"""Edge node running on the ESP32: two independent paths per sample.

1. HardRealTimeGuard (high priority): constant-time threshold logic that latches a
   STOP command (relay) when the current exceeds safety limits. Its worst-case
   execution time is bounded and does not depend on history size, which is what makes
   a hard deadline provable.
2. SoftAnomalyDetector (low priority): learns a healthy baseline and raises
   WARNING/ALARM notifications for the observability platform. Missing its deadline
   degrades the service but never compromises safety.

Execution order per sample: hard path first, soft path after, as two FreeRTOS tasks
with different priorities would behave on the ESP32.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .telemetry import MotorParams


class HardRealTimeGuard:
    CONSECUTIVE_SAMPLES = 2  # debounce against single-sample spikes (e.g. inrush)
    DEADLINE_SAMPLES = 3  # requirement: STOP within 3 samples of the fault

    def __init__(self, params: MotorParams):
        self._i_crit = params.i_critical
        self._i_inst = params.i_instant
        self._count = 0
        self.tripped = False

    def step(self, current: float) -> bool:
        if self.tripped:
            return True
        if current >= self._i_inst:
            self.tripped = True
        elif current >= self._i_crit:
            self._count += 1
            self.tripped = self._count >= self.CONSECUTIVE_SAMPLES
        else:
            self._count = 0
        return self.tripped


class SoftAnomalyDetector:
    """EWMA deviation from a learned healthy baseline of current and power factor."""

    def __init__(
        self,
        warmup_samples: int = 600,
        alpha: float = 0.05,
        warn_z: float = 4.0,
        alarm_z: float = 8.0,
        persistence: int = 30,
    ):
        self.warmup_samples = warmup_samples
        self.alpha = alpha
        self.warn_z = warn_z
        self.alarm_z = alarm_z
        self.persistence = persistence
        self._buf_i: list[float] = []
        self._buf_pf: list[float] = []
        self._mu_i = self._sd_i = self._mu_pf = self._sd_pf = 0.0
        self._ew_i = self._ew_pf = 0.0
        self._above = 0
        self._fitted = False
        self.score = 0.0

    def step(self, current: float, pf: float) -> str:
        if not self._fitted:
            self._buf_i.append(current)
            self._buf_pf.append(pf)
            if len(self._buf_i) == self.warmup_samples:
                self._fit_baseline()
            return "LEARNING"

        self._ew_i += self.alpha * (current - self._ew_i)
        self._ew_pf += self.alpha * (pf - self._ew_pf)
        z_i = (self._ew_i - self._mu_i) / self._sd_i
        z_pf = (self._mu_pf - self._ew_pf) / self._sd_pf
        self.score = max(z_i, z_pf)

        self._above = self._above + 1 if self.score > self.warn_z else 0
        if self._above < self.persistence:
            return "OK"
        return "ALARM" if self.score > self.alarm_z else "WARNING"

    def _fit_baseline(self) -> None:
        i = np.asarray(self._buf_i)
        pf = np.asarray(self._buf_pf)
        self._mu_i, self._mu_pf = float(i.mean()), float(pf.mean())
        # floors avoid division by ~0 with very stable signals or coarse quantization
        self._sd_i = max(float(i.std()), 0.01 * self._mu_i)
        self._sd_pf = max(float(pf.std()), 0.005)
        self._ew_i, self._ew_pf = self._mu_i, self._mu_pf
        self._buf_i.clear()
        self._buf_pf.clear()
        self._fitted = True


class EdgeNode:
    def __init__(self, params: MotorParams = MotorParams(), detector: SoftAnomalyDetector | None = None):
        self.guard = HardRealTimeGuard(params)
        self.detector = detector or SoftAnomalyDetector()


@dataclass
class EdgeResult:
    stop_time: float | None = None
    first_alert_time: float | None = None
    first_alarm_time: float | None = None
    status: list[str] = field(default_factory=list)
    score: list[float] = field(default_factory=list)
    hard_ns: list[int] = field(default_factory=list)
    soft_ns: list[int] = field(default_factory=list)

    def latency_table(self) -> pd.DataFrame:
        rows = []
        for name, data in (("hard_path", self.hard_ns), ("soft_path", self.soft_ns)):
            a = np.asarray(data, dtype=float) / 1000.0
            rows.append(
                {
                    "path": name,
                    "mean_us": a.mean(),
                    "p99_us": np.percentile(a, 99),
                    "max_us": a.max(),
                }
            )
        return pd.DataFrame(rows)


def run_edge(df: pd.DataFrame, node: EdgeNode) -> EdgeResult:
    """Feed samples in order; once STOP is latched the motor is de-energized and the run ends."""
    res = EdgeResult()
    clock = time.perf_counter_ns
    for t, current, pf in zip(df["t"].to_numpy(), df["current"].to_numpy(), df["pf"].to_numpy()):
        t0 = clock()
        stop = node.guard.step(float(current))
        t1 = clock()
        status = node.detector.step(float(current), float(pf))
        t2 = clock()

        res.hard_ns.append(t1 - t0)
        res.soft_ns.append(t2 - t1)
        res.status.append("STOP" if stop else status)
        res.score.append(node.detector.score)
        if status in ("WARNING", "ALARM") and res.first_alert_time is None:
            res.first_alert_time = float(t)
        if status == "ALARM" and res.first_alarm_time is None:
            res.first_alarm_time = float(t)
        if stop:
            res.stop_time = float(t)
            break
    return res
