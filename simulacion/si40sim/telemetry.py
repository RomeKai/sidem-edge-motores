"""Synthetic telemetry of a single-phase induction motor as read by a PZEM-004T.

The PZEM-004T only measures electrical variables (V, I, P, PF, f, energy), so wear
must be inferred from them: mechanical wear (bearings, misalignment, friction)
increases the load torque, which raises the current drawn and lowers the power factor.

Scenarios:
    normal  - healthy motor, small load fluctuation and sensor noise.
    gradual - progressive wear starting at ``onset_s`` (accelerating, quadratic).
    abrupt  - sudden mechanical failure at ``onset_s`` (e.g. bearing seizure):
              current jumps towards locked-rotor values.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MotorParams:
    v_nominal: float = 220.0  # V (Argentine mains)
    f_nominal: float = 50.0  # Hz
    p_input: float = 600.0  # W, ~1 HP motor at ~80% load
    pf_nominal: float = 0.80
    critical_factor: float = 1.8  # hard-RT trip threshold, multiple of nominal current
    instant_factor: float = 3.0  # immediate trip, multiple of nominal current

    @property
    def i_nominal(self) -> float:
        return self.p_input / (self.v_nominal * self.pf_nominal)

    @property
    def i_critical(self) -> float:
        return self.critical_factor * self.i_nominal

    @property
    def i_instant(self) -> float:
        return self.instant_factor * self.i_nominal


# PZEM-004T v3 display resolution
_RES = {"voltage": 0.1, "current": 0.001, "power": 0.1, "pf": 0.01, "frequency": 0.1}

GRADUAL_CURRENT_GAIN = 0.9  # current rises up to +90% at end of life
GRADUAL_PF_DROP = 0.15
ABRUPT_CURRENT_FACTOR = 2.2
ABRUPT_PF = 0.55


def _quantize(x: np.ndarray, step: float) -> np.ndarray:
    return np.round(x / step) * step


def _wear(scenario: str, t: np.ndarray, onset_s: float, time_to_failure_s: float) -> np.ndarray:
    if scenario == "normal":
        return np.zeros_like(t, dtype=float)
    if scenario == "gradual":
        x = np.clip((t - onset_s) / time_to_failure_s, 0.0, 1.0)
        return x**2
    if scenario == "abrupt":
        return np.where(t >= onset_s, 0.95, 0.0)
    raise ValueError(f"unknown scenario: {scenario}")


def _label(health: np.ndarray) -> np.ndarray:
    return np.select([health < 0.3, health < 0.9], [2, 1], default=0)


def generate(
    scenario: str,
    duration_s: int,
    seed: int = 0,
    onset_s: float | None = None,
    time_to_failure_s: float = 6000.0,
    sample_period_s: float = 1.0,
    params: MotorParams = MotorParams(),
) -> pd.DataFrame:
    """Return one row per PZEM sample with ground-truth ``health`` (1 = new, 0 = failed)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s / sample_period_s)
    t = np.arange(n) * sample_period_s
    if onset_s is None:
        onset_s = duration_s / 3

    w = _wear(scenario, t, onset_s, time_to_failure_s)
    health = 1.0 - w

    phase = rng.uniform(0, 2 * np.pi, size=2)
    voltage = (
        params.v_nominal
        + 3.0 * np.sin(2 * np.pi * t / 1800 + phase[0])  # slow mains drift
        + rng.normal(0, 0.8, n)
    )
    load = 1.0 + 0.02 * np.sin(2 * np.pi * t / 900 + phase[1]) + rng.normal(0, 0.008, n)

    if scenario == "abrupt":
        current_factor = np.where(t >= onset_s, ABRUPT_CURRENT_FACTOR, 1.0)
        pf = np.where(t >= onset_s, ABRUPT_PF, params.pf_nominal)
    else:
        current_factor = 1.0 + GRADUAL_CURRENT_GAIN * w
        pf = params.pf_nominal - GRADUAL_PF_DROP * w

    current = params.i_nominal * load * current_factor + rng.normal(0, 0.01, n)
    pf = np.clip(pf + rng.normal(0, 0.004, n), 0.0, 1.0)
    frequency = params.f_nominal + rng.normal(0, 0.03, n)

    voltage = _quantize(voltage, _RES["voltage"])
    current = _quantize(current, _RES["current"])
    pf = _quantize(pf, _RES["pf"])
    frequency = _quantize(frequency, _RES["frequency"])
    power = _quantize(voltage * current * pf, _RES["power"])
    energy_kwh = np.cumsum(power) * sample_period_s / 3.6e6

    return pd.DataFrame(
        {
            "t": t,
            "voltage": voltage,
            "current": current,
            "power": power,
            "pf": pf,
            "frequency": frequency,
            "energy_kwh": energy_kwh,
            "health": health,
            "label": _label(health),
        }
    )
