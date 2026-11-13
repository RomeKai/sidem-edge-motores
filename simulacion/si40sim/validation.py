"""Validate the soft-path detection logic (EWMA deviation from a learned baseline)
on an external dataset with real failure labels.

Same algorithm as SoftAnomalyDetector, generalized to N features and two-sided
deviations, because the external dataset has different sensors than the PZEM-004T.
"""

import numpy as np
import pandas as pd


def detect_alerts(
    df: pd.DataFrame,
    features: list[str],
    warmup: int,
    alpha: float = 0.3,
    z_threshold: float = 4.0,
    persistence: int = 3,
    time_col: str = "datetime",
) -> list[pd.Timestamp]:
    """Return the timestamps where an alert episode starts (rising edges only)."""
    x = df[features].to_numpy(dtype=float)
    base = x[:warmup]
    mu = base.mean(axis=0)
    sd = np.maximum(base.std(axis=0), 1e-9)

    ew = mu.copy()
    above = 0
    active = False
    alerts = []
    times = df[time_col].to_numpy()
    for i in range(warmup, len(x)):
        ew += alpha * (x[i] - ew)
        z = np.max(np.abs(ew - mu) / sd)
        above = above + 1 if z > z_threshold else 0
        if above >= persistence and not active:
            alerts.append(pd.Timestamp(times[i]))
            active = True
        elif above == 0:
            active = False
    return alerts


def score_alerts(
    alerts: list[pd.Timestamp],
    failures: list[pd.Timestamp],
    window_h: float = 48,
    grace_after_h: float = 24,
) -> dict:
    """Event-based scoring.

    TP: failure with at least one alert in [failure - window, failure].
    Alerts in (failure, failure + grace] are ignored (repair still in progress).
    Any other alert is a false positive.
    """
    window = pd.Timedelta(hours=window_h)
    grace = pd.Timedelta(hours=grace_after_h)
    used = set()
    lead_times = []
    tp = 0
    for f in failures:
        in_win = [a for a in alerts if f - window <= a <= f]
        used.update(in_win)
        if in_win:
            tp += 1
            lead_times.append((f - min(in_win)).total_seconds() / 3600)
        used.update(a for a in alerts if f < a <= f + grace)
    fp = sum(1 for a in alerts if a not in used)
    return {
        "true_positives": tp,
        "missed": len(failures) - tp,
        "false_positives": fp,
        "lead_times_h": lead_times,
    }
