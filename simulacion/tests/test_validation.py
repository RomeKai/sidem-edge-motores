import numpy as np
import pandas as pd

from si40sim.validation import detect_alerts, score_alerts


def _machine(n_hours=1000, fail_at=800, anomaly_hours=30, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2015-01-01", periods=n_hours, freq="h")
    volt = 170 + rng.normal(0, 5, n_hours)
    volt[fail_at - anomaly_hours : fail_at] += 40
    return pd.DataFrame({"datetime": t, "volt": volt, "vibration": 40 + rng.normal(0, 2, n_hours)}), t[fail_at]


def test_detects_shift_before_failure_with_positive_lead_time():
    df, fail_time = _machine()
    alerts = detect_alerts(df, ["volt", "vibration"], warmup=168)
    s = score_alerts(alerts, [fail_time], window_h=48)
    assert s["true_positives"] == 1
    assert s["false_positives"] == 0
    assert 0 < s["lead_times_h"][0] <= 30


def test_no_alerts_on_stationary_signal():
    df, _ = _machine(anomaly_hours=0)
    assert detect_alerts(df, ["volt", "vibration"], warmup=168) == []


def test_alert_far_from_failure_is_false_positive():
    t0 = pd.Timestamp("2015-01-01")
    s = score_alerts([t0], [t0 + pd.Timedelta(hours=500)], window_h=48)
    assert s["true_positives"] == 0 and s["false_positives"] == 1 and s["missed"] == 1
