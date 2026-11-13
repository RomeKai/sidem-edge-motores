from si40sim.edge import EdgeNode, HardRealTimeGuard, SoftAnomalyDetector, run_edge
from si40sim.telemetry import MotorParams, generate

P = MotorParams()


def test_hard_guard_trips_on_abrupt_wear_within_deadline_samples():
    onset = 1200
    df = generate("abrupt", duration_s=2000, seed=11, onset_s=onset)
    result = run_edge(df, EdgeNode(P))
    assert result.stop_time is not None
    assert onset <= result.stop_time <= onset + HardRealTimeGuard.DEADLINE_SAMPLES


def test_hard_guard_never_trips_on_normal_operation():
    for seed in range(5):
        df = generate("normal", duration_s=7200, seed=seed)
        assert run_edge(df, EdgeNode(P)).stop_time is None


def test_hard_guard_requires_consecutive_samples_to_reject_single_spike():
    guard = HardRealTimeGuard(P)
    assert guard.step(current=P.i_critical * 1.5) is False
    assert guard.step(current=P.i_nominal) is False
    assert guard.step(current=P.i_critical * 1.5) is False
    assert guard.step(current=P.i_critical * 1.5) is True


def test_soft_detector_alerts_on_gradual_wear_before_health_is_critical():
    onset = 1800
    df = generate("gradual", duration_s=20000, seed=21, onset_s=onset)
    result = run_edge(df, EdgeNode(P))
    critical_time = df.loc[df["health"] < 0.3, "t"].iloc[0]
    assert result.first_alert_time is not None
    assert onset < result.first_alert_time < critical_time


def test_soft_detector_has_no_alerts_on_normal_operation():
    for seed in range(5):
        df = generate("normal", duration_s=7200, seed=100 + seed)
        assert run_edge(df, EdgeNode(P)).first_alert_time is None


def test_soft_detector_learns_baseline_before_alerting():
    det = SoftAnomalyDetector(warmup_samples=10)
    for _ in range(10):
        assert det.step(current=3.4, pf=0.8) == "LEARNING"
    assert det.step(current=3.4, pf=0.8) == "OK"
