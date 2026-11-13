import numpy as np
import pandas as pd

from si40sim.supervised import build_features, make_labels, probabilities_to_alerts

SENSORS = ["volt", "rotate", "pressure", "vibration"]


def _toy(n_hours=200, machines=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2015-01-01", periods=n_hours, freq="h")
    tel = pd.concat(
        [pd.DataFrame({"datetime": t, "machineID": m, **{s: rng.normal(100, 5, n_hours) for s in SENSORS}})
         for m in machines],
        ignore_index=True,
    )
    errors = pd.DataFrame({"datetime": [t[50], t[60]], "machineID": [1, 1], "errorID": ["error1", "error2"]})
    maint = pd.DataFrame({"datetime": [t[0] - pd.Timedelta(days=10), t[100]], "machineID": [1, 1], "comp": ["comp1", "comp1"]})
    machines_df = pd.DataFrame({"machineID": list(machines), "model": ["model1", "model2"][: len(machines)], "age": [5, 10][: len(machines)]})
    return tel, errors, maint, machines_df


def test_features_have_one_row_per_machine_hour_and_no_nans():
    tel, errors, maint, machines = _toy()
    X = build_features(tel, errors, maint, machines)
    assert len(X) == len(tel)
    assert not X.drop(columns=["datetime", "machineID"]).isna().any().any()
    assert {"volt_mean_24h", "vibration_std_24h", "error1_count_24h", "comp1_hours_since"} <= set(X.columns)


def test_features_do_not_look_ahead():
    tel, errors, maint, machines = _toy()
    cut = pd.Timestamp("2015-01-05")
    X1 = build_features(tel, errors, maint, machines)
    tel2 = tel.copy()
    tel2.loc[tel2.datetime > cut, SENSORS] += 1000.0
    errors2 = pd.concat([errors, pd.DataFrame({"datetime": [cut + pd.Timedelta(hours=1)], "machineID": [1], "errorID": ["error1"]})])
    X2 = build_features(tel2, errors2, maint, machines)
    past = X1.datetime <= cut
    pd.testing.assert_frame_equal(X1[past].reset_index(drop=True), X2[past].reset_index(drop=True))


def test_hours_since_maintenance_resets_after_replacement():
    tel, errors, maint, machines = _toy()
    X = build_features(tel, errors, maint, machines)
    m1 = X[X.machineID == 1].reset_index(drop=True)
    assert m1.loc[99, "comp1_hours_since"] == 240 + 99
    assert m1.loc[100, "comp1_hours_since"] == 0


def test_labels_mark_hours_within_horizon_before_failure():
    t = pd.date_range("2015-01-01", periods=100, freq="h")
    X = pd.DataFrame({"datetime": t, "machineID": 1})
    failures = pd.DataFrame({"datetime": [t[80]], "machineID": [1]})
    y = make_labels(X, failures, horizon_h=10)
    assert y.sum() == 10
    assert y.iloc[70:80].all() and not y.iloc[80] and not y.iloc[69]


def test_probabilities_to_alerts_returns_rising_edges_with_persistence():
    t = pd.date_range("2015-01-01", periods=10, freq="h")
    p = np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.9, 0.2, 0.9, 0.9, 0.1])
    alerts = probabilities_to_alerts(t, p, threshold=0.5, persistence=2)
    assert alerts == [t[4], t[8]]
