import numpy as np

from si40sim.telemetry import MotorParams, generate

P = MotorParams()


def test_normal_operation_stays_in_nominal_band():
    df = generate("normal", duration_s=3600, seed=1)
    assert len(df) == 3600
    assert set(df.columns) >= {"t", "voltage", "current", "power", "pf", "frequency", "energy_kwh", "health", "label"}
    assert (df["label"] == 0).all()
    assert df["current"].between(0.8 * P.i_nominal, 1.2 * P.i_nominal).all()
    assert df["pf"].between(0.70, 0.90).all()
    assert df["frequency"].between(49.5, 50.5).all()


def test_energy_is_monotonic():
    df = generate("normal", duration_s=600, seed=2)
    assert np.all(np.diff(df["energy_kwh"]) >= 0)


def test_abrupt_wear_exceeds_critical_current_quickly():
    onset = 1000
    df = generate("abrupt", duration_s=2000, seed=3, onset_s=onset)
    after = df[(df["t"] >= onset) & (df["t"] < onset + 3)]
    assert (after["current"] > P.i_critical).any()
    assert (df.loc[df["t"] >= onset + 3, "label"] == 2).all()


def test_gradual_wear_degrades_health_monotonically():
    onset = 500
    df = generate("gradual", duration_s=5000, seed=4, onset_s=onset)
    h = df.loc[df["t"] >= onset, "health"].to_numpy()
    assert np.all(np.diff(h) <= 1e-12)
    assert h[0] > 0.99 and h[-1] < 0.5
    assert (df.loc[df["t"] < onset, "label"] == 0).all()


def test_generation_is_reproducible():
    a = generate("gradual", duration_s=300, seed=7)
    b = generate("gradual", duration_s=300, seed=7)
    assert a.equals(b)
