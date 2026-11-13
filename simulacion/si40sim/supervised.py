"""Supervised failure prediction for Azure PdM (cloud/digital-twin tier, not the ESP32).

Every feature at hour t uses only data observed at or before t (no look-ahead), and
the label is "a failure happens in (t, t + horizon]".
"""

import numpy as np
import pandas as pd

SENSORS = ["volt", "rotate", "pressure", "vibration"]
ERROR_TYPES = [f"error{i}" for i in range(1, 6)]
COMPONENTS = [f"comp{i}" for i in range(1, 5)]
MODELS = [f"model{i}" for i in range(1, 5)]
NEVER_MAINTAINED_H = 9999.0


def build_features(tel: pd.DataFrame, errors: pd.DataFrame, maint: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    X = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)[["datetime", "machineID", *SENSORS]].copy()
    g = X.groupby("machineID", sort=False)

    for s in SENSORS:
        X[f"{s}_mean_3h"] = g[s].transform(lambda v: v.rolling(3, min_periods=1).mean())
        X[f"{s}_mean_24h"] = g[s].transform(lambda v: v.rolling(24, min_periods=1).mean())
        X[f"{s}_std_24h"] = g[s].transform(lambda v: v.rolling(24, min_periods=2).std()).fillna(0.0)

    err = errors[errors.errorID.isin(ERROR_TYPES)]
    counts = (
        err.assign(n=1).pivot_table(index=["machineID", "datetime"], columns="errorID", values="n", aggfunc="sum")
        .reindex(columns=ERROR_TYPES).fillna(0.0).reset_index()
    )
    X = X.merge(counts, on=["machineID", "datetime"], how="left")
    X[ERROR_TYPES] = X[ERROR_TYPES].fillna(0.0)
    g = X.groupby("machineID", sort=False)
    for e in ERROR_TYPES:
        X[f"{e}_count_24h"] = g[e].transform(lambda v: v.rolling(24, min_periods=1).sum())
    X = X.drop(columns=ERROR_TYPES)

    X = X.sort_values("datetime")  # merge_asof requires sorting by the time key
    for c in COMPONENTS:
        m = maint.loc[maint.comp == c, ["datetime", "machineID"]].sort_values("datetime").rename(columns={"datetime": "last"})
        X = pd.merge_asof(X, m, left_on="datetime", right_on="last", by="machineID", direction="backward")
        X[f"{c}_hours_since"] = ((X["datetime"] - X["last"]).dt.total_seconds() / 3600).fillna(NEVER_MAINTAINED_H)
        X = X.drop(columns="last")
    X = X.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    mach = machines.set_index("machineID")
    X["age"] = X.machineID.map(mach["age"]).astype(float)
    for mdl in MODELS:
        X[f"is_{mdl}"] = (X.machineID.map(mach["model"]) == mdl).astype(float)
    return X


def make_labels(X: pd.DataFrame, failures: pd.DataFrame, horizon_h: float = 48) -> pd.Series:
    f = failures[["datetime", "machineID"]].drop_duplicates().sort_values("datetime").rename(columns={"datetime": "next_failure"})
    left = X[["datetime", "machineID"]].reset_index().sort_values("datetime")
    merged = pd.merge_asof(left, f, left_on="datetime", right_on="next_failure", by="machineID",
                           direction="forward", allow_exact_matches=False)
    delta_h = (merged["next_failure"] - merged["datetime"]).dt.total_seconds() / 3600
    y = (delta_h <= horizon_h).fillna(False)
    return pd.Series(y.to_numpy(), index=merged["index"]).sort_index().reindex(X.index).astype(bool)


def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in ("datetime", "machineID")]


def probabilities_to_alerts(times, proba: np.ndarray, threshold: float, persistence: int = 2) -> list[pd.Timestamp]:
    """Rising edges of 'probability above threshold for `persistence` consecutive hours'."""
    alerts = []
    above = 0
    active = False
    for t, p in zip(times, proba):
        above = above + 1 if p > threshold else 0
        if above >= persistence and not active:
            alerts.append(pd.Timestamp(t))
            active = True
        elif above == 0:
            active = False
    return alerts
