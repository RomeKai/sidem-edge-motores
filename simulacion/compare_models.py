"""Compare the unsupervised edge detector vs. a supervised model on Azure PdM.

Temporal split: train on Jan-Aug 2015, evaluate both detectors on Sep-Dec 2015 with the
same event-based scoring (alert within 48 h before a failure).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from si40sim.supervised import SENSORS, build_features, feature_columns, make_labels, probabilities_to_alerts
from si40sim.validation import detect_alerts, score_alerts

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"
SPLIT = pd.Timestamp("2015-09-01")
HORIZON_H = 48
FA_BREAK_EVEN = 12  # false alarms / motor / year where predictive stops paying off (fig_sensibilidad)


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"PdM_{name}.csv")
    if "datetime" in df:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def evaluate(alerts_by_machine: dict, failures: pd.DataFrame, test_years: float) -> dict:
    tp = missed = fp = 0
    leads = []
    for mid, alerts in alerts_by_machine.items():
        alerts = [a for a in alerts if a >= SPLIT]
        f = list(failures.loc[(failures.machineID == mid) & (failures.datetime >= SPLIT), "datetime"].drop_duplicates())
        s = score_alerts(alerts, f, window_h=HORIZON_H)
        tp, missed, fp = tp + s["true_positives"], missed + s["missed"], fp + s["false_positives"]
        leads += s["lead_times_h"]
    n = len(alerts_by_machine)
    return {
        "Fallas detectadas": tp,
        "Fallas no detectadas": missed,
        "Falsas alarmas": fp,
        "Precisión": round(tp / (tp + fp), 3) if tp + fp else np.nan,
        "Recall": round(tp / (tp + missed), 3) if tp + missed else np.nan,
        "Falsas alarmas/motor/año": round(fp / n / test_years, 1),
        "Anticipación media [h]": round(float(np.mean(leads)), 1) if leads else np.nan,
    }


def main() -> None:
    tel, failures = load("telemetry"), load("failures")
    errors, maint, machines = load("errors"), load("maint"), load("machines")
    test_years = (tel.datetime.max() - SPLIT).total_seconds() / (365 * 24 * 3600)

    print("Building features...")
    X = build_features(tel, errors, maint, machines)
    y = make_labels(X, failures, horizon_h=HORIZON_H)
    all_cols = feature_columns(X)
    # "telemetry" = what a PZEM-like sensor stream offers; "full" adds error logs and maintenance
    # history. Caveat: in this synthetic dataset replacements follow a 15-day grid, so maintenance
    # features carry generator artifacts that inflate the "full" result.
    feature_sets = {
        "telemetría": [c for c in all_cols if c.split("_")[0] in SENSORS],
        "completo": all_cols,
    }
    train, test = X.datetime < SPLIT - pd.Timedelta(hours=HORIZON_H), X.datetime >= SPLIT  # gap avoids label leakage
    print(f"Training on {train.sum()} rows ({y[train].mean():.1%} positive)...")
    for name, cols in feature_sets.items():
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
        model.fit(X.loc[train, cols], y[train])
        X[f"proba_{name}"] = np.nan
        X.loc[test, f"proba_{name}"] = model.predict_proba(X.loc[test, cols])[:, 1]

    rows = []
    feats = ["volt", "rotate", "pressure", "vibration"]
    for z in (3.0, 4.0):
        alerts = {}
        for mid, g in tel.sort_values(["machineID", "datetime"]).groupby("machineID"):
            g = g.copy()
            g[feats] = g[feats].rolling(3, min_periods=1).mean()
            alerts[mid] = detect_alerts(g, feats, warmup=7 * 24, z_threshold=z)
        rows.append({"Modelo": f"No supervisado (edge, z={z:g})", **evaluate(alerts, failures, test_years)})

    Xt = X[test]
    for name in feature_sets:
        for thr in (0.5, 0.7, 0.9, 0.95):
            alerts = {mid: probabilities_to_alerts(g.datetime, g[f"proba_{name}"].to_numpy(), threshold=thr)
                      for mid, g in Xt.groupby("machineID")}
            rows.append({"Modelo": f"Supervisado {name} (p>{thr:g})", **evaluate(alerts, failures, test_years)})

    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "comparacion_modelos.csv", index=False)
    print(res.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in res.iterrows():
        color = "tab:green" if "completo" in r.Modelo else "tab:blue" if "telemetría" in r.Modelo else "tab:red"
        ax.scatter(r["Falsas alarmas/motor/año"], r["Recall"] * 100, s=80,
                   marker="s" if color == "tab:red" else "o", c=color)
        ax.annotate(r.Modelo, (r["Falsas alarmas/motor/año"], r["Recall"] * 100), fontsize=7,
                    xytext=(5, 4), textcoords="offset points")
    ax.axvspan(0, FA_BREAK_EVEN, color="tab:green", alpha=0.08, label=f"Zona rentable (< {FA_BREAK_EVEN} FA/motor/año)")
    ax.set(xlabel="Falsas alarmas por motor por año", ylabel="Fallas detectadas (recall) [%]",
           title="No supervisado (edge) vs. supervisado (nube), período de prueba sep-dic 2015")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_comparacion_modelos.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
