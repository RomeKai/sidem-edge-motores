"""Validate the soft-path detector on Microsoft Azure Predictive Maintenance (Kaggle).

Expected files in ./data:
    PdM_telemetry.csv  (datetime, machineID, volt, rotate, pressure, vibration)
    PdM_failures.csv   (datetime, machineID, failure)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from si40sim.validation import detect_alerts, score_alerts

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"
FEATURES = ["volt", "rotate", "pressure", "vibration"]
WARMUP_H = 7 * 24
WINDOW_H = 48


def main() -> None:
    tel_path, fail_path = DATA / "PdM_telemetry.csv", DATA / "PdM_failures.csv"
    if not tel_path.exists() or not fail_path.exists():
        sys.exit(f"Missing dataset. Put PdM_telemetry.csv and PdM_failures.csv in {DATA}")

    tel = pd.read_csv(tel_path, parse_dates=["datetime"]).sort_values(["machineID", "datetime"])
    fails = pd.read_csv(fail_path, parse_dates=["datetime"])

    rows = []
    for z in (3.0, 4.0, 5.0):
        tp = missed = fp = 0
        leads: list[float] = []
        for mid, g in tel.groupby("machineID"):
            # smooth hourly noise with a 3h rolling mean (an edge node would aggregate the same way)
            g = g.copy()
            g[FEATURES] = g[FEATURES].rolling(3, min_periods=1).mean()
            alerts = detect_alerts(g, FEATURES, warmup=WARMUP_H, z_threshold=z)
            f_times = list(fails.loc[fails.machineID == mid, "datetime"].drop_duplicates())
            s = score_alerts(alerts, f_times, window_h=WINDOW_H)
            tp, missed, fp = tp + s["true_positives"], missed + s["missed"], fp + s["false_positives"]
            leads += s["lead_times_h"]
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + missed) if tp + missed else float("nan")
        rows.append({
            "Umbral z": z,
            "Fallas detectadas": tp,
            "Fallas no detectadas": missed,
            "Falsas alarmas": fp,
            "Precisión": round(precision, 3),
            "Recall": round(recall, 3),
            "Anticipación media [h]": round(float(np.mean(leads)), 1) if leads else None,
        })

    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "validacion_azure.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
