"""Run all scenarios and write figures/tables to ./results for the final report.

Figure labels are in Spanish because the course template requires it.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from si40sim.edge import EdgeNode, HardRealTimeGuard, run_edge
from si40sim.maintenance import FleetAssumptions, simulate_fleet
from si40sim.telemetry import MotorParams, generate

OUT = Path(__file__).parent / "results"
P = MotorParams()
N_RUNS = 50
STRATEGY_LABELS = {
    "corrective": "Correctivo",
    "preventive": "Preventivo",
    "predictive": "Predictivo",
    "prescriptive": "Prescriptivo",
}


def evaluate_edge() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    for i in range(N_RUNS):
        onset = float(rng.integers(900, 3000))
        df = generate("abrupt", duration_s=4000, seed=1000 + i, onset_s=onset)
        r = run_edge(df, EdgeNode(P))
        delay = None if r.stop_time is None else r.stop_time - onset
        rows.append({"scenario": "abrupt", "run": i, "stop_delay_samples": delay,
                     "deadline_met": delay is not None and delay <= HardRealTimeGuard.DEADLINE_SAMPLES})

        onset = float(rng.integers(900, 3000))
        df = generate("gradual", duration_s=onset + 7000, seed=2000 + i, onset_s=onset)
        r = run_edge(df, EdgeNode(P))
        t_crit = df.loc[df["health"] < 0.3, "t"].iloc[0]
        rows.append({"scenario": "gradual", "run": i,
                     "alert_after_onset_s": None if r.first_alert_time is None else r.first_alert_time - onset,
                     "lead_before_critical_s": None if r.first_alert_time is None else t_crit - r.first_alert_time,
                     "health_at_alert": None if r.first_alert_time is None
                     else float(df.loc[df["t"] == r.first_alert_time, "health"].iloc[0]),
                     "stop_time_s": None if r.stop_time is None else r.stop_time - onset})

        df = generate("normal", duration_s=8 * 3600, seed=3000 + i)
        r = run_edge(df, EdgeNode(P))
        rows.append({"scenario": "normal", "run": i,
                     "false_alert": r.first_alert_time is not None, "false_stop": r.stop_time is not None})
    return pd.DataFrame(rows)


def summarize_edge(ev: pd.DataFrame) -> pd.DataFrame:
    ab, gr, no = (ev[ev.scenario == s] for s in ("abrupt", "gradual", "normal"))
    hours_normal = len(no) * 8
    return pd.DataFrame([
        ("Desgaste abrupto: paradas dentro del deadline", f"{ab.deadline_met.mean():.0%} ({int(ab.deadline_met.sum())}/{len(ab)})"),
        ("Desgaste abrupto: retardo de parada máx. [muestras]", f"{ab.stop_delay_samples.max():.0f} (deadline = {HardRealTimeGuard.DEADLINE_SAMPLES})"),
        ("Desgaste gradual: detección (recall)", f"{gr.alert_after_onset_s.notna().mean():.0%}"),
        ("Desgaste gradual: salud media al alertar", f"{gr.health_at_alert.mean():.2f}"),
        ("Desgaste gradual: anticipación media antes de estado crítico [min]", f"{gr.lead_before_critical_s.mean() / 60:.0f}"),
        ("Operación normal: falsas alertas", f"{int(no.false_alert.sum())} en {hours_normal} h"),
        ("Operación normal: paradas en falso", f"{int(no.false_stop.sum())} en {hours_normal} h"),
    ], columns=["Métrica", "Valor"])


def fig_abrupt() -> pd.DataFrame:
    onset = 1500
    df = generate("abrupt", duration_s=1600, seed=42, onset_s=onset)
    r = run_edge(df, EdgeNode(P))
    win = df[(df.t >= onset - 60) & (df.t <= onset + 20)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(win.t - onset, win.current, ".-", label="Corriente medida (PZEM)")
    ax.axhline(P.i_nominal, ls=":", c="gray", label=f"I nominal = {P.i_nominal:.2f} A")
    ax.axhline(P.i_critical, ls="--", c="tab:orange", label=f"Umbral crítico = {P.i_critical:.2f} A")
    ax.axvline(0, c="tab:red", alpha=0.5, label="Inicio de la falla")
    ax.axvline(r.stop_time - onset, c="black", label=f"Señal de PARADA (t+{r.stop_time - onset:.0f} muestras)")
    ax.axvspan(0, HardRealTimeGuard.DEADLINE_SAMPLES, color="tab:green", alpha=0.12, label="Ventana del deadline")
    ax.set(xlabel="Tiempo relativo a la falla [s]", ylabel="Corriente [A]",
           title="Desgaste abrupto: camino de tiempo real duro")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_abrupto.png", dpi=150)
    plt.close(fig)
    return r.latency_table()


def fig_gradual() -> None:
    onset = 1800
    df = generate("gradual", duration_s=onset + 7000, seed=7, onset_s=onset)
    r = run_edge(df, EdgeNode(P))
    df = df.iloc[: len(r.status)].copy()
    df["score"] = r.score
    tm = df.t / 60
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(tm, df.current, lw=0.6, label="Corriente [A] (eje izq.)")
    axes[0].axhline(P.i_critical, ls="--", c="tab:orange", label="Umbral crítico de corriente")
    axes[0].set_ylabel("Corriente [A]")
    ax2 = axes[0].twinx()
    ax2.plot(tm, df.pf, lw=0.6, c="tab:purple", alpha=0.6, label="Factor de potencia (eje der.)")
    ax2.set_ylabel("Factor de potencia", color="tab:purple")
    axes[1].plot(tm, df.score, c="tab:brown", label="Puntaje de anomalía (z EWMA)")
    axes[1].axhline(4, ls="--", c="gold", label="Umbral ADVERTENCIA")
    axes[1].axhline(8, ls="--", c="tab:red", label="Umbral ALARMA")
    axes[1].set_ylabel("Puntaje z")
    axes[2].plot(tm, df.health, c="tab:green", label="Salud real (ground truth)")
    axes[2].axhline(0.3, ls=":", c="tab:red", label="Estado crítico")
    axes[2].set(ylabel="Salud [0-1]", xlabel="Tiempo [min]")
    for ax in axes:
        ax.axvline(onset / 60, c="gray", alpha=0.5, label="Inicio del desgaste" if ax is axes[2] else None)
        for t_ev, c, lab in ((r.first_alert_time, "gold", "1ra advertencia"), (r.first_alarm_time, "tab:red", "1ra alarma"),
                             (r.stop_time, "black", "Parada")):
            if t_ev is not None:
                ax.axvline(t_ev / 60, c=c, lw=1.5, label=lab if ax is axes[2] else None)
    # Legends sit above each panel so they never cover the data
    h0, l0 = axes[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend_kw = dict(fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, borderaxespad=0.2)
    axes[0].legend(h0 + h2, l0 + l2, ncol=3, **legend_kw)
    axes[1].legend(ncol=3, **legend_kw)
    axes[2].legend(ncol=6, **legend_kw)
    fig.suptitle("Desgaste gradual: camino blando (alertas a observabilidad)", fontsize=12)
    fig.tight_layout(h_pad=2.2)
    fig.savefig(OUT / "fig_gradual.png", dpi=150)
    plt.close(fig)


def fig_latency(lat: pd.DataFrame) -> None:
    df = generate("normal", duration_s=20000, seed=5)
    r = run_edge(df, EdgeNode(P))
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.logspace(-1, 3, 60)
    ax.hist(np.asarray(r.hard_ns) / 1000, bins=bins, alpha=0.7, label="Camino duro (umbral)")
    ax.hist(np.asarray(r.soft_ns) / 1000, bins=bins, alpha=0.7, label="Camino blando (detector)")
    ax.set(xscale="log", yscale="log", xlabel="Tiempo de cómputo por muestra [µs]", ylabel="Frecuencia",
           title="Tiempo de ejecución por muestra (PC, sistema operativo de propósito general)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_latencia.png", dpi=150)
    plt.close(fig)
    r.latency_table().round(2).to_csv(OUT / "latencia.csv", index=False)


def fig_strategies() -> pd.DataFrame:
    res = simulate_fleet(n_motors=200, horizon_days=5 * 365, seed=1)
    res["Estrategia"] = res.strategy.map(STRATEGY_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, lab in zip(axes, ["total_cost", "downtime_h", "failures"],
                            ["Costo total [USD]", "Horas de parada", "Fallas no planificadas"]):
        ax.bar(res.Estrategia, res[col], color=["tab:red", "tab:orange", "tab:blue", "tab:green"])
        ax.set_title(lab)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Flota de 200 motores, horizonte de 5 años (Monte Carlo)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_estrategias.png", dpi=150)
    plt.close(fig)
    out = res[["Estrategia", "failures", "planned_interventions", "downtime_h", "total_cost"]].copy()
    out.columns = ["Estrategia", "Fallas no planificadas", "Intervenciones planificadas", "Horas de parada", "Costo total [USD]"]
    base = out["Costo total [USD]"].iloc[0]
    out["Ahorro vs. correctivo"] = (1 - out["Costo total [USD]"] / base).map("{:.0%}".format)
    return out


def fig_sensitivity() -> pd.DataFrame:
    """Savings vs. false-alarm rate: the value of predictive maintenance hinges on precision.

    The 'calibrated' curve uses recall and lead time measured on Azure PdM (validate_azure.py, z=3).
    """
    fa_rates = [0, 1, 2, 5, 10, 15, 20, 27]
    scenarios = {
        "optimista": FleetAssumptions(),
        "calibrado Azure": FleetAssumptions(detection_prob=0.695, lead_days=(0.5, 2.0)),
    }
    rows = []
    for name, base in scenarios.items():
        for fa in fa_rates:
            res = simulate_fleet(n_motors=200, horizon_days=5 * 365, seed=1,
                                 assumptions=replace(base, false_alarms_per_year=fa)).set_index("strategy")
            corrective = res.loc["corrective", "total_cost"]
            for s in ("predictive", "prescriptive"):
                rows.append({"escenario": name, "falsas_alarmas_por_anio": fa, "estrategia": STRATEGY_LABELS[s],
                             "ahorro_vs_correctivo": 1 - res.loc[s, "total_cost"] / corrective})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {"optimista": "-", "calibrado Azure": "--"}
    colors = {"Predictivo": "tab:blue", "Prescriptivo": "tab:green"}
    for (esc, est), g in df.groupby(["escenario", "estrategia"]):
        ax.plot(g.falsas_alarmas_por_anio, g.ahorro_vs_correctivo * 100, styles[esc], marker="o",
                c=colors[est], label=f"{est} ({esc})")
    ax.axhline(0, c="black", lw=1)
    ax.axvline(26.8, c="tab:red", ls=":", label="Tasa medida en Azure PdM (z=3)")
    ax.set(xlabel="Falsas alarmas por motor por año", ylabel="Ahorro vs. correctivo [%]",
           title="Sensibilidad: el valor del mantenimiento predictivo depende de su precisión")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sensibilidad.png", dpi=150)
    plt.close(fig)
    return df


def main() -> None:
    OUT.mkdir(exist_ok=True)
    edge_summary = summarize_edge(evaluate_edge())
    edge_summary.to_csv(OUT / "metricas_edge.csv", index=False)
    lat = fig_abrupt()
    fig_gradual()
    fig_latency(lat)
    strategies = fig_strategies()
    strategies.round(0).to_csv(OUT / "estrategias.csv", index=False)
    sens = fig_sensitivity()
    sens.round(3).to_csv(OUT / "sensibilidad.csv", index=False)

    print("\n== Nodo edge ==")
    print(edge_summary.to_string(index=False))
    print("\n== Latencia por muestra ==")
    print(pd.read_csv(OUT / "latencia.csv").to_string(index=False))
    print("\n== Estrategias de mantenimiento ==")
    print(strategies.round(0).to_string(index=False))
    print("\n== Sensibilidad (ahorro vs. correctivo) ==")
    print(sens.pivot_table(index="falsas_alarmas_por_anio", columns=["escenario", "estrategia"],
                           values="ahorro_vs_correctivo").map("{:.0%}".format).to_string())


if __name__ == "__main__":
    main()
