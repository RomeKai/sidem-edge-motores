"""Architecture diagram for the report (labels in Spanish per course template)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "results" / "fig_arquitectura.png"


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=color, ec="#333333", lw=1.2))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=9.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.75, "\n".join(lines), ha="center", va="top", fontsize=7.8, linespacing=1.4)


def arrow(ax, p0, p1, label="", color="#333333", style="-|>", ls="-", label_offset=(0, 0.15)):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14, color=color, lw=1.6, linestyle=ls))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + label_offset[0], (p0[1] + p1[1]) / 2 + label_offset[1], label,
                ha="center", fontsize=7.5, color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.8, 8.4)
    ax.axis("off")

    ax.text(1.9, 8.15, "PLANTA (OT)", ha="center", fontsize=10, weight="bold", color="#555555")
    ax.text(7.0, 8.15, "BORDE (EDGE)", ha="center", fontsize=10, weight="bold", color="#555555")
    ax.text(12.6, 8.15, "NUBE / GEMELO DIGITAL", ha="center", fontsize=10, weight="bold", color="#555555")
    for x in (4.1, 10.1):
        ax.plot([x, x], [0.2, 8.0], ls=":", c="#999999")

    box(ax, 0.3, 4.6, 3.3, 2.6, "Motor monofásico", ["220 V / 50 Hz", "~1 HP"], "#e8e8e8")
    box(ax, 0.3, 1.0, 3.3, 2.8, "Dispositivo de telemetría", ["PZEM-004T (V, I, P, FP, f, E)", "UART / Modbus-RTU, ~1 Hz",
                                                               "Relé de parada (GPIO)"], "#fff3d6")

    box(ax, 4.6, 4.5, 5.0, 3.0, "Tarea 1: Tiempo real DURO", ["Prioridad alta (FreeRTOS)", "Umbral de corriente + antirrebote",
                                                             "Tiempo de cómputo constante O(1)",
                                                             "Deadline: 3 muestras"], "#f9d5d3")
    box(ax, 4.6, 0.6, 5.0, 3.2, "Tarea 2: Tiempo real BLANDO", ["Prioridad baja", "Línea base aprendida (salud inicial)",
                                                               "Desvío EWMA de I y FP", "ADVERTENCIA / ALARMA"], "#d6e9f8")

    box(ax, 10.6, 4.5, 4.1, 3.0, "Observabilidad", ["Series temporales + dashboard", "Alertas al operador", "(p. ej. MQTT → TSDB)"],
        "#dff2dd")
    box(ax, 10.6, 0.6, 4.1, 3.2, "Gemelo digital", ["Modelo supervisado", "+ contexto (errores, CMMS)", "Análisis qué-pasaría-si",
                                                  "Recomendación prescriptiva"], "#e9def5")

    arrow(ax, (1.95, 4.6), (1.95, 3.8), "")
    arrow(ax, (3.6, 3.4), (4.6, 6.0), "muestras", label_offset=(-0.45, 0.25))
    arrow(ax, (4.6, 5.0), (3.6, 2.6), "PARADA", color="#c0392b", label_offset=(0.05, -0.55))
    arrow(ax, (3.6, 1.8), (4.6, 1.8), "muestras", label_offset=(0, 0.12))
    arrow(ax, (9.6, 2.6), (10.6, 5.4), "eventos", label_offset=(-0.35, 0.1))
    arrow(ax, (9.6, 1.9), (10.6, 1.9), "telemetría", label_offset=(0, 0.12))
    purple = "#7d3c98"
    ax.plot([12.65, 12.65, 1.95], [0.6, 0.15, 0.15], ls="--", c=purple, lw=1.6)
    arrow(ax, (1.95, 0.15), (1.95, 1.0), "", color=purple, ls="--")
    ax.text(7.1, -0.2, "acción prescripta (planificación / reducción de carga)", ha="center", fontsize=7.5, color=purple)
    arrow(ax, (12.65, 3.8), (12.65, 4.5), "")

    ax.text(7.5, -0.6, "Lazo bidireccional: el flujo de vuelta (rojo y violeta) es lo que distingue al gemelo digital de una sombra digital",
            ha="center", fontsize=8, style="italic", color="#444444")
    fig.tight_layout()
    fig.savefig(OUT, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
