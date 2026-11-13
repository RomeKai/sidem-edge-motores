import json
import re

import pandas as pd

from dashboard import build_dashboard, load_results


def _write_minimal_results(d):
    pd.DataFrame({"Métrica": ["Desgaste abrupto: paradas dentro del deadline"], "Valor": ["100% (50/50)"]}).to_csv(
        d / "metricas_edge.csv", index=False)
    pd.DataFrame({"path": ["hard_path", "soft_path"], "mean_us": [0.15, 0.37], "p99_us": [0.3, 0.7],
                  "max_us": [27.7, 145.5]}).to_csv(d / "latencia.csv", index=False)
    pd.DataFrame({"Umbral z": [3.0], "Fallas detectadas": [499], "Fallas no detectadas": [219], "Falsas alarmas": [2678],
                  "Precisión": [0.157], "Recall": [0.695], "Anticipación media [h]": [33.6]}).to_csv(
        d / "validacion_azure.csv", index=False)


def _embedded_data(html: str) -> dict:
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert m, "embedded data block missing"
    return json.loads(m.group(1))


def test_dashboard_embeds_results_and_is_offline(tmp_path):
    _write_minimal_results(tmp_path)
    out = build_dashboard(tmp_path)
    html = out.read_text(encoding="utf-8")
    data = _embedded_data(html)
    assert data["azure"][0]["Precisión"] == 0.157
    assert data["latency"][1]["max_us"] == 145.5
    # must work offline in class: nothing is fetched from the network (the SVG namespace URI is only an identifier)
    assert not re.search(r'(src|href)\s*=\s*["\']https?:|url\(\s*["\']?https?:|@import', html)


def test_missing_results_are_reported_not_crashing(tmp_path):
    _write_minimal_results(tmp_path)
    data = load_results(tmp_path)
    assert data["models"] is None
    assert "compare_models.py" in data["missing"]["models"]


def test_figures_are_listed_only_if_present(tmp_path):
    _write_minimal_results(tmp_path)
    (tmp_path / "fig_arquitectura.png").write_bytes(b"\x89PNG")
    data = load_results(tmp_path)
    assert [f["file"] for f in data["figures"]] == ["fig_arquitectura.png"]
