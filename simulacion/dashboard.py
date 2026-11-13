"""Dashboard for the simulation results.

Two modes:
- ``python dashboard.py`` starts a local server on 127.0.0.1 and opens the browser. The page
  has buttons that run the simulation scripts and refresh the metrics when they finish.
- ``python dashboard.py --build`` only writes the self-contained results/dashboard.html
  (no buttons), which can be opened offline by double-clicking it.

A browser cannot run Python on its own, so the buttons need the local server. Only a fixed
list of scripts can be run; the page never sends a command, just the name of a button.
UI text is in Spanish to match the report.
"""

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

HERE = Path(__file__).parent
RESULTS = HERE / "results"

# Button name -> arguments passed to the current Python interpreter (fixed allow-list)
SCRIPTS = {
    "simulation": ["-u", "run_simulation.py"],
    "azure": ["-u", "validate_azure.py"],
    "models": ["-u", "compare_models.py"],
    "tests": ["-u", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--color=no"],
}
LOG_LIMIT = 200_000  # characters kept from a script's output

SOURCES = {  # key -> (csv, script that produces it)
    "edge": ("metricas_edge.csv", "run_simulation.py"),
    "latency": ("latencia.csv", "run_simulation.py"),
    "strategies": ("estrategias.csv", "run_simulation.py"),
    "sensitivity": ("sensibilidad.csv", "run_simulation.py"),
    "azure": ("validacion_azure.csv", "validate_azure.py"),
    "models": ("comparacion_modelos.csv", "compare_models.py"),
}

FIGURES = [
    ("fig_arquitectura.png", "Arquitectura: planta, borde (ESP32) y nube / gemelo digital"),
    ("fig_abrupto.png", "Desgaste abrupto: la parada llega dentro del deadline"),
    ("fig_gradual.png", "Desgaste gradual: alertas tempranas y parada como última defensa"),
]


def load_results(results_dir: Path = RESULTS) -> dict:
    data: dict = {"missing": {}}
    for key, (csv, script) in SOURCES.items():
        path = results_dir / csv
        if path.exists():
            data[key] = json.loads(pd.read_csv(path).to_json(orient="records", force_ascii=False))
        else:
            data[key] = None
            data["missing"][key] = f"Ejecutá python {script} para ver esta sección."
    data["figures"] = [{"file": f, "caption": c} for f, c in FIGURES if (results_dir / f).exists()]
    data["generated"] = datetime.now().strftime("%d/%m/%Y %H:%M")
    return data


def render_html(results_dir: Path = RESULTS, live: bool = False) -> str:
    data = load_results(results_dir)
    data["live"] = live
    payload = json.dumps(data, ensure_ascii=False, indent=0).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", payload)


def build_dashboard(results_dir: Path = RESULTS) -> Path:
    out = results_dir / "dashboard.html"
    out.write_text(render_html(results_dir, live=False), encoding="utf-8")
    return out


class Runner:
    """Runs one allow-listed script at a time in a background thread and keeps its output."""

    def __init__(self, scripts: dict = SCRIPTS, cwd: Path = HERE):
        self.scripts = scripts
        self.cwd = Path(cwd)
        self._lock = threading.Lock()
        self._proc = None
        self._state = {"running": False, "name": None, "log": "", "returncode": None,
                       "started": None, "seconds": None}

    def state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def start(self, name: str) -> bool:
        args = self.scripts[name]  # KeyError for anything not in the allow-list
        with self._lock:
            if self._state["running"]:
                return False
            self._state = {"running": True, "name": name, "log": "", "returncode": None,
                           "started": time.time(), "seconds": None}
        threading.Thread(target=self._run, args=(args,), daemon=True).start()
        return True

    def stop(self) -> None:
        """Terminate the running script, if any (used when the dashboard is closed)."""
        with self._lock:
            proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()

    def _append(self, text: str) -> None:
        with self._lock:
            self._state["log"] = (self._state["log"] + text)[-LOG_LIMIT:]

    def _run(self, args: list) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
               "MPLBACKEND": "Agg"}
        code = -1
        try:
            with subprocess.Popen([sys.executable, *args], cwd=self.cwd, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace") as proc:
                with self._lock:
                    self._proc = proc
                for line in proc.stdout:
                    self._append(line)
                code = proc.wait()
        except OSError as exc:
            self._append(f"No se pudo ejecutar: {exc}\n")
        with self._lock:
            self._proc = None
            self._state.update(running=False, returncode=code,
                               seconds=round(time.time() - self._state["started"], 1))


class _Server(ThreadingHTTPServer):
    allow_reuse_address = False  # on Windows reuse lets a second server share the port silently
    daemon_threads = True


def make_server(results_dir: Path, runner: Runner, port: int = 8765) -> ThreadingHTTPServer:
    results_dir = Path(results_dir).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal clean
            pass

        def _allowed(self) -> bool:
            """Reject requests from other websites (CSRF) and foreign Host names (DNS rebinding)."""
            port_ = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port_}", f"localhost:{port_}"}
            origin = self.headers.get("Origin")
            if self.headers.get("Host") in hosts and (not origin or urlparse(origin).netloc in hosts):
                return True
            self._send(403, b"Forbidden", "text/plain; charset=utf-8")
            return False

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if not self._allowed():
                return
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html", "/dashboard.html"):
                return self._send(200, render_html(results_dir, live=True).encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/data":
                return self._json(200, load_results(results_dir))
            if path == "/api/job":
                return self._json(200, runner.state())
            target = (results_dir / path.lstrip("/")).resolve()
            if target.parent == results_dir and target.suffix.lower() == ".png" and target.is_file():
                return self._send(200, target.read_bytes(), "image/png")
            self._send(404, b"Not found", "text/plain; charset=utf-8")

        def do_POST(self):
            if not self._allowed():
                return
            path = urlparse(self.path).path
            if not path.startswith("/api/run/"):
                return self._send(404, b"Not found", "text/plain; charset=utf-8")
            try:
                started = runner.start(path.removeprefix("/api/run/"))
            except KeyError:
                return self._json(404, {"started": False, "error": "Script desconocido"})
            if not started:
                return self._json(409, {"started": False, "error": "Ya hay un script corriendo"})
            self._json(202, {"started": True})

    return _Server(("127.0.0.1", port), Handler)


TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tablero SI40</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --zone: rgba(12,163,12,0.10);
  --good: #0ca30c; --critical: #d03b3b; --warning: #a35f00; --up: #006300;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --zone: rgba(12,163,12,0.16); --up: #0ca30c; --warning: #e0a030;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --zone: rgba(12,163,12,0.16); --up: #0ca30c; --warning: #e0a030;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 24px 16px 48px; }
header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0; }
.muted { color: var(--muted); font-size: 13px; }
button.theme { background: var(--surface); color: var(--ink); border: 1px solid var(--ring);
  border-radius: 8px; padding: 6px 12px; cursor: pointer; font: inherit; font-size: 13px; }
.idea { margin: 20px 0; padding: 14px 16px; background: var(--surface); border: 1px solid var(--ring);
  border-left: 4px solid var(--s1); border-radius: 10px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin: 20px 0; }
.tile, .card { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 16px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 32px; font-weight: 600; margin: 2px 0; }
.tile .note { font-size: 13px; color: var(--ink-2); }
.status { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; margin-top: 6px; }
.status::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.status.good { color: var(--good); } .status.bad { color: var(--critical); } .status.warn { color: var(--warning); }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; margin-top: 16px; }
.card { margin-top: 16px; } .grid2 .card { margin-top: 0; }
.card p.what { color: var(--ink-2); margin: 4px 0 10px; font-size: 14px; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--ink-2); margin: 4px 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.key { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.key.line { height: 0; border-top: 2px solid; border-radius: 0; width: 18px; }
.key.dash { border-top-style: dashed; }
.key.dot { border-radius: 50%; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { fill: var(--muted); font-size: 11px; font-family: inherit; }
svg text.val { fill: var(--ink-2); font-size: 11px; }
svg .gridline { stroke: var(--grid); stroke-width: 1; }
svg .baseline { stroke: var(--axis); stroke-width: 1; }
.takeaway { margin: 10px 0 0; padding: 8px 12px; border-radius: 8px; background: var(--page); font-size: 14px; }
details { margin-top: 10px; font-size: 13px; }
summary { cursor: pointer; color: var(--ink-2); }
table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 13px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.tablewrap { overflow-x: auto; }
.missing { color: var(--muted); font-style: italic; }
.figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
.figs img { width: 100%; border-radius: 8px; border: 1px solid var(--ring); background: #fff; }
.figs figcaption { font-size: 13px; color: var(--ink-2); margin-top: 6px; }
#tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink);
  border: 1px solid var(--ring); border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; z-index: 10; max-width: 280px; }
#tip b { display: block; margin-bottom: 2px; }
.runbar { margin: 20px 0 0; padding: 14px 16px; background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; }
.runbar h2 { margin-bottom: 8px; }
.runbtns { display: flex; flex-wrap: wrap; gap: 8px; }
.runbtns button { display: flex; flex-direction: column; align-items: flex-start; gap: 2px; text-align: left;
  background: var(--page); color: var(--ink); border: 1px solid var(--ring); border-radius: 10px;
  padding: 8px 12px; cursor: pointer; font: inherit; font-size: 14px; font-weight: 600; min-width: 180px; }
.runbtns button small { font-weight: 400; font-size: 12px; color: var(--muted); }
.runbtns button:hover:not(:disabled) { border-color: var(--s1); }
.runbtns button.primary { background: var(--s1); color: #fff; border-color: var(--s1); }
.runbtns button.primary small { color: rgba(255,255,255,0.85); }
.runbtns button:disabled { opacity: 0.5; cursor: not-allowed; }
.runbtns button.active { outline: 2px solid var(--s1); outline-offset: 1px; opacity: 1; }
#runstatus { margin-top: 10px; font-size: 14px; min-height: 21px; }
#runstatus.ok { color: var(--good); } #runstatus.fail { color: var(--critical); }
#runlog pre { max-height: 260px; overflow: auto; background: var(--page); border: 1px solid var(--grid);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; white-space: pre-wrap; margin: 6px 0 0; }
.figs img { cursor: zoom-in; }
dialog#zoom { border: 0; padding: 0; background: transparent; max-width: 96vw; max-height: 96vh; }
dialog#zoom::backdrop { background: rgba(0,0,0,0.75); }
dialog#zoom .zbox { background: var(--surface); border-radius: 12px; padding: 12px; max-height: 94vh; overflow: auto; }
dialog#zoom .zbar { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 8px; color: var(--ink); font-size: 14px; }
dialog#zoom img { display: block; max-width: 92vw; max-height: 80vh; margin: 0 auto; background: #fff; border-radius: 6px; cursor: zoom-in; }
dialog#zoom img.full { max-width: none; max-height: none; cursor: zoom-out; }
@media (max-width: 540px) { .grid2 { grid-template-columns: 1fr; } .tile .value { font-size: 26px; } }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Detección de desgaste en motores con Edge Computing</h1>
      <p class="sub">Trabajo final SI40 · UTN FRLP · Monfroglio – Di Grappa</p>
      <p class="muted" id="generated"></p>
    </div>
    <button class="theme" id="theme" type="button">Cambiar tema</button>
  </header>

  <section class="runbar" id="runbar">
    <h2>Correr la simulación</h2>
    <div class="runbtns">
      <button type="button" data-run="simulation">1 · Motor simulado<small>run_simulation.py · sensor, Edge y costos</small></button>
      <button type="button" data-run="azure">2 · Contraste con Azure<small>validate_azure.py · datos externos</small></button>
      <button type="button" data-run="models">3 · Modelo supervisado<small>compare_models.py · tarda más</small></button>
      <button type="button" data-run="all" class="primary">Correr todo<small>1 → 2 → 3, en orden</small></button>
      <button type="button" data-run="tests">Pruebas<small>pytest · verifica el código</small></button>
    </div>
    <div id="runstatus" role="status" aria-live="polite"></div>
    <details id="runlog"><summary>Ver salida del script</summary><pre id="log">Todavía no se corrió nada.</pre></details>
  </section>

  <div class="idea"><b>La idea en una frase:</b> detectar fallas es fácil; detectarlas sin falsas alarmas es lo difícil.
    Por eso la parada la decide una regla simple en el ESP32, y la "inteligencia" solo avisa.</div>

  <section class="tiles" id="tiles"></section>

  <div class="grid2">
    <section class="card" id="edge-card">
      <h2>1 · Nodo Edge con datos simulados</h2>
      <p class="what">50 corridas por escenario. Con datos propios todo funciona perfecto: era esperable.</p>
      <div id="edge"></div>
    </section>
    <section class="card" id="latency-card">
      <h2>2 · Tiempo de cómputo por muestra</h2>
      <p class="what">El camino duro siempre es rápido. Los máximos los produce Windows, no el algoritmo:
        por eso en el ESP32 va en una tarea de FreeRTOS de alta prioridad.</p>
      <div id="latency"></div>
    </section>
  </div>

  <section class="card">
    <h2>3 · Contraste con datos externos (Azure PdM, 100 máquinas)</h2>
    <p class="what">Recall: cuántas fallas detecta. Precisión: cuántas alertas eran reales. Por umbral de sensibilidad z.</p>
    <div id="azure"></div>
  </section>

  <section class="card">
    <h2>4 · Detector simple vs. modelo supervisado</h2>
    <p class="what">Arriba a la izquierda es lo ideal: detecta muchas fallas con pocas falsas alarmas.
      La franja verde es donde el mantenimiento predictivo conviene económicamente. Pasá el mouse por los puntos.</p>
    <div id="models"></div>
  </section>

  <div class="grid2">
    <section class="card">
      <h2>5 · Costo por estrategia (200 motores, 5 años)</h2>
      <p class="what">Con supuestos optimistas: 1 falsa alarma por motor por año.</p>
      <div id="strategies"></div>
    </section>
    <section class="card">
      <h2>6 · Sensibilidad a las falsas alarmas</h2>
      <p class="what">Cada falsa alarma es un técnico que revisa y no encuentra nada. Debajo de 0 %,
        conviene más no hacer nada (correctivo).</p>
      <div id="sensitivity"></div>
    </section>
  </div>

  <section class="card" id="figs-card">
    <h2>Gráficos de la simulación</h2>
    <div class="figs" id="figs"></div>
  </section>
</main>
<div id="tip" role="status"></div>
<dialog id="zoom" aria-label="Gráfico ampliado">
  <div class="zbox">
    <div class="zbar"><span id="zcap"></span><span><span class="muted">Clic en la imagen: tamaño real</span>
      <button class="theme" type="button" id="zclose">Cerrar (Esc)</button></span></div>
    <img id="zimg" alt="">
  </div>
</dialog>

<script id="data" type="application/json">__DATA__</script>
<script>
let D = JSON.parse(document.getElementById("data").textContent);
const fmt = (v, d = 1) => Number(v).toLocaleString("es-AR", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 0) => fmt(v * 100, d) + " %";
const el = (tag, attrs = {}, parent) => {
  const svgTags = ["svg","g","rect","line","circle","path","text"];
  const n = svgTags.includes(tag) ? document.createElementNS("http://www.w3.org/2000/svg", tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === "text" ? (n.textContent = v) : n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};
const tip = document.getElementById("tip");
function showTip(evt, html) {
  tip.innerHTML = html; tip.style.display = "block";
  const x = Math.min(evt.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(evt.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => (tip.style.display = "none");

function missing(id, key) {
  const box = document.getElementById(id);
  box.innerHTML = `<p class="missing">${D.missing[key]}</p>`;
}
function table(parent, cols, rows, open = false) {
  const det = el("details", open ? { open: "" } : {}, parent);
  el("summary", { text: "Ver tabla" }, det);
  const wrap = el("div", { class: "tablewrap" }, det);
  const t = el("table", {}, wrap);
  const tr = el("tr", {}, el("thead", {}, t));
  cols.forEach(c => el("th", { text: c.label, class: c.num ? "n" : "" }, tr));
  const tb = el("tbody", {}, t);
  rows.forEach(r => { const row = el("tr", {}, tb);
    cols.forEach(c => el("td", { text: c.f ? c.f(r[c.key], r) : r[c.key], class: c.num ? "n" : "" }, row)); });
}
function legend(parent, items) {
  const l = el("div", { class: "legend" }, parent);
  items.forEach(i => { const s = el("span", {}, l);
    const k = el("span", { class: "key " + (i.kind || "") }, s);
    if (i.kind && i.kind.includes("line")) k.style.borderTopColor = i.color; else k.style.background = i.color;
    s.appendChild(document.createTextNode(i.label)); });
}
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function niceTicks(min, max, n = 5) {
  const span = max - min, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n);
  const ticks = []; for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}
// Bar path with 4px rounded data-end and square baseline
function barPath(x, y0, w, y1) {
  const r = Math.min(4, Math.abs(y1 - y0) / 2, w / 2), up = y1 < y0;
  if (up) return `M${x},${y0}V${y1 + r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 + r}V${y0}Z`;
  return `M${x},${y0}V${y1 - r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 - r}V${y0}Z`;
}

// ---------- tiles ----------
function tiles() {
  const box = document.getElementById("tiles");
  const add = (label, value, note, status) => {
    const t = el("div", { class: "tile" }, box);
    el("div", { class: "label", text: label }, t);
    el("div", { class: "value", text: value }, t);
    el("div", { class: "note", text: note }, t);
    if (status) el("div", { class: "status " + status[0], text: status[1] }, t);
  };
  const deadline = D.edge && D.edge.find(r => r["Métrica"].includes("paradas dentro"));
  const delay = D.edge && D.edge.find(r => r["Métrica"].includes("retardo"));
  if (deadline) {
    const m = deadline["Valor"].match(/\((\d+)\/(\d+)\)/);
    const maxDelay = delay ? delay["Valor"].split(" ")[0] : null;
    add("Parada de emergencia ante falla abrupta", m ? `${m[1]} de ${m[2]}` : deadline["Valor"],
      "fallas simuladas en las que el ESP32 detuvo el motor antes del plazo de 3 muestras" +
        (maxDelay ? ` (tardó como máximo ${maxDelay} ${maxDelay === "1" ? "muestra" : "muestras"})` : ""),
      ["good", "Cumple el tiempo real duro"]);
  }
  const z3 = D.azure && D.azure.find(r => r["Umbral z"] === 3);
  if (z3) add("Alertas que eran fallas reales (Azure)", `${Math.round(z3["Precisión"] * 100)} de 100`,
    "con el detector simple; el resto fueron falsas alarmas sobre máquinas sanas",
    ["warn", "Por eso solo avisa: nunca detiene el motor"]);
  if (D.models) {
    const edge = D.models.find(r => r.Modelo.startsWith("No supervisado") && r.Modelo.includes("z=3"));
    const best = D.models.find(r => r.Modelo.includes("completo") && r.Modelo.includes("0.9)"));
    if (edge && best) add("Falsas alarmas por motor por año",
      `${fmt(edge["Falsas alarmas/motor/año"])} → ${fmt(best["Falsas alarmas/motor/año"])}`,
      "detector simple vs. modelo que además usa el historial de errores y mantenimiento",
      ["good", "Debajo de 12 por año el predictivo conviene"]);
  }
  if (D.sensitivity) {
    const opt = D.sensitivity.find(r => r.escenario === "optimista" && r.estrategia === "Prescriptivo" && r.falsas_alarmas_por_anio === 1);
    const pred27 = D.sensitivity.find(r => r.escenario === "calibrado Azure" && r.estrategia === "Predictivo" && r.falsas_alarmas_por_anio === 27);
    if (opt && pred27) add("Ahorro frente a reparar cuando se rompe", pct(opt.ahorro_vs_correctivo),
      `con mantenimiento prescriptivo y pocas falsas alarmas. Con las 27 por año medidas, el predictivo cuesta ${pct(-pred27.ahorro_vs_correctivo)} más`,
      null);
  }
}

// ---------- 1. edge ----------
function edge() {
  if (!D.edge) return missing("edge", "edge");
  table(document.getElementById("edge"), [{ key: "Métrica", label: "Métrica" }, { key: "Valor", label: "Resultado", num: true }], D.edge, true);
}

// ---------- 2. latency ----------
function latency() {
  if (!D.latency) return missing("latency", "latency");
  const names = { hard_path: "Camino duro (parada)", soft_path: "Camino blando (alertas)" };
  table(document.getElementById("latency"), [
    { key: "path", label: "Camino", f: v => names[v] || v },
    { key: "mean_us", label: "Media [µs]", num: true, f: v => fmt(v, 2) },
    { key: "p99_us", label: "P99 [µs]", num: true, f: v => fmt(v, 2) },
    { key: "max_us", label: "Máximo [µs]", num: true, f: v => fmt(v, 1) }], D.latency, true);
  const box = document.getElementById("latency");
  el("p", { class: "takeaway", text: "Los máximos cambian en cada corrida: eso muestra que un sistema operativo común no garantiza tiempos." }, box);
}

// ---------- 3. azure grouped columns ----------
function azure() {
  if (!D.azure) return missing("azure", "azure");
  const box = document.getElementById("azure");
  const S = [{ key: "Recall", label: "Recall (fallas detectadas)", c: "--s1" }, { key: "Precisión", label: "Precisión (alertas reales)", c: "--s2" }];
  legend(box, S.map(s => ({ label: s.label, color: css(s.c) })));
  const W = 720, H = 260, m = { l: 44, r: 8, t: 16, b: 34 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Recall y precisión del detector por umbral" }, box);
  const y = v => m.t + (1 - v) * (H - m.t - m.b);
  [0, .25, .5, .75, 1].forEach(v => { el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: v ? "gridline" : "baseline" }, svg);
    el("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end", text: (v * 100) + " %" }, svg); });
  const band = (W - m.l - m.r) / D.azure.length, bw = 24, gap = 2;
  D.azure.forEach((r, i) => {
    const cx = m.l + band * i + band / 2;
    el("text", { x: cx, y: H - 12, "text-anchor": "middle", text: "Umbral z = " + r["Umbral z"] }, svg);
    S.forEach((s, j) => {
      const x = cx - bw - gap / 2 + j * (bw + gap), v = r[s.key];
      const p = el("path", { d: barPath(x, y(0), bw, y(v)), fill: `var(${s.c})` }, svg);
      el("text", { x: x + bw / 2, y: y(v) - 5, "text-anchor": "middle", class: "val", text: pct(v) }, svg);
      const hit = el("rect", { x: x - 2, y: m.t, width: bw + 4, height: H - m.t - m.b, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", e => showTip(e, `<b>z = ${r["Umbral z"]}</b>${s.label}: ${pct(v, 1)}<br>Detectadas: ${r["Fallas detectadas"]} · Falsas alarmas: ${r["Falsas alarmas"]}`));
      hit.addEventListener("mouseleave", hideTip);
    });
  });
  el("p", { class: "takeaway" }, box).innerHTML =
    "Con z = 3 encuentra 7 de cada 10 fallas, pero <b>84 de cada 100 alertas son falsas</b>. Si pudiera parar el motor, casi todas las paradas serían innecesarias.";
  table(box, [{ key: "Umbral z", label: "Umbral z" }, { key: "Fallas detectadas", label: "Detectadas", num: true },
    { key: "Fallas no detectadas", label: "No detectadas", num: true }, { key: "Falsas alarmas", label: "Falsas alarmas", num: true },
    { key: "Precisión", label: "Precisión", num: true, f: v => pct(v, 1) }, { key: "Recall", label: "Recall", num: true, f: v => pct(v, 1) },
    { key: "Anticipación media [h]", label: "Anticipación [h]", num: true, f: v => fmt(v) }], D.azure);
}

// ---------- 4. models scatter ----------
function models() {
  if (!D.models) return missing("models", "models");
  const box = document.getElementById("models");
  const group = r => r.Modelo.startsWith("No supervisado") ? 0 : r.Modelo.includes("telemetr") ? 1 : 2;
  const G = [{ label: "Detector simple (ESP32)", c: "--s1" }, { label: "Supervisado: solo sensores", c: "--s2" },
             { label: "Supervisado: sensores + errores + mantenimiento", c: "--s3" }];
  legend(box, [...G.map(g => ({ label: g.label, color: css(g.c), kind: "dot" })), { label: "Zona rentable (< 12 falsas alarmas/año)", color: css("--zone") }]);
  const W = 720, H = 300, m = { l: 48, r: 16, t: 28, b: 42 };
  const xMax = Math.ceil(Math.max(...D.models.map(r => r["Falsas alarmas/motor/año"])) / 10) * 10;
  const x = v => m.l + v / xMax * (W - m.l - m.r), y = v => m.t + (1 - v) * (H - m.t - m.b);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Recall contra falsas alarmas por modelo" }, box);
  el("rect", { x: x(0), y: m.t, width: x(12) - x(0), height: H - m.t - m.b, fill: "var(--zone)" }, svg);
  [0, .25, .5, .75, 1].forEach(v => { el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: v ? "gridline" : "baseline" }, svg);
    el("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end", text: (v * 100) + " %" }, svg); });
  niceTicks(0, xMax).forEach(v => el("text", { x: x(v), y: H - 24, "text-anchor": "middle", text: v }, svg));
  el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", text: "Falsas alarmas por motor por año →" }, svg);
  el("text", { x: m.l, y: 12, text: "↑ Fallas detectadas (recall)" }, svg);
  D.models.forEach(r => {
    const g = G[group(r)], cx = x(r["Falsas alarmas/motor/año"]), cy = y(r.Recall);
    el("circle", { cx, cy, r: 6, fill: `var(${g.c})`, stroke: "var(--surface)", "stroke-width": 2 }, svg);
    const hit = el("circle", { cx, cy, r: 14, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", e => showTip(e, `<b>${r.Modelo}</b>Recall: ${pct(r.Recall, 1)} · Precisión: ${pct(r["Precisión"], 1)}<br>Falsas alarmas/motor/año: ${fmt(r["Falsas alarmas/motor/año"])}<br>Anticipación: ${fmt(r["Anticipación media [h]"])} h`));
    hit.addEventListener("mouseleave", hideTip);
  });
  el("p", { class: "takeaway" }, box).innerHTML =
    "Cambiar de algoritmo con los mismos sensores <b>no alcanza</b> (naranja). Lo que mejora es sumar contexto de la máquina (aqua): eso es lo que aporta un gemelo digital. " +
    "<span class='muted'>Ojo: parte de la mejora del modelo completo viene de un patrón del dataset (mantenimientos cada 15 días).</span>";
  table(box, [{ key: "Modelo", label: "Modelo" }, { key: "Precisión", label: "Precisión", num: true, f: v => pct(v, 1) },
    { key: "Recall", label: "Recall", num: true, f: v => pct(v, 1) },
    { key: "Falsas alarmas/motor/año", label: "Falsas alarmas/año", num: true, f: v => fmt(v) },
    { key: "Anticipación media [h]", label: "Anticipación [h]", num: true, f: v => fmt(v) }], D.models);
}

// ---------- 5. strategies horizontal bars ----------
function strategies() {
  if (!D.strategies) return missing("strategies", "strategies");
  const box = document.getElementById("strategies");
  const W = 520, rowH = 44, m = { l: 92, r: 150, t: 4 }, H = m.t + rowH * D.strategies.length + 4;
  const max = Math.max(...D.strategies.map(r => r["Costo total [USD]"]));
  const x = v => m.l + v / max * (W - m.l - m.r);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Costo total por estrategia" }, box);
  el("line", { x1: m.l, x2: m.l, y1: 0, y2: H, class: "baseline" }, svg);
  D.strategies.forEach((r, i) => {
    const yc = m.t + rowH * i + rowH / 2, v = r["Costo total [USD]"], bw = 22;
    el("text", { x: m.l - 8, y: yc + 4, "text-anchor": "end", text: r.Estrategia, class: "val" }, svg);
    const xe = x(v), rr = 4;
    el("path", { d: `M${m.l},${yc - bw / 2}H${xe - rr}Q${xe},${yc - bw / 2} ${xe},${yc - bw / 2 + rr}V${yc + bw / 2 - rr}Q${xe},${yc + bw / 2} ${xe - rr},${yc + bw / 2}H${m.l}Z`, fill: "var(--s1)" }, svg);
    el("text", { x: xe + 6, y: yc + 4, class: "val", text: `USD ${fmt(v / 1e6, 2)} M · ahorro ${r["Ahorro vs. correctivo"]}` }, svg);
    const hit = el("rect", { x: m.l, y: yc - rowH / 2, width: W - m.l, height: rowH, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", e => showTip(e, `<b>${r.Estrategia}</b>Costo: USD ${fmt(v, 0)}<br>Fallas no planificadas: ${r["Fallas no planificadas"]}<br>Horas de parada: ${fmt(r["Horas de parada"], 0)}`));
    hit.addEventListener("mouseleave", hideTip);
  });
  table(box, [{ key: "Estrategia", label: "Estrategia" }, { key: "Fallas no planificadas", label: "Fallas", num: true },
    { key: "Horas de parada", label: "Horas de parada", num: true, f: v => fmt(v, 0) },
    { key: "Costo total [USD]", label: "Costo [USD]", num: true, f: v => fmt(v, 0) },
    { key: "Ahorro vs. correctivo", label: "Ahorro", num: true }], D.strategies);
}

// ---------- 6. sensitivity lines with crosshair ----------
function sensitivity() {
  if (!D.sensitivity) return missing("sensitivity", "sensitivity");
  const box = document.getElementById("sensitivity");
  const S = [
    { e: "optimista", s: "Prescriptivo", c: "--s2", dash: false, label: "Prescriptivo (optimista)" },
    { e: "calibrado Azure", s: "Prescriptivo", c: "--s2", dash: true, label: "Prescriptivo (calibrado Azure)" },
    { e: "optimista", s: "Predictivo", c: "--s1", dash: false, label: "Predictivo (optimista)" },
    { e: "calibrado Azure", s: "Predictivo", c: "--s1", dash: true, label: "Predictivo (calibrado Azure)" }];
  legend(box, S.map(s => ({ label: s.label, color: css(s.c), kind: "line" + (s.dash ? " dash" : "") })));
  const xs = [...new Set(D.sensitivity.map(r => r.falsas_alarmas_por_anio))].sort((a, b) => a - b);
  const vals = D.sensitivity.map(r => r.ahorro_vs_correctivo);
  const yMin = Math.floor(Math.min(...vals) * 5) / 5, yMax = Math.ceil(Math.max(...vals) * 5) / 5;
  const W = 520, H = 300, m = { l: 48, r: 12, t: 12, b: 40 }, xMax = xs[xs.length - 1];
  const x = v => m.l + v / xMax * (W - m.l - m.r), y = v => m.t + (yMax - v) / (yMax - yMin) * (H - m.t - m.b);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Ahorro contra falsas alarmas por año" }, box);
  niceTicks(yMin * 100, yMax * 100, 6).forEach(v => { el("line", { x1: m.l, x2: W - m.r, y1: y(v / 100), y2: y(v / 100), class: v === 0 ? "baseline" : "gridline" }, svg);
    el("text", { x: m.l - 6, y: y(v / 100) + 4, "text-anchor": "end", text: v + " %" }, svg); });
  xs.forEach(v => el("text", { x: x(v), y: H - 22, "text-anchor": "middle", text: v }, svg));
  el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", text: "Falsas alarmas por motor por año" }, svg);
  const measured = 26.8;
  el("line", { x1: x(measured), x2: x(measured), y1: m.t, y2: H - m.b, stroke: "var(--critical)", "stroke-width": 1 }, svg);
  el("text", { x: x(measured) - 4, y: m.t + 10, "text-anchor": "end", text: "medido en Azure" }, svg);
  const series = S.map(s => ({ ...s, pts: xs.map(v => D.sensitivity.find(r => r.escenario === s.e && r.estrategia === s.s && r.falsas_alarmas_por_anio === v)) }));
  series.forEach(s => el("path", { d: s.pts.map((r, i) => `${i ? "L" : "M"}${x(r.falsas_alarmas_por_anio)},${y(r.ahorro_vs_correctivo)}`).join(""),
    fill: "none", stroke: `var(${s.c})`, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash ? "6 4" : "none" }, svg));
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" }, svg);
  const dots = series.map(s => el("circle", { r: 4, fill: `var(${s.c})`, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }, svg));
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
  hit.addEventListener("mousemove", e => {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    const i = xs.reduce((b, v, k) => Math.abs(x(v) - p.x) < Math.abs(x(xs[b]) - p.x) ? k : b, 0);
    cross.setAttribute("x1", x(xs[i])); cross.setAttribute("x2", x(xs[i])); cross.setAttribute("visibility", "visible");
    series.forEach((s, k) => { dots[k].setAttribute("cx", x(xs[i])); dots[k].setAttribute("cy", y(s.pts[i].ahorro_vs_correctivo)); dots[k].setAttribute("visibility", "visible"); });
    showTip(e, `<b>${xs[i]} falsas alarmas/año</b>` + series.map(s => `${s.label}: ${pct(s.pts[i].ahorro_vs_correctivo)}`).join("<br>"));
  });
  hit.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); });
  el("p", { class: "takeaway" }, box).innerHTML = "Con las falsas alarmas medidas, el predictivo cuesta <b>más</b> que el correctivo. El prescriptivo resiste más porque filtra alarmas.";
  const rows = xs.map(v => Object.fromEntries([["fa", v], ...series.map(s => [s.label, s.pts.find(r => r.falsas_alarmas_por_anio === v).ahorro_vs_correctivo])]));
  table(box, [{ key: "fa", label: "Falsas alarmas/año" }, ...series.map(s => ({ key: s.label, label: s.label, num: true, f: v => pct(v) }))], rows);
}

// ---------- figures ----------
function figures() {
  const box = document.getElementById("figs");
  if (!D.figures.length) { document.getElementById("figs-card").style.display = "none"; return; }
  D.figures.forEach(f => { const fig = el("figure", { style: "margin:0" }, box);
    const img = el("img", { src: f.file + (figVersion ? "?v=" + figVersion : ""), alt: f.caption, loading: "lazy", tabindex: "0" }, fig);
    el("figcaption", { text: f.caption + " · clic para ampliar" }, fig);
    const open = () => openZoom(img.src, f.caption);
    img.addEventListener("click", open);
    img.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } }); });
}

function render() {
  ["tiles", "edge", "latency", "azure", "models", "strategies", "sensitivity", "figs"].forEach(id => document.getElementById(id).innerHTML = "");
  tiles(); edge(); latency(); azure(); models(); strategies(); sensitivity(); figures();
}
let figVersion = 0;
const setGenerated = () => document.getElementById("generated").textContent = "Datos leídos el " + D.generated + " desde results/";
setGenerated();

// ---------- figure zoom ----------
const zoom = document.getElementById("zoom"), zimg = document.getElementById("zimg");
function openZoom(src, caption) {
  zimg.src = src; zimg.alt = caption; zimg.classList.remove("full");
  document.getElementById("zcap").textContent = caption;
  zoom.showModal();
}
zimg.addEventListener("click", () => zimg.classList.toggle("full"));
document.getElementById("zclose").addEventListener("click", () => zoom.close());
zoom.addEventListener("click", e => { if (e.target === zoom) zoom.close(); });

// ---------- run buttons (only with the local server) ----------
const NAMES = { simulation: "Motor simulado", azure: "Contraste con Azure", models: "Modelo supervisado", tests: "Pruebas" };
const runBtns = [...document.querySelectorAll("[data-run]")];
const runStatus = document.getElementById("runstatus"), logBox = document.getElementById("log");
const setStatus = (text, cls = "") => { runStatus.textContent = text; runStatus.className = cls; };
const sleep = ms => new Promise(r => setTimeout(r, ms));
function lockButtons(active) {
  runBtns.forEach(b => { b.disabled = !!active; b.classList.toggle("active", b.dataset.run === active); });
}
async function runOne(name, step = "") {
  const res = await fetch("/api/run/" + name, { method: "POST" });
  if (!res.ok) throw new Error((await res.json()).error || "No se pudo iniciar");
  const t0 = Date.now();
  let job;
  do {
    await sleep(700);
    job = await (await fetch("/api/job")).json();
    logBox.textContent = (job.log || "(sin salida todavía)").replace(/\x1b\[[0-9;]*m/g, "");
    logBox.scrollTop = logBox.scrollHeight;
    setStatus(`${step}Corriendo ${NAMES[name]}… ${Math.round((Date.now() - t0) / 1000)} s`);
  } while (job.running);
  if (job.returncode !== 0) throw new Error(`${NAMES[name]} terminó con error (código ${job.returncode}). Mirá la salida del script.`);
  return job;
}
async function refreshData() {
  const fresh = await (await fetch("/api/data")).json();
  D = { ...fresh, live: true };
  figVersion = Date.now();
  setGenerated(); render();
}
async function onRun(name) {
  const queue = name === "all" ? ["simulation", "azure", "models"] : [name];
  lockButtons(name);
  try {
    let secs = 0;
    for (let i = 0; i < queue.length; i++) {
      const job = await runOne(queue[i], queue.length > 1 ? `Paso ${i + 1} de ${queue.length} · ` : "");
      secs += job.seconds;
      if (queue[i] !== "tests") await refreshData();
    }
    setStatus(`✓ ${name === "all" ? "Todo" : NAMES[name]} listo en ${fmt(secs)} s.` + (name === "tests" ? "" : " Las métricas ya están actualizadas."), "ok");
  } catch (err) {
    setStatus("✕ " + err.message, "fail");
    document.getElementById("runlog").open = true;
    try { await refreshData(); } catch (_) {}
  } finally { lockButtons(null); }
}
if (D.live) {
  runBtns.forEach(b => b.addEventListener("click", () => onRun(b.dataset.run)));
  setStatus("Elegí qué correr. Si cambiás datos o parámetros, volvé a apretar el botón.");
} else {
  runBtns.forEach(b => b.disabled = true);
  setStatus("Los botones funcionan abriendo el tablero con: python dashboard.py");
}
document.getElementById("theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  render();
});
render();
</script>
</body>
</html>
"""


def main() -> None:
    out = build_dashboard()
    print(f"Tablero estático generado: {out}")
    if "--build" in sys.argv:
        return
    try:
        runner = Runner()
        server = make_server(RESULTS, runner, port=8765)
    except OSError:
        print("El puerto 8765 está ocupado: ¿ya hay otro tablero abierto? Cerralo o usá --build.")
        return
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Tablero con botones en {url}  (Ctrl+C para cerrar)")
    if "--no-open" not in sys.argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        server.server_close()
        build_dashboard()  # leave the static copy up to date with the last runs


if __name__ == "__main__":
    main()
