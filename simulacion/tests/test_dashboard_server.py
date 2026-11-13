import json
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from dashboard import Runner, make_server
from tests.test_dashboard import _write_minimal_results


def _wait_done(runner: Runner, timeout: float = 20.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        state = runner.state()
        if not state["running"]:
            return state
        time.sleep(0.05)
    raise AssertionError("script did not finish in time")


def test_runner_rejects_unknown_script(tmp_path):
    runner = Runner({"ok": ["-c", "print(1)"]}, cwd=tmp_path)
    with pytest.raises(KeyError):
        runner.start("rm -rf")


def test_runner_runs_script_and_captures_output(tmp_path):
    runner = Runner({"hello": ["-c", "print('hola desde el motor')"]}, cwd=tmp_path)
    assert runner.start("hello") is True
    state = _wait_done(runner)
    assert state["name"] == "hello"
    assert state["returncode"] == 0
    assert "hola desde el motor" in state["log"]


def test_runner_refuses_second_job_while_busy(tmp_path):
    runner = Runner({"slow": ["-c", "import time; time.sleep(1)"]}, cwd=tmp_path)
    assert runner.start("slow") is True
    assert runner.start("slow") is False
    _wait_done(runner)


def test_runner_reports_failures(tmp_path):
    runner = Runner({"boom": ["-c", "import sys; print('falló'); sys.exit(3)"]}, cwd=tmp_path)
    runner.start("boom")
    state = _wait_done(runner)
    assert state["returncode"] == 3
    assert "falló" in state["log"]


@pytest.fixture
def server(tmp_path):
    _write_minimal_results(tmp_path)
    (tmp_path / "fig_gradual.png").write_bytes(b"\x89PNGfake")
    runner = Runner({"hello": ["-c", "print('ok')"]}, cwd=tmp_path)
    srv = make_server(tmp_path, runner, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", runner
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def _post(url):
    req = urllib.request.Request(url, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_server_serves_live_dashboard_with_runner_enabled(server):
    base, _ = server
    status, body = _get(base + "/")
    html = body.decode("utf-8")
    assert status == 200
    assert '<script id="data" type="application/json">' in html
    assert '"live": true' in html


def test_server_serves_data_and_figures(server):
    base, _ = server
    _, body = _get(base + "/api/data")
    assert json.loads(body)["azure"][0]["Precisión"] == 0.157
    _, img = _get(base + "/fig_gradual.png")
    assert img == b"\x89PNGfake"


def test_server_blocks_path_traversal(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/..%2Fdashboard.py")
    assert e.value.code == 404


def test_run_endpoint_starts_known_script_and_rejects_unknown(server):
    base, runner = server
    status, body = _post(base + "/api/run/hello")
    assert status == 202 and body["started"] is True
    _wait_done(runner)
    _, state = _get(base + "/api/job")
    assert "ok" in json.loads(state)["log"]
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/api/run/format_c")
    assert e.value.code == 404


def test_static_build_has_runner_disabled(tmp_path):
    from dashboard import build_dashboard
    _write_minimal_results(tmp_path)
    html = build_dashboard(tmp_path).read_text(encoding="utf-8")
    assert '"live": false' in html


def test_runner_uses_current_interpreter(tmp_path):
    runner = Runner({"py": ["-c", "import sys; print(sys.executable)"]}, cwd=tmp_path)
    runner.start("py")
    assert sys.executable in _wait_done(runner)["log"]
