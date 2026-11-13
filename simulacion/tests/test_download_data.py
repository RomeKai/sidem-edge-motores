import io

from download_data import FILES, download


def test_downloads_missing_files_with_pdm_prefix(tmp_path):
    requested = []

    def fake_open(url):
        requested.append(url)
        return io.BytesIO(b"datetime,machineID\n")

    saved = download(tmp_path, opener=fake_open)
    assert sorted(p.name for p in saved) == sorted(f"PdM_{n}.csv" for n in FILES)
    assert all(u.endswith(".csv") and "PredictiveMaintenanceModelingGuide" in u for u in requested)
    assert (tmp_path / "PdM_telemetry.csv").read_bytes() == b"datetime,machineID\n"


def test_skips_files_already_present(tmp_path):
    for n in FILES:
        (tmp_path / f"PdM_{n}.csv").write_text("x")

    def fail_open(url):
        raise AssertionError("should not download")

    assert download(tmp_path, opener=fail_open) == []
