"""Download the Azure Predictive Maintenance dataset into data/.

The CSVs (~80 MB) are not stored in this repository: they belong to Microsoft and are fetched
from the official sample repository (SQL-Server-R-Services-Samples /
PredictiveMaintenanceModelingGuide). Files are saved as data/PdM_<name>.csv, the same names
used by validate_azure.py and compare_models.py. Files already present are skipped.
"""

import shutil
import sys
import urllib.request
from pathlib import Path

BASE_URL = ("https://media.githubusercontent.com/media/microsoft/SQL-Server-R-Services-Samples/"
            "master/PredictiveMaintenanceModelingGuide/Data/")
FILES = ["telemetry", "failures", "errors", "maint", "machines"]
DATA = Path(__file__).parent / "data"


def download(dest: Path = DATA, opener=urllib.request.urlopen) -> list[Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in FILES:
        target = dest / f"PdM_{name}.csv"
        if target.exists():
            print(f"Ya existe {target.name}, se omite.")
            continue
        print(f"Descargando {name}.csv ...", flush=True)
        tmp = target.with_suffix(".part")
        with opener(BASE_URL + f"{name}.csv") as src, open(tmp, "wb") as out:
            shutil.copyfileobj(src, out)
        tmp.replace(target)
        saved.append(target)
    return saved


if __name__ == "__main__":
    try:
        download()
    except OSError as exc:
        sys.exit(f"No se pudo descargar el dataset: {exc}")
    print(f"Dataset listo en {DATA}")
