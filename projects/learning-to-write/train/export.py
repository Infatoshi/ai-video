"""Copy the run's record into the app: train/run/run.json + probe.json -> data/run.json (steps as compact
arrays) + data/probe.json. Run on the Mac after fetching train/run/{run,probe}.json from the GPU host.

  python3 train/export.py [run]
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RUN = HERE / (sys.argv[1] if len(sys.argv) > 1 else "run")
r = json.loads((RUN / "run.json").read_text())
st = r.pop("steps")
r["steps"] = {"step": [s["step"] for s in st], "loss": [s["loss"] for s in st], "t": [s["t"] for s in st]}
(DATA / "run.json").write_text(json.dumps(r, separators=(",", ":")))
(DATA / "probe.json").write_text((RUN / "probe.json").read_text())
ex = HERE.parent / "song" / ("excerpts.json" if RUN.name == "run" else f"excerpts_{RUN.name}.json")
(DATA / "excerpts.json").write_text(ex.read_text())
print("wrote", DATA / "run.json", DATA / "probe.json")
