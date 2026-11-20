"""Run the place-learning forecast job (see ml/README.md).

  python -m ml.run build --data DIR [--max-minutes N] [--release TAG]
      Builds the year chunks that are missing (oldest history first) and
      rebuilds this year's (and, early in the year, last year's) GFS chunk so
      new days keep coming in. With --release, each finished chunk is uploaded
      to that GitHub release right away, so a run that stops keeps its work.
  python -m ml.run train --data DIR --out FILE [--upload]
      Trains, tests on held-back years and writes the model + report. With
      --upload, sends both to the website (MODELS_UPLOAD_KEY).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data import build_chunk, load_stations, save_chunk  # noqa: E402

FIRST_RF, LAST_RF, FIRST_GFS = 2000, 2019, 2021
FRESH_DAYS = 6  # rebuild the current GFS chunk when it's older than this


def chunk_names(today):
    names = [f"rf-{y}" for y in range(FIRST_RF, LAST_RF + 1)] + [f"gfs-{y}" for y in range(FIRST_GFS, today.year + 1)]
    return names


def live(name, today):
    """Chunks still collecting days: this year's GFS, and last year's for the first weeks of January."""
    src, year = name.split("-")
    return src == "gfs" and (int(year) == today.year or (int(year) == today.year - 1 and today.timetuple().tm_yday <= 15))


def built_on(path):
    """The day a chunk was built (saved inside it; a downloaded file's own date is the download)."""
    import numpy as np
    try:
        return dt.date.fromisoformat(str(np.load(path)["built"]))
    except Exception:  # noqa: BLE001  unreadable or old: rebuild
        return dt.date(1970, 1, 1)


def todo(data, today):
    out = []
    for name in chunk_names(today):
        path = os.path.join(data, f"{name}.npz")
        if not os.path.exists(path):
            out.append(name)
        elif live(name, today) and (today - built_on(path)).days > FRESH_DAYS:
            out.append(name)
    # The live chunk first (keeps learning), then history oldest first.
    return sorted(out, key=lambda n: (not live(n, today), n))


def upload_release(tag, path):
    subprocess.run(["gh", "release", "upload", tag, path, "--clobber"], check=True)


def build(a):
    today = dt.date.today()
    os.makedirs(a.data, exist_ok=True)
    stations = load_stations(os.path.join(a.data, "stations.json"), today)
    print(f"{len(stations)} stations", flush=True)
    if a.release:
        upload_release(a.release, os.path.join(a.data, "stations.json"))
    t0 = time.time()
    for name in todo(a.data, today):
        if (time.time() - t0) / 60 > a.max_minutes:
            print("time's up; the next run carries on", flush=True)
            break
        src, year = name.split("-")
        X, Y, M = build_chunk(src, int(year), stations, today=today, log=lambda m: print(m, flush=True))
        path = os.path.join(a.data, f"{name}.npz")
        save_chunk(path, X, Y, M)
        print(f"{name}: {len(X):,} rows saved", flush=True)
        if a.release:
            upload_release(a.release, path)


def train(a):
    import json
    from ml.train import main as train_main
    model = train_main(a.data, a.out, log=lambda m: print(m, flush=True))
    if a.upload:
        if len(os.environ.get("MODELS_UPLOAD_KEY", "")) < 24:
            print("MODELS_UPLOAD_KEY is not set; not uploading")
            return
        from render_models import send
        with open(a.out, "rb") as f:
            send("PUT", "/weather/models/upload/ml/local-v1.json", f.read(), "application/json")
        report = os.path.splitext(a.out)[0] + "-report.json"
        with open(report, "rb") as f:
            send("PUT", "/weather/models/upload/ml/local-v1-report.json", f.read(), "application/json")
        print("uploaded the model and its report", flush=True)
    print(json.dumps(model["report"], indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True)
    b.add_argument("--max-minutes", type=float, default=300)
    b.add_argument("--release", default="")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--upload", action="store_true")
    a = p.parse_args()
    (build if a.cmd == "build" else train)(a)


if __name__ == "__main__":
    main()
