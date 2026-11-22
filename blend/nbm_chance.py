#!/usr/bin/env python3
"""Model-blend rain and snow chance for https://atmosdays.eagleeyeinsightsllc.com

The National Blend of Models (NBM) is NOAA's hourly mix of about 30 weather
models and their ensembles (GFS, European, Canadian, HRRR, NAM and more),
corrected against what actually happened. It is free public data on AWS
(s3://noaa-nbm-grib2-pds), and it is the starting point NWS forecasters use for
their own forecast.

The NWS point forecast is that blend after a forecaster has edited it. Most of
the time the two agree; sometimes a forecaster trims a low chance to near zero
(say 6%) while the models still give 20 to 40%. Weather apps that use model
data then show a much higher number than the NWS. AtmosDays shows both.

This script takes the newest NBM run that reaches 7 days, keeps the chance of
at least 0.01 inch of rain or melted snow (the same measure as the NWS "chance
of precipitation") for every 6-hour and 12-hour window ending at 00, 06, 12 and
18 UTC over the next 7 days, puts it on a plain latitude/longitude grid (0.06
degrees, about 6 km; the chance changes slowly from place to place) and uploads
one binary file plus a small JSON description, so the website can read one
place with a single tiny ranged read.

  python blend/nbm_chance.py --test     # build and print checks, upload nothing
  python blend/nbm_chance.py --upload   # build and upload (needs MODELS_UPLOAD_KEY)

Binary layout: uint8 percent, [row][col][window][p06, p12], rows from south to
north, columns from west to east. 255 = missing. Window end times are in the JSON.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import eccodes
from scipy.spatial import cKDTree

UA = "AtmosDays-Blend/0.1 (https://atmosdays.eagleeyeinsightsllc.com)"
BUCKET = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"

LAT0, LAT1, LON0, LON1, STEP = 22.0, 51.0, -128.0, -64.0, 0.06
MISSING = 255
DAYS = 7


def get(url, headers=None, tries=4, timeout=90):
    err = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 416):
                raise
            err = e
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(2 ** i)
    raise err


def exists(url):
    try:
        get(url, headers={"Range": "bytes=0-0"}, tries=2, timeout=30)
        return True
    except urllib.error.HTTPError:
        return False


def file_url(run, fhr):
    return f"{BUCKET}/blend.{run:%Y%m%d}/{run:%H}/core/blend.t{run:%H}z.core.f{fhr:03d}.co.grib2"


def plan(run):
    """Forecast hours whose valid time is 00, 06, 12 or 18 UTC, from 6 h to 7 days ahead."""
    first = 6 - run.hour % 6 if run.hour % 6 else 6
    return list(range(first, DAYS * 24 + 6, 6))


def newest_run(now=None):
    """Newest NBM run that already has its 7-day file (runs finish about 1.5 hours after their hour)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    start = now.replace(minute=0, second=0, microsecond=0)
    for back in range(1, 13):
        run = start - dt.timedelta(hours=back)
        fh = plan(run)
        if exists(file_url(run, fh[-1]) + ".idx") and exists(file_url(run, fh[0]) + ".idx"):
            return run, fh
    raise SystemExit("no NBM run with 7 days found in the last 12 hours")


def ranges(idx_text, fhr):
    """Byte ranges of the 6-hour and 12-hour chance of 0.01 inch (0.254 mm) ending at this hour."""
    lines = [l.split(":") for l in idx_text.strip().splitlines()]
    out = {}
    for i, p in enumerate(lines):
        if len(p) < 7 or p[3] != "APCP" or p[4] != "surface" or p[6] != "prob >0.254":
            continue
        m = re.match(r"(\d+)-(\d+) hour acc fcst", p[5])
        if not m or int(m.group(2)) != fhr:
            continue
        span = fhr - int(m.group(1))
        if span in (6, 12):
            end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
            out[span] = (int(p[1]), end)
    return out


def decode(msg, want_latlon=False):
    gid = eccodes.codes_new_from_message(msg)
    try:
        vals = eccodes.codes_get_values(gid).astype(np.float64)
        miss = eccodes.codes_get(gid, "missingValue")
        vals[vals == miss] = np.nan
        vals[np.abs(vals) > 1e19] = np.nan
        # NBM stores every other row right to left ("alternative row scanning"),
        # but eccodes gives the latitudes and longitudes in plain order. Put the
        # values in plain order too, or every second row lands on the wrong spot.
        if eccodes.codes_get(gid, "alternativeRowScanning") == 1:
            nx, ny = eccodes.codes_get(gid, "Nx"), eccodes.codes_get(gid, "Ny")
            grid = vals.reshape(ny, nx)
            grid[1::2] = grid[1::2, ::-1].copy()
            vals = grid.ravel()
        ll = None
        if want_latlon:
            ll = (eccodes.codes_get_array(gid, "latitudes"), eccodes.codes_get_array(gid, "longitudes"))
        return vals, ll
    finally:
        eccodes.codes_release(gid)


def fetch_hour(run, fhr):
    url = file_url(run, fhr)
    rng = ranges(get(url + ".idx").decode(), fhr)
    msgs = {}
    for span, (a, b) in rng.items():
        msgs[span] = get(url, headers={"Range": f"bytes={a}-{'' if b is None else b}"})
    return fhr, msgs


def build(run, hours):
    with ThreadPoolExecutor(max_workers=6) as ex:
        got = dict(ex.map(lambda f: fetch_hour(run, f), hours))

    latlon = None
    for fhr in hours:
        for msg in got[fhr].values():
            _, latlon = decode(msg, want_latlon=True)
            break
        if latlon:
            break
    if latlon is None:
        raise SystemExit("no chance fields found in this run")
    lats, lons = latlon
    lons = np.where(lons > 180, lons - 360, lons)
    coslat = np.cos(np.radians(37.0))
    tree = cKDTree(np.column_stack([lats, lons * coslat]))
    glat = np.arange(LAT0, LAT1 + 1e-9, STEP)
    glon = np.arange(LON0, LON1 + 1e-9, STEP)
    ny, nx = len(glat), len(glon)
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    dist, near = tree.query(np.column_stack([LA.ravel(), LO.ravel() * coslat]), distance_upper_bound=0.05)
    inside = np.isfinite(dist)
    near = np.where(inside, near, 0)

    out = np.full((ny * nx, len(hours), 2), MISSING, dtype=np.uint8)
    found = {6: 0, 12: 0}
    for w, fhr in enumerate(hours):
        for k, span in enumerate((6, 12)):
            msg = got[fhr].get(span)
            if msg is None:
                continue
            vals, _ = decode(msg)
            v = vals[near]
            ok = inside & np.isfinite(v)
            out[ok, w, k] = np.clip(np.rint(v[ok]), 0, 100).astype(np.uint8)
            found[span] += 1
    if found[6] < len(hours) * 0.8 or found[12] < len(hours) * 0.8:
        raise SystemExit(f"too few chance fields: {found} of {len(hours)}")
    ends = [(run + dt.timedelta(hours=f)).strftime("%Y-%m-%dT%H:00:00Z") for f in hours]
    meta = {
        "source": "NOAA National Blend of Models (NBM), s3://noaa-nbm-grib2-pds",
        "measure": "chance of at least 0.01 inch of rain or melted snow",
        "lat0": LAT0, "lon0": LON0, "step": STEP, "ny": ny, "nx": nx,
        "ends": ends, "spans": [6, 12], "bytesPerCell": len(hours) * 2, "missing": MISSING,
    }
    return out.reshape(ny, nx, len(hours), 2), meta


def cell(meta, lat, lon):
    return int(round((lat - meta["lat0"]) / meta["step"])), int(round((lon - meta["lon0"]) / meta["step"]))


def put(path, body, ctype):
    key = os.environ["MODELS_UPLOAD_KEY"]
    site = os.environ.get("MODELS_SITE", "https://atmosdays.eagleeyeinsightsllc.com").rstrip("/")
    req = urllib.request.Request(f"{site}/weather/models/upload/{path}", data=body, method="PUT", headers={
        "Authorization": f"Bearer {key}", "Content-Type": ctype, "User-Agent": UA})
    err = None
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                print(r.read().decode())
                return
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise
            err = e
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(2 ** i)
    raise err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--run", help="YYYYMMDDHH (default: newest with 7 days)")
    a = ap.parse_args()
    t0 = time.time()
    if a.run:
        run = dt.datetime.strptime(a.run, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
        hours = plan(run)
    else:
        run, hours = newest_run()
    grid, meta = build(run, hours)
    meta["run"] = run.strftime("%Y-%m-%dT%H:00:00Z")
    # Two files take turns, so the website never reads a half-written one.
    slot = "a" if run.hour % 2 == 0 else "b"
    meta["file"] = f"blend/nbm/{slot}.bin"
    body = grid.tobytes()
    print(f"NBM {meta['run']} -> {meta['ny']}x{meta['nx']} cells x {len(hours)} windows, {len(body) / 1e6:.1f} MB, {time.time() - t0:.0f}s")
    for name, lat, lon in [("Midland TX", 31.9974, -102.0779), ("KMAF airport", 31.9425, -102.2019), ("Dallas TX", 32.90, -97.04), ("Denver CO", 39.86, -104.67), ("Seattle WA", 47.45, -122.31)]:
        r, c = cell(meta, lat, lon)
        row = grid[r, c]
        print(name, " ".join(f"{e[5:13]}:{p6}/{p12}" for e, (p6, p12) in zip(meta["ends"], row.tolist())))
    if a.upload:
        if not os.environ.get("MODELS_UPLOAD_KEY"):
            print("MODELS_UPLOAD_KEY is not set; nothing uploaded")
            return
        put(meta["file"], body, "application/octet-stream")
        put("blend/nbm/latest.json", json.dumps(meta, separators=(",", ":")).encode(), "application/json")
    elif not a.test:
        print("(dry run: pass --upload to send it)")


if __name__ == "__main__":
    sys.exit(main())
