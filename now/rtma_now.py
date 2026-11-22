#!/usr/bin/env python3
"""15-minute "right now" weather for https://atmosdays.eagleeyeinsightsllc.com

NOAA's Real-Time Mesoscale Analysis, Rapid Update (RTMA-RU) is a map of the
weather near the ground over the lower 48 states, made every 15 minutes from
every surface station NOAA receives (airports, state networks, buoys and more)
blended with a short model forecast. It is free public data on AWS
(s3://noaa-rtma-pds). New maps show up about 15 to 20 minutes after their time.

This script takes the newest map, keeps the fields the current-conditions card
shows, puts them on a plain latitude/longitude grid (0.04 degrees, about 4 km)
and uploads one binary file plus a small JSON description, so the website can
read one place with a single tiny ranged read.

  python now/rtma_now.py --test             # build and print checks, upload nothing
  python now/rtma_now.py --upload           # build and upload (needs MODELS_UPLOAD_KEY)

Binary layout: little-endian int16, [row][col][field], rows from south to north,
columns from west to east. -32768 = missing. Fields and scales are in the JSON.
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

import numpy as np
import eccodes
from scipy.spatial import cKDTree

UA = "AtmosDays-Now/0.1 (https://atmosdays.eagleeyeinsightsllc.com)"
BUCKET = "https://noaa-rtma-pds.s3.amazonaws.com"

# Target grid (covers the RTMA CONUS domain).
LAT0, LAT1, LON0, LON1, STEP = 22.0, 51.0, -128.0, -64.0, 0.04

MISSING = -32768
NONE = 32767  # "no ceiling" (sky clear or only scattered clouds)

# name, GRIB idx match, scale note. Values are stored as int16.
FIELDS = [
    ("tF", "TMP:2 m above ground", "deg F x10"),
    ("tdF", "DPT:2 m above ground", "deg F x10"),
    ("mph", "WIND:10 m above ground", "mph x10"),
    ("dir", "WDIR:10 m above ground", "degrees from, true north"),
    ("gust", "GUST:10 m above ground", "mph x10"),
    ("slp", "PRES:surface", "sea-level pressure hPa x10 (reduced from surface pressure)"),
    ("vis", "VIS:surface", "miles x10"),
    ("ceil", "CEIL:cloud ceiling", "feet / 10; 32767 = no ceiling"),
    ("sky", "TCDC:entire atmosphere", "cloud cover %"),
]
EXTRA = ["HGT:surface"]  # terrain height, for the sea-level pressure


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


def newest_run(now=None):
    """Newest analysis file that has its .idx: (valid datetime, url)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    for back in (0, 1):
        day = (now - dt.timedelta(days=back)).strftime("%Y%m%d")
        xml = get(f"{BUCKET}/?list-type=2&prefix=rtma2p5_ru.{day}/&max-keys=1000").decode()
        keys = set(re.findall(r"<Key>([^<]+)</Key>", xml))
        runs = sorted(k for k in keys if k.endswith(".2dvaranl_ndfd.grb2") and k + ".idx" in keys)
        if runs:
            k = runs[-1]
            m = re.search(r"t(\d{2})(\d{2})z", k)
            valid = dt.datetime.strptime(day, "%Y%m%d").replace(hour=int(m.group(1)), minute=int(m.group(2)), tzinfo=dt.timezone.utc)
            return valid, f"{BUCKET}/{k}"
    raise SystemExit("no RTMA-RU analysis found for today or yesterday")


def byte_ranges(idx_text, wanted):
    """{match: (start, end)} from a GRIB .idx file."""
    lines = [l.split(":") for l in idx_text.strip().splitlines()]
    out = {}
    for i, p in enumerate(lines):
        label = ":".join(p[3:5])
        for w in wanted:
            if label.startswith(w):
                start = int(p[1])
                end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
                out[w] = (start, end)
    return out


def decode(msg_bytes, want_latlon=False):
    gid = eccodes.codes_new_from_message(msg_bytes)
    try:
        vals = eccodes.codes_get_values(gid).astype(np.float64)
        miss = eccodes.codes_get(gid, "missingValue")
        vals[vals == miss] = np.nan
        vals[np.abs(vals) > 1e19] = np.nan
        ll = None
        if want_latlon:
            ll = (eccodes.codes_get_array(gid, "latitudes"), eccodes.codes_get_array(gid, "longitudes"))
        return vals, ll
    finally:
        eccodes.codes_release(gid)


def build(url):
    idx = get(url + ".idx").decode()
    wanted = [f[1] for f in FIELDS] + EXTRA
    rng = byte_ranges(idx, wanted)
    missing = [w for w in wanted if w not in rng]
    if missing:
        raise SystemExit(f"fields not in this file: {missing}")
    raw, latlon = {}, None
    for w in wanted:
        a, b = rng[w]
        data = get(url, headers={"Range": f"bytes={a}-{'' if b is None else b}"})
        vals, ll = decode(data, want_latlon=latlon is None)
        if ll is not None:
            latlon = ll
        raw[w] = vals

    lats, lons = latlon
    lons = np.where(lons > 180, lons - 360, lons)
    # Nearest RTMA point for each target cell (2.5 km source, 4 km target).
    coslat = np.cos(np.radians(37.0))
    tree = cKDTree(np.column_stack([lats, lons * coslat]))
    glat = np.arange(LAT0, LAT1 + 1e-9, STEP)
    glon = np.arange(LON0, LON1 + 1e-9, STEP)
    ny, nx = len(glat), len(glon)
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    dist, near = tree.query(np.column_stack([LA.ravel(), LO.ravel() * coslat]), distance_upper_bound=0.05)
    inside = np.isfinite(dist)
    near = np.where(inside, near, 0)

    def pick(arr):
        v = arr[near].astype(np.float64)
        v[~inside] = np.nan
        return v

    k_to_f = lambda k: (k - 273.15) * 9 / 5 + 32  # noqa: E731
    t = pick(raw["TMP:2 m above ground"])
    td = pick(raw["DPT:2 m above ground"])
    p = pick(raw["PRES:surface"])
    z = pick(raw["HGT:surface"])
    # Sea-level pressure from surface pressure, terrain height and temperature (standard reduction).
    slp = p * (1 - 0.0065 * z / (t + 0.0065 * z)) ** -5.257 / 100.0
    ceil_m = pick(raw["CEIL:cloud ceiling"])
    ceil = np.where(ceil_m > 12000, NONE, ceil_m * 3.28084 / 10)
    ceil = np.where(np.isnan(ceil_m), np.nan, ceil)

    cols = {
        "tF": k_to_f(t) * 10,
        "tdF": k_to_f(td) * 10,
        "mph": pick(raw["WIND:10 m above ground"]) * 2.23694 * 10,
        "dir": pick(raw["WDIR:10 m above ground"]),
        "gust": pick(raw["GUST:10 m above ground"]) * 2.23694 * 10,
        "slp": slp * 10,
        "vis": np.minimum(pick(raw["VIS:surface"]) / 1609.344, 100) * 10,
        "ceil": ceil,
        "sky": pick(raw["TCDC:entire atmosphere"]),
    }
    out = np.full((ny * nx, len(FIELDS)), MISSING, dtype="<i2")
    for i, (name, _, _) in enumerate(FIELDS):
        v = cols[name]
        ok = np.isfinite(v)
        out[ok, i] = np.clip(np.rint(v[ok]), -32767, 32767).astype("<i2")
    meta = {
        "source": "NOAA RTMA Rapid Update (2.5 km, every 15 minutes), s3://noaa-rtma-pds",
        "lat0": LAT0, "lon0": LON0, "step": STEP, "ny": ny, "nx": nx,
        "fields": [f[0] for f in FIELDS], "notes": {f[0]: f[2] for f in FIELDS},
        "bytesPerCell": 2 * len(FIELDS), "missing": MISSING, "noCeiling": NONE,
    }
    return out.reshape(ny, nx, len(FIELDS)), meta


def cell(meta, lat, lon):
    r = int(round((lat - meta["lat0"]) / meta["step"]))
    c = int(round((lon - meta["lon0"]) / meta["step"]))
    return r, c


def put(path, body, ctype):
    key = os.environ["MODELS_UPLOAD_KEY"]
    site = os.environ.get("MODELS_SITE", "https://atmosdays.eagleeyeinsightsllc.com").rstrip("/")
    req = urllib.request.Request(f"{site}/weather/models/upload/{path}", data=body, method="PUT", headers={
        "Authorization": f"Bearer {key}", "Content-Type": ctype, "User-Agent": UA})
    err = None
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
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
    a = ap.parse_args()
    t0 = time.time()
    valid, url = newest_run()
    grid, meta = build(url)
    meta["valid"] = valid.strftime("%Y-%m-%dT%H:%M:00Z")
    # Two files take turns, so the website never reads a half-written one.
    slot = "a" if (valid.hour * 4 + valid.minute // 15) % 2 == 0 else "b"
    meta["file"] = f"now/rtma/{slot}.bin"
    body = grid.tobytes()
    print(f"RTMA-RU {meta['valid']} -> {meta['ny']}x{meta['nx']} cells, {len(body) / 1e6:.1f} MB, {time.time() - t0:.0f}s")
    for name, lat, lon in [("Midland TX", 31.95, -102.19), ("Dallas TX", 32.90, -97.04), ("Denver CO", 39.86, -104.67), ("Miami FL", 25.79, -80.29)]:
        r, c = cell(meta, lat, lon)
        print(name, dict(zip(meta["fields"], grid[r, c].tolist())))
    if a.upload:
        if not os.environ.get("MODELS_UPLOAD_KEY"):
            print("MODELS_UPLOAD_KEY is not set; nothing uploaded")
            return
        put(meta["file"], body, "application/octet-stream")
        put("now/rtma/latest.json", json.dumps(meta, separators=(",", ":")).encode(), "application/json")
    elif not a.test:
        print("(dry run: pass --upload to send it)")


if __name__ == "__main__":
    sys.exit(main())
