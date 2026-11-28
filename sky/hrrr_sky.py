#!/usr/bin/env python3
"""Sunset and sunrise sky inputs for https://atmosdays.eagleeyeinsightsllc.com/weather/sunset

The website's "Sunset quality" score (0 to 100, for sunsets and sunrises) comes
from the clouds at the place and toward the sun at the moment the sun sets or
rises. This script reads those clouds from NOAA's HRRR model (3 km, free on AWS,
s3://noaa-hrrr-bdp-pds), downloading only the few fields it needs with byte-range
reads, and writes one small grid per sunset or sunrise:

  for every 0.1 degree cell of the lower 48, at that cell's own sunset (or sunrise)
  time, interpolated between the model hours:
    high, middle, low and total cloud cover overhead (%), visibility, humidity,
    rain in the 6 hours before and in the hour before, and the low and total
    cloud 25, 75 and 150 km out toward the sun.

The score itself is worked out on the website (engine/src/skyscore.js), so it has
one home and can be tuned without touching this job.

  python sky/hrrr_sky.py --test      # build, print checks for a few places, upload nothing
  python sky/hrrr_sky.py --upload    # build and upload (needs MODELS_UPLOAD_KEY)
  python sky/hrrr_sky.py --test --run 2026112612   # a past run (AWS keeps them)

Binary layout per event: uint8, [row][col][field], rows south to north, columns
west to east, 255 = missing. Fields and units are in the event's JSON file.
Each event goes in a slot (date mod 10), so storage never grows:
  sky/sunset-<slot>.bin + sky/sunset-<slot>.json, same for sunrise.
"""
import argparse
import datetime as dt
import email.utils
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import eccodes

UA = "AtmosDays-Sky/0.1 (https://atmosdays.eagleeyeinsightsllc.com)"
BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
GFS_BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

LAT0, LAT1, LON0, LON1, STEP = 22.0, 51.0, -128.0, -64.0, 0.1
MISSING = 255
PATH_KM = (25, 75, 150)

# Stored fields (one byte each) and what the number means.
FIELDS = [
    ("high", "high cloud cover %"),
    ("mid", "middle cloud cover %"),
    ("low", "low cloud cover %"),
    ("total", "total cloud cover %"),
    ("vis", "visibility, tenths of a mile, 250 = 25 mi or more"),
    ("rh", "relative humidity at 2 m %"),
    ("rain6", "rain in the 6 hours before, hundredths of an inch (250 = 2.5 in or more)"),
    ("rain1", "rain in the hour before, hundredths of an inch"),
    ("low25", "low cloud 25 km toward the sun %"),
    ("low75", "low cloud 75 km toward the sun %"),
    ("low150", "low cloud 150 km toward the sun %"),
    ("tot25", "total cloud 25 km toward the sun %"),
    ("tot75", "total cloud 75 km toward the sun %"),
    ("tot150", "total cloud 150 km toward the sun %"),
    ("fh", "model forecast hour at the event"),
    ("az", "sun's compass bearing at the event / 2 (degrees)"),
]

# GRIB index labels (".idx" lines) for the fields read each hour.
CLOUDS = ["LCDC:low cloud layer", "MCDC:middle cloud layer", "HCDC:high cloud layer", "TCDC:entire atmosphere"]
OTHER = ["VIS:surface", "RH:2 m above ground"]
RAIN = "APCP:surface:0-"  # rain since the start of the run


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
    return posted(url) is not None


def posted(url):
    """When a file was posted (from its Last-Modified header), or None when it isn't there."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            lm = r.headers.get("Last-Modified")
            return email.utils.parsedate_to_datetime(lm) if lm else dt.datetime.now(dt.timezone.utc)
    except Exception:  # noqa: BLE001
        return None


class Run:
    """One model run: HRRR (3 km, every hour, 18 or 48 hours out) or GFS (25 km, day 3)."""

    def __init__(self, model, time, hours):
        self.model, self.time, self.hours = model, time, hours

    def url(self, fh):
        t = self.time
        if self.model == "GFS":
            return f"{GFS_BUCKET}/gfs.{t:%Y%m%d}/{t:%H}/atmos/gfs.t{t:%H}z.pgrb2.0p25.f{fh:03d}"
        return f"{BUCKET}/hrrr.{t:%Y%m%d}/conus/hrrr.t{t:%H}z.wrfsfcf{fh:02d}.grib2"

    def timestamp(self):
        return self.time.timestamp()

    def hour_list(self):
        """Forecast hours with a file: GFS is hourly to 120, then every 3 hours."""
        if self.model == "GFS":
            return list(range(0, 121)) + list(range(123, self.hours + 1, 3))
        return list(range(0, self.hours + 1))

    def __repr__(self):
        return f"{self.model} {self.time:%d %HZ} ({self.hours}h)"


def run_length(run):
    """HRRR runs at 00, 06, 12 and 18 UTC go out 48 hours; the others 18."""
    return 48 if run.hour % 6 == 0 else 18


GFS_HOURS = 180  # far enough for 7 sunsets and 7 sunrises (Weather+)
DAYS = 7         # how many sunsets and sunrises ahead


def recent_runs(now, back=8):
    """HRRR runs from newest to oldest whose last hour is posted, then the newest GFS run."""
    base = now.replace(minute=0, second=0, microsecond=0)
    out = []
    for h in range(0, back + 1):
        t = base - dt.timedelta(hours=h)
        r = Run("HRRR", t, run_length(t))
        if exists(r.url(r.hours) + ".idx"):
            out.append(r)
    six = base.replace(hour=base.hour - base.hour % 6)
    for h in range(0, 25, 6):
        r = Run("GFS", six - dt.timedelta(hours=h), GFS_HOURS)
        r.posted = posted(r.url(r.hours) + ".idx")
        if r.posted:
            out.append(r)
            break
    return out


# ---- Sun (vectorized NOAA solar calculator, the same method as engine/src/astro.js) ----

def sun_events(lat, lon, ymd, rising):
    """UTC epoch seconds of sunrise or sunset on local date `ymd` and the sun's bearing then."""
    off = np.round(-lon / 15) * -60  # minutes, rough local offset (only picks the day)
    day0 = dt.datetime.strptime(ymd, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp()
    noon_local = day0 + 12 * 3600 - off * 60

    def one(approx):
        jd = approx / 86400 + 2440587.5
        t = (jd - 2451545) / 36525
        L0 = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360
        M = 357.52911 + t * (35999.05029 - 0.0001537 * t)
        e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
        R = math.pi / 180
        C = np.sin(M * R) * (1.914602 - t * (0.004817 + 0.000014 * t)) + np.sin(2 * M * R) * (0.019993 - 0.000101 * t) + np.sin(3 * M * R) * 0.000289
        omega = 125.04 - 1934.136 * t
        lam = L0 + C - 0.00569 - 0.00478 * np.sin(omega * R)
        eps0 = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
        eps = eps0 + 0.00256 * np.cos(omega * R)
        decl = np.arcsin(np.sin(eps * R) * np.sin(lam * R))
        y = np.tan(eps / 2 * R) ** 2
        eq = 4 / R * (y * np.sin(2 * L0 * R) - 2 * e * np.sin(M * R) + 4 * e * y * np.sin(M * R) * np.cos(2 * L0 * R) - 0.5 * y * y * np.sin(4 * L0 * R) - 1.25 * e * e * np.sin(2 * M * R))
        cosH = np.cos(90.833 * R) / (np.cos(lat * R) * np.cos(decl)) - np.tan(lat * R) * np.tan(decl)
        ha = np.degrees(np.arccos(np.clip(cosH, -1, 1)))
        noon_min = 720 - 4 * lon - eq
        mins = noon_min - 4 * ha if rising else noon_min + 4 * ha
        ms = day0 + mins * 60
        ms = np.where(ms < noon_local - 43200, ms + 86400, np.where(ms > noon_local + 43200, ms - 86400, ms))
        # Bearing at the horizon: cos(az) = sin(decl) / cos(lat), from north.
        az = np.degrees(np.arccos(np.clip(np.sin(decl) / np.cos(lat * R), -1, 1)))
        return ms, (az if rising else 360 - az), np.abs(cosH) <= 1

    a, _, _ = one(noon_local)
    b, az, ok = one(a)
    return np.where(ok, b, np.nan), az


# ---- Grid ----

def target_grid():
    glat = np.arange(LAT0, LAT1 + 1e-9, STEP)
    glon = np.arange(LON0, LON1 + 1e-9, STEP)
    return glat, glon


class Regrid:
    """Mean of the model points inside each 0.1 degree cell (HRRR); the nearest point for a
    coarser model (GFS), whose points are farther apart than the cells."""

    def __init__(self, lats, lons):
        lons = np.where(lons > 180, lons - 360, lons)
        glat, glon = target_grid()
        self.ny, self.nx = len(glat), len(glon)
        r = np.floor((lats - LAT0) / STEP + 0.5).astype(int)
        c = np.floor((lons - LON0) / STEP + 0.5).astype(int)
        ok = (r >= 0) & (r < self.ny) & (c >= 0) & (c < self.nx)
        self.ok = ok
        self.idx = (r * self.nx + c)[ok]
        n = np.bincount(self.idx, minlength=self.ny * self.nx)
        self.near = None
        if (n == 0).mean() > 0.2:
            from scipy.spatial import cKDTree
            sel = (lats > LAT0 - 1) & (lats < LAT1 + 1) & (lons > LON0 - 1) & (lons < LON1 + 1)
            self.src = np.flatnonzero(sel)
            tree = cKDTree(np.column_stack([lats[sel], lons[sel]]))
            LA, LO = np.meshgrid(glat, glon, indexing="ij")
            self.near = tree.query(np.column_stack([LA.ravel(), LO.ravel()]))[1]

    def __call__(self, vals):
        if self.near is not None:
            return vals[self.src][self.near]
        v = vals[self.ok]
        good = np.isfinite(v)
        s = np.bincount(self.idx[good], weights=v[good], minlength=self.ny * self.nx)
        n = np.bincount(self.idx[good], minlength=self.ny * self.nx)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / n, np.nan)


def byte_ranges(idx_text, wanted):
    lines = [l.split(":") for l in idx_text.strip().splitlines()]
    out = {}
    for i, p in enumerate(lines):
        label = ":".join(p[3:6])
        for w in wanted:
            if w not in out and label.startswith(w):
                start = int(p[1])
                end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
                out[w] = (start, end)
    return out


def decode(msg, want_latlon=False):
    gid = eccodes.codes_new_from_message(msg)
    try:
        vals = eccodes.codes_get_values(gid).astype(np.float64)
        miss = eccodes.codes_get(gid, "missingValue")
        vals[vals == miss] = np.nan
        vals[np.abs(vals) > 1e19] = np.nan
        ll = (eccodes.codes_get_array(gid, "latitudes"), eccodes.codes_get_array(gid, "longitudes")) if want_latlon else None
        return vals, ll
    finally:
        eccodes.codes_release(gid)


class Reader:
    """Downloads only the needed fields of each hour, once, and keeps them on the 0.1 degree grid."""

    def __init__(self, run):
        self.run = run
        self.cache = {}
        self.regrid = None
        self.bytes = 0
        self.reads = 0

    def hour(self, fh, rain_only=False):
        key = (fh, rain_only)
        full = self.cache.get((fh, False))
        if full is not None:
            return full
        if key in self.cache:
            return self.cache[key]
        if fh == 0:
            out = {"rain": None}  # nothing has fallen at the start of the run
            self.cache[key] = out
            return out
        url = self.run.url(fh)
        wanted = [RAIN] if rain_only else CLOUDS + OTHER + [RAIN]
        rng = byte_ranges(get(url + ".idx").decode(), wanted)
        missing = [w for w in wanted if w not in rng]
        if missing:
            raise SystemExit(f"{url}: fields not in the file: {missing}")
        # Neighbouring fields come in one read.
        spans = sorted(rng.items(), key=lambda kv: kv[1][0])
        groups = []
        for name, (a, b) in spans:
            if groups and b is not None and groups[-1][2] is not None and a == groups[-1][2] + 1:
                groups[-1][1].append(name)
                groups[-1][2] = b
            else:
                groups.append([a, [name], b])
        out = {}
        for a, names, b in groups:
            data = get(url, headers={"Range": f"bytes={a}-{'' if b is None else b}"})
            self.bytes += len(data)
            self.reads += 1
            pos = 0
            for name in names:
                n = rng[name][1] - rng[name][0] + 1 if rng[name][1] is not None else len(data) - pos
                vals, ll = decode(data[pos:pos + n], want_latlon=self.regrid is None)
                pos += n
                if ll is not None:
                    self.regrid = Regrid(*ll)
                out[name] = self.regrid(vals)
        res = {"rain": out[RAIN] if RAIN in out else None}
        if not rain_only:
            res.update({
                "low": out[CLOUDS[0]], "mid": out[CLOUDS[1]], "high": out[CLOUDS[2]], "total": out[CLOUDS[3]],
                "vis": out[OTHER[0]] / 1609.344, "rh": out[OTHER[1]],
            })
        self.cache[key] = res
        return res


def interp(reader, fhf, name, rain_only=False):
    """A field at each cell's own (fractional) forecast hour, between the model hours around it."""
    hrs = np.array(reader.run.hour_list())
    out = np.full(fhf.shape, np.nan)
    ok = np.isfinite(fhf)
    k = np.searchsorted(hrs, np.where(ok, fhf, 0), side="right") - 1
    k = np.clip(k, 0, len(hrs) - 2)
    for i in np.unique(k[ok]):
        sel = ok & (k == i)
        h0, h1 = int(hrs[i]), int(hrs[i + 1])
        w = (fhf[sel] - h0) / (h1 - h0)
        a = reader.hour(h0, rain_only)[name]
        b = reader.hour(h1, rain_only)[name]
        if name == "rain":
            a = np.zeros_like(b) if a is None else a
        out[sel] = a[sel] * (1 - w) + b[sel] * w
    return out


def build_event(reader, kind, ymd, run):
    glat, glon = target_grid()
    ny, nx = len(glat), len(glon)
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    t, az = sun_events(LA.ravel(), LO.ravel(), ymd, rising=(kind == "sunrise"))
    fhf = (t - run.timestamp()) / 3600
    hours = run.hours
    fhf = np.where((fhf >= 1) & (fhf <= hours - 1e-6), fhf, np.nan)
    cols = {}
    for name in ("high", "mid", "low", "total", "vis", "rh"):
        cols[name] = interp(reader, fhf, name)
    acc = interp(reader, fhf, "rain", rain_only=False)
    acc6 = interp(reader, np.maximum(fhf - 6, 0), "rain", rain_only=True)
    acc1 = interp(reader, np.maximum(fhf - 1, 0), "rain", rain_only=True)
    mm_to_hin = 1 / 25.4 * 100
    cols["rain6"] = np.maximum(acc - acc6, 0) * mm_to_hin
    cols["rain1"] = np.maximum(acc - acc1, 0) * mm_to_hin
    # Toward the sun: the cell 25, 75 and 150 km out along the sun's bearing.
    lat, lon = LA.ravel(), LO.ravel()
    rad = np.radians(az)
    for km in PATH_KM:
        plat = lat + km * np.cos(rad) / 111.2
        plon = lon + km * np.sin(rad) / (111.2 * np.cos(np.radians(lat)))
        r = np.rint((plat - LAT0) / STEP).astype(int)
        c = np.rint((plon - LON0) / STEP).astype(int)
        inside = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
        j = np.where(inside, r * nx + c, 0)
        for src, dst in (("low", f"low{km}"), ("total", f"tot{km}")):
            v = cols[src][j]
            cols[dst] = np.where(inside, v, np.nan)
    cols["vis"] = np.minimum(cols["vis"], 25) * 10
    cols["fh"] = np.rint(fhf)
    cols["az"] = az / 2
    out = np.full((ny * nx, len(FIELDS)), MISSING, dtype=np.uint8)
    valid = np.isfinite(fhf)
    for i, (name, _) in enumerate(FIELDS):
        v = cols[name]
        ok = valid & np.isfinite(v)
        out[ok, i] = np.clip(np.rint(v[ok]), 0, 250).astype(np.uint8)
    tv = t[np.isfinite(fhf)]
    meta = {
        "kind": kind, "date": ymd, "model": run.model, "run": run.time.strftime("%Y-%m-%dT%H:00Z"),
        "source": "NOAA HRRR 3 km (s3://noaa-hrrr-bdp-pds), averaged to 0.1 degree" if run.model == "HRRR"
        else "NOAA GFS 0.25 degree (s3://noaa-gfs-bdp-pds), nearest point",
        "lat0": LAT0, "lon0": LON0, "step": STEP, "ny": ny, "nx": nx,
        "fields": [f[0] for f in FIELDS], "notes": dict(FIELDS), "pathKm": list(PATH_KM),
        "bytesPerCell": len(FIELDS), "missing": MISSING,
        "cells": int(valid.sum()),
        "first": dt.datetime.fromtimestamp(float(tv.min()), dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ") if tv.size else None,
        "last": dt.datetime.fromtimestamp(float(tv.max()), dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ") if tv.size else None,
    }
    return out.reshape(ny, nx, len(FIELDS)), meta


def event_span(kind, ymd):
    """Earliest and latest event time (epoch s) over the grid, from a coarse sample."""
    lat = np.arange(LAT0, LAT1 + 1e-9, 1.0)
    lon = np.arange(LON0, LON1 + 1e-9, 1.0)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    t, _ = sun_events(LA.ravel(), LO.ravel(), ymd, rising=(kind == "sunrise"))
    t = t[np.isfinite(t)]
    return float(t.min()) - 900, float(t.max()) + 900


def plan(now, runs):
    """Each upcoming sunset and sunrise, matched to the newest run that covers all of it."""
    events = []
    for kind in ("sunset", "sunrise"):
        n = 0
        for k in range(-1, DAYS + 2):
            ymd = (now + dt.timedelta(days=k)).strftime("%Y-%m-%d")
            a, b = event_span(kind, ymd)
            if b < now.timestamp():
                continue
            n += 1
            if n > DAYS:  # the next 7 sunsets and the next 7 sunrises
                break
            for run in runs:  # HRRR first (newest first), then GFS
                r = run.timestamp()
                if a >= r + 3600 and b <= r + run.hours * 3600:
                    events.append((kind, ymd, run))
                    break
    return events


def slot_of(ymd):
    return (dt.date.fromisoformat(ymd).toordinal()) % 10


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


CHECKS = [("Fort Worth TX", 32.75, -97.33), ("Denver CO", 39.74, -104.99), ("Seattle WA", 47.61, -122.33),
          ("Miami FL", 25.76, -80.19), ("Boston MA", 42.36, -71.06), ("San Diego CA", 32.72, -117.16)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--run", help="use this run (YYYYMMDDHH) instead of the newest, for checks")
    ap.add_argument("--max-events", type=int, default=2 * DAYS)
    ap.add_argument("--first", action="store_true", help="only the first sunset and the first sunrise (for replays)")
    ap.add_argument("--all-gfs", action="store_true", help="rebuild the GFS days even when the GFS run is not new")
    ap.add_argument("--out", help="also save the files in this folder (for local checks)")
    a = ap.parse_args()
    t0 = time.time()
    if a.run:
        run = dt.datetime.strptime(a.run, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
        runs = [Run("HRRR", run, run_length(run))]
        six = run.replace(hour=run.hour - run.hour % 6)
        runs.append(Run("GFS", six, GFS_HOURS))
        now = run
    else:
        now = dt.datetime.now(dt.timezone.utc)
        runs = recent_runs(now)
    if not runs:
        raise SystemExit("no complete HRRR or GFS run found")
    events = plan(now, runs)
    # GFS changes every 6 hours: redo its days only in the first hour or so after a new run.
    if not a.all_gfs and not a.run:
        events = [e for e in events if e[2].model != "GFS" or (now - e[2].posted).total_seconds() < 75 * 60]
    if a.first:
        events = [next(e for e in events if e[0] == k) for k in ("sunset", "sunrise") if any(e[0] == k for e in events)]
    events = events[: a.max_events]
    print("runs:", runs[:5])
    print("events:", [(k, d, r) for k, d, r in events])
    readers = {}
    for kind, ymd, run in events:
        t1 = time.time()
        reader = readers.setdefault((run.model, run.time), Reader(run))
        grid, meta = build_event(reader, kind, ymd, run)
        slot = slot_of(ymd)
        meta["file"] = f"sky/{kind}-{slot}.bin"
        body = grid.tobytes()
        print(f"{kind} {ymd} from {run.model} {meta['run']}: {meta['cells']} cells, {meta['first']} to {meta['last']}, "
              f"{len(body) / 1e6:.1f} MB, {time.time() - t1:.0f}s")
        for name, lat, lon in CHECKS:
            r = int(round((lat - LAT0) / STEP))
            c = int(round((lon - LON0) / STEP))
            print(f"  {name}:", dict(zip(meta["fields"], grid[r, c].tolist())))
        if a.out:
            os.makedirs(os.path.join(a.out, "sky"), exist_ok=True)
            with open(os.path.join(a.out, meta["file"]), "wb") as f:
                f.write(body)
            with open(os.path.join(a.out, f"sky/{kind}-{slot}.json"), "w") as f:
                json.dump(meta, f, separators=(",", ":"))
        if a.upload:
            if not os.environ.get("MODELS_UPLOAD_KEY"):
                print("MODELS_UPLOAD_KEY is not set; nothing uploaded")
                return
            put(meta["file"], body, "application/octet-stream")
            put(f"sky/{kind}-{slot}.json", json.dumps(meta, separators=(",", ":")).encode(), "application/json")
    total_mb = sum(r.bytes for r in readers.values()) / 1e6
    reads = sum(r.reads for r in readers.values())
    print(f"downloaded {total_mb:.0f} MB in {reads} byte-range reads; done in {time.time() - t0:.0f}s")
    if not a.upload and not a.test:
        print("(dry run: pass --upload to send it)")


if __name__ == "__main__":
    sys.exit(main())
