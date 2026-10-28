"""Draw forecast model maps for AtmosDays Weather (/weather/models).

Downloads the newest NOAA GFS and ECMWF open-data runs (North America, 10
days) and the NOAA HRRR and HREF short-range storm models (lower 48, 2 days)
from their free public copies (no key needed; only the few fields the maps
use, by byte range), draws one picture per map and forecast hour in the same
Mercator projection as the site's maps, and uploads them to the website,
which keeps them in Cloudflare R2 storage.

  python render_models.py --out out            # draw only, into ./out
  python render_models.py --upload             # draw and upload (needs
      MODELS_UPLOAD_KEY and optionally MODELS_SITE in the environment)

Data: NOAA NCEP GFS, HRRR and HREF (public domain) and ECMWF open data (CC BY 4.0).
Runs from .github/workflows/model-maps.yml a few times a day.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import eccodes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
ECMWF = "https://ecmwf-forecasts.s3.amazonaws.com"
HRRR = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HREF = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod"
UA = "AtmosDays-Weather-Models/0.1 (contact@eagleeyeinsightsllc.com)"

# Map areas: (west, south, east, north) and picture width in pixels.
AREAS = {
    "na": ((-135.0, 15.0, -55.0, 60.0), 1400),     # North America, Caribbean and the Gulf
    "conus": ((-126.0, 21.5, -66.0, 50.5), 1600),  # lower 48 (HRRR and HREF cover this)
}
BBOX = AREAS["na"][0]
HOURS = list(range(0, 241, 6))
FIELDS = ["temp", "precip", "wind", "upper"]
KEEP_RUNS = 2  # newest run plus the one before

# HRRR wind at every spot in the lower 48, for the website's "wind at crane
# height" on each job: a 0.1 degree grid, one byte (mph) per field and hour,
# stored cell by cell so the website reads one spot with a single small read.
POINTS = {"west": -126.0, "south": 21.5, "step": 0.1, "nx": 600, "ny": 290,
          "fields": ["w10", "w80", "gust"], "missing": 255}


# ---- colors -------------------------------------------------------------------

def hexc(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


LEGENDS = {
    "temp": {"label": "Temperature", "unit": "°F", "stops": [
        [-30, "#e9dcf5"], [-10, "#8c6bb1"], [0, "#4a3fa8"], [10, "#2c5bc9"], [20, "#3d8fe0"], [32, "#9bd6f2"],
        [33, "#b4e3a8"], [45, "#5cb85c"], [55, "#c6db58"], [65, "#f5e050"], [75, "#f6a838"], [85, "#ef6a2a"],
        [95, "#d42a2a"], [105, "#9c0f3f"], [115, "#5e0a2e"]], "note": "Blue line: freezing"},
    "precip": {"label": "Rain and melted snow, 6 hours", "unit": "in", "stops": [
        [0.01, "#bdf0b5"], [0.1, "#4cc94c"], [0.25, "#1e9e2a"], [0.5, "#f2e33b"], [0.75, "#f39c1f"],
        [1.0, "#e2321e"], [1.5, "#b0157f"], [2.5, "#7a2fbf"]], "note": "Lines: sea-level pressure, mb"},
    "wind": {"label": "Wind at 10 m", "unit": "mph", "stops": [
        [5, "#d8ecf3"], [10, "#9fd3e6"], [20, "#5ab4d6"], [30, "#3b8f4f"], [40, "#e8d54a"], [50, "#f08a2a"],
        [60, "#d7301f"], [75, "#8c1d6e"]]},
    "upper": {"label": "Jet stream (250 mb wind)", "unit": "mph", "stops": [
        [50, "#d6ebff"], [70, "#8fc4ff"], [90, "#4f8ff0"], [110, "#7b4fe0"], [130, "#c43fc4"], [160, "#ff6fae"]],
        "note": "Lines: 500 mb height, dam"},
    "radar": {"label": "Computer radar (what the model expects radar to show)", "unit": "dBZ", "steps": True, "stops": [
        [5, "#9be7f5"], [10, "#3fb7ee"], [15, "#2b6fe0"], [20, "#36d33a"], [25, "#20a72b"], [30, "#147a1d"],
        [35, "#f5ea2c"], [40, "#e7b81b"], [45, "#f58a12"], [50, "#e8221b"], [55, "#c01414"], [60, "#9b0c0c"],
        [65, "#f02df0"], [70, "#9a4fd0"]]},
    "rain1": {"label": "Rain and melted snow, 1 hour", "unit": "in", "stops": [
        [0.01, "#bdf0b5"], [0.05, "#4cc94c"], [0.1, "#1e9e2a"], [0.25, "#f2e33b"], [0.5, "#f39c1f"],
        [0.75, "#e2321e"], [1.0, "#b0157f"], [2.0, "#7a2fbf"]], "note": "Lines: sea-level pressure, mb"},
    "rain6": {"label": "Rain and melted snow, 6 hours", "unit": "in", "stops": [
        [0.01, "#bdf0b5"], [0.1, "#4cc94c"], [0.25, "#1e9e2a"], [0.5, "#f2e33b"], [0.75, "#f39c1f"],
        [1.0, "#e2321e"], [1.5, "#b0157f"], [2.5, "#7a2fbf"]]},
    "gust": {"label": "Wind gusts", "unit": "mph", "stops": [
        [10, "#d8ecf3"], [20, "#9fd3e6"], [30, "#5ab4d6"], [40, "#3b8f4f"], [50, "#e8d54a"], [60, "#f08a2a"],
        [70, "#d7301f"], [85, "#8c1d6e"]]},
    "uh": {"label": "Storm rotation (updraft helicity, 1-hour max)", "unit": "m²/s²", "stops": [
        [25, "#fde68a"], [50, "#f59e0b"], [75, "#ef4444"], [100, "#b91c1c"], [150, "#a21caf"], [250, "#4c1d95"]],
        "note": "Streaks show where the model makes rotating storms"},
    "dew": {"label": "Dew point (how humid it feels)", "unit": "°F", "stops": [
        [-10, "#6b4f3a"], [20, "#8b6b4a"], [30, "#b89a6a"], [40, "#d6c79a"], [50, "#9ed79a"], [55, "#4cbb5c"],
        [60, "#1f9a4b"], [65, "#16a3a3"], [70, "#1f6fd1"], [75, "#5b3fc4"], [80, "#9b2fae"]], "note": "60°F and up feels muggy"},
    "cape": {"label": "Storm fuel (CAPE)", "unit": "J/kg", "stops": [
        [100, "#e0f2fe"], [500, "#a7f3d0"], [1000, "#fde047"], [2000, "#fb923c"], [3000, "#ef4444"], [4000, "#be185d"],
        [5000, "#7e22ce"]], "note": "More fuel means stronger updrafts if storms form"},
    "qpf": {"label": "Total rain and melted snow since the run started", "unit": "in", "stops": [
        [0.01, "#bdf0b5"], [0.25, "#4cc94c"], [0.5, "#1e9e2a"], [1.0, "#f2e33b"], [1.5, "#f39c1f"], [2.0, "#e2321e"],
        [3.0, "#b0157f"], [5.0, "#7a2fbf"], [8.0, "#c4b5fd"]]},
    "snow": {"label": "Total snowfall since the run started", "unit": "in", "stops": [
        [0.1, "#dbeafe"], [1, "#93c5fd"], [2, "#60a5fa"], [4, "#2563eb"], [6, "#1e3a8a"], [8, "#7c3aed"], [12, "#c026d3"],
        [18, "#f472b6"], [24, "#fecdd3"]], "note": "ECMWF: 10 inches of snow per inch of water"},
    "hail": {"label": "Biggest hail in the hour", "unit": "in", "stops": [
        [0.25, "#a7f3d0"], [0.5, "#facc15"], [0.75, "#fb923c"], [1.0, "#ef4444"], [1.5, "#be123c"], [2.0, "#a21caf"],
        [3.0, "#4c1d95"]], "note": "1 inch (quarter size) and up is severe"},
    "ltng": {"label": "Lightning", "unit": "flashes/km² per 5 min", "stops": [
        [0.05, "#fef9c3"], [0.25, "#fde047"], [0.5, "#f59e0b"], [1, "#ef4444"], [2, "#b91c1c"], [4, "#7e22ce"]]},
    "prob": {"label": "Chance among the HREF models", "unit": "%", "stops": [
        [10, "#d9f99d"], [20, "#86efac"], [30, "#22c55e"], [40, "#facc15"], [50, "#fb923c"], [60, "#ef4444"],
        [70, "#dc2626"], [80, "#be185d"], [90, "#7e22ce"]]},
}

# Picture name -> label for the map list on the page.
FIELD_NAMES = {
    "temp": "Temperature", "precip": "Rain and snow, with pressure", "wind": "Wind", "upper": "Jet stream and 500 mb",
    "radar": "Computer radar", "rain1": "Rain, 1 hour", "rain6": "Rain, 6 hours", "gust": "Wind gusts",
    "uh": "Storm rotation tracks", "storms40": "Chance of strong storms", "rotation": "Chance of rotating storms",
    "heavy": "Chance of 1 inch of rain in 6 hours",
    "dew": "Dew point (humidity)", "cape": "Storm fuel (CAPE)", "qpf": "Total rain since the run started",
    "snow": "Total snowfall", "hail": "Hail size", "ltng": "Lightning",
}
# Legend used by each picture.
FIELD_LEGEND = {"precip": "precip", "storms40": "prob", "rotation": "prob", "heavy": "prob"}

# Below these values the map stays see-through.
FADE = {"temp": None, "precip": (0.005, 0.02), "wind": (5, 15), "upper": (45, 70), "radar": (4, 6),
        "rain1": (0.005, 0.015), "rain6": (0.005, 0.02), "gust": (10, 20), "uh": (15, 30), "prob": (5, 12),
        "dew": None, "cape": (100, 300), "qpf": (0.005, 0.03), "snow": (0.05, 0.3), "hail": (0.2, 0.3), "ltng": (0.02, 0.08)}


def colorize(values, field):
    leg = LEGENDS[field]
    stops = leg["stops"]
    xs = np.array([s[0] for s in stops], float)
    cols = np.array([hexc(s[1]) for s in stops], float)
    v = np.nan_to_num(values, nan=-9999)
    rgba = np.zeros(values.shape + (4,), np.uint8)
    if leg.get("steps"):
        idx = np.clip(np.searchsorted(xs, v, side="right") - 1, 0, len(xs) - 1)
        rgba[..., :3] = cols[idx].astype(np.uint8)
    else:
        for c in range(3):
            rgba[..., c] = np.interp(v, xs, cols[:, c]).astype(np.uint8)
    fade = FADE[field]
    if fade is None:
        alpha = np.full(values.shape, 215.0)
    else:
        alpha = np.clip((v - fade[0]) / (fade[1] - fade[0]), 0, 1) * 215
    alpha[np.isnan(values)] = 0
    rgba[..., 3] = alpha.astype(np.uint8)
    return rgba


# ---- projection ---------------------------------------------------------------

def merc_y(lat):
    s = np.sin(np.radians(lat))
    return 0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)


class Area:
    """Latitude and longitude of every output pixel (Mercator) for one map area."""

    def __init__(self, bbox, width):
        w, s, e, n = bbox
        y0, y1 = merc_y(n), merc_y(s)
        self.bbox, self.width = bbox, width
        self.height = int(round(width * (y1 - y0) / ((e - w) / 360)))
        ys = y0 + (np.arange(self.height) + 0.5) / self.height * (y1 - y0)
        self.lats = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * ys))))
        self.lons = w + (np.arange(width) + 0.5) / width * (e - w)


class PointArea:
    """A regular latitude/longitude grid (south to north), for sampling model fields at spots."""

    def __init__(self, spec):
        st = spec["step"]
        self.lats = spec["south"] + np.arange(spec["ny"]) * st
        self.lons = spec["west"] + np.arange(spec["nx"]) * st


_areas = {}


def area(name):
    if name not in _areas:
        _areas[name] = Area(*AREAS[name])
    return _areas[name]


class Grid:
    """A regular lat/lon field; .sample(area) resamples it onto the output pixels."""

    def __init__(self, values, lat0, dlat, lon0, dlon):
        self.values, self.lat0, self.dlat, self.lon0, self.dlon = values, lat0, dlat, lon0, dlon

    def sample(self, ar):
        lon = ar.lons.copy()
        if self.lon0 >= 0:
            lon = np.where(lon < 0, lon + 360, lon)
        rows = (ar.lats - self.lat0) / self.dlat
        cols = (lon - self.lon0) / self.dlon
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        return map_coordinates(self.values, [rr, cc], order=1, mode="nearest")


class LambertGrid:
    """A Lambert conformal field (HRRR, HREF) on a sphere; pixels outside it come out blank."""

    def __init__(self, values, lat0, lon0, lad, lov, latin1, latin2, dx, dy, radius):
        r = np.radians
        self.values, self.dx, self.dy, self.radius = values, dx, dy, radius
        self.lov = r(((lov + 180) % 360) - 180)
        p1, p2 = r(latin1), r(latin2)
        if abs(latin1 - latin2) < 1e-6:
            self.n = np.sin(p1)
        else:
            self.n = np.log(np.cos(p1) / np.cos(p2)) / np.log(np.tan(np.pi / 4 + p2 / 2) / np.tan(np.pi / 4 + p1 / 2))
        self.f = np.cos(p1) * np.tan(np.pi / 4 + p1 / 2) ** self.n / self.n
        self.rho0 = self._rho(r(lad))
        self.x0, self.y0 = self.xy(lat0, ((lon0 + 180) % 360) - 180)

    def _rho(self, phi):
        return self.radius * self.f / np.tan(np.pi / 4 + phi / 2) ** self.n

    def xy(self, lat, lon):
        rho = self._rho(np.radians(lat))
        dl = np.radians(lon) - self.lov
        dl = (dl + np.pi) % (2 * np.pi) - np.pi
        t = self.n * dl
        return rho * np.sin(t), self.rho0 - rho * np.cos(t)

    def sample(self, ar):
        la, lo = np.meshgrid(ar.lats, ar.lons, indexing="ij")
        x, y = self.xy(la, lo)
        cols, rows = (x - self.x0) / self.dx, (y - self.y0) / self.dy
        ny, nx = self.values.shape
        out = map_coordinates(np.nan_to_num(self.values, nan=0), [rows, cols], order=1, mode="nearest")
        bad = map_coordinates(np.isnan(self.values).astype(np.float32), [rows, cols], order=1, mode="nearest") > 0.5
        out[bad | (rows < -0.5) | (cols < -0.5) | (rows > ny - 0.5) | (cols > nx - 0.5)] = np.nan
        return out


def decode(msg_bytes):
    gid = eccodes.codes_new_from_message(msg_bytes)
    try:
        g = lambda k: eccodes.codes_get(gid, k)  # noqa: E731
        vals = eccodes.codes_get_values(gid)
        missing = g("missingValue")
        if g("bitmapPresent"):
            vals = np.where(vals == missing, np.nan, vals)
        if g("gridType") == "lambert":
            nx, ny = g("Nx"), g("Ny")
            vals = vals.reshape(ny, nx)
            if not g("jScansPositively"):
                vals = vals[::-1]
            lat0 = g("latitudeOfFirstGridPointInDegrees")
            if not g("jScansPositively"):
                lat0 = g("latitudeOfLastGridPointInDegrees") if eccodes.codes_is_defined(gid, "latitudeOfLastGridPointInDegrees") else lat0
            try:
                radius = g("radius")
            except Exception:  # noqa: BLE001
                radius = 6371229.0
            return LambertGrid(vals, lat0, g("longitudeOfFirstGridPointInDegrees"), g("LaDInDegrees"), g("LoVInDegrees"),
                               g("Latin1InDegrees"), g("Latin2InDegrees"), g("DxInMetres"), g("DyInMetres"), radius)
        ni, nj = g("Ni"), g("Nj")
        lat0, lat1 = g("latitudeOfFirstGridPointInDegrees"), g("latitudeOfLastGridPointInDegrees")
        lon0, dlon = g("longitudeOfFirstGridPointInDegrees"), g("iDirectionIncrementInDegrees")
        vals = vals.reshape(nj, ni)
    finally:
        eccodes.codes_release(gid)
    if lon0 > 180:
        lon0 -= 360
    dlat = (lat1 - lat0) / (nj - 1)
    return Grid(vals, lat0, dlat, lon0, dlon)


# ---- downloads ----------------------------------------------------------------

_nomads_lock = threading.Lock()
_nomads_last = [0.0]


def polite(url):
    """NOMADS allows about 120 requests a minute; stay well under it."""
    if "nomads.ncep.noaa.gov" not in url:
        return
    with _nomads_lock:
        wait = _nomads_last[0] + 0.7 - time.time()
        if wait > 0:
            time.sleep(wait)
        _nomads_last[0] = time.time()


def get(url, headers=None, tries=6):
    for i in range(tries):
        polite(url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise
            err = e
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(2 ** i)
    raise err


def exists(url):
    polite(url)
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def ranges_get(url, spans):
    return {k: get(url, {"Range": f"bytes={a}-{b}"}) for k, (a, b) in spans.items()}


def idx_spans(text, want):
    """Byte ranges from a NOAA .idx listing. want: {name: test(var, level, fcst, rest)}."""
    rows = [line.split(":") for line in text.splitlines() if line.strip()]
    offs = [int(r[1]) for r in rows]
    spans = {}
    for i, r in enumerate(rows):
        end = offs[i + 1] - 1 if i + 1 < len(rows) else ""
        var, level, fcst, rest = r[3], r[4], r[5] if len(r) > 5 else "", ":".join(r[6:])
        for name, test in want.items():
            if name not in spans and test(var, level, fcst, rest):
                spans[name] = (offs[i], end)
    return spans


def k_to_f(k):
    return (k - 273.15) * 9 / 5 + 32


# GFS: the .idx file lists each field with its byte offset.
GFS_WANT = {
    "t2": "TMP:2 m above ground:", "msl": "PRMSL:mean sea level:", "u10": "UGRD:10 m above ground:",
    "v10": "VGRD:10 m above ground:", "z500": "HGT:500 mb:", "u250": "UGRD:250 mb:", "v250": "VGRD:250 mb:",
    "d2": "DPT:2 m above ground:", "cape": "CAPE:surface:",
}


def gfs_url(run, h):
    return f"{GFS}/gfs.{run[:8]}/{run[8:]}/atmos/gfs.t{run[8:]}z.pgrb2.0p25.f{h:03d}"


# Only the 00 and 12 UTC GFS runs are drawn (like ECMWF): this keeps the job
# inside GitHub's free 2,000 minutes a month.
def gfs_latest(now):
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= dt.timedelta(hours=t.hour % 12)
    for _ in range(4):
        run = t.strftime("%Y%m%d%H")
        if exists(gfs_url(run, HOURS[-1]) + ".idx"):
            return run
        t -= dt.timedelta(hours=12)
    return None


def gfs_fields(run, h):
    idx = get(gfs_url(run, h) + ".idx").decode().splitlines()
    rows = [line.split(":") for line in idx]
    offs = [int(r[1]) for r in rows]
    spans = {}
    for i, r in enumerate(rows):
        key = ":".join(r[3:5]) + ":"
        end = offs[i + 1] - 1 if i + 1 < len(rows) else ""
        for name, want in GFS_WANT.items():
            if key == want and name not in spans:
                spans[name] = (offs[i], end)
        if h >= 6 and r[3] == "APCP" and r[5] == f"{h - 6}-{h} hour acc fcst" and "tp6" not in spans:
            spans["tp6"] = (offs[i], end)
        # Total since the run started: "0-30 hour acc fcst", or "0-2 day acc fcst" on whole days.
        if h >= 6 and r[3] == "APCP" and r[5] in (f"0-{h} hour acc fcst", f"0-{h // 24} day acc fcst") and "tpt" not in spans:
            spans["tpt"] = (offs[i], end)
    raw = ranges_get(gfs_url(run, h), spans)
    ar = area("na")
    g = {k: decode(v).sample(ar) for k, v in raw.items()}
    out = {
        "t2": (g["t2"] - 273.15) * 9 / 5 + 32, "msl": g["msl"] / 100,
        "w10": np.hypot(g["u10"], g["v10"]) * 2.23694, "z500": g["z500"] / 10,
        "w250": np.hypot(g["u250"], g["v250"]) * 2.23694,
        "tp6": g["tp6"] / 25.4 if "tp6" in g else None,  # kg/m2 = mm
        "d2": k_to_f(g["d2"]) if "d2" in g else None, "cape": g.get("cape"),
        "qpf": g["tpt"] / 25.4 if "tpt" in g else None,
    }
    return out


# ECMWF open data: a JSON-lines .index file with byte offsets.
EC_WANT = {("2t", None): "t2", ("msl", None): "msl", ("10u", None): "u10", ("10v", None): "v10",
           ("gh", "500"): "z500", ("u", "250"): "u250", ("v", "250"): "v250", ("tp", None): "tp",
           ("2d", None): "d2", ("mucape", None): "cape", ("sf", None): "sf"}


def ec_url(run, h):
    return f"{ECMWF}/{run[:8]}/{run[8:]}z/ifs/0p25/oper/{run}0000-{h}h-oper-fc"


def ec_latest(now):
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= dt.timedelta(hours=t.hour % 12)
    for _ in range(4):
        run = t.strftime("%Y%m%d%H")
        if exists(ec_url(run, HOURS[-1]) + ".index"):
            return run
        t -= dt.timedelta(hours=12)
    return None


_ec_tp = {}


def ec_tp(run, h):
    """Total precipitation since the start of the run (inches)."""
    if h == 0:
        return None
    key = (run, h)
    if key not in _ec_tp:
        idx = [json.loads(x) for x in get(ec_url(run, h) + ".index").decode().splitlines() if x.strip()]
        r = next(x for x in idx if x["param"] == "tp")
        raw = get(ec_url(run, h) + ".grib2", {"Range": f"bytes={r['_offset']}-{r['_offset'] + r['_length'] - 1}"})
        _ec_tp[key] = decode(raw).sample(area("na")) * 1000 / 25.4  # m -> in
    return _ec_tp[key]


def ec_fields(run, h):
    idx = [json.loads(x) for x in get(ec_url(run, h) + ".index").decode().splitlines() if x.strip()]
    spans = {}
    for r in idx:
        name = EC_WANT.get((r["param"], r.get("levelist")))
        if name and name != "tp" and name not in spans:
            spans[name] = (r["_offset"], r["_offset"] + r["_length"] - 1)
    raw = ranges_get(ec_url(run, h) + ".grib2", spans)
    ar = area("na")
    g = {k: decode(v).sample(ar) for k, v in raw.items()}
    tp6 = None
    if h >= 6:
        now_tp, before = ec_tp(run, h), ec_tp(run, h - 6)
        tp6 = np.clip(now_tp - (before if before is not None else 0), 0, None)
    return {
        "t2": (g["t2"] - 273.15) * 9 / 5 + 32, "msl": g["msl"] / 100,
        "w10": np.hypot(g["u10"], g["v10"]) * 2.23694, "z500": g["z500"] / 10,
        "w250": np.hypot(g["u250"], g["v250"]) * 2.23694, "tp6": tp6,
        "d2": k_to_f(g["d2"]) if "d2" in g else None, "cape": g.get("cape"),
        "qpf": ec_tp(run, h) if h >= 6 else None,
        # Snowfall comes as meters of melted water; 10 inches of snow per inch of water.
        "snow": g["sf"] * 39.37 * 10 if h >= 6 and "sf" in g else None,
    }


# HRRR: NOAA's hourly-updated 3 km storm model; the 00/06/12/18 UTC runs go out
# 48 hours. Same .idx style as GFS.
def hrrr_url(run, h):
    return f"{HRRR}/hrrr.{run[:8]}/conus/hrrr.t{run[8:]}z.wrfsfcf{h:02d}.grib2"


def short_latest(url, last_hour, every=6):
    def latest(now):
        t = now.replace(minute=0, second=0, microsecond=0)
        t -= dt.timedelta(hours=t.hour % every)
        for _ in range(4):
            run = t.strftime("%Y%m%d%H")
            if exists(url(run, last_hour) + ".idx"):
                return run
            t -= dt.timedelta(hours=every)
        return None
    return latest


def hour_window(h, n):
    return f"{h - n}-{h} hour"


def hrrr_fields(run, h):
    win = hour_window(h, 1)
    # Totals since the run started: "0-12 hour acc fcst", or "0-1 day acc fcst" on whole days.
    total = (f"0-{h} hour acc fcst", f"0-{h // 24} day acc fcst")
    want = {
        "refc": lambda v, lv, f, r: v == "REFC" and lv == "entire atmosphere",
        "t2": lambda v, lv, f, r: v == "TMP" and lv == "2 m above ground",
        "gust": lambda v, lv, f, r: v == "GUST" and lv == "surface",
        "msl": lambda v, lv, f, r: v == "MSLMA" and lv == "mean sea level",
        "tp1": lambda v, lv, f, r: h >= 1 and v == "APCP" and lv == "surface" and f == f"{win} acc fcst",
        "uh": lambda v, lv, f, r: h >= 1 and v == "MXUPHL" and lv == "5000-2000 m above ground" and f == f"{win} max fcst",
        "d2": lambda v, lv, f, r: v == "DPT" and lv == "2 m above ground",
        "cape": lambda v, lv, f, r: v == "CAPE" and lv == "surface",
        "tpt": lambda v, lv, f, r: h >= 1 and v == "APCP" and lv == "surface" and f in total,
        "snow": lambda v, lv, f, r: h >= 1 and v == "ASNOW" and lv == "surface" and f in total,
        # Hail at the ground (the "entire atmosphere" one counts hail aloft that melts on the way down).
        "hail": lambda v, lv, f, r: h >= 1 and v == "HAIL" and lv == "surface" and f == f"{win} max fcst",
        "ltng": lambda v, lv, f, r: v == "LTNG" and lv == "entire atmosphere",
        # Wind at 10 m (33 ft) and 80 m (about 260 ft, crane and lift height) for the job pages.
        "u10": lambda v, lv, f, r: v == "UGRD" and lv == "10 m above ground",
        "v10": lambda v, lv, f, r: v == "VGRD" and lv == "10 m above ground",
        "u80": lambda v, lv, f, r: v == "UGRD" and lv == "80 m above ground",
        "v80": lambda v, lv, f, r: v == "VGRD" and lv == "80 m above ground",
    }
    spans = idx_spans(get(hrrr_url(run, h) + ".idx").decode(), want)
    raw = ranges_get(hrrr_url(run, h), spans)
    ar = area("conus")
    grids = {k: decode(v) for k, v in raw.items()}
    pts = None
    if all(k in grids for k in ("u10", "v10", "u80", "v80", "gust")):
        pa = PointArea(POINTS)
        at = {k: grids[k].sample(pa) for k in ("u10", "v10", "u80", "v80", "gust")}
        # Speed doesn't depend on the grid's wind direction convention, so no rotation is needed.
        pts = {"w10": np.hypot(at["u10"], at["v10"]) * 2.23694, "w80": np.hypot(at["u80"], at["v80"]) * 2.23694,
               "gust": at["gust"] * 2.23694}
    g = {k: v.sample(ar) for k, v in grids.items() if k not in ("u10", "v10", "u80", "v80")}
    return {
        "refc": g["refc"], "t2": k_to_f(g["t2"]), "gust": g["gust"] * 2.23694, "msl": g["msl"] / 100,
        "tp1": g["tp1"] / 25.4 if "tp1" in g else None, "uh": g.get("uh"),
        "d2": k_to_f(g["d2"]) if "d2" in g else None, "cape": g.get("cape"),
        "qpf": g["tpt"] / 25.4 if "tpt" in g else None,
        "snow": g["snow"] * 39.37 if "snow" in g else None,  # m -> in
        "hail": g["hail"] * 39.37 if "hail" in g else None,  # m -> in
        "ltng": g.get("ltng"),
        "pts": pts,
    }


# HREF: NOAA's storm-model ensemble (several 3 km models blended), out to 48 hours,
# from NOMADS. Only the 00 and 12 UTC runs are drawn (free Actions minutes). pmmn = blended amounts, mean = average,
# prob = chance among the members (neighborhood probability).
def href_url(run, h, product="pmmn"):
    return f"{HREF}/href.{run[:8]}/ensprod/href.t{run[8:]}z.conus.{product}.f{h:02d}.grib2"


def prob_over(rest, x):
    return f"prob >{x}:" in rest + ":"


def href_fields(run, h):
    w1, w3 = hour_window(h, 1), hour_window(h, 3)
    parts = {
        "pmmn": {
            "refc": lambda v, lv, f, r: v == "REFC" and lv.startswith("entire atmosphere"),
            "tp1": lambda v, lv, f, r: v == "APCP" and f == f"{w1} acc fcst",
        },
        "mean": {
            "t2": lambda v, lv, f, r: v == "TMP" and lv == "2 m above ground",
            "msl": lambda v, lv, f, r: v == "MSLET" and lv == "mean sea level",
        },
        "prob": {
            "p40": lambda v, lv, f, r: v == "REFC" and prob_over(r, 40),
            "puh": lambda v, lv, f, r: v == "MXUPHL" and lv == "5000-2000 m above ground" and prob_over(r, 75),
            "prain": lambda v, lv, f, r: h >= 3 and v == "APCP" and f == f"{w3} acc fcst" and prob_over(r, 25.4),
        },
    }
    ar = area("conus")
    g = {}
    for product, want in parts.items():
        url = href_url(run, h, product)
        spans = idx_spans(get(url + ".idx").decode(), want)
        for k, v in ranges_get(url, spans).items():
            g[k] = decode(v).sample(ar)

    def pct(a):
        if a is None:
            return None
        return a * 100 if np.nanmax(a) <= 1.0 else a
    return {
        "refc": g.get("refc"), "tp1": g["tp1"] / 25.4 if "tp1" in g else None, "t2": k_to_f(g["t2"]),
        "msl": g["msl"] / 100 if "msl" in g else None,
        "p40": pct(g.get("p40")), "puh": pct(g.get("puh")), "prain": pct(g.get("prain")),
    }


# ---- drawing ------------------------------------------------------------------

def picture(ar, fill_rgba, contours=None, sharp=False):
    """RGBA image with optional contour lines [(field, levels, color, width, labels, dashed)]."""
    width, height = ar.width, ar.height
    if not contours and fill_rgba is not None and fill_rgba.shape[:2] == (height, width):
        # No lines to draw: skip matplotlib (much faster, same picture).
        out = io.BytesIO()
        Image.fromarray(fill_rgba, "RGBA").save(out, format="WEBP", quality=80, method=4)
        return out.getvalue()
    if fill_rgba is None and not contours:
        out = io.BytesIO()
        Image.new("RGBA", (width, height), (0, 0, 0, 0)).save(out, format="WEBP", quality=80, method=4)
        return out.getvalue()
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    if fill_rgba is not None:
        ax.imshow(fill_rgba, extent=(0, width, height, 0), interpolation="nearest" if sharp else "bilinear")
    xs, ys = np.arange(width) + 0.5, np.arange(height) + 0.5
    for field, levels, color, width_, labels, dashed in contours or []:
        if field is None or not np.isfinite(field).any():
            continue
        lo, hi = np.nanmin(field), np.nanmax(field)
        levels = [x for x in levels if lo < x < hi]
        if not levels:
            continue
        cs = ax.contour(xs, ys, field, levels=levels, colors=color, linewidths=width_, linestyles="dashed" if dashed else "solid")
        if labels:
            ax.clabel(cs, fmt="%d", fontsize=8, inline=True, inline_spacing=2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=100)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    out = io.BytesIO()
    img.save(out, format="WEBP", quality=80, method=4)
    return out.getvalue()


def smooth(a, n=2):
    """A light box blur so contour lines come out smooth (blank areas stay blank)."""
    from scipy.ndimage import uniform_filter
    ok = np.isfinite(a)
    if ok.all():
        return uniform_filter(a, size=2 * n + 1, mode="nearest")
    num = uniform_filter(np.where(ok, a, 0.0), size=2 * n + 1, mode="nearest")
    den = uniform_filter(ok.astype(float), size=2 * n + 1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~ok] = np.nan
    return out


MSL_LEVELS = np.arange(940, 1068, 4)


def fill(values, legend):
    return colorize(values, legend) if values is not None else None


def temp_pic(ar, t2):
    return picture(ar, colorize(t2, "temp"), [(smooth(t2, 3), [32], "#1d4ed8", 1.4, False, True)])


def rain_pic(ar, tp, legend, msl, n=8):
    lines = [(smooth(msl, n), MSL_LEVELS, "#334155", 0.9, True, False)] if msl is not None else []
    return picture(ar, fill(tp, legend), lines)


def draw(fields, h):
    ar = area("na")
    out = {}
    out["temp"] = temp_pic(ar, fields["t2"])
    out["precip"] = rain_pic(ar, fields["tp6"], "precip", fields["msl"])
    out["wind"] = picture(ar, colorize(fields["w10"], "wind"))
    z = smooth(fields["z500"], 3)
    out["upper"] = picture(ar, colorize(fields["w250"], "upper"), [(z, np.arange(480, 612, 6), "#111827", 1.0, True, False)])
    out["dew"] = picture(ar, fill(fields.get("d2"), "dew"))
    out["cape"] = picture(ar, fill(fields.get("cape"), "cape"))
    out["qpf"] = picture(ar, fill(fields.get("qpf"), "qpf"))
    if "snow" in fields:
        out["snow"] = picture(ar, fill(fields["snow"], "snow"))
    return out


def draw_hrrr(f, h):
    ar = area("conus")
    return {
        "radar": picture(ar, fill(f["refc"], "radar"), sharp=True),
        "temp": temp_pic(ar, f["t2"]),
        "rain1": rain_pic(ar, f["tp1"], "rain1", f["msl"], 14),
        "gust": picture(ar, colorize(f["gust"], "gust")),
        "uh": picture(ar, fill(f["uh"], "uh"), sharp=True),
        "dew": picture(ar, fill(f["d2"], "dew")),
        "cape": picture(ar, fill(f["cape"], "cape")),
        "qpf": picture(ar, fill(f["qpf"], "qpf")),
        "snow": picture(ar, fill(f["snow"], "snow")),
        "hail": picture(ar, fill(f["hail"], "hail"), sharp=True),
        "ltng": picture(ar, fill(f["ltng"], "ltng")),
    }


def draw_href(f, h):
    ar = area("conus")
    return {
        "radar": picture(ar, fill(f["refc"], "radar"), sharp=True),
        "temp": temp_pic(ar, f["t2"]),
        "rain1": rain_pic(ar, f["tp1"], "rain1", f["msl"], 14),
        "storms40": picture(ar, fill(f["p40"], "prob")),
        "rotation": picture(ar, fill(f["puh"], "prob")),
        "heavy": picture(ar, fill(f["prain"], "prob")),
    }


# ---- model soundings (click anywhere on the website) -----------------------------
#
# For every spot on a grid, the temperature, moisture and wind at each pressure
# level, so the website can draw a sounding for any spot with one small read.
# One file per forecast hour; each spot is SND_BYTES bytes in a row:
#   surface: pressure, ground height, 2 m temperature, 2 m dew point gap, 10 m wind u, v
#   then per level (1000 mb up): height, temperature, dew point gap, wind u, v
# Each value is one byte (255 = missing); the scales are in the manifest.

def std_atmos(p_mb):
    """Standard-atmosphere height (m) and temperature (C) at a pressure, for byte offsets."""
    p = np.asarray(p_mb, dtype=float)
    z = 44330.8 * (1 - (p / 1013.25) ** 0.190263)
    z = np.where(p < 226.32, 11000 + 6341.6 * np.log(226.32 / p), z)
    t = np.where(z < 11000, 15 - 0.0065 * z, -56.5)
    return z, t


SND_SCALE = {
    # value = base + byte * step (base per level for z and t)
    "ps": [500.0, 2.5], "zs": [0.0, 20.0], "t2": [-60.0, 0.5], "dd": [0.0, 0.4], "uv": [-127.0, 1.0],
    "z": {"below": 640.0, "step": 5.0}, "t": {"below": 60.0, "step": 0.5},
}
HRRR_SND_LEVELS = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650, 600, 550, 500, 450,
                   400, 350, 300, 250, 200, 150, 100]
GFS_SND_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100]
SND = {
    "hrrr": {"grid": {"west": -126.0, "south": 21.5, "step": 0.125, "nx": 480, "ny": 233},
             "levels": HRRR_SND_LEVELS, "hours": list(range(0, 19)) + list(range(21, 49, 3)), "dew": "DPT"},
    "gfs": {"grid": {"west": -135.0, "south": 15.0, "step": 0.25, "nx": 321, "ny": 181},
            "levels": GFS_SND_LEVELS, "hours": list(range(0, 241, 6)), "dew": "RH"},
}


def snd_info(model):
    spec = SND[model]
    zb, tb = std_atmos(spec["levels"])
    return {"version": 1, **spec["grid"], "levels": spec["levels"],
            "zbase": [round(float(z) - SND_SCALE["z"]["below"]) for z in zb],
            "tbase": [round(float(t) - SND_SCALE["t"]["below"], 1) for t in tb],
            "scale": {k: v for k, v in SND_SCALE.items() if k not in ("z", "t")},
            "zstep": SND_SCALE["z"]["step"], "tstep": SND_SCALE["t"]["step"],
            "bytes": 6 + 5 * len(spec["levels"]), "missing": 255}


def snd_url(model, run, h):
    if model == "hrrr":
        return f"{HRRR}/hrrr.{run[:8]}/conus/hrrr.t{run[8:]}z.wrfprsf{h:02d}.grib2"
    return gfs_url(run, h)


class Sampler:
    """Samples model fields at the sounding grid's spots; the spot positions are worked out once per model grid."""

    def __init__(self, grid):
        self.lats = grid["south"] + np.arange(grid["ny"]) * grid["step"]
        self.lons = grid["west"] + np.arange(grid["nx"]) * grid["step"]
        self.cache = {}

    def coords(self, g):
        key = (type(g).__name__, g.values.shape, getattr(g, "x0", None), getattr(g, "lat0", None), getattr(g, "lon0", None))
        if key not in self.cache:
            la, lo = np.meshgrid(self.lats, self.lons, indexing="ij")
            if isinstance(g, LambertGrid):
                x, y = g.xy(la, lo)
                rows, cols = (y - g.y0) / g.dy, (x - g.x0) / g.dx
                ny, nx = g.values.shape
                out = (rows < -0.5) | (cols < -0.5) | (rows > ny - 0.5) | (cols > nx - 0.5)
                # Grid winds point along the grid; this angle turns them to true north.
                dl = np.radians(lo) - g.lov
                ang = g.n * ((dl + np.pi) % (2 * np.pi) - np.pi)
            else:
                lon = np.where(lo < 0, lo + 360, lo) if g.lon0 >= 0 else lo
                rows, cols = (la - g.lat0) / g.dlat, (lon - g.lon0) / g.dlon
                out, ang = np.zeros(la.shape, bool), None
            self.cache[key] = (rows, cols, out, ang)
        return self.cache[key]

    def sample(self, g):
        rows, cols, out, _ = self.coords(g)
        v = map_coordinates(g.values, [rows, cols], order=1, mode="nearest")
        v[out] = np.nan
        return v

    def wind(self, gu, gv):
        u, v = self.sample(gu), self.sample(gv)
        ang = self.coords(gu)[3]
        if ang is None:
            return u, v
        c, s = np.cos(ang), np.sin(ang)
        return c * u + s * v, -s * u + c * v


def dew_from_rh(t_c, rh):
    rh = np.clip(rh, 1, 100)
    a = np.log(rh / 100) + 17.625 * t_c / (243.04 + t_c)
    return 243.04 * a / (17.625 - a)


def snd_hour(model, run, h, sampler):
    """The packed sounding grid for one forecast hour (bytes)."""
    spec = SND[model]
    levels, dew = spec["levels"], spec["dew"]
    lv = {f"{p} mb": p for p in levels}
    want = {
        "ps": lambda v, l, f, r: v == "PRES" and l == "surface",
        "zs": lambda v, l, f, r: v == "HGT" and l == "surface",
        "t2": lambda v, l, f, r: v == "TMP" and l == "2 m above ground",
        "d2": lambda v, l, f, r: v == "DPT" and l == "2 m above ground",
        "u10": lambda v, l, f, r: v == "UGRD" and l == "10 m above ground",
        "v10": lambda v, l, f, r: v == "VGRD" and l == "10 m above ground",
    }
    for name, var in (("z", "HGT"), ("t", "TMP"), ("d", dew), ("u", "UGRD"), ("v", "VGRD")):
        for lvl, p in lv.items():
            want[f"{name}{p}"] = (lambda var, lvl: lambda v, l, f, r: v == var and l == lvl)(var, lvl)
    url = snd_url(model, run, h)
    spans = idx_spans(get(url + ".idx").decode(), want)
    missing = [k for k in want if k not in spans]
    if missing:
        raise RuntimeError(f"{model} +{h}h missing {missing[:5]}")
    info = snd_info(model)
    ny, nx = spec["grid"]["ny"], spec["grid"]["nx"]
    out = np.full((ny, nx, info["bytes"]), 255, dtype=np.uint8)

    def put(i, vals, base, step):
        q = np.rint((vals - base) / step)
        ok = np.isfinite(q)
        out[:, :, i] = np.where(ok, np.clip(np.nan_to_num(q), 0, 254), 255).astype(np.uint8)

    def grid(k):
        return decode(get(url, {"Range": f"bytes={spans[k][0]}-{spans[k][1]}"}))

    sc = SND_SCALE
    t2 = sampler.sample(grid("t2")) - 273.15
    d2 = sampler.sample(grid("d2")) - 273.15
    put(0, sampler.sample(grid("ps")) / 100, *sc["ps"])
    put(1, sampler.sample(grid("zs")), *sc["zs"])
    put(2, t2, *sc["t2"])
    put(3, np.maximum(t2 - d2, 0), *sc["dd"])
    u, v = sampler.wind(grid("u10"), grid("v10"))
    put(4, u, *sc["uv"])
    put(5, v, *sc["uv"])
    for li, p in enumerate(levels):
        b = 6 + 5 * li
        t = sampler.sample(grid(f"t{p}")) - 273.15
        dv = sampler.sample(grid(f"d{p}"))
        d = dew_from_rh(t, dv) if dew == "RH" else dv - 273.15
        put(b, sampler.sample(grid(f"z{p}")), info["zbase"][li], info["zstep"])
        put(b + 1, t, info["tbase"][li], info["tstep"])
        put(b + 2, np.maximum(t - d, 0), *sc["dd"])
        u, v = sampler.wind(grid(f"u{p}"), grid(f"v{p}"))
        put(b + 3, u, *sc["uv"])
        put(b + 4, v, *sc["uv"])
    return out.tobytes()


def render_soundings(model, run, out_dir, upload, pick=None):
    """Packs and uploads every hour; returns the manifest entry, or None if no hour worked."""
    spec = SND[model]
    hours = [h for h in spec["hours"] if h in pick] if pick else spec["hours"]
    sampler = Sampler(spec["grid"])
    lock = threading.Lock()
    done = []

    def one(h):
        data = snd_hour(model, run, h, sampler)
        key = f"{model}/{run}/snd/{h:03d}.bin"
        if out_dir:
            p = os.path.join(out_dir, key)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(data)
        if upload:
            send("PUT", f"/weather/models/upload/{key}", data, "application/octet-stream")
        with lock:
            done.append(h)
        return len(data)

    size = 0
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(one, h): h for h in hours}
        for fu in cf.as_completed(futs):
            try:
                size = fu.result()
            except Exception as e:  # noqa: BLE001  a missing hour just isn't offered
                print(f"  {model} sounding +{futs[fu]}h failed: {e!r}", flush=True)
    if not done:
        return None
    print(f"{model}: soundings for {len(done)} hours ({size / 1e6:.1f} MB each)", flush=True)
    return {**snd_info(model), "prefix": f"{model}/{run}/snd/", "hours": sorted(done)}


# ---- upload -------------------------------------------------------------------

def site():
    return os.environ.get("MODELS_SITE", "https://atmosdays.eagleeyeinsightsllc.com").rstrip("/")


def send(method, path, body, content_type):
    key = os.environ["MODELS_UPLOAD_KEY"]
    req = urllib.request.Request(f"{site()}{path}", data=body, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": content_type, "User-Agent": UA})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise
            err = e
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(2 ** i)
    raise err


def current_manifest():
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{site()}/weather/api/models", headers={"User-Agent": UA}), timeout=30) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return {"models": {}}


# ---- main ---------------------------------------------------------------------

MODELS = {
    "gfs": {"name": "GFS (NOAA)", "latest": gfs_latest, "fields": gfs_fields, "draw": draw, "area": "na",
            "hours": HOURS, "maps": FIELDS + ["dew", "cape", "qpf"]},
    "ecmwf": {"name": "European (ECMWF)", "latest": ec_latest, "fields": ec_fields, "draw": draw, "area": "na",
              "hours": HOURS, "maps": FIELDS + ["dew", "cape", "qpf", "snow"]},
    "hrrr": {"name": "HRRR (NOAA, 2 days)", "latest": short_latest(hrrr_url, 48), "fields": hrrr_fields, "draw": draw_hrrr,
             "area": "conus", "hours": list(range(0, 49)), "maps": ["radar", "temp", "rain1", "gust", "uh", "dew", "cape", "qpf", "snow", "hail", "ltng"],
             "points": True},
    "href": {"name": "HREF (NOAA blend, 2 days)", "latest": short_latest(href_url, 48, every=12), "fields": href_fields, "draw": draw_href,
             "area": "conus", "hours": list(range(1, 49)), "maps": ["radar", "temp", "rain1", "storms40", "rotation", "heavy"]},
}


def points_bytes(by_hour, hours):
    """Pack {hour: {field: array(ny, nx)}} as uint8 mph, laid out [cell][field][hour]."""
    ny, nx, nf, nh = POINTS["ny"], POINTS["nx"], len(POINTS["fields"]), len(hours)
    out = np.full((ny, nx, nf, nh), POINTS["missing"], dtype=np.uint8)
    for hi, h in enumerate(hours):
        p = by_hour.get(h)
        if not p:
            continue
        for fi, name in enumerate(POINTS["fields"]):
            v = p[name]
            q = np.clip(np.rint(np.nan_to_num(v, nan=-1)), -1, 254)
            out[:, :, fi, hi] = np.where(q < 0, POINTS["missing"], q).astype(np.uint8)
    return out.tobytes()


def render_model(model, run, hours, out_dir, upload, points=None):
    spec = MODELS[model]

    def one(h):
        f = spec["fields"](run, h)
        if points is not None and f.get("pts") is not None:
            points[h] = f["pts"]
        pics = spec["draw"](f, h)
        for field, data in pics.items():
            key = f"{model}/{run}/{field}/{h:03d}.webp"
            if out_dir:
                p = os.path.join(out_dir, key)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.write(data)
            if upload:
                send("PUT", f"/weather/models/upload/{key}", data, "image/webp")
        return h

    done = []
    # ECMWF precipitation needs the hour before, so its hours go in order per worker.
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for h in pool.map(one, hours):
            done.append(h)
            print(f"  {model} {run} +{h}h", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="also save pictures into this folder")
    ap.add_argument("--upload", action="store_true", help="upload to the website")
    ap.add_argument("--hours", help="comma list for a quick test, e.g. 0,6,12")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--force", action="store_true", help="draw even if this run is already up")
    ap.add_argument("--no-soundings", action="store_true", help="skip the click-anywhere sounding grids")
    a = ap.parse_args()
    if a.upload and len(os.environ.get("MODELS_UPLOAD_KEY", "")) < 24:
        print("MODELS_UPLOAD_KEY is not set; nothing to do.")
        return 0
    pick = [int(x) for x in a.hours.split(",")] if a.hours else None
    now = dt.datetime.now(dt.timezone.utc)
    current = current_manifest() if a.upload else {"models": {}}
    manifest = {"updated": now.isoformat(timespec="seconds"), "bbox": list(BBOX), "legends": LEGENDS,
                "names": FIELD_NAMES, "fieldLegend": FIELD_LEGEND,
                "credit": "NOAA NCEP GFS, HRRR and HREF; ECMWF open data, CC BY 4.0",
                "models": dict(current.get("models") or {})}
    order = {k: i for i, k in enumerate(MODELS)}
    failed = []
    for model in a.models.split(","):
        spec = MODELS[model]
        try:
            run = spec["latest"](now)
            if not run:
                print(f"{model}: no complete run found")
                continue
            old = manifest["models"].get(model) or {}
            if old.get("run") == run and not a.force:
                print(f"{model}: run {run} is already up")
                continue
            print(f"{model}: drawing run {run}", flush=True)
            t0 = time.time()
            hours = [h for h in spec["hours"] if h in pick] if pick else spec["hours"]
            points = {} if spec.get("points") else None
            done = render_model(model, run, hours, a.out, a.upload, points)
            print(f"{model}: {len(done)} hours in {time.time() - t0:.0f}s", flush=True)
            point_info = None
            if points:
                ph = spec["hours"]
                data = points_bytes(points, ph)
                key = f"{model}/{run}/wind/points.bin"
                if a.out:
                    p = os.path.join(a.out, key)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "wb") as fh:
                        fh.write(data)
                try:
                    if a.upload:
                        send("PUT", f"/weather/models/upload/{key}", data, "application/octet-stream")
                    point_info = {**POINTS, "key": key, "hours": ph}
                    print(f"{model}: wind at height for {len(points)} hours ({len(data) / 1e6:.1f} MB)", flush=True)
                except Exception as e:  # noqa: BLE001  the maps still go up without it
                    print(f"{model}: wind at height not saved: {e!r}", flush=True)
            snd = None
            if model in SND and not a.no_soundings:
                try:
                    t0 = time.time()
                    snd = render_soundings(model, run, a.out, a.upload, pick)
                    print(f"{model}: soundings took {time.time() - t0:.0f}s", flush=True)
                except Exception as e:  # noqa: BLE001  the maps still go up without them
                    print(f"{model}: soundings not saved: {e!r}", flush=True)
        except Exception as e:  # noqa: BLE001  one model failing shouldn't stop the others
            print(f"{model}: FAILED: {e!r}", flush=True)
            failed.append(model)
            continue
        keep = [r for r in [old.get("run")] if r and r != run][: KEEP_RUNS - 1]
        ar = AREAS[spec["area"]][0]
        manifest["models"][model] = {"name": spec["name"], "run": run, "hours": sorted(done), "fields": spec["maps"],
                                     "bbox": list(ar), "ext": "webp", "keep": keep}
        if point_info:
            manifest["models"][model]["points"] = point_info
        if snd:
            manifest["models"][model]["snd"] = snd
        manifest["models"] = dict(sorted(manifest["models"].items(), key=lambda kv: order.get(kv[0], 9)))
        # Publish after each model so a slow or failed model doesn't hold the others back.
        if a.upload:
            print(send("POST", "/weather/models/upload/manifest", json.dumps(manifest).encode(), "application/json"))
    if a.out:
        with open(os.path.join(a.out, "manifest.json"), "w") as f:
            json.dump(manifest, f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
