"""Training rows for the AtmosDays place-learning forecast (machine learning).

Each row pairs what a forecast model said at an airport weather station with
what that station measured at the same hour:

  - old forecasts: NOAA's GEFS reforecast (today's GFS-family model re-run for
    every day 2000-2019, control member) and the GFS archive from 2021 on, both
    free on AWS open data. Only the few fields and hours needed are read
    (byte ranges from the .idx listings).
  - measured: NOAA's Integrated Surface Database (hourly airport weather), also
    free on AWS open data.

Only public stations are used: no customer place is ever read or written here.

A "chunk" is one source and one year (every 5th day, 00Z run, 6-168 h ahead,
every 6 h), saved as a small .npz file. See ml/README.md.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import datetime as dt
import io
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from render_models import decode, get, idx_spans, ranges_get  # noqa: E402

REFORECAST = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
ISD = "https://noaa-global-hourly-pds.s3.amazonaws.com"
# NOAA retired ISD in late 2025 (its last files are from October 2025) and replaced it
# with GHCN-hourly, which is updated daily. Years from GHCNH_FROM on are read from it.
GHCNH = "https://noaa-ghcnh-pds.s3.amazonaws.com/hourly/access/by-year"
GHCNH_FROM = 2025
# isd-history.csv stopped updating with ISD, so "still reporting" means into 2025.
STILL_REPORTING = "20250101"
ISD_HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

LEADS = list(range(6, 169, 6))
EVERY_DAYS = 5
# Feature columns (the website uses the same order; see FEATURES in the model file).
FEATURES = ["src", "lead", "solar_hour", "doy_sin", "doy_cos", "lat", "lon", "zs_m", "t_f", "td_f", "ws_mph"]
TARGETS = ["t_f", "td_f", "ws_mph"]  # measured; the model learns measured minus forecast
SOURCES = {"rf": 0, "gfs": 1}  # GEFS reforecast (2000-2019), GFS archive (2021 on)


def c_to_f(c):
    return c * 9 / 5 + 32


# ---- stations ----------------------------------------------------------------

def load_stations(path, today=None):
    """US airport stations with a long record, thinned to about one per 1-degree box."""
    if os.path.exists(path):
        with open(path) as f:
            got = json.load(f)
        if len(got) >= 100:  # an empty or tiny list is a failed build: make it again
            return got
    rows = csv.DictReader(io.StringIO(get(ISD_HISTORY).decode("latin-1")))
    best = {}
    for r in rows:
        try:
            lat, lon, elev = float(r["LAT"]), float(r["LON"]), float(r["ELEV(M)"])
        except (TypeError, ValueError):
            continue
        if r["CTRY"] != "US" or not r["ICAO"].startswith("K") or r["WBAN"] == "99999":
            continue
        if not (24 <= lat <= 50 and -125 <= lon <= -66) or r["BEGIN"] > "19991231" or r["END"] < STILL_REPORTING:
            continue
        box = (math.floor(lat), math.floor(lon))
        s = {"id": r["USAF"] + r["WBAN"], "icao": r["ICAO"], "name": r["STATION NAME"].title(), "lat": lat, "lon": lon, "elev": elev, "begin": r["BEGIN"]}
        if box not in best or s["begin"] < best[box]["begin"]:
            best[box] = s
    out = sorted(best.values(), key=lambda s: s["id"])
    if len(out) < 100:
        raise RuntimeError(f"only {len(out)} stations in isd-history.csv; not saving")
    with open(path, "w") as f:
        json.dump(out, f)
    return out


# ---- forecasts ---------------------------------------------------------------

def sample(grid, lats, lons):
    """Bilinear values of a global lat/lon grid at the stations."""
    v = grid.values
    nj, ni = v.shape
    rows = (lats - grid.lat0) / grid.dlat
    cols = ((lons - grid.lon0) % 360) / grid.dlon
    r0, c0 = np.floor(rows).astype(int), np.floor(cols).astype(int)
    fr, fc = rows - r0, cols - c0
    r0 = np.clip(r0, 0, nj - 2)
    c1 = (c0 + 1) % ni
    c0 = c0 % ni
    top = v[r0, c0] * (1 - fc) + v[r0, c1] * fc
    bot = v[r0 + 1, c0] * (1 - fc) + v[r0 + 1, c1] * fc
    return top * (1 - fr) + bot * fr


def dew_from_q(q, p):
    """Dew point (C) from specific humidity (kg/kg) and pressure (Pa)."""
    e = np.maximum(q * p / (0.622 + 0.378 * q), 1e-3)
    a = np.log(e / 611.2)
    return 243.5 * a / (17.67 - a)


def lead_of(fcst):
    """'6 hour fcst' -> 6; anything else (averages, sums) -> None."""
    parts = fcst.split()
    if len(parts) == 3 and parts[1] == "hour" and parts[2] == "fcst" and parts[0].isdigit():
        return int(parts[0])
    return None


def reforecast_day(day, lats, lons):
    """{lead: (t_f, td_f, ws_mph, zs_m)} arrays at the stations for one 00Z reforecast run."""
    ymd = day.strftime("%Y%m%d")
    base = f"{REFORECAST}/{day.year}/{ymd}00/c00/Days:1-10"
    files = {"t": ("tmp_2m", "TMP", "2 m above ground"), "q": ("spfh_2m", "SPFH", "2 m above ground"),
             "p": ("pres_sfc", "PRES", "surface"), "u": ("ugrd_hgt", "UGRD", "10 m above ground"),
             "v": ("vgrd_hgt", "VGRD", "10 m above ground")}

    def one(key):
        name, var, level = files[key]
        url = f"{base}/{name}_{ymd}00_c00.grib2"
        want = {h: (lambda h: lambda v, l, f, r: v == var and l == level and lead_of(f) == h)(h) for h in LEADS}
        spans = idx_spans(get(url + ".idx").decode(), want)
        raw = ranges_get(url, dict(spans))
        return key, {h: sample(decode(raw[h]), lats, lons) for h in raw}

    def terrain():
        url = f"{base}/hgt_sfc_{ymd}00_c00.grib2"
        spans = idx_spans(get(url + ".idx").decode(), {"zs": lambda v, l, f, r: v == "HGT" and l == "surface"})
        return "zs", sample(decode(ranges_get(url, dict(spans))["zs"]), lats, lons)

    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(one, k) for k in files] + [pool.submit(terrain)]
        got = dict(f.result() for f in futs)
    zs = got.pop("zs")
    out = {}
    for h in LEADS:
        if not all(h in got[k] for k in files):
            continue
        t = got["t"][h] - 273.15
        td = np.minimum(dew_from_q(got["q"][h], got["p"][h]), t)
        out[h] = (c_to_f(t), c_to_f(td), np.hypot(got["u"][h], got["v"][h]) * 2.23694, zs)
    return out


def gfs_day(day, lats, lons):
    """Same shape as reforecast_day, from the operational GFS archive (0.25 degree)."""
    ymd = day.strftime("%Y%m%d")
    want = {"t": ("TMP", "2 m above ground"), "td": ("DPT", "2 m above ground"),
            "u": ("UGRD", "10 m above ground"), "v": ("VGRD", "10 m above ground"), "zs": ("HGT", "surface")}

    def one(h):
        # Runs before GFS v16 (March 2021) sit one folder up, without "atmos/".
        url = f"{GFS}/gfs.{ymd}/00/atmos/gfs.t00z.pgrb2.0p25.f{h:03d}"
        try:
            idx = get(url + ".idx")
        except Exception:  # noqa: BLE001
            url = f"{GFS}/gfs.{ymd}/00/gfs.t00z.pgrb2.0p25.f{h:03d}"
            idx = get(url + ".idx")
        tests = {k: (lambda var, lvl: lambda v, l, f, r: v == var and l == lvl)(*vl) for k, vl in want.items()}
        spans = idx_spans(idx.decode(), tests)
        raw = ranges_get(url, dict(spans))
        g = {k: sample(decode(raw[k]), lats, lons) for k in raw}
        t, td = g["t"] - 273.15, g["td"] - 273.15
        return h, (c_to_f(t), c_to_f(np.minimum(td, t)), np.hypot(g["u"], g["v"]) * 2.23694, g.get("zs"))

    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        return dict(pool.map(one, LEADS))


# ---- measured ----------------------------------------------------------------

BAD_Q = set("2367")


def isd_hours(text):
    """{unix hour: (t_f, td_f, ws_mph)} at 00/06/12/18Z from one ISD station-year CSV."""
    out = {}
    best = {}
    for r in csv.DictReader(io.StringIO(text)):
        try:
            when = dt.datetime.fromisoformat(r["DATE"]).replace(tzinfo=dt.timezone.utc)
        except (KeyError, ValueError):
            continue
        hour = round(when.timestamp() / 3600)
        if hour % 6:
            continue
        off = abs(when.timestamp() - hour * 3600)
        if off > 1800 or (hour in best and best[hour] <= off):
            continue

        def val(field, i_val, i_q, scale):
            parts = (r.get(field) or "").split(",")
            if len(parts) <= max(i_val, i_q) or parts[i_q] in BAD_Q:
                return None
            raw = parts[i_val].lstrip("+")
            if not raw.lstrip("-").isdigit() or int(raw) in (9999, -9999):
                return None
            return int(raw) / scale

        t, td = val("TMP", 0, 1, 10), val("DEW", 0, 1, 10)
        ws = val("WND", 3, 4, 10)
        if t is None:
            continue
        best[hour] = off
        out[hour] = (c_to_f(t), None if td is None else c_to_f(td), None if ws is None else ws * 2.23694)
    return out


def ghcnh_hours(text):
    """{unix hour: (t_f, td_f, ws_mph)} at 00/06/12/18Z from one GHCN-hourly station-year PSV."""
    out = {}
    best = {}

    def num(r, field):
        q = (r.get(f"{field}_Quality_Code") or "").strip()
        if q in BAD_Q or q.isalpha():  # letters are GHCNh's failed-check flags
            return None
        try:
            return float(r.get(field) or "")
        except ValueError:
            return None

    for r in csv.DictReader(io.StringIO(text), delimiter="|"):
        try:
            when = dt.datetime.fromisoformat(r["DATE"]).replace(tzinfo=dt.timezone.utc)
        except (KeyError, ValueError):
            continue
        hour = round(when.timestamp() / 3600)
        if hour % 6:
            continue
        off = abs(when.timestamp() - hour * 3600)
        if off > 1800 or (hour in best and best[hour] <= off):
            continue
        t = num(r, "temperature")
        if t is None:
            continue
        td, ws = num(r, "dew_point_temperature"), num(r, "wind_speed")
        best[hour] = off
        out[hour] = (c_to_f(t), None if td is None else c_to_f(td), None if ws is None else ws * 2.23694)
    return out


def ghcnh_id(station):
    """GHCN-hourly id of an airport station: USW000 + its 5-digit WBAN."""
    return "USW000" + station["id"][-5:]


def measured_year(stations, year):
    """{station index: {unix hour: values}} for a year (one file per station)."""
    def one(i):
        s = stations[i]
        try:
            if year >= GHCNH_FROM:
                return i, ghcnh_hours(get(f"{GHCNH}/{year}/psv/GHCNh_{ghcnh_id(s)}_{year}.psv").decode("utf-8", "replace"))
            return i, isd_hours(get(f"{ISD}/{year}/{s['id']}.csv").decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001  a station missing a year is fine
            return i, {}
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        return dict(pool.map(one, range(len(stations))))


# ---- one chunk -----------------------------------------------------------------

def chunk_days(src, year, today):
    start = dt.date(year, 1, 1) + dt.timedelta(days=year % EVERY_DAYS)
    last = min(dt.date(year, 12, 31), today - dt.timedelta(days=10))  # station data posts late
    if src == "rf":
        last = min(last, dt.date(2019, 12, 31))
    days = []
    d = start
    while d <= last:
        days.append(d)
        d += dt.timedelta(days=EVERY_DAYS)
    return days


def build_chunk(src, year, stations, today=None, log=print, days=None):
    """X (rows x FEATURES), Y (rows x TARGETS, NaN when not measured), meta (station, init hour)."""
    today = today or dt.date.today()
    lats = np.array([s["lat"] for s in stations])
    lons = np.array([s["lon"] for s in stations])
    days = days if days is not None else chunk_days(src, year, today)
    obs = measured_year(stations, year)
    if year < today.year:  # a run late in December looks into January
        nxt = measured_year(stations, year + 1)
        for i in obs:
            obs[i].update(nxt.get(i, {}))
    log(f"{src}-{year}: {len(days)} runs, measured at {sum(1 for v in obs.values() if v)} of {len(stations)} stations")
    fetch = reforecast_day if src == "rf" else gfs_day
    X, Y, M = [], [], []
    zs_known = None
    for n, day in enumerate(days):
        try:
            fc = fetch(day, lats, lons)
        except Exception as e:  # noqa: BLE001  a missing day is skipped
            log(f"  {day}: skipped ({e!r})")
            continue
        init = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp() // 3600)
        for h, (t, td, ws, zs) in fc.items():
            if zs is not None:
                zs_known = zs
            zcol = zs_known if zs_known is not None else np.full(len(stations), np.nan)
            vh = init + h
            when = dt.datetime.fromtimestamp(vh * 3600, dt.timezone.utc)
            doy = when.timetuple().tm_yday
            for i in range(len(stations)):
                m = obs.get(i, {}).get(vh)
                if m is None or not np.isfinite(t[i]):
                    continue
                solar = (when.hour + stations[i]["lon"] / 15) % 24
                X.append([SOURCES[src], h, solar, math.sin(2 * math.pi * doy / 365.25), math.cos(2 * math.pi * doy / 365.25),
                          stations[i]["lat"], stations[i]["lon"], zcol[i], t[i], td[i], ws[i]])
                Y.append([m[0], np.nan if m[1] is None else m[1], np.nan if m[2] is None else m[2]])
                M.append([i, vh])
        if n % 10 == 0:
            log(f"  {day}: {len(X)} rows so far")
    return (np.array(X, dtype=np.float32).reshape(-1, len(FEATURES)), np.array(Y, dtype=np.float32).reshape(-1, len(TARGETS)),
            np.array(M, dtype=np.int64).reshape(-1, 2))


def save_chunk(path, X, Y, M):
    np.savez_compressed(path, X=X, Y=Y, M=M, features=np.array(FEATURES), targets=np.array(TARGETS),
                        built=np.array(dt.date.today().isoformat()))
