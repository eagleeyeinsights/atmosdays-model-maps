"""Live scoreboard for the AtmosDays place-learning forecast.

Every day, before the weather happens, this saves three forecasts at the same
public airport stations the model trains on (never customer places):

  - "gfs":  the raw GFS forecast (00Z run, the same model the correction starts from)
  - "ai":   that GFS forecast corrected by the AtmosDays model trained this run
  - "aifs": ECMWF's AI forecast (AIFS, free open data, CC BY 4.0), latest complete run

ECMWF's open-data service only keeps its latest runs, so saving AIFS daily is
also the only way to build up an archive of it for later training.

Once the stations have measured those hours, the saved forecasts are scored:
average miss (°F, mph) by days ahead, on rows where all three have a value, so
it's the same hours and places for each. Weekly numbers show how steady the
difference is. All forecasts were available by the time this runs (07:17 UTC).
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import glob
import json
import math
import os

import numpy as np

from ml.data import FEATURES, LEADS, SOURCES, c_to_f, gfs_day, measured_year, sample
from render_models import decode, ec_get, ec_url, ranges_get

# One saved row: station, valid hour (unix hours), then per source: lead hours, °F, dew °F, mph.
COLS = ["station", "valid", "gfs_lead", "gfs_t", "gfs_td", "gfs_ws", "ai_t", "ai_td", "ai_ws",
        "aifs_lead", "aifs_t", "aifs_td", "aifs_ws"]
C = {k: i for i, k in enumerate(COLS)}
SRCS = ["gfs", "ai", "aifs"]
ECMWF_CREDIT = "Contains ECMWF AIFS data, © ECMWF, CC BY 4.0 (changed: read at stations and scored)."
AIFS_WANT = {"2t": "t", "2d": "td", "10u": "u", "10v": "v"}
OBS_FRESH_DAYS = 2


# ---- forecasts -----------------------------------------------------------------

def aifs_run(day):
    """ECMWF AI run to use: today's 00Z if all of it is out, else yesterday's 12Z."""
    for run in (day.strftime("%Y%m%d") + "00", (day - dt.timedelta(days=1)).strftime("%Y%m%d") + "12"):
        try:
            ec_get(ec_url(run, LEADS[-1] + 12, "aifs-single") + ".index")
            return run
        except Exception:  # noqa: BLE001  not out (or gone): try the older run
            continue
    return None


def aifs_day(run, lats, lons, leads):
    """{lead h: (°F, dew °F, mph)} at the stations from one ECMWF AI run."""
    def one(h):
        path = ec_url(run, h, "aifs-single")
        idx = [json.loads(x) for x in ec_get(path + ".index").decode().splitlines() if x.strip()]
        spans = {}
        for r in idx:
            k = AIFS_WANT.get(r["param"])
            if k and k not in spans:
                spans[k] = (r["_offset"], r["_offset"] + r["_length"] - 1)
        raw = ranges_get(path + ".grib2", spans, fetch=ec_get)
        g = {k: sample(decode(v), lats, lons) for k, v in raw.items()}
        t, td = g["t"] - 273.15, g["td"] - 273.15
        return h, (c_to_f(t), c_to_f(np.minimum(td, t)), np.hypot(g["u"], g["v"]) * 2.23694)

    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        return dict(pool.map(one, leads))


def walk(trees, X):
    """The exported trees (the same arrays the website walks), all rows at once."""
    y = np.zeros(len(X))
    rows = np.arange(len(X))
    for tr in trees:
        f, t, n, left, right, v = (np.asarray(tr[k]) for k in ("f", "t", "n", "l", "r", "v"))
        i = np.zeros(len(X), dtype=np.int64)
        while True:
            inner = f[i] >= 0
            if not inner.any():
                break
            x = X[rows, np.where(inner, f[i], 0)]
            go_left = np.where(np.isnan(x), n[i] == 1, x <= t[i])
            i = np.where(inner, np.where(go_left, left[i], right[i]), i)
        y += v[i]
    return y


def corrected(model, X):
    """Forecast + the model's learned miss; dew point never above temperature, wind not below 0."""
    F = {k: j for j, k in enumerate(FEATURES)}
    out = {}
    for name, col in (("t_f", "t_f"), ("td_f", "td_f"), ("ws_mph", "ws_mph")):
        out[name] = X[:, F[col]] + walk(model["targets"][name]["trees"], X)
    out["td_f"] = np.minimum(out["td_f"], out["t_f"])
    out["ws_mph"] = np.maximum(out["ws_mph"], 0)
    return out


def save_day(day, stations, model, log=print):
    """Rows (len(stations) x len(LEADS)) of today's three forecasts."""
    lats = np.array([s["lat"] for s in stations])
    lons = np.array([s["lon"] for s in stations])
    gfs = gfs_day(day, lats, lons)
    init = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp() // 3600)
    run = aifs_run(day)
    aifs, off = {}, 0
    if run:
        off = 0 if run.endswith("00") and run[:8] == day.strftime("%Y%m%d") else 12
        try:
            aifs = aifs_day(run, lats, lons, [h + off for h in LEADS])
        except Exception as e:  # noqa: BLE001  keep the GFS side; AIFS is missing today
            log(f"AIFS {run}: skipped ({e!r})")
    log(f"live {day}: GFS 00Z, AIFS {run or 'none'}")
    rows, X = [], []
    zs = next((v[3] for v in gfs.values() if v[3] is not None), np.full(len(stations), np.nan))
    for h in LEADS:
        if h not in gfs:
            continue
        t, td, ws, _ = gfs[h]
        vh = init + h
        when = dt.datetime.fromtimestamp(vh * 3600, dt.timezone.utc)
        doy = when.timetuple().tm_yday
        a = aifs.get(h + off)
        for i, s in enumerate(stations):
            if not np.isfinite(t[i]):
                continue
            X.append([SOURCES["gfs"], h, (when.hour + s["lon"] / 15) % 24, math.sin(2 * math.pi * doy / 365.25),
                      math.cos(2 * math.pi * doy / 365.25), s["lat"], s["lon"], zs[i], t[i], td[i], ws[i]])
            rows.append([i, vh, h, t[i], td[i], ws[i], np.nan, np.nan, np.nan,
                         h + off if a else np.nan, a[0][i] if a else np.nan, a[1][i] if a else np.nan, a[2][i] if a else np.nan])
    L = np.array(rows, dtype=np.float64).reshape(-1, len(COLS))
    if len(L) and model:
        ai = corrected(model, np.array(X, dtype=np.float64))
        L[:, C["ai_t"]], L[:, C["ai_td"]], L[:, C["ai_ws"]] = ai["t_f"], ai["td_f"], ai["ws_mph"]
    return L


def month_path(data, day):
    return os.path.join(data, f"live-{day:%Y-%m}.npz")


def add_day(data, day, stations, model, log=print):
    """Save today's forecasts into this month's file (once a day). Returns the file, or None if already saved."""
    path = month_path(data, day)
    L, days = np.zeros((0, len(COLS))), []
    if os.path.exists(path):
        z = np.load(path)
        L, days = z["L"], [str(d) for d in z["days"]]
    if day.isoformat() in days:
        log(f"live {day}: already saved")
        return None
    new = save_day(day, stations, model, log)
    np.savez_compressed(path, L=np.concatenate([L, new]), days=np.array(days + [day.isoformat()]), cols=np.array(COLS))
    log(f"live {day}: {len(new):,} rows saved to {os.path.basename(path)}")
    return path


# ---- measured and scored ---------------------------------------------------------

def obs_path(data, year):
    return os.path.join(data, f"obs-{year}.npz")


def load_obs(data, year, stations, today, log=print):
    """{(station, unix hour): (°F, dew °F, mph)} for a year, cached as obs-YEAR.npz. Returns (obs, path if rebuilt)."""
    path = obs_path(data, year)
    if os.path.exists(path):
        z = np.load(path)
        built = dt.date.fromisoformat(str(z["built"]))
        if year < today.year and built.year > year or (today - built).days < OBS_FRESH_DAYS:
            return {(int(r[0]), int(r[1])): tuple(r[2:]) for r in z["O"]}, None
    got = measured_year(stations, year)
    O = [[i, h, v[0], np.nan if v[1] is None else v[1], np.nan if v[2] is None else v[2]]
         for i, hours in got.items() for h, v in hours.items()]
    O = np.array(O, dtype=np.float64).reshape(-1, 5)
    np.savez_compressed(path, O=O, built=np.array(today.isoformat()))
    log(f"obs-{year}: {len(O):,} measured hours from {sum(1 for v in got.values() if v)} stations")
    return {(int(r[0]), int(r[1])): tuple(r[2:]) for r in O}, path


def mae(a):
    return round(float(np.mean(np.abs(a))), 2) if len(a) else None


def score(L, obs):
    """Average misses by days ahead and by week, same rows for all three forecasts."""
    keep, meas = [], []
    for r in L:
        m = obs.get((int(r[C["station"]]), int(r[C["valid"]])))
        if m is not None:
            keep.append(r)
            meas.append(m)
    if not keep:
        return {"rows": 0, "byDay": [], "weekly": []}
    L, Mx = np.array(keep), np.array(meas)
    have = np.isfinite(L[:, C["ai_t"]]) & np.isfinite(L[:, C["aifs_t"]])
    L, Mx = L[have], Mx[have]
    day = np.ceil(L[:, C["gfs_lead"]] / 24).astype(int)
    out = {"rows": int(len(L)), "byDay": [], "weekly": []}
    for d in range(1, 8):
        sel = day == d
        if not sel.any():
            continue
        row = {"day": d, "n": int(sel.sum())}
        for var, j in (("temp", 0), ("dew", 1), ("wind", 2)):
            ok = sel & np.isfinite(Mx[:, j])
            row[var] = {s: mae(L[ok, C[f"{s}_{['t', 'td', 'ws'][j]}"]] - Mx[ok, j]) for s in SRCS}
        out["byDay"].append(row)
    # Weekly, next-day temperature (day 1), so the week-to-week spread is visible.
    week = (L[:, C["valid"]] // (24 * 7)).astype(int)
    for w in sorted(set(week[day == 1].tolist()))[-12:]:
        sel = (week == w) & (day == 1)
        start = dt.datetime.fromtimestamp(w * 24 * 7 * 3600, dt.timezone.utc).date().isoformat()
        out["weekly"].append({"week": start, "n": int(sel.sum()),
                              "temp": {s: mae(L[sel, C[f"{s}_t"]] - Mx[sel, 0]) for s in SRCS}})
    return out


def scoreboard(data, stations, today, log=print):
    """Score everything saved so far. Returns (scoreboard dict, obs files rebuilt)."""
    files = sorted(glob.glob(os.path.join(data, "live-*.npz")))
    if not files:
        return {"rows": 0, "byDay": [], "weekly": []}, []
    L = np.concatenate([np.load(f)["L"] for f in files])
    years = sorted({dt.datetime.fromtimestamp(int(v) * 3600, dt.timezone.utc).year for v in (L[:, C["valid"]].min(), L[:, C["valid"]].max())})
    obs, rebuilt = {}, []
    for y in range(years[0], years[-1] + 1):
        o, p = load_obs(data, y, stations, today, log)
        obs.update(o)
        if p:
            rebuilt.append(p)
    sb = score(L, obs)
    days = [str(d) for f in files for d in np.load(f)["days"]]
    sb.update({"updated": today.isoformat(), "since": min(days), "daysSaved": len(days), "stations": len(stations),
               "sources": {"gfs": "NOAA GFS, raw", "ai": "AtmosDays AI-corrected GFS", "aifs": "ECMWF AIFS (AI model), raw"},
               # CC BY 4.0: anything that shows the AIFS numbers must show this line.
               "credit": ECMWF_CREDIT})
    return sb, rebuilt
