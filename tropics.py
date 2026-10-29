#!/usr/bin/env python3
"""Hurricane extras for https://atmosdays.eagleeyeinsightsllc.com/weather/hurricanes

For each active storm on the National Hurricane Center's list:
  * spaghetti model tracks and intensity forecasts (ATCF "a-deck", newest run of each model)
  * the storm's history every 6 hours: winds, pressure, wind radii (ATCF "b-deck")
And for the last hours, from the NHC reconnaissance archive:
  * Hurricane Hunter high-density observations (HDOBs, every 30 seconds of flight)
  * vortex data messages (the center fixes: position, lowest pressure, top winds)

All of it is free NOAA public data (ftp.nhc.noaa.gov and www.nhc.noaa.gov).
The result is one small JSON file, uploaded to the website (tropics/latest.json)
so the website itself never has to unpack the big a-deck files.

  python tropics.py --out out/tropics.json     # write locally
  python tropics.py --upload                    # write and upload (needs MODELS_UPLOAD_KEY)
  python tropics.py --test                      # print a summary, upload nothing
"""
import argparse
import datetime as dt
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

UA = "AtmosDays-Weather-Tropics/0.1 (contact@eagleeyeinsightsllc.com)"
NHC = "https://www.nhc.noaa.gov"
FTP = "https://ftp.nhc.noaa.gov/atcf"

# Recon archive folders: HDOBs and vortex messages, Atlantic and Pacific.
HDOB_DIRS = ["AHONT1", "AHOPN1"]
VORTEX_DIRS = ["REPNT2", "REPPN2"]
HDOB_HOURS = 10      # keep this many hours of flight observations
VORTEX_HOURS = 36    # and of center fixes
MODEL_WINDOW_H = 18  # a model run older than the newest run by more than this is left out

# Spaghetti groups: which models go where. Anything else lands in "other".
GROUPS = {
    "official": ["OFCL"],
    "global": ["AVNO", "EMX", "EMX2", "ECMF", "CMC", "NVGM", "UKM", "UKX", "EGRR", "JGSM", "GFSO"],
    "hurricane": ["HFSA", "HFSB", "HWRF", "HMON", "CTCX", "COTC", "HAFS"],
    "consensus": ["TVCN", "TVCA", "TVCE", "TVCX", "HCCA", "FSSE", "GFEX", "RVCN", "CONU"],
    "simple": ["BAMD", "BAMM", "BAMS", "LBAR", "XTRP", "CLP5", "TCLP"],
    "means": ["AEMN", "CEMN", "EEMN", "UEMN", "NEMN"],
}
INTENSITY = {"SHIP", "DSHP", "LGEM", "ICON", "IVCN", "IVDR", "OCD5", "SHF5", "DSF5", "RI25", "RI30", "RI35", "RI40"}
SKIP = {"CARQ", "WRNG", "BEST", "COMS"}
NAMES = {
    "OFCL": "NHC official forecast", "AVNO": "GFS (NOAA)", "AVNI": "GFS, adjusted", "EMX": "ECMWF (European)", "EMXI": "ECMWF, adjusted",
    "ECMF": "ECMWF (European)", "EMX2": "ECMWF, adjusted 12 h", "CMC": "Canadian", "CMCI": "Canadian, adjusted", "NVGM": "Navy NAVGEM",
    "NVGI": "Navy NAVGEM, adjusted", "UKM": "UK Met Office", "UKX": "UK Met Office", "UKMI": "UK Met Office, adjusted", "UKXI": "UK Met Office, adjusted",
    "EGRR": "UK Met Office", "EGRI": "UK Met Office, adjusted", "JGSM": "Japan JMA", "HFSA": "HAFS-A (NOAA hurricane model)",
    "HFAI": "HAFS-A, adjusted", "HFSB": "HAFS-B (NOAA hurricane model)", "HFBI": "HAFS-B, adjusted", "HWRF": "HWRF", "HWFI": "HWRF, adjusted",
    "HMON": "HMON", "HMNI": "HMON, adjusted", "CTCX": "Navy COAMPS-TC", "CTCI": "Navy COAMPS-TC, adjusted", "COTC": "Navy COAMPS-TC",
    "TVCN": "Consensus (TVCN)", "TVCA": "Consensus (TVCA)", "TVCE": "Consensus (TVCE)", "TVCX": "Consensus (TVCX)", "HCCA": "Corrected consensus (HCCA)",
    "FSSE": "Florida State super-ensemble", "GFEX": "GFS + ECMWF average", "RVCN": "Consensus (RVCN)", "CONU": "Consensus (CONU)",
    "BAMD": "Beta-advection, deep", "BAMM": "Beta-advection, medium", "BAMS": "Beta-advection, shallow", "LBAR": "LBAR (simple)",
    "XTRP": "Straight-line motion", "CLP5": "Climatology and persistence", "TCLP": "Climatology and persistence",
    "AEMN": "GFS ensemble average", "AEMI": "GFS ensemble average, adjusted", "CEMN": "Canadian ensemble average", "EEMN": "ECMWF ensemble average",
    "UEMN": "UK ensemble average", "NEMN": "Navy ensemble average", "SHIP": "SHIPS (statistical)", "DSHP": "SHIPS with land decay",
    "LGEM": "LGEM (statistical)", "ICON": "Intensity consensus (ICON)", "IVCN": "Intensity consensus (IVCN)", "OCD5": "Climatology and persistence",
}
MEMBER_PREFIX = {"AP": "GFS ensemble", "AC": "GFS ensemble", "CP": "Canadian ensemble", "CC": "Canadian ensemble", "EP": "ECMWF ensemble",
                 "EN": "ECMWF ensemble", "EE": "ECMWF ensemble", "NP": "Navy ensemble", "UE": "UK ensemble"}


def group_of(tech):
    for g, techs in GROUPS.items():
        if tech in techs:
            return g
    # "Interpolated" early versions: the late model shifted to start at the current position.
    if tech in ("AVNI", "EMXI", "CMCI", "NVGI", "UKMI", "UKXI", "EGRI", "HFAI", "HFBI", "HWFI", "HMNI", "CTCI", "AEMI", "COTI"):
        return "early"
    if re.fullmatch(r"[A-Z]{2}\d\d", tech) and tech[:2] in MEMBER_PREFIX:
        return "members"
    if tech in INTENSITY:
        return "intensity"
    return "other"


def name_of(tech):
    if tech in NAMES:
        return NAMES[tech]
    if re.fullmatch(r"[A-Z]{2}\d\d", tech) and tech[:2] in MEMBER_PREFIX:
        return f"{MEMBER_PREFIX[tech[:2]]} member {int(tech[2:])}"
    return tech


# ---- downloads ----------------------------------------------------------------

def get(url, tries=4):
    err = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
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


def get_text(url):
    b = get(url)
    if url.endswith(".gz") or b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    return b.decode("latin-1")


# ---- ATCF a-deck / b-deck ------------------------------------------------------

def ll(s):
    """'164N' -> 16.4, '749W' -> -74.9 (tenths of a degree)."""
    s = s.strip()
    if not s or not s[:-1].lstrip("-").isdigit():
        return None
    v = int(s[:-1]) / 10
    return -v if s[-1] in "SW" else v


def to_int(s):
    s = s.strip()
    return int(s) if s.lstrip("-").isdigit() else None


def parse_atcf(text):
    """Rows from an ATCF deck: one per (tech, run, hour), wind radii merged in."""
    rows = {}
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 11:
            continue
        tech, run, tau = f[4].upper(), f[2], to_int(f[5])
        lat, lon = ll(f[6]), ll(f[7])
        if tau is None or lat is None or lon is None or not re.fullmatch(r"\d{10}", run):
            continue
        key = (tech, run, tau)
        r = rows.get(key)
        if r is None:
            vmax, mslp = to_int(f[8]), to_int(f[9]) if len(f) > 9 else None
            r = rows[key] = {
                "tech": tech, "run": run, "tau": tau, "lat": lat, "lon": lon,
                "kt": vmax if vmax and vmax > 0 else None,
                "mb": mslp if mslp and 850 < mslp < 1050 else None,
                "type": f[10] if len(f) > 10 else "",
                "radii": {},
            }
            if len(f) > 19:
                rmw = to_int(f[19])
                if rmw and rmw > 0:
                    r["rmw"] = rmw
            if len(f) > 26 and f[25] and f[26]:
                d, s = to_int(f[25]), to_int(f[26])
                if d is not None and s is not None and (d or s):
                    r["dir"], r["spd"] = d, s
            if len(f) > 27 and f[27]:
                r["name"] = f[27]
        if len(f) > 16:
            rad, code = to_int(f[11]), f[12].upper()
            q = [to_int(x) for x in f[13:17]]
            if rad in (34, 50, 64) and code in ("NEQ", "AAA") and any(q):
                r["radii"][str(rad)] = [x or 0 for x in q] if code == "NEQ" else [q[0] or 0] * 4
    return list(rows.values())


def spaghetti(rows):
    """The newest run of every model, as compact tracks."""
    rows = [r for r in rows if r["tech"] not in SKIP and r["tau"] >= 0]
    if not rows:
        return None
    newest = max(r["run"] for r in rows)
    t_new = dt.datetime.strptime(newest, "%Y%m%d%H")
    latest = {}
    for r in rows:
        if r["run"] > latest.get(r["tech"], ""):
            latest[r["tech"]] = r["run"]
    tracks = []
    for tech, run in sorted(latest.items()):
        if (t_new - dt.datetime.strptime(run, "%Y%m%d%H")).total_seconds() > MODEL_WINDOW_H * 3600:
            continue
        pts = sorted((r for r in rows if r["tech"] == tech and r["run"] == run), key=lambda r: r["tau"])
        if not pts:
            continue
        g = group_of(tech)
        tracks.append({
            "id": tech, "name": name_of(tech), "group": g, "run": run,
            # [hour, lat, lon, wind kt, pressure mb]
            "pts": [[p["tau"], round(p["lat"], 1), round(p["lon"], 1), p["kt"], p["mb"]] for p in pts],
        })
    return {"newest": newest, "tracks": tracks}


def history(rows):
    """Best track every 6 hours (the storm's past), with wind radii."""
    best = sorted((r for r in rows if r["tech"] == "BEST"), key=lambda r: r["run"])
    out = []
    for r in best:
        h = {"t": r["run"], "lat": round(r["lat"], 1), "lon": round(r["lon"], 1), "kt": r["kt"], "mb": r["mb"], "type": r["type"]}
        if r.get("rmw"):
            h["rmw"] = r["rmw"]
        if r["radii"]:
            h["radii"] = r["radii"]
        if r.get("dir") is not None:
            h["dir"], h["spd"] = r["dir"], r["spd"]
        out.append(h)
    return out


# ---- recon (Hurricane Hunters) ------------------------------------------------

def stamp_of(name):
    m = re.search(r"\.(\d{12})\.txt$", name)
    return dt.datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc) if m else None


def recent_files(folder, year, hours, now):
    url = f"{NHC}/archive/recon/{year}/{folder}/"
    try:
        html = get(url).decode("latin-1")
    except Exception as e:  # noqa: BLE001
        print(f"  {folder}: no listing ({e})", file=sys.stderr)
        return []
    names = sorted(set(re.findall(rf'href="({folder}-[A-Z]{{4}}\.\d{{12}}\.txt)"', html)))
    keep = [n for n in names if (s := stamp_of(n)) and now - s <= dt.timedelta(hours=hours)]
    return [(n, url + n) for n in keep]


def lat_ddmm(s):
    """'2828N' -> 28.47 (degrees and minutes)."""
    m = re.fullmatch(r"(\d{2})(\d{2})([NS])", s)
    if not m:
        return None
    v = int(m.group(1)) + int(m.group(2)) / 60
    return round(-v if m.group(3) == "S" else v, 3)


def lon_dddmm(s):
    m = re.fullmatch(r"(\d{3})(\d{2})([EW])", s)
    if not m:
        return None
    v = int(m.group(1)) + int(m.group(2)) / 60
    return round(-v if m.group(3) == "W" else v, 3)


def num(s, scale=1.0, signed=False):
    if "/" in s or not s:
        return None
    try:
        v = int(s)
    except ValueError:
        return None
    return v if scale == 1.0 else v * scale


def parse_hdob(text, file_time=None):
    """One HDOB message: header (aircraft, storm, ob number) and its 30-second observations."""
    lines = [l.rstrip() for l in text.splitlines()]
    head = None
    for i, l in enumerate(lines):
        m = re.match(r"^\s*([A-Z]{2,4}\d{1,3})\s+(\w{4,6})\s+(.*?)\s+HDOB\s+(\d+)\s+(\d{8})\s*$", l)
        if m:
            head = (i, m)
            break
    if not head:
        return None
    i0, m = head
    aircraft, mission, storm, ob, date = m.groups()
    day = dt.datetime.strptime(date, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    pts = []
    last_t = None
    for l in lines[i0 + 1:]:
        f = l.split()
        if len(f) < 13 or not re.fullmatch(r"\d{6}", f[0]):
            if l.strip().startswith("$$"):
                break
            continue
        hh, mi, ss = int(f[0][:2]), int(f[0][2:4]), int(f[0][4:])
        if hh > 23 or mi > 59 or ss > 59:
            continue
        t = day + dt.timedelta(hours=hh, minutes=mi, seconds=ss)
        if last_t and t < last_t - dt.timedelta(hours=12):  # past midnight
            t += dt.timedelta(days=1)
        last_t = t
        lat, lon = lat_ddmm(f[1]), lon_dddmm(f[2])
        if lat is None or lon is None:
            continue
        fl_mb = num(f[3], 0.1)
        if fl_mb is not None and fl_mb < 100:  # 4-digit tenths with the leading 1 left off
            fl_mb += 1000
        height_m = num(f[4])
        sfc = num(f[5])
        if sfc is not None:
            # Extrapolated surface pressure in tenths of mb, leading 1 left off (0010 = 1001.0).
            # Above 550 mb the aircraft sends a D-value instead; leave that out.
            sfc = round(sfc / 10 + (1000 if sfc < 5000 else 0), 1) if (fl_mb or 0) > 550 else None
            if sfc is not None and not 850 <= sfc <= 1050:
                sfc = None
        temp = num(f[6], 0.1)
        dew = num(f[7], 0.1)
        wd = f[8]
        wdir = num(wd[:3]) if len(wd) == 6 else None
        wspd = num(wd[3:]) if len(wd) == 6 else None
        peak = num(f[9])
        sfmr = num(f[10])
        rain = num(f[11])
        pts.append([
            t.strftime("%H%M%S"), lat, lon,
            round(fl_mb, 1) if fl_mb is not None else None, height_m, sfc,
            round(temp, 1) if temp is not None else None, round(dew, 1) if dew is not None else None,
            wdir, wspd, peak, sfmr, rain,
        ])
    if not pts:
        return None
    first = day.replace(hour=int(pts[0][0][:2]), minute=int(pts[0][0][2:4]), second=int(pts[0][0][4:]))
    if file_time and first - file_time > dt.timedelta(hours=12):
        first -= dt.timedelta(days=1)
    return {
        "aircraft": aircraft, "mission": mission, "storm": storm.strip(), "ob": int(ob),
        "time": first.strftime("%Y-%m-%dT%H:%M:%SZ"),
        # [hhmmss, lat, lon, flight-level mb, height m, surface mb, temp C, dew point C, wind dir, wind kt, peak 10 s wind kt, SFMR surface wind kt, rain mm/h]
        "pts": pts,
    }


def parse_vortex(text, file_time):
    """A vortex data message (center fix). Item letters follow the 2018+ format."""
    t = " ".join(text.split("\n"))
    out = {"raw": "\n".join(l.rstrip() for l in text.strip().splitlines() if l.strip())[:1600]}
    m = re.search(r"VORTEX DATA MESSAGE\s+([A-Z]{2}\d{6})", t)
    out["storm_id"] = m.group(1) if m else None
    m = re.search(r"\bA\.\s*(\d{2})/(\d{2}):(\d{2}):(\d{2})Z", t)
    if m and file_time:
        d, hh, mi, ss = (int(x) for x in m.groups())
        when = None
        try:
            when = file_time.replace(day=d, hour=hh, minute=mi, second=ss)
            if when > file_time + dt.timedelta(hours=1):  # fix from the previous month
                when = (file_time.replace(day=1) - dt.timedelta(days=1)).replace(day=d, hour=hh, minute=mi, second=ss)
        except ValueError:
            when = file_time
        out["time"] = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        out["time"] = file_time.strftime("%Y-%m-%dT%H:%M:%SZ") if file_time else None
    m = re.search(r"\bB\.\s*([\d.]+)\s*deg\s*([NS])\s+([\d.]+)\s*deg\s*([EW])", t)
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(3))
    out["lat"] = -lat if m.group(2) == "S" else lat
    out["lon"] = -lon if m.group(4) == "W" else lon
    m = re.search(r"\bC\.\s*(\d{3,4})\s*mb\s+(\d+)\s*m", t)
    if m:
        out["level_mb"], out["height_m"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"\bD\.\s*(\d{3,4})\s*mb", t)
    out["mb"] = int(m.group(1)) if m else None
    m = re.search(r"\bE\.\s*(\d{3})\s*deg\s*(\d+)\s*kt", t)
    if m:
        out["center_wind"] = [int(m.group(1)), int(m.group(2))]
    m = re.search(r"\bF\.\s*(.*?)\s+G\.", t)
    if m and m.group(1).strip() not in ("NA", ""):
        out["eye"] = m.group(1).strip()[:40]
    m = re.search(r"\bG\.\s*(.*?)\s+H\.", t)
    if m and m.group(1).strip() not in ("NA", ""):
        out["eye_shape"] = m.group(1).strip()[:40]
    m = re.search(r"\bH\.\s*(\d+)\s*kt", t)
    out["max_sfc_kt"] = int(m.group(1)) if m else None
    m = re.search(r"\bJ\.\s*(\d{3})\s*deg\s*(\d+)\s*kt", t)
    if m:
        out["max_fl_in_kt"] = int(m.group(2))
    m = re.search(r"\bN\.\s*(\d{3})\s*deg\s*(\d+)\s*kt", t)
    if m:
        out["max_fl_out_kt"] = int(m.group(2))
    m = re.search(r"MAX FL WIND\s+(\d+)\s*KT", t)
    if m:
        out["max_fl_kt"] = int(m.group(1))
    m = re.search(r"\b[A-Z]\.\s*([A-Z]{2,4}\d{1,3})\s+(\w{4,6})\s+(.+?)\s+OB\s+(\d+)", t)
    if m:
        out["aircraft"], out["mission"], out["storm"] = m.group(1), m.group(2), m.group(3).strip()
    return out


def recon(now):
    year = now.year
    hdobs, fixes = [], []
    for folder in HDOB_DIRS:
        for name, url in recent_files(folder, year, HDOB_HOURS, now):
            try:
                h = parse_hdob(get_text(url), stamp_of(name))
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: {e}", file=sys.stderr)
                continue
            if h:
                h["basin"] = "Atlantic" if folder.endswith("NT1") else "Pacific"
                hdobs.append(h)
    for folder in VORTEX_DIRS:
        for name, url in recent_files(folder, year, VORTEX_HOURS, now):
            try:
                v = parse_vortex(get_text(url), stamp_of(name))
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: {e}", file=sys.stderr)
                continue
            if v:
                fixes.append(v)
    hdobs.sort(key=lambda h: h["time"])
    fixes.sort(key=lambda v: v.get("time") or "")
    # Group the 10-minute messages into flights.
    flights = {}
    for h in hdobs:
        k = f'{h["aircraft"]} {h["mission"]}'
        f = flights.setdefault(k, {"aircraft": h["aircraft"], "mission": h["mission"], "storm": h["storm"], "basin": h["basin"], "obs": []})
        f["obs"].append({"ob": h["ob"], "time": h["time"], "pts": h["pts"]})
    return {"flights": list(flights.values()), "fixes": fixes}


# ---- storms -----------------------------------------------------------------

def active_storms():
    j = json.loads(get(f"{NHC}/CurrentStorms.json"))
    return [str(s.get("id", "")).lower() for s in j.get("activeStorms", []) if s.get("id")]


def storm_extras(sid):
    """sid like 'al132025'."""
    out = {"id": sid.upper()}
    try:
        b = parse_atcf(get_text(f"{FTP}/btk/b{sid}.dat"))
        out["history"] = history(b)
    except Exception as e:  # noqa: BLE001
        print(f"  {sid} b-deck: {e}", file=sys.stderr)
        out["history"] = []
    try:
        a = parse_atcf(get_text(f"{FTP}/aid_public/a{sid}.dat.gz"))
        out["models"] = spaghetti(a)
    except Exception as e:  # noqa: BLE001
        print(f"  {sid} a-deck: {e}", file=sys.stderr)
        out["models"] = None
    return out


def build(now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    ids = active_storms()
    storms = [storm_extras(s) for s in ids]
    return {"updated": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "storms": storms, "recon": recon(now)}


def summary(doc):
    print(f"updated {doc['updated']}: {len(doc['storms'])} storm(s)")
    for s in doc["storms"]:
        m = s.get("models") or {}
        groups = {}
        for t in m.get("tracks", []):
            groups.setdefault(t["group"], []).append(t["id"])
        print(f"  {s['id']}: {len(s['history'])} best-track points, newest model run {m.get('newest')}")
        for g, ids in sorted(groups.items()):
            print(f"    {g}: {' '.join(ids)}")
        if s["history"]:
            print(f"    latest: {s['history'][-1]}")
    r = doc["recon"]
    print(f"  recon: {len(r['flights'])} flight(s), {len(r['fixes'])} center fix(es)")
    for f in r["flights"]:
        n = sum(len(o["pts"]) for o in f["obs"])
        print(f"    {f['aircraft']} {f['mission']} {f['storm']}: {len(f['obs'])} messages, {n} points, last {f['obs'][-1]['pts'][-1]}")
    for v in r["fixes"][-3:]:
        print(f"    fix {v.get('time')} {v.get('storm_id')} {v.get('lat')},{v.get('lon')} {v.get('mb')} mb sfc {v.get('max_sfc_kt')} kt fl {v.get('max_fl_kt')} kt eye {v.get('eye')}")
    print(f"  JSON size {len(json.dumps(doc, separators=(',', ':'))) / 1024:.0f} KB")


def upload(doc):
    key = os.environ["MODELS_UPLOAD_KEY"]
    site = os.environ.get("MODELS_SITE", "https://atmosdays.eagleeyeinsightsllc.com").rstrip("/")
    body = json.dumps(doc, separators=(",", ":")).encode()
    req = urllib.request.Request(f"{site}/weather/models/upload/tropics/latest.json", data=body, method="PUT", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": UA})
    err = None
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
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
    ap.add_argument("--out")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    doc = build()
    summary(doc)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
    if a.upload and not a.test:
        if not os.environ.get("MODELS_UPLOAD_KEY"):
            print("MODELS_UPLOAD_KEY is not set; nothing uploaded")
            return
        upload(doc)


if __name__ == "__main__":
    main()
