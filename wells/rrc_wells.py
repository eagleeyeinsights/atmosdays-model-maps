"""Texas well index for AtmosDays Well Sites, from the Railroad Commission of Texas.

Downloads two free public RRC data sets for every county and joins them by
API number:

  * Well Layers by County (well<FIPS>.zip): the surface location of every well,
    from the well<FIPS>s.dbf table (API, LAT83, LONG83, SYMNUM, RELIAB).
  * Statewide API Data, dBase (api<FIPS>.dbf): operator, lease name, well
    number and plug date.

and writes one small JSON file per county, an operator index and a manifest,
uploaded to the website through /weather/models/upload/wells/tx/... (the same
upload key as the model maps).

Only the fields the website needs are kept (API number, location, operator,
lease, well number, status). Nothing else from the RRC files is stored.
Source: Railroad Commission of Texas (https://www.rrc.texas.gov/). Not an
official record.

Usage:
  python wells/rrc_wells.py --out out/wells            # build files, upload nothing
  python wells/rrc_wells.py --upload                   # build and upload
  python wells/rrc_wells.py --out out/wells --counties 389,301   # a few counties (test)

Standard library only. The RRC file site is a GoAnywhere "GoDrive" web client:
the file list is an HTML table and each file downloads with a form POST that
names its row (fileTable:<row>:<id>). Rows past the first 250 come from the
table's page-2 AJAX request.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import html
import http.cookiejar
import io
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

MFT = "https://mft.rrc.texas.gov"
WELL_LAYERS = f"{MFT}/link/d551fb20-442e-4b67-84fa-ac3f23ecabb4"  # Well Layers by County (shapefiles)
API_DBF = f"{MFT}/link/1eb94d66-461d-4114-93f7-b4bc04a70674"  # Statewide API Data (dBase)
# A dropped connection (the RRC site sometimes cuts the secure handshake for minutes
# at a time) is waited out with growing pauses, about 8 minutes in all, before a
# request counts as failed. Requests are also spaced out so the job stays polite.
RETRY_WAITS = (10, 30, 60, 120, 240)
MIN_GAP = 0.5  # seconds between requests to one RRC folder


class SourceUnreachable(Exception):
    """The RRC file site could not be reached at all (not a problem in this code)."""


UA = "AtmosDays-WellIndex/1.0 (+https://atmosdays.eagleeyeinsightsllc.com/about)"

# Texas county FIPS codes are assigned in alphabetical order: 001, 003, ... 507.
COUNTIES = [
    "Anderson", "Andrews", "Angelina", "Aransas", "Archer", "Armstrong", "Atascosa", "Austin", "Bailey", "Bandera",
    "Bastrop", "Baylor", "Bee", "Bell", "Bexar", "Blanco", "Borden", "Bosque", "Bowie", "Brazoria",
    "Brazos", "Brewster", "Briscoe", "Brooks", "Brown", "Burleson", "Burnet", "Caldwell", "Calhoun", "Callahan",
    "Cameron", "Camp", "Carson", "Cass", "Castro", "Chambers", "Cherokee", "Childress", "Clay", "Cochran",
    "Coke", "Coleman", "Collin", "Collingsworth", "Colorado", "Comal", "Comanche", "Concho", "Cooke", "Coryell",
    "Cottle", "Crane", "Crockett", "Crosby", "Culberson", "Dallam", "Dallas", "Dawson", "Deaf Smith", "Delta",
    "Denton", "DeWitt", "Dickens", "Dimmit", "Donley", "Duval", "Eastland", "Ector", "Edwards", "Ellis",
    "El Paso", "Erath", "Falls", "Fannin", "Fayette", "Fisher", "Floyd", "Foard", "Fort Bend", "Franklin",
    "Freestone", "Frio", "Gaines", "Galveston", "Garza", "Gillespie", "Glasscock", "Goliad", "Gonzales", "Gray",
    "Grayson", "Gregg", "Grimes", "Guadalupe", "Hale", "Hall", "Hamilton", "Hansford", "Hardeman", "Hardin",
    "Harris", "Harrison", "Hartley", "Haskell", "Hays", "Hemphill", "Henderson", "Hidalgo", "Hill", "Hockley",
    "Hood", "Hopkins", "Houston", "Howard", "Hudspeth", "Hunt", "Hutchinson", "Irion", "Jack", "Jackson",
    "Jasper", "Jeff Davis", "Jefferson", "Jim Hogg", "Jim Wells", "Johnson", "Jones", "Karnes", "Kaufman", "Kendall",
    "Kenedy", "Kent", "Kerr", "Kimble", "King", "Kinney", "Kleberg", "Knox", "Lamar", "Lamb",
    "Lampasas", "La Salle", "Lavaca", "Lee", "Leon", "Liberty", "Limestone", "Lipscomb", "Live Oak", "Llano",
    "Loving", "Lubbock", "Lynn", "McCulloch", "McLennan", "McMullen", "Madison", "Marion", "Martin", "Mason",
    "Matagorda", "Maverick", "Medina", "Menard", "Midland", "Milam", "Mills", "Mitchell", "Montague", "Montgomery",
    "Moore", "Morris", "Motley", "Nacogdoches", "Navarro", "Newton", "Nolan", "Nueces", "Ochiltree", "Oldham",
    "Orange", "Palo Pinto", "Panola", "Parker", "Parmer", "Pecos", "Polk", "Potter", "Presidio", "Rains",
    "Randall", "Reagan", "Real", "Red River", "Reeves", "Refugio", "Roberts", "Robertson", "Rockwall", "Runnels",
    "Rusk", "Sabine", "San Augustine", "San Jacinto", "San Patricio", "San Saba", "Schleicher", "Scurry", "Shackelford", "Shelby",
    "Sherman", "Smith", "Somervell", "Starr", "Stephens", "Sterling", "Stonewall", "Sutton", "Swisher", "Tarrant",
    "Taylor", "Terrell", "Terry", "Throckmorton", "Titus", "Tom Green", "Travis", "Trinity", "Tyler", "Upshur",
    "Upton", "Uvalde", "Val Verde", "Van Zandt", "Victoria", "Walker", "Waller", "Ward", "Washington", "Webb",
    "Wharton", "Wheeler", "Wichita", "Wilbarger", "Willacy", "Williamson", "Wilson", "Winkler", "Wise", "Wood",
    "Yoakum", "Young", "Zapata", "Zavala",
]
assert len(COUNTIES) == 254
FIPS = {f"{2 * i + 1:03d}": name for i, name in enumerate(COUNTIES)}

# RRC well symbol numbers (Digital Map Information user guide, OGA094, Appendix A),
# folded into the few statuses the website shows.
PLUGGED = {7, 8, 10, 116, 117, 118, 119, *range(136, 140), *range(152, 156)}
STATUS = {2: "permit", 3: "dry", 9: "canceled", 19: "shut-in", 20: "shut-in", 86: "horizontal", 87: "directional"}
for s in (4, 5, 6):
    STATUS[s] = "producing"
for s in (11, 21, 22, 23, *range(104, 108), *range(124, 128), *range(144, 148)):
    STATUS[s] = "injection"


def status_of(sym: int | None, plug_date: str) -> str:
    """One short status word for a well: producing, horizontal, plugged, permit, injection..."""
    if sym in PLUGGED or (plug_date and plug_date.strip("0")):
        return "plugged"
    return STATUS.get(sym or 0, "other")


# ---- dBase ---------------------------------------------------------------------

def read_dbf(data: bytes):
    """Rows of a dBase III table as dicts of stripped strings (deleted rows skipped)."""
    if len(data) < 32:
        return
    nrec, hlen, rlen = struct.unpack_from("<IHH", data, 4)
    fields = []
    o = 32
    while o < hlen - 1 and data[o] != 0x0D:
        name = data[o:o + 11].split(b"\0", 1)[0].decode("ascii", "replace")
        fields.append((name, data[o + 16]))
        o += 32
    for r in range(nrec):
        start = hlen + r * rlen
        rec = data[start:start + rlen]
        if len(rec) < rlen:
            break
        if rec[:1] == b"*":
            continue
        row, p = {}, 1
        for name, size in fields:
            row[name] = rec[p:p + size].decode("latin-1").strip()
            p += size
        yield row


def num(s: str):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # NaN


# ---- the RRC GoDrive file site ------------------------------------------------------

class GoDrive:
    """One shared-link folder on mft.rrc.texas.gov: list its files and download them.

    The site is a JSF/PrimeFaces page: a download is a form post that "clicks" a row
    of the file table, and the server only honours a click on a row of the page the
    table is showing right now. So a download first pages the table to that row.
    One folder is one server-side view, so its requests take turns (a lock)."""

    def __init__(self, link: str, page_rows: int = 250):
        self.link = link
        self.page_rows = page_rows
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.action = f"{MFT}/webclient/godrive/PublicGoDrive.xhtml"
        self.fields: dict[str, str] = {}
        self.files: dict[str, dict] = {}
        self.shown: set[int] = set()  # table rows on the page the server is showing
        self.lock = threading.Lock()
        self.last_request = 0.0

    def _open(self, url, data=None, headers=None, timeout=180):
        req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
        last = None
        for wait in (*RETRY_WAITS, None):
            gap = self.last_request + MIN_GAP - time.monotonic()
            if gap > 0:
                time.sleep(gap)
            self.last_request = time.monotonic()
            try:
                with self.opener.open(req, timeout=timeout) as r:
                    return r.read(), dict(r.headers)
            except urllib.error.HTTPError:
                raise  # the server answered; retrying the same stale request won't help
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = e
                if wait is None:
                    break
                print(f"  RRC connection problem ({e}); trying again in {wait} s", flush=True)
                time.sleep(wait)
        raise last

    def _rows(self, text: str):
        """File rows in a page or AJAX update: name, size text, modified text, row id."""
        shown = set()
        for tr in re.split(r"<tr\b", text)[1:]:
            m = re.search(r'id="(fileTable:(\d+):(j_id_[\w]+))"[^>]*>\s*(?:<[^>]+>\s*)*([^<\s][^<]*?)\s*<', tr)
            if not m:
                continue
            cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<td\b[^>]*>(.*?)</td>", tr, re.S)]
            name = html.unescape(m.group(4)).strip()
            modified = next((c for c in cells if re.match(r"\d{1,2}/\d{1,2}/\d{2}", c)), "")
            self.files[name] = {"key": m.group(1), "row": int(m.group(2)), "modified": modified}
            shown.add(int(m.group(2)))
        return shown

    def _form_fields(self, text: str):
        f = re.search(r'<form[^>]+id="fileList".*?</form>', text, re.S)
        body = f.group(0) if f else text
        for m in re.finditer(r"<input\b([^>]*)>", body):
            attrs = dict(re.findall(r'(\w[\w:.-]*)="([^"]*)"', m.group(1)))
            name, typ = attrs.get("name"), attrs.get("type", "text").lower()
            if not name or (typ in ("checkbox", "radio") and "checked" not in m.group(1)):
                continue
            self.fields[name] = html.unescape(attrs.get("value", ""))
        for m in re.finditer(r'<select\b[^>]*name="([^"]+)"[^>]*>(.*?)</select>', body, re.S):
            sel = re.search(r'<option[^>]*value="([^"]*)"[^>]*selected', m.group(2)) or re.search(r'<option[^>]*value="([^"]*)"', m.group(2))
            if sel:
                self.fields[m.group(1)] = sel.group(1)
        vs = re.search(r'name="javax\.faces\.ViewState"[^>]*value="([^"]+)"', body) or re.search(r'name="javax\.faces\.ViewState"[^>]*value="([^"]+)"', text)
        if vs:
            self.fields["javax.faces.ViewState"] = html.unescape(vs.group(1))
        self.fields.setdefault("fileList_SUBMIT", "1")

    def _load(self):
        """Open the folder page (a fresh view, showing the first page of the table)."""
        body, _ = self._open(self.link)
        text = body.decode("utf-8", "replace")
        self.fields = {}
        self._form_fields(text)
        self.shown = self._rows(text)
        return text

    def _page(self, first: int):
        """Show the table page that starts at row `first` (the AJAX call the page's pager makes)."""
        data = urllib.parse.urlencode({
            "javax.faces.partial.ajax": "true", "javax.faces.source": "fileTable",
            "javax.faces.partial.execute": "fileTable", "javax.faces.partial.render": "fileTable",
            "fileTable": "fileTable", "fileTable_pagination": "true", "fileTable_first": str(first),
            "fileTable_rows": str(self.page_rows), "fileTable_encodeFeature": "true", "fileList": "fileList",
            "javax.faces.ViewState": self.fields.get("javax.faces.ViewState", ""),
        }).encode()
        part, _ = self._open(self.action, data, {"Content-Type": "application/x-www-form-urlencoded", "Faces-Request": "partial/ajax", "X-Requested-With": "XMLHttpRequest"})
        ptext = html.unescape(part.decode("utf-8", "replace"))
        vs = re.search(r'ViewState[^>]*><!\[CDATA\[([^\]]+)\]\]>', ptext)
        if vs:
            self.fields["javax.faces.ViewState"] = vs.group(1)
        self.shown = self._rows(ptext)
        return self.shown

    def list(self, page_rows=None):
        if page_rows:
            self.page_rows = page_rows
        with self.lock:
            text = self._load()
            total = re.search(r"Showing\s+\d+\s*-\s*\d+\s+of\s+([\d,]+)", text)
            total = int(total.group(1).replace(",", "")) if total else len(self.files)
            # Ask for the next rows after the last one shown, however many the page held.
            first = max((f["row"] for f in self.files.values()), default=-1) + 1
            while len(self.files) < total and first < total:
                before = len(self.files)
                self._page(first)
                if len(self.files) == before:
                    break
                first = max(f["row"] for f in self.files.values()) + 1
        return self.files

    def get(self, name: str) -> bytes:
        row = self.files[name]["row"]
        last = None
        for attempt in range(3):
            with self.lock:
                try:
                    if attempt:
                        self._load()  # a fresh view after a failed click
                    if row not in self.shown:
                        self._page(row)  # the page that starts at this row, whatever size the server keeps
                    f = self.files[name]
                    data = urllib.parse.urlencode({**self.fields, f["key"]: f["key"]}).encode()
                    body, headers = self._open(self.action, data, {"Content-Type": "application/x-www-form-urlencoded"}, timeout=600)
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                    last = e  # includes HTTPError; start over with a fresh view
                    continue
                ctype = headers.get("Content-Type", headers.get("content-type", ""))
                if "html" in ctype and not body.startswith(b"PK"):
                    last = RuntimeError(f"{name}: got a web page instead of the file")
                    continue
                return body
        raise last


# ---- building the index -----------------------------------------------------------------

def surface_points(zip_bytes: bytes, fips: str):
    """API (8 digits) -> (lat, lon, symbol, reliability) from the county's surface well layer.

    A well with more than one surface point (re-spotted, sidetracks) keeps the most
    reliable one (highest RELIAB code)."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        name = next((n for n in z.namelist() if n.lower() == f"well{fips}s.dbf"), None)
        if not name:
            return out
        for r in read_dbf(z.read(name)):
            api = r.get("API", "")
            lat, lon = num(r.get("LAT83")), num(r.get("LONG83"))
            if not re.fullmatch(r"\d{8}", api) or lat is None or lon is None:
                continue
            if not (25.5 < lat < 36.6 and -106.7 < lon < -93.4):
                continue  # outside Texas: a bad spot
            sym = int(num(r.get("SYMNUM")) or 0)
            rel = int(num(r.get("RELIAB")) or 0)
            old = out.get(api)
            if old is None or rel > old[3]:
                out[api] = (round(lat, 5), round(lon, 5), sym, rel)
    return out


def county_file(fips: str, points: dict, api_rows, updated: str):
    """One county's JSON: operators as a lookup list, one short row per well.

    Kept on purpose to the minimum: API number, surface location, status,
    operator, lease name and well number (plus the RRC's location reliability code)."""
    ops, op_ix = [], {}

    def op(v):
        if not v:
            return -1
        if v not in op_ix:
            op_ix[v] = len(ops)
            ops.append(v)
        return op_ix[v]

    seen, rows = set(), []
    for r in api_rows:
        api = r.get("APINUM", "")
        if not re.fullmatch(r"\d{8}", api) or api in seen:
            continue
        operator = r.get("OPERATOR", "")
        p = points.get(api)
        if not operator and not p:
            continue
        seen.add(api)
        lat, lon, sym, rel = p if p else (None, None, None, None)
        rows.append([api[3:], lat, lon, status_of(sym, r.get("PLUG_DATE", "")), rel, op(operator), r.get("LEASE_NAME", ""), r.get("WELLID", "").lstrip("0")])
    # Points with no API record (rare): keep the location so a lookup still finds them.
    for api, (lat, lon, sym, rel) in points.items():
        if api not in seen:
            rows.append([api[3:], lat, lon, status_of(sym, ""), rel, -1, "", ""])
    rows.sort(key=lambda w: w[0])
    located = [w for w in rows if w[1] is not None and w[3] != "plugged"] or [w for w in rows if w[1] is not None]
    center = [round(sum(w[1] for w in located) / len(located), 4), round(sum(w[2] for w in located) / len(located), 4)] if located else None
    return {
        "v": 1, "state": "TX", "fips": fips, "county": FIPS[fips], "updated": updated, "center": center,
        "cols": ["id5", "lat", "lon", "status", "reliab", "op", "lease", "wellNo"],
        "ops": ops, "wells": rows,
    }


def build(counties, out_dir=None, upload=False, workers=4):
    updated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    wl, ad = GoDrive(WELL_LAYERS), GoDrive(API_DBF)
    try:
        wl.list()
        ad.list()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise SourceUnreachable(f"the RRC file site could not be reached: {e}") from e
    print(f"RRC lists: {len(wl.files)} well layer files, {len(ad.files)} API data files", flush=True)
    operators: dict[str, dict[str, int]] = {}
    leases: dict[tuple, int] = {}  # (lease, fips, operator) -> active wells, for the name search
    summary = {}

    def one(fips):
        zname, dname = f"well{fips}.zip", f"api{fips}.dbf"
        points = surface_points(wl.get(zname), fips) if zname in wl.files else {}
        api_rows = list(read_dbf(ad.get(dname))) if dname in ad.files else []
        data = county_file(fips, points, api_rows, updated)
        body = json.dumps(data, separators=(",", ":")).encode()
        put(f"wells/tx/{fips}.json", body, out_dir, upload)
        return fips, data, len(body)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for fu in cf.as_completed([pool.submit(one, f) for f in counties]):
            try:
                fips, data, size = fu.result()
            except Exception as e:  # noqa: BLE001  one county failing is reported, not fatal
                print(f"  county failed: {e!r}", flush=True)
                continue
            n_loc = sum(1 for w in data["wells"] if w[1] is not None)
            summary[fips] = {"county": data["county"], "wells": len(data["wells"]), "located": n_loc, "center": data["center"], "bytes": size}
            for w in data["wells"]:
                if w[5] >= 0 and w[3] != "plugged":
                    by = operators.setdefault(data["ops"][w[5]], {})
                    by[fips] = by.get(fips, 0) + 1
                if w[6] and w[3] not in ("plugged", "canceled", "dry"):
                    k = (w[6], fips, data["ops"][w[5]] if w[5] >= 0 else "")
                    leases[k] = leases.get(k, 0) + 1
            print(f"  {fips} {data['county']}: {len(data['wells'])} wells, {n_loc} with a surface spot, {size / 1e6:.1f} MB", flush=True)
    return updated, summary, operators, leases


def put(key, body, out_dir, upload):
    if out_dir:
        p = os.path.join(out_dir, key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(body)
    if upload:
        send("PUT", f"/weather/models/upload/{key}", body)


def site():
    return os.environ.get("MODELS_SITE", "https://atmosdays.eagleeyeinsightsllc.com").rstrip("/")


def send(method, path, body):
    req = urllib.request.Request(f"{site()}{path}", data=body, method=method, headers={
        "Authorization": f"Bearer {os.environ['MODELS_UPLOAD_KEY']}", "Content-Type": "application/json", "User-Agent": UA})
    err = None
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise
            err = e
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(5 * (i + 1))
    raise err


def previous_total():
    """Wells in the index the website has now (0 if none yet)."""
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{site()}/weather/api/wells/status", headers={"User-Agent": UA}), timeout=30) as r:
            return int(json.loads(r.read()).get("wells") or 0)
    except Exception:  # noqa: BLE001
        return 0


STALE_DAYS = 4


def site_index_age_days():
    """Days since the website's well index was built, or None if it has none (or can't be read)."""
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{site()}/weather/api/wells/status", headers={"User-Agent": UA}), timeout=30) as r:
            updated = json.loads(r.read()).get("updated")
        if not updated:
            return None
        when = dt.datetime.strptime(updated, "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc)
        return (dt.datetime.now(dt.timezone.utc) - when).total_seconds() / 86400
    except Exception:  # noqa: BLE001
        return None


def source_down(why: str, upload: bool) -> int:
    """The RRC site can't be reached. The website keeps the index it has. The run fails
    (red) only when the website's index is over STALE_DAYS old; otherwise it's a warning."""
    age = site_index_age_days() if upload else None
    if age is None or age <= STALE_DAYS:
        have = "has no well index yet" if age is None else f"keeps its index from {age:.1f} days ago"
        print(f"::warning::Skipped this run: {why}. The website {have}; the next run tries again.")
        return 0
    print(f"::error::{why}. The website's index is {age:.0f} days old; the next run tries again.")
    return 1


PROBE_GAP = 3


def reachable(tries: int = 3) -> bool:
    """Can this computer reach the RRC file site? The site turns some GitHub computers
    away every time (it hangs up right after the secure greeting) and lets others in,
    so a computer that's turned away doesn't wait: another computer tries instead."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(WELL_LAYERS, headers={"User-Agent": UA}), timeout=20):
                return True
        except urllib.error.HTTPError:
            return True  # it answered
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"  RRC file site turned this computer away ({e})", flush=True)
            if i + 1 < tries:
                time.sleep(PROBE_GAP)
    return False


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("--out", help="also write the files to this folder")
    a.add_argument("--upload", action="store_true", help="upload to the website (needs MODELS_UPLOAD_KEY)")
    a.add_argument("--counties", default="", help="comma list of county FIPS codes (default: all 254)")
    a.add_argument("--skip-if-fresh", type=float, default=0, metavar="HOURS",
                   help="with --upload: do nothing if the website's index is newer than this")
    a.add_argument("--computer", default="", metavar="N/OF",
                   help="this is GitHub computer N of OF: if the RRC site turns it away, stop at once so the next one tries")
    args = a.parse_args(argv)
    if args.upload and len(os.environ.get("MODELS_UPLOAD_KEY", "")) < 24:
        print("MODELS_UPLOAD_KEY is not set; nothing to do.")
        return 0
    if args.upload and args.skip_if_fresh:
        age = site_index_age_days()
        if age is not None and age * 24 < args.skip_if_fresh:
            print(f"The website's well index is {age * 24:.1f} hours old; nothing to do this time.")
            return 0
    if args.computer:
        n, of = (int(x) for x in args.computer.split("/"))
        if not reachable():
            if n < of:
                print(f"::notice::The RRC file site turned away GitHub computer {n} of {of}; computer {n + 1} tries next.")
                return 0
            return source_down(f"the RRC file site turned away all {of} GitHub computers", args.upload)
        print(f"GitHub computer {n} of {of} reached the RRC file site.", flush=True)
    counties = [c.strip().zfill(3) for c in args.counties.split(",") if c.strip()] or list(FIPS)
    bad = [c for c in counties if c not in FIPS]
    if bad:
        raise SystemExit(f"not Texas county FIPS codes: {bad}")
    full = len(counties) == len(FIPS)
    # Counties are uploaded as they finish; the manifest and operator list go last,
    # and only after the checks pass, so a broken RRC download never replaces a good index.
    try:
        updated, summary, operators, leases = build(counties, args.out, False if not full else args.upload)
    except SourceUnreachable as e:
        return source_down(str(e), args.upload)
    total = sum(s["wells"] for s in summary.values())
    print(f"{len(summary)} counties, {total} wells", flush=True)
    if full:
        if len(summary) < len(FIPS):
            missing = sorted(set(FIPS) - set(summary))
            raise SystemExit(f"only {len(summary)} of {len(FIPS)} counties built (missing {missing[:10]}); keeping the old index")
        before = previous_total() if args.upload else 0
        if before and abs(total - before) > 0.1 * before:
            raise SystemExit(f"well count moved from {before} to {total} (over 10%); keeping the old index")
    op_list = {"v": 1, "updated": updated, "operators": sorted(([name, sorted(by.items())] for name, by in operators.items()), key=lambda x: x[0])}
    op_names = sorted({k[2] for k in leases})
    op_at = {n: i for i, n in enumerate(op_names)}
    lease_list = {"v": 1, "updated": updated, "cols": ["lease", "fips", "op", "wells"], "ops": op_names,
                  "leases": sorted([lease, fips, op_at[op], n] for (lease, fips, op), n in leases.items())}
    manifest = {"v": 1, "state": "TX", "updated": updated, "wells": total, "counties": summary,
                "source": "Railroad Commission of Texas: Well Layers by County and Statewide API Data (free public downloads). Not an official record."}
    up = args.upload and full
    put("wells/tx/operators.json", json.dumps(op_list, separators=(",", ":")).encode(), args.out, up)
    put("wells/tx/leases.json", json.dumps(lease_list, separators=(",", ":")).encode(), args.out, up)
    put("wells/tx/manifest.json", json.dumps(manifest, separators=(",", ":")).encode(), args.out, up)
    if args.upload and not full:
        print("A partial run uploads nothing (use --out to look at the files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
