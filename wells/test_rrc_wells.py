"""Tests for the Texas well index (python -m unittest wells/test_rrc_wells.py).

A tiny local stand-in for the RRC GoDrive file site serves fake county files in
the same layout as the real ones (checked against mft.rrc.texas.gov on
2026-11-21: the row markup, the page-2 AJAX request and the dBase fields)."""
import http.server
import io
import json
import os
import re
import struct
import sys
import tempfile
import threading
import unittest
import urllib.parse
import zipfile

sys.path.insert(0, os.path.dirname(__file__))
import rrc_wells as W  # noqa: E402


def dbf(fields, rows):
    """A dBase III file: fields = [(name, size)], rows = [dict]."""
    hlen = 32 + 32 * len(fields) + 1
    rlen = 1 + sum(s for _, s in fields)
    out = bytearray(struct.pack("<BBBBIHH20x", 3, 126, 11, 21, len(rows), hlen, rlen))
    for name, size in fields:
        out += name.encode().ljust(11, b"\0") + b"C" + b"\0" * 4 + bytes([size, 0]) + b"\0" * 14
    out += b"\x0d"
    for r in rows:
        out += b" " + b"".join(str(r.get(n, "")).encode("latin-1")[:s].ljust(s) for n, s in fields)
    return bytes(out + b"\x1a")


SURF = [("SURFACE_ID", 10), ("SYMNUM", 5), ("API", 8), ("RELIAB", 2), ("LONG27", 19), ("LAT27", 19), ("LONG83", 19), ("LAT83", 19), ("WELLID", 5)]
APIF = [("ABSTRACT", 6), ("APINUM", 8), ("BLOCK", 10), ("COMPLETION", 8), ("FIELD_NAME", 32), ("LEASE_NAME", 32), ("GAS_RRCID", 6), ("OIL_GAS_CO", 1),
        ("ON_OFF_SCH", 1), ("OPERATOR", 32), ("PERMIT_NUM", 6), ("PLUG_DATE", 8), ("REFER_TO_A", 8), ("SECTION", 8), ("SURVEY", 55), ("TOTAL_DEPT", 5), ("WELLID", 6), ("QUADNUM", 7)]


def county_zip(fips, rows):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"well{fips}s.dbf", dbf(SURF, rows))
        z.writestr(f"well{fips}b.dbf", dbf(SURF, []))
        z.writestr(f"well{fips}s.prj", 'GEOGCS["GCS_North_American_1927"]')
    return b.getvalue()


REEVES_SURF = [
    {"SYMNUM": 86, "API": "38936724", "RELIAB": 55, "LONG83": "-103.31167364", "LAT83": "31.09742981"},
    {"SYMNUM": 86, "API": "38936724", "RELIAB": 15, "LONG83": "-103.4", "LAT83": "31.2"},  # older, less reliable spot
    {"SYMNUM": 7, "API": "38900001", "RELIAB": 40, "LONG83": "-103.5", "LAT83": "31.3"},
    {"SYMNUM": 2, "API": "38999999", "RELIAB": 55, "LONG83": "-103.6", "LAT83": "31.4"},
    {"SYMNUM": 4, "API": "38900002", "RELIAB": 55, "LONG83": "0", "LAT83": "0"},  # bad spot: dropped
]
REEVES_API = [
    {"APINUM": "38936724", "OPERATOR": "PERMIAN ROCK OPERATING LLC", "LEASE_NAME": "RAYBURN 12", "WELLID": "0001H", "FIELD_NAME": "PHANTOM (WOLFCAMP)", "PERMIT_NUM": "719342", "PLUG_DATE": "0"},
    {"APINUM": "38900001", "OPERATOR": "OLD WELLS INC", "LEASE_NAME": "SMITH", "WELLID": "1", "FIELD_NAME": "BUSH (PERMIAN)", "PERMIT_NUM": "000000", "PLUG_DATE": "19870102"},
    {"APINUM": "38900003", "OPERATOR": "PERMIAN ROCK OPERATING LLC", "LEASE_NAME": "RAYBURN 12", "WELLID": "2H", "FIELD_NAME": "PHANTOM (WOLFCAMP)", "PERMIT_NUM": "720001"},
    {"APINUM": "38900004", "OPERATOR": "", "LEASE_NAME": "", "REFER_TO_A": "38934042"},  # refer-to record with nothing else: dropped
]
FILES = {
    "well301.zip": county_zip("301", [{"SYMNUM": 4, "API": "30100010", "RELIAB": 55, "LONG83": "-103.7", "LAT83": "31.8"}]),
    "well389.zip": county_zip("389", REEVES_SURF),
    "api301.dbf": dbf(APIF, [{"APINUM": "30100010", "OPERATOR": "PERMIAN ROCK OPERATING LLC", "LEASE_NAME": "LOVING A", "WELLID": "1", "FIELD_NAME": "WOLFBONE"}]),
    "api389.dbf": dbf(APIF, REEVES_API),
}


def page(names, first):
    rows = "".join(
        f'<tr data-ri="{first + i}" class="GoDriveItem FileItem"><td class="rowCheckbox"><input type="checkbox" name="fileTable_checkbox"></td>'
        f'<td class="NameColumn"><a id="fileTable:{first + i}:j_id_2f" href="#" class="ui-commandlink" onclick="PrimeFaces.addSubmitParam(\'fileList\',{{\'fileTable:{first + i}:j_id_2f\':\'fileTable:{first + i}:j_id_2f\'}}).submit(\'fileList\');return false;">{n}</a></td>'
        f'<td class="ModifiedOnColumn">11/20/26 5:40:01 PM</td><td class="SizeColumn">2.08 MB</td></tr>'
        for i, n in enumerate(names))
    return rows


class Fake(http.server.BaseHTTPRequestHandler):
    per_page = 1  # tiny pages so the AJAX paging is exercised
    pages: dict = {}  # session -> first row of the table page it shows; like the real site, only those rows can be clicked
    sessions = iter(range(1, 10 ** 6))

    def session(self):
        m = re.search(r"JSESSIONID=(\w+)", self.headers.get("Cookie", ""))
        return m.group(1) if m else ""

    def log_message(self, *a):
        pass

    def names(self):
        return sorted(FILES)  # one folder holds every fake file

    def do_GET(self):
        names = self.names()
        sid = self.session() or f"s{next(Fake.sessions)}"
        Fake.pages[sid] = 0
        body = (f'<html><body><form id="fileList" name="fileList" method="post" action="/webclient/godrive/PublicGoDrive.xhtml">'
                f'<input type="hidden" name="fileList_SUBMIT" value="1"><select name="fileTable_rppDD"><option value="250" selected>250</option></select>'
                f'<table>{page(names[:self.per_page], 0)}</table><input type="hidden" name="javax.faces.ViewState" value="vs-1"></form>'
                f'<span>Showing 1 - {self.per_page} of {len(names)}</span></body></html>').encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Set-Cookie", f"JSESSIONID={sid}; Path=/")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        form = dict(urllib.parse.parse_qsl(self.rfile.read(int(self.headers["Content-Length"])).decode()))
        names = self.names()
        if form.get("javax.faces.partial.ajax"):
            first = Fake.pages[self.session()] = int(form["fileTable_first"])
            body = f'<?xml version="1.0"?><partial-response><changes><update id="fileTable"><![CDATA[{page(names[first:first + self.per_page], first)}]]></update><update id="j_id__v_0:javax.faces.ViewState:1"><![CDATA[vs-2]]></update></changes></partial-response>'
            self.send_response(200)
            self.send_header("Content-Type", "text/xml")
            self.end_headers()
            self.wfile.write(body.encode())
            return
        assert form.get("fileList_SUBMIT") == "1" and form.get("javax.faces.ViewState", "").startswith("vs-"), form
        key = next(k for k in form if k.startswith("fileTable:") and k.endswith(":j_id_2f"))
        row = int(key.split(":")[1])
        first = Fake.pages[self.session()]
        if not first <= row < first + self.per_page:
            # The real site ignores a click on a row that isn't on the page shown and sends the page back.
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>file list</body></html>")
            return
        name = names[row]
        self.send_response(200)
        self.send_header("Content-Type", "application/force-download")
        self.end_headers()
        self.wfile.write(FILES[name])


class WellIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        W.MFT = base
        W.WELL_LAYERS = f"{base}/link/d551fb20-x"
        W.API_DBF = f"{base}/link/1eb94d66-x"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_counties_are_alphabetical_fips(self):
        for fips, name in {"001": "Anderson", "003": "Andrews", "135": "Ector", "201": "Harris", "255": "Karnes", "301": "Loving",
                           "307": "McCulloch", "329": "Midland", "389": "Reeves", "453": "Travis", "479": "Webb", "507": "Zavala"}.items():
            self.assertEqual(W.FIPS[fips], name)

    def test_status(self):
        self.assertEqual(W.status_of(86, "0"), "horizontal")
        self.assertEqual(W.status_of(4, "20010101"), "plugged")
        self.assertEqual(W.status_of(7, ""), "plugged")
        self.assertEqual(W.status_of(2, ""), "permit")
        self.assertEqual(W.status_of(None, ""), "other")

    def test_godrive_lists_every_page(self):
        g = W.GoDrive(W.WELL_LAYERS)
        files = g.list(page_rows=1)
        self.assertEqual(sorted(files), sorted(FILES))
        self.assertEqual(files["well389.zip"]["row"], 3)
        self.assertTrue(g.get("well389.zip").startswith(b"PK"))
        # Rows off the page now shown: the table is paged back to them first.
        self.assertEqual(g.get("api301.dbf"), FILES["api301.dbf"])
        self.assertTrue(g.get("well301.zip").startswith(b"PK"))

    def test_build_joins_points_and_operators(self):
        with tempfile.TemporaryDirectory() as d:
            W.main(["--out", d, "--counties", "389,301"])
            reeves = json.load(open(os.path.join(d, "wells/tx/389.json")))
            ops = json.load(open(os.path.join(d, "wells/tx/operators.json")))
            man = json.load(open(os.path.join(d, "wells/tx/manifest.json")))
            leases = json.load(open(os.path.join(d, "wells/tx/leases.json")))
        self.assertEqual(reeves["county"], "Reeves")
        by = {w[0]: w for w in reeves["wells"]}
        self.assertEqual(set(by), {"36724", "00001", "00003", "99999"})
        w = by["36724"]
        self.assertEqual((w[1], w[2], w[3], w[4]), (31.09743, -103.31167, "horizontal", 55))  # the most reliable spot wins
        self.assertEqual(reeves["ops"][w[5]], "PERMIAN ROCK OPERATING LLC")
        self.assertEqual((w[6], w[7]), ("RAYBURN 12", "1H"))
        self.assertEqual(len(w), 8)  # nothing else from the RRC files is kept
        self.assertEqual(by["00001"][3], "plugged")
        self.assertIsNone(by["00003"][1])  # operator record with no surface spot: kept, located later by county
        self.assertEqual(by["99999"][3], "permit")
        self.assertEqual(reeves["center"], [31.2487, -103.4558])  # average of the located, unplugged wells
        op = dict((n, dict(c)) for n, c in ops["operators"])
        self.assertEqual(op["PERMIAN ROCK OPERATING LLC"], {"301": 1, "389": 2})
        self.assertNotIn("OLD WELLS INC", op)  # plugged wells don't count
        # Lease name search list: active wells only, with county and operator.
        named = {(l, f, leases["ops"][o]): n for l, f, o, n in leases["leases"]}
        self.assertEqual(named[("RAYBURN 12", "389", "PERMIAN ROCK OPERATING LLC")], 2)  # 1H and 2H (2H has no surface spot yet)
        self.assertFalse(any(k[2] == "OLD WELLS INC" for k in named))
        self.assertEqual(leases["leases"], sorted(leases["leases"]))
        self.assertEqual(man["wells"], 5)
        self.assertEqual(man["counties"]["389"]["located"], 3)


if __name__ == "__main__":
    unittest.main()
