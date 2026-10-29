"""Temporary: list HiResW file names and one file's fields (NOMADS is only reachable from GitHub)."""
import re, sys, urllib.request
sys.path.insert(0, ".")
import render_models as R

day = sys.argv[1]
html = R.get(f"{R.HIRESW}/hiresw.{day}/").decode()
names = sorted(set(re.sub(r"\.f\d\d\.", ".fXX.", n) for n in re.findall(r'href="(hiresw\.t12z[^"]+)"', html)))
print("\n".join(names))
for u in (R.hiresw_url("arw")(day + "12", 12), R.hiresw_url("fv3")(day + "12", 12)):
    print("==", u)
    print(R.get(u + ".idx").decode()[:6000])
