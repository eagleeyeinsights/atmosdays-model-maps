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
import random
import tempfile
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
# ECMWF open data has identical copies on Google Cloud and AWS. The AWS copy
# often answers 503 "Slow Down" to everyone, so Google goes first and AWS is
# the backup.
ECMWF_MIRRORS = ["https://storage.googleapis.com/ecmwf-open-data", "https://ecmwf-forecasts.s3.amazonaws.com"]
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


PROB_STOPS = [[10, "#d9f99d"], [20, "#86efac"], [30, "#22c55e"], [40, "#facc15"], [50, "#fb923c"], [60, "#ef4444"],
              [70, "#dc2626"], [80, "#be185d"], [90, "#7e22ce"]]
WIND_STOPS = [[5, "#d8ecf3"], [10, "#9fd3e6"], [20, "#5ab4d6"], [30, "#3b8f4f"], [40, "#e8d54a"], [50, "#f08a2a"],
              [60, "#d7301f"], [75, "#8c1d6e"]]

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
    "jet200": {"label": "200 mb wind (about 39,000 ft): the jet stream", "unit": "mph", "stops": [
        [50, "#d6ebff"], [70, "#8fc4ff"], [90, "#4f8ff0"], [110, "#7b4fe0"], [130, "#c43fc4"], [160, "#ff6fae"]],
        "note": "Lines: 200 mb height, dam"},
    "jet300": {"label": "300 mb wind (about 30,000 ft): the jet stream", "unit": "mph", "stops": [
        [50, "#d6ebff"], [70, "#8fc4ff"], [90, "#4f8ff0"], [110, "#7b4fe0"], [130, "#c43fc4"], [160, "#ff6fae"]],
        "note": "Lines: 300 mb height, dam"},
    "jet250": {"label": "250 mb wind (about 34,000 ft): the jet stream", "unit": "mph", "stops": [[50, "#d6ebff"], [70, "#8fc4ff"], [90, "#4f8ff0"], [110, "#7b4fe0"], [130, "#c43fc4"], [160, "#ff6fae"]],
        "note": "Barbs: wind in knots. Purple dashed lines: air spreading out aloft (divergence), which helps air rise below"},
    "jet300d": {"label": "300 mb wind (about 30,000 ft): the jet stream", "unit": "mph", "stops": [[50, "#d6ebff"], [70, "#8fc4ff"], [90, "#4f8ff0"], [110, "#7b4fe0"], [130, "#c43fc4"], [160, "#ff6fae"]],
        "note": "Barbs: wind in knots. Purple dashed lines: air spreading out aloft (divergence), which helps air rise below"},
    "mslp": {"label": "Sea-level pressure", "unit": "mb", "alpha": 110, "stops": [
        [984, "#7c3aed"], [996, "#60a5fa"], [1008, "#f1f5f9"], [1016, "#f1f5f9"], [1026, "#fca5a5"], [1036, "#dc2626"]],
        "note": "Lines every 2 mb. Barbs: wind at 33 ft, knots"},
    "t3h": {"label": "Temperature change, last 3 hours", "unit": "°F", "stops": [
        [-15, "#1e3a8a"], [-10, "#2563eb"], [-5, "#60a5fa"], [-2, "#bfdbfe"], [0, "#f8fafc"], [2, "#fecaca"], [5, "#f87171"],
        [10, "#dc2626"], [15, "#7f1d1d"]], "note": "Blue: cooling (a cold front or storm outflow). Red: warming"},
    "td3h": {"label": "Dew point change, last 3 hours", "unit": "°F", "stops": [
        [-15, "#78350f"], [-10, "#b45309"], [-5, "#f59e0b"], [-2, "#fde68a"], [0, "#f8fafc"], [2, "#bbf7d0"], [5, "#4ade80"],
        [10, "#16a34a"], [15, "#14532d"]], "note": "Green: moisture flowing in. Brown: drying (a dry line or front)"},
    "cape3h": {"label": "Storm fuel change, last 3 hours (MLCAPE)", "unit": "J/kg", "stops": [
        [-2000, "#1e3a8a"], [-1000, "#3b82f6"], [-250, "#bfdbfe"], [0, "#f8fafc"], [250, "#fde68a"], [1000, "#f97316"],
        [2000, "#b91c1c"]], "note": "Red: fuel building (often daytime heating). Blue: fuel used up or cut off"},
    "shear1": {"label": "Wind change, ground to 3,300 ft (0-1 km bulk shear)", "unit": "kt", "stops": [
        [10, "#e0f2fe"], [15, "#7dd3fc"], [20, "#22c55e"], [25, "#facc15"], [30, "#f97316"], [40, "#dc2626"], [50, "#86198f"]],
        "note": "Barbs: the shear, knots. 20 kt and up with storm fuel and low cloud bases favors tornadoes"},
    "shear5": {"label": "Wind change, ground to 18,000 ft (surface to 500 mb bulk shear)", "unit": "kt", "stops": [
        [20, "#e0f2fe"], [30, "#7dd3fc"], [40, "#22c55e"], [50, "#facc15"], [60, "#f97316"], [80, "#dc2626"]],
        "note": "Barbs: the shear, knots. 40 kt and up: storms can organize and rotate"},
    "vort": {"label": "500 mb spin (absolute vorticity, about 18,000 ft)", "unit": "x10⁻⁵/s", "stops": [
        [12, "#fff7bc"], [16, "#fee391"], [20, "#fec44f"], [24, "#fe9929"], [28, "#ec7014"], [32, "#cc4c02"], [40, "#8c2d04"]],
        "note": "Lines: 500 mb height, dam. Spin moving in often sets off storms"},
    "rh700": {"label": "700 mb humidity (about 10,000 ft)", "unit": "%", "stops": [
        [50, "#e5f5e0"], [70, "#a1d99b"], [80, "#74c476"], [90, "#31a354"], [100, "#006d2c"]],
        "note": "Lines: 700 mb height, dam. Green = moist mid levels, clouds and rain likely"},
    "t850": {"label": "850 mb temperature (about 5,000 ft)", "unit": "°C", "stops": [
        [-30, "#e9dcf5"], [-20, "#8c6bb1"], [-10, "#2c5bc9"], [-5, "#3d8fe0"], [0, "#9bd6f2"], [1, "#b4e3a8"], [8, "#5cb85c"],
        [14, "#f5e050"], [20, "#f6a838"], [26, "#d42a2a"], [32, "#7a0c3a"]],
        "note": "Blue line: 0°C, the usual rain/snow dividing line. Black lines: 850 mb height, dam"},
    "mlcape": {"label": "Storm fuel, mixed layer (MLCAPE)", "unit": "J/kg", "stops": [
        [100, "#e0f2fe"], [500, "#a7f3d0"], [1000, "#fde047"], [2000, "#fb923c"], [3000, "#ef4444"], [4000, "#be185d"]]},
    "cin": {"label": "The lid on storms (CIN, mixed layer)", "unit": "J/kg", "stops": [
        [25, "#e0e7ff"], [50, "#a5b4fc"], [100, "#6366f1"], [200, "#3730a3"], [300, "#1e1b4b"]], "note": "Darker = stronger lid"},
    "shear6": {"label": "Wind change, ground to 20,000 ft (0-6 km shear)", "unit": "kt", "stops": [
        [20, "#e0f2fe"], [30, "#7dd3fc"], [40, "#22c55e"], [50, "#facc15"], [60, "#f97316"], [80, "#dc2626"]], "note": "Barbs: the shear, knots. 40 kt and up: supercells possible"},
    "srh1": {"label": "Low-level spin (0-1 km helicity)", "unit": "m²/s²", "stops": [
        [50, "#e0f2fe"], [100, "#93c5fd"], [150, "#facc15"], [250, "#f97316"], [400, "#dc2626"], [600, "#86198f"]], "note": "150 and up with storm fuel: tornado red flag"},
    "lapse": {"label": "How fast air cools with height (700-500 mb lapse rate)", "unit": "°C/km", "stops": [
        [6, "#e0f2fe"], [6.5, "#bae6fd"], [7, "#fde68a"], [7.5, "#fb923c"], [8, "#ef4444"], [9, "#86198f"]], "note": "7.5 and up: steep, bigger hail"},
    "pwat": {"label": "Water in the air (precipitable water)", "unit": "in", "stops": [
        [0.25, "#8b6b4a"], [0.5, "#d6c79a"], [0.75, "#d9f99d"], [1.0, "#86efac"], [1.25, "#22c55e"], [1.5, "#16a3a3"], [1.75, "#1f6fd1"],
        [2.0, "#5b3fc4"], [2.5, "#9b2fae"]]},
    "stp": {"label": "Tornado ingredients (significant tornado parameter)", "unit": "", "stops": [
        [0.5, "#fde68a"], [1, "#fb923c"], [2, "#ef4444"], [4, "#b91c1c"], [6, "#86198f"], [10, "#4c1d95"]], "note": "Over 1: ingredients for strong tornadoes together"},
    "scp": {"label": "Supercell ingredients (supercell composite)", "unit": "", "stops": [
        [1, "#bae6fd"], [2, "#22c55e"], [4, "#facc15"], [8, "#f97316"], [12, "#dc2626"], [20, "#86198f"]], "note": "Over 1: rotating storms favored"},
    "prob": {"label": "Chance among the model runs", "unit": "%", "stops": PROB_STOPS},
    "chance": {"label": "Chance", "unit": "%", "stops": PROB_STOPS},
    "spread": {"label": "How much the GEFS runs disagree at 500 mb (spread)", "unit": "m", "stops": [
        [10, "#e0f2fe"], [20, "#93c5fd"], [30, "#facc15"], [45, "#f97316"], [60, "#dc2626"], [90, "#86198f"]],
        "note": "Lines: average 500 mb height, dam. More spread = less certain"},
    "smoke": {"label": "Wildfire smoke near the ground", "unit": "µg/m³", "stops": [
        [5, "#f5f0dc"], [12, "#e6d3a3"], [35, "#d9a066"], [55, "#b5651d"], [150, "#7f1d1d"], [250, "#4a044e"]],
        "note": "35 and up is unhealthy for sensitive groups (like PM2.5)"},
    "ptype": {"label": "Rain, snow, sleet or freezing rain", "unit": "", "steps": True, "min": "Rain", "max": "Freezing rain",
              "stops": [[1, "#22c55e"], [2, "#3b82f6"], [3, "#f97316"], [4, "#ec4899"]],
              "note": "Green: rain. Blue: snow. Orange: sleet. Pink: freezing rain"},
    "snow6": {"label": "Snowfall, 6 hours", "unit": "in", "stops": [
        [0.1, "#dbeafe"], [0.5, "#93c5fd"], [1, "#60a5fa"], [2, "#2563eb"], [4, "#1e3a8a"], [6, "#7c3aed"], [8, "#c026d3"],
        [12, "#f472b6"]]},
    "ice": {"label": "Ice buildup from freezing rain, 6 hours", "unit": "in", "stops": [
        [0.01, "#fbcfe8"], [0.05, "#f472b6"], [0.1, "#db2777"], [0.25, "#9d174d"], [0.5, "#581c87"], [1, "#1e1b4b"]],
        "note": "0.25 inch and up brings down limbs and power lines"},
    "sky": {"label": "Cloud cover", "unit": "%", "alpha": 170, "stops": [
        [10, "#f1f5f9"], [25, "#cbd5e1"], [50, "#94a3b8"], [75, "#64748b"], [100, "#334155"]]},
    "wind80": {"label": "Wind at 260 ft (80 m)", "unit": "mph", "stops": WIND_STOPS},
    "wind100": {"label": "Wind at 330 ft (100 m, wind-farm hub height)", "unit": "mph", "stops": WIND_STOPS},
}

# Picture name -> label for the map list on the page.
FIELD_NAMES = {
    "temp": "Temperature", "precip": "Rain and snow, with pressure", "wind": "Wind", "upper": "Jet stream and 500 mb",
    "radar": "Computer radar", "rain1": "Rain, 1 hour", "rain6": "Rain, 6 hours", "gust": "Wind gusts",
    "uh": "Storm rotation tracks", "storms40": "Chance of strong storms", "rotation": "Chance of rotating storms",
    "heavy": "Chance of 1 inch of rain in 3 hours",
    "dew": "Dew point (humidity)", "cape": "Storm fuel (CAPE)", "qpf": "Total rain since the run started",
    "snow": "Total snowfall", "hail": "Hail size", "ltng": "Lightning",
    "h200": "200 mb: jet stream (39,000 ft)", "h300": "300 mb: jet stream (30,000 ft)", "h500": "500 mb: heights and spin (18,000 ft)",
    "h700": "700 mb: mid-level moisture (10,000 ft)", "h850": "850 mb: temperature (5,000 ft)",
    "mlcape": "Storm fuel, mixed layer (MLCAPE)", "cin": "Lid on storms (CIN)", "shear6": "Wind change to 20,000 ft (0-6 km shear)",
    "srh1": "Low-level spin (0-1 km helicity)", "lapse": "Mid-level lapse rate", "pwat": "Water in the air (precipitable water)",
    "stp": "Tornado ingredients (STP)", "scp": "Supercell ingredients (SCP)",
    "shear1": "Wind change to 3,300 ft (0-1 km bulk shear)", "shear": "Wind change to 18,000 ft (bulk shear)",
    "mslp": "Sea-level pressure and wind", "t3h": "Temperature change, 3 hours", "td3h": "Dew point change, 3 hours",
    "cape3h": "Storm fuel change, 3 hours", "jet250": "250 mb: jet stream (34,000 ft)", "jet300": "300 mb: jet stream (30,000 ft)",
    # Chance maps (GEFS and REFS), the blend (NBM) and the new storm model (RRFS).
    "rain24half": "Chance of 1/2 inch of rain in 24 hours", "rain24one": "Chance of 1 inch of rain in 24 hours",
    "freeze": "Chance of freezing (below 32°F)", "wind25": "Chance of wind over 25 mph", "spread500": "500 mb: average and spread",
    "pfrzra": "Chance of freezing rain", "wind34": "Chance of wind over 34 mph at 260 ft", "fog": "Chance of dense fog (under 1/4 mile)",
    "smoke": "Wildfire smoke near the ground", "ptype": "Rain, snow, sleet or freezing rain",
    "pop6": "Chance of rain or snow, 6 hours", "tstm": "Chance of thunder, 6 hours", "snow6": "Snowfall, 6 hours",
    "ice6": "Ice buildup, 6 hours", "sky": "Cloud cover", "wind80": "Wind at 260 ft (80 m)",
    "wind100": "Wind at 330 ft (100 m, wind farms)",
}
# Legend used by each picture.
FIELD_LEGEND = {"precip": "precip", "storms40": "prob", "rotation": "prob", "heavy": "prob",
                "h200": "jet200", "h300": "jet300", "h500": "vort", "h700": "rh700", "h850": "t850",
                "jet300": "jet300d", "shear": "shear5",
                **{k: "prob" for k in ("rain24half", "rain24one", "freeze", "wind25", "pfrzra", "wind34", "fog")},
                "pop6": "chance", "tstm": "chance", "spread500": "spread", "ice6": "ice"}

# Below these values the map stays see-through.
FADE = {"temp": None, "precip": (0.005, 0.02), "wind": (5, 15), "upper": (45, 70), "radar": (4, 6),
        "rain1": (0.005, 0.015), "rain6": (0.005, 0.02), "gust": (10, 20), "uh": (15, 30), "prob": (5, 12),
        "dew": None, "cape": (100, 300), "qpf": (0.005, 0.03), "snow": (0.05, 0.3), "hail": (0.2, 0.3), "ltng": (0.02, 0.08),
        "jet200": (45, 70), "jet300": (45, 70), "vort": (10, 14), "rh700": (60, 75), "t850": None,
        "mlcape": (100, 300), "cin": (15, 35), "shear6": (15, 25), "srh1": (40, 80), "lapse": (5.8, 6.3), "pwat": None,
        "stp": (0.2, 0.6), "scp": (0.5, 1.2), "jet250": (45, 70), "jet300d": (45, 70), "mslp": None,
        "shear1": (8, 14), "shear5": (15, 25),
        "chance": (5, 12), "spread": (5, 12), "smoke": (3, 8), "ptype": (0.5, 0.6), "snow6": (0.05, 0.2),
        "ice": (0.005, 0.01), "sky": (5, 30), "wind80": (5, 15), "wind100": (5, 15),
        # Changes: see-through near zero, either way.
        "t3h": ("abs", 1, 3), "td3h": ("abs", 1, 3), "cape3h": ("abs", 100, 300)}


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
    top = leg.get("alpha", 215)
    if fade is None:
        alpha = np.full(values.shape, float(top))
    elif fade[0] == "abs":
        alpha = np.clip((np.abs(v) - fade[1]) / (fade[2] - fade[1]), 0, 1) * top
    else:
        alpha = np.clip((v - fade[0]) / (fade[1] - fade[0]), 0, 1) * top
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

    def rotation(self, ar):
        """Angle (radians) that turns this grid's winds into true east/north winds at each output pixel."""
        dl = np.radians(ar.lons)[None, :] - self.lov
        dl = (dl + np.pi) % (2 * np.pi) - np.pi
        return np.broadcast_to(self.n * dl, (len(ar.lats), len(ar.lons)))

    def sample(self, ar):
        la, lo = np.meshgrid(ar.lats, ar.lons, indexing="ij")
        x, y = self.xy(la, lo)
        cols, rows = (x - self.x0) / self.dx, (y - self.y0) / self.dy
        ny, nx = self.values.shape
        out = map_coordinates(np.nan_to_num(self.values, nan=0), [rows, cols], order=1, mode="nearest")
        bad = map_coordinates(np.isnan(self.values).astype(np.float32), [rows, cols], order=1, mode="nearest") > 0.5
        out[bad | (rows < -0.5) | (cols < -0.5) | (rows > ny - 0.5) | (cols > nx - 0.5)] = np.nan
        return out


def earth_wind(gu, gv, ar):
    """u and v on the output pixels as true east/north winds (HRRR and HREF give them along their own grid)."""
    u, v = gu.sample(ar), gv.sample(ar)
    if not hasattr(gu, "rotation"):
        return u, v
    a = gu.rotation(ar)
    c, s = np.cos(a), np.sin(a)
    return c * u + s * v, -s * u + c * v


R_EARTH = 6371000.0


def kinematics(u, v, ar):
    """Divergence and relative vorticity (1/s) of an east/north wind on a map area."""
    lat = np.radians(ar.lats)
    lon = np.radians(ar.lons)
    cos, tan = np.cos(lat)[:, None], np.tan(lat)[:, None]
    du_x, dv_x = np.gradient(u, lon, axis=1), np.gradient(v, lon, axis=1)
    du_y, dv_y = np.gradient(u, lat, axis=0), np.gradient(v, lat, axis=0)
    div = du_x / (R_EARTH * cos) + dv_y / R_EARTH - v * tan / R_EARTH
    vort = dv_x / (R_EARTH * cos) - du_y / R_EARTH + u * tan / R_EARTH
    return div, vort


def decode(msg_bytes):
    gid = eccodes.codes_new_from_message(msg_bytes)
    try:
        g = lambda k: eccodes.codes_get(gid, k)  # noqa: E731
        vals = eccodes.codes_get_values(gid)
        missing = g("missingValue")
        if g("bitmapPresent") or (eccodes.codes_is_defined(gid, "missingValueManagementUsed") and g("missingValueManagementUsed")):
            vals = np.where(vals == missing, np.nan, vals)
        if g("gridType") == "lambert":
            nx, ny = g("Nx"), g("Ny")
            vals = vals.reshape(ny, nx)
            if eccodes.codes_is_defined(gid, "alternativeRowScanning") and g("alternativeRowScanning"):
                # Every other row is stored right to left (NBM).
                vals[1::2] = vals[1::2, ::-1]
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
            if e.code in (429, 503):  # "Slow Down": back off longer, with jitter so threads spread out
                time.sleep(min(60, 4 * 2 ** i) + random.random() * 3)
                continue
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


def ranges_get(url, spans, gap=1 << 20, fetch=None):
    """Fetch byte ranges, joining ones less than `gap` apart into one request
    (fewer requests keeps S3 from answering 503 Slow Down)."""
    # The last field in a file has no end offset (""): it runs to the end of the file.
    order = sorted(spans.items(), key=lambda kv: kv[1][0])
    groups = []
    for k, (a, b) in order:
        if groups and groups[-1][1] != "" and a - groups[-1][1] <= gap:
            groups[-1][1] = "" if b == "" else max(groups[-1][1], b)
            groups[-1][2].append((k, a, b))
        else:
            groups.append([a, b, [(k, a, b)]])
    out = {}
    for a0, b0, parts in groups:
        raw = (fetch or get)(url, {"Range": f"bytes={a0}-{b0}"})
        for k, a, b in parts:
            out[k] = raw[a - a0:] if b == "" else raw[a - a0:b - a0 + 1]
    return out


def idx_spans(text, want):
    """Byte ranges from a NOAA .idx listing. want: {name: test(var, level, fcst, rest)}."""
    rows = [line.split(":") for line in text.splitlines() if line.strip()]
    offs = [int(r[1]) for r in rows]
    spans = Spans()
    for i, r in enumerate(rows):
        # Some files (NAM) pack u and v wind in one message, listed as 624.1 and 624.2
        # with the same offset: the message ends where the next offset starts.
        nxt = next((o for o in offs[i + 1:] if o > offs[i]), None)
        end = nxt - 1 if nxt is not None else ""
        var, level, fcst, rest = r[3], r[4], r[5] if len(r) > 5 else "", ":".join(r[6:])
        for name, test in want.items():
            if name not in spans and test(var, level, fcst, rest):
                spans[name] = (offs[i], end)
                if "." in r[0]:
                    spans.parts[name] = int(r[0].split(".")[1])
    return spans


class Spans(dict):
    """Byte ranges by name; .parts gives the field number inside a packed message."""

    def __init__(self):
        super().__init__()
        self.parts = {}


def unpack_parts(raw, spans):
    """Turn packed messages (several fields in one) into the one wanted field each."""
    for name, part in spans.parts.items():
        if part > 1 and name in raw:
            eccodes.codes_grib_multi_support_on()
            try:
                with tempfile.TemporaryFile() as f:  # eccodes reads packed fields from a real file
                    f.write(raw[name])
                    f.seek(0)
                    gid = None
                    for _ in range(part):
                        if gid is not None:
                            eccodes.codes_release(gid)
                        gid = eccodes.codes_grib_new_from_file(f)
                    raw[name] = eccodes.codes_get_message(gid)
                    eccodes.codes_release(gid)
            finally:
                eccodes.codes_grib_multi_support_off()
    return raw


def k_to_f(k):
    return (k - 273.15) * 9 / 5 + 32


# GFS: the .idx file lists each field with its byte offset.
GFS_WANT = {
    "t2": "TMP:2 m above ground:", "msl": "PRMSL:mean sea level:", "u10": "UGRD:10 m above ground:",
    "v10": "VGRD:10 m above ground:", "z500": "HGT:500 mb:", "u250": "UGRD:250 mb:", "v250": "VGRD:250 mb:",
    "d2": "DPT:2 m above ground:", "cape": "CAPE:surface:",
    "u500": "UGRD:500 mb:", "v500": "VGRD:500 mb:",
    # Upper-air maps.
    "z200": "HGT:200 mb:", "u200": "UGRD:200 mb:", "v200": "VGRD:200 mb:",
    "z300": "HGT:300 mb:", "u300": "UGRD:300 mb:", "v300": "VGRD:300 mb:", "av500": "ABSV:500 mb:",
    "z700": "HGT:700 mb:", "rh700": "RH:700 mb:", "z850": "HGT:850 mb:", "t850": "TMP:850 mb:",
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
        **surface_wind(g),
        **upper_air(g, lambda: g["av500"] * 1e5),
    }
    return out


KT = 1.94384  # m/s -> knots


def surface_wind(g):
    """10 m wind and the ground-to-500 mb bulk shear (knots), for the pressure and shear maps."""
    out = {"u10kt": g["u10"] * KT, "v10kt": g["v10"] * KT}
    if "u500" in g and "v500" in g:
        out["shr_u"], out["shr_v"] = (g["u500"] - g["u10"]) * KT, (g["v500"] - g["v10"]) * KT
    return out


def coriolis(ar):
    return (2 * 7.2921e-5 * np.sin(np.radians(ar.lats)))[:, None] * np.ones((1, ar.width))


def upper_air(g, vort):
    """Upper-air fields in map units: heights in dam, wind in mph, vorticity x1e-5/s, RH %, temperature C."""
    out = {}
    try:
        for lv in (200, 300):
            out[f"z{lv}"] = g[f"z{lv}"] / 10
            out[f"w{lv}"] = np.hypot(g[f"u{lv}"], g[f"v{lv}"]) * 2.23694
        out["av500"] = vort()
        out["z700"], out["rh700"] = g["z700"] / 10, g["rh700"]
        out["z850"], out["t850"] = g["z850"] / 10, g["t850"] - 273.15
    except KeyError as e:  # a missing field just leaves those maps out
        print(f"  upper air: missing {e}", flush=True)
        return {}
    return out


# ECMWF open data: a JSON-lines .index file with byte offsets.
EC_WANT = {("2t", None): "t2", ("msl", None): "msl", ("10u", None): "u10", ("10v", None): "v10",
           ("gh", "500"): "z500", ("u", "250"): "u250", ("v", "250"): "v250", ("tp", None): "tp",
           ("2d", None): "d2", ("mucape", None): "cape", ("sf", None): "sf",
           ("gh", "200"): "z200", ("u", "200"): "u200", ("v", "200"): "v200", ("gh", "300"): "z300", ("u", "300"): "u300",
           ("v", "300"): "v300", ("vo", "500"): "vo500", ("gh", "700"): "z700", ("r", "700"): "rh700", ("gh", "850"): "z850",
           ("t", "850"): "t850", ("u", "500"): "u500", ("v", "500"): "v500"}


def ec_url(run, h, src="ifs"):
    """Path of one forecast hour, without the mirror in front. src: "ifs" (the regular
    ECMWF model) or "aifs-single" (ECMWF's AI model, same files and names)."""
    return f"/{run[:8]}/{run[8:]}z/{src}/0p25/oper/{run}0000-{h}h-oper-fc"


def ec_get(path, headers=None):
    """Download from the first ECMWF mirror that answers."""
    err = None
    for base in ECMWF_MIRRORS:
        try:
            return get(base + path, headers, tries=3)
        except Exception as e:  # noqa: BLE001  try the next copy
            err = e
    raise err


def ec_exists(path):
    return any(exists(base + path) for base in ECMWF_MIRRORS)


def ec_latest(now, src="ifs", last=HOURS[-1]):
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= dt.timedelta(hours=t.hour % 12)
    for _ in range(4):
        run = t.strftime("%Y%m%d%H")
        if ec_exists(ec_url(run, last, src) + ".index"):
            return run
        t -= dt.timedelta(hours=12)
    return None


_ec_tp = {}


def ec_tp(run, h, src="ifs"):
    """Total precipitation since the start of the run (inches)."""
    if h == 0:
        return None
    key = (src, run, h)
    if key not in _ec_tp:
        idx = [json.loads(x) for x in ec_get(ec_url(run, h, src) + ".index").decode().splitlines() if x.strip()]
        r = next(x for x in idx if x["param"] == "tp")
        raw = ec_get(ec_url(run, h, src) + ".grib2", {"Range": f"bytes={r['_offset']}-{r['_offset'] + r['_length'] - 1}"})
        # The regular model gives meters of water; the AI model gives kg/m2 (= mm).
        _ec_tp[key] = decode(raw).sample(area("na")) * (1000 if src == "ifs" else 1) / 25.4  # -> in
    return _ec_tp[key]


def ec_grids(run, h, src, want):
    """The wanted ECMWF fields of one hour on the North America map pixels (precipitation is read separately)."""
    idx = [json.loads(x) for x in ec_get(ec_url(run, h, src) + ".index").decode().splitlines() if x.strip()]
    spans = {}
    for r in idx:
        name = want.get((r["param"], r.get("levelist")))
        if name and name != "tp" and name not in spans:
            spans[name] = (r["_offset"], r["_offset"] + r["_length"] - 1)
    raw = ranges_get(ec_url(run, h, src) + ".grib2", spans, fetch=ec_get)
    ar = area("na")
    return {k: decode(v).sample(ar) for k, v in raw.items()}


def ec_tp6(run, h, src="ifs"):
    if h < 6:
        return None
    now_tp, before = ec_tp(run, h, src), ec_tp(run, h - 6, src)
    return np.clip(now_tp - (before if before is not None else 0), 0, None)


def ec_fields(run, h):
    g = ec_grids(run, h, "ifs", EC_WANT)
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
        **surface_wind(g),
        # ECMWF gives relative vorticity; add the Earth's spin to match GFS's absolute vorticity.
        **upper_air(g, lambda: (g["vo500"] + coriolis(area("na"))) * 1e5),
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
        # Storm ingredients for the mesoanalysis page.
        "mlcape": lambda v, lv, f, r: v == "CAPE" and lv == "90-0 mb above ground",
        "mlcin": lambda v, lv, f, r: v == "CIN" and lv == "90-0 mb above ground",
        "mucape": lambda v, lv, f, r: v == "CAPE" and lv == "255-0 mb above ground",
        "srh1": lambda v, lv, f, r: v == "HLCY" and lv == "1000-0 m above ground",
        "srh3": lambda v, lv, f, r: v == "HLCY" and lv == "3000-0 m above ground",
        "shu": lambda v, lv, f, r: v == "VUCSH" and lv == "0-6000 m above ground",
        "shv": lambda v, lv, f, r: v == "VVCSH" and lv == "0-6000 m above ground",
        "lcl": lambda v, lv, f, r: v == "HGT" and lv == "level of adiabatic condensation from sfc",
        "t500": lambda v, lv, f, r: v == "TMP" and lv == "500 mb",
        "t700": lambda v, lv, f, r: v == "TMP" and lv == "700 mb",
        "z500": lambda v, lv, f, r: v == "HGT" and lv == "500 mb",
        "z700": lambda v, lv, f, r: v == "HGT" and lv == "700 mb",
        "pwat": lambda v, lv, f, r: v == "PWAT" and lv.startswith("entire atmosphere"),
        "sh1u": lambda v, lv, f, r: v == "VUCSH" and lv == "0-1000 m above ground",
        "sh1v": lambda v, lv, f, r: v == "VVCSH" and lv == "0-1000 m above ground",
        # Upper air: the jet stream and the 500, 700 and 850 mb maps.
        **{f"{c}{p}": (lambda c, p: lambda v, lv, f, r: v == {"u": "UGRD", "v": "VGRD"}[c] and lv == f"{p} mb")(c, p)
           for c in "uv" for p in (250, 300, 500, 700, 850)},
        "z850": lambda v, lv, f, r: v == "HGT" and lv == "850 mb",
        "t850": lambda v, lv, f, r: v == "TMP" and lv == "850 mb",
        "d700": lambda v, lv, f, r: v == "DPT" and lv == "700 mb",
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
    winds = {}
    for a, b in (("u10", "v10"), ("shu", "shv"), ("sh1u", "sh1v"), *((f"u{p}", f"v{p}") for p in (250, 300, 500, 700, 850))):
        if a in grids and b in grids:
            winds[a] = earth_wind(grids[a], grids[b], ar)
    skip = {"u80", "v80", "u10", "v10", "sh1u", "sh1v", *(f"{c}{p}" for c in "uv" for p in (250, 300, 500, 700, 850))}
    g = {k: v.sample(ar) for k, v in grids.items() if k not in skip}
    del grids
    before = hrrr_before(run, h)
    change = {}
    if before:
        if "t2" in before:
            change["t3h"] = k_to_f(g["t2"]) - before["t2"]
        if "d2" in before and "d2" in g:
            change["td3h"] = k_to_f(g["d2"]) - before["d2"]
        if "mlcape" in before and "mlcape" in g:
            change["cape3h"] = g["mlcape"] - before["mlcape"]
    return {
        "refc": g["refc"], "t2": k_to_f(g["t2"]), "gust": g["gust"] * 2.23694, "msl": g["msl"] / 100,
        "tp1": g["tp1"] / 25.4 if "tp1" in g else None, "uh": g.get("uh"),
        "d2": k_to_f(g["d2"]) if "d2" in g else None, "cape": g.get("cape"),
        "qpf": g["tpt"] / 25.4 if "tpt" in g else None,
        "snow": g["snow"] * 39.37 if "snow" in g else None,  # m -> in
        "hail": g["hail"] * 39.37 if "hail" in g else None,  # m -> in
        "ltng": g.get("ltng"),
        "pts": pts,
        "meso": meso_fields(g),
        "change": change,
        "winds": {k: (u * KT, v * KT) for k, (u, v) in winds.items()},
        "z500": g["z500"] / 10 if "z500" in g else None, "z700": g["z700"] / 10 if "z700" in g else None,
        "z850": g["z850"] / 10 if "z850" in g else None, "t850": g["t850"] - 273.15 if "t850" in g else None,
        "rh700": rh_from(g["t700"], g["d700"]) if "t700" in g and "d700" in g else None,
    }


def rh_from(t_k, td_k):
    """Relative humidity (%) from temperature and dew point (Kelvin)."""
    t, td = t_k - 273.15, td_k - 273.15
    return np.clip(100 * np.exp(17.625 * td / (243.04 + td) - 17.625 * t / (243.04 + t)), 0, 100)


def hrrr_before(run, h):
    """Temperature, dew point (F) and MLCAPE three hours before the map time. For the
    first hours of a run this comes from an earlier hourly HRRR run's starting hour."""
    if h >= 3:
        r2, h2 = run, h - 3
    else:
        t = dt.datetime.strptime(run, "%Y%m%d%H") + dt.timedelta(hours=h - 3)
        r2, h2 = t.strftime("%Y%m%d%H"), 0
    want = {
        "t2": lambda v, lv, f, r: v == "TMP" and lv == "2 m above ground",
        "d2": lambda v, lv, f, r: v == "DPT" and lv == "2 m above ground",
        "mlcape": lambda v, lv, f, r: v == "CAPE" and lv == "90-0 mb above ground",
    }
    try:
        spans = idx_spans(get(hrrr_url(r2, h2) + ".idx").decode(), want)
        raw = ranges_get(hrrr_url(r2, h2), spans)
    except Exception as e:  # noqa: BLE001  the change maps are just left out
        print(f"  hrrr 3-hour change for +{h}h not available: {e!r}", flush=True)
        return None
    ar = area("conus")
    out = {k: decode(v).sample(ar) for k, v in raw.items()}
    for k in ("t2", "d2"):
        if k in out:
            out[k] = k_to_f(out[k])
    return out


def meso_fields(g):
    """Storm ingredients (SPC mesoanalysis style) from HRRR fields; missing ones are left out."""
    out = {}
    if "mlcape" in g:
        out["mlcape"] = g["mlcape"]
    if "mlcin" in g:
        out["cin"] = -g["mlcin"]  # shown as a positive "lid" strength
    if "srh1" in g:
        out["srh1"] = g["srh1"]
    if "shu" in g and "shv" in g:
        out["shear6"] = np.hypot(g["shu"], g["shv"]) * 1.94384  # m/s -> kt
    if all(k in g for k in ("t500", "t700", "z500", "z700")):
        with np.errstate(invalid="ignore", divide="ignore"):
            out["lapse"] = (g["t700"] - g["t500"]) / ((g["z500"] - g["z700"]) / 1000)
    if "pwat" in g:
        out["pwat"] = g["pwat"] / 25.4  # kg/m2 = mm -> in
    if all(k in g for k in ("mlcape", "mlcin", "srh1", "shu", "shv", "lcl")):
        # Significant tornado parameter (fixed layer, Thompson et al. 2003/2012), mixed-layer version.
        shr = np.hypot(g["shu"], g["shv"])
        lcl_t = np.clip((2000 - g["lcl"]) / 1000, 0, 1)
        shr_t = np.where(shr < 12.5, 0, np.clip(shr / 20, 0, 1.5))
        cin_t = np.clip((200 + g["mlcin"]) / 150, 0, 1)
        out["stp"] = np.clip(g["mlcape"] / 1500 * lcl_t * (g["srh1"] / 150) * shr_t * cin_t, 0, None)
    if all(k in g for k in ("mucape", "srh3", "shu", "shv")):
        # Supercell composite parameter (Thompson et al.), with 0-6 km shear and 0-3 km helicity.
        shr = np.hypot(g["shu"], g["shv"])
        shr_t = np.where(shr < 10, 0, np.clip(shr / 20, 0, 1))
        out["scp"] = np.clip(g["mucape"] / 1000 * (g["srh3"] / 50) * shr_t, 0, None)
    return out


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


# ---- more models (added October 2026) ------------------------------------------
#
# All from free public copies on Amazon (NOAA Open Data Dissemination) or ECMWF
# open data, so they work without NOMADS. NOAA data is public domain; ECMWF open
# data (including the AIFS AI model) is CC BY 4.0.

RRFS_OPS = "https://noaa-rrfs-ops-pds.s3.amazonaws.com"
GEFS = "https://noaa-gefs-pds.s3.amazonaws.com"
NBM = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
AIGFS = "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com"


def prob_under(rest, x):
    return f"prob <{x}:" in rest + ":"


# REFS: NOAA's new storm-model ensemble (RRFS members plus two HRRR runs). It
# replaces the HREF, which NOAA is retiring along with the NAM and HiResW. Hourly
# to 60 hours over the lower 48; only the 00 and 12 UTC runs are drawn.
def refs_url(run, h, product="pmmn"):
    return f"{RRFS_OPS}/refs.{run[:8]}/{run[8:]}/ensprod/refs.t{run[8:]}z.{product}.f{h:02d}.conus.grib2"


def refs_fields(run, h):
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
            "pfreeze": lambda v, lv, f, r: v == "TMP" and lv == "2 m above ground" and prob_under(r, 273.15),
            "pfrzra": lambda v, lv, f, r: v == "CFRZR" and r.startswith("prob >=1"),
            # 15.4 m/s = 34 mph at 80 m (about 260 ft, crane height).
            "pwind": lambda v, lv, f, r: v == "WIND" and lv == "80 m above ground" and prob_over(r, 15.4),
            # Under 400 m (1/4 mile): dense fog.
            "pfog": lambda v, lv, f, r: v == "VIS" and prob_under(r, 400),
        },
    }
    ar = area("conus")
    g = {}
    for product, want in parts.items():
        url = refs_url(run, h, product)
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
        **{k: pct(g.get(k)) for k in ("p40", "puh", "prain", "pfreeze", "pfrzra", "pwind", "pfog")},
    }


def draw_refs(f, h):
    out = draw_href(f, h)
    ar = area("conus")
    for key, name in (("pfreeze", "freeze"), ("pfrzra", "pfrzra"), ("pwind", "wind34"), ("pfog", "fog")):
        out[name] = picture(ar, fill(f.get(key), "prob"))
    return out


# RRFS: NOAA's new 3 km model for North America. It replaces the NAM (and NAM
# 3 km) and HiResW. The 00/06/12/18 UTC runs go out 84 hours; we draw every hour
# to 48 hours, then every 3 hours, from the 00 and 12 UTC runs.
def rrfs_url(run, h):
    return f"{RRFS_OPS}/rrfs.{run[:8]}/{run[8:]}/rrfs.t{run[8:]}z.2dfld.3km.f{h:03d}.conus.grib2"


RRFS_HOURS = list(range(0, 49)) + list(range(51, 85, 3))


def rrfs_fields(run, h):
    win = hour_window(h, 1)
    total = (f"0-{h} hour acc fcst", f"0-{h // 24} day acc fcst")
    want = {
        "refc": lambda v, lv, f, r: v == "REFC" and lv.startswith("entire atmosphere"),
        "t2": lambda v, lv, f, r: v == "TMP" and lv == "2 m above ground",
        "d2": lambda v, lv, f, r: v == "DPT" and lv == "2 m above ground",
        "gust": lambda v, lv, f, r: v == "GUST" and lv == "surface",
        "msl": lambda v, lv, f, r: v == "MSLET" and lv == "mean sea level",
        "u10": lambda v, lv, f, r: v == "UGRD" and lv == "10 m above ground",
        "v10": lambda v, lv, f, r: v == "VGRD" and lv == "10 m above ground",
        "cape": lambda v, lv, f, r: v == "CAPE" and lv == "surface",
        "tp1": lambda v, lv, f, r: h >= 1 and v == "APCP" and f == f"{win} acc fcst",
        "tpt": lambda v, lv, f, r: h >= 1 and v == "APCP" and f in total,
        "snow": lambda v, lv, f, r: h >= 1 and v == "ASNOW" and f in total,
        "uh": lambda v, lv, f, r: h >= 1 and v == "MXUPHL" and lv == "5000-2000 m above ground" and f == f"{win} max fcst",
        "hail": lambda v, lv, f, r: h >= 1 and v == "HAIL" and lv == "surface" and f == f"{win} max fcst",
        # Smoke: organic particles (from fires) 8 m above the ground.
        "smoke": lambda v, lv, f, r: v == "MASSDEN" and lv == "8 m above ground" and "Particulate organic matter" in r
                                       and "ave" not in f,
        **{k: (lambda name: lambda v, lv, f, r: v == name and lv == "surface" and f.endswith(" fcst") and "-" not in f)(k.upper())
           for k in ("crain", "csnow", "cicep", "cfrzr")},
    }
    spans = idx_spans(get(rrfs_url(run, h) + ".idx").decode(), want)
    raw = ranges_get(rrfs_url(run, h), spans)
    ar = area("conus")
    grids = {k: decode(v) for k, v in raw.items()}
    winds = {}
    if "u10" in grids and "v10" in grids:
        u, v = earth_wind(grids.pop("u10"), grids.pop("v10"), ar)
        winds["u10"] = (u * KT, v * KT)
    g = {k: v.sample(ar) for k, v in grids.items()}
    del grids
    ptype = None
    if all(k in g for k in ("crain", "csnow", "cicep", "cfrzr")):
        # Worst type wins: freezing rain, then sleet, snow, rain. Blank where none falls.
        ptype = np.full(g["crain"].shape, np.nan)
        for k, code in (("crain", 1), ("csnow", 2), ("cicep", 3), ("cfrzr", 4)):
            ptype = np.where(g[k] >= 0.5, code, ptype)
    return {
        "refc": g.get("refc"), "t2": k_to_f(g["t2"]), "d2": k_to_f(g["d2"]) if "d2" in g else None,
        "gust": g["gust"] * 2.23694 if "gust" in g else None, "msl": g["msl"] / 100 if "msl" in g else None,
        "cape": g.get("cape"), "tp1": g["tp1"] / 25.4 if "tp1" in g else None,
        "qpf": g["tpt"] / 25.4 if "tpt" in g else None, "snow": g["snow"] * 39.37 if "snow" in g else None,
        "uh": g.get("uh"), "hail": g["hail"] * 39.37 if "hail" in g else None,
        "smoke": g["smoke"] * 1e9 if "smoke" in g else None,  # kg/m3 -> micrograms/m3
        "ptype": ptype, "winds": winds,
    }


def draw_rrfs(f, h):
    ar = area("conus")
    out = {
        "radar": picture(ar, fill(f["refc"], "radar"), sharp=True),
        "temp": temp_pic(ar, f["t2"]),
        "rain1": rain_pic(ar, f["tp1"], "rain1", f["msl"], 14),
        "gust": picture(ar, fill(f["gust"], "gust")),
        "uh": picture(ar, fill(f["uh"], "uh"), sharp=True),
        "dew": picture(ar, fill(f["d2"], "dew")),
        "cape": picture(ar, fill(f["cape"], "cape")),
        "qpf": picture(ar, fill(f["qpf"], "qpf")),
        "snow": picture(ar, fill(f["snow"], "snow")),
        "hail": picture(ar, fill(f["hail"], "hail"), sharp=True),
        "smoke": picture(ar, fill(f["smoke"], "smoke")),
        "ptype": picture(ar, fill(f["ptype"], "ptype"), sharp=True),
    }
    if f["msl"] is not None and "u10" in f["winds"]:
        out["mslp"] = mslp_pic(ar, f["msl"], *f["winds"]["u10"], 16)
    return out


# GEFS: NOAA's global ensemble, 31 runs (a control run and 30 with small changes)
# out to 16 days. We count how many runs show each event at each spot. 00 and 12
# UTC runs, every 6 hours, from the 0.5 degree files.
GEFS_MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
GEFS_HOURS = list(range(6, 385, 6))


def gefs_url(run, h, member):
    return f"{GEFS}/gefs.{run[:8]}/{run[8:]}/atmos/pgrb2ap5/{member}.t{run[8:]}z.pgrb2a.0p50.f{h:03d}"


def gefs_latest(now):
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= dt.timedelta(hours=t.hour % 12)
    for _ in range(4):
        run = t.strftime("%Y%m%d%H")
        if all(exists(gefs_url(run, GEFS_HOURS[-1], m) + ".idx") for m in (GEFS_MEMBERS[0], GEFS_MEMBERS[-1])):
            return run
        t -= dt.timedelta(hours=12)
    return None


def gefs_member(run, h, member):
    """One member's fields on GEFS's own 0.5 degree grid (small; the chances are
    worked out on this grid and only the results are put on the map pixels)."""
    want = {
        "t2": lambda v, lv, f, r: v == "TMP" and lv == "2 m above ground",
        "u10": lambda v, lv, f, r: v == "UGRD" and lv == "10 m above ground",
        "v10": lambda v, lv, f, r: v == "VGRD" and lv == "10 m above ground",
        "z500": lambda v, lv, f, r: v == "HGT" and lv == "500 mb",
        "tp6": lambda v, lv, f, r: v == "APCP" and f == f"{hour_window(h, 6)} acc fcst",
    }
    url = gefs_url(run, h, member)
    spans = idx_spans(get(url + ".idx").decode(), want)
    return {k: decode(v) for k, v in ranges_get(url, spans).items()}


class GefsRun:
    """Downloads every member hour by hour, in order (24-hour rain adds up each
    member's last four 6-hour amounts), and turns them into chances."""

    def __init__(self, run):
        self.run, self.recent = run, {}

    def fields(self, h):
        n = 0
        freeze = wind = zsum = zsq = None
        rain_half = rain_one = None
        n_rain = 0
        geo = None
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            for m, g in zip(GEFS_MEMBERS, pool.map(lambda m: gefs_member(self.run, h, m), GEFS_MEMBERS)):
                geo = geo or g["t2"]
                t2 = k_to_f(g["t2"].values)
                w = np.hypot(g["u10"].values, g["v10"].values) * 2.23694
                z = g["z500"].values / 10
                if n == 0:
                    freeze, wind = np.zeros_like(t2), np.zeros_like(t2)
                    zsum, zsq = np.zeros_like(t2), np.zeros_like(t2)
                    rain_half, rain_one = np.zeros_like(t2), np.zeros_like(t2)
                freeze += t2 < 32
                wind += w > 25
                zsum += z
                zsq += z * z
                n += 1
                tp6 = g["tp6"].values.astype(np.float32) if "tp6" in g else None
                last = (self.recent.get(m) or [])[-3:] + [tp6]
                self.recent[m] = last
                if h >= 24 and len(last) == 4 and all(x is not None for x in last):
                    r24 = sum(last) / 25.4  # mm -> in
                    rain_half += r24 >= 0.5
                    rain_one += r24 >= 1.0
                    n_rain += 1
        ar = area("na")
        on_map = lambda a: Grid(a, geo.lat0, geo.dlat, geo.lon0, geo.dlon).sample(ar)  # noqa: E731
        mean = zsum / n
        out = {
            "freeze": on_map(freeze * 100 / n), "wind25": on_map(wind * 100 / n),
            "z500": on_map(mean), "spread": on_map(np.sqrt(np.clip(zsq / n - mean * mean, 0, None)) * 10),  # dam -> m
        }
        if n_rain >= n * 0.8:
            out["rain24half"] = on_map(rain_half * 100 / n_rain)
            out["rain24one"] = on_map(rain_one * 100 / n_rain)
        return out


def draw_gefs(f, h):
    ar = area("na")
    out = {
        "freeze": picture(ar, fill(f["freeze"], "prob")),
        "wind25": picture(ar, fill(f["wind25"], "prob")),
        "spread500": picture(ar, colorize(smooth(f["spread"], 2), "spread"),
                             [(smooth(f["z500"], 3), np.arange(480, 612, 6), "#111827", 1.0, True, False)]),
    }
    for k in ("rain24half", "rain24one"):  # blank for the first day
        out[k] = picture(ar, fill(f.get(k), "prob"))
    return out


_gefs_runs = {}


def gefs_fields(run, h):
    return _gefs_runs.setdefault(run, GefsRun(run)).fields(h)


# NBM: the National Blend of Models, the starting point for the National Weather
# Service's own forecasts. Lower 48 at 2.5 km; drawn every 6 hours to 11 days.
NBM_HOURS = list(range(6, 265, 6))


def nbm_url(run, h):
    return f"{NBM}/blend.{run[:8]}/{run[8:]}/core/blend.t{run[8:]}z.core.f{h:03d}.co.grib2"


def nbm_fields(run, h):
    w6, now = f"{hour_window(h, 6)} acc fcst", f"{h} hour fcst"
    want = {
        "t2": lambda v, lv, f, r: v == "TMP" and lv == "2 m above ground" and f == now and r == "",
        "gust": lambda v, lv, f, r: v == "GUST" and lv == "10 m above ground" and f == now and r == "",
        "sky": lambda v, lv, f, r: v == "TCDC" and lv == "surface" and f == now and r == "",
        "w80": lambda v, lv, f, r: v == "WIND" and lv == "80 m above ground" and f == now and r == "",
        "tp6": lambda v, lv, f, r: v == "APCP" and lv == "surface" and f == w6 and r == "",
        "pop6": lambda v, lv, f, r: v == "APCP" and lv == "surface" and f == w6 and r.startswith("prob >0.254"),
        "tstm": lambda v, lv, f, r: v == "TSTM" and f == w6,
        "snow6": lambda v, lv, f, r: v == "ASNOW" and lv == "surface" and f == w6 and r == "",
        "ice6": lambda v, lv, f, r: v == "FICEAC" and lv == "surface" and f == w6 and r == "",
    }
    spans = idx_spans(get(nbm_url(run, h) + ".idx").decode(), want)
    raw = ranges_get(nbm_url(run, h), spans)
    ar = area("conus")
    g = {k: decode(v).sample(ar) for k, v in raw.items()}
    mph = lambda k: g[k] * 2.23694 if k in g else None  # noqa: E731
    return {
        "t2": k_to_f(g["t2"]), "gust": mph("gust"), "w80": mph("w80"), "sky": g.get("sky"),
        "tp6": g["tp6"] / 25.4 if "tp6" in g else None, "pop6": g.get("pop6"), "tstm": g.get("tstm"),
        "snow6": g["snow6"] * 39.37 if "snow6" in g else None,  # m -> in
        "ice6": g["ice6"] / 25.4 if "ice6" in g else None,  # kg/m2 = mm of ice -> in
    }


def draw_nbm(f, h):
    ar = area("conus")
    return {
        "temp": temp_pic(ar, f["t2"]),
        "pop6": picture(ar, fill(f["pop6"], "chance")),
        "tstm": picture(ar, fill(f["tstm"], "chance")),
        "rain6": picture(ar, fill(f["tp6"], "rain6")),
        "snow6": picture(ar, fill(f["snow6"], "snow6")),
        "ice6": picture(ar, fill(f["ice6"], "ice")),
        "gust": picture(ar, fill(f["gust"], "gust")),
        "sky": picture(ar, fill(f["sky"], "sky")),
        "wind80": picture(ar, fill(f["w80"], "wind80")),
    }


# The AI models: NOAA's AIGFS (built on Google DeepMind's GraphCast, run and
# published by NOAA) and ECMWF's AIFS. Both every 6 hours; 00 and 12 UTC runs.
AI_HOURS = list(range(0, 361, 6))
AI_MAPS = ["temp", "precip", "wind", "upper", "dew", "qpf", "mslp", "wind100"]


def aigfs_url(run, h, part):
    return f"{AIGFS}/aigfs.{run[:8]}/{run[8:]}/model/atmos/grib2/aigfs.t{run[8:]}z.{part}.f{h:03d}.grib2"


def aigfs_fields(run, h):
    total = (f"0-{h} hour acc fcst", f"0-{h // 24} day acc fcst")
    sfc = {
        "t2": lambda v, lv, f, r: v == "TMP" and lv == "2 m above ground",
        "d2": lambda v, lv, f, r: v == "DPT" and lv == "2 m above ground",
        "msl": lambda v, lv, f, r: v == "PRMSL",
        "u10": lambda v, lv, f, r: v == "UGRD" and lv == "10 m above ground",
        "v10": lambda v, lv, f, r: v == "VGRD" and lv == "10 m above ground",
        "u100": lambda v, lv, f, r: v == "UGRD" and lv == "100 m above ground",
        "v100": lambda v, lv, f, r: v == "VGRD" and lv == "100 m above ground",
        "tp6": lambda v, lv, f, r: h >= 6 and v == "APCP" and f == f"{hour_window(h, 6)} acc fcst",
        "tpt": lambda v, lv, f, r: h >= 6 and v == "APCP" and f in total,
    }
    pres = {
        "z500": lambda v, lv, f, r: v == "HGT" and lv == "500 mb",
        "u250": lambda v, lv, f, r: v == "UGRD" and lv == "250 mb",
        "v250": lambda v, lv, f, r: v == "VGRD" and lv == "250 mb",
    }
    ar = area("na")
    g = {}
    for part, want in (("sfc", sfc), ("pres", pres)):
        url = aigfs_url(run, h, part)
        spans = idx_spans(get(url + ".idx").decode(), want)
        for k, v in ranges_get(url, spans).items():
            g[k] = decode(v).sample(ar)
    return ai_out(g, g["tp6"] / 25.4 if "tp6" in g else None, g["tpt"] / 25.4 if "tpt" in g else None)


AIFS_WANT = {("2t", None): "t2", ("2d", None): "d2", ("msl", None): "msl", ("10u", None): "u10", ("10v", None): "v10",
             ("100u", None): "u100", ("100v", None): "v100", ("gh", "500"): "z500", ("u", "250"): "u250",
             ("v", "250"): "v250", ("sf", None): "sf"}


def aifs_fields(run, h):
    g = ec_grids(run, h, "aifs-single", AIFS_WANT)
    out = ai_out(g, ec_tp6(run, h, "aifs-single"), ec_tp(run, h, "aifs-single") if h >= 6 else None)
    # Snowfall comes as kg/m2 (mm) of melted water; 10 inches of snow per inch of water.
    out["snow"] = g["sf"] / 25.4 * 10 if h >= 6 and "sf" in g else None
    return out


def ai_out(g, tp6, qpf):
    """Map fields shared by the two AI models (same units as the GFS maps)."""
    out = {
        "t2": k_to_f(g["t2"]), "msl": g["msl"] / 100, "w10": np.hypot(g["u10"], g["v10"]) * 2.23694,
        "z500": g["z500"] / 10, "w250": np.hypot(g["u250"], g["v250"]) * 2.23694, "tp6": tp6, "qpf": qpf,
        "d2": k_to_f(g["d2"]) if "d2" in g else None,
        "u10kt": g["u10"] * KT, "v10kt": g["v10"] * KT,
    }
    if "u100" in g and "v100" in g:
        out["w100"] = np.hypot(g["u100"], g["v100"]) * 2.23694
    return out


def draw_ai(f, h):
    ar = area("na")
    out = {
        "temp": temp_pic(ar, f["t2"]),
        "wind": picture(ar, colorize(f["w10"], "wind")),
        "upper": picture(ar, colorize(f["w250"], "upper"), [(smooth(f["z500"], 3), np.arange(480, 612, 6), "#111827", 1.0, True, False)]),
        "mslp": mslp_pic(ar, f["msl"], f["u10kt"], f["v10kt"], 6),
        "precip": rain_pic(ar, f.get("tp6"), "precip", f["msl"]),
        "dew": picture(ar, fill(f.get("d2"), "dew")),
        "qpf": picture(ar, fill(f.get("qpf"), "qpf")),
        "wind100": picture(ar, fill(f.get("w100"), "wind100")),
    }
    if "snow" in f:  # ECMWF's AIFS has snowfall; NOAA's AIGFS doesn't
        out["snow"] = picture(ar, fill(f["snow"], "snow"))
    return out


# NAM 3 km and the HiResW (HRW) models: NOAA is retiring them (with HREF) once
# RRFS and REFS take over. They stay on the page until NOAA stops making them;
# after SUNSET_DAYS with no new run, the job takes them off the page.
NAM = "https://noaa-nam-pds.s3.amazonaws.com"
HIRESW = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod"
SUNSET = {"href", "nam", "hrwarw", "hrwarw2", "hrwfv3"}
SUNSET_DAYS = 3


def nam_url(run, h):
    return f"{NAM}/nam.{run[:8]}/nam.t{run[8:]}z.conusnest.hiresf{h:02d}.tm00.grib2"


def hiresw_url(kind):
    """kind: arw, arw2 (ARW member 2) or fv3, lower-48 files on NOMADS (2.5 km; ARW
    member 2 only comes at 5 km)."""
    core, tail = {"arw": ("arw_2p5km", "conus"), "arw2": ("arw_5km", "conusmem2"), "fv3": ("fv3_2p5km", "conus")}[kind]

    def url(run, h):
        return f"{HIRESW}/hiresw.{run[:8]}/hiresw.t{run[8:]}z.{core}.f{h:02d}.{tail}.grib2"
    return url


def storm_model_fields(url):
    """Common maps from a single 3 km-class storm model (NAM 3 km, HiResW). Fields a
    model doesn't have come out blank."""
    def fields(run, h):
        win, now = hour_window(h, 1), (f"{h} hour fcst", "anl")
        total = (f"0-{h} hour acc fcst", f"0-{h // 24} day acc fcst")
        # HiResW files also hold the two hours before (e.g. 2 m temperature at 11 and
        # 10 hours in the 12-hour file), so instant fields must match the hour.
        want = {
            "refc": lambda v, lv, f, r: v == "REFC" and lv.startswith("entire atmosphere") and f in now,
            "t2": lambda v, lv, f, r: v == "TMP" and lv == "2 m above ground" and f in now,
            "d2": lambda v, lv, f, r: v == "DPT" and lv == "2 m above ground" and f in now,
            "gust": lambda v, lv, f, r: v == "GUST" and lv == "surface" and f in now,
            "msl": lambda v, lv, f, r: v in ("MSLET", "PRMSL") and lv == "mean sea level" and f in now,
            "u10": lambda v, lv, f, r: v == "UGRD" and lv == "10 m above ground" and f in now,
            "v10": lambda v, lv, f, r: v == "VGRD" and lv == "10 m above ground" and f in now,
            "cape": lambda v, lv, f, r: v == "CAPE" and lv == "surface" and f in now,
            "uh": lambda v, lv, f, r: h >= 1 and v == "MXUPHL" and lv == "5000-2000 m above ground" and f == f"{win} max fcst",
            "tpt": lambda v, lv, f, r: h >= 1 and v == "APCP" and f in total,
        }
        u = url(run, h)
        text = get(u + ".idx").decode()
        spans = idx_spans(text, want)
        if "t2" not in spans:
            print(f"  {u.rsplit('/', 1)[-1]}: no 2 m temperature; fields: "
                  + "; ".join(":".join(x.split(":")[3:6]) for x in text.splitlines()[:40]), flush=True)
        grids = {k: decode(v) for k, v in unpack_parts(ranges_get(u, spans), spans).items()}
        ar = area("conus")
        winds = {}
        if "u10" in grids and "v10" in grids:
            uu, vv = earth_wind(grids.pop("u10"), grids.pop("v10"), ar)
            winds["u10"] = (uu * KT, vv * KT)
        g = {k: v.sample(ar) for k, v in grids.items()}
        return {
            "refc": g.get("refc"), "t2": k_to_f(g["t2"]) if "t2" in g else None, "d2": k_to_f(g["d2"]) if "d2" in g else None,
            "gust": g["gust"] * 2.23694 if "gust" in g else None, "msl": g["msl"] / 100 if "msl" in g else None,
            "cape": g.get("cape"), "uh": g.get("uh"), "qpf": g["tpt"] / 25.4 if "tpt" in g else None, "winds": winds,
        }
    return fields


def draw_storm_model(f, h):
    ar = area("conus")
    out = {
        "radar": picture(ar, fill(f["refc"], "radar"), sharp=True),
        "temp": temp_pic(ar, f["t2"]) if f["t2"] is not None else picture(ar, None),
        "dew": picture(ar, fill(f["d2"], "dew")),
        "gust": picture(ar, fill(f["gust"], "gust")),
        "cape": picture(ar, fill(f["cape"], "cape")),
        "uh": picture(ar, fill(f["uh"], "uh"), sharp=True),
        "qpf": picture(ar, fill(f["qpf"], "qpf")),
    }
    if f["msl"] is not None and "u10" in f["winds"]:
        out["mslp"] = mslp_pic(ar, f["msl"], *f["winds"]["u10"], 16)
    else:
        out["mslp"] = picture(ar, None)
    if f["qpf"] is None:
        del out["qpf"]
    return out


STORM_MAPS = ["radar", "temp", "dew", "gust", "cape", "uh", "qpf", "mslp"]


# ---- drawing ------------------------------------------------------------------

def picture(ar, fill_rgba, contours=None, sharp=False, barbs=None):
    """RGBA image with optional contour lines [(field, levels, color, width, labels, dashed)]
    and wind barbs (u_kt, v_kt, spacing_px, color)."""
    width, height = ar.width, ar.height
    if not contours and barbs is None and fill_rgba is not None and fill_rgba.shape[:2] == (height, width):
        # No lines to draw: skip matplotlib (much faster, same picture).
        out = io.BytesIO()
        Image.fromarray(fill_rgba, "RGBA").save(out, format="WEBP", quality=80, method=4)
        return out.getvalue()
    if fill_rgba is None and not contours and barbs is None:
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
    if barbs is not None:
        bu, bv, step, color = barbs
        o = step // 2
        bu, bv = bu[o::step, o::step], bv[o::step, o::step]
        bx, by = np.meshgrid(xs[o::step], ys[o::step])
        ok = np.isfinite(bu) & np.isfinite(bv)
        if ok.any():
            # Barbs point into the wind: the staff runs toward where the wind comes from.
            ax.barbs(bx[ok], by[ok], bu[ok], bv[ok], length=6.2, linewidth=0.75, color=color,
                     sizes={"emptybarb": 0.12, "spacing": 0.14, "height": 0.38, "width": 0.25})
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
    out["mslp"] = mslp_pic(ar, fields["msl"], fields.get("u10kt"), fields.get("v10kt"), 6)
    if "shr_u" in fields:
        out["shear"] = shear_pic(ar, fields["shr_u"], fields["shr_v"], "shear5")
    out.update(draw_upper(ar, fields))
    return out


BARB_GAP = 46  # pixels between wind barbs


def mslp_pic(ar, msl, u, v, n):
    lines = [(smooth(msl, n), np.arange(940, 1068, 2), "#334155", 0.8, True, False)]
    return picture(ar, colorize(smooth(msl, n), "mslp"), lines, barbs=(u, v, BARB_GAP, "#1f2937") if u is not None else None)


def shear_pic(ar, u, v, legend):
    return picture(ar, colorize(np.hypot(u, v), legend), barbs=(u, v, BARB_GAP, "#111827"))


def jet_pic(ar, u, v, legend, z=None, zlevels=None):
    """Jet stream: wind speed colors (mph), barbs (knots) and upper-level divergence lines."""
    # The 3 km HRRR is noisy aloft; smooth to about 100 km so only the jet-scale pattern shows.
    su, sv = smooth(smooth(u, 24), 24), smooth(smooth(v, 24), 24)
    div, _ = kinematics(su / KT, sv / KT, ar)
    lines = [(smooth(div * 1e5, 10), [2, 3, 4, 6, 8], "#7e22ce", 1.0, True, True)]
    if z is not None:
        lines.append((smooth(z, 3), zlevels, "#111827", 1.0, True, False))
    return picture(ar, colorize(np.hypot(u, v) * 1.15078, legend), lines, barbs=(u, v, BARB_GAP, "#111827"))


def draw_hrrr_upper(ar, f):
    """HRRR jet stream (250, 300 mb) and 500, 700, 850 mb maps, with barbs."""
    w = f.get("winds") or {}
    out = {}
    black, bar = "#111827", lambda k: (w[k][0], w[k][1], BARB_GAP, black) if k in w else None
    if "u250" in w:
        out["jet250"] = jet_pic(ar, *w["u250"], "jet250")
    if "u300" in w:
        out["jet300"] = jet_pic(ar, *w["u300"], "jet300d")
    if "u500" in w and f.get("z500") is not None:
        u, v = w["u500"]
        _, vort = kinematics(smooth(smooth(u, 8), 8) / KT, smooth(smooth(v, 8), 8) / KT, ar)
        av = (vort + coriolis(ar)) * 1e5
        out["h500"] = picture(ar, colorize(smooth(av, 3), "vort"), [(smooth(f["z500"], 4), np.arange(480, 612, 6), black, 1.0, True, False)], barbs=bar("u500"))
    if f.get("rh700") is not None and f.get("z700") is not None:
        out["h700"] = picture(ar, colorize(f["rh700"], "rh700"), [(smooth(f["z700"], 4), np.arange(240, 336, 3), black, 0.9, True, False)], barbs=bar("u700"))
    if f.get("t850") is not None and f.get("z850") is not None:
        out["h850"] = picture(ar, colorize(f["t850"], "t850"), [(smooth(f["t850"], 3), [0], "#1d4ed8", 1.6, False, True),
                                                             (smooth(f["z850"], 4), np.arange(105, 171, 3), black, 0.9, True, False)], barbs=bar("u850"))
    return out


def draw_upper(ar, f):
    """The 200, 300, 500, 700 and 850 mb maps (empty if the model run lacks them)."""
    if "z200" not in f:
        return {}
    black = "#111827"
    return {
        "h200": picture(ar, colorize(f["w200"], "jet200"), [(smooth(f["z200"], 3), np.arange(1080, 1290, 12), black, 1.0, True, False)]),
        "h300": picture(ar, colorize(f["w300"], "jet300"), [(smooth(f["z300"], 3), np.arange(840, 1000, 12), black, 1.0, True, False)]),
        "h500": picture(ar, colorize(smooth(f["av500"], 2), "vort"), [(smooth(f["z500"], 3), np.arange(480, 612, 6), black, 1.0, True, False)]),
        "h700": picture(ar, colorize(f["rh700"], "rh700"), [(smooth(f["z700"], 3), np.arange(240, 336, 3), black, 0.9, True, False)]),
        "h850": picture(ar, colorize(f["t850"], "t850"), [(smooth(f["t850"], 3), [0], "#1d4ed8", 1.6, False, True),
                                                       (smooth(f["z850"], 3), np.arange(105, 171, 3), black, 0.9, True, False)]),
    }


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
        **{k: picture(ar, fill(v, k)) for k, v in (f.get("meso") or {}).items()},
        **draw_hrrr_extra(ar, f),
    }


def draw_hrrr_extra(ar, f):
    """Bulk shear with barbs, sea-level pressure, 3-hour changes and the upper air (HRRR)."""
    w = f.get("winds") or {}
    out = {}
    if "shu" in w:
        out["shear6"] = shear_pic(ar, *w["shu"], "shear6")
    if "sh1u" in w:
        out["shear1"] = shear_pic(ar, *w["sh1u"], "shear1")
    if "u10" in w:
        out["mslp"] = mslp_pic(ar, f["msl"], *w["u10"], 16)
    for k, v in (f.get("change") or {}).items():
        out[k] = picture(ar, colorize(smooth(v, 4), k))
    out.update(draw_hrrr_upper(ar, f))
    return out


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


def retire(model, manifest):
    """Take a model NOAA no longer makes off the website's map list (and delete its pictures)."""
    part = {k: v for k, v in manifest.items() if k != "models"}
    part.update({"retired": [model], "order": list(MODELS)})
    print(send("POST", "/weather/models/upload/manifest-part", json.dumps(part).encode(), "application/json"))


def publish(model, manifest):
    """Add one model's run to the website's map list. Other model groups run as
    separate jobs at the same time, so the website merges this one model into its
    own copy of the list."""
    part = {k: v for k, v in manifest.items() if k != "models"}
    part.update({"model": model, "entry": manifest["models"][model], "retired": sorted(RETIRED), "order": list(MODELS)})
    # A website without manifest-part answers 404. Then the map list is left alone
    # (the pictures are uploaded and show once the website is updated): sending the
    # whole list from parallel jobs could put back older runs and delete newer ones.
    print(send("POST", "/weather/models/upload/manifest-part", json.dumps(part).encode(), "application/json"))


# ---- main ---------------------------------------------------------------------

UPPER_MAPS = ["h200", "h300", "h500", "h700", "h850"]
HRRR_MESO = ["mlcape", "cin", "shear6", "shear1", "srh1", "lapse", "pwat", "stp", "scp", "mslp", "t3h", "td3h", "cape3h",
             "jet250", "jet300", "h500", "h700", "h850"]

MODELS = {
    "gfs": {"name": "GFS", "latest": gfs_latest, "fields": gfs_fields, "draw": draw, "area": "na",
            "hours": HOURS, "maps": FIELDS + ["dew", "cape", "qpf", "mslp", "shear"] + UPPER_MAPS},
    "ecmwf": {"name": "European", "latest": ec_latest, "fields": ec_fields, "draw": draw, "area": "na",
              "hours": HOURS, "maps": FIELDS + ["dew", "cape", "qpf", "snow", "mslp", "shear"] + UPPER_MAPS},
    "aigfs": {"name": "AI GFS", "latest": short_latest(lambda r, h: aigfs_url(r, h, "sfc"), AI_HOURS[-1], every=12),
              "fields": aigfs_fields, "draw": draw_ai, "area": "na", "hours": AI_HOURS,
              "maps": AI_MAPS},
    "aifs": {"name": "European AI", "latest": lambda now: ec_latest(now, "aifs-single", AI_HOURS[-1]),
             "fields": aifs_fields, "draw": draw_ai, "area": "na", "hours": AI_HOURS, "maps": AI_MAPS + ["snow"]},
    "gefs": {"name": "GEFS chances", "latest": gefs_latest, "fields": gefs_fields, "draw": draw_gefs,
             "area": "na", "hours": GEFS_HOURS, "workers": 1,
             "maps": ["rain24half", "rain24one", "freeze", "wind25", "spread500"]},
    "nbm": {"name": "NBM blend", "latest": short_latest(nbm_url, NBM_HOURS[-1], every=12), "fields": nbm_fields,
            "draw": draw_nbm, "area": "conus", "hours": NBM_HOURS,
            "maps": ["temp", "pop6", "tstm", "rain6", "snow6", "ice6", "gust", "sky", "wind80"]},
    "hrrr": {"name": "HRRR", "latest": short_latest(hrrr_url, 48), "fields": hrrr_fields, "draw": draw_hrrr,
             "area": "conus", "hours": list(range(0, 49)), "maps": ["radar", "temp", "rain1", "gust", "uh", "dew", "cape", "qpf", "snow", "hail", "ltng"] + HRRR_MESO,
             "points": True},
    "rrfs": {"name": "RRFS", "latest": short_latest(rrfs_url, RRFS_HOURS[-1], every=12), "fields": rrfs_fields,
             "draw": draw_rrfs, "area": "conus", "hours": RRFS_HOURS,
             "maps": ["radar", "temp", "rain1", "gust", "uh", "dew", "cape", "qpf", "snow", "hail", "smoke", "ptype", "mslp"]},
    "refs": {"name": "REFS chances", "latest": short_latest(refs_url, 60, every=12), "fields": refs_fields,
             "draw": draw_refs, "area": "conus", "hours": list(range(1, 61)),
             "maps": ["radar", "temp", "rain1", "storms40", "rotation", "heavy", "freeze", "pfrzra", "wind34", "fog"]},
    # Being retired by NOAA (see SUNSET): shown until NOAA stops making them.
    "href": {"name": "HREF", "latest": short_latest(href_url, 48, every=12), "fields": href_fields, "draw": draw_href,
             "area": "conus", "hours": list(range(1, 49)), "maps": ["radar", "temp", "rain1", "storms40", "rotation", "heavy"]},
    "nam": {"name": "NAM 3 km", "latest": short_latest(nam_url, 60, every=12), "fields": storm_model_fields(nam_url),
            "draw": draw_storm_model, "area": "conus", "hours": list(range(0, 61)),
            "maps": [m for m in STORM_MAPS if m != "qpf"]},  # NAM rain comes in 3-hour pieces, not run totals
    **{f"hrw{k}": {"name": name, "latest": short_latest(hiresw_url(k), 48, every=12), "fields": storm_model_fields(hiresw_url(k)),
                   "draw": draw_storm_model, "area": "conus", "hours": list(range(0, 49)), "maps": STORM_MAPS}
       for k, name in (("arw", "HRW ARW"), ("arw2", "HRW ARW2"), ("fv3", "HRW FV3"))},
}
DEFAULT_MODELS = list(MODELS)
# Taken off the page by hand: dropped from the map list the next time any model uploads.
RETIRED = set()


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
    # GEFS adds up 24-hour rain from the hours before, so it draws one hour at a time
    # (each hour downloads its 31 members side by side).
    with cf.ThreadPoolExecutor(max_workers=spec.get("workers", 4)) as pool:
        for h in pool.map(one, hours):
            done.append(h)
            print(f"  {model} {run} +{h}h", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="also save pictures into this folder")
    ap.add_argument("--upload", action="store_true", help="upload to the website")
    ap.add_argument("--hours", help="comma list for a quick test, e.g. 0,6,12")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
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
                "credit": "NOAA NCEP GFS, AIGFS, GEFS, NBM, HRRR, RRFS, REFS, HREF, NAM and HiResW; ECMWF open data (IFS and AIFS), CC BY 4.0",
                "models": dict(current.get("models") or {})}
    order = {k: i for i, k in enumerate(MODELS)}
    failed = []
    for model in a.models.split(","):
        spec = MODELS[model]
        try:
            run = spec["latest"](now)
            if not run:
                print(f"{model}: no complete run found")
                last = (manifest["models"].get(model) or {}).get("run")
                if model in SUNSET and last and now - dt.datetime.strptime(last, "%Y%m%d%H").replace(
                        tzinfo=dt.timezone.utc) > dt.timedelta(days=SUNSET_DAYS):
                    # NOAA has stopped this model: take it off the page.
                    print(f"{model}: no new run since {last}; taking it off the page")
                    manifest["models"].pop(model, None)
                    if a.upload:
                        retire(model, manifest)
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
        for gone in RETIRED:
            manifest["models"].pop(gone, None)
        manifest["models"] = dict(sorted(manifest["models"].items(), key=lambda kv: order.get(kv[0], 99)))
        # Publish after each model so a slow or failed model doesn't hold the others back.
        if a.upload:
            publish(model, manifest)
    if a.out:
        with open(os.path.join(a.out, "manifest.json"), "w") as f:
            json.dump(manifest, f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
