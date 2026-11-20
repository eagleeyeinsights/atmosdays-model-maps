# Place-learning forecast (machine learning)

The "AI-powered local forecast and supplementary weather model" for AtmosDays
Weather+. It doesn't forecast the weather from scratch. It learns how the GFS
forecast tends to miss (by time of year, time of day, days ahead, location and
terrain), then corrects temperature, dew point and wind. It's a supplement:
people should still check the National Weather Service forecast, the other
models and official warnings.

## Data (all free, all public)

| What | Where | Years |
|---|---|---|
| Old forecasts: GEFS reforecast, control run (same model family as today's GFS) | AWS `noaa-gefs-retrospective` | 2000-2019 |
| Old forecasts: GFS archive | AWS `noaa-gfs-bdp-pds` | 2021 to now |
| What really happened: hourly airport weather (ISD) | AWS `noaa-global-hourly-pds` | 2000-2024 |
| What really happened, newer: GHCN-hourly (replaced ISD in late 2025, updated daily) | AWS `noaa-ghcnh-pds` | 2025 to now |
| Station list | NOAA NCEI `isd-history.csv` | |

Every 5th day, 00Z run, 6 to 168 hours ahead every 6 hours, at about one long-record
US airport station per 1-degree box. Only public stations are used; customer
places never come here.

## Model

LightGBM gradient-boosted trees, one set each for temperature, dew point and wind.
They predict measured minus forecast; corrected = forecast + prediction.
Features: `src, lead, solar_hour, doy_sin, doy_cos, lat, lon, zs_m, t_f, td_f, ws_mph`.

Testing: whole years (2004, 2012, 2019, 2023) are held back and never trained on;
the report compares the raw GFS-family forecast with the corrected one on them,
by days ahead and season. The shipped model is then retrained on all years.

The trees are exported as plain arrays (`ml/local-v1.json`) that the website
walks itself (`contractor-app/worker/localml.js` in the AtmosDays repo), so
running it costs nothing. `train.py` checks the export matches LightGBM.

## Running

`.github/workflows/place-forecast-ml.yml`, daily. Training data lives as assets of
the `ml-data` release. The first runs collect the history (a few hours each,
picking up where the last stopped); after that each week's run rebuilds this
year's GFS chunk with the newest days and retrains, so it keeps learning.
Public repo, so the GitHub minutes are free; NOAA's AWS data is free.

Locally: `python -m ml.run build --data data --max-minutes 30`, then
`python -m ml.run train --data data --out model/local-v1.json`.

## Live scoreboard

Held-back years test the model on the past. The live scoreboard (`ml/live.py`,
last step of the daily run) tests it on forecasts saved before the weather
happens: each day it saves, at the same stations, the raw GFS forecast, the
AtmosDays AI-corrected GFS, and ECMWF's AI model (AIFS, open data, CC BY 4.0).
Once the stations have measured those hours it scores all three on the same
rows: average miss by days ahead, and next-day temperature week by week. Saved
forecasts are `live-YYYY-MM.npz` and measured hours `obs-YYYY.npz` in the
`ml-data` release; the scores go into the report as `live`.

ECMWF's open data only keeps its latest runs, so this also builds an AIFS
archive at the stations for later training (using AIFS as a model input).

The scoreboard carries a `credit` line ("Contains ECMWF AIFS data, © ECMWF,
CC BY 4.0"); anything that shows the AIFS numbers, or a forecast that learned
from them, must show it too.

Owner's rule: Gemini or any other Google AI product must not be used to build,
train or write these forecasts or models.

## What it doesn't do

It doesn't predict tornadoes, hail or hurricanes. It corrects temperature, dew
point and wind at a place; on stormy days the official warnings come first.
