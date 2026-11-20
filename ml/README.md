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
| What really happened: hourly airport weather (ISD) | AWS `noaa-global-hourly-pds` | same years |
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

## What it doesn't do

It doesn't predict tornadoes, hail or hurricanes. It corrects temperature, dew
point and wind at a place; on stormy days the official warnings come first.
