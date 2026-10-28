# AtmosDays model maps

Draws the forecast model maps shown at
https://atmosdays.eagleeyeinsightsllc.com/weather/models: NOAA GFS, HRRR and
HREF, and the ECMWF (European) open-data model. All the data is free and public.

`render_models.py` finds the newest complete runs, downloads only the fields
it needs, draws WebP maps and uploads them to the website. The scheduled job in
`.github/workflows/model-maps.yml` runs four times a day.

The website's own code is private; only this map-drawing job lives here, so it
can use GitHub's free minutes for public repositories.

ECMWF maps are drawn from ECMWF open data, licensed CC BY 4.0
(Copyright ECMWF). NOAA data is public domain.

Run by hand: Actions → Weather model maps → Run workflow (tick "test_only" to
draw a few hours without uploading).
