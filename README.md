# AtmosDays model maps

Draws the forecast model maps shown at
https://atmosdays.eagleeyeinsightsllc.com/weather/models: NOAA GFS, HRRR and
HREF, and the ECMWF (European) open-data model. All the data is free and public.

`render_models.py` finds the newest complete runs, downloads only the fields
it needs, draws WebP maps and uploads them to the website. For HRRR it also
saves wind at 10 m (33 ft), wind at 80 m (about 260 ft, crane and lift height)
and gusts on a 0.1 degree grid over the lower 48 (`hrrr/<run>/wind/points.bin`,
one byte of mph per field and hour, stored spot by spot), which the website
reads for each job.

For HRRR (2 days) and GFS (10 days) it also saves a sounding grid
(`<model>/<run>/snd/<hour>.bin`, one file per forecast hour): for every spot,
the surface values and the height, temperature, dew point and wind at each
pressure level, one byte each, stored spot by spot. The website's
click-anywhere soundings read one spot with a single small read. HRRR uses a
0.125 degree grid (about 15 MB an hour), GFS 0.25 degree (about 6.5 MB).

The scheduled job in
`.github/workflows/model-maps.yml` checks every hour (GitHub sometimes skips a
scheduled run) and only draws when a model has a new run.

The website's own code is private; only this map-drawing job lives here, so it
can use GitHub's free minutes for public repositories.

ECMWF maps are drawn from ECMWF open data, licensed CC BY 4.0
(Copyright ECMWF). NOAA data is public domain.

Run by hand: Actions → Weather model maps → Run workflow (tick "test_only" to
draw a few hours without uploading).
