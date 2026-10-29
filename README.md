# AtmosDays model maps

Draws the forecast model maps shown at
https://atmosdays.eagleeyeinsightsllc.com/weather/models: NOAA GFS, AI GFS
(AIGFS), GEFS (31-run ensemble chances), NBM (National Blend of Models), HRRR,
RRFS and REFS, and the ECMWF (European) and ECMWF AI (AIFS) open-data models.
All the data is free and public. NOAA is retiring the HREF (and NAM and
HiResW); REFS and RRFS replace them, and HREF is no longer drawn on the
schedule (`--models href` still works by hand while NOAA has it).

The models run as separate groups side by side (see the workflow), and each
upload merges only its own model into the website's map list.

`render_models.py` finds the newest complete runs, downloads only the fields
it needs, draws WebP maps and uploads them to the website. For HRRR it also
saves wind at 10 m (33 ft), wind at 80 m (about 260 ft, crane and lift height)
and gusts on a 0.1 degree grid over the lower 48 (`hrrr/<run>/wind/points.bin`,
one byte of mph per field and hour, stored spot by spot), which the website
reads for each job.

GFS and ECMWF also get upper-air maps: 200 and 300 mb (jet stream wind and
heights), 500 mb (heights and absolute vorticity; ECMWF's relative vorticity
plus the Earth's spin), 700 mb (humidity and heights) and 850 mb (temperature
with the 0°C line, and heights).

HRRR also gets storm-ingredient maps for the website's mesoanalysis page:
mixed-layer CAPE and CIN, 0-6 km shear (VUCSH/VVCSH), 0-1 km helicity, the
700-500 mb lapse rate, precipitable water, and the significant tornado (STP,
fixed layer, with the HRRR's own LCL height) and supercell composite (SCP,
most-unstable CAPE and 0-3 km helicity) parameters computed from those fields.

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

ECMWF and AIFS maps are drawn from ECMWF open data, licensed CC BY 4.0
(Copyright ECMWF; based on data and products of ECMWF). NOAA data is public domain.
NBM files store every other row right to left; `decode()` turns those rows around.

Run by hand: Actions → Weather model maps → Run workflow (tick "test_only" to
draw a few hours without uploading).

## Hurricane extras (`tropics.py`)

Every 15 minutes (`.github/workflows/tropics.yml`) this writes one small JSON file for
the hurricane page: spaghetti model tracks (NHC ATCF a-deck, newest run of each model),
the storm's 6-hourly best track (b-deck) and Hurricane Hunter observations and center
fixes (NHC recon archive). Standard library only. `python tropics.py --test` prints a
summary; `--recon-sample N` reads the newest N recon files however old (for testing).
