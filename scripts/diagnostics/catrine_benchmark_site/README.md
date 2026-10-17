# CATRINE v0.5 benchmark site

Static web site comparing the two AtmosTransport v0.5 CATRINE protocol runs (ERA5 L117 and
MERRA-2 L72, C90, 2021-12 to 2023-12) with GCHP C90. Public location:
`~/www/catrine/benchmark_v0.5/` (https://web.gps.caltech.edu/~cfranken/catrine/benchmark_v0.5/).

## Rebuild and publish

```bash
# build and copy to the public folder (MERRA-2 now; ERA5 is added automatically once all 25 of its
# months are compared)
scripts/diagnostics/catrine_benchmark_site/publish.sh

# show ERA5 before its run is complete
scripts/diagnostics/catrine_benchmark_site/publish.sh --datasets merra2,era5

# build into the staging folder only (nothing public changes)
scripts/diagnostics/catrine_benchmark_site/publish.sh --build-only
```

`publish.sh` runs `../catrine_benchmark_v05_site.py --out $STAGING` (default
`/temp1/cfranken/plan/w3/signoff/site_staging`) and then `rsync --delete`s the build to `$PUBLIC`
(default `~/www/catrine/benchmark_v0.5`). It refuses to write into a non-empty folder that has no
`.catrine_benchmark_site` marker. A rebuild takes seconds; the first build also reads four GCHP files
per month (about 1 s each) for the tropopause, cached in `$STAGING/_cache/` (not published).

`--datasets` (passed through by `publish.sh`): `auto` (default) = MERRA-2, plus ERA5 once all 25 months
are compared; or `merra2`, `era5`, `merra2,era5`. A run that is left out leaves no trace on the pages
(no empty panels, columns or text about it). The status line under the header reads
"<run> run: data through <last compared month> · updated <build time>".

The generator alone: `python3 scripts/diagnostics/catrine_benchmark_v05_site.py --out <folder>`;
`--help` lists the input paths and run labels.

## Inputs (read only)

| Input | Used for |
|---|---|
| `<signoff-dir>/compare/<label>/<season>/` with `DONE`: `summary.json`, `timeseries.npz`, `monthly_fields_c90.npz` (from `catrine_compare_vs_geoschem.py`) | every statistic, monthly field, map and zonal section |
| `<signoff-dir>/SIGNOFF_TABLE.json` (`signoff_summary.py`) | verdict, criteria, drift, mass (C9) and dry-air (C10) numbers |
| `<signoff-dir>/massbal/<label>/<YYYY-MM>.csv` (`catrine_true_mass_balance.jl`) | conservation residual curves |
| GCHP `CATRINE_inst` files (one for the grid; four per month for the tropopause) and GCHP's mass-balance csv | grid, areas, tropopause, GCHP residual |
| run output folders | the transported days, and the station columns (below) |
| GCHP `CATRINE_inst` files at the station cells | station time series (GCHP) and the dry pressure at each site's height |
| ObsPack `obspack_co2_1_OCO2MIP_v5.0.1_2024-11-14` (GLOBALVIEWplus v10.1 records, `--obspack-dir`) | site coordinates, elevation and intake height; NOAA CO2 observations (CC0 datasets only, checked per dataset from `dataset_provider_license`) |

Run labels: ERA5 `era5_protic`, MERRA-2 `m2_protic_fix` (the OF-108 rerun), identical-start MERRA-2
`m2_gcic_fix` (supporting; once it has compared months it appears in one collapsed panel on the
stratosphere page and in the Methods run table, and its signoff criterion C6a is listed with MERRA-2). The
superseded pre-fix labels are never read.

## Station time series

`stations.html` shows eight NOAA sites (`STATIONS` in the generator: MLO, SPO, BRW, ALT, MHD, LEF, SMO,
CGO). Each model is sampled in the nearest C90 cell: surface sites in the lowest layer, MLO at its
altitude and LEF at its intake height above the model surface, where GCHP's `Height_asl` and
`Met_PMIDDRY` give the dry pressure and each run is interpolated in log dry pressure on its own column.
Only the needed columns are read (partial NetCDF reads, `--station-workers` processes, default 8; the
files sit on NFS, so the first extraction of a month takes about a minute per month and dataset). The
extracted series are cached per month and dataset in `$STAGING/_cache/stations/` (keyed by the site
definitions), so rebuilds only read new months. Observations: daily means of the in situ hours with
`obs_flag = 1`, or flask samples with `obs_flag = 1`; datasets whose provider licence is not CC0 are
skipped and listed on the page.

## What updates automatically

Everything numeric. Only seasons with a `DONE` marker are read, so a rebuild while the runs continue
shows the months compared so far (time axes span the months on the site), and the status line shows
the last compared month per run. The "common months" table, the period means of the maps and sections, the
key points on the overview and the drift tiles follow the data. The identical-start panel fills in
when `compare/m2_gcic_fix/` has a finished season. The signoff criteria are read from `SIGNOFF_TABLE.json`
as they are: criteria that are only REPORTED for the runs shown (C5c, C6c, and the ERA5 C8c once the
summary writes the C8a/C8b/C8c split) go to the "reported, not counted" table, and the stratosphere page
switches its C8 text to the split when the table carries it. Hand-written text (Methods, the explanations on the
near-surface and stratosphere pages, the GCHP-side deviations) states facts from the signoff reports
and must be edited by hand if those change.

## Layout

| File | Contents |
|---|---|
| `../catrine_benchmark_v05_site.py` | reads the products, writes `data/*.js`, copies the assets, renders the pages |
| `layout.html` | page frame: header, navigation, status strip, footer |
| `pages/*.html` | page bodies, each with its own small script |
| `site.js` | shared code: lazy data loading, controls, URL state, D3 charts (line, bar, heatmap), map and section grids |
| `site.css` | styles |
| `publish.sh` | build + copy |

Output: `index.html` and seven pages, `assets/` (site.css, site.js, d3.min.js 7.9.0 checksum-verified
and its ISC licence `d3.LICENSE.txt`; the generator downloads D3 once from jsDelivr, or takes `--d3-js <file>`,
and removes any other file from `assets/`), `data/` (`meta.js`, `overview.js`, `monthly.js`, `daily_<tracer>.js`,
`zonal_<tracer>.js`, `mass.js`, `coast.js`, `maps/<tracer>_<band>_<month>.js`). Data files are JSON
wrapped in `SITE.put(key, value)` and loaded with `<script>` tags, so the pages also work from disk.
Maps are 2° means of the C90 fields stored as int16 codes; each page loads only what it shows.

## Design rules

- One colour per dataset everywhere: GCHP near-black, ERA5 blue `#2a78d6`, MERRA-2 orange `#eb6834`
  (the identical-start MERRA-2 run: orange, dotted). In difference plots GCHP is the zero line.
- Text about one run sits in an element with `data-run="<key>"`, text that needs several runs in
  `data-runs="<key> <key>"`; `site.js` removes it unless those runs are on the site. Panel grids, tables,
  legends and key points are built from the runs present.
- Differences use one diverging scale (blue–grey–red, zero = neutral) shared by the difference panels
  of a view; absolute fields use a yellow–green–blue sequential scale.
- One control row per page, kept in the URL hash (views can be linked); the legend toggles datasets.
- Chemical formulas and exponents use `<sub>`/`<sup>`, not Unicode sub/superscripts (missing in some
  fonts); chart axis titles turn them into SVG tspans.
- Charts are D3 (no other library): line and bar charts in SVG, heatmaps as canvas cells (one rectangle
  per cell at device resolution) under an SVG layer with axes, coastlines and the tropopause. Hover with
  the mouse, or tap and slide sideways on a touch screen, for every series at the nearest point; a tap
  elsewhere closes the tooltip. Charts never capture vertical page scrolling (`touch-action: pan-y`).
  Daily and 3-hourly time axes zoom by dragging with the mouse (touch: press *Zoom* first); *Reset zoom*
  or a double-click restores the period (a tap outside the chart disarms *Zoom*). Charts redraw when
  their container width or the screen's pixel ratio changes. There is no image export (take a screenshot)
  and the maps and sections do not zoom.

## Checking a build

Render the pages headlessly before publishing, for example with Playwright and Chrome
(`--use-gl=disabled` is needed on the GPU servers) at 1400 px and 390 px width, and look at every
page, the browser console, and hover/tap/zoom on a line chart and a map; the scripts used for the checks
are not part of the repository.
