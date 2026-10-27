# Performance benchmark site

Static web site that compares the speed of AtmosTransport releases (v0.5.0, v0.6.0, main) on real
MERRA-2 L72 Float32 cubed-sphere runs across devices (L40S, A100, M5 Pro, a 32-thread CPU). It is built
from the timing records of the real-data harness (not from the synthetic cases of `benchmarking/run_benchmarks.jl`).

## Build

```bash
python3 benchmarking/site/build_site.py --data <data> --out <folder>
```

Python 3 standard library only. A rebuild takes about a second and can be repeated while runs are in
progress: devices, versions, grids, tracer counts and operator sets without a successful run are left
out (no empty axes or columns), and the status line under the header shows the progress per device.
The output folder must be empty or an earlier build (marker file `.atmostr_bench_site`); `assets/` and
`data/` are rewritten on every build. Nothing is published by the generator; copy the folder to the web
server after review.

## Inputs (read only)

| Input | Used for |
|---|---|
| `<data>/<device>/records.csv` | every number on the site; schema below |
| `<data>/<device>/PUBLIC_NOTES.md` (optional) | shown on the Methods page under that device (small Markdown subset); the internal `NOTES.md` run log is never published |

`records.csv` columns: `device, backend, float_type, version, commit, pass, grid, tracers, operators,
sim_days, status, steady_seconds_per_day, warm_seconds_per_day, peak_mem_mib, mean_util_percent,
mean_power_w, load_avg_1m, utc, result_dir`. Device folders: `L40S`, `A100`, `M5Pro`, `CPU-wurst`
(another folder is shown in grey with a warning). Versions `v0.5.0`, `v0.6.0`, `main`; operator sets
`none, adv, diff, conv, adv+emis, all`; other values are ignored.

Per configuration (device, version, grid, tracers, operators) and pass, the latest successful record is
used; a pass with only failed records is listed as failed on the Methods page. The planned matrix is
`PLAN` in `build_site.py`; a device with a reduced matrix gets its own entry in `DEVICE_PLAN` (today the
CPU: C24–C90, adv and all, one pass), which the progress counts and the operator page use. The published
`data/records.csv` keeps only the last three components of `result_dir` (run, `timing`, configuration),
so no host paths are published.

## Pages

| Page | Shows |
|---|---|
| `index.html` | status tiles (headline SYPD, speedup since v0.5.0, largest grid, devices), up to five key points computed from the data, grouped bars of SYPD per device and version for one configuration (grid / tracers / operators), compact table |
| `releases.html` | per device: every release-matrix configuration as dot rows (log s/day, both passes), table with speedups |
| `scaling.html` | per device, main (or the newest version measured): s/day, cost per cell and tracer, and peak memory against grid size for 1, 4 and 12 tracers; input-bound grids marked |
| `operators.html` | per device: stacked breakdown at C90 and C180, 12 tracers, one bar per version; table |
| `devices.html` | main: devices side by side per grid, speedup over the CPU |
| `methods.html` | protocol, metric definitions, commits, per-device caveats (load average, flagged passes, input-bound grids, negative operator costs, `PUBLIC_NOTES.md`), failed runs, downloads (`data/records.csv`, `data/configs.json`), how to reproduce |

## Definitions

- steady s/day = median wall time of simulated days ≥ 3 (from the harness); the site shows the mean of
  the passes and both pass values. Passes are flagged when (max − min) / mean > 10 % (`SPREAD_FLAG`).
- SYPD = 86400 / (365 × s/day); speedup = s/day(old) / s/day(new); cells = 6 N² × 72.
- Cost per cell and tracer = s/day / (cells × tracers), in ns per simulated day.
- Breakdown: input floor = none; advection = adv − none; diffusion = diff − none; convection = conv − none;
  emissions = (adv+emis) − adv; rest = all − (none + the four). Shown only when all six sets exist for a
  version; otherwise the page lists what is missing.
- Input-bound: none ≥ 50 % of the selected operator set (`INPUT_BOUND`).
- C720 uses GEOS-FP-derived binaries (`MET` in the generator) and is labelled so.

## Layout

| File | Contents |
|---|---|
| `build_site.py` | reads the records, aggregates, computes tiles / key points / caveats, writes `data/bench.js`, `data/records.csv`, `data/configs.json`, copies the assets, renders the pages |
| `layout.html` | page frame: header, navigation, status strip, footer |
| `pages/*.html` | page bodies, each with its own small script |
| `assets/site.js` | shared code: controls (one row per page, state in the URL hash), legend toggles, tables, D3 charts (dot rows, grouped bars, log-log lines, stacked bars) |
| `assets/site.css` | styles (same look as the CATRINE benchmark site) |
| `assets/d3.min.js`, `assets/d3.LICENSE.txt` | vendored D3 v7.9.0 (ISC), checksum-checked at build time |

## Design rules

- Fixed colours site-wide: v0.5.0 grey `#8f8d86`, v0.6.0 blue `#2a78d6`, main orange `#eb6834`; devices
  L40S green, A100 violet, M5 Pro magenta, CPU amber, each with its own marker shape; operator components
  have their own palette (input grey, advection aqua, diffusion violet, convection yellow, emissions red,
  rest hatched).
- One control row per page; the legend in that row toggles series. Every chart has a tap / hover tooltip
  and never captures vertical scrolling on a phone (`touch-action: pan-y`). No chart zooms.
- Numbers in text (tiles, key points, caveats) are computed from the records at build time; a key point
  appears only when the data supports it.

## Checking a build

Render every page headlessly at 1400 px and 390 px (for example Chrome via its DevTools protocol; wurst has
`google-chrome` but no Playwright), tap a chart, and look for clipped labels, overlaps and console errors.
