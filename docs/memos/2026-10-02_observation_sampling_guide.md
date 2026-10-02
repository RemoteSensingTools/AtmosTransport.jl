# Sampling the model at OCO-2, TCCON, and ObsPack observations

AtmosTransport can now write model CO2 at observation times and places while
it runs, instead of writing full 3-D fields and sampling them afterwards. You
give it a list of observations: satellite soundings, station tables, or ObsPack
files. For each one it writes the model profile from the grid cell that
contains the observation, plus everything needed to apply averaging kernels
offline.

The feature lives on the branch `feature/observation-sampling` (not yet merged
into `main`).

## 1. Get the code

```bash
git clone https://github.com/RemoteSensingTools/AtmosTransport.jl.git
cd AtmosTransport.jl
git switch feature/observation-sampling
julia --project=. -e 'using Pkg; Pkg.instantiate()'
# GPU runs: a separate environment with CUDA (see docs/src/getting_started/installation.md)
julia -e 'using Pkg; Pkg.activate("gpu-env"); Pkg.develop(path="."); Pkg.add("CUDA")'
```

Run on one of the GPU servers with `julia --project=gpu-env`. The L40S cards
need `float_type = "Float32"`.

The ERA5 transport binaries under
`/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/met/` are readable only
by the `cfranken` group. Ask Christian for access if you get a permission
error.

## 2. Run the example first

From the repository root:

```bash
julia --project=gpu-env scripts/run_transport.jl config/examples/observation_sampling_oco2mip.toml
```

This transports two test tracers for 2 days (2021-01-01 and 02) on C90 ERA5
and samples:

- every OCO-2 v11 MIP 10-second average in those 2 days (1,885 soundings);
- profiles at the 33 TCCON sites in `config/examples/tccon_ggg2020_sites.csv`
  at every met-window end.

A **met window** is the time step of the meteorological input, one hour for
these binaries. The model is sampled at the end of each window.

The run takes about a minute and writes four files to
`~/data/AtmosTransport/output/oco2mip_example/`:

```
obs_20210101_soundings.nc   obs_20210101_sites.nc
obs_20210102_soundings.nc   obs_20210102_sites.nc
```

`co2_uniform` starts at 400 ppm everywhere and should stay at 400 ppm, within
Float32 noise of about 0.01 ppm, in every sampled value. Checking that is a
quick way to confirm you are reading the files correctly.

The example uses advection only so it runs fast. Its binaries carry no
convection. For science runs, start from your production config, with
convection and diffusion on, and add the observation settings from section 3.

## 3. Add sampling to your own run

Merge these keys into your config. If it already has an `[output]` table, add
the keys there rather than writing a second `[output]` header, which TOML
rejects.

```toml
[output]
split = "daily"                  # daily files; applies to snapshots and observations
enabled = false                  # turns gridded snapshots off; sampling still runs

[output.observations]
path = "~/data/AtmosTransport/output/myrun/obs_{YYYYMMDD}.nc"
time_interpolation = "linear"    # or "nearest_window"
tracers = ["co2_natural", "co2_fossil"]   # omit to sample every tracer
write_profile_for_sites = true   # full profiles at stations (needed for TCCON)

[[output.observations.sources]]
kind = "oco2_lite"
path = "/kiwi-data/Data/model/OCO2MIP/observation_input/OCO2_b11.2_10sec_GOOD_r2.nc4"
quality_filter = "none"
```

Leave out `enabled = false` if you also want the gridded snapshots.

You can list as many sources as you like. The path becomes
`obs_<date>_soundings.nc` for point measurements and `obs_<date>_sites.nc` for
station time series. A run with no soundings at all writes no soundings files.
If it has any, every day gets one, possibly empty, and likewise for sites.

The run's time origin is `[input].start_date` at 00:00 UTC. If your config
lists `binary_paths` instead of a folder and dates, set
`[output.observations].start_time = "2021-01-01T00:00:00"`. It must fall on the
first binary's date.

In VS Code with the Even Better TOML extension, hovering over a key shows the
allowed values. Inside `[output.observations]` and its sources, unknown keys
are also flagged; elsewhere they are not. This works automatically for files
in `config/examples/` when the repository is your open workspace. For a
config elsewhere, put this on its first line, with the path to your clone:

```toml
#:schema /path/to/AtmosTransport.jl/schemas/atmos_transport_run.schema.json
```

### Satellite soundings (OCO-2, OCO-3)

`kind = "oco2_lite"` reads the MIP files in
`/kiwi-data/Data/model/OCO2MIP/observation_input/`. All of these files have
the same layout:

- `OCO2_b11.2_10sec_GOOD_r2.nc4` and `..._r3.nc4`: OCO-2 10-second averages, about 1,000 a day.
- `OCO3_b11_10sec_GOOD_r2.nc4`: OCO-3 10-second averages.
- `OCO2_b11.2_1sec_GOOD_r1.nc4`: 1-second data, about 6,000 a day.

Check the MIP protocol for which release to use. NASA's daily Lite files work
too, with date tokens in the path, for example
`.../{YYYY}/oco2_LtCO2_{YYMMDD}_*.nc4`.

`quality_filter` decides which records are sampled:

| Setting | Keeps |
|---|---|
| `"none"` | every record; the MIP co-samples all of them |
| `"flag_values"` with `quality_variable = "assimilate_flag"`, `quality_flag_values = [1]` | only assimilated records (0 = not assimilated, 1 = assimilated, 2 = withheld) |
| `"flag_max"` (default) with `quality_flag_max = 0` | records with `xco2_quality_flag <= 0`, the usual Lite-file screening |

In the MIP "GOOD" files every `xco2_quality_flag` is 0, so the default keeps
everything there too. Writing `"none"` makes the intent explicit.

### Stations from a table (TCCON, your own lists)

`kind = "table"` with `mode = "sites"` reads a CSV, TOML, or NetCDF list with
one station per row:

```
id,lat,lon,elevation,intake_height
mlo,19.536,-155.576,3397,40
```

- `elevation` is metres above sea level and is informational only. The model
  has no surface-elevation field.
- `intake_height` is metres above ground and picks the model layer. You can
  give `altitude` (metres above sea level) together with `elevation` instead.
  If both are given, `intake_height` wins. With neither, the lowest model
  layer is used.
- TCCON sites need neither, because you want the whole profile.
- Accepted alternative names are `site_id`, `latitude`, `longitude`,
  `intake_height_m`, `height_agl`, `altitude_agl`, `altitude_asl`,
  `elevation_m`, and `stop_time` for `end_time`. Header case does not matter.
- Any other column is an error, so a misspelt `intake_height` cannot silently
  fall back to the lowest layer.
- CSV files may contain `#` comment lines but no quoted fields.

When a station is sampled depends on which time columns its row fills:

| Row has | Sampled | Written to |
|---|---|---|
| no time columns | at every met-window end | `_sites` |
| `start_time` and `end_time` (both required, both inclusive) | window ends inside that UTC range; NaN outside | `_sites` |
| `times` | once at each listed UTC time, like a sounding (separate several times with `;` in CSV) | `_soundings` |

Times are ISO-8601 UTC strings, for example `2021-12-02T05:00:00`. Plain
numbers are rejected. `config/examples/observation_sites_demo.toml` shows all
three in TOML form. A table with `mode = "soundings"` instead lists one-off
measurements with `id`, `time`, `lat`, `lon` and the same optional height
columns.

### ObsPack

```toml
[[output.observations.sources]]
kind = "obspack"
mode = "soundings"     # every record at its own time and intake height (MIP in-situ protocol)
path = "/path/to/obspack/data/nc/co2_*.nc"
```

`mode` is required. `"sites"` instead gives one time series per ObsPack file
and intake height, sampled at every window end. The ObsPack reader has only
been tested on synthetic files so far, so check the first output carefully
when the real ObsPack arrives.

## 4. What is in the files

Level 1 is the top of the atmosphere and the last level is the surface.
Mixing ratios are dry-air mole fractions (mol/mol; multiply by 1e6 for ppm).
Pressures are dry, which is why their names end in `_dry`. Times are seconds
since 1970-01-01 UTC. Dimensions below are in Python (netCDF4) order. The
reference tables in `docs/src/config/output_schema.md` list them in Julia's
reversed order.

**Soundings file**, one row per observation:

| Variable | Meaning |
|---|---|
| `id`, `source` | `sounding_id`, ObsPack id, or table id; `source` is the 1-based position of the source in your config |
| `time`, `latitude`, `longitude` | the observation |
| `cell_lon`, `cell_lat`, `cell_area` | the model cell that was sampled (`cell_lon` runs 0 to 360) |
| `<tracer>` | profile, `(obs, lev)` |
| `<tracer>_column_mean` | air-mass-weighted column mean (no averaging kernel) |
| `<tracer>_intake` | value in the layer containing `intake_height`; the lowest layer when there is none, as for satellites |
| `p_half_dry`, `ps_dry` | layer-edge pressures `(obs, lev+1)` and surface pressure (Pa) |
| `air_mass_per_area_dry` | layer air mass (kg m⁻²) |
| `sample_time_prev`, `sample_time_next`, `interp_weight` | the two window ends around the observation, and the weight given to the later one |
| `interp_flag` | 0 = normal two-sided blend |

**Sites file**, a `(time, site)` grid:

- `site_id`, `ps_dry`, `<tracer>_intake`, and `<tracer>_surface` (the lowest layer).
- The chosen `intake_level` and its bottom and top heights above the model surface.
- With `write_profile_for_sites = true`, also `<tracer>` and
  `air_mass_per_area_dry` as `(time, site, lev)` and `p_half_dry` as
  `(time, site, lev+1)`.

## 5. Model XCO2 with the OCO averaging kernel (Python)

The MIP formula is XCO2 = x_prior + Σ_j w_j a_j (c_model,j − c_prior,j). Here
w_j is `pressure_weight`, a_j is `xco2_averaging_kernel`, and the 20 retrieval
levels sit at p = `sigma_levels` × surface pressure. Interpolate the model
profile in p/ps, not in absolute pressure (see section 7). If your soundings
files also hold other sources, select the OCO rows by `source` first, because
other ids are not numbers.

```python
import numpy as np, netCDF4 as nc

MIP = "/kiwi-data/Data/model/OCO2MIP/observation_input/OCO2_b11.2_10sec_GOOD_r2.nc4"
MODEL = "obs_20210101_soundings.nc"     # one model output file
TRACER = "co2_blob"                      # a tracer name from [tracers]

with nc.Dataset(MODEL) as m:
    ids = np.array([int(s) for s in m["id"][:]])       # sounding_id as integers
    prof = 1e6 * m[TRACER][:]                           # (obs, lev) ppm, lev 0 = top
    p_half = m["p_half_dry"][:]                         # (obs, lev+1) Pa
    ps = m["ps_dry"][:]                                 # (obs,) Pa

with nc.Dataset(MIP) as o:
    sid = o["sounding_id"][:]
    order = np.argsort(sid)
    k = order[np.searchsorted(sid[order], ids)]         # file row of each model row
    assert np.all(sid[k] == ids)
    lo, hi = k.min(), k.max() + 1                       # read one contiguous block
    rows = k - lo
    sigma = o["sigma_levels"][:]                        # (20,) p/psurf, top first
    ak = o["xco2_averaging_kernel"][lo:hi][rows]
    pw = o["pressure_weight"][lo:hi][rows]
    prior = o["co2_profile_apriori"][lo:hi][rows]
    xprior = o["xco2_apriori"][lo:hi][rows]
    xobs = o["xco2_2019_scale"][lo:hi][rows]            # the observation, for comparison

# Model layer midpoints in normalised pressure, interpolated to the retrieval levels.
# np.interp holds the top and bottom layer values beyond the outermost midpoints.
s_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:]) / ps[:, None]
c_ret = np.array([np.interp(sigma, s_mid[i], prof[i]) for i in range(len(ids))])
xco2_model = xprior + np.sum(pw * ak * (c_ret - prior), axis=1)
```

This was checked on the example output. Setting the kernel to 1 turns the
400 ppm uniform tracer back into 400.00 ppm. With the real kernel the result
is 400.2 to 401.7 ppm, because the a priori enters wherever the kernel is
below 1.

## 6. TCCON-site profiles (Python)

```python
import netCDF4 as nc
with nc.Dataset("obs_20210101_sites.nc") as d:
    ids = list(d["site_id"][:])
    i = ids.index("caltech")
    times = nc.num2date(d["time"][:], d["time"].units)   # every window end
    prof = 1e6 * d["co2_blob"][:, i, :]                  # (time, lev) ppm, lev 0 = top
    p_half = d["p_half_dry"][:, i, :]                    # (time, lev+1) Pa
```

Apply the TCCON averaging kernels the same way, interpolating the model
profile in p/ps.

## 7. Things to know

- **Time.** Each sounding is blended linearly between the two window ends
  around its time. With `nearest_window` it takes the nearest window end
  instead.
- **Space.** Each observation takes the value of the grid cell that contains
  it. There is no horizontal interpolation.
- **Which day's file.** With `linear`, a sounding is written to the file of
  its own day. Each day's sites file starts at 00:00, except that the first
  file also has the initial state, and ends with 00:00 of the next day. So in
  the example, day 1 has 25 records and day 2 has 24. With `nearest_window`,
  soundings in the first half hour after midnight land in the previous day's
  file.
- **Surface pressure.** `ps_dry` is rebuilt from the model's dry air mass. Over
  ocean it matches OCO's total surface pressure (ratio about 1.000). A dry
  pressure should be about 0.4% lower, since water vapour is missing from it.
  The cause is not yet known. Mixing ratios are not affected. Map profiles in
  p/ps, as above, rather than in pascals.
- **Intake heights.** Layer heights are computed from pressure with a
  constant 280 K, because the ERA5 binaries carry no temperature;
  `height_method = 0` records this. The model surface follows the ERA5
  surface pressure, not the station's elevation. Each row records the layer
  used and its bottom and top heights, so you can judge mountain sites
  yourself.
- **Run edges.** Observations outside the run's time span are skipped and
  counted in the startup log. One exactly at the final window end is kept.
  Two chained runs that share a boundary both write an observation sitting
  exactly on it.
- **Startup.** Sources are read at startup, keeping only the records inside
  the run. The whole-mission files take 6 to 11 s and about 2 GB of memory.
- **Partial files.** If a run dies, rows up to the `completed_soundings` or
  `completed_times` attribute are complete. Discard anything beyond.
- **Not done yet.** These are not produced yet: MIP submission files (daily
  co-sample files, TCCON CSV), the 3DCO2 monthly-mean diurnal cycle, and
  applying averaging kernels inside the run.

## 8. More detail

- Every option, with the Julia type it maps to: `docs/src/config/toml_schema.md`,
  section `[output.observations]`.
- Every output variable: `docs/src/config/output_schema.md`, section
  "Observation sampling files".
- Design decisions and known limits:
  `docs/memos/2026-10-02_observation_sampling_output.md`.
