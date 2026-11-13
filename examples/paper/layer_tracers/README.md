# Process experiments with layered tracers (Sect. 6.2)

Idealized tracers without surface fluxes are transported with the meteorology of the CATRINE
runs, once for every combination of advection (A), boundary-layer mixing (P) and convection (C).
They show which process makes two configurations differ. Nothing in `src/` is specific to these
experiments: the runs are ordinary `scripts/run_transport.jl` runs of generated configurations.
They need the `pressure_gaussian` initial condition, and the field swap needs the
`pressure_overlap` mode of `scripts/preprocessing/attach_catrine_tm5_convection_cs.jl`.

## Method

**Tracers.** Seven tracers start as Gaussian profiles of the dry mole fraction in
σ = p / p_surface, the same in every column (`init.kind = "pressure_gaussian"`): centred at the
surface (σ = 1) and at σ = 0.95, 0.85, 0.70, 0.50, 0.25 and 0.07. The standard deviation of each
is the layer thickness of the coarser, 72-layer grid at its centre (0.012 to 0.045), and each
layer holds the mean of the profile over its own σ interval, so the 72-layer MERRA-2 grid and
the 117-layer ERA5 grid start from the same distribution. All tracers carry the same number of
molecules. `make_configs.py --single-layer` gives the earlier initial condition, one model
layer per column.

**Cases.** Two cases with every process combination, at C90 for January, April, July and
October 2022:

| Case name | Meteorology | Mixing | Convection |
|---|---|---|---|
| `era5` | ERA5, 117 layers | TM5 (`tm5_dkg`) | TM5 solver on ERA5-based fields |
| `merra2` | MERRA-2, 72 layers | GEOS-Chem non-local scheme | GEOS-Chem scheme on the archived mass flux |

Three more cases exchange one thing each, for the combinations C and APC:

| Case name | Change | How |
|---|---|---|
| `era5ifs` | IFS solver on the ERA5 fields | `--convection ifs` |
| `merra2tm5matrix` | TM5 matrix solver on the MERRA-2 mass flux | `--convection cmfmc_matrix` |
| `merra2era5conv` | ERA5 convective fields in the MERRA-2 run | binaries from `scripts/preprocessing/attach_catrine_tm5_convection_cs.jl --vertical-mapping pressure_overlap`, then `--convection tm5` |

**Metric.** The vertical distribution of a tracer is described by percentiles of its amount in
σ: the median, the 15.9 % and 84.1 % points (the bounds of the central 68 %, what plus and minus
one standard deviation are for a Gaussian), the 2.3 % and 97.7 % points (central 95 %), and the
mean. Percentiles assume no shape and give the upward and the downward spread separately. The
tracer is taken as uniform within a model layer. σ follows the terrain, so all columns are
used.

## Files

| File | Purpose |
|---|---|
| `run_matrix.sh` | Driver of all runs of the section: configurations, runs, reduction, figures |
| `make_configs.py` | Writes the run configurations of one case (months, process combinations, precisions) |
| `sigma_spread.py` | Per run: cumulative distribution of every tracer in σ at every output time, for all latitudes, three latitude bands and the boxes of `BOXES` |
| `plot_spread.py` | Percentile spread with time, cases overlaid (`--style bands`: one row per case); a table of the percentile points as CSV; `--tracer … --regions …` for one tracer in several regions |
| `column_lift.py` | Per run: 2° maps of the mean σ of each tracer and of its share above σ = 0.5 |
| `paper_figures.py` | The four figures of Sect. 6.2 from the reduced files |
| `zonal_reduce.py`, `plot_zonal.py` | Latitude–pressure sections and band profiles (not used in the paper) |
| `figures/` | The four paper figures, PDF with PNG previews |
| `results/` | Percentile tables behind the attribution figure and table (July 2022, all latitudes). Columns: `start` (centre of the release), `mean`, `median`, `p02`, `p16`, `p84`, `p98` (σ of the 2.3, 15.9, 84.1 and 97.7 % points), `up_1sigma = start − p16`, `down_1sigma = p84 − start`, and the same for `2sigma` with `p02` and `p98`. For the surface tracer the `down_` values are negative: the tracer cannot spread below σ = 1 |

## Reproducing

Large files are not committed: the transport binaries and the model output (2.6 to 8.4 GB per
one-month run with all snapshots, about 400 GB for the 74 runs). `run_matrix.sh` runs
everything from the repository root:

```bash
export LAYER_TRACERS_WORKDIR=/path/to/scratch        # configurations, model output, reduced files
export ERA5_BINARIES=/path/to/era5_c90_binaries      # daily C90 binaries with TM5 convection
export MERRA2_BINARIES=/path/to/merra2_c90_binaries  # daily C90 binaries with GEOS physics
export GPU_LOCK_DIR=/path/to/locks                   # gpu<N>.lock files shared with other jobs
GPUS="0 1" examples/paper/layer_tracers/run_matrix.sh all
# or stage by stage: crossover | configs | run | reduce | figures
```

- `crossover` (not part of `all`) builds the July 2022 MERRA-2 binaries with ERA5-based
  convection for the case `merra2era5conv`. It needs `ERA5_CONVECTION_ROOT` (the 1°, 3-hourly
  TM5 convection archive) and `ERA5_ROOT` (ERA5 N320 native GRIB, for the humidity). Without
  all 31 days the case is left out: `configs` then writes 72 configurations, the attribution
  figure is drawn without it, and the tables in `results/` are not rewritten.
- `configs` writes 74 configurations in Float32: 56 for the two main cases (7 process
  combinations × 4 months), 16 for the two solver swaps and 2 for the field swap.
  `PRECISIONS="Float32 Float64"` doubles them.
- `run` skips runs whose log is complete, so it can be restarted, and fails if any configured
  run has no finished log. One month with advection takes 4 to 6 minutes for MERRA-2 and 7 to
  11 minutes for ERA5 on an NVIDIA L40S, about half of it writing the snapshots.
- `reduce` and `figures` write `<work>/spread`, `<work>/maps`, the figures in `figures/` and
  the tables in `results/`.

Single steps, for other months, process combinations or regions:

```bash
L=examples/paper/layer_tracers
python3 $L/make_configs.py --met merra2 --binaries <dir> --resolution c90 --configs <dir> \
    --output <dir> --months 2022-07 --arms C APC --precisions Float32 \
    --convection cmfmc_matrix --name merra2tm5matrix
python3 $L/plot_spread.py --spread <work>/spread --figures <dir> --month 202201 --arm C
python3 $L/plot_spread.py --spread <work>/spread --figures <dir> --month 202207 --arm C \
    --tracer sfc --regions "Amazon" "Pacific 48°S" "Europe"
```

The Python scripts need numpy, matplotlib and netCDF4; `paper_figures.py` also cartopy for the
coastlines.

## Checks

- `sigma_spread.py` sums ramps with two weighted counts. Against the direct sum over layers the
  cumulative distribution agrees to 3e-13 of the total in double precision; the stored files
  are single precision (3e-8).
- The percentile points at the start of a run reproduce the configured centre and width on
  both grids: the 15.9 % and 84.1 % points of the σ = 0.5 tracer lie at 0.460 and 0.542 for
  ERA5 and at 0.458 and 0.545 for MERRA-2. Over all tracers the two cases start within 0.006.
- Total tracer mass over a one-month Float32 run (the `<tracer>_total_mass` output) changes by
  at most 1.6e-9 without advection and by at most 2.6e-8 with advection.

## Results in brief

Largest difference between the `era5` and `merra2` cases in σ, for any tropospheric tracer and
time, over the four months:

| Processes | Median, 15.9 % and 84.1 % points, mean | Including the 2.3 % and 97.7 % points |
|---|---|---|
| Advection + mixing | 0.018 to 0.024 | 0.031 to 0.041 |
| All processes | 0.08 to 0.09 | 0.20 to 0.23 |
| Convection only | 0.27 to 0.29 | 0.37 to 0.43 |

- `merra2era5conv` follows `era5`: within 0.03 (convection only) and 0.05 (all processes) in
  any of the six statistics, 0.008 in the mean.
- `era5ifs` equals `era5` to 0.009 in any statistic and 0.002 in the mean.
- `merra2tm5matrix` agrees with `merra2` to 0.04 with all processes, and to 0.06 with
  convection alone for the tracers released at σ = 0.95 and above. With convection alone it
  moves nothing out of the two lowest model layers, which hold 95 % of the surface tracer; the
  GEOS-Chem scheme mixes the layers below the cloud base into it. For that tracer the two
  solvers differ by up to 0.65.

So the difference between the two main cases is set by the convective mass fluxes and not by
the rest of the meteorology; the solver does not matter on the ERA5 fields, and on the MERRA-2
mass flux it matters only for the lowest layers when convection acts alone.
