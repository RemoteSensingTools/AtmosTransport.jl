# Idealized advection tests (GMD paper, Sect. 4.1)

Analytic two-dimensional flows on the sphere, run through the production runtime on the cubed
sphere, the latitude–longitude grid and the reduced Gaussian grid with every advection scheme,
in FP32 and FP64. The analytic winds become ordinary transport binaries; the runs use
`scripts/run_transport.jl`, the existing initial-condition kinds and the standard NetCDF output.

## Case table

`cases.toml` (every combination in FP64 and FP32):

| Grid | Resolutions | Flows | Schemes |
|---|---|---|---|
| Cubed sphere (GEOS convention) | C48, C96, C192 | A (tilt 45°), B, C | upwind, slopes, PPM (default limiter), PPM CW84, Lin–Rood 5, Lin–Rood 7 |
| Lat–lon | 3.75° (96 × 48), 1.875° (192 × 96), 0.9375° (384 × 192) | A, B; C at 1.875° | upwind, slopes, PPM (default limiter), PPM CW84 |
| Reduced Gaussian (ECMWF rings) | N48, N96 | A, B; C at N48 | upwind |

Tagged variants, reported separately and left out of the convergence results: the flow-A tilt
series `<grid>_A_tilt0` and `_tilt90` (C96 PPM and Lin–Rood 7, lat–lon 1.875° PPM, N48 upwind;
45° is the untagged case), and the substep-sensitivity case `c96_B_cfl025` (C96 flow B with half
the outflow per flux application, `courant = 0.25`: 16 instead of 8 substeps per window; slopes,
PPM, PPM CW84, Lin–Rood 7, FP64), and the forcing-window cases `c96_B_w15` and `c192_B_w15`
(flow B with 15-minute instead of hourly windows, `windows_per_day = 96`; PPM, PPM CW84,
Lin–Rood 5 and 7, FP64). The ring tables `../grids/n48_pl.txt` and `n96_pl.txt` were
read from the ECMWF eccodes samples `reduced_gg_pl_48` and `reduced_gg_pl_96`.

## Flows and tracers

| Flow | Definition | Tracers (one run carries all) | Exact solution |
|---|---|---|---|
| A | solid-body rotation, axis tilted 45°, one revolution in 12 days (Williamson et al., 1992, case 1) | cosine bell, Gaussian hill, uniform | rotation of the initial field |
| B | non-divergent deformational flow with background rotation, T = 12 days (Lauritzen et al., 2012, Eqs. 18–20, κ = 2; Nair and Lauritzen, 2010, case 4) | Gaussian hills, cosine bells, slotted cylinders, uniform | initial field at T; backward trajectories at T/2 |
| C | divergent deformational flow with background rotation, T = 12 days (Lauritzen et al., 2012, Eqs. 21–22, κ = 1; Nair and Lauritzen, 2010, case 3 plus the background wind) | Gaussian hills, cosine bells, uniform | as B, with the air-mass-weighted cell mean at T/2; the uniform mole fraction must stay uniform while the air mass changes |

In the frame rotating with the background wind (`λ' = λ − 2πt/T`) both deformational flows are a
steady field `W` times `cos(πt/T)`. Parcels therefore follow `W` in the pseudo time
`τ = (T/π) sin(πt/T)`, and the density of flow C is `ρ/ρ0 = exp(−∫ div W dτ)` along the path,
with `div W = −3κ sin λ' sin θ cos²θ` (nondimensional; checked against a finite difference of `W`
to 2e-10). Departure points and densities come from RK4 backward in `τ` with step 0.005 (in units
of T/5; positions converge as h⁴ to 1.5e-9 for B and 1e-11 for C, and agree with a 2880-step
RK4 in physical time).

The vertical coordinate is pure sigma with four layers (Lin–Rood's FV3 vertical profile needs at
least four); every layer is the same 2-D test, scaled by its `ΔB`, and the vertical mass flux is
zero.

## How the winds enter

- **Faces.** Each grid is reduced to one face list: the two cells of each face (positive
  transfer from `left` to `right`) and a path between two corners whose right-hand normal points
  from left to right. Cubed-sphere faces come from the global face table of `Grids`
  (`build_cs_global_face_table`), including the mirror entries on the neighbour panel; corners
  shared by several faces are stored once, so a stream function has one value per corner.
- **Stream-function flows (A, B).** Transfer through a face = column mass per area ×
  `(Ψ(start) − Ψ(end))`, with `Ψ` the stream function integrated over the window (exact for A,
  8-point Gauss–Legendre in time for B). The transfers telescope around every cell, so the air
  mass is constant and `cm = 0`; no Poisson balance is involved.
- **Divergent flow (C).** Transfer = `σ0 ∫∫ (ρ/ρ0) V·n dl dt` with the exact density, by
  Gauss–Legendre with 3 nodes in time and 4 along the face per window (against 6 × 8 nodes:
  3e-10 of the largest transfer), the density by the pseudo-time integration with step 0.01. The
  cell air mass follows by closure, `M(t1) = M(t0) − div F` from `M(0) = σ0 A`, so continuity
  closes exactly and the closure mass tracks the exact cell integral of the density (see the
  probes). Earlier versions integrated continuity in the generator (centred: unstable; donor
  cell: first-order implied wind; Lax–Wendroff sub-steps: first order in time); they are gone.
- **Binaries.** Hourly windows, one binary per day, `mass_basis = dry`, Float64 on disk (the FP32
  runs cast on load). The substep count per window is set so that at most `--courant` (default
  0.5) of a cell's air mass leaves it per flux application; the cubed-sphere runtime follows the
  binary schedule, so this bounds the Courant number of every sweep.
- **Initial conditions** are the analytic cell means. Cubed sphere: a `cs_native` file
  (copied cell for cell). Latitude–longitude: a `file` IC on the model's own cell centres.
  Reduced Gaussian: a `file` IC on the ring latitudes with a 0.1° longitude axis that holds each
  ring's cell means piecewise constant, so the bilinear sample at a cell centre returns the cell
  mean exactly. The analysis checks the snapshot at hour 0 against the reference
  (`ic_error` in `norms.csv`, ≈ 1e-16 in FP64).
- **Reference** fields are air-mass-weighted cell means `∫ρχ dA / ∫ρ dA` (what the model's
  tracer mass / air mass represents; the plain area mean for A and B, where ρ is uniform) from
  sub-cell quadrature: 6 × 6 Gauss–Legendre points in longitude × sin(latitude) on lat-lon and
  reduced-Gaussian cells, 6 × 6 sub-cells of the nested cubed sphere, at the departure points.
  For the slotted cylinders every cell that an edge may cut is evaluated again with 48 × 48
  points: the cells whose 36 samples are not all equal, their face neighbours, both cells of a
  face with different values on its two sides, and those of the next ring of cells in which
  16 × 16 samples are not all equal. The selection is wider than the cells with unequal samples
  because an edge can clip a thin strip or a corner of a cell between the samples. The same
  means are the initial condition. At T the reference is the initial
  mean (field and density are back at their initial state).

## Probes printed by `generate.jl`

- an independent check of the transfers: the stream-function transfer against a direct
  Gauss–Legendre line integral of `V·n` along every face (relative difference ≈ 1e-14); for flow C
  the stored transfers of window 1 against 6 × 8 nodes and a 4× smaller pseudo-time step;
- for flow A on lat-lon: the total eastward transfer across the date line against the analytic
  `σ ω R² 2 cos α Δt`;
- the replay residual of the packed arrays (the per-cell test of the runtime gate); the runs
  also set `[input] validate_replay = true`;
- the largest outflow fraction per window and the resulting substep count;
- trajectories: pseudo-time step h against h/2 and against a 2880-step physical-time RK4 at T/2,
  return of position and density at T;
- slotted cylinders: number of refined cells, and the change of the initial cell means from
  6 × 6 to 48 × 48 and from 24 × 24 to 48 × 48 points;
- flow C: `div W` against a finite difference; the range of the cell-mean density at T/2; the
  global integral of ρ at T/2 against `4πR² ρ0`; the closure air mass against the cell integral of
  ρ at T/2 (with 6 × 6 and 12 × 12 points, which shows the difference is the quadrature of the
  check, not the closure) and at T.

Run `scripts/diagnostics/inspect_transport_binary.jl` on one binary to see the header, the
capabilities and the driver check.

## Files

| File | Role |
|---|---|
| `IdealizedAdvection.jl` | flows, initial fields, face geometry of the three grids, transfers, binary and IC writers, references |
| `generate.jl` | one grid × flow: binaries, `ic.nc`, `reference.nc`, `case.toml`, probe output |
| `cases.toml` | the case table: grids, flows, schemes, precisions, output levels and cadence |
| `make_configs.jl` | one run config per grid × flow × scheme × precision |
| `run_matrix.sh` | driver: `generate` (missing cases only, `REGENERATE=1` for all), `configs`, `run`, `analyze`, `plot`, or `all`; a finished run is reused only if its fingerprint (config and case file) is current, and the run stage fails if any run failed |
| `analyze.py` | norms, extrema, mass drift, uniform-tracer error, FP32 − FP64 → `results/` |
| `plot_figures.py` | `figures/convergence.pdf`, `deformation_maps.pdf`, `conservation.pdf`, `tilt.pdf` (+ PNG) |
| `panel_edge_split.py` | cubed sphere: error near the panel edges against the panel interiors → `results/panel_edge_split.csv` (part of the `analyze` stage) |
| `forcing_window_error.jl` | error of window-mean winds against the analytic flow, without the model → `results/forcing_window_error.csv` |
| `float32_drift/` | the diagnostics of the `Float32` tracer-mass drift (own README) |
| `results/norms.csv` | one row per run × tracer × time (T/2, T) |
| `results/convergence.csv` | observed order between consecutive resolutions of each grid family, from the actual grid-spacing ratio |
| `results/minimal_resolution.csv` | grid spacing at which the flow B cosine bells reach l2 = 0.033 (log–log interpolation between the bracketing resolutions) |
| `results/summary.csv` | per grid family × scheme: orders, l2 at the finest grid, minimal resolution, extrema, FP64/FP32 drift, FP32 − FP64 |
| `results/table_convergence.tex` | LaTeX `tabular` for the paper (all grids; `booktabs` rules) |
| `results/tilt.csv` | flow A errors at the tilts 0°, 45°, 90° |
| `results/mass_series.csv` | relative tracer-mass change of every snapshot (budget output and fields) |
| `results/substep_sensitivity.csv` | `cfl*` variants next to their base case |
| `results/window_sensitivity.csv` | `w*` variants (shorter met windows) next to their base case, with the observed order of both |
| `results/panel_edge_split.csv` | cubed-sphere l2 at T split into the cells within 7.5° of a panel edge and the rest, with the observed order of each |
| `results/forcing_window_error.csv` | normalized l2 difference between advection by window-mean winds and by the analytic winds, for windows of 4 h to 15 min |

Work directory layout (`$IDEALIZED_ADVECTION_WORKDIR`): `binaries/<case>/`,
`inputs/<case>/{ic.nc, reference.nc, case.toml}`, `runs/<case>_<scheme>_<f32|f64>/`
`{config.toml, run.toml, log.txt, output.nc, fingerprint}`, `logs/`, with
`<case> = <grid>_<flow>[_<tag>]`. `analyze.py` reads the runs listed in `runs/runs.txt` and stops
on a run that is unfinished or stale, or whose output lacks a snapshot at 0, T/2 or T.

## Norms

`l1`, `l2`, `linf` are normalized by the exact field (Williamson et al., 1992), area weighted, on
the bottom output level; `layer_diff` is the largest difference between the top and bottom output
levels. `undershoot`/`overshoot` are the differences between the numerical and the exact extrema at the
same time, relative to the initial range (Lauritzen et al., 2012, Appendix A), not defined for the
uniform tracer (use `max_abs_chi_minus_1`). `mass_drift_budget` is the relative change of `<tracer>_total_mass` (the model's
compensated Float64 budget); `mass_drift_fields` recomputes it from mixing ratio × air mass of the
bottom level. `fp32_minus_fp64_l2` is the area-weighted RMS difference of the two precisions
relative to the RMS of the FP64 field, next to the FP64 run's `l2` error (`fp64_l2_same_case`).

Where the runs execute: Float64 on the CPU; the Float32 runs of the grid entries with
`gpu_precisions = ["Float32"]` in `cases.toml` (C192 and lat–lon 0.9375°) on CUDA GPUs, one
run per GPU at a time under `$GPU_LOCK_DIR/gpu<N>.lock` (`GPUS` selects the devices). A lat–lon
0.9375° flow-A run (117 substeps per window) takes 15–85 min in Float64 on 8 CPU threads and
31–47 s in Float32 on an L40S.

Grid spacing (for the convergence orders, the minimal resolution and the figures) is the nominal
spacing in degrees: 90/N on C<N> (the cube's spacing at the equator), 360/Nx on lat–lon, and
90/N on N<N> (the spacing of the Gaussian latitudes). The observed order between two resolutions
is `log(e_coarse/e_fine) / log(h_coarse/h_fine)`.

## Window-mean winds and other sensitivities

An offline model applies one mass transfer per met window at a constant rate, so its parcels
follow the window-mean wind, not the continuously varying analytic wind of the reference
solution. The difference is independent of the grid: it is the error that remains when the
spatial error has converged away (at finite resolution the two can partly cancel).
`forcing_window_error.jl` measures it without the model (backward trajectories through the
window-mean wind, compared with the exact departure points): for hourly windows the normalized
l2 difference at T is 1.3e-3 for the Gaussian hills and 1.7e-3 for the cosine bells of flow B,
and it falls with the square of the window length (8.0e-5 and 1.1e-4 at 15 minutes). For flow
C the transfers are window means of density times wind; the window-mean wind gives an estimate
of the same effect (2.3e-4 and 4.0e-4). Flow A is steady and has no such error.

Three measurements bear on the order of 1.5 of the Lin–Rood option between C96 and C192 in
flow B (after 2.2 between C48 and C96):

- The `w15` cases rerun flow B with 15-minute windows (`results/window_sensitivity.csv`). The
  l2 errors at T change by less than 1 % (Lin–Rood 7 at C192 5.74e-3 → 5.71e-3, order
  1.51 → 1.53); at T/2 the l2 error of the Gaussian hills changes by up to 4 %, and linf by up
  to 9 %. The hourly windows do not set the error at T.
- The error is not concentrated at the panel edges (`results/panel_edge_split.csv`): within
  7.5° of an edge it converges at 1.6, in the panel interiors at 1.5 (Lin–Rood 7). This does
  not show where the error was generated, since errors made at an edge are advected away from
  it.
- More substeps make the errors larger (`results/substep_sensitivity.csv`, C96): with twice the
  substeps the l2 error of the Gaussian hills and cosine bells rises by 3–19 %, that of the
  slotted cylinders by 1–2 %.

The cause of the lower order has not been identified.
