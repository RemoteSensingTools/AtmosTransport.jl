# Validation scripts

These scripts validate production operators, transport binaries and numerical
accuracy. They run from the repository root with `julia --project=.`.

## GPU face quadrature

`gpu_face_quadrature.jl reference|device|emulate T GRID OUTPUT [1|2]` compares TM5
face construction for `ll` (360×180), `rg` (synthetic octahedral N64) and `cs`
(C90). T255 and T639 retain the input harmonics. Each comparison covers a
rotation, sectoral and off-sectoral truncation-edge modes, then every native
level of ERA5 2021-12-01 00 UTC. Input paths resolve under
`~/data/AtmosTransport/met/era5/spectral_hourly`; the reference command caches
only that decoded hour under OUTPUT. Float64 CPU references precede device
runs. `emulate` loads no CUDA and uses the device algorithms with FFTW/BLAS;
it cannot establish CUBLAS/CUFFT rounding or GPU determinism. `sample` is a
bounded CPU investigation of six latitude cells (polar and equatorial), not
the full-grid qualification.

Run `/temp1/cfranken/jobs/gpuquad/run_gpu_validation.sh` on the L40S host.
Each invocation creates a fresh `run_UTC_TIMESTAMP/` directory under
`/temp1/cfranken/jobs/gpuquad/`, preventing stale results from passing a rerun.
The launcher prints its output path and skips dependent GPU comparisons if a
CPU reference fails. `--resume-windows OUTPUT` explicitly resumes the window
phase from complete raw accuracy artifacts; it checks both process memory
reports and refuses to overwrite existing window output directories.
GPU invocations lock `/temp1/cfranken/goldens/gpu1.lock`, select physical GPU 1,
require the Float32 precision policy, disable scalar indexing and use Float32
fields/surfaces. The runner also launches `bench_gpu_face_window.jl` for a
complete LL 1° and C90 00–01 UTC window. Each backend gets one warmup and two
measurements; CPU uses 16 Julia threads and BLAS 1. The benchmark shortens only
the day loop in memory and preserves same-day endpoint reads and production
closure/write gates. It writes no source changes.

CSV metrics include absolute Pa/kg/s error, per-face relative error, and
per-level relative L2. `*_levels.csv` also records the joint u/v face RMS,
maximum error divided by that RMS, and combined horizontal relative L2 at
every level. CS uses its unique canonical faces. Directional zero fields do
not make the joint RMS singular. The production targets are maximum face
error/level RMS ≤1e-5 and balanced T639 cm relative L2 ≤1e-4; historical
counts outside `1e-6*abs(reference)+2e-14*max(field_peak,1)` (the floor handles symmetry
zeros). Aggregate errors never stand in for a per-face bound. Repeated GPU
calls and LL/RG band budgets 32/4 report actual bitwise and numerical changes.
The runner launches two fresh GPU Julia processes per accuracy case. Repetition
1 stores shape-qualified SHA-256 fingerprints; repetition 2 compares against
them and writes `fresh_process_bitwise`. Both process memory peaks are retained.
The four-cell alternate is smaller than the 128-row cap, so the LL 1°
comparison changes actual transform shapes. RG uses N64 so its latitude
bands also contain multiple cells before hitting the row cap. CS has no latitude-band mode. `summarize_gpu_face_quadrature.py` writes
`RESULTS.md` and exits nonzero for missing results, failed repeatability or
scale-aware face-target failures or balanced cm failures; it still summarizes
all completed cases. Nonfinite payloads, metrics and timings, duplicate or missing
case/field/native-level rows, and nonpositive joint horizontal reference norms
fail qualification. An exactly zero reference is allowed only with exactly zero
error and zero scaled metrics. The historical mixed-bound counts remain diagnostic.
`compare_gpu_face_windows.py` compares the written am/bm/cm dry production
payloads and writes `balanced_windows.csv`. Computation uses Float64 on CPU
and Float32 on GPU; both backends store Float64 payloads. The comparator
checks truncation, level mapping and substep schedules before comparison.

`gpu_face_memory.jl` records exact process-lifetime high-water usage/reservation
for the active CUDA pool, including warmup. Independent 100 ms `nvidia-smi`
samples begin inside the locked GPU command and include the context, pool
reservations and vendor allocations; these
whole-device peaks can miss short transients. The two scopes are reported
separately. The runner uses isolated Julia processes to bound live plans.

## Depends on

AtmosTransport production APIs, JLD2/GRIB, the configured Julia environment;
CUDA and nvidia-smi for device qualification only.
Run validator rejection regressions with
`python3 test/diagnostics/test_gpu_face_qualification.py`. The summary and balanced-window comparator use Python and NumPy.

## Used by

Manual developer validation and the L40S runner. The measured T255/T639
results and remaining edge-harmonic maximum-bound failures are documented
in the spectral manual §17.

## Shared helpers owned here

`gpu_face_memory.jl` is included by both qualification drivers.

## Forward<->adjoint mirrors

None.
