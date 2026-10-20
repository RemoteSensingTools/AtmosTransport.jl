# AMD ROCm hardware runbook

The AMDGPU.jl extension selects `GPU(:rocm)`, `ROCArray`, and
`ROCBackend()`. Float32 and Float64 are supported. The implementation loads
without a functional ROCm installation; **MI300X/MI355X hardware validation is
pending**. Run the checks below before using it for scientific production.

## Install and identify the device

Use a Linux ROCm installation supporting your accelerator (MI300X: gfx942;
MI355X: gfx950), including the kernel driver and access to `/dev/kfd` and the
render device. Use Julia 1.10–1.13; local construction checks used Julia
1.12.6 and AMDGPU 2.8.0. The compat range accepts AMDGPU 2.8 and subsequent
compatible 2.x releases. Resolve a fresh environment on each Julia version.

From the checkout, create a separate environment. This leaves AMDGPU optional
in AtmosTransport's own Project.toml and out of test/Project.toml:

```bash
julia --threads=8 -e 'using Pkg; Pkg.activate("rocm-env"); Pkg.develop(path="."); Pkg.add(PackageSpec(name="AMDGPU", version="2.8")); Pkg.precompile()'
HIP_VISIBLE_DEVICES=0 julia --threads=8 --project=rocm-env -e 'using AMDGPU, AtmosTransport; AMDGPU.versioninfo(); @assert AMDGPU.functional(); @assert Base.get_extension(AtmosTransport, :AtmosTransportAMDGPUExt) !== nothing'
```

AMDGPU discovers system ROCm automatically. For a nonstandard installation,
set `ROCM_PATH=/path/to/rocm` before starting Julia. Keep it pointed at the
installation root. `HIP_VISIBLE_DEVICES` selects the accelerator. No
`JULIA_AMDGPU_*` setting is required normally. For installation diagnostics,
`JULIA_AMDGPU_CORE_MUST_LOAD=1` and `JULIA_AMDGPU_HIP_MUST_LOAD=1` turn runtime
initialization failures into errors; leave them unset on machines without
ROCm. These names are checked in AMDGPU 2.8's `src/AMDGPU.jl`. Record
`AMDGPU.versioninfo()`, Julia version, package manifest, and ROCm version with
results. See [AMDGPU installation information](https://amdgpu.juliagpu.org/stable/install_tips/).

## Smoke checks

```bash
ATMOSTR_RUN_ROCM_TESTS=1 HIP_VISIBLE_DEVICES=0 \
  julia --threads=8 --project=rocm-env test/diagnostic/test_rocm_smoke.jl
```

Without the opt-in variable, this file skips before importing AMDGPU. With
it, a nonfunctional runtime fails the test. The small lat-lon and cubed-sphere
models select `backend = "rocm"` through the runner's architecture configuration
path and verify device residency. They take three steps using Upwind and PPM,
TM5 (legacy and collaborative requests), CMFMC, and vertical diffusion. A closed
circulation activates horizontal and vertical advection; cubed-sphere PPM uses
the CATRINE FV3 kord 8 vertical profile, while lat-lon PPM uses its default
profile. A comparison with zero vertical flux verifies that vertical transport
changes the gradient tracer. The tests compare host results for
both precisions, verify mass conservation and uniform VMR, and require the
nonuniform tracer to move. Additional shared helpers test halo exchange,
point operations, CFL reductions, and column-ledger carry. Transfer checks
exercise pinned host arrays, pinned tape and mmap tape read caches.

Float32 comparisons allow 32 eps relative error after three complete steps;
Float64 comparisons use 2e-12. These are smoke tolerances, not measured ROCm
error claims. Float32 collaborative LU retains its portable 85-level,
32 KiB shared-memory limit. ROCm Float64 currently uses the legacy per-thread
solver, including for a collaborative request (with a warning). No levels are
removed or precision reduced. The 32-thread collaborative groups use explicit
workgroup barriers and are valid on wave64; tuning for MI300/MI355 is pending.

Pinned host storage uses AMDGPU's documented
`unsafe_wrap(ROCArray, pointer(array), size(array); own=true)` registration;
the registration is retained until the host array is collected. Registered
arrays must not be resized. The extension installs synchronization hooks for
tape copies and uses `AMDGPU.HIP.reclaim()` to trim the device pool. See
[AMDGPU memory APIs](https://amdgpu.juliagpu.org/stable/api/memory/).

## Production config

Copy a run config that matches your verified version-4 binary, adjust its input
and output paths, then set:

```toml
[architecture]
backend = "rocm"

[numerics]
float_type = "Float64" # or "Float32"; match the binary to avoid host conversion
```

```bash
HIP_VISIBLE_DEVICES=0 julia --threads=8 --project=rocm-env \
  scripts/run_transport.jl /path/to/run_rocm.toml
```

The CLI loads AMDGPU before AtmosTransport. Startup rejects a nonfunctional
ROCm runtime and checks model residency. `backend = "auto"` tries CUDA then
ROCm on Linux; use explicit `"rocm"` for reproducible hardware runs.
`ATMOSTR_PROFILE_MODE` is a CUDA profiler facility; use the ROCm tools below.

## Maintained synthetic benchmarks

For the repository's synthetic [benchmark harness](benchmarks.md), put the
isolated ROCm environment first on the environment stack:

```bash
JULIA_LOAD_PATH="/path/to/rocm-env:@:@stdlib" HIP_VISIBLE_DEVICES=0 \
  julia --threads=8 --project=benchmarking benchmarking/run_benchmarks.jl \
  --backend=rocm --float-type=Float32
```

The benchmark environment declares CUDA 5, whose GPUCompiler requirement
conflicts with AMDGPU 2.8. Resolve AMDGPU separately, then place that environment
first so its compiler wins. Appending it after the CUDA environment fails to
load AMDGPU. Loading both the AMDGPU runtime and benchmark package with the
runtime environment first has been verified without a functional ROCm.

## Transfer the scaling experiment

The existing harness is external to this repository at
`/temp1/cfranken/scaling`. Work on a **copy** on the ROCm machine. Copy
`benchmark.jl`, `common.sh`, `run_timing.sh`, `timing.py`, `configs/`, and,
if needed, `stream_triad.jl`. Keep `run_ncu_roofline.sh` and
`ncu_postprocess.py` as references for metric definitions, not executable ROCm
profilers. Create fresh writable `tmp/`, `cache/`, `logs/`, and `results/`
directories. Recreate the Julia environment there; do not transfer compiled
Julia caches from the NVIDIA machine.

Transfer all seven daily Float32 transport binaries (20211201–20211207) at each
selected resolution C24, C48, C90, C180, C360, C720. C24–C360 use
`bin/C<N>/merra2_transport_YYYYMMDD_float32.bin`. The source C720 payloads
are under `/temp2/cfranken/scaling/bin/C720`, named
`geos_transport_YYYYMMDD_float32.bin`. Start with C24/T1. For T4/T12 also
transfer `fluxes/C<N>/catrine_{fossil_co2,sf6,rn222}_c<N>.nc` and the CATrINE
initial fields and natural-CO2 three-hourly LMDZ flux files referenced by the
configs under `ATMOSTRANSPORT_DATA_ROOT/catrine`. Audit every path in the
chosen TOML; binaries alone do not supply emissions or initial conditions.
Use matching Float64 binaries for a Float64 scaling campaign.

Edit only the copied harness:

- Change `SCALING` and `REPO` in `common.sh`, and absolute paths in copied scripts
  and TOMLs. Set each config's backend to `"rocm"`.
- Replace `using CUDA` with `using AMDGPU` and all timing-boundary
  `CUDA.synchronize()` calls with `AMDGPU.synchronize()` in `benchmark.jl`.
- Replace `CUDA_VISIBLE_DEVICES` and `nvidia-smi` checks in `timing.py`/shell
  wrappers with ROCm device selection/queries. Replace NVIDIA GPU locks with
  a lock file in the copied harness. Do not access the original goldens locks.
- Keep timing mode, which warms the first full day and times subsequent days.
  The `ncu` mode calls CUDA.Profile.start/stop and cannot run unchanged. In the
  copied profiling mode, use the same warm-window/extra-step logic and target
  step count, but remove those CUDA calls, synchronize at both boundaries, and
  record the measured dispatch range (or add ROCTx ranges using the installed
  ROCm tracing API). Preserve `profile_steps.txt` for normalization.

Example with the adapted copy and exported `SCALING` pointing to it:

```bash
HIP_VISIBLE_DEVICES=0 ATMOSTR_TIMERS=1 julia --startup-file=no --threads=8 \
  --project=rocm-env "$SCALING/benchmark.jl" \
  "$SCALING/configs/run_C24_T1.toml" "$SCALING/results/rocm/C24_T1" timing
```

Do throughput timing separately from counter profiling: counters can serialize
kernels and require repeated application runs. Keep cadence, tracer count,
vertical scheme, forcing, precision, and measured steps identical to NVIDIA.

## ROCm counters and roofline

Check the installed tool's options and available counters first:

```bash
rocprofv3 --help
rocprofv3 --list-avail > rocm-counters.txt
# Newer installations also provide:
rocprofv3-avail list --pmc
rocprofv3 --kernel-trace --hip-trace --output-format csv -d trace -- \
  julia --threads=8 --project=rocm-env "$SCALING/benchmark.jl" \
  "$SCALING/configs/run_C24_T1.toml" "$SCALING/results/trace" timing
```

Use the trace to identify warm measured dispatches. For counter collection,
put supported counter groups in a text file, one `pmc:` line per pass, then
run `rocprofv3 -i counters.txt --output-format csv -d counters -- <same command>`.
Select the same warmed dispatch range in each pass; exclude compilation,
initialization and warmup. Counter names and collection limits vary by GPU
and ROCm release; use the installed definitions. See the
[rocprofv3 guide](https://rocmdocs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html).

| NVIDIA harness measurement | ROCm measurement |
|:--|:--|
| `gpu__time_duration.sum` | Kernel end minus start timestamps, summed over selected dispatches |
| `dram__bytes_read.sum`, `dram__bytes_write.sum` | HBM/fabric read and write bytes, aggregated over all relevant instances/XCCs |
| Predicated FP32/FP64 ADD, MUL, FMA instruction counts | Precision-specific VALU arithmetic counts, converted using the documented unit and active lanes; FMA counts twice |

A convenient supported route is ROCm Compute Profiler (formerly Omniperf):

```bash
rocprof-compute profile -n atmos_c24 -- \
  julia --threads=8 --project=rocm-env "$SCALING/benchmark.jl" \
  "$SCALING/configs/run_C24_T1.toml" "$SCALING/results/compute" timing
rocprof-compute analyze -p workloads/atmos_c24/<device-directory>
# Older installations use the analogous executable:
# omniperf profile -n atmos_c24 -- <command>
# omniperf analyze -p workloads/atmos_c24/<device-directory>
```

Use its precision-specific VALU FLOP and HBM byte metrics and restrict analysis
to warmed dispatches. Roofline availability differs across releases, especially
for MI350; use a release supporting your device. See
[Compute Profiler profile mode](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.1.0/how-to/profile/mode.html)
and [analysis](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.0.0/how-to/analyze/cli.html).

For raw counters, inspect definitions such as `SQ_INSTS_VALU_*` and
`TCC_EA_*` on the target device. Do not blindly multiply every instruction
counter by 64: some metrics already count operations, and wave-instruction
counts may include masked lanes. Likewise read/write requests can have
mixed 32/64-byte transaction sizes; use the provided byte formula and sum
instances once. These transport kernels use scalar/vector arithmetic, so
report VALU rather than MFMA/tensor peak performance. If the hardware cannot
provide active-lane counts, label the inferred FLOP count as an estimate.

For each measured step, report FLOPs, HBM bytes, and summed kernel seconds.
Arithmetic intensity is FLOPs/bytes; achieved FLOP/s and byte/s divide totals
by kernel seconds. Divide counts by the recorded number of measured steps.
Keep wall-clock step time separately, since overlapping dispatches make summed
kernel time differ from elapsed time. Archive raw traces, counters, formulas,
config, manifest, tool versions and device partition mode alongside results.
