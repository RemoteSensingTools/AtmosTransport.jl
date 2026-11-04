
`bench_gpu_face_window.jl cpu|gpu ll|cs OUTPUT` warms and measures complete
TM5 00–01 UTC windows on LL 1° or C90. It uses production endpoint, balance,
closure/replay, positivity and write gates, with the day loop shortened only
in memory. GPU runs use Float32 surfaces and fields. The L40S runner and
metric/memory definitions are documented in `../validation/README.md`.

The qualification summary compares the written am/bm/cm payloads with the
CPU Float64 build of the identical T639 window. Timings use two warmed
repetitions, with separate active CUDA-pool and sampled whole-device memory
scopes. The launcher can wait for a CPU correctness-job barrier before timing.

`GPUQUAD_MEASURED_REPS=0` builds only the warmup window for a short
accuracy investigation; the qualification launcher forces two measured
repetitions and rejects incomplete timing files.

Measured L40S complete-window CPU/GPU means are 118.30/84.81 s
for LL 1° and 154.37/41.50 s for C90. Balanced T639 cm relative L2
is 1.87e-07/1.97e-07. [Qualification details and memory scopes](../../docs/src/preprocessing/tm5_spectral_fluxes.md#l40s-qualification-2026-11-03)
also retain the failing edge-harmonic maximum bounds.
