# Core tests

Run each focused file in its own process with `julia --startup-file=no
--project=test test/core/FILE.jl`; see the parent README for environment setup.

Spectral backend tests execute the portable face-staging, polar-coefficient,
Gauss/Fourier and compensated CS gather kernels on KernelAbstractions.CPU.
Pressure-anomaly tests check accurate Float32 exp synthesis against Float64;
FMA cancellation fixtures verify multiplication residuals independently of
summation compensation.
LL/RG complete integration checks include level padding, latitude bands and
resident results. CS checks cover selected-precision surfaces and cardinal-16
maps. CPU fixed-panel and fixed-FFT tests retain their bitwise assertions.
GPU qualification, real-hour accuracy and performance require the separate
owner-run procedure in `scripts/validation/README.md`.

Measured L40S T255/T639 errors, bitwise repetitions and the remaining
edge-harmonic maximum-bound failures are in the spectral manual §17. Full-day
LL72, O24 and C24 CPU goldens remain bitwise identical to `origin/main`.

The backend suite launches `test/helpers/spectral_face_device_policy.jl` in an
isolated Julia process to enable device staging, polar coefficients, pressure
anomalies and compensated Float32 reductions together on KA.CPU. It covers
streamed/resident LL/RG, CS, plan reuse and non-unit-product cancellation in
a launched gather without changing another test’s CPU policy.
