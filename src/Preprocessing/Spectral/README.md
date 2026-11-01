# Preprocessing.Spectral

ERA5 spectral input, LL and batched reduced-Gaussian synthesis, humidity and
native mass support. Spectral files use one `era5_spectral_paths` constructor.
FFTW synthesis keeps the unnormalised inverse transform and all existing
floating-point order and conversions.

## File map

- `Spectral.jl`: module imports and shared interface.
- `spectral_io.jl`: GRIB coefficient decoding and atomic daily cache.
- `discovery.jl`: available dates and next-day hour-zero endpoint.
- `spectral_synthesis.jl`: Legendre recurrence, wind synthesis, LL mass/fluxes.
- `reduced_spectral_synthesis.jl`: batched reduced-Gaussian synthesis for N320.
- `mass_support.jl`: extensive merge adapter, pressure overlap, LL Poisson
  balance, thermo QV readers, drying and global mass pins.

## Depends on

Common geometry, humidity policy and column-correction protocols; Vertical's
extensive reduction; Grids, Parameters, GriddedIO, AtomicFiles; Dates, Logging,
Printf, FFTW, GRIB/ecCodes, SHA, JLD2, NCDatasets, LinearAlgebra and
KernelAbstractions.
Closure implementations register Common methods in the parent; Spectral does
not import the parent or another later stage.

## Used by

The root's RG helpers, topology workspaces, source readers and day workflows.

## Shared helpers owned here

The exports in `Spectral.jl` declare synthesis, spectral paths/cache,
pressure-overlap remap, dry-basis conversion, humidity readers, global mass
pinning, LL balance and mass-target support. Private implementation helpers
stay within this module.

`era5_thermo_path` formats daily thermodynamic filenames for binary and target workflows.

## Forward<->adjoint mirrors

None.

# shared: synthesize_native_fields, SynthesisRings, quadrature_refinement, oversampled_ring_length, lnsp_gradient, complete_wind_coefficients, RingSynthesisPlan, CPURingSynthesis, GPUSpectralSynthesis, synthesize_rings, surface_synthesis, spectral_gpu_adapter, spectral_gpu_precision, spectral_gpu_math_guard, spectral_synthesis_guard, SpectralQuadrature, latitude_quadrature, ring_primitive, ring_prefix!

TM5 ring evaluation uses `synthesis_rings.jl` (geometry, exact LNSP gradient and
complete T+1 wind transform), `device_spectral_synthesis.jl` (bounded batch
GEMM and unnormalised FFT plans), `synthesis_kernels.jl` (portable pack,
scatter and gather), and `spectral_quadrature.jl` (refinement, linear/cos-linear
latitude rules, open polar nodes and periodic prefix primitives).
`CPURingSynthesis` is the reference; `GPUSpectralSynthesis` keeps samples on
the device. CUDA selection, automatic precision and the TF32 guard live in
`ext/AtmosTransportCUDAExt.jl`; core never loads CUDA. Surface fields and
quadrature accumulation use Float64. This exception to runtime Float32 kernel
policy applies only to offline spectral preprocessing.

- `spectral_hour_io.jl`: indexed one-hour reads and per-hour atomic caches.
- `spectral_integral_fluxes.jl`: PS, face and C integration and independent W.
- `tm5_endpoint.jl`: one daily `TM5EndpointPlan`, dry pin and moist reduction;
  endpoints discard native C after constructing retained W.

# shared: SpectralIntegralWorkspace, integrate_surface_pressure!, integrate_face_fluxes!, integrate_vertical_flux!, TM5EndpointPlan, construct_tm5_endpoint, with_spectral_hours, spectral_hour_cache_path, spectral_synthesis_metadata

`read_qv_from_thermo` accepts `variable="t"` for the opt-in live-Calc_Kzz
input writer; the default `variable="q"` retains the original humidity path.
