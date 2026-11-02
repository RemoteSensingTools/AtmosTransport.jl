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

## Dry-mass rounding and column closure

`pin_global_mean_ps_using_qv!` compensates native-layer dry-pressure and
sensitivity sums and every global reduction. The target difference retains
low parts; the routine allocates no scratch and shares no mutable state.
Point-sampled LL, RG and the LL staging of spectral CS, plus TM5 LL/RG endpoints,
all use it. LL column balance leaves its unresolved global mode proportional
to current column mass, diagnoses vertical closure against the original
endpoint tendency, and retains the Float64 replay tolerance of `1e-12`.
All spectral `pre_*` goldens are expected to change at rounding level.
See the manual's mass-conservation page for equations and full-day evidence.

On ERA5 2021-12-01 (1°, Float64 dry mass, 34 levels, 24 windows), replay
maxima decrease from 1.471e-12 to 5.823e-15 for TM5 and from
1.064e-12 to 6.107e-15 for point-sampled construction. Counts above
1e-12 decrease from 536 and 6 to zero. Maximum |global ΔM| per window decreases
from 135,695.388 to 666.004 kg (TM5), and from
121,982.453 to 659.033 kg (point-sampled).

The mass-proportional global-mode shift is enabled only for fixed dry-mass
targets, after Common's compensated mass-normalized budget guard (`1e-14`
for Float64; lower precision allows two-endpoint storage rounding). Moist and
unpinned paths retain their uniform global mode and original endpoint tendency.
TM5 LL/RG layer solves and standalone CS per-level solves retain original
projection arithmetic; only column calls select compensated projection dispatch.
