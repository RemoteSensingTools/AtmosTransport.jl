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
GEMM and unnormalised FFT plans; CUDA orders KA/GEMM/FFT on the task stream
with one completion wait per synthesis call), `synthesis_kernels.jl` (portable pack,
scatter and gather), and `spectral_quadrature.jl` (refinement, linear/cos-linear
latitude rules, open polar nodes and periodic prefix primitives).
`CPURingSynthesis` is the reference; `GPUSpectralSynthesis` keeps samples on
the device. CUDA selection, automatic precision and the TF32 guard live in
`ext/AtmosTransportCUDAExt.jl`; core never loads CUDA. LL/RG surface fields
and quadrature follow the selected precision throughout; the existing host
pin/closure uses Float64 endpoint pressure after transfer. CPU CS retains Float64 surface fields; device CS surfaces follow selected precision.

- `spectral_hour_io.jl`: indexed one-hour reads and per-hour atomic caches.
- `spectral_integral_fluxes.jl`: PS, face and C integration and independent W.
- `latitude_band_integrals.jl`: Gauss/Fourier execution in complete latitude
  cell bands (a total budget of 32 cells), sliced from the original quadrature. Packed native coefficients
  are reused across bands; compatible bands reuse synthesis/FFT buffers and
  refill bounded Legendre tables. Float32 device polar plans are built once per
  compatible band shape and share bounded latitude tables across shapes; sample
  scratch persists, and each endpoint stages its polar coefficients once per
  level batch. Sequential Fourier operations share scratch.
- `cpu_band_synthesis.jl`: host scheduling for small bands. A reusable single
  mode panel preserves the original surface GEMM dimensions and column positions
  so that streaming does not change its accumulation order. Native fields use
  fixed 32-column GEMMs anchored to global latitude indices, with padded partial
  panels in both full and streamed execution. This prevents CPU edge
  microkernels from changing rounding with band size, including Float32.
  Gauss/Fourier CPU FFTs use a fixed one-ring batch through aligned scratch,
  including Fourier integrals; changing latitude batch shapes cannot select
  different FFTW codelets. This correction can change default Gauss/Fourier
  golden payloads at rounding level. Explicit trapezoids are unchanged.

CPU Gauss/Fourier uses up to four band workers, dividing the cell budget
between them, with a fixed two FFT threads per worker, independent of Julia
thread count or host capacity. Plan-local thread counts
preserve the global FFTW setting. CPU trapezoids
retain their existing FFT threading policy. GPU execution is not established
by the CPU memory and bitwise measurements.
- `tm5_endpoint.jl`: one daily `TM5EndpointPlan`, dry pin and moist reduction;
  endpoints discard native C after constructing retained W.

# shared: SpectralIntegralWorkspace, integrate_surface_pressure!, integrate_face_fluxes!, integrate_vertical_flux!, TM5EndpointPlan, construct_tm5_endpoint, with_spectral_hours, spectral_hour_cache_path, spectral_synthesis_metadata

`read_qv_from_thermo` accepts `variable="t"` for the opt-in live-Calc_Kzz
input writer; the default `variable="q"` retains the original humidity path.
The scalar reader and coordinate reader also accept an open `NCDataset`.
`foreach_thermo_record` reads a bounded consecutive block and delivers each
normalized record to a callback, preserving scalar conversion/orientation.
The live PBL writer uses this to decode each temporal chunk once during a
sequential day traversal of the measured three-hour chunk layout, retaining
only interpolated target endpoints. The three-record cap can cause overlapping
chunk reads for other layouts, such as four-hour chunks; skipped or revisited
windows can also reread chunks.

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

## Direct TM5 cubed-sphere integrals

- `cs_point_synthesis.jl`: selected-precision, batched KA direct point reference
  using stable vector harmonics and compensated Fourier accumulation.
- `cs_polar_synthesis.jl`: CS-only stable derivative-based vector transforms
  on polar rings, avoiding cancellation in small pseudo-winds.
- `cs_spectral_integrals.jl`: `CSSpectralQuadrature` selects regular open
  refined rings from the smallest target edge and retained truncation;
  `CSSpectralIntegralWorkspace` applies great-circle midpoint/cardinal16 Cartesian face maps and
  conservative box-overlap maps with KernelAbstractions gathers. The target
  supplies geometry matrices, keeping this stage independent of MassClosure
  and Regridding. Only completed reduced target fields return from the device at the endpoint boundary.
- CPU CS surface synthesis and PS integration use Float64; device surface work
  follows selected precision. CS 3-D integrands and
  gathers use the selected plan precision, including typed constants; the
  new kernels introduce no Float64 promotion. The forcing is a density per
  square metre before conservative area integration, with the IFS/target
  radius ratio retained.
- `construct_tm5_endpoint` shares the same pin, moist merge and W selection
  for LL/RG and CS. CS stores every unique canonical face in `u`, with a
  zero-row `v`; the target owns panel expansion and seam synchronization.

# shared: CSSpectralQuadrature, CSSpectralIntegralWorkspace

`cs_face_gather.jl` stores one scalar stencil and normal per face, with
component-major accumulation identical to the expanded Cartesian map. Float32
uses checked Int8 bit offsets to reproduce rounding of each original Float64
geometry product without Float64 device arithmetic.

# shared: CSFaceGather

`gauss_fourier.jl` owns the Gauss–Legendre rule and Fourier interval factors
shared by production and the §13 diagnostic. Its portable FFT pack/fold/gather
integrates periodic products on arbitrary LL/RG longitude segments.

# shared: gauss_legendre, fourier_interval_factor, fourier_integral_plan, fourier_integrals!

The Gauss/Fourier path packs U/V/D coefficients into one transform batch,
sharing Legendre work and launch overhead while keeping the native-level
batch bound. Explicit trapezoids retain their historical transform sequence.

## Device face quadrature

`device_face_quadrature.jl` owns backend-dispatched level padding and surface
gradient coefficients, and a typed Neumaier accumulator. Gauss latitude sums,
Fourier frequency folds and CS cardinal/overlap gathers use compensation for
Float32 device arrays. CPU arrays retain their previous reduction arithmetic,
aligned GEMM panels and fixed FFT batches. All new kernels are exercised on
the KA CPU backend, including explicit compensated-reduction tests.

Reduced outputs stay on the synthesis backend across level batches and latitude
bands. `integrate_face_fluxes!(...; host_output=false)` retains backend arrays;
the default transfers completed PS/face/forcing fields once for the established
Float64 host pin, balance and closure. Host geometry, sparse stencil assembly
and Float64 Legendre precomputation remain setup work. CS polar coefficients
are formed in a device kernel. Float32 device LL/RG plans reuse the stable
polar vector-harmonic correction; CPU production plans retain their bits.
CS GPU surface synthesis and overlap means now follow selected precision,
so an L40S plan performs no Float64 device work. CPU CS surfaces retain Float64.
Pedantic CUDA math spans the complete face construction, including Fourier
integration; FAST_MATH/TF32 is rejected.

GPU repeatability is scoped to the same device, library stack and plan shapes;
no cross-device bitwise claim is made. Changing bands can select different
CUBLAS/CUFFT algorithms: band-size independence is measured numerically, not
promised bitwise. Compensation does not remove coefficient, GEMM, FFT or
pressure-exponential rounding. A near-cancelling small harmonic fixture has
about 2.2e-5 relative Float32 face error; its smoke tolerance is not a 1e-6
certificate. The qualification driver reports every face outside the bound.
See manual §17 and `scripts/validation/README.md` for the L40S procedure.

Float32 device pressure synthesis separates the constant LNSP coefficient
before transforming the anomaly, then restores its pressure factor. This
avoids losing small spatial pressure signals to rounding the full logarithm.
The scalar reference exponential is prepared on the host; device arithmetic
remains Float32. Compensated face sums also retain real multiplication
residuals with Float32 `fma`, including complex Fourier products. CPU dispatch
retains the original products and additions.

Measured L40S complete-window CPU/GPU means are 118.30/84.81 s
for LL 1° and 154.37/41.50 s for C90. Balanced T639 cm relative L2
is 1.87e-07/1.97e-07. [Qualification details and memory scopes](../../../docs/src/preprocessing/tm5_spectral_fluxes.md#l40s-qualification-2026-11-03)
also retain the failing edge-harmonic maximum bounds.
