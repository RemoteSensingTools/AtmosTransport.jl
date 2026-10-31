# Preprocessing.ReducedGaussianTarget

Reduced-Gaussian spectral workspaces, humidity staging, sliding windows and contracts.

## File map

- [`ReducedGaussian.jl`](ReducedGaussian.jl) — module imports, exports and load order.
- [`reduced_gaussian_contracts.jl`](reduced_gaussian_contracts.jl) — RG stub, replay and per-pass positivity gates.
- [`reduced_spectral_day.jl`](reduced_spectral_day.jl) — RG synthesis, pin/dry/merge, balance and day driver.
- [`reduced_transport_helpers.jl`](reduced_transport_helpers.jl) — ring synthesis, QV, flux sampling and layer merge.
- [`reduced_window_buffer.jl`](reduced_window_buffer.jl) — two-slot window lifecycle and stored-array contracts.

## Depends on

Common, Vertical/Spectral, MassClosure and TransportBinary; LatLonTarget logging; lower grid/replay/writer APIs, FFTW and NetCDF. No parent, source or CS implementation.

## Used by

RG spectral preprocessing through the Preprocessing facade. The deferred ERA5 reader also uses the ring helpers/thread cache until PP4.

## Shared helpers owned here

Ring synthesis/cache helpers, `ReducedFidelitySampler`, RG workspaces, `merge_field_2d!`, pressure-pin rescaling, `SlidingWindowBuffer`, `balance_window!`, `ReducedGaussianContract`, stub/positivity/replay gates and `process_day`.

`allocate_reduced_spectral_thread_cache` owns the ring-cache allocation body;
RG and ERA5 keep their distinct thread-pool sizing policies.

## Forward<->adjoint mirrors

No adjoint. RG checks each horizontal/vertical pass separately and skips polar stub faces; its scan must remain distinct from LL/CS. Stored cm is diagnosed after column balance; native/merged cm scratch is unnecessary.
