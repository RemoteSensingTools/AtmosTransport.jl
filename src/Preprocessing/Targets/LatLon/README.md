# Preprocessing.LatLonTarget

Lat-lon spectral workspaces, humidity staging, contracts and the daily binary workflow.

## File map

- [`LatLon.jl`](LatLon.jl) — module imports, exports and load order.
- [`latlon_contracts.jl`](latlon_contracts.jl) — LL balance, replay, positivity and serialization.
- [`latlon_spectral.jl`](latlon_spectral.jl) — daily LL workflow and deferred writer.
- [`latlon_workspaces.jl`](latlon_workspaces.jl) — spectral/QV staging, storage and merge stages.

## Depends on

Common, Vertical/Spectral, MassClosure, PhysicsInputs and TransportBinary; lower binary IO, replay, mass basis, atomic publication, FFTW and NetCDF/GRIB APIs. No parent, source, RG or CS implementation.

## Used by

LL spectral preprocessing; CS spectral preprocessing reuses its LL staging adapter; RG and CS reuse its progress/mass-pin logging. Public names remain reachable through Preprocessing.

## Shared helpers owned here

`SpectralTransformWorkspace`, `MergeWorkspace`, `WindowStorage`, QV workspaces and interpolation; the workspace overload of `spectral_to_native_fields!`; mass-pin/dry/merge stages; `LatLonContract`, LL gates, accumulator wrappers and serializer; `process_day`; `should_log_window`, `log_mass_fix_configuration`.

## Forward<->adjoint mirrors

No adjoint. LL positivity delegates to the shared rectangular-block palindrome scan in TransportBinary; its replay layout and target-scale conversion stay unchanged.

- [`tm5_pbl_writer.jl`](tm5_pbl_writer.jl) — shared LL/RG opt-in T/q and surface
  input assembly (`load_tm5_pbl_target_window`, `tm5_pbl_sample`,
  `tm5_target_coordinates`). The v5 section group is appended by the existing
  writer when `[preprocessing] include_tm5_pbl=true`. Thermo T/q use target
  point interpolation and moist-mass-weighted vertical merging, through the
  existing `LatLonBilinearWeights`/`interpolate_ll_qv!` sampler (Cartesian or
  paired points). All surface primitives use `Common.target_surface_means`
  and the shared conservative regridder on the actual LL/RG mesh.
  `TM5PBLInputCache` owns day-local surface readers, remap weights and the shared
  thermo endpoint; day drivers close it even on failure.
