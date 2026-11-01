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

`tm5_spectral_day.jl` dispatches `TM5SpectralFluxes` through retained-spectrum
integration, post-integration dry pin, moist layer balance, target-layer drying
and window means, then the existing column repair, explicit-dm closure,
replay/positivity and binary writer. Next-day 00 UTC is mandatory. Positivity
retries rebuild amounts from pristine rates. The new construction writes v5;
`PointSampledSpectralFluxes` retains the legacy v4 path.

`tm5_layer_balance.jl` owns the common TM5 endpoint averaging and guard plus
LL geometry methods. RG implements its geometry methods in its own target.
`tm5_spectral_day.jl` reads one hour at a time, retains two moist endpoints,
closes one window at a time and retains one completed lookahead window for
the historical delta payload. The optional spectral cache is per hour.

# shared: directional_fluxes, tm5_horizontal_convergence, balance_tm5_layers!, tm5_balance_diagnostics, dry_target_fluxes!
