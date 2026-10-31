# Targets

The three target modules are direct children of `Preprocessing`. Their physical
folders group target-specific work without introducing a wrapper module.
The `Target` suffix belongs to the module names; existing folders and entry
filenames stay in place to avoid unrelated file moves.

## File map

- [LatLonTarget](LatLon/README.md) — LL spectral staging, contracts and day workflow.
- [ReducedGaussianTarget](ReducedGaussian/README.md) — RG synthesis, windows and contracts.
- [CubedSphereTarget](CubedSphere/README.md) — CS spectral/regrid/coarsen workflows and contracts.

## Depends on

Common → Vertical → Spectral → MassClosure/PhysicsInputs supply the earlier
pipeline stages. TransportBinary owns the shared window/writer interfaces and
loads before these target implementations. RG and CS reuse LL staging/logging;
LL imports neither sibling. No target imports its parent or a source reader.

## Used by

The Preprocessing facade and the deferred native-source drivers. All retained
public bindings remain reachable through Preprocessing.

## Shared helpers owned here

Each child README lists its own API. This umbrella folder defines no bindings.

## Forward<->adjoint mirrors

No adjoint. Shared LL/CS positivity lives in TransportBinary; RG retains its
separate per-direction positivity contract.
