# Inversion observations

`AtmosTransport.Adjoints.Inversion.Observations` owns observation records,
binding, sampling and IO. It is distinct from `AtmosTransport.Observations`,
which owns the forward sampler, placement and retrieval primitives.

## File Map

- `ObservationsModule.jl`: module imports, includes and exports.
- `Observations.jl`: time sampling, bound observations, placement codes and stencils.
- `ObservationColumns.jl`: meteorology and objective sampling/stencil construction.
- `ObservationBinding.jl`: records → cells, time windows, flags and cell averages.
- `ObservationsIO.jl`: observation records/sets and schema-v2 NetCDF IO.
- `DeparturesIO.jl`: departure records/sets, joins and schema-v2 NetCDF IO.
- `OCO2MIPObservations.jl`: OCO-2/OCO-3 MIP conversion.
- `ForwardSampling.jl`: `_observe_cs_forward`, sampled production replay.

## Depends on

Adjoints core (objectives/functionals), Footprint (replay), Grids, State,
Parameters and AtmosTransport.Observations (placement, retrievals, reader and
writer primitives). External: Dates, NCDatasets. No dependency on Solver.

## Used by

Inversion.Solver, the inversion driver, OCO-2 converter and observation tests.

## Shared helpers owned here

Observation/record/set/stencil types, binding and observation/departure IO.
Solver imports `_observe_cs_forward`; `build_departure_set` has a result-type
adapter defined in Solver to keep the dependency one-directional.

## Forward<->adjoint mirrors

Sampling calls the forward placement and retrieval primitives and builds
sparse functionals in the same arithmetic as before. A time window weights
end-of-step states by overlap/total overlap. Non-assimilated records are
simulated but do not enter the cost or forcing. Departure reconstruction uses
the binding's clamp-policy sampling, including terminal/outside times.
The two ISO parsers have different accepted inputs and remain separate.

# shared: CELL_AVERAGE_PLACEMENT_CODE, CSDepartureSet, CSObservation, CSObservationSet, CSObservationStencil, ClampOutOfRange, _observe_cs_forward, bind_to_mesh, build_departure_set
