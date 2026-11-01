# Inversion observations

`AtmosTransport.Adjoints.Inversion.ObservationBinding` owns observation records,
binding, sampling and IO. It is distinct from `AtmosTransport.Observations`,
which owns the forward sampler, placement and retrieval primitives.
The child name describes the inversion observation flow from records and
columns to bound samples, departures and OCO-2 MIP IO, and avoids shadowing
the package-level module during static inference on Julia 1.10.

## File Map

- `ObservationBindingModule.jl`: module imports, includes and exports.
- `Observations.jl`: time sampling, bound observations, placement codes and stencils.
- `ObservationColumns.jl`: meteorology and objective sampling/stencil construction.
- `ObservationBinding.jl`: records → cells, time windows, flags and cell averages.
- `ObservationsIO.jl`: observation records/sets and schema-v2 NetCDF IO.
- `DeparturesIO.jl`: departure records/sets, joins and schema-v2 NetCDF IO.
- `OCO2MIPObservations.jl`: OCO-2/OCO-3 MIP conversion.
- `ForwardSampling.jl`: `_observation_replay` (checks the observations against
  the run and builds the replay through the last sampled step) and
  `_observe_cs_forward!`, the sampled forward run of a checkpoint driver's run
  (untaped, or recording the tape of the 4D-Var reverse pass).

## Depends on

ConfigChecks, Adjoints core (objectives/functionals), Footprint (replay), Grids, State,
Parameters and AtmosTransport.Observations (placement, retrievals, reader and
writer primitives). External: Dates, NCDatasets. No dependency on Solver.

## Used by

Inversion.Solver, the inversion driver, OCO-2 converter and observation tests.

## Shared helpers owned here

Observation/record/set/stencil types, binding and observation/departure IO.
Solver imports `_observation_replay` and `_observe_cs_forward!`;
`build_departure_set` has a result-type adapter defined in Solver to keep the
dependency one-directional.

## Forward<->adjoint mirrors

Sampling calls the forward placement and retrieval primitives and builds
sparse functionals in the same arithmetic as before. The sampler reads the
state after each step's last stage (after the tape recorder's end-of-step
marker) and does not change it, so a forward run that also records the tape
samples the same values. A time window weights
end-of-step states by overlap/total overlap. Non-assimilated records are
simulated but do not enter the cost or forcing. Departure reconstruction uses
the binding's clamp-policy sampling, including terminal/outside times.
The two ISO parsers have different accepted inputs and remain separate.

# shared: CELL_AVERAGE_PLACEMENT_CODE, CSDepartureSet, CSObservation, CSObservationSet, CSObservationStencil, ClampOutOfRange, _observation_replay, _observe_cs_forward!, bind_to_mesh, build_departure_set

ConfigChecks.parse_choice owns binding-policy and kind parsing. Typed policy
instances pass through; String choices remain case sensitive and reject Symbols.
