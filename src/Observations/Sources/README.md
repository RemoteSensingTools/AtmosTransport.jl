# Sources

Typed observation sources, requests, schedules and file readers.

## File map

- `Sources.jl`
- `observation_readers.jl`
- `observation_requests.jl`
- `observation_sources.jl`

## Depends on

Retrievals, Grids, State, Parameters and ConfigChecks.

## Used by

Sampling and the inversion MIP converter.

## Shared helpers owned here

`pressure_unit_scale`, `parse_iso_utc`, `isvalid_lonlat`, `nc_variable`,
`raw_observation_block`, source/request types and reader entry points.

# shared: _is_plain_int _parse_choice

## Forward<->adjoint mirrors

No adjoint implementation; retrieval fields feed the shared operator.
