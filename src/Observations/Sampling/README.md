# Sampling

Containing-cell location, vertical placement, column gathers and runtime observation output.

## File map

- `Sampling.jl`
- `cell_locator.jl`
- `observation_gather.jl`
- `observation_netcdf.jl`
- `observation_output_spec.jl`
- `observation_placement.jl`
- `observation_sampler.jl`

## Depends on

Sources, Retrievals, Grids, State, Parameters, ConfigChecks and Output.

## Used by

Models runner, GriddedIO terrain readers and Inversion.

## Shared helpers owned here

Placement, layer reconstruction, gather and sampler entry points are exported
and re-exported by Observations. The NetCDF streams use Output's shared lock.

## Forward<->adjoint mirrors

The host and device gather use identical indices. Inversion shares sample_column_layer! and the retrieval-column operator with the forward sampler.
