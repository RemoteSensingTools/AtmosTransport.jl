# Retrievals

Retrieval columns and averaging-kernel projection.

## File map

- `Retrievals.jl`
- `retrieval_column.jl`

## Depends on

Grids, State and Parameters.

## Used by

Sources, Sampling and Inversion.

## Shared helpers owned here

`RetrievalColumn`, `retrieval_profile!`, `retrieval_column_weights!`,
`retrieval_column_value`, unit scales and mapping parsers.

## Forward<->adjoint mirrors

The projection and its linear weights are shared by forward sampling and adjoint seeding.
