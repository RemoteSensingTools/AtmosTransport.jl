# Observations

`AtmosTransport.Observations` owns retrieval columns, observation sources and
requests, containing-cell location, vertical placement, column sampling and
append-only observation NetCDF output. It loads after Output and before
Adjoints. Public observation names remain reachable from the package facade
and are imported from `AtmosTransport.Observations` by callers.

## File map

- `Observations.jl`: assembles and re-exports the three submodules.
- [`Retrievals/`](Retrievals/README.md): retrieval columns and averaging kernels.
- [`Sources/`](Sources/README.md): typed sources, requests and file readers.
- [`Sampling/`](Sampling/README.md): location, placement, gather, configuration,
  runtime sampler and observation NetCDF.

## Depends on

Retrievals, Sources and Sampling, in that order. Their dependencies are lower
modules: Grids, State, Parameters, ConfigChecks and Output. GriddedIO loads
later and supplies concrete terrain readers through this module's interfaces.

## Used by

Models/runner, Adjoints (including Inversion and Footprint), GriddedIO terrain
readers, and inversion scripts.

## Shared helpers owned here

`pressure_unit_scale(units, label)` scales declared pressure units to Pa.
`parse_iso_utc`, `parse_vertical_placement`, `VERTICAL_PLACEMENT_KEYS`,
`handle_unplaced`, `count_unplaced!`, `unplaced_example`, `warn_below_ground`
and `ObservationCounters` are the shared config and placement interfaces.
`nc_variable` and `raw_observation_block` share the observation fill/packing
policy with the MIP converter. `mole_fraction_scale` and
`parse_retrieval_mapping` share retrieval-file interpretation with Inversion.
The 280 K fallback is `Parameters.DEFAULT_OBSERVATION_LAYER_TEMPERATURE_KELVIN`.

## Forward<->adjoint mirrors

The forward sampler and Inversion use `sample_column_layer!` for placed points
and the same retrieval-column operator for averaging-kernel columns. Keep
Float64 top-down pressure accumulation, temporal mass blends, and projection
quadrature unchanged. The gather host/device index expressions must match.

## Validation and follow-ups

Observation sampling relies on focused numerical and IO tests; it has no
whole-run golden coverage. Snapshot goldens are run by the main session.
CF time decoding remains here: sharing it with GriddedIO requires a lower
leaf module and a decision on calendars and fill handling. The inversion ISO
parser, MIP time-source policy and halo-width policy remain separate because
unifying them changes accepted input or arithmetic. NetCDF attribute order,
variable definitions, lock/queue behavior and placement codes are unchanged.
