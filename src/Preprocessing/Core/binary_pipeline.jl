# Shared pipeline interfaces. Deferred source/closure stages add methods in
# Preprocessing; Common itself depends only on these contracts.
"""
How a column mass-budget correction is spread over the levels of a column.

- `MassWeightedColumn()`: in proportion to layer air mass (default).
- `HybridBWeightedColumn(B)`: in proportion to `ΔB_k`, the layer's share of a
  surface-pressure change, as in TM5. Pure-pressure layers (`ΔB = 0`, the
  stratosphere) receive none of it, so a column mismatch cannot appear in
  their vertical mass flux as a coherent mode proportional to pressure.
- `HybridMassWeightedColumn(B)`: in proportion to layer air mass, but only in
  hybrid layers, `ΔB_k ≥ HYBRID_LAYER_MIN_DB` (1e-6); pure-pressure layers,
  including near-pure-pressure ones such as GEOS L72 layer 41 (`ΔB = 8.2e-9`),
  receive none of it.
"""
abstract type AbstractColumnWeights end
"""Spread a column correction in proportion to nonnegative air mass."""
struct MassWeightedColumn <: AbstractColumnWeights end

"""Construct configured column weights for hybrid interfaces `B`."""
function column_weights end
"""Spread one face's column correction over its layers."""
function spread_column_correction! end
"""Validate that column weights match the number of layers."""
function check_column_weight_levels end
"""Allocate target cell degrees and balance scratch from a face table."""
function target_balance_geometry end
"""Read native ERA5 reduced-Gaussian geometry from GRIB."""
function read_era5_reduced_gaussian_geometry end
"""Read a native ERA5 reduced-Gaussian mesh from GRIB."""
function read_era5_reduced_gaussian_mesh end

# Float64 policy value: every caller retains its existing conversion to FT.
const MAX_SPECIFIC_HUMIDITY = 0.999999
const COLUMN_WEIGHT_NAMES = (:mass, :hybrid_b, :hybrid_mass)

const PACKAGE_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
package_path(relpath::AbstractString) = normpath(joinpath(PACKAGE_ROOT, relpath))

"""Reserve a unique staging path for a preprocessor writer."""
function open_preprocessing_staging(path::AbstractString)
    staging, io = open_atomic_staging(path)
    close(io)
    return staging
end
