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

"""
    check_fixed_global_mass(m, endpoint_or_tendency; applications=nothing, mass_type)

Reject a fixed global dry-mass budget if `abs(Σ(m_next-m))/Σm` exceeds
`GLOBAL_DRY_MASS_BUDGET_RTOL`. With `applications`, the second argument is a
per-application tendency and the numerator is `abs(applications * Σdm)`.
Both reductions are compensated Float64 sums of stored values (including CS
panel tuples). Expected global change is zero. Call only for a fixed dry target;
moist and unpinned windows retain their physical global tendency. Stored lower
precision endpoints allow `2eps(mass_type)` for their independent rounding.
"""
function check_fixed_global_mass(m, endpoint_or_tendency;
                                applications::Union{Nothing,Int}=nothing,
                                mass_type::Type=eltype(m isa Tuple ? first(m) : m))
    panels = m isa Tuple ? m : (m,)
    other = endpoint_or_tendency isa Tuple ? endpoint_or_tendency : (endpoint_or_tendency,)
    indices = ((p, i) for p in eachindex(panels) for i in eachindex(panels[p]))
    total = +(_neumaier_sum(indices, Float64) do (p, i)
        Float64(panels[p][i])
    end...)
    change = +(_neumaier_sum(indices, Float64) do (p, i)
        applications === nothing ? Float64(other[p][i]) - Float64(panels[p][i]) :
                                   Float64(other[p][i])
    end...)
    applications === nothing || (change *= applications)
    relative = abs(change) / total
    tolerance = max(GLOBAL_DRY_MASS_BUDGET_RTOL, 2 * Float64(eps(mass_type)))
    isfinite(total) && total > 0 && isfinite(relative) && relative <= tolerance ||
        error("fixed global dry-mass budget mismatch: |ΔM|/M=$relative exceeds $tolerance " *
              "(ΔM=$change kg, M=$total kg; expected global change is zero)")
    return relative
end
