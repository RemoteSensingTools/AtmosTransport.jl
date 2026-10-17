# ===========================================================================
# Regridding of lat-lon surface-flux inventories onto the model mesh
#
# A static inventory (`FileSurfaceFluxField`) goes onto the target mesh by the
# method of `surface_flux.regridding` (a type, `_regridding_method`):
#
#   - `ConservativeFluxRegridding` (default, every topology): the per-cell rate
#     m_s = F_s A_s with the source cell areas A_s (the file's, else the source
#     mesh's) is regridded as an extensive field over the area of each source
#     cell that the target covers, so the target total is the file's own total
#     Σ_s F_s A_s.
#   - `BilinearFluxRegridding` (lat-lon and reduced-Gaussian targets, explicit
#     only): the density is sampled at the target cell centres, and the
#     emitting and the uptake cells are renormalised separately to the source's
#     emission and uptake totals, or the build is refused.
#
# Time-varying series (cubed sphere only) are regridded slice by slice as
# densities with one prebuilt regridder (`_apply_surface_flux_regridder`).
# ===========================================================================

"""
    AbstractFluxRegridding

How a lat-lon surface-flux inventory is put onto the model mesh
(`surface_flux.regridding`): [`ConservativeFluxRegridding`](@ref) (the
default) or [`BilinearFluxRegridding`](@ref).
"""
abstract type AbstractFluxRegridding end

"""
    ConservativeFluxRegridding()

`regridding = "conservative"`, the default on every topology: the source's
per-cell rate ``m_s = F_s A_s`` [kg s⁻¹] is regridded as an extensive field,

```math
r_t = \\sum_s \\frac{W_{ts}}{C_s}\\, m_s, \\qquad C_s = \\sum_t W_{ts}, \\qquad
\\sum_t r_t = \\sum_s F_s A_s ,
```

with the intersection areas ``W_{ts}`` of the conservative regridder and the
covered area ``C_s`` of each source cell (`Regridding.source_coverage`; the
great-circle polygon area of the cell when the target tiles the sphere).
``A_s`` is the file's `cell_area`/`area` variable, else the latitude-circle
area of the source mesh cell, so every target total equals the file's own
total to rounding, also on reduced-Gaussian targets whose great-circle cells
leave slivers uncovered.

On a target that does not cover the sphere (a regional or latitude-truncated
mesh; the meshes of the shipped runs are global), a source cell that the target
covers only in part still puts its whole rate ``m_s`` into the covered part
(its density there is ``m_s / C_s``, raised by ``A_s / C_s``: +4.8 % in the
boundary row of a 1° source on a lat-lon target that ends at 74.75° N), and a
source cell that no target cell covers is dropped. Dividing by ``C_s`` is
what closes the slivers of global targets; telling a sliver from a partial
cover would need an area threshold, so regional targets keep this rule.
"""
struct ConservativeFluxRegridding <: AbstractFluxRegridding end

"""
    BilinearFluxRegridding()

`regridding = "bilinear"` (lat-lon and reduced-Gaussian targets, explicit
only): the flux density is sampled bilinearly at each target cell centre and
multiplied by the target cell area, ``r_t = F(x_t) a_t``. The sample's total
misses the source total; `_renormalize_surface_flux_rate!` rescales the
emitting and the uptake cells separately to the source's two totals, or
refuses when the sample has no cell of a part the source has (one uptake cell
between the sampled centres suffices). The totals are kept, not the cell
values: a factor can be far from 1 (3.53 for the GridFED December 2021 uptake
on LL72), which is logged and, beyond 10 %, a warning.
"""
struct BilinearFluxRegridding <: AbstractFluxRegridding end

const _SURFACE_FLUX_REGRIDDINGS = (:bilinear, :conservative)

"""Parse `surface_flux.regridding` (default `"conservative"`) into its method type."""
function _regridding_method(cfg)
    raw = get(cfg, "regridding", "conservative")
    method = raw isa AbstractString ? Symbol(lowercase(raw)) : nothing
    method in _SURFACE_FLUX_REGRIDDINGS || throw(ArgumentError(
        "surface_flux.regridding must be one of $(join(_SURFACE_FLUX_REGRIDDINGS, ", ")); " *
        "got $(repr(raw))"))
    return method === :conservative ? ConservativeFluxRegridding() : BilinearFluxRegridding()
end

"""
    _surface_flux_rate(method, source, mesh, FT; label) -> Vector{FT}

Physical per-cell rates [kg species s⁻¹] of the static field `source` on
`mesh`, in the mesh's flat cell order (lat-lon: `i` fastest). `label` names
the source in errors.
"""
_surface_flux_rate(::ConservativeFluxRegridding, source::FileSurfaceFluxField, mesh,
                   ::Type{FT}; label::AbstractString = "surface flux") where FT =
    _conservative_surface_flux_rate(source, mesh, FT)

function _surface_flux_rate(::BilinearFluxRegridding, source::FileSurfaceFluxField,
                            mesh::LatLonMesh, ::Type{FT};
                            label::AbstractString = "surface flux") where FT
    rate = Array{FT}(undef, nx(mesh), ny(mesh))
    for j in axes(rate, 2)
        area = Float64(cell_area(mesh, 1, j))
        lat = mesh.φᶜ[j]
        for i in axes(rate, 1)
            density = _sample_bilinear_scalar(source.raw, source.lon, source.lat, mesh.λᶜ[i], lat)
            rate[i, j] = FT(density * area)
        end
    end
    return vec(_renormalize_surface_flux_rate!(rate, source; label))
end

function _surface_flux_rate(::BilinearFluxRegridding, source::FileSurfaceFluxField,
                            mesh::ReducedGaussianMesh, ::Type{FT};
                            label::AbstractString = "surface flux") where FT
    rate = Array{FT}(undef, ncells(mesh))
    for j in 1:nrings(mesh)
        lat = mesh.latitudes[j]
        lons = ring_longitudes(mesh, j)
        for i in eachindex(lons)
            c = cell_index(mesh, i, j)
            density = _sample_bilinear_scalar(source.raw, source.lon, source.lat, lons[i], lat)
            rate[c] = FT(density * Float64(cell_area(mesh, c)))
        end
    end
    return _renormalize_surface_flux_rate!(rate, source; label)
end

"""
    BILINEAR_FACTOR_WARN

A bilinear renormalisation factor (`_renormalize_surface_flux_rate!`) more than
this far from 1 (0.1, i.e. 10 %) is logged as a warning.
"""
const BILINEAR_FACTOR_WARN = 0.1

"""
    _renormalize_surface_flux_rate!(rate, source; label) -> rate

Renormalise bilinearly sampled per-cell rates `rate` [kg s⁻¹] to the totals of
`source`. With the source's emission and uptake totals and the sample's,

```math
P = \\sum_s \\max(F_s, 0) A_s,\\quad N = \\sum_s \\min(F_s, 0) A_s,\\quad
p = \\sum_t \\max(r_t, 0),\\quad n = \\sum_t \\min(r_t, 0),
```

each emitting cell is scaled by ``P/p`` and each uptake cell by ``N/n``. Both
factors are positive, so no cell changes sign, and ``\\sum_t r_t = P + N``,
the source total (to rounding). Renormalising the net total instead fails for
mixed-sign inventories: a net total near zero gives arbitrary factors, and a
sampled net of the other sign than the source's gives a negative factor that
turns every source into a sink. When the source has a part (``P`` or ``N``)
that the sample misses entirely, or the reverse, no factor exists and the
build is an `ArgumentError` that points to `regridding = "conservative"`:
one uptake cell of the source that falls between the sampled centres is
enough. The factors keep the two totals, not the cell values: a part that the
centres sample poorly is scaled by a factor far from 1 (the uptake of the
GridFED December 2021 file sampled on a 5° LL72 grid: ``N/n = 3.53``, every
sampled uptake cell multiplied by 3.53). Both factors are logged, and a factor
more than `BILINEAR_FACTOR_WARN` (10 %) away from 1 is a warning. `label`
names the source.
"""
function _renormalize_surface_flux_rate!(rate::AbstractArray{FT}, source::FileSurfaceFluxField;
                                         label::AbstractString = "surface flux") where FT
    masses = Float64.(source.raw) .* source.cell_area
    scale_pos = _renormalization_factor(sum(m -> max(m, 0.0), masses),
                                        sum(r -> max(Float64(r), 0.0), rate), "emission", label)
    scale_neg = _renormalization_factor(sum(m -> min(m, 0.0), masses),
                                        sum(r -> min(Float64(r), 0.0), rate), "uptake", label)
    @info @sprintf("  %s: bilinear renormalisation factors: emission %.6g, uptake %.6g",
                   label, scale_pos, scale_neg)
    for (part, factor) in (("emission", scale_pos), ("uptake", scale_neg))
        abs(factor - 1) > BILINEAR_FACTOR_WARN &&
            @warn "  $(label): bilinear sampling misses the $(part) total by a factor " *
                  "$(round(factor; sigdigits = 3)); every $(part) cell is scaled by it. " *
                  "regridding = \"conservative\" (the default) keeps the cell values."
    end
    for i in eachindex(rate)
        r = Float64(rate[i])
        rate[i] = r > 0 ? FT(r * scale_pos) : r < 0 ? FT(r * scale_neg) : rate[i]
    end
    return rate
end

function _renormalization_factor(native::Float64, sampled::Float64, part, label)
    native == 0 && sampled == 0 && return 1.0
    native != 0 && sampled != 0 && return native / sampled    # same sign by construction
    throw(ArgumentError(
        "$(label): bilinear regridding cannot be renormalised: the $(part) total of the " *
        "source is $(native) kg/s and that of the bilinear sample $(sampled) kg/s. Use " *
        "regridding = \"conservative\" (the default)."))
end

"""
    _build_surface_flux_regridder(lon, lat, dst_mesh) -> regridder

Build the conservative LL→`dst_mesh` regridder for a source flux grid
defined by its oriented cell centres `lon`/`lat`, with Float64 geometry for
every run precision. The source mesh is `GriddedIO.source_latlon_mesh` with
its default `SnappedFaces()` rule, on the destination mesh's sphere. Factored
out so the time-varying path can build the regridder once and reuse it across
every time slice instead of rebuilding it per slice.
"""
function _build_surface_flux_regridder(lon::Vector{Float64}, lat::Vector{Float64},
                                       dst_mesh::AbstractHorizontalMesh)
    src_mesh = source_latlon_mesh(lon, lat; radius = dst_mesh.radius)
    return build_regridder(src_mesh, dst_mesh; cache_dir = default_regrid_cache_dir())
end

# Log the global totals of a conservative regrid; warn above 1e-6 relative.
function _report_regrid_conservation(src_total::Float64, dst_total::Float64)
    rel_err = abs(dst_total - src_total) / max(abs(src_total), 1e-30)
    @info @sprintf("  Conservative regrid: src_total=%.6e  dst_total=%.6e  rel_err=%.2e kg/s",
                   src_total, dst_total, rel_err)
    rel_err > 1e-6 && @warn @sprintf("  Conservative regrid mass conservation warning: rel_err=%.2e", rel_err)
    return nothing
end

"""
    _apply_surface_flux_regridder(regridder, raw2d, FT; report=true) -> Vector{FT}

Conservatively regrid one 2-D flux-density slice `raw2d` [kg/m²/s] of a
time-varying series using a pre-built `regridder`, returning a flat per-cell
mass rate [kg/s] (area-integrated by the regridder's `dst_areas`). The density
is regridded as an intensive field, so the target total is
``\\sum_s F_s A^{\\rm gc}_s`` over the regridder's great-circle source areas
(the time-varying loader reads no file areas).
"""
function _apply_surface_flux_regridder(regridder, raw2d::AbstractMatrix, ::Type{FT};
                                       report::Bool = true) where FT
    src_flat = vec(Float64.(raw2d))
    n_dst = length(regridder.dst_areas)
    dst_flat = zeros(Float64, n_dst)
    apply_regridder!(dst_flat, regridder, src_flat)

    # Convert flux density [kg/m²/s] → mass rate [kg/s] using regridder areas
    rate = Array{FT}(undef, n_dst)
    for c in 1:n_dst
        rate[c] = FT(dst_flat[c] * regridder.dst_areas[c])
    end

    report && _report_regrid_conservation(sum(src_flat .* regridder.src_areas),
                                          sum(Float64.(rate)))
    return rate
end

"""
    _conservative_surface_flux_rate(source, dst_mesh, FT) -> Vector{FT}

Per-cell rates [kg s⁻¹] of the static field `source` on `dst_mesh`, flat:
its per-cell rate ``m_s`` = `source.raw .* source.cell_area` divided by the
covered area ``C_s`` of each source cell (`Regridding.source_coverage`),
regridded, and multiplied by the regridder's destination areas, so
``r_t = \\sum_s W_{ts} m_s / C_s`` ([`ConservativeFluxRegridding`](@ref)) and
the target total is `source.native_total_mass_rate`, the file's own total, to
rounding. On a regional target, a source cell that no target cell covers is
dropped (the conservation report warns), and a source cell covered only in
part puts its whole rate into the covered part (see
[`ConservativeFluxRegridding`](@ref)). Callers reshape/wrap per topology.
"""
function _conservative_surface_flux_rate(source::FileSurfaceFluxField,
                                         dst_mesh::AbstractHorizontalMesh,
                                         ::Type{FT}) where FT
    regridder = _build_surface_flux_regridder(source.lon, source.lat, dst_mesh)
    mass_rate = vec(Float64.(source.raw) .* source.cell_area)
    coverage = source_coverage(regridder)
    density = [c > 0 ? m / c : 0.0 for (m, c) in zip(mass_rate, coverage)]
    dst = zeros(Float64, length(regridder.dst_areas))
    apply_regridder!(dst, regridder, density)
    dst .*= regridder.dst_areas
    _report_regrid_conservation(source.native_total_mass_rate, sum(dst))
    return FT.(dst)
end
