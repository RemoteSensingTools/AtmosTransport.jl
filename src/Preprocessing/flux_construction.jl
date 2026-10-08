# ---------------------------------------------------------------------------
# Flux construction from cell-centre winds, shared by the wind-derived paths
# (MERRA-2, ERA5 N320): `[preprocessing]` options, face-flux methods and wind
# regridding. See docs/src/theory/vertical_transport.md, section 1.
# ---------------------------------------------------------------------------

# Accepted values of the `[preprocessing]` flux-construction keys.
const FACE_LENGTH_KINDS        = (:cell_centerline, :edge)
const FACE_FLUX_KINDS          = (:panel_average, :vector)
const FACE_INTERPOLATION_KINDS = (:linear, :cubic, :fv3)
const WIND_REGRID_KINDS        = (:scalar, :cartesian)

# Reject unknown flux-construction options; `source` names the met source in errors.
function _validate_flux_construction(s, source::AbstractString)
    for (name, value, kinds) in (("face_lengths", s.face_lengths, FACE_LENGTH_KINDS),
                                 ("face_fluxes", s.face_fluxes, FACE_FLUX_KINDS),
                                 ("face_interpolation", s.face_interpolation, FACE_INTERPOLATION_KINDS),
                                 ("wind_regrid", s.wind_regrid, WIND_REGRID_KINDS))
        value in kinds || throw(ArgumentError(
            "$source $(name) must be one of $(join(kinds, ", ")); got :$(value)"))
    end
    s.face_fluxes === :vector || s.face_interpolation === :linear ||
        throw(ArgumentError("$source face_interpolation applies to face_fluxes = :vector only"))
    haskey(COLUMN_WEIGHT_KINDS, s.column_balance_weights) ||
        throw(ArgumentError("$source column_balance_weights must be one of " *
                            join(keys(COLUMN_WEIGHT_KINDS), ", ") * "; got :$(s.column_balance_weights)"))
    return s
end

# How face fluxes are built from the cell-centre winds.
struct PanelAverageFluxes{L <: AbstractFaceLengths}    # panel components averaged (historical)
    lengths :: L
end
struct VectorFaceFluxes{G <: CSVectorFaceGeometry}     # vectors projected on the face normal
    geom :: G
    face_table :: CSGlobalFaceTable
end
# `face_interpolation` → stencil order across the face and FV3's filter along it.
const _FACE_INTERPOLATION = (linear = (order = 2, along_face_filter = false),
                             cubic  = (order = 4, along_face_filter = false),
                             fv3    = (order = 4, along_face_filter = true))
function _face_flux_method(settings, grid)
    settings.face_fluxes === :vector && return VectorFaceFluxes(
        CSVectorFaceGeometry(grid.mesh, grid.face_table; _FACE_INTERPOLATION[settings.face_interpolation]...),
        grid.face_table)
    lengths = settings.face_lengths === :edge ? EdgeLengths(grid.mesh) :
              CellCenterlineLengths(grid.mesh.Δx, grid.mesh.Δy)
    return PanelAverageFluxes(lengths)
end

# Cell-centre winds the method needs: panel-local face-normal components, or
# the east/north components themselves.
_prepare_cell_winds!(::PanelAverageFluxes, x, u_east, v_north, mesh, Nz) =
    rotate_winds_to_panel_local!(x.u_local, x.v_local, u_east, v_north, mesh, Nz)
function _prepare_cell_winds!(::VectorFaceFluxes, x, u_east, v_north, mesh, Nz)
    foreach(copyto!, x.u_local, u_east)
    foreach(copyto!, x.v_local, v_north)
    return nothing
end

_face_fluxes!(m::PanelAverageFluxes, x, g, dt, Nc, Nz) =
    cs_face_fluxes!(x.am, x.bm, x.u_local, x.v_local, x.dp, m.lengths, g, dt, Nc, Nz)
_face_fluxes!(m::VectorFaceFluxes, x, g, dt, Nc, Nz) =
    cs_vector_face_fluxes!(x.am, x.bm, x.u_local, x.v_local, x.dp, m.geom, m.face_table, g, dt, Nc, Nz)

"""
    AbstractWindRegrid

How the east/north winds are regridded from the source grid to the cube:
[`ScalarWindRegrid`](@ref) or [`CartesianWindRegrid`](@ref)
(`[preprocessing] wind_regrid`).
"""
abstract type AbstractWindRegrid end

"""
    ScalarWindRegrid()

Remap `u` and `v` as two independent scalars (historical default). Their basis
vectors turn with longitude, so near a pole a cube cell averages components that
point in different directions. A C90 cell touching the pole spans 90° of
longitude, and its remapped wind is off by about 5% for a flow across the pole; the
resulting spurious divergence poleward of 88° is several times the global RMS
divergence (`docs/src/theory/vertical_transport.md`).
"""
struct ScalarWindRegrid <: AbstractWindRegrid end

"""
    CartesianWindRegrid(source_mesh, target_mesh, Nz, FT)

Remap the wind as a vector, as GCHP's MAPL does for `UA;VA`
(`MAPL_EsmfRegridder.F90`). The source winds are written in fixed Cartesian
components

    X = −sin λ u − sin φ cos λ v,   Y = cos λ u − sin φ sin λ v,   Z = cos φ v,

the three components are remapped conservatively, and the result is projected
on the east and north unit vectors at each cube-cell centre. The remapped vector
is the area mean of the source vectors, so it is not exactly tangent to the
sphere at the cell centre; the projection drops the small radial part.

`source_mesh` is a `LatLonMesh` (MERRA-2) or a `ReducedGaussianMesh` (ERA5
N320); the source cells are taken in the regridder's order (longitude fastest,
or ring by ring). Owns the three source component buffers (per source cell and
level; 3 × 60 MB for MERRA-2 L72 and 3 × 300 MB for N320 L137 in Float32), the
remapped `Z` panels, and the basis trigonometry of both grids. The buffers are
only used inside one `_regrid_winds!` call, so one instance can serve several
pipelines.
"""
struct CartesianWindRegrid{FT} <: AbstractWindRegrid
    src_xyz  :: NTuple{3, Matrix{FT}}                      # X, Y, Z per source cell and level
    dst_z    :: NTuple{6, Array{FT, 3}}                    # remapped Z
    src_trig :: NamedTuple{(:sinλ, :cosλ, :sinφ, :cosφ), NTuple{4, Vector{Float64}}}   # per source cell
    dst_trig :: NTuple{6, NamedTuple{(:sinλ, :cosλ, :sinφ, :cosφ), NTuple{4, Matrix{Float64}}}}
end

_trig(λ, φ) = (sinλ = sin.(λ), cosλ = cos.(λ), sinφ = sin.(φ), cosφ = cos.(φ))

# Source cell centres (degrees) in the regridder's cell order.
_source_cell_centers(m::LatLonMesh) = (repeat(m.λᶜ; outer = m.Ny), repeat(m.φᶜ; inner = m.Nx))
_source_cell_centers(m::ReducedGaussianMesh) =
    (reduce(vcat, ring_longitudes(m, j) for j in eachindex(m.latitudes)),
     reduce(vcat, fill(m.latitudes[j], m.nlon_per_ring[j]) for j in eachindex(m.latitudes)))

function CartesianWindRegrid(source, target::CubedSphereMesh, Nz::Integer, ::Type{FT}) where FT
    lon, lat = _source_cell_centers(source)
    dst_trig = ntuple(6) do p
        lonp, latp = panel_cell_center_lonlat(target, p)
        _trig(deg2rad.(Float64.(lonp)), deg2rad.(Float64.(latp)))
    end
    Nc = target.Nc
    return CartesianWindRegrid{FT}(ntuple(_ -> zeros(FT, length(lon), Nz), 3),
                                   ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6),
                                   _trig(deg2rad.(Float64.(lon)), deg2rad.(Float64.(lat))), dst_trig)
end

_wind_regrid(kind::Symbol, source, target, Nz, FT) =
    kind === :cartesian ? CartesianWindRegrid(source, target, Nz, FT) : ScalarWindRegrid()
_wind_regrid(method::ScalarWindRegrid, source, target, Nz, FT) = method
function _wind_regrid(method::CartesianWindRegrid, source, target, Nz, FT)    # shared instance
    size(method.src_xyz[1], 2) == Nz || throw(DimensionMismatch(
        "shared CartesianWindRegrid has $(size(method.src_xyz[1], 2)) levels, not $Nz"))
    return method
end

"""
    _regrid_winds!(method, u_cs, v_cs, regrid!, u, v)

Regrid the source east/north winds `u`, `v` to the east/north cube panels `u_cs`,
`v_cs` with `method`, an [`AbstractWindRegrid`](@ref). `regrid!(panels, field)`
remaps one source field, in the layout of `u` (lon × lat × level, or cell ×
level, level last), to cube panels; it must not keep `field`, which may be a
reshaped internal buffer.
"""
function _regrid_winds!(::ScalarWindRegrid, u_cs, v_cs, regrid!, u, v)
    regrid!(u_cs, u)
    regrid!(v_cs, v)
    return nothing
end

function _regrid_winds!(w::CartesianWindRegrid{FT}, u_cs, v_cs, regrid!, u, v) where FT
    X, Y, Z = w.src_xyz
    (size(u) == size(v) && length(u) == length(X) && size(u, ndims(u)) == size(X, 2)) ||
        throw(DimensionMismatch("winds $(size(u)), $(size(v)) do not match the Cartesian buffers $(size(X))"))
    uc, vc = reshape(u, size(X)), reshape(v, size(X))         # source cell × level
    (; sinλ, cosλ, sinφ, cosφ) = w.src_trig
    @inbounds for k in axes(X, 2), c in axes(X, 1)
        uu, vv = Float64(uc[c, k]), Float64(vc[c, k])
        X[c, k] = FT(-sinλ[c] * uu - sinφ[c] * cosλ[c] * vv)
        Y[c, k] = FT( cosλ[c] * uu - sinφ[c] * sinλ[c] * vv)
        Z[c, k] = FT( cosφ[c] * vv)
    end
    regrid!(u_cs, reshape(X, size(u)))      # holds X until projected
    regrid!(v_cs, reshape(Y, size(u)))      # holds Y until projected
    regrid!(w.dst_z, reshape(Z, size(u)))
    for p in 1:6
        (; sinλ, cosλ, sinφ, cosφ) = w.dst_trig[p]
        ue, vn, z = u_cs[p], v_cs[p], w.dst_z[p]
        @inbounds for k in axes(ue, 3), j in axes(ue, 2), i in axes(ue, 1)
            x, y = Float64(ue[i, j, k]), Float64(vn[i, j, k])
            ue[i, j, k] = FT(-sinλ[i, j] * x + cosλ[i, j] * y)
            vn[i, j, k] = FT(-sinφ[i, j] * (cosλ[i, j] * x + sinλ[i, j] * y) + cosφ[i, j] * Float64(z[i, j, k]))
        end
    end
    return nothing
end
