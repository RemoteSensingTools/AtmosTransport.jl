"""
    InitialConditionIO

Single owner of initial-condition I/O, vertical remap, and
topology-dispatched VMR builders for the unified runtime.

## Public API (exported via `Models` → `AtmosTransport`)

- [`build_initial_mixing_ratio`](@ref) — topology-dispatched builder
  returning **dry VMR** on interior cells. Accepts `kind = uniform |
  latitude_step | gaussian_blob | file | netcdf | file_field |
  catrine_co2` for LL/RG meshes, plus `bl_enhanced` on LL. CS supports
  those shared kinds plus `pressure_layer` and signed `cs_native` fields.
- [`pack_initial_tracer_mass`](@ref) — basis-aware VMR → conservative model storage
  conversion. Dispatches on `mass_basis::AbstractMassBasis`:
  - `DryBasis` (the default runtime basis): `rm = vmr .* air_mass`.
  - `MoistBasis`: `rm = vmr .* air_mass .* (1 .- qv)`, because IC VMRs are
    dry and `air_mass` is moist; `qv` must be supplied.
- [`FileInitialConditionSource`](@ref) — container for a loaded IC
  NetCDF (3D VMR + hybrid coefficients + surface pressure).

## Contents

- LL/RG and CS `build_initial_mixing_ratio` for the kinds above. CS
  `pressure_layer` selects a level per column from the stored surface pressure
  and normalizes its dry VMR to the configured global molecule count.
- Basis-aware `pack_initial_tracer_mass` (DryBasis + MoistBasis); CS
  output is halo-padded with the halo zeroed. `_build_source_latlon_mesh`
  helper for LL→CS conservative regridding.
- Surface-flux loaders and builders (`initial_conditions/surface_flux*.jl`):
  static `FileSurfaceFluxField` and time-varying series, and LL/RG/CS
  `build_surface_flux_source` methods. CS `build_surface_flux_source`
  conservatively LL→CS-regrids the flux (the regridder's `dst_areas` ×
  regridded density already yields kg/s per cell) and unpacks to
  `NTuple{6, Matrix{FT}}`, or returns a `TimeVaryingSurfaceFluxSource` when
  `time_varying = true` — satisfying the per-cell rate contract of
  `SurfaceFluxSource` (`src/Operators/SurfaceFlux/sources.jl`).

Private helpers (underscore-prefixed) stay unexported and are
accessed by callers (including the canonical driven runtime)
via `AtmosTransport.Models.InitialConditionIO.<name>` if needed.
"""
module InitialConditionIO

using NCDatasets
using Dates
using ..Models: _config_bool

import ...expand_data_path
using ...Parameters: AVOGADRO, EARTH_RADIUS, SPECIES_MOLAR_MASS, STANDARD_PRESSURE, DRY_AIR_MOLAR_MASS
using ..State: AbstractMassBasis, DryBasis, MoistBasis
using ..Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
                nrings, ring_longitudes, cell_index, cell_area,
                gravity, panel_cell_center_lonlat
# Regridding + Preprocessing are loaded before Models (AtmosTransport.jl)
# so we can pull in the LL→CS conservative regridder + panel unpacking
# helpers for the CS file-based IC path.
using ..Regridding: build_regridder, apply_regridder!
using ..Preprocessing: unpack_flat_to_panels_3d!, unpack_flat_to_panels_2d!,
                       CS_PANEL_COUNT
using ..Operators.SurfaceFlux: SurfaceFluxSource, TimeVaryingSurfaceFluxSource,
                               flux_temporal_scheme
# Grid accessors used by both LL/RG and CS surface-flux builders
using ..Grids: AbstractHorizontalMesh, nx, ny, ncells

using Printf: @sprintf

# ---------------------------------------------------------------------------
# Longitude-wrap helpers
# ---------------------------------------------------------------------------

# NOTE: the source arrays (Catrine, GridFED) may be in [-180, 180)
# convention; these wrap both to a common [0, 360) before looking up.
@inline wrapped_longitude_distance(lon, lon0) = abs(mod(lon - lon0 + 180, 360) - 180)
@inline wrapped_longitude_360(lon) = mod(lon, 360)

# ---------------------------------------------------------------------------
# Config-kind resolvers
# ---------------------------------------------------------------------------

@inline _init_kind(cfg) = Symbol(lowercase(String(get(cfg, "kind", "uniform"))))
@inline _is_file_init_kind(kind::Symbol) = kind in (:file, :netcdf, :file_field, :catrine_co2)
@inline _is_latitude_step_kind(kind::Symbol) =
    kind in (:latitude_step, :lat_step, :hemisphere_step)

function _cfg_get_alias(cfg, key::String, alias::String, default)
    haskey(cfg, key) && return cfg[key]
    haskey(cfg, alias) && return cfg[alias]
    return default
end

function _latitude_step_values(cfg, ::Type{FT}) where FT
    south = FT(_cfg_get_alias(cfg, "south_value", "south",
                              get(cfg, "background", 4.0e-4)))
    north = FT(_cfg_get_alias(cfg, "north_value", "north", 4.4e-4))
    split = FT(get(cfg, "split_lat_deg", 0.0))
    return (; south, north, split)
end

@inline function _latitude_step_value(lat, vals)
    return typeof(vals.split)(lat) >= vals.split ? vals.north : vals.south
end

# ---------------------------------------------------------------------------
# FileInitialConditionSource struct
# ---------------------------------------------------------------------------

"""
    FileInitialConditionSource{FT}

Container for a file-based initial condition (e.g. Catrine startCO2).

# Fields
- `raw`            — 3D mixing ratio field `(nlon_src, nlat_src, nlevel_src)`.
                     Level ordering follows the source file (Catrine: k=1 is
                     SURFACE, k=end is TOA — verify via `ap[1]+bp[1]*ps ≈ ps`).
- `lon`, `lat`     — source coordinate vectors [degrees]. May be in [-180,180)
                     or [0,360); the bilinear sampler wraps to [0,360)
                     internally via `wrapped_longitude_360`.
- `ap`, `bp`       — hybrid half-level coefficients `(nlevel_src + 1)`:
                     `p_half[k] = ap[k] + bp[k] × ps_src`. Units: `ap` [Pa],
                     `bp` [dimensionless].
- `psurf`          — surface pressure `(nlon_src, nlat_src)` [Pa].
- `needs_vinterp`  — `true` if source levels ≠ target levels (triggers
                     log-pressure vertical interpolation in
                     `_interpolate_log_pressure_profile!`).
"""
struct FileInitialConditionSource{FT}
    raw           :: Array{FT, 3}
    lon           :: Vector{Float64}
    lat           :: Vector{Float64}
    ap            :: Vector{Float64}
    bp            :: Vector{Float64}
    psurf         :: Matrix{Float64}
    needs_vinterp :: Bool
end

# ---------------------------------------------------------------------------
# Bracket search + bilinear interpolation
# ---------------------------------------------------------------------------

function _ic_find_coord(ds, candidates::Vector{String})
    for name in candidates
        haskey(ds, name) && return name
    end
    return nothing
end

"""
    _bilinear_bracket(val, arr) -> (lo, w)

Find the 1-based bracket index `lo` and fractional weight `w ∈ [0, 1]`
such that `val ≈ arr[lo] + w × (arr[lo+1] − arr[lo])`. Uses binary
search on a **strictly increasing, non-periodic** array (e.g. latitude).

Clamps: if `val ≤ arr[1]`, returns `(1, 0.0)` (extrapolate to first value).
If `val ≥ arr[end]`, returns `(N, 0.0)` (extrapolate to last value).
"""
function _bilinear_bracket(val::Float64, arr::Vector{Float64})
    N = length(arr)
    N == 1 && return (1, 0.0)
    val <= arr[1] && return (1, 0.0)
    val >= arr[N] && return (N, 0.0)
    lo, hi = 1, N
    while hi - lo > 1
        mid = (lo + hi) >> 1
        if arr[mid] <= val
            lo = mid
        else
            hi = mid
        end
    end
    denom = arr[hi] - arr[lo]
    w = denom > 0 ? (val - arr[lo]) / denom : 0.0
    return lo, w
end

"""
    _periodic_bilinear_bracket(val, arr) -> (ilo, w)

Periodic (longitude) version of `_bilinear_bracket`. `arr` must be
strictly increasing with uniform spacing `Δ = arr[2] − arr[1]` over a
360° domain (e.g. `[-179.5, -178.5, ..., 179.5]` or `[0.5, 1.5, ..., 359.5]`).

Returns 1-based index `ilo` and fractional weight `w` for the bracket
surrounding `mod(val − arr[1], 360)` in the periodic domain. Wraps both
`val` and the index modulo `length(arr)`.

The caller should also compute `ihi = (ilo == N) ? 1 : ilo + 1` for the
upper bracket (periodic wrap at the last cell).
"""
function _periodic_bilinear_bracket(val::Float64, arr::Vector{Float64})
    N = length(arr)
    N == 1 && return (1, 0.0)
    Δ = arr[2] - arr[1]   # assumed uniform spacing [degrees]
    Δ > 0 || throw(ArgumentError("longitude coordinate must be strictly increasing"))
    u = mod(val - arr[1], 360.0) / Δ   # fractional index into the periodic domain
    ilo0 = floor(Int, u)
    return mod1(ilo0 + 1, N), u - ilo0  # (1-indexed, fractional weight)
end

"""
    _horizontal_interp_weights(lon, lat, lon_src, lat_src)

Compute bilinear interpolation indices and weights for a target point
`(lon, lat)` on source arrays with coordinates `lon_src` [degrees, periodic]
and `lat_src` [degrees, non-periodic clamped].

Returns `(ilo, ihi, jlo, jhi, w00, w10, w01, w11)` where `w00` through
`w11` are the four bilinear weights for cells `(ilo, jlo)`, `(ihi, jlo)`,
`(ilo, jhi)`, `(ihi, jhi)` respectively.

**Longitude convention**: `lon` is first wrapped to [0, 360) via
`wrapped_longitude_360` before the periodic bracket. This handles both
`[-180, 180)` and `[0, 360)` input conventions transparently.
"""
function _horizontal_interp_weights(lon::Real, lat::Real, lon_src::Vector{Float64}, lat_src::Vector{Float64})
    lon_m = wrapped_longitude_360(Float64(lon))  # → [0, 360)
    ilo, wx = _periodic_bilinear_bracket(lon_m, lon_src)
    ihi = ilo == length(lon_src) ? 1 : ilo + 1   # periodic wrap at last cell
    jlo, wy = _bilinear_bracket(Float64(lat), lat_src)
    jhi = min(jlo + 1, length(lat_src))           # clamp at poles (no wrap)
    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy
    return ilo, ihi, jlo, jhi, w00, w10, w01, w11
end

function _sample_bilinear_profile!(dest::AbstractVector{FT},
                                   raw::Array{FT, 3},
                                   lon_src::Vector{Float64},
                                   lat_src::Vector{Float64},
                                   lon::Real,
                                   lat::Real) where FT
    ilo, ihi, jlo, jhi, w00, w10, w01, w11 = _horizontal_interp_weights(lon, lat, lon_src, lat_src)
    @inbounds for k in eachindex(dest)
        dest[k] = FT(w00 * raw[ilo, jlo, k] +
                     w10 * raw[ihi, jlo, k] +
                     w01 * raw[ilo, jhi, k] +
                     w11 * raw[ihi, jhi, k])
    end
    return nothing
end

function _sample_bilinear_scalar(raw::AbstractMatrix{T},
                                 lon_src::Vector{Float64},
                                 lat_src::Vector{Float64},
                                 lon::Real,
                                 lat::Real) where T
    ilo, ihi, jlo, jhi, w00, w10, w01, w11 = _horizontal_interp_weights(lon, lat, lon_src, lat_src)
    return w00 * raw[ilo, jlo] +
           w10 * raw[ihi, jlo] +
           w01 * raw[ilo, jhi] +
           w11 * raw[ihi, jhi]
end

# ---------------------------------------------------------------------------
# IC config resolver + NetCDF loader
# ---------------------------------------------------------------------------

function _resolve_file_init(cfg, kind::Symbol)
    default_file, default_variable = if kind === :catrine_co2
        ("~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc", "CO2")
    else
        ("", "")
    end
    file = expand_data_path(String(get(cfg, "file", default_file)))
    variable = String(get(cfg, "variable", default_variable))
    isempty(file) && throw(ArgumentError("file-based init.kind=$(kind) requires init.file"))
    isempty(variable) && throw(ArgumentError("file-based init.kind=$(kind) requires init.variable"))
    time_index = Int(get(cfg, "time_index", 1))
    return file, variable, time_index
end

function _load_file_initial_condition_source(cfg, ::Type{FT}, Nz_target::Integer) where FT
    kind = _init_kind(cfg)
    file, variable, time_index = _resolve_file_init(cfg, kind)
    isfile(file) || throw(ArgumentError("initial-condition file not found: $file"))

    ds = NCDataset(file)
    try
        lon_var = _ic_find_coord(ds, ["lon", "longitude", "x"])
        lat_var = _ic_find_coord(ds, ["lat", "latitude", "y"])
        lev_var = _ic_find_coord(ds, ["lev", "level", "plev", "z", "hybrid", "nhym"])
        isnothing(lon_var) && throw(ArgumentError("could not find longitude coordinate in $file"))
        isnothing(lat_var) && throw(ArgumentError("could not find latitude coordinate in $file"))
        isnothing(lev_var) && throw(ArgumentError("could not find vertical coordinate in $file"))
        haskey(ds, variable) || throw(ArgumentError("variable '$variable' not found in $file"))

        lon_src = Float64.(ds[lon_var][:])
        lat_src = Float64.(ds[lat_var][:])
        lev_src = Float64.(ds[lev_var][:])

        raw_var = ds[variable]
        raw = if ndims(raw_var) == 4
            FT.(nomissing(raw_var[:, :, :, time_index], zero(FT)))
        elseif ndims(raw_var) == 3
            FT.(nomissing(raw_var[:, :, :], zero(FT)))
        else
            throw(ArgumentError("variable '$variable' must be 3D or 4D, got ndims=$(ndims(raw_var))"))
        end

        has_hybrid = haskey(ds, "ap") && haskey(ds, "bp") && haskey(ds, "Psurf")
        ap = has_hybrid ? Float64.(ds["ap"][:]) : Float64[]
        bp = has_hybrid ? Float64.(ds["bp"][:]) : Float64[]
        psurf = has_hybrid ? Float64.(nomissing(ds["Psurf"][:, :], STANDARD_PRESSURE)) : zeros(Float64, 0, 0)

        if length(lat_src) > 1 && lat_src[1] > lat_src[end]
            raw = raw[:, end:-1:1, :]
            lat_src = reverse(lat_src)
            if has_hybrid
                psurf = psurf[:, end:-1:1]
            end
        end

        if minimum(lon_src) < 0
            split = findfirst(>=(0), lon_src)
            if split !== nothing
                idx = vcat(split:length(lon_src), 1:split-1)
                lon_src = mod.(lon_src[idx], 360.0)
                raw = raw[idx, :, :]
                if has_hybrid
                    psurf = psurf[idx, :]
                end
            end
        end

        if length(lev_src) > 1 && lev_src[1] > lev_src[end]
            raw = raw[:, :, end:-1:1]
            lev_src = reverse(lev_src)
            if has_hybrid
                ap = reverse(ap)
                bp = reverse(bp)
            end
        end

        needs_vinterp = has_hybrid && size(raw, 3) != Nz_target
        return FileInitialConditionSource{FT}(raw, lon_src, lat_src, ap, bp, psurf, needs_vinterp)
    finally
        close(ds)
    end
end

# ---------------------------------------------------------------------------
# Vertical log-pressure interpolation
# ---------------------------------------------------------------------------

"""
    _interpolate_log_pressure_profile!(dest, src_q, ap, bp, ps_src, A_tgt, B_tgt, ps_tgt)

Vertically interpolate a source profile `src_q[1:Nsrc]` onto target model
levels `dest[1:Nz]` using log-pressure linear interpolation.

## Source pressure levels

Source half-level pressures from the IC file's hybrid coordinates:

    src_p_half[k] = ap[k] + bp[k] × ps_src     k = 1..Nsrc+1

**Ordering convention** (Catrine startCO2): `ap[1] = 0, bp[1] = 1.0` →
`src_p_half[1] = ps_src` (SURFACE). `ap[end] = 0, bp[end] = 0` →
`src_p_half[end] = 0` (TOA). So `src_p_mid` is **decreasing** from
surface to TOA (k=1 is surface, k=Nsrc is TOA).

## Target pressure levels

Target half-level pressures from the transport binary's hybrid coordinates and
stored surface pressure:

    tgt_p_half[k] = A_tgt[k] + B_tgt[k] × ps_tgt

So `tgt_p_mid` is **increasing** from TOA to surface (k=1 small, k=Nz large).

## Interpolation

For each target level `k`, find the bracket in `src_p_mid` where
`src_p_mid[ks] > p_tgt ≥ src_p_mid[ks+1]`. Then linear interpolation in
log-pressure:

    w = (log p_tgt − log p_src[ks]) / (log p_src[ks+1] − log p_src[ks])
    dest[k] = src_q[ks] + w × (src_q[ks+1] − src_q[ks])

Clamps: if `p_tgt > src_p_mid[1]` (below source surface), use `src_q[1]`.
If `p_tgt < src_p_mid[end]` (above source TOA), use `src_q[end]`.

**NOTE**: source pressure decreases with level index while target pressure
increases with level index. The source bracket must therefore be found
independently for each target level; a persistent monotone source index would
move in the wrong direction after the first interior target level.
"""
function _interpolate_log_pressure_profile!(dest::AbstractVector{FT},
                                            src_q::AbstractVector{FT},
                                            ap::Vector{Float64},
                                            bp::Vector{Float64},
                                            ps_src::Float64,
                                            A_tgt::AbstractVector{<:Real},
                                            B_tgt::AbstractVector{<:Real},
                                            ps_tgt::Real) where FT
    Nsrc = length(src_q)
    Nz = length(dest)
    length(A_tgt) == Nz + 1 || throw(DimensionMismatch(
        "A_tgt has length $(length(A_tgt)), expected Nz+1 = $(Nz + 1)"))
    length(B_tgt) == Nz + 1 || throw(DimensionMismatch(
        "B_tgt has length $(length(B_tgt)), expected Nz+1 = $(Nz + 1)"))
    length(ap) == Nsrc + 1 || throw(DimensionMismatch(
        "ap has length $(length(ap)), expected Nsrc+1 = $(Nsrc + 1)"))
    length(bp) == Nsrc + 1 || throw(DimensionMismatch(
        "bp has length $(length(bp)), expected Nsrc+1 = $(Nsrc + 1)"))

    # Source half-level pressures: src_p_half[1] = ps (surface), src_p_half[end] = 0 (TOA)
    src_p_half = Vector{Float64}(undef, Nsrc + 1)
    @inbounds for k in 1:(Nsrc + 1)
        src_p_half[k] = ap[k] + bp[k] * ps_src
    end
    # Source mid-level pressures (decreasing: surface → TOA)
    src_p_mid = Vector{Float64}(undef, Nsrc)
    @inbounds for k in 1:Nsrc
        src_p_mid[k] = 0.5 * (src_p_half[k] + src_p_half[k + 1])
    end

    # Target half-level pressures from the binary's own hybrid coefficients
    # and surface pressure: `p_half[k] = A[k] + B[k] * ps_tgt`. This is
    # *exact* and decouples vertical remap from `air_mass × g / area`,
    # which can be off by 9-22% on gnomonic CS when `mesh.cell_areas[i, j]`
    # differs from the area the preprocessor used when writing `m` (seen as
    # cube panel-outline structure in a C48 column-mean IC).
    ps_tgt_f = Float64(ps_tgt)
    tgt_p_half = Vector{Float64}(undef, Nz + 1)
    @inbounds for k in 1:(Nz + 1)
        tgt_p_half[k] = Float64(A_tgt[k]) + Float64(B_tgt[k]) * ps_tgt_f
    end

    @inbounds for k in 1:Nz
        p_tgt = 0.5 * (tgt_p_half[k] + tgt_p_half[k + 1])  # target mid-level [Pa]
        if p_tgt >= src_p_mid[1]
            # Below source surface → clamp to surface-level value
            dest[k] = src_q[1]
        elseif p_tgt <= src_p_mid[end]
            # Above source TOA → clamp to TOA-level value
            dest[k] = src_q[end]
        else
            # Find bracket: src_p_mid[ks] > p_tgt ≥ src_p_mid[ks+1]
            # Source levels are ordered surface → TOA (decreasing pressure),
            # while target levels are ordered TOA → surface (increasing
            # pressure). Search independently per target level; carrying a
            # monotone source index across the loop would move in the wrong
            # direction after the first interior level.
            lo = 1
            hi = Nsrc
            while hi - lo > 1
                mid = (lo + hi) >>> 1
                if src_p_mid[mid] >= p_tgt
                    lo = mid
                else
                    hi = mid
                end
            end
            ks = lo
            # Log-pressure linear interpolation
            lp1 = log(max(src_p_mid[ks], floatmin(Float64)))
            lp2 = log(max(src_p_mid[ks + 1], floatmin(Float64)))
            lpt = log(max(p_tgt, floatmin(Float64)))
            w = (lpt - lp1) / (lp2 - lp1)
            dest[k] = FT(src_q[ks] + w * (src_q[ks + 1] - src_q[ks]))
        end
    end
    return nothing
end

function _copy_profile!(dest::AbstractVector{FT}, src_q::AbstractVector{FT}) where FT
    fill!(dest, zero(FT))
    Nz_use = min(length(dest), length(src_q))
    @views copyto!(dest[1:Nz_use], src_q[1:Nz_use])
    return nothing
end

# ---------------------------------------------------------------------------
# Topology-dispatched VMR builder
#
# `build_initial_mixing_ratio` returns **dry VMR** on interior cells.
# Shapes:
#   LL: (Nx, Ny, Nz)
#   RG: (ncells, Nz)
#   CS: NTuple{6, Array{FT, 3}} of (Nc, Nc, Nz)
# ---------------------------------------------------------------------------

"""
    build_initial_mixing_ratio(air_mass, mesh, cfg)
    build_initial_mixing_ratio(air_mass, grid::AtmosGrid, cfg; surface_pressure=nothing)

Construct the initial dry-air volume mixing ratio described by `cfg` on the
horizontal topology and vertical layout of `air_mass`.

Bare lat-lon and reduced-Gaussian meshes support the analytic `uniform`,
`latitude_step`, and `gaussian_blob` modes; `bl_enhanced` is lat-lon only.
Passing an `AtmosGrid` additionally enables file-backed modes (`file`,
`netcdf`, `file_field`, and `catrine_co2`) with topology-aware horizontal
mapping and log-pressure vertical interpolation. Cubed-sphere construction
requires an `AtmosGrid` and also supports `pressure_layer` and `cs_native`.
File-backed and pressure-layer construction require the transport window's
`surface_pressure` where indicated by the selected mode. Each public call owns
fresh output arrays. The driven cubed-sphere runner separately reuses private
interior buffers between analytic initializers before copying each tracer into
its independent packed state slot.

Returns an array shaped like `air_mass` for lat-lon and reduced-Gaussian grids,
or an `NTuple{6}` of interior `(Nc, Nc, Nz)` arrays for cubed-sphere grids.
"""
function build_initial_mixing_ratio(air_mass::AbstractArray{FT}, mesh::LatLonMesh{FT}, cfg) where FT
    kind = _init_kind(cfg)
    background = FT(get(cfg, "background", 4.0e-4))
    if kind === :uniform
        return fill(background, size(air_mass))
    elseif _is_latitude_step_kind(kind)
        vals = _latitude_step_values(cfg, FT)
        q = Array{FT}(undef, size(air_mass))
        for j in axes(q, 2)
            value = _latitude_step_value(mesh.φᶜ[j], vals)
            @views q[:, j, :] .= value
        end
        return q
    elseif kind === :bl_enhanced
        # Flat background + enhancement in the lowest `n_layers` model levels
        # (k = Nz-n_layers+1:Nz, since k=1=TOA, k=Nz=surface). Layer-based
        # threshold follows terrain naturally.
        n_layers = Int(get(cfg, "n_layers", 3))
        enhancement = FT(get(cfg, "enhancement", 1.0e-4))
        Nz = size(air_mass, 3)
        n_layers >= 1 && n_layers <= Nz ||
            throw(ArgumentError("init.kind=bl_enhanced: n_layers=$(n_layers) must satisfy 1 ≤ n_layers ≤ Nz=$(Nz)"))
        q = fill(background, size(air_mass))
        @views q[:, :, (Nz - n_layers + 1):Nz] .+= enhancement
        return q
    elseif kind === :gaussian_blob
        lon0 = FT(get(cfg, "lon0_deg", 0.0))
        lat0 = FT(get(cfg, "lat0_deg", 0.0))
        sigma_lon = FT(get(cfg, "sigma_lon_deg", 10.0))
        sigma_lat = FT(get(cfg, "sigma_lat_deg", 10.0))
        amplitude = FT(get(cfg, "amplitude", background))
        q = Array{FT}(undef, size(air_mass))
        for k in axes(q, 3), j in axes(q, 2), i in axes(q, 1)
            dlon = wrapped_longitude_distance(mesh.λᶜ[i], lon0)
            dlat = mesh.φᶜ[j] - lat0
            q[i, j, k] = background + amplitude * exp(-FT(0.5) * ((dlon / sigma_lon)^2 + (dlat / sigma_lat)^2))
        end
        return q
    else
        throw(ArgumentError("unsupported init.kind=$(kind) for LatLonMesh"))
    end
end

function build_initial_mixing_ratio(air_mass::AbstractArray{FT}, mesh::ReducedGaussianMesh{FT}, cfg) where FT
    kind = _init_kind(cfg)
    background = FT(get(cfg, "background", 4.0e-4))
    if kind === :uniform
        return fill(background, size(air_mass))
    elseif _is_latitude_step_kind(kind)
        vals = _latitude_step_values(cfg, FT)
        q = Array{FT}(undef, size(air_mass))
        for j in 1:nrings(mesh)
            value = _latitude_step_value(mesh.latitudes[j], vals)
            for i in 1:mesh.nlon_per_ring[j]
                c = cell_index(mesh, i, j)
                @views q[c, :] .= value
            end
        end
        return q
    elseif kind === :gaussian_blob
        lon0 = FT(get(cfg, "lon0_deg", 0.0))
        lat0 = FT(get(cfg, "lat0_deg", 0.0))
        sigma_lon = FT(get(cfg, "sigma_lon_deg", 10.0))
        sigma_lat = FT(get(cfg, "sigma_lat_deg", 10.0))
        amplitude = FT(get(cfg, "amplitude", background))
        q = Array{FT}(undef, size(air_mass))
        for j in 1:nrings(mesh)
            lats = mesh.latitudes[j]
            lons = ring_longitudes(mesh, j)
            for i in eachindex(lons)
                c = cell_index(mesh, i, j)
                dlon = wrapped_longitude_distance(lons[i], lon0)
                dlat = lats - lat0
                value = background + amplitude * exp(-FT(0.5) * ((dlon / sigma_lon)^2 + (dlat / sigma_lat)^2))
                @views q[c, :] .= value
            end
        end
        return q
    else
        throw(ArgumentError("unsupported init.kind=$(kind) for ReducedGaussianMesh"))
    end
end

function build_initial_mixing_ratio(air_mass::AbstractArray{FT},
                                    grid::AtmosGrid{<:LatLonMesh},
                                    cfg;
                                    surface_pressure::Union{Nothing, AbstractMatrix} = nothing) where FT
    kind = _init_kind(cfg)
    _is_file_init_kind(kind) || return build_initial_mixing_ratio(air_mass, grid.horizontal, cfg)

    source = _load_file_initial_condition_source(cfg, FT, size(air_mass, 3))
    mesh = grid.horizontal
    q = Array{FT}(undef, size(air_mass))
    src_q = Vector{FT}(undef, size(source.raw, 3))
    A_tgt = grid.vertical.A
    B_tgt = grid.vertical.B
    surface_pressure === nothing && throw(ArgumentError(
        "build_initial_mixing_ratio(::AtmosGrid{<:LatLonMesh}, ...) with " *
        "vertical-interp init kind=$(kind) requires `surface_pressure` (the " *
        "binary's stored ps) so target half-level pressures can be computed " *
        "exactly from the grid's hybrid coefficients. Pass " *
        "`window.surface_pressure` from `load_transport_window` to avoid " *
        "the area-mismatch artifact (2026-04-24)."))
    size(surface_pressure) == size(air_mass)[1:2] || throw(DimensionMismatch(
        "surface_pressure size $(size(surface_pressure)) must match air_mass " *
        "horizontal extent $(size(air_mass)[1:2])"))

    for j in axes(q, 2)
        lat = mesh.φᶜ[j]
        for i in axes(q, 1)
            lon = mesh.λᶜ[i]
            _sample_bilinear_profile!(src_q, source.raw, source.lon, source.lat, lon, lat)
            if source.needs_vinterp
                ps_src = _sample_bilinear_scalar(source.psurf, source.lon, source.lat, lon, lat)
                ps_tgt = Float64(surface_pressure[i, j])
                _interpolate_log_pressure_profile!(@view(q[i, j, :]), src_q,
                                                   source.ap, source.bp, ps_src,
                                                   A_tgt, B_tgt, ps_tgt)
            else
                _copy_profile!(@view(q[i, j, :]), src_q)
            end
        end
    end

    return q
end

function build_initial_mixing_ratio(air_mass::AbstractArray{FT},
                                    grid::AtmosGrid{<:ReducedGaussianMesh},
                                    cfg;
                                    surface_pressure::Union{Nothing, AbstractVector} = nothing) where FT
    kind = _init_kind(cfg)
    _is_file_init_kind(kind) || return build_initial_mixing_ratio(air_mass, grid.horizontal, cfg)

    source = _load_file_initial_condition_source(cfg, FT, size(air_mass, 2))
    mesh = grid.horizontal
    q = Array{FT}(undef, size(air_mass))
    src_q = Vector{FT}(undef, size(source.raw, 3))
    A_tgt = grid.vertical.A
    B_tgt = grid.vertical.B
    surface_pressure === nothing && throw(ArgumentError(
        "build_initial_mixing_ratio(::AtmosGrid{<:ReducedGaussianMesh}, ...) " *
        "with vertical-interp init kind=$(kind) requires `surface_pressure` " *
        "(the binary's stored ps, length = ncells). Pass " *
        "`window.surface_pressure` from `load_transport_window`."))
    length(surface_pressure) == size(air_mass, 1) || throw(DimensionMismatch(
        "surface_pressure length $(length(surface_pressure)) must match " *
        "air_mass cells $(size(air_mass, 1))"))

    for j in 1:nrings(mesh)
        lat = mesh.latitudes[j]
        lons = ring_longitudes(mesh, j)
        for i in eachindex(lons)
            c = cell_index(mesh, i, j)
            lon = lons[i]
            _sample_bilinear_profile!(src_q, source.raw, source.lon, source.lat, lon, lat)
            if source.needs_vinterp
                ps_src = _sample_bilinear_scalar(source.psurf, source.lon, source.lat, lon, lat)
                ps_tgt = Float64(surface_pressure[c])
                _interpolate_log_pressure_profile!(@view(q[c, :]), src_q,
                                                   source.ap, source.bp, ps_src,
                                                   A_tgt, B_tgt, ps_tgt)
            else
                _copy_profile!(@view(q[c, :]), src_q)
            end
        end
    end

    return q
end

# ---------------------------------------------------------------------------
# pack_initial_tracer_mass — basis-aware VMR → tracer-mass conversion
#
# Rule: IC VMRs are dry.
# - DryBasis:   air_mass == m_dry   → rm = vmr .* air_mass
# - MoistBasis: air_mass == m_moist → rm = vmr .* air_mass .* (1 .- qv)
# ---------------------------------------------------------------------------

"""
    pack_initial_tracer_mass(grid, air_mass, vmr_dry; mass_basis::AbstractMassBasis,
                                                      qv = nothing)

Convert dry volume mixing ratio `vmr_dry` to conservative model storage
matching the binary's mass-basis contract. On `DryBasis` that storage is
`vmr_dry × dry_air_mass`; it is not physical kg species. Returns new storage
with the same shape as `air_mass`. On cubed-sphere grids this is a six-tuple
of halo-padded panels, with the halos set to zero.

## Dispatch

- `mass_basis::DryBasis` (the default runtime basis) — `air_mass` is
  `m_dry`. Result: `vmr_dry .* air_mass`. `qv` is ignored.
- `mass_basis::MoistBasis` — `air_mass` is `m_moist`, so the dry VMR is
  converted with `1 - qv`. Result: `vmr_dry .* air_mass .* (1 .- qv)`. `qv`
  must be supplied from the first transport window; missing `qv` errors.

CS dispatch handles per-panel halo packing.

## Arguments

- `grid`           — `AtmosGrid` with a lat-lon, reduced-Gaussian, or cubed-sphere mesh.
- `air_mass`       — storage-shaped air mass from the transport window;
                     a six-tuple of halo-padded 3D panels on cubed-sphere grids.
- `vmr_dry`        — dry volume mixing ratio. On cubed-sphere grids, a six-tuple
                     of interior 3D panels; otherwise the same shape as `air_mass`.
- `mass_basis`     — `DryBasis()` or `MoistBasis()`.
- `qv`             — specific humidity, same shape as `air_mass`;
                     required iff `mass_basis isa MoistBasis`.
"""
function pack_initial_tracer_mass(grid::AtmosGrid, air_mass, vmr_dry;
                                  mass_basis::AbstractMassBasis,
                                  qv = nothing)
    return _pack_tracer_mass(grid, air_mass, vmr_dry, mass_basis, qv)
end

function _pack_tracer_mass(::AtmosGrid{<:LatLonMesh}, air_mass, vmr_dry, ::DryBasis, qv)
    return vmr_dry .* air_mass
end

function _pack_tracer_mass(::AtmosGrid{<:ReducedGaussianMesh}, air_mass, vmr_dry, ::DryBasis, qv)
    return vmr_dry .* air_mass
end

function _pack_tracer_mass(::AtmosGrid{<:LatLonMesh}, air_mass, vmr_dry, ::MoistBasis, qv)
    qv === nothing && throw(ArgumentError(
        "pack_initial_tracer_mass on MoistBasis requires qv (specific humidity) " *
        "from the first transport window to convert dry VMR to moist storage; got qv=nothing."))
    size(qv) == size(air_mass) || throw(DimensionMismatch(
        "qv shape $(size(qv)) must match air_mass shape $(size(air_mass))"))
    return vmr_dry .* air_mass .* (1 .- qv)
end

function _pack_tracer_mass(::AtmosGrid{<:ReducedGaussianMesh}, air_mass, vmr_dry, ::MoistBasis, qv)
    qv === nothing && throw(ArgumentError(
        "pack_initial_tracer_mass on MoistBasis requires qv (specific humidity) " *
        "from the first transport window to convert dry VMR to moist storage; got qv=nothing."))
    size(qv) == size(air_mass) || throw(DimensionMismatch(
        "qv shape $(size(qv)) must match air_mass shape $(size(air_mass))"))
    return vmr_dry .* air_mass .* (1 .- qv)
end

include("initial_conditions/cubed_sphere.jl")
include("initial_conditions/surface_flux.jl")

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export FileInitialConditionSource
export build_initial_mixing_ratio
export pack_initial_tracer_mass
export FileSurfaceFluxField, TimeVaryingFileSurfaceFluxField
export build_surface_flux_source, build_surface_flux_sources

end # module InitialConditionIO
