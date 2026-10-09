# GEOS native reader: per-panel slicers, dry-basis endpoints, mass-flux scaling, settings interface.
# Split from geos.jl (refactor phase 4); included by Preprocessing.jl in this order.

# ---------------------------------------------------------------------------
# Per-panel array slicers.
# ---------------------------------------------------------------------------

"""Read one window of a 3D field as `NTuple{6, Array{FT,3}}`, level-flipped if needed."""
function _read_panels_3d(var, win_idx::Int, orientation::Symbol; FT::Type)
    raw = Array{FT}(var[:, :, :, :, win_idx])     # (Nc, Nc, 6, Nz)
    if orientation === :bottom_up
        return ntuple(p -> reverse(raw[:, :, p, :]; dims=3), 6)
    else
        return ntuple(p -> Array(raw[:, :, p, :]), 6)
    end
end

"""Read one window of a 2D field as `NTuple{6, Matrix{FT}}`."""
function _read_panels_2d(var, win_idx::Int; FT::Type)
    raw = Array{FT}(var[:, :, :, win_idx])        # (Nc, Nc, 6)
    return ntuple(p -> Array(raw[:, :, p]), 6)
end

const GEOS_SURFACE_VAR_CANDIDATES = Dict(
    :pblh  => ("PBLH", "pblh", "ZPBL", "zpbl"),
    :ustar => ("USTAR", "ustar", "UST", "ust"),
    :hflux => ("HFLUX", "hflux", "SH", "sshf", "surface_sensible_heat_flux"),
    :t2m   => ("T2M", "t2m", "T2MDEW", "2t", "2m_temperature"),
)

const GEOS_CONVECTION_VAR_CANDIDATES = Dict(
    :cmfmc  => ("CMFMC", "cmfmc", "conv_mass_flux"),
    :dtrain => ("DTRAIN", "dtrain"),
)

_dim_norm(x) = replace(lowercase(String(x)), "_" => "", "-" => "")

function _find_dim(dims, candidates)
    wanted = Set(_dim_norm.(candidates))
    return findfirst(d -> _dim_norm(d) in wanted, dims)
end

function _find_var_name(ds::NCDataset, candidates)
    wanted = Set(lowercase.(String.(candidates)))
    for name in keys(ds)
        lowercase(String(name)) in wanted && return String(name)
    end
    throw(ArgumentError("NetCDF file is missing required variable; tried " *
                        join(String.(candidates), ", ")))
end

function _geosfp_axis(ds::NCDataset, candidates)
    for name in candidates
        haskey(ds, name) && return Float64.(ds[name][:])
    end
    throw(ArgumentError("GEOS-FP lat-lon physics file is missing coordinate axis; tried " *
                        join(candidates, ", ")))
end

function _centered_lons(lons::AbstractVector{<:Real})
    out = [mod(Float64(lon) + 180.0, 360.0) - 180.0 for lon in lons]
    # Keep the field roll in sync with `_normalize_lon_to_centered`: when the
    # source was 0..360, the roll by Nx/2 produces sorted centered longitudes.
    if !issorted(out)
        out = circshift(out, length(out) ÷ 2)
    end
    return out
end

function _geosfp_latlon_axes(ds::NCDataset)
    lons = _geosfp_axis(ds, ("lon", "longitude", "Xdim", "x"))
    lats = _geosfp_axis(ds, ("lat", "latitude", "Ydim", "y"))
    lons = _centered_lons(lons)
    lats = lats[1] > lats[end] ? reverse(lats) : lats
    return lons, lats
end

function _geos_dim_length(ds::NCDataset, candidates)
    dim_name = nothing
    wanted = Set(_dim_norm.(candidates))
    for name in keys(ds.dim)
        if _dim_norm(name) in wanted
            dim_name = String(name)
            break
        end
    end
    dim_name === nothing && return nothing
    dim = ds.dim[dim_name]
    return dim isa Integer ? Int(dim) : length(dim)
end

function _time_index_for_geosfp_physics(ds::NCDataset, win_idx::Int)
    ntime = _geos_dim_length(ds, ("time",))
    ntime === nothing && return 1
    ntime <= 1 && return 1
    if ntime == 24
        return win_idx
    elseif ntime == 8
        return _a3_index_for_window(win_idx)
    end
    idx = min(win_idx, ntime)
    return idx
end

function _read_latlon_slice(ds::NCDataset, var_name::String,
                            win_idx::Int, ::Val{Rank},
                            ::Type{FT}) where {Rank, FT}
    v = ds[var_name]
    dims = dimnames(v)
    lon_dim = _find_dim(dims, ("lon", "longitude", "xdim", "x"))
    lat_dim = _find_dim(dims, ("lat", "latitude", "ydim", "y"))
    lev_dim = Rank == 3 ? _find_dim(dims, ("lev", "level", "ilev", "edge", "interface")) : nothing
    time_dim = _find_dim(dims, ("time",))
    lon_dim === nothing && throw(ArgumentError("$(var_name) lacks longitude dimension"))
    lat_dim === nothing && throw(ArgumentError("$(var_name) lacks latitude dimension"))
    Rank == 3 && lev_dim === nothing && throw(ArgumentError("$(var_name) lacks level dimension"))

    time_idx = _time_index_for_geosfp_physics(ds, win_idx)
    idx = ntuple(d -> d == lon_dim || d == lat_dim || d == lev_dim ? Colon() :
                      d == time_dim ? time_idx : 1,
                 length(dims))
    raw = Array(v[idx...])
    kept = [dims[d] for d in eachindex(dims)
            if d == lon_dim || d == lat_dim || d == lev_dim]
    perm = if Rank == 2
        [findfirst(==(dims[lon_dim]), kept), findfirst(==(dims[lat_dim]), kept)]
    else
        [findfirst(==(dims[lon_dim]), kept),
         findfirst(==(dims[lat_dim]), kept),
         findfirst(==(dims[lev_dim]), kept)]
    end
    field = perm == collect(1:Rank) ? raw : permutedims(raw, Tuple(perm))

    if haskey(ds, "lat")
        lats_raw = ds["lat"][:]
    elseif haskey(ds, "latitude")
        lats_raw = ds["latitude"][:]
    else
        lats_raw = Float64[]
    end
    if !isempty(lats_raw) && length(lats_raw) == size(field, 2) && lats_raw[1] > lats_raw[end]
        field = Rank == 2 ? field[:, end:-1:1] : field[:, end:-1:1, :]
    end

    if haskey(ds, "lon")
        lons_raw = ds["lon"][:]
    elseif haskey(ds, "longitude")
        lons_raw = ds["longitude"][:]
    else
        lons_raw = Float64[]
    end
    if !isempty(lons_raw)
        if Rank == 2
            tmp = reshape(field, size(field, 1), size(field, 2), 1)
            field = @view _normalize_lon_to_centered(tmp, lons_raw)[:, :, 1]
        else
            field = _normalize_lon_to_centered(field, lons_raw)
        end
    end
    return Array{FT}(field)
end

@inline _wrap_lon180(lon::Real) = mod(Float64(lon) + 180.0, 360.0) - 180.0

function _interp_regular_ll(src::AbstractMatrix{FT}, lons, lats,
                            lon::Real, lat::Real) where FT
    Nx, Ny = size(src)
    λ = _wrap_lon180(lon)
    Δλ = length(lons) > 1 ? lons[2] - lons[1] : 360.0
    Δφ = length(lats) > 1 ? lats[2] - lats[1] : 180.0
    x = (λ - lons[1]) / Δλ + 1.0
    x < 1.0 && (x += Nx)
    x >= Nx + 1 && (x -= Nx)
    y = clamp((Float64(lat) - lats[1]) / Δφ + 1.0, 1.0, Ny)
    i0 = clamp(floor(Int, x), 1, Nx)
    j0 = clamp(floor(Int, y), 1, Ny - 1)
    i1 = i0 == Nx ? 1 : i0 + 1
    j1 = min(j0 + 1, Ny)
    wi = x - floor(x)
    wj = y - j0
    return (1 - wi) * (1 - wj) * src[i0, j0] +
           wi       * (1 - wj) * src[i1, j0] +
           (1 - wi) * wj       * src[i0, j1] +
           wi       * wj       * src[i1, j1]
end

function _interp_regular_ll(src::AbstractArray{FT, 3}, lons, lats,
                            lon::Real, lat::Real, k::Int) where FT
    return _interp_regular_ll(@view(src[:, :, k]), lons, lats, lon, lat)
end

function _interpolate_ll_to_panels!(dst::NTuple{CS_PANEL_COUNT, Matrix{FT}},
                                    src::AbstractMatrix{FT},
                                    physics::GEOSFPLatLonPhysicsFallback) where FT
    @inbounds for p in 1:CS_PANEL_COUNT
        out = dst[p]
        tlon = physics.target_lons[p]
        tlat = physics.target_lats[p]
        for j in axes(out, 2), i in axes(out, 1)
            out[i, j] = _interp_regular_ll(src, physics.lons, physics.lats,
                                           tlon[i, j], tlat[i, j])
        end
    end
    return dst
end

function _interpolate_ll_to_panels!(dst::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                    src::AbstractArray{FT, 3},
                                    physics::GEOSFPLatLonPhysicsFallback) where FT
    @inbounds for p in 1:CS_PANEL_COUNT
        out = dst[p]
        tlon = physics.target_lons[p]
        tlat = physics.target_lats[p]
        for k in axes(out, 3), j in axes(out, 2), i in axes(out, 1)
            out[i, j, k] = _interp_regular_ll(src, physics.lons, physics.lats,
                                              tlon[i, j], tlat[i, j], k)
        end
    end
    return dst
end

function _hflux_to_upward_wm2(raw, ds::NCDataset, var_name::String, ::Type{FT}) where FT
    units = lowercase(String(get(ds[var_name].attrib, "units", "")))
    lname = lowercase(var_name)
    if occursin("j", units) || lname in ("sshf", "surface_sensible_heat_flux")
        return FT.(-raw ./ 3600)
    end
    return FT.(raw)
end

function _validate_geos_surface_panels!(surface, path::String, win_idx::Int)
    for field in (surface.pblh, surface.ustar, surface.hflux, surface.t2m)
        all(p -> all(isfinite, p), field) ||
            error("GEOS surface fallback contains non-finite values in $(path) window $(win_idx)")
    end
    minimum(minimum, surface.pblh) > 0 ||
        error("GEOS surface PBLH must be positive in $(path) window $(win_idx)")
    minimum(minimum, surface.ustar) >= 0 ||
        error("GEOS surface USTAR must be nonnegative in $(path) window $(win_idx)")
    minimum(minimum, surface.t2m) > 150 && maximum(maximum, surface.t2m) < 350 ||
        error("GEOS surface T2M is out of range in $(path) window $(win_idx)")
    maximum(p -> maximum(abs, p), surface.hflux) < 5000 ||
        error("GEOS surface HFLUX magnitude is out of range in $(path) window $(win_idx)")
    return nothing
end

function _validate_geos_convection_panels!(field, name::String, path::String, win_idx::Int)
    field === nothing && return nothing
    all(p -> all(isfinite, p), field) ||
        error("GEOS convection $(name) contains non-finite values in $(path) window $(win_idx)")
    return nothing
end

function _validate_geos_vdiff_panels!(vdiff, win_idx::Int)
    for name in (:u, :v, :t, :qv)
        field = getfield(vdiff, name)
        all(p -> all(isfinite, p), field) ||
            error("GEOS GCHP VDIFF $(name) contains non-finite values in window $(win_idx)")
    end
    maximum(p -> maximum(abs, p), vdiff.u) < 500 ||
        error("GEOS GCHP VDIFF U wind magnitude is out of range in window $(win_idx)")
    maximum(p -> maximum(abs, p), vdiff.v) < 500 ||
        error("GEOS GCHP VDIFF V wind magnitude is out of range in window $(win_idx)")
    minimum(minimum, vdiff.t) > 100 && maximum(maximum, vdiff.t) < 400 ||
        error("GEOS GCHP VDIFF temperature is out of range in window $(win_idx)")
    minimum(minimum, vdiff.qv) >= 0 && maximum(maximum, vdiff.qv) < 0.2 ||
        error("GEOS GCHP VDIFF specific humidity is out of range in window $(win_idx)")
    return nothing
end

"""
    _ps_pa_factor(var) -> FT scaling

GEOS-IT CTM_I1 stores PS in hPa; GEOS-FP native CTM stores PS in Pa. Read
the `units` attribute (case-insensitive) and return the multiplier needed
to land in Pa. Errors loudly on unrecognized units to prevent silent
100x errors.
"""
function _ps_pa_factor(var; FT::Type)
    units = lowercase(strip(get(var.attrib, "units", "")))
    if units == "pa"
        return FT(1)
    elseif units == "hpa" || units == "mbar" || units == "millibars"
        return FT(100)
    else
        error("Unrecognized PS units `$(units)` on GEOS NetCDF variable; expected " *
              "Pa or hPa/mbar")
    end
end

# ---------------------------------------------------------------------------
# Endpoint dry-basis reconstruction.
# Given PS_total (Pa) and QV (kg/kg) at one hour, plus the hybrid
# coordinate, build (PS_dry, DELP_dry) consistent with Σ DELP_dry = PS_dry.
# ---------------------------------------------------------------------------

"""
    endpoint_dry_mass!(delp_dry, ps_dry, ps_total, qv, vc) -> (delp_dry, ps_dry)

Reconstruct dry DELP and dry PS at one endpoint hour from the moist PS and
QV provided by GEOS, using the hybrid coordinate `vc`. The output is on
top-down level convention (k=1 = TOA).

Algorithm:

    DELP_full[k] = ΔA[k] + ΔB[k] * PS_total
    DELP_dry[k]  = (1 - QV[k]) * DELP_full[k]
    PS_dry       = Σ DELP_dry[k]

In-place: writes into `delp_dry` and `ps_dry`.
"""
function endpoint_dry_mass!(delp_dry::AbstractArray{FT,3},
                            ps_dry::AbstractMatrix{FT},
                            ps_total::AbstractMatrix{FT},
                            qv::AbstractArray{FT,3},
                            vc::HybridSigmaPressure) where {FT}
    Nx, Ny, Nz = size(qv)
    @assert size(delp_dry) == size(qv)
    @assert size(ps_dry)   == size(ps_total) == (Nx, Ny)
    A = vc.A
    B = vc.B
    @assert length(A) == length(B) == Nz + 1

    @inbounds for j in 1:Ny, i in 1:Nx
        ps_total_ij = ps_total[i, j]
        ps_dry_ij = zero(FT)
        @simd for k in 1:Nz
            ΔA = FT(A[k+1] - A[k])
            ΔB = FT(B[k+1] - B[k])
            delp_full = ΔA + ΔB * ps_total_ij
            delp_dry_k = (1 - qv[i, j, k]) * delp_full
            delp_dry[i, j, k] = delp_dry_k
            ps_dry_ij += delp_dry_k
        end
        ps_dry[i, j] = ps_dry_ij
    end
    return delp_dry, ps_dry
end

"""Allocate (delp_dry, ps_dry) for one panel and run `endpoint_dry_mass!`."""
function endpoint_dry_mass(ps_total::AbstractMatrix{FT}, qv::AbstractArray{FT,3},
                           vc::HybridSigmaPressure) where {FT}
    delp_dry = similar(qv)
    ps_dry   = similar(ps_total)
    endpoint_dry_mass!(delp_dry, ps_dry, ps_total, qv, vc)
    return delp_dry, ps_dry
end

# ---------------------------------------------------------------------------
# Mass-flux scaling.
#
# MFXC and MFYC in GEOS-IT and GEOS-FP CTM files are ALREADY dry mass fluxes
# (per the in-tree diagnostic `compare_era5_geosit_met.jl` and GMAO docs:
# "GEOS am_moist = MFXC / (g × dt_dyn) / (1 − qv)"; getting MOIST from
# native MFXC requires *dividing* by `(1 − qv)`, so multiplying by it would
# double-dry). The reader therefore only divides by `mass_flux_dt` to convert
# the dynamics-step accumulated quantity to a rate-like quantity; no humidity
# correction is applied here.
# ---------------------------------------------------------------------------

function _scale_flux!(am::AbstractArray{FT,3},
                      mfxc_raw::AbstractArray{FT,3},
                      inv_dt::FT) where {FT}
    @inbounds @simd for i in eachindex(am)
        am[i] = mfxc_raw[i] * inv_dt
    end
    return am
end

# ---------------------------------------------------------------------------
# AbstractMetSettings interface.
# ---------------------------------------------------------------------------

windows_per_day(::GEOSSettings, ::Date) = 24

# Per-source output-filename override: GEOS gets a clearer prefix.
_native_output_filename(::AbstractGEOSSettings, date::Date, FT::Type) =
    "geos_transport_$(Dates.format(date, "yyyymmdd"))_$(FT === Float32 ? "float32" : "float64").bin"

# `has_convection` reports the capability that downstream code can rely on.
# When `settings.include_convection` is `true`, the reader populates
# `RawWindow.cmfmc` (from A3mstE) and `RawWindow.dtrain` (from A3dyn) per
# the 3-hourly hold-constant binding (window 1–3 → t=1, 4–6 → t=2, …);
# the orchestrator threads them into the v4 binary's `:cmfmc` / `:dtrain`
# payload sections, and the runtime `CMFMCConvection` operator consumes
# them via `ConvectionForcing` (GCHP RAS / Grell-Freitas, dry-basis).
has_convection(s::GEOSSettings) = s.include_convection
has_surface(s::GEOSSettings) = s.include_surface || s.include_vdiff_fields
has_vdiff_fields(s::GEOSSettings) = s.include_vdiff_fields
