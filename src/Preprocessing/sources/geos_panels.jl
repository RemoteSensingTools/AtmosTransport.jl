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

function _validated_ll_axis(axis, name)
    all(isfinite, axis) && !isempty(axis) && all(>(0), diff(axis)) ||
        error("GEOS-FP physics $name coordinates must be finite, distinct, and sorted")
    return axis
end

_centered_lons(lons::AbstractVector{<:Real}) = sort(_wrap_lon180.(lons))

function _geosfp_latlon_axes(ds::NCDataset)
    lons = _centered_lons(_geosfp_axis(ds, ("lon", "longitude", "Xdim", "x")))
    lats = sort(_geosfp_axis(ds, ("lat", "latitude", "Ydim", "y")))
    _validated_ll_axis(lons, "longitude")
    _validated_ll_axis(lats, "latitude")
    lons[end] - lons[1] < 360 || error("GEOS-FP longitude axis must span less than 360 degrees")
    return lons, lats
end

# A dimensionless lev=1:N alone says nothing about physical orientation.
# Prefer physical coordinate metadata; otherwise use the physics archive's
# own I3 QV profile (as for MERRA-2), never the native CTM DELP orientation.
function _physics_lev(ds, field_name)
    field = ds[field_name]
    dims = dimnames(field)
    ld = _find_dim(dims, ("lev", "level", "ilev", "edge", "interface"))
    ld === nothing && error("GEOS-FP physics $field_name lacks a level dimension")
    name = dims[ld]
    haskey(ds, name) || error("GEOS-FP physics $field_name lacks coordinate $name")
    v = ds[name]
    dimnames(v) == (name,) && length(v) == size(field, ld) ||
        error("GEOS-FP physics $field_name coordinate $name must describe its level dimension")
    levels = Float64.(v[:])
    all(isfinite, levels) && length(levels) > 1 &&
        (all(>(0), diff(levels)) || all(<(0), diff(levels))) ||
        error("GEOS-FP physics $field_name level coordinate must be finite and strictly monotonic")
    return v, levels
end

function _physics_coordinate_orientation(ds, field_name)
    v, levels = _physics_lev(ds, field_name)
    positive = lowercase(String(get(v.attrib, "positive", "")))
    units = lowercase(String(get(v.attrib, "units", "")))
    if isempty(positive) && units in ("pa", "hpa", "mb", "mbar")
        positive = "down"
    end
    isempty(positive) && return nothing
    positive in ("up", "down") || error("GEOS-FP physics lev positive must be up or down")
    increasing = levels[end] > levels[1]
    return increasing == (positive == "down") ? :top_down : :bottom_up
end

function _physics_qv_orientation(ds, field_name)
    v = ds[field_name]
    dims = dimnames(v)
    ld = _find_dim(dims, ("lev", "level", "ilev", "edge", "interface"))
    td = _find_dim(dims, ("time",))
    ld === nothing && error("GEOS-FP physics QV lacks a level dimension")
    function endpoint_mean(k)
        idx = ntuple(d -> d == ld ? k : d == td ? 1 : Colon(), ndims(v))
        values = Float64.(v[idx...])
        all(isfinite, values) && all(>=(0), values) && maximum(values) < 1 ||
            error("GEOS-FP physics QV must be finite and in [0,1)")
        return sum(values) / length(values)
    end
    q1, qn = endpoint_mean(1), endpoint_mean(size(v, ld))
    q1 > 100qn && return :bottom_up
    qn > 100q1 && return :top_down
    error("Cannot detect GEOS-FP physics level order: endpoint QV means $q1, $qn; expected >100x contrast")
end

function _geosfp_physics_orientation(root, date, Nc, layout, mste, dyn)
    cm_name = _find_var_name(mste, GEOS_CONVECTION_VAR_CANDIDATES[:cmfmc])
    dt_name = _find_var_name(dyn, GEOS_CONVECTION_VAR_CANDIDATES[:dtrain])
    cm_order = _physics_coordinate_orientation(mste, cm_name)
    dt_order = _physics_coordinate_orientation(dyn, dt_name)
    profile_order = nothing
    q_order = nothing
    path = _resolve_geosfp_physics_path(root, date, "I3", Nc, layout)
    if path !== nothing
        NCDataset(path, "r") do ds
            q_name = _find_var_name(ds, ("QV", "qv", "specific_humidity"))
            q_order = _physics_coordinate_orientation(ds, q_name)
            profile_order = _physics_qv_orientation(ds, q_name)
            # The common archive index convention must agree across collections.
            _, qlev = _physics_lev(ds, q_name)
            _, dlev = _physics_lev(dyn, dt_name)
            qlev == dlev || error("GEOS-FP I3 and A3dyn level coordinates disagree")
        end
    end
    if cm_order === nothing || dt_order === nothing
        profile_order === nothing && error(
            "Cannot validate GEOS-FP physics level order: provide the physics I3 QV file " *
            "or physical lev coordinates with a positive=up/down attribute in both A3 files")
        _, clev = _physics_lev(mste, cm_name)
        _, dlev = _physics_lev(dyn, dt_name)
        length(clev) == length(dlev) + 1 &&
            sign(clev[end]-clev[1]) == sign(dlev[end]-dlev[1]) ||
            error("GEOS-FP CMFMC edges and DTRAIN layers must share the archive index convention")
    end
    orders = filter(!isnothing, [cm_order, dt_order, q_order, profile_order])
    all(==(first(orders)), orders) || error("Conflicting GEOS-FP physics level orientations")
    return first(orders)
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
    if ntime == round(Int, 86400 / GEOS_MET_INTERVAL_SECONDS)          # hourly file
        return win_idx
    elseif ntime == round(Int, 86400 / GEOS_3HOURLY_INTERVAL_SECONDS)  # 3-hourly A3 file
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
    # CF decoding retains Union{Missing,T} for variables with _FillValue even
    # when this slice has no missing values. Convert to the requested reader
    # type before longitude normalization, which requires real-valued arrays.
    raw = Array{FT}(v[idx...])
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

    # Apply exactly the same permutations as the coordinate axes, including
    # nonuniform longitudes and arbitrary periodic origins.
    lats_raw = _geosfp_axis(ds, ("lat", "latitude", "Ydim", "y"))
    lons_raw = _geosfp_axis(ds, ("lon", "longitude", "Xdim", "x"))
    ip = sortperm(_wrap_lon180.(lons_raw))
    jp = sortperm(lats_raw)
    return Rank == 2 ? field[ip, jp] : field[ip, jp, :]
end

@inline _wrap_lon180(lon::Real) = mod(Float64(lon) + 180.0, 360.0) - 180.0

function _ll_stencil(lons, lats, lon::Real, lat::Real)
    nx, ny = length(lons), length(lats)
    λ = mod(Float64(lon) - lons[1], 360.0) + lons[1]
    i0 = clamp(searchsortedlast(lons, λ), 1, nx)
    i1 = i0 == nx ? 1 : i0 + 1
    λ1 = i0 == nx ? lons[1] + 360.0 : lons[i1]
    wi = nx == 1 ? 0.0 : (λ - lons[i0]) / (λ1 - lons[i0])
    φ = clamp(Float64(lat), lats[1], lats[end])
    j0 = clamp(searchsortedlast(lats, φ), 1, max(1, ny - 1))
    j1 = min(j0 + 1, ny)
    wj = ny == 1 ? 0.0 : (φ - lats[j0]) / (lats[j1] - lats[j0])
    return GEOSFPLLStencil(i0, i1, j0, j1, wi, wj)
end

@inline function _apply_ll_stencil(src, s::GEOSFPLLStencil)
    i0, i1, j0, j1 = s.i0, s.i1, s.j0, s.j1
    wi, wj = s.wi, s.wj
    @inbounds return (1 - wi) * (1 - wj) * src[i0, j0] +
                     wi       * (1 - wj) * src[i1, j0] +
                     (1 - wi) * wj       * src[i0, j1] +
                     wi       * wj       * src[i1, j1]
end

function _interp_regular_ll(src::AbstractMatrix, lons, lats, lon::Real, lat::Real)
    return _apply_ll_stencil(src, _ll_stencil(lons, lats, lon, lat))
end

function _interp_regular_ll(src::AbstractArray{FT,3}, lons, lats,
                            lon::Real, lat::Real, k::Int) where FT
    return _interp_regular_ll(@view(src[:, :, k]), lons, lats, lon, lat)
end

function _interpolate_ll_to_panels!(dst::NTuple{CS_PANEL_COUNT, Matrix{FT}},
                                    src::AbstractMatrix{FT},
                                    physics::GEOSFPLatLonPhysicsFallback) where FT
    @inbounds for p in 1:CS_PANEL_COUNT
        for ij in eachindex(dst[p])
            dst[p][ij] = _apply_ll_stencil(src, physics.stencils[p][ij])
        end
    end
    return dst
end

function _interpolate_ll_to_panels!(dst::NTuple{CS_PANEL_COUNT, Array{FT,3}},
                                    src::AbstractArray{FT,3},
                                    physics::GEOSFPLatLonPhysicsFallback) where FT
    @inbounds for p in 1:CS_PANEL_COUNT
        out = dst[p]
        for k in axes(out, 3)
            layer = @view src[:, :, k]
            for j in axes(out, 2), i in axes(out, 1)
                out[i, j, k] = _apply_ll_stencil(layer, physics.stencils[p][i, j])
            end
        end
    end
    return dst
end

function _hflux_to_upward_wm2(raw, ds::NCDataset, var_name::String, ::Type{FT}) where FT
    units = lowercase(String(get(ds[var_name].attrib, "units", "")))
    lname = lowercase(var_name)
    if occursin("j", units) || lname in ("sshf", "surface_sensible_heat_flux")
        # J m⁻² accumulated over the ECMWF hourly window. The Int divisor keeps
        # Float32 data in Float32 before the FT conversion.
        return FT.(-raw ./ round(Int, ERA5_MET_INTERVAL_SECONDS))
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
    t_lo, t_hi = SURFACE_AIR_TEMPERATURE_RANGE
    minimum(minimum, surface.t2m) > t_lo && maximum(maximum, surface.t2m) < t_hi ||
        error("GEOS surface T2M is out of range in $(path) window $(win_idx)")
    maximum(p -> maximum(abs, p), surface.hflux) < SURFACE_HEAT_FLUX_MAX_ABS ||
        error("GEOS surface HFLUX magnitude is out of range in $(path) window $(win_idx)")
    return nothing
end

function _validate_geos_convection_panels!(field, name::String, path::String, win_idx::Int)
    field === nothing && return nothing
    all(p -> all(isfinite, p), field) ||
        error("GEOS convection $(name) contains non-finite values in $(path) window $(win_idx)")
    return nothing
end

# Bound on the magnitude of GEOS VDIFF horizontal winds [m s⁻¹]; larger values
# mean a wrong variable or a fill value.
const _VDIFF_WIND_MAX_ABS = 500

function _validate_geos_vdiff_panels!(vdiff, win_idx::Int)
    for name in (:u, :v, :t, :qv)
        field = getfield(vdiff, name)
        all(p -> all(isfinite, p), field) ||
            error("GEOS GCHP VDIFF $(name) contains non-finite values in window $(win_idx)")
    end
    maximum(p -> maximum(abs, p), vdiff.u) < _VDIFF_WIND_MAX_ABS ||
        error("GEOS GCHP VDIFF U wind magnitude is out of range in window $(win_idx)")
    maximum(p -> maximum(abs, p), vdiff.v) < _VDIFF_WIND_MAX_ABS ||
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
# (per the in-tree diagnostic
# `scripts/diagnostics/heritage/compare_era5_geosit_met.jl` and GMAO docs:
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

windows_per_day(::GEOSSettings, ::Date) = round(Int, 86400 / GEOS_MET_INTERVAL_SECONDS)

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
# GEOS-IT A1 carries EFLUX, which GEOS-Chem's non-local PBL needs alongside the
# VDIFF profiles. The GEOS-FP fallback readers do not read it.
has_pbl_eflux(s::GEOSSettings{:geosit}) = s.include_vdiff_fields
