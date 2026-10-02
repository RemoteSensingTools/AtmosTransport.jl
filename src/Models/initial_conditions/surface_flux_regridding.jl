function _renormalize_surface_flux_rate!(rate::AbstractArray{FT}, source::FileSurfaceFluxField) where FT
    isfinite(source.native_total_mass_rate) || return rate
    sampled_total = Float64(sum(rate))
    sampled_total > 0 || return rate
    scale = source.native_total_mass_rate / sampled_total
    rate .*= FT(scale)
    return rate
end

"""Parse regridding method from config, with a caller-owned default."""
_regridding_method(cfg, default::AbstractString = "bilinear") =
    Symbol(lowercase(String(get(cfg, "regridding", default))))

"""
    _build_surface_flux_regridder(lon, lat, dst_mesh, FT) -> regridder

Build the conservative LL→`dst_mesh` regridder for a source flux grid
defined by `lon`/`lat`. Factored out so the time-varying path can build
the regridder once and reuse it across every time slice instead of
rebuilding it per slice.
"""
function _build_surface_flux_regridder(lon::Vector{Float64}, lat::Vector{Float64},
                                       dst_mesh::AbstractHorizontalMesh, ::Type{FT}) where FT
    src_mesh = _build_source_latlon_mesh(lon, lat, FT)
    return build_regridder(src_mesh, dst_mesh; cache_dir = _REGRID_CACHE_DIR)
end

"""
    _apply_surface_flux_regridder(regridder, raw2d, FT; report=true) -> Vector{FT}

Conservatively regrid one 2-D flux-density slice `raw2d` [kg/m²/s] using a
pre-built `regridder`, returning a flat per-cell mass rate [kg/s]
(area-integrated by the regridder's `dst_areas`).
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

    if report
        # Report global mass conservation (warn only; conservative regrid is exact to ~FP ulps)
        src_total = sum(src_flat .* regridder.src_areas)
        dst_total = sum(Float64.(rate))
        rel_err = abs(dst_total - src_total) / max(abs(src_total), 1e-30)
        @info @sprintf("  Conservative regrid: src_total=%.6e  dst_total=%.6e  rel_err=%.2e kg/s",
                       src_total, dst_total, rel_err)
        rel_err > 1e-6 && @warn @sprintf("  Conservative regrid mass conservation warning: rel_err=%.2e", rel_err)
    end

    return rate
end

"""
    _conservative_surface_flux_rate(source, dst_mesh, FT) -> Vector{FT}

Conservatively regrid flux density [kg/m²/s] onto `dst_mesh`; return a
flat vector of per-cell mass rates [kg/s] (already area-integrated).
Callers reshape/wrap per topology.
"""
function _conservative_surface_flux_rate(source::FileSurfaceFluxField,
                                         dst_mesh::AbstractHorizontalMesh,
                                         ::Type{FT}) where FT
    regridder = _build_surface_flux_regridder(source.lon, source.lat, dst_mesh, FT)
    return _apply_surface_flux_regridder(regridder, source.raw, FT)
end

# ---------------------------------------------------------------------------
# build_surface_flux_source — LL / RG / CS
# ---------------------------------------------------------------------------

# Opt-in flag for the time-varying surface-flux path (default false →
# byte-identical static monthly-mean behavior).
@inline _surface_flux_time_varying(cfg) =
    _config_bool(cfg, "time_varying", false, "surface-flux time_varying")

# Kinds for which a time-varying series is supported. LMDZ and GridFED are
# regular lon/lat sources that are conservatively regridded; `:cs_native` is
# an already aligned `(Xdim,Ydim,nf,time)` cubed-sphere density field.
@inline _surface_flux_supports_time_varying(kind::Symbol) =
    kind in (:lmdz_co2, :gridfed_fossil_co2, :cs_native)

function build_surface_flux_source(grid::AtmosGrid{<:LatLonMesh},
                                   tracer_name::Symbol, cfg, ::Type{FT};
                                   reference_time::Union{DateTime, Nothing} = nothing) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing
    _surface_flux_time_varying(cfg) && throw(ArgumentError(
        "time-varying surface flux is CS-only (LatLon support is a follow-up)"))

    source = _load_file_surface_flux_field(cfg, FT)
    method = _regridding_method(cfg)
    mesh = grid.horizontal

    if method === :conservative
        rate_flat = _conservative_surface_flux_rate(source, mesh, FT)
        rate_flat .*= FT(_surface_flux_storage_scale(tracer_name, cfg))
        rate = reshape(rate_flat, nx(mesh), ny(mesh))
    else
        # Legacy bilinear sampling
        rate = Array{FT}(undef, nx(mesh), ny(mesh))
        for j in axes(rate, 2)
            area = Float64(cell_area(mesh, 1, j))
            lat = mesh.φᶜ[j]
            for i in axes(rate, 1)
                lon = mesh.λᶜ[i]
                flux_density = _sample_bilinear_scalar(source.raw, source.lon, source.lat, lon, lat)
                rate[i, j] = FT(flux_density * area)
            end
        end
        _renormalize_surface_flux_rate!(rate, source)
        rate .*= FT(_surface_flux_storage_scale(tracer_name, cfg))
    end

    return SurfaceFluxSource(tracer_name, rate)
end

function build_surface_flux_source(grid::AtmosGrid{<:ReducedGaussianMesh},
                                   tracer_name::Symbol, cfg, ::Type{FT};
                                   reference_time::Union{DateTime, Nothing} = nothing) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing
    _surface_flux_time_varying(cfg) && throw(ArgumentError(
        "time-varying surface flux is CS-only (ReducedGaussian support is a follow-up)"))

    source = _load_file_surface_flux_field(cfg, FT)
    method = _regridding_method(cfg)
    mesh = grid.horizontal

    if method === :conservative
        rate = _conservative_surface_flux_rate(source, mesh, FT)
        rate .*= FT(_surface_flux_storage_scale(tracer_name, cfg))
    else
        # Legacy bilinear sampling
        rate = Array{FT}(undef, ncells(mesh))
        for j in 1:nrings(mesh)
            lat = mesh.latitudes[j]
            lons = ring_longitudes(mesh, j)
            for i in eachindex(lons)
                c = cell_index(mesh, i, j)
                flux_density = _sample_bilinear_scalar(source.raw, source.lon, source.lat, lons[i], lat)
                rate[c] = FT(flux_density * Float64(cell_area(mesh, c)))
            end
        end
        _renormalize_surface_flux_rate!(rate, source)
        rate .*= FT(_surface_flux_storage_scale(tracer_name, cfg))
    end

    return SurfaceFluxSource(tracer_name, rate)
end

"""
    build_surface_flux_source(grid::AtmosGrid{<:CubedSphereMesh},
                              tracer_name, cfg, ::Type{FT})

CS surface-flux builder. Conservatively LL→CS
regrids the 2-D flux density (kg/m²/s) onto the 6 CS panel cell
centres, multiplies by each panel's cell area to yield per-cell kg
species/s, then converts that physical rate to the model storage basis.
Returns a
`SurfaceFluxSource` whose `cell_mass_rate` is an `NTuple{6, Matrix{FT}}`
of interior-only `(Nc, Nc)` panels.

`cfg` must set `kind` (non-`none`); any of the file-based surface-flux
kinds `_load_file_surface_flux_field` understands work
(`gridfed_fossil_co2` or user-supplied `file` + `variable`).
Conservative regridding is enforced — CS bilinear is not supported.

If `cfg["time_varying"] = true`, `:lmdz_co2` and
`:gridfed_fossil_co2` keep every regular-lon/lat time slice and conservatively
regrid it, while `:cs_native` directly loads an aligned
`(Xdim,Ydim,nf,time)` series. All return a
[`TimeVaryingSurfaceFluxSource`](@ref) whose
`cell_mass_rate_series` is an `NTuple{6}` of `(Nc, Nc, ntime)` panels
plus a `times` vector (seconds since `reference_time`). The default
(`time_varying` absent/false) path is byte-identical to before.
"""
function build_surface_flux_source(grid::AtmosGrid{<:CubedSphereMesh},
                                   tracer_name::Symbol, cfg, ::Type{FT};
                                   reference_time::Union{DateTime, Nothing} = nothing) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing

    method = _regridding_method(cfg, "conservative")
    haskey(cfg, "regridding") && method !== :conservative &&
        @warn "CS surface-flux: `regridding = \"$(method)\"` requested; CS supports conservative only — forcing conservative."

    mesh = grid.horizontal
    Nc = mesh.Nc

    if _surface_flux_time_varying(cfg)
        _surface_flux_supports_time_varying(kind) || throw(ArgumentError(
            "time-varying surface flux not supported for kind=$(kind); " *
            "supported: :lmdz_co2, :gridfed_fossil_co2, :cs_native"))
        kind === :cs_native && return _build_native_timevarying_cs_surface_flux_source(
            mesh, tracer_name, cfg, FT, reference_time)
        return _build_timevarying_cs_surface_flux_source(mesh, tracer_name, cfg, FT, reference_time)
    end

    source = _load_file_surface_flux_field(cfg, FT)

    # _conservative_surface_flux_rate already returns kg/s per cell
    # (regridder.dst_areas × regridded flux density), so the panel unpack
    # only needs to reshape the flat `6*Nc^2` vector into 6 × (Nc, Nc).
    rate_flat = _conservative_surface_flux_rate(source, mesh, FT)
    rate_flat .*= FT(_surface_flux_storage_scale(tracer_name, cfg))
    length(rate_flat) == CS_PANEL_COUNT * Nc * Nc || throw(DimensionMismatch(
        "CS surface-flux conservative regrid returned $(length(rate_flat)) cells; expected $(CS_PANEL_COUNT * Nc * Nc)"))

    panels = ntuple(_ -> Matrix{FT}(undef, Nc, Nc), CS_PANEL_COUNT)
    unpack_flat_to_panels_2d!(panels, rate_flat, Nc)

    return SurfaceFluxSource(tracer_name, panels)
end

# Build the time-varying CS source: one regridder, applied per slice into
# stacked `(Nc, Nc, ntime)` panel series.
function _build_timevarying_cs_surface_flux_source(mesh, tracer_name::Symbol, cfg,
                                                   ::Type{FT},
                                                   reference_time::Union{DateTime, Nothing}) where FT
    Nc = mesh.Nc
    field = _load_timevarying_surface_flux_field(cfg, FT, reference_time)
    ntime = length(field.times_sec)
    storage_scale = FT(_surface_flux_storage_scale(tracer_name, cfg))

    regridder = _build_surface_flux_regridder(field.lon, field.lat, mesh, FT)
    panels_series = ntuple(_ -> Array{FT, 3}(undef, Nc, Nc, ntime), CS_PANEL_COUNT)

    slice_panels = ntuple(_ -> Matrix{FT}(undef, Nc, Nc), CS_PANEL_COUNT)
    @inbounds for t in 1:ntime
        # Only report mass conservation on the first slice to avoid log spam.
        rate_flat = _apply_surface_flux_regridder(regridder, @view(field.raw_series[:, :, t]),
                                                  FT; report = (t == 1))
        rate_flat .*= storage_scale
        length(rate_flat) == CS_PANEL_COUNT * Nc * Nc || throw(DimensionMismatch(
            "CS time-varying surface-flux regrid returned $(length(rate_flat)) cells; expected $(CS_PANEL_COUNT * Nc * Nc)"))
        unpack_flat_to_panels_2d!(slice_panels, rate_flat, Nc)
        for p in 1:CS_PANEL_COUNT
            panels_series[p][:, :, t] .= slice_panels[p]
        end
    end

    # Kind-aware default: lmdz_co2 (CAMS) is held PIECEWISE-CONSTANT in 3-hourly
    # blocks by HEMCO/GEOS-Chem (verified: GC's EmisCO2_Total at stamp T equals
    # the CAMS slice at T−Δ), so it MUST default to "stepwise" for GC parity — a
    # linear/interp default smears the diurnal cycle (anomaly corr 0.91 vs 0.998).
    # Other kinds keep the generic "linear" default.
    default_scheme = _surface_flux_kind(cfg) in (:lmdz_co2, :gridfed_fossil_co2) ?
                     "stepwise" : "linear"
    scheme = flux_temporal_scheme(String(get(cfg, "temporal_scheme", default_scheme)))
    return TimeVaryingSurfaceFluxSource(tracer_name, panels_series, field.times_sec, scheme)
end

"""
    _build_native_timevarying_cs_surface_flux_source(mesh, tracer_name, cfg,
                                                      FT, reference_time)

Load an already aligned GEOS-native cubed-sphere flux-density series with
Julia/NCDatasets shape `(Nc, Nc, 6, ntime)`, corresponding to NetCDF dimensions
`(time, nf, Ydim, Xdim)`. The input is kg species m⁻² s⁻¹. It is multiplied by
the runtime mesh's exact cell areas and converted to the model storage basis.
No horizontal interpolation is performed.
"""
function _build_native_timevarying_cs_surface_flux_source(
        mesh::CubedSphereMesh, tracer_name::Symbol, cfg, ::Type{FT},
        reference_time::Union{DateTime, Nothing}) where FT
    file, variable, _ = _resolve_surface_flux_file(cfg, :cs_native)
    isfile(file) || throw(ArgumentError("surface-flux file not found: $file"))

    ds = NCDataset(file)
    try
        time_var = _ic_find_coord(ds, ["time", "t"])
        isnothing(time_var) && throw(ArgumentError(
            "could not find time coordinate in native cubed-sphere flux $file"))
        haskey(ds, variable) || throw(ArgumentError(
            "variable '$variable' not found in $file"))

        raw_var = ds[variable]
        ndims(raw_var) == 4 || throw(ArgumentError(
            "native cubed-sphere surface-flux variable '$variable' must be 4D " *
            "(Xdim,Ydim,nf,time in Julia), got ndims=$(ndims(raw_var))"))
        Nc = mesh.Nc
        size(raw_var, 1) == Nc && size(raw_var, 2) == Nc || throw(DimensionMismatch(
            "native cubed-sphere flux has horizontal shape " *
            "$(size(raw_var, 1))x$(size(raw_var, 2)); transport grid is C$(Nc)"))
        size(raw_var, 3) == CS_PANEL_COUNT || throw(DimensionMismatch(
            "native cubed-sphere flux has nf=$(size(raw_var, 3)); expected $CS_PANEL_COUNT"))
        ntime = size(raw_var, 4)
        ntime > 0 || throw(ArgumentError("native cubed-sphere flux has no time slices"))

        units_norm = _normalize_units_string(get(raw_var.attrib, "units", ""))
        species_scale = if units_norm in ("kgcm-2s-1", "kgc/m2/s", "kgcm2s-1")
            FT(44.0 / 12.0)
        elseif isempty(units_norm) || occursin("/s", units_norm) || occursin("s-1", units_norm)
            one(FT)
        else
            throw(ArgumentError(
                "native cubed-sphere surface flux has unsupported units '$units_norm' " *
                "in $file; expected a per-area, per-second mass flux"))
        end
        scale = species_scale * FT(get(cfg, "scale", 1.0)) *
                FT(_surface_flux_storage_scale(tracer_name, cfg))

        area = reshape(FT.(mesh.cell_areas), Nc, Nc, 1)
        panels_series = ntuple(p -> begin
            density = FT.(nomissing(raw_var[:, :, p, :], zero(FT)))
            all(isfinite, density) || throw(ArgumentError(
                "native cubed-sphere flux contains non-finite values on panel $p"))
            density .* area .* scale
        end, CS_PANEL_COUNT)

        time_units = String(get(ds[time_var].attrib, "units", ""))
        times_sec = _surface_flux_times_seconds(ds[time_var][:], time_units, reference_time)
        issorted(times_sec) || throw(ArgumentError(
            "native cubed-sphere surface-flux times must be ascending"))

        # Hourly flux fields represent interval means and are held constant
        # over their stamped hour unless the config explicitly requests a
        # different reconstruction.
        scheme = flux_temporal_scheme(String(get(cfg, "temporal_scheme", "stepwise")))
        return TimeVaryingSurfaceFluxSource(
            tracer_name, panels_series, times_sec, scheme)
    finally
        close(ds)
    end
end

"""
    build_surface_flux_sources(grid, tracer_specs, ::Type{FT}; reference_time=nothing)

Build surface-flux source instances for every tracer spec that requests
one. Returns a tuple (possibly empty) suitable for the
`surface_sources = (…,)` kwarg on `DrivenSimulation`.

`reference_time` (the run start `DateTime`) is threaded to each
per-tracer builder so the time-varying CS path can align its slice times
to the simulation clock. It is ignored by static sources.
"""
function build_surface_flux_sources(grid, tracer_specs, ::Type{FT};
                                    reference_time::Union{DateTime, Nothing} = nothing) where FT
    sources = Any[]
    for spec in tracer_specs
        source = build_surface_flux_source(grid, spec.name, spec.surface_flux_cfg, FT;
                                           reference_time = reference_time)
        source === nothing || push!(sources, source)
    end
    return Tuple(sources)
end
