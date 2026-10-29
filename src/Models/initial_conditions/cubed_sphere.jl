# ---------------------------------------------------------------------------
# CS build_initial_mixing_ratio
#
# Returns a 6-tuple of interior `(Nc, Nc, Nz)` VMR arrays. The tuple is
# topology-shaped but halo-free; the halo ring is added later by
# `pack_initial_tracer_mass` so that different halo widths can share the
# same IC builder.
# ---------------------------------------------------------------------------

function build_initial_mixing_ratio(air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                    grid::AtmosGrid{<:CubedSphereMesh},
                                    cfg;
                                    surface_pressure::Union{Nothing, NTuple{6, <:AbstractMatrix}} = nothing,
                                    pressure_basis::AbstractSurfacePressureBasis = TotalSurfacePressure(),
                                    qv::Union{Nothing, NTuple{6}} = nothing) where FT
    return _build_cs_initial_mixing_ratio(
        air_mass, grid, cfg, nothing; surface_pressure, pressure_basis, qv)
end

# The allocating public API always owns fresh panels. The runner may reuse its
# private interior buffers between tracers; file/native builders may replace
# those buffers with their own newly allocated output.
function _cs_initial_vmr_storage(::Nothing, ::Type{FT}, Nc, Nz) where FT
    return ntuple(_ -> Array{FT}(undef, Nc, Nc, Nz), CS_PANEL_COUNT)
end

function _cs_initial_vmr_storage(vmr::NTuple{6, <:AbstractArray{FT, 3}},
                                 ::Type{FT}, Nc, Nz) where FT
    for p in 1:CS_PANEL_COUNT
        size(vmr[p]) == (Nc, Nc, Nz) || throw(DimensionMismatch(
            "CS initial VMR panel $p must have shape $((Nc, Nc, Nz)); got $(size(vmr[p]))"))
    end
    return vmr
end

function _build_cs_initial_mixing_ratio(air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                       grid::AtmosGrid{<:CubedSphereMesh}, cfg, vmr;
                                       surface_pressure::Union{Nothing, NTuple{6, <:AbstractMatrix}} = nothing,
                                       pressure_basis::AbstractSurfacePressureBasis = TotalSurfacePressure(),
                                       qv::Union{Nothing, NTuple{6}} = nothing) where FT
    kind = _init_kind(cfg)
    mesh = grid.horizontal
    Nc = mesh.Nc
    Nz = size(air_mass[1], 3)
    background = FT(get(cfg, "background", DEFAULT_BACKGROUND_VMR))

    if kind === :uniform
        vmr = _cs_initial_vmr_storage(vmr, FT, Nc, Nz)
        foreach(panel -> fill!(panel, background), vmr)
        return vmr
    elseif _is_latitude_step_kind(kind)
        vals = _latitude_step_values(cfg, FT)
        vmr = _cs_initial_vmr_storage(vmr, FT, Nc, Nz)
        for p in 1:CS_PANEL_COUNT
            _lons, lats = panel_cell_center_lonlat(mesh, p)
            q = vmr[p]
            for j in 1:Nc, i in 1:Nc
                value = _latitude_step_value(lats[i, j], vals)
                @views q[i, j, :] .= value
            end
        end
        return vmr
    elseif kind === :gaussian_blob
        lon0 = FT(get(cfg, "lon0_deg", 0.0))
        lat0 = FT(get(cfg, "lat0_deg", 0.0))
        sigma_lon = FT(get(cfg, "sigma_lon_deg", DEFAULT_GAUSSIAN_BLOB_SIGMA_DEG))
        sigma_lat = FT(get(cfg, "sigma_lat_deg", DEFAULT_GAUSSIAN_BLOB_SIGMA_DEG))
        amplitude = FT(get(cfg, "amplitude", background))
        vmr = _cs_initial_vmr_storage(vmr, FT, Nc, Nz)
        for p in 1:CS_PANEL_COUNT
            lons, lats = panel_cell_center_lonlat(mesh, p)
            q = vmr[p]
            for j in 1:Nc, i in 1:Nc
                dlon = wrapped_longitude_distance(lons[i, j], lon0)
                dlat = lats[i, j] - lat0
                value = background + amplitude * exp(-FT(0.5) *
                    ((dlon / sigma_lon)^2 + (dlat / sigma_lat)^2))
                @views q[i, j, :] .= value
            end
        end
        return vmr
    elseif kind === :pressure_layer
        surface_pressure === nothing && throw(ArgumentError(
            "init.kind=pressure_layer requires `surface_pressure` " *
            "(NTuple{6, Matrix}) so the target layer can be selected by " *
            "per-column ps. Pass `window.surface_pressure` from the binary."))
        return _build_cs_pressure_layer_ic(air_mass, grid, cfg, FT, surface_pressure,
                                           pressure_basis, qv, vmr)
    elseif kind === :cs_native
        return _build_cs_native_ic(grid, air_mass, cfg, FT;
                                   surface_pressure, pressure_basis, qv)
    elseif _is_file_init_kind(kind)
        surface_pressure === nothing && throw(ArgumentError(
            "build_initial_mixing_ratio(::AtmosGrid{<:CubedSphereMesh}, ...) " *
            "with vertical-interp init kind=$(kind) requires `surface_pressure` " *
            "(the binary's stored ps as `NTuple{6, Matrix}`). Pass " *
            "`window.surface_pressure` from `load_transport_window` so target " *
            "half-level pressures use the binary's hybrid coefficients exactly. " *
            "Without this, the gnomonic `mesh.cell_areas[i,j]` mismatch with " *
            "the preprocessor's area produces a 9-22% pressure drift and " *
            "visible cube-panel artifacts (2026-04-24)."))
        for p in 1:CS_PANEL_COUNT
            size(surface_pressure[p]) == (Nc, Nc) || throw(DimensionMismatch(
                "surface_pressure[$p] size $(size(surface_pressure[p])) must be ($Nc, $Nc)"))
        end
        qv === nothing || all(p -> size(qv[p]) == (Nc, Nc, Nz), 1:CS_PANEL_COUNT) ||
            throw(DimensionMismatch("CS target humidity panels must each have shape ($Nc, $Nc, $Nz)"))
        return _build_cs_file_ic(grid, air_mass, cfg, FT, surface_pressure,
                                 pressure_basis, qv)
    else
        throw(ArgumentError(
            "unsupported init.kind=$(kind) for CubedSphereMesh; " *
            "supported: uniform | latitude_step | gaussian_blob | file | netcdf | file_field | catrine_co2 | pressure_layer | cs_native"))
    end
end

function _build_cs_file_ic(grid::AtmosGrid{<:CubedSphereMesh},
                           air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                           cfg, ::Type{FT},
                           ps_tgt_panels::NTuple{6, <:AbstractMatrix},
                           pressure_basis::AbstractSurfacePressureBasis,
                           target_qv_panels::Union{Nothing, NTuple{6}}) where FT
    mesh = grid.horizontal
    Nc   = mesh.Nc
    Nz   = size(air_mass[1], 3)
    A_tgt = grid.vertical.A
    B_tgt = grid.vertical.B

    source = _load_file_initial_condition_source(cfg, FT, Nz)
    # The loader rolled and flipped the source to [0, 360) longitude and
    # ascending latitude. The source mesh is Float64 for every run precision
    # (Float32 and Float64 runs share weights) and sits on the destination
    # sphere, so the conservative weights compare areas on one sphere.
    src_mesh = source_latlon_mesh(source.lon, source.lat; radius = mesh.radius)
    regridder = build_regridder(src_mesh, mesh)

    # 3D VMR: (Nx_src, Ny_src, Nlev_src) → 6 × (Nc, Nc, Nlev_src)
    Nlev_src = size(source.raw, 3)
    n_src    = length(source.lon) * length(source.lat)
    n_dst    = CS_PANEL_COUNT * Nc * Nc
    src_flat = Matrix{FT}(undef, n_src, Nlev_src)
    dst_flat = Matrix{FT}(undef, n_dst, Nlev_src)
    copyto!(src_flat, reshape(source.raw, n_src, Nlev_src))
    apply_regridder!(dst_flat, regridder, src_flat)
    vmr_src_levels = ntuple(_ -> Array{FT}(undef, Nc, Nc, Nlev_src), CS_PANEL_COUNT)
    unpack_flat_to_panels_3d!(vmr_src_levels, dst_flat, Nc, Nlev_src)

    source_qv_levels = if source.qv === nothing
        nothing
    else
        copyto!(src_flat, reshape(source.qv, n_src, Nlev_src))
        apply_regridder!(dst_flat, regridder, src_flat)
        panels = ntuple(_ -> Array{FT}(undef, Nc, Nc, Nlev_src), CS_PANEL_COUNT)
        unpack_flat_to_panels_3d!(panels, dst_flat, Nc, Nlev_src)
        panels
    end

    # Vertical remap column-by-column into interior `(Nc, Nc, Nz)` tuple.
    vmr = ntuple(_ -> Array{FT}(undef, Nc, Nc, Nz), CS_PANEL_COUNT)
    src_q = Vector{FT}(undef, Nlev_src)
    # Keep the loop-local scratch concrete even when source humidity is absent.
    # Its values are only consumed when `source_qv_levels !== nothing`.
    src_qv = similar(src_q)
    src_p = Vector{Float64}(undef, length(source.ap))
    cumulative = similar(src_p)
    dry_cumulative = similar(src_p)
    source.needs_vinterp &&
        _warn_surface_pressure_approximation(pressure_basis, target_qv_panels)

    for p in 1:CS_PANEL_COUNT
        for j in 1:Nc, i in 1:Nc
            @views copyto!(src_q, vmr_src_levels[p][i, j, :])
            source_qv_levels === nothing || @views copyto!(
                src_qv, source_qv_levels[p][i, j, :])
            if source.needs_vinterp
                target_qv = target_qv_panels === nothing ? nothing :
                            @view(target_qv_panels[p][i, j, :])
                ps_total = _total_surface_pressure(
                    pressure_basis, ps_tgt_panels[p][i, j], A_tgt, B_tgt, target_qv)
                source_qv = source_qv_levels === nothing ? nothing : src_qv
                _remap_hybrid_profile!(@view(vmr[p][i, j, :]), src_q, source_qv,
                                       source.ap, source.bp,
                                       A_tgt, B_tgt, ps_total,
                                       src_p, cumulative, dry_cumulative)
            else
                _copy_profile!(@view(vmr[p][i, j, :]), src_q)
            end
        end
    end

    return vmr
end

# ---------------------------------------------------------------------------
# CS NATIVE IC — read a native cubed-sphere field cell-for-cell (no
# horizontal regrid) and state-align to the binary's own topology. When source
# and target level counts differ, use the same conservative hybrid remap as
# other file ICs. Used to seed tracers from a GEOS-Chem 3D field stored on the
# SAME cube the binary uses (e.g. CATRINE FossilCO2 / Rn222), so only transport
# + emission diverge from GC afterwards.
#
# The source NetCDF carries `<variable>(time, lev, nf, Ydim, Xdim)` in CF
# (surface-first lev, like GEOS-Chem SpeciesConcVV_*). NCDatasets reads it
# in reversed (Julia column-major) order as `(Xdim, Ydim, nf, lev[, time])`,
# so panel axis-1 == Xdim and axis-2 == Ydim — IDENTICAL to the model's own
# CS writer, which stores panel `p` cell `(i, j)` level `k` at `[i, j, p, k]`
# in `(Xdim, Ydim, nf, lev)` order. We therefore map
# `src[i, j, p, k_src]` directly onto interior panel `p` cell `(i, j)`, and
# flip the vertical (source SURFACE-first → model TOA-first) via
# `k = Nz - k_src + 1`. Equal level counts retain that exact copy contract.
# Mismatched counts require `ap`/`bp`, target surface pressure, and optionally
# source/target humidity for dry-carrier weighting.
# ---------------------------------------------------------------------------

function _build_cs_native_ic(grid::AtmosGrid{<:CubedSphereMesh},
                             air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                             cfg, ::Type{FT};
                             surface_pressure::Union{Nothing, NTuple{6, <:AbstractMatrix}} = nothing,
                             pressure_basis::AbstractSurfacePressureBasis = TotalSurfacePressure(),
                             qv::Union{Nothing, NTuple{6}} = nothing) where FT
    mesh = grid.horizontal
    Nc   = mesh.Nc
    Nz   = size(air_mass[1], 3)

    file = expand_data_path(String(get(cfg, "file", "")))
    variable = String(get(cfg, "variable", ""))
    isempty(file) && throw(ArgumentError("init.kind=cs_native requires init.file"))
    isempty(variable) && throw(ArgumentError("init.kind=cs_native requires init.variable"))
    isfile(file) || throw(ArgumentError("cs_native initial-condition file not found: $file"))
    time_index = Int(get(cfg, "time_index", 1))
    ds = NCDataset(file)
    raw, source_qv, ap, bp, has_hybrid = try
        haskey(ds, variable) || throw(ArgumentError("variable '$variable' not found in $file"))
        rv = ds[variable]
        nd = ndims(rv)
        # NCDatasets reverses CF dim order: (Xdim, Ydim, nf, lev[, time]).
        # Fill values become 0 (GriddedIO.MaskedAsZero), read straight into FT.
        values = if nd == 5
            read_raw(FT, ds, variable, MaskedAsZero(), :, :, :, :, time_index)
        elseif nd == 4
            read_raw(FT, ds, variable, MaskedAsZero())
        else
            throw(ArgumentError("init.kind=cs_native: variable '$variable' must be " *
                                "4D (Xdim,Ydim,nf,lev) or 5D (…,time), got ndims=$nd"))
        end

        humidity_name = haskey(cfg, "humidity_variable") ? String(cfg["humidity_variable"]) : nothing
        source_humidity = if humidity_name === nothing
            nothing
        else
            haskey(ds, humidity_name) || throw(ArgumentError(
                "humidity variable '$humidity_name' not found in $file"))
            qv_var = ds[humidity_name]
            ndims(qv_var) == nd && size(qv_var)[1:4] == size(rv)[1:4] ||
                throw(DimensionMismatch(
                    "humidity variable '$humidity_name' must share the cs_native IC " *
                    "field's Xdim, Ydim, panel, and level dimensions"))
            nd == 5 ? read_raw(FT, ds, humidity_name, MaskedAsZero(), :, :, :, :, time_index) :
                      read_raw(FT, ds, humidity_name, MaskedAsZero())
        end

        has_ap, has_bp = haskey(ds, "ap"), haskey(ds, "bp")
        has_ap == has_bp || throw(ArgumentError(
            "cs_native initial condition must provide both ap and bp, or neither"))
        source_ap = has_ap ? Float64.(ds["ap"][:]) : Float64[]
        source_bp = has_bp ? Float64.(ds["bp"][:]) : Float64[]
        has_ap && _validate_ap_units(ds["ap"], source_ap, file)
        values, source_humidity, source_ap, source_bp, has_ap
    finally
        close(ds)
    end

    size(raw, 1) == Nc || throw(DimensionMismatch(
        "cs_native: source Xdim=$(size(raw, 1)) != binary Nc=$Nc (no horizontal " *
        "regrid is performed; the source must be on the SAME cube as the binary)"))
    size(raw, 2) == Nc || throw(DimensionMismatch(
        "cs_native: source Ydim=$(size(raw, 2)) != binary Nc=$Nc"))
    size(raw, 3) == CS_PANEL_COUNT || throw(DimensionMismatch(
        "cs_native: source nf=$(size(raw, 3)) != $CS_PANEL_COUNT panels"))
    Nsrc = size(raw, 4)
    source_qv === nothing || all(v -> isfinite(v) && 0 <= v < 1, source_qv) ||
        throw(ArgumentError("cs_native humidity_variable must contain finite specific humidity in [0, 1)"))

    # Signed tracer contributions are part of the state contract. Preserve
    # negative values by default; physical-species workflows may explicitly
    # request cleanup of source-file noise with `clamp_negative = true`.
    clamp_negative = _config_bool(cfg, "clamp_negative", false, "initial-condition clamp_negative")

    vmr = ntuple(_ -> Array{FT}(undef, Nc, Nc, Nz), CS_PANEL_COUNT)

    surface_first = false
    if has_hybrid
        length(ap) == Nsrc + 1 && length(bp) == Nsrc + 1 || throw(DimensionMismatch(
            "cs_native hybrid coefficients must have source lev + 1 = $(Nsrc + 1) entries; " *
            "got ap=$(length(ap)), bp=$(length(bp))"))
        surface_first = _hybrid_surface_first(ap, bp)
        if haskey(cfg, "vertical_order")
            configured_order = cfg["vertical_order"]
            _parse_vertical_order(configured_order) ===
                (surface_first ? :surface_first : :toa_first) ||
                throw(ArgumentError("init.vertical_order = $(repr(configured_order)) contradicts " *
                                    "the hybrid coefficients of $(file)"))
        end
    end

    if Nsrc == Nz
        if has_hybrid
            A_tgt, B_tgt = grid.vertical.A, grid.vertical.B
            target_surface_first = _hybrid_surface_first(A_tgt, B_tgt)
            if surface_first != target_surface_first
                raw = raw[:, :, :, end:-1:1]
                source_qv === nothing || (source_qv = source_qv[:, :, :, end:-1:1])
                ap = reverse(ap)
                bp = reverse(bp)
            end
            all(isapprox.(ap, A_tgt; atol = 1e-3, rtol = 1e-6)) || throw(ArgumentError(
                "cs_native ap coefficients differ from the transport binary despite equal level counts"))
            all(isapprox.(bp, B_tgt; atol = 1e-8, rtol = 1e-6)) || throw(ArgumentError(
                "cs_native bp coefficients differ from the transport binary despite equal level counts"))
            flip_vertical = false
        else
            flip_vertical = _parse_vertical_order(
                get(cfg, "vertical_order", "surface_first")) === :surface_first
        end
        for p in 1:CS_PANEL_COUNT
            dst = vmr[p]
            @inbounds for k in 1:Nz
                k_src = flip_vertical ? (Nz - k + 1) : k
                for j in 1:Nc, i in 1:Nc
                    v = raw[i, j, p, k_src]
                    dst[i, j, k] = (clamp_negative && v < zero(FT)) ? zero(FT) : v
                end
            end
        end
        return vmr
    end

    has_hybrid || throw(ArgumentError(
        "cs_native initial condition has $Nsrc levels but the binary has $Nz; " *
        "mismatched levels require both hybrid interface coefficients ap and bp"))
    surface_pressure === nothing && throw(ArgumentError(
        "cs_native initial condition has $Nsrc levels but the binary has $Nz; " *
        "pass the binary's surface_pressure panels for conservative hybrid remapping"))
    all(p -> size(surface_pressure[p]) == (Nc, Nc), 1:CS_PANEL_COUNT) ||
        throw(DimensionMismatch("CS target surface-pressure panels must each have shape ($Nc, $Nc)"))
    qv === nothing || all(p -> size(qv[p]) == (Nc, Nc, Nz), 1:CS_PANEL_COUNT) ||
        throw(DimensionMismatch("CS target humidity panels must each have shape ($Nc, $Nc, $Nz)"))

    # The conservative remapper consumes source profiles surface-to-TOA.
    if !surface_first
        raw = raw[:, :, :, end:-1:1]
        source_qv === nothing || (source_qv = source_qv[:, :, :, end:-1:1])
        ap = reverse(ap)
        bp = reverse(bp)
    end

    _warn_surface_pressure_approximation(pressure_basis, qv)
    A_tgt, B_tgt = grid.vertical.A, grid.vertical.B
    src_profile = Vector{FT}(undef, Nsrc)
    src_qv_profile = similar(src_profile)
    src_p = Vector{Float64}(undef, Nsrc + 1)
    cumulative = similar(src_p)
    dry_cumulative = similar(src_p)
    for p in 1:CS_PANEL_COUNT, j in 1:Nc, i in 1:Nc
        @inbounds for k in 1:Nsrc
            value = raw[i, j, p, k]
            src_profile[k] = clamp_negative && value < zero(FT) ? zero(FT) : value
            source_qv === nothing || (src_qv_profile[k] = source_qv[i, j, p, k])
        end
        target_qv = qv === nothing ? nothing : @view(qv[p][i, j, :])
        ps_total = _total_surface_pressure(
            pressure_basis, surface_pressure[p][i, j], A_tgt, B_tgt, target_qv)
        source_qv_profile_or_nothing = source_qv === nothing ? nothing : src_qv_profile
        _remap_hybrid_profile!(@view(vmr[p][i, j, :]), src_profile,
                               source_qv_profile_or_nothing, ap, bp,
                               A_tgt, B_tgt, ps_total,
                               src_p, cumulative, dry_cumulative)
    end
    return vmr
end

# ---------------------------------------------------------------------------
# pressure-layer single-layer IC — places a constant VMR in one model
# layer per column (selected by psurf_fraction × ps_column, or k=Nz when
# `lowest_layer = true`). VMR is computed globally so that the total
# molecule count across all cells matches `total_molecules`.
#
# Used by the convection-comparison experiment (TM5 vs GCHP CMFMC):
# four tracers, each in a different vertical slab at lowest /
# 0.8 / 0.6 / 0.4 × ps_surf, with identical molecule counts so cross-
# tracer redistribution can be compared directly.
# ---------------------------------------------------------------------------

function _build_cs_pressure_layer_ic(air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                      grid::AtmosGrid{<:CubedSphereMesh},
                                      cfg, ::Type{FT},
                                      surface_pressure::NTuple{6, <:AbstractMatrix},
                                      pressure_basis::AbstractSurfacePressureBasis,
                                      qv::Union{Nothing, NTuple{6}},
                                      vmr = nothing) where FT
    mesh = grid.horizontal
    Nc   = mesh.Nc
    Hp   = mesh.Hp
    Nz   = size(air_mass[1], 3)
    A    = grid.vertical.A
    B    = grid.vertical.B

    lowest_layer = _config_bool(cfg, "lowest_layer", false, "initial-condition lowest_layer")
    # Assert the converted configuration types before entering the column loops.
    # Values read from a heterogeneous TOML dictionary otherwise box hot-loop arithmetic.
    psurf_fraction = (lowest_layer ? FT(NaN) :
                      FT(get(cfg, "psurf_fraction", 0.5)))::FT
    total_molecules = Float64(get(cfg, "total_molecules", 1.0e22))::Float64

    if !lowest_layer
        (FT(0) < psurf_fraction <= FT(1)) || throw(ArgumentError(
            "init.kind=pressure_layer requires 0 < psurf_fraction <= 1; got $(psurf_fraction)"))
    end
    qv === nothing || all(p -> size(qv[p]) == (Nc, Nc, Nz), 1:CS_PANEL_COUNT) ||
        throw(DimensionMismatch("CS target humidity panels must each have shape ($Nc, $Nc, $Nz)"))
    !lowest_layer && _warn_surface_pressure_approximation(pressure_basis, qv)
    total_molecules > 0 || throw(ArgumentError(
        "init.kind=pressure_layer requires total_molecules > 0; got $(total_molecules)"))

    # air_mass is halo-padded `(Nc+2Hp, Nc+2Hp, Nz)` (see _cs_pack_interior_into_halo);
    # surface_pressure is interior `(Nc, Nc)` per panel. We index the interior
    # of air_mass via `[Hp+i, Hp+j, :]` so the per-column dry mass we sum matches
    # the cells whose VMR we set.
    k_target = ntuple(_ -> Array{Int}(undef, Nc, Nc), CS_PANEL_COUNT)
    if lowest_layer
        for p in 1:CS_PANEL_COUNT
            fill!(k_target[p], Nz)
        end
    else
        for p in 1:CS_PANEL_COUNT
            ps = surface_pressure[p]
            for j in 1:Nc, i in 1:Nc
                qv_col = qv === nothing ? nothing : @view(qv[p][i, j, :])
                ps_col = _total_surface_pressure(
                    pressure_basis, ps[i, j], A, B, qv_col)
                target_p = psurf_fraction * ps_col
                k_best, dp_best = 1, Inf
                for k in 1:Nz
                    p_top = Float64(A[k])   + Float64(B[k])   * ps_col
                    p_bot = Float64(A[k+1]) + Float64(B[k+1]) * ps_col
                    p_mid = sqrt(max(p_top, eps()) * p_bot)  # log-midpoint
                    dp = abs(p_mid - target_p)
                    if dp < dp_best
                        dp_best = dp
                        k_best = k
                    end
                end
                k_target[p][i, j] = k_best
            end
        end
    end

    # Total dry-air mass in the chosen layer across all (panel, i, j).
    # air_mass is halo-padded, so the interior cell `(i, j)` is `[Hp+i, Hp+j, :]`.
    total_dry_mass = 0.0
    for p in 1:CS_PANEL_COUNT
        m = air_mass[p]
        kt = k_target[p]
        for j in 1:Nc, i in 1:Nc
            total_dry_mass += Float64(m[Hp + i, Hp + j, kt[i, j]])
        end
    end
    total_dry_mass > 0 || throw(ArgumentError(
        "init.kind=pressure_layer: target-layer dry mass summed to zero; binary may be empty"))

    # VMR (mol_co2 / mol_air) chosen so Σ molecules = total_molecules.
    #   molecules_per_cell = VMR × N_A × dry_mass_per_cell / M_air
    #   total_molecules    = VMR × N_A × Σ dry_mass / M_air
    vmr_value = FT(total_molecules * DRY_AIR_MOLAR_MASS /
                   (AVOGADRO * total_dry_mass))

    # Build the VMR panels (interior-shaped `(Nc, Nc, Nz)`): zero except
    # in the chosen layer per column.
    vmr = _cs_initial_vmr_storage(vmr, FT, Nc, Nz)
    foreach(panel -> fill!(panel, zero(FT)), vmr)
    for p in 1:CS_PANEL_COUNT
        kt = k_target[p]
        for j in 1:Nc, i in 1:Nc
            vmr[p][i, j, kt[i, j]] = vmr_value
        end
    end
    return vmr
end

# ---------------------------------------------------------------------------
# CS pack_initial_tracer_mass
#
# Takes interior `NTuple{6, (Nc, Nc, Nz)}` VMR + halo-padded
# `NTuple{6, (Nc+2Hp, Nc+2Hp, Nz)}` air_mass. Returns halo-padded tracer
# mass with the halo ring zeroed; halo exchanges during the run populate
# those cells.
# ---------------------------------------------------------------------------

function _pack_tracer_mass(grid::AtmosGrid{<:CubedSphereMesh},
                           air_mass::NTuple{6, <:AbstractArray},
                           vmr_dry::NTuple{6, <:AbstractArray},
                           ::DryBasis,
                           qv)
    return _cs_pack_interior_into_halo(grid, air_mass, vmr_dry, nothing)
end

function _pack_tracer_mass(grid::AtmosGrid{<:CubedSphereMesh},
                           air_mass::NTuple{6, <:AbstractArray},
                           vmr_dry::NTuple{6, <:AbstractArray},
                           ::MoistBasis,
                           qv)
    qv === nothing && throw(ArgumentError(
        "pack_initial_tracer_mass on MoistBasis requires qv (specific humidity) " *
        "from the first transport window to convert dry VMR to moist storage; got qv=nothing."))
    qv isa NTuple{6} || throw(ArgumentError(
        "CS pack_initial_tracer_mass on MoistBasis requires qv::NTuple{6}; " *
        "got $(typeof(qv))"))
    return _cs_pack_interior_into_halo(grid, air_mass, vmr_dry, qv)
end

function _cs_pack_interior_into_halo(grid::AtmosGrid{<:CubedSphereMesh},
                                     air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                     vmr::NTuple{6, <:AbstractArray{FT, 3}},
                                     qv::Union{Nothing, NTuple{6}}) where FT
    out = ntuple(p -> Array{FT}(undef, size(air_mass[p])), CS_PANEL_COUNT)
    return _cs_pack_interior_into_halo!(out, grid, air_mass, vmr, qv)
end

# Destination panels must be independent of air_mass, vmr, and qv. The allocating
# API owns its output; the driven runner supplies views into fresh packed storage.
function _cs_pack_interior_into_halo!(out::NTuple{6, <:AbstractArray{FT, 3}},
                                      grid::AtmosGrid{<:CubedSphereMesh},
                                      air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                      vmr::NTuple{6, <:AbstractArray{FT, 3}},
                                      qv::Union{Nothing, NTuple{6}}) where FT
    mesh = grid.horizontal
    Nc = mesh.Nc
    Hp = mesh.Hp
    for p in 1:CS_PANEL_COUNT
        size(out[p]) == size(air_mass[p]) || throw(DimensionMismatch(
            "CS panel $p: destination has shape $(size(out[p])), expected $(size(air_mass[p]))"))
        size(vmr[p]) == (Nc, Nc, size(air_mass[p], 3)) || throw(DimensionMismatch(
            "CS panel $p: vmr has shape $(size(vmr[p])), expected $((Nc, Nc, size(air_mass[p], 3)))"))
        fill!(out[p], zero(FT))
        interior_am = @view air_mass[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]
        interior_out = @view out[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]
        if qv === nothing
            interior_out .= vmr[p] .* interior_am
        else
            size(qv[p]) == size(air_mass[p]) || throw(DimensionMismatch(
                "CS panel $p: qv has shape $(size(qv[p])), expected $(size(air_mass[p]))"))
            interior_qv = @view qv[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]
            interior_out .= vmr[p] .* interior_am .* (1 .- interior_qv)
        end
    end
    return out
end
