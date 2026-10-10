# Cubed-sphere validation and panel-native loading specializations for the
# common `TransportBinaryDriver` and `TransportWindow` types.

_supports_runtime_diffusion(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) = has_surface(reader) || :dkg in reader.header.payload_sections

function _validate_replay_consistency_cs(
    reader::TransportBinaryReader{FT, DiskFT, CubedSphereBinaryGeometry},
) where {FT, DiskFT}
    tol_rel = replay_tolerance(FT)
    Nt = window_count(reader)
    Nt >= 1 || return nothing

    worst_rel = 0.0
    worst_abs = 0.0
    worst_win = 0
    worst_idx = (0, 0, 0, 0)

    for k in 1:Nt
        cur = load_window!(reader, k)
        steps = reader.header.steps_per_window_by_window[k]
        if flux_kind(reader) === :full_window_mass_amount
            scale = FT(1) / FT(2 * steps)
            _scale_cs_replay_panels!(cur.am, scale)
            _scale_cs_replay_panels!(cur.bm, scale)
            _scale_cs_replay_panels!(cur.cm, scale)
        end
        m_target = if k < Nt
            load_window!(reader, k + 1).m
        elseif has_flux_delta(reader)
            deltas = load_flux_delta_window!(reader, k)
            if deltas === nothing || !haskey(deltas, :dm)
                cur.m
            else
                ntuple(p -> cur.m[p] .+ deltas.dm[p], length(cur.m))
            end
        else
            cur.m
        end
        diag = verify_window_continuity_cs(cur.m, cur.am, cur.bm, cur.cm, m_target, steps)
        if diag.max_rel_err > worst_rel
            worst_rel = diag.max_rel_err
            worst_abs = diag.max_abs_err
            worst_win = k
            worst_idx = diag.worst_idx
        end
    end

    worst_rel <= tol_rel ||
        throw(ArgumentError(
            "TransportBinaryDriver replay-consistency gate FAILED for " *
            "$(basename(reader.path)): rel=$(worst_rel) > tol=$(tol_rel) at window " *
            "$worst_win cell $worst_idx (abs=$worst_abs kg). Stored CS fluxes do not " *
            "integrate to the stored mass target under palindrome continuity. " *
            "Regenerate the binary with the CS replay-safe preprocessor, or skip this " *
            "load-time check (remove [input] validate_replay = true, or the " *
            "`validate_replay` keyword, or unset ATMOSTR_REPLAY_CHECK)."
        ))

    @info "Replay continuity gate passed: $(basename(reader.path)) " *
          "topology=cubed_sphere worst_rel=$(worst_rel) worst_window=$(worst_win)"
    return (worst_window = worst_win, worst_rel = worst_rel, worst_abs = worst_abs)
end

@inline function _scale_cs_replay_panels!(panels::NTuple{6}, scale)
    @inbounds for p in 1:6
        panels[p] .*= scale
    end
    return panels
end

function _transport_driver_grid(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
    ::CubedSphereBinaryGeometry; FT, arch, Hp,
)
    return load_grid(reader; FT, arch, Hp)
end

function _validate_driver_replay(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
    ::CubedSphereBinaryGeometry, _grid,
)
    return _validate_replay_consistency_cs(reader)
end

@inline _cs_basis_type(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) =
    mass_basis(reader) === :dry ? DryBasis : MoistBasis

@inline function _pad_horizontal(a::AbstractArray{T, N}, Hp::Int) where {T, N}
    dims = ntuple(d -> d <= 2 ? size(a, d) + 2 * Hp : size(a, d), N)
    padded = zeros(T, dims...)
    ranges = ntuple(d -> d <= 2 ? ((Hp + 1):(Hp + size(a, d))) : axes(a, d), N)
    padded[ranges...] .= a
    return padded
end

@inline function _copy_cs_storage!(dest::NTuple{6}, src::NTuple{6})
    @inbounds for p in 1:6
        copyto!(dest[p], src[p])
    end
    return dest
end

@inline function copy_fluxes!(dest::CubedSphereFaceFluxState, src::CubedSphereFaceFluxState)
    _copy_cs_storage!(dest.am, src.am)
    _copy_cs_storage!(dest.bm, src.bm)
    _copy_cs_storage!(dest.cm, src.cm)
    return dest
end

function interpolate_fluxes!(dest::CubedSphereFaceFluxState,
                             window::TransportWindow, λ::Real)
    return copy_fluxes!(dest, window.fluxes)
end

function expected_air_mass!(dest::NTuple{6}, window::TransportWindow, λ::Real)
    _copy_cs_storage!(dest, window.air_mass)
    window.deltas === nothing && return dest
    λ_ft = convert(eltype(dest[1]), λ)
    @inbounds for p in 1:6
        @. dest[p] = dest[p] + λ_ft * window.deltas.dm[p]
    end
    return dest
end

# ---------------------------------------------------------------------------
# In-place window loading into an existing host window
#
# GPU runs copy every loaded window to the device, so the host window is only
# a staging buffer. `load_transport_window!` refills one such window instead
# of allocating new panel arrays (and their padded copies) for every window:
# each payload section is copied straight into its destination, padded fields
# into their interior. Halo cells are never written, so they keep the zeros of
# the first load and the result equals `load_transport_window`.
# ---------------------------------------------------------------------------

# Destination of every payload section in an existing cubed-sphere window:
# padded fields receive their interior, the others the whole panel.
function _cs_section_targets(w::TransportWindow, Hp::Int)
    interior(a) = view(a, (Hp + 1):(size(a, 1) - Hp), (Hp + 1):(size(a, 2) - Hp), :)
    targets = Dict{Symbol, Any}(:m  => map(interior, w.air_mass),
                                :ps => w.surface_pressure,
                                :am => map(interior, w.fluxes.am),
                                :bm => map(interior, w.fluxes.bm),
                                :cm => map(interior, w.fluxes.cm))
    w.deltas === nothing || (targets[:dm] = map(interior, w.deltas.dm))
    c = w.convection
    if c !== nothing
        c.cmfmc === nothing || (targets[:cmfmc] = c.cmfmc)
        c.dtrain === nothing || (targets[:dtrain] = c.dtrain)
        c.cloud_base === nothing || (targets[:cmfmc_cloud_base] = c.cloud_base)
        if c.tm5_fields !== nothing
            for name in (:entu, :detu, :entd, :detd)
                targets[name] = getfield(c.tm5_fields, name)
            end
        end
    end
    sf = w.surface
    if sf !== nothing
        targets[:pblh] = sf.pblh
        targets[:ustar] = sf.ustar
        targets[:pbl_hflux] = sf.hflux
        targets[:t2m] = sf.t2m
        sf.eflux === nothing || (targets[:pbl_eflux] = sf.eflux)
    end
    v = w.vdiff
    if v !== nothing
        targets[:vdiff_u] = v.u
        targets[:vdiff_v] = v.v
        targets[:vdiff_t] = v.t
        targets[:vdiff_qv] = v.qv
    end
    w.dkg === nothing || (targets[:dkg] = w.dkg)
    return targets
end

# One panel of `n` elements starting after offset `o` of the mmap'd payload.
@inline _copy_cs_panel!(dst::Array, data, o::Int, n::Int) = copyto!(dst, 1, data, o + 1, n)

# Interior of a padded 3-D panel: one contiguous row (the fastest index) at a time.
function _copy_cs_panel!(dst::SubArray{<:Any, 3, <:Array}, data, o::Int, n::Int)
    a = parent(dst)
    i0, j0 = first(dst.indices[1]), first(dst.indices[2])
    nx, ny, nz = size(dst)
    nx * ny * nz == n || throw(DimensionMismatch(
        "cubed-sphere panel section has $(n) elements; the window interior holds $(nx * ny * nz)"))
    li = LinearIndices(a)
    src = o
    @inbounds for k in 1:nz, j in 1:ny
        copyto!(a, li[i0, j0 + j - 1, k], data, src + 1, nx)
        src += nx
    end
    return dst
end

"""
    load_transport_window!(window, driver, win) -> window

Refill an existing host-resident cubed-sphere `window` (from
`load_transport_window` on the same binary) with window `win`, without
allocating new field arrays. Equal to `load_transport_window(driver, win)`
provided the halos of `window`'s padded fields still hold the zeros of that
first load: only interiors are written.
"""
function load_transport_window!(
    w::TransportWindow,
    driver::TransportBinaryDriver{FT, ReaderT, <:AtmosGrid{<:CubedSphereMesh}},
    win::Int,
) where {FT, ReaderT}
    reader = driver.reader
    h = reader.header
    np = h.geometry.npanel
    targets = _cs_section_targets(w, driver.grid.horizontal.Hp)
    o = _transport_window_offset(reader, win)
    for section in h.payload_sections
        n = _cs_section_elements(h, section)
        dest = get(targets, section, nothing)
        if dest !== nothing
            per_panel = n ÷ np
            for p in 1:np
                _copy_cs_panel!(dest[p], reader.data, o + (p - 1) * per_panel, per_panel)
            end
        end
        o += n
    end
    return w
end

function load_transport_window(
    driver::TransportBinaryDriver{FT, ReaderT, <:AtmosGrid{<:CubedSphereMesh}},
    win::Int,
) where {FT, ReaderT}
    raw = load_window!(driver.reader, win)
    Hp = driver.grid.horizontal.Hp
    panels_m = ntuple(p -> _pad_horizontal(raw.m[p], Hp), 6)
    panels_ps = raw.ps
    panels_am = ntuple(p -> _pad_horizontal(raw.am[p], Hp), 6)
    panels_bm = ntuple(p -> _pad_horizontal(raw.bm[p], Hp), 6)
    panels_cm = ntuple(p -> _pad_horizontal(raw.cm[p], Hp), 6)
    basis = _cs_basis_type(driver.reader)
    fluxes = CubedSphereFaceFluxState{basis}(panels_am, panels_bm, panels_cm)
    # `raw.tm5_fields` is a NamedTuple of per-panel
    # NTuples `(entu, detu, entd, detd)` when the binary carries TM5
    # sections, or `nothing` otherwise. The runtime validator in
    # DrivenSimulation decides whether TM5Convection can run against
    # this forcing; constructing ConvectionForcing here is
    # capability-preserving (present stays present, absent stays
    # absent).
    has_cmfmc_fwd = raw.cmfmc !== nothing
    has_tm5_fwd   = raw.tm5_fields !== nothing
    convection = if has_cmfmc_fwd || has_tm5_fwd
        ConvectionForcing(raw.cmfmc, raw.dtrain, raw.tm5_fields, raw.cmfmc_cloud_base)
    else
        nothing
    end
    delta_raw = load_flux_delta_window!(driver.reader, win)
    deltas = delta_raw === nothing ? nothing : CubedSphereFluxDeltas(
        ntuple(p -> _pad_horizontal(delta_raw.dm[p], Hp), 6))
    return TransportWindow(panels_m, panels_ps, fluxes;
                           deltas,
                           convection,
                           surface = raw.surface,
                           vdiff = raw.vdiff,
                           dkg = raw.dkg)
end
