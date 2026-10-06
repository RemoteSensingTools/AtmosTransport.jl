# Cubed-sphere validation and panel-native loading specializations for the
# common `TransportBinaryDriver` and `TransportWindow` types.

_supports_runtime_diffusion(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) = has_surface(reader) || :dkg in reader.header.payload_sections

function _validate_replay_consistency_cs(
    reader::TransportBinaryReader{FT, DiskFT, CubedSphereBinaryGeometry},
) where {FT, DiskFT}
    if get(ENV, "ATMOSTR_NO_REPLAY_CHECK", "0") == "1"
        return nothing
    end
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
            "Regenerate the binary with the CS replay-safe preprocessor or bypass " *
            "with ENV[\"ATMOSTR_NO_REPLAY_CHECK\"]=\"1\" for diagnostic runs."
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
