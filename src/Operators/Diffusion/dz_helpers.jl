# ---------------------------------------------------------------------------
# Layer-thickness fill for the implicit-vertical-diffusion workspace.
#
# `apply_vertical_diffusion_vmr!` consumes `workspace.layer_thickness` as the current
# layer thickness in metres. The workspace allocator initializes that array
# to zeros, so the runtime must populate it before each diffusion call —
# otherwise the kernel divides by zero and the entire tracer field NaNs out
# from the next snapshot onward.
#
# Two hydrostatic fills are provided:
#
#   * `fill_dz_hydrostatic_constT!` — constant
#     `T_ref = HYDROSTATIC_REFERENCE_TEMPERATURE` (260 K):
#         delp[i,j,k] = (ak[k+1] - ak[k]) + (bk[k+1] - bk[k]) * ps[i,j]
#         p_ctr[i,j,k] = ½·((ak[k] + ak[k+1]) + (bk[k] + bk[k+1])·ps[i,j])
#         dz[i,j,k]   = R · T_ref / g · delp / p_ctr
#     Related to the preprocessing-side `dz_hydrostatic_constT!` in
#     `src/Preprocessing/tm5_convection_conversion.jl`, whose rounding
#     order and TOA guard differ. Default for any
#     configuration without VDIFF payload (legacy LL / RG / WindowPBLKzField).
#
#   * `fill_dz_hydrostatic_virtualT!` — virtual T from VDIFF:
#         T_v[k]      = T[k] · (1 + VIRTUAL_TEMPERATURE_FACTOR · qv[k])  (qv ≥ 0 clamped)
#         dz[i,j,k]   = R · T_v[k] / g · delp / p_ctr
#     Used by the `LocalHoltslagBovilleKzField` runtime path so the
#     solver layer thickness shares the same column geometry the Kz cache
#     itself uses (closes the virtual-T inconsistency from the audit memo).
#
# Both fills produce the same `dz` units (metres) and the same matrix
# coefficient interpretation in the mass-flux kernel.
#
# `dz` only depends on `(ps, ak/bk, T_v?)`, all window-constant, so the
# fill runs once per met window. `DrivenSimulation._refresh_dz_for_window!`
# dispatches between the two variants via `_fill_dz_for_diffusion!` based
# on the operator's Kz field type.
# ---------------------------------------------------------------------------

"""
    stage_hybrid_coefficients!(coefficients, A, B)

Copy hybrid interface coefficients to persistent backend storage in its own
precision and cache the coordinate traits used by pressure recovery.
"""
function stage_hybrid_coefficients!(coefficients::HybridCoefficientWorkspace,
                                     ak_ifc::AbstractVector,
                                     bk_ifc::AbstractVector)
    length(coefficients.A) == length(ak_ifc) || throw(DimensionMismatch(
        "coefficient workspace length $(length(coefficients.A)) ≠ A length $(length(ak_ifc))"))
    length(coefficients.B) == length(bk_ifc) || throw(DimensionMismatch(
        "coefficient workspace length $(length(coefficients.B)) ≠ B length $(length(bk_ifc))"))
    copyto!(coefficients.A, eltype(coefficients.A).(ak_ifc))
    copyto!(coefficients.B, eltype(coefficients.B).(bk_ifc))
    coefficients.pure_pressure = all(iszero, bk_ifc)
    db = diff(bk_ifc)
    coefficients.positive_b_coordinate =
        all(>=(zero(eltype(db))), db) && any(>(zero(eltype(db))), db)
    return coefficients
end

@kernel function _recover_total_ps_2d_kernel!(ps_total, @Const(ps_dry),
                                              @Const(qv), @Const(ak_ifc),
                                              @Const(bk_ifc), status, Nz::Int)
    i, j = @index(Global, NTuple)
    FT = eltype(ps_total)
    offset = zero(FT)
    slope = zero(FT)
    @inbounds for k in 1:Nz
        dry_fraction = one(FT) - qv[i, j, k]
        offset += dry_fraction * (ak_ifc[k + 1] - ak_ifc[k])
        slope += dry_fraction * (bk_ifc[k + 1] - bk_ifc[k])
    end
    if !(isfinite(slope) && slope > zero(FT))
        @atomic status[1] = one(eltype(status))
    end
    @inbounds ps_total[i, j] = (ps_dry[i, j] - offset) / slope
end

@kernel function _recover_total_ps_1d_kernel!(ps_total, @Const(ps_dry),
                                              @Const(qv), @Const(ak_ifc),
                                              @Const(bk_ifc), status, Nz::Int)
    c = @index(Global)
    FT = eltype(ps_total)
    offset = zero(FT)
    slope = zero(FT)
    @inbounds for k in 1:Nz
        dry_fraction = one(FT) - qv[c, k]
        offset += dry_fraction * (ak_ifc[k + 1] - ak_ifc[k])
        slope += dry_fraction * (bk_ifc[k + 1] - bk_ifc[k])
    end
    if !(isfinite(slope) && slope > zero(FT))
        @atomic status[1] = one(eltype(status))
    end
    @inbounds ps_total[c] = (ps_dry[c] - offset) / slope
end

@inline function _recovery_status(reference)
    status = similar(reference, Int32, 1)
    fill!(status, zero(Int32))
    return status
end

function _check_recovery_status(status)
    only(Array(status)) == zero(eltype(status)) || throw(ArgumentError(
        "cannot recover total surface pressure: humidity-weighted hybrid-B slope " *
        "must be finite and positive in every column"))
    return nothing
end

function _validate_recovery_panel(dest, ps_dry, qv, nA::Int, nB::Int;
                                  label::AbstractString = "")
    prefix = isempty(label) ? "" : "$label "
    ndims(dest) == 2 || throw(DimensionMismatch("$(prefix)destination must be rank 2"))
    ndims(ps_dry) == 2 || throw(DimensionMismatch("$(prefix)dry ps must be rank 2"))
    ndims(qv) == 3 || throw(DimensionMismatch("$(prefix)humidity must be rank 3"))
    Nx, Ny = size(dest)
    size(ps_dry) == (Nx, Ny) || throw(DimensionMismatch("$(prefix)dry ps shape mismatch"))
    size(qv)[1:2] == (Nx, Ny) ||
        throw(DimensionMismatch("$(prefix)humidity horizontal shape mismatch"))
    Nz = size(qv, 3)
    nA == Nz + 1 && nB == Nz + 1 || throw(DimensionMismatch(
        "$(prefix)humidity levels must match hybrid coefficients"))
    return nothing
end

function _validate_recovery_panels(dest::NTuple{6}, ps_dry::NTuple{6},
                                   qv::NTuple{6}, nA::Int, nB::Int)
    @inbounds for p in 1:6
        _validate_recovery_panel(dest[p], ps_dry[p], qv[p], nA, nB;
                                 label = "panel $p")
    end
    return nothing
end

"""
    recover_total_surface_pressure_staged!(dest, ps_dry, qv, coefficients; sync)

Recover total surface pressure from the humidity-weighted hybrid-coordinate
identity `ps_dry = sum((1-qv) * (ΔA + ΔB*ps_total))`. An A-only coordinate
copies `ps_dry` unchanged. With `sync=true`, reject a nonfinite or nonpositive
humidity-weighted B slope after the launch. Supports structured, face-indexed
and six-panel arrays; `coefficients` comes from `stage_hybrid_coefficients!`.
"""
function recover_total_surface_pressure_staged!(dest::AbstractMatrix,
                                                  ps_dry::AbstractMatrix,
                                                  qv::AbstractArray{<:AbstractFloat, 3},
                                                  coefficients::HybridCoefficientWorkspace;
                                                  sync::Bool)
    _validate_recovery_panel(dest, ps_dry, qv, length(coefficients.A), length(coefficients.B))
    if coefficients.pure_pressure
        copyto!(dest, ps_dry)
        sync && synchronize(get_backend(dest))
        return dest
    end
    Nx, Ny = size(dest)
    Nz = size(qv, 3)
    backend = get_backend(dest)
    status = _recovery_status(dest)
    launch!(_recover_total_ps_2d_kernel!, backend, (8, 8), (Nx, Ny),
            dest, ps_dry, qv, coefficients.A, coefficients.B, status, Nz; sync)
    sync && _check_recovery_status(status)
    return dest
end

function recover_total_surface_pressure_staged!(dest::AbstractVector,
                                                  ps_dry::AbstractVector,
                                                  qv::AbstractMatrix,
                                                  coefficients::HybridCoefficientWorkspace;
                                                  sync::Bool)
    ncells = length(dest)
    length(ps_dry) == ncells || throw(DimensionMismatch("dry ps length mismatch"))
    size(qv, 1) == ncells || throw(DimensionMismatch("humidity cell count mismatch"))
    Nz = size(qv, 2)
    length(coefficients.A) == Nz + 1 && length(coefficients.B) == Nz + 1 ||
        throw(DimensionMismatch("humidity levels must match hybrid coefficients"))
    if coefficients.pure_pressure
        copyto!(dest, ps_dry)
        sync && synchronize(get_backend(dest))
        return dest
    end
    ncells, Nz = size(qv)
    backend = get_backend(dest)
    status = _recovery_status(dest)
    launch!(_recover_total_ps_1d_kernel!, backend, 256, ncells,
            dest, ps_dry, qv, coefficients.A, coefficients.B, status, Nz; sync)
    sync && _check_recovery_status(status)
    return dest
end


function recover_total_surface_pressure_staged!(dest::NTuple{6},
                                                  ps_dry::NTuple{6},
                                                  qv::NTuple{6},
                                                  coefficients::HybridCoefficientWorkspace;
                                                  sync::Bool)
    _validate_recovery_panels(dest, ps_dry, qv, length(coefficients.A),
                              length(coefficients.B))
    backend = get_backend(dest[1])
    if coefficients.pure_pressure
        @inbounds for p in 1:6
            copyto!(dest[p], ps_dry[p])
        end
        sync && synchronize(backend)
        return dest
    end
    status = _recovery_status(dest[1])
    @inbounds for p in 1:6
        Nx, Ny = size(dest[p])
        Nz = size(qv[p], 3)
        launch!(_recover_total_ps_2d_kernel!, backend, (8, 8), (Nx, Ny),
                dest[p], ps_dry[p], qv[p], coefficients.A, coefficients.B,
                status, Nz; sync = false)
    end
    if sync
        synchronize(backend)
        _check_recovery_status(status)
    end
    return dest
end

function _validate_constT_panel(dz, ps, nA::Int, nB::Int;
                                label::AbstractString = "")
    prefix = isempty(label) ? "" : "$label "
    ndims(dz) == 3 || throw(DimensionMismatch("$(prefix)dz must be rank 3"))
    ndims(ps) == 2 || throw(DimensionMismatch("$(prefix)ps must be rank 2"))
    Nx, Ny, Nz = size(dz)
    size(ps) == (Nx, Ny) || throw(DimensionMismatch(
        "$(prefix)ps shape $(size(ps)) ≠ ($Nx, $Ny) expected from dz"))
    nA == Nz + 1 || throw(DimensionMismatch(
        "$(prefix)ak_ifc length $nA ≠ Nz+1=$(Nz + 1)"))
    nB == Nz + 1 || throw(DimensionMismatch(
        "$(prefix)bk_ifc length $nB ≠ Nz+1=$(Nz + 1)"))
    return nothing
end

function _validate_constT_panels(dz_panels::NTuple{6}, ps_panels::NTuple{6},
                                 nA::Int, nB::Int)
    @inbounds for p in 1:6
        _validate_constT_panel(dz_panels[p], ps_panels[p], nA, nB;
                               label = "panel $p")
    end
    return nothing
end

@kernel function _dz_hydrostatic_constT_kernel!(dz, @Const(ps),
                                                @Const(ak_ifc), @Const(bk_ifc),
                                                T_ref, R, g, Nz::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ak_lo = ak_ifc[k]
        ak_hi = ak_ifc[k + 1]
        bk_lo = bk_ifc[k]
        bk_hi = bk_ifc[k + 1]
        delp  = (ak_hi - ak_lo) + (bk_hi - bk_lo) * ps[i, j]
        p_ctr = (ak_lo + ak_hi + (bk_lo + bk_hi) * ps[i, j]) * (one(eltype(dz)) / 2)
        dz[i, j, k] = R * T_ref / g * delp / p_ctr
    end
end

"""
    fill_dz_hydrostatic_constT_staged!(dz, ps, coefficients; sync, T_ref, R, gravity)

Fill hydrostatic layer thickness `R*T_ref/gravity * Δp/p_center` from staged
hybrid coefficients and total surface pressure. The default temperature is
`HYDROSTATIC_REFERENCE_TEMPERATURE`. Supports structured, face-indexed and
six-panel arrays; coefficient storage is reused between window refreshes.
"""
function fill_dz_hydrostatic_constT_staged!(dz::AbstractArray{<:AbstractFloat, 3},
                                              ps::AbstractArray{<:AbstractFloat, 2},
                                              coefficients::HybridCoefficientWorkspace;
                                              T_ref::Real = HYDROSTATIC_REFERENCE_TEMPERATURE,
                                              R::Real = R_DRY_AIR,
                                              gravity::Real = STANDARD_GRAVITY,
                                              sync::Bool)
    _validate_constT_panel(dz, ps, length(coefficients.A), length(coefficients.B))
    Nx, Ny, Nz = size(dz)
    FT = eltype(dz)
    backend = get_backend(dz)
    launch!(_dz_hydrostatic_constT_kernel!, backend, (8, 8, 1), (Nx, Ny, Nz), dz, ps,
            coefficients.A, coefficients.B, FT(T_ref), FT(R), FT(gravity), Nz; sync)
    return dz
end

function fill_dz_hydrostatic_constT_staged!(dz_panels::NTuple{6},
                                              ps_panels::NTuple{6},
                                              coefficients::HybridCoefficientWorkspace;
                                              sync::Bool, kwargs...)
    _validate_constT_panels(dz_panels, ps_panels, length(coefficients.A),
                            length(coefficients.B))
    @inbounds for p in 1:6
        fill_dz_hydrostatic_constT_staged!(dz_panels[p], ps_panels[p], coefficients;
                                            kwargs..., sync = false)
    end
    sync && synchronize(get_backend(dz_panels[1]))
    return dz_panels
end

function _validate_virtualT_panel(dz, t_lyr, qv_lyr, ps, nA::Int, nB::Int;
                                  label::AbstractString = "")
    _validate_constT_panel(dz, ps, nA, nB; label)
    prefix = isempty(label) ? "" : "$label "
    expected = size(dz)
    ndims(t_lyr) == 3 || throw(DimensionMismatch("$(prefix)t_lyr must be rank 3"))
    ndims(qv_lyr) == 3 || throw(DimensionMismatch("$(prefix)qv_lyr must be rank 3"))
    size(t_lyr) == expected || throw(DimensionMismatch(
        "$(prefix)t_lyr shape $(size(t_lyr)) ≠ $expected expected from dz"))
    size(qv_lyr) == expected || throw(DimensionMismatch(
        "$(prefix)qv_lyr shape $(size(qv_lyr)) ≠ $expected expected from dz"))
    return nothing
end

function _validate_virtualT_panels(dz_panels::NTuple{6}, t_panels::NTuple{6},
                                   qv_panels::NTuple{6}, ps_panels::NTuple{6},
                                   nA::Int, nB::Int)
    @inbounds for p in 1:6
        _validate_virtualT_panel(dz_panels[p], t_panels[p], qv_panels[p],
                                 ps_panels[p], nA, nB; label = "panel $p")
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Virtual-temperature hydrostatic dz.
#
# When VDIFF fields are present (LocalHoltslagBovilleKzField on CS), the
# Kz cache already uses virtual T per layer to compute its own column
# geometry. Populating `dz_scratch` from the SAME virtual T closes the
# previous inconsistency where the kernel divided by `dz` from a 260 K
# constant-T column while Kz had been computed on a layer-varying virtual-T
# column. The two paths now share the same vertical scale.
#
#     T_v[k] = T[k] · (1 + VIRTUAL_TEMPERATURE_FACTOR · qv[k])
#     dz[k]  = R · T_v[k] / g · delp[k] / p_ctr[k]
# ---------------------------------------------------------------------------

# Convert specific humidity to mixing ratio for the virtual-T factor. We use
# the standard coefficient `VIRTUAL_TEMPERATURE_FACTOR` = 0.61
# (`(1 - epsilon) / epsilon ≈ 0.61` with
# epsilon = R_d / R_v); qv is clamped to non-negative to absorb tiny
# post-regrid negative values.
#
# Note: this helper is intentionally LESS defensive than the Holtslag-Boville
# closure's `_virtual_temperature` in
# `LocalHoltslagBovilleKzField.jl`, which additionally
# clamps T to ≥ 180 K. The HB clamp guards against pathological Kz when the
# diagnosed mixing length collapses for very cold T (its closure divides by
# theta_mid). Here we only need a geometric `dz`, so a physically realistic
# stratospheric T < 180 K is fine: `dz` just gets a little smaller, which
# is the correct hydrostatic answer at that altitude.
@inline _virtual_T_factor(qv::T) where {T<:Real} =
    one(T) + T(VIRTUAL_TEMPERATURE_FACTOR) * max(qv, zero(T))

@kernel function _dz_hydrostatic_virtualT_3d_kernel!(dz, @Const(t_lyr),
                                                     @Const(qv_lyr),
                                                     @Const(ps),
                                                     @Const(ak_ifc), @Const(bk_ifc),
                                                     R, g, Nz::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ak_lo = ak_ifc[k]
        ak_hi = ak_ifc[k + 1]
        bk_lo = bk_ifc[k]
        bk_hi = bk_ifc[k + 1]
        delp  = (ak_hi - ak_lo) + (bk_hi - bk_lo) * ps[i, j]
        p_ctr = (ak_lo + ak_hi + (bk_lo + bk_hi) * ps[i, j]) * (one(eltype(dz)) / 2)
        Tv    = t_lyr[i, j, k] * _virtual_T_factor(qv_lyr[i, j, k])
        dz[i, j, k] = R * Tv / g * delp / p_ctr
    end
end

"""
    fill_dz_hydrostatic_virtualT_staged!(dz, T, qv, ps, coefficients; sync, R, gravity)

Fill hydrostatic layer thickness from staged hybrid coefficients and each
layer's virtual temperature `T * (1 + VIRTUAL_TEMPERATURE_FACTOR * max(qv,0))`.
Supports structured arrays and six-panel tuples. `sync` explicitly controls
the completion barrier; tuples validate every panel before launching.
"""
function fill_dz_hydrostatic_virtualT_staged!(
        dz::AbstractArray{<:AbstractFloat, 3},
        t_lyr::AbstractArray{<:AbstractFloat, 3},
        qv_lyr::AbstractArray{<:AbstractFloat, 3},
        ps::AbstractArray{<:AbstractFloat, 2},
        coefficients::HybridCoefficientWorkspace;
        R::Real = R_DRY_AIR,
        gravity::Real = STANDARD_GRAVITY,
        sync::Bool)
    _validate_virtualT_panel(dz, t_lyr, qv_lyr, ps, length(coefficients.A), length(coefficients.B))
    Nx, Ny, Nz = size(dz)
    FT = eltype(dz)
    backend = get_backend(dz)
    launch!(_dz_hydrostatic_virtualT_3d_kernel!, backend, (8, 8, 1), (Nx, Ny, Nz), dz,
            t_lyr, qv_lyr, ps, coefficients.A, coefficients.B,
            FT(R), FT(gravity), Nz; sync)
    return dz
end

function fill_dz_hydrostatic_virtualT_staged!(
        dz_panels::NTuple{6}, t_panels::NTuple{6}, qv_panels::NTuple{6},
        ps_panels::NTuple{6}, coefficients::HybridCoefficientWorkspace;
        sync::Bool, kwargs...)
    _validate_virtualT_panels(dz_panels, t_panels, qv_panels, ps_panels,
                              length(coefficients.A), length(coefficients.B))
    @inbounds for p in 1:6
        fill_dz_hydrostatic_virtualT_staged!(dz_panels[p], t_panels[p], qv_panels[p],
                                              ps_panels[p], coefficients;
                                              kwargs..., sync = false)
    end
    sync && synchronize(get_backend(dz_panels[1]))
    return dz_panels
end

@kernel function _dz_hydrostatic_constT_face_kernel!(dz, @Const(ps),
                                                     @Const(ak_ifc), @Const(bk_ifc),
                                                     T_ref, R, g, Nz::Int)
    c, k = @index(Global, NTuple)
    @inbounds begin
        ak_lo = ak_ifc[k]
        ak_hi = ak_ifc[k + 1]
        bk_lo = bk_ifc[k]
        bk_hi = bk_ifc[k + 1]
        delp  = (ak_hi - ak_lo) + (bk_hi - bk_lo) * ps[c]
        p_ctr = (ak_lo + ak_hi + (bk_lo + bk_hi) * ps[c]) * (one(eltype(dz)) / 2)
        dz[c, k] = R * T_ref / g * delp / p_ctr
    end
end

function fill_dz_hydrostatic_constT_staged!(dz::AbstractArray{<:AbstractFloat, 2},
                                              ps::AbstractArray{<:AbstractFloat, 1},
                                              coefficients::HybridCoefficientWorkspace;
                                              T_ref::Real = HYDROSTATIC_REFERENCE_TEMPERATURE,
                                              R::Real = R_DRY_AIR,
                                              gravity::Real = STANDARD_GRAVITY,
                                              sync::Bool)
    ncells, Nz = size(dz)
    length(ps) == ncells || throw(DimensionMismatch("ps cell count mismatch"))
    length(coefficients.A) == Nz + 1 && length(coefficients.B) == Nz + 1 ||
        throw(DimensionMismatch("dz levels must match hybrid coefficients"))
    FT = eltype(dz)
    backend = get_backend(dz)
    launch!(_dz_hydrostatic_constT_face_kernel!, backend, (256, 1), (ncells, Nz), dz, ps,
            coefficients.A, coefficients.B, FT(T_ref), FT(R), FT(gravity), Nz; sync)
    return dz
end
