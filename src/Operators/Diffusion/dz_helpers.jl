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
#     Matches the preprocessing-side `dz_hydrostatic_constT!` in
#     `src/Preprocessing/tm5_convection_conversion.jl`. Default for any
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

function _stage_hybrid_coefficients!(coefficients::HybridCoefficientWorkspace,
                                     ak_ifc::AbstractVector,
                                     bk_ifc::AbstractVector)
    length(coefficients.A) == length(ak_ifc) || throw(DimensionMismatch(
        "coefficient workspace length $(length(coefficients.A)) ≠ A length $(length(ak_ifc))"))
    length(coefficients.B) == length(bk_ifc) || throw(DimensionMismatch(
        "coefficient workspace length $(length(coefficients.B)) ≠ B length $(length(bk_ifc))"))
    copyto!(coefficients.A, ak_ifc)
    copyto!(coefficients.B, bk_ifc)
    coefficients.pure_pressure = all(iszero, bk_ifc)
    db = diff(bk_ifc)
    coefficients.positive_b_coordinate =
        all(>=(zero(eltype(db))), db) && any(>(zero(eltype(db))), db)
    return coefficients
end

function _stage_hybrid_coefficients(reference, ak_ifc::AbstractVector,
                                    bk_ifc::AbstractVector)
    FT = eltype(reference)
    coefficients = HybridCoefficientWorkspace(similar(reference, FT, length(ak_ifc)),
                                              similar(reference, FT, length(bk_ifc)))
    return _stage_hybrid_coefficients!(coefficients, ak_ifc, bk_ifc)
end

function _validate_positive_recovery_slopes(qv::Array,
                                            coefficients::HybridCoefficientWorkspace)
    Nz = size(qv, ndims(qv))
    db = diff(coefficients.B)
    @inbounds for column in CartesianIndices(size(qv)[1:end-1])
        slope = zero(promote_type(eltype(qv), eltype(coefficients.B)))
        for k in 1:Nz
            slope += (one(eltype(qv)) - qv[column, k]) * db[k]
        end
        isfinite(slope) && slope > zero(slope) || throw(ArgumentError(
            "cannot recover total surface pressure: humidity-weighted hybrid-B slope " *
            "must be finite and positive; got $slope at column $(Tuple(column))"))
    end
    return nothing
end

# Device arrays cannot be scalar-indexed by the host. Standard hybrid
# coordinates have nondecreasing B; requiring qv < 1 on their B-active layers
# proves every column's humidity-weighted slope is positive before launching.
function _validate_positive_recovery_slopes(qv,
                                            coefficients::HybridCoefficientWorkspace)
    coefficients.positive_b_coordinate ||
        throw(ArgumentError("device dry-to-total recovery requires a nondecreasing, nonconstant hybrid B coordinate"))
    qv_max = maximum(qv)
    isfinite(qv_max) && qv_max < one(qv_max) || throw(ArgumentError(
        "cannot recover total surface pressure: humidity-weighted hybrid-B slope " *
        "is not guaranteed positive because maximum qv is $qv_max"))
    return nothing
end

@kernel function _recover_total_ps_2d_kernel!(ps_total, @Const(ps_dry),
                                              @Const(qv), @Const(ak_ifc),
                                              @Const(bk_ifc), Nz::Int)
    i, j = @index(Global, NTuple)
    FT = eltype(ps_total)
    offset = zero(FT)
    slope = zero(FT)
    @inbounds for k in 1:Nz
        dry_fraction = one(FT) - qv[i, j, k]
        offset += dry_fraction * (ak_ifc[k + 1] - ak_ifc[k])
        slope += dry_fraction * (bk_ifc[k + 1] - bk_ifc[k])
    end
    @inbounds ps_total[i, j] = (ps_dry[i, j] - offset) / slope
end

@kernel function _recover_total_ps_1d_kernel!(ps_total, @Const(ps_dry),
                                              @Const(qv), @Const(ak_ifc),
                                              @Const(bk_ifc), Nz::Int)
    c = @index(Global)
    FT = eltype(ps_total)
    offset = zero(FT)
    slope = zero(FT)
    @inbounds for k in 1:Nz
        dry_fraction = one(FT) - qv[c, k]
        offset += dry_fraction * (ak_ifc[k + 1] - ak_ifc[k])
        slope += dry_fraction * (bk_ifc[k + 1] - bk_ifc[k])
    end
    @inbounds ps_total[c] = (ps_dry[c] - offset) / slope
end

"""
    recover_total_surface_pressure!(dest, ps_dry, qv, A, B)

Recover total surface pressure on the array backend used by `dest`. For the
hybrid interfaces `p_{k+1/2}=A_{k+1/2}+B_{k+1/2}p_{s,total}`, stored dry
surface pressure satisfies

```math
p_{s,dry}=\\sum_k(1-q_{v,k})
  [\\Delta A_k+\\Delta B_k p_{s,total}],\\qquad
p_{s,total}=\\frac{p_{s,dry}-\\sum_k(1-q_{v,k})\\Delta A_k}
                  {\\sum_k(1-q_{v,k})\\Delta B_k}.
```

The implementation uses KernelAbstractions for CPU and the package's supported
CUDA and Metal backends; device arrays are never scalar-indexed by the host.
For an A-only coordinate (`B ≡ 0`) recovery is unobservable and unnecessary,
so `ps_dry` is copied unchanged. Otherwise the humidity-weighted B slope must
be finite and positive or the function throws `ArgumentError`.
"""
function recover_total_surface_pressure!(dest::AbstractMatrix, ps_dry::AbstractMatrix,
                                         qv::AbstractArray{<:AbstractFloat, 3},
                                         ak_ifc::AbstractVector, bk_ifc::AbstractVector)
    Nx, Ny = size(dest)
    size(ps_dry) == (Nx, Ny) || throw(DimensionMismatch("dry ps shape mismatch"))
    size(qv)[1:2] == (Nx, Ny) || throw(DimensionMismatch("humidity horizontal shape mismatch"))
    Nz = size(qv, 3)
    length(ak_ifc) == Nz + 1 && length(bk_ifc) == Nz + 1 ||
        throw(DimensionMismatch("humidity levels must match hybrid coefficients"))
    coefficients = _stage_hybrid_coefficients(dest, ak_ifc, bk_ifc)
    return _recover_total_surface_pressure_staged!(dest, ps_dry, qv,
                                                    coefficients; sync = true)
end

function recover_total_surface_pressure!(dest::AbstractVector, ps_dry::AbstractVector,
                                         qv::AbstractMatrix,
                                         ak_ifc::AbstractVector, bk_ifc::AbstractVector)
    ncells = length(dest)
    length(ps_dry) == ncells || throw(DimensionMismatch("dry ps length mismatch"))
    size(qv, 1) == ncells || throw(DimensionMismatch("humidity cell count mismatch"))
    Nz = size(qv, 2)
    length(ak_ifc) == Nz + 1 && length(bk_ifc) == Nz + 1 ||
        throw(DimensionMismatch("humidity levels must match hybrid coefficients"))
    coefficients = _stage_hybrid_coefficients(dest, ak_ifc, bk_ifc)
    return _recover_total_surface_pressure_staged!(dest, ps_dry, qv,
                                                    coefficients; sync = true)
end

function recover_total_surface_pressure!(dest::NTuple{6}, ps_dry::NTuple{6},
                                         qv::NTuple{6}, ak_ifc, bk_ifc)
    coefficients = _stage_hybrid_coefficients(dest[1], ak_ifc, bk_ifc)
    return _recover_total_surface_pressure_staged!(dest, ps_dry, qv,
                                                    coefficients; sync = true)
end

function _recover_total_surface_pressure_staged!(dest::AbstractMatrix,
                                                  ps_dry::AbstractMatrix,
                                                  qv::AbstractArray{<:AbstractFloat, 3},
                                                  coefficients::HybridCoefficientWorkspace;
                                                  sync::Bool)
    coefficients.pure_pressure && return copyto!(dest, ps_dry)
    _validate_positive_recovery_slopes(qv, coefficients)
    Nx, Ny = size(dest)
    Nz = size(qv, 3)
    backend = get_backend(dest)
    launch!(_recover_total_ps_2d_kernel!, backend, (8, 8), (Nx, Ny),
            dest, ps_dry, qv, coefficients.A, coefficients.B, Nz; sync)
    return dest
end

function _recover_total_surface_pressure_staged!(dest::AbstractVector,
                                                  ps_dry::AbstractVector,
                                                  qv::AbstractMatrix,
                                                  coefficients::HybridCoefficientWorkspace;
                                                  sync::Bool)
    coefficients.pure_pressure && return copyto!(dest, ps_dry)
    _validate_positive_recovery_slopes(qv, coefficients)
    ncells, Nz = size(qv)
    backend = get_backend(dest)
    launch!(_recover_total_ps_1d_kernel!, backend, 256, ncells,
            dest, ps_dry, qv, coefficients.A, coefficients.B, Nz; sync)
    return dest
end


function _recover_total_surface_pressure_staged!(dest::NTuple{6},
                                                  ps_dry::NTuple{6},
                                                  qv::NTuple{6},
                                                  coefficients::HybridCoefficientWorkspace;
                                                  sync::Bool)
    if coefficients.pure_pressure
        @inbounds for p in 1:6
            copyto!(dest[p], ps_dry[p])
        end
        return dest
    end
    @inbounds for p in 1:6
        _recover_total_surface_pressure_staged!(dest[p], ps_dry[p], qv[p],
                                                coefficients; sync = false)
    end
    sync && synchronize(get_backend(dest[1]))
    return dest
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
    fill_dz_hydrostatic_constT!(dz, ps, ak_ifc, bk_ifc;
                                 T_ref = HYDROSTATIC_REFERENCE_TEMPERATURE,
                                 R = R_DRY_AIR, gravity = STANDARD_GRAVITY)

Populate a 3D `(Nx, Ny, Nz)` `dz` array (host or device) from surface
pressure `ps` and hybrid sigma-pressure interface coefficients
`ak_ifc, bk_ifc` (length `Nz + 1`). Backend follows `dz`.
"""
function fill_dz_hydrostatic_constT!(dz::AbstractArray{<:AbstractFloat, 3},
                                      ps::AbstractArray{<:AbstractFloat, 2},
                                      ak_ifc::AbstractVector,
                                      bk_ifc::AbstractVector;
                                      T_ref::Real    = HYDROSTATIC_REFERENCE_TEMPERATURE,
                                      R::Real        = R_DRY_AIR,
                                      gravity::Real  = STANDARD_GRAVITY)
    Nx, Ny, Nz = size(dz)
    size(ps) == (Nx, Ny) || throw(DimensionMismatch(
        "ps shape $(size(ps)) ≠ ($Nx, $Ny) expected from dz"))
    length(ak_ifc) == Nz + 1 || throw(DimensionMismatch(
        "ak_ifc length $(length(ak_ifc)) ≠ Nz+1=$(Nz+1)"))
    length(bk_ifc) == Nz + 1 || throw(DimensionMismatch(
        "bk_ifc length $(length(bk_ifc)) ≠ Nz+1=$(Nz+1)"))
    coefficients = _stage_hybrid_coefficients(dz, ak_ifc, bk_ifc)
    return _fill_dz_hydrostatic_constT_staged!(dz, ps, coefficients;
                                               T_ref, R, gravity, sync = true)
end

function _fill_dz_hydrostatic_constT_staged!(dz::AbstractArray{<:AbstractFloat, 3},
                                              ps::AbstractArray{<:AbstractFloat, 2},
                                              coefficients::HybridCoefficientWorkspace;
                                              T_ref::Real = HYDROSTATIC_REFERENCE_TEMPERATURE,
                                              R::Real = R_DRY_AIR,
                                              gravity::Real = STANDARD_GRAVITY,
                                              sync::Bool)
    Nx, Ny, Nz = size(dz)
    FT = eltype(dz)
    backend = get_backend(dz)
    launch!(_dz_hydrostatic_constT_kernel!, backend, (8, 8, 1), (Nx, Ny, Nz), dz, ps,
            coefficients.A, coefficients.B, FT(T_ref), FT(R), FT(gravity), Nz; sync)
    return dz
end

"""
    fill_dz_hydrostatic_constT!(dz_panels::NTuple{6}, ps_panels::NTuple{6},
                                 ak_ifc, bk_ifc; ...)

Cubed-sphere variant: per-panel 3D `(Nc, Nc, Nz)` `dz` arrays are filled
from per-panel `ps` arrays of shape `(Nc, Nc)` (interior only — the
runtime stores `surface_pressure` without the advection halo).
"""
function fill_dz_hydrostatic_constT!(dz_panels::NTuple{6, <:AbstractArray{<:AbstractFloat, 3}},
                                      ps_panels::NTuple{6, <:AbstractArray{<:AbstractFloat, 2}},
                                      ak_ifc::AbstractVector,
                                      bk_ifc::AbstractVector;
                                      kwargs...)
    coefficients = _stage_hybrid_coefficients(dz_panels[1], ak_ifc, bk_ifc)
    _fill_dz_hydrostatic_constT_staged!(dz_panels, ps_panels, coefficients;
                                        kwargs..., sync = true)
    return dz_panels
end

function _fill_dz_hydrostatic_constT_staged!(dz_panels::NTuple{6},
                                              ps_panels::NTuple{6},
                                              coefficients::HybridCoefficientWorkspace;
                                              sync::Bool, kwargs...)
    @inbounds for p in 1:6
        _fill_dz_hydrostatic_constT_staged!(dz_panels[p], ps_panels[p], coefficients;
                                            kwargs..., sync = false)
    end
    sync && synchronize(get_backend(dz_panels[1]))
    return dz_panels
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
# `src/State/Fields/LocalHoltslagBovilleKzField.jl:83`, which additionally
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
    fill_dz_hydrostatic_virtualT!(dz, t_lyr, qv_lyr, ps, ak_ifc, bk_ifc;
                                   R = R_DRY_AIR, gravity = STANDARD_GRAVITY)

Populate a 3D `(Nx, Ny, Nz)` `dz` array using virtual temperature per
layer: `T_v = T · (1 + VIRTUAL_TEMPERATURE_FACTOR · qv)` (factor 0.61).
Matches the geometry the
[`LocalHoltslagBovilleKzField`](@ref AtmosTransport.State.Fields.LocalHoltslagBovilleKzField)
uses for its column-mid heights, closing the inconsistency between
solver `dz` and Kz-cache `dz`.

`t_lyr`, `qv_lyr` are layer-center 3D fields with shape `(Nx, Ny, Nz)`,
typically pulled from the active window's VDIFF payload.
"""
function fill_dz_hydrostatic_virtualT!(dz::AbstractArray{<:AbstractFloat, 3},
                                        t_lyr::AbstractArray{<:AbstractFloat, 3},
                                        qv_lyr::AbstractArray{<:AbstractFloat, 3},
                                        ps::AbstractArray{<:AbstractFloat, 2},
                                        ak_ifc::AbstractVector,
                                        bk_ifc::AbstractVector;
                                        R::Real       = R_DRY_AIR,
                                        gravity::Real = STANDARD_GRAVITY)
    Nx, Ny, Nz = size(dz)
    size(t_lyr)  == (Nx, Ny, Nz) || throw(DimensionMismatch(
        "t_lyr shape $(size(t_lyr)) ≠ ($Nx, $Ny, $Nz) expected from dz"))
    size(qv_lyr) == (Nx, Ny, Nz) || throw(DimensionMismatch(
        "qv_lyr shape $(size(qv_lyr)) ≠ ($Nx, $Ny, $Nz) expected from dz"))
    size(ps)     == (Nx, Ny)     || throw(DimensionMismatch(
        "ps shape $(size(ps)) ≠ ($Nx, $Ny) expected from dz"))
    length(ak_ifc) == Nz + 1 || throw(DimensionMismatch(
        "ak_ifc length $(length(ak_ifc)) ≠ Nz+1=$(Nz + 1)"))
    length(bk_ifc) == Nz + 1 || throw(DimensionMismatch(
        "bk_ifc length $(length(bk_ifc)) ≠ Nz+1=$(Nz + 1)"))
    coefficients = _stage_hybrid_coefficients(dz, ak_ifc, bk_ifc)
    return _fill_dz_hydrostatic_virtualT_staged!(dz, t_lyr, qv_lyr, ps,
                                                  coefficients;
                                                  R, gravity, sync = true)
end

function _fill_dz_hydrostatic_virtualT_staged!(
        dz::AbstractArray{<:AbstractFloat, 3},
        t_lyr::AbstractArray{<:AbstractFloat, 3},
        qv_lyr::AbstractArray{<:AbstractFloat, 3},
        ps::AbstractArray{<:AbstractFloat, 2},
        coefficients::HybridCoefficientWorkspace;
        R::Real = R_DRY_AIR,
        gravity::Real = STANDARD_GRAVITY,
        sync::Bool)
    Nx, Ny, Nz = size(dz)
    FT = eltype(dz)
    backend = get_backend(dz)
    launch!(_dz_hydrostatic_virtualT_3d_kernel!, backend, (8, 8, 1), (Nx, Ny, Nz), dz,
            t_lyr, qv_lyr, ps, coefficients.A, coefficients.B,
            FT(R), FT(gravity), Nz; sync)
    return dz
end

"""
    fill_dz_hydrostatic_virtualT!(dz_panels::NTuple{6}, t_panels::NTuple{6},
                                   qv_panels::NTuple{6}, ps_panels::NTuple{6},
                                   ak_ifc, bk_ifc; ...)

Cubed-sphere variant of [`fill_dz_hydrostatic_virtualT!`](@ref). Per-panel
3D `(Nc, Nc, Nz)` `dz`/`t_lyr`/`qv_lyr` arrays + per-panel `(Nc, Nc)` `ps`.
The VDIFF payload is panel-native, so passing the panel tuples directly is
the natural runtime shape.
"""
function fill_dz_hydrostatic_virtualT!(
        dz_panels::NTuple{6, <:AbstractArray{<:AbstractFloat, 3}},
        t_panels::NTuple{6, <:AbstractArray{<:AbstractFloat, 3}},
        qv_panels::NTuple{6, <:AbstractArray{<:AbstractFloat, 3}},
        ps_panels::NTuple{6, <:AbstractArray{<:AbstractFloat, 2}},
        ak_ifc::AbstractVector,
        bk_ifc::AbstractVector;
        kwargs...)
    coefficients = _stage_hybrid_coefficients(dz_panels[1], ak_ifc, bk_ifc)
    _fill_dz_hydrostatic_virtualT_staged!(dz_panels, t_panels, qv_panels,
                                          ps_panels, coefficients;
                                          kwargs..., sync = true)
    return dz_panels
end


function _fill_dz_hydrostatic_virtualT_staged!(
        dz_panels::NTuple{6}, t_panels::NTuple{6}, qv_panels::NTuple{6},
        ps_panels::NTuple{6}, coefficients::HybridCoefficientWorkspace;
        sync::Bool, kwargs...)
    @inbounds for p in 1:6
        _fill_dz_hydrostatic_virtualT_staged!(dz_panels[p], t_panels[p], qv_panels[p],
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

"""
    fill_dz_hydrostatic_constT!(dz::AbstractArray{<:Any, 2},
                                 ps::AbstractArray{<:Any, 1},
                                 ak_ifc, bk_ifc; ...)

Face-indexed variant for ReducedGaussian topology: `dz` shape `(ncells, Nz)`,
`ps` shape `(ncells,)`. Same constant-`T_ref` formula as the structured/CS
overloads, just unrolled over the face-indexed cell axis.
"""
function fill_dz_hydrostatic_constT!(dz::AbstractArray{<:AbstractFloat, 2},
                                      ps::AbstractArray{<:AbstractFloat, 1},
                                      ak_ifc::AbstractVector,
                                      bk_ifc::AbstractVector;
                                      T_ref::Real    = HYDROSTATIC_REFERENCE_TEMPERATURE,
                                      R::Real        = R_DRY_AIR,
                                      gravity::Real  = STANDARD_GRAVITY)
    ncells, Nz = size(dz)
    length(ps) == ncells || throw(DimensionMismatch(
        "ps length $(length(ps)) ≠ ncells=$(ncells) expected from dz"))
    length(ak_ifc) == Nz + 1 || throw(DimensionMismatch(
        "ak_ifc length $(length(ak_ifc)) ≠ Nz+1=$(Nz + 1)"))
    length(bk_ifc) == Nz + 1 || throw(DimensionMismatch(
        "bk_ifc length $(length(bk_ifc)) ≠ Nz+1=$(Nz + 1)"))
    coefficients = _stage_hybrid_coefficients(dz, ak_ifc, bk_ifc)
    return _fill_dz_hydrostatic_constT_staged!(dz, ps, coefficients;
                                               T_ref, R, gravity, sync = true)
end

function _fill_dz_hydrostatic_constT_staged!(dz::AbstractArray{<:AbstractFloat, 2},
                                              ps::AbstractArray{<:AbstractFloat, 1},
                                              coefficients::HybridCoefficientWorkspace;
                                              T_ref::Real = HYDROSTATIC_REFERENCE_TEMPERATURE,
                                              R::Real = R_DRY_AIR,
                                              gravity::Real = STANDARD_GRAVITY,
                                              sync::Bool)
    ncells, Nz = size(dz)
    FT = eltype(dz)
    backend = get_backend(dz)
    launch!(_dz_hydrostatic_constT_face_kernel!, backend, (256, 1), (ncells, Nz), dz, ps,
            coefficients.A, coefficients.B, FT(T_ref), FT(R), FT(gravity), Nz; sync)
    return dz
end
