# ---------------------------------------------------------------------------
# Vertical closure — diagnose cm from horizontal convergence
#
# The vertical mass flux cm is diagnosed from the horizontal fluxes am/bm
# via the TM5 dynam0 B-coefficient splitting convention.  This is the
# discrete continuity equation:
#
#   cm[k+1] = cm[k] + (conv_h[k] - Δb[k] × pit)
#
# where:
#   conv_h[k] = am[i,j,k] - am[i+1,j,k] + bm[i,j,k] - bm[i,j+1,k]
#   pit       = Σ_k conv_h[k]   (total column horizontal convergence)
#   Δb[k]     = B[k+1] - B[k]   (hybrid B-coefficient increment)
#
# Boundary conditions: cm[1] = 0 (TOA), cm[Nz+1] ≈ 0 (surface, by design).
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, get_backend, synchronize

"""
    diagnose_cm_from_continuity!(cm, am, bm, Δb, Nx, Ny, Nz)

Diagnose vertical mass flux from horizontal convergence (CPU column loop).

# Arguments
- `cm  :: AbstractArray{FT,3}` — output vertical flux, size `(Nx, Ny, Nz+1)`
- `am  :: AbstractArray{FT,3}` — x-face flux, size `(Nx+1, Ny, Nz)`
- `bm  :: AbstractArray{FT,3}` — y-face flux, size `(Nx, Ny+1, Nz)`
- `Δb  :: AbstractVector{FT}`  — B-coefficient increments, length `Nz`
- `Nx, Ny, Nz :: Int`          — grid dimensions
"""
function diagnose_cm_from_continuity!(cm::AbstractArray{FT, 3},
                                     am::AbstractArray{FT, 3},
                                     bm::AbstractArray{FT, 3},
                                     Δb::AbstractVector{FT},
                                     Nx::Int, Ny::Int, Nz::Int) where FT
    @inbounds for j in 1:Ny, i in 1:Nx
        pit = zero(FT)
        pit_comp = zero(FT)
        for k in 1:Nz
            conv_k = am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
            pit, pit_comp = _kahan_add(pit, pit_comp, conv_k)
        end
        acc = zero(FT)
        acc_comp = zero(FT)
        cm[i, j, 1] = acc
        for k in 1:Nz
            conv_k = am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
            acc, acc_comp = _kahan_add(acc, acc_comp, conv_k - Δb[k] * pit)
            cm[i, j, k+1] = acc
        end
    end
    return nothing
end

"""
    diagnose_cm_from_continuity_vc!(cm, am, bm, vc, Nx, Ny, Nz)

Convenience wrapper that extracts Δb from a `HybridSigmaPressure` vertical
coordinate before calling the core routine.

!!! warning "Deprecated"
    No part of AtmosTransport calls it; it will be removed in the next minor
    release. Equivalent: `diagnose_cm_from_continuity!(cm, am, bm, Δb, Nx, Ny, Nz)` with
    `Δb = eltype(cm)[AtmosTransport.Grids.b_diff(vc, k) for k in 1:Nz]`.
"""
function diagnose_cm_from_continuity_vc!(cm, am, bm, vc, Nx, Ny, Nz)
    Base.depwarn("`diagnose_cm_from_continuity_vc!` is deprecated and will be removed in the " *
                 "next minor release; call `diagnose_cm_from_continuity!(cm, am, bm, Δb, Nx, Ny, Nz)` " *
                 "with `Δb = eltype(cm)[AtmosTransport.Grids.b_diff(vc, k) for k in 1:Nz]`.",
                 :diagnose_cm_from_continuity_vc!)
    FT = eltype(cm)
    Δb = FT[b_diff(vc, k) for k in 1:Nz]
    diagnose_cm_from_continuity!(cm, am, bm, Δb, Nx, Ny, Nz)
end

# ---------------------------------------------------------------------------
# KernelAbstractions version (GPU-compatible)
#
# Each work item handles one column (i, j). The vertical loop is serial
# within the column because cm[k+1] depends on cm[k] (sequential prefix sum).
# ---------------------------------------------------------------------------

@kernel function _cm_continuity_kernel!(cm, @Const(am), @Const(bm),
                                        @Const(Δb), Nz_val::Int32)
    i, j = @index(Global, NTuple)
    FT = eltype(cm)

    pit = zero(FT)
    pit_comp = zero(FT)
    @inbounds for k in 1:Nz_val
        conv_k = am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
        pit, pit_comp = _kahan_add(pit, pit_comp, conv_k)
    end

    acc = zero(FT)
    acc_comp = zero(FT)
    @inbounds cm[i, j, 1] = acc
    @inbounds for k in 1:Nz_val
        conv_k = am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
        acc, acc_comp = _kahan_add(acc, acc_comp, conv_k - Δb[k] * pit)
        cm[i, j, k+1] = acc
    end
end

"""
    diagnose_cm_from_continuity_ka!(cm, am, bm, Δb, Nx, Ny, Nz)

GPU-compatible version using KernelAbstractions.  One thread per column.

!!! warning "Deprecated"
    No part of AtmosTransport calls it; it will be removed in the next minor
    release.
"""
function diagnose_cm_from_continuity_ka!(cm::AbstractArray{FT, 3},
                                         am::AbstractArray{FT, 3},
                                         bm::AbstractArray{FT, 3},
                                         Δb::AbstractVector{FT},
                                         Nx::Int, Ny::Int, Nz::Int) where FT
    Base.depwarn("`diagnose_cm_from_continuity_ka!` is deprecated and will be removed in the " *
                 "next minor release.", :diagnose_cm_from_continuity_ka!)
    backend = get_backend(cm)
    kernel! = _cm_continuity_kernel!(backend, 256)
    kernel!(cm, am, bm, Δb, Int32(Nz); ndrange=(Nx, Ny))
    synchronize(backend)
    return nothing
end

export diagnose_cm_from_continuity!
export diagnose_cm_from_continuity_vc!
export diagnose_cm_from_continuity_ka!
