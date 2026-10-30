# ---------------------------------------------------------------------------
# Multi-tracer kernel shells for structured grids
#
# Fuses the N-tracer loop INTO the GPU kernel, reducing kernel launches
# from 6N to 6 per Strang split (for N tracers).
#
# Performance model
# -----------------
# Separate Nt=1 sweeps would make 6N launches, each reading mass + flux
#   arrays redundantly.
# Multi-tracer kernels: 6 launches, mass update computed ONCE, mass/flux
#   arrays read ONCE per cell, tracer flux computed per-tracer.
#
# For N=50 tracers: 50x fewer kernel launches, 50x fewer mass reads.
#
# Design
# ------
# A lightweight `TracerView` wraps the 4D tracer array and presents it
# as a 3D array with a fixed tracer index.  This lets the multi-tracer
# kernels reuse the EXISTING `_xface_tracer_flux` / `_yface_tracer_flux` /
# `_zface_tracer_flux` functions unchanged — Julia specializes at compile
# time and inlines the TracerView access.
#
# TracerView is a trivial isbits struct (array reference + Int32 index)
# that lives entirely in registers on GPU — zero allocation overhead.
#
# References
# ----------
# - Oceananigans.jl: multi-tracer advection fusion pattern
# - KernelAbstractions.jl: isbits kernel arguments
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const

# =========================================================================
# TracerView — 3D slice adapter for 4D tracer arrays
# =========================================================================

"""
    TracerView{FT, A}

Lightweight wrapper around a 4D array `(Nx, Ny, Nz, Nt)` that presents
it as a 3D array `(Nx, Ny, Nz)` for a fixed tracer index `t`.

This enables the multi-tracer kernels to call the existing `@inline`
face flux functions (which expect 3D `rm` arrays) without any code
duplication.  Julia's compiler inlines the `getindex` call and
eliminates the wrapper entirely — zero overhead on both CPU and GPU.

# Example
```julia
rm_4d = zeros(Float64, 36, 18, 4, 50)  # Nx, Ny, Nz, Nt
rm_t = TracerView(rm_4d, Int32(3))       # view of tracer 3
rm_t[10, 5, 2]  # equivalent to rm_4d[10, 5, 2, 3]
```
"""
struct TracerView{FT, A <: AbstractArray{FT, 4}}
    data::A
    t::Int32
end

Base.@propagate_inbounds Base.getindex(v::TracerView, i, j, k) = v.data[i, j, k, v.t]
@inline Base.eltype(::TracerView{FT}) where FT = FT
@inline Base.eltype(::Type{TracerView{FT, A}}) where {FT, A} = FT

# =========================================================================
# Multi-tracer kernel shells
# =========================================================================
#
# Each kernel:
# 1. Computes the mass update ONCE (shared across all tracers)
# 2. Iterates over tracers, computing face fluxes via the existing
#    @inline functions (dispatched on scheme type via TracerView)
# 3. Updates each tracer's mass in the 4D output array
#
# The mass update is a telescoping sum that conserves total air mass
# to roundoff: summing each face divergence cancels interior faces;
# periodic X faces match and Y/Z boundary fluxes are zero. The tracer mass
# update conserves total tracer mass for each tracer independently.

"""
    _xsweep_mt_kernel!(rm_new_4d, rm_4d, m_new, m, am, scheme, Nx, Nt)

Multi-tracer x-sweep kernel.  Processes all `Nt` tracers in a single
kernel launch, sharing the mass update and stencil index computation.

Periodic boundary conditions in x.
"""
@kernel function _xsweep_mt_kernel!(rm_new_4d, @Const(rm_4d), m_new, @Const(m),
                                     @Const(am), scheme, Nx, Nt, flux_scale)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        am_l = flux_scale * am[i, j, k]
        am_r = flux_scale * am[i + 1, j, k]
        m_new[i, j, k] = m[i, j, k] + am_l - am_r
        for t in Int32(1):Int32(Nt)
            rm_t = TracerView(rm_4d, t)
            flux_L = _xface_tracer_flux(Int32(i), j, k, rm_t, m, am_l, scheme, Nx)
            flux_R = _xface_tracer_flux(Int32(i) + Int32(1), j, k, rm_t, m, am_r, scheme, Nx)
            rm_new_4d[i, j, k, t] = rm_4d[i, j, k, t] + flux_L - flux_R
        end
    end
end

"""
    _ysweep_mt_kernel!(rm_new_4d, rm_4d, m_new, m, bm, scheme, Ny, Nt)

Multi-tracer y-sweep kernel.  Closed boundaries at the poles.
"""
@kernel function _ysweep_mt_kernel!(rm_new_4d, @Const(rm_4d), m_new, @Const(m),
                                     @Const(bm), scheme, Ny, Nt, flux_scale)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        bm_s = flux_scale * bm[i, j, k]
        bm_n = flux_scale * bm[i, j + 1, k]
        m_new[i, j, k] = m[i, j, k] + bm_s - bm_n
        for t in Int32(1):Int32(Nt)
            rm_t = TracerView(rm_4d, t)
            flux_S = _yface_tracer_flux(i, Int32(j), k, rm_t, m, bm_s, scheme, Ny)
            flux_N = _yface_tracer_flux(i, Int32(j) + Int32(1), k, rm_t, m, bm_n, scheme, Ny)
            rm_new_4d[i, j, k, t] = rm_4d[i, j, k, t] + flux_S - flux_N
        end
    end
end

"""
    _zsweep_mt_kernel!(rm_new_4d, rm_4d, m_new, m, cm, scheme, Nz, Nt)

Multi-tracer z-sweep kernel.  Closed boundaries at TOA and surface.
"""
@kernel function _zsweep_mt_kernel!(rm_new_4d, @Const(rm_4d), m_new, @Const(m),
                                     @Const(cm), scheme, Nz, Nt, flux_scale)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        cm_t = flux_scale * cm[i, j, k]
        cm_b = flux_scale * cm[i, j, k + 1]
        m_new[i, j, k] = m[i, j, k] + cm_t - cm_b
        for t in Int32(1):Int32(Nt)
            rm_t = TracerView(rm_4d, t)
            flux_T = _zface_tracer_flux(i, j, Int32(k), rm_t, m, cm_t, scheme, Nz)
            flux_B = _zface_tracer_flux(i, j, Int32(k) + Int32(1), rm_t, m, cm_b, scheme, Nz)
            rm_new_4d[i, j, k, t] = rm_4d[i, j, k, t] + flux_T - flux_B
        end
    end
end

export TracerView
