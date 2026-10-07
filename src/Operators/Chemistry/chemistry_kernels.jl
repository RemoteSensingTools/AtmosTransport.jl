# ---------------------------------------------------------------------------
# Multi-tracer fused decay kernel
#
# One KernelAbstractions launch per `apply!(::ExponentialDecay)` call. The
# kernel iterates over every spatial cell (Cartesian index over all axes
# except the trailing tracer axis) and, for each of the operator's
# `Nt_op ≤ ntracers(state)` selected tracers, multiplies `tracers_raw[I,
# t_idx]` by `exp(-rate * dt)`.
#
# Design notes
# ------------
# - `indices::NTuple{Nt_op, Int32}` is resolved on the host in `apply!` via
#   `tracer_index(state, op.tracer_names[n])`, so the kernel does no symbol
#   lookups.
# - Operating on `tracers_raw` directly (the packed 4D/3D buffer from
#   CellState) means one kernel launch handles every selected
#   tracer without the Julia-level per-tracer loop that the earlier
#   `apply_chemistry!` used.
# - The kernel is rank-agnostic: for structured `(Nx, Ny, Nz, Nt)` it
#   launches `ndrange = (Nx, Ny, Nz)`; for face-indexed `(ncells, Nz, Nt)`
#   it launches `ndrange = (ncells, Nz)`. The trailing tracer axis is
#   indexed with a scalar `t_idx`, which combined with the Cartesian
#   spatial index yields the correct N-dim access.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const

"""
    _exp_decay_kernel!(tracers_raw, indices, decrements, Nt_op)

Apply exponential decay in place to a packed tracer buffer as
`c += c · d` with the precomputed decrement `d = expm1(-rate · dt)`
(see [`decay_decrement`](@ref)). Allocation-free; only `tracers_raw` is written.
"""
@kernel function _exp_decay_kernel!(tracers_raw,
                                     @Const(indices),
                                     @Const(decrements),
                                     Nt_op)
    I = @index(Global, Cartesian)
    @inbounds for n in Int32(1):Nt_op
        t_idx = indices[n]
        c = tracers_raw[I, t_idx]
        tracers_raw[I, t_idx] = muladd(c, decrements[n], c)
    end
end
