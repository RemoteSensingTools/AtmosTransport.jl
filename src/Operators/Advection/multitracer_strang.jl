# Multi-tracer directional sweeps and the multi-tracer Strang palindrome.
# Split from StrangSplitting.jl (refactor phase 4); included by Advection.jl in this order.

# =========================================================================
# Multi-tracer directional sweeps — generated via @eval
# =========================================================================
#
# These operate on 4D tracer arrays (Nx, Ny, Nz, Nt) and use the
# multi-tracer kernel shells from multitracer_kernels.jl.  The mass
# update is computed ONCE per cell, shared across all tracers.

for (sweep_fn, kernel_fn, dim) in (
    (:sweep_x_mt!, :_xsweep_mt_kernel!, 1),
    (:sweep_y_mt!, :_ysweep_mt_kernel!, 2),
    (:sweep_z_mt!, :_zsweep_mt_kernel!, 3),
)
    @eval begin
        # Ping-pong entry point: writes (rm4d_out, m_out), reads (rm4d_in, m_in).
        function $sweep_fn(rm4d_in::AbstractArray{FT,4},  rm4d_out::AbstractArray{FT,4},
                           m_in::AbstractArray{FT,3},     m_out::AbstractArray{FT,3},
                           flux::AbstractArray{FT,3},
                           scheme::AbstractAdvectionScheme,
                           ws::AdvectionWorkspace{FT},
                           flux_scale::FT = one(FT)) where FT
            backend = get_backend(m_in)
            Nt = Int32(size(rm4d_in, 4))
            kernel! = $kernel_fn(backend, 256)
            kernel!(rm4d_out, rm4d_in, m_out, m_in, flux, scheme,
                    Int32(size(m_in, $dim)), Nt, flux_scale;
                    ndrange=size(m_in))
            synchronize(backend)
            return nothing
        end

        """
            $($sweep_fn)(rm_4d, m, flux, scheme, ws[, flux_scale])

        In-place multi-tracer sweep. Writes into workspace B buffers, then
        copies back. Palindrome orchestration should use the 7- or 8-argument
        ping-pong form to avoid the copies.
        """
        function $sweep_fn(rm_4d::AbstractArray{FT,4}, m::AbstractArray{FT,3},
                           flux::AbstractArray{FT,3},
                           scheme::AbstractAdvectionScheme,
                           ws::AdvectionWorkspace{FT},
                           flux_scale::FT = one(FT)) where FT
            $sweep_fn(rm_4d, ws.rm_4d_B, m, ws.m_B, flux, scheme, ws, flux_scale)
            copyto!(rm_4d, ws.rm_4d_B)
            copyto!(m,     ws.m_B)
            return nothing
        end
    end
end

# =========================================================================
# Multi-tracer Strang splitting: (X → Y → Z)^n → midpoint → (Z → Y → X)^n
# =========================================================================

"""
    strang_split_mt!(rm_4d, m, am, bm, cm, scheme, ws)

Multi-tracer Strang-split advection on a packed 4D tracer array.

This is the performance-optimized path: all `Nt = size(rm_4d, 4)` tracers
are processed in a SINGLE kernel launch per sweep direction (6 total),
rather than `6 × Nt` launches in the per-tracer path.

The mass update (m_new = m + flux_in - flux_out) is computed ONCE per
cell per sweep, shared across all tracers.

## Subcycling

One count `n` comes from the palindrome outflow budget
`2 (out_x + out_y + out_z) / (n min(m, m_end)) ≤ cfl_limit` (`m_end` the
cell's mass after the step) that the cubed sphere uses
(`_structured_palindrome_subcycle_count`, default `cfl_limit =
DEFAULT_CFL_LIMIT`). The step then runs the symmetric composition

    (X Y Z)^n → midpoint operators → (Z Y X)^n,     every pass with flux / n,

6n sweeps, second order, and exactly `X Y Z | midpoint | Z Y X` at `n = 1`.

# Arguments
- `rm_4d::AbstractArray{FT,4}` — tracer mass `(Nx, Ny, Nz, Nt)`, mutated
- `m::AbstractArray{FT,3}` — air mass `(Nx, Ny, Nz)`, mutated
- `am, bm, cm` — mass fluxes (x, y, z directions)
- `scheme::AbstractAdvectionScheme` — advection scheme
- `ws::AdvectionWorkspace` — workspace with 4D buffer allocated
"""
function strang_split_mt!(rm_4d::AbstractArray{FT,4}, m::AbstractArray{FT,3},
                          am::AbstractArray{FT,3}, bm::AbstractArray{FT,3},
                          cm::AbstractArray{FT,3},
                          scheme::AbstractAdvectionScheme,
                          ws::AdvectionWorkspace{FT};
                          cfl_limit::Real = DEFAULT_CFL_LIMIT,
                          diffusion_op::AbstractDiffusion = NoDiffusion(),
                          diffusion_workspace = nothing,
                          emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                          tracer_names::Union{Nothing, Tuple} = nothing,
                          meteo = nothing,
                          grid = nothing,
                          dt::Union{Nothing, Real} = nothing) where FT
    _require_structured_vertical(scheme)
    _preflight_advection_workspace(ws, rm_4d, m)
    preflight_diffusion(diffusion_op, diffusion_workspace, dt,
                         m, size(rm_4d, 4))
    cfl_ft = convert(FT, cfl_limit)

    # One palindrome subcycle count for all directions (the cubed-sphere budget).
    n_sub = _structured_palindrome_subcycle_count(am, bm, cm, m, cfl_ft)
    fs = inv(FT(n_sub))

    # Ping-pong state. (rm_cur, m_cur) starts as the caller's buffers;
    # (rm_alt, m_alt) is the workspace B pair. Each kernel launch reads
    # from cur and writes to alt; we then rebind cur ↔ alt so the next
    # launch continues the chain with zero inter-sweep copies.
    rm_cur, m_cur = rm_4d,       m
    rm_alt, m_alt = ws.rm_4d_B,  ws.m_B

    # Inline one palindrome direction at a time. The helper does the
    # rebinding to avoid @eval-ing a parametric loop body.
    @inline function _pass!(sweep_fn, rm_cur_, rm_alt_, m_cur_, m_alt_, flux, fs)
        sweep_fn(rm_cur_, rm_alt_, m_cur_, m_alt_, flux, scheme, ws, fs)
        return rm_alt_, rm_cur_, m_alt_, m_cur_
    end

    # Forward half: (X → Y → Z)^n_sub
    for _ in 1:n_sub
        rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_x_mt!, rm_cur, rm_alt, m_cur, m_alt, am, fs)
        rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_y_mt!, rm_cur, rm_alt, m_cur, m_alt, bm, fs)
        rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_z_mt!, rm_cur, rm_alt, m_cur, m_alt, cm, fs)
    end

    # Palindrome center (`palindrome_center!`, StrangSplitting.jl). Without a
    # surface-flux operator it is one V(dt), a no-op for NoDiffusion, so the
    # default path does no floating-point work here. With one, the diffusion
    # operator's coupling orders the center: V(dt/2) → S(dt) → V(dt/2)
    # (the OPERATOR_COMPOSITION.md §3.2 arrangement) or S(dt) → V(dt).
    # Emissions always enter with the full dt; the symmetric operators around
    # them give the second-order accuracy. Backward Euler is not a semigroup
    # ((I-dt·D)⁻¹ ≠ [(I-dt/2·D)⁻¹]²), so adding a surface flux with the split
    # coupling changes the diffusion by O((dt·D)²). The mass-flux VMR wrapper
    # keeps the diffusion column-mass conserving; it scales the
    # `(Nx, Ny, Nz, Nt)` tracer mass by the `(Nx, Ny, Nz)` air mass. The center
    # closures capture fresh bindings: `rm_cur`/`m_cur` are reassigned by the
    # sweeps, and capturing them would box them.
    emissions_op isa NoSurfaceFlux || tracer_names !== nothing || throw(ArgumentError(
        "strang_split_mt!: `emissions_op` is non-trivial but " *
        "`tracer_names` was not supplied — the surface-flux " *
        "kernel needs per-tracer index resolution. Pass " *
        "`tracer_names = state.tracer_names`."))
    emissions_op isa NoSurfaceFlux || dt !== nothing || throw(ArgumentError(
        "strang_split_mt!: `emissions_op` is non-trivial but `dt` was not supplied"))
    rm_center, m_center = rm_cur, m_cur
    palindrome_center!(
        τ -> apply_vertical_diffusion_vmr!(rm_center, m_center, diffusion_op,
                                           diffusion_workspace, τ, meteo),
        τ -> apply_surface_flux!(rm_center, emissions_op, ws, τ, meteo, grid;
                                 tracer_names = tracer_names),
        diffusion_op, emissions_op, dt)

    # Reverse half: (Z → Y → X)^n_sub
    for _ in 1:n_sub
        rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_z_mt!, rm_cur, rm_alt, m_cur, m_alt, cm, fs)
        rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_y_mt!, rm_cur, rm_alt, m_cur, m_alt, bm, fs)
        rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_x_mt!, rm_cur, rm_alt, m_cur, m_alt, am, fs)
    end

    # The 6 n_sub sweeps are an even count, so the result is back in the
    # caller's arrays; the copy is a guard should the sequence ever change.
    if rm_cur !== rm_4d
        copyto!(rm_4d, rm_cur)
        copyto!(m,     m_cur)
    end
    return nothing
end

export AdvectionWorkspace, strang_split!, strang_split_mt!
