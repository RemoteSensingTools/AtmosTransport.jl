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
# Multi-tracer Strang splitting: X → Y → Z → Z → Y → X
# =========================================================================

"""
    strang_split_mt!(rm_4d, m, am, bm, cm, scheme, ws)

Multi-tracer Strang-split advection on a packed 4D tracer array.

This is the performance-optimized path: all `Nt = size(rm_4d, 4)` tracers
are processed in a SINGLE kernel launch per sweep direction (6 total),
rather than `6 × Nt` launches in the per-tracer path.

The mass update (m_new = m + flux_in - flux_out) is computed ONCE per
cell per sweep, shared across all tracers.

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
                          cfl_limit::Real = one(FT),
                          diffusion_op::AbstractDiffusion = NoDiffusion(),
                          diffusion_workspace = nothing,
                          emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                          tracer_names::Union{Nothing, Tuple} = nothing,
                          meteo = nothing,
                          grid = nothing,
                          dt::Union{Nothing, Real} = nothing) where FT
    _require_structured_vertical(scheme)
    _preflight_advection_workspace(ws, rm_4d, m)
    _preflight_diffusion(diffusion_op, diffusion_workspace, dt,
                         m, size(rm_4d, 4))
    cfl_ft = convert(FT, cfl_limit)

    # CFL subcycling per direction (reuse single-tracer pilot on the 3D mass)
    n_x = _x_subcycling_pass_count(am, m, ws, cfl_ft)
    n_y = _y_subcycling_pass_count(bm, m, ws, cfl_ft)
    n_z = _z_subcycling_pass_count(cm, m, ws, cfl_ft)

    fs_x = inv(FT(n_x))
    fs_y = inv(FT(n_y))
    fs_z = inv(FT(n_z))

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

    # Forward half: X → Y → Z
    for _ in 1:n_x; rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_x_mt!, rm_cur, rm_alt, m_cur, m_alt, am, fs_x); end
    for _ in 1:n_y; rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_y_mt!, rm_cur, rm_alt, m_cur, m_alt, bm, fs_y); end
    for _ in 1:n_z; rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_z_mt!, rm_cur, rm_alt, m_cur, m_alt, cm, fs_z); end

    # Palindrome center.
    # Two configurations:
    #
    # 1. `emissions_op isa NoSurfaceFlux` (the default): single V(dt)
    #    at the palindrome center. NoDiffusion is a dead branch,
    #    so with both defaults this whole block collapses to zero
    #    floating-point work and is bit-exact with the no-op behavior.
    #
    # 2. `emissions_op isa SurfaceFluxOperator`: the OPERATOR_COMPOSITION.md
    #    §3.2 arrangement, V(dt/2) → S(dt) → V(dt/2). Fresh emissions
    #    see vertical mixing before the reverse-half horizontal sweeps
    #    transport them. Emissions enter at the palindrome center with
    #    the FULL dt (not halved) — sources/sinks don't participate in
    #    the Strang half-step dance; symmetric operators around them
    #    provide the 2nd-order accuracy.
    #
    # Linear-operator caveat: V(dt) = V(dt/2) ∘ V(dt/2) is exact for
    # the continuous ODE flow but only O(dt²) for Backward Euler
    # ((I-dt·D)⁻¹ ≠ [(I-dt/2·D)⁻¹]²). Switching from Path 1 to Path 2
    # is therefore NOT bit-exact when `diffusion_op` is non-trivial —
    # the two halves of V differ by O((dt·D)²). Acceptable since Path 2
    # is only reached when the user opts in to emissions.
    # Route through the mass-flux VMR wrapper so the palindrome-
    # center diffusion step is column-mass conserving. `m_cur` is the
    # current air-mass field `(Nx, Ny, Nz)` paired with `rm_cur`'s
    # `(Nx, Ny, Nz, Nt)` tracer storage; the wrapper does the
    # tracer_mass ↔ VMR scaling internally.
    if emissions_op isa NoSurfaceFlux
        apply_vertical_diffusion_vmr!(rm_cur, m_cur, diffusion_op, diffusion_workspace, dt, meteo)
    elseif uses_diffusive_surface_flux_boundary(diffusion_op)
        tracer_names === nothing && throw(ArgumentError(
            "strang_split_mt!: `emissions_op` is non-trivial but " *
            "`tracer_names` was not supplied — the surface-flux " *
            "kernel needs per-tracer index resolution. Pass " *
            "`tracer_names = state.tracer_names`."))
        apply_surface_flux!(rm_cur, emissions_op, ws, dt, meteo, grid;
                            tracer_names = tracer_names)
        apply_vertical_diffusion_vmr!(rm_cur, m_cur, diffusion_op, diffusion_workspace, dt, meteo)
    else
        tracer_names === nothing && throw(ArgumentError(
            "strang_split_mt!: `emissions_op` is non-trivial but " *
            "`tracer_names` was not supplied — the surface-flux " *
            "kernel needs per-tracer index resolution. Pass " *
            "`tracer_names = state.tracer_names`."))
        half_dt = dt === nothing ? nothing : dt / 2
        apply_vertical_diffusion_vmr!(rm_cur, m_cur, diffusion_op, diffusion_workspace, half_dt, meteo)
        apply_surface_flux!(rm_cur, emissions_op, ws, dt, meteo, grid;
                            tracer_names = tracer_names)
        apply_vertical_diffusion_vmr!(rm_cur, m_cur, diffusion_op, diffusion_workspace, half_dt, meteo)
    end

    # Reverse half: Z → Y → X
    for _ in 1:n_z; rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_z_mt!, rm_cur, rm_alt, m_cur, m_alt, cm, fs_z); end
    for _ in 1:n_y; rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_y_mt!, rm_cur, rm_alt, m_cur, m_alt, bm, fs_y); end
    for _ in 1:n_x; rm_cur, rm_alt, m_cur, m_alt = _pass!(sweep_x_mt!, rm_cur, rm_alt, m_cur, m_alt, am, fs_x); end

    # If total parity wound up in the alternate (B) buffer, copy back.
    # When (n_x + n_y + n_z + n_z + n_y + n_x) is even — the common case,
    # e.g. all-ones — the final result lands in the caller's arrays with
    # zero copyto! calls.
    if rm_cur !== rm_4d
        copyto!(rm_4d, rm_cur)
        copyto!(m,     m_cur)
    end
    return nothing
end

export AdvectionWorkspace, strang_split!, strang_split_mt!
