# All kernels use Kahan compensated addition to prevent F32 rounding loss when
# a small emission increment is added to a large background tracer field.
#
# Kahan update: y = x - c; t = s + y; c = (t - s) - y; s = t
# where s = current cell value, x = rate*dt increment, c = running compensation.
# Each kernel takes a `comp` array (same surface shape as `rate`) that persists
# across substeps, carrying the accumulated rounding debt forward.

"""
    _surface_flux_kernel!(q_raw, rate, comp, dt, tracer_idx, Nz)

KernelAbstractions kernel that adds a single source's surface flux to
one tracer slab inside the 4D `tracers_raw` buffer using Kahan compensated
addition.

For structured grids, `q_raw` has shape `(Nx, Ny, Nz, Nt)`. The kernel
is launched over `(Nx, Ny)` and every thread updates the surface layer
at `k = Nz` for the tracer at `tracer_idx`:

    Kahan: y = rate[i,j]*dt - comp[i,j]
           t = q_raw[i,j,Nz,tracer_idx] + y
           comp[i,j] = (t - q_raw[i,j,Nz,tracer_idx]) - y
           q_raw[i,j,Nz,tracer_idx] = t

`comp` has shape `(Nx, Ny)` and is zero-initialised at source construction;
it persists across all substeps so the rounding debt accumulates correctly.
"""
@kernel function _surface_flux_kernel!(q_raw, @Const(rate), comp, dt, tracer_idx, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        x = rate[i, j] * dt
        s = q_raw[i, j, Nz, tracer_idx]
        c = comp[i, j]
        y = x - c
        t = s + y
        comp[i, j]                   = (t - s) - y
        q_raw[i, j, Nz, tracer_idx]  = t
    end
end

"""
    _surface_flux_face_kernel!(q_raw, rate, comp, dt, tracer_idx, Nz)

Face-indexed packed surface-flux kernel with Kahan compensation. `q_raw`
has shape `(ncells, Nz, Nt)` and `rate`/`comp` have shape `(ncells,)`.
"""
@kernel function _surface_flux_face_kernel!(q_raw, @Const(rate), comp, dt, tracer_idx, Nz)
    c_idx = @index(Global, Linear)
    @inbounds begin
        x = rate[c_idx] * dt
        s = q_raw[c_idx, Nz, tracer_idx]
        c = comp[c_idx]
        y = x - c
        t = s + y
        comp[c_idx]                    = (t - s) - y
        q_raw[c_idx, Nz, tracer_idx]   = t
    end
end

"""
    _surface_flux_face_single_kernel!(q_raw, rate, comp, dt, Nz)

Face-indexed single-tracer helper with Kahan compensation for a
`(ncells, Nz)` tracer slice. Used by the reduced-Gaussian advection
palindrome.
"""
@kernel function _surface_flux_face_single_kernel!(q_raw, @Const(rate), comp, dt, Nz)
    c_idx = @index(Global, Linear)
    @inbounds begin
        x = rate[c_idx] * dt
        s = q_raw[c_idx, Nz]
        c = comp[c_idx]
        y = x - c
        t = s + y
        comp[c_idx]   = (t - s) - y
        q_raw[c_idx, Nz] = t
    end
end

# ---------------------------------------------------------------------------
# Cubed sphere: where a column's emitted mass goes.
# ---------------------------------------------------------------------------

"""
    SurfaceLayerDeposit()
    ProfileDeposit(profile)

Vertical placement of the mass emitted into a cubed-sphere column during one
step. `SurfaceLayerDeposit` adds it to the surface layer `k = Nz` with Kahan
compensation. `ProfileDeposit` spreads it over the column with per-layer
fractions `profile[i, j, k]` that sum to one: the non-local PBL transport of
fresh surface emissions in GEOS-Chem's VDIFF (the counter-gradient term
`∂(ρ K γ)/∂z`, linear in the surface flux). A fraction may be negative: that
layer loses tracer to the layers above.

Of a column's emission ``x``, the counter-gradient redistribution moves
``c_k = x\\,(f_k - \\delta_{k N})`` into layer ``k`` (``f_k`` the fraction,
``N = N_z`` the surface layer). As in GEOS-Chem's `qmincg` check, the column
keeps all of ``x`` in the surface layer instead when that redistribution
would make a layer negative or a negative layer more negative,

```math
\\exists\\, k:\\quad c_k < 0 \\quad\\text{and}\\quad q_k + c_k < 0,
```

with ``q_k`` the tracer before the step's emission. A layer the
redistribution raises or leaves unchanged (``c_k \\ge 0``) never triggers the
fallback, so the roundoff negatives that transport leaves in zero-background
tracers do not. GEOS-Chem tests ``q_k + c_k < 0`` on its bottom `npbl`
layers, which hold every nonzero ``c_k``; it also falls back on a layer that
was negative before the step and that the redistribution does not lower.

Both placements add exactly ``x`` to the column. Upper-layer additions are
tracked with `_two_sum`, and the surface layer receives the emission minus what
landed above through the Kahan update, so the column gains exactly the
emitted mass.
"""
struct SurfaceLayerDeposit end
struct ProfileDeposit{P}
    profile :: P
end

Adapt.adapt_structure(to, d::ProfileDeposit) = ProfileDeposit(Adapt.adapt(to, d.profile))

# One cell of a single-tracer (3-D) or packed multi-tracer (4-D) panel.
@inline _tracer_cell(::AbstractArray{<:Any, 3}, i, j, k, _t) = CartesianIndex(i, j, k)
@inline _tracer_cell(::AbstractArray{<:Any, 4}, i, j, k, t) = CartesianIndex(i, j, k, t)

@inline function _deposit!(q, comp, x, ii, jj, Hp, Nz, t, ::SurfaceLayerDeposit)
    @inbounds begin
        cell = _tracer_cell(q, ii + Hp, jj + Hp, Nz, t)
        s = q[cell]
        y = x - comp[ii, jj]
        s_new = s + y
        comp[ii, jj] = (s_new - s) - y
        q[cell] = s_new
    end
    return nothing
end

# The upper-layer shares are added plainly; TwoSum recovers exactly what each
# addition rounded away, and the surface layer (Kahan) receives `x` minus what
# actually landed above, so the column gains exactly `x`.
@inline function _deposit!(q, comp, x, ii, jj, Hp, Nz, t, d::ProfileDeposit)
    f = d.profile
    @inbounds begin
        for k in 1:Nz
            redistributed = x * (f[ii, jj, k] - (k == Nz))       # counter-gradient part only
            # qmincg: only a layer the redistribution lowers can trigger the fallback;
            # a roundoff-negative layer it raises or leaves unchanged does not.
            redistributed < 0 && q[_tracer_cell(q, ii + Hp, jj + Hp, k, t)] + redistributed < 0 &&
                return _deposit!(q, comp, x, ii, jj, Hp, Nz, t, SurfaceLayerDeposit())
        end
        lofted, lofted_err = zero(x), zero(x)          # mass that reached layers 1…Nz-1
        for k in 1:(Nz - 1)
            cell = _tracer_cell(q, ii + Hp, jj + Hp, k, t)
            share = x * f[ii, jj, k]
            q[cell], lost = _two_sum(q[cell], share)
            lofted, e = _two_sum(lofted, share)
            lofted_err += e - lost
        end
    end
    return _deposit!(q, comp, (x - lofted) - lofted_err, ii, jj, Hp, Nz, t, SurfaceLayerDeposit())
end

"""
    CSSurfaceFluxSegment{N, FT}(tracer, slot0, slot1, w0, w1, dt)

One segment of the step of the `N`-th source in the context of a cubed-sphere
surface-flux launch, as a point operation over (column, panel): the increment
`rate[ii, jj] * dt` of a static source, or `(w0 * slots[ii, jj, slot0] +
w1 * slots[ii, jj, slot1]) * dt` of a time-varying one, is deposited into
tracer `tracer` (`tracer = 0`: inactive, an empty index space). The sources
of an operator target distinct tracers and own their compensation arrays, so
their operations write disjoint cells and fuse into one launch per segment.
"""
struct CSSurfaceFluxSegment{N, FT} <: AbstractPointOp
    tracer::Int
    slot0::Int
    slot1::Int
    w0::FT
    w1::FT
    dt::FT
end

# Segment `g` of a source's plan; inactive past its last segment.
function CSSurfaceFluxSegment{N}(plan, g) where N
    FT = typeof(plan.dt)
    (plan.tracer == 0 || g > length(plan.segments)) &&
        return CSSurfaceFluxSegment{N, FT}(0, 1, 1, zero(FT), zero(FT), zero(FT))
    i0, i1, w0, w1, frac = plan.segments[g]
    return CSSurfaceFluxSegment{N, FT}(plan.tracer, i0, i1, FT(w0), FT(w1), plan.dt * FT(frac))
end

index_space(op::CSSurfaceFluxSegment, ctx) = (op.tracer == 0 ? 0 : ctx.Nc * ctx.Ny, 6)
host_parallel(::CSSurfaceFluxSegment) = true      # a column walk per index with `ProfileDeposit`

@inline _cs_surface_flux_increment(rate::AbstractArray{<:Any, 2}, ii, jj, op) =
    @inbounds rate[ii, jj] * op.dt
@inline _cs_surface_flux_increment(slots::AbstractArray{<:Any, 3}, ii, jj, op) =
    @inbounds (op.w0 * slots[ii, jj, op.slot0] + op.w1 * slots[ii, jj, op.slot1]) * op.dt

@inline function apply_point!(op::CSSurfaceFluxSegment{N}, ctx, c, p) where N
    (; q_raw, Nc, Nz, Hp, deposit) = ctx
    rates, comp = ctx.rates[N], ctx.comp[N]
    rates === nothing && return nothing      # a source without its tracer (never active)
    jj, ii = divrem(c - 1, Nc) .+ 1
    x = _cs_surface_flux_increment(select_panel(rates, p), ii, jj, op)
    _deposit!(select_panel(q_raw, p), select_panel(comp, p), x, ii, jj, Hp, Nz, op.tracer,
              _panel_deposit(deposit, p))
    return nothing
end
