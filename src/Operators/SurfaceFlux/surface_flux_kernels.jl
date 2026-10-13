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
    _surface_flux_cs_kernel!(q_raw, rate, comp, dt, tracer_idx, Nz, Hp, deposit)

Cubed-sphere surface-flux kernel for one halo-padded panel, single-tracer
`(Nc + 2Hp, Nc + 2Hp, Nz)` or packed `(…, Nz, Nt)` (`tracer_idx` selects the
slab; ignored for single-tracer panels). `rate`/`comp` are interior `(Nc, Nc)`.
"""
@kernel function _surface_flux_cs_kernel!(q_raw, @Const(rate), comp, dt, tracer_idx, Nz, Hp, deposit)
    ii, jj = @index(Global, NTuple)
    @inbounds _deposit!(q_raw, comp, rate[ii, jj] * dt, ii, jj, Hp, Nz, tracer_idx, deposit)
end

"""
    _surface_flux_cs_interp_kernel!(q_raw, series, comp, w0, w1, i0, i1, dt, tracer_idx, Nz, Hp, deposit)

Time-interpolated variant: the increment is `(w0·series[i0] + w1·series[i1])·dt`.
`series` is one panel of a time-varying source's window buffer, `(Nc, Nc, capacity)`,
and `i0`, `i1` are the buffer slots holding the two slices.
"""
@kernel function _surface_flux_cs_interp_kernel!(q_raw, @Const(series), comp, w0, w1, i0, i1,
                                                  dt, tracer_idx, Nz, Hp, deposit)
    ii, jj = @index(Global, NTuple)
    @inbounds _deposit!(q_raw, comp, (w0 * series[ii, jj, i0] + w1 * series[ii, jj, i1]) * dt,
                        ii, jj, Hp, Nz, tracer_idx, deposit)
end

