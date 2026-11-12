# ---------------------------------------------------------------------------
# IFS convective tracer transport of one column (CUCTRACER + CUBIDIAG).
#
# Layer k = 1 is the model top, k = N the lowest layer. Interface i is the top
# of layer i; interface N + 1 is the ground. The scratch vectors `mu`, `md`,
# `cu`, `cd` hold interface values in slots 1..N; interface N + 1 carries no
# flux and is never stored. Mixing ratios are tracer mass over air mass of the
# same cell (`q[k, t] / m[k]`), so the cell area cancels there; it enters only
# where a mass flux (kg m-2 s-1) meets a layer mass (`m[k] / area`, kg m-2).
#
# Per column the interface fluxes and the Courant number are computed once
# (`_ifs_column_fluxes!`); per tracer and sub-step `_ifs_substep!` builds the
# plume values, the plume flux and solves the implicit compensating flow for
# the increment of the mixing ratio.
# ---------------------------------------------------------------------------

# Column status codes returned in place of the Courant number; the host check
# (`_ifs_check_courant`) turns them into errors before any tracer moves.
const _IFS_NEGATIVE_FORCING = -1     # a negative entu, detu, entd or detd
const _IFS_AIRLESS_LAYER = -2        # a layer without air (m <= 0) among the rewritten layers

"""
    _ifs_column_fluxes!(mu, md, m, entu, detu, entd, detd, area, dt) -> courant

Interface mass fluxes of one column from the layer fields. The updraft is the
cumulative sum `mu[k] = max(mu[k+1] + entu[k] - detu[k], 0)` from the ground
up, closed at the top of its highest detraining layer `kt` (the first `k` with
`detu[k] > 0`): `mu[k] = 0` for `k <= kt`, and no updraft at all if no layer
detrains. The downdraft magnitude `md[k+1] = max(md[k] + entd[k] - detd[k], 0)`
runs from the top down and is closed at the bottom of its lowest detraining
layer `kb` (the last `k` with `detd[k] > 0`): `md[i] = 0` for `i > kb`. Sums of
stored (time- and area-averaged, rounded) fields do not return to zero
exactly; without the closure the remainder would travel on to the model top or
the ground. Nothing crosses either.

Returns the column's largest Courant number over the layers a flux touches,
`dt (max(mu[k], mu[k+1]) + max(md[k], md[k+1])) / (m[k] / area)`. Instead, it
returns `_IFS_NEGATIVE_FORCING` if a field value is negative,
`_IFS_AIRLESS_LAYER` if a layer the transport rewrites (the highest layer a
flux touches and every layer below it) has no air, and `Inf` if a field, the
air mass of such a layer or the Courant number is not finite.
"""
@inline function _ifs_column_fluxes!(mu, md, m, entu, detu, entd, detd, area, dt)
    FT = eltype(mu)
    N = length(mu)
    negative = false
    nonfinite = false
    kt = 0                                       # highest layer with detu > 0
    kb = 0                                       # lowest layer with detd > 0
    @inbounds for k in 1:N
        eu, du, ed, dd = entu[k], detu[k], entd[k], detd[k]
        nonfinite |= !(isfinite(eu) & isfinite(du) & isfinite(ed) & isfinite(dd))
        negative |= (eu < 0) | (du < 0) | (ed < 0) | (dd < 0)
        (kt == 0) & (du > 0) && (kt = k)
        dd > 0 && (kb = k)
    end
    @inbounds begin
        below = zero(FT)                         # mu at interface N + 1
        for k in N:-1:1
            below = k > kt > 0 ? max(below + entu[k] - detu[k], zero(FT)) : zero(FT)
            mu[k] = below
        end
        md[1] = zero(FT)
        above = zero(FT)
        for k in 1:(N - 1)
            above = k < kb ? max(above + entd[k] - detd[k], zero(FT)) : zero(FT)
            md[k + 1] = above
        end
    end
    negative && return FT(_IFS_NEGATIVE_FORCING)
    nonfinite && return FT(Inf)
    worst = zero(FT)
    airless = false
    rewritten = false                            # from the highest layer a flux touches down
    @inbounds for k in 1:N
        mu_below = k < N ? mu[k + 1] : zero(FT)
        md_below = k < N ? md[k + 1] : zero(FT)
        flux = max(mu[k], mu_below) + max(md[k], md_below)
        rewritten |= flux > 0
        rewritten || continue
        mk = m[k]
        if isnan(mk)
            nonfinite = true
        elseif mk <= 0
            airless = true
        elseif flux > 0
            r = dt * flux / (mk / area)
            isfinite(r) || (nonfinite = true)
            worst = max(worst, r)
        end
    end
    airless && return FT(_IFS_AIRLESS_LAYER)
    nonfinite && return FT(Inf)
    return worst
end

# First interface (2..N) that carries an updraft or downdraft flux; 0 if none.
@inline function _ifs_first_active_interface(mu, md)
    @inbounds for i in 2:length(mu)
        (mu[i] > 0 || md[i] > 0) && return i
    end
    return 0
end

# First interface (lo..N) with a downdraft; N + 1 if none.
@inline function _ifs_downdraft_top(md, lo)
    N = length(md)
    @inbounds for i in lo:N
        md[i] > 0 && return i
    end
    return N + 1
end

# Number of sub-steps of a column with Courant number `courant`.
@inline _ifs_substeps(courant, cfl_limit) =
    courant <= cfl_limit ? 1 : unsafe_trunc(Int, ceil(courant / cfl_limit))

# Downdraft value at its first interface from the updraft there and the
# environment of the layer above it.
@inline _ifs_downdraft_start(::IFSCuctracer, cu, c_above) =
    oftype(cu, 0.1) * cu + oftype(cu, 0.9) * c_above
@inline _ifs_downdraft_start(::IFSMonotoneUpstream, cu, c_above) = c_above

"""
    _ifs_substep!(q, t, m, detu, detd, mu, md, cu, cd, area, dt, lo, lfs, variant)

One step of length `dt` of tracer `t` of one column (`q` is the column's
`(N, Nt)` tracer mass, `m` its air mass per cell). `lo` is the first interface
with flux (`_ifs_first_active_interface`), `lfs` the first with a downdraft
(`_ifs_downdraft_top`). Layers `lo - 1 .. N` are updated; layers above are
not touched. `cu` and `cd` are scratch: they hold the plume values, then the
plume flux and the solver coefficients. The update is in flux form, so column
tracer mass is conserved up to rounding.
"""
@inline function _ifs_substep!(q, t, m, detu, detd, mu, md, cu, cd, area, dt,
                               lo, lfs, variant)
    FT = eltype(q)
    N = length(mu)
    @inbounds begin
        # Updraft values at interfaces lo..N, upwards, from the old environment.
        # IFS uses `detu` directly; where the archived detrainment exceeds the
        # flux entering the layer, the excess leaves as entrained air instead.
        cu[N] = q[N, t] / m[N]
        for k in (N - 1):-1:lo
            if mu[k] > 0
                through = mu[k + 1] - min(detu[k], mu[k + 1])
                entrained = mu[k] - through
                cu[k] = (through * cu[k + 1] + entrained * (q[k, t] / m[k])) / mu[k]
            else
                cu[k] = q[k - 1, t] / m[k - 1]
            end
        end
        # Downdraft values, downwards from its first interface `lfs`, through
        # layer i - 1. Interfaces without downdraft get the layer above, so no
        # stale scratch value enters a product with a zero flux.
        for i in lo:min(lfs - 1, N)
            cd[i] = q[i - 1, t] / m[i - 1]
        end
        if lfs <= N
            cd[lfs] = _ifs_downdraft_start(variant, cu[lfs], q[lfs - 1, t] / m[lfs - 1])
            for i in (lfs + 1):N
                if md[i] > 0
                    through = md[i - 1] - min(detd[i - 1], md[i - 1])
                    entrained = md[i] - through
                    cd[i] = (through * cd[i - 1] + entrained * (q[i - 1, t] / m[i - 1])) / md[i]
                else
                    cd[i] = q[i - 1, t] / m[i - 1]
                end
            end
            _ifs_lowest_layer_correction!(variant, q, t, m, mu, cu, md, cd, area, dt)
        end
        # Plume flux M_u C_u - M_d C_d at interfaces lo..N, stored in `cu`.
        for i in lo:N
            cu[i] = mu[i] * cu[i] - md[i] * cd[i]
        end
    end
    _ifs_compensating_solve!(variant, q, t, m, mu, md, cu, cd, area, dt, lo)
    return nothing
end

# IFS correction of the lowest layer (CUCTRACER, below cloud base): where a
# downdraft reaches interface N and the explicit estimate of layer N would be
# negative, the downdraft value there is reset so that the estimate is zero.
@inline function _ifs_lowest_layer_correction!(::IFSCuctracer, q, t, m, mu, cu, md, cd, area, dt)
    N = length(mu)
    @inbounds begin
        md[N] > 0 || return nothing
        m_N = m[N] / area
        c_N = q[N, t] / m[N]
        c_above = q[N - 1, t] / m[N - 1]
        change = dt / m_N * (-mu[N] * cu[N] + md[N] * cd[N] + (mu[N] - md[N]) * c_above)
        if c_N + change < 0
            cd[N] = (mu[N] * cu[N] - (mu[N] - md[N]) * c_above - c_N * m_N / dt) / md[N]
        end
    end
    return nothing
end
@inline _ifs_lowest_layer_correction!(::IFSMonotoneUpstream, args...) = nothing

# Plume flux and net mass flux at interface i (zero above `lo` and at the ground).
@inline _ifs_plume(cu, i, lo, N) = (lo <= i <= N) ? (@inbounds cu[i]) : zero(eltype(cu))
@inline _ifs_net(mu, md, i, lo, N) = (lo <= i <= N) ? (@inbounds mu[i] - md[i]) : zero(eltype(mu))

# Both solves are written for the increment `δ = c' - c` and add `m δ` to the
# stored tracer mass: an increment below half an ulp of the cell leaves it
# bit-identical, so a uniform field does not drift by rounding where the flux
# is tiny. `c` is the mixing ratio at the start of the sub-step.

# IFS (CUBIDIAG, RMFSOLCT = 1): the compensating flow carries the new value of
# the layer above in both directions; a lower bidiagonal system, solved from
# the top down. The plume flux is in `cu`.
#     δ_k = z (ΔF_k + M_k c'_{k-1} - M_{k+1} c_k) / (1 + z M_{k+1}),  c'_{k-1} = c_{k-1} + δ_{k-1}
@inline function _ifs_compensating_solve!(::IFSCuctracer, q, t, m, mu, md, cu, cd, area, dt, lo)
    FT = eltype(q)
    N = length(mu)
    above_new = zero(FT)             # the net flux at interface lo - 1 is zero
    @inbounds for k in (lo - 1):N
        z = dt / (m[k] / area)
        c = q[k, t] / m[k]
        net_top = _ifs_net(mu, md, k, lo, N)
        net_bot = _ifs_net(mu, md, k + 1, lo, N)
        dF = _ifs_plume(cu, k + 1, lo, N) - _ifs_plume(cu, k, lo, N)
        δ = z * (dF + net_top * above_new - net_bot * c) / (1 + z * net_bot)
        q[k, t] += m[k] * δ
        above_new = c + δ
    end
    return nothing
end

# Variant without new extrema: the compensating flow takes the value of the
# layer it comes from (above where the net flux is upward, below where it is
# downward); a tridiagonal system `A δ = r`, solved by the Thomas algorithm.
# The residual is
#     r_k = z (ΔF_k + max(M_k,0) c_{k-1} + min(M_k,0) c_k - max(M_{k+1},0) c_k - min(M_{k+1},0) c_{k+1}).
# The forward sweep stores the reduced residual in `cu` (slot k, once the plume
# flux of interface k has been read) and the reduced diagonal in `cd`.
@inline function _ifs_compensating_solve!(::IFSMonotoneUpstream, q, t, m, mu, md, cu, cd, area, dt, lo)
    FT = eltype(q)
    N = length(mu)
    @inbounds begin
        d_prev = one(FT); r_prev = zero(FT); u_prev = zero(FT)
        c_above = zero(FT)                   # multiplied by the zero flux at lo - 1
        for k in (lo - 1):N
            z = dt / (m[k] / area)
            c = q[k, t] / m[k]
            c_below = k < N ? q[k + 1, t] / m[k + 1] : zero(FT)
            net_top = _ifs_net(mu, md, k, lo, N)
            net_bot = _ifs_net(mu, md, k + 1, lo, N)
            dF = _ifs_plume(cu, k + 1, lo, N) - _ifs_plume(cu, k, lo, N)
            r = z * (dF + max(net_top, zero(FT)) * c_above + min(net_top, zero(FT)) * c -
                     max(net_bot, zero(FT)) * c - min(net_bot, zero(FT)) * c_below)
            d = 1 - z * min(net_top, zero(FT)) + z * max(net_bot, zero(FT))
            if k >= lo
                w = -z * max(net_top, zero(FT)) / d_prev
                d -= w * u_prev
                r -= w * r_prev
            end
            u_prev = z * min(net_bot, zero(FT))
            cu[k] = r
            cd[k] = d
            d_prev = d
            r_prev = r
            c_above = c
        end
        δ = cu[N] / cd[N]
        q[N, t] += m[N] * δ
        for k in (N - 1):-1:(lo - 1)
            z = dt / (m[k] / area)
            upper = z * min(_ifs_net(mu, md, k + 1, lo, N), zero(FT))
            δ = (cu[k] - upper * δ) / cd[k]
            q[k, t] += m[k] * δ
        end
    end
    return nothing
end

"""
    _ifs_restore_column_mass!(q, top, N, t, before, carry) -> carry

Column mass ledger of one sub-step of tracer `t`: the residual between the
compensated column sums before and after is added, with the carry, to the
largest cell of the whole column, so that cell and the carry do not change
with the cloud top. The bound on what counts as rounding comes from the
layers the sub-step rewrote, `top .. N`: at most `16 (N - top + 1)` ulps of
their largest cell. A larger residual is a loss of the scheme and stays in
the field, also when a much larger cell sits above the cloud.
"""
@inline function _ifs_restore_column_mass!(q, top, N, t, before, carry)
    FT = eltype(q)
    kmax = 1
    largest = zero(FT)
    rewritten = zero(FT)
    @inbounds for k in 1:N
        value = abs(q[k, t])
        value > largest && ((kmax, largest) = (k, value))
        k >= top && (rewritten = max(rewritten, value))
    end
    after = _tm5_column_sum(q, 1, N, t)
    r = _ledger_residual(before, after, rewritten, N - top + 1)
    value, carry = _ledger_deposit(@inbounds(q[kmax, t]), r, carry, N)
    @inbounds q[kmax, t] = value
    return carry
end

"""
    _ifs_column!(q, carry, m, detu, detd, mu, md, cu, cd, area, dt, courant, cfl_limit, variant)

All tracers of one column for one operator call: `n` sub-steps of length
`dt / n`, `n = ceil(courant / cfl_limit)`, each followed by the column mass
ledger (`_ifs_restore_column_mass!`): sums, deposit and carry over the whole
column, so the cell that receives the residual and the carry do not depend on
the cloud top, with the rounding bound of the rewritten layers.
`carry` is the column's ledger carry (one per tracer) or `nothing`. The fluxes
in `mu`, `md` come from `_ifs_column_fluxes!`. A column without flux is left
untouched; in the others, the transport rewrites layers `lo - 1 .. N` and the
ledger may change the column's largest cell.
"""
@inline function _ifs_column!(q, carry, m, detu, detd, mu, md, cu, cd, area, dt,
                              courant, cfl_limit, variant)
    FT = eltype(q)
    N = length(mu)
    lo = _ifs_first_active_interface(mu, md)
    lo == 0 && return nothing
    lfs = _ifs_downdraft_top(md, lo)
    n_sub = _ifs_substeps(courant, cfl_limit)
    dt_sub = dt / FT(n_sub)
    @inbounds for t in 1:size(q, 2)
        c = _load_ledger_carry(carry, FT, t)
        for _ in 1:n_sub
            before = _tm5_column_sum(q, 1, N, t)
            _ifs_substep!(q, t, m, detu, detd, mu, md, cu, cd, area, dt_sub, lo, lfs, variant)
            c = _ifs_restore_column_mass!(q, lo - 1, N, t, before, c)
        end
        _store_ledger_carry!(carry, c, t)
    end
    return nothing
end
