# Reference transcription of TM5's zonal slopes advection, used to state what
# `SlopesScheme` shares with TM5 and what it does not.
#
# Source: subroutine `dynamu` in TM5 `base/src/advectx.F90`, the body of the CFL
# loop (the copy used here has it at lines 626-716): the air-mass update, the
# `with_limits` clamp, the fluxes of `rm` and `rxm`, and the `rm`/`rxm` update.
# Only one periodic row (`xcyc = 1`), one tracer and one CFL sub-step
# (`nloop = 1`, i.e. |am| < m) are transcribed; the cross moments `rym` and
# `rzm`, which TM5 transports alongside, do not affect `rm` in an x sweep and
# are left out. It was written separately from the production kernels.
#
# Index convention: `a[i]` is TM5's `am(i)`, the air mass moved through the
# EAST face of cell `i` in one sweep (kg); periodicity gives `am(0) = am(N)`.
# TM5 carries the first moment `rxm` as prognostic state. The moment policy
# fixes what `rxm` is at the start of each sweep:
#
#   PrognosticMoment()       TM5 as written: `rxm` is carried from the last sweep
#   ZeroMoment()             `rxm := 0`, which reduces the flux to donor-cell upwind
#   DiagnosedMinmodMoment()  `rxm := m * minmod((c[i+1] - c[i-1])/2,
#                            2(c[i+1] - c[i]), 2(c[i] - c[i-1])) / 2` from the
#                            neighbouring cell means `c = rm/m`, the slope that
#                            `SlopesScheme(MonotoneLimiter())` re-diagnoses
#
# `limits = true` is TM5's default (`limits = .true.` in `dims.F90` with the
# `with_limits` macro): `rxm` is clamped to `[-rm, rm]` before the fluxes and
# again after the update.

abstract type AbstractTM5MomentPolicy end
struct PrognosticMoment      <: AbstractTM5MomentPolicy end
struct ZeroMoment            <: AbstractTM5MomentPolicy end
struct DiagnosedMinmodMoment <: AbstractTM5MomentPolicy end

tm5_minmod3(a, b, c) =
    (a > 0 && b > 0 && c > 0) ? min(a, b, c) :
    (a < 0 && b < 0 && c < 0) ? max(a, b, c) : zero(a)

tm5_start_moment!(rxm, rm, m, ::PrognosticMoment) = rxm
tm5_start_moment!(rxm, rm, m, ::ZeroMoment) = fill!(rxm, 0)

function tm5_start_moment!(rxm, rm, m, ::DiagnosedMinmodMoment)
    N = length(rm)
    c = rm ./ m
    for i in 1:N
        cl, cc, cr = c[mod1(i - 1, N)], c[i], c[mod1(i + 1, N)]
        rxm[i] = m[i] * tm5_minmod3((cr - cl) / 2, 2 * (cr - cc), 2 * (cc - cl)) / 2
    end
    return rxm
end

tm5_clamp_moment(rxm, rm) = max(min(rxm, rm), -rm)   # TM5: max(min(rxm, rm), -rm)

"""
    tm5_dynamu_sweep!(rm, rxm, m, a, policy; limits = true)

One TM5 `dynamu` x sweep of a periodic row, in place on `rm`, `rxm` and `m`.
"""
function tm5_dynamu_sweep!(rm, rxm, m, a, policy::AbstractTM5MomentPolicy; limits::Bool = true)
    N = length(rm)
    am(i) = a[mod1(i, N)]                       # am(0) = am(N)
    tm5_start_moment!(rxm, rm, m, policy)
    mnew = [m[i] + am(i - 1) - am(i) for i in 1:N]
    if limits
        for i in 1:N
            rxm[i] = tm5_clamp_moment(rxm[i], rm[i])
        end
    end
    f  = zeros(eltype(rm), N + 1)               # f[i + 1] is TM5 f(i), i = 0..N
    pf = zeros(eltype(rm), N + 1)
    for i in 0:N
        if am(i) >= 0                           # donor: cell i (cell 0 is cell N)
            d = mod1(i, N)
            alpha = am(i) / m[d]
            f[i + 1] = alpha * (rm[d] + (1 - alpha) * rxm[d])
        else                                    # donor: cell i + 1
            d = mod1(i + 1, N)
            alpha = am(i) / m[d]
            f[i + 1] = alpha * (rm[d] - (1 + alpha) * rxm[d])
        end
        pf[i + 1] = am(i) * (alpha * alpha * rxm[d] - 3 * f[i + 1])
    end
    f[1] = f[N + 1]                             # left fluxes from periodicity
    pf[1] = pf[N + 1]
    for i in 1:N                                # rm first; rxm uses the NEW rm, OLD m
        rm[i] = rm[i] + (f[i] - f[i + 1])
        rxm[i] = rxm[i] + (pf[i] - pf[i + 1] - (am(i - 1) - am(i)) * rxm[i] +
                           3 * ((am(i - 1) + am(i)) * rm[i] - (f[i] + f[i + 1]) * m[i])) / mnew[i]
        limits && (rxm[i] = tm5_clamp_moment(rxm[i], rm[i]))
    end
    m .= mnew
    return nothing
end

"""
    tm5_dynamu_run(c0, m0, a, nsweeps, policy; limits = true) -> (rm, m)

Cold start (`rxm = 0`, as TM5 starts) from mixing ratio `c0` and air mass `m0`,
then `nsweeps` sweeps with the same face masses `a`.
"""
function tm5_dynamu_run(c0, m0, a, nsweeps, policy::AbstractTM5MomentPolicy; limits::Bool = true)
    m = copy(m0)
    rm = c0 .* m
    rxm = zero(rm)
    for _ in 1:nsweeps
        tm5_dynamu_sweep!(rm, rxm, m, a, policy; limits)
    end
    return rm, m
end
