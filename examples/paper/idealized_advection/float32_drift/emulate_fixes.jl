# Candidate fixes for the Float32 drift of the flux-form update, in the emulation:
#   :two_step   rm + fl - fr                 (current kernels: two roundings at the scale of rm)
#   :one_step   rm + (fl - fr)               (one rounding at the scale of rm)
#   :carry      rm + (fl - fr) with the rounding residual kept per cell and re-applied (TwoSum)
using Printf, Statistics
include(joinpath(@__DIR__, "emulate_upwind_core.jl"))

@inline function two_sum(a::T, b::T) where T
    s = a + b; bb = s - a
    return s, (a - (s - bb)) + (b - bb)
end

@inline function update(::Val{:two_step}, ri, fl, fr, c)
    return ri + fl - fr, c
end
@inline function update(::Val{:one_step}, ri, fl, fr, c)
    return ri + (fl - fr), c
end
# One rounding of the exact three-term sum, in Float32 only (error-free transformations).
@inline function update(::Val{:round_once}, ri, fl, fr, c)
    d, ed = two_sum(fl, -fr)
    s, es = two_sum(ri, d)
    return s + (es + ed), c
end
@inline function update(::Val{:round_once_f64}, ri, fl, fr, c)
    return typeof(ri)(Float64(ri) + Float64(fl) - Float64(fr)), c
end
@inline function update(::Val{:carry}, ri, fl, fr, c)
    d, ed = two_sum(fl, -fr)
    s, es = two_sum(ri, d + (ed + c))
    return s, es
end

function sweep!(rmn, mn, cn, rm, m, c, F, N, dir, mode)
    @inbounds for j in 1:N, i in 1:N
        if dir == 1
            a = (i == 1 ? N : i - 1, j); b = (i == N ? 1 : i + 1, j)
            Fl = F[i, j]; Fr = F[i == N ? 1 : i + 1, j]
        else
            a = (i, j == 1 ? N : j - 1); b = (i, j == N ? 1 : j + 1)
            Fl = F[i, j]; Fr = F[i, j == N ? 1 : j + 1]
        end
        mn[i, j] = m[i, j] + Fl - Fr
        fl = Fl >= 0 ? gflux(Fl, m[a...], rm[a...]) : gflux(Fl, m[i, j], rm[i, j])
        fr = Fr >= 0 ? gflux(Fr, m[i, j], rm[i, j]) : gflux(Fr, m[b...], rm[b...])
        rmn[i, j], cn[i, j] = update(mode, rm[i, j], fl, fr, c[i, j])
    end
end

function run(mode; N = 96, nwin = 288, nsub = 9, cfl = 0.4, field = :uniform)
    T = Float32
    m064 = [1.0e11 * (1 + 0.2 * sin(2π * i / N) * cos(2π * j / N)) for i in 1:N, j in 1:N]
    χ0 = field == :uniform ? ones(N, N) :
         [0.1 + 0.9 * exp(-40 * ((i / N - 0.3)^2 + (j / N - 0.5)^2)) for i in 1:N, j in 1:N]
    m0 = T.(m064); m = copy(m0); rm = T.(χ0) .* m0; mb = similar(m); rb = similar(rm)
    c = zeros(T, N, N); cb = similar(c)
    S0 = sum(Float64, rm)
    series = Float64[]
    for w in 1:nwin
        am64, bm64 = fluxes(N, w - 0.5, nwin, cfl * nsub, m064)
        am = T.(am64 ./ (2nsub)); bm = T.(bm64 ./ (2nsub))
        am[N+1, :] .= am[1, :]; bm[:, N+1] .= bm[:, 1]
        m .= m0
        for half in 1:2, _ in 1:nsub, d in (half == 1 ? (1, 2) : (2, 1))
            sweep!(rb, mb, cb, rm, m, c, d == 1 ? am : bm, N, d, mode)
            rm, rb = rb, rm; m, mb = mb, m; c, cb = cb, c
        end
        w % (nwin ÷ 4) == 0 && push!(series, (sum(Float64, rm) + sum(Float64, c)) / S0 - 1)
    end
    return series, sum(Float64, c) / S0
end

const MODES = (Val(:two_step), Val(:one_step), Val(:round_once), Val(:round_once_f64), Val(:carry))
for field in (:uniform, :hill), mode in MODES
    s, cres = run(mode; field)
    @printf("%-8s %-10s drift at T/4,T/2,3T/4,T: %s   (carry held: %+.1e)\n", field, typeof(mode).parameters[1],
            join((@sprintf("%+.2e", v) for v in s), " "), cres)
end
for N in (48, 192), mode in (Val(:two_step), Val(:round_once), Val(:carry))
    s, _ = run(mode; N, nsub = N == 48 ? 5 : 18)
    @printf("uniform  %-10s N=%d: %s\n", typeof(mode).parameters[1], N, join((@sprintf("%+.2e", v) for v in s), " "))
end
