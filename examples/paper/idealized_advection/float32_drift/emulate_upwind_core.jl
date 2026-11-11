# Emulation of the flux-form upwind update of cs_sweep_x.jl on a doubly periodic 2-D grid, to
# test whether Float32 rounding of `rm + fl - fr` alone biases the global tracer mass.
using Printf, Statistics

function fluxes(N, t, T, cfl, m0::Matrix{Float64})
    ψ(x, y) = (sin(2π * x)^2 * cos(π * y)^2 * cos(π * t / T) * 0.15 + 0.07 * sin(2π * y))
    am = zeros(N + 1, N); bm = zeros(N, N + 1)
    for j in 1:N, i in 1:N+1
        am[i, j] = -(ψ((i - 1) / N, j / N) - ψ((i - 1) / N, (j - 1) / N))
    end
    for j in 1:N+1, i in 1:N
        bm[i, j] = (ψ(i / N, (j - 1) / N) - ψ((i - 1) / N, (j - 1) / N))
    end
    s = cfl * minimum(m0) / max(maximum(abs, am), maximum(abs, bm))
    return am .* s, bm .* s
end

@inline function gflux(F::T, md::T, rd::T) where T
    γ = F >= zero(T) ? clamp(F / md, zero(T), one(T)) : clamp(F / md, -one(T), zero(T))
    return γ * rd
end

function xsweep!(rmn, mn, rm, m, am, N)
    @inbounds for j in 1:N, i in 1:N
        im = i == 1 ? N : i - 1; ip = i == N ? 1 : i + 1
        al = am[i, j]; ar = am[i == N ? 1 : i + 1, j]   # periodic: face N+1 == face 1
        mn[i, j] = m[i, j] + al - ar
        fl = al >= 0 ? gflux(al, m[im, j], rm[im, j]) : gflux(al, m[i, j], rm[i, j])
        fr = ar >= 0 ? gflux(ar, m[i, j], rm[i, j]) : gflux(ar, m[ip, j], rm[ip, j])
        rmn[i, j] = rm[i, j] + fl - fr
    end
end
function ysweep!(rmn, mn, rm, m, bm, N)
    @inbounds for j in 1:N, i in 1:N
        jm = j == 1 ? N : j - 1; jp = j == N ? 1 : j + 1
        bs = bm[i, j]; bn = bm[i, j == N ? 1 : j + 1]
        mn[i, j] = m[i, j] + bs - bn
        fs = bs >= 0 ? gflux(bs, m[i, jm], rm[i, jm]) : gflux(bs, m[i, j], rm[i, j])
        fn = bn >= 0 ? gflux(bn, m[i, j], rm[i, j]) : gflux(bn, m[i, jp], rm[i, jp])
        rmn[i, j] = rm[i, j] + fs - fn
    end
end
