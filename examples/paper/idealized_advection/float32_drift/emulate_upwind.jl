# Float32 drift of the flux-form upwind update in a stand-alone emulation: uniform tracer, time-
# dependent and steady flow, three resolutions, and the Float64 control. See README.md.
using Printf, Statistics
include(joinpath(@__DIR__, "emulate_upwind_core.jl"))

function run(::Type{T}; N = 96, nwin = 288, nsub = 9, cfl = 0.4, reset = true, steady = false) where T
    m064 = [1.0e11 * (1 + 0.2 * sin(2π * i / N) * cos(2π * j / N)) for i in 1:N, j in 1:N]
    m0 = T.(m064)
    m = copy(m0); rm = copy(m0)                       # uniform mixing ratio 1
    mb = similar(m); rb = similar(rm)
    S0 = sum(Float64, rm)
    series = Float64[]
    for w in 1:nwin
        tmid = steady ? 0.0 : (w - 0.5)
        am64, bm64 = fluxes(N, tmid, nwin, cfl * nsub, m064)
        am = T.(am64 ./ (2nsub)); bm = T.(bm64 ./ (2nsub))
        am[N+1, :] .= am[1, :]; bm[:, N+1] .= bm[:, 1]
        reset && (m .= m0)
        for _ in 1:nsub                                # (X Y)^n
            xsweep!(rb, mb, rm, m, am, N); rm, rb = rb, rm; m, mb = mb, m
            ysweep!(rb, mb, rm, m, bm, N); rm, rb = rb, rm; m, mb = mb, m
        end
        for _ in 1:nsub                                # (Y X)^n
            ysweep!(rb, mb, rm, m, bm, N); rm, rb = rb, rm; m, mb = mb, m
            xsweep!(rb, mb, rm, m, am, N); rm, rb = rb, rm; m, mb = mb, m
        end
        w % (nwin ÷ 4) == 0 && push!(series, sum(Float64, rm) / S0 - 1)
    end
    χ = Float64.(rm) ./ Float64.(m0)
    return series, extrema(χ)
end

for (label, kw) in (("time-dependent, reset", (;)), ("time-dependent, no reset", (; reset = false)),
                    ("steady, reset", (; steady = true)), ("N=192", (; N = 192, nsub = 18)),
                    ("N=48", (; N = 48, nsub = 5)))
    s, ex = run(Float32; kw...)
    N = get(kw, :N, 96); nsub = get(kw, :nsub, 9)
    noise = 0.41 * 6e-8 / N * sqrt(288 * nsub * 4)     # random-walk expectation (relative)
    @printf("%-26s drift at T/4,T/2,3T/4,T: %s   chi range %.2e..%.2e  noise~%.1e\n", label,
            join((@sprintf("%+.2e", v) for v in s), " "), ex[1] - 1, ex[2] - 1, noise)
end
s, ex = run(Float64)
@printf("%-26s drift: %s\n", "Float64", join((@sprintf("%+.2e", v) for v in s), " "))
