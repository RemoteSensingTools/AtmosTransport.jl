#!/usr/bin/env julia
# What `SlopesScheme` shares with TM5's slopes advection and what it does not,
# checked against an independent transcription of TM5 `dynamu`
# (test/helpers/tm5_dynamu_reference.jl) on a periodic 1-D row:
#
# - shared: the Russell-Lerner face flux. TM5 with its first moment `rxm`
#   re-diagnosed every sweep by the minmod rule is `SlopesScheme()`, and TM5
#   with `rxm := 0` is `UpwindScheme()`, to rounding.
# - not shared: TM5 carries `rxm` (and `rym`, `rzm`) as prognostic state with
#   its own update equation, and clamps it to |rxm| <= rm. `SlopesScheme` has
#   no moment state, so it is markedly more diffusive than TM5, and it stays
#   equivariant under a constant offset, which TM5's clamp is not.
#
# docs/src/theory/advection_schemes.md and validation_status.md cite this file.

using Test

using AtmosTransport
using .AtmosTransport.Operators: MonotoneLimiter
using .AtmosTransport.Operators.Advection: AdvectionWorkspace, sweep_x!

include(joinpath(@__DIR__, "..", "helpers", "tm5_dynamu_reference.jl"))

# `nsweeps` production x sweeps of a periodic row (Nx = N, Ny = Nz = 1) with the
# same face masses as the reference: the production array holds WEST faces,
# `am[i]` = TM5 `am(i - 1)`.
function production_run(c0, m0, a, nsweeps, scheme)
    N = length(c0)
    m  = reshape(copy(m0), N, 1, 1)
    rm = reshape(c0 .* m0, N, 1, 1)
    am = reshape([a[mod1(i - 1, N)] for i in 1:N + 1], N + 1, 1, 1)
    ws = AdvectionWorkspace(m)
    for _ in 1:nsweeps
        sweep_x!(rm, m, am, scheme, ws)
    end
    return vec(rm), vec(m)
end

relative_l2(c, c0) = sqrt(sum(abs2, c .- c0) / sum(abs2, c0))

const N = 120
const GAUSSIAN = exp.(-((((1:N) .- 0.5) .- (N / 2 + 0.5)) ./ 6.0) .^ 2)
const UNIT_MASS = ones(N)

@testset "TM5 dynamu reference transcription" begin
    a = fill(0.5, N)
    rm, m = tm5_dynamu_run(fill(400e-6, N), UNIT_MASS, a, 10 * 2N, PrognosticMoment())
    @test all(==(400e-6), rm ./ m)                       # uniform field stays exactly uniform
    rm, _ = tm5_dynamu_run(GAUSSIAN, UNIT_MASS, a, 2N, PrognosticMoment())
    @test abs(sum(rm) / sum(GAUSSIAN) - 1) < 1e-14       # tracer mass conserved

    # Second order: one revolution of 2 + sin(2πx) at Courant number 0.5.
    errors = map((25, 50, 100, 200)) do n
        c = 2 .+ sinpi.(2 .* ((1:n) .- 0.5) ./ n)
        rm, m = tm5_dynamu_run(c, ones(n), fill(0.5, n), 2n, PrognosticMoment())
        relative_l2(rm ./ m, c)
    end
    @test all(>(3.9), errors[1:end-1] ./ errors[2:end])
end

@testset "SlopesScheme is the Russell-Lerner flux with a re-diagnosed slope" begin
    flows = (("uniform, eastward", fill(0.5, N), 2N),
             ("uniform, westward", fill(-0.4, N), 2N),
             # Divergent: the air mass changes every sweep (by up to 25 %).
             ("divergent", [0.4 * (1 + 0.3 * sinpi(2i / N)) for i in 1:N], 40))
    @testset "$label" for (label, a, nsweeps) in flows
        # TM5's clamp is switched off: it is inactive for a non-negative field
        # with these moments, and for rm < 0 it turns rxm = 0 into |rm|.
        for c0 in (GAUSSIAN, GAUSSIAN .- 0.5)            # non-negative and signed
            rm_ref, m_ref = tm5_dynamu_run(c0, UNIT_MASS, a, nsweeps,
                                           DiagnosedMinmodMoment(); limits = false)
            rm, m = production_run(c0, UNIT_MASS, a, nsweeps, SlopesScheme(MonotoneLimiter()))
            @test m == m_ref
            @test maximum(abs.(rm .- rm_ref)) < 1e-12

            rm_ref, _ = tm5_dynamu_run(c0, UNIT_MASS, a, nsweeps, ZeroMoment(); limits = false)
            rm, _ = production_run(c0, UNIT_MASS, a, nsweeps, UpwindScheme())
            @test maximum(abs.(rm .- rm_ref)) < 1e-12
        end
    end
end

@testset "SlopesScheme is not TM5's prognostic-moment scheme" begin
    a = fill(0.5, N)
    rm_tm5, m_tm5 = tm5_dynamu_run(GAUSSIAN, UNIT_MASS, a, 2N, PrognosticMoment())
    rm, m = production_run(GAUSSIAN, UNIT_MASS, a, 2N, SlopesScheme())
    c_tm5, c = rm_tm5 ./ m_tm5, rm ./ m
    # One revolution of a Gaussian of half-width 6 cells: TM5 keeps 0.990 of the
    # peak, SlopesScheme 0.892, with an 8.7 times larger L2 error.
    @test maximum(c_tm5) > 0.98
    @test maximum(c) < 0.9
    @test relative_l2(c, GAUSSIAN) > 5 * relative_l2(c_tm5, GAUSSIAN)

    # TM5's clamp |rxm| <= rm refers to zero, so transporting c - 1/2 is not
    # transporting c and subtracting 1/2. SlopesScheme limits against
    # neighbouring values and is offset-equivariant.
    rm_tm5_off, m_tm5_off = tm5_dynamu_run(GAUSSIAN .- 0.5, UNIT_MASS, a, 2N, PrognosticMoment())
    rm_off, m_off = production_run(GAUSSIAN .- 0.5, UNIT_MASS, a, 2N, SlopesScheme())
    @test maximum(abs.(rm_tm5_off ./ m_tm5_off .- (c_tm5 .- 0.5))) > 0.1
    @test maximum(abs.(rm_off ./ m_off .- (c .- 0.5))) < 1e-12
end
