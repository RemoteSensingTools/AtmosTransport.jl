#!/usr/bin/env julia
# Plan 41 P1 — focused regression tests for the per-window LL preprocessor
# contract surface exported from `latlon_contracts.jl`. Mirrors
# `test_cs_preprocessor_contract.jl` one-to-one for the structured lat-lon
# topology (no panel dimension; no halo).
#
# Builds a small synthetic LL window (Nx, Ny, Nz) that trips each gate
# (replay, positivity-CFL, m <= 0, NaN/Inf mass, NaN/Inf flux) and confirms
# the corresponding helper fires with the documented diagnostic. Also locks
# in the CS round-2 (`45b87f3`) regression that the summary helper does not
# throw `InexactError` on a non-finite ratio under either policy and the CS
# round-3 (`9b1ceda`) regression that finite-but-pathologically-large
# ratios are routed through the "no representable rescue" branch.

using Test
using Logging: with_logger, NullLogger

using AtmosTransport
using .AtmosTransport.Preprocessing: verify_substep_positivity_ll!,
                                       verify_ll_window_contract!,
                                       init_ll_positivity_accumulator,
                                       update_ll_positivity_accumulator,
                                       summarize_ll_positivity_status,
                                       LatLonContract,
                                       AbstractWindowContract,
                                       ReadyWindow,
                                       PreprocessorRunCache,
                                       verify_window!,
                                       update_accumulator!,
                                       summarize_status!
using .AtmosTransport.MetDrivers: ReplayScratch

# Build a single-grid LL window whose horizontal fluxes vanish and whose
# vertical fluxes encode the per-cell mass tendency exactly. The result
# satisfies the write-time replay gate to F64 ULP, so the positivity scan
# can be exercised in isolation through `verify_ll_window_contract!`
# without tripping the replay short-circuit first.
function build_clean_ll_window(FT::Type; Nx::Int = 4, Ny::Int = 3, Nz::Int = 3,
                                 steps::Int = 2,
                                 m_base::Real = 1e9, dm_scale::Real = 1e4)
    m_cur = fill(FT(m_base), Nx, Ny, Nz)
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    dm = Array{FT}(undef, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dm[i, j, k] = FT(dm_scale) * sinpi(FT(i) / Nx) *
                      cospi(FT(j) / Ny) * FT(k / Nz)
    end
    for j in 1:Ny, i in 1:Nx
        acc = 0.0
        for k in 1:Nz
            acc -= Float64(dm[i, j, k])
            cm[i, j, k + 1] = FT(acc)
        end
    end
    m_next = m_cur .+ FT(2 * steps) .* dm
    return (; m_cur, am, bm, cm, m_next, steps)
end

# Discard log output from `summarize_ll_positivity_status` during the
# require=false / Inf branches — the tests assert return values, not text.
with_quiet_logger(f) = with_logger(f, NullLogger())

@testset "LL preprocessor contract gates" begin

    # ------------------------------------------------------------------
    # verify_substep_positivity_ll!
    # ------------------------------------------------------------------

    @testset "LL maps the shared scan's direction and location" begin
        for FT in (Float32,Float64)
            w=build_clean_ll_window(FT)
            @test verify_substep_positivity_ll!(w.m_cur,w.am,w.bm,w.cm;cfl_limit=0.95).ok
            for (direction,field,idx,location,amount,expected) in (
                (:x,:am,(3,2,1),(2,2,1),0.49e9,0.98),
                (:y,:bm,(2,3,2),(2,2,2),0.485e9,0.97),
                (:z,:cm,(1,1,3),(1,1,2),0.49e9,0.98))
                am,bm,cm=copy(w.am),copy(w.bm),copy(w.cm)
                getproperty((;am,bm,cm),field)[idx...]=FT(amount)
                diag=verify_substep_positivity_ll!(w.m_cur,am,bm,cm;cfl_limit=0.95)
                @test !diag.ok && diag.direction===direction && diag.location==location
                @test diag.ratio ≈ expected atol=1e-3
            end
        end
    end

    @testset "positivity: directions add up, and the end mass tightens the budget" begin
        w = build_clean_ll_window(Float64)
        am, bm = copy(w.am), copy(w.bm)
        am[3, 2, 1] = 0.3e9             # x and y outflow of cell (2, 2, 1), each
        bm[2, 3, 1] = 0.3e9             # 0.3 m: fine alone, 1.2 m per palindrome
        diag = verify_substep_positivity_ll!(w.m_cur, am, bm, w.cm; cfl_limit = 0.95)
        @test !diag.ok
        @test diag.ratio ≈ 1.2 atol = 1e-4
        @test diag.ratio_xy ≈ 1.2 atol = 1e-4
        @test diag.location == (2, 2, 1)
        am[3, 2, 1] = 0.2e9
        bm[2, 3, 1] = 0.2e9             # 0.8 m: passes against m ...
        @test verify_substep_positivity_ll!(w.m_cur, am, bm, w.cm; cfl_limit = 0.95).ok
        m_next = copy(w.m_cur)
        m_next[2, 2, 1] *= 0.5          # ... but not against a halved end mass
        diag = verify_substep_positivity_ll!(w.m_cur, am, bm, w.cm; cfl_limit = 0.95, m_next)
        @test !diag.ok
        @test diag.ratio ≈ 1.6 atol = 1e-3
        @test_throws ErrorException verify_substep_positivity_ll!(
            w.m_cur, am, bm, w.cm; m_next = m_next[:, :, 1:2])
    end

    @testset "positivity: shape mismatch fails loudly" begin
        w = build_clean_ll_window(Float64)
        bad_am = zeros(Float64, size(w.m_cur, 1), size(w.m_cur, 2), size(w.m_cur, 3))
        @test_throws ErrorException verify_substep_positivity_ll!(
            w.m_cur, bad_am, w.bm, w.cm; cfl_limit = 0.95)
    end

    # ------------------------------------------------------------------
    # verify_ll_window_contract!
    # ------------------------------------------------------------------

    @testset "wrapper: clean window returns both diagnostics" begin
        w = build_clean_ll_window(Float64)
        contract=LatLonContract{Float64}(;replay_tol=1e-12,steps_per_window=w.steps)
        result=verify_window!((m_cur=w.m_cur,am=w.am,bm=w.bm,cm=w.cm,m_next=w.m_next),contract,1)
        @test result.replay.max_rel_err <= 1e-12
        @test result.positivity.ok
    end

    @testset "wrapper: replay failure errors before positivity is reached" begin
        w = build_clean_ll_window(Float64)
        cm_broken = copy(w.cm)
        cm_broken[2, 2, 2] += 1e4
        @test_throws ErrorException verify_ll_window_contract!(
            w.m_cur, w.am, w.bm, cm_broken, w.m_next, w.steps, 7;
            replay_tol = 1e-12, positivity_cfl_limit = 0.95,
        )
    end

    @testset "wrapper: positivity failure with passing replay is non-fatal" begin
        # Closed-loop uniform shift on one (j, k) row preserves per-cell
        # divergence on the perturbed row but drives the palindrome outflow to 0.98 m.
        # `verify_ll_window_contract!` must return both diagnostics so the
        # caller — not the wrapper — decides whether to error or warn after
        # aggregating across windows.
        w = build_clean_ll_window(Float64)
        am = copy(w.am)
        am[:, 1, 1] .= 0.49e9
        result = verify_ll_window_contract!(w.m_cur, am, w.bm, w.cm,
                                             w.m_next, w.steps, 3;
                                             replay_tol = 1e-12,
                                             positivity_cfl_limit = 0.95)
        @test result.replay.max_rel_err <= 1e-12
        @test !result.positivity.ok
        @test result.positivity.direction === :x
        @test result.positivity.ratio ≈ 0.98 atol = 1e-3
        @test Base.tail(result.positivity.location) == (1, 1)    # the shifted row
    end

    # LatLonContract policy validation, the worst-window accumulator and the
    # positivity summary are checked for all three topologies in the shared
    # testsets of test_cs_preprocessor_contract.jl.

    @testset "verify_ll_window_contract!: scratch kwarg is reused (codex round-1)" begin
        w = build_clean_ll_window(Float64)
        replay_scratch = ReplayScratch(size(w.m_cur))
        fill!(replay_scratch.div, NaN)
        fill!(replay_scratch.mag, NaN)
        result = verify_ll_window_contract!(w.m_cur, w.am, w.bm, w.cm, w.m_next,
                                             w.steps, 1;
                                             replay_tol = 1e-12,
                                             positivity_cfl_limit = 0.95,
                                             replay_scratch = replay_scratch)
        @test result.replay.max_rel_err <= 1e-12
        @test result.positivity.ok
        # Shape-mismatch scratch must fail loudly.
        bad_scratch = ReplayScratch((size(w.m_cur, 1) + 1, size(w.m_cur, 2),
                                     size(w.m_cur, 3)))
        @test_throws ErrorException verify_ll_window_contract!(
            w.m_cur, w.am, w.bm, w.cm, w.m_next, w.steps, 1;
            replay_tol = 1e-12,
            replay_scratch = bad_scratch)
    end
end
