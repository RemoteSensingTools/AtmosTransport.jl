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

    @testset "positivity: clean window passes" begin
        for FT in (Float32, Float64)
            w = build_clean_ll_window(FT)
            diag = verify_substep_positivity_ll!(w.m_cur, w.am, w.bm, w.cm;
                                                  cfl_limit = 0.95)
            @test diag.ok
            @test isfinite(diag.ratio)
            @test diag.ratio < 0.95
            @test diag.location isa NTuple{3, Int}
        end
    end

    # The palindrome budget: each direction acts twice per substep, so one
    # application's outflow counts twice (the clean window's `cm` adds < 1e-4).
    @testset "positivity: x-face CFL violation reported with direction and location" begin
        w = build_clean_ll_window(Float64)
        am = copy(w.am)
        am[3, 2, 1] = 0.49e9            # right face of cell (2, 2, 1)
        diag = verify_substep_positivity_ll!(w.m_cur, am, w.bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.direction === :x
        @test diag.ratio ≈ 0.98 atol = 1e-4
        @test diag.location == (2, 2, 1)
    end

    @testset "positivity: y-face CFL violation reported with direction and location" begin
        w = build_clean_ll_window(Float64)
        bm = copy(w.bm)
        bm[2, 3, 2] = 0.485e9           # top face of cell (2, 2, 2)
        diag = verify_substep_positivity_ll!(w.m_cur, w.am, bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.direction === :y
        @test diag.ratio ≈ 0.97 atol = 1e-4
        @test diag.location == (2, 2, 2)
    end

    @testset "positivity: z-face violation is included in the scan" begin
        w = build_clean_ll_window(Float64)
        cm = copy(w.cm)
        cm[1, 1, 3] = 0.49e9            # interface k=3 of cell (1, 1, 2)
        diag = verify_substep_positivity_ll!(w.m_cur, w.am, w.bm, cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.direction === :z
        @test diag.ratio ≈ 0.98 atol = 1e-3
        @test diag.location == (1, 1, 2)
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

    @testset "positivity: m <= 0 is flagged Inf regardless of flux magnitude" begin
        w = build_clean_ll_window(Float64)
        m_cur = copy(w.m_cur)
        m_cur[2, 3, 1] = 0.0
        diag = verify_substep_positivity_ll!(m_cur, w.am, w.bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
        @test diag.direction === :x
        @test diag.location == (2, 3, 1)
    end

    @testset "positivity: NaN mass yields Inf ratio without slipping through" begin
        w = build_clean_ll_window(Float64)
        m_cur = copy(w.m_cur)
        m_cur[1, 1, 1] = NaN
        diag = verify_substep_positivity_ll!(m_cur, w.am, w.bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
        @test diag.location == (1, 1, 1)
    end

    @testset "positivity: NaN flux yields Inf ratio" begin
        w = build_clean_ll_window(Float64)
        am = copy(w.am)
        am[3, 3, 2] = NaN
        diag = verify_substep_positivity_ll!(w.m_cur, am, w.bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
    end

    @testset "positivity: Inf flux yields Inf ratio" begin
        w = build_clean_ll_window(Float64)
        cm = copy(w.cm)
        cm[2, 2, 2] = Inf
        diag = verify_substep_positivity_ll!(w.m_cur, w.am, w.bm, cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
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
        result = verify_ll_window_contract!(w.m_cur, w.am, w.bm, w.cm,
                                             w.m_next, w.steps, 1;
                                             replay_tol = 1e-12,
                                             positivity_cfl_limit = 0.95)
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

    # ------------------------------------------------------------------
    # accumulator
    # ------------------------------------------------------------------

    @testset "accumulator: tracks worst window across the loop" begin
        worst = init_ll_positivity_accumulator()
        @test worst.ratio == 0.0
        @test worst.direction === :none
        @test worst.win == 0
        @test worst.location == (0, 0, 0)

        worst = update_ll_positivity_accumulator(worst,
            (direction = :x, ratio = 0.3, location = (1, 2, 3), ok = true), 5)
        @test worst.ratio ≈ 0.3
        @test worst.direction === :x
        @test worst.win == 5
        @test worst.location == (1, 2, 3)

        worst = update_ll_positivity_accumulator(worst,
            (direction = :y, ratio = 0.1, location = (2, 1, 1), ok = true), 6)
        @test worst.ratio ≈ 0.3
        @test worst.win == 5

        worst = update_ll_positivity_accumulator(worst,
            (direction = :z, ratio = Inf, location = (3, 2, 2), ok = false), 7)
        @test isinf(worst.ratio)
        @test worst.direction === :z
        @test worst.win == 7
        @test worst.location == (3, 2, 2)
    end

    @testset "accumulator: direction === nothing is normalized to :none" begin
        worst = update_ll_positivity_accumulator(init_ll_positivity_accumulator(),
            (direction = nothing, ratio = 0.5, location = (1, 1, 1), ok = true), 1)
        @test worst.direction === :none
        @test worst.ratio ≈ 0.5
    end

    # ------------------------------------------------------------------
    # summarize_ll_positivity_status
    # ------------------------------------------------------------------

    @testset "summary: ratio within limit returns nothing" begin
        worst = (ratio = 0.5, direction = :x, win = 1, location = (1, 1, 1))
        @test summarize_ll_positivity_status(worst; cfl_limit = 0.95,
                                              steps_per_window = 8) === nothing
    end

    @testset "summary: finite violation + require=true errors with rescue advice" begin
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1))
        err = try
            summarize_ll_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = true)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        # The error message must include the recommended steps_per_window
        # (ceil(1.5 / 0.95) * 8 = 16) so the operator can act on it.
        @test occursin("steps_per_window=16", err.msg)
    end

    @testset "summary: finite violation + require=false warns and returns nothing" begin
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1))
        r = with_quiet_logger() do
            summarize_ll_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = false)
        end
        @test r === nothing
    end

    @testset "summary: Inf ratio + require=true throws ErrorException, NOT InexactError (CS round-2 fix)" begin
        worst = (ratio = Inf, direction = :z, win = 4, location = (2, 3, 2))
        err = try
            summarize_ll_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = true)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test !(err isa InexactError)
        @test occursin("no representable", err.msg)
    end

    @testset "summary: Inf ratio + require=false warns (no InexactError)" begin
        worst = (ratio = Inf, direction = :z, win = 4, location = (2, 3, 2))
        r = with_quiet_logger() do
            summarize_ll_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = false)
        end
        @test r === nothing
    end

    @testset "summary: quarantine_path is deleted on error" begin
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1))
        tmp = tempname() * ".bin"
        write(tmp, b"contents")
        @test isfile(tmp)
        @test_throws ErrorException summarize_ll_positivity_status(
            worst; cfl_limit = 0.95, steps_per_window = 8,
            require_substep_positivity = true, quarantine_path = tmp,
        )
        @test !isfile(tmp)
    end

    @testset "summary: missing quarantine_path is benign" begin
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1))
        missing_path = tempname() * ".bin"
        @test !isfile(missing_path)
        @test_throws ErrorException summarize_ll_positivity_status(
            worst; cfl_limit = 0.95, steps_per_window = 8,
            require_substep_positivity = true, quarantine_path = missing_path,
        )
    end

    # ------------------------------------------------------------------
    # Round-3 boundary cases: finite-but-pathologically-large ratios and
    # invalid `cfl_limit` values must route through the "no representable
    # rescue" branch rather than throwing `InexactError`.
    # ------------------------------------------------------------------

    @testset "summary: finite-but-pathologically-large ratio + require=true (round-3)" begin
        worst = (ratio = 1e308, direction = :z, win = 4, location = (2, 3, 2))
        err = try
            summarize_ll_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = true)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test !(err isa InexactError)
        @test occursin("no representable", err.msg)
    end

    @testset "summary: finite-but-pathologically-large ratio + require=false (round-3)" begin
        worst = (ratio = 1e308, direction = :z, win = 4, location = (2, 3, 2))
        r = with_quiet_logger() do
            summarize_ll_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = false)
        end
        @test r === nothing
    end

    @testset "summary: cfl_limit = 0.0 + require=true (round-3)" begin
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1))
        err = try
            summarize_ll_positivity_status(worst; cfl_limit = 0.0,
                                            steps_per_window = 8,
                                            require_substep_positivity = true)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test !(err isa InexactError)
        @test occursin("no representable", err.msg)
    end

    @testset "summary: boundary ratio_factor at typemax(Int) ÷ steps_per_window (round-3)" begin
        max_factor = typemax(Int) ÷ 8
        worst = (ratio = Float64(max_factor) * 0.95, direction = :z, win = 1,
                 location = (1, 1, 1))
        err = try
            summarize_ll_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = true)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test !(err isa InexactError)
    end

    # ------------------------------------------------------------------
    # LatLonContract{FT} — typed Axis-3 concrete (Plan 41 P1).
    # The struct holds policy and accumulator; trait methods delegate to
    # the underlying NamedTuple-based helpers. These tests pin the typed
    # dispatch surface so a future P2 refactor of the orchestrator can
    # rely on `verify_window!(window, contract, w)` returning the same
    # diagnostics as `verify_ll_window_contract!` did directly.
    # ------------------------------------------------------------------

    @testset "LatLonContract: construction validates policy fields" begin
        # replay_tol must be finite and > 0 (codex round-1: Inf would
        # silently disable replay; NaN would fail every window late).
        for bad in (Inf, NaN, 0.0, -1e-12, -Inf)
            @test_throws ErrorException LatLonContract{Float64}(
                replay_tol = bad, positivity_cfl_limit = 0.95,
                steps_per_window = 1)
        end
        @test_throws ErrorException LatLonContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = 0.0, steps_per_window = 1)
        @test_throws ErrorException LatLonContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = 1.5, steps_per_window = 1)
        @test_throws ErrorException LatLonContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = NaN, steps_per_window = 1)
        @test_throws ErrorException LatLonContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = 0.95, steps_per_window = 0)
        c = LatLonContract{Float64}(replay_tol = 1e-12,
                                     positivity_cfl_limit = 0.95,
                                     steps_per_window = 8)
        @test c isa AbstractWindowContract
        @test c.replay_tol == 1e-12
        @test c.positivity_cfl_limit == 0.95
        @test c.require_substep_positivity == true
        @test c.steps_per_window == 8
        @test c.worst.ratio == 0.0
    end

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

    @testset "LatLonContract: verify_window! returns the same diagnostics" begin
        w = build_clean_ll_window(Float64)
        contract = LatLonContract{Float64}(replay_tol = 1e-12,
                                            positivity_cfl_limit = 0.95,
                                            steps_per_window = w.steps)
        result = verify_window!((m_cur = w.m_cur, am = w.am, bm = w.bm,
                                 cm = w.cm, m_next = w.m_next), contract, 1)
        @test result.replay.max_rel_err <= 1e-12
        @test result.positivity.ok
    end

    @testset "P2b skeleton: ReadyWindow forwards payload and cache is typed" begin
        w = build_clean_ll_window(Float64)
        payload = (m_cur = w.m_cur, am = w.am, bm = w.bm,
                   cm = w.cm, m_next = w.m_next)
        ready = ReadyWindow{AtmosTransport.Preprocessing.AbstractTargetGeometry,
                            Float64}(3, payload)
        @test ready.index == 3
        @test ready.payload === payload
        @test ready.m_cur === w.m_cur
        @test :m_cur in propertynames(ready)
        @test !(:payload in propertynames(ready))
        @test :payload in propertynames(ready, true)

        contract = LatLonContract{Float64}(replay_tol = 1e-12,
                                            positivity_cfl_limit = 0.95,
                                            steps_per_window = w.steps)
        result = verify_window!(ready, contract, ready.index)
        @test result.replay.max_rel_err <= 1e-12
        @test result.positivity.ok

        cache = PreprocessorRunCache{
            AtmosTransport.Preprocessing.AbstractTargetGeometry, Float64}()
        @test !haskey(cache, :compressed_laplacian)
        cache[:compressed_laplacian] = :built_once
        @test haskey(cache, :compressed_laplacian)
        @test cache[:compressed_laplacian] === :built_once
        @test get(cache, :missing, :fallback) === :fallback
    end

    # ------------------------------------------------------------------
    # Direct-wrapper policy validation (codex review round 3 of f5224a6).
    # Same bypass pattern as CS — `verify_ll_window_contract!` /
    # `verify_substep_positivity_ll!` must validate even when callers
    # bypass `LatLonContract`.
    # ------------------------------------------------------------------

    @testset "direct wrapper: invalid replay_tol is rejected at verify_ll_window_contract!" begin
        w = build_clean_ll_window(Float64)
        for bad in (Inf, NaN, 0.0, -1e-12, -Inf)
            err = try
                verify_ll_window_contract!(w.m_cur, w.am, w.bm, w.cm, w.m_next,
                                            w.steps, 1;
                                            replay_tol = bad,
                                            positivity_cfl_limit = 0.95)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("replay_tol", err.msg)
        end
    end

    @testset "direct wrapper: invalid positivity_cfl_limit is rejected at verify_ll_window_contract!" begin
        w = build_clean_ll_window(Float64)
        for bad in (Inf, NaN, 0.0, -0.1, 1.5)
            err = try
                verify_ll_window_contract!(w.m_cur, w.am, w.bm, w.cm, w.m_next,
                                            w.steps, 1;
                                            replay_tol = 1e-12,
                                            positivity_cfl_limit = bad)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("cfl_limit", err.msg)
        end
    end

    @testset "direct wrapper: invalid cfl_limit is rejected at verify_substep_positivity_ll!" begin
        w = build_clean_ll_window(Float64)
        for bad in (Inf, NaN, 0.0, -0.1, 1.5)
            err = try
                verify_substep_positivity_ll!(w.m_cur, w.am, w.bm, w.cm;
                                                cfl_limit = bad)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("cfl_limit", err.msg)
        end
    end

    @testset "LatLonContract: scratch is lazily allocated once and reused (codex round-2)" begin
        w = build_clean_ll_window(Float64)
        contract = LatLonContract{Float64}(replay_tol = 1e-12,
                                            positivity_cfl_limit = 0.95,
                                            steps_per_window = w.steps)
        @test contract._replay_scratch === nothing
        verify_window!((m_cur = w.m_cur, am = w.am, bm = w.bm,
                        cm = w.cm, m_next = w.m_next), contract, 1)
        @test contract._replay_scratch isa ReplayScratch{Array{Float64, 3}}
        @test size(contract._replay_scratch) == size(w.m_cur)
        ds = contract._replay_scratch
        verify_window!((m_cur = w.m_cur, am = w.am, bm = w.bm,
                        cm = w.cm, m_next = w.m_next), contract, 2)
        @test contract._replay_scratch === ds
    end

    @testset "LatLonContract: update_accumulator!/summarize_status! lifecycle" begin
        contract = LatLonContract{Float64}(replay_tol = 1e-12,
                                            positivity_cfl_limit = 0.95,
                                            steps_per_window = 8)
        # Push a sub-limit diagnostic; summary should pass without error.
        update_accumulator!(contract,
            (direction = :x, ratio = 0.5, location = (1, 1, 1), ok = true), 2)
        @test contract.worst.ratio ≈ 0.5
        @test contract.worst.win == 2
        @test summarize_status!(contract) === nothing
        # Push a violating diagnostic; require=true → error with rescue
        # advice; require=false → warn.
        update_accumulator!(contract,
            (direction = :y, ratio = 1.5, location = (1, 1, 1), ok = false), 3)
        @test contract.worst.ratio ≈ 1.5
        @test contract.worst.win == 3
        @test_throws ErrorException summarize_status!(contract)
        contract.require_substep_positivity = false
        @test with_quiet_logger() do
            summarize_status!(contract)
        end === nothing
    end
end
