#!/usr/bin/env julia
# Plan 41 P1 — focused regression tests for the per-window RG preprocessor
# contract surface exported from `reduced_gaussian_contracts.jl`. Mirrors
# `test_cs_preprocessor_contract.jl` / `test_ll_preprocessor_contract.jl`
# for the face-indexed reduced-Gaussian topology.
#
# Builds a 4-cell single-ring synthetic RG window whose horizontal face
# fluxes vanish and whose vertical fluxes encode the per-cell mass
# tendency exactly. The result satisfies the write-time replay gate to
# F64 ULP, so the positivity scan can be exercised in isolation through
# `verify_rg_window_contract!` without tripping the replay short-circuit
# first.

using Test
using Logging: with_logger, NullLogger

using AtmosTransport
using .AtmosTransport.Preprocessing: verify_substep_positivity_rg!,
                                       verify_rg_window_contract!,
                                       verify_boundary_stub_flux_rg,
                                       init_rg_positivity_accumulator,
                                       update_rg_positivity_accumulator,
                                       summarize_rg_positivity_status,
                                       ReducedGaussianContract,
                                       AbstractWindowContract,
                                       verify_window!,
                                       update_accumulator!,
                                       summarize_status!

# Build a single-ring RG window (4 cells, 4 ring faces). Each face `f`
# connects cell `f` on its left to cell `mod1(f + 1, 4)` on its right.
# Horizontal fluxes start at zero; vertical `cm` is built so the per-
# substep continuity closes against `m_next = m_cur + 2 * steps * dm`.
function build_clean_rg_window(FT::Type; nc::Int = 4, Nz::Int = 3,
                                 steps::Int = 2,
                                 m_base::Real = 1e9, dm_scale::Real = 1e4)
    face_left  = Int32[1, 2, 3, 4]
    face_right = Int32[2, 3, 4, 1]
    @assert length(face_left) == length(face_right) == nc

    m_cur = fill(FT(m_base), nc, Nz)
    hflux = zeros(FT, length(face_left), Nz)
    cm    = zeros(FT, nc, Nz + 1)
    dm    = Array{FT}(undef, nc, Nz)
    for k in 1:Nz, c in 1:nc
        dm[c, k] = FT(dm_scale) * sinpi(FT(c) / nc) * FT(k / Nz)
    end
    for c in 1:nc
        acc = 0.0
        for k in 1:Nz
            acc -= Float64(dm[c, k])
            cm[c, k + 1] = FT(acc)
        end
    end
    m_next = m_cur .+ FT(2 * steps) .* dm
    return (; m_cur, hflux, cm, m_next, face_left, face_right, steps)
end

# Discard log output from `summarize_rg_positivity_status` during the
# require=false / Inf branches — the tests assert return values, not text.
with_quiet_logger(f) = with_logger(f, NullLogger())

@testset "RG preprocessor contract gates" begin

    # ------------------------------------------------------------------
    # verify_substep_positivity_rg!
    # ------------------------------------------------------------------

    @testset "positivity: clean window passes" begin
        for FT in (Float32, Float64)
            w = build_clean_rg_window(FT)
            diag = verify_substep_positivity_rg!(w.m_cur, w.hflux, w.cm,
                                                  w.face_left, w.face_right;
                                                  cfl_limit = 0.95)
            @test diag.ok
            @test isfinite(diag.ratio)
            @test diag.ratio < 0.95
            @test diag.location isa NTuple{2, Int}
        end
    end

    @testset "positivity: horizontal CFL violation reported with direction and location" begin
        # Face 1 connects cells 1→2. Push positive flux: outflow from cell 1.
        w = build_clean_rg_window(Float64)
        hflux = copy(w.hflux)
        hflux[1, 1] = 0.99e9
        diag = verify_substep_positivity_rg!(w.m_cur, hflux, w.cm,
                                              w.face_left, w.face_right;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.direction === :h
        @test diag.ratio ≈ 0.99 atol = 1e-12
        @test diag.location == (1, 1)
    end

    @testset "positivity: horizontal CFL violation via negative flux on right side" begin
        # Face 4 connects cell 4 → cell 1. Negative flux means outflow
        # from cell 1 (the `right` side of face 4).
        w = build_clean_rg_window(Float64)
        hflux = copy(w.hflux)
        hflux[4, 2] = -0.96e9
        diag = verify_substep_positivity_rg!(w.m_cur, hflux, w.cm,
                                              w.face_left, w.face_right;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.direction === :h
        @test diag.ratio ≈ 0.96 atol = 1e-12
        @test diag.location == (1, 2)
    end

    @testset "positivity: z-face violation is included in the scan" begin
        w = build_clean_rg_window(Float64)
        cm = copy(w.cm)
        cm[1, 3] = 0.98e9                # interface k=3 of cell 1 (level 2)
        diag = verify_substep_positivity_rg!(w.m_cur, w.hflux, cm,
                                              w.face_left, w.face_right;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.direction === :z
        @test diag.ratio ≈ 0.98 atol = 1e-3
        @test diag.location == (1, 2)
    end

    @testset "positivity: m <= 0 is flagged Inf regardless of flux magnitude" begin
        w = build_clean_rg_window(Float64)
        m_cur = copy(w.m_cur)
        m_cur[3, 1] = 0.0
        diag = verify_substep_positivity_rg!(m_cur, w.hflux, w.cm,
                                              w.face_left, w.face_right;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
        @test diag.direction === :h
        @test diag.location == (3, 1)
    end

    @testset "positivity: NaN mass yields Inf ratio without slipping through" begin
        w = build_clean_rg_window(Float64)
        m_cur = copy(w.m_cur)
        m_cur[2, 1] = NaN
        diag = verify_substep_positivity_rg!(m_cur, w.hflux, w.cm,
                                              w.face_left, w.face_right;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
        @test diag.location == (2, 1)
    end

    @testset "positivity: NaN flux yields Inf ratio" begin
        w = build_clean_rg_window(Float64)
        hflux = copy(w.hflux)
        hflux[2, 2] = NaN
        diag = verify_substep_positivity_rg!(w.m_cur, hflux, w.cm,
                                              w.face_left, w.face_right;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
    end

    @testset "positivity: Inf flux yields Inf ratio" begin
        w = build_clean_rg_window(Float64)
        cm = copy(w.cm)
        cm[3, 2] = Inf
        diag = verify_substep_positivity_rg!(w.m_cur, w.hflux, cm,
                                              w.face_left, w.face_right;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
    end

    @testset "positivity: face_left/face_right length mismatch fails loudly" begin
        w = build_clean_rg_window(Float64)
        @test_throws ErrorException verify_substep_positivity_rg!(
            w.m_cur, w.hflux, w.cm, w.face_left[1:3], w.face_right;
            cfl_limit = 0.95)
    end

    # ------------------------------------------------------------------
    # Boundary-stub handling (codex review of `3796526` round-1).
    #
    # Runtime face-indexed advection (`Advection/sweeps.jl`) skips any face
    # where `face_left == 0` OR `face_right == 0`; the cell on the
    # "real" side sees ZERO mass change from that face. So the
    # positivity gate must also skip boundary stubs — otherwise a
    # binary with `(face_left=0, face_right=1, hflux=-0.99e9)` would
    # falsely trip the gate even though the runtime never applies the
    # flux. The codex review caught that the original P1 test only
    # covered the non-outgoing sign and missed the asymmetry.
    # ------------------------------------------------------------------

    @testset "positivity: boundary stub (face_left=0) does NOT count as cell-2 outflow on positive sign" begin
        w = build_clean_rg_window(Float64)
        face_left = copy(w.face_left)
        face_left[1] = 0
        hflux = copy(w.hflux)
        hflux[1, 1] = 0.99e9   # positive flux: would flow toward cell 2
        diag = verify_substep_positivity_rg!(w.m_cur, hflux, w.cm,
                                              face_left, w.face_right;
                                              cfl_limit = 0.95)
        @test diag.ok          # runtime would not apply this flux
    end

    @testset "positivity: boundary stub (face_left=0) does NOT count as cell-2 outflow on NEGATIVE sign (codex round-1 fix)" begin
        # This is the case the original P1 test missed. A negative flux
        # on a face with face_left=0, face_right=2 would imply outflow
        # from cell 2 if we naively followed the sign convention — but
        # the runtime SKIPS this face entirely. The positivity gate
        # must match: no horizontal outflow contribution from cell 2.
        w = build_clean_rg_window(Float64)
        face_left = copy(w.face_left)
        face_left[1] = 0
        hflux = copy(w.hflux)
        hflux[1, 1] = -0.99e9  # negative flux: naively cell-2 outflow
        diag = verify_substep_positivity_rg!(w.m_cur, hflux, w.cm,
                                              face_left, w.face_right;
                                              cfl_limit = 0.95)
        @test diag.ok          # but the runtime skips this face
        @test diag.ratio < 0.95
    end

    @testset "positivity: boundary stub (face_right=0) does NOT count as cell-1 outflow on either sign (codex round-1 fix)" begin
        # Symmetric case for face_right=0 (north-pole singularity).
        w = build_clean_rg_window(Float64)
        face_right = copy(w.face_right)
        face_right[1] = 0
        for sign in (+1.0, -1.0)
            hflux = copy(w.hflux)
            hflux[1, 2] = sign * 0.99e9
            diag = verify_substep_positivity_rg!(w.m_cur, hflux, w.cm,
                                                  w.face_left, face_right;
                                                  cfl_limit = 0.95)
            @test diag.ok
        end
    end

    # ------------------------------------------------------------------
    # `verify_boundary_stub_flux_rg` — explicit-invariant scan for
    # non-zero flux on boundary stubs. Such fluxes are silently
    # discarded by the runtime, so the writer is broken if it emits
    # them. The wrapper errors hard with no `require_*` escape hatch.
    # ------------------------------------------------------------------

    @testset "boundary-stub: zero flux on all stubs is benign" begin
        w = build_clean_rg_window(Float64)
        face_left = copy(w.face_left)
        face_left[1] = 0
        diag = verify_boundary_stub_flux_rg(w.hflux, face_left, w.face_right)
        @test !diag.violated
        @test diag.worst_face == 0
        @test diag.worst_level == 0
    end

    @testset "boundary-stub: non-zero flux is flagged with face + level + value" begin
        w = build_clean_rg_window(Float64)
        face_left = copy(w.face_left)
        face_left[1] = 0
        hflux = copy(w.hflux)
        hflux[1, 2] = -0.5e9
        hflux[1, 3] = +0.3e9        # smaller magnitude — should NOT win
        diag = verify_boundary_stub_flux_rg(hflux, face_left, w.face_right)
        @test diag.violated
        @test diag.worst_face == 1
        @test diag.worst_level == 2
        @test diag.worst_flux ≈ -0.5e9 atol = 1e-12
    end

    @testset "boundary-stub: tol = 1e8 admits |flux| < 1e8" begin
        w = build_clean_rg_window(Float64)
        face_left = copy(w.face_left)
        face_left[1] = 0
        hflux = copy(w.hflux)
        hflux[1, 2] = 5e7           # below tol
        diag = verify_boundary_stub_flux_rg(hflux, face_left, w.face_right;
                                             tol = 1e8)
        @test !diag.violated
    end

    @testset "boundary-stub: NaN/Inf flux is flagged with isinf worst_abs" begin
        w = build_clean_rg_window(Float64)
        face_left = copy(w.face_left)
        face_left[1] = 0
        hflux = copy(w.hflux)
        hflux[1, 1] = NaN
        diag = verify_boundary_stub_flux_rg(hflux, face_left, w.face_right)
        @test diag.violated
        @test isnan(diag.worst_flux) || isinf(diag.worst_flux)
    end

    @testset "boundary-stub: invalid tol is rejected at helper (codex round-2)" begin
        # `verify_boundary_stub_flux_rg` is exported and can be called
        # without going through the `ReducedGaussianContract`
        # constructor. The original P1 code accepted `tol = NaN` /
        # `Inf` / `negative` and silently disabled the gate (`abs(h) > NaN`
        # is always `false`). Helper-level validation closes the
        # bypass.
        w = build_clean_rg_window(Float64)
        face_left = copy(w.face_left)
        face_left[1] = 0
        for bad_tol in (NaN, Inf, -Inf, -1.0)
            @test_throws ErrorException verify_boundary_stub_flux_rg(
                w.hflux, face_left, w.face_right; tol = bad_tol)
        end
    end

    @testset "boundary-stub: invalid boundary_stub_tol is rejected at wrapper (codex round-2)" begin
        w = build_clean_rg_window(Float64)
        for bad_tol in (NaN, Inf, -Inf, -1.0)
            err = try
                verify_rg_window_contract!(w.m_cur, w.hflux, w.cm, w.m_next,
                                            w.face_left, w.face_right,
                                            w.steps, 1;
                                            replay_tol = 1e-12,
                                            positivity_cfl_limit = 0.95,
                                            boundary_stub_tol = bad_tol)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("boundary_stub_tol", err.msg)
        end
    end

    @testset "wrapper: boundary-stub failure errors BEFORE the replay gate (no escape hatch)" begin
        w = build_clean_rg_window(Float64)
        face_left = copy(w.face_left)
        face_left[1] = 0
        hflux = copy(w.hflux)
        hflux[1, 1] = -0.5e9
        # cm closure was built for the un-perturbed window, so replay
        # would also fail — but the wrapper must surface the boundary
        # stub error first because that's the more diagnostic cause.
        err = try
            verify_rg_window_contract!(w.m_cur, hflux, w.cm, w.m_next,
                                        face_left, w.face_right,
                                        w.steps, 1;
                                        replay_tol = 1e-12,
                                        positivity_cfl_limit = 0.95)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("Boundary-stub flux gate FAILED", err.msg)
        @test occursin("Advection/sweeps.jl", err.msg)
    end

    # ------------------------------------------------------------------
    # verify_rg_window_contract!
    # ------------------------------------------------------------------

    @testset "wrapper: clean window returns both diagnostics" begin
        w = build_clean_rg_window(Float64)
        result = verify_rg_window_contract!(w.m_cur, w.hflux, w.cm, w.m_next,
                                             w.face_left, w.face_right,
                                             w.steps, 1;
                                             replay_tol = 1e-12,
                                             positivity_cfl_limit = 0.95)
        @test result.replay.max_rel_err <= 1e-12
        @test result.positivity.ok
    end

    @testset "wrapper: replay failure errors before positivity is reached" begin
        w = build_clean_rg_window(Float64)
        cm_broken = copy(w.cm)
        cm_broken[2, 2] += 1e4
        @test_throws ErrorException verify_rg_window_contract!(
            w.m_cur, w.hflux, cm_broken, w.m_next,
            w.face_left, w.face_right, w.steps, 7;
            replay_tol = 1e-12, positivity_cfl_limit = 0.95,
        )
    end

    @testset "wrapper: positivity failure with passing replay is non-fatal" begin
        # Both endpoints of face 1 (cells 1 and 2) see a uniform horizontal
        # flux at level 1. The per-cell divergence at the perturbed level
        # remains the same (because both cells are touched symmetrically:
        # +hflux on left, -hflux on right contributes zero net to either
        # cell's divergence — wait: this matters for the replay).
        # Simpler: a single ring with all-equal horizontal flux gives
        # zero net divergence because each cell has one inflow and one
        # outflow of equal magnitude.
        w = build_clean_rg_window(Float64)
        hflux = copy(w.hflux)
        hflux[:, 1] .= 0.99e9       # all 4 faces at level 1
        result = verify_rg_window_contract!(w.m_cur, hflux, w.cm, w.m_next,
                                             w.face_left, w.face_right,
                                             w.steps, 3;
                                             replay_tol = 1e-12,
                                             positivity_cfl_limit = 0.95)
        @test result.replay.max_rel_err <= 1e-12
        @test !result.positivity.ok
        @test result.positivity.direction === :h
        @test result.positivity.ratio ≈ 0.99 atol = 1e-12
    end

    # ------------------------------------------------------------------
    # accumulator
    # ------------------------------------------------------------------

    @testset "accumulator: tracks worst window across the loop" begin
        worst = init_rg_positivity_accumulator()
        @test worst.ratio == 0.0
        @test worst.direction === :none
        @test worst.win == 0
        @test worst.location == (0, 0)

        worst = update_rg_positivity_accumulator(worst,
            (direction = :h, ratio = 0.3, location = (1, 2), ok = true), 5)
        @test worst.ratio ≈ 0.3
        @test worst.direction === :h
        @test worst.win == 5
        @test worst.location == (1, 2)

        worst = update_rg_positivity_accumulator(worst,
            (direction = :z, ratio = 0.1, location = (2, 1), ok = true), 6)
        @test worst.ratio ≈ 0.3
        @test worst.win == 5

        worst = update_rg_positivity_accumulator(worst,
            (direction = :z, ratio = Inf, location = (3, 2), ok = false), 7)
        @test isinf(worst.ratio)
        @test worst.direction === :z
        @test worst.win == 7
        @test worst.location == (3, 2)
    end

    @testset "accumulator: direction === nothing is normalized to :none" begin
        worst = update_rg_positivity_accumulator(init_rg_positivity_accumulator(),
            (direction = nothing, ratio = 0.5, location = (1, 1), ok = true), 1)
        @test worst.direction === :none
        @test worst.ratio ≈ 0.5
    end

    # ------------------------------------------------------------------
    # summarize_rg_positivity_status
    # ------------------------------------------------------------------

    @testset "summary: ratio within limit returns nothing" begin
        worst = (ratio = 0.5, direction = :h, win = 1, location = (1, 1))
        @test summarize_rg_positivity_status(worst; cfl_limit = 0.95,
                                              steps_per_window = 8) === nothing
    end

    @testset "summary: finite violation + require=true errors with rescue advice" begin
        worst = (ratio = 1.5, direction = :h, win = 1, location = (1, 1))
        err = try
            summarize_rg_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = true)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("steps_per_window=16", err.msg)
    end

    @testset "summary: finite violation + require=false warns and returns nothing" begin
        worst = (ratio = 1.5, direction = :h, win = 1, location = (1, 1))
        r = with_quiet_logger() do
            summarize_rg_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = false)
        end
        @test r === nothing
    end

    @testset "summary: Inf ratio + require=true throws ErrorException, NOT InexactError (CS round-2 fix)" begin
        worst = (ratio = Inf, direction = :z, win = 4, location = (3, 2))
        err = try
            summarize_rg_positivity_status(worst; cfl_limit = 0.95,
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
        worst = (ratio = Inf, direction = :z, win = 4, location = (3, 2))
        r = with_quiet_logger() do
            summarize_rg_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = false)
        end
        @test r === nothing
    end

    @testset "summary: quarantine_path is deleted on error" begin
        worst = (ratio = 1.5, direction = :h, win = 1, location = (1, 1))
        tmp = tempname() * ".bin"
        write(tmp, b"contents")
        @test isfile(tmp)
        @test_throws ErrorException summarize_rg_positivity_status(
            worst; cfl_limit = 0.95, steps_per_window = 8,
            require_substep_positivity = true, quarantine_path = tmp,
        )
        @test !isfile(tmp)
    end

    @testset "summary: missing quarantine_path is benign" begin
        worst = (ratio = 1.5, direction = :h, win = 1, location = (1, 1))
        missing_path = tempname() * ".bin"
        @test !isfile(missing_path)
        @test_throws ErrorException summarize_rg_positivity_status(
            worst; cfl_limit = 0.95, steps_per_window = 8,
            require_substep_positivity = true, quarantine_path = missing_path,
        )
    end

    # ------------------------------------------------------------------
    # Round-3 boundary cases.
    # ------------------------------------------------------------------

    @testset "summary: finite-but-pathologically-large ratio + require=true (round-3)" begin
        worst = (ratio = 1e308, direction = :z, win = 4, location = (3, 2))
        err = try
            summarize_rg_positivity_status(worst; cfl_limit = 0.95,
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
        worst = (ratio = 1e308, direction = :z, win = 4, location = (3, 2))
        r = with_quiet_logger() do
            summarize_rg_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = false)
        end
        @test r === nothing
    end

    @testset "summary: cfl_limit = 0.0 + require=true (round-3)" begin
        worst = (ratio = 1.5, direction = :h, win = 1, location = (1, 1))
        err = try
            summarize_rg_positivity_status(worst; cfl_limit = 0.0,
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
                 location = (1, 1))
        err = try
            summarize_rg_positivity_status(worst; cfl_limit = 0.95,
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
    # ReducedGaussianContract{FT} — typed Axis-3 concrete (Plan 41 P1).
    # The struct holds policy, face connectivity, and accumulator; trait
    # methods delegate to the NamedTuple-based helpers.
    # ------------------------------------------------------------------

    @testset "ReducedGaussianContract: construction validates policy fields" begin
        face_left  = Int32[1, 2, 3, 4]
        face_right = Int32[2, 3, 4, 1]
        # replay_tol must be finite and > 0 (codex round-1: Inf would
        # silently disable replay; NaN would fail every window late).
        for bad in (Inf, NaN, 0.0, -1e-12, -Inf)
            @test_throws ErrorException ReducedGaussianContract{Float64}(
                replay_tol = bad, positivity_cfl_limit = 0.95,
                steps_per_window = 1,
                face_left = face_left, face_right = face_right)
        end
        @test_throws ErrorException ReducedGaussianContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = 0.0,
            steps_per_window = 1, face_left = face_left, face_right = face_right)
        @test_throws ErrorException ReducedGaussianContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = 1.5,
            steps_per_window = 1, face_left = face_left, face_right = face_right)
        @test_throws ErrorException ReducedGaussianContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = 0.95,
            steps_per_window = 0, face_left = face_left, face_right = face_right)
        @test_throws ErrorException ReducedGaussianContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = 0.95,
            steps_per_window = 1, face_left = face_left[1:3], face_right = face_right)
        # boundary_stub_tol must be finite and ≥ 0 (default is 0.0).
        for bad in (Inf, NaN, -1.0, -Inf)
            @test_throws ErrorException ReducedGaussianContract{Float64}(
                replay_tol = 1e-12, positivity_cfl_limit = 0.95,
                steps_per_window = 1, boundary_stub_tol = bad,
                face_left = face_left, face_right = face_right)
        end
        c = ReducedGaussianContract{Float64}(replay_tol = 1e-12,
                                              positivity_cfl_limit = 0.95,
                                              steps_per_window = 8,
                                              face_left = face_left,
                                              face_right = face_right)
        @test c isa AbstractWindowContract
        @test c.replay_tol == 1e-12
        @test c.positivity_cfl_limit == 0.95
        @test c.require_substep_positivity == true
        @test c.steps_per_window == 8
        @test c.boundary_stub_tol == 0.0
        @test c.face_left == face_left
        @test c.worst.ratio == 0.0
    end

    @testset "ReducedGaussianContract: boundary_stub_tol field threads from struct to wrapper" begin
        # Note on what the boundary-stub gate guards against:
        # Replay's `horizontal_divergence!` IS one-sided
        # (`left > 0 && div_h[left] += flux; right > 0 && div_h[right] -= flux`),
        # while the runtime SKIPS boundary-stub faces entirely. So a
        # non-zero boundary-stub flux silently teaches the
        # writer-side cm closure to "balance" mass that the runtime
        # will never apply — replay passes, runtime drifts. The
        # boundary-stub gate fires BEFORE the replay gate so the
        # writer-bug surface diagnostic is preserved.
        #
        # This test confirms that the contract's `boundary_stub_tol`
        # field actually flows into the wrapper. Under the strict
        # default (0.0) the wrapper errors with the boundary-stub
        # message; loosen it to 1e8 and the message becomes a
        # different error (e.g. replay) — but it must NOT be the
        # boundary-stub message.
        face_left  = Int32[0, 2, 3, 4]
        face_right = Int32[2, 3, 4, 1]
        w = build_clean_rg_window(Float64)
        hflux = copy(w.hflux)
        hflux[1, 1] = 5e7                 # under 1e8 but > 0

        strict = ReducedGaussianContract{Float64}(replay_tol = 1e-12,
                                                   positivity_cfl_limit = 0.95,
                                                   steps_per_window = w.steps,
                                                   face_left = face_left,
                                                   face_right = face_right)
        err_strict = try
            verify_window!((m_cur = w.m_cur, hflux = hflux, cm = w.cm,
                            m_next = w.m_next), strict, 1)
            nothing
        catch e
            e
        end
        @test err_strict isa ErrorException
        @test occursin("Boundary-stub flux gate FAILED", err_strict.msg)

        loose = ReducedGaussianContract{Float64}(replay_tol = 1e-12,
                                                  positivity_cfl_limit = 0.95,
                                                  steps_per_window = w.steps,
                                                  boundary_stub_tol = 1e8,
                                                  face_left = face_left,
                                                  face_right = face_right)
        err_loose = try
            verify_window!((m_cur = w.m_cur, hflux = hflux, cm = w.cm,
                            m_next = w.m_next), loose, 1)
            nothing
        catch e
            e
        end
        # The boundary-stub gate must NOT fire under the relaxed tol;
        # replay still fails downstream because cm wasn't built for
        # the boundary divergence, but THAT is a different error.
        @test err_loose isa ErrorException
        @test !occursin("Boundary-stub flux gate FAILED", err_loose.msg)
        @test occursin("replay gate FAILED", err_loose.msg)
    end

    @testset "verify_rg_window_contract!: scratch kwargs are reused (codex round-1)" begin
        w = build_clean_rg_window(Float64)
        nc, Nz = size(w.m_cur)
        div_scratch = Array{Float64}(undef, nc, Nz)
        outgoing_h  = Array{Float64}(undef, nc, Nz)
        bad_h       = Array{Bool}(undef, nc, Nz)
        # Poison scratch to confirm the helper resets it.
        fill!(div_scratch, NaN)
        fill!(outgoing_h, NaN)
        fill!(bad_h, true)
        result = verify_rg_window_contract!(w.m_cur, w.hflux, w.cm, w.m_next,
                                             w.face_left, w.face_right,
                                             w.steps, 1;
                                             replay_tol = 1e-12,
                                             positivity_cfl_limit = 0.95,
                                             div_scratch = div_scratch,
                                             outgoing_h = outgoing_h,
                                             bad_h = bad_h)
        @test result.replay.max_rel_err <= 1e-12
        @test result.positivity.ok
        # Shape-mismatch scratch must fail loudly.
        @test_throws ErrorException verify_rg_window_contract!(
            w.m_cur, w.hflux, w.cm, w.m_next,
            w.face_left, w.face_right, w.steps, 1;
            replay_tol = 1e-12,
            div_scratch = zeros(Float64, nc + 1, Nz))
    end

    @testset "ReducedGaussianContract: verify_window! returns the same diagnostics" begin
        w = build_clean_rg_window(Float64)
        contract = ReducedGaussianContract{Float64}(replay_tol = 1e-12,
                                                     positivity_cfl_limit = 0.95,
                                                     steps_per_window = w.steps,
                                                     face_left = w.face_left,
                                                     face_right = w.face_right)
        result = verify_window!((m_cur = w.m_cur, hflux = w.hflux, cm = w.cm,
                                 m_next = w.m_next), contract, 1)
        @test result.replay.max_rel_err <= 1e-12
        @test result.positivity.ok
    end

    # ------------------------------------------------------------------
    # Direct-wrapper policy validation (codex review round 3 of f5224a6).
    # Same bypass pattern as CS / LL.
    # ------------------------------------------------------------------

    @testset "direct wrapper: invalid replay_tol is rejected at verify_rg_window_contract!" begin
        w = build_clean_rg_window(Float64)
        for bad in (Inf, NaN, 0.0, -1e-12, -Inf)
            err = try
                verify_rg_window_contract!(w.m_cur, w.hflux, w.cm, w.m_next,
                                            w.face_left, w.face_right,
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

    @testset "direct wrapper: invalid positivity_cfl_limit is rejected at verify_rg_window_contract!" begin
        w = build_clean_rg_window(Float64)
        for bad in (Inf, NaN, 0.0, -0.1, 1.5)
            err = try
                verify_rg_window_contract!(w.m_cur, w.hflux, w.cm, w.m_next,
                                            w.face_left, w.face_right,
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

    @testset "direct wrapper: invalid cfl_limit is rejected at verify_substep_positivity_rg!" begin
        w = build_clean_rg_window(Float64)
        for bad in (Inf, NaN, 0.0, -0.1, 1.5)
            err = try
                verify_substep_positivity_rg!(w.m_cur, w.hflux, w.cm,
                                                w.face_left, w.face_right;
                                                cfl_limit = bad)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("cfl_limit", err.msg)
        end
    end

    @testset "ReducedGaussianContract: scratch is lazily allocated once and reused (codex round-2)" begin
        # P2 watchpoint closed: the trait `verify_window!` allocates
        # scratch on the first call (when the window shape becomes
        # known) and stores it on the contract; subsequent calls reuse
        # the same buffers. We verify by capturing the reference after
        # call 1 and asserting identity after call 2.
        w = build_clean_rg_window(Float64)
        contract = ReducedGaussianContract{Float64}(replay_tol = 1e-12,
                                                     positivity_cfl_limit = 0.95,
                                                     steps_per_window = w.steps,
                                                     face_left = w.face_left,
                                                     face_right = w.face_right)
        @test contract._div_scratch === nothing
        @test contract._outgoing_h  === nothing
        @test contract._bad_h       === nothing
        verify_window!((m_cur = w.m_cur, hflux = w.hflux, cm = w.cm,
                        m_next = w.m_next), contract, 1)
        @test contract._div_scratch isa Matrix{Float64}
        @test contract._outgoing_h  isa Matrix{Float64}
        @test contract._bad_h       isa Matrix{Bool}
        @test size(contract._div_scratch) == size(w.m_cur)
        ds = contract._div_scratch
        oh = contract._outgoing_h
        bh = contract._bad_h
        verify_window!((m_cur = w.m_cur, hflux = w.hflux, cm = w.cm,
                        m_next = w.m_next), contract, 2)
        @test contract._div_scratch === ds
        @test contract._outgoing_h  === oh
        @test contract._bad_h       === bh
    end

    @testset "ReducedGaussianContract: update_accumulator!/summarize_status! lifecycle" begin
        face_left  = Int32[1, 2, 3, 4]
        face_right = Int32[2, 3, 4, 1]
        contract = ReducedGaussianContract{Float64}(replay_tol = 1e-12,
                                                     positivity_cfl_limit = 0.95,
                                                     steps_per_window = 8,
                                                     face_left = face_left,
                                                     face_right = face_right)
        update_accumulator!(contract,
            (direction = :h, ratio = 0.5, location = (1, 1), ok = true), 2)
        @test contract.worst.ratio ≈ 0.5
        @test contract.worst.win == 2
        @test summarize_status!(contract) === nothing
        update_accumulator!(contract,
            (direction = :z, ratio = 1.5, location = (1, 1), ok = false), 3)
        @test contract.worst.ratio ≈ 1.5
        @test_throws ErrorException summarize_status!(contract)
        contract.require_substep_positivity = false
        @test with_quiet_logger() do
            summarize_status!(contract)
        end === nothing
    end
end
