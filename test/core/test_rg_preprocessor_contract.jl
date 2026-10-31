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
using .AtmosTransport.MetDrivers: ReplayScratch

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
    end

    # ------------------------------------------------------------------
    # verify_rg_window_contract!
    # ------------------------------------------------------------------

    @testset "wrapper: clean window returns both diagnostics" begin
        w = build_clean_rg_window(Float64)
        contract=ReducedGaussianContract{Float64}(;replay_tol=1e-12,steps_per_window=w.steps,
            face_left=w.face_left,face_right=w.face_right)
        result=verify_window!((m_cur=w.m_cur,hflux=w.hflux,cm=w.cm,m_next=w.m_next),contract,1)
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
    # ReducedGaussianContract{FT} — typed Axis-3 concrete (Plan 41 P1).
    # The struct holds policy, face connectivity, and accumulator; trait
    # methods delegate to the NamedTuple-based helpers.
    # Policy validation, the accumulator and the positivity summary are
    # checked for all three topologies in test_cs_preprocessor_contract.jl.
    # ------------------------------------------------------------------

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
    end

    @testset "verify_rg_window_contract!: scratch kwargs are reused (codex round-1)" begin
        w = build_clean_rg_window(Float64)
        nc, Nz = size(w.m_cur)
        replay_scratch = ReplayScratch((nc, Nz))
        outgoing_h  = Array{Float64}(undef, nc, Nz)
        bad_h       = Array{Bool}(undef, nc, Nz)
        # Poison scratch to confirm the helper resets it.
        fill!(replay_scratch.div, NaN)
        fill!(replay_scratch.mag, NaN)
        fill!(outgoing_h, NaN)
        fill!(bad_h, true)
        result = verify_rg_window_contract!(w.m_cur, w.hflux, w.cm, w.m_next,
                                             w.face_left, w.face_right,
                                             w.steps, 1;
                                             replay_tol = 1e-12,
                                             positivity_cfl_limit = 0.95,
                                             replay_scratch = replay_scratch,
                                             outgoing_h = outgoing_h,
                                             bad_h = bad_h)
        @test result.replay.max_rel_err <= 1e-12
        @test result.positivity.ok
        # Shape-mismatch scratch must fail loudly.
        @test_throws ErrorException verify_rg_window_contract!(
            w.m_cur, w.hflux, w.cm, w.m_next,
            w.face_left, w.face_right, w.steps, 1;
            replay_tol = 1e-12,
            replay_scratch = ReplayScratch((nc + 1, Nz)))
    end
end
