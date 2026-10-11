#!/usr/bin/env julia
# Plan 39 Commit H — regression tests for the write-time (Commit E) and
# load-time (Commit F) replay-consistency gates.
#
# These tests would have caught the dry-basis Δb×pit cm closure bug that
# motivated the plan-39 fix (F64 probe day-boundary mismatch 7.491e-3).
# After the fix, synthetic binaries that satisfy continuity pass the gate
# to F64 ULP (≤ 1e-12); synthetic binaries with a deliberate cm corruption
# fire the gate with a clear diagnostic.

using Test

using AtmosTransport
using .AtmosTransport.Preprocessing: verify_window_continuity_ll,
                                       verify_window_continuity_rg,
                                       verify_window_continuity_cs,
                                       verify_storage_continuity_ll!,
                                       allocate_window_storage
using .AtmosTransport.MetDrivers: replay_tolerance, ReplayScratch
using .AtmosTransport.Parameters: REPLAY_TOLERANCE_FLOAT64, REPLAY_TOLERANCE_FLOAT32

# ----------------------------------------------------------------------------
# Helper: build a (m, am, bm, cm) tuple that satisfies continuity exactly
# for a tiny LL grid, with a user-specified per-cell dm_target and hand-picked
# horizontal fluxes. Returns (m_cur, am, bm, cm, m_next).
# ----------------------------------------------------------------------------
function build_continuity_consistent_window_ll(FT::Type; Nx::Int=4, Ny::Int=3, Nz::Int=3,
                                                 steps::Int=2,
                                                 m_base::Real=1e9,
                                                 dm_scale::Real=1e5)
    m_cur = fill(FT(m_base), Nx, Ny, Nz)
    # Synthesize a smooth dm_target per cell — the target per-application mass
    # tendency that (am, bm, cm) must integrate to after 2*steps applications.
    dm_target = Array{FT}(undef, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dm_target[i, j, k] = FT(dm_scale) * sinpi(FT(i) / Nx) * cospi(FT(j) / Ny) * FT(k / Nz)
    end

    # Use divergence-free horizontal fluxes: am = 0, bm = 0. Then cm MUST
    # encode the entire vertical divergence to close continuity:
    #     (cm[k+1] - cm[k]) = -dm_target[k]     per application
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    for j in 1:Ny, i in 1:Nx
        acc = 0.0
        for k in 1:Nz
            acc -= Float64(dm_target[i, j, k])
            cm[i, j, k + 1] = FT(acc)
        end
    end

    # m_next implied by continuity after 2*steps applications.
    m_next = similar(m_cur)
    two_steps = FT(2 * steps)
    @. m_next = m_cur + two_steps * dm_target
    return (; m_cur, am, bm, cm, m_next, dm_target, steps)
end

function build_continuity_consistent_window_cs(FT::Type; Nc::Int=4, Nz::Int=3,
                                               steps::Int=2,
                                               m_base::Real=1e9,
                                               dm_scale::Real=1e5)
    panels = ntuple(6) do p
        build_continuity_consistent_window_ll(
            FT; Nx=Nc, Ny=Nc, Nz=Nz, steps=steps,
            m_base=m_base, dm_scale=dm_scale * (1 + 0.05 * p))
    end
    return (
        m_cur = ntuple(p -> panels[p].m_cur, 6),
        am = ntuple(p -> panels[p].am, 6),
        bm = ntuple(p -> panels[p].bm, 6),
        cm = ntuple(p -> panels[p].cm, 6),
        m_next = ntuple(p -> panels[p].m_next, 6),
        steps = steps,
    )
end

@testset "Plan 39 Commit H — replay-consistency gate regressions" begin

    @testset "verify_window_continuity_ll: continuity-consistent data passes" begin
        for FT in (Float32, Float64)
            w = build_continuity_consistent_window_ll(FT)
            diag = verify_window_continuity_ll(w.m_cur, w.am, w.bm, w.cm, w.m_next, w.steps)
            tol_rel = FT === Float32 ? 1e-6 : 1e-12
            @test diag.max_rel_err <= tol_rel
            @test diag.max_abs_err >= 0
            @test diag.worst_idx isa Tuple{Int, Int, Int}
        end
    end

    @testset "verify_window_continuity_ll: deliberately broken cm fires" begin
        w = build_continuity_consistent_window_ll(Float64)
        # Perturb cm at one interior cell by ~1% of a dm magnitude — this is
        # far above replay_tolerance(Float64) but below cm sanity limits.
        cm_broken = copy(w.cm)
        cm_broken[2, 2, 2] += Float64(1e4)
        diag = verify_window_continuity_ll(w.m_cur, w.am, w.bm, cm_broken, w.m_next, w.steps)
        @test diag.max_rel_err > replay_tolerance(Float64)
        # The cm perturbation at interface k=2 shifts dm at level k=1 (cm[2]−cm[1] up by 1e4
        # gives extra +1e4·2·steps = +4e4 mass change) and at k=2 (cm[3]−cm[2] down by 1e4
        # gives -4e4). Both cells (2,2,1) and (2,2,2) should be worst candidates.
        @test diag.worst_idx[1] == 2 && diag.worst_idx[2] == 2
        @test diag.worst_idx[3] in (1, 2)
    end

    @testset "verify_window_continuity_ll: deliberately broken horizontal flux fires" begin
        w = build_continuity_consistent_window_ll(Float64)
        am_broken = copy(w.am)
        # Perturb an interior x-face flux — this creates a divergence imbalance
        # at the two adjacent cells.
        am_broken[3, 2, 1] += Float64(1e4)
        diag = verify_window_continuity_ll(w.m_cur, am_broken, w.bm, w.cm, w.m_next, w.steps)
        @test diag.max_rel_err > replay_tolerance(Float64)
        # Cell (2, 2, 1) sees +am outflow → m decreases more than expected.
        # Cell (3, 2, 1) sees +am inflow → m increases more than expected.
        @test diag.worst_idx[1] in (2, 3)
        @test diag.worst_idx[2] == 2
        @test diag.worst_idx[3] == 1
    end

    @testset "verify_window_continuity_ll: zero-flux + zero-tendency passes to ULP" begin
        # Trivial case — both m's identical, all fluxes zero.
        m = fill(Float64(1e9), 4, 3, 3)
        am = zeros(Float64, 5, 3, 3)
        bm = zeros(Float64, 4, 4, 3)
        cm = zeros(Float64, 4, 3, 4)
        diag = verify_window_continuity_ll(m, am, bm, cm, m, 2)
        @test diag.max_rel_err == 0
        @test diag.max_abs_err == 0
    end

    @testset "verify_window_continuity_ll: zero-flux + nonzero-dm fires" begin
        # Replicates the existing test-binary helper pattern (zero fluxes, m
        # scales with window index): gate correctly detects the continuity
        # violation.
        m_cur = fill(Float64(1e9), 4, 3, 3)
        m_next = fill(Float64(2e9), 4, 3, 3)
        am = zeros(Float64, 5, 3, 3)
        bm = zeros(Float64, 4, 4, 3)
        cm = zeros(Float64, 4, 3, 4)
        diag = verify_window_continuity_ll(m_cur, am, bm, cm, m_next, 2)
        # |m_cur - m_next| / (|m_cur| + |m_next|) = 1e9 / 3e9 in every cell
        @test diag.max_rel_err ≈ 1 / 3 atol=1e-12
        @test diag.max_abs_err ≈ 1e9
    end

    @testset "replay_tolerance: Parameters values, less precise type of a pair" begin
        @test replay_tolerance(Float64) == REPLAY_TOLERANCE_FLOAT64 == 1e-12
        @test replay_tolerance(Float32) == REPLAY_TOLERANCE_FLOAT32 == 1e-5
        # A binary stored in one type and loaded in the other carries Float32 rounding.
        @test replay_tolerance(Float64, Float32) == REPLAY_TOLERANCE_FLOAT32
        @test replay_tolerance(Float32, Float64) == REPLAY_TOLERANCE_FLOAT32
        @test replay_tolerance(Float64, Float64) == REPLAY_TOLERANCE_FLOAT64
    end

    @testset "verify_window_continuity_ll: residual is measured per cell" begin
        # One column: a thin top cell (1e4 kg) over a thick one (1e10 kg), no flux.
        # A 1e-4 kg mismatch in the thin cell is 5e-9 of its own budget scale
        # (|m_cur| + |m_next| = 2e4 kg) but only 1e-14 of the largest cell mass,
        # so a test normalised by the global maximum mass would pass it.
        m_cur = reshape([1e4, 1e10], 1, 1, 2)
        m_next = copy(m_cur)
        m_next[1, 1, 1] += 1e-4
        am = zeros(2, 1, 2); bm = zeros(1, 2, 2); cm = zeros(1, 1, 3)
        diag = verify_window_continuity_ll(m_cur, am, bm, cm, m_next, 2)
        @test diag.worst_idx == (1, 1, 1)
        @test diag.max_rel_err ≈ 1e-4 / (2e4 + 1e-4) rtol=1e-6
        @test diag.max_rel_err > replay_tolerance(Float64)
        @test 1e-4 / maximum(m_next) < replay_tolerance(Float64)
    end

    @testset "verify_window_continuity_ll: scale is the summed magnitude of the budget terms" begin
        # One cell, two substeps: 10 kg in through the west face, 4 kg out through
        # the east face, 3 kg out through the bottom per application. Net outflow
        # per application -3 kg; m_next is 1 kg off the exact 1000 + 12 = 1012 kg.
        m_cur = fill(1000.0, 1, 1, 1)
        m_next = fill(1013.0, 1, 1, 1)
        am = reshape([10.0, 4.0], 2, 1, 1)
        bm = zeros(1, 2, 1)
        cm = reshape([0.0, 3.0], 1, 1, 2)
        diag = verify_window_continuity_ll(m_cur, am, bm, cm, m_next, 2)
        @test diag.max_abs_err ≈ 1.0
        @test diag.max_rel_err ≈ 1.0 / (1000 + 1013 + 4 * (10 + 4 + 0 + 3))
    end

    @testset "verify_window_continuity_ll: NaN in the stored data fails" begin
        m = fill(Float64(1e9), 4, 3, 3)
        am = zeros(Float64, 5, 3, 3)
        bm = zeros(Float64, 4, 4, 3)
        cm = zeros(Float64, 4, 3, 4)
        am[3, 2, 2] = NaN
        diag = verify_window_continuity_ll(m, am, bm, cm, m, 2)
        @test diag.max_rel_err == Inf
        @test diag.worst_idx in ((2, 2, 2), (3, 2, 2))
        @test !(diag.max_rel_err <= replay_tolerance(Float64))
    end

    @testset "verify_storage_continuity_ll!: final zero-tendency fallback is checked" begin
        # With `last_hour_next === nothing`, the final LL window should replay
        # against `m_cur` itself, not be skipped.
        m = fill(Float64(1e9), 4, 3, 3)
        am = zeros(Float64, 5, 3, 3)
        bm = zeros(Float64, 4, 4, 3)
        cm = zeros(Float64, 4, 3, 4)
        cm[2, 2, 2] = 1e4  # deliberate inconsistency in the only/final window

        storage = allocate_window_storage(1, Float64)
        storage.all_m[1] = copy(m)
        storage.all_am[1] = copy(am)
        storage.all_bm[1] = copy(bm)
        storage.all_cm[1] = copy(cm)
        storage.all_ps[1] = zeros(Float64, 4, 3)

        @test_throws ErrorException verify_storage_continuity_ll!(storage, nothing, 2, Float64)
    end

    # -------------------------------------------------------------------
    # RG helper — minimal face-indexed continuity-consistent tuple.
    # -------------------------------------------------------------------
    @testset "verify_window_continuity_rg: continuity-consistent data passes" begin
        # Tiny 2-ring face-indexed mesh: 6 cells, 3 faces (non-polar).
        # Use zero hflux so continuity depends only on cm vs dm_target.
        nc = 6
        Nz = 3
        steps = 2
        m_cur = fill(Float64(1e9), nc, Nz)
        dm_target = zeros(Float64, nc, Nz)
        for k in 1:Nz, c in 1:nc
            dm_target[c, k] = 1e4 * sinpi(c / nc) * (k / Nz)
        end
        # No horizontal faces on this synthetic test — null face list.
        face_left  = Int32[]
        face_right = Int32[]
        hflux      = zeros(Float64, 0, Nz)
        cm         = zeros(Float64, nc, Nz + 1)
        for c in 1:nc
            acc = 0.0
            for k in 1:Nz
                acc -= dm_target[c, k]
                cm[c, k + 1] = acc
            end
        end
        m_next = similar(m_cur)
        @. m_next = m_cur + 2 * steps * dm_target
        diag = verify_window_continuity_rg(m_cur, hflux, cm, m_next,
                                            face_left, face_right, steps)
        @test diag.max_rel_err <= 1e-12
    end

    @testset "verify_window_continuity_rg: deliberately broken cm fires" begin
        nc = 6
        Nz = 3
        steps = 2
        m_cur = fill(Float64(1e9), nc, Nz)
        face_left  = Int32[]
        face_right = Int32[]
        hflux      = zeros(Float64, 0, Nz)
        cm         = zeros(Float64, nc, Nz + 1)
        # All-zero cm with non-zero m_next → continuity violated.
        m_next = fill(Float64(2e9), nc, Nz)
        diag = verify_window_continuity_rg(m_cur, hflux, cm, m_next,
                                            face_left, face_right, steps)
        @test diag.max_rel_err ≈ 1 / 3 atol=1e-12   # 1e9 / (1e9 + 2e9)
    end

    @testset "verify_window_continuity_rg: face fluxes enter the scale of both cells" begin
        # Two cells joined by one face carrying 5 kg per application from cell 1
        # to cell 2; one level, one substep; cell 2's m_next is 2 kg too high.
        face_left, face_right = Int32[1], Int32[2]
        m_cur = fill(100.0, 2, 1)
        m_next = reshape([90.0, 112.0], 2, 1)
        hflux = fill(5.0, 1, 1)
        cm = zeros(2, 2)
        diag = verify_window_continuity_rg(m_cur, hflux, cm, m_next,
                                            face_left, face_right, 1)
        @test diag.worst_idx == (2, 1)
        @test diag.max_abs_err ≈ 2.0
        @test diag.max_rel_err ≈ 2.0 / (100 + 112 + 2 * 5)
    end

    @testset "verify_window_continuity_cs: continuity-consistent data passes" begin
        for FT in (Float32, Float64)
            w = build_continuity_consistent_window_cs(FT)
            diag = verify_window_continuity_cs(w.m_cur, w.am, w.bm, w.cm, w.m_next, w.steps)
            tol_rel = FT === Float32 ? 1e-6 : 1e-12
            @test diag.max_rel_err <= tol_rel
            @test diag.worst_idx isa NTuple{4, Int}
        end
    end

    @testset "verify_window_continuity_cs: deliberately broken cm fires" begin
        w = build_continuity_consistent_window_cs(Float64)
        cm_broken = ntuple(6) do p
            cm = copy(w.cm[p])
            p == 4 && (cm[2, 2, 2] += 1e4)
            cm
        end
        diag = verify_window_continuity_cs(w.m_cur, w.am, w.bm, cm_broken, w.m_next, w.steps)
        @test diag.max_rel_err > replay_tolerance(Float64)
        @test diag.worst_idx[1] == 4
    end

    @testset "verify_window_continuity_cs: caller scratch is reused" begin
        w = build_continuity_consistent_window_cs(Float64; Nc = 16, Nz = 8)
        scratch = ReplayScratch(size(w.m_cur[1]))
        check(w, scratch) = verify_window_continuity_cs(w.m_cur, w.am, w.bm, w.cm,
                                                        w.m_next, w.steps; scratch)
        reference = verify_window_continuity_cs(w.m_cur, w.am, w.bm, w.cm,
                                                w.m_next, w.steps)
        @test check(w, scratch) === reference
        # One panel-sized Float64 array; a shared scratch must allocate far less.
        @test @allocated(check(w, scratch)) < sizeof(scratch.div) ÷ 2
        Nx, Ny, Nz = size(w.m_cur[1])
        @test_throws ErrorException check(w, ReplayScratch((Nx + 1, Ny, Nz)))
    end
end

# What docs/src/concepts/binary_format.md ("What the replay gate certifies")
# states about the gate. Tightening the gate (per-cell normalisation,
# independent flux checks) must update this testset and that section together.
@testset "Replay gate scope" begin
    MD = AtmosTransport.MetDrivers
    Nx, Ny, Nz, steps = 8, 6, 4, 2

    # A 1 % error in a thin top-level cell is measured against that cell's
    # own budget terms, independent of the much larger cells below it.
    m = fill(1e9, Nx, Ny, Nz)
    m[:, :, 1] .= 1e5
    m_next = copy(m)
    m_next[1, 1, 1] *= 1.01
    diag = verify_window_continuity_ll(m, zeros(Nx + 1, Ny, Nz), zeros(Nx, Ny + 1, Nz),
                                       zeros(Nx, Ny, Nz + 1), m_next, steps)
    @test diag.max_rel_err ≈ 1000 / 201000 rtol = 1e-9
    @test diag.max_rel_err > MD.replay_tolerance(Float32)

    # As the preprocessors wire it: cm is diagnosed from the stored horizontal
    # fluxes and the endpoint tendency right before the gate.
    function diagnosed_cm(m_cur, m_next, am, bm)
        dm = (m_next .- m_cur) ./ (2steps)
        cm = zeros(Nx, Ny, Nz + 1)
        MD.recompute_cm_from_dm_target!(MD.structured_replay_layout(), zeros(Nx, Ny, Nz),
                                        cm, m_cur, dm, am, bm)
        return cm
    end
    gate_as_wired(m_cur, m_next, am, bm; cm = diagnosed_cm(m_cur, m_next, am, bm)) =
        verify_window_continuity_ll(m_cur, am, bm, cm, m_next, steps).max_rel_err
    m_cur = [1e9 * (1 + 0.1 * sinpi(i / 4 + j / 3) * k / Nz) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    am = [3e5 * (1 + cospi(2(i - 1) / Nx)) * sinpi((j - 0.5) / Ny) * (1 + k / Nz)
          for i in 1:Nx + 1, j in 1:Ny, k in 1:Nz]
    am[Nx + 1, :, :] .= am[1, :, :]
    bm = [j in (1, Ny + 1) ? 0.0 : 5e4 * sinpi(2i / Nx + 0.3k) for i in 1:Nx, j in 1:Ny + 1, k in 1:Nz]
    div_h = zeros(Nx, Ny, Nz)
    MD.horizontal_divergence!(div_h, MD.structured_replay_layout(), am, bm)
    # Endpoint tendency that closes every column's budget; its vertical split
    # differs from the horizontal divergence.
    shift = [1e3 * cospi(i / 4 + j / 3 + k / 2) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    shift .-= sum(shift; dims = 3) ./ Nz
    m_next = m_cur .+ 2steps .* (shift .- div_h)
    # A non-divergent flux from a streamfunction on cell corners.
    ψ = [j in (1, Ny + 1) ? 0.0 : 2e5 * sinpi(2(i - 1) / Nx + 0.2j) for i in 1:Nx + 1, j in 1:Ny + 1]
    ψ[Nx + 1, :] .= ψ[1, :]
    am_rot = [ψ[i, j + 1] - ψ[i, j] for i in 1:Nx + 1, j in 1:Ny]
    bm_rot = [ψ[i, j] - ψ[i + 1, j] for i in 1:Nx, j in 1:Ny + 1]
    broken = copy(am)
    broken[3, 3, :] .*= 2

    @test gate_as_wired(m_cur, m_next, am, bm) < 1e-14
    cm_through = diagnosed_cm(m_cur, m_next, am, bm)
    cm_through[2, 2, :] .+= 1e6                       # same flux through top and surface
    @test gate_as_wired(m_cur, m_next, am, bm; cm = cm_through) < 1e-14
    @test gate_as_wired(m_cur, m_next, am .+ am_rot, bm .+ bm_rot) < 1e-14  # rotational error
    @test gate_as_wired(m_cur, m_next, am[:, :, end:-1:1], bm[:, :, end:-1:1]) < 1e-14  # level flip
    @test gate_as_wired(m_cur, m_next, broken, bm) > 1e-4                   # column budget broken
end
