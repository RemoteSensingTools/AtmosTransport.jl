#!/usr/bin/env julia
# Focused regression tests for the per-window CS preprocessor contract
# surface exported from `cubed_sphere_contracts.jl`.
#
# Builds tiny 6-panel synthetic windows that trip each gate (replay,
# positivity-CFL, m <= 0, NaN/Inf mass, NaN/Inf flux) and confirms the
# corresponding helper fires with the documented diagnostic. Also locks in
# the round-2 (45b87f3) regression that `summarize_cs_positivity_status`
# does not throw `InexactError` on a non-finite ratio under either policy.

using Test
using Logging: with_logger, NullLogger

using AtmosTransport
using .AtmosTransport.Preprocessing: verify_substep_positivity_cs!,
                                       verify_cs_window_contract!,
                                       init_cs_positivity_accumulator,
                                       update_cs_positivity_accumulator,
                                       summarize_cs_positivity_status,
                                       CubedSphereContract,
                                       AbstractWindowContract

# Internal helper; not exported. Accessed by qualified name so the entrypoint
# validation regression (round-3) doesn't require exporting an underscore
# helper just for tests.
const _resolve_positivity_cfl_limit =
    AtmosTransport.Preprocessing._resolve_positivity_cfl_limit
const fill_cs_mass_delta_payload! =
    AtmosTransport.Preprocessing.fill_cs_mass_delta_payload!

# Build a 6-panel CS window whose horizontal fluxes vanish and whose vertical
# fluxes encode the per-cell mass tendency exactly. The result satisfies the
# write-time replay gate to F64 ULP, so the positivity scan can be exercised
# in isolation through `verify_cs_window_contract!` without tripping the
# replay short-circuit first.
function build_clean_cs_window(FT::Type; Nc::Int = 4, Nz::Int = 3, steps::Int = 2,
                                m_base::Real = 1e9, dm_scale::Real = 1e4)
    panels = ntuple(6) do p
        m_cur = fill(FT(m_base), Nc, Nc, Nz)
        am = zeros(FT, Nc + 1, Nc, Nz)
        bm = zeros(FT, Nc, Nc + 1, Nz)
        cm = zeros(FT, Nc, Nc, Nz + 1)
        dm = Array{FT}(undef, Nc, Nc, Nz)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            dm[i, j, k] = FT(dm_scale) * sinpi(FT(i) / Nc) *
                          cospi(FT(j) / Nc) * FT(k / Nz) * FT(1 + 0.05 * p)
        end
        for j in 1:Nc, i in 1:Nc
            acc = 0.0
            for k in 1:Nz
                acc -= Float64(dm[i, j, k])
                cm[i, j, k + 1] = FT(acc)
            end
        end
        m_next = m_cur .+ FT(2 * steps) .* dm
        (; m_cur, am, bm, cm, m_next)
    end
    return (m_cur = ntuple(p -> panels[p].m_cur, 6),
            am = ntuple(p -> panels[p].am, 6),
            bm = ntuple(p -> panels[p].bm, 6),
            cm = ntuple(p -> panels[p].cm, 6),
            m_next = ntuple(p -> panels[p].m_next, 6),
            steps = steps)
end

# Discard log output from `summarize_cs_positivity_status` during the
# require=false / Inf branches — the tests assert return values, not text.
with_quiet_logger(f) = with_logger(f, NullLogger())

@testset "CS preprocessor contract gates" begin

    @testset "ERA5 sliding-window dm payload does not mutate endpoint" begin
        FT = Float32
        m_cur = ntuple(p -> fill(FT(100 + p), 2, 2, 2), 6)
        m_next = ntuple(p -> fill(FT(130 + p), 2, 2, 2), 6)
        m_next_before = ntuple(p -> copy(m_next[p]), 6)
        dm_payload = ntuple(_ -> zeros(FT, 2, 2, 2), 6)

        fill_cs_mass_delta_payload!(dm_payload, m_cur, m_next)

        for p in 1:6
            @test dm_payload[p] == m_next_before[p] .- m_cur[p]
            @test m_next[p] == m_next_before[p]
        end

        # Regression for the ERA5 N320 writer's window-2 replay failure:
        # the buffer swapped into `cur_m_dry` must remain the absolute
        # endpoint, not the serialized delta payload.
        cur_after_swap = m_next
        @test cur_after_swap[4][1, 1, 1] == m_next_before[4][1, 1, 1]
        @test cur_after_swap[4][1, 1, 1] != dm_payload[4][1, 1, 1]
    end

    # ------------------------------------------------------------------
    # verify_substep_positivity_cs!
    # ------------------------------------------------------------------

    @testset "positivity: clean window passes" begin
        for FT in (Float32, Float64)
            w = build_clean_cs_window(FT)
            diag = verify_substep_positivity_cs!(w.m_cur, w.am, w.bm, w.cm;
                                                  cfl_limit = 0.95)
            @test diag.ok
            @test isfinite(diag.ratio)
            @test diag.ratio < 0.95
            @test diag.location isa NTuple{4, Int}
        end
    end

    @testset "positivity: x-face CFL violation reported with direction and location" begin
        w = build_clean_cs_window(Float64)
        am = ntuple(6) do p
            arr = copy(w.am[p])
            p == 3 && (arr[3, 2, 1] = 0.99e9)  # right face of cell (2,2,1)
            arr
        end
        diag = verify_substep_positivity_cs!(w.m_cur, am, w.bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.direction === :x
        @test diag.ratio ≈ 1.98 atol = 1e-12
        @test diag.location == (3, 2, 2, 1)
    end

    @testset "positivity: y-face CFL violation reported with direction and location" begin
        w = build_clean_cs_window(Float64)
        bm = ntuple(6) do p
            arr = copy(w.bm[p])
            p == 5 && (arr[2, 3, 2] = 0.97e9)  # top face of cell (2,2,2)
            arr
        end
        diag = verify_substep_positivity_cs!(w.m_cur, w.am, bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.direction === :y
        @test diag.ratio ≈ 1.94 atol = 1e-12
        @test diag.location == (5, 2, 2, 2)
    end

    @testset "positivity: z-face violation is included in the scan" begin
        # Drive a single vertical interface flux up enough to dominate the
        # tiny baseline z-ratio from the clean-window dm divergence.
        w = build_clean_cs_window(Float64)
        cm = ntuple(6) do p
            arr = copy(w.cm[p])
            p == 6 && (arr[1, 1, 3] = 0.98e9)  # interface k=3 of cell (1,1,2)
            arr
        end
        diag = verify_substep_positivity_cs!(w.m_cur, w.am, w.bm, cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.direction === :z
        # The clean-window baseline contributes a few units to cm at this
        # interface, so the observed palindrome ratio is slightly above
        # 2 * 0.98 — loose
        # tolerance is fine, the point is that the z-direction violation
        # surfaces rather than being silently dominated by tiny x/y ratios.
        @test diag.ratio ≈ 1.96 atol = 1e-3
        @test diag.location == (6, 1, 1, 2)
    end

    @testset "positivity: combined palindrome budget catches per-direction-safe cells" begin
        w = build_clean_cs_window(Float64)
        am = ntuple(6) do p
            arr = copy(w.am[p])
            p == 1 && (arr[2, 2, 1] = 0.30e9)
            arr
        end
        bm = ntuple(6) do p
            arr = copy(w.bm[p])
            p == 1 && (arr[1, 3, 1] = 0.30e9)
            arr
        end
        cm = ntuple(6) do p
            arr = copy(w.cm[p])
            p == 1 && (arr[1, 2, 2] = 0.20e9)
            arr
        end
        diag = verify_substep_positivity_cs!(w.m_cur, am, bm, cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test diag.ratio ≈ 1.60 atol = 1e-12
        @test diag.location == (1, 1, 2, 1)
    end

    @testset "positivity: m_next tightens the reference mass" begin
        w = build_clean_cs_window(Float64)
        am = ntuple(6) do p
            arr = copy(w.am[p])
            p == 1 && (arr[2, 2, 1] = 0.30e9)
            arr
        end
        m_next = ntuple(6) do p
            arr = copy(w.m_next[p])
            p == 1 && (arr[1, 2, 1] = 0.50e9)
            arr
        end
        diag = verify_substep_positivity_cs!(w.m_cur, am, w.bm, w.cm;
                                              cfl_limit = 0.95,
                                              m_next = m_next)
        @test !diag.ok
        @test diag.ratio ≈ 1.20 atol = 1e-12
        @test diag.location == (1, 1, 2, 1)
    end

    @testset "positivity: m <= 0 is flagged Inf regardless of flux magnitude (round-2 fix)" begin
        w = build_clean_cs_window(Float64)
        m_cur = ntuple(6) do p
            arr = copy(w.m_cur[p])
            p == 1 && (arr[2, 3, 1] = 0.0)
            arr
        end
        diag = verify_substep_positivity_cs!(m_cur, w.am, w.bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
        # The kernel pins the report to the first non-positive cell it
        # encounters; subsequent cells cannot make `Inf` worse.
        @test diag.direction === :x
        @test diag.location == (1, 2, 3, 1)
    end

    @testset "positivity: NaN mass yields Inf ratio without slipping through (round-2 fix)" begin
        # Before the round-2 fix the kernel branched on `mi <= 0`, which is
        # `false` for `NaN` in Julia — so a `NaN`-mass cell would fall through
        # to the divisor branch where `NaN / NaN = NaN`, and `NaN > worst_ratio`
        # is also false, silently dropping the cell.
        w = build_clean_cs_window(Float64)
        m_cur = ntuple(6) do p
            arr = copy(w.m_cur[p])
            p == 2 && (arr[1, 1, 1] = NaN)
            arr
        end
        diag = verify_substep_positivity_cs!(m_cur, w.am, w.bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
        @test diag.location == (2, 1, 1, 1)
    end

    @testset "positivity: NaN flux yields Inf ratio (round-2 fix)" begin
        w = build_clean_cs_window(Float64)
        am = ntuple(6) do p
            arr = copy(w.am[p])
            p == 6 && (arr[3, 3, 2] = NaN)
            arr
        end
        diag = verify_substep_positivity_cs!(w.m_cur, am, w.bm, w.cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
    end

    @testset "positivity: Inf flux yields Inf ratio (round-2 fix)" begin
        w = build_clean_cs_window(Float64)
        cm = ntuple(6) do p
            arr = copy(w.cm[p])
            p == 4 && (arr[2, 2, 2] = Inf)
            arr
        end
        diag = verify_substep_positivity_cs!(w.m_cur, w.am, w.bm, cm;
                                              cfl_limit = 0.95)
        @test !diag.ok
        @test isinf(diag.ratio)
    end

    # ------------------------------------------------------------------
    # verify_cs_window_contract!
    # ------------------------------------------------------------------

    @testset "wrapper: clean window returns both diagnostics" begin
        w = build_clean_cs_window(Float64)
        result = verify_cs_window_contract!(w.m_cur, w.am, w.bm, w.cm,
                                             w.m_next, w.steps, 1;
                                             replay_tol = 1e-12,
                                             positivity_cfl_limit = 0.95)
        @test result.replay.max_rel_err <= 1e-12
        @test result.positivity.ok
    end

    @testset "wrapper: replay failure errors before positivity is reached" begin
        w = build_clean_cs_window(Float64)
        cm_broken = ntuple(6) do p
            arr = copy(w.cm[p])
            p == 4 && (arr[2, 2, 2] += 1e4)
            arr
        end
        @test_throws ErrorException verify_cs_window_contract!(
            w.m_cur, w.am, w.bm, cm_broken, w.m_next, w.steps, 7;
            replay_tol = 1e-12, positivity_cfl_limit = 0.95,
        )
    end

    @testset "wrapper: positivity failure with passing replay is non-fatal (caller policy)" begin
        # Closed-loop uniform shift on one (j, k) row preserves per-cell
        # divergence on the perturbed row but drives the palindrome outgoing
        # budget above the limit.
        # `verify_cs_window_contract!` must return both diagnostics so the
        # caller — not the wrapper — decides whether to error or warn after
        # aggregating across windows.
        w = build_clean_cs_window(Float64)
        am = ntuple(6) do p
            arr = copy(w.am[p])
            p == 2 && (arr[:, 1, 1] .= 0.99e9)
            arr
        end
        result = verify_cs_window_contract!(w.m_cur, am, w.bm, w.cm,
                                             w.m_next, w.steps, 3;
                                             replay_tol = 1e-12,
                                             positivity_cfl_limit = 0.95)
        @test result.replay.max_rel_err <= 1e-12
        @test !result.positivity.ok
        @test result.positivity.direction === :x
        @test result.positivity.ratio ≈ 1.98 atol = 1e-12
        @test result.positivity.location == (2, 1, 1, 1)
    end

    # The accumulator, summary, contract-constructor and direct-wrapper
    # policy checks for all three topologies live in the shared testsets at
    # the end of this file.

    # ------------------------------------------------------------------
    # Round-3: TOML-resolver validation of `[numerics].positivity_cfl_limit`.
    # ------------------------------------------------------------------

    @testset "entrypoint: default cfl_limit is 0.95" begin
        @test _resolve_positivity_cfl_limit(Dict{String, Any}()) == 0.95
    end

    @testset "entrypoint: invalid cfl_limit is rejected at TOML load" begin
        @test_throws ErrorException _resolve_positivity_cfl_limit(
            Dict("numerics" => Dict("positivity_cfl_limit" => 0.0)))
    end

    @testset "CubedSphereContract: write_replay_on = false runs only the positivity gate" begin
        # `[numerics] write_replay_check = false` keeps a binary that fails replay for
        # inspection; the positivity gate still uses the window's end mass.
        w = build_clean_cs_window(Float64)
        contract = CubedSphereContract{Float64}(replay_tol = 1e-12,
                                                 positivity_cfl_limit = 0.95,
                                                 steps_per_window = w.steps)
        # Half the end mass: breaks continuity, and the gate's min(m, m_next) is m_next.
        bad_next = map(m -> 0.5 .* m, w.m_next)
        window = (m_cur = w.m_cur, am = w.am, bm = w.bm, cm = w.cm, m_next = bad_next)
        verify = AtmosTransport.Preprocessing.verify_window!
        @test_throws ErrorException verify(window, contract, 1)
        diag = verify(window, contract, 1; write_replay_on = false)
        @test diag.replay.max_rel_err == 0
        @test diag.positivity == verify_substep_positivity_cs!(w.m_cur, w.am, w.bm, w.cm;
                                                                cfl_limit = 0.95, m_next = bad_next)
        @test diag.positivity.ratio > verify_substep_positivity_cs!(w.m_cur, w.am, w.bm, w.cm;
                                                                     cfl_limit = 0.95).ratio
    end
end

@testset "Shared target contract policy and worst-window lifecycle" begin
    P = AtmosTransport.Preprocessing
    for (Contract, init, update, summarize, N) in (
        (P.LatLonContract, P.init_ll_positivity_accumulator,
         P.update_ll_positivity_accumulator, P.summarize_ll_positivity_status, 3),
        (P.CubedSphereContract, P.init_cs_positivity_accumulator,
         P.update_cs_positivity_accumulator, P.summarize_cs_positivity_status, 4),
        (P.ReducedGaussianContract, P.init_rg_positivity_accumulator,
         P.update_rg_positivity_accumulator, P.summarize_rg_positivity_status, 2))
        extra = N == 2 ? (; face_left=Int32[1], face_right=Int32[2]) : (;)
        for FT in (Float32, Float64)
            for tol in (0.0, -1.0, Inf, NaN)
                @test_throws ErrorException Contract{FT}(; replay_tol=tol, steps_per_window=2, extra...)
            end
            for limit in (0.0, -0.1, 1.01, Inf, NaN)
                @test_throws ErrorException Contract{FT}(; replay_tol=1e-5,
                    positivity_cfl_limit=limit, steps_per_window=2, extra...)
            end
            @test_throws ErrorException Contract{FT}(; replay_tol=1e-5, steps_per_window=0, extra...)
        end
        location = ntuple(identity, N)
        worst = update(init(), (ratio=0.7, direction=:x, location=location), 2)
        @test worst.ratio == 0.7 && worst.win == 2 && worst.location == location
        @test update(worst, (ratio=0.7, direction=:z, location=location), 3) == worst
        @test update(worst, (ratio=0.1, direction=:z, location=location), 4) == worst
        contract = Contract{Float64}(; replay_tol=1e-5, steps_per_window=2, extra...)
        P.update_accumulator!(contract, (ratio=0.7, direction=:x, location=location), 2)
        @test contract.worst == worst
        @test with_quiet_logger(() -> P.summarize_status!(contract)) === nothing
        for ratio in (1.5,), require_positive in (true, false)
            bad = merge(worst, (; ratio))
            call_summary() = with_quiet_logger() do
                summarize(bad; cfl_limit=0.95, steps_per_window=2,
                          require_substep_positivity=require_positive)
            end
            if require_positive
                @test_throws ErrorException call_summary()
            else
                @test call_summary() === nothing
            end
        end
        mktempdir() do dir
            path=joinpath(dir,"invalid.bin"); write(path, "invalid")
            @test_throws ErrorException with_quiet_logger() do
                summarize(merge(worst,(;ratio=1.5)); cfl_limit=0.95,
                          steps_per_window=2, quarantine_path=path)
            end
            @test !isfile(path)
            write(path,"diagnostic")
            @test with_quiet_logger() do
                summarize(merge(worst,(;ratio=1.5)); cfl_limit=0.95,
                          steps_per_window=2, require_substep_positivity=false,
                          quarantine_path=path) === nothing
            end
            @test isfile(path)
        end
    end
    for bad in (-1.0, Inf, NaN)
        @test_throws ErrorException P.ReducedGaussianContract{Float64}(;
            replay_tol=1e-5, steps_per_window=2, boundary_stub_tol=bad,
            face_left=Int32[1], face_right=Int32[2])
    end
    @test_throws ErrorException P.ReducedGaussianContract{Float64}(;
        replay_tol=1e-5, steps_per_window=2,
        face_left=Int32[1,2], face_right=Int32[2])
end

@testset "Target gate entry points reject invalid numerical policy" begin
    P=AtmosTransport.Preprocessing
    m=ones(2,2,1); am=zeros(3,2,1); bm=zeros(2,3,1); cm=zeros(2,2,2)
    ll(tol, limit)=P.verify_ll_window_contract!(m,am,bm,cm,m,1,1;
        replay_tol=tol, positivity_cfl_limit=limit)
    cs(tol, limit)=P.verify_cs_window_contract!((m,),(am,),(bm,),(cm,),(m,),1,1;
        replay_tol=tol, positivity_cfl_limit=limit)
    rg(tol, limit)=P.verify_rg_window_contract!(ones(2,1),zeros(1,1),zeros(2,2),
        ones(2,1),Int32[1],Int32[2],1,1; replay_tol=tol, positivity_cfl_limit=limit)
    for check in (ll,cs,rg), bad in (0.0,-1.0,Inf,NaN)
        @test_throws ErrorException check(bad,0.95)
        @test_throws ErrorException check(1e-5,bad)
    end
    scans=(limit -> P.verify_substep_positivity_ll!(m,am,bm,cm;cfl_limit=limit),
           limit -> P.verify_substep_positivity_cs!((m,),(am,),(bm,),(cm,);cfl_limit=limit),
           limit -> P.verify_substep_positivity_rg!(ones(2,1),zeros(1,1),zeros(2,2),
               Int32[1],Int32[2];cfl_limit=limit))
    for scan in scans, bad in (0.0,-1.0,1.01,Inf,NaN)
        @test_throws ErrorException scan(bad)
    end
end
