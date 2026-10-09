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
const _fill_cs_mass_delta_payload! =
    AtmosTransport.Preprocessing._fill_cs_mass_delta_payload!

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

        _fill_cs_mass_delta_payload!(dm_payload, m_cur, m_next)

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

    @testset "positivity: halo_width skips halo cells" begin
        # Build a haloed buffer with the violation seeded in the halo region;
        # the interior is clean (all-zero fluxes, m = 1e9), so positivity
        # must report ok = true.
        FT = Float64
        Nc = 4
        Nz = 3
        Hp = 1
        m = ntuple(_ -> begin
            arr = fill(FT(1e9), Nc + 2Hp, Nc + 2Hp, Nz)
            arr[1, 1, 1] = -1.0  # in the halo
            arr
        end, 6)
        am = ntuple(_ -> zeros(FT, Nc + 2Hp + 1, Nc + 2Hp, Nz), 6)
        bm = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp + 1, Nz), 6)
        cm = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz + 1), 6)
        diag = verify_substep_positivity_cs!(m, am, bm, cm;
                                              cfl_limit = 0.95, halo_width = Hp)
        @test diag.ok
        @test diag.ratio == 0.0
        # Same buffer with halo_width = 0 must catch the same violation.
        diag0 = verify_substep_positivity_cs!(m, am, bm, cm;
                                               cfl_limit = 0.95, halo_width = 0)
        @test !diag0.ok
        @test isinf(diag0.ratio)
        @test diag0.location == (1, 1, 1, 1)
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

    # ------------------------------------------------------------------
    # accumulator
    # ------------------------------------------------------------------

    @testset "accumulator: tracks worst window across the loop" begin
        worst = init_cs_positivity_accumulator()
        @test worst.ratio == 0.0
        @test worst.direction === :none
        @test worst.win == 0
        @test worst.location == (0, 0, 0, 0)

        worst = update_cs_positivity_accumulator(worst,
            (direction = :x, ratio = 0.3, location = (1, 2, 3, 4), ok = true), 5)
        @test worst.ratio ≈ 0.3
        @test worst.direction === :x
        @test worst.win == 5
        @test worst.location == (1, 2, 3, 4)

        # Smaller ratio is ignored — the older worst is preserved.
        worst = update_cs_positivity_accumulator(worst,
            (direction = :y, ratio = 0.1, location = (2, 1, 1, 1), ok = true), 6)
        @test worst.ratio ≈ 0.3
        @test worst.win == 5

        worst = update_cs_positivity_accumulator(worst,
            (direction = :z, ratio = Inf, location = (3, 2, 2, 2), ok = false), 7)
        @test isinf(worst.ratio)
        @test worst.direction === :z
        @test worst.win == 7
        @test worst.location == (3, 2, 2, 2)
    end

    @testset "accumulator: direction === nothing is normalized to :none" begin
        worst = update_cs_positivity_accumulator(init_cs_positivity_accumulator(),
            (direction = nothing, ratio = 0.5, location = (1, 1, 1, 1), ok = true), 1)
        @test worst.direction === :none
        @test worst.ratio ≈ 0.5
    end

    # ------------------------------------------------------------------
    # summarize_cs_positivity_status
    # ------------------------------------------------------------------

    @testset "summary: ratio within limit returns nothing" begin
        worst = (ratio = 0.5, direction = :x, win = 1, location = (1, 1, 1, 1))
        @test summarize_cs_positivity_status(worst; cfl_limit = 0.95,
                                              steps_per_window = 8) === nothing
    end

    @testset "summary: finite violation + require=true errors with rescue advice" begin
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1, 1))
        err = try
            summarize_cs_positivity_status(worst; cfl_limit = 0.95,
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
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1, 1))
        r = with_quiet_logger() do
            summarize_cs_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = false)
        end
        @test r === nothing
    end

    @testset "summary: Inf ratio + require=true throws ErrorException, NOT InexactError (round-2 fix)" begin
        # Before the round-2 fix this branch hit `ceil(Int, Inf)` and threw
        # `InexactError(:Int64, Inf)` BEFORE the intended error/warn path.
        # That broke the `require_substep_positivity = false` escape hatch for
        # exactly the failure modes that produce `Inf`/`NaN` ratios — the only
        # ones an operator might legitimately want to record-and-continue.
        worst = (ratio = Inf, direction = :z, win = 4, location = (2, 3, 3, 2))
        err = try
            summarize_cs_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = true)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test !(err isa InexactError)
        # Round-2 used a dedicated "non-finite" message; round-3 merged the
        # non-finite and finite-but-pathologically-large branches under one
        # "no representable rescue" message so both are handled uniformly.
        @test occursin("no representable", err.msg)
    end

    @testset "summary: Inf ratio + require=false warns (no InexactError) (round-2 fix)" begin
        worst = (ratio = Inf, direction = :z, win = 4, location = (2, 3, 3, 2))
        r = with_quiet_logger() do
            summarize_cs_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = false)
        end
        @test r === nothing
    end

    @testset "summary: quarantine_path is deleted on error" begin
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1, 1))
        tmp = tempname() * ".bin"
        write(tmp, b"contents")
        @test isfile(tmp)
        @test_throws ErrorException summarize_cs_positivity_status(
            worst; cfl_limit = 0.95, steps_per_window = 8,
            require_substep_positivity = true, quarantine_path = tmp,
        )
        @test !isfile(tmp)
    end

    @testset "summary: missing quarantine_path is benign" begin
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1, 1))
        missing_path = tempname() * ".bin"
        @test !isfile(missing_path)
        @test_throws ErrorException summarize_cs_positivity_status(
            worst; cfl_limit = 0.95, steps_per_window = 8,
            require_substep_positivity = true, quarantine_path = missing_path,
        )
    end

    # ------------------------------------------------------------------
    # Round-3: the `isfinite(worst.ratio)` guard from round-2 only covered
    # `Inf`/`NaN`; `ceil(Int, ratio / cfl_limit)` could still throw
    # `InexactError` for finite-but-huge ratios (e.g. `1e308`) or when an
    # invalid `cfl_limit = 0.0` drives the divide to `Inf`. The fix
    # routes both through the same "no representable rescue" branch and
    # validates `0 < cfl_limit <= 1` at TOML resolve time.
    # ------------------------------------------------------------------

    @testset "summary: finite-but-pathologically-large ratio + require=true (round-3 fix)" begin
        # ceil(Int, 1e308 / 0.95) overflows Int — the pre-round-3 path
        # threw InexactError before the intended error/warn branch.
        worst = (ratio = 1e308, direction = :z, win = 4, location = (2, 3, 3, 2))
        err = try
            summarize_cs_positivity_status(worst; cfl_limit = 0.95,
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

    @testset "summary: finite-but-pathologically-large ratio + require=false (round-3 fix)" begin
        worst = (ratio = 1e308, direction = :z, win = 4, location = (2, 3, 3, 2))
        r = with_quiet_logger() do
            summarize_cs_positivity_status(worst; cfl_limit = 0.95,
                                            steps_per_window = 8,
                                            require_substep_positivity = false)
        end
        @test r === nothing
    end

    @testset "summary: cfl_limit = 0.0 + require=true (round-3 fix)" begin
        # 1.5 / 0.0 = Inf → ceil(Int, Inf) hit InexactError pre-round-3.
        # The entrypoint resolver now rejects cfl_limit = 0 at TOML load
        # time, but the summary helper is also called from inspectors and
        # direct callers, so it must defend itself.
        worst = (ratio = 1.5, direction = :x, win = 1, location = (1, 1, 1, 1))
        err = try
            summarize_cs_positivity_status(worst; cfl_limit = 0.0,
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
        # The Float64 representation of `typemax(Int) ÷ 8` is at the edge
        # of Int64 exactness; the guard must route the case through the
        # "no representable rescue" branch rather than risk silent
        # `ceil(Int, _) * 8` overflow.
        max_factor = typemax(Int) ÷ 8
        worst = (ratio = Float64(max_factor) * 0.95, direction = :z, win = 1,
                 location = (1, 1, 1, 1))
        err = try
            summarize_cs_positivity_status(worst; cfl_limit = 0.95,
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
    # Round-3: TOML-resolver validation of `[numerics].positivity_cfl_limit`.
    # ------------------------------------------------------------------

    @testset "entrypoint: default cfl_limit is 0.95" begin
        @test _resolve_positivity_cfl_limit(Dict{String, Any}()) == 0.95
    end

    @testset "entrypoint: valid cfl_limit values are accepted" begin
        for v in (0.1, 0.5, 0.95, 1.0)
            cfg = Dict("numerics" => Dict("positivity_cfl_limit" => v))
            @test _resolve_positivity_cfl_limit(cfg) == v
        end
    end

    @testset "entrypoint: invalid cfl_limit values are rejected at TOML load" begin
        for v in (0.0, -0.5, 1.5, Inf, -Inf, NaN)
            cfg = Dict("numerics" => Dict("positivity_cfl_limit" => v))
            err = try
                _resolve_positivity_cfl_limit(cfg)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("positivity_cfl_limit", err.msg)
            @test occursin("0, 1", err.msg)
        end
    end

    # ------------------------------------------------------------------
    # CubedSphereContract construction validation (codex round-1 — Inf
    # would silently disable replay; NaN would fail every window late).
    # The typed concrete validates every policy knob at construction so
    # an invalid TOML value fails BEFORE any window is preprocessed.
    # ------------------------------------------------------------------

    @testset "CubedSphereContract: construction validates policy fields" begin
        for bad in (Inf, NaN, 0.0, -1e-12, -Inf)
            @test_throws ErrorException CubedSphereContract{Float64}(
                replay_tol = bad, positivity_cfl_limit = 0.95,
                steps_per_window = 1)
        end
        @test_throws ErrorException CubedSphereContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = 0.0,
            steps_per_window = 1)
        @test_throws ErrorException CubedSphereContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = 1.5,
            steps_per_window = 1)
        @test_throws ErrorException CubedSphereContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = NaN,
            steps_per_window = 1)
        @test_throws ErrorException CubedSphereContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = 0.95,
            steps_per_window = 0)
        @test_throws ErrorException CubedSphereContract{Float64}(
            replay_tol = 1e-12, positivity_cfl_limit = 0.95,
            steps_per_window = 1, halo_width = -1)
        c = CubedSphereContract{Float64}(replay_tol = 1e-12,
                                          positivity_cfl_limit = 0.95,
                                          steps_per_window = 8)
        @test c isa AbstractWindowContract
        @test c.replay_tol == 1e-12
        @test c.positivity_cfl_limit == 0.95
        @test c.require_substep_positivity == true
        @test c.steps_per_window == 8
        @test c.halo_width == 0
        @test c.worst.ratio == 0.0
    end

    # ------------------------------------------------------------------
    # Direct-wrapper policy validation (codex review round 3 of f5224a6).
    # The CS production paths in `cubed_sphere_spectral.jl`,
    # the GEOS workflow, and `cubed_sphere_regrid.jl` call
    # `verify_cs_window_contract!` / `verify_substep_positivity_cs!` /
    # `verify_write_replay_cs!` directly without going through
    # `CubedSphereContract`. Codex reproduced a bypass where
    # `replay_tol = Inf` returned passing replay; the wrappers must
    # validate independently of the contract constructor.
    # ------------------------------------------------------------------

    @testset "direct wrapper: invalid replay_tol is rejected at verify_cs_window_contract!" begin
        w = build_clean_cs_window(Float64)
        for bad in (Inf, NaN, 0.0, -1e-12, -Inf)
            err = try
                verify_cs_window_contract!(w.m_cur, w.am, w.bm, w.cm, w.m_next,
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

    @testset "direct wrapper: invalid positivity_cfl_limit is rejected at verify_cs_window_contract!" begin
        w = build_clean_cs_window(Float64)
        for bad in (Inf, NaN, 0.0, -0.1, 1.5)
            err = try
                verify_cs_window_contract!(w.m_cur, w.am, w.bm, w.cm, w.m_next,
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

    @testset "direct wrapper: invalid cfl_limit is rejected at verify_substep_positivity_cs!" begin
        w = build_clean_cs_window(Float64)
        for bad in (Inf, NaN, 0.0, -0.1, 1.5)
            err = try
                verify_substep_positivity_cs!(w.m_cur, w.am, w.bm, w.cm;
                                                cfl_limit = bad)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("cfl_limit", err.msg)
        end
    end

    @testset "CubedSphereContract: scratch is lazily allocated once and reused (codex round-2)" begin
        # P2 watchpoint closed: the trait `verify_window!` allocates
        # the panel-shared `_div_scratch` on the first call and reuses
        # it thereafter. Eliminates the per-window `Array{Float64}`
        # allocation the original `verify_window_continuity_cs` would
        # produce inside the legacy convenience wrapper.
        w = build_clean_cs_window(Float64)
        contract = CubedSphereContract{Float64}(replay_tol = 1e-12,
                                                 positivity_cfl_limit = 0.95,
                                                 steps_per_window = w.steps)
        @test contract._div_scratch === nothing
        # Build a window NamedTuple matching the typed-concrete surface.
        window = (m_cur = w.m_cur, am = w.am, bm = w.bm,
                  cm = w.cm, m_next = w.m_next)
        AtmosTransport.Preprocessing.verify_window!(window, contract, 1)
        @test contract._div_scratch isa Array{Float64, 3}
        @test size(contract._div_scratch) == size(w.m_cur[1])
        ds = contract._div_scratch
        AtmosTransport.Preprocessing.verify_window!(window, contract, 2)
        @test contract._div_scratch === ds
    end
end
