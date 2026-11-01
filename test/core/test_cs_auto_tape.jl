using KernelAbstractions: CPU
using Test
import AtmosTransport
const AT = AtmosTransport
const T = AT.Tape
const F = AT.Adjoints.Footprint
include(joinpath(@__DIR__, "..", "helpers", "cs_adjoint_tape_harness.jl"))
const H = CSAdjointTapeHarness

function _auto_replay(P)
    return first(F._cs_replay_for_call(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh;
                                       base_emission_rates = P.rates, H._common(P)...))
end

@testset "AutoTape constraints and typed reservation errors" begin
    @test AT.AutoTape().storage === nothing
    @test_throws MethodError AT.AutoTape(; reuse = AT.AirMassTape())
    @test_throws ArgumentError AT.AutoTape(; budget_bytes = -1)
    @test_throws ArgumentError AT.AutoTape(; storage = AT.DeviceCSTapeStorage(), scratch_dir = "x")
    @test_throws ArgumentError AT.AutoTape(; scratch_dir = " ")
    @test_throws ArgumentError AT.PinnedHostCSTapeStorage(; max_bytes = -1)
    @test_throws ArgumentError AT.StrideCheckpoint(2; resident_tail = -1)
    @test_throws ArgumentError AT.StrideCheckpoint(2; snapshots = AT.MmapCSTapeStorage("x"))
    @test occursin("estimate 10", sprint(showerror, AT.TapeReservationError(10, 2, "test")))
    @test occursin("reservation of 2", sprint(showerror, AT.TapeReservationExceeded(2, 10)))
end

@testset "binary size model equals counting hooks and every stride bound" begin
    for FT in (Float32, Float64), scheme in (AT.UpwindScheme(), AT.PPMScheme(), AT.PPMScheme(AT.CW84Limiter())),
        reset in (:none, :preserve_tracer_mass, :preserve_vmr), cadence in (:window, :substep)
        P = H.problem(H.Case(; name = "size", FT, scheme, reset, cadence, conv = :tm5, diffusion = :split))
        replay = _auto_replay(P)
        size = F._closed_tape_size(replay, P.nsteps)
        counter = F._CSCountingRecorder(AT.StateTape(), scheme)
        F._replay_steps!(replay, 1:P.nsteps; recorder = counter)
        counted = F._TapeSizing(counter.layouts, counter.pending_slot_bytes .> 0, F._replay_m(replay))
        @test [F._counts_bytes(F._segment_counts(size, 1:s)) for s in 1:P.nsteps] ==
              [F._counts_bytes(F._segment_counts(counted, 1:s)) for s in 1:P.nsteps]
        for K in (1, 2, 4), tail in (0, 2)
            tape = AT.AdjointTape(; checkpoint = AT.StrideCheckpoint(K; resident_tail = tail))
            @test F._candidate_memory(tape, size, P.nsteps, 0) ==
                  F._candidate_memory(tape, counted, P.nsteps, 0)
        end
    end
end

@testset "CPU candidate selection with schedules and budgets" begin
    for steps in ([2, 3, 2], [1, 1, 5], [4, 4, 3]), scheme in (AT.UpwindScheme(), AT.PPMScheme())
        P = H.problem(H.Case(; name = "pick", steps, scheme))
        replay = _auto_replay(P)
        size = F._closed_tape_size(replay, P.nsteps)
        sample = F._snapshot_operator_state(replay)
        snapbytes = 2 * T._bytes_per_panel_tuple(P.m0) + F._snapshot_bytes(sample)
        F._release_snapshot_state!(sample)
        fullbytes = F._layout_total(F._segment_layout(size, 1:P.nsteps))
        for fraction in (1.1, 0.8, 0.6)
            budget = floor(Int, fullbytes * fraction)
            F._with_tape_runtime(AT.AutoTape(; budget_bytes = budget)) do runtime
                tape = F._prepare_tape!(runtime, replay, P.nsteps)
                mem = F._candidate_memory(tape, runtime.sizing, P.nsteps, snapbytes)
                @test mem.device <= budget
                @test fraction > 1 ? tape.checkpoint isa AT.FullCheckpoint : tape.checkpoint isa AT.StrideCheckpoint
                @test runtime.pool !== nothing
                if tape.checkpoint isa AT.StrideCheckpoint
                    @test tape.checkpoint.resident_tail >= 0
                    # Independent small-problem oracle: six m states per
                    # step, six code states for PPM, two prognostic states
                    # plus persistent operator state per snapshot.
                    stepbytes = 6 * T._tape_slot_bytes(P.m0) +
                        (scheme isa AT.UpwindScheme ? 0 : 6 * T._tape_slot_bytes(UInt8, P.m0))
                    feasible = [tail for K in 1:P.nsteps for tail in 0:(P.nsteps - 1)
                        if stepbytes * (min(K, P.nsteps - tail) + tail) +
                           cld(P.nsteps - tail, K) * snapbytes <= budget]
                    @test tape.checkpoint.resident_tail == maximum(feasible)
                    @test length(runtime.snapshots) == mem.nsnaps
                end
            end
        end
        F._with_tape_runtime(AT.AutoTape(; budget_bytes = 0)) do runtime
            @test_throws AT.TapeReservationError F._prepare_tape!(runtime, replay, P.nsteps)
            @test runtime.pool === nothing
        end
        # Host snapshots are the last resort when one device tape step fits
        # but device checkpoints do not.
        minimum = F._candidate_memory(AT.AdjointTape(; checkpoint = AT.StrideCheckpoint(1;
            snapshots = AT.PinnedHostCSTapeStorage())), size, P.nsteps, snapbytes).device
        request = AT.AutoTape(; budget_bytes = minimum, scratch_dir = mktempdir())
        budgets = (; device = minimum, host = typemax(Int), disk = typemax(Int), shared = false)
        choices = F._auto_candidates(request, replay, size, P.nsteps, snapbytes, budgets)
        fitting = filter(t -> F._fits_tape(t, size, P.nsteps, snapbytes, budgets), choices)
        @test first(fitting).storage isa AT.DeviceCSTapeStorage
        @test first(fitting).checkpoint.snapshots isa AT.PinnedHostCSTapeStorage
        @test findfirst(t -> t.storage isa AT.MmapCSTapeStorage, fitting) > 1
    end
end

@testset "constraints, reserve failure, and no silent growth" begin
    P = H.problem(H.Case(; name = "constraints", scheme = AT.UpwindScheme()))
    replay = _auto_replay(P)
    mktempdir() do dir
        for storage in (AT.DeviceCSTapeStorage(), AT.PinnedHostCSTapeStorage(), AT.MmapCSTapeStorage(dir)),
            checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2; resident_tail = 2,
                                                        snapshots = AT.PinnedHostCSTapeStorage()))
            request = AT.AutoTape(; storage, checkpoint)
            F._with_tape_runtime(request) do runtime
                tape = F._prepare_tape!(runtime, replay, P.nsteps)
                @test tape.storage === storage
                @test tape.checkpoint === checkpoint
                size = runtime.sizing
                @test F._prepare_tape!(runtime, replay, P.nsteps) === tape
                @test runtime.sizing === size
            end
        end
        # Failed explicit reservations throw the public typed error.
        F._with_tape_runtime(AT.AdjointTape(; storage = AT.PinnedHostCSTapeStorage(; max_bytes = 0))) do runtime
            @test_throws AT.TapeReservationError F._prepare_tape!(runtime, replay, P.nsteps)
        end
        for storage in (AT.DeviceCSTapeStorage(), AT.PinnedHostCSTapeStorage(), AT.MmapCSTapeStorage(dir))
            layout = Any[(P.m0, Float64)]
            pool = T._reserve_tape_pool(storage, layout)
            try
                T._stage_panels(pool, P.m0)
                @test_throws AT.TapeReservationExceeded T._stage_panels(pool, P.m0)
            finally
                T._release_tape_pool!(pool)
            end
        end
    end
end

@testset "CFL counting is cached across optimization evaluations" begin
    P = H.problem(H.Case(; name = "cfl", scheme = AT.UpwindScheme(), contract = false, flow = 2.4))
    replay = _auto_replay(P)
    F._with_tape_runtime(AT.AutoTape()) do runtime
        F._prepare_tape!(runtime, replay, P.nsteps)
        @test !isempty(runtime.slot_bytes_through_step)
        size = runtime.sizing
        steps = runtime.slot_bytes_through_step
        F._prepare_tape!(runtime, _auto_replay(P), P.nsteps)
        @test runtime.sizing === size
        @test runtime.slot_bytes_through_step === steps
    end
end

# Fail an actual reserve attempt (after candidate selection), rather than
# just rejecting its byte estimate. The next permitted schedule must run.
struct _FailFirstReservation <: T.AbstractCSTapeStorage
    attempts::Base.RefValue{Int}
end
F._tape_storage_tier(::_FailFirstReservation) = :device
function T._reserve_tape_pool(storage::_FailFirstReservation, layout)
    storage.attempts[] += 1
    storage.attempts[] == 1 && throw(OutOfMemoryError())
    return T._reserve_tape_pool(AT.DeviceCSTapeStorage(), layout)
end

@testset "Auto retries a failed reserve; explicit choices never change" begin
    P = H.problem(H.Case(; name = "reserve", scheme = AT.UpwindScheme()))
    replay = _auto_replay(P)
    storage = _FailFirstReservation(Ref(0))
    F._with_tape_runtime(AT.AutoTape(; storage)) do runtime
        chosen = F._prepare_tape!(runtime, replay, P.nsteps)
        @test storage.attempts[] == 2 + (runtime.tail_pool !== nothing)
        @test chosen.storage === storage
        @test chosen.checkpoint isa AT.StrideCheckpoint
        @test !isempty(runtime.pool.banks)
    end
    storage = _FailFirstReservation(Ref(0))
    F._with_tape_runtime(AT.AdjointTape(; storage)) do runtime
        @test_throws AT.TapeReservationError F._prepare_tape!(runtime, replay, P.nsteps)
        @test storage.attempts[] == 1
        @test runtime.pool === nothing && runtime.tail_pool === nothing
    end
end

@testset "AutoTape AirMass constraint resolves once and reuses its recording" begin
    P = H.problem(H.Case(; name = "air_auto", scheme = AT.UpwindScheme()))
    F._with_tape_runtime(AT.AutoTape(; content = AT.AirMassTape(), storage = AT.PinnedHostCSTapeStorage())) do runtime
        replay = _auto_replay(P)
        run = F._begin_taped_run(replay, runtime, P.nsteps)
        try
            F._run_forward!(run, replay)
        finally
            F._end_taped_run!(run)
        end
        @test runtime.tape.content === AT.AirMassTape()
        @test runtime.tape.checkpoint === AT.FullCheckpoint()
        ops = runtime.held.ops
        replay = _auto_replay(P)
        reused = F._begin_taped_run(replay, runtime, P.nsteps)
        @test reused isa F._ReplayedTapedRun
        @test reused.ops === ops
        F._run_forward!(reused, replay)
        F._end_taped_run!(reused)
    end
    F._with_tape_runtime(AT.AutoTape(; checkpoint = AT.FullCheckpoint(), budget_bytes = 0)) do runtime
        @test_throws AT.TapeReservationError F._prepare_tape!(runtime, _auto_replay(P), P.nsteps)
        @test runtime.pool === nothing
    end
end

@testset "reclaimable cgroup cache and host-wide MemAvailable" begin
    mktempdir() do dir
        GiB = 1 << 30
        root = joinpath(dir, "cgroup")
        leaf = joinpath(root, "job", "task")
        mkpath(leaf)
        meminfo, cgroup = joinpath(dir, "meminfo"), joinpath(dir, "self_cgroup")
        write(meminfo, "MemTotal: $(64GiB ÷ 1024) kB\nMemAvailable: $(40GiB ÷ 1024) kB\n")
        write(cgroup, "0::/job/task\n")
        for (path, cap, used, cache) in ((root, 32GiB, 30GiB, 20GiB),
                (dirname(leaf), 16GiB, 15GiB, 10GiB))
            write(joinpath(path, "memory.max"), string(cap))
            write(joinpath(path, "memory.current"), string(used))
            write(joinpath(path, "memory.stat"), "anon $(used-cache)\nfile $(cache)\ninactive_file $(cache)\n")
        end
        write(joinpath(leaf, "memory.max"), "max")
        write(joinpath(leaf, "memory.current"), "0")
        inputs = (; meminfo, cgroup, cgroup_root = root)
        @test T._host_tape_memory(; inputs...) == (11GiB, 16GiB)
        @test T._host_tape_budget(; inputs...) == 3GiB
        write(meminfo, "MemTotal: $(64GiB ÷ 1024) kB\nMemAvailable: $(9GiB ÷ 1024) kB\n")
        @test T._host_tape_memory(; inputs...) == (9GiB, 16GiB)
        # Missing stats conservatively count all current memory as used.
        rm(joinpath(dirname(leaf), "memory.stat"))
        @test first(T._host_tape_memory(; inputs...)) == GiB
    end
end

@testset "Auto margin scales on CPU; explicit and overridden budgets bypass it" begin
    GiB = 1 << 30
    @test F._auto_memory_budgets(CPU(), 6GiB, 8GiB, 0, AT.AutoTape(), 0) ==
        (6GiB - cld(8GiB, 10), 6GiB - cld(8GiB, 10))
    @test F._auto_memory_budgets(CPU(), 0, 8GiB, GiB,
        AT.AutoTape(; budget_bytes = 1_000_000), 0) == (1_000_000, 1_000_000)
    P = H.problem(H.Case(; name = "budget_override", scheme = AT.UpwindScheme()))
    replay = _auto_replay(P)
    explicit = AT.AdjointTape()
    budgets = F._tape_budgets(replay, P.nsteps, 1, explicit)
    @test budgets.device == budgets.host == typemax(Int)
    F._with_tape_runtime(explicit) do runtime
        @test F._prepare_tape!(runtime, replay, P.nsteps) === explicit
    end
    size = F._closed_tape_size(replay, P.nsteps)
    bytes = F._layout_total(F._segment_layout(size, 1:P.nsteps))
    request = AT.AutoTape(; budget_bytes = bytes)
    budgets = F._tape_budgets(replay, P.nsteps, 1, request)
    @test budgets.device == budgets.host == bytes
    F._with_tape_runtime(request) do runtime
        @test F._prepare_tape!(runtime, replay, P.nsteps).checkpoint isa AT.FullCheckpoint
    end
    # CPU query dispatch accepts wrappers with a CPU backend.
    @test T._device_tape_memory(view(P.m0[1], :, :, :)) isa Tuple{Int, Int}
end

struct _BrokenReservation <: T.AbstractCSTapeStorage end
F._tape_storage_tier(::_BrokenReservation) = :device
T._reserve_tape_pool(::_BrokenReservation, _layout) = throw(DomainError(:reservation_bug))
@testset "non-reservation errors propagate for Auto and explicit tapes" begin
    P = H.problem(H.Case(; name = "bug", scheme = AT.UpwindScheme()))
    for request in (AT.AutoTape(; storage = _BrokenReservation()),
                    AT.AdjointTape(; storage = _BrokenReservation()))
        F._with_tape_runtime(request) do runtime
            @test_throws DomainError F._prepare_tape!(runtime, _auto_replay(P), P.nsteps)
        end
    end
end

@testset "resident tail follows the selected storage and window arithmetic" begin
    P = H.problem(H.Case(; name = "tail_storage", scheme = AT.UpwindScheme()))
    mktempdir() do dir
        for storage in (AT.DeviceCSTapeStorage(), AT.PinnedHostCSTapeStorage(), AT.MmapCSTapeStorage(dir))
            checkpoint = AT.StrideCheckpoint(2; resident_tail = 2)
            F._with_tape_runtime(AT.AdjointTape(; storage, checkpoint)) do runtime
                F._prepare_tape!(runtime, _auto_replay(P), P.nsteps)
                @test typeof(runtime.tail_pool) === typeof(runtime.pool)
                @test F._candidate_memory(runtime.tape, runtime.sizing, P.nsteps, 0).tailbytes > 0
            end
        end
    end
    for n in (0, 1, 7), K in (1, 2, 10), tail in (0, 1, 3, 10)
        schedule = AT.StrideCheckpoint(K; resident_tail = tail)
        windows = [AT.checkpoint_window_range(schedule, w, n)
            for w in 1:AT.checkpoint_window_count(schedule, n)]
        @test reduce(vcat, collect.(windows); init = Int[]) == collect(1:n)
        @test F._head_windows(schedule, n) == (tail == 0 ? windows : windows[1:end-1])
    end
    @test AT.AutoTape(; checkpoint = AT.StrideSearch()).checkpoint isa AT.StrideSearch
    @test_throws TypeError AT.AdjointTape(; checkpoint = AT.StrideSearch())
end

mutable struct _ReleaseProbe
    released::Bool
    fail::Bool
end
function T._release_tape_pool!(pool::_ReleaseProbe)
    pool.released = true
    pool.fail && throw(DomainError(:release_bug))
    return nothing
end
@testset "runtime releases every pool after a release error" begin
    runtime = F._TapeRuntime(AT.AdjointTape())
    pools = (_ReleaseProbe(false, true), _ReleaseProbe(false, false), _ReleaseProbe(false, false))
    runtime.pool, runtime.tail_pool, runtime.snapshot_pool = pools
    @test_throws DomainError F._release_tape_runtime!(runtime)
    @test all(p -> p.released, pools)
    @test runtime.pool === runtime.tail_pool === runtime.snapshot_pool === nothing
    @test F._release_tape_runtime!(runtime) === nothing
end

@testset "reset targets are validated before tape reservation" begin
    P = H.problem(H.Case(; name = "reset_shape", reset = :preserve_vmr, scheme = AT.UpwindScheme()))
    for targets in (map(t -> map(a -> Float32.(a), t), P.schedule.window_start_air_mass),
                    map(t -> map(a -> zeros(eltype(a), 1, 1, 1), t), P.schedule.window_start_air_mass))
        schedule = AT.CSStepSchedule(P.schedule.plan;
            steps_per_window = P.schedule.steps_per_window, window_dt = P.schedule.window_dt,
            binary_substeps = P.schedule.binary_substeps, window_start_air_mass = targets,
            window_end_air_mass = P.schedule.window_end_air_mass)
        @test_throws ArgumentError _auto_replay(merge(P, (; schedule)))
    end
end
