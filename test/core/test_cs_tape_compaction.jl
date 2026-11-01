#!/usr/bin/env julia
#
# What a split-sweep adjoint tape stages (`StateTape`), on the rough problems
# of the bitwise harness (`helpers/cs_adjoint_tape_harness.jl`):
#
#  1. Records share air masses that are already on the tape: the palindrome
#     center's diffusion records and the reverse half's first Z sweep read one
#     slot; a convection record reads the next step's first sweep's slot, or the
#     old air mass of a following `:preserve_vmr` reset, and stages its own only
#     before a `:preserve_tracer_mass` reset or at the end of the recording. A
#     limited PPM sweep stages its limiter branch codes (six `UInt8` panels
#     shaped like the air mass) instead of the tracer mass. Lin–Rood records
#     share nothing.
#  2. The counting recorder and `cs_tape_byte_estimate` count exactly what the
#     recorder stages; a stride window's count bounds its recording.
#  3. Bytes per step under the binary window contract with substep convection
#     and diffusion: six air masses (the linear schemes; main staged eight)
#     plus six code panels for limited PPM (main: fourteen air masses).
#  4. An `AirMassTape` optimization whose air-mass resets refresh the halos of
#     fresh targets in place reuses its first tape and equals `StateTape`.
#
# That the shared slots and the codes give the gradients of the uncompacted
# tape bit for bit is the bitwise gate (`test_cs_adjoint_tape_bitwise.jl`).
using Test
using KernelAbstractions: get_backend, synchronize

include(joinpath(@__DIR__, "..", "helpers", "cs_adjoint_tape_harness.jl"))
using .CSAdjointTapeHarness: Case
const H = CSAdjointTapeHarness
const AT = H.AT
const FP = AT.Adjoints.Footprint
const TP = AT.Tape

# The replay of a harness case (at its initial state).
_replay(P) = first(FP._cs_replay_for_call(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh;
                                          H._common(P)..., base_emission_rates = P.rates))

# The device tape of steps `steps` of a fresh replay, after running the steps
# before them without a recorder; and the pool it was recorded into.
function _record(P, steps = 1:P.nsteps)
    replay = _replay(P)
    runtime = FP._TapeRuntime(AT.AdjointTape())
    size = FP._size_tape!(runtime, replay, P.nsteps)
    pool = TP._reserve_tape_pool(TP.DeviceCSTapeStorage(),
        filter(TP._layout_pooled, FP._segment_layout(size, steps)))
    FP._replay_steps!(replay, 1:(first(steps) - 1))
    return FP._record_cs_tape!(replay, steps, TP.StateTape(), pool), pool
end

# Staged bytes and mmap slot bytes of a recorded device pool.
_pool_bytes(pool) = sum(TP._bytes_per_panel_tuple, pool.slabs; init = 0)
_pool_slot_bytes(pool) = sum(TP._tape_slot_bytes, pool.slabs; init = 0)

_slot(x) = x isa TP.CSTapeSlot ? x.panels : nothing
_slots(op::TP._CSSweepRecord) = (_slot(op.panels_m), _slot(op.panels_tracer))
_slots(op::Union{TP._CSDiffusionRecord, TP._CSConvectionRecord}) = (_slot(op.panels_m),)
_slots(op::TP._CSAirMassResetRecord) = (_slot(op.panels_old_m), _slot(op.panels_new_m))
_slots(_op) = ()

const SPLIT_CASES = [
    Case(name = "monotone_tm5sub_split_vmr", scheme = AT.PPMScheme(AT.MonotoneLimiter()), conv = :tm5,
         diffusion = :split, cadence = :substep, reset = :preserve_vmr),
    Case(name = "cw84_cmfmcwin_boundary_mass", FT = Float32, scheme = AT.PPMScheme(AT.CW84Limiter()),
         conv = :cmfmc, diffusion = :boundary, reset = :preserve_tracer_mass),
    Case(name = "upwind_tm5sub_boundary_mass_cfl", scheme = AT.UpwindScheme(), conv = :tm5,
         diffusion = :boundary, cadence = :substep, contract = false, reset = :preserve_tracer_mass,
         flow = 0.6),
    Case(name = "slopes_matrixwin_none_vmr", FT = Float32, scheme = AT.SlopesScheme(AT.NoLimiter()),
         conv = :cmfmc_matrix, reset = :preserve_vmr),
    Case(name = "ppm_none_cmfmcsub_split_none", scheme = AT.PPMScheme(AT.NoLimiter()), conv = :cmfmc,
         diffusion = :split, cadence = :substep),
]
const LINROOD_CASE = Case(name = "linrood5_tm5win_boundary_vmr", scheme = AT.LinRoodPPMScheme(5),
                          conv = :tm5, diffusion = :boundary, reset = :preserve_vmr)

@testset "split-sweep records share the air masses on the tape ($(c.name))" for c in SPLIT_CASES
    P = H.problem(c)
    ops, pool = _record(P)
    limited = H._limited(c.scheme)
    @test all(bank -> bank.cursor == length(bank.slabs), values(pool.banks))
    # Every slab of the pool is one staged slot, read by at least one record.
    read = IdDict{Any, Int}()
    foreach(op -> foreach(s -> s === nothing || (read[s] = get(read, s, 0) + 1), _slots(op)), ops)
    @test length(read) == length(pool.slabs)
    @test all(s -> haskey(read, s), pool.slabs)
    staged_convection = 0
    for (i, op) in enumerate(ops)
        if op isa TP._CSSweepRecord
            m = op.panels_m.panels
            if limited
                codes = op.panels_tracer.panels
                @test eltype(codes[1]) == UInt8 && all(p -> size(codes[p]) == size(m[p]), 1:6)
            else
                @test op.panels_tracer === nothing
            end
        elseif op isa TP._CSDiffusionRecord
            # The reverse half's first sweep (Z) reads the center's slot.
            j = findnext(o -> o isa TP._CSSweepRecord, ops, i)
            @test ops[j].direction === :z && ops[j].panels_m === op.panels_m
        elseif op isa TP._CSConvectionRecord
            j = findnext(o -> o isa TP._CSSweepRecord || o isa TP._CSAirMassResetRecord, ops, i)
            next = j === nothing ? nothing : ops[j]
            if next isa TP._CSSweepRecord
                @test next.direction === :x && op.panels_m === next.panels_m
            elseif next isa TP._CSAirMassResetRecord && next.mode === :preserve_vmr
                @test op.panels_m === next.panels_old_m
            else    # before a :preserve_tracer_mass reset, or the last one
                @test read[op.panels_m.panels] == 1
                staged_convection += 1
            end
        end
    end
    # The estimate's plan walk finds the same convection records.
    replay = _replay(P)
    @test staged_convection ==
          FP._staged_convection_records(AT.Operators.Advection.CSSplitSweepStyle(), replay.plan, replay.steps)
end

@testset "Lin–Rood records stage their own air masses" begin
    P = H.problem(LINROOD_CASE)
    ops, pool = _record(P)
    slots = [s for op in ops for s in _slots(op) if s !== nothing]
    diffusion_slots = Set(objectid(op.panels_m.panels) for op in ops if op isa TP._CSDiffusionRecord)
    # One slot per record (a center's two half solves of a split coupling would
    # share one; this case's boundary coupling has one solve per center).
    @test length(unique(objectid, slots)) == length(slots) == length(pool.slabs)
    @test !isempty(diffusion_slots)
    FP._release_records!(ops)
end

@testset "the count is the tape ($(c.name))" for c in [SPLIT_CASES; LINROOD_CASE]
    P = H.problem(c)
    ops, pool = _record(P)
    counter = FP._CSCountingRecorder(TP.StateTape(), c.scheme)
    FP._replay_steps!(_replay(P), 1:P.nsteps; recorder = counter)
    FP._finish_recording!(counter)
    @test counter.slot_bytes == _pool_slot_bytes(pool)
    linrood = c.scheme isa AT.LinRoodPPMScheme
    if !linrood    # Lin–Rood horizontal records are copies outside the pool
        @test counter.state == length(pool.slabs)
        @test counter.state_bytes == _pool_bytes(pool)
        est = AT.cs_tape_byte_estimate(P.m0, P.am, P.bm, P.cm, P.mesh, c.scheme; schedule = P.schedule,
                                       (k => v for (k, v) in pairs(H.physics_kwargs(P))
                                        if k in (:diffusion_op, :convection_op))...)
        @test est.state_records == length(pool.slabs)
        @test est.state_bytes == _pool_bytes(pool)
        @test est.sweep_records == count(op -> op isa TP._CSSweepRecord, ops)
        @test est.convection_records == count(op -> op isa TP._CSConvectionRecord, ops)
    end
    FP._release_records!(ops)
    # Stride windows: the count of a window bounds its recording (by at most
    # one air mass) and is exact from step 1.
    runtime = FP._TapeRuntime(AT.AdjointTape())
    FP._store_tape_count!(runtime, counter)
    for steps in (1:3, 3:5, 4:P.nsteps, 6:P.nsteps)
        window_ops, window_pool = _record(P, steps)
        actual = _pool_slot_bytes(window_pool)
        counted = FP._counted_slot_bytes(runtime, steps)
        @test actual <= counted <= actual + TP._tape_slot_bytes(P.m0)
        first(steps) == 1 && @test counted == actual
        FP._release_records!(window_ops)
    end
end

@testset "bytes per step: $(H._limited(scheme) ? nameof(typeof(scheme.limiter)) : nameof(typeof(scheme))), $FT" for
        scheme in (AT.UpwindScheme(), AT.PPMScheme(AT.MonotoneLimiter()), AT.PPMScheme(AT.CW84Limiter())),
        FT in (Float64, Float32)
    # Binary window contract (one palindrome pass per step), convection and
    # diffusion every step, no reset: six air masses per step, the last
    # convection's own, and six code panels per step for limited PPM.
    c = Case(; name = "bytes", FT, scheme, conv = :tm5, diffusion = :split, cadence = :substep)
    P = H.problem(c)
    _, pool = _record(P)
    S = TP._bytes_per_panel_tuple(P.m0)
    n = P.nsteps
    codes = H._limited(scheme) ? 6n * S ÷ sizeof(FT) : 0
    @test _pool_bytes(pool) == (6n + 1) * S + codes
    @test length(pool.slabs) == 6n + 1 + (H._limited(scheme) ? 6n : 0)
end

# An `AirMassTape` optimization on fresh reset targets: the first forward run
# records with the targets' halos as given, the reset refreshes them in place,
# and the later evaluations reuse the tape. They must equal `StateTape`, which
# records every evaluation, bit for bit (one thread). Each run gets its own
# problem, so no earlier run refreshed its targets.
@testset "AirMassTape reuse with air-mass resets on fresh targets ($(c.name))" for c in [
        Case(name = "upwind_tm5win_vmr", scheme = AT.UpwindScheme(), conv = :tm5, reset = :preserve_vmr),
        Case(name = "slopes_cmfmcsub_split_mass", scheme = AT.SlopesScheme(AT.NoLimiter()), conv = :cmfmc,
             diffusion = :split, cadence = :substep, reset = :preserve_tracer_mass)]
    function optimize(content)
        P = H.problem(c)
        probe = AT.cs_surface_flux_4dvar(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh, H._observations(P),
                                         H._controls(P); H._common(H.problem(c))...)
        step = H._norm([x.value for x in H._controls(P)]) / (10 * H._norm(probe.gradients))
        optimizer = AT.CSGradientDescent(; iterations = 2, initial_step = step, min_step = step,
                                         line_search = false)
        return AT.cs_surface_flux_4dvar_optimize(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh, H._observations(P),
                                                 H._controls(P); optimizer, H._common(P)...,
                                                 tape = AT.AdjointTape(; content))
    end
    reused, fresh = optimize(AT.AirMassTape()), optimize(AT.StateTape())
    @test length(reused.cost_history) == 3
    if Threads.nthreads() == 1     # CPU atomics are reproducible on one thread only
        @test reused.cost_history == fresh.cost_history
        @test reused.last.gradients == fresh.last.gradients
    else
        @test reused.cost_history ≈ fresh.cost_history rtol = 1e-12
    end
end


@testset "met checksum reductions preserve bit patterns and cell positions" begin
    for FT in (Float32, Float64, Int32, UInt64)
        a = reshape(FT.(mod.(0:40039, 97)), 13, 22, 140)
        if FT <: AbstractFloat
            a[1:4] .= (FT(0.0), FT(-0.0), FT(Inf), FT(NaN))
        end
        arrays = Any[a, view(a, 2:11, 3:20, :), FT[], reshape(FT[1], 1, 1, 1)]
        partials = zeros(UInt64, 128, length(arrays))
        result = zeros(UInt64, length(arrays))
        backend = get_backend(a)
        for (slot, x) in enumerate(arrays)
            ngroups = min(128, max(1, cld(length(x), 256)))
            FP._met_checksum_partials!(backend, 256)(partials, x, slot, ngroups;
                                                       ndrange = 256 * ngroups)
        end
        FP._met_checksum_finish!(backend, 256)(result, partials; ndrange = length(arrays))
        synchronize(backend)
        # The device algorithm (executed here on KA's CPU backend) agrees
        # exactly with the scalar host reduction, including strided views.
        @test result == map(FP._content_checksum, arrays)
        @test FP._met_checksums(arrays) ==
              [hash((size(x), eltype(x), result[i])) for (i, x) in enumerate(arrays)]
        permuted = reverse(a; dims = 3)
        @test FP._content_checksum(a) != FP._content_checksum(permuted)
    end
end

@testset "AirMassTape hold checks the recorded payload" begin
    for c in (SPLIT_CASES[1], LINROOD_CASE, SPLIT_CASES[3])
        P = H.problem(c)
        ops, pool = _record(P)
        runtime = FP._TapeRuntime(AT.AdjointTape(; content = AT.AirMassTape()))
        runtime.pool = pool
        try
            identity = FP._tape_identity(_replay(P), P.nsteps)
            if H._limited(c.scheme) || c === LINROOD_CASE
                @test_throws ArgumentError FP._hold_air_mass_tape!(runtime, ops, identity)
                @test runtime.held === nothing
            else
                FP._hold_air_mass_tape!(runtime, ops, identity)
                @test runtime.held.ops === ops
            end
        finally
            # These StateTape records own their Lin–Rood arrays, even when
            # the air-mass hold rejected them.
            runtime.held = nothing
            FP._release_records!(ops)
            FP._release_tape_runtime!(runtime)
        end
    end
end
