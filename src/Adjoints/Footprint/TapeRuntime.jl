# ---------------------------------------------------------------------------
# The runtime of an `AdjointTape`: what one reverse-pass call (or one
# `cs_surface_flux_4dvar_optimize` call, for all its evaluations) records into.
#
#   * The tape pool, opened from `tape.storage` before the first tape
#     (`_open_tape_pool`), rewound before each further tape (every stride
#     window, every later 4D-Var evaluation) so they all reuse its memory,
#     and released explicitly when the call returns (`_release_tape_runtime!`):
#     device arrays and pinned host chunks are freed now, not at a later GC,
#     and an mmap file is unmapped and closed.
#   * The counted tape bytes per replay step (`_CSCountingRecorder`), which an
#     mmap pool reserves. They depend on the meteorology and the step plan
#     only (the air-mass trajectory and its CFL subcycles do not depend on the
#     tracer or the emissions), so a runtime counts once: the later
#     evaluations of an optimization skip the counting run.
#   * With `AirMassTape` content, the tape its first forward run recorded and
#     the identity of that run (`_TapeIdentity`), which every later forward
#     run reuses instead of recording after checking its own identity
#     against it (`_held_air_mass_tape`).
#
# The drivers (`StrideCheckpoint.jl`) take the runtime; the entry points build
# it with `_with_tape_runtime`.
# ---------------------------------------------------------------------------

mutable struct _TapeRuntime{T <: AdjointTape}
    const tape :: T
    pool :: Any                             # `nothing` until the first tape
    slot_bytes_through_step :: Vector{Int}  # counted bytes through each step (empty: not counted)
    pending_slot_bytes :: Vector{Int}       # bytes a recording that ends after the step adds
    held :: Any                             # `AirMassTape`: the recorded `_HeldAirMassTape`, or nothing
end

_TapeRuntime(tape::AdjointTape) = _TapeRuntime(tape, nothing, Int[], Int[], nothing)

"""
    _with_tape_runtime(f, tape::AdjointTape)

`f(runtime)` with a fresh runtime of `tape`, whose pool is released when `f`
returns or throws.
"""
function _with_tape_runtime(f, tape::AdjointTape)
    runtime = _TapeRuntime(tape)
    try
        return f(runtime)
    finally
        _release_tape_runtime!(runtime)
    end
end

function _release_tape_runtime!(runtime::_TapeRuntime)
    held = runtime.held
    runtime.held = nothing
    held === nothing || _release_records!(held.ops)
    pool = runtime.pool
    runtime.pool = nothing
    pool === nothing || _release_tape_pool!(pool)
    return nothing
end

"""
    _tape_pool!(runtime, nbytes) -> pool

The pool the next tape of `runtime` records into, rewound: the open pool when
it holds `nbytes` (counted bytes; 0 unless the storage reserves them), else a
new one. Device and host pools grow on demand, and a runtime's counted bytes
are fixed by its first count, so in practice only an mmap file opened for a
smaller tape than a later one is replaced (released first).
"""
function _tape_pool!(runtime::_TapeRuntime, nbytes::Integer)
    pool = runtime.pool
    pool !== nothing && _holds_tape_bytes(pool, nbytes) && return _rewind_tape_pool!(pool)
    _release_tape_runtime!(runtime)
    return runtime.pool = _open_tape_pool(runtime.tape.storage, nbytes)
end

# The counted bytes of a recording of `steps`: the bytes the counting run
# (from step 1) staged in them, plus what a recording ending after
# `last(steps)` stages at its end (a waiting convection record's air mass).
# Exact for a recording from step 1. A later window may stage one state less
# (a convection air mass the counting run staged at a `:preserve_tracer_mass`
# reset in its first step belongs to the previous window): an upper bound.
function _counted_slot_bytes(runtime::_TapeRuntime, steps::AbstractUnitRange)
    through = runtime.slot_bytes_through_step
    isempty(through) && return 0
    before = first(steps) == 1 ? 0 : through[first(steps) - 1]
    return through[last(steps)] - before + runtime.pending_slot_bytes[last(steps)]
end

# The recorder that counts the tape bytes during a forward run from step 1,
# when the storage reserves them and the runtime has not counted them yet.
_tape_size_counter(runtime::_TapeRuntime, replay::_CSReplay, taped_steps::Int) =
    _needs_tape_size(runtime.tape.storage) && length(runtime.slot_bytes_through_step) < taped_steps ?
        _CSCountingRecorder(runtime.tape.content, replay.advection) : NoStepRecorder()

_store_tape_count!(::_TapeRuntime, ::NoStepRecorder) = nothing
function _store_tape_count!(runtime::_TapeRuntime, counter::_CSCountingRecorder)
    runtime.slot_bytes_through_step = counter.slot_bytes_through_step
    runtime.pending_slot_bytes = counter.pending_slot_bytes
    return nothing
end

# The bytes of the tape of `steps` for `runtime`'s storage: 0 when it does not
# reserve them; else the count, made now by a counting run of the replay from
# step 1 (restored to its initial state afterwards, which reproduces the
# recording run bit for bit) unless the runtime counted these steps before.
function _counted_tape_bytes!(runtime::_TapeRuntime, replay::_CSReplay, steps::AbstractUnitRange)
    _needs_tape_size(runtime.tape.storage) || return 0
    counter = _tape_size_counter(runtime, replay, last(steps))
    if counter isa _CSCountingRecorder
        initial = _replay_snapshot(replay)
        _replay_steps!(replay, 1:last(steps); recorder = counter)
        _replay_restore!(replay, initial)
        _release_snapshot!(initial)
        _store_tape_count!(runtime, counter)
    end
    return _counted_slot_bytes(runtime, steps)
end

# ---------------------------------------------------------------------------
# AirMassTape: one recording per runtime, reused after an identity check
# ---------------------------------------------------------------------------

"""
    _TapeIdentity

What an `AirMassTape` recording depends on (a reuse compares `grid`, `steps`,
`window`, `scheme`, then `met`):

- `met`: content checksums (a position-weighted wrapping sum of bit patterns,
  the same on every backend and thread count) of the met inputs the records read or the
  air-mass trajectory follows: the initial air mass, every step's face fluxes,
  convection forcing and diffusion layer thickness, and the interiors of the
  air-mass reset targets (whose halos the forward run refreshes in place),
  each array once, in step order;
- `window`: met window, first/last flags, step length and clock of every
  taped step, and the window length;
- `scheme`: the advection scheme and the step plan (values), the binary
  subcycle contract and whether the plan has a surface flux, and the
  diffusion and convection operators, the caller's diffusion workspaces and
  meteorology (objects the records hold, compared by identity: the contents
  of a Kz field are not checksummed);
- `grid`: `Nc`, `Hp`, `Nz` and the float type;
- `steps`: the number of taped steps.

The pool generation of the recording is held next to it (`_HeldAirMassTape`).
Within one call or optimization the keywords that fix the replay (the CFL
limit, the flux scale) are the same for every forward run; the identity
guards against a recording that no longer fits (an input changed in place, a
rewound pool), not against a different call.
"""
struct _TapeIdentity{W, S, G}
    met    :: Vector{UInt64}
    window :: W
    scheme :: S
    grid   :: G
    steps  :: Int
end

# A recorded air-mass tape, the identity of its forward run and the pool
# generation its slots carry.
struct _HeldAirMassTape{I <: _TapeIdentity}
    ops        :: Vector{Any}
    identity   :: I
    generation :: Int
end

_bits64(x::Float64) = reinterpret(UInt64, x)
_bits64(x::Float32) = UInt64(reinterpret(UInt32, x))
_bits64(x::Float16) = UInt64(reinterpret(UInt16, x))
_bits64(x::Integer) = x % UInt64

# Position-weighted wrapping sum: integer addition is independent of reduction
# order, while moving unequal bit patterns between cells changes the checksum.
# This is a within-runtime guard, not a collision-free content fingerprint.
@inline function _checksum_weight(i)
    x = UInt64(i) + 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end
@inline _checksum_term(a, i) = _bits64(a[i]) * _checksum_weight(i)
_content_checksum(a::AbstractArray) =
    mapreduce(i -> _checksum_term(a, i), +, 1:length(a); init = zero(UInt64))

# Each met array writes one column of partial sums, without a host read or
# barrier. Linear indexing also handles non-contiguous reset-interior views.
# At most 128 workgroups per array; each lane sums a disjoint strided subset.
@kernel function _met_checksum_partials!(partials, @Const(a), slot, ngroups)
    lane = @index(Local, Linear)
    group = @index(Group, Linear)
    scratch = @localmem UInt64 (256,)
    value = zero(UInt64)
    for i in ((group - 1) * 256 + lane):(ngroups * 256):length(a)
        @inbounds value += _checksum_term(a, i)
    end
    scratch[lane] = value
    @synchronize
    for offset in (128, 64, 32, 16, 8, 4, 2, 1)
        if lane <= offset
            @inbounds scratch[lane] += scratch[lane + offset]
        end
        @synchronize
    end
    if lane == 1
        @inbounds partials[group, slot] = scratch[1]
    end
end

@kernel function _met_checksum_finish!(sums, @Const(partials))
    slot = @index(Global, Linear)
    value = zero(UInt64)
    for group in axes(partials, 1)
        @inbounds value += partials[group, slot]
    end
    @inbounds sums[slot] = value
end

# Host arrays are reduced on the host. All device arrays feed one result
# vector; its single D2H copy is the only host synchronization of the check.
function _met_checksums(arrays)
    sums = Vector{UInt64}(undef, length(arrays))
    device = Int[]
    for (i, a) in enumerate(arrays)
        if get_backend(a) isa KACPU
            sums[i] = _content_checksum(a)
        else
            push!(device, i)
        end
    end
    if !isempty(device)
        template = arrays[first(device)]
        backend = get_backend(template)
        partials = similar(template, UInt64, 128, length(device))
        device_sums = similar(template, UInt64, length(device))
        try
            fill!(partials, zero(UInt64))
            for (slot, i) in enumerate(device)
                a = arrays[i]
                ngroups = min(128, max(1, cld(length(a), 256)))
                _met_checksum_partials!(backend, 256)(partials, a, slot, ngroups;
                                                       ndrange = 256 * ngroups)
            end
            _met_checksum_finish!(backend, 256)(device_sums, partials; ndrange = length(device))
            sums[device] = Array(device_sums)
        finally
            _free_tape_array!(device_sums)
            _free_tape_array!(partials)
        end
    end
    return UInt64[hash((size(a), eltype(a), sums[i])) for (i, a) in enumerate(arrays)]
end

# The float arrays of a met input (a panel tuple, a flux state, a convection
# forcing, a vector of reset targets), each checksummed once per replay, in a
# fixed order.
function _push_met_arrays!(arrays::Vector{Any}, seen::IdDict{Any, Nothing}, a::AbstractArray)
    eltype(a) <: Union{Base.IEEEFloat, Integer} ||
        return (foreach(x -> _push_met_arrays!(arrays, seen, x), a); arrays)   # arrays of arrays
    haskey(seen, a) && return arrays
    seen[a] = nothing
    push!(arrays, a)
    return arrays
end
_push_met_arrays!(arrays, seen, x::Union{Tuple, NamedTuple}) =
    (foreach(y -> _push_met_arrays!(arrays, seen, y), x); arrays)
_push_met_arrays!(arrays, seen, f::CubedSphereFaceFluxState) =
    _push_met_arrays!(arrays, seen, (f.am, f.bm, f.cm))

# The interiors of the air-mass reset targets (a vector of six-panel tuples,
# one per window). A reset refreshes its target's halos in place before it
# installs it (`fill_panel_halos!(target; dir = 1)`, idempotent), so a later
# forward run sees refreshed halos where the recording may have seen others;
# the reset's reverse multiplies the target's halos only by the adjoint state
# the following halo record has just zeroed there, so the tape does not
# depend on them.
function _push_reset_target_arrays!(arrays::Vector{Any}, seen::IdDict{Any, Nothing}, targets,
                                       mesh::CubedSphereMesh)
    targets === nothing && return arrays
    I = (mesh.Hp + 1):(mesh.Hp + mesh.Nc)
    for target in targets, a in target
        haskey(seen, a) && continue
        seen[a] = nothing
        interior = view(a, I, I, :)
        push!(arrays, interior)
    end
    return arrays
end
_push_met_arrays!(arrays, seen, f::ConvectionForcing) =
    _push_met_arrays!(arrays, seen, (f.cmfmc, f.dtrain, f.tm5_fields, f.cloud_base))
_push_met_arrays!(arrays, _seen, ::Nothing) = arrays

"""
    _tape_identity(replay, taped_steps) -> _TapeIdentity

The identity of a forward run of `replay` (at its initial state) that tapes
steps `1:taped_steps`.
"""
function _tape_identity(replay::_CSReplay, taped_steps::Int)
    arrays = Any[]
    seen = IdDict{Any, Nothing}()
    _push_met_arrays!(arrays, seen, _replay_m(replay))
    for s in 1:taped_steps
        _push_met_arrays!(arrays, seen, replay.fluxes[s])
        _push_met_arrays!(arrays, seen, _forcing_at(replay.convection_forcing, s))
        diffusion_ws = _sequence_at(replay.diffusion_user_ws, s)
        diffusion_ws === nothing || _push_met_arrays!(arrays, seen, diffusion_ws.layer_thickness)
    end
    _push_reset_target_arrays!(arrays, seen, replay.start_targets, replay.mesh)
    _push_reset_target_arrays!(arrays, seen, replay.end_targets, replay.mesh)
    steps = replay.steps
    n = 1:taped_steps
    window = (steps.window[n], steps.first[n], steps.last[n], steps.dt[n], steps.start_time[n],
              steps.end_time[n], steps.window_dt)
    scheme = (replay.advection, replay.plan, replay.meteo.binary_substeps, _has_surface_flux(replay),
              objectid(replay.diffusion_op), objectid(replay.convection),
              objectid(replay.diffusion_user_ws), objectid(replay.meteo.diffusion_meteo))
    m = _replay_m(replay)
    grid = (replay.mesh.Nc, replay.mesh.Hp, size(m[1], 3), eltype(m[1]))
    return _TapeIdentity(_met_checksums(arrays), window, scheme, grid, taped_steps)
end

const _TAPE_IDENTITY_FIELDS = (
    grid = "grid (Nc, Hp, Nz or float type)",
    steps = "number of taped steps",
    window = "time windows (met window, step length or clock of the taped steps)",
    scheme = "scheme, step plan or operators",
    met = "met inputs (the contents of the initial air mass, face fluxes, convection forcing, " *
          "diffusion layer thickness or air-mass reset targets)")

"""
    _held_air_mass_tape(runtime, identity) -> Union{Nothing, Vector{Any}}

The tape an `AirMassTape` runtime recorded, for a forward run with
`identity`: `nothing` before the first recording, else its records after
checking that `identity` equals the recording's and that the pool still has
the recording's generation. A difference is an `ArgumentError` naming it.
"""
function _held_air_mass_tape(runtime::_TapeRuntime, identity::_TapeIdentity)
    held = runtime.held
    held === nothing && return nothing
    for (field, what) in pairs(_TAPE_IDENTITY_FIELDS)
        isequal(getfield(held.identity, field), getfield(identity, field)) || throw(ArgumentError(
            "the AirMassTape recorded by this call's first forward run does not fit this forward " *
            "run: its $what differ. A recorded air-mass tape is reused only for the same " *
            "meteorology, windows, scheme, operators and grid."))
    end
    pool = runtime.pool
    pool !== nothing && !pool.released && pool.generation == held.generation || throw(ArgumentError(
        "the AirMassTape recorded by this call's first forward run is stale: its tape pool was " *
        "rewound or released since (generation $(held.generation), now " *
        "$(pool === nothing ? "released" : pool.generation))."))
    return held.ops
end

# Hold the tape a runtime's first `AirMassTape` forward run recorded.
function _hold_air_mass_tape!(runtime::_TapeRuntime, ops, identity::_TapeIdentity)
    for op in ops
        if (op isa _CSSweepRecord && op.panels_tracer !== nothing) || op isa _CSLinRoodHorizRecord
            throw(ArgumentError("AirMassTape cannot hold tracer-dependent records ($(typeof(op)))."))
        end
    end
    runtime.held = _HeldAirMassTape(ops, identity, runtime.pool.generation)
    return nothing
end

# ---------------------------------------------------------------------------
# Explicit release of what a tape holds outside its pool
# ---------------------------------------------------------------------------

# A replay snapshot's tracer and air masses (operator state is small and left
# to the GC).
function _release_snapshot!(snap)
    foreach(_free_tape_array!, snap.rm)
    foreach(_free_tape_array!, snap.m)
    return nothing
end

# Records that own arrays outside the pool: the Lin–Rood horizontal records
# copy their states and fluxes (`TapeRecorder.jl`). Released once every start
# has walked the tape.
_release_records!(ops) = (foreach(_release_record!, ops); nothing)
_release_record!(_op) = nothing
function _release_record!(op::_CSLinRoodHorizRecord)
    foreach(_free_tape_panels!, (op.panels_rm, op.panels_m, op.panels_q_buf_phase2,
                                 op.panels_q_buf_phase3, op.panels_fx_in, op.panels_fx_out,
                                 op.panels_fy_in, op.panels_am, op.panels_bm))
    return nothing
end
