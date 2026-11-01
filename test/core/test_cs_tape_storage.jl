#!/usr/bin/env julia
#
# Tape storage of the cubed-sphere adjoint: the runtime pools a call opens
# from the storage of its `AdjointTape`, and the `tape` values a reverse-pass
# call accepts and rejects.
#
# Every pool is reused: a rewind starts the next tape (stride window, 4D-Var
# evaluation) in the same memory and makes the previous tape's slots stale
# (`StaleTapeSlotError`), and a release frees the memory now. The device pool
# reuses its arrays; the pinned-host arena allocates and registers (pins) its
# host chunks once; `MmapCSTapeStorage(dir)` counts the largest tape, checks
# the free space of `dir`, and reserves the extent in one anonymous (unlinked)
# file before recording, so a full disk is an `ArgumentError` with the
# required and available bytes, not a SIGBUS while recording. Host slots are
# 64-byte aligned views into the pool's memory that never leave it: reads
# copy into caches keyed by read role and panel shape (on the device for GPU
# panels, emulated here). Gradient parity of every storage and checkpoint
# schedule is the bitwise gate `test_cs_adjoint_tape_bitwise.jl`.
using Test

import AtmosTransport
const AT = AtmosTransport
const Adv = AT.Operators.Advection
const TapeMod = AT.Tape

function _shape_panel(::Val{:m}, Nc, Hp, Nz, FT)
    N = Nc + 2Hp
    return [FT(1000 + 100i + 10j + k) for i in 1:N, j in 1:N, k in 1:Nz]
end
_shape_panel(::Val{:am}, Nc, Hp, Nz, FT) = fill(FT(0.5), Nc + 2Hp + 1, Nc + 2Hp, Nz)
_shape_panel(::Val{:bm}, Nc, Hp, Nz, FT) = fill(FT(-0.25), Nc + 2Hp, Nc + 2Hp + 1, Nz)
_shape_panel(::Val{:cm}, Nc, Hp, Nz, FT) = fill(FT(0.125), Nc + 2Hp, Nc + 2Hp, Nz + 1)

_build_panels(kind::Symbol; Nc = 3, Hp = 3, Nz = 4, FT = Float32, seed = 1) =
    ntuple(p -> _shape_panel(Val(kind), Nc, Hp, Nz, FT) .+ FT(0.001) * FT(seed) * FT(p), 6)

_panel_bytes(panels) = sum(p -> sizeof(eltype(p)) * length(p), panels)

# Byte offsets of a host slot's panel views from `base`.
_offsets(slot, base) = ntuple(p -> Int(pointer(slot.panels[p]) - Ptr{eltype(slot.panels[p])}(base)), 6)

# Mappings of this process into files under `dir` (Linux).
_mappings_under(dir) = Sys.islinux() ?
    count(line -> occursin(realpath(dir), line), eachline("/proc/self/maps")) : 0

# CPU reverse kernels accumulate with atomics: bit for bit on one thread.
_same(a, b) = all(s -> all(p -> Threads.nthreads() == 1 ? a[s][p] == b[s][p] :
                                isapprox(a[s][p], b[s][p]; rtol = 1e-12, atol = 1e-15), 1:6),
                  eachindex(a, b))

# A stand-in for device arrays: host pools copy reads into caches `similar` to
# it, and the arena registers its chunks for it, as for CUDA and AMDGPU arrays.
struct _PseudoDevArray{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
end
Base.size(a::_PseudoDevArray) = size(a.data)
Base.@propagate_inbounds Base.getindex(a::_PseudoDevArray, i::Int...) = a.data[i...]
Base.@propagate_inbounds Base.setindex!(a::_PseudoDevArray, v, i::Int...) = (a.data[i...] = v)
Base.IndexStyle(::Type{<:_PseudoDevArray}) = IndexLinear()
Base.similar(::_PseudoDevArray, ::Type{S}, dims::Dims) where {S} = _PseudoDevArray(Array{S}(undef, dims))
Base.copyto!(dst::_PseudoDevArray, src::AbstractArray) = (copyto!(dst.data, src); dst)
Base.copyto!(dst::AbstractArray, src::_PseudoDevArray) = (copyto!(dst, src.data); dst)
Base.copyto!(dst::_PseudoDevArray, src::_PseudoDevArray) = (copyto!(dst.data, src.data); dst)
wrap6(panels) = ntuple(p -> _PseudoDevArray(panels[p]), 6)

# Its backend's chunk registrations, as the extensions make them.
const _REGISTERED = Dict{Ptr{UInt8}, Int}()
struct _PseudoRegistration
    ptr::Ptr{UInt8}
end
function TapeMod._pin_host_chunk(::_PseudoDevArray, ptr::Ptr, nbytes::Integer)
    haskey(_REGISTERED, ptr) && error("chunk registered twice")
    _REGISTERED[Ptr{UInt8}(ptr)] = nbytes
    return _PseudoRegistration(Ptr{UInt8}(ptr))
end
TapeMod._unpin_host_chunk(r::_PseudoRegistration) = (pop!(_REGISTERED, r.ptr); nothing)

@testset "device pool: arrays reused across tapes, stale slots refused" begin
    src = _build_panels(:m; Nz = 3)
    pool = TapeMod._open_tape_pool(AT.DeviceCSTapeStorage(), 0)
    slots = [TapeMod._stage_panels(pool, map(a -> a .+ k, src)) for k in 1:3]
    @test all(k -> all(p -> TapeMod._tape_panels(slots[k], TapeMod.AirMassRead())[p] == src[p] .+ k, 1:6), 1:3)
    arrays = [slot.panels for slot in slots]
    # The next tape (a stride window, an evaluation) records into the same arrays.
    TapeMod._rewind_tape_pool!(pool)
    @test_throws TapeMod.StaleTapeSlotError TapeMod._tape_panels(slots[1], TapeMod.AirMassRead())
    again = [TapeMod._stage_panels(pool, src) for _ in 1:2]
    @test all(k -> again[k].panels === arrays[k], 1:2)
    # A state of another element type takes a new array in that place.
    other = TapeMod._stage_panels(pool, map(a -> Float64.(a), src))
    @test other.panels !== arrays[3] && eltype(other.panels[1]) === Float64
    @test length(pool.slabs) == 3
    @test TapeMod._tape_panels(other, TapeMod.TracerMassRead())[2] == Float64.(src[2])
    TapeMod._release_tape_pool!(pool)
    @test pool.released && isempty(pool.slabs)
    @test_throws TapeMod.StaleTapeSlotError TapeMod._tape_panels(other, TapeMod.AirMassRead())
    @test_throws TapeMod.StaleTapeSlotError TapeMod._stage_panels(pool, src)
end

@testset "pinned-host arena: chunks allocated and pinned once, reused across tapes" begin
    # A 2.24 MB state: the first chunk (64 MiB) holds 29 of them.
    src = wrap6(_build_panels(:m; Nc = 48, Hp = 3, Nz = 16, FT = Float64))
    nbytes = TapeMod._tape_slot_bytes(src)
    arena = TapeMod._open_tape_pool(AT.PinnedHostCSTapeStorage(), 0)
    slots = [TapeMod._stage_panels(arena, src) for _ in 1:40]
    nchunks = length(arena.chunks)
    @test nchunks == 2 && length(_REGISTERED) == 2
    @test all(c -> haskey(_REGISTERED, c.base) && _REGISTERED[c.base] == c.capacity, arena.chunks)
    @test arena.chunks[1].capacity == TapeMod._HOST_ARENA_MIN_CHUNK
    @test all(slot -> all(p -> UInt(pointer(slot.panels[p])) % 64 == 0, 1:6), slots)
    # Reads copy into the role's device cache.
    got = TapeMod._tape_panels(slots[end], TapeMod.AirMassRead())
    @test got[3] isa _PseudoDevArray && got[3].data == src[3].data
    # The next tape reuses the chunks in order: no allocation, no registration.
    bases = [c.base for c in arena.chunks]
    TapeMod._rewind_tape_pool!(arena)
    @test_throws TapeMod.StaleTapeSlotError TapeMod._tape_panels(slots[1], TapeMod.AirMassRead())
    doubled = map(a -> _PseudoDevArray(a.data .* 2), src)
    again = [TapeMod._stage_panels(arena, doubled) for _ in 1:40]
    @test [c.base for c in arena.chunks] == bases && length(_REGISTERED) == 2
    @test pointer(again[1].panels[1]) == pointer(slots[1].panels[1])
    @test TapeMod._tape_panels(again[40], TapeMod.TracerMassRead())[6].data == 2 .* src[6].data
    # Release unregisters and frees every chunk now.
    TapeMod._release_tape_pool!(arena)
    @test isempty(_REGISTERED) && isempty(arena.chunks) && isempty(arena.read_caches)
    @test_throws TapeMod.StaleTapeSlotError TapeMod._tape_panels(again[1], TapeMod.AirMassRead())
    @test_throws TapeMod.StaleTapeSlotError TapeMod._stage_panels(arena, src)
    # Plain host panels need no registration.
    cpu = TapeMod._open_tape_pool(AT.PinnedHostCSTapeStorage(), 0)
    slot = TapeMod._stage_panels(cpu, map(a -> a.data, src))
    @test cpu.chunks[1].registration === nothing
    @test TapeMod._tape_panels(slot, TapeMod.AirMassRead())[1] == src[1].data
    TapeMod._release_tape_pool!(cpu)
end

@testset "mmap tape file: reservation and slots" begin
    @testset "anonymous file, reserved extent, bit-exact slots" begin
        mktempdir() do dir
            src = _build_panels(:m; Nc = 4, Hp = 3, Nz = 5, FT = Float64)
            nbytes = TapeMod._tape_slot_bytes(src)
            # Each panel starts on a 64-byte boundary: 4000 bytes take 4032.
            stride = cld(sizeof(Float64) * length(src[1]), 64) * 64
            @test nbytes == 6stride > _panel_bytes(src)
            storage = TapeMod._MmapTapeFile(dir, 2nbytes)
            # The file is unlinked at creation, its extent reserved and mapped once.
            @test isempty(readdir(dir))
            @test filesize(storage.io) == 2nbytes
            @test storage.mapping != C_NULL
            slot = TapeMod._stage_panels(storage, src)
            @test slot isa TapeMod.HostCSTapeSlot && eltype(slot.panels[1]) === Float64
            @test _offsets(slot, storage.mapping) == (0, stride, 2stride, 3stride, 4stride, 5stride)
            @test storage.cursor == nbytes
            got = TapeMod._tape_panels(slot, TapeMod.AirMassRead())
            @test all(p -> got[p] == src[p] && eltype(got[p]) === Float64, 1:6)
            # A read is a copy in the role's cache, never a view of the file.
            @test all(p -> got[p] !== slot.panels[p], 1:6)

            # Rewinding reuses the extent (one stride window after another);
            # the previous tape's slots are stale.
            TapeMod._rewind_tape_pool!(storage)
            @test storage.cursor == 0
            again = TapeMod._stage_panels(storage, map(a -> a .+ 1, src))
            @test _offsets(again, storage.mapping) == _offsets(slot, storage.mapping)
            @test_throws TapeMod.StaleTapeSlotError TapeMod._tape_panels(slot, TapeMod.AirMassRead())
            @test_throws TapeMod.StaleTapeSlotError TapeMod.stage_panels!(slot, src)
            TapeMod._stage_panels(storage, src)
            # Growth past the reservation is a counting bug, not a silent extension.
            @test_throws ErrorException TapeMod._stage_panels(storage, src)
            @test filesize(storage.io) == 2nbytes
            @test TapeMod._holds_tape_bytes(storage, 2nbytes) && !TapeMod._holds_tape_bytes(storage, 2nbytes + 1)

            TapeMod._release_tape_pool!(storage)
            @test storage.released && !TapeMod._holds_tape_bytes(storage, 0)
            # The slots are refused after the unmap rather than read.
            @test_throws TapeMod.StaleTapeSlotError TapeMod._tape_panels(again, TapeMod.AirMassRead())
            @test_throws TapeMod.StaleTapeSlotError TapeMod._stage_panels(storage, src)
            TapeMod._release_tape_pool!(storage)           # a second release is a no-op
            @test isempty(readdir(dir))
        end
    end

    @testset "multiple slots, heterogeneous shapes, reverse reads" begin
        mktempdir() do dir
            # Mixed shapes and element types: every view stays aligned.
            srcs = [_build_panels(:m; seed = 1), _build_panels(:am; seed = 2, FT = Float64),
                    _build_panels(:bm; seed = 3), _build_panels(:cm; seed = 4, FT = Float64)]
            storage = TapeMod._MmapTapeFile(dir, sum(TapeMod._tape_slot_bytes, srcs))
            slots = [TapeMod._stage_panels(storage, src) for src in srcs]
            @test storage.cursor == storage.capacity
            offsets = [_offsets(slot, storage.mapping) for slot in slots]
            @test issorted(first.(offsets))
            @test all(o -> all(x -> x % 64 == 0, o), offsets)
            for idx in 4:-1:1      # the reverse walk's order
                got = TapeMod._tape_panels(slots[idx], TapeMod.AirMassRead())
                @test all(p -> size(got[p]) == size(srcs[idx][p]) && got[p] == srcs[idx][p] &&
                               eltype(got[p]) === eltype(srcs[idx][p]), 1:6)
            end
            TapeMod._release_tape_pool!(storage)
        end
    end

    @testset "free space is checked before the file is created" begin
        mktempdir() do dir
            capacity = Int64(1) << 62               # more than any file system holds (no wait)
            err = try
                TapeMod._MmapTapeFile(joinpath(dir, "scratch"), capacity)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            msg = sprint(showerror, err)
            @test occursin("needs $(capacity) bytes", msg) && occursin("are available", msg)
            # The directory is created; nothing is left in it.
            @test isempty(readdir(joinpath(dir, "scratch")))
            @test_throws ArgumentError TapeMod._MmapTapeFile(dir, -1)
        end
    end

    @testset "file systems that cannot reserve blocks are named" begin
        mktempdir() do dir
            fstype = TapeMod._filesystem_type(dir)
            @test fstype isa String && !isempty(fstype)
            Sys.islinux() && @test fstype != "unknown"
            @test TapeMod._check_tape_filesystem(dir, fstype) === false
            err = try
                TapeMod._check_tape_filesystem(dir, "zfs")
                nothing
            catch e
                e
            end
            @test err isa ArgumentError && occursin("zfs", sprint(showerror, err))
        end
    end

    @testset "release unmaps and closes: nothing accumulates over calls" begin
        mktempdir() do dir
            src = _build_panels(:m; Nc = 4, Hp = 3, Nz = 5, FT = Float64)
            nbytes = TapeMod._tape_slot_bytes(src)
            before = _mappings_under(dir)
            for _ in 1:3
                storage = TapeMod._MmapTapeFile(dir, 8nbytes)
                slots = [TapeMod._stage_panels(storage, src) for _ in 1:8]
                foreach(slot -> TapeMod._tape_panels(slot, TapeMod.AirMassRead()), slots)
                Sys.islinux() && @test _mappings_under(dir) == before + 1   # one mapping per file
                TapeMod._release_tape_pool!(storage)
                @test storage.released && storage.mapping == C_NULL && !isopen(storage.io)
                # No GC in between: the mapping and the descriptor are gone.
                @test _mappings_under(dir) == before
            end
            TapeMod._release_tape_pool!(TapeMod._MmapTapeFile(dir, 0))   # empty tape
            @test isempty(readdir(dir))
        end
    end

    @testset "a failed reservation is an ArgumentError" begin
        mktempdir() do dir
            path, io = mktemp(dir)
            close(io)
            # A stream that cannot be written fails without allocating anything
            # (fallocate: EBADF; zero fill: a write error).
            open(path, "r") do readonly
                @test_throws ArgumentError TapeMod._reserve_tape_extent!(readonly, Int64(4096), dir)
                @test_throws ArgumentError TapeMod._zero_fill_tape_extent!(readonly, Int64(4096), dir)
            end
            # The portable zero-fill reservation (used off Linux).
            open(path, "w+") do io
                TapeMod._zero_fill_tape_extent!(io, Int64(100_000), dir)
                @test filesize(io) == 100_000
            end
        end
    end

    @testset "role- and shape-keyed device read caches" begin
        # Device panels (`_PseudoDevArray`): the pool records their backend, so
        # reads copy into per-role, per-shape caches on it, as with CUDA and
        # AMDGPU arrays.
        shape_key(src) = ntuple(p -> size(src[p]), 6)
        air, tracer = TapeMod.AirMassRead(), TapeMod.TracerMassRead()
        mktempdir() do dir
            shapes = [:m, :am, :bm, :cm, :m, :am]
            sources = [wrap6(_build_panels(s; Nz = 2, seed = i)) for (i, s) in enumerate(shapes)]
            storage = TapeMod._MmapTapeFile(dir, sum(TapeMod._tape_slot_bytes, sources))
            slots = [TapeMod._stage_panels(storage, src) for src in sources]
            @test storage.read_template isa _PseudoDevArray
            @test isempty(storage.read_caches)
            for (slot, src) in zip(slots, sources)
                got = TapeMod._tape_panels(slot, air)
                @test all(p -> got[p] isa _PseudoDevArray && got[p].data == src[p].data, 1:6)
                @test got === storage.read_caches[(air, shape_key(src))]
            end
            @test length(storage.read_caches) == 4   # one per distinct shape
            # The other role reads into its own buffers (issue #57).
            m_read = TapeMod._tape_panels(slots[1], air)
            rm_read = TapeMod._tape_panels(slots[5], tracer)
            @test all(p -> m_read[p] !== rm_read[p], 1:6)
            @test all(p -> m_read[p].data == sources[1][p].data && rm_read[p].data == sources[5][p].data, 1:6)
            @test length(storage.read_caches) == 5
            TapeMod._release_tape_pool!(storage)
            @test isempty(storage.read_caches)
        end
    end
end

# ---------------------------------------------------------------------------
# `tape` values of the reverse-pass entry points
# ---------------------------------------------------------------------------

function _problem(; Nc = 4, Nz = 3, nsteps = 4, FT = Float64)
    mesh = AT.CubedSphereMesh(Nc = Nc, Hp = 3, FT = FT)
    Hp = mesh.Hp
    N = Nc + 2Hp
    m = ntuple(p -> [FT(2.0 + 0.25k + 0.01p + 0.0001(i + j)) for i in 1:N, j in 1:N, k in 1:Nz], 6)
    rm = ntuple(p -> m[p] .* FT(0.05), 6)
    Adv.fill_panel_halos!(m, mesh; dir = 0)
    Adv.fill_panel_halos!(rm, mesh; dir = 0)
    am = [ntuple(p -> (a = zeros(FT, N + 1, N, Nz);
                       for k in 1:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc+1
                           a[i, j, k] = FT(0.013) * sin(FT(0.21s + 0.31p + 0.71i + 0.41j + 0.23k))
                       end; a), 6) for s in 1:nsteps]
    bm = [ntuple(p -> (b = zeros(FT, N, N + 1, Nz);
                       for k in 1:Nz, j in Hp+1:Hp+Nc+1, i in Hp+1:Hp+Nc
                           b[i, j, k] = FT(0.011) * cos(FT(0.33s + 0.51p + 0.41i + 0.61j + 0.13k))
                       end; b), 6) for s in 1:nsteps]
    cm = [ntuple(p -> (c = zeros(FT, N, N, Nz + 1);
                       for k in 2:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc
                           c[i, j, k] = -FT(0.009) * (1 + FT(0.1) * sin(FT(i + j + k + p + s)))
                       end; c), 6) for s in 1:nsteps]
    return mesh, m, rm, am, bm, cm
end

@testset "tape values" begin
    mesh, m, rm, am, bm, cm = _problem()
    objective = AT.CSColumnMeanObjective(1, 2, 2)
    seed = map(zero, rm)
    scheme = AT.PPMScheme(AT.MonotoneLimiter())
    footprint(; kw...) = AT.cs_surface_emission_footprint(rm, m, am, bm, cm, mesh, objective;
                                                          scheme, dt = 1.0, kw...)
    reference = footprint()

    @testset "AdjointTape" begin
        @test AT.AdjointTape() === AT.AdjointTape(AT.StateTape(), AT.DeviceCSTapeStorage(),
                                                  AT.FullCheckpoint())
        tape = AT.AdjointTape(; storage = AT.PinnedHostCSTapeStorage(), checkpoint = AT.StrideCheckpoint(2))
        @test tape isa AT.AdjointTape{AT.StateTape, AT.PinnedHostCSTapeStorage, AT.StrideCheckpoint}
        @test isbits(tape)                        # a descriptor: no runtime state
        for bad in ((; storage = :device), (; storage = "mmap"), (; checkpoint = :full),
                    (; content = nothing))
            @test_throws TypeError AT.AdjointTape(; bad...)
        end
        @test_throws ArgumentError AT.StrideCheckpoint(0)
        # The Symbol API is gone.
        @test_throws MethodError footprint(; tape_storage = :device)
        @test_throws TypeError footprint(; tape = :device)
    end

    @testset "MmapCSTapeStorage needs a directory and leaves nothing in it" begin
        @test_throws ArgumentError AT.MmapCSTapeStorage("")
        @test_throws MethodError AT.MmapCSTapeStorage()
        @test AT.MmapCSTapeStorage("relative/dir").dir == abspath("relative/dir")
        mktempdir() do root
            dir = joinpath(root, "tape")
            storage = AT.MmapCSTapeStorage(dir)
            @test !ispath(dir)                       # created by the call, not the constructor
            for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(3))
                result = footprint(; tape = AT.AdjointTape(; storage, checkpoint))
                @test _same(result.footprints, reference.footprints)
                # Unmapped and closed when the call returns, without a GC (the
                # file is unlinked at creation: no name is ever left).
                @test isempty(readdir(dir))
                @test _mappings_under(dir) == 0
            end
        end
    end

    @testset "a runtime reuses its pool; a larger mmap tape gets a new file" begin
        Footprint = AT.Adjoints.Footprint
        mktempdir() do root
            dir = joinpath(root, "tape")
            runtime = Footprint._TapeRuntime(AT.AdjointTape(; storage = AT.MmapCSTapeStorage(dir)))
            first_pool = Footprint._tape_pool!(runtime, 1 << 16)
            @test Footprint._tape_pool!(runtime, 1 << 12) === first_pool   # rewound, reused
            @test first_pool.generation == 2
            larger = Footprint._tape_pool!(runtime, 1 << 20)
            @test larger !== first_pool && first_pool.released && larger.capacity == 1 << 20
            Footprint._release_tape_runtime!(runtime)
            @test larger.released && runtime.pool === nothing
            @test _mappings_under(dir) == 0
        end
        runtime = Footprint._TapeRuntime(AT.AdjointTape())
        pool = Footprint._tape_pool!(runtime, 0)
        @test Footprint._tape_pool!(runtime, 1 << 40) === pool          # device pools grow on demand
        Footprint._release_tape_runtime!(runtime)
        @test pool.released
    end

    @testset "one tape serves every call and schedule" begin
        for storage in (AT.DeviceCSTapeStorage(), AT.PinnedHostCSTapeStorage()),
            checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2))
            tape = AT.AdjointTape(; storage, checkpoint)
            @test _same(footprint(; tape).footprints, reference.footprints)
            @test _same(footprint(; tape).footprints, reference.footprints)
        end
    end

    @testset "rejected values" begin
        mktempdir() do root
            dir = joinpath(root, "tape")
            for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2)),
                storage in (AT.PinnedHostCSTapeStorage(), AT.MmapCSTapeStorage(dir))
                tape = AT.AdjointTape(; storage, checkpoint)
                # The Lin-Rood tape is device-only.
                @test_throws ArgumentError AT.cs_surface_emission_footprint(
                    rm, m, am, bm, cm, mesh, objective; scheme = AT.LinRoodPPMScheme(), tape)
                @test_throws ArgumentError AT.cs_surface_emission_footprint_from_seed(
                    seed, m, am, bm, cm, mesh; scheme = AT.LinRoodPPMScheme(), tape)
            end
            @test !ispath(dir)
        end
    end
end
