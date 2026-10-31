#!/usr/bin/env julia
#
# Tape storage of the cubed-sphere adjoint: the `tape_storage` values a
# reverse-pass call accepts and rejects, and the on-disk tier.
#
# `MmapCSTapeStorage(dir)` names a required scratch directory. Each call counts
# its largest tape, checks the free space of `dir`, and reserves the extent in
# one anonymous (unlinked) file before recording, so a full disk is an
# `ArgumentError` with the required and available bytes, not a SIGBUS while
# recording. The file is mapped once; slots are 64-byte aligned views into
# that mapping, and releasing the storage unmaps and closes it, so nothing of
# it outlives the call. Reads come back as views (CPU) or through device
# caches keyed by read role and panel shape (GPU, emulated here). Gradient
# parity of every storage and checkpoint schedule is the bitwise gate
# `test_cs_adjoint_tape_bitwise.jl`.
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

# Mappings of this process into files under `dir` (Linux).
_mappings_under(dir) = Sys.islinux() ?
    count(line -> occursin(realpath(dir), line), eachline("/proc/self/maps")) : 0

# CPU reverse kernels accumulate with atomics: bit for bit on one thread.
_same(a, b) = all(s -> all(p -> Threads.nthreads() == 1 ? a[s][p] == b[s][p] :
                                isapprox(a[s][p], b[s][p]; rtol = 1e-12, atol = 1e-15), 1:6),
                  eachindex(a, b))

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
            @test slot.eltype === Float64
            @test slot.offsets == (0, stride, 2stride, 3stride, 4stride, 5stride)
            @test storage.cursor == nbytes
            got = TapeMod._tape_panels(slot, TapeMod.AirMassRead())
            @test all(p -> got[p] == src[p] && eltype(got[p]) === Float64, 1:6)

            # Rewinding reuses the extent (one stride window after another).
            TapeMod._rewind_tape_storage!(storage)
            @test storage.cursor == 0
            again = TapeMod._stage_panels(storage, map(a -> a .+ 1, src))
            @test again.offsets == slot.offsets
            TapeMod._stage_panels(storage, src)
            # Growth past the reservation is a counting bug, not a silent extension.
            @test_throws ErrorException TapeMod._stage_panels(storage, src)
            @test filesize(storage.io) == 2nbytes

            TapeMod._release_tape_storage!(storage)
            @test storage.closed
            @test_throws ArgumentError TapeMod._tape_panels(slot, TapeMod.AirMassRead())
            @test_throws ArgumentError TapeMod._stage_panels(storage, src)
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
            @test issorted([slot.offsets[1] for slot in slots])
            @test all(slot -> all(o -> o % 64 == 0, slot.offsets), slots)
            for idx in 4:-1:1      # the reverse walk's order
                got = TapeMod._tape_panels(slots[idx], TapeMod.AirMassRead())
                @test all(p -> size(got[p]) == size(srcs[idx][p]) && got[p] == srcs[idx][p] &&
                               eltype(got[p]) === eltype(srcs[idx][p]), 1:6)
            end
            TapeMod._release_tape_storage!(storage)
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
                TapeMod._release_tape_storage!(storage)
                @test storage.closed && storage.mapping == C_NULL && !isopen(storage.io)
                # No GC in between: the mapping and the descriptor are gone.
                @test _mappings_under(dir) == before
            end
            TapeMod._release_tape_storage!(TapeMod._MmapTapeFile(dir, 0))   # empty tape
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
        # A stand-in for device arrays: the generic `_mmap_prepare_for_panels!`
        # records the backend, so reads copy into per-role, per-shape caches as
        # the CUDA and AMDGPU extensions do.
        struct _PseudoDevArray{T, N} <: AbstractArray{T, N}
            data::Array{T, N}
        end
        Base.size(a::_PseudoDevArray) = size(a.data)
        Base.@propagate_inbounds Base.getindex(a::_PseudoDevArray, i::Int...) = a.data[i...]
        Base.@propagate_inbounds Base.setindex!(a::_PseudoDevArray, v, i::Int...) =
            (a.data[i...] = v)
        Base.IndexStyle(::Type{<:_PseudoDevArray}) = IndexLinear()
        Base.similar(::_PseudoDevArray, ::Type{S}, dims::Dims) where {S} =
            _PseudoDevArray(Array{S}(undef, dims))
        Base.copyto!(dst::_PseudoDevArray, src::AbstractArray) = (copyto!(dst.data, src); dst)
        Base.copyto!(dst::AbstractArray, src::_PseudoDevArray) = (copyto!(dst, src.data); dst)
        Base.copyto!(dst::_PseudoDevArray, src::_PseudoDevArray) = (copyto!(dst.data, src.data); dst)

        wrap6(panels) = ntuple(p -> _PseudoDevArray(panels[p]), 6)
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
            TapeMod._release_tape_storage!(storage)
            @test isempty(storage.read_caches)
        end
    end
end

# ---------------------------------------------------------------------------
# `tape_storage` values of the reverse-pass entry points
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

@testset "tape_storage values" begin
    mesh, m, rm, am, bm, cm = _problem()
    objective = AT.CSColumnMeanObjective(1, 2, 2)
    seed = map(zero, rm)
    scheme = AT.PPMScheme(AT.MonotoneLimiter())
    footprint(; kw...) = AT.cs_surface_emission_footprint(rm, m, am, bm, cm, mesh, objective;
                                                          scheme, dt = 1.0, kw...)
    reference = footprint()

    @testset "MmapCSTapeStorage needs a directory and leaves nothing in it" begin
        @test_throws ArgumentError AT.MmapCSTapeStorage("")
        @test_throws MethodError AT.MmapCSTapeStorage()
        @test AT.MmapCSTapeStorage("relative/dir").dir == abspath("relative/dir")
        mktempdir() do root
            dir = joinpath(root, "tape")
            storage = AT.MmapCSTapeStorage(dir)
            @test !ispath(dir)                       # created by the call, not the constructor
            for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(3), AT.RevolveCheckpoint())
                result = footprint(; tape_storage = storage, checkpoint)
                @test _same(result.footprints, reference.footprints)
                # Unmapped and closed when the call returns, without a GC.
                @test isempty(readdir(dir))
                @test _mappings_under(dir) == 0
            end
        end
    end

    @testset "a pre-built mmap tape file stays the caller's" begin
        mktempdir() do dir
            file = TapeMod._MmapTapeFile(dir, 1 << 22)
            for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2), AT.RevolveCheckpoint())
                @test _same(footprint(; tape_storage = file, checkpoint).footprints,
                            reference.footprints)
                @test !file.closed            # neither sized nor released by the call
            end
            TapeMod._release_tape_storage!(file)
            @test _mappings_under(dir) == 0
        end
    end

    @testset "a pre-constructed storage serves every schedule" begin
        for storage in (AT.DeviceCSTapeStorage(), AT.PinnedHostCSTapeStorage()),
            checkpoint in (AT.StrideCheckpoint(2), AT.RevolveCheckpoint())
            @test _same(footprint(; tape_storage = storage, checkpoint).footprints, reference.footprints)
        end
    end

    @testset "rejected values" begin
        mktempdir() do root
            dir = joinpath(root, "tape")
            for bad in (:mmap, :disk, "device", nothing)
                @test_throws ArgumentError footprint(; tape_storage = bad)
            end
            for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2), AT.RevolveCheckpoint()),
                tape_storage in (:pinned_host, AT.PinnedHostCSTapeStorage(), AT.MmapCSTapeStorage(dir))
                # The Lin-Rood tape is device-only.
                @test_throws ArgumentError AT.cs_surface_emission_footprint(
                    rm, m, am, bm, cm, mesh, objective;
                    scheme = AT.LinRoodPPMScheme(), checkpoint, tape_storage)
                @test_throws ArgumentError AT.cs_surface_emission_footprint_from_seed(
                    seed, m, am, bm, cm, mesh;
                    scheme = AT.LinRoodPPMScheme(), checkpoint, tape_storage)
            end
            @test !ispath(dir)
        end
    end
end
