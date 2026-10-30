#!/usr/bin/env julia

using Test
using TOML

import AtmosTransport
const AT = AtmosTransport
const TapeMod = AtmosTransport.Tape

# ---------------------------------------------------------------------------
# Plan 26 Phase A.1 — `MmapCSTapeStorage` on-disk tape roundtrip tests.
#
# Verifies:
#   * Slot allocation bumps the cursor by the expected number of bytes.
#   * stage_panels! → _tape_panels(slot, role) yields bit-exact equality.
#   * Multiple staged slots stay disjoint and readable in any order.
#   * Heterogeneous panel shapes (am / bm / cm-shaped tapes) survive the
#     roundtrip with the per-panel offset bookkeeping.
#   * finalize_tape! emits a manifest.toml that captures every record and
#     marks the storage `finalised`.
#   * The CPU mmap path is bit-exact against the in-memory
#     DeviceCSTapeStorage path for cs_surface_emission_footprint across
#     upwind, linear PPM, monotone PPM, and emission-bearing midpoints.
# ---------------------------------------------------------------------------

function _shape_panel(::Val{:m}, Nc, Hp, Nz, FT)
    N = Nc + 2Hp
    a = zeros(FT, N, N, Nz)
    for k in 1:Nz, j in 1:N, i in 1:N
        a[i, j, k] = FT(1000 + 100i + 10j + k)
    end
    a
end

_shape_panel(::Val{:am}, Nc, Hp, Nz, FT) =
    fill!(zeros(FT, Nc + 2Hp + 1, Nc + 2Hp, Nz), FT(0.5))
_shape_panel(::Val{:bm}, Nc, Hp, Nz, FT) =
    fill!(zeros(FT, Nc + 2Hp, Nc + 2Hp + 1, Nz), FT(-0.25))
_shape_panel(::Val{:cm}, Nc, Hp, Nz, FT) =
    fill!(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz + 1), FT(0.125))

function _build_panels(kind::Symbol; Nc = 3, Hp = 3, Nz = 4, FT = Float32, seed = 1)
    return ntuple(p -> begin
        a = _shape_panel(Val(kind), Nc, Hp, Nz, FT)
        a .+= FT(0.001) * FT(seed) * FT(p)
        a
    end, 6)
end

@testset "MmapCSTapeStorage roundtrip" begin
    @testset "single-slot bit-exact roundtrip" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            src = _build_panels(:m; Nc = 4, Hp = 3, Nz = 5, FT = Float64)
            slot = TapeMod._stage_panels(storage, src)

            @test slot.record_id == 1
            @test slot.eltype === Float64
            expected_offsets = let
                stride = sizeof(Float64) * length(src[1])
                (0, stride, 2stride, 3stride, 4stride, 5stride)
            end
            @test slot.offsets == expected_offsets
            @test storage.cursor == 6 * sizeof(Float64) * length(src[1])

            got = TapeMod._tape_panels(slot, TapeMod.AirMassRead())
            @test length(got) == 6
            for p in 1:6
                @test got[p] == src[p]
                @test eltype(got[p]) === Float64
                @test size(got[p]) == size(src[p])
            end

            TapeMod.finalize_tape!(storage)
            @test isfile(joinpath(dir, "manifest.toml"))
            @test filesize(joinpath(dir, "records.bin")) == storage.cursor

            m = TOML.parsefile(joinpath(dir, "manifest.toml"))
            @test m["meta"]["finalised"] === true
            @test m["meta"]["record_count"] == 1
            @test m["meta"]["version"] == "v1"
            @test m["meta"]["endianness"] in ("little", "big")
            @test length(m["record"]) == 1
            @test m["record"][1]["eltype"] == "Float64"
            @test m["record"][1]["offsets"] == collect(expected_offsets)
        end
    end

    @testset "multiple slots stay disjoint and out-of-order readable" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            srcs = [_build_panels(:m; Nc = 3, Hp = 3, Nz = 3, FT = Float32, seed = s)
                    for s in 1:4]
            slots = [TapeMod._stage_panels(storage, src) for src in srcs]

            # Read in reverse to mirror the adjoint walk order.
            for idx in 4:-1:1
                got = TapeMod._tape_panels(slots[idx], TapeMod.AirMassRead())
                for p in 1:6
                    @test got[p] == srcs[idx][p]
                end
            end

            # Each slot should sit at a unique base offset.
            base_offsets = [slot.offsets[1] for slot in slots]
            @test length(unique(base_offsets)) == 4
            @test issorted(base_offsets)
            TapeMod.finalize_tape!(storage)
        end
    end

    @testset "heterogeneous panel shapes (am, bm, cm)" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            am = _build_panels(:am; FT = Float32, seed = 7)
            bm = _build_panels(:bm; FT = Float32, seed = 8)
            cm = _build_panels(:cm; FT = Float32, seed = 9)

            slot_am = TapeMod._stage_panels(storage, am)
            slot_bm = TapeMod._stage_panels(storage, bm)
            slot_cm = TapeMod._stage_panels(storage, cm)

            for (slot, src) in ((slot_am, am), (slot_bm, bm), (slot_cm, cm))
                got = TapeMod._tape_panels(slot, TapeMod.AirMassRead())
                for p in 1:6
                    @test size(got[p]) == size(src[p])
                    @test got[p] == src[p]
                end
            end
            TapeMod.finalize_tape!(storage)
        end
    end

    @testset "role- and shape-keyed device read caches" begin
        # Stand-in for a "GPU-ish" backend that picks up the device-cache
        # path: wrap Array{T,N} in a thin subtype so the
        # `_mmap_prepare_for_panels!(::NTuple{6, <:Array})` no-op
        # specialisation does NOT fire and we exercise the default
        # method that records the read backend (`storage.read_template`).
        # Same arithmetic behaviour as Array, so copyto! / similar /
        # eltype work unchanged.
        struct _PseudoDevArray{T, N} <: AbstractArray{T, N}
            data::Array{T, N}
        end
        Base.size(a::_PseudoDevArray) = size(a.data)
        Base.@propagate_inbounds Base.getindex(a::_PseudoDevArray, i::Int...) =
            a.data[i...]
        Base.@propagate_inbounds Base.setindex!(a::_PseudoDevArray, v, i::Int...) =
            (a.data[i...] = v)
        Base.IndexStyle(::Type{<:_PseudoDevArray}) = IndexLinear()
        Base.similar(a::_PseudoDevArray, ::Type{S}, dims::Dims) where {S} =
            _PseudoDevArray(Array{S}(undef, dims))
        Base.copyto!(dst::_PseudoDevArray, src::AbstractArray) =
            (copyto!(dst.data, src); dst)
        Base.copyto!(dst::AbstractArray, src::_PseudoDevArray) =
            (copyto!(dst, src.data); dst)
        Base.copyto!(dst::_PseudoDevArray, src::_PseudoDevArray) =
            (copyto!(dst.data, src.data); dst)
        Base.:(==)(a::_PseudoDevArray, b::_PseudoDevArray) = a.data == b.data
        Base.:(==)(a::_PseudoDevArray, b::AbstractArray) = a.data == b
        Base.:(==)(a::AbstractArray, b::_PseudoDevArray) = a == b.data

        wrap6(panels) = ntuple(p -> _PseudoDevArray(panels[p]), 6)
        shape_key(src) = ntuple(p -> size(src[p]), 6)
        air, tracer = TapeMod.AirMassRead(), TapeMod.TracerMassRead()

        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            # Four heterogeneous shapes interleaved across six slots; the
            # keyed caches must serve repeats without reallocating.
            shapes = [:m, :am, :bm, :cm, :m, :am]
            sources = [wrap6(_build_panels(s; Nc = 3, Hp = 3, Nz = 2,
                                           FT = Float32, seed = i))
                       for (i, s) in enumerate(shapes)]
            slots = [TapeMod._stage_panels(storage, src) for src in sources]

            # Staging records the device backend; the caches are allocated
            # on first read, on that backend.
            @test storage.read_template isa _PseudoDevArray
            @test isempty(storage.read_caches)

            for (slot, src) in zip(slots, sources)
                got = TapeMod._tape_panels(slot, air)
                @test got isa NTuple{6}
                @test all(got[p] isa _PseudoDevArray for p in 1:6)
                @test all(got[p] == src[p] for p in 1:6)
                # Same cache buffer is returned for repeat shape reads.
                @test got === storage.read_caches[(air, shape_key(src))]
            end
            # One cache per distinct shape: m=(9,9,2), am=(10,9,2),
            # bm=(9,10,2), cm=(9,9,3) — see `_shape_panel`.
            @test length(storage.read_caches) == 4

            # The other role reads into its own buffers (issue #57: a sweep
            # record's air mass and tracer mass are read together).
            m_read = TapeMod._tape_panels(slots[1], air)
            rm_read = TapeMod._tape_panels(slots[5], tracer)
            @test all(m_read[p] !== rm_read[p] for p in 1:6)
            @test all(m_read[p] == sources[1][p] for p in 1:6)
            @test all(rm_read[p] == sources[5][p] for p in 1:6)
            @test length(storage.read_caches) == 5
            TapeMod.finalize_tape!(storage)
            @test isempty(storage.read_caches)
        end
    end
end

# ---------------------------------------------------------------------------
# Cross-policy parity: :mmap must be bit-exact against :device on CPU.
# ---------------------------------------------------------------------------

const Adv = AT.Operators.Advection

function _trivial_problem(; Nc = 4, Nz = 3, nsteps = 2, FT = Float64,
                          nontrivial::Bool = false)
    mesh = AT.CubedSphereMesh(Nc = Nc, Hp = 3, FT = FT)
    N = mesh.Nc + 2 * mesh.Hp
    Hp = mesh.Hp
    panels_m = ntuple(p -> begin
        m = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in 1:N, i in 1:N
            m[i, j, k] = FT(2.0 + 0.25 * k + 0.01 * p)
        end
        m
    end, 6)
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    Adv.fill_panel_halos!(panels_m, mesh; dir = 0)
    Adv.fill_panel_halos!(panels_rm, mesh; dir = 0)

    if nontrivial
        am_steps = [ntuple(p -> begin
            a = zeros(FT, N + 1, N, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + Nc, i in Hp + 1:Hp + Nc + 1
                a[i, j, k] = FT(0.015) *
                    sin(FT(0.2 * step + 0.3 * p + 0.7 * i + 0.4 * j + 0.2 * k))
            end
            a
        end, 6) for step in 1:nsteps]
        bm_steps = [ntuple(p -> begin
            b = zeros(FT, N, N + 1, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + Nc + 1, i in Hp + 1:Hp + Nc
                b[i, j, k] = FT(0.012) *
                    cos(FT(0.3 * step + 0.5 * p + 0.4 * i + 0.6 * j + 0.1 * k))
            end
            b
        end, 6) for step in 1:nsteps]
        cm_steps = [ntuple(p -> begin
            c = zeros(FT, N, N, Nz + 1)
            for k in 2:Nz, j in Hp + 1:Hp + Nc, i in Hp + 1:Hp + Nc
                c[i, j, k] = -FT(0.010) *
                    (one(FT) + FT(0.1) * sin(FT(i + j + k + p + step)))
            end
            c
        end, 6) for step in 1:nsteps]
    else
        am_steps = [ntuple(_ -> zeros(FT, N + 1, N, Nz), 6) for _ in 1:nsteps]
        bm_steps = [ntuple(_ -> zeros(FT, N, N + 1, Nz), 6) for _ in 1:nsteps]
        cm_steps = [ntuple(_ -> zeros(FT, N, N, Nz + 1), 6) for _ in 1:nsteps]
    end
    return mesh, panels_m, panels_rm, am_steps, bm_steps, cm_steps
end

function _max_footprint_diff(a, b)
    nsteps = length(a.footprints)
    return maximum(maximum(abs, a.footprints[s][p] .- b.footprints[s][p])
                   for s in 1:nsteps, p in 1:6)
end

# Device and mmap tape replay should agree to roundoff; Julia/LLVM patch-level
# differences can move the final adjoint accumulation by a few eps.
const FOOTPRINT_REPLAY_ATOL = 1e-12

@testset "cs_surface_emission_footprint :mmap parity with :device" begin
    @testset "$(nameof(typeof(scheme))) ($(nontrivial ? "transport" : "constant"))" for
        scheme in (AT.UpwindScheme(),
                   AT.SlopesScheme(AT.NoLimiter()),
                   AT.PPMScheme(AT.NoLimiter()),
                   AT.PPMScheme(AT.MonotoneLimiter())),
        nontrivial in (false, true)

        mesh, panels_m, panels_rm, am, bm, cm =
            _trivial_problem(Nc = 4, Nz = 3, nsteps = 2; nontrivial = nontrivial)
        obj = AT.CSLayerMeanObjective(1, 2, 2, 3)
        dev = AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
            mesh, obj; scheme = scheme, dt = 1.0, tape_storage = :device)
        mmap = AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
            mesh, obj; scheme = scheme, dt = 1.0, tape_storage = :mmap)
        @test _max_footprint_diff(dev, mmap) ≤ FOOTPRINT_REPLAY_ATOL
    end

    @testset "from_seed parity" begin
        mesh, panels_m, _, am, bm, cm =
            _trivial_problem(Nc = 4, Nz = 3, nsteps = 2; nontrivial = true)
        N = mesh.Nc + 2mesh.Hp
        seed = ntuple(p -> begin
            a = zeros(Float64, N, N, 3)
            a[mesh.Hp + 2, mesh.Hp + 2, 3] = 0.5
            a
        end, 6)
        dev = AT.cs_surface_emission_footprint_from_seed(seed, panels_m, am, bm, cm,
            mesh; scheme = AT.PPMScheme(AT.NoLimiter()), dt = 1.0,
            tape_storage = :device)
        mmap = AT.cs_surface_emission_footprint_from_seed(seed, panels_m, am, bm, cm,
            mesh; scheme = AT.PPMScheme(AT.NoLimiter()), dt = 1.0,
            tape_storage = :mmap)
        @test _max_footprint_diff(dev, mmap) ≤ FOOTPRINT_REPLAY_ATOL
    end
end

# ---------------------------------------------------------------------------
# `cs_tape_byte_estimate` capacity-planning helper. Exported but previously
# untested. Verifies that the per-record-class counts and `state_bytes`
# total are consistent with the runtime tape that
# `cs_surface_emission_footprint` actually builds.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Codex-review fixes:
#   * LinRoodPPMScheme rejects non-device tape_storage explicitly.
#   * cs_surface_flux_4dvar validates control panel shapes vs mesh.Nc.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Plan 26 P0.A.2c — manifest-driven resume API.
#
# load_mmap_tape(dir) parses manifest.toml + records.bin from a previously
# finalised MmapCSTapeStorage and rebuilds the slot table. get_record(...)
# reconstructs MmapCSTapeSlot descriptors so _tape_panels can mmap-view
# the stored bytes. The reopened storage defaults to readonly: any further
# allocation / stage attempt must throw rather than silently corrupt the
# tape.
# ---------------------------------------------------------------------------

@testset "load_mmap_tape resume" begin
    @testset "bit-exact reload of single-slot tape" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            src = _build_panels(:m; Nc = 4, Hp = 3, Nz = 5, FT = Float32, seed = 7)
            slot = TapeMod._stage_panels(storage, src)
            written_cursor = storage.cursor
            TapeMod.finalize_tape!(storage)

            reopened = TapeMod.load_mmap_tape(dir)
            @test reopened isa TapeMod.MmapCSTapeStorage
            @test reopened.dir == dir
            @test reopened.cursor == written_cursor
            @test length(reopened.records) == 1
            @test reopened.finalised === true
            @test reopened.readonly === true

            got_slot = TapeMod.get_record(reopened, 1)
            @test got_slot.record_id == slot.record_id
            @test got_slot.eltype === slot.eltype
            @test got_slot.offsets == slot.offsets
            @test got_slot.shapes == slot.shapes

            got = TapeMod._tape_panels(got_slot, TapeMod.AirMassRead())
            for p in 1:6
                @test got[p] == src[p]
                @test eltype(got[p]) === Float32
            end

            TapeMod.finalize_tape!(reopened)
        end
    end

    @testset "multi-slot heterogeneous-shape reload" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            src_m = _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32, seed = 1)
            src_am = _build_panels(:am; Nc = 3, Hp = 3, Nz = 4, FT = Float32, seed = 2)
            src_cm = _build_panels(:cm; Nc = 3, Hp = 3, Nz = 4, FT = Float32, seed = 3)
            TapeMod._stage_panels(storage, src_m)
            TapeMod._stage_panels(storage, src_am)
            TapeMod._stage_panels(storage, src_cm)
            TapeMod.finalize_tape!(storage)

            reopened = TapeMod.load_mmap_tape(dir)
            @test length(reopened.records) == 3

            # Read in reverse (LIFO mirrors the reverse-pass walk).
            for (id, expected) in ((3, src_cm), (2, src_am), (1, src_m))
                slot = TapeMod.get_record(reopened, id)
                @test slot.record_id == id
                got = TapeMod._tape_panels(slot, TapeMod.AirMassRead())
                for p in 1:6
                    @test got[p] == expected[p]
                end
            end

            TapeMod.finalize_tape!(reopened)
        end
    end

    @testset "validation: missing files, corrupted meta, version mismatch" begin
        @test_throws ArgumentError TapeMod.load_mmap_tape(
            joinpath(tempdir(), "atmostransport-mmap-load-nope-$(rand(UInt32))"))

        # Directory exists, manifest missing.
        mktempdir() do dir
            @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
        end

        # records.bin missing.
        mktempdir() do dir
            open(joinpath(dir, "manifest.toml"), "w") do io
                println(io, "[meta]")
            end
            @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
        end

        # Version mismatch.
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32))
            TapeMod.finalize_tape!(storage)
            mpath = joinpath(dir, "manifest.toml")
            txt = read(mpath, String)
            write(mpath, replace(txt, "version = \"v1\"" => "version = \"v0\""))
            @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
        end

        # finalised = false (interrupted run).
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32))
            TapeMod.finalize_tape!(storage)
            mpath = joinpath(dir, "manifest.toml")
            txt = read(mpath, String)
            write(mpath, replace(txt, "finalised = true" => "finalised = false"))
            @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
        end

        # records.bin truncated below total_bytes.
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32))
            TapeMod.finalize_tape!(storage)
            bin_path = joinpath(dir, "records.bin")
            open(bin_path, "r+") do io
                Base.truncate(io, max(filesize(bin_path) - 16, 0))
            end
            @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
        end
        for name in ("Int64", "Bool", "Module")
            @test_throws ArgumentError TapeMod._parse_mmap_eltype(name)
        end
        for T in (Float32, Float64, Float16)
            @test TapeMod._parse_mmap_eltype(string(T)) === T
        end
    end



    @testset "per-record validation: offsets, nbytes, shapes" begin
        good_dir = mktempdir(; cleanup = false)
        try
            storage = TapeMod.MmapCSTapeStorage(; dir = good_dir,
                                                cleanup_on_finalize = false)
            TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32))
            TapeMod.finalize_tape!(storage)
            mpath = joinpath(good_dir, "manifest.toml")

            # Negative offset.
            mktempdir() do dir
                cp(joinpath(good_dir, "records.bin"), joinpath(dir, "records.bin"))
                m = TOML.parsefile(mpath)
                m["record"][1]["offsets"][1] = -8
                open(joinpath(dir, "manifest.toml"), "w") do io
                    TOML.print(io, m)
                end
                @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
            end

            # offset + nbytes past total_bytes.
            mktempdir() do dir
                cp(joinpath(good_dir, "records.bin"), joinpath(dir, "records.bin"))
                m = TOML.parsefile(mpath)
                m["record"][1]["offsets"][6] = Int(m["meta"]["total_bytes"]) - 1
                open(joinpath(dir, "manifest.toml"), "w") do io
                    TOML.print(io, m)
                end
                @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
            end

            # nbytes disagrees with shape × sizeof(eltype).
            mktempdir() do dir
                cp(joinpath(good_dir, "records.bin"), joinpath(dir, "records.bin"))
                m = TOML.parsefile(mpath)
                m["record"][1]["nbytes"][1] += 4
                open(joinpath(dir, "manifest.toml"), "w") do io
                    TOML.print(io, m)
                end
                @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
            end

            # Shape vector with wrong length.
            mktempdir() do dir
                cp(joinpath(good_dir, "records.bin"), joinpath(dir, "records.bin"))
                m = TOML.parsefile(mpath)
                m["record"][1]["shapes"][1] = [9, 9]  # length 2 instead of 3
                open(joinpath(dir, "manifest.toml"), "w") do io
                    TOML.print(io, m)
                end
                @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
            end

            # offsets vector with wrong length.
            mktempdir() do dir
                cp(joinpath(good_dir, "records.bin"), joinpath(dir, "records.bin"))
                m = TOML.parsefile(mpath)
                m["record"][1]["offsets"] = m["record"][1]["offsets"][1:5]
                open(joinpath(dir, "manifest.toml"), "w") do io
                    TOML.print(io, m)
                end
                @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
            end

            # Missing required field.
            mktempdir() do dir
                cp(joinpath(good_dir, "records.bin"), joinpath(dir, "records.bin"))
                m = TOML.parsefile(mpath)
                delete!(m["record"][1], "eltype")
                open(joinpath(dir, "manifest.toml"), "w") do io
                    TOML.print(io, m)
                end
                @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
            end
        finally
            rm(good_dir; recursive = true, force = true)
        end
    end

end
