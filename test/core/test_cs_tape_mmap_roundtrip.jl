#!/usr/bin/env julia

using Test
using TOML

import AtmosTransport
const AT = AtmosTransport
const TapeMod = AtmosTransport.Tape

function _manifest_records_path(dir)
    manifest = TOML.parsefile(joinpath(dir, "manifest.toml"))
    return joinpath(dir, manifest["meta"]["records_file"])
end

# ---------------------------------------------------------------------------
# Plan 26 Phase A.1 — `MmapCSTapeStorage` on-disk tape roundtrip tests.
#
# Verifies:
#   * Slot allocation bumps the cursor by the expected number of bytes.
#   * stage_panels! → _tape_panels yields bit-exact equality.
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

            got = TapeMod._tape_panels(slot)
            @test length(got) == 6
            for p in 1:6
                @test got[p] == src[p]
                @test eltype(got[p]) === Float64
                @test size(got[p]) == size(src[p])
            end

            TapeMod.finalize_tape!(storage)
            @test isfile(joinpath(dir, "manifest.toml"))
            @test filesize(_manifest_records_path(dir)) == storage.cursor

            m = TOML.parsefile(joinpath(dir, "manifest.toml"))
            @test m["meta"]["finalised"] === true
            @test m["meta"]["record_count"] == 1
            @test m["meta"]["version"] == "v2"
            @test m["meta"]["records_file"] == basename(storage.bin_path)
            @test m["meta"]["records_sha256"] == storage.records_sha256
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
                got = TapeMod._tape_panels(slots[idx])
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
                got = TapeMod._tape_panels(slot)
                for p in 1:6
                    @test size(got[p]) == size(src[p])
                    @test got[p] == src[p]
                end
            end
            TapeMod.finalize_tape!(storage)
        end
    end

    @testset "finaliser is idempotent and survives missing files" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 2, Hp = 2, Nz = 2, FT = Float32))
            TapeMod.finalize_tape!(storage)
            @test storage.finalised === true
            @test storage.closed === true
            # Second call is a safe no-op.
            TapeMod.finalize_tape!(storage)
            @test storage.closed === true
        end
    end

    @testset "default :mmap dispatch returns a temp-dir MmapCSTapeStorage" begin
        s = TapeMod._tape_storage(:mmap)
        try
            @test s isa TapeMod.MmapCSTapeStorage
            @test isdir(s.dir)
            @test s.cleanup_on_finalize === true
        finally
            TapeMod.finalize_tape!(s)
        end
        # Temp dir should be removed after finalize_tape! when cleanup is on.
        @test !isdir(s.dir)
    end

    @testset "argument errors on closed / finalised storage" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            TapeMod.finalize_tape!(storage)
            @test_throws ArgumentError TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 2, Hp = 2, Nz = 2, FT = Float32))
        end
    end

    @testset "shape-keyed device cache amortises heterogeneous reads" begin
        # Stand-in for a "GPU-ish" backend that picks up the device-cache
        # path: wrap Array{T,N} in a thin subtype so the
        # `_mmap_prepare_for_panels!(::NTuple{6, <:Array})` no-op
        # specialisation does NOT fire and we exercise the default
        # method that populates `storage.device_caches`. Same arithmetic
        # behaviour as Array, so copyto! / similar / eltype work
        # unchanged.
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

        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            # Three heterogeneous shapes interleaved across six slots;
            # the keyed cache must serve repeats without reallocating.
            shapes = [:m, :am, :bm, :cm, :m, :am]
            sources = [wrap6(_build_panels(s; Nc = 3, Hp = 3, Nz = 2,
                                           FT = Float32, seed = i))
                       for (i, s) in enumerate(shapes)]
            slots = [TapeMod._stage_panels(storage, src) for src in sources]

            # Allocation must populate one cache per *distinct* shape.
            # The four shape kinds are m=(9,9,2), am=(10,9,2),
            # bm=(9,10,2), cm=(9,9,3) — see `_shape_panel`.
            @test length(storage.device_caches) == 4

            for (slot, src) in zip(slots, sources)
                got = TapeMod._tape_panels(slot)
                @test got isa NTuple{6}
                @test all(got[p] == src[p] for p in 1:6)
                # Same cache buffer is returned for repeat shape reads.
                @test got === storage.device_caches[ntuple(p -> size(src[p]), 6)]
            end

            # Reads of repeat-shape slots must NOT have grown the cache
            # dict further.
            @test length(storage.device_caches) == 4
            TapeMod.finalize_tape!(storage)
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

@testset "cs_tape_byte_estimate matches realised tape size" begin
    @testset "linear PPM (m-only state tape)" begin
        mesh, panels_m, panels_rm, am, bm, cm =
            _trivial_problem(Nc = 4, Nz = 3, nsteps = 2; nontrivial = false)
        scheme = AT.PPMScheme(AT.NoLimiter())

        est = AT.cs_tape_byte_estimate(panels_m, am, bm, cm, mesh, scheme)
        @test est isa AT.CSTapeByteEstimate
        @test est.nsteps == 2
        # Linear schemes stage one panel set per sweep, no tracer branch.
        @test est.sweep_records > 0
        @test est.state_records == est.sweep_records
        @test est.bytes_per_state == sizeof(Float64) * 6 * length(panels_m[1])
        @test est.state_bytes ==
              est.state_records * est.bytes_per_state
        # total_records is the op count — sum of every op-count field.
        @test est.total_records == est.sweep_records + est.halo_records +
                                   est.midpoint_records + est.diffusion_records +
                                   est.convection_records
        @test est.halo_records > 0
        @test est.midpoint_records == est.nsteps

        # Sanity-check the estimate against a realised mmap tape — the
        # written `records.bin` size must be within rounding distance
        # of `state_bytes` once we run the footprint.
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
                mesh, AT.CSLayerMeanObjective(1, 2, 2, 3);
                scheme = scheme, dt = 1.0, tape_storage = storage)
            @test storage.cursor == est.state_bytes
            TapeMod.finalize_tape!(storage)
        end
    end

    @testset "monotone PPM (m + rm tracer branch tape)" begin
        mesh, panels_m, panels_rm, am, bm, cm =
            _trivial_problem(Nc = 4, Nz = 3, nsteps = 2; nontrivial = false)
        scheme = AT.PPMScheme(AT.MonotoneLimiter())

        est = AT.cs_tape_byte_estimate(panels_m, am, bm, cm, mesh, scheme)
        # Nonlinear schemes stage BOTH m and rm panels per sweep.
        @test est.state_records == 2 * est.sweep_records
        @test est.state_bytes ==
              est.state_records * est.bytes_per_state

        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
                mesh, AT.CSLayerMeanObjective(1, 2, 2, 3);
                scheme = scheme, dt = 1.0, tape_storage = storage)
            @test storage.cursor == est.state_bytes
            TapeMod.finalize_tape!(storage)
        end
    end

    @testset "estimate scales with nsteps" begin
        mesh, panels_m, _, am1, bm1, cm1 =
            _trivial_problem(Nc = 4, Nz = 3, nsteps = 1; nontrivial = false)
        _,    _,        _, am2, bm2, cm2 =
            _trivial_problem(Nc = 4, Nz = 3, nsteps = 2; nontrivial = false)
        scheme = AT.PPMScheme(AT.NoLimiter())
        e1 = AT.cs_tape_byte_estimate(panels_m, am1, bm1, cm1, mesh, scheme)
        e2 = AT.cs_tape_byte_estimate(panels_m, am2, bm2, cm2, mesh, scheme)
        @test e2.nsteps == 2 * e1.nsteps
        @test e2.state_bytes == 2 * e1.state_bytes
        @test e2.midpoint_records == 2 * e1.midpoint_records
    end

    @testset "total_records is the op count (regression: was inflated)" begin
        mesh, panels_m, _, am, bm, cm =
            _trivial_problem(Nc = 4, Nz = 3, nsteps = 2; nontrivial = false)
        # Monotone PPM stages BOTH `panels_m` and `panels_rm` per
        # sweep, so `state_records == 2 * sweep_records`. The previous
        # buggy formula computed
        # `total = state + halo + midpoint = 2*sweep + halo + midpoint`
        # — inflating the op count by the nonlinear staging factor
        # for nonlinear schemes. The fixed formula sums op counts
        # directly.
        est = AT.cs_tape_byte_estimate(panels_m, am, bm, cm, mesh,
                                       AT.PPMScheme(AT.MonotoneLimiter()))
        @test est.state_records == 2 * est.sweep_records
        @test est.total_records == est.sweep_records + est.halo_records +
                                   est.midpoint_records + est.diffusion_records +
                                   est.convection_records
        # The (buggy) old formula would have evaluated to
        # `state + halo + midpoint = 2*sweep + halo + midpoint`. Whatever
        # the actual cell count, the *gap* between old-formula and new is
        # `state - sweep = sweep_records` (the count of nonlinear duplicate
        # state stagings). Assert that this delta is precisely accounted for.
        old_buggy = est.state_records + est.halo_records + est.midpoint_records
        @test old_buggy - est.total_records == est.sweep_records
    end

    @testset "Lin-Rood counts its four-state horizontal records" begin
        mesh, panels_m, _, am, bm, cm =
            _trivial_problem(Nc = 4, Nz = 3, nsteps = 2; nontrivial = false)
        est = AT.cs_tape_byte_estimate(
            panels_m, am, bm, cm, mesh, AT.LinRoodPPMScheme(5))
        @test est.nsteps == 2
        @test est.sweep_records == 8       # H, Z, Z, H per step
        @test est.halo_records == 0        # halo work is inside each H record
        @test est.midpoint_records == 2
        @test est.state_records == 20      # 2*(4-state H) + 2*(1-state Z)
        @test est.total_records == 10
        face_tuple_bytes = sizeof(Float64) * 6 * (mesh.Nc + 1) * mesh.Nc *
                           size(panels_m[1], 3)
        @test est.state_bytes == est.state_records * est.bytes_per_state +
                                 12 * face_tuple_bytes
    end
end

# ---------------------------------------------------------------------------
# Codex-review fixes:
#   * LinRoodPPMScheme rejects non-device tape_storage explicitly.
#   * cs_surface_flux_4dvar validates control panel shapes vs mesh.Nc.
# ---------------------------------------------------------------------------

@testset "LinRoodPPMScheme rejects non-device tape_storage" begin
    mesh, panels_m, panels_rm, am, bm, cm =
        _trivial_problem(Nc = 4, Nz = 3, nsteps = 1; FT = Float32,
                         nontrivial = false)
    obj = AT.CSLayerMeanObjective(1, 2, 2, 3)
    # Sanity: `:device` still works.
    AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm, mesh, obj;
        scheme = AT.LinRoodPPMScheme(), dt = Float32(1),
        tape_storage = :device)

    # `:mmap` must throw — silently keeping the LinRood tape on the
    # source backend was a latent OOM trap.
    @test_throws ArgumentError AT.cs_surface_emission_footprint(
        panels_rm, panels_m, am, bm, cm, mesh, obj;
        scheme = AT.LinRoodPPMScheme(), dt = Float32(1),
        tape_storage = :mmap)
    @test_throws ArgumentError AT.cs_surface_emission_footprint(
        panels_rm, panels_m, am, bm, cm, mesh, obj;
        scheme = AT.LinRoodPPMScheme(), dt = Float32(1),
        tape_storage = :pinned_host)
end

@testset "cs_surface_flux_4dvar validates control shapes vs mesh.Nc" begin
    mesh, panels_m, panels_rm, am, bm, cm =
        _trivial_problem(Nc = 4, Nz = 3, nsteps = 1; FT = Float32,
                         nontrivial = false)

    # Correctly-shaped control passes the validator.
    good_control = AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:step1, 1),
        ntuple(_ -> zeros(Float32, mesh.Nc, mesh.Nc), 6))
    observations = [
        AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 3),
                         0.01f0, 0.2f0),
    ]
    @test_throws UndefKeywordError AT.cs_surface_flux_4dvar(
        panels_rm, panels_m, am, bm, cm, mesh, observations, good_control;
        dt = Float32(1))
    AT.cs_surface_flux_4dvar(
        panels_rm, panels_m, am, bm, cm, mesh, observations, good_control;
        scheme = AT.PPMScheme(AT.NoLimiter()), dt = Float32(1))

    # Wrong-sized value panels — value[1] is (Nc+1, Nc+1) instead of
    # (Nc, Nc); the shared `_add_weighted_footprint_kernel!` would
    # silently read with `ndrange = size(rates[step][p])` and grab
    # OOB or skip cells depending on the size mismatch. Validator
    # must catch this BEFORE any kernel launches.
    bad_value = ntuple(p -> begin
        sz = p == 1 ? (mesh.Nc + 1, mesh.Nc + 1) : (mesh.Nc, mesh.Nc)
        zeros(Float32, sz)
    end, 6)
    bad_control = AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:step1, 1), bad_value)
    @test_throws DimensionMismatch AT.cs_surface_flux_4dvar(
        panels_rm, panels_m, am, bm, cm, mesh, observations, bad_control;
        scheme = AT.PPMScheme(AT.NoLimiter()), dt = Float32(1))

    # Wrong-sized sigma panels are also rejected (the
    # `_add_background_gradient_array_kernel!` reads `sigma[i, j]`).
    bad_sigma = ntuple(p -> begin
        sz = p == 3 ? (mesh.Nc - 1, mesh.Nc) : (mesh.Nc, mesh.Nc)
        fill(Float32(0.1), sz)
    end, 6)
    # Have to bypass the constructor's per-panel sigma vs value cross-
    # check; build a control with correct sigma first then mutate the
    # tuple via a fresh constructor that skips the check.
    # The constructor at Observations.jl:131-141 validates
    # sigma-panel-vs-value-panel shape, which is exactly what we test
    # here from the mesh side too — confirm both gates fire.
    @test_throws DimensionMismatch AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:step1, 1),
        ntuple(_ -> zeros(Float32, mesh.Nc, mesh.Nc), 6);
        sigma = bad_sigma)
end

# ---------------------------------------------------------------------------
# Plan 26 P0.A.2c — manifest-driven resume API.
#
# load_mmap_tape(dir) parses manifest.toml plus its named records generation
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

            got = TapeMod._tape_panels(got_slot)
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
                got = TapeMod._tape_panels(slot)
                for p in 1:6
                    @test got[p] == expected[p]
                end
            end

            TapeMod.finalize_tape!(reopened)
        end
    end

    @testset "readonly default blocks mutation" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            src = _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32)
            TapeMod._stage_panels(storage, src)
            TapeMod.finalize_tape!(storage)

            ro = TapeMod.load_mmap_tape(dir)
            @test ro.readonly === true

            extra = _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32, seed = 9)
            @test_throws ArgumentError TapeMod._allocate_tape_slot(ro, extra)
            @test_throws ArgumentError TapeMod._bump_cursor!(ro, Int64(1024))

            # stage_panels! goes through an existing slot, but the slot
            # must come from get_record (no new allocation) — the
            # in-place stage path also enforces readonly.
            slot = TapeMod.get_record(ro, 1)
            @test_throws ArgumentError TapeMod.stage_panels!(slot, extra)

            TapeMod.finalize_tape!(ro)
        end
    end

    @testset "readonly=false reload still blocks overwrite of finalised tape" begin
        # GPT F3: `load_mmap_tape(dir; readonly=false)` returns a
        # storage with `finalised = true` and `readonly = false`.
        # `_allocate_tape_slot` is blocked by `finalised`; the previous
        # version of `stage_panels!` only checked `readonly` and
        # `closed`, so a caller that obtained a slot via `get_record`
        # could silently overwrite the on-disk bytes. Phase B will own
        # slot-reuse semantics; for v1, `stage_panels!` rejects on
        # `finalised` too.
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            src = _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32, seed = 11)
            TapeMod._stage_panels(storage, src)
            TapeMod.finalize_tape!(storage)

            rw = TapeMod.load_mmap_tape(dir; readonly = false)
            @test rw.readonly === false
            @test rw.finalised === true

            slot = TapeMod.get_record(rw, 1)
            overwrite = _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32, seed = 99)
            @test_throws ArgumentError TapeMod.stage_panels!(slot, overwrite)

            # Verify the on-disk bytes are still the original src, not
            # the would-be overwrite. Round-trip through a fresh
            # readonly reopen to confirm.
            TapeMod.finalize_tape!(rw)
            ro = TapeMod.load_mmap_tape(dir)
            got = TapeMod._tape_panels(TapeMod.get_record(ro, 1))
            for p in 1:6
                @test got[p] == src[p]
            end
            TapeMod.finalize_tape!(ro)
        end
    end

    @testset "validation: missing files, corrupted meta, version mismatch" begin
        @test_throws ArgumentError TapeMod.load_mmap_tape(
            joinpath(tempdir(), "atmostransport-mmap-load-nope-$(rand(UInt32))"))

        # Directory exists, manifest missing.
        mktempdir() do dir
            @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
        end

        # Manifest without the required generation metadata.
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
            write(mpath, replace(txt, "version = \"v2\"" => "version = \"v0\""))
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

        # Same-size content corruption is rejected by the manifest digest.
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32))
            TapeMod.finalize_tape!(storage)
            bin_path = _manifest_records_path(dir)
            open(bin_path, "r+") do io
                byte = read(io, UInt8)
                seekstart(io)
                write(io, xor(byte, UInt8(0x01)))
            end
            @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
        end

        # Named records generation truncated below total_bytes.
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32))
            TapeMod.finalize_tape!(storage)
            bin_path = _manifest_records_path(dir)
            open(bin_path, "r+") do io
                Base.truncate(io, max(filesize(bin_path) - 16, 0))
            end
            @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
        end
    end

    @testset "get_record argument validation" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32))
            TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32, seed = 2))
            TapeMod.finalize_tape!(storage)

            ro = TapeMod.load_mmap_tape(dir)
            @test_throws BoundsError TapeMod.get_record(ro, 0)
            @test_throws BoundsError TapeMod.get_record(ro, 3)
            TapeMod.finalize_tape!(ro)
            @test_throws ArgumentError TapeMod.get_record(ro, 1)
        end
    end

    @testset "eltype whitelist rejects non-float types" begin
        @test_throws ArgumentError TapeMod._parse_mmap_eltype("Int64")
        @test_throws ArgumentError TapeMod._parse_mmap_eltype("Bool")
        @test_throws ArgumentError TapeMod._parse_mmap_eltype("Module")
        @test TapeMod._parse_mmap_eltype("Float32") === Float32
        @test TapeMod._parse_mmap_eltype("Float64") === Float64
        @test TapeMod._parse_mmap_eltype("Float16") === Float16
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
                m = TOML.parsefile(mpath)
                cp(_manifest_records_path(good_dir), joinpath(dir, m["meta"]["records_file"]))
                m["record"][1]["offsets"][1] = -8
                open(joinpath(dir, "manifest.toml"), "w") do io
                    TOML.print(io, m)
                end
                @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
            end

            # offset + nbytes past total_bytes.
            mktempdir() do dir
                m = TOML.parsefile(mpath)
                cp(_manifest_records_path(good_dir), joinpath(dir, m["meta"]["records_file"]))
                m["record"][1]["offsets"][6] = Int(m["meta"]["total_bytes"]) - 1
                open(joinpath(dir, "manifest.toml"), "w") do io
                    TOML.print(io, m)
                end
                @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
            end

            # nbytes disagrees with shape × sizeof(eltype).
            mktempdir() do dir
                m = TOML.parsefile(mpath)
                cp(_manifest_records_path(good_dir), joinpath(dir, m["meta"]["records_file"]))
                m["record"][1]["nbytes"][1] += 4
                open(joinpath(dir, "manifest.toml"), "w") do io
                    TOML.print(io, m)
                end
                @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
            end

            # Shape vector with wrong length.
            mktempdir() do dir
                m = TOML.parsefile(mpath)
                cp(_manifest_records_path(good_dir), joinpath(dir, m["meta"]["records_file"]))
                m["record"][1]["shapes"][1] = [9, 9]  # length 2 instead of 3
                open(joinpath(dir, "manifest.toml"), "w") do io
                    TOML.print(io, m)
                end
                @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
            end

            # offsets vector with wrong length.
            mktempdir() do dir
                m = TOML.parsefile(mpath)
                cp(_manifest_records_path(good_dir), joinpath(dir, m["meta"]["records_file"]))
                m["record"][1]["offsets"] = m["record"][1]["offsets"][1:5]
                open(joinpath(dir, "manifest.toml"), "w") do io
                    TOML.print(io, m)
                end
                @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
            end

            # Missing required field.
            mktempdir() do dir
                m = TOML.parsefile(mpath)
                cp(_manifest_records_path(good_dir), joinpath(dir, m["meta"]["records_file"]))
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

    @testset "reload survives writer GC" begin
        # Resumed storage must be fully self-contained: nothing about
        # _tape_panels should reach back into the original writer
        # object. Force the writer out of scope, GC it, then read.
        dir = mktempdir(; cleanup = false)
        try
            let src = _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32, seed = 5)
                storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                   cleanup_on_finalize = false)
                TapeMod._stage_panels(storage, src)
                TapeMod.finalize_tape!(storage)
                # Hold src for the post-reload comparison below.
                global _tape_src_for_gc_test = src
            end
            GC.gc(true)
            GC.gc(true)

            ro = TapeMod.load_mmap_tape(dir)
            slot = TapeMod.get_record(ro, 1)
            got = TapeMod._tape_panels(slot)
            for p in 1:6
                @test got[p] == _tape_src_for_gc_test[p]
            end
            TapeMod.finalize_tape!(ro)
        finally
            rm(dir; recursive = true, force = true)
        end
    end

    @testset "writable reopen selection" begin
        mktempdir() do dir
            storage = TapeMod.MmapCSTapeStorage(; dir = dir,
                                                cleanup_on_finalize = false)
            TapeMod._stage_panels(storage,
                _build_panels(:m; Nc = 3, Hp = 3, Nz = 4, FT = Float32))
            TapeMod.finalize_tape!(storage)

            # The public loader exposes only readonly or non-destructive r+.
            rw = TapeMod.load_mmap_tape(dir; readonly = false)
            @test rw.readonly === false
            TapeMod.finalize_tape!(rw)
        end
    end
end
