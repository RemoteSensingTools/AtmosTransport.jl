#!/usr/bin/env julia

using Test

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection
const TapeMod = AtmosTransport.Tape

# ---------------------------------------------------------------------------
# Plan 26 P0.A3 — `tape_path` kwarg on `cs_surface_emission_footprint` and
# `cs_surface_emission_footprint_from_seed`.
#
# Verifies:
#   * FullCheckpoint + `:mmap` + `tape_path` produces `records.bin` +
#     `manifest.toml` at the user-supplied directory and matches the
#     temp-dir default to roundoff.
#   * The user's directory is preserved after the call (no cleanup).
#   * `load_mmap_tape(tape_path)` can reopen the finalised tape.
#   * StrideCheckpoint with tape_path creates per-window subdirectories
#     (`window_NNNNN/{records.bin,manifest.toml}`) and matches the
#     temp-dir default to roundoff.
#   * RevolveCheckpoint with tape_path creates per-base-case subdirs
#     (`step_NNNNN/{records.bin,manifest.toml}`).
#   * Validation: `tape_path` with non-`:mmap` storage, with
#     pre-constructed `AbstractCSTapeStorage`, with LinRood, and with
#     empty string all raise `ArgumentError`.
#   * `tape_path` survives a missing-leaf-dir (auto-`mkpath`).
# ---------------------------------------------------------------------------

function _problem(; Nc = 4, Nz = 3, nsteps = 4, FT = Float64)
    mesh = AT.CubedSphereMesh(Nc = Nc, Hp = 3, FT = FT)
    N = mesh.Nc + 2mesh.Hp
    Hp = mesh.Hp

    panels_m = ntuple(6) do p
        m = zeros(FT, N, N, Nz)
        @inbounds for k in 1:Nz, j in 1:N, i in 1:N
            m[i, j, k] = FT(2.0 + 0.25k + 0.01p + 0.0001(i + j))
        end
        m
    end
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    Adv.fill_panel_halos!(panels_m, mesh; dir = 0)
    Adv.fill_panel_halos!(panels_rm, mesh; dir = 0)

    panels_am_steps = Vector{Any}(undef, nsteps)
    panels_bm_steps = Vector{Any}(undef, nsteps)
    panels_cm_steps = Vector{Any}(undef, nsteps)
    for step in 1:nsteps
        panels_am_steps[step] = ntuple(6) do p
            am = zeros(FT, N + 1, N, Nz)
            @inbounds for k in 1:Nz, j in Hp + 1:Hp + mesh.Nc, i in Hp + 1:Hp + mesh.Nc + 1
                am[i, j, k] = FT(0.013) * sin(FT(0.21step + 0.31p + 0.71i + 0.41j + 0.23k))
            end
            am
        end
        panels_bm_steps[step] = ntuple(6) do p
            bm = zeros(FT, N, N + 1, Nz)
            @inbounds for k in 1:Nz, j in Hp + 1:Hp + mesh.Nc + 1, i in Hp + 1:Hp + mesh.Nc
                bm[i, j, k] = FT(0.011) * cos(FT(0.33step + 0.51p + 0.41i + 0.61j + 0.13k))
            end
            bm
        end
        panels_cm_steps[step] = ntuple(6) do p
            cm = zeros(FT, N, N, Nz + 1)
            @inbounds for k in 2:Nz, j in Hp + 1:Hp + mesh.Nc, i in Hp + 1:Hp + mesh.Nc
                cm[i, j, k] = -FT(0.009) * (one(FT) + FT(0.1) * sin(FT(i + j + k + p + step)))
            end
            cm
        end
    end
    return mesh, panels_m, panels_rm, panels_am_steps, panels_bm_steps, panels_cm_steps
end

_objective(mesh::AT.CubedSphereMesh) =
    AT.CSColumnMeanObjective(1, max(2, mesh.Nc ÷ 2), max(2, mesh.Nc ÷ 2))

const FOOTPRINT_REPLAY_ATOL = 1e-12
const FOOTPRINT_REPLAY_RTOL = 1e-10

function _footprints_equal(a, b; atol = FOOTPRINT_REPLAY_ATOL, rtol = FOOTPRINT_REPLAY_RTOL)
    length(a.footprints) == length(b.footprints) || return false
    for step in eachindex(a.footprints)
        for p in 1:6
            if atol == 0.0 && rtol == 0.0
                a.footprints[step][p] == b.footprints[step][p] || return false
            else
                isapprox(a.footprints[step][p], b.footprints[step][p];
                         atol = atol, rtol = rtol) || return false
            end
        end
    end
    return true
end

@testset "tape_path — FullCheckpoint single-tape directory" begin
    mesh, panels_m, panels_rm, am, bm, cm = _problem(; Nc = 4, Nz = 3, nsteps = 3)
    obj = _objective(mesh)
    scheme = AT.PPMScheme(AT.NoLimiter())

    # Baseline: default temp dir.
    base = AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
        mesh, obj; scheme = scheme, dt = 1.0, tape_storage = :mmap)

    mktempdir() do parent
        dir = joinpath(parent, "fullcp_tape")
        @test !isdir(dir)
        got = AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
            mesh, obj; scheme = scheme, dt = 1.0,
            tape_storage = :mmap, tape_path = dir)

        # The user-supplied directory is created, finalised, and preserved.
        @test isdir(dir)
        @test isfile(joinpath(dir, "records.bin"))
        @test isfile(joinpath(dir, "manifest.toml"))

        # Loader reopens the finalised tape.
        ro = TapeMod.load_mmap_tape(dir)
        @test ro.cursor > 0
        @test length(ro.records) > 0
        TapeMod.finalize_tape!(ro)

        # Roundoff parity with the temp-dir default — same kernels, same order.
        @test _footprints_equal(got, base)
    end
end

@testset "tape_path — FullCheckpoint auto-creates leaf directory" begin
    mesh, panels_m, panels_rm, am, bm, cm = _problem(; Nc = 4, Nz = 3, nsteps = 2)
    obj = _objective(mesh)
    scheme = AT.PPMScheme(AT.NoLimiter())

    mktempdir() do parent
        dir = joinpath(parent, "deeply", "nested", "tape_root")
        @test !isdir(dir)
        AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
            mesh, obj; scheme = scheme, dt = 1.0,
            tape_storage = :mmap, tape_path = dir)
        @test isdir(dir)
        @test isfile(joinpath(dir, "manifest.toml"))
    end
end

@testset "tape_path — StrideCheckpoint per-window subdirectories" begin
    mesh, panels_m, panels_rm, am, bm, cm = _problem(; Nc = 4, Nz = 3, nsteps = 6)
    obj = _objective(mesh)
    scheme = AT.PPMScheme(AT.NoLimiter())
    K = 2

    base = AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
        mesh, obj; scheme = scheme, dt = 1.0,
        tape_storage = :mmap, checkpoint = TapeMod.StrideCheckpoint(K))

    mktempdir() do parent
        dir = joinpath(parent, "stride_tape")
        got = AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
            mesh, obj; scheme = scheme, dt = 1.0,
            tape_storage = :mmap, tape_path = dir,
            checkpoint = TapeMod.StrideCheckpoint(K))

        nw = TapeMod.checkpoint_window_count(TapeMod.StrideCheckpoint(K), 6)
        @test nw == 3
        for w in 1:nw
            sub = joinpath(dir, "window_" * lpad(w, 5, '0'))
            @test isdir(sub)
            @test isfile(joinpath(sub, "records.bin"))
            @test isfile(joinpath(sub, "manifest.toml"))
        end

        # Roundoff parity with the temp-dir default for the same K.
        @test _footprints_equal(got, base)
    end
end

@testset "tape_path — RevolveCheckpoint per-base-case subdirectories" begin
    mesh, panels_m, panels_rm, am, bm, cm = _problem(; Nc = 4, Nz = 3, nsteps = 4)
    obj = _objective(mesh)
    scheme = AT.PPMScheme(AT.NoLimiter())

    base = AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
        mesh, obj; scheme = scheme, dt = 1.0,
        tape_storage = :mmap, checkpoint = TapeMod.RevolveCheckpoint())

    mktempdir() do parent
        dir = joinpath(parent, "revolve_tape")
        got = AT.cs_surface_emission_footprint(panels_rm, panels_m, am, bm, cm,
            mesh, obj; scheme = scheme, dt = 1.0,
            tape_storage = :mmap, tape_path = dir,
            checkpoint = TapeMod.RevolveCheckpoint())

        # Revolve hits each step's base case exactly once: nsteps subdirs.
        for step in 1:4
            sub = joinpath(dir, "step_" * lpad(step, 5, '0'))
            @test isdir(sub)
            @test isfile(joinpath(sub, "records.bin"))
            @test isfile(joinpath(sub, "manifest.toml"))
        end

        @test _footprints_equal(got, base)
    end
end

@testset "tape_path — from-seed entry point" begin
    mesh, panels_m, panels_rm, am, bm, cm = _problem(; Nc = 4, Nz = 3, nsteps = 2)
    scheme = AT.PPMScheme(AT.NoLimiter())
    FT = Float64
    final_seed = ntuple(p -> begin
        a = similar(panels_m[p])
        fill!(a, zero(FT))
        a[mesh.Hp + 2, mesh.Hp + 2, 1] = FT(1.0)
        a
    end, 6)

    base = AT.cs_surface_emission_footprint_from_seed(final_seed, panels_m, am, bm, cm,
        mesh; scheme = scheme, dt = 1.0, tape_storage = :mmap)

    mktempdir() do parent
        dir = joinpath(parent, "from_seed_tape")
        got = AT.cs_surface_emission_footprint_from_seed(final_seed, panels_m, am, bm, cm,
            mesh; scheme = scheme, dt = 1.0,
            tape_storage = :mmap, tape_path = dir)
        @test isfile(joinpath(dir, "manifest.toml"))
        @test _footprints_equal(got, base)
    end
end

@testset "tape_path — validation rejections" begin
    mesh, panels_m, panels_rm, am, bm, cm = _problem(; Nc = 4, Nz = 3, nsteps = 2)
    obj = _objective(mesh)
    scheme = AT.PPMScheme(AT.NoLimiter())

    mktempdir() do parent
        # tape_path requires :mmap storage.
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am, bm, cm, mesh, obj;
            scheme = scheme, dt = 1.0,
            tape_storage = :device, tape_path = joinpath(parent, "x"))
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am, bm, cm, mesh, obj;
            scheme = scheme, dt = 1.0,
            tape_storage = :pinned_host, tape_path = joinpath(parent, "x"))

        # Empty tape_path is rejected.
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am, bm, cm, mesh, obj;
            scheme = scheme, dt = 1.0,
            tape_storage = :mmap, tape_path = "")

        # Pre-constructed storage + tape_path is rejected.
        pre_dir = joinpath(parent, "pre")
        mkpath(pre_dir)
        pre_storage = TapeMod.MmapCSTapeStorage(; dir = pre_dir,
                                                cleanup_on_finalize = false)
        try
            @test_throws ArgumentError AT.cs_surface_emission_footprint(
                panels_rm, panels_m, am, bm, cm, mesh, obj;
                scheme = scheme, dt = 1.0,
                tape_storage = pre_storage,
                tape_path = joinpath(parent, "x"))
        finally
            TapeMod.finalize_tape!(pre_storage; quiet = true)
        end

        # LinRood: tape_path is rejected (storage is :device only).
        linrood = AT.LinRoodPPMScheme(5)
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am, bm, cm, mesh, obj;
            scheme = linrood, dt = 1.0,
            tape_storage = :device, tape_path = joinpath(parent, "x"))

        # LinRood + Stride + tape_path rejected.
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am, bm, cm, mesh, obj;
            scheme = linrood, dt = 1.0,
            tape_storage = :device, tape_path = joinpath(parent, "x"),
            checkpoint = TapeMod.StrideCheckpoint(2))

        # LinRood + Revolve + tape_path rejected.
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am, bm, cm, mesh, obj;
            scheme = linrood, dt = 1.0,
            tape_storage = :device, tape_path = joinpath(parent, "x"),
            checkpoint = TapeMod.RevolveCheckpoint())
    end
end

@testset "tape_path — LinRood reject is side-effect free (P2)" begin
    # Reviewer P2: a LinRood + tape_path request used to mkpath the
    # directory before LinRood's storage validation fired, leaving an
    # empty records.bin / manifest.toml behind. After the fix,
    # `_require_tape_path_supported` rejects at the public API entry
    # before `_resolve_tape_path` ever calls `mkpath`.
    mesh, panels_m, panels_rm, am, bm, cm = _problem(; Nc = 4, Nz = 3, nsteps = 2)
    obj = _objective(mesh)
    linrood = AT.LinRoodPPMScheme(5)

    # NOTE: pass a non-existent parent so `mkpath` would be observable.
    mktempdir() do parent
        dir = joinpath(parent, "linrood_should_not_exist")
        @test !ispath(dir)

        # FullCheckpoint LinRood + tape_storage=:mmap + tape_path.
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am, bm, cm, mesh, obj;
            scheme = linrood, dt = 1.0,
            tape_storage = :mmap, tape_path = dir)
        @test !ispath(dir)   # no mkpath happened

        # Stride LinRood + tape_storage=:mmap + tape_path.
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am, bm, cm, mesh, obj;
            scheme = linrood, dt = 1.0,
            tape_storage = :mmap, tape_path = dir,
            checkpoint = TapeMod.StrideCheckpoint(2))
        @test !ispath(dir)

        # Revolve LinRood + tape_storage=:mmap + tape_path.
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am, bm, cm, mesh, obj;
            scheme = linrood, dt = 1.0,
            tape_storage = :mmap, tape_path = dir,
            checkpoint = TapeMod.RevolveCheckpoint())
        @test !ispath(dir)

        # from-seed entry: same guard.
        FT = Float64
        final_seed = ntuple(p -> begin
            a = similar(panels_m[p])
            fill!(a, zero(FT))
            a[mesh.Hp + 2, mesh.Hp + 2, 1] = FT(1.0)
            a
        end, 6)
        @test_throws ArgumentError AT.cs_surface_emission_footprint_from_seed(
            final_seed, panels_m, am, bm, cm, mesh;
            scheme = linrood, dt = 1.0,
            tape_storage = :mmap, tape_path = dir)
        @test !ispath(dir)
    end
end

@testset "tape_path — strict finalize surfaces manifest failure (P1)" begin
    mktempdir() do parent
        dir = joinpath(parent, "broken_manifest")
        storage = TapeMod.MmapCSTapeStorage(; dir)
        # Point the final rename below a parent that does not exist. This is
        # deterministic even for privileged test processes, unlike chmod or
        # pre-creating the destination (atomic `mv(...; force=true)` replaces
        # both files and directories).
        storage.manifest_path = joinpath(dir, "missing", "manifest.toml")
        @test_throws Exception TapeMod.finalize_tape!(storage; strict = true)
        @test storage.closed
    end

    # Sanity: without tape_path, manifest failure is still
    # warn-and-continue (temp-dir behaviour unchanged) — the API
    # returns successfully because there's nothing the caller cares
    # to keep. Reproducible via finalize_tape!(temp_storage;
    # strict=false) which is the existing call path.
    @testset "finalize_tape! warn-mode preserved for temp-dir use" begin
        storage = TapeMod.MmapCSTapeStorage()
        storage.manifest_path = joinpath(storage.dir, "missing", "manifest.toml")
        TapeMod.finalize_tape!(storage)
        @test storage.closed
    end
end

@testset "tape_path — strict finalize for per-window storage (P1)" begin
    mktempdir() do parent
        dir = joinpath(parent, "stride_broken_manifest")
        storage = TapeMod._build_window_storage(:mmap, dir, "window_00001")
        storage.manifest_path = joinpath(storage.dir, "missing", "manifest.toml")
        @test_throws Exception TapeMod.finalize_tape!(storage; strict = true)
        @test storage.closed
    end
end
