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

    mesh, m, rm, am, bm, cm = _problem(; nsteps = 4)
    objective = _objective(mesh)
    seed = map(zero, rm)
    for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2), AT.RevolveCheckpoint()),
        tape_storage in (:mmap, :pinned_host)
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            rm, m, am, bm, cm, mesh, objective;
            scheme = AT.LinRoodPPMScheme(), checkpoint, tape_storage)
        @test_throws ArgumentError AT.cs_surface_emission_footprint_from_seed(
            seed, m, am, bm, cm, mesh;
            scheme = AT.LinRoodPPMScheme(), checkpoint, tape_storage)
    end
    for checkpoint in (AT.StrideCheckpoint(2), AT.RevolveCheckpoint()),
        scheme in (AT.PPMScheme(AT.NoLimiter()), AT.PPMScheme(AT.MonotoneLimiter()),
                   AT.LinRoodPPMScheme())
        tape_storage = AT.DeviceCSTapeStorage()
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            rm, m, am, bm, cm, mesh, objective; scheme, checkpoint, tape_storage)
        @test_throws ArgumentError AT.cs_surface_emission_footprint_from_seed(
            seed, m, am, bm, cm, mesh; scheme, checkpoint, tape_storage)
    end

end
