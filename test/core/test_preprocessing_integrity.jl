#!/usr/bin/env julia

using Test
using Dates

import AtmosTransport
const AT = AtmosTransport
const Prep = AtmosTransport.Preprocessing

@testset "cubed-sphere geometry rejects obsolete configuration aliases" begin
    cfg = Dict{String,Any}("Nc" => 2, "convention" => "geos_native")
    @test_throws ArgumentError Prep.build_target_geometry(
        Val(:cubed_sphere), cfg, Float64)
end

@testset "reduced dry-basis conversion skips polar stub faces" begin
    mesh = AT.ReducedGaussianMesh(Float64[-60, -20, 20, 60], [8, 12, 12, 8])
    grid = Prep.ReducedGaussianTargetGeometry(
        mesh, "synthetic", 2, copy(mesh.nlon_per_ring), copy(mesh.latitudes),
        [collect(range(0.0, 360.0; length=n + 1))[1:end-1] for n in mesh.nlon_per_ring])
    work = Prep.allocate_reduced_transform_workspace(grid, 3, 2)
    fill!(work.m_arr, 1.0)
    fill!(work.hflux_arr, 1.0)
    qv = fill(0.2, size(work.m_arr))

    Prep.apply_dry_basis_reduced!(work, qv)

    stub_faces = findall(f -> work.face_left[f] == 0 || work.face_right[f] == 0,
                         eachindex(work.face_left))
    real_faces = setdiff(collect(eachindex(work.face_left)), stub_faces)
    @test !isempty(stub_faces)
    @test all(iszero, work.hflux_arr[stub_faces, :])
    @test all(==(0.8), work.hflux_arr[real_faces, :])
    @test all(==(0.8), work.m_arr)
end

@testset "spectral coverage fails closed" begin
    hours = 0:2
    complete = Dict(hour => trues(3) for hour in hours)
    @test Prep.validate_spectral_coverage(
        hours, complete, deepcopy(complete); expected_hours=hours, nlevels=3) == collect(hours)

    missing_level = deepcopy(complete)
    missing_level[1][2] = false
    @test_throws ArgumentError Prep.validate_spectral_coverage(
        hours, missing_level, complete; expected_hours=hours, nlevels=3)

    missing_hour = Dict(0 => trues(3), 2 => trues(3))
    @test_throws ArgumentError Prep.validate_spectral_coverage(
        hours, missing_hour, complete; expected_hours=hours, nlevels=3)
    @test Prep.SPECTRAL_DAY_CACHE_VERSION == 2
end

@testset "hour-0 reader distinguishes absent and incomplete inputs" begin
    date = Date(2021, 12, 1)
    mktempdir() do dir
        @test Prep.read_hour0_spectral(dir, date) === nothing

        lnsp = joinpath(dir, "era5_spectral_20211201_lnsp.gb")
        touch(lnsp)
        @test_throws ArgumentError Prep.read_hour0_spectral(dir, date)
        rm(lnsp)

        vo_d = joinpath(dir, "era5_spectral_20211201_vo_d.gb")
        touch(vo_d)
        @test_throws ArgumentError Prep.read_hour0_spectral(dir, date)
    end
end
