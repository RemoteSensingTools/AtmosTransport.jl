#!/usr/bin/env julia

using Test
import NCDatasets: NCDataset, defDim, defVar

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Operators.Convection: TM5Workspace

const _RUNTIME_RECIPE_AIR_MASS = 1e16
const _RUNTIME_RECIPE_JANUARY_SECONDS = 31 * 86400

function _make_runtime_recipe_tm5_fields(::Type{FT}, Nx, Ny, Nz) where FT
    entu = zeros(FT, Nx, Ny, Nz)
    detu = zeros(FT, Nx, Ny, Nz)
    entd = zeros(FT, Nx, Ny, Nz)
    detd = zeros(FT, Nx, Ny, Nz)
    return (; entu, detu, entd, detd)
end

function write_runtime_recipe_binary(path::AbstractString; FT::Type{<:AbstractFloat} = Float64)
    Nx, Ny, Nz = 4, 3, 5
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(
        FT[0, 100, 300, 600, 1000, 2000],
        FT[0, 0, 0.1, 0.3, 0.7, 1],
    )
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    m = fill(FT(_RUNTIME_RECIPE_AIR_MASS), Nx, Ny, Nz)
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    ps = fill(FT(95_000), Nx, Ny)

    windows = [(
        m = m,
        am = am,
        bm = bm,
        cm = cm,
        ps = ps,
        tm5_fields = _make_runtime_recipe_tm5_fields(FT, Nx, Ny, Nz),
    )]

    write_transport_binary(path, grid, windows;
                           FT = FT,
                           dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0,
                           steps_per_window = 1,
                           mass_basis = :dry,
                           surface_pressure_basis = TotalSurfacePressure(),
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant)
    return nothing
end

function write_runtime_recipe_surface_flux_file(path::AbstractString)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "longitude", 4)
        defDim(ds, "latitude", 2)
        defDim(ds, "time", 1)

        vlon = defVar(ds, "longitude", Float64, ("longitude",))
        vlat = defVar(ds, "latitude", Float64, ("latitude",))
        vtime = defVar(ds, "time", Float64, ("time",))
        vtotal = defVar(ds, "TOTAL", Float64, ("longitude", "latitude", "time"))
        varea = defVar(ds, "cell_area", Float64, ("longitude", "latitude"))

        vlon[:] = [-135.0, -45.0, 45.0, 135.0]
        vlat[:] = [45.0, -45.0]
        vtime[:] = [1.0]
        raw = zeros(Float64, 4, 2)
        raw[1, 1] = 7 * _RUNTIME_RECIPE_JANUARY_SECONDS
        vtotal[:, :, 1] = raw
        varea[:, :] = reshape(Float64.(1:8), 4, 2)
        vtotal.attrib["units"] = "kgCO2/month/m2"
    finally
        close(ds)
    end
    return nothing
end

@testset "run_transport builds structured runtime recipes" begin
    mktempdir() do dir
        path = joinpath(dir, "tm5_runtime.bin")
        write_runtime_recipe_binary(path)

        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path]),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("start_window" => 1),
            "advection" => Dict("scheme" => "ppm"),
            "diffusion" => Dict("kind" => "constant", "value" => 2.0),
            "convection" => Dict("kind" => "tm5"),
            "init" => Dict("kind" => "uniform", "background" => 4.0e-4),
        )

        # Plan 40 Commit 6a: `make_model` is gone; run through the
        # unified library entry point. The model is returned at the end of
        # the loop, so for a 1-window input we can inspect it directly.
        model = run_driven_simulation(cfg)

        @test model.advection isa PPMScheme
        @test model.diffusion isa ImplicitVerticalDiffusion
        @test model.convection isa TM5Convection
        @test model.workspace.convection_ws isa TM5Workspace{Float64}
    end
end

@testset "run_transport accepts multiple binary paths" begin
    mktempdir() do dir
        path1 = joinpath(dir, "tm5_runtime_day1.bin")
        path2 = joinpath(dir, "tm5_runtime_day2.bin")
        write_runtime_recipe_binary(path1)
        write_runtime_recipe_binary(path2)

        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path1, path2]),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("start_window" => 1),
            "advection" => Dict("scheme" => "upwind"),
            "init" => Dict("kind" => "uniform", "background" => 4.0e-4),
        )

        model = run_driven_simulation(cfg)
        @test total_air_mass(model.state) ≈ 60 * _RUNTIME_RECIPE_AIR_MASS rtol = 1e-12
        @test total_mass(model.state, :CO2) ≈
              60 * _RUNTIME_RECIPE_AIR_MASS * 4.0e-4 rtol = 1e-12
    end
end

@testset "run_transport supports multi-tracer surface flux sources" begin
    mktempdir() do dir
        path = joinpath(dir, "tm5_runtime.bin")
        flux_path = joinpath(dir, "gridfed.nc")
        write_runtime_recipe_binary(path)
        write_runtime_recipe_surface_flux_file(flux_path)

        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path]),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("start_window" => 1),
            "advection" => Dict("scheme" => "upwind"),
            "tracers" => Dict(
                "natural_co2" => Dict(
                    "init" => Dict("kind" => "uniform", "background" => 4.0e-4),
                ),
                "fossil_co2" => Dict(
                    "init" => Dict("kind" => "uniform", "background" => 0.0),
                    "surface_flux" => Dict(
                        "kind" => "gridfed_fossil_co2",
                        "file" => flux_path,
                        "time_index" => 1,
                        "year" => 2021,
                    ),
                ),
            ),
        )

        model = run_driven_simulation(cfg)
        @test total_mass(model.state, :natural_co2) ≈
              60 * _RUNTIME_RECIPE_AIR_MASS * 4.0e-4 rtol = 1e-12
        storage_per_kg_co2 = AtmosTransport.Parameters.DRY_AIR_MOLAR_MASS /
                             AtmosTransport.Parameters.SPECIES_MOLAR_MASS.co2
        @test total_mass(model.state, :fossil_co2) ≈
              7.0 * storage_per_kg_co2 * 3600.0 rtol = 1e-12
    end
end

@testset "run_transport runtime recipe wiring" begin
    mktempdir() do dir
        path = joinpath(dir, "tm5_runtime.bin")
        write_runtime_recipe_binary(path)

        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path]),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("start_window" => 1),
            "advection" => Dict("scheme" => "ppm"),
            "convection" => Dict("kind" => "tm5"),
            "init" => Dict("kind" => "uniform", "background" => 4.0e-4),
        )

        model = run_driven_simulation(cfg)

        @test model.convection_forcing.tm5_fields !== nothing
        @test total_air_mass(model.state) ≈ 60 * _RUNTIME_RECIPE_AIR_MASS rtol = 1e-12
        @test total_mass(model.state, :CO2) ≈ 60 * _RUNTIME_RECIPE_AIR_MASS * 4.0e-4 rtol = 1e-12
    end
end

@testset "run_transport rejects unsupported recipe capabilities" begin
    mktempdir() do dir
        path = joinpath(dir, "tm5_runtime.bin")
        write_runtime_recipe_binary(path)

        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path]),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("start_window" => 1),
            "convection" => Dict("kind" => "cmfmc"),
            "init" => Dict("kind" => "uniform", "background" => 4.0e-4),
        )

        @test_throws ArgumentError run_driven_simulation(cfg)
    end
end
