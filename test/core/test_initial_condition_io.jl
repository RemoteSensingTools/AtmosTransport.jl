#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# test_initial_condition_io.jl — plan 40 Commit 1b regression tests.
#
# Covers:
#   - `build_initial_mixing_ratio` bit-exact behaviour post-hoist on LL
#     (uniform + gaussian_blob) and RG (uniform + gaussian_blob) — uses
#     the same algebra as the pre-hoist LL/RG runner path, so equivalence is
#     proven by "same output for the same inputs".
#   - `pack_initial_tracer_mass` basis-aware packing per
#     feedback_vmr_to_mass_basis_aware:
#       * DryBasis   → rm = vmr .* air_mass
#       * MoistBasis → rm = vmr .* air_mass .* (1 .- qv)  (+ error without qv)
#
# Also exercises the file-based IC path with a synthetic NetCDF fixture; the
# real Catrine equivalence check is skipped when external data are absent.
# CS dispatch is added in plan 40 Commit 1c.
# ---------------------------------------------------------------------------

using Test
using Dates
using TOML
using Printf: @sprintf
import NCDatasets: NCDataset, defDim, defVar, nomissing

using AtmosTransport
using .AtmosTransport.Grids: cell_index, nrings, ring_longitudes
using .AtmosTransport.Models.InitialConditionIO: build_surface_flux_source,
                                                build_surface_flux_sources

const FT = Float64
const ICIO = AtmosTransport.Models.InitialConditionIO
const _SYNTHETIC_SECONDS_JAN_2021 = 31 * 86400

_synthetic_file_ic_value(lon_deg, lat_deg) =
    3.5e-4 + 1e-6 * mod(lon_deg, 360.0) + 2e-6 * lat_deg

function _write_synthetic_ic_file(path::AbstractString; with_humidity::Bool = false,
                                  humidity_name::AbstractString = "QV",
                                  humidity_standard_name::Bool = true)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "longitude", 4)
        defDim(ds, "latitude", 2)
        defDim(ds, "level", 3)
        defDim(ds, "hlevel", 4)

        vlon = defVar(ds, "longitude", Float64, ("longitude",))
        vlat = defVar(ds, "latitude", Float64, ("latitude",))
        vlev = defVar(ds, "level", Float64, ("level",))
        vhlev = defVar(ds, "hlevel", Float64, ("hlevel",))
        vap = defVar(ds, "ap", Float64, ("hlevel",), attrib = Dict("units" => "Pa"))
        vbp = defVar(ds, "bp", Float64, ("hlevel",))
        vco2 = defVar(ds, "CO2", Float64, ("longitude", "latitude", "level"))
        vqv = with_humidity ?
              defVar(ds, humidity_name, Float64,
                     ("longitude", "latitude", "level")) : nothing

        lon = [-135.0, -45.0, 45.0, 135.0]
        lat = [45.0, -45.0]  # descending on purpose
        vlon[:] = lon
        vlat[:] = lat
        vlev[:] = [0.0, 1.0, 2.0]
        vhlev[:] = [0.0, 1.0, 2.0, 3.0]
        vap[:] = [0.0, 0.0, 0.0, 0.0]
        vbp[:] = [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]
        raw = Array{Float64}(undef, 4, 2, 3)
        for k in 1:3, j in 1:2, i in 1:4
            raw[i, j, k] = _synthetic_file_ic_value(lon[i], lat[j])
        end
        vco2[:, :, :] = raw
        vco2.attrib["units"] = "mol mol-1"
        if vqv !== nothing
            for k in 1:3
                vqv[:, :, k] .= (0.35, 0.08, 0.001)[k]
            end
            humidity_standard_name &&
                (vqv.attrib["standard_name"] = "specific_humidity")
            vqv.attrib["units"] = "kg kg-1"
        end
    finally
        close(ds)
    end
    return nothing
end

function _write_synthetic_cs_native_ic_file(path::AbstractString; Nc=2, Nz=2)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "Xdim", Nc)
        defDim(ds, "Ydim", Nc)
        defDim(ds, "nf", 6)
        defDim(ds, "lev", Nz)
        defDim(ds, "ilev", Nz + 1)
        tracer = defVar(ds, "SIGNED", Float64,
                        ("Xdim", "Ydim", "nf", "lev"))
        # Equal level counts must retain the exact-copy path even when the
        # native file carries otherwise usable hybrid metadata.
        defVar(ds, "ap", Float64, ("ilev",), attrib = Dict("units" => "Pa"))[:] = zeros(Nz + 1)
        defVar(ds, "bp", Float64, ("ilev",))[:] = collect(range(1.0, 0.0; length = Nz + 1))
        values = Array{Float64}(undef, Nc, Nc, 6, Nz)
        for k in 1:Nz, p in 1:6, j in 1:Nc, i in 1:Nc
            values[i, j, p, k] = (isodd(i + j + p + k) ? -1.0 : 1.0) *
                                  (1e-6 + 1e-8 * (i + 2j + 3p + 4k))
        end
        tracer[:, :, :, :] = values
    finally
        close(ds)
    end
    return nothing
end

# A cs_native IC whose level k (file order) holds `level_values[k]` everywhere,
# with optional hybrid coefficients: `ap`/`bp` = `nothing` omits the variable,
# `ap_units` = `nothing` omits its units attribute.
function _write_hybrid_cs_native_ic_file(path::AbstractString; Nc, level_values,
                                         ap, bp, ap_units = "Pa")
    Nsrc = length(level_values)
    NCDataset(path, "c") do ds
        for (name, n) in (("Xdim", Nc), ("Ydim", Nc), ("nf", 6),
                          ("lev", Nsrc), ("ilev", Nsrc + 1))
            defDim(ds, name, n)
        end
        tracer = defVar(ds, "TRACER", Float64, ("Xdim", "Ydim", "nf", "lev"))
        if ap !== nothing
            attrib = ap_units === nothing ? Dict{String, Any}() :
                                            Dict{String, Any}("units" => ap_units)
            defVar(ds, "ap", Float64, ("ilev",); attrib)[:] = ap
        end
        bp === nothing || (defVar(ds, "bp", Float64, ("ilev",))[:] = bp)
        for k in 1:Nsrc
            tracer[:, :, :, k] .= level_values[k]
        end
    end
    return path
end

function _write_synthetic_surface_flux_file(path::AbstractString)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "longitude", 4)
        defDim(ds, "latitude", 2)
        defDim(ds, "time", 2)

        vlon = defVar(ds, "longitude", Float64, ("longitude",))
        vlat = defVar(ds, "latitude", Float64, ("latitude",))
        vtime = defVar(ds, "time", Float64, ("time",),
                       attrib = Dict("units" => "months since 2021-01-01"))
        vtotal = defVar(ds, "TOTAL", Float64, ("longitude", "latitude", "time"))
        varea = defVar(ds, "cell_area", Float64, ("longitude", "latitude"))

        vlon[:] = [-135.0, -45.0, 45.0, 135.0]
        vlat[:] = [45.0, -45.0]  # descending on purpose
        vtime[:] = [0.0, 1.0]
        # Every 90° × 90° cell has the area R² (π/2)(sin 90° − sin 0°) [m²] (the
        # loader checks a file's cell areas against its coordinates); one cell
        # emits 7 kg CO₂ s⁻¹ in January.
        area = AtmosTransport.Parameters.EARTH_RADIUS^2 * π / 2
        raw1 = zeros(Float64, 4, 2)
        raw1[1, 1] = 7 * _SYNTHETIC_SECONDS_JAN_2021 / area
        raw2 = 2 .* raw1
        vtotal[:, :, 1] = raw1
        vtotal[:, :, 2] = raw2
        varea[:, :] = fill(area, 4, 2)
        varea.attrib["units"] = "m2"
        vtotal.attrib["units"] = "kgCO2/month/m2"
    finally
        close(ds)
    end
    return nothing
end

function _write_mismatched_cs_native_ic_file(path::AbstractString; Nc=2)
    Nsrc = 3
    NCDataset(path, "c") do ds
        for (name, n) in (("Xdim", Nc), ("Ydim", Nc), ("nf", 6),
                          ("lev", Nsrc), ("ilev", Nsrc + 1))
            defDim(ds, name, n)
        end
        raw = defVar(ds, "TRACER", Float64, ("Xdim", "Ydim", "nf", "lev"))
        humidity = defVar(ds, "QV", Float64, ("Xdim", "Ydim", "nf", "lev"))
        defVar(ds, "ap", Float64, ("ilev",), attrib = Dict("units" => "Pa"))[:] = zeros(Nsrc + 1)
        defVar(ds, "bp", Float64, ("ilev",))[:] = [1.0, 2 / 3, 1 / 3, 0.0]
        for k in 1:Nsrc
            raw[:, :, :, k] .= (6.0, 3.0, 0.0)[k]
            humidity[:, :, :, k] .= (0.0, 0.5, 0.0)[k]
        end
    end
    return nothing
end

function _write_synthetic_lmdz_flux_file(path::AbstractString, year::Int, month::Int)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "longitude", 4)
        defDim(ds, "latitude", 2)
        defDim(ds, "time", 2)
        lon = defVar(ds, "longitude", Float64, ("longitude",))
        lat = defVar(ds, "latitude", Float64, ("latitude",))
        time = defVar(ds, "time", Float64, ("time",))
        flux = defVar(ds, "flux_apos", Float32,
                      ("longitude", "latitude", "time"))
        lon[:] = [-135.0, -45.0, 45.0, 135.0]
        lat[:] = [45.0, -45.0]
        time[:] = [0.0, 3.0]
        time.attrib["units"] = @sprintf("hours since %04d-%02d-01 00:00:00", year, month)
        flux[:, :, 1] .= Float32(month)
        flux[:, :, 2] .= Float32(month + 0.5)
        flux.attrib["units"] = "kgC m-2 s-1"
    finally
        close(ds)
    end
end

function _write_synthetic_monthly_gridfed_file(path::AbstractString, year::Int)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "longitude", 4)
        defDim(ds, "latitude", 2)
        defDim(ds, "time", 12)  # GridFED has a time dimension but no coordinate variable.
        lon = defVar(ds, "longitude", Float64, ("longitude",))
        lat = defVar(ds, "latitude", Float64, ("latitude",))
        flux = defVar(ds, "TOTAL", Float64,
                      ("longitude", "latitude", "time"))
        lon[:] = [-135.0, -45.0, 45.0, 135.0]
        lat[:] = [45.0, -45.0]
        for month in 1:12
            month_seconds = Dates.daysinmonth(year, month) * 86400.0
            flux[:, :, month] .= month * month_seconds
        end
        flux.attrib["units"] = "kgCO2/month/m2"
    finally
        close(ds)
    end
end

@testset "plan 40 Commit 1b — InitialConditionIO hoist" begin

    # ---------------------------- LL uniform + blob ------------------------
    @testset "build_initial_mixing_ratio — LatLonMesh" begin
        mesh = LatLonMesh(; Nx = 4, Ny = 3,
                          longitude = (0.0, 360.0),
                          latitude  = (-90.0, 90.0))
        Nz = 2
        air_mass = ones(FT, 4, 3, Nz)   # shape-only; builder does not read values

        # uniform
        q = build_initial_mixing_ratio(air_mass, mesh,
                                       Dict("kind" => "uniform",
                                            "background" => 1.5e-4))
        @test size(q) == size(air_mass)
        @test all(q .== FT(1.5e-4))

        q_step = build_initial_mixing_ratio(air_mass, mesh,
                                            Dict("kind" => "latitude_step",
                                                 "south_value" => 4.0e-4,
                                                 "north_value" => 4.4e-4,
                                                 "split_lat_deg" => 0.0))
        @test size(q_step) == size(air_mass)
        @test all(q_step[:, 1, :] .== FT(4.0e-4))
        @test all(q_step[:, 2:3, :] .== FT(4.4e-4))

        # gaussian_blob — non-uniform profile centered at (0, 0)
        q2 = build_initial_mixing_ratio(air_mass, mesh,
                                        Dict("kind" => "gaussian_blob",
                                             "lon0_deg" => 0.0, "lat0_deg" => 0.0,
                                             "sigma_lon_deg" => 10.0,
                                             "sigma_lat_deg" => 10.0,
                                             "amplitude" => 1.0e-3,
                                             "background" => 4.0e-4))
        @test size(q2) == size(air_mass)
        @test minimum(q2) ≥ FT(4.0e-4)
        @test maximum(q2) > minimum(q2)   # non-trivial profile
    end

    # ---------------------------- RG uniform + blob ------------------------
    @testset "build_initial_mixing_ratio — ReducedGaussianMesh" begin
        latitudes = [-75.0, -25.0, 25.0, 75.0]
        nlon_per_ring = [4, 8, 8, 4]
        mesh = ReducedGaussianMesh(latitudes, nlon_per_ring; FT = FT)
        Nz = 2
        ncells_ = ncells(mesh)
        air_mass = ones(FT, ncells_, Nz)

        q = build_initial_mixing_ratio(air_mass, mesh,
                                       Dict("kind" => "uniform",
                                            "background" => 2.0e-4))
        @test size(q) == (ncells_, Nz)
        @test all(q .== FT(2.0e-4))

        q_step = build_initial_mixing_ratio(air_mass, mesh,
                                            Dict("kind" => "hemisphere_step",
                                                 "south_value" => 4.0e-4,
                                                 "north_value" => 4.4e-4))
        @test all(q_step[1:12, :] .== FT(4.0e-4))
        @test all(q_step[13:end, :] .== FT(4.4e-4))

        q2 = build_initial_mixing_ratio(air_mass, mesh,
                                        Dict("kind" => "gaussian_blob",
                                             "lon0_deg" => 0.0, "lat0_deg" => 0.0,
                                             "sigma_lon_deg" => 15.0,
                                             "sigma_lat_deg" => 15.0,
                                             "amplitude" => 5.0e-4,
                                             "background" => 4.0e-4))
        @test minimum(q2) ≥ FT(4.0e-4)
        @test maximum(q2) > minimum(q2)
    end

    # ---------------------------- LL grid-level dispatch -------------------
    @testset "build_initial_mixing_ratio — AtmosGrid{LatLonMesh} forwards" begin
        # A non-file kind should forward to the mesh method and be bit-exact.
        mesh = LatLonMesh(; Nx = 4, Ny = 3,
                          longitude = (0.0, 360.0),
                          latitude  = (-90.0, 90.0))
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        Nz = 2
        air_mass = ones(FT, 4, 3, Nz)
        cfg = Dict("kind" => "uniform", "background" => 3.14e-4)
        q_mesh = build_initial_mixing_ratio(air_mass, mesh, cfg)
        q_grid = build_initial_mixing_ratio(air_mass, grid, cfg)
        @test q_mesh == q_grid   # bit-exact
    end

    @testset "file-based initial conditions support latlon and reduced grids" begin
        mktempdir() do dir
            ic_path = joinpath(dir, "test_ic.nc")
            _write_synthetic_ic_file(ic_path)
            humid_ic_path = joinpath(dir, "test_ic_humid.nc")
            _write_synthetic_ic_file(humid_ic_path; with_humidity = true)

            init_cfg = Dict{String, Any}(
                "kind" => "file",
                "file" => ic_path,
                "variable" => "CO2",
            )
            humid_cfg = merge(init_cfg, Dict("file" => humid_ic_path,
                                              "humidity_variable" => "QV"))
            humid_source = ICIO._load_file_initial_condition_source(humid_cfg, Float64, 2)
            @test humid_source.qv !== nothing
            @test humid_source.qv[1, 1, :] == [0.35, 0.08, 0.001]

            vertical = HybridSigmaPressure(Float64[0, 0, 0], Float64[0, 0.5, 1])

            latlon_mesh = LatLonMesh(; FT = Float64, Nx = 4, Ny = 2)
            latlon_grid = AtmosGrid(latlon_mesh, vertical, CPU(); FT = Float64)
            air_mass_ll = ones(Float64, 4, 2, 2)
            ps_ll = fill(9.0e4, 4, 2)
            q_ll = build_initial_mixing_ratio(air_mass_ll, latlon_grid, init_cfg;
                                              surface_pressure = ps_ll)
            for j in 1:2, i in 1:4
                expected = _synthetic_file_ic_value(latlon_mesh.λᶜ[i], latlon_mesh.φᶜ[j])
                @test q_ll[i, j, 1] ≈ expected atol = 1e-12
                @test q_ll[i, j, 2] ≈ expected atol = 1e-12
            end

            reduced_mesh = ReducedGaussianMesh([-45.0, 45.0], [4, 4]; FT = Float64)
            reduced_grid = AtmosGrid(reduced_mesh, vertical, CPU(); FT = Float64)
            air_mass_rg = ones(Float64, ncells(reduced_mesh), 2)
            ps_rg = fill(9.0e4, ncells(reduced_mesh))
            q_rg = build_initial_mixing_ratio(air_mass_rg, reduced_grid, init_cfg;
                                              surface_pressure = ps_rg)
            for j in 1:nrings(reduced_mesh)
                lons = ring_longitudes(reduced_mesh, j)
                lat = reduced_mesh.latitudes[j]
                for i in eachindex(lons)
                    c = cell_index(reduced_mesh, i, j)
                    expected = _synthetic_file_ic_value(lons[i], lat)
                    @test q_rg[c, 1] ≈ expected atol = 1e-12
                    @test q_rg[c, 2] ≈ expected atol = 1e-12
                end
            end
        end
    end

    # ---------------------------- pack_initial_tracer_mass ------------------
    @testset "pack_initial_tracer_mass — DryBasis (LL)" begin
        mesh = LatLonMesh(; Nx = 4, Ny = 3,
                          longitude = (0.0, 360.0),
                          latitude  = (-90.0, 90.0))
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        air_mass = fill(FT(1.2e10), 4, 3, 2)
        vmr      = fill(FT(4.11e-4), 4, 3, 2)
        rm = pack_initial_tracer_mass(grid, air_mass, vmr; mass_basis = DryBasis())
        @test rm == vmr .* air_mass
        # qv is ignored on DryBasis
        qv_ignored = fill(FT(0.5), 4, 3, 2)   # absurd humidity — would break MoistBasis
        rm2 = pack_initial_tracer_mass(grid, air_mass, vmr;
                                       mass_basis = DryBasis(), qv = qv_ignored)
        @test rm2 == rm
    end

    @testset "pack_initial_tracer_mass — MoistBasis (LL)" begin
        mesh = LatLonMesh(; Nx = 4, Ny = 3,
                          longitude = (0.0, 360.0),
                          latitude  = (-90.0, 90.0))
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        air_mass = fill(FT(1.2e10), 4, 3, 2)
        vmr      = fill(FT(4.11e-4), 4, 3, 2)
        qv       = fill(FT(0.02), 4, 3, 2)   # 2 % humidity
        rm = pack_initial_tracer_mass(grid, air_mass, vmr;
                                      mass_basis = MoistBasis(), qv = qv)
        @test rm == vmr .* air_mass .* (1 .- qv)
        # Missing qv errors loudly
        @test_throws ArgumentError pack_initial_tracer_mass(grid, air_mass, vmr;
                                                            mass_basis = MoistBasis())
        # Shape mismatch errors
        @test_throws DimensionMismatch pack_initial_tracer_mass(grid, air_mass, vmr;
                                                                mass_basis = MoistBasis(),
                                                                qv = fill(FT(0.02), 3, 3, 2))
    end

    @testset "pack_initial_tracer_mass — RG both bases" begin
        latitudes = [-75.0, -25.0, 25.0, 75.0]
        nlon_per_ring = [4, 8, 8, 4]
        mesh = ReducedGaussianMesh(latitudes, nlon_per_ring; FT = FT)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        ncells_ = ncells(mesh)
        air_mass = fill(FT(1.2e10), ncells_, 2)
        vmr      = fill(FT(4.11e-4), ncells_, 2)

        rm_dry = pack_initial_tracer_mass(grid, air_mass, vmr;
                                          mass_basis = DryBasis())
        @test rm_dry == vmr .* air_mass

        qv = fill(FT(0.01), ncells_, 2)
        rm_moist = pack_initial_tracer_mass(grid, air_mass, vmr;
                                            mass_basis = MoistBasis(), qv = qv)
        @test rm_moist == vmr .* air_mass .* (1 .- qv)
    end

    # --------- plan 40 Commit 1c: CubedSphere IC + packer ---------------
    @testset "build_initial_mixing_ratio — CubedSphereMesh uniform" begin
        Nc = 4
        Hp = 1
        Nz = 3
        mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        # Halo-padded 6-panel air_mass tuple (matches TransportBinaryDriver layout).
        air_mass = ntuple(_ -> fill(FT(1e10), Nc + 2 * Hp, Nc + 2 * Hp, Nz), 6)

        vmr = build_initial_mixing_ratio(air_mass, grid,
                                         Dict("kind" => "uniform",
                                              "background" => 4.11e-4))
        @test vmr isa NTuple{6, Array{FT, 3}}
        @test all(size(vmr[p]) == (Nc, Nc, Nz) for p in 1:6)   # interior only
        @test all(all(vmr[p] .== FT(4.11e-4)) for p in 1:6)

        vmr_step = build_initial_mixing_ratio(air_mass, grid,
                                              Dict("kind" => "latitude_step",
                                                   "south_value" => 4.0e-4,
                                                   "north_value" => 4.4e-4))
        @test vmr_step isa NTuple{6, Array{FT, 3}}
        @test all(size(vmr_step[p]) == (Nc, Nc, Nz) for p in 1:6)
        step_vals = vcat((vec(vmr_step[p]) for p in 1:6)...)
        @test all(v -> v == FT(4.0e-4) || v == FT(4.4e-4), step_vals)
        @test minimum(step_vals) == FT(4.0e-4)
        @test maximum(step_vals) == FT(4.4e-4)

        vmr_blob = build_initial_mixing_ratio(air_mass, grid,
                                              Dict("kind" => "gaussian_blob",
                                                   "lon0_deg" => 0.0,
                                                   "lat0_deg" => 35.0,
                                                   "sigma_lon_deg" => 30.0,
                                                   "sigma_lat_deg" => 20.0,
                                                   "amplitude" => 8.0e-5,
                                                   "background" => 4.0e-4))
        @test vmr_blob isa NTuple{6, Array{FT, 3}}
        @test all(size(vmr_blob[p]) == (Nc, Nc, Nz) for p in 1:6)
        blob_vals = vcat((vec(vmr_blob[p]) for p in 1:6)...)
        @test minimum(blob_vals) ≥ FT(4.0e-4)
        @test maximum(blob_vals) > FT(4.0e-4)
        @test maximum(blob_vals) < FT(4.8e-4)

        # Unsupported kind errors with a helpful message
        @test_throws ArgumentError build_initial_mixing_ratio(air_mass, grid,
                                                              Dict("kind" => "bl_enhanced"))
    end

    @testset "cs_native preserves signed values by default" begin
        mktempdir() do dir
            Nc, Hp, Nz = 2, 1, 2
            path = joinpath(dir, "signed_cs_native.nc")
            _write_synthetic_cs_native_ic_file(path; Nc, Nz)

            mesh = CubedSphereMesh(; FT, Nc, Hp)
            vertical = HybridSigmaPressure(zeros(FT, Nz + 1),
                                           collect(FT, range(1.0, 0.0; length = Nz + 1)))
            grid = AtmosGrid(mesh, vertical, CPU(); FT)
            air_mass = ntuple(_ -> fill(FT(1e10), Nc + 2Hp, Nc + 2Hp, Nz), 6)
            base_cfg = Dict{String, Any}(
                "kind" => "cs_native",
                "file" => path,
                "variable" => "SIGNED",
                "vertical_order" => "surface_first",
            )

            signed = build_initial_mixing_ratio(air_mass, grid, base_cfg)
            @test minimum(vcat((vec(signed[p]) for p in 1:6)...)) < 0
            @test maximum(vcat((vec(signed[p]) for p in 1:6)...)) > 0
            # Equal level counts retain the pre-E7 cell-for-cell copy exactly.
            for k in 1:Nz, p in 1:6, j in 1:Nc, i in 1:Nc
                expected = (isodd(i + j + p + k) ? -1.0 : 1.0) *
                           (1e-6 + 1e-8 * (i + 2j + 3p + 4k))
                @test signed[p][i, j, k] === FT(expected)
            end

            clamped_cfg = copy(base_cfg)
            clamped_cfg["clamp_negative"] = true
            clamped = build_initial_mixing_ratio(air_mass, grid, clamped_cfg)
            @test minimum(vcat((vec(clamped[p]) for p in 1:6)...)) == 0
            @test maximum(vcat((vec(clamped[p]) for p in 1:6)...)) > 0
        end
    end

    @testset "cs_native hybrid coefficients: orientation, agreement and units" begin
        mktempdir() do dir
            Nc, Hp, Nz = 2, 1, 3
            mesh = CubedSphereMesh(; FT, Nc, Hp)
            # A normal top-first target: interface 1 is the model top.
            A_top = [100.0, 3000.0, 6000.0, 0.0]
            B_top = [0.0, 0.2, 0.6, 1.0]
            grid = AtmosGrid(mesh, HybridSigmaPressure(FT.(A_top), FT.(B_top)), CPU(); FT)
            air_mass = ntuple(_ -> fill(FT(1e10), Nc + 2Hp, Nc + 2Hp, Nz), 6)
            # Source levels surface-first: level 1 is the surface layer.
            surface_first_values = [1e-6, 2e-6, 3e-6]
            cfg_for(path) = Dict{String, Any}("kind" => "cs_native", "file" => path,
                                              "variable" => "TRACER")
            build(path) = build_initial_mixing_ratio(air_mass, grid, cfg_for(path))
            message(f) = try
                f()
                ""
            catch err
                err isa ArgumentError || err isa DimensionMismatch || rethrow()
                sprint(showerror, err)
            end

            # Opposite orientation, equal level counts: the source column is
            # reversed onto the target, so the surface value lands at k = Nz.
            # ap of 6000 Pa without a units attribute is unambiguously Pa.
            reversed = _write_hybrid_cs_native_ic_file(joinpath(dir, "surface_first.nc");
                Nc, level_values = surface_first_values,
                ap = reverse(A_top), bp = reverse(B_top), ap_units = nothing)
            vmr = build(reversed)
            for p in 1:6, k in 1:Nz
                @test all(vmr[p][:, :, k] .== FT(surface_first_values[Nz - k + 1]))
            end
            # A consistent vertical_order is accepted; a contradicting one is not.
            consistent = merge(cfg_for(reversed), Dict{String, Any}("vertical_order" => "surface_first"))
            @test build_initial_mixing_ratio(air_mass, grid, consistent) == vmr
            contradicting = merge(cfg_for(reversed), Dict{String, Any}("vertical_order" => "toa_first"))
            @test occursin("contradicts the hybrid coefficients",
                           message(() -> build_initial_mixing_ratio(air_mass, grid, contradicting)))

            # Same orientation as the target: copied level for level.
            same = _write_hybrid_cs_native_ic_file(joinpath(dir, "top_first.nc");
                Nc, level_values = reverse(surface_first_values), ap = A_top, bp = B_top)
            @test build(same) == vmr

            # Equal level counts with other coefficients are another vertical
            # grid: rejected, never copied cell for cell.
            other_bp = _write_hybrid_cs_native_ic_file(joinpath(dir, "other_bp.nc");
                Nc, level_values = surface_first_values,
                ap = reverse(A_top), bp = [1.0, 0.5, 0.2, 0.0])
            @test occursin("bp coefficients differ", message(() -> build(other_bp)))
            other_ap = _write_hybrid_cs_native_ic_file(joinpath(dir, "other_ap.nc");
                Nc, level_values = surface_first_values,
                ap = [0.0, 7000.0, 3000.0, 100.0], bp = reverse(B_top))
            @test occursin("ap coefficients differ", message(() -> build(other_ap)))

            # Only one of ap and bp is an error, whatever the level counts.
            only_ap = _write_hybrid_cs_native_ic_file(joinpath(dir, "only_ap.nc");
                Nc, level_values = surface_first_values, ap = reverse(A_top), bp = nothing)
            only_bp = _write_hybrid_cs_native_ic_file(joinpath(dir, "only_bp.nc");
                Nc, level_values = surface_first_values, ap = nothing, bp = reverse(B_top))
            for path in (only_ap, only_bp)
                @test occursin("must provide both ap and bp, or neither", message(() -> build(path)))
            end

            # ap in hPa, or unit-less with hPa-sized values, is rejected, not guessed.
            hpa = _write_hybrid_cs_native_ic_file(joinpath(dir, "hpa.nc");
                Nc, level_values = surface_first_values,
                ap = reverse(A_top) ./ 100, bp = reverse(B_top), ap_units = "hPa")
            @test occursin("must use Pa; got units = \"hPa\"", message(() -> build(hpa)))
            ambiguous = _write_hybrid_cs_native_ic_file(joinpath(dir, "ambiguous.nc");
                Nc, level_values = surface_first_values,
                ap = reverse(A_top) ./ 100, bp = reverse(B_top), ap_units = nothing)
            @test occursin("ambiguous with hPa", message(() -> build(ambiguous)))
            # Explicit Pa spellings and a unit-less pure-sigma (all-zero) ap pass.
            pascal = _write_hybrid_cs_native_ic_file(joinpath(dir, "pascal.nc");
                Nc, level_values = surface_first_values,
                ap = reverse(A_top), bp = reverse(B_top), ap_units = "Pascal")
            @test build(pascal) == vmr
            sigma_grid = AtmosGrid(mesh, HybridSigmaPressure(zeros(FT, Nz + 1), FT.(B_top)), CPU(); FT)
            sigma = _write_hybrid_cs_native_ic_file(joinpath(dir, "sigma.nc");
                Nc, level_values = surface_first_values,
                ap = zeros(Nz + 1), bp = reverse(B_top), ap_units = nothing)
            sigma_vmr = build_initial_mixing_ratio(air_mass, sigma_grid, cfg_for(sigma))
            @test sigma_vmr == vmr
        end
    end

    @testset "cs_native conservatively remaps mismatched hybrid levels" begin
        mktempdir() do dir
            Nc, Hp, Nz = 2, 1, 2
            path = joinpath(dir, "mismatched_cs_native.nc")
            _write_mismatched_cs_native_ic_file(path; Nc)
            mesh = CubedSphereMesh(; FT, Nc, Hp)
            vertical = HybridSigmaPressure(FT[0, 0, 0], FT[0, 0.5, 1])
            grid = AtmosGrid(mesh, vertical, CPU(); FT)
            air_mass = ntuple(_ -> fill(FT(1e10), Nc + 2Hp, Nc + 2Hp, Nz), 6)
            cfg = Dict{String, Any}(
                "kind" => "cs_native", "file" => path, "variable" => "TRACER",
                "humidity_variable" => "QV")

            ps_total = ntuple(_ -> fill(FT(90000), Nc, Nc), 6)
            total = build_initial_mixing_ratio(
                air_mass, grid, cfg; surface_pressure = ps_total,
                pressure_basis = TotalSurfacePressure())
            for p in 1:6
                @test total[p][:, :, 1] == fill(FT(0.6), Nc, Nc)
                @test total[p][:, :, 2] == fill(FT(5.4), Nc, Nc)
                # Dry-carrier pressure integral is conserved in every column.
                @test sum(Float64.(total[p][1, 1, :])) * 37500 == 225000
            end

            # The typed dry-pressure path recovers the same 90 kPa total ps
            # from target qv before placing source and target interfaces.
            target_qv = ntuple(_ -> fill(FT(0.1), Nc, Nc, Nz), 6)
            ps_dry = ntuple(_ -> fill(FT(81000), Nc, Nc), 6)
            dry = build_initial_mixing_ratio(
                air_mass, grid, cfg; surface_pressure = ps_dry,
                pressure_basis = DrySurfacePressure(), qv = target_qv)
            @test dry == total

            @test_throws ArgumentError build_initial_mixing_ratio(air_mass, grid, cfg)
        end
    end

    @testset "pack_initial_tracer_mass — CubedSphereMesh (DryBasis)" begin
        Nc = 4
        Hp = 1
        Nz = 3
        mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        air_mass = ntuple(_ -> fill(FT(1.2e10), Nc + 2 * Hp, Nc + 2 * Hp, Nz), 6)
        vmr_interior = ntuple(_ -> fill(FT(4.11e-4), Nc, Nc, Nz), 6)

        rm = pack_initial_tracer_mass(grid, air_mass, vmr_interior;
                                      mass_basis = DryBasis())
        @test rm isa NTuple{6, Array{FT, 3}}
        for p in 1:6
            @test size(rm[p]) == (Nc + 2 * Hp, Nc + 2 * Hp, Nz)
            interior = @view rm[p][Hp + 1 : Hp + Nc, Hp + 1 : Hp + Nc, :]
            expected = vmr_interior[p] .* FT(1.2e10)   # air_mass × vmr (interior)
            @test interior == expected
            # Halo ring is zero — halo exchanges at runtime populate it.
            @test rm[p][1, 1, 1] == zero(FT)
            @test rm[p][end, end, end] == zero(FT)
        end
    end

    @testset "surface-anchored conservative hybrid remap ($T)" for T in (Float32, Float64)
        function remap!(dest, src_q, src_qv, ap, bp, A_tgt, B_tgt, ps,
                        target_dp_dry, src_p = zeros(Float64, length(src_q) + 1),
                        cumulative = similar(src_p), dry_cumulative = similar(src_p))
            ICIO._remap_hybrid_profile!(dest, src_q, src_qv, ap, bp, A_tgt,
                                        B_tgt, ps, src_p, cumulative, dry_cumulative)
        end

        # Different source and target partitions of the same pressure column,
        # including a strongly humid source bottom layer. The tracer integral
        # is conserved on dry pressure, not total pressure.
        src_q = T[9.0, 2.0, 5.0]                         # surface -> TOA
        src_qv = T[0.35, 0.08, 0.001]                    # strong humidity
        ap = zeros(T, 4)
        bp = T[1.0, 0.7, 0.2, 0.0]                      # surface -> TOA
        A_tgt = zeros(T, 5)
        B_tgt = T[0.0, 0.1, 0.4, 0.8, 1.0]              # TOA -> surface
        ps = T(100_000)
        src_interfaces = Float64.(ap) .+ Float64.(bp) .* Float64(ps)
        tgt_p = Float64.(A_tgt) .+ Float64.(B_tgt) .* Float64(ps)
        target_dp_dry = zeros(Float64, 4)
        for k in eachindex(target_dp_dry), s in eachindex(src_qv)
            overlap = max(0.0, min(tgt_p[k + 1], src_interfaces[s]) -
                               max(tgt_p[k], src_interfaces[s + 1]))
            target_dp_dry[k] += (1 - Float64(src_qv[s])) * overlap
        end
        dest = zeros(T, 4)
        scratch_p = zeros(Float64, 4)
        scratch_cumulative = similar(scratch_p)
        scratch_dry_cumulative = similar(scratch_p)
        remap!(dest, src_q, src_qv, ap, bp, A_tgt, B_tgt, ps, target_dp_dry,
               scratch_p, scratch_cumulative, scratch_dry_cumulative)
        source_integral = sum(Float64.(src_q) .* (1 .- Float64.(src_qv)) .*
                              (-diff(src_interfaces)))
        target_integral = sum(Float64.(dest) .* target_dp_dry)
        @test target_integral ≈ source_integral rtol = 64eps(T)
        @test !isapprox(sum(Float64.(dest) .* diff(tgt_p)),
                        sum(Float64.(src_q) .* (-diff(src_interfaces))); rtol = 1e-3)
        @test @allocated(remap!(dest, src_q, src_qv, ap, bp, A_tgt, B_tgt,
                                ps, target_dp_dry, scratch_p,
                                scratch_cumulative, scratch_dry_cumulative)) == 0

        uniform = similar(dest)
        remap!(uniform, fill(T(400e-6), length(src_q)), src_qv, ap, bp,
               A_tgt, B_tgt, ps, fill(1.0, length(dest)), scratch_p,
               scratch_cumulative, scratch_dry_cumulative)
        @test uniform ≈ fill(T(400e-6), length(uniform)) rtol = 8eps(T)

        # Without source-file humidity the target dry factor cancels exactly,
        # even for a target layer with 40% humidity.
        target_qv = T[0.0, 0.4, 0.12, 0.02]
        binary_dp_dry = (1 .- Float64.(target_qv)) .* diff(tgt_p)
        fallback = zeros(T, 4)
        remap!(fallback, src_q, nothing, ap, bp, A_tgt, B_tgt, ps,
               binary_dp_dry, scratch_p, scratch_cumulative, scratch_dry_cumulative)
        expected = zeros(Float64, 4)
        for k in eachindex(expected), s in eachindex(src_q)
            overlap = max(0.0, min(tgt_p[k + 1], src_interfaces[s]) -
                               max(tgt_p[k], src_interfaces[s + 1]))
            expected[k] += Float64(src_q[s]) * overlap / (tgt_p[k + 1] - tgt_p[k])
        end
        @test Float64.(fallback) ≈ expected rtol = 32eps(T)

        # Caller-owned scratch makes the production kernel allocation-free.
        remap!(fallback, src_q, nothing, ap, bp, A_tgt, B_tgt, ps,
               binary_dp_dry, scratch_p, scratch_cumulative)
        @test @allocated(remap!(fallback, src_q, nothing, ap, bp, A_tgt, B_tgt,
                                ps, binary_dp_dry, scratch_p,
                                scratch_cumulative, scratch_dry_cumulative)) == 0

        # A near-surface peak stays on the same sigma layers when every column
        # receives a uniform surface-pressure shift.
        peaked_q = T[50.0, 7.0, 3.0, 1.0]
        peaked_bp = T[1.0, 0.95, 0.8, 0.4, 0.0]
        shifted_1 = zeros(T, 5)
        shifted_2 = similar(shifted_1)
        target_bp = T[0.0, 0.1, 0.3, 0.6, 0.85, 1.0]
        remap!(shifted_1, peaked_q, nothing, zeros(T, 5), peaked_bp,
               zeros(T, 6), target_bp, T(98_325),
               diff(Float64.(target_bp) .* 98_325))
        remap!(shifted_2, peaked_q, nothing, zeros(T, 5), peaked_bp,
               zeros(T, 6), target_bp, T(98_725),
               diff(Float64.(target_bp) .* 98_725))
        @test shifted_1 ≈ shifted_2 rtol = 32eps(T)

        # A pure pressure-level source keeps its absolute levels. Its surface
        # end layer extends when the source bottom is above the target surface,
        # and is truncated when it is below the target surface.
        pressure_q = T[10.0, 2.0]
        pressure_ap = T[800.0, 600.0, 300.0]
        pressure_bp = zeros(T, 3)
        target_A = T[0.0, 300.0, 0.0]
        target_B = T[0.0, 0.0, 1.0]
        shallower = zeros(T, 2)
        deeper = similar(shallower)
        remap!(shallower, pressure_q, nothing, pressure_ap, pressure_bp,
               target_A, target_B, T(700),
               diff(Float64.(target_A) .+ Float64.(target_B) .* 700))
        remap!(deeper, pressure_q, nothing, pressure_ap, pressure_bp,
               target_A, target_B, T(1_000),
               diff(Float64.(target_A) .+ Float64.(target_B) .* 1_000))
        @test shallower ≈ T[2.0, 4.0] rtol = 16eps(T)
        @test deeper ≈ T[2.0, 4600 / 700] rtol = 16eps(T)

        @test_throws DimensionMismatch remap!(zeros(T, 3), src_q, nothing,
                                              ap[1:3], bp[1:3], A_tgt,
                                              B_tgt, ps, ones(3))
        @test :psurf ∉ fieldnames(ICIO.FileInitialConditionSource{T})
        @test :qv ∈ fieldnames(ICIO.FileInitialConditionSource{T})
    end

    @testset "LL, RG, and CS binary surface-pressure basis traits ($T)" for T in (Float32, Float64)
        vertical = HybridSigmaPressure(zeros(T, 3), T[0.0, 0.5, 1.0])
        ps_total = T(100_000)
        dp_dry = T[30_000.0, 45_000.0] # qv = [0.40, 0.10]

        ll_mesh = LatLonMesh(; FT = T, Nx = 4, Ny = 2)
        ll_grid = AtmosGrid(ll_mesh, vertical, CPU(); FT = T)
        ll_air = Array{T}(undef, 4, 2, 2)
        for j in 1:2, i in 1:4, k in 1:2
            ll_air[i, j, k] = dp_dry[k] * cell_area(ll_mesh, i, j) /
                              AtmosTransport.Grids.gravity(ll_grid)
        end
        @test ICIO._stored_surface_pressure_basis(
            ll_grid, ll_air, fill(ps_total, 4, 2), DryBasis()) isa ICIO.TotalSurfacePressure

        rg_mesh = ReducedGaussianMesh([-45.0, 45.0], [4, 4]; FT = T)
        rg_grid = AtmosGrid(rg_mesh, vertical, CPU(); FT = T)
        rg_air = Array{T}(undef, ncells(rg_mesh), 2)
        for c in 1:ncells(rg_mesh), k in 1:2
            rg_air[c, k] = dp_dry[k] * cell_area(rg_mesh, c) /
                           AtmosTransport.Grids.gravity(rg_grid)
        end
        @test ICIO._stored_surface_pressure_basis(
            rg_grid, rg_air, fill(ps_total, ncells(rg_mesh)), DryBasis()) isa
            ICIO.TotalSurfacePressure

        Nc, Hp = 2, 1
        cs_mesh = CubedSphereMesh(; FT = T, Nc, Hp)
        cs_grid = AtmosGrid(cs_mesh, vertical, CPU(); FT = T)
        cs_air = ntuple(_ -> zeros(T, Nc + 2Hp, Nc + 2Hp, 2), 6)
        for p in 1:6, j in 1:Nc, i in 1:Nc, k in 1:2
            cs_air[p][i + Hp, j + Hp, k] =
                dp_dry[k] * cell_area(cs_mesh, i, j) /
                AtmosTransport.Grids.gravity(cs_grid)
        end
        cs_ps_total = ntuple(_ -> fill(ps_total, Nc, Nc), 6)
        cs_ps_dry = ntuple(_ -> fill(sum(dp_dry), Nc, Nc), 6)
        @test ICIO._stored_surface_pressure_basis(
            cs_grid, cs_air, cs_ps_total, DryBasis()) isa ICIO.TotalSurfacePressure
        @test ICIO._stored_surface_pressure_basis(
            cs_grid, cs_air, cs_ps_dry, DryBasis()) isa ICIO.DrySurfacePressure
        recovered = ICIO._total_surface_pressure(
            ICIO.DrySurfacePressure(), sum(dp_dry), vertical.A, vertical.B, T[0.4, 0.1])
        @test recovered ≈ ps_total rtol = 16eps(T)
    end

    @test_logs (:warn, r"no humidity payload.*approximating total surface pressure") begin
        ICIO._warn_surface_pressure_approximation(ICIO.DrySurfacePressure(), nothing)
        ICIO._warn_surface_pressure_approximation(ICIO.DrySurfacePressure(), nothing)
    end
    @test ICIO._total_surface_pressure(
        ICIO.DrySurfacePressure(), 98_000.0, zeros(3), [0.0, 0.5, 1.0], nothing) == 98_000.0

    runner = AtmosTransport.Models.DrivenRunner
    @test runner._initial_condition_qv((qv_start = :endpoint,
                                        vdiff = (qv = :vdiff,))) === :endpoint
    @test runner._initial_condition_qv((qv_start = nothing,
                                        vdiff = (qv = :vdiff,))) === :vdiff
    @test runner._initial_condition_qv((qv_start = nothing,
                                        vdiff = nothing)) === nothing

    @testset "validated humidity config reaches LL/RG/CS builders ($T)" for T in (Float32, Float64)
        mktempdir() do dir
            path = joinpath(dir, "q_source.nc")
            _write_synthetic_ic_file(path; with_humidity = true,
                                     humidity_name = "q",
                                     humidity_standard_name = false)
            init_cfg = Dict{String, Any}(
                "kind" => "file",
                "file" => path,
                "variable" => "CO2",
                "humidity_variable" => "q",
            )
            cfg = Dict{String, Any}(
                "input" => Dict("binary_paths" => [path]),
                "numerics" => Dict("float_type" => string(T)),
                "tracers" => Dict(
                    "nested" => Dict("init" => init_cfg),
                    "flat" => copy(init_cfg),
                ),
            )
            @test validate_config(cfg) == (true, String[])
            specs = AtmosTransport.Models.DrivenRunner._parse_tracer_specs(cfg)
            @test length(specs) == 2
            @test all(spec -> spec.init_cfg["humidity_variable"] == "q", specs)

            # Humidity is opt-in: a conventional `q` name does not silently
            # change the conservative carrier.
            autodetect_cfg = copy(init_cfg)
            delete!(autodetect_cfg, "humidity_variable")
            source = ICIO._load_file_initial_condition_source(autodetect_cfg, T, 2)
            @test source.qv === nothing

            vertical = HybridSigmaPressure(T[0, 0, 0], T[0, 0.5, 1])
            ps_total = T(100_000)
            src_bp = [1.0, 2 / 3, 1 / 3, 0.0] # file order: surface -> TOA
            src_qv = [0.35, 0.08, 0.001]
            tgt_p = [0.0, 50_000.0, 100_000.0]
            src_p = src_bp .* Float64(ps_total)
            dp_dry = zeros(Float64, 2)
            for k in 1:2, s in 1:3
                overlap = max(0.0, min(tgt_p[k + 1], src_p[s]) -
                                   max(tgt_p[k], src_p[s + 1]))
                dp_dry[k] += (1 - src_qv[s]) * overlap
            end
            ps_dry = T(sum(dp_dry))
            target_qv = T.(1 .- dp_dry ./ diff(tgt_p))
            cases = (
                (ICIO.TotalSurfacePressure(), ps_total, nothing),
                (ICIO.DrySurfacePressure(), ps_dry, target_qv),
                (ICIO.DrySurfacePressure(), ps_dry, nothing),
            )

            ll_mesh = LatLonMesh(; FT = T, Nx = 4, Ny = 2)
            ll_grid = AtmosGrid(ll_mesh, vertical, CPU(); FT = T)
            ll_air = Array{T}(undef, 4, 2, 2)
            for j in 1:2, i in 1:4, k in 1:2
                ll_air[i, j, k] = T(dp_dry[k] * cell_area(ll_mesh, i, j) /
                                      AtmosTransport.Grids.gravity(ll_grid))
            end
            ll_qv = repeat(reshape(target_qv, 1, 1, 2), 4, 2, 1)
            ll_results = map(cases) do (basis, ps, qv_profile)
                qv = qv_profile === nothing ? nothing : ll_qv
                build_initial_mixing_ratio(ll_air, ll_grid, init_cfg;
                    surface_pressure = fill(ps, 4, 2), pressure_basis = basis,
                    mass_basis = DryBasis(), qv)
            end
            @test ll_results[1] ≈ ll_results[2] rtol = 128eps(T)
            @test all(isfinite, ll_results[3])

            rg_mesh = ReducedGaussianMesh([-45.0, 45.0], [4, 4]; FT = T)
            rg_grid = AtmosGrid(rg_mesh, vertical, CPU(); FT = T)
            rg_air = Array{T}(undef, ncells(rg_mesh), 2)
            for c in 1:ncells(rg_mesh), k in 1:2
                rg_air[c, k] = T(dp_dry[k] * cell_area(rg_mesh, c) /
                                   AtmosTransport.Grids.gravity(rg_grid))
            end
            rg_qv = repeat(reshape(target_qv, 1, 2), ncells(rg_mesh), 1)
            rg_results = map(cases) do (basis, ps, qv_profile)
                qv = qv_profile === nothing ? nothing : rg_qv
                build_initial_mixing_ratio(rg_air, rg_grid, init_cfg;
                    surface_pressure = fill(ps, ncells(rg_mesh)),
                    pressure_basis = basis, mass_basis = DryBasis(), qv)
            end
            @test rg_results[1] ≈ rg_results[2] rtol = 128eps(T)
            @test all(isfinite, rg_results[3])

            Nc, Hp = 2, 1
            cs_mesh = CubedSphereMesh(; FT = T, Nc, Hp)
            cs_grid = AtmosGrid(cs_mesh, vertical, CPU(); FT = T)
            cs_air = ntuple(_ -> zeros(T, Nc + 2Hp, Nc + 2Hp, 2), 6)
            for p in 1:6, j in 1:Nc, i in 1:Nc, k in 1:2
                cs_air[p][i + Hp, j + Hp, k] =
                    T(dp_dry[k] * cell_area(cs_mesh, i, j) /
                      AtmosTransport.Grids.gravity(cs_grid))
            end
            cs_qv = ntuple(_ -> repeat(reshape(target_qv, 1, 1, 2), Nc, Nc, 1), 6)
            cs_results = map(cases) do (basis, ps, qv_profile)
                qv = qv_profile === nothing ? nothing : cs_qv
                build_initial_mixing_ratio(cs_air, cs_grid, init_cfg;
                    surface_pressure = ntuple(_ -> fill(ps, Nc, Nc), 6),
                    pressure_basis = basis, mass_basis = DryBasis(), qv)
            end
            for p in 1:6
                @test cs_results[1][p] ≈ cs_results[2][p] rtol = 128eps(T)
                @test all(isfinite, cs_results[3][p])
            end
        end
    end

    # --------- plan 40 Commit 2: catrine_co2 semantic unification -----
    #
    # After Commit 2, `kind = "catrine_co2"` on every topology is a
    # convenience alias for `kind = "file"` with the default Catrine
    # NetCDF path. The flat-411 stub branch in `build_cs_tracer_panels`
    # was deleted. Verify the equivalence on all three topologies.
    #
    # Requires the real Catrine NetCDF file — skipped when missing
    # (CI or fresh checkouts without `~/data/AtmosTransport/`).
    @testset "`catrine_co2` ≡ `file` with default Catrine path" begin
        catrine_ic_path =
            expanduser("~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc")
        if !isfile(catrine_ic_path)
            @info "Skipping Catrine equivalence tests; file not present at $catrine_ic_path"
            @test_skip false
        else
            cfg_catrine = Dict("kind" => "catrine_co2")
            cfg_file    = Dict("kind" => "file",
                                "file" => catrine_ic_path,
                                "variable" => "CO2")

            # LL
            mesh_ll = LatLonMesh(; Nx = 36, Ny = 18,
                                longitude = (0.0, 360.0),
                                latitude  = (-90.0, 90.0))
            vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[0, 0.5, 1])
            grid_ll = AtmosGrid(mesh_ll, vertical, CPU(); FT = FT)
            air_mass_ll = fill(FT(1e10), 36, 18, 2)
            ps_ll = fill(FT(101325), 36, 18)
            q_catrine_ll = build_initial_mixing_ratio(air_mass_ll, grid_ll, cfg_catrine;
                                                       surface_pressure = ps_ll)
            q_file_ll    = build_initial_mixing_ratio(air_mass_ll, grid_ll, cfg_file;
                                                       surface_pressure = ps_ll)
            @test q_catrine_ll == q_file_ll

            # RG
            latitudes = [-75.0, -25.0, 25.0, 75.0]
            nlon_per_ring = [4, 8, 8, 4]
            mesh_rg = ReducedGaussianMesh(latitudes, nlon_per_ring; FT = FT)
            grid_rg = AtmosGrid(mesh_rg, vertical, CPU(); FT = FT)
            air_mass_rg = fill(FT(1e10), ncells(mesh_rg), 2)
            ps_rg = fill(FT(101325), ncells(mesh_rg))
            q_catrine_rg = build_initial_mixing_ratio(air_mass_rg, grid_rg, cfg_catrine;
                                                       surface_pressure = ps_rg)
            q_file_rg    = build_initial_mixing_ratio(air_mass_rg, grid_rg, cfg_file;
                                                       surface_pressure = ps_rg)
            @test q_catrine_rg == q_file_rg

            # CS — build_initial_mixing_ratio output is interior NTuple{6}
            Nc = 4
            Hp = 1
            Nz = 2
            mesh_cs = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
            grid_cs = AtmosGrid(mesh_cs, vertical, CPU(); FT = FT)
            air_mass_cs = ntuple(_ -> fill(FT(1e10), Nc + 2Hp, Nc + 2Hp, Nz), 6)
            ps_cs = ntuple(_ -> fill(FT(101325), Nc, Nc), 6)
            q_catrine_cs = build_initial_mixing_ratio(air_mass_cs, grid_cs, cfg_catrine;
                                                       surface_pressure = ps_cs)
            q_file_cs    = build_initial_mixing_ratio(air_mass_cs, grid_cs, cfg_file;
                                                       surface_pressure = ps_cs)
            for p in 1:6
                @test q_catrine_cs[p] == q_file_cs[p]
            end
            # Sanity: CS CO2 values are in a plausible atmospheric range.
            all_vals = vcat((vec(q_catrine_cs[p]) for p in 1:6)...)
            @test minimum(all_vals) > 1e-4    # > 100 ppm
            @test maximum(all_vals) < 1e-3    # < 1000 ppm
        end
    end

    @testset "CATRINE C90 near-surface band is preserved" begin
        catrine_ic_path =
            expanduser("~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc")
        if !isfile(catrine_ic_path)
            @info "Skipping CATRINE C90 conservative-remap regression; file not present at $catrine_ic_path"
            @test_skip false
        else
            # Independent bottom-fraction mean for one column. Source files
            # are surface first; model columns are top first.
            function surface_band(q, dp, fraction; surface_first)
                total = sum(dp)
                remaining = fraction * total
                weighted = 0.0
                indices = surface_first ? eachindex(q) : Iterators.reverse(eachindex(q))
                for k in indices
                    take = min(remaining, dp[k])
                    weighted += Float64(q[k]) * take
                    remaining -= take
                    remaining <= 16eps(Float64) * total && break
                end
                return weighted / (fraction * total)
            end

            coeff_path = normpath(joinpath(@__DIR__, "..", "..", "config",
                                           "geos_L72_coefficients.toml"))
            coeff = TOML.parsefile(coeff_path)["coefficients"]
            A = Float64.(coeff["a"])
            B = Float64.(coeff["b"])
            Nz = length(A) - 1

            mesh = CubedSphereMesh(; FT = Float64, Nc = 90, Hp = 1)
            grid = AtmosGrid(mesh, HybridSigmaPressure(A, B), CPU(); FT = Float64)
            air = ntuple(_ -> ones(Float64, 92, 92, Nz), 6)
            cfg = Dict{String, Any}("kind" => "file", "file" => catrine_ic_path,
                                    "variable" => "CO2")

            # Put the source topography on C90, then shift every column by the
            # same 400 Pa. The correct surface-anchored remap is invariant to
            # this pin-sized shift; the replaced absolute-pressure mapping was
            # not. `Psurf` is read only to construct this independent test
            # perturbation and is not part of FileInitialConditionSource.
            source = ICIO._load_file_initial_condition_source(cfg, Float64, Nz)
            source_ps = NCDataset(catrine_ic_path) do ds
                lon = Float64.(ds["longitude"][:])
                lat = Float64.(ds["latitude"][:])
                raw_ps = Float64.(nomissing(ds["Psurf"][:, :], 100_000.0))
                orientation = AtmosTransport.GriddedIO.latlon_orientation(lon, lat)
                AtmosTransport.GriddedIO.orient(raw_ps, orientation)
            end
            source_mesh = AtmosTransport.GriddedIO.source_latlon_mesh(
                source.lon, source.lat; radius = mesh.radius)
            ps_regridder = AtmosTransport.Regridding.build_regridder(source_mesh, mesh)
            ps_flat = Vector{Float64}(undef, 6 * 90 * 90)
            AtmosTransport.Regridding.apply_regridder!(ps_flat, ps_regridder,
                                                       reshape(source_ps, :))
            ps_base = ntuple(_ -> Matrix{Float64}(undef, 90, 90), 6)
            AtmosTransport.Preprocessing.unpack_flat_to_panels_2d!(ps_base, ps_flat, 90)
            ps = ntuple(p -> ps_base[p] .+ 400.0, 6)
            mapped = build_initial_mixing_ratio(air, grid, cfg; surface_pressure = ps)

            target_weighted = 0.0
            target_area = 0.0
            for p in 1:6, j in 1:90, i in 1:90
                area = Float64(cell_area(mesh, i, j))
                target_dp = diff(A .+ B .* ps[p][i, j])
                target_weighted += area * surface_band(@view(mapped[p][i, j, :]),
                                                       target_dp, 0.1;
                                                       surface_first = false)
                target_area += area
            end
            target_mean = target_weighted / target_area

            source_mean = NCDataset(catrine_ic_path) do ds
                q = Float64.(nomissing(ds["CO2"][:, :, :], 0.0))
                source_ps = Float64.(nomissing(ds["Psurf"][:, :], 100_000.0))
                ap = Float64.(ds["ap"][:])
                bp = Float64.(ds["bp"][:])
                lat = Float64.(ds["latitude"][:])
                weighted = 0.0
                area_sum = 0.0
                for j in eachindex(lat), i in axes(q, 1)
                    # Exact spherical strip weight for the regular 1-degree
                    # source grid; the longitude-width factor cancels.
                    south = max(-90.0, lat[j] - 0.5)
                    north = min(90.0, lat[j] + 0.5)
                    area = sind(north) - sind(south)
                    interfaces = ap .+ bp .* source_ps[i, j]
                    dp = -diff(interfaces)
                    weighted += area * surface_band(@view(q[i, j, :]), dp, 0.1;
                                                    surface_first = true)
                    area_sum += area
                end
                weighted / area_sum
            end

            difference_ppm = (target_mean - source_mean) * 1e6
            @info "CATRINE C90 conservative IC regression" target_ppm=target_mean * 1e6 source_ppm=source_mean * 1e6 difference_ppm
            @test abs(difference_ppm) <= 1e-3
        end
    end

    # --------- plan 40 Commit 1d: surface-flux builders --------------
    @testset "build_surface_flux_source supports latlon and reduced grids" begin
        mktempdir() do dir
            flux_path = joinpath(dir, "gridfed.nc")
            _write_synthetic_surface_flux_file(flux_path)

            flux_cfg = Dict{String, Any}(
                "kind" => "gridfed_fossil_co2",
                "file" => flux_path,
                "time_index" => 1,
                "year" => 2021,
            )

            vertical = HybridSigmaPressure(Float64[0, 0], Float64[1, 0])
            native_total = 7.0
            storage_scale = AtmosTransport.Parameters.DRY_AIR_MOLAR_MASS / 44.0095e-3

            latlon_mesh = LatLonMesh(; FT = Float64, Nx = 4, Ny = 2)
            latlon_grid = AtmosGrid(latlon_mesh, vertical, CPU(); FT = Float64)
            ll_source = build_surface_flux_source(latlon_grid, :fossil_co2, flux_cfg, Float64)
            @test sum(ll_source.cell_mass_rate) ≈ native_total * storage_scale rtol = 1e-12

            reduced_mesh = ReducedGaussianMesh([-67.5, -22.5, 22.5, 67.5],
                                               [8, 8, 8, 8]; FT = Float64)
            reduced_grid = AtmosGrid(reduced_mesh, vertical, CPU(); FT = Float64)
            rg_source = build_surface_flux_source(reduced_grid, :fossil_co2, flux_cfg, Float64)
            @test sum(rg_source.cell_mass_rate) ≈ native_total * storage_scale rtol = 1e-12
        end
    end

    @testset "calendar-aware multi-month LMDZ and GridFED loaders" begin
        mktempdir() do dir
            @test ICIO._infer_year_from_path(
                "GCP-GridFEDv2024.0_2022.short.nc") == 2022
            @test ICIO._infer_year_from_path("GCP-GridFEDv2024.0_2021.short.nc") == 2021
            @test ICIO._infer_year_from_path("flux_2021_v2024.nc") == 2021
            @test ICIO._infer_year_from_path("z_cams_l_cams55_202112_FT24r2.nc") === nothing
            @test_throws ArgumentError ICIO._infer_year_from_path("flux_2021_2022.nc")
            for month in 1:12
                _write_synthetic_lmdz_flux_file(
                    joinpath(dir, @sprintf("lmdz_%04d%02d.nc", 2022, month)),
                    2022, month)
            end
            lmdz_cfg = Dict{String, Any}(
                "kind" => "lmdz_co2",
                "file_pattern" => joinpath(dir, "lmdz_{YYYYMM}.nc"),
                "year" => 2022,
                "time_varying" => true,
            )
            lmdz = ICIO._load_timevarying_surface_flux_field(
                lmdz_cfg, Float32, DateTime(2022, 1, 1); species = :co2)
            @test size(lmdz.raw_series) == (4, 2, 24)
            @test lmdz.times_sec[1:2] == [0.0, 3 * 3600.0]
            @test lmdz.times_sec[end] ==
                  Dates.value(DateTime(2022, 12, 1, 3) - DateTime(2022, 1, 1)) / 1000.0
            c_to_co2 = AtmosTransport.Parameters.SPECIES_MOLAR_MASS.co2 /
                       AtmosTransport.Parameters.CARBON_MOLAR_MASS
            @test all(lmdz.raw_series[:, :, 1] .== Float32(c_to_co2))
            @test all(lmdz.raw_series[:, :, 3] .== Float32(2 * c_to_co2))

            gridfed_path = joinpath(dir, "gridfed_2022.nc")
            _write_synthetic_monthly_gridfed_file(gridfed_path, 2022)
            gridfed_cfg = Dict{String, Any}(
                "kind" => "gridfed_fossil_co2",
                "file" => gridfed_path,
                "year" => 2022,
                "time_varying" => true,
            )
            gridfed = ICIO._load_timevarying_surface_flux_field(
                gridfed_cfg, Float64, DateTime(2022, 1, 1); species = :co2)
            @test size(gridfed.raw_series) == (4, 2, 12)
            @test gridfed.times_sec[1] == 0.0
            @test gridfed.times_sec[2] == 31 * 86400.0
            for month in 1:12
                @test all(gridfed.raw_series[:, :, month] .== month)
            end

            # The ordinal fallback belongs only to the documented GridFED
            # layout, even when another kind points at the same data variable.
            not_gridfed = copy(gridfed_cfg)
            not_gridfed["kind"] = "lmdz_co2"
            not_gridfed["variable"] = "TOTAL"
            @test_throws ArgumentError ICIO._load_timevarying_surface_flux_field(
                not_gridfed, Float64, DateTime(2022, 1, 1); species = :co2)
        end
    end

    @testset "per-month slices use CF coordinates, bounds, and calendar" begin
        function write_permonth(path; times, values, calendar, bounds = nothing)
            NCDataset(path, "c") do ds
                defDim(ds, "longitude", 4)
                defDim(ds, "latitude", 2)
                defDim(ds, "time", length(times))
                defVar(ds, "longitude", Float64, ("longitude",))[:] =
                    [-135.0, -45.0, 45.0, 135.0]
                defVar(ds, "latitude", Float64, ("latitude",))[:] = [45.0, -45.0]
                attrib = Dict("units" => "days since 2021-01-01", "calendar" => calendar)
                bounds === nothing || (attrib["bounds"] = "time_bnds")
                defVar(ds, "time", Float64, ("time",); attrib)[:] = times
                if bounds !== nothing
                    defDim(ds, "nv", 2)
                    defVar(ds, "time_bnds", Float64, ("nv", "time"))[:, :] = bounds
                end
                flux = defVar(ds, "TOTAL", Float64,
                              ("longitude", "latitude", "time"),
                              attrib = Dict("units" => "kgCO2/month/m2"))
                for t in eachindex(values)
                    flux[:, :, t] .= values[t]
                end
            end
            return path
        end

        mktempdir() do dir
            # A partial year in file order April, February, March. Bounds assign
            # each amount its duration; the coordinate sorts the returned series.
            bounds = [90.0 31.0 59.0; 120.0 59.0 90.0]
            markers = [4.0, 2.0, 3.0]
            values = markers .* vec(bounds[2, :] .- bounds[1, :]) .* 86400
            path = write_permonth(joinpath(dir, "partial_out_of_order.nc");
                                  times = vec(sum(bounds; dims = 1) ./ 2), values,
                                  calendar = "proleptic_gregorian", bounds)
            cfg = Dict{String, Any}(
                "kind" => "gridfed_fossil_co2", "file" => path,
                "time_varying" => true)
            series = ICIO._load_timevarying_surface_flux_field(
                cfg, Float64, DateTime(2021, 1, 1); species = :co2)
            @test size(series.raw_series, 3) == 3
            @test all(series.raw_series[:, :, 1] .== 2.0)
            @test all(series.raw_series[:, :, 2] .== 3.0)
            @test all(series.raw_series[:, :, 3] .== 4.0)
            # Stepwise slices switch at interval starts, not their mid-month
            # coordinate stamps.
            @test series.times_sec == [31.0, 59.0, 90.0] .* 86400

            # Julian bounds are instants: Julian 2021-01-01 is proleptic
            # Gregorian 2021-01-14, so each switch is 13 days after the
            # Julian calendar fields, as for the decoded coordinate.
            julian_bounds = [0.0 31.0 59.0; 31.0 59.0 90.0]
            julian = write_permonth(joinpath(dir, "julian.nc");
                                    times = vec(sum(julian_bounds; dims = 1) ./ 2),
                                    values = [1.0, 2.0, 3.0] .*
                                             vec(julian_bounds[2, :] .- julian_bounds[1, :]) .* 86400,
                                    calendar = "julian", bounds = julian_bounds)
            julian_series = ICIO._load_timevarying_surface_flux_field(
                Dict{String, Any}("kind" => "gridfed_fossil_co2", "file" => julian,
                                  "time_varying" => true),
                Float64, DateTime(2021, 1, 1); species = :co2)
            @test julian_series.times_sec == ([0.0, 31.0, 59.0] .+ 13) .* 86400
            for month in 1:3
                @test all(julian_series.raw_series[:, :, month] .== month)
            end

            # time_index remains explicitly positional: slice 1 is April.
            static_cfg = Dict{String, Any}(
                "kind" => "gridfed_fossil_co2", "file" => path, "time_index" => 1)
            static = ICIO._load_file_surface_flux_field(
                static_cfg, Float64; species = :co2)
            @test all(static.raw .== 4.0)

            # month is semantic when a coordinate exists, even though the
            # partial-year file is out of order.
            february = ICIO._load_file_surface_flux_field(
                Dict{String, Any}("kind" => "gridfed_fossil_co2", "file" => path,
                                  "month" => 2), Float64; species = :co2)
            @test all(february.raw .== 2.0)
            @test_throws ArgumentError ICIO._load_file_surface_flux_field(
                Dict{String, Any}("kind" => "gridfed_fossil_co2", "file" => path,
                                  "month" => 1), Float64; species = :co2)

            # Fixed-unit bounds must span exactly the represented calendar
            # month; a daily interval cannot normalize a monthly amount.
            bad_bounds = [31.0 59.0 90.0; 32.0 60.0 91.0]
            bad = write_permonth(joinpath(dir, "daily_bounds.nc");
                                 times = vec(sum(bad_bounds; dims = 1) ./ 2),
                                 values = ones(3), calendar = "proleptic_gregorian",
                                 bounds = bad_bounds)
            @test_throws ArgumentError ICIO._load_timevarying_surface_flux_field(
                Dict{String, Any}("kind" => "gridfed_fossil_co2", "file" => bad,
                                  "time_varying" => true),
                Float64, DateTime(2021, 1, 1); species = :co2)

            # Without bounds, month length comes from the coordinate's own
            # 360_day calendar: February is 30 days, not Gregorian 28 days.
            values360 = [m * 30 * 86400.0 for m in 1:3]
            path360 = write_permonth(joinpath(dir, "partial_360_day.nc");
                                     times = [15.0, 45.0, 75.0], values = values360,
                                     calendar = "360_day")
            cfg360 = Dict{String, Any}(
                "kind" => "gridfed_fossil_co2", "file" => path360,
                "time_varying" => true)
            series360 = ICIO._load_timevarying_surface_flux_field(
                cfg360, Float64, DateTime(2021, 1, 1); species = :co2)
            for month in 1:3
                @test all(series360.raw_series[:, :, month] .== month)
            end

            # A coordinate-less partial year is not the legacy twelve-slice
            # GridFED layout and therefore cannot fall back to ordinal months.
            partial_no_coordinate = joinpath(dir, "partial_no_coordinate.nc")
            NCDataset(partial_no_coordinate, "c") do ds
                defDim(ds, "longitude", 4)
                defDim(ds, "latitude", 2)
                defDim(ds, "time", 3)
                defVar(ds, "longitude", Float64, ("longitude",))[:] =
                    [-135.0, -45.0, 45.0, 135.0]
                defVar(ds, "latitude", Float64, ("latitude",))[:] = [45.0, -45.0]
                flux = defVar(ds, "TOTAL", Float64,
                              ("longitude", "latitude", "time"),
                              attrib = Dict("units" => "kgCO2/month/m2"))
                flux[:, :, :] .= 1.0
            end
            partial_cfg = Dict{String, Any}(
                "kind" => "gridfed_fossil_co2", "file" => partial_no_coordinate,
                "time_varying" => true, "year" => 2021)
            @test_throws ArgumentError ICIO._load_timevarying_surface_flux_field(
                partial_cfg, Float64, DateTime(2021, 1, 1); species = :co2)
        end
    end

    @testset "time-varying loaders read only the run period" begin
        times = [0.0, 10.0, 20.0, 30.0]
        @test ICIO._needed_slice_window(times, (12.0, 18.0)) == (10.0, 20.0)
        @test ICIO._needed_slice_window(times, (10.0, 20.0)) == (10.0, 20.0)
        @test ICIO._needed_slice_window(times, (-5.0, 3.0)) == (0.0, 10.0)
        @test ICIO._needed_slice_window(times, (25.0, 99.0)) == (20.0, 30.0)
        @test ICIO._needed_slice_window([30.0, 0.0, 10.0], (12.0, 18.0)) == (10.0, 30.0)
        @test_throws ArgumentError ICIO._needed_slice_window(Float64[], (0.0, 1.0))
        @test_throws ArgumentError ICIO._needed_slice_window(times, (5.0, 1.0))

        mktempdir() do dir
            for month in 1:12
                _write_synthetic_lmdz_flux_file(
                    joinpath(dir, @sprintf("lmdz_%04d%02d.nc", 2022, month)), 2022, month)
            end
            # Each synthetic month holds two 3-hourly slices, so the files do
            # not cover their months; this test is about slice selection.
            cfg = Dict{String, Any}("kind" => "lmdz_co2",
                                    "file_pattern" => joinpath(dir, "lmdz_{YYYYMM}.nc"),
                                    "year" => 2022, "time_varying" => true,
                                    "hold_outside_coverage" => true)
            ref = DateTime(2022, 1, 1)
            full = ICIO._load_timevarying_surface_flux_field(cfg, Float32, ref; species = :co2)
            sec(d) = Dates.value(d - ref) / 1000.0
            # Mar 1 03:00 + 1 h to Apr 1 00:00: Mar 03:00 slice through the Apr 00:00 slice.
            span = (sec(DateTime(2022, 3, 1, 4)), sec(DateTime(2022, 4, 1)))
            sub = ICIO._load_timevarying_surface_flux_field(cfg, Float32, ref; species = :co2,
                                                            run_span = span)
            idx = findall(t -> sec(DateTime(2022, 3, 1, 3)) <= t <= sec(DateTime(2022, 4, 1)),
                          full.times_sec)
            @test length(idx) == 2
            @test sub.times_sec == full.times_sec[idx]
            @test sub.raw_series == full.raw_series[:, :, idx]
            @test sub.lon == full.lon && sub.lat == full.lat
            # Every scheme gives the same emitted increment from the subset: steps
            # starting on and between knots, crossing knots, and at the run ends.
            SF = AtmosTransport.Operators.SurfaceFlux
            increment(field, segs) = sum(f * (w0 .* field.raw_series[:, :, i0] .+
                                              w1 .* field.raw_series[:, :, i1])
                                         for (i0, i1, w0, w1, f) in segs)
            # Feb 1 01:00 to Mar 1 02:00 has knots inside (Feb 1 03:00, Mar 1 00:00).
            span2 = (sec(DateTime(2022, 2, 1, 1)), sec(DateTime(2022, 3, 1, 2)))
            sub2 = ICIO._load_timevarying_surface_flux_field(cfg, Float32, ref; species = :co2,
                                                             run_span = span2)
            @test length(sub2.times_sec) == 4
            n_steps = 0
            for (sp, field) in ((span, sub), (span2, sub2))
                knots = filter(t -> sp[1] <= t <= sp[2], full.times_sec)
                starts = sort(unique(vcat(knots, knots .- 1800, sp[1],
                                          range(sp...; length = 9)[1:end-1])))
                for scheme in (AtmosTransport.StepwiseFlux(), AtmosTransport.LinearInterpFlux(),
                               AtmosTransport.ConservativeMeanFlux()),
                    t in starts, dt in (900.0, 4 * 3600.0, 3 * 86400.0)
                    sp[1] <= t && t + dt <= sp[2] || continue
                    n_steps += 1
                    @test increment(field, SF._flux_temporal_segments(scheme, field.times_sec, t, dt)) ==
                          increment(full, SF._flux_temporal_segments(scheme, full.times_sec, t, dt))
                end
            end
            @test n_steps > 60

            gridfed_path = joinpath(dir, "gridfed_2022.nc")
            _write_synthetic_monthly_gridfed_file(gridfed_path, 2022)
            gcfg = Dict{String, Any}("kind" => "gridfed_fossil_co2", "file" => gridfed_path,
                                     "year" => 2022, "time_varying" => true)
            g = ICIO._load_timevarying_surface_flux_field(
                gcfg, Float64, ref; species = :co2,
                run_span = (sec(DateTime(2022, 6, 10)), sec(DateTime(2022, 7, 2))))
            @test g.times_sec == [sec(DateTime(2022, m, 1)) for m in 6:8]
            # Monthly totals divided by each slice's own month length.
            for (k, month) in enumerate(6:8)
                @test all(g.raw_series[:, :, k] .== month)
            end
        end
    end

    @testset "build_surface_flux_source — `kind = none` returns nothing" begin
        mesh = LatLonMesh(; Nx = 4, Ny = 3,
                          longitude = (0.0, 360.0),
                          latitude  = (-90.0, 90.0))
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        @test build_surface_flux_source(grid, :co2, Dict("kind" => "none"), FT) === nothing

        rg_mesh = ReducedGaussianMesh([-75.0, -25.0, 25.0, 75.0],
                                      [4, 8, 8, 4]; FT = FT)
        rg_grid = AtmosGrid(rg_mesh, vertical, CPU(); FT = FT)
        @test build_surface_flux_source(rg_grid, :co2, Dict("kind" => "none"), FT) === nothing

        cs_mesh = CubedSphereMesh(; FT = FT, Nc = 4, Hp = 1)
        cs_grid = AtmosGrid(cs_mesh, vertical, CPU(); FT = FT)
        @test build_surface_flux_source(cs_grid, :co2, Dict("kind" => "none"), FT) === nothing

        # build_surface_flux_sources with no tracer specs returns empty tuple
        @test build_surface_flux_sources(grid, (), FT) === ()
    end

    @testset "build_surface_flux_source — CS panel shape contract" begin
        # Writes a tiny synthetic LL NetCDF emission file; conservative
        # LL→CS regrid should produce 6 × (Nc, Nc) panels of per-cell
        # model-storage rates.
        # Acceptance: shape is NTuple{6, Matrix{FT}}, each (Nc, Nc);
        # global mass rate is preserved within conservative-regrid tolerance.
        import NCDatasets: NCDataset, defVar, defDim
        Nx_src = 16
        Ny_src = 8
        path = joinpath(mktempdir(), "flux.nc")
        ds = NCDataset(path, "c")
        defDim(ds, "lon", Nx_src)
        defDim(ds, "lat", Ny_src)
        defDim(ds, "time", 1)
        lon_v  = defVar(ds, "lon",  Float64, ("lon",))
        lat_v  = defVar(ds, "lat",  Float64, ("lat",))
        flux_v = defVar(ds, "FLUX", Float32, ("lon", "lat", "time"),
                        attrib = Dict("units" => "kg/m2/s"))
        lon_v[:] = [(i - 0.5) * 360.0 / Nx_src for i in 1:Nx_src]
        lat_v[:] = [-90.0 + (j - 0.5) * 180.0 / Ny_src for j in 1:Ny_src]
        # Uniform 1.0 flux density → dst_total = area of full sphere
        flux_v[:, :, 1] .= 1.0
        close(ds)

        Nc = 6
        Hp = 1
        cs_mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        cs_grid  = AtmosGrid(cs_mesh, vertical, CPU(); FT = FT)

        cfg = Dict("kind" => "file", "file" => path, "variable" => "FLUX",
                   "regridding" => "conservative", "time_index" => 1)
        src = build_surface_flux_source(cs_grid, :co2, cfg, FT)
        @test src !== nothing
        @test src.tracer_name === :co2
        @test src.cell_mass_rate isa NTuple{6, Matrix{FT}}
        for p in 1:6
            @test size(src.cell_mass_rate[p]) == (Nc, Nc)
        end

        # Global integral check: uniform 1 kg/m²/s over the full sphere,
        # converted to dry-air-equivalent storage for CO2.
        R = Float64(cs_mesh.radius)
        storage_scale = AtmosTransport.Parameters.DRY_AIR_MOLAR_MASS / 44.0095e-3
        expected_total = 4π * R^2 * storage_scale
        actual_total = sum(sum(panel) for panel in src.cell_mass_rate)
        @test isapprox(Float64(actual_total), expected_total; rtol = 1e-3)
    end

    @testset "build_surface_flux_source — native time-varying CS density" begin
        Nc = 4
        ntime = 2
        path = joinpath(mktempdir(), "native_cs_flux.nc")
        ds = NCDataset(path, "c")
        defDim(ds, "time", ntime)
        defDim(ds, "nf", 6)
        defDim(ds, "Ydim", Nc)
        defDim(ds, "Xdim", Nc)
        time_v = defVar(ds, "time", Float64, ("time",),
                        attrib = Dict("units" => "hours since 2021-12-01 00:00:00"))
        flux_v = defVar(ds, "CO2_FLUX", Float32,
                        ("Xdim", "Ydim", "nf", "time"),
                        attrib = Dict("units" => "kg CO2 m-2 s-1"))
        time_v[:] = [0.0, 1.0]
        for t in 1:ntime, p in 1:6
            flux_v[:, :, p, t] .= Float32(p + 10t)
        end
        close(ds)

        mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = 1)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        cfg = Dict{String, Any}(
            "kind" => "cs_native",
            "file" => path,
            "variable" => "CO2_FLUX",
            "time_varying" => true,
            "species" => "co2",
        )
        src = build_surface_flux_source(
            grid, :co2_test, cfg, FT; reference_time = DateTime(2021, 12, 1))
        @test src isa AtmosTransport.TimeVaryingSurfaceFluxSource
        @test src.times == [0.0, 3600.0]
        @test src.scheme isa AtmosTransport.StepwiseFlux

        storage_scale = AtmosTransport.Parameters.DRY_AIR_MOLAR_MASS / 44.0095e-3
        for p in 1:6, t in 1:ntime
            expected = (p + 10t) .* mesh.cell_areas .* storage_scale
            @test src.cell_mass_rate_series[p][:, :, t] ≈ expected
        end

        # A run span inside the second hour keeps only the bracketing slice.
        spanned = build_surface_flux_source(
            grid, :co2_test, cfg, FT; reference_time = DateTime(2021, 12, 1),
            run_span = (3600.0, 5400.0))
        @test spanned.times == [3600.0]
        for p in 1:6
            @test spanned.cell_mass_rate_series[p][:, :, 1] ≈ src.cell_mass_rate_series[p][:, :, 2]
        end
        Runner = AtmosTransport.Models.DrivenRunner
        @test Runner._check_flux_run_span((0.0, 7201.0), 7201.0, (spanned,), "a.bin") === nothing
        @test_throws ArgumentError Runner._check_flux_run_span(
            (0.0, 7201.0), 7201.5, (spanned,), "a.bin")
        @test Runner._check_flux_run_span(nothing, 10800.0, (spanned,), "a.bin") === nothing
        @test Runner._check_flux_run_span((0.0, 7201.0), 10800.0, (), "a.bin") === nothing
    end

    @testset "run span from the binaries" begin
        Runner = AtmosTransport.Models.DrivenRunner
        listed = Dict("input" => Dict("binary_paths" => ["a.bin"]))
        dated = Dict("input" => Dict("start_date" => "2021-12-01", "end_date" => "2021-12-02"))
        # The binaries' windows (run_hours) set the span; end_date adds nothing.
        @test Runner._run_time_span(6.0) == (0.0, 6 * 3600.0 + 1)
        @test Runner._flux_run_span(dated, 6.0) == (0.0, 6 * 3600.0 + 1)
        # Without a start date there are no time-varying fluxes (validate_config requires it).
        @test Runner._flux_run_span(listed, 6.0) === nothing
        # Mixed window lengths: the layout gives the run end exactly.
        @test Runner._layout_run_hours([(3600.0, 24), (10800.0, 8)]) == 48.0
        # Window lengths as the run clock sees them (Float32 runs round them).
        @test Runner._runtime_window_layout([(3599.99988, 2)], Float32) ==
              [(Float64(Float32(3599.99988)), 2)]
        @test Runner._runtime_window_layout([(3600.0, 2)], Float32) == [(3600.0, 2)]
    end

    @testset "pack_initial_tracer_mass — CubedSphereMesh (MoistBasis)" begin
        Nc = 4
        Hp = 1
        Nz = 3
        mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        air_mass = ntuple(_ -> fill(FT(1.2e10), Nc + 2 * Hp, Nc + 2 * Hp, Nz), 6)
        vmr_interior = ntuple(_ -> fill(FT(4.11e-4), Nc, Nc, Nz), 6)
        qv = ntuple(_ -> fill(FT(0.02), Nc + 2 * Hp, Nc + 2 * Hp, Nz), 6)

        rm = pack_initial_tracer_mass(grid, air_mass, vmr_interior;
                                      mass_basis = MoistBasis(), qv = qv)
        for p in 1:6
            interior = @view rm[p][Hp + 1 : Hp + Nc, Hp + 1 : Hp + Nc, :]
            expected = vmr_interior[p] .* FT(1.2e10) .* (1 - FT(0.02))
            @test interior == expected
        end

        # MoistBasis without qv errors
        @test_throws ArgumentError pack_initial_tracer_mass(grid, air_mass, vmr_interior;
                                                            mass_basis = MoistBasis())
        # MoistBasis with wrong qv type errors
        qv_flat = fill(FT(0.02), Nc + 2 * Hp, Nc + 2 * Hp, Nz)   # not an NTuple{6}
        @test_throws ArgumentError pack_initial_tracer_mass(grid, air_mass, vmr_interior;
                                                            mass_basis = MoistBasis(),
                                                            qv = qv_flat)
    end

end

@testset "surface-flux units (GriddedIO grammar)" begin
    # The loaders parse the units attribute with GriddedIO.parse_flux_units and
    # convert a slice to kg species m⁻² s⁻¹ by kg_per_unit and period_seconds.
    units(u, species = :co2) = ICIO._flux_units((; attrib = Dict("units" => u)), "F", "f.nc", species)
    density(u, species; iv = nothing, area = nothing, v = 2.0) =
        only(ICIO._to_flux_density!(fill(v, 1, 1), units(u, species), species, iv, area))
    # kg of the species (or unnamed kg) per m² per second: unchanged, bit for bit.
    for u in ("kg m-2 s-1", "kg/m2/s", "kgCO2/m2/s", "kg(CO2) m^-2 s^-1", "kg/m²/s", "kg CO2 m-2 s-1")
        @test density(u, :co2) === 2.0
    end
    @test density("kg m-2 s-1", :rn222) === 2.0
    @test density("kg SF6 m-2 s-1", :sf6) === 2.0
    # kg of carbon → CO2 conserves carbon atoms: M(CO2)/M(C).
    K = AtmosTransport.Parameters.SPECIES_MOLAR_MASS.co2 / AtmosTransport.Parameters.CARBON_MOLAR_MASS
    @test density("kgC m-2 s-1", :co2; v = 1.0) === K
    # Grams and days convert (the former per-second-only parser refused them).
    @test density("g m-2 s-1", :co2; v = 1.0) === 1e-3
    @test density("gC m-2 d-1", :co2; v = 86400.0) ≈ 1e-3 * K
    # Per-month values need their calendar month (GridFED); February 2024 has 29 days.
    @test density("kgCO2/month/m2", :co2; v = 29 * 86400.0,
                  iv = AtmosTransport.GriddedIO.interval(AtmosTransport.GriddedIO.MonthKey(2024, 2))) === 1.0
    @test_throws ArgumentError density("kgCO2/month/m2", :co2)
    # Per-cell amounts need the cell areas (time-varying loaders read densities).
    @test density("Tonnes", :sf6; v = 1.0, area = fill(2.0, 1, 1)) ===
          1000.0 / (2.0 * AtmosTransport.Parameters.DAYS_PER_JULIAN_YEAR * 86400)
    @test_throws ArgumentError density("kg/s", :co2)
    # Missing per-cell areas come from the same snapped source mesh used by
    # conservative regridding, including half-width polar caps. Preserve the
    # caller's latitude row order when the file stores north to south.
    lon = collect(45.0:90.0:315.0)
    lat = [-90.0, 0.0, 90.0]
    mesh = AtmosTransport.GriddedIO.source_latlon_mesh(
        lon, lat; radius = AtmosTransport.Parameters.EARTH_RADIUS)
    expected_area = [cell_area(mesh, i, j) for i in 1:4, j in 1:3]
    @test ICIO._lonlat_cell_areas_m2(lon, lat) == expected_area
    @test ICIO._lonlat_cell_areas_m2(lon, reverse(lat)) == expected_area[:, end:-1:1]
    old_uniform = AtmosTransport.GriddedIO.source_latlon_mesh(
        lon, lat, AtmosTransport.GriddedIO.UniformSpanFaces();
        radius = AtmosTransport.Parameters.EARTH_RADIUS)
    @test expected_area != [cell_area(old_uniform, i, j) for i in 1:4, j in 1:3]
    # A molar flux that names no substance is of the tracer's species
    # (ECCO-Darwin writes mmol m-2 s-1 of CO2).
    M = AtmosTransport.Parameters.SPECIES_MOLAR_MASS
    @test density("mmol m-2 s-1", :co2; v = 1.0) ≈ 1e-3 * M.co2 rtol = 1e-15
    @test density("mol m-2 s-1", :sf6; v = 1.0) ≈ M.sf6 rtol = 1e-15
    @test density("umol/m2/s", :co2; v = 1.0) ≈ 1e-6 * M.co2 rtol = 1e-15
    # A substance named in the string wins over the tracer's species, with or
    # without a space or parentheses.
    K = AtmosTransport.Parameters.SPECIES_MOLAR_MASS.co2 / AtmosTransport.Parameters.CARBON_MOLAR_MASS
    for u in ("mol C m-2 s-1", "molC m-2 s-1", "mol (C) m-2 s-1")
        @test density(u, :co2; v = 1.0) ≈ AtmosTransport.Parameters.CARBON_MOLAR_MASS * K rtol = 1e-15
    end
    # Units of another species than the tracer's, missing and unknown units
    # are errors that name the variable and file.
    @test_throws ArgumentError density("kgCO2/m2/s", :sf6)
    @test_throws ArgumentError density("kgC m-2 s-1", :rn222)
    for u in ("", "kg m-2", "Tg/yr")
        err = try units(u); nothing catch e; e end
        @test err isa ArgumentError && occursin("'F' in f.nc", err.msg)
    end
    @test_throws ArgumentError ICIO._require_reference_time(nothing)
end

@testset "species of a surface flux" begin
    M = AtmosTransport.Parameters.SPECIES_MOLAR_MASS
    MD = AtmosTransport.Parameters.DRY_AIR_MOLAR_MASS
    species = AtmosTransport.Models.InitialConditionIO.surface_flux_species
    # `species` names an entry of SPECIES_MOLAR_MASS ...
    @test species(:fossil, Dict("species" => "co2")) === :co2
    @test species(:anything, Dict("species" => "sf6")) === :sf6
    @test species(:co2, Dict("species" => "rn222")) === :rn222          # explicit wins
    # ... or the tracer name implies it.
    for (name, s) in ((:co2, :co2), (:co2_natural, :co2), (:co2_fossil, :co2),
                      (:fossil_co2, :co2), (:sf6, :sf6), (:rn222, :rn222))
        @test species(name, Dict{String, Any}()) === s
    end
    # Unknown species, unknown tracer names without `species`, and the removed
    # molar_mass_kg_mol are errors naming the known species.
    for (name, cfg) in ((:fossil, Dict("species" => "n2o")), (:fossil, Dict("species" => "CO2")),
                        (:fossil, Dict("species" => 44.0)), (:fossil, Dict{String, Any}()),
                        (:co2, Dict("molar_mass_kg_mol" => 0.0440095)))
        err = try species(name, cfg); nothing catch e; e end
        @test err isa ArgumentError && occursin("\"co2\", \"sf6\", \"rn222\"", err.msg)
    end
    # The storage factor is DRY_AIR_MOLAR_MASS / M(species), bit for bit.
    @test ICIO._surface_flux_storage_scale(:fossil, Dict("species" => "co2")) === MD / M.co2
    @test ICIO._surface_flux_storage_scale(:sf6, Dict{String, Any}()) === MD / M.sf6
    @test ICIO._surface_flux_storage_scale(:x, Dict("species" => "rn222")) === MD / M.rn222

    # validate_config reports a missing or unknown species, and the old key.
    base = Dict{String, Any}("input" => Dict{String, Any}("binary_paths" => ["a.bin"]))
    check(tracers) = AtmosTransport.validate_config(merge(base, Dict{String, Any}("tracers" => tracers)))
    sf(extra...) = Dict{String, Any}("kind" => "edgar_sf6", extra...)
    tracer(flux) = Dict{String, Any}("init" => Dict{String, Any}("kind" => "uniform"),
                                     "surface_flux" => flux)
    ok, errors = check(Dict("fossil" => tracer(sf("species" => "sf6"))))
    @test !any(e -> occursin("species", e), errors)
    ok, errors = check(Dict("fossil" => tracer(sf())))
    @test any(e -> occursin("Set surface_flux.species", e), errors)
    ok, errors = check(Dict("fossil" => tracer(sf("species" => "ch4"))))
    @test any(e -> occursin("\"ch4\" is not a known species", e), errors)
    ok, errors = check(Dict("sf6" => tracer(sf("molar_mass_kg_mol" => 0.146055))))
    @test any(e -> occursin("molar_mass_kg_mol was replaced by species", e), errors)
    @test any(e -> occursin("unknown option", e) && occursin("molar_mass_kg_mol", e), errors)
    # No source, no species needed.
    ok, errors = check(Dict("blob" => tracer(Dict{String, Any}("kind" => "none"))))
    @test !any(e -> occursin("species", e), errors)
end

@testset "time-varying loaders decode time axes through GriddedIO" begin
    # Units are classified before any CFTime call: `months since` gives the
    # first instant of each month (CFTime alone would use 30.44-day months),
    # a noleap `days since` axis is accepted, `years since` is refused.
    function write_series(path, units; calendar = nothing, times = [0.0, 1.0, 2.0])
        NCDataset(path, "c") do ds
            defDim(ds, "lon", 4); defDim(ds, "lat", 2); defDim(ds, "time", length(times))
            defVar(ds, "lon", Float64, ("lon",))[:] = [45.0, 135.0, 225.0, 315.0]
            defVar(ds, "lat", Float64, ("lat",))[:] = [-45.0, 45.0]
            attrib = calendar === nothing ? Dict("units" => units) :
                                            Dict("units" => units, "calendar" => calendar)
            defVar(ds, "time", Float64, ("time",); attrib)[:] = times
            v = defVar(ds, "flux_apos", Float64, ("lon", "lat", "time"),
                       attrib = Dict("units" => "kg m-2 s-1"))
            v[:, :, :] = ones(4, 2, length(times))
        end
        return path
    end
    ref = DateTime(2021, 1, 1)
    load(path) = ICIO._load_timevarying_surface_flux_field(
        Dict{String, Any}("kind" => "lmdz_co2", "time_varying" => true, "file" => path),
        Float64, ref; species = :co2)
    sec(t) = Dates.value(t - ref) / 1000.0
    mktempdir() do dir
        months = load(write_series(joinpath(dir, "months.nc"), "months since 2021-01-16 12:00:00";
                                   calendar = "365_day"))
        @test months.times_sec == sec.([DateTime(2021, 1, 1), DateTime(2021, 2, 1), DateTime(2021, 3, 1)])
        noleap = load(write_series(joinpath(dir, "noleap.nc"), "days since 2021-02-27";
                                   calendar = "noleap"))
        @test noleap.times_sec == sec.([DateTime(2021, 2, 27), DateTime(2021, 2, 28), DateTime(2021, 3, 1)])
        hours = load(write_series(joinpath(dir, "utc.nc"), "hours since 2021-01-01 00:00:00 UTC"))
        @test hours.times_sec == [0.0, 3600.0, 7200.0]
        @test_throws ArgumentError load(write_series(joinpath(dir, "years.nc"), "years since 2021-01-01"))
    end
end
