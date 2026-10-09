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
using Printf: @sprintf
import NCDatasets: NCDataset, defDim, defVar

using AtmosTransport
using .AtmosTransport.Grids: cell_index, nrings, ring_longitudes
using .AtmosTransport.Models.InitialConditionIO: build_surface_flux_source,
                                                build_surface_flux_sources

const FT = Float64
const ICIO = AtmosTransport.Models.InitialConditionIO
const _SYNTHETIC_SECONDS_JAN_2021 = 31 * 86400

_synthetic_file_ic_value(lon_deg, lat_deg) =
    3.5e-4 + 1e-6 * mod(lon_deg, 360.0) + 2e-6 * lat_deg

function _write_synthetic_ic_file(path::AbstractString)
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
        vap = defVar(ds, "ap", Float64, ("hlevel",))
        vbp = defVar(ds, "bp", Float64, ("hlevel",))
        vps = defVar(ds, "Psurf", Float64, ("longitude", "latitude"))
        vco2 = defVar(ds, "CO2", Float64, ("longitude", "latitude", "level"))

        lon = [-135.0, -45.0, 45.0, 135.0]
        lat = [45.0, -45.0]  # descending on purpose
        vlon[:] = lon
        vlat[:] = lat
        vlev[:] = [0.0, 1.0, 2.0]
        vhlev[:] = [0.0, 1.0, 2.0, 3.0]
        vap[:] = [0.0, 0.0, 0.0, 0.0]
        vbp[:] = [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]
        vps[:, :] = fill(9.0e4, 4, 2)

        raw = Array{Float64}(undef, 4, 2, 3)
        for k in 1:3, j in 1:2, i in 1:4
            raw[i, j, k] = _synthetic_file_ic_value(lon[i], lat[j])
        end
        vco2[:, :, :] = raw
        vco2.attrib["units"] = "mol mol-1"
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
        tracer = defVar(ds, "SIGNED", Float64,
                        ("Xdim", "Ydim", "nf", "lev"))
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

function _write_synthetic_surface_flux_file(path::AbstractString)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "longitude", 4)
        defDim(ds, "latitude", 2)
        defDim(ds, "time", 2)

        vlon = defVar(ds, "longitude", Float64, ("longitude",))
        vlat = defVar(ds, "latitude", Float64, ("latitude",))
        vtime = defVar(ds, "time", Float64, ("time",))
        vtotal = defVar(ds, "TOTAL", Float64, ("longitude", "latitude", "time"))
        varea = defVar(ds, "cell_area", Float64, ("longitude", "latitude"))

        vlon[:] = [-135.0, -45.0, 45.0, 135.0]
        vlat[:] = [45.0, -45.0]  # descending on purpose
        vtime[:] = [1.0, 2.0]
        raw1 = zeros(Float64, 4, 2)
        raw1[1, 1] = 7 * _SYNTHETIC_SECONDS_JAN_2021
        raw2 = 2 .* raw1
        vtotal[:, :, 1] = raw1
        vtotal[:, :, 2] = raw2
        varea[:, :] = reshape(Float64.(1:8), 4, 2)
        vtotal.attrib["units"] = "kgCO2/month/m2"
    finally
        close(ds)
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
            month_seconds = ICIO._days_in_month(year, month) * 86400.0
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

            init_cfg = Dict{String, Any}(
                "kind" => "file",
                "file" => ic_path,
                "variable" => "CO2",
            )

            vertical = HybridSigmaPressure(Float64[0, 0, 0], Float64[1, 0.5, 0])

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
            vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
            grid = AtmosGrid(mesh, vertical, CPU(); FT)
            air_mass = ntuple(_ -> fill(FT(1e10), Nc + 2Hp, Nc + 2Hp, Nz), 6)
            base_cfg = Dict{String, Any}(
                "kind" => "cs_native",
                "file" => path,
                "variable" => "SIGNED",
                "vertical_order" => "toa_first",
            )

            signed = build_initial_mixing_ratio(air_mass, grid, base_cfg)
            @test minimum(vcat((vec(signed[p]) for p in 1:6)...)) < 0
            @test maximum(vcat((vec(signed[p]) for p in 1:6)...)) > 0

            clamped_cfg = copy(base_cfg)
            clamped_cfg["clamp_negative"] = true
            clamped = build_initial_mixing_ratio(air_mass, grid, clamped_cfg)
            @test minimum(vcat((vec(clamped[p]) for p in 1:6)...)) == 0
            @test maximum(vcat((vec(clamped[p]) for p in 1:6)...)) > 0
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

    @testset "file IC log-pressure interpolation handles TOA-to-surface targets" begin
        f = ICIO._interpolate_log_pressure_profile!
        src_half = Float64[100000.0, 50000.0, 10000.0, 1000.0]
        src_mid = 0.5 .* (src_half[1:end-1] .+ src_half[2:end])
        # Deliberately non-log-linear in pressure: this catches regressions
        # where the source bracket index is carried in the wrong direction
        # while targets are visited TOA -> surface.
        src_q = (src_mid ./ 1000.0) .^ 2  # source levels are surface -> TOA
        A_tgt = Float64[1000.0, 8000.0, 40000.0, 100000.0]
        B_tgt = zeros(Float64, length(A_tgt))
        dest = zeros(Float64, 3)

        f(dest, src_q, src_half, zeros(Float64, 4), 0.0, A_tgt, B_tgt, 0.0)

        logp_interp(q1, q2, p1, p2, p) =
            q1 + (log(p) - log(p1)) / (log(p2) - log(p1)) * (q2 - q1)
        expected = Float64[
            src_q[3],  # target midpoint above source TOA clamps to TOA value
            logp_interp(src_q[2], src_q[3], src_mid[2], src_mid[3], 24000.0),
            logp_interp(src_q[1], src_q[2], src_mid[1], src_mid[2], 70000.0),
        ]
        @test dest ≈ expected rtol=1e-12 atol=1e-12
        @test_throws DimensionMismatch f(zeros(Float64, 3), src_q,
                                         src_half[1:3], zeros(Float64, 3),
                                         0.0, A_tgt, B_tgt, 0.0)
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
            vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
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
                lmdz_cfg, Float32, DateTime(2022, 1, 1))
            @test size(lmdz.raw_series) == (4, 2, 24)
            @test lmdz.times_sec[1:2] == [0.0, 3 * 3600.0]
            @test lmdz.times_sec[end] ==
                  Dates.value(DateTime(2022, 12, 1, 3) - DateTime(2022, 1, 1)) / 1000.0
            @test all(lmdz.raw_series[:, :, 1] .== Float32(44 / 12))
            @test all(lmdz.raw_series[:, :, 3] .== Float32(2 * 44 / 12))

            gridfed_path = joinpath(dir, "gridfed_2022.nc")
            _write_synthetic_monthly_gridfed_file(gridfed_path, 2022)
            gridfed_cfg = Dict{String, Any}(
                "kind" => "gridfed_fossil_co2",
                "file" => gridfed_path,
                "year" => 2022,
                "time_varying" => true,
            )
            gridfed = ICIO._load_timevarying_surface_flux_field(
                gridfed_cfg, Float64, DateTime(2022, 1, 1))
            @test size(gridfed.raw_series) == (4, 2, 12)
            @test gridfed.times_sec[1] == 0.0
            @test gridfed.times_sec[2] == 31 * 86400.0
            for month in 1:12
                @test all(gridfed.raw_series[:, :, month] .== month)
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
            "molar_mass_kg_mol" => 0.0440095,
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
