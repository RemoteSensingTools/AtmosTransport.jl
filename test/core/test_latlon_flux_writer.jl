#!/usr/bin/env julia
#
# Lat-lon surface-flux files (`GriddedIO.write_latlon_flux`, design 2.4) for
# `surface_flux.kind = "file"`: static CF (lon, lat) Float32 densities without
# fill values and without a time axis (the runtime reads one static field),
# `lon`/`lat` centres the source-mesh rule turns back into the mesh faces,
# `lon_bnds`/`lat_bnds`, the mesh `cell_area`, atomic publication; the runtime
# reads such a file (conservative regrid onto a cubed sphere conserves the
# emitted mass). The runtime rebuilds snapped faces from the file coordinates,
# so the writer refuses meshes whose faces that rule does not reproduce bit for
# bit; the regrid of an accepted mesh uses exactly its cells.

using Test
using Dates
using NCDatasets
using FastGaussQuadrature: gausslegendre

using AtmosTransport
using AtmosTransport: CubedSphereMesh, LatLonMesh, HybridSigmaPressure, AtmosGrid, CPU,
                      SurfaceFluxSource
using AtmosTransport.GriddedIO: write_latlon_flux, TimeInterval, source_latlon_mesh,
                                find_coordinate, LongitudeAxis, LatitudeAxis,
                                latlon_orientation, orient_coordinates
using AtmosTransport.Grids: cell_area
using AtmosTransport.Models.InitialConditionIO: build_surface_flux_source
using AtmosTransport.Parameters: DRY_AIR_MOLAR_MASS, SPECIES_MOLAR_MASS, EARTH_RADIUS

pattern(i, j, t) = Float32(1f-9 * (i + 100j + 10000t))

# The mesh the runtime regrids a written file on: the file's centres, oriented,
# with snapped faces (the `kind = "file"` loader's rule).
function runtime_mesh(path, radius)
    lon, lat = NCDataset(ds -> (Float64.(ds["lon"][:]), Float64.(ds["lat"][:])), path)
    lon_o, lat_o = orient_coordinates(lon, lat, latlon_orientation(lon, lat))
    return source_latlon_mesh(lon_o, lat_o; radius)
end

@testset "lat-lon flux writer" begin
    mktempdir() do dir
        mesh = LatLonMesh(; FT = Float64, Nx = 36, Ny = 18)
        Nx, Ny = mesh.Nx, mesh.Ny
        static = [pattern(i, j, 0) for i in 1:Nx, j in 1:Ny]
        series = [pattern(i, j, t) for i in 1:Nx, j in 1:Ny, t in 1:3]
        path = joinpath(dir, "ocean.nc")
        @test write_latlon_flux(path, mesh,
                                ["co2_flux_ocean" => (static, Dict("long_name" => "air-sea CO2 flux"))];
                                species = :co2, attributes = Dict("source" => "test"),
                                config_text = "x = 1\n") == path
        @test isempty(filter(f -> occursin(".tmp-", f), readdir(dir)))
        NCDataset(path) do ds
            v = ds["co2_flux_ocean"]
            @test Tuple(dimnames(v)) == ("lon", "lat")
            @test !haskey(ds, "time") && !haskey(ds.dim, "time") && !haskey(ds, "time_bnds")
            @test eltype(v.var) === Float32
            @test v.var[:, :] == static
            @test !haskey(v.attrib, "_FillValue") && !haskey(v.attrib, "missing_value")
            @test v.attrib["units"] == "kg CO2 m-2 s-1"
            @test v.attrib["sign_convention"] == "positive into the atmosphere"
            @test v.attrib["cell_measures"] == "area: cell_area"
            @test v.attrib["long_name"] == "air-sea CO2 flux"
            @test NCDatasets.chunking(v.var) == (:chunked, (Nx, Ny))
            @test find_coordinate(ds, LongitudeAxis()) == "lon"
            @test find_coordinate(ds, LatitudeAxis()) == "lat"
            @test ds["lon"][:] == Float64.(mesh.λᶜ)
            @test ds["lat"][:] == Float64.(mesh.φᶜ)
            @test ds["lon_bnds"][:, :] == permutedims(hcat(mesh.λᶠ[1:Nx], mesh.λᶠ[2:end]))
            @test ds["lat_bnds"][:, :] == permutedims(hcat(mesh.φᶠ[1:Ny], mesh.φᶠ[2:end]))
            @test ds["cell_area"][:, :] == [cell_area(mesh, i, j) for i in 1:Nx, j in 1:Ny]
            # The source-mesh rule rebuilds the mesh faces from the stored centres.
            rebuilt = source_latlon_mesh(ds; radius = mesh.radius)
            @test rebuilt.λᶠ == mesh.λᶠ && rebuilt.φᶠ == mesh.φᶠ
            @test ds.attrib["species"] == "co2"
            @test ds.attrib["source"] == "test"
        end

        # Two variables, both spec forms in one vector, and the runtime reading
        # the file (kind = "file").
        spath = joinpath(dir, "static.nc")
        write_latlon_flux(spath, mesh, ["F" => static, "G" => (2 .* static, Dict("long_name" => "g"))];
                          species = :co2)
        NCDataset(spath) do ds
            @test Tuple(dimnames(ds["F"])) == ("lon", "lat")
            @test ds["F"].var[:, :] == static
            @test ds["G"].var[:, :] == 2 .* static
            @test ds["G"].attrib["long_name"] == "g"
        end
        withenv("ATMOSTR_REGRID_CACHE_DIR" => mkpath(joinpath(dir, "cache"))) do
            grid = AtmosGrid(CubedSphereMesh(; FT = Float64, Nc = 6, Hp = 1),
                             HybridSigmaPressure([0.0, 50000.0, 0.0], [1.0, 0.5, 0.0]), CPU(); FT = Float64)
            cfg = Dict{String, Any}("kind" => "file", "file" => spath, "variable" => "F",
                                    "species" => "co2")
            source = build_surface_flux_source(grid, :co2, cfg, Float64)
            @test source isa SurfaceFluxSource
            emitted = sum(sum, source.cell_mass_rate) / (DRY_AIR_MOLAR_MASS / SPECIES_MOLAR_MASS.co2)
            @test emitted ≈ sum(Float64(static[i, j]) * cell_area(mesh, i, j) for i in 1:Nx, j in 1:Ny) rtol = 1e-10
        end

        # Static only: a series (lon, lat, time) is refused, and there is no
        # time-interval argument (kind = "file" is not a time-varying kind).
        @test_throws DimensionMismatch write_latlon_flux(joinpath(dir, "bad.nc"), mesh, ["F" => series]; species = :co2)
        days = [TimeInterval(DateTime(2008, 12, d), DateTime(2008, 12, d + 1)) for d in 1:3]
        @test_throws MethodError write_latlon_flux(joinpath(dir, "bad.nc"), mesh, days, ["F" => series]; species = :co2)
        @test_throws MethodError write_latlon_flux(joinpath(dir, "bad.nc"), mesh, nothing, ["F" => static]; species = :co2)

        # Errors: non-finite values, wrong type or shape, reserved names and attributes.
        bad = copy(static)
        bad[3, 4] = NaN32
        @test_throws ArgumentError write_latlon_flux(joinpath(dir, "bad.nc"), mesh, ["F" => bad]; species = :co2)
        @test_throws ArgumentError write_latlon_flux(joinpath(dir, "bad.nc"), mesh, ["F" => Float64.(static)]; species = :co2)
        @test_throws DimensionMismatch write_latlon_flux(joinpath(dir, "bad.nc"), mesh, ["F" => static[1:end-1, :]]; species = :co2)
        @test_throws ArgumentError write_latlon_flux(joinpath(dir, "bad.nc"), mesh, ["lat" => static]; species = :co2)
        @test_throws ArgumentError write_latlon_flux(joinpath(dir, "bad.nc"), mesh,
                                                     ["F" => (static, Dict("_FillValue" => -999f0))]; species = :co2)
        @test_throws ArgumentError write_latlon_flux(joinpath(dir, "bad.nc"), mesh, ["F" => static]; species = :ch4)
        @test_throws ArgumentError write_latlon_flux(joinpath(dir, "bad.nc"), mesh, ["F" => 1.0f0]; species = :co2)
        @test_throws ArgumentError write_latlon_flux(joinpath(dir, "bad.nc"), mesh, ["F" => static, "F" => static];
                                                     species = :co2)
        @test !isfile(joinpath(dir, "bad.nc"))
        @test isempty(filter(f -> occursin(".tmp-", f), readdir(dir)))
    end
end

@testset "only meshes the runtime rebuilds exactly" begin
    mktempdir() do dir
        R = EARTH_RADIUS
        field(m) = [pattern(i, j, 0) for i in 1:m.Nx, j in 1:m.Ny]
        write(m, name) = write_latlon_flux(joinpath(dir, name), m, ["F" => field(m)]; species = :co2)
        # Uniform cell-centred meshes: written, and the runtime regrids on exactly
        # the mesh's cells (in its orientation: longitudes from 0°).
        accepted = [
            ("1°, -180..180", LatLonMesh(; FT = Float64, Nx = 360, Ny = 180)),
            ("1°, 0..360", LatLonMesh(; FT = Float64, Nx = 360, Ny = 180, longitude = (0, 360))),
            ("0.5° (ECCO-Darwin)", LatLonMesh(; FT = Float64, Nx = 720, Ny = 360)),
            ("2.5° × 2°", LatLonMesh(; FT = Float64, Nx = 144, Ny = 90)),
            ("snapped Float32 0.1° (GridFED)",
             source_latlon_mesh(Float64.(Float32.([-180 + (k + 0.5) / 10 for k in 0:3599])),
                                Float64.(Float32.([-90 + (k + 0.5) / 10 for k in 0:1799]));
                                radius = R)),
            ("regional", LatLonMesh(; FT = Float64, Nx = 30, Ny = 20, longitude = (0, 30), latitude = (-10, 10))),
            ("regional, west of 0°", LatLonMesh(; FT = Float64, Nx = 50, Ny = 30, longitude = (-100, -50),
                                                latitude = (20, 50))),
        ]
        for (name, m) in accepted
            @testset "accepted: $name" begin
                path = write(m, "ok_$(hash(name)).nc")
                @test isfile(path)
                rt = runtime_mesh(path, m.radius)
                @test rt.φᶠ == m.φᶠ
                @test sort(mod.(rt.λᶠ[1:end-1], 360)) == sort(mod.(m.λᶠ[1:end-1], 360))
                @test rt.Δλ == m.Δλ
            end
        end
        # Meshes the runtime would regrid on other cells: refused, nothing written.
        lon = collect(0.0:2.5:357.5)
        x, _ = gausslegendre(48)
        refused = [
            ("pole-centred (LMDZ rows)", source_latlon_mesh(lon, collect(range(-90.0, 90.0; length = 73)); radius = R)),
            ("truncated (FATES rows)", source_latlon_mesh(lon, Float32.([-90 + (k + 16) * 180 / 95 for k in 0:79]); radius = R)),
            ("Gaussian (T31)", source_latlon_mesh(collect(0.0:3.75:356.25), asind.(x); radius = R)),
            ("regional across 0°", LatLonMesh(; FT = Float64, Nx = 60, Ny = 30, longitude = (-30, 30), latitude = (20, 50))),
        ]
        for (name, m) in refused
            @testset "refused: $name" begin
                file = joinpath(dir, "no_$(hash(name)).nc")
                err = try write(m, basename(file)); nothing catch e; e end
                @test err isa ArgumentError
                @test !isfile(file)
            end
        end
        @test isempty(filter(f -> occursin(".tmp-", f), readdir(dir)))
    end
end
