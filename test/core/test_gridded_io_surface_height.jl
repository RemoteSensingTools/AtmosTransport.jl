# Static ground fields for observation placement (GriddedIO/surface_height.jl):
# the per-grid model surface height dataset (write → read round trip on every
# topology, grid and unit gates) and the point DEM (bilinear lookups, fill,
# coverage, longitude conventions).
using Test, AtmosTransport, NCDatasets
using AtmosTransport.GriddedIO: write_model_surface_height, read_model_surface_height,
                                read_point_elevation, LatLonPointElevation
using AtmosTransport.Observations: ModelSurfaceHeight, CellLocation, model_surface_height,
                                   point_elevation, NoPointElevation, NoModelSurfaceHeight
const O = AtmosTransport.Output

@testset "model surface height: lat-lon round trip and grid gates" begin
    mktempdir() do dir
        mesh = LatLonMesh(; FT = Float64, Nx = 4, Ny = 3)
        field = [100.0 * i + 10.0 * j for i in 1:4, j in 1:3]
        path = write_model_surface_height(joinpath(dir, "zs_ll.nc"), mesh, field;
                                          attributes = Dict("source_file" => "synthetic"))
        h = read_model_surface_height(path, mesh)
        @test h isa ModelSurfaceHeight{Matrix{Float64}}
        @test h.field == field && h.path == path
        # A Float32 run mesh reads the same file (centres within Float32 rounding).
        @test read_model_surface_height(path, LatLonMesh(; FT = Float32, Nx = 4, Ny = 3)).field == field
        cell = AtmosTransport.Observations.locate(AtmosTransport.Observations.cell_locator(mesh), 50.0, 40.0)        # i = 3, j = 3
        @test model_surface_height(h, cell) == field[3, 3]
        @test isnan(model_surface_height(NoModelSurfaceHeight(), cell))
        NCDataset(path) do ds
            @test ds["surface_height"].attrib["units"] == "m"
            @test ds.attrib["source_file"] == "synthetic"
            @test haskey(ds.attrib, "framework_commit")
        end
        # Another grid, a non-finite value, a wrong shape, wrong units.
        @test_throws Exception read_model_surface_height(path, LatLonMesh(; FT = Float64, Nx = 8, Ny = 3))
        @test_throws ArgumentError read_model_surface_height(path, LatLonMesh(; FT = Float64, Nx = 4, Ny = 3,
                                                                             latitude = (-60, 60)))
        @test_throws ArgumentError write_model_surface_height(joinpath(dir, "nan.nc"), mesh, fill(NaN, 4, 3))
        @test_throws DimensionMismatch write_model_surface_height(joinpath(dir, "shape.nc"), mesh, zeros(3, 4))
        bad_units = joinpath(dir, "km.nc")
        cp(path, bad_units)
        NCDataset(bad_units, "a") do ds
            ds["surface_height"].attrib["units"] = "km"
        end
        @test_throws ArgumentError read_model_surface_height(bad_units, mesh)
        @test_throws ArgumentError read_model_surface_height(joinpath(dir, "missing.nc"), mesh)
    end
end

@testset "model surface height: cubed-sphere and reduced-Gaussian round trips" begin
    mktempdir() do dir
        cs = CubedSphereMesh(; FT = Float64, Nc = 4, convention = GEOSNativePanelConvention())
        panels = Array{Float64}(undef, 4, 4, 6)
        for p in 1:6, j in 1:4, i in 1:4
            panels[i, j, p] = 1000.0 * p + 10.0 * j + i
        end
        path = write_model_surface_height(joinpath(dir, "zs_cs.nc"), cs, panels)
        h = read_model_surface_height(path, cs)
        @test h.field isa NTuple{6, Matrix{Float64}}
        @test all(h.field[p] == panels[:, :, p] for p in 1:6)
        # A Float32 C90 run mesh reads a file written on the Float64 mesh (its
        # Float32 centres are 2.7e-6 from the Float64 ones).
        c90 = CubedSphereMesh(; FT = Float64, Nc = 90, convention = GEOSNativePanelConvention())
        p90 = write_model_surface_height(joinpath(dir, "zs_c90.nc"), c90, fill(5.0, 90, 90, 6))
        c90_32 = CubedSphereMesh(; FT = Float32, Nc = 90, convention = GEOSNativePanelConvention())
        @test all(p -> all(==(5.0), p), read_model_surface_height(p90, c90_32).field)
        loc = AtmosTransport.Observations.cell_locator(cs; halo_width = 0)
        cell = AtmosTransport.Observations.locate(loc, 10.0, 80.0)
        @test model_surface_height(h, cell) == panels[cell.i, cell.j, cell.panel]
        # The same Nc with another panel convention is another grid.
        other = CubedSphereMesh(; FT = Float64, Nc = 4, convention = GnomonicPanelConvention())
        @test_throws ArgumentError read_model_surface_height(path, other)
        @test_throws Exception read_model_surface_height(path, CubedSphereMesh(; FT = Float64, Nc = 8,
                                                                               convention = GEOSNativePanelConvention()))

        rg = ReducedGaussianMesh(Float64[-45, 45], [4, 6]; FT = Float64)
        values = collect(1.0:O.ncells(rg))
        rpath = write_model_surface_height(joinpath(dir, "zs_rg.nc"), rg, values)
        rh = read_model_surface_height(rpath, rg)
        @test rh.field == values
        rcell = AtmosTransport.Observations.locate(AtmosTransport.Observations.cell_locator(rg), 100.0, 30.0)
        @test model_surface_height(rh, rcell) == values[rcell.cell]
    end
end

# A DEM file with the given coordinates and values (Julia order (lon, lat)).
# `z` is given as (lon, lat); `dims = ("lat", "lon")` stores it transposed (Julia order).
function write_dem(path, lon, lat, z; units = "m", fill_value = nothing, name = "z", dims = ("lon", "lat"))
    NCDataset(path, "c") do ds
        defDim(ds, "lon", length(lon))
        defDim(ds, "lat", length(lat))
        defVar(ds, "lon", Float64, ("lon",))[:] = lon
        defVar(ds, "lat", Float64, ("lat",))[:] = lat
        attrib = Dict{String, Any}("units" => units)
        fill_value === nothing || (attrib["_FillValue"] = Float32(fill_value))
        stored = dims == ("lon", "lat") ? Float32.(z) : permutedims(Float32.(z))
        for d in dims
            haskey(ds.dim, d) || defDim(ds, d, size(stored, findfirst(==(d), dims)))
        end
        defVar(ds, name, Float32, dims; attrib)[:, :] = stored
    end
    return path
end

@testset "point DEM: bilinear lookups, fill, coverage, longitude conventions" begin
    mktempdir() do dir
        # Global 1° DEM on -179.5..179.5, latitudes stored north to south; the field
        # is linear in latitude and in longitude away from the date line.
        lon = collect(-179.5:1.0:179.5)
        lat = collect(89.5:-1.0:-89.5)
        f(λ, φ) = 2.0 * λ + 10.0 * φ + 1000.0
        global_dem = write_dem(joinpath(dir, "global.nc"), lon, lat, [f(λ, φ) for λ in lon, φ in lat])
        dem = read_point_elevation(global_dem)
        @test dem isa LatLonPointElevation
        @test dem.periodic && issorted(dem.lat)
        @test AtmosTransport.Observations.terrain_source(dem) == global_dem
        for (λ, φ) in ((-100.25, 10.7), (12.3, -45.6), (0.0, 0.0), (179.0, 33.3))
            @test point_elevation(dem, λ, φ) ≈ f(λ, φ) atol = 1e-3
        end
        @test point_elevation(dem, 260.0 - 0.25, 10.7) ≈ f(-100.25, 10.7) atol = 1e-3   # 0..360 convention
        # Across the date line the field jumps; the lookup blends the two edge columns.
        @test point_elevation(dem, 179.9, 0.0) ≈ 0.6 * f(179.5, 0.5) / 2 + 0.6 * f(179.5, -0.5) / 2 +
                                                 0.4 * f(-179.5, 0.5) / 2 + 0.4 * f(-179.5, -0.5) / 2 atol = 1e-3
        # Poleward of the outermost centre: that row.
        @test point_elevation(dem, 10.5, 89.9) ≈ f(10.5, 89.5) atol = 1e-3
        @test isnan(point_elevation(NoPointElevation(), 0.0, 0.0))

        # Regional DEM with an ocean fill: corners with fill are left out; outside is NaN.
        rlon = collect(-10.0:1.0:-5.0)
        rlat = collect(40.0:1.0:44.0)
        z = [100.0 * (λ + 11) for λ in rlon, φ in rlat]
        z[1, :] .= -9999
        regional = read_point_elevation(write_dem(joinpath(dir, "regional.nc"), rlon, rlat, z;
                                                  fill_value = -9999, name = "elevation");
                                        variable = "elevation")
        @test !regional.periodic
        @test point_elevation(regional, -7.5, 41.5) ≈ 100.0 * 3.5 atol = 1e-3
        @test point_elevation(regional, -9.5, 41.0) ≈ 200.0 atol = 1e-3       # fill corner dropped
        @test point_elevation(regional, 355.0 - 0.5, 41.0) ≈ 100.0 * 5.5 atol = 1e-3   # -5.5° in the 0..360 convention
        @test isnan(point_elevation(regional, 0.0, 41.0))
        @test isnan(point_elevation(regional, -7.0, 50.0))
        @test isnan(point_elevation(regional, -10.0, 41.0))                   # on a fill column

        @test_throws ArgumentError read_point_elevation(write_dem(joinpath(dir, "ft.nc"), rlon, rlat, z; units = "ft"))
        @test_throws ArgumentError read_point_elevation(global_dem; variable = "elevation")
        @test_throws ArgumentError read_point_elevation(joinpath(dir, "none.nc"))
        # Bathymetry (ETOPO-like ocean depths) or an undeclared fill value would put
        # ocean records kilometres above the ground: rejected. The Dead Sea shore is land.
        bathy = copy(z)
        bathy[1, :] .= -4000
        err = try
            read_point_elevation(write_dem(joinpath(dir, "bathy.nc"), rlon, rlat, bathy)); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("bathymetry", sprint(showerror, err))
        @test_throws ArgumentError read_point_elevation(write_dem(joinpath(dir, "nofill.nc"), rlon, rlat, z))
        dead_sea = copy(z)
        dead_sea[1, :] .= -430
        @test point_elevation(read_point_elevation(write_dem(joinpath(dir, "deadsea.nc"), rlon, rlat, dead_sea)),
                              -10.0, 41.0) ≈ -430.0
    end
end

@testset "point DEM: axis order, gridline seam, fill above the highest land" begin
    mktempdir() do dir
        # A square DEM stored (lat, lon) in Julia order: the dimension names, not
        # the lengths, tell the axes apart (it used to be read transposed).
        lon = collect(10.0:1.0:14.0)
        lat = collect(40.0:1.0:44.0)
        z = [100.0 * i for i in 1:5, j in 1:5]           # rises with longitude only
        for (name, dims) in (("square_lonlat.nc", ("lon", "lat")), ("square_latlon.nc", ("lat", "lon")))
            dem = read_point_elevation(write_dem(joinpath(dir, name), lon, lat, z; dims))
            @test dem.values == Float32.(z)
            @test point_elevation(dem, 14.0, 40.0) ≈ 500.0
            @test point_elevation(dem, 12.5, 43.0) ≈ 350.0
        end
        rect = read_point_elevation(write_dem(joinpath(dir, "rect_latlon.nc"), collect(10.0:1.0:15.0), lat,
                                              [100.0 * i for i in 1:6, j in 1:5]; dims = ("lat", "lon")))
        @test point_elevation(rect, 15.0, 42.0) ≈ 600.0
        # Dimensions that are not the coordinate dimensions are an error.
        err = try
            read_point_elevation(write_dem(joinpath(dir, "xy.nc"), lon, lat, z; dims = ("x", "y"))); nothing
        catch e
            e
        end
        @test err isa DimensionMismatch && occursin("dimensions", sprint(showerror, err))

        # Gridline-registered global DEM, -180..180 inclusive: the repeated
        # antimeridian column is dropped and the grid wraps.
        glon = collect(-180.0:1.0:180.0)
        glat = collect(-90.0:1.0:90.0)
        g(λ, φ) = 1000.0 + 2.0 * abs(λ) + φ              # equal at -180 and 180
        gridline = read_point_elevation(write_dem(joinpath(dir, "gridline.nc"), glon, glat,
                                                  [g(λ, φ) for λ in glon, φ in glat]))
        @test gridline.periodic && length(gridline.lon) == 360 && gridline.lon[end] == 179.0
        @test point_elevation(gridline, 180.0, 10.0) ≈ g(180.0, 10.0)
        @test point_elevation(gridline, -180.0, 10.0) ≈ g(180.0, 10.0)
        @test point_elevation(gridline, 179.5, 10.0) ≈ (g(179.0, 10.0) + g(180.0, 10.0)) / 2
        @test point_elevation(gridline, -12.25, -33.5) ≈ g(-12.25, -33.5)
        # A grid that spans more than 360° is still refused.
        @test_throws ArgumentError read_point_elevation(write_dem(joinpath(dir, "over.nc"), collect(-180.0:1.0:181.0),
                                                                  glat, zeros(362, 181)))

        # An undeclared positive fill (32767) is refused: it would put records
        # kilometres below the ground. Everest is land.
        high = fill(10.0, 5, 5)
        high[3, 3] = 32767
        err = try
            read_point_elevation(write_dem(joinpath(dir, "fill32767.nc"), lon, lat, high)); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("highest land surface", sprint(showerror, err))
        high[3, 3] = 8849
        @test point_elevation(read_point_elevation(write_dem(joinpath(dir, "everest.nc"), lon, lat, high)),
                              12.0, 42.0) ≈ 8849.0
    end
end

# scripts/preprocessing/build_model_surface_height.jl: surface geopotential on a
# lat-lon grid → conservative regrid → the per-grid dataset the runtime reads.
module SurfaceHeightScript
include(joinpath(@__DIR__, "..", "..", "scripts", "preprocessing", "build_model_surface_height.jl"))
end

# A 2° lat-lon surface field, `(lon, lat)` or `(lon, lat, time)`, latitudes
# north to south as in ERA5 when `descending`.
function write_geopotential(path, values; units, name = "PHIS", descending = false, ntime = 0)
    lon = collect(-179.0:2.0:179.0)
    lat = descending ? collect(89.0:-2.0:-89.0) : collect(-89.0:2.0:89.0)
    z = [values(λ, φ) for λ in lon, φ in lat]
    NCDataset(path, "c") do ds
        defDim(ds, "lon", length(lon)); defDim(ds, "lat", length(lat))
        defVar(ds, "lon", Float64, ("lon",); attrib = Dict("units" => "degrees_east"))[:] = lon
        defVar(ds, "lat", Float64, ("lat",); attrib = Dict("units" => "degrees_north"))[:] = lat
        if ntime == 0
            defVar(ds, name, Float64, ("lon", "lat"); attrib = Dict("units" => units))[:, :] = z
        else
            defDim(ds, "time", ntime)
            defVar(ds, name, Float64, ("lon", "lat", "time"); attrib = Dict("units" => units))[:, :, :] =
                repeat(z, 1, 1, ntime)
        end
    end
    return path
end

@testset "build_model_surface_height.jl: geopotential → per-grid surface height" begin
    mktempdir() do dir
        S = SurfaceHeightScript
        g = AtmosTransport.Parameters.STANDARD_GRAVITY
        # MERRA-2 PHIS units; a constant 1000 m ground is reproduced on every cell.
        flat = write_geopotential(joinpath(dir, "phis_flat.nc"), (λ, φ) -> 1000.0 * g; units = "m+2 s-2")
        cs = CubedSphereMesh(; FT = Float64, Nc = 8, convention = GEOSNativePanelConvention())
        r = S.build_model_surface_height(flat, "PHIS", cs, joinpath(dir, "zs_c8.nc"); cache_dir = nothing)
        @test size(r.field) == (8, 8, 6)
        @test all(x -> isapprox(x, 1000.0; rtol = 1e-10), r.field)
        h = read_model_surface_height(r.path, cs)
        @test all(p -> all(x -> isapprox(x, 1000.0; rtol = 1e-10), p), h.field)
        NCDataset(r.path) do ds
            @test ds.attrib["source_variable"] == "PHIS"
            @test ds.attrib["source_file"] == abspath(flat)
        end
        # ERA5 layout: `z` (m**2 s**-2) with a time axis and descending latitudes. The
        # area mean is conserved, and a field rising to the north stays monotone.
        tilted = write_geopotential(joinpath(dir, "z_era5.nc"), (λ, φ) -> g * (2000.0 + 10.0 * φ);
                                    units = "m**2 s**-2", name = "z", descending = true, ntime = 2)
        ll = LatLonMesh(; FT = Float64, Nx = 36, Ny = 18)
        r = S.build_model_surface_height(tilted, "z", ll, joinpath(dir, "zs_ll.nc"); time_index = 2,
                                         cache_dir = nothing)
        @test isapprox(r.target_mean, r.source_mean; rtol = 1e-10)
        @test size(r.field) == (36, 18)
        @test all(diff(r.field[1, :]) .> 0)
        # Northernmost row 80..90°N: the area-weighted mean of the five 2° source
        # bands (weights sin φ₂ − sin φ₁), not the value at 85° (2850 m). The
        # regridder's cells have great-circle edges, so the overlap areas differ
        # from the latitude bands by about 4e-5 of the value.
        bands = [(φ, φ + 2.0) for φ in 80.0:2.0:88.0]
        w = [sind(b) - sind(a) for (a, b) in bands]
        expected = sum(w .* [2000.0 + 10.0 * (a + 1.0) for (a, _) in bands]) / sum(w)
        @test isapprox(r.field[1, 18], expected; rtol = 1e-4)
        @test read_model_surface_height(r.path, ll).field == r.field
        # Heights in metres are taken as they are; anything else is refused.
        metres = write_geopotential(joinpath(dir, "orog.nc"), (λ, φ) -> 250.0; units = "m", name = "orog")
        @test all(x -> isapprox(x, 250.0; rtol = 1e-10),
                  S.build_model_surface_height(metres, "orog", ll, joinpath(dir, "zs_m.nc"); cache_dir = nothing).field)
        bad = write_geopotential(joinpath(dir, "bad.nc"), (λ, φ) -> 1.0; units = "J kg-1")
        @test_throws ArgumentError S.build_model_surface_height(bad, "PHIS", ll, joinpath(dir, "zs_bad.nc");
                                                                cache_dir = nothing)
        @test_throws ArgumentError S.build_model_surface_height(flat, "Z", ll, joinpath(dir, "zs_none.nc");
                                                                cache_dir = nothing)
    end
end
