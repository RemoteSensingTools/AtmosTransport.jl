using Test
using NCDatasets

import AtmosTransport

using .AtmosTransport.Visualization

function write_ll_snapshot(path)
    NCDataset(path, "c") do ds
        defDim(ds, "lon", 4)
        defDim(ds, "lat", 3)
        defDim(ds, "time", 2)
        defVar(ds, "lon", Float64, ("lon",))[:] = [-135.0, -45.0, 45.0, 135.0]
        defVar(ds, "lat", Float64, ("lat",))[:] = [-60.0, 0.0, 60.0]
        defVar(ds, "time", Float64, ("time",))[:] = [0.0, 6.0]
        v = defVar(ds, "co2_column_mean", Float64, ("lon", "lat", "time"),
                   attrib = Dict("units" => "mol mol-1"))
        v[:, :, 1] = fill(400e-6, 4, 3)
        v[:, :, 2] = fill(401e-6, 4, 3)
    end
    return path
end

function write_cs_snapshot(path; panel_convention="gnomonic", geometry=nothing, Nc=2)
    NCDataset(path, "c") do ds
        defDim(ds, "Xdim", Nc)
        defDim(ds, "Ydim", Nc)
        defDim(ds, "nf", 6)
        defDim(ds, "lev", 2)
        defDim(ds, "time", 2)
        ds.attrib["Nc"] = Nc
        ds.attrib["panel_convention"] = panel_convention
        geometry === nothing || foreach(((k, v),) -> ds.attrib[k] = v, geometry)
        defVar(ds, "time", Float64, ("time",))[:] = [0.0, 6.0]
        air = defVar(ds, "air_mass", Float64, ("Xdim", "Ydim", "nf", "lev", "time"))
        co2 = defVar(ds, "co2", Float64, ("Xdim", "Ydim", "nf", "lev", "time"))
        air[:, :, :, 1, :] = fill(2.0, Nc, Nc, 6, 2)
        air[:, :, :, 2, :] = fill(1.0, Nc, Nc, 6, 2)
        co2[:, :, :, 1, 1] = fill(300e-6, Nc, Nc, 6)
        co2[:, :, :, 2, 1] = fill(600e-6, Nc, Nc, 6)
        co2[:, :, :, 1, 2] = fill(330e-6, Nc, Nc, 6)
        co2[:, :, :, 2, 2] = fill(660e-6, Nc, Nc, 6)
    end
    return path
end

@testset "Visualization snapshot field views" begin
    mktempdir() do dir
        ll_path = write_ll_snapshot(joinpath(dir, "ll.nc"))
        ll = open_snapshot(ll_path)
        @test snapshot_topology(ll) isa LatLonSnapshotTopology
        @test available_variables(ll) == [:co2]
        @test snapshot_times(ll) == [0.0, 6.0]

        ll_field = fieldview(ll, :co2; time=6.0, unit=:ppm)
        @test ll_field.values == fill(401.0, 4, 3)
        ll_raster = as_raster(ll_field)
        @test ll_raster.values == fill(401.0, 4, 3)
        @test ll_raster.units == "ppm"

        cs_path = write_cs_snapshot(joinpath(dir, "cs.nc"))
        cs = open_snapshot(cs_path)
        @test snapshot_topology(cs) isa CubedSphereSnapshotTopology
        @test available_variables(cs) == [:co2]

        cs_field = fieldview(cs, :co2; transform=:column_mean, time=1, unit=:ppm)
        @test size(cs_field.values) == (2, 2, 6)
        @test all(isapprox.(cs_field.values, 400.0; atol=1e-12))

        level_field = fieldview(cs, :co2; transform=:level_slice, level=2, time=2, unit=:ppm)
        @test all(isapprox.(level_field.values, 660.0; atol=1e-12))
        @test robust_colorrange([ll_raster]) == (401.0, 401.0 + max(401.0, 1.0) * eps(Float64))

        geos_path = write_cs_snapshot(joinpath(dir, "cs_geos.nc"); panel_convention="geos_native")
        geos = open_snapshot(geos_path)
        @test snapshot_topology(geos).panel_convention === :geos_native
        geos_field = fieldview(geos, :co2; transform=:column_mean, time=1, unit=:ppm)
        geos_raster = as_raster(geos_field; resolution=(24, 12))
        @test size(geos_raster.values) == (24, 12)
        @test maximum(abs.(geos_raster.values .- 400.0)) < 1e-10
    end
end

@testset "CS snapshots regrid on the mesh they were written on" begin
    G = AtmosTransport.Grids
    gmao_laws(offset) = ["cs_definition" => "custom",
                         "cs_coordinate_law" => "gmao_equal_distance_gnomonic",
                         "cs_center_law" => "four_corner_normalized",
                         "longitude_of_central_meridian" => offset]
    # Cell-indexed values, so moving any cell boundary changes the raster.
    cache = SnapshotRegridCache()            # one cache: entries must be per definition
    function cell_raster(snap)
        field = fieldview(snap, :co2; transform=:level_slice, level=1, time=1)
        Nc = snapshot_topology(snap).Nc
        values = [i + 10.0j + 100.0p for i in 1:Nc, j in 1:Nc, p in 1:6]
        cells = AtmosTransport.Visualization.HorizontalField(
            field.topology, values, field.name, field.units, field.time,
            field.time_index, field.transform, field.level, field.source_path)
        return as_raster(cells; resolution=(72, 36), cache).values
    end
    mktempdir() do dir
        plain = open_snapshot(write_cs_snapshot(joinpath(dir, "plain.nc"); Nc=8))
        gmao = open_snapshot(write_cs_snapshot(joinpath(dir, "gmao.nc"); Nc=8, geometry=gmao_laws(0.0)))
        rotated = open_snapshot(write_cs_snapshot(joinpath(dir, "rotated.nc"); Nc=8, geometry=gmao_laws(30.0)))

        @test snapshot_topology(plain).definition == G.EquiangularCubedSphereDefinition()
        def = snapshot_topology(rotated).definition
        @test G.coordinate_law(def) isa G.GMAOEqualDistanceGnomonic
        @test G.center_law(def) isa G.FourCornerNormalizedCenter
        @test G.panel_convention(def) isa G.GnomonicPanelConvention
        @test G.longitude_offset_deg(def) == 30.0

        r_plain, r_gmao, r_rotated = cell_raster(plain), cell_raster(gmao), cell_raster(rotated)
        @test all(isfinite, r_plain) && all(isfinite, r_gmao) && all(isfinite, r_rotated)
        @test maximum(abs, r_gmao .- r_plain) > 0.1           # the coordinate law moves cell edges
        @test maximum(abs, r_rotated .- r_gmao) > 10          # the offset moves cells across panels
        @test length(cache.entries) == 3
    end
end
