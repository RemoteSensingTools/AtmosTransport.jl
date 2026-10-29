#!/usr/bin/env julia
#
# Output.define_cs_horizontal_geometry!(ds, mesh) and
# Output.define_provenance_attributes!(ds): the public pieces split out of the
# snapshot writer's private CS geometry and provenance blocks, so that
# generated cubed-sphere input files share one definition of the layout. The
# snapshot writer must produce the same dimensions, variables, data and
# attributes as before the split.

using Test
using NCDatasets

using AtmosTransport
using AtmosTransport.Grids: CubedSphereMesh, GnomonicPanelConvention, GEOSNativePanelConvention,
                            GMAOCubedSphereDefinition, panel_cell_center_lonlat,
                            panel_cell_corner_lonlat, cell_area
using AtmosTransport.Output: define_cs_horizontal_geometry!
using AtmosTransport.Parameters: IFS_EARTH_RADIUS

function write_with(f, path)
    NCDataset(path, "c") do ds
        f(ds)
    end
    return path
end

@testset "Output CS geometry helper" begin
    @testset "define_cs_horizontal_geometry! on its own" begin
        mesh = CubedSphereMesh(; FT = Float64, Nc = 4, radius = IFS_EARTH_RADIUS,
                               convention = GEOSNativePanelConvention(),
                               definition = GMAOCubedSphereDefinition(;
                                   convention = GEOSNativePanelConvention()))
        mktempdir() do dir
            path = write_with(ds -> @test(define_cs_horizontal_geometry!(ds, mesh) === nothing),
                              joinpath(dir, "geometry.nc"))
            NCDataset(path, "r") do ds
                @test collect(keys(ds.dim)) == ["Xdim", "Ydim", "Xcorner", "Ycorner", "nf"]
                @test !haskey(ds, "lev") && !haskey(ds, "time")
                @test ds.attrib["Nc"] == 4
                @test ds.attrib["panel_convention"] == "geos_native"
                @test ds.attrib["longitude_of_central_meridian"] == -10.0
                gm = ds["cubed_sphere"]
                @test gm.attrib["semi_major_axis"] === IFS_EARTH_RADIUS
                @test gm.attrib["inverse_flattening"] === 0.0
                @test gm.attrib["cs_definition"] == ds.attrib["cs_definition"]
                for p in 1:6
                    lon, lat = panel_cell_center_lonlat(mesh, p)
                    clon, clat = panel_cell_corner_lonlat(mesh, p)
                    @test ds["lons"][:, :, p] == lon && ds["lats"][:, :, p] == lat
                    @test ds["corner_lons"][:, :, p] == clon && ds["corner_lats"][:, :, p] == clat
                end
                area = Array(ds["cell_area"])
                @test size(area) == (4, 4, 6)
                @test eltype(area) == Float64
                @test sum(area) ≈ 4π * IFS_EARTH_RADIUS^2 rtol = 1e-12
                @test ds["cell_area"].attrib["coordinates"] == "lons lats"
            end
        end
    end

end
