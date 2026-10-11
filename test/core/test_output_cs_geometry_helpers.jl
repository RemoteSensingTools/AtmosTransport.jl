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
using AtmosTransport.Output: define_cs_horizontal_geometry!, define_provenance_attributes!
using AtmosTransport.Parameters: IFS_EARTH_RADIUS

const O = AtmosTransport.Output

# The snapshot writer's CS geometry block before the split, frozen verbatim
# (helpers qualified). The new path is `Output._define_geometry!`.
function frozen_define_cs_geometry!(ds, mesh::CubedSphereMesh, Nz::Integer, times)
    Nc = mesh.Nc
    defDim(ds, "Xdim", Nc)
    defDim(ds, "Ydim", Nc)
    defDim(ds, "Xcorner", Nc + 1)
    defDim(ds, "Ycorner", Nc + 1)
    defDim(ds, "nf", 6)
    O._define_lev!(ds, Nz)
    O._define_time!(ds, times)

    convention_tag = O._cs_convention_tag(mesh)
    central_meridian = O._cs_central_meridian(mesh)
    coordinate_tag = O.coordinate_law_tag(O.coordinate_law(mesh))
    center_tag = O.center_law_tag(O.center_law(mesh))
    definition_tag = String(O.cs_definition_tag(O.cs_definition(mesh)))
    ds.attrib["Nc"] = Nc
    ds.attrib["cs_definition"] = definition_tag
    ds.attrib["cs_coordinate_law"] = coordinate_tag
    ds.attrib["cs_center_law"] = center_tag
    ds.attrib["panel_convention"] = convention_tag
    ds.attrib["grid_mapping_name"] = "gnomonic cubed-sphere"
    ds.attrib["longitude_of_central_meridian"] = central_meridian

    defVar(ds, "Xdim", Float64, ("Xdim",),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "Fake Longitude for GrADS Compatibility",
                         "axis" => "X"))[:] = collect(1.0:Nc)
    defVar(ds, "Ydim", Float64, ("Ydim",),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "Fake Latitude for GrADS Compatibility",
                         "axis" => "Y"))[:] = collect(1.0:Nc)
    defVar(ds, "nf", Int32, ("nf",),
           attrib = Dict("axis" => "e",
                         "grads_dim" => "e",
                         "long_name" => "cubed-sphere face"))[:] = Int32.(1:6)

    lons, lats = O._cs_panel_center_arrays(mesh)
    clons, clats = O._cs_panel_corner_arrays(mesh)
    defVar(ds, "lons", Float64, ("Xdim", "Ydim", "nf"),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "cell-center longitude",
                         "standard_name" => "longitude"))[:, :, :] = lons
    defVar(ds, "lats", Float64, ("Xdim", "Ydim", "nf"),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "cell-center latitude",
                         "standard_name" => "latitude"))[:, :, :] = lats
    defVar(ds, "corner_lons", Float64, ("Xcorner", "Ycorner", "nf"),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "cell-corner longitude"))[:, :, :] = clons
    defVar(ds, "corner_lats", Float64, ("Xcorner", "Ycorner", "nf"),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "cell-corner latitude"))[:, :, :] = clats

    area = Array{Float64}(undef, Nc, Nc, 6)
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        area[i, j, p] = Float64(cell_area(mesh, i, j))
    end
    defVar(ds, "cell_area", Float64, ("Xdim", "Ydim", "nf"),
           attrib = Dict("units" => "m2",
                         "long_name" => "horizontal cell area",
                         "standard_name" => "cell_area",
                         "coordinates" => "lons lats"))[:, :, :] = area

    gm = defVar(ds, "cubed_sphere", Int32, (),
                attrib = Dict("grid_mapping_name" => "gnomonic cubed-sphere",
                              "cs_definition" => definition_tag,
                              "cs_coordinate_law" => coordinate_tag,
                              "cs_center_law" => center_tag,
                              "panel_convention" => convention_tag,
                              "longitude_of_central_meridian" => central_meridian,
                              "semi_major_axis" => Float64(mesh.radius),
                              "inverse_flattening" => 0.0,
                              "comment" => "Use lons/lats and corner_lons/corner_lats as authoritative native-cell coordinates."))
    gm[] = Int32(0)
    return (;)
end

# Everything a reader sees, by name: dimensions (in order) and lengths, global
# attributes (in order, with value types), and per variable the stored element
# type, dimension names, attributes and raw (untransformed) data.
function file_contents(path)
    NCDataset(path, "r") do ds
        dims = [(name, ds.dim[name]) for name in keys(ds.dim)]
        globals = [(k, v, typeof(v)) for (k, v) in ds.attrib]
        vars = Dict(name => (eltype(ds[name].var), dimnames(ds[name]),
                             Dict((k, (v, typeof(v))) for (k, v) in ds[name].attrib),
                             Array(ds[name].var))
                    for name in keys(ds))
        return (; dims, globals, vars)
    end
end

function write_with(f, path)
    NCDataset(path, "c") do ds
        f(ds)
    end
    return path
end

@testset "Output CS geometry and provenance helpers" begin
    meshes = (CubedSphereMesh(; FT = Float64, Nc = 4, convention = GEOSNativePanelConvention(),
                              definition = GMAOCubedSphereDefinition(;
                                  convention = GEOSNativePanelConvention())),
              CubedSphereMesh(; FT = Float32, Nc = 4, convention = GnomonicPanelConvention()))

    @testset "snapshot geometry is unchanged by the split ($(summary(mesh)), times=$(repr(times)))" for
            mesh in meshes, times in ([0.0, 6.0], nothing)
        mktempdir() do dir
            Nz = 3
            before = write_with(ds -> frozen_define_cs_geometry!(ds, mesh, Nz, times),
                                joinpath(dir, "before.nc"))
            after = write_with(ds -> @test(O._define_geometry!(ds, mesh, Nz, times) === (;)),
                               joinpath(dir, "after.nc"))
            a, b = file_contents(before), file_contents(after)
            @test a.dims == b.dims                      # same names, lengths and order
            @test a.globals == b.globals
            @test sort(collect(keys(a.vars))) == sort(collect(keys(b.vars)))
            for name in keys(a.vars)
                @test isequal(a.vars[name], b.vars[name])
            end
        end
    end

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

    @testset "define_provenance_attributes!" begin
        mktempdir() do dir
            local result
            path = write_with(ds -> (result = define_provenance_attributes!(ds)),
                              joinpath(dir, "provenance.nc"))
            @test keys(result) == (:creation_date, :git_commit, :git_dirty)
            creation_date, git_commit, git_dirty = result       # positional destructuring
            @test occursin(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", creation_date)
            @test (git_commit, git_dirty) == Tuple(AtmosTransport.source_revision())
            @test git_dirty in ("clean", "dirty", "unknown")
            NCDataset(path, "r") do ds
                @test ds.attrib["creation_date"] == creation_date
                @test ds.attrib["framework"] == "AtmosTransport.jl"
                @test ds.attrib["framework_commit"] == git_commit
                @test ds.attrib["framework_dirty"] == git_dirty
                @test occursin("Julia $(VERSION)", ds.attrib["runtime"])
                @test haskey(ds.attrib, "hostname") && haskey(ds.attrib, "user")
            end
        end
    end

    @testset "public, documented" begin
        for name in (:define_cs_horizontal_geometry!, :define_provenance_attributes!)
            @test name in names(AtmosTransport.Output)
            binding = Base.Docs.Binding(AtmosTransport.Output, name)
            @test haskey(Base.Docs.meta(AtmosTransport.Output), binding)
        end
        @test !isdefined(AtmosTransport.Output, :_define_cs_geometry!)
        @test !isdefined(AtmosTransport.Output, :_define_provenance_attributes!)
    end
end
