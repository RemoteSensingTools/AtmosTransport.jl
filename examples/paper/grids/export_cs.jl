# Cell corners (lon, lat) of the model's GEOS-convention cubed sphere, for the grid figure.
using AtmosTransport
using AtmosTransport.Grids: CubedSphereMesh, GMAOCubedSphereDefinition, cs_corner_xyz
Nc = parse(Int, ARGS[1])
mesh = CubedSphereMesh(; Nc, definition = GMAOCubedSphereDefinition())
open(ARGS[2], "w") do io
    println(io, "panel,i,j,lon,lat")
    for p in 1:6, j in 1:Nc+1, i in 1:Nc+1
        x, y, z = cs_corner_xyz(mesh, i, j, p)
        println(io, p, ",", i, ",", j, ",", atand(y, x), ",", asind(clamp(z, -1, 1)))
    end
end
println(typeof(mesh))
