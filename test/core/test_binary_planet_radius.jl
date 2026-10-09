# Transport binaries record the radius of the mesh on which the preprocessor
# computed cell areas and air masses (`planet_radius_m`); the runtime builds its
# mesh on that sphere, and runtime regridding puts the source lat-lon mesh on
# the same sphere. Binaries without the key read as EARTH_RADIUS, the radius the
# runtime used for them before. ATMSNAP snapshots record the radius too.

using Test
using JSON3
using NCDatasets
using AtmosTransport
using AtmosTransport.Parameters: EARTH_RADIUS, IFS_EARTH_RADIUS
using AtmosTransport.Output: SnapshotFrame, write_snapshot_binary
const MD = AtmosTransport.MetDrivers

const VERTICAL = HybridSigmaPressure([0.0, 500.0, 0.0], [0.0, 0.5, 1.0])

# One-window lat-lon binary written from a grid of the given radius.
function latlon_binary(path, radius)
    Nx, Ny = 4, 3
    grid = AtmosGrid(LatLonMesh(; Nx, Ny, radius), VERTICAL, CPU(); radius)
    window = (m = fill(1e16, Nx, Ny, 2), am = zeros(Nx + 1, Ny, 2), bm = zeros(Nx, Ny + 1, 2),
              cm = zeros(Nx, Ny, 3), ps = fill(1e5, Nx, Ny))
    write_transport_binary(path, grid, [window]; steps_per_window = 1, mass_basis = :dry,
                           source_flux_sampling = :window_start_endpoint, flux_sampling = :window_constant)
end

# One-window reduced-Gaussian binary written from a grid of the given radius.
function reduced_binary(path, radius)
    mesh = ReducedGaussianMesh([-45.0, 45.0], [4, 4]; radius)
    grid = AtmosGrid(mesh, VERTICAL, CPU(); radius)
    nc, nf = ncells(mesh), nfaces(mesh)
    window = (m = fill(1e16, nc, 2), hflux = zeros(nf, 2), cm = zeros(nc, 3), ps = fill(1e5, nc))
    write_transport_binary(path, grid, [window]; steps_per_window = 1, mass_basis = :dry,
                           source_flux_sampling = :window_start_endpoint, flux_sampling = :window_constant)
end

# One-window cubed-sphere binary; `kw` reaches the writer.
function cs_binary(path; kw...)
    Nc = 2
    window = (m = ntuple(_ -> fill(1e16, Nc, Nc, 2), 6), am = ntuple(_ -> zeros(Nc + 1, Nc, 2), 6),
              bm = ntuple(_ -> zeros(Nc, Nc + 1, 2), 6), cm = ntuple(_ -> zeros(Nc, Nc, 3), 6),
              ps = ntuple(_ -> fill(1e5, Nc, Nc), 6))
    writer = MD.open_streaming_cs_transport_binary(path, Nc, 6, 2, 1, VERTICAL; mass_basis = :dry, kw...)
    MD.write_streaming_cs_window!(writer, window, Nc, 6)
    MD.close_streaming_transport_binary!(writer)
end

function loaded_grid(path, FT)
    reader = MD.TransportBinaryReader(path; FT)
    try
        return reader.header.planet_radius_m, MD.load_grid(reader; FT)
    finally
        close(reader)
    end
end

# Apply `f!` to a binary's JSON header and rewrite it in place.
function rewrite_header!(f!, path)
    reader = MD.TransportBinaryReader(path; FT = Float64)
    nbytes = reader.header.header_bytes
    close(reader)
    raw = open(io -> read(io, nbytes), path)
    header = JSON3.read(String(raw[1:findfirst(==(0x00), raw) - 1]), Dict{String, Any})
    f!(header)
    json = Vector{UInt8}(JSON3.write(header))
    open(io -> write(io, json, zeros(UInt8, nbytes - length(json))), path, "r+")
end

# As written before the key existed.
drop_radius_key!(path) = rewrite_header!(h -> delete!(h, "planet_radius_m"), path)

@testset "binaries carry the mesh radius to the runtime" begin
    mktempdir() do dir
        for radius in (EARTH_RADIUS, IFS_EARTH_RADIUS), FT in (Float32, Float64)
            for (name, write!) in (("ll", latlon_binary), ("rg", reduced_binary),
                                   ("cs", (p, r) -> cs_binary(p; planet_radius = r)))
                path = joinpath(dir, "$name.bin")
                write!(path, radius)
                recorded, grid = loaded_grid(path, FT)
                @test recorded == radius
                @test (grid.horizontal.radius, grid.planet.radius) === (FT(radius), FT(radius))
            end
            mesh = loaded_grid(joinpath(dir, "ll.bin"), FT)[2].horizontal
            @test sum(cell_area(mesh, i, j) for i in 1:mesh.Nx, j in 1:mesh.Ny) ≈ 4π * radius^2 rtol = 1e-6
        end

        # A caller cannot record another radius through extra_header.
        @test_throws ArgumentError cs_binary(joinpath(dir, "bad.bin"); planet_radius = IFS_EARTH_RADIUS,
                                             extra_header = Dict("planet_radius_m" => EARTH_RADIUS))

        # Binaries written before the key existed read as EARTH_RADIUS.
        for (name, write!) in (("ll", latlon_binary), ("rg", reduced_binary),
                               ("cs", (p, r) -> cs_binary(p; planet_radius = r)))
            path = joinpath(dir, "old_$name.bin")
            write!(path, IFS_EARTH_RADIUS)
            drop_radius_key!(path)
            recorded, grid = loaded_grid(path, Float64)
            @test recorded == EARTH_RADIUS
            @test grid.horizontal.radius == EARTH_RADIUS
        end

        # A recorded radius must be a positive length.
        for bad in (0.0, -1.0, true, "6371229")
            path = joinpath(dir, "bad_radius.bin")
            latlon_binary(path, IFS_EARTH_RADIUS)
            rewrite_header!(h -> h["planet_radius_m"] = bad, path)
            @test_throws ArgumentError MD.TransportBinaryReader(path; FT = Float64)
        end
    end
end

@testset "ATMSNAP snapshots record the mesh radius" begin
    mktempdir() do dir
        mesh = CubedSphereMesh(; FT = Float32, Nc = 2, radius = IFS_EARTH_RADIUS)
        grid = AtmosGrid(mesh, HybridSigmaPressure(Float32[0, 1], Float32[0, 1]), CPU();
                         FT = Float32, radius = IFS_EARTH_RADIUS)
        air = ntuple(_ -> ones(Float32, 2, 2, 1), 6)
        frame = SnapshotFrame(0.0, air, Dict(:tracer => air), :dry)
        path = write_snapshot_binary(joinpath(dir, "r.atmsnap"), [frame], grid; mass_basis = :dry)
        header = open(path, "r") do io
            read(io, 8)
            JSON3.read(String(read(io, Int(read(io, UInt64)))), Dict{String, Any})
        end
        @test header["grid"]["planet_radius_m"] == IFS_EARTH_RADIUS
    end
end

@testset "runtime regridding source mesh shares the destination sphere" begin
    ICIO = AtmosTransport.Models.InitialConditionIO
    lon, lat = collect(0.5:1.0:359.5), collect(-89.5:1.0:89.5)
    @test ICIO._build_source_latlon_mesh(lon, lat).radius == EARTH_RADIUS
    @test ICIO._build_source_latlon_mesh(lon, lat; radius = IFS_EARTH_RADIUS).radius == IFS_EARTH_RADIUS
end

@testset "per-cell inventory totals do not depend on the destination radius" begin
    ICIO = AtmosTransport.Models.InitialConditionIO
    mktempdir() do dir
        # EDGAR-style annual tonnes per cell on a 10° grid, no cell-area variable.
        path = joinpath(dir, "edgar.nc")
        lon, lat = collect(-175.0:10.0:175.0), collect(-85.0:10.0:85.0)
        tonnes = [1.0 + i + 2j for i in eachindex(lon), j in eachindex(lat)]
        NCDataset(path, "c") do ds
            defDim(ds, "lon", length(lon)); defDim(ds, "lat", length(lat))
            defVar(ds, "lon", lon, ("lon",)); defVar(ds, "lat", lat, ("lat",))
            defVar(ds, "emissions", tonnes, ("lon", "lat"); attrib = Dict("units" => "Tonnes"))
        end
        cfg = Dict{String, Any}("kind" => "edgar_sf6", "file" => path, "variable" => "emissions")
        expected = 1000 * sum(tonnes) / (365.25 * 86400) * ICIO._surface_flux_storage_scale(:sf6, cfg)
        totals = map((EARTH_RADIUS, IFS_EARTH_RADIUS)) do radius
            grid = AtmosGrid(CubedSphereMesh(; Nc = 4, radius), VERTICAL, CPU(); radius)
            sum(sum, ICIO.build_surface_flux_source(grid, :sf6, cfg, Float64).cell_mass_rate)
        end
        @test totals[1] ≈ totals[2] rtol = 1e-12
        # The regridder's 10° cells have great-circle edges, 1e-3 smaller than latitude bands.
        @test totals[2] ≈ expected rtol = 2e-3
    end
end
