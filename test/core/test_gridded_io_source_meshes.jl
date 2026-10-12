#!/usr/bin/env julia
#
# GriddedIO source meshes: the cell faces a file's centres imply, on
# `LatLonMesh(λᶠ, φᶠ; radius)`. The cases are those measured in step P0 of the
# surface-flux port (docs/memos/2026-10-10_python_flux_port_p0_facts.md,
# Sections 2, 5 and 6): the snapping rule for regular axes (GFED5 exact,
# FLUXCOM-X round-off ≤ 2.84e-14°), centres on 0° or ±180° (CLASSIC, CLM,
# ELM, JSBACH, FATES), the wrapped CLM-FATES axis, descending latitude (VISIT),
# pole-centred rows (CLM, ELM, iMAPLE), the truncated FATES grid (southern face
# -60.6316°), Gaussian latitudes (JSBACH) and bounds variables that must be
# ignored.
#
# The runtime's former source mesh (`_build_source_latlon_mesh`) is frozen
# below: the snapped faces equal it bit for bit on exactly stored regular grids
# and regional grids; on Float32-stored centres (GridFED 0.1°) the old west face
# carried the coordinate round-off (7.45e-10°), the snapped one does not. The
# legacy face rule `UniformSpanFaces` reproduces it bit for bit everywhere.

using Test
using NCDatasets
using FastGaussQuadrature: gausslegendre

using AtmosTransport
using AtmosTransport.Grids: LatLonMesh, cell_areas_by_latitude, latitude_row_order
using AtmosTransport.GriddedIO: source_latlon_mesh, latlon_orientation, orient_coordinates,
                                SnappedFaces, UniformSpanFaces
using AtmosTransport.Parameters: EARTH_RADIUS, IFS_EARTH_RADIUS

const Gr = AtmosTransport.Grids
const R = EARTH_RADIUS

# ---- Frozen runtime source mesh (src/Models/initial_conditions/cubed_sphere.jl) ----
function frozen_build_source_latlon_mesh(lon_src::Vector{Float64}, lat_src::Vector{Float64};
                                         radius::Real = EARTH_RADIUS)
    Nx_src = length(lon_src)
    Ny_src = length(lat_src)
    dlon = (lon_src[end] - lon_src[1]) / (Nx_src - 1)
    dlat = (lat_src[end] - lat_src[1]) / (Ny_src - 1)
    lon_west  = lon_src[1]   - dlon / 2
    lon_east  = lon_src[end] + dlon / 2
    lat_south = lat_src[1]   - dlat / 2
    lat_north = lat_src[end] + dlat / 2
    snap = 1e-3
    lat_south < -90 + snap * dlat && (lat_south = -90.0)
    lat_north >  90 - snap * dlat && (lat_north =  90.0)
    lon_east - lon_west > 360 - snap * dlon && (lon_east = lon_west + 360)
    return LatLonMesh(; FT = Float64, Nx = Nx_src, Ny = Ny_src,
                      longitude = (lon_west, lon_east), latitude = (lat_south, lat_north),
                      radius = Float64(radius))
end

function same_mesh(a, b)
    for name in fieldnames(LatLonMesh)
        x, y = getfield(a, name), getfield(b, name)
        (typeof(x) == typeof(y) && isequal(x, y)) || return false
    end
    return true
end

full_sphere_area(m) = sum(cell_areas_by_latitude(m)) * m.Nx
centres(n, x0, span) = [x0 + (k + 0.5) * span / n for k in 0:(n - 1)]

@testset "regular grids: identical to the runtime mesh where it was exact" begin
    cases = [
        ("CAMS 1°, oriented 0..360", collect(0.5:1.0:359.5), collect(-89.5:1.0:89.5)),
        ("0.5°, -180..180", collect(-179.75:0.5:179.75), collect(-89.75:0.5:89.75)),
        ("GFED5 0.25°", centres(1440, -180, 360), centres(720, -90, 180)),
        ("regional", collect(0.5:1.0:29.5), collect(-9.5:1.0:9.5)),
        ("regional, negative lon", collect(-99.5:1.0:-50.5), collect(20.5:1.0:49.5)),
        ("2.5° × 2°", collect(1.25:2.5:358.75), collect(-89.0:2.0:89.0)),
    ]
    for (name, lon, lat) in cases
        @testset "$name" begin
            for radius in (EARTH_RADIUS, IFS_EARTH_RADIUS)
                @test same_mesh(source_latlon_mesh(lon, lat; radius),
                                frozen_build_source_latlon_mesh(lon, lat; radius))
            end
        end
    end

    # GridFED: Float32 centres stored as Float64 (-179.9499969482422, …).
    lon = Float64.(Float32.(centres(3600, -180, 360)))
    lat = Float64.(Float32.(centres(1800, -90, 180)))
    m = source_latlon_mesh(lon, lat; radius = R)
    @test m.λᶠ == [Float64(-180 * 3600 + 360j) / 3600 for j in 0:3600]   # from integers
    @test m.λᶠ[1] === -180.0 && m.λᶠ[end] === 180.0
    @test m.φᶠ[1] === -90.0 && m.φᶠ[end] === 90.0
    # The runtime oriented the centres first (0.05 … 359.95, mod 360) and took
    # the faces from their span: its west face was 7.45e-10°, not 0.
    o = latlon_orientation(lon, lat)
    lon_o, lat_o = orient_coordinates(lon, lat, o)
    new = source_latlon_mesh(lon_o, lat_o; radius = R)
    old = frozen_build_source_latlon_mesh(lon_o, lat_o)
    @test new.λᶠ == [Float64(360j) / 3600 for j in 0:3600]
    @test old.φᶠ == new.φᶠ
    @test 7e-10 < old.λᶠ[1] < 8e-10
    @test 0 < maximum(abs.(old.λᶠ .- new.λᶠ)) < 1e-9
    # Without the storage-precision allowance these centres would not snap:
    # their residual (3e-5 cells) exceeds the 1e-6 bound.
    @test abs((lon[1] + 180) / 0.1 - 0.5) > 1e-6
end

@testset "snapping rule" begin
    # GFED5 finalized: Float32 coordinates equal the exact centres.
    lon32 = Float32.(centres(1440, -180, 360))
    lat32 = Float32.(centres(720, -90, 180))
    exact = source_latlon_mesh(centres(1440, -180, 360), centres(720, -90, 180); radius = R)
    @test same_mesh(source_latlon_mesh(lon32, lat32; radius = R), exact)
    @test exact.λᶠ[1] === -180.0 && exact.λᶠ[end] === 180.0
    @test exact.φᶠ[1] === -90.0 && exact.φᶠ[end] === 90.0
    @test isapprox(full_sphere_area(exact), 4π * R^2; rtol = 1e-12)

    # FLUXCOM-X: Float64 centres with round-off up to 2.84e-14° (≤ 1.1e-13
    # cells), latitude descending. The faces come from integers: the same mesh.
    lon = centres(1440, -180, 360) .+ 2.84e-14 .* sin.(1:1440)
    lat = reverse(centres(720, -90, 180)) .+ 7.1e-15 .* cos.(1:720)
    @test lon != centres(1440, -180, 360)
    m = source_latlon_mesh(lon, lat; radius = R)
    @test same_mesh(m, exact)
    @test latitude_row_order([90.0, 0.0, -90.0]) == 2:-1:1

    # A half-cell offset or a wrong resolution throws.
    @test_throws ArgumentError source_latlon_mesh(collect(-179.5:1.0:179.5), collect(-89.0:1.0:90.0); radius = R)
    @test_throws ArgumentError source_latlon_mesh(collect(-179.5:1.0:179.5), collect(-90.0:1.0:89.0); radius = R)
    @test_throws ArgumentError source_latlon_mesh(collect(-179.75:1.0:179.25), collect(-89.5:1.0:89.5); radius = R)
    bad = centres(720, -90, 180); bad[100] += 0.3 * 0.25
    @test_throws ArgumentError source_latlon_mesh(centres(1440, -180, 360), bad; radius = R)
    badlon = centres(360, -180, 360); badlon[7] += 0.01
    @test_throws ArgumentError source_latlon_mesh(badlon, centres(180, -90, 180); radius = R)
end

@testset "centres on 0° or ±180°" begin
    lat = collect(-89.5:1.0:89.5)
    for (name, lon, first_face) in (("CLASSIC", collect(-180.0:1.0:179.0), -180.5),
                                    ("CLM", collect(0.0:1.25:358.75), -0.625),
                                    ("ELM", collect(-180.0:1.25:178.75), -180.625),
                                    ("JSBACH", collect(-180.0:1.875:178.125), -180.9375),
                                    ("ELM-FATES", collect(0.0:2.5:357.5), -1.25))
        m = source_latlon_mesh(Float32.(lon), lat; radius = R)
        @test m.λᶠ[1] == first_face
        @test m.λᶠ[end] == first_face + 360
        @test m.λᶜ == lon                                  # centres reproduced exactly
        @test isapprox(full_sphere_area(m), 4π * R^2; rtol = 1e-12)
    end
    # CLM-FATES: 0 … 177.5, then -180 … -2.5. Columns keep their file order;
    # the faces are unwrapped (-1.25 … 358.75).
    wrapped = Float32.([collect(0.0:2.5:177.5); collect(-180.0:2.5:-2.5)])
    m = source_latlon_mesh(wrapped, lat; radius = R)
    @test m.λᶠ[1] == -1.25 && m.λᶠ[end] == 358.75
    @test m.λᶜ == [collect(0.0:2.5:177.5); collect(180.0:2.5:357.5)]
    @test all(diff(m.λᶠ) .== 2.5)
    # The same grid oriented by the runtime (rolled, mod 360) gives the same faces.
    @test source_latlon_mesh(collect(0.0:2.5:357.5), lat; radius = R).λᶠ == m.λᶠ
    # A quarter-cell registration is neither.
    @test_throws ArgumentError source_latlon_mesh(collect(-179.75:1.0:179.25), lat; radius = R)
end

@testset "pole-centred, truncated and Gaussian latitudes" begin
    lon = collect(0.0:1.25:358.75)
    # CLM/ELM: 192 rows from pole to pole, Float32-rounded centres.
    clm = Float32.([-90 + k * 180 / 191 for k in 0:191])
    m = source_latlon_mesh(lon, clm; radius = R)
    φ = Float64.(clm)
    @test m.φᶠ == [-90.0; [(φ[k] + φ[k + 1]) / 2 for k in 1:191]; 90.0]
    @test isnan(m.Δφ)
    @test_throws ArgumentError Gr.dy(m)
    @test isapprox(full_sphere_area(m), 4π * R^2; rtol = 1e-12)
    @test m.φᶠ[2] - m.φᶠ[1] ≈ 90 / 191 atol = 1e-5                       # half-width polar cap
    # iMAPLE: 181 rows of 1°, pole to pole.
    m = source_latlon_mesh(collect(-179.5:1.0:179.5), Float32.(collect(-90.0:1.0:90.0)); radius = R)
    @test m.φᶠ == [-90.0; collect(-89.5:1.0:89.5); 90.0]
    @test m.Ny == 181
    # FATES: f19 rows 16 … 95, truncated south of -59.684°, north row on the pole.
    fates = Float32.([-90 + (k + 16) * 180 / 95 for k in 0:79])
    m = source_latlon_mesh(collect(0.0:2.5:357.5), fates; radius = R)
    @test isapprox(m.φᶠ[1], -90 + 15.5 * 180 / 95; atol = 1e-5)             # -60.6316°
    @test m.φᶠ[end] == 90.0
    @test isapprox(m.φᶠ[2], -90 + 16.5 * 180 / 95; atol = 1e-5)
    # JSBACH: T63 Gaussian latitudes. Midpoint faces, polar outer faces (the
    # rule that reproduces the Python audit), non-uniform.
    x, _ = gausslegendre(96)
    jsb = asind.(x)
    m = source_latlon_mesh(collect(-180.0:1.875:178.125), jsb; radius = R)
    @test m.φᶠ == [-90.0; [(jsb[k] + jsb[k + 1]) / 2 for k in 1:95]; 90.0]
    @test isnan(m.Δφ)
    @test_throws ArgumentError Gr.dx(m, 1)
    @test isapprox(full_sphere_area(m), 4π * R^2; rtol = 1e-12)
    # A regional latitude axis: outer faces at half a spacing.
    m = source_latlon_mesh(lon, collect(20.25:0.5:49.75); radius = R)
    @test m.φᶠ[1] == 20.0 && m.φᶠ[end] == 50.0
end

@testset "descending latitude (VISIT)" begin
    lon = collect(-179.75:0.5:179.75)
    up = source_latlon_mesh(lon, collect(-89.75:0.5:89.75); radius = R)
    down = source_latlon_mesh(lon, collect(89.75:-0.5:-89.75); radius = R)
    @test same_mesh(up, down)
    # Descending pole-centred rows store the same faces as ascending ones.
    asc = source_latlon_mesh(lon, collect(-90.0:1.0:90.0); radius = R)
    desc = source_latlon_mesh(lon, collect(90.0:-1.0:-90.0); radius = R)
    @test asc.φᶠ == desc.φᶠ
end

@testset "faces never from bounds variables" begin
    dir = mktempdir()
    path = joinpath(dir, "fluxcom_like.nc")
    lon = centres(1440, -180, 360)
    lat = reverse(centres(720, -90, 180))
    NCDataset(path, "c") do ds
        defDim(ds, "lon", 1440); defDim(ds, "lat", 720); defDim(ds, "bnds", 2)
        defVar(ds, "lon", Float64, ("lon",); attrib = Dict("units" => "degrees_north",
                                                           "long_name" => "latitude"))[:] = lon
        defVar(ds, "lat", Float64, ("lat",))[:] = lat
        lb = hcat(lon .- 0.125, lon .+ 0.125)'
        lb[1, 1] = -179.99999999999997; lb[2, end] = 180.00000000000003
        defVar(ds, "lon_bnds", Float64, ("bnds", "lon"))[:, :] = lb
        defVar(ds, "lat_bnds", Float64, ("bnds", "lat"))[:, :] = hcat(lat .+ 0.471, lat .- 0.471)'
    end
    m = NCDataset(ds -> source_latlon_mesh(ds; radius = R), path)
    @test same_mesh(m, source_latlon_mesh(lon, lat; radius = R))
    @test m.λᶠ[1] === -180.0 && m.λᶠ[end] === 180.0
    @test m.φᶠ[1] === -90.0 && m.φᶠ[end] === 90.0
    @test m.radius == R
    # Float32 coordinates in the file are read raw (their precision is kept).
    path32 = joinpath(dir, "gridfed_like.nc")
    NCDataset(path32, "c") do ds
        defDim(ds, "longitude", 3600); defDim(ds, "latitude", 1800)
        defVar(ds, "longitude", Float32, ("longitude",))[:] = Float32.(centres(3600, -180, 360))
        defVar(ds, "latitude", Float32, ("latitude",))[:] = Float32.(centres(1800, -90, 180))
    end
    m = NCDataset(ds -> source_latlon_mesh(ds; radius = IFS_EARTH_RADIUS), path32)
    @test m.λᶠ[1] === -180.0 && m.φᶠ[end] === 90.0 && m.radius == IFS_EARTH_RADIUS
end

@testset "malformed axes" begin
    lon = collect(-179.5:1.0:179.5)
    lat = collect(-89.5:1.0:89.5)
    @test_throws ArgumentError source_latlon_mesh([0.5], lat; radius = R)
    @test_throws ArgumentError source_latlon_mesh(lon, [0.0]; radius = R)
    @test_throws ArgumentError source_latlon_mesh(lon, [0.0, 10.0, 5.0]; radius = R)       # not monotone
    @test_throws ArgumentError source_latlon_mesh(lon, [80.0, 95.0]; radius = R)           # beyond the pole
    @test_throws ArgumentError source_latlon_mesh([0.0, 10.0, 0.0, 10.0, 0.0], lat; radius = R)   # wraps twice
    @test_throws ArgumentError source_latlon_mesh([0.0, 1.0, 3.0, 4.0], lat; radius = R)  # non-uniform
    @test_throws ArgumentError source_latlon_mesh(collect(0.0:2.0:720.0), lat; radius = R) # > 360°
    @test_throws ArgumentError source_latlon_mesh(lon, [NaN, 1.0]; radius = R)
end

@testset "UniformSpanFaces: the former runtime mesh, bit for bit" begin
    oriented(lon, lat) = orient_coordinates(lon, lat, latlon_orientation(lon, lat))
    gridfed = oriented(Float64.(Float32.(centres(3600, -180, 360))),
                       Float64.(Float32.(centres(1800, -90, 180))))
    cases = [
        ("CAMS 1°, oriented", collect(0.5:1.0:359.5), collect(-89.5:1.0:89.5)),
        ("GridFED Float32 0.1°, oriented", gridfed...),
        ("Zhang 0.5°, oriented", oriented(Float64.(Float32.(collect(-179.75:0.5:179.75))),
                                          Float64.(Float32.(collect(-89.75:0.5:89.75))))...),
        # Pole-centred LMDZ rows (CATRINE SF6 initial state, 144 × 143): the
        # uniform faces are shifted from the cells, as the former runtime had them.
        ("LMDZ pole-centred", collect(0.0:2.5:357.5), collect(range(-90.0, 90.0; length = 143))),
        ("regional", collect(0.5:1.0:29.5), collect(-9.5:1.0:9.5)),
        ("regional, negative lon", collect(-99.5:1.0:-50.5), collect(20.5:1.0:49.5)),
    ]
    for (name, lon, lat) in cases, radius in (EARTH_RADIUS, IFS_EARTH_RADIUS)
        @testset "$name, R = $radius" begin
            @test same_mesh(source_latlon_mesh(lon, lat, UniformSpanFaces(); radius),
                            frozen_build_source_latlon_mesh(lon, lat; radius))
        end
    end
    lon, lat = gridfed
    @test 7e-10 < source_latlon_mesh(lon, lat, UniformSpanFaces(); radius = R).λᶠ[1] < 8e-10
    @test source_latlon_mesh(lon, lat, SnappedFaces(); radius = R).λᶠ[1] === 0.0
    @test same_mesh(source_latlon_mesh(lon, lat, SnappedFaces(); radius = R),
                    source_latlon_mesh(lon, lat; radius = R))          # the default rule
    # Float32 centre vectors give the same mesh as their Float64 values.
    @test same_mesh(source_latlon_mesh(Float32.(lon), Float32.(lat), UniformSpanFaces(); radius = R),
                    source_latlon_mesh(Float64.(Float32.(lon)), Float64.(Float32.(lat)),
                                       UniformSpanFaces(); radius = R))
    # The dataset form reads the coordinates by name.
    path = joinpath(mktempdir(), "cams_like.nc")
    NCDataset(path, "c") do ds
        defDim(ds, "longitude", 360); defDim(ds, "latitude", 180)
        defVar(ds, "longitude", Float64, ("longitude",))[:] = collect(0.5:1.0:359.5)
        defVar(ds, "latitude", Float64, ("latitude",))[:] = collect(-89.5:1.0:89.5)
    end
    @test same_mesh(NCDataset(ds -> source_latlon_mesh(ds, UniformSpanFaces(); radius = R), path),
                    frozen_build_source_latlon_mesh(collect(0.5:1.0:359.5), collect(-89.5:1.0:89.5)))
    @test_throws ArgumentError source_latlon_mesh([0.5], collect(-89.5:1.0:89.5), UniformSpanFaces(); radius = R)
    @test_throws UndefKeywordError source_latlon_mesh(lon, lat, UniformSpanFaces())
end
