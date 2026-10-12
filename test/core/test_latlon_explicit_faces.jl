#!/usr/bin/env julia
#
# LatLonMesh(λᶠ, φᶠ; FT, radius): latitude-longitude meshes from explicit cell
# faces, for the source grids of input files. The face cases are the ones
# measured on the TRENDY v14 S3 models in step P0 of the surface-flux port
# (docs/memos/2026-10-10_python_flux_port_p0_facts.md, Section 2): first
# longitude faces that are not ±180 or 0, a wrapped longitude sequence, 0…360
# grids, descending latitude, pole-centred latitude (CLM/ELM 192 rows, iMAPLE
# 181 rows), the truncated FATES grid and T63 Gaussian latitudes (JSBACH).

using Test
using FastGaussQuadrature: gausslegendre

using AtmosTransport
using AtmosTransport.Grids: LatLonMesh, cell_area, cell_areas_by_latitude,
                            latitude_row_order, face_length
using AtmosTransport.Parameters: EARTH_RADIUS, IFS_EARTH_RADIUS

const Gr = AtmosTransport.Grids
const R = EARTH_RADIUS

# The pre-P1 regular constructor, frozen: faces from `range`, centres as face
# midpoints, spacing from the interval end points. The keyword constructor now
# delegates to the explicit-face constructor and must still produce this mesh.
function frozen_regular_mesh(FT, Nx, Ny, longitude, latitude, radius)
    λ_west, λ_east = FT.(longitude)
    φ_south, φ_north = FT.(latitude)
    faces(a, b, N) = collect(range(a, b, length = N + 1) .|> FT)
    centers(f) = FT[(f[i] + f[i + 1]) / 2 for i in 1:length(f) - 1]
    λᶠ = faces(λ_west, λ_east, Nx)
    φᶠ = faces(φ_south, φ_north, Ny)
    return (; Nx, Ny, Δλ = (λ_east - λ_west) / Nx, Δφ = (φ_north - φ_south) / Ny,
            λᶜ = centers(λᶠ), λᶠ, φᶜ = centers(φᶠ), φᶠ, radius = FT(radius))
end

# Bitwise field equality (isequal distinguishes -0.0 from 0.0 and compares NaN).
function same_mesh(m, ref)
    for name in (:Nx, :Ny, :Δλ, :Δφ, :λᶜ, :λᶠ, :φᶜ, :φᶠ, :radius)
        a, b = getfield(m, name), getproperty(ref, name)
        (typeof(a) == typeof(b) && isequal(a, b)) || return false
    end
    return true
end

# Voronoi faces of ascending centres with the given outer faces.
voronoi_faces(centres, south, north) =
    [south; [(centres[k] + centres[k + 1]) / 2 for k in 1:length(centres) - 1]; north]

global_area(m) = sum(cell_areas_by_latitude(m)) * m.Nx
const FULL_SPHERE = 4π * R^2

# Every Δφ-based metric refuses a mesh with non-uniform latitude faces.
function test_nonuniform_metrics(m)
    @test isnan(m.Δφ)
    @test_throws ArgumentError Gr.dy(m)
    @test_throws ArgumentError Gr.dx(m, 1)
    @test_throws ArgumentError Gr._face_length_x(m, 1)
    @test_throws ArgumentError face_length(m, 1)                          # x-face
    @test isfinite(face_length(m, (m.Nx + 1) * m.Ny + m.Nx + 1))         # y-face
    @test occursin("non-uniform faces", sprint(show, m))
end

@testset "LatLonMesh from explicit faces" begin
    @testset "regular meshes reproduce the frozen pre-P1 constructor bit for bit" begin
        cases = ((Float64, 720, 360, (-180, 180), (-90, 90), EARTH_RADIUS),
                 (Float32, 96, 48, (-180, 180), (-90, 90), EARTH_RADIUS),
                 (Float32, 1440, 720, (-180, 180), (-90, 90), EARTH_RADIUS),
                 (Float64, 288, 181, (0, 360), (-90, 90), IFS_EARTH_RADIUS),
                 (Float32, 7, 5, (-30.5, 60.25), (-10, 35), EARTH_RADIUS),
                 (Float64, 1, 1, (-180, 180), (-90, 90), EARTH_RADIUS))
        for (FT, Nx, Ny, lon, lat, radius) in cases
            ref = frozen_regular_mesh(FT, Nx, Ny, lon, lat, radius)
            kw = LatLonMesh(; FT, Nx, Ny, longitude = lon, latitude = lat, radius = FT(radius))
            @test same_mesh(kw, ref)
            # The faces of a regular mesh, passed explicitly, give the same mesh.
            @test same_mesh(LatLonMesh(kw.λᶠ, kw.φᶠ; FT, radius = FT(radius)), ref)
            # Descending latitude faces are stored ascending: the same mesh again.
            @test same_mesh(LatLonMesh(kw.λᶠ, reverse(kw.φᶠ); FT, radius = FT(radius)), ref)
        end
        # Float32 faces of a regular mesh step unevenly by an ulp; they stay
        # uniform for a Float64 mesh too.
        f32 = LatLonMesh(; FT = Float32, Nx = 360, Ny = 181)
        @test !isnan(LatLonMesh(f32.λᶠ, f32.φᶠ; FT = Float64, radius = R).Δφ)
    end

    @testset "first longitude face anywhere (P0 cases)" begin
        φᶠ = collect(-90.0:1.0:90.0)
        # (model, first face, Nx): a centre on 0° or ±180° puts the date line inside a cell
        for (model, λ₀, Nx) in (("CLASSIC", -180.5, 360), ("CLM", -0.625, 288),
                                ("CLM-FATES/ELM-FATES", -1.25, 144), ("ELM", -180.625, 288),
                                ("JSBACH", -180.9375, 192))
            λᶠ = [λ₀ + k * (360 / Nx) for k in 0:Nx]
            m = LatLonMesh(λᶠ, φᶠ; radius = R)
            @test m.λᶠ == λᶠ
            @test m.Nx == Nx
            @test m.Δλ ≈ 360 / Nx rtol = 1e-14
            @test m.λᶜ ≈ [λ₀ + (k + 0.5) * (360 / Nx) for k in 0:Nx - 1] atol = 1e-12
            @test global_area(m) ≈ FULL_SPHERE rtol = 1e-13
        end
        # CLASSIC: the integer longitudes are the centres
        classic = LatLonMesh([-180.5 + k for k in 0:360], φᶠ; radius = R)
        @test classic.λᶜ == collect(-180.0:179.0)
    end

    @testset "longitude 0 … 360 (ISAM, CLM, ELM-FATES)" begin
        m = LatLonMesh(collect(0.0:0.5:360.0), collect(-90.0:0.5:90.0); radius = R)
        @test m.λᶜ[1] == 0.25 && m.λᶜ[end] == 359.75
        @test m.Δλ == 0.5 && m.Δφ == 0.5
        @test global_area(m) ≈ FULL_SPHERE rtol = 1e-13
    end

    @testset "wrapped longitude sequence (CLM-FATES) is unwrapped, not rolled" begin
        file_centres = [collect(0.0:2.5:177.5); collect(-180.0:2.5:-2.5)]   # file order
        wrapped_faces = [mod(-1.25 + 2.5k + 180, 360) - 180 for k in 0:144]
        @test wrapped_faces[1] == -1.25 && wrapped_faces[73] == 178.75 &&
              wrapped_faces[74] == -178.75 && wrapped_faces[end] == -1.25
        m = LatLonMesh(wrapped_faces, collect(-90.0:2.5:90.0); radius = R)
        @test m.λᶠ == collect(-1.25:2.5:358.75)
        @test m.Δλ == 2.5
        # mesh column i is file column i
        @test mod.(m.λᶜ, 360) == mod.(file_centres, 360)
        @test global_area(m) ≈ FULL_SPHERE rtol = 1e-13
        # The observation locator wraps points into the stored face range.
        loc = AtmosTransport.Output.cell_locator(m)
        @test AtmosTransport.Output.locate(loc, -2.0, 0.0).i == 144     # centre -2.5
        @test AtmosTransport.Output.locate(loc, 1.0, 0.0).i == 1        # centre 0
        @test AtmosTransport.Output.locate(loc, -179.0, 0.0).i == 73    # centre -180
    end

    @testset "descending latitude (VISIT) is stored ascending" begin
        φᶠ_file = collect(90.0:-0.5:-90.0)
        φᶜ_file = collect(89.75:-0.5:-89.75)
        m = LatLonMesh(collect(-180.0:0.5:180.0), φᶠ_file; radius = R)
        rows = latitude_row_order(φᶠ_file)
        @test rows == 360:-1:1
        @test rows isa StepRange{Int, Int}
        @test m.φᶠ == collect(-90.0:0.5:90.0)
        @test m.Δφ == 0.5
        @test m.φᶜ == φᶜ_file[rows]
        # a file field indexed (lon, lat) lands on the mesh through the row order
        field = repeat(reshape(φᶜ_file, 1, :), 720, 1)
        @test view(field, :, rows)[1, :] == m.φᶜ
        @test latitude_row_order(m.φᶠ) == 1:1:360
        @test same_mesh(m, frozen_regular_mesh(Float64, 720, 360, (-180, 180), (-90, 90), R))
    end

    @testset "pole-centred latitude (CLM/ELM 192 rows, iMAPLE 181 rows)" begin
        λᶠ = collect(-180.625:1.25:179.375)
        clm_centres = [-90 + k * 180 / 191 for k in 0:191]
        clm = LatLonMesh(λᶠ, voronoi_faces(clm_centres, -90.0, 90.0); radius = R)
        @test clm.Ny == 192
        @test clm.φᶠ[1] == -90 && clm.φᶠ[end] == 90
        @test clm.φᶠ[2] ≈ -90 + 90 / 191 rtol = 1e-14        # half-width polar cap
        @test global_area(clm) ≈ FULL_SPHERE rtol = 1e-13
        test_nonuniform_metrics(clm)

        imaple = LatLonMesh(collect(-180.0:1.0:180.0),
                            voronoi_faces(collect(-90.0:1.0:90.0), -90.0, 90.0); radius = R)
        @test imaple.Ny == 181
        @test imaple.φᶠ[1:3] == [-90.0, -89.5, -88.5]
        @test imaple.φᶜ[2] == -89.0                          # interior rows keep the file centre
        @test global_area(imaple) ≈ FULL_SPHERE rtol = 1e-13
        test_nonuniform_metrics(imaple)
    end

    @testset "truncated latitude (CLM-FATES / ELM-FATES): physical southern face" begin
        # CESM f19 rows 16 … 95: centre_k = -90 + (k + 16)·180/95; north pole only
        centres = [-90 + (k + 16) * 180 / 95 for k in 0:79]
        south = -90 + 15.5 * 180 / 95
        @test south ≈ -60.6316 atol = 1e-4
        m = LatLonMesh(collect(-1.25:2.5:358.75), voronoi_faces(centres, south, 90.0); radius = R)
        @test m.φᶠ[1] == south
        @test m.φᶠ[2] ≈ -90 + 16.5 * 180 / 95 rtol = 1e-14
        @test m.φᶠ[end] == 90
        test_nonuniform_metrics(m)                            # the northern polar row is a half row
        @test global_area(m) ≈ 2π * R^2 * (1 - sind(south)) rtol = 1e-13
        @test cell_area(m, 1, 1) ≈ R^2 * deg2rad(2.5) * (sind(m.φᶠ[2]) - sind(south)) rtol = 1e-14
    end

    @testset "Gaussian latitude (JSBACH T63, 96 rows, midpoint faces)" begin
        x, _ = gausslegendre(96)
        centres = asind.(x)                                   # ascending
        spacing = diff(centres)
        @test maximum(spacing) - minimum(spacing) > 1e-3      # genuinely non-uniform
        m = LatLonMesh(collect(-180.9375:1.875:179.0625), voronoi_faces(centres, -90.0, 90.0);
                       radius = R)
        @test m.Ny == 96 && m.Nx == 192
        test_nonuniform_metrics(m)
        @test global_area(m) ≈ FULL_SPHERE rtol = 1e-13
        # descending Gaussian faces give the same mesh
        desc = LatLonMesh(m.λᶠ, reverse(m.φᶠ); radius = R)
        @test desc.φᶠ == m.φᶠ && isnan(desc.Δφ)
    end

    @testset "cell areas from the faces" begin
        m = LatLonMesh(collect(0.0:30.0:360.0), [-90.0, -30.0, 10.0, 90.0]; FT = Float32,
                       radius = IFS_EARTH_RADIUS)
        @test eltype(m) == Float32
        @test isnan(m.Δφ)
        Rf = Float32(IFS_EARTH_RADIUS)
        for j in 1:3
            @test cell_area(m, 1, j) == Rf^2 * deg2rad(m.Δλ) * abs(sind(m.φᶠ[j + 1]) - sind(m.φᶠ[j]))
        end
        @test sum(cell_areas_by_latitude(m)) * 12 ≈ 4π * IFS_EARTH_RADIUS^2 rtol = 1e-6
    end

    @testset "invalid faces" begin
        φ = collect(-90.0:30.0:90.0)
        λ = collect(0.0:30.0:360.0)
        @test_throws UndefKeywordError LatLonMesh(λ, φ)                         # radius required
        @test_throws ArgumentError LatLonMesh([0.0, 1.0, 2.0, 3.5], φ; radius = R)   # non-uniform
        @test_throws ArgumentError LatLonMesh(collect(0.0:1.0:361.0), φ; radius = R) # span > 360
        @test_throws ArgumentError LatLonMesh(collect(180.0:-30.0:-180.0), φ; radius = R)  # westward
        @test_throws ArgumentError LatLonMesh([0.0, 0.0, 1.0], φ; radius = R)        # zero width
        @test_throws ArgumentError LatLonMesh([0.0], φ; radius = R)                  # no cell
        @test_throws ArgumentError LatLonMesh([0.0, NaN, 2.0], φ; radius = R)
        @test_throws ArgumentError LatLonMesh(λ, [-91.0, 0.0, 91.0]; radius = R)
        @test_throws ArgumentError LatLonMesh(λ, [-90.0, 0.0, -10.0, 90.0]; radius = R)
        @test_throws ArgumentError LatLonMesh(λ, [-90.0, 0.0, 0.0, 90.0]; radius = R)
        @test_throws ArgumentError LatLonMesh(λ, [-90.0, Inf]; radius = R)
        @test_throws ArgumentError latitude_row_order([-90.0, 10.0, 0.0])
        @test_throws ArgumentError latitude_row_order([0.0])
    end

    # The binary header records only the latitude interval, so the lat-lon
    # writer refuses non-uniform faces instead of writing a uniform mesh.
    @testset "transport-binary writer refuses non-uniform latitude faces" begin
        φ = voronoi_faces(collect(-75.0:15.0:75.0), -90.0, 90.0)   # half-width polar caps
        mesh = LatLonMesh(collect(0.0:30.0:360.0), φ; FT = Float32, radius = R)
        @test isnan(mesh.Δφ)
        vc = Gr.HybridSigmaPressure(Float32[0, 1000], Float32[0, 1])
        grid = Gr.AtmosGrid(mesh, vc, AtmosTransport.Architectures.CPU(); FT = Float32)
        mktempdir() do dir
            path = joinpath(dir, "nonuniform.bin")
            @test_throws "non-uniform latitude faces" AtmosTransport.MetDrivers.write_transport_binary(
                path, grid, Any[]; source_flux_sampling = :window_start_endpoint,
                surface_pressure_basis = DrySurfacePressure())
            @test isempty(readdir(dir))
        end
    end

    @testset "public, documented" begin
        @test :latitude_row_order in names(AtmosTransport.Grids)
        @test haskey(Base.Docs.meta(AtmosTransport.Grids),
                     Base.Docs.Binding(AtmosTransport.Grids, :latitude_row_order))
    end
end
