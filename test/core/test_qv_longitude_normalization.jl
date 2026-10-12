#!/usr/bin/env julia

"""
Unit tests for QV longitude/latitude normalization helpers in
`src/Preprocessing/mass_support.jl` and the bilinear interpolator in
`src/Preprocessing/transport_binary/latlon_workspaces.jl`.

Codex findings #2 + #3 (regrid-bug audit, 2026-04-28):

- `_normalize_lon_to_centered` should detect ERA5 NetCDF longitudes that
  start at 0° and roll the field by `Nx ÷ 2` columns so it lands in the
  staging-mesh `[-180, 180)` cell-centered convention. Files already
  centered at -180° must be left alone.

- `_interpolate_ll_qv!` target latitudes must use the cell-centered
  staging-mesh convention (`-90 + (j - 0.5) × 180/Ny_dst`), not the
  pole-inclusive convention used by ERA5 source. Without this, the
  interpolation places QV at face latitudes instead of cell centers,
  producing a max ~Δlat/2 mismatch that drifts dry-basis mass.
"""

using Test

using AtmosTransport
using AtmosTransport.Preprocessing: _normalize_lon_to_centered, _interpolate_ll_qv!

@testset "QV longitude normalization (Codex finding #2)" begin
    Nx, Ny, Nz = 8, 4, 2

    @testset "0..360 NetCDF lons rolled to -180..180" begin
        # Mark every column with its "expected lon center" in the staging
        # frame. ERA5 NetCDF lon centers are at 0, 45, 90, …, 315.
        # Staging cell centers are at -157.5, -112.5, …, +157.5.
        # The roll by Nx÷2 = 4 columns maps NetCDF column 1 (lon=0°) to
        # staging column 5 (lon=22.5°), etc. — i.e. the data at NetCDF lon=0
        # ends up at the staging cell whose center is closest to 0°.
        Δλ = 360.0 / Nx
        netcdf_lons = collect(0.0:Δλ:(360 - Δλ))   # 0, 45, …, 315
        # Field encodes (i, j, k) so we can detect mis-rolls
        field = zeros(Float32, Nx, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            field[i, j, k] = netcdf_lons[i]   # value = source longitude
        end
        rolled = _normalize_lon_to_centered(field, netcdf_lons)
        # After rolling by Nx÷2: rolled[1] should now contain the value
        # that was at netcdf column 1+Nx÷2 = column 5 → lon = 180°
        @test rolled[1, 1, 1] == 180f0
        @test rolled[Nx ÷ 2 + 1, 1, 1] == 0f0   # lon=0 lands mid-array
    end

    @testset "Already-centered -180..180 lons are pass-through" begin
        Δλ = 360.0 / Nx
        centered_lons = collect((-180 + Δλ/2):Δλ:(180 - Δλ/2))
        field = rand(Float32, Nx, Ny, Nz)
        rolled = _normalize_lon_to_centered(field, centered_lons)
        @test rolled === field   # same array, no allocation
    end

    @testset "Length mismatch is a no-op" begin
        field = rand(Float32, Nx, Ny, Nz)
        rolled = _normalize_lon_to_centered(field, Float64[1, 2, 3])
        @test rolled === field
    end

    @testset "Union{Missing,Float64} lon vectors without missing values are accepted" begin
        Δλ = 360.0 / Nx
        lons = Union{Missing, Float64}[x for x in 0.0:Δλ:(360 - Δλ)]
        field = zeros(Float32, Nx, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            field[i, j, k] = Float32(lons[i])
        end
        rolled = _normalize_lon_to_centered(field, lons)
        @test rolled[1, 1, 1] == 180f0
        @test rolled[Nx ÷ 2 + 1, 1, 1] == 0f0
    end
end

@testset "_interpolate_ll_qv! target uses cell-centered lats (finding #3)" begin
    # Build a source whose QV equals its own latitude. After interpolation,
    # the destination value at staging-grid cell j should match
    # lat = -90 + (j - 0.5) × 180/Ny_dst — the LatLonMesh cell-center
    # latitude — to within bilinear-interpolation precision.
    Nx_src = 36
    Ny_src = 19            # pole-inclusive: lats -90, -80, …, 90
    Nx_dst = 36
    Ny_dst = 18            # cell-centered:  lats -85, -75, …, 85
    Nz = 2

    Δlat_src = 180.0 / (Ny_src - 1)
    src = zeros(Float64, Nx_src, Ny_src, Nz)
    for k in 1:Nz, j in 1:Ny_src, i in 1:Nx_src
        src[i, j, k] = -90.0 + (j - 1) * Δlat_src
    end

    dst = zeros(Float64, Nx_dst, Ny_dst, Nz)
    _interpolate_ll_qv!(dst, src, Nx_dst, Ny_dst, Nx_src, Ny_src, Nz)

    Δlat_dst = 180.0 / Ny_dst
    @testset "target lat j corresponds to cell-centered lat" begin
        for j in 1:Ny_dst
            expected = -90.0 + (j - 0.5) * Δlat_dst
            actual   = dst[1, j, 1]
            @test isapprox(actual, expected; atol = 1e-10)
        end
    end

    @testset "Old (pole-inclusive) target convention is NOT used" begin
        # Sanity: confirm the new convention differs from the pre-fix
        # `-90 + (j-1) × 180/(Ny_dst - 1)` formula at j=1 (would give -90,
        # whereas the cell-centered convention gives -85).
        old_lat_at_j1 = -90.0 + 0.0 * (180.0 / (Ny_dst - 1))
        new_lat_at_j1 = -90.0 + 0.5 * Δlat_dst
        @test old_lat_at_j1 == -90.0
        @test new_lat_at_j1 == -85.0
        @test isapprox(dst[1, 1, 1], -85.0; atol = 1e-10)
    end
end
