#!/usr/bin/env julia

using Test
using Dates
using NCDatasets

import AtmosTransport

const _fill_masked_surface = AtmosTransport.Preprocessing._fill_masked_surface

@testset "masked-surface neighbour fill" begin
    # All-finite slice round-trips unchanged.
    A = Float32[1 2 3; 4 5 6; 7 8 9]
    @test _fill_masked_surface(A, Float32) == A

    # A single stray masked cell takes the mean of its finite 4-neighbours,
    # NOT the global mean — local structure is preserved.
    B = Union{Missing,Float32}[10 10 10; 10 missing 10; 10 10 10]
    out = _fill_masked_surface(B, Float32)
    @test all(isfinite, out)
    @test out[2, 2] == 10.0f0            # neighbours are all 10 (global mean is also 10 here)

    # Local-gradient case: the filled cell follows its neighbours, not the global mean.
    C = Union{Missing,Float32}[1 2 3; 2 missing 4; 3 4 100]
    oc = _fill_masked_surface(C, Float32)
    # 4-neighbours of (2,2): (1,2)=2 [lon+1 wrap], (3,2)=4 [lon-1 wrap], (2,1)=2, (2,3)=4 → mean 3
    @test oc[2, 2] == 3.0f0
    gmean = (1 + 2 + 3 + 2 + 4 + 3 + 4 + 100) / 8
    @test oc[2, 2] != Float32(gmean)     # decidedly not the global mean (~14.9)

    # Longitude wraps: a masked cell in the first column can be filled from the last.
    D = Union{Missing,Float32}[missing; 5; 5;;]   # 3×1 (lon × lat) column
    od = _fill_masked_surface(D, Float32)
    @test all(isfinite, od)
    @test od[1, 1] == 5.0f0              # neighbours (2,1)=5 and wrap (3,1)=5

    # Fully-isolated masked region (no finite neighbour ever) falls back to the
    # global finite mean.
    E = Matrix{Union{Missing,Float32}}(missing, 2, 2)
    E[1, 1] = 7.0f0
    oe = _fill_masked_surface(E, Float32)
    @test all(isfinite, oe)
    @test oe[1, 1] == 7.0f0
    @test all(oe .== 7.0f0)              # the lone finite cell propagates everywhere
end

@testset "ERA5 surface reader normalizes raw PBL fields" begin
    mktempdir() do dir
        path = joinpath(dir, "era5_surface_20211201.nc")
        NCDataset(path, "c") do ds
            ds.dim["longitude"] = 4
            ds.dim["latitude"] = 3
            ds.dim["time"] = 24
            defVar(ds, "longitude", [0.0, 90.0, 180.0, 270.0], ("longitude",))
            defVar(ds, "latitude", [90.0, 0.0, -90.0], ("latitude",))
            defVar(ds, "time", collect(0:23), ("time",))

            base = zeros(Float64, 4, 3, 24)
            for t in 1:24, j in 1:3, i in 1:4
                base[i, j, t] = 100i + 10j + t
            end
            defVar(ds, "blh", base .+ 1000, ("longitude", "latitude", "time"))
            defVar(ds, "zust", fill(0.25, 4, 3, 24), ("longitude", "latitude", "time"))
            defVar(ds, "sshf", fill(-360.0, 4, 3, 24), ("longitude", "latitude", "time");
                   attrib = ["units" => "J m**-2"])
            defVar(ds, "t2m", fill(290.0, 4, 3, 24), ("longitude", "latitude", "time"))
        end

        reader = AtmosTransport.Preprocessing.open_era5_surface_reader(dir, Date(2021, 12, 1), 4, 3)
        try
            mesh = AtmosTransport.Preprocessing.era5_surface_source_mesh(
                reader, AtmosTransport.Parameters.EARTH_RADIUS)
            @test mesh.λᶜ == [-180.0, -90.0, 0.0, 90.0]
            @test mesh.φᶠ == [-90.0, -45.0, 45.0, 90.0]
            @test isnan(mesh.Δφ)
            surface = AtmosTransport.Preprocessing.load_era5_surface_window(reader, 2, Float64)
            @test size(surface.pblh) == (4, 3)
            # Latitude flips S->N, then longitude rolls 0..360 to centered [-180,180).
            raw = [100i + 10j + 2 + 1000 for i in 1:4, j in 1:3]
            expected = circshift(raw[:, end:-1:1], (2, 0))
            @test surface.pblh == expected
            @test all(surface.ustar .== 0.25)
            @test all(surface.hflux .== 0.1)
            @test all(surface.t2m .== 290.0)
        finally
            AtmosTransport.Preprocessing.close_era5_surface_reader(reader)
        end
    end
end

@testset "ERA5 0.25-degree surface mesh uses file-coordinate snapped faces" begin
    mktempdir() do dir
        path = joinpath(dir, "era5_surface_20211201.nc")
        NCDataset(path, "c") do ds
            ds.dim["longitude"] = 1440
            ds.dim["latitude"] = 721
            defVar(ds, "longitude", collect(0.0:0.25:359.75), ("longitude",))
            defVar(ds, "latitude", collect(90.0:-0.25:-90.0), ("latitude",))
        end

        reader = AtmosTransport.Preprocessing.open_era5_surface_reader(
            dir, Date(2021, 12, 1), 1440, 721)
        try
            mesh = AtmosTransport.Preprocessing.era5_surface_source_mesh(
                reader, AtmosTransport.Parameters.EARTH_RADIUS)
            @test mesh.λᶜ[1:3] == [-180.0, -179.75, -179.5]
            @test mesh.λᶠ[1:3] == [-180.125, -179.875, -179.625]
            @test mesh.φᶠ[1:3] == [-90.0, -89.875, -89.625]
            @test mesh.φᶠ[end-2:end] == [89.625, 89.875, 90.0]
            @test isnan(mesh.Δφ)
            @test sum(AtmosTransport.Grids.cell_areas_by_latitude(mesh)) * mesh.Nx ≈
                  4π * AtmosTransport.Parameters.EARTH_RADIUS^2 rtol = 1e-12
        finally
            AtmosTransport.Preprocessing.close_era5_surface_reader(reader)
        end
    end
end

@testset "ERA5 surface reader returns latent heat flux on request" begin
    mktempdir() do dir
        path = joinpath(dir, "era5_surface_20211201.nc")
        NCDataset(path, "c") do ds
            ds.dim["longitude"] = 4
            ds.dim["latitude"] = 3
            ds.dim["time"] = 24
            defVar(ds, "longitude", [0.0, 90.0, 180.0, 270.0], ("longitude",))
            defVar(ds, "latitude", [90.0, 0.0, -90.0], ("latitude",))
            defVar(ds, "time", collect(0:23), ("time",))
            defVar(ds, "blh", fill(1000.0, 4, 3, 24), ("longitude", "latitude", "time"))
            defVar(ds, "zust", fill(0.25, 4, 3, 24), ("longitude", "latitude", "time"))
            # Both turbulent fluxes are accumulated J m⁻², downward-positive.
            defVar(ds, "sshf", fill(-360.0, 4, 3, 24), ("longitude", "latitude", "time");
                   attrib = ["units" => "J m**-2"])
            defVar(ds, "slhf", fill(-720.0, 4, 3, 24), ("longitude", "latitude", "time");
                   attrib = ["units" => "J m**-2"])
            defVar(ds, "t2m", fill(290.0, 4, 3, 24), ("longitude", "latitude", "time"))
        end

        reader = AtmosTransport.Preprocessing.open_era5_surface_reader(dir, Date(2021, 12, 1), 4, 3)
        try
            # Default path is unchanged and never requires slhf in the file.
            base = AtmosTransport.Preprocessing.load_era5_surface_window(reader, 2, Float64)
            @test !haskey(base, :lhflux)

            # Opt-in path adds upward-positive latent flux in W m⁻²:
            #   -(-720 J m⁻²) / 3600 s = +0.2 W m⁻², same convention as hflux.
            withlh = AtmosTransport.Preprocessing.load_era5_surface_window(
                reader, 2, Float64; with_latent = true)
            @test haskey(withlh, :lhflux)
            @test size(withlh.lhflux) == (4, 3)
            @test all(withlh.lhflux .== 0.2)
            @test all(withlh.hflux .== 0.1)
        finally
            AtmosTransport.Preprocessing.close_era5_surface_reader(reader)
        end
    end
end

@testset "ERA5 surface reader opens ARCO per-variable layout" begin
    mktempdir() do dir
        date = Date(2021, 12, 1)
        arco_dir = joinpath(dir, "arco", "20211201")
        mkpath(arco_dir)
        fields = (
            "boundary_layer_height" => ("blh", 1000.0, "m"),
            "friction_velocity" => ("zust", 0.25, "m s**-1"),
            "surface_sensible_heat_flux" => ("sshf", -360.0, "J m**-2"),
            "surface_latent_heat_flux" => ("slhf", -720.0, "J m**-2"),
            "2m_temperature" => ("t2m", 290.0, "K"),
        )
        for (filename, (variable, value, units)) in fields
            NCDataset(joinpath(arco_dir, "$filename.nc"), "c") do ds
                ds.dim["longitude"] = 4
                ds.dim["latitude"] = 3
                ds.dim["time"] = 24
                defVar(ds, "longitude", [0.0, 90.0, 180.0, 270.0], ("longitude",))
                defVar(ds, "latitude", [90.0, 0.0, -90.0], ("latitude",))
                defVar(ds, "time", collect(0:23), ("time",))
                defVar(ds, variable, fill(value, 4, 3, 24),
                       ("longitude", "latitude", "time"); attrib = ["units" => units])
            end
        end

        reader = AtmosTransport.Preprocessing.open_era5_surface_reader(
            dir, date, 4, 3)
        try
            surface = AtmosTransport.Preprocessing.load_era5_surface_window(
                reader, 1, Float64; with_latent = true)
            @test all(surface.pblh .== 1000.0)
            @test all(surface.ustar .== 0.25)
            @test all(surface.hflux .== 0.1)
            @test all(surface.lhflux .== 0.2)
            @test all(surface.t2m .== 290.0)
        finally
            AtmosTransport.Preprocessing.close_era5_surface_reader(reader)
        end
        @test isdir(arco_dir)
    end
end

@testset "ERA5 surface reader opens split CDS ZIP payloads" begin
    mktempdir() do dir
        inst_path = joinpath(dir, "data_stream-oper_stepType-instant.nc")
        accum_path = joinpath(dir, "data_stream-oper_stepType-accum.nc")
        zip_path = joinpath(dir, "era5_surface_20211202.nc")

        NCDataset(inst_path, "c") do ds
            ds.dim["longitude"] = 4
            ds.dim["latitude"] = 3
            ds.dim["valid_time"] = 24
            defVar(ds, "longitude", [0.0, 90.0, 180.0, 270.0], ("longitude",))
            defVar(ds, "latitude", [90.0, 0.0, -90.0], ("latitude",))
            defVar(ds, "valid_time", collect(0:23), ("valid_time",))

            base = zeros(Float64, 24, 3, 4)
            for t in 1:24, j in 1:3, i in 1:4
                base[t, j, i] = 100i + 10j + t
            end
            defVar(ds, "blh", base .+ 1000, ("valid_time", "latitude", "longitude"))
            defVar(ds, "u10", fill(3.0, 24, 3, 4), ("valid_time", "latitude", "longitude"))
            defVar(ds, "v10", fill(4.0, 24, 3, 4), ("valid_time", "latitude", "longitude"))
            defVar(ds, "t2m", fill(291.0, 24, 3, 4), ("valid_time", "latitude", "longitude"))
        end

        NCDataset(accum_path, "c") do ds
            ds.dim["longitude"] = 4
            ds.dim["latitude"] = 3
            ds.dim["valid_time"] = 24
            defVar(ds, "longitude", [0.0, 90.0, 180.0, 270.0], ("longitude",))
            defVar(ds, "latitude", [90.0, 0.0, -90.0], ("latitude",))
            defVar(ds, "valid_time", collect(0:23), ("valid_time",))
            defVar(ds, "sshf", fill(-720.0, 24, 3, 4),
                   ("valid_time", "latitude", "longitude");
                   attrib = ["units" => "J m**-2"])
        end

        run(`zip -j -q $zip_path $inst_path $accum_path`)

        reader = AtmosTransport.Preprocessing.open_era5_surface_reader(dir, Date(2021, 12, 2), 4, 3)
        try
            surface = AtmosTransport.Preprocessing.load_era5_surface_window(reader, 3, Float64)
            raw = [100i + 10j + 3 + 1000 for i in 1:4, j in 1:3]
            expected = circshift(raw[:, end:-1:1], (2, 0))
            @test surface.pblh == expected
            @test all(isapprox.(surface.ustar, sqrt(1.2e-3) * 5.0; rtol = 1e-12))
            @test all(surface.hflux .== 0.2)
            @test all(surface.t2m .== 291.0)
        finally
            AtmosTransport.Preprocessing.close_era5_surface_reader(reader)
        end
    end
end
