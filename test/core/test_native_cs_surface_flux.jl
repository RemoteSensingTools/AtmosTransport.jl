using Test
using Dates
using NCDatasets
using AtmosTransport
using AtmosTransport.Models.InitialConditionIO: build_surface_flux_source

@testset "Native CS time-varying surface flux" begin
    for FT in (Float32, Float64)
        mktempdir() do dir
            Nc = 4
            path = joinpath(dir, "native.nc")
            NCDataset(path, "c") do ds
                for (name, n) in (("Xdim", Nc), ("Ydim", Nc), ("nf", 6), ("time", 3))
                    defDim(ds, name, n)
                end
                defVar(ds, "time", Float64, ("time",), attrib=Dict(
                    "units" => "hours since 2014-01-01 00:00:00 UTC"))[:] = [0,24,48]
                v = defVar(ds, "NPP", Float32, ("Xdim","Ydim","nf","time"),
                           attrib=Dict("units" => "kg CO2 m-2 s-1"))
                # Unequal i/j, panel, and time patterns expose any transposition.
                for t in 1:3, p in 1:6, j in 1:Nc, i in 1:Nc
                    v[i,j,p,t] = -Float32(i + 10j + 100p + 1000t)*1f-10
                end
            end
            mesh = CubedSphereMesh(; FT, Nc, Hp=1)
            vertical = HybridSigmaPressure(FT[0,50000,0], FT[1,0.5,0])
            grid = AtmosGrid(mesh, vertical, CPU(); FT)
            cfg = Dict{String,Any}("kind"=>"cs_native", "file"=>path, "variable"=>"NPP",
                "time_varying"=>true, "species"=>"co2", "scale"=>2.0)
            source = build_surface_flux_source(grid, :custom_npp, cfg, FT;
                                               reference_time=DateTime(2014,1,2))
            @test source isa TimeVaryingSurfaceFluxSource
            @test source.scheme isa StepwiseFlux
            @test source.times == [-86400,0,86400]
            for p in 1:6, t in 1:3
                density = [-Float32(i+10j+100p+1000t)*1f-10 for i in 1:Nc, j in 1:Nc]
                expected = Float64.(density) .* Float64.(mesh.cell_areas) .* (2 * AtmosTransport.Parameters.DRY_AIR_MOLAR_MASS / 0.0440095)
                actual = source.cell_mass_rate_series[p][:,:,t]
                @test actual ≈ expected rtol=3e-7
                @test all(<(0), actual)
            end
            NCDataset(path, "a") do ds
                ds["NPP"][1,1,1,1] = NaN32
            end
            @test_throws ArgumentError build_surface_flux_source(grid, :custom_npp, cfg, FT)
            NCDataset(path, "a") do ds
                ds["NPP"][1,1,1,1] = 0
                ds["time"][:] = [48,24,0]
            end
            @test_throws ArgumentError build_surface_flux_source(grid, :custom_npp, cfg, FT)
        end
    end

    @testset "units, fill values, cell areas, and misplaced configs" begin
        FT = Float64
        Nc = 4
        mesh = CubedSphereMesh(; FT, Nc, Hp=1)
        grid = AtmosGrid(mesh, HybridSigmaPressure(FT[0,50000,0], FT[1,0.5,0]), CPU(); FT)
        function write_native(path; units = "kg CO2 m-2 s-1", area = nothing, with_fill = false)
            NCDataset(path, "c") do ds
                for (name, n) in (("Xdim", Nc), ("Ydim", Nc), ("nf", 6), ("time", 2))
                    defDim(ds, name, n)
                end
                defVar(ds, "time", Float64, ("time",),
                       attrib=Dict("units" => "hours since 2014-01-01 00:00:00"))[:] = [0, 24]
                v = defVar(ds, "F", Float64, ("Xdim","Ydim","nf","time"),
                           attrib=Dict("units" => units, "_FillValue" => -9999.0))
                v[:, :, :, :] = fill(1e-9, Nc, Nc, 6, 2)
                with_fill && (v[1, 1, 1, 1] = missing)
                area === nothing ||
                    (defVar(ds, "cell_area", Float64, ("Xdim","Ydim","nf"))[:, :, :] = area)
            end
            return path
        end
        cfg(path) = Dict{String,Any}("kind"=>"cs_native", "file"=>path, "variable"=>"F",
                                     "time_varying"=>true, "species"=>"co2")
        build(path) = build_surface_flux_source(grid, :co2, cfg(path), FT;
                                                reference_time = DateTime(2014, 1, 1))
        mktempdir() do dir
            kgc = build(write_native(joinpath(dir, "kgc.nc"); units = "kgC m-2 s-1"))
            kgco2 = build(write_native(joinpath(dir, "kgco2.nc")))
            @test kgc.cell_mass_rate_series[1] ≈ kgco2.cell_mass_rate_series[1] .* (44.0095 / 12.0107)
            for bad in ("mol m-2 s-1", "g C m-2 s-1", "umol m-2 s-1")
                @test_throws ArgumentError build(write_native(joinpath(dir, "u$(hash(bad)).nc"); units = bad))
            end
            @test_throws ArgumentError build(write_native(joinpath(dir, "fill.nc"); with_fill = true))
            areas = repeat(Float64.(mesh.cell_areas), 1, 1, 6)
            @test build(write_native(joinpath(dir, "area_ok.nc"); area = areas)) isa TimeVaryingSurfaceFluxSource
            @test_throws ArgumentError build(write_native(joinpath(dir, "area_bad.nc"); area = 1.01 .* areas))
            static = merge(cfg(joinpath(dir, "kgco2.nc")), Dict{String,Any}("time_varying" => false))
            @test_throws ArgumentError build_surface_flux_source(grid, :co2, static, FT)
            ll = AtmosGrid(LatLonMesh(; FT, Nx = 8, Ny = 4), HybridSigmaPressure(FT[0,50000,0], FT[1,0.5,0]), CPU(); FT)
            @test_throws ArgumentError build_surface_flux_source(ll, :co2, static, FT)
        end
    end
end

using AtmosTransport.GriddedIO: TimeInterval, open_cs_native_writer, write_slices!
using AtmosTransport.Parameters: DRY_AIR_MOLAR_MASS, SPECIES_MOLAR_MASS, CARBON_MOLAR_MASS

const NATIVE_NC = 4
# ---- Fixtures ----

native_test_grid(FT; Nc = NATIVE_NC) = AtmosGrid(CubedSphereMesh(; FT, Nc, Hp = 1),
                                 HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0]), CPU(); FT)

native_test_daily(start::DateTime, n) = [TimeInterval(start + Day(k - 1), start + Day(k)) for k in 1:n]

# Distinct values per (i, j, panel, slice) expose any transposition.
native_test_pattern(i, j, p, t; s = 1) = Float32(s * (i + 10j + 100p + 1000t) * 1f-10)
native_test_density(Nc, n; s = 1) = [native_test_pattern(i, j, p, t; s) for i in 1:Nc, j in 1:Nc, p in 1:6, t in 1:n]

# A hand-written cs_native file in the runtime layout (as the Python drivers wrote them).
function native_test_file(path; Nc = NATIVE_NC, T = Float32, units = "kg CO2 m-2 s-1",
                        time_units = "hours since 2014-01-01 00:00:00 UTC", times = [0.0, 24.0, 48.0],
                        area = nothing, fillvalue = nothing, dims = ("Xdim", "Ydim", "nf", "time"),
                        values = nothing)
    n = length(times)
    NCDataset(path, "c") do ds
        for (name, len) in zip(dims, (Nc, Nc, 6, n))
            defDim(ds, name, len)
        end
        defVar(ds, dims[4], Float64, (dims[4],), attrib = Dict("units" => time_units))[:] = times
        attrib = Dict{String, Any}("units" => units)
        fillvalue === nothing || (attrib["_FillValue"] = T(fillvalue))
        v = defVar(ds, "F", T, dims; attrib)
        v[:, :, :, :] = values === nothing ? T.(native_test_density(Nc, n)) : values
        area === nothing || (defVar(ds, "cell_area", Float64, dims[1:ndims(area)])[ntuple(_ -> :, ndims(area))...] = area)
    end
    return path
end

native_test_cfg(path; scale = 1.0) =
    Dict{String, Any}("kind" => "cs_native", "file" => path, "variable" => "F",
                      "time_varying" => true, "species" => "co2", "scale" => scale)

@testset "native loader units, storage precision and time encodings" begin
    mktempdir() do dir
        files = [
            native_test_file(joinpath(dir, "co2_f32.nc")),
            native_test_file(joinpath(dir, "kgc_f32.nc"); units = "kgC m-2 s-1"),
            native_test_file(joinpath(dir, "plain_f64.nc"); T = Float64, units = "kg m-2 s-1",
                           values = Float64.(native_test_density(NATIVE_NC, 3)) .* (1 + 1e-9)),
            native_test_file(joinpath(dir, "days.nc"); time_units = "days since 2013-12-31", times = [1.0, 2.0, 3.0]),
            native_test_file(joinpath(dir, "seconds.nc"); time_units = "seconds since 2014-01-01T00:00:00",
                           times = [0.0, 86400.0, 172800.0],
                           area = repeat(Float64.(CubedSphereMesh(; FT = Float64, Nc = NATIVE_NC).cell_areas), 1, 1, 6)),
        ]
        # A file from the writer, as the CATRINE protocol script now writes them.
        mesh64 = CubedSphereMesh(; FT = Float64, Nc = NATIVE_NC)
        written = joinpath(dir, "written.nc")
        open_cs_native_writer(written, mesh64, native_test_daily(DateTime(2014, 1, 1), 3), ["F"]; species = :co2, chunk_time = 1) do w
            write_slices!(w, "F", 1:3, native_test_density(NATIVE_NC, 3; s = -3))
        end
        push!(files, written)
        for FT in (Float32, Float64), path in files
            grid = native_test_grid(FT)
            mesh = grid.horizontal
            cfg = native_test_cfg(path; scale = 2.5)
            for (reference_time, run_span) in ((DateTime(2014, 1, 2), nothing),
                                               (DateTime(2014, 1, 2), (0.0, 3600.0)),
                                               (DateTime(2014, 1, 1, 12), (0.0, 86400.0 + 1.0)),
                                               (DateTime(2013, 12, 1), (0.0, 86400.0)))
                # A run outside the file's time coverage holds an end slice
                # only on request.
                covered = reference_time >= DateTime(2014, 1, 1)
                covered || @test_throws ArgumentError build_surface_flux_source(
                    grid, :co2, cfg, FT; reference_time, run_span)
                cfg_run = covered ? cfg : merge(cfg, Dict{String, Any}("hold_outside_coverage" => true))
                source = build_surface_flux_source(grid, :co2, cfg_run, FT; reference_time, run_span)
                # Expected slices for these explicit daily fixtures and run spans.
                kr = run_span === nothing ? (1:3) : reference_time == DateTime(2014, 1, 2) ?
                     (2:3) : reference_time == DateTime(2014, 1, 1, 12) ? (1:3) : (1:1)
                stamps = [DateTime(2014, 1, d) for d in 1:3]
                @test source.times == [Dates.value(t - reference_time) / 1000.0 for t in stamps[kr]]
                units_factor = basename(path) == "kgc_f32.nc" ?
                               SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS : 1.0
                scale = FT(units_factor) * FT(2.5) * FT(DRY_AIR_MOLAR_MASS / SPECIES_MOLAR_MASS.co2)
                for panel in 1:6
                    stored = NCDataset(ds -> Array(NCDatasets.variable(ds, "F")[:, :, panel, kr]), path)
                    expected = FT.(stored) .* reshape(FT.(mesh.cell_areas), NATIVE_NC, NATIVE_NC, 1) .* scale
                    @test reinterpret((FT === Float32 ? UInt32 : UInt64), vec(source.cell_mass_rate_series[panel])) ==
                          reinterpret((FT === Float32 ? UInt32 : UInt64), vec(expected))
                end
                @test source.scheme isa StepwiseFlux
            end
        end
    end
end
