using Test
using Dates
using NCDatasets
using AtmosTransport
using AtmosTransport.Models.InitialConditionIO: build_surface_flux_source

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

@testset "build_surface_flux_source — native time-varying CS density" begin
    Nc = 4
    ntime = 2
    path = joinpath(mktempdir(), "native_cs_flux.nc")
    ds = NCDataset(path, "c")
    defDim(ds, "time", ntime)
    defDim(ds, "nf", 6)
    defDim(ds, "Ydim", Nc)
    defDim(ds, "Xdim", Nc)
    time_v = defVar(ds, "time", Float64, ("time",),
                    attrib = Dict("units" => "hours since 2021-12-01 00:00:00"))
    flux_v = defVar(ds, "CO2_FLUX", Float32,
                    ("Xdim", "Ydim", "nf", "time"),
                    attrib = Dict("units" => "kg CO2 m-2 s-1"))
    time_v[:] = [0.0, 1.0]
    for t in 1:ntime, p in 1:6
        flux_v[:, :, p, t] .= Float32(p + 10t)
    end
    close(ds)

    mesh = CubedSphereMesh(; FT = Float64, Nc = Nc, Hp = 1)
    vertical = HybridSigmaPressure(Float64[0, 50000, 0], Float64[1, 0.5, 0])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = Float64)
    cfg = Dict{String, Any}(
        "kind" => "cs_native",
        "file" => path,
        "variable" => "CO2_FLUX",
        "time_varying" => true,
        "species" => "co2",
    )
    src = build_surface_flux_source(
        grid, :co2_test, cfg, Float64; reference_time = DateTime(2021, 12, 1))
    @test src isa AtmosTransport.TimeVaryingSurfaceFluxSource
    @test src.times == [0.0, 3600.0]
    @test src.scheme isa AtmosTransport.StepwiseFlux

    storage_scale = AtmosTransport.Parameters.DRY_AIR_MOLAR_MASS / 44.0095e-3
    for p in 1:6, t in 1:ntime
        expected = (p + 10t) .* mesh.cell_areas .* storage_scale
        @test src.cell_mass_rate_series[p][:, :, t] ≈ expected
    end

    # A run span inside the second hour keeps only the bracketing slice.
    spanned = build_surface_flux_source(
        grid, :co2_test, cfg, Float64; reference_time = DateTime(2021, 12, 1),
        run_span = (3600.0, 5400.0))
    @test spanned.times == [3600.0]
    for p in 1:6
        @test spanned.cell_mass_rate_series[p][:, :, 1] ≈ src.cell_mass_rate_series[p][:, :, 2]
    end
    Runner = AtmosTransport.Models.DrivenRunner
    @test Runner._check_flux_run_span((0.0, 7201.0), 7201.0, (spanned,), "a.bin") === nothing
    @test_throws ArgumentError Runner._check_flux_run_span(
        (0.0, 7201.0), 7201.5, (spanned,), "a.bin")
    @test Runner._check_flux_run_span(nothing, 10800.0, (spanned,), "a.bin") === nothing
    @test Runner._check_flux_run_span((0.0, 7201.0), 10800.0, (), "a.bin") === nothing
end
