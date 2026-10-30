#!/usr/bin/env julia
#
# The `cs_native` surface-flux file, defined once in GriddedIO
# (`src/GriddedIO/cs_native_flux.jl`, design 2.4):
#
# - writer → reader round trip on C4: densities bitwise, start-stamped times,
#   `time_bnds`, CF `hours since <start> 00:00:00` (proleptic Gregorian), units
#   from the species, no fill attribute, chunks (Nc, Nc, 1, chunk_time), the
#   mesh geometry and `cell_area = mesh.cell_areas`;
# - the reader's checks (layout, Nc, fill, non-finite, time order, units);
# - streaming panel layout and precision, and the norm-based cell-area gate.
# Runtime units/storage/time-encoding coverage lives in test_native_cs_surface_flux.jl.

using Test
using Dates
using NCDatasets

using AtmosTransport
using AtmosTransport: CubedSphereMesh, LatLonMesh, HybridSigmaPressure, AtmosGrid, CPU,
                      TimeVaryingSurfaceFluxSource, StepwiseFlux
using AtmosTransport.GriddedIO: CS_NATIVE_FLUX_DIMS, CSNativeFluxWriter, open_cs_native_writer,
                                write_slices!, publish!, abandon!, read_cs_native_flux,
                                read_cs_native_flux!, cs_native_flux_info,
                                check_cell_area, TimeInterval, FluxUnits, Species, Carbon,
                                TargetSpecies, PerSquareMetre, PerSecond, canonical_units_string
using AtmosTransport.Models.InitialConditionIO: build_surface_flux_source
using AtmosTransport.Parameters: DRY_AIR_MOLAR_MASS, SPECIES_MOLAR_MASS, CARBON_MOLAR_MASS

const NC = 4

# ---- Fixtures ----

daily(start::DateTime, n) = [TimeInterval(start + Day(k - 1), start + Day(k)) for k in 1:n]

# Distinct values per (i, j, panel, slice) expose any transposition.
pattern(i, j, p, t; s = 1) = Float32(s * (i + 10j + 100p + 1000t) * 1f-10)
density_array(Nc, n; s = 1) = [pattern(i, j, p, t; s) for i in 1:Nc, j in 1:Nc, p in 1:6, t in 1:n]

# A hand-written cs_native file in the runtime layout (as the Python drivers wrote them).
function write_handmade(path; Nc = NC, T = Float32, units = "kg CO2 m-2 s-1",
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
        v[:, :, :, :] = values === nothing ? T.(density_array(Nc, n)) : values
        area === nothing || (defVar(ds, "cell_area", Float64, dims[1:ndims(area)])[ntuple(_ -> :, ndims(area))...] = area)
    end
    return path
end

# ---- Writer → reader ----

@testset "writer → reader round trip" begin
    mktempdir() do dir
        mesh = CubedSphereMesh(; FT = Float64, Nc = NC)
        intervals = daily(DateTime(2014, 1, 1), 3)
        gpp = density_array(NC, 3; s = -1)
        ter = density_array(NC, 3; s = 2)
        path = joinpath(dir, "daily.nc")
        config = "[target]\nNc = 4\n"
        w = open_cs_native_writer(path, mesh, intervals,
                                  ["GPP" => Dict("long_name" => "gross primary production"), "TER"];
                                  species = :co2, chunk_time = 1, config_text = config,
                                  attributes = Dict("title" => "test fluxes"))
        @test w isa CSNativeFluxWriter
        @test isopen(w)
        @test !isfile(path)                                   # nothing published yet
        write_slices!(w, "GPP", 1:3, gpp)
        write_slices!(w, "TER", 3:3, ter[:, :, :, 3:3])        # any order
        write_slices!(w, "TER", 1:2, view(ter, :, :, :, 1:2))  # views are accepted
        @test publish!(w) == path
        @test !isopen(w)
        @test isfile(path)
        @test_throws ArgumentError write_slices!(w, "GPP", 1:1, gpp[:, :, :, 1:1])
        @test_throws ArgumentError publish!(w)
        @test abandon!(w) === nothing                          # a closed writer: no-op
        @test isfile(path)

        for (name, data) in (("GPP", gpp), ("TER", ter))
            flux = read_cs_native_flux(path, name; mesh)
            @test flux.density == data
            @test eltype(flux.density) === Float32
            @test flux.times == [iv.start for iv in intervals]
            @test flux.units isa FluxUnits{Species, PerSquareMetre, PerSecond}
            @test flux.units.substance == Species(:co2)
        end
        # Slice selection: a range, or a function of the decoded times.
        sub = read_cs_native_flux(path, "GPP"; mesh, slices = 2:3)
        @test sub.density == gpp[:, :, :, 2:3]
        @test sub.times == [DateTime(2014, 1, 2), DateTime(2014, 1, 3)]
        byt = read_cs_native_flux(path, "GPP"; mesh,
                                  slices = times -> findfirst(==(DateTime(2014, 1, 2)), times):length(times))
        @test byt.density == sub.density && byt.times == sub.times
        @test_throws ArgumentError read_cs_native_flux(path, "GPP"; mesh, slices = 3:4)
        @test_throws ArgumentError read_cs_native_flux(path, "GPP"; mesh, slices = 2:1)

        NCDataset(path) do ds
            v = ds["GPP"]
            @test Tuple(dimnames(v)) == CS_NATIVE_FLUX_DIMS
            @test !haskey(v.attrib, "_FillValue")
            @test !haskey(v.attrib, "missing_value")
            @test v.attrib["units"] == "kg CO2 m-2 s-1" == canonical_units_string(:co2)
            @test v.attrib["sign_convention"] == "positive into the atmosphere"
            @test eltype(v.var) === Float32
            t = ds["time"]
            @test t.attrib["units"] == "hours since 2014-01-01 00:00:00"
            @test t.attrib["calendar"] == "proleptic_gregorian"
            @test t.attrib["bounds"] == "time_bnds"
            @test t.var[:] == [0.0, 24.0, 48.0]
            @test ds["time_bnds"].var[:, :] == [0.0 24.0 48.0; 24.0 48.0 72.0]
            @test ds["time_bnds"][:, :] == permutedims(hcat([iv.start for iv in intervals],
                                                            [iv.stop for iv in intervals]))
            @test ds["cell_area"][:, :, :] == repeat(Float64.(mesh.cell_areas), 1, 1, 6)
            @test ds["cubed_sphere"].attrib["semi_major_axis"] == mesh.radius
            @test haskey(ds, "lons") && haskey(ds, "corner_lats")
            @test ds.attrib["species"] == "co2"
        end
    end
end

@testset "hourly chunks, month intervals and the do-block form" begin
    mktempdir() do dir
        mesh = CubedSphereMesh(; FT = Float64, Nc = NC)
        start = DateTime(2021, 12, 31, 5)
        hourly = [TimeInterval(start + Hour(k - 1), start + Hour(k)) for k in 1:30]
        data = density_array(NC, 30)
        path = joinpath(dir, "hourly.nc")
        @test open_cs_native_writer(path, mesh, hourly, ("F",); species = :co2, chunk_time = 24) do w
            for d in 0:1                                       # one output day per write
                k = (24d + 1):min(24d + 24, 30)
                write_slices!(w, "F", k, data[:, :, :, k])
            end
        end == path
        flux = read_cs_native_flux(path, "F"; mesh)
        @test flux.density == data
        @test flux.times == [iv.start for iv in hourly]
        NCDataset(path) do ds
            @test ds["time"].attrib["units"] == "hours since 2021-12-31 00:00:00"
            @test ds["time"].var[1:2] == [5.0, 6.0]
        end
        # Calendar months (unequal lengths) and a chunk longer than the series.
        months = [TimeInterval(DateTime(2023, m, 1), DateTime(2023, m + 1, 1)) for m in 1:2]
        mpath = joinpath(dir, "monthly.nc")
        open_cs_native_writer(mpath, mesh, months, ["sf6"]; species = :sf6, chunk_time = 24) do w
            write_slices!(w, "sf6", 1:2, density_array(NC, 2))
        end
        flux = read_cs_native_flux(mpath, "sf6"; mesh)
        @test flux.times == [DateTime(2023, 1, 1), DateTime(2023, 2, 1)]
        @test flux.units.substance == Species(:sf6)
        NCDataset(mpath) do ds
            @test ds["sf6"].attrib["units"] == "kg SF6 m-2 s-1"
            @test ds["time_bnds"].var[:, :] == [0.0 744.0; 744.0 1416.0]
        end
    end
end

@testset "writer rejects non-finite values and invalid intervals" begin
    mktempdir() do dir
        mesh = CubedSphereMesh(; FT = Float64, Nc = NC)
        intervals = daily(DateTime(2014, 1, 1), 2)
        path = joinpath(dir, "flux.nc")
        w = open_cs_native_writer(path, mesh, intervals, ["F"]; species = :co2, chunk_time = 1)
        bad = density_array(NC, 2)
        for value in (NaN32, Inf32)
            bad[2, 3, 4, 2] = value
            @test_throws ArgumentError write_slices!(w, "F", 1:2, bad)
        end
        abandon!(w)
        gap = [TimeInterval(DateTime(2014, 1, 1), DateTime(2014, 1, 2)),
               TimeInterval(DateTime(2014, 1, 3), DateTime(2014, 1, 4))]
        overlap = [TimeInterval(DateTime(2014, 1, 1), DateTime(2014, 1, 3)),
                   TimeInterval(DateTime(2014, 1, 2), DateTime(2014, 1, 4))]
        for ivs in (gap, overlap, TimeInterval[])
            @test_throws ArgumentError open_cs_native_writer(path, mesh, ivs, ["F"]; species = :co2, chunk_time = 1)
        end
    end
end

# ---- Reader checks ----

@testset "reader checks" begin
    mktempdir() do dir
        mesh = CubedSphereMesh(; FT = Float64, Nc = NC)
        file(name; kw...) = write_handmade(joinpath(dir, name); kw...)
        # Units the runtime has always accepted: kg of the species (or unnamed), or kgC.
        for (units, substance) in (("kg CO2 m-2 s-1", Species(:co2)), ("kg m-2 s-1", TargetSpecies()),
                                   ("kgC m-2 s-1", Carbon()), ("kg/m2/s", TargetSpecies()),
                                   ("kg(CO2) m^-2 s^-1", Species(:co2)))
            u = read_cs_native_flux(file("u$(hash(units)).nc"; units), "F"; mesh).units
            @test u.substance == substance
        end
        # Gram, molar, per-cell, other time units and missing units are refused.
        for units in ("g C m-2 s-1", "mol m-2 s-1", "umol m-2 s-1", "kg s-1", "kg m-2 d-1",
                      "g C per month", "", "Tonnes")
            @test_throws ArgumentError read_cs_native_flux(file("bad$(hash(units)).nc"; units), "F"; mesh)
        end
        @test_throws DimensionMismatch read_cs_native_flux(file("c8.nc"; Nc = 8), "F"; mesh)
        @test_throws ArgumentError read_cs_native_flux(file("dims.nc"; dims = ("x", "y", "face", "time")), "F"; mesh)
        @test_throws ArgumentError read_cs_native_flux(file("ok.nc"), "G"; mesh)
        @test_throws ArgumentError read_cs_native_flux(joinpath(dir, "absent.nc"), "F"; mesh)
        # Time axes: decreasing or repeated stamps; `months since` never reaches CFTime.
        @test_throws ArgumentError read_cs_native_flux(file("desc.nc"; times = [48.0, 24.0, 0.0]), "F"; mesh)
        @test_throws ArgumentError read_cs_native_flux(file("rep.nc"; times = [0.0, 24.0, 24.0]), "F"; mesh)
        months = read_cs_native_flux(file("months.nc"; time_units = "months since 2014-01-01", times = [0.0, 1.0, 2.0]), "F"; mesh)
        @test months.times == [DateTime(2014, 1, 1), DateTime(2014, 2, 1), DateTime(2014, 3, 1)]
        # Fill values (through `_FillValue`) and non-finite values.
        vals = density_array(NC, 3)
        vals[1, 2, 3, 2] = -999f0
        @test_throws ArgumentError read_cs_native_flux(file("fill.nc"; fillvalue = -999, values = vals), "F"; mesh)
        vals[1, 2, 3, 2] = NaN32
        @test_throws ArgumentError read_cs_native_flux(file("nan.nc"; values = vals), "F"; mesh)
        # A NaN in a slice that is not read does not matter.
        @test read_cs_native_flux(file("nan2.nc"; values = vals), "F"; mesh, slices = 3:3).density ==
              density_array(NC, 3)[:, :, :, 3:3]
        # Float64 storage is returned as stored.
        f64 = read_cs_native_flux(file("f64.nc"; T = Float64, values = Float64.(density_array(NC, 3)) .* (1 + 1e-12)), "F"; mesh)
        @test eltype(f64.density) === Float64
        @test f64.density == Float64.(density_array(NC, 3)) .* (1 + 1e-12)
    end
end

# ---- Streaming reader ----

panels_of(FT, Nc, n) = ntuple(_ -> Array{FT}(undef, Nc, Nc, n), 6)

@testset "streaming reader: panel by panel into the caller's arrays" begin
    mktempdir() do dir
        mesh = CubedSphereMesh(; FT = Float64, Nc = NC)
        data = density_array(NC, 3; s = -2)
        path = joinpath(dir, "daily.nc")
        open_cs_native_writer(path, mesh, daily(DateTime(2014, 1, 1), 3), ["F"]; species = :co2, chunk_time = 1) do w
            write_slices!(w, "F", 1:3, data)
        end
        whole = read_cs_native_flux(path, "F"; mesh)
        # The checks without the values, then the values into the caller's arrays.
        info = cs_native_flux_info(path, "F"; mesh, slices = times -> 2:length(times))
        @test info.slices == 2:3
        @test info.times == [DateTime(2014, 1, 2), DateTime(2014, 1, 3)]
        @test info.units == whole.units
        for FT in (Float32, Float64)
            dest = panels_of(FT, NC, 3)
            got = read_cs_native_flux!(dest, path, "F"; mesh)
            @test got.slices == 1:3 && got.times == whole.times && got.units == whole.units
            @test all(p -> eltype(dest[p]) === FT && dest[p] == FT.(whole.density[:, :, p, :]), 1:6)
            sub = panels_of(FT, NC, 2)
            NCDataset(ds -> read_cs_native_flux!(sub, ds, "F", info.slices; mesh), path)
            @test all(p -> sub[p] == FT.(data[:, :, p, 2:3]), 1:6)
            # Views of a larger array are filled too.
            cube = zeros(FT, NC, NC, 6, 3)
            read_cs_native_flux!(ntuple(p -> view(cube, :, :, p, :), 6), path, "F", 1:3; mesh)
            @test cube == FT.(whole.density)
        end
        @test_throws DimensionMismatch read_cs_native_flux!(panels_of(Float32, NC, 2), path, "F", 1:3; mesh)
        @test_throws MethodError read_cs_native_flux!(ntuple(_ -> zeros(Float32, NC, NC, 3), 5), path, "F", 1:3; mesh)
        @test_throws ArgumentError read_cs_native_flux!(panels_of(Float32, NC, 3), joinpath(dir, "absent.nc"), "F", 1:3; mesh)

        # The info checks need no values; the values are checked on reading.
        vals = density_array(NC, 3)
        vals[1, 2, 3, 2] = NaN32
        nan = write_handmade(joinpath(dir, "nan.nc"); values = vals)
        @test cs_native_flux_info(nan, "F"; mesh).slices == 1:3
        @test_throws ArgumentError read_cs_native_flux!(panels_of(Float32, NC, 3), nan, "F"; mesh)
        @test_throws ArgumentError cs_native_flux_info(write_handmade(joinpath(dir, "g.nc"); units = "g C m-2 s-1"), "F"; mesh)
        vals = density_array(NC, 3)
        vals[1, 2, 3, 2] = -999f0
        fill = write_handmade(joinpath(dir, "fill.nc"); fillvalue = -999, values = vals)
        @test_throws ArgumentError read_cs_native_flux!(panels_of(Float32, NC, 3), fill, "F"; mesh)
        # A Float64 value beyond the Float32 range is not finite once read into
        # Float32 (the former loader's check, after its conversion).
        big = Float64.(density_array(NC, 3))
        big[2, 2, 5, 1] = 1e300
        f64 = write_handmade(joinpath(dir, "big.nc"); T = Float64, values = big)
        @test_throws ArgumentError read_cs_native_flux!(panels_of(Float32, NC, 3), f64, "F"; mesh)
        dest64 = panels_of(Float64, NC, 3)
        read_cs_native_flux!(dest64, f64, "F"; mesh)
        @test dest64[5][2, 2, 1] == 1e300
    end
end

# ---- Cell-area gate: today's runtime semantics (TODO(P11): element-wise) ----

@testset "cell-area gate keeps the runtime semantics" begin
    mktempdir() do dir
        mesh = CubedSphereMesh(; FT = Float64, Nc = NC)
        A = Float64.(mesh.cell_areas)
        A6 = repeat(A, 1, 1, 6)
        one_cell(rel) = (a = copy(A6); a[2, 3, 3] *= 1 + rel; a)
        cases = [
            ("identical (Nc, Nc, 6)", A6, true),
            ("identical (Nc, Nc)", A, true),
            ("uniform +5e-5", A6 .* (1 + 5e-5), true),
            ("uniform +2e-4", A6 .* (1 + 2e-4), false),
            ("one cell +1.75e-4 (Float32-era C90 polar)", one_cell(1.75e-4), true),
            ("one cell +2e-3", one_cell(2e-3), false),
            ("one cell +1e-2", one_cell(1e-2), false),
            ("uniform +1 %", A6 .* 1.01, false),
        ]
        for (label, area, passes) in cases
            path = write_handmade(joinpath(dir, "area_$(hash(label)).nc"); area)
            @testset "$label" begin
                if passes
                    @test NCDataset(ds -> check_cell_area(ds, mesh; file = path), path) === nothing
                    @test read_cs_native_flux(path, "F"; mesh).density == density_array(NC, 3)
                else
                    @test_throws ArgumentError NCDataset(ds -> check_cell_area(ds, mesh; file = path), path)
                    @test_throws ArgumentError read_cs_native_flux(path, "F"; mesh)
                end
            end
        end
        # A file without cell_area passes; a wrong shape or a NaN does not.
        @test NCDataset(ds -> check_cell_area(ds, mesh), write_handmade(joinpath(dir, "none.nc"))) === nothing
        shape = write_handmade(joinpath(dir, "shape.nc"))
        NCDataset(shape, "a") do ds
            defDim(ds, "one", 1)
            defVar(ds, "cell_area", Float64, ("Xdim", "Ydim", "one"))[:, :, :] = A6[:, :, 1:1]
        end
        @test_throws DimensionMismatch NCDataset(ds -> check_cell_area(ds, mesh), shape)
        nan = copy(A6)
        nan[1, 1, 1] = NaN
        @test_throws ArgumentError NCDataset(ds -> check_cell_area(ds, mesh), write_handmade(joinpath(dir, "nan.nc"); area = nan))
        # P2 keeps the runtime gate, which does not compare `semi_major_axis`;
        # TODO(P11): a mismatch becomes an error with the element-wise gate.
        other = CubedSphereMesh(; FT = Float64, Nc = NC, radius = 6.4e6)
        rpath = joinpath(dir, "radius.nc")
        open_cs_native_writer(rpath, mesh, daily(DateTime(2014, 1, 1), 1), ["F"]; species = :co2, chunk_time = 1) do w
            write_slices!(w, "F", 1:1, density_array(NC, 1))
        end
        NCDataset(rpath, "a") do ds
            ds["cubed_sphere"].attrib["semi_major_axis"] = other.radius
        end
        @test read_cs_native_flux(rpath, "F"; mesh).density == density_array(NC, 1)
        @test_throws ArgumentError read_cs_native_flux(rpath, "F"; mesh = other)    # areas differ
    end
end

@testset "runtime native flux refuses static and structured-grid configs" begin
    mktempdir() do dir
        path = joinpath(dir, "native.nc")
        mesh = CubedSphereMesh(; Nc = 4, Hp = 1, FT = Float64)
        grid = AtmosGrid(mesh, HybridSigmaPressure([0.0, 50000, 0], [1.0, 0.5, 0]), CPU(); FT = Float64)
        NCDataset(path, "c") do ds
            for (name, n) in (("Xdim", 4), ("Ydim", 4), ("nf", 6), ("time", 2))
                defDim(ds, name, n)
            end
            defVar(ds, "time", Float64, ("time",),
                   attrib = Dict("units" => "hours since 2014-01-01 00:00:00"))[:] = [0, 24]
            defVar(ds, "F", Float64, ("Xdim", "Ydim", "nf", "time"),
                   attrib = Dict("units" => "kg CO2 m-2 s-1"))[:, :, :, :] = fill(1e-9, 4, 4, 6, 2)
        end
        cfg = Dict{String, Any}("kind" => "cs_native", "file" => path, "variable" => "F",
                                "time_varying" => false, "species" => "co2")
        @test_throws ArgumentError build_surface_flux_source(grid, :co2, cfg, Float64)
        ll = AtmosGrid(LatLonMesh(; Nx = 8, Ny = 4), grid.vertical, CPU(); FT = Float64)
        @test_throws ArgumentError build_surface_flux_source(ll, :co2, cfg, Float64)
    end
end
