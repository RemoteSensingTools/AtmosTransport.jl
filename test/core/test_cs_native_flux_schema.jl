#!/usr/bin/env julia
#
# The `cs_native` surface-flux file, defined once in GriddedIO
# (`src/GriddedIO/cs_native_flux.jl`, design 2.4):
#
# - writer → reader round trip on C4: densities bitwise, start-stamped times,
#   `time_bnds`, CF `hours since <start> 00:00:00` (proleptic Gregorian), units
#   from the species, no fill attribute, chunks (Nc, Nc, 1, chunk_time), the
#   mesh geometry and `cell_area = mesh.cell_areas`, provenance and config SHA-1;
# - atomic publication: nothing at the path until `publish!`; an incomplete,
#   non-finite or failed write leaves the previous file and no staging file;
# - the reader's checks (layout, Nc, fill, non-finite, time order, units);
# - the streaming reader (`cs_native_flux_info`, `read_cs_native_flux!`) fills
#   the caller's arrays panel by panel, converting each value once, and
#   allocates one panel-slab beyond its checks (the runtime holds the series
#   once, not the whole cube twice);
# - the cell-area gate keeps the runtime's norm-based semantics (rtol 1e-4 per
#   panel) in P2, compared outcome by outcome with the frozen runtime gate;
# - the migrated runtime loader (`kind = "cs_native"`) gives the frozen loader's
#   source bit for bit (Float32 and Float64 runs, Float32 and Float64 files,
#   kg CO2 / kgC / unnamed units, run spans), also on files from the writer.

using Test
using Dates
using NCDatasets
using SHA: sha1

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

# ---- Frozen runtime loader (src/Models/initial_conditions/surface_flux_native.jl
#      and its helpers in surface_flux.jl before P2a) ----

function frozen_ic_find_coord(ds, candidates)
    for name in candidates
        haskey(ds, name) && return name
    end
    return nothing
end

function frozen_parse_cf_time_origin(s::AbstractString)
    ss = replace(strip(String(s)), "T" => " ")
    for fmt in (dateformat"y-m-d H:M:S", dateformat"y-m-d H:M", dateformat"y-m-d")
        try
            return DateTime(ss, fmt)
        catch
        end
    end
    throw(ArgumentError("cannot parse time origin '$(s)'"))
end

function frozen_times_seconds(time_vals, units::AbstractString, reference_time::DateTime)
    n = length(time_vals)
    times_sec = Vector{Float64}(undef, n)
    if first(time_vals) isa Real
        m = match(r"(\w+)\s+since\s+(.+)", strip(String(units)))
        unit_word = lowercase(m.captures[1])
        per_unit_sec = unit_word in ("second", "seconds", "sec", "s") ? 1.0 :
                       unit_word in ("minute", "minutes", "min")       ? 60.0 :
                       unit_word in ("hour", "hours", "hr", "h")        ? 3600.0 :
                       unit_word in ("day", "days", "d")                ? 86400.0 : error("unit")
        origin = frozen_parse_cf_time_origin(strip(m.captures[2]))
        ref_offset_sec = Dates.value(origin - reference_time) / 1000.0
        for k in 1:n
            times_sec[k] = ref_offset_sec + Float64(time_vals[k]) * per_unit_sec
        end
    else
        for k in 1:n
            times_sec[k] = Dates.value(DateTime(time_vals[k]) - reference_time) / 1000.0
        end
    end
    return times_sec
end

const FROZEN_KG_PER_M2_S = r"^kg(\(?[a-z][a-z0-9]*\)?)?(/m2/s|m-2s-1|/m2s|/s/m2|s-1m-2)$"
frozen_normalize(units) = lowercase(replace(strip(String(units)), " " => "", "^" => "", "²" => "2"))
function frozen_species_scale(units_norm)
    m = match(FROZEN_KG_PER_M2_S, units_norm)
    m === nothing && throw(ArgumentError("unsupported surface-flux units '$(units_norm)'"))
    species = m.captures[1] === nothing ? "" : strip(m.captures[1], ['(', ')'])
    return species == "c" ? SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS : 1.0
end

function frozen_needed_slice_window(times, run_span)
    t0, t1 = run_span
    sorted = sort(times)
    i_lo = searchsortedlast(sorted, t0)
    i_hi = searchsortedfirst(sorted, t1)
    return sorted[max(i_lo, 1)], sorted[min(i_hi, length(sorted))]
end

function frozen_check_cell_area(ds, mesh, file)
    haskey(ds, "cell_area") || return nothing
    file_area = Float64.(nomissing(Array(ds["cell_area"]), NaN))
    mesh_area = Float64.(mesh.cell_areas)
    Nc = mesh.Nc
    panels = if size(file_area) == (Nc, Nc)
        (file_area,)
    elseif size(file_area) == (Nc, Nc, 6)
        ntuple(p -> view(file_area, :, :, p), 6)
    else
        throw(DimensionMismatch("cell_area shape"))
    end
    for (p, a) in enumerate(panels)
        isapprox(a, mesh_area; rtol = 1e-4) || throw(ArgumentError("cell_area panel $(p)"))
    end
    return nothing
end

# The storage factor of a CO2 tracer before P2c (`DRY_AIR_MOLAR_MASS /
# molar_mass_kg_mol` with the configs' 0.0440095); P2c's `species = "co2"` gives
# `DRY_AIR_MOLAR_MASS / SPECIES_MOLAR_MASS.co2`, the same Float64.
const FROZEN_CO2_STORAGE_SCALE = DRY_AIR_MOLAR_MASS / 0.0440095

# (panels_series, times_sec) of the pre-P2a loader for a CO2 tracer.
function frozen_native_source(mesh, cfg, ::Type{FT}, reference_time; run_span = nothing) where FT
    file, variable = cfg["file"], cfg["variable"]
    ds = NCDataset(file)
    try
        time_var = frozen_ic_find_coord(ds, ["time", "t"])
        raw_var = ds[variable]
        Nc = mesh.Nc
        ntime = size(raw_var, 4)
        time_units = String(get(ds[time_var].attrib, "units", ""))
        times_all = frozen_times_seconds(ds[time_var][:], time_units, reference_time)
        issorted(times_all) || throw(ArgumentError("times"))
        kr = if run_span === nothing
            1:ntime
        else
            t_lo, t_hi = frozen_needed_slice_window(times_all, run_span)
            searchsortedfirst(times_all, t_lo):searchsortedlast(times_all, t_hi)
        end
        times_sec = times_all[kr]
        units_norm = frozen_normalize(get(raw_var.attrib, "units", ""))
        species_scale = FT(frozen_species_scale(units_norm))
        scale = species_scale * FT(get(cfg, "scale", 1.0)) *
                FT(FROZEN_CO2_STORAGE_SCALE)
        frozen_check_cell_area(ds, mesh, file)
        area = reshape(FT.(mesh.cell_areas), Nc, Nc, 1)
        panels_series = ntuple(p -> begin
            panel = raw_var[:, :, p, kr]
            any(ismissing, panel) && throw(ArgumentError("fill"))
            density = FT.(panel)
            all(isfinite, density) || throw(ArgumentError("non-finite"))
            density .* area .* scale
        end, 6)
        return panels_series, times_sec
    finally
        close(ds)
    end
end

# ---- Fixtures ----

cs_grid(FT; Nc = NC) = AtmosGrid(CubedSphereMesh(; FT, Nc, Hp = 1),
                                 HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0]), CPU(); FT)

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

native_cfg(path; scale = 1.0) =
    Dict{String, Any}("kind" => "cs_native", "file" => path, "variable" => "F",
                      "time_varying" => true, "species" => "co2", "scale" => scale)

staging_files(dir) = filter(f -> occursin(".tmp-", f), readdir(dir))

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
        @test length(staging_files(dir)) == 1
        write_slices!(w, "GPP", 1:3, gpp)
        write_slices!(w, "TER", 3:3, ter[:, :, :, 3:3])        # any order
        write_slices!(w, "TER", 1:2, view(ter, :, :, :, 1:2))  # views are accepted
        @test publish!(w) == path
        @test !isopen(w)
        @test isfile(path)
        @test isempty(staging_files(dir))
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
            @test v.attrib["grid_mapping"] == "cubed_sphere"
            @test v.attrib["coordinates"] == "lons lats"
            @test v.attrib["long_name"] == "gross primary production"
            @test eltype(v.var) === Float32
            @test NCDatasets.chunking(v.var) == (:chunked, (NC, NC, 1, 1))
            @test NCDatasets.deflate(v.var) == (true, true, 1)
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
            @test ds.attrib["Conventions"] == "CF-1.8"
            @test ds.attrib["species"] == "co2"
            @test ds.attrib["framework"] == "AtmosTransport.jl"
            @test ds.attrib["config_text"] == config
            @test ds.attrib["config_sha1"] == bytes2hex(sha1(config))
            @test ds.attrib["title"] == "test fluxes"
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
            @test NCDatasets.chunking(ds["F"].var) == (:chunked, (NC, NC, 1, 24))
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
            @test NCDatasets.chunking(ds["sf6"].var) == (:chunked, (NC, NC, 1, 2))
            @test ds["sf6"].attrib["units"] == "kg SF6 m-2 s-1"
            @test ds["time_bnds"].var[:, :] == [0.0 744.0; 744.0 1416.0]
        end
    end
end

@testset "atomic publication and writer errors" begin
    mktempdir() do dir
        mesh = CubedSphereMesh(; FT = Float64, Nc = NC)
        intervals = daily(DateTime(2014, 1, 1), 2)
        good = density_array(NC, 2)
        path = joinpath(dir, "flux.nc")
        write(path, "previous file")
        open_w() = open_cs_native_writer(path, mesh, intervals, ["F"]; species = :co2, chunk_time = 1)

        # A failure inside the do-block abandons the file; the previous file stays.
        @test_throws ErrorException open_cs_native_writer(path, mesh, intervals, ["F"];
                                                           species = :co2, chunk_time = 1) do w
            write_slices!(w, "F", 1:1, good[:, :, :, 1:1])
            error("builder failed")
        end
        @test read(path, String) == "previous file"
        @test isempty(staging_files(dir))

        # An incomplete file is never published.
        w = open_w()
        write_slices!(w, "F", 1:1, good[:, :, :, 1:1])
        @test_throws ArgumentError publish!(w)
        @test !isopen(w)
        @test read(path, String) == "previous file"
        @test isempty(staging_files(dir))

        # Non-finite values, wrong shapes, unknown series, out-of-range slices.
        w = open_w()
        bad = copy(good)
        bad[2, 3, 4, 2] = NaN32
        @test_throws ArgumentError write_slices!(w, "F", 1:2, bad)
        bad[2, 3, 4, 2] = Inf32
        @test_throws ArgumentError write_slices!(w, "F", 1:2, bad)
        @test_throws DimensionMismatch write_slices!(w, "F", 1:2, good[:, :, :, 1:1])
        @test_throws ArgumentError write_slices!(w, "G", 1:2, good)
        @test_throws ArgumentError write_slices!(w, "F", 2:3, good)
        @test_throws MethodError write_slices!(w, "F", 1:2, Float64.(good))   # round to Float32 first
        abandon!(w)
        @test !isopen(w)
        @test isempty(staging_files(dir))
        @test read(path, String) == "previous file"

        # Arguments rejected before anything is created.
        gap = [TimeInterval(DateTime(2014, 1, 1), DateTime(2014, 1, 2)),
               TimeInterval(DateTime(2014, 1, 3), DateTime(2014, 1, 4))]
        overlap = [TimeInterval(DateTime(2014, 1, 1), DateTime(2014, 1, 3)),
                   TimeInterval(DateTime(2014, 1, 2), DateTime(2014, 1, 4))]
        for ivs in (gap, overlap, TimeInterval[])
            @test_throws ArgumentError open_cs_native_writer(path, mesh, ivs, ["F"]; species = :co2, chunk_time = 1)
        end
        @test_throws ArgumentError open_cs_native_writer(path, mesh, intervals, ["F"]; species = :ch4, chunk_time = 1)
        @test_throws ArgumentError open_cs_native_writer(path, mesh, intervals, ["F"]; species = :co2, chunk_time = 0)
        @test_throws ArgumentError open_cs_native_writer(path, mesh, intervals, ["F"]; species = :co2, chunk_time = 1, deflate = 10)
        @test_throws ArgumentError open_cs_native_writer(path, mesh, intervals, String[]; species = :co2, chunk_time = 1)
        @test_throws ArgumentError open_cs_native_writer(path, mesh, intervals, ["F", "F"]; species = :co2, chunk_time = 1)
        for reserved in ("units", "_FillValue", "missing_value", "scale_factor", "sign_convention")
            @test_throws ArgumentError open_cs_native_writer(path, mesh, intervals, ["F" => Dict(reserved => 1)];
                                                             species = :co2, chunk_time = 1)
        end
        for name in ("time", "cell_area", "lons")
            @test_throws ArgumentError open_cs_native_writer(path, mesh, intervals, [name]; species = :co2, chunk_time = 1)
        end
        for key in ("species", "Conventions", "framework", "cs_definition")
            @test_throws ArgumentError open_cs_native_writer(path, mesh, intervals, ["F"]; species = :co2,
                                                             chunk_time = 1, attributes = Dict(key => "x"))
        end
        @test isempty(staging_files(dir))
        @test read(path, String) == "previous file"

        # A relative path is fixed at opening, whatever the working directory at publication.
        w = cd(() -> open_cs_native_writer("relative.nc", mesh, intervals, ["F"]; species = :co2, chunk_time = 1), dir)
        write_slices!(w, "F", 1:2, good)
        @test cd(() -> publish!(w), mktempdir()) == joinpath(dir, "relative.nc")
        @test read_cs_native_flux(joinpath(dir, "relative.nc"), "F"; mesh).density == good
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

# Bytes allocated by one call (after a warm-up call), measured in a function so
# that no closure boxing is counted.
function stream_bytes(dest, ds, mesh, kr)
    read_cs_native_flux!(dest, ds, "F", kr; mesh)
    return @allocated read_cs_native_flux!(dest, ds, "F", kr; mesh)
end
function info_bytes(ds, mesh, kr)
    cs_native_flux_info(ds, "F"; mesh, slices = kr)
    return @allocated cs_native_flux_info(ds, "F"; mesh, slices = kr)
end
function builder_bytes(grid, cfg, ::Type{FT}) where FT
    build_surface_flux_source(grid, :co2, cfg, FT; reference_time = DateTime(2014, 1, 1))
    return @allocated build_surface_flux_source(grid, :co2, cfg, FT; reference_time = DateTime(2014, 1, 1))
end

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

@testset "streaming reader allocates one panel-slab" begin
    mktempdir() do dir
        Nc, n = 24, 96                                   # one panel-slab: 221 kB in Float32
        mesh = CubedSphereMesh(; FT = Float64, Nc)
        path = joinpath(dir, "hourly.nc")
        hourly = [TimeInterval(DateTime(2014, 1, 1) + Hour(k - 1), DateTime(2014, 1, 1) + Hour(k)) for k in 1:n]
        open_cs_native_writer(path, mesh, hourly, ["F"]; species = :co2, chunk_time = 24) do w
            write_slices!(w, "F", 1:n, density_array(Nc, n))
        end
        slab = Nc * Nc * n * sizeof(Float32)
        allowance = 32 * 1024                            # per-call overhead, independent of the slab
        NCDataset(path) do ds
            checks = info_bytes(ds, mesh, 1:n)
            for FT in (Float32, Float64)
                extra = stream_bytes(panels_of(FT, Nc, n), ds, mesh, 1:n) - checks
                @test extra <= slab + allowance          # the whole cube would be ≥ 6 slabs
            end
        end
        # The runtime builder holds the series once: its own arrays, plus one
        # read slab and the checks (run by both calls), never a second cube.
        checks = NCDataset(ds -> info_bytes(ds, mesh, 1:n), path)
        cfg = Dict{String, Any}("kind" => "cs_native", "file" => path, "variable" => "F",
                                "time_varying" => true, "species" => "co2")
        for FT in (Float32, Float64)
            series = 6 * Nc * Nc * n * sizeof(FT)
            @test builder_bytes(cs_grid(FT; Nc), cfg, FT) - series <= slab + 2 * checks + 4 * allowance
        end
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
            frozen_passes = NCDataset(ds -> (try frozen_check_cell_area(ds, mesh, path); true
                                              catch err; err isa ArgumentError || rethrow(); false end), path)
            @testset "$label" begin
                @test frozen_passes == passes
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

# ---- The migrated runtime loader equals the frozen one, bit for bit ----

function same_source(source, frozen)
    panels, times = frozen
    source isa TimeVaryingSurfaceFluxSource || return false
    source.times == times && all(t -> t isa Float64, source.times) || return false
    return all(p -> eltype(source.cell_mass_rate_series[p]) === eltype(panels[p]) &&
                    isequal(source.cell_mass_rate_series[p], panels[p]), 1:6)
end

@testset "runtime loader is bit-identical to the frozen loader" begin
    mktempdir() do dir
        files = [
            write_handmade(joinpath(dir, "co2_f32.nc")),
            write_handmade(joinpath(dir, "kgc_f32.nc"); units = "kgC m-2 s-1"),
            write_handmade(joinpath(dir, "plain_f64.nc"); T = Float64, units = "kg m-2 s-1",
                           values = Float64.(density_array(NC, 3)) .* (1 + 1e-9)),
            write_handmade(joinpath(dir, "days.nc"); time_units = "days since 2013-12-31", times = [1.0, 2.0, 3.0]),
            write_handmade(joinpath(dir, "seconds.nc"); time_units = "seconds since 2014-01-01T00:00:00",
                           times = [0.0, 86400.0, 172800.0],
                           area = repeat(Float64.(CubedSphereMesh(; FT = Float64, Nc = NC).cell_areas), 1, 1, 6)),
        ]
        # A file from the writer, as the CATRINE protocol script now writes them.
        mesh64 = CubedSphereMesh(; FT = Float64, Nc = NC)
        written = joinpath(dir, "written.nc")
        open_cs_native_writer(written, mesh64, daily(DateTime(2014, 1, 1), 3), ["F"]; species = :co2, chunk_time = 1) do w
            write_slices!(w, "F", 1:3, density_array(NC, 3; s = -3))
        end
        push!(files, written)
        for FT in (Float32, Float64), path in files
            grid = cs_grid(FT)
            mesh = grid.horizontal
            cfg = native_cfg(path; scale = 2.5)
            for (reference_time, run_span) in ((DateTime(2014, 1, 2), nothing),
                                               (DateTime(2014, 1, 2), (0.0, 3600.0)),
                                               (DateTime(2014, 1, 1, 12), (0.0, 86400.0 + 1.0)),
                                               (DateTime(2013, 12, 1), (0.0, 86400.0)))
                # A run outside the file's time coverage holds an end slice, as
                # the frozen loader did, only on request.
                covered = reference_time >= DateTime(2014, 1, 1)
                covered || @test_throws ArgumentError build_surface_flux_source(
                    grid, :co2, cfg, FT; reference_time, run_span)
                cfg_run = covered ? cfg : merge(cfg, Dict{String, Any}("hold_outside_coverage" => true))
                source = build_surface_flux_source(grid, :co2, cfg_run, FT; reference_time, run_span)
                frozen = frozen_native_source(mesh, cfg, FT, reference_time; run_span)
                @test same_source(source, frozen)
                @test source.scheme isa StepwiseFlux
            end
        end
    end
end
