"""
`AtmosTransport.Analysis.GEOSChemCATRINEArchive` (`src/Analysis/geoschem_archive.jl`)
and the `--model geoschem` mode of `scripts/diagnostics/catrine_true_mass_balance.jl`,
on a tiny synthetic GEOS-Chem CATRINE archive written into `mktempdir()`
(2×2 cells per panel, 3 levels, GEOS-Chem's variable names and attributes).

Asserts:
- `gc_path` prefers `YYYY/MM/<name>` and falls back to the flat name;
  `gc_times` is the inclusive output grid;
- `global_totals` equals a direct Float64 computation from the fixture values:
  Σ Met_AD, Σ χ·Met_AD and (Σ E·Met_AREAM2)·M_air/M_species with GEOS-Chem's
  molar masses; a `_FillValue` cell counts as zero; the threaded batch equals
  the single-time calls bit for bit;
- a garbage file, a missing file and a file without one variable read as
  `nothing` and are listed by `unreadable_files` in time order with a reason;
  a non-file error inside the reader propagates;
- the constructor rejects empty or repeated tracers, a non-positive interval,
  molar mass or half-life; `CATRINE_GEOSCHEM_TRACERS` takes its molar masses
  and the Rn-222 half-life from `Parameters`;
- the script writes the csv columns of the Python script and of the
  AtmosTransport mode, and its rows equal an independent port of the Python
  algorithm (rate over the interval before each file, the next readable rate
  over an unreadable file's interval, rows only for readable files, t₀ at the
  first readable file, the exact decay update for Rn-222);
- `exact_decay` matches the closed form of B' = E − λB; the summary lines
  print r, r/∫E and r/air; misplaced command-line options are rejected.
"""

using Test
using Dates
using Logging
using NCDatasets
using Printf
using TOML
using AtmosTransport
using AtmosTransport: HybridSigmaPressure
using AtmosTransport.Analysis: GEOSChemTracer, CATRINE_GEOSCHEM_TRACERS, GEOSChemCATRINEArchive,
                               gc_path, gc_times, with_gc_file, global_totals, unreadable_files
using AtmosTransport.Parameters: DRY_AIR_MOLAR_MASS, GEOSCHEM_SPECIES_MOLAR_MASS, RN222_HALF_LIFE,
                                 SPECIES_MOLAR_MASS, EARTH_RADIUS
const MD = AtmosTransport.MetDrivers

const SCRIPT = joinpath(@__DIR__, "..", "..", "scripts", "diagnostics", "catrine_true_mass_balance.jl")
const MB = Module()
Base.include(MB, SCRIPT)
mb_call(f::Symbol, args...; kw...) = Base.invokelatest(getproperty(MB, f), args...; kw...)

const NX, NF, NLEV = 2, 6, 3
const FILL = 1.0f15

gc_name(t) = "GEOSChem.CATRINE_inst." * Dates.format(t, dateformat"yyyymmdd_HHMM") * "z.nc4"

# Fixture fields of output number k (Float32, as GEOS-Chem writes them).
function fixture_fields(k)
    air = Float32[1.0e3 * (1 + 0.1i + 0.01j + 0.001p) * (1 + 0.5l) + k for i in 1:NX, j in 1:NX, p in 1:NF, l in 1:NLEV]
    area = Float32[1.0e6 * (1 + 0.2i + 0.03j + 0.004p) for i in 1:NX, j in 1:NX, p in 1:NF]
    conc = Dict(tr.concentration => Float32[(1 + 0.05k) * 1e-4 * (s + 0.1i + 0.01j + 0.001p + 0.0001l)
                                            for i in 1:NX, j in 1:NX, p in 1:NF, l in 1:NLEV]
                for (s, tr) in enumerate(CATRINE_GEOSCHEM_TRACERS))
    emis = Dict(tr.emission => Float32[(1 + 0.3k) * 1e-9 * (s + 0.2i - 0.05j + 0.01p)
                                       for i in 1:NX, j in 1:NX, p in 1:NF]
                for (s, tr) in enumerate(CATRINE_GEOSCHEM_TRACERS))
    return (; air, area, conc, emis)
end

# One GEOS-Chem-like file: (Xdim, Ydim, nf[, lev], time) Float32 with the
# _FillValue, missing_value, scale_factor and add_offset attributes GEOS-Chem
# writes. `omit` leaves a variable out; `fill_cell` stores the fill value in one
# Met_AD cell; `no_record` leaves the (unlimited) time axis empty.
function write_gc_file(path, t, f; omit = (), fill_cell = false, no_record = false)
    mkpath(dirname(path))
    attrib = ["missing_value" => FILL, "scale_factor" => 1.0f0, "add_offset" => 0.0f0]
    NCDataset(path, "c") do ds
        defDim(ds, "Xdim", NX); defDim(ds, "Ydim", NX); defDim(ds, "nf", NF)
        defDim(ds, "lev", NLEV); defDim(ds, "time", no_record ? Inf : 1)
        tv = defVar(ds, "time", Float64, ("time",), attrib = ["units" => "minutes since $(t)"])
        no_record || (tv.var[1] = 0.0)
        function put(name, A)
            name in omit && return
            dims = ndims(A) == 4 ? ("Xdim", "Ydim", "nf", "lev", "time") : ("Xdim", "Ydim", "nf", "time")
            v = defVar(ds, name, Float32, dims; fillvalue = FILL, attrib)
            no_record || (v.var[ntuple(_ -> Colon(), ndims(A))..., 1] = A)
        end
        put("Met_AD", f.air)
        put("Met_AREAM2", f.area)
        foreach(kv -> put(kv...), f.conc)
        foreach(kv -> put(kv...), f.emis)
        fill_cell && (ds["Met_AD"].var[1, 1, 1, 1, 1] = FILL)
    end
    return path
end

# Direct Float64 totals of fixture fields, independent of the module's code path.
function direct_totals(f; zero_first_air_cell = false)
    air = Float64.(f.air)
    zero_first_air_cell && (air[1, 1, 1, 1] = 0.0)
    area = Float64.(f.area)
    burden = [sum(Float64.(f.conc[tr.concentration]) .* air) for tr in CATRINE_GEOSCHEM_TRACERS]
    rate = [sum(Float64.(f.emis[tr.emission]) .* area) * DRY_AIR_MOLAR_MASS / tr.molar_mass
            for tr in CATRINE_GEOSCHEM_TRACERS]
    return (; dry_air = sum(air), burden, rate)
end


# ---------------------------------------------------------------------------
# A tiny AtmosTransport run for the default (`atmostransport`) mode: two daily
# C3 binaries of eight 3-hour windows with zero mass fluxes (transport moves
# nothing) and 1e16 kg of dry air per cell and level, a daily `cs_native` flux
# file, a stable `co2` tracer and a decaying `rn222` tracer, 6-hourly snapshots.
# ---------------------------------------------------------------------------
const RUN_NC, RUN_NZ, RUN_AIR = 3, 2, 1e16

function write_run_binary(path)
    vc = HybridSigmaPressure(fill(100.0, RUN_NZ + 1), collect(range(0.0, 1.0; length = RUN_NZ + 1)))
    writer = MD.open_streaming_cs_transport_binary(path, RUN_NC, 6, RUN_NZ, 8, vc; planet_radius = EARTH_RADIUS,
                                                   FT = Float64, dt_met_seconds = 10800.0,
                                                   steps_per_window = 2, mass_basis = :dry,
                                                   surface_pressure_basis = TotalSurfacePressure())
    try
        window = (; m = ntuple(_ -> fill(RUN_AIR, RUN_NC, RUN_NC, RUN_NZ), 6),
                  am = ntuple(_ -> zeros(RUN_NC + 1, RUN_NC, RUN_NZ), 6),
                  bm = ntuple(_ -> zeros(RUN_NC, RUN_NC + 1, RUN_NZ), 6),
                  cm = ntuple(_ -> zeros(RUN_NC, RUN_NC, RUN_NZ + 1), 6),
                  ps = ntuple(_ -> fill(1e5, RUN_NC, RUN_NC), 6))
        for _ in 1:8
            MD.write_streaming_cs_window!(writer, window, RUN_NC, 6)
        end
    finally
        MD.close_streaming_transport_binary!(writer)
    end
    return path
end

co2_density(i, j, p, day) = 1e-8 * (0.5 + 0.1i - 0.07j + 0.03p) * (day == 1 ? 1.0 : -0.5)
rn_density(i, j, p) = 2e-20 * (1 + 0.2i + 0.05j + 0.01p)

function write_run_flux(path, areas)
    NCDataset(path, "c") do ds
        defDim(ds, "Xdim", RUN_NC); defDim(ds, "Ydim", RUN_NC); defDim(ds, "nf", 6); defDim(ds, "time", 3)
        defVar(ds, "time", Float64, ("time",),
               attrib = Dict("units" => "hours since 2021-01-01 00:00:00",
                             "calendar" => "proleptic_gregorian"))[:] = [0.0, 24.0, 48.0]
        attrib = Dict("units" => "kg m-2 s-1")
        co2 = defVar(ds, "CO2_FLUX", Float64, ("Xdim", "Ydim", "nf", "time"); attrib)
        rn = defVar(ds, "RN_FLUX", Float64, ("Xdim", "Ydim", "nf", "time"); attrib)
        for (k, day) in enumerate((1, 2, 2))
            co2[:, :, :, k] = [co2_density(i, j, p, day) for i in 1:RUN_NC, j in 1:RUN_NC, p in 1:6]
            rn[:, :, :, k] = [rn_density(i, j, p) for i in 1:RUN_NC, j in 1:RUN_NC, p in 1:6]
        end
        defVar(ds, "cell_area", Float64, ("Xdim", "Ydim", "nf"))[:, :, :] = areas
    end
    return path
end

function run_config(dir, binaries, flux)
    surface(variable, species) = Dict{String, Any}("kind" => "cs_native", "file" => flux, "variable" => variable,
                                                   "species" => species, "time_varying" => true,
                                                   "temporal_scheme" => "stepwise")
    return Dict{String, Any}(
        "input" => Dict{String, Any}("binary_paths" => binaries, "start_date" => "2021-01-01"),
        "architecture" => Dict{String, Any}("use_gpu" => false),
        "numerics" => Dict{String, Any}("float_type" => "Float64"),
        "advection" => Dict{String, Any}("scheme" => "upwind"),
        "diffusion" => Dict{String, Any}("kind" => "none"),
        "convection" => Dict{String, Any}("kind" => "none"),
        "chemistry" => Dict{String, Any}("kind" => "decay",
                                         "half_lives_seconds" => Dict{String, Any}("rn222" => RN222_HALF_LIFE)),
        "tracers" => Dict{String, Any}(
            "co2" => Dict{String, Any}("init" => Dict{String, Any}("kind" => "uniform", "background" => 4e-4),
                                       "surface_flux" => surface("CO2_FLUX", "co2")),
            "rn222" => Dict{String, Any}("init" => Dict{String, Any}("kind" => "uniform", "background" => 0.0),
                                         "surface_flux" => surface("RN_FLUX", "rn222"))),
        "output" => Dict{String, Any}("format" => "netcdf", "split" => "daily",
                                      "snapshot_interval_hours" => 6,
                                      "path" => joinpath(dir, "output", "tiny_{YYYYMMDD}.nc")))
end

@testset "GEOS-Chem CATRINE archive" begin
    dir = mktempdir()
    t0 = DateTime(2021, 12, 1, 3)
    times = collect(t0:Hour(3):t0 + Hour(15))              # 6 output times
    # k = 1 garbage, 2 nested, 3 flat with a fill cell, 4 missing, 5 nested, 6 without EmisSF6
    write(joinpath(dir, gc_name(times[1])), "not a netcdf file")
    nested(t) = joinpath(dir, Dates.format(t, dateformat"yyyy"), Dates.format(t, dateformat"mm"), gc_name(t))
    write_gc_file(nested(times[2]), times[2], fixture_fields(2))
    write_gc_file(joinpath(dir, gc_name(times[3])), times[3], fixture_fields(3); fill_cell = true)
    write_gc_file(nested(times[5]), times[5], fixture_fields(5))
    write_gc_file(joinpath(dir, gc_name(times[6])), times[6], fixture_fields(6); omit = ("EmisSF6",))
    t_empty = times[end] + Hour(3)                                             # outside the script's range
    write_gc_file(joinpath(dir, gc_name(t_empty)), t_empty, fixture_fields(7); no_record = true)

    @testset "paths and times" begin
        a = GEOSChemCATRINEArchive(dir)
        @test gc_path(a, times[2]) == nested(times[2])
        @test gc_path(a, times[3]) == joinpath(dir, gc_name(times[3]))
        @test gc_path(a, times[4]) == joinpath(dir, gc_name(times[4]))            # missing: the flat name
        @test basename(gc_path(a, DateTime(2022, 1, 15, 21))) == "GEOSChem.CATRINE_inst.20220115_2100z.nc4"
        @test gc_times(a, t0, t0 + Hour(15)) == times
        @test gc_times(a, t0, t0 + Hour(16)) == times                             # off-grid end
        @test gc_times(GEOSChemCATRINEArchive(dir; interval = Hour(1)), t0, t0 + Hour(2)) == t0 .+ Hour.(0:2)
        @test_throws ArgumentError gc_times(a, t0, t0 - Hour(3))
        @test occursin("4 tracers, every 3 hours", sprint(show, a))
    end

    @testset "global totals" begin
        a = GEOSChemCATRINEArchive(dir)
        for (k, zero_cell) in ((2, false), (3, true), (5, false))
            g = global_totals(a, times[k])
            d = direct_totals(fixture_fields(k); zero_first_air_cell = zero_cell)
            @test g.dry_air ≈ d.dry_air rtol = 1e-14
            @test all(isapprox.(g.burden, d.burden; rtol = 1e-14))
            @test all(isapprox.(g.emission_rate, d.rate; rtol = 1e-14))
            @test g.burden isa NTuple{4, Float64} && g.emission_rate isa NTuple{4, Float64}
        end
        # GEOS-Chem's molar masses, not the model's: the CO2 rate is the kg CO2 s⁻¹ flux × M_air / 44.01 g/mol.
        g, f = global_totals(a, times[2]), fixture_fields(2)
        kg_co2 = sum(Float64.(f.emis["EmisCO2_Total"]) .* Float64.(f.area))
        @test g.emission_rate[1] == kg_co2 * DRY_AIR_MOLAR_MASS / 44.01e-3
        @test isempty(unreadable_files(a))

        batch = global_totals(a, times)
        @test [b === nothing for b in batch] == [true, false, false, true, false, true]
        for k in (2, 3, 5)
            @test batch[k] === global_totals(a, times[k])
        end
    end

    @testset "unreadable files" begin
        a = GEOSChemCATRINEArchive(dir)
        @test global_totals(a, times[6]) === nothing
        @test global_totals(a, times[4]) === nothing
        @test global_totals(a, times[1]) === nothing
        @test global_totals(a, t_empty) === nothing
        bad = unreadable_files(a)
        @test first.(bad) == vcat(times[[1, 4, 6]], t_empty)                       # time order
        @test occursin("Unknown file format", bad[1].second)
        @test occursin("No such file", bad[2].second)
        @test occursin("EmisSF6", bad[3].second)
        @test bad[4].second == "`Met_AD` has no time record"
        # The file is fine for a reader that does not need EmisSF6.
        @test with_gc_file(ds -> size(ds["Met_AD"]), a, times[6]) == (NX, NX, NF, NLEV, 1)
        # A bug in the reader is not an unreadable file, whatever its error type.
        @test_throws DivideError with_gc_file(ds -> div(1, 0), a, times[2])
        @test_throws KeyError with_gc_file(ds -> Dict{String, Int}()["Met_AD"], a, times[2])
        @test_throws BoundsError with_gc_file(ds -> ds["Met_AD"][1, 1, 1, 1, 2], a, times[2])
        @test length(unreadable_files(a)) == 4
    end

    @testset "construction" begin
        tr = CATRINE_GEOSCHEM_TRACERS
        @test [t.name for t in tr] == [:co2_natural, :co2_fossil, :sf6, :rn222]
        @test [t.molar_mass for t in tr] == [GEOSCHEM_SPECIES_MOLAR_MASS.co2, GEOSCHEM_SPECIES_MOLAR_MASS.co2,
                                             GEOSCHEM_SPECIES_MOLAR_MASS.sf6, GEOSCHEM_SPECIES_MOLAR_MASS.rn222]
        @test [t.half_life for t in tr] == [Inf, Inf, Inf, RN222_HALF_LIFE]
        @test GEOSChemTracer(:x, "A", "B", 1).half_life == Inf
        @test_throws ArgumentError GEOSChemCATRINEArchive(dir; tracers = ())
        @test_throws ArgumentError GEOSChemCATRINEArchive(dir; tracers = (tr[1], tr[1]))
        @test_throws ArgumentError GEOSChemCATRINEArchive(dir; interval = Hour(0))
        @test_throws ArgumentError GEOSChemCATRINEArchive(dir; tracers = (GEOSChemTracer(:x, "A", "B", -1.0),))
        @test_throws ArgumentError GEOSChemCATRINEArchive(dir; tracers = (GEOSChemTracer(:x, "A", "B", 1.0; half_life = 0),))
    end

    @testset "exact decay" begin
        λ, E, B0, Δ = log(2) / RN222_HALF_LIFE, 3.0e-8, 2.0e-3, 10800.0
        B = B0
        for _ in 1:40
            B = mb_call(:exact_decay, B, E, λ, Δ)
        end
        T = 40Δ
        @test B ≈ B0 * exp(-λ * T) + E / λ * (1 - exp(-λ * T)) rtol = 1e-13
        @test mb_call(:exact_decay, 2.0, 0.5, 0.0, 4.0) == 4.0
    end

    @testset "catrine_true_mass_balance.jl --model geoschem" begin
        out = joinpath(mktempdir(), "gc.csv")
        origin = DateTime(2021, 12, 1)
        io = IOBuffer()
        mb_call(:main, ["--model", "geoschem", "--gc-dir", dir, "--start", string(times[1]),
                        "--end", string(times[end]), "--origin", string(origin), "--out", out]; io)
        text = String(take!(io))
        lines = readlines(out)
        # The csv columns of the Python script (and of the AtmosTransport mode).
        @test lines[1] == "hours,dry_air_kg,co2_natural_burden,co2_natural_source_cum,co2_natural_residual," *
                          "co2_fossil_burden,co2_fossil_source_cum,co2_fossil_residual,sf6_burden,sf6_source_cum," *
                          "sf6_residual,rn222_burden,rn222_source_cum,rn222_residual"
        rows = [parse.(Float64, split(l, ",")) for l in lines[2:end]]
        @test [r[1] for r in rows] == [6.0, 9.0, 15.0]                           # readable files only, hours since origin

        # Independent port of the Python algorithm on the direct totals.
        readable = Dict(2 => direct_totals(fixture_fields(2)), 3 => direct_totals(fixture_fields(3); zero_first_air_cell = true),
                        5 => direct_totals(fixture_fields(5)))
        rate_from = Dict(3 => 3, 4 => 5, 5 => 5, 6 => 5)                          # next readable, else the last
        λ = [log(2) / tr.half_life for tr in CATRINE_GEOSCHEM_TRACERS]
        c, ref = zeros(4), copy(readable[2].burden)
        expected = Dict{Int, Vector{Float64}}()
        for k in 2:6
            if k > 2
                r = readable[rate_from[k]].rate
                for i in 1:4
                    c[i] += r[i] * 10800
                    ref[i] = λ[i] > 0 ? ref[i] * exp(-λ[i] * 10800) + r[i] * (1 - exp(-λ[i] * 10800)) / λ[i] :
                                        ref[i] + r[i] * 10800
                end
            end
            haskey(readable, k) || continue
            d = readable[k]
            expected[k] = vcat(3.0 * k, d.dry_air, vcat(([d.burden[i], c[i], d.burden[i] - ref[i]] for i in 1:4)...))
        end
        for (row, k) in zip(rows, (2, 3, 5))
            e = expected[k]
            @test row[1] == e[1]
            @test row[2] ≈ e[2] rtol = 1e-14
            for i in 1:4
                burden, source, residual = row[3i:3i + 2]
                @test burden ≈ e[3i] rtol = 1e-14
                @test source ≈ e[3i + 1] rtol = 1e-14
                @test abs(residual - e[3i + 2]) <= 1e-12 * e[3i]                # residual to 1e-12 of the burden
            end
        end
        @test all(rows[1][[4, 5, 7, 8, 10, 11, 13, 14]] .== 0)                  # t₀: no source, no residual

        # Summary: unreadable files, then one line per tracer at the last row.
        @test occursin("3 unreadable files", text)
        @test occursin("($(joinpath(dir, gc_name(times[4])))): cannot read the file", text)
        last_row = rows[end]
        for (i, tr) in enumerate(CATRINE_GEOSCHEM_TRACERS)
            r, cum, air = last_row[3i + 2], last_row[3i + 1], last_row[2]
            line = @sprintf("  %-24s r = %+.4e kg-storage   r/∫E = %+.3e   r/air = %+.4e mol/mol", tr.name, r, r / cum, r / air)
            @test occursin(line, text)
        end
    end


    @testset "catrine_true_mass_balance.jl (atmostransport) on a tiny run" begin
        rdir = mktempdir()
        binaries = [write_run_binary(joinpath(rdir, "tiny_20210101.bin")),
                    write_run_binary(joinpath(rdir, "tiny_20210102.bin"))]
        mesh = MD.driver_grid(MD.TransportBinaryDriver(binaries[1]; FT = Float64)).horizontal
        areas = repeat(Float64.(mesh.cell_areas), 1, 1, 6)
        cfg = run_config(rdir, binaries, write_run_flux(joinpath(rdir, "flux.nc"), areas))
        with_logger(NullLogger()) do
            redirect_stdout(devnull) do
                redirect_stderr(() -> AtmosTransport.run_driven_simulation(cfg), devnull)
            end
        end
        cfg_path = joinpath(rdir, "run.toml")
        open(io -> TOML.print(io, cfg), cfg_path, "w")
        out = joinpath(rdir, "balance.csv")
        io = IOBuffer()
        mb = with_logger(NullLogger()) do
            mb_call(:main, [cfg_path, joinpath(rdir, "output", "tiny_*.nc"), out]; io)
        end
        text = String(take!(io))
        lines = readlines(out)
        @test lines[1] == "hours,dry_air_kg,co2_burden,co2_source_cum,co2_residual,rn222_burden,rn222_source_cum,rn222_residual"
        rows = [parse.(Float64, split(l, ",")) for l in lines[2:end]]
        hours = [r[1] for r in rows]
        @test hours == collect(6.0:6.0:48.0)                                     # from start_hours = 3; boundary frame once
        @test all(r -> isapprox(r[2], RUN_AIR * RUN_NC^2 * 6 * RUN_NZ; rtol = 1e-12), rows)

        # The source: Σ density × area × M_air / M over the stepwise daily slices.
        storage(density, M) = sum(density .* areas) * DRY_AIR_MOLAR_MASS / M
        co2_rate(day) = storage([co2_density(i, j, p, day) for i in 1:RUN_NC, j in 1:RUN_NC, p in 1:6], SPECIES_MOLAR_MASS.co2)
        rn_rate = storage([rn_density(i, j, p) for i in 1:RUN_NC, j in 1:RUN_NC, p in 1:6], SPECIES_MOLAR_MASS.rn222)
        for r in rows
            s1 = (min(r[1], 24.0) - 6.0) * 3600
            s2 = max(r[1] - 24.0, 0.0) * 3600
            @test r[4] ≈ co2_rate(1) * s1 + co2_rate(2) * s2 rtol = 1e-12
            @test r[7] ≈ rn_rate * (r[1] - 6.0) * 3600 rtol = 1e-12
            # Transport moves nothing, so the run conserves: r ≈ 0 for the stable tracer
            # and, with the model's symmetric decay/emission split, for Rn-222.
            @test abs(r[5]) <= 1e-12 * r[3]
            @test abs(r[8]) <= 1e-12 * max(r[6], rn_rate * 10800)
        end
        @test rows[1][[4, 5, 7, 8]] == zeros(4)
        # Midpoint emissions leave only the negative O((λT)²) Strang-splitting bias.
        m = match(r"rn222\s+splitting bias of the model's decay vs the exact solution: ([+-]\d\.\d+e[+-]\d+)", text)
        @test m !== nothing
        @test -1e-4 < parse(Float64, only(m.captures)) < 0
        @test occursin(r"\s+co2\s+r = [+-]\d\.\d{4}e[+-]\d+ kg-storage", text)
        @test mb.tracers == [:co2, :rn222]
    end

    @testset "command line" begin
        @test_throws ArgumentError mb_call(:parse_command_line, ["--model", "geoschem", "--gc-dir", dir])     # no --out
        @test_throws ArgumentError mb_call(:parse_command_line, ["--model", "geoschem", "--out", "x.csv", "extra"])
        @test_throws ArgumentError mb_call(:parse_command_line, ["run.toml", "out/*.nc", "x.csv", "--gc-dir", dir])
        @test_throws ArgumentError mb_call(:parse_command_line, ["run.toml", "out/*.nc"])
        run, out = mb_call(:parse_command_line, ["run.toml", "out/*.nc", "x.csv"])
        @test (run.config, run.pattern, run.start_hours, out) == ("run.toml", "out/*.nc", 3.0, "x.csv")
        run, out = mb_call(:parse_command_line, ["--model", "geoschem", "--out", "x.csv"])
        @test (run.start, run.stop, run.origin) ==
              (DateTime(2021, 12, 1, 3), DateTime(2024, 1, 1), DateTime(2021, 12, 1))
        @test run.archive.dir == expanduser("~/data/AtmosTransport/catrine-geoschem-runs/C90")
    end
end
