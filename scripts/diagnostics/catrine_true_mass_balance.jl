#!/usr/bin/env julia
# =============================================================================
# catrine_true_mass_balance.jl — true global mass balance of a multi-day run
# =============================================================================
#
# For every output time t and tracer, the residual
#
#     r(t) = B(t) − B̂(t),     B̂(t₀) = B(t₀)
#
# is the mass the model gained (r > 0) or lost (r < 0) beyond what its surface
# sources put in. B̂ is the burden of a model that conserves mass exactly: for
# a stable tracer B̂(t) = B(t₀) + ∫_{t₀}^{t} E dt, so r = B(t) − B(t₀) − ∫E dt;
# for a decaying tracer B̂ also decays (below). Transport only moves mass, so r
# is the true conservation error of the whole model (advection, convection,
# diffusion, mass fixers). Surface sources and radioactive decay are the only
# non-transport terms accounted for. All masses are in dry-air-equivalent
# storage units (kg of dry air carrying the tracer's dry mole fraction).
#
# --model atmostransport (default): an AtmosTransport run, from its snapshot
# files and its config. Runs with deposition are rejected.
#
#   * B(t) is the run's own compensated Float64 global storage mass,
#     `<tracer>_total_mass` = Σ VMR_dry · m_dry (kg dry-air equivalent).
#   * E is the run's own applied source: the sources are rebuilt with the
#     model's `build_surface_flux_sources` from the run's config (same files,
#     regridding, unit and storage scaling) and integrated between output
#     times with the surface operator's own `emission_segments` (one step per
#     interval, split at every source slice time). In storage units the molar
#     masses cancel, so no inventory rate or unit factor enters.
#   * Decaying tracers (Rn-222): exact half-window decay steps bracket every
#     met window T. The reference follows that order,
#     B̂ ← (B̂ e^{−λT/2} + ∫_window E dt) e^{−λT/2}, so r is again the
#     conservation error. The remaining O((λT)²) departure from the exact
#     solution of B' = E − λB is reported separately as the splitting bias.
#   * The dry-air mass m_dry comes from the output's dry
#     `column_air_mass_per_area`. The output `time` axis is taken as hours since
#     [input].start_date 00 UTC, the origin of the model's source times.
#     Cubed-sphere runs with stepwise or conservative-mean sources only.
#
# --model geoschem: the GEOS-Chem (GCHP) CATRINE reference run, from its own
# 3-hourly instantaneous output (`AtmosTransport.Analysis.GEOSChemCATRINEArchive`,
# files `GEOSChem.CATRINE_inst.YYYYMMDD_HHMMz.nc4` flat or under YYYY/MM/).
#
#   * B = Σ SpeciesConcVV · Met_AD and m_dry = Σ Met_AD.
#   * E = (Σ Emis · Met_AREAM2) · M_air / M_species, the emission flux
#     (kg species s⁻¹) converted to storage units with GEOS-Chem's own molar
#     masses (`GEOSCHEM_SPECIES_MOLAR_MASS`, and `DRY_AIR_MOLAR_MASS` = AIRMW).
#     The emission fields are instantaneous: each is taken as the constant rate
#     over the interval before its output time (the emission step that ends
#     there). Over months this convention shifts the cumulative source by at
#     most one 3-hour slice.
#   * Rn-222: B̂ is the exact solution of B̂' = E − λB̂ with E constant on each
#     interval (λ = ln 2 / `RN222_HALF_LIFE`), so GEOS-Chem's Rn-222 residual
#     includes its own emission/decay splitting.
#   * An unreadable file (missing, truncated, lacking a variable) is listed;
#     its row is skipped, and the source over its interval uses the rate of the
#     next readable file (of the last readable one at the end of the series).
#     t₀ is the first readable time.
#   * Files are read into memory on all Julia threads (`julia -t 8`; each
#     thread holds one file, about 145 MB at C90 and 580 MB at C180); the
#     NetCDF decoding is serialised.
#
# r is reported in kg-storage, relative to the cumulative source, and as a
# global-mean dry mole fraction r / m_dry (mol/mol; ×1e6 for ppm, ×1e12 for
# ppt). Both models write the same csv columns: hours, dry_air_kg and, per
# tracer, <tracer>_burden, <tracer>_source_cum and <tracer>_residual (kg);
# `scripts/diagnostics/catrine_benchmark_page.py` plots them.
#
#   julia --project=. scripts/diagnostics/catrine_true_mass_balance.jl \
#       <run.toml> '<directory>/<file glob>' <out.csv> [start_hours]
#
#   julia -t 8 --project=. scripts/diagnostics/catrine_true_mass_balance.jl --model geoschem \
#       [--gc-dir ~/data/AtmosTransport/catrine-geoschem-runs/C90] \
#       [--start 2021-12-01T03:00] [--end 2024-01-01T00:00] [--origin 2021-12-01T00:00] \
#       --out <out.csv>
#
# start_hours (default 3) is the first output time of an AtmosTransport run
# used as t₀; 3 h matches the first GEOS-Chem CATRINE output. For GEOS-Chem,
# --start and --end bound the output times, and --origin is the time origin of
# the csv hours (the run start). The defaults are the CATRINE protocol run.
# =============================================================================

using AtmosTransport, NCDatasets, TOML, Printf, Dates, ArgParse
using AtmosTransport.Operators: emission_segments
using AtmosTransport.Analysis: GEOSChemCATRINEArchive, gc_path, gc_times, global_totals, unreadable_files

const AT   = AtmosTransport
const DR   = AT.Models.DrivenRunner
const ICIO = AT.Models.InitialConditionIO

# -----------------------------------------------------------------------------
# The residual series, shared by both models
# -----------------------------------------------------------------------------

"""
Per output time: hours since the origin, dry-air mass (kg), and per tracer the
burden B, the cumulative source ∫E dt and the residual r = B − B̂ (kg-storage).
"""
struct MassBalance
    hours    :: Vector{Float64}
    dry_air  :: Vector{Float64}
    tracers  :: Vector{Symbol}
    burden   :: Dict{Symbol, Vector{Float64}}
    source   :: Dict{Symbol, Vector{Float64}}
    residual :: Dict{Symbol, Vector{Float64}}
end

MassBalance(tracers) = MassBalance(Float64[], Float64[], collect(Symbol, tracers),
                                   Dict(n => Float64[] for n in tracers), Dict(n => Float64[] for n in tracers),
                                   Dict(n => Float64[] for n in tracers))

# Exact solution of B' = r − λB over a length ℓ with r constant (λ = 0: B + rℓ).
exact_decay(B, r, λ, ℓ) = λ > 0 ? B * exp(-λ * ℓ) - r * expm1(-λ * ℓ) / λ : B + r * ℓ

function write_csv(path, mb::MassBalance)
    open(path, "w") do io
        println(io, "hours,dry_air_kg,", join(("$(n)_burden,$(n)_source_cum,$(n)_residual" for n in mb.tracers), ","))
        for k in eachindex(mb.hours)
            println(io, mb.hours[k], ",", mb.dry_air[k], ",",
                    join((x for n in mb.tracers for x in (mb.burden[n][k], mb.source[n][k], mb.residual[n][k])), ","))
        end
    end
end

# The splitting bias of decaying tracers (if any), then one line per tracer at the last time.
function print_summary(io, mb::MassBalance, bias)
    for n in mb.tracers
        haskey(bias, n) || continue
        @printf(io, "  %-24s splitting bias of the model's decay vs the exact solution: %+.3e of the burden\n",
                n, bias[n])
    end
    for n in mb.tracers
        r, c = mb.residual[n][end], mb.source[n][end]
        @printf(io, "  %-24s r = %+.4e kg-storage   r/∫E = %+.3e   r/air = %+.4e mol/mol   max|r|/air = %.4e\n",
                n, r, c == 0 ? NaN : r / c, r / mb.dry_air[end], maximum(abs, mb.residual[n] ./ mb.dry_air))
    end
end

# -----------------------------------------------------------------------------
# --model atmostransport
# -----------------------------------------------------------------------------

"An AtmosTransport run: its config, a glob of its snapshot files, and t₀ in output hours."
struct AtmosTransportRun
    config      :: String
    pattern     :: String
    start_hours :: Float64
end

# One integral per output interval (or met window) equals the run's substep sum
# when the step integral is exact (static, stepwise, conservative-mean sources).
# LinearInterpFlux samples its rate at every substep start instead.
function check_interval_integrable(s::AT.TimeVaryingSurfaceFluxSource)
    s.scheme isa AT.LinearInterpFlux &&
        error("$(s.tracer_name): LinearInterpFlux samples per substep; one integral per output interval would not match")
    return nothing
end
check_interval_integrable(_) = nothing

# (global storage rate kg/s, length s) pieces of the source over [t, t + Δ], as
# the surface operator applies them; a tracer without a source adds nothing.
segments(src, t, Δ) = emission_segments(src, t, Δ)
segments(::Nothing, t, Δ) = ((0.0, Δ),)

emitted(segs) = sum(r * ℓ for (r, ℓ) in segs; init = 0.0)

# Snapshot files matching a shell-style glob ('*' only) in one directory, sorted.
function snapshot_files(pattern)
    dir, glob = splitdir(pattern)
    dir = isempty(dir) ? "." : dir
    isdir(dir) || error("no directory $dir for pattern $pattern")
    # '*' → '.*', the literal parts escaped (a shell-style glob without a Glob.jl dependency)
    escaped = [replace(p, r"[\\^$.|?*+()\[\]{}]" => s"\\\0") for p in split(glob, "*")]
    pattern_re = Regex("^" * join(escaped, ".*") * "\$")
    files = joinpath.(dir, sort(filter(f -> occursin(pattern_re, f), readdir(dir))))
    isempty(files) && error("no files match $pattern")
    return files
end

function mass_balance(run::AtmosTransportRun)
    cfg = TOML.parsefile(run.config)
    haskey(cfg, "deposition") && error("deposition is not accounted for; the residual would include it")
    FT = DR.configured_float_type(cfg)
    driver = AT.MetDrivers.TransportBinaryDriver(first(AT.expand_binary_paths(cfg["input"])); FT, arch = AT.CPU(), Hp = 1)
    specs = DR.parse_tracer_specs(cfg)
    sources = ICIO.build_surface_flux_sources(AT.MetDrivers.driver_grid(driver), specs, FT;
                                              reference_time = DR.run_reference_time(cfg))
    source = Dict(s.tracer_name => s for s in sources)
    window = Float64(AT.MetDrivers.window_dt(driver))      # met window T (s); decay is applied once per window
    chem = get(cfg, "chemistry", Dict{String, Any}())
    λ = get(chem, "kind", "") == "decay" ?
        Dict(Symbol(n) => log(2) / Float64(h) for (n, h) in get(chem, "half_lives_seconds", Dict())) :
        Dict{Symbol, Float64}()
    tracers = [s.name for s in specs]
    mb = MassBalance(tracers)

    # Output series; daily files share their boundary frame, which is read once.
    for f in snapshot_files(run.pattern)
        NCDataset(f) do ds
            area = Float64.(ds["cell_area"][:, :, :])
            # Keep the model-relative elapsed hours used by the budget equations.
            # NCDatasets decodes the runtime CF time coordinate to DateTime, so
            # read the stored numeric coordinate through the underlying variable.
            for (k, h) in enumerate(Float64.(ds["time"].var[:]))
                (h < run.start_hours || (!isempty(mb.hours) && h == mb.hours[end])) && continue
                push!(mb.hours, h)
                foreach(n -> push!(mb.burden[n], Float64(ds["$(n)_total_mass"][k])), tracers)
                push!(mb.dry_air, sum(Float64.(ds["column_air_mass_per_area"][:, :, :, k]) .* area))
            end
        end
    end
    isempty(mb.hours) && error("no output time at or after $(run.start_hours) h in $(run.pattern)")
    issorted(mb.hours) || error("output times are not increasing; check the file glob")
    t = mb.hours .* 3600

    # Residual series per tracer: cumulative source, r(t) = B(t) − B̂(t) and, for
    # decaying tracers, the splitting bias B̂_model − B̂_exact.
    bias = Dict{Symbol, Float64}()
    for n in tracers
        src = get(source, n, nothing)
        check_interval_integrable(src)
        l = get(λ, n, 0.0)
        B = mb.burden[n]
        c, ref, exact = 0.0, B[1], B[1]
        cum, res = zeros(length(t)), zeros(length(t))
        for k in 2:length(t)
            if l > 0      # window by window: symmetric half-decay around emissions; exact ODE alongside
                nw = round(Int, (t[k] - t[k - 1]) / window)
                nw * window ≈ t[k] - t[k - 1] ||
                    error("output interval $(t[k] - t[k - 1]) s is not a multiple of the met window $window s")
                for a in t[k - 1] .+ window .* (0:nw - 1)
                    segs = segments(src, a, window)
                    e = emitted(segs)
                    c += e
                    half_decay = exp(-l * window / 2)
                    ref = (ref * half_decay + e) * half_decay
                    for (r, ℓ) in segs
                        exact = exact_decay(exact, r, l, ℓ)
                    end
                end
            else
                e = emitted(segments(src, t[k - 1], t[k] - t[k - 1]))
                c += e; ref += e
            end
            cum[k], res[k] = c, B[k] - ref
        end
        append!(mb.source[n], cum)
        append!(mb.residual[n], res)
        l > 0 && exact > 0 && (bias[n] = (ref - exact) / exact)
    end
    return mb, bias
end

function describe(io, run::AtmosTransportRun, mb::MassBalance, out_csv)
    println(io, "true mass balance from t₀ = $(run.start_hours) h to $(mb.hours[end]) h ($(length(mb.hours)) times) → $out_csv")
    println(io, "  residual r = B(t) − B(t₀) − ∫E dt (decay: B − B̂ with the model's symmetric half-window order); mole fraction = r / dry-air mass")
end

# -----------------------------------------------------------------------------
# --model geoschem
# -----------------------------------------------------------------------------

"The GEOS-Chem CATRINE run: its archive, the first and last output time, and the csv time origin."
struct GEOSChemRun
    archive :: GEOSChemCATRINEArchive
    start   :: DateTime
    stop    :: DateTime
    origin  :: DateTime
end

const MS_PER_HOUR = Dates.value(Millisecond(Hour(1)))

function mass_balance(run::GEOSChemRun)
    a = run.archive
    times = gc_times(a, run.start, run.stop)
    totals = global_totals(a, times)
    ok = findall(!isnothing, totals)
    if isempty(ok)
        bad = unreadable_files(a)
        error("no readable GEOS-Chem file among the $(length(times)) times $(run.start) .. $(run.stop) under $(a.dir)" *
              (isempty(bad) ? "" : "; $(first(bad).first): $(first(bad).second)"))
    end

    # The rate over the interval ending at times[k] comes from the first readable
    # file at or after it, or from the last readable file at the end of the series.
    rate_file = Vector{Int}(undef, length(times))
    next = last(ok)
    for k in reverse(eachindex(times))
        totals[k] === nothing || (next = k)
        rate_file[k] = next
    end

    Δ = Dates.value(a.interval) / 1000                           # s
    tracers = [tr.name for tr in a.tracers]
    λ = [log(2) / tr.half_life for tr in a.tracers]               # 0 for a stable tracer
    c = zeros(length(tracers))
    ref = collect(totals[first(ok)].burden)
    mb = MassBalance(tracers)
    for k in eachindex(times)
        if k > first(ok)
            e = totals[rate_file[k]].emission_rate
            for i in eachindex(tracers)
                c[i] += e[i] * Δ
                ref[i] = exact_decay(ref[i], e[i], λ[i], Δ)
            end
        end
        tot = totals[k]
        tot === nothing && continue
        push!(mb.hours, Dates.value(times[k] - run.origin) / MS_PER_HOUR)
        push!(mb.dry_air, tot.dry_air)
        for (i, n) in enumerate(tracers)
            push!(mb.burden[n], tot.burden[i])
            push!(mb.source[n], c[i])
            push!(mb.residual[n], tot.burden[i] - ref[i])
        end
    end
    return mb, Dict{Symbol, Float64}()
end

function describe(io, run::GEOSChemRun, mb::MassBalance, out_csv)
    a = run.archive
    bad = unreadable_files(a)
    for (t, reason) in bad
        println(io, "unreadable $(t) ($(gc_path(a, t))): $(reason)")
    end
    isempty(bad) || println(io, "$(length(bad)) unreadable files (rate taken from the next readable one; burden skipped)")
    at(hours) = run.origin + Millisecond(round(Int64, hours * MS_PER_HOUR))
    println(io, "GEOS-Chem true mass balance $(at(mb.hours[1])) .. $(at(mb.hours[end])) ($(length(mb.hours)) times) → $out_csv")
    println(io, "  residual r = B(t) − B(t₀) − ∫E dt (decay: B − B̂, the exact solution of B̂' = E − λB̂); mole fraction = r / dry-air mass")
end

# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------

const GEOSCHEM_OPTIONS = ("gc-dir", "start", "end", "origin", "out")

function argparse_settings()
    s = ArgParseSettings(prog = "catrine_true_mass_balance.jl",
                         description = "True global mass balance of an AtmosTransport run or of the GEOS-Chem CATRINE run.")
    @add_arg_table! s begin
        "--model"
            help = "atmostransport (a run's snapshot files and config) or geoschem (the GEOS-Chem CATRINE archive)"
            default = "atmostransport"
            range_tester = in(("atmostransport", "geoschem"))
        "--gc-dir"
            help = "geoschem: directory of the GEOSChem.CATRINE_inst files, flat or YYYY/MM/ " *
                   "(default ~/data/AtmosTransport/catrine-geoschem-runs/C90)"
        "--start"
            help = "geoschem: first output time (default 2021-12-01T03:00)"
        "--end"
            help = "geoschem: last output time (default 2024-01-01T00:00)"
        "--origin"
            help = "geoschem: time origin of the csv hours, the run start (default 2021-12-01T00:00)"
        "--out"
            help = "geoschem: output csv"
        "arguments"
            help = "atmostransport: <run.toml> '<dir>/<file glob>' <out.csv> [start_hours]"
            nargs = '*'
    end
    return s
end

"Parse the command line into (run, out_csv)."
function parse_command_line(argv)
    opts = parse_args(argv, argparse_settings())
    positional = opts["arguments"]
    if opts["model"] == "geoschem"
        isempty(positional) ||
            throw(ArgumentError("--model geoschem takes no positional arguments; got $(positional)"))
        opts["out"] === nothing && throw(ArgumentError("--model geoschem needs --out <csv>"))
        option(key, default) = something(opts[key], default)
        archive = GEOSChemCATRINEArchive(expanduser(option("gc-dir", "~/data/AtmosTransport/catrine-geoschem-runs/C90")))
        run = GEOSChemRun(archive, DateTime(option("start", "2021-12-01T03:00")),
                          DateTime(option("end", "2024-01-01T00:00")),
                          DateTime(option("origin", "2021-12-01T00:00")))
        return run, opts["out"]
    end
    given = filter(k -> opts[k] !== nothing, GEOSCHEM_OPTIONS)
    isempty(given) || throw(ArgumentError("--$(first(given)) applies to --model geoschem only"))
    length(positional) in 3:4 ||
        throw(ArgumentError("usage: catrine_true_mass_balance.jl <run.toml> '<dir>/<file glob>' <out.csv> [start_hours]"))
    start_hours = parse(Float64, get(positional, 4, "3"))
    return AtmosTransportRun(positional[1], expanduser(positional[2]), start_hours), positional[3]
end

function main(argv = ARGS; io = stdout)
    run, out_csv = parse_command_line(argv)
    mb, bias = mass_balance(run)
    write_csv(out_csv, mb)
    describe(io, run, mb, out_csv)
    print_summary(io, mb, bias)
    return mb
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
