#!/usr/bin/env julia
"""
Compare convection schemes on the 4-tracer single-layer IC experiment.
Each scheme is one NetCDF written by `scripts/run_transport.jl` with
NoAdvection + NoDiffusion + the scheme under test. The 4 tracers all
carry identical molecule counts but are concentrated in 4 different
pressure layers initially (lowest model layer + 0.8/0.6/0.4 × psurf).

For each tracer × scheme, emits a CSV with the global-mean per-layer
VMR for every snapshot time, plus a markdown summary that lists the
peak layer (k_argmax) and percent-mass-above-initial-layer at the
final time, plus the total mass evolution (mass-conservation diagnostic).

Usage:
  julia --project=. scripts/diagnostics/heritage/compare_convection_schemes.jl \\
      --label=tm5_n1:/temp1/convection_compare_4tracer/convonly_tm5_era5_c180_3day.nc \\
      --label=tm5_n2:/temp1/convection_compare_4tracer/convonly_tm5_n2_era5_c180_3day.nc \\
      --label=tm5_n3:/temp1/convection_compare_4tracer/convonly_tm5_n3_era5_c180_3day.nc \\
      --label=cmfmc:/temp1/convection_compare_4tracer/convonly_cmfmc_geosit_c180_3day.nc \\
      --output /temp1/convection_compare_4tracer/comparison
"""

using NCDatasets
using Printf
using Statistics
using Dates

"""
Coerce a time-axis value (could be `DateTime` from one writer or
`Float64` hours from another) into a Float64 hour offset relative
to the first entry. Lets the CSV always carry numeric hours.
"""
@inline _hours_from_start(t0, t) = t isa Dates.DateTime ?
    Dates.value(Dates.Millisecond(t - t0)) / 3600_000.0 :
    Float64(t) - Float64(t0)

const TRACER_NAMES = ("co2_lowest_layer", "co2_p08", "co2_p06", "co2_p04")

function parse_cli(args)
    labels_paths = Pair{String,String}[]
    output_prefix = ""
    for a in args
        if startswith(a, "--label=")
            spec = a[length("--label=")+1:end]
            colon = findfirst(==(':'), spec)
            colon === nothing && error("--label=<name>:<path> required")
            push!(labels_paths, String(spec[1:colon-1]) => String(spec[colon+1:end]))
        elseif startswith(a, "--output=")
            output_prefix = a[length("--output=")+1:end]
        elseif a == "--output"
            # positional next-arg fallback handled by caller; treat as no-op here
        end
    end
    # Allow `--output <prefix>` two-arg form too.
    i = 1
    while i <= length(args)
        if args[i] == "--output" && i+1 <= length(args)
            output_prefix = args[i+1]
            i += 2
        else
            i += 1
        end
    end
    isempty(labels_paths) && error("at least one --label=<name>:<path> required")
    isempty(output_prefix) && error("--output <prefix> required")
    return labels_paths, output_prefix
end

"""
    layer_profile(ds, tracer)

Returns `(times_hours, profile)` where `profile[k, t]` is the
global-mean VMR at layer `k` and snapshot `t`. Treats missing as 0.
"""
function layer_profile(ds::NCDataset, tracer::AbstractString)
    times = Array(ds["time"][:])
    # Read one snapshot at a time to avoid materializing the full
    # ~28 GB array; vectorize the spatial sum per snapshot. Was
    # written as a scalar loop initially — 65 min/tracer that way.
    var = ds[tracer]
    dims = size(var)                    # (Xdim, Ydim, nf, lev, time)
    Nx, Ny, Nf, Nz, Nt = dims
    ncells = Nx * Ny * Nf
    profile = Array{Float64}(undef, Nz, Nt)
    for t in 1:Nt
        slab = Array(var[:, :, :, :, t]) # (Xdim, Ydim, nf, lev)
        slab = coalesce.(slab, 0.0f0)
        sums = dropdims(sum(Float64.(slab); dims = (1, 2, 3)); dims = (1, 2, 3))
        profile[:, t] .= sums ./ ncells
    end
    return times, profile
end

function initial_peak_k(profile::AbstractMatrix)
    p0 = view(profile, :, 1)
    return argmax(p0)
end

"""
    fraction_above(profile, kref, t)

Of the total VMR at time t, what fraction sits at layer indices `< kref`
(i.e. above the initial layer, since k=1 is TOA)?
"""
function fraction_above(profile::AbstractMatrix, kref::Int, t::Int)
    total = sum(@view profile[:, t])
    above = sum(@view profile[1:kref-1, t])
    return total == 0 ? 0.0 : above / total
end

function write_csv(path::AbstractString, times::AbstractVector,
                    profile::AbstractMatrix, tracer::AbstractString, scheme::AbstractString)
    Nz, Nt = size(profile)
    t0 = times[1]
    open(path, "w") do io
        write(io, "scheme,tracer,t_hours,k,mean_vmr\n")
        for t in 1:Nt, k in 1:Nz
            @printf(io, "%s,%s,%.4f,%d,%.9e\n",
                    scheme, tracer, _hours_from_start(t0, times[t]), k, profile[k, t])
        end
    end
end

"""
    tracer_total_mass(ds, tracer, t)

Sum of `co2_*_column_mass_per_area × cell_area` at snapshot `t` — the
stored tracer-mass total. Returns 0 if the column-mass variable is
absent. Used for mass-conservation diagnostics.
"""
function tracer_total_mass(ds::NCDataset, tracer::AbstractString, t::Int)
    name = tracer * "_column_mass_per_area"
    haskey(ds, name) || return NaN
    haskey(ds, "cell_area") || return NaN
    slab = coalesce.(Array(ds[name][:, :, :, t]), 0.0f0)   # (Xdim, Ydim, nf)
    cell_area = coalesce.(Array(ds["cell_area"]), 0.0f0)   # avoid [:] flatten
    # Reshape to match the slab if NCDatasets ever returns flat.
    cell_area3 = reshape(cell_area, size(slab)...)
    return sum(Float64.(slab) .* Float64.(cell_area3))
end

function main()
    labels_paths, output_prefix = parse_cli(ARGS)

    isdir(dirname(output_prefix)) || mkpath(dirname(output_prefix))

    summary_lines = String[]
    push!(summary_lines, "# 4-tracer convection comparison\n")
    push!(summary_lines, "Each tracer is initialized as a single-layer spike (lowest model layer,")
    push!(summary_lines, "or layer whose log-midpoint p ≈ 0.8/0.6/0.4 × ps_col), all with identical")
    push!(summary_lines, "molecule counts. NoAdvection + NoDiffusion + the scheme under test.")
    push!(summary_lines, "")
    push!(summary_lines, "Columns:")
    push!(summary_lines, "- `k_peak`     = layer with highest global-mean VMR at t=0 (k=1=TOA, k=Nz=surface)")
    push!(summary_lines, "- `%above`     = fraction of column-mean VMR at the final hour that sits ABOVE k_peak")
    push!(summary_lines, "                 (i.e. mass that's been lifted higher than its initial layer)")
    push!(summary_lines, "- `mass_drift` = (total tracer mass at t_final - total at t=0) / total at t=0,")
    push!(summary_lines, "                 a mass-conservation check (0 = perfectly conserved)")
    push!(summary_lines, "")

    push!(summary_lines, "| Tracer | Scheme | k_peak | t_final (h) | %above | mass_drift |")
    push!(summary_lines, "|---|---|---:|---:|---:|---:|")

    for tracer in TRACER_NAMES
        for (label, path) in labels_paths
            isfile(path) || begin
                push!(summary_lines, @sprintf("| %s | %s | (missing file %s) | | | |", tracer, label, path))
                continue
            end
            NCDataset(path) do ds
                times, profile = layer_profile(ds, tracer)
                kref = initial_peak_k(profile)
                t_final = length(times)
                frac = fraction_above(profile, kref, t_final)
                m0 = tracer_total_mass(ds, tracer, 1)
                mf = tracer_total_mass(ds, tracer, t_final)
                drift = (isnan(m0) || m0 == 0) ? NaN : (mf - m0) / m0
                csv_path = output_prefix * "_$(label)_$(tracer).csv"
                write_csv(csv_path, times, profile, tracer, label)
                drift_str = isnan(drift) ? "n/a" : @sprintf("%+.3e", drift)
                hours_final = _hours_from_start(times[1], times[t_final])
                push!(summary_lines,
                      @sprintf("| %s | %s | %d | %.1f | %.2f%% | %s |",
                              tracer, label, kref, hours_final, 100*frac, drift_str))
            end
        end
    end

    md_path = output_prefix * "_summary.md"
    open(md_path, "w") do io
        for line in summary_lines
            println(io, line)
        end
    end

    println("Wrote per-tracer per-scheme CSVs + summary to:")
    println("  ", output_prefix, "_*.csv")
    println("  ", md_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
