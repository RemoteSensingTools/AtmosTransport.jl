#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Check global dry-mass balance closure across the Dec 2021 GEOS-IT (or any
# v4 CS) transport binaries.
#
# For each binary, opens every window and reports:
#   - total dry-air mass Σ m[panel, i, j, k] (kg)
#   - drift from the pinned target (relative ppm)
#   - max-abs window-to-window drift within the day
#
# Across days reports min/max/mean of the per-window totals and the
# largest day-to-day jump.
#
# Pinned target convention: 5.135313897e18 kg corresponds to a
# global-mean dry surface pressure of 98726.0 Pa per the
# `[mass_fix].target_ps_dry_pa` knob in the preprocessing TOMLs.
#
# Usage:
#   julia --project=. scripts/diagnostics/check_mass_balance_dec2021.jl \
#       ~/data/AtmosTransport/met/geosit/C180/transport_binary_dec2021_catrine_f32 \
#       [--target-kg 5.135313897e18]
# ---------------------------------------------------------------------------

using Printf
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.MetDrivers: TransportBinaryReader

const DEFAULT_TARGET_KG = 5.135313897e18

function _parse_cli(args::Vector{String})
    isempty(args) &&
        error("usage: check_mass_balance_dec2021.jl <bin_dir> [--target-kg X.YZ]")
    bin_dir = expanduser(args[1])
    isdir(bin_dir) || error("not a directory: $bin_dir")
    target_kg = DEFAULT_TARGET_KG
    i = 2
    while i <= length(args)
        if args[i] == "--target-kg" && i + 1 <= length(args)
            target_kg = parse(Float64, args[i + 1]); i += 2
        else
            i += 1
        end
    end
    return (; bin_dir, target_kg)
end

# Section-size helpers — mirrored from
# `scripts/completed_experiments/compare_c180_binary_mass_fluxes.jl` so the layout walk
# is self-contained.
# Elements of one CS payload section (the library's table, which knows every section).
_cs_section_elements(h, section::Symbol) = AtmosTransport.MetDrivers._cs_section_elements(h, section)

"""Sum the `:m` payload (dry mass per cell per layer, kg) across all 6 panels
of one window without materialising the per-panel arrays beyond a single
panel-sized scratch."""
function _sum_window_mass(reader::TransportBinaryReader{FT}, win::Int) where FT
    h = reader.header
    nc, nz, np = h.geometry.Nc, h.nlevel, h.geometry.npanel
    offset = (win - 1) * h.elems_per_window
    scratch = Array{FT}(undef, nc, nc, nz)
    total = 0.0
    saw_m = false
    for section in h.payload_sections
        n = _cs_section_elements(h, section)
        if section === :m
            for p in 1:np
                copyto!(scratch, 1, reader.data, offset + 1, length(scratch))
                total += sum(Float64, scratch)
                offset += length(scratch)
            end
            saw_m = true
        else
            offset += n
        end
    end
    saw_m || error("Binary $(reader.path) missing :m payload section")
    return total
end

function _check_one_binary(path::AbstractString)
    reader = TransportBinaryReader(path)
    try
        h = reader.header
        h.mass_basis === :dry ||
            @warn "Binary `$(basename(path))` is mass_basis=$(h.mass_basis), not :dry; \
                   total-mass check still runs but its interpretation differs."
        nwin = Int(h.nwindow)
        per_win = Vector{Float64}(undef, nwin)
        for w in 1:nwin
            per_win[w] = _sum_window_mass(reader, w)
        end
        return (; path, nwin, totals_kg = per_win)
    finally
        finalize(reader)
    end
end

function main(args::Vector{String} = ARGS)
    cli = _parse_cli(args)
    @info "Mass balance check" cli.bin_dir cli.target_kg

    files = sort(filter(f -> endswith(f, ".bin"),
                        readdir(cli.bin_dir; join = true)))
    isempty(files) && error("no .bin files under $(cli.bin_dir)")
    @info "Found $(length(files)) binaries."

    println()
    @printf("%-50s  %5s  %16s  %12s  %12s  %12s\n",
            "binary", "nwin", "day mean (kg)", "drift (ppm)",
            "spread (ppm)", "max jump (ppm)")
    println(repeat('-', 122))

    all_totals = Float64[]
    per_day_means = Float64[]
    binary_summaries = Tuple{String, Vector{Float64}}[]

    for path in files
        s = _check_one_binary(path)
        day_mean = mean(s.totals_kg)
        max_minus_min = maximum(s.totals_kg) - minimum(s.totals_kg)
        adj_diff = maximum(abs.(diff(s.totals_kg)))
        drift_ppm = 1e6 * (day_mean - cli.target_kg) / cli.target_kg
        spread_ppm = 1e6 * max_minus_min / day_mean
        jump_ppm = 1e6 * adj_diff / day_mean
        @printf("%-50s  %5d  %.10e  %+12.3f  %12.3f  %12.3f\n",
                basename(path), s.nwin, day_mean, drift_ppm,
                spread_ppm, jump_ppm)
        append!(all_totals, s.totals_kg)
        push!(per_day_means, day_mean)
        push!(binary_summaries, (basename(path), s.totals_kg))
    end

    println()
    println("=== global summary ===")
    overall_mean = mean(all_totals)
    overall_spread = maximum(all_totals) - minimum(all_totals)
    day_spread = maximum(per_day_means) - minimum(per_day_means)
    @printf("Total windows checked      : %d (%d days × 24 windows expected)\n",
            length(all_totals), length(files))
    @printf("Per-window total range     : [%.10e, %.10e] kg\n",
            minimum(all_totals), maximum(all_totals))
    @printf("Per-window mean / target   : %.10e / %.10e kg\n",
            overall_mean, cli.target_kg)
    @printf("Drift of per-window mean   : %+.3f ppm\n",
            1e6 * (overall_mean - cli.target_kg) / cli.target_kg)
    @printf("Largest per-window spread  : %.3e (= %.3f ppm of mean)\n",
            overall_spread, 1e6 * overall_spread / overall_mean)
    @printf("Largest day-mean spread    : %.3e (= %.3f ppm of mean)\n",
            day_spread, 1e6 * day_spread / overall_mean)

    closure_threshold_ppm = 1.0
    spread_ppm = 1e6 * overall_spread / overall_mean
    if spread_ppm < closure_threshold_ppm
        @printf("\n✓ Mass balance closed: %.3f ppm spread across all windows (< %.1f ppm threshold).\n",
                spread_ppm, closure_threshold_ppm)
    else
        @printf("\n✗ Mass balance NOT closed: %.3f ppm spread exceeds %.1f ppm threshold.\n",
                spread_ppm, closure_threshold_ppm)
    end
    return (; all_totals, per_day_means, binary_summaries, target_kg = cli.target_kg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
