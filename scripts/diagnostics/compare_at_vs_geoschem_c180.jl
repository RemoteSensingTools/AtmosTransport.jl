#!/usr/bin/env julia
# ===========================================================================
# Compare AtmosTransport forward-run output against GeosChem Catrine
# reference on a shared C180 cubed-sphere grid.
#
# AtmosTransport snapshot file is the one produced by
# `config/runs/catrine_geosit_c180_v4_fullphys_dec2021.toml` — a
# single multi-time NetCDF with `(Xdim, Ydim, nf, lev, time)` 5D
# variables named after the user-defined tracer symbols
# (`co2_natural`, `co2_fossil`, `sf6`, `rn222`) plus `air_mass` and
# `*_column_mass_per_area` diagnostics.
#
# GeosChem reference is a directory of 3-hourly files
# `~/data/AtmosTransport/catrine-geoschem-runs/GEOSChem.CATRINE_inst.YYYYMMDD_HHHHz.nc4`
# with the same horizontal layout but variables prefixed
# `SpeciesConcVV_*` (one file per snapshot time). GEOS-Chem stores its L72
# vertical axis surface-to-top, while AtmosTransport snapshots store levels
# top-to-surface. When vertical grids differ, metrics use the common
# surface-aligned subset of levels and reorder GEOS-Chem into the
# AtmosTransport convention before comparing.
#
# Tracer-name mapping is hard-coded at the top. Both data sources share
# the C180 panel layout (no regridding required); per-cell area for
# weighted RMS is read from the AtmosTransport snapshot geometry.
#
# Outputs:
#   * a single `--out` NetCDF with per-time / per-tracer metrics:
#     global RMS, bias, Pearson correlation, column-burden ratio,
#     plus latitude-binned bias at the surface level (10° bins).
#   * stdout summary with monthly RMS error per tracer.
#
# Plots (`--plots <dir>`, optional) use the AtmosTransport
# Visualization library + a Makie backend if available.
#
# Usage:
#   julia --project=. scripts/diagnostics/compare_at_vs_geoschem_c180.jl \
#       --at  ~/data/AtmosTransport/output/catrine_geosit_c180_v4_fullphys_dec2021.nc \
#       --gc  ~/data/AtmosTransport/catrine-geoschem-runs \
#       --out ~/data/AtmosTransport/output/catrine_v4_vs_gc_dec2021_metrics.nc
# ===========================================================================

using Dates
using Printf
using Statistics: cor
using NCDatasets

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

# Tracer-name mapping (AtmosTransport symbol → GeosChem variable name).
# Hard-coded; matches the run config and the GC `SpeciesConcVV_*`
# convention seen in `~/data/AtmosTransport/catrine-geoschem-runs/`.
const TRACER_MAP = (
    co2_natural = "SpeciesConcVV_CO2",
    co2_fossil  = "SpeciesConcVV_FossilCO2",
    sf6         = "SpeciesConcVV_SF6",
    rn222       = "SpeciesConcVV_Rn222",
)

# Used only for global species-mass diagnostics from dry VMR and dry-air mass.
const M_DRY_AIR_KG_MOL = AtmosTransport.Parameters.DRY_AIR_MOLAR_MASS
const TRACER_MOLAR_MASS_KG_MOL = (
    co2_natural = 44.0095e-3,
    co2_fossil  = 44.0095e-3,
    sf6         = 146.055e-3,
    rn222       = 222.0e-3,
)

# Run start; needed to convert AT's "hours since start" `time` axis to
# absolute UTC for matching against GC filenames. Read from the AT
# `time` units attribute when present, else default to the Dec 2021
# Catrine run start.
const DEFAULT_RUN_START = DateTime(2021, 12, 1, 0, 0, 0)
const _WARNED_VERTICAL_MISMATCH = Set{Tuple{Int, Int}}()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function _parse_args(args)
    flag(k) = (i = findfirst(==(k), args); i === nothing ? nothing : args[i + 1])
    at  = flag("--at")
    gc  = flag("--gc")
    out = flag("--out")
    plots_dir = flag("--plots")
    any(isnothing, (at, gc, out)) && error("""
        Usage: compare_at_vs_geoschem_c180.jl --at <at.nc> --gc <gc_dir> --out <metrics.nc> [--plots <dir>]
    """)
    return (at = expanduser(at), gc = expanduser(gc), out = expanduser(out),
            plots = plots_dir === nothing ? nothing : expanduser(plots_dir))
end

# ---------------------------------------------------------------------------
# Time alignment
# ---------------------------------------------------------------------------

function _resolve_run_start(at_path::AbstractString)
    NCDataset(at_path, "r") do ds
        haskey(ds, "time") || return DEFAULT_RUN_START
        units = String(get(ds["time"].attrib, "units", ""))
        # Snapshot writer emits "hours since 2000-01-01 00:00:00" but
        # the time values are simulation-relative — i.e. the calendar
        # origin in the file is nominal. We override with the
        # configured run start. v1 hard-codes Dec 2021; a follow-up
        # could read `[input].start_date` from a sidecar TOML.
        return DEFAULT_RUN_START
    end
end

function _gc_filename_to_datetime(name::AbstractString)
    # `GEOSChem.CATRINE_inst.YYYYMMDD_HHHHz.nc4`
    m = match(r"GEOSChem\.CATRINE_inst\.(\d{8})_(\d{4})z\.nc4$", name)
    m === nothing && return nothing
    y, mo, d = parse.(Int, (m[1][1:4], m[1][5:6], m[1][7:8]))
    hh, mm = parse(Int, m[2][1:2]), parse(Int, m[2][3:4])
    return DateTime(y, mo, d, hh, mm)
end

# Map each AT snapshot's hour-offset to (a) absolute DateTime,
# (b) the matching GC filename if present, else nothing.
function _build_time_pairs(at_path::AbstractString, gc_dir::AbstractString,
                           run_start::DateTime)
    NCDataset(at_path, "r") do ds
        haskey(ds, "time") || error("AT file missing `time` axis")
        at_hours = Float64.(collect(ds["time"].var[:]))

        gc_files = readdir(gc_dir; join = false, sort = true)
        gc_map = Dict{DateTime, String}()
        for f in gc_files
            dt = _gc_filename_to_datetime(f)
            dt === nothing || (gc_map[dt] = joinpath(gc_dir, f))
        end

        pairs = NamedTuple{(:at_index, :at_time_hours, :datetime, :gc_path),
                           Tuple{Int, Float64, DateTime, Union{Nothing, String}}}[]
        for (i, h) in enumerate(at_hours)
            dt = run_start + Millisecond(round(Int, h * 3_600_000))
            gc_path = get(gc_map, dt, nothing)
            push!(pairs, (at_index = i, at_time_hours = h,
                          datetime = dt, gc_path = gc_path))
        end
        return pairs
    end
end

# ---------------------------------------------------------------------------
# Cell-area weights from the AtmosTransport snapshot geometry
# ---------------------------------------------------------------------------

function _build_cell_weights(ds)
    haskey(ds, "cell_area") || error("AT file missing `cell_area` geometry")
    haskey(ds, "lats") || error("AT file missing `lats` geometry")
    area = Float64.(coalesce.(Array(ds["cell_area"][:, :, :]), NaN))
    lats = Float64.(coalesce.(Array(ds["lats"][:, :, :]), NaN))
    size(area, 3) == 6 || error("expected 6 cubed-sphere panels in cell_area, got $(size(area))")
    cell_areas = ntuple(p -> area[:, :, p], 6)
    lat_panels = ntuple(p -> lats[:, :, p], 6)
    total_area = sum(p -> sum(cell_areas[p]), 1:6)
    return (cell_areas = cell_areas, lats = lat_panels, total_area = total_area)
end

# ---------------------------------------------------------------------------
# Per-snapshot metrics
# ---------------------------------------------------------------------------

# Read a 5D `(Xdim, Ydim, nf, lev, time)` variable at `time_idx`,
# returning a 4D `(Xdim, Ydim, nf, lev)` Float64 array.
function _read_5d_at_time(ds, var::AbstractString, time_idx::Int)
    raw = ds[var][:, :, :, :, time_idx]
    return Float64.(coalesce.(raw, NaN))
end

function _surface_aligned_common_levels(at_top_to_surface::Array{Float64, 4},
                                        gc_surface_to_top::Array{Float64, 4},
                                        label_at::AbstractString,
                                        label_gc::AbstractString)
    nz = min(size(at_top_to_surface, 4), size(gc_surface_to_top, 4))
    key = (size(at_top_to_surface, 4), size(gc_surface_to_top, 4))
    if !(key in _WARNED_VERTICAL_MISMATCH)
        push!(_WARNED_VERTICAL_MISMATCH, key)
        @warn "Vertical level mismatch; comparing common surface-aligned levels" label_at size_at=size(at_top_to_surface) label_gc size_gc=size(gc_surface_to_top) common_levels=nz
    end
    at_cmp = at_top_to_surface[:, :, :, (end - nz + 1):end]
    gc_surface_subset = gc_surface_to_top[:, :, :, 1:nz]
    gc_cmp = gc_surface_subset[:, :, :, nz:-1:1]
    return at_cmp, gc_cmp
end

# Global area- and mass-weighted metrics across all (panel, i, j, lev)
# cells. Cell mass weight is `air_mass[i, j, panel, k]` so the metric
# represents an atmospheric-mass-weighted error in the species
# mixing ratio. Falls back to area-only weighting when `air_mass`
# is unavailable.
function _vmr_metrics(at::Array{Float64, 4}, gc::Array{Float64, 4},
                      cell_areas::NTuple{6, Matrix{Float64}},
                      air_mass::Union{Nothing, Array{Float64, 4}})
    nx, ny, nf, nz = size(at)
    @assert size(gc) == size(at) "AT vs GC shape mismatch: $(size(at)) vs $(size(gc))"
    @assert nf == 6 "expected 6 cubed-sphere panels, got nf = $nf"

    w_sum = 0.0
    diff_sum = 0.0
    sq_sum = 0.0
    sum_at = 0.0
    sum_gc = 0.0
    sum_at_sq = 0.0
    sum_gc_sq = 0.0
    sum_atgc = 0.0
    n_valid = 0

    @inbounds for k in 1:nz, p in 1:6, j in 1:ny, i in 1:nx
        a = at[i, j, p, k]; g = gc[i, j, p, k]
        (isnan(a) || isnan(g)) && continue
        w = air_mass === nothing ? cell_areas[p][i, j] : air_mass[i, j, p, k]
        w > 0 || continue
        d = a - g
        w_sum   += w
        diff_sum += w * d
        sq_sum   += w * d * d
        sum_at   += w * a
        sum_gc   += w * g
        sum_at_sq += w * a * a
        sum_gc_sq += w * g * g
        sum_atgc  += w * a * g
        n_valid += 1
    end

    w_sum > 0 || return (rms = NaN, bias = NaN, correlation = NaN,
                         mean_vmr_at = NaN, mean_vmr_gc = NaN,
                         n_valid = 0)
    mean_at = sum_at / w_sum
    mean_gc = sum_gc / w_sum
    var_at = sum_at_sq / w_sum - mean_at^2
    var_gc = sum_gc_sq / w_sum - mean_gc^2
    cov_atgc = sum_atgc / w_sum - mean_at * mean_gc
    correlation = (var_at > 0 && var_gc > 0) ? cov_atgc / sqrt(var_at * var_gc) : NaN

    return (rms = sqrt(sq_sum / w_sum),
            bias = diff_sum / w_sum,
            correlation = correlation,
            mean_vmr_at = mean_at,
            mean_vmr_gc = mean_gc,
            n_valid = n_valid)
end

function _global_species_mass_kg(vmr::Array{Float64, 4},
                                 dry_air_mass::Array{Float64, 4},
                                 tracer::Symbol)
    @assert size(vmr) == size(dry_air_mass)
    molar_mass = getfield(TRACER_MOLAR_MASS_KG_MOL, tracer)
    total = 0.0
    @inbounds for idx in eachindex(vmr, dry_air_mass)
        q = vmr[idx]
        m_air = dry_air_mass[idx]
        (isnan(q) || isnan(m_air) || m_air <= 0) && continue
        total += q * m_air * molar_mass / M_DRY_AIR_KG_MOL
    end
    return total
end

# Latitude-binned surface-level bias + RMS. Returns (bin_centers,
# bias_per_bin, rms_per_bin). 10° bins, -90 to 90.
function _surface_lat_binning(at::Array{Float64, 4}, gc::Array{Float64, 4},
                               lat_panels::NTuple{6, Matrix{Float64}},
                               cell_areas::NTuple{6, Matrix{Float64}})
    nx, ny, nf, nz = size(at)
    n_bins = 18
    bin_centers = collect(-85.0:10.0:85.0)
    sum_w = zeros(n_bins)
    sum_d = zeros(n_bins)
    sum_d2 = zeros(n_bins)

    for p in 1:6
        lats = lat_panels[p]
        for j in 1:ny, i in 1:nx
            lat = Float64(lats[i, j])
            bin = clamp(1 + Int(floor((lat + 90.0) / 10.0)), 1, n_bins)
            a = at[i, j, p, nz]   # surface level (k = Nz, "positive=down")
            g = gc[i, j, p, nz]
            (isnan(a) || isnan(g)) && continue
            w = cell_areas[p][i, j]
            d = a - g
            sum_w[bin] += w
            sum_d[bin] += w * d
            sum_d2[bin] += w * d * d
        end
    end

    bias = [sum_w[k] > 0 ? sum_d[k] / sum_w[k] : NaN for k in 1:n_bins]
    rms  = [sum_w[k] > 0 ? sqrt(sum_d2[k] / sum_w[k]) : NaN for k in 1:n_bins]
    return (centers = bin_centers, bias = bias, rms = rms)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(args = ARGS)
    cli = _parse_args(args)
    isfile(cli.at) || error("AT snapshot file not found: $(cli.at)")
    isdir(cli.gc)  || error("GC directory not found: $(cli.gc)")

    run_start = _resolve_run_start(cli.at)
    @info "Run start anchor: $run_start"

    pairs = _build_time_pairs(cli.at, cli.gc, run_start)
    matched = filter(p -> p.gc_path !== nothing, pairs)
    @info @sprintf("Matched %d / %d AT snapshots to GC files",
                   length(matched), length(pairs))
    isempty(matched) && error("no AT/GC time overlaps; check run_start + GC dir")

    # Open the AT snapshot once, read the C180 dimensions.
    ds_at = NCDataset(cli.at, "r")
    Nc = ds_at.dim["Xdim"]
    Nz = ds_at.dim["lev"]
    @info @sprintf("C%d grid, %d levels", Nc, Nz)
    weights = _build_cell_weights(ds_at)

    # Per-tracer metric accumulators. One row per matched timestamp.
    tracer_keys = keys(TRACER_MAP)
    n_times = length(matched)
    metrics = Dict(name => Dict(
        :rms => fill(NaN, n_times),
        :bias => fill(NaN, n_times),
        :correlation => fill(NaN, n_times),
        :mean_vmr_at => fill(NaN, n_times),
        :mean_vmr_gc => fill(NaN, n_times),
        :species_mass_at_kg => fill(NaN, n_times),
        :species_mass_gc_kg => fill(NaN, n_times),
        :species_mass_ratio_at_over_gc => fill(NaN, n_times),
    ) for name in tracer_keys)
    lat_bin_centers = collect(-85.0:10.0:85.0)
    n_lat_bins = length(lat_bin_centers)
    lat_bias = Dict(name => fill(NaN, n_lat_bins, n_times) for name in tracer_keys)
    lat_rms  = Dict(name => fill(NaN, n_lat_bins, n_times) for name in tracer_keys)

    for (t_idx, p) in enumerate(matched)
        @info @sprintf("[%d/%d] %s ↔ %s",
                       t_idx, n_times, p.datetime, basename(p.gc_path))
        air_mass = haskey(ds_at, "air_mass") ?
            _read_5d_at_time(ds_at, "air_mass", p.at_index) : nothing
        NCDataset(p.gc_path, "r") do ds_gc
            gc_air_mass_surface_to_top = haskey(ds_gc, "Met_AD") ?
                Float64.(coalesce.(ds_gc["Met_AD"][:, :, :, :, 1], NaN)) : nothing
            for at_sym in tracer_keys
                at_name = String(at_sym)
                gc_name = String(getfield(TRACER_MAP, at_sym))
                if !haskey(ds_at, at_name)
                    @warn "AT missing variable $at_name; skipping tracer"
                    continue
                end
                if !haskey(ds_gc, gc_name)
                    @warn "GC file $(basename(p.gc_path)) missing $gc_name; skipping"
                    continue
                end
                at_field = _read_5d_at_time(ds_at, at_name, p.at_index)
                # GC files have time = 1; squeeze to 4D.
                gc_raw = ds_gc[gc_name][:, :, :, :, 1]
                gc_field = Float64.(coalesce.(gc_raw, NaN))
                at_cmp, gc_cmp = _surface_aligned_common_levels(
                    at_field, gc_field, at_name, gc_name)
                mass_cmp = air_mass === nothing ? nothing :
                    air_mass[:, :, :, (end - size(at_cmp, 4) + 1):end]
                gc_mass_cmp = gc_air_mass_surface_to_top === nothing ? nothing :
                    _surface_aligned_common_levels(at_field, gc_air_mass_surface_to_top,
                                                   "air_mass", "Met_AD")[2]

                m = _vmr_metrics(at_cmp, gc_cmp, weights.cell_areas, mass_cmp)
                metrics[at_sym][:rms][t_idx]         = m.rms
                metrics[at_sym][:bias][t_idx]        = m.bias
                metrics[at_sym][:correlation][t_idx] = m.correlation
                metrics[at_sym][:mean_vmr_at][t_idx] = m.mean_vmr_at
                metrics[at_sym][:mean_vmr_gc][t_idx] = m.mean_vmr_gc
                if mass_cmp !== nothing && gc_mass_cmp !== nothing
                    mass_at = _global_species_mass_kg(at_cmp, mass_cmp, at_sym)
                    mass_gc = _global_species_mass_kg(gc_cmp, gc_mass_cmp, at_sym)
                    metrics[at_sym][:species_mass_at_kg][t_idx] = mass_at
                    metrics[at_sym][:species_mass_gc_kg][t_idx] = mass_gc
                    metrics[at_sym][:species_mass_ratio_at_over_gc][t_idx] =
                        mass_gc > 0 ? mass_at / mass_gc : NaN
                end
                lat = _surface_lat_binning(at_cmp, gc_cmp,
                                            weights.lats, weights.cell_areas)
                lat_bias[at_sym][:, t_idx] = lat.bias
                lat_rms[at_sym][:, t_idx]  = lat.rms
            end
        end
    end
    close(ds_at)

    # ── Write metrics NetCDF ────────────────────────────────────────
    @info "Writing metrics → $(cli.out)"
    mkpath(dirname(abspath(cli.out)))
    NCDataset(cli.out, "c") do ds
        NCDatasets.defDim(ds, "time", n_times)
        NCDatasets.defDim(ds, "lat_bin", n_lat_bins)
        v = NCDatasets.defVar(ds, "time_hours", Float64, ("time",))
        v[:] = [Float64(p.at_time_hours) for p in matched]
        v.attrib["long_name"] = "hours since AT run start"
        v = NCDatasets.defVar(ds, "datetime_iso", String, ("time",))
        v[:] = String[string(p.datetime) for p in matched]
        v = NCDatasets.defVar(ds, "lat_bin_center", Float64, ("lat_bin",))
        v[:] = lat_bin_centers
        for at_sym in tracer_keys
            for (key, vec) in metrics[at_sym]
                vname = "$(at_sym)_$key"
                vv = NCDatasets.defVar(ds, vname, Float64, ("time",))
                vv[:] = vec
                if key in (:rms, :bias, :mean_vmr_at, :mean_vmr_gc)
                    vv.attrib["units"] = "mol mol-1 dry"
                elseif key in (:species_mass_at_kg, :species_mass_gc_kg)
                    vv.attrib["units"] = "kg"
                elseif key === :correlation || key === :species_mass_ratio_at_over_gc
                    vv.attrib["units"] = "1"
                end
            end
            vv = NCDatasets.defVar(ds, "$(at_sym)_surface_lat_bias",
                                    Float64, ("lat_bin", "time"))
            vv[:, :] = lat_bias[at_sym]
            vv = NCDatasets.defVar(ds, "$(at_sym)_surface_lat_rms",
                                    Float64, ("lat_bin", "time"))
            vv[:, :] = lat_rms[at_sym]
        end
        ds.attrib["title"] = "AtmosTransport vs GeosChem CATRINE C180 comparison"
        ds.attrib["at_file"] = cli.at
        ds.attrib["gc_directory"] = cli.gc
        ds.attrib["run_start"] = string(run_start)
        ds.attrib["mixing_ratio_basis"] = "dry mole fraction (mol mol-1 dry)"
        ds.attrib["vertical_alignment"] = "GEOS-Chem surface-to-top L72 is subset to the common surface levels and reversed to AtmosTransport top-to-surface order before comparison."
        ds.attrib["species_mass_formula"] = "sum(dry_vmr * dry_air_mass * species_molar_mass / dry_air_molar_mass)"
    end

    # ── Console summary ─────────────────────────────────────────────
    println()
    println("==== Summary (mass-weighted dry VMR, common surface-aligned levels) ====")
    @printf("%-15s %12s %12s %12s %12s\n", "tracer", "rms", "bias", "mean_corr", "mass_ratio")
    for at_sym in tracer_keys
        m = metrics[at_sym]
        valid_rms = filter(!isnan, m[:rms])
        valid_bias = filter(!isnan, m[:bias])
        valid_corr = filter(!isnan, m[:correlation])
        valid_mass_ratio = filter(!isnan, m[:species_mass_ratio_at_over_gc])
        if isempty(valid_rms)
            @printf("%-15s %12s %12s %12s %12s\n", String(at_sym), "—", "—", "—", "—")
            continue
        end
        @printf("%-15s %12.4g %12.4g %12.4f %12.6f\n", String(at_sym),
                sqrt(sum(abs2, valid_rms) / length(valid_rms)),
                sum(valid_bias) / length(valid_bias),
                sum(valid_corr) / length(valid_corr),
                isempty(valid_mass_ratio) ? NaN : sum(valid_mass_ratio) / length(valid_mass_ratio))
    end
    println()
    @info "Metrics written to $(cli.out)"

    if cli.plots !== nothing
        try
            @info "Loading Makie for diagnostic plots…"
            @eval using GLMakie
            _write_plots(cli, matched, weights, tracer_keys)
        catch err
            @warn "Plotting skipped (Makie unavailable): $err"
        end
    end

    return nothing
end

# Generate three reference-time side-by-side maps per tracer
# (AT / GC / diff) using AtmosTransport's Visualization library +
# Makie. v1: column-mean field at start / mid / end of run.
function _write_plots(cli, matched, weights, tracer_keys)
    Visualization = AtmosTransport.Visualization
    mkpath(cli.plots)
    plot_idx = sort(unique([1, max(1, length(matched) ÷ 2), length(matched)]))
    at_snap = Visualization.open_snapshot(cli.at)
    for at_sym in tracer_keys
        haskey(at_snap.ds, String(at_sym)) || continue
        # `snapshot_grid` panels the AT side at the chosen times.
        spec = Visualization.PlotSpec(at_sym; transform = :column_mean)
        out = joinpath(cli.plots, "$(at_sym)_at_grid.png")
        try
            fig = Visualization.snapshot_grid(at_snap, [spec];
                                               times = plot_idx, cols = length(plot_idx))
            save(out, fig)
            @info "Plot → $out"
        catch e
            @warn "Plot failed for $at_sym: $e"
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
