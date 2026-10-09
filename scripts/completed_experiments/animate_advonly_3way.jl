#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# 3-way pure advection comparison: dry q-space | moist q-space | dry rm-space
#
# Isolates whether SH oscillations come from:
#   - dry_correction (compare col 1 vs col 2)
#   - q-space vs rm-space (compare col 1 vs col 3)
#   - advection/remap algorithm (if all 3 show it)
#
# Layout: 6 panels (2 rows × 3 cols)
#   Row 1: Surface  — q-dry | q-moist | rm-dry
#   Row 2: ~750 hPa — q-dry | q-moist | rm-dry
#
# Optional: add GC reference with AT_GC_DIR env var
# ---------------------------------------------------------------------------

using CairoMakie
using Dates, Printf, Statistics
include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

# ── Configuration ─────────────────────────────────────────────────────────
OUTDIR   = get(ENV, "OUTDIR", "/temp1/catrine/output")
GC_DIR   = get(ENV, "GC_DIR", expanduser("~/data/AtmosTransport/catrine/geos-chem"))
OUTFILE  = get(ENV, "OUTFILE", joinpath(OUTDIR, "advonly_3way_comparison.gif"))

PATTERNS = ["advonly_qspace_dry", "advonly_qspace_moist", "advonly_rmspace_dry"]
LABELS   = ["q-space dry", "q-space moist", "rm-space dry"]

SPECIES  = "co2_3d"
SCALE    = 1e6   # mol/mol → ppm
LEV_SFC  = 1
LEV_750  = 10

DATE_START = DateTime(2021, 12, 1)
DATE_END   = DateTime(2021, 12, 4)

# ── Build regrid map ─────────────────────────────────────────────────────
coord_file = expanduser("~/code/gitHub/AtmosTransportModel/data/grids/cs_c180_gridspec.nc")
@info "Building 2° regrid map..."
cs_lons, cs_lats = load_cs_coordinates(coord_file)
rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=2.0, dlat=2.0)

# ── Load all 3 datasets ──────────────────────────────────────────────────
datasets = []
for (pat, lbl) in zip(PATTERNS, LABELS)
    @info "Loading $lbl..."
    d = load_cs_daily_nc(OUTDIR, pat, rmap, SPECIES, [LEV_SFC, LEV_750];
                          date_start=DATE_START, date_end=DATE_END,
                          scale=Float64(SCALE), label=lbl)
    push!(datasets, d)
end

# Use first dataset for time axis (all should match)
times = datasets[1].times
nt = length(times)

# ── Color ranges (from first dataset, late snapshot) ──────────────────────
ref_idx = max(1, nt ÷ 2)
ref_sfc = vec(datasets[1].fields[1][:, :, ref_idx])
ref_750 = vec(datasets[1].fields[2][:, :, ref_idx])
sfc_lo = floor(Float64(quantile(ref_sfc, 0.02)))
sfc_hi = ceil(Float64(quantile(ref_sfc, 0.98)))
mid_lo = floor(Float64(quantile(ref_750, 0.02)))
mid_hi = ceil(Float64(quantile(ref_750, 0.98)))

@info "Color ranges: surface [$sfc_lo, $sfc_hi], 750hPa [$mid_lo, $mid_hi] ppm"

# ── Create figure ─────────────────────────────────────────────────────────
fig = Figure(size=(1500, 700), fontsize=12)

title_obs = Observable("2021-12-01 00:00")
Label(fig[0, 1:3], title_obs; fontsize=16, font=:bold)

lon, lat = rmap.lon, rmap.lat
cmap = :YlOrRd

# Create axes
axes_sfc = Axis[]
axes_750 = Axis[]
for (col, lbl) in enumerate(LABELS)
    ax_s = Axis(fig[1, col]; title="$lbl sfc", aspect=DataAspect())
    ax_m = Axis(fig[2, col]; title="$lbl 750", aspect=DataAspect(), xlabel="lon")
    for ax in [ax_s, ax_m]
        ax.xticklabelsize = 9; ax.yticklabelsize = 9
        xlims!(ax, -180, 180); ylims!(ax, -90, 90)
    end
    hidexdecorations!(ax_s; ticks=false, grid=false)
    col > 1 && hideydecorations!(ax_s; ticks=false, grid=false)
    col > 1 && hideydecorations!(ax_m; ticks=false, grid=false)
    col == 1 && (ax_s.ylabel = "lat"; ax_m.ylabel = "lat")
    push!(axes_sfc, ax_s)
    push!(axes_750, ax_m)
end

# Observables + heatmaps
z_sfc = [Observable(datasets[i].fields[1][:, :, 1]) for i in 1:3]
z_750 = [Observable(datasets[i].fields[2][:, :, 1]) for i in 1:3]

for i in 1:2
    heatmap!(axes_sfc[i], lon, lat, z_sfc[i]; colormap=cmap, colorrange=(sfc_lo, sfc_hi))
    heatmap!(axes_750[i], lon, lat, z_750[i]; colormap=cmap, colorrange=(mid_lo, mid_hi))
end
hm_sfc = heatmap!(axes_sfc[3], lon, lat, z_sfc[3]; colormap=cmap, colorrange=(sfc_lo, sfc_hi))
hm_750 = heatmap!(axes_750[3], lon, lat, z_750[3]; colormap=cmap, colorrange=(mid_lo, mid_hi))
Colorbar(fig[1, 4], hm_sfc; label="ppm", width=12, ticklabelsize=9)
Colorbar(fig[2, 4], hm_750; label="ppm", width=12, ticklabelsize=9)

# ── Animate ───────────────────────────────────────────────────────────────
framerate = min(10, max(4, nt ÷ 8))
frame_step = max(1, nt ÷ 120)
frame_indices = 1:frame_step:nt

@info "Recording $(length(frame_indices)) frames at $(framerate) fps → $OUTFILE"

record(fig, OUTFILE, frame_indices; framerate) do ti
    for i in 1:3
        # Find matching time index for each dataset
        ti_i = argmin(abs.(Dates.value.(datasets[i].times .- times[ti])))
        z_sfc[i][] = datasets[i].fields[1][:, :, ti_i]
        z_750[i][] = datasets[i].fields[2][:, :, ti_i]
    end
    title_obs[] = "CO₂ advection-only  " * Dates.format(times[ti], "yyyy-mm-dd HH:MM")
end

@info "Animation saved: $OUTFILE"

