#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Fast q-space vs GEOS-Chem comparison animation
#
# Skips GeoMakie projections for speed — uses plain CairoMakie heatmaps.
#
# Layout: 6 panels (2 rows × 3 cols)
#   Row 1: Surface  — GC | AT | AT−GC difference
#   Row 2: ~750 hPa — GC | AT | AT−GC difference
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_qspace_vs_geoschem_fast.jl
# ---------------------------------------------------------------------------

using CairoMakie
using Dates, Printf, Statistics
include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

# ── Configuration ─────────────────────────────────────────────────────────
AT_DIR  = get(ENV, "AT_DIR",  "/temp1/catrine/output")
GC_DIR  = get(ENV, "GC_DIR",  expanduser("~/data/AtmosTransport/catrine/geos-chem"))
OUTFILE = get(ENV, "OUTFILE", "/temp1/catrine/output/qspace_vs_geoschem.gif")

AT_PATTERN = get(ENV, "AT_PATTERN", "catrine_qspace_7d")
SPECIES    = "co2_3d"
SCALE      = 1e6          # mol/mol → ppm
GC_VAR     = "SpeciesConcVV_CO2"
GC_SCALE   = 1e6

LEV_SFC  = 1
LEV_750  = 10

DATE_START = DateTime(2021, 12, 1)
DATE_END   = DateTime(2021, 12, 9)

# ── Build regrid map (2° for speed) ──────────────────────────────────────
coord_file = expanduser("~/code/gitHub/AtmosTransportModel/data/grids/cs_c180_gridspec.nc")
@info "Building 2° regrid map..."
cs_lons, cs_lats = load_cs_coordinates(coord_file)
rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=2.0, dlat=2.0)

# ── Load data ─────────────────────────────────────────────────────────────
@info "Loading AtmosTransport output..."
at_data = load_cs_daily_nc(AT_DIR, AT_PATTERN, rmap, SPECIES,
                            [LEV_SFC, LEV_750];
                            date_start=DATE_START, date_end=DATE_END,
                            scale=Float64(SCALE), label="AT q-space")

@info "Loading GEOS-Chem data..."
gc_data = load_geoschem_nc(GC_DIR, rmap, GC_VAR, [LEV_SFC, LEV_750];
                            date_start=DATE_START, date_end=DATE_END,
                            scale=Float64(GC_SCALE))

at_sfc, at_750 = at_data.fields[1], at_data.fields[2]
gc_sfc, gc_750 = gc_data.fields[1], gc_data.fields[2]
at_times = at_data.times
gc_times = gc_data.times
nt_at, nt_gc = length(at_times), length(gc_times)
@info "AT: $nt_at snapshots, GC: $nt_gc snapshots"

# ── Color ranges ──────────────────────────────────────────────────────────
ref_idx = max(1, nt_at ÷ 2)
sfc_lo = floor(Float64(quantile(vec(at_sfc[:, :, ref_idx]), 0.02)))
sfc_hi = ceil(Float64(quantile(vec(at_sfc[:, :, ref_idx]), 0.98)))
mid_lo = floor(Float64(quantile(vec(at_750[:, :, ref_idx]), 0.02)))
mid_hi = ceil(Float64(quantile(vec(at_750[:, :, ref_idx]), 0.98)))
diff_max = 15.0  # ppm, symmetric for RdBu

@info "Color ranges: surface [$sfc_lo, $sfc_hi] ppm, 750hPa [$mid_lo, $mid_hi] ppm"

# ── Nearest GC index ──────────────────────────────────────────────────────
function nearest_gc_idx(at_time, gc_times)
    isempty(gc_times) && return 1
    _, idx = findmin(abs.(Dates.value.(gc_times .- at_time)))
    return idx
end

# ── Create figure ─────────────────────────────────────────────────────────
@info "Creating animation ($nt_at frames)..."

fig = Figure(size=(1500, 700), fontsize=12)

title_obs = Observable("2021-12-01 00:00")
Label(fig[0, 1:3], title_obs; fontsize=16, font=:bold)

gc_sfc_title = Observable("GEOS-Chem sfc (Dec 01 03Z)")
gc_750_title = Observable("GEOS-Chem 750 (Dec 01 03Z)")

ax_gc_sfc = Axis(fig[1, 1]; title=gc_sfc_title, ylabel="lat", aspect=DataAspect())
ax_at_sfc = Axis(fig[1, 2]; title="AT q-space surface", aspect=DataAspect())
ax_df_sfc = Axis(fig[1, 3]; title="AT − GC surface", aspect=DataAspect())
ax_gc_750 = Axis(fig[2, 1]; title=gc_750_title, xlabel="lon", ylabel="lat", aspect=DataAspect())
ax_at_750 = Axis(fig[2, 2]; title="AT q-space ~750 hPa", xlabel="lon", aspect=DataAspect())
ax_df_750 = Axis(fig[2, 3]; title="AT − GC ~750 hPa", xlabel="lon", aspect=DataAspect())

for ax in [ax_gc_sfc, ax_at_sfc, ax_df_sfc, ax_gc_750, ax_at_750, ax_df_750]
    ax.xticklabelsize = 9; ax.yticklabelsize = 9
    xlims!(ax, -180, 180); ylims!(ax, -90, 90)
end
hidexdecorations!(ax_gc_sfc; ticks=false, grid=false)
hidexdecorations!(ax_at_sfc; ticks=false, grid=false)
hidexdecorations!(ax_df_sfc; ticks=false, grid=false)
hideydecorations!(ax_at_sfc; ticks=false, grid=false)
hideydecorations!(ax_df_sfc; ticks=false, grid=false)
hideydecorations!(ax_at_750; ticks=false, grid=false)
hideydecorations!(ax_df_750; ticks=false, grid=false)

lon = rmap.lon
lat = rmap.lat

# Initial data
gc_idx_0 = nearest_gc_idx(at_times[1], gc_times)
diff_sfc_0 = at_sfc[:, :, 1] .- gc_sfc[:, :, gc_idx_0]
diff_750_0 = at_750[:, :, 1] .- gc_750[:, :, gc_idx_0]

z_gc_sfc = Observable(gc_sfc[:, :, gc_idx_0])
z_at_sfc = Observable(at_sfc[:, :, 1])
z_df_sfc = Observable(diff_sfc_0)
z_gc_750 = Observable(gc_750[:, :, gc_idx_0])
z_at_750 = Observable(at_750[:, :, 1])
z_df_750 = Observable(diff_750_0)

cmap = :YlOrRd
cmap_diff = :RdBu

heatmap!(ax_gc_sfc, lon, lat, z_gc_sfc; colormap=cmap, colorrange=(sfc_lo, sfc_hi))
hm_sfc = heatmap!(ax_at_sfc, lon, lat, z_at_sfc; colormap=cmap, colorrange=(sfc_lo, sfc_hi))
Colorbar(fig[1, 4], hm_sfc; label="ppm", width=12, ticklabelsize=9)

heatmap!(ax_gc_750, lon, lat, z_gc_750; colormap=cmap, colorrange=(mid_lo, mid_hi))
hm_750 = heatmap!(ax_at_750, lon, lat, z_at_750; colormap=cmap, colorrange=(mid_lo, mid_hi))
Colorbar(fig[2, 4], hm_750; label="ppm", width=12, ticklabelsize=9)

# Difference panels (separate colorbars)
hm_df_sfc = heatmap!(ax_df_sfc, lon, lat, z_df_sfc; colormap=cmap_diff, colorrange=(-diff_max, diff_max))
hm_df_750 = heatmap!(ax_df_750, lon, lat, z_df_750; colormap=cmap_diff, colorrange=(-diff_max, diff_max))

# ── Animate ───────────────────────────────────────────────────────────────
framerate = min(12, max(4, nt_at ÷ 10))
frame_step = max(1, nt_at ÷ 120)
frame_indices = 1:frame_step:nt_at

@info "Recording $(length(frame_indices)) frames at $(framerate) fps → $OUTFILE"

record(fig, OUTFILE, frame_indices; framerate) do ti
    gc_idx = nearest_gc_idx(at_times[ti], gc_times)

    # Update all panels
    z_at_sfc[] = at_sfc[:, :, ti]
    z_at_750[] = at_750[:, :, ti]
    z_gc_sfc[] = gc_sfc[:, :, gc_idx]
    z_gc_750[] = gc_750[:, :, gc_idx]
    z_df_sfc[] = at_sfc[:, :, ti] .- gc_sfc[:, :, gc_idx]
    z_df_750[] = at_750[:, :, ti] .- gc_750[:, :, gc_idx]

    # Update titles with GC timestamp
    gc_ts = Dates.format(gc_times[gc_idx], "dd HH")
    gc_sfc_title[] = "GC sfc (Dec $gc_ts" * "Z)"
    gc_750_title[] = "GC 750 (Dec $gc_ts" * "Z)"

    title_obs[] = "CO₂  " * Dates.format(at_times[ti], "yyyy-mm-dd HH:MM")
end

@info "Animation saved: $OUTFILE"
