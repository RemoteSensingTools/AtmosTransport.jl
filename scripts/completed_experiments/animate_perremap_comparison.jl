#!/usr/bin/env julia
# ===========================================================================
# Side-by-side comparison: 1 remap/hour vs 8 remaps/hour (per-substep)
#
# Layout (3 rows × 3 cols):
#   Row 1: Surface CO2    (1-remap/hr | 8-remaps/hr | Difference)
#   Row 2: Surface SF6    (1-remap/hr | 8-remaps/hr | Difference)
#   Row 3: Surface Rn222  (1-remap/hr | 8-remaps/hr | Difference)
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_perremap_comparison.jl
# ===========================================================================

using CairoMakie
using GeoMakie
using NCDatasets
using Dates
using Statistics

include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

const AT_DIR       = get(ENV, "AT_DIR", "/temp1/catrine/output")
const PAT_NOREMAP  = get(ENV, "PAT_NOREMAP",  "catrine_gchp_v4_7d")
const PAT_PERREMAP = get(ENV, "PAT_PERREMAP", "catrine_gchp_v4_7d_perremap")
const OUT_GIF      = get(ENV, "OUT_GIF", "/temp1/catrine/output/perremap_comparison.gif")
const FPS          = parse(Int, get(ENV, "FPS", "4"))

const LEV_SURFACE = 1
const DATE_START  = DateTime(2021, 12, 1, 3)
const DATE_END    = DateTime(2021, 12, 8, 21)

# ---------------------------------------------------------------------------
# Loader — returns named tuple of field arrays [nlon × nlat × nt]
# ---------------------------------------------------------------------------
function load_run(at_dir, pattern, rmap, target_times)
    # Match files that are exactly `pattern_YYYYMMDD.nc` (date follows immediately)
    daily_files = sort(filter(f -> endswith(f, ".nc") &&
                                   occursin(Regex("^" * pattern * "_\\d{8}\\.nc\$"), f),
                               readdir(at_dir)))

    nt  = length(target_times)
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    co2  = zeros(Float32, rmap.nlon, rmap.nlat, nt)
    sf6  = zeros(Float32, rmap.nlon, rmap.nlat, nt)
    rn   = zeros(Float32, rmap.nlon, rmap.nlat, nt)

    for fname in daily_files
        NCDataset(joinpath(at_dir, fname), "r") do ds
            at_times = ds["time"][:]
            for (ti, tgt) in enumerate(target_times)
                diffs = [abs(Dates.value(at_t - tgt)) for at_t in at_times]
                best_idx = argmin(diffs)
                diffs[best_idx] / 60_000 > 90 && continue

                if haskey(ds, "co2_3d")
                    data_cs = Float32.(ds["co2_3d"][:, :, :, LEV_SURFACE, best_idx]) .* 1f6
                    regrid_cs!(buf, data_cs, rmap)
                    co2[:, :, ti] .= buf
                end
                if haskey(ds, "sf6_3d")
                    data_cs = Float32.(ds["sf6_3d"][:, :, :, LEV_SURFACE, best_idx]) .* 1f12
                    regrid_cs!(buf, data_cs, rmap)
                    sf6[:, :, ti] .= buf
                end
                if haskey(ds, "rn222_3d")
                    data_cs = Float32.(ds["rn222_3d"][:, :, :, LEV_SURFACE, best_idx])
                    regrid_cs!(buf, data_cs, rmap)
                    rn[:, :, ti] .= buf
                end
            end
        end
    end
    @info "Loaded '$pattern': $nt timesteps"
    return (; co2, sf6, rn)
end

# ---------------------------------------------------------------------------
# Collect shared timesteps from the no-remap run
# ---------------------------------------------------------------------------
function collect_times(at_dir, pattern; date_start, date_end)
    files = sort(filter(f -> endswith(f, ".nc") &&
                             occursin(Regex("^" * pattern * "_\\d{8}\\.nc\$"), f),
                         readdir(at_dir)))
    times = DateTime[]
    for fname in files
        NCDataset(joinpath(at_dir, fname), "r") do ds
            for t in ds["time"][:]
                date_start <= t <= date_end && push!(times, t)
            end
        end
    end
    return sort(unique(times))
end

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
function make_animation(nr, pr, times, rmap; fps=FPS)
    nframes = length(times)
    @info "Animating $nframes frames at $fps fps → $OUT_GIF"

    lon2d, lat2d = lon_lat_meshes(rmap)

    # Color ranges
    co2_lo, co2_hi   = 395f0, 500f0
    sf6_lo, sf6_hi   = 9.5f0, 11.5f0
    rn_lo,  rn_hi    = 0f0, 2f-20

    co2_dmax  = 5f0
    sf6_dmax  = 0.3f0
    rn_dmax   = 3f-21

    fig = Figure(size=(2100, 1100), fontsize=13)

    # --- Row 1: CO2 ---
    ax_co2_nr  = GeoAxis(fig[1,1]; dest="+proj=robin", title="Surface CO₂ — 1 remap/hr")
    ax_co2_pr  = GeoAxis(fig[1,2]; dest="+proj=robin", title="Surface CO₂ — 8 remaps/hr")
    ax_co2_df  = GeoAxis(fig[1,3]; dest="+proj=robin", title="Δ CO₂ (per-step − 1/hr)")

    # --- Row 2: SF6 ---
    ax_sf6_nr  = GeoAxis(fig[2,1]; dest="+proj=robin", title="Surface SF₆ — 1 remap/hr [ppt]")
    ax_sf6_pr  = GeoAxis(fig[2,2]; dest="+proj=robin", title="Surface SF₆ — 8 remaps/hr [ppt]")
    ax_sf6_df  = GeoAxis(fig[2,3]; dest="+proj=robin", title="Δ SF₆ (per-step − 1/hr) [ppt]")

    # --- Row 3: Rn222 ---
    ax_rn_nr   = GeoAxis(fig[3,1]; dest="+proj=robin", title="Surface ²²²Rn — 1 remap/hr")
    ax_rn_pr   = GeoAxis(fig[3,2]; dest="+proj=robin", title="Surface ²²²Rn — 8 remaps/hr")
    ax_rn_df   = GeoAxis(fig[3,3]; dest="+proj=robin", title="Δ ²²²Rn (per-step − 1/hr)")

    # Observables
    z_co2_nr  = Observable(nr.co2[:,:,1]')
    z_co2_pr  = Observable(pr.co2[:,:,1]')
    z_co2_df  = Observable((pr.co2[:,:,1] .- nr.co2[:,:,1])')

    z_sf6_nr  = Observable(nr.sf6[:,:,1]')
    z_sf6_pr  = Observable(pr.sf6[:,:,1]')
    z_sf6_df  = Observable((pr.sf6[:,:,1] .- nr.sf6[:,:,1])')

    z_rn_nr   = Observable(nr.rn[:,:,1]')
    z_rn_pr   = Observable(pr.rn[:,:,1]')
    z_rn_df   = Observable((pr.rn[:,:,1] .- nr.rn[:,:,1])')

    cm_seq  = Reverse(:RdYlBu)
    cm_rn   = :YlOrRd
    cm_diff = :RdBu

    sf_co2_nr = surface!(ax_co2_nr, lon2d, lat2d, z_co2_nr; shading=NoShading,
        colormap=cm_seq, colorrange=(co2_lo, co2_hi))
    surface!(ax_co2_pr, lon2d, lat2d, z_co2_pr; shading=NoShading,
        colormap=cm_seq, colorrange=(co2_lo, co2_hi))
    sf_co2_df = surface!(ax_co2_df, lon2d, lat2d, z_co2_df; shading=NoShading,
        colormap=cm_diff, colorrange=(-co2_dmax, co2_dmax))

    sf_sf6_nr = surface!(ax_sf6_nr, lon2d, lat2d, z_sf6_nr; shading=NoShading,
        colormap=cm_seq, colorrange=(sf6_lo, sf6_hi))
    surface!(ax_sf6_pr, lon2d, lat2d, z_sf6_pr; shading=NoShading,
        colormap=cm_seq, colorrange=(sf6_lo, sf6_hi))
    sf_sf6_df = surface!(ax_sf6_df, lon2d, lat2d, z_sf6_df; shading=NoShading,
        colormap=cm_diff, colorrange=(-sf6_dmax, sf6_dmax))

    sf_rn_nr  = surface!(ax_rn_nr, lon2d, lat2d, z_rn_nr; shading=NoShading,
        colormap=cm_rn, colorrange=(rn_lo, rn_hi))
    surface!(ax_rn_pr, lon2d, lat2d, z_rn_pr; shading=NoShading,
        colormap=cm_rn, colorrange=(rn_lo, rn_hi))
    sf_rn_df  = surface!(ax_rn_df, lon2d, lat2d, z_rn_df; shading=NoShading,
        colormap=cm_diff, colorrange=(-rn_dmax, rn_dmax))

    for ax in [ax_co2_nr, ax_co2_pr, ax_co2_df,
               ax_sf6_nr, ax_sf6_pr, ax_sf6_df,
               ax_rn_nr,  ax_rn_pr,  ax_rn_df]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    Colorbar(fig[1,4], sf_co2_nr; label="CO₂ [ppm]",   width=14)
    Colorbar(fig[1,5], sf_co2_df; label="Δ [ppm]",      width=14)
    Colorbar(fig[2,4], sf_sf6_nr; label="SF₆ [ppt]",   width=14)
    Colorbar(fig[2,5], sf_sf6_df; label="Δ [ppt]",      width=14)
    Colorbar(fig[3,4], sf_rn_nr;  label="²²²Rn [mol/mol]", width=14)
    Colorbar(fig[3,5], sf_rn_df;  label="Δ [mol/mol]",  width=14)

    title_obs = Observable(Dates.format(times[1], "yyyy-mm-dd HH:MM") *
        " UTC — GCHP vertical remap: 1/hr (left) vs 8/hr per-substep (right)")
    Label(fig[0, 1:5], title_obs; fontsize=17, font=:bold)

    Makie.record(fig, OUT_GIF, 1:nframes; framerate=fps) do fn
        z_co2_nr[]  = nr.co2[:,:,fn]'
        z_co2_pr[]  = pr.co2[:,:,fn]'
        z_co2_df[]  = (pr.co2[:,:,fn] .- nr.co2[:,:,fn])'
        z_sf6_nr[]  = nr.sf6[:,:,fn]'
        z_sf6_pr[]  = pr.sf6[:,:,fn]'
        z_sf6_df[]  = (pr.sf6[:,:,fn] .- nr.sf6[:,:,fn])'
        z_rn_nr[]   = nr.rn[:,:,fn]'
        z_rn_pr[]   = pr.rn[:,:,fn]'
        z_rn_df[]   = (pr.rn[:,:,fn] .- nr.rn[:,:,fn])'
        title_obs[] = Dates.format(times[fn], "yyyy-mm-dd HH:MM") *
            " UTC — GCHP vertical remap: 1/hr (left) vs 8/hr per-substep (right)"
    end
    @info "Saved: $OUT_GIF ($nframes frames)"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    gc_dir = joinpath(homedir(), "data", "AtmosTransport", "catrine-geoschem-runs")
    gc_files = sort(filter(f -> endswith(f, ".nc4"), readdir(gc_dir)))
    @info "Building CS regrid map..."
    cs_lons, cs_lats = load_cs_coordinates(joinpath(gc_dir, gc_files[1]))
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    @info "Collecting timesteps from 1-remap/hr run..."
    times = collect_times(AT_DIR, PAT_NOREMAP; date_start=DATE_START, date_end=DATE_END)
    isempty(times) && error("No output found for pattern '$PAT_NOREMAP' in $AT_DIR")
    @info "  $(length(times)) timesteps: $(times[1]) → $(times[end])"

    @info "Loading 1-remap/hr run..."
    nr = load_run(AT_DIR, PAT_NOREMAP, rmap, times)

    @info "Loading 8-remaps/hr run..."
    pr = load_run(AT_DIR, PAT_PERREMAP, rmap, times)

    make_animation(nr, pr, times, rmap)
end

main()
