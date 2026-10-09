#!/usr/bin/env julia
# ===========================================================================
# Animation + snapshot: ERA5 merged-level (68L) vs GeosChem C180
# 3 rows × 3 cols: GC | AT | Difference  ×  Surface CO₂ | 750 hPa | XCO₂
# Uses heatmap! (no transpose issues) + proper mass-weighted XCO2
# ===========================================================================

using CairoMakie
using NCDatasets
using Dates
using Statistics
using Printf
using GeoMakie
import GeoMakie.GeometryBasics

include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

const GC_DIR  = expanduser("~/data/AtmosTransport/catrine-geoschem-runs")
const AT_FILE = expanduser("~/data/AtmosTransport/catrine/output/era5_hourly_merged_dec2021/catrine_era5_hourly.bin")
const OUT_DIR = expanduser("~/data/AtmosTransport/catrine/output/era5_hourly_merged_dec2021")
const OUT_MP4 = joinpath(OUT_DIR, "era5_merged_vs_geoschem.mp4")
const OUT_PNG = joinpath(OUT_DIR, "snapshot_day5_vs_geoschem.png")
const FPS = 8

const SIM_START  = DateTime(2021, 12, 1, 0)
const DATE_START = DateTime(2021, 12, 1, 1)   # first hourly output
const DATE_END   = DateTime(2021, 12, 5, 0)
const SNAP_DT    = DateTime(2021, 12, 4, 0)

const LEV_950_GC = 3    # ~946 hPa in GC C180
const LEV_750_GC = 15   # ~757 hPa
const LEV_950_AT = 63   # ~943 hPa in AT 68L merged
const LEV_750_AT = 50   # ~751 hPa

const GRAV = 9.81f0
const MW_RATIO = 28.97f0 / 44.01f0  # M_air / M_co2

# ---------------------------------------------------------------------------
# Downsample 720×361 → 360×180  +  lon shift 0..360 → -180..180
# ---------------------------------------------------------------------------
function downsample_shift(data::AbstractMatrix)
    Nx, Ny = size(data)
    Nx2, Ny2 = Nx ÷ 2, min(Ny, 360) ÷ 2
    out = zeros(Float32, Nx2, Ny2)
    for j in 1:Ny2, i in 1:Nx2
        i2, j2 = 2i-1, 2j-1
        out[i,j] = (data[i2,j2] + data[i2+1,j2] + data[i2,j2+1] + data[i2+1,j2+1]) / 4f0
    end
    return circshift(out, (Nx2 ÷ 2, 0))
end

# ---------------------------------------------------------------------------
# Load GeosChem (C180 → 1° LL)
# ---------------------------------------------------------------------------
function load_geoschem(gc_dir, rmap, times)
    files = sort(filter(f -> endswith(f, ".nc4") && contains(f, "CATRINE_inst"), readdir(gc_dir)))
    nt = length(times)
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    sfc  = zeros(Float32, rmap.nlon, rmap.nlat, nt)
    hpa  = zeros(Float32, rmap.nlon, rmap.nlat, nt)
    xco2 = zeros(Float32, rmap.nlon, rmap.nlat, nt)

    for (ti, tgt) in enumerate(times)
        best_f, best_dt = nothing, Inf
        for f in files
            m = match(r"(\d{8})_(\d{4})z", f)
            m === nothing && continue
            d = abs(Dates.value(DateTime(m[1]*m[2], dateformat"yyyymmddHHMM") - tgt)) / 60_000
            d < best_dt && (best_dt = d; best_f = f)
        end
        (best_f === nothing || best_dt > 120) && continue
        try
            NCDataset(joinpath(gc_dir, best_f)) do ds
                co2 = ds["SpeciesConcVV_CO2"]
                regrid_cs!(buf, Float32.(co2[:,:,:,LEV_950_GC,1]) .* 1f6, rmap)
                sfc[:,:,ti] .= buf
                regrid_cs!(buf, Float32.(co2[:,:,:,LEV_750_GC,1]) .* 1f6, rmap)
                hpa[:,:,ti] .= buf

                # Mass-weighted XCO2 using Met_AD (dry air mass per cell)
                vmr = Float32.(co2[:,:,:,:,1])
                Nc, _, Np, Nz = size(vmr)
                if haskey(ds, "Met_AD")
                    ad = Float32.(ds["Met_AD"][:,:,:,:,1])
                    xcs = zeros(Float32, Nc, Nc, Np)
                    for p in 1:Np, j in 1:Nc, i in 1:Nc
                        sm, sw = 0f0, 0f0
                        for k in 1:Nz
                            w = ad[i,j,p,k]
                            sm += vmr[i,j,p,k] * w
                            sw += w
                        end
                        xcs[i,j,p] = sw > 0f0 ? sm / sw * 1f6 : 0f0
                    end
                else
                    # Fallback: simple average
                    xcs = zeros(Float32, Nc, Nc, Np)
                    for p in 1:Np, j in 1:Nc, i in 1:Nc
                        s = 0f0; for k in 1:Nz; s += vmr[i,j,p,k]; end
                        xcs[i,j,p] = s / Nz * 1f6
                    end
                end
                regrid_cs!(buf, xcs, rmap)
                xco2[:,:,ti] .= buf
            end
        catch e
            @warn "Skip GC frame $ti: $e" maxlog=3
        end
    end
    return (; sfc, hpa, xco2)
end

# ---------------------------------------------------------------------------
# Load AT (NC → downsample + shift, XCO2 from column mass)
# ---------------------------------------------------------------------------
function load_at(path, times)
    ds = NCDataset(path)
    at_t = ds["time"][:]; Nz = ds.dim["lev"]
    lev_950 = min(LEV_950_AT, Nz)
    lev_750 = min(LEV_750_AT, Nz)

    # Map AT times to real dates
    out_interval_h = length(at_t) >= 2 ?
        Dates.value(at_t[2] - at_t[1]) / 3_600_000.0 : 6.0
    at_real_dt = [SIM_START + Millisecond(round(Int, i * out_interval_h * 3_600_000))
                  for i in 1:length(at_t)]

    nt = length(times); nlon, nlat = 360, 180
    sfc  = zeros(Float32, nlon, nlat, nt)
    hpa  = zeros(Float32, nlon, nlat, nt)
    xco2 = zeros(Float32, nlon, nlat, nt)

    has_colmass = haskey(ds, "co2_column_mass")

    for (ti, tgt) in enumerate(times)
        diffs = [abs(Dates.value(a - tgt)) / 60_000 for a in at_real_dt]
        bi = argmin(diffs); diffs[bi] > 120 && continue

        # Surface + 750 hPa from 3D field (clipped for visualization)
        co2_raw = Float32.(ds["co2_3d"][:,:,:,bi]) .* 1f6
        co2 = clamp.(co2_raw, 350f0, 500f0)
        sfc[:,:,ti]  .= downsample_shift(co2[:,:,lev_950])
        hpa[:,:,ti]  .= downsample_shift(co2[:,:,lev_750])

        # XCO2 from column mass (proper mass-weighted)
        if has_colmass
            cm = Float32.(ds["co2_column_mass"][:,:,bi])
            ps = Float32.(ds["surface_pressure"][:,:,bi])
            xco2_raw = (cm ./ (ps ./ GRAV)) .* MW_RATIO .* 1f6
            xco2[:,:,ti] .= downsample_shift(xco2_raw)
        else
            # Fallback: simple column average of clipped VMR
            Nx, Ny, Nzz = size(co2)
            xf = zeros(Float32, Nx, Ny)
            for j in 1:Ny, i in 1:Nx
                s = 0f0; for k in 1:Nzz; s += co2[i,j,k]; end
                xf[i,j] = s / Nzz
            end
            xco2[:,:,ti] .= downsample_shift(xf)
        end
    end
    close(ds)
    @info "AT: $nt steps, Nz=$Nz, lev_950=$lev_950, lev_750=$lev_750, colmass=$has_colmass"
    return (; sfc, hpa, xco2)
end

# ---------------------------------------------------------------------------
# Coastlines
# ---------------------------------------------------------------------------
function get_coastlines()
    coast = GeoMakie.coastlines()
    xs, ys = Float32[], Float32[]
    geoms = coast isa AbstractVector ? coast : coast.geometry
    for geom in geoms
        pts = GeometryBasics.coordinates(geom)
        for pt in pts
            push!(xs, Float32(pt[1])); push!(ys, Float32(pt[2]))
        end
        push!(xs, NaN32); push!(ys, NaN32)
    end
    return xs, ys
end

# ---------------------------------------------------------------------------
# Create animation using heatmap! (no transpose issues)
# ---------------------------------------------------------------------------
function animate(gc, at, times)
    nf = length(times)
    @info "Animating $nf frames"
    coast_x, coast_y = get_coastlines()

    lons = range(-179.5f0, 179.5f0; length=360)
    lats = range(-89.5f0, 89.5f0; length=180)

    sfc_rng  = (390f0, 450f0)
    hpa_rng  = (405f0, 420f0)
    xco2_rng = (406f0, 416f0)
    diff_sfc = (-8f0, 8f0)
    diff_hpa = (-5f0, 5f0)
    diff_xco2 = (-4f0, 4f0)

    fig = Figure(size=(1300, 850), fontsize=10)
    title_obs = Observable("")
    Label(fig[1, 1:4], title_obs; fontsize=14, font=:bold)

    col_labels = ["GeosChem C180", "AT ERA5 (68L)", "AT − GC"]
    row_labels = ["~950 hPa CO₂", "~750 hPa CO₂", "XCO₂"]
    ranges = [(sfc_rng, diff_sfc), (hpa_rng, diff_hpa), (xco2_rng, diff_xco2)]

    # Observables for animated data
    gc_obs  = [[Observable(gc.sfc[:,:,1]),  Observable(gc.hpa[:,:,1]),  Observable(gc.xco2[:,:,1])],
               [Observable(at.sfc[:,:,1]),  Observable(at.hpa[:,:,1]),  Observable(at.xco2[:,:,1])]]
    diff_obs = [Observable(at.sfc[:,:,1] .- gc.sfc[:,:,1]),
                Observable(at.hpa[:,:,1] .- gc.hpa[:,:,1]),
                Observable(at.xco2[:,:,1] .- gc.xco2[:,:,1])]

    cmap_co2  = Reverse(:RdYlBu)
    cmap_diff = :RdBu

    for r in 1:3
        (crng, drng) = ranges[r]
        for c in 1:3
            ttl = r == 1 ? col_labels[c] : ""
            ax = Axis(fig[r+1, c]; title=ttl, aspect=DataAspect(),
                limits=(-180, 180, -90, 90),
                xticklabelsvisible=false, yticklabelsvisible=false,
                xticksvisible=false, yticksvisible=false)
            if c <= 2
                heatmap!(ax, lons, lats, gc_obs[c][r]; colorrange=crng, colormap=cmap_co2)
            else
                heatmap!(ax, lons, lats, diff_obs[r]; colorrange=drng, colormap=cmap_diff)
            end
            lines!(ax, coast_x, coast_y; color=(:black, 0.5), linewidth=0.5)
        end
        Label(fig[r+1, 0], row_labels[r]; fontsize=11, font=:bold, rotation=π/2, tellheight=false)
        Colorbar(fig[r+1, 4], colormap=cmap_co2, limits=ranges[r][1], label="ppm", width=10)
    end

    @info "Recording $nf frames → $OUT_MP4"
    record(fig, OUT_MP4, 1:nf; framerate=FPS) do fn
        for r in 1:3
            gc_data = (gc.sfc, gc.hpa, gc.xco2)[r]
            at_data = (at.sfc, at.hpa, at.xco2)[r]
            gc_obs[1][r][] = gc_data[:,:,fn]
            gc_obs[2][r][] = at_data[:,:,fn]
            diff_obs[r][]  = at_data[:,:,fn] .- gc_data[:,:,fn]
        end
        title_obs[] = Dates.format(times[fn], "yyyy-mm-dd HH:MM") *
            " UTC — ERA5 68L vs GeosChem C180"
    end
    @info "Saved: $OUT_MP4"
end

# ---------------------------------------------------------------------------
# Save snapshot
# ---------------------------------------------------------------------------
function save_snapshot(gc, at, times, snap_dt)
    diffs_t = [abs(Dates.value(t - snap_dt)) / 60_000 for t in times]
    fi = argmin(diffs_t)
    @info "Snapshot: frame $fi ($(times[fi]))"

    coast_x, coast_y = get_coastlines()
    lons = range(-179.5f0, 179.5f0; length=360)
    lats = range(-89.5f0, 89.5f0; length=180)

    sfc_rng  = (390f0, 450f0)
    hpa_rng  = (405f0, 420f0)
    xco2_rng = (406f0, 416f0)

    fig = Figure(size=(1300, 850), fontsize=10)
    Label(fig[1, 1:4], Dates.format(times[fi], "yyyy-mm-dd HH:MM") *
          " UTC — ERA5 68L vs GeosChem C180"; fontsize=14, font=:bold)

    col_labels = ["GeosChem C180", "AT ERA5 (68L)", "AT − GC"]
    row_labels = ["~950 hPa CO₂ (ppm)", "~750 hPa CO₂ (ppm)", "XCO₂ (ppm)"]
    ranges = [sfc_rng, hpa_rng, xco2_rng]
    diff_ranges = [(-8f0, 8f0), (-5f0, 5f0), (-4f0, 4f0)]

    gc_fields  = [gc.sfc[:,:,fi],  gc.hpa[:,:,fi],  gc.xco2[:,:,fi]]
    at_fields  = [at.sfc[:,:,fi],  at.hpa[:,:,fi],  at.xco2[:,:,fi]]

    for r in 1:3
        g, a = gc_fields[r], at_fields[r]
        nlon = min(size(g,1), size(a,1))
        nlat = min(size(g,2), size(a,2))
        g2, a2 = g[1:nlon, 1:nlat], a[1:nlon, 1:nlat]
        d = a2 .- g2

        crng = ranges[r]; drng = diff_ranges[r]
        lons_r = range(-179.5f0, 179.5f0; length=nlon)
        lats_r = range(-89.5f0, 89.5f0; length=nlat)

        for (c, (dat, rng, cm)) in enumerate([
            (g2, crng, Reverse(:RdYlBu)),
            (a2, crng, Reverse(:RdYlBu)),
            (d,  drng, :RdBu)])

            ttl = r == 1 ? col_labels[c] : ""
            ax = Axis(fig[r+1, c]; title=ttl, aspect=DataAspect(),
                limits=(-180, 180, -90, 90),
                xticklabelsvisible=false, yticklabelsvisible=false,
                xticksvisible=false, yticksvisible=false)
            heatmap!(ax, lons_r, lats_r, dat; colorrange=rng, colormap=cm)
            lines!(ax, coast_x, coast_y; color=(:black, 0.5), linewidth=0.5)
        end
        Label(fig[r+1, 0], row_labels[r]; fontsize=11, font=:bold, rotation=π/2, tellheight=false)
        Colorbar(fig[r+1, 4], colormap=Reverse(:RdYlBu), limits=crng, label="ppm", width=10)

        # Stats
        r_val = cor(vec(a2), vec(g2))
        bias  = mean(d)
        rmse  = sqrt(mean(d.^2))
        Label(fig[r+1, 3], @sprintf("R=%.3f\nbias=%+.1f\nRMSE=%.1f", r_val, bias, rmse);
              fontsize=9, halign=:right, valign=:top, color=:white,
              tellwidth=false, tellheight=false, padding=(0, 5, 5, 0))
    end

    save(OUT_PNG, fig; px_per_unit=2)
    @info "Snapshot saved: $OUT_PNG"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    times = collect(DATE_START:Hour(1):DATE_END)
    @info "$(length(times)) snapshots: $(times[1]) to $(times[end])"

    gc_files = sort(filter(f -> endswith(f, ".nc4") && contains(f, "CATRINE_inst"), readdir(GC_DIR)))
    cs_lons, cs_lats = load_cs_coordinates(joinpath(GC_DIR, gc_files[1]))
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    @info "Loading GeosChem..."; gc = load_geoschem(GC_DIR, rmap, times)
    @info "Loading AT...";       at = load_at(AT_FILE, times)

    animate(gc, at, times)
    save_snapshot(gc, at, times, SNAP_DT)

    # Stats summary
    println("\n=== AT vs GC Diagnostics ===")
    nt = length(times)
    for (name, at_f, gc_f) in [("950hPa", at.sfc, gc.sfc),
                                ("750hPa", at.hpa, gc.hpa),
                                ("XCO2", at.xco2, gc.xco2)]
        for ti in [1, nt÷2, nt]
            a, g = at_f[:,:,ti], gc_f[:,:,ti]
            n = min(size(a,1),size(g,1)); m = min(size(a,2),size(g,2))
            av, gv = vec(a[1:n,1:m]), vec(g[1:n,1:m])
            r = cor(av, gv); b = mean(av .- gv)
            @info @sprintf("  %-8s t=%2d: R=%.3f bias=%+.1f ppm", name, ti, r, b)
        end
    end
end

main()
