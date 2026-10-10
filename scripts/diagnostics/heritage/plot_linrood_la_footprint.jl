#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plot the 6-hour LinRood LA footprint saved by
# `scripts/diagnostics/linrood_la_footprint.jl`. Renders a global lat/lon map (CS panels
# unstitched onto a single equirectangular plot) showing log|dJ/dE|
# with the LA receptor marked.
# ---------------------------------------------------------------------------

using CairoMakie
using Printf

include(joinpath(@__DIR__, "..", "..", "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport

const FOOTPRINT_PATH = joinpath(@__DIR__, "..", "..", "..",
                                "artifacts", "linrood_la_footprint_c24_6h.bin")

function _load_footprint(path)
    open(path, "r") do io
        Nc      = read(io, Int64)
        Nz      = read(io, Int64)
        nsteps  = read(io, Int64)
        la_pan  = read(io, Int64)
        la_i    = read(io, Int64)
        la_j    = read(io, Int64)
        panels = ntuple(_ -> read!(io, Array{Float64, 2}(undef, Nc, Nc)), 6)
        return (Nc=Nc, Nz=Nz, nsteps=nsteps,
                la_panel=la_pan, la_i=la_i, la_j=la_j,
                panels=panels)
    end
end

function main()
    isfile(FOOTPRINT_PATH) || error("Footprint binary missing — run scripts/diagnostics/linrood_la_footprint.jl first")
    fp = _load_footprint(FOOTPRINT_PATH)
    @printf("Loaded C%d footprint (nsteps=%d) — LA at panel %d, (%d, %d)\n",
            fp.Nc, fp.nsteps, fp.la_panel, fp.la_i, fp.la_j)

    mesh = AT.CubedSphereMesh(Nc=fp.Nc, Hp=3, FT=Float64)

    # Gather per-cell (lon, lat, dJ/dE) for all 6 panels.
    lons_all = Float64[]
    lats_all = Float64[]
    vals_all = Float64[]
    for p in 1:6
        lons, lats = AT.Grids.panel_cell_center_lonlat(mesh, p)
        for j in 1:fp.Nc, i in 1:fp.Nc
            push!(lons_all, lons[i, j])
            push!(lats_all, lats[i, j])
            push!(vals_all, fp.panels[p][i, j])
        end
    end

    # Wrap longitudes to [-180, 180] for a more readable map.
    lons_wrapped = [l > 180 ? l - 360 : l for l in lons_all]

    abs_vals = abs.(vals_all)
    vmax = maximum(abs_vals)
    log_vals = [v > 0 ? log10(v / vmax) : -Inf for v in abs_vals]
    # Mask very weak cells for the map; keep the top ~3 decades.
    mask = log_vals .> -3
    @printf("Non-trivial cells (|dJ/dE| > vmax/1e3): %d / %d\n",
            count(mask), length(mask))
    @printf("Peak |dJ/dE| = %.4e\n", vmax)

    # Get the LA receptor lat/lon.
    la_lons, la_lats = AT.Grids.panel_cell_center_lonlat(mesh, fp.la_panel)
    la_lon = la_lons[fp.la_i, fp.la_j]
    la_lat = la_lats[fp.la_i, fp.la_j]
    la_lon > 180 && (la_lon -= 360)

    fig = Figure(size=(1400, 800))
    ax = Axis(fig[1, 1],
              title = @sprintf("6-hour LinRood adjoint footprint, LA receptor at (%.2f°, %.2f°)",
                                la_lat, la_lon),
              xlabel = "Longitude [°E]",
              ylabel = "Latitude [°N]",
              limits = (-180, 180, -90, 90))

    # Coastline-free background: just a continent box.
    sc = scatter!(ax, lons_wrapped[mask], lats_wrapped_indices(lats_all, mask);
                  color = log_vals[mask],
                  colormap = :viridis,
                  colorrange = (-3, 0),
                  markersize = 14)
    Colorbar(fig[1, 2], sc, label = "log₁₀(|dJ/dE| / max)")

    # Mark the LA receptor.
    scatter!(ax, [la_lon], [la_lat]; color = :red, markersize = 22,
             marker = :xcross, strokewidth = 2, strokecolor = :black)
    text!(ax, la_lon + 2, la_lat + 2; text = "LA receptor",
          color = :red, fontsize = 14)

    out_path = joinpath(@__DIR__, "..", "..", "..", "artifacts",
                        "linrood_la_footprint_c24_6h.png")
    save(out_path, fig; px_per_unit = 2)
    @printf("Plot saved to %s (%.1f KB)\n", out_path,
            filesize(out_path) / 1024)
end

# Mirror the latitudes via the same mask for the scatter call.
lats_wrapped_indices(lats, mask) = lats[mask]

main()
