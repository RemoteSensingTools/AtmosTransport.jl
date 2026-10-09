#!/usr/bin/env julia
# Quick animation of LMDZ CO2 emissions from the preprocessed CS binary
# Shows first ~3 days (24 timesteps at 3-hourly cadence)

using CairoMakie
using GeoMakie
using NCDatasets
using Dates
using JSON3
using Printf

include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

const BIN_PATH = expanduser("~/data/AtmosTransport/catrine/preprocessed_c180/lmdz_co2_cs_c180_float32.bin")
const GC_DIR = joinpath(homedir(), "data", "AtmosTransport", "catrine-geoschem-runs")
const OUT_GIF = "/temp1/catrine/output/lmdz_co2_emissions_3d.gif"
const N_FRAMES = 24  # 3 days at 3-hourly = 24 frames
function load_emission_frames(bin_path, n_frames)
    io = open(bin_path, "r")
    # Read initial 4096 to find header_bytes field
    initial = read(io, 4096)
    initial_str = String(initial[1:something(findfirst(==(0x00), initial), 4097) - 1])
    m = match(r"\"header_bytes\"\s*:\s*(\d+)", initial_str)
    header_size = m !== nothing ? parse(Int, m[1]) : 4096

    # Re-read full header
    seek(io, 0)
    hdr_bytes = read(io, header_size)
    json_end = something(findfirst(==(0x00), hdr_bytes), header_size + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))

    Nc = Int(hdr.Nc)
    Nt = Int(get(hdr, :Nt, 1))
    time_hours = haskey(hdr, :time_hours) ? Float64.(hdr.time_hours) :
                 [(t - 1) * 3.0 for t in 1:Nt]

    seek(io, header_size)

    n = min(n_frames, Nt)
    buf = Array{Float32}(undef, Nc, Nc)
    frames = Vector{NTuple{6, Matrix{Float32}}}()

    for t in 1:n
        panels = ntuple(6) do _
            read!(io, buf)
            copy(buf)
        end
        push!(frames, panels)
    end
    close(io)

    @info "Loaded $n emission frames, Nc=$Nc, Nt_total=$Nt"
    return frames, time_hours[1:n], Nc
end

function main()
    # Load CS coordinates from GC reference file
    gc_files = sort(filter(f -> endswith(f, ".nc4"), readdir(GC_DIR)))
    @info "Loading CS coordinates..."
    cs_lons, cs_lats = load_cs_coordinates(joinpath(GC_DIR, gc_files[1]))
    @info "Building regrid map..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    # Load emissions
    frames, time_hours, Nc = load_emission_frames(BIN_PATH, N_FRAMES)

    # Regrid all frames
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    regridded = [zeros(Float32, rmap.nlon, rmap.nlat) for _ in 1:length(frames)]
    for (i, panels) in enumerate(frames)
        # Stack panels into (Nc, Nc, 6) for regrid_cs!
        data_cs = zeros(Float32, Nc, Nc, 6)
        for p in 1:6
            data_cs[:, :, p] .= panels[p]
        end
        # Convert kg/m²/s → µmol/m²/s for visualization
        data_cs .*= 1f6 / 44.009f-3  # kg/m²/s → mol/m²/s → µmol/m²/s
        regrid_cs!(buf, data_cs, rmap)
        regridded[i] .= buf
    end

    # Animation
    lon2d, lat2d = lon_lat_meshes(rmap)
    fig = Figure(size=(1000, 500), fontsize=14)
    ax = GeoAxis(fig[1, 1]; dest="+proj=robin", title="LMDZ CO2 Emissions")

    z = Observable(regridded[1]')
    sf = surface!(ax, lon2d, lat2d, z; shading=NoShading,
        colormap=:OrRd, colorrange=(-5f0, 30f0))
    lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    Colorbar(fig[1, 2], sf; label="CO2 flux [µmol/m²/s]", width=14)

    title_obs = Observable(@sprintf("LMDZ CO2 Emissions — t = %.1f h", time_hours[1]))
    Label(fig[0, 1:2], title_obs; fontsize=16, font=:bold)

    nframes = length(frames)
    @info "Writing $nframes frames to $OUT_GIF"
    Makie.record(fig, OUT_GIF, 1:nframes; framerate=4) do fn
        z[] = regridded[fn]'
        title_obs[] = @sprintf("LMDZ CO2 Emissions — t = %.1f h (day %.1f)", time_hours[fn], time_hours[fn]/24)
    end
    @info "Saved: $OUT_GIF"
end

main()
