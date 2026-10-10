#!/usr/bin/env julia
# Per-column active-depth histogram for the production C180 binary.
#
# Active depth = Nz - icltop + 1, where icltop is the smallest k with
# detu[k] > 0 (the cloud-top index in AtmosTransport orientation).
# This is the size of the per-column dense LU block when the matrix is
# anchored at the surface (which it is, because subsidence propagates to
# k=Nz regardless of where detu vanishes below cloud base).
#
# The script answers a single design question raised in the 2026-05-22
# convection-perf investigation: "if we use the per-column variable shift
# proposed by the user, what is the global max active depth and how many
# columns sit above each candidate cuBLAS cap (32 / 48 / 64)?"
#
# Output: prints the distribution and an artifact CSV with one row per
# (window, depth) bin.

using Printf
using Statistics

include(joinpath(@__DIR__, "..", "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.MetDrivers: TransportBinaryReader

const DEFAULT_BIN = "/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps48_v3_20260520/era5_transport_20211202_merged1000Pa_float32.bin"

# Elements of one CS payload section (the library's table, which knows every section).
_section_elements(h, section::Symbol) = AtmosTransport.MetDrivers._cs_section_elements(h, section)

function _section_offset(h, win::Int, section::Symbol)
    o = (win - 1) * h.elems_per_window
    for s in h.payload_sections
        s === section && return o
        o += _section_elements(h, s)
    end
    error("missing section: $section")
end

function _panel_view(reader, win::Int, section::Symbol, panel::Int)
    h = reader.header
    Nc, Nz = h.geometry.Nc, h.nlevel
    panel_elems = Nc * Nc * Nz
    sec_off = _section_offset(h, win, section)
    lo = sec_off + (panel - 1) * panel_elems + 1
    hi = lo + panel_elems - 1
    return reshape(@view(reader.data[lo:hi]), Nc, Nc, Nz)
end

function main()
    bin = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_BIN
    @info "Scanning per-column depth" bin
    reader = TransportBinaryReader(bin; FT = Float32)
    h = reader.header
    Nc, Nz, np, nw = h.geometry.Nc, h.nlevel, h.geometry.npanel, h.nwindow
    threshold = Float32(1e-9)

    # Histogram of active depths.  depth = Nz - icltop + 1 ∈ [1, Nz].  We
    # also count "identity" columns (icltop > Nz, i.e., detu never > 0).
    depth_hist = zeros(Int, Nz + 1)  # index = depth; depth_hist[0] handled separately
    identity_count = 0
    code_active = 0
    total_columns = 0

    for w in 1:nw, p in 1:np
        detu = _panel_view(reader, w, :detu, p)
        @inbounds for j in 1:Nc, i in 1:Nc
            first_detu = Nz + 1
            for k in 1:Nz
                if Float64(detu[i, j, k]) > threshold
                    first_detu = k; break
                end
            end
            total_columns += 1
            if first_detu > Nz
                identity_count += 1
            else
                depth = Nz - first_detu + 1
                depth_hist[depth] += 1
                code_active += 1
            end
        end
    end
    close(reader)

    @info "Scan complete" total_columns identity_count code_active

    # Cumulative coverage as a function of lmax.
    println()
    println("Per-column active-depth distribution across the whole binary")
    println("($(nw) windows × $(np) panels × $(Nc)² = $(total_columns) columns; Nz = $(Nz)).")
    println("`detu > $(threshold)` defines active.")
    println()
    @printf "Identity columns (no detu): %d (%.2f%%)\n" identity_count 100*identity_count/total_columns
    @printf "Active columns:             %d (%.2f%%)\n" code_active 100*code_active/total_columns
    println()
    println("Cumulative active-column coverage by lmax cap:")
    @printf "  %-6s %-12s %-12s %-12s\n" "lmax" "≤ lmax" "frac of all" "frac of active"
    cum_le = 0
    for lmax in (16, 24, 32, 40, 48, 56, 60, 62, 63, 64, 66, 67, 70, 73, 75, Nz)
        cum_le = sum(depth_hist[d] for d in 1:lmax)
        # All identity columns trivially fit any lmax — count them as "≤ lmax".
        cum_le_total = cum_le + identity_count
        @printf "  %-6d %-12d %-12.4f %-12.4f\n" lmax cum_le_total cum_le_total/total_columns (code_active > 0 ? cum_le/code_active : 0)
    end
    println()
    # Worst-case across all columns
    max_d = findlast(!iszero, depth_hist)
    @printf "Global max active depth: %d  (anchor: surface)\n" something(max_d, 0)
    # Sketch the histogram.
    println()
    println("Active-depth histogram (column count per bucket of 5 layers):")
    for d_lo in 1:5:Nz
        d_hi = min(d_lo + 4, Nz)
        n = sum(depth_hist[d_lo:d_hi])
        bar = "#" ^ min(60, round(Int, 60 * n / maximum(depth_hist)))
        @printf "  %2d–%2d: %8d  %s\n" d_lo d_hi n bar
    end
    mkpath("artifacts/diagnostics")
    open("artifacts/diagnostics/tm5_per_column_depth_histogram.csv", "w") do io
        println(io, "depth,column_count")
        println(io, "0,$(identity_count)")
        for d in 1:Nz
            depth_hist[d] > 0 && println(io, "$d,$(depth_hist[d])")
        end
    end
    @info "Wrote artifacts/diagnostics/tm5_per_column_depth_histogram.csv"
end

main()
