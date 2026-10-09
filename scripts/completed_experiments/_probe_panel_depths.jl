#!/usr/bin/env julia
# Quick probe: dump icllfs / icltop / n_T / n_B distribution for a panel.
# Used to inform Round-2 P13 cache sizing (n_B_max) and P8 shallow-panel claim.

using Printf
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
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

function probe_panel(bin::String, win::Int, panel::Int)
    reader = TransportBinaryReader(bin; FT = Float32)
    h = reader.header
    Nc, Nz = h.geometry.Nc, h.nlevel
    detu = collect(_panel_view(reader, win, :detu, panel))
    entd = collect(_panel_view(reader, win, :entd, panel))
    close(reader)

    n_T = Int[]
    n_B = Int[]
    icltops = Int[]
    icllfss = Int[]
    no_conv_count = 0
    deep_count = 0   # active depth > 50
    shallow_count = 0  # active depth <= 25
    for j in 1:Nc, i in 1:Nc
        icltop = Nz + 1
        icllfs = Nz + 1
        for k in 1:Nz
            d = detu[i, j, k]
            if d > 0f0 && icltop == Nz + 1
                icltop = k
            end
            e = entd[i, j, k]
            if e > 0f0 && icllfs == Nz + 1
                icllfs = k
            end
        end
        if icltop > Nz
            no_conv_count += 1
            continue
        end
        icltop_eff = min(icllfs, max(icltop, 2) - 1)
        k_lo = max(icltop_eff, 1)
        icllfs_eff = max(min(icllfs, Nz + 1), k_lo)
        nt_v = icllfs_eff - k_lo
        nb_v = Nz - icllfs_eff + 1
        active_depth = Nz - k_lo + 1
        active_depth > 50 && (deep_count += 1)
        active_depth <= 25 && (shallow_count += 1)
        push!(n_T, nt_v)
        push!(n_B, nb_v)
        push!(icltops, icltop)
        push!(icllfss, icllfs)
    end
    total = Nc * Nc
    active = total - no_conv_count
    @printf "Panel (win=%d, panel=%d): Nc=%d Nz=%d total cols=%d\n" win panel Nc Nz total
    @printf "  no-conv: %d (%.1f%%)\n" no_conv_count 100*no_conv_count/total
    @printf "  active : %d (%.1f%%)\n" active 100*active/total
    if active > 0
        @printf "  shallow (active depth <= 25 layers): %d (%.1f%% of active)\n" shallow_count 100*shallow_count/active
        @printf "  deep    (active depth >  50 layers): %d (%.1f%% of active)\n" deep_count 100*deep_count/active
        @printf "  n_T  : min=%d  p50=%d  p75=%d  p95=%d  max=%d  mean=%.1f\n" minimum(n_T) Int(round(median(n_T))) Int(round(quantile(n_T, 0.75))) Int(round(quantile(n_T, 0.95))) maximum(n_T) mean(n_T)
        @printf "  n_B  : min=%d  p50=%d  p75=%d  p95=%d  max=%d  mean=%.1f\n" minimum(n_B) Int(round(median(n_B))) Int(round(quantile(n_B, 0.75))) Int(round(quantile(n_B, 0.95))) maximum(n_B) mean(n_B)
        @printf "  icltop : p50=%d  max=%d\n" Int(round(median(icltops))) maximum(icltops)
        @printf "  icllfs : p50=%d  max=%d  (Nz+1 = no downdraft)\n" Int(round(median(icllfss))) maximum(icllfss)
        @printf "  Nz+1=%d means no downdraft.  fraction columns w/ icllfs == Nz+1 (no DD): %.1f%%\n" (Nz + 1) 100*count(==(Nz+1), icllfss)/active
    end
    println()
end

function main()
    bin = DEFAULT_BIN
    args = ARGS
    pairs = Tuple{Int,Int}[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--binary"
            bin = args[i+1]; i += 2
        elseif a == "--probe"
            push!(pairs, (parse(Int, args[i+1]), parse(Int, args[i+2])))
            i += 3
        else
            error("unknown arg `$a`")
        end
    end
    if isempty(pairs)
        # Probes the panels recommended in the Round-2 prompt.
        pairs = [(1, 1), (12, 3), (18, 5), (6, 1)]
    end
    for (w, p) in pairs
        probe_panel(bin, w, p)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
