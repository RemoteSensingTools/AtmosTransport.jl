#!/usr/bin/env julia

using Printf

include(joinpath(@__DIR__, "..", "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.MetDrivers: TransportBinaryReader, load_window!, window_count,
    mesh_definition
using .AtmosTransport.Grids: CubedSphereMesh
using .AtmosTransport.Preprocessing: sync_all_cs_boundary_mirrors!

const USAGE = """
Usage:
  julia --project=. scripts/diagnostics/heritage/check_cs_binary_seam_mirrors.jl \\
      <transport_cs.bin> [--windows 1,24,48] [--all] [--threshold 0.0]

Checks whether raw on-disk cubed-sphere horizontal mass fluxes are already
invariant under the same oriented seam-mirror operation used by preprocessing.
"""

function parse_windows(spec::AbstractString, nwindow::Int)
    wins = Int[]
    for part in split(spec, ',')
        s = strip(part)
        isempty(s) && continue
        if occursin(':', s)
            bits = split(s, ':')
            length(bits) in (2, 3) || error("Invalid window range: $s")
            lo = parse(Int, bits[1])
            step = length(bits) == 3 ? parse(Int, bits[2]) : 1
            hi = parse(Int, bits[end])
            append!(wins, lo:step:hi)
        else
            push!(wins, parse(Int, s))
        end
    end
    isempty(wins) && error("No windows selected")
    for w in wins
        1 <= w <= nwindow || error("Window $w outside 1:$nwindow")
    end
    return unique(wins)
end

function parse_args(argv::Vector{String})
    path = ""
    windows = ""
    all_windows = false
    threshold = 0.0
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--windows"
            i + 1 <= length(argv) || error("Missing value for --windows")
            windows = argv[i + 1]
            i += 2
        elseif a == "--all"
            all_windows = true
            i += 1
        elseif a == "--threshold"
            i + 1 <= length(argv) || error("Missing value for --threshold")
            threshold = parse(Float64, argv[i + 1])
            i += 2
        elseif a in ("-h", "--help")
            println(USAGE)
            exit(0)
        elseif startswith(a, "--")
            error("Unknown argument: $a\n$USAGE")
        elseif isempty(path)
            path = expanduser(a)
            i += 1
        else
            error("Unexpected positional argument: $a\n$USAGE")
        end
    end
    isempty(path) && error("Missing transport binary path\n$USAGE")
    all_windows && !isempty(windows) &&
        error("Use either --all or --windows, not both")
    return (path = path, windows = windows, all_windows = all_windows,
            threshold = threshold)
end

function max_flux_delta(original, synced)
    max_abs = 0.0
    denom = eps(Float64)
    worst = (:none, 0, 0, 0, 0, 0)
    for (label, a, b) in ((:am, original.am, synced.am), (:bm, original.bm, synced.bm))
        for p in eachindex(a)
            @inbounds for idx in CartesianIndices(a[p])
                before = Float64(a[p][idx])
                after = Float64(b[p][idx])
                err = abs(after - before)
                denom = max(denom, abs(before), abs(after))
                if err > max_abs
                    max_abs = err
                    worst = (label, p, Tuple(idx)...)
                end
            end
        end
    end
    return (max_abs = max_abs, max_rel = max_abs / denom, worst = worst)
end

function check_window(reader, mesh, win::Int)
    raw = load_window!(reader, win)
    synced_am = deepcopy(raw.am)
    synced_bm = deepcopy(raw.bm)
    sync_all_cs_boundary_mirrors!(synced_am, synced_bm, mesh.connectivity,
                                  reader.header.geometry.Nc, reader.header.nlevel)
    return max_flux_delta((am = raw.am, bm = raw.bm),
                          (am = synced_am, bm = synced_bm))
end

function main(argv = ARGS)
    opts = parse_args(Vector{String}(argv))
    reader = TransportBinaryReader(opts.path; FT = Float64)
    try
        h = reader.header
        wins = opts.all_windows ? collect(1:window_count(reader)) :
               isempty(opts.windows) ? [1, min(24, h.nwindow), min(48, h.nwindow)] :
               parse_windows(opts.windows, h.nwindow)
        wins = unique(wins)
        mesh = CubedSphereMesh(; FT = Float64, Nc = h.geometry.Nc, Hp = 0,
                               definition = mesh_definition(reader))

        @info "CS binary seam mirror validation"
        @info "  binary: $(opts.path)"
        @info @sprintf("  grid: C%d levels=%d windows=%d", h.geometry.Nc, h.nlevel, h.nwindow)
        @info "  selected windows: $(join(wins, ","))"

        global_abs = 0.0
        global_rel = 0.0
        global_win = 0
        global_worst = (:none, 0, 0, 0, 0, 0)
        for win in wins
            d = check_window(reader, mesh, win)
            @printf("window=%d max_abs=%.12e max_rel=%.12e worst=%s panel=%d i=%d j=%d k=%d\n",
                    win, d.max_abs, d.max_rel, String(d.worst[1]), d.worst[2],
                    d.worst[3], d.worst[4], d.worst[5])
            if d.max_abs > global_abs
                global_abs = d.max_abs
                global_rel = d.max_rel
                global_win = win
                global_worst = d.worst
            end
        end

        @printf("worst_window=%d max_abs=%.12e max_rel=%.12e worst=%s panel=%d i=%d j=%d k=%d\n",
                global_win, global_abs, global_rel, String(global_worst[1]),
                global_worst[2], global_worst[3], global_worst[4], global_worst[5])
        if global_abs > opts.threshold
            @error @sprintf("Seam mirror mismatch exceeds threshold %.12e", opts.threshold)
            exit(1)
        end
    finally
        close(reader)
    end
end

main()
