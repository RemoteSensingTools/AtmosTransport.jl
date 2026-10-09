#!/usr/bin/env julia

# Scan cubed-sphere transport binaries for TM5 convection active-layer depth.
# This reads only the four TM5 sections (entu, detu, entd, detd) by mmap view,
# so it can answer "how high does convection actually reach?" without loading
# the full advection payload into memory.

using Printf
using JSON3

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.MetDrivers: TransportBinaryReader

const USAGE = """
Usage:
  julia --project=. scripts/validation/diagnose_tm5_active_layers.jl OUT_PREFIX BIN [THRESHOLD] [MAX_WINDOWS]

Writes:
  OUT_PREFIX.summary.csv
  OUT_PREFIX.levels.csv

THRESHOLD defaults to 0.0.  The solver-active definition follows the
current TM5 column solver: a column is active if detu[k] > THRESHOLD for
any level k.  The any-active definition counts abs(entu/detu/entd/detd)
above threshold and is intended to catch data inconsistencies.
"""

function _on_disk_float_type(path::AbstractString)
    open(path, "r") do io
        raw = read(io, min(filesize(path), 262144))
        json_end = something(findfirst(==(0x00), raw), length(raw) + 1) - 1
        hdr = JSON3.read(String(raw[1:json_end]))
        float_bytes = Int(get(hdr, :float_bytes, 4))
        return float_bytes == 8 ? Float64 : Float32
    end
end

function _output_paths(prefix::AbstractString)
    root = endswith(lowercase(prefix), ".csv") ? prefix[1:(end - 4)] : prefix
    return root * ".summary.csv", root * ".levels.csv"
end

# Elements of one CS payload section (the library's table, which knows every section).
_section_elements(h, section::Symbol) = AtmosTransport.MetDrivers._cs_section_elements(h, section)

function _section_offset(h, win::Int, section::Symbol)
    o = (win - 1) * h.elems_per_window
    for s in h.payload_sections
        s === section && return o
        o += _section_elements(h, s)
    end
    error("CS binary is missing required section :$section")
end

function _panel_view(reader, section_offsets, section::Symbol, panel::Int)
    h = reader.header
    Nc, Nz = h.geometry.Nc, h.nlevel
    panel_elems = Nc * Nc * Nz
    lo = section_offsets[section] + (panel - 1) * panel_elems + 1
    hi = lo + panel_elems - 1
    return reshape(@view(reader.data[lo:hi]), Nc, Nc, Nz)
end

function _q(sorted_values::Vector{Int}, q::Float64)
    isempty(sorted_values) && return 0
    idx = clamp(ceil(Int, q * length(sorted_values)), 1, length(sorted_values))
    return sorted_values[idx]
end

function _scan_window(reader, win::Int, threshold::Float64)
    h = reader.header
    all(s in h.payload_sections for s in (:entu, :detu, :entd, :detd)) ||
        error("CS binary does not carry all TM5 sections (:entu, :detu, :entd, :detd)")

    Nc, Nz, np = h.geometry.Nc, h.nlevel, h.geometry.npanel
    section_offsets = Dict(s => _section_offset(h, win, s)
                           for s in (:entu, :detu, :entd, :detd))

    any_by_level = zeros(Int, Nz)
    detu_by_level = zeros(Int, Nz)
    entd_by_level = zeros(Int, Nz)
    max_abs_by_level = zeros(Float64, Nz)
    top_code = Int[]
    top_any = Int[]
    depth_code = Int[]
    depth_any = Int[]
    any_without_detu = 0
    max_abs_without_detu = 0.0

    ncols = np * Nc * Nc
    sizehint!(top_code, ncols)
    sizehint!(top_any, ncols)
    sizehint!(depth_code, ncols)
    sizehint!(depth_any, ncols)

    @inbounds for p in 1:np
        entu = _panel_view(reader, section_offsets, :entu, p)
        detu = _panel_view(reader, section_offsets, :detu, p)
        entd = _panel_view(reader, section_offsets, :entd, p)
        detd = _panel_view(reader, section_offsets, :detd, p)

        for j in 1:Nc, i in 1:Nc
            first_detu = Nz + 1
            first_any = Nz + 1
            col_max_abs = 0.0
            for k in 1:Nz
                vu = Float64(entu[i, j, k])
                vd = Float64(detu[i, j, k])
                ve = Float64(entd[i, j, k])
                vf = Float64(detd[i, j, k])
                vmax = max(abs(vu), abs(vd), abs(ve), abs(vf))
                col_max_abs = max(col_max_abs, vmax)

                detu_active = vd > threshold
                entd_active = ve > threshold
                any_active = abs(vu) > threshold ||
                             abs(vd) > threshold ||
                             abs(ve) > threshold ||
                             abs(vf) > threshold

                if detu_active
                    detu_by_level[k] += 1
                    first_detu == Nz + 1 && (first_detu = k)
                end
                if entd_active
                    entd_by_level[k] += 1
                end
                if any_active
                    any_by_level[k] += 1
                    first_any == Nz + 1 && (first_any = k)
                end
                max_abs_by_level[k] = max(max_abs_by_level[k], vmax)
            end

            if first_detu <= Nz
                push!(top_code, first_detu)
                push!(depth_code, Nz - first_detu + 1)
            end
            if first_any <= Nz
                push!(top_any, first_any)
                push!(depth_any, Nz - first_any + 1)
            end
            if first_any <= Nz && first_detu > Nz
                any_without_detu += 1
                max_abs_without_detu = max(max_abs_without_detu, col_max_abs)
            end
        end
    end

    sort!(top_code)
    sort!(top_any)
    sort!(depth_code)
    sort!(depth_any)

    return (
        ncols = ncols,
        any_by_level = any_by_level,
        detu_by_level = detu_by_level,
        entd_by_level = entd_by_level,
        max_abs_by_level = max_abs_by_level,
        code_active = length(top_code),
        any_active = length(top_any),
        any_without_detu = any_without_detu,
        max_abs_without_detu = max_abs_without_detu,
        min_top_code = _q(top_code, 0.0),
        p05_top_code = _q(top_code, 0.05),
        median_top_code = _q(top_code, 0.50),
        p95_top_code = _q(top_code, 0.95),
        max_depth_code = _q(depth_code, 1.0),
        p95_depth_code = _q(depth_code, 0.95),
        median_depth_code = _q(depth_code, 0.50),
        min_top_any = _q(top_any, 0.0),
        p05_top_any = _q(top_any, 0.05),
        median_top_any = _q(top_any, 0.50),
        p95_top_any = _q(top_any, 0.95),
        max_depth_any = _q(depth_any, 1.0),
        p95_depth_any = _q(depth_any, 0.95),
        median_depth_any = _q(depth_any, 0.50),
    )
end

function diagnose_binary!(summary_io, levels_io, path::String,
                          threshold::Float64, max_windows::Union{Nothing, Int})
    FT = _on_disk_float_type(path)
    reader = TransportBinaryReader(path; FT = FT)
    try
        h = reader.header
        nw = isnothing(max_windows) ? h.nwindow : min(max_windows, h.nwindow)
        for w in 1:nw
            stats = _scan_window(reader, w, threshold)
            @printf(summary_io,
                    "%s,%d,%.12e,%d,%d,%.12e,%d,%.12e,%d,%.12e,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    path, w, threshold, stats.ncols,
                    stats.code_active, stats.code_active / stats.ncols,
                    stats.any_active, stats.any_active / stats.ncols,
                    stats.any_without_detu, stats.max_abs_without_detu,
                    stats.min_top_code, stats.p05_top_code,
                    stats.median_top_code, stats.p95_top_code,
                    stats.max_depth_code, stats.p95_depth_code,
                    stats.median_depth_code,
                    stats.min_top_any, stats.p05_top_any,
                    stats.median_top_any, stats.p95_top_any,
                    stats.max_depth_any, stats.p95_depth_any,
                    stats.median_depth_any)

            for k in 1:h.nlevel
                @printf(levels_io,
                        "%s,%d,%d,%.12e,%d,%d,%.12e,%d,%.12e,%d,%.12e,%.12e\n",
                        path, w, k, threshold, stats.ncols,
                        stats.any_by_level[k],
                        stats.any_by_level[k] / stats.ncols,
                        stats.detu_by_level[k],
                        stats.detu_by_level[k] / stats.ncols,
                        stats.entd_by_level[k],
                        stats.entd_by_level[k] / stats.ncols,
                        stats.max_abs_by_level[k])
            end
            @info "scanned TM5 active layers" path window=w windows=nw
        end
    finally
        close(reader)
    end
end

function main(args)
    (2 <= length(args) <= 4) || (println(USAGE); exit(2))
    out_prefix = args[1]
    path = args[2]
    threshold = length(args) >= 3 ? parse(Float64, args[3]) : 0.0
    threshold >= 0 || error("THRESHOLD must be nonnegative")
    max_windows = length(args) >= 4 ? parse(Int, args[4]) : nothing
    isnothing(max_windows) || max_windows >= 1 || error("MAX_WINDOWS must be positive")

    summary_path, levels_path = _output_paths(out_prefix)
    mkpath(dirname(summary_path))
    mkpath(dirname(levels_path))
    open(summary_path, "w") do summary_io
        open(levels_path, "w") do levels_io
            println(summary_io,
                    "path,window,threshold,n_columns,code_active_columns,code_active_fraction,any_active_columns,any_active_fraction,any_without_detu_columns,max_abs_without_detu,min_top_code,p05_top_code,median_top_code,p95_top_code,max_depth_code,p95_depth_code,median_depth_code,min_top_any,p05_top_any,median_top_any,p95_top_any,max_depth_any,p95_depth_any,median_depth_any")
            println(levels_io,
                    "path,window,level,threshold,n_columns,any_columns,any_fraction,detu_columns,detu_fraction,entd_columns,entd_fraction,max_abs_rate")
            diagnose_binary!(summary_io, levels_io, path, threshold, max_windows)
        end
    end
    @info "wrote TM5 active-layer diagnostics" summary=summary_path levels=levels_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
