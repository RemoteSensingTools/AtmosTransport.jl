#!/usr/bin/env julia

# Diagnose cubed-sphere transport-binary Courant/CFL-style ratios by window
# and vertical layer. This is intentionally read-only: it opens one or more CS
# binaries, scans stored mass/flux/convection sections, and writes a compact
# CSV suitable for deciding whether full-L72 binaries need layer merging.

using Printf

include(joinpath(@__DIR__, "..", "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.MetDrivers: TransportBinaryReader, load_window!, load_grid

function _usage()
    println("""
    Usage:
      julia --project=. scripts/validation/heritage/diagnose_cs_courant.jl OUT.csv BIN [BIN...]

    Columns:
      path,window,level,x_ratio,y_ratio,z_ratio,cmfmc_ratio,tm5_rate_ratio,x_panel,x_i,x_j,y_panel,y_i,y_j,z_panel,z_i,z_j,cmfmc_panel,cmfmc_i,cmfmc_j,tm5_panel,tm5_i,tm5_j
    """)
end

function _max_layer_ratios(win, level::Int, cell_areas, conv_dt::Float64)
    np = length(win.m)
    x_ratio = 0.0
    x_loc = (0, 0, 0)
    y_ratio = 0.0
    y_loc = (0, 0, 0)
    z_ratio = 0.0
    z_loc = (0, 0, 0)
    cmfmc_ratio = 0.0
    cmfmc_loc = (0, 0, 0)
    tm5_ratio = 0.0
    tm5_loc = (0, 0, 0)

    for p in 1:np
        m = win.m[p]
        am = win.am[p]
        bm = win.bm[p]
        cm = win.cm[p]
        tm5 = win.tm5_fields
        Nc1, Nc2, _ = size(m)
        for j in 1:Nc2, i in 1:Nc1
            mi = Float64(m[i, j, level])
            mi > 0 || continue
            bmass = mi / Float64(cell_areas[i, j])

            outgoing_x = max(0.0, -Float64(am[i, j, level])) +
                         max(0.0,  Float64(am[i + 1, j, level]))
            rx = outgoing_x / mi
            if rx > x_ratio
                x_ratio = rx
                x_loc = (p, i, j)
            end

            outgoing_y = max(0.0, -Float64(bm[i, j, level])) +
                         max(0.0,  Float64(bm[i, j + 1, level]))
            ry = outgoing_y / mi
            if ry > y_ratio
                y_ratio = ry
                y_loc = (p, i, j)
            end

            outgoing_z = max(0.0, -Float64(cm[i, j, level])) +
                         max(0.0,  Float64(cm[i, j, level + 1]))
            rz = outgoing_z / mi
            if rz > z_ratio
                z_ratio = rz
                z_loc = (p, i, j)
            end

            if win.cmfmc !== nothing
                c = win.cmfmc[p]
                outgoing_c = max(0.0, -Float64(c[i, j, level])) +
                             max(0.0,  Float64(c[i, j, level + 1]))
                rc = outgoing_c * conv_dt / bmass
                if rc > cmfmc_ratio
                    cmfmc_ratio = rc
                    cmfmc_loc = (p, i, j)
                end
            end

            if tm5 !== nothing
                turnover = max(0.0, Float64(tm5.entu[p][i, j, level])) +
                           max(0.0, Float64(tm5.detu[p][i, j, level])) +
                           max(0.0, Float64(tm5.entd[p][i, j, level])) +
                           max(0.0, Float64(tm5.detd[p][i, j, level]))
                rt = turnover * conv_dt / bmass
                if rt > tm5_ratio
                    tm5_ratio = rt
                    tm5_loc = (p, i, j)
                end
            end
        end
    end
    return x_ratio, y_ratio, z_ratio, cmfmc_ratio, tm5_ratio,
           x_loc, y_loc, z_loc, cmfmc_loc, tm5_loc
end

function diagnose_binary!(io, path::String)
    reader = TransportBinaryReader(path; FT = Float32)
    try
        h = reader.header
        grid = load_grid(reader; FT = Float64, Hp = 0)
        cell_areas = grid.horizontal.cell_areas
        conv_dt = h.dt_met_seconds / h.steps_per_window
        for w in 1:h.nwindow
            win = load_window!(reader, w)
            for k in 1:h.nlevel
                xr, yr, zr, cr, tr, xloc, yloc, zloc, cloc, tloc =
                    _max_layer_ratios(win, k, cell_areas, conv_dt)
                @printf(io, "%s,%d,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                        path, w, k, xr, yr, zr, cr, tr,
                        xloc[1], xloc[2], xloc[3],
                        yloc[1], yloc[2], yloc[3],
                        zloc[1], zloc[2], zloc[3],
                        cloc[1], cloc[2], cloc[3],
                        tloc[1], tloc[2], tloc[3])
            end
        end
    finally
        close(reader)
    end
end

function main(args)
    length(args) >= 2 || (_usage(); exit(2))
    out = args[1]
    bins = args[2:end]
    open(out, "w") do io
        println(io, "path,window,level,x_ratio,y_ratio,z_ratio,cmfmc_ratio,tm5_rate_ratio,x_panel,x_i,x_j,y_panel,y_i,y_j,z_panel,z_i,z_j,cmfmc_panel,cmfmc_i,cmfmc_j,tm5_panel,tm5_i,tm5_j")
        for path in bins
            diagnose_binary!(io, path)
        end
    end
    @info "wrote Courant diagnostics" out bins=length(bins)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
