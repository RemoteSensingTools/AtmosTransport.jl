#!/usr/bin/env julia

# Evaluate candidate vertical layer groups for a cubed-sphere transport binary.
# The script sums mass and face fluxes across each proposed native-layer group
# and reports the resulting CFL/positivity ratios by merged group.
#
# It also reports objective merge-fidelity proxies:
#   * vector_rel_rms: mass-weighted RMS spread of layer transport vectors
#     around the group mean, normalized by native-layer vector energy.
#   * angle_rms_deg: flow-weighted RMS layer/group direction error.
#   * max_internal_z: max absolute internal vertical-flux turnover within
#     the group. Internal interfaces disappear after merging, so this is a
#     direct "do not merge here if large" signal.

using Printf

include(joinpath(@__DIR__, "..", "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.MetDrivers: TransportBinaryReader, load_window!, load_grid

const USAGE = """
Usage:
  julia --project=. scripts/validation/heritage/assess_cs_merge_groups.jl OUT.csv BIN SPEC [MAX_WINDOWS]

SPEC is a comma-separated top-to-bottom list of native layer ranges.
Append xN to split a range into chunks of size N.

Examples:
  1:30x2,31:65x1,66:89x2,90:137x1
  1:24x3,25:65x1,66:89x2,90:137x1
"""

function _parse_groups(spec::AbstractString, Nz::Int)
    groups = Tuple{Int, Int}[]
    for raw_part in split(spec, ",")
        part = strip(raw_part)
        isempty(part) && continue
        m = match(r"^(\d+):(\d+)(?:x(\d+))?$", part)
        m === nothing && error("Bad SPEC part `$part`.\n$USAGE")
        a = parse(Int, m.captures[1])
        b = parse(Int, m.captures[2])
        step = m.captures[3] === nothing ? 1 : parse(Int, m.captures[3])
        1 <= a <= b <= Nz || error("SPEC part `$part` outside 1:$Nz")
        step >= 1 || error("SPEC chunk size must be positive in `$part`")
        s = a
        while s <= b
            e = min(s + step - 1, b)
            push!(groups, (s, e))
            s = e + 1
        end
    end
    isempty(groups) && error("SPEC produced no groups")

    seen = falses(Nz)
    for (a, b) in groups
        for k in a:b
            seen[k] && error("SPEC overlaps native layer $k")
            seen[k] = true
        end
    end
    all(seen) || error("SPEC does not cover all native layers 1:$Nz")
    return groups
end

@inline _centered_x_courant(am, m, i, j, k) =
    0.5 * (Float64(am[i, j, k]) + Float64(am[i + 1, j, k])) / Float64(m[i, j, k])

@inline _centered_y_courant(bm, m, i, j, k) =
    0.5 * (Float64(bm[i, j, k]) + Float64(bm[i, j + 1, k])) / Float64(m[i, j, k])

function _max_group_ratios_and_fidelity(win, k1::Int, k2::Int, cell_areas, conv_dt::Float64)
    np = length(win.m)
    x_ratio = 0.0; x_loc = (0, 0, 0)
    y_ratio = 0.0; y_loc = (0, 0, 0)
    z_ratio = 0.0; z_loc = (0, 0, 0)
    tm5_ratio = 0.0; tm5_loc = (0, 0, 0)
    vec_sse = 0.0
    vec_energy = 0.0
    angle_sse = 0.0
    angle_weight = 0.0
    internal_z_max = 0.0

    for p in 1:np
        m = win.m[p]
        am = win.am[p]
        bm = win.bm[p]
        cm = win.cm[p]
        tm5 = win.tm5_fields
        Nc1, Nc2, _ = size(m)
        for j in 1:Nc2, i in 1:Nc1
            mass = 0.0
            x_left = 0.0
            x_right = 0.0
            y_left = 0.0
            y_right = 0.0
            tm5_turnover = 0.0
            cx_mass = 0.0
            cy_mass = 0.0
            for k in k1:k2
                mk = Float64(m[i, j, k])
                mass += mk
                x_left += Float64(am[i, j, k])
                x_right += Float64(am[i + 1, j, k])
                y_left += Float64(bm[i, j, k])
                y_right += Float64(bm[i, j + 1, k])
                if mk > 0
                    cx = _centered_x_courant(am, m, i, j, k)
                    cy = _centered_y_courant(bm, m, i, j, k)
                    cx_mass += mk * cx
                    cy_mass += mk * cy
                end
                if tm5 !== nothing
                    tm5_turnover += max(0.0, Float64(tm5.entu[p][i, j, k])) +
                                     max(0.0, Float64(tm5.detu[p][i, j, k])) +
                                     max(0.0, Float64(tm5.entd[p][i, j, k])) +
                                     max(0.0, Float64(tm5.detd[p][i, j, k]))
                end
            end
            mass > 0 || continue
            bmass = mass / Float64(cell_areas[i, j])

            rx = (max(0.0, -x_left) + max(0.0, x_right)) / mass
            if rx > x_ratio
                x_ratio = rx
                x_loc = (p, i, j)
            end

            ry = (max(0.0, -y_left) + max(0.0, y_right)) / mass
            if ry > y_ratio
                y_ratio = ry
                y_loc = (p, i, j)
            end

            rz = (max(0.0, -Float64(cm[i, j, k1])) +
                  max(0.0,  Float64(cm[i, j, k2 + 1]))) / mass
            if rz > z_ratio
                z_ratio = rz
                z_loc = (p, i, j)
            end

            rt = tm5_turnover * conv_dt / bmass
            if rt > tm5_ratio
                tm5_ratio = rt
                tm5_loc = (p, i, j)
            end

            if k2 > k1
                internal_z = 0.0
                for kk in (k1 + 1):k2
                    internal_z += abs(Float64(cm[i, j, kk]))
                end
                internal_z_ratio = internal_z / mass
                internal_z_ratio > internal_z_max && (internal_z_max = internal_z_ratio)
            end

            cx_bar = cx_mass / mass
            cy_bar = cy_mass / mass
            speed_bar = hypot(cx_bar, cy_bar)
            for k in k1:k2
                mk = Float64(m[i, j, k])
                mk > 0 || continue
                cx = _centered_x_courant(am, m, i, j, k)
                cy = _centered_y_courant(bm, m, i, j, k)
                dcx = cx - cx_bar
                dcy = cy - cy_bar
                vec_sse += mk * (dcx * dcx + dcy * dcy)
                speed2 = cx * cx + cy * cy
                vec_energy += mk * speed2
                speed = sqrt(speed2)
                if speed > 1e-14 && speed_bar > 1e-14
                    c = clamp((cx * cx_bar + cy * cy_bar) / (speed * speed_bar), -1.0, 1.0)
                    angle = acos(c)
                    w = mk * speed
                    angle_sse += w * angle * angle
                    angle_weight += w
                end
            end
        end
    end
    return x_ratio, y_ratio, z_ratio, tm5_ratio,
           vec_sse, vec_energy, angle_sse, angle_weight, internal_z_max,
           x_loc, y_loc, z_loc, tm5_loc
end

_required_steps(current_steps::Int, ratio::Float64; threshold::Float64 = 0.95) =
    max(1, ceil(Int, current_steps * ratio / threshold))

function assess!(io, path::String, groups; max_windows::Union{Nothing, Int} = nothing)
    reader = TransportBinaryReader(path; FT = Float32)
    try
        h = reader.header
        grid = load_grid(reader; FT = Float64, Hp = 0)
        cell_areas = grid.horizontal.cell_areas
        conv_dt = h.dt_met_seconds / h.steps_per_window
        nw = isnothing(max_windows) ? h.nwindow : min(max_windows, h.nwindow)

        best = [
            (x = 0.0, y = 0.0, z = 0.0, tm5 = 0.0,
             vec_sse = 0.0, vec_energy = 0.0,
             angle_sse = 0.0, angle_weight = 0.0,
             internal_z = 0.0,
             wx = 0, wy = 0, wz = 0, wt = 0,
             xloc = (0, 0, 0), yloc = (0, 0, 0),
             zloc = (0, 0, 0), tloc = (0, 0, 0))
            for _ in groups
        ]

        for w in 1:nw
            win = load_window!(reader, w)
            for (gidx, (k1, k2)) in enumerate(groups)
                xr, yr, zr, tr, vec_sse, vec_energy, angle_sse, angle_weight,
                    internal_z, xloc, yloc, zloc, tloc =
                    _max_group_ratios_and_fidelity(win, k1, k2, cell_areas, conv_dt)
                cur = best[gidx]
                best[gidx] = (
                    x = max(cur.x, xr),
                    y = max(cur.y, yr),
                    z = max(cur.z, zr),
                    tm5 = max(cur.tm5, tr),
                    vec_sse = cur.vec_sse + vec_sse,
                    vec_energy = cur.vec_energy + vec_energy,
                    angle_sse = cur.angle_sse + angle_sse,
                    angle_weight = cur.angle_weight + angle_weight,
                    internal_z = max(cur.internal_z, internal_z),
                    wx = xr > cur.x ? w : cur.wx,
                    wy = yr > cur.y ? w : cur.wy,
                    wz = zr > cur.z ? w : cur.wz,
                    wt = tr > cur.tm5 ? w : cur.wt,
                    xloc = xr > cur.x ? xloc : cur.xloc,
                    yloc = yr > cur.y ? yloc : cur.yloc,
                    zloc = zr > cur.z ? zloc : cur.zloc,
                    tloc = tr > cur.tm5 ? tloc : cur.tloc,
                )
            end
            w % 4 == 0 && GC.gc(false)
        end

        println(io, "group,k_start,k_end,n_native,max_x,max_y,max_z,max_tm5_rate,min_steps_x,min_steps_y,min_steps_z,vector_rel_rms,angle_rms_deg,max_internal_z,win_x,win_y,win_z,win_tm5,x_panel,x_i,x_j,y_panel,y_i,y_j,z_panel,z_i,z_j,tm5_panel,tm5_i,tm5_j")
        for (gidx, (k1, k2)) in enumerate(groups)
            b = best[gidx]
            req_x = _required_steps(h.steps_per_window, b.x)
            req_y = _required_steps(h.steps_per_window, b.y)
            req_z = _required_steps(h.steps_per_window, b.z)
            vector_rel = b.vec_energy > 0 ? sqrt(b.vec_sse / b.vec_energy) : 0.0
            angle_deg = b.angle_weight > 0 ? sqrt(b.angle_sse / b.angle_weight) * 180.0 / pi : 0.0
            @printf(io, "%d,%d,%d,%d,%.12e,%.12e,%.12e,%.12e,%d,%d,%d,%.12e,%.12e,%.12e,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    gidx, k1, k2, k2 - k1 + 1,
                    b.x, b.y, b.z, b.tm5,
                    req_x, req_y, req_z,
                    vector_rel, angle_deg, b.internal_z,
                    b.wx, b.wy, b.wz, b.wt,
                    b.xloc[1], b.xloc[2], b.xloc[3],
                    b.yloc[1], b.yloc[2], b.yloc[3],
                    b.zloc[1], b.zloc[2], b.zloc[3],
                    b.tloc[1], b.tloc[2], b.tloc[3])
        end
    finally
        close(reader)
    end
end

function main(args)
    (3 <= length(args) <= 4) || (println(USAGE); exit(2))
    out, path, spec = args
    max_windows = length(args) == 4 ? parse(Int, args[4]) : nothing
    isnothing(max_windows) || max_windows >= 1 || error("MAX_WINDOWS must be positive")
    reader = TransportBinaryReader(path; FT = Float32)
    Nz = reader.header.nlevel
    close(reader)
    groups = _parse_groups(spec, Nz)
    mkpath(dirname(out))
    open(out, "w") do io
        assess!(io, path, groups; max_windows = max_windows)
    end
    @info "wrote CS merge-group assessment" out groups=length(groups) max_windows=max_windows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
