#!/usr/bin/env julia

using Dates
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))

using .AtmosTransport.Grids: CubedSphereMesh, LatLonMesh,
    panel_cell_local_tangent_basis
using .AtmosTransport.MetDrivers: TransportBinaryReader, mesh_definition
using .AtmosTransport.Regridding: build_regridder, apply_regridder!

const GRAV = 9.80665
const R_EARTH_M = 6.371229e6

const ERA_BIN_DIR = get(ENV, "ERA_BIN_DIR",
    "/home/cfranken/data/AtmosTransport/met/era5/cs_c180/transport_binary_v2_full137_dec2021_f32_tm5_surface_pblrename_steps24")
const GEOS_BIN_DIR = get(ENV, "GEOS_BIN_DIR",
    "/temp1/c180_geosit_native_v4_dec2021_f32")
const OUT_DIR = get(ENV, "OUT_DIR", "/temp1/c180_binary_mass_flux_audit")

const DATES = split(get(ENV, "DATES", "20211202,20211203,20211204"), ",")
const WINDOWS_SPEC = get(ENV, "WINDOWS", "all")
const TARGET_HPA = parse.(Float64, split(get(ENV, "TARGET_HPA", "850,500,250"), ","))
const LL_NX = parse(Int, get(ENV, "LL_NX", "180"))
const LL_NY = parse(Int, get(ENV, "LL_NY", "90"))
const EDGE_BAND = parse(Int, get(ENV, "EDGE_BAND", "3"))
const DX_FIELD = Symbol("\u0394x")
const DY_FIELD = Symbol("\u0394y")

struct SourceSpec
    name::String
    dir::String
    prefix::String
    suffix::String
end

const SOURCES = (
    SourceSpec("ERA5", ERA_BIN_DIR, "era5_transport_", "_merged1000Pa_float32.bin"),
    SourceSpec("GEOSIT", GEOS_BIN_DIR, "geos_transport_", "_float32.bin"),
)

function binary_path(spec::SourceSpec, date::AbstractString)
    path = joinpath(spec.dir, spec.prefix * date * spec.suffix)
    isfile(path) || error("Binary not found: $path")
    return path
end

function parse_windows(spec::AbstractString, nwindow::Int)
    spec == "all" && return collect(1:nwindow)
    if occursin(":", spec)
        a, b = parse.(Int, split(spec, ":"))
        return collect(max(1, a):min(nwindow, b))
    end
    return [parse(Int, part) for part in split(spec, ",")]
end

# Elements of one CS payload section (the library's table, which knows every section).
cs_section_elements(h, section::Symbol) = AtmosTransport.MetDrivers._cs_section_elements(h, section)

function copy_panel_section!(panels, reader, offset::Int)
    next = offset
    for p in eachindex(panels)
        n = length(panels[p])
        copyto!(panels[p], 1, reader.data, next + 1, n)
        next += n
    end
    return next
end

function load_mass_flux_window(reader::TransportBinaryReader{FT}, win::Int) where FT
    h = reader.header
    nc, nz, np = h.geometry.Nc, h.nlevel, h.geometry.npanel
    offset = (win - 1) * h.elems_per_window
    panels_m = nothing
    panels_am = nothing
    panels_bm = nothing
    panels_cm = nothing

    for section in h.payload_sections
        if section === :m
            panels_m = ntuple(_ -> Array{FT}(undef, nc, nc, nz), np)
            offset = copy_panel_section!(panels_m, reader, offset)
        elseif section === :am
            panels_am = ntuple(_ -> Array{FT}(undef, nc + 1, nc, nz), np)
            offset = copy_panel_section!(panels_am, reader, offset)
        elseif section === :bm
            panels_bm = ntuple(_ -> Array{FT}(undef, nc, nc + 1, nz), np)
            offset = copy_panel_section!(panels_bm, reader, offset)
        elseif section === :cm
            panels_cm = ntuple(_ -> Array{FT}(undef, nc, nc, nz + 1), np)
            offset = copy_panel_section!(panels_cm, reader, offset)
        else
            offset += cs_section_elements(h, section)
        end
    end

    panels_m === nothing && error("Missing m section in $(reader.path)")
    panels_am === nothing && error("Missing am section in $(reader.path)")
    panels_bm === nothing && error("Missing bm section in $(reader.path)")
    panels_cm === nothing && error("Missing cm section in $(reader.path)")
    return (; m = panels_m, am = panels_am, bm = panels_bm, cm = panels_cm)
end

function source_mesh(reader)
    return CubedSphereMesh(; FT = Float64, Nc = reader.header.geometry.Nc, Hp = 0,
        radius = R_EARTH_M, definition = mesh_definition(reader))
end

function latlon_mesh()
    return LatLonMesh(; FT = Float64, Nx = LL_NX, Ny = LL_NY,
        longitude = (-180.0, 180.0), latitude = (-90.0, 90.0),
        radius = R_EARTH_M)
end

function latlon_area_vector(mesh::LatLonMesh)
    area = Vector{Float64}(undef, mesh.Nx * mesh.Ny)
    dlon = deg2rad(360.0 / mesh.Nx)
    for j in 1:mesh.Ny
        lat_s = -90.0 + (j - 1) * 180.0 / mesh.Ny
        lat_n = -90.0 + j * 180.0 / mesh.Ny
        strip = mesh.radius^2 * dlon * (sind(lat_n) - sind(lat_s))
        for i in 1:mesh.Nx
            area[i + (j - 1) * mesh.Nx] = strip
        end
    end
    return area
end

function cs_area_vector(mesh)
    nc = mesh.Nc
    area = Vector{Float64}(undef, 6 * nc * nc)
    for p in 1:6, j in 1:nc, i in 1:nc
        area[i + (j - 1) * nc + (p - 1) * nc * nc] = Float64(mesh.cell_areas[i, j])
    end
    return area
end

function cs_edge_mask(mesh)
    nc = mesh.Nc
    mask = Vector{Bool}(undef, 6 * nc * nc)
    for p in 1:6, j in 1:nc, i in 1:nc
        edge = (i <= EDGE_BAND) || (i > nc - EDGE_BAND) ||
               (j <= EDGE_BAND) || (j > nc - EDGE_BAND)
        mask[i + (j - 1) * nc + (p - 1) * nc * nc] = edge
    end
    return mask
end

function weighted_mean(x, w)
    acc = 0.0
    wacc = 0.0
    for idx in eachindex(x, w)
        xi = Float64(x[idx])
        wi = Float64(w[idx])
        if isfinite(xi) && isfinite(wi) && wi > 0
            acc += wi * xi
            wacc += wi
        end
    end
    return wacc > 0 ? acc / wacc : NaN
end

function weighted_corr(x, y, w)
    mx = weighted_mean(x, w)
    my = weighted_mean(y, w)
    isfinite(mx) && isfinite(my) || return NaN
    cov = 0.0
    vx = 0.0
    vy = 0.0
    wacc = 0.0
    for idx in eachindex(x, y, w)
        xi = Float64(x[idx])
        yi = Float64(y[idx])
        wi = Float64(w[idx])
        if isfinite(xi) && isfinite(yi) && isfinite(wi) && wi > 0
            dx = xi - mx
            dy = yi - my
            cov += wi * dx * dy
            vx += wi * dx * dx
            vy += wi * dy * dy
            wacc += wi
        end
    end
    (wacc == 0 || vx <= 0 || vy <= 0) && return NaN
    return cov / sqrt(vx * vy)
end

function collect_weighted_values(x, w; mask = nothing, invert_mask = false)
    vals = Float64[]
    weights = Float64[]
    sizehint!(vals, length(x))
    sizehint!(weights, length(x))
    for idx in eachindex(x, w)
        if mask !== nothing
            keep = Bool(mask[idx])
            invert_mask && (keep = !keep)
            keep || continue
        end
        xi = Float64(x[idx])
        wi = Float64(w[idx])
        if isfinite(xi) && isfinite(wi) && wi > 0
            push!(vals, xi)
            push!(weights, wi)
        end
    end
    return vals, weights
end

function weighted_quantile(vals, weights, q)
    isempty(vals) && return NaN
    order = sortperm(vals)
    total = sum(weights)
    threshold = q * total
    acc = 0.0
    for idx in order
        acc += weights[idx]
        acc >= threshold && return vals[idx]
    end
    return vals[order[end]]
end

function weighted_stats(x, w; mask = nothing, invert_mask = false)
    vals, weights = collect_weighted_values(x, w; mask, invert_mask)
    isempty(vals) && return (; mean = NaN, std = NaN, min = NaN, p50 = NaN,
        p95 = NaN, p99 = NaN, max = NaN, n = 0)
    total = sum(weights)
    mean = sum(vals .* weights) / total
    var = sum(weights .* (vals .- mean).^2) / total
    return (;
        mean,
        std = sqrt(max(var, 0.0)),
        min = minimum(vals),
        p50 = weighted_quantile(vals, weights, 0.50),
        p95 = weighted_quantile(vals, weights, 0.95),
        p99 = weighted_quantile(vals, weights, 0.99),
        max = maximum(vals),
        n = length(vals),
    )
end

function paired_stats(era, geos, w)
    diff = similar(era)
    for idx in eachindex(era, geos)
        diff[idx] = geos[idx] - era[idx]
    end
    era_mean = weighted_mean(era, w)
    geos_mean = weighted_mean(geos, w)
    bias = weighted_mean(diff, w)
    mae = weighted_mean(abs.(diff), w)
    rmse = sqrt(weighted_mean(diff .* diff, w))
    corr = weighted_corr(era, geos, w)
    return (; era_mean, geos_mean, bias, mae, rmse,
        maxabs = maximum(abs.(diff)), corr)
end

function face_normal_to_geographic(up, vp, basis, panel, i, j)
    x_east, x_north, y_east, y_north = basis[panel]
    xe = Float64(x_east[i, j])
    xn = Float64(x_north[i, j])
    ye = Float64(y_east[i, j])
    yn = Float64(y_north[i, j])
    c = clamp(xe * ye + xn * yn, -1.0, 1.0)
    denom = max(1.0 - c * c, eps(Float64))
    s = sqrt(denom)
    nx_east = (xe - c * ye) / s
    nx_north = (xn - c * yn) / s
    ny_east = (ye - c * xe) / s
    ny_north = (yn - c * xn) / s
    ax = (up + c * vp) / denom
    ay = (vp + c * up) / denom
    return (;
        u_east = ax * nx_east + ay * ny_east,
        v_north = ax * nx_north + ay * ny_north,
    )
end

function speed_fields_at_pressures(window, mesh, basis, dt_factor, targets)
    nc = mesh.Nc
    nt = length(targets)
    fields = [Vector{Float64}(undef, 6 * nc * nc) for _ in targets]
    dx = getfield(mesh, DX_FIELD)
    dy = getfield(mesh, DY_FIELD)
    for p in 1:6
        m = window.m[p]
        am = window.am[p]
        bm = window.bm[p]
        nz = size(m, 3)
        for j in 1:nc, i in 1:nc
            area = Float64(mesh.cell_areas[i, j])
            ptop = 0.0
            bestdiff = fill(Inf, nt)
            bestspeed = fill(NaN, nt)
            for k in 1:nz
                mass = Float64(m[i, j, k])
                dp = mass * GRAV / area
                pmid_hpa = (ptop + 0.5 * dp) / 100.0
                speed = 0.0
                if dp > 0
                    xsec_x = Float64(dy[i, j]) * mass / area * dt_factor
                    xsec_y = Float64(dx[i, j]) * mass / area * dt_factor
                    up = xsec_x > 0 ?
                        0.5 * (Float64(am[i, j, k]) + Float64(am[i + 1, j, k])) / xsec_x :
                        0.0
                    vp = xsec_y > 0 ?
                        0.5 * (Float64(bm[i, j, k]) + Float64(bm[i, j + 1, k])) / xsec_y :
                        0.0
                    geo = face_normal_to_geographic(up, vp, basis, p, i, j)
                    speed = hypot(geo.u_east, geo.v_north)
                end
                for t in 1:nt
                    d = abs(pmid_hpa - targets[t])
                    if d < bestdiff[t]
                        bestdiff[t] = d
                        bestspeed[t] = speed
                    end
                end
                ptop += dp
            end
            idx = i + (j - 1) * nc + (p - 1) * nc * nc
            for t in 1:nt
                fields[t][idx] = bestspeed[t]
            end
        end
    end
    return fields
end

function cm_abs_rate_fields_at_pressures(window, mesh, dt_factor, targets)
    nc = mesh.Nc
    nt = length(targets)
    fields = [Vector{Float64}(undef, 6 * nc * nc) for _ in targets]
    for p in 1:6
        m = window.m[p]
        cm = window.cm[p]
        nz = size(m, 3)
        for j in 1:nc, i in 1:nc
            area = Float64(mesh.cell_areas[i, j])
            pifc_hpa = Vector{Float64}(undef, nz + 1)
            ptop = 0.0
            pifc_hpa[1] = 0.0
            for k in 1:nz
                ptop += Float64(m[i, j, k]) * GRAV / area
                pifc_hpa[k + 1] = ptop / 100.0
            end
            idx = i + (j - 1) * nc + (p - 1) * nc * nc
            for t in 1:nt
                kbest = argmin(abs.(pifc_hpa .- targets[t]))
                fields[t][idx] = abs(Float64(cm[i, j, kbest])) / (area * dt_factor)
            end
        end
    end
    return fields
end

function neighbor_roughness(map, nx::Int, ny::Int)
    vals = Float64[]
    sizehint!(vals, 2 * nx * ny)
    for j in 1:ny, i in 1:nx
        idx = i + (j - 1) * nx
        inext = i == nx ? 1 : i + 1
        push!(vals, abs(map[inext + (j - 1) * nx] - map[idx]))
        if j < ny
            push!(vals, abs(map[i + j * nx] - map[idx]))
        end
    end
    sort!(vals)
    p95 = vals[max(1, ceil(Int, 0.95 * length(vals)))]
    return (; mean_abs = sum(vals) / length(vals), p95, max = vals[end])
end

function temporal_stats(prev, cur, w)
    diff = cur .- prev
    return (;
        mean_abs = weighted_mean(abs.(diff), w),
        rmse = sqrt(weighted_mean(diff .* diff, w)),
        maxabs = maximum(abs.(diff)),
        corr = weighted_corr(prev, cur, w),
    )
end

function write_metadata(io, spec, date, path, reader)
    h = reader.header
    raw = h.raw_header
    @printf(io, "%s,%s,%s,%d,%d,%d,%g,%d,%s,%s,%s,%s,%s,%g,%s,%s,%s,%s\n",
        spec.name, date, path, h.geometry.Nc, h.geometry.npanel, h.nlevel, h.dt_met_seconds,
        h.steps_per_window, String(h.mass_basis), String(h.geometry.panel_convention),
        String(h.geometry.definition), String(h.geometry.coordinate_law), String(h.geometry.center_law),
        h.geometry.longitude_offset_deg,
        String(get(raw, "flux_kind", "")),
        String(get(raw, "source_flux_sampling", "")),
        String(get(raw, "air_mass_sampling", "")),
        join(String.(h.payload_sections), "|"))
end

function build_context(spec, sample_date, ll)
    reader = TransportBinaryReader(binary_path(spec, sample_date); FT = Float32)
    mesh = source_mesh(reader)
    println("Building common-grid regridder for $(spec.name) $(reader.header.geometry.panel_convention) $(reader.header.geometry.definition)")
    regridder = build_regridder(mesh, ll; normalize = false)
    basis = ntuple(p -> panel_cell_local_tangent_basis(mesh, p), 6)
    area = cs_area_vector(mesh)
    edge = cs_edge_mask(mesh)
    # Keep the mmap-backed reader alive until process teardown. Explicitly
    # closing the underlying IO while the mmap vector is still live can trigger
    # a Julia shutdown-time segfault on some runs.
    return (; mesh, regridder, basis, area, edge)
end

function write_worst_map(path, ll, date, win, target_hpa, era, geos)
    open(path, "w") do io
        println(io, "date,window,target_hpa,lon,lat,era_speed_mps,geosit_speed_mps,geosit_minus_era_mps")
        for j in 1:ll.Ny, i in 1:ll.Nx
            idx = i + (j - 1) * ll.Nx
            lon = -180.0 + (i - 0.5) * 360.0 / ll.Nx
            lat = -90.0 + (j - 0.5) * 180.0 / ll.Ny
            @printf(io, "%s,%d,%.1f,%.6f,%.6f,%.9g,%.9g,%.9g\n",
                date, win, target_hpa, lon, lat, era[idx], geos[idx], geos[idx] - era[idx])
        end
    end
end

function main()
    mkpath(OUT_DIR)
    ll = latlon_mesh()
    ll_area = latlon_area_vector(ll)
    contexts = Dict(spec.name => build_context(spec, first(DATES), ll) for spec in SOURCES)

    metadata = open(joinpath(OUT_DIR, "binary_metadata.csv"), "w")
    global_speed = open(joinpath(OUT_DIR, "global_speed_stats.csv"), "w")
    edge_speed = open(joinpath(OUT_DIR, "edge_vs_interior_speed_stats.csv"), "w")
    vertical_cm = open(joinpath(OUT_DIR, "vertical_cm_abs_rate_stats.csv"), "w")
    roughness = open(joinpath(OUT_DIR, "common_grid_speed_roughness.csv"), "w")
    paired = open(joinpath(OUT_DIR, "common_grid_era_geosit_speed_metrics.csv"), "w")
    temporal = open(joinpath(OUT_DIR, "common_grid_temporal_speed_jumps.csv"), "w")

    try
        println(metadata, "source,date,path,Nc,npanel,Nz,dt_met_seconds,steps_per_window,mass_basis,panel_convention,cs_definition,coordinate_law,center_law,longitude_offset_deg,flux_kind,source_flux_sampling,air_mass_sampling,payload_sections")
        println(global_speed, "source,date,window,target_hpa,mean_mps,std_mps,min_mps,p50_mps,p95_mps,p99_mps,max_mps,n")
        println(edge_speed, "source,date,window,target_hpa,region,mean_mps,std_mps,min_mps,p50_mps,p95_mps,p99_mps,max_mps,n")
        println(vertical_cm, "source,date,window,target_hpa,mean_kg_m2_s,std_kg_m2_s,min_kg_m2_s,p50_kg_m2_s,p95_kg_m2_s,p99_kg_m2_s,max_kg_m2_s,n")
        println(roughness, "source,date,window,target_hpa,mean_neighbor_delta_mps,p95_neighbor_delta_mps,max_neighbor_delta_mps")
        println(paired, "date,window,target_hpa,era_mean_mps,geosit_mean_mps,bias_geosit_minus_era_mps,mae_mps,rmse_mps,maxabs_mps,corr")
        println(temporal, "source,prev_date,prev_window,date,window,target_hpa,mean_abs_jump_mps,rmse_jump_mps,maxabs_jump_mps,corr")

        previous_maps = Dict{Tuple{String, Int}, NamedTuple}()
        worst = Dict{Int, NamedTuple}()

        for date in DATES
            readers = Dict{String, TransportBinaryReader{Float32}}()
            for spec in SOURCES
                reader = TransportBinaryReader(binary_path(spec, date); FT = Float32)
                readers[spec.name] = reader
                write_metadata(metadata, spec, date, reader.path, reader)
            end
            flush(metadata)

            nwindow = minimum(reader.header.nwindow for reader in values(readers))
            windows = parse_windows(WINDOWS_SPEC, nwindow)
            for win in windows
                println("Processing $date window $win")
                common_maps = Dict{String, Vector{Vector{Float64}}}()

                for spec in SOURCES
                    reader = readers[spec.name]
                    ctx = contexts[spec.name]
                    steps = reader.header.steps_per_window_by_window[win]
                    # see compare_c180_binary_winds.jl: full-window storage
                    # normalizes by the full window dt, per-substep by dt/(2*steps)
                    fk = get(reader.header.raw_header, "flux_kind", "substep_mass_amount")
                    dt_factor = String(fk) == "full_window_mass_amount" ?
                        Float64(reader.header.dt_met_seconds) :
                        Float64(reader.header.dt_met_seconds) /
                        (2.0 * Float64(steps))
                    window = load_mass_flux_window(reader, win)
                    speeds = speed_fields_at_pressures(window, ctx.mesh, ctx.basis, dt_factor, TARGET_HPA)
                    cmrates = cm_abs_rate_fields_at_pressures(window, ctx.mesh, dt_factor, TARGET_HPA)
                    maps = [zeros(Float64, LL_NX * LL_NY) for _ in TARGET_HPA]

                    for t in eachindex(TARGET_HPA)
                        s = weighted_stats(speeds[t], ctx.area)
                        @printf(global_speed, "%s,%s,%d,%.1f,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n",
                            spec.name, date, win, TARGET_HPA[t], s.mean, s.std, s.min,
                            s.p50, s.p95, s.p99, s.max, s.n)

                        es = weighted_stats(speeds[t], ctx.area; mask = ctx.edge)
                        is = weighted_stats(speeds[t], ctx.area; mask = ctx.edge, invert_mask = true)
                        @printf(edge_speed, "%s,%s,%d,%.1f,edge,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n",
                            spec.name, date, win, TARGET_HPA[t], es.mean, es.std, es.min,
                            es.p50, es.p95, es.p99, es.max, es.n)
                        @printf(edge_speed, "%s,%s,%d,%.1f,interior,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n",
                            spec.name, date, win, TARGET_HPA[t], is.mean, is.std, is.min,
                            is.p50, is.p95, is.p99, is.max, is.n)

                        vs = weighted_stats(cmrates[t], ctx.area)
                        @printf(vertical_cm, "%s,%s,%d,%.1f,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n",
                            spec.name, date, win, TARGET_HPA[t], vs.mean, vs.std, vs.min,
                            vs.p50, vs.p95, vs.p99, vs.max, vs.n)

                        apply_regridder!(maps[t], ctx.regridder, speeds[t])
                        rs = neighbor_roughness(maps[t], LL_NX, LL_NY)
                        @printf(roughness, "%s,%s,%d,%.1f,%.9g,%.9g,%.9g\n",
                            spec.name, date, win, TARGET_HPA[t], rs.mean_abs, rs.p95, rs.max)

                        key = (spec.name, t)
                        if haskey(previous_maps, key)
                            prev = previous_maps[key]
                            ts = temporal_stats(prev.map, maps[t], ll_area)
                            @printf(temporal, "%s,%s,%d,%s,%d,%.1f,%.9g,%.9g,%.9g,%.9g\n",
                                spec.name, prev.date, prev.window, date, win, TARGET_HPA[t],
                                ts.mean_abs, ts.rmse, ts.maxabs, ts.corr)
                        end
                        previous_maps[key] = (; date, window = win, map = copy(maps[t]))
                    end

                    common_maps[spec.name] = maps
                    window = nothing
                    speeds = nothing
                    cmrates = nothing
                    GC.gc()
                end

                era_maps = common_maps["ERA5"]
                geos_maps = common_maps["GEOSIT"]
                for t in eachindex(TARGET_HPA)
                    ps = paired_stats(era_maps[t], geos_maps[t], ll_area)
                    @printf(paired, "%s,%d,%.1f,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                        date, win, TARGET_HPA[t], ps.era_mean, ps.geos_mean, ps.bias,
                        ps.mae, ps.rmse, ps.maxabs, ps.corr)
                    if !haskey(worst, t) || ps.rmse > worst[t].rmse
                        worst[t] = (; rmse = ps.rmse, date, window = win,
                            era = copy(era_maps[t]), geos = copy(geos_maps[t]))
                    end
                end

                flush(global_speed)
                flush(edge_speed)
                flush(vertical_cm)
                flush(roughness)
                flush(paired)
                flush(temporal)
            end

            empty!(readers)
        end

        for t in eachindex(TARGET_HPA)
            haskey(worst, t) || continue
            w = worst[t]
            suffix = replace(@sprintf("%.0f", TARGET_HPA[t]), "." => "p")
            write_worst_map(joinpath(OUT_DIR, "worst_pair_speed_map_$(suffix)hpa.csv"),
                ll, w.date, w.window, TARGET_HPA[t], w.era, w.geos)
        end

        open(joinpath(OUT_DIR, "README.txt"), "w") do io
            println(io, "C180 ERA5 vs GEOS-IT binary mass-flux audit")
            println(io, "Generated: ", Dates.now())
            println(io, "Dates: ", join(DATES, ","))
            println(io, "Windows: ", WINDOWS_SPEC)
            println(io, "Targets hPa: ", join(string.(TARGET_HPA), ","))
            println(io, "Common grid: ", LL_NX, "x", LL_NY)
            println(io, "Speed normalization: horizontal substep mass amount divided by (layer mass/area * face length * dt_met/(2*steps_per_window)).")
            println(io, "Outputs:")
            println(io, "  binary_metadata.csv")
            println(io, "  global_speed_stats.csv")
            println(io, "  edge_vs_interior_speed_stats.csv")
            println(io, "  vertical_cm_abs_rate_stats.csv")
            println(io, "  common_grid_speed_roughness.csv")
            println(io, "  common_grid_era_geosit_speed_metrics.csv")
            println(io, "  common_grid_temporal_speed_jumps.csv")
            println(io, "  worst_pair_speed_map_*hpa.csv")
        end
    finally
        close(metadata)
        close(global_speed)
        close(edge_speed)
        close(vertical_cm)
        close(roughness)
        close(paired)
        close(temporal)
    end

    println("Wrote ", OUT_DIR)
end

main()

# This script has repeatedly hit a Julia shutdown-time segfault after all CSVs
# are closed and the final "Wrote ..." line is printed, likely from a finalizer
# in the mmap/regridding stack. Hard-exit by default after successful writes so
# batch runs return cleanly. Set HARD_EXIT_AFTER_AUDIT=0 to debug finalizers.
if get(ENV, "HARD_EXIT_AFTER_AUDIT", "1") == "1"
    flush(stdout)
    flush(stderr)
    ccall(:_exit, Cvoid, (Cint,), 0)
end
