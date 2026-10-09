"""
Makie extension for AtmosTransport visualization.

Load a Makie backend before plotting, for example:

```julia
using CairoMakie
using AtmosTransport
```

The core `AtmosTransport.Visualization` module remains Makie-free; this
extension adds rendering methods for topology-aware snapshot fields.
"""
module AtmosTransportMakieExt

import AtmosTransport
using AtmosTransport.Parameters: DRY_AIR_MOLAR_MASS, SPECIES_MOLAR_MASS, STANDARD_GRAVITY
import AtmosTransport.Visualization:
    HorizontalField, RasterField, SnapshotDataset, SnapshotRegridCache, PlotSpec,
    fieldview, frame_indices, as_raster, robust_colorrange,
    mapplot, mapplot!, snapshot_grid, movie, movie_grid, catrine_map_curtains,
    catrine_map_curtains_3way

using Makie
using Dates
using JSON3
using NCDatasets
using Printf

function _label(raster::RasterField)
    isempty(raster.units) ? String(raster.name) : "$(raster.name) [$(raster.units)]"
end

function _title(raster::RasterField; override=nothing)
    override !== nothing && return String(override)
    return "$(raster.name) $(raster.transform), t=$(round(raster.time; digits=2)) h"
end

function _colorrange(rasters, colorrange)
    colorrange === :auto && return robust_colorrange(rasters)
    return colorrange
end

function _heatmap_axis!(fig, slot, raster::RasterField;
                        colormap=:viridis,
                        colorrange=:auto,
                        title=nothing,
                        hide_ticks::Bool=true)
    ax = Makie.Axis(fig[slot...];
                    aspect=Makie.DataAspect(),
                    title=_title(raster; override=title),
                    xlabel="longitude",
                    ylabel="latitude")
    if hide_ticks
        ax.xticksvisible[] = false
        ax.yticksvisible[] = false
        ax.xticklabelsvisible[] = false
        ax.yticklabelsvisible[] = false
    end
    hm = Makie.heatmap!(ax, raster.lons, raster.lats, raster.values;
                        colormap=colormap,
                        colorrange=_colorrange([raster], colorrange))
    return ax, hm
end

"""
    mapplot(field; kwargs...) -> Figure

Render one topology-native `HorizontalField` as a regular lon-lat debug map.
CS fields are conservatively regridded through `as_raster`.
"""
function mapplot(field::HorizontalField;
                 resolution::Tuple{Int, Int}=(360, 181),
                 cache::SnapshotRegridCache=SnapshotRegridCache(),
                 colormap=:viridis,
                 colorrange=:auto,
                 title=nothing,
                 colorbar::Bool=true,
                 size=(1000, 520))
    raster = as_raster(field; resolution=resolution, cache=cache)
    fig = Makie.Figure(size=size)
    ax, hm = _heatmap_axis!(fig, (1, 1), raster;
                            colormap=colormap,
                            colorrange=colorrange,
                            title=title,
                            hide_ticks=false)
    colorbar && Makie.Colorbar(fig[1, 2], hm; label=_label(raster))
    return fig
end

"""
    mapplot!(ax, field; kwargs...)

Draw one field into an existing Makie axis and return the heatmap plot object.
"""
function mapplot!(ax, field::HorizontalField;
                  resolution::Tuple{Int, Int}=(360, 181),
                  cache::SnapshotRegridCache=SnapshotRegridCache(),
                  colormap=:viridis,
                  colorrange=:auto)
    raster = as_raster(field; resolution=resolution, cache=cache)
    return Makie.heatmap!(ax, raster.lons, raster.lats, raster.values;
                          colormap=colormap,
                          colorrange=_colorrange([raster], colorrange))
end

function _grid_size(n::Int, cols::Int)
    c = max(1, cols)
    r = cld(n, c)
    return r, c
end

"""
    snapshot_grid(snapshot, variable; times=:all, cols=4, kwargs...) -> Figure

Render a multi-panel figure containing one variable at several snapshot times.
"""
function snapshot_grid(snapshot::SnapshotDataset,
                       variable::Union{Symbol, AbstractString};
                       transform::Symbol=:column_mean,
                       times=:all,
                       level::Union{Nothing, Int}=nothing,
                       unit::Symbol=:native,
                       cols::Int=4,
                       resolution::Tuple{Int, Int}=(360, 181),
                       cache::SnapshotRegridCache=SnapshotRegridCache(),
                       colormap=:viridis,
                       colorrange=:auto,
                       size=nothing)
    indices = frame_indices(snapshot, times)
    fields = [fieldview(snapshot, variable;
                        transform=transform,
                        time=i,
                        level=level,
                        unit=unit) for i in indices]
    rasters = [as_raster(f; resolution=resolution, cache=cache) for f in fields]
    cr = _colorrange(rasters, colorrange)
    rows, columns = _grid_size(length(rasters), cols)
    fig_size = size === nothing ? (260 * columns + 90, 180 * rows + 70) : size
    fig = Makie.Figure(size=fig_size)

    hm = nothing
    for (idx, raster) in enumerate(rasters)
        row = div(idx - 1, columns) + 1
        col = mod(idx - 1, columns) + 1
        ax, h = _heatmap_axis!(fig, (row, col), raster;
                               colormap=colormap,
                               colorrange=cr,
                               title="t=$(round(raster.time; digits=2)) h")
        hm === nothing && (hm = h)
    end
    hm !== nothing && Makie.Colorbar(fig[1:rows, columns + 1], hm;
                                     label=_label(first(rasters)),
                                     height=Makie.Relative(0.9))
    return fig
end

function _spec_title(spec::PlotSpec, raster::RasterField)
    spec.title === nothing || return spec.title
    if spec.transform === :level_slice || spec.transform === :surface_slice
        lev = spec.level === nothing ? "" : " level $(spec.level)"
        return "$(spec.variable) $(spec.transform)$(lev)"
    end
    return "$(spec.variable) $(spec.transform)"
end

function _spec_rasters(snapshot::SnapshotDataset, spec::PlotSpec, indices;
                       resolution, cache)
    fields = [fieldview(snapshot, spec.variable;
                        transform=spec.transform,
                        time=i,
                        level=spec.level,
                        unit=spec.unit) for i in indices]
    return [as_raster(f; resolution=resolution, cache=cache) for f in fields]
end

"""
    movie_grid(snapshot, specs, out; times=:all, fps=8, kwargs...) -> String

Record a multi-panel GIF/MP4 from snapshot fields. Each `PlotSpec` becomes one
panel, and expensive topology geometry is cached across all frames.
"""
function movie_grid(snapshot::SnapshotDataset,
                    specs::AbstractVector{PlotSpec},
                    out::AbstractString;
                    times=:all,
                    fps::Real=8,
                    cols::Int=max(1, length(specs)),
                    resolution::Tuple{Int, Int}=(360, 181),
                    cache::SnapshotRegridCache=SnapshotRegridCache(),
                    colormap=:viridis,
                    colorrange=:auto,
                    size=nothing)
    isempty(specs) && throw(ArgumentError("movie_grid requires at least one PlotSpec"))
    indices = frame_indices(snapshot, times)
    isempty(indices) && throw(ArgumentError("movie_grid has no frames"))

    rasters_by_spec = [_spec_rasters(snapshot, spec, indices;
                                     resolution=resolution,
                                     cache=cache) for spec in specs]
    ranges = [colorrange === :auto ? robust_colorrange(rasters) : colorrange
              for rasters in rasters_by_spec]
    rows, columns = _grid_size(length(specs), cols)
    fig_size = size === nothing ? (340 * columns + 90, 220 * rows + 70) : size
    fig = Makie.Figure(size=fig_size)

    data_obs = Makie.Observable[]
    axes = Any[]
    for (idx, spec) in enumerate(specs)
        raster = rasters_by_spec[idx][1]
        row = div(idx - 1, columns) + 1
        col = mod(idx - 1, columns) + 1
        ax = Makie.Axis(fig[row, col];
                        aspect=Makie.DataAspect(),
                        title="$( _spec_title(spec, raster) ), t=$(round(raster.time; digits=2)) h")
        ax.xticksvisible[] = false
        ax.yticksvisible[] = false
        ax.xticklabelsvisible[] = false
        ax.yticklabelsvisible[] = false
        obs = Makie.Observable(raster.values)
        hm = Makie.heatmap!(ax, raster.lons, raster.lats, obs;
                            colormap=colormap,
                            colorrange=ranges[idx])
        Makie.Colorbar(fig[row, columns + 1], hm; label=_label(raster))
        push!(data_obs, obs)
        push!(axes, ax)
    end

    mkpath(dirname(String(out)))
    Makie.record(fig, String(out), eachindex(indices); framerate=round(Int, fps)) do frame
        for idx in eachindex(specs)
            raster = rasters_by_spec[idx][frame]
            data_obs[idx][] = raster.values
            axes[idx].title[] = "$(_spec_title(specs[idx], raster)), t=$(round(raster.time; digits=2)) h"
        end
    end
    return String(out)
end

"""
    movie(snapshot, variable, out; kwargs...) -> String

Record a one-panel movie for one variable.
"""
function movie(snapshot::SnapshotDataset,
               variable::Union{Symbol, AbstractString},
               out::AbstractString;
               transform::Symbol=:column_mean,
               level::Union{Nothing, Int}=nothing,
               unit::Symbol=:native,
               kwargs...)
    spec = PlotSpec(Symbol(variable); transform=transform, level=level, unit=unit)
    return movie_grid(snapshot, [spec], out; kwargs...)
end

const _CATRINE_RUN_START = DateTime(2021, 12, 1)
const _CATRINE_GC_VAR = Dict(
    :co2_natural => "SpeciesConcVV_CO2",
    :co2_fossil => "SpeciesConcVV_FossilCO2",
    :sf6 => "SpeciesConcVV_SF6",
    :rn222 => "SpeciesConcVV_Rn222",
)
const _CATRINE_GC_FLUX_VAR = Dict(
    :co2_natural => "EmisCO2_Total",
    :co2_fossil => "Emis_FossilCO2_Total",
    :sf6 => "EmisSF6",
    :rn222 => "EmisRn_Soil",
)
const _CATRINE_MOLAR_MASS = Dict(
    :co2_natural => SPECIES_MOLAR_MASS.co2,
    :co2_fossil => SPECIES_MOLAR_MASS.co2,
    :sf6 => SPECIES_MOLAR_MASS.sf6,
    :rn222 => SPECIES_MOLAR_MASS.rn222,
)
const _CATRINE_TITLE_DATEFORMAT = DateFormat("yyyy-mm-dd HH:MM")

function _gc_datetime(path::AbstractString)
    m = match(r"(\d{8})_(\d{4})z\.nc4$", basename(path))
    m === nothing && return nothing
    return DateTime(m.captures[1] * m.captures[2], dateformat"yyyymmddHHMM")
end

function _matched_catrine_times(at_path::AbstractString, gc_dir::AbstractString, max_frames::Integer)
    gc_map = Dict{DateTime, String}()
    for file in sort(filter(f -> occursin(r"GEOSChem\.CATRINE_inst\..*\.nc4$", f), readdir(gc_dir)))
        path = joinpath(gc_dir, file)
        dt = _gc_datetime(path)
        dt === nothing || (gc_map[dt] = path)
    end
    pairs = Tuple{Int, DateTime, String}[]
    NCDataset(at_path, "r") do ds
        hours = Float64.(collect(ds["time"].var[:]))
        for (i, hour) in enumerate(hours)
            dt = _CATRINE_RUN_START + Hour(round(Int, hour))
            haskey(gc_map, dt) && push!(pairs, (i, dt, gc_map[dt]))
        end
    end
    return max_frames <= 0 ? pairs : pairs[1:min(end, max_frames)]
end

_wrap_lon_180(lon) = mod(lon + 180.0, 360.0) - 180.0

function _robinson_xy(lon_deg::Real, lat_deg::Real)
    # Robinson tabular projection at 5 degree intervals, normalized to a
    # radius-one sphere. Coefficients follow the public-domain Snyder table.
    xcoef = (1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600,
             0.9427, 0.9216, 0.8962, 0.8679, 0.8350, 0.7986, 0.7597,
             0.7186, 0.6732, 0.6213, 0.5722, 0.5322)
    ycoef = (0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720,
             0.4340, 0.4958, 0.5571, 0.6176, 0.6769, 0.7346, 0.7903,
             0.8435, 0.8936, 0.9394, 0.9761, 1.0000)
    lat = clamp(Float64(lat_deg), -90.0, 90.0)
    lon = Float64(_wrap_lon_180(lon_deg))
    a = abs(lat) / 5.0
    i = clamp(floor(Int, a) + 1, 1, 18)
    f = clamp(a - (i - 1), 0.0, 1.0)
    xfac = (1 - f) * xcoef[i] + f * xcoef[i + 1]
    yfac = (1 - f) * ycoef[i] + f * ycoef[i + 1]
    x = 0.8487 * deg2rad(lon) * xfac
    y = 1.3523 * sign(lat) * yfac
    return Makie.Point2f(x, y)
end

function _robinson_polyline(lons, lats)
    pts = Makie.Point2f[]
    for (lon, lat) in zip(lons, lats)
        push!(pts, _robinson_xy(lon, lat))
    end
    return pts
end

function _coastline_geojson_path()
    for depot in DEPOT_PATH
        root = joinpath(depot, "packages", "GeoMakie")
        isdir(root) || continue
        for entry in sort(readdir(root); rev=true)
            path = joinpath(root, entry, "assets", "vector", "110m_coastline.geojson")
            isfile(path) && return path
        end
    end
    return nothing
end

function _load_robinson_coastlines()
    path = _coastline_geojson_path()
    path === nothing && return Vector{Vector{Makie.Point2f}}()
    parsed = JSON3.read(read(path, String))
    lines = Vector{Vector{Makie.Point2f}}()
    add_coords! = function (coords)
        current = Makie.Point2f[]
        prev_lon = nothing
        for c in coords
            lon = Float64(c[1])
            lat = Float64(c[2])
            if prev_lon !== nothing && abs(_wrap_lon_180(lon) - _wrap_lon_180(prev_lon)) > 180.0
                length(current) > 1 && push!(lines, current)
                current = Makie.Point2f[]
            end
            push!(current, _robinson_xy(lon, lat))
            prev_lon = lon
        end
        length(current) > 1 && push!(lines, current)
        return nothing
    end
    for feature in parsed.features
        geom = feature.geometry
        if String(geom.type) == "LineString"
            add_coords!(geom.coordinates)
        elseif String(geom.type) == "MultiLineString"
            for coords in geom.coordinates
                add_coords!(coords)
            end
        end
    end
    return lines
end

function _read_cs3(ds, name::AbstractString)
    return Array{Float64}(ds[name][:, :, :])
end

function _read_cs4_time(ds, name::AbstractString, ti::Integer)
    return Array{Float64}(ds[name][:, :, :, :, ti])
end

function _read_gc4(ds, name::AbstractString)
    return Array{Float64}(ds[name][:, :, :, :, 1])
end

function _at_pressure_hpa(air_mass, area)
    nx, ny, nf, nz = size(air_mass)
    out = similar(air_mass)
    @inbounds for p in 1:nf, j in 1:ny, i in 1:nx
        acc = 0.0
        a = area[i, j, p]
        for k in 1:nz
            dp = air_mass[i, j, p, k] / a * STANDARD_GRAVITY
            acc += dp
            out[i, j, p, k] = (acc - 0.5 * dp) / 100.0
        end
    end
    return out
end

function _catrine_fields(at_ds, gc_ds, species::Symbol, ti::Integer)
    at_vmr = _read_cs4_time(at_ds, String(species), ti) .* 1e6
    at_air = _read_cs4_time(at_ds, "air_mass", ti)
    area = _read_cs3(at_ds, "cell_area")
    at_p = _at_pressure_hpa(at_air, area)

    gc_vmr_st = _read_gc4(gc_ds, _CATRINE_GC_VAR[species]) .* 1e6
    gc_air_st = _read_gc4(gc_ds, "Met_AD")
    gc_p_st = _read_gc4(gc_ds, "Met_PMIDDRY")

    nz = min(size(at_vmr, 4), size(gc_vmr_st, 4))
    at_vmr = at_vmr[:, :, :, (end - nz + 1):end]
    at_air = at_air[:, :, :, (end - nz + 1):end]
    at_p = at_p[:, :, :, (end - nz + 1):end]
    gc_vmr = gc_vmr_st[:, :, :, nz:-1:1]
    gc_air = gc_air_st[:, :, :, nz:-1:1]
    gc_p = gc_p_st[:, :, :, nz:-1:1]
    return at_vmr, at_air, at_p, gc_vmr, gc_air, gc_p
end

function _column_mean_ppm(vmr_ppm, air_mass)
    nx, ny, nf, nz = size(vmr_ppm)
    out = zeros(Float64, nx, ny, nf)
    @inbounds for p in 1:nf, j in 1:ny, i in 1:nx
        num = 0.0
        den = 0.0
        for k in 1:nz
            m = air_mass[i, j, p, k]
            num += vmr_ppm[i, j, p, k] * m
            den += m
        end
        out[i, j, p] = den > 0 ? num / den : NaN
    end
    return out
end

function _global_burden_kg(vmr_ppm, air_mass, species::Symbol)
    factor = _CATRINE_MOLAR_MASS[species] / DRY_AIR_MOLAR_MASS * 1e-6
    return sum(vmr_ppm .* air_mass) * factor
end

function _gc_flux_kg_s(ds, species::Symbol)
    vname = get(_CATRINE_GC_FLUX_VAR, species, "")
    isempty(vname) && return nothing
    haskey(ds, vname) || return nothing
    flux = Array{Float64}(ds[vname][:, :, :, 1])
    area = Array{Float64}(ds["Met_AREAM2"][:, :, :, 1])
    return sum(flux .* area)
end

function _read_at_flux_kg_s(path::AbstractString, species::Symbol)
    isfile(path) || return nothing
    pat = Regex("Surface source $(species) total model-storage rate:\\s+([0-9.eE+-]+)\\s+kg_air_equiv/s")
    m = match(pat, read(path, String))
    m === nothing && return nothing
    storage_rate = parse(Float64, m.captures[1])
    return storage_rate * _CATRINE_MOLAR_MASS[species] / DRY_AIR_MOLAR_MASS
end

function _budget_text(storage_kg, flux_kg_s, elapsed_s)
    if flux_kg_s === nothing
        return @sprintf("Global storage: %.4e kg\nGlobal Sum(flux): n/a", storage_kg)
    end
    return @sprintf("Global storage: %.4e kg\nGlobal Sum(flux) dt: %.4e kg  (%.4e kg/s)",
                    storage_kg, flux_kg_s * elapsed_s, flux_kg_s)
end

function _unit_xyz(lon_deg, lat_deg)
    lon = deg2rad(Float64(lon_deg))
    lat = deg2rad(Float64(lat_deg))
    return (cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))
end

function _build_sections(lons, lats, section_lats, dlon)
    lon_samples = collect(-180.0:dlon:180.0)
    centers = vec([(lons[i], lats[i]) for i in eachindex(lons)])
    xyz = [_unit_xyz(lon, lat) for (lon, lat) in centers]
    sections = []
    for lat in section_lats
        idxs = Vector{CartesianIndex{3}}(undef, length(lon_samples))
        for (n, lon) in enumerate(lon_samples)
            x, y, z = _unit_xyz(lon, lat)
            best = 1
            bestd = Inf
            @inbounds for q in eachindex(xyz)
                dx = xyz[q][1] - x
                dy = xyz[q][2] - y
                dz = xyz[q][3] - z
                d = dx * dx + dy * dy + dz * dz
                if d < bestd
                    bestd = d
                    best = q
                end
            end
            idxs[n] = CartesianIndices(lons)[best]
        end
        push!(sections, (; lat = Float64(lat), lons = lon_samples, idxs))
    end
    return sections
end

function _curtain(vmr_ppm, p_hpa, section, p_grid)
    out = fill(NaN, length(section.lons), length(p_grid))
    @inbounds for n in eachindex(section.lons)
        idx = section.idxs[n]
        p = vec(p_hpa[idx[1], idx[2], idx[3], :])
        v = vec(vmr_ppm[idx[1], idx[2], idx[3], :])
        ok = findall(i -> isfinite(p[i]) && isfinite(v[i]), eachindex(p))
        length(ok) < 2 && continue
        order = sortperm(p[ok])
        pp = p[ok][order]
        vv = v[ok][order]
        for (m, target) in enumerate(p_grid)
            if target < first(pp) || target > last(pp)
                continue
            end
            hi = searchsortedfirst(pp, target)
            if hi <= 1
                out[n, m] = vv[1]
            else
                p0, p1 = pp[hi - 1], pp[hi]
                v0, v1 = vv[hi - 1], vv[hi]
                out[n, m] = v0 + (v1 - v0) * (target - p0) / (p1 - p0)
            end
        end
    end
    return out
end

_lat_label(lat) = abs(lat) < 1e-9 ? "Eq" : @sprintf("%g%s", abs(lat), lat > 0 ? "N" : "S")

function _catrine_tracer_label(species::Symbol)
    species === :co2_fossil && return "Fossil CO₂"
    species === :co2_natural && return "Natural CO₂"
    species === :sf6 && return "SF₆"
    species === :rn222 && return "²²²Rn"
    return String(species)
end

function _symlog01(x, vmax; linthresh=0.05)
    y = max(Float64(x), 0.0)
    return asinh(y / linthresh) / asinh(Float64(vmax) / linthresh)
end

function _symlog_array(values, vmax; linthresh=0.05)
    out = similar(values, Float64)
    @inbounds for i in eachindex(values)
        out[i] = isfinite(values[i]) ? _symlog01(values[i], vmax; linthresh) : NaN
    end
    return out
end

# Linear normalization to [0, 1] across (vmin, vmax) — clamped at both ends so
# values that overshoot don't push the colormap past its dark-end.
function _linear01(x, vmin, vmax)
    vmin_f = Float64(vmin)
    vmax_f = Float64(vmax)
    den = vmax_f - vmin_f
    den == 0.0 && return 0.0
    return clamp((Float64(x) - vmin_f) / den, 0.0, 1.0)
end

function _linear_array(values, vmin, vmax)
    out = similar(values, Float64)
    @inbounds for i in eachindex(values)
        out[i] = isfinite(values[i]) ? _linear01(values[i], vmin, vmax) : NaN
    end
    return out
end

function _cs_cell_polygons(corner_lons, corner_lats)
    nx = size(corner_lons, 1) - 1
    ny = size(corner_lons, 2) - 1
    nf = size(corner_lons, 3)
    polygons = Vector{Vector{Makie.Point2f}}()
    indices = CartesianIndex{3}[]
    for p in 1:nf, j in 1:ny, i in 1:nx
        lons = [_wrap_lon_180(corner_lons[i, j, p]),
                _wrap_lon_180(corner_lons[i + 1, j, p]),
                _wrap_lon_180(corner_lons[i + 1, j + 1, p]),
                _wrap_lon_180(corner_lons[i, j + 1, p])]
        maximum(lons) - minimum(lons) > 180.0 && continue
        lats = [corner_lats[i, j, p],
                corner_lats[i + 1, j, p],
                corner_lats[i + 1, j + 1, p],
                corner_lats[i, j + 1, p]]
        push!(polygons, _expand_polygon([_robinson_xy(lons[q], lats[q]) for q in 1:4]))
        push!(indices, CartesianIndex(i, j, p))
    end
    return polygons, indices
end

function _expand_polygon(poly; factor=1.002)
    cx = sum(p[1] for p in poly) / length(poly)
    cy = sum(p[2] for p in poly) / length(poly)
    return [Makie.Point2f(cx + factor * (p[1] - cx),
                          cy + factor * (p[2] - cy)) for p in poly]
end

function _polygon_colors(field, indices, vmax)
    colors = Vector{Float64}(undef, length(indices))
    @inbounds for n in eachindex(indices)
        idx = indices[n]
        colors[n] = _symlog01(field[idx], vmax)
    end
    return colors
end

function _polygon_colors_linear(field, indices, vmin, vmax)
    colors = Vector{Float64}(undef, length(indices))
    @inbounds for n in eachindex(indices)
        idx = indices[n]
        colors[n] = _linear01(field[idx], vmin, vmax)
    end
    return colors
end

function _decorate_robinson_axis!(ax, section_lats)
    for lat in -60:30:60
        pts = _robinson_polyline(-180:2:180, fill(lat, 181))
        Makie.lines!(ax, pts; color=(:gray35, 0.20), linewidth=0.45)
    end
    for lon in -150:30:150
        pts = _robinson_polyline(fill(lon, 91), -90:2:90)
        Makie.lines!(ax, pts; color=(:gray35, 0.22), linewidth=0.45)
    end
    for lat in section_lats
        pts = _robinson_polyline(-180:2:180, fill(lat, 181))
        Makie.lines!(ax, pts; color=(:black, 0.45), linewidth=0.9, linestyle=:dot)
    end
    outline = _robinson_polyline(vcat(-180:2:180, fill(180, 91), 180:-2:-180, fill(-180, 91)),
                                 vcat(fill(-90, 181), -90:2:90, fill(90, 181), 90:-2:-90))
    Makie.lines!(ax, outline; color=:gray25, linewidth=1.0)
    Makie.hidedecorations!(ax)
    Makie.hidespines!(ax)
    Makie.xlims!(ax, -3.05, 3.05)
    Makie.ylims!(ax, -1.56, 1.56)
    return ax
end

function _draw_robinson_coastlines!(ax, coastlines)
    for line in coastlines
        Makie.lines!(ax, line; color=(:gray12, 0.76), linewidth=0.8)
    end
    return ax
end

function _draw_curtain_guides!(ax; longitude_step=60, pressure_step=250)
    for lon in -180:longitude_step:180
        Makie.vlines!(ax, [lon]; color=(:gray20, 0.16), linewidth=0.65)
    end
    for p in 0:pressure_step:1000
        Makie.hlines!(ax, [p]; color=(:gray20, 0.18), linewidth=0.65)
    end
    return ax
end

function _scene_rect(scene)
    return hasproperty(scene, :viewport) ? scene.viewport[] : scene.px_area[]
end

function _fig_y_extent(fig)
    r = _scene_rect(fig.scene)
    return Float32(r.origin[2]), Float32(r.origin[2] + r.widths[2])
end

function _fig_x_extent(fig)
    r = _scene_rect(fig.scene)
    return Float32(r.origin[1]), Float32(r.origin[1] + r.widths[1])
end

function _axis_y_extent(ax)
    r = _scene_rect(ax.scene)
    return Float32(r.origin[2]), Float32(r.origin[2] + r.widths[2])
end

# Axis block bbox — adds the title strip (above the plotting area) and the
# bottom decorations to the axis content bbox so the lower-row background
# rectangle covers the per-axis titles ("Column Mean", "40N", …) instead
# of stopping at the plotting area.
#
# `layoutobservables.computedbbox` is the axis's plotting-area cell;
# `layoutobservables.protrusions` carries the extra space allocated for
# the title (`.top`) and axis decorations (`.bottom`, `.left`, `.right`).
# A small `extra_top` adds a few pixels above the title so the title's
# own padding is covered too (otherwise the colored band stops right at
# the title baseline).
#
# When a Makie version omits `layoutobservables`, we estimate the title
# strip from `titlesize` + `titlegap`.
function _axis_block_y_extent(ax; extra_top::Real = 14)
    if hasproperty(ax, :layoutobservables)
        bbox  = ax.layoutobservables.computedbbox[]
        protr = ax.layoutobservables.protrusions[]
        ylo = Float32(bbox.origin[2] - protr.bottom)
        yhi = Float32(bbox.origin[2] + bbox.widths[2] + protr.top + extra_top)
        return ylo, yhi
    end
    ylo, yhi = _axis_y_extent(ax)
    title_size = hasproperty(ax, :titlesize) ? Float32(ax.titlesize[]) : 14.0f0
    title_gap  = hasproperty(ax, :titlegap)  ? Float32(ax.titlegap[])  : 4.0f0
    return ylo, yhi + title_size + title_gap + Float32(extra_top)
end

function _page_rect!(fig, x0, x1, y0, y1; color, z=-1000)
    xa, xb = minmax(Float32(x0), Float32(x1))
    ya, yb = minmax(Float32(y0), Float32(y1))
    p = Makie.poly!(fig.scene,
        Makie.Point2f[(xa, ya), (xb, ya), (xb, yb), (xa, yb)];
        color=color, strokewidth=0)
    Makie.translate!(p, 0, 0, z)
    return p
end

function _catrine_frame_data(at_path, gc_path, species, ti, sections, p_grid)
    NCDataset(at_path, "r") do at_ds
        NCDataset(gc_path, "r") do gc_ds
            at_vmr, at_air, at_p, gc_vmr, gc_air, gc_p =
                _catrine_fields(at_ds, gc_ds, species, ti)
            gc_col = _column_mean_ppm(gc_vmr, gc_air)
            at_col = _column_mean_ppm(at_vmr, at_air)
            gc_curtains = [_curtain(gc_vmr, gc_p, s, p_grid) for s in sections]
            at_curtains = [_curtain(at_vmr, at_p, s, p_grid) for s in sections]
            return (; gc_col, at_col, gc_curtains, at_curtains,
                    gc_burden = _global_burden_kg(gc_vmr, gc_air, species),
                    at_burden = _global_burden_kg(at_vmr, at_air, species),
                    gc_flux = _gc_flux_kg_s(gc_ds, species))
        end
    end
end

"""
    _precompute_catrine_frames(at_path, pairs, species, sections, p_grid)

Walk every (`at_ti`, `gc_path`) pair once with the AT NCDataset held open
across all frames (win #1) and the GC NCDatasets opened in parallel
threads (win #3). For each frame, computes the AT and GC column means
from VMR × air_mass on the fly (NOT from the pre-stored
`co2_*_column_mean` field — GC's pre-stored column-mass is known to be
biased and we want AT/GC to share the same computation). Returns
NamedTuple of cached arrays the per-frame record loop just looks up.
"""
function _precompute_catrine_frames(at_path::AbstractString,
                                     pairs::Vector,
                                     species::Symbol,
                                     sections,
                                     p_grid)
    n = length(pairs)
    at_col_all      = Vector{Array{Float64, 3}}(undef, n)
    gc_col_all      = Vector{Array{Float64, 3}}(undef, n)
    at_curtains_all = Vector{Vector{Matrix{Float64}}}(undef, n)
    gc_curtains_all = Vector{Vector{Matrix{Float64}}}(undef, n)

    # --- AT side: one NC handle, sequential frames. ---
    NCDataset(at_path, "r") do at_ds
        area = _read_cs3(at_ds, "cell_area")
        for i in 1:n
            ti = pairs[i][1]
            at_vmr = _read_cs4_time(at_ds, String(species), ti) .* 1e6
            at_air = _read_cs4_time(at_ds, "air_mass", ti)
            at_p   = _at_pressure_hpa(at_air, area)
            at_col_all[i]      = _column_mean_ppm(at_vmr, at_air)
            at_curtains_all[i] = [_curtain(at_vmr, at_p, s, p_grid) for s in sections]
        end
    end

    # --- GC side: one file per frame, parallel opens. ---
    Threads.@threads for i in 1:n
        gc_path = pairs[i][3]
        NCDataset(gc_path, "r") do gc_ds
            gc_vmr_st = _read_gc4(gc_ds, _CATRINE_GC_VAR[species]) .* 1e6
            gc_air_st = _read_gc4(gc_ds, "Met_AD")
            gc_p_st   = _read_gc4(gc_ds, "Met_PMIDDRY")
            nz = size(gc_vmr_st, 4)
            gc_vmr = gc_vmr_st[:, :, :, nz:-1:1]
            gc_air = gc_air_st[:, :, :, nz:-1:1]
            gc_p   = gc_p_st[:, :, :, nz:-1:1]
            gc_col_all[i]      = _column_mean_ppm(gc_vmr, gc_air)
            gc_curtains_all[i] = [_curtain(gc_vmr, gc_p, s, p_grid) for s in sections]
        end
    end

    return (; at_col = at_col_all, gc_col = gc_col_all,
              at_curtains = at_curtains_all, gc_curtains = gc_curtains_all)
end

"""
    _percentile_range(values, low_pct, high_pct) -> (lo, hi)

Sort the finite entries of `values` and return the `low_pct`/`high_pct`-th
percentile pair. Used by the day-1 auto-range path.
"""
function _percentile_range(values::AbstractVector, low_pct::Real, high_pct::Real)
    finite_only = filter(isfinite, values)
    isempty(finite_only) && return (0.0, 1.0)
    sorted = sort!(finite_only)
    n = length(sorted)
    lo_idx = max(1, ceil(Int, low_pct  / 100 * n))
    hi_idx = min(n, max(lo_idx, ceil(Int, high_pct / 100 * n)))
    return (Float64(sorted[lo_idx]), Float64(sorted[hi_idx]))
end

"""
    catrine_map_curtains(at_path, gc_dir; kwargs...) -> NamedTuple

Create the CATRINE GEOS-Chem vs AtmosTransport column-map plus
longitude-pressure curtain figure using the active Makie backend. The core
package only defines the API; this method lives in `AtmosTransportMakieExt`.
"""
function catrine_map_curtains(at_path::AbstractString,
                              gc_dir::AbstractString;
                              species::Symbol=:co2_fossil,
                              out_dir::AbstractString=joinpath(homedir(), "data", "AtmosTransport", "output", "catrine_makie_animation"),
                              fps::Integer=3,
                              max_frames::Integer=0,
                              map_vmax::Real=8.0,
                              map_vmin::Real=0.0,
                              curtain_vmax::Real=40.0,
                              curtain_vmin::Real=0.0,
                              scale::Symbol=:symlog,
                              auto_range_day1::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
                              latitudes=(40.0, 0.0, -40.0),
                              dlon::Real=2.0,
                              dp::Real=10.0,
                              at_log::Union{Nothing, AbstractString}=nothing,
                              write_animation::Bool=true,
                              size=(1896, 936))
    species = Symbol(species)
    haskey(_CATRINE_GC_VAR, species) ||
        throw(ArgumentError("unsupported CATRINE species $(species)"))
    scale in (:symlog, :linear) ||
        throw(ArgumentError("scale must be :symlog or :linear, got $(scale)"))
    at_path = expanduser(String(at_path))
    gc_dir = expanduser(String(gc_dir))
    out_dir = expanduser(String(out_dir))
    mkpath(out_dir)

    pairs = _matched_catrine_times(at_path, gc_dir, max_frames)
    isempty(pairs) && throw(ArgumentError("no matched CATRINE AT/GEOS-Chem frames"))

    lons = lats = corner_lons = corner_lats = nothing
    NCDataset(at_path, "r") do ds
        lons = _read_cs3(ds, "lons")
        lats = _read_cs3(ds, "lats")
        corner_lons = _read_cs3(ds, "corner_lons")
        corner_lats = _read_cs3(ds, "corner_lats")
    end

    section_lats = Float64.(collect(latitudes))
    sections = _build_sections(lons, lats, section_lats, Float64(dlon))
    p_grid = collect(0.0:Float64(dp):1000.0)
    polygons, poly_indices = _cs_cell_polygons(corner_lons, corner_lats)
    lon_grid = sections[1].lons

    # --- Win #1-3: pre-compute every frame once. AT NC held open across all
    # frames; GC opened in parallel threads. Per-frame record loop is a
    # straight lookup of cached arrays after this point.
    t_pre = time()
    precomp = _precompute_catrine_frames(at_path, pairs, species, sections, p_grid)
    @info @sprintf("  Pre-computed %d frame(s) in %.1fs", length(pairs), time() - t_pre)
    _, dt0, _ = first(pairs)
    default_log = joinpath(homedir(), "data", "AtmosTransport", "output", "logs",
                           "$(splitext(basename(at_path))[1]).log")
    at_flux_path = at_log === nothing ? default_log : expanduser(String(at_log))
    at_flux = _read_at_flux_kg_s(at_flux_path, species)
    elapsed0 = Dates.value(dt0 - _CATRINE_RUN_START) / 1000.0

    # --- Auto-range from day 1 (first 8 frames @ 3-hourly) when requested.
    # Both panels share the same vmin/vmax so AT vs GC differences are
    # actually visible. The same percentile pair feeds the column-map and
    # the curtain plots so the two colorbars stay comparable.
    if auto_range_day1 !== nothing
        low_pct  = Float64(auto_range_day1[1])
        high_pct = Float64(auto_range_day1[2])
        n_day1 = min(8, length(pairs))
        col_vals = Float64[]
        for i in 1:n_day1
            append!(col_vals, vec(precomp.at_col[i]))
            append!(col_vals, vec(precomp.gc_col[i]))
        end
        map_vmin, map_vmax = _percentile_range(col_vals, low_pct, high_pct)

        cur_vals = Float64[]
        for i in 1:n_day1, s_idx in eachindex(sections)
            append!(cur_vals, vec(precomp.at_curtains[i][s_idx]))
            append!(cur_vals, vec(precomp.gc_curtains[i][s_idx]))
        end
        curtain_vmin, curtain_vmax = _percentile_range(cur_vals, low_pct, high_pct)
        @info @sprintf("  Auto-range day-1 %.1f-%.1f pct: map=[%.3f, %.3f]  curtain=[%.3f, %.3f]",
                        low_pct, high_pct, map_vmin, map_vmax, curtain_vmin, curtain_vmax)
    end
    frame0 = (gc_col = precomp.gc_col[1], at_col = precomp.at_col[1],
              gc_curtains = precomp.gc_curtains[1], at_curtains = precomp.at_curtains[1])

    # Colormap pairs the colour scheme to the value transform.
    #   :linear → RdBu_10 reversed — Makie's ColorBrewer divergent palette,
    #     10 discrete steps; for column-mean CO2 in a narrow 405-425 ppm
    #     window the eye groups concentration bands more clearly than a
    #     smooth ramp, and the diverging palette puts the centre of the
    #     range at white so deviations either way are readable.
    #   :symlog → original sequential white→yellow→orange→red ramp, which
    #     suits 0-anchored anomaly fields (e.g. fossil CO2 perturbation
    #     above background) where the value of interest is the tail above
    #     zero, not a deviation around a centre.
    cmap = scale === :linear ?
        Makie.cgrad(:RdBu_10, rev=true) :
        Makie.cgrad([:white, "#fee8a8", "#fca85d", "#e34a33", "#7f0000"])
    map_color(field) = scale === :linear ?
        _polygon_colors_linear(field, poly_indices, map_vmin, map_vmax) :
        _polygon_colors(field, poly_indices, map_vmax)
    curtain_norm(values) = scale === :linear ?
        _linear_array(values, curtain_vmin, curtain_vmax) :
        _symlog_array(values, curtain_vmax)
    gc_colors = Makie.Observable(map_color(frame0.gc_col))
    at_colors = Makie.Observable(map_color(frame0.at_col))
    curtain_obs = [
        [Makie.Observable(curtain_norm(frame0.gc_curtains[i])) for i in eachindex(sections)],
        [Makie.Observable(curtain_norm(frame0.at_curtains[i])) for i in eachindex(sections)],
    ]
    tracer_label = _catrine_tracer_label(species)
    title_obs = Makie.Observable("Time: $(Dates.format(dt0, _CATRINE_TITLE_DATEFORMAT)) UTC    Tracer: $(tracer_label)")

    upper_row_color = "#edf1f5"
    lower_row_color = "#f3eee5"
    header_color = "#1f2328"
    fig = Makie.Figure(size=size, fontsize=14, backgroundcolor=upper_row_color, figure_padding=0)
    Makie.rowgap!(fig.layout, 0)
    Makie.colgap!(fig.layout, 0)
    Makie.Box(fig[0, 1:6]; color=(header_color, 1.0), strokecolor=(:transparent, 0.0))
    Makie.Label(fig[0, 1:6], title_obs; fontsize=22, font=:bold, color=:white)
    Makie.Label(fig[1:3, 1], "GEOS-Chem"; rotation=pi / 2, fontsize=22, font=:bold,
                tellwidth=false)
    Makie.Label(fig[4:6, 1], "AtmosTransport"; rotation=pi / 2, fontsize=22, font=:bold,
                tellwidth=false)
    ax_gc_map = Makie.Axis(fig[1:3, 2]; title="Column mean", aspect=Makie.DataAspect(),
                           backgroundcolor=:transparent)
    ax_at_map = Makie.Axis(fig[4:6, 2]; title="Column mean", aspect=Makie.DataAspect(),
                           backgroundcolor=:transparent)
    coastlines = _load_robinson_coastlines()
    p_gc = Makie.poly!(ax_gc_map, polygons; color=gc_colors, colormap=cmap, colorrange=(0, 1),
                       strokewidth=0)
    Makie.poly!(ax_at_map, polygons; color=at_colors, colormap=cmap, colorrange=(0, 1),
                strokewidth=0)
    _decorate_robinson_axis!(ax_gc_map, section_lats)
    _decorate_robinson_axis!(ax_at_map, section_lats)
    _draw_robinson_coastlines!(ax_gc_map, coastlines)
    _draw_robinson_coastlines!(ax_at_map, coastlines)

    if scale === :linear
        # ~5 evenly spaced labelled ticks across [vmin, vmax].
        nticks_map = 5
        tick_values_map = collect(range(Float64(map_vmin), Float64(map_vmax); length=nticks_map))
        tick_pos_map = [_linear01(v, map_vmin, map_vmax) for v in tick_values_map]
    else
        tick_values_map = [0.0, 0.05, 0.1, 0.5, 1, 2, 4, Float64(map_vmax)]
        tick_pos_map = [_symlog01(v, map_vmax) for v in tick_values_map]
    end
    Makie.Colorbar(fig[1:6, 3], p_gc; label="X$(species) [ppm]",
                   ticks=(tick_pos_map, string.(tick_values_map)), width=14)

    curtain_axes = Vector{Any}(undef, 6)
    curtain_plots = Vector{Any}(undef, 6)
    row_slots = [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)]
    labels = [_lat_label(s.lat) for s in sections]
    append!(labels, [_lat_label(s.lat) for s in sections])
    k = 1
    for row in 1:2
        for col in eachindex(sections)
            slot = row_slots[k]
            ax = Makie.Axis(fig[slot...]; title=labels[k],
                            ylabel=k == 3 ? "pressure [hPa]" : "",
                            xlabel=k == 6 ? "longitude" : "",
                            yreversed=true,
                            backgroundcolor=:transparent)
            Makie.hidespines!(ax)
            ax.xticksvisible[] = k == 6
            ax.xticklabelsvisible[] = k == 6
            ax.yaxisposition[] = :right
            ax.xticks = -180:60:180
            ax.yticks = [0, 250, 500, 750, 1000]
            ax.xgridvisible[] = true
            ax.ygridvisible[] = true
            ax.xgridcolor[] = (:gray35, 0.18)
            ax.ygridcolor[] = (:gray35, 0.22)
            ax.xgridwidth[] = 0.6
            ax.ygridwidth[] = 0.6
            k in (3, 6) || (ax.yticklabelsvisible[] = false)
            hm = Makie.heatmap!(ax, lon_grid, p_grid, curtain_obs[row][col];
                                colormap=cmap, colorrange=(0, 1), nan_color=:white)
            _draw_curtain_guides!(ax)
            curtain_axes[k] = ax
            curtain_plots[k] = hm
            k += 1
        end
    end
    if scale === :linear
        nticks_c = 6
        tick_values_c = collect(range(Float64(curtain_vmin), Float64(curtain_vmax); length=nticks_c))
        tick_pos_c = [_linear01(v, curtain_vmin, curtain_vmax) for v in tick_values_c]
    else
        tick_values_c = [0.0, 0.1, 0.5, 1, 5, 10, 20, Float64(curtain_vmax)]
        tick_pos_c = [_symlog01(v, curtain_vmax) for v in tick_values_c]
    end
    Makie.Colorbar(fig[1:6, 5], curtain_plots[1]; label="$(species) [ppm]",
                   ticks=(tick_pos_c, string.(tick_values_c)), width=14)
    Makie.colsize!(fig.layout, 1, 54)
    Makie.colsize!(fig.layout, 2, Makie.Relative(0.43))
    Makie.colsize!(fig.layout, 6, 110)
    Makie.resize_to_layout!(fig)
    x0, x1 = _fig_x_extent(fig)
    fig_y0, _ = _fig_y_extent(fig)
    # `_axis_block_y_extent` includes the title strip above the plotting
    # area; otherwise the lower-row background stops below "Column Mean"
    # and "40N" on the upper edge of the AT row.
    _, lower_top = _axis_block_y_extent(ax_at_map)
    _page_rect!(fig, x0, x1, fig_y0, lower_top; color=(lower_row_color, 1.0))

    function update_frame!(frame)
        _, dt, _ = pairs[frame]
        gc_colors[] = map_color(precomp.gc_col[frame])
        at_colors[] = map_color(precomp.at_col[frame])
        for i in eachindex(sections)
            curtain_obs[1][i][] = curtain_norm(precomp.gc_curtains[frame][i])
            curtain_obs[2][i][] = curtain_norm(precomp.at_curtains[frame][i])
        end
        title_obs[] = "Time: $(Dates.format(dt, _CATRINE_TITLE_DATEFORMAT)) UTC    Tracer: $(tracer_label)"
        return nothing
    end

    scale_tag = scale === :linear ? "linear" : "symlog"
    stem = "$(species)_column_map_curtains_$(scale_tag)_at_vs_geoschem_makie"
    png = joinpath(out_dir, "$(stem)_first_frame.png")
    gif = joinpath(out_dir, "$(stem).gif")
    Makie.save(png, fig)
    if write_animation
        Makie.record(fig, gif, eachindex(pairs); framerate=fps) do frame
            update_frame!(frame)
        end
    end
    return (; fig, png, gif = write_animation ? gif : nothing, frames = length(pairs))
end

"""
    catrine_map_curtains_3way(at_geos_path, at_era5_path, gc_dir; kwargs...) -> NamedTuple

Same column-map + longitude-pressure-curtain plot as [`catrine_map_curtains`],
but with THREE stacked sections instead of two: GeosChem (top), AtmosTransport
driven by GEOS-IT met (middle), and AtmosTransport driven by ERA5 met (bottom).
Both AT runs must share the GeosChem time base; they are precomputed against the
same matched frames. Reuses every helper of the 2-section renderer.
"""
function catrine_map_curtains_3way(at_geos_path::AbstractString,
                                   at_era5_path::AbstractString,
                                   gc_dir::AbstractString;
                                   species::Symbol=:co2_fossil,
                                   out_dir::AbstractString=joinpath(homedir(), "data", "AtmosTransport", "output", "catrine_3way_animation"),
                                   fps::Integer=3,
                                   max_frames::Integer=0,
                                   map_vmax::Real=8.0,
                                   map_vmin::Real=0.0,
                                   curtain_vmax::Real=40.0,
                                   curtain_vmin::Real=0.0,
                                   scale::Symbol=:symlog,
                                   auto_range_day1::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
                                   latitudes=(40.0, 0.0, -40.0),
                                   dlon::Real=2.0,
                                   dp::Real=10.0,
                                   write_animation::Bool=true,
                                   row_labels=("GEOS-Chem", "AT (GEOS-IT)", "AT (ERA5)"),
                                   size=(1896, 1380))
    species = Symbol(species)
    haskey(_CATRINE_GC_VAR, species) ||
        throw(ArgumentError("unsupported CATRINE species $(species)"))
    scale in (:symlog, :linear) ||
        throw(ArgumentError("scale must be :symlog or :linear, got $(scale)"))
    at_geos_path = expanduser(String(at_geos_path))
    at_era5_path = expanduser(String(at_era5_path))
    gc_dir = expanduser(String(gc_dir))
    out_dir = expanduser(String(out_dir))
    mkpath(out_dir)

    pairs = _matched_catrine_times(at_era5_path, gc_dir, max_frames)
    isempty(pairs) && throw(ArgumentError("no matched CATRINE AT/GEOS-Chem frames"))

    lons = lats = corner_lons = corner_lats = nothing
    NCDataset(at_era5_path, "r") do ds
        lons = _read_cs3(ds, "lons")
        lats = _read_cs3(ds, "lats")
        corner_lons = _read_cs3(ds, "corner_lons")
        corner_lats = _read_cs3(ds, "corner_lats")
    end

    section_lats = Float64.(collect(latitudes))
    sections = _build_sections(lons, lats, section_lats, Float64(dlon))
    p_grid = collect(0.0:Float64(dp):1000.0)
    polygons, poly_indices = _cs_cell_polygons(corner_lons, corner_lats)
    lon_grid = sections[1].lons

    t_pre = time()
    precomp_e = _precompute_catrine_frames(at_era5_path, pairs, species, sections, p_grid)
    precomp_g = _precompute_catrine_frames(at_geos_path, pairs, species, sections, p_grid)
    @info @sprintf("  Pre-computed %d frame(s) ×2 (GEOS+ERA5) in %.1fs", length(pairs), time() - t_pre)
    _, dt0, _ = first(pairs)

    # Three data rows, top→bottom: GeosChem, AT(GEOS-IT), AT(ERA5).
    row_col = (precomp_e.gc_col, precomp_g.at_col, precomp_e.at_col)
    row_cur = (precomp_e.gc_curtains, precomp_g.at_curtains, precomp_e.at_curtains)

    if auto_range_day1 !== nothing
        low_pct  = Float64(auto_range_day1[1])
        high_pct = Float64(auto_range_day1[2])
        n_day1 = min(8, length(pairs))
        col_vals = Float64[]
        for i in 1:n_day1, rc in row_col
            append!(col_vals, vec(rc[i]))
        end
        map_vmin, map_vmax = _percentile_range(col_vals, low_pct, high_pct)
        cur_vals = Float64[]
        for i in 1:n_day1, rcur in row_cur, s_idx in eachindex(sections)
            append!(cur_vals, vec(rcur[i][s_idx]))
        end
        curtain_vmin, curtain_vmax = _percentile_range(cur_vals, low_pct, high_pct)
        @info @sprintf("  Auto-range day-1 %.1f-%.1f pct: map=[%.3f, %.3f]  curtain=[%.3f, %.3f]",
                        low_pct, high_pct, map_vmin, map_vmax, curtain_vmin, curtain_vmax)
    end

    cmap = scale === :linear ?
        Makie.cgrad(:RdBu_10, rev=true) :
        Makie.cgrad([:white, "#fee8a8", "#fca85d", "#e34a33", "#7f0000"])
    map_color(field) = scale === :linear ?
        _polygon_colors_linear(field, poly_indices, map_vmin, map_vmax) :
        _polygon_colors(field, poly_indices, map_vmax)
    curtain_norm(values) = scale === :linear ?
        _linear_array(values, curtain_vmin, curtain_vmax) :
        _symlog_array(values, curtain_vmax)

    map_obs = [Makie.Observable(map_color(row_col[r][1])) for r in 1:3]
    curtain_obs = [[Makie.Observable(curtain_norm(row_cur[r][1][i])) for i in eachindex(sections)] for r in 1:3]
    tracer_label = _catrine_tracer_label(species)
    title_obs = Makie.Observable("Time: $(Dates.format(dt0, _CATRINE_TITLE_DATEFORMAT)) UTC    Tracer: $(tracer_label)")

    bg_top = "#edf1f5"
    bg_bot = "#f3eee5"
    header_color = "#1f2328"
    fig = Makie.Figure(size=size, fontsize=14, backgroundcolor=bg_top, figure_padding=0)
    Makie.rowgap!(fig.layout, 0)
    Makie.colgap!(fig.layout, 0)
    Makie.Box(fig[0, 1:6]; color=(header_color, 1.0), strokecolor=(:transparent, 0.0))
    Makie.Label(fig[0, 1:6], title_obs; fontsize=22, font=:bold, color=:white)

    coastlines = _load_robinson_coastlines()
    map_axes = Vector{Any}(undef, 3)
    map_plots = Vector{Any}(undef, 3)
    for r in 1:3
        rows = (3r - 2):(3r)   # 1:3, 4:6, 7:9
        Makie.Label(fig[rows, 1], row_labels[r]; rotation=pi / 2, fontsize=20,
                    font=:bold, tellwidth=false)
        ax = Makie.Axis(fig[rows, 2]; title = r == 1 ? "Column mean" : "",
                        aspect=Makie.DataAspect(), backgroundcolor=:transparent)
        map_plots[r] = Makie.poly!(ax, polygons; color=map_obs[r], colormap=cmap,
                                   colorrange=(0, 1), strokewidth=0)
        _decorate_robinson_axis!(ax, section_lats)
        _draw_robinson_coastlines!(ax, coastlines)
        map_axes[r] = ax
    end

    if scale === :linear
        tick_values_map = collect(range(Float64(map_vmin), Float64(map_vmax); length=5))
        tick_pos_map = [_linear01(v, map_vmin, map_vmax) for v in tick_values_map]
    else
        tick_values_map = [0.0, 0.05, 0.1, 0.5, 1, 2, 4, Float64(map_vmax)]
        tick_pos_map = [_symlog01(v, map_vmax) for v in tick_values_map]
    end
    Makie.Colorbar(fig[1:9, 3], map_plots[1]; label="X$(species) [ppm]",
                   ticks=(tick_pos_map, string.(tick_values_map)), width=14)

    # 9 curtain panels: 3 sections per data row, in column 4, rows 1..9.
    ncur = 3 * length(sections)
    curtain_axes = Vector{Any}(undef, ncur)
    curtain_plots = Vector{Any}(undef, ncur)
    k = 1
    for r in 1:3
        for col in eachindex(sections)
            ax = Makie.Axis(fig[k, 4]; title = _lat_label(sections[col].lat),
                            ylabel = k == 5 ? "pressure [hPa]" : "",
                            xlabel = k == ncur ? "longitude" : "",
                            yreversed=true, backgroundcolor=:transparent)
            Makie.hidespines!(ax)
            ax.xticksvisible[] = k == ncur
            ax.xticklabelsvisible[] = k == ncur
            ax.yaxisposition[] = :right
            ax.xticks = -180:60:180
            ax.yticks = [0, 250, 500, 750, 1000]
            ax.xgridvisible[] = true
            ax.ygridvisible[] = true
            ax.xgridcolor[] = (:gray35, 0.18)
            ax.ygridcolor[] = (:gray35, 0.22)
            ax.xgridwidth[] = 0.6
            ax.ygridwidth[] = 0.6
            (k % length(sections) == 0) || (ax.yticklabelsvisible[] = false)
            hm = Makie.heatmap!(ax, lon_grid, p_grid, curtain_obs[r][col];
                                colormap=cmap, colorrange=(0, 1), nan_color=:white)
            _draw_curtain_guides!(ax)
            curtain_axes[k] = ax
            curtain_plots[k] = hm
            k += 1
        end
    end

    if scale === :linear
        tick_values_c = collect(range(Float64(curtain_vmin), Float64(curtain_vmax); length=6))
        tick_pos_c = [_linear01(v, curtain_vmin, curtain_vmax) for v in tick_values_c]
    else
        tick_values_c = [0.0, 0.1, 0.5, 1, 5, 10, 20, Float64(curtain_vmax)]
        tick_pos_c = [_symlog01(v, curtain_vmax) for v in tick_values_c]
    end
    Makie.Colorbar(fig[1:9, 5], curtain_plots[1]; label="$(species) [ppm]",
                   ticks=(tick_pos_c, string.(tick_values_c)), width=14)
    Makie.colsize!(fig.layout, 1, 54)
    Makie.colsize!(fig.layout, 2, Makie.Relative(0.43))
    Makie.colsize!(fig.layout, 6, 110)
    Makie.resize_to_layout!(fig)
    x0, x1 = _fig_x_extent(fig)
    fig_y0, _ = _fig_y_extent(fig)
    # Tint the two AT sections (rows 4:9) with the warm background; GeosChem
    # (rows 1:3) keeps the cool page colour. The rect spans from the bottom of
    # the figure up to the top of the AT(GEOS-IT) map block.
    _, lower_top = _axis_block_y_extent(map_axes[2])
    _page_rect!(fig, x0, x1, fig_y0, lower_top; color=(bg_bot, 1.0))

    function update_frame!(frame)
        _, dt, _ = pairs[frame]
        for r in 1:3
            map_obs[r][] = map_color(row_col[r][frame])
            for i in eachindex(sections)
                curtain_obs[r][i][] = curtain_norm(row_cur[r][frame][i])
            end
        end
        title_obs[] = "Time: $(Dates.format(dt, _CATRINE_TITLE_DATEFORMAT)) UTC    Tracer: $(tracer_label)"
        return nothing
    end

    scale_tag = scale === :linear ? "linear" : "symlog"
    stem = "$(species)_column_map_curtains_$(scale_tag)_3way_makie"
    png = joinpath(out_dir, "$(stem)_first_frame.png")
    gif = joinpath(out_dir, "$(stem).gif")
    Makie.save(png, fig)
    if write_animation
        Makie.record(fig, gif, eachindex(pairs); framerate=fps) do frame
            update_frame!(frame)
        end
    end
    return (; fig, png, gif = write_animation ? gif : nothing, frames = length(pairs))
end

end # module AtmosTransportMakieExt
