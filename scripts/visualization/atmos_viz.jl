#!/usr/bin/env julia
# Topology-aware snapshot visualization CLI.

using ArgParse

function _requested_backend(argv)
    for i in eachindex(argv)
        if argv[i] == "--backend" && i < length(argv)
            return lowercase(argv[i + 1])
        end
    end
    return "cairo"
end

const _BACKEND_NAME = _requested_backend(ARGS)
if _BACKEND_NAME == "cairo"
    using CairoMakie
    const VIZ_BACKEND = CairoMakie
elseif _BACKEND_NAME == "gl"
    using GLMakie
    const VIZ_BACKEND = GLMakie
else
    error("--backend must be cairo or gl")
end

using AtmosTransport
using AtmosTransport.Visualization

function _parse_resolution(s::AbstractString)
    m = match(r"^(\d+)x(\d+)$", lowercase(s))
    m === nothing && throw(ArgumentError("--resolution must look like 360x181"))
    return (parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

function _settings()
    s = ArgParseSettings(description = "Plot AtmosTransport snapshot NetCDF files.")
    @add_arg_table! s begin
        "--input"
            help = "Snapshot NetCDF file"
            arg_type = String
            required = true
        "--tracer"
            help = "Tracer/variable name, e.g. co2 or co2_natural"
            arg_type = String
            required = true
        "--out"
            help = "Output PNG, PDF, GIF, or MP4 path"
            arg_type = String
            required = true
        "--kind"
            help = "Output kind: frame, grid, or movie"
            arg_type = String
            default = "grid"
        "--transform"
            help = "Field transform: column_mean, surface_slice, level_slice, column_sum"
            arg_type = String
            default = "column_mean"
        "--level"
            help = "Vertical level for level_slice or surface_slice override"
            arg_type = Int
            default = 0
        "--times"
            help = "Comma-separated snapshot hours or indices; default all"
            arg_type = String
            default = "all"
        "--cols"
            help = "Panel columns for grids/multi-panel movies"
            arg_type = Int
            default = 4
        "--fps"
            help = "Movie frame rate"
            arg_type = Float64
            default = 8.0
        "--resolution"
            help = "Debug raster resolution for CS, e.g. 360x181"
            arg_type = String
            default = "360x181"
        "--backend"
            help = "Makie backend: cairo or gl"
            arg_type = String
            default = "cairo"
        "--ppm"
            help = "Scale VMR-like fields to ppm"
            action = :store_true
        "--colormap"
            help = "Makie colormap symbol"
            arg_type = String
            default = "viridis"
    end
    return s
end

function _parse_times(raw::AbstractString)
    lowercase(raw) == "all" && return AllFrames()
    vals = split(raw, ",")
    return [parse(Float64, strip(v)) for v in vals if !isempty(strip(v))]
end

function main(argv=ARGS)
    opts = parse_args(argv, _settings())
    lowercase(opts["backend"]) == _BACKEND_NAME ||
        throw(ArgumentError("backend is selected during script startup; restart with --backend $(opts["backend"])"))
    backend = VIZ_BACKEND

    snapshot = open_snapshot(expanduser(opts["input"]))
    level = opts["level"] == 0 ? nothing : opts["level"]
    plot_spec = PlotSpec(Symbol(opts["tracer"]);
                         transform=opts["transform"], level,
                         unit=opts["ppm"] ? PPMUnit() : NativeUnit())
    transform = plot_spec.transform
    unit = plot_spec.unit
    times = _parse_times(opts["times"])
    resolution = _parse_resolution(opts["resolution"])
    colormap = Symbol(opts["colormap"])
    out = expanduser(opts["out"])
    mkpath(dirname(out))

    kind = lowercase(opts["kind"])
    if kind == "frame"
        first_time = times isa AllFrames ? 1 : first(times)
        field = fieldview(snapshot, opts["tracer"];
                          transform=transform,
                          time=first_time,
                          level=level,
                          unit=unit)
        fig = mapplot(field; resolution=resolution, colormap=colormap)
        backend.save(out, fig)
    elseif kind == "grid"
        fig = snapshot_grid(snapshot, opts["tracer"];
                            transform=transform,
                            times=times,
                            level=level,
                            unit=unit,
                            cols=opts["cols"],
                            resolution=resolution,
                            colormap=colormap)
        backend.save(out, fig)
    elseif kind == "movie"
        movie(snapshot, opts["tracer"], out;
              transform=transform,
              times=times,
              level=level,
              unit=unit,
              fps=opts["fps"],
              resolution=resolution,
              colormap=colormap)
    else
        throw(ArgumentError("--kind must be frame, grid, or movie"))
    end

    @info "Saved $(out)"
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
