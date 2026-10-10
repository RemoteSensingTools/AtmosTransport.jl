#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plot a two-day CS column-mean surface-emission adjoint footprint.
#
# The default map computes one receptor column and plots the full accumulated
# d(column-mean mixing ratio) / dE field over a user-defined lookback window.
# Receptor-grid averaging is available with --grid.
#
# Usage:
#   julia --project=docs scripts/diagnostics/heritage/plot_cs_column_adjoint_2day_map.jl \
#       --out artifacts/cs_column_adjoint_2day_map.png
# ---------------------------------------------------------------------------

using CairoMakie
using Printf
import Adapt

const _CUDA_RUNTIME = Ref{Any}(nothing)

function _argv_requests_cuda(argv)
    for i in 1:length(argv) - 1
        argv[i] == "--backend" && return Symbol(argv[i + 1]) === :cuda
    end
    return false
end

if _argv_requests_cuda(ARGS)
    try
        @eval using CUDA
        CUDA.functional() || error("CUDA runtime is not functional on this host")
        CUDA.allowscalar(false)
        _CUDA_RUNTIME[] = CUDA
    catch err
        error("failed to initialize CUDA for --backend cuda: $err")
    end
end

try
    @eval using AtmosTransport
    @eval using AtmosTransport.Operators.Advection: fill_panel_halos!
    @eval using AtmosTransport.Operators.Diffusion: fill_dz_hydrostatic_constT!
catch
    include(joinpath(@__DIR__, "..", "..", "..", "src", "AtmosTransport.jl"))
    @eval using .AtmosTransport
    @eval using .AtmosTransport.Operators.Advection: fill_panel_halos!
    @eval using .AtmosTransport.Operators.Diffusion: fill_dz_hydrostatic_constT!
end

const USAGE = """
Usage: julia --project=docs scripts/diagnostics/heritage/plot_cs_column_adjoint_2day_map.jl \\
           [--out <png>] [--nc N] [--days D] [--dt-hours H] \\
           [--grid-spacing DEG] [--threshold X] [--grid] [--single] \\
           [--global|--regional] [--map-resolution DEG] \\
           [--log-mode positive|floor|signed] [--log-decades N] \\
           [--cs-binary PATH] [--start-window N] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] \\
           [--horizontal-cfl X] [--vertical-cfl X] [--float-type Float32|Float64] \\
           [--receptor-lon LON] [--receptor-lat LAT] \\
           [--physics transport|diffusion|full] \\
           [--diffusion-kind tm5_beljaars_viterbo|geoschem_holtslag_boville] \\
           [--scheme upwind|slopes|ppm_unlimited|ppm_limited|linrood] \\
           [--backend cpu|cuda] [--tape-storage auto|device|pinned_host]
"""

function _parse_args(argv)
    out = joinpath("artifacts", "cs_column_adjoint_2day_map.png")
    nc = 180
    days = 2.0
    dt_hours = 1.0
    grid_spacing = 20.0
    threshold = 0.0
    map_resolution = 0.5
    log_mode = :floor
    log_decades = 12.0
    cs_binary = nothing
    start_window = 1
    start_date = nothing
    end_date = nothing
    receptor_lon = -95.0
    receptor_lat = 40.0
    mode = :single
    global_view = true
    physics = :transport
    diffusion_kind = :tm5_beljaars_viterbo
    scheme = :upwind
    backend = :cpu
    tape_storage = :auto
    horizontal_cfl = 0.35
    vertical_cfl = 0.01
    float_type = Float32
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--out" && i + 1 <= length(argv)
            out = argv[i + 1]
            i += 2
        elseif arg == "--nc" && i + 1 <= length(argv)
            nc = parse(Int, argv[i + 1])
            i += 2
        elseif arg == "--days" && i + 1 <= length(argv)
            days = parse(Float64, argv[i + 1])
            i += 2
        elseif arg == "--dt-hours" && i + 1 <= length(argv)
            dt_hours = parse(Float64, argv[i + 1])
            i += 2
        elseif arg == "--grid-spacing" && i + 1 <= length(argv)
            grid_spacing = parse(Float64, argv[i + 1])
            i += 2
        elseif arg == "--threshold" && i + 1 <= length(argv)
            threshold = parse(Float64, argv[i + 1])
            i += 2
        elseif arg == "--map-resolution" && i + 1 <= length(argv)
            map_resolution = parse(Float64, argv[i + 1])
            i += 2
        elseif arg == "--log-mode" && i + 1 <= length(argv)
            log_mode = Symbol(argv[i + 1])
            log_mode in (:positive, :floor, :signed) ||
                error("--log-mode must be positive, floor, or signed")
            i += 2
        elseif arg == "--log-decades" && i + 1 <= length(argv)
            log_decades = parse(Float64, argv[i + 1])
            i += 2
        elseif arg == "--cs-binary" && i + 1 <= length(argv)
            cs_binary = argv[i + 1]
            i += 2
        elseif arg == "--start-window" && i + 1 <= length(argv)
            start_window = parse(Int, argv[i + 1])
            i += 2
        elseif arg == "--start-date" && i + 1 <= length(argv)
            start_date = argv[i + 1]
            i += 2
        elseif arg == "--end-date" && i + 1 <= length(argv)
            end_date = argv[i + 1]
            i += 2
        elseif arg == "--horizontal-cfl" && i + 1 <= length(argv)
            horizontal_cfl = parse(Float64, argv[i + 1])
            i += 2
        elseif arg == "--vertical-cfl" && i + 1 <= length(argv)
            vertical_cfl = parse(Float64, argv[i + 1])
            i += 2
        elseif arg == "--float-type" && i + 1 <= length(argv)
            value = argv[i + 1]
            float_type = value == "Float32" ? Float32 :
                         value == "Float64" ? Float64 :
                         error("--float-type must be Float32 or Float64")
            i += 2
        elseif arg == "--receptor-lon" && i + 1 <= length(argv)
            receptor_lon = parse(Float64, argv[i + 1])
            i += 2
        elseif arg == "--receptor-lat" && i + 1 <= length(argv)
            receptor_lat = parse(Float64, argv[i + 1])
            i += 2
        elseif arg == "--physics" && i + 1 <= length(argv)
            physics = Symbol(argv[i + 1])
            physics in (:transport, :diffusion, :full) ||
                error("--physics must be transport, diffusion, or full")
            i += 2
        elseif arg == "--diffusion-kind" && i + 1 <= length(argv)
            diffusion_kind = Symbol(replace(lowercase(argv[i + 1]), '-' => '_'))
            diffusion_kind in (:tm5_beljaars_viterbo, :geoschem_holtslag_boville) ||
                error("--diffusion-kind must be tm5_beljaars_viterbo or geoschem_holtslag_boville")
            i += 2
        elseif arg == "--scheme" && i + 1 <= length(argv)
            scheme = _parse_scheme(argv[i + 1])
            i += 2
        elseif arg == "--backend" && i + 1 <= length(argv)
            backend = Symbol(argv[i + 1])
            backend in (:cpu, :cuda) ||
                error("--backend must be cpu or cuda")
            i += 2
        elseif arg == "--tape-storage" && i + 1 <= length(argv)
            tape_storage = Symbol(argv[i + 1])
            tape_storage in (:auto, :device, :pinned_host) ||
                error("--tape-storage must be auto, device, or pinned_host")
            i += 2
        elseif arg == "--single"
            mode = :single
            i += 1
        elseif arg == "--grid"
            mode = :grid
            i += 1
        elseif arg == "--global"
            global_view = true
            i += 1
        elseif arg == "--regional"
            global_view = false
            i += 1
        elseif arg in ("-h", "--help")
            println(USAGE)
            exit(0)
        else
            error("Unknown argument `$arg`.\n$USAGE")
        end
    end
    nc >= 4 || error("--nc must be at least 4 for the CS map demo")
    days > 0 || error("--days must be positive")
    dt_hours > 0 || error("--dt-hours must be positive")
    grid_spacing > 0 || error("--grid-spacing must be positive")
    map_resolution > 0 || error("--map-resolution must be positive")
    log_decades > 0 || error("--log-decades must be positive")
    start_window >= 1 || error("--start-window must be positive")
    0 <= threshold <= 1 || error("--threshold must be in [0, 1]")
    0 < horizontal_cfl < 0.95 || error("--horizontal-cfl must be in (0, 0.95)")
    0 <= vertical_cfl < 0.95 || error("--vertical-cfl must be in [0, 0.95)")
    abs(receptor_lat) <= 90 || error("--receptor-lat must be in [-90, 90]")
    return (; out, nc, days, dt_hours, grid_spacing, threshold, map_resolution,
            log_mode, log_decades, cs_binary, start_window,
            start_date, end_date,
            receptor_lon, receptor_lat, mode, physics, scheme, horizontal_cfl,
            vertical_cfl, float_type, global_view, backend, tape_storage,
            diffusion_kind)
end

function _parse_scheme(value::AbstractString)
    normalized = Symbol(replace(lowercase(value), '-' => '_'))
    if normalized === :ppm
        @warn "--scheme ppm currently means unlimited split-sweep PPM in this adjoint diagnostic; use --scheme ppm_unlimited explicitly, or --scheme ppm_limited for monotone-limited split-sweep PPM"
        return :ppm_unlimited
    end
    normalized in (:upwind, :slopes, :ppm_unlimited, :ppm_limited, :linrood) ||
        error("--scheme must be upwind, slopes, ppm_unlimited, ppm_limited, or linrood")
    return normalized
end

abstract type _DiagnosticBackend end
struct _CPUBackend <: _DiagnosticBackend end
struct _CUDABackend <: _DiagnosticBackend end

function _diagnostic_backend(name::Symbol)
    if name === :cpu
        return _CPUBackend()
    elseif name === :cuda
        _load_cuda_runtime!()
        return _CUDABackend()
    else
        error("unknown backend $name")
    end
end

_backend_name(::_CPUBackend) = :cpu
_backend_name(::_CUDABackend) = :cuda
_backend_label(backend::_DiagnosticBackend) = String(_backend_name(backend))

function _load_cuda_runtime!()
    if _CUDA_RUNTIME[] === nothing
        isdefined(@__MODULE__, :CUDA) || Core.eval(@__MODULE__, :(using CUDA))
        cuda = Core.eval(@__MODULE__, :CUDA)
        Base.invokelatest(getproperty(cuda, :functional)) ||
            error("CUDA runtime is not functional on this host")
        Base.invokelatest(getproperty(cuda, :allowscalar), false)
        _CUDA_RUNTIME[] = cuda
    end
    return _CUDA_RUNTIME[]
end

_to_backend_array(::_CPUBackend, a::AbstractArray) = a
_to_backend_array(::_CUDABackend, a::AbstractArray) =
    Base.invokelatest(getproperty(_load_cuda_runtime!(), :CuArray), a)
_to_backend_tuple(backend::_DiagnosticBackend, panels::NTuple{6}) =
    ntuple(p -> _to_backend_array(backend, panels[p]), 6)
_to_backend_steps(backend::_DiagnosticBackend, steps::AbstractVector) =
    [_to_backend_tuple(backend, step) for step in steps]

_adapt_backend(::_CPUBackend, x) = x
_adapt_backend(backend::_CUDABackend, x) =
    x === nothing ? nothing :
    Base.invokelatest(Adapt.adapt, getproperty(_load_cuda_runtime!(), :CuArray), x)

function _adapt_backend_sequence(backend::_DiagnosticBackend, x)
    if x === nothing || x isa NoDiffusion
        return x
    elseif x isa AbstractVector
        cache = IdDict{Any, Any}()
        return [get!(cache, item) do
                    _adapt_backend(backend, item)
                end for item in x]
    else
        return _adapt_backend(backend, x)
    end
end

_sync_backend!(::_CPUBackend) = nothing
function _sync_backend!(::_CUDABackend)
    Base.invokelatest(getproperty(_load_cuda_runtime!(), :synchronize))
    return nothing
end

function _resolved_tape_storage(backend::_DiagnosticBackend, requested::Symbol)
    requested === :auto || return requested
    return backend isa _CUDABackend ? :pinned_host : :device
end

function _demo_problem(; Nc::Int, Nz::Int=5, nsteps::Int, FT=Float32,
                       horizontal_cfl::Real=0.35, vertical_cfl::Real=0.05)
    mesh = CubedSphereMesh(Nc=Nc, Hp=3, FT=FT)
    N = mesh.Nc + 2mesh.Hp
    Hp = mesh.Hp
    hcfl = FT(horizontal_cfl)
    vcfl = FT(vertical_cfl)

    panels_m = ntuple(6) do p
        m = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in 1:N, i in 1:N
            m[i, j, k] = FT(1.0 + 0.25k + 0.01p)
        end
        m
    end
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    fill_panel_halos!(panels_m, mesh; dir=0)
    fill_panel_halos!(panels_rm, mesh; dir=0)

    panels_am_steps = Vector{Any}(undef, nsteps)
    panels_bm_steps = Vector{Any}(undef, nsteps)
    panels_cm_steps = Vector{Any}(undef, nsteps)
    for step in 1:nsteps
        diurnal = FT(sin(2π * (step - 1) / 24))
        panels_am_steps[step] = ntuple(6) do p
            am = zeros(FT, N + 1, N, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + Nc, i in Hp + 1:Hp + Nc + 1
                am[i, j, k] = hcfl * (
                    FT(0.80) +
                    FT(0.10) * sin(FT(0.20step + 0.37p + 0.18j + 0.12k)) +
                    FT(0.05) * diurnal)
            end
            am
        end
        panels_bm_steps[step] = ntuple(6) do p
            bm = zeros(FT, N, N + 1, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + Nc + 1, i in Hp + 1:Hp + Nc
                bm[i, j, k] = FT(0.25) * hcfl *
                              sin(FT(0.16step + 0.29p + 0.22i + 0.08k))
            end
            bm
        end
        panels_cm_steps[step] = ntuple(6) do p
            cm = zeros(FT, N, N, Nz + 1)
            for k in 2:Nz, j in Hp + 1:Hp + Nc, i in Hp + 1:Hp + Nc
                cm[i, j, k] = -vcfl *
                              (one(FT) + FT(0.20) *
                               sin(FT(0.11step + i + j + k + p)))
            end
            cm
        end
    end
    return mesh, panels_m, panels_rm, panels_am_steps, panels_bm_steps, panels_cm_steps
end

function _demo_diffusion(mesh, prototype; kz=2.5, dz=60.0)
    FT = eltype(prototype)
    ws = DiffusionWorkspace(ntuple(_ -> prototype, 6), mesh.Hp, 0)
    for p in 1:6
        fill!(ws.layer_thickness[p], FT(dz))
    end
    kz_field = CubedSphereField(ntuple(_ -> ConstantField{FT, 3}(FT(kz)), 6))
    return ImplicitVerticalDiffusion(; kz_field), ws
end

function _demo_tm5_convection(mesh, panels_m)
    FT = eltype(panels_m[1])
    Nc = mesh.Nc
    Nz = size(panels_m[1], 3)
    entu = ntuple(_ -> begin
        e = zeros(FT, Nc, Nc, Nz)
        e[:, :, 2:min(4, Nz - 1)] .= FT(0.004)
        e
    end, 6)
    detu = ntuple(_ -> begin
        e = zeros(FT, Nc, Nc, Nz)
        e[:, :, 2:min(4, Nz - 1)] .= FT(0.0025)
        e
    end, 6)
    entd = ntuple(_ -> begin
        e = zeros(FT, Nc, Nc, Nz)
        e[:, :, 3:min(4, Nz - 1)] .= FT(0.0015)
        e
    end, 6)
    detd = ntuple(_ -> begin
        e = zeros(FT, Nc, Nc, Nz)
        e[:, :, 3:min(4, Nz - 1)] .= FT(0.0008)
        e
    end, 6)
    forcing = ConvectionForcing(nothing, nothing, (; entu, detu, entd, detd))
    metrics = ntuple(_ -> ones(FT, Nc, Nc), 6)
    ws = TM5Workspace(panels_m; tile_columns=Nc * Nc, cell_metrics=metrics)
    return TM5Convection(), forcing, ws
end

_wrap_lon(lon) = mod(lon + 180.0, 360.0) - 180.0

function _angular_distance_deg(lon1, lat1, lon2, lat2)
    λ1, φ1 = deg2rad(_wrap_lon(lon1)), deg2rad(lat1)
    λ2, φ2 = deg2rad(_wrap_lon(lon2)), deg2rad(lat2)
    c = sin(φ1) * sin(φ2) + cos(φ1) * cos(φ2) * cos(λ1 - λ2)
    return rad2deg(acos(clamp(c, -1.0, 1.0)))
end

function _nearest_cs_cell(mesh, lon, lat)
    best = (dist=Inf, panel=1, i=1, j=1, lon=0.0, lat=0.0)
    for p in 1:6
        lons, lats = panel_cell_center_lonlat(mesh, p)
        for j in 1:mesh.Nc, i in 1:mesh.Nc
            d = _angular_distance_deg(lon, lat, lons[i, j], lats[i, j])
            if d < best.dist
                best = (dist=d, panel=p, i=i, j=j,
                        lon=_wrap_lon(Float64(lons[i, j])), lat=Float64(lats[i, j]))
            end
        end
    end
    return best
end

function _aggregate_footprint(result)
    Nc = size(result.footprints[1][1], 1)
    total = ntuple(_ -> zeros(Float64, Nc, Nc), 6)
    for step in eachindex(result.footprints), p in 1:6
        total[p] .+= Array(result.footprints[step][p])
    end
    return total
end

function _flatten_map(mesh, panels)
    n = 6 * mesh.Nc^2
    lons = Vector{Float64}(undef, n)
    lats = Vector{Float64}(undef, n)
    values = Vector{Float64}(undef, n)
    idx = 0
    for p in 1:6
        panel_lons, panel_lats = panel_cell_center_lonlat(mesh, p)
        for j in 1:mesh.Nc, i in 1:mesh.Nc
            idx += 1
            lons[idx] = _wrap_lon(panel_lons[i, j])
            lats[idx] = panel_lats[i, j]
            values[idx] = panels[p][i, j]
        end
    end
    return lons, lats, values
end

function _log_view(values::AbstractVector, threshold::Real,
                   log_mode::Symbol, log_decades::Real)
    if log_mode === :signed
        absvalues = abs.(values)
        nonzero = [v > 0 for v in absvalues]
        any(nonzero) || error("footprint is exactly zero")
        maxabs = maximum(absvalues[nonzero])
        floor_fraction = 10.0^(-log_decades)
        cutoff = maxabs * max(threshold, floor_fraction)
        keep = [abs(v) >= cutoff for v in values]
        any(keep) || error("threshold removed all signed sensitivities")
        signed_decades = log10(maxabs / cutoff)
        positive = [v > 0 for v in values]
        negative = [v < 0 for v in values]
        return (; keep, colorrange=(-signed_decades, signed_decades),
                maxpos=any(positive) ? maximum(values[positive]) : NaN,
                maxabs, cutoff, npositive=count(positive),
                nnegative=count(negative), nnonzero=count(nonzero),
                nkeep=count(keep), log_mode, log_decades)
    end
    positive = [v > 0 for v in values]
    any(positive) || error("footprint has no positive sensitivities to plot")
    maxpos = maximum(values[positive])
    floor_fraction = log_mode === :floor ? 10.0^(-log_decades) : 0.0
    cutoff = maxpos * max(threshold, floor_fraction)
    keep = [v > 0 && v >= cutoff for v in values]
    any(keep) || error("threshold removed all positive sensitivities")

    hi = log10(maxpos)
    lo = if log_mode === :floor
        log10(cutoff)
    else
        minimum(log10.(values[keep]))
    end
    colorrange = lo == hi ? (lo - one(lo), hi + one(hi)) : (lo, hi)
    return (; keep, colorrange, maxpos, maxabs=maximum(abs, values), cutoff,
            npositive=count(positive), nnegative=count(v -> v < 0, values),
            nnonzero=count(!=(0), values), nkeep=count(keep),
            log_mode, log_decades)
end

function _regular_lonlat_grid(map_resolution::Real)
    nlon = max(1, round(Int, 360 / map_resolution))
    nlat = max(1, round(Int, 180 / map_resolution))
    dlon = 360.0 / nlon
    dlat = 180.0 / nlat
    lons = collect(range(-180.0 + dlon / 2, 180.0 - dlon / 2; length=nlon))
    lats = collect(range(-90.0 + dlat / 2, 90.0 - dlat / 2; length=nlat))
    return lons, lats, dlon, dlat
end

@inline function _lonlat_to_xyz(lon::Real, lat::Real)
    λ = deg2rad(_wrap_lon(lon))
    φ = deg2rad(lat)
    cφ = cos(φ)
    return (cφ * cos(λ), cφ * sin(λ), sin(φ))
end

@inline function _rotate_z(x::Real, y::Real, z::Real, offset_deg::Real)
    θ = deg2rad(offset_deg)
    c = cos(θ)
    s = sin(θ)
    return (c * x - s * y, s * x + c * y, z)
end

function _gnomonic_panel_coords(x::Real, y::Real, z::Real)
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay && ax >= az
        if x >= 0
            return 1, y / x, z / x
        else
            return 3, y / x, -z / x
        end
    elseif ay >= ax && ay >= az
        if y >= 0
            return 2, -x / y, z / y
        else
            return 4, -x / y, -z / y
        end
    else
        if z >= 0
            return 5, y / z, -x / z
        else
            return 6, -y / z, -x / z
        end
    end
end

function _mesh_panel_coords(mesh::CubedSphereMesh, lon::Real, lat::Real)
    x, y, z = _lonlat_to_xyz(lon, lat)
    x, y, z = _rotate_z(x, y, z, -longitude_offset_deg(cs_definition(mesh)))
    gpanel, ξg, ηg = _gnomonic_panel_coords(x, y, z)
    conv = panel_convention(mesh)
    if conv isa GnomonicPanelConvention
        return gpanel, ξg, ηg
    elseif conv isa GEOSNativePanelConvention
        if gpanel == 1
            return 1, ξg, ηg
        elseif gpanel == 2
            return 2, ξg, ηg
        elseif gpanel == 5
            return 3, ηg, -ξg
        elseif gpanel == 3
            return 4, -ηg, ξg
        elseif gpanel == 4
            return 5, -ηg, ξg
        else
            return 6, ξg, ηg
        end
    else
        error("unsupported cubed-sphere panel convention $(typeof(conv))")
    end
end

function _edge_coordinate_index(mesh::CubedSphereMesh, ξ::Real)
    Nc = mesh.Nc
    law = coordinate_law(mesh)
    s = if law isa EquiangularGnomonic
        1.0 + (atan(ξ) + π / 4) * (2Nc / π)
    elseif law isa GMAOEqualDistanceGnomonic
        r = inv(sqrt(3.0))
        α0 = asin(r)
        β = atan(ξ * r / cos(α0))
        (β * Nc / α0 + Nc + 2) / 2
    else
        error("unsupported cubed-sphere coordinate law $(typeof(law))")
    end
    return clamp(floor(Int, s), 1, Nc)
end

function _cell_for_lonlat(mesh::CubedSphereMesh, lon::Real, lat::Real)
    panel, ξ, η = _mesh_panel_coords(mesh, lon, lat)
    return panel, _edge_coordinate_index(mesh, ξ), _edge_coordinate_index(mesh, η)
end

function _rasterize_log_map(mesh::CubedSphereMesh, panels, logview,
                            map_resolution::Real)
    lons, lats, dlon, dlat = _regular_lonlat_grid(map_resolution)
    raster = fill(NaN, length(lons), length(lats))
    cutoff = logview.cutoff
    for jj in eachindex(lats), ii in eachindex(lons)
        p, i, j = _cell_for_lonlat(mesh, lons[ii], lats[jj])
        value = panels[p][i, j]
        if logview.log_mode === :floor
            raster[ii, jj] = log10(max(value, cutoff))
        elseif logview.log_mode === :signed
            av = abs(value)
            raster[ii, jj] = av >= cutoff ?
                sign(value) * (log10(av) - log10(cutoff)) :
                0.0
        elseif value > 0 && value >= cutoff
            raster[ii, jj] = log10(value)
        end
    end
    return lons, lats, raster, dlon, dlat
end

function _log_colorbar_label(logview, objective_label::AbstractString)
    if logview.log_mode === :signed
        return "signed log10(|$objective_label| / floor)"
    elseif logview.log_mode === :floor
        return @sprintf("log10 %s, floored %.0f decades below max", objective_label,
                        logview.log_decades)
    else
        return "log10 positive $objective_label"
    end
end

_map_colormap(logview) = logview.log_mode === :signed ? :balance : :viridis

function _shown_label(logview)
    if logview.log_mode === :signed
        return @sprintf("shown %d/%d nonzero CS cells (positive=%d, negative=%d)",
                        logview.nkeep, logview.nnonzero,
                        logview.npositive, logview.nnegative)
    else
        return @sprintf("shown %d/%d positive CS cells",
                        logview.nkeep, logview.npositive)
    end
end

function _duration_label(nsteps::Int, dt_hours::Real)
    hours = nsteps * Float64(dt_hours)
    if hours < 24
        return @sprintf("%.2f h", hours)
    else
        return @sprintf("%.2f days", hours / 24)
    end
end

function _plot_limits(lons, lats, keep; global_view::Bool)
    global_view && return (-180, 180, -90, 90)
    xs = lons[keep]
    ys = lats[keep]
    lon_min = maximum((-180.0, minimum(xs) - 4.0))
    lon_max = minimum((180.0, maximum(xs) + 4.0))
    lat_min = maximum((-90.0, minimum(ys) - 4.0))
    lat_max = minimum((90.0, maximum(ys) + 4.0))
    if lon_max - lon_min < 12
        mid = (lon_min + lon_max) / 2
        lon_min = max(-180.0, mid - 6.0)
        lon_max = min(180.0, mid + 6.0)
    end
    if lat_max - lat_min < 12
        mid = (lat_min + lat_max) / 2
        lat_min = max(-90.0, mid - 6.0)
        lat_max = min(90.0, mid + 6.0)
    end
    return (lon_min, lon_max, lat_min, lat_max)
end

function _receptor_grid(mesh, spacing)
    lons = collect(-160.0:spacing:160.0)
    lats = collect(-80.0:spacing:80.0)
    receptors = NamedTuple[]
    seen = Set{Tuple{Int, Int, Int}}()
    for lat in lats, lon in lons
        receptor = _nearest_cs_cell(mesh, lon, lat)
        key = (receptor.panel, receptor.i, receptor.j)
        if !(key in seen)
            push!(seen, key)
            push!(receptors, receptor)
        end
    end
    return receptors
end

struct _GridColumnMeanObjective <: AbstractCSFootprintObjective
    receptors::Vector{NamedTuple}
    normalize::Bool
end

function AtmosTransport.Adjoints._validate_objective(obj::_GridColumnMeanObjective,
                                                     mesh::CubedSphereMesh,
                                                     Nz::Int)
    isempty(obj.receptors) && throw(ArgumentError("receptor grid is empty"))
    for receptor in obj.receptors
        1 <= receptor.panel <= 6 ||
            throw(ArgumentError("panel must be in 1:6, got $(receptor.panel)"))
        1 <= receptor.i <= mesh.Nc ||
            throw(ArgumentError("i must be in 1:$(mesh.Nc), got $(receptor.i)"))
        1 <= receptor.j <= mesh.Nc ||
            throw(ArgumentError("j must be in 1:$(mesh.Nc), got $(receptor.j)"))
    end
    return nothing
end

function AtmosTransport.Adjoints._seed_objective!(lambda_panels,
                                                  obj::_GridColumnMeanObjective,
                                                  final_m,
                                                  mesh::CubedSphereMesh)
    FT = eltype(lambda_panels[1])
    for p in 1:6
        fill!(lambda_panels[p], zero(FT))
    end
    scale = obj.normalize ? inv(FT(length(obj.receptors))) : one(FT)
    for receptor in obj.receptors
        p = receptor.panel
        ii = mesh.Hp + receptor.i
        jj = mesh.Hp + receptor.j
        denom = sum(@view final_m[p][ii, jj, :])
        weight = scale / max(FT(denom), eps(FT))
        for k in axes(lambda_panels[p], 3)
            lambda_panels[p][ii, jj, k] += weight
        end
    end
    return nothing
end

function _advection_scheme(scheme::Symbol)
    if scheme === :upwind
        return UpwindScheme()
    elseif scheme === :slopes
        return SlopesScheme(NoLimiter())
    elseif scheme === :ppm_unlimited
        return PPMScheme(NoLimiter())
    elseif scheme === :ppm_limited
        return PPMScheme()
    elseif scheme === :linrood
        error(
            "--scheme linrood is the FV3/Lin-Rood CS PPM path. Its " *
            "adjoint needs a dedicated Lin-Rood horizontal VJP and is not " *
            "wired into this diagnostic yet.")
    else
        error("unknown advection scheme $scheme")
    end
end

_scheme_halo_width(::Val{:upwind}) = 1
_scheme_halo_width(::Val{:slopes}) = 2
_scheme_halo_width(::Val{:ppm_unlimited}) = 3
_scheme_halo_width(::Val{:ppm_limited}) = 3
_scheme_halo_width(::Val{:linrood}) = 3
_scheme_halo_width(scheme::Symbol) = _scheme_halo_width(Val(scheme))

function _physics_kwargs(mesh, panels_rm, panels_m, physics::Symbol,
                         scheme::Symbol, dt, backend::_DiagnosticBackend,
                         tape_storage::Symbol, diffusion_kind::Symbol)
    kwargs = (; scheme=_advection_scheme(scheme), dt=dt,
              tape_storage=tape_storage)
    if physics in (:diffusion, :full)
        diffusion_kind === :tm5_beljaars_viterbo ||
            error("--diffusion-kind=$diffusion_kind is only supported with --cs-binary; synthetic diagnostics use tm5_beljaars_viterbo")
        diffusion_op, diffusion_ws = _demo_diffusion(mesh, panels_rm[1])
        kwargs = merge(kwargs, (;
            diffusion_op=_adapt_backend(backend, diffusion_op),
            diffusion_workspace=_adapt_backend(backend, diffusion_ws)))
    end
    if physics == :full
        convection_op, convection_forcing, convection_ws =
            _demo_tm5_convection(mesh, panels_m)
        kwargs = merge(kwargs, (;
            convection_op=_adapt_backend(backend, convection_op),
            convection_forcing=_adapt_backend(backend, convection_forcing),
            convection_workspace=_adapt_backend(backend, convection_ws)))
    end
    return kwargs
end

function _resolve_cs_binary(path::AbstractString)
    expanded = abspath(expanduser(path))
    if isdir(expanded)
        bins = sort(filter(p -> endswith(lowercase(p), ".bin"),
                           readdir(expanded; join=true)))
        isempty(bins) && error("no .bin files found under --cs-binary directory $expanded")
        return first(bins)
    end
    isfile(expanded) || error("--cs-binary path does not exist: $expanded")
    return expanded
end

function _zero_panel_tuple_like(panels::NTuple{6})
    return ntuple(p -> begin
        a = similar(panels[p])
        fill!(a, zero(eltype(a)))
        a
    end, 6)
end

function _copy_haloed_air_mass(window, mesh::CubedSphereMesh)
    panels_m = ntuple(p -> copy(window.air_mass[p]), 6)
    fill_panel_halos!(panels_m, mesh; dir=0)
    return panels_m
end

function _physical_air_mass(panel_m, mesh::CubedSphereMesh)
    if size(panel_m, 1) == mesh.Nc && size(panel_m, 2) == mesh.Nc
        return panel_m
    end
    Hp = mesh.Hp
    return @view panel_m[Hp + 1:Hp + mesh.Nc, Hp + 1:Hp + mesh.Nc, :]
end

function _validate_positive_air_mass!(panels_m::NTuple{6},
                                      mesh::CubedSphereMesh,
                                      context::AbstractString)
    min_m = Inf
    max_m = -Inf
    nbad = 0
    for panel_m in panels_m
        interior = _physical_air_mass(panel_m, mesh)
        nbad += count(!isfinite, interior)
        min_m = min(min_m, Float64(minimum(interior)))
        max_m = max(max_m, Float64(maximum(interior)))
    end
    nbad == 0 && min_m > 0 && return nothing
    error(
        "CS binary has invalid stored air mass at $context: " *
        "min=$min_m max=$max_m nonfinite=$nbad. The checkpointed " *
        "adjoint requires positive replay masses. A GEOS-native " *
        "`chain_mass=true` file can be replay-safe but still nonpositive " *
        "when the pressure-fixer endpoint overshoots; use reset-aware " *
        "adjoint semantics or repair the native mass/flux balance before " *
        "using this diagnostic.")
end

function _replay_safety_tolerance(::Type{Float32})
    return 1.0f-5
end

function _replay_safety_tolerance(::Type{Float64})
    return 1.0e-10
end

function _validate_replay_safe_boundaries!(reader, start_window::Int,
                                           final_window::Int,
                                           ::Type{FT}) where {FT <: AbstractFloat}
    tol = Float64(_replay_safety_tolerance(FT))
    worst = (rel = 0.0, abs = 0.0, window = 0, idx = nothing)
    for win in start_window:final_window
        win < reader.header.nwindow || continue
        cur = load_window!(reader, win)
        nxt = load_window!(reader, win + 1)
        steps = reader.header.steps_per_window_by_window[win]
        # the verifier expects per-substep amounts (it multiplies the
        # divergence by 2*steps); full-window storage must be scaled first.
        # load_window! returns fresh arrays, so in-place is safe.
        fscale = AtmosTransport.MetDrivers.flux_storage_substep_scale(
            FT, steps, AtmosTransport.MetDrivers.flux_kind(reader))
        if fscale != one(FT)
            for pn in 1:6
                cur.am[pn] .*= fscale
                cur.bm[pn] .*= fscale
                cur.cm[pn] .*= fscale
            end
        end
        diag = AtmosTransport.MetDrivers.verify_window_continuity_cs(
            cur.m, cur.am, cur.bm, cur.cm, nxt.m, steps)
        if diag.max_rel_err > worst.rel
            worst = (rel = Float64(diag.max_rel_err),
                     abs = Float64(diag.max_abs_err),
                     window = win,
                     idx = diag.worst_idx)
        end
    end
    worst.window == 0 && return nothing
    worst.rel <= tol || error(
        "CS binary is not replay-safe across stored window boundaries: " *
        "worst rel=$(worst.rel) abs=$(worst.abs) at window $(worst.window) " *
        "cell $(worst.idx), tolerance=$tol. The checkpointed adjoint map " *
        "assumes the stored next-window air mass equals the mass obtained by " *
        "replaying the stored fluxes. Binaries generated with `chain_mass=false` " *
        "and runtime `reset_air_mass_each_window=true` need a reset-aware " *
        "adjoint transpose; use a chained, replay-safe, positive-mass binary " *
        "for this diagnostic until that reset operator is implemented.")
    return nothing
end

function _real_binary_problem(path::AbstractString; start_window::Int,
                              requested_days::Real,
                              FT::Type{<:AbstractFloat},
                              scheme::Symbol,
                              physics::Symbol,
                              diffusion_kind::Symbol)
    physics === :full && error(
        "--physics full is not implemented for --cs-binary in this diagnostic; " *
        "use --physics transport or --physics diffusion")
    bin_path = _resolve_cs_binary(path)
    Hp = _scheme_halo_width(scheme)
    driver = TransportBinaryDriver(bin_path; FT=FT, Hp=Hp,
                                        validate_replay=false)
    try
        step_schedule = steps_per_window_schedule(driver)
        nwin = total_windows(driver)
        1 <= start_window <= nwin ||
            error("--start-window must be in 1:$nwin for $(basename(bin_path))")
        window = load_transport_window(driver, start_window)
        mesh = driver.grid.horizontal
        requested_seconds = Float64(requested_days) * 86400.0
        window_seconds = Float64(window_dt(driver))
        windows_needed = max(1, ceil(Int, requested_seconds / window_seconds))
        final_window = start_window + windows_needed - 1
        if final_window > nwin
            max_days = Float64((nwin - start_window + 1) * window_dt(driver)) / 86400
            error(
                "requested $(requested_days) days reaches window $final_window, " *
                "but $(basename(bin_path)) only has $nwin windows. " *
                "Use --days <= $(max_days), lower --start-window, or pass a " *
                "binary containing more windows.")
        end
        requested_steps = sum(@view step_schedule[start_window:final_window])
        reset_aware = false
        try
            _validate_replay_safe_boundaries!(driver.reader, start_window,
                                              final_window, FT)
        catch err
            if :dm in driver.reader.header.payload_sections
                reset_aware = true
                @warn "CS binary is not replay-continuous; using reset-aware adjoint boundary transpose" reason=sprint(showerror, err)
            else
                rethrow()
            end
        end
        if reset_aware && final_window >= nwin
            error(
                "reset-aware adjoint needs the raw air mass from window " *
                "$(final_window + 1) as the final boundary, but " *
                "$(basename(bin_path)) only has $nwin windows. Lower --days " *
                "or pass a binary containing the next window.")
        end

        panels_m = _copy_haloed_air_mass(window, mesh)
        panels_rm = _zero_panel_tuple_like(panels_m)
        panels_am = Vector{Any}(undef, requested_steps)
        panels_bm = Vector{Any}(undef, requested_steps)
        panels_cm = Vector{Any}(undef, requested_steps)
        diffusion_ops = physics === :diffusion ? Vector{Any}(undef, requested_steps) : nothing
        diffusion_wss = physics === :diffusion ? Vector{Any}(undef, requested_steps) : nothing
        chunks = Any[]

        kwargs_extra = (;)
        step_idx = 0
        for win in start_window:final_window
            win_window = win == start_window ? window : load_transport_window(driver, win)
            _validate_positive_air_mass!(
                win_window.air_mass, mesh,
                "window $win of $(basename(bin_path))")
            steps_this = Int(step_schedule[win])
            # full-window binaries store unscaled met-window amounts; the
            # substep slots below alias these panel arrays, so scale ONCE
            # per loaded window (mirrors DrivenSimulation's refresh scale).
            fscale_win = AtmosTransport.MetDrivers.flux_storage_substep_scale(
                FT, steps_this, AtmosTransport.MetDrivers.flux_kind(driver))
            if fscale_win != one(FT)
                for pn in 1:6
                    win_window.fluxes.am[pn] .*= fscale_win
                    win_window.fluxes.bm[pn] .*= fscale_win
                    win_window.fluxes.cm[pn] .*= fscale_win
                end
            end
            n_this = min(steps_this, requested_steps - step_idx)
            dt_this = Float64(window_dt(driver)) / steps_this
            chunk_m0 = _copy_haloed_air_mass(win_window, mesh)
            chunk_am = Vector{Any}(undef, n_this)
            chunk_bm = Vector{Any}(undef, n_this)
            chunk_cm = Vector{Any}(undef, n_this)
            chunk_diffusion_ops = physics === :diffusion ? Vector{Any}(undef, n_this) : NoDiffusion()
            chunk_diffusion_wss = physics === :diffusion ? Vector{Any}(undef, n_this) : nothing
            if physics === :diffusion
                win_window.surface === nothing && error(
                    "$(basename(bin_path)) does not carry pblh/ustar/pbl_hflux/t2m; " *
                    "cannot run --physics diffusion")
                diffusion_ws = DiffusionWorkspace(panels_rm, mesh.Hp, 0)
                fill_dz_hydrostatic_constT!(diffusion_ws.layer_thickness,
                                            win_window.surface_pressure,
                                            driver.grid.vertical.A,
                                            driver.grid.vertical.B)
                Nz = size(panels_m[1], 3)
                kz_cache = ntuple(_ -> zeros(FT, mesh.Nc, mesh.Nc, Nz), 6)
                if diffusion_kind === :tm5_beljaars_viterbo
                    diffusion_op = ImplicitVerticalDiffusion(;
                        kz_field = WindowPBLKzField(kz_cache))
                    refresh_pbl_kz_cache!(diffusion_op.kz_field, win_window.surface,
                                          chunk_m0, mesh.cell_areas;
                                          halo_width = mesh.Hp)
                elseif diffusion_kind === :geoschem_holtslag_boville
                    diffusion_op = ImplicitVerticalDiffusion(;
                        kz_field = LocalHoltslagBovilleKzField(kz_cache))
                    refresh_local_holtslag_boville_kz_cache!(
                        diffusion_op.kz_field, win_window.surface,
                        win_window.vdiff, chunk_m0, mesh.cell_areas;
                        halo_width = mesh.Hp)
                else
                    error("unknown diffusion kind $diffusion_kind")
                end
            end
            for local_step in 1:n_this
                step_idx += 1
                panels_am[step_idx] = win_window.fluxes.am
                panels_bm[step_idx] = win_window.fluxes.bm
                panels_cm[step_idx] = win_window.fluxes.cm
                chunk_am[local_step] = win_window.fluxes.am
                chunk_bm[local_step] = win_window.fluxes.bm
                chunk_cm[local_step] = win_window.fluxes.cm
                if physics === :diffusion
                    diffusion_ops[step_idx] = diffusion_op
                    diffusion_wss[step_idx] = diffusion_ws
                    chunk_diffusion_ops[local_step] = diffusion_op
                    chunk_diffusion_wss[local_step] = diffusion_ws
                end
            end
            push!(chunks, (;
                window = win,
                panels_m0 = chunk_m0,
                panels_am_steps = chunk_am,
                panels_bm_steps = chunk_bm,
                panels_cm_steps = chunk_cm,
                diffusion_op = chunk_diffusion_ops,
                diffusion_workspace = chunk_diffusion_wss,
                dt = dt_this))
        end
        if physics === :diffusion
            kwargs_extra = (;
                diffusion_op = diffusion_ops,
                diffusion_workspace = diffusion_wss)
        end
        final_window_state = if final_window < nwin
            load_transport_window(driver, final_window + 1)
        else
            nothing
        end
        if final_window_state !== nothing
            _validate_positive_air_mass!(
                final_window_state.air_mass, mesh,
                "final boundary window $(final_window + 1) of $(basename(bin_path))")
        end
        final_m = final_window_state === nothing ? nothing :
            _copy_haloed_air_mass(final_window_state, mesh)

        h = driver.reader.header
        boundary_label = reset_aware ? "reset-aware boundaries" : "replay-continuous"
        source_label = @sprintf(
            "real %s; windows %d-%d/%d; native %.0f s window, %d:%d substeps/window; %s; %s; %s",
            basename(bin_path), start_window, final_window, nwin,
            Float64(window_dt(driver)),
            minimum(@view step_schedule[start_window:final_window]),
            maximum(@view step_schedule[start_window:final_window]),
            String(h.geometry.panel_convention),
            String(h.geometry.definition),
            boundary_label)
        return (; mesh, panels_m, panels_rm,
                panels_am_steps = panels_am,
                panels_bm_steps = panels_bm,
                panels_cm_steps = panels_cm,
                dt = Float64(window_dt(driver)) / step_schedule[start_window],
                nsteps = requested_steps,
                source_label, kwargs_extra, chunks, final_m, reset_aware)
    finally
        close(driver)
    end
end

function _column_footprint(mesh, panels_rm, panels_m, panels_am, panels_bm,
                           panels_cm, receptor, kwargs)
    objective = CSColumnMeanObjective(receptor.panel, receptor.i, receptor.j)
    result = cs_surface_emission_footprint(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh,
        objective; kwargs...)
    return _aggregate_footprint(result)
end

function _stage_chunk_for_backend(chunk, backend::_DiagnosticBackend)
    return (;
        window = chunk.window,
        dt = chunk.dt,
        panels_m0 = _to_backend_tuple(backend, chunk.panels_m0),
        panels_am_steps = _to_backend_steps(backend, chunk.panels_am_steps),
        panels_bm_steps = _to_backend_steps(backend, chunk.panels_bm_steps),
        panels_cm_steps = _to_backend_steps(backend, chunk.panels_cm_steps),
        diffusion_op = _adapt_backend_sequence(backend, chunk.diffusion_op),
        diffusion_workspace = _adapt_backend_sequence(backend, chunk.diffusion_workspace))
end

function _apply_reset_air_mass_adjoint!(lambda_panels, reset_target_m,
                                        replayed_m, mesh::CubedSphereMesh)
    FT = eltype(lambda_panels[1])
    floor_m = eps(FT)
    @inbounds for p in 1:6
        λ = lambda_panels[p]
        new_m = reset_target_m[p]
        old_m = replayed_m[p]
        @. λ = ifelse(old_m > floor_m, λ * new_m / old_m, zero(FT))
    end
    return lambda_panels
end

function _column_footprint_checkpointed(mesh::CubedSphereMesh, chunks,
                                        final_m, receptor, scheme::Symbol,
                                        _dt, backend::_DiagnosticBackend,
                                        tape_storage::Symbol,
                                        reset_aware::Bool = false)
    final_m === nothing && error(
        "checkpointed real-binary footprint currently needs the final state " *
        "to coincide with a stored met-window boundary")
    objective = CSColumnMeanObjective(receptor.panel, receptor.i, receptor.j)
    final_m_backend = _to_backend_tuple(backend, final_m)
    FT = eltype(final_m_backend[1])
    lambda_panels = ntuple(p -> begin
        a = similar(final_m_backend[p])
        fill!(a, zero(FT))
        a
    end, 6)
    AtmosTransport.Adjoints._seed_objective!(lambda_panels, objective,
                                             final_m_backend, mesh)
    reset_target_m = final_m_backend
    total = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
    adv_scheme = _advection_scheme(scheme)
    storage = tape_storage === :pinned_host ?
        PinnedHostCSTapeStorage() :
        tape_storage
    for chunk in Iterators.reverse(chunks)
        staged = _stage_chunk_for_backend(chunk, backend)
        @info "processing checkpoint chunk" window=staged.window backend=_backend_name(backend) tape_storage
        chunk_rm0 = _zero_panel_tuple_like(staged.panels_m0)
        ops, chunk_end_m = AtmosTransport.Adjoints._record_cs_adjoint_tape(
            chunk_rm0,
            staged.panels_m0,
            staged.panels_am_steps,
            staged.panels_bm_steps,
            staged.panels_cm_steps,
            mesh,
            adv_scheme;
            cfl_limit = 0.95,
            flux_scale = one(FT),
            dt = FT(staged.dt),
            diffusion_op = staged.diffusion_op,
            diffusion_workspace = staged.diffusion_workspace,
            tape_storage = storage)
        if reset_aware
            _apply_reset_air_mass_adjoint!(lambda_panels, reset_target_m,
                                           chunk_end_m, mesh)
        end
        result = AtmosTransport.Adjoints._collect_surface_footprints(
            lambda_panels, ops, staged.panels_m0, mesh,
            CSSeedObjective(), FT(staged.dt);
            diffusion_workspace = nothing,
            diffusion_meteo = nothing,
            convection_workspace = nothing)
        for step in eachindex(result.footprints), p in 1:6
            total[p] .+= Array(result.footprints[step][p])
        end
        _sync_backend!(backend)
        reset_target_m = staged.panels_m0
        GC.gc()
    end
    return total
end

function _plot_single(out::AbstractString; Nc::Int, days::Real, dt_hours::Real,
                      receptor_lon::Real, receptor_lat::Real, threshold::Real,
                      physics::Symbol, scheme::Symbol, horizontal_cfl::Real,
                      vertical_cfl::Real, FT::Type{<:AbstractFloat},
                      global_view::Bool, map_resolution::Real,
                      log_mode::Symbol, log_decades::Real,
                      cs_binary, start_window::Int,
                      backend::_DiagnosticBackend,
                      tape_storage::Symbol,
                      diffusion_kind::Symbol)
    tape_storage = _resolved_tape_storage(backend, tape_storage)
    checkpoint_chunks = nothing
    final_m_checkpoint = nothing
    reset_aware = false
    if cs_binary === nothing
        nsteps = round(Int, days * 24 / dt_hours)
        nsteps >= 1 || error("lookback window rounds to zero model steps")
        dt = 3600.0 * dt_hours
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _demo_problem(; Nc=Nc, nsteps=nsteps, FT=FT,
                          horizontal_cfl=horizontal_cfl,
                          vertical_cfl=vertical_cfl)
        panels_m = _to_backend_tuple(backend, panels_m)
        panels_rm = _to_backend_tuple(backend, panels_rm)
        panels_am = _to_backend_steps(backend, panels_am)
        panels_bm = _to_backend_steps(backend, panels_bm)
        panels_cm = _to_backend_steps(backend, panels_cm)
        source_label = @sprintf("synthetic C%d demo winds; CFL=(%.2f horiz, %.2f vert)",
                                Nc, horizontal_cfl, vertical_cfl)
        kwargs_extra = (;)
    else
        problem = _real_binary_problem(String(cs_binary);
            start_window = start_window,
            requested_days = days,
            FT = FT,
            scheme = scheme,
            physics = physics,
            diffusion_kind = diffusion_kind)
        mesh = problem.mesh
        panels_m = problem.panels_m
        panels_rm = problem.panels_rm
        panels_am = problem.panels_am_steps
        panels_bm = problem.panels_bm_steps
        panels_cm = problem.panels_cm_steps
        dt = problem.dt
        dt_hours = dt / 3600
        nsteps = problem.nsteps
        source_label = problem.source_label
        kwargs_extra = problem.kwargs_extra
        checkpoint_chunks = problem.chunks
        final_m_checkpoint = problem.final_m
        reset_aware = problem.reset_aware
        Nc = mesh.Nc
    end
    receptor = _nearest_cs_cell(mesh, receptor_lon, receptor_lat)
    kwargs = cs_binary === nothing ?
        _physics_kwargs(mesh, panels_rm, panels_m, physics, scheme, dt,
                        backend, tape_storage, diffusion_kind) :
        (; scheme=_advection_scheme(scheme), dt=dt, tape_storage=tape_storage)
    kwargs = merge(kwargs, kwargs_extra)
    @info "computing single-receptor adjoint footprint" Nc nsteps physics scheme receptor backend=_backend_name(backend) tape_storage
    total = if cs_binary === nothing
        _column_footprint(mesh, panels_rm, panels_m, panels_am, panels_bm,
                          panels_cm, receptor, kwargs)
    else
        Base.invokelatest(_column_footprint_checkpointed, mesh,
                          checkpoint_chunks, final_m_checkpoint, receptor,
                          scheme, dt, backend, tape_storage, reset_aware)
    end
    lons, lats, values = _flatten_map(mesh, total)

    logview = _log_view(values, threshold, log_mode, log_decades)
    raster_lons, raster_lats, raster, dlon, dlat =
        _rasterize_log_map(mesh, total, logview, map_resolution)
    limits = _plot_limits(lons, lats, logview.keep; global_view)

    fig = Figure(size=(1260, 700))
    ax = Axis(fig[1, 1],
        xlabel="longitude",
        ylabel="latitude",
        title="$(_duration_label(nsteps, dt_hours)) full surface-emission adjoint footprint for one column mean",
        limits=limits,
        aspect=DataAspect())
    for lon in -180:20:180
        lines!(ax, [lon, lon], [-90, 90], color=(:gray40, 0.25), linewidth=0.6)
    end
    for lat in -80:10:80
        lines!(ax, [-180, 180], [lat, lat], color=(:gray40, 0.25), linewidth=0.6)
    end
    h = heatmap!(ax, raster_lons, raster_lats, raster;
        colormap=_map_colormap(logview),
        colorrange=logview.colorrange,
        nan_color=(:gray90, 0.0))
    scatter!(ax, [receptor.lon], [receptor.lat];
        color=:black,
        marker=:xcross,
        markersize=12,
        strokewidth=2)
    Colorbar(fig[1, 2], h;
        label=_log_colorbar_label(logview, "Σ previous-window d(column mean) / dE"))

    summary = @sprintf(
        "CS C%d, %d × %.3f h steps; receptor panel %d i=%d j=%d (%.2f°, %.2f°); raster %.3g° × %.3g°; %s (threshold %.2g max; log-mode=%s)\nphysics=%s; scheme=%s; backend=%s; tape=%s; source=%s; max positive=%.3e; max abs=%.3e",
        Nc, nsteps, dt_hours, receptor.panel, receptor.i, receptor.j,
        receptor.lon, receptor.lat, dlon, dlat, _shown_label(logview),
        threshold, String(log_mode), String(physics), String(scheme),
        _backend_label(backend), String(tape_storage), source_label,
        logview.maxpos, logview.maxabs)
    Label(fig[2, 1:2], summary; fontsize=11, tellwidth=false)

    mkpath(dirname(out))
    save(out, fig)
    println(out)
    return out
end

function _plot_grid(out::AbstractString; Nc::Int, days::Real, dt_hours::Real,
                    grid_spacing::Real, threshold::Real, physics::Symbol,
                    scheme::Symbol, horizontal_cfl::Real, vertical_cfl::Real,
                    FT::Type{<:AbstractFloat}, map_resolution::Real,
                    log_mode::Symbol, log_decades::Real,
                    diffusion_kind::Symbol)
    nsteps = round(Int, days * 24 / dt_hours)
    nsteps >= 1 || error("lookback window rounds to zero model steps")
    dt = 3600.0 * dt_hours

    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _demo_problem(; Nc=Nc, nsteps=nsteps, FT=FT,
                      horizontal_cfl=horizontal_cfl,
                      vertical_cfl=vertical_cfl)
    kwargs = _physics_kwargs(mesh, panels_rm, panels_m, physics, scheme, dt,
                             _CPUBackend(), :device, diffusion_kind)
    receptors = _receptor_grid(mesh, grid_spacing)
    objective = _GridColumnMeanObjective(receptors, true)
    @info "computing combined receptor-grid adjoint footprint" receptors=length(receptors) nsteps physics scheme
    result = cs_surface_emission_footprint(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh,
        objective; kwargs...)
    total = _aggregate_footprint(result)
    lons, lats, values = _flatten_map(mesh, total)
    logview = _log_view(values, threshold, log_mode, log_decades)
    raster_lons, raster_lats, raster, dlon, dlat =
        _rasterize_log_map(mesh, total, logview, map_resolution)

    fig = Figure(size=(1240, 680))
    ax = Axis(fig[1, 1],
        xlabel="longitude",
        ylabel="latitude",
        title="$(_duration_label(nsteps, dt_hours)) adjoint footprint for average column mean sampled about every $(grid_spacing)°",
        limits=(-180, 180, -90, 90),
        aspect=DataAspect())
    for lon in -180:60:180
        lines!(ax, [lon, lon], [-90, 90], color=(:gray40, 0.22), linewidth=0.6)
    end
    for lat in -60:30:60
        lines!(ax, [-180, 180], [lat, lat], color=(:gray40, 0.22), linewidth=0.6)
    end
    h = heatmap!(ax, raster_lons, raster_lats, raster;
        colormap=_map_colormap(logview),
        colorrange=logview.colorrange,
        nan_color=(:gray90, 0.0))
    Colorbar(fig[1, 2], h;
        label=_log_colorbar_label(logview, "d(mean sampled column mean) / dE, summed over $(days) days"))

    summary = @sprintf(
        "CS C%d, %d × %.2f h steps = %.1f days; %d unique receptor columns; raster %.3g° × %.3g°; %s (threshold %.2g max; log-mode=%s)\nphysics=%s; scheme=%s; synthetic CFL=(%.2f horiz, %.2f vert); max positive=%.3e; max abs=%.3e",
        Nc, nsteps, dt_hours, nsteps * dt_hours / 24, length(receptors),
        dlon, dlat, _shown_label(logview), threshold, String(log_mode),
        String(physics), String(scheme), horizontal_cfl, vertical_cfl,
        logview.maxpos, logview.maxabs)
    Label(fig[2, 1:2], summary; fontsize=11, tellwidth=false)

    mkpath(dirname(out))
    save(out, fig)
    println(out)
    return out
end

function main(argv=ARGS)
    opts = _parse_args(argv)
    backend = _diagnostic_backend(opts.backend)
    if opts.mode === :single
        return _plot_single(opts.out; Nc=opts.nc, days=opts.days,
                            dt_hours=opts.dt_hours,
                            receptor_lon=opts.receptor_lon,
                            receptor_lat=opts.receptor_lat,
                            threshold=opts.threshold,
                            physics=opts.physics,
                            scheme=opts.scheme,
                            horizontal_cfl=opts.horizontal_cfl,
                            vertical_cfl=opts.vertical_cfl,
                            FT=opts.float_type,
                            global_view=opts.global_view,
                            map_resolution=opts.map_resolution,
                            log_mode=opts.log_mode,
                            log_decades=opts.log_decades,
                            cs_binary=opts.cs_binary,
                            start_window=opts.start_window,
                            backend=backend,
                            tape_storage=opts.tape_storage,
                            diffusion_kind=opts.diffusion_kind)
    else
        opts.backend === :cpu ||
            error("--grid receptor averaging is currently CPU-only; use --single for CUDA maps")
        opts.cs_binary === nothing ||
            error("--grid is not implemented with --cs-binary in this diagnostic")
        return _plot_grid(opts.out; Nc=opts.nc, days=opts.days,
                          dt_hours=opts.dt_hours,
                          grid_spacing=opts.grid_spacing,
                          threshold=opts.threshold,
                          physics=opts.physics,
                          scheme=opts.scheme,
                          horizontal_cfl=opts.horizontal_cfl,
                          vertical_cfl=opts.vertical_cfl,
                          FT=opts.float_type,
                          map_resolution=opts.map_resolution,
                          log_mode=opts.log_mode,
                          log_decades=opts.log_decades,
                          diffusion_kind=opts.diffusion_kind)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
