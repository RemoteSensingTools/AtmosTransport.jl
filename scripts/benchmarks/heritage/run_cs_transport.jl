#!/usr/bin/env julia
#
# Low-level cubed-sphere advection benchmark runner (CPU + GPU).
#
# Usage:
#   julia --project=. scripts/benchmarks/heritage/run_cs_transport.jl config/runs/completed_experiments/catrine_2day_cs_ppm.toml

using Printf, TOML, NCDatasets, Adapt

if !isempty(ARGS)
    _cfg_path = expanduser(ARGS[1])
    if isfile(_cfg_path)
        _cfg = try TOML.parsefile(_cfg_path) catch; nothing end
        if _cfg !== nothing
            _arch = get(_cfg, "architecture", Dict{String, Any}())
            _use_gpu = Bool(get(_arch, "use_gpu", false))
            _backend = lowercase(String(get(_arch, "backend",
                                            _use_gpu ? "auto" : "cpu")))
            _backend = replace(_backend, '-' => '_', ' ' => '_')
            _gpu_requested = _use_gpu ||
                             _backend in ("auto", "gpu", "cuda", "nvidia",
                                          "metal", "apple", "apple_metal")
            if _gpu_requested
                if _backend in ("metal", "apple", "apple_metal") ||
                   (_backend in ("auto", "gpu") && Sys.isapple())
                    @info "Preloading Metal (GPU backend)"
                    using Metal
                elseif _backend in ("cuda", "nvidia") ||
                       _backend in ("auto", "gpu")
                    @info "Preloading CUDA (GPU backend)"
                    using CUDA
                end
            end
        end
    end
end

include(joinpath(@__DIR__, "..", "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Architectures: architecture_from_config, ensure_runtime!,
    array_adapter, architecture_label, assert_float_type!
using .AtmosTransport.Operators.Advection: fill_panel_halos!,
    strang_split_cs!, CSAdvectionWorkspace,
    LinRoodWorkspace, strang_split_linrood_ppm!,
    UpwindScheme, SlopesScheme, PPMScheme

# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

configured_architecture(cfg) =
    architecture_from_config(get(cfg, "architecture", Dict{String,Any}()))

# ---------------------------------------------------------------------------
# Panel padding (zero-fill halo around raw binary arrays)
# ---------------------------------------------------------------------------

function _pad(a::AbstractArray{T,3}, Hp) where T
    Nx, Ny, Nz = size(a)
    p = zeros(T, Nx + 2Hp, Ny + 2Hp, Nz)
    p[Hp+1:Hp+Nx, Hp+1:Hp+Ny, :] .= a
    return p
end

@inline _scheme_label(scheme::AbstractAdvectionScheme) = nameof(typeof(scheme))
@inline _linrood_order(::LinRoodPPMScheme{ORD}) where ORD = ORD

function _make_cs_advection_workspaces(::AbstractAdvectionScheme, mesh, Nz; FT, array_type)
    return CSAdvectionWorkspace(mesh, Nz; FT, array_type = array_type), nothing
end

function _make_cs_advection_workspaces(::LinRoodPPMScheme, mesh, Nz; FT, array_type)
    return CSAdvectionWorkspace(mesh, Nz; FT, array_type = array_type),
           LinRoodWorkspace(mesh; FT, Nz)
end

# `storage_scale` converts stored flux amounts to legacy per-substep units
# (1 for substep_mass_amount binaries; 1/(2*steps) for full_window ones), so
# both scheme paths below behave identically across flux_kind formats.
function _prepare_cs_flux_panels(::AbstractAdvectionScheme,
                                 pam_w, pbm_w, pcm_w, _pm_w,
                                 Hp, AT, FT, _steps_per_window, _win;
                                 storage_scale = one(FT))
    fs = FT(storage_scale)
    return 1,
           ntuple(p -> AT(_pad(pam_w[p] .* fs, Hp)), 6),
           ntuple(p -> AT(_pad(pbm_w[p] .* fs, Hp)), 6),
           ntuple(p -> AT(_pad(pcm_w[p] .* fs, Hp)), 6)
end

function _prepare_cs_flux_panels(::LinRoodPPMScheme,
                                 pam_w, pbm_w, pcm_w, pm_w,
                                 Hp, AT, FT, steps_per_window, win;
                                 storage_scale = one(FT))
    max_cfl = _linrood_max_cfl(pam_w, pbm_w, pm_w, steps_per_window)
    n_lr = max(1, ceil(Int, max_cfl / 0.85))
    fs = FT(storage_scale) / (steps_per_window * n_lr)
    @printf("  Win %d: CFL %.1f → %d LR sub-passes\n", win, max_cfl, n_lr)
    return n_lr,
           ntuple(p -> AT(_pad(pam_w[p] .* fs, Hp)), 6),
           ntuple(p -> AT(_pad(pbm_w[p] .* fs, Hp)), 6),
           ntuple(p -> AT(_pad(pcm_w[p] .* fs, Hp)), 6)
end

function _apply_cs_transport!(scheme::AbstractAdvectionScheme,
                              rm, pm, pam_p, pbm_p, pcm_p,
                              mesh, ws, _ws_lr)
    strang_split_cs!(rm, pm, pam_p, pbm_p, pcm_p, mesh, scheme, ws;
                     cfl_limit = 0.95)
    return nothing
end

function _apply_cs_transport!(scheme::LinRoodPPMScheme,
                              rm, pm, pam_p, pbm_p, pcm_p,
                              mesh, ws, ws_lr)
    strang_split_linrood_ppm!(rm, pm, pam_p, pbm_p, pcm_p,
                              mesh, Val(_linrood_order(scheme)), ws, ws_lr)
    return nothing
end

# ---------------------------------------------------------------------------
# Snapshot export — GCHP/MAPL-compatible NetCDF
# ---------------------------------------------------------------------------

function _write_snapshots(nc_path, snap_data, snap_m, snapshot_hours, Nc, Nz)
    ntime = length(snap_data)
    cs_lons = [panel_cell_center_lonlat(Nc, p, Float64)[1] for p in 1:6]
    cs_lats = [panel_cell_center_lonlat(Nc, p, Float64)[2] for p in 1:6]
    cs_clons = [panel_cell_corner_lonlat(Nc, p, Float64)[1] for p in 1:6]
    cs_clats = [panel_cell_corner_lonlat(Nc, p, Float64)[2] for p in 1:6]

    isfile(nc_path) && rm(nc_path)
    mkpath(dirname(nc_path))
    ds = NCDataset(nc_path, "c")

    defDim(ds, "Xdim", Nc); defDim(ds, "Ydim", Nc); defDim(ds, "nf", 6)
    defDim(ds, "XCdim", Nc+1); defDim(ds, "YCdim", Nc+1); defDim(ds, "time", ntime)
    ds.attrib["Conventions"] = "CF"
    ds.attrib["Source"] = "AtmosTransport.jl"
    ds.attrib["Nc"] = Nc

    defVar(ds, "Xdim", Float64, ("Xdim",),
           attrib=Dict("units"=>"degrees_east"))[:] = cs_lons[1][:, 1]
    defVar(ds, "Ydim", Float64, ("Ydim",),
           attrib=Dict("units"=>"degrees_north"))[:] = cs_lats[1][1, :]
    defVar(ds, "nf", Int32, ("nf",),
           attrib=Dict("axis"=>"e"))[:] = Int32.(1:6)

    for (var, data, dim) in [("lons", cs_lons, ("Xdim","Ydim","nf")),
                              ("lats", cs_lats, ("Xdim","Ydim","nf")),
                              ("corner_lons", cs_clons, ("XCdim","YCdim","nf")),
                              ("corner_lats", cs_clats, ("XCdim","YCdim","nf"))]
        v = defVar(ds, var, Float64, dim, attrib=Dict("units" =>
            contains(var, "lon") ? "degrees_east" : "degrees_north"))
        for p in 1:6; v[:, :, p] = data[p]; end
    end

    defVar(ds, "time", Float64, ("time",),
           attrib=Dict("units"=>"hours since start"))[:] = snapshot_hours[1:ntime]

    coord_attr = Dict("units"=>"mol mol-1 dry", "coordinates"=>"lons lats")
    for name in keys(first(snap_data))
        col_v = defVar(ds, "$(name)_column_mean", Float64, ("Xdim","Ydim","nf","time"),
                        attrib=coord_attr)
        sfc_v = defVar(ds, "$(name)_surface", Float64, ("Xdim","Ydim","nf","time"),
                        attrib=coord_attr)
        for t in 1:ntime, p in 1:6
            rm, m = snap_data[t][name][p], snap_m[t][p]
            col = dropdims(sum(rm, dims=3), dims=3) ./
                  max.(dropdims(sum(m, dims=3), dims=3), eps(Float64))
            col_v[:, :, p, t] = col
            sfc_v[:, :, p, t] = rm[:, :, Nz] ./ max.(m[:, :, Nz], eps(Float64))
        end
    end
    close(ds)
    println("Saved: $nc_path (C$Nc × 6 × $ntime snapshots)")
end

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

function run_cs(cfg)
    FT = cfg_float_type(cfg)
    arch = configured_architecture(cfg)
    assert_float_type!(arch, FT)
    ensure_runtime!(arch)
    AT = array_adapter(arch)

    input_cfg = get(cfg, "input", Dict{String,Any}())
    binary_paths = [expanduser(String(p)) for p in input_cfg["binary_paths"]]

    run_cfg = get(cfg, "run", Dict{String,Any}())
    scheme = AtmosTransport.Models.build_runtime_advection(
        cfg, AtmosTransport.Models.CubedSphereRuntimeRecipeStyle())
    Hp = configured_halo_width(cfg, scheme)
    start_window = Int(get(run_cfg, "start_window", 1))
    stop_window_override = get(run_cfg, "stop_window", nothing)

    output_cfg = get(cfg, "output", Dict{String,Any}())
    snap_hours = Float64.(get(output_cfg, "snapshot_hours", Float64.(0:6:48)))
    snap_file  = expanduser(get(output_cfg, "snapshot_file", "cs_snapshot.nc"))

    # --- Tracers config ---
    tracers_cfg = get(cfg, "tracers", Dict{String,Any}())
    tracer_defs = if isempty(tracers_cfg)
        Dict(:CO2 => Dict("kind"=>"uniform", "background"=>4.11e-4))
    else
        Dict(Symbol(n) => get(c, "init", Dict("kind"=>"uniform","background"=>0.0))
             for (n, c) in tracers_cfg)
    end

    # --- Grid & scheme ---
    reader1 = TransportBinaryReader(first(binary_paths); FT)
    h = reader1.header
    Nc, Nz = h.geometry.Nc, h.nlevel
    window_hours = h.dt_met_seconds / 3600.0
    mesh = CubedSphereMesh(; Nc, Hp, definition=mesh_definition(reader1))

    println("="^60)
    @printf("CS transport  C%d × %d levels  Hp=%d  %s  %s\n",
            Nc, Nz, Hp, _scheme_label(scheme), architecture_label(arch))
    @printf("Tracers: %s   Binaries: %d   Output: %s\n",
            join(String.(keys(tracer_defs)), ", "), length(binary_paths), snap_file)
    println("="^60)

    # --- Initialise state ---
    pm_w1, _, _, _, _ = load_window!(reader1, 1)
    pm = ntuple(p -> AT(_pad(pm_w1[p], Hp)), 6)
    fill_panel_halos!(pm, mesh; dir=1)

    tracers = Dict{Symbol, NTuple{6, typeof(pm[1])}}()
    for (name, init) in tracer_defs
        kind = Symbol(lowercase(get(init, "kind", "uniform")))
        bg = FT(kind == :catrine_co2 ? 4.11e-4 : get(init, "background", 0.0))
        tracers[name] = ntuple(p -> pm[p] .* bg, 6)
        fill_panel_halos!(tracers[name], mesh; dir=1)
    end

    ws, ws_lr = _make_cs_advection_workspaces(scheme, mesh, Nz; FT, array_type = AT)

    # --- Snapshot storage (always CPU for I/O) ---
    interior(a) = Array(a[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :])
    snap_data = Dict{Symbol, NTuple{6, Array{FT,3}}}[]
    snap_m    = NTuple{6, Array{FT,3}}[]

    function capture!()
        push!(snap_m, ntuple(p -> interior(pm[p]), 6))
        push!(snap_data, Dict(n => ntuple(p -> interior(tracers[n][p]), 6)
                              for n in keys(tracers)))
    end

    snap_idx = 1; total_hour = 0.0
    if snap_idx <= length(snap_hours) && abs(snap_hours[snap_idx]) < 0.5
        capture!(); @printf("  Snapshot %d at t=%.0fh\n", snap_idx, 0.0); snap_idx += 1
    end

    readers = [reader1; [TransportBinaryReader(expanduser(p); FT) for p in binary_paths[2:end]]]
    t0 = time()

    # --- Window loop ---
    for reader in readers
        n_win = reader.header.nwindow
        stop_win = stop_window_override === nothing ? n_win : min(Int(stop_window_override), n_win)

        for win in start_window:stop_win
            steps_per_window = reader.header.steps_per_window_by_window[win]
            pm_w, _, pam_w, pbm_w, pcm_w = load_window!(reader, win)

            for p in 1:6; pm[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :] .= AT(pm_w[p]); end
            fill_panel_halos!(pm, mesh; dir=1)

            n_lr, pam_p, pbm_p, pcm_p =
                _prepare_cs_flux_panels(scheme, pam_w, pbm_w, pcm_w, pm_w,
                                        Hp, AT, FT, steps_per_window, win;
                                        storage_scale =
                                            AtmosTransport.MetDrivers.flux_storage_substep_scale(
                                                FT, steps_per_window,
                                                AtmosTransport.MetDrivers.flux_kind(reader)))

            for _ in 1:steps_per_window, _ in 1:n_lr
                pm_snap = ntuple(p -> copy(pm[p]), 6)
                for (name, rm) in tracers
                    for p in 1:6; pm[p] .= pm_snap[p]; end
                    _apply_cs_transport!(scheme, rm, pm, pam_p, pbm_p, pcm_p,
                                         mesh, ws, ws_lr)
                end
            end

            total_hour += window_hours
            while snap_idx <= length(snap_hours) &&
                  abs(total_hour - snap_hours[snap_idx]) < 0.5
                capture!(); @printf("  Snapshot %d at t=%.0fh\n", snap_idx, total_hour)
                snap_idx += 1
            end
        end
        start_window = 1
    end

    @printf("\nDone: %.1fs (%d windows)\n", time() - t0, length(snap_data))
    _write_snapshots(snap_file, snap_data, snap_m, snap_hours, Nc, Nz)

    for (name, rm) in tracers
        total = sum(sum(Array(rm[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :])) for p in 1:6)
        @printf("  %s total mass: %.12e kg\n", name, total)
    end
end

# ---------------------------------------------------------------------------
# LinRood CFL estimate
# ---------------------------------------------------------------------------

function _linrood_max_cfl(pam_w, pbm_w, pm_raw, steps_per_window)
    cfl = 0.0
    for p in 1:6
        Nc_f = size(pam_w[p], 1) - 1
        for k in axes(pam_w[p], 3), j in 1:Nc_f, i in 1:Nc_f+1
            f = abs(pam_w[p][i, j, k]) / steps_per_window
            d = (i >= 2 && i <= Nc_f) ? pm_raw[p][min(i, Nc_f), j, k] :
                                        pm_raw[p][max(1, i-1), j, k]
            d > 0 && (cfl = max(cfl, f / d))
        end
        Nc_g = size(pbm_w[p], 2) - 1
        for k in axes(pbm_w[p], 3), j in 1:Nc_g+1, i in 1:Nc_g
            f = abs(pbm_w[p][i, j, k]) / steps_per_window
            d = (j >= 2 && j <= Nc_g) ? pm_raw[p][i, min(j, Nc_g), k] :
                                        pm_raw[p][i, max(1, j-1), k]
            d > 0 && (cfl = max(cfl, f / d))
        end
    end
    return cfl
end

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

cfg_float_type(cfg) = let s = get(get(cfg, "numerics", Dict()), "float_type", "Float64")
    s == "Float32" ? Float32 : Float64
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function main()
    isempty(ARGS) && error("Usage: julia --project=. scripts/benchmarks/heritage/run_cs_transport.jl <config.toml>")
    cfg = TOML.parsefile(expanduser(ARGS[1]))
    run_cs(cfg)
end

main()
