#!/usr/bin/env julia

using Dates
using Printf
using Statistics
using NCDatasets

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.Grids: CubedSphereMesh, LatLonMesh,
    GnomonicPanelConvention, GEOSNativePanelConvention
using .AtmosTransport.Regridding: build_regridder, apply_regridder!

const R_EARTH_M = 6.371229e6
const GRAV = 9.80665
const PPM = 1e6

const ERA_DIR = get(ENV, "ERA_DIR", "/temp1/c180_full137_3d")
const GEOS_DIR = get(ENV, "GEOS_DIR", "/temp1/c180_geosit_native_3d")
const OUT_DIR = get(ENV, "OUT_DIR", "/temp1/c180_era_geos_window_tendencies")
const LL_NX = parse(Int, get(ENV, "LL_NX", "180"))
const LL_NY = parse(Int, get(ENV, "LL_NY", "90"))
const LAG_STEPS = Tuple(parse.(Int, split(get(ENV, "LAG_STEPS", "1,2,4"), ",")))

const RUNS = Tuple(String.(strip.(split(get(ENV, "RUNS",
    "advonly_ppm,advdiff_ppm,fullphysics_ppm"), ","))))
const TRACERS = Tuple(String.(strip.(split(get(ENV, "TRACERS",
    "co2_natural,co2_fossil"), ","))))
const PRESSURE_TARGETS = (
    ("lower_850hPa", 850.0),
    ("mid_500hPa", 500.0),
    ("upper_250hPa", 250.0),
)
const DIAGNOSTICS = ("column_mean", "surface",
                     PRESSURE_TARGETS[1][1], PRESSURE_TARGETS[2][1],
                     PRESSURE_TARGETS[3][1])

struct SimulationFields
    source::String
    run::String
    path::String
    times::Vector
    area::Vector{Float64}
    fields::Dict{Tuple{String,String}, Matrix{Float64}}
end

function _snapshot_path(root, run)
    path = joinpath(root, run * ".nc")
    isfile(path) || error("Snapshot file not found: $path")
    return path
end

function _source_mesh(ds)
    conv = lowercase(String(get(ds.attrib, "panel_convention", "gnomonic")))
    convention = if conv in ("geos_native", "geosnative", "geos-native")
        GEOSNativePanelConvention()
    elseif conv in ("gnomonic", "gnomic")
        GnomonicPanelConvention()
    else
        error("Unsupported panel_convention=$(conv)")
    end
    return CubedSphereMesh(; FT=Float64, Nc=Int(ds.attrib["Nc"]),
                           convention, radius=R_EARTH_M)
end

function _ll_mesh()
    return LatLonMesh(; FT=Float64, Nx=LL_NX, Ny=LL_NY,
                      longitude=(-180.0, 180.0), latitude=(-90.0, 90.0),
                      radius=R_EARTH_M)
end

function _latlon_cell_areas(mesh::LatLonMesh)
    areas = Matrix{Float64}(undef, mesh.Nx, mesh.Ny)
    dlon = deg2rad(mesh.Δλ)
    @inbounds for j in 1:mesh.Ny
        strip = mesh.radius^2 * dlon *
                (sind(Float64(mesh.φᶠ[j + 1])) - sind(Float64(mesh.φᶠ[j])))
        for i in 1:mesh.Nx
            areas[i, j] = strip
        end
    end
    return areas
end

function _copy_regridded!(dest::Matrix{Float64}, t::Int, tmp::Vector{Float64},
                          regridder, src)
    fill!(tmp, 0.0)
    apply_regridder!(tmp, regridder, vec(src))
    @views dest[:, t] .= tmp
    return nothing
end

function _pressure_target_indices(mass::Array{Float64,4}, area::Array{Float64,3})
    nx, ny, nf, nz = size(mass)
    ntarget = length(PRESSURE_TARGETS)
    indices = Array{Int16}(undef, nx, ny, nf, ntarget)
    best = Vector{Float64}(undef, ntarget)
    @inbounds for p in 1:nf, j in 1:ny, i in 1:nx
        fill!(best, Inf)
        ptop_hpa = 0.0
        a = area[i, j, p]
        for k in 1:nz
            dp_hpa = mass[i, j, p, k] * GRAV / a / 100.0
            pmid_hpa = ptop_hpa + 0.5 * dp_hpa
            for target in 1:ntarget
                diff = abs(pmid_hpa - PRESSURE_TARGETS[target][2])
                if diff < best[target]
                    best[target] = diff
                    indices[i, j, p, target] = Int16(k)
                end
            end
            ptop_hpa += dp_hpa
        end
    end
    return indices
end

function _extract_pressure_field!(dest::Array{Float64,3}, q::Array{Float64,4},
                                  indices::Array{Int16,4}, target::Int)
    nx, ny, nf = size(dest)
    @inbounds for p in 1:nf, j in 1:ny, i in 1:nx
        dest[i, j, p] = q[i, j, p, Int(indices[i, j, p, target])]
    end
    return dest
end

function _load_simulation_fields(source::String, run::String, path::String)
    println("Loading and regridding $source $run")
    ll = _ll_mesh()
    ll_area = vec(_latlon_cell_areas(ll))
    ds = NCDataset(path)
    try
        mesh = _source_mesh(ds)
        regridder = build_regridder(mesh, ll; normalize=false)
        times = collect(ds["time"][:])
        nll = LL_NX * LL_NY
        nt = length(times)
        fields = Dict{Tuple{String,String}, Matrix{Float64}}()
        for tracer in TRACERS, diagnostic in DIAGNOSTICS
            fields[(tracer, diagnostic)] = Matrix{Float64}(undef, nll, nt)
        end

        area = Float64.(ds["cell_area"][:, :, :])
        tmp = zeros(Float64, nll)
        pressure_field = similar(area)

        for t in 1:nt
            println("  $source $run snapshot $t/$nt")
            mass = Float64.(ds["air_mass"][:, :, :, :, t])
            pressure_indices = _pressure_target_indices(mass, area)
            nz = size(mass, 4)

            for tracer in TRACERS
                column_var = tracer * "_column_mean"
                column = Float64.(ds[column_var][:, :, :, t]) .* PPM
                _copy_regridded!(fields[(tracer, "column_mean")], t, tmp,
                                 regridder, column)

                q = Float64.(ds[tracer][:, :, :, :, t]) .* PPM
                _copy_regridded!(fields[(tracer, "surface")], t, tmp,
                                 regridder, @view(q[:, :, :, nz]))

                for target in eachindex(PRESSURE_TARGETS)
                    diagnostic = PRESSURE_TARGETS[target][1]
                    _extract_pressure_field!(pressure_field, q, pressure_indices, target)
                    _copy_regridded!(fields[(tracer, diagnostic)], t, tmp,
                                     regridder, pressure_field)
                end
            end
        end

        return SimulationFields(source, run, path, times, ll_area, fields)
    finally
        close(ds)
    end
end

function _dt_hours(t0, t1)
    delta = t1 - t0
    if delta isa Dates.Millisecond
        return Dates.value(delta) / 3.6e6
    elseif delta isa Dates.Second
        return Dates.value(delta) / 3600.0
    elseif delta isa Dates.Minute
        return Dates.value(delta) / 60.0
    elseif delta isa Dates.Hour
        return Float64(Dates.value(delta))
    elseif delta isa Dates.Day
        return 24.0 * Dates.value(delta)
    elseif delta isa Number
        return Float64(delta)
    else
        error("Unsupported time delta type: $(typeof(delta))")
    end
end

function _weighted_mean(x, w)
    return sum(x .* w) / sum(w)
end

function _weighted_std(x, w, mean)
    return sqrt(sum(w .* (x .- mean).^2) / sum(w))
end

function _weighted_corr(x, y, w)
    mx = _weighted_mean(x, w)
    my = _weighted_mean(y, w)
    sx = _weighted_std(x, w, mx)
    sy = _weighted_std(y, w, my)
    (sx == 0 || sy == 0) && return NaN
    return sum(w .* (x .- mx) .* (y .- my)) / sum(w) / (sx * sy)
end

function _metrics(a, b, w)
    mask = isfinite.(a) .& isfinite.(b) .& isfinite.(w) .& (w .> 0)
    n_valid = count(mask)
    n_valid == 0 && return (;
        a_mean=NaN, b_mean=NaN, bias=NaN, a_std=NaN, b_std=NaN,
        mae=NaN, rmse=NaN, maxabs=NaN, corr=NaN, n_valid=0)
    aa = a[mask]
    bb = b[mask]
    ww = w[mask]
    diff = bb .- aa
    a_mean = _weighted_mean(aa, ww)
    b_mean = _weighted_mean(bb, ww)
    mae = _weighted_mean(abs.(diff), ww)
    rmse = sqrt(_weighted_mean(diff.^2, ww))
    return (;
        a_mean,
        b_mean,
        bias=b_mean - a_mean,
        a_std=_weighted_std(aa, ww, a_mean),
        b_std=_weighted_std(bb, ww, b_mean),
        mae,
        rmse,
        maxabs=maximum(abs.(diff)),
        corr=_weighted_corr(aa, bb, ww),
        n_valid)
end

function _write_window_metrics(path, rows)
    open(path, "w") do io
        println(io, "source_a,run_a,source_b,run_b,tracer,diagnostic,lag_steps,dt_hours,window_index,start_time,end_time,a_mean_ppm_hr,b_mean_ppm_hr,bias_b_minus_a_ppm_hr,a_std_ppm_hr,b_std_ppm_hr,mae_ppm_hr,rmse_ppm_hr,maxabs_ppm_hr,corr,n_valid")
        for r in rows
            @printf(io, "%s,%s,%s,%s,%s,%s,%d,%.9g,%d,%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n",
                r.source_a, r.run_a, r.source_b, r.run_b, r.tracer,
                r.diagnostic, r.lag_steps, r.dt_hours, r.window_index,
                string(r.start_time), string(r.end_time),
                r.a_mean, r.b_mean, r.bias, r.a_std, r.b_std, r.mae,
                r.rmse, r.maxabs, r.corr, r.n_valid)
        end
    end
end

function _finite(values, field)
    out = Float64[]
    for row in values
        value = getfield(row, field)
        isfinite(value) && push!(out, value)
    end
    return out
end

function _mean_or_nan(x)
    isempty(x) && return NaN
    return mean(x)
end

function _median_or_nan(x)
    isempty(x) && return NaN
    return median(x)
end

function _min_or_nan(x)
    isempty(x) && return NaN
    return minimum(x)
end

function _max_or_nan(x)
    isempty(x) && return NaN
    return maximum(x)
end

function _write_summary(path, rows; matched_only=false)
    selected = if matched_only
        [r for r in rows if r.source_a != r.source_b && r.run_a == r.run_b]
    else
        rows
    end

    keys = unique((r.source_a, r.run_a, r.source_b, r.run_b, r.tracer,
                   r.diagnostic, r.lag_steps, r.dt_hours) for r in selected)
    sort!(keys; by=k -> (k[1], k[2], k[3], k[4], k[5], k[6], k[7]))

    open(path, "w") do io
        println(io, "source_a,run_a,source_b,run_b,tracer,diagnostic,lag_steps,dt_hours,window_count,mean_corr,median_corr,min_corr,mean_abs_bias_ppm_hr,mean_mae_ppm_hr,mean_rmse_ppm_hr,max_rmse_ppm_hr,mean_a_std_ppm_hr,mean_b_std_ppm_hr")
        for key in keys
            group = [r for r in selected if (r.source_a, r.run_a, r.source_b,
                     r.run_b, r.tracer, r.diagnostic, r.lag_steps,
                     r.dt_hours) == key]
            corrs = _finite(group, :corr)
            biases = [abs(r.bias) for r in group if isfinite(r.bias)]
            maes = _finite(group, :mae)
            rmses = _finite(group, :rmse)
            a_stds = _finite(group, :a_std)
            b_stds = _finite(group, :b_std)
            @printf(io, "%s,%s,%s,%s,%s,%s,%d,%.9g,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                key[1], key[2], key[3], key[4], key[5], key[6], key[7],
                key[8], length(group), _mean_or_nan(corrs),
                _median_or_nan(corrs), _min_or_nan(corrs),
                _mean_or_nan(biases), _mean_or_nan(maes),
                _mean_or_nan(rmses), _max_or_nan(rmses),
                _mean_or_nan(a_stds), _mean_or_nan(b_stds))
        end
    end
end

function _write_metadata(path, sims)
    open(path, "w") do io
        println(io, "source,run,path,Nc,Nz,panel_convention,cs_definition,time_count,start_time,end_time")
        for sim in sims
            NCDataset(sim.path) do ds
                @printf(io, "%s,%s,%s,%d,%d,%s,%s,%d,%s,%s\n",
                    sim.source, sim.run, sim.path,
                    Int(ds.attrib["Nc"]), size(ds["air_mass"], 4),
                    String(get(ds.attrib, "panel_convention", "")),
                    String(get(ds.attrib, "cs_definition", "")),
                    length(sim.times), string(sim.times[1]), string(sim.times[end]))
            end
        end
    end
end

function _compute_pairwise_rows(sims)
    rows = NamedTuple[]
    nsim = length(sims)
    for ia in 1:(nsim - 1), ib in (ia + 1):nsim
        a = sims[ia]
        b = sims[ib]
        length(a.times) == length(b.times) ||
            error("time length mismatch: $(a.source) $(a.run) vs $(b.source) $(b.run)")
        for i in eachindex(a.times)
            a.times[i] == b.times[i] ||
                error("time mismatch at index $i: $(a.source) $(a.run) vs $(b.source) $(b.run)")
        end
        println("Computing tendencies: $(a.source) $(a.run) vs $(b.source) $(b.run)")
        for tracer in TRACERS, diagnostic in DIAGNOSTICS, lag in LAG_STEPS
            lag < length(a.times) ||
                error("lag_steps=$lag is too large for $(length(a.times)) snapshots")
            afield = a.fields[(tracer, diagnostic)]
            bfield = b.fields[(tracer, diagnostic)]
            for t0 in 1:(length(a.times) - lag)
                t1 = t0 + lag
                dt = _dt_hours(a.times[t0], a.times[t1])
                atend = @views (afield[:, t1] .- afield[:, t0]) ./ dt
                btend = @views (bfield[:, t1] .- bfield[:, t0]) ./ dt
                m = _metrics(atend, btend, a.area)
                push!(rows, (;
                    source_a=a.source, run_a=a.run,
                    source_b=b.source, run_b=b.run,
                    tracer, diagnostic, lag_steps=lag, dt_hours=dt,
                    window_index=t0, start_time=a.times[t0], end_time=a.times[t1],
                    a_mean=m.a_mean, b_mean=m.b_mean, bias=m.bias,
                    a_std=m.a_std, b_std=m.b_std, mae=m.mae, rmse=m.rmse,
                    maxabs=m.maxabs, corr=m.corr, n_valid=m.n_valid))
            end
        end
    end
    return rows
end

function main()
    mkpath(OUT_DIR)
    sims = SimulationFields[]
    for (source, root) in (("ERA5", ERA_DIR), ("GEOS", GEOS_DIR)), run in RUNS
        push!(sims, _load_simulation_fields(source, run, _snapshot_path(root, run)))
    end

    _write_metadata(joinpath(OUT_DIR, "metadata.csv"), sims)
    rows = _compute_pairwise_rows(sims)
    _write_window_metrics(joinpath(OUT_DIR, "all_sim_window_tendency_metrics.csv"), rows)
    _write_summary(joinpath(OUT_DIR, "all_sim_window_tendency_summary.csv"), rows)
    _write_summary(joinpath(OUT_DIR, "matched_era_geos_window_tendency_summary.csv"),
                   rows; matched_only=true)

    open(joinpath(OUT_DIR, "README.txt"), "w") do io
        println(io, "C180 all-simulation window-tendency diagnostics")
        println(io, "Generated: ", Dates.now())
        println(io, "ERA_DIR: ", ERA_DIR)
        println(io, "GEOS_DIR: ", GEOS_DIR)
        println(io, "Common grid: ", LL_NX, " x ", LL_NY)
        println(io, "Lag steps: ", join(LAG_STEPS, ", "),
                " snapshot intervals; snapshot spacing is 6 h for these runs")
        println(io, "Tendencies are field(t + lag) - field(t), divided by dt_hours.")
        println(io, "Column means use saved column variables. Surface uses the bottom model level.")
        println(io, "Pressure diagnostics use the model level whose midpoint pressure is nearest the target at each source-grid column/time.")
        println(io, "Outputs:")
        println(io, "  metadata.csv")
        println(io, "  all_sim_window_tendency_metrics.csv")
        println(io, "  all_sim_window_tendency_summary.csv")
        println(io, "  matched_era_geos_window_tendency_summary.csv")
        println(io, "Optional plots from scripts/completed_experiments/plot_c180_window_tendency_summary.py:")
        println(io, "  matched_era_geos_mean_corr_6h.png")
        println(io, "  matched_era_geos_mean_corr_24h.png")
        println(io, "  matched_era_geos_mean_rmse_6h.png")
        println(io, "  matched_era_geos_mean_rmse_24h.png")
        println(io, "  within_path_mean_corr_6h.png")
        println(io, "  within_path_mean_corr_24h.png")
        println(io, "  within_path_mean_rmse_6h.png")
        println(io, "  within_path_mean_rmse_24h.png")
    end
    println("Wrote ", OUT_DIR)
end

main()
