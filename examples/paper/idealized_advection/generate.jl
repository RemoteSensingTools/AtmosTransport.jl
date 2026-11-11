#!/usr/bin/env julia
#
# Analytic winds -> transport binaries, initial-condition file and analytic reference for one
# grid and one flow of the idealized advection tests (GMD paper, Sect. 4.1).
#
#   julia --project=. -t 16 examples/paper/idealized_advection/generate.jl \
#         --grid c48 --flow B --workdir <work directory>
#
# Writes <workdir>/binaries/<grid>_<flow>/*.bin (one per day), <workdir>/inputs/<grid>_<flow>/
# {ic.nc, reference.nc, case.toml}, and prints the probe numbers (transfer magnitudes, an
# independent check of the transfers, the replay residual, the outflow Courant number and the
# substep count, trajectory and quadrature convergence, and for flow C the density checks). The
# work directory defaults to $IDEALIZED_ADVECTION_WORKDIR; `--tag` names a variant (e.g. a
# different `--courant`) of the same grid and flow.

using ArgParse
using Dates
using Printf
using TOML
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "IdealizedAdvection.jl"))
using .IdealizedAdvection
const IA = IdealizedAdvection
using AtmosTransport.Parameters: EARTH_RADIUS, SECONDS_PER_DAY, GLOBAL_MEAN_DRY_SURFACE_PRESSURE

function parse_arguments()
    s = ArgParseSettings(description = "Write transport binaries and references for an idealized advection case.")
    @add_arg_table! s begin
        "--grid"
            help = "c<N> (cubed sphere), ll<Nx>x<Ny> (latitude-longitude) or n<N> (reduced Gaussian)"
            required = true
        "--flow"
            help = "A (solid-body rotation), B (non-divergent deformation), C (divergent deformation)"
            required = true
        "--workdir"
            help = "work directory for binaries and inputs (default: \$IDEALIZED_ADVECTION_WORKDIR)"
            default = get(ENV, "IDEALIZED_ADVECTION_WORKDIR", "")
        "--courant"
            help = "largest outflow fraction of a cell's air mass per flux application"
            arg_type = Float64
            default = 0.5
        "--days"
            arg_type = Int
            default = 12
        "--windows-per-day"
            arg_type = Int
            default = 24
        "--start-date"
            default = "2001-01-01"
        "--quadrature"
            help = "sub-cell quadrature points per direction for the analytic cell means"
            arg_type = Int
            default = 6
        "--quadrature-fine"
            help = "points per direction for cells cut by a discontinuity (slotted cylinders)"
            arg_type = Int
            default = 48
        "--tilt"
            help = "flow A: tilt of the rotation axis from the pole [degrees]"
            arg_type = Float64
            default = 45.0
        "--tag"
            help = "suffix of the case name for a variant of the same grid and flow (e.g. cfl025)"
            default = ""
        "--sigma-levels"
            help = "pure-sigma interface coefficients B (top to surface); " *
                   "Lin-Rood (FV3 vertical) needs at least 4 layers"
            default = "0.0,0.15,0.4,0.7,1.0"
        "--surface-pressure"
            help = "uniform dry surface pressure [Pa] (default: the model's global-mean dry value)"
            arg_type = Float64
            default = GLOBAL_MEAN_DRY_SURFACE_PRESSURE
        "--pl-dir"
            help = "directory with the reduced-Gaussian ring-size tables n<N>_pl.txt"
            default = joinpath(@__DIR__, "..", "grids")
    end
    return parse_args(s)
end

"""
Independent check of the stored transfers for a stream-function flow: the face transfer against a
direct line integral of V·n along the face (Gauss-Legendre 8 × 8 in space and time), relative to
the largest transfer. For the divergent flow: the stored transfers of window 1 against 6 × 8
nodes and a 4× smaller pseudo-time step.
"""
function probe_transfers(g, flow::IA.StreamFunctionFlow, F, windows, ps)
    t0, t1 = windows[1]
    direct = IA.window_volume_transfer!(zeros(IA.nface(g)), flow, g, t0, t1; ntime = 8, nspace = 8)
    return maximum(abs, IA.column_mass_per_area(ps) .* direct .- F[:, 1]) / maximum(abs, F[:, 1])
end
function probe_transfers(g, flow, F, windows, ps)
    t0, t1 = windows[1]
    fine = IA.window_mass_transfer!(zeros(IA.nface(g)), flow, g, t0, t1; σ0 = IA.column_mass_per_area(ps),
                                    ntime = 6, nspace = 8, h = IA.FLUX_PSEUDO_STEP / 4)
    return maximum(abs, fine .- F[:, 1]) / maximum(abs, fine)
end

"""Analytic `div W̃` against a centred finite difference of `W̃` (max abs difference)."""
function probe_divergence(flow::IA.DivergentDeformation, sample)
    worst = 0.0
    d = 1e-5
    for x in sample
        abs(x[3]) > 0.99 && continue
        λ, θ = IA.lon_of(x), IA.lat_of(x)
        u(λ, θ) = dot(IA.deformation_wind(flow, IA.unit_xyz(λ, θ)), IA.east_unit(λ))
        v(λ, θ) = dot(IA.deformation_wind(flow, IA.unit_xyz(λ, θ)), IA.north_unit(λ, θ))
        fd = ((u(λ + d, θ) - u(λ - d, θ)) + (v(λ, θ + d) * cos(θ + d) - v(λ, θ - d) * cos(θ - d))) / (2d * cos(θ))
        worst = max(worst, abs(fd - IA.deformation_divergence(flow, x)))
    end
    return worst
end

"""Trajectory probes on a point sample: step convergence and the physical-time RK4 at T/2, return at T."""
function probe_trajectories(flow::IA.DeformationalFlow, sample)
    T = flow.period
    h = IA.DEFAULT_PSEUDO_STEP
    base = IA.trace(flow, sample, T / 2; h)
    half = IA.trace(flow, sample, T / 2; h = h / 2)
    rk4 = [IA.departure_point_rk4(flow, x, T / 2; steps = 2880) for x in sample]
    back = IA.trace(flow, sample, T; h)
    @printf("trajectories (%d points): pseudo-time h vs h/2 at T/2: |dx| %.2e, |dlog rho| %.2e; ",
            length(sample), maximum(norm.(base[1] .- half[1])), maximum(abs.(base[2] .- half[2])))
    @printf("vs 2880-step physical-time RK4: |dx| %.2e; at T: |x(0) - x| %.2e, |log rho(T)| %.2e\n",
            maximum(norm.(base[1] .- rk4)), maximum(norm.(back[1] .- sample)), maximum(abs.(back[2])))
end
probe_trajectories(flow, sample) = println("trajectories: exact rotation")

"""Total eastward transfer across the meridian at the lat-lon date line, against the analytic value."""
function probe_meridian(g::IA.GridGeometry{IA.LatLonGrid}, flow::IA.SolidBodyRotation, F, windows, ps)
    Nx, Ny = g.grid.mesh.Nx, g.grid.mesh.Ny
    model = sum(F[1 + (j - 1) * Nx, 1] for j in 1:Ny)
    t0, t1 = windows[1]
    analytic = IA.column_mass_per_area(ps) * IA.angular_speed(flow) * EARTH_RADIUS^2 *
               2 * cosd(flow.tilt_deg) * (t1 - t0)
    return model, analytic
end
probe_meridian(g, flow, F, windows, ps) = nothing

function main()
    args = parse_arguments()
    workdir = args["workdir"]
    isempty(workdir) && error("set --workdir or IDEALIZED_ADVECTION_WORKDIR")
    grid_label, flow_id = args["grid"], args["flow"]
    case = isempty(args["tag"]) ? "$(grid_label)_$(flow_id)" : "$(grid_label)_$(flow_id)_$(args["tag"])"
    B_ifc = parse.(Float64, split(args["sigma-levels"], ","))
    ps = args["surface-pressure"]
    dt = SECONDS_PER_DAY / args["windows-per-day"]
    nw = args["days"] * args["windows-per-day"]
    windows = [((w - 1) * dt, w * dt) for w in 1:nw]

    # `case.toml` marks a complete case (run_matrix.sh): drop it before anything is overwritten,
    # so an interrupted regeneration cannot leave old outputs looking current.
    rm(joinpath(workdir, "inputs", case, "case.toml"); force = true)

    flow, tracers = IA.flow_case(flow_id; tilt_deg = args["tilt"])
    T = flow.period
    abs(nw * dt - T) < 1e-6 || @warn "run length $(nw * dt / 3600) h differs from the flow period $(T / 3600) h"

    t_start = time()
    g = IA.build_geometry(grid_label; pl_dir = args["pl-dir"])
    @printf("grid %s: %d cells, %d faces, %d unique corners, area/4piR2 = %.15f\n", grid_label,
            IA.ncell(g), IA.nface(g), length(g.corners), sum(g.cell_area) / (4π * EARTH_RADIUS^2))
    println("flow $(flow_id): ", IA.flow_description(flow))

    F, M = IA.column_transfers(flow, g, windows; ps)
    outflow = IA.max_outflow_fraction(g, F, M)
    steps = max(1, ceil(Int, outflow / (2 * args["courant"])))
    @printf("transfers: max |F| %.4e kg, mean |F| %.4e kg per window (column)\n",
            maximum(abs, F), sum(abs, F) / length(F))
    @printf("largest outflow per window %.3f of the cell mass -> %d substeps per window, %.3f per flux application\n",
            outflow, steps, outflow / (2 * steps))
    @printf("air mass: min/max column %.6e / %.6e kg; global change over the run %.3e (relative)\n",
            minimum(M), maximum(M), (sum(M[:, end]) - sum(M[:, 1])) / sum(M[:, 1]))
    @printf("line-integral check of the transfers (see probe_transfers): %.3e\n",
            probe_transfers(g, flow, F, windows, ps))
    meridian = probe_meridian(g, flow, F, windows, ps)
    meridian === nothing ||
        @printf("eastward transfer across the date line: model %.10e kg, analytic %.10e kg, rel diff %.2e\n",
                meridian..., (meridian[1] - meridian[2]) / meridian[2])
    replay = IA.replay_residual(g, F, M, B_ifc, steps)
    @printf("replay residual (per-cell |r|/scale, packed Float64 arrays): %.3e (window %d)\n",
            replay.max_rel_err, replay.window)

    bindir = joinpath(workdir, "binaries", case)
    paths = IA.write_binaries(g, F, M; outdir = bindir, prefix = "idealized_$(case)",
        start_date = Date(args["start-date"]), windows_per_day = args["windows-per-day"], dt, steps, B_ifc,
        extra = Dict{String, Any}("idealized_flow" => IA.flow_description(flow), "idealized_case" => case,
                                  "mass_fix_enabled" => false,
                                  "idealized_outflow_fraction_per_application" => outflow / (2 * steps)))
    @printf("wrote %d binaries to %s (%.1f s)\n", length(paths), bindir, time() - t_start)

    # Analytic references: air-mass-weighted cell means at 0 and T/2; at T the field and the
    # density are back at their initial state, so T uses the initial means.
    k, kf = args["quadrature"], args["quadrature-fine"]
    initial, area0, refined = IA.analytic_cell_means(flow, tracers, g, 0.0; k, k_fine = kf)
    middle, air_half, refined_half = IA.analytic_cell_means(flow, tracers, g, T / 2; k, k_fine = kf)
    fields = [name => hcat(initial[name], middle[name], initial[name]) for (name, _) in tracers]
    for (name, field) in tracers
        IA.is_discontinuous(field) || continue
        half_fine = IA.analytic_cell_means(flow, [name => field], g, 0.0; k, k_fine = kf ÷ 2)[1][name]
        coarse = IA.analytic_cell_means(flow, [name => field], g, 0.0; k, k_fine = k)[1][name]
        @printf("%s: %d / %d cells refined at t = 0 (%d at T/2); initial cell means %dx%d vs %dx%d max |d| %.3e, ",
                name, refined[name], IA.ncell(g), refined_half[name], k, k, kf, kf,
                maximum(abs, coarse .- initial[name]))
        @printf("%dx%d vs %dx%d max |d| %.3e\n", kf ÷ 2, kf ÷ 2, kf, kf, maximum(abs, half_fine .- initial[name]))
    end
    rng = Random.Xoshiro(1)
    sample = [IA.unit_xyz(2π * rand(rng), asin(2rand(rng) - 1)) for _ in 1:2000]
    probe_trajectories(flow, sample)
    if flow isa IA.DivergentDeformation
        σ0 = IA.column_mass_per_area(ps)
        ρcell = air_half ./ g.cell_area
        @printf("div W: analytic vs finite difference max |d| %.2e\n", probe_divergence(flow, sample))
        @printf("density at T/2: cell-mean rho/rho0 in [%.4f, %.4f]; global integral / 4 pi R^2 - 1 = %.2e\n",
                minimum(ρcell), maximum(ρcell), sum(air_half) / (4π * EARTH_RADIUS^2) - 1)
        mid = nw ÷ 2 + 1
        air_fine = IA.analytic_cell_means(flow, Pair{String, IA.UniformField}[], g, T / 2; k = 2k)[2]
        @printf("closure air mass vs cell integral of rho (%dx%d / %dx%d points): T/2 max rel %.2e / %.2e; ",
                k, k, 2k, 2k, maximum(abs.(M[:, mid] ./ (σ0 .* air_half) .- 1)),
                maximum(abs.(M[:, mid] ./ (σ0 .* air_fine) .- 1)))
        @printf("T max rel %.2e\n", maximum(abs.(M[:, end] ./ (σ0 .* g.cell_area) .- 1)))
    end

    indir = mkpath(joinpath(workdir, "inputs", case))
    Nz = length(B_ifc) - 1
    ic = IA.write_initial_conditions(g, joinpath(indir, "ic.nc"), [name => a[:, 1] for (name, a) in fields], Nz)
    attributes = Dict{String, Any}("flow" => IA.flow_description(flow), "start_date" => args["start-date"],
                                   "quadrature_points_per_cell" => k * k,
                                   "quadrature_points_per_cut_cell" => kf * kf,
                                   "cell_mean" => "air-mass weighted")
    ref = IA.write_reference(joinpath(indir, "reference.nc"), g, [0.0, T / 7200, T / 3600], fields; attributes)
    meta = Dict{String, Any}(
        "case" => case, "grid" => grid_label, "flow" => flow_id, "tag" => args["tag"],
        "courant" => args["courant"], "tilt_deg" => args["tilt"],
        "flow_description" => IA.flow_description(flow),
        "topology" => g.grid isa IA.CubedSphereGrid ? "cubed_sphere" :
                      g.grid isa IA.LatLonGrid ? "latlon" : "reduced_gaussian",
        "ncell" => IA.ncell(g), "nlevel" => Nz, "sigma_interfaces" => B_ifc,
        "start_date" => args["start-date"], "days" => args["days"], "period_hours" => T / 3600,
        "windows_per_day" => args["windows-per-day"], "steps_per_window" => steps,
        "outflow_fraction_per_application" => outflow / (2 * steps),
        "replay_residual" => replay.max_rel_err,
        "binaries" => paths, "initial_conditions" => ic, "reference" => ref,
        "tracers" => first.(tracers))
    open(joinpath(indir, "case.toml"), "w") do io
        TOML.print(io, meta)
    end
    @printf("wrote %s, %s (total %.1f s)\n", ic, ref, time() - t_start)
end

main()
