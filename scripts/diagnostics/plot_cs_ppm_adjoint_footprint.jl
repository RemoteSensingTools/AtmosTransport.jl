#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plot a small CS PPM surface-emission footprint prototype.
#
# This uses AtmosTransport.Adjoints.cs_surface_emission_footprint, which runs
# a kernelized CS PPM reverse pass for diagnostics/tests. The plot shows how
# final receptor objectives respond to surface emissions applied at earlier
# midpoint steps, including the midpoint vertical-diffusion slot.
#
# Usage:
#   julia --project=docs scripts/diagnostics/plot_cs_ppm_adjoint_footprint.jl \
#       --out artifacts/cs_ppm_adjoint_footprint.png
# ---------------------------------------------------------------------------

using CairoMakie

try
    @eval using AtmosTransport
    @eval using AtmosTransport.Operators.Advection: fill_panel_halos!
catch
    include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
    @eval using .AtmosTransport
    @eval using .AtmosTransport.Operators.Advection: fill_panel_halos!
end

const USAGE = """
Usage: julia --project=docs scripts/diagnostics/plot_cs_ppm_adjoint_footprint.jl \\
           [--out <png>] [--nc N] [--steps N]
"""

function _parse_args(argv)
    out = joinpath("artifacts", "cs_ppm_adjoint_footprint.png")
    nc = 6
    steps = 3
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--out" && i + 1 <= length(argv)
            out = argv[i + 1]
            i += 2
        elseif arg == "--nc" && i + 1 <= length(argv)
            nc = parse(Int, argv[i + 1])
            i += 2
        elseif arg == "--steps" && i + 1 <= length(argv)
            steps = parse(Int, argv[i + 1])
            i += 2
        elseif arg in ("-h", "--help")
            println(USAGE)
            exit(0)
        else
            error("Unknown argument `$arg`.\n$USAGE")
        end
    end
    nc >= 4 || error("--nc must be at least 4 for the PPM footprint demo")
    steps >= 1 || error("--steps must be positive")
    return (; out, nc, steps)
end

function _demo_problem(; Nc::Int, Nz::Int=4, nsteps::Int=3, FT=Float64)
    mesh = CubedSphereMesh(Nc=Nc, Hp=3, FT=FT)
    N = mesh.Nc + 2mesh.Hp
    Hp = mesh.Hp

    panels_m = ntuple(6) do p
        m = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in 1:N, i in 1:N
            # Larger lower-atmosphere mass, with a tiny panel perturbation so
            # column and layer objectives are visibly different.
            m[i, j, k] = FT(1.0 + 0.35k + 0.02p)
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
        panels_am_steps[step] = ntuple(6) do p
            am = zeros(FT, N + 1, N, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + Nc, i in Hp + 1:Hp + Nc + 1
                am[i, j, k] = FT(0.018) * sin(FT(0.4step + 0.2p + 0.7i + 0.3j + 0.1k))
            end
            am
        end
        panels_bm_steps[step] = ntuple(6) do p
            bm = zeros(FT, N, N + 1, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + Nc + 1, i in Hp + 1:Hp + Nc
                bm[i, j, k] = FT(0.014) * cos(FT(0.2step + 0.4p + 0.5i + 0.6j + 0.2k))
            end
            bm
        end
        panels_cm_steps[step] = ntuple(6) do p
            cm = zeros(FT, N, N, Nz + 1)
            for k in 2:Nz, j in Hp + 1:Hp + Nc, i in Hp + 1:Hp + Nc
                # Negative is upward in this convention, moving surface
                # emissions into the overlying layers over successive steps.
                cm[i, j, k] = -FT(0.012) * (one(FT) + FT(0.15) * sin(FT(i + j + k + p + step)))
            end
            cm
        end
    end
    return mesh, panels_m, panels_rm, panels_am_steps, panels_bm_steps, panels_cm_steps
end

function _row_range(result, panel)
    mx = 0.0
    for fp in result.footprints
        mx = max(mx, maximum(abs, fp[panel]))
    end
    return mx == 0.0 ? (-1.0, 1.0) : (-mx, mx)
end

function _demo_diffusion(mesh, prototype; kz=5.0, dz=50.0)
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
        e[:, :, 2:min(3, Nz - 1)] .= FT(0.003)
        e
    end, 6)
    detu = ntuple(_ -> begin
        e = zeros(FT, Nc, Nc, Nz)
        e[:, :, 2:min(3, Nz - 1)] .= FT(0.002)
        e
    end, 6)
    entd = ntuple(_ -> begin
        e = zeros(FT, Nc, Nc, Nz)
        e[:, :, 3:min(4, Nz - 1)] .= FT(0.001)
        e
    end, 6)
    detd = ntuple(_ -> begin
        e = zeros(FT, Nc, Nc, Nz)
        e[:, :, 3:min(4, Nz - 1)] .= FT(0.0005)
        e
    end, 6)
    forcing = ConvectionForcing(nothing, nothing, (; entu, detu, entd, detd))
    metrics = ntuple(_ -> ones(FT, Nc, Nc), 6)
    ws = TM5Workspace(panels_m; tile_columns=Nc * Nc, cell_metrics=metrics)
    return TM5Convection(), forcing, ws
end

function _plot(out::AbstractString; Nc::Int, nsteps::Int)
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _demo_problem(; Nc=Nc, nsteps=nsteps)
    diffusion_op, diffusion_ws = _demo_diffusion(mesh, panels_rm[1])
    convection_op, convection_forcing, convection_ws =
        _demo_tm5_convection(mesh, panels_m)

    receptor_panel = 1
    receptor_i = cld(Nc, 2)
    receptor_j = cld(Nc, 2)
    dt = 120.0
    scheme = PPMScheme(NoLimiter())
    objectives = [
        ("surface layer", CSLayerMeanObjective(receptor_panel, receptor_i, receptor_j, 4)),
        ("free-trop layer", CSLayerMeanObjective(receptor_panel, receptor_i, receptor_j, 2)),
        ("full column", CSColumnMeanObjective(receptor_panel, receptor_i, receptor_j)),
    ]

    results = map(objectives) do (_, obj)
        cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            diffusion_op=diffusion_op,
            diffusion_workspace=diffusion_ws,
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws)
    end

    fig = Figure(size=(260nsteps + 130, 230length(results) + 80))
    for (row, ((label, _), result)) in enumerate(zip(objectives, results))
        crange = _row_range(result, receptor_panel)
        hm = nothing
        for step in 1:nsteps
            ax = Axis(fig[row, step];
                aspect=DataAspect(),
                title=row == 1 ? "lag $(result.lag_steps[step]) step" : "",
                xlabel= row == length(results) ? "CS i" : "",
                ylabel= step == 1 ? "CS j" : "")
            hidedecorations!(ax; label=false)
            hm = heatmap!(ax, 1:Nc, 1:Nc, result.footprints[step][receptor_panel];
                colormap=:RdBu, colorrange=crange)
            scatter!(ax, [receptor_i], [receptor_j];
                color=:black, markersize=9, marker=:xcross)
        end
        Label(fig[row, 0], label; rotation=pi / 2, tellheight=false)
        Colorbar(fig[row, nsteps + 1], hm; label="dJ / dE")
    end

    mkpath(dirname(out))
    save(out, fig)
    println(out)
    return out
end

function main(argv=ARGS)
    opts = _parse_args(argv)
    return _plot(opts.out; Nc=opts.nc, nsteps=opts.steps)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
