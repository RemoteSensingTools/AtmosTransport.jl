#!/usr/bin/env julia

using Printf

include(joinpath(@__DIR__, "..", "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Operators: AdvectionWorkspace, CSAdvectionWorkspace,
    CSLinRoodAdvectionWorkspace, strang_split!
using .AtmosTransport.Operators.Advection: fill_panel_halos!, strang_split_cs!
using .AtmosTransport.State: CubedSphereFaceFluxState, DryBasis,
    StructuredFaceFluxState, total_air_mass, total_mass

function relerr(a, b)
    return (a - b) / max(abs(b), eps(Float64))
end

function total_cs(panels, mesh, Nz)
    Nc, Hp = mesh.Nc, mesh.Hp
    s = 0.0
    for p in 1:6
        s += sum(@view panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, 1:Nz])
    end
    return s
end

function run_latlon_closed(; FT=Float64, Nx=48, Ny=24, Nz=4, scheme=UpwindScheme(), steps=8)
    mesh = LatLonMesh(; FT, Nx, Ny)
    vc = HybridSigmaPressure(FT[0, 100, 500, 2000, 0], FT[0, 0, 0.1, 0.5, 1])
    grid = AtmosGrid(mesh, vc, CPU(); FT)

    m = Array{FT}(undef, Nx, Ny, Nz)
    rm = similar(m)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m[i, j, k] = FT(1.0e9) * (1 + FT(0.02) * k + FT(0.001) * j)
        χ = FT(390e-6) +
            FT(35e-6) * sin(FT(2π) * FT(i - 1) / FT(Nx)) +
            FT(25e-6) * cos(FT(π) * FT(j - 1) / FT(Ny - 1)) +
            FT(5e-6) * FT(k - 1)
        rm[i, j, k] = m[i, j, k] * χ
    end

    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    for k in 1:Nz
        for j in 2:Ny-1, i in 2:Nx
            am[i, j, k] = FT(1.5e6) * sin(FT(2π) * FT(i - 1) / FT(Nx)) *
                          cos(FT(π) * FT(j - 1) / FT(Ny - 1))
        end
        for j in 2:Ny, i in 2:Nx-1
            bm[i, j, k] = FT(1.0e6) * cos(FT(2π) * FT(i - 1) / FT(Nx)) *
                          sin(FT(π) * FT(j - 1) / FT(Ny - 1))
        end
    end

    state = CellState(DryBasis, copy(m); tracer=copy(rm))
    fluxes = StructuredFaceFluxState{DryBasis}(copy(am), copy(bm), copy(cm))
    ws = AdvectionWorkspace(state)
    m0 = sum(state.air_mass)
    rm0 = sum(state.tracers.tracer)
    for _ in 1:steps
        strang_split!(state, fluxes, grid, scheme; workspace=ws)
    end
    return relerr(sum(state.air_mass), m0), relerr(sum(state.tracers.tracer), rm0)
end

function make_cs_state_fluxes(; FT=Float64, Nc=24, Nz=4, scheme=UpwindScheme(),
                              cross_panel::Bool, mirrored_seams::Bool=false)
    Hp = AtmosTransport.Operators.required_halo_width(scheme)
    mesh = CubedSphereMesh(; FT, Nc, Hp)
    N = Nc + 2Hp

    panels_m = ntuple(6) do p
        m = Array{FT}(undef, N, N, Nz)
        fill!(m, zero(FT))
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            m[Hp+i, Hp+j, k] = FT(1.0e9) * (1 + FT(0.01) * k + FT(0.002) * p)
        end
        m
    end
    panels_rm = ntuple(6) do p
        rm = Array{FT}(undef, N, N, Nz)
        fill!(rm, zero(FT))
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            χ = FT(390e-6) +
                FT(30e-6) * sin(FT(2π) * FT(i - 1) / FT(Nc)) +
                FT(20e-6) * cos(FT(2π) * FT(j - 1) / FT(Nc)) +
                FT(8e-6) * FT(p - 3) +
                FT(5e-6) * FT(k - 1)
            rm[Hp+i, Hp+j, k] = panels_m[p][Hp+i, Hp+j, k] * χ
        end
        rm
    end
    fill_panel_halos!(panels_m, mesh; dir=0)
    fill_panel_halos!(panels_rm, mesh; dir=0)

    margin = cross_panel ? 1 : Hp + 2
    raw_am = nothing
    raw_bm = nothing
    if cross_panel && mirrored_seams
        raw_am = ntuple(6) do p
            am = zeros(FT, Nc + 1, Nc, Nz)
            for k in 1:Nz, s in 1:Nc
                am[1,      s, k] = FT(1.0e6) * sin(FT(0.3p + 0.2s + 0.1k))
                am[Nc + 1, s, k] = FT(1.0e6) * cos(FT(0.2p - 0.4s + 0.1k))
            end
            am
        end
        raw_bm = ntuple(6) do p
            bm = zeros(FT, Nc, Nc + 1, Nz)
            for k in 1:Nz, s in 1:Nc
                bm[s, 1,      k] = FT(0.8e6) * cos(FT(0.1p + 0.3s - 0.2k))
                bm[s, Nc + 1, k] = FT(0.8e6) * sin(FT(0.4p - 0.1s + 0.2k))
            end
            bm
        end
        AtmosTransport.Preprocessing.sync_all_cs_boundary_mirrors!(
            raw_am, raw_bm, mesh.connectivity, Nc, Nz)
    end

    panels_am = ntuple(6) do p
        am = zeros(FT, N + 1, N, Nz)
        if cross_panel
            if mirrored_seams
                for k in 1:Nz, j in 1:Nc, i in 1:(Nc + 1)
                    am[Hp + i, Hp + j, k] = raw_am[p][i, j, k]
                end
            else
                for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
                    am[i, j, k] = FT(2.0e6)
                end
            end
        else
            for k in 1:Nz, j in (Hp+margin):(Hp+Nc-margin+1),
                    i in (Hp+margin):(Hp+Nc-margin+2)
                am[i, j, k] = FT(2.5e6) * sin(FT(0.3) * FT(p) + FT(0.4) * FT(i) -
                                              FT(0.2) * FT(j) + FT(0.1) * FT(k))
            end
        end
        am
    end
    panels_bm = ntuple(6) do p
        bm = zeros(FT, N, N + 1, Nz)
        if cross_panel && mirrored_seams
            for k in 1:Nz, j in 1:(Nc + 1), i in 1:Nc
                bm[Hp + i, Hp + j, k] = raw_bm[p][i, j, k]
            end
        elseif !cross_panel
            for k in 1:Nz, j in (Hp+margin):(Hp+Nc-margin+2),
                    i in (Hp+margin):(Hp+Nc-margin+1)
                bm[i, j, k] = FT(2.0e6) * cos(FT(0.2) * FT(p) - FT(0.3) * FT(i) +
                                              FT(0.5) * FT(j) + FT(0.1) * FT(k))
            end
        end
        bm
    end
    panels_cm = ntuple(_ -> zeros(FT, N, N, Nz + 1), 6)
    return mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm
end

function run_cs_closed(; FT=Float64, scheme=UpwindScheme(), cross_panel::Bool,
                       mirrored_seams::Bool=false, steps=8)
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        make_cs_state_fluxes(; FT, scheme, cross_panel, mirrored_seams)
    Nz = size(panels_m[1], 3)
    if scheme isa LinRoodPPMScheme
        vertical = HybridSigmaPressure(FT[0, 100, 500, 2000, 0],
                                       FT[0, 0, 0.1, 0.5, 1])
        grid = AtmosGrid(mesh, vertical, CPU(); FT)
        state = CubedSphereState(DryBasis, mesh, panels_m; tracer=panels_rm)
        fluxes = CubedSphereFaceFluxState{DryBasis}(panels_am, panels_bm, panels_cm)
        ws = CSLinRoodAdvectionWorkspace(mesh, state.air_mass[1])
        m0 = total_air_mass(state)
        rm0 = total_mass(state, :tracer)
        for _ in 1:steps
            strang_split!(state, fluxes, grid, scheme; workspace=ws)
        end
        return relerr(total_air_mass(state), m0), relerr(total_mass(state, :tracer), rm0)
    end

    ws = CSAdvectionWorkspace(mesh, Nz)
    m0 = total_cs(panels_m, mesh, Nz)
    rm0 = total_cs(panels_rm, mesh, Nz)
    for _ in 1:steps
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, scheme, ws)
    end
    return relerr(total_cs(panels_m, mesh, Nz), m0),
           relerr(total_cs(panels_rm, mesh, Nz), rm0)
end

function main()
    cases = (
        ("LL closed", false, run_latlon_closed),
        ("CS interior-only", false, (; kwargs...) -> run_cs_closed(; kwargs..., cross_panel=false)),
        # Negative control: uniform same-sign face values placed across panel
        # boundaries are not a valid cubed-sphere seam-flux field.
        ("CS cross-panel", true, (; kwargs...) -> run_cs_closed(; kwargs..., cross_panel=true)),
        ("CS mirrored-seam", true, (; kwargs...) -> run_cs_closed(; kwargs..., cross_panel=true, mirrored_seams=true)),
    )
    schemes = (
        ("upwind", UpwindScheme()),
        ("ppm", PPMScheme()),
        ("linrood5", LinRoodPPMScheme(5)),
        ("linrood7", LinRoodPPMScheme(7)),
    )

    @printf("%-17s %-8s %-5s %5s %14s %14s\n", "case", "scheme", "FT", "steps", "air_rel", "tracer_rel")
    for (label, _is_cs, runner) in cases
        for (scheme_name, scheme) in schemes
            label == "LL closed" && scheme isa LinRoodPPMScheme && continue
            for FT in (Float64, Float32)
                for steps in (1, 8)
                    air_rel, tracer_rel = runner(; FT, scheme, steps)
                    @printf("%-17s %-8s %-5s %5d %+14.6e %+14.6e\n",
                        label, scheme_name, FT === Float64 ? "F64" : "F32",
                        steps, air_rel, tracer_rel)
                end
            end
        end
    end
end

main()
