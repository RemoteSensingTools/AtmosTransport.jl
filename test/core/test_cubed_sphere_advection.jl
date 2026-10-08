#!/usr/bin/env julia

using Test
using Logging, Random

using AtmosTransport
using .AtmosTransport.Grids: reciprocal_edge
using .AtmosTransport.Operators: MonotoneLimiter, required_halo_width
using .AtmosTransport.Operators.Advection: fill_panel_halos!, strang_split_cs!,
    strang_split_cs_mt!, strang_split!, CSAdvectionWorkspace,
    _sweep_x_panel_mt!, _sweep_y_panel_mt!, _sweep_z_panel_mt!,
    _sweep_x_panels_mt_pingpong!, _sweep_y_panels_mt_pingpong!,
    _sweep_z_panels_mt_pingpong!, _strang_split_cs_mt_copyback!,
    strang_split_cs_mt_pingpong!
using .AtmosTransport.State: CubedSphereFaceFluxState, DryBasis,
    total_air_mass, total_mass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function total_interior(panels, Nc, Hp, Nz)
    s = 0.0
    for p in 1:6, k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
        s += panels[p][i, j, k]
    end
    return s
end

function max_vmr_deviation(panels_rm, panels_m, Nc, Hp, Nz, target)
    dev = 0.0
    for p in 1:6, k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
        vmr = panels_rm[p][i, j, k] / panels_m[p][i, j, k]
        dev = max(dev, abs(vmr - target))
    end
    return dev
end

function max_interior_absdiff(a, b, Nc, Hp, Nz)
    dev = 0.0
    for p in 1:6, k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
        dev = max(dev, abs(a[p][i, j, k] - b[p][i, j, k]))
    end
    return dev
end

function max_interior_absdiff_4d(a, b, Nc, Hp, Nz, Nt)
    dev = 0.0
    for t in 1:Nt, p in 1:6, k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
        dev = max(dev, abs(a[p][i, j, k, t] - b[p][i, j, k, t]))
    end
    return dev
end

function make_cs_test_state(; Nc=12, Hp=1, Nz=4, FT=Float64, vmr=411.0)
    mesh = CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
    N = Nc + 2Hp
    panels_m  = ntuple(_ -> ones(FT, N, N, Nz), 6)
    panels_rm = ntuple(_ -> fill!(zeros(FT, N, N, Nz), FT(vmr)), 6)
    fill_panel_halos!(panels_m, mesh; dir=0)
    fill_panel_halos!(panels_rm, mesh; dir=0)
    return mesh, panels_m, panels_rm
end

function total_cs_surface_rate(rates)
    s = zero(eltype(rates[1]))
    for p in 1:6
        s += sum(rates[p])
    end
    return s
end

function make_structured_cs_state(; FT=Float64, Nc=8, Hp=1, Nz=2, convention=nothing)
    mesh = CubedSphereMesh(; FT, Nc, Hp, convention)
    N = Nc + 2Hp
    panels_m = ntuple(6) do p
        m = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            m[Hp+i, Hp+j, k] = FT(1.0e9) * (1 + FT(0.01) * k + FT(0.002) * p)
        end
        m
    end
    panels_rm = ntuple(6) do p
        rm = zeros(FT, N, N, Nz)
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
    return mesh, panels_m, panels_rm
end

function make_mirrored_cs_horizontal_fluxes(mesh::CubedSphereMesh{FT}, Nz::Int) where FT
    Prep = AtmosTransport.Preprocessing
    Nc, Hp = mesh.Nc, mesh.Hp
    N = Nc + 2Hp
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
    Prep.sync_all_cs_boundary_mirrors!(raw_am, raw_bm, mesh.connectivity, Nc, Nz)

    panels_am = ntuple(6) do p
        am = zeros(FT, N + 1, N, Nz)
        for k in 1:Nz, j in 1:Nc, i in 1:(Nc + 1)
            am[Hp + i, Hp + j, k] = raw_am[p][i, j, k]
        end
        am
    end
    panels_bm = ntuple(6) do p
        bm = zeros(FT, N, N + 1, Nz)
        for k in 1:Nz, j in 1:(Nc + 1), i in 1:Nc
            bm[Hp + i, Hp + j, k] = raw_bm[p][i, j, k]
        end
        bm
    end
    panels_cm = ntuple(_ -> zeros(FT, N, N, Nz + 1), 6)
    return panels_am, panels_bm, panels_cm
end

function run_mirrored_seam_advection_conservation(scheme; FT=Float64, Nc=8, Nz=2, steps=2,
                                                  flux_gain=1, convention=nothing)
    Hp = required_halo_width(scheme)
    mesh, panels_m, panels_rm = make_structured_cs_state(; FT, Nc, Hp, Nz, convention)
    panels_am, panels_bm, panels_cm = make_mirrored_cs_horizontal_fluxes(mesh, Nz)
    panels_am = map(a -> a .* FT(flux_gain), panels_am)
    panels_bm = map(a -> a .* FT(flux_gain), panels_bm)
    if scheme isa LinRoodPPMScheme
        vertical = HybridSigmaPressure(FT[0, 100, 500], FT[0, 0.2, 1])
        grid = AtmosGrid(mesh, vertical, CPU(); FT)
        state = CubedSphereState(DryBasis, mesh, panels_m; tracer=panels_rm)
        fluxes = CubedSphereFaceFluxState{DryBasis}(panels_am, panels_bm, panels_cm)
        ws = AtmosTransport.Operators.CSLinRoodAdvectionWorkspace(mesh, state.air_mass[1])
        m0 = total_air_mass(state)
        rm0 = total_mass(state, :tracer)
        for _ in 1:steps
            strang_split!(state, fluxes, grid, scheme; workspace=ws)
        end
        return (total_air_mass(state) - m0) / m0, (total_mass(state, :tracer) - rm0) / rm0
    end

    ws = CSAdvectionWorkspace(mesh, Nz; FT)
    m0 = total_interior(panels_m, Nc, Hp, Nz)
    rm0 = total_interior(panels_rm, Nc, Hp, Nz)
    for _ in 1:steps
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, scheme, ws; subcycle_count=1)
    end
    return (total_interior(panels_m, Nc, Hp, Nz) - m0) / m0,
           (total_interior(panels_rm, Nc, Hp, Nz) - rm0) / rm0
end

# ---------------------------------------------------------------------------
# Panel connectivity
# ---------------------------------------------------------------------------

@testset "CubedSphereMesh geometry" begin
    @testset "Construction and area" begin
        for Nc in [8, 24, 48]
            mesh = CubedSphereMesh(Nc=Nc)
            @test ncells(mesh) == 6 * Nc^2
            total_area = 6 * sum(mesh.cell_areas)
            expected = 4π * mesh.radius^2
            @test abs(total_area - expected) / expected < 1e-12
        end
    end

    @testset "F32 construction" begin
        mesh = CubedSphereMesh(Nc=12, FT=Float32)
        @test eltype(mesh) == Float32
        total_area = 6 * sum(mesh.cell_areas)
        expected = 4f0 * Float32(π) * mesh.radius^2
        @test abs(total_area - expected) / expected < 1f-5
    end

    @testset "Connectivity reciprocal" begin
        mesh = CubedSphereMesh(Nc=8)
        conn = mesh.connectivity
        for p in 1:6, e in 1:4
            nb = conn.neighbors[p][e]
            re = reciprocal_edge(conn, p, e)
            back = conn.neighbors[nb.panel][re]
            @test back.panel == p
        end
    end

    @testset "Metric symmetry" begin
        mesh = CubedSphereMesh(Nc=24)
        # All panels should have the same areas by gnomonic symmetry
        # (areas are computed for panel 1 and shared)
        @test all(mesh.cell_areas .> 0)
        @test all(mesh.Δx .> 0)
        @test all(mesh.Δy .> 0)
        # Cell area should be larger at panel center than edges
        mid = div(mesh.Nc, 2)
        @test mesh.cell_areas[mid, mid] > mesh.cell_areas[1, 1]
    end
end

# ---------------------------------------------------------------------------
# Halo exchange
# ---------------------------------------------------------------------------

@testset "Halo exchange" begin
    @testset "Edge fill — no zeros" begin
        mesh = CubedSphereMesh(Nc=8, Hp=1)
        Nc, Hp = mesh.Nc, mesh.Hp
        N = Nc + 2Hp; Nz = 2

        panels = ntuple(6) do p
            q = zeros(Float64, N, N, Nz)
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                q[Hp+i, Hp+j, k] = 1000.0*p + 100.0*k + 10.0*j + i
            end
            q
        end

        fill_panel_halos!(panels, mesh; dir=0)

        for p in 1:6, k in 1:Nz, s in 1:Nc, d in 1:Hp
            @test panels[p][Hp+s, Hp+Nc+d, k] != 0.0  # north
            @test panels[p][Hp+s, Hp+1-d, k] != 0.0    # south
            @test panels[p][Hp+Nc+d, Hp+s, k] != 0.0   # east
            @test panels[p][Hp+1-d, Hp+s, k] != 0.0    # west
        end
    end

    @testset "Edge consistency — P1 east ↔ P2 west (aligned)" begin
        mesh = CubedSphereMesh(Nc=8, Hp=1)
        Nc, Hp = mesh.Nc, mesh.Hp
        N = Nc + 2Hp; Nz = 2

        panels = ntuple(6) do p
            q = zeros(Float64, N, N, Nz)
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                q[Hp+i, Hp+j, k] = 1000.0*p + 100.0*k + 10.0*j + i
            end
            q
        end

        fill_panel_halos!(panels, mesh; dir=0)

        # P1 east halo should match P2 west interior
        for k in 1:Nz, s in 1:Nc
            @test panels[1][Hp+Nc+1, Hp+s, k] == panels[2][Hp+1, Hp+s, k]
        end
    end

    @testset "Corner fill — no zeros (dir=1 and dir=2)" begin
        mesh = CubedSphereMesh(Nc=8, Hp=1)
        Nc, Hp = mesh.Nc, mesh.Hp
        N = Nc + 2Hp; Nz = 2

        for dir in [1, 2]
            panels = ntuple(6) do p
                q = zeros(Float64, N, N, Nz)
                for k in 1:Nz, j in 1:Nc, i in 1:Nc
                    q[Hp+i, Hp+j, k] = 1000.0*p + 10.0*j + i
                end
                q
            end
            fill_panel_halos!(panels, mesh; dir=dir)

            for p in 1:6, k in 1:Nz, dj in 1:Hp, di in 1:Hp
                @test panels[p][Hp+1-di, Hp+1-dj, k] != 0.0      # SW
                @test panels[p][Hp+Nc+di, Hp+1-dj, k] != 0.0     # SE
                @test panels[p][Hp+Nc+di, Hp+Nc+dj, k] != 0.0    # NE
                @test panels[p][Hp+1-di, Hp+Nc+dj, k] != 0.0     # NW
            end
        end
    end
end

# ---------------------------------------------------------------------------
# CS Strang splitting — Upwind scheme
# ---------------------------------------------------------------------------

@testset "CS Strang splitting — UpwindScheme" begin
    @testset "Uniform field invariance" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=1, Nz=4, vmr=411.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        # Small uniform eastward flux
        base_am = zeros(Float64, N+1, N, Nz)
        for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
            base_am[i, j, k] = 0.03
        end

        panels_am = ntuple(_ -> copy(base_am), 6)
        panels_bm = ntuple(_ -> zeros(Float64, N, N+1, Nz), 6)
        panels_cm = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        scheme = UpwindScheme()
        ws = CSAdvectionWorkspace(mesh, Nz)

        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, scheme, ws)

        dev = max_vmr_deviation(panels_rm, panels_m, Nc, Hp, Nz, 411.0)
        @test dev < 1e-10
    end

    @testset "Packed multi-tracer path matches single-tracer reference" begin
        for scheme in (UpwindScheme(), PPMScheme())
            Hp = required_halo_width(scheme)
            mesh, panels_m0, panels_rm0 = make_cs_test_state(Nc=8, Hp=Hp, Nz=3, vmr=100.0)
            Nc, Nz = mesh.Nc, 3
            N = Nc + 2Hp
            panels_rm2 = ntuple(p -> panels_rm0[p] .* 1.7, 6)
            panels_am = ntuple(6) do p
                am = zeros(Float64, N + 1, N, Nz)
                for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
                    am[i, j, k] = 0.015 * sin(0.2p + 0.7i - 0.4j + 0.3k)
                end
                am
            end
            panels_bm = ntuple(6) do p
                bm = zeros(Float64, N, N + 1, Nz)
                for k in 1:Nz, j in (Hp+1):(Hp+Nc+1), i in (Hp+1):(Hp+Nc)
                    bm[i, j, k] = 0.012 * cos(0.3p - 0.5i + 0.6j + 0.2k)
                end
                bm
            end
            panels_cm = ntuple(6) do p
                cm = zeros(Float64, N, N, Nz + 1)
                for k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                    cm[i, j, k] = 0.01 * sin(0.1p + 0.2i + 0.3j - 0.7k)
                end
                cm
            end

            m_ref0 = deepcopy(panels_m0)
            m_ref = deepcopy(panels_m0)
            rm_ref1 = deepcopy(panels_rm0)
            rm_ref2 = deepcopy(panels_rm2)
            ws_ref = CSAdvectionWorkspace(mesh, Nz)
            strang_split_cs!(rm_ref1, m_ref, panels_am, panels_bm, panels_cm,
                             mesh, scheme, ws_ref; subcycle_count = 1)
            copyto!.(m_ref, m_ref0)
            strang_split_cs!(rm_ref2, m_ref, panels_am, panels_bm, panels_cm,
                             mesh, scheme, ws_ref; subcycle_count = 1)

            rm_mt = ntuple(p -> cat(panels_rm0[p], panels_rm2[p]; dims = 4), 6)
            m_mt = deepcopy(panels_m0)
            ws_mt = CSAdvectionWorkspace(mesh, Nz; n_tracers = 2)
            strang_split_cs_mt!(rm_mt, m_mt, panels_am, panels_bm, panels_cm,
                                mesh, scheme, ws_mt; subcycle_count = 1)
            rm_ref = ntuple(p -> cat(rm_ref1[p], rm_ref2[p]; dims = 4), 6)

            @test max_interior_absdiff_4d(rm_mt, rm_ref, Nc, Hp, Nz, 2) < 1e-12
            @test max_interior_absdiff(m_mt, m_ref, Nc, Hp, Nz) < 1e-12
        end
    end

    @testset "Packed sweep ping-pong prototypes match copy-back path" begin
        for scheme in (UpwindScheme(), PPMScheme())
            Hp = required_halo_width(scheme)
            mesh, panels_m0, panels_rm0 = make_cs_test_state(Nc=8, Hp=Hp, Nz=3, vmr=100.0)
            Nc, Nz, Nt = mesh.Nc, 3, 2
            N = Nc + 2Hp
            panels_rm2 = ntuple(p -> panels_rm0[p] .* 1.7, 6)
            panels_am = ntuple(6) do p
                am = zeros(Float64, N + 1, N, Nz)
                for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
                    am[i, j, k] = 0.015 * sin(0.2p + 0.7i - 0.4j + 0.3k)
                end
                am
            end
            panels_bm = ntuple(6) do p
                bm = zeros(Float64, N, N + 1, Nz)
                for k in 1:Nz, j in (Hp+1):(Hp+Nc+1), i in (Hp+1):(Hp+Nc)
                    bm[i, j, k] = 0.012 * cos(0.3p - 0.5i + 0.6j + 0.2k)
                end
                bm
            end
            panels_cm = ntuple(6) do p
                cm = zeros(Float64, N, N, Nz + 1)
                for k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                    cm[i, j, k] = 0.01 * sin(0.1p + 0.2i + 0.3j - 0.7k)
                end
                cm
            end

            rm_in = ntuple(p -> cat(panels_rm0[p], panels_rm2[p]; dims = 4), 6)
            m_in = deepcopy(panels_m0)
            rm_ref = deepcopy(rm_in)
            m_ref = deepcopy(m_in)
            ws_ref = CSAdvectionWorkspace(mesh, Nz; n_tracers = Nt)
            AtmosTransport.Operators.Advection._sweep_cs_horizontal!(
                rm_ref, m_ref, panels_am, mesh, scheme, ws_ref, Val(1); flux_scale=0.75)
            fill_panel_halos!(rm_ref, mesh; dir = 1)
            fill_panel_halos!(m_ref, mesh; dir = 1)

            rm_out = ntuple(p -> similar(rm_in[p]), 6)
            m_out = ntuple(p -> similar(m_in[p]), 6)
            _sweep_x_panels_mt_pingpong!(rm_out, m_out, rm_in, m_in, panels_am,
                                         mesh, scheme; flux_scale = 0.75)
            fill_panel_halos!(rm_out, mesh; dir = 1)
            fill_panel_halos!(m_out, mesh; dir = 1)

            @test max_interior_absdiff_4d(rm_out, rm_ref, Nc, Hp, Nz, Nt) < 1e-12
            @test max_interior_absdiff(m_out, m_ref, Nc, Hp, Nz) < 1e-12

            rm_ref = deepcopy(rm_in)
            m_ref = deepcopy(m_in)
            AtmosTransport.Operators.Advection._sweep_cs_horizontal!(
                rm_ref, m_ref, panels_bm, mesh, scheme, ws_ref, Val(2); flux_scale=0.75)
            fill_panel_halos!(rm_ref, mesh; dir = 2)
            fill_panel_halos!(m_ref, mesh; dir = 2)

            rm_out = ntuple(p -> similar(rm_in[p]), 6)
            m_out = ntuple(p -> similar(m_in[p]), 6)
            _sweep_y_panels_mt_pingpong!(rm_out, m_out, rm_in, m_in, panels_bm,
                                         mesh, scheme; flux_scale = 0.75)
            fill_panel_halos!(rm_out, mesh; dir = 2)
            fill_panel_halos!(m_out, mesh; dir = 2)

            @test max_interior_absdiff_4d(rm_out, rm_ref, Nc, Hp, Nz, Nt) < 1e-12
            @test max_interior_absdiff(m_out, m_ref, Nc, Hp, Nz) < 1e-12

            rm_ref = deepcopy(rm_in)
            m_ref = deepcopy(m_in)
            for p in 1:6
                _sweep_z_panel_mt!(rm_ref[p], m_ref[p], panels_cm[p],
                                   scheme, ws_ref.rm_4d_A, ws_ref.m_A,
                                   Nc, Hp, Nz, Nt; flux_scale = 0.75)
            end

            rm_out = ntuple(p -> similar(rm_in[p]), 6)
            m_out = ntuple(p -> similar(m_in[p]), 6)
            _sweep_z_panels_mt_pingpong!(rm_out, m_out, rm_in, m_in, panels_cm,
                                         mesh, scheme, ws_ref; flux_scale = 0.75)

            @test max_interior_absdiff_4d(rm_out, rm_ref, Nc, Hp, Nz, Nt) < 1e-12
            @test max_interior_absdiff(m_out, m_ref, Nc, Hp, Nz) < 1e-12
        end
    end

    @testset "Packed Strang ping-pong prototype matches copy-back path" begin
        for scheme in (UpwindScheme(), PPMScheme())
            Hp = required_halo_width(scheme)
            mesh, panels_m0, panels_rm0 = make_cs_test_state(Nc=8, Hp=Hp, Nz=3, vmr=100.0)
            Nc, Nz, Nt = mesh.Nc, 3, 2
            N = Nc + 2Hp
            panels_rm2 = ntuple(p -> panels_rm0[p] .* 1.7, 6)
            panels_am = ntuple(6) do p
                am = zeros(Float64, N + 1, N, Nz)
                for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
                    am[i, j, k] = 0.015 * sin(0.2p + 0.7i - 0.4j + 0.3k)
                end
                am
            end
            panels_bm = ntuple(6) do p
                bm = zeros(Float64, N, N + 1, Nz)
                for k in 1:Nz, j in (Hp+1):(Hp+Nc+1), i in (Hp+1):(Hp+Nc)
                    bm[i, j, k] = 0.012 * cos(0.3p - 0.5i + 0.6j + 0.2k)
                end
                bm
            end
            panels_cm = ntuple(6) do p
                cm = zeros(Float64, N, N, Nz + 1)
                for k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                    cm[i, j, k] = 0.01 * sin(0.1p + 0.2i + 0.3j - 0.7k)
                end
                cm
            end

            rm_ref = ntuple(p -> cat(panels_rm0[p], panels_rm2[p]; dims = 4), 6)
            m_ref = deepcopy(panels_m0)
            ws_ref = CSAdvectionWorkspace(mesh, Nz; n_tracers = Nt)
            _strang_split_cs_mt_copyback!(rm_ref, m_ref, panels_am, panels_bm, panels_cm,
                                          mesh, scheme, ws_ref; subcycle_count = 1)

            rm_ping = ntuple(p -> cat(panels_rm0[p], panels_rm2[p]; dims = 4), 6)
            m_ping = deepcopy(panels_m0)
            ws_ping = CSAdvectionWorkspace(mesh, Nz; n_tracers = Nt)
            rm_final, m_final = strang_split_cs_mt_pingpong!(
                rm_ping, m_ping, ws_ping.rm_4d_pp_buf, ws_ping.m_pp_buf,
                panels_am, panels_bm, panels_cm,
                mesh, scheme, ws_ping; subcycle_count = 1)

            @test rm_final === rm_ping
            @test m_final === m_ping
            @test max_interior_absdiff_4d(rm_final, rm_ref, Nc, Hp, Nz, Nt) < 1e-12
            @test max_interior_absdiff(m_final, m_ref, Nc, Hp, Nz) < 1e-12

            # Contract: the ping-pong path needs a buffer-aware (2-arg) midpoint!;
            # a 0-arg midpoint! would silently mutate the stale buffer, so it must
            # error loudly instead.
            rm_bad = ntuple(p -> cat(panels_rm0[p], panels_rm2[p]; dims = 4), 6)
            m_bad = deepcopy(panels_m0)
            ws_bad = CSAdvectionWorkspace(mesh, Nz; n_tracers = Nt)
            @test_throws ArgumentError strang_split_cs_mt_pingpong!(
                rm_bad, m_bad, ws_bad.rm_4d_pp_buf, ws_bad.m_pp_buf,
                panels_am, panels_bm, panels_cm,
                mesh, scheme, ws_bad; subcycle_count = 1, midpoint! = () -> nothing)
        end
    end

    @testset "Mass conservation — interior fluxes" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=1, Nz=4, vmr=100.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        panels_am = ntuple(6) do _
            am = zeros(Float64, N+1, N, Nz)
            for k in 1:Nz, j in (Hp+3):(Hp+Nc-2), i in (Hp+3):(Hp+Nc-1)
                am[i, j, k] = 0.04 * sin(Float64(i)*0.7 + Float64(j)*1.3)
            end
            am
        end
        panels_bm = ntuple(6) do _
            bm = zeros(Float64, N, N+1, Nz)
            for k in 1:Nz, j in (Hp+3):(Hp+Nc-1), i in (Hp+3):(Hp+Nc-2)
                bm[i, j, k] = 0.04 * cos(Float64(i)*1.1 + Float64(j)*0.9)
            end
            bm
        end
        panels_cm = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        rm0 = total_interior(panels_rm, Nc, Hp, Nz)
        ws = CSAdvectionWorkspace(mesh, Nz)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, UpwindScheme(), ws)
        rm1 = total_interior(panels_rm, Nc, Hp, Nz)

        @test abs(rm1 - rm0) / rm0 < 1e-13
    end
end

# ---------------------------------------------------------------------------
# CS Strang splitting — SlopesScheme
# ---------------------------------------------------------------------------

@testset "CS Strang splitting — SlopesScheme{MonotoneLimiter}" begin
    @testset "Uniform field invariance" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=2, Nz=4, vmr=411.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        panels_am = ntuple(6) do _
            am = zeros(Float64, N+1, N, Nz)
            for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
                am[i, j, k] = 0.05 * sin(Float64(i)*0.7 + Float64(j)*1.3 + Float64(k)*0.5)
            end
            am
        end
        panels_bm = ntuple(6) do _
            bm = zeros(Float64, N, N+1, Nz)
            for k in 1:Nz, j in (Hp+1):(Hp+Nc+1), i in (Hp+1):(Hp+Nc)
                bm[i, j, k] = 0.05 * cos(Float64(i)*1.1 + Float64(j)*0.9 + Float64(k)*0.3)
            end
            bm
        end
        panels_cm = ntuple(6) do _
            cm = zeros(Float64, N, N, Nz+1)
            for k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                cm[i, j, k] = 0.025 * sin(Float64(i)*0.3 + Float64(k)*2.1)
            end
            cm
        end

        ws = CSAdvectionWorkspace(mesh, Nz)
        scheme = SlopesScheme(MonotoneLimiter())
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, scheme, ws)

        dev = max_vmr_deviation(panels_rm, panels_m, Nc, Hp, Nz, 411.0)
        @test dev < 1e-10
    end

    @testset "Mass conservation — interior fluxes" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=16, Hp=2, Nz=4, vmr=350.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        margin = 3
        panels_am = ntuple(6) do _
            am = zeros(Float64, N+1, N, Nz)
            for k in 1:Nz, j in (Hp+margin):(Hp+Nc-margin+1)
                for i in (Hp+margin):(Hp+Nc-margin+2)
                    am[i, j, k] = 0.05 * sin(Float64(i)*0.7 + Float64(j)*1.3 + Float64(k)*0.5)
                end
            end
            am
        end
        panels_bm = ntuple(6) do _
            bm = zeros(Float64, N, N+1, Nz)
            for k in 1:Nz, j in (Hp+margin):(Hp+Nc-margin+2)
                for i in (Hp+margin):(Hp+Nc-margin+1)
                    bm[i, j, k] = 0.05 * cos(Float64(i)*1.1 + Float64(j)*0.9 + Float64(k)*0.3)
                end
            end
            bm
        end
        panels_cm = ntuple(6) do _
            cm = zeros(Float64, N, N, Nz+1)
            for k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                cm[i, j, k] = 0.025 * sin(Float64(i)*0.3 + Float64(k)*2.1)
            end
            cm
        end

        rm0 = total_interior(panels_rm, Nc, Hp, Nz)
        m0  = total_interior(panels_m, Nc, Hp, Nz)
        ws = CSAdvectionWorkspace(mesh, Nz)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, SlopesScheme(MonotoneLimiter()), ws)
        rm1 = total_interior(panels_rm, Nc, Hp, Nz)
        m1  = total_interior(panels_m, Nc, Hp, Nz)

        @test abs(rm1 - rm0) / rm0 < 1e-13
        @test abs(m1 - m0) / m0 < 1e-13
    end

    @testset "Cross-panel conservation — uniform flux" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=2, Nz=4, vmr=100.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        # Identical uniform eastward flux for all panels
        base_am = zeros(Float64, N+1, N, Nz)
        for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
            base_am[i, j, k] = 0.02
        end
        panels_am = ntuple(_ -> copy(base_am), 6)
        panels_bm = ntuple(_ -> zeros(Float64, N, N+1, Nz), 6)
        panels_cm = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        rm0 = total_interior(panels_rm, Nc, Hp, Nz)
        ws = CSAdvectionWorkspace(mesh, Nz)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, SlopesScheme(MonotoneLimiter()), ws)
        rm1 = total_interior(panels_rm, Nc, Hp, Nz)

        @test abs(rm1 - rm0) / rm0 < 1e-13
    end

    @testset "Panel interior symmetry — central cells match across panels" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=2, Nz=4, vmr=411.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        # Non-uniform initial tracer (same pattern on all panels)
        for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
            panels_rm[p][Hp+i, Hp+j, k] = 411.0 + 10.0*sin(Float64(i)/Nc*π) * cos(Float64(j)/Nc*π)
        end
        fill_panel_halos!(panels_rm, mesh; dir=0)

        # Identical fluxes, zero at boundaries
        margin = 3
        base_am = zeros(Float64, N+1, N, Nz)
        for k in 1:Nz, j in (Hp+margin):(Hp+Nc-margin+1)
            for i in (Hp+margin):(Hp+Nc-margin+2)
                base_am[i, j, k] = 0.02
            end
        end
        panels_am = ntuple(_ -> copy(base_am), 6)
        panels_bm = ntuple(_ -> zeros(Float64, N, N+1, Nz), 6)
        panels_cm = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        ws = CSAdvectionWorkspace(mesh, Nz)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, SlopesScheme(MonotoneLimiter()), ws)

        # Central cells (far from panel edges) should be identical across panels
        # because they don't see the halo differences
        center = (Hp+margin+1):(Hp+Nc-margin)
        for p in 2:6
            @test panels_rm[p][center, center, :] ≈ panels_rm[1][center, center, :] atol=1e-14
        end
    end
end

# ---------------------------------------------------------------------------
# F32 precision
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CS Strang splitting — PPMScheme
# ---------------------------------------------------------------------------

@testset "CS Strang splitting — PPMScheme" begin
    @testset "Uniform field invariance" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=3, Nz=4, vmr=411.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        panels_am = ntuple(6) do _
            am = zeros(Float64, N+1, N, Nz)
            for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
                am[i, j, k] = 0.03 * sin(Float64(i)*0.5 + Float64(j)*1.1 + Float64(k)*0.4)
            end
            am
        end
        panels_bm = ntuple(6) do _
            bm = zeros(Float64, N, N+1, Nz)
            for k in 1:Nz, j in (Hp+1):(Hp+Nc+1), i in (Hp+1):(Hp+Nc)
                bm[i, j, k] = 0.03 * cos(Float64(i)*1.3 + Float64(j)*0.7 + Float64(k)*0.2)
            end
            bm
        end
        panels_cm = ntuple(6) do _
            cm = zeros(Float64, N, N, Nz+1)
            for k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                cm[i, j, k] = 0.015 * sin(Float64(i)*0.3 + Float64(k)*2.1)
            end
            cm
        end

        ws = CSAdvectionWorkspace(mesh, Nz)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, PPMScheme(), ws)

        dev = max_vmr_deviation(panels_rm, panels_m, Nc, Hp, Nz, 411.0)
        @test dev < 1e-10
    end

    @testset "Mass conservation — interior fluxes" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=16, Hp=3, Nz=4, vmr=350.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        margin = 4  # PPM has wider stencil
        panels_am = ntuple(6) do _
            am = zeros(Float64, N+1, N, Nz)
            for k in 1:Nz, j in (Hp+margin):(Hp+Nc-margin+1)
                for i in (Hp+margin):(Hp+Nc-margin+2)
                    am[i, j, k] = 0.04 * sin(Float64(i)*0.7 + Float64(j)*1.3 + Float64(k)*0.5)
                end
            end
            am
        end
        panels_bm = ntuple(6) do _
            bm = zeros(Float64, N, N+1, Nz)
            for k in 1:Nz, j in (Hp+margin):(Hp+Nc-margin+2)
                for i in (Hp+margin):(Hp+Nc-margin+1)
                    bm[i, j, k] = 0.04 * cos(Float64(i)*1.1 + Float64(j)*0.9 + Float64(k)*0.3)
                end
            end
            bm
        end
        panels_cm = ntuple(6) do _
            cm = zeros(Float64, N, N, Nz+1)
            for k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                cm[i, j, k] = 0.02 * sin(Float64(i)*0.3 + Float64(k)*2.1)
            end
            cm
        end

        rm0 = total_interior(panels_rm, Nc, Hp, Nz)
        m0  = total_interior(panels_m, Nc, Hp, Nz)
        ws = CSAdvectionWorkspace(mesh, Nz)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, PPMScheme(), ws)
        rm1 = total_interior(panels_rm, Nc, Hp, Nz)
        m1  = total_interior(panels_m, Nc, Hp, Nz)

        @test abs(rm1 - rm0) / rm0 < 1e-13
        @test abs(m1 - m0) / m0 < 1e-13
    end
end

@testset "CS advection conserves structured tracers across mirrored seams" begin
    # Mirrors GCHP/FV3's contract: C-grid seam mass fluxes are oriented by the
    # cubed-sphere contact map, then exchanged into each panel's boundary slots.
    # A structured tracer background must conserve in F64 under that contract.
    schemes = (
        ("upwind", UpwindScheme(), 1e-9),
        ("ppm", PPMScheme(), 1e-9),
        ("linrood5", LinRoodPPMScheme(5), 1e-9),
        ("linrood7", LinRoodPPMScheme(7), 1e-9),
    )
    for (label, scheme, tol) in schemes
        @testset "$label" begin
            air_rel, tracer_rel = run_mirrored_seam_advection_conservation(scheme)
            @test abs(air_rel) < 1e-12
            @test abs(tracer_rel) < tol
        end
    end
end

@testset "CS schemes conserve across seams at appreciable Courant number" begin
    # The old seam fixture used ~0.001 Courant numbers and loose tolerances,
    # hiding a truncation error from independently evaluated boundary fluxes.
    for convention in (AtmosTransport.Grids.GnomonicPanelConvention(),
                       AtmosTransport.Grids.GEOSNativePanelConvention()),
            FT in (Float32, Float64), scheme in
            (UpwindScheme(), SlopesScheme(MonotoneLimiter()), PPMScheme(),
             LinRoodPPMScheme(5), LinRoodPPMScheme(7))
        air_rel, tracer_rel = run_mirrored_seam_advection_conservation(
            scheme; FT, flux_gain=100, convention)
        tolerance = FT == Float32 ? 3e-7 : 2e-14
        @test abs(air_rel) < tolerance
        @test abs(tracer_rel) < tolerance
    end
end

@testset "Lin-Rood q-space seam conservation at valid CFL" begin
    Adv = AtmosTransport.Operators.Advection
    for FT in (Float32, Float64), ord in (5, 7), convention in
            (AtmosTransport.Grids.GnomonicPanelConvention(),
             AtmosTransport.Grids.GEOSNativePanelConvention())
        Nc, Hp, Nz = 8, 3, 2
        mesh, m, rm = make_structured_cs_state(; FT, Nc, Hp, Nz, convention)
        am, bm, _ = make_mirrored_cs_horizontal_fluxes(mesh, Nz)
        am = map(a -> Adv._cs_flux_x_interior(a .* FT(100), Nc, Hp), am)
        bm = map(a -> Adv._cs_flux_y_interior(a .* FT(100), Nc, Hp), bm)
        q = map((r, a) -> r ./ a, rm, m)
        mq = map(copy, m)
        ws = Adv.CSLinRoodAdvectionWorkspace(mesh, m[1])
        wsq = Adv.CSLinRoodAdvectionWorkspace(mesh, m[1])
        total0 = total_interior(rm, Nc, Hp, Nz)
        air0 = total_interior(m, Nc, Hp, Nz)
        for _ in 1:2
            Adv.fv_tp_2d_cs!(rm, m, am, bm, mesh, Val(ord), ws.cs, ws.linrood)
            Adv.fv_tp_2d_cs_q!(q, mq, am, bm, mesh, Val(ord), wsq.cs, wsq.linrood)
        end
        mass_from_q = map((v, a) -> v .* a, q, mq)
        tol = FT == Float32 ? 2e-7 : 2e-14
        @test abs(total_interior(mass_from_q, Nc, Hp, Nz) / total0 - 1) < tol
        @test abs(total_interior(mq, Nc, Hp, Nz) / air0 - 1) < tol
        @test max_interior_absdiff(mass_from_q, rm, Nc, Hp, Nz) / (total0 / (6Nc^2*Nz)) < 5tol
        @test max_interior_absdiff(mq, m, Nc, Hp, Nz) / (air0 / (6Nc^2*Nz)) < 5tol
    end
end

@testset "CS monotone advection preserves signed tracer offsets" begin
    FT = Float64
    Nc, Nz = 8, 2
    q0 = FT(400e-6)
    schemes = (
        SlopesScheme(MonotoneLimiter()),
        PPMScheme(MonotoneLimiter()),
        LinRoodPPMScheme(5),
    )

    function advance_signed(scheme, mesh, panels_m, panels_rm,
                            panels_am, panels_bm, panels_cm)
        if scheme isa LinRoodPPMScheme
            vertical = HybridSigmaPressure(FT[0, 100, 500], FT[0, 0.2, 1])
            grid = AtmosGrid(mesh, vertical, CPU(); FT)
            state = CubedSphereState(DryBasis, mesh, panels_m; tracer=panels_rm)
            fluxes = CubedSphereFaceFluxState{DryBasis}(
                panels_am, panels_bm, panels_cm)
            ws = AtmosTransport.Operators.CSLinRoodAdvectionWorkspace(
                mesh, state.air_mass[1])
            strang_split!(state, fluxes, grid, scheme; workspace=ws)
            return state.air_mass, state.tracers.tracer
        end

        ws = CSAdvectionWorkspace(mesh, Nz; FT)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, scheme, ws; subcycle_count=1)
        return panels_m, panels_rm
    end

    for scheme in schemes
        @testset "$(nameof(typeof(scheme)))" begin
            Hp = required_halo_width(scheme)
            mesh, m_initial, rm_absolute =
                make_structured_cs_state(; FT, Nc, Hp, Nz)
            am, bm, cm = make_mirrored_cs_horizontal_fluxes(mesh, Nz)

            rm_anomaly = ntuple(p -> rm_absolute[p] .- q0 .* m_initial[p], 6)
            rm_negative = ntuple(p -> -q0 .* m_initial[p], 6)

            m_abs, rm_abs_out = advance_signed(
                scheme, mesh, map(copy, m_initial), map(copy, rm_absolute),
                map(copy, am), map(copy, bm), map(copy, cm))
            m_anom, rm_anom_out = advance_signed(
                scheme, mesh, map(copy, m_initial), map(copy, rm_anomaly),
                map(copy, am), map(copy, bm), map(copy, cm))
            m_neg, rm_neg_out = advance_signed(
                scheme, mesh, map(copy, m_initial), map(copy, rm_negative),
                map(copy, am), map(copy, bm), map(copy, cm))

            @test max_interior_absdiff(m_abs, m_anom, Nc, Hp, Nz) == 0
            @test max_interior_absdiff(m_abs, m_neg, Nc, Hp, Nz) == 0

            offset_error = 0.0
            negative_vmr_error = 0.0
            absolute_scale = 0.0
            for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
                ii, jj = Hp + i, Hp + j
                reconstructed = rm_anom_out[p][ii, jj, k] +
                                q0 * m_anom[p][ii, jj, k]
                offset_error = max(offset_error,
                                   abs(reconstructed - rm_abs_out[p][ii, jj, k]))
                negative_vmr_error = max(negative_vmr_error,
                    abs(rm_neg_out[p][ii, jj, k] / m_neg[p][ii, jj, k] + q0))
                absolute_scale = max(absolute_scale,
                                     abs(rm_abs_out[p][ii, jj, k]))
            end
            @test offset_error <= 512eps(absolute_scale)
            @test negative_vmr_error <= 128eps(q0)
        end
    end
end

@testset "CS source mass closure — PPMScheme" begin
    @testset "Surface source stays equal to integrated source through transport" begin
        FT = Float64
        Nc, Hp, Nz = 12, 3, 4
        mesh = CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp
        vertical = HybridSigmaPressure(FT[0, 100, 300, 600, 1000],
                                       FT[0, 0, 0, 0.5, 1])
        grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)

        panels_m = ntuple(6) do p
            m = zeros(FT, N, N, Nz)
            for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                m[i, j, k] = FT(1.0e6) * (1 + FT(0.05) * k + FT(0.001) * p)
            end
            m
        end
        tracer = ntuple(_ -> zeros(FT, N, N, Nz), 6)
        fill_panel_halos!(panels_m, mesh; dir=0)
        fill_panel_halos!(tracer, mesh; dir=0)
        state = CubedSphereState(DryBasis, mesh, panels_m; FossilCO2=tracer)

        fluxes = allocate_face_fluxes(mesh, Nz; FT=FT, basis=DryBasis)
        margin = 4
        for p in 1:6, k in 1:Nz
            for j in (Hp+margin):(Hp+Nc-margin+1), i in (Hp+margin):(Hp+Nc-margin+2)
                fluxes.am[p][i, j, k] = FT(25.0) * sin(FT(0.17) * i + FT(0.11) * j + FT(0.13) * k + p)
            end
            for j in (Hp+margin):(Hp+Nc-margin+2), i in (Hp+margin):(Hp+Nc-margin+1)
                fluxes.bm[p][i, j, k] = FT(20.0) * cos(FT(0.09) * i - FT(0.14) * j + FT(0.07) * k + p)
            end
        end
        for p in 1:6, k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
            fluxes.cm[p][i, j, k] = FT(10.0) * sin(FT(0.05) * i + FT(0.08) * j - FT(0.19) * k + p)
        end

        rates = ntuple(6) do p
            [FT(0.02) * (1 + FT(0.01) * p + FT(0.001) * i + FT(0.002) * j)
             for i in 1:Nc, j in 1:Nc]
        end
        source_rate = total_cs_surface_rate(rates)
        emissions = SurfaceFluxOperator(SurfaceFluxSource(:FossilCO2, rates))
        ws = CSAdvectionWorkspace(mesh, Nz; n_tracers=1)
        dt = FT(600)

        initial_air = total_air_mass(state)
        expected = zero(FT)
        for _ in 1:6
            strang_split!(state, fluxes, grid, PPMScheme();
                          workspace=ws,
                          cfl_limit=FT(0.95),
                          diffusion_op=NoDiffusion(),
                          emissions_op=emissions,
                          dt=dt)
            expected += source_rate * dt
            storage = total_mass(state, :FossilCO2)
            @test isapprox(storage, expected; rtol=1e-12, atol=1e-8)
            @test isapprox(total_air_mass(state), initial_air; rtol=1e-13, atol=1e-6)
        end
    end
end

# ---------------------------------------------------------------------------
# Halo validation
# ---------------------------------------------------------------------------

@testset "CS halo validation" begin
    @testset "SlopesScheme requires Hp ≥ 2" begin
        mesh_hp1 = CubedSphereMesh(Nc=8, Hp=1)
        N = 8 + 2; Nz = 2
        ws = CSAdvectionWorkspace(mesh_hp1, Nz)
        pr = ntuple(_ -> ones(Float64, N, N, Nz), 6)
        pm = ntuple(_ -> ones(Float64, N, N, Nz), 6)
        pa = ntuple(_ -> zeros(Float64, N+1, N, Nz), 6)
        pb = ntuple(_ -> zeros(Float64, N, N+1, Nz), 6)
        pc = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        @test_throws ErrorException strang_split_cs!(pr, pm, pa, pb, pc,
                                                      mesh_hp1, SlopesScheme(), ws)
    end

    @testset "PPMScheme requires Hp ≥ 3" begin
        mesh_hp2 = CubedSphereMesh(Nc=8, Hp=2)
        N = 8 + 4; Nz = 2
        ws = CSAdvectionWorkspace(mesh_hp2, Nz)
        pr = ntuple(_ -> ones(Float64, N, N, Nz), 6)
        pm = ntuple(_ -> ones(Float64, N, N, Nz), 6)
        pa = ntuple(_ -> zeros(Float64, N+1, N, Nz), 6)
        pb = ntuple(_ -> zeros(Float64, N, N+1, Nz), 6)
        pc = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        @test_throws ErrorException strang_split_cs!(pr, pm, pa, pb, pc,
                                                      mesh_hp2, PPMScheme(), ws)
    end
end

# ---------------------------------------------------------------------------
# CS Poisson balance (LLPoissonWorkspace zero-allocation)
# ---------------------------------------------------------------------------

@testset "LLPoissonWorkspace zero-allocation balance" begin
    Prep = AtmosTransport.Preprocessing
    if isdefined(Prep, :LLPoissonWorkspace) && isdefined(Prep, :balance_mass_fluxes!)
        Nx, Ny, Nz = 24, 12, 4
        am = rand(Float64, Nx+1, Ny, Nz) .* 0.01
        bm = rand(Float64, Nx, Ny+1, Nz) .* 0.01
        dm = rand(Float64, Nx, Ny, Nz) .* 1e-6

        ws = Prep.LLPoissonWorkspace(Nx, Ny)

        # Warm up
        Prep.balance_mass_fluxes!(copy(am), copy(bm), copy(dm), ws)
        prev_logger = global_logger(NullLogger())
        try
            Prep.balance_mass_fluxes!(copy(am), copy(bm), copy(dm), ws)

            # Measure allocation
            am2 = copy(am); bm2 = copy(bm); dm2 = copy(dm)
            alloc = @allocated Prep.balance_mass_fluxes!(am2, bm2, dm2, ws)
            # Should be near-zero once logging allocations are removed.
            @test alloc < 10_000  # < 10 KB
        finally
            global_logger(prev_logger)
        end
    else
        @test true  # skip if Preprocessing not available
    end
end

# ---------------------------------------------------------------------------
# F32 precision
# ---------------------------------------------------------------------------

@testset "CS advection — Float32" begin
    mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=2, Nz=4, FT=Float32, vmr=411f0)
    Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
    N = Nc + 2Hp

    panels_am = ntuple(_ -> zeros(Float32, N+1, N, Nz), 6)
    panels_bm = ntuple(_ -> zeros(Float32, N, N+1, Nz), 6)
    panels_cm = ntuple(_ -> zeros(Float32, N, N, Nz+1), 6)

    for p in 1:6, k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
        panels_am[p][i, j, k] = 0.02f0 * sin(Float32(i)*0.7f0)
    end

    rm0 = Float64(total_interior(panels_rm, Nc, Hp, Nz))
    ws = CSAdvectionWorkspace(mesh, Nz; FT=Float32)
    strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                     mesh, SlopesScheme(MonotoneLimiter()), ws)

    dev = max_vmr_deviation(panels_rm, panels_m, Nc, Hp, Nz, 411.0)
    @test dev < 1f-4  # F32 has ~7 digits
end

@testset "Shared-seam horizontal adjoint at appreciable CFL" begin
    Adv = AtmosTransport.Operators.Advection
    Adj = AtmosTransport.Adjoints
    for ord in (5, 7), convention in
            (AtmosTransport.Grids.GnomonicPanelConvention(),
             AtmosTransport.Grids.GEOSNativePanelConvention())
        Nc, Hp, Nz = 8, 3, 2
        mesh, m, rm = make_structured_cs_state(; Nc, Hp, Nz, convention)
        # Perturb the smooth field reproducibly to avoid symmetric limiter ties.
        # The transport adjoint differentiates tracers at fixed meteorology.
        rng = MersenneTwister(1918)
        for p in 1:6, k in 1:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc
            rm[p][i, j, k] *= 1 + 0.002randn(rng)
        end
        am, bm, _ = make_mirrored_cs_horizontal_fluxes(mesh, Nz)
        am = map(a -> a .* 100, am)
        bm = map(a -> a .* 100, bm)
        inner_am = map(a -> Adv._cs_flux_x_interior(a, Nc, Hp), am)
        inner_bm = map(a -> Adv._cs_flux_y_interior(a, Nc, Hp), bm)
        out, mout = map(copy, rm), map(copy, m)
        record = Adj._record_linrood_horizontal_substep!(
            out, mout, am, bm, mesh, 1.0, Val(ord))
        production, pm = map(copy, rm), map(copy, m)
        ws = Adv.CSLinRoodAdvectionWorkspace(mesh, m[1])
        Adv.fv_tp_2d_cs!(production, pm, inner_am, inner_bm,
                        mesh, Val(ord), ws.cs, ws.linrood)
        @test max_interior_absdiff(production, out, Nc, Hp, Nz) == 0
        @test max_interior_absdiff(pm, mout, Nc, Hp, Nz) == 0

        direction, seed = ntuple(_ -> map(a -> zero(a), m), 2)
        for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
            ii, jj = Hp+i, Hp+j
            direction[p][ii, jj, k] = rm[p][ii, jj, k] * 0.02sin(0.3i + 0.2j + 0.1k + p)
            seed[p][ii, jj, k] = sin(0.4i + 0.3j + 0.2k + p)
        end
        adjoint_rm, adjoint_m = map(copy, seed), map(a -> zero(a), m)
        Adj._apply_cs_linrood_horizontal_adjoint!(adjoint_rm, adjoint_m, record, mesh)
        predicted = sum(sum(adjoint_rm[p] .* direction[p]) for p in 1:6)
        function objective(h)
            perturbed = map((r, d) -> r .+ h .* d, rm, direction)
            mass = map(copy, m)
            Adv.fv_tp_2d_cs!(perturbed, mass, inner_am, inner_bm,
                            mesh, Val(ord), ws.cs, ws.linrood)
            return sum(sum(seed[p] .* perturbed[p]) for p in 1:6)
        end
        h = 1e-5
        fd = (objective(h) - objective(-h)) / (2h)
        @test predicted ≈ fd rtol=2e-8
    end
end
