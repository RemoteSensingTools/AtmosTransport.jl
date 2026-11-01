#!/usr/bin/env julia

using Test
using AtmosTransport.Operators.StepPlans: transport_step!
using Logging, Random

using AtmosTransport
using .AtmosTransport.Grids: reciprocal_edge
using .AtmosTransport.Operators: MonotoneLimiter, CW84Limiter, required_halo_width
using .AtmosTransport.Operators.Advection: fill_panel_halos!,
    strang_split_cs_mt!, strang_split!, CSAdvectionWorkspace,
    _sweep_x_panels_mt_pingpong!, _sweep_y_panels_mt_pingpong!,
    _sweep_z_panels_mt_pingpong!,
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
    vertical = HybridSigmaPressure(FT[0, 100, 500], FT[0, 0.2, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT)
    state = CubedSphereState(DryBasis, mesh, panels_m; tracer=panels_rm)
    fluxes = CubedSphereFaceFluxState{DryBasis}(panels_am, panels_bm, panels_cm)
    ws = scheme isa LinRoodPPMScheme ?
        AtmosTransport.Operators.CSLinRoodAdvectionWorkspace(mesh, state.air_mass[1]) :
        CSAdvectionWorkspace(mesh, Nz; FT, n_tracers=1)
    m0, rm0 = total_air_mass(state), total_mass(state, :tracer)
    for _ in 1:steps
        strang_split!(state, fluxes, grid, scheme; workspace=ws)
    end
    return (total_air_mass(state) - m0) / m0, (total_mass(state, :tracer) - rm0) / rm0
end

# ---------------------------------------------------------------------------
# Panel connectivity
# ---------------------------------------------------------------------------

@testset "Packed CS transport" begin
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
            ws_ref = CSAdvectionWorkspace(mesh, Nz; n_tracers=1)
            strang_split_cs_mt!(map(a -> reshape(a, size(a)..., 1), rm_ref1), m_ref, panels_am, panels_bm, panels_cm,
                             mesh, scheme, ws_ref; subcycle_count = 1)
            copyto!.(m_ref, m_ref0)
            strang_split_cs_mt!(map(a -> reshape(a, size(a)..., 1), rm_ref2), m_ref, panels_am, panels_bm, panels_cm,
                             mesh, scheme, ws_ref; subcycle_count = 1)

            rm_mt = ntuple(p -> cat(panels_rm0[p], panels_rm2[p]; dims = 4), 6)
            m_mt = deepcopy(panels_m0)
            ws_mt = CSAdvectionWorkspace(mesh, Nz; n_tracers = 2)
            strang_split_cs_mt!(rm_mt, m_mt, panels_am, panels_bm, panels_cm,
                                mesh, scheme, ws_mt; subcycle_count = 1)
            rm_ref = ntuple(p -> cat(rm_ref1[p], rm_ref2[p]; dims = 4), 6)

            I = Hp+1:Hp+Nc
            @test all(p -> isequal(rm_mt[p][I,I,:,:], rm_ref[p][I,I,:,:]), 1:6)
            @test all(p -> isequal(m_mt[p][I,I,:], m_ref[p][I,I,:]), 1:6)
        end
    end

end

function reference_cs_z(rm, m, cm, mesh, scheme, scale)
    Adv = AtmosTransport.Operators.Advection
    rout, mout = deepcopy(rm), deepcopy(m)
    Nc, Hp, Nz, Nt = mesh.Nc, mesh.Hp, size(m[1], 3), size(rm[1], 4)
    for p in 1:6, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc, k in 1:Nz
        ft, fb = scale * cm[p][i,j,k], scale * cm[p][i,j,k+1]
        mout[p][i,j,k] = m[p][i,j,k] + ft - fb
        for t in 1:Nt
            r = view(rm[p], :, :, :, t)
            if scheme isa UpwindScheme
                flux(F, donor) = m[p][i,j,donor] > 0 ?
                    clamp(F / m[p][i,j,donor], F >= 0 ? 0 : -1, F >= 0 ? 1 : 0) * r[i,j,donor] : zero(F)
                top = k == 1 ? zero(ft) : flux(ft, ft >= 0 ? k-1 : k)
                bottom = k == Nz ? zero(fb) : flux(fb, fb >= 0 ? k : k+1)
            else
                top = Adv._zface_tracer_flux(i, j, Int32(k), r, m[p], ft, scheme, Int32(Nz))
                bottom = Adv._zface_tracer_flux(i, j, Int32(k+1), r, m[p], fb, scheme, Int32(Nz))
            end
            rout[p][i,j,k,t] = r[i,j,k] + top - bottom
        end
    end
    return rout, mout
end

@testset "CS Z ping-pong sweep matches scalar reference" begin
    for FT in (Float32, Float64), Nz in (1, 5), Nt in (1, 3), scheme in
            (UpwindScheme(), SlopesScheme(MonotoneLimiter()), PPMScheme(), PPMScheme(CW84Limiter()))
        Nc, Hp = 5, required_halo_width(scheme)
        N = Nc + 2Hp
        mesh = CubedSphereMesh(; Nc, Hp, FT)
        m = ntuple(p -> FT[2 + 0.01(i+j+k+p) for i in 1:N, j in 1:N, k in 1:Nz], 6)
        rm = ntuple(p -> FT[(-1)^t * (0.4 + 0.1sin(i+j+k+t+p)) * m[p][i,j,k]
                            for i in 1:N, j in 1:N, k in 1:Nz, t in 1:Nt], 6)
        cm = ntuple(p -> FT[k in (1, Nz+1) ? 0 : 0.2sin(i+2j+k+p)
                            for i in 1:N, j in 1:N, k in 1:Nz+1], 6)
        source_rm, source_m = deepcopy(rm), deepcopy(m)
        expected_rm, expected_m = reference_cs_z(rm, m, cm, mesh, scheme, FT(0.75))
        actual_rm, actual_m = deepcopy(rm), deepcopy(m)
        ws = CSAdvectionWorkspace(mesh, m[1]; n_tracers=Nt)
        _sweep_z_panels_mt_pingpong!(actual_rm, actual_m, rm, m, cm, mesh, scheme, ws;
                                     flux_scale=FT(0.75))
        @test isequal(actual_rm, expected_rm)
        @test isequal(actual_m, expected_m)
        @test isequal(rm, source_rm)
        @test isequal(m, source_m)
    end
end

    @testset "Packed Strang ping-pong prototype returns home buffers and validates midpoint" begin
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

            rm_ping = ntuple(p -> cat(panels_rm0[p], panels_rm2[p]; dims = 4), 6)
            m_ping = deepcopy(panels_m0)
            ws_ping = CSAdvectionWorkspace(mesh, Nz; n_tracers = Nt)
            rm_final, m_final = strang_split_cs_mt_pingpong!(
                rm_ping, m_ping, ws_ping.rm_4d_pp_buf, ws_ping.m_pp_buf,
                panels_am, panels_bm, panels_cm,
                mesh, scheme, ws_ping; subcycle_count = 1)

            @test rm_final === rm_ping
            @test m_final === m_ping


        end
    end


@testset "CS schemes conserve across seams at appreciable Courant number" begin
    # The old seam fixture used ~0.001 Courant numbers and loose tolerances,
    # hiding a truncation error from independently evaluated boundary fluxes.
    for convention in (AtmosTransport.Grids.GnomonicPanelConvention(),
                       AtmosTransport.Grids.GEOSNativePanelConvention()),
            FT in (Float32, Float64), scheme in
            (UpwindScheme(), SlopesScheme(MonotoneLimiter()), PPMScheme(),
             PPMScheme(CW84Limiter()), LinRoodPPMScheme(5), LinRoodPPMScheme(7))
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
        PPMScheme(CW84Limiter()),
        LinRoodPPMScheme(5),
    )

    function advance_signed(scheme, mesh, panels_m, panels_rm,
                            panels_am, panels_bm, panels_cm)
        vertical = HybridSigmaPressure(FT[0, 100, 500], FT[0, 0.2, 1])
        grid = AtmosGrid(mesh, vertical, CPU(); FT)
        state = CubedSphereState(DryBasis, mesh, panels_m; tracer=panels_rm)
        fluxes = CubedSphereFaceFluxState{DryBasis}(panels_am, panels_bm, panels_cm)
        ws = scheme isa LinRoodPPMScheme ?
            AtmosTransport.Operators.CSLinRoodAdvectionWorkspace(mesh, state.air_mass[1]) :
            CSAdvectionWorkspace(mesh, Nz; FT, n_tracers=1)
        strang_split!(state, fluxes, grid, scheme; workspace=ws)
        return state.air_mass, state.tracers.tracer
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
            transport_step!(state, fluxes, grid, PPMScheme(), dt;
                          workspace=ws,
                          cfl_limit=FT(0.95),
                          diffusion_op=NoDiffusion(),
                          emissions_op=emissions)
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
        ws = CSAdvectionWorkspace(mesh_hp1, Nz; n_tracers=1)
        pr = ntuple(_ -> ones(Float64, N, N, Nz, 1), 6)
        pm = ntuple(_ -> ones(Float64, N, N, Nz), 6)
        pa = ntuple(_ -> zeros(Float64, N+1, N, Nz), 6)
        pb = ntuple(_ -> zeros(Float64, N, N+1, Nz), 6)
        pc = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        @test_throws ErrorException strang_split_cs_mt!(pr, pm, pa, pb, pc,
                                                      mesh_hp1, SlopesScheme(), ws)
    end

    @testset "PPMScheme requires Hp ≥ 3" begin
        mesh_hp2 = CubedSphereMesh(Nc=8, Hp=2)
        N = 8 + 4; Nz = 2
        ws = CSAdvectionWorkspace(mesh_hp2, Nz; n_tracers=1)
        pr = ntuple(_ -> ones(Float64, N, N, Nz, 1), 6)
        pm = ntuple(_ -> ones(Float64, N, N, Nz), 6)
        pa = ntuple(_ -> zeros(Float64, N+1, N, Nz), 6)
        pb = ntuple(_ -> zeros(Float64, N, N+1, Nz), 6)
        pc = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        @test_throws ErrorException strang_split_cs_mt!(pr, pm, pa, pb, pc,
                                                      mesh_hp2, PPMScheme(), ws)
    end
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
        # The tape record is built from the production operator's recorder
        # hooks, so recording leaves the production result unchanged.
        out, mout = map(copy, rm), map(copy, m)
        ws = Adv.CSLinRoodAdvectionWorkspace(mesh, m[1])
        recorder = Adj.Footprint._CSTapeRecorder(AtmosTransport.StateTape(),
            AtmosTransport.Tape._reserve_tape_pool(AtmosTransport.DeviceCSTapeStorage(), Any[]),
            AtmosTransport.LinRoodPPMScheme(ord))
        Adv.fv_tp_2d_cs!(out, mout, inner_am, inner_bm, mesh, Val(ord), ws.cs, ws.linrood;
                         recorder)
        record = only(recorder.ops)
        production, pm = map(copy, rm), map(copy, m)
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
