#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Runtime CFL subcycling composition (issue #61).
#
# When one palindrome's outflow budget exceeds `cfl_limit`, a transport step
# runs `(X Y Z)^n → midpoint → (Z Y X)^n` with every flux divided by `n`
# (`H V` / `V H` on reduced Gaussian). The earlier order `X^n Y^n Z^n | Z^n Y^n
# X^n` subcycled each direction separately: the passes of one leg still remove
# the full-flux outflow of that direction before any other direction refills
# the cell, so subcycling bounded the Courant number of a pass but not the
# mass loss within a leg. Exactly non-divergent flows then drove air mass
# negative and lost a uniform tracer.
#
# These tests check, through the public entry points and with the count the
# runtime pilot chooses:
#   * a uniform tracer stays uniform (and air mass positive) in solid-body
#     rotation and strain on the cubed sphere, in strain on lat-lon, and in an
#     overturning circulation on reduced Gaussian;
#   * the step is exactly the composition built from the sweep primitives,
#     at n > 1 and (the plain palindrome) at n = 1;
#   * the footprint tape records the same composition;
#   * a strongly divergent (draining) cell keeps positive air mass: the count
#     budgets against min(m, m_end), and invalid flux sets throw.
# ---------------------------------------------------------------------------

using Test
using LinearAlgebra
using Random

import AtmosTransport
const AT  = AtmosTransport
const Adv = AT.Operators.Advection
const G   = AT.Grids
using AtmosTransport.Operators: UpwindScheme, SlopesScheme, PPMScheme, MonotoneLimiter, CW84Limiter

# ----------------------------------------------------------------- cubed sphere
# Exactly non-divergent C12 flow from a streamfunction on the cell corners:
#   :rotation   ψ = A (0.2x + 0.7y − 0.4z)/|·|   (solid-body rotation, tilted axis)
#   :strain     ψ = A (x y + 0.3 z)              (corner flow plus a weak rotation)
function _cs_flow_problem(flow, A; Nc = 12, convention = G.GEOSNativePanelConvention())
    Hp = 3
    mesh = AT.CubedSphereMesh(; Nc, Hp, FT = Float64, radius = 1.0, convention)
    def = G.cs_definition(mesh)
    N, Nz = Nc + 2Hp, 1
    axis = normalize([0.2, 0.7, -0.4])
    ψ(v) = flow === :rotation ? A * dot(axis, v) : A * (v[1] * v[2] + 0.3 * v[3])
    m  = ntuple(_ -> zeros(N, N, Nz), 6)
    am = ntuple(_ -> zeros(N + 1, N, Nz), 6)
    bm = ntuple(_ -> zeros(N, N + 1, Nz), 6)
    cm = ntuple(_ -> zeros(N, N, Nz + 1), 6)
    for p in 1:6
        corner(i, j) = collect(G._corner_xyz(def, Nc, i, j, p, Float64))
        handed = sign(dot(cross(corner(2, 1) - corner(1, 1), corner(1, 2) - corner(1, 1)), corner(1, 1)))
        psi = [ψ(corner(i, j)) for i in 1:Nc+1, j in 1:Nc+1]
        for j in 1:Nc, i in 1:Nc;   m[p][i+Hp, j+Hp, 1] = mesh.cell_areas[i, j]; end
        for j in 1:Nc, i in 1:Nc+1; am[p][i+Hp, j+Hp, 1] =  handed * (psi[i, j+1] - psi[i, j]); end
        for j in 1:Nc+1, i in 1:Nc; bm[p][i+Hp, j+Hp, 1] = -handed * (psi[i+1, j] - psi[i, j]); end
    end
    x = map(a -> copy(a[Hp+1:Hp+Nc+1, Hp+1:Hp+Nc, :]), am)
    y = map(a -> copy(a[Hp+1:Hp+Nc, Hp+1:Hp+Nc+1, :]), bm)
    AT.Preprocessing.sync_all_cs_boundary_mirrors!(x, y, mesh.connectivity, Nc, Nz)
    for p in 1:6
        am[p][Hp+1:Hp+Nc+1, Hp+1:Hp+Nc, :] .= x[p]
        bm[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc+1, :] .= y[p]
    end
    Adv.fill_panel_halos!(m, mesh; dir = 1)
    return mesh, m, am, bm, cm
end

_cs_interior(mesh) = mesh.Hp+1:mesh.Hp+mesh.Nc
_scheme_label(s) = hasproperty(s, :limiter) ? "$(nameof(typeof(s)))($(nameof(typeof(s.limiter))))" :
                                              string(nameof(typeof(s)))
_cs_packed(panels) = ntuple(p -> reshape(copy(panels[p]), size(panels[p])..., 1), 6)
_cs_max_q_error(rm, m, q, I) = maximum(maximum(abs.(rm[p][I, I, :, 1] ./ m[p][I, I, :] .- q)) for p in 1:6)

@testset "CS subcycling keeps a uniform tracer uniform (rotation and strain)" begin
    q0 = 0.4
    for (flow, A, convention) in ((:rotation, 0.15, G.GEOSNativePanelConvention()),
                                  (:rotation, 0.20, G.GnomonicPanelConvention()),
                                  (:strain,   0.30, G.GEOSNativePanelConvention())),
        scheme in (UpwindScheme(), PPMScheme(MonotoneLimiter()), PPMScheme(CW84Limiter()))
        mesh, m0, am, bm, cm = _cs_flow_problem(flow, A; convention)
        I = _cs_interior(mesh)
        n = Adv._cs_static_palindrome_subcycle_count(am, bm, cm, m0, mesh.Nc, mesh.Hp, 1,
                                                     AT.Parameters.DEFAULT_CFL_LIMIT)
        label = "$(flow) A=$(A) $(nameof(typeof(convention))) $(_scheme_label(scheme)) n=$(n)"
        @testset "$label" begin
            @test n >= 5

            # Packed ping-pong driver (production), the midpoint observes the air mass.
            rm = _cs_packed(map(a -> q0 .* a, m0)); m = map(copy, m0)
            Adv.fill_panel_halos!(rm, mesh; dir = 1)
            ws = Adv.CSAdvectionWorkspace(mesh, m0[1]; n_tracers = 1)
            mid_min = Ref(Inf)
            midpoint! = (ar, amass) -> (mid_min[] = minimum(minimum(amass[p][I, I, :] ./ m0[p][I, I, :]) for p in 1:6))
            Adv.strang_split_cs_mt!(rm, m, am, bm, cm, mesh, scheme, ws; midpoint!)
            @test _cs_max_q_error(rm, m, q0, I) < 1e-12
            @test mid_min[] > 0.5
            @test minimum(minimum(m[p][I, I, :]) for p in 1:6) > 0

            # Packed copy-back driver.
            rm = _cs_packed(map(a -> q0 .* a, m0)); m = map(copy, m0)
            Adv.fill_panel_halos!(rm, mesh; dir = 1)
            Adv._strang_split_cs_mt_copyback!(rm, m, am, bm, cm, mesh, scheme,
                                               Adv.CSAdvectionWorkspace(mesh, m0[1]; n_tracers = 1))
            @test _cs_max_q_error(rm, m, q0, I) < 1e-12

            # Single-tracer driver.
            rm = map(a -> q0 .* a, m0); m = map(copy, m0)
            Adv.strang_split_cs!(rm, m, am, bm, cm, mesh, scheme, Adv.CSAdvectionWorkspace(mesh, 1))
            @test maximum(maximum(abs.(rm[p][I, I, :] ./ m[p][I, I, :] .- q0)) for p in 1:6) < 1e-12
        end
    end
end

# The composition `(X Y Z)^n | midpoint | (Z Y X)^n` built from the sweep
# primitives, with the halo fills of `strang_split_cs_mt_pingpong!`.
function _cs_explicit_composition(rm, m, am, bm, cm, mesh, scheme, ws, n, midpoint!)
    spare_rm = ntuple(p -> zero(rm[p]), 6); spare_m = ntuple(p -> zero(m[p]), 6)
    fs = 1.0 / n
    function sweep!(dir)
        if dir === :z
            Adv._sweep_z_panels_mt_pingpong!(spare_rm, spare_m, rm, m, cm, mesh, scheme, ws; flux_scale = fs)
        else
            sweep = dir === :x ? Adv._sweep_x_panels_mt_pingpong! : Adv._sweep_y_panels_mt_pingpong!
            sweep(spare_rm, spare_m, rm, m, dir === :x ? am : bm, mesh, scheme;
                  flux_scale = fs, seam_flux = ws.seam_flux)
            Adv.fill_panel_halos!(spare_rm, mesh; dir = dir === :x ? 1 : 2)
            Adv.fill_panel_halos!(spare_m, mesh; dir = dir === :x ? 1 : 2)
        end
        rm, spare_rm = spare_rm, rm
        m, spare_m = spare_m, m
    end
    for pass in 1:n
        pass > 1 && (Adv.fill_panel_halos!(rm, mesh; dir = 1); Adv.fill_panel_halos!(m, mesh; dir = 1))
        sweep!(:x); sweep!(:y); sweep!(:z)
    end
    midpoint!(rm, m)
    for _ in 1:n
        sweep!(:z)
        Adv.fill_panel_halos!(rm, mesh; dir = 2); Adv.fill_panel_halos!(m, mesh; dir = 2)
        sweep!(:y)
        Adv.fill_panel_halos!(rm, mesh; dir = 1); Adv.fill_panel_halos!(m, mesh; dir = 1)
        sweep!(:x)
    end
    return rm, m
end

@testset "CS subcycled step is (X Y Z)^n | midpoint | (Z Y X)^n" begin
    mesh, m0, am, bm, cm = _cs_flow_problem(:rotation, 0.06)
    I = _cs_interior(mesh)
    q = ntuple(p -> [1 + 0.3 * sin(3i + 2j + p) for i in 1:mesh.Nc+6, j in 1:mesh.Nc+6, k in 1:1], 6)
    rm0 = ntuple(p -> q[p] .* m0[p], 6)
    Adv.fill_panel_halos!(rm0, mesh; dir = 1)
    # The midpoint scales the tracer, so its position in the sequence matters.
    scale!(ar, _) = foreach(p -> (ar[p] .*= 1.01), 1:6)
    for scheme in (UpwindScheme(), PPMScheme(MonotoneLimiter())), n in (1, 3)
        rm = _cs_packed(rm0); m = map(copy, m0)
        ws = Adv.CSAdvectionWorkspace(mesh, m0[1]; n_tracers = 1)
        out_rm, out_m = Adv.strang_split_cs_mt_pingpong!(rm, m, ws.rm_4d_pp_buf, ws.m_pp_buf,
                                                         am, bm, cm, mesh, scheme, ws;
                                                         subcycle_count = n, midpoint! = scale!)
        ref_rm, ref_m = _cs_explicit_composition(_cs_packed(rm0), map(copy, m0), am, bm, cm, mesh,
                                                 scheme, Adv.CSAdvectionWorkspace(mesh, m0[1]; n_tracers = 1),
                                                 n, scale!)
        @test all(p -> out_rm[p][I, I, :, :] == ref_rm[p][I, I, :, :], 1:6)
        @test all(p -> out_m[p][I, I, :] == ref_m[p][I, I, :], 1:6)
    end
end

# ----------------------------------------------------------------------- lat-lon
# Exactly non-divergent strain on a 32×16 periodic-x, closed-y grid (unit mass).
function _ll_strain_fluxes(A; Nx = 32, Ny = 16)
    ψ(i, j) = A * sin(2π * (i - 1) / Nx) * sin(π * (j - 1) / Ny)
    am = zeros(Nx + 1, Ny, 1); bm = zeros(Nx, Ny + 1, 1); cm = zeros(Nx, Ny, 2)
    for j in 1:Ny, i in 1:Nx+1; am[i, j, 1] = ψ(i, j + 1) - ψ(i, j); end
    for j in 1:Ny+1, i in 1:Nx;  bm[i, j, 1] = -(ψ(i + 1, j) - ψ(i, j)); end
    return am, bm, cm
end

@testset "lat-lon subcycling keeps a uniform tracer uniform (strain)" begin
    for A in (15.0, 25.0, 40.0),
        scheme in (UpwindScheme(), SlopesScheme(MonotoneLimiter()), PPMScheme(MonotoneLimiter()))
        am, bm, cm = _ll_strain_fluxes(A)
        m = ones(32, 16, 1)
        n = Adv._structured_palindrome_subcycle_count(am, bm, cm, m, AT.Parameters.DEFAULT_CFL_LIMIT)
        @testset "A=$(A) $(_scheme_label(scheme)) n=$(n)" begin
            @test n > 1
            rm = reshape(0.4 .* m, 32, 16, 1, 1)
            Adv.strang_split_mt!(rm, m, am, bm, cm, scheme, Adv.AdvectionWorkspace(m; n_tracers = 1))
            @test maximum(abs.(rm[:, :, :, 1] ./ m .- 0.4)) < 1e-12
            @test minimum(m) > 0
        end
    end
end

@testset "lat-lon subcycled step is (X Y Z)^n | (Z Y X)^n" begin
    Nx, Ny, Nz = 12, 8, 3
    m0 = [1.0 + 0.1 * sin(i + 2j + 3k) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    q  = [0.4 + 0.1 * cos(2i - j + k) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    am = [0.2 * sin(0.7i + 0.3j + k) for i in 1:Nx+1, j in 1:Ny, k in 1:Nz]; am[Nx+1, :, :] .= am[1, :, :]
    bm = [0.2 * cos(0.4i + 0.9j + k) for i in 1:Nx, j in 1:Ny+1, k in 1:Nz]; bm[:, [1, Ny+1], :] .= 0
    cm = [0.2 * sin(0.5i + 0.2j + 2k) for i in 1:Nx, j in 1:Ny, k in 1:Nz+1]; cm[:, :, [1, Nz+1]] .= 0
    # scale 0.5: n = 2 (at scale 1 some cells would end the step with negative mass)
    for scale in (0.1, 0.5), scheme in (UpwindScheme(), SlopesScheme(MonotoneLimiter()), PPMScheme(MonotoneLimiter()))
        a, b, c = scale .* am, scale .* bm, scale .* cm
        n = Adv._structured_palindrome_subcycle_count(a, b, c, m0, AT.Parameters.DEFAULT_CFL_LIMIT)
        @test (n == 1) == (scale == 0.1)
        m = copy(m0); rm = reshape(q .* m0, Nx, Ny, Nz, 1)
        Adv.strang_split_mt!(rm, m, a, b, c, scheme, Adv.AdvectionWorkspace(m; n_tracers = 1))
        mr = copy(m0); rr = reshape(q .* m0, Nx, Ny, Nz, 1); ws = Adv.AdvectionWorkspace(mr; n_tracers = 1)
        fs = 1.0 / n
        for _ in 1:n
            Adv.sweep_x_mt!(rr, mr, a, scheme, ws, fs); Adv.sweep_y_mt!(rr, mr, b, scheme, ws, fs)
            Adv.sweep_z_mt!(rr, mr, c, scheme, ws, fs)
        end
        for _ in 1:n
            Adv.sweep_z_mt!(rr, mr, c, scheme, ws, fs); Adv.sweep_y_mt!(rr, mr, b, scheme, ws, fs)
            Adv.sweep_x_mt!(rr, mr, a, scheme, ws, fs)
        end
        @test isequal(rm, rr)
        @test isequal(m, mr)
    end
end

# A strongly divergent cell: one cell of mass 1 among cells of mass 100, inflow
# `I` through both faces of one direction and outflow `O` through both faces of
# the other, so after the step it holds m_end = 1 + 2 (I − O). Budgeting
# against the step-start mass alone chose n = 5 for (O, I) = (2.0, 1.51)
# (m_end = 0.02): the reverse half then removed more than the cell held, air
# mass went negative and PPM returned q = −1.7e43. The budget against
# min(m, m_end) guarantees that every pass removes at most cfl_limit / 2 of the
# mass the cell holds when the pass starts.
function _ll_draining_cell(; O, I, div_dir)
    m = fill(100.0, 8, 8, 1); m[4, 4, 1] = 1.0
    am = zeros(9, 8, 1); bm = zeros(8, 9, 1); cm = zeros(8, 8, 2)
    if div_dir === :y
        am[4, 4, 1] = I / 2; am[5, 4, 1] = -I / 2
        bm[4, 4, 1] = -O / 2; bm[4, 5, 1] = O / 2
    else
        bm[4, 4, 1] = I / 2; bm[4, 5, 1] = -I / 2
        am[4, 4, 1] = -O / 2; am[5, 4, 1] = O / 2
    end
    return m, am, bm, cm
end

_ll_outflow(am, bm, cm) = (max.(0, -am[1:end-1, :, :]) .+ max.(0, am[2:end, :, :]),
                           max.(0, -bm[:, 1:end-1, :]) .+ max.(0, bm[:, 2:end, :]),
                           max.(0, -cm[:, :, 1:end-1]) .+ max.(0, cm[:, :, 2:end]))

@testset "lat-lon subcycling of a draining cell keeps air mass positive" begin
    cfl = AT.Parameters.DEFAULT_CFL_LIMIT
    for div_dir in (:y, :x), (O, I) in ((2.0, 1.51), (2.0, 1.75), (1.2, 0.95))
        m0, am, bm, cm = _ll_draining_cell(; O, I, div_dir)
        m_end = 1 + 2 * (I - O)
        n = Adv._structured_palindrome_subcycle_count(am, bm, cm, m0, cfl)
        @testset "$(div_dir) divergent, O=$(O) I=$(I) n=$(n)" begin
            @test n == ceil(Int, 2O / (m_end * cfl))
            # Pass by pass: no pass removes more than cfl/2 of a cell's current mass.
            out = _ll_outflow(am, bm, cm)
            m = copy(m0); rr = reshape(0.4 .* m0, size(m0)..., 1)
            ws = Adv.AdvectionWorkspace(m; n_tracers = 1)
            fs = 1.0 / n
            worst_fraction = 0.0
            min_mass = Inf
            pass!(sweep!, flux, d) = begin
                worst_fraction = max(worst_fraction, maximum(fs .* out[d] ./ m))
                sweep!(rr, m, flux, UpwindScheme(), ws, fs)
                min_mass = min(min_mass, m[4, 4, 1])
            end
            for _ in 1:n
                pass!(Adv.sweep_x_mt!, am, 1); pass!(Adv.sweep_y_mt!, bm, 2); pass!(Adv.sweep_z_mt!, cm, 3)
            end
            for _ in 1:n
                pass!(Adv.sweep_z_mt!, cm, 3); pass!(Adv.sweep_y_mt!, bm, 2); pass!(Adv.sweep_x_mt!, am, 1)
            end
            @test worst_fraction <= cfl / 2 + 1e-12
            @test min_mass >= (1 - cfl / 2) * m_end - 1e-12
            @test m[4, 4, 1] ≈ m_end rtol = 1e-12

            # The production step keeps a uniform tracer uniform with every scheme.
            for scheme in (UpwindScheme(), SlopesScheme(MonotoneLimiter()), PPMScheme(MonotoneLimiter()))
                m = copy(m0); rm = reshape(0.4 .* m0, size(m0)..., 1)
                Adv.strang_split_mt!(rm, m, am, bm, cm, scheme, Adv.AdvectionWorkspace(m; n_tracers = 1))
                @test minimum(m) > 0
                @test maximum(abs.(rm[:, :, :, 1] ./ m .- 0.4)) < 1e-12
            end
        end
    end
end

@testset "cubed-sphere and reduced-Gaussian counts budget against min(m, m_end)" begin
    cfl = AT.Parameters.DEFAULT_CFL_LIMIT
    # Cubed sphere: one interior cell of panel 3, inflow 1.51 in x, outflow 2.0 in y.
    mesh, m0, am, bm, cm = _cs_flow_problem(:rotation, 0.0)
    foreach(p -> fill!(m0[p], 100.0), 1:6)
    i = j = mesh.Hp + 4
    m0[3][i, j, 1] = 1.0
    am[3][i, j, 1] = 0.755; am[3][i + 1, j, 1] = -0.755
    bm[3][i, j, 1] = -1.0;  bm[3][i, j + 1, 1] = 1.0
    @test Adv._cs_static_palindrome_subcycle_count(am, bm, cm, m0, mesh.Nc, mesh.Hp, 1, cfl) ==
          ceil(Int, 4.0 / (0.02 * cfl))
    # flux_scale enters the end mass too: at half the flux m_end = 0.51.
    @test Adv._cs_static_palindrome_subcycle_count(am, bm, cm, m0, mesh.Nc, mesh.Hp, 1, cfl;
                                                   flux_scale = 0.5) == ceil(Int, 2.0 / (0.51 * cfl))

    # Reduced Gaussian: level 1 of cell l loses 1.0 through a side face and gains
    # 0.6 from level 2 per application, so m_end = 1 − 2 · 0.4 = 0.2.
    FT = Float64
    rg = AT.ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT)
    grid = AT.AtmosGrid(rg, AT.HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1]), AT.CPU(); FT)
    nc, Nz = AT.ncells(rg), 2
    f = findfirst(f -> all(>(0), G.face_cells(rg, f)), 1:G.nfaces(rg))
    l, r = G.face_cells(rg, f)
    fluxes = AT.allocate_face_fluxes(rg, Nz; FT, basis = AT.DryBasis)
    fluxes.horizontal_flux .= 0; fluxes.cm .= 0
    fluxes.horizontal_flux[f, 1] = 1.0       # l → r on level 1
    fluxes.cm[l, 2] = -0.6                    # level 2 of l → level 1 of l (upward)
    m0 = fill(10.0, nc, Nz); m0[l, 1] = 1.0
    n = Adv._face_indexed_palindrome_subcycle_count(fluxes.horizontal_flux, fluxes.cm, m0, rg, FT(cfl))
    @test n == ceil(Int, 2.0 / (0.2 * cfl))
    state = AT.CellState(AT.DryBasis, copy(m0); CO2 = 0.4 .* m0)
    AT.Operators.apply!(state, fluxes, grid, UpwindScheme(), 1.0;
                        workspace = Adv.AdvectionWorkspace(state.air_mass))
    @test minimum(state.air_mass) > 0
    @test state.air_mass[l, 1] ≈ 0.2 rtol = 1e-12
    @test maximum(abs.(state.tracers_raw[:, :, 1] ./ state.air_mass .- 0.4)) < 1e-12
end

@testset "invalid flux sets throw (non-positive reference mass with outflow)" begin
    cfl = AT.Parameters.DEFAULT_CFL_LIMIT
    # Lat-lon: a cell the step would leave with negative mass (m_end = 1 + 2 (1 − 2) = −1) ...
    m0, am, bm, cm = _ll_draining_cell(; O = 2.0, I = 1.0, div_dir = :y)
    @test_throws ArgumentError Adv._structured_palindrome_subcycle_count(am, bm, cm, m0, cfl)
    @test_throws ArgumentError Adv.strang_split_mt!(reshape(copy(m0), 8, 8, 1, 1), copy(m0), am, bm, cm,
                                                    UpwindScheme(), Adv.AdvectionWorkspace(m0; n_tracers = 1))
    # ... a cell with no mass but outflow, and a non-finite flux.
    m0, am, bm, cm = _ll_draining_cell(; O = 0.0, I = 0.0, div_dir = :y)
    am[4, 4, 1] = -0.1; m0[4, 4, 1] = 0.0
    @test_throws ArgumentError Adv._structured_palindrome_subcycle_count(am, bm, cm, m0, cfl)
    am[4, 4, 1] = NaN; m0[4, 4, 1] = 1.0
    @test_throws ArgumentError Adv._structured_palindrome_subcycle_count(am, bm, cm, m0, cfl)
    # A cell without mass and without outflow is fine (it only receives).
    m0, am, bm, cm = _ll_draining_cell(; O = 0.0, I = 0.0, div_dir = :y)
    m0[4, 4, 1] = 0.0; am[4, 4, 1] = 0.1
    @test Adv._structured_palindrome_subcycle_count(am, bm, cm, m0, cfl) == 1

    # Cubed sphere: an empty cell with outflow.
    mesh, m0, am, bm, cm = _cs_flow_problem(:rotation, 0.0)
    i = mesh.Hp + 2
    m0[5][i, i, 1] = 0.0; am[5][i + 1, i, 1] = 1e-3
    @test_throws ArgumentError Adv._cs_static_palindrome_subcycle_count(am, bm, cm, m0, mesh.Nc, mesh.Hp, 1, cfl)

    # Reduced Gaussian: random fluxes that empty cells (the adversarial review's case).
    FT = Float64
    rng = MersenneTwister(5)
    rg = AT.ReducedGaussianMesh(FT[-60, -20, 20, 60], [6, 10, 10, 6]; FT)
    grid = AT.AtmosGrid(rg, AT.HybridSigmaPressure(FT[0, 100, 300, 600], FT[0, 0, 0.3, 1]), AT.CPU(); FT)
    nc, Nz = AT.ncells(rg), 3
    fluxes = AT.allocate_face_fluxes(rg, Nz; FT, basis = AT.DryBasis)
    fluxes.horizontal_flux .= 0; fluxes.cm .= 0
    for f in 1:G.nfaces(rg), k in 1:Nz
        l, r = G.face_cells(rg, f)
        (l > 0 && r > 0) && (fluxes.horizontal_flux[f, k] = 0.5 * randn(rng))
    end
    fluxes.cm[:, 2:Nz] .= 0.5 .* randn(rng, nc, Nz - 1)
    m0 = 1 .+ rand(rng, nc, Nz)
    state = AT.CellState(AT.DryBasis, copy(m0); CO2 = 0.4 .* m0)
    @test_throws ArgumentError AT.Operators.apply!(state, fluxes, grid, UpwindScheme(), 1.0;
                                                   workspace = Adv.AdvectionWorkspace(state.air_mass))
end

# ------------------------------------------------------------- reduced Gaussian
# An overturning circulation between two neighbouring cells of a two-level
# reduced-Gaussian grid: level 1 flows l → r, level 2 r → l, closed by an
# upward flux in l and a downward flux in r. Every cell is non-divergent, but
# one horizontal leg alone removes 2.4 times the cell mass.
@testset "reduced-Gaussian subcycling keeps a uniform tracer uniform (overturning)" begin
    FT = Float64
    mesh = AT.ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT)
    grid = AT.AtmosGrid(mesh, AT.HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1]), AT.CPU(); FT)
    nc, Nz = AT.ncells(mesh), 2
    f = findfirst(f -> all(>(0), G.face_cells(mesh, f)), 1:G.nfaces(mesh))
    l, r = G.face_cells(mesh, f)
    F = 2.4
    fluxes = AT.allocate_face_fluxes(mesh, Nz; FT, basis = AT.DryBasis)
    fluxes.horizontal_flux .= 0
    fluxes.cm .= 0
    fluxes.horizontal_flux[f, 1] = F
    fluxes.horizontal_flux[f, 2] = -F
    fluxes.cm[l, 2] = -F
    fluxes.cm[r, 2] = F
    m0 = ones(FT, nc, Nz)
    n = Adv._face_indexed_palindrome_subcycle_count(fluxes.horizontal_flux, fluxes.cm, m0, mesh,
                                                    FT(AT.Parameters.DEFAULT_CFL_LIMIT))
    @test n == 6
    state = AT.CellState(AT.DryBasis, copy(m0); CO2 = 0.4 .* m0, SF6 = 0.1 .* m0)
    AT.Operators.apply!(state, fluxes, grid, UpwindScheme(), 1.0;
                        workspace = Adv.AdvectionWorkspace(state.air_mass))
    @test maximum(abs.(state.tracers_raw[:, :, 1] ./ state.air_mass .- 0.4)) < 1e-12
    @test maximum(abs.(state.tracers_raw[:, :, 2] ./ state.air_mass .- 0.1)) < 1e-12
    @test state.air_mass ≈ m0 atol = 1e-12
end

# ---------------------------------------------------------------- footprint tape
@testset "footprint tape records [halo] (X Y Z)^n | midpoint | (Z Y X)^n" begin
    mesh, m0, am, bm, cm = _cs_flow_problem(:strain, 0.20)
    I = _cs_interior(mesh)
    q = ntuple(p -> [1 + 0.3 * sin(3i + 2j + p) for i in 1:mesh.Nc+6, j in 1:mesh.Nc+6, k in 1:1], 6)
    rm0 = ntuple(p -> q[p] .* m0[p], 6)
    Adv.fill_panel_halos!(rm0, mesh; dir = 0)
    n = Adv._cs_static_palindrome_subcycle_count(am, bm, cm, m0, mesh.Nc, mesh.Hp, 1,
                                                 AT.Parameters.DEFAULT_CFL_LIMIT)
    @test n > 1
    for scheme in (UpwindScheme(), PPMScheme(MonotoneLimiter()))
        record = scheme isa UpwindScheme ?
            (() -> (AT.Adjoints._record_cs_mass_tape(m0, [am], [bm], [cm], mesh, scheme;
                                                     cfl_limit = AT.Parameters.DEFAULT_CFL_LIMIT,
                                                     flux_scale = 1.0, dt = 1.0)[1], nothing)) :
            (() -> begin
                ops, rm_tape, m_tape = AT.Adjoints._record_cs_tracer_tape(
                    rm0, m0, [am], [bm], [cm], mesh, scheme;
                    cfl_limit = AT.Parameters.DEFAULT_CFL_LIMIT, flux_scale = 1.0, dt = 1.0)
                (ops, (rm_tape, m_tape))
            end)
        ops, final = record()
        kinds = map(ops) do op
            op isa AT.Tape._CSSweepRecord ? op.direction :
            op isa AT.Tape._CSHaloRecord ? Symbol("h", op.dir) :
            op isa AT.Tape._CSMidpointRecord ? :mid : :other
        end
        forward = reduce(vcat, [vcat(pass > 1 ? [:h1] : Symbol[], [:x, :h1, :y, :h2, :z]) for pass in 1:n])
        reverse_half = repeat([:z, :h2, :y, :h2, :h1, :x, :h1], n)
        # The leading X halo record is production's step-start refresh
        # (`fill_cs_step_start_halos!`).
        @test kinds == vcat([:h1], forward, [:mid], reverse_half)

        est = AT.cs_tape_byte_estimate(m0, [am], [bm], [cm], mesh, scheme)
        @test est.sweep_records == count(in((:x, :y, :z)), kinds)
        @test est.halo_records == count(in((:h1, :h2)), kinds)

        if final !== nothing
            # The tape's forward trajectory is the production step.
            rm = map(copy, rm0); m = map(copy, m0)
            Adv.fill_panel_halos!(m, mesh; dir = 0)
            Adv.strang_split_cs!(rm, m, am, bm, cm, mesh, scheme, Adv.CSAdvectionWorkspace(mesh, 1))
            @test all(p -> final[1][p][I, I, :] == rm[p][I, I, :], 1:6)
            @test all(p -> final[2][p][I, I, :] == m[p][I, I, :], 1:6)
        end
    end
end
