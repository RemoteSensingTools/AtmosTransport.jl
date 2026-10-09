"""
NoAdvection no-op scheme — covers all three topologies (LatLon, ReducedGaussian,
CubedSphere) plus the recipe parse and the companion-operator rejection.

Why this exists:
- A "convection-only" production timing run needs to disable advection on a CS
  binary without also disabling diffusion plumbing in a way that silently drops
  Strang-half-stepping.
- NoAdvection must therefore be a *typed* no-op (so dispatch picks it up at
  compile time and bypasses the sweep kernels entirely) and must hard-reject
  non-NoDiffusion / non-NoSurfaceFlux companion operators (because both are
  integrated at the palindrome center of the advection block).
"""

using Test
using AtmosTransport
using AtmosTransport: CellState, CubedSphereState, DryBasis,
                      NoAdvection, UpwindScheme,
                      AdvectionWorkspace, StructuredFaceFluxState,
                      allocate_face_fluxes,
                      NoDiffusion, ImplicitVerticalDiffusion, DiffusionWorkspace,
                      ConstantField,
                      SurfaceFluxSource, SurfaceFluxOperator, NoSurfaceFlux,
                      apply!,
                      LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
                      HybridSigmaPressure, AtmosGrid,
                      reconstruction_order, required_halo_width
using AtmosTransport.Operators.Advection: cs_advection_style, CSSplitSweepStyle
using AtmosTransport.Grids: ncells
using AtmosTransport.Models: build_runtime_advection,
                             CubedSphereRuntimeRecipeStyle,
                             LatLonRuntimeRecipeStyle,
                             ReducedGaussianRuntimeRecipeStyle

const FT = Float64

# ---------------------------------------------------------------------------
# Trait coverage — NoAdvection participates in the standard query surface
# ---------------------------------------------------------------------------

@testset "NoAdvection scheme traits" begin
    s = NoAdvection()
    @test s isa AtmosTransport.AbstractAdvectionScheme
    @test reconstruction_order(s) == -1     # sentinel "not applicable"
    @test required_halo_width(s) == 0       # no horizontal stencil
    @test cs_advection_style(s) === CSSplitSweepStyle()
end

# ---------------------------------------------------------------------------
# LatLon: tracer + air mass are bit-identical after NoAdvection apply!
# ---------------------------------------------------------------------------

@testset "NoAdvection LatLon — apply! is a no-op" begin
    Nx, Ny, Nz = 6, 4, 2
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    air = ones(FT, Nx, Ny, Nz)
    co2 = fill(FT(400e-6), Nx, Ny, Nz) .* air
    co2[1, 1, 1] = FT(800e-6)  # break uniformity so a no-op is observable
    state = CellState(DryBasis, copy(air); CO2=copy(co2))

    # Non-zero fluxes — would normally drift the field; NoAdvection must
    # leave it untouched.
    am = fill(FT(0.01), Nx + 1, Ny, Nz)
    bm = fill(FT(0.02), Nx, Ny + 1, Nz)
    cm = fill(FT(0.001), Nx, Ny, Nz + 1)
    fluxes = StructuredFaceFluxState{DryBasis}(am, bm, cm)
    ws = AdvectionWorkspace(state.air_mass)

    air_before = copy(state.air_mass)
    co2_before = copy(state.tracers.CO2)

    apply!(state, fluxes, grid, NoAdvection(), FT(1800); workspace=ws)

    @test state.air_mass == air_before          # bit-identical
    @test state.tracers.CO2 == co2_before       # bit-identical
end

# ---------------------------------------------------------------------------
# ReducedGaussian (face-indexed): tracer is bit-identical after NoAdvection
# ---------------------------------------------------------------------------

@testset "NoAdvection ReducedGaussian — apply! is a no-op" begin
    Nz = 2
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    m = ones(FT, ncells(mesh), Nz)
    rm = m .* FT(400e-6); rm[1, 1] = FT(800e-6) * m[1, 1]
    state = CellState(DryBasis, copy(m); CO2=copy(rm))
    fluxes = allocate_face_fluxes(mesh, Nz; FT=FT, basis=DryBasis)
    fluxes.horizontal_flux .= FT(0.05)
    fluxes.cm .= FT(0.001)

    air_before = copy(state.air_mass)
    co2_before = copy(state.tracers.CO2)

    apply!(state, fluxes, grid, NoAdvection(), FT(1800))

    @test state.air_mass == air_before
    @test state.tracers.CO2 == co2_before
end

# ---------------------------------------------------------------------------
# CubedSphere: tracer is bit-identical after NoAdvection apply!
# ---------------------------------------------------------------------------

@testset "NoAdvection CubedSphere — apply! is a no-op + workspace is nothing" begin
    Nc, Hp, Nz = 4, 1, 2
    N = Nc + 2Hp
    mesh = CubedSphereMesh(; Nc=Nc, Hp=Hp, FT=FT)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    panels_m  = ntuple(_ -> ones(FT, N, N, Nz), 6)
    panels_rm = ntuple(p -> fill(FT(400e-6), N, N, Nz), 6)
    panels_rm[1][2, 2, 1] = FT(800e-6)  # observable perturbation
    cs_state = CubedSphereState(DryBasis, mesh, panels_m; CO2=panels_rm)
    cs_fluxes = allocate_face_fluxes(mesh, Nz; FT=FT, basis=DryBasis)

    # CS workspace allocator returns nothing for NoAdvection — no scratch
    # buffers needed.
    @test AtmosTransport.Models._cs_advection_workspace_for(
              NoAdvection(), cs_state, grid) === nothing

    air_before = deepcopy(cs_state.air_mass)
    co2_before = deepcopy(cs_state.tracers.CO2)

    apply!(cs_state, cs_fluxes, grid, NoAdvection(), FT(1800); workspace=nothing)

    @test all(cs_state.air_mass[p] == air_before[p] for p in 1:6)
    @test all(cs_state.tracers.CO2[p] == co2_before[p] for p in 1:6)
end

@testset "with_diffusion allocates independent CS diffusion scratch" begin
    Nc, Hp, Nz = 2, 1, 2
    N = Nc + 2Hp
    mesh = CubedSphereMesh(; Nc, Hp, FT)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, AtmosTransport.CPU(); FT)
    panels_m = ntuple(_ -> ones(FT, N, N, Nz), 6)
    panels_rm = ntuple(_ -> fill(FT(400e-6), N, N, Nz), 6)
    state = CubedSphereState(DryBasis, mesh, panels_m; CO2=panels_rm)
    fluxes = allocate_face_fluxes(mesh, Nz; FT, basis=DryBasis)
    model = TransportModel(state, fluxes, grid, NoAdvection())
    @test model.workspace.advection_ws === nothing

    kz = AtmosTransport.State.CubedSphereField(
        ntuple(_ -> ConstantField{FT, 3}(one(FT)), 6))
    with_kz = with_diffusion(model, ImplicitVerticalDiffusion(; kz_field=kz))
    @test with_kz.workspace.advection_ws === nothing
    @test with_kz.workspace.diffusion_ws isa DiffusionWorkspace
    @test size(with_kz.workspace.diffusion_ws.layer_thickness[1]) ==
          (Nc, Nc, Nz)
end

# ---------------------------------------------------------------------------
# Companion-operator rejection — diffusion + emissions both throw cleanly
# ---------------------------------------------------------------------------

@testset "NoAdvection — diffusion runs, surface emissions still rejected" begin
    # NoAdvection + diffusion is a SUPPORTED setup ("diffusion-only"
    # experiment); the V(dt) step is applied directly via the mass-flux
    # VMR wrapper (no Strang palindrome is needed since diffusion is
    # mass-conserving on its own). Surface emissions remain rejected
    # because they're Strang-wrapped around the advection block.
    Nx, Ny, Nz = 6, 4, 2
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    air = ones(FT, Nx, Ny, Nz)
    state = CellState(DryBasis, copy(air); CO2=copy(air) .* FT(400e-6))
    fluxes = StructuredFaceFluxState{DryBasis}(zeros(FT, Nx + 1, Ny, Nz),
                                                zeros(FT, Nx, Ny + 1, Nz),
                                                zeros(FT, Nx, Ny, Nz + 1))
    ws = AdvectionWorkspace(state.air_mass)

    # NoDiffusion + NoSurfaceFlux: still a no-op, no throw.
    @test (apply!(state, fluxes, grid, NoAdvection(), FT(1800);
                  workspace=ws,
                  diffusion_op=NoDiffusion(),
                  emissions_op=NoSurfaceFlux()); true)

    # NoAdvection + diffusion: V(dt) step runs without error and
    # preserves column tracer mass to roundoff (mass-flux wrapper).
    # `layer_thickness` is populated externally in production by
    # `_refresh_dz_for_window!`; here we set it directly to 1 m so the
    # implicit solve has well-defined coefficients.
    kz = ConstantField{FT, 3}(FT(1.0))
    diff_op = ImplicitVerticalDiffusion(; kz_field=kz)
    diffusion_ws = DiffusionWorkspace(state)
    fill!(diffusion_ws.layer_thickness, one(FT))
    air_pre = copy(state.air_mass)
    co2_mass_pre = sum(state.tracers_raw[:, :, :, 1])
    apply!(state, fluxes, grid, NoAdvection(), FT(1800);
           workspace=ws, diffusion_workspace=diffusion_ws,
           diffusion_op=diff_op)
    @test state.air_mass == air_pre              # NoAdvection is air-inert
    @test isapprox(sum(state.tracers_raw[:, :, :, 1]), co2_mass_pre;
                   rtol = 1e-12, atol = 0)

    # Direct-API guard: missing workspace + non-trivial diffusion must
    # surface a clear ArgumentError rather than a cryptic FieldError
    # from the kernel-internal `w_scratch` access.
    @test_throws ArgumentError apply!(state, fluxes, grid, NoAdvection(), FT(1800);
                                      workspace=nothing,
                                      diffusion_workspace=nothing,
                                      diffusion_op=diff_op)

    # Non-NoSurfaceFlux → ArgumentError pointing at the TOML knob.
    em_op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(FT(1.0), Nx, Ny)))
    @test_throws ArgumentError apply!(state, fluxes, grid, NoAdvection(), FT(1800);
                                      workspace=ws, emissions_op=em_op)
    try
        apply!(state, fluxes, grid, NoAdvection(), FT(1800);
               workspace=ws, emissions_op=em_op)
        @test false
    catch err
        @test err isa ArgumentError
        @test occursin("surface_flux", err.msg)
    end
end

# ---------------------------------------------------------------------------
# Companion-op rejection on the face-indexed (RG) and CS entry points —
# the rejection helper is shared, but each topology has its own apply!
# method, so each path must wire the helper in.
# ---------------------------------------------------------------------------

@testset "NoAdvection on RG + CS: diffusion runs, emissions still rejected" begin
    # RG
    rg_mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    rg_vc   = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    rg_grid = AtmosGrid(rg_mesh, rg_vc, AtmosTransport.CPU(); FT=FT)
    rg_m   = ones(FT, ncells(rg_mesh), 2)
    rg_state  = CellState(DryBasis, copy(rg_m); CO2=copy(rg_m) .* FT(400e-6))
    rg_fluxes = allocate_face_fluxes(rg_mesh, 2; FT=FT, basis=DryBasis)
    rg_kz     = ConstantField{FT, 2}(FT(1.0))
    rg_diff   = ImplicitVerticalDiffusion(; kz_field=rg_kz)

    # Without a workspace, the apply! API surfaces a clear ArgumentError
    # rather than a kernel-internal FieldError on `w_scratch`.
    @test_throws ArgumentError apply!(rg_state, rg_fluxes, rg_grid, NoAdvection(),
                                      FT(1800); diffusion_op=rg_diff)

    # With a workspace + populated `layer_thickness`, RG NoAdvection +
    # diffusion runs and preserves column tracer mass to roundoff.
    rg_ws = AdvectionWorkspace(rg_state.air_mass)
    rg_diffusion_ws = DiffusionWorkspace(rg_state)
    fill!(rg_diffusion_ws.layer_thickness, one(FT))
    rg_co2_pre = sum(rg_state.tracers_raw[:, :, 1])
    apply!(rg_state, rg_fluxes, rg_grid, NoAdvection(), FT(1800);
           workspace=rg_ws, diffusion_workspace=rg_diffusion_ws,
           diffusion_op=rg_diff)
    @test isapprox(sum(rg_state.tracers_raw[:, :, 1]), rg_co2_pre;
                   rtol = 1e-12, atol = 0)

    rg_em = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(FT(1.0), ncells(rg_mesh))))
    @test_throws ArgumentError apply!(rg_state, rg_fluxes, rg_grid, NoAdvection(),
                                      FT(1800); emissions_op=rg_em)

    # The reduced-Gaussian palindrome has no diffusive surface-flux boundary:
    # asking for it is an error before the state is touched.
    rg_diff_boundary = ImplicitVerticalDiffusion(; kz_field=rg_kz,
                                                 surface_flux_coupling=AtmosTransport.Operators.Diffusion.DiffusiveSurfaceFluxBoundary())
    # Nonzero fluxes and a nonuniform tracer, so a sweep before the check would show.
    rg_fluxes.horizontal_flux .= FT(1e-3) .* (1:size(rg_fluxes.horizontal_flux, 1))
    rg_state.tracers_raw[:, :, 1] .*= (1:ncells(rg_mesh)) .+ FT(1)
    tracers_before, mass_before = copy(rg_state.tracers_raw), copy(rg_state.air_mass)
    @test_throws ArgumentError apply!(rg_state, rg_fluxes, rg_grid, UpwindScheme(), FT(1800);
                                      workspace=rg_ws, diffusion_workspace=rg_diffusion_ws,
                                      diffusion_op=rg_diff_boundary, emissions_op=rg_em)
    @test rg_state.tracers_raw == tracers_before
    @test rg_state.air_mass == mass_before

    # CS
    cs_Nc, cs_Hp, cs_Nz = 4, 1, 2
    cs_N = cs_Nc + 2cs_Hp
    cs_mesh   = CubedSphereMesh(; Nc=cs_Nc, Hp=cs_Hp, FT=FT)
    cs_vc     = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    cs_grid   = AtmosGrid(cs_mesh, cs_vc, AtmosTransport.CPU(); FT=FT)
    cs_pm     = ntuple(_ -> ones(FT, cs_N, cs_N, cs_Nz), 6)
    cs_prm    = ntuple(_ -> fill(FT(400e-6), cs_N, cs_N, cs_Nz), 6)
    cs_state  = CubedSphereState(DryBasis, cs_mesh, cs_pm; CO2=cs_prm)
    cs_fluxes = allocate_face_fluxes(cs_mesh, cs_Nz; FT=FT, basis=DryBasis)
    cs_kz_panels = ntuple(_ -> ConstantField{FT, 3}(FT(1.0)), 6)
    cs_kz     = AtmosTransport.State.CubedSphereField(cs_kz_panels)
    cs_diff   = ImplicitVerticalDiffusion(; kz_field=cs_kz)

    # CS workspace guard: same actionable ArgumentError as LL/RG.
    @test_throws ArgumentError apply!(cs_state, cs_fluxes, cs_grid, NoAdvection(),
                                      FT(1800); diffusion_op=cs_diff)

    cs_em = SurfaceFluxOperator(SurfaceFluxSource(:CO2,
                                                  ntuple(_ -> fill(FT(1.0), cs_N, cs_N), 6)))
    @test_throws ArgumentError apply!(cs_state, cs_fluxes, cs_grid, NoAdvection(),
                                      FT(1800); emissions_op=cs_em)
end

# ---------------------------------------------------------------------------
# Recipe parse — `scheme = "none"` builds a NoAdvection on every topology
# ---------------------------------------------------------------------------

@testset "Recipe parses [advection] scheme = \"none\" to NoAdvection" begin
    cfg = Dict("advection" => Dict("scheme" => "none"))
    @test build_runtime_advection(cfg, LatLonRuntimeRecipeStyle()) isa NoAdvection
    @test build_runtime_advection(cfg, ReducedGaussianRuntimeRecipeStyle()) isa NoAdvection
    @test build_runtime_advection(cfg, CubedSphereRuntimeRecipeStyle()) isa NoAdvection
end
