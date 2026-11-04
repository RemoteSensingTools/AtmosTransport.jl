"""
NoAdvection no-op scheme — covers all three topologies (LatLon, ReducedGaussian,
CubedSphere) plus the recipe parse and the companion-operator rejection.

Why this exists:
- A "convection-only" production timing run needs to disable advection on a CS
  binary without also disabling diffusion plumbing in a way that silently drops
  Strang-half-stepping.
- NoAdvection must therefore be a *typed* no-op (`transport_step!` dispatches
  on it at compile time and bypasses the sweep kernels entirely) and must
  reject surface fluxes wherever they sit inside the advection palindrome.
  StepPlan composes standalone diffusion when advection is disabled, and on the
  cubed sphere with DiffusiveSurfaceFluxBoundary the S(dt) -> V(dt) center.
"""

using Test
using AtmosTransport.Operators.StepPlans: transport_step!
using AtmosTransport
using AtmosTransport: CellState, CubedSphereState, DryBasis,
                      NoAdvection, UpwindScheme, PPMScheme, LinRoodPPMScheme,
                      AdvectionWorkspace, StructuredFaceFluxState,
                      allocate_face_fluxes,
                      NoDiffusion, ImplicitVerticalDiffusion, DiffusionWorkspace,
                      ConstantField,
                      SurfaceFluxSource, SurfaceFluxOperator, NoSurfaceFlux,
                      LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
                      HybridSigmaPressure, AtmosGrid,
                      required_halo_width
using AtmosTransport.Operators.Advection: cs_advection_style, CSSplitSweepStyle
using AtmosTransport.Grids: ncells

const FT = Float64

# ---------------------------------------------------------------------------
# Trait coverage — NoAdvection participates in the standard query surface
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LatLon: tracer + air mass are bit-identical after a NoAdvection step
# ---------------------------------------------------------------------------

@testset "NoAdvection LatLon — transport_step! is a no-op" begin
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

    transport_step!(state, fluxes, grid, NoAdvection(), FT(1800); workspace=ws)

    @test state.air_mass == air_before          # bit-identical
    @test state.tracers.CO2 == co2_before       # bit-identical
end

# ---------------------------------------------------------------------------
# ReducedGaussian (face-indexed): tracer is bit-identical after NoAdvection
# ---------------------------------------------------------------------------

@testset "NoAdvection ReducedGaussian — transport_step! is a no-op" begin
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

    transport_step!(state, fluxes, grid, NoAdvection(), FT(1800))

    @test state.air_mass == air_before
    @test state.tracers.CO2 == co2_before
end

# ---------------------------------------------------------------------------
# CubedSphere: tracer is bit-identical after a NoAdvection step; the cubed-
# sphere adjoint tape has no NoAdvection record, so a recorder is rejected.
# ---------------------------------------------------------------------------

struct _TapeRecorderStandIn <: AtmosTransport.Operators.AbstractStepRecorder end

@testset "NoAdvection CubedSphere — transport_step! is a no-op + workspace is nothing" begin
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
    @test AtmosTransport.Operators.StepPlans._cs_advection_workspace_for(
              NoAdvection(), cs_state, grid) === nothing

    air_before = deepcopy(cs_state.air_mass)
    co2_before = deepcopy(cs_state.tracers.CO2)

    transport_step!(cs_state, cs_fluxes, grid, NoAdvection(), FT(1800); workspace=nothing)

    @test all(cs_state.air_mass[p] == air_before[p] for p in 1:6)
    @test all(cs_state.tracers.CO2[p] == co2_before[p] for p in 1:6)

    @test_throws ArgumentError transport_step!(cs_state, cs_fluxes, grid, NoAdvection(),
                                               FT(1800); workspace=nothing,
                                               recorder=_TapeRecorderStandIn())
    @test all(cs_state.tracers.CO2[p] == co2_before[p] for p in 1:6)
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
    @test (transport_step!(state, fluxes, grid, NoAdvection(), FT(1800);
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
    transport_step!(state, fluxes, grid, NoAdvection(), FT(1800);
           workspace=ws, diffusion_workspace=diffusion_ws,
           diffusion_op=diff_op)
    @test state.air_mass == air_pre              # NoAdvection is air-inert
    @test isapprox(sum(state.tracers_raw[:, :, :, 1]), co2_mass_pre;
                   rtol = 1e-12, atol = 0)

    # Direct-API guard: missing workspace + non-trivial diffusion must
    # surface a clear ArgumentError rather than a cryptic FieldError
    # from the kernel-internal `w_scratch` access.
    @test_throws ArgumentError transport_step!(state, fluxes, grid, NoAdvection(), FT(1800);
                                      workspace=nothing,
                                      diffusion_workspace=nothing,
                                      diffusion_op=diff_op)

    # Non-NoSurfaceFlux → ArgumentError pointing at the TOML knob.
    em_op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(FT(1.0), Nx, Ny)))
    @test_throws ArgumentError transport_step!(state, fluxes, grid, NoAdvection(), FT(1800);
                                      workspace=ws, emissions_op=em_op)
    try
        transport_step!(state, fluxes, grid, NoAdvection(), FT(1800);
               workspace=ws, emissions_op=em_op)
        @test false
    catch err
        @test err isa ArgumentError
        @test occursin("surface_flux", err.msg)
    end
end

# ---------------------------------------------------------------------------
# Companion-op rejection on the face-indexed (RG) and CS states — the
# NoAdvection `transport_step!` is shared, so check each topology's
# preflight and diffusion dispatch.
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

    # Without a workspace, transport_step! surfaces a clear ArgumentError
    # rather than a kernel-internal FieldError on `w_scratch`.
    @test_throws ArgumentError transport_step!(rg_state, rg_fluxes, rg_grid, NoAdvection(),
                                      FT(1800); diffusion_op=rg_diff)

    # With a workspace + populated `layer_thickness`, RG NoAdvection +
    # diffusion runs and preserves column tracer mass to roundoff.
    rg_ws = AdvectionWorkspace(rg_state.air_mass)
    rg_diffusion_ws = DiffusionWorkspace(rg_state)
    fill!(rg_diffusion_ws.layer_thickness, one(FT))
    rg_co2_pre = sum(rg_state.tracers_raw[:, :, 1])
    transport_step!(rg_state, rg_fluxes, rg_grid, NoAdvection(), FT(1800);
           workspace=rg_ws, diffusion_workspace=rg_diffusion_ws,
           diffusion_op=rg_diff)
    @test isapprox(sum(rg_state.tracers_raw[:, :, 1]), rg_co2_pre;
                   rtol = 1e-12, atol = 0)

    rg_em = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(FT(1.0), ncells(rg_mesh))))
    @test_throws ArgumentError transport_step!(rg_state, rg_fluxes, rg_grid, NoAdvection(),
                                      FT(1800); emissions_op=rg_em)

    # The reduced-Gaussian palindrome has no diffusive surface-flux boundary:
    # asking for it is an error before the state is touched.
    rg_diff_boundary = ImplicitVerticalDiffusion(; kz_field=rg_kz,
                                                 surface_flux_coupling=AtmosTransport.Operators.Diffusion.DiffusiveSurfaceFluxBoundary())
    # Nonzero fluxes and a nonuniform tracer, so a sweep before the check would show.
    rg_fluxes.horizontal_flux .= FT(1e-3) .* (1:size(rg_fluxes.horizontal_flux, 1))
    rg_state.tracers_raw[:, :, 1] .*= (1:ncells(rg_mesh)) .+ FT(1)
    tracers_before, mass_before = copy(rg_state.tracers_raw), copy(rg_state.air_mass)
    @test_throws ArgumentError transport_step!(rg_state, rg_fluxes, rg_grid, UpwindScheme(), FT(1800);
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
    @test_throws ArgumentError transport_step!(cs_state, cs_fluxes, cs_grid, NoAdvection(),
                                      FT(1800); diffusion_op=cs_diff)

    cs_em = SurfaceFluxOperator(SurfaceFluxSource(:CO2,
                                                  ntuple(_ -> fill(FT(1.0), cs_N, cs_N), 6)))
    @test_throws ArgumentError transport_step!(cs_state, cs_fluxes, cs_grid, NoAdvection(),
                                      FT(1800); emissions_op=cs_em)
end

# ---------------------------------------------------------------------------
# Emissions with advection off: supported only where the palindrome center has
# no half-steps (cubed sphere + DiffusiveSurfaceFluxBoundary, S(dt) -> V(dt))
# ---------------------------------------------------------------------------

const _SFB = AtmosTransport.Operators.Diffusion

# A C4 cube with three tracers (two emitted) and four layers. `kz = :gchp` uses
# GEOS-Chem's non-local field with a fixed exchange and an emission profile that
# spreads the emission over three layers (`ProfileDeposit`); `kz = :constant` a
# constant Kz (surface-layer deposit). Air mass varies by column and layer.
function _noadv_cs_case(; kz = :gchp, coupling = _SFB.DiffusiveSurfaceFluxBoundary(),
                        dkg_value = 5e3)
    Nc, Hp, Nz = 4, 3, 4
    N = Nc + 2Hp
    mesh = CubedSphereMesh(; Nc, Hp, FT)
    grid = AtmosGrid(mesh, HybridSigmaPressure(FT[0, 100, 200, 300, 400], FT[0, 0, 0, 0, 1]),
                     AtmosTransport.CPU(); FT)
    air = ntuple(p -> [FT(1e4) * (1 + 0.1p + 0.01i + 0.02j + 0.3k) for i in 1:N, j in 1:N, k in 1:Nz], 6)
    vmr(p, base) = [base * (1 + 0.05sin(i + 2j + 3k + p)) for i in 1:N, j in 1:N, k in 1:Nz]
    new_state() = CubedSphereState(DryBasis, mesh, map(copy, air);
                                   CO2 = ntuple(p -> air[p] .* vmr(p, FT(400e-6)), 6),
                                   SF6 = ntuple(p -> air[p] .* vmr(p, FT(1e-11)), 6),
                                   RN  = ntuple(p -> air[p] .* vmr(p, FT(1e-20)), 6))
    if kz === :gchp
        field = AtmosTransport.State.GCHPNonlocalPBLField(Nc, Nz, FT)
        for p in 1:6
            field.dkg.panels[p].data[:, :, 1:(Nz - 1)] .= FT(dkg_value)   # zero below the surface layer
            field.emission_profile[p] .= reshape(FT[0, 0.2, 0.3, 0.5], 1, 1, Nz)
        end
    else
        field = AtmosTransport.State.CubedSphereField(ntuple(_ -> ConstantField{FT, 3}(FT(1.0)), 6))
    end
    diff = ImplicitVerticalDiffusion(; kz_field = field, surface_flux_coupling = coupling)
    rates(scale) = ntuple(p -> [FT(scale * (p + 0.1i + 0.01j)) for i in 1:Nc, j in 1:Nc], 6)
    em = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rates(1e-6)),
                             SurfaceFluxSource(:RN, rates(1e-18)))
    function diffusion_ws(state)
        ws = DiffusionWorkspace(state)
        foreach(p -> fill!(ws.layer_thickness[p], one(FT)), 1:6)
        return ws
    end
    return (; Nc, Hp, Nz, mesh, grid, new_state, diff, em, rates, diffusion_ws,
            fluxes = allocate_face_fluxes(mesh, Nz; FT, basis = DryBasis))
end

_interior(c, x, p) = x[p][(c.Hp + 1):(c.Hp + c.Nc), (c.Hp + 1):(c.Hp + c.Nc), ntuple(_ -> :, ndims(x[p]) - 2)...]
# Exact (BigFloat) sums, so the budget checks see the step, not the summation.
_tracer_total(c, state, t) = sum(sum(big, _interior(c, state.tracers_raw, p)[:, :, :, t]) for p in 1:6)

function _noadv_step!(c, scheme; kz_dt = FT(1800), kwargs...)
    state = c.new_state()
    ws = AtmosTransport.Operators.StepPlans._cs_advection_workspace_for(scheme, state, c.grid)
    transport_step!(state, c.fluxes, c.grid, scheme, kz_dt; workspace = ws,
                    diffusion_workspace = c.diffusion_ws(state), diffusion_op = c.diff,
                    emissions_op = c.em, kwargs...)
    return state
end

@testset "NoAdvection CS + DiffusiveSurfaceFluxBoundary: emissions run S(dt) -> V(dt)" begin
    for kz in (:gchp, :constant)
        c = _noadv_cs_case(; kz)
        dt = FT(1800)
        s0 = c.new_state()
        s_none = _noadv_step!(c, NoAdvection())

        # Exactly the emitted mass is added to each emitted tracer; the other
        # tracer and the air mass are untouched.
        emitted(scale) = sum(sum(r) for r in c.rates(scale)) * dt
        @test isapprox(_tracer_total(c, s_none, 1) - _tracer_total(c, s0, 1), emitted(1e-6); rtol = 1e-13)
        @test isapprox(_tracer_total(c, s_none, 3) - _tracer_total(c, s0, 3), emitted(1e-18); rtol = 1e-13)
        @test isapprox(_tracer_total(c, s_none, 2), _tracer_total(c, s0, 2); rtol = 1e-14)
        @test all(s_none.air_mass[p] == s0.air_mass[p] for p in 1:6)

        # Same composition as the advection palindrome center: equal to an
        # advection step with zero mass fluxes, through the packed midpoint
        # (Upwind, PPM) and the single-tracer-panel midpoint (Lin–Rood).
        for scheme in (UpwindScheme(), PPMScheme(), LinRoodPPMScheme(5))
            s_adv = _noadv_step!(c, scheme)
            @test all(_interior(c, s_adv.air_mass, p) ≈ _interior(c, s0.air_mass, p) for p in 1:6)
            for t in 1:3, p in 1:6
                a = _interior(c, s_none.tracers_raw, p)[:, :, :, t]
                b = _interior(c, s_adv.tracers_raw, p)[:, :, :, t]
                @test maximum(abs.(a .- b)) <= 1e-13 * maximum(abs.(b))
            end
        end
    end

    # The deposit is the scheme's: with no exchange (dkg = 0) the emission stays
    # where GEOS-Chem's non-local profile puts it, 0 / 0.2 / 0.3 / 0.5 top-down.
    c = _noadv_cs_case(; kz = :gchp, dkg_value = 0)
    dt = FT(1800)
    s0, s_none = c.new_state(), _noadv_step!(c, NoAdvection())
    rate = c.rates(1e-6)
    for p in 1:6
        added = _interior(c, s_none.tracers_raw, p)[:, :, :, 1] .- _interior(c, s0.tracers_raw, p)[:, :, :, 1]
        expected = rate[p] .* dt .* reshape(FT[0, 0.2, 0.3, 0.5], 1, 1, c.Nz)
        @test isapprox(added, expected; rtol = 1e-9)
    end
end

@testset "NoAdvection emissions: every other coupling and topology is rejected" begin
    # Cubed sphere with the split coupling, or with no diffusion: the emission sits
    # between two diffusion half-steps, so the step throws before touching the state.
    for diff in (_noadv_cs_case(; coupling = _SFB.SplitSurfaceFluxCoupling()).diff, NoDiffusion())
        c = _noadv_cs_case()
        state = c.new_state()
        before = deepcopy(state.tracers_raw)
        err = try
            transport_step!(state, c.fluxes, c.grid, NoAdvection(), FT(1800);
                            diffusion_workspace = c.diffusion_ws(state), diffusion_op = diff,
                            emissions_op = c.em)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("DiffusiveSurfaceFluxBoundary", err.msg)
        @test occursin("SplitSurfaceFluxCoupling", err.msg)
        @test state.tracers_raw == before
    end

    # Lat-lon keeps the rejection even with DiffusiveSurfaceFluxBoundary.
    Nx, Ny, Nz = 6, 4, 2
    mesh = LatLonMesh(; Nx, Ny, FT)
    grid = AtmosGrid(mesh, HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1]), AtmosTransport.CPU(); FT)
    air = ones(FT, Nx, Ny, Nz)
    state = CellState(DryBasis, copy(air); CO2 = air .* FT(400e-6))
    fluxes = StructuredFaceFluxState{DryBasis}(zeros(FT, Nx + 1, Ny, Nz), zeros(FT, Nx, Ny + 1, Nz),
                                               zeros(FT, Nx, Ny, Nz + 1))
    diff = ImplicitVerticalDiffusion(; kz_field = ConstantField{FT, 3}(FT(1.0)),
                                     surface_flux_coupling = _SFB.DiffusiveSurfaceFluxBoundary())
    dws = DiffusionWorkspace(state)
    fill!(dws.layer_thickness, one(FT))
    em = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(FT(1.0), Nx, Ny)))
    before = copy(state.tracers_raw)
    err = try
        transport_step!(state, fluxes, grid, NoAdvection(), FT(1800);
                        workspace = AdvectionWorkspace(state.air_mass), diffusion_workspace = dws,
                        diffusion_op = diff, emissions_op = em)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("lat-lon or reduced-Gaussian", err.msg)
    @test state.tracers_raw == before
end

# ---------------------------------------------------------------------------
# Recipe parse — `scheme = "none"` builds a NoAdvection on every topology
# ---------------------------------------------------------------------------
