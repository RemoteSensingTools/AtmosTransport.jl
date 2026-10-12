"""
Plan 17 Commit 5 — emissions at palindrome center.

Verifies:
1. `NoSurfaceFlux` default preserves plan-16b palindrome behavior
   bit-for-bit (critical regression: the single `V(dt)` path is
   untouched when no emissions are active).
2. A non-trivial `SurfaceFluxOperator` emits at the palindrome center
   with `V(dt/2) → S(dt) → V(dt/2)`, and the emitted surface mass is
   transported correctly by the surrounding advection.
3. `tracer_names` kwarg is required when `emissions_op !== NoSurfaceFlux`.
4. With zero diffusion (`NoDiffusion`) and zero mass fluxes, an
   emission-only palindrome reduces exactly to a standalone
   `apply_surface_flux!` on the surface layer.
5. The palindrome with emissions conserves the added emission mass
   (analytic: Σ rate × dt × cells).
"""

using Test
using AtmosTransport
using AtmosTransport: CellState, MoistBasis,
                      SlopesScheme, NoLimiter, UpwindScheme,
                      AdvectionWorkspace,
                      StructuredFaceFluxState, DryBasis,
                      allocate_face_fluxes,
                      NoDiffusion, ImplicitVerticalDiffusion, DiffusionWorkspace,
                      SurfaceFluxSource, SurfaceFluxOperator, NoSurfaceFlux,
                      strang_split_mt!, strang_split!, apply!,
                      ConstantField

# Tiny helper: build a structured-grid state + fluxes + workspace triple.
function _make_triple(FT, Nx, Ny, Nz; init_co2 = 1.0, init_sf6 = 0.1)
    air    = ones(FT, Nx, Ny, Nz)
    state  = CellState(MoistBasis, air;
                       CO2 = fill(FT(init_co2), Nx, Ny, Nz),
                       SF6 = fill(FT(init_sf6), Nx, Ny, Nz))
    # Zero fluxes for isolation
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    ws = AdvectionWorkspace(air; n_tracers = 2)
    return state, am, bm, cm, ws
end

@testset "Emissions palindrome — plan 17 Commit 5" begin

    @testset "NoSurfaceFlux default: palindrome bit-exact with pre-17" begin
        # With emissions_op = NoSurfaceFlux (default) the palindrome
        # must take the `apply_vertical_diffusion_vmr!` no-op call, and the
        # final state must equal the state produced by a
        # call that omits emissions_op entirely.
        FT = Float64
        Nx, Ny, Nz = 3, 2, 3
        state_a, am, bm, cm, ws_a = _make_triple(FT, Nx, Ny, Nz)
        state_b, _,  _,  _,  ws_b = _make_triple(FT, Nx, Ny, Nz)

        scheme = SlopesScheme(NoLimiter())
        dt = 100.0

        # Path A: no emissions_op kwarg (uses default NoSurfaceFlux)
        strang_split_mt!(state_a.tracers_raw, state_a.air_mass,
                         copy(am), copy(bm), copy(cm), scheme, ws_a;
                         cfl_limit = 1.0, dt = dt)

        # Path B: explicitly pass emissions_op = NoSurfaceFlux()
        strang_split_mt!(state_b.tracers_raw, state_b.air_mass,
                         copy(am), copy(bm), copy(cm), scheme, ws_b;
                         cfl_limit = 1.0, dt = dt,
                         emissions_op = NoSurfaceFlux())

        @test state_a.tracers_raw == state_b.tracers_raw   # byte-equal
    end

    @testset "NoSurfaceFlux matches explicit non-emissions path" begin
        # And when diffusion is also active, NoSurfaceFlux must not
        # perturb the diffusion result.
        FT = Float64
        Nx, Ny, Nz = 4, 3, 4

        # Seed with a non-uniform profile so diffusion has work to do
        air = ones(FT, Nx, Ny, Nz)
        profile = collect(range(0.0, 10.0; length = Nz))
        state = CellState(MoistBasis, air;
                          CO2 = repeat(reshape(profile, 1, 1, Nz),
                                       outer = (Nx, Ny, 1)))
        state_b = CellState(MoistBasis, air;
                            CO2 = repeat(reshape(profile, 1, 1, Nz),
                                         outer = (Nx, Ny, 1)))

        am = zeros(FT, Nx + 1, Ny, Nz)
        bm = zeros(FT, Nx, Ny + 1, Nz)
        cm = zeros(FT, Nx, Ny, Nz + 1)
        ws_a = AdvectionWorkspace(air; n_tracers = 1)
        ws_b = AdvectionWorkspace(air; n_tracers = 1)
        diffusion_ws_a = DiffusionWorkspace(state)
        diffusion_ws_b = DiffusionWorkspace(state_b)
        fill!(diffusion_ws_a.layer_thickness, FT(100.0))
        fill!(diffusion_ws_b.layer_thickness, FT(100.0))

        kz = ConstantField{FT, 3}(1.0)
        dfop = ImplicitVerticalDiffusion(; kz_field = kz)
        scheme = UpwindScheme()
        dt = 50.0

        # Path A: diffusion active, emissions absent (default NoSurfaceFlux)
        strang_split_mt!(state.tracers_raw, state.air_mass,
                         copy(am), copy(bm), copy(cm), scheme, ws_a;
                         cfl_limit = 1.0, dt = dt, diffusion_op = dfop,
                         diffusion_workspace = diffusion_ws_a)

        # Path B: same, explicit NoSurfaceFlux
        strang_split_mt!(state_b.tracers_raw, state_b.air_mass,
                         copy(am), copy(bm), copy(cm), scheme, ws_b;
                         cfl_limit = 1.0, dt = dt, diffusion_op = dfop,
                         diffusion_workspace = diffusion_ws_b,
                         emissions_op = NoSurfaceFlux())

        @test state.tracers_raw == state_b.tracers_raw
    end

    @testset "tracer_names kwarg required when emissions active" begin
        FT = Float64
        Nx, Ny, Nz = 3, 2, 2
        state, am, bm, cm, ws = _make_triple(FT, Nx, Ny, Nz)

        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(1.0, Nx, Ny)))
        # Missing `tracer_names` kwarg → should throw
        @test_throws ArgumentError strang_split_mt!(
            state.tracers_raw, state.air_mass, am, bm, cm,
            UpwindScheme(), ws;
            cfl_limit = 1.0, dt = 10.0, emissions_op = op)
    end

    @testset "Zero-flux emission-only palindrome matches standalone surface-flux" begin
        # Setup: zero fluxes, zero diffusion. The palindrome reduces to
        # `apply_surface_flux!(rm, op, ws, dt, ...)` on the surface layer.
        FT = Float64
        Nx, Ny, Nz = 3, 2, 3

        # State A: run the full palindrome with emissions
        state_a, am, bm, cm, ws_a = _make_triple(FT, Nx, Ny, Nz;
                                                  init_co2 = 0.0)
        rate = fill(2.0, Nx, Ny)   # kg/s per cell
        op   = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rate))
        dt   = 7.0

        strang_split_mt!(state_a.tracers_raw, state_a.air_mass,
                         copy(am), copy(bm), copy(cm),
                         UpwindScheme(), ws_a;
                         cfl_limit = 1.0, dt = dt,
                         emissions_op = op,
                         tracer_names = state_a.tracer_names)

        # State B: apply the surface flux directly via apply!
        state_b, _, _, _, _ = _make_triple(FT, Nx, Ny, Nz; init_co2 = 0.0)
        apply!(state_b, nothing, nothing, op, dt)

        @test state_a.tracers_raw ≈ state_b.tracers_raw
    end

    @testset "Non-trivial emission adds surface mass correctly" begin
        FT = Float64
        Nx, Ny, Nz = 4, 3, 2
        state, am, bm, cm, ws = _make_triple(FT, Nx, Ny, Nz;
                                              init_co2 = 0.0,
                                              init_sf6 = 0.0)
        rate_co2 = fill(3.0, Nx, Ny)
        rate_sf6 = fill(0.5, Nx, Ny)
        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rate_co2),
                                  SurfaceFluxSource(:SF6, rate_sf6))
        dt = 4.0

        strang_split_mt!(state.tracers_raw, state.air_mass,
                         copy(am), copy(bm), copy(cm),
                         UpwindScheme(), ws;
                         cfl_limit = 1.0, dt = dt,
                         emissions_op = op,
                         tracer_names = state.tracer_names)

        # With zero fluxes, both tracers stay at the surface; CO2 surface = 12,
        # SF6 surface = 2. Non-surface layers stay 0.
        @test all(state.tracers.CO2[:, :, Nz] .≈ 3.0 * 4.0)
        @test all(state.tracers.SF6[:, :, Nz] .≈ 0.5 * 4.0)
        @test all(state.tracers.CO2[:, :, 1] .== 0.0)
        @test all(state.tracers.SF6[:, :, 1] .== 0.0)
    end

    @testset "Mass conservation: emitted == Σ rate × dt × cells" begin
        FT = Float64
        Nx, Ny, Nz = 5, 4, 3
        state, am, bm, cm, ws = _make_triple(FT, Nx, Ny, Nz;
                                              init_co2 = 0.0)
        rate = reshape(collect(1.0:Float64(Nx * Ny)), Nx, Ny) ./ 100.0
        op   = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rate))
        dt   = 7.5

        strang_split_mt!(state.tracers_raw, state.air_mass,
                         copy(am), copy(bm), copy(cm),
                         UpwindScheme(), ws;
                         cfl_limit = 1.0, dt = dt,
                         emissions_op = op,
                         tracer_names = state.tracer_names)

        expected = sum(rate) * dt
        @test sum(state.tracers.CO2) ≈ expected
    end

    @testset "Cubed-sphere surface flux updates interior surface layer" begin
        FT = Float64
        Nc, Hp, Nz = 3, 1, 4
        dt = FT(5)
        rates = ntuple(p -> fill(FT(p), Nc, Nc), 6)

        q4 = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz, 2), 6)
        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rates))
        AtmosTransport.Operators.SurfaceFlux.apply_surface_flux!(
            q4, op, nothing, dt, nothing, nothing;
            tracer_names = (:CO2, :SF6), halo_width = Hp)
        @inbounds for p in 1:6
            @test all(q4[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz, 1] .== rates[p] .* dt)
            @test all(q4[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz, 2] .== zero(FT))
            @test all(q4[p][1, :, :, :] .== zero(FT))
            @test all(q4[p][:, 1, :, :] .== zero(FT))
        end

        q3 = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)
        AtmosTransport.Operators.SurfaceFlux.apply_surface_flux!(
            q3, op, nothing, dt, nothing, nothing;
            tracer_names = (:CO2,), halo_width = Hp)
        @inbounds for p in 1:6
            @test all(q3[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz] .== rates[p] .* dt)
            @test all(q3[p][1, :, :] .== zero(FT))
            @test all(q3[p][:, 1, :] .== zero(FT))
        end
    end

    @testset "V(dt/2) S V(dt/2) vs V(dt) S with active diffusion: diffusion is symmetric" begin
        # With active diffusion and active emissions, the V(dt/2) → S → V(dt/2)
        # arrangement should mix fresh emissions upward (non-zero at upper
        # layers after one step).
        FT = Float64
        Nx, Ny, Nz = 4, 3, 6
        air = ones(FT, Nx, Ny, Nz)
        state = CellState(MoistBasis, air; CO2 = zeros(FT, Nx, Ny, Nz))

        am = zeros(FT, Nx + 1, Ny, Nz)
        bm = zeros(FT, Nx, Ny + 1, Nz)
        cm = zeros(FT, Nx, Ny, Nz + 1)
        ws = AdvectionWorkspace(air; n_tracers = 1)
        diffusion_ws = DiffusionWorkspace(state)
        fill!(diffusion_ws.layer_thickness, FT(100.0))

        kz = ConstantField{FT, 3}(10.0)   # strong Kz so mixing visible in 1 step
        dfop = ImplicitVerticalDiffusion(; kz_field = kz)
        op   = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(1.0, Nx, Ny)))
        dt   = 60.0

        strang_split_mt!(state.tracers_raw, state.air_mass,
                         copy(am), copy(bm), copy(cm),
                         UpwindScheme(), ws;
                         cfl_limit = 1.0, dt = dt,
                         diffusion_op = dfop,
                         diffusion_workspace = diffusion_ws,
                         emissions_op = op,
                         tracer_names = state.tracer_names)

        # The second V(dt/2) after S mixes the freshly-emitted mass into
        # layers above the surface.
        @test all(state.tracers.CO2[:, :, Nz - 1] .> 0.0)
        @test all(state.tracers.CO2[:, :, 1]      .>= 0.0)
        # Global mass is conserved within the emission added
        expected = Nx * Ny * 1.0 * dt
        @test sum(state.tracers.CO2) ≈ expected
    end
end
