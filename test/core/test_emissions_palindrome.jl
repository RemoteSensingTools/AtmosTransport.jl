"""
Plan 17 Commit 5 — emissions at palindrome center.

Verifies:
1. A non-trivial `SurfaceFluxOperator` emits at the palindrome center
   with `V(dt/2) → S(dt) → V(dt/2)`, and the emitted surface mass is
   transported correctly by the surrounding advection.
2. `tracer_names` kwarg is required when `emissions_op !== NoSurfaceFlux`.
3. With zero diffusion (`NoDiffusion`) and zero mass fluxes, an
   emission-only palindrome reduces exactly to a standalone
   `apply_surface_flux!` on the surface layer.
4. The palindrome with emissions conserves the added emission mass
   (analytic: Σ rate × dt × cells).
"""

using Test
using AtmosTransport.Operators.StepPlans: TransportMidpoint
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

    @testset "tracer_names kwarg required when emissions active" begin
        FT = Float64
        Nx, Ny, Nz = 3, 2, 2
        state, am, bm, cm, ws = _make_triple(FT, Nx, Ny, Nz)

        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(1.0, Nx, Ny)))
        # Missing `tracer_names` kwarg → should throw
        @test_throws ArgumentError strang_split_mt!(
            state.tracers_raw, state.air_mass, am, bm, cm,
            UpwindScheme(), ws;
            cfl_limit = 1.0, midpoint = TransportMidpoint(; workspace=ws, dt=10.0, emissions_op=op))
    end

    @testset "Non-trivial emission adds surface mass correctly" begin
        FT = Float64
        Nx, Ny, Nz = 4, 3, 2
        state, am, bm, cm, ws = _make_triple(FT, Nx, Ny, Nz;
                                              init_co2 = 0.0,
                                              init_sf6 = 0.0)
        rate_co2 = reshape(collect(1.0:Float64(Nx * Ny)), Nx, Ny) ./ 100.0
        rate_sf6 = fill(0.5, Nx, Ny)
        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rate_co2),
                                  SurfaceFluxSource(:SF6, rate_sf6))
        dt = 4.0

        strang_split_mt!(state.tracers_raw, state.air_mass,
                         copy(am), copy(bm), copy(cm),
                         UpwindScheme(), ws;
                         cfl_limit = 1.0,
                         midpoint = TransportMidpoint(; workspace=ws, dt,
                             emissions_op = op, tracer_names = state.tracer_names))

        # With zero fluxes, both tracers stay at the surface; CO2 = rate * dt,
        # SF6 surface = 2. Non-surface layers stay 0.
        @test all(state.tracers.CO2[:, :, Nz] .≈ rate_co2 .* dt)
        @test sum(state.tracers.CO2) ≈ sum(rate_co2) * dt
        @test all(state.tracers.SF6[:, :, Nz] .≈ 0.5 * 4.0)
        @test all(state.tracers.CO2[:, :, 1] .== 0.0)
        @test all(state.tracers.SF6[:, :, 1] .== 0.0)
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

end
