# Per-cell, per-panel interpolation through the packed production operator.
# Scheme integrals and knot crossings live in test_surface_flux_integrals.jl;
# host/device adaptation lives in test_surface_flux_window_buffer.jl.
using Test, AtmosTransport
using AtmosTransport: apply_surface_flux!, flux_temporal_scheme
using AtmosTransport.Operators.SurfaceFlux: TimeVaryingSurfaceFluxSource

struct StubClock
    t::Float64
end
AtmosTransport.MetDrivers.current_time(c::StubClock) = c.t

@testset "TimeVarying surface flux — constructor contracts" begin
    series = ntuple(_ -> zeros(2, 2, 2), 6)
    @test_throws ArgumentError TimeVaryingSurfaceFluxSource(:CO2, series, Float64[])
    @test_throws ArgumentError TimeVaryingSurfaceFluxSource(:CO2, series, [0.0, 0.0])
    @test_throws ArgumentError TimeVaryingSurfaceFluxSource(:CO2, series, [0.0, Inf])
    @test_throws DimensionMismatch TimeVaryingSurfaceFluxSource(:CO2, series, [0.0])
end

@testset "TimeVarying surface flux — CS apply + interpolation" begin
    for (name, T) in (("stepwise", StepwiseFlux), ("block", StepwiseFlux),
                      ("linear", LinearInterpFlux), ("conservative", ConservativeMeanFlux))
        @test flux_temporal_scheme(name) isa T
    end
    @test_throws ArgumentError flux_temporal_scheme("bogus")

    Nc, Hp, Nz = 4, 1, 2
    N = Nc + 2Hp
    mesh = CubedSphereMesh(; Nc, Hp)
    vertical = HybridSigmaPressure([0.0, 100.0, 300.0], [0.0, 0.0, 1.0])
    grid = AtmosGrid(mesh, vertical, CPU())
    state = CubedSphereState(DryBasis, mesh, ntuple(_ -> ones(N, N, Nz), 6);
                             CO2 = ntuple(_ -> zeros(N, N, Nz), 6))
    times = [0.0, 3600.0, 7200.0]
    # Fingerprints vary by cell as well as panel, exposing transpositions.
    series = ntuple(p -> [10.0p + i + 0.1j + k - 1
                         for i in 1:Nc, j in 1:Nc, k in 1:3], 6)
    dt = 100.0
    for (time, offset) in ((0.0, 0.0), (3600.0, 1.0), (7200.0, 2.0),
                            (1800.0, 0.5), (4500.0, 1.25), (-500.0, 0.0), (1e6, 2.0))
        foreach(q -> fill!(q, 0), state.tracers.CO2)
        op = SurfaceFluxOperator(PerTracerFluxMap(
            TimeVaryingSurfaceFluxSource(:CO2, series, times, LinearInterpFlux())))
        apply_surface_flux!(state.tracers_raw, op, nothing, dt, StubClock(time), grid;
                            tracer_names=state.tracer_names, halo_width=Hp)
        for p in 1:6
            panel = get_tracer(state, :CO2)[p]
            expected = [(10.0p + i + 0.1j + offset) * dt for i in 1:Nc, j in 1:Nc]
            @test panel[2:5, 2:5, Nz] ≈ expected
            @test all(iszero, panel[:, :, 1])
            @test all(iszero, panel[[1,end], :, :])
            @test all(iszero, panel[:, [1,end], :])
        end
    end
end
