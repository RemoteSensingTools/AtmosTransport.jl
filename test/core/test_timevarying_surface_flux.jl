"""
Time-varying (3-hourly) cubed-sphere surface-flux source.

Covers the opt-in `TimeVaryingSurfaceFluxSource` path added so the CAMS/LMDZ
natural-CO2 flux advances through its sub-monthly slices with the simulation
clock (instead of collapsing to a monthly mean), matching GeosChem.

Asserts:
- `_time_interp_bracket` returns the right bracket indices + linear weights,
  including constant-extrapolation clamping outside the series.
- The CS `apply_surface_flux!` increments the surface layer by `(linear interp
  of the two bracketing slices) * dt` at exact slice times, a midpoint, and
  clamped ends — driven by a stub `meteo` exposing `current_time`. This drives
  `state.tracers_raw` (the packed multi-tracer CS buffer used by the production
  `scheme = "ppm"` split-sweep path).
- `Adapt.adapt(Array, src)` is identity on CPU (series, compensation and
  window buffer shared) and keeps `times` host-side.
- `PerTracerFluxMap` accepts the new abstract source type.
"""

using Test
using AtmosTransport
using AtmosTransport: CubedSphereState, DryBasis,
                      SurfaceFluxSource, SurfaceFluxOperator,
                      PerTracerFluxMap, apply_surface_flux!,
                      CubedSphereMesh, HybridSigmaPressure, AtmosGrid,
                      allocate_face_fluxes, get_tracer
using AtmosTransport.Operators.SurfaceFlux: TimeVaryingSurfaceFluxSource,
                                            AbstractSurfaceFluxSource,
                                            _time_interp_bracket, _flux_temporal_weights
using AtmosTransport: StepwiseFlux, LinearInterpFlux, ConservativeMeanFlux,
                      flux_temporal_scheme
import Adapt

const FT = Float64

@testset "TimeVarying surface flux — constructor contracts" begin
    series = ntuple(_ -> zeros(FT, 2, 2, 2), 6)
    @test_throws ArgumentError TimeVaryingSurfaceFluxSource(:CO2, series, Float64[])
    @test_throws ArgumentError TimeVaryingSurfaceFluxSource(:CO2, series, [0.0, 0.0])
    @test_throws ArgumentError TimeVaryingSurfaceFluxSource(:CO2, series, [0.0, Inf])
    @test_throws DimensionMismatch TimeVaryingSurfaceFluxSource(:CO2, series, [0.0])
end

# A tiny stand-in for the production `DrivenSimulation` clock: returns a fixed
# elapsed-seconds value. `current_time(meteo)` is the one method the operator
# calls to resolve the time-interpolation bracket.
struct StubClock
    t :: Float64
end
AtmosTransport.MetDrivers.current_time(c::StubClock) = c.t

@testset "TimeVarying surface flux — _time_interp_bracket" begin
    times = [0.0, 3600.0, 7200.0]

    # Exact slice times.
    @test _time_interp_bracket(times, 0.0)    == (1, 1, 1.0, 0.0)
    @test _time_interp_bracket(times, 3600.0) == (2, 3, 1.0, 0.0)   # frac 0 → w on i0
    @test _time_interp_bracket(times, 7200.0) == (3, 3, 1.0, 0.0)   # clamp at end

    # Midpoint: equal weights.
    i0, i1, w0, w1 = _time_interp_bracket(times, 1800.0)
    @test (i0, i1) == (1, 2)
    @test w0 ≈ 0.5
    @test w1 ≈ 0.5

    # General interior fraction.
    i0, i1, w0, w1 = _time_interp_bracket(times, 3600.0 + 900.0)  # 25% into [1,2]→[2,3]
    @test (i0, i1) == (2, 3)
    @test w0 ≈ 0.75
    @test w1 ≈ 0.25

    # Clamp below first and above last (constant extrapolation).
    @test _time_interp_bracket(times, -100.0)  == (1, 1, 1.0, 0.0)
    @test _time_interp_bracket(times, 1.0e6)   == (3, 3, 1.0, 0.0)

    # Single-slice degenerate case.
    @test _time_interp_bracket([5.0], 999.0) == (1, 1, 1.0, 0.0)
end

@testset "TimeVarying surface flux — CS apply + interpolation" begin
    Nc, Hp, Nz = 4, 1, 2
    N = Nc + 2Hp
    mesh = CubedSphereMesh(; Nc = Nc, Hp = Hp, FT = FT)
    vc   = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)

    panels_m  = ntuple(_ -> ones(FT, N, N, Nz), 6)
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    state = CubedSphereState(DryBasis, mesh, panels_m; CO2 = panels_rm)

    # Three time slices with known, panel-distinct per-cell rates so the
    # interpolation is observable cell-by-cell. Slice k carries the constant
    # value `base[p] + (k-1)` on panel p.
    times = [0.0, 3600.0, 7200.0]
    base  = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    series = ntuple(p -> begin
        a = Array{FT, 3}(undef, Nc, Nc, length(times))
        for k in 1:length(times)
            a[:, :, k] .= FT(base[p] + (k - 1))
        end
        a
    end, 6)

    src = TimeVaryingSurfaceFluxSource(:CO2, series, times)
    @test src isa AbstractSurfaceFluxSource
    op  = SurfaceFluxOperator(PerTracerFluxMap(src))

    dt = 100.0

    # Helper: zero the tracer, apply at clock time t, return the surface-layer
    # interior increment for a chosen panel (cell (1,1)).
    function apply_and_probe(t)
        for p in 1:6
            state.tracers.CO2[p] .= zero(FT)
        end
        apply_surface_flux!(state.tracers_raw, op, nothing, dt, StubClock(t), grid;
                            tracer_names = state.tracer_names, halo_width = Hp)
        return ntuple(p -> state.tracers.CO2[p][Hp + 1, Hp + 1, Nz], 6)
    end

    # Exact slice 1 (t=0): value = base[p] + 0.
    probe = apply_and_probe(0.0)
    for p in 1:6
        @test probe[p] ≈ (base[p] + 0.0) * dt
    end

    # Exact slice 2 (t=3600): value = base[p] + 1.
    probe = apply_and_probe(3600.0)
    for p in 1:6
        @test probe[p] ≈ (base[p] + 1.0) * dt
    end

    # Midpoint t=1800: 0.5*slice1 + 0.5*slice2 = base[p] + 0.5.
    probe = apply_and_probe(1800.0)
    for p in 1:6
        @test probe[p] ≈ (base[p] + 0.5) * dt
    end

    # Clamp below the first slice (t<0): equals slice 1.
    probe = apply_and_probe(-500.0)
    for p in 1:6
        @test probe[p] ≈ (base[p] + 0.0) * dt
    end

    # Clamp above the last slice (t>7200): equals slice 3 = base[p] + 2.
    probe = apply_and_probe(1.0e6)
    for p in 1:6
        @test probe[p] ≈ (base[p] + 2.0) * dt
    end

    # The full interior is updated (not just (1,1)); surface layer only.
    state.tracers.CO2[1] .= zero(FT)
    apply_surface_flux!(state.tracers_raw, op, nothing, dt, StubClock(1800.0), grid;
                        tracer_names = state.tracer_names, halo_width = Hp)
    @test all(state.tracers.CO2[1][Hp + 1:Hp + Nc, Hp + 1:Hp + Nc, Nz] .≈
              (base[1] + 0.5) * dt)
    @test all(state.tracers.CO2[1][:, :, 1] .== zero(FT))   # non-surface untouched
end

@testset "TimeVarying surface flux — temporal schemes (dispatch)" begin
    times = [0.0, 3600.0, 7200.0]

    # Config-string mapping.
    @test flux_temporal_scheme("stepwise")     isa StepwiseFlux
    @test flux_temporal_scheme("block")        isa StepwiseFlux
    @test flux_temporal_scheme("linear")       isa LinearInterpFlux
    @test flux_temporal_scheme("conservative") isa ConservativeMeanFlux
    @test_throws ArgumentError flux_temporal_scheme("bogus")

    # Weight contract at t=1800, dt=600.
    @test _flux_temporal_weights(StepwiseFlux(), times, 1800.0, 600.0) == (1, 1, 1.0, 0.0)
    let (i0, i1, w0, w1) = _flux_temporal_weights(LinearInterpFlux(), times, 1800.0, 600.0)
        @test (i0, i1) == (1, 2); @test w0 ≈ 0.5; @test w1 ≈ 0.5
    end
    # Conservative evaluates the linear reconstruction at the step centre t+dt/2=2100.
    let (i0, i1, w0, w1) = _flux_temporal_weights(ConservativeMeanFlux(), times, 1800.0, 600.0)
        @test (i0, i1) == (1, 2); @test w1 ≈ 2100.0 / 3600.0; @test w0 ≈ 1 - 2100.0 / 3600.0
    end

    # End-to-end emission through each scheme (CS packed production path).
    Nc, Hp, Nz = 4, 1, 2
    N = Nc + 2Hp
    mesh = CubedSphereMesh(; Nc = Nc, Hp = Hp, FT = FT)
    vc   = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
    panels_m  = ntuple(_ -> ones(FT, N, N, Nz), 6)
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    state = CubedSphereState(DryBasis, mesh, panels_m; CO2 = panels_rm)
    # slice k on panel 1 = 10 + (k-1)
    series = ntuple(_ -> begin
        a = Array{FT,3}(undef, Nc, Nc, 3)
        for k in 1:3; a[:, :, k] .= FT(10.0 + (k - 1)); end
        a
    end, 6)
    dt = 600.0
    probe(scheme, t) = begin
        for p in 1:6; state.tracers.CO2[p] .= zero(FT); end
        op = SurfaceFluxOperator(PerTracerFluxMap(
            TimeVaryingSurfaceFluxSource(:CO2, series, times, scheme)))
        apply_surface_flux!(state.tracers_raw, op, nothing, dt, StubClock(t), grid;
                            tracer_names = state.tracer_names, halo_width = Hp)
        state.tracers.CO2[1][Hp + 1, Hp + 1, Nz]
    end
    # t=1800 (block [0,3600)): stepwise holds slice1 (10); linear→10.5; conservative→10+2100/3600.
    @test probe(StepwiseFlux(),       1800.0) ≈ 10.0 * dt
    @test probe(LinearInterpFlux(),   1800.0) ≈ 10.5 * dt
    @test probe(ConservativeMeanFlux(), 1800.0) ≈ (10.0 + 2100.0 / 3600.0) * dt
    # Default (no scheme arg) == linear.
    @test probe(LinearInterpFlux(), 1800.0) ≈
          (begin
              for p in 1:6; state.tracers.CO2[p] .= zero(FT); end
              op = SurfaceFluxOperator(PerTracerFluxMap(
                  TimeVaryingSurfaceFluxSource(:CO2, series, times)))
              apply_surface_flux!(state.tracers_raw, op, nothing, dt, StubClock(1800.0), grid;
                                  tracer_names = state.tracer_names, halo_width = Hp)
              state.tracers.CO2[1][Hp + 1, Hp + 1, Nz]
          end)
end

@testset "TimeVarying surface flux — ConservativeMeanFlux knot-crossing integral" begin
    # Non-collinear slices so the flux has a genuine slope change at the 3600 s
    # knot (10 → 11 → 20). A step that straddles the knot must integrate the
    # piecewise-linear reconstruction across BOTH sub-intervals, not just sample
    # the step centre. Exact integral over [t, t+dt] (trapezoidal per piece):
    times  = [0.0, 3600.0, 7200.0]
    vals   = (10.0, 11.0, 20.0)
    SF = AtmosTransport.Operators.SurfaceFlux

    f(tau) = begin   # piecewise-linear reconstruction (clamped)
        tau <= times[1] && return vals[1]
        tau >= times[end] && return vals[end]
        k = searchsortedlast(times, tau)
        frac = (tau - times[k]) / (times[k+1] - times[k])
        (1 - frac) * vals[k] + frac * vals[k+1]
    end
    t, dt = 3000.0, 1200.0     # spans [3000, 4200], crossing the 3600 knot
    exact = (0.5*(f(3000)+f(3600))*600 + 0.5*(f(3600)+f(4200))*600)  # ∫ f dτ

    # Segments must reproduce the exact integral; naive single-midpoint would give f(3600)*dt.
    segs = SF._flux_temporal_segments(ConservativeMeanFlux(), times, t, dt)
    @test length(segs) == 2
    eff = sum(((i0,i1,w0,w1,frac),) -> (w0*vals[i0] + w1*vals[i1]) * frac, segs) * dt
    @test eff ≈ exact
    @test !(eff ≈ f(3600) * dt)   # the fix actually changed the answer

    # End-to-end through the CS packed apply path.
    Nc, Hp, Nz = 4, 1, 2; N = Nc + 2Hp
    mesh = CubedSphereMesh(; Nc = Nc, Hp = Hp, FT = FT)
    vc   = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
    state = CubedSphereState(DryBasis, mesh, ntuple(_ -> ones(FT, N, N, Nz), 6);
                             CO2 = ntuple(_ -> zeros(FT, N, N, Nz), 6))
    series = ntuple(_ -> begin
        a = Array{FT,3}(undef, Nc, Nc, 3); for k in 1:3; a[:,:,k] .= FT(vals[k]); end; a
    end, 6)
    op = SurfaceFluxOperator(PerTracerFluxMap(
        TimeVaryingSurfaceFluxSource(:CO2, series, times, ConservativeMeanFlux())))
    apply_surface_flux!(state.tracers_raw, op, nothing, dt, StubClock(t), grid;
                        tracer_names = state.tracer_names, halo_width = Hp)
    @test state.tracers.CO2[1][Hp+1, Hp+1, Nz] ≈ exact
end

@testset "TimeVarying surface flux — Adapt identity + PerTracerFluxMap" begin
    times  = [0.0, 3600.0]
    series = ntuple(_ -> fill(FT(1.0), 2, 2, 2), 6)
    src    = TimeVaryingSurfaceFluxSource(:CO2, series, times)

    # Adapt-to-Array is identity on CPU; times stays a host Vector, and the
    # series, compensation and window buffer are the source's own.
    src2 = Adapt.adapt(Array, src)
    @test src2 isa TimeVaryingSurfaceFluxSource
    @test src2.times === src.times
    @test src2.times isa Vector{Float64}
    @test src2.cell_mass_rate_series === src.cell_mass_rate_series
    @test src2.compensation === src.compensation
    @test src2.window_buffer === src.window_buffer

    # PerTracerFluxMap accepts the abstract source type (mixed with a static one).
    static = SurfaceFluxSource(:SF6, ntuple(_ -> fill(FT(2.0), 2, 2), 6))
    m = PerTracerFluxMap(src, static)
    @test length(m) == 2
    @test AtmosTransport.flux_for(m, :CO2) === src
    @test AtmosTransport.flux_for(m, :SF6) === static
end
