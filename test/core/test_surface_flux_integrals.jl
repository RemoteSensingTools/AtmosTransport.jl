"""
Emission totals of surface sources: `total_emission_rate`, `emission_segments`
and `integrated_emission` (`src/Operators/SurfaceFlux/integrals.jl`).

Asserts:
- `integrated_emission(src, t0, t1, dt)` equals the mass the CPU operator adds
  when stepped over `[t0, t1]` with step `dt`, for a static source and every
  temporal scheme, to 1e-14 relative (Float64 storage), with steps that
  straddle slice times, start on a slice time, and run past both series ends;
- for `StepwiseFlux` and `ConservativeMeanFlux` the integral equals the
  analytic integral of the piecewise-constant / piecewise-linear
  reconstruction and does not depend on `dt`; for `LinearInterpFlux` it is the
  step-start Riemann sum;
- `total_emission_rate(src, t)` sums static and time-varying sources in
  Float64 and evaluates each scheme's reconstruction at `t`;
- `emission_segments` covers the step and sums to the one-step integral;
- invalid step arguments are rejected;
- a source adapted to a device array type whose Float64 reductions fail (the
  Metal case, emulated on the CPU) gives the host source's totals bit for bit:
  static rates and series built on the device are summed on host copies, and
  an adapted time-varying source keeps its series in host memory.
"""

using Test
using Adapt
using KernelAbstractions: KernelAbstractions as KA
using Random
using AtmosTransport
using AtmosTransport: CubedSphereState, DryBasis, SurfaceFluxSource, SurfaceFluxOperator,
                      TimeVaryingSurfaceFluxSource, PerTracerFluxMap, apply_surface_flux!,
                      CubedSphereMesh, LatLonMesh, HybridSigmaPressure, AtmosGrid,
                      StepwiseFlux, LinearInterpFlux, ConservativeMeanFlux
using AtmosTransport.Operators: total_emission_rate, emission_segments, integrated_emission

# Stand-in for the `DrivenSimulation` clock: the operator reads only `current_time`.
struct IntegralTestClock
    t :: Float64
end
AtmosTransport.MetDrivers.current_time(c::IntegralTestClock) = c.t

# Stand-in for a Metal array on the CPU: its backend is a device, and a Float64
# reduction over it or over a view of it throws (Metal has no Float64).
struct NoFloat64Backend <: KA.GPU end
struct NoFloat64Array{T, N} <: AbstractArray{T, N}
    data :: Array{T, N}
end
Base.size(a::NoFloat64Array) = size(a.data)
Base.getindex(a::NoFloat64Array{T, N}, i::Vararg{Int, N}) where {T, N} = a.data[i...]
KA.get_backend(::NoFloat64Array) = NoFloat64Backend()
const NoFloat64Arrays = Union{NoFloat64Array, SubArray{<:Any, <:Any, <:NoFloat64Array}}
function Base.mapreduce(f, op, a::NoFloat64Arrays; kw...)
    Base.promote_op(f, eltype(a)) === Float64 &&
        throw(ArgumentError("this device has no Float64"))
    return mapreduce(f, op, collect(a); kw...)
end
struct ToNoFloat64Device end
Adapt.adapt_storage(::ToNoFloat64Device, x::Array) = NoFloat64Array(x)

const NC, HP, NZ = 4, 1, 2

cs_grid(FT) = AtmosGrid(CubedSphereMesh(; Nc = NC, Hp = HP, FT = FT),
                        HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1]), AtmosTransport.CPU(); FT)

# Positive per-cell rates that differ by cell, panel and slice.
function random_series(FT, ntime; seed = 7)
    rng = MersenneTwister(seed)
    return ntuple(p -> FT[1e-3 * (1 + rand(rng)) * (1 + 0.5k) * p for i in 1:NC, j in 1:NC, k in 1:ntime], 6)
end

# Mass the CPU operator adds when stepped from t0 with `nsteps` steps of `dt`.
function stepped_application(src, ::Type{FT}, t0, dt, nsteps) where FT
    grid = cs_grid(FT)
    N = NC + 2HP
    state = CubedSphereState(DryBasis, grid.horizontal, ntuple(_ -> ones(FT, N, N, NZ), 6);
                             CO2 = ntuple(_ -> zeros(FT, N, N, NZ), 6))
    op = SurfaceFluxOperator(PerTracerFluxMap(src))
    for n in 0:(nsteps - 1)
        apply_surface_flux!(state.tracers_raw, op, nothing, dt, IntegralTestClock(t0 + n * dt), grid;
                            tracer_names = state.tracer_names, halo_width = HP)
    end
    interior = HP + 1:HP + NC
    return sum(p -> sum(Float64, @view state.tracers.CO2[p][interior, interior, :]), 1:6)
end

slice_totals(series) = [sum(p -> sum(Float64, @view p[:, :, k]), series) for k in axes(series[1], 3)]

# R(t) of each reconstruction, written out independently of `sources.jl`.
function stepwise_rate(R, times, t)
    t <= times[1] && return R[1]
    return R[searchsortedlast(times, t)]
end
function linear_rate(R, times, t)
    t <= times[1] && return R[1]
    t >= times[end] && return R[end]
    k = searchsortedlast(times, t)
    f = (t - times[k]) / (times[k + 1] - times[k])
    return (1 - f) * R[k] + f * R[k + 1]
end

# ∫_{a}^{b} of a reconstruction that is constant (stepwise) or linear between
# knots and constant outside them: break at the knots, integrate each piece exactly.
function exact_integral(R, times, a, b; linear::Bool)
    edges = sort(unique([a; filter(t -> a < t < b, times); b]))
    total = 0.0
    for (lo, hi) in zip(edges[1:end-1], edges[2:end])
        total += linear ? 0.5 * (linear_rate(R, times, lo) + linear_rate(R, times, hi)) * (hi - lo) :
                          stepwise_rate(R, times, 0.5 * (lo + hi)) * (hi - lo)
    end
    return total
end

@testset "integrated_emission equals the stepped operator" begin
    FT = Float64
    # Knots: before the first step's end, on a step boundary (2800 = 4·700),
    # inside steps, and a last slice before the run end (constant extrapolation).
    times = [600.0, 2800.0, 3600.0, 5400.0, 9000.0, 9600.0]
    series = random_series(FT, length(times))
    R = slice_totals(series)
    dt = 700.0
    for (t0, nsteps) in ((0.0, 20), (2100.0, 7))
        t1 = t0 + nsteps * dt
        for scheme in (StepwiseFlux(), LinearInterpFlux(), ConservativeMeanFlux())
            src = TimeVaryingSurfaceFluxSource(:CO2, series, times, scheme)
            E = integrated_emission(src, t0, t1, dt)
            applied = stepped_application(src, FT, t0, dt, nsteps)
            @test isapprox(E, applied; rtol = 1e-14)

            if scheme isa LinearInterpFlux
                # Point value at each step start: a Riemann sum that depends on dt.
                riemann = sum(linear_rate(R, times, t0 + n * dt) * dt for n in 0:(nsteps - 1))
                @test isapprox(E, riemann; rtol = 1e-14)
                @test !isapprox(integrated_emission(src, t0, t1, t1 - t0), E; rtol = 1e-6)
            else
                # Exact integral of the reconstruction, independent of the step.
                exact = exact_integral(R, times, t0, t1; linear = scheme isa ConservativeMeanFlux)
                @test isapprox(E, exact; rtol = 1e-14)
                @test isapprox(integrated_emission(src, t0, t1, t1 - t0), E; rtol = 1e-14)
                @test isapprox(integrated_emission(src, t0, t1, dt / 4), E; rtol = 1e-14)
            end
        end
    end

    rates = ntuple(p -> series[p][:, :, 2], 6)
    static = SurfaceFluxSource(:CO2, rates)
    E = integrated_emission(static, 0.0, 20 * dt, dt)
    @test isapprox(E, stepped_application(static, FT, 0.0, dt, 20); rtol = 1e-14)
    @test E == total_emission_rate(static, 0.0) * (20 * dt)

    # Float32 storage: the operator rounds every increment to Float32; the
    # integral (Float64 throughout) matches to that rounding.
    times32 = Float64[0, 3600, 7200]
    series32 = random_series(Float32, 3)
    src32 = TimeVaryingSurfaceFluxSource(:CO2, series32, times32, ConservativeMeanFlux())
    @test isapprox(integrated_emission(src32, 0.0, 9000.0, 450.0),
                   stepped_application(src32, Float32, 0.0, 450.0, 20); rtol = 1e-6)
end

@testset "emission_segments and step arguments" begin
    times = [0.0, 1000.0, 2500.0]
    series = random_series(Float64, 3)
    for scheme in (StepwiseFlux(), LinearInterpFlux(), ConservativeMeanFlux())
        src = TimeVaryingSurfaceFluxSource(:CO2, series, times, scheme)
        segs = emission_segments(src, 700.0, 2400.0)       # crosses 1000 and 2500
        @test length(segs) == (scheme isa LinearInterpFlux ? 1 : 3)
        @test sum(last, segs) ≈ 2400.0 rtol = 1e-15
        @test sum(r * τ for (r, τ) in segs) ≈ integrated_emission(src, 700.0, 3100.0, 2400.0) rtol = 1e-15
    end
    static = SurfaceFluxSource(:CO2, ntuple(p -> series[p][:, :, 1], 6))
    @test emission_segments(static, 5.0, 60.0) == [(total_emission_rate(static, 5.0), 60.0)]

    src = TimeVaryingSurfaceFluxSource(:CO2, series, times, StepwiseFlux())
    @test integrated_emission(src, 100.0, 100.0, 60.0) == 0.0
    @test integrated_emission(static, 100.0, 100.0, 60.0) == 0.0
    for s in (src, static)
        @test_throws ArgumentError integrated_emission(s, 0.0, 600.0, 0.0)
        @test_throws ArgumentError integrated_emission(s, 0.0, 600.0, -60.0)
        @test_throws ArgumentError integrated_emission(s, 600.0, 0.0, 60.0)
        @test_throws ArgumentError integrated_emission(s, 0.0, 610.0, 60.0)   # not whole steps
        @test_throws ArgumentError integrated_emission(s, 0.0, Inf, 60.0)
        @test_throws ArgumentError integrated_emission(s, 0.0, 600.0, Inf)
        @test_throws ArgumentError integrated_emission(s, NaN, 600.0, 60.0)
    end
    # The runner's step 3600/7 s: whole steps to rounding are accepted.
    @test integrated_emission(static, 0.0, 3600.0, 3600 / 7) ≈ total_emission_rate(static, 0.0) * 3600 rtol = 1e-14
end

@testset "total_emission_rate uses Float64 totals at the requested time" begin
    @testset "static sources ($FT)" for FT in (Float64, Float32)
        rng = MersenneTwister(3)
        ll = SurfaceFluxSource(:CO2, FT.(1e-4 .* (1 .+ rand(rng, 36, 18))))
        cs = SurfaceFluxSource(:CO2, ntuple(p -> FT.(1e-4 * p .* (1 .+ rand(rng, NC, NC))), 6))
        for src in (ll, cs)
            r = src.cell_mass_rate
            @test total_emission_rate(src, 0.0) === total_emission_rate(src, 1.0e7)
            @test total_emission_rate(src, 0.0) === (r isa Tuple ? sum(p -> sum(Float64, p), r) : sum(Float64, r))
        end
    end

    @testset "time-varying sources at the run start and at t" begin
        times = [0.0, 3600.0, 7200.0]
        series = random_series(Float64, 3)
        R = slice_totals(series)
        for scheme in (StepwiseFlux(), LinearInterpFlux(), ConservativeMeanFlux())
            src = TimeVaryingSurfaceFluxSource(:CO2, series, times, scheme)
            @test total_emission_rate(src, 0.0) === R[1]
            @test total_emission_rate(src, -50.0) == R[1]           # before the series
            @test total_emission_rate(src, 9000.0) == R[3]          # after the series
            expected = scheme isa StepwiseFlux ? R[1] : 0.75 * R[1] + 0.25 * R[2]
            @test total_emission_rate(src, 900.0) ≈ expected rtol = 1e-15
        end
        series32 = random_series(Float32, 3)
        src32 = TimeVaryingSurfaceFluxSource(:CO2, series32, times, StepwiseFlux())
        @test total_emission_rate(src32, 0.0) === slice_totals(series32)[1]

        # Without a slice at the run start, linear schemes report the
        # interpolated run-start rate; the former log showed the first slice.
        early = TimeVaryingSurfaceFluxSource(:CO2, series, [-3600.0, 3600.0, 7200.0], LinearInterpFlux())
        @test total_emission_rate(early, 0.0) ≈ 0.5 * (R[1] + R[2]) rtol = 1e-15
        stepwise_early = TimeVaryingSurfaceFluxSource(:CO2, series, [-3600.0, 3600.0, 7200.0], StepwiseFlux())
        @test total_emission_rate(stepwise_early, 0.0) === R[1]
    end

end

@testset "device sources are summed on host copies" begin
    rng = MersenneTwister(11)
    times = [0.0, 3600.0, 7200.0]
    statics = (SurfaceFluxSource(:CO2, 1e-4 .* (1 .+ rand(rng, 36, 18))),
               SurfaceFluxSource(:CO2, ntuple(p -> Float32.(1e-4 * p .* (1 .+ rand(rng, NC, NC))), 6)))
    series = (random_series(Float64, 3)[2], random_series(Float32, 3), random_series(Float64, 3))
    schemes = (StepwiseFlux(), LinearInterpFlux(), ConservativeMeanFlux())
    varying = map((s, scheme) -> TimeVaryingSurfaceFluxSource(:CO2, s, times, scheme), series, schemes)

    # Adapting a time-varying source moves only its compensation and window
    # buffer: the series stays in host memory and is summed in place.
    for host in varying
        device = Adapt.adapt(ToNoFloat64Device(), host)
        @test device.cell_mass_rate_series === host.cell_mass_rate_series
        slots = device.window_buffer.slots
        @test (slots isa Tuple ? first(slots) : slots) isa NoFloat64Array
        for t in (0.0, 1800.0, 9000.0)
            @test total_emission_rate(device, t) === total_emission_rate(host, t)
        end
        @test integrated_emission(device, 0.0, 7200.0, 450.0) === integrated_emission(host, 0.0, 7200.0, 450.0)
    end

    # Static rates adapted to the device, and a series built on the device,
    # are copied to the host for their sums.
    on_device = (map(host -> (host, Adapt.adapt(ToNoFloat64Device(), host)), statics)...,
                 map((host, s) -> (host, TimeVaryingSurfaceFluxSource(:CO2, Adapt.adapt(ToNoFloat64Device(), s),
                                                                       times, host.scheme)),
                     varying, series)...)
    for (host, device) in on_device
        rates = host isa SurfaceFluxSource ? device.cell_mass_rate : device.cell_mass_rate_series
        array = rates isa Tuple ? first(rates) : rates
        @test array isa NoFloat64Array
        @test_throws ArgumentError sum(Float64, array)                 # the stand-in rejects Float64
        @test_throws ArgumentError sum(Float64, selectdim(array, ndims(array), 1))
        for t in (0.0, 1800.0, 9000.0)
            @test total_emission_rate(device, t) === total_emission_rate(host, t)
        end
        @test emission_segments(device, 700.0, 2400.0) == emission_segments(host, 700.0, 2400.0)
        @test integrated_emission(device, 0.0, 7200.0, 450.0) === integrated_emission(host, 0.0, 7200.0, 450.0)
    end
end
