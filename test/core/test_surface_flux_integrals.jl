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
- `total_emission_rate(src, t)` reproduces the runner's former log value
  (static lat-lon and cubed-sphere sources, time-varying and `cs_native`
  sources at the run start), bit for bit for Float64 sources and to Float32
  rounding for Float32 sources (now summed in Float64), and evaluates each
  scheme's reconstruction at `t`;
- `emission_segments` covers the step and sums to the one-step integral;
- invalid step arguments are rejected;
- a source adapted to a device array type whose Float64 reductions fail (the
  Metal case, emulated on the CPU) gives the host source's totals bit for bit:
  the sums run on host copies.
"""

using Test
using Adapt
using Dates
using KernelAbstractions: KernelAbstractions as KA
using NCDatasets
using Random
using AtmosTransport
using AtmosTransport: CubedSphereState, DryBasis, SurfaceFluxSource, SurfaceFluxOperator,
                      TimeVaryingSurfaceFluxSource, PerTracerFluxMap, apply_surface_flux!,
                      CubedSphereMesh, LatLonMesh, HybridSigmaPressure, AtmosGrid,
                      StepwiseFlux, LinearInterpFlux, ConservativeMeanFlux
using AtmosTransport.Operators: total_emission_rate, emission_segments, integrated_emission
using AtmosTransport.Models.InitialConditionIO: build_surface_flux_source

# Stand-in for the `DrivenSimulation` clock: the operator reads only `current_time`.
struct IntegralTestClock
    t :: Float64
end
AtmosTransport.MetDrivers.current_time(c::IntegralTestClock) = c.t

# The runner's surface-source log value before `total_emission_rate` replaced it
# (`DrivenRunner._surface_source_total_rate`): a sum in the storage float type,
# over the first slice of a time-varying source.
function legacy_logged_rate(source)
    if hasproperty(source, :cell_mass_rate)
        r = source.cell_mass_rate
        return r isa Tuple ? Float64(sum(sum, r)) : Float64(sum(r))
    else
        series = source.cell_mass_rate_series
        return Float64(sum(p -> sum(@view p[:, :, 1]), series))
    end
end

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

@testset "total_emission_rate replaces the runner's log value" begin
    @testset "static sources ($FT)" for FT in (Float64, Float32)
        rng = MersenneTwister(3)
        ll = SurfaceFluxSource(:CO2, FT.(1e-4 .* (1 .+ rand(rng, 36, 18))))
        cs = SurfaceFluxSource(:CO2, ntuple(p -> FT.(1e-4 * p .* (1 .+ rand(rng, NC, NC))), 6))
        for src in (ll, cs)
            r = src.cell_mass_rate
            @test total_emission_rate(src, 0.0) === total_emission_rate(src, 1.0e7)
            @test total_emission_rate(src, 0.0) === (r isa Tuple ? sum(p -> sum(Float64, p), r) : sum(Float64, r))
            if FT === Float64
                @test total_emission_rate(src, 0.0) === legacy_logged_rate(src)
            else   # the former log summed in Float32
                @test total_emission_rate(src, 0.0) ≈ legacy_logged_rate(src) rtol = 1e-6
            end
        end
    end

    @testset "time-varying sources at the run start and at t" begin
        times = [0.0, 3600.0, 7200.0]
        series = random_series(Float64, 3)
        R = slice_totals(series)
        for scheme in (StepwiseFlux(), LinearInterpFlux(), ConservativeMeanFlux())
            src = TimeVaryingSurfaceFluxSource(:CO2, series, times, scheme)
            @test total_emission_rate(src, 0.0) === legacy_logged_rate(src) === R[1]
            @test total_emission_rate(src, -50.0) == R[1]           # before the series
            @test total_emission_rate(src, 9000.0) == R[3]          # after the series
            expected = scheme isa StepwiseFlux ? R[1] : 0.75 * R[1] + 0.25 * R[2]
            @test total_emission_rate(src, 900.0) ≈ expected rtol = 1e-15
        end
        series32 = random_series(Float32, 3)
        src32 = TimeVaryingSurfaceFluxSource(:CO2, series32, times, StepwiseFlux())
        @test total_emission_rate(src32, 0.0) === slice_totals(series32)[1]
        @test total_emission_rate(src32, 0.0) ≈ legacy_logged_rate(src32) rtol = 1e-6

        # Without a slice at the run start, linear schemes report the
        # interpolated run-start rate; the former log showed the first slice.
        early = TimeVaryingSurfaceFluxSource(:CO2, series, [-3600.0, 3600.0, 7200.0], LinearInterpFlux())
        @test total_emission_rate(early, 0.0) ≈ 0.5 * (R[1] + R[2]) rtol = 1e-15
        stepwise_early = TimeVaryingSurfaceFluxSource(:CO2, series, [-3600.0, 3600.0, 7200.0], StepwiseFlux())
        @test total_emission_rate(stepwise_early, 0.0) === legacy_logged_rate(stepwise_early)
    end

    @testset "cs_native source ($FT)" for FT in (Float64, Float32)
        mktempdir() do dir
            path = joinpath(dir, "native.nc")
            density(i, j, p, t) = 1e-9 * (i + 10j + 100p + 1000t)        # kg CO2 m⁻² s⁻¹
            NCDataset(path, "c") do ds
                for (name, n) in (("Xdim", NC), ("Ydim", NC), ("nf", 6), ("time", 3))
                    defDim(ds, name, n)
                end
                defVar(ds, "time", Float64, ("time",),
                       attrib = Dict("units" => "hours since 2014-01-01 00:00:00"))[:] = [0, 24, 48]
                v = defVar(ds, "F", Float64, ("Xdim", "Ydim", "nf", "time"),
                           attrib = Dict("units" => "kg CO2 m-2 s-1"))
                v[:, :, :, :] = [density(i, j, p, t) for i in 1:NC, j in 1:NC, p in 1:6, t in 1:3]
            end
            grid = cs_grid(FT)
            M = 0.0440095
            cfg = Dict{String, Any}("kind" => "cs_native", "file" => path, "variable" => "F",
                                    "time_varying" => true, "molar_mass_kg_mol" => M, "scale" => 2.0)
            # The runner passes the run span; the series then starts at the slice
            # in effect at the run start (2014-01-02 00 UTC = file hour 24).
            src = build_surface_flux_source(grid, :co2, cfg, FT;
                                            reference_time = DateTime(2014, 1, 2),
                                            run_span = (0.0, 86400.0 + 1.0))
            @test src isa TimeVaryingSurfaceFluxSource
            @test src.times == [0.0, 86400.0]
            areas = Float64.(grid.horizontal.cell_areas)
            storage = 2.0 * AtmosTransport.Parameters.DRY_AIR_MOLAR_MASS / M
            from_file = sum(density(i, j, p, 2) * areas[i, j] * storage for i in 1:NC, j in 1:NC, p in 1:6)
            rate = total_emission_rate(src, 0.0)
            if FT === Float64
                @test rate === legacy_logged_rate(src)
                @test rate ≈ from_file rtol = 1e-13
            else
                @test rate ≈ legacy_logged_rate(src) rtol = 1e-6
                @test rate ≈ from_file rtol = 1e-6
            end
            @test integrated_emission(src, 0.0, 86400.0, 450.0) ≈ rate * 86400 rtol = 1e-14
        end
    end
end

@testset "device sources are summed on host copies" begin
    rng = MersenneTwister(11)
    times = [0.0, 3600.0, 7200.0]
    hosts = (SurfaceFluxSource(:CO2, 1e-4 .* (1 .+ rand(rng, 36, 18))),
             SurfaceFluxSource(:CO2, ntuple(p -> Float32.(1e-4 * p .* (1 .+ rand(rng, NC, NC))), 6)),
             TimeVaryingSurfaceFluxSource(:CO2, random_series(Float64, 3)[2], times, StepwiseFlux()),
             TimeVaryingSurfaceFluxSource(:CO2, random_series(Float32, 3), times, LinearInterpFlux()),
             TimeVaryingSurfaceFluxSource(:CO2, random_series(Float64, 3), times, ConservativeMeanFlux()))
    for host in hosts
        device = Adapt.adapt(ToNoFloat64Device(), host)
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
