# Shared fixtures of the time-varying surface-flux window-buffer tests
# (test/core/test_surface_flux_window_buffer.jl on the CPU,
# test/diagnostic/test_surface_flux_window_buffer_gpu.jl on CUDA): a clock, slice
# layouts, the runner's step times, the pre-R1 full-series operator as a
# reference, a small cubed-sphere state on any array type, and a window driver.

using Adapt
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using AtmosTransport
using AtmosTransport: CubedSphereState, DryBasis, SurfaceFluxOperator, PerTracerFluxMap,
                      apply_surface_flux!, CubedSphereMesh, HybridSigmaPressure, AtmosGrid,
                      StepwiseFlux, LinearInterpFlux, ConservativeMeanFlux,
                      TimeVaryingSurfaceFluxSource, SurfaceFluxSource

const SF = AtmosTransport.Operators.SurfaceFlux
const MD = AtmosTransport.MetDrivers

struct WindowTestClock
    t :: Float64
end
AtmosTransport.MetDrivers.current_time(c::WindowTestClock) = c.t

const SCHEMES = (StepwiseFlux(), LinearInterpFlux(), ConservativeMeanFlux())
const NC, HP, NZ = 4, 1, 3

# Bitwise equality of float arrays (or tuples of them): a NaN payload and the
# sign of a zero count.
bitequal(a::AbstractArray{T}, b::AbstractArray{T}) where {T <: AbstractFloat} =
    size(a) == size(b) && all(((x, y),) -> reinterpret(Unsigned, x) == reinterpret(Unsigned, y), zip(a, b))
bitequal(a::Tuple, b::Tuple) = length(a) == length(b) && all(map(bitequal, a, b))

# Per-cell rates that differ by cell, panel and slice.
series_for(FT, ntime, seed; nc = NC, scale = 1e-3) =
    ntuple(p -> FT[scale * (1 + 0.37 * sin(seed + i + 3j + 7p + 11k)) * (1 + 0.1k)
                   for i in 1:nc, j in 1:nc, k in 1:ntime], 6)

# The runner's step `n` of window `w` (both from 0): `DrivenSimulation`'s clock
# (`_clock_time`) and its step length `window_dt / steps` in the storage type.
step_time(t_start, W::FT, w, n, steps) where FT = t_start + Float64(W) * (w + n / steps)
step_length(W::FT, steps) where FT = W / FT(steps)

# The slices `_flux_temporal_segments` references for the steps of window `w`.
function slices_read(scheme, times, W::FT, w, steps; t_start = 0.0) where FT
    ks = Set{Int}()
    for n in 0:(steps - 1), (i0, i1, _, _, _) in
            SF._flux_temporal_segments(scheme, times, step_time(t_start, W, w, n, steps),
                                       Float64(step_length(W, steps)))
        push!(ks, i0, i1)
    end
    return ks
end

window_span(W, w; t_start = 0.0) = (t_start + Float64(W) * w, t_start + Float64(W) * (w + 1))

# Slice-time layouts: hourly (31 slices, more than any buffer holds), 3-hourly
# from before the run start to before its end, and irregular.
const HOURLY = collect(0.0:3600.0:(30 * 3600.0))
const THREE_HOURLY = collect(-5400.0:10800.0:(20 * 3600.0))
const IRREGULAR = [100.0, 2000.0, 3600.0, 3700.0, 9000.0, 20000.5, 50000.0]
const WINDOWS = ((3600.0, 32), (5400.0, 21), (1800.0, 64), (10800.0, 11))   # (length, count)
const STEPS = (1, 4, 7, 11)

# ---------------------------------------------------------------------------
# Reference: the operator before R1, which read slices `i0`, `i1` from the full
# series (moved to the device whole by the old Adapt hook: `to = CuArray` puts
# the series there again). It carries its own Kahan compensation, as the source
# did.
# ---------------------------------------------------------------------------
struct FullSeriesReference{S, T, Sch, C} <: SF.AbstractSurfaceFluxSource
    tracer_name  :: Symbol
    series       :: S
    times        :: T
    scheme       :: Sch
    compensation :: C
end
FullSeriesReference(src::TimeVaryingSurfaceFluxSource, to = Array) =
    FullSeriesReference(src.tracer_name, Adapt.adapt(to, src.cell_mass_rate_series), src.times,
                        src.scheme, Adapt.adapt(to, map(c -> zero(c), src.compensation)))

# Reproduce the pre-R1 lifecycle in DrivenSimulation tests: adapting the same
# host reference for each binary creates a fresh full device series and fresh
# compensation, just as adapting a host TimeVaryingSurfaceFluxSource creates a
# fresh window buffer and compensation.
function Adapt.adapt_structure(to, src::FullSeriesReference)
    series = Adapt.adapt(to, src.series)
    compensation = Adapt.adapt(to, src.compensation)
    return FullSeriesReference(src.tracer_name, series, src.times, src.scheme, compensation)
end

# In the operator, the reference reads slices `i0`, `i1` of its full series.
function SF._cs_surface_flux_plan(q_raw::NTuple{6}, src::FullSeriesReference, dt_FT,
                                  t_idx::Integer, Hp, meteo)
    segments = SF._flux_temporal_segments(src.scheme, src.times, Float64(SF.current_time(meteo)),
                                          Float64(dt_FT))
    SF._check_cs_panel_shapes(q_raw, src.series, src, Hp)
    return SF._cs_surface_flux_plan(src.series, src.compensation, t_idx, dt_FT, segments)
end

# The launch structure before the fused operator, as a reference with the
# production cell arithmetic (`SF._deposit!`): one launch per source, segment
# and panel, sources in map order.
@kernel function reference_cs_interp_kernel!(q_raw, @Const(series), comp, w0, w1, i0, i1,
                                             dt, tracer_idx, Nz, Hp, deposit)
    ii, jj = @index(Global, NTuple)
    @inbounds SF._deposit!(q_raw, comp, (w0 * series[ii, jj, i0] + w1 * series[ii, jj, i1]) * dt,
                           ii, jj, Hp, Nz, tracer_idx, deposit)
end
@kernel function reference_cs_static_kernel!(q_raw, @Const(rate), comp, dt, tracer_idx, Nz, Hp,
                                             deposit)
    ii, jj = @index(Global, NTuple)
    @inbounds SF._deposit!(q_raw, comp, rate[ii, jj] * dt, ii, jj, Hp, Nz, tracer_idx, deposit)
end

function reference_cs_apply!(q_raw::NTuple{6}, src::FullSeriesReference, dt_FT::FT, t_idx,
                             Hp, meteo, deposit) where FT
    backend = KernelAbstractions.get_backend(q_raw[1])
    segments = SF._flux_temporal_segments(src.scheme, src.times, Float64(SF.current_time(meteo)),
                                          Float64(dt_FT))
    kernel = reference_cs_interp_kernel!(backend, (16, 16))
    for p in 1:6
        Nc, Ny, Nz = SF._cs_interior_shape(q_raw[p], Hp, src.series[p], src, p)
        for (i0, i1, w0, w1, frac) in segments
            kernel(q_raw[p], src.series[p], src.compensation[p], FT(w0), FT(w1), i0, i1,
                   dt_FT * FT(frac), t_idx, Nz, Hp, SF._panel_deposit(deposit, p); ndrange = (Nc, Ny))
        end
    end
    return nothing
end

function reference_cs_apply!(q_raw::NTuple{6}, src::SurfaceFluxSource, dt_FT::FT, t_idx,
                             Hp, meteo, deposit) where FT
    backend = KernelAbstractions.get_backend(q_raw[1])
    kernel = reference_cs_static_kernel!(backend, (16, 16))
    for p in 1:6
        Nc, Ny, Nz = SF._cs_interior_shape(q_raw[p], Hp, src.cell_mass_rate[p], src, p)
        kernel(q_raw[p], src.cell_mass_rate[p], src.compensation[p], dt_FT, t_idx, Nz, Hp,
               SF._panel_deposit(deposit, p); ndrange = (Nc, Ny))
    end
    return nothing
end

function reference_cs_apply!(q_raw::NTuple{6}, sources::Tuple, dt_FT, meteo, tracer_names;
                             halo_width, deposit = SF.SurfaceLayerDeposit())
    for src in sources
        t_idx = findfirst(==(src.tracer_name), tracer_names)
        t_idx === nothing && continue
        reference_cs_apply!(q_raw, src, dt_FT, t_idx, halo_width, meteo, deposit)
    end
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(q_raw[1]))
    return nothing
end

function SF._check_surface_source_compatibility(state::CubedSphereState, src::FullSeriesReference)
    SF._check_surface_source_compatibility(state,
        TimeVaryingSurfaceFluxSource(src.tracer_name, src.series, src.times, src.scheme))
end

# A small cubed-sphere state whose panels are `arr` arrays (`CuArray` on CUDA).
function cs_state(FT; packed = true, arr = identity)
    mesh = CubedSphereMesh(; Nc = NC, Hp = HP, FT = FT)
    grid = AtmosGrid(mesh, HybridSigmaPressure(FT[0, 100, 300, 600], FT[0, 0, 0.5, 1]),
                     AtmosTransport.CPU(); FT = FT)
    N = NC + 2HP
    air = ntuple(_ -> arr(fill(FT(1e4), N, N, NZ)), 6)
    # A large background makes each increment round (and the Kahan terms matter).
    state = packed ?
        CubedSphereState(DryBasis, mesh, air; co2 = map(a -> a .* FT(4e-4), air),
                         tag = map(a -> a .* FT(2e-6), air)) :
        CubedSphereState(DryBasis, mesh, air; co2 = map(a -> a .* FT(4e-4), air))
    return state, grid
end

# Step `sources` through `nwin` windows of `steps` steps as `DrivenSimulation`
# does: stage each window (unless `stage = false`), then apply every step.
# Returns host copies of the tracers and of the compensation of every source
# and, per window, the number of slices the steps copied after staging.
# `single = true` applies to the first tracer's `(N, N, Nz)` panels instead of
# the packed `(N, N, Nz, Nt)` buffer.
function drive!(state, grid, sources, W::FT, nwin, steps; stage = true, deposit = nothing,
                single = false) where FT
    op = SurfaceFluxOperator(PerTracerFluxMap(sources))
    target = single ? AtmosTransport.get_tracer(state, 1) : state.tracers_raw
    names = single ? (state.tracer_names[1],) : state.tracer_names
    kw = deposit === nothing ? (;) : (; deposit)
    dt = step_length(W, steps)
    copied_during_steps = Int[]
    buffers = [s.window_buffer for s in sources if s isa TimeVaryingSurfaceFluxSource]
    for w in 0:(nwin - 1)
        stage && SF.stage_surface_flux_window!(op, window_span(W, w)...)
        before = sum(b -> b.uploads, buffers; init = 0)
        for n in 0:(steps - 1)
            apply_surface_flux!(target, op, nothing, dt,
                                WindowTestClock(step_time(0.0, W, w, n, steps)), grid;
                                tracer_names = names, halo_width = HP, kw...)
        end
        push!(copied_during_steps, sum(b -> b.uploads, buffers; init = 0) - before)
    end
    return map(Array, state.tracers_raw), map(s -> map(Array, s.compensation), sources),
           copied_during_steps
end
