"""
Host-resident time-varying surface-flux series and their device window buffer
(`FluxWindowBuffer`, `stage_surface_flux_window!`, `_flux_window_slices` in
`src/Operators/SurfaceFlux/sources.jl`).

Asserts:
- `_flux_window_slices(scheme, times, t0, t1)` contains every slice that
  `_flux_temporal_segments` references for the steps of a met window, with the
  runner's clock (exact Float64) and step length (storage float), for every
  scheme, window lengths aligned and not aligned with the slices, steps that
  end after the window in Float32, and series that start before and end inside
  the run; for stepwise and conservative-mean sources it holds at most one
  slice more than the steps read;
- once a window is staged, the operator's steps copy nothing more (the staged
  range covers what the operator reads);
- a multi-window run with the window buffer equals, bit for bit (tracers and
  Kahan compensation), the pre-R1 operator that read the full series on the
  device (kept in `test/helpers/surface_flux_window_buffer.jl` as a
  reference), for every scheme, Float32 and Float64,
  window lengths that do and do not align with slices, one to eleven steps per
  window, packed and single-tracer panels and profile deposits;
- the buffer capacity stays the largest window's slice count, independent of
  the run length, while the run reads more slices than fit in it;
- `Adapt` moves the compensation and the window buffer to the device array
  type and leaves the series and times in host memory; adapting to the
  backend a buffer is already on shares it;
- a two-binary `DrivenSimulation` stages each window when it becomes current
  (no extra copies during the steps) and equals the reference bit for bit.
"""

using Test
include(joinpath(@__DIR__, "..", "helpers", "surface_flux_window_buffer.jl"))
using AtmosTransport.MetDrivers: driver_grid, load_transport_window

const MD = AtmosTransport.MetDrivers

@testset "_flux_window_slices covers the operator's reads" begin
    for FT in (Float64, Float32), scheme in SCHEMES, times in (HOURLY, THREE_HOURLY, IRREGULAR),
        (Wd, nwin) in WINDOWS, steps in STEPS
        W = FT(Wd)
        for w in 0:(nwin - 1)
            read = slices_read(scheme, times, W, w, steps)
            staged = SF._flux_window_slices(scheme, times, window_span(W, w)...)
            @test issubset(read, staged)
            if !(scheme isa LinearInterpFlux)
                @test length(staged) <= length(read) + 1
            end
        end
    end

    # A Float32 step of 5400/7 s ends after its window; a slice time inside that
    # overshoot is split off and read (stepwise, conservative), and only the
    # padded window end includes it.
    W, steps, w = 5400.0f0, 7, 1
    t1 = window_span(W, w)[2]
    last_end = step_time(0.0, W, w, steps - 1, steps) + Float64(step_length(W, steps))
    @test last_end > t1 + 2eps(t1)
    times = [0.0, 3600.0, 0.5 * (t1 + last_end), 20000.0]
    for scheme in SCHEMES
        read = slices_read(scheme, times, W, w, steps)
        staged = SF._flux_window_slices(scheme, times, window_span(W, w)...)
        unpadded = first(staged):last(SF._slices_read_at(scheme, times, t1))
        @test issubset(read, staged)
        @test issubset(read, unpadded) == (scheme isa LinearInterpFlux)
    end

    @test SF._flux_window_slices(StepwiseFlux(), HOURLY, 7200.0, 10800.0) == 3:4
    @test SF._flux_window_slices(LinearInterpFlux(), HOURLY, 7200.0, 10800.0) == 3:5
    @test SF._flux_window_slices(ConservativeMeanFlux(), HOURLY, 9000.0, 9000.0) == 3:4
    @test SF._flux_window_slices(StepwiseFlux(), HOURLY, -7200.0, -3600.0) == 1:1   # clamped
    @test SF._flux_window_slices(LinearInterpFlux(), HOURLY, 2.0e6, 2.1e6) == 31:31
    @test_throws ArgumentError SF._flux_window_slices(StepwiseFlux(), HOURLY, 3600.0, 0.0)
end

@testset "windowed series equal the full-series operator bit for bit" begin
    for FT in (Float64, Float32), scheme in SCHEMES, (Wd, nwin) in WINDOWS, steps in STEPS,
        (times_a, times_b) in ((HOURLY, THREE_HOURLY), (IRREGULAR, HOURLY))
        W = FT(Wd)
        new_sources = (TimeVaryingSurfaceFluxSource(:co2, series_for(FT, length(times_a), 1), times_a, scheme),
                       TimeVaryingSurfaceFluxSource(:tag, series_for(FT, length(times_b), 2), times_b, scheme))
        ref_sources = map(FullSeriesReference, new_sources)
        state, grid = cs_state(FT)
        q_new, comp_new, copied = drive!(state, grid, new_sources, W, nwin, steps)
        state, grid = cs_state(FT)
        q_ref, comp_ref, _ = drive!(state, grid, ref_sources, W, nwin, steps)
        @test bitequal(q_new, q_ref)
        @test all(map(bitequal, comp_new, comp_ref))
        @test all(iszero, copied)                  # staging covered every step
    end

    # Without staging (direct operator calls, as in budget scripts), the
    # operator stages each step's slices itself: same result. Single-tracer
    # panels, as the per-tracer cubed-sphere advection passes them.
    for FT in (Float64, Float32), scheme in SCHEMES
        W = FT(5400)
        src = TimeVaryingSurfaceFluxSource(:co2, series_for(FT, length(HOURLY), 3), HOURLY, scheme)
        state, grid = cs_state(FT; packed = false)
        q_new, comp_new, _ = drive!(state, grid, (src,), W, 21, 7; stage = false, single = true)
        state, grid = cs_state(FT; packed = false)
        q_ref, comp_ref, _ = drive!(state, grid, (FullSeriesReference(src),), W, 21, 7; single = true)
        @test bitequal(q_new, q_ref) && all(map(bitequal, comp_new, comp_ref))
        @test SF._capacity(src.window_buffer) <= 3        # one step's slices at a time
    end

    # Profile deposit (GEOS-Chem non-local VDIFF) and a static source next to
    # a time-varying one.
    for FT in (Float64, Float32), scheme in SCHEMES
        W = FT(3600)
        profile = ntuple(p -> SF.ProfileDeposit(FT[k / 6 for i in 1:NC, j in 1:NC, k in 1:NZ]), 6)
        static() = SurfaceFluxSource(:tag, map(s -> s[:, :, 1], series_for(FT, 1, 4)))   # own compensation
        src = TimeVaryingSurfaceFluxSource(:co2, series_for(FT, length(THREE_HOURLY), 5),
                                           THREE_HOURLY, scheme)
        state, grid = cs_state(FT)
        q_new, _, _ = drive!(state, grid, (src, static()), W, 24, 11; deposit = profile)
        state, grid = cs_state(FT)
        q_ref, _, _ = drive!(state, grid, (FullSeriesReference(src), static()), W, 24, 11;
                             deposit = profile)
        @test bitequal(q_new, q_ref)
    end
end

@testset "buffer capacity is the largest window's slice count, not the run length" begin
    for scheme in SCHEMES
        W = 3600.0
        capacities = map((2, 8, 32)) do nwin
            src = TimeVaryingSurfaceFluxSource(:co2, series_for(Float64, length(HOURLY), 1), HOURLY, scheme)
            state, grid = cs_state(Float64; packed = false)
            drive!(state, grid, (src,), W, nwin, 4)
            buffer = src.window_buffer
            @test SF._capacity(buffer) == size(buffer.slots[1], 3)
            @test SF._capacity(buffer) == maximum(w -> length(SF._flux_window_slices(scheme, HOURLY, window_span(W, w)...)),
                                                 0:(nwin - 1))
            (SF._capacity(buffer), buffer.uploads)
        end
        @test first(capacities[1]) == first(capacities[2]) == first(capacities[3]) <= 3
        # The 32-window run reads all 31 slices through a buffer of at most 3.
        @test last(capacities[3]) >= length(HOURLY)
        # A slice that two windows read is copied once.
        @test last(capacities[3]) <= length(HOURLY) + 1
    end

    # The buffer starts empty: a source that never emits holds no slots.
    src = TimeVaryingSurfaceFluxSource(:co2, series_for(Float32, 5, 1), collect(0.0:3600.0:14400.0),
                                       StepwiseFlux())
    @test SF._capacity(src.window_buffer) == 0 && size(src.window_buffer.slots[1]) == (NC, NC, 0)
    @test eltype(src.window_buffer.slots[1]) === Float32
    # Staging a longer range grows the slots once and keeps the backend.
    SF.stage_surface_flux_window!(src, 0.0, 3 * 3600.0)
    @test SF._capacity(src.window_buffer) == 4
    @test all(p -> src.window_buffer.slots[p][:, :, SF._buffer_slot(src.window_buffer, 3)] ==
                   src.cell_mass_rate_series[p][:, :, 3], 1:6)
    SF.stage_surface_flux_window!(src, 3600.0, 2 * 3600.0)
    @test SF._capacity(src.window_buffer) == 4 && src.window_buffer.uploads == 4
end

# A device stand-in on the CPU: its own array type (so Adapt must move data
# into it), elementwise writes, and `similar` that stays on the "device".
struct DeviceStandIn{T, N} <: AbstractArray{T, N}
    data :: Array{T, N}
end
Base.size(a::DeviceStandIn) = size(a.data)
Base.getindex(a::DeviceStandIn{T, N}, i::Vararg{Int, N}) where {T, N} = a.data[i...]
Base.setindex!(a::DeviceStandIn{T, N}, v, i::Vararg{Int, N}) where {T, N} = (a.data[i...] = v)
Base.similar(a::DeviceStandIn, ::Type{T}, dims::Dims) where {T} = DeviceStandIn(similar(a.data, T, dims))
struct ToDeviceStandIn end
Adapt.adapt_storage(::ToDeviceStandIn, x::Array) = DeviceStandIn(copy(x))

const BULK_STAGE_CALLS = Ref(0)
function SF._bulk_copy_flux_slices!(dest::DeviceStandIn, copies::AbstractVector)
    BULK_STAGE_CALLS[] += 1
    @test all(copy -> SF._first_flux_array(copy.buffer.slots) isa DeviceStandIn,
              copies)
    return invoke(SF._bulk_copy_flux_slices!, Tuple{AbstractArray, AbstractVector},
                  dest, copies)
end

@testset "Adapt keeps the series on the host and moves the window buffer" begin
    FT = Float32
    host = TimeVaryingSurfaceFluxSource(:co2, series_for(FT, length(HOURLY), 1), HOURLY, ConservativeMeanFlux())
    device = Adapt.adapt(ToDeviceStandIn(), host)
    @test device.cell_mass_rate_series === host.cell_mass_rate_series     # no device copy
    @test device.times === host.times && device.scheme === host.scheme
    @test device.compensation[1] isa DeviceStandIn{FT, 2}
    @test device.window_buffer.slots[1] isa DeviceStandIn{FT, 3}
    @test device.window_buffer !== host.window_buffer
    # Adapting to the backend the buffer is already on is an idempotent direct
    # API operation; on the CPU, `Array` is that backend.
    @test Adapt.adapt(ToDeviceStandIn(), device).window_buffer === device.window_buffer
    @test Adapt.adapt(Array, host).window_buffer === host.window_buffer
    @test Adapt.adapt(Array, host).compensation === host.compensation

    # Staging copies host slices into the device slots; the operator then
    # gives the host source's result.
    state, grid = cs_state(FT; packed = false)
    q_device, _, copied = drive!(state, grid, (device,), FT(5400), 21, 7)
    @test all(iszero, copied)
    @test device.window_buffer.slots[1] isa DeviceStandIn{FT, 3}
    @test SF._capacity(device.window_buffer) <= 4
    state, grid = cs_state(FT; packed = false)
    q_host, _, _ = drive!(state, grid, (host,), FT(5400), 21, 7)
    @test bitequal(q_device, q_host)
    @test SF._capacity(host.window_buffer) == SF._capacity(device.window_buffer)

    # A buffer adapted with resident slices keeps them valid on the new backend.
    moved = Adapt.adapt(ToDeviceStandIn(), host)
    @test moved.window_buffer.resident == host.window_buffer.resident
    @test all(p -> moved.window_buffer.slots[p].data == host.window_buffer.slots[p], 1:6)
    @test moved.window_buffer.uploads == host.window_buffer.uploads > 0

    # Missing slices are partitioned by destination dispatch group: the two
    # stand-in sources share one bulk call even with a CPU source between them,
    # and that call never receives the CPU copy. A fully resident repeat invokes
    # no stand-in hook.
    bulk_hosts = (
        TimeVaryingSurfaceFluxSource(:co2, series_for(FT, length(HOURLY), 8), HOURLY,
                                     StepwiseFlux()),
        TimeVaryingSurfaceFluxSource(:tag, series_for(FT, length(HOURLY), 9), HOURLY,
                                     LinearInterpFlux()))
    bulk_devices = map(s -> Adapt.adapt(ToDeviceStandIn(), s), bulk_hosts)
    cpu_source = TimeVaryingSurfaceFluxSource(
        :ch4, series_for(FT, length(HOURLY), 10), HOURLY, ConservativeMeanFlux())
    bulk_op = SurfaceFluxOperator(PerTracerFluxMap((bulk_devices[1], cpu_source,
                                                    bulk_devices[2])))
    BULK_STAGE_CALLS[] = 0
    SF.stage_surface_flux_window!(bulk_op, 0.0, 7200.0)
    @test BULK_STAGE_CALLS[] == 1
    @test sum(s -> s.window_buffer.uploads, bulk_devices) > 2
    @test cpu_source.window_buffer.uploads > 0
    SF.stage_surface_flux_window!(bulk_op, 0.0, 7200.0)
    @test BULK_STAGE_CALLS[] == 1
end

# ---------------------------------------------------------------------------
# DrivenSimulation over two cubed-sphere binaries (the runner's per-file loop).
# ---------------------------------------------------------------------------
function write_emission_binary(path; FT, nwin, steps, Nc = 3, Nz = 4)
    vc = HybridSigmaPressure(fill(FT(100), Nz + 1), FT.(range(0, 1; length = Nz + 1)))
    writer = MD.open_streaming_cs_transport_binary(path, Nc, 6, Nz, nwin, vc;
        planet_radius = AtmosTransport.Parameters.EARTH_RADIUS, FT,
        dt_met_seconds = 3600.0, steps_per_window = steps, mass_basis = :dry,
        surface_pressure_basis = TotalSurfacePressure())
    try
        for _ in 1:nwin
            MD.write_streaming_cs_window!(writer, (;
                m = ntuple(_ -> fill(FT(1e16), Nc, Nc, Nz), 6),
                am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), 6),
                bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), 6),
                cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6),
                ps = ntuple(_ -> fill(FT(100000), Nc, Nc), 6)), Nc, 6)
        end
    finally
        MD.close_streaming_transport_binary!(writer)
    end
end

function run_two_binaries(paths, sources; FT, nwin)
    driver = TransportBinaryDriver(paths[1]; FT, arch = AtmosTransport.CPU(), Hp = 1)
    grid = driver_grid(driver)
    air = map(copy, load_transport_window(driver, 1).air_mass)
    state = CubedSphereState(DryBasis, grid.horizontal, air;
                             co2 = map(a -> a .* FT(4e-4), air), tag = map(a -> a .* FT(2e-6), air),
                             ch4 = map(a -> a .* FT(1.9e-6), air))
    fluxes = allocate_face_fluxes(grid.horizontal, size(air[1], 3); FT, basis = DryBasis)
    model = TransportModel(state, fluxes, grid, UpwindScheme())
    staged_at_start = Bool[]
    for (b, path) in enumerate(paths)
        driver = b == 1 ? driver : TransportBinaryDriver(path; FT, arch = AtmosTransport.CPU(), Hp = 1)
        start_time = (b - 1) * nwin * 3600.0
        sim = DrivenSimulation(model, driver; initialize_air_mass = b == 1,
                               surface_sources = sources, start_time)
        # The first window is staged when the simulation is built, before any step.
        for src in sources
            src isa TimeVaryingSurfaceFluxSource || continue
            ks = SF._flux_window_slices(src.scheme, src.times, start_time, start_time + 3600.0)
            push!(staged_at_start, all(k -> src.window_buffer.resident[SF._buffer_slot(src.window_buffer, k)] == k, ks))
        end
        run!(sim)
        model = sim.model
        close(driver)
    end
    return model, staged_at_start
end

@testset "DrivenSimulation stages each window and equals the full-series operator" begin
    mktempdir() do dir
        for FT in (Float64, Float32), steps in (2, 7)
            nwin = 5
            paths = [joinpath(dir, "day$(b)_$(FT)_$(steps).bin") for b in 1:2]
            foreach(p -> write_emission_binary(p; FT, nwin, steps), paths)
            half_hourly = collect(0.0:1800.0:(12 * 3600.0))
            three_hourly = collect(-5400.0:10800.0:(12 * 3600.0))
            # Rates of 1e8 per cell and second change the 1e12-1e13 tracer
            # masses at every step, in Float32 too.
            make_sources() = (
                TimeVaryingSurfaceFluxSource(:co2, series_for(FT, length(half_hourly), 1; nc = 3, scale = 1e8),
                                             half_hourly, StepwiseFlux()),
                TimeVaryingSurfaceFluxSource(:tag, series_for(FT, length(three_hourly), 2; nc = 3, scale = 1e8),
                                             three_hourly, ConservativeMeanFlux()),
                TimeVaryingSurfaceFluxSource(:ch4, series_for(FT, length(half_hourly), 3; nc = 3, scale = 1e8),
                                             half_hourly, LinearInterpFlux()))
            sources = make_sources()
            model_new, staged = run_two_binaries(paths, sources; FT, nwin)
            @test all(staged)
            ref_sources = map(FullSeriesReference, make_sources())
            model_ref, _ = run_two_binaries(paths, ref_sources; FT, nwin)
            @test bitequal(model_new.state.tracers_raw, model_ref.state.tracers_raw)
            @test all(map((a, b) -> bitequal(a.compensation, b.compensation), sources, ref_sources))
            model_none, _ = run_two_binaries(paths, (); FT, nwin)
            for t in 1:3        # every source emits
                @test !bitequal(map(q -> q[:, :, :, t], model_new.state.tracers_raw),
                                map(q -> q[:, :, :, t], model_none.state.tracers_raw))
            end

            # The steps copied nothing beyond the per-window staging: replaying
            # only the staging of the ten windows gives the same copies.
            for src in sources
                replay = SF.FluxWindowBuffer(src.cell_mass_rate_series)
                for w in 0:(2nwin - 1)
                    SF._stage_flux_slices!(replay, src.cell_mass_rate_series,
                                           SF._flux_window_slices(src.scheme, src.times, window_span(3600.0, w)...))
                end
                @test src.window_buffer.uploads == replay.uploads
                @test src.window_buffer.resident == replay.resident
                @test SF._capacity(src.window_buffer) ==
                      maximum(w -> length(SF._flux_window_slices(src.scheme, src.times, window_span(3600.0, w)...)),
                              0:(2nwin - 1)) <= 4
            end
        end
    end
end
