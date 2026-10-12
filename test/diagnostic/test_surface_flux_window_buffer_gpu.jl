# Time-varying surface fluxes on CUDA: the series stays in host memory, the
# device window buffer is filled by host-to-device copies on the task's stream,
# and the result equals, bit for bit, the pre-R1 operator that read the whole
# series on the device. Set CUDA_VISIBLE_DEVICES explicitly and opt in before
# running:
#
#   CUDA_VISIBLE_DEVICES=<id> ATMOSTR_RUN_SURFACE_FLUX_GPU_TESTS=1 \
#       julia --project=. test/diagnostic/test_surface_flux_window_buffer_gpu.jl
using Test
if get(ENV, "ATMOSTR_RUN_SURFACE_FLUX_GPU_TESTS", "0") != "1"
    @info "Skipping opt-in time-varying surface-flux GPU tests"
else
    using CUDA, Adapt
    using AtmosTransport
    CUDA.allowscalar(false)
    include(joinpath(@__DIR__, "..", "helpers", "surface_flux_window_buffer.jl"))

    function write_cuda_emission_binary(path; FT, nwin, steps, Nc = 3, Nz = 4)
        vc = HybridSigmaPressure(fill(FT(100), Nz + 1),
                                 FT.(range(0, 1; length = Nz + 1)))
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

    function cuda_model_for(path; FT)
        driver = TransportBinaryDriver(path; FT, arch = AtmosTransport.CPU(), Hp = 1)
        try
            grid = MD.driver_grid(driver)
            air = map(copy, MD.load_transport_window(driver, 1).air_mass)
            state = CubedSphereState(DryBasis, grid.horizontal, air;
                co2 = map(a -> a .* FT(4e-4), air),
                tag = map(a -> a .* FT(2e-6), air),
                ch4 = map(a -> a .* FT(1.9e-6), air))
            fluxes = allocate_face_fluxes(grid.horizontal, size(air[1], 3);
                                          FT, basis = DryBasis)
            return Adapt.adapt(CuArray,
                TransportModel(state, fluxes, grid, UpwindScheme()))
        finally
            close(driver)
        end
    end

    function run_two_cuda_binaries(paths, host_sources; FT, nwin)
        model = cuda_model_for(first(paths); FT)
        compensation_at_start = Any[]
        compensation_at_end = Any[]
        buffer_stats = Any[]
        for (b, path) in enumerate(paths)
            driver = TransportBinaryDriver(path; FT, arch = AtmosTransport.GPU(:cuda), Hp = 1)
            try
                start_time = (b - 1) * nwin * 3600.0
                sim = DrivenSimulation(model, driver; initialize_air_mass = b == 1,
                                       surface_sources = host_sources, start_time)
                adapted = sim.model.emissions.flux_map.sources
                push!(compensation_at_start,
                      map(s -> map(Array, s.compensation), adapted))
                run!(sim)
                push!(compensation_at_end,
                      map(s -> map(Array, s.compensation), adapted))
                if all(s -> s isa TimeVaryingSurfaceFluxSource, adapted)
                    push!(buffer_stats, map(adapted) do s
                        (capacity = SF._capacity(s.window_buffer),
                         resident = copy(s.window_buffer.resident),
                         uploads = s.window_buffer.uploads,
                         host_series = s.cell_mass_rate_series ===
                                       host_sources[findfirst(==(s.tracer_name),
                                                              map(x -> x.tracer_name, host_sources))].cell_mass_rate_series)
                    end)
                end
                model = sim.model
            finally
                close(driver)
            end
        end
        return model, compensation_at_start, compensation_at_end, buffer_stats
    end

    @testset "CUDA window buffer equals the full device series" begin
        for FT in (Float32, Float64), scheme in SCHEMES,
            (Wd, nwin) in ((3600.0, 32), (5400.0, 21), (10800.0, 11)), steps in (1, 7)
            W = FT(Wd)
            hosts = (TimeVaryingSurfaceFluxSource(:co2, series_for(FT, length(HOURLY), 1), HOURLY, scheme),
                     TimeVaryingSurfaceFluxSource(:tag, series_for(FT, length(THREE_HOURLY), 2),
                                                  THREE_HOURLY, scheme))
            devices = map(h -> Adapt.adapt(CuArray, h), hosts)
            @test all(map((d, h) -> d.cell_mass_rate_series === h.cell_mass_rate_series, devices, hosts))
            @test all(d -> d.window_buffer.slots[1] isa CuArray{FT, 3}, devices)
            @test all(d -> d.compensation[1] isa CuArray{FT, 2}, devices)

            state, grid = cs_state(FT; arr = CuArray)
            q_new, comp_new, copied = drive!(state, grid, devices, W, nwin, steps)
            state, grid = cs_state(FT; arr = CuArray)
            q_ref, comp_ref, _ = drive!(state, grid, map(h -> FullSeriesReference(h, CuArray), hosts),
                                        W, nwin, steps)
            @test bitequal(q_new, q_ref)
            @test all(map(bitequal, comp_new, comp_ref))
            @test all(iszero, copied)                      # staging covered every step
            # The capacity is the largest window's slice range.
            @test all(d -> SF._capacity(d.window_buffer) ==
                           maximum(w -> length(SF._flux_window_slices(d.scheme, d.times, window_span(W, w)...)),
                                   0:(nwin - 1)), devices)

            # The same operator on the CPU agrees to rounding (the two backends
            # need not round alike).
            state, grid = cs_state(FT)
            q_cpu, _, _ = drive!(state, grid, hosts, W, nwin, steps)
            rtol = FT === Float32 ? 1e-6 : 1e-14
            @test all(map((a, b) -> isapprox(a, b; rtol), q_new, q_cpu))
        end

        # Unstaged single-tracer panels (the operator stages each step itself)
        # and the GEOS-Chem profile deposit next to a static source.
        for FT in (Float32, Float64), scheme in SCHEMES
            host = TimeVaryingSurfaceFluxSource(:co2, series_for(FT, length(HOURLY), 3), HOURLY, scheme)
            device = Adapt.adapt(CuArray, host)
            state, grid = cs_state(FT; packed = false, arr = CuArray)
            q_new, _, _ = drive!(state, grid, (device,), FT(5400), 21, 7; stage = false, single = true)
            state, grid = cs_state(FT; packed = false, arr = CuArray)
            q_ref, _, _ = drive!(state, grid, (FullSeriesReference(host, CuArray),), FT(5400), 21, 7;
                                 single = true)
            @test bitequal(q_new, q_ref)

            profile = ntuple(p -> SF.ProfileDeposit(CuArray(FT[k / 6 for i in 1:NC, j in 1:NC, k in 1:NZ])), 6)
            static() = Adapt.adapt(CuArray, SurfaceFluxSource(:tag, map(s -> s[:, :, 1], series_for(FT, 1, 4))))
            host = TimeVaryingSurfaceFluxSource(:co2, series_for(FT, length(THREE_HOURLY), 5), THREE_HOURLY, scheme)
            state, grid = cs_state(FT; arr = CuArray)
            q_new, _, _ = drive!(state, grid, (Adapt.adapt(CuArray, host), static()), FT(3600), 24, 11;
                                 deposit = profile)
            state, grid = cs_state(FT; arr = CuArray)
            q_ref, _, _ = drive!(state, grid, (FullSeriesReference(host, CuArray), static()), FT(3600), 24, 11;
                                 deposit = profile)
            @test bitequal(q_new, q_ref)
        end

        # Directly adapting a device source to CUDA again shares its buffer;
        # separately, slots of a populated host buffer move with their residency
        # map and lifetime upload count.
        host = TimeVaryingSurfaceFluxSource(:co2, series_for(Float32, length(HOURLY), 1), HOURLY, StepwiseFlux())
        SF.stage_surface_flux_window!(host, 0.0, 7200.0)
        device = Adapt.adapt(CuArray, host)
        @test Adapt.adapt(CuArray, device).window_buffer === device.window_buffer
        @test device.window_buffer.resident == host.window_buffer.resident
        @test device.window_buffer.uploads == host.window_buffer.uploads
        @test all(p -> Array(device.window_buffer.slots[p]) == host.window_buffer.slots[p], 1:6)

        # Match the runner's per-binary lifecycle: every DrivenSimulation is
        # constructed from the same host tuple, creating a fresh CUDA window
        # buffer and compensation. Compare both files with a fresh full-series
        # device reference at the same boundary.
        mktempdir() do dir
            FT, nwin, steps = Float32, 4, 7
            paths = [joinpath(dir, "cuda_part$(b).bin") for b in 1:2]
            foreach(p -> write_cuda_emission_binary(p; FT, nwin, steps), paths)
            times = collect(0.0:1800.0:(10 * 3600.0))
            make_hosts() = (
                TimeVaryingSurfaceFluxSource(:co2,
                    series_for(FT, length(times), 31; nc = 3, scale = 1e8),
                    times, StepwiseFlux()),
                TimeVaryingSurfaceFluxSource(:tag,
                    series_for(FT, length(times), 32; nc = 3, scale = 1e8),
                    times, ConservativeMeanFlux()),
                TimeVaryingSurfaceFluxSource(:ch4,
                    series_for(FT, length(times), 33; nc = 3, scale = 1e8),
                    times, LinearInterpFlux()))
            hosts = make_hosts()
            model_new, start_new, end_new, stats =
                run_two_cuda_binaries(paths, hosts; FT, nwin)
            refs = map(FullSeriesReference, make_hosts())
            model_ref, start_ref, end_ref, _ =
                run_two_cuda_binaries(paths, refs; FT, nwin)

            @test bitequal(map(Array, model_new.state.tracers_raw),
                           map(Array, model_ref.state.tracers_raw))
            @test all(i -> all(map(bitequal, end_new[i], end_ref[i])), eachindex(end_new))
            @test all(run -> all(comp -> all(panel -> all(iszero, panel), comp), run),
                      start_new)
            @test all(run -> all(comp -> all(panel -> all(iszero, panel), comp), run),
                      start_ref)
            @test length(stats) == 2
            @test all(run -> all(s -> s.capacity <= 4 && s.capacity > 0 &&
                                      s.uploads > 0 && any(!iszero, s.resident) && s.host_series,
                                  run), stats)
            @test all(s -> SF._capacity(s.window_buffer) == 0 &&
                           s.window_buffer.uploads == 0 &&
                           all(panel -> all(iszero, panel), s.compensation),
                      hosts)
        end
    end
end
