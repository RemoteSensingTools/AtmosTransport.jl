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

        # Adapting a device source to CUDA again shares its buffer, as the runner
        # does for every input file; slots of a host buffer move with their
        # residency map.
        host = TimeVaryingSurfaceFluxSource(:co2, series_for(Float32, length(HOURLY), 1), HOURLY, StepwiseFlux())
        SF.stage_surface_flux_window!(host, 0.0, 7200.0)
        device = Adapt.adapt(CuArray, host)
        @test Adapt.adapt(CuArray, device).window_buffer === device.window_buffer
        @test device.window_buffer.resident == host.window_buffer.resident
        @test all(p -> Array(device.window_buffer.slots[p]) == host.window_buffer.slots[p], 1:6)
    end
end
