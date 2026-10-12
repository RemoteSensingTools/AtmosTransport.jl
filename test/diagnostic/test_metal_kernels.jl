# Opt-in Metal (Apple GPU, Float32) checks of the point operations, the
# cubed-sphere halo exchange and the CMFMC CFL scan against host references.
# On an Apple-silicon Mac, from the repository root:
#   ATMOSTR_RUN_METAL_TESTS=1 julia --project=<env with Metal + this package> test/diagnostic/test_metal_kernels.jl
using Test
if get(ENV, "ATMOSTR_RUN_METAL_TESTS", "0") != "1"
    @info "Skipping opt-in Metal kernel tests"
else
    using Metal              # before AtmosTransport, so its Metal extension loads
    @assert Metal.functional() "Metal is not functional on this machine"
    Metal.allowscalar(false)
    helpers = joinpath(@__DIR__, "..", "helpers")
    include(joinpath(helpers, "point_ops.jl"))
    include(joinpath(helpers, "cs_halo_fill.jl"))
    include(joinpath(helpers, "cmfmc_cfl.jl"))
    include(joinpath(helpers, "surface_flux_window_buffer.jl"))
    @testset "Metal point operations" begin
        # Separate launches with bound contexts on Metal (faster there).
        @test PointArch.fusion_policy(Metal.MetalBackend()) === PointArch.SeparateLaunches()
        check_point_ops_on_device(MtlArray)
    end
    @testset "Metal halo exchange matches the reference" begin
        results = check_halo_fill(library_fill!, library_corners!, MtlArray;
                                  cases = ((Float32, 3), (Float32, 4)))
        @test length(results) == 32
        @test all(results)
    end
    @testset "Metal CMFMC CFL scan matches the host reference" begin
        check_cmfmc_cfl_on_device(MtlArray; float_types = (Float32,))
    end
    @testset "Metal surface-flux window staging matches the full series" begin
        for scheme in SCHEMES
            hosts = (
                TimeVaryingSurfaceFluxSource(:co2,
                    series_for(Float32, length(HOURLY), 21), HOURLY, scheme),
                TimeVaryingSurfaceFluxSource(:tag,
                    series_for(Float32, length(THREE_HOURLY), 22), THREE_HOURLY, scheme))
            devices = map(h -> Adapt.adapt(MtlArray, h), hosts)
            state, grid = cs_state(Float32; arr = MtlArray)
            q_new, comp_new, copied = drive!(state, grid, devices, 5400.0f0, 8, 7)
            state, grid = cs_state(Float32; arr = MtlArray)
            refs = map(h -> FullSeriesReference(h, MtlArray), hosts)
            q_ref, comp_ref, _ = drive!(state, grid, refs, 5400.0f0, 8, 7)
            @test bitequal(q_new, q_ref)
            @test all(map(bitequal, comp_new, comp_ref))
            @test all(iszero, copied)
        end

        # One Metal dispatch group may mix the optimized dense host path with
        # generic copies. Device-resident series and host SubArrays must stay on
        # `_copy_slice!`; neither is valid input to the unsafe Array-to-MtlArray
        # transfer primitive.
        dense_arrays = series_for(Float32, length(HOURLY), 31)
        dense = Adapt.adapt(MtlArray,
            TimeVaryingSurfaceFluxSource(:co2, dense_arrays, HOURLY,
                                         StepwiseFlux()))
        view_arrays = map(a -> view(a, :, :, :),
                          series_for(Float32, length(HOURLY), 32))
        views = Adapt.adapt(MtlArray,
            TimeVaryingSurfaceFluxSource(:tag, view_arrays, HOURLY,
                                         StepwiseFlux()))
        device_arrays = map(MtlArray,
                            series_for(Float32, length(HOURLY), 33))
        resident = TimeVaryingSurfaceFluxSource(:ch4, device_arrays, HOURLY,
                                                StepwiseFlux())
        mixed = (dense, views, resident)
        SF.stage_surface_flux_window!(
            SurfaceFluxOperator(PerTracerFluxMap(mixed)), 0.0, 7200.0)

        @test views.cell_mass_rate_series === view_arrays
        @test all(a -> a isa SubArray, views.cell_mass_rate_series)
        @test all(a -> a isa MtlArray, resident.cell_mass_rate_series)
        for source in mixed, k in 1:3, panel in 1:6
            slot = SF._buffer_slot(source.window_buffer, k)
            staged = Array(source.window_buffer.slots[panel])[:, :, slot]
            expected = Array(source.cell_mass_rate_series[panel])[:, :, k]
            @test staged == expected
        end
    end
end
