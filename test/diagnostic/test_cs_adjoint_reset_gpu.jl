# Opt in with CUDA_VISIBLE_DEVICES explicitly selected, in the root project:
# ATMOSTR_RUN_CS_ADJOINT_RESET_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=1 \
# julia --project=. test/diagnostic/test_cs_adjoint_reset_gpu.jl
using Test

if get(ENV, "ATMOSTR_RUN_CS_ADJOINT_RESET_GPU_TESTS", "0") != "1"
    @info "Skipping opt-in CS adjoint preserve_vmr GPU tests"
else
    using CUDA
    using AtmosTransport
    using Adapt
    CUDA.functional(true) || error("CUDA is unavailable")
    CUDA.allowscalar(false)
    include(joinpath(@__DIR__, "..", "helpers", "cs_adjoint_step_plan.jl"))

    function _gpu_reset_simulation(c)
        cpu = _simulation(c)
        model = Adapt.adapt(CuArray, cpu.model)
        return AT.DrivenSimulation(model, cpu.driver; physics_cadence = c.cadence,
                                   air_mass_reset_mode = c.reset, prefetch_windows = false)
    end

    @testset "CUDA captured preserve_vmr reset targets and replay" begin
        for scheme in (AT.UpwindScheme(), AT.PPMScheme(AT.MonotoneLimiter()))
            c = Case(; scheme, diffusion = :split, reset = :preserve_vmr, steps = [2, 2])
            sim = _gpu_reset_simulation(c)
            inp = AT.cs_adjoint_inputs(_gpu_reset_simulation(c))
            schedule = inp.kwargs.schedule
            for targets in (schedule.window_start_air_mass, schedule.window_end_air_mass)
                @test all(p -> p isa CuArray, Iterators.flatten(targets))
            end
            @test inp.panels_m0[1] isa CuArray
            AT.run!(sim)
            expected = map(Array, _interior(_rm(sim)))
            replay() = first(ADJ._cs_replay_for_call(inp.panels_rm0, inp.panels_m0,
                inp.panels_am_steps, inp.panels_bm_steps, inp.panels_cm_steps, inp.mesh;
                inp.kwargs...))
            forward = replay()
            ADJ._replay_steps!(forward, 1:sum(c.steps))
            @test map(Array, _interior(ADJ._replay_rm(forward))) == expected
            recorded = replay()
            tape = ADJ._record_cs_tape!(recorded, 1:sum(c.steps), AT.DeviceCSTapeStorage())
            @test map(Array, _interior(ADJ._replay_rm(recorded))) == expected
            resets = filter(op -> op isa AT.Tape._CSAirMassResetRecord, tape)
            @test length(resets) == 3 # two window ends and one window advance
            @test all(op -> all(p -> p isa CuArray, op.panels_new_m), resets)
            kwargs = Base.structdiff(inp.kwargs, NamedTuple{(:surface_flux, :emission_rates)})
            footprint(checkpoint) = AT.cs_surface_emission_footprint(inp.panels_rm0, inp.panels_m0,
                inp.panels_am_steps, inp.panels_bm_steps, inp.panels_cm_steps, inp.mesh,
                AT.CSLayerMeanObjective(2, 2, 3, NZ); kwargs...,
                base_emission_rates = inp.kwargs.emission_rates, checkpoint)
            full = footprint(AT.FullCheckpoint())
            stride = footprint(AT.StrideCheckpoint(2))
            @test full.initial_adjoint[1] isa CuArray
            @test all(isapprox(Array(full.initial_adjoint[p]), Array(stride.initial_adjoint[p]);
                               rtol = 1e-13) for p in 1:6)
            @test all(isapprox(Array(full.footprints[s][p]), Array(stride.footprints[s][p]);
                               rtol = 1e-13) for s in eachindex(full.footprints), p in 1:6)
        end
    end
end
