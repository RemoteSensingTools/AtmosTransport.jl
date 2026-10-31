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

    # The convection workspace's cell areas come from the host mesh; the
    # convection cases check that they reach the kernels as device arrays.
    _metrics(ws::AT.Operators.Convection.CMFMCMatrixWorkspace) = ws.tm5_workspace.cell_metrics
    _metrics(ws) = ws.cell_metrics

    @testset "CUDA captured preserve_vmr reset targets and replay" begin
        for (scheme, conv) in ((AT.UpwindScheme(), :none), (AT.PPMScheme(AT.MonotoneLimiter()), :none),
                               (AT.UpwindScheme(), :cmfmc), (AT.UpwindScheme(), :tm5))
            c = Case(; scheme, conv, diffusion = :split, reset = :preserve_vmr, steps = [2, 2])
            sim = _gpu_reset_simulation(c)
            inp = AT.cs_adjoint_inputs(_gpu_reset_simulation(c))
            schedule = inp.kwargs.schedule
            for targets in (schedule.window_start_air_mass, schedule.window_end_air_mass)
                @test all(p -> p isa CuArray, Iterators.flatten(targets))
            end
            @test inp.panels_m0[1] isa CuArray
            conv === :none ||
                @test all(a -> a isa CuArray, _metrics(inp.kwargs.convection_workspace))
            AT.run!(sim)
            expected = map(Array, _interior(_rm(sim)))
            replay() = first(ADJ.Footprint._cs_replay_for_call(inp.panels_rm0, inp.panels_m0,
                inp.panels_am_steps, inp.panels_bm_steps, inp.panels_cm_steps, inp.mesh;
                inp.kwargs...))
            forward = replay()
            ADJ.Footprint._replay_steps!(forward, 1:sum(c.steps))
            @test map(Array, _interior(ADJ.Footprint._replay_rm(forward))) == expected
            recorded = replay()
            tape = ADJ.Footprint._record_cs_tape!(recorded, 1:sum(c.steps), AT.FullStateTape(),
                                                  AT.Tape._open_tape_pool(AT.DeviceCSTapeStorage(), 0))
            @test map(Array, _interior(ADJ.Footprint._replay_rm(recorded))) == expected
            resets = filter(op -> op isa AT.Tape._CSAirMassResetRecord, tape)
            @test length(resets) == 3 # two window ends and one window advance
            # The record stores its reset target in a tape slot (device storage).
            @test all(op -> all(p -> p isa CuArray, op.panels_new_m.panels), resets)
            kwargs = Base.structdiff(inp.kwargs, NamedTuple{(:surface_flux, :emission_rates)})
            footprint(checkpoint) = AT.cs_surface_emission_footprint(inp.panels_rm0, inp.panels_m0,
                inp.panels_am_steps, inp.panels_bm_steps, inp.panels_cm_steps, inp.mesh,
                AT.CSLayerMeanObjective(2, 2, 3, NZ); kwargs...,
                base_emission_rates = inp.kwargs.emission_rates, tape = AT.AdjointTape(; checkpoint))
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
