using Test
using AtmosTransport: AtmosGrid, CPU, HybridSigmaPressure, ReducedGaussianMesh
using AtmosTransport.Grids: ncells, nfaces
using AtmosTransport.Preprocessing:
    SubstepSchedulePolicy,
    initial_substeps,
    next_substeps,
    required_substeps,
    rescale_substep_amounts!
using AtmosTransport.MetDrivers:
    close_streaming_transport_binary!, open_streaming_transport_binary,
    set_transport_header_steps_per_window_schedule!

@testset "shared preprocessor substep schedule policy" begin
    policy = SubstepSchedulePolicy(
        adaptive_substeps = true,
        substep_cfl_target = 0.95,
        min_steps_per_window = 1,
        max_steps_per_window = 64)

    @test initial_substeps(policy, 4) == 4
    @test required_substeps(policy, 8, 1.233) == 11
    @test next_substeps(policy, 8, 0.25) == 8
    @test next_substeps(policy, 8, Inf) == 64

    fixed = SubstepSchedulePolicy(
        adaptive_substeps = false,
        substep_cfl_target = 0.95,
        min_steps_per_window = 1,
        max_steps_per_window = 64)
    @test next_substeps(fixed, 8, 10.0) == 8

    a = fill(2.0, 2, 2)
    b = fill(4.0, 2, 2)
    rescale_substep_amounts!((a, b), 4, 10)
    @test all(==(0.8), a)
    @test all(==(1.6), b)

    @test_throws ArgumentError SubstepSchedulePolicy(
        adaptive_substeps = true,
        substep_cfl_target = 0.0)
    @test_throws ArgumentError SubstepSchedulePolicy(
        adaptive_substeps = true,
        substep_cfl_target = 0.95,
        min_steps_per_window = 8,
        max_steps_per_window = 4)
end

@testset "transport header schedule patch keeps Poisson contract synchronized" begin
    mesh = ReducedGaussianMesh([-30.0, 30.0], [4, 4])
    vertical = HybridSigmaPressure([0.0, 0.0], [0.0, 1.0])
    grid = AtmosGrid(mesh, vertical, CPU())
    ncell = ncells(mesh)
    nface = nfaces(mesh)
    sample = (
        m = ones(ncell, 1), hflux = zeros(nface, 1),
        cm = zeros(ncell, 2), ps = fill(100_000.0, ncell),
        dhflux = zeros(nface, 1), dcm = zeros(ncell, 2), dm = zeros(ncell, 1),
    )

    mktempdir() do dir
        writer = open_streaming_transport_binary(
            joinpath(dir, "schedule.bin"), grid, 3, sample;
            steps_per_window = 4,
            source_flux_sampling = :window_start_endpoint,
            air_mass_sampling = :window_start_endpoint,
            flux_sampling = :window_constant,
            flux_kind = :substep_mass_amount,
            humidity_sampling = :none,
            delta_semantics = :forward_window_endpoint_difference,
            mass_basis = :dry)
        try
            header = writer.header
            set_transport_header_steps_per_window_schedule!(header, [4, 7, 5])
            @test header["steps_per_window"] == 7
            @test header["steps_per_window_by_window"] == [4, 7, 5]
            @test header["time_step_schedule"] == "per_window"
            @test header["poisson_balance_target_semantics"] ==
                  "forward_window_mass_difference / (2 * steps_per_window_by_window[win])"
            @test header["poisson_balance_target_scale_by_window"] ==
                  [1 / 8, 1 / 14, 1 / 10]
            @test header["poisson_balance_target_scale"] == 1 / 14
        finally
            try
                close_streaming_transport_binary!(writer)
            catch err
                err isa ArgumentError || rethrow()
            end
        end
    end
end
