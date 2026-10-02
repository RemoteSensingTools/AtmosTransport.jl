using Adapt
using AtmosChemistry
using AtmosTransport
using CUDA
using Test

@testset "AtmosChemistry backend-resident transport coupling" begin
    if !CUDA.functional()
        @test_skip "CUDA is unavailable"
    else
        for FT in (Float64, Float32)
            species = (
                Species(:A; molar_mass = 10, atoms = (:X => 1,)),
                Species(:B; molar_mass = 10, atoms = (:X => 1,)),
            )
            reactions = (
                Reaction((:A => 1,), (:B => 1,), ConstantRate(1 // 1_000)),
            )
            architecture = NativeGPU(
                CUDA.CUDABackend(), FixedWorkgroup(64),
                TabulatedReactions(UniformEffectiveRates()), CellTeam(4))
            compiled = compile_mechanism(
                Mechanism(species, reactions); architecture, precision = FT)
            solver = Rodas3(NativeSolver(SparseLinearSolver()))
            policy = SolverPolicy(FT;
                abstol = FT(1e-8), reltol = FT(1e-7), positivity = :allow,
                allow_uncertified_float32 = FT === Float32)
            chemistry = ChemistryModel(
                compiled; solver, policy, architecture)
            forcing = ChemistryForcing(FT;
                temperature = FT(298), air_number_density = FT(2.5e19))
            operator = AtmosChemistryOperator(chemistry, forcing)
            @test operator.dry_air_molar_mass isa FT

            air_mass = reshape(FT[1, 2], 2, 1)
            state_cpu = CellState(DryBasis, air_mass;
                A = reshape(FT[0.1, 0.2], 2, 1),
                B = zeros(FT, 2, 1),
                inert = fill(FT(3), 2, 1))
            state = Adapt.adapt(CUDA.CuArray, state_cpu)
            workspace = AtmosTransport.Models._chemistry_workspace_for(
                operator, state, nothing)
            @test workspace.concentrations isa CUDA.CuArray{FT, 2}
            @test workspace.tracer_indices isa CUDA.CuArray{Int32, 1}
            @test workspace.molar_masses isa CUDA.CuArray{FT, 1}
            @test workspace.chemistry.state_scratch isa CUDA.CuArray
            @test workspace.host_forcing_stage === nothing
            plan = chemistry_workspace_plan(workspace)
            measured = chemistry_workspace_storage_bytes(workspace)
            @test measured.total == plan.required_backend_bytes +
                  plan.host_persistent_bytes

            apply!(state, nothing, nothing, operator, FT(100); workspace)
            a = Array(state.tracers.A)
            b = Array(state.tracers.B)
            tolerance = FT === Float32 ? FT(2e-5) : FT(2e-7)
            @test vec(a) ≈ FT[0.1, 0.2] .* exp(FT(-0.1)) rtol = tolerance
            @test a .+ b ≈ state_cpu.tracers.A rtol = tolerance
            @test Array(state.tracers.inert) == state_cpu.tracers.inert
            @test all(==(Int8(1)), workspace.diagnostics.status)

            bad_air = reshape(FT[0, 2], 2, 1)
            bad_cpu = CellState(DryBasis, bad_air;
                A = reshape(FT[0.1, 0.2], 2, 1),
                B = zeros(FT, 2, 1))
            bad_state = Adapt.adapt(CUDA.CuArray, bad_cpu)
            bad_workspace = AtmosTransport.Models._chemistry_workspace_for(
                operator, bad_state, nothing)
            before = Array(bad_state.tracers_raw)
            @test_throws ArgumentError apply!(
                bad_state, nothing, nothing, operator, FT(100);
                workspace = bad_workspace)
            @test Array(bad_state.tracers_raw) == before
        end

        @testset "selected species map into a larger tracer state" begin
            FT = Float64
            species = (
                Species(:A; molar_mass = 10, atoms = (:X => 1,)),
                Species(:B; molar_mass = 10, atoms = (:X => 1,)),
                Species(:C; molar_mass = 20, atoms = (:Y => 1,)),
            )
            reactions = (
                Reaction((:A => 1, :C => 1),
                         (:B => 1, :C => 1), ConstantRate(1e-15)),
            )
            architecture = NativeGPU(
                CUDA.CUDABackend(), FixedWorkgroup(64),
                TabulatedReactions(UniformEffectiveRates()), CellTeam(4))
            compiled = compile_mechanism(
                Mechanism(species, reactions);
                selection = SelectedSpecies((:A, :B);
                                            closure = :fixed_background),
                architecture, precision = FT)
            @test species_names(compiled) == (:A, :B)
            @test required_forcings(compiled).fixed_species == (:C,)

            policy = SolverPolicy(FT;
                abstol = 1e-10, reltol = 1e-9, positivity = :allow)
            chemistry = ChemistryModel(compiled;
                solver = Rodas3(NativeSolver(SparseLinearSolver())),
                policy, architecture)
            forcing = ChemistryForcing(FT;
                temperature = 298, air_number_density = 2.5e19,
                fixed_species = (C = 1e12, EXTRA = 9e12))
            operator = AtmosChemistryOperator(chemistry, forcing)

            cell_count = 7
            initial_a = FT.(range(0.05, 0.11; length = cell_count))
            initial_b = FT.(range(0.01, 0.04; length = cell_count))
            state_cpu = CellState(DryBasis, ones(FT, cell_count, 1);
                inert = fill(FT(3), cell_count, 1),
                B = reshape(initial_b, cell_count, 1),
                A = reshape(initial_a, cell_count, 1),
                C = fill(FT(0.7), cell_count, 1))
            state = Adapt.adapt(CUDA.CuArray, state_cpu)
            workspace = AtmosTransport.Models._chemistry_workspace_for(
                operator, state, nothing)
            plan = chemistry_workspace_plan(workspace)

            @test state.tracer_names == (:inert, :B, :A, :C)
            @test Array(workspace.tracer_indices) == Int32[3, 2]
            @test size(workspace.concentrations) == (cell_count, 2)
            @test plan.species_count == 2
            @test workspace.host_forcing_stage === nothing
            @test propertynames(workspace.forcing.fixed_species) == (:C,)
            @test Array(workspace.forcing.fixed_species.values) == FT[1e12]

            apply!(state, nothing, nothing, operator, FT(100); workspace)
            final_a = vec(Array(state.tracers.A))
            final_b = vec(Array(state.tracers.B))
            @test final_a ≈ initial_a .* exp(FT(-0.1)) rtol = 2e-7
            @test final_a .+ final_b ≈ initial_a .+ initial_b rtol = 2e-7
            @test Array(state.tracers.inert) == state_cpu.tracers.inert
            @test Array(state.tracers.C) == state_cpu.tracers.C
            @test all(==(Int8(1)), workspace.diagnostics.status)
        end

        FT = Float64
        species = (
            Species(:A; molar_mass = 10, atoms = (:X => 1,)),
            Species(:B; molar_mass = 10, atoms = (:X => 1,)),
        )
        mechanism = Mechanism(species, (
            Reaction((:A => 1,), (:B => 1,), ConstantRate(1e-3)),))
        architecture = NativeGPU(
            CUDA.CUDABackend(), FixedWorkgroup(64),
            TabulatedReactions(), CellTeam(4))
        compiled = compile_mechanism(mechanism; architecture, precision = FT)
        policy = SolverPolicy(FT; abstol = 1e-10, reltol = 1e-9,
                              positivity = :allow)
        chemistry = ChemistryModel(compiled;
            solver = Rodas3(NativeSolver(SparseLinearSolver())),
            policy, architecture)
        panel_density = ntuple(6) do panel
            reshape(FT.(1:16) .* FT(1e18 + panel * 1e15), 4, 4, 1)
        end
        forcing = ntuple(6) do panel
            ChemistryForcing(FT;
                temperature = 298,
                air_number_density = panel_density[panel])
        end
        operator = AtmosChemistryOperator(chemistry, forcing)

        panel_air = ntuple(_ -> ones(FT, 4, 4, 1), 6)
        panel_a = ntuple(_ -> fill(FT(0.1), 4, 4, 1), 6)
        panel_b = ntuple(_ -> zeros(FT, 4, 4, 1), 6)
        cs_cpu = CubedSphereState(DryBasis, panel_air; halo_width = 1,
                                  A = panel_a, B = panel_b)
        cs_state = Adapt.adapt(CUDA.CuArray, cs_cpu)
        workspaces = AtmosTransport.Models._chemistry_workspace_for(
            operator, cs_state, nothing)
        @test workspaces.concentrations isa CUDA.CuArray{FT, 2}
        @test Array(workspaces.forcing[1].air_number_density) ==
              vec(@view panel_density[1][2:3, 2:3, :])

        apply!(cs_state, nothing, nothing, operator, FT(100);
               workspace = workspaces)
        for panel in cs_state.tracers.A
            host = Array(panel)
            @test all(@view(host[2:3, 2:3, :]) .≈ FT(0.1) * exp(FT(-0.1)))
            @test host[1, 1, 1] == FT(0.1)
            @test host[4, 4, 1] == FT(0.1)
        end

        cell_count = 4103
        linear_forcing = ChemistryForcing(FT;
            temperature = 298, air_number_density = 2.5e19)
        linear_operator = AtmosChemistryOperator(chemistry, linear_forcing)
        tiled_cpu = CellState(DryBasis, ones(FT, cell_count, 1);
            A = fill(FT(0.1), cell_count, 1),
            B = zeros(FT, cell_count, 1))
        tiled_state = Adapt.adapt(CUDA.CuArray, tiled_cpu)
        tiled_workspace = AtmosTransport.Models._chemistry_workspace_for(
            linear_operator, tiled_state, nothing)
        @test size(tiled_workspace.concentrations, 1) == 4096
        @test size(tiled_workspace.concentrations, 1) < cell_count
        @test chemistry_workspace_plan(tiled_workspace).tile_cells == 4096
        apply!(tiled_state, nothing, nothing, linear_operator, FT(100);
               workspace = tiled_workspace)
        tiled_a = Array(tiled_state.tracers.A)
        @test tiled_a[1] ≈ FT(0.1) * exp(FT(-0.1)) rtol = 2e-7
        @test tiled_a[end] ≈ FT(0.1) * exp(FT(-0.1)) rtol = 2e-7
        @test all(==(Int8(1)), tiled_workspace.diagnostics.status)

        failure_policy(throw_on_failure) = SolverPolicy(FT;
            abstol = 1e-10, reltol = 1e-9, initial_step = 1e-6,
            max_step = 1e-6, max_steps = 1, positivity = :allow,
            throw_on_failure)
        failure_model(throw_on_failure) = ChemistryModel(compiled;
            solver = Rodas3(NativeSolver(SparseLinearSolver())),
            policy = failure_policy(throw_on_failure), architecture)

        quiet_operator = AtmosChemistryOperator(
            failure_model(false), linear_forcing)
        quiet_cpu = CellState(DryBasis, ones(FT, 5, 1);
            A = fill(FT(0.1), 5, 1), B = zeros(FT, 5, 1))
        quiet_state = Adapt.adapt(CUDA.CuArray, quiet_cpu)
        quiet_workspace = AtmosTransport.Models._chemistry_workspace_for(
            quiet_operator, quiet_state, nothing)
        quiet_before = Array(quiet_state.tracers_raw)
        apply!(quiet_state, nothing, nothing, quiet_operator, FT(100);
               workspace = quiet_workspace)
        @test all(==(Int8(-1)), quiet_workspace.diagnostics.status)
        @test Array(quiet_state.tracers_raw) == quiet_before

        throwing_operator = AtmosChemistryOperator(
            failure_model(true), ntuple(_ -> linear_forcing, 6))
        failed_cs = Adapt.adapt(CUDA.CuArray, cs_cpu)
        failed_workspace = AtmosTransport.Models._chemistry_workspace_for(
            throwing_operator, failed_cs, nothing)
        failed_before = map(Array, failed_cs.tracers_raw)
        @test_throws ChemistrySolveError apply!(
            failed_cs, nothing, nothing, throwing_operator, FT(100);
            workspace = failed_workspace)
        @test all(==(Int8(-1)), failed_workspace.diagnostics.status)
        @test all(panel -> Array(failed_cs.tracers_raw[panel]) ==
                  failed_before[panel], 1:6)

        photolysis_mechanism = Mechanism(species, (
            Reaction((:A => 1,), (:B => 1,), PhotolysisRate(:J_A)),))
        photolysis_compiled = compile_mechanism(
            photolysis_mechanism; architecture, precision = FT)
        photolysis_model = ChemistryModel(photolysis_compiled;
            solver = Rodas3(NativeSolver(SparseLinearSolver())),
            policy, architecture)
        initial_photolysis_forcing = ChemistryForcing(FT;
            temperature = 298, air_number_density = 2.5e19,
            photolysis = (J_A = 0,))
        photolysis_provider = CallUpdatedChemistryForcing(
            initial_photolysis_forcing,
            (initial, meteo, grid) -> ChemistryForcing(FT;
                temperature = meteo.temperature,
                air_number_density = initial.air_number_density,
                photolysis = (J_A = meteo.J_A,)))
        photolysis_operator = AtmosChemistryOperator(
            photolysis_model, photolysis_provider)
        photolysis_cpu = CellState(DryBasis, ones(FT, 2, 1);
            A = fill(FT(0.1), 2, 1), B = zeros(FT, 2, 1))
        photolysis_state = Adapt.adapt(CUDA.CuArray, photolysis_cpu)
        photolysis_workspace = AtmosTransport.Models._chemistry_workspace_for(
            photolysis_operator, photolysis_state, nothing)
        @test photolysis_workspace.host_forcing_stage isa Vector{FT}
        photolysis_plan = chemistry_workspace_plan(photolysis_workspace)
        @test photolysis_plan.construction_transient_host_bytes == sizeof(FT)
        @test photolysis_plan.construction_peak_host_bytes ==
              photolysis_plan.host_persistent_bytes + sizeof(FT)
        @test chemistry_workspace_storage_bytes(photolysis_workspace).total ==
              photolysis_plan.required_backend_bytes +
              photolysis_plan.host_persistent_bytes
        photolysis_storage = photolysis_workspace.forcing.photolysis.values
        apply!(photolysis_state, (temperature = FT(299), J_A = FT(1e-3)),
               nothing, photolysis_operator, FT(100);
               workspace = photolysis_workspace)
        @test photolysis_workspace.forcing.photolysis.values ===
              photolysis_storage
        @test only(Array(photolysis_storage)) == FT(1e-3)
        @test vec(Array(photolysis_state.tracers.A)) ≈
              fill(FT(0.1) * exp(FT(-0.1)), 2) rtol = 2e-7

        resident_values = CuArray(FT[1e-3 9; 2e-3 8])
        resident_fields = PackedForcingFields(
            (:J_A, :EXTRA), resident_values, CellForcingFields())
        mixed_forcing = ChemistryForcing(FT;
            temperature = fill(FT(298), 2),
            air_number_density = CuArray(fill(FT(2.5e19), 2)),
            photolysis = resident_fields)
        mixed_operator = AtmosChemistryOperator(
            photolysis_model, mixed_forcing)
        mixed_state = Adapt.adapt(CUDA.CuArray,
            CellState(DryBasis, ones(FT, 2, 1);
                A = fill(FT(0.1), 2, 1), B = zeros(FT, 2, 1)))
        mixed_plan = chemistry_workspace_plan(
            mixed_operator, mixed_state, nothing)
        mixed_workspace = AtmosTransport.Models._chemistry_workspace_for(
            mixed_operator, mixed_state, nothing)
        mixed_measured = chemistry_workspace_storage_bytes(mixed_workspace)
        @test mixed_plan.peak_backend_bytes ==
              mixed_plan.required_backend_bytes
        @test mixed_plan.construction_transient_host_bytes == 0
        @test mixed_measured.total == mixed_plan.required_backend_bytes +
              mixed_plan.host_persistent_bytes
        @test propertynames(mixed_workspace.forcing.photolysis) == (:J_A,)
        @test Array(mixed_workspace.forcing.photolysis.values) ==
              reshape(FT[1e-3, 2e-3], 2, 1)
        @test mixed_workspace.forcing.photolysis.values !== resident_values

        cpu_photolysis_compiled = compile_mechanism(
            photolysis_mechanism; precision = FT)
        cpu_photolysis_model = ChemistryModel(cpu_photolysis_compiled;
            solver = Rodas3(NativeSolver(SparseLinearSolver())), policy)
        wrong_source_operator = AtmosChemistryOperator(
            cpu_photolysis_model, ChemistryForcing(FT;
                temperature = 298, air_number_density = 2.5e19,
                photolysis = resident_fields))
        wrong_source_state = CellState(DryBasis, ones(FT, 2, 1);
            A = fill(FT(0.1), 2, 1), B = zeros(FT, 2, 1))
        @test_throws ArgumentError chemistry_workspace_plan(
            wrong_source_operator, wrong_source_state, nothing)
        apply!(mixed_state, nothing, nothing, mixed_operator, FT(10);
               workspace = mixed_workspace)
        @test vec(Array(mixed_state.tracers.A)) ≈
              FT[0.1exp(-0.01), 0.1exp(-0.02)] rtol = 2e-7

        halo_j_host = FT.(1:16) .* FT(1e-4)
        resident_halo_values = CuArray(hcat(halo_j_host, fill(FT(7), 16)))
        resident_halo_density = CuArray(fill(FT(2.5e19), 16))
        resident_halo_forcing = ntuple(6) do _
            ChemistryForcing(FT;
                temperature = 298,
                air_number_density = resident_halo_density,
                photolysis = PackedForcingFields(
                    (:J_A, :EXTRA), resident_halo_values,
                    CellForcingFields()))
        end
        resident_halo_operator = AtmosChemistryOperator(
            photolysis_model, resident_halo_forcing)
        resident_halo_plan = chemistry_workspace_plan(
            resident_halo_operator, cs_state, nothing)
        resident_halo_workspace =
            AtmosTransport.Models._chemistry_workspace_for(
                resident_halo_operator, cs_state, nothing)
        @test resident_halo_plan.required_backend_bytes ==
              chemistry_workspace_plan(
                  resident_halo_workspace).required_backend_bytes
        @test chemistry_workspace_storage_bytes(
                  resident_halo_workspace).total ==
              resident_halo_plan.required_backend_bytes +
              resident_halo_plan.host_persistent_bytes
        @test size(resident_halo_workspace.forcing[1].photolysis.values) ==
              (4, 1)
        @test vec(Array(
            resident_halo_workspace.forcing[1].photolysis.values)) ==
              halo_j_host[[6, 7, 10, 11]]

        host_cell_fields = PackedForcingFields(
            (:J_A,), zeros(FT, 2, 1), CellForcingFields())
        host_cell_initial = ChemistryForcing(FT;
            temperature = 298, air_number_density = 2.5e19,
            photolysis = host_cell_fields)
        host_cell_provider = CallUpdatedChemistryForcing(
            host_cell_initial,
            (initial, meteo, grid) -> ChemistryForcing(FT;
                temperature = initial.temperature,
                air_number_density = initial.air_number_density,
                photolysis = PackedForcingFields(
                    (:J_A,), meteo, CellForcingFields())))
        host_cell_operator = AtmosChemistryOperator(
            photolysis_model, host_cell_provider)
        host_cell_state = Adapt.adapt(CUDA.CuArray,
            CellState(DryBasis, ones(FT, 2, 1);
                A = fill(FT(0.1), 2, 1), B = zeros(FT, 2, 1)))
        host_cell_plan = chemistry_workspace_plan(
            host_cell_operator, host_cell_state, nothing)
        @test host_cell_plan.construction_transient_host_bytes ==
              2 * sizeof(FT) + sizeof(Int32)
        host_cell_workspace = AtmosTransport.Models._chemistry_workspace_for(
            host_cell_operator, host_cell_state, nothing)
        host_cell_storage =
            host_cell_workspace.forcing.photolysis.values
        updated_j = reshape(FT[1e-3, 2e-3], 2, 1)
        apply!(host_cell_state, updated_j, nothing, host_cell_operator,
               FT(10); workspace = host_cell_workspace)
        @test host_cell_workspace.forcing.photolysis.values ===
              host_cell_storage
        @test Array(host_cell_storage) == updated_j
        @test vec(Array(host_cell_state.tracers.A)) ≈
              FT[0.1exp(-0.01), 0.1exp(-0.02)] rtol = 2e-7

        initial_panel_forcing = ntuple(6) do panel
            ChemistryForcing(FT;
                temperature = 298,
                air_number_density = CuArray(panel_density[panel]))
        end
        updated_panel_density = ntuple(6) do panel
            CuArray(panel_density[panel] .* FT(2))
        end
        updated_panel_forcing = ntuple(6) do panel
            ChemistryForcing(FT;
                temperature = 299,
                air_number_density = updated_panel_density[panel])
        end
        panel_provider = CallUpdatedChemistryForcing(
            initial_panel_forcing, (initial, meteo, grid) -> meteo)
        panel_operator = AtmosChemistryOperator(chemistry, panel_provider)
        panel_state = Adapt.adapt(CUDA.CuArray, cs_cpu)
        panel_workspace = AtmosTransport.Models._chemistry_workspace_for(
            panel_operator, panel_state, nothing)
        @test panel_workspace.forcing[1].air_number_density !==
              initial_panel_forcing[1].air_number_density
        panel_density_storage =
            panel_workspace.forcing[1].air_number_density
        apply!(panel_state, updated_panel_forcing, nothing, panel_operator,
               FT(100); workspace = panel_workspace)
        @test panel_workspace.forcing[1].air_number_density ===
              panel_density_storage
        @test Array(panel_density_storage) ==
              vec(@view(Array(updated_panel_density[1])[2:3, 2:3, :]))

        host_panel_initial = ntuple(6) do panel
            ChemistryForcing(FT;
                temperature = 298,
                air_number_density = panel_density[panel])
        end
        host_panel_density = ntuple(6) do panel
            panel_density[panel] .* FT(3)
        end
        host_panel_updated = ntuple(6) do panel
            ChemistryForcing(FT;
                temperature = 299,
                air_number_density = host_panel_density[panel])
        end
        host_panel_provider = CallUpdatedChemistryForcing(
            host_panel_initial, (initial, meteo, grid) -> meteo)
        host_panel_operator = AtmosChemistryOperator(
            chemistry, host_panel_provider)
        host_panel_state = Adapt.adapt(CUDA.CuArray, cs_cpu)
        host_panel_plan = chemistry_workspace_plan(
            host_panel_operator, host_panel_state, nothing)
        @test host_panel_plan.construction_transient_host_bytes ==
              6 * 4 * sizeof(FT)
        host_panel_workspace = AtmosTransport.Models._chemistry_workspace_for(
            host_panel_operator, host_panel_state, nothing)
        @test host_panel_workspace.host_forcing_stage isa Vector{FT}
        host_panel_storage =
            host_panel_workspace.forcing[1].air_number_density
        apply!(host_panel_state, host_panel_updated, nothing,
               host_panel_operator, FT(100);
               workspace = host_panel_workspace)
        @test host_panel_workspace.forcing[1].air_number_density ===
              host_panel_storage
        @test Array(host_panel_storage) ==
              vec(@view(host_panel_density[1][2:3, 2:3, :]))

        pressure = FT(2.5e19) * FT(1e6) * FT(1.380649e-23) * FT(298)
        pressure_forcing = ChemistryForcing(FT;
            temperature = 298, pressure)
        pressure_operator = AtmosChemistryOperator(
            chemistry, pressure_forcing)
        pressure_state = Adapt.adapt(CUDA.CuArray,
            CellState(DryBasis, ones(FT, 2, 1);
                A = fill(FT(0.1), 2, 1), B = zeros(FT, 2, 1)))
        apply!(pressure_state, nothing, nothing, pressure_operator, FT(100))
        @test vec(Array(pressure_state.tracers.A)) ≈
              fill(FT(0.1) * exp(FT(-0.1)), 2) rtol = 2e-7

        short_forcing = ChemistryForcing(FT;
            temperature = 298, air_number_density = FT[2.5e19])
        short_operator = AtmosChemistryOperator(chemistry, short_forcing)
        short_state = Adapt.adapt(CUDA.CuArray,
            CellState(DryBasis, ones(FT, 2, 1);
                A = fill(FT(0.1), 2, 1), B = zeros(FT, 2, 1)))
        short_before = Array(short_state.tracers_raw)
        @test_throws DimensionMismatch AtmosTransport.Models._chemistry_workspace_for(
            short_operator, short_state, nothing)
        @test Array(short_state.tracers_raw) == short_before
    end
end
