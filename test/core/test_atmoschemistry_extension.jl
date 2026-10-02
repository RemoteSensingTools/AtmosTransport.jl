using Test
using AtmosTransport

if Base.find_package("AtmosChemistry") === nothing
    @test_skip "AtmosChemistry weak dependency is unavailable"
else
    using AtmosChemistry

    @testset "AtmosChemistry dry-mass adapter" begin
        species = (
            Species(:A; molar_mass = 10.0, atoms = (:X => 1,)),
            Species(:B; molar_mass = 10.0, atoms = (:X => 1,)),
        )
        reactions = (Reaction((:A => 1,), (:B => 1,), ConstantRate(1e-3)),)
        mechanism = compile_mechanism(Mechanism(species, reactions))
        policy = SolverPolicy(Float64; abstol = 1e-10, reltol = 1e-9,
                              positivity = :allow)
        model = ChemistryModel(mechanism; policy)
        forcing = ChemistryForcing(temperature = 298.0,
                                   air_number_density = 2.5e19)
        operator = AtmosChemistryOperator(model, forcing)

        air_mass = reshape([1.0, 2.0], 2, 1)
        state = CellState(DryBasis, air_mass;
            A = reshape([0.1, 0.2], 2, 1),
            B = zeros(2, 1),
            inert = fill(3.0, 2, 1))
        initial_reactive_mass = state.tracers.A .+ state.tracers.B
        initial_inert = copy(state.tracers.inert)

        plan = chemistry_workspace_plan(operator, state, nothing)
        workspace = AtmosTransport.Models._chemistry_workspace_for(
            operator, state, nothing)
        measured = chemistry_workspace_storage_bytes(workspace)
        @test workspace.host_forcing_stage === nothing
        @test chemistry_workspace_plan(workspace).tile_cells == plan.tile_cells
        @test chemistry_workspace_plan(workspace).required_backend_bytes ==
              plan.required_backend_bytes
        @test plan.tile_cells == 2
        @test measured.total == plan.required_backend_bytes
        @test plan.construction_transient_host_bytes == 0
        @test plan.construction_peak_host_bytes == plan.host_persistent_bytes
        plan_summary = sprint(show, plan)
        @test occursin("cells=2", plan_summary)
        @test occursin("tile_cells=2", plan_summary)
        @test occursin("budget=", plan_summary)
        @test_throws ArgumentError AtmosChemistryOperator(
            model, forcing; dry_air_molar_mass = Inf)

        shared_forcing_field = fill(298.0, 2)
        aliased_forcing = ChemistryForcing(
            temperature = shared_forcing_field,
            pressure = shared_forcing_field)
        aliased_operator = AtmosChemistryOperator(model, aliased_forcing)
        aliased_plan = chemistry_workspace_plan(
            aliased_operator, state, nothing)
        aliased_workspace = AtmosTransport.Models._chemistry_workspace_for(
            aliased_operator, state, nothing)
        @test chemistry_workspace_storage_bytes(aliased_workspace).total ==
              aliased_plan.required_backend_bytes
        aliased_limit = aliased_plan.required_backend_bytes - 1
        aliased_tight_operator = AtmosChemistryOperator(
            model, aliased_forcing;
            workspace_policy = WorkspacePolicy(
                max_bytes = aliased_limit, fixed_tile_cells = 2))
        @test_throws ArgumentError chemistry_workspace_plan(
            aliased_tight_operator, state, nothing)

        apply!(state, nothing, nothing, operator, 100.0; workspace)

        @test vec(state.tracers.A) ≈ [0.1, 0.2] .* exp(-0.1) rtol = 2e-7
        @test state.tracers.A .+ state.tracers.B ≈ initial_reactive_mass
        @test state.tracers.inert == initial_inert

        moist_state = CellState(MoistBasis, copy(air_mass);
            A = reshape([0.1, 0.2], 2, 1), B = zeros(2, 1))
        moist_before = copy(moist_state.tracers_raw)
        @test_throws ArgumentError apply!(
            moist_state, nothing, nothing, operator, 100.0)
        @test moist_state.tracers_raw == moist_before

        panel_air = ntuple(_ -> ones(4, 4, 1), 6)
        panel_a = ntuple(_ -> fill(0.1, 4, 4, 1), 6)
        panel_b = ntuple(_ -> zeros(4, 4, 1), 6)
        panel_inert = ntuple(_ -> fill(3.0, 4, 4, 1), 6)
        cs_state = CubedSphereState(DryBasis, panel_air; halo_width = 1,
                                    A = panel_a, B = panel_b, inert = panel_inert)

        apply!(cs_state, nothing, nothing, operator, 100.0)

        @test all(panel -> all(@view(panel[2:3, 2:3, :]) .≈ 0.1exp(-0.1)),
                  cs_state.tracers.A)
        @test all(panel -> panel[1, 1, 1] == 0.1, cs_state.tracers.A)
        @test all(panel -> panel[1, 1, 1] == 0.0, cs_state.tracers.B)
        @test all(panel -> all(panel .== 3.0), cs_state.tracers.inert)

        moist_cs = CubedSphereState(MoistBasis, panel_air; halo_width = 1,
                                    A = panel_a, B = panel_b)
        @test_throws ArgumentError apply!(
            moist_cs, nothing, nothing, operator, 100.0)

        photolysis = Reaction((:A => 1,), (:B => 1,),
                              PhotolysisRate(:J_A))
        photolysis_mechanism = compile_mechanism(
            Mechanism(species, (photolysis,)))
        photolysis_model = ChemistryModel(photolysis_mechanism; policy)
        extra_constant_forcing = ChemistryForcing(
            temperature = 298.0, air_number_density = 2.5e19,
            photolysis = (J_A = 0.0, EXTRA = 1.0))
        extra_constant_operator = AtmosChemistryOperator(
            photolysis_model, extra_constant_forcing)
        extra_constant_plan = chemistry_workspace_plan(
            extra_constant_operator, state, nothing)
        extra_constant_workspace =
            AtmosTransport.Models._chemistry_workspace_for(
                extra_constant_operator, state, nothing)
        @test propertynames(
            extra_constant_workspace.forcing.photolysis) == (:J_A,)
        @test chemistry_workspace_storage_bytes(
                  extra_constant_workspace).total ==
              extra_constant_plan.required_backend_bytes
        cell_fields = PackedForcingFields(
            (:J_A, :EXTRA), [0.0 4.0; 0.0 5.0], CellForcingFields())
        cell_operator = AtmosChemistryOperator(
            photolysis_model, ChemistryForcing(
                temperature = 298.0, air_number_density = 2.5e19,
                photolysis = cell_fields))
        cell_plan = chemistry_workspace_plan(cell_operator, state, nothing)
        cell_workspace = AtmosTransport.Models._chemistry_workspace_for(
            cell_operator, state, nothing)
        @test cell_workspace.forcing.photolysis.values == zeros(2, 1)
        @test chemistry_workspace_storage_bytes(cell_workspace).total ==
              cell_plan.required_backend_bytes
        @test cell_plan.construction_transient_host_bytes == 0
        tight_cell_operator = AtmosChemistryOperator(
            photolysis_model, cell_operator.forcing_provider;
            workspace_policy = WorkspacePolicy(
                max_bytes = cell_plan.required_backend_bytes - 1,
                fixed_tile_cells = 2))
        @test_throws ArgumentError chemistry_workspace_plan(
            tight_cell_operator, state, nothing)
        initial_forcing = ChemistryForcing(
            temperature = 298.0, air_number_density = 2.5e19,
            photolysis = (J_A = 0.0,))
        provider = CallUpdatedChemistryForcing(
            initial_forcing,
            (initial, meteo, grid) -> ChemistryForcing(
                temperature = meteo.temperature,
                air_number_density = initial.air_number_density,
                photolysis = (J_A = meteo.J_A,)))
        updated_operator = AtmosChemistryOperator(photolysis_model, provider)
        @test updated_operator.forcing_provider === provider
        updated_state = CellState(DryBasis, ones(2, 1);
            A = fill(0.1, 2, 1), B = zeros(2, 1))
        updated_workspace = AtmosTransport.Models._chemistry_workspace_for(
            updated_operator, updated_state, nothing)
        photolysis_storage = updated_workspace.forcing.photolysis.values
        apply!(updated_state, (temperature = 299.0, J_A = 1e-3), nothing,
               updated_operator, 100.0; workspace = updated_workspace)
        @test vec(updated_state.tracers.A) ≈
              fill(0.1exp(-0.1), 2) rtol = 2e-7
        @test updated_workspace.forcing.photolysis.values === photolysis_storage
        @test only(photolysis_storage) == 1e-3

        invalid_provider = CallUpdatedChemistryForcing(
            initial_forcing,
            (initial, meteo, grid) -> ChemistryForcing(
                temperature = initial.temperature,
                air_number_density = initial.air_number_density,
                photolysis = (J_A = fill(1e-3, 2),)))
        invalid_operator = AtmosChemistryOperator(
            photolysis_model, invalid_provider)
        invalid_state = CellState(DryBasis, ones(2, 1);
            A = fill(0.1, 2, 1), B = zeros(2, 1))
        invalid_workspace = AtmosTransport.Models._chemistry_workspace_for(
            invalid_operator, invalid_state, nothing)
        invalid_before = copy(invalid_state.tracers_raw)
        @test_throws ArgumentError apply!(
            invalid_state, nothing, nothing, invalid_operator, 100.0;
            workspace = invalid_workspace)
        @test invalid_state.tracers_raw == invalid_before

        extra_provider = CallUpdatedChemistryForcing(
            initial_forcing,
            (initial, meteo, grid) -> ChemistryForcing(
                temperature = initial.temperature,
                air_number_density = initial.air_number_density,
                photolysis = (J_A = 1e-3, EXTRA = 2e-3)))
        extra_operator = AtmosChemistryOperator(
            photolysis_model, extra_provider)
        extra_state = CellState(DryBasis, ones(2, 1);
            A = fill(0.1, 2, 1), B = zeros(2, 1))
        extra_workspace = AtmosTransport.Models._chemistry_workspace_for(
            extra_operator, extra_state, nothing)
        extra_before = copy(extra_state.tracers_raw)
        @test_throws ArgumentError apply!(
            extra_state, nothing, nothing, extra_operator, 100.0;
            workspace = extra_workspace)
        @test extra_state.tracers_raw == extra_before

        composite_state = CellState(DryBasis, ones(2, 1);
            A = fill(0.1, 2, 1), B = zeros(2, 1))
        composite = CompositeChemistry(operator, NoChemistry())
        composite_workspace = AtmosTransport.Models._chemistry_workspace_for(
            composite, composite_state, nothing)
        @test composite_workspace isa CompositeChemistryWorkspace
        @test length(composite_workspace.workspaces) == 2
        composite_plan = chemistry_workspace_plan(
            composite, composite_state, nothing)
        @test composite_plan isa CompositeChemistryPlan
        @test composite_plan.plans[1].tile_cells == 2
        @test composite_plan.plans[2] === nothing
        composite_storage = chemistry_workspace_storage_bytes(
            composite_workspace)
        @test composite_storage.total ==
              chemistry_workspace_storage_bytes(
                  composite_workspace.workspaces[1]).total
        @test composite_plan.backend_bytes == composite_storage.total
        @test composite_plan.host_bytes == 0
        @test composite_plan.total_bytes == composite_storage.total
        apply!(composite_state, nothing, nothing, composite, 100.0;
               workspace = composite_workspace)
        @test vec(composite_state.tracers.A) ≈
              fill(0.1exp(-0.1), 2) rtol = 2e-7

        child_bytes = chemistry_workspace_plan(
            operator, composite_state, nothing).required_backend_bytes
        capped_operator = AtmosChemistryOperator(
            model, forcing;
            workspace_policy = WorkspacePolicy(
                max_bytes = child_bytes + 100, fixed_tile_cells = 2))
        over_budget_composite = CompositeChemistry(
            capped_operator, capped_operator)
        @test_throws ArgumentError chemistry_workspace_plan(
            over_budget_composite, composite_state, nothing)
        @test_throws ArgumentError AtmosTransport.Models._chemistry_workspace_for(
            over_budget_composite, composite_state, nothing)

        tiny_policy = WorkspacePolicy(max_bytes = 1)
        tiny_operator = AtmosChemistryOperator(
            model, forcing; workspace_policy = tiny_policy)
        @test_throws ArgumentError chemistry_workspace_plan(
            tiny_operator, state, nothing)

        pressure = 2.5e19 * 1e6 * 1.380649e-23 * 298.0
        pressure_forcing = ChemistryForcing(
            temperature = 298.0, pressure = pressure)
        pressure_operator = AtmosChemistryOperator(model, pressure_forcing)
        pressure_state = CellState(DryBasis, ones(2, 1);
            A = fill(0.1, 2, 1), B = zeros(2, 1))
        apply!(pressure_state, nothing, nothing, pressure_operator, 100.0)
        @test vec(pressure_state.tracers.A) ≈
              fill(0.1exp(-0.1), 2) rtol = 2e-7

        halo_density = ntuple(6) do panel
            fill(2.5e19 + panel, 4, 4, 1)
        end
        halo_forcing = ntuple(6) do panel
            ChemistryForcing(
                temperature = 298.0,
                air_number_density = halo_density[panel])
        end
        halo_operator = AtmosChemistryOperator(model, halo_forcing)
        halo_plan = chemistry_workspace_plan(
            halo_operator, cs_state, nothing)
        halo_workspace = AtmosTransport.Models._chemistry_workspace_for(
            halo_operator, cs_state, nothing)
        @test chemistry_workspace_plan(halo_workspace).required_backend_bytes ==
              halo_plan.required_backend_bytes
        @test chemistry_workspace_storage_bytes(halo_workspace).total ==
              halo_plan.required_backend_bytes

        short_forcing = ChemistryForcing(
            temperature = 298.0, air_number_density = [2.5e19])
        short_operator = AtmosChemistryOperator(model, short_forcing)
        short_state = CellState(DryBasis, ones(2, 1);
            A = fill(0.1, 2, 1), B = zeros(2, 1))
        short_before = copy(short_state.tracers_raw)
        @test_throws DimensionMismatch AtmosTransport.Models._chemistry_workspace_for(
            short_operator, short_state, nothing)
        @test short_state.tracers_raw == short_before

        planning_cells = 64
        planning_state = CellState(DryBasis, ones(planning_cells, 1);
            A = fill(0.1, planning_cells, 1),
            B = zeros(planning_cells, 1))
        core_plan = workspace_plan(model, planning_cells;
            policy = WorkspacePolicy(fixed_tile_cells = planning_cells))
        core_limit = core_plan.persistent_bytes + core_plan.tile_bytes
        fixed_policy = WorkspacePolicy(
            max_bytes = core_limit, fixed_tile_cells = planning_cells)
        fixed_operator = AtmosChemistryOperator(
            model, forcing; workspace_policy = fixed_policy)
        @test_throws ArgumentError chemistry_workspace_plan(
            fixed_operator, planning_state, nothing)
        automatic_operator = AtmosChemistryOperator(
            model, forcing;
            workspace_policy = WorkspacePolicy(max_bytes = core_limit))
        automatic_plan = chemistry_workspace_plan(
            automatic_operator, planning_state, nothing)
        @test automatic_plan.tile_cells < planning_cells
        next_tile = automatic_plan.tile_cells + 1
        next_operator = AtmosChemistryOperator(
            model, forcing;
            workspace_policy = WorkspacePolicy(
                max_bytes = core_limit, fixed_tile_cells = next_tile))
        @test_throws ArgumentError chemistry_workspace_plan(
            next_operator, planning_state, nothing)
    end
end
