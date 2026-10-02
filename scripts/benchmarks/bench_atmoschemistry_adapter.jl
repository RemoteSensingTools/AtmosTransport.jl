#!/usr/bin/env julia

using Adapt
using AtmosChemistry
using AtmosTransport
using CUDA
using Printf
using SHA
using Statistics
using TOML

function percentile(values, fraction)
    ordered = sort(values)
    index = clamp(ceil(Int, fraction * length(ordered)), 1, length(ordered))
    return ordered[index]
end

function elapsed_cuda(f)
    start = time_ns()
    f()
    CUDA.synchronize()
    return (time_ns() - start) / 1e9
end

function reset_transport_state!(state, initial_storage, workspace)
    copyto!(state.tracers_raw, initial_storage)
    fill!(workspace.warm_steps, zero(eltype(workspace.warm_steps)))
    CUDA.synchronize()
    return nothing
end

function reset_core_state!(concentrations, initial, workspace)
    copyto!(concentrations, initial)
    fill!(workspace.last_step, zero(eltype(workspace.last_step)))
    CUDA.synchronize()
    return nothing
end

measure_adapter!(state, operator, duration, workspace) = elapsed_cuda() do
    apply!(state, nothing, nothing, operator, duration; workspace)
end

measure_core!(batch, model, forcing, duration, workspace, diagnostics) =
    elapsed_cuda() do
        advance!(batch, model, forcing, duration; workspace, diagnostics)
    end

file_digest(path) = open(path) do io
    bytes2hex(sha256(io))
end

function source_snapshot_digest(root)
    files = String[]
    for entry in ("Project.toml", "src", "ext")
        path = joinpath(root, entry)
        if isfile(path)
            push!(files, path)
        elseif isdir(path)
            for (directory, _, names) in walkdir(path), name in names
                push!(files, joinpath(directory, name))
            end
        end
    end
    sort!(files; by = path -> relpath(path, root))
    io = IOBuffer()
    for path in files
        relative = relpath(path, root)
        write(io, string(ncodeunits(relative)), ':', relative, ';')
        write(io, file_digest(path), ';')
    end
    return bytes2hex(sha256(take!(io)))
end

function benchmark_adapter(::Type{FT}, cell_count; samples = 10) where FT
    CUDA.functional() || error("CUDA is not functional")
    transport_root = normpath(joinpath(@__DIR__, "..", ".."))
    chemistry_root = normpath(joinpath(transport_root, "..", "AtmosChemistry.jl"))
    provenance = Dict(
        "benchmark_script_sha256" => file_digest(@__FILE__),
        "test_manifest_sha256" => file_digest(joinpath(
            transport_root, "test", "atmoschemistry", "Manifest.toml")),
        "AtmosChemistry_source_snapshot_sha256" =>
            source_snapshot_digest(chemistry_root),
        "AtmosTransport_source_snapshot_sha256" =>
            source_snapshot_digest(transport_root),
    )

    species = (
        Species(:A; molar_mass = 10, atoms = (:X => 1,)),
        Species(:B; molar_mass = 10, atoms = (:X => 1,)),
    )
    mechanism = Mechanism(species, (
        Reaction((:A => 1,), (:B => 1,), ConstantRate(FT(1e-3))),))
    architecture = NativeGPU(
        CUDA.CUDABackend(), FixedWorkgroup(64),
        TabulatedReactions(), CellTeam(4))
    compiled = compile_mechanism(
        mechanism; architecture, precision = FT)
    policy = SolverPolicy(FT;
        abstol = FT(1e-8), reltol = FT(1e-7), positivity = :allow,
        allow_uncertified_float32 = FT === Float32)
    model = ChemistryModel(compiled;
        solver = Rodas3(NativeSolver(SparseLinearSolver())),
        policy, architecture)
    forcing = ChemistryForcing(FT;
        temperature = FT(298), air_number_density = FT(2.5e19))
    operator = AtmosChemistryOperator(model, forcing)

    host_state = CellState(DryBasis, ones(FT, cell_count, 1);
        A = fill(FT(0.1), cell_count, 1),
        B = zeros(FT, cell_count, 1))
    state_staging = @timed begin
        staged = Adapt.adapt(CUDA.CuArray, host_state)
        CUDA.synchronize()
        staged
    end
    state = state_staging.value
    initial_transport_storage = copy(state.tracers_raw)
    workspace_allocation = @timed begin
        allocated = AtmosTransport.Models._chemistry_workspace_for(
            operator, state, nothing)
        CUDA.synchronize()
        allocated
    end
    workspace = workspace_allocation.value
    extension = Base.get_extension(
        AtmosTransport, :AtmosTransportAtmosChemistryExt)
    extension === nothing && error("AtmosChemistry extension did not load")
    air_mass = reshape(state.air_mass, :)

    extension._validate_storage!(
        workspace, air_mass, workspace.forcing, workspace.layout,
        "benchmark")
    first_coupling = @timed begin
        apply!(state, nothing, nothing, operator, FT(10); workspace)
        CUDA.synchronize()
    end

    concentration_a = FT(0.1) * operator.dry_air_molar_mass / FT(10) *
                      FT(2.5e19)
    raw_initial = CuArray(hcat(
        fill(concentration_a, cell_count), zeros(FT, cell_count)))
    raw_cells = CuArray(Int32.(1:cell_count))
    checked_concentrations = copy(raw_initial)
    prevalidated_concentrations = copy(raw_initial)
    checked_batch = ChemistryBatch(checked_concentrations, raw_cells)
    prevalidated_batch = ChemistryBatch(
        prevalidated_concentrations, raw_cells, PrevalidatedForcingCells())
    raw_forcing = pack_forcing(forcing, compiled, architecture)
    checked_workspace = allocate_workspace(model, cell_count;
        policy = WorkspacePolicy(fixed_tile_cells = 4096))
    prevalidated_workspace = allocate_workspace(model, cell_count;
        policy = WorkspacePolicy(fixed_tile_cells = 4096))
    checked_diagnostics = allocate_diagnostics(FT, cell_count)
    prevalidated_diagnostics = allocate_diagnostics(FT, cell_count)

    terminal_capacity = workspace.plan.tile_cells
    terminal_active_cells = min(7, terminal_capacity)
    terminal_initial = CuArray(hcat(
        fill(concentration_a, terminal_capacity),
        zeros(FT, terminal_capacity)))
    terminal_active_concentrations = copy(terminal_initial)
    terminal_padded_concentrations = copy(terminal_initial)
    terminal_cells = CuArray(Int32.(1:terminal_capacity))
    terminal_active_batch = ChemistryBatch(
        terminal_active_concentrations, terminal_cells,
        PrevalidatedForcingCells(); active_cells = terminal_active_cells)
    terminal_padded_batch = ChemistryBatch(
        terminal_padded_concentrations, terminal_cells,
        PrevalidatedForcingCells())
    terminal_active_workspace = allocate_workspace(model, terminal_capacity;
        policy = WorkspacePolicy(fixed_tile_cells = terminal_capacity))
    terminal_padded_workspace = allocate_workspace(model, terminal_capacity;
        policy = WorkspacePolicy(fixed_tile_cells = terminal_capacity))
    terminal_active_diagnostics = allocate_diagnostics(FT, terminal_capacity)
    terminal_padded_diagnostics = allocate_diagnostics(FT, terminal_capacity)

    reset_transport_state!(state, initial_transport_storage, workspace)
    apply!(state, nothing, nothing, operator, FT(10); workspace)
    reset_core_state!(checked_concentrations, raw_initial, checked_workspace)
    advance!(checked_batch, model, raw_forcing, FT(10);
             workspace = checked_workspace, diagnostics = checked_diagnostics)
    reset_core_state!(prevalidated_concentrations, raw_initial,
                      prevalidated_workspace)
    advance!(prevalidated_batch, model, raw_forcing, FT(10);
             workspace = prevalidated_workspace,
             diagnostics = prevalidated_diagnostics)
    reset_core_state!(terminal_active_concentrations, terminal_initial,
                      terminal_active_workspace)
    advance!(terminal_active_batch, model, raw_forcing, FT(10);
             workspace = terminal_active_workspace,
             diagnostics = terminal_active_diagnostics)
    reset_core_state!(terminal_padded_concentrations, terminal_initial,
                      terminal_padded_workspace)
    advance!(terminal_padded_batch, model, raw_forcing, FT(10);
             workspace = terminal_padded_workspace,
             diagnostics = terminal_padded_diagnostics)
    CUDA.synchronize()

    validation_times = Vector{Float64}(undef, samples)
    coupling_times = Vector{Float64}(undef, samples)
    raw_checked_times = Vector{Float64}(undef, samples)
    raw_prevalidated_times = Vector{Float64}(undef, samples)
    terminal_active_times = Vector{Float64}(undef, samples)
    terminal_padded_times = Vector{Float64}(undef, samples)
    for sample in 1:samples
        validation_times[sample] = @elapsed extension._validate_storage!(
            workspace, air_mass, workspace.forcing, workspace.layout,
            "benchmark")
        reset_transport_state!(state, initial_transport_storage, workspace)
        reset_core_state!(checked_concentrations, raw_initial,
                          checked_workspace)
        reset_core_state!(prevalidated_concentrations, raw_initial,
                          prevalidated_workspace)
        if isodd(sample)
            coupling_times[sample] = measure_adapter!(
                state, operator, FT(10), workspace)
            raw_prevalidated_times[sample] = measure_core!(
                prevalidated_batch, model, raw_forcing, FT(10),
                prevalidated_workspace, prevalidated_diagnostics)
            raw_checked_times[sample] = measure_core!(
                checked_batch, model, raw_forcing, FT(10),
                checked_workspace, checked_diagnostics)
        else
            raw_checked_times[sample] = measure_core!(
                checked_batch, model, raw_forcing, FT(10),
                checked_workspace, checked_diagnostics)
            raw_prevalidated_times[sample] = measure_core!(
                prevalidated_batch, model, raw_forcing, FT(10),
                prevalidated_workspace, prevalidated_diagnostics)
            coupling_times[sample] = measure_adapter!(
                state, operator, FT(10), workspace)
        end
        reset_core_state!(terminal_active_concentrations, terminal_initial,
                          terminal_active_workspace)
        reset_core_state!(terminal_padded_concentrations, terminal_initial,
                          terminal_padded_workspace)
        if isodd(sample)
            terminal_active_times[sample] = elapsed_cuda() do
                advance!(terminal_active_batch, model, raw_forcing, FT(10);
                    workspace = terminal_active_workspace,
                    diagnostics = terminal_active_diagnostics)
            end
            terminal_padded_times[sample] = elapsed_cuda() do
                advance!(terminal_padded_batch, model, raw_forcing, FT(10);
                    workspace = terminal_padded_workspace,
                    diagnostics = terminal_padded_diagnostics)
            end
        else
            terminal_padded_times[sample] = elapsed_cuda() do
                advance!(terminal_padded_batch, model, raw_forcing, FT(10);
                    workspace = terminal_padded_workspace,
                    diagnostics = terminal_padded_diagnostics)
            end
            terminal_active_times[sample] = elapsed_cuda() do
                advance!(terminal_active_batch, model, raw_forcing, FT(10);
                    workspace = terminal_active_workspace,
                    diagnostics = terminal_active_diagnostics)
            end
        end
    end

    Array(checked_concentrations) ≈ Array(prevalidated_concentrations) ||
        error("checked and prevalidated core paths disagree")
    checked_diagnostics.counter_storage ==
        prevalidated_diagnostics.counter_storage ||
        error("checked and prevalidated counters disagree")
    workspace.diagnostics.counter_storage ==
        prevalidated_diagnostics.counter_storage ||
        error("transport adapter and raw core counters disagree")
    Array(terminal_active_concentrations)[1:terminal_active_cells, :] ≈
        Array(terminal_padded_concentrations)[1:terminal_active_cells, :] ||
        error("active and padded terminal paths disagree")
    terminal_active_diagnostics.counter_storage[1:terminal_active_cells, :] ==
        terminal_padded_diagnostics.counter_storage[1:terminal_active_cells, :] ||
        error("active and padded terminal counters disagree")
    adapter_concentrations = reshape(
        Array(state.tracers_raw), cell_count, length(species)) .* (
            operator.dry_air_molar_mass / FT(10) * FT(2.5e19))
    adapter_concentrations ≈ Array(prevalidated_concentrations) ||
        error("transport adapter and raw core paths disagree")

    reset_transport_state!(state, initial_transport_storage, workspace)
    apply!(state, nothing, nothing, operator, FT(10); workspace)
    reset_core_state!(checked_concentrations, raw_initial, checked_workspace)
    advance!(checked_batch, model, raw_forcing, FT(10);
             workspace = checked_workspace, diagnostics = checked_diagnostics)
    reset_core_state!(prevalidated_concentrations, raw_initial,
                      prevalidated_workspace)
    advance!(prevalidated_batch, model, raw_forcing, FT(10);
             workspace = prevalidated_workspace,
             diagnostics = prevalidated_diagnostics)
    reset_core_state!(terminal_active_concentrations, terminal_initial,
                      terminal_active_workspace)
    advance!(terminal_active_batch, model, raw_forcing, FT(10);
             workspace = terminal_active_workspace,
             diagnostics = terminal_active_diagnostics)
    reset_core_state!(terminal_padded_concentrations, terminal_initial,
                      terminal_padded_workspace)
    advance!(terminal_padded_batch, model, raw_forcing, FT(10);
             workspace = terminal_padded_workspace,
             diagnostics = terminal_padded_diagnostics)

    warm_coupling_times = Vector{Float64}(undef, samples)
    warm_raw_checked_times = Vector{Float64}(undef, samples)
    warm_raw_prevalidated_times = Vector{Float64}(undef, samples)
    warm_terminal_active_times = Vector{Float64}(undef, samples)
    warm_terminal_padded_times = Vector{Float64}(undef, samples)
    for sample in 1:samples
        if isodd(sample)
            warm_coupling_times[sample] = measure_adapter!(
                state, operator, FT(10), workspace)
            warm_raw_prevalidated_times[sample] = measure_core!(
                prevalidated_batch, model, raw_forcing, FT(10),
                prevalidated_workspace, prevalidated_diagnostics)
            warm_raw_checked_times[sample] = measure_core!(
                checked_batch, model, raw_forcing, FT(10),
                checked_workspace, checked_diagnostics)
            warm_terminal_active_times[sample] = elapsed_cuda() do
                advance!(terminal_active_batch, model, raw_forcing, FT(10);
                    workspace = terminal_active_workspace,
                    diagnostics = terminal_active_diagnostics)
            end
            warm_terminal_padded_times[sample] = elapsed_cuda() do
                advance!(terminal_padded_batch, model, raw_forcing, FT(10);
                    workspace = terminal_padded_workspace,
                    diagnostics = terminal_padded_diagnostics)
            end
        else
            warm_raw_checked_times[sample] = measure_core!(
                checked_batch, model, raw_forcing, FT(10),
                checked_workspace, checked_diagnostics)
            warm_raw_prevalidated_times[sample] = measure_core!(
                prevalidated_batch, model, raw_forcing, FT(10),
                prevalidated_workspace, prevalidated_diagnostics)
            warm_coupling_times[sample] = measure_adapter!(
                state, operator, FT(10), workspace)
            warm_terminal_padded_times[sample] = elapsed_cuda() do
                advance!(terminal_padded_batch, model, raw_forcing, FT(10);
                    workspace = terminal_padded_workspace,
                    diagnostics = terminal_padded_diagnostics)
            end
            warm_terminal_active_times[sample] = elapsed_cuda() do
                advance!(terminal_active_batch, model, raw_forcing, FT(10);
                    workspace = terminal_active_workspace,
                    diagnostics = terminal_active_diagnostics)
            end
        end
    end

    Array(checked_concentrations) ≈ Array(prevalidated_concentrations) ||
        error("warm checked and prevalidated core paths disagree")
    checked_diagnostics.counter_storage ==
        prevalidated_diagnostics.counter_storage ||
        error("warm checked and prevalidated counters disagree")
    workspace.diagnostics.counter_storage ==
        prevalidated_diagnostics.counter_storage ||
        error("warm transport adapter and raw core counters disagree")
    Array(terminal_active_concentrations)[1:terminal_active_cells, :] ≈
        Array(terminal_padded_concentrations)[1:terminal_active_cells, :] ||
        error("warm active and padded terminal paths disagree")
    terminal_active_diagnostics.counter_storage[1:terminal_active_cells, :] ==
        terminal_padded_diagnostics.counter_storage[1:terminal_active_cells, :] ||
        error("warm active and padded terminal counters disagree")
    warm_adapter_concentrations = reshape(
        Array(state.tracers_raw), cell_count, length(species)) .* (
            operator.dry_air_molar_mass / FT(10) * FT(2.5e19))
    warm_adapter_concentrations ≈ Array(prevalidated_concentrations) ||
        error("warm transport adapter and raw core paths disagree")

    plan = chemistry_workspace_plan(workspace)
    storage = chemistry_workspace_storage_bytes(workspace)
    validation_median = median(validation_times)
    coupling_median = median(coupling_times)
    raw_checked_median = median(raw_checked_times)
    raw_prevalidated_median = median(raw_prevalidated_times)
    terminal_active_median = median(terminal_active_times)
    terminal_padded_median = median(terminal_padded_times)
    warm_coupling_median = median(warm_coupling_times)
    warm_raw_checked_median = median(warm_raw_checked_times)
    warm_raw_prevalidated_median = median(warm_raw_prevalidated_times)
    warm_terminal_active_median = median(warm_terminal_active_times)
    warm_terminal_padded_median = median(warm_terminal_padded_times)
    return Dict(
        "schema_version" => 4,
        "benchmark" => "AtmosTransport AtmosChemistry adapter",
        "julia_version" => string(VERSION),
        "device" => CUDA.name(CUDA.device()),
        "gpu_uuid" => string(CUDA.uuid(CUDA.device())),
        "cuda_runtime_version" => string(CUDA.runtime_version()),
        "cuda_driver_version" => string(CUDA.driver_version()),
        "device_memory_bytes" => CUDA.total_memory(),
        "precision" => string(FT),
        "cells" => cell_count,
        "species" => length(species),
        "tile_cells" => plan.tile_cells,
        "validation_chunk_cells" => extension._VALIDATION_CHUNK_CELLS,
        "validation_median_seconds" => validation_median,
        "validation_p95_seconds" => percentile(validation_times, 0.95),
        "validation_cells_per_second" => cell_count / validation_median,
        "state_staging_seconds" => state_staging.time,
        "state_staging_host_bytes" => state_staging.bytes,
        "workspace_allocation_seconds" => workspace_allocation.time,
        "workspace_allocation_host_bytes" => workspace_allocation.bytes,
        "first_coupling_seconds" => first_coupling.time,
        "first_coupling_host_bytes" => first_coupling.bytes,
        "adapter_state_to_first_result_seconds" =>
            state_staging.time + workspace_allocation.time +
            first_coupling.time,
        "coupling_median_seconds" => warm_coupling_median,
        "coupling_p95_seconds" => percentile(warm_coupling_times, 0.95),
        "raw_checked_median_seconds" => warm_raw_checked_median,
        "raw_prevalidated_median_seconds" => warm_raw_prevalidated_median,
        "adapter_to_raw_prevalidated_ratio" =>
            warm_coupling_median / warm_raw_prevalidated_median,
        "cold_start_coupling_median_seconds" => coupling_median,
        "cold_start_raw_checked_median_seconds" => raw_checked_median,
        "cold_start_raw_prevalidated_median_seconds" =>
            raw_prevalidated_median,
        "cold_start_adapter_to_raw_prevalidated_ratio" =>
            coupling_median / raw_prevalidated_median,
        "terminal_capacity_cells" => terminal_capacity,
        "terminal_active_cells" => terminal_active_cells,
        "terminal_active_median_seconds" => warm_terminal_active_median,
        "terminal_padded_median_seconds" => warm_terminal_padded_median,
        "terminal_active_speedup" =>
            warm_terminal_padded_median / warm_terminal_active_median,
        "cold_start_terminal_active_median_seconds" =>
            terminal_active_median,
        "cold_start_terminal_padded_median_seconds" =>
            terminal_padded_median,
        "cold_start_terminal_active_speedup" =>
            terminal_padded_median / terminal_active_median,
        "planned_backend_bytes" => plan.required_backend_bytes,
        "planned_host_bytes" => plan.host_persistent_bytes,
        "planned_host_forcing_stage_bytes" =>
            plan.allocations.host_forcing_stage_bytes,
        "has_host_forcing_stage" => workspace.host_forcing_stage !== nothing,
        "measured_incremental_logical_owned_bytes" => storage.total,
        "memory_accounting" => "incremental logical owned bytes; excludes " *
            "caller-owned state, source forcing, allocator pools, JIT, and driver",
        "matched_state_checks_passed" => true,
        "comparison_order" => "alternating_adapter_and_core",
        "provenance" => provenance,
        "samples" => samples,
    )
end

function main(args)
    cell_count = isempty(args) ? 65_536 : parse(Int, args[1])
    precision = length(args) < 2 ? "float64" : lowercase(args[2])
    FT = precision == "float32" ? Float32 :
         precision == "float64" ? Float64 :
         error("precision must be float32 or float64")
    output = length(args) < 3 ? nothing : args[3]
    result = benchmark_adapter(FT, cell_count)

    @printf("validation: %.3f ms (%.3e cells/s)\n",
            1e3 * result["validation_median_seconds"],
            result["validation_cells_per_second"])
    @printf("coupling:   %.3f ms\n",
            1e3 * result["coupling_median_seconds"])
    @printf("startup:    %.3f ms state, %.3f ms workspace, %.3f ms first call\n",
            1e3 * result["state_staging_seconds"],
            1e3 * result["workspace_allocation_seconds"],
            1e3 * result["first_coupling_seconds"])
    @printf("raw core:   %.3f ms checked, %.3f ms prevalidated\n",
            1e3 * result["raw_checked_median_seconds"],
            1e3 * result["raw_prevalidated_median_seconds"])
    @printf("tail core:  %.3f ms active, %.3f ms padded (%.2fx)\n",
            1e3 * result["terminal_active_median_seconds"],
            1e3 * result["terminal_padded_median_seconds"],
            result["terminal_active_speedup"])
    @printf("tile:       %d cells\n", result["tile_cells"])

    if output === nothing
        TOML.print(stdout, result; sorted = true)
        println()
    else
        open(output, "w") do io
            TOML.print(io, result; sorted = true)
        end
        println("wrote $output")
    end
end

main(ARGS)
