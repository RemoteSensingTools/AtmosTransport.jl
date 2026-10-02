module AtmosTransportAtmosChemistryExt

import AtmosChemistry
import AtmosTransport
import KernelAbstractions
using KernelAbstractions: @index, @kernel

const ATC = AtmosTransport.Operators.Chemistry
const ATM = AtmosTransport.Models
const _DEFAULT_TRANSPORT_CHEMISTRY_TILE_CELLS = 4096
const _VALIDATION_CHUNK_CELLS = 256

# Model and workspace types

function AtmosTransport.AtmosChemistryOperator(
        model::AtmosChemistry.ChemistryModel, forcing;
        tracer_names = AtmosChemistry.species_names(model),
        workspace_policy = nothing,
        dry_air_molar_mass::Real = 28.9647)
    names = Tuple(Symbol(name) for name in tracer_names)
    length(names) == length(AtmosChemistry.species_names(model)) ||
        throw(DimensionMismatch(
            "one transport tracer name is required for every active " *
            "chemistry species"))
    length(unique(names)) == length(names) || throw(ArgumentError(
        "chemistry tracer names must be unique"))
    workspace_policy === nothing ||
        workspace_policy isa AtmosChemistry.WorkspacePolicy ||
        throw(ArgumentError(
            "workspace_policy must be an AtmosChemistry.WorkspacePolicy"))
    FT = eltype(model)
    isfinite(dry_air_molar_mass) && dry_air_molar_mass > 0 ||
        throw(ArgumentError(
            "dry_air_molar_mass must be finite and positive"))
    provider = ATC.chemistry_forcing_provider(forcing)
    return AtmosTransport.AtmosChemistryOperator{
        typeof(model), typeof(provider), typeof(names),
        typeof(workspace_policy), FT}(
            model, provider, names, workspace_policy, FT(dry_air_molar_mass))
end

abstract type AbstractTransportChemistryLayout end

"Packed cells with no excluded halo storage."
struct LinearCellLayout <: AbstractTransportChemistryLayout end

"Interior cells of one cubed-sphere panel; halo width is a type parameter."
struct CubedSphereInteriorLayout{H} <: AbstractTransportChemistryLayout
    x_cells :: Int32
    y_cells :: Int32
end

struct TransportChemistryWorkspace{
        C, W, D, TD, DS, DC, F, I, M, E, V, H, G, B, S, P,
        L <: AbstractTransportChemistryLayout}
    concentrations :: C
    chemistry      :: W
    diagnostics    :: D
    tile_diagnostics:: TD
    backend_status :: DS
    backend_counters:: DC
    forcing        :: F
    tracer_indices :: I
    molar_masses   :: M
    conversion_error:: E
    validation_chunks:: V
    conversion_error_host:: H
    host_forcing_stage:: G
    batch_cells    :: B
    warm_steps     :: S
    plan           :: P
    layout         :: L
    cell_count     :: Int
end

struct TransportChemistryPlan{C, A}
    cell_count                       :: Int
    diagnostic_count                 :: Int
    panel_count                      :: Int
    tile_cells                       :: Int
    species_count                    :: Int
    chemistry                        :: C
    allocations                      :: A
    adapter_tile_bytes               :: Int
    adapter_persistent_bytes         :: Int
    required_backend_bytes           :: Int
    construction_peak_backend_bytes :: Int
    peak_backend_bytes               :: Int
    host_persistent_bytes            :: Int
    construction_transient_host_bytes:: Int
    construction_peak_host_bytes     :: Int
    planned_backend_bytes            :: Int
    planned_host_bytes               :: Int
    budget_bytes                     :: Int
end

function ATC.chemistry_plan_accounting(plan::TransportChemistryPlan)
    return (; backend = plan.planned_backend_bytes,
            host = plan.planned_host_bytes,
            peak_backend = plan.peak_backend_bytes,
            budget = plan.budget_bytes)
end

function Base.show(io::IO, plan::TransportChemistryPlan)
    print(io, "TransportChemistryPlan(cells=", plan.cell_count,
          ", tile_cells=", plan.tile_cells,
          ", species=", plan.species_count,
          ", backend=", plan.required_backend_bytes, " bytes",
          ", peak=", plan.peak_backend_bytes, " bytes",
          ", host=", plan.host_persistent_bytes, " bytes",
          ", host_peak=", plan.construction_peak_host_bytes, " bytes",
          ", budget=", plan.budget_bytes, " bytes)")
end

function _tracer_indices(operator, state)
    names = state.tracer_names
    indices = Tuple(findfirst(==(name), names) for name in operator.tracer_names)
    missing = Tuple(operator.tracer_names[i] for i in eachindex(indices)
                    if indices[i] === nothing)
    isempty(missing) || throw(ArgumentError(
        "transport state is missing active chemistry tracers $missing"))
    return Tuple(Int32(index) for index in indices)
end

_workspace_policy(operator) = operator.workspace_policy === nothing ?
    AtmosChemistry.WorkspacePolicy() : operator.workspace_policy

_needs_host_forcing_stage(::ATC.ConstantChemistryForcing) = false
_needs_host_forcing_stage(::ATC.CallUpdatedChemistryForcing) = true

function _tile_workspace_policy(policy, tile_cells)
    return AtmosChemistry.WorkspacePolicy(
        memory_fraction = policy.memory_fraction,
        reserve_bytes = policy.reserve_bytes,
        max_bytes = policy.max_bytes,
        fixed_tile_cells = tile_cells)
end

_structural_storage_bytes(::AtmosChemistry.NoField, layout, full_shape) = 0
_structural_storage_bytes(value::Number, layout, full_shape) = sizeof(value)

function _projected_cell_count(values::AbstractArray, layout, full_shape)
    layout isa LinearCellLayout && return size(values, 1)
    interior_cells = Int(layout.x_cells) * Int(layout.y_cells) * full_shape[3]
    size(values, 1) in (interior_cells, prod(full_shape)) ||
        throw(DimensionMismatch(
            "cubed-sphere forcing has $(size(values, 1)) rows; expected " *
            "$(prod(full_shape)) halo-padded or $interior_cells interior rows"))
    return interior_cells
end

_structural_storage_bytes(values::AbstractArray,
                          ::LinearCellLayout, full_shape) =
    length(values) * sizeof(eltype(values))

function _structural_storage_bytes(
        values::AbstractArray, layout::CubedSphereInteriorLayout, full_shape)
    interior_cells = Int(layout.x_cells) * Int(layout.y_cells) * full_shape[3]
    length(values) in (interior_cells, prod(full_shape)) ||
        throw(DimensionMismatch(
            "cubed-sphere forcing has $(length(values)) values; expected " *
            "$(prod(full_shape)) halo-padded or $interior_cells interior values"))
    return interior_cells * sizeof(eltype(values))
end

function _structural_storage_bytes(values::NamedTuple, layout, full_shape)
    return sum(value -> _structural_storage_bytes(value, layout, full_shape),
               values; init = 0)
end

function _structural_storage_bytes(
        fields::AtmosChemistry.PackedForcingFields{
            Names, AtmosChemistry.ConstantForcingFields},
        layout, full_shape) where Names
    values = getfield(fields, :values)
    return length(values) * sizeof(eltype(values))
end

function _structural_storage_bytes(
        fields::AtmosChemistry.PackedForcingFields{
            Names, AtmosChemistry.CellForcingFields},
        layout, full_shape) where Names
    values = getfield(fields, :values)
    rows = _projected_cell_count(values, layout, full_shape)
    return rows * size(values, 2) * sizeof(eltype(values))
end

function _packed_field_storage_bytes(
        values::NamedTuple, required, ::Type{FT}, layout, full_shape) where FT
    for name in required
        hasproperty(values, name) || throw(KeyError(name))
    end
    return length(required) * sizeof(FT)
end

function _packed_field_storage_bytes(
        fields::AtmosChemistry.PackedForcingFields{
            Available, AtmosChemistry.ConstantForcingFields},
        required, ::Type{FT}, layout, full_shape) where {Available, FT}
    all(name -> name in Available, required) ||
        throw(ArgumentError("packed forcing is missing required fields"))
    return length(required) * sizeof(FT)
end


function _packed_field_storage_bytes(
        fields::AtmosChemistry.PackedForcingFields{
            Available, AtmosChemistry.CellForcingFields},
        required, ::Type{FT}, layout, full_shape) where {Available, FT}
    all(name -> name in Available, required) ||
        throw(ArgumentError("packed forcing is missing required fields"))
    rows = _projected_cell_count(
        getfield(fields, :values), layout, full_shape)
    return rows * length(required) * sizeof(FT)
end

function _canonical_forcing_storage_bytes(
        forcing::AtmosChemistry.ChemistryForcing, mechanism,
        ::Type{FT}, layout, full_shape) where FT
    requirements = AtmosChemistry.required_forcings(mechanism)
    return _structural_storage_bytes(
               forcing.temperature, layout, full_shape) +
           _structural_storage_bytes(
               forcing.pressure, layout, full_shape) +
           _structural_storage_bytes(
               forcing.air_number_density, layout, full_shape) +
           _packed_field_storage_bytes(
               forcing.fixed_species, requirements.fixed_species,
               FT, layout, full_shape) +
           _packed_field_storage_bytes(
               forcing.photolysis, requirements.photolysis,
               FT, layout, full_shape) +
           _structural_storage_bytes(
               forcing.auxiliaries, layout, full_shape)
end

function _canonical_forcing_storage_bytes(
        forcing::Tuple, mechanism, ::Type{FT}, layout, full_shape) where FT
    return sum(value -> _canonical_forcing_storage_bytes(
                   value, mechanism, FT, layout, full_shape),
               forcing; init = 0)
end

_host_source(value::AbstractArray) =
    KernelAbstractions.get_backend(value) isa KernelAbstractions.CPU

_construction_host_stage_bytes(
    value, required, ::Type, architecture, layout, full_shape) = 0

function _construction_host_stage_bytes(
        values::NamedTuple, required, ::Type{FT},
        ::AtmosChemistry.NativeGPU, layout, full_shape) where FT
    return isempty(required) ? 0 : length(required) * sizeof(FT)
end

function _construction_host_stage_bytes(
        fields::AtmosChemistry.PackedForcingFields{
            <:Any, AtmosChemistry.ConstantForcingFields},
        required, ::Type{FT}, ::AtmosChemistry.NativeGPU,
        layout, full_shape) where FT
    isempty(required) && return 0
    return _host_source(fields.values) ? length(required) * sizeof(FT) : 0
end

function _construction_host_stage_bytes(
        fields::AtmosChemistry.PackedForcingFields{
            <:Any, AtmosChemistry.CellForcingFields},
        required, ::Type{FT}, ::AtmosChemistry.NativeGPU,
        layout, full_shape) where FT
    isempty(required) && return 0
    _host_source(fields.values) || return 0
    rows = _projected_cell_count(fields.values, layout, full_shape)
    index_bytes = layout isa LinearCellLayout ?
        length(required) * sizeof(Int32) : 0
    return rows * length(required) * sizeof(FT) + index_bytes
end

_construction_field_host_stage_bytes(
    value::AbstractArray, ::Type, architecture, ::LinearCellLayout,
    full_shape) = 0
_construction_field_host_stage_bytes(
    value::Number, ::Type, architecture, layout, full_shape) = 0
_construction_field_host_stage_bytes(
    value::AtmosChemistry.NoField, ::Type, architecture, layout, full_shape) = 0

function _construction_field_host_stage_bytes(
        value::AbstractArray, ::Type{FT}, ::AtmosChemistry.NativeGPU,
        layout::CubedSphereInteriorLayout, full_shape) where FT
    _host_source(value) || return 0
    interior_cells = Int(layout.x_cells) * Int(layout.y_cells) * full_shape[3]
    return length(value) == interior_cells ? 0 : interior_cells * sizeof(FT)
end

function _construction_auxiliary_host_stage_bytes(
        values::NamedTuple, ::Type{FT}, architecture, layout, full_shape) where FT
    return sum(value -> _construction_field_host_stage_bytes(
                   value, FT, architecture, layout, full_shape),
               values; init = 0)
end

function _construction_auxiliary_host_stage_bytes(
        fields::AtmosChemistry.PackedForcingFields{Names}, ::Type{FT},
        architecture, layout, full_shape) where {Names, FT}
    return _construction_host_stage_bytes(
        fields, Names, FT, architecture, layout, full_shape)
end

function _construction_forcing_host_stage_bytes(
        forcing::AtmosChemistry.ChemistryForcing, mechanism, ::Type{FT},
        architecture::AtmosChemistry.NativeGPU, layout, full_shape) where FT
    requirements = AtmosChemistry.required_forcings(mechanism)
    return sum((
        _construction_field_host_stage_bytes(
            forcing.temperature, FT, architecture, layout, full_shape),
        _construction_field_host_stage_bytes(
            forcing.pressure, FT, architecture, layout, full_shape),
        _construction_field_host_stage_bytes(
            forcing.air_number_density, FT, architecture, layout, full_shape),
        _construction_host_stage_bytes(
            forcing.fixed_species, requirements.fixed_species, FT,
            architecture, layout, full_shape),
        _construction_host_stage_bytes(
            forcing.photolysis, requirements.photolysis, FT,
            architecture, layout, full_shape),
        _construction_auxiliary_host_stage_bytes(
            forcing.auxiliaries, FT, architecture, layout, full_shape)))
end

_construction_forcing_host_stage_bytes(
    forcing, mechanism, ::Type, architecture, layout, full_shape) = 0

function _construction_forcing_host_stage_bytes(
        forcing::Tuple, mechanism, ::Type{FT}, architecture, layout,
        full_shape) where FT
    return sum(value -> _construction_forcing_host_stage_bytes(
                   value, mechanism, FT, architecture, layout, full_shape),
               forcing; init = 0)
end

function _projected_forcing_storage_bytes(
        forcing::AtmosChemistry.ChemistryForcing,
                                          layout, full_shape)
    return _structural_storage_bytes(
               forcing.temperature, layout, full_shape) +
           _structural_storage_bytes(
               forcing.pressure, layout, full_shape) +
           _structural_storage_bytes(
               forcing.air_number_density, layout, full_shape) +
           _structural_storage_bytes(
               forcing.fixed_species, layout, full_shape) +
           _structural_storage_bytes(
               forcing.photolysis, layout, full_shape) +
           _structural_storage_bytes(
               forcing.auxiliaries, layout, full_shape)
end

function _projected_forcing_storage_bytes(forcing::Tuple, layout, full_shape)
    return sum(value -> _projected_forcing_storage_bytes(
                   value, layout, full_shape), forcing; init = 0)
end

_forcing_storage_bytes(forcing) = _projected_forcing_storage_bytes(
    forcing, LinearCellLayout(), nothing)

# Canonical memory planning

function _transport_tile_plan(model, cell_count, diagnostic_count,
                              forcing, policy, needs_host_stage, layout,
                              full_shape = nothing)
    _validate_source_forcing(model, forcing, cell_count)
    requested = min(cell_count,
        policy.fixed_tile_cells == 0 ?
        _DEFAULT_TRANSPORT_CHEMISTRY_TILE_CELLS : policy.fixed_tile_cells)
    probe_policy = policy.fixed_tile_cells == 0 ?
        AtmosChemistry.WorkspacePolicy(
            memory_fraction = policy.memory_fraction,
            reserve_bytes = policy.reserve_bytes,
            max_bytes = policy.max_bytes) :
        _tile_workspace_policy(policy, requested)
    automatic = AtmosChemistry.workspace_plan(
        model, requested; policy = probe_policy)
    candidate = min(requested, automatic.tile_cells)
    FT = eltype(model)
    species_count = length(AtmosChemistry.species_names(model))
    mechanism = AtmosChemistry.compiled_mechanism(model)
    architecture = AtmosChemistry.architecture(model)
    forcing_bytes = _canonical_forcing_storage_bytes(
        forcing, mechanism, FT, layout, full_shape)
    construction_transient_host_bytes =
        _construction_forcing_host_stage_bytes(
            forcing, mechanism, FT, architecture, layout, full_shape)
    panel_count = div(diagnostic_count, cell_count)
    validation_chunks_per_panel = cld(
        cell_count, _VALIDATION_CHUNK_CELLS)
    validation_chunk_count = panel_count * validation_chunks_per_panel
    diagnostic_bytes_per_cell =
        sizeof(Int8) + 5 * sizeof(Int32) + sizeof(FT)
    backend_diagnostic_bytes =
        architecture isa AtmosChemistry.NativeGPU ?
        diagnostic_count * (sizeof(Int8) + 5 * sizeof(Int32)) : 0
    host_forcing_stage_bytes =
        architecture isa AtmosChemistry.NativeGPU && needs_host_stage ?
        cell_count * sizeof(FT) : 0

    function candidate_plan(candidate)
        chemistry_policy = _tile_workspace_policy(policy, candidate)
        chemistry_plan = AtmosChemistry.workspace_plan(
            model, candidate; policy = chemistry_policy,
            forcing = forcing isa Tuple ? nothing : forcing)
        host_diagnostic_bytes = architecture isa AtmosChemistry.CPU ?
            (diagnostic_count + candidate) * diagnostic_bytes_per_cell +
                sizeof(Int32) :
            diagnostic_count * diagnostic_bytes_per_cell + sizeof(Int32)
        adapter_tile_bytes = candidate *
            (species_count * sizeof(FT) + sizeof(Int32))
        adapter_persistent_bytes =
            species_count * (sizeof(FT) + sizeof(Int32)) +
            (1 + validation_chunk_count) * sizeof(Int32) +
            diagnostic_count * sizeof(FT) + forcing_bytes +
            backend_diagnostic_bytes
        architecture isa AtmosChemistry.CPU &&
            (adapter_persistent_bytes += host_diagnostic_bytes)
        required = chemistry_plan.persistent_bytes +
                   chemistry_plan.tile_bytes + adapter_tile_bytes +
                   adapter_persistent_bytes
        construction_transient_backend_bytes = 0
        construction_peak_backend_bytes = required
        peak_backend_bytes = required
        feasible = peak_backend_bytes <= chemistry_plan.budget_bytes
        plan = if feasible
            allocations = (
                concentration_tile_bytes = candidate * species_count * sizeof(FT),
                batch_index_bytes = candidate * sizeof(Int32),
                species_mapping_bytes = species_count * sizeof(Int32),
                molar_mass_bytes = species_count * sizeof(FT),
                warm_step_bytes = diagnostic_count * sizeof(FT),
                validation_chunk_bytes =
                    validation_chunk_count * sizeof(Int32),
                backend_diagnostic_bytes,
                forcing_bytes,
                host_diagnostic_bytes,
                host_forcing_stage_bytes,
                construction_transient_backend_bytes,
                construction_transient_host_bytes,
            )
            host_persistent_bytes = host_diagnostic_bytes +
                                    host_forcing_stage_bytes
            construction_peak_host_bytes = host_persistent_bytes +
                                           construction_transient_host_bytes
            planned_host_bytes =
                architecture isa AtmosChemistry.NativeGPU ?
                host_persistent_bytes : 0
            TransportChemistryPlan(
                cell_count, diagnostic_count, panel_count, candidate,
                species_count, chemistry_plan, allocations,
                adapter_tile_bytes, adapter_persistent_bytes, required,
                construction_peak_backend_bytes, peak_backend_bytes,
                host_persistent_bytes, construction_transient_host_bytes,
                construction_peak_host_bytes, required, planned_host_bytes,
                chemistry_plan.budget_bytes)
        else
            nothing
        end
        chemistry_bytes = chemistry_plan.persistent_bytes +
                          chemistry_plan.tile_bytes
        return (; feasible, plan, chemistry_plan, chemistry_bytes, required,
                peak_backend_bytes, adapter_tile_bytes,
                adapter_persistent_bytes)
    end

    first_result = candidate_plan(candidate)
    first_result.feasible && return first_result.plan
    if policy.fixed_tile_cells != 0
        throw(ArgumentError(
            "transport chemistry fixed tile exceeds its workspace budget: " *
            "budget=$(first_result.chemistry_plan.budget_bytes), " *
            "required=$(first_result.required), " *
            "peak=$(first_result.peak_backend_bytes), " *
            "chemistry=$(first_result.chemistry_bytes), " *
            "adapter_tile=$(first_result.adapter_tile_bytes), " *
            "adapter_persistent=$(first_result.adapter_persistent_bytes)"))
    end

    lower = 1
    upper = candidate - 1
    best = nothing
    smallest = first_result
    while lower <= upper
        midpoint = lower + div(upper - lower, 2)
        result = candidate_plan(midpoint)
        midpoint == 1 && (smallest = result)
        if result.feasible
            best = result.plan
            lower = midpoint + 1
        else
            upper = midpoint - 1
        end
    end
    best !== nothing && return best

    smallest = smallest === first_result ? candidate_plan(1) : smallest
    throw(ArgumentError(
        "transport chemistry workspace budget is infeasible: " *
        "budget=$(smallest.chemistry_plan.budget_bytes), " *
        "required=$(smallest.required), peak=$(smallest.peak_backend_bytes), " *
        "chemistry=$(smallest.chemistry_bytes), " *
        "adapter_tile=$(smallest.adapter_tile_bytes), " *
        "adapter_persistent=$(smallest.adapter_persistent_bytes)"))
end

function _validate_source_forcing(model, forcing, cell_count)
    FT = eltype(model)
    AtmosChemistry._validate_forcing_precision(FT, forcing)
    _validate_initial_forcing_source(
        AtmosChemistry.architecture(model), forcing, "forcing")
    AtmosChemistry._validate_forcing_schema(
        AtmosChemistry.compiled_mechanism(model), forcing,
        Base.OneTo(cell_count))
    return nothing
end

function _validate_source_forcing(model, forcing::Tuple, cell_count)
    for value in forcing
        _validate_source_forcing(model, value, cell_count)
    end
    return nothing
end

# Backend storage and residency

_allocate_storage(::AtmosChemistry.CPU, ::Type{T}, dims::Tuple) where T =
    Array{T}(undef, dims)

_allocate_host_forcing_stage(
    ::AtmosChemistry.NativeGPU, ::Type{T}, cell_count, ::Val{true}) where T =
    Vector{T}(undef, cell_count)
_allocate_host_forcing_stage(
    ::AtmosChemistry.NativeGPU, ::Type, cell_count, ::Val{false}) = nothing
_allocate_host_forcing_stage(
    ::AtmosChemistry.CPU, ::Type, cell_count, ::Val) = nothing

_allocate_tile_diagnostics(
    ::AtmosChemistry.CPU, ::Type{FT}, tile_cells) where FT =
    AtmosChemistry.allocate_diagnostics(FT, tile_cells)
_allocate_tile_diagnostics(
    ::AtmosChemistry.NativeGPU, ::Type, tile_cells) = nothing

_allocate_backend_diagnostics(
    ::AtmosChemistry.CPU, diagnostic_count) = (nothing, nothing)
function _allocate_backend_diagnostics(
        architecture::AtmosChemistry.NativeGPU, diagnostic_count)
    status = _allocate_storage(architecture, Int8, (diagnostic_count,))
    counters = _allocate_storage(
        architecture, Int32, (diagnostic_count, 5))
    return status, counters
end

function _allocate_storage(architecture::AtmosChemistry.NativeGPU,
                           ::Type{T}, dims::Tuple) where T
    return KernelAbstractions.allocate(architecture.backend, T, dims)
end

function _allocate_storage(architecture, ::Type, ::Tuple)
    throw(ArgumentError(
        "AtmosTransport coupling supports AtmosChemistry CPU and NativeGPU " *
        "architectures; received $(typeof(architecture))"))
end

function _validate_backend(::AtmosChemistry.CPU, array, label)
    backend = KernelAbstractions.get_backend(array)
    backend isa KernelAbstractions.CPU || throw(ArgumentError(
        "$label resides on $(typeof(backend)), but chemistry uses CPU()"))
    return nothing
end

function _validate_backend(architecture::AtmosChemistry.NativeGPU, array, label)
    backend = KernelAbstractions.get_backend(array)
    typeof(backend) === typeof(architecture.backend) || throw(ArgumentError(
        "$label resides on $(typeof(backend)), but chemistry uses " *
        "$(typeof(architecture.backend))"))
    AtmosChemistry._validate_native_gpu_device(
        architecture.backend, array, label)
    return nothing
end

function _validate_backend(architecture, array, label)
    throw(ArgumentError(
        "AtmosTransport coupling does not support $(typeof(architecture))"))
end

_validate_forcing_backend(architecture, value::Number, label) = nothing
_validate_forcing_backend(architecture, ::AtmosChemistry.NoField, label) = nothing
_validate_forcing_backend(
    architecture, ::AtmosChemistry.AbstractForcingEnvelope, label) = nothing

function _validate_forcing_backend(architecture, value::AbstractArray, label)
    return _validate_backend(architecture, value, label)
end

function _validate_forcing_backend(architecture, values::NamedTuple, label)
    for name in propertynames(values)
        _validate_forcing_backend(
            architecture, getproperty(values, name), "$label.$name")
    end
    return nothing
end


function _validate_forcing_backend(architecture, values::Tuple, label)
    for index in eachindex(values)
        _validate_forcing_backend(
            architecture, values[index], "$label[$index]")
    end
    return nothing
end

function _validate_forcing_backend(
        architecture, fields::AtmosChemistry.PackedForcingFields, label)
    return _validate_forcing_backend(
        architecture, fields.values, "$label.values")
end

function _validate_forcing_backend(
        architecture, forcing::AtmosChemistry.ChemistryForcing, label)
    _validate_forcing_backend(
        architecture, forcing.temperature, "$label.temperature")
    _validate_forcing_backend(
        architecture, forcing.pressure, "$label.pressure")
    _validate_forcing_backend(
        architecture, forcing.air_number_density,
        "$label.air_number_density")
    _validate_forcing_backend(
        architecture, forcing.fixed_species, "$label.fixed_species")
    _validate_forcing_backend(
        architecture, forcing.photolysis, "$label.photolysis")
    _validate_forcing_backend(
        architecture, forcing.auxiliaries, "$label.auxiliaries")
    return nothing
end

function _validate_transport_forcing(model, forcing, cell_count)
    FT = eltype(model)
    AtmosChemistry._validate_forcing_precision(FT, forcing)
    AtmosChemistry._validate_forcing_schema(
        AtmosChemistry.compiled_mechanism(model), forcing,
        Base.OneTo(cell_count))
    _validate_forcing_backend(
        AtmosChemistry.architecture(model), forcing, "forcing")
    return nothing
end

function _validate_transport_forcing(model, forcing::Tuple, cell_count)
    for panel in eachindex(forcing)
        _validate_transport_forcing(model, forcing[panel], cell_count)
    end
    return nothing
end

function _validate_workspace_backend!(architecture, workspace)
    storage = (
        ("chemistry concentration tile", workspace.concentrations),
        ("chemistry tracer indices", workspace.tracer_indices),
        ("chemistry molar masses", workspace.molar_masses),
        ("chemistry conversion result", workspace.conversion_error),
        ("chemistry validation chunks", workspace.validation_chunks),
        ("chemistry batch cells", workspace.batch_cells),
        ("chemistry warm steps", workspace.warm_steps),
    )
    for (label, array) in storage
        _validate_backend(architecture, array, label)
    end
    _validate_forcing_backend(
        architecture, workspace.forcing, "workspace forcing")
    return nothing
end

function _validate_apply_backend!(architecture,
                                  state::AtmosTransport.CellState,
                                  workspace)
    _validate_backend(architecture, state.air_mass, "transport air mass")
    _validate_backend(
        architecture, state.tracers_raw, "transport tracer storage")
    return _validate_workspace_backend!(architecture, workspace)
end

function _validate_apply_backend!(architecture,
                                  state::AtmosTransport.CubedSphereState,
                                  workspace)
    for panel in 1:6
        _validate_backend(
            architecture, state.air_mass[panel], "panel $panel air mass")
        _validate_backend(architecture, state.tracers_raw[panel],
                          "panel $panel tracer storage")
    end
    return _validate_workspace_backend!(architecture, workspace)
end

# Canonical forcing ownership

_validate_initial_forcing_source(architecture, source::Number, label) = nothing
_validate_initial_forcing_source(
    architecture, source::AtmosChemistry.NoField, label) = nothing
_validate_initial_forcing_source(
    architecture, source::AtmosChemistry.AbstractForcingEnvelope, label) = nothing

function _validate_initial_forcing_source(architecture, source::NamedTuple, label)
    for name in propertynames(source)
        _validate_initial_forcing_source(
            architecture, getproperty(source, name), "$label.$name")
    end
    return nothing
end

function _validate_initial_forcing_source(architecture, source::Tuple, label)
    for index in eachindex(source)
        _validate_initial_forcing_source(
            architecture, source[index], "$label[$index]")
    end
    return nothing
end

function _validate_initial_forcing_source(
        architecture, source::AtmosChemistry.PackedForcingFields, label)
    return _validate_initial_forcing_source(
        architecture, source.values, "$label.values")
end

function _validate_initial_forcing_source(
        architecture, source::AtmosChemistry.ChemistryForcing, label)
    for name in (:temperature, :pressure, :air_number_density,
                 :fixed_species, :photolysis, :auxiliaries)
        _validate_initial_forcing_source(
            architecture, getproperty(source, name), "$label.$name")
    end
    return nothing
end

function _validate_initial_forcing_source(
        ::AtmosChemistry.CPU, source::AbstractArray, label)
    KernelAbstractions.get_backend(source) isa KernelAbstractions.CPU ||
        throw(ArgumentError(
            "$label must be host-resident for CPU chemistry"))
    return nothing
end

function _validate_initial_forcing_source(
        architecture::AtmosChemistry.NativeGPU,
        source::AbstractArray, label)
    KernelAbstractions.get_backend(source) isa KernelAbstractions.CPU &&
        return nothing
    return _validate_backend(architecture, source, label)
end

function _copy_initial_forcing!(destination, source, architecture, label)
    _validate_initial_forcing_source(architecture, source, label)
    copyto!(destination, source)
    return destination
end

function _owned_linear_forcing_field(
        source::AbstractArray, architecture, label)
    _validate_initial_forcing_source(architecture, source, label)
    destination = _allocate_storage(
        architecture, eltype(source), size(source))
    return _copy_initial_forcing!(
        destination, source, architecture, label)
end

_owned_forcing_field(source::Number, architecture, layout,
                     full_shape, label) = source
_owned_forcing_field(source::AtmosChemistry.NoField, architecture, layout,
                     full_shape, label) = source

function _owned_forcing_field(source::AbstractArray, architecture,
                              ::LinearCellLayout, full_shape, label)
    return _owned_linear_forcing_field(source, architecture, label)
end

function _copy_projected_forcing!(destination, source, architecture,
                                  layout, label)
    _validate_initial_forcing_source(architecture, source, label)
    source_backend = KernelAbstractions.get_backend(source)
    destination_backend = KernelAbstractions.get_backend(destination)
    if source_backend isa KernelAbstractions.CPU &&
       destination_backend isa KernelAbstractions.CPU
        @inbounds for cell in eachindex(destination)
            destination[cell] = source[_full_panel_index(cell, layout)]
        end
    elseif source_backend isa KernelAbstractions.CPU
        stage = Array{eltype(source)}(undef, size(destination))
        @inbounds for cell in eachindex(stage)
            stage[cell] = source[_full_panel_index(cell, layout)]
        end
        copyto!(destination, stage)
    else
        project! = _project_panel_field!(destination_backend)
        project!(destination, source, layout;
                 ndrange = length(destination))
    end
    return destination
end

function _owned_forcing_field(
        source::AbstractArray, architecture,
        layout::CubedSphereInteriorLayout, full_shape, label)
    _validate_initial_forcing_source(architecture, source, label)
    interior_cells = Int(layout.x_cells) * Int(layout.y_cells) * full_shape[3]
    if length(source) == interior_cells
        destination = _allocate_storage(
            architecture, eltype(source), (interior_cells,))
        return _copy_initial_forcing!(
            destination, source, architecture, label)
    end
    length(source) == prod(full_shape) || throw(DimensionMismatch(
        "$label has shape $(size(source)); expected halo-padded $full_shape " *
        "or $interior_cells packed interior values"))
    destination = _allocate_storage(
        architecture, eltype(source), (interior_cells,))
    return _copy_projected_forcing!(
        destination, source, architecture, layout, label)
end

function _required_field_indices(available, required, label)
    return ntuple(length(required)) do field
        index = findfirst(==(required[field]), available)
        index === nothing && throw(KeyError(
            "$label is missing required field $(required[field])"))
        Int32(index)
    end
end

function _owned_required_forcing_fields(
        values::NamedTuple, required, ::Type{FT}, architecture,
        layout, full_shape, label) where FT
    isempty(required) && return NamedTuple()
    entries = ntuple(length(required)) do field
        name = required[field]
        hasproperty(values, name) || throw(KeyError(
            "$label is missing required field $name"))
        value = getproperty(values, name)
        value isa FT || throw(ArgumentError(
            "$label.$name uses $(typeof(value)); expected $FT"))
        value
    end
    destination = _allocate_storage(
        architecture, FT, (length(required),))
    if architecture isa AtmosChemistry.CPU
        @inbounds for field in eachindex(entries)
            destination[field] = entries[field]
        end
    else
        _copy_initial_forcing!(
            destination, collect(entries), architecture, label)
    end
    return AtmosChemistry.PackedForcingFields(
        required, destination, AtmosChemistry.ConstantForcingFields())
end


function _copy_selected_constant_fields!(destination, source, indices,
                                         architecture, label)
    _validate_initial_forcing_source(architecture, source, label)
    source_backend = KernelAbstractions.get_backend(source)
    destination_backend = KernelAbstractions.get_backend(destination)
    if source_backend isa KernelAbstractions.CPU &&
       destination_backend isa KernelAbstractions.CPU
        @inbounds for field in eachindex(indices)
            destination[field] = source[indices[field]]
        end
    elseif source_backend isa KernelAbstractions.CPU
        selected = [source[index] for index in indices]
        copyto!(destination, selected)
    else
        gather! = _gather_constant_forcing_fields!(destination_backend)
        gather!(destination, source, indices;
                ndrange = length(indices))
    end
    return destination
end

function _owned_required_forcing_fields(
        fields::AtmosChemistry.PackedForcingFields{
            Available, AtmosChemistry.ConstantForcingFields},
        required, ::Type{FT}, architecture, layout, full_shape,
        label) where {Available, FT}
    isempty(required) && return NamedTuple()
    indices = _required_field_indices(Available, required, label)
    _validate_initial_forcing_source(
        architecture, getfield(fields, :values), label)
    destination = _allocate_storage(
        architecture, FT, (length(required),))
    _copy_selected_constant_fields!(
        destination, getfield(fields, :values), indices,
        architecture, label)
    return AtmosChemistry.PackedForcingFields(
        required, destination, AtmosChemistry.ConstantForcingFields())
end

function _copy_selected_cell_fields!(destination, source, indices,
                                     architecture, ::LinearCellLayout,
                                     full_shape, label)
    _validate_initial_forcing_source(architecture, source, label)
    source_backend = KernelAbstractions.get_backend(source)
    destination_backend = KernelAbstractions.get_backend(destination)
    if source_backend isa KernelAbstractions.CPU &&
       destination_backend isa KernelAbstractions.CPU
        @inbounds for field in axes(destination, 2),
                      cell in axes(destination, 1)
            destination[cell, field] = source[cell, indices[field]]
        end
    elseif source_backend isa KernelAbstractions.CPU
        selected = source[:, collect(indices)]
        copyto!(destination, selected)
    else
        gather! = _gather_cell_forcing_fields!(destination_backend)
        gather!(destination, source, indices; ndrange = size(destination))
    end
    return destination
end

function _copy_selected_cell_fields!(
        destination, source, indices, architecture,
        layout::CubedSphereInteriorLayout, full_shape, label)
    _validate_initial_forcing_source(architecture, source, label)
    interior_cells = size(destination, 1)
    source_rows = size(source, 1)
    source_rows in (interior_cells, prod(full_shape)) ||
        throw(DimensionMismatch(
            "$label has $source_rows rows; expected $interior_cells packed " *
            "interior or $(prod(full_shape)) halo-padded rows"))
    source_backend = KernelAbstractions.get_backend(source)
    destination_backend = KernelAbstractions.get_backend(destination)
    if source_backend isa KernelAbstractions.CPU &&
       destination_backend isa KernelAbstractions.CPU
        @inbounds for field in axes(destination, 2),
                      cell in axes(destination, 1)
            source_cell = source_rows == interior_cells ? cell :
                          _full_panel_index(cell, layout)
            destination[cell, field] = source[source_cell, indices[field]]
        end
    elseif source_backend isa KernelAbstractions.CPU
        selected = Matrix{eltype(source)}(
            undef, size(destination))
        @inbounds for field in axes(selected, 2), cell in axes(selected, 1)
            source_cell = source_rows == interior_cells ? cell :
                          _full_panel_index(cell, layout)
            selected[cell, field] = source[source_cell, indices[field]]
        end
        copyto!(destination, selected)
    elseif source_rows == interior_cells
        gather! = _gather_cell_forcing_fields!(destination_backend)
        gather!(destination, source, indices; ndrange = size(destination))
    else
        gather! = _project_gather_panel_forcing_fields!(destination_backend)
        gather!(destination, source, indices, layout;
                ndrange = size(destination))
    end
    return destination
end

function _owned_required_forcing_fields(
        fields::AtmosChemistry.PackedForcingFields{
            Available, AtmosChemistry.CellForcingFields},
        required, ::Type{FT}, architecture, layout, full_shape,
        label) where {Available, FT}
    isempty(required) && return NamedTuple()
    indices = _required_field_indices(Available, required, label)
    _validate_initial_forcing_source(
        architecture, getfield(fields, :values), label)
    rows = _projected_cell_count(
        getfield(fields, :values), layout, full_shape)
    destination = _allocate_storage(
        architecture, FT, (rows, length(required)))
    _copy_selected_cell_fields!(
        destination, getfield(fields, :values), indices,
        architecture, layout, full_shape, label)
    return AtmosChemistry.PackedForcingFields(
        required, destination, AtmosChemistry.CellForcingFields())
end

function _owned_auxiliary_fields(values::NamedTuple, ::Type{FT}, architecture,
                                 layout, full_shape, label) where FT
    names = propertynames(values)
    owned = map(names) do name
        _owned_forcing_field(
            getproperty(values, name), architecture, layout, full_shape,
            "$label.$name")
    end
    return NamedTuple{names}(owned)
end

function _owned_auxiliary_fields(
        fields::AtmosChemistry.PackedForcingFields{Names}, ::Type{FT},
        architecture, layout, full_shape, label) where {Names, FT}
    return _owned_required_forcing_fields(
        fields, Names, FT, architecture, layout, full_shape, label)
end

function _owned_transport_forcing(
        forcing::AtmosChemistry.ChemistryForcing, mechanism, ::Type{FT},
        architecture, layout, full_shape) where FT
    requirements = AtmosChemistry.required_forcings(mechanism)
    return AtmosChemistry.ChemistryForcing(
        temperature = _owned_forcing_field(
            forcing.temperature, architecture, layout, full_shape,
            "forcing.temperature"),
        pressure = _owned_forcing_field(
            forcing.pressure, architecture, layout, full_shape,
            "forcing.pressure"),
        air_number_density = _owned_forcing_field(
            forcing.air_number_density, architecture, layout, full_shape,
            "forcing.air_number_density"),
        fixed_species = _owned_required_forcing_fields(
            forcing.fixed_species, requirements.fixed_species, FT,
            architecture, layout, full_shape, "forcing.fixed_species"),
        photolysis = _owned_required_forcing_fields(
            forcing.photolysis, requirements.photolysis, FT,
            architecture, layout, full_shape, "forcing.photolysis"),
        auxiliaries = _owned_auxiliary_fields(
            forcing.auxiliaries, FT, architecture, layout, full_shape,
            "forcing.auxiliaries"),
        envelope = forcing.envelope)
end

function _owned_transport_forcing(forcing::Tuple, mechanism, ::Type{FT},
                                  architecture, layout, full_shape) where FT
    return map(value -> _owned_transport_forcing(
                   value, mechanism, FT, architecture, layout, full_shape),
               forcing)
end

_synchronize_owned_forcing(::AtmosChemistry.CPU) = nothing
_synchronize_owned_forcing(architecture::AtmosChemistry.NativeGPU) =
    KernelAbstractions.synchronize(architecture.backend)

# Workspace allocation and accounting

function _allocate_transport_workspace(operator, state, cell_count,
                                       diagnostic_count, forcing, layout,
                                       storage_arrays;
                                       full_shape = nothing)
    architecture = AtmosChemistry.architecture(operator.model)
    for (label, array) in storage_arrays
        _validate_backend(architecture, array, label)
    end
    indices_host = _tracer_indices(operator, state)
    FT = eltype(operator.model)
    species_count = length(indices_host)
    policy = _workspace_policy(operator)
    plan = _transport_tile_plan(
        operator.model, cell_count, diagnostic_count, forcing, policy,
        _needs_host_forcing_stage(operator.forcing_provider), layout,
        full_shape)
    tile_cells = plan.tile_cells
    mechanism = AtmosChemistry.compiled_mechanism(operator.model)
    packed_forcing = _owned_transport_forcing(
        forcing, mechanism, FT, architecture, layout, full_shape)
    _synchronize_owned_forcing(architecture)
    _validate_transport_forcing(operator.model, packed_forcing, cell_count)
    concentrations = _allocate_storage(
        architecture, FT, (tile_cells, species_count))
    tracer_indices = _allocate_storage(
        architecture, Int32, (species_count,))
    molar_masses = _allocate_storage(
        architecture, FT, (species_count,))
    conversion_error = _allocate_storage(architecture, Int32, (1,))
    validation_chunks = _allocate_storage(
        architecture, Int32,
        (plan.panel_count *
         cld(cell_count, _VALIDATION_CHUNK_CELLS),))
    conversion_error_host = zeros(Int32, 1)
    host_forcing_stage = _allocate_host_forcing_stage(
        architecture, FT, cell_count,
        Val(_needs_host_forcing_stage(operator.forcing_provider)))
    batch_cells = _allocate_storage(architecture, Int32, (tile_cells,))
    warm_steps = _allocate_storage(architecture, FT, (diagnostic_count,))
    fill!(warm_steps, zero(FT))
    copyto!(tracer_indices, collect(indices_host))
    copyto!(molar_masses,
            collect(AtmosChemistry.species_molar_masses(operator.model)))
    chemistry = AtmosChemistry.allocate_workspace(
        operator.model, tile_cells;
        policy = _tile_workspace_policy(policy, tile_cells))
    diagnostics = AtmosChemistry.allocate_diagnostics(FT, diagnostic_count)
    tile_diagnostics = _allocate_tile_diagnostics(
        architecture, FT, tile_cells)
    backend_status, backend_counters = _allocate_backend_diagnostics(
        architecture, diagnostic_count)
    return TransportChemistryWorkspace(
        concentrations, chemistry, diagnostics, tile_diagnostics,
        backend_status, backend_counters,
        packed_forcing, tracer_indices, molar_masses, conversion_error,
        validation_chunks, conversion_error_host, host_forcing_stage,
        batch_cells, warm_steps, plan, layout,
        cell_count)
end

ATC.chemistry_workspace_plan(workspace::TransportChemistryWorkspace) =
    workspace.plan

_storage_bytes(array::AbstractArray) = length(array) * sizeof(eltype(array))
_storage_bytes(::Nothing) = 0

function _diagnostic_storage_bytes(diagnostics)
    return _storage_bytes(diagnostics.status) +
           _storage_bytes(diagnostics.counter_storage) +
           _storage_bytes(diagnostics.last_step)
end

_diagnostic_storage_bytes(::Nothing) = 0

function ATC.chemistry_workspace_storage_bytes(
        workspace::TransportChemistryWorkspace)
    chemistry = AtmosChemistry.workspace_storage_bytes(workspace.chemistry)
    adapter = _storage_bytes(workspace.concentrations) +
              _storage_bytes(workspace.tracer_indices) +
              _storage_bytes(workspace.molar_masses) +
              _storage_bytes(workspace.conversion_error) +
              _storage_bytes(workspace.validation_chunks) +
              _storage_bytes(workspace.conversion_error_host) +
              _storage_bytes(workspace.host_forcing_stage) +
              _storage_bytes(workspace.batch_cells) +
              _storage_bytes(workspace.warm_steps) +
              _forcing_storage_bytes(workspace.forcing) +
              _diagnostic_storage_bytes(workspace.diagnostics) +
              _diagnostic_storage_bytes(workspace.tile_diagnostics) +
              _storage_bytes(workspace.backend_status) +
              _storage_bytes(workspace.backend_counters)
    return (; chemistry, adapter, total = chemistry + adapter)
end

# Transport topology adapters

function ATM._chemistry_workspace_for(operator::AtmosTransport.AtmosChemistryOperator,
                                      state::AtmosTransport.CellState{
                                          AtmosTransport.DryBasis}, grid)
    storage = (("transport air mass", state.air_mass),
               ("transport tracer storage", state.tracers_raw))
    forcing = ATC.initial_chemistry_forcing(operator.forcing_provider)
    return _allocate_transport_workspace(
        operator, state, length(state.air_mass), length(state.air_mass),
        forcing,
        LinearCellLayout(), storage)
end

function ATC.chemistry_workspace_plan(
        operator::AtmosTransport.AtmosChemistryOperator,
        state::AtmosTransport.CellState{AtmosTransport.DryBasis}, grid)
    cells = length(state.air_mass)
    forcing = ATC.initial_chemistry_forcing(operator.forcing_provider)
    return _transport_tile_plan(
        operator.model, cells, cells, forcing, _workspace_policy(operator),
        _needs_host_forcing_stage(operator.forcing_provider),
        LinearCellLayout())
end

@inline function _full_panel_index(
        cell, layout::CubedSphereInteriorLayout{H}) where H
    x, y, z = _panel_indices(cell, layout)
    full_x = layout.x_cells + Int32(2H)
    full_y = layout.y_cells + Int32(2H)
    return x + (y - 1) * full_x + (z - 1) * full_x * full_y
end

@kernel function _project_panel_field!(interior, panel_field, layout)
    cell = @index(Global, Linear)
    @inbounds interior[cell] = panel_field[_full_panel_index(cell, layout)]
end

@kernel function _gather_constant_forcing_fields!(destination, source,
                                                  indices)
    field = @index(Global, Linear)
    @inbounds destination[field] = source[indices[field]]
end

@kernel function _gather_cell_forcing_fields!(destination, source, indices)
    cell, field = @index(Global, NTuple)
    @inbounds destination[cell, field] = source[cell, indices[field]]
end

@kernel function _project_gather_panel_forcing_fields!(
        destination, source, indices, layout)
    cell, field = @index(Global, NTuple)
    @inbounds destination[cell, field] =
        source[_full_panel_index(cell, layout), indices[field]]
end

function ATM._chemistry_workspace_for(operator::AtmosTransport.AtmosChemistryOperator,
                                      state::AtmosTransport.CubedSphereState{
                                          AtmosTransport.DryBasis}, grid)
    Hp = state.halo_width
    panel_size = size(state.air_mass[1])
    x_cells = panel_size[1] - 2Hp
    y_cells = panel_size[2] - 2Hp
    x_cells > 0 && y_cells > 0 || throw(ArgumentError(
        "cubed-sphere halo width leaves no chemistry interior cells"))
    panel_cells = x_cells * y_cells * panel_size[3]
    layout = CubedSphereInteriorLayout{Hp}(Int32(x_cells), Int32(y_cells))
    source_forcing = ATC.initial_chemistry_forcing(operator.forcing_provider)
    forcing = source_forcing isa Tuple ? source_forcing :
              ntuple(_ -> source_forcing, 6)
    storage = ntuple(12) do index
        panel = div(index + 1, 2)
        isodd(index) ?
            ("panel $panel air mass", state.air_mass[panel]) :
            ("panel $panel tracer storage", state.tracers_raw[panel])
    end
    return _allocate_transport_workspace(
        operator, state, panel_cells, 6 * panel_cells,
        forcing, layout, storage; full_shape = panel_size)
end

function ATC.chemistry_workspace_plan(
        operator::AtmosTransport.AtmosChemistryOperator,
        state::AtmosTransport.CubedSphereState{AtmosTransport.DryBasis}, grid)
    Hp = state.halo_width
    panel_size = size(state.air_mass[1])
    x_cells = panel_size[1] - 2Hp
    y_cells = panel_size[2] - 2Hp
    x_cells > 0 && y_cells > 0 || throw(ArgumentError(
        "cubed-sphere halo width leaves no chemistry interior cells"))
    panel_cells = x_cells * y_cells * panel_size[3]
    layout = CubedSphereInteriorLayout{Hp}(
        Int32(x_cells), Int32(y_cells))
    forcing = ATC.initial_chemistry_forcing(operator.forcing_provider)
    forcing_tuple = forcing isa Tuple ? forcing : ntuple(_ -> forcing, 6)
    return _transport_tile_plan(
        operator.model, panel_cells, 6 * panel_cells, forcing_tuple,
        _workspace_policy(operator),
        _needs_host_forcing_stage(operator.forcing_provider), layout,
        panel_size)
end

function ATM._chemistry_workspace_for(operator::AtmosTransport.AtmosChemistryOperator,
                                      state::Union{
                                          AtmosTransport.CellState,
                                          AtmosTransport.CubedSphereState}, grid)
    throw(ArgumentError(
        "AtmosChemistry coupling requires DryBasis transport state; " *
        "MoistBasis conversion needs water-vapor mass before chemistry"))
end

@inline _tile_cell(row, tile_start, tile_length) =
    tile_start + min(row, tile_length) - 1

@inline function _panel_indices(cell, layout::CubedSphereInteriorLayout{H}) where H
    offset = cell - 1
    x = rem(offset, layout.x_cells) + Int32(H + 1)
    yz = div(offset, layout.x_cells)
    y = rem(yz, layout.y_cells) + Int32(H + 1)
    z = div(yz, layout.y_cells) + 1
    return x, y, z
end

# Pre-mutation conversion validation

@inline function _conversion_code(dry_mass, density)
    (!isfinite(dry_mass) || dry_mass <= zero(dry_mass)) && return Int32(2)
    (!isfinite(density) || density <= zero(density)) && return Int32(3)
    return Int32(0)
end

@inline _encoded_conversion_error(cell, code) = Int32(4) * (cell - 1) + code

@kernel function _validate_linear_chunks!(
    chunk_errors, air_mass, forcing, cell_count)
    chunk_index = @index(Global, Linear)
    chunk = Int32(chunk_index)
    first_cell = (chunk - 1) * Int32(_VALIDATION_CHUNK_CELLS) + 1
    last_cell = min(cell_count,
                    first_cell + Int32(_VALIDATION_CHUNK_CELLS - 1))
    first = typemax(Int32)
    @inbounds for cell in first_cell:last_cell
        dry_mass = air_mass[cell]
        density = AtmosChemistry.number_density(forcing, cell)
        code = _conversion_code(dry_mass, density)
        code == 0 || (first = min(first,
            _encoded_conversion_error(cell, code)))
    end
    @inbounds chunk_errors[chunk] = first
end

@kernel function _validate_panel_chunks!(
    chunk_errors, air_mass, forcing, cell_count, layout,
    chunk_offset, cell_offset)
    chunk_index = @index(Global, Linear)
    chunk = Int32(chunk_index)
    first_cell = (chunk - 1) * Int32(_VALIDATION_CHUNK_CELLS) + 1
    last_cell = min(cell_count,
                    first_cell + Int32(_VALIDATION_CHUNK_CELLS - 1))
    first = typemax(Int32)
    @inbounds for cell in first_cell:last_cell
        x, y, z = _panel_indices(cell, layout)
        dry_mass = air_mass[x, y, z]
        density = AtmosChemistry.number_density(forcing, cell)
        code = _conversion_code(dry_mass, density)
        code == 0 || (first = min(first,
            _encoded_conversion_error(cell + cell_offset, code)))
    end
    @inbounds chunk_errors[chunk_offset + chunk] = first
end

@kernel function _reduce_validation_chunks!(
        first_error, chunk_errors, chunk_count)
    first = typemax(Int32)
    @inbounds for chunk in Int32(1):chunk_count
        first = min(first, chunk_errors[chunk])
    end
    @inbounds first_error[1] = first
end

function _finish_conversion_validation!(workspace, backend, label;
                                        cells_per_panel = 0)
    KernelAbstractions.synchronize(backend)
    copyto!(workspace.conversion_error_host, workspace.conversion_error)
    encoded = only(workspace.conversion_error_host)
    encoded == typemax(Int32) && return nothing
    code = rem(encoded, Int32(4))
    cell = div(encoded, Int32(4)) + 1
    field = code == Int32(2) ? "dry-air mass" : "air number density"
    location = if iszero(cells_per_panel)
        "$label cell $cell"
    else
        panel = div(cell - 1, cells_per_panel) + 1
        panel_cell = rem(cell - 1, cells_per_panel) + 1
        "$label panel $panel cell $panel_cell"
    end
    throw(ArgumentError(
        "chemistry requires finite positive $field in $location"))
end

function _validate_storage!(workspace, air_mass, forcing,
                            ::LinearCellLayout, label)
    backend = KernelAbstractions.get_backend(workspace.concentrations)
    chunk_count = cld(workspace.cell_count, _VALIDATION_CHUNK_CELLS)
    validate! = _validate_linear_chunks!(backend)
    validate!(workspace.validation_chunks, air_mass, forcing,
              Int32(workspace.cell_count); ndrange = chunk_count)
    reduce! = _reduce_validation_chunks!(backend)
    reduce!(workspace.conversion_error, workspace.validation_chunks,
            Int32(chunk_count); ndrange = 1)
    return _finish_conversion_validation!(workspace, backend, label)
end

function _validate_storage!(workspace, air_mass, forcing,
                            layout::CubedSphereInteriorLayout, label)
    backend = KernelAbstractions.get_backend(workspace.concentrations)
    chunk_count = cld(workspace.cell_count, _VALIDATION_CHUNK_CELLS)
    validate! = _validate_panel_chunks!(backend)
    validate!(workspace.validation_chunks, air_mass, forcing,
              Int32(workspace.cell_count), layout, Int32(0), Int32(0);
              ndrange = chunk_count)
    reduce! = _reduce_validation_chunks!(backend)
    reduce!(workspace.conversion_error, workspace.validation_chunks,
            Int32(chunk_count); ndrange = 1)
    return _finish_conversion_validation!(workspace, backend, label)
end

function _validate_cubed_sphere_storage!(workspace, air_mass, forcing,
                                         layout)
    backend = KernelAbstractions.get_backend(workspace.concentrations)
    chunks_per_panel = cld(
        workspace.cell_count, _VALIDATION_CHUNK_CELLS)
    validate! = _validate_panel_chunks!(backend)
    for panel in 1:6
        chunk_offset = (panel - 1) * chunks_per_panel
        cell_offset = (panel - 1) * workspace.cell_count
        validate!(workspace.validation_chunks, air_mass[panel], forcing[panel],
                  Int32(workspace.cell_count), layout,
                  Int32(chunk_offset), Int32(cell_offset);
                  ndrange = chunks_per_panel)
    end
    reduce! = _reduce_validation_chunks!(backend)
    reduce!(workspace.conversion_error, workspace.validation_chunks,
            Int32(length(workspace.validation_chunks)); ndrange = 1)
    return _finish_conversion_validation!(
        workspace, backend, "cubed-sphere";
        cells_per_panel = workspace.cell_count)
end

@kernel function _fill_batch_cells!(cells, tile_start, tile_length)
    row = @index(Global, Linear)
    @inbounds cells[row] = _tile_cell(row, tile_start, tile_length)
end

@kernel function _load_warm_steps!(last_step, warm_steps, tile_start,
                                   tile_length, diagnostic_offset)
    row = @index(Global, Linear)
    cell = diagnostic_offset + _tile_cell(row, tile_start, tile_length)
    @inbounds last_step[row] = warm_steps[cell]
end

@kernel function _store_warm_steps!(warm_steps, last_step, tile_start,
                                    diagnostic_offset)
    row = @index(Global, Linear)
    cell = diagnostic_offset + tile_start + row - 1
    @inbounds warm_steps[cell] = last_step[row]
end

# Tile conversion and execution

@kernel function _load_linear_concentrations!(
        concentrations, tracer_storage, air_mass, forcing,
        tracer_indices, molar_masses, dry_air_molar_mass,
        tile_start, tile_length)
    row, species = @index(Global, NTuple)
    cell = _tile_cell(row, tile_start, tile_length)
    dry_mass = @inbounds air_mass[cell]
    density = AtmosChemistry.number_density(forcing, cell)
    tracer = @inbounds tracer_indices[species]
    molecular_mass = @inbounds molar_masses[species]
    transported_mass = @inbounds tracer_storage[cell, tracer]
    @inbounds concentrations[row, species] =
        (transported_mass / dry_mass) *
        (dry_air_molar_mass / molecular_mass) * density
end

@kernel function _store_linear_concentrations!(
        tracer_storage, concentrations, status, air_mass, forcing,
        tracer_indices, molar_masses, dry_air_molar_mass, tile_start)
    row, species = @index(Global, NTuple)
    if @inbounds status[row] == Int8(1)
        cell = tile_start + row - 1
        density = AtmosChemistry.number_density(forcing, cell)
        tracer = @inbounds tracer_indices[species]
        molecular_mass = @inbounds molar_masses[species]
        mole_fraction = @inbounds concentrations[row, species] / density
        @inbounds tracer_storage[cell, tracer] =
            mole_fraction * (molecular_mass / dry_air_molar_mass) * air_mass[cell]
    end
end

@kernel function _load_panel_concentrations!(
        concentrations, tracer_storage, air_mass, forcing,
        tracer_indices, molar_masses, dry_air_molar_mass, layout,
        tile_start, tile_length)
    row, species = @index(Global, NTuple)
    cell = _tile_cell(row, tile_start, tile_length)
    x, y, z = _panel_indices(cell, layout)
    dry_mass = @inbounds air_mass[x, y, z]
    density = AtmosChemistry.number_density(forcing, cell)
    tracer = @inbounds tracer_indices[species]
    molecular_mass = @inbounds molar_masses[species]
    transported_mass = @inbounds tracer_storage[x, y, z, tracer]
    @inbounds concentrations[row, species] =
        (transported_mass / dry_mass) *
        (dry_air_molar_mass / molecular_mass) * density
end

@kernel function _store_panel_concentrations!(
        tracer_storage, concentrations, status, air_mass, forcing,
        tracer_indices, molar_masses, dry_air_molar_mass, layout, tile_start)
    row, species = @index(Global, NTuple)
    if @inbounds status[row] == Int8(1)
        cell = tile_start + row - 1
        x, y, z = _panel_indices(cell, layout)
        density = AtmosChemistry.number_density(forcing, cell)
        tracer = @inbounds tracer_indices[species]
        molecular_mass = @inbounds molar_masses[species]
        mole_fraction = @inbounds concentrations[row, species] / density
        @inbounds tracer_storage[x, y, z, tracer] =
            mole_fraction * (molecular_mass / dry_air_molar_mass) * air_mass[x, y, z]
    end
end

function _load_tile!(workspace, tracer_storage, air_mass, forcing,
                     dry_air_molar_mass, tile_start, tile_length,
                     ::LinearCellLayout)
    backend = KernelAbstractions.get_backend(workspace.concentrations)
    kernel! = _load_linear_concentrations!(backend)
    kernel!(workspace.concentrations, tracer_storage, air_mass,
        forcing, workspace.tracer_indices,
        workspace.molar_masses, dry_air_molar_mass,
        Int32(tile_start), Int32(tile_length);
        ndrange = (tile_length, size(workspace.concentrations, 2)))
    return nothing
end

function _load_tile!(workspace, tracer_storage, air_mass, forcing,
                     dry_air_molar_mass, tile_start, tile_length,
                     layout::CubedSphereInteriorLayout)
    backend = KernelAbstractions.get_backend(workspace.concentrations)
    kernel! = _load_panel_concentrations!(backend)
    kernel!(workspace.concentrations, tracer_storage, air_mass,
        forcing, workspace.tracer_indices,
        workspace.molar_masses, dry_air_molar_mass, layout,
        Int32(tile_start), Int32(tile_length);
        ndrange = (tile_length, size(workspace.concentrations, 2)))
    return nothing
end

function _store_tile!(tracer_storage, workspace, air_mass, forcing,
                      dry_air_molar_mass, tile_start, tile_length,
                      ::LinearCellLayout)
    backend = KernelAbstractions.get_backend(workspace.concentrations)
    kernel! = _store_linear_concentrations!(backend)
    kernel!(tracer_storage, workspace.concentrations,
        workspace.chemistry.status, air_mass, forcing,
        workspace.tracer_indices, workspace.molar_masses,
        dry_air_molar_mass, Int32(tile_start);
        ndrange = (tile_length, size(workspace.concentrations, 2)))
    return nothing
end

function _store_tile!(tracer_storage, workspace, air_mass, forcing,
                      dry_air_molar_mass, tile_start, tile_length,
                      layout::CubedSphereInteriorLayout)
    backend = KernelAbstractions.get_backend(workspace.concentrations)
    kernel! = _store_panel_concentrations!(backend)
    kernel!(tracer_storage, workspace.concentrations,
        workspace.chemistry.status, air_mass, forcing,
        workspace.tracer_indices, workspace.molar_masses,
        dry_air_molar_mass, layout, Int32(tile_start);
        ndrange = (tile_length, size(workspace.concentrations, 2)))
    return nothing
end

function _finish_transport_diagnostics!(workspace, ::AtmosChemistry.CPU)
    backend = KernelAbstractions.get_backend(workspace.concentrations)
    KernelAbstractions.synchronize(backend)
    return workspace.diagnostics
end

function _finish_transport_diagnostics!(
        workspace, ::AtmosChemistry.NativeGPU)
    backend = KernelAbstractions.get_backend(workspace.concentrations)
    KernelAbstractions.synchronize(backend)
    copyto!(workspace.diagnostics.status, workspace.backend_status)
    copyto!(workspace.diagnostics.counter_storage,
            workspace.backend_counters)
    copyto!(workspace.diagnostics.last_step, workspace.warm_steps)
    return workspace.diagnostics
end

function _record_tile_diagnostics!(workspace, tile_start, tile_length,
                                   diagnostic_offset,
                                   ::AtmosChemistry.CPU)
    source = workspace.tile_diagnostics
    destination = workspace.diagnostics
    @inbounds for row in 1:tile_length
        cell = diagnostic_offset + tile_start + row - 1
        destination.status[cell] = source.status[row]
        destination.function_calls[cell] = source.function_calls[row]
        destination.jacobian_calls[cell] = source.jacobian_calls[row]
        destination.accepted_steps[cell] = source.accepted_steps[row]
        destination.rejected_steps[cell] = source.rejected_steps[row]
        destination.linear_solves[cell] = source.linear_solves[row]
        destination.last_step[cell] = source.last_step[row]
    end
    return nothing
end

@kernel function _record_backend_diagnostics!(
        destination_status, destination_counters, source_status,
        source_counters, tile_start, diagnostic_offset)
    row = @index(Global, Linear)
    cell = diagnostic_offset + tile_start + row - 1
    @inbounds begin
        destination_status[cell] = source_status[row]
        for counter in 1:5
            destination_counters[cell, counter] =
                source_counters[row, counter]
        end
    end
end

function _record_tile_diagnostics!(workspace, tile_start, tile_length,
                                   diagnostic_offset,
                                   ::AtmosChemistry.NativeGPU)
    backend = KernelAbstractions.get_backend(workspace.concentrations)
    record! = _record_backend_diagnostics!(backend)
    record!(workspace.backend_status, workspace.backend_counters,
            workspace.chemistry.status, workspace.chemistry.counters,
            Int32(tile_start), Int32(diagnostic_offset);
            ndrange = tile_length)
    return nothing
end


function _advance_transport_tile!(
        ::AtmosChemistry.CPU, batch, model, forcing, duration, workspace)
    return AtmosChemistry.advance!(
        batch, model, forcing, duration;
        workspace = workspace.chemistry,
        diagnostics = workspace.tile_diagnostics)
end

function _advance_transport_tile!(
        ::AtmosChemistry.NativeGPU, batch, model, forcing, duration,
        workspace)
    return AtmosChemistry.advance!(
        batch, model, forcing, duration;
        workspace = workspace.chemistry, diagnostics = nothing,
        diagnostic_transfer = AtmosChemistry.BackendDiagnosticTransfer())
end

function _prepare_tile!(workspace, backend, tile_start, tile_length,
                        diagnostic_offset)
    fill_cells! = _fill_batch_cells!(backend)
    fill_cells!(workspace.batch_cells, Int32(tile_start), Int32(tile_length);
                ndrange = tile_length)
    load_steps! = _load_warm_steps!(backend)
    load_steps!(workspace.chemistry.last_step, workspace.warm_steps,
                Int32(tile_start), Int32(tile_length),
                Int32(diagnostic_offset);
                ndrange = tile_length)
    return nothing
end


function _save_warm_steps!(workspace, backend, tile_start, tile_length,
                           diagnostic_offset)
    store_steps! = _store_warm_steps!(backend)
    store_steps!(workspace.warm_steps, workspace.chemistry.last_step,
                 Int32(tile_start), Int32(diagnostic_offset);
                 ndrange = tile_length)
    return nothing
end

function _apply_to_storage!(tracer_storage, air_mass, operator, forcing,
                            duration, workspace, diagnostic_offset)
    backend = KernelAbstractions.get_backend(workspace.concentrations)
    architecture = AtmosChemistry.architecture(operator.model)
    tile_cells = size(workspace.concentrations, 1)
    for tile_start in 1:tile_cells:workspace.cell_count
        tile_length = min(tile_cells, workspace.cell_count - tile_start + 1)
        _prepare_tile!(workspace, backend, tile_start, tile_length,
                       diagnostic_offset)
        _load_tile!(workspace, tracer_storage, air_mass, forcing,
                    operator.dry_air_molar_mass, tile_start, tile_length,
                    workspace.layout)
        try
            _advance_transport_tile!(
                architecture,
                AtmosChemistry.ChemistryBatch(
                    workspace.concentrations, workspace.batch_cells,
                    AtmosChemistry.PrevalidatedForcingCells();
                    active_cells = tile_length),
                operator.model, forcing, duration, workspace)
        catch error
            error isa AtmosChemistry.ChemistrySolveError || rethrow()
        end
        _record_tile_diagnostics!(
            workspace, tile_start, tile_length, diagnostic_offset,
            architecture)
        _save_warm_steps!(workspace, backend, tile_start, tile_length,
                          diagnostic_offset)
        _store_tile!(tracer_storage, workspace, air_mass, forcing,
                     operator.dry_air_molar_mass, tile_start, tile_length,
                     workspace.layout)
    end
    return nothing
end

_failed_cells(diagnostics) = findall(==(Int8(-1)), diagnostics.status)

_forcing_for_call(workspace, ::ATC.ConstantChemistryForcing,
                  operator, state, meteo, grid) = workspace.forcing

# Dynamic forcing refresh

function _forcing_schema_error(label, initial, updated)
    throw(ArgumentError(
        "$label changed schema from $(typeof(initial)) to $(typeof(updated)); " *
        "call-updated forcing must preserve names, precision, field layout, " *
        "and envelope type"))
end

_validate_update_schema(::AtmosChemistry.NoField,
                        ::AtmosChemistry.NoField, label) = nothing

function _validate_update_schema(initial::Number, updated::Number, label)
    typeof(updated) === typeof(initial) ||
        _forcing_schema_error(label, initial, updated)
    return nothing
end

function _validate_update_schema(
        initial::AbstractArray, updated::AbstractArray, label)
    eltype(updated) === eltype(initial) ||
        _forcing_schema_error(label, initial, updated)
    size(updated) == size(initial) || throw(DimensionMismatch(
        "$label changed shape from $(size(initial)) to $(size(updated))"))
    return nothing
end

function _validate_update_schema(
        initial::NamedTuple{Names}, updated::NamedTuple, label) where Names
    propertynames(updated) == Names || throw(ArgumentError(
        "$label changed names from $Names to $(propertynames(updated))"))
    for name in Names
        _validate_update_schema(
            getproperty(initial, name), getproperty(updated, name),
            "$label.$name")
    end
    return nothing
end

function _validate_update_schema(
        initial::AtmosChemistry.PackedForcingFields{Names, Layout},
        updated::AtmosChemistry.PackedForcingFields,
        label) where {Names, Layout}
    typeof(updated.layout) === Layout ||
        _forcing_schema_error(label, initial, updated)
    propertynames(updated) == Names || throw(ArgumentError(
        "$label changed names from $Names to $(propertynames(updated))"))
    return _validate_update_schema(
        initial.values, updated.values, "$label.values")
end

function _validate_update_schema(
        initial::AtmosChemistry.ChemistryForcing,
        updated::AtmosChemistry.ChemistryForcing, label)
    typeof(updated.envelope) === typeof(initial.envelope) ||
        _forcing_schema_error("$label.envelope", initial.envelope,
                              updated.envelope)
    for name in (:temperature, :pressure, :air_number_density,
                 :fixed_species, :photolysis, :auxiliaries)
        _validate_update_schema(
            getproperty(initial, name), getproperty(updated, name),
            "$label.$name")
    end
    return nothing
end

function _validate_update_schema(initial::Tuple, updated::Tuple, label)
    length(updated) == length(initial) || throw(DimensionMismatch(
        "$label changed tuple length from $(length(initial)) to " *
        "$(length(updated))"))
    for index in eachindex(initial)
        _validate_update_schema(
            initial[index], updated[index], "$label[$index]")
    end
    return nothing
end

_validate_update_schema(initial, updated, label) =
    _forcing_schema_error(label, initial, updated)

function _forcing_layout_error(label, destination, source)
    throw(ArgumentError(
        "$label changed forcing storage from $(typeof(destination)) to " *
        "$(typeof(source)); call-updated forcing must preserve the schema " *
        "and scalar/spatial layout used to allocate its workspace"))
end

_refresh_forcing_field!(destination::AtmosChemistry.NoField,
                        source::AtmosChemistry.NoField, label) = destination

function _refresh_forcing_field!(destination::Number, source::Number, label)
    typeof(source) === typeof(destination) ||
        _forcing_layout_error(label, destination, source)
    return source
end

function _refresh_forcing_field!(destination::AbstractArray,
                                 source::AbstractArray, label)
    size(source) == size(destination) || throw(DimensionMismatch(
        "$label changed shape from $(size(destination)) to $(size(source))"))
    if source !== destination
        _same_refresh_backend(destination, source, label)
        copyto!(destination, source)
    end
    return destination
end

function _refresh_forcing_field!(destination::AbstractArray,
                                 source::Number, label)
    fill!(destination, source)
    return destination
end

_refresh_forcing_field!(destination, source, label) =
    _forcing_layout_error(label, destination, source)

@kernel function _refresh_constant_fields!(destination, values)
    field = @index(Global, Linear)
    @inbounds destination[field] = values[field]
end

@kernel function _refresh_cell_field!(destination, source, field)
    cell = @index(Global, Linear)
    @inbounds destination[cell, field] =
        AtmosChemistry.forcing_value(source, cell)
end

@kernel function _refresh_panel_cell_field!(destination, source, field, layout)
    cell = @index(Global, Linear)
    source_cell = source isa Number ? cell : _full_panel_index(cell, layout)
    @inbounds destination[cell, field] =
        AtmosChemistry.forcing_value(source, source_cell)
end

function _same_refresh_backend(destination, source, label)
    destination_backend = KernelAbstractions.get_backend(destination)
    source_backend = KernelAbstractions.get_backend(source)
    if typeof(destination_backend) === typeof(source_backend)
        AtmosChemistry._validate_native_gpu_device(
            destination_backend, source, label)
        return true
    end
    source_backend isa KernelAbstractions.CPU || throw(ArgumentError(
        "$label cannot be refreshed across $(typeof(source_backend)) and " *
        "$(typeof(destination_backend)); move forcing to the model backend"))
    return false
end

function _refresh_cell_column!(destination, source::Number, field, label)
    backend = KernelAbstractions.get_backend(destination)
    kernel! = _refresh_cell_field!(backend)
    kernel!(destination, source, Int32(field);
            ndrange = size(destination, 1))
    return nothing
end

function _refresh_cell_column!(
        destination, source::AbstractArray, field, label)
    cell_count = size(destination, 1)
    length(source) == cell_count || throw(DimensionMismatch(
        "$label has $(length(source)) cells; expected $cell_count"))
    if _same_refresh_backend(destination, source, label)
        backend = KernelAbstractions.get_backend(destination)
        kernel! = _refresh_cell_field!(backend)
        kernel!(destination, source, Int32(field); ndrange = cell_count)
    else
        copyto!(@view(destination[:, field]), reshape(source, :))
    end
    return nothing
end

_named_forcing_source(source::NamedTuple, name) = getproperty(source, name)
_named_forcing_source(source::AtmosChemistry.PackedForcingFields, name) =
    getproperty(source, name)

function _require_forcing_names(source, names, label)
    missing = Tuple(name for name in names if !hasproperty(source, name))
    isempty(missing) || throw(ArgumentError(
        "$label is missing required fields $missing"))
    return nothing
end

function _refresh_constant_forcing_fields!(
        destination::AtmosChemistry.PackedForcingFields{
            Names, AtmosChemistry.ConstantForcingFields},
        source::AtmosChemistry.PackedForcingFields{
            Names, AtmosChemistry.ConstantForcingFields}, label) where Names
    source.values === destination.values ||
        copyto!(destination.values, source.values)
    return destination
end

function _refresh_constant_forcing_fields!(
        destination::AtmosChemistry.PackedForcingFields{
            Names, AtmosChemistry.ConstantForcingFields},
        source, label) where Names
    _require_forcing_names(source, Names, label)
    values = ntuple(length(Names)) do index
        value = _named_forcing_source(source, Names[index])
        value isa Number || _forcing_layout_error(
            "$label.$(Names[index])", destination, value)
        value
    end
    backend = KernelAbstractions.get_backend(destination.values)
    kernel! = _refresh_constant_fields!(backend)
    kernel!(destination.values, values; ndrange = length(Names))
    return destination
end

function _refresh_cell_forcing_fields!(
        destination::AtmosChemistry.PackedForcingFields{
            Names, AtmosChemistry.CellForcingFields},
        source::AtmosChemistry.PackedForcingFields{
            Names, AtmosChemistry.CellForcingFields}, label) where Names
    size(source.values) == size(destination.values) ||
        throw(DimensionMismatch(
            "$label changed shape from $(size(destination.values)) to " *
            "$(size(source.values))"))
    source.values === destination.values ||
        copyto!(destination.values, source.values)
    return destination
end

function _refresh_cell_forcing_fields!(
        destination::AtmosChemistry.PackedForcingFields{
            Names, AtmosChemistry.CellForcingFields},
        source, label) where Names
    _require_forcing_names(source, Names, label)
    for field in eachindex(Names)
        values = _named_forcing_source(source, Names[field])
        _refresh_cell_column!(
            destination.values, values, field, "$label.$(Names[field])")
    end
    return destination
end

function _refresh_forcing_fields!(destination::NamedTuple{Names},
                                  source::NamedTuple, label) where Names
    propertynames(source) == Names || throw(ArgumentError(
        "$label changed names from $Names to $(propertynames(source))"))
    values = map(Names) do name
        _refresh_forcing_field!(getproperty(destination, name),
                                getproperty(source, name), "$label.$name")
    end
    return NamedTuple{Names}(values)
end

_refresh_forcing_fields!(destination::AtmosChemistry.PackedForcingFields{
        <:Any, AtmosChemistry.ConstantForcingFields}, source, label) =
    _refresh_constant_forcing_fields!(destination, source, label)

_refresh_forcing_fields!(destination::AtmosChemistry.PackedForcingFields{
        <:Any, AtmosChemistry.CellForcingFields}, source, label) =
    _refresh_cell_forcing_fields!(destination, source, label)

function _refresh_forcing!(destination, source)
    source === destination && return destination
    return AtmosChemistry.ChemistryForcing(
        temperature = _refresh_forcing_field!(
            destination.temperature, source.temperature,
            "forcing.temperature"),
        pressure = _refresh_forcing_field!(
            destination.pressure, source.pressure, "forcing.pressure"),
        air_number_density = _refresh_forcing_field!(
            destination.air_number_density, source.air_number_density,
            "forcing.air_number_density"),
        fixed_species = _refresh_forcing_fields!(
            destination.fixed_species, source.fixed_species,
            "forcing.fixed_species"),
        photolysis = _refresh_forcing_fields!(
            destination.photolysis, source.photolysis,
            "forcing.photolysis"),
        auxiliaries = _refresh_forcing_fields!(
            destination.auxiliaries, source.auxiliaries,
            "forcing.auxiliaries"),
        envelope = source.envelope)
end

function _project_panel_field_host!(stage, source, layout)
    @inbounds for cell in eachindex(stage)
        stage[cell] = source[_full_panel_index(Int32(cell), layout)]
    end
    return stage
end

function _refresh_panel_forcing_field!(destination::AbstractArray,
                                       source::AbstractArray, layout,
                                       full_shape, host_stage, label)
    interior_cells = length(destination)
    if length(source) == interior_cells
        if source !== destination
            _same_refresh_backend(destination, source, label)
            copyto!(destination, source)
        end
    elseif length(source) == prod(full_shape)
        if _same_refresh_backend(destination, source, label)
            backend = KernelAbstractions.get_backend(destination)
            kernel! = _project_panel_field!(backend)
            kernel!(destination, source, layout; ndrange = interior_cells)
        else
            host_stage === nothing && throw(ArgumentError(
                "$label needs host staging for cross-backend halo projection"))
            _project_panel_field_host!(host_stage, source, layout)
            copyto!(destination, host_stage)
        end
    else
        throw(DimensionMismatch(
            "$label has $(length(source)) cells; expected $interior_cells " *
            "packed interior or $(prod(full_shape)) halo-padded cells"))
    end
    return destination
end

_refresh_panel_forcing_field!(destination::AbstractArray, source::Number,
                              layout, full_shape, host_stage, label) =
    _refresh_forcing_field!(destination, source, label)

_refresh_panel_forcing_field!(destination, source, layout, full_shape,
                              host_stage, label) =
    _refresh_forcing_field!(destination, source, label)

function _refresh_panel_cell_column!(destination, source::Number, field,
                                     layout, full_shape, host_stage, label)
    return _refresh_cell_column!(destination, source, field, label)
end

function _refresh_panel_cell_column!(destination, source::AbstractArray, field,
                                     layout, full_shape, host_stage, label)
    interior_cells = size(destination, 1)
    if length(source) == interior_cells
        return _refresh_cell_column!(destination, source, field, label)
    elseif length(source) != prod(full_shape)
        throw(DimensionMismatch(
            "$label has $(length(source)) cells; expected $interior_cells " *
            "packed interior or $(prod(full_shape)) halo-padded cells"))
    end

    if _same_refresh_backend(destination, source, label)
        backend = KernelAbstractions.get_backend(destination)
        kernel! = _refresh_panel_cell_field!(backend)
        kernel!(destination, source, Int32(field), layout;
                ndrange = interior_cells)
    else
        host_stage === nothing && throw(ArgumentError(
            "$label needs host staging for cross-backend halo projection"))
        _project_panel_field_host!(host_stage, source, layout)
        copyto!(@view(destination[:, field]), host_stage)
    end
    return nothing
end

function _refresh_panel_cell_forcing_fields!(
        destination::AtmosChemistry.PackedForcingFields{
            Names, AtmosChemistry.CellForcingFields},
        source::AtmosChemistry.PackedForcingFields{
            Names, AtmosChemistry.CellForcingFields},
        layout, full_shape, host_stage, label) where Names
    source_rows = size(source.values, 1)
    interior_cells = size(destination.values, 1)
    if source_rows == interior_cells
        if source.values !== destination.values
            _same_refresh_backend(destination.values, source.values, label)
            copyto!(destination.values, source.values)
        end
        return destination
    end
    source_rows == prod(full_shape) || throw(DimensionMismatch(
        "$label has $source_rows rows; expected $interior_cells packed " *
        "interior or $(prod(full_shape)) halo-padded rows"))
    for field in eachindex(Names)
        source_column = @view(source.values[:, field])
        _refresh_panel_cell_column!(
            destination.values, source_column, field, layout, full_shape,
            host_stage, "$label.$(Names[field])")
    end
    return destination
end

function _refresh_panel_cell_forcing_fields!(
        destination::AtmosChemistry.PackedForcingFields{
            Names, AtmosChemistry.CellForcingFields},
        source, layout, full_shape, host_stage, label) where Names
    _require_forcing_names(source, Names, label)
    for field in eachindex(Names)
        values = _named_forcing_source(source, Names[field])
        _refresh_panel_cell_column!(
            destination.values, values, field, layout, full_shape,
            host_stage, "$label.$(Names[field])")
    end
    return destination
end

function _refresh_panel_forcing_fields!(destination::NamedTuple{Names},
                                        source::NamedTuple, layout,
                                        full_shape, host_stage,
                                        label) where Names
    propertynames(source) == Names || throw(ArgumentError(
        "$label changed names from $Names to $(propertynames(source))"))
    values = map(Names) do name
        _refresh_panel_forcing_field!(getproperty(destination, name),
            getproperty(source, name), layout, full_shape, host_stage,
            "$label.$name")
    end
    return NamedTuple{Names}(values)
end

_refresh_panel_forcing_fields!(
        destination::AtmosChemistry.PackedForcingFields{
            <:Any, AtmosChemistry.ConstantForcingFields},
        source, layout, full_shape, host_stage, label) =
    _refresh_constant_forcing_fields!(destination, source, label)

_refresh_panel_forcing_fields!(
        destination::AtmosChemistry.PackedForcingFields{
            <:Any, AtmosChemistry.CellForcingFields},
        source, layout, full_shape, host_stage, label) =
    _refresh_panel_cell_forcing_fields!(
        destination, source, layout, full_shape, host_stage, label)

function _refresh_panel_forcing!(destination, source, layout, full_shape,
                                 host_stage)
    source === destination && return destination
    return AtmosChemistry.ChemistryForcing(
        temperature = _refresh_panel_forcing_field!(
            destination.temperature, source.temperature, layout, full_shape,
            host_stage, "forcing.temperature"),
        pressure = _refresh_panel_forcing_field!(
            destination.pressure, source.pressure, layout, full_shape,
            host_stage, "forcing.pressure"),
        air_number_density = _refresh_panel_forcing_field!(
            destination.air_number_density, source.air_number_density,
            layout, full_shape, host_stage, "forcing.air_number_density"),
        fixed_species = _refresh_panel_forcing_fields!(
            destination.fixed_species, source.fixed_species, layout,
            full_shape, host_stage, "forcing.fixed_species"),
        photolysis = _refresh_panel_forcing_fields!(
            destination.photolysis, source.photolysis, layout, full_shape,
            host_stage, "forcing.photolysis"),
        auxiliaries = _refresh_panel_forcing_fields!(
            destination.auxiliaries, source.auxiliaries, layout, full_shape,
            host_stage, "forcing.auxiliaries"),
        envelope = source.envelope)
end

function _forcing_for_call(workspace, provider::ATC.CallUpdatedChemistryForcing,
                           operator, state::AtmosTransport.CellState,
                           meteo, grid)
    source = ATC.chemistry_forcing(provider, meteo, grid)
    _validate_update_schema(provider.initial, source, "forcing")
    return _refresh_forcing!(workspace.forcing, source)
end

function _forcing_for_call(workspace, provider::ATC.CallUpdatedChemistryForcing,
                           operator, state::AtmosTransport.CubedSphereState,
                           meteo, grid)
    source = ATC.chemistry_forcing(provider, meteo, grid)
    _validate_update_schema(provider.initial, source, "forcing")
    panel_size = size(state.air_mass[1])
    return ntuple(6) do panel
        panel_source = source isa Tuple ? source[panel] : source
        _refresh_panel_forcing!(workspace.forcing[panel], panel_source,
                                workspace.layout, panel_size,
                                workspace.host_forcing_stage)
    end
end

# Public execution boundary

function ATC.apply!(state::AtmosTransport.CellState{AtmosTransport.DryBasis},
                    meteo, grid,
                    operator::AtmosTransport.AtmosChemistryOperator, duration;
                    workspace = nothing)
    workspace === nothing &&
        (workspace = ATM._chemistry_workspace_for(operator, state, grid))
    architecture = AtmosChemistry.architecture(operator.model)
    _validate_apply_backend!(architecture, state, workspace)
    tracers = reshape(state.tracers_raw, length(state.air_mass), :)
    air_mass = reshape(state.air_mass, :)
    forcing = _forcing_for_call(
        workspace, operator.forcing_provider, operator, state, meteo, grid)
    _validate_transport_forcing(
        operator.model, forcing, workspace.cell_count)
    _validate_storage!(workspace, air_mass, forcing,
                       workspace.layout, "transport")
    _apply_to_storage!(
        tracers, air_mass, operator, forcing,
        duration, workspace, 0)
    _finish_transport_diagnostics!(
        workspace, architecture)
    failures = _failed_cells(workspace.diagnostics)
    operator.model.policy.throw_on_failure && !isempty(failures) &&
        throw(AtmosChemistry.ChemistrySolveError(failures))
    return state
end

function ATC.apply!(state::AtmosTransport.CubedSphereState{
                        AtmosTransport.DryBasis}, meteo, grid,
                    operator::AtmosTransport.AtmosChemistryOperator, duration;
                    workspace = nothing)
    workspace === nothing &&
        (workspace = ATM._chemistry_workspace_for(operator, state, grid))
    architecture = AtmosChemistry.architecture(operator.model)
    _validate_apply_backend!(architecture, state, workspace)
    forcing = _forcing_for_call(
        workspace, operator.forcing_provider, operator, state, meteo, grid)
    _validate_transport_forcing(
        operator.model, forcing, workspace.cell_count)
    _validate_cubed_sphere_storage!(
        workspace, state.air_mass, forcing, workspace.layout)
    @inbounds for panel in 1:6
        _apply_to_storage!(
            state.tracers_raw[panel], state.air_mass[panel], operator,
            forcing[panel], duration, workspace,
            (panel - 1) * workspace.cell_count)
    end
    _finish_transport_diagnostics!(
        workspace, architecture)
    failures = _failed_cells(workspace.diagnostics)
    operator.model.policy.throw_on_failure && !isempty(failures) &&
        throw(AtmosChemistry.ChemistrySolveError(failures))
    return state
end

function ATC.apply!(state::Union{
                        AtmosTransport.CellState,
                        AtmosTransport.CubedSphereState}, meteo, grid,
                    operator::AtmosTransport.AtmosChemistryOperator, duration;
                    workspace = nothing)
    throw(ArgumentError(
        "AtmosChemistry coupling requires DryBasis transport state; " *
        "MoistBasis conversion needs water-vapor mass before chemistry"))
end

end
