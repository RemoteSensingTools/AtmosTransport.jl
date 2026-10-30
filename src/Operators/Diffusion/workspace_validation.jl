# Workspace validation shared by forward solves, advection preflight and adjoint wiring.

function _check_workspace_array(array, expected, name, label; reference = nothing)
    array isa AbstractArray && size(array) == expected || throw(DimensionMismatch(
        "diffusion workspace `$name` $label has shape " *
        "$(array isa AbstractArray ? size(array) : typeof(array)); expected $expected. " *
        "Build the workspace with DiffusionWorkspace(state)."))
    if reference !== nothing
        eltype(array) === eltype(reference) || throw(ArgumentError(
            "diffusion workspace $label and state must share one element type"))
        typeof(get_backend(array)) === typeof(get_backend(reference)) || throw(ArgumentError(
            "diffusion workspace $label and state must use the same backend"))
    end
    return nothing
end

function _check_column_workspace(arrays::NamedTuple, expected, label; reference = nothing)
    for (name, array) in pairs(arrays)
        _check_workspace_array(array, expected, name, label; reference)
    end
    return nothing
end

"""
    _column_ledger_carry(form, carry, columns, Nt, packed, tracer, label) -> (carry, first_slot)

Ledger carry of a column solve and the carry slot of its first tracer. The
flux form keeps a carry per column and tracer (`DiffusionWorkspace.ledger_carry`,
shaped `(columns..., Nt)`): packed tracers use slots `1:Nt`; a single-tracer
array uses slot `tracer`, its index in the state, and keeps no carry with
`tracer = nothing`, so the call is a function of its input alone (adjoint
replays rely on this). A workspace without a carry gives the ledger without one.

"""
function _column_ledger_carry(::_TracerMassFluxForm, carry, columns, Nt, packed, tracer, label)
    if packed
        carry === nothing && return nothing, 1
        size(carry) == (columns..., Nt) || throw(DimensionMismatch(
            "diffusion ledger carry $label has shape $(size(carry)); packed tracers need " *
            "$((columns..., Nt)). Build the workspace with DiffusionWorkspace(state)."))
        return carry, 1
    end
    tracer === nothing && return nothing, 1
    carry isa AbstractArray && size(carry)[1:end-1] == columns &&
        1 <= tracer <= size(carry, ndims(carry)) || throw(ArgumentError(
            "diffusion tracer = $tracer needs a workspace ledger carry with that tracer slot " *
            "and columns $columns $label; build the workspace with DiffusionWorkspace(state)."))
    return carry, Int(tracer)
end

"""
    _check_packed_carry(workspace, columns, n_tracers, FT, backend)
    _check_packed_carry(workspace, q_raw)

Per-tracer column array of a packed lat-lon or reduced-Gaussian workspace (the
shape of its ledger carry), checked against the columns `(Nx, Ny)` or
`(ncells,)`, the tracer count, the element type and the backend of the state,
so a workspace built for another state fails before any tracer is touched.
"""
function _check_packed_carry(workspace::DiffusionWorkspace, columns::Tuple, n_tracers::Integer,
                            ::Type{FT}, backend) where FT
    carry = workspace.ledger_carry
    shape = (columns..., Int(n_tracers))
    carry isa AbstractArray && size(carry) == shape && eltype(carry) === FT &&
        typeof(get_backend(carry)) === typeof(backend) && return carry
    throw(DimensionMismatch(
        "packed diffusion needs per-tracer column carry of shape $shape, element type $FT " *
        "and the state's backend; build the workspace with DiffusionWorkspace(state) or " *
        "DiffusionWorkspace(air_mass, n_tracers)"))
end
_check_packed_carry(workspace::DiffusionWorkspace, q_raw::AbstractArray) =
    _check_packed_carry(workspace, size(q_raw)[1:end-2], size(q_raw, ndims(q_raw)), eltype(q_raw),
                       get_backend(q_raw))

# Validate the direct operator API before any advection sweep mutates state.
# TransportModel supplies this workspace automatically; direct callers must
# make the operator's storage ownership explicit.
@inline preflight_diffusion(::NoDiffusion, _, _, ::AbstractArray, ::Integer) = nothing
@inline preflight_diffusion(::NoDiffusion, _, _, ::NTuple{6}, ::Integer, ::Integer) = nothing

"""
    check_diffusion_workspace(workspace, air_mass, n_tracers; halo_width)

Validate diffusion storage against a state before its tracers are changed.
Structured and face-indexed arrays use their full shape; panel tuples use
interior scratch and require `halo_width`. The ledger carry binds storage to
the tracer count, precision and backend without a second unused allocation.
"""
function check_diffusion_workspace(workspace, air_mass::AbstractArray, n_tracers::Integer)
    workspace isa DiffusionWorkspace || throw(ArgumentError("requires DiffusionWorkspace(state)"))
    _check_column_workspace((factors = workspace.factors,
                             layer_thickness = workspace.layer_thickness),
                            size(air_mass), ""; reference = air_mass)
    _check_packed_carry(workspace, size(air_mass)[1:end-1], n_tracers,
                        eltype(air_mass), get_backend(air_mass))
    return nothing
end

function check_diffusion_workspace(workspace, air_mass::NTuple{6}, n_tracers::Integer;
                                    halo_width::Integer)
    workspace isa DiffusionWorkspace || throw(ArgumentError("requires DiffusionWorkspace(state)"))
    Hp = Int(halo_width)
    Nxi, Nyi, Nz = size(air_mass[1])
    expected = (Nxi - 2Hp, Nyi - 2Hp, Nz)
    workspace.factors isa NTuple{6} && workspace.layer_thickness isa NTuple{6} &&
        workspace.ledger_carry isa NTuple{6} || throw(DimensionMismatch(
            "cubed-sphere diffusion workspace must contain six factor, layer-thickness, and ledger carry panels"))
    for p in 1:6
        _check_column_workspace((factors = workspace.factors[p],
                                 layer_thickness = workspace.layer_thickness[p]),
                                expected, "panel $p"; reference = air_mass[p])
        _check_workspace_array(workspace.ledger_carry[p], (expected[1:2]..., Int(n_tracers)),
                                :ledger_carry, "panel $p"; reference = air_mass[p])
    end
    return nothing
end

"""
    preflight_diffusion(op, workspace, dt, air_mass, n_tracers[, halo_width])

Check the timestep and diffusion workspace before an advection sweep changes
state. A `NoDiffusion` operator skips these checks.
"""
function preflight_diffusion(op::AbstractDiffusion, workspace, dt, air_mass, n_tracers,
                             halo_width = nothing)
    workspace isa DiffusionWorkspace || throw(ArgumentError(
        "$(typeof(op)) requires `diffusion_workspace = DiffusionWorkspace(state)`; " *
        "got $(typeof(workspace))."))
    dt isa Real || throw(ArgumentError("$(typeof(op)) requires a real-valued `dt`; got $(repr(dt))."))
    if halo_width === nothing
        check_diffusion_workspace(workspace, air_mass, n_tracers)
    else
        check_diffusion_workspace(workspace, air_mass, n_tracers; halo_width)
    end
    return nothing
end

_ledger_panel(carry::NTuple{6}, p) = carry[p]
_ledger_panel(carry, p) = carry
_select_ledger_panel(carry, p, packed, tracer) =
    packed || tracer !== nothing ? _ledger_panel(carry, p) : nothing

function check_diffusion_workspace(workspace)
    workspace === nothing && throw(ArgumentError(
        "CS adjoint diffusion requires a panel-native DiffusionWorkspace"))
    hasproperty(workspace, :factors) && hasproperty(workspace, :layer_thickness) ||
        throw(ArgumentError(
            "CS adjoint diffusion requires a workspace with panel-native " *
            "`factors` and `layer_thickness` tuples"))
    w_scratch = workspace.factors
    dz_scratch = workspace.layer_thickness
    length(w_scratch) == 6 && length(dz_scratch) == 6 ||
        throw(DimensionMismatch("CS adjoint diffusion workspace must provide 6 panel scratch arrays"))
    return w_scratch, dz_scratch
end


# Adjoint panel validation shares the same shape checks as the forward solve.
function check_diffusion_workspace(workspace, lambda::AbstractArray, air_mass::AbstractArray;
                                    panel::Integer, columns::Tuple)
    size(lambda) == size(air_mass) || throw(DimensionMismatch(
        "adjoint tracer panel $panel shape $(size(lambda)) does not match air_mass shape $(size(air_mass))"))
    expected = (columns..., size(lambda, 3))
    _check_column_workspace((factors = workspace.factors[panel],
                             layer_thickness = workspace.layer_thickness[panel]),
                            expected, "panel $panel")
    return nothing
end
