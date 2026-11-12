# Persistent convection state for checkpoint replay.

"""
    convection_checkpoint_state(op, workspace, state::CubedSphereState, mesh::CubedSphereMesh)

Copy persistent convection state for a preallocated checkpoint, sizing lazy
column carries for the mesh and the state's tracer count without changing the
workspace. An unused carry is equivalent to a zero carry of that shape.
"""
convection_checkpoint_state(::NoConvection, workspace, _state, _mesh) = operator_state(workspace)

function convection_checkpoint_state(::TM5Convection, ws::TM5Workspace,
                                     state::CubedSphereState, mesh::CubedSphereMesh)
    dims = (mesh.Nc, mesh.Nc, size(first(state.tracers_raw), 4))
    return (carry = _fitted_ledger_carry(_copy_state(ws.ledger_carry), dims),)
end

function convection_checkpoint_state(op::CMFMCConvection, ws::CMFMCWorkspace,
                                     state::CubedSphereState, mesh::CubedSphereMesh)
    snap = operator_state(ws)
    op.clamp && return snap
    dims = (mesh.Nc, mesh.Nc, size(first(state.tracers_raw), 4))
    return merge(snap, (carry = _fitted_ledger_carry(snap.carry, dims),))
end

convection_checkpoint_state(op::CMFMCMatrixConvection, ws::CMFMCMatrixWorkspace,
                            state::CubedSphereState, mesh::CubedSphereMesh) =
    convection_checkpoint_state(op.inner, ws.tm5_workspace, state, mesh)

# CMFMC convection: the ledger carry and the window's cached CFL sub-steps.
operator_state(ws::CMFMCWorkspace) =
    (carry = _copy_state(ws.ledger_carry[]), n_sub = ws.cached_n_sub[],
     valid = ws.cache_valid[], clamp = ws.cached_clamp[])
function restore_operator_state!(ws::CMFMCWorkspace, snap::NamedTuple)
    ws.ledger_carry[] = _copy_state(snap.carry)
    ws.cached_n_sub[] = snap.n_sub
    ws.cache_valid[] = snap.valid
    ws.cached_clamp[] = snap.clamp
    return nothing
end
function reset_operator_state!(ws::CMFMCWorkspace)
    _zero_state!(ws.ledger_carry[])
    ws.cache_valid[] = false
    return nothing
end

# TM5 convection: the ledger carry.
operator_state(ws::TM5Workspace) = (carry = _copy_state(ws.ledger_carry),)
restore_operator_state!(ws::TM5Workspace, snap::NamedTuple) =
    (ws.ledger_carry = _copy_state(snap.carry); nothing)
reset_operator_state!(ws::TM5Workspace) = (_zero_state!(ws.ledger_carry); nothing)

# IFS convection: the ledger carry. (No checkpoint method: the operator has
# no adjoint, so no replay checkpoints it.)
operator_state(ws::IFSWorkspace) = (carry = _copy_state(ws.ledger_carry),)
restore_operator_state!(ws::IFSWorkspace, snap::NamedTuple) =
    (ws.ledger_carry = _copy_state(snap.carry); nothing)
reset_operator_state!(ws::IFSWorkspace) = (_zero_state!(ws.ledger_carry); nothing)

# CMFMC-matrix convection: its TM5 ledger carry. The derived TM5 rates are a
# function of the window's forcing alone, so a restore marks them stale and
# the next call derives them again, bit for bit.
operator_state(ws::CMFMCMatrixWorkspace) = operator_state(ws.tm5_workspace)
function restore_operator_state!(ws::CMFMCMatrixWorkspace, snap::NamedTuple)
    restore_operator_state!(ws.tm5_workspace, snap)
    ws.derived_valid[] = false
    return nothing
end
function reset_operator_state!(ws::CMFMCMatrixWorkspace)
    reset_operator_state!(ws.tm5_workspace)
    ws.derived_valid[] = false
    return nothing
end
