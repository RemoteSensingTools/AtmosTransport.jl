# Persistent convection state for checkpoint replay.
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
