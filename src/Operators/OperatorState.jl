# Persistent operator state used by checkpoint replay.
_copy_state(::Nothing) = nothing
_copy_state(a::AbstractArray) = copy(a)
_copy_state(t::Tuple) = map(_copy_state, t)

const _zero_state! = fill_zero!

_copy_state!(::Nothing, ::Nothing) = nothing
_copy_state!(dst::AbstractArray, src::AbstractArray) = copyto!(dst, src)
_copy_state!(dst::Tuple, src::Tuple) = (foreach(_copy_state!, dst, src); dst)

"""
    operator_state(x) -> snapshot

The state `x` (an operator or workspace) carries from one call to the next,
copied; `nothing` when it carries none. See `restore_operator_state!` and
`reset_operator_state!`.
"""
operator_state(_x) = nothing
"""    restore_operator_state!(x, snapshot)  — install a snapshot of `operator_state(x)`."""
restore_operator_state!(_x, ::Nothing) = nothing
"""    reset_operator_state!(x)  — give `x` the persistent state of a fresh run."""
reset_operator_state!(_x) = nothing
