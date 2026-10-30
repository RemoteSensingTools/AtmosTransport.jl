"""
    CMFMCWorkspace{FT, QC, CA, L}

Per-sim pre-allocated workspace for [`CMFMCConvection`](@ref).

# Fields

- `qc_scratch :: QC` — updraft concentration buffer, one entry per
  cell, shape matching `air_mass` (`(Nx, Ny, Nz)` for structured
  grids, `(ncells, Nz)` for face-indexed grids). Reused across all
  substeps and all tracers.
- `cell_metrics :: CA` — pre-adapted cell-area metric vector used by
  the CFL scan and the kernel:
  `cell_areas_by_latitude(mesh)` for structured grids,
  per-cell `cell_area(mesh, c)` for face-indexed grids.
- `cached_n_sub :: Base.RefValue{Int}` — the CFL-derived sub-step
  count for the current met window. Stays valid as long as
  `cache_valid[] == true`; re-computed from the current CMFMC /
  air-mass state on the next `apply!` call when invalid. Int (not
  Float) because it's the integer sub-step count.
- `cache_valid :: Base.RefValue{Bool}` — sentinel; cleared via
  [`invalidate_cmfmc_cache!`](@ref) when the met window advances
  (`DrivenSimulation._maybe_advance_window!`).
- `ledger_carry :: Base.RefValue{L}` — persistent state of the column mass
  ledger, as in [`TM5Workspace`](@ref): per column and tracer, the part of
  the last update's rounding residual that the column's largest layer could
  not hold. Empty until the first `apply!` sizes it with zeros; not stored
  in restarts.

# Usage

Constructed once at `DrivenSimulation` setup time and reused for the
whole run. `Adapt.adapt_structure` preserves the cached scalar state
on the host while adapting `qc_scratch` to the requested backend.
"""
struct CMFMCWorkspace{FT, QC, CA, L}
    qc_scratch   :: QC
    cell_metrics :: CA
    cached_n_sub :: Base.RefValue{Int}
    cache_valid  :: Base.RefValue{Bool}
    cached_clamp :: Base.RefValue{Bool}   # clamp mode the cached n_sub was computed for
    ledger_carry :: Base.RefValue{L}      # persistent; sized by `_cmfmc_ledger_carry!`
end

"""
    CMFMCWorkspace(air_mass::AbstractArray{FT}; cell_metrics = nothing) -> CMFMCWorkspace

Construct a fresh workspace from an air-mass payload. `air_mass` may
be a single array (structured / face-indexed) or a cubed-sphere panel
tuple. Shape of `qc_scratch` matches `air_mass`.
"""
@inline _cmfmc_scratch_like(air_mass::AbstractArray{FT}) where FT = similar(air_mass)
@inline function _cmfmc_scratch_like(air_mass::NTuple{N, <:AbstractArray{FT}}) where {N, FT}
    return ntuple(i -> similar(air_mass[i]), N)
end

function CMFMCWorkspace(air_mass::AbstractArray{FT}; cell_metrics = nothing) where FT
    qc_scratch = _cmfmc_scratch_like(air_mass)
    metrics = if cell_metrics === nothing
        nothing
    else
        _cmfmc_metric_buffer(cell_metrics, FT)
    end
    carry = _empty_ledger_carry(air_mass, FT)
    return CMFMCWorkspace{FT, typeof(qc_scratch), typeof(metrics), typeof(carry)}(
        qc_scratch,
        metrics,
        Ref{Int}(1),
        Ref{Bool}(false),
        Ref{Bool}(false),
        Ref(carry),
    )
end

function CMFMCWorkspace(air_mass::NTuple{N, <:AbstractArray{FT}}; cell_metrics = nothing) where {N, FT}
    qc_scratch = _cmfmc_scratch_like(air_mass)
    metrics = if cell_metrics === nothing
        nothing
    else
        _cmfmc_metric_buffer(cell_metrics, FT)
    end
    carry = _empty_ledger_carry(air_mass, FT)
    return CMFMCWorkspace{FT, typeof(qc_scratch), typeof(metrics), typeof(carry)}(
        qc_scratch,
        metrics,
        Ref{Int}(1),
        Ref{Bool}(false),
        Ref{Bool}(false),
        Ref(carry),
    )
end

function Adapt.adapt_structure(to, ws::CMFMCWorkspace{FT}) where FT
    qc_scratch = Adapt.adapt(to, ws.qc_scratch)
    cell_metrics = ws.cell_metrics === nothing ? nothing : Adapt.adapt(to, ws.cell_metrics)
    carry = Adapt.adapt(to, ws.ledger_carry[])       # persistent state, moves with the workspace
    return CMFMCWorkspace{FT, typeof(qc_scratch), typeof(cell_metrics), typeof(carry)}(
        qc_scratch,
        cell_metrics,
        Ref{Int}(ws.cached_n_sub[]),
        Ref{Bool}(ws.cache_valid[]),
        Ref{Bool}(ws.cached_clamp[]),
        Ref(carry),
    )
end

"""
    invalidate_cmfmc_cache!(ws::CMFMCWorkspace) -> nothing

Mark the cached CFL `n_sub` as stale. Called on met-window
advance by `DrivenSimulation._maybe_advance_window!`. The next
`apply!` recomputes `n_sub` from the fresh CMFMC array.
"""
function invalidate_cmfmc_cache!(ws::CMFMCWorkspace)
    ws.cache_valid[] = false
    return nothing
end

# No-op for non-CMFMC workspaces (e.g. TM5Workspace); simplifies the
# call site in DrivenSimulation.
invalidate_cmfmc_cache!(::Any) = nothing


_cmfmc_ledger_carry!(ws::CMFMCWorkspace, dims::Dims) =
    ws.ledger_carry[] = _fitted_ledger_carry(ws.ledger_carry[], dims)
