# ===========================================================================
# TM5Workspace
# ===========================================================================

"""
    TM5Workspace{FT, M, P, C, F, A, CA, L}

Per-sim pre-allocated workspace for [`TM5Convection`](@ref).

# Fields

- `conv1 :: M` — `conv1 = I - dt·D` matrix slab, one `(Nz, Nz)`
  block per column. Parametric on array type so
  `Adapt.adapt_structure` can swap CPU ↔ GPU without changing the
  `TM5Workspace` type constructor.
  Shapes per topology:
  - Structured LatLon: `(Nz, Nz, Nx, Ny)` — 4D.
  - Face-indexed ReducedGaussian: `(Nz, Nz, ncells)` — 3D.
  - Panel-native CubedSphere: `NTuple{6, AbstractArray{FT, 4}}`
    with per-panel shape `(Nz, Nz, Nc, Nc)`.
- `pivots :: P` — permutation vector from partial-pivot LU, one
  Nz-length Int slice per column. Preserved so the adjoint can
  replay the same factorization with transposed back-substitution.
  Shapes strip the leading Nz
  from `conv1` shape: `(Nz, Nx, Ny)` / `(Nz, ncells)` /
  `NTuple{6, (Nz, Nc, Nc)}`.
- `cloud_dims :: C` — per-column `(icltop, iclbas, icllfs)` triple
  in AtmosTransport indexing (k=1=TOA, k=Nz=surface; the
  preprocessor delivers forcings in this orientation so the solver
  has zero orientation logic). Shape `(3, Nx, Ny)` /
  `(3, ncells)` / `NTuple{6, (3, Nc, Nc)}`.
- `f_scratch :: F` — per-column intermediate matrix for the
  matrix build (TM5 `f(0:lmx, 1:lmx)`, with the updraft merged into
  `f`). Pre-allocated so `_tm5_build_conv1!` runs without heap
  allocation inside KA kernels (mandatory on GPU; same contract on
  CPU for parity). `f_scratch` aliases `conv1` in the production
  workspace because `conv1` is only needed after `f` has been
  converted into `I - dt*D`; this saves one dense `(Nz, Nz)` slab
  per column. There is no standalone `fu_scratch` field — the
  updraft and downdraft passes write disjoint index ranges, so the
  builder writes directly into `f`.
- `amu_scratch :: A`, `amd_scratch :: A` — length-`(Nz+1)`
  per-column boundary-aware mass-flux vectors (TM5 `amu(0:lmx)` /
  `amd(0:lmx)`). Same allocation policy as the scratch matrices.
- `cell_metrics :: CA` — pre-adapted cell-area metrics used to
  convert runtime `air_mass` from kg per grid cell to the kg/m²
  column mass expected by the TM5 matrix. `nothing` is allowed only
  for standalone unit-area column tests; topology-level `apply!`
  requires this field to be populated.
- `ledger_carry :: L` — persistent state of the column mass ledger: per
  column and tracer, the part of the last solve's rounding residual that
  the column's largest cell could not hold (about half its ulp at most; see
  `_ledger_deposit`). Shape `(Nx, Ny, Nt)` / `(ncells, Nt)` /
  `NTuple{6, (Nc, Nc, Nt)}`. The constructor allocates it empty, with the
  topology's storage type; the first `apply!` sizes it with zeros, and a
  call with another tracer layout starts from zero again. Restarts do not
  store it: a restarted run drops about half an ulp of each column's
  largest cell, once.

# Usage

Constructed once at `DrivenSimulation` setup time via
`convection_workspace_for(::TM5Convection, state, grid)`.
`Adapt.adapt_structure` adapts each array to the requested backend
without reallocating on the host.
"""
mutable struct TM5Workspace{FT, M, P, C, F, A, CA, L}
    conv1       :: M
    pivots      :: P
    cloud_dims  :: C
    f_scratch   :: F
    amu_scratch :: A
    amd_scratch :: A
    cell_metrics :: CA
    scratch_columns :: Int
    defer_scratch :: Bool
    ledger_carry :: L          # persistent; sized by `_tm5_ledger_carry!`
end

# Topology-shape introspection helpers used by the tile workspace
# constructor. Single dispatch on the air-mass payload picks Nz,
# the per-launch cell count, and a "template" array (used by
# `similar` to inherit backend / element type) — the workspace
# itself is topology-agnostic.

@inline _tm5_template(air_mass::AbstractArray) = air_mass
@inline _tm5_template(air_mass::NTuple)        = air_mass[1]

# Number of vertical layers — the leading two `(Nz, Nz)` dims of
# the matrix slab.
@inline _tm5_extract_Nz(air_mass::AbstractArray{<:Any, 3}) = size(air_mass, 3)
@inline _tm5_extract_Nz(air_mass::AbstractArray{<:Any, 2}) = size(air_mass, 2)
@inline _tm5_extract_Nz(air_mass::NTuple{N, <:AbstractArray{<:Any, 3}}) where N =
    size(air_mass[1], 3)

# Per-launch cell count — the largest single kernel ndrange the
# workspace has to cover. CS panels are launched one at a time,
# so the workspace is sized for one panel (`Nc²`), not all six.
@inline _tm5_total_cells_per_launch(air_mass::AbstractArray{<:Any, 3}) =
    size(air_mass, 1) * size(air_mass, 2)
@inline _tm5_total_cells_per_launch(air_mass::AbstractArray{<:Any, 2}) =
    size(air_mass, 1)
@inline _tm5_total_cells_per_launch(air_mass::NTuple{N, <:AbstractArray{<:Any, 3}}) where N =
    size(air_mass[1], 1) * size(air_mass[1], 2)

"""
    derive_tile_columns(::Type{FT}, Nz, budget_gib, total_cells) -> Int

Pick a tile size `B` from a per-cell memory cost and a target
budget. The cost model accounts for every per-cell field that
[`TM5Workspace`](@ref) tiles:

- `conv1` : `Nz²` × `sizeof(FT)`
- `amu_scratch` + `amd_scratch` : `2(Nz+1)` × `sizeof(FT)`
- `pivots` : `Nz` × `sizeof(Int)`
- `cloud_dims` : `3` × `sizeof(Int)`

`f_scratch` is a structural alias for `conv1` (saves one matrix
slab per cell). `B` is clamped between 256 (avoid pathological
launches) and `total_cells` (one tile covers the whole topology
when the budget allows it).
"""
function derive_tile_columns(::Type{FT}, Nz::Integer,
                             budget_gib::Real, total_cells::Integer) where FT
    Nz > 0 || throw(ArgumentError("derive_tile_columns: Nz must be > 0"))
    total_cells > 0 ||
        throw(ArgumentError("derive_tile_columns: total_cells must be > 0"))
    budget_gib > 0 ||
        throw(ArgumentError("derive_tile_columns: budget_gib must be > 0"))
    per_cell_bytes = sizeof(FT) * (Nz * Nz + 2 * (Nz + 1)) +
                     sizeof(Int) * (Nz + 3)
    budget_bytes = round(Int, budget_gib * (1 << 30))
    raw = fld(budget_bytes, per_cell_bytes)
    # Floor of 256 avoids pathologically tiny launches; the
    # `min(total_cells, …)` cap keeps small topologies from
    # over-allocating beyond what they need.
    return Int(min(Int(total_cells), max(256, Int(raw))))
end

"""
    TM5Workspace(air_mass; tile_columns = …,
                            tile_workspace_gib = nothing,
                            cell_metrics = nothing,
                            defer_scratch = false) -> TM5Workspace

Construct a fresh workspace from an air-mass payload. `air_mass`
may be a single array (structured `(Nx, Ny, Nz)` or face-indexed
`(ncells, Nz)`) or a cubed-sphere panel tuple — the workspace
itself is topology-agnostic. The per-column element type of
`conv1` matches `eltype(air_mass)`.

The per-launch column count `B` is set by exactly one of:

- `tile_columns::Integer` (explicit). Default
  `_tm5_total_cells_per_launch(air_mass)` — one tile covers the
  whole launch and the workspace is bit-equal to the earlier
  per-cell allocator. Production paths use this branch when they
  already know `B`.
- `tile_workspace_gib::Real` (budget). Picks `B` via
  [`derive_tile_columns`](@ref) from
  [`TM5Convection`](@ref)'s `tile_workspace_gib` field. Bypassed
  if `tile_columns` is also passed (explicit wins).
- `cell_metrics` carries topology cell areas. Production
  `convection_workspace_for(::TM5Convection, ...)` supplies this;
  leaving it `nothing` is only for unit-area solver tests.

Specifying both tile settings is an error. `defer_scratch=true` retains the
configured tile size but allocates no legacy matrix payload until a legacy
solve actually needs it. Adaptation drops this recomputable scratch while
preserving cell metrics; each backend allocates its own tile on demand.
"""
function TM5Workspace(air_mass;
                      tile_columns::Union{Integer, Nothing} = nothing,
                      tile_workspace_gib::Union{Real, Nothing} = nothing,
                      cell_metrics = nothing,
                      defer_scratch::Bool = false)
    Nz          = _tm5_extract_Nz(air_mass)
    template    = _tm5_template(air_mass)
    FT          = eltype(template)
    total_cells = _tm5_total_cells_per_launch(air_mass)
    B = if tile_columns !== nothing && tile_workspace_gib !== nothing
        throw(ArgumentError(
            "TM5Workspace: pass either `tile_columns` or `tile_workspace_gib`, not both"))
    elseif tile_columns !== nothing
        Int(tile_columns)
    elseif tile_workspace_gib !== nothing
        derive_tile_columns(FT, Nz, Float64(tile_workspace_gib), total_cells)
    else
        total_cells
    end
    B > 0 || throw(ArgumentError("TM5Workspace: tile size must be > 0"))
    allocated_columns = defer_scratch ? 0 : B
    conv1       = similar(template, FT,  Nz, Nz, allocated_columns)
    pivots      = similar(template, Int, Nz,     allocated_columns)
    cloud_dims  = similar(template, Int, 3,      allocated_columns)
    f_scratch   = conv1                     # alias — see docstring
    amu_scratch = similar(template, FT,  Nz + 1, allocated_columns)
    amd_scratch = similar(template, FT,  Nz + 1, allocated_columns)
    metrics = cell_metrics === nothing ? nothing : _cmfmc_metric_buffer(cell_metrics, FT)
    carry = _empty_ledger_carry(air_mass, FT)
    return TM5Workspace{FT,
                        typeof(conv1), typeof(pivots), typeof(cloud_dims),
                        typeof(f_scratch), typeof(amu_scratch),
                        typeof(metrics), typeof(carry)}(
        conv1, pivots, cloud_dims,
        f_scratch, amu_scratch, amd_scratch, metrics, B, defer_scratch, carry,
    )
end

# Only legacy entry points call this. Retain the configured tile size rather
# than falling back to single-column launches. Allocate all buffers before
# installing them, so an allocation failure leaves a deferred workspace intact.
function _ensure_tm5_scratch!(ws::TM5Workspace{FT}) where FT
    size(ws.conv1, 3) > 0 && return ws
    Nz = size(ws.conv1, 1)
    B = ws.scratch_columns
    conv1 = similar(ws.conv1, FT, Nz, Nz, B)
    pivots = similar(ws.pivots, Int, Nz, B)
    cloud_dims = similar(ws.cloud_dims, Int, 3, B)
    amu = similar(ws.amu_scratch, FT, Nz+1, B)
    amd = similar(ws.amd_scratch, FT, Nz+1, B)
    ws.conv1 = conv1
    ws.f_scratch = conv1
    ws.pivots = pivots
    ws.cloud_dims = cloud_dims
    ws.amu_scratch = amu
    ws.amd_scratch = amd
    return ws
end

function Adapt.adapt_structure(to, ws::TM5Workspace{FT}) where FT
    if ws.defer_scratch
        # Scratch is recomputed by every legacy solve. Preserve persistent
        # metrics/cache below, but never transfer an unused legacy tile.
        Nz = size(ws.conv1, 1)
        conv1 = Adapt.adapt(to, similar(ws.conv1, FT, Nz, Nz, 0))
        pivots = Adapt.adapt(to, similar(ws.pivots, Int, Nz, 0))
        cloud_dims = Adapt.adapt(to, similar(ws.cloud_dims, Int, 3, 0))
        f_scratch = conv1
        amu_scratch = Adapt.adapt(to, similar(ws.amu_scratch, FT, Nz+1, 0))
        amd_scratch = Adapt.adapt(to, similar(ws.amd_scratch, FT, Nz+1, 0))
    else
        f_aliases_conv1 = ws.conv1 === ws.f_scratch
        conv1       = Adapt.adapt(to, ws.conv1)
        pivots      = Adapt.adapt(to, ws.pivots)
        cloud_dims  = Adapt.adapt(to, ws.cloud_dims)
        f_scratch   = f_aliases_conv1 ? conv1 : Adapt.adapt(to, ws.f_scratch)
        amu_scratch = Adapt.adapt(to, ws.amu_scratch)
        amd_scratch = Adapt.adapt(to, ws.amd_scratch)
    end
    cell_metrics = ws.cell_metrics === nothing ? nothing : Adapt.adapt(to, ws.cell_metrics)
    # The carry is persistent state, not scratch: it moves with the workspace.
    carry = Adapt.adapt(to, ws.ledger_carry)
    return TM5Workspace{FT,
                        typeof(conv1), typeof(pivots), typeof(cloud_dims),
                        typeof(f_scratch), typeof(amu_scratch),
                        typeof(cell_metrics), typeof(carry)}(
        conv1, pivots, cloud_dims,
        f_scratch, amu_scratch, amd_scratch, cell_metrics,
        ws.scratch_columns, ws.defer_scratch, carry,
    )
end


_tm5_ledger_carry!(ws::TM5Workspace, dims::Dims) =
    ws.ledger_carry = _fitted_ledger_carry(ws.ledger_carry, dims)
