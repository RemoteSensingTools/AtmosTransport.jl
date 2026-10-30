# ---------------------------------------------------------------------------
# CMFMCMatrixConvection — GEOS-derived rates (cmfmc, dtrain) routed through
# the TM5 backward-Euler LU solver.
#
# Motivation
#
# The GCHP-audited "two-pass on the env reservoir + qc cloud reservoir"
# CMFMC update (our `CMFMCConvection`) is not column-conservative under
# our state variables when run in isolation. GCHP itself relies on a
# dynamics-core PBL air-mass rescale that lives outside this operator.
# Our kernel also intentionally diverges from line-by-line GCHP in two
# places (cloud-base proxied from cmfmc rather than DQRCU, plus an added
# Pass-0 cloud-base mass-closure update for column self-consistency), so
# the label is "GCHP-audited" not "GCHP-faithful". Empirically, on the
# C180/L72 4-tracer convection-only experiment, this kernel drifts
# `Σ tracer_mass` by ~1–10% per 3-day run.
#
# This operator preserves `Σ tracer_mass` to roundoff for inert tracers by
# (a) deriving non-negative entrainment/detrainment rates from the GEOS
# binary's `cmfmc/dtrain` and (b) solving the resulting backward-Euler
# matrix LU — the same conservative core used by `TM5Convection`. The
# backward-Euler matrix has unit column sums under the `Σ entu = Σ detu`
# closure enforced by the derivation kernels. Its off-diagonal entries need
# not be nonnegative, so the matrix itself is not column-stochastic.
#
# It is NOT GCHP RAS numerics — it is GEOS-derived rates through TM5
# transport numerics. Use it where the conservation contract is required
# (4D-Var inversion adjoint, long integrations) and use `CMFMCConvection`
# for parity comparison.
# ---------------------------------------------------------------------------

"""
    CMFMCMatrixConvection(; tile_workspace_gib = 1.0,
                            use_collab_lu = false,
                            lmax_conv = 0,
                            n_merge = 1)

GEOS-derived rates through the TM5 LU solver. The kwarg-tuned knobs all
pass through to the inner [`TM5Convection`](@ref) that performs the
column solve — see its docstring for `tile_workspace_gib`,
`use_collab_lu`, `lmax_conv`, and `n_merge` semantics.

# Forcing requirements

`forcing.cmfmc` and `forcing.dtrain` must be populated on the GEOS binary's
basis (whatever `state.air_mass` is — `dry` for our production runtime).
`forcing.tm5_fields` is ignored. The operator runs once-per-met-window
derivation to convert `(cmfmc, dtrain) → (entu, detu, 0, 0)`, caching the
result in [`CMFMCMatrixWorkspace`](@ref) for reuse across all substeps.

# Conservation contract

`Σ(m·q)` is preserved to floating-point roundoff for any inert tracer:
the backward-Euler matrix has unit column sums once `Σ entu = Σ detu`,
which the derivation kernels guarantee by absorbing any boundary residual
at the surface layer. The solve therefore preserves the sum of its
right-hand side up to rounding.

# Adjoint

The derivation is constant in the state vector — it depends only on
forcings — so the state-space adjoint is exactly the inner TM5 LU adjoint
applied to the cached derived rates. See `Adjoints/ConvectionAdjoint.jl`.
"""
struct CMFMCMatrixConvection{FT} <: AbstractConvection
    inner :: TM5Convection{FT}
end

function CMFMCMatrixConvection(; tile_workspace_gib::Real = 1.0,
                                  use_collab_lu::Bool = false,
                                  lmax_conv::Integer = 0,
                                  n_merge::Integer = 1)
    inner = TM5Convection(; tile_workspace_gib = tile_workspace_gib,
                            use_collab_lu = use_collab_lu,
                            lmax_conv = Int(lmax_conv),
                            n_merge = Int(n_merge))
    return CMFMCMatrixConvection(inner)
end

# =========================================================================
# Workspace — composes a TM5Workspace with the per-window derived rate
# slabs. Lives in the convection workspace slot on the model, just like
# CMFMCWorkspace / TM5Workspace.
# =========================================================================

"""
    CMFMCMatrixWorkspace{FT, TS, EU, ED, V}

Per-sim pre-allocated workspace for [`CMFMCMatrixConvection`](@ref).

# Fields

- `tm5_workspace :: TS` — the underlying [`TM5Workspace`](@ref) carrying
  the column-tile matrix slab, pivots, and TM5 cell-area metrics.
- `derived_entu :: EU` — derived updraft entrainment rate, same shape and
  basis as `forcing.dtrain` (LL: `(Nx,Ny,Nz)`, RG: `(ncells,Nz)`, CS:
  `NTuple{6, (Nc,Nc,Nz)}`). Refreshed on every met-window advance via
  [`invalidate_cmfmc_matrix_cache!`](@ref); reused bit-exact across all
  substeps within a window.
- `derived_detu :: EU` — derived updraft detrainment rate, same shape as
  `derived_entu`. Includes the explicit GEOS `dtrain` plus any
  negative-entrainment residual folded in to keep `entu ≥ 0`.
- `zero_entd :: ED` — immutable zeros, same shape as `derived_entu`.
  Passed as the TM5 downdraft entrainment field (CMFMC has no downdraft
  in our binary contract).
- `zero_detd :: ED` — immutable zeros, same shape (TM5 downdraft
  detrainment).
- `derived_valid :: V` — `Base.RefValue{Bool}` flagging whether the
  cached derivation matches the current met window. Cleared via
  [`invalidate_cmfmc_matrix_cache!`](@ref) on window advance.

# Lifecycle

The derived rates are computed once per met-window from
`forcing.cmfmc + forcing.dtrain` by `_launch_cmfmc_matrix_derivation!`
(see `cmfmc_matrix_kernels.jl`). Within a window, `apply!` reuses the
cached arrays directly — the LU solve sees the same `(entu, detu, 0, 0)`
quadruplet every substep.
"""
struct CMFMCMatrixWorkspace{FT, TS, EU, ED, V}
    tm5_workspace :: TS
    derived_entu  :: EU
    derived_detu  :: EU
    zero_entd     :: ED
    zero_detd     :: ED
    derived_valid :: V
end

# Shape-matching `similar` for the (entu, detu) cache slabs against the
# expected shape of `forcing.dtrain`. LL / RG: same shape as `air_mass`.
# CS: forcings are halo-free `(Nc, Nc, Nz)` per panel while `air_mass` is
# halo-padded `(Nc+2Hp, Nc+2Hp, Nz)` — strip the halo when allocating.
@inline _cmfmc_matrix_rate_like(air_mass::AbstractArray{<:Any, 3},
                                  ::Type{FT}, halo::Int) where {FT} =
    similar(air_mass, FT)
@inline _cmfmc_matrix_rate_like(air_mass::AbstractArray{<:Any, 2},
                                  ::Type{FT}, halo::Int) where {FT} =
    similar(air_mass, FT)
@inline function _cmfmc_matrix_rate_like(air_mass::NTuple{N, <:AbstractArray{<:Any, 3}},
                                          ::Type{FT}, halo::Int) where {N, FT}
    return ntuple(i -> begin
        n1, n2, Nz = size(air_mass[i])
        similar(air_mass[i], FT, n1 - 2 * halo, n2 - 2 * halo, Nz)
    end, N)
end

"""
    CMFMCMatrixWorkspace(air_mass; tile_workspace_gib = nothing,
                          tile_columns = nothing,
                          cell_metrics = nothing,
                          halo_width = 0) -> CMFMCMatrixWorkspace

Construct a fresh workspace from an air-mass payload. `air_mass` may be a
single array (LL / RG) or a panel tuple (CS). The inner `TM5Workspace`
is built with the same tile-budget rules as a bare [`TM5Workspace`](@ref);
the derived-rate slabs are sized to mirror the layout of `forcing.dtrain`
(LL/RG: same as `air_mass`; CS: halo-stripped from `air_mass`).

`cell_metrics` is the topology cell-area metric (mandatory in production —
the TM5 matrix needs it to convert kg-per-cell air mass to the kg/m² basis
the convective fluxes share).

`halo_width` is the per-panel halo half-width `Hp` for cubed-sphere
payloads (the production CS factory passes `grid.horizontal.Hp`). LL and
RG leave it at 0.
"""
function CMFMCMatrixWorkspace(air_mass;
                              tile_workspace_gib::Union{Real, Nothing} = nothing,
                              tile_columns::Union{Integer, Nothing} = nothing,
                              cell_metrics = nothing,
                              halo_width::Integer = 0,
                              defer_scratch::Bool = false)
    FT = eltype(_tm5_template(air_mass))
    tm5_ws = TM5Workspace(air_mass;
                          tile_workspace_gib = tile_workspace_gib,
                          tile_columns = tile_columns,
                          cell_metrics = cell_metrics,
                          defer_scratch = defer_scratch)
    Hp = Int(halo_width)
    derived_entu = _cmfmc_matrix_rate_like(air_mass, FT, Hp)
    derived_detu = _cmfmc_matrix_rate_like(air_mass, FT, Hp)
    zero_entd    = _cmfmc_matrix_rate_like(air_mass, FT, Hp)
    zero_detd    = _cmfmc_matrix_rate_like(air_mass, FT, Hp)
    _fill_zero!(zero_entd)
    _fill_zero!(zero_detd)
    valid = Ref{Bool}(false)
    return CMFMCMatrixWorkspace{FT,
                                typeof(tm5_ws),
                                typeof(derived_entu),
                                typeof(zero_entd),
                                typeof(valid)}(
        tm5_ws, derived_entu, derived_detu, zero_entd, zero_detd, valid,
    )
end

@inline _fill_zero!(arr::AbstractArray) = (fill!(arr, zero(eltype(arr))); nothing)
@inline _fill_zero!(t::NTuple{N, <:AbstractArray}) where {N} =
    (for a in t; fill!(a, zero(eltype(a))); end; nothing)

function Adapt.adapt_structure(to, ws::CMFMCMatrixWorkspace{FT}) where {FT}
    tm5 = Adapt.adapt(to, ws.tm5_workspace)
    de  = Adapt.adapt(to, ws.derived_entu)
    dd  = Adapt.adapt(to, ws.derived_detu)
    ze  = Adapt.adapt(to, ws.zero_entd)
    zd  = Adapt.adapt(to, ws.zero_detd)
    valid = Ref{Bool}(ws.derived_valid[])
    return CMFMCMatrixWorkspace{FT, typeof(tm5), typeof(de), typeof(ze), typeof(valid)}(
        tm5, de, dd, ze, zd, valid,
    )
end

"""
    invalidate_cmfmc_matrix_cache!(ws::CMFMCMatrixWorkspace) -> nothing
    invalidate_cmfmc_matrix_cache!(::Any) -> nothing

Mark the derived `(entu, detu)` cache stale so the next `apply!`
re-derives from `forcing.cmfmc + forcing.dtrain`. Called on met-window
advance from `DrivenSimulation._maybe_advance_window!` via the shared
[`invalidate_cmfmc_cache!`](@ref) dispatch.
"""
function invalidate_cmfmc_matrix_cache!(ws::CMFMCMatrixWorkspace)
    ws.derived_valid[] = false
    return nothing
end
invalidate_cmfmc_matrix_cache!(::Any) = nothing

# Specialise the window-advance hook so DrivenSimulation's generic call
# site does not need to know the concrete convection operator.
invalidate_cmfmc_cache!(ws::CMFMCMatrixWorkspace) = invalidate_cmfmc_matrix_cache!(ws)

# =========================================================================
# Apply — per topology. Derives rates if stale, then dispatches into the
# existing TM5 `apply!` with a synthetic ConvectionForcing pointing at
# the workspace's cached derived rates.
# =========================================================================

@inline function _assert_cmfmc_matrix_forcing(forcing::ConvectionForcing)
    forcing.cmfmc === nothing && throw(ArgumentError(
        "CMFMCMatrixConvection requires `forcing.cmfmc` to be populated. " *
        "This binary has no CMFMC payload; pick a different convection scheme."))
    forcing.dtrain === nothing && throw(ArgumentError(
        "CMFMCMatrixConvection requires `forcing.dtrain` to be populated. " *
        "GCHP-style binaries deliver dtrain alongside cmfmc; check the " *
        "binary preprocessor flow."))
    return nothing
end

@inline function _refresh_cmfmc_matrix_rates!(ws::CMFMCMatrixWorkspace,
                                               forcing::ConvectionForcing)
    ws.derived_valid[] && return nothing
    _launch_cmfmc_matrix_derivation!(ws.derived_entu, ws.derived_detu,
                                     forcing.cmfmc, forcing.dtrain)
    ws.derived_valid[] = true
    return nothing
end

# Build a synthetic ConvectionForcing whose `tm5_fields` aliases the
# workspace's cached arrays. Allocation is one NamedTuple per substep —
# cheap host-side, no device traffic.
@inline function _cmfmc_matrix_synthetic_forcing(ws::CMFMCMatrixWorkspace)
    return ConvectionForcing(nothing, nothing,
        (entu = ws.derived_entu, detu = ws.derived_detu,
         entd = ws.zero_entd,    detd = ws.zero_detd))
end

# ---- LatLon ----

function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:LatLonMesh},
                op::CMFMCMatrixConvection,
                dt::Real;
                workspace::CMFMCMatrixWorkspace) where {B, A, Raw <: AbstractArray{<:Any, 4}, Names}
    _assert_cmfmc_matrix_forcing(forcing)
    _refresh_cmfmc_matrix_rates!(workspace, forcing)
    synth = _cmfmc_matrix_synthetic_forcing(workspace)
    return apply!(state, synth, grid, op.inner, dt;
                  workspace = workspace.tm5_workspace)
end

# ---- ReducedGaussian ----

function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:ReducedGaussianMesh},
                op::CMFMCMatrixConvection,
                dt::Real;
                workspace::CMFMCMatrixWorkspace) where {B, A, Raw <: AbstractArray{<:Any, 3}, Names}
    _assert_cmfmc_matrix_forcing(forcing)
    _refresh_cmfmc_matrix_rates!(workspace, forcing)
    synth = _cmfmc_matrix_synthetic_forcing(workspace)
    return apply!(state, synth, grid, op.inner, dt;
                  workspace = workspace.tm5_workspace)
end

# ---- CubedSphere ----

function apply!(state::CubedSphereState{B},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:CubedSphereMesh},
                op::CMFMCMatrixConvection,
                dt::Real;
                workspace::CMFMCMatrixWorkspace) where {B}
    _assert_cmfmc_matrix_forcing(forcing)
    _refresh_cmfmc_matrix_rates!(workspace, forcing)
    synth = _cmfmc_matrix_synthetic_forcing(workspace)
    return apply!(state, synth, grid, op.inner, dt;
                  workspace = workspace.tm5_workspace)
end
