# ---------------------------------------------------------------------------
# CMFMCConvection — GCHP-audited two-pass CMFMC convective transport.
# Basis-polymorphic, with well-mixed sub-cloud treatment, mandatory CFL
# sub-cycling, and no positivity clamp to preserve adjoint structure.
#
# Intentional divergences from line-by-line GCHP RAS (`convection_mod.F90`):
#   1. By default the cloud base is the lowest layer with non-zero
#      `cmfmc[k+1]` (`CMFMCEdgeCloudBase`) rather than `DQRCU(K) > 0`; the
#      DQRCU base is available on the cubed sphere (`ArchivedCloudBase`)
#      when the binary carries it.
#   2. No `Q + DELQ < 0` clip (by default; see `clamp`), and the model-top
#      layer is updated (GEOS-Chem stops at KTOP = NLAY - 1).
#
# The sub-cloud mixing, the updraft and the level update follow GEOS-Chem's
# order: the updraft entrains the cloud-base layer as it was, and the layer
# takes the sub-cloud exchange through its bottom edge in the level update
# (`T1 + T4`). GEOS-Chem's routine conserves column tracer mass apart from
# its clip, and so does this one. Equations: "The RAS column update" in
# docs/src/for_tm5_gchp_users/operators_on_binaries.md; a literal port of the
# GEOS-Chem routine is the reference of test/core/test_cmfmc_ras_cloud_base.jl.
# ---------------------------------------------------------------------------


# `clamp = true` opts into the GCHP positivity clamp (Q+DELQ<0 → DELQ=−Q) made
# conservative by a post-update per-column rescale. The clamp absorbs the CFL
# overshoot of strong convection so the scheme stays stable at FEW sub-steps (the
# explicit flux-divergence form alone is CFL-limited and would need thousands);
# the rescale restores the pre-convection column tracer mass exactly, so the
# scheme remains conservative. The default `clamp = false` is the pure
# (unclamped) conservative explicit scheme — exactly mass-conserving but
# CFL-substep-limited.
"""
    CMFMCEdgeCloudBase()
    ArchivedCloudBase()

Where `CMFMCConvection` puts the convective cloud base, below which the
column is mixed before the updraft starts. `CMFMCEdgeCloudBase` (default)
takes the lowest layer with updraft inflow through its bottom edge.
`ArchivedCloudBase` takes it from the met forcing (`forcing.cloud_base`):
GEOS-Chem's lowest layer with convective rain production DQRCU > 0
(`convection_mod.F90`, `DO_RAS_CLOUD_CONVECTION`). Cubed sphere only.
"""
abstract type AbstractCloudBase end
struct CMFMCEdgeCloudBase <: AbstractCloudBase end
struct ArchivedCloudBase <: AbstractCloudBase end

"""
    CMFMCConvection()

CMFMC convective transport in the order of GEOS-Chem's RAS
(`DO_RAS_CLOUD_CONVECTION`): sub-cloud mixing, updraft, flux-form level
update. See the module-top comment for the differences from GEOS-Chem
(default cloud base, no positivity clip).

No struct fields — the forcing arrays (`cmfmc`, optionally `dtrain`)
arrive via `TransportModel.convection_forcing`, populated at the first substep
of each window by `DrivenSimulation._refresh_forcing!` from `sim.window.convection`.

# Basis convention

The operator is basis-polymorphic. `cmfmc` and `dtrain` must be on the
SAME basis as `state.air_mass` (dry in
CATRINE usage, moist if the driver supplies moist forcing). The
driver is responsible for basis correction upstream.

# Fields required on `ConvectionForcing`

- `forcing.cmfmc :: AbstractArray{FT, 3}` at interfaces, shape
  `(Nx, Ny, Nz+1)`. Units kg / m² / s on the state's basis.
- `forcing.dtrain :: Union{AbstractArray{FT, 3}, Nothing}` at centers,
  shape `(Nx, Ny, Nz)`. When `nothing`, the kernel runs
  Tiedtke-style single-flux transport.

Face-indexed ReducedGaussian uses `(ncell, Nz+1)` / `(ncell, Nz)`.
CubedSphere uses `NTuple{6}` of per-panel `(Nc, Nc, Nz+1)` /
`(Nc, Nc, Nz)` arrays.

# CFL sub-cycling

The kernel sub-cycles internally based on the CMFMC profile:

    n_sub = max(1, ceil(max_over_grid(cmfmc × dt / bmass) / 0.5))

`n_sub` is cached per window in `workspace.cached_n_sub[]`; the
driver clears the cache on window roll via
`invalidate_cmfmc_cache!(workspace)`. Bit-exact: one call with `dt`
matches `n_sub` manual calls with `sdt = dt / n_sub`.

# Well-mixed sub-cloud layer

Applies GCHP's pressure-weighted well-mixed treatment below cloud
base (`convection_mod.F90:738-782`). As in GEOS-Chem the updraft then
entrains the cloud-base layer as it was, and the layer takes the
sub-cloud exchange through its bottom edge in the level update: the
column is conserved, and the inflow is not counted twice in the
cloud-base layer (which could drive it negative). See the kernel
Pass-0 block in `cmfmc_kernels.jl`.

# Column mass ledger

The update conserves column tracer mass by construction, but each
layer's new mass is rounded when it is written. A
compensated column ledger returns that rounding to the column's
largest layer and keeps the sub-ulp remainder as a per-column,
per-tracer carry in `CMFMCWorkspace` (as for TM5 convection; see the
single-precision page of the manual). Columns with a flux through the
model top or a layer without air, and `clamp = true`, are not ledgered.

# Scope

`CMFMCConvection` now supports structured LatLon, face-indexed
ReducedGaussian, and panel-native CubedSphere runtime state. The CS
path keeps forcing panel-native too: `forcing.cmfmc` and
`forcing.dtrain` are `NTuple{6}` payloads loaded by the CS transport
driver and applied column-locally on the halo-free panel interior.

# Adjoint path

With `clamp = false`, the forward operator is linear in tracer mixing ratio
and its transposed two-pass kernel ships in `adjoint_kernels.jl`.
Adjoint-identity and end-to-end footprint finite-difference tests cover this
default path. `clamp = true` is nonlinear; the footprint API rejects that
variant until its branch decisions are taped and transposed. The four-term
scavenging-restoring form remains a wet-deposition follow-up.
"""
struct CMFMCConvection{CB <: AbstractCloudBase} <: AbstractConvection
    clamp      :: Bool
    cloud_base :: CB
end
CMFMCConvection(; clamp::Bool = false, cloud_base::AbstractCloudBase = CMFMCEdgeCloudBase()) =
    CMFMCConvection(clamp, cloud_base)

# Paths that only implement the CMFMC-edge cloud base.
_require_edge_cloud_base(::CMFMCEdgeCloudBase, _where) = nothing
_require_edge_cloud_base(::AbstractCloudBase, where) = throw(ArgumentError(
    "CMFMCConvection with an archived cloud base is not implemented for $(where)."))

# Per-panel cloud-base array handed to the kernel (`nothing` = derive from CMFMC).
_panel_cloud_base(::CMFMCEdgeCloudBase, _forcing, _p) = nothing
function _panel_cloud_base(::ArchivedCloudBase, forcing, p)
    forcing.cloud_base === nothing && throw(ArgumentError(
        "CMFMCConvection(cloud_base = ArchivedCloudBase()) requires `forcing.cloud_base`; " *
        "regenerate the binary with include_convective_cloud_base = true."))
    return forcing.cloud_base[p]
end

# =========================================================================
# Array-level entry: apply_convection!
# =========================================================================

"""
    apply_convection!(q_raw, air_mass, forcing::ConvectionForcing,
                       op::CMFMCConvection, dt, workspace::CMFMCWorkspace,
                       grid::AtmosGrid) -> nothing

Apply one transport step of CMFMC + optional DTRAIN convection on
the structured lat-lon grid. Operates on `q_raw` and `air_mass` in
place; `workspace.qc_scratch` is used as per-column updraft storage
and `workspace.cached_n_sub` / `workspace.cache_valid` manage the
CFL cache.

`dt` is the block-level step length; the kernel internally
sub-cycles `n_sub` times with `sdt = dt / n_sub`.
"""
function _cmfmc_dtrain_array(cmfmc::AbstractArray{FT, 3},
                             dtrain::Nothing,
                             air_mass::AbstractArray{FT, 3}) where FT
    Nx_c, Ny_c, Nz_c = size(air_mass)
    darr = similar(cmfmc, FT, Nx_c, Ny_c, Nz_c)
    @views darr .= max.(zero(FT), cmfmc[:, :, 2:Nz_c + 1] .- cmfmc[:, :, 1:Nz_c])
    return darr
end

_cmfmc_dtrain_array(cmfmc, dtrain, air_mass) = dtrain

function _cmfmc_dtrain_array(cmfmc::NTuple{6, <:AbstractArray{FT, 3}},
                             dtrain::Nothing,
                             air_mass::NTuple{6, <:AbstractArray{FT, 3}}) where FT
    return ntuple(6) do p
        Nx_c, Ny_c, Nzp1 = size(cmfmc[p])
        Nz_c = Nzp1 - 1
        darr = similar(cmfmc[p], FT, Nx_c, Ny_c, Nz_c)
        @views darr .= max.(zero(FT), cmfmc[p][:, :, 2:Nzp1] .- cmfmc[p][:, :, 1:Nz_c])
        darr
    end
end

function apply_convection!(q_raw::AbstractArray{FT, 4},
                            air_mass::AbstractArray{FT, 3},
                            forcing::ConvectionForcing,
                            op::CMFMCConvection,
                            dt,
                            workspace::CMFMCWorkspace,
                            grid::AtmosGrid{<:LatLonMesh}) where FT
    forcing.cmfmc === nothing && throw(ArgumentError(
        "CMFMCConvection requires `forcing.cmfmc` to be populated; got nothing. " *
        "Install via `TransportModel.convection_forcing` or " *
        "`with_convection_forcing(model, ConvectionForcing(cmfmc, dtrain, nothing))`."))
    op.clamp && throw(ArgumentError(
        "CMFMCConvection(clamp=true) is currently implemented for cubed-sphere only; " *
        "the lat-lon column kernel does not yet apply the clamp + column rescale."))
    _require_edge_cloud_base(op.cloud_base, "lat-lon grids")

    cmfmc = forcing.cmfmc

    # DTRAIN-missing Tiedtke-style fallback:
    # when `forcing.dtrain === nothing`, derive per-layer detrainment
    # from the updraft-flux divergence:
    #
    #     dtrain[k] = max(0, cmfmc[k+1] - cmfmc[k])
    #
    # (cmfmc[k+1] = flux at the BOTTOM of layer k going up;
    #  cmfmc[k]   = flux at the TOP of layer k going up.
    # A decreasing cmfmc as k decreases — i.e. flux goes up through
    # the column and diminishes — implies the updraft is shedding
    # mass into the environment at that layer. Without an explicit
    # DTRAIN field, this diagnosed detrainment closes the mass
    # balance. Matches the legacy lat-lon fallback behavior of the
    # earlier Julia port (git commit ec2d2c0,
    # `src_legacy/Convection/ras_convection.jl:365-370`), which
    # delegated to a single-flux Tiedtke operator that made the same
    # diagnosis implicitly.)
    dtrain_arr = _cmfmc_dtrain_array(cmfmc, forcing.dtrain, air_mass)
    # Always true at the kernel level once we've closed the mass
    # balance via derived or provided dtrain.
    has_dtrain_val = true

    Nx, Ny, Nz, Nt = size(q_raw)
    cell_areas_y = workspace.cell_metrics === nothing ?
                   cell_areas_by_latitude(grid.horizontal) :
                   workspace.cell_metrics

    # Cache the CFL sub-step count per met window.
    n_sub = _get_or_compute_n_sub!(workspace, cmfmc, air_mass,
                                    cell_areas_y, dt)
    sdt = FT(dt) / FT(n_sub)

    backend = get_backend(q_raw)
    kernel = _cmfmc_column_kernel!(backend, (16, 16))
    carry = _cmfmc_ledger_carry!(workspace, (Nx, Ny, Nt))

    for _ in 1:n_sub
        kernel(q_raw, carry, air_mass, cmfmc, dtrain_arr, cell_areas_y,
               workspace.qc_scratch,
               Nz, Nt, sdt, Val(has_dtrain_val);
               ndrange = (Nx, Ny))
    end
    synchronize(backend)
    return nothing
end

function apply_convection!(q_raw::AbstractArray{FT, 3},
                            air_mass::AbstractArray{FT, 2},
                            forcing::ConvectionForcing,
                            op::CMFMCConvection,
                            dt,
                            workspace::CMFMCWorkspace,
                            grid::AtmosGrid{<:ReducedGaussianMesh}) where FT
    op.clamp && throw(ArgumentError(
        "CMFMCConvection(clamp=true) is currently implemented for cubed-sphere only; " *
        "the reduced-Gaussian column kernel does not yet apply the clamp + column rescale."))
    _require_edge_cloud_base(op.cloud_base, "reduced-Gaussian grids")
    forcing.cmfmc === nothing && throw(ArgumentError(
        "CMFMCConvection requires `forcing.cmfmc` to be populated; got nothing. " *
        "Install via `TransportModel.convection_forcing` or " *
        "`with_convection_forcing(model, ConvectionForcing(cmfmc, dtrain, nothing))`."))

    cell_areas = workspace.cell_metrics
    cell_areas === nothing && throw(ArgumentError(
        "Face-indexed CMFMCConvection requires `workspace.cell_metrics` " *
        "to carry per-cell areas. Build the model with `with_convection` " *
        "or construct `CMFMCWorkspace(air_mass; cell_metrics=...)`."))

    cmfmc = forcing.cmfmc
    dtrain_arr = _cmfmc_dtrain_array(cmfmc, forcing.dtrain, air_mass)
    has_dtrain_val = true

    ncell, Nz, Nt = size(q_raw)
    n_sub = _get_or_compute_n_sub!(workspace, cmfmc, air_mass,
                                   cell_areas, dt)
    sdt = FT(dt) / FT(n_sub)

    backend = get_backend(q_raw)
    kernel = _cmfmc_faceindexed_column_kernel!(backend, 256)
    carry = _cmfmc_ledger_carry!(workspace, (ncell, Nt))

    for _ in 1:n_sub
        kernel(q_raw, carry, air_mass, cmfmc, dtrain_arr, cell_areas,
               workspace.qc_scratch,
               Nz, Nt, sdt, Val(has_dtrain_val);
               ndrange = ncell)
    end
    synchronize(backend)
    return nothing
end

function apply_convection!(q_raw::AbstractArray{FT, 3},
                            air_mass::AbstractArray{FT, 2},
                            ::ConvectionForcing,
                            ::CMFMCConvection,
                            dt, workspace, grid::AtmosGrid) where FT
    throw(ArgumentError(
        "`CMFMCConvection` supports face-indexed state only on " *
        "`ReducedGaussianMesh`; got $(typeof(grid.horizontal))."))
end

function apply_convection!(q_raw::NTuple{6, <:AbstractArray{FT, 4}},
                           air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                           forcing::ConvectionForcing,
                           op::CMFMCConvection,
                           dt,
                           workspace::CMFMCWorkspace,
                           grid::AtmosGrid{<:CubedSphereMesh}) where FT
    forcing.cmfmc === nothing && throw(ArgumentError(
        "CMFMCConvection requires `forcing.cmfmc` to be populated; got nothing. " *
        "Install via `TransportModel.convection_forcing` or " *
        "`with_convection_forcing(model, ConvectionForcing(cmfmc, dtrain, nothing))`."))

    cell_areas = workspace.cell_metrics
    cell_areas === nothing && throw(ArgumentError(
        "Cubed-sphere CMFMCConvection requires `workspace.cell_metrics` " *
        "to carry per-panel cell-area matrices. Build the model with " *
        "`with_convection` or construct `CMFMCWorkspace(air_mass; cell_metrics=...)`."))

    cmfmc = forcing.cmfmc
    dtrain_arr = _cmfmc_dtrain_array(cmfmc, forcing.dtrain, air_mass)
    has_dtrain_val = true

    mesh = grid.horizontal
    Hp = mesh.Hp
    Nc = mesh.Nc
    Nz = size(q_raw[1], 3)
    Nt = size(q_raw[1], 4)
    n_sub = _get_or_compute_n_sub!(workspace, cmfmc, air_mass,
                                   cell_areas, dt; allow_clamp = op.clamp)
    sdt = FT(dt) / FT(n_sub)

    backend = get_backend(q_raw[1])
    # One launch per substep covers the six panels (the third index selects
    # the panel); each column runs as in a one-panel launch.
    kernel = _cmfmc_cs_panels_column_kernel!(backend, (16, 16))
    # The clamped variant is not ledgered and keeps no carry.
    carry = op.clamp ? nothing : _cmfmc_ledger_carry!(workspace, (Nc, Nc, Nt))
    carry_panels = ntuple(p -> carry === nothing ? nothing : carry[p], 6)
    cloud_base = ntuple(p -> _panel_cloud_base(op.cloud_base, forcing, p), 6)

    for _ in 1:n_sub
        kernel(q_raw, carry_panels, air_mass, cmfmc, dtrain_arr, cell_areas, cloud_base,
               workspace.qc_scratch, Nz, Nt, sdt, Hp, Val(has_dtrain_val), Val(op.clamp);
               ndrange = (Nc, Nc, 6))
    end
    synchronize(backend)
    return nothing
end

# =========================================================================
# State-level entry: apply!
# =========================================================================

"""
    apply!(state::CellState, forcing::ConvectionForcing, grid::AtmosGrid,
           op::CMFMCConvection, dt::Real; workspace) -> state

State-level delegate. Unpacks `state.tracers_raw` and
`state.air_mass`, then forwards to `apply_convection!`. Requires an
allocated `CMFMCWorkspace` — `DrivenSimulation` construction
allocates this (caller-owned pre-allocation).
"""
function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:LatLonMesh},
                op::CMFMCConvection,
                dt::Real;
                workspace::CMFMCWorkspace) where {B, A, Raw <: AbstractArray{<:Any, 4}, Names}
    apply_convection!(state.tracers_raw, state.air_mass, forcing, op, dt,
                       workspace, grid)
    return state
end

function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:ReducedGaussianMesh},
                op::CMFMCConvection,
                dt::Real;
                workspace::CMFMCWorkspace) where {B, A, Raw <: AbstractArray{<:Any, 3}, Names}
    apply_convection!(state.tracers_raw, state.air_mass, forcing, op, dt,
                      workspace, grid)
    return state
end

function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing, grid::AtmosGrid,
                op::CMFMCConvection, dt::Real;
                workspace = nothing) where {B, A, Raw <: AbstractArray{<:Any, 3}, Names}
    throw(ArgumentError(
        "`CMFMCConvection` supports face-indexed state only on " *
        "`ReducedGaussianMesh`; got $(typeof(grid.horizontal))."))
end

function apply!(state::CubedSphereState{B},
                forcing::ConvectionForcing,
                grid::AtmosGrid,
                op::CMFMCConvection,
                dt::Real;
                workspace = nothing) where {B}
    throw(ArgumentError(
        "`CMFMCConvection` supports `CubedSphereState` only on " *
        "`CubedSphereMesh`; got $(typeof(grid.horizontal))."))
end

function apply!(state::CubedSphereState{B},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:CubedSphereMesh},
                op::CMFMCConvection,
                dt::Real;
                workspace::CMFMCWorkspace) where {B}
    apply_convection!(state.tracers_raw, state.air_mass, forcing, op, dt,
                      workspace, grid)
    return state
end
