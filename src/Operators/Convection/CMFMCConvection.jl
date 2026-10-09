# ---------------------------------------------------------------------------
# CMFMCConvection — GCHP-audited two-pass CMFMC convective transport.
# Basis-polymorphic, with well-mixed sub-cloud treatment, mandatory CFL
# sub-cycling, and no positivity clamp to preserve adjoint structure.
#
# Intentional divergences from line-by-line GCHP RAS (`convection_mod.F90`):
#   1. Cloud-base detection uses the lowest non-zero `cmfmc[k+1]` rather
#      than `DQRCU(K) > 0` (GCHP RAS proxy). Our binary contract delivers
#      cmfmc + dtrain only — no DQRCU — so we proxy from the available
#      mass flux. Same physical intent (lowest level with updraft inflow),
#      different signal source. See cmfmc_kernels.jl:493.
#   2. Pass-0 adds a local cloud-base mass-closing tracer update for
#      column self-consistency (kg/m²-weighted well-mix). GCHP RAS does
#      not have this step in the column kernel; it relies on the
#      dynamics-core PBL air-mass rescale to absorb residual imbalance.
#      In standalone offline transport this rescale is absent, so the
#      added closure exists to keep our env-only column update truthful.
#      See cmfmc_kernels.jl:559.
#
# These divergences are documented in the audit memo
# `convection_reference_audit_2026_05_24.md`. The label is
# "GCHP-audited" (matches the high-level operator design + ~95% of the
# kernel mechanics) rather than "GCHP-faithful" (line-by-line parity).
# ---------------------------------------------------------------------------

"""
    CMFMCConvection()

GCHP-audited CMFMC convective transport operator (env + qc two-pass).
NOT line-by-line GCHP RAS — see the module-top comment for the two
intentional divergences (cloud-base proxy + Pass-0 closure).

No struct fields — the forcing arrays (`cmfmc`, optionally `dtrain`)
arrive via `TransportModel.convection_forcing`, populated each substep by
`DrivenSimulation._refresh_forcing!` from `sim.window.convection`.

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
base (`convection_mod.F90:742-782`). Absent in legacy Julia;
deliberate improvement for surface-source tracers. See the kernel
Pass-0 block in `cmfmc_kernels.jl`.

# Scope

`CMFMCConvection` now supports structured LatLon, face-indexed
ReducedGaussian, and panel-native CubedSphere runtime state. The CS
path keeps forcing panel-native too: `forcing.cmfmc` and
`forcing.dtrain` are `NTuple{6}` payloads loaded by the CS transport
driver and applied column-locally on the halo-free panel interior.

# Adjoint path

With `clamp = false`, the forward operator is linear in tracer mixing ratio
and its transposed two-pass kernel ships in `Adjoints/ConvectionAdjoint.jl`.
Adjoint-identity and end-to-end footprint finite-difference tests cover this
default path. `clamp = true` is nonlinear; the footprint API rejects that
variant until its branch decisions are taped and transposed. The four-term
scavenging-restoring form remains a wet-deposition follow-up.
"""
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
    CMFMCConvection(; clamp=false, cloud_base=CMFMCEdgeCloudBase())

Explicit, conservative convection driven by per-layer convective mass flux and
detrainment from ConvectionForcing. Supported on lat-lon, reduced-Gaussian,
and cubed-sphere states. Setting clamp=true applies the positivity correction
and conservative column rescaling used for strong-CFL forcing. `cloud_base`
selects the cloud-base rule (see [`AbstractCloudBase`](@ref)).
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

    for _ in 1:n_sub
        kernel(q_raw, air_mass, cmfmc, dtrain_arr, cell_areas_y,
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

    for _ in 1:n_sub
        kernel(q_raw, air_mass, cmfmc, dtrain_arr, cell_areas,
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
    kernel = _cmfmc_cs_panel_column_kernel!(backend, (16, 16))

    for _ in 1:n_sub
        for p in 1:6
            kernel(q_raw[p], air_mass[p], cmfmc[p], dtrain_arr[p], cell_areas[p],
                   _panel_cloud_base(op.cloud_base, forcing, p),
                   workspace.qc_scratch[p],
                   Nz, Nt, sdt, Hp, Val(has_dtrain_val), Val(op.clamp);
                   ndrange = (Nc, Nc))
        end
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
