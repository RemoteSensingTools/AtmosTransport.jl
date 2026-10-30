# CS convection adjoint validation and tape dispatch. Numerical kernels live
# beside their forward operators in Operators/Convection/{CMFMC,TM5}.

# The CMFMC forward (replayed below) and its adjoint use the production
# operator's definitions: `_cmfmc_tiny`, the cloud base derived from CMFMC
# (`_cmfmc_cloud_base(nothing, …)`) and the detrainment array of
# `_cmfmc_dtrain_array` (DTRAIN, or max(0, CMFMC[k+1] − CMFMC[k]) without it).

function _require_cmfmc_convection_workspace(workspace)
    workspace isa CMFMCWorkspace || throw(ArgumentError(
        "CS CMFMC adjoint convection requires a `CMFMCWorkspace`; got $(typeof(workspace))"))
    workspace.cell_metrics === nothing && throw(ArgumentError(
        "CS CMFMC adjoint convection requires `workspace.cell_metrics` with per-panel cell areas"))
    return workspace
end

function _assert_cmfmc_adjoint_forcing(forcing)
    forcing isa ConvectionForcing || throw(ArgumentError(
        "CS CMFMC adjoint convection requires a `ConvectionForcing`; got $(typeof(forcing))"))
    forcing.cmfmc === nothing && throw(ArgumentError(
        "CS CMFMC adjoint convection requires `forcing.cmfmc` panel fields"))
    return forcing.cmfmc, forcing.dtrain
end

function _require_tm5_convection_workspace(workspace)
    workspace isa TM5Workspace || throw(ArgumentError(
        "CS TM5 adjoint convection requires a `TM5Workspace`; got $(typeof(workspace))"))
    workspace.cell_metrics === nothing && throw(ArgumentError(
        "CS TM5 adjoint convection requires `workspace.cell_metrics` with per-panel cell areas"))
    return workspace
end

function _assert_tm5_adjoint_forcing(forcing)
    forcing isa ConvectionForcing || throw(ArgumentError(
        "CS TM5 adjoint convection requires a `ConvectionForcing`; got $(typeof(forcing))"))
    forcing.tm5_fields === nothing && throw(ArgumentError(
        "CS TM5 adjoint convection requires `forcing.tm5_fields` (:entu, :detu, :entd, :detd)"))
    return forcing.tm5_fields
end

_require_supported_cs_convection_variant(::NoConvection) = nothing

function _require_supported_cs_convection_variant(op::CMFMCConvection)
    op.clamp && throw(ArgumentError(
        "CS adjoint footprints currently require `CMFMCConvection(clamp = false)`: " *
        "the positivity clamp is nonlinear and its branch decisions are not taped."))
    _require_edge_cloud_base(op.cloud_base, "CS adjoint footprints (the adjoint kernel derives the cloud base from CMFMC)")
    return nothing
end

function _require_supported_cs_convection_variant(op::TM5Convection)
    if op.use_collab_lu || op.lmax_conv != 0 || op.n_merge != 1
        throw(ArgumentError(
            "CS adjoint footprints currently support only the full-column, unmerged " *
            "TM5 solve (`use_collab_lu = false`, `lmax_conv = 0`, `n_merge = 1`); " *
            "got use_collab_lu=$(op.use_collab_lu), lmax_conv=$(op.lmax_conv), " *
            "n_merge=$(op.n_merge)."))
    end
    return nothing
end

_require_supported_cs_convection_variant(op::CMFMCMatrixConvection) =
    _require_supported_cs_convection_variant(op.inner)

_require_cs_convection_workspace(::NoConvection, workspace) = nothing
function _require_cs_convection_workspace(op::CMFMCConvection, workspace)
    _require_supported_cs_convection_variant(op)
    return _require_cmfmc_convection_workspace(workspace)
end
function _require_cs_convection_workspace(op::TM5Convection, workspace)
    _require_supported_cs_convection_variant(op)
    return _require_tm5_convection_workspace(workspace)
end
function _require_cs_convection_workspace(op::CMFMCMatrixConvection, workspace)
    _require_supported_cs_convection_variant(op)
    workspace isa CMFMCMatrixWorkspace || throw(ArgumentError(
        "CS CMFMCMatrix adjoint convection requires a `CMFMCMatrixWorkspace`; got $(typeof(workspace))"))
    return _require_tm5_convection_workspace(workspace.tm5_workspace)
end
function _require_cs_convection_workspace(op, workspace)
    throw(ArgumentError("CS adjoint footprint supports `NoConvection`, " *
                        "`CMFMCConvection`, `TM5Convection`, and `CMFMCMatrixConvection`; " *
                        "got $(typeof(op))"))
end

# CMFMCMatrixConvection adjoint forcing must look like the forward forcing:
# cmfmc + dtrain populated. We derive (entu, detu) into the workspace cache
# and build a synthetic TM5-form forcing that the TM5 forward / adjoint
# kernels consume unchanged.
function _assert_cmfmc_matrix_adjoint_forcing(forcing)
    forcing isa ConvectionForcing || throw(ArgumentError(
        "CS CMFMCMatrix adjoint convection requires a `ConvectionForcing`; got $(typeof(forcing))"))
    forcing.cmfmc === nothing && throw(ArgumentError(
        "CS CMFMCMatrix adjoint convection requires `forcing.cmfmc` panel fields"))
    forcing.dtrain === nothing && throw(ArgumentError(
        "CS CMFMCMatrix adjoint convection requires `forcing.dtrain` panel fields"))
    return forcing.cmfmc, forcing.dtrain
end

@inline function _refresh_and_synthesize_cmfmc_matrix_forcing!(
        workspace::CMFMCMatrixWorkspace, forcing::ConvectionForcing)
    _assert_cmfmc_matrix_adjoint_forcing(forcing)
    invalidate_cmfmc_matrix_cache!(workspace)
    _refresh_cmfmc_matrix_rates!(workspace, forcing)
    return _cmfmc_matrix_synthetic_forcing(workspace)
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       ::NoConvection, dt, workspace,
                                       mesh::CubedSphereMesh; n_sub = 1)
    return nothing
end

# `n_sub` is the CFL sub-step count of the forward call, which production
# caches per met window (from the air mass of the window's first call): the
# tape records it, and the reverse transposes the same sub-steps. Without it
# (`nothing`, the per-operator identity tests) the count is computed from
# `panels_m` without touching the workspace cache.
function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       op::CMFMCConvection, dt,
                                       workspace::CMFMCWorkspace,
                                       mesh::CubedSphereMesh; n_sub = nothing)
    _require_supported_cs_convection_variant(op)
    cmfmc, dtrain = _assert_cmfmc_adjoint_forcing(forcing)
    _require_cmfmc_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    if n_sub === nothing
        snap = operator_state(workspace)
        invalidate_cmfmc_cache!(workspace)
        n_sub = _get_or_compute_n_sub!(workspace, cmfmc, panels_m, cell_areas, dt)
        restore_operator_state!(workspace, snap)
    end
    dtrain_arr = _cmfmc_dtrain_array(cmfmc, dtrain, panels_m)
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(lambda_panels[1], 3)
    backend = get_backend(lambda_panels[1])
    kernel! = _cmfmc_cs_panel_column_single_adjoint_kernel!(backend, (16, 16))
    FT = eltype(lambda_panels[1])
    sdt = FT(dt) / FT(n_sub)
    @inbounds for _ in 1:n_sub
        for p in 1:6
            kernel!(lambda_panels[p], panels_m[p], cmfmc[p], dtrain_arr[p],
                    cell_areas[p], workspace.qc_scratch[p],
                    Nz, sdt, Hp;
                    ndrange = (Nc, Nc))
        end
    end
    return nothing
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       op::TM5Convection, dt,
                                       workspace::TM5Workspace,
                                       mesh::CubedSphereMesh; n_sub = 1)
    _require_supported_cs_convection_variant(op)
    tm5 = _assert_tm5_adjoint_forcing(forcing)
    _require_tm5_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    Nc = mesh.Nc
    Hp = mesh.Hp
    N_total = Nc * Nc
    _ensure_tm5_scratch!(workspace)
    B = size(workspace.conv1, 3)
    backend = get_backend(lambda_panels[1])
    kernel! = _tm5_cs_panel_column_adjoint_kernel!(backend)
    FT = eltype(lambda_panels[1])
    @inbounds for p in 1:6
        for tile_off in 0:B:(N_total - 1)
            n = min(B, N_total - tile_off)
            kernel!(lambda_panels[p], panels_m[p],
                    tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p],
                    cell_areas[p],
                    workspace.conv1, workspace.pivots, workspace.cloud_dims,
                    workspace.f_scratch,
                    workspace.amu_scratch, workspace.amd_scratch,
                    Hp, Int(tile_off), Nc, FT(dt);
                    ndrange = n)
        end
    end
    return nothing
end

# CMFMCMatrixConvection adjoint: same path as forward — derive (entu, detu)
# from the forcing, then delegate to the TM5 adjoint kernel. The state-space
# operator is purely the TM5 LU (the derivation is independent of the state),
# so the adjoint-identity ⟨y, L·x⟩ = ⟨Lᵀ·y, x⟩ inherits from the TM5 LU
# adjoint with no chain-rule contribution from the derivation step.
function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       op::CMFMCMatrixConvection, dt,
                                       workspace::CMFMCMatrixWorkspace,
                                       mesh::CubedSphereMesh; n_sub = 1)
    _require_supported_cs_convection_variant(op)
    synth = _refresh_and_synthesize_cmfmc_matrix_forcing!(workspace, forcing)
    return _apply_cs_convection_adjoint!(lambda_panels, panels_m, synth,
                                          op.inner, dt, workspace.tm5_workspace, mesh)
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       op, dt, workspace,
                                       mesh::CubedSphereMesh; n_sub = 1)
    throw(ArgumentError("CS adjoint footprint reverse helper supports `NoConvection`, " *
                        "`CMFMCConvection`, `TM5Convection`, and `CMFMCMatrixConvection`; " *
                        "got $(typeof(op))"))
end
