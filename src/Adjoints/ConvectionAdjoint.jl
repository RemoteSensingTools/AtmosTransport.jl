# ---------------------------------------------------------------------------
# Adjoint of CS convection (CMFMC + TM5).
#
# Reverse-mode of `CMFMCConvection` and `TM5Convection` column operators.
# Contains:
#   * TM5 forward Thomas/LU solves (`_tm5_solve_vector!`,
#     `_tm5_solve_column_vector!`) used by the reverse-mode forward
#     replay path.
#   * TM5 transpose solves (`_tm5_solve_vector_transpose!`,
#     `_tm5_solve_column_vector_adjoint!`).
#   * Per-panel column kernels for both TM5 and CMFMC (forward + adjoint).
#   * Workspace + forcing validation helpers.
#   * `_apply_cs_convection_forward!` and `_apply_cs_convection_adjoint!`
#     dispatch arms for `NoConvection` / `CMFMCConvection` / `TM5Convection`.
# ---------------------------------------------------------------------------

function _tm5_solve_vector!(rm_col, conv1, pivots, Nz::Integer;
                            icltop_eff::Integer = 1)
    Nz == 0 && return nothing
    k_lo = max(Int(icltop_eff), 1)
    @inbounds begin
        for k in k_lo:Nz
            piv = pivots[k]
            if piv != k
                tmp = rm_col[k]
                rm_col[k] = rm_col[piv]
                rm_col[piv] = tmp
            end
        end
        for k in k_lo:Nz
            s = rm_col[k]
            for j in k_lo:(k - 1)
                s -= conv1[k, j] * rm_col[j]
            end
            rm_col[k] = s
        end
        for k in Nz:-1:k_lo
            s = rm_col[k]
            for j in (k + 1):Nz
                s -= conv1[k, j] * rm_col[j]
            end
            rm_col[k] = s / conv1[k, k]
        end
    end
    return nothing
end

function _tm5_solve_vector_transpose!(lambda_col, conv1, pivots, Nz::Integer;
                                      icltop_eff::Integer = 1)
    Nz == 0 && return nothing
    k_lo = max(Int(icltop_eff), 1)
    @inbounds begin
        # U' z = lambda, where U is stored in the upper triangle.
        for k in k_lo:Nz
            s = lambda_col[k]
            for j in k_lo:(k - 1)
                s -= conv1[j, k] * lambda_col[j]
            end
            lambda_col[k] = s / conv1[k, k]
        end
        # L' y = z, where L is unit diagonal and stored below the diagonal.
        for k in Nz:-1:k_lo
            s = lambda_col[k]
            for j in (k + 1):Nz
                s -= conv1[j, k] * lambda_col[j]
            end
            lambda_col[k] = s
        end
        # Forward solve applies pivots in ascending order; the transpose
        # applies the inverse permutation, so replay swaps in reverse.
        for k in Nz:-1:k_lo
            piv = pivots[k]
            if piv != k
                tmp = lambda_col[k]
                lambda_col[k] = lambda_col[piv]
                lambda_col[piv] = tmp
            end
        end
    end
    return nothing
end

@inline function _tm5_effective_cloud_top(icltop, icllfs)
    return min(Int(icllfs), max(Int(icltop), 2) - 1)
end

# Unpivoted Hessenberg LU has bidiagonal L. U and the active-row convention
# remain unchanged; these helpers use the same ordering as the general solves.
function _tm5_solve_vector_bidiagonal!(q, A, Nz; icltop_eff=1)
    lo = max(Int(icltop_eff), 1)
    @inbounds begin
        for k in (lo+1):Nz
            q[k] -= A[k,k-1] * q[k-1]
        end
        for k in Nz:-1:lo
            value = q[k]
            for j in (k+1):Nz
                value -= A[k,j] * q[j]
            end
            q[k] = value / A[k,k]
        end
    end
    return nothing
end

function _tm5_solve_vector_transpose_bidiagonal!(q, A, Nz; icltop_eff=1)
    lo = max(Int(icltop_eff), 1)
    @inbounds begin
        for k in lo:Nz
            value = q[k]
            for j in lo:(k-1)
                value -= A[j,k] * q[j]
            end
            q[k] = value / A[k,k]
        end
        for k in (Nz-1):-1:lo
            q[k] -= A[k+1,k] * q[k+1]
        end
    end
    return nothing
end

function _tm5_solve_column_vector!(rm_col, m_col,
                                   entu_col, detu_col, entd_col, detd_col,
                                   conv1_buf, pivots_buf, cloud_dims, dt;
                                   cell_area = one(eltype(rm_col)),
                                   f_buf = conv1_buf,
                                   amu_buf,
                                   amd_buf)
    FT = eltype(rm_col)
    Nz = length(m_col)
    Nz == 0 && return nothing
    icltop, iclbas, icllfs = _tm5_diagnose_cloud_dims(detu_col, entd_col, Nz)
    cloud_dims[1] = icltop
    cloud_dims[2] = iclbas
    cloud_dims[3] = icllfs
    icltop > Nz && return nothing

    icltop_eff = _tm5_effective_cloud_top(icltop, icllfs)
    _tm5_build_conv1!(conv1_buf,
                      entu_col, detu_col, entd_col, detd_col, m_col,
                      icltop, icllfs, FT(dt), Nz;
                      cell_area = FT(cell_area),
                      f = f_buf, amu = amu_buf, amd = amd_buf)
    _tm5_factorize!(conv1_buf, pivots_buf, Nz, icllfs; icltop_eff = icltop_eff)
    # Same column mass ledger as the forward operator, so the replayed states
    # are the forward run's (the ledger is rounding-level; the transpose below
    # is the adjoint of the exact solve). A vector is a one-tracer column.
    lo = max(icltop_eff, 1)
    before = _tm5_column_sum(rm_col, lo, Nz, 1)
    if icllfs > Nz && _tm5_identity_pivots(pivots_buf, Nz, icltop_eff)
        _tm5_solve_vector_bidiagonal!(rm_col, conv1_buf, Nz; icltop_eff)
    else
        _tm5_solve_vector!(rm_col, conv1_buf, pivots_buf, Nz; icltop_eff)
    end
    _tm5_restore_column_mass!(rm_col, lo, Nz, 1, before)
    return nothing
end

function _tm5_solve_column_vector_adjoint!(lambda_col, m_col,
                                           entu_col, detu_col, entd_col, detd_col,
                                           conv1_buf, pivots_buf, cloud_dims, dt;
                                           cell_area = one(eltype(lambda_col)),
                                           f_buf = conv1_buf,
                                           amu_buf,
                                           amd_buf)
    FT = eltype(lambda_col)
    Nz = length(m_col)
    Nz == 0 && return nothing
    icltop, iclbas, icllfs = _tm5_diagnose_cloud_dims(detu_col, entd_col, Nz)
    cloud_dims[1] = icltop
    cloud_dims[2] = iclbas
    cloud_dims[3] = icllfs
    icltop > Nz && return nothing

    icltop_eff = _tm5_effective_cloud_top(icltop, icllfs)
    _tm5_build_conv1!(conv1_buf,
                      entu_col, detu_col, entd_col, detd_col, m_col,
                      icltop, icllfs, FT(dt), Nz;
                      cell_area = FT(cell_area),
                      f = f_buf, amu = amu_buf, amd = amd_buf)
    _tm5_factorize!(conv1_buf, pivots_buf, Nz, icllfs; icltop_eff = icltop_eff)
    if icllfs > Nz && _tm5_identity_pivots(pivots_buf, Nz, icltop_eff)
        _tm5_solve_vector_transpose_bidiagonal!(lambda_col, conv1_buf, Nz; icltop_eff)
    else
        _tm5_solve_vector_transpose!(lambda_col, conv1_buf, pivots_buf, Nz; icltop_eff)
    end
    return nothing
end

@kernel function _tm5_cs_panel_column_single_kernel!(
    q_raw_panel, @Const(air_mass_panel),
    @Const(entu_panel), @Const(detu_panel),
    @Const(entd_panel), @Const(detd_panel),
    @Const(cell_areas_panel),
    conv1_panel, pivots_panel, cloud_panel,
    f_panel, amu_panel, amd_panel,
    Hp::Int, tile_offset::Int, Nc::Int, dt)
    t = @index(Global)
    c_global = tile_offset + t
    c1 = ((c_global - 1) % Nc) + 1
    c2 = ((c_global - 1) ÷ Nc) + 1
    i = c1 + Hp
    j = c2 + Hp
    @inbounds begin
        rm_col = @view q_raw_panel[i, j, :]
        m_col = @view air_mass_panel[i, j, :]
        entu_col = @view entu_panel[c1, c2, :]
        detu_col = @view detu_panel[c1, c2, :]
        entd_col = @view entd_panel[c1, c2, :]
        detd_col = @view detd_panel[c1, c2, :]
        conv1_col = @view conv1_panel[:, :, t]
        pivots_col = @view pivots_panel[:, t]
        cloud_col = @view cloud_panel[:, t]
        f_col = @view f_panel[:, :, t]
        amu_col = @view amu_panel[:, t]
        amd_col = @view amd_panel[:, t]
        _tm5_solve_column_vector!(
            rm_col, m_col, entu_col, detu_col, entd_col, detd_col,
            conv1_col, pivots_col, cloud_col, dt;
            cell_area = cell_areas_panel[c1, c2],
            f_buf = f_col, amu_buf = amu_col, amd_buf = amd_col)
    end
end

@kernel function _tm5_cs_panel_column_adjoint_kernel!(
    lambda_panel, @Const(air_mass_panel),
    @Const(entu_panel), @Const(detu_panel),
    @Const(entd_panel), @Const(detd_panel),
    @Const(cell_areas_panel),
    conv1_panel, pivots_panel, cloud_panel,
    f_panel, amu_panel, amd_panel,
    Hp::Int, tile_offset::Int, Nc::Int, dt)
    t = @index(Global)
    c_global = tile_offset + t
    c1 = ((c_global - 1) % Nc) + 1
    c2 = ((c_global - 1) ÷ Nc) + 1
    i = c1 + Hp
    j = c2 + Hp
    @inbounds begin
        lambda_col = @view lambda_panel[i, j, :]
        m_col = @view air_mass_panel[i, j, :]
        entu_col = @view entu_panel[c1, c2, :]
        detu_col = @view detu_panel[c1, c2, :]
        entd_col = @view entd_panel[c1, c2, :]
        detd_col = @view detd_panel[c1, c2, :]
        conv1_col = @view conv1_panel[:, :, t]
        pivots_col = @view pivots_panel[:, t]
        cloud_col = @view cloud_panel[:, t]
        f_col = @view f_panel[:, :, t]
        amu_col = @view amu_panel[:, t]
        amd_col = @view amd_panel[:, t]
        _tm5_solve_column_vector_adjoint!(
            lambda_col, m_col, entu_col, detu_col, entd_col, detd_col,
            conv1_col, pivots_col, cloud_col, dt;
            cell_area = cell_areas_panel[c1, c2],
            f_buf = f_col, amu_buf = amu_col, amd_buf = amd_col)
    end
end

# The CMFMC forward (replayed below) and its adjoint use the production
# operator's definitions: `_cmfmc_tiny`, the cloud base derived from CMFMC
# (`_cmfmc_cloud_base(nothing, …)`) and the detrainment array of
# `_cmfmc_dtrain_array` (DTRAIN, or max(0, CMFMC[k+1] − CMFMC[k]) without it).

@kernel function _cmfmc_cs_panel_column_single_adjoint_kernel!(
    lambda_panel,
    @Const(air_mass_panel),
    @Const(cmfmc_panel),
    @Const(dtrain_panel),
    @Const(cell_areas_panel),
    lambda_qc_panel,
    Nz::Int,
    dt,
    Hp::Int)
    # Transpose of the production CMFMC forward operator (post-audit:
    # GG1 surface-up cloud base, CC1 kg/m² well-mix with cloud-base
    # closure, C3 entrn≥0 guard). Derivation: forward operator is
    # linear in `q = rm/m`, so the adjoint is rm-to-rm linear. Walks
    # Pass 2 → Pass 1 → Pass 0 in reverse order, accumulating gradient
    # contributions. See comments at each pass for the per-step
    # Jacobian terms. The lambda_qc_panel scratch carries λ_qc through
    # Pass 1 just as qc_scratch carries qc through the forward Pass 1.
    i, j = @index(Global, NTuple)
    FT = eltype(lambda_panel)
    tiny = _cmfmc_tiny(FT)
    ii = i + Hp
    jj = j + Hp
    cell_area = FT(cell_areas_panel[i, j])
    dt_ft = FT(dt)

    @inbounds begin
        for k in 1:Nz
            lambda_qc_panel[ii, jj, k] = zero(FT)
        end

        # ── Pass 2 adjoint (transpose of the flux-divergence update) ──
        # Forward (k = 1 … cldbase): q_new[k] = q_post0[k]·(1 − α·cbot)
        #   + α·cb·q_post0[k−1] − α·cb·qc[k] + α·cbot·qc[k+1],
        # with α = dt/bmass, cb = cmfmc[k] (top interface), cbot = cmfmc[k+1]
        # (bottom interface; 0 at the cloud base, which Pass 0 already closed).
        # For k > cldbase the forward leaves q_post0 untouched → adjoint identity
        # (λ_panel[k] kept as-is). λ_qc accumulates from BOTH q_new[k] (the
        # −α·cb·qc[k] term) and q_new[k−1] (the +α·cbot·qc[k] term), so it is +=
        # into the zeroed scratch. The self-read of λ_panel[k] precedes any
        # cross-write into it (which only ever comes from iteration k+1).
        cldbase_k_p2 = _cmfmc_cloud_base(nothing, cmfmc_panel, i, j, Nz, tiny)
        for k in 1:cldbase_k_p2
            m_k = air_mass_panel[ii, jj, k]
            lambda_out = lambda_panel[ii, jj, k]
            lambda_panel[ii, jj, k] = zero(FT)
            if m_k > tiny
                lambda_qnew = lambda_out * m_k
                bmass = m_k / cell_area
                if bmass > tiny
                    alpha = dt_ft / bmass
                    cb   = cmfmc_panel[i, j, k]
                    cbot = k < cldbase_k_p2 ? cmfmc_panel[i, j, k + 1] : zero(FT)
                    # ∂q_new[k]/∂q_post0[k] = 1 − α·cbot
                    lambda_panel[ii, jj, k] +=
                        lambda_qnew * (one(FT) - alpha * cbot) / m_k
                    # ∂q_new[k]/∂q_post0[k−1] = α·cb  (cb = 0 at k = 1)
                    if k > 1
                        m_prev = air_mass_panel[ii, jj, k - 1]
                        if m_prev > tiny
                            lambda_panel[ii, jj, k - 1] +=
                                lambda_qnew * alpha * cb / m_prev
                        end
                    end
                    # ∂q_new[k]/∂qc[k] = −α·cb
                    lambda_qc_panel[ii, jj, k] += lambda_qnew * (-(alpha * cb))
                    # ∂q_new[k]/∂qc[k+1] = α·cbot  (only k < cldbase)
                    if k < cldbase_k_p2
                        lambda_qc_panel[ii, jj, k + 1] += lambda_qnew * alpha * cbot
                    end
                else
                    lambda_panel[ii, jj, k] += lambda_qnew / m_k
                end
            end
        end

        # ── Pass 1 adjoint (transposes the GCHP-style entrn≥0 guard) ─
        # Forward: qc[k] = (cmfmc_bot · qc[k+1] + entrn · q_post0[k]) / cmout
        #          when (entrn ≥ 0 ∧ cmout > tiny); else qc[k] = qc[k+1].
        # Walk k = 1..Nz so λ_qc[k]'s contribution to λ_qc[k+1]
        # accumulates by the time iteration k+1 reads it.
        for k in 1:Nz
            lambda_qc = lambda_qc_panel[ii, jj, k]
            cmfmc_bot = k < Nz ? cmfmc_panel[i, j, k + 1] : zero(FT)
            cmfmc_top = cmfmc_panel[i, j, k]
            dtrain_k = dtrain_panel[i, j, k]
            cmout = cmfmc_top + dtrain_k
            entrn = cmout - cmfmc_bot
            if entrn >= zero(FT) && cmout > tiny
                coeff_below = cmfmc_bot / cmout
                coeff_q     = entrn / cmout
            else
                coeff_below = one(FT)
                coeff_q     = zero(FT)
            end
            m_k = air_mass_panel[ii, jj, k]
            if m_k > tiny
                lambda_panel[ii, jj, k] += lambda_qc * coeff_q / m_k
            end
            if k < Nz
                lambda_qc_panel[ii, jj, k + 1] += lambda_qc * coeff_below
            end
        end

        # ── Pass 0 adjoint (well-mix + cloud-base closure) ─────────
        # Forward writes
        #   rm[k] = qc_mixed · m_k                            (k > cb)
        #   rm[cb] = (q_cb_old + γ/m_cb_pa · (qc_mixed − q_cb_old)) · m_cb
        # with γ = cmfmc_at_cb · dt and
        #   qc_mixed = (mb_pa · qb + γ · q_cb_old) / (mb_pa + γ),
        #   qb = Σ_{k>cb} q_init[k] · m_k_pa / mb_pa.
        # All quantities are linear in q_init, so the adjoint
        # accumulates λ_q_init[k] from λ_q_post0[k] for k ≥ cb. For
        # k > cb, λ_rm_init[k] depends only on the column-summed
        # λ_qc_mixed (every sub-cloud layer collapses to the same
        # value), which is why the post-loop store overwrites
        # lambda_panel[k>cb] with a single coefficient rather than
        # accumulating per-layer.
        cldbase_k = _cmfmc_cloud_base(nothing, cmfmc_panel, i, j, Nz, tiny)
        if cldbase_k != 0 && cldbase_k < Nz
            cmfmc_at_cldbase = cmfmc_panel[i, j, cldbase_k + 1]
            if cmfmc_at_cldbase > tiny
                # Compensated sums, as in the forward's sub-cloud mean.
                mb_pa, mb_pa_comp = zero(FT), zero(FT)
                lambda_qc_mixed, lambda_qc_comp = zero(FT), zero(FT)
                for k in (cldbase_k + 1):Nz
                    m_k = air_mass_panel[ii, jj, k]
                    mb_pa, mb_pa_comp = _kahan_add(mb_pa, mb_pa_comp, m_k / cell_area)
                    # lambda_panel[k] at this point is λ_rm_post0[k].
                    # λ_q_post0[k] = m_k · λ_rm_post0[k]; for k > cb,
                    # all of it flows into λ_qc_mixed.
                    lambda_qc_mixed, lambda_qc_comp =
                        _kahan_add(lambda_qc_mixed, lambda_qc_comp, lambda_panel[ii, jj, k] * m_k)
                end
                if mb_pa > zero(FT)
                    gamma_cb = cmfmc_at_cldbase * dt_ft
                    denom = mb_pa + gamma_cb
                    m_cb = air_mass_panel[ii, jj, cldbase_k]
                    m_cb_pa = m_cb / cell_area
                    # Cloud-base layer: split λ_q_post0[cb] into a
                    # direct λ_q_init[cb] contribution and a
                    # closure-mediated λ_qc_mixed contribution.
                    factor_cb = (m_cb > tiny && m_cb_pa > tiny) ?
                                gamma_cb / m_cb_pa : zero(FT)
                    if m_cb > tiny
                        lambda_q_post0_cb = lambda_panel[ii, jj, cldbase_k] * m_cb
                        lambda_qc_mixed += lambda_q_post0_cb * factor_cb
                        lambda_panel[ii, jj, cldbase_k] =
                            lambda_panel[ii, jj, cldbase_k] * (one(FT) - factor_cb)
                    end
                    # Now propagate λ_qc_mixed into λ_q_init:
                    #   ∂qc_mixed/∂q_init[cb]  = γ_cb / denom
                    #   ∂qc_mixed/∂q_init[k>cb] = m_k_pa / denom
                    # In rm-space, λ_rm_init[k>cb] is layer-independent
                    # (m_k_pa/m_k = 1/cell_area).
                    if denom > tiny
                        if m_cb > tiny
                            lambda_panel[ii, jj, cldbase_k] +=
                                lambda_qc_mixed * gamma_cb / denom / m_cb
                        end
                        coeff_sub = lambda_qc_mixed / (cell_area * denom)
                        for k in (cldbase_k + 1):Nz
                            m_k = air_mass_panel[ii, jj, k]
                            lambda_panel[ii, jj, k] = m_k > tiny ? coeff_sub : zero(FT)
                        end
                    else
                        # Edge case: denom collapsed; ensure sub-cloud
                        # gradients are zeroed rather than carrying the
                        # pre-Pass-0 λ_rm_post0 (which is meaningless
                        # at this point since we already drained it
                        # into λ_qc_mixed).
                        for k in (cldbase_k + 1):Nz
                            lambda_panel[ii, jj, k] = zero(FT)
                        end
                    end
                end
            end
        end
    end
end

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
    # ALWAYS invalidate at the entry of the footprint forward/adjoint helpers.
    # Reason: these helpers are called from `Footprint/ReverseLoop.jl` with a
    # per-step forcing slice that can differ step-to-step. Production
    # `DrivenSimulation` invalidates the cache on met-window advance, but the
    # footprint path bypasses that hook. Mirrors the CMFMC pattern at
    # `:622` / `:708` where `invalidate_cmfmc_cache!(workspace)` is called
    # before every kernel launch for the same reason.
    invalidate_cmfmc_matrix_cache!(workspace)
    if !workspace.derived_valid[]
        _launch_cmfmc_matrix_derivation!(workspace.derived_entu, workspace.derived_detu,
                                          forcing.cmfmc, forcing.dtrain)
        workspace.derived_valid[] = true
    end
    return ConvectionForcing(nothing, nothing,
        (entu = workspace.derived_entu, detu = workspace.derived_detu,
         entd = workspace.zero_entd,    detd = workspace.zero_detd))
end

function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       ::NoConvection, dt, workspace,
                                       mesh::CubedSphereMesh)
    return nothing
end

function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       op::CMFMCConvection, dt,
                                       workspace::CMFMCWorkspace,
                                       mesh::CubedSphereMesh)
    _require_supported_cs_convection_variant(op)
    cmfmc, dtrain = _assert_cmfmc_adjoint_forcing(forcing)
    _require_cmfmc_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    invalidate_cmfmc_cache!(workspace)
    n_sub = _get_or_compute_n_sub!(workspace, cmfmc, panels_m, cell_areas, dt)
    dtrain_arr = _cmfmc_dtrain_array(cmfmc, dtrain, panels_m)
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(panels_rm[1], 3)
    backend = get_backend(panels_rm[1])
    kernel! = _cmfmc_cs_panel_column_kernel!(backend, (16, 16))    # the production forward
    FT = eltype(panels_rm[1])
    sdt = FT(dt) / FT(n_sub)
    @inbounds for _ in 1:n_sub
        for p in 1:6
            rm_packed = reshape(panels_rm[p], size(panels_rm[p])..., 1)   # one packed tracer
            kernel!(rm_packed, panels_m[p], cmfmc[p], dtrain_arr[p], cell_areas[p],
                    nothing, workspace.qc_scratch[p],
                    Nz, 1, sdt, Hp, Val(true), Val(false);
                    ndrange = (Nc, Nc))
        end
    end
    synchronize(backend)
    return nothing
end

function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       op::TM5Convection, dt,
                                       workspace::TM5Workspace,
                                       mesh::CubedSphereMesh)
    _require_supported_cs_convection_variant(op)
    tm5 = _assert_tm5_adjoint_forcing(forcing)
    _require_tm5_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    Nc = mesh.Nc
    Hp = mesh.Hp
    N_total = Nc * Nc
    _ensure_tm5_scratch!(workspace)
    B = size(workspace.conv1, 3)
    backend = get_backend(panels_rm[1])
    kernel! = _tm5_cs_panel_column_single_kernel!(backend)
    FT = eltype(panels_rm[1])
    @inbounds for p in 1:6
        for tile_off in 0:B:(N_total - 1)
            n = min(B, N_total - tile_off)
            kernel!(panels_rm[p], panels_m[p],
                    tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p],
                    cell_areas[p],
                    workspace.conv1, workspace.pivots, workspace.cloud_dims,
                    workspace.f_scratch,
                    workspace.amu_scratch, workspace.amd_scratch,
                    Hp, Int(tile_off), Nc, FT(dt);
                    ndrange = n)
        end
    end
    synchronize(backend)
    return nothing
end

# CMFMCMatrixConvection forward = refresh-derive + TM5 forward. Cheap
# alias-and-delegate; no kernel duplication.
function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       op::CMFMCMatrixConvection, dt,
                                       workspace::CMFMCMatrixWorkspace,
                                       mesh::CubedSphereMesh)
    _require_supported_cs_convection_variant(op)
    synth = _refresh_and_synthesize_cmfmc_matrix_forcing!(workspace, forcing)
    return _apply_cs_convection_forward!(panels_rm, panels_m, synth,
                                          op.inner, dt, workspace.tm5_workspace, mesh)
end

function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       op, dt, workspace,
                                       mesh::CubedSphereMesh)
    throw(ArgumentError("CS adjoint footprint forward helper supports `NoConvection`, " *
                        "`CMFMCConvection`, `TM5Convection`, and `CMFMCMatrixConvection`; " *
                        "got $(typeof(op))"))
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       ::NoConvection, dt, workspace,
                                       mesh::CubedSphereMesh)
    return nothing
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       op::CMFMCConvection, dt,
                                       workspace::CMFMCWorkspace,
                                       mesh::CubedSphereMesh)
    _require_supported_cs_convection_variant(op)
    cmfmc, dtrain = _assert_cmfmc_adjoint_forcing(forcing)
    _require_cmfmc_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    invalidate_cmfmc_cache!(workspace)
    n_sub = _get_or_compute_n_sub!(workspace, cmfmc, panels_m, cell_areas, dt)
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
    synchronize(backend)
    return nothing
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       op::TM5Convection, dt,
                                       workspace::TM5Workspace,
                                       mesh::CubedSphereMesh)
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
    synchronize(backend)
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
                                       mesh::CubedSphereMesh)
    _require_supported_cs_convection_variant(op)
    synth = _refresh_and_synthesize_cmfmc_matrix_forcing!(workspace, forcing)
    return _apply_cs_convection_adjoint!(lambda_panels, panels_m, synth,
                                          op.inner, dt, workspace.tm5_workspace, mesh)
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       op, dt, workspace,
                                       mesh::CubedSphereMesh)
    throw(ArgumentError("CS adjoint footprint reverse helper supports `NoConvection`, " *
                        "`CMFMCConvection`, `TM5Convection`, and `CMFMCMatrixConvection`; " *
                        "got $(typeof(op))"))
end
