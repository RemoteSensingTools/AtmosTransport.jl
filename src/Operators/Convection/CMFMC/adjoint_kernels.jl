# Transpose of the CMFMC CS column operator.
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
