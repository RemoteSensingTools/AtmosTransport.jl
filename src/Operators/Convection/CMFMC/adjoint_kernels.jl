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
    # Transpose of one sub-step of the forward column `_cmfmc_cs_column!`
    # (CMFMC-edge cloud base, no clamp). The forward is linear in `q = rm/m`,
    # so the adjoint is rm-to-rm linear. With cloud base `cb`, sub-cloud air
    # mass `mb_pa` and `γ = cmfmc[cb+1]·dt` (kg/m²), the forward
    #
    #   (0) mixes the sub-cloud layers:
    #         qc_mixed = (Σ_{k>cb} q[k]·m_k_pa + γ·q[cb]) / (mb_pa + γ),
    #         q[k > cb] ← qc_mixed  (`cb_inflow`; q[cb] is left as it is);
    #   (1) computes the updraft bottom-up, entering the cloud base with
    #         qc_mixed: qc[k] = (cmfmc[k+1]·qc[k+1] + entrn·q[k]) / cmout,
    #         qc[cb+1] ≡ qc_mixed, or qc[k] = qc[k+1] outside the guard;
    #   (2) updates k = 1 … cb by the edge fluxes
    #         Φ(k) = cmfmc[k]·(qc[k] − q[k−1]),
    #         Φ(cb+1) = cmfmc[cb+1]·(qc_mixed − q[cb]) (0 without inflow),
    #         q_new[k] = q[k] + α·(Φ(k+1) − Φ(k)),  α = dt / bmass.
    #
    # The reverse walks (2) → (1) → (0). `lambda_qc_panel` carries λ_qc
    # through (1) as `qc_scratch` carries qc in the forward, and `lambda_qcm`
    # collects λ of qc_mixed from all three steps.
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
        cldbase_k = _cmfmc_cloud_base(nothing, cmfmc_panel, i, j, Nz, tiny)

        # Step (0) constants, as in the forward: the sub-cloud exchange runs
        # (`cb_inflow`) when there is inflow through the cloud-base bottom
        # edge and air below it.
        cb_inflow = false
        mb_pa = zero(FT)
        gamma_cb = zero(FT)
        if cldbase_k != 0 && cldbase_k < Nz
            cmfmc_at_cldbase = cmfmc_panel[i, j, cldbase_k + 1]
            if cmfmc_at_cldbase > tiny
                # Compensated sum, as in the forward's sub-cloud mean.
                mb_pa_comp = zero(FT)
                for k in (cldbase_k + 1):Nz
                    mb_pa, mb_pa_comp = _kahan_add(mb_pa, mb_pa_comp, air_mass_panel[ii, jj, k] / cell_area)
                end
                if mb_pa > zero(FT)
                    cb_inflow = true
                    gamma_cb = cmfmc_at_cldbase * dt_ft
                end
            end
        end
        lambda_qcm = zero(FT)

        # ── Step (2) adjoint (transpose of the flux-divergence update) ──
        # q_new[k] = q[k]·(1 − α·cbot) + α·cb·q[k−1] − α·cb·qc[k] + α·cbot·qc[k+1],
        # cb = cmfmc[k], cbot = cmfmc[k+1]; at the cloud base qc[cb+1] is
        # qc_mixed and cbot is 0 without inflow. For k > cldbase step (2)
        # leaves q untouched → identity (λ kept). λ_qc accumulates from
        # q_new[k] and q_new[k−1], so it is += into the zeroed scratch. The
        # self-read of λ[k] precedes any cross-write into it (which only comes
        # from iteration k+1).
        for k in 1:cldbase_k
            m_k = air_mass_panel[ii, jj, k]
            lambda_out = lambda_panel[ii, jj, k]
            lambda_panel[ii, jj, k] = zero(FT)
            if m_k > tiny
                lambda_qnew = lambda_out * m_k
                bmass = m_k / cell_area
                if bmass > tiny
                    alpha = dt_ft / bmass
                    cb   = cmfmc_panel[i, j, k]
                    cbot = (k < cldbase_k || cb_inflow) ? cmfmc_panel[i, j, k + 1] : zero(FT)
                    # ∂q_new[k]/∂q[k] = 1 − α·cbot
                    lambda_panel[ii, jj, k] +=
                        lambda_qnew * (one(FT) - alpha * cbot) / m_k
                    # ∂q_new[k]/∂q[k−1] = α·cb  (cb = 0 at k = 1)
                    if k > 1
                        m_prev = air_mass_panel[ii, jj, k - 1]
                        if m_prev > tiny
                            lambda_panel[ii, jj, k - 1] +=
                                lambda_qnew * alpha * cb / m_prev
                        end
                    end
                    # ∂q_new[k]/∂qc[k] = −α·cb
                    lambda_qc_panel[ii, jj, k] += lambda_qnew * (-(alpha * cb))
                    # ∂q_new[k]/∂qc[k+1] = α·cbot; at the cloud base qc[k+1]
                    # is qc_mixed.
                    if k < cldbase_k
                        lambda_qc_panel[ii, jj, k + 1] += lambda_qnew * alpha * cbot
                    elseif cb_inflow
                        lambda_qcm += lambda_qnew * alpha * cbot
                    end
                else
                    lambda_panel[ii, jj, k] += lambda_qnew / m_k
                end
            end
        end

        # ── Step (1) adjoint (transposes the GCHP-style entrn≥0 guard) ─
        # qc[k] = (cmfmc_bot · qc_below + entrn · q[k]) / cmout when
        # (entrn ≥ 0 ∧ cmout > tiny), else qc[k] = qc_below, where qc_below is
        # qc[k+1], or qc_mixed at the cloud base with inflow. Walk k = 1..Nz so
        # λ_qc[k]'s contribution to λ_qc[k+1] accumulates by the time
        # iteration k+1 reads it.
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
            if cb_inflow && k == cldbase_k
                lambda_qcm += lambda_qc * coeff_below
            elseif k < Nz
                lambda_qc_panel[ii, jj, k + 1] += lambda_qc * coeff_below
            end
        end

        # ── Step (0) adjoint (sub-cloud mixing) ─────────────────────
        # rm[k > cb] = qc_mixed · m_k, and q[cb] enters qc_mixed only. With
        # m_k_pa / m_k = 1 / cell_area, λ_rm[k > cb] is the same in every
        # sub-cloud layer that holds air: λ_qcm / (cell_area · (mb_pa + γ)).
        if cb_inflow
            lambda_qcm_comp = zero(FT)
            for k in (cldbase_k + 1):Nz
                # λ[k] here is λ_rm after step (0); λ_q = m_k · λ_rm all flows
                # into λ_qcm.
                lambda_qcm, lambda_qcm_comp =
                    _kahan_add(lambda_qcm, lambda_qcm_comp, lambda_panel[ii, jj, k] * air_mass_panel[ii, jj, k])
            end
            denom = mb_pa + gamma_cb
            coeff_sub = lambda_qcm / (cell_area * denom)
            for k in (cldbase_k + 1):Nz
                lambda_panel[ii, jj, k] = air_mass_panel[ii, jj, k] > tiny ? coeff_sub : zero(FT)
            end
            m_cb = air_mass_panel[ii, jj, cldbase_k]
            if m_cb > tiny
                lambda_panel[ii, jj, cldbase_k] += lambda_qcm * gamma_cb / denom / m_cb
            end
        end
    end
end
