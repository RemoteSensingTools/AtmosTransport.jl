# ---------------------------------------------------------------------------
# Vertical diffusion kernels.
#
# All topology variants use the same mass-flux formulation. The Backward-
# Euler solve `Ã·q_new = q_old` acts on dry mass-mixing ratio `q`; `Ã` is
# constructed so the equivalent tracer-mass operator is column-stochastic and
# therefore preserves `Σ m·q` to roundoff for every inert tracer.
#
# Reference for the mass-flux form: TM5's `TM5_Diff_Matrix` at
# `deps/tm5-cy3-4dvar/base/src/tm5_diff.F90:36-129`.
# `src/Adjoints/DiffusionAdjoint.jl` implements the transpose.
#
# Packed tracers share one tridiagonal system per atmospheric column. Each
# packed kernel therefore assigns one work item to a horizontal column,
# computes every Thomas factor once, and advances all tracers inside that
# work item. `w_scratch` remains tracer-independent without concurrent writes.
#
# Coefficients per level k (with `m_k = air_mass[..., k]`, `Kz` at cell
# centers, `dz` thickness in meters):
#
#     dkg[k-½] = (m_{k-1} + m_k) · (Kz_{k-1} + Kz_k) / (dz_{k-1} + dz_k)²
#     dkg[k+½] = (m_k + m_{k+1}) · (Kz_k + Kz_{k+1}) / (dz_k + dz_{k+1})²
#
#     a_k = -dt · dkg[k-½] / m_k                       (sub-diagonal)
#     b_k = 1 + dt · (dkg[k-½] + dkg[k+½]) / m_k       (diagonal)
#     c_k = -dt · dkg[k+½] / m_k                       (super-diagonal)
#
# Boundary: dkg[½] = dkg[Nz+½] = 0 (zero-flux at TOA and surface).
#
# Reference: TM5's `TM5_Diff_Matrix` at `deps/tm5-cy3-4dvar/base/src/
# tm5_diff.F90:36-129` uses the same `fu = dt·dkg/m(k)`, `fd = dt·dkg/m(k+1)`
# fraction-of-mass-exchanged form on the tracer-mass vector directly.
# We operate on VMR with `Ã = M⁻¹·A·M`, which has the same conservation
# guarantee via `Σ m·q_new = Σ m·q_old`.
#
# The previous geometric form `D = Kz / (dz_k · dz_iface)` (kept in git
# history at commit 400410b~) conserved `Σ q·dz` but not `Σ m·q`,
# leaking tracer mass by ~1–10 % per 3-day CS180 run.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LL packed mass-flux kernel — used by `apply_vertical_diffusion_vmr!` for
# the LL state path. Identical math to the CS kernels (mass-flux form,
# preserves `Σ m·q` to roundoff for inert tracers); same TM5 reference at
# `deps/tm5-cy3-4dvar/base/src/tm5_diff.F90:36-129`.
# ---------------------------------------------------------------------------
@kernel function _vertical_diffusion_kernel_mass_flux!(q, @Const(air_mass),
                                                       kz_field,
                                                       @Const(dz),
                                                       w_scratch, reference_scratch,
                                                       dt, Nz::Int, Nt::Int)
    i, j = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        dt_ft = FT(dt)

        # Anomaly diffusion (F32 conservation fix), as in the cubed-sphere
        # kernels: solve for the departure from the column minimum of each
        # tracer and add the minimum back (the operator keeps a uniform column).
        for t in 1:Nt
            cref = q[i, j, 1, t]
            for k in 2:Nz
                v = q[i, j, k, t]
                cref = v < cref ? v : cref
            end
            reference_scratch[i, j, t] = cref
        end

        Kz_prev = zero(FT)
        dz_prev = zero(FT)
        m_prev  = zero(FT)
        w_prev  = zero(FT)

        Kz_k = field_value(kz_field, (i, j, 1))
        dz_k = dz[i, j, 1]
        m_k  = air_mass[i, j, 1]

        for k in 1:Nz
            dkg_above = zero(FT)
            dkg_below = zero(FT)
            Kz_next   = zero(FT)
            dz_next   = zero(FT)
            m_next    = zero(FT)

            if k > 1
                sum_dz_above = dz_prev + dz_k
                dkg_above = (m_prev + m_k) * (Kz_prev + Kz_k) /
                            (sum_dz_above * sum_dz_above)
            end

            if k < Nz
                Kz_next  = field_value(kz_field, (i, j, k + 1))
                dz_next  = dz[i, j, k + 1]
                m_next   = air_mass[i, j, k + 1]
                sum_dz_below = dz_k + dz_next
                dkg_below = (m_k + m_next) * (Kz_k + Kz_next) /
                            (sum_dz_below * sum_dz_below)
            end

            inv_m_k = m_k > zero(FT) ? one(FT) / m_k : zero(FT)
            a_k = (k > 1)  ? -dt_ft * dkg_above * inv_m_k : zero(FT)
            c_k = (k < Nz) ? -dt_ft * dkg_below * inv_m_k : zero(FT)
            b_k = one(FT) + dt_ft * (dkg_above + dkg_below) * inv_m_k
            if k == 1
                denom = b_k
                w_k   = c_k / denom
            else
                denom = b_k - a_k * w_prev
                w_k   = c_k / denom
            end

            w_scratch[i, j, k] = w_k
            for t in 1:Nt
                d_k = q[i, j, k, t] - reference_scratch[i, j, t]
                g_k = k == 1 ? d_k / denom :
                      (d_k - a_k * q[i, j, k - 1, t]) / denom
                q[i, j, k, t] = g_k
            end

            if k < Nz
                w_prev  = w_k
                Kz_prev = Kz_k
                dz_prev = dz_k
                m_prev  = m_k
                Kz_k    = Kz_next
                dz_k    = dz_next
                m_k     = m_next
            end
        end

        for k in (Nz - 1):-1:1, t in 1:Nt
            q[i, j, k, t] -= w_scratch[i, j, k] * q[i, j, k + 1, t]
        end

        # restore the per-column reference removed before the solve
        for k in 1:Nz, t in 1:Nt
            q[i, j, k, t] += reference_scratch[i, j, t]
        end
    end
end

"""
    _vertical_diffusion_cs_single_kernel!(q, kz_field, dz, w_scratch, dt, Nz, Hp)

Cubed-sphere single-tracer diffusion kernel.

`q` is one halo-padded panel `(Nc + 2Hp, Nc + 2Hp, Nz)` while `dz` and
`w_scratch` are interior `(Nc, Nc, Nz)` workspaces. The structured
column solve is unchanged; only the panel halo offset differs.
"""
@kernel function _vertical_diffusion_cs_single_kernel!(q, @Const(air_mass),
                                                       kz_field,
                                                       @Const(dz),
                                                       w_scratch,
                                                       dt, Nz::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        dt_ft = FT(dt)

        # Anomaly diffusion (F32 conservation fix): the implicit operator
        # preserves a uniform column exactly (each tridiagonal row sums to 1
        # under the zero-flux BCs), so subtracting a per-column reference VMR
        # before the solve and adding it back is an identity in exact arithmetic
        # (L⁻¹(c·1)=c·1) — it only changes the F32 ROUNDING, keeping the
        # elimination on the small anomaly instead of the large background.
        # Without it, the Thomas elimination on a background-dominated VMR loses
        # ~1e-5 relative mass, contaminating small emission budgets (the SF6
        # ~2.5% deficit). Never worse than the plain solve: a column whose
        # current min is exactly 0 (e.g. a still-zero IC=0 tracer) gives cref=0
        # and is bit-identical; otherwise it strictly improves F32 conservation.
        cref = q[i, j, 1]
        for k in 2:Nz
            v = q[i, j, k]
            cref = v < cref ? v : cref
        end

        Kz_prev = zero(FT)
        dz_prev = zero(FT)
        m_prev  = zero(FT)
        w_prev  = zero(FT)
        g_prev  = zero(FT)

        Kz_k = field_value(kz_field, (ii, jj, 1))
        dz_k = dz[ii, jj, 1]
        m_k  = air_mass[i, j, 1]

        for k in 1:Nz
            dkg_above = zero(FT)
            dkg_below = zero(FT)
            Kz_next   = zero(FT)
            dz_next   = zero(FT)
            m_next    = zero(FT)

            if k > 1
                sum_dz_above = dz_prev + dz_k
                dkg_above = (m_prev + m_k) * (Kz_prev + Kz_k) /
                            (sum_dz_above * sum_dz_above)
            end

            if k < Nz
                Kz_next  = field_value(kz_field, (ii, jj, k + 1))
                dz_next  = dz[ii, jj, k + 1]
                m_next   = air_mass[i, j, k + 1]
                sum_dz_below = dz_k + dz_next
                dkg_below = (m_k + m_next) * (Kz_k + Kz_next) /
                            (sum_dz_below * sum_dz_below)
            end

            inv_m_k = m_k > zero(FT) ? one(FT) / m_k : zero(FT)
            a_k = (k > 1)  ? -dt_ft * dkg_above * inv_m_k : zero(FT)
            c_k = (k < Nz) ? -dt_ft * dkg_below * inv_m_k : zero(FT)
            b_k = one(FT) + dt_ft * (dkg_above + dkg_below) * inv_m_k
            d_k = q[i, j, k] - cref

            if k == 1
                denom = b_k
                w_k   = c_k / denom
                g_k   = d_k / denom
            else
                denom = b_k - a_k * w_prev
                w_k   = c_k / denom
                g_k   = (d_k - a_k * g_prev) / denom
            end

            w_scratch[ii, jj, k] = w_k
            q[i, j, k]           = g_k

            if k < Nz
                w_prev  = w_k
                g_prev  = g_k
                Kz_prev = Kz_k
                dz_prev = dz_k
                m_prev  = m_k
                Kz_k    = Kz_next
                dz_k    = dz_next
                m_k     = m_next
            end
        end

        for k in (Nz - 1):-1:1
            q[i, j, k] = q[i, j, k] - w_scratch[ii, jj, k] * q[i, j, k + 1]
        end

        # restore the per-column reference removed before the solve
        for k in 1:Nz
            q[i, j, k] += cref
        end
    end
end

# Exact precomputed-TM5 path. `dkg_field[..., k]` is the exchange rate
# [kg s⁻¹] through the interface between top-down layers k and k+1, with the
# final entry zero at the surface. Geometry, Kvh, and the chosen mass basis are
# already baked by preprocessing, so this kernel must not reconstruct them.
@kernel function _vertical_diffusion_cs_single_dkg_kernel!(q, @Const(air_mass),
                                                           dkg_field, w_scratch,
                                                           dt, Nz::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        dt_ft = FT(dt)
        cref = q[i, j, 1]
        for k in 2:Nz
            cref = min(cref, q[i, j, k])
        end
        w_prev = zero(FT)
        g_prev = zero(FT)
        for k in 1:Nz
            dkg_above = k > 1  ? field_value(dkg_field, (ii, jj, k - 1)) : zero(FT)
            dkg_below = k < Nz ? field_value(dkg_field, (ii, jj, k))     : zero(FT)
            m_k = air_mass[i, j, k]
            inv_m_k = m_k > zero(FT) ? one(FT) / m_k : zero(FT)
            a_k = k > 1  ? -dt_ft * dkg_above * inv_m_k : zero(FT)
            c_k = k < Nz ? -dt_ft * dkg_below * inv_m_k : zero(FT)
            b_k = one(FT) + dt_ft * (dkg_above + dkg_below) * inv_m_k
            d_k = q[i, j, k] - cref
            if k == 1
                w_k = c_k / b_k
                g_k = d_k / b_k
            else
                denom = b_k - a_k * w_prev
                w_k = c_k / denom
                g_k = (d_k - a_k * g_prev) / denom
            end
            w_scratch[ii, jj, k] = w_k
            q[i, j, k] = g_k
            w_prev = w_k
            g_prev = g_k
        end
        for k in (Nz - 1):-1:1
            q[i, j, k] -= w_scratch[ii, jj, k] * q[i, j, k + 1]
        end
        for k in 1:Nz
            q[i, j, k] += cref
        end
    end
end

"""
    _vertical_diffusion_cs_kernel!(q, air_mass, kz_field, dz, w_scratch,
                                   reference_scratch, dt, Nz, Nt, Hp)

Packed cubed-sphere diffusion kernel. `q` is one halo-padded panel
`(Nc + 2Hp, Nc + 2Hp, Nz, Nt)`. One work item owns one interior
`(ii, jj)` column, computes its tracer-independent tridiagonal factors once,
and advances all `Nt` tracers. `reference_scratch[:, :, t]` stores the
per-tracer column offset used to limit Float32 cancellation.
"""
@kernel function _vertical_diffusion_cs_kernel!(q, @Const(air_mass),
                                                kz_field,
                                                @Const(dz),
                                                w_scratch, reference_scratch,
                                                dt, Nz::Int, Nt::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        dt_ft = FT(dt)

        # Anomaly diffusion (F32 conservation fix) — see the single-column
        # kernel above for the rationale. Per-tracer per-column reference: the
        # tridiagonal coefficients are shared across tracers but the reference
        # is the column min of THIS tracer slice. A column whose current min is
        # exactly 0 gives cref=0 (bit-identical to the plain solve); otherwise
        # it strictly improves F32 conservation.
        for t in 1:Nt
            cref = q[i, j, 1, t]
            for k in 2:Nz
                v = q[i, j, k, t]
                cref = v < cref ? v : cref
            end
            reference_scratch[ii, jj, t] = cref
        end

        Kz_prev = zero(FT)
        dz_prev = zero(FT)
        m_prev  = zero(FT)
        w_prev  = zero(FT)

        Kz_k = field_value(kz_field, (ii, jj, 1))
        dz_k = dz[ii, jj, 1]
        m_k  = air_mass[i, j, 1]

        for k in 1:Nz
            dkg_above = zero(FT)
            dkg_below = zero(FT)
            Kz_next   = zero(FT)
            dz_next   = zero(FT)
            m_next    = zero(FT)

            if k > 1
                sum_dz_above = dz_prev + dz_k
                dkg_above = (m_prev + m_k) * (Kz_prev + Kz_k) /
                            (sum_dz_above * sum_dz_above)
            end

            if k < Nz
                Kz_next  = field_value(kz_field, (ii, jj, k + 1))
                dz_next  = dz[ii, jj, k + 1]
                m_next   = air_mass[i, j, k + 1]
                sum_dz_below = dz_k + dz_next
                dkg_below = (m_k + m_next) * (Kz_k + Kz_next) /
                            (sum_dz_below * sum_dz_below)
            end

            inv_m_k = m_k > zero(FT) ? one(FT) / m_k : zero(FT)
            a_k = (k > 1)  ? -dt_ft * dkg_above * inv_m_k : zero(FT)
            c_k = (k < Nz) ? -dt_ft * dkg_below * inv_m_k : zero(FT)
            b_k = one(FT) + dt_ft * (dkg_above + dkg_below) * inv_m_k
            if k == 1
                denom = b_k
                w_k   = c_k / denom
            else
                denom = b_k - a_k * w_prev
                w_k   = c_k / denom
            end

            w_scratch[ii, jj, k] = w_k
            for t in 1:Nt
                d_k = q[i, j, k, t] - reference_scratch[ii, jj, t]
                g_k = k == 1 ? d_k / denom :
                      (d_k - a_k * q[i, j, k - 1, t]) / denom
                q[i, j, k, t] = g_k
            end

            if k < Nz
                w_prev  = w_k
                Kz_prev = Kz_k
                dz_prev = dz_k
                m_prev  = m_k
                Kz_k    = Kz_next
                dz_k    = dz_next
                m_k     = m_next
            end
        end

        for k in (Nz - 1):-1:1, t in 1:Nt
            q[i, j, k, t] -= w_scratch[ii, jj, k] * q[i, j, k + 1, t]
        end

        # restore the per-column reference removed before the solve
        for k in 1:Nz, t in 1:Nt
            q[i, j, k, t] += reference_scratch[ii, jj, t]
        end
    end
end

@kernel function _vertical_diffusion_cs_dkg_kernel!(q, @Const(air_mass),
                                                    dkg_field, w_scratch,
                                                    reference_scratch,
                                                    dt, Nz::Int, Nt::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        dt_ft = FT(dt)
        for t in 1:Nt
            cref = q[i, j, 1, t]
            for k in 2:Nz
                cref = min(cref, q[i, j, k, t])
            end
            reference_scratch[ii, jj, t] = cref
        end
        w_prev = zero(FT)
        for k in 1:Nz
            dkg_above = k > 1  ? field_value(dkg_field, (ii, jj, k - 1)) : zero(FT)
            dkg_below = k < Nz ? field_value(dkg_field, (ii, jj, k))     : zero(FT)
            m_k = air_mass[i, j, k]
            inv_m_k = m_k > zero(FT) ? one(FT) / m_k : zero(FT)
            a_k = k > 1  ? -dt_ft * dkg_above * inv_m_k : zero(FT)
            c_k = k < Nz ? -dt_ft * dkg_below * inv_m_k : zero(FT)
            b_k = one(FT) + dt_ft * (dkg_above + dkg_below) * inv_m_k
            if k == 1
                w_k = c_k / b_k
                denom = b_k
            else
                denom = b_k - a_k * w_prev
                w_k = c_k / denom
            end
            w_scratch[ii, jj, k] = w_k
            for t in 1:Nt
                d_k = q[i, j, k, t] - reference_scratch[ii, jj, t]
                g_k = k == 1 ? d_k / denom :
                      (d_k - a_k * q[i, j, k - 1, t]) / denom
                q[i, j, k, t] = g_k
            end
            w_prev = w_k
        end
        for k in (Nz - 1):-1:1, t in 1:Nt
            q[i, j, k, t] -= w_scratch[ii, jj, k] * q[i, j, k + 1, t]
        end
        for k in 1:Nz, t in 1:Nt
            q[i, j, k, t] += reference_scratch[ii, jj, t]
        end
    end
end

@kernel function _cs_tracer_mass_to_vmr_kernel!(q, @Const(air_mass), Hp::Int)
    ii, jj, k = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        m = air_mass[i, j, k]
        q[i, j, k] = m > zero(FT) ? q[i, j, k] / m : zero(FT)
    end
end

@kernel function _cs_vmr_to_tracer_mass_kernel!(q, @Const(air_mass), Hp::Int)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        q[i, j, k] *= air_mass[i, j, k]
    end
end

@kernel function _cs_tracer_mass_to_vmr_4d_kernel!(q, @Const(air_mass), Hp::Int)
    ii, jj, k, t = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        m = air_mass[i, j, k]
        q[i, j, k, t] = m > zero(FT) ? q[i, j, k, t] / m : zero(FT)
    end
end

@kernel function _cs_vmr_to_tracer_mass_4d_kernel!(q, @Const(air_mass), Hp::Int)
    ii, jj, k, t = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        q[i, j, k, t] *= air_mass[i, j, k]
    end
end

# ---------------------------------------------------------------------------
# Face-indexed (ReducedGaussian) mass-flux kernels — used by the LL/RG
# branch of `apply_vertical_diffusion_vmr!`. Same TM5-style mass-flux
# coefficients as the LL packed / CS kernels above; only the storage
# layout changes (2D `(ncells, Nz)` interior).
# ---------------------------------------------------------------------------

@kernel function _vertical_diffusion_face_kernel_mass_flux!(q, @Const(air_mass),
                                                             kz_field,
                                                             @Const(dz),
                                                             w_scratch, reference_scratch,
                                                             dt, Nz::Int, Nt::Int)
    c = @index(Global, Linear)
    FT = eltype(q)
    @inbounds begin
        dt_ft = FT(dt)

        # Anomaly diffusion (F32 conservation fix): per-tracer column minimum.
        for t in 1:Nt
            cref = q[c, 1, t]
            for k in 2:Nz
                v = q[c, k, t]
                cref = v < cref ? v : cref
            end
            reference_scratch[c, t] = cref
        end

        Kz_prev = zero(FT)
        dz_prev = zero(FT)
        m_prev  = zero(FT)
        w_prev  = zero(FT)

        Kz_k = field_value(kz_field, (c, 1))
        dz_k = dz[c, 1]
        m_k  = air_mass[c, 1]

        for k in 1:Nz
            dkg_above = zero(FT)
            dkg_below = zero(FT)
            Kz_next   = zero(FT)
            dz_next   = zero(FT)
            m_next    = zero(FT)

            if k > 1
                sum_dz_above = dz_prev + dz_k
                dkg_above = (m_prev + m_k) * (Kz_prev + Kz_k) /
                            (sum_dz_above * sum_dz_above)
            end

            if k < Nz
                Kz_next  = field_value(kz_field, (c, k + 1))
                dz_next  = dz[c, k + 1]
                m_next   = air_mass[c, k + 1]
                sum_dz_below = dz_k + dz_next
                dkg_below = (m_k + m_next) * (Kz_k + Kz_next) /
                            (sum_dz_below * sum_dz_below)
            end

            inv_m_k = m_k > zero(FT) ? one(FT) / m_k : zero(FT)
            a_k = (k > 1)  ? -dt_ft * dkg_above * inv_m_k : zero(FT)
            c_k = (k < Nz) ? -dt_ft * dkg_below * inv_m_k : zero(FT)
            b_k = one(FT) + dt_ft * (dkg_above + dkg_below) * inv_m_k
            if k == 1
                denom = b_k
                w_k   = c_k / denom
            else
                denom = b_k - a_k * w_prev
                w_k   = c_k / denom
            end

            w_scratch[c, k] = w_k
            for t in 1:Nt
                d_k = q[c, k, t] - reference_scratch[c, t]
                g_k = k == 1 ? d_k / denom :
                      (d_k - a_k * q[c, k - 1, t]) / denom
                q[c, k, t] = g_k
            end

            if k < Nz
                w_prev  = w_k
                Kz_prev = Kz_k
                dz_prev = dz_k
                m_prev  = m_k
                Kz_k    = Kz_next
                dz_k    = dz_next
                m_k     = m_next
            end
        end

        for k in (Nz - 1):-1:1, t in 1:Nt
            q[c, k, t] -= w_scratch[c, k] * q[c, k + 1, t]
        end

        # restore the per-column reference removed before the solve
        for k in 1:Nz, t in 1:Nt
            q[c, k, t] += reference_scratch[c, t]
        end
    end
end

@kernel function _vertical_diffusion_face_single_kernel_mass_flux!(q, @Const(air_mass),
                                                                    kz_field,
                                                                    @Const(dz),
                                                                    w_scratch,
                                                                    dt, Nz::Int)
    c = @index(Global, Linear)
    FT = eltype(q)
    @inbounds begin
        dt_ft = FT(dt)

        # Anomaly diffusion (F32 conservation fix): column minimum, kept in a register.
        cref = q[c, 1]
        for k in 2:Nz
            v = q[c, k]
            cref = v < cref ? v : cref
        end

        Kz_prev = zero(FT)
        dz_prev = zero(FT)
        m_prev  = zero(FT)
        w_prev  = zero(FT)
        g_prev  = zero(FT)

        Kz_k = field_value(kz_field, (c, 1))
        dz_k = dz[c, 1]
        m_k  = air_mass[c, 1]

        for k in 1:Nz
            dkg_above = zero(FT)
            dkg_below = zero(FT)
            Kz_next   = zero(FT)
            dz_next   = zero(FT)
            m_next    = zero(FT)

            if k > 1
                sum_dz_above = dz_prev + dz_k
                dkg_above = (m_prev + m_k) * (Kz_prev + Kz_k) /
                            (sum_dz_above * sum_dz_above)
            end

            if k < Nz
                Kz_next  = field_value(kz_field, (c, k + 1))
                dz_next  = dz[c, k + 1]
                m_next   = air_mass[c, k + 1]
                sum_dz_below = dz_k + dz_next
                dkg_below = (m_k + m_next) * (Kz_k + Kz_next) /
                            (sum_dz_below * sum_dz_below)
            end

            inv_m_k = m_k > zero(FT) ? one(FT) / m_k : zero(FT)
            a_k = (k > 1)  ? -dt_ft * dkg_above * inv_m_k : zero(FT)
            c_k = (k < Nz) ? -dt_ft * dkg_below * inv_m_k : zero(FT)
            b_k = one(FT) + dt_ft * (dkg_above + dkg_below) * inv_m_k
            d_k = q[c, k] - cref

            if k == 1
                denom = b_k
                w_k   = c_k / denom
                g_k   = d_k / denom
            else
                denom = b_k - a_k * w_prev
                w_k   = c_k / denom
                g_k   = (d_k - a_k * g_prev) / denom
            end

            w_scratch[c, k] = w_k
            q[c, k]         = g_k

            if k < Nz
                w_prev  = w_k
                g_prev  = g_k
                Kz_prev = Kz_k
                dz_prev = dz_k
                m_prev  = m_k
                Kz_k    = Kz_next
                dz_k    = dz_next
                m_k     = m_next
            end
        end

        for k in (Nz - 1):-1:1
            q[c, k] = q[c, k] - w_scratch[c, k] * q[c, k + 1]
        end

        # restore the column reference removed before the solve
        for k in 1:Nz
            q[c, k] += cref
        end
    end
end

# ---------------------------------------------------------------------------
# Pre/post mass-VMR scaling kernels — LL packed + RG face-indexed.
# Mirror of the CS `_cs_tracer_mass_to_vmr_kernel!` / `_cs_vmr_to_tracer_mass_kernel!`
# but for the structured `(Nx, Ny, Nz, Nt)` and face-indexed
# `(ncells, Nz, Nt)` / `(ncells, Nz)` layouts.
# ---------------------------------------------------------------------------

@kernel function _ll_tracer_mass_to_vmr_kernel!(q, @Const(air_mass))
    i, j, k, t = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        m = air_mass[i, j, k]
        q[i, j, k, t] = m > zero(FT) ? q[i, j, k, t] / m : zero(FT)
    end
end

@kernel function _ll_vmr_to_tracer_mass_kernel!(q, @Const(air_mass))
    i, j, k, t = @index(Global, NTuple)
    @inbounds q[i, j, k, t] *= air_mass[i, j, k]
end

@kernel function _face_tracer_mass_to_vmr_kernel!(q, @Const(air_mass))
    c, k, t = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        m = air_mass[c, k]
        q[c, k, t] = m > zero(FT) ? q[c, k, t] / m : zero(FT)
    end
end

@kernel function _face_vmr_to_tracer_mass_kernel!(q, @Const(air_mass))
    c, k, t = @index(Global, NTuple)
    @inbounds q[c, k, t] *= air_mass[c, k]
end

@kernel function _face_single_tracer_mass_to_vmr_kernel!(q, @Const(air_mass))
    c, k = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        m = air_mass[c, k]
        q[c, k] = m > zero(FT) ? q[c, k] / m : zero(FT)
    end
end

@kernel function _face_single_vmr_to_tracer_mass_kernel!(q, @Const(air_mass))
    c, k = @index(Global, NTuple)
    @inbounds q[c, k] *= air_mass[c, k]
end
