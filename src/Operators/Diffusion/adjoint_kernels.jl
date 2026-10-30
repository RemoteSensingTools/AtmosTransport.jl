# ---------------------------------------------------------------------------
# Adjoint of CS implicit vertical diffusion.
#
# Reverse-mode of `ImplicitVerticalDiffusion`. Builds the same Thomas-
# solve coefficients as the forward pass on the fly, then runs the
# transpose solve (upper-triangular sweep first, then back-substitution).
# Mass-aware: enters and exits in lambda-on-tracer-mass space.
#
# The CS adjoint kernel transposes the TM5-style mass-flux forward
# coefficients. The forward+adjoint share the same `(dkg, m_k, dt)`
# ingredients; `a_T[k] = c[k-1]` and `c_T[k] = a[k+1]` reduce to mass-flux
# entries with the "other layer's" m as normalizer. See the file-top doc of
# `src/Operators/Diffusion/diffusion_kernels.jl` for the forward derivation
# and `docs/src/theory/adjoint_status.md` for the adjoint support status.
# ---------------------------------------------------------------------------

# Mass-flux adjoint: transpose of the forward `Ã = M⁻¹·A·M` (on VMR),
# which is equivalent to the column-stochastic A on tracer mass. The
# transposed tridiagonal has:
#
#     a_T[k] = c[k-1]    super-of-row-(k-1) becomes sub-of-row-k
#     b_T[k] = b[k]
#     c_T[k] = a[k+1]    sub-of-row-(k+1) becomes super-of-row-k
#
# Substituting the forward coefficients
#     a[k]   = -dt·dkg[k-½]/m_k
#     c[k]   = -dt·dkg[k+½]/m_k
# at the transposed indices gives the adjoint entries in terms of the
# `dkg` evaluated at the same interface but with the OTHER layer's
# air mass as the normalizer:
#     a_T[k] = c[k-1] = -dt·dkg[k-½]/m_{k-1}      (note m_{k-1}, not m_k)
#     c_T[k] = a[k+1] = -dt·dkg[k+½]/m_{k+1}      (note m_{k+1}, not m_k)
#     b_T[k] = 1 + dt·(dkg[k-½] + dkg[k+½])/m_k
@kernel function _vertical_diffusion_cs_single_adjoint_kernel!(
    lambda, @Const(air_mass), kz_field, @Const(dz), w_scratch,
    dt, Nz::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    FT = eltype(lambda)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        dt_ft = FT(dt)

        # Scalar-carry pattern matching the forward kernel: amortize
        # global-memory loads of (Kz, dz, air_mass) across consecutive
        # vertical levels. Without this, every iteration re-loaded the
        # k-1 and k+1 neighbors, tripling the load count on the column.
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
            a_T = zero(FT)
            c_T = zero(FT)

            if k > 1
                dkg_above = _kz_exchange(m_prev, m_k, Kz_prev, Kz_k, dz_prev, dz_k)
                inv_m_prev = _safe_inverse_mass(m_prev)
                a_T = -dt_ft * dkg_above * inv_m_prev
            end

            if k < Nz
                Kz_next = field_value(kz_field, (ii, jj, k + 1))
                dz_next = dz[ii, jj, k + 1]
                m_next  = air_mass[i, j, k + 1]
                dkg_below = _kz_exchange(m_k, m_next, Kz_k, Kz_next, dz_k, dz_next)
                inv_m_next = _safe_inverse_mass(m_next)
                c_T = -dt_ft * dkg_below * inv_m_next
            end

            inv_m_k = _safe_inverse_mass(m_k)
            b_T = one(FT) + dt_ft * (dkg_above + dkg_below) * inv_m_k
            d_k = m_k > zero(FT) ? m_k * lambda[i, j, k] : zero(FT)

            if k == 1
                denom = b_T
                w_k = c_T / denom
                g_k = d_k / denom
            else
                denom = b_T - a_T * w_prev
                w_k = c_T / denom
                g_k = (d_k - a_T * g_prev) / denom
            end

            w_scratch[ii, jj, k] = w_k
            lambda[i, j, k] = g_k

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
            lambda[i, j, k] = lambda[i, j, k] -
                              w_scratch[ii, jj, k] * lambda[i, j, k + 1]
        end

        for k in 1:Nz
            m_div = air_mass[i, j, k]
            lambda[i, j, k] = m_div > zero(FT) ? lambda[i, j, k] / m_div : zero(FT)
        end
    end
end

# Transpose each partition using its smaller weight directly. This preserves
# both constant mass-objective seeds and weak recipient sensitivities.
@inline function _dkg_transpose_partition(seed, neighbor_seed, ratio)
    iszero(ratio) && return seed
    if ratio > one(ratio)
        fraction = _dkg_small_fraction(ratio)
        return neighbor_seed + fraction * (seed - neighbor_seed)
    else
        fraction = _dkg_small_fraction(ratio)
        return seed + fraction * (neighbor_seed - seed)
    end
end

@kernel function _vertical_diffusion_cs_single_dkg_adjoint_kernel!(
    lambda, @Const(air_mass), dkg_field, w_scratch,
    dt, Nz::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    FT = eltype(lambda)
    i, j = ii + Hp, jj + Hp
    @inbounds begin
        coupling = zero(FT)
        previous_seed = zero(FT)
        for k in 1:Nz
            m = air_mass[i, j, k]
            next_m = k < Nz ? air_mass[i, j, k + 1] : zero(FT)
            exchange = k < Nz ? FT(dt) * field_value(dkg_field, (ii, jj, k)) : zero(FT)
            forward_ratio, backward_ratio, coupling = _dkg_transfer_ratios(m, next_m, exchange, coupling)
            w_scratch[ii, jj, k] = forward_ratio
            seed = m > zero(FT) ? lambda[i, j, k] : zero(FT)
            previous_seed = _dkg_transpose_partition(seed, previous_seed, backward_ratio)
            lambda[i, j, k] = previous_seed
        end
        next_seed = zero(FT)
        for k in Nz:-1:1
            next_seed = _dkg_transpose_partition(lambda[i, j, k], next_seed, w_scratch[ii, jj, k])
            lambda[i, j, k] = air_mass[i, j, k] > zero(FT) ? next_seed : zero(FT)
        end
    end
end
