# Backward Euler in tracer-mass space, with column-conservative LU factors.
# For interface k, u_k = dt D_k/m_k and d_k = dt D_k/m_{k+1}.
# Scaling the ordinary bidiagonal LU factors gives two matrices whose columns
# sum to one. Their inverses are directed retention/transfer passes:
#   e_k = 1 + d_{k-1} a_{k-1}; a_k = e_k/(e_k + u_k); b_k = 1/e_k.
# The forward pass retains a_k of the incoming mass and sends the remainder
# down; the backward pass retains b_k and sends the remainder up. This solves
# the same implicit diffusion equation without a mass/VMR round trip or a
# correction of the column total. Positive mass stays positive at safe inputs.

# Store the ratio of transferred to retained mass, rather than 1-retention.
# This preserves very weak exchange without cancellation when retention rounds
# to one, and very strong exchange without losing the small retained fraction.
@inline function _dkg_transfer_ratios(m, next_m, exchange, previous_coupling)
    FT = typeof(m)
    u = m > zero(FT) ? exchange / m : zero(FT)
    d = next_m > zero(FT) ? exchange / next_m : zero(FT)
    forward_ratio = u / (one(FT) + previous_coupling)
    return forward_ratio, previous_coupling, d / (one(FT) + forward_ratio)
end

# Each transfer keeps its rounding residual in a second scalar (`_two_sum`).
# This avoids losing small layer masses when a stiff solve carries a large
# column subtotal.
@inline function _dkg_partition(value, incoming, correction, ratio)
    amount, error = _two_sum(value, incoming)
    low = error + correction
    iszero(ratio) && return amount + low, zero(incoming), zero(correction)
    if ratio > one(ratio)
        # Retain the smaller portion directly, and carry the large remainder
        # with its rounding residual to the next layer.
        fraction = one(ratio) / (one(ratio) + ratio)
        retained = muladd(fraction, amount, fraction * low)
        remainder, error = _two_sum(amount, -retained)
        incoming, correction = _two_sum(remainder, error + low)
    else
        # Compute a weak transfer directly. Reconstructing it as amount minus
        # rounded retained mass would erase sub-ulp exchange into empty cells.
        fraction = ratio / (one(ratio) + ratio)
        incoming = fraction * amount
        correction = fma(fraction, amount, -incoming) + fraction * low
        remainder, error = _two_sum(amount, -incoming)
        retained = remainder + (error + low - correction)
    end
    return retained, incoming, correction
end

@inline function _dkg_factor_column!(factors, air_mass, dkg_field, dt, ii, jj, Nz, Hp)
    FT = eltype(air_mass)
    i, j = ii + Hp, jj + Hp
    coupling = zero(FT)
    @inbounds for k in 1:Nz
        m = air_mass[i, j, k]
        next_m = k < Nz ? air_mass[i, j, k + 1] : zero(FT)
        exchange = k < Nz ? FT(dt) * field_value(dkg_field, (ii, jj, k)) : zero(FT)
        forward_ratio, _, coupling = _dkg_transfer_ratios(m, next_m, exchange, coupling)
        factors[ii, jj, k] = forward_ratio
    end
    return nothing
end

@inline _dkg_mass_value(rm::AbstractArray{FT,3}, i, j, k, t) where FT = @inbounds rm[i,j,k]
@inline _dkg_mass_value(rm::AbstractArray{FT,4}, i, j, k, t) where FT = @inbounds rm[i,j,k,t]
@inline _set_dkg_mass!(rm::AbstractArray{FT,3}, i, j, k, t, value) where FT = (@inbounds rm[i,j,k] = value)
@inline _set_dkg_mass!(rm::AbstractArray{FT,4}, i, j, k, t, value) where FT = (@inbounds rm[i,j,k,t] = value)

@inline function _dkg_isolated_layer(dkg, dt, ii, jj, k, Nz)
    iszero(dt) && return true
    above_closed = k == 1 || iszero(dt * field_value(dkg, (ii, jj, k - 1)))
    below_closed = k == Nz || iszero(dt * field_value(dkg, (ii, jj, k)))
    return above_closed && below_closed
end

@inline function _dkg_diffuse_mass_column!(rm, air_mass, dkg_field, factors, dt, ii, jj, Nz, Hp, t)
    FT = eltype(rm)
    i, j = ii + Hp, jj + Hp
    @inbounds begin
        qmin, qmax = typemax(FT), -typemax(FT)
        positive_carrier = true
        before = (zero(FT), zero(FT))            # compensated column sum, for the ledger
        for k in 1:Nz
            m = air_mass[i, j, k]
            value = _dkg_mass_value(rm, i, j, k, t)
            before = _neumaier_add(before..., value)
            positive_carrier &= m > zero(FT)
            q = m > zero(FT) ? value / m : zero(FT)
            qmin, qmax = min(qmin, q), max(qmax, q)
        end
        # A constant mixing ratio is stationary only when all carrier masses
        # are positive. Keep the existing absorbing zero-carrier convention.
        # Use the range endpoint closest to zero. A profile spanning zero has
        # no removable background; subtracting its minimum would amplify a
        # signed solve's cancellation near the mixed equilibrium.
        cref = !positive_carrier ? zero(FT) : qmin > zero(FT) ? qmin :
               qmax < zero(FT) ? qmax : zero(FT)
        incoming, correction = zero(FT), zero(FT)
        for k in 1:Nz
            m = air_mass[i, j, k]
            if _dkg_isolated_layer(dkg_field, dt, ii, jj, k, Nz)
                # Even subtracting and restoring a background can change the
                # last bit. Preserve cells with no exchange exactly.
                m > zero(FT) || _set_dkg_mass!(rm, i, j, k, t, zero(FT))
                incoming, correction = zero(FT), zero(FT)
                continue
            end
            value = m > zero(FT) ? _dkg_mass_value(rm, i, j, k, t) - cref * m : zero(FT)
            retained, incoming, correction = _dkg_partition(
                value, incoming, correction, factors[ii, jj, k])
            _set_dkg_mass!(rm, i, j, k, t, retained)
        end
        incoming, correction = zero(FT), zero(FT)
        after, kmax, largest, exchanging = (zero(FT), zero(FT)), 0, zero(FT), 0
        for k in Nz:-1:1
            m = air_mass[i, j, k]
            if _dkg_isolated_layer(dkg_field, dt, ii, jj, k, Nz)
                after = _neumaier_add(after..., _dkg_mass_value(rm, i, j, k, t))
                incoming, correction = zero(FT), zero(FT)
                continue
            end
            # Recover the backward ratio from the preceding forward ratio;
            # no extra factor array or per-tracer column scratch is needed.
            exchange = k > 1 ? FT(dt) * field_value(dkg_field, (ii, jj, k - 1)) : zero(FT)
            d = m > zero(FT) ? exchange / m : zero(FT)
            backward_ratio = k > 1 ? d / (one(FT) + factors[ii, jj, k - 1]) : zero(FT)
            retained, incoming, correction = _dkg_partition(
                _dkg_mass_value(rm, i, j, k, t), incoming, correction, backward_ratio)
            value = m > zero(FT) ? retained + cref * m : zero(FT)
            _set_dkg_mass!(rm, i, j, k, t, value)
            after = _neumaier_add(after..., value)
            exchanging += 1
            abs(value) > largest && ((kmax, largest) = (k, abs(value)))
        end
        # Column mass ledger. The exchange form conserves the column exactly,
        # but the final `retained + cref·m` writes drop sub-ulp exchanges
        # (≈1e-7 of SF₆ per month in Float32). The residual, about one ulp of
        # the largest exchanging layer, is added back there. In about a quarter
        # of the columns it is below half an ulp and rounds away, so the bias
        # is removed statistically rather than exactly.
        # Layers without exchange stay bit-exact, and zero-carrier columns keep
        # their absorbing convention.
        if positive_carrier && kmax > 0
            _set_dkg_mass!(rm, i, j, kmax, t, _dkg_mass_value(rm, i, j, kmax, t) +
                           _ledger_residual(before, after, largest, exchanging))
        end
    end
    return nothing
end

@kernel function _vertical_diffusion_cs_mass_dkg_kernel!(rm, @Const(air_mass),
                                                        dkg_field, factors,
                                                        dt, Nz::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    _dkg_factor_column!(factors, air_mass, dkg_field, dt, ii, jj, Nz, Hp)
    _dkg_diffuse_mass_column!(rm, air_mass, dkg_field, factors, dt, ii, jj, Nz, Hp, 1)
end

@kernel function _vertical_diffusion_cs_mass_dkg_packed_kernel!(rm, @Const(air_mass),
                                                               dkg_field, factors,
                                                               dt, Nz::Int, Nt::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    _dkg_factor_column!(factors, air_mass, dkg_field, dt, ii, jj, Nz, Hp)
    for t in 1:Nt
        _dkg_diffuse_mass_column!(rm, air_mass, dkg_field, factors, dt, ii, jj, Nz, Hp, t)
    end
end

# Backend extensions may separate the tracer solves from factor construction.
# Every tracer still uses the same column arithmetic and read-only factors.
@inline _cs_dkg_mass_workgroupsize(backend, ::Type) = (8, 8)
@inline _cs_dkg_tracer_workgroupsize(backend, ::Type) = nothing

@kernel function _vertical_diffusion_cs_dkg_factors_kernel!(factors, @Const(air_mass),
                                                           dkg_field, dt,
                                                           Nz::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    _dkg_factor_column!(factors, air_mass, dkg_field, dt, ii, jj, Nz, Hp)
end

@kernel function _vertical_diffusion_cs_mass_dkg_tracers_kernel!(rm, @Const(air_mass),
                                                               dkg_field, @Const(factors),
                                                               dt, Nz::Int, Hp::Int)
    ii, jj, t = @index(Global, NTuple)
    _dkg_diffuse_mass_column!(rm, air_mass, dkg_field, factors, dt, ii, jj, Nz, Hp, t)
end
