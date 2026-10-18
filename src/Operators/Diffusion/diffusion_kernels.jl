# ---------------------------------------------------------------------------
# Implicit vertical diffusion: a matrix solve for the mixing ratio and a
# flux-form update of the tracer mass.
#
# Backward Euler for one column of Nz layers (k = 1 at the model top), with air
# mass m_k [kg] and interface exchange D_{k+½} [kg s⁻¹] between layers k and
# k+1:
#
#     m_k (q′_k − q_k) = F_{k−½} − F_{k+½},   F_{k+½} = Δt D_{k+½} (q′_k − q′_{k+1}),
#
# with no flux through the model top or the surface (F_½ = F_{Nz+½} = 0).
# For a cell-centred Kz [m² s⁻¹] and layer thickness dz [m],
#
#     D_{k+½} = (m_k + m_{k+1}) (Kz_k + Kz_{k+1}) / (dz_k + dz_{k+1})²,
#
# the mass-flux form of TM5's `TM5_Diff_Matrix`
# (`deps/tm5-cy3-4dvar/base/src/tm5_diff.F90:36-129`); a precomputed TM5 `dkg`
# field supplies D_{k+½} directly. Dividing by m_k gives the tridiagonal system
# A q′ = q for the new mixing ratio,
#
#     a_k = −Δt D_{k−½}/m_k,   b_k = 1 + Δt (D_{k−½} + D_{k+½})/m_k,   c_k = −Δt D_{k+½}/m_k.
#
# The Thomas algorithm solves it for the departure g = q′ − c from the column's
# smallest mixing ratio c. Each row of A sums to one, so a uniform column is
# stationary and the shift is exact; the elimination then works on the small
# anomaly instead of a 400 ppm background (the earlier Float32 fix for
# cancellation). The previous geometric form D = Kz/(dz_k dz_iface) (git
# history at 400410b~) conserved Σ q·dz, not Σ m·q.
#
# Updating tracer mass. Row k of the system times m_k states that
# m_k (g_k + c) = rm_k + F_{k−½} − F_{k+½} with F_{k+½} = Δt D_{k+½} (g_k − g_{k+1}),
# so in exact arithmetic writing back rm′_k = m_k (g_k + c) conserves the column.
# In Float32 it does not: the computed g is the exact solution of a matrix whose
# entries were rounded, among them b_k = fl(1 + x_k) with
# x_k = Δt (D_{k−½} + D_{k+½})/m_k ≪ 1. In the pure-pressure levels near the
# model top, x_k is the same in every column (m_k and D_{k±½} scale with the
# cell area, which cancels), so the rounding of b_k has the same sign in every
# column, and the anomaly g ≥ 0. Writing back m_k (g_k + c) then changes the
# global mass by a bias of one sign on every solve: −2.0e-6 of a tracer per day
# in the Float32 lat-lon L72 golden run, which applies about 376 solves a day
# (finding OF-113).
#
# The tracer mass is therefore updated in flux form: from the solved g, each
# interface flux F_{k+½} is computed once and applied with opposite signs to its
# two layers,
#
#     rm′_k = (rm_k + F_{k−½}) − F_{k+½}.
#
# The column total telescopes: it changes only through the rounding of the
# 2 Nz additions, about half an ulp of each layer. (Adding the net flux
# F_{k−½} − F_{k+½} in one step instead rounds a small net change into a
# large background, which in the golden run lost 1.7e-7 of the CO₂ per day
# without the ledger, against 9.4e-9 for the two additions.) Where the fluxes
# keep one
# direction that rounding can still keep one sign, so the column mass ledger
# with persistent carry of the conservative `dkg` solve returns it (see
# `_diffusion_write_back!`). In the golden run above, diffusion's own change of
# each total falls from −2.0e-6 (matrix form) to about 1e-8 per day (flux form)
# and below 5e-10 per day (flux form with the ledger). The solve for g, and
# therefore the fluxes, are those of the matrix form, and the two forms agree in
# exact arithmetic, so the transpose in `src/Adjoints/DiffusionAdjoint.jl` is
# unchanged. Layers without exchange keep their mass bit for bit. A layer with
# no air (m_k = 0) keeps the absorbing convention of the matrix form: it is
# emptied.
#
# The VMR-level reference solve (`apply_vertical_diffusion!`, mixing ratio in
# and out) keeps the matrix form q′ = g + c; with no tracer mass to update
# there is nothing to telescope.
#
# One work item owns one column. It computes the tracer-independent factors
# once (`exchange`: D_{k+½}, `factors`: the Thomas ratios w_k = c_k/denom_k) and
# then solves each tracer in turn: forward elimination into the column scratch
# `elimination`, back substitution with the flux-form update. Packed tracers
# therefore give the same values as separate single-tracer solves.
# ---------------------------------------------------------------------------

"""
    KzExchange(kz, dz)

Interface exchange of a column built from a cell-centred `Kz` field [m² s⁻¹]
and the layer thickness `dz` [m]:
`D_{k+½} = (m_k + m_{k+1}) (Kz_k + Kz_{k+1}) / (dz_k + dz_{k+1})²`.
"""
struct KzExchange{K, Z}
    kz :: K
    dz :: Z
end

"""
    DkgExchange(dkg)

Interface exchange read from a precomputed TM5 `dkg` field: entry `k` is the
exchange [kg s⁻¹] between layers `k` and `k+1`.
"""
struct DkgExchange{K}
    dkg :: K
end

Adapt.@adapt_structure KzExchange
Adapt.@adapt_structure DkgExchange

# What the solved array holds, and so how the solution is written back.
# `_MixingRatioMatrixForm`: mixing ratios in and out, q′ = g + c (reference).
# `_TracerMassFluxForm`: tracer mass in and out, rm′_k = (rm_k + F_{k−½}) − F_{k+½}.
struct _MixingRatioMatrixForm end
struct _TracerMassFluxForm end

# Column accessors. `I` indexes the column in the tracer and air-mass arrays
# (halo offset included on cubed-sphere panels), `J` in the interior-shaped
# workspace arrays and fields. A tracer array has one more dimension than the
# air mass when it packs tracers; `t` is ignored otherwise.
@inline _tracer_value(x::AbstractArray{T, N}, I::NTuple{M, Int}, k, t) where {T, N, M} =
    _tracer_value(Val(N - M), x, I, k, t)
@inline _tracer_value(::Val{1}, x, I, k, t) = @inbounds x[I..., k]
@inline _tracer_value(::Val{2}, x, I, k, t) = @inbounds x[I..., k, t]

@inline _set_tracer_value!(x::AbstractArray{T, N}, I::NTuple{M, Int}, k, t, value) where {T, N, M} =
    _set_tracer_value!(Val(N - M), x, I, k, t, value)
@inline _set_tracer_value!(::Val{1}, x, I, k, t, value) = (@inbounds x[I..., k] = value; nothing)
@inline _set_tracer_value!(::Val{2}, x, I, k, t, value) = (@inbounds x[I..., k, t] = value; nothing)

@inline _mixing_ratio(::_MixingRatioMatrixForm, x, m, I, k, t) = _tracer_value(x, I, k, t)

@inline function _mixing_ratio(::_TracerMassFluxForm, x, m, I, k, t)
    FT = eltype(x)
    m_k = @inbounds m[I..., k]
    return m_k > zero(FT) ? _tracer_value(x, I, k, t) / m_k : zero(FT)
end

# Coefficients (a_k, b_k, c_k) of row k from the exchanges above and below the
# layer. Factor and solve passes share this so they agree bit for bit.
@inline function _diffusion_coefficients(m_k, exchange_above, exchange_below, dt, k, Nz)
    FT = typeof(m_k)
    inv_m_k = m_k > zero(FT) ? one(FT) / m_k : zero(FT)
    a_k = k > 1  ? -dt * exchange_above * inv_m_k : zero(FT)
    c_k = k < Nz ? -dt * exchange_below * inv_m_k : zero(FT)
    b_k = one(FT) + dt * (exchange_above + exchange_below) * inv_m_k
    return a_k, b_k, c_k
end

# Store D_{k+½} and w_k of layer k; returns w_k for the next layer.
@inline function _store_diffusion_factor!(exchange, factors, J, k, Nz, m_k, exchange_above,
                                          exchange_below, w_prev, dt)
    a_k, b_k, c_k = _diffusion_coefficients(m_k, exchange_above, exchange_below, dt, k, Nz)
    denom = k == 1 ? b_k : b_k - a_k * w_prev
    w_k = c_k / denom
    @inbounds exchange[J..., k] = exchange_below
    @inbounds factors[J..., k] = w_k
    return w_k
end

"""
    _diffusion_factors!(exchange, factors, source, m, dt, I, J, Nz)

Tracer-independent factors of one column: `exchange[J..., k] = D_{k+½}` (zero
below the surface layer) and the Thomas ratios `factors[J..., k] = c_k/denom_k`.
The Kz form carries the neighbouring layer's Kz, dz and m in registers, so each
value is loaded once.
"""
@inline function _diffusion_factors!(exchange, factors, source::KzExchange, m, dt, I, J, Nz)
    FT = eltype(factors)
    @inbounds begin
        Kz_k = field_value(source.kz, (J..., 1))
        dz_k = source.dz[J..., 1]
        m_k  = m[I..., 1]
        exchange_above = zero(FT)
        w_prev = zero(FT)
        for k in 1:Nz
            exchange_below = zero(FT)
            Kz_next = zero(FT)
            dz_next = zero(FT)
            m_next  = zero(FT)
            if k < Nz
                Kz_next = field_value(source.kz, (J..., k + 1))
                dz_next = source.dz[J..., k + 1]
                m_next  = m[I..., k + 1]
                sum_dz = dz_k + dz_next
                exchange_below = (m_k + m_next) * (Kz_k + Kz_next) / (sum_dz * sum_dz)
            end
            w_prev = _store_diffusion_factor!(exchange, factors, J, k, Nz, m_k, exchange_above,
                                              exchange_below, w_prev, dt)
            exchange_above = exchange_below
            Kz_k = Kz_next
            dz_k = dz_next
            m_k  = m_next
        end
    end
    return nothing
end

@inline function _diffusion_factors!(exchange, factors, source::DkgExchange, m, dt, I, J, Nz)
    FT = eltype(factors)
    @inbounds begin
        exchange_above = zero(FT)
        w_prev = zero(FT)
        for k in 1:Nz
            exchange_below = k < Nz ? field_value(source.dkg, (J..., k)) : zero(FT)
            w_prev = _store_diffusion_factor!(exchange, factors, J, k, Nz, m[I..., k],
                                              exchange_above, exchange_below, w_prev, dt)
            exchange_above = exchange_below
        end
    end
    return nothing
end

# Forward elimination of tracer `t` on the departure from the column minimum c.
# The matrix form keeps the eliminated values in the tracer array; the flux
# form keeps them in `elimination`, because the tracer mass is still needed.
@inline _store_eliminated!(::_MixingRatioMatrixForm, x, elimination, I, J, k, t, g) =
    _set_tracer_value!(x, I, k, t, g)
@inline _store_eliminated!(::_TracerMassFluxForm, x, elimination, I, J, k, t, g) =
    (@inbounds elimination[J..., k] = g; nothing)
"""
    _diffusion_solve!(form, x, elimination, exchange, factors, m, dt, I, J, Nz, t,
                      carry, slot)

Solve one column of tracer `t` with the factors of `_diffusion_factors!` and
write the result back in `form` (see the file header). The flux form keeps the
column ledger with carry slot `slot` of `carry` (`nothing`: no carry).
"""
@inline function _diffusion_solve!(form, x, elimination, exchange, factors, m, dt, I, J, Nz, t,
                                   carry, slot)
    FT = eltype(x)
    @inbounds begin
        cref = _mixing_ratio(form, x, m, I, 1, t)
        for k in 2:Nz
            v = _mixing_ratio(form, x, m, I, k, t)
            cref = v < cref ? v : cref
        end
        exchange_above = zero(FT)
        w_prev = zero(FT)
        g_prev = zero(FT)
        for k in 1:Nz
            exchange_below = k < Nz ? exchange[J..., k] : zero(FT)
            a_k, b_k, _ = _diffusion_coefficients(m[I..., k], exchange_above, exchange_below,
                                                  dt, k, Nz)
            denom = k == 1 ? b_k : b_k - a_k * w_prev
            d_k = _mixing_ratio(form, x, m, I, k, t) - cref
            g_k = k == 1 ? d_k / denom : (d_k - a_k * g_prev) / denom
            _store_eliminated!(form, x, elimination, I, J, k, t, g_k)
            w_prev = factors[J..., k]
            g_prev = g_k
            exchange_above = exchange_below
        end
        _diffusion_write_back!(form, x, elimination, exchange, factors, m, dt, I, J, Nz, t, cref,
                               carry, slot)
    end
    return nothing
end

# Matrix form: back substitution in place, then restore the column minimum.
@inline function _diffusion_write_back!(form::_MixingRatioMatrixForm, x, elimination, exchange,
                                        factors, m, dt, I, J, Nz, t, cref, carry, slot)
    @inbounds for k in (Nz - 1):-1:1
        _set_tracer_value!(x, I, k, t, _tracer_value(x, I, k, t) -
                                       factors[J..., k] * _tracer_value(x, I, k + 1, t))
    end
    for k in 1:Nz
        _set_tracer_value!(x, I, k, t, _tracer_value(x, I, k, t) + cref)
    end
    return nothing
end

# Flux form: back substitution from the surface up. Layer L receives
# F_{L−½} = Δt D_{L−½} (g_{L−1} − g_L) as soon as g_{L−1} is known; its lower
# flux F_{L+½} is the upper flux of the layer below, so both layers apply the
# same rounded value.
#
# Column mass ledger (as for the conservative `dkg` solve): the flux form
# changes the column total only by the rounding of the two additions per
# layer, but where the fluxes keep one direction that rounding can keep one sign
# (a flux below half an ulp of the giving layer and above half an ulp of the
# receiving one moves mass that only one side records). The compensated column
# sums before and after give the residual, which is added, with the column's
# carry, to the largest exchanging layer (`_ledger_deposit`); the part that
# layer cannot hold stays in `carry` for the next solve. Layers without
# exchange stay bit-exact and receive nothing. A column with an empty layer
# keeps the absorbing convention and its carry and is not ledgered.
@inline function _diffusion_write_back!(form::_TracerMassFluxForm, x, elimination, exchange,
                                        factors, m, dt, I, J, Nz, t, cref, carry, slot)
    FT = eltype(x)
    @inbounds begin
        g_layer = elimination[J..., Nz]
        flux_below = zero(FT)
        exchange_below = zero(FT)
        before, after = (zero(FT), zero(FT)), (zero(FT), zero(FT))
        positive_carrier = true
        kmax, largest, exchanging = 0, zero(FT), 0
        for L in Nz:-1:1
            g_above, exchange_above, flux_above = zero(FT), zero(FT), zero(FT)
            if L > 1
                g_above = elimination[J..., L - 1] - factors[J..., L - 1] * g_layer
                exchange_above = exchange[J..., L - 1]
                flux_above = dt * exchange_above * (g_above - g_layer)
            end
            m_L = m[I..., L]
            old = _tracer_value(x, I, L, t)
            new = m_L > zero(FT) ? (old + flux_above) - flux_below : zero(FT)
            _set_tracer_value!(x, I, L, t, new)
            before = _neumaier_add(before..., old)
            after = _neumaier_add(after..., new)
            positive_carrier &= m_L > zero(FT)
            if !(iszero(dt * exchange_above) && iszero(dt * exchange_below))
                exchanging += 1
                abs(new) > largest && ((kmax, largest) = (L, abs(new)))
            end
            g_layer = g_above
            flux_below = flux_above
            exchange_below = exchange_above
        end
        if positive_carrier && kmax > 0
            value, column_carry = _ledger_deposit(
                _tracer_value(x, I, kmax, t), _ledger_residual(before, after, largest, exchanging),
                _load_ledger_carry(carry, FT, J..., slot), exchanging)
            _set_tracer_value!(x, I, kmax, t, value)
            _store_ledger_carry!(carry, column_carry, J..., slot)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Kernels: one work item per column; `Nt` tracers share the column's factors
# (`Nt = 1` for a single-tracer array). `form` selects the write-back. Tracer
# `t` uses ledger carry slot `first_slot + t − 1` of `carry` (`nothing`: no
# carry): 1 for packed tracers, the tracer's index for a single-tracer array.
#
# `air_mass` is deliberately not marked `@Const`. On the CPU backend `@Const`
# wraps the kernel in `Base.Experimental.@aliasscope`, and with Julia 1.12.6 at
# -O2 the back substitution then read the `elimination` scratch as it was
# before the forward elimination of the same work item had stored to it, so the
# solve left the tracer mass unchanged (correct at -O0 and -O1, and without
# `@Const`).
# ---------------------------------------------------------------------------

# Structured lat-lon: tracers (Nx, Ny, Nz, Nt), air mass and workspace (Nx, Ny, Nz).
@kernel function _vertical_diffusion_structured_kernel!(form, x, air_mass, source,
                                                        exchange, factors, elimination, carry,
                                                        dt, Nz::Int, Nt::Int, first_slot::Int)
    i, j = @index(Global, NTuple)
    I = (i, j)
    _diffusion_factors!(exchange, factors, source, air_mass, dt, I, I, Nz)
    for t in 1:Nt
        _diffusion_solve!(form, x, elimination, exchange, factors, air_mass, dt, I, I, Nz, t,
                          carry, first_slot + t - 1)
    end
end

# Face-indexed reduced Gaussian: tracers (ncells, Nz, Nt) or (ncells, Nz).
@kernel function _vertical_diffusion_face_kernel!(form, x, air_mass, source,
                                                  exchange, factors, elimination, carry,
                                                  dt, Nz::Int, Nt::Int, first_slot::Int)
    c = @index(Global, Linear)
    I = (c,)
    _diffusion_factors!(exchange, factors, source, air_mass, dt, I, I, Nz)
    for t in 1:Nt
        _diffusion_solve!(form, x, elimination, exchange, factors, air_mass, dt, I, I, Nz, t,
                          carry, first_slot + t - 1)
    end
end

# Cubed sphere, all six panels in one launch: the third index selects the panel.
# Arguments are six-panel tuples (`carry` a tuple of panel carries or of
# `nothing`s); tracer and air-mass panels are halo-padded, workspace panels and
# fields interior-shaped.
@kernel function _vertical_diffusion_panels_kernel!(form, x, air_mass, source,
                                                    exchange, factors, elimination, carry,
                                                    dt, Nz::Int, Nt::Int, Hp::Int,
                                                    first_slot::Int)
    ii, jj, p = @index(Global, NTuple)
    x_p, m_p = select_panel(x, p), select_panel(air_mass, p)
    exchange_p, factors_p = select_panel(exchange, p), select_panel(factors, p)
    elimination_p, carry_p = select_panel(elimination, p), select_panel(carry, p)
    I = (ii + Hp, jj + Hp)
    J = (ii, jj)
    _diffusion_factors!(exchange_p, factors_p, select_panel(source, p), m_p, dt, I, J, Nz)
    for t in 1:Nt
        _diffusion_solve!(form, x_p, elimination_p, exchange_p, factors_p, m_p, dt, I, J, Nz, t,
                          carry_p, first_slot + t - 1)
    end
end

# ---------------------------------------------------------------------------
# Mass <-> mixing-ratio scaling of cubed-sphere panels, for VMR-level reference
# comparisons of tracer-mass states.
# ---------------------------------------------------------------------------

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
