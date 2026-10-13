# ---------------------------------------------------------------------------
# Adjoint of the FV3 vertical sweep (scalar_profile, kord = 8)
#
# The forward sweep in `vertical_fv3_profile.jl` maps the tracer mass of a
# column, for fixed air mass m and interface fluxes F, as
#
#   q̄ = S rm,  S = diag(1 / m),            layer means
#   e = T q̄,   T = A(m)⁻¹ B(m),            compact edges (Thomas sweep)
#   (q_L, q_R, q_6)_k = P_k(β) x_k,        limited parabola of layer k
#   f_e = F_e^d c(α_e) · profile_donor,     interface flux
#   rm'_k = rm_k + f_k − f_{k+1},
#
# with the local inputs x_k = (e_k, e_{k+1}, q̄_{k−2}, …, q̄_{k+2}). Every
# limiter step selects one of a few linear forms of x_k; its comparisons choose
# the branch β. For fixed β the sweep is linear, rm' = W(β) rm, and the reverse
# pass applies
#
#   λ_rm = λ' + S (Tᵀ λ_e + λ_q̄),  (λ_e, λ_q̄) = P(β)ᵀ Cᵀ Dᵀ λ',
#
# the transpose at the branches the forward took (frozen-limiter adjoint).
#
# The branches and the 3 × 7 matrices P_k(β) come from the forward code itself:
# `_fv3_edge`, `_fv3_layer_profile` and `_fv3_interface_flux` are evaluated on a
# `_LinearForm`, a value with its partial derivatives with respect to x_k. Its
# arithmetic on the value is the Float arithmetic of the forward, so every
# comparison, and with it β, is the forward's; the partials are the derivative
# of the transcribed `fv_mapz.F90` code along that branch.
# ---------------------------------------------------------------------------

"""
    _LinearForm{T, N}(value, partials)

A number carrying its partial derivatives with respect to `N` local inputs.
Arithmetic on `value` is exactly the Float arithmetic of the forward code, so a
function evaluated on `_LinearForm`s takes the forward's branches; `partials`
is its derivative along those branches. Only the operations the FV3 profile
code uses are defined.
"""
struct _LinearForm{T <: AbstractFloat, N}
    value    :: T
    partials :: NTuple{N, T}
end

_LinearForm{T, N}(x::Real) where {T, N} = _LinearForm{T, N}(T(x), ntuple(_ -> zero(T), Val(N)))

const _LF = _LinearForm

@inline Base.zero(::Type{_LF{T, N}}) where {T, N} = _LF{T, N}(zero(T))
@inline Base.one(::Type{_LF{T, N}}) where {T, N}  = _LF{T, N}(one(T))
@inline Base.zero(a::_LF) = zero(typeof(a))

@inline Base.:+(a::_LF, b::_LF) = _LF(a.value + b.value, map(+, a.partials, b.partials))
@inline Base.:-(a::_LF, b::_LF) = _LF(a.value - b.value, map(-, a.partials, b.partials))
@inline Base.:-(a::_LF) = _LF(-a.value, map(-, a.partials))
@inline Base.:+(a::_LF, b::Real) = _LF(a.value + b, a.partials)
@inline Base.:+(a::Real, b::_LF) = _LF(a + b.value, b.partials)
@inline Base.:-(a::_LF, b::Real) = _LF(a.value - b, a.partials)
@inline Base.:-(a::Real, b::_LF) = _LF(a - b.value, map(-, b.partials))
@inline Base.:*(a::Real, b::_LF) = _LF(a * b.value, map(p -> a * p, b.partials))
@inline Base.:*(a::_LF, b::Real) = _LF(a.value * b, map(p -> p * b, a.partials))
@inline Base.:*(a::_LF, b::_LF) =
    _LF(a.value * b.value, map((pa, pb) -> pa * b.value + a.value * pb, a.partials, b.partials))
@inline Base.:/(a::_LF, b::Real) = _LF(a.value / b, map(p -> p / b, a.partials))
@inline function Base.:/(a::_LF, b::_LF)
    v = a.value / b.value
    return _LF(v, map((pa, pb) -> (pa - v * pb) / b.value, a.partials, b.partials))
end
# Float `x^2` is `x * x`; keep the same rounding.
@inline Base.literal_pow(::typeof(^), a::_LF, ::Val{2}) = a * a
@inline Base.abs(a::_LF) = _LF(abs(a.value), ifelse(signbit(a.value), map(-, a.partials), a.partials))

# `min`/`max` return the forward's value; on an exact tie the derivative is the
# first argument's (a kink, where either one-sided derivative is valid).
@inline Base.min(a::_LF, b::_LF) = _LF(min(a.value, b.value), ifelse(b.value < a.value, b.partials, a.partials))
@inline Base.max(a::_LF, b::_LF) = _LF(max(a.value, b.value), ifelse(b.value > a.value, b.partials, a.partials))

@inline Base.:<(a::_LF, b::_LF)   = a.value < b.value
@inline Base.:<(a::_LF, b::Real)  = a.value < b
@inline Base.:<(a::Real, b::_LF)  = a < b.value
@inline Base.:<=(a::_LF, b::_LF)  = a.value <= b.value
@inline Base.:<=(a::_LF, b::Real) = a.value <= b
@inline Base.:<=(a::Real, b::_LF) = a <= b.value

# ---------------------------------------------------------------------------
# Linearized column: the FV3 profile of one layer as a function of its inputs
# ---------------------------------------------------------------------------

# Number of local inputs of a layer's limited parabola: its two edges and the
# means of the five layers k − 2 … k + 2.
const _FV3_LOCAL_INPUTS = 7

"""
    FV3LinearizedColumn(scratch, ii, jj, k)

The column `FV3Column(scratch, ii, jj, 1)` seen from layer `k`: `_edge` and
`_qbar` return `_LinearForm`s whose partials are unit vectors in the local
inputs `x_k = (e_k, e_{k+1}, q̄_{k−2}, q̄_{k−1}, q̄_k, q̄_{k+1}, q̄_{k+2})`.
`_fv3_layer_profile(col, k, Nz, profile)` on it returns layer `k`'s limited
parabola together with its 3 × 7 Jacobian at the forward's limiter branches.
"""
struct FV3LinearizedColumn{A} <: AbstractFV3Column
    scratch :: A
    ii :: Int
    jj :: Int
    k  :: Int
end

@inline _unit_partials(::Type{T}, slot) where {T} =
    ntuple(n -> ifelse(n == slot, one(T), zero(T)), Val(_FV3_LOCAL_INPUTS))

@inline _primal(c::FV3LinearizedColumn) = FV3Column(c.scratch, c.ii, c.jj, 1)

Base.@propagate_inbounds _edge(c::FV3LinearizedColumn, e) =
    _LF(_edge(_primal(c), e), _unit_partials(eltype(c.scratch), 1 + (e - c.k)))
Base.@propagate_inbounds _qbar(c::FV3LinearizedColumn, e) =
    _LF(_qbar(_primal(c), e), _unit_partials(eltype(c.scratch), 5 + (e - c.k)))

# ---------------------------------------------------------------------------
# Adjoint column storage
# ---------------------------------------------------------------------------

# Slices of the adjoint column scratch: the forward replay (`FV3Column` with
# t = 1: edges, elimination factors, layer means) and the adjoints of the
# unlimited edges and of the layer means.
const _FV3_ADJOINT_SCRATCH_SLICES = 5

Base.@propagate_inbounds _edge_bar(c::FV3Column, k) = c.scratch[c.ii, c.jj, k, 4]
Base.@propagate_inbounds _qbar_bar(c::FV3Column, k) = c.scratch[c.ii, c.jj, k, 5]
Base.@propagate_inbounds _add_edge_bar!(c::FV3Column, k, v) =
    (c.scratch[c.ii, c.jj, k, 4] += v)
Base.@propagate_inbounds _add_qbar_bar!(c::FV3Column, k, v) =
    (c.scratch[c.ii, c.jj, k, 5] += v)

"""
    _fv3_adjoint_column_scratch(prototype, Nc, Nz, needed::Bool)

Working storage of the FV3 sweep's reverse pass, `Nc × Nc × (Nz+1) × 5`, on
`prototype`'s backend; empty unless `needed`.
"""
_fv3_adjoint_column_scratch(prototype, Nc, Nz, needed::Bool) =
    needed ? similar(prototype, Nc, Nc, Nz + 1, _FV3_ADJOINT_SCRATCH_SLICES) :
             similar(prototype, 0, 0, 0, 0)

function _check_fv3_adjoint_column_scratch(scratch, Nc, Nz)
    size(scratch) == (Nc, Nc, Nz + 1, _FV3_ADJOINT_SCRATCH_SLICES) || throw(ArgumentError(
        "The FV3 vertical sweep's adjoint needs column scratch of size " *
        "$((Nc, Nc, Nz + 1, _FV3_ADJOINT_SCRATCH_SLICES)), got $(size(scratch)). " *
        "Build the adjoint workspace with `CSAdjointWorkspace(mesh, prototype; " *
        "column_scratch = true)`."))
    return nothing
end

# ---------------------------------------------------------------------------
# Transposes of the column steps
# ---------------------------------------------------------------------------

# Add `w · ∂f/∂x_kd` to the adjoints of the unlimited edges and layer means of
# donor layer `kd`. Inputs outside the column have zero partials (the profile
# of layer kd never reads them).
@inline function _add_local_adjoint!(adj::FV3Column, kd, Nz, w, d)
    @inbounds begin
        _add_edge_bar!(adj, kd, w * d[1])
        _add_edge_bar!(adj, kd + 1, w * d[2])
        kd - 2 >= 1  && _add_qbar_bar!(adj, kd - 2, w * d[3])
        kd - 1 >= 1  && _add_qbar_bar!(adj, kd - 1, w * d[4])
        _add_qbar_bar!(adj, kd, w * d[5])
        kd + 1 <= Nz && _add_qbar_bar!(adj, kd + 1, w * d[6])
        kd + 2 <= Nz && _add_qbar_bar!(adj, kd + 2, w * d[7])
    end
    return nothing
end

"""
    _fv3_unlimited_edges_adjoint!(adj::FV3Column, m, i, j, Nz)

Transpose of `_fv3_unlimited_edges!`'s Thomas sweep: adds `Tᵀ λ_e` (minus the
`S` scaling) to the layer-mean adjoints. Reads the elimination factors the
forward replay stored in `adj`'s primal slices; the forward's back substitution
is reversed first, then its elimination rows from the surface up.
"""
@inline function _fv3_unlimited_edges_adjoint!(adj::FV3Column, m, i, j, Nz)
    FT = eltype(adj.scratch)
    @inbounds begin
        for k in 1:Nz                                       # e_k ← e_k − g_k e_{k+1}
            _add_edge_bar!(adj, k + 1, -_elim(adj, k) * _edge_bar(adj, k))
        end

        d4 = _layer_mass(m, i, j, Nz - 1) / _layer_mass(m, i, j, Nz)
        a_bot = 1 + d4 * (d4 + FT(1.5))
        s = _edge_bar(adj, Nz + 1) / (d4 * (d4 + FT(0.5)) - a_bot * _elim(adj, Nz))
        _add_qbar_bar!(adj, Nz, 2 * d4 * (d4 + 1) * s)
        _add_qbar_bar!(adj, Nz - 1, s)
        _add_edge_bar!(adj, Nz, -a_bot * s)

        for k in Nz:-1:2
            d4  = _layer_mass(m, i, j, k - 1) / _layer_mass(m, i, j, k)
            bet = 2 + d4 + d4 - _elim(adj, k - 1)
            s = _edge_bar(adj, k) / bet
            _add_qbar_bar!(adj, k - 1, 3 * s)
            _add_qbar_bar!(adj, k, 3 * d4 * s)
            _add_edge_bar!(adj, k - 1, -s)
        end

        grat = _layer_mass(m, i, j, 2) / _layer_mass(m, i, j, 1)
        s = _edge_bar(adj, 1) / (grat * (grat + FT(0.5)))
        _add_qbar_bar!(adj, 1, (grat + grat) * (grat + 1) * s)
        _add_qbar_bar!(adj, 2, s)
    end
    return nothing
end

"""
Reverse of `_cs_zsweep_fv3_kernel!` for one panel and one tracer, in place on
the tracer-mass adjoint `lambda`; one work item per interior column. `rm` and
`m` are the tracer and air mass that entered the forward sweep. Each work item
reads all of its column's `lambda` before writing it; halo columns, which the
forward sweep does not change, keep their adjoint.
"""
@kernel function _cs_zsweep_fv3_adjoint_kernel!(lambda, @Const(rm), @Const(m), @Const(cm),
                                                scratch, profile, Nz, Hp, flux_scale)
    ii, jj = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(lambda)
        i = ii + Hp
        j = jj + Hp
        adj = FV3Column(scratch, ii, jj, 1)
        _fv3_unlimited_edges!(adj, rm, m, i, j, Nz)          # the forward's e, g, q̄
        for k in Int32(1):(Nz + Int32(1))
            adj.scratch[ii, jj, k, 4] = zero(FT)
            adj.scratch[ii, jj, k, 5] = zero(FT)
        end

        # Dᵀ, Cᵀ and P(β)ᵀ: each interior interface flux, through its donor layer.
        for e in Int32(2):Nz
            λ_f = lambda[i, j, e] - lambda[i, j, e - Int32(1)]
            F   = flux_scale * cm[i, j, e]
            kd  = ifelse(F >= 0, e - Int32(1), e)            # donor, as in _fv3_interface_flux
            donor = _fv3_layer_profile(FV3LinearizedColumn(scratch, ii, jj, Int(kd)), kd, Nz, profile)
            f = _fv3_interface_flux(F, donor, _layer_mass(m, i, j, e - Int32(1)),
                                    donor, _layer_mass(m, i, j, e))
            _add_local_adjoint!(adj, kd, Nz, λ_f, f.partials)
        end

        _fv3_unlimited_edges_adjoint!(adj, m, i, j, Nz)       # Tᵀ
        for k in Int32(1):Nz                                  # identity + S
            lambda[i, j, k] += _qbar_bar(adj, k) / _layer_mass(m, i, j, k)
        end
    end
end

"""
    _sweep_z_panels_fv3_adjoint!(lambda_panels, panels_rm, panels_m, panels_cm,
                                 profile, scratch, mesh; flux_scale)

Apply the transpose of one FV3-profile vertical sweep of all six panels to the
tracer-mass adjoint `lambda_panels`, in place. `panels_rm` and `panels_m` are
the tracer and air mass that entered the forward sweep, `panels_cm` and
`flux_scale` its interface fluxes; the limiter branches are the forward's at
that state. `scratch` is `_fv3_adjoint_column_scratch(…, true)`.
"""
function _sweep_z_panels_fv3_adjoint!(lambda_panels::NTuple{6}, panels_rm::NTuple{6},
                                      panels_m::NTuple{6}, panels_cm::NTuple{6},
                                      profile::FV3ScalarProfile, scratch,
                                      mesh::CubedSphereMesh;
                                      flux_scale = one(eltype(lambda_panels[1])))
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(lambda_panels[1], 3)
    Nz >= 4 || throw(ArgumentError(
        "the FV3 vertical-profile adjoint needs at least 4 layers, got Nz = $Nz."))
    FT = eltype(lambda_panels[1])
    _check_fv3_adjoint_column_scratch(scratch, Nc, Nz)
    for p in 1:6
        backend = get_backend(lambda_panels[p])
        kernel! = _cs_zsweep_fv3_adjoint_kernel!(backend, (32, 4))
        kernel!(lambda_panels[p], panels_rm[p], panels_m[p], panels_cm[p], scratch, profile,
                Int32(Nz), Int32(Hp), FT(flux_scale); ndrange = (Nc, Nc))
        synchronize(backend)
    end
    return lambda_panels
end
