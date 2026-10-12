# ---------------------------------------------------------------------------
# Convective mass fluxes on the binary's air-mass basis.
#
# Met centres archive convective mass fluxes per unit of TOTAL (moist) air:
# ERA5 UDMF/DDMF/UDRF/DDRF, the TM5-archive eu/du/ed/dd derived from them,
# and GEOS CMFMC/DTRAIN. The convection operators divide each flux by the
# runtime air mass (`TM5Convection`: entu / m, ...), so the rate is right
# only when flux and mass are on the same basis. A dry-basis binary
# therefore carries dry-air fluxes, F_dry = (1 - q) F_moist, with the
# specific humidity q taken where the flux lives: at the layer centre for
# centre fields, and at the interface for interface fields by the rule of
# `interface_specific_humidity`. GEOS (`_moist_to_dry_cmfmc!`) and ERA5 use
# the same interface rule.
# ---------------------------------------------------------------------------

"""
    interface_specific_humidity(k, Nz, q...) -> q̂_k
    interface_specific_humidity(k, Nz, i, j, q...) -> q̂_k

Specific humidity at interface `k` of a top-first column of `Nz` layers
(interface `k` is the top of layer `k`; `k = 1` is the model top and
`k = Nz + 1` the surface), the mean of the adjacent layer centres:

```math
\\hat q_k = \\tfrac12 (q_{k-1} + q_k) \\quad (2 \\le k \\le N_z), \\qquad
\\hat q_1 = q_1, \\qquad \\hat q_{N_z+1} = q_{N_z},
```

where `q_k` is the value at the centre of layer `k` and `q̂_k` the value at
interface `k`.

Each `q` is a column of layer-centre values indexed by level; with several
columns (time samples, e.g. the start and end of a window) the result is also
averaged over them, in the order `q[1][k-1], q[1][k], q[2][k-1], ...`, which
is the four-corner mean of GEOS `CMFMC` (`_moist_to_dry_cmfmc!`).

The `(i, j, q...)` method applies the identical rule directly to columns of
three-dimensional arrays without constructing a view in the GEOS grid loop.
"""
@inline function interface_specific_humidity(k::Integer, Nz::Integer,
                                             q::AbstractVector{FT},
                                             qs::AbstractVector{FT}...) where FT
    above = max(k - 1, 1)
    below = min(k, Nz)
    s = zero(FT)
    n = 0
    @inbounds for column in (q, qs...)
        s += column[above]
        n += 1
        if below != above
            s += column[below]
            n += 1
        end
    end
    return s / n
end


@inline function interface_specific_humidity(k::Integer, Nz::Integer,
                                             i::Integer, j::Integer,
                                             q::AbstractArray{FT, 3},
                                             qs::AbstractArray{FT, 3}...) where FT
    above = max(k - 1, 1)
    below = min(k, Nz)
    s = zero(FT)
    n = 0
    @inbounds for field in (q, qs...)
        s += field[i, j, above]
        n += 1
        if below != above
            s += field[i, j, below]
            n += 1
        end
    end
    return s / n
end

"""
    ecmwf_convection_to_basis!(basis, udmf, ddmf, udrf, ddrf, q) -> nothing

Put one column of ECMWF convective fields (top-first, `Nz = length(q)`
layers) on the air-mass `basis` of the transport binary before
[`ec2tm_from_rates!`](@ref) turns them into TM5 entrainment and detrainment.

- `udmf`, `ddmf` (length `Nz + 1`): updraft and downdraft mass fluxes at the
  interfaces, kg m⁻² s⁻¹; interface `k` is the top of layer `k`.
- `udrf`, `ddrf` (length `Nz`): updraft and downdraft detrainment rates at
  the layer centres, kg m⁻³ s⁻¹.
- `q` (length `Nz`): specific humidity at the layer centres, kg kg⁻¹.

ECMWF fluxes are per unit of total (moist) air. For `DryBasis()` they become
dry-air fluxes,

```math
M^{\\mathrm{dry}}_k = (1 - \\hat q_k)\\, M_k \\;\\; (1 \\le k \\le N_z + 1), \\qquad
D^{\\mathrm{dry}}_k = (1 - q_k)\\, D_k \\;\\; (1 \\le k \\le N_z),
```

for the interface fluxes `M` (udmf, ddmf) and the centre rates `D` (udrf,
ddrf), with the interface humidity `q̂_k` of
[`interface_specific_humidity`](@ref), so the TM5 rates `entu / m_dry` equal TM5's own `entu / m_moist`. Converting
the inputs keeps the ec2tm budget `entu − detu = udmf[k] − udmf[k+1]` exact.
`MoistBasis()` leaves the fields unchanged. ec2tm's noise thresholds then act
on the converted values.
"""
ecmwf_convection_to_basis!(::MoistBasis, udmf, ddmf, udrf, ddrf, q) = nothing

function ecmwf_convection_to_basis!(::DryBasis,
                                    udmf::AbstractVector{FT}, ddmf::AbstractVector{FT},
                                    udrf::AbstractVector{FT}, ddrf::AbstractVector{FT},
                                    q::AbstractVector{FT}) where {FT <: AbstractFloat}
    Nz = length(q)
    length(udmf) == length(ddmf) == Nz + 1 ||
        throw(DimensionMismatch("udmf/ddmf lengths $(length(udmf))/$(length(ddmf)) ≠ Nz+1 = $(Nz + 1)"))
    length(udrf) == length(ddrf) == Nz ||
        throw(DimensionMismatch("udrf/ddrf lengths $(length(udrf))/$(length(ddrf)) ≠ Nz = $Nz"))
    @inbounds for k in 1:(Nz + 1)
        dry = one(FT) - interface_specific_humidity(k, Nz, q)
        udmf[k] *= dry
        ddmf[k] *= dry
    end
    @inbounds for k in 1:Nz
        dry = one(FT) - q[k]
        udrf[k] *= dry
        ddrf[k] *= dry
    end
    return nothing
end

"""
    tm5_convection_to_basis!(basis, entu, detu, entd, detd, q) -> nothing

Put one top-first column of TM5 layer-centre convection fields (kg m⁻² s⁻¹,
per unit of total air, as in the TM5 meteo archive) on the air-mass `basis`.
For `MoistBasis()` nothing changes.

For `DryBasis()` the draft mass fluxes are rebuilt from the TM5 budgets,
converted where they live, and the entrainment re-derived from the converted
budgets, as [`ecmwf_convection_to_basis!`](@ref) followed by ec2tm would do:

```math
\\begin{aligned}
M^u_{k} &= M^u_{k+1} + e^u_k - d^u_k, \\quad M^u_{N_z+1} = 0
  &&\\text{(updraft, upward, top interface of layer } k)\\\\
M^d_{k+1} &= M^d_{k} + e^d_k - d^d_k, \\quad M^d_{1} = 0
  &&\\text{(downdraft, downward, bottom interface)}\\\\
d'_k &= (1 - q_k)\\, d_k, \\qquad M'_k = (1 - \\hat q_k)\\, M_k\\\\
e'^u_k &= d'^u_k + M'^u_{k} - M'^u_{k+1}, \\qquad
e'^d_k = d'^d_k + M'^d_{k+1} - M'^d_{k}
\\end{aligned}
```

with the interface humidity `q̂_k` of
[`interface_specific_humidity`](@ref). Scaling every
field by `1 − q_k` instead would break the column closure
`Σ e = Σ d`, because a draft entrains moist air low down and detrains drier
air aloft. Each draft is converted only over its active span (the levels
where its entrainment or detrainment is non-zero), so no rounding-level
values appear above the cloud top, where `TM5Convection` diagnoses the cloud
dimensions. A negative re-derived rate is moved to its partner rate as in
TM5's `ECconv_to_TMconv`, which keeps `e − d` and so the fluxes.
"""
tm5_convection_to_basis!(::MoistBasis, entu, detu, entd, detd, q) = nothing

function tm5_convection_to_basis!(::DryBasis,
                                  entu::AbstractVector{FT}, detu::AbstractVector{FT},
                                  entd::AbstractVector{FT}, detd::AbstractVector{FT},
                                  q::AbstractVector{FT}) where {FT <: AbstractFloat}
    Nz = length(q)
    length(entu) == length(detu) == length(entd) == length(detd) == Nz ||
        throw(DimensionMismatch("TM5 fields must have length Nz = $Nz"))
    _tm5_updraft_to_dry!(entu, detu, q, Nz)
    _tm5_downdraft_to_dry!(entd, detd, q, Nz)
    return nothing
end

# Active span of one draft: first and last level with non-zero entrainment or
# detrainment (0:-1 when the draft is absent).
function _tm5_draft_span(ent, det, Nz)
    top = findfirst(k -> ent[k] != 0 || det[k] != 0, 1:Nz)
    top === nothing && return 1:0
    bottom = findlast(k -> ent[k] != 0 || det[k] != 0, 1:Nz)
    return top:bottom
end

# Updraft: upward flux rebuilt from the bottom interface of the span, where
# it is zero; `flux` is the interface below layer k, `flux_top` interface k.
function _tm5_updraft_to_dry!(ent::AbstractVector{FT}, det::AbstractVector{FT},
                              q::AbstractVector{FT}, Nz::Int) where FT
    span = _tm5_draft_span(ent, det, Nz)
    flux, flux_dry = zero(FT), zero(FT)
    @inbounds for k in reverse(span)
        flux_top = flux + ent[k] - det[k]
        flux_top_dry = (one(FT) - interface_specific_humidity(k, Nz, q)) * flux_top
        det[k] *= one(FT) - q[k]
        ent[k] = det[k] + flux_top_dry - flux_dry
        flux, flux_dry = flux_top, flux_top_dry
    end
    foreach(k -> _redistribute_negative_tm5_pair!(ent, det, k), span)
    return nothing
end

# Downdraft: downward flux rebuilt from the top interface of the span, where
# it is zero; `flux` is interface k (above layer k), `flux_bottom` interface k + 1.
function _tm5_downdraft_to_dry!(ent::AbstractVector{FT}, det::AbstractVector{FT},
                                q::AbstractVector{FT}, Nz::Int) where FT
    span = _tm5_draft_span(ent, det, Nz)
    flux, flux_dry = zero(FT), zero(FT)
    @inbounds for k in span
        flux_bottom = flux + ent[k] - det[k]
        flux_bottom_dry = (one(FT) - interface_specific_humidity(k + 1, Nz, q)) * flux_bottom
        det[k] *= one(FT) - q[k]
        ent[k] = det[k] + flux_bottom_dry - flux_dry
        flux, flux_dry = flux_bottom, flux_bottom_dry
    end
    foreach(k -> _redistribute_negative_tm5_pair!(ent, det, k), span)
    return nothing
end

# TM5 `ECconv_to_TMconv` (phys_convec_ec2tm.F90): a negative entrainment is
# added to the detrainment and zeroed, and vice versa; `ent − det` is kept.
# Returns which of the pair was negative.
@inline function _redistribute_negative_tm5_pair!(ent::AbstractVector{FT},
                                                  det::AbstractVector{FT}, k) where FT
    ent_negative = ent[k] < zero(FT)
    if ent_negative
        det[k] -= ent[k]
        ent[k] = zero(FT)
    end
    det_negative = det[k] < zero(FT)
    if det_negative
        ent[k] -= det[k]
        det[k] = zero(FT)
    end
    return (ent_negative, det_negative)
end

# Record the air-mass basis of the TM5 convection sections in a binary
# header (`TM5_CONVECTION_MASS_BASIS_KEY`); `nothing` when the binary carries
# no TM5 sections, or copies sections whose basis is not recorded.
_with_tm5_convection_basis(header::AbstractDict, ::Nothing) = header

function _with_tm5_convection_basis(header::AbstractDict, basis::AbstractMassBasis)
    header[TM5_CONVECTION_MASS_BASIS_KEY] = String(mass_basis_symbol(basis))
    return header
end
