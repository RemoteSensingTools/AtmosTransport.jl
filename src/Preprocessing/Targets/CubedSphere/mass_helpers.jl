# Shared CS endpoint mass construction and dry-mass pins. Precision policies stay separate.

"""
    ps_from_air_mass!(ps, m, area, g, Nc, Nz)

Set `ps[i,j] = (Σ_k m[i,j,k]) · g / area[i,j]` (Pa). Used to keep the
binary's stored `ps` consistent with the chained pressure-fixer mass.
"""
function ps_from_air_mass!(ps::AbstractMatrix{FT},
                            m::AbstractArray{FT, 3},
                            cell_areas::AbstractMatrix{FT},
                            g::FT, Nc::Int, Nz::Int) where {FT}
    @inbounds for j in 1:Nc, i in 1:Nc
        s = zero(FT)
        for k in 1:Nz
            s += m[i, j, k]
        end
        ps[i, j] = s * g / cell_areas[i, j]
    end
    return ps
end

function cs_total_air_mass(panels_m::NTuple{CS_PANEL_COUNT, <:AbstractArray})
    total = 0.0
    for p in 1:CS_PANEL_COUNT
        total += sum(Float64, panels_m[p])
    end
    return total
end


"""
    pin_cs_global_air_mass!(panels_m, panels_qv, cell_areas, g, target_kg, dB)

Apply a uniform moist-surface-pressure offset to a cubed-sphere dry-air mass
state so its global mass equals `target_kg`. For layer `k`, the correction is

```math
\\Delta m_k = \\Delta B_k\\,\\Delta p_s\\,(1-q_k)\\,A/g.
```

This is the same hybrid-coordinate correction as the spectral path: layers
with `\\Delta B_k = 0` are unchanged. `panels_qv` and `dB` are on the same
target vertical grid as `panels_m`.
"""
function pin_cs_global_air_mass!(panels_m::NTuple{CS_PANEL_COUNT, <:AbstractArray{FT, 3}},
                                  panels_qv::NTuple{CS_PANEL_COUNT, <:AbstractArray{<:Real, 3}},
                                  cell_areas::AbstractMatrix,
                                  g::FT,
                                  target_kg::Real,
                                  dB::AbstractVector) where FT
    current = cs_total_air_mass(panels_m)
    Nz = size(panels_m[1], 3)
    length(dB) == Nz || throw(DimensionMismatch(
        "native dry-mass pin dB length $(length(dB)) does not match Nz=$Nz"))
    sensitivity_kg_per_pa = 0.0
    @inbounds for p in 1:CS_PANEL_COUNT
        m = panels_m[p]
        qv = panels_qv[p]
        size(qv) == size(m) || throw(DimensionMismatch(
            "native dry-mass pin qv panel $p shape $(size(qv)) does not match mass $(size(m))"))
        for k in 1:Nz, j in axes(m, 2), i in axes(m, 1)
            q = clamp(Float64(qv[i, j, k]), 0.0, MAX_SPECIFIC_HUMIDITY)
            sensitivity_kg_per_pa += Float64(dB[k]) * (1.0 - q) *
                                     Float64(cell_areas[i, j]) / Float64(g)
        end
    end
    delta_ps = mass_target_offset(target_kg, current, sensitivity_kg_per_pa)

    @inbounds for p in 1:CS_PANEL_COUNT
        m = panels_m[p]
        qv = panels_qv[p]
        for k in 1:Nz, j in axes(m, 2), i in axes(m, 1)
            q = clamp(Float64(qv[i, j, k]), 0.0, MAX_SPECIFIC_HUMIDITY)
            delta_m = Float64(dB[k]) * delta_ps * (1.0 - q) *
                      Float64(cell_areas[i, j]) / Float64(g)
            m[i, j, k] = FT(Float64(m[i, j, k]) + delta_m)
        end
    end

    final = cs_total_air_mass(panels_m)
    return (before_kg = current,
            after_kg = final,
            target_kg = Float64(target_kg),
            delta_ps_pa = delta_ps,
            residual_kg = final - Float64(target_kg))
end

"""
    pin_cs_global_air_mass_from_sensitivity!(panels_m, sensitivity, target_kg)

Apply the global pin using a precomputed extensive sensitivity
`dm/dps = dB * (1-q) * area/g` in kg/Pa. Building this field on native levels
and summing it through a vertical transform preserves humidity covariance
inside merged hybrid layers.
"""
function pin_cs_global_air_mass_from_sensitivity!(
        panels_m::NTuple{CS_PANEL_COUNT, <:AbstractArray{FT, 3}},
        sensitivity::NTuple{CS_PANEL_COUNT, <:AbstractArray{<:Real, 3}},
        target_kg::Real) where FT
    for p in 1:CS_PANEL_COUNT
        size(sensitivity[p]) == size(panels_m[p]) || throw(DimensionMismatch(
            "native dry-mass pin sensitivity panel $p shape $(size(sensitivity[p])) " *
            "does not match mass $(size(panels_m[p]))"))
    end
    current = cs_total_air_mass(panels_m)
    sensitivity_kg_per_pa = cs_total_air_mass(sensitivity)
    delta_ps = mass_target_offset(target_kg, current, sensitivity_kg_per_pa)
    for p in 1:CS_PANEL_COUNT
        @. panels_m[p] += FT(delta_ps) * sensitivity[p]
    end
    final = cs_total_air_mass(panels_m)
    return (before_kg = current,
            after_kg = final,
            target_kg = Float64(target_kg),
            delta_ps_pa = delta_ps,
            residual_kg = final - Float64(target_kg))
end

"""
    derive_c180_dry_mass!(m_dry, ps_dry, ps_dry_acc,
                           ps_panels, qv_panels, vc, cell_areas; grav=STANDARD_GRAVITY) -> nothing

Cubed-sphere dry-mass construction in Float64. Builds the dry-air
layer mass and dry surface pressure for each of
the 6 C-tier panels from the regridded moist PS + Q on C180. Same formula
as the GEOS endpoint dry-mass derivation so `Σ_k DELP_dry = PS_dry` to
roundoff on the target mesh too.

Inputs and outputs are `NTuple{6, ...}` panel tuples; `ps_dry_acc` is a
panel-tuple Float64 accumulator (so a Float32 output preserves the
multi-level summation precision). `cell_areas[i, j]` is shared across all
6 panels (the CS mesh is isotropic per panel).
"""
function derive_c180_dry_mass!(m_dry::NTuple{6, AbstractArray{<:Real, 3}},
                                 ps_dry::NTuple{6, AbstractMatrix{<:Real}},
                                 ps_dry_acc::NTuple{6, Matrix{Float64}},
                                 ps_panels::NTuple{6, AbstractMatrix{<:Real}},
                                 qv_panels::NTuple{6, AbstractArray{<:Real, 3}},
                                 vc::HybridSigmaPressure,
                                 cell_areas::AbstractMatrix{<:Real};
                                 grav::Real = STANDARD_GRAVITY)
    Nc, _, Nz = size(m_dry[1])
    length(vc.A) == length(vc.B) == Nz + 1 ||
        throw(DimensionMismatch("hybrid A/B length $(length(vc.A))/$(length(vc.B)) ≠ Nz+1 = $(Nz + 1)"))
    size(cell_areas) == (Nc, Nc) ||
        throw(DimensionMismatch("cell_areas $(size(cell_areas)) ≠ (Nc=$Nc, Nc)"))

    A = vc.A
    B = vc.B
    inv_g = 1.0 / Float64(grav)

    @inbounds for p in 1:6
        fill!(ps_dry_acc[p], 0.0)
        # k outer / i,j inner — column-major over (Nc, Nc, Nz).
        for k in 1:Nz
            dA = Float64(A[k + 1]) - Float64(A[k])
            dB = Float64(B[k + 1]) - Float64(B[k])
            for j in 1:Nc, i in 1:Nc
                ps_total   = Float64(ps_panels[p][i, j])
                area       = Float64(cell_areas[i, j])
                delp_full  = dA + dB * ps_total
                delp_dry_k = (1.0 - Float64(qv_panels[p][i, j, k])) * delp_full
                FT = eltype(m_dry[p])
                m_dry[p][i, j, k]    = FT(delp_dry_k * area * inv_g)
                ps_dry_acc[p][i, j] += delp_dry_k
            end
        end
        FT = eltype(ps_dry[p])
        for j in 1:Nc, i in 1:Nc
            ps_dry[p][i, j] = FT(ps_dry_acc[p][i, j])
        end
    end
    return nothing
end
