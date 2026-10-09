"""
    AbstractCSDkgField{FT}

Cubed-sphere diffusion fields that supply interface air-mass exchange `dkg`
[kg s⁻¹] directly to the conservative implicit solver (`panel_field(f, p)`
returns a rank-3 `(Nc, Nc, Nz)` field; the last level is the zero-flux surface).
"""
abstract type AbstractCSDkgField{FT} <: AbstractCubedSphereField{FT} end

"""
    PrecomputedCSDkgField(host_cache)

Cubed-sphere interface air-mass exchange field loaded from a binary `:dkg`
payload. `dkg[i,j,k]` is the TM5 vertical-diffusion exchange rate [kg s⁻¹]
between top-down layers `k` and `k+1`; the final level is the zero-flux surface
boundary.

Unlike a Kz field, this payload already includes the preprocessing-time layer
mass and virtual-temperature geometry. The runtime must therefore consume it
directly rather than reconstructing an interface coefficient from layer-centre
Kz and a second, potentially inconsistent, `dz` profile.
"""
struct PrecomputedCSDkgField{FT, F <: PreComputedKzField{FT, 3}, H} <: AbstractCSDkgField{FT}
    panels     :: NTuple{6, F}
    host_cache :: H
end

function PrecomputedCSDkgField(host_cache::NTuple{6, Array{FT, 3}}) where FT
    panels = ntuple(p -> PreComputedKzField(host_cache[p]), 6)
    return PrecomputedCSDkgField{FT, typeof(panels[1]), typeof(host_cache)}(panels, host_cache)
end

@inline panel_field(f::PrecomputedCSDkgField, p::Integer) = f.panels[Int(p)]
update_field!(f::PrecomputedCSDkgField, ::Real) = f

function Adapt.adapt_structure(to, f::PrecomputedCSDkgField)
    panels = Adapt.adapt(to, f.panels)
    return PrecomputedCSDkgField{_precomputed_cs_dkg_eltype(f), typeof(panels[1]),
                                 typeof(f.host_cache)}(panels, f.host_cache)
end
@inline _precomputed_cs_dkg_eltype(::PrecomputedCSDkgField{FT}) where {FT} = FT

function refresh_precomputed_cs_dkg_cache!(field::PrecomputedCSDkgField, dkg_panels)
    dkg_panels === nothing && throw(ArgumentError(
        "TM5 precomputed diffusion requires a `:dkg` payload in the cubed-sphere " *
        "transport window; regenerate with include_tm5_diffusion=true."))
    @inbounds for p in 1:6
        copyto!(field.panels[p].data, dkg_panels[p])
    end
    return field
end
