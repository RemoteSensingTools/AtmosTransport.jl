"""Version-5 construction identity; the default writer remains version 4."""
function tm5_construction_metadata(c::TM5SpectralFluxes,T::Integer)
    # Apply the same TM5-0 format gate as the generic RG writer.
    TM5SpectralFluxesHeader(tm5_construction_revision(c),T,c.refinement_steps,c.vod2uv_radius)
    return Dict{String,Any}("format_version"=>5,"flux_construction"=>"tm5_spectral",
        "flux_construction_revision"=>tm5_construction_revision(c),"spectral_truncation_used"=>Int(T),
        "face_quadrature"=>face_quadrature_name(c.face_quadrature),
        "quadrature_refinement_steps"=>c.refinement_steps,"vod2uv_radius_m"=>c.vod2uv_radius)
end

# The analytic/Gauss tests exercise production integration. No independent
# spectral re-evaluator is attached to the runtime certificate yet.
function tm5_flux_evidence(::Type{G},::Type{FT}) where {G,FT}
    return FluxFidelityEvidenceBundle(
        NoFluxFidelityEvidence{G,FT,EvidenceNotAttached}(EvidenceNotAttached()),
        NoSourceOmegaEvidence{G,FT,SourceHasNoOmega}(SourceHasNoOmega()),NoColumnClosureRecord())
end

tm5_construction_revision(::TM5SpectralFluxes{GaussFourier}) = 2
tm5_construction_revision(::TM5SpectralFluxes{TM5Trapezoid}) = 1
