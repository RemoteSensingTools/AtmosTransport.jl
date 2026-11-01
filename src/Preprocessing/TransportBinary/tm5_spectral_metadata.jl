"""Version-5 construction identity; the default writer remains version 4."""
function tm5_construction_metadata(c::TM5SpectralFluxes,T::Integer)
    # Apply the same TM5-0 format gate as the generic RG writer.
    TM5SpectralFluxesHeader(1,T,c.refinement_steps,c.vod2uv_radius)
    return Dict{String,Any}("format_version"=>5,"flux_construction"=>"tm5_spectral",
        "flux_construction_revision"=>1,"spectral_truncation_used"=>Int(T),
        "quadrature_refinement_steps"=>c.refinement_steps,"vod2uv_radius_m"=>c.vod2uv_radius)
end

# The analytic/Gauss tests exercise production integration. No independent
# spectral re-evaluator is attached to the runtime certificate yet.
function tm5_flux_evidence(::Type{G},::Type{FT}) where {G,FT}
    return FluxFidelityEvidenceBundle(
        NoFluxFidelityEvidence{G,FT,EvidenceNotAttached}(EvidenceNotAttached()),
        NoSourceOmegaEvidence{G,FT,SourceHasNoOmega}(SourceHasNoOmega()),NoColumnClosureRecord())
end
