# Explicit payload-only evidence for synthetic writer/lifecycle fixtures.
function payload_only_evidence(::Type{G}, ::Type{FT}) where {G,FT}
    P = AtmosTransport.Preprocessing
    return P.FluxFidelityEvidenceBundle(
        P.NoFluxFidelityEvidence{G,FT,P.SourcePayloadOnly}(P.SourcePayloadOnly()),
        P.NoSourceOmegaEvidence{G,FT,P.SourceHasNoOmega}(P.SourceHasNoOmega()),
        P.NoColumnClosureRecord())
end
