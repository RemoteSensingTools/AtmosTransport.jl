# Stage input coefficients exactly as the runtime does, on the tested backend.
function diffusion_coefficients(reference, A, B)
    ref = reference isa Tuple ? reference[1] : reference
    FT = eltype(ref)
    D = AtmosTransport.Operators.Diffusion
    storage = D.HybridCoefficientWorkspace(similar(ref, FT, length(A)),
                                           similar(ref, FT, length(B)))
    return D.stage_hybrid_coefficients!(storage, A, B)
end
