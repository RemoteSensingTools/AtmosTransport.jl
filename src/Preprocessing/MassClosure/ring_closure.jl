"""
    project_mean_zero!(v)

Subtract the mean of `v` in-place. Used to keep Conjugate-Gradient
iterates in the range of the singular graph Laplacian (which has a
1-dimensional constant null space on the CS and reduced-Gaussian graphs).
"""
@inline function project_mean_zero!(v::AbstractVector{Float64})
    s = sum(v) / length(v)
    @. v -= s
    return v
end

function recompute_faceindexed_cm_from_divergence!(cm::AbstractMatrix{FT},
                                                   hflux::AbstractMatrix{FT},
                                                   face_left::Vector{Int32},
                                                   face_right::Vector{Int32},
                                                   div_scratch::AbstractMatrix{Float64};
                                                   B_ifc::Vector{<:Real}) where FT
    nc = size(cm, 1)
    Nz = size(cm, 2) - 1
    length(B_ifc) == Nz + 1 || throw(ArgumentError(
        "recompute_faceindexed_cm_from_divergence!: B_ifc must contain Nz + 1 = $(Nz + 1) interfaces; got $(length(B_ifc))"))
    fill!(cm, zero(FT))
    fill!(div_scratch, 0.0)

    @inbounds for k in 1:Nz, f in eachindex(face_left)
        flux = Float64(hflux[f, k])
        left = Int(face_left[f])
        right = Int(face_right[f])
        left > 0 && (div_scratch[left, k] += flux)
        right > 0 && (div_scratch[right, k] -= flux)
    end

    @inbounds for c in 1:nc
        pit = 0.0
        for k in 1:Nz
            pit += div_scratch[c, k]
        end
        acc = 0.0
        for k in 1:Nz
            acc = acc - div_scratch[c, k] + (Float64(B_ifc[k + 1]) - Float64(B_ifc[k])) * pit
            cm[c, k + 1] = FT(acc)
        end
    end
    return nothing
end
