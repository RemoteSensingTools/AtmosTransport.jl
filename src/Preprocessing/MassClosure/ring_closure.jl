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
