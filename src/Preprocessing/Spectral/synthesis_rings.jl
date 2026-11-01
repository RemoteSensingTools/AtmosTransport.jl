"""Longitude rings in radians, with variable lengths and arbitrary origins."""
struct SynthesisRings
    latitudes::Vector{Float64}
    nlon::Vector{Int}
    origins::Vector{Float64}
    offsets::Vector{Int}
    function SynthesisRings(latitudes, nlon, origins)
        length(latitudes) == length(nlon) == length(origins) || throw(DimensionMismatch("ring geometry"))
        all(>(0), nlon) || throw(ArgumentError("ring lengths must be positive"))
        all(x -> abs(x) <= pi/2, latitudes) || throw(ArgumentError("ring latitudes must be in radians"))
        new(Float64.(latitudes), Int.(nlon), Float64.(origins), cumsum([1; Int.(nlon)]))
    end
end

"""Number of intervals in TM5 ShRefinement (grid_interpol.F90:1602–1630)."""
quadrature_refinement(T::Integer, degrees::Real, steps::Integer=2) =
    max(1, ceil(Int, steps * abs(degrees) * (2(T+1)) / 360))

"""FFT ring length retaining every mode, including the T+1 wind/gradient row."""
function oversampled_ring_length(K::Integer, T::Integer)
    J = Int(K)
    while J < 2T+1
        J *= 2
    end
    return J
end

"""Exact longitude and cos-scaled latitude gradients, through degree T+1."""
function lnsp_gradient(H::AbstractMatrix{ComplexF64}, T::Integer)
    zonal = zeros(ComplexF64, T+2, T+2)
    meridional = similar(zonal)
    fill!(meridional, 0)
    for m in 0:T, n in m:(T+1)
        n <= T && (zonal[n+1,m+1] = im*m*H[n+1,m+1])
        upper = n < T ? H[n+2,m+1] : 0im
        lower = n > m ? H[n,m+1] : 0im
        # eps(m,n) = -n*delta(m,n); correct minus sign at the top edge.
        meridional[n+1,m+1] = -(n+2)*(n+1)*_delta(m,n+1)*upper +
                              (n-1)*n*_delta(m,n)*lower
    end
    return zonal, meridional
end

"""Complete vector transform; unlike the legacy transform, retains degree T+1."""
function complete_wind_coefficients(vo, d, T::Integer, radius::Real=IFS_EARTH_RADIUS)
    nf = size(vo, 3)
    u = zeros(ComplexF64, T+2, T+2, nf)
    v = similar(u)
    fill!(v, 0)
    for k in 1:nf
        # Preserve the validated production formula inside [m,T].
        uu = zeros(ComplexF64,T+1,T+1); vv = similar(uu)
        vod2uv!(uu, vv, view(vo,:,:,k), view(d,:,:,k), Int(T))
        u[1:T+1,1:T+1,k] .= uu .* (radius/IFS_EARTH_RADIUS)
        v[1:T+1,1:T+1,k] .= vv .* (radius/IFS_EARTH_RADIUS)
        for m in 0:T
            u[T+2,m+1,k] = radius*_delta(m,T+1)*vo[T+1,m+1,k]
            v[T+2,m+1,k] = -radius*_delta(m,T+1)*d[T+1,m+1,k]
        end
    end
    return u,v
end
