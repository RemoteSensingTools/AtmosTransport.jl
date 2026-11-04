# CS-only vector-harmonic evaluation. Scalar wind coefficients cancel terms
# of order one to recover U=O(cos(phi)^2) for a zonal rotation. Evaluating the
# latitude derivative first avoids that subtraction in the selected precision.
function cs_vector_legendre(rings, T)
    nr = length(rings.latitudes)
    derivative = [zeros(T-m+1,nr) for m in 0:T]
    divided = [zeros(T-m+1,nr) for m in 0:T]
    return cs_vector_legendre!(derivative, divided, rings, T)
end

function cs_vector_legendre!(derivative, divided, rings, T)
    for (j,phi) in enumerate(rings.latitudes)
        s, c = sincos(phi)
        diagonal, ddiagonal = 1., 0.
        for m in 0:T
            if m > 0
                a = sqrt((2m+1)/(2m))
                diagonal, ddiagonal = a*c*diagonal, a*(c*ddiagonal-s*diagonal)
            end
            p0, d0 = 0., 0.
            p1, d1 = diagonal, ddiagonal
            for n in m:T
                if n == m+1
                    a = sqrt(2m+3.)
                    p0, d0, p1, d1 = p1, d1, a*s*p1, a*(c*p1+s*d1)
                elseif n > m+1
                    a = sqrt((4n*n-1)/(n*n-m*m))
                    b = sqrt((2n+1)*((n-1)^2-m*m)/((2n-3)*(n*n-m*m)))
                    p0, d0, p1, d1 = p1, d1, a*s*p1-b*p0, a*(c*p1+s*d1)-b*d0
                end
                derivative[m+1][n-m+1,j] = d1
                divided[m+1][n-m+1,j] = m == 0 ? 0. : p1/c
            end
        end
    end
    return derivative, divided
end

function cs_polar_synthesis(s, T; batch=s.plan.batch, rings=s.plan.rings, threshold=0.025, tables=nothing)
    p = s.plan
    # eps(Float32)/cos(phi)^2 bounds the cancellation scale. Use the stable
    # formula in both precisions so comparisons exercise the same evaluation.
    rows = findall(phi -> abs(cos(phi)) < threshold, rings.latitudes)
    isempty(rows) && return nothing
    r = SynthesisRings(rings.latitudes[rows],rings.nlon[rows],rings.origins[rows])
    dp, pc = tables === nothing ? cs_vector_legendre(r,T) : tables
    FT = eltype(p.coeffs)
    wrap(plan) = s isa CPURingSynthesis ? CPURingSynthesis(plan) : GPUSpectralSynthesis{FT,typeof(plan)}(plan)
    derivative = wrap(RingSynthesisPlan(r,T,2batch,FT; adapter=p.adapter,legendre=tables === nothing ? p.adapter.(FT.(a) for a in dp) : dp))
    divided = wrap(RingSynthesisPlan(r,T,2batch,FT; adapter=p.adapter,legendre=tables === nothing ? p.adapter.(FT.(a) for a in pc) : pc))
    ids = vcat([collect(rings.offsets[j]:rings.offsets[j+1]-1) for j in rows]...)
    cosine = vcat([fill(FT(cos(rings.latitudes[j])),rings.nlon[j]) for j in rows]...)
    samples = ntuple(_ -> similar(p.coeffs, FT, r.offsets[end]-1, 2batch), 2)
    return (; derivative,divided,indices=p.adapter(ids),cosine=p.adapter(cosine),samples)
end

@kernel function cs_replace_polar_winds!(U,V,derivative,divided,indices,cosine,nf)
    i,k = @index(Global,NTuple)
    if i <= length(indices) && k <= nf
        c = indices[i]
        U[c,k] = cosine[i]*(derivative[i,k]-divided[i,nf+k])
        V[c,k] = cosine[i]*(-derivative[i,nf+k]-divided[i,k])
    end
end

cs_stable_polar_winds!(U,V,vo,d,R,::Nothing; coefficients=nothing) = nothing
function cs_stable_polar_winds!(U,V,vo,d,R,polar; coefficients=nothing)
    p = polar.derivative.plan
    T, nf, FT = p.T, p.batch÷2, eltype(p.coeffs)
    potentials, longitude = coefficients === nothing ?
        cs_polar_coefficients(get_backend(U), vo, d, R, p) : coefficients
    dd = synthesize_rings(potentials,polar.derivative; out=polar.samples[1])
    ll = synthesize_rings(longitude,polar.divided; out=polar.samples[2])
    backend = get_backend(U)
    cs_replace_polar_winds!(backend,256)(U,V,dd,ll,polar.indices,polar.cosine,nf;ndrange=(length(polar.indices),nf))
    synchronize(backend)
    return nothing
end

function cs_polar_coefficients(::KernelAbstractions.CPU, vo, d, R, p)
    T, nf, FT = p.T, p.batch÷2, eltype(p.coeffs)
    # Coefficients are host inputs already; conversion precedes transfer.
    potentials = zeros(Complex{FT},T+1,T+1,2nf)
    longitude = similar(potentials)
    fill!(longitude,0)
    for k in 1:nf, m in 0:T, n in max(1,m):T
        scale = FT(R)/(FT(n)*FT(n+1))
        potentials[n+1,m+1,k] = scale*Complex{FT}(vo[n+1,m+1,k])
        potentials[n+1,m+1,nf+k] = scale*Complex{FT}(d[n+1,m+1,k])
        longitude[n+1,m+1,k] = complex(zero(FT),FT(m))*potentials[n+1,m+1,k]
        longitude[n+1,m+1,nf+k] = complex(zero(FT),FT(m))*potentials[n+1,m+1,nf+k]
    end
    return potentials, longitude
end

function cs_polar_coefficients(backend, vo, d, R, p)
    FT = eltype(p.coeffs)
    potentials = similar(p.coeffs, Complex{FT}, p.T+1, p.T+1, p.batch)
    longitude = similar(potentials)
    cs_polar_coefficients_kernel!(backend, 256)(potentials, longitude,
        p.adapter(Complex{FT}.(vo)), p.adapter(Complex{FT}.(d)), FT(R), p.T, p.batch÷2;
        ndrange=(p.T+1, p.T+1, p.batch÷2))
    synchronize(backend)
    return potentials, longitude
end

@kernel function cs_polar_coefficients_kernel!(potentials, longitude, vo, d, R, T, nf)
    ni, mi, k = @index(Global, NTuple)
    n, m = ni-1, mi-1
    if ni <= T+1 && mi <= T+1 && k <= nf
        FT = typeof(R)
        v, div = zero(eltype(potentials)), zero(eltype(potentials))
        if n >= max(1, m)
            scale = R/(FT(n)*FT(n+1))
            v, div = scale*vo[ni, mi, k], scale*d[ni, mi, k]
        end
        potentials[ni, mi, k], potentials[ni, mi, nf+k] = v, div
        phase = complex(zero(FT), FT(m))
        longitude[ni, mi, k], longitude[ni, mi, nf+k] = phase*v, phase*div
    end
end
