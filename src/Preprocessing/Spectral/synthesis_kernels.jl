@kernel function pack_ring_coefficients!(packed, spec, m, T, nf)
    q, k = @index(Global, NTuple)
    if k <= nf && q <= T-m+1
        z = spec[q+m,m+1,k]
        packed[k,q] = real(z)
        packed[nf+k,q] = imag(z)
    end
end

@kernel function scatter_ring_spectrum!(half, sums, rows, phases, m, nf, T, J)
    r, k = @index(Global, NTuple)
    if k <= nf && r <= length(rows)
        j = rows[r]
        if m <= min(T,(J-1)÷2)
            z = complex(sums[k,j], sums[nf+k,j])
            half[m+1,k,r] = z * phases[m+1,j]
        end
    end
end

@kernel function gather_ring_samples!(out, samples, rows, offsets, J, nf)
    i,k,r = @index(Global, NTuple)
    if i <= J && k <= nf && r <= length(rows)
        out[offsets[rows[r]]+i-1,k] = samples[i,k,r]
    end
end

@kernel function complete_wind_spectral_kernel!(u,v,divergence,vo,d,T,R,nf)
    ni,mi,k = @index(Global,NTuple)
    n=ni-1
    m=mi-1
    if ni<=T+2 && mi<=T+2 && k<=nf
        z=zero(eltype(u))
        uu=z
        vv=z
        dd=z
        if m<=min(n,T)
            FT=typeof(real(z))
            lower_vo=n>m ? vo[n,mi,k] : z
            lower_d=n>m ? d[n,mi,k] : z
            upper_vo=n<T ? vo[n+2,mi,k] : z
            upper_d=n<T ? d[n+2,mi,k] : z
            center_vo=n<=T ? vo[ni,mi,k] : z
            center_d=n<=T ? d[ni,mi,k] : z
            delta=n==0 ? zero(FT) : -sqrt(FT(n*n-m*m)/FT(4n*n-1))/FT(n)
            next_delta=-sqrt(FT((n+1)^2-m*m)/FT(4(n+1)^2-1))/FT(n+1)
            sigma=n==0 ? zero(FT) : -FT(m)/(FT(n)*FT(n+1))
            uu=FT(R)*(delta*lower_vo+complex(zero(FT),sigma)*center_d-next_delta*upper_vo)
            vv=FT(R)*(-delta*lower_d+complex(zero(FT),sigma)*center_vo+next_delta*upper_d)
            dd=center_d
        end
        u[ni,mi,k]=uu
        v[ni,mi,k]=vv
        divergence[ni,mi,k]=dd
    end
end
