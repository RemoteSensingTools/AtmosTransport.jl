"""Direct vector-harmonic reference at arbitrary points, entirely in eltype(U)."""
@kernel function cs_exact_point_kernel!(U,V,H,vo,d,lnsp,longitude,latitude,T,R)
    i,k = @index(Global,NTuple)
    if i <= length(latitude) && k <= size(U,2)
        FT = eltype(U)
        s,c = sincos(latitude[i])
        ud,vd,hh = zero(FT),zero(FT),zero(FT)
        pu,pv,ph = zero(FT),zero(FT),zero(FT)
        diagonal,ddiagonal = one(FT),zero(FT)
        for m in 0:T
            if m > 0
                a = sqrt(FT(2m+1)/FT(2m))
                diagonal,ddiagonal = a*c*diagonal,a*(c*ddiagonal-s*diagonal)
            end
            sn,cs = sincos(FT(m)*longitude[i])
            phase = complex(cs,sn)
            factor = m == 0 ? one(FT) : FT(2)
            p0,d0 = zero(FT),zero(FT)
            p1,d1 = diagonal,ddiagonal
            for n in m:T
                if n == m+1
                    a = sqrt(FT(2m+3))
                    p0,d0,p1,d1 = p1,d1,a*s*p1,a*(c*p1+s*d1)
                elseif n > m+1
                    a = sqrt(FT(4n*n-1)/FT(n*n-m*m))
                    b = sqrt(FT((2n+1)*((n-1)^2-m*m))/FT((2n-3)*(n*n-m*m)))
                    p0,d0,p1,d1 = p1,d1,a*s*p1-b*p0,a*(c*p1+s*d1)-b*d0
                end
                # Compensated real sums; every scalar, including recurrence
                # constants and trig arguments, has the selected precision.
                term = factor*real(lnsp[n+1,m+1]*phase)*p1-ph
                total = hh+term
                ph = (total-hh)-term
                hh = total
                if n > 0
                    scale = R/(FT(n)*FT(n+1))
                    zonal = complex(zero(FT),FT(m))*p1
                    uu = factor*scale*real((c*d1*vo[n+1,m+1,k]-zonal*d[n+1,m+1,k])*phase)-pu
                    vv = factor*scale*real((-c*d1*d[n+1,m+1,k]-zonal*vo[n+1,m+1,k])*phase)-pv
                    totalu,totalv = ud+uu,vd+vv
                    pu,pv = (totalu-ud)-uu,(totalv-vd)-vv
                    ud,vd = totalu,totalv
                end
            end
        end
        U[i,k],V[i,k],H[i,k] = ud,vd,hh
    end
end

"""Batched, bounded-memory exact samples; usable with any KA array adapter."""
function cs_exact_samples(H,vo,d,T,R,lon,lat; precision=Float64,adapter=identity)
    FT = precision
    u = adapter(zeros(FT,length(lat),size(vo,3)))
    v,h = similar(u),similar(u)
    backend = get_backend(u)
    cs_exact_point_kernel!(backend,64)(u,v,h,adapter(Complex{FT}.(vo)),adapter(Complex{FT}.(d)),
        adapter(Complex{FT}.(H)),adapter(FT.(lon)),adapter(FT.(lat)),T,FT(R);ndrange=size(u))
    synchronize(backend)
    return u,v,h
end
