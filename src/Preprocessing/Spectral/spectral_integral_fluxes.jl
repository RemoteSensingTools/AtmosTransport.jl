# Direct target integrals. TM5 tmm.F90:1866–1876, 2135–2147, 2167–2179,
# 2428–2473. R*Rv/g (not R²/g) follows the two-radius Gauss identity.

@kernel function surface_pressure_samples!(ps,surface)
    i = @index(Global)
    i <= length(ps) && (ps[i,1] = exp(surface[i,1]))
end

@kernel function spectral_flux_integrands!(fu,fv,forcing,U,V,D,ps,surface,
                                          offsets,lengths,latitudes,dA,dB,R,Rv,nf)
    i,row,k = @index(Global,NTuple)
    if row <= length(lengths) && i <= lengths[row] && k <= nf
        c = offsets[row]+i-1
        cosine = cos(latitudes[row])
        if abs(cosine) < 1e-14
            fu[c,k] = 0
            fv[c,k] = 0
            forcing[c,k] = 0
        else
            p = ps[c,1]
            dp = dA[k]+dB[k]*p
            u = Float64(U[c,k])
            v = Float64(V[c,k])
            fu[c,k] = (R/STANDARD_GRAVITY)*u*dp/cosine
            fv[c,k] = (R/STANDARD_GRAVITY)*v*dp
            forcing[c,k] = (R*Rv/STANDARD_GRAVITY)*Float64(D[c,k])*dp*cosine +
                (R/STANDARD_GRAVITY)*p*dB[k]/cosine*(u*surface[c,2]+v*surface[c,3])
        end
    end
end

@kernel function integrate_cell_rows!(ps_out,u_out,C_out,ps,psprefix,fu,C,Cprefix,
                                     lengths,offsets,node_offsets,node_rows,cell_offsets,nlon,
                                     weights,cosweights,areas,R,nf)
    i,j,k = @index(Global,NTuple)
    if j <= length(nlon) && i <= nlon[j] && k <= nf
        cell = cell_offsets[j]+i-1
        pressure = 0.0
        zonal = 0.0
        forcing = 0.0
        for node in node_offsets[j]:node_offsets[j+1]-1
            row = node_rows[node]
            J = lengths[row]
            start = offsets[row]
            a = Float64(i-1)*J/nlon[j]
            b = Float64(i)*J/nlon[j]
            k == 1 && (pressure += cosweights[node]*(
                ring_primitive(ps,psprefix,start,J,1,b)-ring_primitive(ps,psprefix,start,J,1,a)))
            sample = start+round(Int,a)
            zonal += weights[node]*fu[sample,k]
            forcing += weights[node]*(ring_primitive(C,Cprefix,start,J,k,b)-
                                      ring_primitive(C,Cprefix,start,J,k,a))
        end
        k == 1 && (ps_out[cell] = pressure*R^2/areas[cell])
        u_out[cell,k] = zonal
        C_out[cell,k] = forcing
    end
end

@kernel function integrate_boundary_rows!(v_out,fv,prefix,lengths,offsets,rows,counts,bounds,nf)
    i,b,k = @index(Global,NTuple)
    if b <= length(counts) && i <= counts[b] && k <= nf
        f = bounds[b]+i-1
        if b == 1 || b == length(counts)
            v_out[f,k] = 0.0
        else
            row = rows[b]
            J = lengths[row]
            start = offsets[row]
            a = Float64(i-1)*J/counts[b]
            z = Float64(i)*J/counts[b]
            v_out[f,k] = ring_primitive(fv,prefix,start,J,k,z)-ring_primitive(fv,prefix,start,J,k,a)
        end
    end
end

"""Reusable quadrature and synthesis plans. Batches bound the refined-grid working set."""
struct SpectralIntegralWorkspace{S,F,Q}
    quadrature::SpectralQuadrature
    synthesis::S
    surface::F
    geometry::Q
end

function SpectralIntegralWorkspace(grid,T,c::TM5SpectralFluxes; synthesis=CPUSynthesis(),
                                   precision=AutoSynthesisPrecision(),batch::Integer=8)
    q = SpectralQuadrature(grid,T,c.refinement_steps)
    s = build_ring_synthesis(synthesis,q.rings,T+1,batch,precision)
    surface = surface_synthesis(s,3)
    adapt = s.plan.adapter
    g = (;lengths=adapt(q.rings.nlon),offsets=adapt(q.rings.offsets),
        latitudes=adapt(q.rings.latitudes),node_offsets=adapt(q.node_offsets),node_rows=adapt(q.node_rows),
        cell_offsets=adapt(q.cell_offsets),nlon=adapt(q.nlon),
        weights=adapt(q.plain_weights),cosweights=adapt(q.cosine_weights),
        areas=adapt(q.areas),rows=adapt(q.boundary_rows),counts=adapt(q.boundary_counts),
        bounds=adapt(q.boundary_offsets))
    return SpectralIntegralWorkspace(q,s,surface,g)
end

"""Sample exp(LNSP), then integrate its piecewise-linear interpolant over cells."""
function integrate_surface_pressure!(ws::SpectralIntegralWorkspace,H,T)
    gradients = lnsp_gradient(H,T)
    spectra = zeros(ComplexF64,T+2,T+2,3)
    spectra[1:T+1,1:T+1,1] .= H
    spectra[:,:,2] .= gradients[1]
    spectra[:,:,3] .= gradients[2]
    surface = synthesize_rings(spectra,ws.surface)
    p = ws.synthesis.plan
    ps = similar(p.coeffs,Float64,size(surface,1),1)
    backend = get_backend(ps)
    surface_pressure_samples!(backend,256)(ps,surface;ndrange=length(ps))
    prefix = similar(ps)
    ring_prefix!(prefix,ps,ws.geometry,1,maximum(ws.quadrature.rings.nlon))
    synchronize(backend)
    return (;surface,ps,prefix)
end

"""Integrate native-level face rates and independent spectral divergence C_k."""
function integrate_face_fluxes!(ws::SpectralIntegralWorkspace,H,vo,d,T,dA,dB,c::TM5SpectralFluxes)
    q = ws.quadrature
    g = ws.geometry
    p = ws.synthesis.plan
    surf = integrate_surface_pressure!(ws,H,T)
    nc = length(q.areas)
    nv = q.boundary_offsets[end]-1
    nz = length(dA)
    ps_out = similar(p.coeffs,Float64,nc)
    u_out = zeros(nc,nz)
    v_out = zeros(nv,nz)
    C_out = zeros(nc,nz)
    backend = get_backend(ps_out)
    nf = p.batch
    ns = q.rings.offsets[end]-1
    for firstlevel in 1:nf:nz
        levels = firstlevel:min(firstlevel+nf-1,nz)
        nb = length(levels)
        vb = zeros(ComplexF64,T+1,T+1,nf)
        db = similar(vb)
        fill!(db,0)
        vb[:,:,1:nb] .= view(vo,:,:,levels)
        db[:,:,1:nb] .= view(d,:,:,levels)
        U,V,D = synthesize_native_fields(vb,db,T,c.vod2uv_radius,ws.synthesis)
        fu = similar(p.coeffs,Float64,ns,nf)
        fv = similar(fu)
        forcing = similar(fu)
        aa=zeros(nf)
        bb=zeros(nf)
        aa[1:nb].=dA[levels]
        bb[1:nb].=dB[levels]
        spectral_flux_integrands!(backend,256)(fu,fv,forcing,U,V,D,surf.ps,surf.surface,
            g.offsets,g.lengths,g.latitudes,p.adapter(aa),p.adapter(bb),q.radius,c.vod2uv_radius,nf;
            ndrange=(maximum(q.rings.nlon),length(q.rings.nlon),nf))
        fp = similar(fv)
        cp = similar(forcing)
        for (prefix,values) in ((fp,fv),(cp,forcing))
            ring_prefix!(prefix,values,g,nf,maximum(q.rings.nlon))
        end
        uu=similar(p.coeffs,Float64,nc,nf)
        vv=similar(p.coeffs,Float64,nv,nf)
        cc=similar(p.coeffs,Float64,nc,nf)
        integrate_cell_rows!(backend,256)(ps_out,uu,cc,surf.ps,surf.prefix,fu,forcing,cp,
            g.lengths,g.offsets,g.node_offsets,g.node_rows,g.cell_offsets,g.nlon,g.weights,g.cosweights,
            g.areas,q.radius,nf;
            ndrange=(maximum(q.nlon),length(q.nlon),nf))
        integrate_boundary_rows!(backend,256)(vv,fv,fp,g.lengths,g.offsets,g.rows,g.counts,g.bounds,nf;
            ndrange=(maximum(q.boundary_counts),length(q.boundary_counts),nf))
        synchronize(backend)
        u_out[:,levels] .= Array(uu)[:,1:nb]
        v_out[:,levels] .= Array(vv)[:,1:nb]
        C_out[:,levels] .= Array(cc)[:,1:nb]
    end
    return (;ps=Array(ps_out),u=u_out,v=v_out,C=C_out)
end

"""Upward independent interface flux W_e = Σ_{k<e} C_k − B_e Σ_k C_k."""
function integrate_vertical_flux!(W,C,B)
    nc,nz=size(C)
    first(B)==0 && last(B)==1 || throw(ArgumentError("TM5 W requires B=0 at the top and B=1 at the surface"))
    size(W)==(nc,nz+1) && length(B)==nz+1 || throw(DimensionMismatch("vertical integral interfaces"))
    for cell in 1:nc
        total=sum(view(C,cell,:))
        acc=0.0
        W[cell,1]=0.0
        for k in 1:nz-1
            acc += C[cell,k]
            W[cell,k+1]=acc-B[k+1]*total
        end
        W[cell,end]=0.0
    end
    return W
end
