using AtmosTransport, LinearAlgebra
source=read("test/core/test_cubed_sphere_advection.jl",String)
include_string(Main,first(split(source,"# Panel connectivity")))
const Adv=AtmosTransport.Operators.Advection
const G=AtmosTransport.Grids
function grouped_horizontal(rm,m,am,bm,mesh,scheme,dir,scale)
    Nc,Hp=mesh.Nc,mesh.Hp; Nz=size(m[1],3); N=Int32(Nc+2Hp)
    rnew,mnew=map(copy,rm),map(copy,m)
    for p in 1:6,k in 1:Nz,jj in 1:Nc,ii in 1:Nc
        i,j=Hp+ii,Hp+jj
        if dir==1
            a=ii==1 ? 0. : scale*am[p][i,j,k]
            b=ii==Nc ? 0. : scale*am[p][i+1,j,k]
            left=Adv._xface_tracer_flux(Int32(i),j,k,rm[p],m[p],a,scheme,N)
            right=Adv._xface_tracer_flux(Int32(i+1),j,k,rm[p],m[p],b,scheme,N)
        else
            a=jj==1 ? 0. : scale*bm[p][i,j,k]
            b=jj==Nc ? 0. : scale*bm[p][i,j+1,k]
            left=Adv._yface_tracer_flux(i,Int32(j),k,rm[p],m[p],a,scheme,N)
            right=Adv._yface_tracer_flux(i,Int32(j+1),k,rm[p],m[p],b,scheme,N)
        end
        rnew[p][i,j,k]=rm[p][i,j,k]+left-right
        mnew[p][i,j,k]=m[p][i,j,k]+a-b
    end
    location=AtmosTransport.Grids.cs_edge_face_location
    for p in 1:6,edge in 1:4
        neighbor=mesh.connectivity.neighbors[p][edge];q=neighbor.panel
        p<q || continue
        other=G.reciprocal_edge(mesh.connectivity,p,edge)
        axis,_,_=location(edge,1,Nc)
        axis==dir || continue
        sign=edge in (G.EDGE_EAST,G.EDGE_NORTH) ? 1. : -1.
        for k in 1:Nz,s in 1:Nc
            t=neighbor.orientation==0 ? s : Nc+1-s
            _,i,j=location(edge,s,Nc);_,u,v=location(other,t,Nc)
            ii,jj=Hp+clamp(i,1,Nc),Hp+clamp(j,1,Nc)
            uu,vv=Hp+clamp(u,1,Nc),Hp+clamp(v,1,Nc)
            flux=scale*(axis==1 ? am[p][Hp+i,Hp+j,k] : bm[p][Hp+i,Hp+j,k])
            tracer=axis==1 ? Adv._xface_tracer_flux(Int32(Hp+i),Hp+j,k,rm[p],m[p],flux,scheme,N) : Adv._yface_tracer_flux(Hp+i,Int32(Hp+j),k,rm[p],m[p],flux,scheme,N)
            rnew[p][ii,jj,k]-=sign*tracer; rnew[q][uu,vv,k]+=sign*tracer
            mnew[p][ii,jj,k]-=sign*flux; mnew[q][uu,vv,k]+=sign*flux
        end
    end
    fill_panel_halos!(rnew,mesh;dir); fill_panel_halos!(mnew,mesh;dir)
    return rnew,mnew
end
function run_grouped(n; uniform=false)
    mesh,m,rm=make_structured_cs_state(;Nc=8,Hp=3,Nz=2)
    uniform && (rm=map(a->a.*4e-4,m))
    am,bm,_=make_mirrored_cs_horizontal_fluxes(mesh,2)
    initial=total_interior(rm,8,3,2)
    for _ in 1:n,dir in (1,2,2,1)
        rm,m=grouped_horizontal(rm,m,am,bm,mesh,PPMScheme(),dir,100. / n)
    end
    r=vcat([vec(rm[p][4:11,4:11,:]) for p in 1:6]...)
    a=vcat([vec(m[p][4:11,4:11,:]) for p in 1:6]...)
    return r,a,initial
end
println("PROBE: F64 Nc8 Nz2, m about 1e9 kg, seam Courant about 0.1 per half step")
rref,_,_=run_grouped(512)
for n in (1,2,4,8,16,32,64,128)
    r,a,total=run_grouped(n)
    println("GROUPED ",n," drift=",(sum(r)-total)/total," reference_error=",norm(r-rref)/norm(rref)," min_air=",minimum(a))
end
r,a,total=run_grouped(2;uniform=true)
println("UNIFORM drift=",(sum(r)-total)/total," vmr_error=",maximum(abs,r./a.-4e-4))
