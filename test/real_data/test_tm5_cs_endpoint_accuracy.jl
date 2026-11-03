using Test, AtmosTransport, LinearAlgebra, Dates, TOML
const PP=AtmosTransport.Preprocessing
const SP=PP.Spectral
const MC=PP.MassClosure
const G=AtmosTransport.Grids
const RV=AtmosTransport.Parameters.IFS_EARTH_RADIUS

function cs_endpoint_normal_samples(U,V,H,lon,lat,n,dA,dB)
    result=zeros(size(U))
    for k in axes(U,2),i in axes(U,1)
        sl,cl=sincos(lon[i]);sp,cp=sincos(lat[i])
        east=-n[1]*sl+n[2]*cl
        north=-sp*(n[1]*cl+n[2]*sl)+n[3]*cp
        result[i,k]=(east*U[i,k]+north*V[i,k])/cp*(dA[k]+dB[k]*exp(H[i]))
    end
    return result
end

@testset "ERA5 2021-12-01 C90 direct face reference" begin
    dir=get(ENV,"ATMOSTR_ERA5_SPECTRAL_DIR",expanduser("~/data/AtmosTransport/met/era5/spectral_hourly"))
    paths=SP.era5_spectral_paths(dir,Date(2021,12,1))
    if !isfile(paths.vo_d) || !isfile(paths.lnsp)
        @test_skip false
    else
        grid=PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>90),Float64)
        ft=grid.face_table
        faces=[findfirst(f->ft.face_panel[f]==p && ft.face_idx_i[f]==i && ft.face_idx_j[f]==j,1:ft.nf)
            for (p,i,j) in ((1,46,45),(1,68,65),(3,46,45),(1,1,45))]
        ab=TOML.parsefile(joinpath(pkgdir(AtmosTransport),"config/era5_L137_coefficients.toml"))["coefficients"]
        SP.with_spectral_hours(paths) do read_hour
            for hour in (0,1)
                endpoint=read_hour(hour);T=endpoint.T
                levels=[1,70,137]
                vo=endpoint.vo[:,:,levels];d=endpoint.d[:,:,levels];H=endpoint.lnsp
                q=SP.CSSpectralQuadrature(grid,T,PP.TM5SpectralFluxes())
                synthesis=SP.CPURingSynthesis(q.rings,T+1,3)
                U,V,_=SP.synthesize_native_fields(vo,d,T,RV,synthesis)
                SP.cs_stable_polar_winds!(U,V,vo,d,RV,SP.cs_polar_synthesis(synthesis,T))
                surface=SP.surface_synthesis(synthesis,1)
                hp=zeros(ComplexF64,T+2,T+2,1);hp[1:T+1,1:T+1,1].=H
                hs=SP.synthesize_rings(hp,surface)[:,1]
                dA=diff(ab["a"])[levels];dB=diff(ab["b"])[levels]
                for f in faces
                    a,b=MC.cs_face_corners(grid.mesh,ft,f)
                    n=G.unit3(G.cross3(a,b))
                    ids,w=MC.cs_cardinal_integral_weights(q.mesh,a,b,q.samples_per_face)
                    lon=[deg2rad(q.mesh.λᶜ[mod(i-1,q.mesh.Nx)+1]) for i in ids]
                    lat=[deg2rad(q.mesh.φᶜ[(i-1)÷q.mesh.Nx+1]) for i in ids]
                    values=cs_endpoint_normal_samples(U[ids,:],V[ids,:],hs[ids],lon,lat,n,dA,dB)
                    # Production interpolates Cartesian products. Projecting
                    # at source nodes is identical because n is constant.
                    approximate=vec(transpose(w./q.radius)*values)
                    rule=eigen(SymTridiagonal(zeros(256),[i/sqrt(4i*i-1) for i in 1:255]))
                    theta=G.arc3(a,b)
                    points=[G.slerp3(a,b,theta,(t+1)/2) for t in rule.values]
                    lon=[atan(x[2],x[1]) for x in points]
                    lat=[asin(clamp(x[3],-1.,1.)) for x in points]
                    u,v,h=SP.cs_exact_samples(H,vo,d,T,RV,lon,lat)
                    exactvalues=cs_endpoint_normal_samples(u,v,h[:,1],lon,lat,n,dA,dB)
                    weights=theta.*rule.vectors[1,:].^2
                    exact=vec(transpose(weights)*exactvalues)
                    envelope=vec(transpose(weights)*abs.(exactvalues))
                    relative=maximum(abs.(approximate-exact)./max.(abs.(exact),0.1.*envelope))
                    @info "ERA5 CS face" hour f T levels relative approximate exact
                    @test relative < 1e-3
                end
                GC.gc()
            end
        end
    end
end
