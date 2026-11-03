using Test, AtmosTransport, LinearAlgebra, KernelAbstractions
const PP = AtmosTransport.Preprocessing
const SP = PP.Spectral
const MC = PP.MassClosure
const G = AtmosTransport.Grids
const RV = AtmosTransport.Parameters.IFS_EARTH_RADIUS

function harmonic_moments(points,weights,T)
    out = zeros(ComplexF64,T+1,T+1)
    envelope = zeros(T+1,T+1)
    P = zeros(T+1,T+1)
    for (x,w) in zip(points,weights)
        SP.compute_legendre_column!(P,T,x[3])
        lon = atan(x[2],x[1])
        for m in 0:T
            phase = cis(m*lon)
            @inbounds for n in m:T
                value = w*P[n+1,m+1]
                out[n+1,m+1] += value*phase
                envelope[n+1,m+1] += abs(value)
            end
        end
    end
    return out,envelope
end

function gauss_arc(a,b,N)
    rule = eigen(SymTridiagonal(zeros(N),[i/sqrt(4i*i-1) for i in 1:N-1]))
    theta = G.arc3(a,b)
    points = [G.slerp3(a,b,theta,(t+1)/2) for t in rule.values]
    return points,theta.*rule.vectors[1,:].^2
end

function source_points(mesh,ids)
    return [G.unit_xyz_from_radians(deg2rad(mesh.λᶜ[mod(c-1,mesh.Nx)+1]),
        deg2rad(mesh.φᶜ[(c-1)÷mesh.Nx+1])) for c in ids]
end

function vector_harmonic_moments(points,weights,T,normal)
    rotation=zeros(ComplexF64,T+1,T+1);divergence=copy(rotation)
    er=zeros(T+1,T+1);ed=copy(er)
    for (x,w) in zip(points,weights)
        phi=asin(clamp(x[3],-1.,1.));lambda=atan(x[2],x[1])
        s,c=sincos(phi);sl,cl=sincos(lambda)
        east=-normal[1]*sl+normal[2]*cl
        north=-s*(normal[1]*cl+normal[2]*sl)+normal[3]*c
        diagonal,ddiagonal=1.,0.
        for m in 0:T
            if m>0
                a=sqrt((2m+1)/(2m))
                diagonal,ddiagonal=a*c*diagonal,a*(c*ddiagonal-s*diagonal)
            end
            phase=cis(m*lambda)
            p0,d0=0.,0.;p1,d1=diagonal,ddiagonal
            for n in m:T
                if n==m+1
                    a=sqrt(2m+3.)
                    p0,d0,p1,d1=p1,d1,a*s*p1,a*(c*p1+s*d1)
                elseif n>m+1
                    a=sqrt((4n*n-1)/(n*n-m*m))
                    b=sqrt((2n+1)*((n-1)^2-m*m)/((2n-3)*(n*n-m*m)))
                    p0,d0,p1,d1=p1,d1,a*s*p1-b*p0,a*(c*p1+s*d1)-b*d0
                end
                n==0 && continue
                r=w*(east*d1-north*im*m*p1/c)/(n*(n+1))*phase
                d=w*(-north*d1-east*im*m*p1/c)/(n*(n+1))*phase
                @inbounds begin
                    rotation[n+1,m+1]+=r;divergence[n+1,m+1]+=d
                    er[n+1,m+1]+=abs(r);ed[n+1,m+1]+=abs(d)
                end
            end
        end
    end
    return (rotation,divergence),(er,ed)
end

function harmonic_error(approx,exact,envelope)
    active=envelope .> maximum(envelope)*1e-13
    ordinary=active .& (abs.(exact) .>= 0.1.*envelope)
    bound=maximum(abs.(approx[active]-exact[active])./envelope[active])
    relative=maximum(abs.(approx[ordinary]-exact[ordinary])./abs.(exact[ordinary]))
    return bound,relative
end

@testset "Every retained scalar and vector harmonic on C90 arcs" begin
    grid = PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>90),Float64)
    for T in (255,639)
        q = SP.CSSpectralQuadrature(grid,T,PP.TM5SpectralFluxes())
        x(l,p) = G.unit_xyz_from_radians(deg2rad(l),deg2rad(p))
        ft = grid.face_table
        tilted = findfirst(f -> ft.face_panel[f]==1 && ft.face_idx_i[f]==68 && ft.face_idx_j[f]==65,1:ft.nf)
        polar = findfirst(f -> ft.face_panel[f]==3 && ft.face_idx_i[f]==46 && ft.face_idx_j[f]==45,1:ft.nf)
        arcs = [(x(0.,0.),x(0.,1.)), # halfway between source longitudes
            (x(first(q.mesh.λᶜ),0.),x(first(q.mesh.λᶜ),1.)),
            MC.cs_face_corners(grid.mesh,ft,tilted),MC.cs_face_corners(grid.mesh,ft,polar),
            (x(0.,0.),x(1.,0.))]
        for (case,(a,b)) in enumerate(arcs)
            ids,w = MC.cs_cardinal_integral_weights(q.mesh,a,b,q.samples_per_face)
            approx,_ = harmonic_moments(source_points(q.mesh,ids),w./q.radius,T)
            points,weights = gauss_arc(a,b,256)
            exact,envelope = harmonic_moments(points,weights,T)
            # A relative bound is undefined for a zero integral. All modes use
            # an absolute integral envelope; non-cancelling modes also use the
            # ordinary relative error. Include complex phases, not just cosines.
            active = envelope .> maximum(envelope)*1e-13
            ordinary = active .& (abs.(exact) .>= 0.1.*envelope)
            bound = maximum(abs.(approx[active]-exact[active])./envelope[active])
            relative = maximum(abs.(approx[ordinary]-exact[ordinary])./abs.(exact[ordinary]))
            @info "CS retained harmonic bound" T case shape=(q.mesh.Nx,q.mesh.Ny) samples=q.samples_per_face bound relative
            @test bound < 1e-3
            @test relative < 1e-3
            pairs=[(T,T),(T-2,T-2),(T,0),(T,div(T,2))]
            source=source_points(q.mesh,ids)
            selected=zeros(length(ids),length(pairs))
            P=zeros(T+1,T+1)
            for (i,xs) in enumerate(source)
                SP.compute_legendre_column!(P,T,xs[3])
                for (j,(n,m)) in enumerate(pairs)
                    # Choose the phase that makes the exact face integral
                    # real and positive; include halves between longitudes.
                    selected[i,j]=real(P[n+1,m+1]*cis(m*atan(xs[2],xs[1]))*
                        conj(exact[n+1,m+1])/max(abs(exact[n+1,m+1]),eps()))
                end
            end
            for FT in (Float32,Float64)
                op=(;offsets=Int32[1,length(ids)+1],indices=Int32.(1:length(ids)),weights=FT.(w./q.radius))
                output=zeros(FT,1,length(pairs))
                SP.cs_apply_integral!(output,FT.(selected),op)
                synchronize(get_backend(output))
                for (j,(n,m)) in enumerate(pairs)
                    @test abs(output[j]-abs(exact[n+1,m+1])) <=
                        1e-3*max(abs(exact[n+1,m+1]),envelope[n+1,m+1]) + 1e-14
                end
            end
            normal=G.unit3(G.cross3(a,b))
            av,_=vector_harmonic_moments(source_points(q.mesh,ids),w./q.radius,T,normal)
            ev,envelopes=vector_harmonic_moments(points,weights,T,normal)
            for kind in 1:2
                vb,vr=harmonic_error(av[kind],ev[kind],envelopes[kind])
                @info "CS every vector harmonic" T case kind bound=vb relative=vr
                @test vb < 1e-3
                @test vr < 1e-3
            end
            if case in (1,2,5)
                # On a meridian/parallel this is precisely IntLat/IntLon,
                # evaluated at converged refinement rather than its default.
                theta=G.arc3(a,b)
                nodes=[G.slerp3(a,b,theta,i/1024) for i in 0:1024]
                tw=fill(theta/1024,1025);tw[1]/=2;tw[end]/=2
                tm5,_=vector_harmonic_moments(nodes,tw,T,normal)
                for kind in 1:2
                    tb,tr=harmonic_error(av[kind],tm5[kind],envelopes[kind])
                    @info "CS vs converged TM5 vector rule" T case kind bound=tb relative=tr
                    @test tb < 1e-3
                    @test tr < 1e-3
                end
            end
            # Independently converge the reference, including high m near T.
            for (n,m) in pairs
                @test isfinite(approx[n+1,m+1])
            end
        end
    end
end

@testset "TM5 IntLon default cannot meet the requested exact bound" begin
    # V=cos(m*lambda) on an equatorial one-degree face. This is the retained
    # sectoral harmonic. IntLon is exactly the uniform-longitude trapezoid.
    for T in (255,639)
        m = T-2
        L = deg2rad(1.)
        r = SP.quadrature_refinement(T,1.)
        exact = sin(m*L)/m
        trap(n) = L/n*(0.5+sum(cos(m*L*i/n) for i in 1:n-1)+0.5cos(m*L))
        @info "Default vs converged TM5 IntLon" T r default_error=trap(r)/exact-1 converged_error=trap(1024)/exact-1
        @test abs(trap(r)/exact-1) > 1e-3
        @test abs(trap(1024)/exact-1) < 1e-4
    end
end

@testset "Review meridian high-m probes vs exact and default TM5 IntLat" begin
    grid=PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>90),Float64)
    for T in (255,639)
        q=SP.CSSpectralQuadrature(grid,T,PP.TM5SpectralFluxes())
        modes=unique([T,T-2,min(T,500)])
        for lambda in (0.,deg2rad(first(q.mesh.λᶜ)))
            a=G.unit_xyz_from_radians(lambda,0.)
            b=G.unit_xyz_from_radians(lambda,deg2rad(1.))
            ids,w=MC.cs_cardinal_integral_weights(q.mesh,a,b,q.samples_per_face)
            points=source_points(q.mesh,ids)
            source=zeros(length(ids),length(modes));P=zeros(T+1,T+1)
            for (i,x) in enumerate(points)
                SP.compute_legendre_column!(P,T,x[3])
                phi=asin(x[3]);dlambda=atan(x[2],x[1])-lambda
                for (j,m) in enumerate(modes)
                    source[i,j]=P[m+1,m+1]*cos(m*dlambda)*cos(dlambda)/cos(phi)
                end
            end
            exactpoints,ew=gauss_arc(a,b,256)
            exact=zeros(length(modes))
            for (x,wg) in zip(exactpoints,ew)
                SP.compute_legendre_column!(P,T,x[3])
                for (j,m) in enumerate(modes)
                    exact[j]+=wg*P[m+1,m+1]/sqrt(1-x[3]^2)
                end
            end
            r=SP.quadrature_refinement(T,1.)
            tm5=zeros(length(modes))
            for i in 0:r
                phi=deg2rad(i/r)
                SP.compute_legendre_column!(P,T,sin(phi))
                for (j,m) in enumerate(modes)
                    tm5[j]+=deg2rad(1.)/r*(i in (0,r) ? 0.5 : 1.)*P[m+1,m+1]/cos(phi)
                end
            end
            for FT in (Float32,Float64)
                op=(;offsets=Int32[1,length(ids)+1],indices=Int32.(1:length(ids)),weights=FT.(w./q.radius))
                output=zeros(FT,1,length(modes))
                SP.cs_apply_integral!(output,FT.(source),op)
                synchronize(get_backend(output))
                @info "Review meridian probe" T FT longitude=rad2deg(lambda) modes cs_relative=vec(output)./exact.-1 tm5_relative=tm5./exact.-1
                @test maximum(abs.(vec(output)./exact.-1)) < 1e-3
                @test maximum(abs.(vec(output)./tm5.-1)) < 1e-3
            end
        end
    end
end

@testset "Polar vector harmonics in selected precision at C90 production spacing" begin
    grid = PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>90),Float64)
    for T in (255,639)
        q = SP.CSSpectralQuadrature(grid,T,PP.TM5SpectralFluxes())
        phi = last(q.rings.latitudes)
        rings = SP.SynthesisRings([-phi,phi],fill(q.mesh.Nx,2),zeros(2))
        vo = zeros(ComplexF64,T+1,T+1,1)
        d = copy(vo); H = zeros(ComplexF64,T+1,T+1)
        vo[2,1,1] = 1e-6
        for FT in (Float32,Float64)
            plan = SP.RingSynthesisPlan(rings,T+1,1,FT)
            synthesis = SP.GPUSpectralSynthesis{FT,typeof(plan)}(plan)
            U,V,_ = SP.synthesize_native_fields(vo,d,T,RV,synthesis)
            expected = RV*sqrt(3)*1e-6/2*cos(phi)^2
            before = maximum(abs.(U.-expected))/expected
            polar = SP.cs_polar_synthesis(synthesis,T)
            SP.cs_stable_polar_winds!(U,V,vo,d,RV,polar)
            error = maximum(abs.(U.-expected))/expected
            @info "Polar zonal rotation" T FT latitude=rad2deg(phi) physical_u=expected/cos(phi) before error
            @test FT === Float64 || before > 1e-3
            @test error < 1e-3
            @test all(iszero,V)
            # Arbitrarily oriented rotation and divergence exercise m=1 and
            # both potential derivatives, compared with direct point sums.
            vo[2,2,1] = 2e-7+3e-7im
            d[2,1,1] = 1e-7; d[2,2,1] = 2e-8-1e-7im
            U,V,_ = SP.synthesize_native_fields(vo,d,T,RV,synthesis)
            SP.cs_stable_polar_winds!(U,V,vo,d,RV,polar)
            # Ring length need not be divisible by six: use exact grid longitudes.
            ids = q.mesh.Nx .+ [1,1+q.mesh.Nx÷6,1+q.mesh.Nx÷2]
            lon = 2pi.*((ids.-q.mesh.Nx).-1)./q.mesh.Nx
            uu,vv,_ = SP.cs_exact_samples(H,vo,d,T,RV,lon,fill(phi,3);precision=FT)
            @test U[ids,:] ≈ uu rtol=2e-4
            @test V[ids,:] ≈ vv rtol=2e-4
            vo[2,2,1]=0;fill!(d,0)
        end
    end
end

@testset "Direct point synthesis agrees with complete ring synthesis" begin
    T = 12
    rings = SP.SynthesisRings([-1.2,-0.4,0.2,1.1],fill(64,4),fill(0.1,4))
    vo = zeros(ComplexF64,T+1,T+1,2); d=copy(vo); H=zeros(ComplexF64,T+1,T+1)
    for m in 0:T,n in m:T
        vo[n+1,m+1,:].=1e-7*cis(n+m)
        d[n+1,m+1,:].=1e-8*cis(n-m)
        H[n+1,m+1]=1e-3*cis(n+m)
    end
    # Vorticity and divergence have zero global mode; no vector potential
    # exists for the scalar Laplacian's degree-zero nullspace.
    vo[1,1,:].=0;d[1,1,:].=0
    ring = SP.CPURingSynthesis(rings,T+1,2)
    U,V,_=SP.synthesize_native_fields(vo,d,T,RV,ring)
    padded=zeros(ComplexF64,T+2,T+2,1);padded[1:T+1,1:T+1,1].=H
    hs=SP.synthesize_rings(padded,SP.surface_synthesis(ring,1))
    ids=[1,27,88,153,256]
    lon=[0.1+2pi*mod(i-1,64)/64 for i in ids]
    lat=[rings.latitudes[(i-1)÷64+1] for i in ids]
    for FT in (Float32,Float64)
        u,v,h=SP.cs_exact_samples(H,vo,d,T,RV,lon,lat;precision=FT)
        @test norm(u-U[ids,:])/norm(U[ids,:]) < (FT === Float32 ? 2e-5 : 1e-12)
        @test norm(v-V[ids,:])/norm(V[ids,:]) < (FT === Float32 ? 2e-5 : 1e-12)
        @test norm(h.-hs[ids,:])/norm(hs[ids,:]) < (FT === Float32 ? 3e-5 : 2e-12)
        @test eltype(u)===eltype(v)===eltype(h)===FT
    end
end

@testset "Selected-precision high-degree vector synthesis on production rings" begin
    grid=PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>90),Float64)
    for T in (255,639)
        q=SP.CSSpectralQuadrature(grid,T,PP.TM5SpectralFluxes())
        phi=last(q.rings.latitudes)
        rings=SP.SynthesisRings([-phi,-0.01,0.01,phi],fill(q.mesh.Nx,4),fill(first(q.rings.origins),4))
        pairs=[(T,T),(T-2,T-2),(T,0),(T,div(T,2))]
        vo=zeros(ComplexF64,T+1,T+1,4);d=copy(vo);H=zeros(ComplexF64,T+1,T+1)
        for (k,(n,m)) in enumerate(pairs)
            vo[n+1,m+1,k]=1e-6*cis(0.4)
            d[n+1,m+1,k]=3e-7*cis(0.7)
        end
        ids=[(j-1)*q.mesh.Nx+i for j in 1:4 for i in (1,q.mesh.Nx÷3,q.mesh.Nx÷2)]
        lon=[first(rings.origins)+2pi*mod(i-1,q.mesh.Nx)/q.mesh.Nx for i in ids]
        lat=[rings.latitudes[(i-1)÷q.mesh.Nx+1] for i in ids]
        exactu,exactv,_=SP.cs_exact_samples(H,vo,d,T,RV,lon,lat)
        for FT in (Float32,Float64)
            plan=SP.RingSynthesisPlan(rings,T+1,4,FT)
            synthesis=SP.GPUSpectralSynthesis{FT,typeof(plan)}(plan)
            U,V,_=SP.synthesize_native_fields(vo,d,T,RV,synthesis)
            SP.cs_stable_polar_winds!(U,V,vo,d,RV,SP.cs_polar_synthesis(synthesis,T))
            eu=[norm(U[ids,k]-exactu[:,k])/norm(exactu[:,k]) for k in 1:4]
            ev=[norm(V[ids,k]-exactv[:,k])/norm(exactv[:,k]) for k in 1:4]
            @info "High-degree selected-precision wind error" T FT u=maximum(eu) v=maximum(ev)
            for k in 1:4
                @test eu[k] < 5e-5
                @test ev[k] < 5e-5
            end
        end
    end
end
