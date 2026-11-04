using Test, AtmosTransport, LinearAlgebra, SparseArrays, KernelAbstractions, Random
const PP = AtmosTransport.Preprocessing
const SP = PP.Spectral
const MC = PP.MassClosure
const G = AtmosTransport.Grids
const R = AtmosTransport.Parameters.EARTH_RADIUS
const RV = AtmosTransport.Parameters.IFS_EARTH_RADIUS
const GRAV = AtmosTransport.Parameters.STANDARD_GRAVITY

function cs_integral_workspace(grid, T, c; kwargs...)
    q = SP.CSSpectralQuadrature(grid, T, c)
    faces = MC.cs_spectral_face_geometry(q.mesh, grid.mesh, grid.face_table; samples_per_face=q.samples_per_face)
    overlap = AtmosTransport.Regridding.build_regridder(q.mesh, grid.mesh)
    return SP.CSSpectralIntegralWorkspace(q, T, faces, overlap; kwargs...)
end

@testset "Portable CS face gather and analytic integrals" begin
    grid = PP.build_target_geometry(Dict("type"=>"cubed_sphere", "Nc"=>4), Float64)
    ft = grid.face_table
    centre = MC.cs_cell_bases(grid.mesh)[1]
    errors = Dict(field=>Float64[] for field in (:rotation, :harmonic))
    for ny in (45, 90, 180)
        source = G.LatLonMesh(; Nx=2ny, Ny=ny, FT=Float64, radius=R)
        op = MC.LineIntegralFaceFluxes(source, grid.mesh, ft; samples_per_face=64)
        xyz = [G.unit_xyz_from_radians(deg2rad(l), deg2rad(p)) for p in source.φᶜ for l in source.λᶜ]
        for field in (:rotation, :harmonic)
            values = field === :rotation ? [G.cross3((0.2,-0.3,1.),x) for x in xyz] :
                [(0.,0.,1.).-x[3].*x for x in xyz] # grad_s Y10, unit normalization
            exact = map(1:ft.nf) do f
                a, b = MC.cs_face_corners(grid.mesh, ft, f)
                n = G.unit3(G.cross3(a,b))
                sign = G.dot3(n, centre[ft.face_right[f]].-centre[ft.face_left[f]]) >= 0 ? 1 : -1
                R*sign*(field === :rotation ? G.dot3((0.2,-0.3,1.), a.-b) : G.arc3(a,b)*n[3])
            end
            for FT in (Float64, Float32)
                input = reshape(FT.(vcat([[v[d] for v in values] for d in 1:3]...)), :, 1)
                output = zeros(FT, ft.nf, 1)
                SP.cs_apply_integral!(output, input, SP.cs_gather_operator(op.L, identity, FT))
                synchronize(get_backend(output))
                reference = op.L*input[:, 1]
                @test output[:, 1] ≈ reference rtol=(FT === Float64 ? 1e-12 : 1e-5) atol=(FT === Float64 ? 1e-8 : 1.)
                FT === Float64 && push!(errors[field], maximum(abs, output[:,1]-exact)/maximum(abs,exact))
            end
        end
    end
    @info "CS refined-grid analytic errors" errors
    for e in Base.values(errors)
        @test e[3] < e[2]/3 < e[1]/9
    end
    # Midpoint rule alone, on exact Cartesian samples: second order along
    # great circles independently of the bilinear interpolation floor.
    a, b = MC.cs_face_corners(grid.mesh, ft, 5)
    theta = G.arc3(a,b)
    n = G.unit3(G.cross3(a,b))
    omega = (0.2,-0.3,1.)
    exact = G.dot3(omega, a.-b)
    errs = [abs(sum(G.dot3(G.cross3(omega,G.slerp3(a,b,theta,(i-0.5)/ns)),n) for i in 1:ns)*theta/ns-exact) for ns in (4,8,16)]
    @test errs[3] < errs[2]/3.9 < errs[1]/3.9^2
end


@testset "Compact CS geometry preserves expanded Cartesian bits" begin
    rng = MersenneTwister(1739)
    for convention in ("gnomonic", "geos_native")
        grid = PP.build_target_geometry(Dict("type"=>"cubed_sphere", "Nc"=>2,
            "panel_convention"=>convention), Float64)
        q = SP.CSSpectralQuadrature(grid, 3, PP.TM5SpectralFluxes())
        geometry = MC.cs_spectral_face_geometry(q.mesh, grid.mesh, grid.face_table;
            samples_per_face=q.samples_per_face)
        rows, normals = geometry.stencils, geometry.normals
        ns, nf = size(rows)
        # Independent expansion of the previous map, including signed zeros.
        offsets = cumsum([1; 3 .* diff(rows.colptr)])
        indices = Int[]; weights = Float64[]
        for f in 1:nf, d in 1:3, t in rows.colptr[f]:rows.colptr[f+1]-1
            push!(indices, (d-1)*ns+rows.rowval[t])
            push!(weights, normals[d,f]*rows.nzval[t])
        end
        expanded = sparse(transpose(SparseMatrixCSC(3ns,nf,offsets,indices,weights)))
        values = randn(rng,3ns,4) .* exp2.(rand(rng,-20:20,3ns,4))
        @test any(iszero, weights)
        for FT in (Float64, Float32)
            compact = SP.cs_gather_operator(geometry,identity,FT)
            previous = SP.cs_gather_operator(expanded,identity,FT)
            input = FT.(values)
            actual = zeros(FT,nf,4); expected = similar(actual)
            SP.cs_apply_integral!(actual,input,compact)
            SP.cs_apply_integral!(expected,input,previous)
            synchronize(get_backend(actual))
            @test reinterpret(UInt8,vec(actual)) == reinterpret(UInt8,vec(expected))
            @test length(compact.weights)*3 == length(previous.weights)
            @test Base.summarysize(compact) < Base.summarysize(previous)*(FT===Float64 ? .36 : .52)
            FT===Float32 && @test any(!iszero,compact.rounding)
        end
    end
end

@testset "CS spectral Gauss, meridian LL parity, constant PS and W" begin
    grid = PP.build_target_geometry(Dict("type"=>"cubed_sphere", "Nc"=>2), Float64)
    T = 3
    h = zeros(ComplexF64,T+1,T+1); h[1] = log(100000.)
    vo = zeros(ComplexF64,T+1,T+1,1); d = similar(vo); fill!(d,0)
    vo[2,1,1] = 1e-6
    errors = Float64[]
    variable_errors = Float64[]
    parity_errors = Float64[]
    for refinement in (8,16,32,64)
        c = PP.TM5SpectralFluxes(refinement)
        ws = cs_integral_workspace(grid,T,c; batch=1)
        raw = SP.integrate_face_fluxes!(ws,h,vo,d,T,[0.],[1.],c)
        @test raw.ps ≈ fill(100000.,length(raw.ps)) rtol=2e-12
        @test all(iszero,raw.C)
        W = zeros(length(raw.ps),2)
        SP.integrate_vertical_flux!(W,raw.C,[0.,1.])
        @test all(iszero,W)
        # On the central x edge of panel 1 the face is a meridian. Compare
        # the LL closed-form integral, including the independent IFS radius.
        ft = grid.face_table
        f = findfirst(1:ft.nf) do f
            a,b = MC.cs_face_corners(grid.mesh,ft,f)
            ft.face_dir[f] == 1 && ft.face_idx_i[f] == 2 && ft.face_panel[f] == 1 &&
                abs(a[1]*b[2]-a[2]*b[1]) < 1e-12
        end
        @test f !== nothing
        a,b = MC.cs_face_corners(grid.mesh,ft,f)
        exact = R*RV*sqrt(3)*1e-6*100000/(2GRAV)*abs(b[3]-a[3])
        push!(errors, abs(abs(raw.u[f,1])-exact)/exact)
        ll = PP.build_target_geometry(Dict("nlon"=>8,"nlat"=>4),Float64)
        llws = SP.SpectralIntegralWorkspace(ll,T,c;batch=1)
        llraw = SP.integrate_face_fluxes!(llws,h,vo,d,T,[0.],[1.],c)
        south = min(asind(a[3]),asind(b[3]))
        north = max(asind(a[3]),asind(b[3]))
        j = findfirst(j->isapprox(ll.mesh.φᶠ[j],south;atol=1e-10) &&
            isapprox(ll.mesh.φᶠ[j+1],north;atol=1e-10),1:ll.mesh.Ny)
        @test j !== nothing
        longitude = atand(a[2],a[1])
        i = findfirst(i -> abs(mod(ll.mesh.λᶠ[i]-longitude+180,360)-180)<1e-10,1:ll.mesh.Nx)
        @test i !== nothing
        llrate = llraw.u[i+(j-1)*ll.mesh.Nx,1]
        push!(parity_errors,abs(abs(raw.u[f,1])-llrate)/abs(llrate))
        # A divergent degree-one mode exercises independent spectral C.
        fill!(vo,0); d[2,1,1] = 1e-7
        divergent = SP.integrate_face_fluxes!(ws,h,vo,d,T,[0.],[1.],c)
        div = zeros(length(raw.ps))
        for f in 1:ft.nf
            div[ft.face_left[f]] += divergent.u[f,1]
            div[ft.face_right[f]] -= divergent.u[f,1]
        end
        residual = norm(div-divergent.C[:,1])/norm(divergent.C)
        @info "CS spectral Gauss and meridian parity" refinement shape=(ws.quadrature.mesh.Nx,ws.quadrature.mesh.Ny) residual meridian_error=errors[end]
        @test residual < 0.03
        fill!(d,0); vo[2,1,1] = 1e-6
        h[3,2] = 0.03
        varying = SP.integrate_face_fluxes!(ws,h,vo,d,T,[0.],[1.],c)
        fill!(div,0)
        for f in 1:ft.nf
            div[ft.face_left[f]] += varying.u[f,1]
            div[ft.face_right[f]] -= varying.u[f,1]
        end
        push!(variable_errors,norm(div-varying.C[:,1])/norm(varying.C))
        h[3,2] = 0
    end
    @test errors[end] < errors[end-1] < errors[1]
    # The coarse grids can cancel interpolation and midpoint errors. The
    # last three resolutions are in the second-order asymptotic range.
    @test variable_errors[end] < variable_errors[end-1]/3.5 < variable_errors[end-2]/3.5^2
    @test parity_errors[end] < parity_errors[end-1]/3.5 < parity_errors[end-2]/3.5^2
    @info "CS vs production LL meridian and pressure-gradient Gauss" parity_errors variable_errors
end


# Exercise the device implementation and its selected precision on CPU arrays,
# independently of CUDA availability (the same convention as backend tests).
struct CSHostDeviceSynthesis{FT} <: PP.Common.AbstractSynthesisBackend end
function SP.build_ring_synthesis(::CSHostDeviceSynthesis{FT},rings,T,batch,precision) where FT
    plan=SP.RingSynthesisPlan(rings,T,batch,FT)
    return SP.GPUSpectralSynthesis{FT,typeof(plan)}(plan)
end
@testset "CS full portable integration in selected precision" begin
    grid=PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>2),Float64)
    c=PP.TM5SpectralFluxes(4)
    h=zeros(ComplexF64,4,4);h[1]=log(100000.);h[3,2]=0.02
    vo=zeros(ComplexF64,4,4,2);d=copy(vo)
    vo[2,1,:].=1e-6;d[3,2,:].=1e-8
    cpu=cs_integral_workspace(grid,3,c;batch=2)
    reference=SP.integrate_face_fluxes!(cpu,h,vo,d,3,[2000.,-2000.],[0.3,0.7],c)
    for FT in (Float64,Float32)
        portable=cs_integral_workspace(grid,3,c;batch=2,synthesis=CSHostDeviceSynthesis{FT}())
        actual=SP.integrate_face_fluxes!(portable,h,vo,d,3,[2000.,-2000.],[0.3,0.7],c)
        @test eltype(portable.geometry.face.weights)===FT
        @test actual.ps ≈ reference.ps rtol=(FT === Float64 ? 1e-12 : 3e-6)
        tolerance=FT === Float64 ? 1e-12 : 1e-5
        @test norm(actual.u-reference.u)/norm(reference.u)<=tolerance
        @test norm(actual.C-reference.C)/norm(reference.C)<=tolerance
    end
end


if get(ENV,"ATMOSTR_TEST_SPECTRAL_GPU","0") == "1"
    @eval using CUDA
    @testset "CS device face and spectral operators vs CPU" begin
        CUDA.allowscalar(false)
        grid=PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>2),Float64)
        c=PP.TM5SpectralFluxes(4)
        h=zeros(ComplexF64,4,4);h[1]=log(100000.);h[3,2]=0.02
        vo=zeros(ComplexF64,4,4,2);d=copy(vo)
        vo[2,1,:].=1e-6;d[3,2,:].=1e-8
        cpu=cs_integral_workspace(grid,3,c;batch=2)
        reference=SP.integrate_face_fluxes!(cpu,h,vo,d,3,[2000.,-2000.],[0.3,0.7],c)
        for FT in (Float64,Float32)
            gpu=cs_integral_workspace(grid,3,c;batch=2,synthesis=PP.Common.GPUSynthesis(),precision=FT)
            actual=SP.integrate_face_fluxes!(gpu,h,vo,d,3,[2000.,-2000.],[0.3,0.7],c)
            @test actual.ps ≈ reference.ps rtol=(FT === Float64 ? 1e-12 : 3e-6)
            tolerance=FT === Float64 ? 1e-12 : 1e-5
            @test norm(actual.u-reference.u)/norm(reference.u)<=tolerance
            @test norm(actual.C-reference.C)/norm(reference.C)<=tolerance
        end
    end
end
