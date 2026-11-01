using Test, AtmosTransport, LinearAlgebra, Random
const SS=AtmosTransport.Preprocessing.Spectral

function check_ring_synthesis(adapter=identity; T=63)
    rng=MersenneTwister(35);nf=3
    lengths=T>=255 ? [2T+3,2T+5,2T+7,2T+4,2T+3] : [129,35,131,128,129]
    rings=SS.SynthesisRings([-1.56,-0.4,0.,0.4,1.56],lengths,[0.,0.1,-pi,-0.2,1.2])
    coeff=zeros(ComplexF64,T+1,T+1,nf)
    for k in 1:nf,m in 0:T,n in m:T
        coeff[n+1,m+1,k]=randn(rng)+im*randn(rng)
    end
    reference=zeros(rings.offsets[end]-1,nf)
    P=zeros(T+1,T+1)
    for j in eachindex(rings.nlon)
        SS.compute_legendre_column!(P,T,sin(rings.latitudes[j]))
        for k in 1:nf
            buf=zeros(ComplexF64,rings.nlon[j])
            SS.fill_ring_spectrum!(buf,view(coeff,:,:,k),P,T,rings.origins[j])
            reference[rings.offsets[j]:rings.offsets[j+1]-1,k] .= real.(SS.bfft(buf))
        end
    end
    for FT in (Float64,Float32)
        plan=SS.RingSynthesisPlan(rings,T,nf,FT;adapter)
        synthesis=SS.GPUSpectralSynthesis{FT,typeof(plan)}(plan)
        actual=Array(SS.synthesize_rings(coeff,synthesis))
        @test norm(actual-reference)/norm(reference) < (FT===Float64 ? 1e-13 : 1e-6)
        surface=SS.surface_synthesis(synthesis,3)
        @test eltype(surface.plan.coeffs)===Float64
        FT===Float64 && @test surface.plan.legendre === plan.legendre
        @test Array(SS.synthesize_rings(coeff,surface)) ≈ reference rtol=1e-13
        windplan=SS.RingSynthesisPlan(rings,T+1,nf,FT;adapter)
        windbackend=SS.GPUSpectralSynthesis{FT,typeof(windplan)}(windplan)
        vo=coeff.*1e-6;d=coeff.*(1e-7im)
        devicewinds=SS.synthesize_native_fields(vo,d,T,AtmosTransport.Parameters.IFS_EARTH_RADIUS,windbackend)
        cpuwinds=SS.synthesize_native_fields(vo,d,T,AtmosTransport.Parameters.IFS_EARTH_RADIUS,
            SS.CPURingSynthesis(rings,T+1,nf))
        for (actual,ref) in zip(devicewinds,cpuwinds)
            @test norm(Array(actual)-ref)/norm(ref) < (FT===Float64 ? 1e-12 : 1e-6)
        end
    end
end

@testset "Portable spectral synthesis, CPU array backend" begin
    check_ring_synthesis()
    c=AtmosTransport.Preprocessing.Common
    @test c.resolve_spectral_construction(Dict()) == NamedTuple()
    config=Dict{String,Any}("preprocessing"=>Dict{String,Any}("spectral_flux_construction"=>"tm5_spectral","spectral_synthesis"=>"cpu"))
    settings=c.resolve_spectral_construction(config)
    @test settings.spectral_flux_construction isa c.TM5SpectralFluxes
    @test settings.spectral_synthesis isa c.CPUSynthesis
    @test_throws ArgumentError AtmosTransport.Preprocessing.TransportBinary.tm5_construction_metadata(settings.spectral_flux_construction,0)
    for (key,value) in (("spectral_synthesis",true),("spectral_synthesis_float","half"),("quadrature_refinement_steps",false))
        bad=deepcopy(config);bad["preprocessing"]=Dict{String,Any}(bad["preprocessing"]);bad["preprocessing"][key]=value
        @test_throws ArgumentError c.resolve_spectral_construction(bad)
    end
    if get(ENV,"ATMOSTR_TEST_SPECTRAL_GPU","0")!="1"
        @test_throws ArgumentError c.validate_gpu_spectral_backend()
    end
end

# Explicit opt-in only: ordinary CPU CI never imports CUDA or touches a device.
if get(ENV,"ATMOSTR_TEST_SPECTRAL_GPU","0")=="1"
    @eval using CUDA
    @testset "CUDA spectral synthesis and TF32 guard" begin
        if CUDA.functional()
            original_mode=CUDA.math_mode()
            check_ring_synthesis(CUDA.CuArray)
            @test CUDA.math_mode()==original_mode
            check_ring_synthesis(CUDA.CuArray; T=255)
            PP=AtmosTransport.Preprocessing;MC=PP.MassClosure
            grid=PP.build_target_geometry(Dict("nlon"=>12,"nlat"=>6),Float64)
            for T in (7,255)
                h=zeros(ComplexF64,T+1,T+1);h[1]=log(100000.);h[3,2]=0.01
                vo=zeros(ComplexF64,T+1,T+1,2);d=similar(vo);fill!(d,0)
                vo[2,1,:].=1e-6;d[4,2,:].=1e-7
                vo[T+1,T+1,:].=2e-8
                c=PP.TM5SpectralFluxes()
                cpu=SS.integrate_face_fluxes!(SS.SpectralIntegralWorkspace(grid,T,c;batch=2),
                    h,vo,d,T,[0.,0.],[0.3,0.7],c)
                for FT in (Float64,Float32)
                    gpu=SS.integrate_face_fluxes!(SS.SpectralIntegralWorkspace(grid,T,c;
                        batch=2,synthesis=PP.Common.GPUSynthesis(),precision=FT),h,vo,d,T,[0.,0.],[0.3,0.7],c)
                    @test gpu.ps ≈ cpu.ps rtol=1e-12
                    for name in (:u,:v,:C)
                        @test norm(gpu[name]-cpu[name])/max(norm(cpu[name]),eps()) <
                            (FT===Float64 ? 1e-12 : 1e-6)
                    end
                end
            end
            @test AtmosTransport.Preprocessing.Common.validate_gpu_spectral_backend()===nothing
            CUDA.math_mode!(CUDA.FAST_MATH)
            @test_throws ArgumentError SS.spectral_gpu_math_guard()
            CUDA.math_mode!(CUDA.DEFAULT_MATH)
        else
            @test_skip false
        end
    end
end

@testset "Spectral-only config options fail closed" begin
    common=AtmosTransport.Preprocessing.Common
    for key in common.SPECTRAL_CONSTRUCTION_KEYS
        for source in (AtmosTransport.Preprocessing.ERA5N320Settings,
                       AtmosTransport.Preprocessing.GEOSITSettings,
                       AtmosTransport.Preprocessing.MERRA2Settings)
            cfg=Dict("preprocessing"=>Dict(key=>"cpu"))
            @test_throws ArgumentError common.reject_unsupported_flux_keys(source,cfg)
        end
    end
    @test_throws ArgumentError common.resolve_spectral_construction(
        Dict("preprocessing"=>Dict("spectral_flux_constructon"=>"tm5_spectral")))
    @test_throws ArgumentError common.resolve_spectral_construction(
        Dict("preprocessing"=>Dict("quadrature_refinement_steps"=>2)))
end
