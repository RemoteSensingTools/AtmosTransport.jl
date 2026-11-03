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
    # Live Calc_Kzz inputs share the spectral [preprocessing] table.
    @test c.resolve_spectral_construction(Dict("preprocessing"=>Dict("include_tm5_pbl"=>true))) == NamedTuple()
    combined=c.resolve_spectral_construction(Dict("preprocessing"=>
        Dict("spectral_flux_construction"=>"tm5_spectral","include_tm5_pbl"=>true)))
    @test combined.spectral_flux_construction isa c.TM5SpectralFluxes
    config=Dict{String,Any}("preprocessing"=>Dict{String,Any}("spectral_flux_construction"=>"tm5_spectral","spectral_synthesis"=>"cpu"))
    settings=c.resolve_spectral_construction(config)
    @test settings.spectral_flux_construction isa c.TM5SpectralFluxes
    @test c.TM5SpectralFluxes().refinement_steps == 2
    @test settings.spectral_flux_construction.refinement_steps == 2
    @test settings.spectral_flux_construction.face_quadrature isa c.GaussFourier
    @test c.TM5SpectralFluxes().face_quadrature isa c.GaussFourier
    explicit=deepcopy(config);explicit["preprocessing"]["quadrature_refinement_steps"]=2
    explicit["preprocessing"]["face_quadrature"]="tm5_trapezoid"
    tm5=c.resolve_spectral_construction(explicit).spectral_flux_construction
    @test tm5.refinement_steps == c.TM5SpectralFluxes(2).refinement_steps == 2
    @test tm5.face_quadrature isa c.TM5Trapezoid
    # Independent ShRefinement rule: ceil(nstep * edge / spectral spacing).
    for T in (255,639), edge in (0.1,1.,5.)
        expected=max(1,ceil(Int,2*edge/(360/(2*(T+1)))))
        @test SS.quadrature_refinement(T,edge,tm5.refinement_steps) == expected
    end
    @test SS.quadrature_refinement(255,1.,tm5.refinement_steps) == 3
    @test SS.quadrature_refinement(639,1.,tm5.refinement_steps) == 8
    @test settings.spectral_synthesis isa c.CPUSynthesis
    @test_throws ArgumentError AtmosTransport.Preprocessing.TransportBinary.tm5_construction_metadata(settings.spectral_flux_construction,0)
    for (key,value) in (("spectral_synthesis",true),("spectral_synthesis_float","half"),("quadrature_refinement_steps",false),("face_quadrature","bogus"),("face_quadrature",true))
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
                    @test gpu.ps ≈ cpu.ps rtol=(FT===Float64 ? 1e-12 : 1e-6)
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

@testset "Streamed CPU surface keeps the full contraction's rounding" begin
    # A narrow six-row GEMM at atmospheric truncation exercises the CPU
    # accumulation change that is absent from the small-degree day fixtures.
    T = 255
    rng = MersenneTwister(617)
    rings = SS.SynthesisRings(collect(range(-1.5, 1.5; length=600)), fill(513, 600), fill(-pi, 600))
    spectra = randn(rng, ComplexF64, T+1, T+1, 3)
    full = SS.CPURingSynthesis(SS.RingSynthesisPlan(rings, T, 3; stable_poles=true))
    expected = SS.synthesize_rings(spectra, full)
    rows = [1:20; 301:305; 596:600]
    bandrings = SS.SynthesisRings(rings.latitudes[rows], rings.nlon[rows], rings.origins[rows])
    band = SS.CPURingSynthesis(SS.RingSynthesisPlan(bandrings, T, 3; stable_poles=true))
    output = similar(expected, bandrings.offsets[end]-1, 3)
    actual = SS.band_synthesis(spectra, band; out=output, layout=(; rows, total=length(rings.nlon)))
    samples = reduce(vcat, [collect(rings.offsets[r]:rings.offsets[r+1]-1) for r in rows])
    @test actual === output
    @test reinterpret(UInt64, vec(actual)) == reinterpret(UInt64, vec(expected[samples, :]))
end

@testset "Native CPU panels preserve global column positions" begin
    # Dense spectra catch edge GEMM reductions even when low-degree wind
    # fixtures have zero coefficients there. Include a one-level native
    # batch (three packed fields), tile edges, gaps and the padded final tile.
    rng = MersenneTwister(718)
    T = 63
    rings = SS.SynthesisRings(collect(range(-1.5, 1.5; length=70)), fill(135, 70), fill(-pi, 70))
    for FT in (Float32, Float64), nf in (3, 6, 24)
        spectra = randn(rng, Complex{FT}, T+1, T+1, nf)
        full = SS.CPURingSynthesis(SS.RingSynthesisPlan(rings, T, nf, FT; stable_poles=true, fft_threads=2))
        expected = SS.band_synthesis(spectra, full; native_panels=true)
        for rows in ([1:7; 31:34; 69:70], collect(28:66))
            bandrings = SS.SynthesisRings(rings.latitudes[rows], rings.nlon[rows], rings.origins[rows])
            band = SS.CPURingSynthesis(SS.RingSynthesisPlan(bandrings, T, nf, FT; stable_poles=true, fft_threads=2))
            actual = SS.band_synthesis(spectra, band; layout=(; rows, total=70), native_panels=true)
            samples = reduce(vcat, [collect(rings.offsets[r]:rings.offsets[r+1]-1) for r in rows])
            UT = FT === Float32 ? UInt32 : UInt64
            @test reinterpret(UT, vec(actual)) == reinterpret(UT, vec(expected[samples, :]))
        end
    end
end

@testset "CPU Fourier integrals keep their FFT shape across bands" begin
    PP = AtmosTransport.Preprocessing
    grid = PP.build_target_geometry(Val(:synthetic_reduced_gaussian),
        Dict("gaussian_number"=>8, "nlon_mode"=>"octahedral"), Float64)
    T = 31
    c = PP.TM5SpectralFluxes()
    q = SS.SpectralQuadrature(grid, T, c.face_quadrature)
    rng = MersenneTwister(910)
    for FT in (Float32, Float64), threads in (1, 2)
        values = randn(rng, FT, q.rings.offsets[end]-1, 2)
        s = SS.CPURingSynthesis(SS.RingSynthesisPlan(q.rings, T+1, 2, FT; fft_threads=threads))
        plan = SS.fourier_integral_plan(q.rings, q.interval_counts, s, 2; selected_rows=unique(q.boundary_rows))
        expected = SS.fourier_integrals!(values, plan, 2)
        for cells in (1:1, 2:3, 4:8)
            band = SS.quadrature_band(q, cells)
            bq = band.quadrature
            bs = SS.CPURingSynthesis(SS.RingSynthesisPlan(bq.rings, T+1, 2, FT; fft_threads=threads))
            bp = SS.fourier_integral_plan(bq.rings, bq.interval_counts, bs, 2; selected_rows=unique(bq.boundary_rows))
            samples = reduce(vcat, [collect(q.rings.offsets[r]:q.rings.offsets[r+1]-1) for r in band.rows])
            actual = SS.fourier_integrals!(values[samples, :], bp, 2)
            for row in unique(bq.boundary_rows)
                globalrow = band.rows[row]
                a = actual[bp.offsets[row]:bp.offsets[row+1]-1, :]
                b = expected[plan.offsets[globalrow]:plan.offsets[globalrow+1]-1, :]
                UT = FT === Float32 ? UInt32 : UInt64
                @test reinterpret(UT, vec(a)) == reinterpret(UT, vec(b))
            end
        end
    end
end
