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
    # Without a loaded CUDA extension, or with a CUDA that is not functional, the
    # GPU backend is rejected. Once another test in the same process has loaded
    # CUDA on a GPU host, the extension's method accepts it.
    cuda_ext=Base.get_extension(AtmosTransport,:AtmosTransportCUDAExt)
    if cuda_ext!==nothing && cuda_ext.CUDA.functional()
        @test c.validate_gpu_spectral_backend()===nothing
    else
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

# The device branch is deliberately invoked on KA.CPU: these checks execute
# every new kernel without requiring CUDA or changing CPU production policy.
@testset "Device face staging and compensated reductions on KA CPU" begin
    KA = SS.KernelAbstractions
    backend = KA.CPU()
    rng = MersenneTwister(996)
    T, nf = 7, 2
    rings = SS.SynthesisRings([0.3], [19], [0.0])
    for FT in (Float32, Float64)
        p = SS.RingSynthesisPlan(rings, T, 2nf, FT)
        H = randn(rng, ComplexF64, T+1, T+1)
        vo, d = randn(rng, ComplexF64, T+1, T+1, nf), randn(rng, ComplexF64, T+1, T+1, nf)
        # Spectral inputs have a triangular support.
        for m in 1:T+1, n in 1:m-1
            H[n,m] = 0
            vo[n,m,:] .= 0
            d[n,m,:] .= 0
        end
        source = Complex{FT}.(H)
        staged = invoke(SS.surface_spectra, Tuple{Any,Any,Any,Any}, backend, H, T, p)
        reference = SS.surface_spectra(H, T, p)
        @test eltype(staged) === Complex{FT}
        @test staged ≈ reference rtol=(FT === Float32 ? 3e-7 : 1e-14)
        batch = invoke(SS.spectral_level_batch, Tuple{Any,Any,Any,Any,Any}, backend, vo, 2:2, 3, p)
        @test batch[:,:,1] == Complex{FT}.(vo[:,:,2])
        @test all(iszero, batch[:,:,2:3])
        actual = invoke(SS.cs_polar_coefficients, Tuple{Any,Any,Any,Any,Any}, backend, vo, d, 6.371f6, p)
        expected = SS.cs_polar_coefficients(backend, vo, d, 6.371f6, p)
        @test actual == expected
    end
    # Cancellation fixture: Float32 alone loses the middle unit. Neumaier
    # retains it using Float32 correction, including both complex components.
    for CT in (Float32, ComplexF32)
        total, correction = zero(CT), zero(CT)
        for x in CT.([1e8, 1, -1e8])
            total, correction = SS.face_sum_step(total, correction, x, Val(true))
        end
        @test SS.face_sum_result(total, correction, Val(true)) === one(CT)
    end
    op = (; offsets=[1,4], indices=[1,2,3], weights=ones(Float32,3))
    values = reshape(Float32[1e8,1,-1e8],3,1)
    output = zeros(Float32,1,1)
    SS.cs_apply_integral!(output, values, op; accumulation=Val(true))
    @test output[1] === 1f0
    SS.cs_apply_integral!(output, values, op)
    @test output[1] === 0f0 # historical CPU policy
    compact = SS.CSFaceGather([1,4], [1,2,3], ones(Float32,3), ones(Float32,3,1), nothing, 3)
    SS.cs_apply_integral!(output, repeat(values,3,1), compact; accumulation=Val(true))
    @test output[1] === 3f0
    # Three latitude nodes exercise all three compensated Gauss sums.
    pressure, u, C = zeros(Float32,1), zeros(Float32,1,1), zeros(Float32,1,1)
    SS.integrate_gauss_cells!(backend, 256)(pressure,u,C,values,values,values,
        [1,1,1],[1,2,3],[1,2,3],[1,4],[1,2,3],[1,2],[1],
        ones(Float32,3),ones(Float32,3),ones(Float32,1),1f0,1,Val(true); ndrange=(1,1,1))
    @test pressure == u[:] == C[:] == [1f0]
    # Complex frequency folding uses the same compensated primitive.
    folded = zeros(ComplexF32,1,1,1)
    SS.fold_fourier_integrals!(backend,256)(folded,reshape(ComplexF32[1,1,1],3,1,1),
        ComplexF32[1e8,1,-2e8,1,1e8],4,1,1,Val(true);ndrange=(1,1,1))
    @test folded[1] == ComplexF32(2)
end

struct HostFaceDevice{FT} <: AtmosTransport.Preprocessing.Common.AbstractSynthesisBackend end
function SS.build_ring_synthesis(::HostFaceDevice{FT},rings,T,batch,precision;stable_poles=false) where FT
    plan = SS.RingSynthesisPlan(rings,T,batch,FT;stable_poles)
    return SS.GPUSpectralSynthesis{FT,typeof(plan)}(plan)
end
@testset "Portable LL/RG complete face construction, streamed and resident" begin
    PP = AtmosTransport.Preprocessing
    T = 7
    h = zeros(ComplexF64,T+1,T+1); h[1] = log(100000.); h[3,2] = 0.02
    vo = zeros(ComplexF64,T+1,T+1,3); d = copy(vo)
    vo[2,1,:] .= 1e-6; d[T+1,T,:] .= 1e-8
    for grid in (PP.build_target_geometry(Dict("nlon"=>12,"nlat"=>6),Float64),
                 PP.build_target_geometry(Val(:synthetic_reduced_gaussian),Dict("gaussian_number"=>4,"nlon_mode"=>"octahedral"),Float64))
        c = PP.TM5SpectralFluxes()
        args = (h,vo,d,T,[2000.,0.,-2000.],[0.2,0.3,0.5],c)
        reference = SS.integrate_face_fluxes!(SS.SpectralIntegralWorkspace(grid,T,c;batch=2),args...)
        for FT in (Float32,Float64), cells in (0,1,3)
            ws = SS.SpectralIntegralWorkspace(grid,T,c;batch=2,latitude_band_cells=cells,synthesis=HostFaceDevice{FT}())
            actual = SS.integrate_face_fluxes!(ws,args...;host_output=false)
            repeated = SS.integrate_face_fluxes!(ws,args...)
            for name in keys(reference)
                @test reinterpret(UInt8,vec(actual[name])) == reinterpret(UInt8,vec(repeated[name]))
                error = norm(actual[name]-reference[name])/max(norm(reference[name]),eps())
                # This m=6 cosine nearly cancels on 12 longitude intervals;
                # Float32 retains ~2.2e-5 relative in the small covariance flux.
                # Qualification reports per-face residuals, without treating
                # this smoke tolerance as a 1e-6 accuracy certificate.
                @test error < (FT === Float32 ? 3e-5 : 1e-12)
                @test eltype(actual[name]) === FT
            end
        end
    end
end

@testset "Float32 pressure retains the log-pressure anomaly" begin
    T = 7
    rings = SS.SynthesisRings([-0.4,0.2],fill(31,2),zeros(2))
    H = zeros(ComplexF64,T+1,T+1)
    H[1,1] = log(100000.)
    H[3,2] = 0.01+0.003im
    cpu = SS.CPURingSynthesis(SS.RingSynthesisPlan(rings,T+1,3,Float64))
    plan = SS.RingSynthesisPlan(rings,T+1,3,Float32)
    device = SS.GPUSpectralSynthesis{Float32,typeof(plan)}(plan)
    spectra, scale = SS.surface_pressure_spectra(H,T,device)
    @test iszero(spectra[1,1,1])
    @test scale isa Float32
    samples = SS.synthesize_rings(spectra,device)
    pressure = zeros(Float32,size(samples,1),1)
    SS.surface_pressure_samples!(SS.KernelAbstractions.CPU(),256)(pressure,samples,scale;ndrange=length(pressure))
    expected = exp.(SS.synthesize_rings(SS.surface_spectra(H,T,cpu.plan),cpu)[:,1])
    @test maximum(abs.(vec(pressure).-expected)./expected) < 2e-7
end

@testset "Compensated Float32 products retain multiplication residuals" begin
    total,correction = SS.face_product_step(0f0,0f0,4097f0,4097f0,Val(true))
    total,correction = SS.face_sum_step(total,correction,-16785408f0,Val(true))
    @test SS.face_sum_result(total,correction,Val(true)) === 1f0
    total,correction = SS.face_product_step(0f0,0f0,4097f0,4097f0,Val(false))
    total,correction = SS.face_sum_step(total,correction,-16785408f0,Val(false))
    @test SS.face_sum_result(total,correction,Val(false)) === 0f0
    total,correction = SS.face_product_step(0f0+0f0im,0f0+0f0im,4097f0+4097f0im,4097f0+0f0im,Val(true))
    total,correction = SS.face_sum_step(total,correction,-16785408f0-16785408f0im,Val(true))
    @test SS.face_sum_result(total,correction,Val(true)) === 1f0+1f0im
end

@testset "Isolated complete Float32 device policy" begin
    driver = normpath(joinpath(@__DIR__, "../helpers/spectral_face_device_policy.jl"))
    project = dirname(Base.active_project())
    command = `$(Base.julia_cmd()) --startup-file=no --project=$project $driver`
    @test success(pipeline(addenv(command, "CUDA_VISIBLE_DEVICES"=>"", "OPENBLAS_NUM_THREADS"=>"1"); stdout=stdout, stderr=stderr))
end
