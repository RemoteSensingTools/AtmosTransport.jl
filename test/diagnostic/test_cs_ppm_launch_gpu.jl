# Opt-in launch-layout regression on the explicitly authorized CUDA device.
# ATMOSTR_RUN_CS_PPM_LAUNCH_GPU_TESTS=1 ATMOSTR_PPM_GPU_NAME=V100 \
# CUDA_VISIBLE_DEVICES=<authorized UUID> julia --project=. test/diagnostic/test_cs_ppm_launch_gpu.jl
using Test
if get(ENV,"ATMOSTR_RUN_CS_PPM_LAUNCH_GPU_TESTS","0") != "1"
    @info "Skipping opt-in CS PPM launch tests"
else
    using CUDA, AtmosTransport, KernelAbstractions
    using AtmosTransport.Operators.Advection: MonotoneLimiter
    const LaunchAdv = AtmosTransport.Operators.Advection
    expected_device = get(ENV,"ATMOSTR_PPM_GPU_NAME","A100")
    isempty(expected_device) && error("ATMOSTR_PPM_GPU_NAME must name the authorized device")
    @assert occursin(expected_device,CUDA.name(CUDA.device())) "Wrong device for CS PPM launch tests"
    CUDA.allowscalar(false)

    function check_ppm_panel_launch(FT, Nc, Nz, Nt, direction)
        Hp=3; N=Nc+2Hp
        m=FT[1000 + 5i + 3j + k for i in 1:N, j in 1:N, k in 1:Nz]
        rm=Array{FT}(undef,N,N,Nz,Nt)
        for t in 1:Nt, k in 1:Nz, j in 1:N, i in 1:N
            q=t%3==0 ? zero(FT) : FT((isodd(t) ? -1 : 1)*(4e-4+1e-5*sin(i/3)+2e-5*cos(j/4)+1e-5*sin(k+t)))
            rm[i,j,k,t]=q*m[i,j,k]
        end
        dims=direction==:x ? (N+1,N,Nz) : direction==:y ? (N,N+1,Nz) : (N,N,Nz+1)
        flux=FT[40sin(i/3+j/4+k) for i in 1:dims[1],j in 1:dims[2],k in 1:dims[3]]
        if direction==:z
            flux[:,:,1].=0;flux[:,:,end].=0
        end
        drm,dm,df=CuArray(rm),CuArray(m),CuArray(flux)
        expected_rm=CUDA.fill(FT(-999),size(rm));expected_m=CUDA.fill(FT(-999),size(m))
        actual_rm=similar(expected_rm);fill!(actual_rm,FT(-999))
        actual_m=similar(expected_m);fill!(actual_m,FT(-999))
        scheme=PPMScheme(MonotoneLimiter())
        kernel=getproperty(LaunchAdv,Symbol("_cs_",direction,"sweep_mt_kernel!"))
        size_arg=Int32(direction==:z ? Nz : Nc)
        # The original 256-thread kernel is the device reference. CPU uses the
        # same mathematical reconstruction through its independent KA backend.
        kernel(get_backend(drm),256)(expected_rm,drm,expected_m,dm,df,scheme,
            size_arg,Int32(Hp),Int32(Nt),FT(0.75);ndrange=(Nc,Nc,Nz))
        launch=getproperty(LaunchAdv,Symbol("_sweep_",direction,"_panel_mt_pingpong!"))
        launch(actual_rm,actual_m,drm,dm,df,scheme,Nc,Hp,Nz,Nt;flux_scale=FT(0.75))
        CUDA.synchronize()
        reference_rm,reference_m=Array(expected_rm),Array(expected_m)
        @test Array(actual_rm)==reference_rm
        @test Array(actual_m)==reference_m
        @test Array(drm)==rm
        @test Array(dm)==m
        @test all(isfinite,Array(actual_rm))

        cpu_rm=fill(FT(-999),size(rm));cpu_m=fill(FT(-999),size(m))
        kernel(KernelAbstractions.CPU(),256)(cpu_rm,rm,cpu_m,m,flux,scheme,
            size_arg,Int32(Hp),Int32(Nt),FT(0.75);ndrange=(Nc,Nc,Nz))
        KernelAbstractions.synchronize(KernelAbstractions.CPU())
        interior=(Hp+1:Hp+Nc,Hp+1:Hp+Nc,Colon())
        @test reference_rm[interior...,:] ≈ cpu_rm[interior...,:] rtol=16eps(FT)
        @test reference_m[interior...] ≈ cpu_m[interior...] rtol=4eps(FT)


    end

    @testset "CS PPM launch layout preserves signed storage and halos" begin
        for FT in (Float32,Float64), Nc in (5,35), Nt in (1,6,7,32,65), direction in (:x,:y,:z)
            check_ppm_panel_launch(FT,Nc,7,Nt,direction)
        end
    end
end
