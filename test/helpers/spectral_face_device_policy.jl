# Isolated process: replacing CPU dispatch here cannot change another test's
# production CPU policy. No CUDA is imported, including on GPU test hosts.
using Test, AtmosTransport, LinearAlgebra, KernelAbstractions
const PP = AtmosTransport.Preprocessing
const SP = PP.Spectral
struct EmulatedFacePolicy <: PP.Common.AbstractSynthesisBackend end
function SP.build_ring_synthesis(::EmulatedFacePolicy, rings, T, batch, precision; stable_poles=false)
    p = SP.RingSynthesisPlan(rings, T, batch, Float32; stable_poles)
    return SP.GPUSpectralSynthesis{Float32,typeof(p)}(p)
end
const T = 7
const h = zeros(ComplexF64,T+1,T+1)
h[1] = log(100000.); h[3,2] = 0.02+0.003im
const vo = zeros(ComplexF64,T+1,T+1,3)
const d = copy(vo)
vo[2,1,:] .= 1e-6; vo[7,4,:] .= 2e-8+3e-8im; d[6,3,:] .= 1e-8
const c = PP.TM5SpectralFluxes()
const args = (h,vo,d,T,[2000.,0.,-2000.],[0.2,0.3,0.5],c)
const grids = (PP.build_target_geometry(Dict("nlon"=>24,"nlat"=>12),Float64),
    PP.build_target_geometry(Val(:synthetic_reduced_gaussian),Dict("gaussian_number"=>4,"nlon_mode"=>"octahedral"),Float64))
const references = map(g -> SP.integrate_face_fluxes!(SP.SpectralIntegralWorkspace(g,T,c;batch=2),args...), grids)
const csgrid = PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>3),Float64)
const csreference = SP.integrate_face_fluxes!(PP.CubedSphereTarget.tm5_cs_integral_workspace(csgrid,T,c;batch=2),args...)

@eval SP begin
    face_accumulation(::KernelAbstractions.CPU, ::Type{Float32}) = Val(true)
    surface_spectra(b::KernelAbstractions.CPU,H,T,p) =
        invoke(surface_spectra,Tuple{Any,Any,Any,Any},b,H,T,p)
    spectral_level_batch(b::KernelAbstractions.CPU,spec,levels,nf,p) =
        invoke(spectral_level_batch,Tuple{Any,Any,Any,Any,Any},b,spec,levels,nf,p)
    cs_polar_coefficients(b::KernelAbstractions.CPU,vo,d,R,p) =
        invoke(cs_polar_coefficients,Tuple{Any,Any,Any,Any,Any},b,vo,d,R,p)
end

@testset "Full Float32 device policy on KA.CPU" begin
    @test SP.face_accumulation(KernelAbstractions.CPU(),Float32) === Val(true)
    for (grid, reference) in zip(grids,references), cells in (0,1,3)
        ws = SP.SpectralIntegralWorkspace(grid,T,c;batch=2,latitude_band_cells=cells,synthesis=EmulatedFacePolicy())
        @test eltype(SP.spectral_level_batch(vo,3:3,2,ws.synthesis.plan)) === ComplexF32
        surface, scale = SP.surface_pressure_spectra(h,T,ws.surface)
        @test eltype(surface) === ComplexF32
        @test iszero(surface[1,1,1]) && scale isa Float32
        if ws.polar_cache !== nothing
            plans = copy(ws.polar_cache.plans)
            # All shapes share bounded tables instead of retaining a separate
            # latitude table family for every band or endpoint.
            @test all(parent(p.derivative.plan.legendre[1]) === ws.polar_cache.tables[1][1] for p in Base.values(plans))
        end
        actual = SP.integrate_face_fluxes!(ws,args...;host_output=false)
        repeated = SP.integrate_face_fluxes!(ws,args...;host_output=false)
        changed = SP.integrate_face_fluxes!(ws,h,2vo,2d,T,args[5],args[6],c;host_output=false)
        for field in keys(reference)
            @test eltype(actual[field]) === Float32
            @test get_backend(actual[field]) isa KernelAbstractions.CPU
            @test reinterpret(UInt32,vec(actual[field])) == reinterpret(UInt32,vec(repeated[field]))
            @test norm(actual[field]-reference[field])/max(norm(reference[field]),eps()) < 2e-5
            @test changed[field] == (field === :ps ? actual[field] : 2actual[field])
        end
        if ws.polar_cache !== nothing
            @test keys(plans) == keys(ws.polar_cache.plans)
            @test all(ws.polar_cache.plans[k] === p for (k,p) in plans)
        end
    end
    ws = PP.CubedSphereTarget.tm5_cs_integral_workspace(csgrid,T,c;batch=2,synthesis=EmulatedFacePolicy())
    actual = SP.integrate_face_fluxes!(ws,args...;host_output=false)
    for field in keys(csreference)
        @test eltype(actual[field]) === Float32
        @test norm(actual[field]-csreference[field])/max(norm(csreference[field]),eps()) < 2e-5
    end
    # Exercise FMA multiplication residuals inside a launched production
    # gather, with the automatic production policy and non-unit products.
    op = (; offsets=[1,3], indices=[1,2], weights=Float32[4097,1])
    values = reshape(Float32[4097,-16785408],2,1)
    output = zeros(Float32,1,1)
    SP.cs_apply_integral!(output,values,op)
    @test output[1] === 1f0
    SP.cs_apply_integral!(output,values,op;accumulation=Val(false))
    @test output[1] === 0f0
    @test !isdefined(Main,:CUDA)
end
