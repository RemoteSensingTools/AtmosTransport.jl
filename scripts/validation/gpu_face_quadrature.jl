# Float32 device qualification. CPU reference and emulation never load CUDA.
# julia --project=. scripts/validation/gpu_face_quadrature.jl MODE T GRID OUTPUT
# MODE: reference | device | emulate; GRID: ll | rg | cs | sample
using AtmosTransport, LinearAlgebra, FFTW, Dates, TOML, JLD2, KernelAbstractions, SHA
const PP = AtmosTransport.Preprocessing
const SP = PP.Spectral
const REPO = normpath(joinpath(@__DIR__, "../.."))
include(joinpath(@__DIR__,"gpu_face_memory.jl"))
BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

struct EmulatedFaceDevice <: PP.Common.AbstractSynthesisBackend end
function SP.build_ring_synthesis(::EmulatedFaceDevice, rings, T, batch, precision; stable_poles=false)
    p = SP.RingSynthesisPlan(rings,T,batch,Float32;stable_poles)
    return SP.GPUSpectralSynthesis{Float32,typeof(p)}(p)
end

function validation_grid(name)
    name == "cs" && return PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>90),Float64)
    name == "rg" && return PP.build_target_geometry(Val(:synthetic_reduced_gaussian),
        Dict("gaussian_number"=>64,"nlon_mode"=>"octahedral"),Float64)
    name in ("ll","sample") || error("unknown grid $name")
    return PP.build_target_geometry(Dict("nlon"=>360,"nlat"=>180),Float64)
end

function validation_workspace(grid, name, T, c, synthesis; cells=32)
    opts = (; synthesis,precision=synthesis isa PP.Common.CPUSynthesis ? Float64 : Float32,batch=8)
    name == "cs" && return PP.CubedSphereTarget.tm5_cs_integral_workspace(grid,T,c;opts...)
    return SP.SpectralIntegralWorkspace(grid,T,c;opts...,latitude_band_cells=cells)
end

function integrate_validation(ws, spec, aa, bb, c)
    raw = SP.integrate_face_fluxes!(ws,spec.lnsp,spec.vo,spec.d,spec.T,aa,bb,c;host_output=false)
    # In device mode this catches an accidental Float64 output/host fallback.
    if ws.synthesis isa SP.GPUSpectralSynthesis{Float32}
        for x in raw
            eltype(x) === Float32 || error("Float64 result on Float32 device path")
            typeof(get_backend(x)) === typeof(get_backend(ws.synthesis.plan.coeffs)) || error("host fallback")
        end
    end
    return map(Array,raw)
end

function validation_cases(T, out)
    paths = SP.era5_spectral_paths(expanduser("~/data/AtmosTransport/met/era5/spectral_hourly"),Date(2021,12,1))
    real = SP.with_spectral_hours(paths;T_target=T,cache_dir=joinpath(out,"hour_cache")) do hour
        hour(0)
    end
    real.T == T || error("input truncation $(real.T) cannot supply T$T")
    coef = TOML.parsefile(joinpath(REPO,"config/era5_L137_coefficients.toml"))["coefficients"]
    h = zeros(ComplexF64,T+1,T+1); h[1] = log(100000.); h[3,2] = 0.01
    vo = zeros(ComplexF64,T+1,T+1,3); d = copy(vo)
    vo[2,1,1] = 1e-6
    vo[T+1,T+1,2] = (1+0.3im)*1e-7
    d[T+1,T÷2+1,3] = (0.4+im)*1e-7
    harmonics = (;lnsp=h,vo,d,T)
    return (("harmonics",harmonics,zeros(3),ones(3)),
        ("era5_20211201_00",real,diff(coef["a"]),diff(coef["b"])))
end

function level_metrics(actual, reference; face_rms=nothing)
    a = reshape(Float64.(actual), :, ndims(actual) == 1 ? 1 : size(actual,ndims(actual)))
    r = reshape(reference, size(a))
    return map(axes(r,2)) do k
        ref = view(r,:,k)
        delta = view(a,:,k).-ref
        rms = face_rms === nothing ? norm(ref)/sqrt(max(length(ref),1)) : face_rms[k]
        denominator = max(rms,eps(Float64))
        (;level=k,reference_rms=rms,max_absolute=maximum(abs,delta;init=0.0),
          rms_scaled=maximum(abs,delta;init=0.0)/denominator,
          relative_l2=norm(delta)/max(norm(ref),eps(Float64)))
    end
end

function face_metrics(actual, reference; face_rms=nothing)
    delta = abs.(Float64.(actual).-reference)
    peak = isempty(reference) ? 0.0 : maximum(abs,reference)
    scale = max(peak,1.0)
    # Per-face mixed bound: 1e-6 relative plus a 2e-14 * field-peak floor
    # for symmetry zeros. Report absolute error too; units are Pa or kg/s.
    bound = @. 1e-6*abs(reference)+2e-14*scale
    resolved = abs.(reference) .> 1e-12*scale
    relative = delta[resolved]./abs.(reference[resolved])
    return (;max_absolute=isempty(delta) ? 0.0 : maximum(delta),peak_scaled=isempty(delta) ? 0.0 : maximum(delta)/scale,
        max_relative=isempty(relative) ? 0.0 : maximum(relative),
        outside_bound=count(delta.>bound),relative_l2=norm(delta)/max(norm(reference),eps()),
        max_level_rms_scaled=maximum(m.rms_scaled for m in level_metrics(actual,reference;face_rms)),
        max_level_relative_l2=maximum(m.relative_l2 for m in level_metrics(actual,reference;face_rms)))
end

function validate_faces(mode,T,name,out; repetition=1)
    repetition in (1,2) || error("repetition must be 1 or 2")
    mkpath(out)
    if mode == "device"
        CUDA.functional() || error("CUDA unavailable")
        CUDA.allowscalar(false)
        CUDA.math_mode!(CUDA.DEFAULT_MATH)
        SP.spectral_gpu_precision() === Float32 || error("qualification requires the Float32 GPU policy")
        CUDA.versioninfo()
        start_face_device_memory(joinpath(out,"device_T$(T)_$(name)_run$(repetition)_memory.csv"))
    elseif mode == "emulate"
        @info "CPU emulation: vendor GPU GEMM/FFT error is not measured"
    elseif mode != "reference"
        error("unknown mode $mode")
    end
    synthesis = mode == "reference" ? PP.Common.CPUSynthesis() :
        mode == "device" ? PP.Common.GPUSynthesis() : EmulatedFaceDevice()
    c = PP.TM5SpectralFluxes()
    grid = validation_grid(name)
    cases = validation_cases(T,out)
    # A sample mode permits bounded CPU investigations at the poles and equator.
    # Qualification uses full ll/rg/cs targets, never sample mode.
    workspaces = if name == "sample"
        q = SP.SpectralQuadrature(grid,T,c.face_quadrature,c.refinement_steps)
        [SP.spectral_integral_workspace(SP.quadrature_band(q,rows).quadrature,T,c;
            synthesis,precision=mode == "reference" ? Float64 : Float32,batch=8) for rows in (1:2,90:91,179:180)]
    else
        [validation_workspace(grid,name,T,c,synthesis)]
    end
    reference_path = joinpath(out,"reference_T$(T)_$(name).jld2")
    reference = mode == "reference" ? Dict{String,Any}() : JLD2.load(reference_path,"results")
    fingerprint_path = joinpath(out,"device_T$(T)_$(name)_fingerprints.toml")
    previous_hashes = mode == "device" && repetition == 2 ? TOML.parsefile(fingerprint_path) : Dict{String,Any}()
    fingerprints = Dict{String,String}()
    report = joinpath(out,"$(mode)_T$(T)_$(name).csv")
    level_report = joinpath(out,"$(mode)_T$(T)_$(name)_levels.csv")
    level_io = open(level_report,"w")
    println(level_io,"case,region,field,level,reference_rms,max_absolute,rms_scaled,relative_l2")
    open(report,"w") do io
        println(io,"case,region,field,max_absolute,peak_scaled,max_relative_resolved_faces,outside_1e6_mixed_bound,relative_l2,max_level_rms_scaled,max_level_relative_l2,repeat_bitwise,band_bitwise,band_peak_scaled,fresh_process_bitwise")
        for (label,spec,aa,bb) in cases, (region,ws) in enumerate(workspaces)
            key = "$(label)_$(region)"
            raw = integrate_validation(ws,spec,aa,bb,c)
            if mode == "reference"
                reference[key] = raw
                continue
            end
            repeated = integrate_validation(ws,spec,aa,bb,c)
            other = if name in ("ll","rg")
                alternate = validation_workspace(grid,name,T,c,synthesis;cells=4)
                integrate_validation(alternate,spec,aa,bb,c)
            else
                raw
            end
            for field in keys(raw)
                ref = reference[key]
                # A shared horizontal-face RMS avoids a singular denominator
                # for a direction that vanishes by symmetry (zonal rotation).
                face_rms = field in (:u,:v) ? [sqrt((sum(abs2,view(ref.u,:,k))+sum(abs2,view(ref.v,:,k)))/
                    (size(ref.u,1)+size(ref.v,1))) for k in axes(ref.u,2)] : nothing
                metrics = face_metrics(raw[field],ref[field];face_rms)
                for m in level_metrics(raw[field],ref[field];face_rms)
                    println(level_io,join((label,region,field,values(m)...),','))
                end
                if field === :u
                    for k in axes(ref.u,2)
                        du = Float64.(view(raw.u,:,k)).-view(ref.u,:,k)
                        dv = Float64.(view(raw.v,:,k)).-view(ref.v,:,k)
                        absolute = max(maximum(abs,du;init=0.0),maximum(abs,dv;init=0.0))
                        relative_l2 = sqrt((sum(abs2,du)+sum(abs2,dv))/
                            max(sum(abs2,view(ref.u,:,k))+sum(abs2,view(ref.v,:,k)),eps(Float64)^2))
                        println(level_io,join((label,region,"horizontal",k,face_rms[k],absolute,
                            absolute/max(face_rms[k],eps(Float64)),relative_l2),','))
                    end
                end
                flush(level_io)
                fingerprint_key = "$(key)_$(field)"
                fingerprint = string(size(raw[field]), ":", bytes2hex(sha256(reinterpret(UInt8,vec(raw[field])))))
                fingerprints[fingerprint_key] = fingerprint
                fresh_bits = mode != "device" || repetition == 1 || get(previous_hashes,fingerprint_key,"") == fingerprint
                repeat_bits = reinterpret(UInt8,vec(raw[field])) == reinterpret(UInt8,vec(repeated[field]))
                band_bits = reinterpret(UInt8,vec(raw[field])) == reinterpret(UInt8,vec(other[field]))
                sensitivity = face_metrics(other[field],raw[field]).peak_scaled
                println(io,join((label,region,field,values(metrics)...,repeat_bits,band_bits,sensitivity,fresh_bits),','))
                flush(io)
                @info "face accuracy" mode T name label region field metrics repeat_bits band_bits sensitivity fresh_bits
                all(isfinite,raw[field]) || error("nonfinite face output")
            end
        end
    end
    close(level_io)
    mode == "reference" && JLD2.jldsave(reference_path;results=reference)
    if mode == "device" && repetition == 1
        open(io->TOML.print(io,fingerprints),fingerprint_path,"w")
    end
    mode == "device" && record_face_device_memory(joinpath(out,"device_T$(T)_$(name)_run$(repetition)_pool.toml"))
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (4,5) || error("usage: reference|device|emulate T ll|rg|cs|sample OUTPUT [1|2]")
    ARGS[1] == "device" && (@eval using CUDA)
    if ARGS[1] == "emulate"
        # Override only this standalone process, at top level before entering
        # validation. Execute the device branches on KA.CPU with current-world
        # dispatch, including staging kernels and compensated reductions.
        @eval SP begin
            face_accumulation(::KernelAbstractions.CPU,::Type{Float32}) = Val(true)
            surface_spectra(b::KernelAbstractions.CPU,H,T,p) =
                invoke(surface_spectra,Tuple{Any,Any,Any,Any},b,H,T,p)
            spectral_level_batch(b::KernelAbstractions.CPU,spec,levels,nf,p) =
                invoke(spectral_level_batch,Tuple{Any,Any,Any,Any,Any},b,spec,levels,nf,p)
            cs_polar_coefficients(b::KernelAbstractions.CPU,vo,d,R,p) =
                invoke(cs_polar_coefficients,Tuple{Any,Any,Any,Any,Any},b,vo,d,R,p)
        end
    end
    validate_faces(ARGS[1],parse(Int,ARGS[2]),ARGS[3],abspath(ARGS[4]); repetition=length(ARGS) == 5 ? parse(Int,ARGS[5]) : 1)
end
