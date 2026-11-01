# All GPU dependencies are extension hooks. The same KA/GEMM/FFT implementation
# runs on CPU arrays, including in ordinary CI; CUDA never loads from core.
"""Portable batched ring transform buffers (CPU arrays form the reference)."""
struct RingSynthesisPlan{A,L,C,S,G,O,D}
    rings::SynthesisRings
    T::Int
    batch::Int
    adapter::A
    legendre::L
    coeffs::C
    sums::S
    groups::G
    phases::O
    offsets::D
end

"""Reference ring synthesis using host GEMM and unnormalised FFTW."""
struct CPURingSynthesis{P} <: AbstractSpectralSynthesis
    plan::P
end

"""Device ring synthesis; FT controls 3-D fields, surface synthesis is Float64."""
struct GPUSpectralSynthesis{FT,A} <: AbstractSpectralSynthesis
    plan::A
end

spectral_gpu_adapter() = spectral_gpu_adapter(Val(:cuda))
spectral_gpu_adapter(::Val) = throw(ArgumentError("load CUDA before requesting GPU spectral synthesis"))
spectral_gpu_precision() = spectral_gpu_precision(Val(:cuda))
spectral_gpu_precision(::Val) = Float64
spectral_gpu_math_guard() = spectral_gpu_math_guard(Val(:cuda))
spectral_gpu_math_guard(::Val) = nothing
with_spectral_math(f, backend) = f()
with_spectral_math(f, ::KernelAbstractions.CPU) = f()
build_ring_synthesis(::CPUSynthesis,rings,T,batch,precision) = CPURingSynthesis(rings,T,batch)
build_ring_synthesis(::GPUSynthesis,rings,T,batch,precision) = GPUSpectralSynthesis(rings,T,batch;precision)

function RingSynthesisPlan(rings::SynthesisRings, T::Integer, batch::Integer,
                           ::Type{FT}=Float64;
                           adapter=identity, legendre=nothing) where FT
    nr = length(rings.nlon)
    tables = legendre
    if tables === nothing
        tables = [zeros(FT,T-m+1,nr) for m in 0:T]
        columns = [zeros(T+1,T+1) for _ in 1:Threads.maxthreadid()]
        Threads.@threads :static for j in 1:nr
            P = columns[Threads.threadid()]
            compute_legendre_column!(P,Int(T),sin(rings.latitudes[j]))
            for m in 0:T, n in m:T
                tables[m+1][n-m+1,j] = FT(P[n+1,m+1])
            end
        end
        tables = adapter.(tables)
    end
    groups = map(sort(unique(rings.nlon))) do J
        rows = findall(==(J),rings.nlon)
        half = adapter(zeros(Complex{FT}, J÷2+1,batch,length(rows)))
        plan = plan_brfft(half,J,1)
        samples = adapter(zeros(FT,J,batch,length(rows)))
        (; J, rows=adapter(rows), half, plan, samples)
    end
    return RingSynthesisPlan(rings,Int(T),Int(batch),adapter,tables,
        adapter(zeros(FT,2batch,T+1)),adapter(zeros(FT,2batch,nr)),groups,
        adapter(Complex{FT}.([cis(m*origin) for m in 0:T, origin in rings.origins])),adapter(rings.offsets))
end

CPURingSynthesis(rings::SynthesisRings,T::Integer,batch::Integer=1) =
    CPURingSynthesis(RingSynthesisPlan(rings,T,batch))

function GPUSpectralSynthesis(rings::SynthesisRings,T::Integer,batch::Integer=1;
                              precision=AutoSynthesisPrecision())
    FT = precision isa AutoSynthesisPrecision ? spectral_gpu_precision() : precision
    FT in (Float32,Float64) || throw(ArgumentError("GPU spectral precision must be Float32 or Float64"))
    plan = RingSynthesisPlan(rings,T,batch,FT; adapter=spectral_gpu_adapter())
    @info "GPU spectral synthesis" float_type=FT surface_float_type=Float64 batch
    return GPUSpectralSynthesis{FT,typeof(plan)}(plan)
end

"""Synthesize batch columns; returned samples stay on the selected backend."""
function synthesize_rings(spec, synthesis::Union{CPURingSynthesis,GPUSpectralSynthesis})
    p = synthesis.plan
    size(spec) == (p.T+1,p.T+1,p.batch) || throw(DimensionMismatch("ring spectral batch"))
    return with_spectral_math(get_backend(p.coeffs)) do
        spectral_synthesis_guard(synthesis)
        source = p.adapter(spec)
        backend = get_backend(p.coeffs)
        nf = p.batch
        for group in p.groups
            fill!(group.half,0)
        end
        for m in 0:p.T
            pack_ring_coefficients!(backend,256)(p.coeffs,source,m,p.T,nf;
                ndrange=(p.T-m+1,nf))
            synchronize(backend)
            mul!(p.sums,view(p.coeffs,:,1:p.T-m+1),p.legendre[m+1])
            for group in p.groups
                scatter_ring_spectrum!(backend,256)(group.half,p.sums,group.rows,
                    p.phases,m,nf,p.T,group.J;
                    ndrange=(length(group.rows),nf))
            end
        end
        out = similar(p.coeffs,eltype(p.coeffs),p.rings.offsets[end]-1,nf)
        for group in p.groups
            synchronize(backend)
            mul!(group.samples,group.plan,group.half)
            gather_ring_samples!(backend,256)(out,group.samples,group.rows,p.offsets,
                group.J,nf;
                ndrange=(group.J,nf,length(group.rows)))
        end
        synchronize(backend)
        return out
    end
end
spectral_synthesis_guard(::CPURingSynthesis) = nothing
spectral_synthesis_guard(s::GPUSpectralSynthesis) = spectral_gpu_math_guard(get_backend(s.plan.coeffs))
spectral_gpu_math_guard(::KernelAbstractions.CPU) = nothing

"""Surface coefficients always use Float64, including on reduced-FP64 GPUs."""
function surface_synthesis(s::GPUSpectralSynthesis, batch::Integer=3)
    p = s.plan
    plan = RingSynthesisPlan(p.rings,p.T,batch,Float64;adapter=p.adapter,
        legendre=eltype(p.coeffs)===Float64 ? p.legendre : nothing)
    return GPUSpectralSynthesis{Float64,typeof(plan)}(plan)
end
surface_synthesis(s::CPURingSynthesis,batch::Integer=3) =
    CPURingSynthesis(RingSynthesisPlan(s.plan.rings,s.plan.T,batch;legendre=s.plan.legendre))

"""Synthesize native winds and divergence, using the portable vector kernel on devices."""
function synthesize_native_fields(vo,d,T,radius,s::CPURingSynthesis)
    u,v=complete_wind_coefficients(vo,d,T,radius)
    padded=zeros(ComplexF64,T+2,T+2,s.plan.batch)
    padded[1:T+1,1:T+1,:] .= d
    return synthesize_rings(u,s),synthesize_rings(v,s),synthesize_rings(padded,s)
end
function synthesize_native_fields(vo,d,T,radius,s::GPUSpectralSynthesis{FT}) where FT
    adapt=s.plan.adapter
    nf=s.plan.batch
    vv=adapt(Complex{FT}.(vo))
    dd=adapt(Complex{FT}.(d))
    u=similar(s.plan.coeffs,Complex{FT},T+2,T+2,nf)
    v=similar(u)
    divergence=similar(u)
    backend=get_backend(u)
    complete_wind_spectral_kernel!(backend,256)(u,v,divergence,vv,dd,T,radius,nf;
        ndrange=(T+2,T+2,nf))
    synchronize(backend)
    return synthesize_rings(u,s),synthesize_rings(v,s),synthesize_rings(divergence,s)
end

synthesis_provenance(::CPURingSynthesis) = "cpu"
synthesis_provenance(::GPUSpectralSynthesis) = "gpu"
function spectral_synthesis_metadata(ws)
    return Dict{String,Any}("spectral_synthesis" => synthesis_provenance(ws.synthesis),
        "spectral_synthesis_float" => string(eltype(ws.synthesis.plan.coeffs)),
        "spectral_surface_float" => "Float64")
end
