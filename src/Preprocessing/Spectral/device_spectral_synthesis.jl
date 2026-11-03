# All GPU dependencies are extension hooks. The same KA/GEMM/FFT implementation
# runs on CPU arrays, including in ordinary CI; CUDA never loads from core.
"""Portable batched ring transform buffers (CPU arrays form the reference)."""
struct RingSynthesisPlan{A, L, C, S, G, O, D}
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
    fft_threads::Int
end

"""Reference ring synthesis using host GEMM and unnormalised FFTW."""
struct CPURingSynthesis{P} <: AbstractSpectralSynthesis
    plan::P
end

"""Device ring synthesis in the selected precision; surfaces may choose it too."""
struct GPUSpectralSynthesis{FT, A} <: AbstractSpectralSynthesis
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
cpu_band_workers() = min(4, max(1, Threads.nthreads()÷4))
cpu_band_fft_threads() = min(2, Threads.nthreads()÷cpu_band_workers())
build_ring_synthesis(::CPUSynthesis, rings, T, batch, precision; stable_poles=false) =
    CPURingSynthesis(RingSynthesisPlan(rings, T, batch, precision isa AutoSynthesisPrecision ? Float64 : precision;
        stable_poles, fft_threads=stable_poles ? cpu_band_fft_threads() : 0))
build_ring_synthesis(::GPUSynthesis, rings, T, batch, precision; stable_poles=false) = GPUSpectralSynthesis(rings, T, batch; precision, stable_poles)

function RingSynthesisPlan(rings::SynthesisRings, T::Integer, batch::Integer,
                           ::Type{FT} = Float64;
                           adapter = identity, legendre = nothing, stable_poles=false, fft_threads=0) where FT
    nr = length(rings.nlon)
    tables = legendre
    if tables === nothing
        tables = [zeros(FT, T-m+1, nr) for m in 0:T]
        fill_ring_legendre!(tables, rings, T; stable_poles)
        tables = adapter.(tables)
    end
    groups = map(sort(unique(rings.nlon))) do J
        rows = findall(==(J), rings.nlon)
        half = adapter(zeros(Complex{FT}, J÷2+1, batch, length(rows)))
        plan = fft_threads == 0 ? plan_brfft(half, J, 1) : plan_brfft(half, J, 1; num_threads=fft_threads)
        samples = adapter(zeros(FT, J, batch, length(rows)))
        (; J, rows = adapter(rows), half, plan, samples)
    end
    return RingSynthesisPlan(rings, Int(T), Int(batch), adapter, tables,
        adapter(zeros(FT, 2batch, T+1)), adapter(zeros(FT, 2batch, nr)), groups,
        adapter(Complex{FT}.([cis(m*origin) for m in 0:T, origin in rings.origins])), adapter(rings.offsets), Int(fft_threads))
end

function fill_ring_legendre!(tables, rings, T; stable_poles=false)
    nr = length(rings.nlon)
    FT = eltype(first(tables))
    columns = [zeros(T+1, T+1) for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for j in 1:nr
        P = columns[Threads.threadid()]
        if stable_poles
            compute_legendre_column!(P, Int(T), sin(rings.latitudes[j]), cos(rings.latitudes[j]))
        else
            compute_legendre_column!(P, Int(T), sin(rings.latitudes[j]))
        end
        for m in 0:T, n in m:T
            tables[m+1][n-m+1, j] = FT(P[n+1, m+1])
        end
    end
    return tables
end

CPURingSynthesis(rings::SynthesisRings, T::Integer, batch::Integer = 1) =
    CPURingSynthesis(RingSynthesisPlan(rings, T, batch))

function GPUSpectralSynthesis(rings::SynthesisRings, T::Integer, batch::Integer = 1;
                              precision = AutoSynthesisPrecision(), stable_poles=false)
    FT = precision isa AutoSynthesisPrecision ? spectral_gpu_precision() : precision
    FT in (Float32, Float64) || throw(ArgumentError("GPU spectral precision must be Float32 or Float64"))
    plan = RingSynthesisPlan(rings, T, batch, FT; adapter = spectral_gpu_adapter(), stable_poles)
    @info "GPU spectral synthesis" float_type = FT batch
    return GPUSpectralSynthesis{FT, typeof(plan)}(plan)
end

"""
Evaluate a batch by real/imaginary Legendre GEMM, then unnormalised inverse FFT.

Rings sharing a longitude count share an FFT batch. Phase factors account for
each ring's origin; the returned samples remain on the plan's backend.
"""
function synthesize_rings(spec, synthesis::Union{CPURingSynthesis, GPUSpectralSynthesis}; out=nothing)
    p = synthesis.plan
    size(spec) == (p.T+1, p.T+1, p.batch) || throw(DimensionMismatch("ring spectral batch"))
    return with_spectral_math(get_backend(p.coeffs)) do
        spectral_synthesis_guard(synthesis)
        CT = Complex{eltype(p.coeffs)}
        source = p.adapter(eltype(spec) === CT ? spec : CT.(spec))
        backend = get_backend(p.coeffs)
        nf = p.batch
        for group in p.groups
            fill!(group.half, 0)
        end
        for m in 0:p.T
            pack_ring_coefficients!(backend, 256)(p.coeffs, source, m, p.T, nf;
                ndrange = (p.T-m+1, nf))
            synchronize(backend)
            mul!(p.sums, view(p.coeffs, :, 1:p.T-m+1), p.legendre[m+1])
            for group in p.groups
                scatter_ring_spectrum!(backend, 256)(group.half, p.sums, group.rows,
                    p.phases, m, nf, p.T, group.J;
                    ndrange = (length(group.rows), nf))
            end
        end
        out === nothing && (out = similar(p.coeffs, eltype(p.coeffs), p.rings.offsets[end]-1, nf))
        for group in p.groups
            synchronize(backend)
            mul!(group.samples, group.plan, group.half)
            gather_ring_samples!(backend, 256)(out, group.samples, group.rows, p.offsets,
                group.J, nf;
                ndrange = (group.J, nf, length(group.rows)))
        end
        synchronize(backend)
        return out
    end
end
spectral_synthesis_guard(::CPURingSynthesis) = nothing
spectral_synthesis_guard(s::GPUSpectralSynthesis) = spectral_gpu_math_guard(get_backend(s.plan.coeffs))
spectral_gpu_math_guard(::KernelAbstractions.CPU) = nothing

"""Surface plan shares the selected tables when its precision matches the field plan."""
function surface_synthesis(s::GPUSpectralSynthesis, batch::Integer = 3; precision=Float64)
    p = s.plan
    plan = RingSynthesisPlan(p.rings, p.T, batch, precision; adapter = p.adapter,
        legendre = eltype(p.coeffs) === precision ? p.legendre : nothing)
    return GPUSpectralSynthesis{precision, typeof(plan)}(plan)
end
surface_synthesis(s::CPURingSynthesis, batch::Integer = 3; precision=Float64) =
    CPURingSynthesis(RingSynthesisPlan(s.plan.rings, s.plan.T, batch, precision;
        legendre = eltype(s.plan.coeffs) === precision ? s.plan.legendre : nothing,
        fft_threads=s.plan.fft_threads))

"""Synthesize native winds and divergence, using the portable vector kernel on devices."""
function synthesize_native_fields(vo, d, T, radius, s::Union{CPURingSynthesis,GPUSpectralSynthesis})
    u, v, divergence = native_spectral_components(vo, d, T, radius, s)
    return synthesize_rings(u, s), synthesize_rings(v, s), synthesize_rings(divergence, s)
end

synthesis_provenance(::CPURingSynthesis) = "cpu"
synthesis_provenance(::GPUSpectralSynthesis) = "gpu"
function spectral_synthesis_metadata(ws)
    metadata = Dict{String, Any}("spectral_synthesis" => synthesis_provenance(ws.synthesis),
        "spectral_synthesis_float" => string(eltype(ws.synthesis.plan.coeffs)),
        "spectral_surface_float" => string(eltype(ws.surface.plan.coeffs)))
    return metadata
end

# The Gauss/Fourier path transforms three components together. This gives the
# same Legendre operator a wider GEMM and removes two sets of per-mode launches.
native_transform_batch(::TM5Trapezoid, batch) = batch
native_transform_batch(::GaussFourier, batch) = 3batch
synthesize_quadrature_fields(::TM5Trapezoid, vo, d, T, radius, s) =
    synthesize_native_fields(vo, d, T, radius, s)

function native_spectral_components(vo, d, T, radius, s::CPURingSynthesis)
    u, v = complete_wind_coefficients(vo, d, T, radius)
    divergence = zeros(ComplexF64, T+2, T+2, size(vo, 3))
    divergence[1:T+1, 1:T+1, :] .= d
    return u, v, divergence
end
function native_spectral_components(vo, d, T, radius, s::GPUSpectralSynthesis{FT}) where FT
    nf = size(vo, 3)
    u = similar(s.plan.coeffs, Complex{FT}, T+2, T+2, nf)
    v = similar(u)
    divergence = similar(u)
    backend = get_backend(u)
    complete_wind_spectral_kernel!(backend, 256)(u, v, divergence,
        s.plan.adapter(Complex{FT}.(vo)), s.plan.adapter(Complex{FT}.(d)), T, FT(radius), nf;
        ndrange=(T+2, T+2, nf))
    synchronize(backend)
    return u, v, divergence
end

@kernel function pack_native_components!(spectra, u, v, divergence, nf)
    n, m, k = @index(Global, NTuple)
    if n <= size(u, 1) && m <= size(u, 2) && k <= nf
        spectra[n, m, k] = u[n, m, k]
        spectra[n, m, nf+k] = v[n, m, k]
        spectra[n, m, 2nf+k] = divergence[n, m, k]
    end
end

function quadrature_spectra(vo, d, T, radius, s)
    nf = size(vo, 3)
    u, v, divergence = native_spectral_components(vo, d, T, radius, s)
    FT = eltype(s.plan.coeffs)
    spectra = similar(s.plan.coeffs, Complex{FT}, T+2, T+2, 3nf)
    backend = get_backend(spectra)
    # Host coefficients are converted before adapting, so Float32 device
    # kernels never read Float64 arrays, including the component pack.
    adapt = s.plan.adapter
    pack_native_components!(backend, 256)(spectra, adapt(Complex{FT}.(u)),
        adapt(Complex{FT}.(v)), adapt(Complex{FT}.(divergence)), nf;
        ndrange=(T+2, T+2, nf))
    synchronize(backend)
    return spectra
end

function synthesize_quadrature_fields(::GaussFourier, vo, d, T, radius, s)
    nf = size(vo, 3)
    samples = synthesize_rings(quadrature_spectra(vo, d, T, radius, s), s)
    return view(samples, :, 1:nf), view(samples, :, nf+1:2nf), view(samples, :, 2nf+1:3nf)
end
