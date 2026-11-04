# Preserve historical CPU reductions. Device Float32 uses Neumaier sums;
# compensation cannot recover error already introduced by GEMM or FFT.
face_accumulation(backend, ::Type{FT}) where FT = Val(FT === Float32)
face_accumulation(::KernelAbstractions.CPU, ::Type) = Val(false)
@inline face_sum_step(total, correction, term, ::Val{false}) = (total+term, correction)
@inline function face_sum_step(total, correction, term, ::Val{true})
    next = total+term
    correction += abs(total) >= abs(term) ? (total-next)+term : (term-next)+total
    return next, correction
end
@inline face_sum_result(total, correction, ::Val{false}) = total
@inline face_sum_result(total, correction, ::Val{true}) = total+correction
# Fourier folding is complex: compensate its two real components separately.
@inline function face_sum_step(total::Complex, correction::Complex, term::Complex, mode::Val{true})
    re, cre = face_sum_step(real(total), real(correction), real(term), mode)
    im, cim = face_sum_step(imag(total), imag(correction), imag(term), mode)
    return complex(re, im), complex(cre, cim)
end

# Retain the rounding residual of each real product as well as the sum.
# fma recovers that residual using only the selected scalar precision.
@inline face_product_step(total,correction,a,b,mode::Val{false}) =
    face_sum_step(total,correction,a*b,mode)
@inline function face_product_step(total,correction,a::Real,b::Real,mode::Val{true})
    product = a*b
    total,correction = face_sum_step(total,correction,product,mode)
    return total,correction+fma(a,b,-product)
end
@inline function face_product_step(total::Complex,correction::Complex,a::Complex,b::Complex,mode::Val{true})
    re,cre = face_product_step(real(total),real(correction),real(a),real(b),mode)
    re,cre = face_product_step(re,cre,-imag(a),imag(b),mode)
    im,cim = face_product_step(imag(total),imag(correction),real(a),imag(b),mode)
    im,cim = face_product_step(im,cim,imag(a),real(b),mode)
    return complex(re,im),complex(cre,cim)
end

"""Pad a native-level batch on the array backend, converting before transfer."""
function spectral_level_batch(spec, levels, nf, p)
    return spectral_level_batch(get_backend(p.coeffs), spec, levels, nf, p)
end
function spectral_level_batch(::KernelAbstractions.CPU, spec, levels, nf, p)
    batch = zeros(ComplexF64, size(spec, 1), size(spec, 2), nf)
    batch[:, :, 1:length(levels)] .= view(spec, :, :, levels)
    return batch
end
function spectral_level_batch(backend, spec, levels, nf, p)
    FT = eltype(p.coeffs)
    source = p.adapter(Complex{FT}.(view(spec, :, :, levels)))
    batch = similar(p.coeffs, Complex{FT}, size(spec, 1), size(spec, 2), nf)
    pad_spectral_levels!(backend, 256)(batch, source; ndrange=size(batch))
    synchronize(backend)
    return batch
end
@kernel function pad_spectral_levels!(out, source)
    n, m, k = @index(Global, NTuple)
    if n <= size(out, 1) && m <= size(out, 2) && k <= size(out, 3)
        out[n, m, k] = k <= size(source, 3) ? source[n, m, k] : zero(eltype(out))
    end
end

"""Surface coefficients and exact gradients on the selected backend."""
surface_spectra(H, T, p) = surface_spectra(get_backend(p.coeffs), H, T, p)
function surface_spectra(::KernelAbstractions.CPU, H, T, p)
    gradients = lnsp_gradient(H, T)
    spectra = zeros(ComplexF64, T+2, T+2, 3)
    spectra[1:T+1, 1:T+1, 1] .= H
    spectra[:, :, 2] .= gradients[1]
    spectra[:, :, 3] .= gradients[2]
    return spectra
end
function surface_spectra(backend, H, T, p)
    FT = eltype(p.coeffs)
    spectra = similar(p.coeffs, Complex{FT}, T+2, T+2, 3)
    source = p.adapter(Complex{FT}.(H))
    surface_gradient_coefficients!(backend, 256)(spectra, source, T; ndrange=(T+2, T+2))
    synchronize(backend)
    return spectra
end
@kernel function surface_gradient_coefficients!(out, H, T)
    ni, mi = @index(Global, NTuple)
    n, m = ni-1, mi-1
    if ni <= T+2 && mi <= T+2
        z = zero(eltype(out))
        FT = typeof(real(z))
        h, dx, dy = z, z, z
        if m <= min(n, T)
            h = n <= T ? H[ni, mi] : z
            dx = complex(zero(FT), FT(m))*h
            upper = n < T ? H[ni+1, mi] : z
            lower = n > m ? H[ni-1, mi] : z
            delta = n == 0 ? zero(FT) : -sqrt(FT(n*n-m*m)/FT(4n*n-1))/FT(n)
            next_delta = -sqrt(FT((n+1)^2-m*m)/FT(4(n+1)^2-1))/FT(n+1)
            dy = -FT(n+2)*FT(n+1)*next_delta*upper+FT(n-1)*FT(n)*delta*lower
        end
        out[ni, mi, 1], out[ni, mi, 2], out[ni, mi, 3] = h, dx, dy
    end
end

face_host_result(raw) = map(Array, raw)

face_host_result(raw, ::Type{FT}) where FT = map(x -> FT.(Array(x)), raw)

# Separate the large (~11.5) constant log-pressure mode before a Float32
# transform. Expanding around its pressure factor keeps the anomaly's bits
# instead of rounding the full logarithm at every FFT butterfly.
surface_pressure_spectra(H,T,s) = (surface_spectra(H,T,s.plan),nothing)
function surface_pressure_spectra(H,T,s::GPUSpectralSynthesis{Float32})
    spectra = surface_spectra(H,T,s.plan)
    backend = get_backend(spectra)
    clear_surface_mean!(backend,1)(spectra;ndrange=1)
    synchronize(backend)
    # Native spectral inputs are host arrays. Only this scalar exponential
    # uses host Float64; every field operation and device scalar is Float32.
    return spectra,Float32(exp(real(H[1,1])))
end
@kernel function clear_surface_mean!(spectra)
    spectra[1,1,1] = complex(zero(real(spectra[1,1,1])),imag(spectra[1,1,1]))
end
@inline surface_pressure_value(logp,::Nothing) = exp(logp)
@inline surface_pressure_value(anomaly,scale) = scale*exp(anomaly)
