# Shared with the §13 raw-face diagnostic. Geometry and Fourier multipliers
# are prepared on the host; transforms and gathers use the selected backend.
"""Gauss–Legendre nodes and weights on [-1,1] (Golub–Welsch)."""
function gauss_legendre(n::Integer)
    e = eigen(SymTridiagonal(zeros(n), [i/sqrt(4i*i-1) for i in 1:n-1]))
    return e.values, 2e.vectors[1, :].^2
end

"""Integral of exp(im*m*λ) over [0,width], including the zero mode."""
fourier_interval_factor(m::Integer, width::Real) = width*sinc(m*width/(2pi))*cis(m*width/2)

function gauss_latitude_quadrature(south, north, T, rule::GaussFourier)
    base = max(10, ceil(Int, (T+2)*(north-south))+3)
    n = cld(rule.order*base, 8)
    x, w = gauss_legendre(n)
    nodes = @. (south+north)/2+(north-south)/2*x
    weights = w.*((north-south)/2)
    return nodes, weights, weights.*cos.(nodes)
end

# FFT evaluation after folding frequencies modulo the number of intervals
# handles arbitrary RG segments without an LCM-sized synthesis grid. Unlike
# interpolation of a primitive, this is exact for the sampled Fourier series.
@kernel function pack_fourier_rings!(samples, values, rows, offsets, J, nf)
    i, k, r = @index(Global, NTuple)
    if i <= J && k <= nf && r <= length(rows)
        samples[i, k, r] = values[offsets[rows[r]]+i-1, k]
    end
end

@kernel function fold_fourier_integrals!(out, spectrum, factors, J, n, nf)
    qi, k, r = @index(Global, NTuple)
    if qi <= size(out, 1) && k <= nf && r <= size(out, 3)
        q = qi-1
        total = zero(eltype(out))
        half = J÷2
        firstmode = -half+mod(q+half, n)
        for m in firstmode:n:half
            z = m >= 0 ? spectrum[m+1, k, r] : conj(spectrum[1-m, k, r])
            total += z*factors[m+half+1]
        end
        out[qi, k, r] = total
    end
end

@kernel function unpack_fourier_integrals!(out, samples, rows, offsets, n, nf)
    i, k, r = @index(Global, NTuple)
    if i <= n && k <= nf && r <= length(rows)
        out[offsets[rows[r]]+i-1, k] = samples[i, k, r]
    end
end

fourier_scratch_view(buffer, dims...) = reshape(view(vec(buffer), 1:prod(dims)), dims...)

"""Reusable FFT plans for exact box integrals of sampled periodic products."""
function fourier_integral_plan(rings, counts, synthesis, batch;
                               selected_rows=collect(eachindex(counts)), scratch=Dict())
    p = synthesis.plan
    FT = eltype(p.coeffs)
    adapt = p.adapter
    active = falses(length(counts))
    active[selected_rows] .= true
    keys = sort(unique(collect(zip(rings.nlon[selected_rows], counts[selected_rows]))))
    groups = map(keys) do (J, n)
        rows = findall(i -> active[i] && rings.nlon[i] == J && counts[i] == n, eachindex(counts))
        buffers = get!(scratch, (J, n)) do
            (; samples=adapt(zeros(FT, J, batch, length(rows))),
               spectrum=adapt(zeros(Complex{FT}, J÷2+1, batch, length(rows))),
               half=adapt(zeros(Complex{FT}, n÷2+1, batch, length(rows))),
               integrals=adapt(zeros(FT, n, batch, length(rows))))
        end
        # Sequential surface/forcing/boundary transforms can use the same
        # storage. Each plan retains only its own shape and row mapping.
        size(buffers.samples, 2) >= batch && size(buffers.samples, 3) >= length(rows) ||
            throw(DimensionMismatch("Fourier scratch capacity"))
        samples = fourier_scratch_view(buffers.samples, J, batch, length(rows))
        forward = p.fft_threads == 0 ? plan_rfft(samples, 1) : plan_rfft(samples, 1; num_threads=p.fft_threads)
        spectrum = fourier_scratch_view(buffers.spectrum, J÷2+1, batch, length(rows))
        half = fourier_scratch_view(buffers.half, n÷2+1, batch, length(rows))
        inverse = p.fft_threads == 0 ? plan_brfft(half, n, 1) : plan_brfft(half, n, 1; num_threads=p.fft_threads)
        integrals = fourier_scratch_view(buffers.integrals, n, batch, length(rows))
        factors = [fourier_interval_factor(m, 2pi/n)/J for m in -(J÷2):J÷2]
        # Split the real Nyquist cosine into its positive and negative modes.
        if iseven(J)
            factors[1] /= 2
            factors[end] /= 2
        end
        (; J, n, rows=adapt(rows), samples, forward, spectrum, half, inverse,
           integrals, factors=adapt(Complex{FT}.(factors)))
    end
    return (; groups, offsets=adapt(cumsum([1; counts])), source_offsets=p.offsets,
              size=sum(counts), prototype=p.coeffs)
end

function fourier_integrals!(values, plan, nf)
    backend = get_backend(values)
    out = similar(plan.prototype, eltype(values), plan.size, nf)
    for g in plan.groups
        pack_fourier_rings!(backend, 256)(g.samples, values, g.rows, plan.source_offsets, g.J, nf;
            ndrange=(g.J, nf, length(g.rows)))
        synchronize(backend)
        mul!(g.spectrum, g.forward, g.samples)
        fold_fourier_integrals!(backend, 256)(g.half, g.spectrum, g.factors, g.J, g.n, nf;
            ndrange=size(g.half))
        synchronize(backend)
        mul!(g.integrals, g.inverse, g.half)
        unpack_fourier_integrals!(backend, 256)(out, g.integrals, g.rows, plan.offsets, g.n, nf;
            ndrange=size(g.integrals))
    end
    synchronize(backend)
    return out
end
