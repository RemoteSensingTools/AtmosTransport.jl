# ============================================================================
# Batched spectral → reduced-Gaussian synthesis.
#
# A field given by spherical-harmonic coefficients f̂_n^m (triangular truncation
# T, ECMWF's fully normalised P̃_n^m) is, on ring j of a reduced Gaussian grid,
#
#     f(λ_i, φ_j) = Re Σ_{m=0}^{M_j} w_m G_m(φ_j) e^{i m λ_i},
#     G_m(φ_j)    = Σ_{n=m}^{T} f̂_n^m P̃_n^m(sin φ_j),
#
# with M_j = min(T, nlon_j ÷ 2), w_0 = 1, w_m = 2 for 0 < m < nlon_j / 2, and
# cell centres λ_i = (i − ½) Δλ_j. Wavenumbers above nlon_j ÷ 2 are dropped on
# ring j. The Nyquist term m = nlon_j / 2 of an even ring enters with w = 1,
# as in the per-ring path (`spectral_to_ring!`); ERA5's ring lengths keep its
# amplitude small. The
# Legendre sums G_m are the expensive part. For one m, all levels k and all
# rings j they form one matrix product, G_m[k, j] = Σ_n F_m[k, n] P_m[n, j],
# with P_m[n − m + 1, j] = P̃_n^m(sin φ_j); the longitude sums are one inverse
# real FFT per ring and level.
#
# Evaluating the Legendre table once per process and running the sums as BLAS
# products over all levels replaces the per-ring, per-level recurrence and the
# scalar sums of `spectral_to_ring!`: 2.6 s instead of 39–45 s per ERA5 N320
# window (T639, 137 levels, u, v, T; 12 threads, 2026-10-08). Results agree with
# `spectral_to_ring!` to round-off.
# ============================================================================

"""
    ReducedSpectralSynthesis(grid, T, Nf)

Precomputed Legendre tables, FFT plans and buffers for synthesising `Nf`
columns (model levels) of a spectral field with truncation `T` on the cell
centres of the reduced Gaussian `grid`. Calls to [`synthesize_reduced!`](@ref)
reuse the buffers, so one object serves any number of fields, but not two
calls at once. The Legendre table holds `n_rings × (T + 1)(T + 2)/2` values
(1.1 GB for N320 at T639) and the half-spectrum buffer `Nf × Σ_j (nlon_j ÷ 2 + 1)`
complex values (0.6 GB for 137 levels).
"""
struct ReducedSpectralSynthesis{G <: ReducedGaussianTargetGeometry, P}
    grid          :: G
    T             :: Int
    Nf            :: Int
    legendre      :: Vector{Matrix{Float64}}   # [m + 1][n − m + 1, j] = P̃_n^m(sin φ_j)
    half_offsets  :: Vector{Int}               # first column of ring j in `half`
    coeffs        :: Matrix{Float64}           # (2Nf, T + 1): Re, Im of f̂_n^m for one m
    sums          :: Matrix{Float64}           # (2Nf, n_rings): Re, Im of G_m on every ring
    half          :: Matrix{ComplexF64}        # (Nf, Σ_j nlon_j ÷ 2 + 1): half spectra of all rings
    plans         :: Dict{Int, P}              # inverse real FFT per ring length
    fft_in        :: Vector{Vector{ComplexF64}}  # per thread
    fft_out       :: Vector{Vector{Float64}}     # per thread
end

function ReducedSpectralSynthesis(grid::ReducedGaussianTargetGeometry, T::Integer, Nf::Integer)
    T >= 1 && Nf >= 1 || throw(ArgumentError("need T ≥ 1 and Nf ≥ 1, got T = $T, Nf = $Nf"))
    mesh = grid.mesh
    nr = nrings(mesh)
    nlon = mesh.nlon_per_ring
    legendre = [zeros(T - m + 1, nr) for m in 0:T]
    columns = [zeros(T + 1, T + 1) for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for j in 1:nr
        P = columns[Threads.threadid()]
        compute_legendre_column!(P, T, sind(Float64(grid.lats[j])))
        for m in 0:T, n in m:T
            legendre[m + 1][n - m + 1, j] = P[n + 1, m + 1]
        end
    end
    half_offsets = cumsum([1; [n ÷ 2 + 1 for n in nlon[1:end-1]]])
    maxlen = maximum(nlon)
    fft_in = [zeros(ComplexF64, maxlen ÷ 2 + 1) for _ in 1:Threads.maxthreadid()]
    fft_out = [zeros(maxlen) for _ in 1:Threads.maxthreadid()]
    plans = Dict(n => plan_brfft(view(fft_in[1], 1:(n ÷ 2 + 1)), n; flags = FFTW.ESTIMATE)
                 for n in unique(nlon))
    return ReducedSpectralSynthesis(grid, Int(T), Int(Nf), legendre, half_offsets,
                                    zeros(2Nf, T + 1), zeros(2Nf, nr),
                                    zeros(ComplexF64, Nf, sum(n ÷ 2 + 1 for n in nlon)),
                                    plans, fft_in, fft_out)
end

"""
    synthesize_reduced!(out, spec, s::ReducedSpectralSynthesis) -> out

Synthesise the spectral coefficients `spec[n + 1, m + 1, k]` of `s.Nf` levels
`k` onto the reduced Gaussian cell centres: `out[c, k]` for cell `c`.
"""
function synthesize_reduced!(out::AbstractMatrix{<:Real}, spec::AbstractArray{ComplexF64, 3},
                             s::ReducedSpectralSynthesis)
    (; T, Nf, legendre, half_offsets, coeffs, sums, half) = s
    mesh = s.grid.mesh
    nr = nrings(mesh)
    size(spec) == (T + 1, T + 1, Nf) ||
        throw(DimensionMismatch("spectral coefficients $(size(spec)) ≠ ($(T + 1), $(T + 1), $Nf)"))
    size(out) == (ncells(mesh), Nf) ||
        throw(DimensionMismatch("output $(size(out)) ≠ ($(ncells(mesh)), $Nf)"))

    # Legendre sums G_m(φ_j) for all levels and rings; m ≤ nlon_j ÷ 2 enters
    # ring j's half spectrum, shifted to the cell centres by e^{i m Δλ_j / 2}.
    # Entries with m > T are never written and stay zero from the allocation.
    for m in 0:T
        F = view(coeffs, :, 1:(T - m + 1))
        @inbounds for k in 1:Nf, n in m:T
            c = spec[n + 1, m + 1, k]
            F[k, n - m + 1] = real(c)
            F[Nf + k, n - m + 1] = imag(c)
        end
        mul!(sums, F, legendre[m + 1])
        @inbounds for j in 1:nr
            nlon = mesh.nlon_per_ring[j]
            m <= nlon ÷ 2 || continue
            shift = m == 0 ? one(ComplexF64) : exp(im * m * (pi / nlon))
            col = half_offsets[j] + m
            for k in 1:Nf
                half[k, col] = complex(sums[k, j], sums[Nf + k, j]) * shift
            end
        end
    end

    # Longitude sums: one inverse real FFT per ring and level. `brfft` is the
    # unnormalised inverse: it adds each 0 < m < nlon/2 term twice (m and its
    # mirror −m) and the Nyquist term once.
    Threads.@threads :static for j in 1:nr
        tid = Threads.threadid()
        nlon = mesh.nlon_per_ring[j]
        nh = nlon ÷ 2 + 1
        hin = view(s.fft_in[tid], 1:nh)
        rout = view(s.fft_out[tid], 1:nlon)
        plan = s.plans[nlon]
        c0 = half_offsets[j] - 1
        cells = mesh.ring_offsets[j]:(mesh.ring_offsets[j + 1] - 1)
        @inbounds for k in 1:Nf
            for i in 1:nh
                hin[i] = half[k, c0 + i]
            end
            mul!(rout, plan, hin)
            copyto!(view(out, cells, k), rout)
        end
    end
    return out
end
