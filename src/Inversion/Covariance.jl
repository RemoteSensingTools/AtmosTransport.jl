# ---------------------------------------------------------------------------
# Surface-flux background-error covariance B for the CS 4D-Var path.
#
# Preconditioned-4D-Var convention:
#     x   = x_b + B^(1/2) * χ              (physical-space control)
#     B   = B^(1/2) * (B^(1/2))^T          (symmetric PSD)
#     J   = 0.5 ‖χ‖²  +  0.5 (Hx − y)^T R^(−1) (Hx − y)
#     ∇_χ = χ + (B^(1/2))^T H^T R^(−1) (Hx − y)
#
# Factorization used here:
#     B^(1/2) = D · L            (NOT symmetric in general)
#     B       = D · L² · D       (with L² ≡ C, the correlation)
#
# Where:
#   * D is the diagonal `(per-cell σ)` scaling.
#   * L is a symmetric square root of the correlation operator C:
#       - For `DiagonalCSCovariance`, L = I (no correlation), so
#         B = D² and B^(1/2) is self-adjoint.
#       - For `IsotropicGaussianCSCovariance`, C is panel-local periodic
#         Gaussian; L is its symmetric spectral square root, computed
#         once at construction as `sqrt(C̃)` of the FFT eigenvalues of
#         the wrapped Gaussian kernel.
#
# Not covered:
#   * Cross-panel correlation. Each panel is smoothed independently
#     with periodic-on-panel boundary conditions, so correlation wraps
#     around inside a panel instead of crossing panel edges and corners.
#   * Temporal correlation. The operators act on a single
#     CSSurfaceFluxControl `value` shape (`NTuple{6, Matrix{FT}}`);
#     there is no multi-window temporal smoothing.
#   * GPU storage. The FFTW path is CPU-only; there is no CUFFT path.
# ---------------------------------------------------------------------------

import FFTW

"""
    AbstractCSSurfaceFluxCovariance{FT, A}

Supertype for CS surface-flux background-error covariance operators.
Concrete subtypes implement [`apply_B_half!`](@ref) and
[`apply_B_half_adjoint!`](@ref) on a single
[`CSSurfaceFluxControl`](@ref)-shaped tuple `NTuple{6, A}` of `(Nc, Nc)`
matrices.
"""
abstract type AbstractCSSurfaceFluxCovariance{FT <: AbstractFloat,
                                              A <: AbstractMatrix{FT}} end

# ---------------------------------------------------------------------------
# DiagonalCSCovariance — `B = D²`, no correlation.
# ---------------------------------------------------------------------------

"""
    DiagonalCSCovariance(sigma::NTuple{6, AbstractMatrix})

Pure-diagonal CS surface-flux covariance `B = diag(σ)²`. `sigma[p][i, j]`
is the background standard deviation at panel `p`, interior cell
`(i, j)`. All six panels must share the same `(Nc, Nc)` shape and every
entry must be finite and strictly positive.

This is the cleanest baseline; `B^(1/2)` is self-adjoint, so
[`apply_B_half_adjoint!`](@ref) reduces to [`apply_B_half!`](@ref).
"""
struct DiagonalCSCovariance{FT, A} <: AbstractCSSurfaceFluxCovariance{FT, A}
    sigma::NTuple{6, A}
    Nc::Int
end

function DiagonalCSCovariance(sigma::NTuple{6, A}) where {FT <: AbstractFloat,
                                                          A <: AbstractMatrix{FT}}
    sh = size(sigma[1])
    sh[1] == sh[2] || throw(DimensionMismatch(
        "DiagonalCSCovariance requires square panels, got $sh"))
    Nc = sh[1]
    @inbounds for p in 1:6
        size(sigma[p]) == (Nc, Nc) || throw(DimensionMismatch(
            "DiagonalCSCovariance sigma panel $p has shape " *
            "$(size(sigma[p])); expected $((Nc, Nc))"))
        _validate_positive_finite(sigma[p], "DiagonalCSCovariance sigma panel $p")
    end
    return DiagonalCSCovariance{FT, A}(sigma, Nc)
end

# ---------------------------------------------------------------------------
# IsotropicGaussianCSCovariance — `B = D · C · D` with panel-local
# periodic Gaussian correlation `C` of length `L_cells` (in interior
# cells). Built from the spectral square root of the wrapped Gaussian.
# ---------------------------------------------------------------------------

"""
    IsotropicGaussianCSCovariance(sigma::NTuple{6, AbstractMatrix},
                                  correlation_length_cells::Real)

Panel-local CS surface-flux covariance `B = D · C · D` with
isotropic separable Gaussian correlation of length
`correlation_length_cells` (in interior-cell units).

Each panel's `(Nc, Nc)` grid is treated as periodic and the
correlation matrix `C` is the circulant whose first row is the
wrapped Gaussian. `L = C^(1/2)` (the symmetric square root) is
applied via FFT: in spectral space `L` is multiplication by
`sqrt(C̃)` where `C̃` is the real, non-negative spectral eigenvalues
of `C`. The factorization is `B^(1/2) = D · L`, so
`(B^(1/2))^T = L^T · D = L · D` (since `D` and `L` are both
symmetric).

The wrapped-Gaussian normalization pins `C[i, i] = 1`, hence
`B[i, i] = σ_i²` — the diagonal of `B` recovers the per-cell
variance, independent of correlation length.

To keep `B^(-1/2)` finite at operational Float32 resolutions, each
one-dimensional factor of the separable spectrum is floored at its maximum
times `sqrt(eps(FT))`, then renormalized to preserve the unit diagonal. The
corresponding minimum two-dimensional covariance eigenvalue is therefore
approximately its maximum times `eps(FT)`. This is a precision-aware
regularization of unresolved high-frequency modes.

**Limitations**:

- Cross-panel correlation is dropped. Each panel is smoothed in
  isolation; the implicit wrap-around at panel boundaries leaves
  edge artefacts.
- The FFT path is CPU-only.
- The struct carries mutable `fft_buf` / `fft_scratch` scratch
  buffers reused by every `apply_B_half!` / `_adjoint!` /
  `_inverse!` call. As a consequence the covariance is **not
  thread-safe** — concurrent calls on the same instance race on
  the scratch. Build one instance per worker thread if needed.
"""
struct IsotropicGaussianCSCovariance{FT, A, P, IP} <: AbstractCSSurfaceFluxCovariance{FT, A}
    sigma::NTuple{6, A}
    correlation_length_cells::FT
    Nc::Int
    # Spectral square root of the panel-local 2D Gaussian correlation
    # eigenvalues. Real, non-negative, length `Nc × Nc` per panel.
    L_transfer_sqrt::Matrix{FT}
    # Pre-allocated scratch buffers + pre-built FFTW plans reused by
    # every `apply_B_half!` / `_adjoint!` / `_inverse!` call.
    # Avoids per-call matrix allocations (~50 KB at C48) and per-call
    # FFTW-plan construction (plain `fft!` / `ifft!` rebuild a plan on
    # every invocation). Mirrors the `LLPoissonWorkspace` pattern in
    # `src/Preprocessing/mass_support.jl`.
    fft_buf::Matrix{Complex{FT}}
    fft_scratch::Matrix{FT}
    fft_plan::P
    ifft_plan::IP
end

function IsotropicGaussianCSCovariance(sigma::NTuple{6, A},
                                       correlation_length_cells::Real
                                       ) where {FT <: AbstractFloat,
                                                A <: AbstractMatrix{FT}}
    isfinite(correlation_length_cells) || throw(ArgumentError(
        "IsotropicGaussianCSCovariance correlation_length_cells must be " *
        "finite, got $correlation_length_cells"))
    correlation_length_cells > 0 || throw(ArgumentError(
        "IsotropicGaussianCSCovariance correlation_length_cells must be " *
        "positive, got $correlation_length_cells"))
    sh = size(sigma[1])
    sh[1] == sh[2] || throw(DimensionMismatch(
        "IsotropicGaussianCSCovariance requires square panels, got $sh"))
    Nc = sh[1]
    @inbounds for p in 1:6
        size(sigma[p]) == (Nc, Nc) || throw(DimensionMismatch(
            "IsotropicGaussianCSCovariance sigma panel $p has shape " *
            "$(size(sigma[p])); expected $((Nc, Nc))"))
        _validate_positive_finite(sigma[p],
            "IsotropicGaussianCSCovariance sigma panel $p")
    end
    L_cells = FT(correlation_length_cells)
    transfer_sqrt = _gaussian_transfer_sqrt_2d(Nc, L_cells)
    fft_buf = Matrix{Complex{FT}}(undef, Nc, Nc)
    fft_scratch = Matrix{FT}(undef, Nc, Nc)
    fft_plan = FFTW.plan_fft!(fft_buf)
    ifft_plan = FFTW.plan_ifft!(fft_buf)
    return IsotropicGaussianCSCovariance{FT, A,
                                          typeof(fft_plan),
                                          typeof(ifft_plan)}(
        sigma, L_cells, Nc, transfer_sqrt,
        fft_buf, fft_scratch, fft_plan, ifft_plan)
end

# ---------------------------------------------------------------------------
# Spectral helpers — wrapped 1D periodic Gaussian + 2D separable
# spectral square root.
# ---------------------------------------------------------------------------

# Periodic 1D Gaussian kernel `k[n]` for `n ∈ 0..Nc-1`, summed over
# enough periodic images that the tails are below Float64 precision.
# Returns the unnormalized kernel; the caller normalizes diagonal-of-C
# to 1 by dividing by `k[1]`.
function _gaussian_periodic_kernel(Nc::Int, L::FT) where FT
    inv2L2 = inv(2 * L * L)
    # Number of wraps. For `L < Nc/2` two images each side already give
    # < exp(-Nc²/2L²) ≪ 1e-15. Safety floor of 3 in case L ≳ Nc.
    M = max(3, ceil(Int, 4 * L / Nc) + 1)
    kernel = zeros(FT, Nc)
    @inbounds for n in 0:(Nc - 1)
        s = zero(FT)
        for m in (-M):M
            d = n + m * Nc
            s += exp(-FT(d) * FT(d) * inv2L2)
        end
        kernel[n + 1] = s
    end
    return kernel
end

# Spectral square root of the panel-local 2D isotropic Gaussian
# correlation. Returns `sqrt(C̃[kx, ky])` where `C̃[kx, ky] =
# C̃_1d[kx] · C̃_1d[ky]` is the separable spectral density of the
# wrapped Gaussian, normalized so that `C[i, i] = 1` (kernel diagonal
# matches a unit correlation).
function _gaussian_transfer_sqrt_2d(Nc::Int, L::FT) where FT
    k1d = _gaussian_periodic_kernel(Nc, L)
    # Normalize so `C[0, 0] = k[1] = 1` after division. `k[1]` is the
    # diagonal value (n = 0) and is strictly positive.
    k1d ./= k1d[1]
    # FFT eigenvalues of a circulant matrix are the FFT of its first
    # row. The kernel is real and symmetric (`k[Nc - n] = k[n]` modulo
    # periodicity), so the spectrum is real. Finite precision can make
    # the smallest eigenvalues negative or exactly zero, especially for
    # Float32 at operational panel sizes. A zero makes B^(-1/2)
    # undefined, so regularize the one-dimensional spectrum to a
    # precision-scaled positive floor. This bounds the condition number
    # of the two-dimensional square-root transfer at O(inv(sqrt(eps))).
    spec_complex = FFTW.fft(k1d)
    spec_1d = real.(spec_complex)
    max_spec = maximum(spec_1d)
    isfinite(max_spec) && max_spec > zero(FT) || throw(ArgumentError(
        "Gaussian covariance produced an invalid spectral maximum $max_spec"))
    spectral_floor = max_spec * sqrt(eps(FT))
    @inbounds for k in eachindex(spec_1d)
        spec_1d[k] = max(spec_1d[k], spectral_floor)
    end
    # Flooring slightly changes the zero-lag variance. Renormalize the
    # spectrum so the corresponding circulant kernel still has unit
    # diagonal and B[i,i] remains sigma[i]^2.
    spec_1d .*= FT(Nc) / sum(spec_1d)
    sqrt_spec = sqrt.(spec_1d)
    # Outer product gives the 2D separable spectral square root.
    out = Matrix{FT}(undef, Nc, Nc)
    @inbounds for ky in 1:Nc, kx in 1:Nc
        out[kx, ky] = sqrt_spec[kx] * sqrt_spec[ky]
    end
    return out
end

# ---------------------------------------------------------------------------
# Forward + adjoint application
# ---------------------------------------------------------------------------

"""
    apply_B_half!(y::NTuple{6, AbstractMatrix},
                  cov::AbstractCSSurfaceFluxCovariance,
                  chi::NTuple{6, AbstractMatrix}) -> y

Apply the (non-symmetric) square root `B^(1/2)` to `chi` and write the
result into `y`. Concretely, `y = D · L · chi`, where `D` is the
diagonal scaling and `L` is the symmetric correlation square root.
`y` and `chi` may alias; the implementation writes through a scratch
buffer when needed.
"""
function apply_B_half! end

"""
    apply_B_half_adjoint!(g_chi::NTuple{6, AbstractMatrix},
                          cov::AbstractCSSurfaceFluxCovariance,
                          g_phys::NTuple{6, AbstractMatrix}) -> g_chi

Apply the adjoint `(B^(1/2))^T` to `g_phys`. With `B^(1/2) = D · L` and
`D`, `L` both symmetric, this is `g_chi = L · D · g_phys`.
"""
function apply_B_half_adjoint! end

# ---- DiagonalCSCovariance ----------------------------------------------------

function apply_B_half!(y::NTuple{6, A},
                      cov::DiagonalCSCovariance{FT, A},
                      chi::NTuple{6, A}) where {FT, A}
    _validate_panel_shapes(y, cov.Nc, "y")
    _validate_panel_shapes(chi, cov.Nc, "chi")
    @inbounds for p in 1:6
        @. y[p] = cov.sigma[p] * chi[p]
    end
    return y
end

# `B^(1/2) = D` is self-adjoint for the diagonal case.
function apply_B_half_adjoint!(g_chi::NTuple{6, A},
                               cov::DiagonalCSCovariance{FT, A},
                               g_phys::NTuple{6, A}) where {FT, A}
    return apply_B_half!(g_chi, cov, g_phys)
end

# ---- IsotropicGaussianCSCovariance ------------------------------------------

# Apply the symmetric correlation square root `L` to a single panel
# in-place via 2D FFT. Real input, real output (within float noise).
# Uses pre-built `fft_plan` / `ifft_plan` so the per-call cost is the
# actual FFT work, not plan construction.
function _apply_L_panel!(buf::Matrix{Complex{FT}}, input::AbstractMatrix{FT},
                        output::AbstractMatrix{FT},
                        transfer_sqrt::Matrix{FT},
                        fft_plan, ifft_plan) where FT
    Nc = size(transfer_sqrt, 1)
    @inbounds for j in 1:Nc, i in 1:Nc
        buf[i, j] = Complex{FT}(input[i, j])
    end
    fft_plan * buf
    @inbounds for j in 1:Nc, i in 1:Nc
        buf[i, j] *= transfer_sqrt[i, j]
    end
    ifft_plan * buf
    @inbounds for j in 1:Nc, i in 1:Nc
        output[i, j] = real(buf[i, j])
    end
    return output
end

function apply_B_half!(y::NTuple{6, A},
                      cov::IsotropicGaussianCSCovariance{FT, A},
                      chi::NTuple{6, A}) where {FT, A}
    _validate_panel_shapes(y, cov.Nc, "y")
    _validate_panel_shapes(chi, cov.Nc, "chi")
    @inbounds for p in 1:6
        # B^(1/2) = D · L. Apply L first (smooth chi), then multiply
        # by D (per-cell sigma). `cov.fft_scratch` holds `L · chi[p]`.
        _apply_L_panel!(cov.fft_buf, chi[p], cov.fft_scratch,
                        cov.L_transfer_sqrt, cov.fft_plan, cov.ifft_plan)
        @. y[p] = cov.sigma[p] * cov.fft_scratch
    end
    return y
end

function apply_B_half_adjoint!(g_chi::NTuple{6, A},
                               cov::IsotropicGaussianCSCovariance{FT, A},
                               g_phys::NTuple{6, A}) where {FT, A}
    _validate_panel_shapes(g_chi, cov.Nc, "g_chi")
    _validate_panel_shapes(g_phys, cov.Nc, "g_phys")
    @inbounds for p in 1:6
        # (B^(1/2))^T = L · D. Apply D first (per-cell sigma), then L.
        @. cov.fft_scratch = cov.sigma[p] * g_phys[p]
        _apply_L_panel!(cov.fft_buf, cov.fft_scratch, g_chi[p],
                        cov.L_transfer_sqrt, cov.fft_plan, cov.ifft_plan)
    end
    return g_chi
end

# ---------------------------------------------------------------------------
# Inverse — `B^(-1/2)`. Needed by the preconditioner to map a
# physical-space initial guess back into χ-space and for the
# LogNormal bijection. `apply_B_half_inverse!` is `D^(-1) · L^(-1)`:
# elementwise divide by σ, then deconvolve via spectral `1/sqrt(C̃)`.
#
# For `DiagonalCSCovariance` the inverse is exact and self-adjoint.
# For `IsotropicGaussianCSCovariance`, the symmetric `L` is its own
# inverse in spectral space (multiplication by `1/sqrt(C̃)`); the
# composition `L · L^(-1) = I` is exact up to FFT roundoff, but
# applying `L^(-1)` to a noisy input amplifies any signal that lies
# in the high-frequency tail where `transfer_sqrt` is tiny. Callers
# are responsible for keeping input clean.
# ---------------------------------------------------------------------------

"""
    apply_B_half_inverse!(y::NTuple{6, AbstractMatrix},
                          cov::AbstractCSSurfaceFluxCovariance,
                          x::NTuple{6, AbstractMatrix}) -> y

Apply `B^(-1/2)` to `x` and write the result into `y`. With the
factorization `B^(1/2) = D · L` (`D` diagonal, `L` symmetric),
`B^(-1/2) = L^(-1) · D^(-1)`. Used by the preconditioner to invert
the change of variables `x = x_b + B^(1/2) χ` (Linear) or
`x = x_b ⊙ exp(B^(1/2) χ)` (LogNormal).

The Gaussian inverse is regularized in the high-frequency tail by the
precision-scaled spectral floor documented on
[`IsotropicGaussianCSCovariance`](@ref). It is finite for arbitrary finite
inputs, but poorly resolved high-frequency modes remain sensitive to
roundoff.
"""
function apply_B_half_inverse! end

function apply_B_half_inverse!(y::NTuple{6, A},
                                cov::DiagonalCSCovariance{FT, A},
                                x::NTuple{6, A}) where {FT, A}
    _validate_panel_shapes(y, cov.Nc, "y")
    _validate_panel_shapes(x, cov.Nc, "x")
    @inbounds for p in 1:6
        @. y[p] = x[p] / cov.sigma[p]
    end
    return y
end

# Apply `L^(-1)` to a single panel in-place via 2D FFT. Real input,
# real output. Mirror of `_apply_L_panel!` but divides by
# `transfer_sqrt` instead of multiplying.
function _apply_L_inverse_panel!(buf::Matrix{Complex{FT}},
                                 input::AbstractMatrix{FT},
                                 output::AbstractMatrix{FT},
                                 transfer_sqrt::Matrix{FT},
                                 fft_plan, ifft_plan) where FT
    Nc = size(transfer_sqrt, 1)
    @inbounds for j in 1:Nc, i in 1:Nc
        buf[i, j] = Complex{FT}(input[i, j])
    end
    fft_plan * buf
    @inbounds for j in 1:Nc, i in 1:Nc
        # Construction applies a precision-scaled positive spectral
        # floor, so division is finite while still damping unsupported
        # high-frequency inverse modes.
        buf[i, j] /= transfer_sqrt[i, j]
    end
    ifft_plan * buf
    @inbounds for j in 1:Nc, i in 1:Nc
        output[i, j] = real(buf[i, j])
    end
    return output
end

function apply_B_half_inverse!(y::NTuple{6, A},
                                cov::IsotropicGaussianCSCovariance{FT, A},
                                x::NTuple{6, A}) where {FT, A}
    _validate_panel_shapes(y, cov.Nc, "y")
    _validate_panel_shapes(x, cov.Nc, "x")
    @inbounds for p in 1:6
        # B^(-1/2) = L^(-1) · D^(-1). Divide by σ first, then
        # deconvolve via `1/sqrt(C̃)` in spectral space.
        @. cov.fft_scratch = x[p] / cov.sigma[p]
        _apply_L_inverse_panel!(cov.fft_buf, cov.fft_scratch, y[p],
                                cov.L_transfer_sqrt,
                                cov.fft_plan, cov.ifft_plan)
    end
    return y
end

# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

function _validate_positive_finite(arr::AbstractMatrix, name::AbstractString)
    @inbounds for j in axes(arr, 2), i in axes(arr, 1)
        v = arr[i, j]
        isfinite(v) || throw(ArgumentError(
            "$name has non-finite entry at ($i, $j): $v"))
        v > 0 || throw(ArgumentError(
            "$name has non-positive entry at ($i, $j): $v"))
    end
    return nothing
end

function _validate_panel_shapes(panels::NTuple{6, <:AbstractMatrix},
                                Nc::Int, name::AbstractString)
    @inbounds for p in 1:6
        size(panels[p]) == (Nc, Nc) || throw(DimensionMismatch(
            "$name panel $p has shape $(size(panels[p])); expected $((Nc, Nc))"))
    end
    return nothing
end
