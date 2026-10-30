"""
    StepwiseField{FT, N, A, B, W}(samples, boundaries, current_window)

A rank-`N` `AbstractTimeVaryingField{FT, N}` that is **piecewise-constant
in time**. Stores a sequence of `N_win` spatial snapshots plus
`N_win + 1` window boundaries. `field_value` reads from the snapshot for
the window containing the current time; `update_field!(f, t)` binary-
searches the boundaries and caches the index of the current window.

Designed for CATRINE-style emission inventories that arrive as window-
averaged values (monthly, annual, weekly). The sample for window `n` is
the average value over `[boundaries[n], boundaries[n+1])`.

# Layout

```
samples     :: AbstractArray{FT, N + 1}   # rank N + 1
                                          # last dim iterates over windows
                                          # size(samples)[1:N] = spatial shape
                                          # size(samples, N+1) = N_win
boundaries  :: AbstractVector{<:Real}     # length N_win + 1, sorted
                                          # boundaries[n], boundaries[n+1]
                                          # bound window n
current_window :: AbstractVector{Int}     # length 1, holds the index that
                                          # `field_value` reads
```

The spatial prefix of `samples` matches the field's rank: rank-2 field →
`samples::AbstractArray{FT, 3}` with shape `(Nx, Ny, N_win)`. Rank-3 field
→ `samples::AbstractArray{FT, 4}` with `(Nx, Ny, Nz, N_win)`.

# Why `current_window` is a 1-element array

`Base.RefValue{Int}` would be the idiomatic mutable Int cache, but it is
not kernel-safe on GPU — a `Ref` dereference inside a KA kernel pulls from
host memory. Storing the index in a 1-element array lets `Adapt.adapt`
convert it to device storage alongside `samples`, so the kernel reads
`current_window[1]` as a device-local memory access. Overhead: one extra
device load per field_value call, broadcast across all kernel threads
(same value for every thread), effectively free.

# Kernel-safety

After `update_field!(f, t)` has been called by the operator, the kernel
can call `field_value(f, idx)` freely. The call is allocation-free and
type-stable. The operator is responsible for calling `update_field!` on
the host before launching the kernel.

# Out-of-bounds time

`update_field!(f, t)` with `t < boundaries[1]` or
`t >= boundaries[end]` throws an `ArgumentError`. This is stricter than
clamping but avoids silently extrapolating the first/last window — which
would bias emission totals. Callers that need cyclic coverage can wrap
in a `CyclicField` (future plan) or extend `boundaries` explicitly.

# Examples

```julia
# Rank-2 (surface) StepwiseField: monthly CO2 emission inventory
# over 12 months, Nx × Ny = 144 × 72.
samples    = rand(144, 72, 12)                        # kg/s per cell
boundaries = [FT(m * 30 * 86400) for m in 0:12]       # month starts, seconds
f          = StepwiseField(samples, boundaries)

update_field!(f, 15.0 * 86400)   # day 15 → window 1
field_value(f, (10, 20))         # = samples[10, 20, 1]

update_field!(f, 65.0 * 86400)   # day 65 → window 3
field_value(f, (10, 20))         # = samples[10, 20, 3]
```

# Adapt / GPU

```julia
using CUDA, Adapt
f_gpu = Adapt.adapt(CuArray, f)
# f_gpu.samples isa CuArray{FT, 3}
# f_gpu.current_window isa CuArray{Int, 1}
# field_value(f_gpu, (i, j)) now kernel-safe on GPU
```
"""
struct StepwiseField{FT <: AbstractFloat, N,
                     A <: AbstractArray,
                     B <: AbstractVector,
                     W <: AbstractVector{Int}} <: AbstractTimeVaryingField{FT, N}
    samples        :: A
    boundaries     :: B
    current_window :: W

    function StepwiseField{FT, N, A, B, W}(samples::A, boundaries::B,
                                            current_window::W) where {
            FT <: AbstractFloat, N,
            A <: AbstractArray{FT},
            B <: AbstractVector,
            W <: AbstractVector{Int}}
        ndims(samples) == N + 1 ||
            throw(ArgumentError("StepwiseField{$FT, $N}: samples has ndims=$(ndims(samples)), expected $(N + 1)"))
        n_win = size(samples, N + 1)
        length(boundaries) == n_win + 1 ||
            throw(ArgumentError("StepwiseField: boundaries has length $(length(boundaries)), expected $(n_win + 1) (n_win + 1)"))
        issorted(boundaries) ||
            throw(ArgumentError("StepwiseField: boundaries must be sorted"))
        length(current_window) == 1 ||
            throw(ArgumentError("StepwiseField: current_window must be a 1-element vector, got length $(length(current_window))"))
        new{FT, N, A, B, W}(samples, boundaries, current_window)
    end

    # Unchecked path for Adapt reconstruction: the host-side
    # StepwiseField has already validated its inputs, so the adapted
    # copy skips `issorted` (which does not work on device arrays)
    # and length checks (which iterate shape metadata only — safe
    # but redundant).
    function StepwiseField{FT, N, A, B, W}(samples::A, boundaries::B,
                                            current_window::W,
                                            ::Val{:unchecked}) where {
            FT <: AbstractFloat, N,
            A <: AbstractArray{FT},
            B <: AbstractVector,
            W <: AbstractVector{Int}}
        new{FT, N, A, B, W}(samples, boundaries, current_window)
    end
end

"""
    StepwiseField(samples, boundaries)

Outer convenience constructor. Infers `FT` from `eltype(samples)` and
`N` from `ndims(samples) - 1`. Initializes `current_window = [1]`; the
first call to `update_field!(f, t)` sets the real index.
"""
function StepwiseField(samples::AbstractArray{FT}, boundaries::AbstractVector) where FT
    N = ndims(samples) - 1
    current_window = Int[1]
    StepwiseField{FT, N, typeof(samples), typeof(boundaries),
                  typeof(current_window)}(samples, boundaries, current_window)
end

"""
    StepwiseField(samples, boundaries, current_window)

Three-argument form that preserves an explicit `current_window` buffer.
Primarily used by `Adapt.adapt_structure` to carry the cached window index
across a host → device conversion without losing state.
"""
function StepwiseField(samples::AbstractArray{FT}, boundaries::AbstractVector,
                       current_window::AbstractVector{Int}) where FT
    N = ndims(samples) - 1
    StepwiseField{FT, N, typeof(samples), typeof(boundaries),
                  typeof(current_window)}(samples, boundaries, current_window)
end

# ==========================================================================
# TimeVaryingField interface
# ==========================================================================

# `current_window[1]` is the device-local read; splatting `idx` places
# the window index on the rightmost dimension. Marked @inbounds because
# (a) inner constructor validates rank/shape, (b) `update_field!`
# validates the window index.
@inline function field_value(f::StepwiseField{FT, N}, idx::NTuple{N, Int}) where {FT, N}
    @inbounds f.samples[idx..., f.current_window[1]]
end

"""
    update_field!(f::StepwiseField, t::Real) -> f

Binary-search `f.boundaries` for the window `n` such that
`f.boundaries[n] <= t < f.boundaries[n+1]` and cache `n` into
`f.current_window[1]`. Host-side; not kernel-safe itself.

Throws `ArgumentError` if `t` is outside `[f.boundaries[1], f.boundaries[end])`.
"""
function update_field!(f::StepwiseField, t::Real)
    # Adapt keeps boundaries on the active backend so the complete field
    # remains a valid isbits kernel argument. Bring this small one-dimensional
    # coordinate vector to the host for the control-plane binary search; never
    # scalar-index a device vector from the host.
    b = f.boundaries isa Array ? f.boundaries : Array(f.boundaries)
    # searchsortedlast(b, t) returns the largest n with b[n] <= t, or 0
    # if t < b[1]. Clamp/check to valid window range [1, n_win].
    n_win = length(b) - 1
    n = searchsortedlast(b, t)
    (1 <= n <= n_win) ||
        throw(ArgumentError("StepwiseField: time $t is outside the covered range [$(b[1]), $(b[end]))"))
    # Exact equality with the upper boundary of the last window is out of
    # bounds by the half-open convention.
    (n == n_win && t == b[end]) &&
        throw(ArgumentError("StepwiseField: time $t is at the exclusive upper boundary $(b[end])"))
    fill!(f.current_window, n)
    return f
end

# ==========================================================================
# Adapt — GPU dispatch
# ==========================================================================

# After `Adapt.adapt(CuArray, f)`:
#   f.samples         :: CuArray{FT, N+1}
#   f.boundaries      :: CuArray{<:Real, 1}  (if Adapt.adapt(CuArray, Vector{T}))
#   f.current_window  :: CuArray{Int, 1}     (1-element, device-visible)
# `field_value` then does pure device-side indexing.
#
# Uses the unchecked inner-constructor path (Val(:unchecked)) because
# `issorted(boundaries)` on a `CuDeviceVector` is not callable on the
# host, and `f` has already validated its inputs. Preserves the current
# window index across the host → device copy: `Adapt.adapt(to, current_window)`
# converts `Vector{Int}` → `CuArray{Int, 1}` with the stored value.
function Adapt.adapt_structure(to, f::StepwiseField{FT, N}) where {FT, N}
    samples_new        = Adapt.adapt(to, f.samples)
    boundaries_new     = Adapt.adapt(to, f.boundaries)
    current_window_new = Adapt.adapt(to, f.current_window)
    StepwiseField{FT, N, typeof(samples_new), typeof(boundaries_new),
                  typeof(current_window_new)}(samples_new, boundaries_new,
                                              current_window_new, Val(:unchecked))
end
