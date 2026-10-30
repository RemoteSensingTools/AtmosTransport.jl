"""
    PreComputedKzField{FT, N, A}(data::AbstractArray{FT, N})

A rank-2/3 `AbstractTimeVaryingField{FT, N}` backed directly by a
spatial array. `field_value(f, idx)` returns `f.data[idx...]`
— the no-abstraction path for Kz profiles read from a pre-computed
source (met binary snapshot, GCHP-style 3D diffusivity export, etc.).

Time variation, if any, is handled by the caller: mutate `f.data`
in place when the current met window advances, then call (or skip)
`update_field!` — this method is a no-op and does not own the
storage. A more structured stepwise-in-time backing (e.g. a 4D
buffer with an internal window index) can be added later as a
separate concrete type; keeping this one minimal makes the contract
transparent.

# Backend

`A <: AbstractArray{FT, N}` is parametric: `Array{FT, N}` for CPU,
`CuArray{FT, N}` / `MtlArray{FT, N}` for GPU runs. `field_value`
indexes the array element-wise, which is kernel-safe on all three
backends.

# Examples
```julia
# CPU: static 3D Kz snapshot
Kz = fill(1.0, 144, 72, 34)
f  = PreComputedKzField(Kz)

field_value(f, (1, 1, 1))      # 1.0
field_value(f, (144, 72, 34))  # 1.0

# Updating the snapshot between met windows (caller-owned)
Kz .= 2.0
field_value(f, (1, 1, 1))      # 2.0  — same f, updated data
```
"""
struct PreComputedKzField{FT <: AbstractFloat,
                          N,
                          A <: AbstractArray} <: AbstractTimeVaryingField{FT, N}
    data :: A

    function PreComputedKzField{FT, N, A}(data::A) where {FT, N, A <: AbstractArray{FT, N}}
        (N == 2 || N == 3) || throw(ArgumentError(
            "PreComputedKzField: data rank must be 2 or 3, got $N"))
        new{FT, N, A}(data)
    end
end

PreComputedKzField(data::A) where {FT, A <: AbstractArray{FT, 2}} =
    PreComputedKzField{FT, 2, A}(data)

PreComputedKzField(data::A) where {FT, A <: AbstractArray{FT, 3}} =
    PreComputedKzField{FT, 3, A}(data)

@inline field_value(f::PreComputedKzField{FT, N}, idx::NTuple{N, Int}) where {FT, N} =
    @inbounds f.data[idx...]

update_field!(f::PreComputedKzField, ::Real) = f

# Adapt hook: lets KA kernels receive a device-shaped `PreComputedKzField`.
# When `data::CuArray{...}` (non-bitstype, holds a host-side pointer
# wrapper), `Adapt.adapt(to, f.data)` becomes a `CuDeviceArray` usable
# from inside the kernel.
Adapt.adapt_structure(to, f::PreComputedKzField) =
    PreComputedKzField(Adapt.adapt(to, f.data))
