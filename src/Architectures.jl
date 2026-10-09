"""
    Architectures

Execution architectures and their optional runtime integration.

An architecture is part of a grid's type-level model contract: `CPU()` selects
host execution, while `GPU(:cuda)` and `GPU(:metal)` select a concrete GPU
runtime. The same object controls array adaptation, device synchronization, and
runtime validation, so grid metadata cannot diverge from model storage.
"""
module Architectures

using DocStringExtensions
using KernelAbstractions: KernelAbstractions as KA

export AbstractArchitecture, CPU, GPU
export array_type, device, architecture, architecture_from_config
export autodetect_gpu_architecture, is_gpu, ensure_runtime!, array_adapter
export architecture_label, device_name, backend_name, synchronize_architecture!
export array_adapter_for, assert_residency!, assert_float_type!
export reclaim_backend_pool!, _kahan_add

abstract type AbstractArchitecture end

"""
$(TYPEDEF)

Host CPU execution architecture.
"""
struct CPU <: AbstractArchitecture end

"""
$(TYPEDEF)

GPU execution architecture for backend `B`.

Construct with `GPU(:cuda)` or `GPU(:metal)`. Making the backend explicit keeps
CUDA and Metal methods unambiguous when both optional packages are loaded.
"""
struct GPU{B} <: AbstractArchitecture end

function GPU(backend::Symbol)
    backend in (:cuda, :metal) || throw(ArgumentError(
        "unsupported GPU backend $(repr(backend)); expected :cuda or :metal"))
    return GPU{backend}()
end

GPU(backend::AbstractString) = GPU(_architecture_symbol(backend))

array_type(::CPU) = Array
device(::CPU) = KA.CPU()

# GPU array and KernelAbstractions device methods live in the CUDA and Metal
# package extensions. Without the corresponding optional package, calling
# `array_type` or `device` for that architecture intentionally has no method.

function architecture end

is_gpu(::CPU) = false
is_gpu(::GPU) = true

backend_name(::CPU) = :cpu
backend_name(::GPU{B}) where {B} = B

const _RUNTIME_PACKAGE_IDS = (
    CUDA = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"),
    Metal = Base.PkgId(Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"),
)

function _load_runtime_package!(name::Symbol)
    hasproperty(_RUNTIME_PACKAGE_IDS, name) ||
        throw(ArgumentError("unsupported runtime package $(name)"))
    pkgid = getproperty(_RUNTIME_PACKAGE_IDS, name)
    try
        return Base.require(pkgid)
    catch err
        throw(ArgumentError(
            "$(name) backend requested, but $(name).jl could not be loaded from " *
            "the active environment: $(sprint(showerror, err))"))
    end
end

function _architecture_symbol(raw)
    name = replace(lowercase(String(raw)), '-' => '_', ' ' => '_')
    name in ("cpu", "host") && return :cpu
    name in ("cuda", "nvidia") && return :cuda
    name in ("metal", "apple", "apple_metal") && return :metal
    name in ("auto", "gpu") && return :auto
    throw(ArgumentError(
        "unknown architecture.backend = \"$(raw)\"; supported values are " *
        "\"cpu\", \"cuda\", \"metal\", and \"auto\"."))
end

_architecture(::Val{:cpu}) = CPU()
_architecture(::Val{:cuda}) = GPU(:cuda)
_architecture(::Val{:metal}) = GPU(:metal)

"""
    architecture_from_config(config) -> AbstractArchitecture

Resolve an `[architecture]` configuration table to one concrete execution
architecture. An omitted backend selects `CPU()` unless `use_gpu = true`, in
which case a usable GPU runtime is detected.
"""
function architecture_from_config(config)
    use_gpu = get(config, "use_gpu", false)
    use_gpu isa Bool || throw(ArgumentError(
        "[architecture].use_gpu must be true or false; got $(repr(use_gpu))"))
    raw_backend = get(config, "backend", nothing)

    raw_backend === nothing && return use_gpu ? autodetect_gpu_architecture() : CPU()

    backend = _architecture_symbol(raw_backend)
    backend === :cpu && use_gpu && throw(ArgumentError(
        "[architecture] use_gpu = true conflicts with backend = \"cpu\""))
    backend === :auto && return autodetect_gpu_architecture()
    return _architecture(Val(backend))
end

function _try_architecture!(arch::GPU)
    try
        ensure_runtime!(arch)
        return true, nothing
    catch err
        return false, err
    end
end

"""
    autodetect_gpu_architecture() -> GPU

Return the first functional supported GPU architecture on this host.
"""
function autodetect_gpu_architecture()
    candidates = Sys.isapple() ?
        (GPU(:metal), GPU(:cuda)) :
        (GPU(:cuda), GPU(:metal))

    failures = String[]
    for arch in candidates
        arch isa GPU{:metal} && !Sys.isapple() && continue
        ok, err = _try_architecture!(arch)
        ok && return arch
        push!(failures, "$(backend_name(arch)): $(sprint(showerror, err))")
    end

    detail = isempty(failures) ? "No candidate backend was attempted." :
             "Tried " * join(failures, "; ")
    throw(ArgumentError(
        "[architecture] requested GPU backend auto-detection, but no supported " *
        "GPU backend is usable on this host. $(detail)"))
end

ensure_runtime!(::CPU) = true

function ensure_runtime!(::GPU{:cuda})
    CUDA = _load_runtime_package!(:CUDA)
    Base.invokelatest(getproperty(CUDA, :functional)) ||
        throw(ArgumentError("CUDA runtime is not functional on this host"))
    isdefined(CUDA, :allowscalar) &&
        Base.invokelatest(getproperty(CUDA, :allowscalar), false)
    return true
end

function ensure_runtime!(::GPU{:metal})
    Sys.isapple() ||
        throw(ArgumentError("Metal backend requires macOS on Apple Silicon"))
    Metal = _load_runtime_package!(:Metal)
    if isdefined(Metal, :functional)
        Base.invokelatest(getproperty(Metal, :functional)) ||
            throw(ArgumentError("Metal runtime is not functional on this host"))
    end
    isdefined(Metal, :device) && Base.invokelatest(getproperty(Metal, :device))
    isdefined(Metal, :allowscalar) &&
        Base.invokelatest(getproperty(Metal, :allowscalar), false)
    return true
end

array_adapter(::CPU) = Array

function array_adapter(arch::GPU{:cuda})
    ensure_runtime!(arch)
    return getproperty(_load_runtime_package!(:CUDA), :CuArray)
end

function array_adapter(arch::GPU{:metal})
    ensure_runtime!(arch)
    return getproperty(_load_runtime_package!(:Metal), :MtlArray)
end

device_name(::CPU) = "CPU"

function device_name(arch::GPU{:cuda})
    ensure_runtime!(arch)
    CUDA = _load_runtime_package!(:CUDA)
    return string(Base.invokelatest(getproperty(CUDA, :name),
                                    Base.invokelatest(getproperty(CUDA, :device))))
end

function device_name(arch::GPU{:metal})
    ensure_runtime!(arch)
    Metal = _load_runtime_package!(:Metal)
    dev = isdefined(Metal, :device) ?
          Base.invokelatest(getproperty(Metal, :device)) :
          nothing
    dev === nothing && return "Metal device"
    return hasproperty(dev, :name) ? string(getproperty(dev, :name)) : string(dev)
end

architecture_label(::CPU) = "CPU"
architecture_label(arch::GPU{:cuda}) = "GPU (CUDA, $(device_name(arch)))"
architecture_label(arch::GPU{:metal}) = "GPU (Metal, $(device_name(arch)))"

synchronize_architecture!(::CPU) = nothing

function synchronize_architecture!(arch::GPU{:cuda})
    ensure_runtime!(arch)
    Base.invokelatest(getproperty(_load_runtime_package!(:CUDA), :synchronize))
    return nothing
end

function synchronize_architecture!(arch::GPU{:metal})
    ensure_runtime!(arch)
    Metal = _load_runtime_package!(:Metal)
    if isdefined(Metal, :synchronize)
        Base.invokelatest(getproperty(Metal, :synchronize))
    else
        KA.synchronize(getproperty(Metal, :MetalBackend)())
    end
    return nothing
end

function array_adapter_for(reference_array)
    ref = reference_array isa Tuple ? reference_array[1] : reference_array
    # Optional-package extensions may have loaded after this caller was
    # compiled, so cross the world-age boundary at this single startup lookup.
    adaptor = Base.invokelatest(_array_adapter_for, ref)
    # CLI entry points include the working-tree module directly. Package
    # extensions then belong to a separately loaded package module and cannot
    # extend this local module's `_array_adapter_for`. Recover the adapter from
    # an already-loaded runtime package so host forcing windows are copied to
    # the same backend as the model state.
    return adaptor === Array ? _loaded_gpu_adapter(ref) : adaptor
end

_array_adapter_for(::Any) = Array

function _loaded_gpu_adapter(reference_array, loaded_modules = Base.loaded_modules)
    for (pkgid, adapter_name) in ((_RUNTIME_PACKAGE_IDS.CUDA, :CuArray),
                                  (_RUNTIME_PACKAGE_IDS.Metal, :MtlArray))
        runtime = get(loaded_modules, pkgid, nothing)
        runtime === nothing && continue
        adaptor = getproperty(runtime, adapter_name)
        reference_array isa adaptor && return adaptor
    end
    return Array
end

"""
    reclaim_backend_pool!(reference_array)

Release device allocator caches associated with `reference_array` after
startup transients become unreachable. CPU arrays are a no-op.
"""
function reclaim_backend_pool!(reference_array)
    ref = reference_array isa Tuple ? reference_array[1] : reference_array
    return Base.invokelatest(_reclaim_backend_pool!, ref)
end

_reclaim_backend_pool!(::Any) = nothing

_is_architecture_array(::CPU, backing) = backing isa Array

function _is_architecture_array(arch::GPU{:cuda}, backing)
    return backing isa array_adapter(arch)
end

function _is_architecture_array(arch::GPU{:metal}, backing)
    return backing isa array_adapter(arch)
end

"""
    assert_residency!(storage, architecture; label="storage")

Verify that an array or tuple of arrays is resident on `architecture`. CPU
storage is returned directly; a GPU mismatch aborts rather than falling back
silently to host execution.
"""
function assert_residency!(storage, arch::AbstractArchitecture;
                           label::AbstractString = "storage")
    backing = storage isa Tuple ? parent(storage[1]) : parent(storage)
    is_gpu(arch) || return backing
    _is_architecture_array(arch, backing) || throw(ErrorException(
        "[gpu residency check] expected $(label) to live on $(backend_name(arch)) " *
        "but found $(typeof(backing)). CPU fallback aborted."))
    return backing
end

assert_float_type!(::AbstractArchitecture, ::Type{<:AbstractFloat}) = nothing

function assert_float_type!(::GPU{:metal}, ::Type{FT}) where {FT <: AbstractFloat}
    FT === Float32 || throw(ArgumentError(
        "Metal backend requires [numerics] float_type = \"Float32\"; got $(FT). " *
        "Apple Metal does not support Float64 kernels for this runtime."))
    return nothing
end

@inline function _kahan_add(s::T, c::T, x::T) where {T <: Union{Float16, Float32}}
    y = x - c
    t = s + y
    c_new = (t - s) - y
    return (t, c_new)
end

@inline _kahan_add(s::T, c::T, x::T) where {T <: Float64} = (s + x, zero(T))

"""
    _two_sum(a, b) -> (s, e)

Error-free sum (Knuth's TwoSum): `s = fl(a + b)` and `a + b == s + e` exactly.
"""
@inline function _two_sum(a::T, b::T) where T <: AbstractFloat
    s = a + b
    b_virtual = s - a
    return s, (a - (s - b_virtual)) + (b - b_virtual)
end

"""
    _neumaier_add(s, c, x) -> (s, c)

Compensated running sum: `s + c` tracks `Σx` to about one rounding of the
total, also when a term exceeds the running sum (Neumaier's variant of Kahan).
"""
@inline function _neumaier_add(s::T, c::T, x::T) where T <: AbstractFloat
    s_new, e = _two_sum(s, x)
    return s_new, c + e
end

"""
    _neumaier_sum(f, ks, T) -> (s, c)

Compensated sum of `f(k)` over `ks` in precision `T`; the total is `s + c`.
"""
@inline function _neumaier_sum(f, ks, ::Type{T}) where T
    s = c = zero(T)
    for k in ks
        s, c = _neumaier_add(s, c, f(k))
    end
    return s, c
end

"""
    _neumaier_gap(before, after) -> Σbefore − Σafter

Difference of two compensated sums of nearly equal totals. `s₀ − s₁` is exact
(Sterbenz), so the result carries one rounding of the small correction terms.
"""
@inline _neumaier_gap((s0, c0), (s1, c1)) = (s0 - s1) + (c0 - c1)

"""
    _ledger_residual(before, after, largest, n) -> r

Column mass ledger of a mass-conserving solve over `n` cells: the residual
`Σbefore − Σafter` of the compensated sums, to be added back to the column's
largest cell (`|largest|`). Only rounding-sized residuals, at most `16n` ulps
of the largest cell, are returned. A larger one means the operator itself
loses mass; it stays in the field so budgets reveal it.
"""
@inline function _ledger_residual(before, after, largest, n)
    r = _neumaier_gap(before, after)
    return abs(r) <= 16n * eps(typeof(r)) * largest ? r : zero(r)
end

# --- Compensated Float64 totals -----------------------------------------------
#
# Global diagnostics must resolve drifts of 1e-9, below what a Float32
# reduction can see. Device lanes return both terms of a compensated Float64
# sum so the host combination keeps residuals across lanes. Backends without
# Float64 kernels (Metal) copy bounded slabs to the host instead.

"""Precision diagnostic-total kernels may use on `backend` (Float32 on Metal)."""
_total_accumulator_type(_backend) = Float64

KA.@kernel function _compensated_total_lanes!(pairs, values, nlanes, nvalues)
    lane = KA.@index(Global, Linear)
    s, c = 0.0, 0.0
    @inbounds for i in lane:nlanes:nvalues
        s, c = _neumaier_add(s, c, Float64(values[i]))
    end
    @inbounds pairs[1, lane] = s
    @inbounds pairs[2, lane] = c
end

function _accumulate_host_total(s, c, values)
    @inbounds for value in values
        s, c = _neumaier_add(s, c, Float64(value))
    end
    return s, c
end

function _accumulate_total(s, c, values::AbstractArray)
    backend = KA.get_backend(values)
    backend isa KA.CPU && return _accumulate_host_total(s, c, values)
    isempty(values) && return s, c
    if _total_accumulator_type(backend) === Float64
        nlanes = min(4096, cld(length(values), 256))
        pairs = similar(values, Float64, (2, nlanes))
        _compensated_total_lanes!(backend, 256)(pairs, values, nlanes, length(values);
                                                ndrange = nlanes)
        KA.synchronize(backend)
        return _accumulate_host_total(s, c, Array(pairs))
    end
    axis = ndims(values)
    for first in 1:16:size(values, axis)
        slab = Array(selectdim(values, axis, first:min(first + 15, size(values, axis))))
        s, c = _accumulate_host_total(s, c, slab)
    end
    return s, c
end

_accumulate_total(s, c, parts::Tuple) =
    foldl((sc, part) -> _accumulate_total(sc..., part), parts; init = (s, c))

"""
    _compensated_total(values) -> Float64

Compensated Float64 total of a host or device array, or of a tuple of arrays
(cubed-sphere panels). CPU and CUDA arrays are summed in place.
"""
_compensated_total(values) = +(_accumulate_total(0.0, 0.0, values)...)

end # module Architectures
