# Device-memory estimate of a benchmark case and the decision to skip a case
# that does not fit. Base only, so the test suite checks it without loading
# AtmosTransport.
#
# The estimate is a linear model of the case's array allocations, fitted to
# measurements on CPU (benchmarking/README.md, "Device memory"): the summed
# size of every array reachable from the model after construction and one
# step (`Base.summarysize`), for C8-C64, L16/L40, 1 and 3 tracers, validated
# at C90 and C180 L72 with 2 tracers (fit within 0.02 %). Sizes are counted in
# elements of the float type; the coefficients are the fitted values rounded
# up, and a negative fitted term is dropped, so the estimate is never below
# the measured size: 1-17 % above it at C90-C180 (the fixed allowance
# dominates on tiny grids).

"Halo width of the synthetic meshes, as in the harness."
const MEMORY_MODEL_HALO = 3

"""
Per-operator coefficients of the model-memory estimate, in elements of the
float type: `volume` per haloed cell `6 (Nc+2Hp)² Nz`, `tracer_volume` per
haloed cell and tracer, `interior_volume` per interior cell `6 Nc² Nz`, and
`area` per haloed column `6 (Nc+2Hp)²`.
"""
const MODEL_MEMORY_COEFFICIENTS = Dict(
    :advection  => (volume = 5.91, tracer_volume = 2.20, interior_volume = 0.00, area = 1.5),
    :diffusion  => (volume = 4.11, tracer_volume = 1.03, interior_volume = 1.85, area = 3.1),
    :convection => (volume = 4.17, tracer_volume = 1.00, interior_volume = 3.84, area = 2.3),
    :full       => (volume = 5.82, tracer_volume = 2.22, interior_volume = 5.43, area = 4.0),
)

"""
Adjoint allocations beyond its payload, in elements per haloed cell: a base
term plus a term per step (tape and reverse sweep). Bounds the measured
cumulative allocation of `cs_surface_emission_footprint` on CPU (7.9-8.4
elements per cell per step at C32 L40), which is itself an upper bound of the
live memory.
"""
const ADJOINT_MEMORY_ELEMENTS = (base = 3.0, per_step = 8.5)

"Fixed allowance for small allocations (reductions, launch parameters) per case."
const MEMORY_FIXED_BYTES = 16 * 2^20

"""
Margin between the estimate and the memory a case may use: 5 % plus 0.5 GiB
for kernel temporaries and memory-pool fragmentation on the device.
"""
const MEMORY_SAFETY = (factor = 1.05, bytes = 2^29)

_haloed_volume(Nc, Nz) = 6 * (Nc + 2MEMORY_MODEL_HALO)^2 * Nz
_haloed_area(Nc) = 6 * (Nc + 2MEMORY_MODEL_HALO)^2

"""
    tm5_legacy_tile_bytes(float_bytes, Nc, Nz) -> Int

Bytes of the TM5 convection column tile (`TM5Workspace`, 1 GiB budget,
`derive_tile_columns`), which the per-thread LU solve allocates on first use:
on the CPU and wherever the collaborative GPU solve does not engage. A launch
covers one haloed panel.
"""
function tm5_legacy_tile_bytes(float_bytes::Integer, Nc::Integer, Nz::Integer)
    per_column = float_bytes * (Nz * Nz + 2 * (Nz + 1)) + sizeof(Int) * (Nz + 3)
    columns = min((Nc + 2MEMORY_MODEL_HALO)^2, max(256, fld(2^30, per_column)))
    return columns * per_column
end

"""
    io_payload_bytes(float_bytes, Nc, Nz, Nt) -> Int

Exact device bytes of the `io` payload: air mass, `Nt` tracers, and the x, y,
and level-interface face fluxes on six haloed panels.
"""
function io_payload_bytes(float_bytes::Integer, Nc::Integer, Nz::Integer, Nt::Integer)
    N = Nc + 2MEMORY_MODEL_HALO
    elements = 6 * ((1 + Nt) * N * N * Nz + 2 * (N + 1) * N * Nz + N * N * (Nz + 1))
    return float_bytes * elements
end

"""
    estimate_case_bytes(operator, float_bytes, Nc, Nz, Nt; steps, tm5_legacy_tile, device_is_host) -> Int

Estimated device memory of one benchmark case: the fitted model-memory
estimate (`MODEL_MEMORY_COEFFICIENTS`), plus the TM5 column tile for
`convection` and `full` when `tm5_legacy_tile` (the collaborative solve does
not engage); for `io` the payload, twice when `device_is_host` (CPU runs:
the read-back copy is allocated while the payload is live; a GPU's read-back
copy is in host memory); the payload plus the adjoint's tape allowance for
`adjoint`. `MEMORY_FIXED_BYTES` is added to every case.
"""
function estimate_case_bytes(operator::Symbol, float_bytes::Integer, Nc::Integer, Nz::Integer,
                             Nt::Integer; steps::Integer, tm5_legacy_tile::Bool, device_is_host::Bool)
    V = _haloed_volume(Nc, Nz)
    bytes = if operator === :io
        (device_is_host ? 2 : 1) * io_payload_bytes(float_bytes, Nc, Nz, Nt)
    elseif operator === :adjoint
        io_payload_bytes(float_bytes, Nc, Nz, 1) +
            float_bytes * V * (ADJOINT_MEMORY_ELEMENTS.base + ADJOINT_MEMORY_ELEMENTS.per_step * steps)
    else
        c = MODEL_MEMORY_COEFFICIENTS[operator]
        elements = c.volume * V + c.tracer_volume * V * Nt +
                   c.interior_volume * 6 * Nc^2 * Nz + c.area * _haloed_area(Nc)
        tile = operator in (:convection, :full) && tm5_legacy_tile ?
               tm5_legacy_tile_bytes(float_bytes, Nc, Nz) : 0
        float_bytes * elements + tile
    end
    return ceil(Int, bytes) + MEMORY_FIXED_BYTES
end

"Device memory a case of `estimate_bytes` may need, with `MEMORY_SAFETY`."
required_device_bytes(estimate_bytes::Integer) =
    ceil(Int, MEMORY_SAFETY.factor * estimate_bytes) + MEMORY_SAFETY.bytes

gib(bytes::Real) = round(bytes / 2^30; digits = 2)

"""
    memory_skip_reason(required_bytes, available_bytes) -> Union{Nothing, String}

`nothing` when the case fits; otherwise the reason recorded with a
`skipped_memory` case. An unknown `available_bytes` (`nothing`) never skips.
"""
function memory_skip_reason(required_bytes::Integer, available_bytes::Union{Nothing, Integer})
    available_bytes === nothing && return nothing
    required_bytes <= available_bytes && return nothing
    return "needs an estimated $(gib(required_bytes)) GiB of device memory, " *
           "$(gib(available_bytes)) GiB available"
end

"""
    is_out_of_memory(err) -> Bool

Whether `err` (possibly wrapped in a task or composite exception) is an
out-of-memory error: Julia's `OutOfMemoryError`, CUDA.jl's
`OutOfGPUMemoryError`, or an allocation failure reported by message (Metal).
"""
function is_out_of_memory(err)
    err isa OutOfMemoryError && return true
    err isa TaskFailedException && return is_out_of_memory(err.task.exception)
    err isa CompositeException && return any(is_out_of_memory, err.exceptions)
    err isa CapturedException && return is_out_of_memory(err.ex)
    nameof(typeof(err)) === :OutOfGPUMemoryError && return true
    message = try
        sprint(showerror, err)
    catch
        ""
    end
    return occursin(r"out of (gpu |device )?memory|insufficient memory|failed to allocate|allocation fail"i, message)
end
