# Benchmark series naming shared by the harness (`AtmosTransportBenchmarks`)
# and the lab lane (`benchmarking/lab/LabLane.jl`). Base only, so tests and
# the lab runner can include it without loading AtmosTransport.
#
# Dashboard series are keyed by record name, so every rule here is part of
# the published history: changing one starts new series.

"Separator between the fields of a dashboard record name."
const SERIES_SEPARATOR = " / "

"Advection scheme whose name is omitted from case names (keeps CI history)."
const DEFAULT_SERIES_SCHEME = "ppm"

"First field of every operator-phase record name."
const BREAKDOWN_SERIES_PREFIX = "Operator Breakdown"

# Vendor and product-line words that carry no information about which chip
# ran the benchmark ("NVIDIA L40S" and "L40S" are the same series).
const _DEVICE_VENDOR_PREFIX = r"^(?:NVIDIA|Apple|AMD|Intel)\s+"i
const _DEVICE_LINE_PREFIX = r"^(?:GeForce|Tesla|Quadro)\s+"i
# Data-centre board suffixes: form factor and memory size of one chip.
const _DEVICE_BOARD_SUFFIX = r"[-\s](?:PCIE|SXM\d*|NVL\d*|HGX)(?:-\d+GB)?$"i
const _DEVICE_MEMORY_SUFFIX = r"(?:[-\s]+\d+\s*GB(?:\s+HBM\d*e?)?)$"i

"""
    normalize_device_label(device_name) -> String

Short, stable device label for dashboard series names, derived from the
driver-reported device name:

1. drop the vendor prefix (`NVIDIA`, `Apple`, `AMD`, `Intel`) and a product
   line prefix (`GeForce`, `Tesla`, `Quadro`);
2. drop a data-centre board suffix (`-PCIE`, ` PCIe`, `-SXM4`, `-NVL`,
   optionally with `-40GB`) and a trailing memory size (`80GB`, `80GB HBM3`);
3. replace the series separator `/` with `-` and collapse whitespace.

Examples: `"NVIDIA L40S"` -> `"L40S"`, `"NVIDIA A100-PCIE-40GB"` -> `"A100"`,
`"Apple M5 Pro"` -> `"M5 Pro"`, `"NVIDIA GeForce RTX 4090"` -> `"RTX 4090"`.

Board variants of one chip (A100 40 GB PCIe and A100 80 GB SXM) collapse to
the same label; pass `--device-label` to keep them apart.
"""
function normalize_device_label(device_name::AbstractString)
    label = strip(String(device_name))
    label = replace(label, _DEVICE_VENDOR_PREFIX => "")
    label = replace(label, _DEVICE_LINE_PREFIX => "")
    label = replace(label, _DEVICE_BOARD_SUFFIX => "")
    label = replace(label, _DEVICE_MEMORY_SUFFIX => "")
    label = _clean_label(label)
    isempty(label) && return _clean_label(String(device_name))
    return label
end

_clean_label(s::AbstractString) = String(strip(replace(replace(String(s), "/" => "-"), r"\s+" => " ")))

"""
    validate_device_label(label) -> String

Check a user-supplied `--device-label` override: non-empty, no `/` (the
series separator), no leading or trailing whitespace.
"""
function validate_device_label(label::AbstractString)
    s = String(label)
    isempty(strip(s)) && throw(ArgumentError("device label must not be empty"))
    occursin('/', s) &&
        throw(ArgumentError("device label $(repr(s)) must not contain '/', the series separator"))
    s == strip(s) ||
        throw(ArgumentError("device label $(repr(s)) has leading or trailing whitespace"))
    return s
end

"""
    backend_series_label(backend_class, device_label) -> String

Backend field of a series name. CPU classes (`CPU-AMD`, `CPU-Intel`,
`CPU-Apple`) stay as they are so the CI history continues; GPU classes carry
the device: `GPU-CUDA L40S`, `GPU-CUDA A100`, `GPU-Metal M5 Pro`.
"""
function backend_series_label(backend_class::AbstractString, device_label)
    startswith(backend_class, "CPU") && return String(backend_class)
    device_label === nothing &&
        throw(ArgumentError("GPU backend class $(backend_class) needs a device label"))
    return string(backend_class, " ", validate_device_label(device_label))
end

"""
    case_series_name(; group, grid, levels, operator, tracers, scheme,
                       backend_label, float_type) -> String

Dashboard name of one benchmark case, for example
`"Lab Synthetic CS Sweep / C180 L72 full 12tr / GPU-CUDA A100 / Float64"`.
The scheme is appended to the case field only when it differs from
`$(DEFAULT_SERIES_SCHEME)`, so existing series keep their names.
"""
function case_series_name(; group::AbstractString, grid::AbstractString,
                          levels::Integer, operator::AbstractString,
                          tracers::Integer, scheme::AbstractString,
                          backend_label::AbstractString,
                          float_type::AbstractString)
    case = "$(grid) L$(levels) $(operator) $(tracers)tr"
    scheme == DEFAULT_SERIES_SCHEME || (case *= " " * scheme)
    return join((group, case, backend_label, float_type), SERIES_SEPARATOR)
end

"""
    breakdown_series_name(case_name, phase) -> String

Dashboard name of one operator-phase record: `$(BREAKDOWN_SERIES_PREFIX)`, the
full case name (`case_series_name`), and the phase, for example
`"$(BREAKDOWN_SERIES_PREFIX) / Lab Synthetic CS Sweep / C90 L72 full 12tr / GPU-CUDA L40S / Float32 / convection"`.
Every case field (group, grid, levels, operator, tracers, scheme, backend,
float type) is part of the name, so two cases never share a phase series.
"""
breakdown_series_name(case_name::AbstractString, phase::AbstractString) =
    join((BREAKDOWN_SERIES_PREFIX, case_name, phase), SERIES_SEPARATOR)
