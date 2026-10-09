# Transport-binary contract construction and validation.

# Header JSON is null-terminated inside the fixed-width header region. Read it
# incrementally so metadata may exceed the historical 256 KiB probe without
# scanning an unbounded payload when a file is corrupt.
const _TRANSPORT_HEADER_READ_CHUNK_BYTES = 262_144
const _TRANSPORT_MAX_HEADER_JSON_BYTES = 16 * 1024 * 1024

function _read_transport_header_json(io::IO; source::AbstractString = "transport binary")
    seekstart(io)
    json_bytes = UInt8[]
    sizehint!(json_bytes, _TRANSPORT_HEADER_READ_CHUNK_BYTES)

    while length(json_bytes) < _TRANSPORT_MAX_HEADER_JSON_BYTES
        remaining = _TRANSPORT_MAX_HEADER_JSON_BYTES - length(json_bytes)
        chunk = read(io, min(_TRANSPORT_HEADER_READ_CHUNK_BYTES, remaining))
        isempty(chunk) && throw(ArgumentError(
            "$(source) JSON header is not null-terminated before end of file"))

        null_index = findfirst(==(0x00), chunk)
        if null_index !== nothing
            append!(json_bytes, @view chunk[1:(null_index - 1)])
            isempty(json_bytes) && throw(ArgumentError(
                "$(source) JSON header is empty"))
            return json_bytes
        end
        append!(json_bytes, chunk)
    end

    throw(ArgumentError(
        "$(source) JSON header is not null-terminated within " *
        "$(_TRANSPORT_MAX_HEADER_JSON_BYTES) bytes"))
end

"""
    TransportBinaryContract(; source_flux_sampling, air_mass_sampling,
                              flux_sampling, flux_kind, delta_semantics,
                              humidity_sampling,
                              poisson_balance_target_scale,
                              poisson_balance_target_semantics)

Self-describing transport-binary timing/basis contract. All eight fields
are required — no defaults — so a writer cannot produce an ambiguous
binary. Readers call [`validate_transport_contract!`](@ref) on the parsed
header to decide whether the file is trustworthy.

Canonical usage: construct via [`canonical_window_constant_contract`](@ref)
for window-constant, per-substep mass fluxes.

Symbol fields are validated against the `_TRANSPORT_ALLOWED_*` tuples at
construction time. Combinations are also checked:
- `delta_semantics === :forward_window_endpoint_difference` requires the
  payload to carry `dm` (or `dm + dhflux`); the writer is responsible for
  honoring this.
- `humidity_sampling === :window_endpoints` requires `qv_start` + `qv_end`
  in the payload; `:none` requires neither.
"""
struct TransportBinaryContract
    source_flux_sampling             :: Symbol
    air_mass_sampling                :: Symbol
    flux_sampling                    :: Symbol
    flux_kind                        :: Symbol
    delta_semantics                  :: Symbol
    humidity_sampling                :: Symbol
    poisson_balance_target_scale     :: Float64
    poisson_balance_target_semantics :: String

    function TransportBinaryContract(source_flux_sampling::Symbol,
                                     air_mass_sampling::Symbol,
                                     flux_sampling::Symbol,
                                     flux_kind::Symbol,
                                     delta_semantics::Symbol,
                                     humidity_sampling::Symbol,
                                     poisson_balance_target_scale::Real,
                                     poisson_balance_target_semantics::AbstractString)
        sfs = _transport_validate_source_flux_sampling(source_flux_sampling)
        ams = _transport_normalize_symbol(air_mass_sampling)
        fs  = _transport_normalize_symbol(flux_sampling)
        fk  = _transport_normalize_symbol(flux_kind)
        ds  = _transport_normalize_symbol(delta_semantics)
        hs  = _transport_normalize_symbol(humidity_sampling)
        ams in _TRANSPORT_ALLOWED_AIR_MASS_SAMPLINGS ||
            throw(ArgumentError("air_mass_sampling=$(ams) not in $(Tuple(_TRANSPORT_ALLOWED_AIR_MASS_SAMPLINGS))"))
        fs in _TRANSPORT_ALLOWED_FLUX_SAMPLINGS ||
            throw(ArgumentError("flux_sampling=$(fs) not in $(Tuple(_TRANSPORT_ALLOWED_FLUX_SAMPLINGS))"))
        fk in _TRANSPORT_ALLOWED_FLUX_KINDS ||
            throw(ArgumentError("flux_kind=$(fk) not in $(Tuple(_TRANSPORT_ALLOWED_FLUX_KINDS))"))
        ds in _TRANSPORT_ALLOWED_DELTA_SEMANTICS ||
            throw(ArgumentError("delta_semantics=$(ds) not in $(Tuple(_TRANSPORT_ALLOWED_DELTA_SEMANTICS))"))
        hs in _TRANSPORT_ALLOWED_HUMIDITY_SAMPLINGS ||
            throw(ArgumentError("humidity_sampling=$(hs) not in $(Tuple(_TRANSPORT_ALLOWED_HUMIDITY_SAMPLINGS))"))
        scale = Float64(poisson_balance_target_scale)
        isfinite(scale) && scale > 0 ||
            throw(ArgumentError("poisson_balance_target_scale must be finite and > 0"))
        new(sfs, ams, fs, fk, ds, hs, Float64(poisson_balance_target_scale),
            String(poisson_balance_target_semantics))
    end
end

# Keyword constructor — all fields required.
function TransportBinaryContract(; source_flux_sampling::Symbol,
                                   air_mass_sampling::Symbol,
                                   flux_sampling::Symbol,
                                   flux_kind::Symbol,
                                   delta_semantics::Symbol,
                                   humidity_sampling::Symbol,
                                   poisson_balance_target_scale::Real,
                                   poisson_balance_target_semantics::AbstractString)
    return TransportBinaryContract(source_flux_sampling, air_mass_sampling,
                                   flux_sampling, flux_kind, delta_semantics,
                                   humidity_sampling,
                                   poisson_balance_target_scale,
                                   poisson_balance_target_semantics)
end

"""
    canonical_window_constant_contract(; steps_per_window,
                                         humidity_sampling = :none,
                                         source_flux_sampling = :window_start_endpoint,
                                         include_flux_delta = true) -> TransportBinaryContract

Build the canonical contract for window-constant, per-substep mass amounts.
The
Poisson target scale is `1 / (2 * steps_per_window)` — matching the TM5
r1112 horizontal-sweep count of `2 * steps_per_window` per window.

`include_flux_delta = true` implies `delta_semantics =
:forward_window_endpoint_difference` (the writer must include `dm` in the
payload); `false` implies `:none`.
"""
function canonical_window_constant_contract(;
        steps_per_window::Integer,
        humidity_sampling::Symbol = :none,
        source_flux_sampling::Symbol = :window_start_endpoint,
        include_flux_delta::Bool = true)
    return TransportBinaryContract(
        source_flux_sampling = source_flux_sampling,
        air_mass_sampling    = :window_start_endpoint,
        flux_sampling        = :window_constant,
        flux_kind            = :substep_mass_amount,
        delta_semantics      = include_flux_delta ? :forward_window_endpoint_difference : :none,
        humidity_sampling    = humidity_sampling,
        poisson_balance_target_scale = 1.0 / (2 * Int(steps_per_window)),
        poisson_balance_target_semantics = "forward_window_mass_difference / (2 * steps_per_window)",
    )
end

# The cubed-sphere header keys the RUNTIME reads to decide execution cadence and
# capabilities (`uses_binary_substep_contract`, `binary_capabilities`). Emitted
# with defaults by `open_streaming_cs_transport_binary` so they can never go
# missing; listed here once so the writer-side guard and the emitter agree.
const _CS_WRITER_CONTRACT_KEYS = ("runtime_substep_contract",
                                  "preprocessor_contract",
                                  "adaptive_substeps")

const _TRANSPORT_STRUCTURAL_HEADER_KEYS = Set((
    "magic", "format_version", "header_bytes", "float_type", "float_bytes",
    "grid_type", "horizontal_topology", "planet_radius_m", "ncell", "nface_h", "nlevel",
    "nwindow", "A_ifc", "B_ifc", "mass_basis", "payload_sections",
    "elems_per_window", "n_geometry_elems", "Nx", "Ny", "Nc", "npanel",
    "lons", "lats", "longitude_interval", "latitude_interval",
    "nlat", "latitudes", "nlon_per_ring",
    "panel_convention", "cs_definition", "cs_coordinate_law",
    "cs_center_law", "longitude_offset_deg",
    "include_qv_endpoints", "include_flux_delta",
))

"""Merge caller metadata without permitting it to rewrite the binary layout."""
function _merge_transport_extra_header!(header::Dict{String, Any}, extra_header)
    for (raw_key, value) in pairs(extra_header)
        key = String(raw_key)
        if key in _TRANSPORT_STRUCTURAL_HEADER_KEYS && haskey(header, key) &&
           !isequal(header[key], value)
            throw(ArgumentError(
                "extra_header cannot override structural field $(repr(key)): " *
                "writer computed $(repr(header[key])), caller supplied $(repr(value))"))
        end
        header[key] = value
    end
    return header
end

function _transport_required_int(header, key::String; positive::Bool = true)
    haskey(header, key) || throw(ArgumentError(
        "Transport-binary contract violation — missing $(key)"))
    value = header[key]
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "Transport-binary contract violation — $(key) must be an integer; got $(repr(value))"))
    result = Int(value)
    positive && result <= 0 && throw(ArgumentError(
        "Transport-binary contract violation — $(key) must be positive; got $(result)"))
    return result
end

function _validate_transport_layout!(header::AbstractDict)
    header_bytes = _transport_required_int(header, "header_bytes")
    float_bytes = _transport_required_int(header, "float_bytes")
    float_type = String(get(header, "float_type", ""))
    expected_float_bytes = float_type == "Float32" ? 4 : float_type == "Float64" ? 8 : nothing
    expected_float_bytes === nothing && throw(ArgumentError(
        "Transport-binary contract violation — float_type must be Float32 or Float64; got $(repr(float_type))"))
    float_bytes == expected_float_bytes || throw(ArgumentError(
        "Transport-binary contract violation — float_bytes=$(float_bytes) does not match float_type=$(float_type)"))

    ncell = _transport_required_int(header, "ncell")
    nface_h = _transport_required_int(header, "nface_h")
    nlevel = _transport_required_int(header, "nlevel")
    nwindow = _transport_required_int(header, "nwindow")
    elems_per_window = _transport_required_int(header, "elems_per_window")
    n_geometry_elems = _transport_required_int(header, "n_geometry_elems"; positive = false)
    n_geometry_elems >= 0 || throw(ArgumentError(
        "Transport-binary contract violation — n_geometry_elems must be nonnegative"))

    basis = lowercase(String(get(header, "mass_basis", "")))
    basis in ("dry", "moist") || throw(ArgumentError(
        "Transport-binary contract violation — mass_basis must be dry or moist; got $(repr(basis))"))
    for key in ("dt_met_seconds", "half_dt_seconds")
        value = try Float64(header[key]) catch; NaN end
        isfinite(value) && value > 0 || throw(ArgumentError(
            "Transport-binary contract violation — $(key) must be finite and positive"))
    end
    encoded_header_bytes = try
        ncodeunits(JSON3.write(header))
    catch
        throw(ArgumentError(
            "Transport-binary contract violation — header contains values that cannot be encoded as JSON"))
    end
    encoded_header_bytes < _TRANSPORT_MAX_HEADER_JSON_BYTES || throw(ArgumentError(
        "Transport-binary contract violation — encoded JSON header must be smaller than " *
        "$(_TRANSPORT_MAX_HEADER_JSON_BYTES) bytes"))
    header_bytes > encoded_header_bytes || throw(ArgumentError(
        "Transport-binary contract violation — header_bytes leaves no room for a null terminator"))

    A = try Float64.(collect(header["A_ifc"])) catch; Float64[] end
    B = try Float64.(collect(header["B_ifc"])) catch; Float64[] end
    length(A) == nlevel + 1 || throw(ArgumentError(
        "Transport-binary contract violation — A_ifc must have nlevel + 1 entries"))
    length(B) == nlevel + 1 || throw(ArgumentError(
        "Transport-binary contract violation — B_ifc must have nlevel + 1 entries"))
    all(isfinite, A) && all(isfinite, B) || throw(ArgumentError(
        "Transport-binary contract violation — A_ifc and B_ifc must be finite"))

    sections = try Symbol.(lowercase.(String.(collect(header["payload_sections"])))) catch; Symbol[] end
    isempty(sections) && throw(ArgumentError(
        "Transport-binary contract violation — payload_sections must be a nonempty list"))
    length(unique(sections)) == length(sections) || throw(ArgumentError(
        "Transport-binary contract violation — payload_sections contains duplicates"))
    :qv in sections && throw(ArgumentError(
        "Transport-binary contract violation — single-field qv is obsolete; " *
        "regenerate with qv_start/qv_end endpoints"))

    endpoint_humidity = (:qv_start in sections, :qv_end in sections)
    endpoint_humidity[1] == endpoint_humidity[2] || throw(ArgumentError(
        "Transport-binary contract violation — qv_start and qv_end must appear together"))
    tm5 = map(s -> s in sections, (:entu, :detu, :entd, :detd))
    all(tm5) || !any(tm5) || throw(ArgumentError(
        "Transport-binary contract violation — TM5 convection requires entu, detu, entd, and detd together"))
    surface = map(s -> s in sections, _PBL_SURFACE_PAYLOAD_SECTIONS)
    all(surface) || !any(surface) || throw(ArgumentError(
        "Transport-binary contract violation — surface payload sections must be complete"))
    (:pbl_eflux in sections && !all(surface)) && throw(ArgumentError(
        "Transport-binary contract violation — pbl_eflux requires the PBL surface sections"))
    (:cmfmc_cloud_base in sections && !(:cmfmc in sections)) && throw(ArgumentError(
        "Transport-binary contract violation — cmfmc_cloud_base requires cmfmc"))
    vdiff = map(s -> s in sections, _GCHP_VDIFF_PAYLOAD_SECTIONS)
    all(vdiff) || !any(vdiff) || throw(ArgumentError(
        "Transport-binary contract violation — GCHP VDIFF payload sections must be complete"))
    :kz in sections && throw(ArgumentError(
        "Transport-binary contract violation — :kz is not part of format v$(TRANSPORT_BINARY_FORMAT_VERSION); " *
        "write exact dry-air interface exchange as :dkg"))
    (:dkg in sections && basis != "dry") && throw(ArgumentError(
        "Transport-binary contract violation — exact :dkg requires mass_basis=dry"))

    grid_type = lowercase(String(get(header, "grid_type", "")))
    topology = lowercase(String(get(header, "horizontal_topology", "")))
    expected_elems = if grid_type == "latlon" && topology == "structureddirectional"
        all(s -> s in sections, (:m, :am, :bm, :cm, :ps)) || throw(ArgumentError(
            "Transport-binary contract violation — latlon payload requires m, am, bm, cm, and ps"))
        Nx = _transport_required_int(header, "Nx")
        Ny = _transport_required_int(header, "Ny")
        Nx * Ny == ncell || throw(ArgumentError(
            "Transport-binary contract violation — ncell must equal Nx * Ny"))
        for (key, expected_length) in (("lons", Nx), ("lats", Ny))
            haskey(header, key) || throw(ArgumentError(
                "Transport-binary contract violation — latlon header is missing $(key)"))
            values = header[key]
            values isa AbstractVector && length(values) == expected_length ||
                throw(ArgumentError(
                    "Transport-binary contract violation — $(key) must contain $(expected_length) coordinates"))
            all(value -> value isa Real && !(value isa Bool) && isfinite(value), values) ||
                throw(ArgumentError(
                    "Transport-binary contract violation — $(key) coordinates must be finite numbers"))
        end
        lons = Float64.(header["lons"])
        lats = Float64.(header["lats"])
        all(>(0), diff(lons)) || throw(ArgumentError(
            "Transport-binary contract violation — lons must be strictly increasing"))
        all(>(0), diff(lats)) && all(lat -> -90 <= lat <= 90, lats) ||
            throw(ArgumentError(
                "Transport-binary contract violation — lats must be strictly increasing within [-90, 90]"))
        expected_nface_h = (Nx + 1) * Ny + Nx * (Ny + 1)
        nface_h == expected_nface_h || throw(ArgumentError(
            "Transport-binary contract violation — nface_h=$(nface_h), expected $(expected_nface_h) for Nx=$(Nx), Ny=$(Ny)"))
        sum(_transport_structured_section_elements(Nx, Ny, ncell, nlevel, section)
            for section in sections)
    elseif grid_type == "reduced_gaussian" && topology == "faceindexed"
        all(s -> s in sections, (:m, :hflux, :cm, :ps)) || throw(ArgumentError(
            "Transport-binary contract violation — reduced-Gaussian payload requires m, hflux, cm, and ps"))
        nlat = _transport_required_int(header, "nlat")
        for key in ("latitudes", "nlon_per_ring")
            haskey(header, key) || throw(ArgumentError(
                "Transport-binary contract violation — reduced-Gaussian header is missing $(key)"))
            header[key] isa AbstractVector && length(header[key]) == nlat ||
                throw(ArgumentError(
                    "Transport-binary contract violation — $(key) must contain nlat=$(nlat) entries"))
        end
        latitudes = header["latitudes"]
        all(value -> value isa Real && !(value isa Bool) && isfinite(value), latitudes) ||
            throw(ArgumentError(
                "Transport-binary contract violation — latitudes must be finite numbers"))
        latitudes_f64 = Float64.(latitudes)
        all(>(0), diff(latitudes_f64)) &&
            all(lat -> -90 < lat < 90, latitudes_f64) || throw(ArgumentError(
                "Transport-binary contract violation — reduced-Gaussian latitudes must be " *
                "strictly increasing within (-90, 90)"))
        nlon_per_ring = header["nlon_per_ring"]
        all(value -> value isa Integer && !(value isa Bool) && value > 0,
            nlon_per_ring) || throw(ArgumentError(
                "Transport-binary contract violation — nlon_per_ring entries must be positive integers"))
        sum(nlon_per_ring) == ncell || throw(ArgumentError(
            "Transport-binary contract violation — sum(nlon_per_ring) must equal ncell"))
        expected_nface_h = nfaces(ReducedGaussianMesh(
            latitudes_f64, Int.(nlon_per_ring); FT=Float64))
        nface_h == expected_nface_h || throw(ArgumentError(
            "Transport-binary contract violation — nface_h=$(nface_h), expected " *
            "$(expected_nface_h) from reduced-Gaussian ring geometry"))
        sum(_transport_faceindexed_section_elements(ncell, nface_h, nlevel, section)
            for section in sections)
    elseif grid_type == "cubed_sphere" && topology == "structureddirectional"
        all(s -> s in sections, (:m, :am, :bm, :cm, :ps)) || throw(ArgumentError(
            "Transport-binary contract violation — cubed-sphere payload requires m, am, bm, cm, and ps"))
        Nc = _transport_required_int(header, "Nc")
        npanel = _transport_required_int(header, "npanel")
        npanel == 6 || throw(ArgumentError(
            "Transport-binary contract violation — cubed-sphere npanel must equal 6"))
        npanel * Nc * Nc == ncell || throw(ArgumentError(
            "Transport-binary contract violation — ncell must equal npanel * Nc^2"))
        expected_nface_h = npanel * 2 * Nc * (Nc + 1)
        nface_h == expected_nface_h || throw(ArgumentError(
            "Transport-binary contract violation — nface_h=$(nface_h), expected $(expected_nface_h) for C$(Nc)"))
        for key in ("panel_convention", "cs_definition", "cs_coordinate_law",
                    "cs_center_law", "longitude_offset_deg")
            haskey(header, key) || throw(ArgumentError(
                "Transport-binary contract violation — cubed-sphere header is missing $(key)"))
        end
        convention = lowercase(String(header["panel_convention"]))
        definition = lowercase(String(header["cs_definition"]))
        coordinate_law = lowercase(String(header["cs_coordinate_law"]))
        center_law = lowercase(String(header["cs_center_law"]))
        expected_geometry = convention == "gnomonic" ?
            ("equiangular_gnomonic", "equiangular_gnomonic", "angular_midpoint") :
            convention == "geos_native" ?
            ("gmao_equal_distance", "gmao_equal_distance_gnomonic", "four_corner_normalized") :
            throw(ArgumentError(
                "Transport-binary contract violation — panel_convention must be gnomonic or geos_native"))
        (definition, coordinate_law, center_law) == expected_geometry || throw(ArgumentError(
            "Transport-binary contract violation — cubed-sphere geometry tags do not match " *
            "panel_convention=$(convention); expected $(expected_geometry)"))
        offset = header["longitude_offset_deg"]
        offset isa Real && !(offset isa Bool) && isfinite(offset) || throw(ArgumentError(
            "Transport-binary contract violation — longitude_offset_deg must be finite"))
        sum(_cs_section_elements(Nc, npanel, nlevel, section) for section in sections)
    else
        throw(ArgumentError(
            "Transport-binary contract violation — unsupported grid/topology $(repr(grid_type))/$(repr(topology))"))
    end
    expected_elems == elems_per_window || throw(ArgumentError(
        "Transport-binary contract violation — elems_per_window=$(elems_per_window), expected $(expected_elems) from payload_sections"))

    flux_kind = _transport_normalize_symbol(header["flux_kind"])
    if grid_type != "cubed_sphere" && flux_kind !== :substep_mass_amount
        throw(ArgumentError(
            "Transport-binary contract violation — $(grid_type) runtime requires " *
            "flux_kind=substep_mass_amount; got $(flux_kind)"))
    end

    humidity = _transport_normalize_symbol(header["humidity_sampling"])
    expected_humidity = endpoint_humidity[1] ? :window_endpoints : :none
    humidity == expected_humidity || throw(ArgumentError(
        "Transport-binary contract violation — humidity_sampling=$(humidity) does not match payload sections"))
    has_delta = any(s -> s in sections, (:dam, :dbm, :dhflux, :dcm, :dm))
    delta = _transport_normalize_symbol(header["delta_semantics"])
    (has_delta == (delta === :forward_window_endpoint_difference)) || throw(ArgumentError(
        "Transport-binary contract violation — delta_semantics=$(delta) does not match delta payload sections"))

    include_qv_endpoints = get(header, "include_qv_endpoints", nothing)
    include_qv_endpoints isa Bool && include_qv_endpoints == endpoint_humidity[1] ||
        throw(ArgumentError(
            "Transport-binary contract violation — include_qv_endpoints must match qv_start/qv_end"))
    include_flux_delta = get(header, "include_flux_delta", nothing)
    include_flux_delta isa Bool && include_flux_delta == has_delta || throw(ArgumentError(
        "Transport-binary contract violation — include_flux_delta must match delta payload sections"))
    for (key, section) in (("n_qv_start", :qv_start), ("n_qv_end", :qv_end))
        declared = _transport_required_int(header, key; positive=false)
        expected = section in sections ? ncell * nlevel : 0
        declared == expected || throw(ArgumentError(
            "Transport-binary contract violation — $(key)=$(declared), expected $(expected)"))
    end

    expected_bytes = try
        payload_elems = Base.checked_add(
            n_geometry_elems, Base.checked_mul(nwindow, elems_per_window))
        Base.checked_add(header_bytes, Base.checked_mul(payload_elems, float_bytes))
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError(
            "Transport-binary contract violation — declared payload byte count overflows Int"))
    end
    return expected_bytes
end

"""
    validate_cs_writer_contract!(header::AbstractDict)

Write-time guard: assert every runtime-read cubed-sphere contract key
(`_CS_WRITER_CONTRACT_KEYS`) is present before a binary is finalized — the
writer-side mirror of [`validate_transport_contract!`]. The single choke point
`open_streaming_cs_transport_binary` emits these keys with defaults, so this
never fires in normal use; it exists to fail LOUDLY if a future refactor drops
the default emission, rather than silently shipping a binary that makes the
runtime run convection/chemistry per advection substep (the 2026-05-31 N320
regression: a new source path omitted `runtime_substep_contract`).
"""
function validate_cs_writer_contract!(header::AbstractDict)
    absent = [k for k in _CS_WRITER_CONTRACT_KEYS if !haskey(header, k)]
    isempty(absent) || error(
        "CS transport-binary writer contract violation — runtime-read header " *
        "keys absent: $(join(absent, ", ")). These are emitted with defaults by " *
        "`open_streaming_cs_transport_binary`; a caller or refactor has bypassed " *
        "that single source of truth.")
    return nothing
end

"""
    validate_transport_contract!(header::AbstractDict)

Assert that `header` declares the current transport-binary contract and that
the timing metadata is self-consistent. `format_version` is a hard boundary:
only `TRANSPORT_BINARY_FORMAT_VERSION` is accepted. Older files are obsolete
and must be regenerated rather than loaded through compatibility defaults.

Shared between `TransportBinaryDriver`, `TransportBinaryReader`, and the
`scripts/diagnostics/inspect_transport_binary.jl` tool so there is ONE
validator every reader-facing tool calls.
"""
function validate_transport_contract!(header::AbstractDict)
    missing_or_unknown = String[]
    magic = get(header, "magic", nothing)
    magic == "MFLX" || throw(ArgumentError(
        "Transport-binary contract violation — expected magic=\"MFLX\", got $(repr(magic)). " *
        "This is not a current transport binary."))

    haskey(header, "format_version") || throw(ArgumentError(
        "Transport-binary contract violation — missing format_version. " *
        "All pre-v$(TRANSPORT_BINARY_FORMAT_VERSION) transport binaries are obsolete; regenerate."))
    format_version = try
        Int(header["format_version"])
    catch
        throw(ArgumentError("Transport-binary contract violation — invalid format_version=$(repr(header["format_version"]))"))
    end
    format_version == TRANSPORT_BINARY_FORMAT_VERSION || throw(ArgumentError(
        "Obsolete transport binary format_version=$(format_version); current runtime requires " *
        "format_version=$(TRANSPORT_BINARY_FORMAT_VERSION). Regenerate this file with the current " *
        "preprocessor so the header carries the per-window substep schedule and runtime contract."))

    radius = get(header, "planet_radius_m", nothing)       # absent in older binaries
    radius === nothing || (radius isa Real && !(radius isa Bool) && isfinite(radius) && radius > 0) ||
        throw(ArgumentError("Transport-binary contract violation — planet_radius_m must be a positive " *
                            "length in m; got $(repr(radius))"))

    runtime_contract = get(header, "runtime_substep_contract", nothing)
    if runtime_contract !== nothing
        String(runtime_contract) == "binary_schedule" ||
            throw(ArgumentError("Transport-binary contract violation — unknown runtime_substep_contract=$(repr(runtime_contract))"))
        grid_type = lowercase(String(get(header, "grid_type", "")))
        if grid_type != "cubed_sphere"
            throw(ArgumentError(
                "Transport-binary contract violation — runtime_substep_contract=\"binary_schedule\" " *
                "is currently supported only by cubed-sphere runtime forcing. LL/RG binaries " *
                "would otherwise fall back to the runtime CFL pilot and double-subcycle adaptive " *
                "schedules. Add an LL/RG runtime contract before writing adaptive LL/RG binaries."
            ))
        end
    end

    fields = ("source_flux_sampling", "air_mass_sampling", "flux_sampling",
              "flux_kind", "delta_semantics", "humidity_sampling",
              "poisson_balance_target_scale", "poisson_balance_target_semantics",
              "nwindow", "steps_per_window", "steps_per_window_by_window",
              "time_step_schedule", "poisson_balance_target_scale_by_window")

    for f in fields
        if !haskey(header, f)
            push!(missing_or_unknown, "$f (missing)")
        else
            val = header[f]
            if f == "poisson_balance_target_scale"
                # NaN or ≤0 → unknown
                vf = try Float64(val) catch; NaN end
                (isnan(vf) || vf <= 0) && push!(missing_or_unknown, "$f (value=$val)")
            elseif f == "poisson_balance_target_semantics"
                isempty(String(val)) && push!(missing_or_unknown, "$f (empty)")
            elseif f in ("source_flux_sampling", "air_mass_sampling", "flux_sampling",
                         "flux_kind", "delta_semantics", "humidity_sampling")
                sym = _transport_normalize_symbol(val)
                sym === :unknown && push!(missing_or_unknown, "$f (:unknown)")
            end
        end
    end

    if !isempty(missing_or_unknown)
        msg = "Transport-binary contract violation — the following fields are missing " *
              "or unknown in the header:\n  " *
              join(missing_or_unknown, "\n  ") *
              "\nThis binary was produced by a preprocessor that does not declare the " *
              "runtime forcing contract. Regenerate via the current preprocessor " *
              "(scripts/preprocessing/preprocess_transport_binary.jl)."
        throw(ArgumentError(msg))
    end

    # All fields present — validate ranges via a roundtrip construction.
    # This catches e.g. an unknown value for `flux_sampling` that slipped in.
    try
        TransportBinaryContract(
            source_flux_sampling = _transport_normalize_symbol(header["source_flux_sampling"]),
            air_mass_sampling    = _transport_normalize_symbol(header["air_mass_sampling"]),
            flux_sampling        = _transport_normalize_symbol(header["flux_sampling"]),
            flux_kind            = _transport_normalize_symbol(header["flux_kind"]),
            delta_semantics      = _transport_normalize_symbol(header["delta_semantics"]),
            humidity_sampling    = _transport_normalize_symbol(header["humidity_sampling"]),
            poisson_balance_target_scale = Float64(header["poisson_balance_target_scale"]),
            poisson_balance_target_semantics = String(header["poisson_balance_target_semantics"]),
        )
    catch e
        rethrow(e)
    end

    nwindow = try
        Int(header["nwindow"])
    catch
        throw(ArgumentError("Transport-binary contract violation — nwindow must be an integer"))
    end
    nwindow > 0 ||
        throw(ArgumentError("Transport-binary contract violation — nwindow must be positive"))
    steps_per_window = try
        Int(header["steps_per_window"])
    catch
        throw(ArgumentError("Transport-binary contract violation — steps_per_window must be an integer"))
    end
    steps_per_window > 0 ||
        throw(ArgumentError("Transport-binary contract violation — steps_per_window must be positive"))

    schedule = try
        Int.(collect(header["steps_per_window_by_window"]))
    catch
        throw(ArgumentError("Transport-binary contract violation — steps_per_window_by_window must be an integer vector"))
    end
    length(schedule) == nwindow ||
        throw(ArgumentError("Transport-binary contract violation — steps_per_window_by_window length $(length(schedule)) " *
                            "does not match nwindow=$(nwindow)"))
    all(>=(1), schedule) ||
        throw(ArgumentError("Transport-binary contract violation — steps_per_window_by_window must contain only positive integers"))
    steps_per_window == maximum(schedule) ||
        throw(ArgumentError("Transport-binary contract violation — steps_per_window=$(steps_per_window) must equal " *
                            "maximum(steps_per_window_by_window)=$(maximum(schedule))"))

    variable_steps = _has_variable_step_schedule(schedule)
    expected_time_step_schedule = variable_steps ? "per_window" : "constant"
    time_step_schedule = String(header["time_step_schedule"])
    time_step_schedule == expected_time_step_schedule ||
        throw(ArgumentError("Transport-binary contract violation — time_step_schedule=$(repr(time_step_schedule)) " *
                            "but schedule requires $(repr(expected_time_step_schedule))"))

    flux_kind = _transport_normalize_symbol(header["flux_kind"])
    full_window_flux = flux_kind === :full_window_mass_amount

    scalar_scale = Float64(header["poisson_balance_target_scale"])
    expected_scalar_scale = full_window_flux ? 1.0 : 1.0 / (2 * steps_per_window)
    isapprox(scalar_scale, expected_scalar_scale; atol=eps(Float64) * 8, rtol=0.0) ||
        throw(ArgumentError("Transport-binary contract violation — poisson_balance_target_scale=$(scalar_scale), " *
                            "expected $(expected_scalar_scale) for flux_kind=$(flux_kind)"))
    expected_semantics = if full_window_flux
        "forward_window_mass_difference"
    elseif variable_steps
        "forward_window_mass_difference / (2 * steps_per_window_by_window[win])"
    else
        "forward_window_mass_difference / (2 * steps_per_window)"
    end
    semantics = String(header["poisson_balance_target_semantics"])
    semantics == expected_semantics ||
        throw(ArgumentError("Transport-binary contract violation — poisson_balance_target_semantics=$(repr(semantics)), " *
                            "expected $(repr(expected_semantics))"))

    scale_schedule = try
        Float64.(collect(header["poisson_balance_target_scale_by_window"]))
    catch
        throw(ArgumentError("Transport-binary contract violation — poisson_balance_target_scale_by_window must be a numeric vector"))
    end
    length(scale_schedule) == nwindow ||
        throw(ArgumentError("Transport-binary contract violation — poisson_balance_target_scale_by_window length $(length(scale_schedule)) " *
                            "does not match nwindow=$(nwindow)"))
    for win in 1:nwindow
        expected = full_window_flux ? 1.0 : 1.0 / (2 * schedule[win])
        isapprox(scale_schedule[win], expected; atol=eps(Float64) * 8, rtol=0.0) ||
            throw(ArgumentError("Transport-binary contract violation — poisson_balance_target_scale_by_window[$win]=" *
                                "$(scale_schedule[win]), expected $(expected) for flux_kind=$(flux_kind)"))
    end
    _validate_transport_layout!(header)
    return nothing
end
