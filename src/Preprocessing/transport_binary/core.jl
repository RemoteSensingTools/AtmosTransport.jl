# Shared binary payload metadata, header construction, and low-level output helpers.

"""
    write_array!(io, arr) -> Int

Write a dense Julia array to the binary stream without intermediate copies and
return the number of bytes written.
"""
function write_array!(io::IO, arr::Array{FT}) where FT
    nb = sizeof(arr)
    GC.@preserve arr begin
        written = unsafe_write(io, pointer(arr), nb)
    end
    written == nb || error("Short write: expected $nb bytes, got $written")
    return written
end

# Full L137 payloads with TM5 + surface metadata exceed the original 16 KiB
# budget. Readers use the serialized `header_bytes` field, so enlarging this
# only affects newly written files.
const HEADER_SIZE = 32768

_settings_include_surface(settings) =
    hasproperty(settings, :include_surface) && Bool(settings.include_surface)

@inline function exact_steps_per_window(dt_met::Real, dt::Real)
    ratio = Float64(dt_met) / Float64(dt)
    steps = round(Int, ratio)
    isapprox(ratio, steps; atol=1e-12, rtol=0.0) ||
        error("met_interval / dt must be an integer for the current transport-binary contract; got dt_met=$(dt_met), dt=$(dt), ratio=$(ratio)")
    steps >= 1 || error("steps_per_window must be >= 1; got $(steps)")
    return steps
end

# The Poisson balance target is the forward-window mass difference divided
# by `2 × steps_per_window`. The factor of 2 comes from the Strang splitting:
# each full time step applies horizontal fluxes TWICE (once in the forward
# sweep X-Y-Z and once in the reverse Z-Y-X), so the per-application
# mass tendency is half the total tendency. This matches TM5 r1112's
# `poisson_balance_target_scale` in grid_type_ll.F90.
@inline poisson_balance_target_scale(steps_per_window::Integer, ::Type{FT}=Float64) where FT =
    FT(inv(2 * max(Int(steps_per_window), 1)))

"""
    script_provenance(; caller_file=nothing) -> NamedTuple

Collect reproducibility metadata. `caller_file` is the path of the CLI script
that invoked the preprocessing (set by the entry point, not the library).
"""
function script_provenance(; caller_file::Union{String, Nothing}=nothing)
    preprocess_src_dir = dirname(@__DIR__)
    script_path = caller_file !== nothing ? abspath(caller_file) : preprocess_src_dir
    script_mtime = isfile(script_path) ? mtime(script_path) : 0.0
    git_commit, state = source_revision()
    git_dirty = state != "clean"       # an unidentified tree never reuses a binary

    return (
        script_path = script_path,
        script_mtime = script_mtime,
        git_commit = git_commit,
        git_dirty = git_dirty,
        creation_time = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
    )
end

function _generation_value_signature(value::AbstractArray)
    isbitstype(eltype(value)) || return repr(value)
    bytes = reinterpret(UInt8, vec(Array(value)))
    return (size=collect(size(value)), eltype=string(eltype(value)),
            sha256=bytes2hex(SHA.sha256(bytes)))
end

_generation_value_signature(value::NamedTuple) =
    Dict(String(key) => _generation_value_signature(getfield(value, key))
         for key in keys(value))
_generation_value_signature(value) = repr(value)

function _generation_file_signature(path::AbstractString)
    absolute = abspath(path)
    isfile(absolute) || return (path=absolute, missing=true)
    info = stat(absolute)
    return (path=absolute, size=info.size, mtime=info.mtime)
end

function generation_fingerprint(; settings, spectral_resolution,
                                source_paths, next_day_hour0, provenance)
    payload = Dict{String, Any}(
        "settings" => repr(settings),
        "spectral_resolution" => Int(spectral_resolution),
        "sources" => [_generation_file_signature(path) for path in source_paths],
        "next_day_hour0" => _generation_value_signature(next_day_hour0),
        "code" => Dict(
            "script_path" => provenance.script_path,
            "script_mtime_unix" => provenance.script_mtime,
            "git_commit" => provenance.git_commit,
            "git_dirty" => provenance.git_dirty,
            # A dirty tree has no stable code revision. Deliberately make its
            # fingerprint invocation-specific so it can never reuse a binary
            # whose producing source cannot be identified exactly.
            "dirty_nonce" => provenance.git_dirty ? provenance.creation_time : "",
        ),
    )
    return bytes2hex(SHA.sha256(codeunits(JSON3.write(payload))))
end

"""
    expected_payload_sections(settings) -> Vector{String}

Canonical v4 LL payload-section list for the writer-side schema. Used
by both `build_v4_header` and the SKIP-vs-regen gate in `process_day`,
so the on-disk header and the existence check stay in lockstep.
"""
function expected_payload_sections(settings)
    sections = String["m", "am", "bm", "cm", "ps"]
    settings.include_qv && append!(sections, ["qv_start", "qv_end"])
    append!(sections, ["dam", "dbm", "dcm", "dm"])
    # TM5 sections follow the deltas, matching the
    # ordering in _transport_push_optional_sections!.
    settings.tm5_convection_enable && append!(sections, ["entu", "detu", "entd", "detd"])
    _settings_include_surface(settings) && append!(sections, ["pblh", "t2m", "ustar", "pbl_hflux"])
    return sections
end

"""
    build_v4_header(date, grid, vertical, settings, FT, counts, sizes, provenance)

Construct the fixed-size JSON header for one v4 output file, including payload
layout, vertical metadata, humidity timing semantics, and mass-fix provenance.
"""
function build_v4_header(date::Date,
                         grid::LatLonTargetGeometry,
                         vertical,
                         settings,
                         FT::Type,
                         counts,
                         sizes,
                         provenance)
    payload_sections = expected_payload_sections(settings)
    var_names = copy(payload_sections)
    include_surface = _settings_include_surface(settings)

    ncell = sizes.Nx * sizes.Ny
    nface_h = (sizes.Nx + 1) * sizes.Ny + sizes.Nx * (sizes.Ny + 1)

    # Declare the self-describing window-constant transport contract.
    contract = canonical_window_constant_contract(
        steps_per_window   = sizes.steps_per_met,
        humidity_sampling  = settings.include_qv ? :window_endpoints : :none,
        source_flux_sampling = :window_start_endpoint,
        include_flux_delta = true,
    )

    step_schedule = fill(Int(sizes.steps_per_met), Int(sizes.Nt))
    poisson_scale_schedule = [1.0 / (2 * s) for s in step_schedule]

    header = Dict{String, Any}(
        "magic" => "MFLX",
        "format_version" => TRANSPORT_BINARY_FORMAT_VERSION,
        "header_bytes" => HEADER_SIZE,
        "grid_type" => "latlon", "horizontal_topology" => "StructuredDirectional",
        "ncell" => ncell, "nface_h" => nface_h, "nlevel" => sizes.Nz, "nwindow" => sizes.Nt,
        "Nx" => sizes.Nx, "Ny" => sizes.Ny, "Nz" => sizes.Nz,
        "Nz_native" => sizes.Nz_native, "Nt" => sizes.Nt,
        "float_type" => string(FT), "float_bytes" => sizeof(FT),
        "mass_basis" => String(settings.mass_basis),
        "payload_sections" => payload_sections,
        "elems_per_window" => counts.elems_per_window,
        "n_geometry_elems" => 0,
        # Timing and sampling semantics are always explicit.
        "source_flux_sampling"            => String(contract.source_flux_sampling),
        "air_mass_sampling"               => String(contract.air_mass_sampling),
        "flux_sampling"                   => String(contract.flux_sampling),
        "flux_kind"                       => String(contract.flux_kind),
        "delta_semantics"                 => String(contract.delta_semantics),
        "humidity_sampling"               => String(contract.humidity_sampling),
        "window_bytes" => counts.bytes_per_window,
        "n_m" => counts.n_m, "n_am" => counts.n_am, "n_bm" => counts.n_bm,
        "n_cm" => counts.n_cm, "n_ps" => counts.n_ps,
        "n_qv_start" => counts.n_qv_start, "n_qv_end" => counts.n_qv_end,
        "n_cmfmc" => 0,
        "n_entu" => counts.n_entu, "n_detu" => counts.n_detu,
        "n_entd" => counts.n_entd, "n_detd" => counts.n_detd,
        "n_pblh" => counts.n_pblh, "n_t2m" => counts.n_t2m,
        "n_ustar" => counts.n_ustar, "n_pbl_hflux" => counts.n_hflux,
        "n_temperature" => 0,
        "n_dam" => counts.n_dam, "n_dbm" => counts.n_dbm, "n_dcm" => counts.n_dcm, "n_dm" => counts.n_dm,
        "include_flux_delta" => true,
        "include_qv_endpoints" => settings.include_qv,
        "include_cmfmc" => false,
        "include_tm5conv" => settings.tm5_convection_enable,
        "include_surface" => include_surface,
        "surface_payload" => include_surface ? "pbl_raw_v2" : "none",
        "include_temperature" => false,
        "dt_met_seconds" => settings.met_interval,
        "dt_seconds" => settings.dt,
        "half_dt_seconds" => settings.half_dt,
        "steps_per_window" => sizes.steps_per_met,
        "steps_per_window_by_window" => step_schedule,
        "time_step_schedule" => "constant",
        "steps_per_met_window" => sizes.steps_per_met,
        "level_top" => first(vertical.level_range), "level_bot" => last(vertical.level_range),
        "A_ifc" => Float64.(vertical.merged_vc.A), "B_ifc" => Float64.(vertical.merged_vc.B),
        "merge_map" => vertical.merge_map, "merge_min_thickness_Pa" => settings.min_dp,
        "vertical_mapping_method" => String(vertical_mapping_method(vertical)),
        "target_vertical_name" => hasproperty(vertical, :target_vertical_name) ?
            vertical.target_vertical_name : "",
        "target_coefficients" => hasproperty(vertical, :target_coefficients) ?
            vertical.target_coefficients : "",
        "var_names" => var_names,
        "date" => Dates.format(date, "yyyy-mm-dd"),
        "spectral_half_dt_seconds" => settings.half_dt,
        # Poisson fields driven by the same TransportBinaryContract above — single source of truth.
        "poisson_balance_target_scale"     => contract.poisson_balance_target_scale,
        "poisson_balance_target_semantics" => contract.poisson_balance_target_semantics,
        "poisson_balance_target_scale_by_window" => poisson_scale_schedule,
        "horizontal_balance" => balance_tag(effective_horizontal_balance(get(settings, :horizontal_balance, nothing), ColumnBalance())),
        "script_path" => provenance.script_path,
        "script_mtime_unix" => provenance.script_mtime,
        "git_commit" => provenance.git_commit,
        "git_dirty" => provenance.git_dirty,
        "creation_time" => provenance.creation_time,
        "mass_fix_enabled" => settings.mass_fix_enable,
        "mass_fix_target_ps_dry_pa" => settings.target_ps_dry_pa,
        "mass_fix_qv_global_climatology" => settings.qv_global_climatology,
        "mass_fix_qv_mode" => settings.include_qv || settings.mass_basis == :dry ?
            "native_hourly_qv" : "global_qv_climatology",
        "ps_offsets_pa_per_window" => zeros(Float64, sizes.Nt),
        "ps_offsets_next_day_hour0_pa" => 0.0,
        "qv_source_type" => isempty(settings.thermo_dir) ? "none" : "era5_thermo_netcdf",
        "qv_source_directory" => settings.thermo_dir,
        "qv_variable_name" => settings.include_qv || settings.mass_basis == :dry ? "q" : "",
        "qv_units" => settings.include_qv || settings.mass_basis == :dry ? "kg kg-1" : "",
        "qv_time_alignment" => settings.include_qv || settings.mass_basis == :dry ?
            "qv_start uses same-day thermo time index i; qv_end uses time index i+1, with next-day 00 UTC for the final window" : "",
        "qv_next_day_alignment" => settings.include_qv || settings.mass_basis == :dry ?
            "final-window qv_end uses next-day thermo file time index 1 (00 UTC) when available" : "",
        "qv_latitude_handling" => settings.include_qv || settings.mass_basis == :dry ?
            "flip latitude to south-to-north if thermo file is stored north-to-south" : "",
    )

    merge!(header, target_header_metadata(grid))
    validate_transport_contract!(header)
    return header
end

"""
    resolve_qv_requirements(date, settings) -> (; needs_qv, thermo_path)

Determine whether the current run needs humidity data and resolve the matching
daily thermo file path.

Humidity is required whenever `output.include_qv=true` or `mass_basis=:dry`.
The time convention is:
- window `i` uses the same day's thermo time index `i`
- the next-day delta path uses the next day's time index `1`
"""
function resolve_qv_requirements(date::Date, settings)
    needs_qv_for_dry = settings.mass_basis == :dry
    needs_qv = settings.include_qv || needs_qv_for_dry
    needs_qv && isempty(settings.thermo_dir) &&
        error("mass_basis=:dry requires [input].thermo_dir")

    thermo_path = if needs_qv
        path = joinpath(settings.thermo_dir, "era5_thermo_ml_$(Dates.format(date, "yyyymmdd")).nc")
        isfile(path) || error("Thermo file not found: $path")
        path
    else
        ""
    end

    if needs_qv
        @info "  Humidity source: ERA5 thermo NetCDF `q`"
        @info "    Window i -> same-day thermo time index i (00..23 UTC)"
        @info "    Next-day delta -> next-day thermo time index 1 (00 UTC)"
    end

    return (; needs_qv, thermo_path)
end

"""
    window_element_counts(grid, Nz; include_qv=false) -> NamedTuple

Return the per-window element counts for every payload section in the output
binary.
"""
function window_element_counts(grid::LatLonTargetGeometry, Nz::Int;
                                include_qv::Bool=false,
                                tm5_convection::Bool=false,
                                include_surface::Bool=false)
    Nx = nlon(grid)
    Ny = nlat(grid)

    n_m = Int64(Nx) * Ny * Nz
    n_am = Int64(Nx + 1) * Ny * Nz
    n_bm = Int64(Nx) * (Ny + 1) * Nz
    n_cm = Int64(Nx) * Ny * (Nz + 1)
    n_ps = Int64(Nx) * Ny
    n_qv_start = include_qv ? n_m : Int64(0)
    n_qv_end = include_qv ? n_m : Int64(0)
    n_tm5 = tm5_convection ? n_m : Int64(0)  # each of entu/detu/entd/detd is (Nx, Ny, Nz)
    n_sfc = include_surface ? n_ps : Int64(0)

    return (
        n_m = n_m,
        n_am = n_am,
        n_bm = n_bm,
        n_cm = n_cm,
        n_ps = n_ps,
        n_qv_start = n_qv_start,
        n_qv_end = n_qv_end,
        n_dam = n_am,
        n_dbm = n_bm,
        n_dcm = n_cm,
        n_dm = n_m,
        n_entu = n_tm5,
        n_detu = n_tm5,
        n_entd = n_tm5,
        n_detd = n_tm5,
        n_pblh = n_sfc,
        n_t2m = n_sfc,
        n_ustar = n_sfc,
        n_hflux = n_sfc,
        elems_per_window = n_m + n_am + n_bm + n_cm + n_ps +
                           n_qv_start + n_qv_end +
                           n_am + n_bm + n_cm + n_m +
                           4 * n_tm5 + 4 * n_sfc,
    )
end

"""
    window_byte_sizes(counts, FT, Nt) -> (; bytes_per_window, total_bytes)

Translate payload element counts into byte counts for one window and the full
daily file.
"""
function window_byte_sizes(counts, ::Type{FT}, Nt::Int) where FT
    bytes_per_window = counts.elems_per_window * sizeof(FT)
    total_bytes = Int64(HEADER_SIZE) + Int64(bytes_per_window) * Nt
    return (; bytes_per_window, total_bytes)
end

"""
    output_binary_path(date, out_dir, min_dp, FT) -> String

Build the canonical output filename for one daily v4 binary.
"""
function output_binary_path(date::Date, out_dir::String, min_dp::Float64, ::Type{FT}) where FT
    dp_tag = @sprintf("merged%dPa", round(Int, min_dp))
    ft_tag = FT == Float64 ? "float64" : "float32"
    date_str = Dates.format(date, "yyyymmdd")
    return joinpath(out_dir, "era5_transport_$(date_str)_$(dp_tag)_$(ft_tag).bin")
end

# Peek the JSON header at byte 0 (null-terminated; matches both
# `_parse_transport_header` and `TransportBinaryReader`). Returns
# `nothing` on any failure so callers can fall back gracefully.
function _peek_transport_header(bin_path::AbstractString)
    isfile(bin_path) || return nothing
    raw = open(bin_path, "r") do io
        read(io, min(filesize(bin_path), HEADER_SIZE))
    end
    json_end = findfirst(==(0x00), raw)
    json_end === nothing && return nothing
    hdr = try
        JSON3.read(String(raw[1:json_end - 1]))
    catch
        return nothing
    end
    return Dict{String, Any}(String(k) => v for (k, v) in pairs(hdr))
end

function _peek_payload_sections(bin_path::AbstractString)
    hdr = _peek_transport_header(bin_path)
    hdr === nothing && return nothing
    haskey(hdr, "payload_sections") || return nothing
    return Set(String.(hdr["payload_sections"]))
end

const _OUTPUT_REUSE_CONTRACT_KEYS = (
    "format_version", "float_type", "float_bytes", "mass_basis",
    "grid_type", "horizontal_topology", "planet_radius_m", "Nx", "Ny", "nlevel", "nwindow",
    "payload_sections", "elems_per_window", "A_ifc", "B_ifc", "merge_map",
    "merge_min_thickness_Pa", "vertical_mapping_method", "target_vertical_name",
    "target_coefficients", "dt_met_seconds", "dt_seconds", "half_dt_seconds",
    "steps_per_window", "source_flux_sampling", "air_mass_sampling",
    "flux_sampling", "flux_kind", "delta_semantics", "humidity_sampling",
    "mass_fix_enabled", "mass_fix_target_ps_dry_pa", "mass_fix_qv_global_climatology",
    "qv_source_type", "qv_source_directory", "include_qv_endpoints",
    "include_tm5conv", "include_surface", "date", "generation_fingerprint",
)

"""
    existing_output_schema_matches(bin_path, total_bytes, expected_sections,
                                   expected_header=nothing) -> (skip, reason)

SKIP-or-regen decision used by `process_day` writers. Returns
`(true, "")` when the file exists, byte size matches, AND the on-disk
`payload_sections` set matches the caller's `expected_sections`. When an
`expected_header` is supplied, the generation-defining metadata must match as
well; provenance and run-produced diagnostics are intentionally excluded.
Returns `(false, reason)` otherwise; `reason` is a human-readable diff
(`"size 41g vs 23g"` or `"schema added :pbl_hflux removed :hflux"`)
the caller should `@info` so the user knows *why* the binary is being
rebuilt rather than skipped.
"""
function existing_output_schema_matches(bin_path::AbstractString,
                                        total_bytes::Integer,
                                        expected_sections,
                                        expected_header::Union{Nothing, AbstractDict}=nothing)
    isfile(bin_path) || return (false, "no existing file")
    fs = filesize(bin_path)
    fs == total_bytes ||
        return (false, @sprintf("size %.2f GB vs expected %.2f GB", fs / 1e9, total_bytes / 1e9))

    on_disk = _peek_payload_sections(bin_path)
    on_disk === nothing && return (false, "header unparseable (likely pre-v4)")
    expected = Set(String.(expected_sections))
    if on_disk == expected
        if expected_header !== nothing
            disk_header = _peek_transport_header(bin_path)
            disk_header === nothing && return (false, "header unparseable")
            changed = String[]
            for key in _OUTPUT_REUSE_CONTRACT_KEYS
                if !haskey(disk_header, key) || !haskey(expected_header, key) ||
                   JSON3.write(disk_header[key]) != JSON3.write(expected_header[key])
                    push!(changed, key)
                end
            end
            # Optional key: absent means the write-time replay gate ran.
            get(disk_header, "write_replay_check", true) == get(expected_header, "write_replay_check", true) ||
                push!(changed, "write_replay_check")
            isempty(changed) || return (
                false, "generation contract changed: " * join(changed, ","))
        end
        return (true, "")
    end
    added   = sort!(collect(setdiff(expected, on_disk)))
    removed = sort!(collect(setdiff(on_disk, expected)))
    parts = String[]
    isempty(added)   || push!(parts, "added " * join(added, ","))
    isempty(removed) || push!(parts, "removed " * join(removed, ","))
    return (false, "schema mismatch: " * join(parts, "; "))
end
