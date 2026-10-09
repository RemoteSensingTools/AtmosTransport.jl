# Typed geometry metadata, common header schema, and version-4 parsing.

const _PBL_SURFACE_PAYLOAD_SECTIONS = (:pblh, :ustar, :pbl_hflux, :t2m)
const _PBL_SURFACE_FIELD_NAMES = (:pblh, :ustar, :hflux, :t2m)
const _GCHP_VDIFF_PAYLOAD_SECTIONS = (:vdiff_u, :vdiff_v, :vdiff_t, :vdiff_qv)
const _GCHP_VDIFF_FIELD_NAMES = (:u, :v, :t, :qv)
# Optional 2-D cubed-sphere sections: surface latent heat flux for the GCHP
# non-local PBL scheme, and the convective cloud-base layer (top-down index).
const _CS_OPTIONAL_2D_SECTIONS = (:pbl_eflux, :cmfmc_cloud_base)

# Sections stored as one (Nc, Nc) field per panel.
@inline _is_cs_2d_section(section::Symbol) =
    section === :ps || _is_pbl_surface_payload_section(section) || section in _CS_OPTIONAL_2D_SECTIONS
const TRANSPORT_BINARY_FORMAT_VERSION = 4

@inline _is_pbl_surface_payload_section(section::Symbol) =
    section in _PBL_SURFACE_PAYLOAD_SECTIONS
@inline _pbl_surface_field_name(section::Symbol) =
    section === :pbl_hflux ? :hflux : section
@inline _is_gchp_vdiff_payload_section(section::Symbol) =
    section in _GCHP_VDIFF_PAYLOAD_SECTIONS
@inline function _gchp_vdiff_field_name(section::Symbol)
    section === :vdiff_u  && return :u
    section === :vdiff_v  && return :v
    section === :vdiff_t  && return :t
    section === :vdiff_qv && return :qv
    throw(ArgumentError("not a GCHP VDIFF payload section: $(section)"))
end

"""
    AbstractTransportBinaryGeometry

Geometry metadata embedded in a version-4 transport binary. Concrete geometry
types carry only the fields meaningful for their topology; reader and grid
construction code dispatches on this type instead of branching on string tags.
"""
abstract type AbstractTransportBinaryGeometry end

"""Geometry metadata for a regular longitude-latitude binary."""
struct LatLonBinaryGeometry <: AbstractTransportBinaryGeometry
    Nx                 :: Int
    Ny                 :: Int
    longitudes         :: Vector{Float64}
    latitudes          :: Vector{Float64}
    longitude_interval :: Vector{Float64}
    latitude_interval  :: Vector{Float64}
end

"""Geometry metadata for a face-indexed reduced-Gaussian binary."""
struct ReducedGaussianBinaryGeometry <: AbstractTransportBinaryGeometry
    latitudes     :: Vector{Float64}
    nlon_per_ring :: Vector{Int}
end

"""Geometry metadata for a six-panel cubed-sphere binary."""
struct CubedSphereBinaryGeometry <: AbstractTransportBinaryGeometry
    Nc                   :: Int
    npanel               :: Int
    panel_convention     :: Symbol
    definition           :: Symbol
    coordinate_law       :: Symbol
    center_law           :: Symbol
    longitude_offset_deg :: Float64
end

grid_type(::LatLonBinaryGeometry) = :latlon
grid_type(::ReducedGaussianBinaryGeometry) = :reduced_gaussian
grid_type(::CubedSphereBinaryGeometry) = :cubed_sphere

horizontal_topology(::LatLonBinaryGeometry) = :structureddirectional
horizontal_topology(::ReducedGaussianBinaryGeometry) = :faceindexed
horizontal_topology(::CubedSphereBinaryGeometry) = :structureddirectional

"""
    TransportBinaryHeader{G<:AbstractTransportBinaryGeometry}

Validated metadata for the current transport-binary format. `geometry` selects
the horizontal topology through dispatch, while the remaining fields describe
the common timing, vertical coordinate, mass basis, and payload contract.

`planet_radius_m` is the radius of the sphere on which the preprocessor computed
cell areas, and so the air masses from pressure thickness (`m = Δp A / g`); the
runtime builds its mesh with it. Binaries written before the key existed read as
`EARTH_RADIUS`, the radius the runtime used for them.

`A_ifc` and `B_ifc` define interface pressure as
`p_half[k] = A_ifc[k] + B_ifc[k] * surface_pressure`, with `k = 1` at the
top of atmosphere. `flux_kind` distinguishes mass per transport substep from a
full-window mass amount.
"""
struct TransportBinaryHeader{G <: AbstractTransportBinaryGeometry}
    format_version       :: Int
    header_bytes         :: Int
    on_disk_float_type   :: Symbol      # :Float32 or :Float64
    float_bytes          :: Int         # 4 or 8
    geometry             :: G
    planet_radius_m      :: Float64     # radius of the preprocessing mesh [m]
    ncell                :: Int         # total horizontal cells
    nface_h              :: Int         # total horizontal faces
    nlevel               :: Int         # vertical levels (k=1 TOA, k=nlevel surface)
    nwindow              :: Int         # windows per day (typically 24)
    dt_met_seconds       :: Float64     # met-data window interval [s] (3600 for hourly)
    half_dt_seconds      :: Float64     # half-step time [s] for flux scaling
    steps_per_window     :: Int         # scalar compatibility summary (constant value or max schedule)
    steps_per_window_by_window :: Vector{Int}
    source_flux_sampling :: Symbol
    air_mass_sampling    :: Symbol
    flux_sampling        :: Symbol      # :window_constant
    flux_kind            :: Symbol      # :substep_mass_amount or :full_window_mass_amount
    humidity_sampling    :: Symbol
    delta_semantics      :: Symbol
    poisson_balance_target_scale :: Float64
    poisson_balance_target_semantics :: String
    poisson_balance_target_scale_by_window :: Vector{Float64}
    A_ifc                :: Vector{Float64}  # hybrid A [Pa], length nlevel+1
    B_ifc                :: Vector{Float64}  # hybrid B [1],  length nlevel+1
    mass_basis           :: Symbol           # :moist or :dry
    payload_sections     :: Vector{Symbol}
    n_geometry_elems     :: Int
    elems_per_window     :: Int
    raw_header           :: Dict{String, Any}
end

binary_geometry(h::TransportBinaryHeader) = h.geometry
grid_type(h::TransportBinaryHeader) = grid_type(h.geometry)
horizontal_topology(h::TransportBinaryHeader) = horizontal_topology(h.geometry)

function _parse_steps_per_window_schedule(hdr, nwindow::Integer,
                                          steps_per_window::Integer)
    haskey(hdr, :steps_per_window_by_window) ||
        throw(ArgumentError("format_version=$(TRANSPORT_BINARY_FORMAT_VERSION) requires " *
                            "`steps_per_window_by_window`; regenerate this binary"))
    schedule = Int.(collect(hdr.steps_per_window_by_window))
    length(schedule) == Int(nwindow) ||
        throw(ArgumentError("steps_per_window_by_window length $(length(schedule)) " *
                            "does not match nwindow=$(nwindow)"))
    all(>=(1), schedule) ||
        throw(ArgumentError("steps_per_window_by_window must contain only positive integers; got $(schedule)"))
    Int(steps_per_window) == maximum(schedule) ||
        throw(ArgumentError("steps_per_window must be maximum(steps_per_window_by_window); " *
                            "got steps_per_window=$(steps_per_window), schedule=$(schedule)"))
    return schedule
end

function _parse_poisson_scale_schedule(hdr, nwindow::Integer,
                                       scalar_scale::Real)
    haskey(hdr, :poisson_balance_target_scale_by_window) ||
        throw(ArgumentError("format_version=$(TRANSPORT_BINARY_FORMAT_VERSION) requires " *
                            "`poisson_balance_target_scale_by_window`; regenerate this binary"))
    scales = Float64.(collect(hdr.poisson_balance_target_scale_by_window))
    length(scales) == Int(nwindow) ||
        throw(ArgumentError("poisson_balance_target_scale_by_window length $(length(scales)) " *
                            "does not match nwindow=$(nwindow)"))
    all(s -> isfinite(s) && s > 0, scales) ||
        throw(ArgumentError("poisson_balance_target_scale_by_window must contain only finite positive values"))
    isfinite(Float64(scalar_scale)) && Float64(scalar_scale) > 0 ||
        throw(ArgumentError("poisson_balance_target_scale must be finite and positive"))
    return scales
end

@inline _has_variable_step_schedule(schedule::AbstractVector{<:Integer}) =
    !isempty(schedule) && any(!=(first(schedule)), schedule)

function _steps_per_window_summary(steps::Integer,
                                   schedule::AbstractVector{<:Integer})
    if _has_variable_step_schedule(schedule)
        return string(Int(steps), " max (per-window ",
                      minimum(schedule), ":", maximum(schedule), ")")
    end
    return string(Int(steps))
end

@inline _transport_geometry_summary(g::LatLonBinaryGeometry, h::TransportBinaryHeader) =
    string(g.Nx, "×", g.Ny, " structured cells, ", h.nlevel, " levels")
@inline _transport_geometry_summary(::ReducedGaussianBinaryGeometry, h::TransportBinaryHeader) =
    string(h.ncell, " cells, ", h.nface_h, " faces, ", h.nlevel, " levels")
@inline _transport_geometry_summary(g::CubedSphereBinaryGeometry, h::TransportBinaryHeader) =
    string("C", g.Nc, ", panels=", g.npanel, ", ", h.nlevel, " levels")
@inline _transport_geometry_summary(h::TransportBinaryHeader) =
    string(_transport_geometry_summary(h.geometry, h), ", radius ", h.planet_radius_m, " m")

@inline function _transport_qv_summary(h::TransportBinaryHeader)
    return (:qv_start in h.payload_sections && :qv_end in h.payload_sections) ?
           "qv_start/qv_end" : "none"
end

@inline function _transport_semantics_summary(h::TransportBinaryHeader)
    return string(
        "air_mass=", h.air_mass_sampling,
        ", flux=", h.flux_sampling, "/", h.flux_kind,
        ", humidity=", h.humidity_sampling,
        ", delta=", h.delta_semantics,
        h.source_flux_sampling === :unknown ? "" : string(", source_flux=", h.source_flux_sampling)
    )
end

function Base.summary(h::TransportBinaryHeader)
    return string(
        "TransportBinaryHeader(v", h.format_version, ", ",
        grid_type(h), "/", horizontal_topology(h), ", ",
        h.nwindow, " windows)"
    )
end

function Base.show(io::IO, h::TransportBinaryHeader)
    print(io, summary(h), "\n",
          "├── geometry:      ", _transport_geometry_summary(h), "\n",
          "├── storage:       ", h.on_disk_float_type, " on disk, basis=", h.mass_basis, "\n",
          "├── timing:        dt=", h.dt_met_seconds, " s, steps/window=",
              _steps_per_window_summary(h.steps_per_window, h.steps_per_window_by_window), "\n",
          "├── payload:       ", join(String.(h.payload_sections), ", "), "\n",
          "├── humidity:      ", _transport_qv_summary(h), "\n",
          "├── semantics:     ", _transport_semantics_summary(h), "\n",
          "├── poisson:       ", isnan(h.poisson_balance_target_scale) ? "unspecified" :
                               string("scale=", h.poisson_balance_target_scale, ", ", h.poisson_balance_target_semantics), "\n",
          "└── header bytes:  ", h.header_bytes)
end

_transport_is_structured(h::TransportBinaryHeader) = h.geometry isa LatLonBinaryGeometry
_transport_is_faceindexed(h::TransportBinaryHeader) = h.geometry isa ReducedGaussianBinaryGeometry
_transport_is_cubed_sphere(h::TransportBinaryHeader) = h.geometry isa CubedSphereBinaryGeometry

function _transport_disk_float_type(sym::Symbol)
    sym === :Float64 ? Float64 : Float32
end

function _transport_parse_on_disk_float_type(hdr)
    haskey(hdr, :float_type) ||
        throw(ArgumentError("transport binary header is missing float_type"))
    ft_str = string(hdr.float_type)
    if ft_str == "Float64"
        return :Float64, 8
    elseif ft_str == "Float32"
        return :Float32, 4
    end
    throw(ArgumentError("unsupported transport binary float_type=$(repr(ft_str)); expected Float32 or Float64"))
end

function _transport_parse_mass_basis(hdr)
    haskey(hdr, :mass_basis) ||
        throw(ArgumentError("transport binary header is missing mass_basis; regenerate the binary"))
    basis_str = lowercase(string(hdr.mass_basis))
    basis_str in ("dry", "moist") || throw(ArgumentError(
        "unsupported transport binary mass_basis=$(repr(basis_str)); expected dry or moist"))
    return Symbol(basis_str)
end

@inline function _transport_normalize_symbol(value)
    return Symbol(replace(lowercase(String(value)), '-' => '_', ' ' => '_'))
end

const _TRANSPORT_ALLOWED_SOURCE_FLUX_SAMPLINGS = (
    :window_start_endpoint,
    :window_end_endpoint,
    :window_mean,
    :interval_integrated,
)

@inline function _transport_validate_source_flux_sampling(value)
    sym = _transport_normalize_symbol(value)
    sym in _TRANSPORT_ALLOWED_SOURCE_FLUX_SAMPLINGS || throw(ArgumentError(
        "unsupported source_flux_sampling=$(value); supported values are " *
        join(string.(Tuple(_TRANSPORT_ALLOWED_SOURCE_FLUX_SAMPLINGS)), ", ")
    ))
    return sym
end

@inline function _transport_parse_symbol_key(hdr, key::Symbol, default::Symbol)
    return _transport_normalize_symbol(get(hdr, key, String(default)))
end

@inline function _transport_default_humidity_sampling(payload_sections::AbstractVector{Symbol})
    if (:qv_start in payload_sections) || (:qv_end in payload_sections)
        return :window_endpoints
    else
        return :none
    end
end

@inline function _transport_default_delta_semantics(payload_sections::AbstractVector{Symbol})
    return any(section in (:dam, :dbm, :dcm, :dm, :dhflux) for section in payload_sections) ?
           :forward_window_endpoint_difference : :none
end

# Accepted timing and sampling semantics. Writers declare these explicitly;
# readers reject missing or unsupported values.

const _TRANSPORT_ALLOWED_AIR_MASS_SAMPLINGS = (:window_start_endpoint,)
const _TRANSPORT_ALLOWED_FLUX_SAMPLINGS     = (:window_start_endpoint, :window_constant, :window_mean)
const _TRANSPORT_ALLOWED_FLUX_KINDS         = (:substep_mass_amount, :full_window_mass_amount)
const _TRANSPORT_ALLOWED_DELTA_SEMANTICS    = (:forward_window_endpoint_difference, :none)
const _TRANSPORT_ALLOWED_HUMIDITY_SAMPLINGS = (:window_endpoints, :none)

_transport_parse_grid_type(hdr) = Symbol(lowercase(string(hdr.grid_type)))
_transport_parse_topology(hdr) = Symbol(lowercase(string(hdr.horizontal_topology)))

function _transport_parse_sections(hdr)
    sections = haskey(hdr, :payload_sections) ? collect(hdr.payload_sections) : ["m", "hflux", "cm", "ps"]
    return Symbol.(lowercase.(String.(sections)))
end

function _transport_header_axis(hdr, n::Int, key::Symbol)
    haskey(hdr, key) && return Float64.(collect(getproperty(hdr, key)))
    n == 0 && return Float64[]
    error("Transport binary header missing $(key)")
end

function _transport_header_interval(hdr, key::Symbol)
    return haskey(hdr, key) ? Float64.(collect(getproperty(hdr, key))) : Float64[]
end

function _parse_transport_geometry(hdr, grid::Symbol, topology::Symbol)
    if grid === :latlon && topology === :structureddirectional
        Nx, Ny = Int(hdr.Nx), Int(hdr.Ny)
        return LatLonBinaryGeometry(
            Nx,
            Ny,
            _transport_header_axis(hdr, Nx, :lons),
            _transport_header_axis(hdr, Ny, :lats),
            _transport_header_interval(hdr, :longitude_interval),
            _transport_header_interval(hdr, :latitude_interval),
        )
    elseif grid === :reduced_gaussian && topology === :faceindexed
        return ReducedGaussianBinaryGeometry(
            Float64.(collect(hdr.latitudes)),
            Int.(collect(hdr.nlon_per_ring)),
        )
    elseif grid === :cubed_sphere && topology === :structureddirectional
        return CubedSphereBinaryGeometry(
            Int(hdr.Nc),
            Int(hdr.npanel),
            Symbol(lowercase(String(hdr.panel_convention))),
            Symbol(lowercase(String(hdr.cs_definition))),
            Symbol(lowercase(String(hdr.cs_coordinate_law))),
            Symbol(lowercase(String(hdr.cs_center_law))),
            Float64(hdr.longitude_offset_deg),
        )
    end
    throw(ArgumentError("unsupported transport-binary geometry $(grid)/$(topology)"))
end

function _parse_transport_header(raw_bytes::Vector{UInt8})
    json_end = something(findfirst(==(0x00), raw_bytes), length(raw_bytes) + 1) - 1
    hdr = JSON3.read(String(raw_bytes[1:json_end]))

    haskey(hdr, :format_version) ||
        error("TransportBinaryReader requires the topology-generic binary family header (`format_version` missing)")

    # No silent defaults for missing contract fields.
    # `validate_transport_contract!` (called by `TransportBinaryReader`
    # before we get here) has already verified the 8 fields are present —
    # before parsing the typed header.
    format_version = Int(hdr.format_version)
    format_version == TRANSPORT_BINARY_FORMAT_VERSION ||
        throw(ArgumentError("Obsolete transport binary format_version=$(format_version); " *
                            "current runtime requires format_version=$(TRANSPORT_BINARY_FORMAT_VERSION). Regenerate."))
    header_bytes = Int(get(hdr, :header_bytes, 16384))
    disk_ft, float_bytes = _transport_parse_on_disk_float_type(hdr)
    grid_type = _transport_parse_grid_type(hdr)
    topology = _transport_parse_topology(hdr)
    geometry = _parse_transport_geometry(hdr, grid_type, topology)
    planet_radius_m = Float64(get(hdr, :planet_radius_m, EARTH_RADIUS))
    isfinite(planet_radius_m) && planet_radius_m > 0 ||
        throw(ArgumentError("planet_radius_m must be finite and positive; got $(planet_radius_m)"))
    ncell = Int(hdr.ncell)
    nface_h = Int(hdr.nface_h)
    nlevel = Int(hdr.nlevel)
    nwindow = Int(hdr.nwindow)
    A_ifc = Float64.(collect(hdr.A_ifc))
    B_ifc = Float64.(collect(hdr.B_ifc))
    payload_sections = _transport_parse_sections(hdr)
    source_flux_sampling = _transport_parse_symbol_key(hdr, :source_flux_sampling, :unknown)
    air_mass_sampling    = _transport_parse_symbol_key(hdr, :air_mass_sampling,    :unknown)
    flux_sampling        = _transport_parse_symbol_key(hdr, :flux_sampling,        :unknown)
    flux_kind            = _transport_parse_symbol_key(hdr, :flux_kind,            :unknown)
    humidity_sampling    = _transport_parse_symbol_key(hdr, :humidity_sampling,    :unknown)
    delta_semantics      = _transport_parse_symbol_key(hdr, :delta_semantics,      :unknown)
    poisson_balance_target_scale = haskey(hdr, :poisson_balance_target_scale) ?
                                   Float64(hdr.poisson_balance_target_scale) : NaN
    poisson_balance_target_semantics = haskey(hdr, :poisson_balance_target_semantics) ?
                                       String(hdr.poisson_balance_target_semantics) : ""
    steps_per_window = Int(hdr.steps_per_window)
    steps_schedule = _parse_steps_per_window_schedule(hdr, nwindow, steps_per_window)
    poisson_scale_schedule = _parse_poisson_scale_schedule(
        hdr, nwindow, poisson_balance_target_scale)
    n_geometry_elems = Int(get(hdr, :n_geometry_elems, 0))
    elems_per_window = Int(hdr.elems_per_window)
    raw_header = Dict{String, Any}(String(k) => v for (k, v) in pairs(hdr))

    return TransportBinaryHeader(
        format_version,
        header_bytes,
        disk_ft,
        float_bytes,
        geometry,
        planet_radius_m,
        ncell,
        nface_h,
        nlevel,
        nwindow,
        Float64(hdr.dt_met_seconds),
        Float64(hdr.half_dt_seconds),
        steps_per_window,
        steps_schedule,
        source_flux_sampling,
        air_mass_sampling,
        flux_sampling,
        flux_kind,
        humidity_sampling,
        delta_semantics,
        poisson_balance_target_scale,
        poisson_balance_target_semantics,
        poisson_scale_schedule,
        A_ifc,
        B_ifc,
        _transport_parse_mass_basis(hdr),
        payload_sections,
        n_geometry_elems,
        elems_per_window,
        raw_header,
    )
end

function _transport_basis_symbol(sym::Symbol)
    basis = Symbol(lowercase(String(sym)))
    basis in (:dry, :moist) ||
        throw(ArgumentError("mass_basis must be :dry or :moist; got $(repr(sym))"))
    return basis
end
_transport_basis_symbol(::DryBasis) = :dry
_transport_basis_symbol(::MoistBasis) = :moist

function _transport_common_header(grid_type::String,
                                  horizontal_topology::String,
                                  ncell::Int,
                                  nface_h::Int,
                                  nlevel::Int,
                                  nwindow::Int,
                                  vc,
                                  payload_sections::Vector{Symbol},
                                  elems_per_window::Int;
                                  FT::Type{<:AbstractFloat},
                                  planet_radius::Real,
                                  header_bytes::Int,
                                  dt_met_seconds::Real,
                                  half_dt_seconds::Real,
                                  steps_per_window::Integer,
                                  mass_basis::Symbol,
                                  source_flux_sampling::Symbol,
                                  air_mass_sampling::Symbol,
                                  flux_sampling::Symbol,
                                  flux_kind::Symbol,
                                  humidity_sampling::Symbol,
                                  delta_semantics::Symbol)
    n_qv_start = (:qv_start in payload_sections) ? ncell * nlevel : 0
    n_qv_end = (:qv_end in payload_sections) ? ncell * nlevel : 0
    n_surface = (:pblh in payload_sections) ? ncell : 0
    n_tm5 = (:entu in payload_sections) ? ncell * nlevel : 0
    n_vdiff = (:vdiff_u in payload_sections) ? ncell * nlevel : 0
    humidity_sampling = humidity_sampling === :auto ? _transport_default_humidity_sampling(payload_sections) : _transport_normalize_symbol(humidity_sampling)
    delta_semantics = delta_semantics === :auto ? _transport_default_delta_semantics(payload_sections) : _transport_normalize_symbol(delta_semantics)
    steps = Int(steps_per_window)
    steps > 0 || throw(ArgumentError("steps_per_window must be positive"))
    step_schedule = fill(steps, nwindow)
    poisson_scale_schedule = [1.0 / (2 * s) for s in step_schedule]
    contract = TransportBinaryContract(
        source_flux_sampling = source_flux_sampling,
        air_mass_sampling = air_mass_sampling,
        flux_sampling = flux_sampling,
        flux_kind = flux_kind,
        delta_semantics = delta_semantics,
        humidity_sampling = humidity_sampling,
        poisson_balance_target_scale = 1.0 / (2 * steps),
        poisson_balance_target_semantics = "forward_window_mass_difference / (2 * steps_per_window)",
    )

    return Dict{String, Any}(
        "magic" => "MFLX",
        "format_version" => TRANSPORT_BINARY_FORMAT_VERSION,
        "header_bytes" => header_bytes,
        "float_type" => string(FT),
        "float_bytes" => sizeof(FT),
        "grid_type" => grid_type,
        "horizontal_topology" => horizontal_topology,
        "planet_radius_m" => Float64(planet_radius),
        "ncell" => ncell,
        "nface_h" => nface_h,
        "nlevel" => nlevel,
        "nwindow" => nwindow,
        "vertical_coordinate_type" => "hybrid_sigma_pressure",
        "A_ifc" => Float64.(vc.A),
        "B_ifc" => Float64.(vc.B),
        "dt_met_seconds" => Float64(dt_met_seconds),
        "half_dt_seconds" => Float64(half_dt_seconds),
        "steps_per_window" => steps,
        "steps_per_window_by_window" => step_schedule,
        "time_step_schedule" => "constant",
        "source_flux_sampling" => String(contract.source_flux_sampling),
        "air_mass_sampling" => String(contract.air_mass_sampling),
        "flux_sampling" => String(contract.flux_sampling),
        "flux_kind" => String(contract.flux_kind),
        "humidity_sampling" => String(contract.humidity_sampling),
        "delta_semantics" => String(contract.delta_semantics),
        "poisson_balance_target_scale" => contract.poisson_balance_target_scale,
        "poisson_balance_target_semantics" => contract.poisson_balance_target_semantics,
        "poisson_balance_target_scale_by_window" => poisson_scale_schedule,
        "mass_basis" => String(mass_basis),
        "payload_sections" => String.(payload_sections),
        "include_qv_endpoints" => (:qv_start in payload_sections) || (:qv_end in payload_sections),
        "include_flux_delta" => any(section in (:dam, :dbm, :dcm, :dm, :dhflux) for section in payload_sections),
        "include_surface" => n_surface > 0,
        "surface_payload" => n_surface > 0 ? "pbl_raw_v2" : "none",
        "include_tm5conv" => n_tm5 > 0,
        "include_gchp_vdiff" => n_vdiff > 0,
        "gchp_vdiff_payload" => n_vdiff > 0 ? "u_v_t_qv_layer_center_v1" : "none",
        "n_qv_start" => n_qv_start,
        "n_qv_end" => n_qv_end,
        "n_pblh" => n_surface,
        "n_ustar" => n_surface,
        "n_pbl_hflux" => n_surface,
        "n_t2m" => n_surface,
        "n_entu" => n_tm5,
        "n_detu" => n_tm5,
        "n_entd" => n_tm5,
        "n_detd" => n_tm5,
        "n_vdiff_u" => n_vdiff,
        "n_vdiff_v" => n_vdiff,
        "n_vdiff_t" => n_vdiff,
        "n_vdiff_qv" => n_vdiff,
        "n_geometry_elems" => 0,
        "elems_per_window" => elems_per_window,
    )
end
