"""
    TransportBinaryDriver

Meteorological driver backed by a validated version-4 transport binary.
Geometry dispatch reconstructs the appropriate grid and loads structured,
face-indexed, or panel-native windows through the same public interface.
"""
struct TransportBinaryDriver{FT, ReaderT, GridT} <: AbstractMassFluxMetDriver
    reader :: ReaderT
    grid   :: GridT
end

@inline function _cm_interface_mass(m, i, j, k, Nz)
    if k <= 1
        return m[i, j, 1]
    elseif k > Nz
        return m[i, j, Nz]
    else
        return max(m[i, j, k - 1], m[i, j, k])
    end
end

@inline function _cm_interface_mass(m, c, k, Nz)
    if k <= 1
        return m[c, 1]
    elseif k > Nz
        return m[c, Nz]
    else
        return max(m[c, k - 1], m[c, k])
    end
end

function _window_max_rel_cm(m::AbstractArray{FT, 3}, fluxes::StructuredFaceFluxState) where FT
    Nz = size(m, 3)
    worst = 0.0
    @inbounds for k in 1:size(fluxes.cm, 3), j in 1:size(fluxes.cm, 2), i in 1:size(fluxes.cm, 1)
        denom = max(Float64(_cm_interface_mass(m, i, j, k, Nz)), floatmin(Float64))
        ratio = abs(Float64(fluxes.cm[i, j, k])) / denom
        worst = max(worst, ratio)
    end
    return worst
end

function _window_max_rel_cm(m::AbstractMatrix{FT}, fluxes::FaceIndexedFluxState) where FT
    Nz = size(m, 2)
    worst = 0.0
    @inbounds for k in 1:size(fluxes.cm, 2), c in 1:size(fluxes.cm, 1)
        denom = max(Float64(_cm_interface_mass(m, c, k, Nz)), floatmin(Float64))
        ratio = abs(Float64(fluxes.cm[c, k])) / denom
        worst = max(worst, ratio)
    end
    return worst
end

function _validate_window_cm_sanity(reader::TransportBinaryReader; max_rel_cm::Real=1.0)
    threshold = Float64(max_rel_cm)
    worst_ratio = 0.0
    worst_window = 0

    for win in 1:window_count(reader)
        m, _ps, fluxes = load_window!(reader, win)
        ratio = _window_max_rel_cm(m, fluxes)
        if ratio > worst_ratio
            worst_ratio = ratio
            worst_window = win
        end
        if ratio > threshold
            throw(ArgumentError(
                "TransportBinaryDriver sanity check failed for $(basename(reader.path)) " *
                "at window $win: max(abs(cm)/m)=$(ratio) exceeds threshold $(threshold). " *
                "This transport binary has a vertical interface flux larger than the local " *
                "cell mass. Regenerate it with the current preprocessor or disable validation explicitly."
            ))
        end
    end

    return worst_window, worst_ratio
end

@inline function _replay_window_pair(::StructuredDirectionalReplayLayout,
                                     div_scratch::AbstractArray{Float64, 3},
                                     m_cur::AbstractArray{FT, 3},
                                     fluxes::StructuredFaceFluxState,
                                     m_next::AbstractArray{FT, 3},
                                     steps_per_window::Integer) where FT
    return verify_window_continuity(structured_replay_layout(), div_scratch,
                                    m_cur, fluxes.cm, m_next, steps_per_window,
                                    fluxes.am, fluxes.bm)
end

"""
    _validate_replay_consistency_ll(reader::TransportBinaryReader)

Load-time replay gate for LL structured binaries.
Walks every consecutive window pair (k, k+1) and asserts

    m[k] − 2·steps·(∇·am + ∇·bm + ∂_k cm)  ≈  m[k+1]

to within `tol_rel = 1e-10` (Float64) / `1e-4` (Float32). This mirrors the
write-time gate but fires at driver construction so a
binary produced by an older preprocessor (with the dry-basis Δb×pit cm
closure bug) is rejected before any runtime integration.

Bypass with env var `ATMOSTR_NO_REPLAY_CHECK=1`.
"""
function _validate_replay_consistency_ll(reader::TransportBinaryReader{FT}) where FT
    if get(ENV, "ATMOSTR_NO_REPLAY_CHECK", "0") == "1"
        return nothing
    end
    tol_rel = replay_tolerance(FT)
    Nt = window_count(reader)
    Nt >= 2 || return nothing

    m_cur, _ps_cur, fluxes = load_window!(reader, 1)
    div_scratch = Array{Float64}(undef, size(m_cur))
    layout = structured_replay_layout()
    worst_rel = 0.0
    worst_abs = 0.0
    worst_win = 0
    worst_idx = (0, 0, 0)
    for k in 1:(Nt - 1)
        m_next, _ps_next, fluxes_next = load_window!(reader, k + 1)
        steps = reader.header.steps_per_window_by_window[k]
        diag = _replay_window_pair(layout, div_scratch, m_cur, fluxes, m_next, steps)
        if diag.max_rel_err > worst_rel
            worst_rel = diag.max_rel_err
            worst_abs = diag.max_abs_err
            worst_win = k
            worst_idx = diag.worst_idx
        end
        # Advance: next window becomes current.
        m_cur = m_next
        fluxes = fluxes_next
    end

    worst_rel <= tol_rel ||
        throw(ArgumentError(
            "TransportBinaryDriver replay-consistency gate FAILED for " *
            "$(basename(reader.path)): rel=$(worst_rel) > tol=$(tol_rel) at window " *
            "$worst_win cell $worst_idx (abs=$worst_abs kg). Stored fluxes do not " *
            "integrate to stored m_next under palindrome continuity. Regenerate the " *
            "binary with explicit mass-delta continuity closure or " *
            "bypass with ENV[\"ATMOSTR_NO_REPLAY_CHECK\"]=\"1\" for diagnostic runs."
        ))

    @info "Replay continuity gate passed: $(basename(reader.path)) " *
          "topology=latlon worst_rel=$(worst_rel) worst_window=$(worst_win)"
    return (worst_window = worst_win, worst_rel = worst_rel, worst_abs = worst_abs)
end

# No-op fallback for arguments that are not a `TransportBinaryReader`.
# Cubed-sphere binaries have their own load-time gate,
# `_validate_replay_consistency_cs` (`cubed_sphere_driver.jl`).
_validate_replay_consistency_ll(::Any) = nothing

@inline function _rg_face_connectivity(mesh)
    nf = Grids.nfaces(mesh)
    left  = Vector{Int32}(undef, nf)
    right = Vector{Int32}(undef, nf)
    @inbounds for f in 1:nf
        l, r = Grids.face_cells(mesh, f)
        left[f]  = Int32(l)
        right[f] = Int32(r)
    end
    return left, right
end

@inline function _replay_window_pair(layout::FaceIndexedReplayLayout,
                                     div_scratch::AbstractMatrix{Float64},
                                     m_cur::AbstractMatrix{FT},
                                     fluxes::FaceIndexedFluxState,
                                     m_next::AbstractMatrix{FT},
                                     steps_per_window::Integer) where FT
    return verify_window_continuity(layout, div_scratch,
                                    m_cur, fluxes.cm, m_next, steps_per_window,
                                    fluxes.horizontal_flux)
end

"""
    _validate_replay_consistency_rg(reader::TransportBinaryReader, grid)

Load-time replay gate for RG (`:faceindexed`) binaries.
Uses the `ReducedGaussianMesh` from `grid.horizontal` to build face-cell
connectivity, then walks consecutive window pairs and asserts

    m[k] − 2·steps·(div_face_flux + ∂_k cm) ≈ m[k+1]

to within `tol_rel = 1e-10` (Float64) / `1e-4` (Float32). Bypass with
`ENV["ATMOSTR_NO_REPLAY_CHECK"]="1"`.
"""
function _validate_replay_consistency_rg(reader::TransportBinaryReader{FT}, grid) where FT
    if get(ENV, "ATMOSTR_NO_REPLAY_CHECK", "0") == "1"
        return nothing
    end
    tol_rel = replay_tolerance(FT)
    Nt = window_count(reader)
    Nt >= 2 || return nothing

    face_left, face_right = _rg_face_connectivity(grid.horizontal)
    layout = faceindexed_replay_layout(face_left, face_right)

    m_cur, _, fluxes = load_window!(reader, 1)
    _, Nz = size(m_cur)
    div_scratch = zeros(Float64, size(m_cur, 1), Nz)
    worst_rel = 0.0
    worst_abs = 0.0
    worst_win = 0
    worst_idx = (0, 0)
    for k in 1:(Nt - 1)
        m_next, _, fluxes_next = load_window!(reader, k + 1)
        steps = reader.header.steps_per_window_by_window[k]
        diag = _replay_window_pair(layout, div_scratch, m_cur, fluxes, m_next, steps)
        if diag.max_rel_err > worst_rel
            worst_rel = diag.max_rel_err
            worst_abs = diag.max_abs_err
            worst_win = k
            worst_idx = diag.worst_idx
        end
        m_cur = m_next
        fluxes = fluxes_next
    end

    worst_rel <= tol_rel ||
        throw(ArgumentError(
            "TransportBinaryDriver replay-consistency gate FAILED for " *
            "$(basename(reader.path)): rel=$(worst_rel) > tol=$(tol_rel) at window " *
            "$worst_win cell $worst_idx (abs=$worst_abs kg). Stored fluxes do not " *
            "integrate to stored m_next under palindrome continuity. Regenerate the " *
            "binary with explicit mass-delta continuity closure or " *
            "bypass with ENV[\"ATMOSTR_NO_REPLAY_CHECK\"]=\"1\" for diagnostic runs."
        ))

    @info "Replay continuity gate passed: $(basename(reader.path)) " *
          "topology=reduced_gaussian worst_rel=$(worst_rel) worst_window=$(worst_win)"
    return (worst_window = worst_win, worst_rel = worst_rel, worst_abs = worst_abs)
end

_validate_replay_consistency_rg(::Any, ::Any) = nothing

function _validate_runtime_semantics(
    reader::TransportBinaryReader,
    ::Union{LatLonBinaryGeometry, ReducedGaussianBinaryGeometry},
)
    h = reader.header
    variable_steps = _has_variable_step_schedule(h.steps_per_window_by_window)
    expected_poisson_scale = 1.0 / (2 * h.steps_per_window)
    expected_poisson_semantics = variable_steps ?
        "forward_window_mass_difference / (2 * steps_per_window_by_window[win])" :
        "forward_window_mass_difference / (2 * steps_per_window)"

    h.flux_kind === :substep_mass_amount ||
        throw(ArgumentError("TransportBinaryDriver requires flux_kind = :substep_mass_amount, got $(h.flux_kind)"))

    h.air_mass_sampling === :window_start_endpoint ||
        throw(ArgumentError("TransportBinaryDriver requires air_mass_sampling = :window_start_endpoint, got $(h.air_mass_sampling)"))

    if has_flux_delta(reader)
        h.flux_sampling in (:window_start_endpoint, :window_constant) ||
            throw(ArgumentError("TransportBinaryDriver requires flux_sampling = :window_start_endpoint or :window_constant when deltas are present, got $(h.flux_sampling)"))
        h.delta_semantics === :forward_window_endpoint_difference ||
            throw(ArgumentError("TransportBinaryDriver requires delta_semantics = :forward_window_endpoint_difference, got $(h.delta_semantics)"))
    else
        h.flux_sampling in (:window_start_endpoint, :window_mean, :window_constant) ||
            throw(ArgumentError("TransportBinaryDriver supports flux_sampling = :window_start_endpoint, :window_mean, or :window_constant without deltas, got $(h.flux_sampling)"))
    end

    if has_qv_endpoints(reader)
        h.humidity_sampling === :window_endpoints ||
            throw(ArgumentError("TransportBinaryDriver requires humidity_sampling = :window_endpoints when qv_start/qv_end are present, got $(h.humidity_sampling)"))
    else
        h.humidity_sampling === :none ||
            throw(ArgumentError("TransportBinaryDriver requires humidity_sampling = :none when humidity endpoints are absent, got $(h.humidity_sampling)"))
    end

    if has_flux_delta(reader)
        poisson_scale = h.poisson_balance_target_scale
        isfinite(poisson_scale) ||
            throw(ArgumentError("TransportBinaryDriver requires finite poisson_balance_target_scale metadata for delta-bearing transport binaries"))

        poisson_semantics = h.poisson_balance_target_semantics
        isempty(poisson_semantics) &&
            throw(ArgumentError("TransportBinaryDriver requires poisson_balance_target_semantics metadata for delta-bearing transport binaries"))

        if variable_steps
            length(h.poisson_balance_target_scale_by_window) == h.nwindow ||
                throw(ArgumentError("TransportBinaryDriver requires poisson_balance_target_scale_by_window length $(h.nwindow) for variable-step binaries"))
            for win in 1:h.nwindow
                expected_win_scale = 1.0 / (2 * h.steps_per_window_by_window[win])
                isapprox(h.poisson_balance_target_scale_by_window[win], expected_win_scale;
                         atol=eps(Float64)*8, rtol=0.0) ||
                    throw(ArgumentError("TransportBinaryDriver requires poisson_balance_target_scale_by_window[$win]=$(expected_win_scale), got $(h.poisson_balance_target_scale_by_window[win])"))
            end
        else
            isapprox(poisson_scale, expected_poisson_scale; atol=eps(Float64)*8, rtol=0.0) ||
                throw(ArgumentError("TransportBinaryDriver requires poisson_balance_target_scale=$(expected_poisson_scale), got $(poisson_scale)"))
        end
        poisson_semantics == expected_poisson_semantics ||
            throw(ArgumentError("TransportBinaryDriver requires poisson_balance_target_semantics = '$(expected_poisson_semantics)', got $(repr(poisson_semantics))"))
    end

    return nothing
end

function _validate_runtime_semantics(reader::TransportBinaryReader,
                                     ::CubedSphereBinaryGeometry)
    h = reader.header
    h.air_mass_sampling === :window_start_endpoint || throw(ArgumentError(
        "TransportBinaryDriver requires air_mass_sampling = :window_start_endpoint, " *
        "got $(h.air_mass_sampling)"))
    h.flux_sampling === :window_constant || throw(ArgumentError(
        "cubed-sphere forcing requires flux_sampling = :window_constant, " *
        "got $(h.flux_sampling)"))
    h.flux_kind in (:substep_mass_amount, :full_window_mass_amount) ||
        throw(ArgumentError("unsupported cubed-sphere flux_kind $(h.flux_kind)"))
    h.humidity_sampling === :none || throw(ArgumentError(
        "cubed-sphere forcing requires humidity_sampling = :none, " *
        "got $(h.humidity_sampling)"))
    return nothing
end

function _validate_transport_windows(reader::TransportBinaryReader,
                                     ::Union{LatLonBinaryGeometry,
                                             ReducedGaussianBinaryGeometry};
                                     max_rel_cm::Real)
    return _validate_window_cm_sanity(reader; max_rel_cm)
end

_validate_transport_windows(::TransportBinaryReader,
                            ::CubedSphereBinaryGeometry;
                            max_rel_cm::Real) = nothing

function _transport_driver_grid(reader::TransportBinaryReader,
                                ::Union{LatLonBinaryGeometry,
                                        ReducedGaussianBinaryGeometry};
                                FT, arch, Hp)
    return load_grid(reader; FT, arch)
end

function _validate_driver_replay(reader::TransportBinaryReader,
                                 ::LatLonBinaryGeometry, _grid)
    return _validate_replay_consistency_ll(reader)
end

function _validate_driver_replay(reader::TransportBinaryReader,
                                 ::ReducedGaussianBinaryGeometry, grid)
    return _validate_replay_consistency_rg(reader, grid)
end

function Base.summary(driver::TransportBinaryDriver{FT}) where {FT}
    return string(
        "TransportBinaryDriver{", FT, "}(",
        basename(driver.reader.path), ", ", grid_type(driver.reader), "/", horizontal_topology(driver.reader), ")"
    )
end

function Base.show(io::IO, driver::TransportBinaryDriver)
    reader = driver.reader
    h = reader.header
    print(io, summary(driver), "\n",
          "├── grid:          ", summary(driver.grid.horizontal), "\n",
          "├── basis:         ", air_mass_basis(driver), "\n",
          "├── timing:        dt=", window_dt(driver), " s, steps/window=",
              _steps_per_window_summary(steps_per_window(driver), steps_per_window_schedule(driver)), "\n",
          "├── payload:       ", join(String.(h.payload_sections), ", "), "\n",
          "├── humidity:      ", has_qv_endpoints(reader) ? "qv_start/qv_end" : "none", "\n",
          "├── semantics:     air_mass=", h.air_mass_sampling, ", flux=", h.flux_sampling, "/", h.flux_kind, "\n",
          "└── windows:       ", total_windows(driver))
end

"""
    TransportBinaryDriver(reader; arch=CPU(), Hp=1,
                          validate_windows=true, validate_replay=false)
    TransportBinaryDriver(path; FT=Float64, arch=CPU(), Hp=1,
                          validate_windows=true, validate_replay=false)

Open a version-4 binary and construct its runtime grid. `Hp` is the horizontal
halo width for cubed-sphere grids and is ignored by other geometries. Optional
validation checks vertical-flux magnitude for LL/RG and replay continuity for
all geometries. The path constructor closes its internally opened reader if
construction fails; callers retain ownership of a reader passed directly.
"""
function TransportBinaryDriver(reader::TransportBinaryReader{FT};
                               arch = CPU(), Hp::Int = 1,
                               validate_windows::Bool = true,
                               validate_replay::Bool = false,
                               max_rel_cm::Real = 1.0) where FT
    geometry = binary_geometry(reader)
    _validate_runtime_semantics(reader, geometry)
    validate_windows &&
        _validate_transport_windows(reader, geometry; max_rel_cm)
    grid = _transport_driver_grid(reader, geometry; FT, arch, Hp)
    # Load-time replay-consistency gate. Opt-in because the write-time
    # gate already guarantees continuity for
    # binaries we produce; the load-time gate is for suspect binaries
    # (manual imports, file corruption, older preprocessor versions).
    # Set `validate_replay=true` or `ENV["ATMOSTR_REPLAY_CHECK"]="1"` to
    # enable; disable the in-flight check with `ATMOSTR_NO_REPLAY_CHECK=1`.
    replay_on = validate_replay || get(ENV, "ATMOSTR_REPLAY_CHECK", "0") == "1"
    replay_on && _validate_driver_replay(reader, geometry, grid)
    return TransportBinaryDriver{FT, typeof(reader), typeof(grid)}(reader, grid)
end

function TransportBinaryDriver(path::AbstractString;
                               FT::Type{<:AbstractFloat} = Float64,
                               arch = CPU(), Hp::Int = 1,
                               validate_windows::Bool = true,
                               validate_replay::Bool = false,
                               max_rel_cm::Real = 1.0)
    reader = TransportBinaryReader(String(path); FT)
    try
        return TransportBinaryDriver(reader; arch, Hp, validate_windows,
                                     validate_replay, max_rel_cm)
    catch
        close(reader)
        rethrow()
    end
end

Base.close(driver::TransportBinaryDriver) = close(driver.reader)

"""
    release_payload!(driver)

On Linux, advise the kernel that pages faulted from the driver's read-only mmap
may be reclaimed. Later access safely faults them back from disk. This bounds
page-cache pressure in multi-file runs; it is a no-op on other platforms.
"""
function release_payload!(driver::TransportBinaryDriver)
    Sys.islinux() || return nothing
    data = driver.reader.data
    isempty(data) && return nothing
    MADV_DONTNEED = Cint(4)
    page_size = ccall(:getpagesize, Cint, ())
    address = UInt(pointer(data))
    base = address & ~(UInt(page_size) - 1)
    length_bytes = Csize_t(sizeof(data) + (address - base))
    ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint),
          Ptr{Cvoid}(base), length_bytes, MADV_DONTNEED)
    return nothing
end

total_windows(driver::TransportBinaryDriver) = window_count(driver.reader)
window_dt(driver::TransportBinaryDriver{FT}) where {FT} = FT(driver.reader.header.dt_met_seconds)
steps_per_window(driver::TransportBinaryDriver) = driver.reader.header.steps_per_window
steps_per_window(driver::TransportBinaryDriver, win::Integer) =
    driver.reader.header.steps_per_window_by_window[Int(win)]
steps_per_window_schedule(driver::TransportBinaryDriver) =
    copy(driver.reader.header.steps_per_window_by_window)
"""Return the binary's declared air-mass basis (`:dry` or `:moist`)."""
air_mass_basis(driver::TransportBinaryDriver) = mass_basis(driver.reader)
supports_moisture(driver::TransportBinaryDriver) = has_qv_endpoints(driver.reader)
supports_native_vertical_flux(::TransportBinaryDriver) = true
supports_convection(driver::TransportBinaryDriver) =
    has_tm5_convection(driver.reader) || has_cmfmc(driver.reader)
supports_diffusion(driver::TransportBinaryDriver) =
    _supports_runtime_diffusion(driver.reader)
_supports_runtime_diffusion(::TransportBinaryReader) = false
"""Return the grid reconstructed and owned by the runtime driver."""
driver_grid(driver::TransportBinaryDriver) = driver.grid
flux_interpolation_mode(driver::TransportBinaryDriver) =
    has_flux_delta(driver.reader) && driver.reader.header.flux_sampling !== :window_constant ? :interpolate : :constant
flux_kind(driver::TransportBinaryDriver) = flux_kind(driver.reader)
uses_binary_substep_contract(driver::TransportBinaryDriver) =
    get(driver.reader.header.raw_header, "runtime_substep_contract", nothing) ==
    "binary_schedule"

@inline function _interpolate_field!(dest, base, delta, λ)
    @. dest = base + λ * delta
    return dest
end

@inline function copy_fluxes!(dest::StructuredFaceFluxState, src::StructuredFaceFluxState)
    copyto!(dest.am, src.am)
    copyto!(dest.bm, src.bm)
    copyto!(dest.cm, src.cm)
    return dest
end

@inline function copy_fluxes!(dest::FaceIndexedFluxState, src::FaceIndexedFluxState)
    copyto!(dest.horizontal_flux, src.horizontal_flux)
    copyto!(dest.cm, src.cm)
    return dest
end

function interpolate_fluxes!(dest::StructuredFaceFluxState,
                             window::TransportWindow, λ::Real)
    λ_ft = convert(eltype(dest.am), λ)
    if window.deltas === nothing
        return copy_fluxes!(dest, window.fluxes)
    end

    _interpolate_field!(dest.am, window.fluxes.am, window.deltas.dam, λ_ft)
    _interpolate_field!(dest.bm, window.fluxes.bm, window.deltas.dbm, λ_ft)
    _interpolate_field!(dest.cm, window.fluxes.cm, window.deltas.dcm, λ_ft)
    return dest
end

function interpolate_fluxes!(dest::FaceIndexedFluxState,
                             window::TransportWindow, λ::Real)
    λ_ft = convert(eltype(dest.horizontal_flux), λ)
    if window.deltas === nothing
        return copy_fluxes!(dest, window.fluxes)
    end

    _interpolate_field!(dest.horizontal_flux, window.fluxes.horizontal_flux, window.deltas.dhflux, λ_ft)
    _interpolate_field!(dest.cm, window.fluxes.cm, window.deltas.dcm, λ_ft)
    return dest
end

function expected_air_mass!(dest, window::TransportWindow, λ::Real)
    if window.deltas === nothing
        copyto!(dest, window.air_mass)
        return dest
    end
    λ_ft = convert(eltype(dest), λ)
    _interpolate_field!(dest, window.air_mass, window.deltas.dm, λ_ft)
    return dest
end

function interpolate_qv!(dest, window::TransportWindow, λ::Real)
    has_humidity_endpoints(window) || throw(ArgumentError("transport window does not carry qv_start/qv_end"))
    λ_ft = convert(eltype(dest), λ)
    @. dest = window.qv_start + λ_ft * (window.qv_end - window.qv_start)
    return dest
end

function _make_transport_window(m, ps, fluxes::StructuredFaceFluxState;
                                qv_pair = nothing, deltas = nothing,
                                convection = nothing)
    delta_obj = if deltas === nothing
        nothing
    else
        haskey(deltas, :dam) || throw(ArgumentError("structured transport deltas require `dam`"))
        haskey(deltas, :dbm) || throw(ArgumentError("structured transport deltas require `dbm`"))
        haskey(deltas, :dcm) || throw(ArgumentError("structured transport deltas require `dcm`"))
        haskey(deltas, :dm) || throw(ArgumentError("structured transport deltas require `dm`"))
        StructuredFluxDeltas(deltas.dam, deltas.dbm, deltas.dcm, deltas.dm)
    end
    return TransportWindow(m, ps, fluxes;
                           qv_start = qv_pair === nothing ? nothing : qv_pair.qv_start,
                           qv_end = qv_pair === nothing ? nothing : qv_pair.qv_end,
                           deltas = delta_obj,
                           convection = convection)
end

function _make_transport_window(m, ps, fluxes::FaceIndexedFluxState;
                                qv_pair = nothing, deltas = nothing,
                                convection = nothing)
    delta_obj = if deltas === nothing
        nothing
    else
        haskey(deltas, :dhflux) || throw(ArgumentError("face-indexed transport deltas require `dhflux`"))
        haskey(deltas, :dcm) || throw(ArgumentError("face-indexed transport deltas require `dcm`"))
        haskey(deltas, :dm) || throw(ArgumentError("face-indexed transport deltas require `dm`"))
        FaceIndexedFluxDeltas(deltas.dhflux, deltas.dcm, deltas.dm)
    end
    return TransportWindow(m, ps, fluxes;
                           qv_start = qv_pair === nothing ? nothing : qv_pair.qv_start,
                           qv_end = qv_pair === nothing ? nothing : qv_pair.qv_end,
                           deltas = delta_obj,
                           convection = convection)
end

"""
    load_transport_window(driver, win)

Load one typed forcing window from the transport binary.
"""
function load_transport_window(driver::TransportBinaryDriver{FT, ReaderT, <:AtmosGrid{H}}, win::Int) where {FT, ReaderT, H <: AbstractStructuredMesh}
    m, ps, fluxes_any = load_window!(driver.reader, win)
    fluxes = fluxes_any::StructuredFaceFluxState
    qv_pair = load_qv_pair_window!(driver.reader, win)
    deltas = load_flux_delta_window!(driver.reader, win)
    convection = _load_transport_binary_convection_forcing(driver.reader, win)
    return _make_transport_window(m, ps, fluxes; qv_pair=qv_pair, deltas=deltas,
                                    convection=convection)
end

function load_transport_window(driver::TransportBinaryDriver{FT, ReaderT, <:AtmosGrid{<:ReducedGaussianMesh}}, win::Int) where {FT, ReaderT}
    m, ps, fluxes_any = load_window!(driver.reader, win)
    fluxes = fluxes_any::FaceIndexedFluxState
    qv_pair = load_qv_pair_window!(driver.reader, win)
    deltas = load_flux_delta_window!(driver.reader, win)
    convection = _load_transport_binary_convection_forcing(driver.reader, win)
    return _make_transport_window(m, ps, fluxes; qv_pair=qv_pair, deltas=deltas,
                                    convection=convection)
end

# Returns a ConvectionForcing with populated
# tm5_fields when the LL/RG binary carries TM5 sections; nothing
# otherwise.  CMFMC isn't yet written to LL/RG transport binaries
# (only CS), so cmfmc/dtrain stay nothing on this path.  The
# DrivenSimulation validator enforces the operator-forcing
# compatibility at runtime.
function _load_transport_binary_convection_forcing(reader::TransportBinaryReader, win::Int)
    has_tm5_convection(reader) || return nothing
    tm5 = load_tm5_convection_window!(reader, win)
    return ConvectionForcing(nothing, nothing, tm5)
end

export StructuredFluxDeltas, FaceIndexedFluxDeltas, CubedSphereFluxDeltas
export TransportWindow
export TransportBinaryDriver, driver_grid, air_mass_basis, load_transport_window
export has_humidity_endpoints, interpolate_fluxes!, expected_air_mass!, interpolate_qv!, copy_fluxes!
