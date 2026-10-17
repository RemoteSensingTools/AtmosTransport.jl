# Cubed-sphere validation and panel-native loading specializations for the
# common `TransportBinaryDriver` and `TransportWindow` types.

_supports_runtime_diffusion(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) = has_surface(reader) || :dkg in reader.header.payload_sections

function _validate_replay_consistency_cs(
    reader::TransportBinaryReader{FT, DiskFT, CubedSphereBinaryGeometry},
) where {FT, DiskFT}
    tol_rel = replay_tolerance(FT, DiskFT)
    Nt = window_count(reader)
    Nt >= 1 || return nothing

    worst_rel = 0.0
    worst_abs = 0.0
    worst_win = 0
    worst_idx = (0, 0, 0, 0)
    Nc = reader.header.geometry.Nc
    scratch = ReplayScratch((Nc, Nc, reader.header.nlevel))

    for k in 1:Nt
        cur = load_window!(reader, k)
        steps = reader.header.steps_per_window_by_window[k]
        if flux_kind(reader) === :full_window_mass_amount
            scale = FT(1) / FT(2 * steps)
            _scale_cs_replay_panels!(cur.am, scale)
            _scale_cs_replay_panels!(cur.bm, scale)
            _scale_cs_replay_panels!(cur.cm, scale)
        end
        m_target = if k < Nt
            load_window!(reader, k + 1).m
        elseif has_flux_delta(reader)
            deltas = load_flux_delta_window!(reader, k)
            if deltas === nothing || !haskey(deltas, :dm)
                cur.m
            else
                ntuple(p -> cur.m[p] .+ deltas.dm[p], length(cur.m))
            end
        else
            cur.m
        end
        diag = verify_window_continuity_cs(cur.m, cur.am, cur.bm, cur.cm, m_target, steps;
                                           scratch)
        if diag.max_rel_err > worst_rel
            worst_rel = diag.max_rel_err
            worst_abs = diag.max_abs_err
            worst_win = k
            worst_idx = diag.worst_idx
        end
    end

    worst_rel <= tol_rel ||
        throw(ArgumentError(
            "TransportBinaryDriver replay-consistency gate FAILED for " *
            "$(basename(reader.path)): rel=$(worst_rel) > tol=$(tol_rel) at window " *
            "$worst_win cell $worst_idx (abs=$worst_abs kg). Stored CS fluxes do not " *
            "integrate to the stored mass target under palindrome continuity. " *
            "Regenerate the binary with the CS replay-safe preprocessor, or skip this " *
            "load-time check (remove [input] validate_replay = true, or the " *
            "`validate_replay` keyword)."
        ))

    @info "Replay continuity gate passed: $(basename(reader.path)) " *
          "topology=cubed_sphere worst_rel=$(worst_rel) worst_window=$(worst_win)"
    return (worst_window = worst_win, worst_rel = worst_rel, worst_abs = worst_abs)
end

@inline function _scale_cs_replay_panels!(panels::NTuple{6}, scale)
    @inbounds for p in 1:6
        panels[p] .*= scale
    end
    return panels
end

function _transport_driver_grid(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
    ::CubedSphereBinaryGeometry; FT, arch, Hp,
)
    return load_grid(reader; FT, arch, Hp)
end

function _validate_driver_replay(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
    ::CubedSphereBinaryGeometry, _grid,
)
    return _validate_replay_consistency_cs(reader)
end

@inline _cs_basis_type(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) =
    mass_basis(reader) === :dry ? DryBasis : MoistBasis

@inline function _pad_horizontal(a::AbstractArray{T, N}, Hp::Int) where {T, N}
    dims = ntuple(d -> d <= 2 ? size(a, d) + 2 * Hp : size(a, d), N)
    padded = zeros(T, dims...)
    ranges = ntuple(d -> d <= 2 ? ((Hp + 1):(Hp + size(a, d))) : axes(a, d), N)
    padded[ranges...] .= a
    return padded
end

@inline function _copy_cs_storage!(dest::NTuple{6}, src::NTuple{6})
    @inbounds for p in 1:6
        copyto!(dest[p], src[p])
    end
    return dest
end

@inline function copy_fluxes!(dest::CubedSphereFaceFluxState, src::CubedSphereFaceFluxState)
    _copy_cs_storage!(dest.am, src.am)
    _copy_cs_storage!(dest.bm, src.bm)
    _copy_cs_storage!(dest.cm, src.cm)
    return dest
end

function interpolate_fluxes!(dest::CubedSphereFaceFluxState,
                             window::TransportWindow, λ::Real)
    return copy_fluxes!(dest, window.fluxes)
end

function expected_air_mass!(dest::NTuple{6}, window::TransportWindow, λ::Real)
    _copy_cs_storage!(dest, window.air_mass)
    window.deltas === nothing && return dest
    λ_ft = convert(eltype(dest[1]), λ)
    @inbounds for p in 1:6
        @. dest[p] = dest[p] + λ_ft * window.deltas.dm[p]
    end
    return dest
end

# ---------------------------------------------------------------------------
# In-place window loading into an existing host window
#
# GPU runs copy every loaded window to the device, so the host window is only
# a staging buffer. `load_transport_window!` refills one such window instead
# of allocating new panel arrays (and their padded copies) for every window:
# each payload section is copied straight into its destination, padded fields
# into their interior. Halo cells are never written, so they keep the zeros of
# the first load and the result equals `load_transport_window`.
# ---------------------------------------------------------------------------

# The panels of an existing cubed-sphere window that receive each payload
# section, and whether they are halo-padded (padded fields receive their
# interior, the others the whole panel).
function _cs_section_fields(w::TransportWindow)
    fields = Dict{Symbol, Tuple{Any, Bool}}(:m  => (w.air_mass, true),
                                            :ps => (w.surface_pressure, false),
                                            :am => (w.fluxes.am, true),
                                            :bm => (w.fluxes.bm, true),
                                            :cm => (w.fluxes.cm, true))
    w.deltas === nothing || (fields[:dm] = (w.deltas.dm, true))
    for (section, panels) in _cs_unpadded_section_fields(w)
        fields[section] = (panels, false)
    end
    return fields
end

# Destination of every payload section in an existing host window: the
# interior of a padded field, the whole panel otherwise.
function _cs_section_targets(w::TransportWindow, Hp::Int)
    interior(a) = view(a, (Hp + 1):(size(a, 1) - Hp), (Hp + 1):(size(a, 2) - Hp), :)
    return Dict{Symbol, Any}(section => (padded ? map(interior, panels) : panels)
                             for (section, (panels, padded)) in _cs_section_fields(w))
end

# The whole-panel (unpadded) fields of a window by payload section.
function _cs_unpadded_section_fields(w::TransportWindow)
    targets = Dict{Symbol, Any}()
    c = w.convection
    if c !== nothing
        c.cmfmc === nothing || (targets[:cmfmc] = c.cmfmc)
        c.dtrain === nothing || (targets[:dtrain] = c.dtrain)
        c.cloud_base === nothing || (targets[:cmfmc_cloud_base] = c.cloud_base)
        if c.tm5_fields !== nothing
            for name in (:entu, :detu, :entd, :detd)
                targets[name] = getfield(c.tm5_fields, name)
            end
        end
    end
    sf = w.surface
    if sf !== nothing
        targets[:pblh] = sf.pblh
        targets[:ustar] = sf.ustar
        targets[:pbl_hflux] = sf.hflux
        targets[:t2m] = sf.t2m
        sf.eflux === nothing || (targets[:pbl_eflux] = sf.eflux)
    end
    v = w.vdiff
    if v !== nothing
        targets[:vdiff_u] = v.u
        targets[:vdiff_v] = v.v
        targets[:vdiff_t] = v.t
        targets[:vdiff_qv] = v.qv
    end
    w.dkg === nothing || (targets[:dkg] = w.dkg)
    return targets
end

# ---------------------------------------------------------------------------
# Raw window staging (device runs)
#
# A window's payload is one contiguous run of `elems_per_window` elements of
# the binary. A device run reads it with one `read!` into a page-locked host
# buffer (no page faults on the file mapping), uploads that buffer with one
# transfer, and unpacks it on the device: each section panel is copied into its
# window panel, a padded field receiving its interior and zero halo cells. The
# device window is then exactly the upload of a host window loaded with
# `load_transport_window!` (whose halos keep the zeros of the first load).
# ---------------------------------------------------------------------------

"""
    RawWindowStaging(host, device)
    raw_window_staging(driver, reference_array) -> RawWindowStaging or nothing

A page-locked host buffer and a device buffer, each holding one window payload
of a cubed-sphere transport binary in its on-disk element type, for device
runs: `read_window_payload!` fills the host buffer, `unpack_window_payload!`
uploads it and fills a device window. `raw_window_staging` builds the buffers
on the backend of `reference_array` (a device array or six-panel tuple), or
returns `nothing` for drivers without raw window payloads.
"""
struct RawWindowStaging{H <: AbstractVector, D <: AbstractVector}
    host   :: H
    device :: D
end

raw_window_staging(::AbstractMetDriver, _reference_array) = nothing

function raw_window_staging(driver::TransportBinaryDriver{FT, <:TransportBinaryReader{FT, DiskFT},
                                                          <:AtmosGrid{<:CubedSphereMesh}},
                            reference_array) where {FT, DiskFT}
    ref = reference_array isa Tuple ? reference_array[1] : reference_array
    n = driver.reader.header.elems_per_window
    host = pin_host_memory!(Vector{DiskFT}(undef, n), ref)
    return RawWindowStaging(host, similar(ref, DiskFT, n))
end

"""
    read_window_payload!(staging::RawWindowStaging, driver, win) -> staging

Read window `win` of `driver`'s binary into `staging.host`: on Unix as
`_RAW_READ_CHUNKS` concurrent `pread`s on libuv's thread pool (`@threadcall`,
no Julia thread is blocked), elsewhere with one `read!`.
"""
function read_window_payload!(staging::RawWindowStaging,
                              driver::TransportBinaryDriver{FT, <:TransportBinaryReader{FT, DiskFT},
                                                            <:AtmosGrid{<:CubedSphereMesh}},
                              win::Int) where {FT, DiskFT}
    reader = driver.reader
    h = reader.header
    buffer = staging.host
    eltype(buffer) === DiskFT && length(buffer) == h.elems_per_window || throw(DimensionMismatch(
        "raw window staging holds $(length(buffer)) $(eltype(buffer)) values; " *
        "$(basename(reader.path)) has $(h.elems_per_window) $(DiskFT) values per window"))
    1 <= win <= h.nwindow || throw(BoundsError(1:h.nwindow, win))
    position = h.header_bytes + _transport_window_offset(reader, win) * sizeof(DiskFT)
    if Sys.isunix()
        _concurrent_pread!(buffer, reader.io, position)
    else
        lock(reader.io)
        try
            seek(reader.io, position)
            read!(reader.io, buffer)
        finally
            unlock(reader.io)
        end
    end
    return staging
end

# A single reader copies a window from the page cache at about 6 GB/s; four
# concurrent reads of its quarters reach 10-17 GB/s (C90 L72 window, 156 MB).
const _RAW_READ_CHUNKS = 4

# Fill `buffer` from byte `position` of the file behind `io` with concurrent
# `pread`s (which leave the stream's file offset alone).
function _concurrent_pread!(buffer::Vector, io::IOStream, position::Integer)
    descriptor = Base.cconvert(Cint, fd(io))
    nbytes = sizeof(buffer)
    chunk = cld(nbytes, _RAW_READ_CHUNKS)
    GC.@preserve buffer begin
        base = Ptr{UInt8}(pointer(buffer))
        @sync for c in 0:(_RAW_READ_CHUNKS - 1)
            first = c * chunk
            n = min(chunk, nbytes - first)
            n > 0 && @async _pread_all(descriptor, base + first, n, Int64(position + first))
        end
    end
    return buffer
end

function _pread_all(descriptor::Cint, destination::Ptr{UInt8}, nbytes::Int, offset::Int64)
    done = 0
    while done < nbytes
        got = @threadcall(:pread, Cssize_t, (Cint, Ptr{UInt8}, Csize_t, Int64),
                          descriptor, destination + done, nbytes - done, offset + done)
        got < 0 && Base.systemerror("pread of a transport binary window", true)
        got == 0 && throw(EOFError())
        done += got
    end
    return nothing
end

"""
Copy panel `p` of one payload section from the raw device buffer `ctx.raw`,
starting after element `offset`, into `ctx.dest[p]`: with `ctx.halo > 0` (a
padded field) the section fills the interior and the halo cells become zero,
otherwise it fills the whole panel. Values convert to the panel's element type.
"""
struct RawPanelUnpack <: AbstractPointOp
    p::Int
    offset::Int
end

index_space(::RawPanelUnpack, ctx) = size(ctx.dest[1])
bound_context(op::RawPanelUnpack, ctx) = merge(ctx, (panel = ctx.dest[op.p],))
@inline _unpack_panel(op::RawPanelUnpack, ctx) =
    hasproperty(ctx, :panel) ? ctx.panel : select_panel(ctx.dest, op.p)

@inline _raw_index(ii, jj, nx, ny) = ii + nx * (jj - 1)
@inline _raw_index(ii, jj, nx, ny, k) = ii + nx * (jj - 1) + nx * ny * (k - 1)

@inline function apply_point!(op::RawPanelUnpack, ctx, i, j, K...)
    (; raw, halo) = ctx
    a = _unpack_panel(op, ctx)
    nx, ny = size(a, 1) - 2halo, size(a, 2) - 2halo
    ii, jj = i - halo, j - halo
    T = eltype(a)
    @inbounds a[i, j, K...] = 1 <= ii <= nx && 1 <= jj <= ny ?
        T(raw[op.offset + _raw_index(ii, jj, nx, ny, K...)]) : zero(T)
    return nothing
end

"""
    unpack_window_payload!(window, driver, staging::RawWindowStaging) -> window

Upload `staging.host` (from `read_window_payload!`) to `staging.device` and
unpack it into the device `window` (one `Fused` launch of six
`RawPanelUnpack`s per payload section). The transfer and kernels are queued on
the current task's device queue; synchronize before using `window` on another
task or refilling `staging.host`.
"""
function unpack_window_payload!(w::TransportWindow,
                                driver::TransportBinaryDriver{FT, R, <:AtmosGrid{<:CubedSphereMesh}},
                                staging::RawWindowStaging) where {FT, R}
    h = driver.reader.header
    h.geometry.npanel == 6 || throw(ArgumentError(
        "raw window unpacking needs six panels; the binary has $(h.geometry.npanel)"))
    Hp = driver.grid.horizontal.Hp
    copyto!(staging.device, staging.host)
    backend = get_backend(staging.device)
    fields = _cs_section_fields(w)
    o = 0
    for section in h.payload_sections
        n = _cs_section_elements(h, section)
        entry = get(fields, section, nothing)
        if entry !== nothing
            panels, padded = entry
            per_panel = n ÷ 6
            launch!(Fused(ntuple(p -> RawPanelUnpack(p, o + (p - 1) * per_panel), Val(6))),
                    (; dest = panels, raw = staging.device, halo = padded ? Hp : 0),
                    backend; sync = false)
        end
        o += n
    end
    return w
end

# One panel of `n` elements starting after offset `o` of the mmap'd payload.
@inline _copy_cs_panel!(dst::Array, data, o::Int, n::Int) = copyto!(dst, 1, data, o + 1, n)

# Interior of a padded 3-D panel: one contiguous row (the fastest index) at a time.
function _copy_cs_panel!(dst::SubArray{<:Any, 3, <:Array}, data, o::Int, n::Int)
    a = parent(dst)
    i0, j0 = first(dst.indices[1]), first(dst.indices[2])
    nx, ny, nz = size(dst)
    nx * ny * nz == n || throw(DimensionMismatch(
        "cubed-sphere panel section has $(n) elements; the window interior holds $(nx * ny * nz)"))
    li = LinearIndices(a)
    src = o
    @inbounds for k in 1:nz, j in 1:ny
        copyto!(a, li[i0, j0 + j - 1, k], data, src + 1, nx)
        src += nx
    end
    return dst
end

# Two cubed-sphere binaries fill the same window storage (including a raw
# staging buffer) when their panel geometry, levels, payload sections,
# on-disk element type and runtime halo agree (the runtime element type is the
# driver's type parameter).
function same_window_layout(a::TransportBinaryDriver{FT, <:Any, <:AtmosGrid{<:CubedSphereMesh}},
                            b::TransportBinaryDriver{FT, <:Any, <:AtmosGrid{<:CubedSphereMesh}}) where FT
    ha, hb = a.reader.header, b.reader.header
    return ha.geometry.Nc == hb.geometry.Nc && ha.geometry.npanel == hb.geometry.npanel &&
           ha.nlevel == hb.nlevel && ha.payload_sections == hb.payload_sections &&
           ha.on_disk_float_type == hb.on_disk_float_type &&
           a.grid.horizontal.Hp == b.grid.horizontal.Hp
end

"""
    load_transport_window!(window, driver, win) -> window

Refill an existing host-resident cubed-sphere `window` (from
`load_transport_window` on the same binary, or on one with the same layout,
`same_window_layout`) with window `win`, without allocating new field arrays. Equal to `load_transport_window(driver, win)`
provided the halos of `window`'s padded fields still hold the zeros of that
first load: only interiors are written.
"""
function load_transport_window!(
    w::TransportWindow,
    driver::TransportBinaryDriver{FT, ReaderT, <:AtmosGrid{<:CubedSphereMesh}},
    win::Int,
) where {FT, ReaderT}
    reader = driver.reader
    h = reader.header
    np = h.geometry.npanel
    targets = _cs_section_targets(w, driver.grid.horizontal.Hp)
    o = _transport_window_offset(reader, win)
    for section in h.payload_sections
        n = _cs_section_elements(h, section)
        dest = get(targets, section, nothing)
        if dest !== nothing
            per_panel = n ÷ np
            for p in 1:np
                _copy_cs_panel!(dest[p], reader.data, o + (p - 1) * per_panel, per_panel)
            end
        end
        o += n
    end
    return w
end

function load_transport_window(
    driver::TransportBinaryDriver{FT, ReaderT, <:AtmosGrid{<:CubedSphereMesh}},
    win::Int,
) where {FT, ReaderT}
    raw = load_window!(driver.reader, win)
    Hp = driver.grid.horizontal.Hp
    panels_m = ntuple(p -> _pad_horizontal(raw.m[p], Hp), 6)
    panels_ps = raw.ps
    panels_am = ntuple(p -> _pad_horizontal(raw.am[p], Hp), 6)
    panels_bm = ntuple(p -> _pad_horizontal(raw.bm[p], Hp), 6)
    panels_cm = ntuple(p -> _pad_horizontal(raw.cm[p], Hp), 6)
    basis = _cs_basis_type(driver.reader)
    fluxes = CubedSphereFaceFluxState{basis}(panels_am, panels_bm, panels_cm)
    # `raw.tm5_fields` is a NamedTuple of per-panel
    # NTuples `(entu, detu, entd, detd)` when the binary carries TM5
    # sections, or `nothing` otherwise. The runtime validator in
    # DrivenSimulation decides whether TM5Convection can run against
    # this forcing; constructing ConvectionForcing here is
    # capability-preserving (present stays present, absent stays
    # absent).
    has_cmfmc_fwd = raw.cmfmc !== nothing
    has_tm5_fwd   = raw.tm5_fields !== nothing
    convection = if has_cmfmc_fwd || has_tm5_fwd
        ConvectionForcing(raw.cmfmc, raw.dtrain, raw.tm5_fields, raw.cmfmc_cloud_base)
    else
        nothing
    end
    delta_raw = load_flux_delta_window!(driver.reader, win)
    deltas = delta_raw === nothing ? nothing : CubedSphereFluxDeltas(
        ntuple(p -> _pad_horizontal(delta_raw.dm[p], Hp), 6))
    return TransportWindow(panels_m, panels_ps, fluxes;
                           deltas,
                           convection,
                           surface = raw.surface,
                           vdiff = raw.vdiff,
                           dkg = raw.dkg)
end
