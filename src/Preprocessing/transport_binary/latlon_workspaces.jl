# Lat-lon spectral preprocessing workspaces and per-window staging helpers.

"""
    SpectralTransformWorkspace

Scratch buffers and native-level output arrays for one spectral-to-gridpoint
hour transform.
"""
struct SpectralTransformWorkspace
    u_cc       :: Array{Float64, 3}
    v_cc       :: Array{Float64, 3}
    u_stag     :: Array{Float64, 3}
    v_stag     :: Array{Float64, 3}
    dp         :: Array{Float64, 3}
    m_arr      :: Array{Float64, 3}
    am_arr     :: Array{Float64, 3}
    bm_arr     :: Array{Float64, 3}
    cm_arr     :: Array{Float64, 3}
    sp         :: Matrix{Float64}
    P_buf      :: Matrix{Float64}
    fft_buf    :: Vector{ComplexF64}
    field_2d   :: Matrix{Float64}
    P_buf_t    :: Vector{Matrix{Float64}}
    fft_buf_t  :: Vector{Vector{ComplexF64}}
    fft_out_t  :: Vector{Vector{ComplexF64}}
    u_spec_t   :: Vector{Matrix{ComplexF64}}
    v_spec_t   :: Vector{Matrix{ComplexF64}}
    field_2d_t :: Vector{Matrix{Float64}}
    bfft_plans :: Vector{FFTW.cFFTWPlan{ComplexF64, 1, false, 1, Tuple{Int64}}}
end

"""
    MergeWorkspace

Scratch storage used while collapsing native fields onto the merged transport
levels and while forming forward differences for `dam`, `dbm`, `dcm`, and `dm`.
"""
struct MergeWorkspace{FT}
    m_merged       :: Array{FT, 3}
    am_merged      :: Array{FT, 3}
    bm_merged      :: Array{FT, 3}
    cm_merged      :: Array{FT, 3}
    m_native       :: Array{FT, 3}
    am_native      :: Array{FT, 3}
    bm_native      :: Array{FT, 3}
    dam_merged     :: Array{FT, 3}
    dbm_merged     :: Array{FT, 3}
    dcm_merged     :: Array{FT, 3}
    dm_merged      :: Array{FT, 3}
    m_next_merged  :: Array{FT, 3}
    am_next_merged :: Array{FT, 3}
    bm_next_merged :: Array{FT, 3}
    ps_xface       :: Matrix{Float64}
    ps_yface       :: Matrix{Float64}
end

"""
    WindowStorage

In-memory storage for all windows of the current day before the final binary
write.
"""
struct WindowStorage{FT}
    all_m        :: Vector{Array{FT, 3}}
    all_am       :: Vector{Array{FT, 3}}
    all_bm       :: Vector{Array{FT, 3}}
    all_cm       :: Vector{Array{FT, 3}}
    all_ps       :: Vector{Array{FT, 2}}
    all_qv_start :: Vector{Array{FT, 3}}   # empty vector when include_qv=false
    all_qv_end   :: Vector{Array{FT, 3}}   # empty vector when include_qv=false
    all_pblh     :: Vector{Array{FT, 2}}
    all_t2m      :: Vector{Array{FT, 2}}
    all_ustar    :: Vector{Array{FT, 2}}
    all_hflux    :: Vector{Array{FT, 2}}
end

abstract type AbstractQVWorkspace{FT} end

"""
    NoQVWorkspace

Placeholder humidity workspace for moist-only runs.
"""
struct NoQVWorkspace{FT} <: AbstractQVWorkspace{FT} end

"""
    LatLonBilinearWeights(src_lon, src_lat, dst_lon, dst_lat)

Separable bilinear interpolation from point values on a latitude-longitude
source grid to the target points `(dst_lon[i], dst_lat[j])`, all in degrees.
`src_lon` is strictly increasing within one 360° period (periodic), `src_lat`
strictly increasing. Target longitude `i` lies between the source columns
`i0[i]` and `i1[i]` with weight `wi[i]` on `i1[i]`; target latitude `j`
between the source rows `j0[j]` and `j1[j]` with weight `wj[j]` on `j1[j]`.
A target latitude beyond the outermost source row takes that row's values.

The weights come from the source coordinates themselves, so a source whose
points coincide with the target points (no shift) reduces to a copy, and a
point-registered source is never read as cell-centred (issue #63).
"""
struct LatLonBilinearWeights
    i0 :: Vector{Int}
    i1 :: Vector{Int}
    wi :: Vector{Float64}
    j0 :: Vector{Int}
    j1 :: Vector{Int}
    wj :: Vector{Float64}
end

function LatLonBilinearWeights(src_lon::AbstractVector{<:Real}, src_lat::AbstractVector{<:Real},
                               dst_lon::AbstractVector{<:Real}, dst_lat::AbstractVector{<:Real})
    Nx, Ny = length(src_lon), length(src_lat)
    Nx >= 2 && Ny >= 2 || throw(ArgumentError(
        "bilinear source grid needs at least 2 longitudes and 2 latitudes; got $(Nx)×$(Ny)"))
    all(>(0), diff(src_lon)) && src_lon[end] - src_lon[1] < 360 || throw(ArgumentError(
        "source longitudes must increase strictly within one 360° period"))
    all(>(0), diff(src_lat)) || throw(ArgumentError(
        "source latitudes must increase strictly (south to north)"))
    λ0 = Float64(src_lon[1])
    i0 = Vector{Int}(undef, length(dst_lon)); i1 = similar(i0); wi = Vector{Float64}(undef, length(dst_lon))
    for (n, λ) in pairs(dst_lon)
        x = λ0 + mod(Float64(λ) - λ0, 360.0)            # in [λ0, λ0 + 360)
        a = searchsortedlast(src_lon, x)                 # ≥ 1
        λb = a == Nx ? λ0 + 360.0 : Float64(src_lon[a + 1])
        i0[n], i1[n] = a, a == Nx ? 1 : a + 1
        wi[n] = (x - src_lon[a]) / (λb - src_lon[a])
    end
    j0 = Vector{Int}(undef, length(dst_lat)); j1 = similar(j0); wj = Vector{Float64}(undef, length(dst_lat))
    for (n, φ) in pairs(dst_lat)
        a = clamp(searchsortedlast(src_lat, φ), 1, Ny - 1)
        j0[n], j1[n] = a, a + 1
        wj[n] = clamp((Float64(φ) - src_lat[a]) / (src_lat[a + 1] - src_lat[a]), 0.0, 1.0)
    end
    return LatLonBilinearWeights(i0, i1, wi, j0, j1, wj)
end

"""
    NativeQVWorkspace

Humidity workspace backed by daily ERA5 thermo NetCDF files.

`qv_thermo` holds the read on the thermo grid and `weights` bilinear-
interpolates it into `qv_native` on the target cell centres, always: the
thermo values are point values at the file coordinates
([`read_thermo_coordinates`](@ref)), which are not the target cell centres
even when the two grids have the same dimensions. `qv_daily` is an optional
in-memory daily preload in thermo-grid order; it avoids repeated NetCDF chunk
decompression.
"""
struct NativeQVWorkspace{FT} <: AbstractQVWorkspace{FT}
    thermo_path   :: String
    qv_native     :: Array{Float64, 3}   # (Nx_target, Ny_target, Nz_native)
    qv_native_ft  :: Array{FT, 3}
    qv_merged     :: Array{FT, 3}
    include_qv    :: Bool
    qv_thermo     :: Array{Float64, 3}   # (Nx_thermo, Ny_thermo, Nz_native)
    weights       :: LatLonBilinearWeights
    qv_daily      :: Array{Float64, 4}   # (Nx_thermo, Ny_thermo, Nz_native, Nt) or empty
end

Base.summary(ws::SpectralTransformWorkspace) =
    string("SpectralTransformWorkspace(", size(ws.sp, 1), "×", size(ws.sp, 2), "×", size(ws.m_arr, 3), ")")

Base.summary(ws::MergeWorkspace{FT}) where FT =
    string("MergeWorkspace{", FT, "}(", size(ws.m_merged, 1), "×", size(ws.m_merged, 2), "×", size(ws.m_merged, 3), ")")

Base.summary(ws::WindowStorage{FT}) where FT =
    string("WindowStorage{", FT, "}(", length(ws.all_m), " windows)")

"""
    allocate_transform_workspace(grid, T, Nz_native) -> SpectralTransformWorkspace

Allocate the scratch state needed to synthesize one day's hourly spectral input
onto the native target grid.
"""
function allocate_transform_workspace(grid::LatLonTargetGeometry, T::Int, Nz_native::Int)
    Nx = nlon(grid)
    Ny = nlat(grid)

    u_cc = Array{Float64}(undef, Nx, Ny, Nz_native)
    v_cc = Array{Float64}(undef, Nx, Ny, Nz_native)
    u_stag = Array{Float64}(undef, Nx + 1, Ny, Nz_native)
    v_stag = Array{Float64}(undef, Nx, Ny + 1, Nz_native)
    dp = Array{Float64}(undef, Nx, Ny, Nz_native)
    m_arr = Array{Float64}(undef, Nx, Ny, Nz_native)
    am_arr = Array{Float64}(undef, Nx + 1, Ny, Nz_native)
    bm_arr = Array{Float64}(undef, Nx, Ny + 1, Nz_native)
    cm_arr = Array{Float64}(undef, Nx, Ny, Nz_native + 1)
    sp = Array{Float64}(undef, Nx, Ny)

    nt = Threads.nthreads()
    nt_max = nt + 2  # small margin over nthreads() for thread-pool headroom
    P_buf = zeros(Float64, T + 1, T + 1)
    fft_buf = zeros(ComplexF64, Nx)
    field_2d = Array{Float64}(undef, Nx, Ny)
    P_buf_t = [zeros(Float64, T + 1, T + 1) for _ in 1:nt_max]
    fft_buf_t = [zeros(ComplexF64, Nx) for _ in 1:nt_max]
    fft_out_t = [zeros(ComplexF64, Nx) for _ in 1:nt_max]
    u_spec_t = [zeros(ComplexF64, T + 1, T + 1) for _ in 1:nt_max]
    v_spec_t = [zeros(ComplexF64, T + 1, T + 1) for _ in 1:nt_max]
    field_2d_t = [Array{Float64}(undef, Nx, Ny) for _ in 1:nt_max]
    bfft_plans = [plan_bfft(fft_buf_t[i]) for i in 1:nt_max]

    return SpectralTransformWorkspace(u_cc, v_cc, u_stag, v_stag, dp,
                                      m_arr, am_arr, bm_arr, cm_arr, sp,
                                      P_buf, fft_buf, field_2d,
                                      P_buf_t, fft_buf_t, fft_out_t,
                                      u_spec_t, v_spec_t, field_2d_t,
                                      bfft_plans)
end

"""
    allocate_merge_workspace(grid, Nz_native, Nz, FT) -> MergeWorkspace{FT}

Allocate merged-level work arrays and delta buffers for one day.
"""
function allocate_merge_workspace(grid::LatLonTargetGeometry, Nz_native::Int, Nz::Int, ::Type{FT}) where FT
    Nx = nlon(grid)
    Ny = nlat(grid)

    return MergeWorkspace{FT}(Array{FT}(undef, Nx, Ny, Nz),
                              Array{FT}(undef, Nx + 1, Ny, Nz),
                              Array{FT}(undef, Nx, Ny + 1, Nz),
                              Array{FT}(undef, Nx, Ny, Nz + 1),
                              Array{FT}(undef, Nx, Ny, Nz_native),
                              Array{FT}(undef, Nx + 1, Ny, Nz_native),
                              Array{FT}(undef, Nx, Ny + 1, Nz_native),
                              Array{FT}(undef, Nx + 1, Ny, Nz),
                              Array{FT}(undef, Nx, Ny + 1, Nz),
                              Array{FT}(undef, Nx, Ny, Nz + 1),
                              Array{FT}(undef, Nx, Ny, Nz),
                              Array{FT}(undef, Nx, Ny, Nz),
                              Array{FT}(undef, Nx + 1, Ny, Nz),
                              Array{FT}(undef, Nx, Ny + 1, Nz),
                              Array{Float64}(undef, Nx + 1, Ny),
                              Array{Float64}(undef, Nx, Ny + 1))
end

"""
    allocate_window_storage(Nt, FT; include_qv=false,
                            include_surface=false) -> WindowStorage{FT}

Allocate per-window storage for the daily output payloads.
"""
function allocate_window_storage(Nt::Int, ::Type{FT};
                                 include_qv::Bool=false,
                                 include_surface::Bool=false) where FT
    _qv_vec() = include_qv ? Vector{Array{FT, 3}}(undef, Nt) : Vector{Array{FT, 3}}()
    _sfc_vec() = include_surface ? Vector{Array{FT, 2}}(undef, Nt) : Vector{Array{FT, 2}}()
    return WindowStorage{FT}(Vector{Array{FT, 3}}(undef, Nt),
                             Vector{Array{FT, 3}}(undef, Nt),
                             Vector{Array{FT, 3}}(undef, Nt),
                             Vector{Array{FT, 3}}(undef, Nt),
                             Vector{Array{FT, 2}}(undef, Nt),
                             _qv_vec(), _qv_vec(),
                             _sfc_vec(), _sfc_vec(), _sfc_vec(), _sfc_vec())
end

"""
    allocate_qv_workspace(grid, settings, date, Nz_native, Nz, FT)

Allocate the humidity workspace for the current day. Returns `NoQVWorkspace`
when humidity is not required.
"""
function allocate_qv_workspace(grid::LatLonTargetGeometry,
                               settings,
                               date::Date,
                               Nz_native::Int,
                               Nz::Int,
                               ::Type{FT}) where FT
    setup = resolve_qv_requirements(date, settings)
    Nx = nlon(grid)
    Ny = nlat(grid)

    setup.needs_qv || return NoQVWorkspace{FT}()

    # The thermo values are point values at the file coordinates; interpolate
    # them to the target cell centres with weights built from those
    # coordinates, also when the dimensions match (issue #63).
    thermo_lon, thermo_lat = read_thermo_coordinates(setup.thermo_path)
    Nx_thermo, Ny_thermo = length(thermo_lon), length(thermo_lat)
    weights = LatLonBilinearWeights(thermo_lon, thermo_lat, grid.lons, grid.lats)
    @info @sprintf("  QV interpolation: thermo %d×%d points → target %d×%d cell centres",
                   Nx_thermo, Ny_thermo, Nx, Ny)

    qv_daily = maybe_preload_qv_day(setup.thermo_path, Nx_thermo, Ny_thermo,
                                    Nz_native, settings)

    return NativeQVWorkspace{FT}(setup.thermo_path,
                                 Array{Float64}(undef, Nx, Ny, Nz_native),
                                 Array{FT}(undef, Nx, Ny, Nz_native),
                                 Array{FT}(undef, Nx, Ny, Nz),
                                 settings.include_qv,
                                 Array{Float64}(undef, Nx_thermo, Ny_thermo, Nz_native),
                                 weights,
                                 qv_daily)
end

@inline has_daily_qv(qv::NativeQVWorkspace) = size(qv.qv_daily, 4) > 0

read_window_qv!(::NoQVWorkspace, args...) = nothing

"""
    read_window_qv!(qv, win_idx, Nx, Ny, Nz_native)

Load the humidity field associated with one analysis window into the native
humidity workspace: read it on the thermo grid and bilinear-interpolate it to
the target cell centres (`qv.weights`).
"""
function read_window_qv!(qv::NativeQVWorkspace,
                         win_idx::Int,
                         Nx::Int,
                         Ny::Int,
                         Nz_native::Int)
    size(qv.qv_native) == (Nx, Ny, Nz_native) || throw(DimensionMismatch(
        "humidity workspace is $(size(qv.qv_native)), the window needs ($Nx, $Ny, $Nz_native)"))
    Nx_t, Ny_t = size(qv.qv_thermo, 1), size(qv.qv_thermo, 2)
    if has_daily_qv(qv)
        @views qv.qv_thermo .= qv.qv_daily[:, :, :, win_idx]
    else
        qv.qv_thermo .= read_qv_from_thermo(qv.thermo_path, win_idx, Nx_t, Ny_t, Nz_native; FT=Float64)
    end
    _interpolate_ll_qv!(qv.qv_native, qv.qv_thermo, qv.weights)
    return nothing
end

"""
    _interpolate_ll_qv!(dst, src, weights::LatLonBilinearWeights)

Bilinear-interpolate QV from the thermo grid `src` (point values, in the
order of [`read_qv_from_thermo`](@ref)) to the target cell centres `dst`
with the separable `weights` built from both grids' coordinates. For the
ERA5 archive the source points are `(−180 + (i − 1)Δ, −90 + (j − 1)Δ)`
(pole-inclusive) and the targets the `LatLonMesh` cell centres
`(−180 + (i − ½)Δλ, −90 + (j − ½)Δφ)`, so the interpolated QV lands at the
same points as `transform.sp`, `merged.m_merged` and the other staging
fields.
"""
function _interpolate_ll_qv!(dst::AbstractArray{<:Real, 3}, src::AbstractArray{<:Real, 3},
                              w::LatLonBilinearWeights)
    size(dst, 1) == length(w.i0) && size(dst, 2) == length(w.j0) ||
        throw(DimensionMismatch("interpolation weights are for $(length(w.i0))×$(length(w.j0)) " *
                                "targets, the field is $(size(dst, 1))×$(size(dst, 2))"))
    size(src, 3) == size(dst, 3) || throw(DimensionMismatch(
        "source has $(size(src, 3)) levels, the target $(size(dst, 3))"))
    @inbounds for k in axes(dst, 3), j in axes(dst, 2)
        j0, j1, wj = w.j0[j], w.j1[j], w.wj[j]
        for i in axes(dst, 1)
            i0, i1, wi = w.i0[i], w.i1[i], w.wi[i]
            dst[i, j, k] = (1 - wj) * ((1 - wi) * src[i0, j0, k] + wi * src[i1, j0, k]) +
                           wj       * ((1 - wi) * src[i0, j1, k] + wi * src[i1, j1, k])
        end
    end
    return dst
end

read_next_day_qv!(::NoQVWorkspace, args...) = nothing

"""
    read_next_day_qv!(qv, date, settings, Nx, Ny, Nz_native)

Load next day's hour-0 humidity so the final window can form forward deltas
with consistent dry-basis metadata.
"""
function read_next_day_qv!(qv::NativeQVWorkspace,
                           date::Date,
                           settings,
                           Nx::Int,
                           Ny::Int,
                           Nz_native::Int)
    next_thermo_path = joinpath(settings.thermo_dir,
                                "era5_thermo_ml_$(Dates.format(date + Day(1), "yyyymmdd")).nc")
    isfile(next_thermo_path) || error("Thermo file not found: $next_thermo_path")
    Nx_t, Ny_t = size(qv.qv_thermo, 1), size(qv.qv_thermo, 2)
    qv.qv_thermo .= read_qv_from_thermo(next_thermo_path, 1, Nx_t, Ny_t, Nz_native; FT=Float64)
    _interpolate_ll_qv!(qv.qv_native, qv.qv_thermo, qv.weights)
    return nothing
end

apply_dry_basis_if_needed!(::Val{:moist}, native, qv) = nothing

function apply_dry_basis_if_needed!(::Val{:dry},
                                    native::SpectralTransformWorkspace,
                                    qv::NativeQVWorkspace)
    apply_dry_basis_native!(native.m_arr, native.am_arr, native.bm_arr, qv.qv_native)
    return nothing
end

function apply_dry_basis_if_needed!(::Val{:dry},
                                    native::SpectralTransformWorkspace,
                                    ::NoQVWorkspace)
    error("Dry-mass preprocessing requires a humidity workspace")
end

"""
    apply_dry_basis_if_needed!(basis, native, qv)

Conditionally convert the native hourly mass and horizontal flux fields from
moist to dry basis.
"""
apply_dry_basis_if_needed!(basis::Symbol, native, qv) =
    apply_dry_basis_if_needed!(Val(basis), native, qv)

merge_qv_output!(::NoQVWorkspace, merged, vertical, settings, ps_cell=nothing) = nothing

"""
    merge_qv_output!(qv, merged, vertical, settings)

Merge native humidity to the output vertical grid when `qv` is requested in the
binary payload.
"""
function merge_qv_output!(qv::NativeQVWorkspace{FT},
                          merged::MergeWorkspace{FT},
                          vertical,
                          settings,
                          ps_cell=nothing) where FT
    settings.include_qv || return nothing
    @. qv.qv_native_ft = FT(qv.qv_native)
    if vertical_mapping_method(vertical) === :pressure_overlap
        ps_cell === nothing &&
            error("pressure-overlap qv remap requires the cell surface-pressure field")
        remap_qv_pressure_overlap!(qv.qv_merged, qv.qv_native_ft, merged.m_native,
                                   ps_cell, vertical.vc_native, vertical.merged_vc)
    else
        merge_qv!(qv.qv_merged, qv.qv_native_ft, merged.m_native, vertical.merge_map)
    end
    return nothing
end

store_qv_output!(storage::WindowStorage, ::NoQVWorkspace, win_idx::Int) = nothing

"""
    store_qv_output!(storage, qv, win_idx)

Persist the merged `qv` field for one window into the daily storage bundle.
"""
function store_qv_output!(storage::WindowStorage,
                          qv::NativeQVWorkspace,
                          win_idx::Int)
    isempty(storage.all_qv_start) && return nothing
    storage.all_qv_start[win_idx] = copy(qv.qv_merged)
    return nothing
end

"""
    close_pole_faces!(bm)

Zero the meridional fluxes through the two pole faces of a global lat-lon
mesh, `bm[:, 1, :]` (`φ = −90°`) and `bm[:, Ny+1, :]` (`φ = +90°`). These faces
have zero length and are the only closed faces of the mesh. The pole rows are
ordinary cells and keep their zonal fluxes `am[:, 1, :]` and `am[:, Ny, :]`:
zeroing those leaves a flow across the pole no path through the polar cap, and
the column balance then re-routes `bm` at every latitude (issue #58; the same
zeroing was removed once before, see
`docs/resources/bug_archive/SESSION1_GRID_CM_FIX_2026-04-03.md`, section 3).
"""
function close_pole_faces!(bm::AbstractArray{FT, 3}) where FT
    @views bm[:, 1, :] .= zero(FT)
    @views bm[:, size(bm, 2), :] .= zero(FT)
    return bm
end

"""
    merge_native_window!(merged, native, qv, vertical, settings)

Collapse native hourly fields to the merged transport levels, recompute merged
`cm`, and optionally produce merged `qv`.
"""
function merge_native_window!(merged::MergeWorkspace{FT},
                              native::SpectralTransformWorkspace,
                              qv::AbstractQVWorkspace{FT},
                              vertical,
                              settings) where FT
    @. merged.m_native = FT(native.m_arr)
    @. merged.am_native = FT(native.am_arr)
    @. merged.bm_native = FT(native.bm_arr)

    if vertical_mapping_method(vertical) === :pressure_overlap
        remap_extensive_field_pressure_overlap!(merged.m_merged,
                                                merged.m_native,
                                                native.sp,
                                                vertical.vc_native,
                                                vertical.merged_vc)
        fill_xface_surface_pressure!(merged.ps_xface, native.sp)
        fill_yface_surface_pressure!(merged.ps_yface, native.sp)
        remap_extensive_field_pressure_overlap!(merged.am_merged,
                                                merged.am_native,
                                                merged.ps_xface,
                                                vertical.vc_native,
                                                vertical.merged_vc)
        remap_extensive_field_pressure_overlap!(merged.bm_merged,
                                                merged.bm_native,
                                                merged.ps_yface,
                                                vertical.vc_native,
                                                vertical.merged_vc)
    else
        merge_cell_field!(merged.m_merged, merged.m_native, vertical.merge_map)
        merge_cell_field!(merged.am_merged, merged.am_native, vertical.merge_map)
        merge_cell_field!(merged.bm_merged, merged.bm_native, vertical.merge_map)
    end

    close_pole_faces!(merged.bm_merged)

    recompute_cm_from_divergence!(merged.cm_merged, merged.am_merged, merged.bm_merged, merged.m_merged;
                                  B_ifc=vertical.merged_vc.B)
    @views merged.cm_merged[:, :, 1] .= zero(FT)
    @views merged.cm_merged[:, :, size(merged.cm_merged, 3)] .= zero(FT)

    merge_qv_output!(qv, merged, vertical, settings, native.sp)

    return nothing
end

"""
    store_window_fields!(storage, merged, ps, qv, win_idx)

Copy one processed window into the daily storage bundle.
"""
function store_window_fields!(storage::WindowStorage{FT},
                              merged::MergeWorkspace{FT},
                              ps::AbstractMatrix,
                              qv::AbstractQVWorkspace{FT},
                              win_idx::Int) where FT
    storage.all_m[win_idx] = copy(merged.m_merged)
    storage.all_am[win_idx] = copy(merged.am_merged)
    storage.all_bm[win_idx] = copy(merged.bm_merged)
    storage.all_cm[win_idx] = copy(merged.cm_merged)
    storage.all_ps[win_idx] = FT.(ps)
    store_qv_output!(storage, qv, win_idx)
    return nothing
end

"""
    log_mass_fix_configuration(settings)

Emit a short description of the dry-surface-pressure mass-fix mode selected for
the current run.
"""
function log_mass_fix_configuration(settings)
    if settings.mass_fix_enable
        if settings.include_qv || settings.mass_basis == :dry
            @info @sprintf("  Mass fix ENABLED: pin ⟨ps_dry⟩→%.2f Pa using hourly native qv",
                           settings.target_ps_dry_pa)
        else
            target_total = settings.target_ps_dry_pa / (1.0 - settings.qv_global_climatology)
            @info @sprintf("  Mass fix ENABLED: pin ⟨ps_dry⟩→%.2f Pa (⟨ps_total⟩→%.2f Pa, qv_global=%.5f)",
                           settings.target_ps_dry_pa, target_total, settings.qv_global_climatology)
        end
    else
        @info "  Mass fix DISABLED"
    end
    return nothing
end

"""
    should_log_window(win_idx, Nt) -> Bool

Return `true` for the subset of windows that should emit per-window timing
messages.
"""
@inline should_log_window(win_idx::Int, Nt::Int) =
    win_idx <= 3 || win_idx == Nt || win_idx % 8 == 0

apply_mass_fix_if_needed!(::NoQVWorkspace, args...) = nothing

"""
    apply_mass_fix_if_needed!(qv_workspace, transform, grid, vertical, settings, ps_offsets, slot)

Apply the configured global dry-mass fix to the current hourly surface-pressure
field and recompute native mass/flux fields afterward.

When humidity is available, the target dry surface pressure is matched using the
actual hourly native `q` field. Otherwise the fallback global `qv` climatology
is used.
"""
function apply_mass_fix_if_needed!(qv::NoQVWorkspace,
                                   transform::SpectralTransformWorkspace,
                                   grid::LatLonTargetGeometry,
                                   vertical,
                                   settings,
                                   ps_offsets::AbstractVector{<:Real},
                                   ps_offset_slot::Int)
    settings.mass_fix_enable || return nothing
    ps_offset = pin_global_mean_ps!(transform.sp, grid.area;
                                    target_ps_dry_pa=settings.target_ps_dry_pa,
                                    qv_global=settings.qv_global_climatology)
    recompute_native_mass_fields!(transform, vertical.ab, grid, settings.half_dt)
    ps_offsets[ps_offset_slot] = ps_offset
    return nothing
end

function apply_mass_fix_if_needed!(qv::NativeQVWorkspace,
                                   transform::SpectralTransformWorkspace,
                                   grid::LatLonTargetGeometry,
                                   vertical,
                                   settings,
                                   ps_offsets::AbstractVector{<:Real},
                                   ps_offset_slot::Int)
    settings.mass_fix_enable || return nothing
    ps_offset = pin_global_mean_ps_using_qv!(transform.sp, grid.area,
                                             vertical.ab.dA, vertical.ab.dB, qv.qv_native;
                                             target_ps_dry_pa=settings.target_ps_dry_pa)
    recompute_native_mass_fields!(transform, vertical.ab, grid, settings.half_dt)
    ps_offsets[ps_offset_slot] = ps_offset
    return nothing
end

"""
    process_window!(win_idx, hour, spec, grid, vertical, settings, transform, merged, qv, storage, ps_offsets)

Run the full hourly processing chain for one analysis window:
spectral synthesis, humidity load, optional dry-mass fix, optional dry-basis
conversion, vertical merging, and storage.
"""
function process_window!(win_idx::Int,
                         hour::Int,
                         spec,
                         grid::LatLonTargetGeometry,
                         vertical,
                         settings,
                         transform::SpectralTransformWorkspace,
                         merged::MergeWorkspace{FT},
                         qv::AbstractQVWorkspace{FT},
                         storage::WindowStorage{FT},
                         ps_offsets::AbstractVector{<:Real};
                         surface_reader = nothing) where FT
    Nx = size(transform.sp, 1)
    Ny = size(transform.sp, 2)
    t0 = time()

    spectral_to_native_fields!(
        transform.m_arr, transform.am_arr, transform.bm_arr, transform.cm_arr, transform.sp,
        transform.u_cc, transform.v_cc, transform.u_stag, transform.v_stag, transform.dp,
        spec.lnsp_all[hour], spec.vo_by_hour[hour], spec.d_by_hour[hour],
        spec.T, vertical.level_range, vertical.ab, grid, settings.half_dt,
        transform.P_buf, transform.fft_buf, transform.field_2d,
        transform.P_buf_t, transform.fft_buf_t, transform.fft_out_t,
        transform.u_spec_t, transform.v_spec_t, transform.field_2d_t,
        transform.bfft_plans)

    read_window_qv!(qv, win_idx, Nx, Ny, vertical.Nz_native)
    apply_mass_fix_if_needed!(qv, transform, grid, vertical, settings, ps_offsets, win_idx)
    apply_dry_basis_if_needed!(settings.mass_basis, transform, qv)
    merge_native_window!(merged, transform, qv, vertical, settings)
    store_window_fields!(storage, merged, transform.sp, qv, win_idx)

    if _settings_include_surface(settings) && surface_reader !== nothing
        _store_window_surface_fields!(storage, win_idx, surface_reader, FT)
    end

    elapsed = round(time() - t0, digits=2)
    should_log_window(win_idx, length(storage.all_m)) &&
        @info(@sprintf("    Window %d/%d (hour %02d): %.2fs  ps_offset=%+.3f Pa",
                       win_idx, length(storage.all_m), hour, elapsed, ps_offsets[win_idx]))

    return nothing
end

function _store_window_surface_fields!(storage::WindowStorage{FT},
                                       win_idx::Int,
                                       surface_reader::ERA5SurfaceReader,
                                       ::Type{FT}) where FT
    surface = load_era5_surface_window(surface_reader, win_idx, FT)
    storage.all_pblh[win_idx] = surface.pblh
    storage.all_t2m[win_idx] = surface.t2m
    storage.all_ustar[win_idx] = surface.ustar
    storage.all_hflux[win_idx] = surface.hflux
    return nothing
end
