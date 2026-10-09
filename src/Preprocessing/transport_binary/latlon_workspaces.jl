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
    # Four TM5 convection sections on the merged
    # transport grid, one per window.  Empty vectors when
    # tm5_convection=false.  Shape per window: (Nx, Ny, Nz).
    all_entu     :: Vector{Array{FT, 3}}
    all_detu     :: Vector{Array{FT, 3}}
    all_entd     :: Vector{Array{FT, 3}}
    all_detd     :: Vector{Array{FT, 3}}
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
    NativeQVWorkspace

Humidity workspace backed by daily ERA5 thermo NetCDF files.

When the thermo grid resolution differs from the target grid (e.g. thermo is
720×361 but target is 96×48), `qv_thermo` holds the raw read and
`_interpolate_qv_to_target!` bilinear-interpolates into `qv_native`.
When resolutions match, `qv_thermo` is empty and the read goes directly
into `qv_native`. `qv_daily` is an optional in-memory daily preload in
thermo-grid coordinates; it avoids repeated NetCDF chunk decompression.
"""
struct NativeQVWorkspace{FT} <: AbstractQVWorkspace{FT}
    thermo_path   :: String
    qv_native     :: Array{Float64, 3}   # (Nx_target, Ny_target, Nz_native)
    qv_native_ft  :: Array{FT, 3}
    qv_merged     :: Array{FT, 3}
    include_qv    :: Bool
    qv_thermo     :: Array{Float64, 3}   # (Nx_thermo, Ny_thermo, Nz_native) or empty
    needs_interp  :: Bool
    qv_daily      :: Array{Float64, 4}   # (Nx_thermo, Ny_thermo, Nz_native, Nt) or empty
end

struct TM5SourcePressureWorkspace{G, P}
    grid      :: G
    sp        :: Matrix{Float64}
    P_buf     :: Matrix{Float64}
    fft_buf   :: Vector{ComplexF64}
    field_2d  :: Matrix{Float64}
    fft_out   :: Vector{ComplexF64}
    bfft_plan :: P
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

function allocate_tm5_source_pressure_workspace(grid::LatLonTargetGeometry, T::Int)
    Nx = nlon(grid)
    Ny = nlat(grid)
    fft_buf = zeros(ComplexF64, Nx)
    bfft_plan = plan_bfft(fft_buf)
    return TM5SourcePressureWorkspace(
        grid,
        Array{Float64}(undef, Nx, Ny),
        zeros(Float64, T + 1, T + 1),
        fft_buf,
        Array{Float64}(undef, Nx, Ny),
        zeros(ComplexF64, Nx),
        bfft_plan,
    )
end

function synthesize_tm5_source_pressure!(ws::TM5SourcePressureWorkspace,
                                         lnsp_spec::AbstractMatrix{ComplexF64},
                                         T::Integer;
                                         ps_offset::Real = 0.0)
    lon_shift = deg2rad(ws.grid.lons[1])
    spectral_to_grid!(ws.field_2d, lnsp_spec, Int(T), ws.grid.lats,
                      nlon(ws.grid), ws.P_buf, ws.fft_buf;
                      fft_out = ws.fft_out,
                      bfft_plan = ws.bfft_plan,
                      lon_shift_rad = lon_shift)
    @. ws.sp = exp(ws.field_2d) + Float64(ps_offset)
    return ws.sp
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
    allocate_window_storage(Nt, FT; include_qv=false, tm5_convection=false,
                            include_surface=false) -> WindowStorage{FT}

Allocate per-window storage for the daily output payloads. When
`tm5_convection=true`, also allocate the four TM5 section vectors.
"""
function allocate_window_storage(Nt::Int, ::Type{FT};
                                 include_qv::Bool=false,
                                 tm5_convection::Bool=false,
                                 include_surface::Bool=false) where FT
    _qv_vec() = include_qv ? Vector{Array{FT, 3}}(undef, Nt) : Vector{Array{FT, 3}}()
    _tm5_vec() = tm5_convection ? Vector{Array{FT, 3}}(undef, Nt) : Vector{Array{FT, 3}}()
    _sfc_vec() = include_surface ? Vector{Array{FT, 2}}(undef, Nt) : Vector{Array{FT, 2}}()
    return WindowStorage{FT}(Vector{Array{FT, 3}}(undef, Nt),
                             Vector{Array{FT, 3}}(undef, Nt),
                             Vector{Array{FT, 3}}(undef, Nt),
                             Vector{Array{FT, 3}}(undef, Nt),
                             Vector{Array{FT, 2}}(undef, Nt),
                             _qv_vec(), _qv_vec(),
                             _tm5_vec(), _tm5_vec(), _tm5_vec(), _tm5_vec(),
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

    # Detect thermo/target grid mismatch
    Nx_thermo, Ny_thermo = _detect_thermo_grid_size(setup.thermo_path)
    needs_interp = (Nx_thermo != Nx || Ny_thermo != Ny)
    qv_thermo = needs_interp ? Array{Float64}(undef, Nx_thermo, Ny_thermo, Nz_native) :
                               Array{Float64}(undef, 0, 0, 0)
    if needs_interp
        @info @sprintf("  QV interpolation: thermo %d×%d → target %d×%d",
                       Nx_thermo, Ny_thermo, Nx, Ny)
    end

    qv_daily = maybe_preload_qv_day(setup.thermo_path, Nx_thermo, Ny_thermo,
                                    Nz_native, settings)

    return NativeQVWorkspace{FT}(setup.thermo_path,
                                 Array{Float64}(undef, Nx, Ny, Nz_native),
                                 Array{FT}(undef, Nx, Ny, Nz_native),
                                 Array{FT}(undef, Nx, Ny, Nz),
                                 settings.include_qv,
                                 qv_thermo,
                                 needs_interp,
                                 qv_daily)
end

"""
    _detect_thermo_grid_size(thermo_path) -> (Nx, Ny)

Read the lon/lat dimensions of the thermo NetCDF file.
"""
function _detect_thermo_grid_size(thermo_path::String)
    NCDataset(thermo_path) do ds
        q_var = ds["q"]
        dims = dimnames(q_var)
        if dims[1] == "longitude"
            return (size(q_var, 1), size(q_var, 2))
        else
            return (size(q_var, 4), size(q_var, 3))
        end
    end
end

@inline has_daily_qv(qv::NativeQVWorkspace) = size(qv.qv_daily, 4) > 0

read_window_qv!(::NoQVWorkspace, args...) = nothing

"""
    read_window_qv!(qv, win_idx, Nx, Ny, Nz_native)

Load the humidity field associated with one analysis window into the native
humidity workspace. When the thermo grid differs from the target grid,
bilinear-interpolates QV to the target resolution.
"""
function read_window_qv!(qv::NativeQVWorkspace,
                         win_idx::Int,
                         Nx::Int,
                         Ny::Int,
                         Nz_native::Int)
    if qv.needs_interp
        Nx_t = size(qv.qv_thermo, 1)
        Ny_t = size(qv.qv_thermo, 2)
        if has_daily_qv(qv)
            @views qv.qv_thermo .= qv.qv_daily[:, :, :, win_idx]
        else
            qv.qv_thermo .= read_qv_from_thermo(qv.thermo_path, win_idx, Nx_t, Ny_t, Nz_native; FT=Float64)
        end
        _interpolate_ll_qv!(qv.qv_native, qv.qv_thermo, Nx, Ny, Nx_t, Ny_t, Nz_native)
    else
        if has_daily_qv(qv)
            @views qv.qv_native .= qv.qv_daily[:, :, :, win_idx]
        else
            qv.qv_native .= read_qv_from_thermo(qv.thermo_path, win_idx, Nx, Ny, Nz_native; FT=Float64)
        end
    end
    return nothing
end

"""
    _interpolate_ll_qv!(dst, src, Nx_dst, Ny_dst, Nx_src, Ny_src, Nz)

Bilinear-interpolate QV from a regular LL source grid (ERA5 reanalysis,
pole-inclusive: lats include both `-90` and `+90`) to the staging mesh
target grid (cell-centered: lats span `-90 + Δ/2` to `90 - Δ/2` where
`Δ = 180/Ny_dst`).

Both source and target use cell-centered longitudes in `[-180, 180)` —
caller is responsible for normalizing the source longitude convention
upstream (see [`_normalize_lon_to_centered`](@ref) in `mass_support.jl`).

The target latitude convention here matches `LatLonMesh.φᶜ`, so the
interpolated QV lands at the same physical points as `transform.sp`,
`merged.m_merged`, and the rest of the staging fields.

Source grid layout (ERA5):
- Longitudes: cell-centered, `Nx_src` points spanning a full `360°`
  with spacing `Δλ = 360/Nx_src`. After [`_normalize_lon_to_centered`](@ref)
  the array is in `[-180, 180)` convention.
- Latitudes: pole-inclusive, `Ny_src` points spanning the full `180°`
  with spacing `Δφ = 180/(Ny_src - 1)`, including both poles.

Target grid layout (staging `LatLonMesh`):
- Longitudes: cell-centered, `Nx_dst` points, spacing `Δλ = 360/Nx_dst`.
- Latitudes: cell-centered, `Ny_dst` points, spacing `Δφ = 180/Ny_dst`,
  starting at `-90 + Δφ/2` (south pole face is *not* a cell center).
"""
function _interpolate_ll_qv!(dst::AbstractArray{<:Real, 3}, src::AbstractArray{<:Real, 3},
                              Nx_dst::Int, Ny_dst::Int,
                              Nx_src::Int, Ny_src::Int,
                              Nz::Int)
    Δlon_src = 360.0 / Nx_src
    Δlat_src = 180.0 / (Ny_src - 1)   # source pole-inclusive
    Δlon_dst = 360.0 / Nx_dst
    Δlat_dst = 180.0 / Ny_dst         # target cell-centered (no -1)

    @inbounds for k in 1:Nz, j in 1:Ny_dst, i in 1:Nx_dst
        # Target cell center in degrees
        lon = -180.0 + (i - 0.5) * Δlon_dst
        lat = -90.0 + (j - 0.5) * Δlat_dst

        # Source fractional indices — source is pole-inclusive so
        # j_src = 1 corresponds to lat = -90, j_src = Ny_src to lat = +90.
        if_val = (lon + 180.0) / Δlon_src + 0.5
        jf_val = (lat + 90.0) / Δlat_src + 1.0

        i0 = clamp(floor(Int, if_val), 1, Nx_src)
        j0 = clamp(floor(Int, jf_val), 1, Ny_src - 1)
        i1 = i0 == Nx_src ? 1 : i0 + 1  # periodic longitude
        j1 = min(j0 + 1, Ny_src)

        wi = if_val - floor(if_val)
        wj = jf_val - j0

        dst[i, j, k] = (1 - wi) * (1 - wj) * src[i0, j0, k] +
                        wi       * (1 - wj) * src[i1, j0, k] +
                        (1 - wi) * wj       * src[i0, j1, k] +
                        wi       * wj       * src[i1, j1, k]
    end
    return nothing
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
    if qv.needs_interp
        Nx_t = size(qv.qv_thermo, 1)
        Ny_t = size(qv.qv_thermo, 2)
        qv.qv_thermo .= read_qv_from_thermo(next_thermo_path, 1, Nx_t, Ny_t, Nz_native; FT=Float64)
        _interpolate_ll_qv!(qv.qv_native, qv.qv_thermo, Nx, Ny, Nx_t, Ny_t, Nz_native)
    else
        qv.qv_native .= read_qv_from_thermo(next_thermo_path, 1, Nx, Ny, Nz_native; FT=Float64)
    end
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
    apply_structured_pole_constraints!(am, bm)

Enforce the structured lat-lon polar boundary conditions after vertical merging.
"""
function apply_structured_pole_constraints!(am::AbstractArray{FT, 3},
                                            bm::AbstractArray{FT, 3}) where FT
    Ny = size(am, 2)
    @views am[:, 1, :] .= zero(FT)
    @views am[:, Ny, :] .= zero(FT)
    @views bm[:, 1, :] .= zero(FT)
    @views bm[:, Ny + 1, :] .= zero(FT)
    return nothing
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

    apply_structured_pole_constraints!(merged.am_merged, merged.bm_merged)

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
                         physics_reader = nothing,
                         tm5_ws = nothing,
                         tm5_source_ps = nothing,
                         tm5_stats = nothing,
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

    # TM5 convection step — runs only when the caller wired up a physics BIN
    # reader and TM5 workspace.  The physics BIN omits PS.  Native LL keeps
    # using the target PS fast path; source!=target LL synthesizes PS on the
    # physics source grid and applies the same scalar mass-fix offset as the
    # target transport grid before TM5 conversion.
    if settings.tm5_convection_enable && physics_reader !== nothing
        ps_tm5 = tm5_source_ps === nothing ? transform.sp :
                 synthesize_tm5_source_pressure!(
                     tm5_source_ps, spec.lnsp_all[hour], spec.T;
                     ps_offset = ps_offsets[win_idx])
        _store_window_tm5_fields!(storage, win_idx,
                                   physics_reader, tm5_ws,
                                   ps_tm5, vertical, tm5_stats, FT)
    end
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

# Per-window TM5 compute + store.
# `ps_source` is on the TM5 physics source grid. For source==target this is
# the main transport `transform.sp`; for lower-resolution LL targets it comes
# from `TM5SourcePressureWorkspace`.
# `win_idx` is the 1-indexed physics-BIN hour slot (matches the
# 24 hourly slices in the Commit-2 daily BIN).
function _store_window_tm5_fields!(storage::WindowStorage{FT},
                                    win_idx::Int,
                                    physics_reader,
                                    tm5_ws::TM5PreprocessingWorkspace{FT},
                                    ps_source::AbstractMatrix,
                                    vertical,
                                    tm5_stats,
                                    ::Type{FT}) where FT
    compute_tm5_merged_hour_on_source!(
        tm5_ws, physics_reader, win_idx, ps_source,
        vertical.ab.a_ifc, vertical.ab.b_ifc,
        vertical.Nz_native, vertical.merge_map;
        stats = tm5_stats)

    Nx_t, Ny_t, Nz = tm5_ws.target_nlon, tm5_ws.target_nlat, size(tm5_ws.entu_merged_src, 3)
    storage.all_entu[win_idx] = Array{FT, 3}(undef, Nx_t, Ny_t, Nz)
    storage.all_detu[win_idx] = Array{FT, 3}(undef, Nx_t, Ny_t, Nz)
    storage.all_entd[win_idx] = Array{FT, 3}(undef, Nx_t, Ny_t, Nz)
    storage.all_detd[win_idx] = Array{FT, 3}(undef, Nx_t, Ny_t, Nz)
    tm5_copy_or_regrid_ll!(storage.all_entu[win_idx], tm5_ws.entu_merged_src, tm5_ws)
    tm5_copy_or_regrid_ll!(storage.all_detu[win_idx], tm5_ws.detu_merged_src, tm5_ws)
    tm5_copy_or_regrid_ll!(storage.all_entd[win_idx], tm5_ws.entd_merged_src, tm5_ws)
    tm5_copy_or_regrid_ll!(storage.all_detd[win_idx], tm5_ws.detd_merged_src, tm5_ws)
    return nothing
end
