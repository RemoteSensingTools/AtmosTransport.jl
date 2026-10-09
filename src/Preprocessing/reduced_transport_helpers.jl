struct ReducedSpectralThreadCache
    P_buf        :: Matrix{Float64}
    fft_buffers  :: Dict{Int, Vector{ComplexF64}}
    real_buffers :: Dict{Int, Vector{Float64}}
    u_spec       :: Matrix{ComplexF64}
    v_spec       :: Matrix{ComplexF64}
end

struct ReducedTransformWorkspace
    sp             :: Vector{Float64}
    lnsp           :: Vector{Float64}
    dp             :: Matrix{Float64}
    m_arr          :: Matrix{Float64}
    hflux_arr      :: Matrix{Float64}
    cm_arr         :: Matrix{Float64}
    cell_areas     :: Vector{Float64}
    face_left      :: Vector{Int32}
    face_right     :: Vector{Int32}
    div_scratch    :: Matrix{Float64}
    # Scratch buffers for the Poisson-balance CG solver (cell-indexed).
    balance_psi    :: Vector{Float64}
    balance_rhs    :: Vector{Float64}
    balance_r      :: Vector{Float64}
    balance_p      :: Vector{Float64}
    balance_Ap     :: Vector{Float64}
    balance_z      :: Vector{Float64}
    caches         :: Vector{ReducedSpectralThreadCache}
end

struct ReducedMergeWorkspace{FT}
    m_native_ft     :: Matrix{FT}
    hflux_native_ft :: Matrix{FT}
    m_merged        :: Matrix{FT}
    hflux_merged    :: Matrix{FT}
    cm_merged       :: Matrix{FT}
    div_scratch     :: Matrix{Float64}
end

"""
    ReducedQVWorkspace

Humidity workspace for dry-basis conversion on reduced Gaussian grids.
Loads QV from ERA5 thermo NetCDF on a regular LL grid and bilinear-interpolates
to RG cell centers. The interpolation mapping is precomputed at allocation time.
"""
struct ReducedQVWorkspace
    qv_cell    :: Matrix{Float64}   # (nc, Nz_native) — QV at RG cell centers
    qv_ll      :: Array{Float64, 3} # (Nx_ll, Ny_ll, Nz_native) — LL buffer
    qv_daily_ll:: Array{Float64, 4} # optional daily LL preload or empty
    i0         :: Vector{Int32}     # bilinear: left lon index per cell
    j0         :: Vector{Int32}     # bilinear: bottom lat index per cell
    wi         :: Vector{Float64}   # bilinear: lon fractional weight
    wj         :: Vector{Float64}   # bilinear: lat fractional weight
    Nx_ll      :: Int
    Ny_ll      :: Int
end

"""
    allocate_reduced_qv_workspace(grid, Nz_native, thermo_path)

Build the RG QV workspace with precomputed bilinear interpolation from the LL
thermo grid to RG cell centers. The LL grid dimensions are read from the first
available thermo file.
"""
function allocate_reduced_qv_workspace(grid::ReducedGaussianTargetGeometry,
                                       Nz_native::Int,
                                       thermo_path::String;
                                       settings=nothing)
    # Discover LL grid dimensions from thermo file
    Nx_ll, Ny_ll = NCDataset(thermo_path) do ds
        q_var = ds["q"]
        dims = dimnames(q_var)
        if dims[1] == "longitude"
            (size(q_var, 1), size(q_var, 2))
        else
            (size(q_var, 4), size(q_var, 3))
        end
    end

    mesh = grid.mesh
    nc = ncells(mesh)

    i0 = Vector{Int32}(undef, nc)
    j0 = Vector{Int32}(undef, nc)
    wi = Vector{Float64}(undef, nc)
    wj = Vector{Float64}(undef, nc)

    Δlon = 360.0 / Nx_ll
    Δlat = 180.0 / (Ny_ll - 1)

    n_rings = length(grid.lats)
    for j_ring in 1:n_rings
        lat = grid.lats[j_ring]
        nlon = grid.nlon_per_ring[j_ring]
        start = mesh.ring_offsets[j_ring]

        jf = (lat + 90.0) / Δlat + 1.0
        j0_val = clamp(floor(Int32, jf), Int32(1), Int32(Ny_ll - 1))
        wj_val = jf - j0_val

        for ic in 0:(nlon - 1)
            c = start + ic
            lon = grid.lons_by_ring[j_ring][ic + 1]

            if_val = (lon + 180.0) / Δlon + 1.0
            i0_val = mod1(floor(Int32, if_val), Int32(Nx_ll))
            wi_val = if_val - floor(if_val)

            i0[c] = i0_val
            j0[c] = j0_val
            wi[c] = wi_val
            wj[c] = wj_val
        end
    end

    qv_daily = settings === nothing ?
        zeros(Float64, 0, 0, 0, 0) :
        maybe_preload_qv_day(thermo_path, Nx_ll, Ny_ll, Nz_native, settings)

    return ReducedQVWorkspace(
        zeros(Float64, nc, Nz_native),
        zeros(Float64, Nx_ll, Ny_ll, Nz_native),
        qv_daily,
        i0, j0, wi, wj, Nx_ll, Ny_ll)
end

"""
    load_rg_qv!(qv_ws, thermo_path, hour_idx, Nz_native)

Read QV from ERA5 thermo NetCDF for one hour, then bilinear-interpolate from
the regular LL grid to all RG cell centers.
"""
function load_rg_qv!(qv_ws::ReducedQVWorkspace,
                     thermo_path::String,
                     hour_idx::Int,
                     Nz_native::Int)
    if size(qv_ws.qv_daily_ll, 4) > 0
        @views qv_ws.qv_ll .= qv_ws.qv_daily_ll[:, :, :, hour_idx]
    else
        qv_ws.qv_ll .= read_qv_from_thermo(thermo_path, hour_idx,
                                             qv_ws.Nx_ll, qv_ws.Ny_ll, Nz_native;
                                             FT=Float64)
    end
    _interpolate_ll_to_rg!(qv_ws)
    return nothing
end

"""
    _interpolate_ll_to_rg!(qv_ws)

Bilinear-interpolate `qv_ll` (regular LL grid) → `qv_cell` (RG cell centers)
using the precomputed index/weight mapping.
"""
function _interpolate_ll_to_rg!(qv_ws::ReducedQVWorkspace)
    nc = size(qv_ws.qv_cell, 1)
    Nz = size(qv_ws.qv_cell, 2)
    Nx = qv_ws.Nx_ll
    Ny = qv_ws.Ny_ll

    @inbounds for k in 1:Nz, c in 1:nc
        i0 = Int(qv_ws.i0[c])
        j0 = Int(qv_ws.j0[c])
        i1 = i0 == Nx ? 1 : i0 + 1   # periodic in longitude
        j1 = min(j0 + 1, Ny)          # clamp at pole
        w_i = qv_ws.wi[c]
        w_j = qv_ws.wj[c]

        qv_ws.qv_cell[c, k] =
            (1 - w_i) * (1 - w_j) * qv_ws.qv_ll[i0, j0, k] +
            w_i       * (1 - w_j) * qv_ws.qv_ll[i1, j0, k] +
            (1 - w_i) * w_j       * qv_ws.qv_ll[i0, j1, k] +
            w_i       * w_j       * qv_ws.qv_ll[i1, j1, k]
    end
    return nothing
end

"""
    apply_dry_basis_reduced!(work, qv_cell)

Convert RG native-level moist mass and horizontal face fluxes to dry basis.
Face humidity is the average of the two adjacent cell values.
"""
function apply_dry_basis_reduced!(work::ReducedTransformWorkspace,
                                  qv_cell::Matrix{Float64})
    nc = size(work.m_arr, 1)
    Nz = size(work.m_arr, 2)
    nf = size(work.hflux_arr, 1)

    @inbounds for k in 1:Nz, c in 1:nc
        q = clamp(qv_cell[c, k], 0.0, 0.999999)
        work.m_arr[c, k] *= (1.0 - q)
    end

    @inbounds for k in 1:Nz, f in 1:nf
        left  = work.face_left[f]
        right = work.face_right[f]
        if left == 0 || right == 0
            # Polar stub faces represent a closed boundary, not a real
            # cell-to-cell connection. Enforce the zero-flux invariant before
            # touching qv_cell; indexing cell 0 under @inbounds is undefined.
            work.hflux_arr[f, k] = 0.0
            continue
        end
        q_face = 0.5 * (qv_cell[left, k] + qv_cell[right, k])
        q_face = clamp(q_face, 0.0, 0.999999)
        work.hflux_arr[f, k] *= (1.0 - q_face)
    end
    return nothing
end

function allocate_reduced_transform_workspace(grid::ReducedGaussianTargetGeometry,
                                              T::Int,
                                              Nz_native::Int)
    mesh = grid.mesh
    nc = ncells(mesh)
    nf = nfaces(mesh)
    nt = Threads.nthreads()
    nt_max = max(nt, 2 * nt) + 4

    cell_areas = [cell_area(mesh, c) for c in 1:nc]
    buffer_lengths = sort!(unique(vcat(collect(mesh.nlon_per_ring), collect(mesh.boundary_counts))))
    face_left = Vector{Int32}(undef, nf)
    face_right = Vector{Int32}(undef, nf)
    for f in 1:nf
        left, right = face_cells(mesh, f)
        face_left[f] = Int32(left)
        face_right[f] = Int32(right)
    end

    caches = [ReducedSpectralThreadCache(
                 zeros(Float64, T + 1, T + 1),
                 Dict(n => zeros(ComplexF64, n) for n in buffer_lengths),
                 Dict(n => zeros(Float64, n) for n in buffer_lengths),
                 zeros(ComplexF64, T + 1, T + 1),
                 zeros(ComplexF64, T + 1, T + 1),
             ) for _ in 1:nt_max]

    return ReducedTransformWorkspace(
        zeros(Float64, nc),
        zeros(Float64, nc),
        zeros(Float64, nc, Nz_native),
        zeros(Float64, nc, Nz_native),
        zeros(Float64, nf, Nz_native),
        zeros(Float64, nc, Nz_native + 1),
        cell_areas,
        face_left,
        face_right,
        zeros(Float64, nc, Nz_native),
        zeros(Float64, nc),   # balance_psi
        zeros(Float64, nc),   # balance_rhs
        zeros(Float64, nc),   # balance_r
        zeros(Float64, nc),   # balance_p
        zeros(Float64, nc),   # balance_Ap
        zeros(Float64, nc),   # balance_z (preconditioned residual)
        caches,
    )
end

function allocate_reduced_merge_workspace(grid::ReducedGaussianTargetGeometry,
                                          Nz_native::Int,
                                          Nz::Int,
                                          ::Type{FT}) where FT
    mesh = grid.mesh
    nc = ncells(mesh)
    nf = nfaces(mesh)
    return ReducedMergeWorkspace{FT}(
        zeros(FT, nc, Nz_native),
        zeros(FT, nf, Nz_native),
        zeros(FT, nc, Nz),
        zeros(FT, nf, Nz),
        zeros(FT, nc, Nz + 1),
        zeros(Float64, nc, Nz),
    )
end

@inline function _fft_buffer!(cache::ReducedSpectralThreadCache, n::Int)
    return cache.fft_buffers[n]
end

@inline function _real_buffer!(cache::ReducedSpectralThreadCache, n::Int)
    return cache.real_buffers[n]
end

function spectral_to_ring!(dest::AbstractVector{Float64},
                           spec::AbstractMatrix{ComplexF64},
                           T::Int,
                           lat_deg::Real,
                           cache::ReducedSpectralThreadCache;
                           lon_shift_rad::Real = 0.0)
    Nlon = length(dest)
    lat_deg64 = Float64(lat_deg)
    lon_shift_rad64 = Float64(lon_shift_rad)
    compute_legendre_column!(cache.P_buf, T, sind(lat_deg64))
    fft_buf = _fft_buffer!(cache, Nlon)
    fill!(fft_buf, zero(ComplexF64))

    for m in 0:min(T, div(Nlon, 2))
        Gm = zero(ComplexF64)
        @inbounds for n in m:T
            Gm += spec[n + 1, m + 1] * cache.P_buf[n + 1, m + 1]
        end
        if lon_shift_rad64 != 0.0 && m > 0
            Gm *= exp(im * m * lon_shift_rad64)
        end
        fft_buf[m + 1] = Gm
    end

    # Hermitian mirror of every 0 < m < Nlon/2 (for odd Nlon this includes
    # m = (Nlon − 1)/2); the Nyquist term of an even ring has no mirror.
    for m in 1:min(T, (Nlon - 1) ÷ 2)
        fft_buf[Nlon - m + 1] = conj(fft_buf[m + 1])
    end

    FFTW.bfft!(fft_buf)
    @inbounds for i in 1:Nlon
        dest[i] = real(fft_buf[i])
    end
    return dest
end

function spectral_to_reduced_scalar!(field::Vector{Float64},
                                     spec::AbstractMatrix{ComplexF64},
                                     T::Int,
                                     grid::ReducedGaussianTargetGeometry,
                                     cache::ReducedSpectralThreadCache;
                                     centered::Bool = true)
    mesh = grid.mesh
    @inbounds for j in 1:nrings(mesh)
        start = mesh.ring_offsets[j]
        stop = mesh.ring_offsets[j + 1] - 1
        shift = centered ? (pi / mesh.nlon_per_ring[j]) : 0.0
        spectral_to_ring!(@view(field[start:stop]), spec, T, grid.lats[j], cache; lon_shift_rad=shift)
    end
    return field
end

function spectral_to_reduced_boundary!(dest::AbstractVector{Float64},
                                       spec::AbstractMatrix{ComplexF64},
                                       T::Int,
                                       lat_deg::Float64,
                                       cache::ReducedSpectralThreadCache)
    spectral_to_ring!(dest, spec, T, lat_deg, cache; lon_shift_rad=pi / length(dest))
    return dest
end

function compute_reduced_dp_and_mass!(dp::Matrix{Float64},
                                      m_arr::Matrix{Float64},
                                      sp::Vector{Float64},
                                      cell_areas::Vector{Float64},
                                      dA,
                                      dB)
    nc = length(sp)
    Nz = length(dA)
    inv_g = 1.0 / STANDARD_GRAVITY
    @inbounds for k in 1:Nz, c in 1:nc
        dp_face = abs(dA[k] + dB[k] * sp[c])
        dp[c, k] = dp_face
        m_arr[c, k] = dp_face * cell_areas[c] * inv_g
    end
    return nothing
end

function compute_reduced_horizontal_fluxes!(hflux::AbstractVector{Float64},
                                            lnsp_center::Vector{Float64},
                                            u_spec::AbstractMatrix{ComplexF64},
                                            v_spec::AbstractMatrix{ComplexF64},
                                            T::Int,
                                            dA_k::Float64,
                                            dB_k::Float64,
                                            grid::ReducedGaussianTargetGeometry,
                                            half_dt::Float64,
                                            cache::ReducedSpectralThreadCache)
    mesh = grid.mesh
    R_g = mesh.radius / STANDARD_GRAVITY
    fill!(hflux, 0.0)

    @inbounds for j in 1:nrings(mesh)
        nlon = mesh.nlon_per_ring[j]
        ring_vals = _real_buffer!(cache, nlon)
        spectral_to_ring!(ring_vals, u_spec, T, grid.lats[j], cache; lon_shift_rad=0.0)
        ring_start = mesh.ring_offsets[j]
        cos_lat = cosd(grid.lats[j])
        dlat = deg2rad(mesh.lat_faces[j + 1] - mesh.lat_faces[j])
        for i in 1:nlon
            face = ring_start + i - 1
            left = ring_start + (i == 1 ? nlon - 1 : i - 2)
            right = ring_start + i - 1
            ps_face = exp((lnsp_center[left] + lnsp_center[right]) / 2)
            dp_face = abs(dA_k + dB_k * ps_face)
            hflux[face] = ring_vals[i] / cos_lat * dp_face * R_g * dlat * half_dt
        end
    end

    @inbounds for b in 2:nrings(mesh)
        nseg = mesh.boundary_counts[b]
        seg_vals = _real_buffer!(cache, nseg)
        spectral_to_reduced_boundary!(seg_vals, v_spec, T, mesh.lat_faces[b], cache)
        south_ring = b - 1
        north_ring = b
        nlon_s = mesh.nlon_per_ring[south_ring]
        nlon_n = mesh.nlon_per_ring[north_ring]
        dlon = 2pi / nseg
        face0 = mesh._ncells + mesh.boundary_offsets[b] - 1
        for seg in 1:nseg
            face = face0 + seg
            south_i = ((seg - 1) * nlon_s) ÷ nseg + 1
            north_i = ((seg - 1) * nlon_n) ÷ nseg + 1
            south = mesh.ring_offsets[south_ring] + south_i - 1
            north = mesh.ring_offsets[north_ring] + north_i - 1
            ps_face = exp((lnsp_center[south] + lnsp_center[north]) / 2)
            dp_face = abs(dA_k + dB_k * ps_face)
            hflux[face] = seg_vals[seg] * dp_face * R_g * dlon * half_dt
        end
    end

    return nothing
end

"""
    _project_mean_zero!(v)

Subtract the mean of `v` in-place. Used to keep Conjugate-Gradient
iterates in the range of the singular graph Laplacian (which has a
1-dimensional constant null space on the reduced-Gaussian mesh with
interior-only degree).
"""
@inline function _project_mean_zero!(v::AbstractVector{Float64})
    s = sum(v) / length(v)
    @. v -= s
    return v
end

function recompute_faceindexed_cm_from_divergence!(cm::AbstractMatrix{FT},
                                                   hflux::AbstractMatrix{FT},
                                                   face_left::Vector{Int32},
                                                   face_right::Vector{Int32},
                                                   div_scratch::AbstractMatrix{Float64};
                                                   B_ifc::Vector{<:Real}=Float64[]) where FT
    nc = size(cm, 1)
    Nz = size(cm, 2) - 1
    fill!(cm, zero(FT))
    fill!(div_scratch, 0.0)

    @inbounds for k in 1:Nz, f in eachindex(face_left)
        flux = Float64(hflux[f, k])
        left = Int(face_left[f])
        right = Int(face_right[f])
        left > 0 && (div_scratch[left, k] += flux)
        right > 0 && (div_scratch[right, k] -= flux)
    end

    if !isempty(B_ifc) && length(B_ifc) == Nz + 1
        @inbounds for c in 1:nc
            pit = 0.0
            for k in 1:Nz
                pit += div_scratch[c, k]
            end
            acc = 0.0
            for k in 1:Nz
                acc = acc - div_scratch[c, k] + (Float64(B_ifc[k + 1]) - Float64(B_ifc[k])) * pit
                cm[c, k + 1] = FT(acc)
            end
        end
    else
        @inbounds for c in 1:nc
            acc = 0.0
            for k in 1:Nz
                acc = acc - div_scratch[c, k]
                cm[c, k + 1] = FT(acc)
            end
        end
    end
    return nothing
end

function spectral_to_native_fields!(work::ReducedTransformWorkspace,
                                    lnsp_spec::Matrix{ComplexF64},
                                    vo_hour::Array{ComplexF64, 3},
                                    d_hour::Array{ComplexF64, 3},
                                    T::Int,
                                    level_range::UnitRange{Int},
                                    ab,
                                    grid::ReducedGaussianTargetGeometry,
                                    half_dt::Float64)
    cache1 = work.caches[1]
    spectral_to_reduced_scalar!(work.lnsp, lnsp_spec, T, grid, cache1; centered=true)
    @. work.sp = exp(work.lnsp)
    compute_reduced_dp_and_mass!(work.dp, work.m_arr, work.sp, work.cell_areas, ab.dA, ab.dB)

    Nz = length(level_range)
    Threads.@threads :static for kk in 1:Nz
        level = level_range[kk]
        cache = work.caches[Threads.threadid()]
        vod2uv!(cache.u_spec, cache.v_spec,
                @view(vo_hour[:, :, level]),
                @view(d_hour[:, :, level]),
                T)
        compute_reduced_horizontal_fluxes!(@view(work.hflux_arr[:, kk]),
                                           work.lnsp,
                                           cache.u_spec,
                                           cache.v_spec,
                                           T,
                                           Float64(ab.dA[kk]),
                                           Float64(ab.dB[kk]),
                                           grid,
                                           half_dt,
                                           cache)
    end

    recompute_faceindexed_cm_from_divergence!(work.cm_arr, work.hflux_arr,
                                              work.face_left, work.face_right,
                                              work.div_scratch; B_ifc=ab.b_ifc)
    return nothing
end

function merge_field_2d!(merged::AbstractMatrix{FT}, native::AbstractMatrix{FT}, mm::Vector{Int}) where FT
    fill!(merged, zero(FT))
    @inbounds for k in 1:length(mm)
        @views merged[:, mm[k]] .+= native[:, k]
    end
    return nothing
end

function merge_reduced_window!(merged::ReducedMergeWorkspace{FT},
                               native::ReducedTransformWorkspace,
                               vertical) where FT
    @. merged.m_native_ft = FT(native.m_arr)
    @. merged.hflux_native_ft = FT(native.hflux_arr)
    merge_field_2d!(merged.m_merged, merged.m_native_ft, vertical.merge_map)
    merge_field_2d!(merged.hflux_merged, merged.hflux_native_ft, vertical.merge_map)
    recompute_faceindexed_cm_from_divergence!(merged.cm_merged,
                                              merged.hflux_merged,
                                              native.face_left,
                                              native.face_right,
                                              merged.div_scratch;
                                              B_ifc=vertical.merged_vc.B)
    return nothing
end

"""
    SlidingWindowBuffer{FT}

Two-slot circular buffer for streaming RG preprocessing.  Only two windows'
worth of `(m, hflux, cm, ps)` are kept in memory at any time, enabling
O160/O320 binary generation without OOM.
"""
struct SlidingWindowBuffer{FT}
    m     :: Vector{Matrix{FT}}     # length-2 circular buffer
    hflux :: Vector{Matrix{FT}}
    cm    :: Vector{Matrix{FT}}
    ps    :: Vector{Vector{FT}}
end

function allocate_sliding_window_buffer(nc::Int, nf::Int, Nz::Int, ::Type{FT}) where FT
    SlidingWindowBuffer{FT}(
        [zeros(FT, nc, Nz)     for _ in 1:2],
        [zeros(FT, nf, Nz)     for _ in 1:2],
        [zeros(FT, nc, Nz + 1) for _ in 1:2],
        [zeros(FT, nc)         for _ in 1:2],
    )
end

"""
    fill_buffer_slot!(buf, slot, merged, ps_vec, FT)

Copy merged results into the given slot (1 or 2) of the sliding buffer.
"""
function fill_buffer_slot!(buf::SlidingWindowBuffer{FT},
                           slot::Int,
                           merged::ReducedMergeWorkspace{FT},
                           ps_vec::AbstractVector) where FT
    copyto!(buf.m[slot],     merged.m_merged)
    copyto!(buf.hflux[slot], merged.hflux_merged)
    copyto!(buf.cm[slot],    merged.cm_merged)
    buf.ps[slot] .= ps_vec   # broadcast handles Float64→FT conversion in-place
    return nothing
end

mutable struct ReducedGaussianSpectralWindowWorkspace{FT, TW, MW, BW, QW, CL}
    work           :: TW
    merged         :: MW
    buf            :: BW
    qv_ws          :: QW
    thermo_path    :: String
    ps_offsets     :: Vector{Float64}
    cL             :: CL
    hflux_work     :: Matrix{Float64}
    m_cur_work     :: Matrix{Float64}
    m_next_work    :: Matrix{Float64}
    cm_work        :: Matrix{Float64}
    div_scratch    :: Matrix{Float64}
    dm_target_work :: Matrix{Float64}
    steps_schedule :: Vector{Int}
    cur            :: Int
    nxt            :: Int
end

_rg_compressed_laplacian_cache_key(grid::ReducedGaussianTargetGeometry) =
    Symbol("rg_compressed_laplacian_", grid.gaussian_number)

function _get_or_build_rg_compressed_laplacian!(cache,
                                                grid::ReducedGaussianTargetGeometry,
                                                work::ReducedTransformWorkspace,
                                                nc::Int,
                                                nf::Int)
    cache_key = _rg_compressed_laplacian_cache_key(grid)
    cached = cache === nothing ? nothing : get(cache, cache_key, nothing)
    t_cl = time()
    if cached !== nothing
        cL = cached
        total_entries = cL.row_ptr[end] - 1
        avg_neighbors = total_entries / nc
        @info @sprintf("  Compressed Laplacian: reused %d unique entries (avg %.1f neighbors/cell) from %d faces (%.0f× compression, %.2fs)",
                       total_entries, avg_neighbors, nf,
                       nf / max(total_entries, 1), time() - t_cl)
        return cL
    end

    cL = build_compressed_laplacian(work.face_left, work.face_right, nc)
    cache === nothing || (cache[cache_key] = cL)
    total_entries = cL.row_ptr[end] - 1
    avg_neighbors = total_entries / nc
    @info @sprintf("  Compressed Laplacian: %d unique entries (avg %.1f neighbors/cell) from %d faces (%.0f× compression, %.2fs)",
                   total_entries, avg_neighbors, nf,
                   nf / max(total_entries, 1), time() - t_cl)
    return cL
end

function allocate_window_workspace(grid::ReducedGaussianTargetGeometry,
                                   settings,
                                   vertical,
                                   spec,
                                   date::Date,
                                   ::Type{FT};
                                   cache = nothing) where FT
    mesh = grid.mesh
    nc = ncells(mesh)
    nf = nfaces(mesh)
    Nz_native = vertical.Nz_native
    Nz = vertical.Nz

    work = allocate_reduced_transform_workspace(grid, spec.T, Nz_native)
    merged = allocate_reduced_merge_workspace(grid, Nz_native, Nz, FT)
    buf = allocate_sliding_window_buffer(nc, nf, Nz, FT)
    ps_offsets = zeros(Float64, spec.n_times + 1)   # the last entry: the next day's 00 UTC

    thermo_path = ""
    qv_ws = nothing
    if settings.mass_basis == :dry
        date_str = Dates.format(date, "yyyymmdd")
        thermo_path = joinpath(settings.thermo_dir,
                               "era5_thermo_ml_$(date_str).nc")
        isfile(thermo_path) ||
            error("Thermo file not found for dry-basis conversion: $thermo_path")
        qv_ws = allocate_reduced_qv_workspace(grid, Nz_native, thermo_path;
                                              settings = settings)
        @info "  Dry-basis: QV from $thermo_path → $(qv_ws.Nx_ll)×$(qv_ws.Ny_ll) LL → $(nc) RG cells"
    end

    cL = _get_or_build_rg_compressed_laplacian!(cache, grid, work, nc, nf)

    hflux_work = zeros(Float64, nf, Nz)
    m_cur_work = zeros(Float64, nc, Nz)
    m_next_work = zeros(Float64, nc, Nz)
    cm_work = zeros(Float64, nc, Nz + 1)
    div_scratch = zeros(Float64, nc, Nz)
    dm_target_work = zeros(Float64, nc, Nz)

    return ReducedGaussianSpectralWindowWorkspace{
        FT, typeof(work), typeof(merged), typeof(buf), typeof(qv_ws), typeof(cL)}(
            work, merged, buf, qv_ws, thermo_path, ps_offsets, cL,
            hflux_work, m_cur_work, m_next_work, cm_work, div_scratch,
            dm_target_work, fill(exact_steps_per_window(settings.met_interval,
                                                         settings.dt),
                                  spec.n_times),
            1, 2)
end

function ingest_window!(workspace::ReducedGaussianSpectralWindowWorkspace,
                        slot::Int,
                        win_idx::Int,
                        hour::Int,
                        spec,
                        grid::ReducedGaussianTargetGeometry,
                        vertical,
                        settings)
    t0 = time()
    synthesize_and_merge_window!(workspace.work, workspace.merged, hour, spec,
                                 grid, vertical, settings,
                                 workspace.ps_offsets, win_idx;
                                 qv_ws = workspace.qv_ws,
                                 thermo_path = workspace.thermo_path)
    fill_buffer_slot!(workspace.buf, slot, workspace.merged, workspace.work.sp)
    return time() - t0
end

function drain_ready_windows!(workspace::ReducedGaussianSpectralWindowWorkspace{FT},
                              contract,
                              win_idx::Int,
                              steps_per_window::Int;
                              write_replay_on::Bool,
                              substep_policy) where FT
    _ = steps_per_window
    steps = initial_substeps(substep_policy, workspace.steps_schedule[win_idx])
    old_steps = workspace.steps_schedule[win_idx]
    diag = nothing
    contract_diag = nothing
    while true
        old_steps == steps ||
            rescale_substep_amounts!((workspace.buf.hflux[workspace.cur],),
                                     old_steps, steps)
        old_steps = steps
        workspace.steps_schedule[win_idx] = steps
        contract.steps_per_window = steps
        diag = balance_window!(workspace.hflux_work, workspace.m_cur_work,
                               workspace.m_next_work, workspace.cm_work,
                               workspace.div_scratch, workspace.dm_target_work,
                               workspace.buf, workspace.cur,
                               workspace.buf.m[workspace.nxt],
                               workspace.work, workspace.cL, steps)
        contract_diag = _verify_rg_balanced_window!(
            contract, workspace.m_cur_work, workspace.hflux_work, workspace.cm_work,
            workspace.m_next_work, win_idx; write_replay_on = write_replay_on,
            accumulate = false)
        next_steps = next_substeps(substep_policy, steps,
                                   contract_diag.positivity.ratio)
        next_steps == steps && break
        steps = next_steps
    end
    update_accumulator!(contract, contract_diag.positivity, win_idx)
    ready = ReadyWindow{ReducedGaussianTargetGeometry, FT}(
        win_idx,
        (m = workspace.buf.m[workspace.cur],
         hflux = workspace.buf.hflux[workspace.cur],
         cm = workspace.buf.cm[workspace.cur],
         ps = workspace.buf.ps[workspace.cur]))
    workspace.cur, workspace.nxt = workspace.nxt, workspace.cur
    return (ready = ready, balance = diag, contract = contract_diag)
end

function flush_final_windows!(workspace::ReducedGaussianSpectralWindowWorkspace{FT},
                              contract,
                              win_idx::Int,
                              steps_per_window::Int;
                              write_replay_on::Bool,
                              substep_policy,
                              m_next = workspace.buf.m[workspace.cur]) where FT
    _ = steps_per_window
    steps = initial_substeps(substep_policy, workspace.steps_schedule[win_idx])
    old_steps = workspace.steps_schedule[win_idx]
    diag = nothing
    contract_diag = nothing
    while true
        old_steps == steps ||
            rescale_substep_amounts!((workspace.buf.hflux[workspace.cur],),
                                     old_steps, steps)
        old_steps = steps
        workspace.steps_schedule[win_idx] = steps
        contract.steps_per_window = steps
        diag = balance_window!(workspace.hflux_work, workspace.m_cur_work,
                               workspace.m_next_work, workspace.cm_work,
                               workspace.div_scratch, workspace.dm_target_work,
                               workspace.buf, workspace.cur, m_next,
                               workspace.work, workspace.cL, steps)
        contract_diag = _verify_rg_balanced_window!(
            contract, workspace.m_cur_work, workspace.hflux_work, workspace.cm_work,
            workspace.m_next_work, win_idx; write_replay_on = write_replay_on,
            accumulate = false)
        next_steps = next_substeps(substep_policy, steps,
                                   contract_diag.positivity.ratio)
        next_steps == steps && break
        steps = next_steps
    end
    update_accumulator!(contract, contract_diag.positivity, win_idx)
    ready = ReadyWindow{ReducedGaussianTargetGeometry, FT}(
        win_idx,
        (m = workspace.buf.m[workspace.cur],
         hflux = workspace.buf.hflux[workspace.cur],
         cm = workspace.buf.cm[workspace.cur],
         ps = workspace.buf.ps[workspace.cur]))
    return (ready = ready, balance = diag, contract = contract_diag)
end

# =========================================================================
# RG window synthesis and process_day — mirrors the LL path
# =========================================================================

"""
    synthesize_and_merge_window!(work, merged, hour, spec, grid, vertical,
                                 settings, ps_offsets, win_idx;
                                 qv_ws=nothing, thermo_path="")

Spectral synthesis → native fields → mass fix → dry conversion → level merge
for one window. Results are left in `merged.m_merged`, `merged.hflux_merged`,
`merged.cm_merged` and `work.sp` (surface pressure). No allocation.

When `qv_ws` is provided and `settings.mass_basis == :dry`, loads QV from the
thermo file and interpolates it to RG cells; the window is then finished by
[`pin_convert_merge_window!`](@ref).
"""
function synthesize_and_merge_window!(work::ReducedTransformWorkspace,
                                      merged::ReducedMergeWorkspace{FT},
                                      hour::Int,
                                      spec,
                                      grid::ReducedGaussianTargetGeometry,
                                      vertical,
                                      settings,
                                      ps_offsets::Vector{Float64},
                                      win_idx::Int;
                                      qv_ws::Union{ReducedQVWorkspace, Nothing}=nothing,
                                      thermo_path::String="") where FT
    spectral_to_native_fields!(work,
        spec.lnsp_all[hour], spec.vo_by_hour[hour], spec.d_by_hour[hour],
        spec.T, vertical.level_range, vertical.ab, grid, settings.half_dt)
    dry = settings.mass_basis == :dry && qv_ws !== nothing
    dry && load_rg_qv!(qv_ws, thermo_path, win_idx, vertical.Nz_native)
    pin_convert_merge_window!(work, merged, vertical, settings, ps_offsets, win_idx, dry ? qv_ws : nothing)
    return nothing
end

"""
    synthesize_next_day_hour0!(work, merged, next_day_hour0, date, grid, vertical,
                               settings, ps_offsets; qv_ws=nothing)

The next day's 00 UTC state, the end point of the day's last window, through
the same synthesis, pin (offset stored in `ps_offsets[end]`), dry conversion
(the next day's thermo file, time index 1) and merge as every window.
"""
function synthesize_next_day_hour0!(work::ReducedTransformWorkspace,
                                    merged::ReducedMergeWorkspace,
                                    next_day_hour0,
                                    date::Date,
                                    grid::ReducedGaussianTargetGeometry,
                                    vertical,
                                    settings,
                                    ps_offsets::Vector{Float64};
                                    qv_ws::Union{ReducedQVWorkspace, Nothing}=nothing)
    spectral_to_native_fields!(work, next_day_hour0.lnsp, next_day_hour0.vo, next_day_hour0.d,
                               next_day_hour0.T, vertical.level_range, vertical.ab, grid,
                               settings.half_dt)
    dry = settings.mass_basis == :dry && qv_ws !== nothing
    if dry
        path = joinpath(settings.thermo_dir,
                        "era5_thermo_ml_$(Dates.format(date + Day(1), "yyyymmdd")).nc")
        isfile(path) || error("Thermo file not found for the next-day endpoint: $path")
        qv_ws.qv_ll .= read_qv_from_thermo(path, 1, qv_ws.Nx_ll, qv_ws.Ny_ll, vertical.Nz_native;
                                           FT = Float64)
        _interpolate_ll_to_rg!(qv_ws)
    end
    pin_convert_merge_window!(work, merged, vertical, settings, ps_offsets, length(ps_offsets),
                              dry ? qv_ws : nothing)
    return nothing
end

"""
    pin_convert_merge_window!(work, merged, vertical, settings, ps_offsets, slot, qv_ws)

Finish a synthesized window: the global mass fix (offset stored in
`ps_offsets[slot]`), the dry-basis conversion when `qv_ws` holds the window's
humidity, and the level merge.

With humidity the fix pins the global dry surface pressure, as the lat-lon and
cubed-sphere paths do: the vertical-flux closure needs the same global dry
mass at both ends of a window. Without humidity it pins the total surface
pressure with the climatological `qv_global_climatology`. The horizontal
fluxes follow the pinned pressure: each is wind × Δp at the face, so it scales
by the ratio of the face Δp after and before the pin.
"""
function pin_convert_merge_window!(work::ReducedTransformWorkspace,
                                   merged::ReducedMergeWorkspace,
                                   vertical,
                                   settings,
                                   ps_offsets::Vector{Float64},
                                   slot::Int,
                                   qv_ws::Union{ReducedQVWorkspace, Nothing})
    if settings.mass_fix_enable
        ab = vertical.ab
        ps_offsets[slot] = qv_ws === nothing ?
            pin_global_mean_ps!(work.sp, work.cell_areas; target_ps_dry_pa = settings.target_ps_dry_pa,
                                qv_global = settings.qv_global_climatology) :
            pin_global_mean_ps_using_qv!(work.sp, work.cell_areas, ab.dA, ab.dB, qv_ws.qv_cell;
                                         target_ps_dry_pa = settings.target_ps_dry_pa)
        compute_reduced_dp_and_mass!(work.dp, work.m_arr, work.sp, work.cell_areas, ab.dA, ab.dB)
        rescale_reduced_fluxes_to_ps!(work.hflux_arr, work.lnsp, work.sp, work.face_left,
                                      work.face_right, ab.dA, ab.dB)
        @. work.lnsp = log(work.sp)
    end
    qv_ws === nothing || apply_dry_basis_reduced!(work, qv_ws.qv_cell)
    merge_reduced_window!(merged, work, vertical)
    return nothing
end

"""
    rescale_reduced_fluxes_to_ps!(hflux, lnsp_old, sp_new, face_left, face_right, dA, dB)

Scale the face mass fluxes `hflux[f, k]` (wind × Δp at the face) from the
surface pressure `exp.(lnsp_old)` to `sp_new`: the face pressure is the
geometric mean of the two cells' (as in `compute_reduced_horizontal_fluxes!`)
and Δp = |dA_k + dB_k p|. Polar stub faces (a zero neighbour) are set to zero.
"""
function rescale_reduced_fluxes_to_ps!(hflux::AbstractMatrix{Float64},
                                       lnsp_old::AbstractVector{Float64},
                                       sp_new::AbstractVector{Float64},
                                       face_left::AbstractVector{<:Integer},
                                       face_right::AbstractVector{<:Integer},
                                       dA::AbstractVector, dB::AbstractVector)
    @inbounds for k in axes(hflux, 2), f in axes(hflux, 1)
        left, right = face_left[f], face_right[f]
        if left == 0 || right == 0
            hflux[f, k] = 0.0
            continue
        end
        p_old = exp((lnsp_old[left] + lnsp_old[right]) / 2)
        p_new = sqrt(sp_new[left] * sp_new[right])
        dp_old = abs(dA[k] + dB[k] * p_old)
        dp_old > 0 && (hflux[f, k] *= abs(dA[k] + dB[k] * p_new) / dp_old)
    end
    return hflux
end

"""
    balance_window!(hflux_work, m_cur_work, m_next_work, cm_work, div_scratch,
                    dm_target_work, buf, slot, m_next, work, cL, steps_per_window;
                    tol, max_iter)

Poisson-balance the horizontal fluxes in buffer `slot` using the
compressed Laplacian `cL` for the CG solver.  Mathematically identical
to the old face-indexed CG but ~16-27× faster because the compressed
MatVec iterates over ~4×ncells entries instead of nfaces (millions).

All scratch arrays are preallocated Float64 buffers — no per-call allocation.
Returns a diagnostics NamedTuple.
"""
function balance_window!(hflux_work::Matrix{Float64},
                         m_cur_work::Matrix{Float64},
                         m_next_work::Matrix{Float64},
                         cm_work::Matrix{Float64},
                         div_scratch::Matrix{Float64},
                         dm_target_work::Matrix{Float64},
                         buf::SlidingWindowBuffer{FT},
                         slot::Int,
                         m_next::AbstractMatrix{FT},
                         work::ReducedTransformWorkspace,
                         cL::CompressedLaplacian,
                         steps_per_window::Int;
                         tol::Float64 = 1e-14,
                         max_iter::Int = 20000) where FT

    copyto!(hflux_work, buf.hflux[slot])
    copyto!(m_cur_work, buf.m[slot])
    copyto!(m_next_work, m_next)

    scratch = (psi = work.balance_psi, rhs = work.balance_rhs,
               r = work.balance_r, p = work.balance_p,
               Ap = work.balance_Ap, z = work.balance_z)
    replay_layout = faceindexed_replay_layout(work.face_left, work.face_right)

    diag = balance_compressed_horizontal_fluxes!(
        hflux_work, m_cur_work, m_next_work,
        work.face_left, work.face_right, cL,
        steps_per_window, scratch; tol=tol, max_iter=max_iter)

    # Store balanced hflux back as FT.
    buf.hflux[slot] .= FT.(hflux_work)

    # Recompute cm from balanced hflux using the explicit-dm closure. The
    # Δb × pit closure does not hold on the dry basis: dm[k] = dB[k] Σ dm
    # holds only for moist hybrid coordinates, and the humidity profile
    # breaks it by ~27%.
    size(dm_target_work) == size(m_cur_work) ||
        error("balance_window!: dm_target_work shape $(size(dm_target_work)) " *
              "!= m_cur_work shape $(size(m_cur_work)).")
    inv_scale = 1.0 / (2 * max(Int(steps_per_window), 1))
    @. dm_target_work = (m_next_work - m_cur_work) * inv_scale
    recompute_cm_from_dm_target!(replay_layout, div_scratch, cm_work,
                                 m_cur_work, dm_target_work, hflux_work)
    buf.cm[slot] .= FT.(cm_work)

    return diag
end

function _verify_rg_balanced_window!(window_contract,
                                     m_cur_work::Matrix{Float64},
                                     hflux_work::Matrix{Float64},
                                     cm_work::Matrix{Float64},
                                     m_next_work::Matrix{Float64},
                                     win_idx::Int;
                                     write_replay_on::Bool,
                                     accumulate::Bool = true)
    if write_replay_on
        diag = verify_window!((m_cur = m_cur_work,
                               hflux = hflux_work,
                               cm = cm_work,
                               m_next = m_next_work),
                              window_contract, win_idx)
    else
        stub = verify_boundary_stub_flux_rg(hflux_work,
                                             window_contract.face_left,
                                             window_contract.face_right;
                                             tol = window_contract.boundary_stub_tol)
        stub.violated &&
            error("Boundary-stub flux gate FAILED for RG window $(win_idx): " *
                  "hflux=$(stub.worst_flux) on face=$(stub.worst_face) " *
                  "level=$(stub.worst_level) where " *
                  "face_left=$(window_contract.face_left[stub.worst_face]) " *
                  "face_right=$(window_contract.face_right[stub.worst_face]); " *
                  "runtime advection (`StrangSplitting.jl:279`) will silently " *
                  "discard this flux.")

        m_shape = size(m_cur_work)
        if window_contract._outgoing_h === nothing ||
           size(window_contract._outgoing_h) != m_shape
            window_contract._outgoing_h = Array{Float64}(undef, m_shape)
            window_contract._bad_h = Array{Bool}(undef, m_shape)
        end
        positivity = verify_substep_positivity_rg!(m_cur_work, hflux_work, cm_work,
                                                    window_contract.face_left,
                                                    window_contract.face_right;
                                                    cfl_limit =
                                                        window_contract.positivity_cfl_limit,
                                                    outgoing_h =
                                                        window_contract._outgoing_h,
                                                    bad_h =
                                                        window_contract._bad_h)
        diag = (replay = nothing, positivity = positivity)
    end

    accumulate && update_accumulator!(window_contract, diag.positivity, win_idx)
    return diag
end

mutable struct RGSpectralUnifiedDriverContext{G, S, V, SP, P, N}
    grid              :: G
    settings          :: S
    vertical          :: V
    spec              :: SP
    substep_policy    :: P
    date              :: Date
    next_day_hour0    :: N          # next day's 00 UTC spectral state, or nothing
    steps_per_window  :: Int
    write_replay_on   :: Bool
    worst_pre_raw     :: Float64
    worst_post_proj   :: Float64
    worst_post_raw    :: Float64
    worst_iter        :: Int
    worst_replay_rel  :: Float64
    worst_replay_abs  :: Float64
    worst_replay_win  :: Int
    worst_replay_idx  :: Tuple{Int, Int}
end

driver_windows_per_day(::Nothing, ctx::RGSpectralUnifiedDriverContext) =
    ctx.spec.n_times

function driver_ingest_window!(workspace::ReducedGaussianSpectralWindowWorkspace,
                               ::Nothing,
                               win::Int,
                               ctx::RGSpectralUnifiedDriverContext)
    slot = win == 1 ? workspace.cur : workspace.nxt
    t_synth = ingest_window!(workspace, slot, win, ctx.spec.hours[win],
                             ctx.spec, ctx.grid, ctx.vertical, ctx.settings)
    should_log_window(win, ctx.spec.n_times) &&
        @info @sprintf("    Window %2d/%d (hour %02d): synth %.2fs  offset=%+.3f Pa",
                       win, ctx.spec.n_times, ctx.spec.hours[win], t_synth,
                       workspace.ps_offsets[win])
    return nothing
end

function _rg_unified_record_diag!(ctx::RGSpectralUnifiedDriverContext,
                                  ready_diag,
                                  win::Int)
    diag = ready_diag.balance
    ctx.worst_pre_raw = max(ctx.worst_pre_raw, diag.max_pre_raw_residual)
    ctx.worst_post_proj = max(ctx.worst_post_proj, diag.max_post_projected)
    ctx.worst_post_raw = max(ctx.worst_post_raw, diag.max_post_raw_residual)
    ctx.worst_iter = max(ctx.worst_iter, diag.max_cg_iter)

    contract_diag = ready_diag.contract
    if ctx.write_replay_on && contract_diag.replay.max_rel_err > ctx.worst_replay_rel
        ctx.worst_replay_rel = contract_diag.replay.max_rel_err
        ctx.worst_replay_abs = contract_diag.replay.max_abs_err
        ctx.worst_replay_win = win
        ctx.worst_replay_idx = contract_diag.replay.worst_idx
    end
    return nothing
end

function driver_drain_ready_windows!(workspace::ReducedGaussianSpectralWindowWorkspace,
                                     contract,
                                     win::Int,
                                     ctx::RGSpectralUnifiedDriverContext)
    win == 1 && return ()
    written_win = win - 1
    t_bal = time()
    ready_diag = drain_ready_windows!(workspace, contract, written_win,
                                      ctx.steps_per_window;
                                      write_replay_on = ctx.write_replay_on,
                                      substep_policy = ctx.substep_policy)
    t_bal = time() - t_bal
    _rg_unified_record_diag!(ctx, ready_diag, written_win)
    diag = ready_diag.balance
    should_log_window(written_win, ctx.spec.n_times) &&
        @info @sprintf("    Window %2d/%d: wrote (bal %.2fs pre_raw=%.2e post_proj=%.2e iter=%d)",
                       written_win, ctx.spec.n_times, t_bal,
                       diag.max_pre_raw_residual, diag.max_post_projected,
                       diag.max_cg_iter)
    return (PreverifiedWindow(ready_diag.ready, ready_diag.contract;
                              accumulated = true),)
end

function driver_flush_final_windows!(workspace::ReducedGaussianSpectralWindowWorkspace,
                                     ::Nothing,
                                     contract,
                                     ctx::RGSpectralUnifiedDriverContext)
    Nt = ctx.spec.n_times
    # The last window ends at the next day's 00 UTC; without it the window keeps
    # its own mass at both ends (zero tendency).
    m_next = workspace.buf.m[workspace.cur]
    if ctx.next_day_hour0 !== nothing
        synthesize_next_day_hour0!(workspace.work, workspace.merged, ctx.next_day_hour0, ctx.date,
                                   ctx.grid, ctx.vertical, ctx.settings, workspace.ps_offsets;
                                   qv_ws = workspace.qv_ws)
        fill_buffer_slot!(workspace.buf, workspace.nxt, workspace.merged, workspace.work.sp)
        m_next = workspace.buf.m[workspace.nxt]
    end
    t_bal = time()
    ready_diag = flush_final_windows!(workspace, contract, Nt,
                                      ctx.steps_per_window;
                                      write_replay_on = ctx.write_replay_on,
                                      substep_policy = ctx.substep_policy,
                                      m_next)
    t_bal = time() - t_bal
    _rg_unified_record_diag!(ctx, ready_diag, Nt)
    diag = ready_diag.balance
    @info @sprintf("    Window %2d/%d (last): bal %.2fs  pre_raw=%.2e post_proj=%.2e iter=%d",
                   Nt, Nt, t_bal, diag.max_pre_raw_residual,
                   diag.max_post_projected, diag.max_cg_iter)
    return (PreverifiedWindow(ready_diag.ready, ready_diag.contract;
                              accumulated = true),)
end

function driver_before_close_writer!(workspace::ReducedGaussianSpectralWindowWorkspace,
                                     ::Nothing,
                                     contract,
                                     writer,
                                     ::RGSpectralUnifiedDriverContext)
    set_streaming_steps_per_window_schedule!(writer.inner, workspace.steps_schedule)
    writer.inner.header["ps_offsets_pa_per_window"] = workspace.ps_offsets[1:end - 1]
    writer.inner.header["ps_offsets_next_day_hour0_pa"] = workspace.ps_offsets[end]
    set_contract_steps_schedule!(contract, workspace.steps_schedule)
    return nothing
end

"""
    process_day(date, grid::ReducedGaussianTargetGeometry, settings, vertical;
                next_day_hour0=nothing, positivity_cfl_limit=0.95,
                require_substep_positivity=true)

Streaming one-day preprocessing for reduced-Gaussian targets.

Uses a 2-window sliding buffer: at any time only two windows' worth of
`(m, hflux, cm, ps)` are held in memory.  Each window is Poisson-balanced
and written to disk before the next pair is computed.  This reduces peak
memory from `O(Nt)` to `O(1)` and enables O160/O320 binary generation.

Pipeline per window:
  spectral synthesis → mass fix → level merge → (wait for next window) →
  Poisson balance using (m_cur, m_next) → cm recomputation → stream-write
"""
function process_day(date::Date,
                     grid::ReducedGaussianTargetGeometry,
                     settings::ERA5SpectralSettings,
                     vertical;
                     next_day_hour0=nothing,
                     positivity_cfl_limit::Real = 0.95,
                     require_substep_positivity::Bool = true,
                     substep_policy =
                         SubstepSchedulePolicy(
                             adaptive_substeps = false,
                             substep_cfl_target = positivity_cfl_limit),
                     run_cache = nothing)
    FT = settings.output_float_type
    settings.include_qv && throw(ArgumentError(
        "reduced-Gaussian preprocessing does not support output.include_qv=true; " *
        "set include_qv=false (humidity may still be used internally for dry-basis conversion)"))
    mesh = grid.mesh
    nc = ncells(mesh)
    nf = nfaces(mesh)
    Nz = vertical.Nz
    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    date_str = Dates.format(date, "yyyymmdd")

    vo_d_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    if !isfile(vo_d_path) || !isfile(lnsp_path)
        @warn "Missing GRIB files for $date_str, skipping"
        return nothing
    end

    t_day = time()
    @info "  Reading spectral data for $date_str..."
    spec = read_day_spectral(vo_d_path, lnsp_path;
                             T_target=settings.T_target,
                             cache_dir=settings.spectral_cache_dir)
    @info @sprintf("  Spectral data read: T=%d, %d hours (%.1fs)",
                   spec.T, spec.n_times, time() - t_day)

    Nt = spec.n_times
    mkpath(settings.out_dir)
    bin_path = output_binary_path(date, settings.out_dir, settings.min_dp, FT)

    workspace = allocate_window_workspace(grid, settings, vertical, spec, date, FT;
                                          cache = run_cache)
    work = workspace.work
    buf = workspace.buf
    ps_offsets = workspace.ps_offsets
    write_replay_on = get(ENV, "ATMOSTR_NO_WRITE_REPLAY_CHECK", "0") != "1"
    write_replay_on ||
        @info "  Write-time replay gate SKIPPED (ATMOSTR_NO_WRITE_REPLAY_CHECK=1)"
    window_contract = ReducedGaussianContract{FT}(
        replay_tol = replay_tolerance(FT),
        positivity_cfl_limit = positivity_cfl_limit,
        require_substep_positivity = require_substep_positivity,
        steps_per_window = steps_per_met,
        face_left = work.face_left,
        face_right = work.face_right,
    )

    # Open the streaming binary writer
    vc_merged = vertical.merged_vc
    transport_grid = AtmosGrid(mesh, vc_merged, CPU(); FT=FT, radius=mesh.radius)
    sample_window = (m = buf.m[1], hflux = buf.hflux[1],
                     cm = buf.cm[1], ps = buf.ps[1])

    # Declare the canonical window_constant contract
    # explicitly. The sliding-window payload stores already balanced,
    # window-constant fluxes and no endpoint deltas, so delta semantics must
    # be `none`. Poisson metadata still records how those fluxes were built.
    rg_contract = canonical_window_constant_contract(
        steps_per_window     = steps_per_met,
        humidity_sampling    = :none,
        source_flux_sampling = :window_start_endpoint,
        include_flux_delta   = false,
    )

    writer = open_streaming_transport_binary(
        bin_path, transport_grid, Nt, sample_window;
        FT = FT,
        dt_met_seconds       = settings.met_interval,
        half_dt_seconds      = settings.half_dt,
        steps_per_window     = steps_per_met,
        source_flux_sampling = rg_contract.source_flux_sampling,
        air_mass_sampling    = rg_contract.air_mass_sampling,
        flux_sampling        = rg_contract.flux_sampling,
        flux_kind            = rg_contract.flux_kind,
        humidity_sampling    = rg_contract.humidity_sampling,
        delta_semantics      = rg_contract.delta_semantics,
        mass_basis           = Symbol(settings.mass_basis),
        extra_header = Dict{String, Any}(
            "preprocessor"     => "preprocess_transport_binary.jl",
            "source_type"      => "era5_spectral",
            "target_type"      => "reduced_gaussian",
            "gaussian_number"  => grid.gaussian_number,
            "poisson_balanced" => true,
            "mass_fix_enabled" => settings.mass_fix_enable,
            "mass_fix_target_ps_dry_pa" => settings.target_ps_dry_pa,
            "mass_fix_qv_global_climatology" => settings.qv_global_climatology,
            "mass_fix_qv_mode" => settings.mass_basis == :dry ? "native_hourly_qv" : "global_qv_climatology",
            # Poisson fields via extra_header — _transport_common_header
            # doesn't accept these directly. Single source
            # of truth: the contract.
            "poisson_balance_target_scale"     => rg_contract.poisson_balance_target_scale,
            "poisson_balance_target_semantics" => rg_contract.poisson_balance_target_semantics,
        ))

    bytes_per_window = writer.elems_per_window * sizeof(eltype(writer.pack_buffer))
    expected_total = writer.header_bytes + Nt * bytes_per_window
    @info @sprintf("  Output: %s (%.2f GB, %d windows, %d cells, %d faces)",
                   basename(bin_path), expected_total / 1e9, Nt, nc, nf)

    log_mass_fix_configuration(settings)
    @info "  Streaming: synthesize → balance → write (2-window sliding buffer)..."

    window_writer = ReducedGaussianBinaryWriter(
        writer,
        mass_basis_from_symbol(Symbol(settings.mass_basis));
        final_path = bin_path)
    ctx = RGSpectralUnifiedDriverContext(
        grid, settings, vertical, spec, substep_policy, date, next_day_hour0,
        steps_per_met, write_replay_on,
        0.0, 0.0, 0.0, 0, 0.0, 0.0, 0, (0, 0))
    run_unified_preprocessor_day!(
        UnifiedPreprocessorDay(nothing, workspace, window_contract, window_writer;
                               context = ctx);
        close_reader = false)

    if settings.mass_fix_enable
        day_offsets = @view ps_offsets[1:Nt]
        @info @sprintf("  Mass-fix offsets (Pa) min/max/mean: %+.3f / %+.3f / %+.3f",
                       minimum(day_offsets), maximum(day_offsets), sum(day_offsets) / Nt)
    end

    if write_replay_on
        @info @sprintf("  Write-time RG replay gate: max rel=%.3e abs=%.3e win=%d cell=%s",
                       ctx.worst_replay_rel, ctx.worst_replay_abs,
                       ctx.worst_replay_win, ctx.worst_replay_idx)
    end

    @info @sprintf("  Poisson balance summary: pre_raw=%.3e  post_proj=%.3e  post_raw=%.3e  max_cg_iter=%d",
                   ctx.worst_pre_raw, ctx.worst_post_proj,
                   ctx.worst_post_raw, ctx.worst_iter)

    actual = filesize(bin_path)
    @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(bin_path),
                   actual / 1e9, time() - t_day)
    actual == expected_total ||
        @warn @sprintf("File size mismatch: expected %d bytes, got %d", expected_total, actual)

    return bin_path
end
