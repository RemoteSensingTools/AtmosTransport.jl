# Reduced-Gaussian spectral preprocessing: workspaces, humidity, spectral synthesis,
# horizontal fluxes and the level merge. The window buffer and the day workflow
# follow in reduced_window_buffer.jl and reduced_spectral_day.jl.

struct ReducedSpectralThreadCache
    P_buf        :: Matrix{Float64}
    fft_buffers  :: Dict{Int, Vector{ComplexF64}}
    real_buffers :: Dict{Int, Vector{Float64}}
    u_spec       :: Matrix{ComplexF64}
    v_spec       :: Matrix{ComplexF64}
end

"""
    ReducedFidelitySampler(grid, face_left, face_right; samples = 256, lines = 32)

Faces of a reduced-Gaussian target sampled by the write-time flux-fidelity
check (`rg_hashed_latitude_lines_v1`). At most `lines` latitude lines are
chosen by the SplitMix64 hash among the rings (zonal faces) and the interior
ring boundaries (meridional faces), then faces within each chosen line, about
`samples` in total. The check re-synthesizes the winds of the chosen lines
only, with the same spectral transform as production (shared code, so this
check does not test the transform itself), and keeps `samples × levels`
values; production synthesizes `2 n_rings − 1` lines.

Per sample it stores the line, the point on the line, the face-table cells
(`face_left`/`face_right` of `Grids.face_cells`: their `ln p_s` set the face
pressure, and they drive the pin rescale and humidity), the edge length factor
and `cos φ` (1 for meridional faces).
"""
struct ReducedFidelitySampler
    faces       :: Vector{Int}
    line_zonal  :: Vector{Bool}
    line_index  :: Vector{Int}           # ring (zonal) or ring boundary (meridional)
    line_slots  :: Vector{Vector{Int}}   # sample slots on each line
    segment     :: Vector{Int}
    face_left   :: Vector{Int32}
    face_right  :: Vector{Int32}
    edge        :: Vector{Float64}
    cos_lat     :: Vector{Float64}
end

function ReducedFidelitySampler(grid::ReducedGaussianTargetGeometry,
                                face_left::AbstractVector{<:Integer},
                                face_right::AbstractVector{<:Integer};
                                samples::Integer = 256, lines::Integer = 32)
    mesh = grid.mesh
    nr = nrings(mesh)
    seed = FLUX_FIDELITY_SAMPLE_SEED
    # Line ids: rings 1..nr (zonal), then interior boundaries 2..nr (meridional).
    line_ids = collect(1:(2nr - 1))
    line_ids = ranked_fidelity_ids(line_ids, seed, 0x5a)
    chosen = sort(line_ids[1:min(Int(lines), length(line_ids))])
    per_line = cld(Int(samples), length(chosen))
    picks = Tuple{Int, Bool, Int, Int}[]          # (face, zonal, line index, segment)
    for id in chosen
        zonal = id <= nr
        index = zonal ? id : id - nr + 1
        npts = zonal ? mesh.nlon_per_ring[index] : mesh.boundary_counts[index]
        first_face = zonal ? mesh.ring_offsets[index] :
                             ncells(mesh) + mesh.boundary_offsets[index]
        segments = sort(collect(1:npts); by = seg -> splitmix64(xor(
            seed, UInt64(first_face + seg - 1), UInt64(0x7f))))
        for seg in sort(segments[1:min(per_line, npts)])
            push!(picks, (first_face + seg - 1, zonal, index, seg))
        end
    end
    sort!(picks; by = first)
    ns = length(picks)
    line_key(zonal, index) = zonal ? index : nr + index - 1
    slot_of_line = Dict(id => Int[] for id in chosen)
    faces = Vector{Int}(undef, ns); segment = Vector{Int}(undef, ns)
    edge = Vector{Float64}(undef, ns); cos_lat = Vector{Float64}(undef, ns)
    for (q, (f, zonal, index, seg)) in pairs(picks)
        faces[q] = f; segment[q] = seg
        push!(slot_of_line[line_key(zonal, index)], q)
        if zonal
            edge[q] = deg2rad(mesh.lat_faces[index + 1] - mesh.lat_faces[index])
            cos_lat[q] = cosd(grid.lats[index])
        else
            count = mesh.boundary_counts[index]
            edge[q] = 2pi / count
            cos_lat[q] = 1.0
        end
    end
    return ReducedFidelitySampler(faces,
        [id <= nr for id in chosen], [id <= nr ? id : id - nr + 1 for id in chosen],
        [slot_of_line[id] for id in chosen], segment,
        Int32.(face_left[faces]), Int32.(face_right[faces]), edge, cos_lat)
end

nsamples(s::ReducedFidelitySampler) = length(s.faces)

struct ReducedTransformWorkspace
    sp             :: Vector{Float64}
    lnsp           :: Vector{Float64}
    dp             :: Matrix{Float64}
    m_arr          :: Matrix{Float64}
    hflux_arr      :: Matrix{Float64}
    fidelity_sampler :: ReducedFidelitySampler
    fidelity_native :: Matrix{Float64}       # (samples, Nz_native), Float64
    cell_areas     :: Vector{Float64}
    face_left      :: Vector{Int32}
    face_right     :: Vector{Int32}
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
    fidelity_native_ft :: Matrix{FT}        # (samples, Nz_native)
    m_merged        :: Matrix{FT}
    hflux_merged    :: Matrix{FT}
    fidelity_merged :: Matrix{FT}           # (samples, Nz)
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
        q = clamp(qv_cell[c, k], 0.0, MAX_SPECIFIC_HUMIDITY)
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
        q_face = clamp(q_face, 0.0, MAX_SPECIFIC_HUMIDITY)
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

    cell_areas = Grids.cell_areas(mesh)
    buffer_lengths = sort!(unique(vcat(collect(mesh.nlon_per_ring), collect(mesh.boundary_counts))))
    face_left, face_right = face_connectivity(mesh)

    caches = [allocate_reduced_spectral_thread_cache(T, buffer_lengths) for _ in 1:nt_max]
    sampler = ReducedFidelitySampler(grid, face_left, face_right)

    return ReducedTransformWorkspace(
        zeros(Float64, nc),
        zeros(Float64, nc),
        zeros(Float64, nc, Nz_native),
        zeros(Float64, nc, Nz_native),
        zeros(Float64, nf, Nz_native),
        sampler,
        zeros(Float64, nsamples(sampler), Nz_native),
        cell_areas,
        face_left,
        face_right,
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
                                          ::Type{FT};
                                          fidelity_samples::Int) where FT
    mesh = grid.mesh
    nc = ncells(mesh)
    nf = nfaces(mesh)
    return ReducedMergeWorkspace{FT}(
        zeros(FT, nc, Nz_native),
        zeros(FT, nf, Nz_native),
        zeros(FT, fidelity_samples, Nz_native),
        zeros(FT, nc, Nz),
        zeros(FT, nf, Nz),
        zeros(FT, fidelity_samples, Nz),
    )
end

@inline function _fft_buffer!(cache::ReducedSpectralThreadCache, n::Int)
    return cache.fft_buffers[n]
end

@inline function _real_buffer!(cache::ReducedSpectralThreadCache, n::Int)
    return cache.real_buffers[n]
end

"""
    spectral_to_ring!(dest, spec, T, lat_deg, cache; lon_shift_rad = 0.0) -> dest

Synthesise the spectral field `spec[n + 1, m + 1] = f̂_n^m` (truncation `T`) on
the `N = length(dest)` equally spaced points `λ_k = lon_shift_rad + 2πk/N` of
the ring at latitude `lat_deg`:

    dest[k + 1] = G_0 + 2 Re Σ_{m=1}^{M} G_m e^{i m λ_k},
    G_m = Σ_{n=m}^{T} f̂_n^m P̃_n^m(sin φ),   M = ring_max_wavenumber(T, N),

through [`fill_ring_spectrum!`](@ref) (shared with `spectral_to_grid!`) and
one unnormalised inverse FFT. The term `m = N/2` of an even ring is dropped
(see [`ring_max_wavenumber`](@ref AtmosTransport.Preprocessing.Spectral.ring_max_wavenumber)).
"""
function spectral_to_ring!(dest::AbstractVector{Float64},
                           spec::AbstractMatrix{ComplexF64},
                           T::Int,
                           lat_deg::Real,
                           cache::ReducedSpectralThreadCache;
                           lon_shift_rad::Real = 0.0)
    Nlon = length(dest)
    compute_legendre_column!(cache.P_buf, T, sind(Float64(lat_deg)))
    fft_buf = _fft_buffer!(cache, Nlon)
    fill_ring_spectrum!(fft_buf, spec, cache.P_buf, T, Float64(lon_shift_rad))
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
                                     cache::ReducedSpectralThreadCache)
    mesh = grid.mesh
    @inbounds for j in 1:nrings(mesh)
        start = mesh.ring_offsets[j]
        stop = mesh.ring_offsets[j + 1] - 1
        shift = pi / mesh.nlon_per_ring[j]
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
        face0 = ncells(mesh) + mesh.boundary_offsets[b] - 1
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
    sample_reduced_horizontal_fluxes!(out, sampler, lnsp_center, u_spec, v_spec, T,
                                      dA_k, dB_k, grid, half_dt, cache)

Scalar re-evaluation of the sampled reduced-grid face amounts of one level for
the write-time fidelity check,
`F = u Δp R Δφ Δt/(g cos φ)` (zonal) and `F = v Δp R Δλ Δt/g` (meridional),
with `Δp = |ΔA + ΔB exp((ln p_{s,a} + ln p_{s,b})/2)|`. Only the sampler's
lines are synthesized; `out` has one value per sample.
"""
function sample_reduced_horizontal_fluxes!(out::AbstractVector{Float64},
                                           s::ReducedFidelitySampler,
                                           lnsp_center::Vector{Float64},
                                           u_spec::AbstractMatrix{ComplexF64},
                                           v_spec::AbstractMatrix{ComplexF64},
                                           T::Int, dA_k::Float64, dB_k::Float64,
                                           grid::ReducedGaussianTargetGeometry,
                                           half_dt::Float64,
                                           cache::ReducedSpectralThreadCache)
    mesh = grid.mesh
    factor = Float64(mesh.radius) * half_dt / Float64(STANDARD_GRAVITY)
    @inbounds for (line, slots) in pairs(s.line_slots)
        index = s.line_index[line]
        if s.line_zonal[line]
            wind = _real_buffer!(cache, mesh.nlon_per_ring[index])
            spectral_to_ring!(wind, u_spec, T, grid.lats[index], cache; lon_shift_rad=0.0)
        else
            wind = _real_buffer!(cache, mesh.boundary_counts[index])
            spectral_to_reduced_boundary!(wind, v_spec, T, mesh.lat_faces[index], cache)
        end
        for q in slots
            ps_face = exp(0.5 * (lnsp_center[s.face_left[q]] + lnsp_center[s.face_right[q]]))
            thickness = abs(dA_k + dB_k * ps_face)
            out[q] = wind[s.segment[q]] * thickness * factor * s.edge[q] / s.cos_lat[q]
        end
    end
    return out
end

# The pin and dry-basis steps of `pin_convert_merge_window!` on the sampled
# values, in the same order and arithmetic as on the full field.
function rescale_reduced_samples_to_ps!(values::AbstractMatrix{Float64},
                                        s::ReducedFidelitySampler,
                                        lnsp_old::AbstractVector{Float64},
                                        sp_new::AbstractVector{Float64},
                                        dA::AbstractVector, dB::AbstractVector)
    return rescale_reduced_fluxes_to_ps!(values, lnsp_old, sp_new,
                                         s.face_left, s.face_right, dA, dB)
end

function apply_dry_basis_reduced_samples!(values::AbstractMatrix{Float64},
                                          s::ReducedFidelitySampler,
                                          qv_cell::AbstractMatrix{Float64})
    @inbounds for k in axes(values, 2), q in axes(values, 1)
        left, right = s.face_left[q], s.face_right[q]
        if left == 0 || right == 0
            values[q, k] = 0.0
        else
            q_face = clamp(0.5 * (qv_cell[left, k] + qv_cell[right, k]), 0.0, MAX_SPECIFIC_HUMIDITY)
            values[q, k] *= 1 - q_face
        end
    end
    return values
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
    spectral_to_reduced_scalar!(work.lnsp, lnsp_spec, T, grid, cache1)
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
        sample_reduced_horizontal_fluxes!(@view(work.fidelity_native[:, kk]),
                                          work.fidelity_sampler,
                                          work.lnsp, cache.u_spec, cache.v_spec,
                                          T, Float64(ab.dA[kk]), Float64(ab.dB[kk]),
                                          grid, half_dt, cache)
    end

    return nothing
end

function merge_field_2d!(merged::AbstractMatrix{FT}, native::AbstractMatrix{FT}, mm::Vector{Int}) where FT
    sum_merged_layers!(merged, native, mm)
    return nothing
end

function merge_reduced_window!(merged::ReducedMergeWorkspace{FT},
                               native::ReducedTransformWorkspace,
                               vertical) where FT
    @. merged.m_native_ft = FT(native.m_arr)
    @. merged.hflux_native_ft = FT(native.hflux_arr)
    @. merged.fidelity_native_ft = FT(native.fidelity_native)
    merge_field_2d!(merged.m_merged, merged.m_native_ft, vertical.merge_map)
    merge_field_2d!(merged.hflux_merged, merged.hflux_native_ft, vertical.merge_map)
    merge_field_2d!(merged.fidelity_merged,
                    merged.fidelity_native_ft, vertical.merge_map)
    return nothing
end

"""Allocate one ring synthesis cache; the caller retains its thread-count policy."""
function allocate_reduced_spectral_thread_cache(T::Int, buffer_lengths)
    nc = T + 1
    return ReducedSpectralThreadCache(
        zeros(Float64, nc, nc),
        Dict(n => zeros(ComplexF64, n) for n in buffer_lengths),
        Dict(n => zeros(Float64, n) for n in buffer_lengths),
        zeros(ComplexF64, nc, nc),
        zeros(ComplexF64, nc, nc))
end
