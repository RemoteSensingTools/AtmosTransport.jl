"""
    merge_cell_field!(merged, native, mm)

Accumulate a native-level cell field onto merged output levels using the
native-to-merged level map `mm`.
"""
function merge_cell_field!(merged::Array{FT, 3}, native::Array{FT, 3}, mm::Vector{Int}) where FT
    sum_merged_layers!(merged, native, mm)
    return nothing
end

@inline vertical_mapping_method(vertical) =
    hasproperty(vertical, :vertical_mapping_method) ? vertical.vertical_mapping_method : :merge_map

"""
    remap_extensive_field_pressure_overlap!(dst, src, ps, src_vc, dst_vc)

Conservatively remap a layer-integrated field from one hybrid coordinate to
another using pressure-thickness overlap in each column.

For a source layer `s` and target layer `k`, interface pressures are evaluated
with the column surface pressure,

```math
p^s_r = A^s_r + B^s_r p_s,\\qquad
p^t_r = A^t_r + B^t_r p_s,
```

and the target layer integral is

```math
X^t_k =
\\sum_s X^s_s
\\frac{\\max(0, \\min(p^s_{s+1}, p^t_{k+1})
              - \\max(p^s_s, p^t_k))}
     {p^s_{s+1} - p^s_s}.
```

`src` and `dst` are extensive in the vertical direction: air mass and
half-step horizontal mass fluxes are both valid inputs. For horizontal fluxes,
pass the face surface pressure field used to construct the flux thickness.
"""
function remap_extensive_field_pressure_overlap!(dst::Array{FT, 3},
                                                 src::Array{FT, 3},
                                                 ps::AbstractMatrix{<:Real},
                                                 src_vc::HybridSigmaPressure,
                                                 dst_vc::HybridSigmaPressure) where FT
    Nx, Ny, Ns = size(src)
    size(dst, 1) == Nx && size(dst, 2) == Ny ||
        throw(DimensionMismatch("dst horizontal shape $(size(dst)[1:2]) != src shape ($Nx, $Ny)"))
    size(ps) == (Nx, Ny) ||
        throw(DimensionMismatch("ps shape $(size(ps)) != field shape ($Nx, $Ny)"))
    Ns == n_levels(src_vc) ||
        throw(DimensionMismatch("src levels $Ns != source coordinate levels $(n_levels(src_vc))"))
    Nt = size(dst, 3)
    Nt == n_levels(dst_vc) ||
        throw(DimensionMismatch("dst levels $Nt != target coordinate levels $(n_levels(dst_vc))"))

    Threads.@threads for j in 1:Ny
        @inbounds for i in 1:Nx
            remap_extensive_column_pressure_overlap!(
                view(dst, i, j, :), view(src, i, j, :), Float64(ps[i, j]), src_vc, dst_vc)
        end
    end
    return dst
end

# One column of `remap_extensive_field_pressure_overlap!` at surface pressure
# `psij`; shared with the write-time flux-fidelity check, which evaluates only
# the sampled columns.
function remap_extensive_column_pressure_overlap!(dst::AbstractVector{FT},
                                                   src::AbstractVector,
                                                   psij::Float64,
                                                   src_vc::HybridSigmaPressure,
                                                   dst_vc::HybridSigmaPressure) where FT
    Ns = length(src)
    Nt = length(dst)
    @inbounds begin
        for kt in 1:Nt
            dst[kt] = zero(FT)
        end

        ks0 = 1
        for kt in 1:Nt
            t0 = Float64(dst_vc.A[kt]) + Float64(dst_vc.B[kt]) * psij
            t1 = Float64(dst_vc.A[kt + 1]) + Float64(dst_vc.B[kt + 1]) * psij
            tlo = min(t0, t1)
            thi = max(t0, t1)
            acc = 0.0

            while ks0 <= Ns
                s0 = Float64(src_vc.A[ks0]) + Float64(src_vc.B[ks0]) * psij
                s1 = Float64(src_vc.A[ks0 + 1]) + Float64(src_vc.B[ks0 + 1]) * psij
                max(s0, s1) <= tlo && (ks0 += 1; continue)
                break
            end

            ks = ks0
            while ks <= Ns
                s0 = Float64(src_vc.A[ks]) + Float64(src_vc.B[ks]) * psij
                s1 = Float64(src_vc.A[ks + 1]) + Float64(src_vc.B[ks + 1]) * psij
                slo = min(s0, s1)
                shi = max(s0, s1)
                slo >= thi && break
                overlap = min(shi, thi) - max(slo, tlo)
                if overlap > 0
                    thick = shi - slo
                    thick > 0 && (acc += Float64(src[ks]) * overlap / thick)
                end
                shi >= thi && break
                ks += 1
            end
            dst[kt] = FT(acc)
        end
    end
    return dst
end

@inline function _geometric_mean_ps(a::Real, b::Real)
    aa = max(Float64(a), eps(Float64))
    bb = max(Float64(b), eps(Float64))
    return exp((log(aa) + log(bb)) / 2)
end

"""
    fill_xface_surface_pressure!(ps_xface, ps_cell)

Fill west/east face surface pressures with the same Jensen/geometric mean used
by `compute_mass_fluxes!` for the native zonal mass flux thickness.
"""
function fill_xface_surface_pressure!(ps_xface::AbstractMatrix{<:Real},
                                      ps_cell::AbstractMatrix{<:Real})
    Nx, Ny = size(ps_cell)
    size(ps_xface) == (Nx + 1, Ny) ||
        throw(DimensionMismatch("x-face ps shape $(size(ps_xface)) != ($Nx+1, $Ny)"))
    @inbounds for j in 1:Ny, i in 1:(Nx + 1)
        ps_xface[i, j] = xface_surface_pressure(ps_cell, i, j)
    end
    return ps_xface
end

# Surface pressure of x-face `(i, j)` (periodic in longitude) and of y-face
# `(i, j)` (pole rows copy the adjacent cell), as `fill_*face_surface_pressure!`.
@inline function xface_surface_pressure(ps_cell::AbstractMatrix{<:Real}, i::Int, j::Int)
    Nx = size(ps_cell, 1)
    il = i == 1 ? Nx : i - 1
    ir = i <= Nx ? i : 1
    return _geometric_mean_ps(ps_cell[il, j], ps_cell[ir, j])
end
@inline function yface_surface_pressure(ps_cell::AbstractMatrix{<:Real}, i::Int, j::Int)
    Ny = size(ps_cell, 2)
    j == 1 && return Float64(ps_cell[i, 1])
    j == Ny + 1 && return Float64(ps_cell[i, Ny])
    return _geometric_mean_ps(ps_cell[i, j - 1], ps_cell[i, j])
end

"""
    fill_yface_surface_pressure!(ps_yface, ps_cell)

Fill south/north face surface pressures with the same Jensen/geometric mean
used by `compute_mass_fluxes!` for native meridional mass flux thickness.
Polar boundary values are copied from the adjacent pole row; the corresponding
mass fluxes are zeroed separately by `close_pole_faces!`.
"""
function fill_yface_surface_pressure!(ps_yface::AbstractMatrix{<:Real},
                                      ps_cell::AbstractMatrix{<:Real})
    Nx, Ny = size(ps_cell)
    size(ps_yface) == (Nx, Ny + 1) ||
        throw(DimensionMismatch("y-face ps shape $(size(ps_yface)) != ($Nx, $Ny+1)"))
    @inbounds for j in 1:(Ny + 1), i in 1:Nx
        ps_yface[i, j] = yface_surface_pressure(ps_cell, i, j)
    end
    return ps_yface
end

"""
    remap_qv_pressure_overlap!(qv_dst, qv_src, m_src, ps, src_vc, dst_vc)

Mass-weighted vertical remap of specific humidity onto an independent target
hybrid coordinate. The same pressure-overlap fractions used for extensive mass
fields are applied to the source dry-air mass, then `qv` is averaged over each
target layer:

```math
q^t_k =
\\frac{\\sum_s q^s_s m^s_s f_{s,k}}
     {\\sum_s m^s_s f_{s,k}}.
```

This is only for optional diagnostic humidity output. Transport masses and
fluxes are remapped by `remap_extensive_field_pressure_overlap!`.
"""
function remap_qv_pressure_overlap!(qv_dst::Array{FT, 3},
                                    qv_src::Array{FT, 3},
                                    m_src::Array{FT, 3},
                                    ps::AbstractMatrix{<:Real},
                                    src_vc::HybridSigmaPressure,
                                    dst_vc::HybridSigmaPressure) where FT
    Nx, Ny, Ns = size(qv_src)
    size(m_src) == (Nx, Ny, Ns) ||
        throw(DimensionMismatch("m_src shape $(size(m_src)) != qv_src shape ($Nx, $Ny, $Ns)"))
    size(ps) == (Nx, Ny) ||
        throw(DimensionMismatch("ps shape $(size(ps)) != qv horizontal shape ($Nx, $Ny)"))
    Nt = size(qv_dst, 3)
    size(qv_dst, 1) == Nx && size(qv_dst, 2) == Ny ||
        throw(DimensionMismatch("qv_dst horizontal shape $(size(qv_dst)[1:2]) != ($Nx, $Ny)"))
    Ns == n_levels(src_vc) && Nt == n_levels(dst_vc) ||
        throw(DimensionMismatch("vertical coordinate levels do not match qv arrays"))

    Threads.@threads for j in 1:Ny
        mass_acc = zeros(Float64, Nt)
        q_acc = zeros(Float64, Nt)
        @inbounds for i in 1:Nx
            fill!(mass_acc, 0.0)
            fill!(q_acc, 0.0)
            psij = Float64(ps[i, j])
            ks0 = 1
            for kt in 1:Nt
                t0 = Float64(dst_vc.A[kt]) + Float64(dst_vc.B[kt]) * psij
                t1 = Float64(dst_vc.A[kt + 1]) + Float64(dst_vc.B[kt + 1]) * psij
                tlo = min(t0, t1)
                thi = max(t0, t1)

                while ks0 <= Ns
                    s0 = Float64(src_vc.A[ks0]) + Float64(src_vc.B[ks0]) * psij
                    s1 = Float64(src_vc.A[ks0 + 1]) + Float64(src_vc.B[ks0 + 1]) * psij
                    max(s0, s1) <= tlo && (ks0 += 1; continue)
                    break
                end

                ks = ks0
                while ks <= Ns
                    s0 = Float64(src_vc.A[ks]) + Float64(src_vc.B[ks]) * psij
                    s1 = Float64(src_vc.A[ks + 1]) + Float64(src_vc.B[ks + 1]) * psij
                    slo = min(s0, s1)
                    shi = max(s0, s1)
                    slo >= thi && break
                    overlap = min(shi, thi) - max(slo, tlo)
                    if overlap > 0
                        thick = shi - slo
                        if thick > 0
                            frac = overlap / thick
                            mass = Float64(m_src[i, j, ks]) * frac
                            mass_acc[kt] += mass
                            q_acc[kt] += Float64(qv_src[i, j, ks]) * mass
                        end
                    end
                    shi >= thi && break
                    ks += 1
                end
            end
            for kt in 1:Nt
                qv_dst[i, j, kt] = mass_acc[kt] > 0 ? FT(q_acc[kt] / mass_acc[kt]) : zero(FT)
            end
        end
    end
    return qv_dst
end

"""
    LLPoissonWorkspace

Pre-allocated scratch and cached FFT plans for the LL Poisson mass-flux balance.
Construct once per grid size and reuse across all windows and levels.
"""
struct LLPoissonWorkspace{P, Q}
    fac        :: Matrix{Float64}       # Laplacian eigenvalues (Nx × Ny)
    psi        :: Matrix{Float64}       # streamfunction scratch
    residual   :: Matrix{Float64}       # residual scratch
    cmplx_buf  :: Matrix{ComplexF64}    # in-place FFT I/O buffer
    fft_plan   :: P                     # plan_fft!(cmplx_buf) — concrete FFTW plan
    ifft_plan  :: Q                     # plan_ifft!(cmplx_buf) — ScaledPlan wrapper
end

function LLPoissonWorkspace(Nx::Int, Ny::Int)
    fac = Array{Float64}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        fac[i, j] = 2.0 * (cos(2π * (i - 1) / Nx) + cos(2π * (j - 1) / Ny) - 2.0)
    end
    fac[1, 1] = 1.0
    cmplx_buf = zeros(ComplexF64, Nx, Ny)
    return LLPoissonWorkspace(fac,
                              zeros(Float64, Nx, Ny),
                              zeros(Float64, Nx, Ny),
                              cmplx_buf,
                              plan_fft!(cmplx_buf),
                              plan_ifft!(cmplx_buf))
end

"""
    balance_mass_fluxes!(am, bm, dm_dt, ws::LLPoissonWorkspace)

FFT Poisson solve of the lat-lon balance (the solver of TM5 r1112
`BalanceMassFluxes_m`, grid_type_ll.F90:2536-2653), applied to every level
`k` of the arrays it is given.

Adjusts `am` and `bm` in-place so that the discrete horizontal convergence
at each cell `(i, j)` matches the prescribed mass tendency `dm_dt`:

    (am[i] − am[i+1]) + (bm[j] − bm[j+1])  =  dm_dt[i, j]     ∀ (i, j)

The correction is a streamfunction `ψ` found by solving `∇²ψ = residual` on
the periodic lat-lon grid via FFT division by the discrete Laplacian eigenvalues.
The `ws::LLPoissonWorkspace` provides pre-computed eigenvalues, scratch arrays,
and cached in-place FFT plans for zero-allocation operation.

Preprocessing calls it on the column-integrated fluxes only (one level, from
`balance_column_mass_fluxes!`). Balancing every model level to its
own mass change would leave nothing for the vertical mass flux `cm`: TM5's
per-level balance keeps vertical motion only because its target includes the
IFS vertical flux, which our preprocessing does not have.
"""
function balance_mass_fluxes!(am::Array{FT, 3}, bm::Array{FT, 3},
                              dm_dt::Array{FT, 3},
                              ws::LLPoissonWorkspace;
                              log_summary::Bool=true) where FT
    Nx = size(am, 1) - 1
    Ny = size(bm, 2) - 1
    Nz = size(am, 3)

    n_balanced = 0
    max_residual = 0.0

    for k in 1:Nz
        @inbounds for j in 1:Ny, i in 1:Nx
            conv = (Float64(am[i, j, k]) - Float64(am[i + 1, j, k])) +
                   (Float64(bm[i, j, k]) - Float64(bm[i, j + 1, k]))
            ws.residual[i, j] = conv - Float64(dm_dt[i, j, k])
        end

        max_res_k = maximum(abs, ws.residual)
        max_residual = max(max_residual, max_res_k)
        max_res_k < 1e-10 && continue

        # In-place FFT solve: ψ̂ = R̂ / eigenvalue
        @inbounds for j in 1:Ny, i in 1:Nx
            ws.cmplx_buf[i, j] = complex(ws.residual[i, j])
        end
        ws.fft_plan * ws.cmplx_buf
        @inbounds for j in 1:Ny, i in 1:Nx
            ws.cmplx_buf[i, j] /= ws.fac[i, j]
        end
        ws.cmplx_buf[1, 1] = 0.0 + 0.0im
        ws.ifft_plan * ws.cmplx_buf
        @inbounds for j in 1:Ny, i in 1:Nx
            ws.psi[i, j] = real(ws.cmplx_buf[i, j])
        end

        # Zonal flux correction
        @inbounds for j in 1:Ny
            u_wrap = ws.psi[1, j] - ws.psi[Nx, j]
            for i in 2:Nx
                du = (ws.psi[i, j] - ws.psi[i - 1, j]) - u_wrap
                am[i, j, k] += FT(du)
            end
            am[1, j, k] += FT(0)
            am[Nx + 1, j, k] += FT(0)
        end

        # Meridional flux correction
        @inbounds for i in 1:Nx
            v_wrap = ws.psi[i, 1] - ws.psi[i, Ny]
            for j in 2:Ny
                dv = (ws.psi[i, j] - ws.psi[i, j - 1]) - v_wrap
                bm[i, j, k] += FT(dv)
            end
        end

        n_balanced += 1
    end

    log_summary &&
        @info "Poisson balance: corrected $n_balanced/$Nz levels, " *
              "max pre-balance residual: $(round(max_residual, sigdigits=3)) kg"
end

"""
    balance_mass_fluxes!(am, bm, dm_dt)

Convenience overload that allocates a temporary workspace. Prefer the
`LLPoissonWorkspace` overload when calling in a loop (e.g., per window).
"""
function balance_mass_fluxes!(am::Array{FT, 3}, bm::Array{FT, 3},
                              dm_dt::Array{FT, 3}) where FT
    Nx = size(am, 1) - 1
    Ny = size(bm, 2) - 1
    ws = LLPoissonWorkspace(Nx, Ny)
    balance_mass_fluxes!(am, bm, dm_dt, ws)
end

"""
    balance_column_mass_fluxes!(am, bm, m, dm_dt, ws::LLPoissonWorkspace;
                                weights = MassWeightedColumn())

Column Poisson balance of a lat-lon window: one FFT Poisson solve
(`balance_mass_fluxes!`) makes the convergence of the
column-integrated fluxes `Σ_k am`, `Σ_k bm` equal the column mass tendency
`Σ_k dm_dt` in every cell, and each face's column correction `δ` is spread
over the layers by `spread_column_correction!` with the column `weights`
(`AbstractColumnWeights`; production passes the configured
`column_balance_weights`, `hybrid_mass` by default, built from the output
coordinate's `B`). The layer winds keep their vertical structure, and `cm`
diagnosed afterwards from the explicit tendency carries the vertical motion
and closes at the top and the surface.
"""
function balance_column_mass_fluxes!(am::Array{FT, 3}, bm::Array{FT, 3},
                                     m::Array{FT, 3}, dm_dt::Array{FT, 3},
                                     ws::LLPoissonWorkspace;
                                     weights::AbstractColumnWeights = MassWeightedColumn()) where FT
    Nx = size(am, 1) - 1
    Ny = size(bm, 2) - 1
    Nz = size(am, 3)
    size(bm, 1) == Nx && size(bm, 2) == Ny + 1 && size(bm, 3) == Nz ||
        error("balance_column_mass_fluxes!: bm shape $(size(bm)) incompatible with am $(size(am))")
    size(m) == (Nx, Ny, Nz) ||
        error("balance_column_mass_fluxes!: m shape $(size(m)) incompatible with am $(size(am))")
    size(dm_dt) == size(m) ||
        error("balance_column_mass_fluxes!: dm_dt shape $(size(dm_dt)) incompatible with m $(size(m))")
    check_column_weight_levels(weights, Nz)

    col_am = zeros(FT, Nx + 1, Ny, 1)
    col_bm = zeros(FT, Nx, Ny + 1, 1)
    col_dm = zeros(FT, Nx, Ny, 1)

    @inbounds for k in 1:Nz
        for j in 1:Ny, i in 1:Nx + 1
            col_am[i, j, 1] += am[i, j, k]
        end
        for j in 1:Ny + 1, i in 1:Nx
            col_bm[i, j, 1] += bm[i, j, k]
        end
        for j in 1:Ny, i in 1:Nx
            col_dm[i, j, 1] += dm_dt[i, j, k]
        end
    end

    pre_residual = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        conv = (Float64(col_am[i, j, 1]) - Float64(col_am[i + 1, j, 1])) +
               (Float64(col_bm[i, j, 1]) - Float64(col_bm[i, j + 1, 1]))
        pre_residual = max(pre_residual, abs(conv - Float64(col_dm[i, j, 1])))
    end

    col_am_before = copy(col_am)
    col_bm_before = copy(col_bm)
    balance_mass_fluxes!(col_am, col_bm, col_dm, ws; log_summary=false)

    max_face_delta = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx + 1
        delta = Float64(col_am[i, j, 1] - col_am_before[i, j, 1])
        max_face_delta = max(max_face_delta, abs(delta))
        delta == 0.0 && continue
        i_w = i == 1 ? Nx : i - 1          # periodic in longitude
        i_e = i == Nx + 1 ? 1 : i
        spread_column_correction!(view(am, i, j, :), delta, weights,
                                  view(m, i_w, j, :), view(m, i_e, j, :))
    end

    @inbounds for j in 1:Ny + 1, i in 1:Nx
        delta = Float64(col_bm[i, j, 1] - col_bm_before[i, j, 1])
        max_face_delta = max(max_face_delta, abs(delta))
        delta == 0.0 && continue
        spread_column_correction!(view(bm, i, j, :), delta, weights,
                                  view(m, i, max(j - 1, 1), :), view(m, i, min(j, Ny), :))
    end

    post_residual = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        conv = 0.0
        for k in 1:Nz
            conv += (Float64(am[i, j, k]) - Float64(am[i + 1, j, k])) +
                    (Float64(bm[i, j, k]) - Float64(bm[i, j + 1, k]))
        end
        post_residual = max(post_residual, abs(conv - Float64(col_dm[i, j, 1])))
    end

    return (max_pre_residual = pre_residual,
            max_post_residual = post_residual,
            max_face_delta = max_face_delta)
end

"""
    merge_qv!(qv_merged, qv_native, m_native, mm)

Merge native-level specific humidity to the output vertical grid using native
air mass as the averaging weight.
"""
function merge_qv!(qv_merged::Array{FT, 3}, qv_native::Array{FT, 3},
                   m_native::Array{FT, 3}, mm::Vector{Int}) where FT
    Nx, Ny = size(qv_merged, 1), size(qv_merged, 2)
    Nz_merged = size(qv_merged, 3)
    fill!(qv_merged, zero(FT))
    m_sum = zeros(FT, Nx, Ny, Nz_merged)
    @inbounds for k in 1:length(mm)
        km = mm[k]
        @views qv_merged[:, :, km] .+= qv_native[:, :, k] .* m_native[:, :, k]
        @views m_sum[:, :, km] .+= m_native[:, :, k]
    end
    @inbounds for km in 1:Nz_merged
        @views qv_merged[:, :, km] ./= max.(m_sum[:, :, km], FT(1))
    end
end

# Horizontal size `(Nx, Ny)` of the thermo `q` variable, stored either
# `(longitude, latitude, level, time)` or `(time, level, latitude, longitude)`.
function _thermo_horizontal_size(q_var)
    return dimnames(q_var)[1] == "longitude" ? (size(q_var, 1), size(q_var, 2)) :
                                               (size(q_var, 4), size(q_var, 3))
end

# The thermo file's longitude and latitude coordinate values in file order,
# found by name as `GriddedIO.find_coordinate` finds them and checked against
# the horizontal size of `q`.
function _thermo_file_coordinates(ds)
    Nx, Ny = _thermo_horizontal_size(ds["q"])
    lon = ds[find_coordinate(ds, LongitudeAxis(); length = Nx)][:]
    lat = ds[find_coordinate(ds, LatitudeAxis(); length = Ny)][:]
    return lon, lat
end

"""
    read_qv_from_thermo(thermo_path, hour_idx, Nx, Ny, Nz; FT=Float32)

Read hourly ERA5 model-level specific humidity from the thermo NetCDF
sidecar as `(longitude, south_to_north_latitude, level)`.

Two reorderings relative to the raw NetCDF data, both decided from the file's
coordinate variables (found as `GriddedIO.find_coordinate` finds them):

1. **Latitude orientation.** ERA5 latitudes run north to south
   (`latitude[1] = +90`); the array is then reversed along latitude so `j = 1`
   is the southernmost row.
2. **Longitude order.** A `0 … 360 − Δ` longitude axis is rolled by `Nx ÷ 2`
   columns, so the column of longitude 180° comes first.

The values stay where the file put them. ERA5 thermo fields are point values
at longitudes `0, Δ, …, 360 − Δ` and latitudes `90, 90 − Δ, …, −90`
(`scripts/downloads/download_era5_physics.py` checks this layout), so after
the reordering index `(i, j)` holds the value at
`(−180 + (i − 1)Δ, −90 + (j − 1)Δ)`. These are not the cell centres of a
lat-lon staging mesh (`−180 + (i − ½)Δ`); [`read_thermo_coordinates`](@ref)
returns the coordinates of the reordered array, and the lat-lon workspaces
interpolate with them.
"""
function read_qv_from_thermo(thermo_path::String, hour_idx::Int, Nx::Int, Ny::Int, Nz::Int;
                             FT::Type{<:AbstractFloat}=Float32)
    NCDataset(thermo_path) do ds
        q_var = ds["q"]
        dims = dimnames(q_var)
        if dims[1] == "longitude"
            q = FT.(q_var[:, :, :, hour_idx])
        else
            q_raw = FT.(q_var[hour_idx, :, :, :])
            q = permutedims(q_raw, (3, 2, 1))
        end
        lon, lat = _thermo_file_coordinates(ds)
        if size(q, 2) == Ny && lat[1] > lat[end]
            q = q[:, end:-1:1, :]
        end
        q = normalize_lon_to_centered(q, lon)
        return q
    end
end

"""
    read_thermo_coordinates(thermo_path) -> (lon, lat)

Longitudes and latitudes [degrees] of the humidity values in the arrays
[`read_qv_from_thermo`](@ref) and [`read_daily_qv_from_thermo`](@ref) return:
the file's coordinate variables (found by name through
`GriddedIO.find_coordinate`), reordered as the data are (latitude south to
north; a `0 … 360` longitude axis rolled by `Nx ÷ 2`) and written in
`[−180°, 180°)`. For the ERA5 thermo archive this is
`lon[i] = −180 + (i − 1)Δ` and `lat[j] = −90 + (j − 1)Δ`.
"""
function read_thermo_coordinates(thermo_path::String)
    NCDataset(thermo_path) do ds
        lon_raw, lat_raw = _thermo_file_coordinates(ds)
        lon_file = _coordinate_values(lon_raw, "longitude")
        lat = _coordinate_values(lat_raw, "latitude")
        lat[1] > lat[end] && reverse!(lat)
        lon = vec(normalize_lon_to_centered(reshape(lon_file, :, 1, 1), lon_file))
        return mod.(lon .+ 180, 360) .- 180, lat
    end
end

function _coordinate_values(values::AbstractVector, name::AbstractString)
    any(ismissing, values) && throw(ArgumentError(
        "thermo $(name) coordinate contains missing values"))
    return Float64.(collect(values))
end

"""
    normalize_lon_to_centered(field, lons) -> field

Roll a 3D `(Nx, Ny, Nz)` field along the longitude axis into the
`[-180, 180)` order. Detects the source longitude convention from the
NetCDF `longitude` variable: when source longitudes start at or near 0°
(i.e. `0..360-Δ` convention), rolls by `Nx ÷ 2` columns; when source
already starts near -180°, no-op. The roll reorders values; it does not
change where they sit: point values at `0, Δ, …` land at `-180, -180 + Δ, …`
(see [`read_thermo_coordinates`](@ref)).

Returns `field` unchanged when `length(lons) ≠ size(field, 1)` (mismatch
that the caller will catch downstream) or when the source is already in
the staging convention.
"""
function normalize_lon_to_centered(field::AbstractArray{<:Real, 3},
                                     lons::AbstractVector{<:Real})
    Nx = size(field, 1)
    length(lons) == Nx || return field
    # Source convention check: if first longitude ≥ 0 and last > 180, the
    # array is in 0..360 layout. If first < 0, it's already centered.
    if lons[1] >= 0 && lons[end] > 180
        return circshift(field, (Nx ÷ 2, 0, 0))
    end
    return field
end

function normalize_lon_to_centered(field::AbstractArray{<:Real, 3},
                                     lons::AbstractVector)
    any(ismissing, lons) &&
        throw(ArgumentError("longitude coordinate contains missing values; cannot normalize QV longitude order"))
    return normalize_lon_to_centered(field, Float64.(lons))
end

"""
    read_daily_qv_from_thermo(thermo_path, Nx, Ny, Nz; FT=Float64, time_block=3)

Read a complete daily ERA5 thermo NetCDF `q` field into
`(longitude, south_to_north_latitude, level, time)`.

ERA5 thermo files are usually deflated and chunked across multiple times. The
reader therefore pulls chunk-aligned time blocks instead of 24 independent hour
slices, avoiding repeated decompression of the same HDF5 chunks while keeping
the result in the exact orientation expected by the dry-basis preprocessing
paths.
"""
function read_daily_qv_from_thermo(thermo_path::String,
                                   Nx::Int,
                                   Ny::Int,
                                   Nz::Int;
                                   FT::Type{<:AbstractFloat}=Float64,
                                   time_block::Int=3)
    NCDataset(thermo_path) do ds
        q_var = ds["q"]
        dims = dimnames(q_var)
        Nt = _qv_time_count(q_var, dims)
        out = Array{FT}(undef, Nx, Ny, Nz, Nt)
        lons, lats = _thermo_file_coordinates(ds)
        lat_descending = lats[1] > lats[end]
        # Same reordering as read_qv_from_thermo: roll a 0..360 longitude axis
        # into -180..180 order (values keep their positions; see
        # `read_thermo_coordinates`). Detected from the coordinate variables
        # so files in either layout are handled.
        roll_lon = (length(lons) == Nx && lons[1] >= 0 && lons[end] > 180)
        block = max(time_block, 1)

        for t0 in 1:block:Nt
            t1 = min(t0 + block - 1, Nt)
            tr = t0:t1
            if dims[1] == "longitude"
                q_raw = FT.(q_var[:, :, :, tr])
                q = lat_descending ? @view(q_raw[:, end:-1:1, :, :]) : q_raw
            else
                q_raw = FT.(q_var[tr, :, :, :])
                q_perm = permutedims(q_raw, (4, 3, 2, 1))
                q = lat_descending ? @view(q_perm[:, end:-1:1, :, :]) : q_perm
            end
            if roll_lon
                @views out[:, :, :, tr] .= circshift(q, (Nx ÷ 2, 0, 0, 0))
            else
                @views out[:, :, :, tr] .= q
            end
        end
        return out
    end
end

function _qv_time_count(q_var, dims)
    tidx = findfirst(==("time"), dims)
    tidx === nothing && error("Thermo NetCDF q variable has no time dimension")
    return size(q_var, tidx)
end

"""
    maybe_preload_qv_day(thermo_path, Nx, Ny, Nz, settings)

Return a daily in-memory QV cache when enabled and within the configured memory
cap. The empty `0×0×0×0` array means the caller should use the historical
hourly NetCDF read path.
"""
function maybe_preload_qv_day(thermo_path::String,
                              Nx::Int,
                              Ny::Int,
                              Nz::Int,
                              settings)
    settings.qv_preload || return zeros(Float64, 0, 0, 0, 0)
    records_per_day = round(Int, 86400 / ERA5_MET_INTERVAL_SECONDS)   # hourly thermo records
    bytes = Int64(Nx) * Int64(Ny) * Int64(Nz) * Int64(records_per_day) * Int64(sizeof(Float64))
    if bytes > settings.qv_preload_max_bytes
        @info @sprintf("  QV daily preload skipped: %.1f GiB exceeds cap %.1f GiB",
                       bytes / 1024.0^3, settings.qv_preload_max_bytes / 1024.0^3)
        return zeros(Float64, 0, 0, 0, 0)
    end
    t0 = time()
    qv_daily = read_daily_qv_from_thermo(thermo_path, Nx, Ny, Nz; FT=Float64)
    @info @sprintf("  QV daily preload: %.1f GiB from %s (%.1fs)",
                   Base.summarysize(qv_daily) / 1024.0^3,
                   basename(thermo_path), time() - t0)
    return qv_daily
end

"""
    apply_dry_basis_native!(m, am, bm, qv)

Convert native-level moist mass and horizontal fluxes to dry basis before level
merging. Face humidity is reconstructed by simple two-point averaging.
"""
function apply_dry_basis_native!(m::AbstractArray{<:Real, 3},
                                 am::AbstractArray{<:Real, 3},
                                 bm::AbstractArray{<:Real, 3},
                                 qv::AbstractArray{<:Real, 3})
    Nx, Ny, Nz = size(m)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        q = clamp(qv[i, j, k], 0.0, MAX_SPECIFIC_HUMIDITY)
        m[i, j, k] *= (1.0 - q)
    end

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:(Nx + 1)
        i_l = i == 1 ? Nx : i - 1
        i_r = i <= Nx ? i : 1
        q_face = 0.5 * (qv[i_l, j, k] + qv[i_r, j, k])
        q_face = clamp(q_face, 0.0, MAX_SPECIFIC_HUMIDITY)
        am[i, j, k] *= (1.0 - q_face)
    end

    @inbounds for k in 1:Nz, j in 1:(Ny + 1), i in 1:Nx
        if j == 1 || j == Ny + 1
            bm[i, j, k] = 0.0
        else
            q_face = 0.5 * (qv[i, j - 1, k] + qv[i, j, k])
            q_face = clamp(q_face, 0.0, MAX_SPECIFIC_HUMIDITY)
            bm[i, j, k] *= (1.0 - q_face)
        end
    end
    return nothing
end

"""
    recompute_cm_from_divergence!(cm, am, bm, m; B_ifc)

Recompute merged-level vertical mass flux from merged horizontal divergence.
Uses the TM5/ERA5 hybrid-coordinate `Δb × pit` closure. `B_ifc` must
contain exactly `Nz + 1` interface coefficients.

For the post-Poisson-balance correctness-critical path, prefer
[`recompute_cm_from_dm_target!`](@ref AtmosTransport.MetDrivers.recompute_cm_from_dm_target!)
which uses an explicit dm target and is basis-agnostic. This legacy function is retained for the pre-Poisson
staging / on-the-fly synthesis code paths that only seed `merged.cm_merged`
for the unused `:window_constant` delta payload.
"""
function recompute_cm_from_divergence!(cm::Array{FT, 3}, am::Array{FT, 3},
                                       bm::Array{FT, 3}, m::Array{FT, 3};
                                       B_ifc::Vector{<:Real}) where FT
    Nx = size(m, 1)
    Ny = size(m, 2)
    Nz = size(m, 3)
    length(B_ifc) == Nz + 1 || throw(ArgumentError(
        "recompute_cm_from_divergence!: B_ifc must contain Nz + 1 = $(Nz + 1) interfaces; got $(length(B_ifc))"))
    fill!(cm, zero(FT))

    @inbounds for j in 1:Ny, i in 1:Nx
        pit = 0.0
        for k in 1:Nz
            pit += (Float64(am[i + 1, j, k]) - Float64(am[i, j, k])) +
                   (Float64(bm[i, j + 1, k]) - Float64(bm[i, j, k]))
        end
        acc = 0.0
        for k in 1:Nz
            div_h = (Float64(am[i + 1, j, k]) - Float64(am[i, j, k])) +
                    (Float64(bm[i, j + 1, k]) - Float64(bm[i, j, k]))
            acc = acc - div_h + (Float64(B_ifc[k + 1]) - Float64(B_ifc[k])) * pit
            cm[i, j, k + 1] = FT(acc)
        end
    end
end

"""
    mass_fix_target_ps_dry_record(target_ps_dry_pa, target_kg, total_area_m2, g) -> Float64

Header value `mass_fix_target_ps_dry_pa` of a native-source dry-mass pin. A
configured target (`[mass_fix] mode = "target_ps_dry"`, `target_ps_dry_pa`
finite) is recorded exactly: recomputing it from the pinned air mass rounds
(98325.00000000001 Pa) or, with a Float32 `g`, is off by 1.6e-3 Pa. Without
one (GEOS `mode = "initial_endpoint"`, or a caller that passes only a kg
target, `target_ps_dry_pa = NaN`) the record is the mean dry surface pressure
of the pinned mass, `target_kg g / A`.
"""
mass_fix_target_ps_dry_record(target_ps_dry_pa::Real, target_kg::Real,
                              total_area_m2::Real, g::Real) =
    isfinite(target_ps_dry_pa) ? Float64(target_ps_dry_pa) :
        Float64(target_kg) * Float64(g) / Float64(total_area_m2)

"""
    pin_global_mean_ps!(sp, area; target_ps_dry_pa, qv_global) -> Float64

Apply a uniform additive offset to `sp` so that ⟨sp⟩_area corresponds to a
prescribed dry-air mass target inferred from a global humidity climatology.
`sp` and `area` hold one value per column: lat-lon `(Nx, Ny)` or
reduced-Gaussian `(ncell,)`. The defaults are `GLOBAL_MEAN_DRY_SURFACE_PRESSURE`
(98325 Pa) and `GLOBAL_MEAN_SPECIFIC_HUMIDITY` (0.00247 kg kg⁻¹).
"""
function pin_global_mean_ps!(sp::AbstractArray{<:Real},
                             area::AbstractArray{<:Real};
                             target_ps_dry_pa::Real = GLOBAL_MEAN_DRY_SURFACE_PRESSURE,
                             qv_global::Real = GLOBAL_MEAN_SPECIFIC_HUMIDITY)
    size(area) == size(sp) || error("area shape $(size(area)) does not match sp shape $(size(sp))")
    target_ps_total = target_ps_dry_pa / (1.0 - qv_global)
    sum_ps_area = 0.0
    sum_area = 0.0
    @inbounds for c in eachindex(IndexLinear(), sp, area)
        a = area[c]
        sum_ps_area += sp[c] * a
        sum_area += a
    end
    ps_mean_current = sum_ps_area / sum_area
    offset = target_ps_total - ps_mean_current
    @. sp += offset
    return offset
end

"""
    pin_global_mean_ps_using_qv!(sp, area, dA, dB, qv; target_ps_dry_pa) -> Float64

Apply a uniform additive offset to `sp` so that the area-weighted global mean
dry surface pressure implied by the native-layer hourly humidity field matches
the prescribed target.

The correction is still spatially uniform in `ps`; only the global dry-mass
targeting uses the instantaneous humidity pattern. `sp` and `area` hold one
value per column and `qv` adds the level dimension last: lat-lon `(Nx, Ny)`
and `(Nx, Ny, Nz)`, or reduced-Gaussian `(ncell,)` and `(ncell, Nz)`. The
default target is `GLOBAL_MEAN_DRY_SURFACE_PRESSURE` (98325 Pa).
"""
function pin_global_mean_ps_using_qv!(sp::AbstractArray{<:Real},
                                      area::AbstractArray{<:Real},
                                      dA::AbstractVector,
                                      dB::AbstractVector,
                                      qv::AbstractArray{<:Real};
                                      target_ps_dry_pa::Real = GLOBAL_MEAN_DRY_SURFACE_PRESSURE)
    Nz = size(qv, ndims(qv))
    length(dA) == Nz || error("length(dA)=$(length(dA)) does not match qv levels $Nz")
    length(dB) == Nz || error("length(dB)=$(length(dB)) does not match qv levels $Nz")
    size(qv) == (size(sp)..., Nz) || error("sp shape $(size(sp)) does not match qv horizontal shape $(size(qv)[1:end-1])")
    size(area) == size(sp) || error("area shape $(size(area)) does not match sp shape $(size(sp))")
    qv_columns = reshape(qv, length(sp), Nz)

    sum_dry_ps_area = 0.0
    sum_alpha_area = 0.0
    sum_area = 0.0

    @inbounds for c in eachindex(IndexLinear(), sp, area)
        ps_c = sp[c]
        dry_ps_col = 0.0
        alpha_col = 0.0
        for k in 1:Nz
            q = clamp(qv_columns[c, k], 0.0, MAX_SPECIFIC_HUMIDITY)
            dry_ps_col += (Float64(dA[k]) + Float64(dB[k]) * ps_c) * (1.0 - q)
            alpha_col += Float64(dB[k]) * (1.0 - q)
        end
        a = area[c]
        sum_dry_ps_area += dry_ps_col * a
        sum_alpha_area += alpha_col * a
        sum_area += a
    end

    mean_dry_ps_current = sum_dry_ps_area / sum_area
    mean_alpha = sum_alpha_area / sum_area
    offset = mass_target_offset(target_ps_dry_pa, mean_dry_ps_current, mean_alpha)
    @. sp += offset
    return offset
end

"""
    mass_target_offset(target, current, sensitivity) -> Float64

Solve the linear dry-mass pin shared by the spectral and native paths,

```math
\\Delta p_s = \\frac{M_{\\mathrm{target}}-M_{\\mathrm{current}}}
                   {\\partial M_{\\mathrm{dry}}/\\partial p_s}.
```

The inputs may be pressure or mass so long as `target` and `current` share
units and `sensitivity` is their change per pascal. For a hybrid layer,
`\\partial \\Delta p_{dry,k}/\\partial p_s = \\Delta B_k(1-q_k)`. Keeping this
algebra in one function ensures the spectral surface-pressure pin and every
native mass pin solve the same physical correction.
"""
function mass_target_offset(target::Real, current::Real, sensitivity::Real)
    target_f = Float64(target)
    current_f = Float64(current)
    sensitivity_f = Float64(sensitivity)
    isfinite(target_f) || error("Dry-mass pin target must be finite; got $target")
    isfinite(current_f) || error("Dry-mass pin current value must be finite; got $current")
    isfinite(sensitivity_f) && sensitivity_f > 0.0 ||
        error("Computed non-positive dry-pressure sensitivity: $sensitivity")
    return (target_f - current_f) / sensitivity_f
end
