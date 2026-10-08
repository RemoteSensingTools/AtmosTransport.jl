# ===========================================================================
# MERRA-2 wind-derived → C180 cubed-sphere transport-binary writer.
#
# Reproduces the validated GEOS-Chem CO₂ transport input path: derive the
# horizontal mass fluxes from MERRA-2 WINDS (U/V) + a Cameron-Smith column
# pressure-fix (the Poisson balance), instead of GEOS native cubed-sphere
# MFXC. Purely additive — the GEOS-native and ERA5 paths are untouched.
#
# This is a near-clone of `process_era5_n320_to_cs_day`
# (transport_binary/era5_n320_regrid.jl): identical mass-derivation, global
# dry-mass pin, wind rotation, flux reconstruction, Poisson balance
# (= the pressure-fixer), cm diagnosis, adaptive substep policy, contract
# verification, and streaming writer. The ONLY substantive change is replacing
# the ERA5 spectral pipeline with a direct MERRA-2 NetCDF read + conservative
# regrid to C180, and `nwindow = 8` instead of 24.
#
# Drives one UTC day end-to-end:
#
#   per window (8 × 3-hourly):
#     1. Read native MERRA-2 LL fields (PS/QV from inst3 slice `win`, U/V from
#        tavg3 slice `win` = the 3-hr time-average advecting winds) and
#        conservatively regrid PS / U / V / QV to the C180 target.
#     2. Re-derive dry-mass on C180 from the regridded moist PS + QV so the
#        target-side column closure Σ_k DELP_dry = PS_dry holds to roundoff.
#     3. Rotate cell-centre winds geographic → panel-local using the CS
#        tangent basis.
#     4. Reconstruct Arakawa-C face mass fluxes (am, bm) from rotated U/V
#        and panel DELP via the existing CS helper.
#
#   per window transition (windows 2..8):
#     5. Read the next window's fields so we can close continuity against the
#        explicit endpoint-mass target.
#     6. Poisson-balance the current window's horizontal fluxes against the
#        next-window mass tendency (the Cameron-Smith column pressure-fix).
#     7. Diagnose cm from the balanced fluxes + endpoint mass tendency.
#     8. Verify the per-substep positivity gate and the write-time replay gate.
#     9. Convert the next-window mass target into the forward `dm` payload and
#        stream-write the window to the staging binary.
#
# The final window's right endpoint is the next day's inst3 slice-1 PS/QV; on
# the archive boundary a zero-tendency fallback is used with a warning,
# mirroring the ERA5 N320 writer.
#
# Output windows: `dt_met_seconds = 10800` writes one window per 3-hour
# MERRA-2 block (8 per day); `dt_met_seconds = 3600` splits every block into
# three hourly windows (24 per day) so the hourly A1 PBL fields reach the
# runtime unaveraged. Within a block the endpoint dry mass (and the moist PS,
# QV and T) is linear in time between the 3-hourly I3 states, the 3-hour mean
# winds are reused, and each hourly window is Poisson-balanced against its own
# third of the block's mass change.
#
# Optional physics sections (GEOS-Chem archive only, see `MERRA2Settings`),
# all conservatively regridded as intensive fields:
#   * `cmfmc` (Nz+1 edges) and `dtrain`: A3 3-hour averages, converted to the
#     dry basis with the output window's endpoint QV exactly as the GEOS
#     cubed-sphere path does (`_moist_to_dry_cmfmc!`, `_moist_to_dry_dtrain!`).
#   * `cmfmc_cloud_base`: GCHP's convective cloud base, the lowest layer whose
#     regridded A3mstC DQRCU is positive (surface layer if none), stored as a
#     top-down layer index.
#   * `pblh, ustar, pbl_hflux, t2m` (and `pbl_eflux` with VDIFF): mean of the
#     hourly A1 fields inside the output window (the single hour for hourly
#     windows).
#   * `vdiff_{u,v,t,qv}`: geographic U/V (the advecting winds, unrotated),
#     window-start T and QV — the GEOS VDIFF payload convention.
# TM5 convection / `:dkg` sections are not written.
# ===========================================================================

# Cap on adaptive substep refinements per window (mirror of the N320 path).
const _MERRA2_ADAPTIVE_SUBSTEP_MAX_REFINEMENTS = 8

"""
    MERRA2ToC180Pipeline{FT, R, P, E}

Per-day MERRA-2 → C180 preprocessing workspace. Owns the conservative LL→CS
regridder, the CS preprocess scratch, and the per-window regridded C180
scalar fields (`c180_fields.{ps, qv, u, v}`), laid out as `NTuple{6, …}`
panels so the shared CS helpers (`derive_c180_dry_mass!`,
`rotate_winds_to_panel_local!`, `reconstruct_cs_fluxes!`) work unchanged.

`phys` holds the regridded optional physics panels (`cmfmc`, `dtrain`, `t`,
`dqrcu` and its `cloud_base`, and `surface_hours`, the three hourly A1 records
of the block as `(pblh, ustar, hflux, t2m[, eflux])` panels — only those
requested; empty without physics);
`edge_src`/`edge_dst` are the regridder buffers for the Nz+1-edge CMFMC.

Two pipelines per day (block start and end), reused across the 8 blocks.
"""
struct MERRA2ToC180Pipeline{FT <: AbstractFloat, R, P, E}
    regridder   :: R
    ws          :: CubedSpherePreprocessWorkspace{FT}
    Nz          :: Int
    c180_fields :: NamedTuple{(:ps, :qv, :u, :v),
                              Tuple{NTuple{6, Matrix{FT}},
                                    NTuple{6, Array{FT, 3}},
                                    NTuple{6, Array{FT, 3}},
                                    NTuple{6, Array{FT, 3}}}}
    phys        :: P
    edge_src    :: E
    edge_dst    :: E
end

"""
    _merra2_source_latlon_mesh(FT, radius) -> LatLonMesh

Build the MERRA-2 native source mesh with cell CENTERS coincident with the
archive coordinates — lon centers at `-180:0.625:179.375` (periodic; faces at
±0.3125° so each data point IS its cell center, not the west edge) and lat
points at `-90:0.5:90` with the two POLAR cells as half-width caps clamped at
±90 (the GEOS-5 point-registered grid includes the poles as points). This makes
the conservative regridder use the correct source-cell geometry. A plain
`LatLonMesh(longitude=(-180,180), latitude=(-90,90))` offsets every center by a
half-cell (centers at -179.6875°, lat spacing 180/361) → a ~0.3° spatial shift
and non-archive areas (Codex P1, 2026-06-04). Faces are passed explicitly to the
inner constructor; `ConservativeRegridding` builds cell polygons from `λᶠ`/`φᶠ`.
"""
function _merra2_source_latlon_mesh(::Type{FT}, radius) where FT
    Δλ = FT(360) / MERRA2_NX                        # 0.625°
    Δφ = FT(180) / (MERRA2_NY - 1)                  # 0.5°
    λᶜ = FT[FT(-180) + (i - 1) * Δλ for i in 1:MERRA2_NX]           # archive lon centers
    λᶠ = FT[FT(-180) - Δλ / 2 + (i - 1) * Δλ for i in 1:MERRA2_NX + 1]
    φᶜ = FT[FT(-90) + (j - 1) * Δφ for j in 1:MERRA2_NY]            # -90:0.5:90 (poles incl.)
    φᶠ = Vector{FT}(undef, MERRA2_NY + 1)
    φᶠ[1] = FT(-90)
    @inbounds for j in 2:MERRA2_NY
        φᶠ[j] = FT(-90) + (FT(j) - FT(1.5)) * Δφ    # midpoint(φᶜ[j-1], φᶜ[j])
    end
    φᶠ[MERRA2_NY + 1] = FT(90)
    return LatLonMesh{FT}(MERRA2_NX, MERRA2_NY, Δλ, Δφ, λᶜ, λᶠ, φᶜ, φᶠ, FT(radius))
end

"""
    allocate_merra2_to_c180_pipeline(target_grid; Nz, cache_dir, settings) -> MERRA2ToC180Pipeline

Build (or JLD2-load from `cache_dir`) the MERRA-2 LL → C180 conservative
regridder and allocate every per-window buffer, including the physics panels
`settings` requests. The source LL mesh is built with the TARGET mesh radius
so the two manifolds match (`build_regridder` rejects a radius mismatch).
"""
function allocate_merra2_to_c180_pipeline(target_grid::CubedSphereTargetGeometry{FT};
                                          Nz::Integer,
                                          cache_dir::Union{Nothing, AbstractString} = nothing,
                                          settings::MERRA2Settings) where FT
    Nz_int = Int(Nz)
    Nz_int >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    Nc = target_grid.mesh.Nc

    source_ll_mesh = _merra2_source_latlon_mesh(FT, target_grid.mesh.radius)
    regridder = build_regridder(source_ll_mesh, target_grid.mesh;
                                normalize = false, cache_dir = cache_dir)
    n_src = length(regridder.src_areas)
    n_dst = length(regridder.dst_areas)
    n_src == MERRA2_NX * MERRA2_NY ||
        throw(DimensionMismatch("regridder src_areas length $n_src ≠ MERRA-2 cells $(MERRA2_NX * MERRA2_NY)"))
    n_dst == ncells(target_grid.mesh) ||
        throw(DimensionMismatch("regridder dst_areas length $n_dst ≠ C180 cells $(ncells(target_grid.mesh))"))

    ws = allocate_cs_preprocess_workspace(Nc, MERRA2_NX, MERRA2_NY, Nz_int,
                                          n_src, n_dst, FT)
    c180_fields = (
        ps = ntuple(_ -> zeros(FT, Nc, Nc), 6),
        qv = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        u  = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        v  = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
    )
    panels2() = ntuple(_ -> zeros(FT, Nc, Nc), 6)
    panels3(nz) = ntuple(_ -> zeros(FT, Nc, Nc, nz), 6)
    conv = has_convection(settings)
    phys = (;)
    conv && (phys = merge(phys, (; cmfmc = panels3(Nz_int + 1), dtrain = panels3(Nz_int))))
    has_cmfmc_cloud_base(settings) &&
        (phys = merge(phys, (; dqrcu = panels3(Nz_int), cloud_base = panels2())))
    surface_names = has_pbl_eflux(settings) ? (:pblh, :ustar, :hflux, :t2m, :eflux) :
                                              (:pblh, :ustar, :hflux, :t2m)
    has_surface(settings) && (phys = merge(phys, (; surface_hours = ntuple(
        _ -> NamedTuple{surface_names}(ntuple(_ -> panels2(), length(surface_names))), 3))))
    has_vdiff_fields(settings) && (phys = merge(phys, (; t = panels3(Nz_int))))
    edge_src = conv ? zeros(FT, n_src, Nz_int + 1) : nothing
    edge_dst = conv ? zeros(FT, n_dst, Nz_int + 1) : nothing
    return MERRA2ToC180Pipeline{FT, typeof(regridder), typeof(phys), typeof(edge_src)}(
        regridder, ws, Nz_int, c180_fields, phys, edge_src, edge_dst)
end

"""
    process_merra2_window!(pipe, handles, win; FT) -> pipe

Read native MERRA-2 LL fields for window `win` (PS/QV from inst3 slice `win`,
U/V from tavg3 slice `win`) and conservatively regrid PS (2D intensive) and
QV/U/V (3D intensive) onto the C180 panels. U/V/QV are intensive → default
field type, as in the ERA5 path. The readers return top-down levels whatever
the file order; requested physics fields are regridded into `pipe.phys`.
"""
function process_merra2_window!(pipe::MERRA2ToC180Pipeline{FT},
                                handles::MERRA2DayHandles, win::Integer) where FT
    Nc = size(pipe.c180_fields.ps[1], 1)
    fields = read_merra2_window_fields(handles, win, pipe.Nz; FT = FT)
    regrid_2d_to_cs_panels!(pipe.c180_fields.ps, pipe.regridder, fields.ps,
                            pipe.ws, Nc, IntensiveCellField())
    regrid_3d_to_cs_panels!(pipe.c180_fields.qv, pipe.regridder, fields.qv, pipe.ws, Nc)
    regrid_3d_to_cs_panels!(pipe.c180_fields.u,  pipe.regridder, fields.u,  pipe.ws, Nc)
    regrid_3d_to_cs_panels!(pipe.c180_fields.v,  pipe.regridder, fields.v,  pipe.ws, Nc)
    isempty(pipe.phys) || _regrid_merra2_physics!(pipe, handles, win)
    return pipe
end

# Regrid the native physics fields of window `win` onto `pipe.phys` (still
# moist; the dry conversion needs both endpoints and happens at write time).
function _regrid_merra2_physics!(pipe::MERRA2ToC180Pipeline{FT},
                                 handles::MERRA2DayHandles, win::Integer) where FT
    Nc = size(pipe.c180_fields.ps[1], 1)
    raw = read_merra2_physics_window(handles, win, pipe.Nz; FT = FT)
    phys = pipe.phys
    if haskey(raw, :cmfmc)
        # CMFMC lives on Nz+1 edges; the shared workspace is sized to Nz layers.
        copyto!(pipe.edge_src, reshape(raw.cmfmc, size(pipe.edge_src)...))
        apply_regridder!(pipe.edge_dst, pipe.regridder, pipe.edge_src)
        unpack_flat_to_panels_3d!(phys.cmfmc, pipe.edge_dst, Nc, pipe.Nz + 1)
        regrid_3d_to_cs_panels!(phys.dtrain, pipe.regridder, raw.dtrain, pipe.ws, Nc)
    end
    if haskey(raw, :dqrcu)
        regrid_3d_to_cs_panels!(phys.dqrcu, pipe.regridder, raw.dqrcu, pipe.ws, Nc)
        convective_cloud_base!(phys.cloud_base, phys.dqrcu)
    end
    if haskey(raw, :pblh)
        for h in 1:3, name in keys(phys.surface_hours[h])
            regrid_2d_to_cs_panels!(getfield(phys.surface_hours[h], name), pipe.regridder,
                                    view(getfield(raw, name), :, :, h),
                                    pipe.ws, Nc, IntensiveCellField())
        end
    end
    haskey(raw, :t) && regrid_3d_to_cs_panels!(phys.t, pipe.regridder, raw.t, pipe.ws, Nc)
    return pipe
end

"""
    convective_cloud_base!(cloud_base, dqrcu) -> cloud_base

GCHP's convective cloud base (`convection_mod.F90`, `DO_RAS_CLOUD_CONVECTION`):
the lowest layer with convective rain production DQRCU > 0, or the surface
layer when there is none. Levels are top-down, so this is the largest such
`k`, stored as a float layer index.
"""
function convective_cloud_base!(cloud_base, dqrcu)
    for p in eachindex(cloud_base)
        cb, rain = cloud_base[p], dqrcu[p]
        fill!(cb, size(rain, 3))
        for k in axes(rain, 3), j in axes(cb, 2), i in axes(cb, 1)   # last positive k wins
            rain[i, j, k] > 0 && (cb[i, j] = k)
        end
    end
    return cloud_base
end

"""
    allocate_merra2_window_physics(settings, Nc, Nz, FT) -> NamedTuple

Output buffers for one window's physics sections (see
[`merra2_window_physics!`](@ref)); only the requested ones are allocated.
"""
function allocate_merra2_window_physics(settings::MERRA2Settings, Nc::Integer, Nz::Integer,
                                        ::Type{FT}) where FT
    panels2() = ntuple(_ -> zeros(FT, Nc, Nc), 6)
    panels3(nz) = ntuple(_ -> zeros(FT, Nc, Nc, nz), 6)
    out = (; qv_a = panels3(Nz))
    has_convection(settings) &&
        (out = merge(out, (; qv_b = panels3(Nz), cmfmc = panels3(Nz + 1), dtrain = panels3(Nz))))
    has_surface(settings) &&
        (out = merge(out, (; pblh = panels2(), ustar = panels2(), hflux = panels2(), t2m = panels2())))
    has_pbl_eflux(settings) && (out = merge(out, (; eflux = panels2())))
    has_vdiff_fields(settings) && (out = merge(out, (; t = panels3(Nz))))
    return out
end

# dst = (1 - f) a + f b per panel; exact at f = 0 and f = 1.
function _lerp_panels!(dst, a, b, f::Real)
    for p in eachindex(dst)
        T = eltype(dst[p])
        w0, w1 = T(1 - f), T(f)
        @. dst[p] = w0 * a[p] + w1 * b[p]
    end
    return dst
end

"""
    merra2_window_physics!(out, settings, cur, nxt, h, nsub)
        -> (; surface, cmfmc, dtrain, vdiff, cmfmc_cloud_base)

Physics sections of output window `h` of the `nsub` equal windows that split
the 3-hour block held in `cur` (`nxt` holds the block's right-endpoint QV and
T). Sections not requested are `nothing`.

  - The window's start and end QV, and the VDIFF temperature, are linear in
    time between the block endpoints (`cur` exactly when `nsub == 1`).
  - CMFMC and DTRAIN (3-hour averages) are copied into `out` and converted to
    dry air there with the window's start and end QV, so `cur.phys` stays
    moist and can serve every window of the block.
  - The PBL surface fields (and EFLUX) average the A1 hours inside the window.
  - The convective cloud base is the block's (from 3-hour DQRCU).
"""
function merra2_window_physics!(out, settings::MERRA2Settings, cur, nxt, h::Integer, nsub::Integer)
    f0, f1 = (h - 1) / nsub, h / nsub
    qv_a = _lerp_panels!(out.qv_a, cur.c180_fields.qv, nxt.c180_fields.qv, f0)
    cmfmc = dtrain = surface = vdiff = cmfmc_cloud_base = nothing
    if has_convection(settings)
        qv_b = _lerp_panels!(out.qv_b, cur.c180_fields.qv, nxt.c180_fields.qv, f1)
        foreach(copyto!, out.cmfmc, cur.phys.cmfmc)
        foreach(copyto!, out.dtrain, cur.phys.dtrain)
        _moist_to_dry_dtrain!(out.dtrain, qv_a, qv_b)
        _moist_to_dry_cmfmc!(out.cmfmc, qv_a, qv_b)
        cmfmc, dtrain = out.cmfmc, out.dtrain
    end
    if has_surface(settings)
        hours = (3 * (h - 1) ÷ nsub + 1):(3 * h ÷ nsub)
        for name in keys(cur.phys.surface_hours[1]), p in 1:6
            dst = getfield(out, name)[p]
            fill!(dst, zero(eltype(dst)))
            for hr in hours
                dst .+= getfield(cur.phys.surface_hours[hr], name)[p]
            end
            dst ./= length(hours)
        end
        surface = has_pbl_eflux(settings) ?
            (pblh = out.pblh, ustar = out.ustar, hflux = out.hflux, t2m = out.t2m, eflux = out.eflux) :
            (pblh = out.pblh, ustar = out.ustar, hflux = out.hflux, t2m = out.t2m)
    end
    has_cmfmc_cloud_base(settings) && (cmfmc_cloud_base = cur.phys.cloud_base)
    if has_vdiff_fields(settings)
        t = _lerp_panels!(out.t, cur.phys.t, nxt.phys.t, f0)
        vdiff = (u = cur.c180_fields.u, v = cur.c180_fields.v, t = t, qv = qv_a)
    end
    return (; surface, cmfmc, dtrain, vdiff, cmfmc_cloud_base)
end

# ---------------------------------------------------------------------------
# Day driver.
# ---------------------------------------------------------------------------

# Length of one MERRA-2 block (the I3/A3 cadence) [s].
const MERRA2_BLOCK_SECONDS = 10800.0

"""
    merra2_windows_per_block(dt_met_seconds) -> Int

Output windows per 3-hour MERRA-2 block: 1 for 3-hourly windows, 3 for hourly.
"""
function merra2_windows_per_block(dt_met_seconds::Real)
    nsub = round(Int, MERRA2_BLOCK_SECONDS / dt_met_seconds)
    (nsub in (1, 3) && nsub * dt_met_seconds == MERRA2_BLOCK_SECONDS) ||
        throw(ArgumentError("MERRA-2 dt_met_seconds must be 10800 (3-hourly) or 3600 " *
                            "(hourly windows); got $(dt_met_seconds)"))
    return nsub
end

"""
    MERRA2BlockState(pipe)

One 3-hourly MERRA-2 endpoint on the cube: the regridded fields in `pipe` and
the dry air mass, layer thickness and surface pressure derived from them.
"""
struct MERRA2BlockState{FT, P <: MERRA2ToC180Pipeline{FT}}
    pipe       :: P
    m_dry      :: NTuple{6, Array{FT, 3}}
    delp_dry   :: NTuple{6, Array{FT, 3}}
    ps_dry     :: NTuple{6, Matrix{FT}}
    ps_dry_acc :: NTuple{6, Matrix{Float64}}
end

function MERRA2BlockState(pipe::MERRA2ToC180Pipeline{FT}) where FT
    Nc, Nz = size(pipe.c180_fields.qv[1], 1), pipe.Nz
    panels(T, dims...) = ntuple(_ -> zeros(T, dims...), 6)
    return MERRA2BlockState(pipe, panels(FT, Nc, Nc, Nz), panels(FT, Nc, Nc, Nz),
                            panels(FT, Nc, Nc), panels(Float64, Nc, Nc))
end

"""
    MERRA2DayDriver

Everything one day's windows share: settings and open files, the target grid
and vertical coordinate, the writer, the substep and balance policy, the
global dry-mass target (`nothing` when unpinned) and the per-window scratch.
"""
struct MERRA2DayDriver{S, H, G, V, W, B, F, X, Q}
    settings     :: S
    handles      :: H
    grid         :: G
    vc           :: V
    writer       :: W
    nsub         :: Int
    dt_window    :: Float64
    policy       :: SubstepSchedulePolicy
    steps_min    :: Int
    cfl_target   :: Float64
    balance      :: B          # (; tol, project_every, global_solve, weights)
    flux         :: F          # (; method, thickness) of the flux reconstruction
    positivity_limit :: Float64
    replay_tol   :: Float64
    replay_check :: Bool
    mass_target  :: Union{Nothing, Float64}
    scratch      :: X
    win_phys     :: Q
end

"""Running gate statistics of one day."""
mutable struct MERRA2DayDiagnostics
    steps      :: Vector{Int}
    pre        :: Float64
    post       :: Float64
    cg_iter    :: Int
    replay_rel :: Float64
    replay_abs :: Float64
    replay_win :: Int
    positivity :: CSWorst
end

MERRA2DayDiagnostics(nwindow, steps) =
    MERRA2DayDiagnostics(fill(steps, nwindow), 0.0, 0.0, 0, 0.0, 0.0, 0,
                         init_cs_positivity_accumulator())

function _merra2_window_scratch(::Type{FT}, Nc, Nz) where FT
    panels(dims...) = ntuple(_ -> zeros(FT, dims...), 6)
    return (am = panels(Nc + 1, Nc, Nz), bm = panels(Nc, Nc + 1, Nz), cm = panels(Nc, Nc, Nz + 1),
            dm = panels(Nc, Nc, Nz), dp = panels(Nc, Nc, Nz),
            u_local = panels(Nc, Nc, Nz), v_local = panels(Nc, Nc, Nz),
            m_a = panels(Nc, Nc, Nz), m_b = panels(Nc, Nc, Nz),
            ps_dry = panels(Nc, Nc), ps_moist = panels(Nc, Nc))
end

# Global dry-air mass pin, mirroring the GEOS-CS / N320 paths: rescale the
# endpoint mass to a fixed target and recompute `ps` from it, so every day of a
# day-threaded archive shares one dry-air baseline.
_pin_endpoint_mass!(::Nothing, _grid, _m_dry, _ps_dry) = nothing
function _pin_endpoint_mass!(target::Float64, grid, m_dry, ps_dry)
    areas = grid.mesh.cell_areas
    g = eltype(areas)(GRAV)
    _pin_cs_global_air_mass!(m_dry, areas, g, target)
    for p in 1:6
        _ps_from_air_mass!(ps_dry[p], m_dry[p], areas, g, size(areas, 1), size(m_dry[p], 3))
    end
    return nothing
end

function _derive_endpoint_mass!(d::MERRA2DayDriver, s::MERRA2BlockState)
    derive_c180_dry_mass!(s.m_dry, s.delp_dry, s.ps_dry, s.ps_dry_acc,
                          s.pipe.c180_fields.ps, s.pipe.c180_fields.qv, d.vc, d.grid.mesh.cell_areas)
    _pin_endpoint_mass!(d.mass_target, d.grid, s.m_dry, s.ps_dry)
    return s
end

# Read and regrid block `blk` and derive its start dry mass.
function _read_block!(d::MERRA2DayDriver, s::MERRA2BlockState, blk::Integer)
    process_merra2_window!(s.pipe, d.handles, blk)
    return _derive_endpoint_mass!(d, s)
end

# The day's last right endpoint: next-day inst3 slice 1 (PS, QV and, for VDIFF,
# T). `nxt`'s winds and physics stay stale; only endpoint state is needed.
function _read_next_day_endpoint!(d::MERRA2DayDriver, nxt::MERRA2BlockState, cur::MERRA2BlockState)
    pipe, Nc = nxt.pipe, d.grid.Nc
    endpoint = read_merra2_next_day_endpoint(d.handles, pipe.Nz; FT = eltype(nxt.m_dry[1]))
    if endpoint === nothing
        # HACK: zero-tendency fallback for the final day of the archive (no
        # next-day inst3 file). The positivity gate will likely warn/fail, as
        # it should: the last block's continuity does not close. TODO: accept
        # only on the configured last day and fail otherwise.
        @warn "process_merra2_to_cs_day: archive-boundary fallback — no next-day inst3 file " *
              "for $(d.handles.date + Day(1)), using zero-tendency m_next. Final window's " *
              "continuity will not close."
        for p in 1:6
            copyto!(nxt.m_dry[p], cur.m_dry[p]); copyto!(nxt.ps_dry[p], cur.ps_dry[p])
            copyto!(pipe.c180_fields.ps[p], cur.pipe.c180_fields.ps[p])
            copyto!(pipe.c180_fields.qv[p], cur.pipe.c180_fields.qv[p])
            haskey(pipe.phys, :t) && copyto!(pipe.phys.t[p], cur.pipe.phys.t[p])
        end
        return nxt
    end
    regrid_2d_to_cs_panels!(pipe.c180_fields.ps, pipe.regridder, endpoint.ps, pipe.ws, Nc,
                            IntensiveCellField())
    regrid_3d_to_cs_panels!(pipe.c180_fields.qv, pipe.regridder, endpoint.qv, pipe.ws, Nc)
    haskey(endpoint, :t) && regrid_3d_to_cs_panels!(pipe.phys.t, pipe.regridder, endpoint.t, pipe.ws, Nc)
    return _derive_endpoint_mass!(d, nxt)
end

# Layer thickness that multiplies the winds in the face fluxes.
struct MoistFluxThickness end      # from the moist surface pressure (historical)
struct DryMassFluxThickness end    # g m_dry / A: the dry air the fluxes transport
_flux_thickness(kind::Symbol) = (moist = MoistFluxThickness(), dry_mass = DryMassFluxThickness())[kind]

# How face fluxes are built from the cell-centre winds.
struct PanelAverageFluxes{L <: AbstractFaceLengths}    # panel components averaged (historical)
    lengths :: L
end
struct VectorFaceFluxes                                # vectors projected on the face normal
    geom :: CSVectorFaceGeometry
    face_table :: CSGlobalFaceTable
end
function _face_flux_method(settings, grid)
    settings.face_fluxes === :vector && return VectorFaceFluxes(
        CSVectorFaceGeometry(grid.mesh, grid.face_table), grid.face_table)
    lengths = settings.face_lengths === :edge ? EdgeLengths(grid.mesh) :
              CellCenterlineLengths(grid.mesh.Δx, grid.mesh.Δy)
    return PanelAverageFluxes(lengths)
end

# Cell-centre winds the method needs: panel-local face-normal components, or
# the east/north components themselves.
_prepare_cell_winds!(::PanelAverageFluxes, x, u_east, v_north, mesh, Nz) =
    rotate_winds_to_panel_local!(x.u_local, x.v_local, u_east, v_north, mesh, Nz)
function _prepare_cell_winds!(::VectorFaceFluxes, x, u_east, v_north, mesh, Nz)
    foreach(copyto!, x.u_local, u_east)
    foreach(copyto!, x.v_local, v_north)
    return nothing
end

_face_fluxes!(m::PanelAverageFluxes, x, g, dt, Nc, Nz) =
    cs_face_fluxes!(x.am, x.bm, x.u_local, x.v_local, x.dp, m.lengths, g, dt, Nc, Nz)
_face_fluxes!(m::VectorFaceFluxes, x, g, dt, Nc, Nz) =
    cs_vector_face_fluxes!(x.am, x.bm, x.u_local, x.v_local, x.dp, m.geom, m.face_table, g, dt, Nc, Nz)

_fill_flux_thickness!(::MoistFluxThickness, x, ps_moist, m_a, vc, mesh, Nc, Nz) =
    fill_cs_layer_thickness!(x.dp, ps_moist, vc.A, vc.B, Nc, Nz)
function _fill_flux_thickness!(::DryMassFluxThickness, x, ps_moist, m_a, vc, mesh, Nc, Nz)
    g, areas = eltype(x.dp[1])(GRAV), mesh.cell_areas
    @inbounds for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
        x.dp[p][i, j, k] = g * m_a[p][i, j, k] / areas[i, j]
    end
    return nothing
end

function _reconstruct_window_fluxes!(flux, x, ps_moist, m_a, vc, mesh, g, dt, Nc, Nz)
    _fill_flux_thickness!(flux.thickness, x, ps_moist, m_a, vc, mesh, Nc, Nz)
    return _face_fluxes!(flux.method, x, g, dt, Nc, Nz)
end

# Fluxes from the panel-local winds, Poisson-balanced against the window's mass
# change (the column pressure fix), then `dm` and `cm`, at `steps` substeps.
function _balance_window!(d::MERRA2DayDriver, ps_moist, m_a, m_b, steps)
    x, mesh, vc = d.scratch, d.grid.mesh, d.vc
    FT = eltype(m_a[1])
    Nc, Nz = d.grid.Nc, size(m_a[1], 3)
    _reconstruct_window_fluxes!(d.flux, x, ps_moist, m_a, vc, mesh,
                                FT(GRAV), FT(d.dt_window / (2 * steps)), Nc, Nz)
    g, tol, project_every = d.grid, d.balance.tol, d.balance.project_every
    diag = if d.balance.global_solve     # per-level solve (diagnostic option)
        balance_cs_global_mass_fluxes!(x.am, x.bm, m_a, m_b, g.face_table, g.cell_degree,
                                       steps, g.poisson_scratch; tol, max_iter = 20000,
                                       project_every)
    else                                 # column solve, correction spread by `weights`
        balance_cs_column_mass_fluxes!(x.am, x.bm, m_a, m_b, g.face_table, g.cell_degree,
                                       steps, g.poisson_scratch; tol, max_iter = 20000,
                                       project_every, weights = d.balance.weights)
    end
    sync_all_cs_boundary_mirrors!(x.am, x.bm, mesh.connectivity, Nc, Nz)
    fill_cs_window_mass_tendency!(x.dm, m_a, m_b, steps)
    foreach(c -> fill!(c, zero(FT)), x.cm)
    diagnose_cs_cm!(x.cm, x.am, x.bm, x.dm, m_a, Nc, Nz, d.balance.weights)
    return diag
end

# Raise the substep count until the per-substep CFL is under the target.
function _adapt_window!(d::MERRA2DayDriver, ps_moist, m_a, m_b)
    steps = d.steps_min
    diag = _balance_window!(d, ps_moist, m_a, m_b, steps)
    d.policy.adaptive_substeps || return steps, diag
    for _ in 1:_MERRA2_ADAPTIVE_SUBSTEP_MAX_REFINEMENTS
        pos = verify_substep_positivity_cs!(m_a, x.am, x.bm, x.cm; cfl_limit = d.cfl_target,
                                            m_next = m_b)
        next = next_substeps(d.policy, steps, pos.ratio)
        next == steps && break
        steps = next
        diag = _balance_window!(d, ps_moist, m_a, m_b, steps)
    end
    return steps, diag
end

function _record_window!(diag::MERRA2DayDiagnostics, d::MERRA2DayDriver, m_a, m_b, win, steps, balance)
    x = d.scratch
    diag.steps[win] = steps
    diag.pre = max(diag.pre, balance.max_pre_residual)
    diag.post = max(diag.post, balance.max_post_residual)
    diag.cg_iter = max(diag.cg_iter, balance.max_cg_iter)
    positivity = if d.replay_check
        contract = verify_cs_window_contract!(m_a, x.am, x.bm, x.cm, m_b, steps, win;
                                              replay_tol = d.replay_tol,
                                              positivity_cfl_limit = d.positivity_limit)
        if diag.replay_win == 0 || contract.replay.max_rel_err > diag.replay_rel
            diag.replay_rel = contract.replay.max_rel_err
            diag.replay_abs = contract.replay.max_abs_err
            diag.replay_win = win
        end
        contract.positivity
    else
        verify_substep_positivity_cs!(m_a, x.am, x.bm, x.cm; cfl_limit = d.positivity_limit)
    end
    diag.positivity = update_cs_positivity_accumulator(diag.positivity, positivity, win)
    return diag
end

# Write the `nsub` windows of block `blk`, from `cur` (block start) to `nxt`
# (block end). Within the block the endpoint state is linear in time.
function _emit_block!(d::MERRA2DayDriver, cur::MERRA2BlockState, nxt::MERRA2BlockState,
                      blk::Integer, diag::MERRA2DayDiagnostics)
    x, FT = d.scratch, eltype(cur.m_dry[1])
    _prepare_cell_winds!(d.flux.method, x, cur.pipe.c180_fields.u, cur.pipe.c180_fields.v,
                         d.grid.mesh, cur.pipe.Nz)
    for h in 1:d.nsub
        win = (blk - 1) * d.nsub + h
        f0, f1 = (h - 1) / d.nsub, h / d.nsub
        _lerp_panels!(x.m_a, cur.m_dry, nxt.m_dry, f0)
        _lerp_panels!(x.m_b, cur.m_dry, nxt.m_dry, f1)
        _lerp_panels!(x.ps_dry, cur.ps_dry, nxt.ps_dry, f0)
        _lerp_panels!(x.ps_moist, cur.pipe.c180_fields.ps, nxt.pipe.c180_fields.ps, f0)
        t_balance = time()
        steps, balance = _adapt_window!(d, x.ps_moist, x.m_a, x.m_b)
        t_balance = time() - t_balance
        _record_window!(diag, d, x.m_a, x.m_b, win, steps, balance)
        _fill_cs_mass_delta_payload!(x.dm, x.m_a, x.m_b)
        payload = merge((m = x.m_a, am = x.am, bm = x.bm, cm = x.cm, ps = x.ps_dry, dm = x.dm),
                        merra2_window_physics!(d.win_phys, d.settings, cur.pipe, nxt.pipe, h, d.nsub))
        write_window!(d.writer, ReadyWindow{CubedSphereTargetGeometry, FT}(win, payload))
        @info @sprintf("    Window %2d/%d: wrote (steps=%d bal %.2fs pre=%.2e post=%.2e iter=%d)",
                       win, length(diag.steps), steps, t_balance, balance.max_pre_residual,
                       balance.max_post_residual, balance.max_cg_iter)
    end
    return diag
end

# Header entries describing the MERRA-2 inputs; only the requested physics.
function _merra2_provenance(settings::MERRA2Settings, handles, nsub)
    entries = Dict{String, Any}(
        "source_type" => "merra2_native_latlon",
        "source_root" => settings.root_dir,
        "merra2_archive" => merra2_archive_name(settings.archive),
        "merra2_level_order" => string(nameof(typeof(handles.level_order))),
        "winds_collection" => String(settings.winds_collection),
        "merra2_windows_per_block" => nsub,
        "column_balance_weights" => String(settings.column_balance_weights),
        "flux_face_lengths" => settings.face_fluxes === :vector ? "edge" : String(settings.face_lengths),
        "flux_thickness" => String(settings.flux_thickness),
        "face_fluxes" => String(settings.face_fluxes),
        "merra2_window_interpolation" => nsub == 1 ? "none" :
            "dry mass, PS, QV, T linear in time between 3-hourly I3; 3-hour mean winds")
    has_convection(settings) && (entries["cmfmc_dtrain_source"] =
        "A3mstE CMFMC + A3dyn DTRAIN 3-hour means, dry via the window's endpoint qv")
    has_cmfmc_cloud_base(settings) && (entries["cmfmc_cloud_base_source"] =
        "A3mstC DQRCU, conservatively regridded; lowest layer with DQRCU > 0")
    has_surface(settings) && (entries["pbl_surface_source"] =
        "A1 PBLH/USTAR/HFLUX/T2M" * (has_pbl_eflux(settings) ? "/EFLUX" : "") *
        " mean of the hourly records in the window")
    has_vdiff_fields(settings) && (entries["vdiff_source"] =
        "A3dyn geographic U/V, I3 T and QV at window start")
    return entries
end

function _open_merra2_writer(settings, handles, grid::CubedSphereTargetGeometry{FT}, vc, out_path,
                             nwindow, nsub, dt_window, steps, policy, mass_target) where FT
    mkpath(dirname(out_path))
    tmp_path = out_path * ".tmp"
    isfile(tmp_path) && rm(tmp_path)
    header = merge(_merra2_provenance(settings, handles, nsub), Dict{String, Any}(
        "preprocessor" => "process_merra2_to_cs_day",
        # Advection at the baked substep cadence; convection and chemistry once
        # per met window.
        "runtime_substep_contract" => "binary_schedule",
        "preprocessor_contract" => "plan41_variable_substeps",
        "adaptive_substeps" => policy.adaptive_substeps,
        "target_type" => "cubed_sphere",
        "regrid_method" => "conservative",
        "poisson_balanced" => true,
        # The column Poisson balance plays the role of the Cameron-Smith
        # pressure fix (`pjc_pfix_mod.F90`, GEOS-Chem Classic): it forces the
        # column flux convergence to match the analyzed dry-mass tendency.
        # GCHP applies no pressure fix; FV3 remaps to its advected surface
        # pressure instead.
        "wind_flux_pressure_fix" => "cameron_smith_column_balance",
        "global_mass_pin_enabled" => mass_target !== nothing,
        "global_mass_pin_target_kg" => mass_target))
    inner = open_streaming_cs_transport_binary(
        tmp_path, grid.Nc, 6, length(vc.A) - 1, nwindow, vc;
        FT, dt_met_seconds = dt_window, half_dt_seconds = dt_window / 2,
        steps_per_window = steps, include_flux_delta = true, mass_basis = :dry,
        include_surface = has_surface(settings), include_cmfmc = has_convection(settings),
        include_dtrain = has_convection(settings), include_gchp_vdiff = has_vdiff_fields(settings),
        include_pbl_eflux = has_pbl_eflux(settings),
        include_cmfmc_cloud_base = has_cmfmc_cloud_base(settings),
        panel_convention = _cs_panel_convention_tag(grid), cs_definition = _cs_definition_tag(grid),
        cs_coordinate_law = _cs_coordinate_law_tag(grid), cs_center_law = _cs_center_law_tag(grid),
        longitude_offset_deg = longitude_offset_deg(cs_definition(grid.mesh)),
        extra_header = header)
    return CubedSphereBinaryWriter(inner, mass_basis_from_symbol(:dry); Nc = grid.Nc, npanel = 6,
                                   final_path = String(out_path))
end

"""
    process_merra2_to_cs_day(date, settings, target_grid; out_path, dt_met_seconds, …)

Write a v4 cubed-sphere transport binary for one UTC `date` from MERRA-2 to
`out_path` (staged as `out_path.tmp`, promoted when the gates pass).

Horizontal mass fluxes come from the regridded 3-hour mean winds and are
Poisson-balanced (a column pressure fix) against the dry-mass change between
the 3-hourly inst3 endpoints. `dt_met_seconds = 10800` writes one window per
3-hour block (8 per day); `3600` splits each block into three hourly windows
(24 per day) with endpoint mass, PS, QV and T linear in time, so the hourly A1
boundary-layer fields reach the runtime unaveraged. Optional convection, PBL,
VDIFF and cloud-base sections follow `settings`. Only `mass_basis = :dry`.
"""
function process_merra2_to_cs_day(date::Date,
                                  settings::MERRA2Settings,
                                  target_grid::CubedSphereTargetGeometry{FT};
                                  out_path::AbstractString,
                                  Nz::Integer = MERRA2_NATIVE_LEVEL_COUNT,
                                  mass_basis::Symbol = :dry,
                                  dt_met_seconds::Real = MERRA2_BLOCK_SECONDS,
                                  steps_per_window::Integer = 1,
                                  adaptive_substeps::Bool = true,
                                  substep_cfl_target::Real = 0.95,
                                  max_steps_per_window::Integer = typemax(Int),
                                  cs_balance_tol::Real = 1e-14,
                                  cs_balance_project_every::Integer = 50,
                                  positivity_cfl_limit::Real = 0.95,
                                  require_substep_positivity::Bool = true,
                                  cache_dir::Union{Nothing, AbstractString} = nothing,
                                  global_mass_pin::Bool = false,
                                  global_mass_target_kg::Real = NaN) where FT
    mass_basis === :dry ||
        throw(ArgumentError("MERRA-2 → CS writer only supports mass_basis=:dry; got $(mass_basis)"))
    steps_per_window >= 1 || throw(ArgumentError("steps_per_window must be ≥ 1; got $(steps_per_window)"))
    nsub = merra2_windows_per_block(dt_met_seconds)
    vc = load_hybrid_coefficients(expand_data_path(settings.coefficients_file))
    length(vc.A) == length(vc.B) == Nz + 1 ||
        throw(DimensionMismatch("hybrid A/B length $(length(vc.A))/$(length(vc.B)) ≠ Nz+1 = $(Nz + 1); " *
                                "check `settings.coefficients_file` vs the requested Nz"))
    policy = SubstepSchedulePolicy(; adaptive_substeps, substep_cfl_target = Float64(substep_cfl_target),
                                   min_steps_per_window = Int(steps_per_window),
                                   max_steps_per_window = Int(max_steps_per_window))
    mass_target = global_mass_pin && isfinite(global_mass_target_kg) ? Float64(global_mass_target_kg) : nothing
    t_start = time()
    @info @sprintf("Process MERRA-2 → CS day: date=%s, Nc=%d, Nz=%d, FT=%s, winds=%s, %d windows/block",
                   string(date), target_grid.Nc, Nz, string(FT), String(settings.winds_collection), nsub)
    mass_target === nothing ||
        @info @sprintf("  Global dry-mass pin ON: target=%.9e kg (%.3f Pa dry ⟨ps⟩)", mass_target,
                       mass_target * GRAV / (6 * sum(Float64, target_grid.mesh.cell_areas)))

    global_solve = horizontal_poisson_balance_enabled()
    global_solve && settings.column_balance_weights !== :mass && throw(ArgumentError(
        "column_balance_weights = $(settings.column_balance_weights) applies to the column " *
        "Poisson balance; it cannot be combined with ATMOSTR_ENABLE_HORIZONTAL_POISSON_BALANCE=1"))
    handles = open_merra2_day(settings, date; next_day_handle = true)
    try
        new_block() = MERRA2BlockState(allocate_merra2_to_c180_pipeline(target_grid; Nz, cache_dir, settings))
        cur, nxt = new_block(), new_block()
        nblock = windows_per_day(settings, date)
        nwindow = nblock * nsub
        writer = _open_merra2_writer(settings, handles, target_grid, vc, out_path, nwindow, nsub,
                                     Float64(dt_met_seconds), Int(steps_per_window), policy, mass_target)
        @info @sprintf("  Output: %s (Nc=%d, Nz=%d, FT=%s)", basename(out_path), target_grid.Nc, Nz, string(FT))
        d = MERRA2DayDriver(
            settings, handles, target_grid, vc, writer, nsub, Float64(dt_met_seconds), policy,
            Int(steps_per_window), Float64(substep_cfl_target),
            (tol = Float64(cs_balance_tol), project_every = Int(cs_balance_project_every),
             global_solve, weights = column_weights(settings.column_balance_weights, vc.B)),
            (method = _face_flux_method(settings, target_grid),
             thickness = _flux_thickness(settings.flux_thickness)),
            Float64(positivity_cfl_limit), replay_tolerance(FT),
            get(ENV, "ATMOSTR_NO_WRITE_REPLAY_CHECK", "0") != "1", mass_target,
            _merra2_window_scratch(FT, target_grid.Nc, Nz),
            allocate_merra2_window_physics(settings, target_grid.Nc, Nz, FT))
        diag = MERRA2DayDiagnostics(nwindow, Int(steps_per_window))

        _read_block!(d, cur, 1)
        for blk in 1:nblock
            t_read = time()
            blk < nblock ? _read_block!(d, nxt, blk + 1) : _read_next_day_endpoint!(d, nxt, cur)
            t_read = time() - t_read
            _emit_block!(d, cur, nxt, blk, diag)
            @info @sprintf("    Block %d/%d done (next endpoint read %.2fs)", blk, nblock, t_read)
            cur, nxt = nxt, cur
        end

        set_streaming_steps_per_window_schedule!(writer.inner, diag.steps)
        # Gate positivity before promoting, so a failing day is quarantined.
        summarize_cs_positivity_status(diag.positivity; cfl_limit = positivity_cfl_limit,
                                       require_substep_positivity,
                                       steps_per_window = maximum(diag.steps),
                                       quarantine_path = writer_staging_path(writer))
        promote_streaming_binary!(writer)

        elapsed = time() - t_start
        @info @sprintf("MERRA-2 → CS day complete: %.1fs (%.2fs/window). substeps=[%d..%d]. Worst bal pre=%.2e post=%.2e iter=%d.",
                       elapsed, elapsed / nwindow, extrema(diag.steps)..., diag.pre, diag.post, diag.cg_iter)
        diag.replay_win > 0 &&
            @info @sprintf("  Worst replay: rel=%.2e abs=%.2e at win=%d",
                           diag.replay_rel, diag.replay_abs, diag.replay_win)
        return nothing
    finally
        close_merra2_day!(handles)
    end
end

# ===========================================================================
# Unified-CLI dispatch — wires the per-day driver into
# `preprocess_transport_binary.jl` via the standard
# `process_day(date, grid, settings, vertical; ...)` extension point.
# ===========================================================================

"""
    process_day(date, grid::CubedSphereTargetGeometry, settings::MERRA2Settings,
                vertical; out_path, mass_basis, dt_met_seconds, …)

Adapter that the unified preprocessor CLI calls into. Forwards to
[`process_merra2_to_cs_day`](@ref) with the kwargs the underlying function
accepts; the rest of the unified-CLI day-kwargs (e.g. `chain_mass`,
`seed_m`, `balance_mode`, `cm_closure`) are absorbed by the trailing
`kwargs...` and ignored — MERRA-2 has no day-to-day mass-chain state and the
flux balance is the fixed Cameron-Smith column pressure-fix.

Returns `(; final_m = nothing, global_mass_target_kg)` so the unified CLI's
`seed_m`/`global_mass_target_kg` chain remains a no-op.
"""
function process_day(date::Date,
                     grid::CubedSphereTargetGeometry,
                     settings::MERRA2Settings,
                     vertical;
                     out_path::AbstractString,
                     mass_basis::Symbol = :dry,
                     dt_met_seconds::Real = 10800.0,
                     positivity_cfl_limit::Real = 0.95,
                     min_steps_per_window::Union{Integer, Nothing} = nothing,
                     adaptive_substeps::Bool = true,
                     substep_cfl_target::Real = 0.95,
                     max_steps_per_window::Integer = typemax(Int),
                     require_substep_positivity::Bool = true,
                     global_mass_pin::Bool = false,
                     global_mass_target_kg::Real = NaN,
                     kwargs...)
    steps_floor = min_steps_per_window === nothing ? 1 : Int(min_steps_per_window)
    process_merra2_to_cs_day(date, settings, grid;
        out_path              = out_path,
        Nz                    = vertical.Nz,
        mass_basis            = mass_basis,
        dt_met_seconds        = dt_met_seconds,
        steps_per_window      = steps_floor,
        adaptive_substeps     = adaptive_substeps,
        substep_cfl_target    = substep_cfl_target,
        max_steps_per_window  = max_steps_per_window,
        positivity_cfl_limit  = positivity_cfl_limit,
        require_substep_positivity = require_substep_positivity,
        cache_dir             = grid.cache_dir,
        global_mass_pin       = global_mass_pin,
        global_mass_target_kg = global_mass_target_kg)
    return (; final_m = nothing,
            global_mass_target_kg = global_mass_target_kg)
end

# Source/target support matrix entry for the TOML entrypoint.
preprocessor_pair_supported(::CubedSphereTargetGeometry, ::MERRA2Settings) = true

# Output filename: clear MERRA-2 prefix, matching the other native writers.
_native_output_filename(::MERRA2Settings, date::Date, FT::Type) =
    "merra2_transport_$(Dates.format(date, "yyyymmdd"))_$(FT === Float32 ? "float32" : "float64").bin"
