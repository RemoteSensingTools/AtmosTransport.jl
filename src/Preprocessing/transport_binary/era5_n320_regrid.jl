# ===========================================================================
# ERA5 N320 → cubed-sphere transport-binary writer. The target may be any
# `Nc`; helpers and buffers named `c180` work for every target resolution.
#
# Drives one UTC day end-to-end:
#
#   per window (24 hourly):
#     1. Run the ERA5 per-window pipeline (B/C/D/E from sources/era5*.jl):
#        synthesise U/V/T/Q/PS on N320, derive dry-basis mass on the source
#        mesh, read UDMF/DDMF/UDRF/DDRF convection (optional), and
#        conservatively regrid PS / U / V / T / Q to the CS target.
#     2. Re-derive dry mass on the target from the regridded moist PS + Q so
#        the target-side column closure Σ_k DELP_dry = PS_dry holds to
#        roundoff, then apply the vertical plan and the optional global
#        dry-mass pin.
#     3. Rotate cell-centre winds geographic → panel-local using the CS
#        tangent basis.
#     4. Reconstruct Arakawa-C face mass fluxes (am, bm) with the configured
#        flux-construction method.
#
#   per window transition (windows 2..24):
#     5. Read the next window's pipeline output so we can close continuity
#        against the explicit endpoint-mass target.
#     6. Column Poisson-balance the current window's horizontal fluxes
#        against the next-window mass tendency.
#     7. Diagnose cm from the balanced fluxes + endpoint mass tendency.
#        With adaptive substeps, steps 6-7 repeat at a higher substep count
#        until the per-substep CFL is under `substep_cfl_target`; the
#        per-window schedule is stamped onto the header.
#     8. Verify the per-substep positivity gate and the write-time replay
#        gate. Update the worst-case accumulators.
#     9. Convert the next-window mass target into the forward `dm` payload
#        and stream-write the window to the staging binary.
#
# The final window closes against the next day's hour-0 endpoint, read from
# the next day's core GRIB. Only when that file is missing (archive boundary)
# does it fall back to a zero-tendency endpoint (`m_next = m_cur`) with a
# warning.
#
# Surface (PBL) fields (pblh, ustar, hflux, t2m) are written when
# `include_surface = true`: they are read from the 0.25° regular lat-lon
# surface data under `sfc_an_native/` (`era5_surface_reader.jl`) and
# conservatively regridded to the target. With
# `include_tm5_diffusion = true` the TM5 boundary-layer `dkg` payload is
# computed on the target from those fields.
#
# TM5 convection (entu/detu/entd/detd) is written when
# `include_convection = true`. The N320 forecast (UDMF/DDMF/UDRF/DDRF)
# is conservatively mapped to the target first, then converted to TM5 fields
# via `ec2tm_from_rates!` using target-grid thermodynamic geometry, and
# attached to the per-window writer payload as `window.tm5_fields`.
# CMFMC/DTRAIN is NOT written from this preprocessor; consumers that
# want CMFMC should read from a GEOS-IT binary or convert from TM5
# downstream.
# ===========================================================================

"""
    _fill_cs_mass_delta_payload!(dm_payload, m_cur, m_next)

Fill the on-disk forward endpoint-difference payload without mutating the
absolute next endpoint. The ERA5 writer's sliding-window loop swaps
`m_next` into `m_cur` after each write, so in-place conversion of the endpoint
would make the following window start from `m_next - m_cur`.
"""
function _fill_cs_mass_delta_payload!(dm_payload::NTuple{NP, <:AbstractArray{FT, 3}},
                                      m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                      m_next::NTuple{NP, <:AbstractArray{FT, 3}}) where {FT, NP}
    for p in 1:NP
        @inbounds for idx in eachindex(dm_payload[p])
            dm_payload[p][idx] = m_next[p][idx] - m_cur[p][idx]
        end
    end
    return dm_payload
end

"""
    _merge_era5_c180_state!(out_fields, out_m, out_delp_dry,
                            native_fields, native_m, native_delp_dry,
                            native_delp_moist, plan)

Collapse a target-grid ERA5 state from native levels onto `plan`. Extensive
dry mass and pressure thickness are summed exactly. Winds and thermodynamic
fields are pressure-mass weighted, which provides the merged state required
by TM5 boundary-layer diffusion; transport face fluxes are merged separately
after native-level reconstruction and therefore do not depend on these
averaged winds.
"""
function _merge_era5_c180_state!(out_fields::ERA5C180RegridFields,
                                  out_m, out_delp_dry,
                                  native_fields::ERA5C180RegridFields,
                                  native_m, native_delp_dry,
                                  native_delp_moist,
                                  plan::VerticalPlan)
    Nz_out = plan.Nz_output
    Threads.@threads :dynamic for p in 1:6
        copyto!(out_fields.ps[p], native_fields.ps[p])
        for a in (out_m[p], out_delp_dry[p], out_fields.u[p],
                  out_fields.v[p], out_fields.t[p], out_fields.qv[p])
            fill!(a, zero(eltype(a)))
        end

        @inbounds for k in 1:plan.Nz_native
            l = plan.merge_map[k]
            for j in axes(native_m[p], 2), i in axes(native_m[p], 1)
                w = native_delp_moist[p][i, j, k]
                out_m[p][i, j, l] += native_m[p][i, j, k]
                out_delp_dry[p][i, j, l] += native_delp_dry[p][i, j, k]
                out_fields.u[p][i, j, l] += native_fields.u[p][i, j, k] * w
                out_fields.v[p][i, j, l] += native_fields.v[p][i, j, k] * w
                out_fields.t[p][i, j, l] += native_fields.t[p][i, j, k] * w
                out_fields.qv[p][i, j, l] += native_fields.qv[p][i, j, k] * w
            end
        end

        @inbounds for l in 1:Nz_out, j in axes(out_m[p], 2), i in axes(out_m[p], 1)
            w = zero(eltype(native_delp_moist[p]))
            for k in plan.groups[l]
                w += native_delp_moist[p][i, j, k]
            end
            w > zero(w) || error("non-positive merged moist Δp at panel=$p i=$i j=$j level=$l")
            out_fields.u[p][i, j, l] /= w
            out_fields.v[p][i, j, l] /= w
            out_fields.t[p][i, j, l] /= w
            out_fields.qv[p][i, j, l] /= w
        end
    end
    return nothing
end

function _merge_cs_center_extensive!(out, native, plan::VerticalPlan,
                                      kind::AbstractFieldKind)
    for p in 1:6
        apply_vertical!(out[p], native[p], plan, kind)
    end
    return out
end

function _fill_era5_pin_sensitivity!(out, qv, dB, cell_areas, gravity)
    inv_g = inv(Float64(gravity))
    for p in 1:6
        s = out[p]
        q = qv[p]
        size(s) == size(q) || throw(DimensionMismatch(
            "ERA5 pin sensitivity panel $p shape $(size(s)) does not match qv $(size(q))"))
        @inbounds for k in axes(s, 3), j in axes(s, 2), i in axes(s, 1)
            s[i, j, k] = eltype(s)(Float64(dB[k]) *
                (1.0 - clamp(Float64(q[i, j, k]), 0.0, 0.999999)) *
                Float64(cell_areas[i, j]) * inv_g)
        end
    end
    return out
end

abstract type AbstractERA5PinSensitivityStrategy end
struct NoERA5MassPin <: AbstractERA5PinSensitivityStrategy end
struct NativeMergedPinSensitivity <: AbstractERA5PinSensitivityStrategy end
struct PurePressureMergePinCompatibility <: AbstractERA5PinSensitivityStrategy end

function _era5_pin_sensitivity_strategy(plan::VerticalPlan)
    native_dB = diff(plan.native_vc.B)
    merges_hybrid = any(plan.groups) do group
        length(group) > 1 && any(k -> !iszero(native_dB[k]), group)
    end
    return merges_hybrid ? NativeMergedPinSensitivity() :
                           PurePressureMergePinCompatibility()
end

_allocate_era5_pin_sensitivity(::NoERA5MassPin, _FT, _Nc, _Nz_native, _Nz_output) =
    (native = nothing, merged = nothing)
_allocate_era5_pin_sensitivity(::PurePressureMergePinCompatibility,
                               _FT, _Nc, _Nz_native, _Nz_output) =
    (native = nothing, merged = nothing)
function _allocate_era5_pin_sensitivity(::NativeMergedPinSensitivity,
                                        ::Type{FT}, Nc, Nz_native, Nz_output) where FT
    return (
        native = ntuple(_ -> zeros(FT, Nc, Nc, Nz_native), 6),
        merged = ntuple(_ -> zeros(FT, Nc, Nc, Nz_output), 6),
    )
end

function _pin_era5_endpoint_mass!(::NoERA5MassPin,
                                  _m_dry, _native_qv, _merged_qv,
                                  _native_pin_sensitivity, _merged_pin_sensitivity,
                                  _plan, _cell_areas, _gravity, _target_kg)
    return nothing
end

function _pin_era5_endpoint_mass!(::NativeMergedPinSensitivity,
                                  m_dry, native_qv, _merged_qv,
                                  native_pin_sensitivity, merged_pin_sensitivity,
                                  plan, cell_areas, gravity, target_kg)
    _fill_era5_pin_sensitivity!(native_pin_sensitivity, native_qv,
                                diff(plan.native_vc.B), cell_areas, gravity)
    _merge_cs_center_extensive!(merged_pin_sensitivity,
                                native_pin_sensitivity, plan, MassField())
    return _pin_cs_global_air_mass_from_sensitivity!(
        m_dry, merged_pin_sensitivity, target_kg)
end

function _pin_era5_endpoint_mass!(::PurePressureMergePinCompatibility,
                                  m_dry, _native_qv, merged_qv,
                                  _native_pin_sensitivity, _merged_pin_sensitivity,
                                  plan, cell_areas, gravity, target_kg)
    # Pure-pressure-only merges have zero sensitivity in every merged dB=0
    # group. Keep the direct merged-q path for the remaining singleton hybrid
    # layers so this no-op merge class retains its established rounding.
    return _pin_cs_global_air_mass!(m_dry, merged_qv, cell_areas, gravity,
                                    target_kg, diff(plan.merged_vc.B))
end

function _merge_era5_tm5_fields!(out::ERA5C180TM5ConvectionFields,
                                  native::ERA5C180TM5ConvectionFields,
                                  plan::VerticalPlan)
    for name in (:entu, :detu, :entd, :detd)
        _merge_cs_center_extensive!(getproperty(out, name), getproperty(native, name),
                                    plan, ConvectionTendencyField())
    end
    return out
end

"""
    _next_day_core_only_handle(handles)

Build a minimal next-day `ERA5GRIBDayHandles` that points only at the recorded
`next_core_path`. Used for the final-window mass-endpoint look-ahead, which
needs PS + Q from hour 0 of `date + 1` but not the convection or surface
streams. Returns `nothing` when `handles.next_core_path` is absent (archive
boundary day).
"""
function _next_day_core_only_handle(handles::ERA5GRIBDayHandles)
    handles.next_core_path === nothing && return nothing
    next_date = handles.date + Day(1)
    # In ARCO PS mode the endpoint PS comes from next_date's ARCO sp netCDF.
    arco_sp = nothing
    if handles.settings.arco_surface_pressure
        candidate = era5_arco_sp_path(handles.settings, next_date)
        isfile(candidate) ||
            error("ERA5 ARCO surface-pressure netCDF not found for endpoint " *
                  "$(next_date): $candidate")
        arco_sp = candidate
    end
    return ERA5GRIBDayHandles(
        handles.settings,
        next_date,
        handles.next_core_path,
        nothing,  # convection_path — not needed for mass endpoint
        nothing,  # surface_path
        nothing,  # next_core_path (chain stops here)
        nothing,  # prev_convection_path
        arco_sp,  # arco_sp_path (next_date)
    )
end

"""
    _endpoint_face_fluxes!(method, thickness, x, pipe, vc, gravity, dt_factor, Nc, Nz)

Face mass fluxes `x.am`, `x.bm` of one window endpoint (`pipe`, one hour) on the
native levels. `MoistFluxThickness` uses hybrid pressure thickness from total
surface pressure. `DryMassFluxThickness` uses the dry thickness
`(1-q) Δp = g m_dry / A` on the target or source grid, respectively.
"""
function _fill_era5_flux_thickness!(::MoistFluxThickness, x, pipe, vc, Nc, Nz)
    fill_cs_layer_thickness!(x.dp, pipe.c180_fields.ps, vc.A, vc.B, Nc, Nz)
    return nothing
end

function _fill_era5_flux_thickness!(::DryMassFluxThickness, x, pipe, vc, Nc, Nz)
    A, B = vc.A, vc.B
    @inbounds for p in 1:6, k in 1:Nz
        dA = Float64(A[k + 1]) - Float64(A[k])
        dB = Float64(B[k + 1]) - Float64(B[k])
        for j in 1:Nc, i in 1:Nc
            dp = abs(dA + dB * Float64(pipe.c180_fields.ps[p][i, j]))
            x.dp[p][i, j, k] = (1 - pipe.c180_fields.qv[p][i, j, k]) * dp
        end
    end
    return nothing
end

function _endpoint_face_fluxes!(method::Union{PanelAverageFluxes, VectorFaceFluxes}, thickness,
                                x, pipe, vc, gravity, dt_factor, Nc, Nz)
    _fill_era5_flux_thickness!(thickness, x, pipe, vc, Nc, Nz)
    _face_fluxes!(method, x, gravity, dt_factor, Nc, Nz)
    return nothing
end

function _endpoint_face_fluxes!(method::LineIntegralFaceFluxes, ::MoistFluxThickness,
                                x, pipe, vc, gravity, dt_factor, Nc, Nz)
    w = pipe.window_fields
    line_integral_face_fluxes!(x.am, x.bm, method, w.u, w.v, w.ps, vc.A, vc.B, gravity, dt_factor, Nz)
    return nothing
end

function _endpoint_face_fluxes!(method::LineIntegralFaceFluxes, ::DryMassFluxThickness,
                                x, pipe, vc, gravity, dt_factor, Nc, Nz)
    w = pipe.window_fields
    line_integral_face_fluxes!(x.am, x.bm, method, w.u, w.v,
                               pipe.dry_fields.delp_dry, gravity, dt_factor, Nz)
    return nothing
end

function _capture_era5_endpoint_flux_fidelity(method::Union{PanelAverageFluxes,VectorFaceFluxes},
                                              _thickness, x, pipe, ft, gravity,
                                              dt_factor, Nc, Nz, smap, _rows)
    return capture_cs_wind_flux_samples(method, x.am, x.bm, x.u_local, x.v_local,
                                        x.dp, ft, gravity, dt_factor, Nc, Nz;
                                        sample_map = smap)
end
function _capture_era5_endpoint_flux_fidelity(method::LineIntegralFaceFluxes,
                                              ::MoistFluxThickness, x, pipe, ft,
                                              gravity, dt_factor, Nc, Nz, smap, rows)
    w = pipe.window_fields
    # Thickness at the sampled stencil cells only, as the production moist
    # line integral computes it; no source-grid array per endpoint/retry.
    return capture_cs_wind_flux_samples(method, x.am, x.bm, w.u, w.v,
                                        HybridLayerThickness(pipe.vc.A, pipe.vc.B, w.ps),
                                        ft, gravity, dt_factor, Nc, Nz;
                                        sample_map = smap, line_rows = rows)
end
function _capture_era5_endpoint_flux_fidelity(method::LineIntegralFaceFluxes,
                                              ::DryMassFluxThickness, x, pipe, ft,
                                              gravity, dt_factor, Nc, Nz, smap, rows)
    w = pipe.window_fields
    return capture_cs_wind_flux_samples(method, x.am, x.bm, w.u, w.v,
                                        pipe.dry_fields.delp_dry, ft, gravity,
                                        dt_factor, Nc, Nz; sample_map = smap,
                                        line_rows = rows)
end

"""
    fill_tm5_dkg_payload!(dkg_c180, c180_fields, air_mass, hflux, lhflux,
                          ustar, A, B, Nc, constants, scratches)

Compute the exact TM5 interface exchange payload on C180. Unlike the legacy
centre-Kz payload, `dkg` already contains Kvh, target dry mass, and the
virtual-temperature geometry used during bldiff preprocessing.
"""
function fill_tm5_dkg_payload!(dkg_c180, c180_fields, air_mass,
                               hflux, lhflux, ustar, A, B, Nc::Int,
                               c::BLDiffConstants{FT},
                               scratches::Vector{BLDiffColumnScratch{FT}}) where FT
    for s in scratches
        _reset!(s.diag)
    end
    # `:static` pins each iteration to one thread, which the per-thread
    # `scratches[threadid()]` indexing requires.
    Threads.@threads :static for p in 1:6
        scratch = scratches[Threads.threadid()]
        dp, mp = dkg_c180[p], air_mass[p]
        tp, qp = c180_fields.t[p], c180_fields.qv[p]
        up, vp, psp = c180_fields.u[p], c180_fields.v[p], c180_fields.ps[p]
        hf, lf, us = hflux[p], lhflux[p], ustar[p]
        @inbounds for j in 1:Nc, i in 1:Nc
            tm5_bldiff_dkg_column!(
                view(dp, i, j, :), view(tp, i, j, :), view(qp, i, j, :),
                view(up, i, j, :), view(vp, i, j, :), view(mp, i, j, :),
                psp[i, j], hf[i, j], lf[i, j], us[i, j], A, B, c, scratch)
        end
    end
    entr = sum(s.diag.entr_fallback for s in scratches)
    floored = sum(s.diag.kvh_floored for s in scratches)
    max_kz = maximum(s.diag.max_kz for s in scratches; init = zero(FT))
    return (; entr_fallback = entr, kvh_floored = floored, max_kz,
            total_columns = 6 * Nc * Nc)
end

"""
    process_era5_n320_to_cs_day(date, settings, target_grid;
                                 out_path,
                                 Nz = ERA5_NATIVE_LEVEL_COUNT,
                                 vertical_plan = nothing,
                                 mass_basis = :dry,
                                 dt_met_seconds = ERA5_MET_INTERVAL_SECONDS,
                                 steps_per_window = 8,
                                 adaptive_substeps = true,
                                 substep_cfl_target = ERA5_N320_SUBSTEP_CFL_TARGET,
                                 max_steps_per_window = typemax(Int),
                                 cs_balance_tol = DEFAULT_POISSON_BALANCE_TOL,
                                 cs_balance_project_every = DEFAULT_POISSON_PROJECT_EVERY,
                                 positivity_cfl_limit = DEFAULT_POSITIVITY_CFL_LIMIT,
                                 require_substep_positivity = true,
                                 write_replay_check = true,
                                 cache_dir = nothing,
                                 include_convection = false,
                                 global_mass_pin = false,
                                 global_mass_target_kg = NaN,
                                 mass_fix_target_ps_dry_pa = NaN)

The float type is the target grid's (`target_grid::CubedSphereTargetGeometry{FT}`).

Generate a v4 cubed-sphere transport binary for one UTC `date` from the
ERA5 native-GRIB source described by `settings`, written to `out_path`.

`mass_basis` is fixed to `:dry` here — the writer pulls dry-basis layer
mass and dry surface pressure (re-derived on the target from regridded
PS + Q). Dry basis is the runtime default; a `:moist` request errors.

`steps_per_window` is the number of Strang substeps per met window. With
`adaptive_substeps = true` (default) it is the floor: each window's count is
raised until the per-substep CFL is under `substep_cfl_target` (at most
`max_steps_per_window`), and the per-window schedule is written into the
header. Each `am` / `bm` per-face slot stores the substep-mass amount. Per
face, the Courant number of a substep is `|F| / m_donor` (the donor cell
depends on the flux sign); the runtime subcycling gate sums each cell's
outgoing x, y and z face fluxes over the palindrome and divides by its mass
(`cs_subcycling.jl`).

The function writes through a unique sibling staging file and atomically
publishes `out_path` on success — a partial run leaves no usable file at the
requested path.
"""
function process_era5_n320_to_cs_day(date::Date,
                                       settings::ERA5N320Settings,
                                       target_grid::CubedSphereTargetGeometry{FT};
                                       out_path::AbstractString,
                                       Nz::Integer = ERA5_NATIVE_LEVEL_COUNT,
                                       vertical_plan::Union{Nothing, VerticalPlan} = nothing,
                                       mass_basis::Symbol = :dry,
                                       dt_met_seconds::Real = ERA5_MET_INTERVAL_SECONDS,
                                       steps_per_window::Integer = 8,
                                       adaptive_substeps::Bool = true,
                                       substep_cfl_target::Real = ERA5_N320_SUBSTEP_CFL_TARGET,
                                       max_steps_per_window::Integer = typemax(Int),
                                       cs_balance_tol::Real = DEFAULT_POISSON_BALANCE_TOL,
                                       cs_balance_project_every::Integer = DEFAULT_POISSON_PROJECT_EVERY,
                                       positivity_cfl_limit::Real = DEFAULT_POSITIVITY_CFL_LIMIT,
                                       require_substep_positivity::Bool = true,
                                       write_replay_check::Bool = true,
                                       cache_dir::Union{Nothing, AbstractString} = nothing,
                                       include_convection::Bool = false,
                                       global_mass_pin::Bool = false,
                                       global_mass_target_kg::Real = NaN,
                                       mass_fix_target_ps_dry_pa::Real = NaN,
                                       configured_last_day::Union{Nothing, Date} = nothing) where FT
    mass_basis === :dry ||
        throw(ArgumentError("ERA5 N320 → CS writer only supports mass_basis=:dry; got $(mass_basis)"))
    _validate_flux_construction(settings, "ERA5")
    _validate_flux_thickness(settings.flux_thickness, "ERA5")
    native_vc_cfg = load_hybrid_coefficients(settings.coefficients_file)
    native_vc_ft = HybridSigmaPressure(FT.(native_vc_cfg.A), FT.(native_vc_cfg.B))
    plan = if vertical_plan === nothing
        Int(Nz) == n_levels(native_vc_ft) ||
            throw(ArgumentError("`Nz` can no longer request a truncated ERA5 native " *
                                "column; pass a VerticalPlan to reduce L137 safely"))
        plan_vertical(IdentityVertical(), native_vc_ft)
    else
        vertical_plan
    end
    plan.Nz_native == n_levels(native_vc_ft) ||
        throw(DimensionMismatch("vertical plan native Nz=$(plan.Nz_native) does not " *
                                "match ERA5 coefficients Nz=$(n_levels(native_vc_ft))"))
    (plan.native_vc.A == native_vc_ft.A && plan.native_vc.B == native_vc_ft.B) ||
        throw(ArgumentError("vertical plan native A/B coefficients do not match " *
                            "`settings.coefficients_file`"))
    length(plan.merge_map) == plan.Nz_native ||
        throw(DimensionMismatch("vertical plan merge_map length $(length(plan.merge_map)) " *
                                "does not match native Nz=$(plan.Nz_native)"))
    Nz_native = plan.Nz_native
    Nz_int = plan.Nz_output
    steps_per_met = Int(steps_per_window)
    steps_per_met >= 1 || throw(ArgumentError("steps_per_window must be ≥ 1; got $(steps_per_met)"))
    # Adaptive per-window substep schedule (mirrors the GEOS path): start at
    # `steps_per_met` (the floor) and, per window, raise the substep count until
    # the per-substep vertical/horizontal CFL drops under `substep_cfl_target`
    # (kept below the hard `positivity_cfl_limit` gate). The runtime then honors
    # the recorded per-window schedule with no further substepping. A fixed
    # count cannot be correct across a day — see KEY_PARADIGMS §A5.
    substep_policy = SubstepSchedulePolicy(;
        adaptive_substeps = adaptive_substeps,
        substep_cfl_target = Float64(substep_cfl_target),
        min_steps_per_window = steps_per_met,
        max_steps_per_window = Int(max_steps_per_window))

    t_start = time()
    Nc = target_grid.Nc

    @info @sprintf("Process ERA5 N320 → CS day: date=%s, Nc=%d, Nz=%d, FT=%s",
                   string(date), Nc, Nz_int, string(FT))

    handles = open_era5_day(settings, date; next_day_handle = true)
    writer = nothing
    try
        # --- Allocate two ERA5 pipelines: current + next sliding-window. ---
        @info "  Allocating ERA5 N320 → C180 pipelines (×2, sliding window)..."
        cur_pipe = allocate_era5_n320_to_c180_pipeline(
            handles, target_grid; Nz = Nz_native,
            cache_dir = cache_dir,
            include_convection = include_convection,
            wind_regrid = settings.wind_regrid)
        nxt_pipe = allocate_era5_n320_to_c180_pipeline(
            handles, target_grid; Nz = Nz_native,
            cache_dir = cache_dir,
            include_convection = include_convection,
            wind_regrid = cur_pipe.wind,          # one set of wind-regrid buffers for both
            synthesis = cur_pipe.spectral_ws.synthesis)   # and one synthesis (reads never overlap)

        native_vc = cur_pipe.vc
        vc = plan.merged_vc
        mesh = target_grid.mesh

        do_mass_pin = global_mass_pin && isfinite(global_mass_target_kg)
        pin_strategy = do_mass_pin ? _era5_pin_sensitivity_strategy(plan) : NoERA5MassPin()
        pin_sensitivity = _allocate_era5_pin_sensitivity(
            pin_strategy, FT, Nc, Nz_native, Nz_int)

        # --- CS-side workspaces (dry mass on target, face flux, balance). ---
        cs_cell_areas = mesh.cell_areas   # (Nc, Nc), shared across panels
        native_m_dry    = ntuple(_ -> zeros(FT, Nc, Nc, Nz_native), 6)
        native_delp_dry = ntuple(_ -> zeros(FT, Nc, Nc, Nz_native), 6)
        native_ps_dry   = ntuple(_ -> zeros(FT, Nc, Nc), 6)
        native_ps_acc   = ntuple(_ -> zeros(Float64, Nc, Nc), 6)
        native_dp       = ntuple(_ -> zeros(FT, Nc, Nc, Nz_native), 6)
        native_u_local  = ntuple(_ -> zeros(FT, Nc, Nc, Nz_native), 6)
        native_v_local  = ntuple(_ -> zeros(FT, Nc, Nc, Nz_native), 6)
        native_am       = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz_native), 6)
        native_bm       = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz_native), 6)
        cur_fields      = allocate_era5_c180_regrid_fields(target_grid, Nz_int)
        nxt_fields      = allocate_era5_c180_regrid_fields(target_grid, Nz_int)
        cur_m_dry      = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        cur_delp_dry   = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        cur_ps_dry     = ntuple(_ -> zeros(FT, Nc, Nc), 6)
        cur_ps_dry_acc = ntuple(_ -> zeros(Float64, Nc, Nc), 6)
        cur_am         = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz_int), 6)
        cur_bm         = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz_int), 6)
        cur_cm         = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int + 1), 6)
        cur_dm_dry     = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        nxt_m_dry      = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        nxt_delp_dry   = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        nxt_ps_dry     = ntuple(_ -> zeros(FT, Nc, Nc), 6)
        nxt_ps_dry_acc = ntuple(_ -> zeros(Float64, Nc, Nc), 6)
        tm5_merged = include_convection ?
            allocate_era5_c180_tm5_convection_fields(target_grid, Nz_int) : nothing

        gravity = FT(STANDARD_GRAVITY)
        # Flux construction (`[preprocessing]` keys; defaults reproduce the
        # historical panel-average fluxes and mass-weighted column balance).
        flux_method = _face_flux_method(settings, target_grid; source = cur_pipe.source_grid.mesh)
        flux_thickness = _flux_thickness(settings.flux_thickness)
        flux_x = (am = native_am, bm = native_bm, u_local = native_u_local,
                  v_local = native_v_local, dp = native_dp)
        # Window-mean fluxes also need the face fluxes at the window end, and the
        # cell-centre methods their own winds and Δp there.
        window_mean = settings.flux_time_sampling === :window_mean
        flux_x_end = window_mean ? map(x -> map(zero, x), flux_x) : nothing
        fidelity_map = FidelitySampleMap(target_grid.face_table)
        fidelity_rows = flux_method isa LineIntegralFaceFluxes ?
            LineIntegralFidelityStencil(flux_method, fidelity_map) : nothing
        fidelity_capture = Ref{Any}(nothing)
        closure_record = Ref{AbstractColumnClosureRecord}(NoColumnClosureRecord())
        balance_weights = column_weights(settings.column_balance_weights, vc.B)
        # Per-substep flux scaling for a window taking `steps` substeps. The
        # face flux is the substep-mass amount, so it scales as 1/steps; the
        # adaptive loop re-reconstructs at the chosen `steps`.
        out_dt_factor_for(steps) = FT(dt_met_seconds / (2 * steps))
        # --- Surface PBL payload setup. ---
        # Surface fields live on a SEPARATE point-registered lat-lon 0.25° NetCDF
        # (`sfc_an_native/era5_surface_YYYYMM.nc`), read via the shared
        # `era5_surface_reader` (handles ERA5 unit/sign + orientation: pblh=blh,
        # ustar=zust, hflux=-sshf/3600, t2m=2t; lat S→N, lon centered). They are
        # regridded to C180 with a dedicated snapped-face LL → CS regridder (the same
        # `build_regridder` used by the LL→CS path), then written as the
        # `surface` payload.
        do_surface = settings.include_surface
        # Precompute the exact TM5 boundary-layer interface exchange on C180.
        # This needs latent heat flux in addition to the four PBL fields.
        do_tm5_diffusion = settings.include_tm5_diffusion
        if do_surface
            surf_Nx, surf_Ny = 1440, 721      # ERA5 0.25° single-levels regular-ll
            surf_reader = open_era5_surface_reader(
                joinpath(settings.root_dir, "sfc_an_native"), date, surf_Nx, surf_Ny)
            # Derive faces from the actual file coordinates. ERA5's 721
            # latitude rows include both poles, so SnappedFaces gives
            # half-width polar caps instead of treating the rows as 721
            # uniform cells spread over -90..90.
            surf_ll_mesh = era5_surface_source_mesh(surf_reader, mesh.radius)
            surf_regridder = build_regridder(surf_ll_mesh, mesh;
                                             normalize = false, cache_dir = cache_dir)
            surf_ws = allocate_cs_preprocess_workspace(
                Nc, surf_Nx, surf_Ny, 1,
                length(surf_regridder.src_areas),
                length(surf_regridder.dst_areas), FT)
            surf_pblh  = ntuple(_ -> zeros(FT, Nc, Nc), 6)
            surf_ustar = ntuple(_ -> zeros(FT, Nc, Nc), 6)
            surf_hflux = ntuple(_ -> zeros(FT, Nc, Nc), 6)
            surf_t2m   = ntuple(_ -> zeros(FT, Nc, Nc), 6)
            @info "  Surface PBL payload ENABLED (file-coordinate SnappedFaces 0.25° LL → C180)"
            if do_tm5_diffusion
                surf_lhflux = ntuple(_ -> zeros(FT, Nc, Nc), 6)   # latent flux on C180
                dkg_c180    = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
                # One scratch per thread; the panel loop threads over the 6
                # panels. Size by maxthreadid() (see synthesis-threading note).
                kz_scratch  = [BLDiffColumnScratch{FT}(Nz_int)
                               for _ in 1:Threads.maxthreadid()]
                kz_const    = BLDiffConstants{FT}()
                @info "  TM5 diffusion (bldiff) exact :dkg payload ENABLED"
            end
        end

        # Read + regrid the four PBL fields for the written window's hour
        # (0-indexed) into the shared `surf_*` CS panels. No-op when surface is
        # off. `load_era5_surface_window` takes a 1-based within-day window idx.
        fill_surface_payload! = function (written_hour::Int)
            do_surface || return nothing
            s = load_era5_surface_window(surf_reader, written_hour + 1, FT;
                                         with_latent = do_tm5_diffusion)
            regrid_2d_to_cs_panels!(surf_pblh,  surf_regridder, s.pblh,  surf_ws, Nc, IntensiveCellField())
            regrid_2d_to_cs_panels!(surf_ustar, surf_regridder, s.ustar, surf_ws, Nc, IntensiveCellField())
            regrid_2d_to_cs_panels!(surf_hflux, surf_regridder, s.hflux, surf_ws, Nc, IntensiveCellField())
            regrid_2d_to_cs_panels!(surf_t2m,   surf_regridder, s.t2m,   surf_ws, Nc, IntensiveCellField())
            do_tm5_diffusion &&
                regrid_2d_to_cs_panels!(surf_lhflux, surf_regridder, s.lhflux, surf_ws, Nc, IntensiveCellField())
            return nothing
        end

        # Build the surface/diffusion payload addition for the written window.
        surface_vdiff_payload = function (fields, air_mass, win_idx)
            extra = NamedTuple()
            if do_surface
                extra = merge(extra, (surface = (pblh = surf_pblh, ustar = surf_ustar,
                                                 hflux = surf_hflux, t2m = surf_t2m),))
            end
            if do_tm5_diffusion
                kzdiag = fill_tm5_dkg_payload!(dkg_c180, fields, air_mass,
                                               surf_hflux, surf_lhflux, surf_ustar,
                                               vc.A, vc.B, Nc, kz_const, kz_scratch)
                # Entrainment fallbacks are expected on a handful of cells; a
                # non-zero `kvh_floored` (the finite-payload guard) or a large
                # `entr_fallback` fraction flags a met problem to investigate.
                if kzdiag.kvh_floored > 0
                    @warn @sprintf("  Window %2d/%d: :dkg source-Kvh guard floored %d non-finite cell(s) — investigate met input",
                                   win_idx, nwindow, kzdiag.kvh_floored)
                elseif kzdiag.entr_fallback > 0
                    @info @sprintf("  Window %2d/%d: :dkg entrainment fallback on %d/%d cell(s) (%.4f%%), max Kvh=%.1f m²/s",
                                   win_idx, nwindow, kzdiag.entr_fallback, kzdiag.total_columns,
                                   100 * kzdiag.entr_fallback / kzdiag.total_columns, kzdiag.max_kz)
                end
                extra = merge(extra, (dkg = dkg_c180,))
            end
            return extra
        end

        # Global dry-air mass pin (mirror of the GEOS-CS path's
        # `_geos_pin_global_mass_if_needed!`). Removes the residual global-mean
        # dry-mass drift that ERA reanalysis carries through PS/Q, so the
        # binary's absolute dry mass matches a fixed target shared with the
        # GEOS path (consistency across driver families; see
        # docs/src/theory/mass_conservation.md). Applied to each
        # freshly derived endpoint mass; `ps` is recomputed from the pinned
        # mass so the stored surface pressure stays consistent. The fixed
        # `target_ps_dry`-derived target is independent of the start
        # day/window, which is required for the day-threaded build to produce
        # a globally coherent multi-day archive.
        do_mass_pin && @info @sprintf("  Global dry-mass pin ON: target=%.9e kg (%.3f Pa dry ⟨ps⟩)",
                                      Float64(global_mass_target_kg),
                                      Float64(global_mass_target_kg) * Float64(gravity) /
                                      (6 * sum(Float64, cs_cell_areas)))
        pin_endpoint_mass! = function (m_dry, ps_dry, native_qv, merged_qv)
            do_mass_pin || return nothing
            _pin_era5_endpoint_mass!(
                pin_strategy, m_dry, native_qv, merged_qv,
                pin_sensitivity.native, pin_sensitivity.merged,
                plan, cs_cell_areas, gravity, global_mass_target_kg)
            for p in 1:6
                _ps_from_air_mass!(ps_dry[p], m_dry[p], cs_cell_areas, gravity, Nc, Nz_int)
            end
            return nothing
        end

        # --- Open the streaming writer. ---
        nwindow = windows_per_day(settings, date)
        # Per-window substep schedule, filled adaptively below and stamped onto
        # the binary header before close so the runtime executes it directly.
        steps_schedule = fill(steps_per_met, nwindow)
        mkpath(dirname(out_path))
        tmp_path = _open_preprocessing_staging(out_path)
        @info @sprintf("  Output: %s (Nc=%d, Nz=%d, FT=%s)",
                       basename(out_path), Nc, Nz_int, string(FT))
        # The header records it and the gate uses it.
        write_replay_on = write_replay_check
        inner_writer = try
            open_streaming_cs_transport_binary(
            tmp_path, Nc, 6, Nz_int, nwindow, vc;
            FT = FT,
            dt_met_seconds = Float64(dt_met_seconds),
            half_dt_seconds = Float64(dt_met_seconds) / 2,
            steps_per_window = steps_per_met,
            source_flux_sampling = window_mean ? :window_mean : :window_start_endpoint,
            include_flux_delta = true,
            include_tm5conv = include_convection,
            include_surface = settings.include_surface,
            include_precomputed_dkg = settings.include_tm5_diffusion,
            mass_basis = mass_basis,
            surface_pressure_basis = DrySurfacePressure(),
            panel_convention = _cs_panel_convention_tag(target_grid),
            cs_definition = _cs_definition_tag(target_grid),
            cs_coordinate_law = _cs_coordinate_law_tag(target_grid),
            cs_center_law = _cs_center_law_tag(target_grid),
            longitude_offset_deg = longitude_offset_deg(cs_definition(mesh)),
            planet_radius = mesh.radius,
            extra_header = _with_replay_record(Dict{String, Any}(
                "preprocessor" => "process_era5_n320_to_cs_day",
                # Declare the per-window advection substep contract so the
                # runtime applies advection at the baked substep cadence but
                # runs convection + chemistry ONCE per met window (not per
                # substep). Without this flag `uses_binary_substep_contract`
                # is false and the driven loop falls into the per-substep
                # `step!` branch, running convection ~25× too often — the GEOS
                # cubed-sphere spectral writer sets the same key. The substep
                # schedule itself is already baked per window (steps_per_window).
                "runtime_substep_contract" => "binary_schedule",
                "preprocessor_contract" => "plan41_variable_substeps",
                # Declarative capability flag (matches the GEOS writers). The
                # runtime surfaces this into `caps.adaptive_substeps`; a config
                # that sets `[input].require_adaptive_substeps = true` rejects
                # binaries that omit it, even though the schedule IS adaptive.
                "adaptive_substeps" => substep_policy.adaptive_substeps,
                "source_type"  => "era5_n320_native_grib",
                "source_root"  => settings.root_dir,
                "target_type"  => "cubed_sphere",
                "regrid_method" => "conservative",
                # Fixed 2026-10-08; binaries without these keys predate the fixes.
                "regrid_coverage_normalized" => true,
                "reduced_gg_registration" => "interpolated_to_cell_centres",
                "vertical_transform" => string(nameof(typeof(plan.transform))),
                "vertical_Nz_native" => Nz_native,
                "vertical_Nz_output" => Nz_int,
                "merge_map" => plan.merge_map,
                "poisson_balanced" => true,
                "horizontal_balance" => HORIZONTAL_BALANCE_TAG,
                "column_balance_weights" => String(settings.column_balance_weights),
                "face_fluxes" => String(settings.face_fluxes),
                "flux_thickness" => String(settings.flux_thickness),
                "flux_face_lengths" => settings.face_fluxes === :panel_average ? String(settings.face_lengths) : "edge",
                "flux_time_sampling" => String(settings.flux_time_sampling),
                "face_interpolation" => flux_header_face_interpolation(flux_method, settings),
                "wind_regrid" => flux_header_wind_regrid(flux_method, settings),
                "tm5_convection_source" => include_convection ?
                    "ec2tm_from_rates(udmf,ddmf,udrf,ddrf)" : "none",
                "global_mass_pin_enabled" => do_mass_pin,
                "global_mass_pin_target_kg" => do_mass_pin ?
                    Float64(global_mass_target_kg) : nothing,
            ), write_replay_on))
        catch
            abandon_atomic_staging(tmp_path)
            rethrow()
        end
        do_mass_pin &&
            (inner_writer.header["mass_fix_target_ps_dry_pa"] = mass_fix_target_ps_dry_record(
                mass_fix_target_ps_dry_pa, global_mass_target_kg,
                6.0 * sum(Float64, cs_cell_areas), STANDARD_GRAVITY))
        writer = CubedSphereBinaryWriter(inner_writer,
                                          mass_basis_from_symbol(mass_basis);
                                          Nc = Nc,
                                          npanel = 6,
                                          final_path = String(out_path))

        replay_tol = replay_tolerance(FT)

        # Drive the native-L137 pipeline, derive native target-grid mass, then
        # collapse the target state with the configured vertical plan.
        function _process_window_to_cs!(win::Int,
                                          pipe::ERA5N320ToC180Pipeline,
                                          merged_fields::ERA5C180RegridFields,
                                          m_dry, delp_dry, ps_dry, ps_dry_acc)
            hour = win - 1
            process_era5_n320_window!(pipe, handles, date, hour)

            derive_c180_dry_mass!(native_m_dry, native_delp_dry,
                                   native_ps_dry, native_ps_acc,
                                   pipe.c180_fields.ps, pipe.c180_fields.qv,
                                   native_vc, cs_cell_areas)

            @inbounds for p in 1:6
                for k in 1:Nz_native
                    dA = Float64(native_vc.A[k + 1]) - Float64(native_vc.A[k])
                    dB = Float64(native_vc.B[k + 1]) - Float64(native_vc.B[k])
                    for j in 1:Nc, i in 1:Nc
                        native_dp[p][i, j, k] =
                            FT(abs(dA + dB * Float64(pipe.c180_fields.ps[p][i, j])))
                    end
                end
            end
            _merge_era5_c180_state!(merged_fields, m_dry, delp_dry,
                                     pipe.c180_fields, native_m_dry,
                                     native_delp_dry, native_dp, plan)
            for p in 1:6
                copyto!(ps_dry[p], native_ps_dry[p])
                copyto!(ps_dry_acc[p], native_ps_acc[p])
            end
            pin_endpoint_mass!(m_dry, ps_dry, pipe.c180_fields.qv, merged_fields.qv)
            return nothing
        end

        worst_pre = 0.0; worst_post = 0.0; worst_iter = 0
        worst_replay_rel = 0.0; worst_replay_abs = 0.0; worst_replay_win = 0
        worst_positivity = init_cs_positivity_accumulator()

        # Reconstruct + Poisson-balance + diagnose `cm` for the CURRENT window
        # (`pipe` outputs, balanced against `m_next`) at a given substep count.
        # `pipe` rotation + Δp are substep-independent, so they are computed once
        # before the adaptive loop; only the flux scaling (`out_dt_factor_for`),
        # the balance, the mass tendency, and `cm` depend on `steps`.
        _balance_window_at_steps! = function (pipe, pipe_end, m_dry, m_next, am, bm, cm, dm, steps)
            dt = out_dt_factor_for(steps)
            _endpoint_face_fluxes!(flux_method, flux_thickness, flux_x, pipe,
                                   native_vc, gravity, dt, Nc, Nz_native)
            start_capture = _capture_era5_endpoint_flux_fidelity(
                flux_method, flux_thickness, flux_x, pipe,
                target_grid.face_table, gravity, dt, Nc, Nz_native,
                fidelity_map, fidelity_rows)
            if window_mean    # trapezoidal rule: mean of the start and end fluxes
                _endpoint_face_fluxes!(flux_method, flux_thickness, flux_x_end, pipe_end,
                                       native_vc, gravity, dt, Nc, Nz_native)
                end_capture = _capture_era5_endpoint_flux_fidelity(
                    flux_method, flux_thickness, flux_x_end, pipe_end,
                    target_grid.face_table, gravity, dt, Nc, Nz_native,
                    fidelity_map, fidelity_rows)
                for p in 1:6
                    native_am[p] .= (native_am[p] .+ flux_x_end.am[p]) ./ 2
                    native_bm[p] .= (native_bm[p] .+ flux_x_end.bm[p]) ./ 2
                end
                start_capture = combine_captured_flux_samples(
                    start_capture, end_capture, native_am, native_bm,
                    target_grid.face_table; mean_endpoints = true)
            end
            _merge_cs_center_extensive!(am, native_am, plan, MassFluxField())
            _merge_cs_center_extensive!(bm, native_bm, plan, MassFluxField())
            fidelity_capture[] = merge_captured_flux_samples(
                start_capture, plan, am, bm, target_grid.face_table)
            bal_diag = balance_cs_column_mass_fluxes!(
                am, bm, m_dry, m_next,
                target_grid.face_table, target_grid.cell_degree, steps,
                target_grid.poisson_scratch; tol = Float64(cs_balance_tol),
                max_iter = DEFAULT_POISSON_BALANCE_MAX_ITER,
                project_every = Int(cs_balance_project_every),
                weights = balance_weights)
            sync_all_cs_boundary_mirrors!(am, bm, mesh.connectivity, Nc, Nz_int)
            fill_cs_window_mass_tendency!(dm, m_dry, m_next, steps)
            for p in 1:6; fill!(cm[p], zero(FT)); end
            cm_diag = _diagnose_balanced_cs_cm!(cm, am, bm, dm, m_dry, Nc, Nz_int,
                                                bal_diag, "ERA5 N320 window", balance_weights)
            closure_record[] = column_closure_record(bal_diag, cm_diag)
            return bal_diag
        end

        # Rotate `pipe` winds + build Δp (substep-independent), then adaptively
        # raise the substep count until the per-substep CFL drops under the
        # target (or the schedule converges). Returns the chosen `steps` and the
        # final balance diagnostics. Re-prepares at each candidate `steps`
        # (mirrors the GEOS path; guarantees continuity closes at that count).
        _adapt_window! = function (pipe, pipe_end, m_dry, m_next, am, bm, cm, dm)
            _prepare_cell_winds!(flux_method, flux_x, pipe.c180_fields.u, pipe.c180_fields.v,
                                 mesh, Nz_native)
            window_mean && _prepare_cell_winds!(flux_method, flux_x_end, pipe_end.c180_fields.u,
                                                pipe_end.c180_fields.v, mesh, Nz_native)
            @inbounds for p in 1:6
                for k in 1:Nz_native
                    dA = Float64(native_vc.A[k + 1]) - Float64(native_vc.A[k])
                    dB = Float64(native_vc.B[k + 1]) - Float64(native_vc.B[k])
                    for j in 1:Nc, i in 1:Nc
                        native_dp[p][i, j, k] =
                            FT(abs(dA + dB * Float64(pipe.c180_fields.ps[p][i, j])))
                    end
                end
            end
            steps = steps_per_met
            bal_diag = _balance_window_at_steps!(pipe, pipe_end, m_dry, m_next, am, bm, cm, dm, steps)
            if substep_policy.adaptive_substeps
                # `required_substeps` jumps directly to the count implied by the measured
                # CFL ratio, so 1-2 iterations converge; the cap only guards a
                # pathological non-converging window.
                for _ in 1:ADAPTIVE_SUBSTEP_MAX_REFINEMENTS
                    # Pass `m_next` so the refinement uses the SAME full-palindrome
                    # positivity ratio as the final `verify_cs_window_contract!`
                    # gate (the single-endpoint ratio without it undershoots →
                    # borderline windows pick too-few substeps and the gate
                    # rejects them). Mirrors the GEOS path.
                    pos = verify_substep_positivity_cs!(m_dry, am, bm, cm;
                                                        cfl_limit = substep_cfl_target,
                                                        m_next = m_next)
                    next = next_substeps(substep_policy, steps, pos.ratio)
                    next == steps && break
                    steps = next
                    bal_diag = _balance_window_at_steps!(pipe, pipe_end, m_dry, m_next, am, bm, cm, dm, steps)
                end
            end
            return steps, bal_diag
        end

        # --- Window 1. ---
        t0 = time()
        _process_window_to_cs!(1, cur_pipe, cur_fields,
                                cur_m_dry, cur_delp_dry, cur_ps_dry, cur_ps_dry_acc)
        @info @sprintf("    Window  1/%d: pipeline+regrid+vertical-merge %.2fs",
                       nwindow, time() - t0)

        # --- Windows 2..24: read next, balance current, diagnose cm, write. ---
        for win in 2:nwindow
            t0 = time()
            _process_window_to_cs!(win, nxt_pipe, nxt_fields,
                                    nxt_m_dry, nxt_delp_dry, nxt_ps_dry, nxt_ps_dry_acc)
            t_read = time() - t0

            # Restore + adaptively balance the CURRENT window against the NEXT
            # window's mass target. `_process_window_to_cs!` above overwrote
            # cur_am/bm while reading the next window; `_adapt_window!` rebuilds
            # them from the preserved `cur_pipe` outputs and raises the substep
            # count until the per-substep CFL clears the target. Returns the
            # per-window `win_steps` recorded into the schedule.
            t_bal = time()
            win_steps, bal_diag = _adapt_window!(cur_pipe, nxt_pipe, cur_m_dry, nxt_m_dry,
                                                 cur_am, cur_bm, cur_cm, cur_dm_dry)
            t_bal = time() - t_bal
            steps_schedule[win - 1] = win_steps

            worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
            worst_post = max(worst_post, bal_diag.max_post_residual)
            worst_iter = max(worst_iter, bal_diag.max_cg_iter)

            pos_diag = if write_replay_on
                contract = verify_cs_window_contract!(cur_m_dry, cur_am, cur_bm, cur_cm,
                                                       nxt_m_dry,
                                                       win_steps, win - 1;
                                                       replay_tol = replay_tol,
                                                       positivity_cfl_limit = positivity_cfl_limit)
                if worst_replay_win == 0 || contract.replay.max_rel_err > worst_replay_rel
                    worst_replay_rel = contract.replay.max_rel_err
                    worst_replay_abs = contract.replay.max_abs_err
                    worst_replay_win = win - 1
                end
                contract.positivity
            else
                verify_substep_positivity_cs!(cur_m_dry, cur_am, cur_bm, cur_cm;
                                              cfl_limit = positivity_cfl_limit,
                                              m_next = nxt_m_dry)
            end
            worst_positivity = update_cs_positivity_accumulator(worst_positivity, pos_diag, win - 1)

            _fill_cs_mass_delta_payload!(cur_dm_dry, cur_m_dry, nxt_m_dry)

            # Surface PBL for the written window (window win-1 → hour win-2).
            fill_surface_payload!(win - 2)

            base_payload = (m = cur_m_dry, am = cur_am, bm = cur_bm, cm = cur_cm,
                            ps = cur_ps_dry, dm = cur_dm_dry)
            payload = if include_convection
                _merge_era5_tm5_fields!(tm5_merged, cur_pipe.tm5_c180_fields, plan)
                merge(base_payload, (; tm5_fields = (
                    entu = tm5_merged.entu, detu = tm5_merged.detu,
                    entd = tm5_merged.entd, detd = tm5_merged.detd)))
            else
                base_payload
            end
            payload = merge(payload,
                            surface_vdiff_payload(cur_fields, cur_m_dry, win - 1))
            flux_evidence = wind_derived_evidence(
                CubedSphereTargetGeometry, fidelity_capture[], cur_am, cur_bm,
                target_grid.face_table;
                method = flux_fidelity_method(flux_method),
                samples_per_window = length(fidelity_map.faces),
                time_sampling = String(settings.flux_time_sampling),
                amount_seconds = dt_met_seconds / (2 * win_steps))
            omega_evidence = NoSourceOmegaEvidence{
                CubedSphereTargetGeometry,FT,SourceHasNoOmega}(SourceHasNoOmega())
            fidelity = FluxFidelityEvidenceBundle(flux_evidence, omega_evidence,
                                                  closure_record[])
            write_window!(writer, ReadyWindow{CubedSphereTargetGeometry, FT}(
                win - 1, payload, fidelity))

            @info @sprintf("    Window %2d/%d: wrote (steps=%d bal %.2fs pre=%.2e post=%.2e iter=%d) | read %2d (%.2fs)",
                            win - 1, nwindow, win_steps, t_bal, bal_diag.max_pre_residual,
                            bal_diag.max_post_residual, bal_diag.max_cg_iter,
                            win, t_read)

            # Swap current ↔ next. The pipelines themselves swap so the
            # GRIB read state stays paired with the dry mass we cached.
            cur_pipe, nxt_pipe = nxt_pipe, cur_pipe
            cur_fields, nxt_fields = nxt_fields, cur_fields
            cur_m_dry, nxt_m_dry         = nxt_m_dry, cur_m_dry
            cur_delp_dry, nxt_delp_dry   = nxt_delp_dry, cur_delp_dry
            cur_ps_dry, nxt_ps_dry       = nxt_ps_dry, cur_ps_dry
            cur_ps_dry_acc, nxt_ps_dry_acc = nxt_ps_dry_acc, cur_ps_dry_acc
        end

        # --- Final window: next-day hour-0 endpoint look-ahead. ---
        # The closed contract for window h is m(h+1) - m(h); the final
        # window's m_next must be the next day's hour-0 endpoint, not a
        # zero-tendency copy of m_cur. A zero-tendency fallback forces
        # Poisson balance to absorb the actual ERA5 wind divergence into
        # `cm`, which the positivity gate then (correctly) rejects.
        # Reuse `nxt_pipe`'s window/regrid buffers — those are convection-
        # agnostic so we skip the heavier `process_era5_n320_window!` path.
        next_handles = _next_day_core_only_handle(handles)
        if next_handles !== nothing
            try
                read_era5_n320_window_fields!(nxt_pipe.window_fields,
                                               nxt_pipe.spectral_ws,
                                               next_handles,
                                               handles.date + Day(1), 0)
            finally
                close_era5_day!(next_handles)
            end
            derive_n320_dry_mass!(nxt_pipe.dry_fields, nxt_pipe.window_fields,
                                  nxt_pipe.vc, nxt_pipe.cell_areas)
            regrid_n320_to_c180!(nxt_pipe.c180_fields,
                                   nxt_pipe.window_fields,
                                   nxt_pipe.regrid_ws,
                                   target_grid, nxt_pipe.wind)
            derive_c180_dry_mass!(native_m_dry, native_delp_dry,
                                   native_ps_dry, native_ps_acc,
                                   nxt_pipe.c180_fields.ps, nxt_pipe.c180_fields.qv,
                                   native_vc, cs_cell_areas)
            @inbounds for p in 1:6
                for k in 1:Nz_native
                    dA = Float64(native_vc.A[k + 1]) - Float64(native_vc.A[k])
                    dB = Float64(native_vc.B[k + 1]) - Float64(native_vc.B[k])
                    for j in 1:Nc, i in 1:Nc
                        native_dp[p][i, j, k] =
                            FT(abs(dA + dB * Float64(nxt_pipe.c180_fields.ps[p][i, j])))
                    end
                end
            end
            _merge_era5_c180_state!(nxt_fields, nxt_m_dry, nxt_delp_dry,
                                     nxt_pipe.c180_fields, native_m_dry,
                                     native_delp_dry, native_dp, plan)
            for p in 1:6
                copyto!(nxt_ps_dry[p], native_ps_dry[p])
                copyto!(nxt_ps_dry_acc[p], native_ps_acc[p])
            end
            pin_endpoint_mass!(nxt_m_dry, nxt_ps_dry,
                               nxt_pipe.c180_fields.qv, nxt_fields.qv)
        else
            _require_configured_terminal_day(
                handles.date, configured_last_day, "ERA5 N320 preprocessor",
                "next-day core endpoint for $(handles.date + Day(1))")
            # The configured archive boundary has no next_core_path on disk.
            # The positivity gate will likely
            # warn/fail in this case — that's correct: a zero-tendency
            # closure for the last hour is a known artifact, not a passing
            # binary. Run with [numerics].require_substep_positivity = false
            # only when you understand and accept this for boundary days.
            @warn "process_era5_n320_to_cs_day: archive-boundary fallback — " *
                  "no next-day core file for $(handles.date + Day(1)). The final " *
                  "window ends at an artificial zero-tendency endpoint (m_next = m_start) " *
                  "and its fluxes use the window-start winds; they are still balanced and " *
                  "cm diagnosed against that endpoint, so continuity closes algebraically " *
                  "and the replay gate runs, but the window's mass change is not physical."
            for p in 1:6
                copyto!(nxt_m_dry[p], cur_m_dry[p])
            end
        end

        # Final window: same adaptive balance against the next-day hour-0 mass
        # endpoint (or the zero-tendency boundary fallback above).
        # Without a next day the window end is unknown; the start fluxes stand in.
        final_steps, bal_diag = _adapt_window!(cur_pipe, next_handles === nothing ? cur_pipe : nxt_pipe,
                                               cur_m_dry, nxt_m_dry,
                                               cur_am, cur_bm, cur_cm, cur_dm_dry)
        steps_schedule[nwindow] = final_steps
        final_pos_diag = if write_replay_on
            contract = verify_cs_window_contract!(cur_m_dry, cur_am, cur_bm, cur_cm,
                                                   nxt_m_dry,
                                                   final_steps, nwindow;
                                                   replay_tol = replay_tol,
                                                   positivity_cfl_limit = positivity_cfl_limit)
            if worst_replay_win == 0 || contract.replay.max_rel_err > worst_replay_rel
                worst_replay_rel = contract.replay.max_rel_err
                worst_replay_abs = contract.replay.max_abs_err
                worst_replay_win = nwindow
            end
            contract.positivity
        else
            verify_substep_positivity_cs!(cur_m_dry, cur_am, cur_bm, cur_cm;
                                          cfl_limit = positivity_cfl_limit,
                                          m_next = nxt_m_dry)
        end
        worst_positivity = update_cs_positivity_accumulator(worst_positivity, final_pos_diag, nwindow)
        _fill_cs_mass_delta_payload!(cur_dm_dry, cur_m_dry, nxt_m_dry)

        # Surface PBL for the final window (window nwindow → hour nwindow-1).
        # VDIFF still comes from cur_pipe (current-day last hour); the look-ahead
        # above overwrote only nxt_pipe.c180_fields with next-day hour-0 winds.
        fill_surface_payload!(nwindow - 1)

        final_base_payload = (m = cur_m_dry, am = cur_am, bm = cur_bm, cm = cur_cm,
                              ps = cur_ps_dry, dm = cur_dm_dry)
        final_payload = if include_convection
            _merge_era5_tm5_fields!(tm5_merged, cur_pipe.tm5_c180_fields, plan)
            merge(final_base_payload, (; tm5_fields = (
                entu = tm5_merged.entu, detu = tm5_merged.detu,
                entd = tm5_merged.entd, detd = tm5_merged.detd)))
        else
            final_base_payload
        end
        final_payload = merge(final_payload,
                              surface_vdiff_payload(cur_fields, cur_m_dry, nwindow))
        # Without a next day the window-mean trapezoid uses the start fluxes
        # twice: the final window's effective sampling is the fallback.
        boundary_fallback = window_mean && next_handles === nothing
        boundary_fallback && (writer.inner.header["flux_time_sampling_effective"] =
                                  "window_start_boundary_fallback")
        flux_evidence = wind_derived_evidence(
            CubedSphereTargetGeometry, fidelity_capture[], cur_am, cur_bm,
            target_grid.face_table;
            method = flux_fidelity_method(flux_method),
            samples_per_window = length(fidelity_map.faces),
            time_sampling = boundary_fallback ? "window_start_boundary_fallback" :
                            String(settings.flux_time_sampling),
            amount_seconds = dt_met_seconds / (2 * final_steps))
        omega_evidence = NoSourceOmegaEvidence{
            CubedSphereTargetGeometry,FT,SourceHasNoOmega}(SourceHasNoOmega())
        final_fidelity = FluxFidelityEvidenceBundle(flux_evidence, omega_evidence,
                                                    closure_record[])
        write_window!(writer, ReadyWindow{CubedSphereTargetGeometry, FT}(
            nwindow, final_payload, final_fidelity))

        worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
        worst_post = max(worst_post, bal_diag.max_post_residual)
        worst_iter = max(worst_iter, bal_diag.max_cg_iter)

        # Stamp the adaptive per-window substep schedule onto the header so the
        # runtime executes it directly (no runtime CFL adaptation).
        set_streaming_steps_per_window_schedule!(writer.inner, steps_schedule)

        # Summarize positivity BEFORE publishing so a failed-gate day
        # quarantines the staged file (matches cubed_sphere_regrid.jl:610-617).
        summarize_cs_positivity_status(worst_positivity;
                                       cfl_limit = positivity_cfl_limit,
                                       steps_per_window = maximum(steps_schedule),
                                       require_substep_positivity = require_substep_positivity,
                                       quarantine_path = writer_staging_path(writer))
        promote_streaming_binary!(writer)

        elapsed = time() - t_start
        @info @sprintf("ERA5 N320 → C180 day complete: %.1fs (%.2fs/window). substeps=[%d..%d]. Worst bal pre=%.2e post=%.2e iter=%d.",
                        elapsed, elapsed / nwindow,
                        minimum(steps_schedule), maximum(steps_schedule),
                        worst_pre, worst_post, worst_iter)
        worst_replay_win > 0 &&
            @info @sprintf("  Worst replay: rel=%.2e abs=%.2e at win=%d",
                            worst_replay_rel, worst_replay_abs, worst_replay_win)
        return nothing
    finally
        if writer !== nothing && !writer.promoted
            try
                quarantine_streaming_binary!(writer)
            catch err
                @warn("ERA5 N320 preprocessor: failed to close staging writer during cleanup",
                      exception = (err, catch_backtrace()))
            end
        end
        close_era5_day!(handles)
        @isdefined(surf_reader) && close_era5_surface_reader(surf_reader)
    end
end

# ===========================================================================
# Unified-CLI dispatch — wires the per-day driver into
# `preprocess_transport_binary.jl` via the standard
# `process_day(date, grid, settings, vertical; ...)` extension point.
# ===========================================================================

"""
    process_day(date, grid::CubedSphereTargetGeometry, settings::AbstractERA5GRIBSettings,
                vertical; out_path, FT, mass_basis, dt_met_seconds, positivity_cfl_limit,
                kwargs...)

Adapter that the unified preprocessor CLI calls into. Forwards to
`process_era5_n320_to_cs_day` the substep policy (`min_steps_per_window` as
the floor, `adaptive_substeps`, `substep_cfl_target`, `max_steps_per_window`),
the positivity settings, `write_replay_check`, and the global dry-mass pin.
The remaining
unified-CLI day-kwargs (e.g. `chain_mass`, `seed_m`) are absorbed by the
trailing `kwargs...` and ignored — ERA5 N320 has no day-to-day mass-chain
state. The removed `horizontal_balance` writer keyword is rejected rather than
silently absorbed.

Returns `(; final_m = nothing)` so the unified CLI's `seed_m = get(result,
:final_m, nothing)` chain remains a no-op.
"""
function process_day(date::Date,
                     grid::CubedSphereTargetGeometry,
                     settings::AbstractERA5GRIBSettings,
                     vertical;
                     out_path::AbstractString,
                     mass_basis::Symbol = :dry,
                     dt_met_seconds::Real = ERA5_MET_INTERVAL_SECONDS,
                     positivity_cfl_limit::Real = DEFAULT_POSITIVITY_CFL_LIMIT,
                     require_substep_positivity::Bool = true,
                     write_replay_check::Bool = true,
                     min_steps_per_window::Union{Integer, Nothing} = nothing,
                     adaptive_substeps::Bool = true,
                     substep_cfl_target::Real = ERA5_N320_SUBSTEP_CFL_TARGET,
                     max_steps_per_window::Integer = typemax(Int),
                     global_mass_pin::Bool = false,
                     global_mass_target_kg::Real = NaN,
                     mass_fix_target_ps_dry_pa::Real = NaN,
                     configured_last_day::Union{Nothing, Date} = nothing,
                     kwargs...)
    reject_removed_horizontal_balance_keyword(kwargs)
    # The substep floor is the policy's min_steps_per_window (the entrypoint
    # resolves it from [numerics]); adaptive scheduling raises it per window to
    # satisfy CFL. Default floor 1 when the CLI passes nothing.
    steps_floor = min_steps_per_window === nothing ? 1 : Int(min_steps_per_window)
    process_era5_n320_to_cs_day(date, settings, grid;
        out_path                  = out_path,
        Nz                        = vertical.Nz_native,
        vertical_plan             = vertical.plan,
        mass_basis                = mass_basis,
        dt_met_seconds            = dt_met_seconds,
        steps_per_window          = steps_floor,
        adaptive_substeps         = adaptive_substeps,
        substep_cfl_target        = substep_cfl_target,
        max_steps_per_window      = max_steps_per_window,
        positivity_cfl_limit      = positivity_cfl_limit,
        require_substep_positivity = require_substep_positivity,
        write_replay_check        = write_replay_check,
        cache_dir                 = grid.cache_dir,
        include_convection        = settings.include_convection,
        global_mass_pin           = global_mass_pin,
        global_mass_target_kg     = global_mass_target_kg,
        mass_fix_target_ps_dry_pa = mass_fix_target_ps_dry_pa,
        configured_last_day       = configured_last_day)
    # Surface the fixed target so the unified driver's serial path can echo it
    # across days (no-op for the threaded path, which uses the config target).
    return (; final_m = nothing,
            global_mass_target_kg = global_mass_target_kg)
end

# Output filename matches the standalone CLI script's naming convention so
# downstream tools that already grep for `era5_n320_to_cNNN_transport_…`
# pick up the unified-CLI output without changes.
function _native_output_filename(::AbstractERA5GRIBSettings, date::Date, FT::Type)
    return "era5_n320_transport_$(Dates.format(date, "yyyymmdd"))_$(FT === Float32 ? "float32" : "float64").bin"
end
