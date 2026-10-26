# DrivenSimulation: per-window physics refresh (diffusion dz and Kz, convection forcing) and midpoint forcing.
# Split from DrivenSimulation.jl (refactor phase 4); included by Models.jl in this order.

# Surface-source shape and compatibility helpers live with the operator.
using ..Operators.SurfaceFlux: _surface_shape,
                                _check_surface_source_compatibility,
                                stage_surface_flux_window!
using ..Operators.Diffusion: NoDiffusion, ImplicitVerticalDiffusion,
                              _stage_hybrid_coefficients!,
                              _recover_total_surface_pressure_staged!,
                              _fill_dz_hydrostatic_constT_staged!,
                              _fill_dz_hydrostatic_virtualT_staged!

# ---------------------------------------------------------------------------
# Diffusion layer-thickness refresh.
#
# `apply_vertical_diffusion_vmr!` uses `dz` per cell. The workspace allocator
# intentionally leaves `layer_thickness` undefined, so it must be refreshed
# from the just-loaded window's
# surface pressure + the grid's hybrid-σp coefficients each time the
# simulation advances to a new met window.
#
# `dz` depends on total `ps` in the hybrid coordinate. A dry-pressure binary is
# first reconstructed from window-start/VDIFF humidity. The Thomas-factor
# workspace's first vertical plane is safe scratch here because every diffusion
# solve rebuilds those factors after `dz` is refreshed.
# ---------------------------------------------------------------------------
@inline _refresh_dz_for_window!(sim::DrivenSimulation) =
    _refresh_dz_for_window!(sim, sim.model.diffusion)

@inline _refresh_dz_for_window!(_sim::DrivenSimulation, ::NoDiffusion) = nothing

# Dkg fields advance tracer mass with precomputed conservative exchange
# factors and never read `layer_thickness`; do not reconstruct an unused ps.
@inline _refresh_dz_for_window!(_sim::DrivenSimulation,
    ::ImplicitVerticalDiffusion{FT, <:AbstractCSDkgField}) where FT = nothing

@inline function _refresh_dz_for_window!(sim::DrivenSimulation, diffusion)
    diffusion_workspace = sim.model.workspace.diffusion_ws
    diffusion_workspace === nothing && return nothing
    layer_thickness = diffusion_workspace.layer_thickness
    vertical = sim.model.grid.vertical
    coefficients = _stage_hybrid_coefficients!(
        diffusion_workspace.hybrid_coefficients, vertical.A, vertical.B)
    ps = _diffusion_total_surface_pressure!(
        surface_pressure_basis(sim.driver), diffusion_workspace.factors,
        sim.window.surface_pressure, vertical.A, vertical.B,
        _diffusion_pressure_humidity(sim.window), coefficients)
    _fill_dz_for_diffusion!(layer_thickness, ps, vertical.A, vertical.B,
                             diffusion, sim.window, coefficients)
    return nothing
end

_diffusion_pressure_humidity(window) = window.qv_start !== nothing ? window.qv_start :
                                       window.vdiff === nothing ? nothing : window.vdiff.qv

@inline _diffusion_total_surface_pressure!(::TotalSurfacePressure, _factors,
                                           ps, _A, _B, _qv, _coefficients) = ps

function _diffusion_total_surface_pressure!(::DrySurfacePressure, _factors,
                                            ps, _A, _B, ::Nothing, _coefficients)
    @warn "diffusion received dry surface pressure without endpoint or VDIFF humidity; approximating total surface pressure as dry ps for hybrid layer thickness (typically 0.5-2% low in humid columns)" maxlog=1
    return ps
end

_diffusion_ps_scratch(factors::AbstractArray) = selectdim(factors, ndims(factors), 1)
_diffusion_ps_scratch(factors::NTuple{6}) =
    ntuple(p -> selectdim(factors[p], ndims(factors[p]), 1), 6)

function _diffusion_total_surface_pressure!(::DrySurfacePressure, factors,
                                            ps, _A, _B, qv, coefficients)
    scratch = _diffusion_ps_scratch(factors)
    _recover_total_surface_pressure_staged!(scratch, ps, qv, coefficients; sync = true)
    return scratch
end

# When the diffusion operator uses LocalHoltslagBovilleKzField (which
# itself derives column geometry from VDIFF virtual-T), populate
# layer thickness from the SAME virtual-T-per-layer the Kz cache uses. Closes
# the previous inconsistency where the kernel divided by a 260 K-constant
# `dz` while Kz had been computed on layer-varying `dz`.
#
# All other diffusion configurations stay on the constant-T_ref path.
@inline _fill_dz_for_diffusion!(layer_thickness, ps, _ak, _bk, _diffop, _window,
                                coefficients) =
    _fill_dz_hydrostatic_constT_staged!(layer_thickness, ps, coefficients; sync = true)

@inline function _fill_dz_for_diffusion!(
        layer_thickness, ps, ak, bk,
        op::ImplicitVerticalDiffusion{FT, <:LocalHoltslagBovilleKzField},
        window, coefficients) where FT
    vdiff = window.vdiff
    # Defensive fallback: if VDIFF isn't actually present on the window
    # (shouldn't happen — the diffusion runtime validator rejects this
    # case at config-load time), drop back to the constant-T_ref path and
    # warn loudly so a silently degraded config doesn't go unnoticed.
    if vdiff === nothing || !hasproperty(vdiff, :t) || !hasproperty(vdiff, :qv)
        @warn """
        _fill_dz_for_diffusion!: LocalHoltslagBovilleKzField was selected
        but the active window lacks `vdiff.t` / `vdiff.qv`. Falling back
        to constant-T_ref hydrostatic dz, which is INCONSISTENT with the
        Kz cache's virtual-T column geometry. Check the binary's VDIFF
        payload and the [diffusion] runtime config.
        """
        return _fill_dz_hydrostatic_constT_staged!(layer_thickness, ps, coefficients;
                                                   sync = true)
    end
    _fill_dz_hydrostatic_virtualT_staged!(layer_thickness, vdiff.t, vdiff.qv, ps,
                                          coefficients; sync = true)
    return layer_thickness
end

@inline _refresh_pbl_kz_for_window!(_field, _sim::DrivenSimulation) = nothing

function _refresh_pbl_kz_for_window!(field::WindowPBLKzField,
                                     sim::DrivenSimulation)
    mesh = sim.model.grid.horizontal
    refresh_pbl_kz_cache!(field, sim.window.surface, sim.window.air_mass,
                           mesh.cell_areas; halo_width = mesh.Hp)
    return nothing
end

function _refresh_pbl_kz_for_window!(field::LocalHoltslagBovilleKzField,
                                     sim::DrivenSimulation)
    mesh = sim.model.grid.horizontal
    refresh_local_holtslag_boville_kz_cache!(
        field, sim.window.surface, sim.window.vdiff, sim.window.air_mass,
        mesh.cell_areas; halo_width = mesh.Hp)
    return nothing
end

function _refresh_pbl_kz_for_window!(field::GCHPNonlocalPBLField,
                                     sim::DrivenSimulation)
    mesh = sim.model.grid.horizontal
    refresh_gchp_nonlocal_pbl!(field, sim.window.surface, sim.window.vdiff, sim.window.air_mass,
                               mesh.cell_areas, sim.model.grid.vertical;
                               halo_width = mesh.Hp)
    return nothing
end

function _refresh_pbl_kz_for_window!(field::PrecomputedCSDkgField,
                                     sim::DrivenSimulation)
    refresh_precomputed_cs_dkg_cache!(field, sim.window.dkg)
    return nothing
end

@inline function _fill_dz_for_diffusion!(layer_thickness, _ps, _ak, _bk,
        ::ImplicitVerticalDiffusion{FT, <:AbstractCSDkgField}, _window,
        _coefficients) where FT
    return layer_thickness
end

@inline _refresh_pbl_kz_for_window!(::NoDiffusion, _sim::DrivenSimulation) = nothing

function _refresh_pbl_kz_for_window!(op::ImplicitVerticalDiffusion,
                                     sim::DrivenSimulation)
    _refresh_pbl_kz_for_window!(op.kz_field, sim)
    return nothing
end

"""
    _validate_convection_window!(op, window, driver) -> nothing

Per-operator validation of a loaded transport window. Operator
authors add a method for their concrete type; the fallback method
throws `ArgumentError` naming the operator and pointing at this
function as the place to add a method.

Validation dispatches on the operator type rather than an
`if/elseif op isa …` chain, so adding `TM5Convection` (or any future
operator) only requires adding a method here.
"""
_validate_convection_window!(::NoConvection,
                              ::TransportWindow,
                              ::AbstractMetDriver) = nothing

function _validate_convection_window!(::CMFMCConvection,
                                       window::TransportWindow,
                                       driver::AbstractMetDriver)
    window.convection.cmfmc === nothing &&
        throw(ArgumentError(
            "CMFMCConvection requires `window.convection.cmfmc` to be populated; " *
            "driver $(typeof(driver)) provided convection forcing without CMFMC."))
    return nothing
end

function _validate_convection_window!(::TM5Convection,
                                       window::TransportWindow,
                                       driver::AbstractMetDriver)
    window.convection.tm5_fields === nothing &&
        throw(ArgumentError(
            "TM5Convection requires `window.convection.tm5_fields` " *
            "(NamedTuple with :entu, :detu, :entd, :detd) to be populated; " *
            "driver $(typeof(driver)) provided convection forcing without TM5 fields. " *
            "Build TM5 convection on the cubed sphere: an ERA5 N320 met source with " *
            "`include_convection = true` in its [preprocessing] table, or " *
            "`scripts/preprocessing/attach_catrine_tm5_convection_cs.jl`; or fall back to " *
            "`CMFMCConvection()` if you have GEOS-FP CMFMC data instead."))
    return nothing
end

# CMFMCMatrixConvection reads the SAME binary sections as CMFMCConvection
# (cmfmc + dtrain) — only the runtime numerics differ (GCHP two-pass vs
# the conservative TM5 LU on derived rates).
function _validate_convection_window!(::CMFMCMatrixConvection,
                                       window::TransportWindow,
                                       driver::AbstractMetDriver)
    window.convection.cmfmc === nothing &&
        throw(ArgumentError(
            "CMFMCMatrixConvection requires `window.convection.cmfmc` to be populated; " *
            "driver $(typeof(driver)) provided convection forcing without CMFMC."))
    window.convection.dtrain === nothing &&
        throw(ArgumentError(
            "CMFMCMatrixConvection requires `window.convection.dtrain` to be populated " *
            "(the matrix variant uses dtrain as the explicit detrainment rate); " *
            "driver $(typeof(driver)) provided CMFMC but no DTRAIN."))
    return nothing
end

function _validate_convection_window!(op::AbstractConvection,
                                       ::TransportWindow,
                                       ::AbstractMetDriver)
    throw(ArgumentError(
        "DrivenSimulation does not support convection operator $(typeof(op)) yet. " *
        "Add a `_validate_convection_window!(::$(typeof(op)), window, driver)` " *
        "method in `src/Models/driven_physics_refresh.jl` that checks its forcing " *
        "requirements."))
end

_validate_convection_runtime(model::TransportModel,
                             driver::AbstractMetDriver,
                             window::TransportWindow) =
    _validate_convection_runtime(model.convection, model, driver, window)

@inline _validate_convection_runtime(::NoConvection, ::TransportModel,
                                     ::AbstractMetDriver,
                                     ::TransportWindow) = nothing

function _validate_convection_runtime(op::AbstractConvection,
                                      model::TransportModel,
                                      driver::AbstractMetDriver,
                                      window::TransportWindow)
    window.convection === nothing &&
        throw(ArgumentError(
            "DrivenSimulation loaded a transport window without convection forcing, " *
            "but model.convection = $(typeof(op)) is active. " *
            "Install a driver/window path that populates `window.convection` " *
            "for this operator."))

    _validate_convection_window!(op, window, driver)
    return nothing
end

function _install_convection_forcing(model::TransportModel,
                                     driver::AbstractMetDriver,
                                     window::TransportWindow)
    _validate_convection_runtime(model, driver, window)
    return _install_convection_forcing(model.convection, model, window)
end

@inline _install_convection_forcing(::NoConvection, model::TransportModel,
                                    ::TransportWindow) = model

function _install_convection_forcing(::AbstractConvection, model::TransportModel,
                                     window::TransportWindow)
    forcing = allocate_convection_forcing_like(window.convection, model.state.air_mass)
    copy_convection_forcing!(forcing, window.convection)
    # A retained workspace must not reuse the previous driver's derived forcing.
    invalidate_cmfmc_cache!(model.workspace.convection_ws)
    return with_convection_forcing(model, forcing)
end

# A simulation that refills the previous simulation's window storage
# (`reuse_buffers_of`) also refills that model's convection forcing buffers.
function _reinstall_convection_forcing(model::TransportModel, driver::AbstractMetDriver,
                                       window::TransportWindow)
    _validate_convection_runtime(model, driver, window)
    _refresh_convection_forcing!(model.convection, model, window)
    invalidate_cmfmc_cache!(model.workspace.convection_ws)
    return model
end

@inline _refresh_convection_forcing!(::NoConvection, ::TransportModel,
                                     ::TransportWindow) = nothing

function _refresh_convection_forcing!(::AbstractConvection, model::TransportModel,
                                      window::TransportWindow)
    copy_convection_forcing!(model.convection_forcing, window.convection)
    return nothing
end

# Forcing of the step `substep` of the current window. Interpolated fluxes
# and humidity change every step. Window-constant fluxes and the convection
# forcing change only with the window: the operators read the model's copies
# and never write them, so they are copied (and the fluxes scaled) at the
# window's first step only. `sim.expected_air_mass` is not refreshed here: its
# only reader, the window-end air-mass reset (`air_mass_reset_target`),
# computes it itself.
function _refresh_forcing!(sim::DrivenSimulation, substep::Int)
    λ = _substep_fraction(substep, sim.steps_per_window, typeof(sim.Δt), sim.use_midpoint_forcing)
    if sim.interpolate_fluxes_within_window
        interpolate_fluxes!(sim.model.fluxes, sim.window, λ)
        _apply_runtime_flux_storage_scale!(sim)
    elseif substep == 1
        copy_fluxes!(sim.model.fluxes, sim.window.fluxes)
        _apply_runtime_flux_storage_scale!(sim)
    end
    if sim.qv_buffer !== nothing
        interpolate_qv!(sim.qv_buffer, sim.window, λ)
    end
    substep == 1 && _refresh_convection_forcing!(sim.model.convection, sim.model, sim.window)
    return λ
end
