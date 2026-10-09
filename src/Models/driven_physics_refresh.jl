# DrivenSimulation: per-window physics refresh (diffusion dz and Kz, convection forcing) and midpoint forcing.
# Split from DrivenSimulation.jl (refactor phase 4); included by Models.jl in this order.

# Surface-source shape and compatibility helpers live with the operator.
using ..Operators.SurfaceFlux: _surface_shape,
                                _check_surface_source_compatibility
using ..Operators.Diffusion: NoDiffusion, ImplicitVerticalDiffusion,
                              fill_dz_hydrostatic_constT!,
                              fill_dz_hydrostatic_virtualT!

# ---------------------------------------------------------------------------
# Diffusion layer-thickness refresh.
#
# `apply_vertical_diffusion_vmr!` uses `dz` per cell. The workspace allocator
# intentionally leaves `layer_thickness` undefined, so it must be refreshed
# from the just-loaded window's
# surface pressure + the grid's hybrid-σp coefficients each time the
# simulation advances to a new met window.
#
# `dz` only depends on (ps, ak, bk) (constant-T_ref hydrostatic); within a
# window ps is fixed, so one fill per window is correct and cheap.
# ---------------------------------------------------------------------------
@inline function _refresh_dz_for_window!(sim::DrivenSimulation)
    sim.model.diffusion isa NoDiffusion && return nothing
    diffusion_workspace = sim.model.workspace.diffusion_ws
    diffusion_workspace === nothing && return nothing
    layer_thickness = diffusion_workspace.layer_thickness
    vertical = sim.model.grid.vertical
    ps = sim.window.surface_pressure
    _fill_dz_for_diffusion!(layer_thickness, ps, vertical.A, vertical.B,
                             sim.model.diffusion, sim.window)
    return nothing
end

# When the diffusion operator uses LocalHoltslagBovilleKzField (which
# itself derives column geometry from VDIFF virtual-T), populate
# layer thickness from the SAME virtual-T-per-layer the Kz cache uses. Closes
# the previous inconsistency where the kernel divided by a 260 K-constant
# `dz` while Kz had been computed on layer-varying `dz`.
#
# All other diffusion configurations stay on the constant-T_ref path.
@inline _fill_dz_for_diffusion!(layer_thickness, ps, ak, bk, _diffop, _window) =
    fill_dz_hydrostatic_constT!(layer_thickness, ps, ak, bk)

@inline function _fill_dz_for_diffusion!(
        layer_thickness, ps, ak, bk,
        op::ImplicitVerticalDiffusion{FT, <:LocalHoltslagBovilleKzField},
        window) where FT
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
        return fill_dz_hydrostatic_constT!(layer_thickness, ps, ak, bk)
    end
    fill_dz_hydrostatic_virtualT!(layer_thickness, vdiff.t, vdiff.qv, ps, ak, bk)
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
        ::ImplicitVerticalDiffusion{FT, <:AbstractCSDkgField}, _window) where FT
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
            "Preprocess the binary with `scripts/preprocessing/preprocess_transport_binary.jl` " *
            "and `[tm5_convection] enable = true` in the preprocessing config, or fall back to " *
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

@inline _refresh_convection_forcing!(::NoConvection, ::TransportModel,
                                     ::TransportWindow) = nothing

function _refresh_convection_forcing!(::AbstractConvection, model::TransportModel,
                                      window::TransportWindow)
    copy_convection_forcing!(model.convection_forcing, window.convection)
    return nothing
end

function _refresh_forcing!(sim::DrivenSimulation, substep::Int)
    λ = _substep_fraction(substep, sim.steps_per_window, typeof(sim.Δt), sim.use_midpoint_forcing)
    if sim.interpolate_fluxes_within_window
        interpolate_fluxes!(sim.model.fluxes, sim.window, λ)
    else
        copy_fluxes!(sim.model.fluxes, sim.window.fluxes)
    end
    _apply_runtime_flux_storage_scale!(sim)
    expected_air_mass!(sim.expected_air_mass, sim.window, λ)
    if sim.qv_buffer !== nothing
        interpolate_qv!(sim.qv_buffer, sim.window, λ)
    end
    _refresh_convection_forcing!(sim.model.convection, sim.model, sim.window)
    return λ
end
