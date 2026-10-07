# ---------------------------------------------------------------------------
# Abstract met driver types and capability traits
#
# Met drivers are responsible for:
#   1. Reading native meteorological fields
#   2. Reconstructing pressures and layer masses
#   3. Computing basis-aware mass fluxes
#   4. Diagnosing or ingesting vertical fluxes
#   5. Enforcing continuity / closure
#
# The transport core should never see raw met fields.
# Met drivers produce AbstractFaceFluxState + CellState.air_mass.
# ---------------------------------------------------------------------------

"""
    AbstractMetDriver

Supertype for all meteorological data drivers.

## Required methods
    total_windows(driver) -> Int
    window_dt(driver) -> FT (seconds per met window)
    steps_per_window(driver) -> Int
    steps_per_window(driver, win_index) -> Int
    steps_per_window_schedule(driver) -> Vector{Int}
    load_transport_window(driver, win_index)
    driver_grid(driver)
    air_mass_basis(driver)
"""
abstract type AbstractMetDriver end

"""
    AbstractMassFluxMetDriver <: AbstractMetDriver

Reads pre-computed mass fluxes (am, bm, cm, m).
"""
abstract type AbstractMassFluxMetDriver <: AbstractMetDriver end

# --- Interface stubs ---
function total_windows end
function window_dt end
function steps_per_window end
function steps_per_window_schedule end
function load_transport_window end
function driver_grid end
function air_mass_basis end
function flux_interpolation_mode end
function flux_kind end
function uses_binary_substep_contract end

steps_per_window(d::AbstractMetDriver, _win::Integer) = steps_per_window(d)
steps_per_window_schedule(d::AbstractMetDriver) =
    fill(steps_per_window(d), total_windows(d))
flux_kind(::AbstractMetDriver) = :substep_mass_amount

"""
    flux_application_seconds(dt_seconds, steps, fk::Symbol) -> Float64

Seconds spanned by ONE stored flux amount — the normalization interval for
converting stored `am/bm/cm` mass amounts [kg] to rates or winds.
Per-substep storage: one palindrome application = `dt / (2 * steps)`.
Full-window storage: the full met window = `dt`. Every diagnostic that
divides a stored flux by a time interval must use this (not a hand-rolled
`dt/(2*steps)`), or it is wrong by `2 * steps` on full-window binaries.
"""
function flux_application_seconds(dt_seconds::Real, steps::Integer, fk::Symbol)
    fk === :full_window_mass_amount && return Float64(dt_seconds)
    fk === :substep_mass_amount && return Float64(dt_seconds) / (2 * steps)
    throw(ArgumentError("unknown flux_kind $(fk)"))
end

"""
    flux_storage_substep_scale(::Type{FT}, steps, fk::Symbol) -> FT

Multiplier converting STORED flux amounts to per-palindrome-application
amounts (what the transport kernels consume): `1` for per-substep storage,
`1/(2*steps)` for full-window storage. Transport-style script consumers
that feed raw window fluxes into kernels must scale by this (the runtime's
`DrivenSimulation` does so automatically at every forcing refresh).
"""
function flux_storage_substep_scale(::Type{FT}, steps::Integer, fk::Symbol) where {FT}
    fk === :full_window_mass_amount && return FT(1) / FT(2 * steps)
    fk === :substep_mass_amount && return one(FT)
    throw(ArgumentError("unknown flux_kind $(fk)"))
end

"""
    current_time(meteo) -> Float64

Simulation time [s] at the start of the next step. Threaded through
operator `apply!` methods:

    apply!(state, meteo, grid, op, dt; workspace)

Every operator that consumes time (`ExponentialDecay` rates,
`ImplicitVerticalDiffusion` Kz refresh, future emission-rate
`StepwiseField`s, etc.) reads `current_time(meteo)` once per call
and passes the resulting scalar to each `update_field!(f, t)`.

# Canonical usage

- **Production**: `meteo = sim::DrivenSimulation`; returns `sim.time`
  (Float64), set from the window and step counters after each `step!(sim)`.
  See `src/Models/DrivenSimulation.jl`.
- **Unit tests without a sim**: `meteo = nothing`; returns `0.0`.
"""
current_time(::Nothing) = 0.0

# ---------------------------------------------------------------------------
# Capability traits — what physics the met data supports
# ---------------------------------------------------------------------------

"""Does this driver provide diffusivity fields for boundary-layer diffusion?"""
supports_diffusion(::AbstractMetDriver) = false

"""Does this driver provide convective mass flux / detrainment for convection?"""
supports_convection(::AbstractMetDriver) = false

"""Does this driver provide native vertical mass fluxes (vs diagnosing from continuity)?"""
supports_native_vertical_flux(::AbstractMetDriver) = false

"""How should flux forcing vary within a met window?"""
flux_interpolation_mode(::AbstractMetDriver) = :constant

"""Does this driver provide a verified per-window timestep contract?"""
uses_binary_substep_contract(::Any) = false
uses_binary_substep_contract(::AbstractMetDriver) = false
uses_binary_substep_contract(::Nothing) = false

"""Does this driver provide specific humidity for dry-mass correction?"""
supports_moisture(::AbstractMetDriver) = false

export AbstractMetDriver, AbstractMassFluxMetDriver
export total_windows, window_dt, steps_per_window, steps_per_window_schedule,
       load_transport_window
export driver_grid, air_mass_basis, flux_interpolation_mode
export uses_binary_substep_contract
export supports_diffusion, supports_convection
export supports_native_vertical_flux, supports_moisture
export current_time
