#!/usr/bin/env julia
# Plan 39 F64 day-boundary probe — run on curry (A100, F64-capable).
#
# Purpose: separate F32 accumulation noise from a real cross-day
# hand-off inconsistency. The F32 2-day demo (wurst) shows a 2.6 ppm
# undershoot at t=30h (first snapshot after the day boundary) on a
# natural-CO₂ IC with min=400 ppm. If the undershoot collapses to F64
# ULP here, the residual is F32 precision. If it persists at the same
# magnitude, it is a real cross-day contract mismatch.
#
# Measures at every substep + day boundary:
#   1. tracer min/max
#   2. max |state.air_mass - next_window.m| / max(|m|)
#   3. boundary mismatch echoed via the canonical driven-runtime style
#
# The binaries must be the Float64 post-plan-39 canonical-contract
# files produced by `preprocess_transport_binary.jl` with
# `era5_ll720x361_v4_transport_binary.toml`. Output on NFS so curry
# sees them: ~/data/AtmosTransport/met/era5/ll720x361_v4/
#                 transport_binary_v2_tropo34_dec2021_f64/

using Printf

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.State: CellState, DryBasis, allocate_face_fluxes, get_tracer
using .AtmosTransport.Operators: UpwindScheme
using .AtmosTransport.MetDrivers: TransportBinaryDriver, load_transport_window,
                                   total_windows, steps_per_window
using .AtmosTransport.Models: TransportModel, DrivenSimulation, step!, run_window!

const DATA_DIR = expanduser("~/data/AtmosTransport/met/era5/ll720x361_v4/transport_binary_v2_tropo34_dec2021_f64")
const DAY1_BIN = joinpath(DATA_DIR, "era5_transport_20211202_merged1000Pa_float64.bin")
const DAY2_BIN = joinpath(DATA_DIR, "era5_transport_20211203_merged1000Pa_float64.bin")

function main()
    FT = Float64
    @info "F64 day-boundary probe"
    @info "  Day 1 binary: $DAY1_BIN"
    @info "  Day 2 binary: $DAY2_BIN"
    isfile(DAY1_BIN) || error("missing: $DAY1_BIN")
    isfile(DAY2_BIN) || error("missing: $DAY2_BIN")

    # === Day 1 setup ===
    driver1 = TransportBinaryDriver(DAY1_BIN; FT = FT)
    win1 = load_transport_window(driver1, 1)
    air_mass = copy(win1.air_mass)
    Nx, Ny, Nz = size(air_mass)
    @info "  Grid: ($Nx, $Ny, $Nz). steps/window=$(steps_per_window(driver1))"

    # ─── Uniform IC: min = max = IC_val to F64 ULP if advection is bit-flat.
    vmr_uni = FT(400e-6)
    tracer_uni = fill(vmr_uni, Nx, Ny, Nz) .* air_mass
    # ─── Variable IC (matches F32 2-day demo): natural-style,
    #     400 ppm SH + 30*sin²(lat) NH enhancement, min=400, max=430.
    lats = driver1.grid.horizontal.φᶜ
    vmr_var = zeros(FT, Nx, Ny, Nz)
    for j in 1:Ny
        ppm = FT(400) + FT(30) * max(sind(lats[j]), zero(FT))^2
        vmr_var[:, j, :] .= ppm * FT(1e-6)
    end
    tracer_var = air_mass .* vmr_var
    ic_var_min = minimum(vmr_var)
    ic_var_max = maximum(vmr_var)
    @info @sprintf("  Variable IC: min=%.4f max=%.4f ppm", ic_var_min*1e6, ic_var_max*1e6)

    for (label, tracer) in (("uniform", tracer_uni), ("variable", tracer_var))
        println()
        @info "════════ IC = $label ════════"
        state = CellState(air_mass; CO2 = copy(tracer))
        fluxes = allocate_face_fluxes(driver1.grid.horizontal, Nz; FT = FT, basis = DryBasis)
        model = TransportModel(state, fluxes, driver1.grid, UpwindScheme())
        sim1 = DrivenSimulation(model, driver1; start_window = 1, stop_window = total_windows(driver1))

        tracer_init = copy(get_tracer(sim1.model.state, :CO2))
        init_min_ppm = minimum(tracer_init ./ sim1.model.state.air_mass) * 1e6
        init_max_ppm = maximum(tracer_init ./ sim1.model.state.air_mass) * 1e6
        total_mass_init = sum(tracer_init)

        println("  DAY 1 (Dec 2 2021, 24 windows)")
        println("  step | win | VMR min (ppm) | VMR max (ppm) | below_IC_min (ppm) | above_IC_max (ppm)")
        while sim1.iteration < sim1.final_iteration
            step!(sim1)
            s = sim1.iteration
            if s in (1, 4, 8, 12, 24, 48, 72, 96)
                rm = get_tracer(sim1.model.state, :CO2) ./ sim1.model.state.air_mass
                mn, mx = extrema(rm)
                below = (init_min_ppm - mn*1e6)
                above = (mx*1e6 - init_max_ppm)
                @printf "  %4d | %3d | %13.6f | %13.6f | %+18.3e | %+18.3e\n" s sim1.current_window_index mn*1e6 mx*1e6 below above
            end
        end

        # Record state at day-1 end for cross-day compare
        state_day1_end_air_mass = copy(sim1.model.state.air_mass)

        # === Day boundary diagnostic ===
        driver2 = TransportBinaryDriver(DAY2_BIN; FT = FT)
        day2_win1_air_mass = copy(load_transport_window(driver2, 1).air_mass)
        boundary_abs = maximum(abs.(state_day1_end_air_mass .- day2_win1_air_mass))
        boundary_rel = boundary_abs / max(maximum(abs.(day2_win1_air_mass)), eps(FT))
        @info @sprintf("  DAY BOUNDARY: max|m_runtime_end_day1 - m_stored_day2_win1| / max|m| = %.3e (abs=%.3e kg)",
                        boundary_rel, boundary_abs)

        # === Day 2 run ===
        fluxes2 = allocate_face_fluxes(driver2.grid.horizontal, Nz; FT = FT, basis = DryBasis)
        model2 = TransportModel(state, fluxes2, driver2.grid, UpwindScheme())
        sim2 = DrivenSimulation(model2, driver2; start_window = 1, stop_window = total_windows(driver2))

        println()
        println("  DAY 2 (Dec 3 2021, 24 windows) — iteration count resets")
        println("  step | win | VMR min (ppm) | VMR max (ppm) | below_IC_min (ppm) | above_IC_max (ppm)")
        # Snapshot at t=0 of day 2 (before any step) — this is the state right
        # after the day-2 driver's window-1 air_mass was loaded at construction.
        rm0 = get_tracer(sim2.model.state, :CO2) ./ sim2.model.state.air_mass
        mn, mx = extrema(rm0)
        @printf "  %4d | %3d | %13.6f | %13.6f | %+18.3e | %+18.3e <- post-handoff, pre-step\n" 0 sim2.current_window_index mn*1e6 mx*1e6 (init_min_ppm - mn*1e6) (mx*1e6 - init_max_ppm)

        while sim2.iteration < sim2.final_iteration
            step!(sim2)
            s = sim2.iteration
            if s in (1, 4, 8, 24, 48, 72, 96)
                rm = get_tracer(sim2.model.state, :CO2) ./ sim2.model.state.air_mass
                mn, mx = extrema(rm)
                @printf "  %4d | %3d | %13.6f | %13.6f | %+18.3e | %+18.3e\n" s sim2.current_window_index mn*1e6 mx*1e6 (init_min_ppm - mn*1e6) (mx*1e6 - init_max_ppm)
            end
        end
        total_mass_end = sum(get_tracer(sim2.model.state, :CO2))
        mass_drift_pct = (total_mass_end - total_mass_init) / total_mass_init * 100
        @info @sprintf("  Total mass drift over 2 days (%s IC): %+.3e %%", label, mass_drift_pct)

        close(driver2)
    end
    close(driver1)
    @info "Done."
end

main()
