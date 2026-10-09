#!/usr/bin/env julia
# Plan 24 Commit 6 demo: 2-day TM5 sim with two CO2 tracers
# (natural + fossil) on LL 720×361 Dec 2-3 2021.
#
# Tracers:
#   co2_natural : 400 ppm + 30 ppm × max(sin(lat), 0)² pattern
#                 (NH winter CO2 enhancement, well-mixed vertically)
#   co2_fossil  : 400 ppm global + 50 ppm at surface over NH
#                 30°-60°N latitude band (industrial belt pattern)
#
# No surface emissions — demonstrates pure transport + TM5 convection
# on two distinct initial conditions across 48 hours.
#
# Output: one NetCDF per snapshot at t = 0, 12, 24, 36, 48 h, each
# containing both tracers at the full (Nx, Ny, Nz) grid plus
# precomputed diagnostics (surface, 750 hPa layer, column-mean).

using Logging
using Printf
using NCDatasets
using Dates

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.State: CellState, DryBasis, allocate_face_fluxes, get_tracer
using .AtmosTransport.Operators: TM5Convection, UpwindScheme
using .AtmosTransport.MetDrivers: TransportBinaryDriver, load_transport_window,
                                   total_windows, steps_per_window
using .AtmosTransport.Models: TransportModel, DrivenSimulation, step!,
                               run_window!, window_index

# ─── Paths ───
const DAY1_BIN = "/temp1/tm5_smoke/transport_bin/era5_transport_20211202_merged1000Pa_float32.bin"
const DAY2_BIN = "/temp1/tm5_smoke/transport_bin/era5_transport_20211203_merged1000Pa_float32.bin"
const OUT_DIR  = "/temp1/tm5_smoke/snapshots"
mkpath(OUT_DIR)

const FT = Float32

# ─── Build ICs ───
#
# Called once using day-1's air_mass + grid coordinates.  Returns
# tracer arrays (Nx, Ny, Nz) in absolute mass units (kg/cell).
function build_initial_conditions(driver, air_mass)
    Nx, Ny, Nz = size(air_mass)
    lats = driver.grid.horizontal.φᶜ   # -90..90 in degrees

    baseline_ppm = 400.0f0
    # co2_natural: NH winter enhancement, lat-weighted.
    # "NH winter" pattern: max amplitude 30 ppm at NH mid-latitudes.
    co2_natural = zeros(FT, Nx, Ny, Nz)
    for j in 1:Ny
        lat_deg = lats[j]
        nh_factor = max(sind(lat_deg), 0f0)^2   # 0 in SH, peaks at NP
        ppm = baseline_ppm + 30f0 * nh_factor
        co2_natural[:, j, :] .= ppm * 1f-6
    end
    # co2_fossil: industrial-belt enhancement at surface k=Nz.
    co2_fossil = fill(baseline_ppm * 1f-6, Nx, Ny, Nz)
    for j in 1:Ny
        lat_deg = lats[j]
        if 30f0 <= lat_deg <= 60f0
            co2_fossil[:, j, Nz] = fill((baseline_ppm + 50f0) * 1f-6, Nx)
        end
    end

    # Convert volume mixing ratio to mass amount (kg per cell).
    return air_mass .* co2_natural, air_mass .* co2_fossil
end

# ─── Snapshot writer ───
function write_snapshot(path, state, driver, time_hours, air_mass)
    Nx, Ny, Nz = size(air_mass)
    lons = driver.grid.horizontal.λᶜ
    lats = driver.grid.horizontal.φᶜ

    nat = get_tracer(state, :co2_natural) ./ air_mass   # back to VMR
    fos = get_tracer(state, :co2_fossil)  ./ air_mass

    # 750 hPa index: the TM5 ab coefficients give pressure at
    # interfaces; pick the full-level whose midpoint is closest to
    # 75000 Pa.  Computed once from the grid's vertical coords.
    vc = driver.grid.vertical
    A_half = vc.A       # (Nz+1,)
    B_half = vc.B
    ps_mean = 98000f0   # crude; not used for sim, just for layer pick
    p_mid = [(FT(A_half[k]) + FT(B_half[k]) * ps_mean +
               FT(A_half[k+1]) + FT(B_half[k+1]) * ps_mean) / 2f0
              for k in 1:Nz]
    k_750 = argmin(abs.(p_mid .- 75000f0))

    # Diagnostics.
    nat_surface = nat[:, :, Nz]
    nat_750 = nat[:, :, k_750]
    # Mass-weighted column mean.
    mass_sum = sum(air_mass; dims = 3)[:, :, 1]
    nat_col  = sum(nat .* air_mass; dims = 3)[:, :, 1] ./ mass_sum

    fos_surface = fos[:, :, Nz]
    fos_750 = fos[:, :, k_750]
    fos_col = sum(fos .* air_mass; dims = 3)[:, :, 1] ./ mass_sum

    ds = NCDataset(path, "c")
    try
        defDim(ds, "longitude", Nx)
        defDim(ds, "latitude",  Ny)
        defDim(ds, "level",     Nz)
        defVar(ds, "longitude", Float32.(lons), ("longitude",))
        defVar(ds, "latitude",  Float32.(lats), ("latitude",))
        defVar(ds, "level",     Float32.(collect(1:Nz)), ("level",))
        ds.attrib["time_hours"] = Float64(time_hours)
        ds.attrib["k_750hPa"]   = Int32(k_750)
        ds.attrib["p_mid_Pa_at_k_750"] = Float64(p_mid[k_750])
        ds.attrib["baseline_ppm"] = 400.0

        # Full 3D tracers in ppm for inspection.
        defVar(ds, "co2_natural", Float32.(nat) .* 1f6,
                ("longitude", "latitude", "level"))
        defVar(ds, "co2_fossil",  Float32.(fos) .* 1f6,
                ("longitude", "latitude", "level"))

        # Pre-computed 2D diagnostics in ppm.
        defVar(ds, "co2_natural_surface", Float32.(nat_surface) .* 1f6,
                ("longitude", "latitude"))
        defVar(ds, "co2_natural_750hPa",  Float32.(nat_750) .* 1f6,
                ("longitude", "latitude"))
        defVar(ds, "co2_natural_column_mean", Float32.(nat_col) .* 1f6,
                ("longitude", "latitude"))
        defVar(ds, "co2_fossil_surface",  Float32.(fos_surface) .* 1f6,
                ("longitude", "latitude"))
        defVar(ds, "co2_fossil_750hPa",   Float32.(fos_750) .* 1f6,
                ("longitude", "latitude"))
        defVar(ds, "co2_fossil_column_mean", Float32.(fos_col) .* 1f6,
                ("longitude", "latitude"))
    finally
        close(ds)
    end
    @info "  Snapshot saved: $path (t=$(time_hours)h, k_750=$k_750)"
end

# ─── Run one day's worth of windows on the given driver ───
function run_one_day!(sim, driver, state, air_mass,
                      t_start_hours, snapshot_hours_within_day)
    n_windows = total_windows(driver)
    @info "  Starting day run: $n_windows windows, dt_met=$(driver.reader.header.dt_met_seconds)s"

    for w_target in 1:n_windows
        run_window!(sim)
        time_now = t_start_hours + w_target

        if (time_now - t_start_hours) in snapshot_hours_within_day
            snap_path = joinpath(OUT_DIR, @sprintf("snap_t%02d.nc", Int(time_now)))
            write_snapshot(snap_path, state, driver, time_now, air_mass)
        end
    end
end

# ─── Main ───
function main()
    @info "==============================================="
    @info "Plan 24 Commit 6 demo: 2-day TM5 sim, Dec 2-3 2021"
    @info "==============================================="

    isfile(DAY1_BIN) || error("Missing day-1 binary: $DAY1_BIN")
    isfile(DAY2_BIN) || error("Missing day-2 binary: $DAY2_BIN")

    # === Day 1 ===
    driver1 = TransportBinaryDriver(DAY1_BIN; FT = FT)
    win1 = load_transport_window(driver1, 1)
    air_mass = copy(win1.air_mass)
    Nx, Ny, Nz = size(air_mass)
    @info "Grid: ($Nx, $Ny, $Nz), Nt/day: $(total_windows(driver1))"

    co2_nat_init, co2_fos_init = build_initial_conditions(driver1, air_mass)
    state = CellState(air_mass; co2_natural = co2_nat_init, co2_fossil = co2_fos_init)

    # Save t=0 snapshot BEFORE building sim (sim may rearrange internals).
    write_snapshot(joinpath(OUT_DIR, "snap_t00.nc"), state, driver1, 0.0, air_mass)

    fluxes = allocate_face_fluxes(driver1.grid.horizontal, Nz; FT = FT, basis = DryBasis)
    model  = TransportModel(state, fluxes, driver1.grid, UpwindScheme();
                             convection = TM5Convection())
    sim1   = DrivenSimulation(model, driver1;
                               start_window = 1,
                               stop_window  = total_windows(driver1))

    initial_total_nat = sum(get_tracer(state, :co2_natural))
    initial_total_fos = sum(get_tracer(state, :co2_fossil))
    @info @sprintf("Initial mass — natural: %.6e kg, fossil: %.6e kg",
                    initial_total_nat, initial_total_fos)

    @info "--- Day 1 (Dec 2 2021) ---"
    run_one_day!(sim1, driver1, state, air_mass, 0, Set([6, 12, 18, 24]))
    close(driver1)

    # Day 2 (Dec 3) — rebuild sim around the next-day driver; state
    # (including tracer and air_mass) carries over. Plan 39 Commit G
    # removed the window-boundary air_mass reset, so cross-driver
    # hand-off no longer injects the pre-plan-39 discontinuity.
    driver2 = TransportBinaryDriver(DAY2_BIN; FT = FT)
    fluxes2 = allocate_face_fluxes(driver2.grid.horizontal, Nz; FT = FT, basis = DryBasis)
    model2  = TransportModel(state, fluxes2, driver2.grid, UpwindScheme();
                              convection = TM5Convection())
    sim2    = DrivenSimulation(model2, driver2;
                                start_window = 1,
                                stop_window  = total_windows(driver2))

    @info "--- Day 2 (Dec 3 2021) ---"
    run_one_day!(sim2, driver2, state, air_mass, 24, Set([6, 12, 18, 24]))
    close(driver2)

    # === Summary ===
    final_total_nat = sum(get_tracer(state, :co2_natural))
    final_total_fos = sum(get_tracer(state, :co2_fossil))
    drift_nat = (final_total_nat - initial_total_nat) / initial_total_nat * 100
    drift_fos = (final_total_fos - initial_total_fos) / initial_total_fos * 100
    @info @sprintf("Final   mass — natural: %.6e kg (drift %+.5f%%)",
                    final_total_nat, drift_nat)
    @info @sprintf("Final   mass — fossil:  %.6e kg (drift %+.5f%%)",
                    final_total_fos, drift_fos)
    @info "==============================================="
    @info "Done.  Snapshots in $OUT_DIR"
end

main()
