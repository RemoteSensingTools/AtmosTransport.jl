#!/usr/bin/env julia
# ===========================================================================
# compare_era5_geosit_met.jl — ERA5 CS vs GEOS-IT CS met data audit
#
# Compares mass fluxes, pressure thickness, humidity, and surface pressure
# between ERA5-derived cubed-sphere binary (MFLX format) and native GEOS-IT
# C180 NetCDF data for the same date (2021-12-01, window 1 = 00–01 UTC).
#
# CRITICAL: the two sources use DIFFERENT mass-flux bases:
#   ERA5  binary: am, bm are MOIST mass fluxes  (mass_basis="moist")
#   GEOS-IT raw:  MFXC, MFYC are DRY mass fluxes (FV3 dynamics)
#                 DELP is MOIST pressure thickness
#                 QV is specific humidity (from I3 collection)
#
# This script reports:
#   1. Per-panel statistics of each field (mean, std, min, max)
#   2. Derived comparisons on a common basis (moist→dry or dry→moist)
#   3. Global budget cross-checks (total atmospheric mass, flux magnitudes)
#
# Usage:
#   julia --project=. scripts/diagnostics/compare_era5_geosit_met.jl
# ===========================================================================

using Printf
using Statistics
using NCDatasets

# --- Paths -----------------------------------------------------------------

const ERA5_CS_BIN = expanduser(
    "~/data/AtmosTransport/met/era5/C90/transport_binary_v2_tropo34_dec2021_f64/era5_transport_v2_cs90_20211201_float64.bin"
)
const GEOSIT_CTM  = expanduser(
    "~/data/AtmosTransport/met/geosit_c180/raw_catrine/20211201/GEOSIT.20211201.CTM_A1.C180.nc"
)
const GEOSIT_I3   = expanduser(
    "~/data/AtmosTransport/met/geosit_c180/raw_catrine/20211201/GEOSIT.20211201.I3.C180.nc"
)

# --- Physical constants ----------------------------------------------------

const GRAV  = 9.80616   # m/s² (GEOS standard)
const R_EARTH = 6.371e6 # m

# --- Load ERA5 C90 binary (window 1) via TransportBinaryReader -----------

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

function load_era5_cs_window1(path)
    reader = AtmosTransport.MetDrivers.TransportBinaryReader(path; FT=Float64)
    h = reader.header
    Nc, Nz, npanel = h.geometry.Nc, h.nlevel, h.geometry.npanel
    @info @sprintf("ERA5 CS binary: C%d, %d levels, %d panels, mass_basis=%s",
                   Nc, Nz, npanel, h.mass_basis)
    @info @sprintf("  dt_met=%.0fs, window1 steps=%d, half_dt=%.0fs",
                   h.dt_met_seconds, h.steps_per_window_by_window[1],
                   h.dt_met_seconds/2)

    sections = h.payload_sections
    @info "  Payload sections: $(join(String.(sections), ", "))"

    # Parse header for element counts
    raw = h.raw_header
    n_m  = Int(raw["n_m"])
    n_am = Int(raw["n_am"])
    n_bm = Int(raw["n_bm"])
    n_cm = Int(raw["n_cm"])
    n_ps = Int(raw["n_ps"])

    # Window 1 data starts at offset 0 in the mmap'd data vector
    d = reader.data
    off = 0

    # m: (Nc, Nc, Nz) × 6 panels = n_m elements
    m_raw = d[off+1 : off+n_m]; off += n_m
    # am: (Nc+1, Nc, Nz) × 6 panels
    am_raw = d[off+1 : off+n_am]; off += n_am
    # bm: (Nc, Nc+1, Nz) × 6 panels
    bm_raw = d[off+1 : off+n_bm]; off += n_bm
    # cm: (Nc, Nc, Nz+1) × 6 panels
    cm_raw = d[off+1 : off+n_cm]; off += n_cm
    # ps: (Nc, Nc) × 6 panels
    ps_raw = d[off+1 : off+n_ps]; off += n_ps

    return (; Nc, Nz, npanel,
              m=m_raw, am=am_raw, bm=bm_raw, cm=cm_raw, ps=ps_raw,
              mass_basis=h.mass_basis, header=h)
end

# --- Load GEOS-IT C180 raw NetCDF (timestep 1) ----------------------------

function load_geosit_window1(ctm_path, i3_path)
    local mfxc, mfyc, delp, ps_ctm
    local qv, ps_i3, t_air

    NCDataset(ctm_path) do ds
        # Dimensions: (Xdim=180, Ydim=180, nf=6, lev=72, time=24)
        # Julia reads as (Xdim, Ydim, nf, lev, time)
        @info @sprintf("GEOS-IT CTM_A1: C%d, %d levels, %d panels, %d timesteps",
                       size(ds["MFXC"], 1), size(ds["MFXC"], 4),
                       size(ds["MFXC"], 3), size(ds["MFXC"], 5))

        tidx = 1  # First 3-hourly window
        mfxc = Float64.(ds["MFXC"][:, :, :, :, tidx])  # (Nc, Nc, 6, Nz) Pa·m²
        mfyc = Float64.(ds["MFYC"][:, :, :, :, tidx])
        delp = Float64.(ds["DELP"][:, :, :, :, tidx])   # Pa
        ps_ctm = Float64.(ds["PS"][:, :, :, tidx])      # (Nc, Nc, 6)

        # MFXC units: "Pa m+2" — accumulated over dynamics dt (~450s)
        @info @sprintf("  MFXC units: %s", ds["MFXC"].attrib["units"])
        @info @sprintf("  DELP units: %s", ds["DELP"].attrib["units"])

        # Check if vertical is bottom-to-top (GEOS-IT) or top-to-bottom
        mid = div(size(delp, 1), 2)
        delp_top = delp[mid, mid, 1, 1]
        delp_bot = delp[mid, mid, 1, end]
        if delp_top > 10 * delp_bot
            @info "  Detected bottom-to-top vertical ordering — flipping to TOA-first"
            mfxc = reverse(mfxc, dims=4)
            mfyc = reverse(mfyc, dims=4)
            delp = reverse(delp, dims=4)
        end
    end

    NCDataset(i3_path) do ds
        # I3 has 8 3-hourly timesteps (00, 03, 06, ..., 21 UTC)
        tidx = 1  # 00 UTC
        qv = Float64.(ds["QV"][:, :, :, :, tidx])   # (Nc, Nc, 6, Nz) kg/kg
        ps_i3 = Float64.(ds["PS"][:, :, :, tidx])  # (Nc, Nc, 6)
        t_air = Float64.(ds["T"][:, :, :, :, tidx])  # K

        @info @sprintf("GEOS-IT I3: QV shape=%s, PS shape=%s",
                       string(size(qv)), string(size(ps_i3)))

        # Same vertical flip
        mid = div(size(qv, 1), 2)
        if qv[mid, mid, 1, 1] < qv[mid, mid, 1, end] / 10
            # QV should be larger near surface; if level 1 << level end, flip
        else
            @info "  Detected bottom-to-top QV — flipping to TOA-first"
            qv = reverse(qv, dims=4)
            t_air = reverse(t_air, dims=4)
        end
    end

    Nc = size(mfxc, 1)
    Nz = size(mfxc, 4)
    return (; Nc, Nz,
              mfxc, mfyc, delp, ps=ps_ctm,
              qv, ps_i3, t_air)
end

# --- Reporting utilities ---------------------------------------------------

function field_stats(name, data; units="")
    mn = mean(data)
    sd = std(data)
    lo = minimum(data)
    hi = maximum(data)
    @info @sprintf("  %-20s  mean=%12.4e  std=%12.4e  min=%12.4e  max=%12.4e  %s",
                   name, mn, sd, lo, hi, units)
end

# ===========================================================================
# Main comparison
# ===========================================================================

function main()
    println("=" ^ 72)
    println("ERA5 CS vs GEOS-IT CS Met Data Audit")
    println("Date: 2021-12-01, Window 1 (00–01 UTC ERA5 / 00–03 UTC GEOS-IT)")
    println("=" ^ 72)

    # --- Check files exist ---
    for (label, path) in [("ERA5 CS binary", ERA5_CS_BIN),
                           ("GEOS-IT CTM_A1", GEOSIT_CTM),
                           ("GEOS-IT I3", GEOSIT_I3)]
        isfile(path) || error("$label not found: $path")
    end

    # --- Load data ---
    println("\n--- Loading ERA5 CS binary ---")
    era5 = load_era5_cs_window1(ERA5_CS_BIN)

    println("\n--- Loading GEOS-IT C180 raw ---")
    geos = load_geosit_window1(GEOSIT_CTM, GEOSIT_I3)

    # --- Resolution summary ---
    println("\n--- Resolution comparison ---")
    @info @sprintf("ERA5:    C%d  (%d levels, mass_basis=%s)", era5.Nc, era5.Nz, era5.mass_basis)
    @info @sprintf("GEOS-IT: C%d  (%d levels)", geos.Nc, geos.Nz)
    @info "NOTE: C90 vs C180 — statistics not cell-matched, comparing distributions only"

    # =======================================================================
    # Section 1: Raw field statistics
    # =======================================================================
    println("\n" * "=" ^ 72)
    println("Section 1: Raw field statistics")
    println("=" ^ 72)

    println("\n--- ERA5 CS binary (window 1) ---")
    field_stats("m (air mass)", era5.m; units="kg")
    field_stats("am (x-flux)", era5.am; units="kg/substep")
    field_stats("bm (y-flux)", era5.bm; units="kg/substep")
    field_stats("cm (z-flux)", era5.cm; units="kg/substep")
    field_stats("ps (surf press)", era5.ps; units="Pa")

    println("\n--- GEOS-IT raw (timestep 1) ---")
    field_stats("DELP", geos.delp; units="Pa")
    field_stats("MFXC (DRY)", geos.mfxc; units="Pa·m² accum")
    field_stats("MFYC (DRY)", geos.mfyc; units="Pa·m² accum")
    field_stats("PS", geos.ps; units="Pa")
    field_stats("QV", geos.qv; units="kg/kg")
    field_stats("T", geos.t_air; units="K")

    # =======================================================================
    # Section 2: Derived comparisons on common basis
    # =======================================================================
    println("\n" * "=" ^ 72)
    println("Section 2: Derived comparisons (common basis)")
    println("=" ^ 72)

    # --- 2a: Surface pressure ---
    println("\n--- Surface pressure [Pa] ---")
    era5_ps_mean = mean(era5.ps)
    geos_ps_mean = mean(geos.ps)
    @info @sprintf("  ERA5 <ps> = %.2f Pa  (from binary)", era5_ps_mean)
    @info @sprintf("  GEOS <ps> = %.2f Pa  (from CTM_A1)", geos_ps_mean)
    @info @sprintf("  Difference: %.2f Pa (%.4f%%)", era5_ps_mean - geos_ps_mean,
                   100 * (era5_ps_mean - geos_ps_mean) / geos_ps_mean)

    # --- 2b: Total atmospheric mass ---
    println("\n--- Total atmospheric mass [kg] ---")
    era5_total_m = sum(era5.m)
    # GEOS-IT mass from DELP: m_cell = DELP × area / g
    # Since we don't have per-cell areas for C180, approximate:
    # total mass ≈ sum(DELP_all_cells) × avg_cell_area / g
    # Better: M_atm ≈ 4π R² × <ps> / g
    era5_M = 4π * era5.header.planet_radius_m^2 * era5_ps_mean / GRAV
    geos_M = 4π * R_EARTH^2 * geos_ps_mean / GRAV
    @info @sprintf("  ERA5 Σm (from binary)     = %.6e kg", era5_total_m)
    @info @sprintf("  ERA5 M (from <ps>)        = %.6e kg", era5_M)
    @info @sprintf("  GEOS M (from <ps>)        = %.6e kg", geos_M)
    @info @sprintf("  ERA5/GEOS mass ratio      = %.6f", era5_M / geos_M)

    # --- 2c: Humidity budget ---
    println("\n--- Humidity (GEOS-IT QV) ---")
    geos_qv_global_mean = mean(geos.qv)
    geos_qv_surface = mean(geos.qv[:, :, :, end])
    geos_qv_tropo   = mean(geos.qv[:, :, :, max(1, geos.Nz - 10):geos.Nz])
    @info @sprintf("  <QV> global          = %.6e kg/kg", geos_qv_global_mean)
    @info @sprintf("  <QV> surface level   = %.6e kg/kg", geos_qv_surface)
    @info @sprintf("  <QV> bottom 10 levs  = %.6e kg/kg", geos_qv_tropo)

    # --- 2d: Mass flux magnitude comparison ---
    println("\n--- Mass flux magnitudes ---")
    println("  NOTE: ERA5 am/bm are MOIST [kg per substep]")
    println("        GEOS MFXC/MFYC are DRY [Pa·m² accumulated over dt_dyn≈450s]")

    # ERA5 flux stats
    era5_am_rms = sqrt(mean(era5.am .^ 2))
    era5_bm_rms = sqrt(mean(era5.bm .^ 2))

    # GEOS-IT: convert MFXC from Pa·m² to kg/s:
    #   mass_flux_kg_per_s = MFXC / (g × dt_dyn)
    # where MFXC is accumulated over dt_dyn ≈ 450 s
    dt_dyn = 450.0  # GEOS dynamics timestep [s]
    geos_am_kgs = geos.mfxc ./ (GRAV * dt_dyn)  # DRY kg/s
    geos_bm_kgs = geos.mfyc ./ (GRAV * dt_dyn)

    geos_am_rms = sqrt(mean(geos_am_kgs .^ 2))
    geos_bm_rms = sqrt(mean(geos_bm_kgs .^ 2))

    # ERA5 substep flux to kg/s: am is already in kg per substep
    # substep = dt_met / steps_per_window_by_window[1]
    era5_dt_sub = era5.header.dt_met_seconds /
                  era5.header.steps_per_window_by_window[1]
    era5_am_kgs_rms = era5_am_rms / era5_dt_sub
    era5_bm_kgs_rms = era5_bm_rms / era5_dt_sub

    @info @sprintf("  ERA5 |am| RMS (moist, kg/s) = %.4e", era5_am_kgs_rms)
    @info @sprintf("  GEOS |MFX| RMS (dry, kg/s)  = %.4e", geos_am_rms)
    @info @sprintf("  ERA5 |bm| RMS (moist, kg/s) = %.4e", era5_bm_kgs_rms)
    @info @sprintf("  GEOS |MFY| RMS (dry, kg/s)  = %.4e", geos_bm_rms)

    # Expected relationship: am_moist ≈ am_dry / (1 - qv)
    # For global mean qv ≈ 2.5e-3: factor ≈ 1.0025 — a 0.25% correction
    @info @sprintf("  Expected moist/dry ratio:     %.4f (from 1/(1-<qv>))",
                   1.0 / (1.0 - geos_qv_global_mean))

    # --- 2e: Vertical mass flux (ERA5 cm) ---
    println("\n--- Vertical mass flux (ERA5 binary only) ---")
    field_stats("cm (z-flux)", era5.cm; units="kg/substep")
    @info "  GEOS-IT raw does NOT include cm — would need to be diagnosed from continuity"

    # =======================================================================
    # Section 3: Key diagnostics
    # =======================================================================
    println("\n" * "=" ^ 72)
    println("Section 3: Key diagnostics & cross-checks")
    println("=" ^ 72)

    # --- 3a: Column pressure sum (GEOS DELP) ---
    geos_col_ps = dropdims(sum(geos.delp, dims=4), dims=4)  # (Nc, Nc, 6)
    @info @sprintf("  GEOS Σ(DELP) vs PS: mean diff = %.2f Pa  (should be ~0)",
                   mean(geos_col_ps .- geos.ps))
    @info @sprintf("  GEOS Σ(DELP) vs PS: max |diff| = %.2f Pa",
                   maximum(abs.(geos_col_ps .- geos.ps)))

    # --- 3b: Dry surface pressure (derived) ---
    geos_col_qv_dp = dropdims(sum(geos.qv .* geos.delp, dims=4), dims=4)
    geos_pw = geos_col_qv_dp  # column-integrated moisture [Pa]
    geos_ps_dry = geos.ps .- geos_pw
    @info @sprintf("  GEOS <ps_dry> = %.2f Pa  (ps - Σ(qv·dp))", mean(geos_ps_dry))
    @info @sprintf("  GEOS <ps_wet> = %.2f Pa  (reported PS)", mean(geos.ps))
    @info @sprintf("  GEOS <pw>     = %.2f Pa  (column water vapor pressure)", mean(geos_pw))

    # --- 3c: GEOS-IT vertical ordering check ---
    println("\n--- Vertical ordering check ---")
    @info @sprintf("  GEOS DELP[k=1] mean = %.4f Pa  (should be thin, TOA)", mean(geos.delp[:,:,:,1]))
    @info @sprintf("  GEOS DELP[k=Nz] mean = %.2f Pa  (should be thick, surface)", mean(geos.delp[:,:,:,end]))
    @info @sprintf("  GEOS QV[k=1] mean = %.4e  (should be dry, TOA)", mean(geos.qv[:,:,:,1]))
    @info @sprintf("  GEOS QV[k=Nz] mean = %.4e  (should be moist, surface)", mean(geos.qv[:,:,:,end]))

    # =======================================================================
    # Summary
    # =======================================================================
    println("\n" * "=" ^ 72)
    println("SUMMARY: Dry vs Moist Mass Flux Conventions")
    println("=" ^ 72)
    println("""
    ERA5 CS binary (MFLX format):
      - mass_basis = "moist"
      - am, bm are MOIST mass fluxes [kg per substep]
      - m is MOIST air mass [kg]
      - cm diagnosed from MOIST continuity
      - Poisson-balanced on MOIST basis
      - 34 merged levels (from ERA5 L137)

    GEOS-IT raw (CTM_A1.C180.nc):
      - MFXC, MFYC are DRY mass fluxes [Pa·m² accumulated over 450s]
      - DELP is MOIST pressure thickness [Pa]
      - QV from I3 is specific humidity [kg/kg]
      - 72 native levels (bottom-to-top in file, flipped to TOA-first above)
      - NO cm, NO Poisson balance in raw data

    To compare on a COMMON DRY basis:
      ERA5: am_dry = am_moist × (1 - qv_face)
            m_dry  = m_moist × (1 - qv)
      GEOS: am_dry = MFXC / (g × dt_dyn)  [already dry]
            m_dry  = DELP × area / g × (1 - qv)

    To compare on a COMMON MOIST basis:
      GEOS: am_moist = MFXC / (g × dt_dyn) / (1 - qv_face)
            m_moist  = DELP × area / g
    """)

    println("Audit complete.")
end

main()
