#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Catrine emission loaders — EDGAR SF6, Zhang Rn222, LMDZ/CAMS natural CO2.
#
# Verifies each new `surface_flux.kind` dispatch:
#   * resolves the canonical Catrine file path,
#   * loads the right variable on the right grid,
#   * applies the correct unit conversion to kg/m²/s,
#   * produces a global mass rate in the right order of magnitude.
#
# Requires the Catrine emission directory at
# `~/data/AtmosTransport/catrine/Emissions/`. The test is skipped when
# any of the source files is absent, so it can run on CI hosts that
# don't ship the dataset.
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport
using .AT.Models.InitialConditionIO: _load_file_surface_flux_field

const CATRINE_DIR = expanduser("~/data/AtmosTransport/catrine/Emissions")
const EDGAR_FILE  = joinpath(CATRINE_DIR, "edgar_v8",
                              "v8.0_FT2022_GHG_SF6_2022_TOTALS_emi.nc")
const ZHANG_FILE  = joinpath(CATRINE_DIR, "ZHANG_Rn222",
                              "Rn222_Emis_Zhang_Liu_et_al_05x05_mass.nc")
const LMDZ_FILE   = joinpath(CATRINE_DIR, "LMDZ_fluxes",
                              "z_cams_l_cams55_202112_FT24r2_ra_sfc_3h_co2_flux.nc")
const GRIDFED_FILE = joinpath(CATRINE_DIR, "gridfed",
                               "GCP-GridFEDv2024.0_2021.short.nc")

const HAS_ALL_DATA = all(isfile, (EDGAR_FILE, ZHANG_FILE, LMDZ_FILE, GRIDFED_FILE))

if !HAS_ALL_DATA
    @info "[Catrine emissions] one or more emission files missing — skipping"
    @testset "Catrine emissions — data unavailable" begin
        @test_skip true
    end
else
    @testset "edgar_sf6 — Tonnes/cell/year → kg/m²/s" begin
        f = _load_file_surface_flux_field(Dict("kind" => "edgar_sf6"), Float32; species = :sf6)
        @test f isa AT.Models.InitialConditionIO.FileSurfaceFluxField
        @test size(f.raw) == (3600, 1800)            # 0.1° × 0.1°
        @test minimum(f.lat) ≈ -89.95 atol = 0.1
        @test maximum(f.lat) ≈  89.95 atol = 0.1
        # All entries must be non-negative (emissions never sink).
        @test minimum(f.raw) >= 0
        # Order-of-magnitude check: peak ~3e-11 kg/m²/s (max per-cell
        # SF6 in industrial source clusters); won't exceed 1e-9.
        @test maximum(f.raw) < 1e-9
        # Global SF6 emissions ≈ 0.25 kg/s = ~8 kt/yr.
        global_rate = sum(Float64.(f.raw) .* AT.Models.InitialConditionIO._lonlat_cell_areas_m2(f.lon, f.lat))
        @test 0.05 < global_rate < 1.0
    end

    @testset "zhang_rn222 — kg/m²/s pass-through with monthly time index" begin
        f = _load_file_surface_flux_field(
            Dict("kind" => "zhang_rn222", "time_index" => 6), Float32; species = :rn222)
        @test f isa AT.Models.InitialConditionIO.FileSurfaceFluxField
        @test size(f.raw) == (720, 360)              # 0.5° × 0.5°
        @test minimum(f.raw) >= 0
        # Rn222 flux peak ~1e-20 kg/m²/s over high-radon land.
        @test maximum(f.raw) > 0
        @test maximum(f.raw) < 1e-18
        # Monthly index switching produces a different field.
        f2 = _load_file_surface_flux_field(
            Dict("kind" => "zhang_rn222", "time_index" => 1), Float32; species = :rn222)
        @test sum(abs, f2.raw .- f.raw) > 0
    end

    @testset "lmdz_co2 — kgC m⁻² s⁻¹ → kgCO2 m⁻² s⁻¹ with monthly mean" begin
        f = _load_file_surface_flux_field(Dict("kind" => "lmdz_co2"), Float32; species = :co2)
        @test f isa AT.Models.InitialConditionIO.FileSurfaceFluxField
        @test size(f.raw) == (360, 180)              # 1° × 1°
        # CO2 land flux can be either sign (net source or sink).
        @test minimum(f.raw) < 0    # at least one sink cell
        @test maximum(f.raw) > 0    # at least one source cell
        # Peak ~1.4 µg/m²/s after the kgC→kgCO2 (× M(CO2)/M(C)) conversion;
        # plausible for tropical biome cells.
        @test maximum(abs, f.raw) < 1e-5
        # Total net rate should be smaller than gross CO2 cycle.
        @test isfinite(f.native_total_mass_rate)
        @test abs(f.native_total_mass_rate) < 1e8
    end

    @testset "gridfed_fossil_co2 — kgCO2/month/m² → calendar-month /s" begin
        f = _load_file_surface_flux_field(
            Dict("kind" => "gridfed_fossil_co2", "time_index" => 12), Float32; species = :co2)
        @test f isa AT.Models.InitialConditionIO.FileSurfaceFluxField
        # GridFED monthly averaging produces a handful of cells with
        # tiny negative values from numerical artifacts; reject only
        # large negatives.
        @test minimum(f.raw) > -1e-6
        # December 2021 is converted with 31 calendar days, not the
        # old average-month convention. This matches GEOS-Chem's
        # embedded Emis_FossilCO2_Total for the CATRINE run.
        @test f.native_total_mass_rate ≈ 1.22939955e6 rtol = 2e-6
    end
end
