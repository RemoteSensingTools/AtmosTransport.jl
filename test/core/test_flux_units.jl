#!/usr/bin/env julia
#
# GriddedIO flux units: one grammar for every units string of a surface-flux
# input (design 2.4; P0 facts memo Section 3), conversion factors built from
# `Parameters` constants, species factors, calendar-month durations and the
# canonical strings written to flux files.

using Test
using Dates

using AtmosTransport
using AtmosTransport.GriddedIO: parse_flux_units, FluxUnits, TargetSpecies, Carbon, Species,
                                PerSquareMetre, PerCell, PerSecond, PerDay, PerMonth, PerYear,
                                kg_per_unit, period_seconds, mass_rate_factor,
                                canonical_units_string, MonthKey, TimeInterval, interval
using AtmosTransport.Parameters: SPECIES_MOLAR_MASS, CARBON_MOLAR_MASS, DAYS_PER_JULIAN_YEAR

const K = SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS        # M(CO2)/M(C)

@testset "every units string in use parses" begin
    per_m2_s(s, substance) = (u = parse_flux_units(s);
        u.area == PerSquareMetre() && u.time == PerSecond() && !u.molar &&
        u.amount_per_unit == 1.0 && u.substance == substance)
    # TRENDY v14 S3 (P0 facts memo, Section 3).
    @test per_m2_s("kg m-2 s-1", TargetSpecies())
    @test per_m2_s("kg  m-2 s-1", TargetSpecies())                         # ISAM, two spaces
    @test per_m2_s("kg.m-2.s-1", TargetSpecies())                          # CARDAMOM
    @test per_m2_s("kg m\$^{-2}\$ s\$^{-1}\$", TargetSpecies())             # CLASSIC
    @test per_m2_s("kg m-2 s-1, per unit land area occupied by the PFT", TargetSpecies())   # SDGVM ra
    # TRENDY fluxes are kg of carbon: the source declares it.
    u = parse_flux_units("kg m-2 s-1"; substance = :carbon)
    @test u.substance == Carbon()
    @test parse_flux_units("kg  m-2 s-1"; substance = "C").substance == Carbon()
    # CAMS/LMDZ and the other products.
    @test per_m2_s("kgC m-2 s-1", Carbon())
    u = parse_flux_units("gC m-2 d-1")
    @test (u.amount_per_unit, u.substance, u.area, u.time) == (1e-3, Carbon(), PerSquareMetre(), PerDay())
    u = parse_flux_units("g C per month")                                   # GFED5, per cell
    @test (u.amount_per_unit, u.substance, u.area, u.time) == (1e-3, Carbon(), PerCell(), PerMonth())
    u = parse_flux_units("g C month-1 cell-1")
    @test (u.amount_per_unit, u.substance, u.area, u.time) == (1e-3, Carbon(), PerCell(), PerMonth())
    u = parse_flux_units("mmol m-2 s-1"; substance = :co2)                  # ECCO-Darwin
    @test (u.amount_per_unit, u.molar, u.substance, u.area, u.time) ==
          (1e-3, true, Species(:co2), PerSquareMetre(), PerSecond())
    u = parse_flux_units("kgCO2/month/m2")                                   # GridFED
    @test (u.substance, u.area, u.time) == (Species(:co2), PerSquareMetre(), PerMonth())
    u = parse_flux_units("Tonnes")                                           # EDGAR
    @test (u.amount_per_unit, u.substance, u.area, u.time) == (1e3, TargetSpecies(), PerCell(), PerYear())
    # The runtime's per-second spellings.
    for s in ("kg/m2/s", "kg m-2 s-1", "kg/m2s", "kg/s/m2", "kg s-1 m-2", "kg/m²/s", "kg m^-2 s^-1",
              "KG M-2 S-1", " kg m-2 s-1 ")
        @test per_m2_s(s, TargetSpecies())
    end
    @test per_m2_s("kgCO2/m2/s", Species(:co2))
    @test per_m2_s("kg(CO2) m^-2 s^-1", Species(:co2))
    @test per_m2_s("kg CO2 m-2 s-1", Species(:co2))
    @test per_m2_s("kg(C)/m2/s", Carbon())
    @test per_m2_s("kgc/m2/s", Carbon())
    @test per_m2_s("kgsf6 m-2 s-1", Species(:sf6))
    # Other amounts and times.
    @test parse_flux_units("umol m-2 s-1"; substance = :co2).amount_per_unit == 1e-6
    @test parse_flux_units("µmol m-2 s-1"; substance = :co2).amount_per_unit == 1e-6
    @test parse_flux_units("mol C m-2 s-1").substance == Carbon()
    @test parse_flux_units("kgC m-2 yr-1").time == PerYear()
    @test parse_flux_units("kg m⁻² s⁻¹").time == PerSecond()
    @test parse_flux_units("kg/s").area == PerCell()
    @test parse_flux_units("kg m-2 s-1").text == "kg m-2 s-1"
    @test occursin("carbon", sprint(show, parse_flux_units("kgC m-2 s-1")))
end

@testset "rejected units" begin
    for s in ("", "   ", "mol m-2 s-1", "umol/m2/s", "mmol m-2 s-1", "kg m-2", "Tg/yr", "kg",
              "kg m-2 s-1, per unit grid-cell area", "kg(co2 m-2 s-1", "kg m-2 m-2 s-1",
              "kg m-2 s-1 d-1", "kelvin", "kg m-3 s-1")
        @test_throws ArgumentError parse_flux_units(s)
    end
    # A substance in the string and a different one from the source.
    @test_throws ArgumentError parse_flux_units("kgC m-2 s-1"; substance = :co2)
    @test_throws ArgumentError parse_flux_units("kgCO2 m-2 s-1"; substance = :carbon)
    @test parse_flux_units("kgCO2 m-2 s-1"; substance = :co2).substance == Species(:co2)
end

@testset "conversion factors from Parameters" begin
    feb2024 = interval(MonthKey(2024, 2))
    # Carbon → CO2 conserves carbon atoms: exactly M(CO2)/M(C).
    @test mass_rate_factor(parse_flux_units("kgC m-2 s-1"), :co2) === K
    @test kg_per_unit(parse_flux_units("kg m-2 s-1"; substance = :carbon), :co2) === K
    @test mass_rate_factor(parse_flux_units("gC m-2 d-1"), :co2) == 1e-3 * SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS / 86400
    @test mass_rate_factor(parse_flux_units("g C per month"), :co2, feb2024) ==
          1e-3 * SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS / (29 * 86400)
    @test mass_rate_factor(parse_flux_units("mmol m-2 s-1"; substance = :co2), :co2) == 1e-3 * SPECIES_MOLAR_MASS.co2
    @test mass_rate_factor(parse_flux_units("mol C m-2 s-1"), :co2) == SPECIES_MOLAR_MASS.co2
    @test mass_rate_factor(parse_flux_units("kgCO2/month/m2"), :co2, feb2024) == 1 / (29 * 86400)
    @test mass_rate_factor(parse_flux_units("Tonnes"), :sf6) == 1000 / (DAYS_PER_JULIAN_YEAR * 86400)
    @test mass_rate_factor(parse_flux_units("kg m-2 s-1"), :rn222) === 1.0
    @test mass_rate_factor(parse_flux_units("kgsf6 m-2 s-1"), :sf6) === 1.0
    @test period_seconds(parse_flux_units("kg m-2 d-1")) == 86400.0
    @test period_seconds(parse_flux_units("kg m-2 yr-1")) == DAYS_PER_JULIAN_YEAR * 86400
    # A per-month unit inside a day of the month: the month's length.
    day = TimeInterval(DateTime(2023, 2, 10), DateTime(2023, 2, 11))
    @test period_seconds(parse_flux_units("kgCO2/month/m2"), day) == 28 * 86400.0
    # Errors: no interval, an interval across months, wrong species.
    @test_throws ArgumentError period_seconds(parse_flux_units("kgCO2/month/m2"))
    @test_throws ArgumentError mass_rate_factor(parse_flux_units("kgCO2/month/m2"), :co2)
    @test_throws ArgumentError period_seconds(parse_flux_units("kgCO2/month/m2"),
                                              TimeInterval(DateTime(2023, 2, 20), DateTime(2023, 3, 2)))
    @test_throws ArgumentError mass_rate_factor(parse_flux_units("kgCO2 m-2 s-1"), :sf6)
    @test_throws ArgumentError mass_rate_factor(parse_flux_units("kgC m-2 s-1"), :sf6)
    @test_throws ArgumentError mass_rate_factor(parse_flux_units("mmol m-2 s-1"; substance = :n2o), :n2o)
end

@testset "canonical units string" begin
    @test canonical_units_string(:co2) == "kg CO2 m-2 s-1"
    @test canonical_units_string(:sf6) == "kg SF6 m-2 s-1"
    @test canonical_units_string(:rn222) == "kg Rn222 m-2 s-1"
    for species in propertynames(SPECIES_MOLAR_MASS)
        u = parse_flux_units(canonical_units_string(species))
        @test (u.substance, u.area, u.time, u.molar) == (Species(species), PerSquareMetre(), PerSecond(), false)
        @test mass_rate_factor(u, species) === 1.0
    end
    @test_throws ArgumentError canonical_units_string(:n2o)
end

const ICIO = AtmosTransport.Models.InitialConditionIO

@testset "surface-flux units (GriddedIO grammar)" begin
    # The loaders parse the units attribute with GriddedIO.parse_flux_units and
    # convert a slice to kg species m⁻² s⁻¹ by kg_per_unit and period_seconds.
    units(u, species = :co2) = ICIO._flux_units((; attrib = Dict("units" => u)), "F", "f.nc", species)
    density(u, species; iv = nothing, area = nothing, v = 2.0) =
        only(ICIO._to_flux_density!(fill(v, 1, 1), units(u, species), species, iv, area))
    # kg of the species (or unnamed kg) per m² per second: unchanged, bit for bit.
    for u in ("kg m-2 s-1", "kg/m2/s", "kgCO2/m2/s", "kg(CO2) m^-2 s^-1", "kg/m²/s", "kg CO2 m-2 s-1")
        @test density(u, :co2) === 2.0
    end
    @test density("kg m-2 s-1", :rn222) === 2.0
    @test density("kg SF6 m-2 s-1", :sf6) === 2.0
    # kg of carbon → CO2 conserves carbon atoms: M(CO2)/M(C).
    K = AtmosTransport.Parameters.SPECIES_MOLAR_MASS.co2 / AtmosTransport.Parameters.CARBON_MOLAR_MASS
    @test density("kgC m-2 s-1", :co2; v = 1.0) === K
    # Grams and days convert (the former per-second-only parser refused them).
    @test density("g m-2 s-1", :co2; v = 1.0) === 1e-3
    @test density("gC m-2 d-1", :co2; v = 86400.0) ≈ 1e-3 * K
    # Per-month values need their calendar month (GridFED); February 2024 has 29 days.
    @test density("kgCO2/month/m2", :co2; v = 29 * 86400.0,
                  iv = AtmosTransport.GriddedIO.interval(AtmosTransport.GriddedIO.MonthKey(2024, 2))) === 1.0
    @test_throws ArgumentError density("kgCO2/month/m2", :co2)
    # Per-cell amounts need the cell areas (time-varying loaders read densities).
    @test density("Tonnes", :sf6; v = 1.0, area = fill(2.0, 1, 1)) ===
          1000.0 / (2.0 * AtmosTransport.Parameters.DAYS_PER_JULIAN_YEAR * 86400)
    @test_throws ArgumentError density("kg/s", :co2)
    # Missing per-cell areas come from the same snapped source mesh used by
    # conservative regridding, including half-width polar caps. Preserve the
    # caller's latitude row order when the file stores north to south.
    lon = collect(45.0:90.0:315.0)
    lat = [-90.0, 0.0, 90.0]
    mesh = AtmosTransport.GriddedIO.source_latlon_mesh(
        lon, lat; radius = AtmosTransport.Parameters.EARTH_RADIUS)
    expected_area = [cell_area(mesh, i, j) for i in 1:4, j in 1:3]
    @test ICIO._lonlat_cell_areas_m2(lon, lat) == expected_area
    @test ICIO._lonlat_cell_areas_m2(lon, reverse(lat)) == expected_area[:, end:-1:1]
    # A molar flux that names no substance is of the tracer's species
    # (ECCO-Darwin writes mmol m-2 s-1 of CO2).
    M = AtmosTransport.Parameters.SPECIES_MOLAR_MASS
    @test density("mmol m-2 s-1", :co2; v = 1.0) ≈ 1e-3 * M.co2 rtol = 1e-15
    @test density("mol m-2 s-1", :sf6; v = 1.0) ≈ M.sf6 rtol = 1e-15
    @test density("umol/m2/s", :co2; v = 1.0) ≈ 1e-6 * M.co2 rtol = 1e-15
    # A substance named in the string wins over the tracer's species, with or
    # without a space or parentheses.
    K = AtmosTransport.Parameters.SPECIES_MOLAR_MASS.co2 / AtmosTransport.Parameters.CARBON_MOLAR_MASS
    for u in ("mol C m-2 s-1", "molC m-2 s-1", "mol (C) m-2 s-1")
        @test density(u, :co2; v = 1.0) ≈ AtmosTransport.Parameters.CARBON_MOLAR_MASS * K rtol = 1e-15
    end
    # Units of another species than the tracer's, missing and unknown units
    # are errors that name the variable and file.
    @test_throws ArgumentError density("kgCO2/m2/s", :sf6)
    @test_throws ArgumentError density("kgC m-2 s-1", :rn222)
    for u in ("", "kg m-2", "Tg/yr")
        err = try units(u); nothing catch e; e end
        @test err isa ArgumentError && occursin("'F' in f.nc", err.msg)
    end
    @test_throws ArgumentError ICIO._require_reference_time(nothing)
end
