#!/usr/bin/env julia
#
# GriddedIO flux units: one grammar for every units string of a surface-flux
# input (design 2.4; P0 facts memo Section 3), conversion factors built from
# `Parameters` constants, and bit-identity with the runtime's unit handling it
# replaces (`_normalize_units_string` + `_per_second_flux_scale`, the GridFED
# month-length branch with `_days_in_month`, the EDGAR tonnes-per-year branch).

using Test
using Dates

using AtmosTransport
using AtmosTransport.GriddedIO: parse_flux_units, FluxUnits, TargetSpecies, Carbon, Species,
                                PerSquareMetre, PerCell, PerSecond, PerDay, PerMonth, PerYear,
                                kg_per_unit, period_seconds, mass_rate_factor,
                                canonical_units_string, MonthKey, TimeInterval, interval
using AtmosTransport.Parameters: SPECIES_MOLAR_MASS, CARBON_MOLAR_MASS, DAYS_PER_JULIAN_YEAR

const K = SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS        # M(CO2)/M(C)

# ---- Frozen runtime unit handling (src/Models/initial_conditions/surface_flux.jl) ----

const FROZEN_KG_PER_M2_S = r"^kg(\(?[a-z][a-z0-9]*\)?)?(/m2/s|m-2s-1|/m2s|/s/m2|s-1m-2)$"
frozen_normalize(units) = lowercase(replace(strip(String(units)), " " => "", "^" => "", "²" => "2"))
function frozen_per_second_flux_scale(units_norm)
    m = match(FROZEN_KG_PER_M2_S, units_norm)
    m === nothing && throw(ArgumentError("unsupported surface-flux units '$(units_norm)'"))
    species = m.captures[1] === nothing ? "" : strip(m.captures[1], ['(', ')'])
    return species == "c" ? SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS : 1.0
end
frozen_is_leap_year(year) = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
function frozen_days_in_month(year, month)
    month == 2 && frozen_is_leap_year(year) && return 29
    return (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month]
end
frozen_gridfed_month_seconds(year, month) = Float64(frozen_days_in_month(year, month)) * 86400.0
const FROZEN_EDGAR_SECONDS_PER_YEAR = DAYS_PER_JULIAN_YEAR * 86400

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

@testset "bit-identical to the runtime unit handling" begin
    # Per-second units the runtime accepted: same species factor, bit for bit.
    for s in ("kg m-2 s-1", "kg/m2/s", "kg/m2s", "kg/s/m2", "kg s-1 m-2", "kg/m²/s", "kg m^-2 s^-1",
              "kgCO2/m2/s", "kg(CO2) m^-2 s^-1", "kgC m-2 s-1", "kg(C)/m2/s", "kgc/m2/s", "KGC M-2 S-1")
        old = frozen_per_second_flux_scale(frozen_normalize(s))
        @test mass_rate_factor(parse_flux_units(s), :co2) === old
    end
    for s in ("kgsf6 m-2 s-1", "kg m-2 s-1")
        @test mass_rate_factor(parse_flux_units(s), :sf6) === frozen_per_second_flux_scale(frozen_normalize(s))
    end
    # GridFED kgCO2/month/m2: the runtime divides by the month's seconds.
    u = parse_flux_units("kgCO2/month/m2")
    for year in 1999:2025, month in 1:12
        @test period_seconds(u, interval(MonthKey(year, month))) === frozen_gridfed_month_seconds(year, month)
        @test kg_per_unit(u, :co2) === 1.0
        @test Dates.daysinmonth(year, month) == frozen_days_in_month(year, month)
    end
    for year in 1600:2400
        @test Dates.isleapyear(year) == frozen_is_leap_year(year)
    end
    # EDGAR Tonnes: 1000 kg per tonne over the runtime's Julian-year seconds.
    u = parse_flux_units("Tonnes")
    @test period_seconds(u) === FROZEN_EDGAR_SECONDS_PER_YEAR
    @test kg_per_unit(u, :sf6) === 1000.0
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
