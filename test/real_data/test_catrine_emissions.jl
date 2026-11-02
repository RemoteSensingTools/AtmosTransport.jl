using Test, AtmosTransport
using AtmosTransport.Models.InitialConditionIO: _load_file_surface_flux_field
const GRIDFED_FILE = expanduser("~/data/AtmosTransport/catrine/Emissions/gridfed/GCP-GridFEDv2024.0_2021.short.nc")
if !isfile(GRIDFED_FILE)
    @testset "GridFED pinned monthly total (data unavailable)" begin
        @test_skip true
    end
else
    @testset "gridfed_fossil_co2 — kgCO2/month/m² → calendar-month /s" begin
        f = _load_file_surface_flux_field(
            Dict("kind" => "gridfed_fossil_co2", "time_index" => 12), Float32; species = :co2)
        # December 2021 is converted with 31 calendar days, not the
        # old average-month convention. This matches GEOS-Chem's
        # embedded Emis_FossilCO2_Total for the CATRINE run.
        @test f.native_total_mass_rate ≈ 1.22939955e6 rtol = 2e-6
    end
end
