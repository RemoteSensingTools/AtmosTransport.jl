# One end-to-end smoke test of the cs_native comparison CLI (T08).
using Test
using Dates
using JSON3
using AtmosTransport
using AtmosTransport.Grids: CubedSphereMesh, GEOSNativePanelConvention
using AtmosTransport.GriddedIO: TimeInterval, open_cs_native_writer, write_slices!

const CMP = Module()
Base.include(CMP, joinpath(@__DIR__, "..", "..", "scripts", "validation", "compare_cs_flux_files.jl"))

@testset "compare_cs_flux_files CLI round trip" begin
    mktempdir() do dir
        mesh = CubedSphereMesh(; Nc = 4, FT = Float64, Hp = 0, convention = GEOSNativePanelConvention())
        intervals = [TimeInterval(DateTime(2021, 12, 30) + Day(k),
                                  DateTime(2021, 12, 31) + Day(k)) for k in 0:3]
        reference = [Float32(1e-9 * (i + 10j + 100p + 1000t))
                     for i in 1:4, j in 1:4, p in 1:6, t in 1:4]
        candidate = reference .* 1.001f0
        ref, cand = joinpath(dir, "ref.nc"), joinpath(dir, "cand.nc")
        for (path, data) in ((ref, reference), (cand, candidate))
            open_cs_native_writer(path, mesh, intervals, ["FLUX"]; species = :co2, chunk_time = 1) do w
                write_slices!(w, "FLUX", 1:4, data)
            end
        end
        json = joinpath(dir, "report.json")
        summary = joinpath(dir, "summary.txt")
        args = ["--reference", ref, "--candidate", cand, "--nc", "4",
                "--json", json, "--summary", summary, "--tol", "cell_max=0.01"]
        @test redirect_stdout(devnull) do
            Base.invokelatest(getproperty(CMP, :main), args)
        end == 0
        report = JSON3.read(read(json, String))
        @test report.pass === true
        variable = only(report.variables)
        ref_totals = [sum(Float64(reference[i, j, p, t]) * mesh.cell_areas[i, j]
                          for i in 1:4, j in 1:4, p in 1:6) for t in 1:4]
        cand_totals = [sum(Float64(candidate[i, j, p, t]) * mesh.cell_areas[i, j]
                           for i in 1:4, j in 1:4, p in 1:6) for t in 1:4]
        @test variable.slice_totals.reference ≈ ref_totals rtol = 1e-13
        @test variable.slice_totals.candidate ≈ cand_totals rtol = 1e-13
        @test [m.month for m in variable.month_totals.months] == ["2021-12", "2022-01"]
        @test startswith(read(summary, String), "compare_cs_flux_files: PASS")
    end
end
