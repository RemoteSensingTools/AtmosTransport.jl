using Test, AtmosTransport, NCDatasets, Dates, Logging
include(joinpath(@__DIR__, "..", "fixtures", "cs_multifile.jl"))
using .CSDriverHandoffFixtures

const ORIGIN = DateTime(2021, 12, 2)
const G = 9.80665

# Two-window 2×2 lat-lon binary with no fluxes: a uniform tracer stays uniform.
# The single layer holds the dry air between p_top = 10 Pa and ps = 95 kPa.
function synthetic_ll_binary(path; nwindows = 2)
    mesh = LatLonMesh(; FT = Float64, Nx = 2, Ny = 2)
    grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[10.0, 0.0], Float64[0.0, 1.0]), CPU(); FT = Float64)
    m = [(95000.0 - 10.0) * cell_area(mesh, i, j) / G for i in 1:2, j in 1:2, k in 1:1]
    window = (m = m, am = zeros(Float64, 3, 2, 1), bm = zeros(Float64, 2, 3, 1),
              cm = zeros(Float64, 2, 2, 2), ps = fill(95000.0, 2, 2))
    write_transport_binary(path, grid, fill(window, nwindows); FT = Float64, dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0, steps_per_window = 1, mass_basis = :dry,
                           source_flux_sampling = :window_start_endpoint, flux_sampling = :window_constant)
    return path, mesh
end

function quiet(f)
    with_logger(NullLogger()) do
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                f()
            end
        end
    end
end

@testset "lat-lon run samples soundings and sites without gridded output" begin
    mktempdir() do dir
        bin, mesh = synthetic_ll_binary(joinpath(dir, "transport_20211202.bin"))
        csv = joinpath(dir, "soundings.csv")
        write(csv, "id,time,lat,lon\nfirst,2021-12-02T00:30:00,10,-100\nsecond,2021-12-02T01:30:00,-40,120\n")
        sites = joinpath(dir, "sites.csv")
        write(sites, "id,lat,lon,intake_height\nmlo,19.5,-155.6,40\n")
        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [bin]),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("tracer_name" => "co2"),
            "init" => Dict("kind" => "uniform", "background" => 400e-6),
            "output" => Dict{String, Any}("enabled" => false,
                "observations" => Dict{String, Any}(
                    "path" => joinpath(dir, "obs.nc"),
                    "start_time" => "2021-12-02T00:00:00",
                    "sources" => Any[Dict{String, Any}("kind" => "table", "mode" => "soundings", "path" => csv),
                                     Dict{String, Any}("kind" => "table", "mode" => "sites", "path" => sites)])))
        ok, errors = validate_config(cfg)
        @test ok
        quiet(() -> run_driven_simulation(cfg))
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test ds.attrib["completed_soundings"] == 2
            @test ds.attrib["n_after_end"] == 0
            @test ds["id"][:] == ["first", "second"]
            @test ds["time"][:] == [ORIGIN + Minute(30), ORIGIN + Minute(90)]
            @test ds["sample_time_prev"][:] == [ORIGIN, ORIGIN + Hour(1)]
            @test ds["sample_time_next"][:] == [ORIGIN + Hour(1), ORIGIN + Hour(2)]
            @test ds["interp_weight"][:] == Float32[0.5, 0.5]
            # Uniform tracer: exact in the model, Float32 on disk.
            @test all(isapprox.(ds["co2"][:, :], 400e-6; rtol = 1e-6))
            @test all(isapprox.(ds["co2_column_mean"][:], 400e-6; rtol = 1e-6))
            p_half = ds["p_half_dry"][:, :]
            per_area = ds["air_mass_per_area_dry"][:, :]
            @test all(p_half[1, :] .== 10)                              # A_ifc[1]
            @test all(isapprox.(p_half[2, :], 10 .+ G .* per_area[1, :]; rtol = 1e-6))
            @test all(isapprox.(ds["ps_dry"][:], 95000; rtol = 1e-6))   # reconstructed from air mass
            @test ds["cell_i"][:] == Int32[1, 2] && ds["cell_j"][:] == Int32[2, 1]
        end
        NCDataset(joinpath(dir, "obs_sites.nc"), "r") do ds
            @test ds.attrib["completed_times"] == 3                    # t = 0, 1 h, 2 h
            @test ds["time"][:] == [ORIGIN, ORIGIN + Hour(1), ORIGIN + Hour(2)]
            @test all(isapprox.(ds["co2_intake"][:, :], 400e-6; rtol = 1e-6))
            @test all(ds["intake_level"][:, :] .== 1)
            @test all(ds["height_method"][:, :] .== 0)
        end
    end
end

@testset "daily files carry soundings across the binary handoff" begin
    mktempdir() do dir
        bin1, _ = synthetic_ll_binary(joinpath(dir, "transport_20211202.bin"))
        bin2, _ = synthetic_ll_binary(joinpath(dir, "transport_20211203.bin"))
        csv = joinpath(dir, "soundings.csv")
        # 2.5 h is bracketed by the last window end of day 1 (2 h) and the first of day 2 (3 h).
        write(csv, "id,time,lat,lon\nday1,2021-12-02T00:30:00,10,-100\nstraddle,2021-12-02T02:30:00,10,-100\nday2,2021-12-02T03:30:00,10,-100\n")
        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [bin1, bin2]),
            "numerics" => Dict("float_type" => "Float32"),
            "run" => Dict("tracer_name" => "co2"),
            "init" => Dict("kind" => "uniform", "background" => 400e-6),
            "output" => Dict{String, Any}("path" => joinpath(dir, "snap_{date}.nc"), "hours" => [0.0, 2.0, 4.0],
                "split" => "daily",
                "observations" => Dict{String, Any}(
                    "path" => joinpath(dir, "obs_{date}.nc"),
                    "start_time" => "2021-12-02T00:00:00",
                    "sources" => Any[Dict{String, Any}("kind" => "table", "mode" => "soundings", "path" => csv)])))
        quiet(() -> run_driven_simulation(cfg))
        NCDataset(joinpath(dir, "obs_20211202_soundings.nc"), "r") do ds
            @test ds["id"][:] == ["day1"]
            @test ds.attrib["completed_soundings"] == 1
        end
        NCDataset(joinpath(dir, "obs_20211203_soundings.nc"), "r") do ds
            @test ds["id"][:] == ["straddle", "day2"]
            @test ds["sample_time_prev"][:] == [ORIGIN + Hour(2), ORIGIN + Hour(3)]
            @test ds["sample_time_next"][:] == [ORIGIN + Hour(3), ORIGIN + Hour(4)]
            @test all(ds["interp_flag"][:] .== 0)
            @test all(isapprox.(ds["co2"][:, :], 400e-6; rtol = 1e-6))   # Float32 run
            @test ds.attrib["n_after_end"] == 0
        end
        # The gridded daily snapshots are still written alongside.
        @test isfile(joinpath(dir, "snap_20211202.nc")) && isfile(joinpath(dir, "snap_20211203.nc"))
    end
end

@testset "sampling leaves the gridded output unchanged" begin
    mktempdir() do dir
        bin, _ = synthetic_ll_binary(joinpath(dir, "transport_20211202.bin"))
        csv = joinpath(dir, "soundings.csv")
        write(csv, "id,time,lat,lon\nfirst,2021-12-02T00:30:00,10,-100\n")
        function run_with(out, observations)
            cfg = Dict{String, Any}(
                "input" => Dict("binary_paths" => [bin]),
                "numerics" => Dict("float_type" => "Float64"),
                "run" => Dict("tracer_name" => "co2"),
                "init" => Dict("kind" => "gaussian_blob", "background" => 400e-6, "amplitude" => 50e-6,
                               "lon0_deg" => 0.0, "lat0_deg" => 0.0, "sigma_lon_deg" => 60.0, "sigma_lat_deg" => 30.0),
                "output" => Dict{String, Any}("path" => out, "hours" => [0.0, 1.0, 2.0], "split" => "single"))
            observations === nothing || (cfg["output"]["observations"] = observations)
            quiet(() -> run_driven_simulation(cfg))
            return NCDataset(out, "r") do ds
                (Array(ds["co2"][:, :, :, :]), Array(ds["co2_column_mean"][:, :, :]))
            end
        end
        reference = run_with(joinpath(dir, "reference.nc"), nothing)
        sampled = run_with(joinpath(dir, "sampled.nc"), Dict{String, Any}(
            "path" => joinpath(dir, "obs.nc"), "start_time" => "2021-12-02T00:00:00",
            "sources" => Any[Dict{String, Any}("kind" => "table", "mode" => "soundings", "path" => csv)]))
        @test reference[1] == sampled[1]
        @test reference[2] == sampled[2]
        @test isfile(joinpath(dir, "obs_soundings.nc"))
    end
end

@testset "cubed-sphere sampling is identical across a file handoff" begin
    mktempdir() do dir
        combined, first, second = joinpath.(dir, ["combined.bin", "first.bin", "second.bin"])
        cs_handoff_fixture(combined, [1.0, 1.0, 8.0, 8.0])
        cs_handoff_fixture(first, [1.0, 1.0])
        cs_handoff_fixture(second, [8.0, 8.0])
        csv = joinpath(dir, "soundings.csv")
        write(csv, "id,time,lat,lon\na,2021-12-02T00:30:00,10,-100\nb,2021-12-02T02:30:00,-40,120\nc,2021-12-02T03:15:00,60,0\n")
        sites = joinpath(dir, "sites.csv")
        write(sites, "id,lat,lon,intake_height\np1,0,0,10\np2,-80,170,500\n")
        function run_paths(paths, obs_root)
            cfg = Dict{String, Any}(
                "input" => Dict("binary_paths" => paths),
                "numerics" => Dict("float_type" => "Float64"),
                "advection" => Dict("scheme" => "ppm"),
                "diffusion" => Dict("kind" => "constant", "value" => 10.0),
                "convection" => Dict("kind" => "cmfmc"),
                "tracers" => Dict("co2" => Dict("init" => Dict("kind" => "pressure_layer", "lowest_layer" => true,
                                                              "total_molecules" => 1e35))),
                "output" => Dict{String, Any}("enabled" => false,
                    "observations" => Dict{String, Any}(
                        "path" => obs_root, "start_time" => "2021-12-02T00:00:00",
                        "sources" => Any[Dict{String, Any}("kind" => "table", "mode" => "soundings", "path" => csv),
                                         Dict{String, Any}("kind" => "table", "mode" => "sites", "path" => sites)])))
            quiet(() -> run_driven_simulation(cfg))
        end
        run_paths([combined], joinpath(dir, "cont.nc"))
        run_paths([first, second], joinpath(dir, "split.nc"))
        NCDataset(joinpath(dir, "cont_soundings.nc"), "r") do c
            NCDataset(joinpath(dir, "split_soundings.nc"), "r") do s
                @test c["id"][:] == s["id"][:] == ["a", "b", "c"]
                @test c["cell_panel"][:] == s["cell_panel"][:]
                @test c["air_mass_per_area_dry"][:, :] == s["air_mass_per_area_dry"][:, :]
                @test c["co2"][:, :] ≈ s["co2"][:, :] rtol = 1e-10
                @test c["co2_column_mean"][:] ≈ s["co2_column_mean"][:] rtol = 1e-10
                @test c["sample_time_prev"][2] == ORIGIN + Hour(2) && c["sample_time_next"][2] == ORIGIN + Hour(3)
                @test all(c["interp_flag"][:] .== 0)
            end
        end
        NCDataset(joinpath(dir, "cont_sites.nc"), "r") do c
            NCDataset(joinpath(dir, "split_sites.nc"), "r") do s
                @test c.attrib["completed_times"] == s.attrib["completed_times"] == 5
                @test c["co2_intake"][:, :] ≈ s["co2_intake"][:, :] rtol = 1e-10
                @test c["intake_level"][:, :] == s["intake_level"][:, :]
                # Toy fixture: 5 equal-sigma layers (~20% of the column each, lowest ~1.8 km),
                # so both intakes sit in layer 5. Real-grid placement is tested in
                # test_observation_gather.jl on GEOS L72 (surface layer ~124 m).
                @test all(c["intake_level"][:, :] .== 5)
                @test all(c["intake_layer_top_agl"][2, :] .> 500)
            end
        end
    end
end

function ll_cfg(dir, bins; observations, start_time = "2021-12-02T00:00:00", run = Dict{String, Any}(),
                snapshots = false)
    obs = Dict{String, Any}("path" => joinpath(dir, "obs.nc"), "sources" => observations)
    start_time === nothing || (obs["start_time"] = start_time)
    output = snapshots ? Dict{String, Any}("path" => joinpath(dir, "snap.nc"), "hours" => [0.0]) :
                         Dict{String, Any}("enabled" => false)
    output["observations"] = obs
    return Dict{String, Any}("input" => Dict("binary_paths" => bins),
                             "numerics" => Dict("float_type" => "Float64"),
                             "run" => merge(Dict{String, Any}("tracer_name" => "co2"), run),
                             "init" => Dict("kind" => "uniform", "background" => 400e-6),
                             "output" => output)
end
table(path; mode = "soundings") = Any[Dict{String, Any}("kind" => "table", "mode" => mode, "path" => path)]

@testset "start_window > 1 shifts observation times by the skipped windows" begin
    mktempdir() do dir
        bin, _ = synthetic_ll_binary(joinpath(dir, "transport_20211202.bin"); nwindows = 4)
        csv = joinpath(dir, "s.csv")
        write(csv, "id,time,lat,lon\nearly,2021-12-02T00:30:00,10,-100\nmid,2021-12-02T01:30:00,10,-100\nlate,2021-12-02T03:30:00,10,-100\n")
        cfg = ll_cfg(dir, [bin]; observations = table(csv), run = Dict{String, Any}("start_window" => 2, "stop_window" => 3))
        quiet(() -> run_driven_simulation(cfg))
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test ds["id"][:] == ["mid"]
            @test ds["sample_time_prev"][1] == ORIGIN + Hour(1)
            @test ds["sample_time_next"][1] == ORIGIN + Hour(2)
            @test ds.attrib["n_before_start"] == 0          # "early" is outside the transported span
            @test ds.attrib["n_after_end"] == 0              # so is "late"
        end
    end
end

@testset "the sounding window is the transported span, not calendar days" begin
    mktempdir() do dir
        bin, _ = synthetic_ll_binary(joinpath(dir, "transport_20211202.bin"); nwindows = 24)
        csv = joinpath(dir, "s.csv")
        write(csv, "id,time,lat,lon\na,2021-12-02T23:30:00,10,-100\nb,2021-12-03T00:30:00,10,-100\n")
        quiet(() -> run_driven_simulation(ll_cfg(dir, [bin]; observations = table(csv))))
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test ds["id"][:] == ["a"]
        end
        # An origin on another day than the first binary's label is refused.
        bad = ll_cfg(dir, [bin]; observations = table(csv), start_time = "2021-12-03T00:00:00")
        @test_throws ArgumentError quiet(() -> run_driven_simulation(bad))
        # start_date and start_time must agree when both are given.
        both = ll_cfg(dir, [bin]; observations = table(csv), start_time = "2021-12-02T06:00:00")
        both["input"] = Dict{String, Any}("folder" => dir, "start_date" => "2021-12-02", "end_date" => "2021-12-02")
        ok, errors = validate_config(both)
        @test !ok && any(e -> occursin("disagrees with", e), errors)
    end
end

@testset "sampling without snapshots leaves the transported state unchanged" begin
    mktempdir() do dir
        bin, _ = synthetic_ll_binary(joinpath(dir, "transport_20211202.bin"))
        csv = joinpath(dir, "s.csv")
        write(csv, "id,time,lat,lon\na,2021-12-02T00:30:00,10,-100\n")
        plain = ll_cfg(dir, [bin]; observations = table(csv))
        plain["output"] = Dict{String, Any}("enabled" => false)
        reference = quiet(() -> run_driven_simulation(plain))
        sampled = quiet(() -> run_driven_simulation(ll_cfg(dir, [bin]; observations = table(csv))))
        @test sampled.state.air_mass == reference.state.air_mass
        @test sampled.state.tracers_raw == reference.state.tracers_raw
    end
end
