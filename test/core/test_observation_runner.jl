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
                           surface_pressure_basis = TotalSurfacePressure(),
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
                    "pressure_basis" => "dry",   # the synthetic binary carries no humidity
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
                    "pressure_basis" => "dry",   # the synthetic binary carries no humidity
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
            "path" => joinpath(dir, "obs.nc"), "start_time" => "2021-12-02T00:00:00", "pressure_basis" => "dry",
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
                        "path" => obs_root, "start_time" => "2021-12-02T00:00:00", "pressure_basis" => "dry",
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

# `pressure_basis = "dry"` for binaries without humidity (`synthetic_ll_binary`).
function ll_cfg(dir, bins; observations, start_time = "2021-12-02T00:00:00", run = Dict{String, Any}(),
                snapshots = false, pressure_basis = nothing)
    obs = Dict{String, Any}("path" => joinpath(dir, "obs.nc"), "sources" => observations)
    start_time === nothing || (obs["start_time"] = start_time)
    pressure_basis === nothing || (obs["pressure_basis"] = pressure_basis)
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
        cfg = ll_cfg(dir, [bin]; observations = table(csv), run = Dict{String, Any}("start_window" => 2, "stop_window" => 3),
                     pressure_basis = "dry")
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
        quiet(() -> run_driven_simulation(ll_cfg(dir, [bin]; observations = table(csv), pressure_basis = "dry")))
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test ds["id"][:] == ["a"]
        end
        # An origin on another day than the first binary's label is refused.
        bad = ll_cfg(dir, [bin]; observations = table(csv), start_time = "2021-12-03T00:00:00", pressure_basis = "dry")
        @test_throws ArgumentError quiet(() -> run_driven_simulation(bad))
        # start_date and start_time must agree when both are given.
        both = ll_cfg(dir, [bin]; observations = table(csv), start_time = "2021-12-02T06:00:00", pressure_basis = "dry")
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
        plain = ll_cfg(dir, [bin]; observations = table(csv), pressure_basis = "dry")
        plain["output"] = Dict{String, Any}("enabled" => false)
        reference = quiet(() -> run_driven_simulation(plain))
        sampled = quiet(() -> run_driven_simulation(ll_cfg(dir, [bin]; observations = table(csv), pressure_basis = "dry")))
        @test sampled.state.air_mass == reference.state.air_mass
        @test sampled.state.tracers_raw == reference.state.tracers_raw
    end
end

# Two-window 2×2 lat-lon binary with three layers whose total interfaces are
# 10, 28500, 66500 and 95000 Pa under the humidity qa: dry air mass
# (1 − qa) Δp A / g, unchanged in time (no fluxes). With `humid`, window 1
# carries qv_start = qa, qv_end = 0 and window 2 qv_start = 0, qv_end = 2qa.
const QA = [0.001, 0.01, 0.02]
function humid_ll_binary(path; humid = true)
    mesh = LatLonMesh(; FT = Float64, Nx = 2, Ny = 2)
    A, B = [10.0, 0.0, 0.0, 0.0], [0.0, 0.3, 0.7, 1.0]
    grid = AtmosGrid(mesh, HybridSigmaPressure(A, B), CPU(); FT = Float64)
    dp = diff(A .+ B .* 95000.0)
    m = [(1 - QA[k]) * dp[k] * cell_area(mesh, i, j) / G for i in 1:2, j in 1:2, k in 1:3]
    q(v) = repeat(reshape(v, 1, 1, 3), 2, 2, 1)
    base = (m = m, am = zeros(3, 2, 3), bm = zeros(2, 3, 3), cm = zeros(2, 2, 4), ps = fill(95000.0, 2, 2))
    windows = humid ? [merge(base, (qv_start = q(QA), qv_end = q(zeros(3)))),
                       merge(base, (qv_start = q(zeros(3)), qv_end = q(2 .* QA)))] : [base, base]
    write_transport_binary(path, grid, windows; FT = Float64, dt_met_seconds = 3600.0, half_dt_seconds = 1800.0,
                           steps_per_window = 1, mass_basis = :dry, surface_pressure_basis = TotalSurfacePressure(),
                           source_flux_sampling = :window_start_endpoint, flux_sampling = :window_constant)
    return path
end

function obspack_pressures(path, times, pressures_hpa)
    NCDataset(path, "c") do ds
        defDim(ds, "obs", length(times))
        defVar(ds, "time", Float64, ("obs",); attrib = Dict("units" => "seconds since 1970-01-01T00:00:00Z"))[:] =
            datetime2unix.(times)
        defVar(ds, "latitude", Float32, ("obs",))[:] = fill(10f0, length(times))
        defVar(ds, "longitude", Float32, ("obs",))[:] = fill(-100f0, length(times))
        defVar(ds, "pressure", Float32, ("obs",); attrib = Dict("units" => "hPa"))[:] = Float32.(pressures_hpa)
        defVar(ds, "obspack_id", String, ("obs",))[:] = ["p~$(k)" for k in eachindex(times)]
    end
    return path
end

@testset "pressure placement takes the binary's humidity at each window end (issue #20)" begin
    mktempdir() do dir
        # Bottom interface of layer 2: 66500 Pa with qa, 66091.5 Pa dry (q = 0), 66916.3 Pa with 2qa.
        # 664 hPa at t = 0 is in layer 2 only with qv_start of window 1 (qa); 662 hPa at 1 h is in layer 3
        # with qv_end of window 1 (0); 667 hPa at 2 h is in layer 2 only with qv_end of window 2 (2qa).
        pack = obspack_pressures(joinpath(dir, "co2_x_aircraft-insitu_1.nc"),
                                 [ORIGIN, ORIGIN + Hour(1), ORIGIN + Hour(2)], [664.0, 662.0, 667.0])
        sources = Any[Dict{String, Any}("kind" => "obspack", "mode" => "soundings", "path" => pack)]
        levels(path) = NCDataset(ds -> Int.(ds["intake_level"][:]), path, "r")
        humid = humid_ll_binary(joinpath(dir, "transport_20211202.bin"))
        quiet(() -> run_driven_simulation(ll_cfg(dir, [humid]; observations = sources)))
        @test levels(joinpath(dir, "obs_soundings.nc")) == [2, 3, 2]
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test ds.attrib["pressure_basis"] == "total"
            @test all(ds["vertical_placement"][:] .== 2)
        end
        # A binary without humidity needs the explicit dry opt-in, which compares with dry interfaces.
        dry_dir = mkpath(joinpath(dir, "dry"))
        dry = humid_ll_binary(joinpath(dry_dir, "transport_20211202.bin"); humid = false)
        err = try
            quiet(() -> run_driven_simulation(ll_cfg(dry_dir, [dry]; observations = sources)))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("pressure_basis = \"dry\"", sprint(showerror, err))
        cfg = ll_cfg(dry_dir, [dry]; observations = sources)
        cfg["output"]["observations"]["pressure_basis"] = "dry"
        quiet(() -> run_driven_simulation(cfg))
        @test levels(joinpath(dry_dir, "obs_soundings.nc")) == [3, 3, 3]
    end
end

@testset "layer heights take the binary's humidity at each window end (OF-117)" begin
    mktempdir() do dir
        # A 10 m site samples the surface layer (66500-95000 Pa under qa) at 0, 1 and 2 h. Its top is
        # R_d·280·(1 + c·q)/g·ln(p_s/p_2) on the total pressures of that window end's humidity (qa,
        # 0, 2qa; the lat-lon sampler's constant 280 K) with the virtual temperature; the dry rule
        # (before OF-117) takes the dry interfaces and T at every end.
        R, c = AtmosTransport.Parameters.R_DRY_AIR, AtmosTransport.Parameters.VIRTUAL_TEMPERATURE_FACTOR
        dp = diff([10.0, 0.3 * 95000, 0.7 * 95000, 95000.0])
        dry_dp = (1 .- QA) .* dp
        top(q) = begin
            p = 10.0 .+ vcat(0.0, cumsum(dry_dp ./ (1 .- q)))
            R * 280 * (1 + c * q[3]) / G * log(p[4] / p[3])
        end
        @test top(QA) ≈ R * 280 * (1 + c * QA[3]) / G * log(95000 / 66500) rtol = 1e-14
        sites = joinpath(dir, "sites.csv")
        write(sites, "id,lat,lon,intake_height\nlow,10,-100,10\n")
        humid = humid_ll_binary(joinpath(dir, "transport_20211202.bin"))
        tops(path) = NCDataset(ds -> Float64.(ds["intake_layer_top_agl"][1, :]), path, "r")
        quiet(() -> run_driven_simulation(ll_cfg(dir, [humid]; observations = table(sites; mode = "sites"))))
        @test tops(joinpath(dir, "obs_sites.nc")) ≈ [top(QA), top(zeros(3)), top(2 .* QA)] rtol = 1e-6
        dry_dir = mkpath(joinpath(dir, "dry"))
        quiet(() -> run_driven_simulation(ll_cfg(dry_dir, [humid]; observations = table(sites; mode = "sites"),
                                                 pressure_basis = "dry")))
        @test tops(joinpath(dry_dir, "obs_sites.nc")) ≈ fill(top(zeros(3)), 3) rtol = 1e-6
        @test top(2 .* QA) / top(zeros(3)) - 1 > 0.04          # 2qa: the dry rule is 4 % low here
    end
end

@testset "runner humidity: sampled fields and per-binary capability (issue #20)" begin
    R = AtmosTransport.Models.DrivenRunner
    qs, qe, vq = fill(0.01, 2, 2, 3), fill(0.02, 2, 2, 3), ntuple(_ -> fill(0.03, 4, 4, 3), 6)
    structured = (qv_start = qs, qv_end = qe, vdiff = nothing)
    @test R._window_humidity(structured, true) === qs && R._window_humidity(structured, false) === qe
    cubed = (qv_start = nothing, qv_end = nothing, vdiff = (t = vq, qv = vq))
    @test R._window_humidity(cubed, true) === vq && R._window_humidity(cubed, false) === vq
    @test R._window_humidity((qv_start = nothing, qv_end = nothing, vdiff = nothing), false) === nothing
    caps(humidity, grid, sections = Symbol[]) = (humidity = humidity, grid_type = grid, payload_sections = sections)
    @test R._binary_observation_humidity(caps(true, :latlon))
    @test R._binary_observation_humidity(caps(false, :cubed_sphere, [:m, :vdiff_t, :vdiff_qv]))
    @test !R._binary_observation_humidity(caps(false, :latlon, [:vdiff_qv]))   # lat-lon windows load no VDIFF
    @test !R._binary_observation_humidity(caps(false, :cubed_sphere, [:m]))
    @test R._run_observation_humidity([true, true], ["a", "b"], false)
    @test !R._run_observation_humidity([true, false], ["a", "b"], true)
    @test R._run_observation_humidity(nothing, ["a"], true)

    # A later binary without humidity stops the run before transport, not midway.
    mktempdir() do dir
        pack = obspack_pressures(joinpath(dir, "co2_x_aircraft-insitu_1.nc"), [ORIGIN + Minute(30)], [664.0])
        sources = Any[Dict{String, Any}("kind" => "obspack", "mode" => "soundings", "path" => pack)]
        day1 = humid_ll_binary(joinpath(dir, "transport_20211202.bin"))
        day2 = humid_ll_binary(joinpath(dir, "transport_20211203.bin"); humid = false)
        err = try
            quiet(() -> run_driven_simulation(ll_cfg(dir, [day1, day2]; observations = sources)))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("do not all carry humidity", sprint(showerror, err))
        @test !isfile(joinpath(dir, "obs_soundings.nc"))
    end
end

# Cubed-sphere binary (Nc = 3, two layers, no fluxes) with a GCHP VDIFF payload
# whose layer-1 humidity is `q1[w]` in window w (layer 2 dry). The dry air is
# fixed, (1 − 0.01) Δp A / g with Δp = 49900 and 50000 Pa, so the total
# interface between the layers is 100 + 49401 / (1 − q1).
function cs_vdiff_binary(path, q1)
    FT, Nc, Nz = Float64, 3, 2
    vc = HybridSigmaPressure(FT[100, 0, 0], FT[0, 0.5, 1])
    area = CubedSphereMesh(; FT, Nc).cell_areas          # the mesh the runner rebuilds from the header
    m = ntuple(_ -> cat((1 - 0.01) * 49900 .* area ./ G, 50000 .* area ./ G; dims = 3), 6)
    p3(v) = ntuple(_ -> fill(FT(v), Nc, Nc, Nz), 6)
    writer = AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(path, Nc, 6, Nz, length(q1), vc;
        planet_radius = AtmosTransport.Parameters.EARTH_RADIUS, FT, dt_met_seconds = 3600.0, steps_per_window = 1,
        mass_basis = :dry, surface_pressure_basis = DrySurfacePressure(), include_gchp_vdiff = true)
    try
        for q in q1
            qv = ntuple(_ -> cat(fill(q, Nc, Nc), zeros(Nc, Nc); dims = 3), 6)
            window = (m, am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), 6), bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), 6),
                      cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6),
                      ps = ntuple(_ -> fill(FT(100 + 0.99 * 49900 + 50000), Nc, Nc), 6),
                      vdiff = (u = p3(0), v = p3(0), t = p3(280), qv))
            AtmosTransport.MetDrivers.write_streaming_cs_window!(writer, window, Nc, 6)
        end
    finally
        AtmosTransport.MetDrivers.close_streaming_transport_binary!(writer)
    end
    return path
end

@testset "cubed sphere: a window end takes the next window's VDIFF humidity (issue #20)" begin
    mktempdir() do dir
        # q1 = 0.01, 0.03 (binary a), 0.05, 0.07 (binary b): the total interface is 50000, 51028.9,
        # 52101.6 and 53219.4 Pa. VDIFF qv is a window-start value, so the humidity at 1 h, 2 h and 3 h
        # is that of the next window: 505, 516 and 527 hPa are in layer 1 with it and in layer 2 with the
        # ended window's (the lag). The final window end (4 h) takes the ended window's (0.07).
        a = cs_vdiff_binary(joinpath(dir, "a.bin"), [0.01, 0.03])
        b = cs_vdiff_binary(joinpath(dir, "b.bin"), [0.05, 0.07])
        pack = obspack_pressures(joinpath(dir, "co2_cs_aircraft-insitu_1.nc"),
                                 [ORIGIN + Hour(h) for h in 1:4], [505.0, 516.0, 527.0, 527.0])
        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [a, b]),
            "numerics" => Dict("float_type" => "Float64"),
            "advection" => Dict("scheme" => "upwind"),
            "tracers" => Dict("co2" => Dict("init" => Dict("kind" => "uniform", "background" => 400e-6))),
            "output" => Dict{String, Any}("enabled" => false,
                "observations" => Dict{String, Any}("path" => joinpath(dir, "obs.nc"),
                    "start_time" => "2021-12-02T00:00:00",
                    "sources" => Any[Dict{String, Any}("kind" => "obspack", "mode" => "soundings", "path" => pack)])))
        quiet(() -> run_driven_simulation(cfg))
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test all(ds["vertical_placement"][:] .== 2)
            @test Int.(ds["intake_level"][:]) == [1, 1, 1, 1]       # [2, 2, 2, 1] with the ended window's qv
        end
    end
end
