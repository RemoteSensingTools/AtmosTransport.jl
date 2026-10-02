using Test, AtmosTransport, NCDatasets, Dates
using AtmosTransport.Output: observation_output_spec, build_observation_sampler, begin_observation_day!,
                             observe_window_boundary!, finish_observations!, ObservationSampler,
                             NoObservationOutput, NoObservationSampler, SoundingNetCDFStream,
                             SiteNetCDFStream, SoundingBatch, SiteRecord, append_soundings!,
                             append_site_record!, CellLocation, GatherPlan, samples_observations
const O = AtmosTransport.Output

const ORIGIN = DateTime(2021, 12, 2)
unix(dt) = datetime2unix(dt)
const G = 9.80665

# A 4×3 lat-lon column model with 3 levels, p_top = 100 Pa, uniform co2.
function fake_model(; q = 400e-6, FT = Float64)
    mesh = LatLonMesh(; FT = FT, Nx = 4, Ny = 3)
    A = FT[100.0, 0.0, 0.0, 0.0]
    B = FT[0.0, 0.1, 0.5, 1.0]
    grid = AtmosGrid(mesh, HybridSigmaPressure(A, B), CPU(); FT = FT)
    area = [cell_area(mesh, i, j) for i in 1:4, j in 1:3]
    ps = 98_000.0
    air = Array{FT}(undef, 4, 3, 3)
    for k in 1:3, j in 1:3, i in 1:4
        dp = (A[k + 1] + B[k + 1] * ps) - (A[k] + B[k] * ps)
        air[i, j, k] = FT(dp * area[i, j] / G * (1 + 0.01 * i))   # slightly column dependent
    end
    state = CellState(DryBasis, air; co2 = air .* FT(q), ch4 = air .* FT(1.9e-6))
    return (; state, grid), mesh
end

function sampler_spec(dir; split = "single", sources, kwargs...)
    obs_path = split == "daily" ? joinpath(dir, "obs_{YYYYMMDD}.nc") : joinpath(dir, "obs.nc")
    output_cfg = Dict{String, Any}("path" => joinpath(dir, "snap.nc"), "hours" => [0.0], "split" => split,
        "observations" => Dict{String, Any}("path" => obs_path,
                                            "sources" => sources, Dict(String(k) => v for (k, v) in kwargs)...))
    return observation_output_spec(output_cfg)
end

table_source(path; mode = "soundings") = Dict{String, Any}("kind" => "table", "mode" => mode, "path" => path)

@testset "linear bracketing across window ends" begin
    mktempdir() do dir
        csv = joinpath(dir, "soundings.csv")
        write(csv, """
            id,time,lat,lon
            before,2021-12-01T23:59:50,10,-100
            at0,2021-12-02T00:00:00,10,-100
            half,2021-12-02T00:30:00,10,-100
            late,2021-12-02T00:59:59,-20,30
            at1,2021-12-02T01:00:00,10,-100
            end,2021-12-02T01:59:59,10,-100
            held,2021-12-02T02:30:00,10,-100
            far,2021-12-02T05:33:20,10,-100
            """)
        sites = joinpath(dir, "sites.csv")
        write(sites, "id,lat,lon,intake_height,elevation\nlow,10,-100,10,0\nhigh,10,-100,100000,0\nnone,-20,30,,\n")
        spec = sampler_spec(dir; sources = Any[table_source(csv), table_source(sites; mode = "sites")],
                            tracers = ["co2"])
        model, mesh = fake_model()
        s = build_observation_sampler(spec, model.state, model.grid; origin = ORIGIN,
                                      window_seconds = (-86400.0, 86400.0), halo_width = 0)
        @test s isa ObservationSampler
        @test length(s.soundings) == 8 && length(s.sites) == 3
        @test issorted(s.sounding_times)
        @test_throws ArgumentError observe_window_boundary!(s, model.state, 0.0; next_window_seconds = 3600.0)
        begin_observation_day!(s, "20211202", 1)
        observe_window_boundary!(s, model.state, 0.0; next_window_seconds = 3600.0)
        @test s.counters.dropped_before_start == 1
        @test s.pending !== nothing && length(s.pending.range) == 3      # at0, half, late
        model.state.tracers_raw[:, :, :, 1] .*= 2                           # co2 → 800 ppm
        observe_window_boundary!(s, model.state, 3600.0; next_window_seconds = 3600.0)
        @test s.counters.emitted == 3
        @test_throws ArgumentError observe_window_boundary!(s, model.state, 3600.0; next_window_seconds = 3600.0)
        model.state.tracers_raw[:, :, :, 1] .*= 2                           # co2 → 1600 ppm
        observe_window_boundary!(s, model.state, 7200.0; next_window_seconds = 3600.0)
        @test s.counters.emitted == 5
        finish_observations!(s)
        @test s.counters.dropped_after_end == 2                              # held + far
        @test_throws ArgumentError observe_window_boundary!(s, model.state, 10800.0; next_window_seconds = 3600.0)
        close(s)
        @test close(s) === nothing

        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test ds.attrib["output_contract"] == "AtmosTransport observations v1"
            @test ds.attrib["completed_soundings"] == 5
            @test ds.attrib["n_before_start"] == 1
            @test ds.attrib["n_after_end"] == 2
            @test ds.attrib["n_one_sided"] == 0
            @test ds.attrib["mass_basis"] == "dry"
            @test ds.attrib["time_interpolation"] == "linear"
            @test occursin("\"kind\":\"table\"", ds.attrib["sources"])
            @test ds["id"][:] == ["at0", "half", "late", "at1", "end"]
            # CF time decodes to DateTime on read; the raw variable holds unix seconds.
            @test ds["time"][:] == [ORIGIN, ORIGIN + Minute(30), ORIGIN + Second(3599),
                                    ORIGIN + Hour(1), ORIGIN + Second(7199)]
            @test ds["time"].var[:] == [unix(ORIGIN), unix(ORIGIN + Minute(30)), unix(ORIGIN + Second(3599)),
                                        unix(ORIGIN + Hour(1)), unix(ORIGIN + Second(7199))]
            @test ds["time"].attrib["units"] == "seconds since 1970-01-01 00:00:00"
            w = ds["interp_weight"][:]
            @test w[1] == 0 && w[2] == 0.5f0 && w[3] ≈ 3599 / 3600 && w[4] == 0 && w[5] ≈ 7199 / 3600 - 1
            @test all(ds["interp_flag"][:] .== 0)
            @test ds["sample_time_prev"][:] == [ORIGIN, ORIGIN, ORIGIN, ORIGIN + Hour(1), ORIGIN + Hour(1)]
            @test ds["sample_time_next"][:] == [ORIGIN + Hour(1), ORIGIN + Hour(1), ORIGIN + Hour(1),
                                                ORIGIN + Hour(2), ORIGIN + Hour(2)]
            co2 = ds["co2"][:, :]
            @test size(co2) == (3, 5)
            @test all(co2[:, 1] .≈ 400e-6)                         # w = 0 → first sample
            @test all(co2[:, 2] .≈ 600e-6)                         # half way between 400 and 800
            @test all(co2[:, 4] .≈ 800e-6)
            @test all(isapprox.(co2[:, 5], 1600e-6 - 800e-6 * (1 / 3600); rtol = 1e-5))   # w = 3599/3600 between 800 and 1600
            means = ds["co2_column_mean"][:]
            @test means[2] ≈ 600e-6 rtol = 1e-6
            @test ds["latitude"][:] == [10, 10, -20, 10, 10]
            @test ds["longitude"][3] == 30
            @test ds["cell_i"][3] == 3 && ds["cell_j"][3] == 2        # (30°E, 20°S) on a 4×3 mesh
            @test !haskey(ds, "cell_panel")
            ps = ds["ps_dry"][:]
            p_half = ds["p_half_dry"][:, :]
            @test size(p_half) == (4, 5)
            @test all(p_half[1, :] .== 100)
            @test all(p_half[end, :] .== ps)
            # Reconstructed surface pressure of cell (1,1): 1.01 × 98000 of dry mass above p_top.
            @test ps[1] ≈ 100 + (98_000 - 100) * 1.01 rtol = 2e-6
            per_area = ds["air_mass_per_area_dry"][:, :]
            @test all(diff(p_half[:, 1]) .≈ per_area[:, 1] .* G)
            @test haskey(ds, "cell_area") && haskey(ds, "source")
            @test ds["source"][:] == Int32[1, 1, 1, 1, 1]
            @test NCDatasets.fillvalue(ds["co2"].var) == 1f15
        end
        NCDataset(joinpath(dir, "obs_sites.nc"), "r") do ds
            @test ds.attrib["completed_times"] == 3
            @test ds["site_id"][:] == ["low", "high", "none"]
            @test ds["time"][:] == [ORIGIN, ORIGIN + Hour(1), ORIGIN + Hour(2)]
            co2 = ds["co2"][:, :]
            @test size(co2) == (3, 3)
            @test all(co2[:, 1] .≈ 400e-6) && all(co2[:, 2] .≈ 800e-6) && all(co2[:, 3] .≈ 1600e-6)
            @test ds["co2_surface"][:, :] ≈ co2
            levels = ds["intake_level"][:, :]
            @test all(levels[1, :] .== 3)                  # 10 m lies in the lowest layer
            @test all(levels[2, :] .== 1)                  # 100 km is above the column top
            @test all(levels[3, :] .== 3)                  # unknown intake → lowest layer
            @test all(ds["height_method"][:, :] .== 0)
            @test all(ds["intake_layer_bottom_agl"][1, :] .== 0)
            @test all(ds["intake_layer_top_agl"][1, :] .> 10)
            @test all(ds["intake_layer_top_agl"][2, :] .> 50_000)    # column top with p_top = 100 Pa (~56 km)
            @test isnan(ds["intake_height"][3])
            @test !haskey(ds, "co2_profile")
            @test ds.attrib["n_records"] == 3
        end
    end
end

@testset "nearest-window sampling, daily files, site profiles" begin
    mktempdir() do dir
        csv = joinpath(dir, "soundings.csv")
        write(csv, """
            id,time,lat,lon
            a,2021-12-02T00:00:00,10,-100
            b,2021-12-02T00:30:00,10,-100
            c,2021-12-02T00:59:59,10,-100
            d,2021-12-02T01:00:00,10,-100
            e,2021-12-02T01:59:59,10,-100
            """)
        sites = joinpath(dir, "sites.csv")
        write(sites, "id,lat,lon,intake_height\nlow,10,-100,10\n")
        spec = sampler_spec(dir; split = "daily", sources = Any[table_source(csv), table_source(sites; mode = "sites")],
                            time_interpolation = "nearest_window", write_profile_for_sites = true)
        model, _ = fake_model()
        s = build_observation_sampler(spec, model.state, model.grid; origin = ORIGIN,
                                      window_seconds = (0.0, 86400.0), halo_width = 0)
        begin_observation_day!(s, "20211202", 1)
        observe_window_boundary!(s, model.state, 0.0; next_window_seconds = 3600.0)       # emits a
        model.state.tracers_raw[:, :, :, 1] .*= 2
        observe_window_boundary!(s, model.state, 3600.0; next_window_seconds = 3600.0)    # emits b, c, d
        begin_observation_day!(s, "20211203", 2)
        model.state.tracers_raw[:, :, :, 1] .*= 2
        observe_window_boundary!(s, model.state, 7200.0; next_window_seconds = 3600.0)    # emits e
        finish_observations!(s)
        close(s)
        NCDataset(joinpath(dir, "obs_20211202_soundings.nc"), "r") do ds
            @test ds["id"][:] == ["a", "b", "c", "d"]
            @test all(isnan, ds["interp_weight"][:])
            @test all(ds["interp_flag"][:] .== 0)
            @test ds["sample_time_prev"][:] == ds["sample_time_next"][:]
            @test ds["co2"][1, :] ≈ [400e-6, 800e-6, 800e-6, 800e-6]
            @test ds.attrib["completed_soundings"] == 4
        end
        NCDataset(joinpath(dir, "obs_20211203_soundings.nc"), "r") do ds
            @test ds["id"][:] == ["e"]
            @test ds["co2"][1, 1] ≈ 1600e-6
            @test ds.attrib["n_after_end"] == 0
        end
        NCDataset(joinpath(dir, "obs_20211203_sites.nc"), "r") do ds
            @test ds.attrib["completed_times"] == 1
            @test size(ds["co2_profile"]) == (3, 1, 1)
            @test all(ds["co2_profile"][:, 1, 1] .≈ 1600e-6)
            @test size(ds["p_half_dry"]) == (4, 1, 1)
            @test ds["p_half_dry"][1, 1, 1] == 100
            @test size(ds["air_mass_per_area_dry"]) == (3, 1, 1)
        end
        @test s.counters.emitted == 5
    end
end

@testset "a changed window length yields one-sided rows instead of silent gaps" begin
    mktempdir() do dir
        csv = joinpath(dir, "soundings.csv")
        write(csv, """
            id,time,lat,lon
            p,2021-12-02T00:16:40,10,-100
            q,2021-12-02T00:41:40,10,-100
            r,2021-12-02T01:40:00,10,-100
            """)
        spec = sampler_spec(dir; sources = Any[table_source(csv)])
        model, _ = fake_model()
        s = build_observation_sampler(spec, model.state, model.grid; origin = ORIGIN,
                                      window_seconds = (0.0, 86400.0), halo_width = 0)
        begin_observation_day!(s, "20211202", 1)
        observe_window_boundary!(s, model.state, 0.0; next_window_seconds = 3600.0)       # holds p, q (guess: 1 h)
        @test length(s.pending.range) == 2
        observe_window_boundary!(s, model.state, 1800.0; next_window_seconds = 3600.0)    # actual window was 30 min
        @test s.counters.emitted == 1                                                     # p bracketed
        @test length(s.pending.range) == 1                                                # q re-held
        observe_window_boundary!(s, model.state, 7200.0; next_window_seconds = 3600.0)    # actual window 90 min
        @test s.counters.emitted == 3
        @test s.counters.one_sided == 1                                                   # r had no first sample
        finish_observations!(s); close(s)
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test ds["id"][:] == ["p", "q", "r"]
            @test ds["interp_flag"][:] == Int8[0, 0, 1]
            @test ds["interp_weight"][:] ≈ Float32[1000 / 1800, (2500 - 1800) / 5400, 1]
            @test ds["sample_time_prev"][3] == ds["sample_time_next"][3] == ORIGIN + Hour(2)
            @test ds.attrib["n_one_sided"] == 1
        end
    end
end

@testset "build-time validation and unlocated requests" begin
    mktempdir() do dir
        csv = joinpath(dir, "soundings.csv")
        write(csv, "id,time,lat,lon\na,2021-12-02T00:10:00,10,-100\n")
        model, _ = fake_model()
        bad = sampler_spec(dir; sources = Any[table_source(csv)], tracers = ["sf6"])
        @test_throws ArgumentError build_observation_sampler(bad, model.state, model.grid; origin = ORIGIN,
                                                             window_seconds = (0.0, 86400.0), halo_width = 0)
        hybrid_top = AtmosGrid(model.grid.horizontal, HybridSigmaPressure([0.0, 0.0, 0.0, 0.0], [0.01, 0.1, 0.5, 1.0]), CPU(); FT = Float64)
        spec = sampler_spec(dir; sources = Any[table_source(csv)])
        @test_throws ArgumentError build_observation_sampler(spec, model.state, hybrid_top; origin = ORIGIN,
                                                             window_seconds = (0.0, 86400.0), halo_width = 0)
        # Regional mesh: requests outside the domain are counted, not sampled.
        regional = LatLonMesh(; FT = Float64, Nx = 4, Ny = 3, longitude = (0, 40), latitude = (0, 30))
        rgrid = AtmosGrid(regional, model.grid.vertical, CPU(); FT = Float64)
        sites = joinpath(dir, "sites.csv")
        write(sites, "id,lat,lon\ninside,10,20\noutside,-50,100\n")
        rspec = sampler_spec(dir; sources = Any[table_source(csv), table_source(sites; mode = "sites")])
        rs = build_observation_sampler(rspec, model.state, rgrid; origin = ORIGIN, window_seconds = (0.0, 86400.0), halo_width = 0)
        @test isempty(rs.soundings) && rs.counters.unlocated_soundings == 1
        @test length(rs.sites) == 1 && rs.counters.unlocated_sites == 1
        begin_observation_day!(rs, "20211202", 1)
        observe_window_boundary!(rs, model.state, 0.0; next_window_seconds = 3600.0)
        finish_observations!(rs); close(rs)
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test ds.attrib["completed_soundings"] == 0
            @test ds.attrib["n_unlocated_soundings"] == 1
            @test ds.dim["obs"] == 0
        end
        NCDataset(joinpath(dir, "obs_sites.nc"), "r") do ds
            @test ds["site_id"][:] == ["inside"]
            @test ds.attrib["n_unlocated_sites"] == 1
        end
    end
end

@testset "gather plans group by panel" begin
    cells = [CellLocation(p, 1, 1, 1, c, 0.0, 0.0, 1.0) for (p, c) in ((3, 30), (1, 10), (3, 31), (6, 60), (1, 11))]
    plan = GatherPlan(cells, 6)
    @test plan.columns == Int32[10, 11, 30, 31, 60]
    @test plan.panel_ranges == [1:2, 3:2, 3:4, 5:4, 5:4, 5:5]
    @test plan.position == [3, 1, 4, 5, 2]
    single = GatherPlan(cells[2:2], 1)
    @test single.columns == Int32[10] && single.panel_ranges == [1:1] && single.position == [1]
end

@testset "stream contracts: poisoning, dimension checks, idempotent close" begin
    mktempdir() do dir
        mesh = LatLonMesh(; FT = Float64, Nx = 2, Ny = 2)
        stream = SoundingNetCDFStream(joinpath(dir, "s.nc"), mesh, 2, [:co2]; mass_basis = :dry, origin = ORIGIN)
        cell = CellLocation(1, 1, 1, 1, 1, 0.0, 0.0, 1.0)
        good = SoundingBatch(["x"], Int32[1], [0.0], [0.0], [0.0], [cell], [0.0], [0.0], Float32[0], Int8[0],
                             reshape([1.0, 2.0, 3.0], 3, 1), reshape([1.0, 1.0], 2, 1),
                             reshape([4e-4, 4e-4], 2, 1, 1), reshape([4e-4], 1, 1))
        @test append_soundings!(stream, good) == 1
        # Inconsistent batches cannot be constructed; consistent ones of the wrong depth are refused.
        @test_throws DimensionMismatch SoundingBatch(["y"], Int32[1], [0.0], [0.0], [0.0], [cell], [0.0], [0.0],
                                                     Float32[0], Int8[0], reshape([1.0, 2.0], 2, 1),
                                                     reshape([1.0, 1.0], 2, 1), reshape([4e-4, 4e-4], 2, 1, 1),
                                                     reshape([4e-4], 1, 1))
        @test_throws DimensionMismatch SoundingBatch(["y", "z"], Int32[1], [0.0], [0.0], [0.0], [cell], [0.0], [0.0],
                                                     Float32[0], Int8[0], reshape([1.0, 2.0, 3.0], 3, 1),
                                                     reshape([1.0, 1.0], 2, 1), reshape([4e-4, 4e-4], 2, 1, 1),
                                                     reshape([4e-4], 1, 1))
        deep = SoundingBatch(["y"], Int32[1], [0.0], [0.0], [0.0], [cell], [0.0], [0.0], Float32[0], Int8[0],
                             reshape(collect(1.0:4.0), 4, 1), reshape([1.0, 1.0, 1.0], 3, 1),
                             reshape(fill(4e-4, 3), 3, 1, 1), reshape([4e-4], 1, 1))
        @test_throws DimensionMismatch append_soundings!(stream, deep)
        # A failed append poisons the stream; the published count is unchanged.
        poisoned = SoundingNetCDFStream(joinpath(dir, "p.nc"), mesh, 2, [:co2]; mass_basis = :dry, origin = ORIGIN)
        close(poisoned.dataset)    # sabotage the dataset underneath the stream
        @test_throws Exception append_soundings!(poisoned, good)
        @test poisoned.failed && poisoned.closed
        @test_throws ArgumentError append_soundings!(poisoned, good)
        close(stream)
        @test close(stream) === nothing
        @test_throws ArgumentError append_soundings!(stream, good)
        NCDataset(joinpath(dir, "s.nc"), "r") do ds
            @test ds.attrib["completed_soundings"] == 1
            @test ds["id"][:] == ["x"]
        end
        @test_throws ArgumentError SoundingNetCDFStream(joinpath(dir, "t.nc"), mesh, 2, Symbol[]; mass_basis = :dry, origin = ORIGIN)
        site = O.SiteRequest("a", 0.0, 0.0, NaN, NaN, 1)
        sstream = SiteNetCDFStream(joinpath(dir, "sites.nc"), mesh, 2, [:co2], [site], [cell]; mass_basis = :moist, origin = ORIGIN)
        record = SiteRecord(0.0, 1, 2, 1; profiles = false)
        record.ps .= 1e5; record.intake_level .= 2; record.layer_bottom .= 0; record.layer_top .= 10
        record.height_method .= 0; record.values .= 4e-4; record.surface_values .= 4e-4
        @test size(record.profiles) == (0, 0, 0)
        @test size(SiteRecord(0.0, 3, 5, 2; profiles = true).profiles) == (5, 3, 2)
        @test append_site_record!(sstream, record) == 1
        short = SiteRecord(0.0, Float64[], Int32[], Float64[], Float64[], Int8[], zeros(0, 1), zeros(0, 1),
                           zeros(0, 0), zeros(0, 0), zeros(0, 0, 0))
        @test_throws DimensionMismatch append_site_record!(sstream, short)
        close(sstream)
        NCDataset(joinpath(dir, "sites.nc"), "r") do ds
            @test haskey(ds, "ps") && !haskey(ds, "ps_dry")      # moist basis: no suffix
            @test ds.attrib["mass_basis"] == "moist"
        end
    end
end

@testset "a window without soundings followed by a longer one emits one-sided rows" begin
    mktempdir() do dir
        csv = joinpath(dir, "soundings.csv")
        write(csv, "id,time,lat,lon\ngap1,2021-12-02T01:30:00,10,-100\ngap2,2021-12-02T01:40:00,10,-100\nlater,2021-12-02T02:30:00,10,-100\n")
        spec = sampler_spec(dir; sources = Any[table_source(csv)])
        model, _ = fake_model()
        s = build_observation_sampler(spec, model.state, model.grid; origin = ORIGIN,
                                      window_seconds = (0.0, 86400.0), halo_width = 0)
        begin_observation_day!(s, "20211202", 1)
        begin_observation_day!(s, "20211203", 2)          # single file: later calls are no-ops
        observe_window_boundary!(s, model.state, 0.0; next_window_seconds = 3600.0)       # nothing held
        @test s.pending === nothing
        observe_window_boundary!(s, model.state, 7200.0; next_window_seconds = 3600.0)    # window was 2 h
        @test s.counters.dropped_before_start == 0
        @test s.counters.emitted == 2 && s.counters.one_sided == 2
        observe_window_boundary!(s, model.state, 10800.0; next_window_seconds = 3600.0)
        finish_observations!(s); close(s)
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test ds["id"][:] == ["gap1", "gap2", "later"]
            @test ds["interp_flag"][:] == Int8[1, 1, 0]
            @test ds["sample_time_prev"][1] == ds["sample_time_next"][1] == ORIGIN + Hour(2)
            @test ds.attrib["n_one_sided"] == 2 && ds.attrib["n_before_start"] == 0
        end
    end
end

@testset "per-layer temperature drives site heights; Float32 and cubed sphere" begin
    mktempdir() do dir
        sites = joinpath(dir, "sites.csv")
        write(sites, "id,lat,lon,intake_height\nlow,10,-100,10\n")
        csv = joinpath(dir, "soundings.csv")
        write(csv, "id,time,lat,lon\na,2021-12-02T00:30:00,10,-100\n")
        spec = sampler_spec(dir; sources = Any[table_source(csv), table_source(sites; mode = "sites")])
        model, _ = fake_model(; FT = Float32)
        s = build_observation_sampler(spec, model.state, model.grid; origin = ORIGIN,
                                      window_seconds = (0.0, 86400.0), halo_width = 0)
        begin_observation_day!(s, "20211202", 1)
        warm = fill(Float32(300), size(model.state.air_mass))
        observe_window_boundary!(s, model.state, 0.0; next_window_seconds = 3600.0, temperature = warm)
        observe_window_boundary!(s, model.state, 3600.0; next_window_seconds = 3600.0)
        finish_observations!(s); close(s)
        NCDataset(joinpath(dir, "obs_sites.nc"), "r") do ds
            @test ds["height_method"][1, :] == Int8[2, 0]
            # 300 K vs the 280 K fallback: the lowest layer is 300/280 thicker.
            top = ds["intake_layer_top_agl"][1, :]
            @test top[1] / top[2] ≈ 300 / 280 rtol = 1e-5
        end
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test all(isapprox.(ds["co2"][:, 1], 400e-6; rtol = 1e-6))
        end
    end
    mktempdir() do dir
        mesh = CubedSphereMesh(; FT = Float64, Nc = 4, Hp = 2)
        Np = 8
        A = [100.0, 0.0, 0.0]; B = [0.0, 0.5, 1.0]
        grid = AtmosGrid(mesh, HybridSigmaPressure(A, B), CPU(); FT = Float64)
        air = ntuple(p -> fill(1e15 * p, Np, Np, 2), 6)
        state = CubedSphereState(DryBasis, air; co2 = map(m -> m .* 400e-6, air), halo_width = 2)
        csv = joinpath(dir, "soundings.csv")
        write(csv, "id,time,lat,lon\nnp,2021-12-02T00:30:00,89,0\nsp,2021-12-02T00:30:00,-89,0\neq,2021-12-02T00:30:00,0,100\n")
        spec = sampler_spec(dir; sources = Any[table_source(csv)])
        sites = joinpath(dir, "sites.csv")
        write(sites, "id,lat,lon,intake_height\nn,60,10,10\ns,-60,200,10\n")
        spec = sampler_spec(dir; sources = Any[table_source(csv), table_source(sites; mode = "sites")])
        s = build_observation_sampler(spec, state, grid; origin = ORIGIN, window_seconds = (0.0, 86400.0), halo_width = 2)
        begin_observation_day!(s, "20211202", 1)
        # GCHP VDIFF temperature panels are interior-only (Nc × Nc × Nz) while air mass is halo-padded.
        warm = ntuple(p -> [250.0 + 10p + i + 0.1j for i in 1:4, j in 1:4, k in 1:2], 6)
        observe_window_boundary!(s, state, 0.0; next_window_seconds = 3600.0, temperature = warm)
        observe_window_boundary!(s, state, 3600.0; next_window_seconds = 3600.0)
        bad = ntuple(_ -> fill(300.0, 5, 5, 2), 6)
        @test_throws DimensionMismatch observe_window_boundary!(s, state, 7200.0; next_window_seconds = 3600.0,
                                                                temperature = bad)
        finish_observations!(s); close(s)
        NCDataset(joinpath(dir, "obs_sites.nc"), "r") do ds
            @test ds["height_method"][:, 1] == Int8[2, 2]
            @test ds["height_method"][:, 2] == Int8[0, 0]
            top = ds["intake_layer_top_agl"][:, :]
            expected = [(250 + 10ds["cell_panel"][n] + ds["cell_i"][n] + 0.1ds["cell_j"][n]) / 280 for n in 1:2]
            @test top[:, 1] ./ top[:, 2] ≈ expected rtol = 1e-5
            @test length(unique(ds["cell_panel"][:])) == 2
        end
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            panels = ds["cell_panel"][:]
            @test length(unique(panels)) == 3
            @test all(ds["co2"][:, :] .≈ 400e-6)
            # Per-panel air mass differs (1e15·p), so each row must carry its own panel's mass.
            per_area = ds["air_mass_per_area_dry"][1, :]
            areas = ds["cell_area"][:]
            @test per_area .* areas ≈ 1e15 .* panels rtol = 1e-6
        end
    end
end

@testset "no-op sampler interface" begin
    @test build_observation_sampler(NoObservationOutput()) isa NoObservationSampler
    @test !samples_observations(NoObservationSampler())
end
