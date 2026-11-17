# Time-mean ([output.time_mean]) and local-solar-time ([output.local_time]) output:
# boundary weights, synthetic states with known answers on every topology,
# config errors, and driven runs checked against hourly snapshots.
using Test
using Dates
using Logging
using NCDatasets
using AtmosTransport
using AtmosTransport.Output
using AtmosTransport.Output: aggregate_output_specs, TimeAggregateOutputs, sample_window_boundary!,
                             finish_time_aggregates!, boundary_weights, time_weightings,
                             TimeMeanAggregate, LocalSolarTimeAggregate, TrapezoidalWeights,
                             CellWeights
include(joinpath(@__DIR__, "..", "fixtures", "cs_multifile.jl"))
using .CSDriverHandoffFixtures

const ORIGIN = DateTime(2021, 12, 1)

# Sum the boundary weights of one period, cell by cell (sparse local-time
# weights scattered into one vector over all horizontal parts).
weight_sum(w::Real) = w
weight_sum(w::AbstractArray) = vec(w)
weight_sum(w::Tuple) = reduce(vcat, map(vec, w))
function weight_sum(w::CellWeights)
    parts = [zeros(n) for n in w.ncell]
    foreach((part, c, v) -> (part[c] .= v), parts, w.cells, w.weights)
    return reduce(vcat, parts)
end
period_weight_sums(weighting, n, Δ) = sum(k -> weight_sum(boundary_weights(weighting, k, n, Δ)), 0:n)

function aggregates_for(grid, output_cfg; names = [:co2], window_hours = 1.0)
    specs = aggregate_output_specs(output_cfg, Float64)
    return TimeAggregateOutputs(specs, grid, names; window_hours, halo_width = 0, mass_basis = :dry,
                                physics_attributes = Dict("physics_advection_scheme" => "test"),
                                time_origin = ORIGIN)
end

# Feed boundaries 0..nboundary of full snapshot frames `frame_at(t)`.
function feed!(aggregates, frame_at, nboundary; window_hours = 1.0)
    for b in 0:nboundary
        t = b * window_hours
        sample_window_boundary!(aggregates, nothing, t; shared_frame = frame_at(t))
    end
    finish_time_aggregates!(aggregates)
    close(aggregates)
    return nothing
end

ll_grid(Nx, Ny, Nz) = AtmosGrid(LatLonMesh(; FT = Float64, Nx, Ny),
                                HybridSigmaPressure(collect(Float64, 0:Nz), collect(range(0.0, 1.0; length = Nz + 1))),
                                CPU(); FT = Float64)

@testset "boundary weights sum to one in every cell" begin
    for n in (1, 8, 24)
        @test period_weight_sums(TrapezoidalWeights(), n, 24 / n) ≈ 1 atol = 1e-15
    end
    ll = LatLonMesh(; FT = Float64, Nx = 36, Ny = 4)
    cs = CubedSphereMesh(; FT = Float64, Nc = 6, convention = GEOSNativePanelConvention())
    rg = ReducedGaussianMesh([-45.0, 45.0], [4, 8]; FT = Float64)
    for mesh in (ll, cs, rg), Δ in (1.0, 3.0)
        n = round(Int, 24 / Δ)
        for w in time_weightings(LocalSolarTimeAggregate([0.0, 4.0, 13.5, 23.99]), mesh, Δ)
            sums = period_weight_sums(w, n, Δ)
            @test all(s -> isapprox(s, 1; atol = 1e-12), sums)
            @test all(t -> 0 <= t <= 24, weight_sum(w.utc_hours))
            # At most two boundaries weight a cell; the sparse lists hold only those.
            @test sum(k -> sum(length, boundary_weights(w, k, n, Δ).cells), 0:n) <= 2 * length(sums)
        end
    end
end

@testset "time mean of a state linear in time is the midpoint value" begin
    mktempdir() do dir
        grid = ll_grid(6, 3, 2)
        air = [1e9 * (1 + 0.1i + 0.01j + 0.5k) for i in 1:6, j in 1:3, k in 1:2]
        vmr(t) = [4e-4 + 1e-6 * (i + j) + 2e-6 * k + 1e-7 * t for i in 1:6, j in 1:3, k in 1:2]
        frame_at(t) = SnapshotFrame(t, air, Dict(:co2 => vmr(t) .* air), :dry)
        all_fields = Dict("layers" => "full", "column_mass_per_area" => true, "air_mass" => true,
                          "air_mass_per_area" => true, "column_air_mass_per_area" => true)
        for (window_hours, nwindow) in ((1.0, 48), (3.0, 16))
            path = joinpath(dir, "mean_$(Int(window_hours)).nc")
            aggregates = aggregates_for(grid, Dict("time_mean" => Dict("path" => path, "fields" => all_fields));
                                        window_hours)
            feed!(aggregates, frame_at, nwindow; window_hours)
            NCDataset(path) do ds
                @test ds["time"].var[:] == [12.0, 36.0]
                @test ds["time_bnds"].var[:, :] == [0.0 24.0; 24.0 48.0]
                @test ds["time"].attrib["bounds"] == "time_bnds"
                @test ds.attrib["output_product"] == "time_mean"
                @test ds.attrib["completed_records"] == 2
                @test ds["co2_column_mean"].attrib["cell_methods"] == "lev: mean time: mean"
                @test ds["co2"].attrib["cell_methods"] == "time: mean"
                @test ds["co2_total_mass"].attrib["cell_methods"] == "time: mean"
                for (p, mid) in enumerate((12.0, 36.0))
                    expected = vmr(mid)
                    @test ds["co2"][:, :, :, p] ≈ expected rtol = 1e-14
                    cm = sum(expected .* air; dims = 3)[:, :, 1] ./ sum(air; dims = 3)[:, :, 1]
                    @test ds["co2_column_mean"][:, :, p] ≈ cm rtol = 1e-14
                    @test ds["air_mass"][:, :, :, p] ≈ air rtol = 1e-14
                    @test ds["co2_total_mass"][p] ≈ sum(expected .* air) rtol = 1e-14
                end
            end
        end
    end
end

# A mixing ratio linear in time within each window: the local-time composite
# must equal f at each cell's UTC sample hour.
f_lt(t, lon, k) = 4e-4 + 1e-7 * t + 1e-6 * sind(lon) + 3e-6 * k

@testset "local-time composite of f(t, lon) on $(label)" for (label, grid) in (
        ("lat-lon", ll_grid(24, 3, 2)),
        ("cubed sphere", AtmosGrid(CubedSphereMesh(; FT = Float64, Nc = 4,
                                                   convention = GEOSNativePanelConvention()),
                                   HybridSigmaPressure([0.0, 1.0, 2.0], [0.0, 0.5, 1.0]), CPU(); FT = Float64)),
        ("reduced Gaussian", AtmosGrid(ReducedGaussianMesh([-45.0, 45.0], [4, 8]; FT = Float64),
                                       HybridSigmaPressure([0.0, 1.0, 2.0], [0.0, 0.5, 1.0]), CPU(); FT = Float64)))
    mktempdir() do dir
        mesh = grid.horizontal
        lons = AtmosTransport.Output._cell_longitudes(mesh)
        layered(g) = lons isa Tuple ? map(l -> cat((g.(l, k) for k in 1:2)...; dims = 3), lons) :
                     cat((g.(lons, k) for k in 1:2)...; dims = ndims(lons) + 1)
        air = layered((lon, k) -> 1e9 * (1 + 0.3k))
        storage(a::Tuple, x::Tuple) = map(.*, a, x)
        storage(a, x) = a .* x
        frame_at(t) = SnapshotFrame(t, air, Dict(:co2 => storage(air, layered((lon, k) -> f_lt(t, lon, k)))),
                                    :dry)
        aggregates = aggregates_for(grid, Dict("split" => "daily",
            "local_time" => Dict("path" => joinpath(dir, "lt_{YYYYMMDD}.nc"), "hours" => [13.5, 0.0, 4.0],
                                 "fields" => Dict("layers" => "full"))))
        @test_logs (:warn, r"incomplete") match_mode = :any feed!(aggregates, frame_at, 54)
        @test sort(readdir(dir)) == ["lt_20211201.nc", "lt_20211202.nc"]
        for (day, file) in enumerate(("lt_20211201.nc", "lt_20211202.nc"))
            NCDataset(joinpath(dir, file)) do ds
                @test ds["local_time"][:] == [0.0, 4.0, 13.5]
                @test ds["time_bnds"].var[:, 1] == [24.0(day - 1), 24.0day]
                hdims = label == "cubed sphere" ? 3 : label == "lat-lon" ? 2 : 1
                file_lons = label == "cubed sphere" ? ds["lons"][:, :, :] :
                            label == "lat-lon" ? repeat(ds["lon"][:], 1, 3) : ds["cell_lon"][:]
                for (j, T) in enumerate((0.0, 4.0, 13.5))
                    utc = mod.(T .- file_lons ./ 15, 24)
                    sel = ntuple(_ -> Colon(), hdims)
                    @test ds["utc_sample_hour"][sel..., j] ≈ utc atol = 1e-12
                    t = 24.0(day - 1) .+ utc
                    cm_name = label == "reduced Gaussian" ? "co2_column_mean_native" : "co2_column_mean"
                    @test ds[cm_name][sel..., j, 1] ≈
                          (f_lt.(t, file_lons, 1) .* 1.3 .+ f_lt.(t, file_lons, 2) .* 1.6) ./ 2.9 rtol = 1e-13
                    @test ds["co2"][sel..., 2, j, 1] ≈ f_lt.(t, file_lons, 2) rtol = 1e-13
                end
                @test !haskey(ds, "co2_total_mass")
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Driven runs
# ---------------------------------------------------------------------------

function write_ll_binary(path; nwindow, Nx = 12, Ny = 6, Nz = 2, window_seconds = 3600.0)
    grid = ll_grid(Nx, Ny, Nz)
    windows = [(m = fill(1e10, Nx, Ny, Nz), am = fill(1e9 * window_seconds / 3600, Nx + 1, Ny, Nz),
                bm = zeros(Nx, Ny + 1, Nz), cm = zeros(Nx, Ny, Nz + 1), ps = fill(95000.0, Nx, Ny))
               for _ in 1:nwindow]
    write_transport_binary(path, grid, windows; FT = Float64, dt_met_seconds = window_seconds,
                           half_dt_seconds = window_seconds / 2, steps_per_window = 1, mass_basis = :dry,
                           surface_pressure_basis = TotalSurfacePressure(),
                           source_flux_sampling = :window_start_endpoint, flux_sampling = :window_constant)
    return path
end

ll_run_cfg(bin, output; run = Dict{String, Any}(), float_type = "Float64") = Dict{String, Any}(
    "input" => Dict("binary_paths" => bin isa AbstractVector ? bin : [bin], "start_date" => "2021-12-01"),
    "numerics" => Dict("float_type" => float_type),
    "logging" => Dict("progress" => "none"),
    "run" => run,
    "advection" => Dict("scheme" => "upwind"),
    "chemistry" => Dict("kind" => "decay", "half_lives_seconds" => Dict("co2" => 2e5)),
    "tracers" => Dict("co2" => Dict("init" => Dict("kind" => "gaussian_blob", "background" => 4e-4,
                                                    "lon0_deg" => 90.0, "lat0_deg" => 0.0,
                                                    "sigma_lon_deg" => 40.0, "sigma_lat_deg" => 30.0,
                                                    "amplitude" => 1e-4))),
    "output" => output)

quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Warn))

const COLUMN_MEAN_ONLY = Dict("layers" => "none", "air_mass" => false, "air_mass_per_area" => false,
                              "column_air_mass_per_area" => false, "column_mass_per_area" => false)

# Recompute the daily mean and the local-time composites of day `day` (0-based)
# from hourly column means `cm[..., hour + 1]` and per-cell longitudes `lons`.
function check_against_hourly(cm, lons, mean, lt_values, lt_hours, utc_hours; day = 0)
    hdims = ndims(cm) - 1
    sel = ntuple(_ -> Colon(), hdims)
    hour(h) = cm[sel..., 24day + h + 1]
    ref_mean = sum(h -> hour(h) * (h == 0 || h == 24 ? 1 / 48 : 1 / 24), 0:24)
    @test mean ≈ ref_mean rtol = 1e-13
    @test maximum(abs.(mean .- hour(0))) > 1e-3 * maximum(abs.(hour(0)))  # the field changes
    for (j, T) in enumerate(lt_hours)
        tc = mod.(T .- lons ./ 15, 24)
        @test utc_hours[sel..., j] ≈ tc atol = 1e-12
        k0 = min.(floor.(Int, tc), 23)
        ref = [(1 - (tc[c] - k0[c])) * cm[c, 24day + k0[c] + 1] + (tc[c] - k0[c]) * cm[c, 24day + k0[c] + 2]
               for c in CartesianIndices(tc)]
        @test lt_values[sel..., j] ≈ ref rtol = 1e-13
    end
end

@testset "lat-lon run: products match the hourly snapshots" begin
    mktempdir() do dir
        bin = write_ll_binary(joinpath(dir, "transport_20211201.bin"); nwindow = 30)
        cfg = ll_run_cfg(bin, Dict("path" => joinpath(dir, "snap.nc"), "cadence_hours" => 1.0,
                                   "fields" => COLUMN_MEAN_ONLY,
                                   "time_mean" => Dict("path" => joinpath(dir, "mean.nc")),
                                   "local_time" => Dict("path" => joinpath(dir, "lt.nc"),
                                                        "hours" => [13.5, 4.0, 0.0])))
        @test validate_config(cfg) == (true, String[])
        warn_mean = (:warn, r"\[output.time_mean\].*incomplete")
        warn_lt = (:warn, r"\[output.local_time\].*incomplete")
        @test_logs warn_mean warn_lt match_mode = :any run_driven_simulation(cfg)
        NCDataset(joinpath(dir, "snap.nc")) do snap
            cm = snap["co2_column_mean"][:, :, :]
            @test size(cm, 3) == 31
            lons = repeat(snap["lon"][:], 1, size(cm, 2))
            NCDataset(joinpath(dir, "mean.nc")) do mean
                NCDataset(joinpath(dir, "lt.nc")) do lt
                    @test size(mean["co2_column_mean"]) == (12, 6, 1)
                    @test lt["local_time"][:] == [0.0, 4.0, 13.5]
                    @test mean.attrib["physics_advection_scheme"] == "upwind"
                    @test haskey(mean.attrib, "framework_commit")
                    check_against_hourly(cm, lons, mean["co2_column_mean"][:, :, 1],
                                         lt["co2_column_mean"][:, :, :, 1], lt["local_time"][:],
                                         lt["utc_sample_hour"][:, :, :])
                end
            end
        end
    end
end

@testset "cubed-sphere run: products match the hourly snapshots" begin
    mktempdir() do dir
        bin = joinpath(dir, "cs.bin")
        cs_handoff_fixture(bin, fill(1.0, 25))
        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [bin], "start_date" => "2021-12-01"),
            "numerics" => Dict("float_type" => "Float64"),
            "logging" => Dict("progress" => "none"),
            "advection" => Dict("scheme" => "upwind"),
            "convection" => Dict("kind" => "cmfmc"),
            "chemistry" => Dict("kind" => "decay", "half_lives_seconds" => Dict("co2" => 1e5)),
            "tracers" => Dict("co2" => Dict("init" => Dict("kind" => "pressure_layer",
                                                            "lowest_layer" => true,
                                                            "total_molecules" => 1e35))),
            "output" => Dict("path" => joinpath(dir, "snap.nc"), "cadence_hours" => 1.0,
                             "fields" => COLUMN_MEAN_ONLY,
                             "time_mean" => Dict("path" => joinpath(dir, "mean.nc")),
                             "local_time" => Dict("path" => joinpath(dir, "lt.nc"), "hours" => [13.5, 4.0])))
        @test validate_config(cfg) == (true, String[])
        quiet(() -> run_driven_simulation(cfg))
        NCDataset(joinpath(dir, "snap.nc")) do snap
            cm = snap["co2_column_mean"][:, :, :, :]
            @test size(cm, 4) == 26
            NCDataset(joinpath(dir, "mean.nc")) do mean
                NCDataset(joinpath(dir, "lt.nc")) do lt
                    @test haskey(mean, "cubed_sphere")
                    @test dimnames(lt["co2_column_mean"]) == ("Xdim", "Ydim", "nf", "local_time", "time")
                    check_against_hourly(cm, snap["lons"][:, :, :], mean["co2_column_mean"][:, :, :, 1],
                                         lt["co2_column_mean"][:, :, :, :, 1], lt["local_time"][:],
                                         lt["utc_sample_hour"][:, :, :, :])
                end
            end
        end
    end
end

@testset "run without snapshots; incomplete period" begin
    mktempdir() do dir
        bin = write_ll_binary(joinpath(dir, "transport_20211201.bin"); nwindow = 24)
        output = Dict("time_mean" => Dict("path" => joinpath(dir, "mean.nc"), "period_hours" => 6),
                      "local_time" => Dict("path" => joinpath(dir, "lt.nc"), "hours" => [12.0],
                                           "fields" => Dict("tracers" => ["co2"], "layers" => "full")))
        cfg = ll_run_cfg(bin, output)
        @test validate_config(cfg) == (true, String[])
        @test_logs min_level = Logging.Warn run_driven_simulation(cfg)
        @test sort(readdir(dir)) == ["lt.nc", "mean.nc", "transport_20211201.bin"]
        NCDataset(joinpath(dir, "mean.nc")) do ds
            @test ds["time"].var[:] == [3.0, 9.0, 15.0, 21.0]
            @test ds.attrib["completed_records"] == 4
            @test endswith(ds.attrib["history"], "with 4 record(s)")
            @test !haskey(ds, "co2")
        end
        NCDataset(joinpath(dir, "lt.nc")) do ds
            @test size(ds["co2"]) == (12, 6, 2, 1, 1)
        end

        short = ll_run_cfg(bin, Dict("time_mean" => Dict("path" => joinpath(dir, "short.nc"))),
                           run = Dict("stop_window" => 12))
        @test_logs (:warn, r"incomplete \(12 of 24 windows\)") match_mode = :any run_driven_simulation(short)
        @test !isfile(joinpath(dir, "short.nc"))
    end
end

@testset "config errors" begin
    errors_of(output) = last(validate_config(Dict{String, Any}(
        "input" => Dict("binary_paths" => ["/nonexistent/transport.bin"]),
        "tracers" => Dict("co2" => Dict("init" => Dict("kind" => "uniform", "background" => 4e-4))),
        "output" => output)))
    has_error(output, pattern) = any(e -> occursin(pattern, e), errors_of(output))
    @test has_error(Dict("time_mean" => Dict("path" => "m.nc", "period" => 24)),
                    r"Unknown \[output.time_mean\] option.*period.*did you mean `period_hours`")
    @test has_error(Dict("time_mean" => Dict("path" => "m.nc", "period_hours" => -24)), "finite and positive")
    @test has_error(Dict("time_mean" => Dict("path" => "m.nc", "period_hours" => "24")), "must be a number")
    @test has_error(Dict("time_mean" => Dict("period_hours" => 24)), "needs a `path`")
    @test has_error(Dict("time_mean" => "m.nc"), "[output.time_mean] must be a TOML table")
    @test has_error(Dict("local_time" => Dict("path" => "l.nc", "hours" => [24.0])), "[0, 24)")
    @test has_error(Dict("local_time" => Dict("path" => "l.nc", "hours" => [-1])), "[0, 24)")
    @test has_error(Dict("local_time" => Dict("path" => "l.nc", "hours" => 13.5)), "must be an array")
    @test has_error(Dict("local_time" => Dict("path" => "l.nc")), "must be an array")
    @test has_error(Dict("local_time" => Dict("path" => "l.nc", "hours" => [1, 1])), "more than once")
    @test has_error(Dict("local_time" => Dict("path" => "l.nc", "hours" => [true])), "must be a number")
    @test has_error(Dict("local_time" => Dict("path" => "l.nc", "hours" => [1.0],
                                              "fields" => Dict("column_means" => true))),
                    r"\[output.local_time.fields\]: unknown option.*column_means")
    @test has_error(Dict("local_time" => Dict("path" => "l.nc", "hours" => [1.0],
                                              "fields" => Dict("per_tracer" => Dict("ch4" => Dict())))),
                    "`ch4` is not a tracer of the run")
    @test has_error(Dict("time_mean" => Dict("path" => "a.nc"),
                         "local_time" => Dict("path" => "a.nc", "hours" => [1.0])), "would both write")
    @test has_error(Dict("path" => "a.nc", "hours" => [0.0], "time_mean" => Dict("path" => "a.nc")),
                    "would both write")
    # Collisions after day expansion: a daily template and a plain path name one file.
    @test has_error(Dict("path" => "x_{date}.nc", "split" => "daily", "hours" => [0.0],
                         "time_mean" => Dict("path" => "x.nc")),
                    r"\[output\].path and \[output.time_mean\].path would both write .*x_20000101.nc")
    @test has_error(Dict("split" => "daily", "time_mean" => Dict("path" => "m_{YYYYMMDD}.nc"),
                         "local_time" => Dict("path" => "m.nc", "hours" => [1.0])), "would both write")
    @test has_error(Dict("observations" => Dict("path" => "o.nc", "sources" => [Dict("kind" => "table",
                             "mode" => "sites", "path" => "sites.csv")]),
                         "time_mean" => Dict("path" => "o_sites.nc")),
                    r"\[output.time_mean\].path and \[output.observations\].path would both write .*o_sites.nc")
    @test !has_error(Dict("path" => "x_{date}.nc", "split" => "daily", "hours" => [0.0],
                          "time_mean" => Dict("path" => "x_mean.nc")), "would both write")
    # Field selections: selected layers need levels; tracers must be tracers of the run.
    @test has_error(Dict("time_mean" => Dict("path" => "m.nc", "fields" => Dict("layers" => "selected"))),
                    "[output.time_mean.fields].levels must be non-empty")
    # The same check covers written snapshot fields only: not disabled output, not the
    # binary format (it reads no field selection), not a table of a tracer left out.
    selected = Dict("layers" => "selected")
    @test has_error(Dict("path" => "s.nc", "hours" => [0.0], "fields" => selected),
                    "[output.fields].levels must be non-empty")
    for unread in (Dict("path" => "s.nc", "hours" => [0.0], "enabled" => false, "fields" => selected),
                   Dict("path" => "s.bin", "hours" => [0.0], "format" => "binary_mmap", "fields" => selected))
        @test !has_error(unread, "levels must be non-empty")
    end
    two_tracers = Dict{String, Any}(
        "input" => Dict("binary_paths" => ["/nonexistent/transport.bin"]),
        "tracers" => Dict(name => Dict("init" => Dict("kind" => "uniform", "background" => 4e-4))
                          for name in ("co2", "sf6")),
        "output" => Dict("path" => "s.nc", "hours" => [0.0],
                         "fields" => Dict("tracers" => ["co2"], "layers" => "none",
                                          "per_tracer" => Dict("sf6" => selected))))
    @test !any(e -> occursin("[output", e), last(validate_config(two_tracers)))
    two_tracers["output"]["fields"]["tracers"] = ["co2", "sf6"]
    @test any(e -> occursin("[output.fields].levels must be non-empty", e), last(validate_config(two_tracers)))
    @test has_error(Dict("local_time" => Dict("path" => "l.nc", "hours" => [1.0],
                                              "fields" => Dict("tracers" => ["ch4"]))),
                    "[output.local_time.fields].tracers: `ch4` is not a tracer of the run")
    # Only the new products, no snapshot path or schedule: valid.
    @test !any(e -> occursin("output", e), errors_of(Dict("time_mean" => Dict("path" => "m.nc"))))
    @test !any(e -> occursin("output", e),
               errors_of(Dict("time_mean" => Dict("path" => "m.nc", "enabled" => false, "period_hours" => 7))))
    # A disabled product needs no path or hours; its keys are still checked.
    @test !any(e -> occursin("output", e), errors_of(Dict("local_time" => Dict("enabled" => false))))
    @test has_error(Dict("local_time" => Dict("enabled" => false, "hour" => [1.0])), "Unknown [output.local_time]")

    mktempdir() do dir
        bin = write_ll_binary(joinpath(dir, "transport_20211201.bin"); nwindow = 4)
        odd = ll_run_cfg(bin, Dict("time_mean" => Dict("path" => joinpath(dir, "m.nc"), "period_hours" => 1.5)))
        @test validate_config(odd) == (true, String[])
        @test_throws r"not a positive multiple of the 1.0 h met window" quiet(() -> run_driven_simulation(odd))
        late = ll_run_cfg(bin, Dict("local_time" => Dict("path" => joinpath(dir, "l.nc"), "hours" => [1.0]));
                          run = Dict("start_window" => 2))
        @test_throws r"must start at 00 UTC" quiet(() -> run_driven_simulation(late))
        @test !isfile(joinpath(dir, "m.nc")) && !isfile(joinpath(dir, "l.nc"))
    end
end

@testset "one capture per instant: a covering snapshot frame is shared" begin
    mktempdir() do dir
        grid = ll_grid(6, 3, 2)
        air = fill(1e9, 6, 3, 2)
        model = (; state = CellState(DryBasis, air; co2 = air .* 4e-4), grid)
        aggregates = aggregates_for(grid, Dict("time_mean" => Dict("path" => joinpath(dir, "m.nc"))))
        covering = capture_snapshot(model; fields = output_field_spec(Dict{String, Any}(COLUMN_MEAN_ONLY)))
        full = capture_snapshot(model)
        lacking = capture_snapshot(model; fields = output_field_spec(Dict{String, Any}(
            "layers" => "none", "column_mean" => false, "column_mass_per_area" => false,
            "column_air_mass_per_area" => false, "air_mass" => false, "air_mass_per_area" => false)))
        # `sample_window_boundary!` returns the frame it sampled.
        @test sample_window_boundary!(aggregates, model, 0.0; shared_frame = covering) === covering
        @test sample_window_boundary!(aggregates, model, 1.0; shared_frame = full) === full
        captured = sample_window_boundary!(aggregates, model, 2.0; shared_frame = lacking)
        @test captured !== lacking
        @test captured isa SelectedSnapshotFrame && haskey(captured.column_tracers, :co2)
        @test sample_window_boundary!(aggregates, model, 3.0) isa SelectedSnapshotFrame
        @test_throws r"expected window boundary 4" sample_window_boundary!(aggregates, model, 5.0)
        close(aggregates)
    end
end

@testset "daily binaries, 3 h windows, Float32, 12 h means, local hour 0" begin
    mktempdir() do dir
        Δ = 3.0
        bins = [write_ll_binary(joinpath(dir, "transport_2021120$(d).bin"); nwindow = 8,
                                window_seconds = 3600Δ) for d in 1:3]
        output = Dict("path" => joinpath(dir, "snap_{date}.nc"), "split" => "daily", "cadence_hours" => Δ,
                      "fields" => COLUMN_MEAN_ONLY,
                      "time_mean" => Dict("path" => joinpath(dir, "mean_{date}.nc"), "period_hours" => 12),
                      "local_time" => Dict("path" => joinpath(dir, "lt_{date}.nc"),
                                           "hours" => [0.0, 1.5, 13.5, 22.5]))
        cfg = ll_run_cfg(bins, output; float_type = "Float32")
        @test validate_config(cfg) == (true, String[])
        @test_logs min_level = Logging.Warn run_driven_simulation(cfg)
        # Day files hold boundaries 0..8 (day 1) and 1..8 (later days, 3 h apart).
        cm = cat([NCDataset(ds -> Float64.(ds["co2_column_mean"][:, :, :]),
                            joinpath(dir, "snap_2021120$(d).nc")) for d in 1:3]...; dims = 3)
        @test size(cm, 3) == 25
        lons = NCDataset(ds -> repeat(Float64.(ds["lon"][:]), 1, size(cm, 2)), joinpath(dir, "snap_20211201.nc"))
        n = 8
        for d in 1:3
            NCDataset(joinpath(dir, "mean_2021120$(d).nc")) do ds
                @test eltype(ds["co2_column_mean"].var) == Float32
                @test ds["time"].var[:] == [24.0(d - 1) + 6, 24.0(d - 1) + 18]
                for r in 1:2
                    p = 2(d - 1) + r - 1
                    ref = sum(k -> cm[:, :, p * 4 + k + 1] * (k == 0 || k == 4 ? 0.5 : 1.0) / 4, 0:4)
                    @test Float64.(ds["co2_column_mean"][:, :, r]) ≈ ref rtol = 1e-6
                end
            end
            NCDataset(joinpath(dir, "lt_2021120$(d).nc")) do ds
                @test ds["time_bnds"].var[:, 1] == [24.0(d - 1), 24.0d]
                @test ds["local_time"][:] == [0.0, 1.5, 13.5, 22.5]
                for (j, T) in enumerate(ds["local_time"][:])
                    tc = mod.(T .- lons ./ 15, 24)
                    ref = [begin
                               k0 = min(floor(Int, tc[c] / Δ), n - 1)
                               f = tc[c] / Δ - k0
                               (1 - f) * cm[c, (d - 1) * n + k0 + 1] + f * cm[c, (d - 1) * n + k0 + 2]
                           end for c in CartesianIndices(tc)]
                    @test Float64.(ds["co2_column_mean"][:, :, j, 1]) ≈ ref rtol = 1e-6
                end
            end
        end
    end
end

@testset "time mean of a run starting at start_window = 3" begin
    mktempdir() do dir
        bin = write_ll_binary(joinpath(dir, "transport_20211201.bin"); nwindow = 30)
        cfg = ll_run_cfg(bin, Dict("path" => joinpath(dir, "snap.nc"), "cadence_hours" => 1.0,
                                   "fields" => COLUMN_MEAN_ONLY,
                                   "time_mean" => Dict("path" => joinpath(dir, "mean.nc")));
                         run = Dict("start_window" => 3))
        @test validate_config(cfg) == (true, String[])
        @test_logs (:warn, r"incomplete \(4 of 24 windows\)") match_mode = :any run_driven_simulation(cfg)
        cm = NCDataset(ds -> ds["co2_column_mean"][:, :, :], joinpath(dir, "snap.nc"))
        @test size(cm, 3) == 29
        NCDataset(joinpath(dir, "mean.nc")) do ds
            @test ds["time"].attrib["units"] == "hours since 2021-12-01 02:00:00"
            @test ds["time_bnds"].var[:, 1] == [0.0, 24.0]
            ref = sum(h -> cm[:, :, h + 1] * (h == 0 || h == 24 ? 1 / 48 : 1 / 24), 0:24)
            @test ds["co2_column_mean"][:, :, 1] ≈ ref rtol = 1e-13
        end
    end
end
