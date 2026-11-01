# Run-config checks before any binary is opened: unknown keys (errors with
# "did you mean" suggestions) and unread keys (warnings), physics/run/output
# settings parsed by `validate_config`, and the snapshot-schedule checks at setup.
using Test
using AtmosTransport
using Logging
using Dates: Date

const CC = AtmosTransport.ConfigChecks
const Runner = AtmosTransport.Models.DrivenRunner

# Every key finding (errors and warnings) of a run config.
key_messages(cfg) = (f = Runner._config_key_findings(cfg); vcat(f.errors, f.warnings))

@testset "ConfigChecks helpers" begin
    @test CC.key_suggestion("diffussion", ("advection", "diffusion")) == "diffusion"
    @test CC.key_suggestion("kz_max", ("kind", "Kz_max")) == "Kz_max"
    @test CC.key_suggestion("order", ("scheme", "ppm_order")) == "ppm_order"
    @test CC.key_suggestion("zzz", ("scheme", "ppm_order")) === nothing
    @test CC.unknown_key_messages(Dict("kindd" => 1, "kind" => 2), ("kind",)) ==
          ["kindd (did you mean `kind`?)"]
    err = try
        CC.check_known_keys(Dict("pth" => 1), ("path", "mode"), "[x]"); nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Unknown [x] option(s): pth (did you mean `path`?). Supported: path, mode.",
                   err.msg)
    @test CC.config_bool(true, "a") === true
    @test_throws ArgumentError CC.config_bool("true", "a")
    @test CC.config_bool(Dict("k" => false), "k", true, "a") === false
    @test CC.config_float(2, "a") === 2.0
    @test_throws ArgumentError CC.config_float(true, "a")
    @test CC.parse_choice("Per-Day", (daily = :d, var"per-day" = :d, single = :s), "a") === :d
    @test_throws ArgumentError CC.parse_choice("weekly", (daily = :d, single = :s), "a")
    @test CC.config_int(3, "a"; min = 1) === 3
    for bad in (true, 3.0, big(typemax(Int)) + 1)
        @test_throws ArgumentError CC.config_int(bad, "a")
    end
    @test_throws ArgumentError CC.config_int(0, "a"; min = 1)
    @test CC.config_date("2024-02-29", "a") == Date(2024, 2, 29)
    @test CC.config_date(Date(2024, 2, 29), "a") == Date(2024, 2, 29)
    @test_throws ArgumentError CC.config_date("2023-02-29", "a")
    @test CC.parse_choice(:dry, (dry = 1,), "a"; allow_symbol = true) === 1
    @test_throws ArgumentError CC.parse_choice(:dry, (dry = 1,), "a")
    @test_throws ArgumentError CC.parse_choice("DRY", (dry = 1,), "a"; case_sensitive = true)
    withenv("ATMOSTR_F1_TEST_FLAG" => "TRUE") do
        @test CC.env_flag("ATMOSTR_F1_TEST_FLAG")
        @test !CC.env_flag("ATMOSTR_F1_TEST_FLAG"; true_values = ("1",), case_sensitive = true)
    end
    withenv("ATMOSTR_F1_TEST_FLAG" => nothing) do
        @test !CC.env_flag("ATMOSTR_F1_TEST_FLAG")
    end
end

@testset "float_type: one strict parser on the runtime and both preprocessing paths" begin
    numerics(raw) = Dict{String, Any}("numerics" => Dict{String, Any}("float_type" => raw))
    resolvers = ((Runner._cfg_float_type, Float64),
                 (AtmosTransport.Preprocessing.resolve_output_float_type, Float32),
                 (AtmosTransport.Preprocessing._resolve_float_type, Float64))
    for (resolve, default) in resolvers
        @test resolve(Dict{String, Any}("numerics" => Dict{String, Any}())) === default
        @test resolve(numerics("Float32")) === Float32
        @test resolve(numerics("Float64")) === Float64
        @test resolve(numerics("float64")) === Float64
        for bad in ("Float16", "Float32 ", "double", "F64", 64)
            err = try resolve(numerics(bad)); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("\"Float32\" or \"Float64\"", err.msg)
        end
    end
end

@testset "unknown and unread keys" begin
    has(ws, parts...) = any(w -> all(p -> occursin(p, w), parts), ws)
    @test isempty(key_messages(Dict(
        "input" => Dict("folder" => "/x", "start_date" => "2021-12-01", "end_date" => "2021-12-02"),
        "run" => Dict("prefetch_windows" => false),
        "advection" => Dict("scheme" => "linrood", "ppm_order" => 7),
        "diffusion" => Dict("kind" => "constant", "value" => 2.0),
        "tracers" => Dict("co2" => Dict("init" => Dict("kind" => "uniform", "background" => 4e-4))),
        "output" => Dict("path" => "/o.nc", "cadence_hours" => 3, "start_hour" => 3))))

    w = key_messages(Dict(
        "diffussion" => Dict("kind" => "constant"),
        "input" => Dict("binary_paths" => ["a.bin"], "end_date" => "2021-12-02"),
        "run" => Dict("tracer_name" => "co2", "Hp" => 3),
        "advection" => Dict("scheme" => "upwind", "order" => 7, "ppm_order" => 7),
        "diffusion" => Dict("kind" => "tm5_dkg", "value" => 1.0, "type" => "pbl"),
        "convection" => Dict("kind" => "tm5", "clamp" => true),
        "chemistry" => Dict("kind" => "none", "half_lives_seconds" => Dict("rn222" => 1.0)),
        "init" => Dict("kind" => "uniform"),
        "tracers" => Dict(
            "co2" => Dict("background" => 1.0,
                          "init" => Dict("kind" => "uniform", "lon0_deg" => 3.0),
                          "surface_flux" => Dict("kind" => "lmdz_co2", "month" => 2,
                                                 "files" => ["a.nc"], "scal" => 2.0))),
        "output" => Dict("path" => "/o.nc", "snapshot_file" => "/p.nc", "hours" => [3],
                         "stop_hour" => 9, "format" => "binary_mmap", "deflate_level" => 1)))
    @test has(w, "diffussion (did you mean `diffusion`?)")
    @test has(w, "[input]", "end_date", "with `binary_paths`")
    @test has(w, "[run]", "tracer_name", "because [tracers]")
    @test has(w, "[advection]", "order (did you mean `ppm_order`?)")
    @test has(w, "[advection]", "ppm_order", "scheme = \"upwind\"")
    @test has(w, "[diffusion]", "value", "unless kind = \"constant\"")
    @test has(w, "[diffusion]", "type", "legacy")
    @test has(w, "[convection]", "clamp", "unless kind = \"cmfmc\"")
    @test has(w, "[chemistry]", "half_lives_seconds", "kind = \"none\"")
    @test has(w, "top-level [init] is ignored")
    @test has(w, "[tracers.co2]", "background", "[tracers.co2.init] exists")
    @test has(w, "[tracers.co2.init]", "lon0_deg", "kind = \"uniform\"")
    @test has(w, "[tracers.co2.surface_flux]", "scal (did you mean `scale`?)")
    @test has(w, "[tracers.co2.surface_flux]", "files", "time_varying = true")
    @test has(w, "[output]", "snapshot_file", "because `path` is set")
    @test has(w, "[output]", "stop_hour", "without an interval key")
    @test has(w, "[output]", "deflate_level", "binary_mmap")
    # Every accepted binary alias must warn about settings its writer ignores.
    for (alias, format) in pairs(AtmosTransport.Output.OUTPUT_FORMATS)
        format isa AtmosTransport.Output.BinaryMmapOutput || continue
        w = key_messages(Dict("output" => Dict("format" => uppercase(String(alias)),
                                              "deflate_level" => 1)))
        @test has(w, "[output]", "deflate_level", "binary_mmap")
    end
    fields = Dict{String, Any}(k => false for k in AtmosTransport.Output.OUTPUT_COLUMN_MASS_KEYS)
    merge!(fields, Dict(k => Dict("CO2" => copy(fields))
                       for k in AtmosTransport.Output.OUTPUT_PER_TRACER_KEYS))
    w = key_messages(Dict("output" => Dict("fields" => fields)))
    @test has(w, "[output.fields]", "tracer_fields", "because `per_tracer` is set")
    @test has(w, "[output.fields]", "column_mass", "because `column_mass_per_area` is set")
    @test has(w, "[output.fields.per_tracer.CO2]", "column_mass", "because `column_mass_per_area` is set")
    # Unknown keys are errors; known keys a setting leaves unread are warnings.
    f = Runner._config_key_findings(Dict(
        "diffussion" => Dict("kind" => "constant"),
        "advection" => Dict("scheme" => "upwind", "order" => 7, "ppm_order" => 7)))
    @test has(f.errors, "diffussion") && has(f.errors, "order (did you mean `ppm_order`?)")
    @test !has(f.errors, "ppm_order", "scheme") && has(f.warnings, "ppm_order", "scheme = \"upwind\"")

    w = key_messages(Dict(
        "tracers" => Dict("co2" => Dict("init" => Dict("kind" => "uniform"))),
        "output" => Dict("fields" => Dict("per_tracer" => Dict("c02" => Dict("layers" => "none",
                                                                             "colum_mean" => true))))))
    @test has(w, "[output.fields.per_tracer.c02]", "not a tracer")
    @test has(w, "[output.fields.per_tracer.c02]", "colum_mean (did you mean `column_mean`?)")
    # Without [tracers] the run carries [run].tracer_name.
    w = key_messages(Dict("run" => Dict("tracer_name" => "CO2"),
        "output" => Dict("fields" => Dict("per_tracer" => Dict("C02" => Dict("layers" => "none"))))))
    @test has(w, "[output.fields.per_tracer.C02]", "not a tracer")
end

@testset "validate_config parses physics, run and output settings" begin
    mktempdir() do dir
        bin = joinpath(dir, "day.bin")
        touch(bin)
        base() = Dict{String, Any}(
            "input" => Dict{String, Any}("binary_paths" => [bin]),
            "tracers" => Dict{String, Any}("co2" => Dict("init" => Dict("kind" => "uniform"))))
        errors_of(cfg) = with_logger(NullLogger()) do
            last(validate_config(cfg))
        end
        @test isempty(errors_of(base()))

        bad(f) = (cfg = base(); f(cfg); errors_of(cfg))
        @test any(contains("ppm_order"),
                  bad(c -> c["advection"] = Dict("scheme" => "ppm", "ppm_order" => 7)))
        @test any(contains("prognostic_slopes"),
                  bad(c -> c["advection"] = Dict("scheme" => "slopes",
                                                  "prognostic_slopes" => true)))
        @test any(contains("must be 5 or 7"),
                  bad(c -> c["advection"] = Dict("scheme" => "linrood", "ppm_order" => 3)))
        @test any(contains("moved to the [advection] table"),
                  bad(c -> c["run"] = Dict("scheme" => "linrood")))
        @test any(contains("moved to the [advection] table"),
                  bad(c -> (c["advection"] = Dict("scheme" => "ppm");
                            c["run"] = Dict("vertical" => "fv3_kord8"))))
        @test any(contains("[diffusion].kind must be one of"), bad(c -> c["diffusion"] = Dict("kind" => "pbl")))
        @test any(contains("names no tracer"),
                  bad(c -> c["chemistry"] = Dict("kind" => "decay",
                                                 "half_lives_seconds" => Dict("rn222" => 3.3e5))))
        @test any(contains("air_mass_reset_mode"),
                  bad(c -> c["run"] = Dict("air_mass_reset_mode" => "always")))
        @test any(contains("reset_air_mass_each_window"),
                  bad(c -> c["run"] = Dict("reset_air_mass_each_window" => true)))
        @test any(contains("physics_cadence"), bad(c -> c["run"] = Dict("physics_cadence" => "step")))
        @test any(contains("[run].prefetch_windows"), bad(c -> c["run"] = Dict("prefetch_windows" => "no")))
        @test isempty(bad(c -> c["run"] = Dict("prefetch_windows" => false)))
        @test any(contains("check_binary_cfl"),
                  bad(c -> c["advection"] = Dict("scheme" => "linrood", "check_binary_cfl" => true)))
        @test any(contains("belongs in [advection]"), bad(c -> c["run"] = Dict("check_binary_cfl" => true)))
        @test isempty(bad(c -> c["advection"] = Dict("scheme" => "ppm", "check_binary_cfl" => true)))
        varying = Dict{String, Any}("co2" => Dict{String, Any}(
            "init" => Dict("kind" => "uniform"),
            "surface_flux" => Dict{String, Any}("kind" => "lmdz_co2", "time_varying" => true)))
        @test any(contains("needs [input].start_date"), bad(c -> c["tracers"] = varying))
        @test isempty(bad(c -> (c["tracers"] = varying; c["input"]["start_date"] = "2021-12-01")))
        @test any(contains("[logging] must be a TOML table"), bad(c -> c["logging"] = "quiet"))
        @test any(contains("[checks] must be a TOML table"), bad(c -> c["checks"] = true))
        @test any(contains("[logging].level"), bad(c -> c["logging"] = Dict("level" => "loud")))
        @test any(contains("no snapshot times"), bad(c -> c["output"] = Dict("path" => "/o.nc")))
        @test any(contains("no `path`"), bad(c -> c["output"] = Dict("hours" => [6, 12])))
        @test isempty(bad(c -> c["output"] = Dict("path" => "/o.nc", "enabled" => false)))
        @test isempty(bad(c -> c["output"] = Dict("snapshot_file" => "/o.nc", "snapshot_hours" => [])))
        @test isempty(bad(c -> c["output"] = Dict("path" => "/o.nc", "cadence_hours" => 6)))

        # Unknown keys are errors; unread keys are warnings.
        cfg = base(); cfg["diffussion"] = Dict("kind" => "constant")
        ok, errors = with_logger(() -> validate_config(cfg), NullLogger())
        @test !ok && any(contains("diffussion (did you mean `diffusion`?)"), errors)
        cfg = base(); cfg["diffusion"] = Dict("kind" => "none", "value" => 1.0)
        logs, (ok, errors) = Test.collect_test_logs() do
            validate_config(cfg)
        end
        @test ok && isempty(errors)
        @test any(l -> l.level == Logging.Warn && occursin("value", string(l.message)), logs)
    end
end

@testset "snapshot schedule at setup" begin
    spec(cfg) = AtmosTransport.Output.runtime_output_spec(cfg, Float32; default_cap_hours = 48)
    hourly = Runner._window_end_hours([(3600.0, 24), (3600.0, 24)])
    threehourly = Runner._window_end_hours([(3 * 3600.0, 8), (3 * 3600.0, 8)])
    @test hourly == collect(1.0:48.0)
    @test Runner._window_end_hours([(3600.0, 24)]; start_window = 3, stop_window_override = 5) ==
          [1.0, 2.0, 3.0]
    @test Runner._check_snapshot_schedule(spec(Dict("path" => "/o.nc", "hours" => [0, 3, 6])),
                                          hourly) === nothing
    @test_throws ArgumentError Runner._check_snapshot_schedule(
        spec(Dict("path" => "/o.nc", "hours" => [1, 2])), threehourly)
    @test_throws ArgumentError Runner._check_snapshot_schedule(
        spec(Dict("path" => "/o.nc", "cadence_hours" => 1.5)), hourly)
    @test Runner._check_snapshot_schedule(spec(Dict("path" => "/o.nc", "cadence_hours" => 6)),
                                          threehourly) === nothing
    @test_logs (:warn, r"after the end of the run") Runner._check_snapshot_schedule(
        spec(Dict("path" => "/o.nc", "hours" => [24, 72])), hourly)
    # Binaries of different window lengths: hour 25 ends a window of the second.
    mixed = Runner._window_end_hours([(3 * 3600.0, 8), (3600.0, 24)])
    @test Runner._check_snapshot_schedule(spec(Dict("path" => "/o.nc", "hours" => [3, 25])),
                                          mixed) === nothing
    @test_throws ArgumentError Runner._check_snapshot_schedule(
        spec(Dict("path" => "/o.nc", "hours" => [1])), mixed)
    # A repeated hour would block every later snapshot.
    @test_throws ArgumentError Runner._check_snapshot_schedule(
        spec(Dict("path" => "/o.nc", "hours" => [0, 0, 3])), hourly)
    @test_throws ArgumentError Runner._check_snapshot_schedule(
        spec(Dict("path" => "/o.nc", "hours" => [3.0, 3.0000005])), hourly)
    # Disabled output is not checked.
    @test Runner._check_snapshot_schedule(spec(Dict("path" => "/o.nc", "hours" => [1.5],
                                                    "enabled" => false)), hourly) === nothing
    # Matching: within half a window and at most half an hour of a window end.
    @test Runner._snapshot_due(3.0, 3.0, 1.0) && !Runner._snapshot_due(2.0, 3.0, 1.0)
    @test Runner._snapshot_due(10 / 60, 10 / 60, 10 / 60)
    @test !Runner._snapshot_due(10 / 60, 20 / 60, 10 / 60)   # 10-minute windows
    @test !Runner._snapshot_due(0.0, 10 / 60, 10 / 60)
    @test Runner._check_window_range([(3600.0, 4), (1800.0, 6)];
                                     stop_window_override = 4) === nothing
    @test_throws ArgumentError Runner._check_window_range(
        [(3600.0, 4), (1800.0, 3)]; stop_window_override = 4)
    @test_throws ArgumentError Runner._check_window_range(
        [(3600.0, 2)]; start_window = 3)
end

@testset "preflight details" begin
    mktempdir() do dir
        bin = joinpath(dir, "day.bin"); touch(bin)
        errors_of(cfg) = with_logger(NullLogger()) do
            last(validate_config(cfg))
        end
        input = Dict{String, Any}("binary_paths" => [bin])
        # An omitted stop_hour is not capped before the binaries are known.
        @test isempty(errors_of(Dict{String, Any}("input" => input,
            "output" => Dict("path" => "/o.nc", "cadence_hours" => 24, "start_hour" => 9000))))
        # Without [tracers] the run carries the legacy [run].tracer_name tracer.
        legacy(name) = Dict{String, Any}("input" => input, "init" => Dict("kind" => "uniform"),
            "run" => Dict("tracer_name" => "rn222"),
            "chemistry" => Dict("kind" => "decay", "half_lives_seconds" => Dict(name => 3.3e5)))
        @test isempty(errors_of(legacy("rn222")))
        @test any(contains("names no tracer"), errors_of(legacy("co2")))
        # Types of the binary expectations, the legacy [init] table, regridding.
        @test any(contains("expected_nlevel must be a positive integer"),
                  errors_of(Dict{String, Any}("input" => merge(input, Dict("expected_nlevel" => true)))))
        @test any(contains("required_preprocessor_contract must be a string"),
                  errors_of(Dict{String, Any}("input" => merge(input, Dict("required_preprocessor_contract" => 1)))))
        @test any(contains("require_adaptive_substeps"),
                  errors_of(Dict{String, Any}("input" => merge(input, Dict("require_adaptive_substeps" => "yes")))))
        @test any(contains("[init] must be a TOML table"),
                  errors_of(Dict{String, Any}("input" => input, "init" => 4e-4)))
        # Surface-flux kinds must be known sources, nested or flat.
        @test any(e -> contains(e, "is not a known source") && contains(e, "did you mean `gridfed_fossil_co2`"),
                  errors_of(Dict{String, Any}("input" => input, "tracers" => Dict("co2" => Dict(
                      "init" => Dict("kind" => "uniform"),
                      "surface_flux" => Dict("kind" => "gridfed", "file" => "f.nc", "variable" => "TOTAL"))))))
        @test any(contains("is not a known source"),
                  errors_of(Dict{String, Any}("input" => input, "tracers" => Dict("co2" => Dict(
                      "surface_flux_kind" => "eccodarwin_ocean_co2", "surface_flux_file" => "f.nc")))))
        @test any(contains("surface_flux.regridding must be one of"),
                  errors_of(Dict{String, Any}("input" => input, "tracers" => Dict("co2" => Dict(
                      "init" => Dict("kind" => "uniform"),
                      "surface_flux" => Dict("kind" => "file", "regridding" => "conservativ"))))))
        # The time-coverage opt-in of time-varying sources must be a Bool (#59).
        coverage(value) = errors_of(Dict{String, Any}("input" => input, "tracers" => Dict("co2" => Dict(
            "init" => Dict("kind" => "uniform"),
            "surface_flux" => Dict("kind" => "lmdz_co2", "time_varying" => true,
                                   "hold_outside_coverage" => value)))))
        @test !any(contains("hold_outside_coverage"), coverage(true))
        @test any(contains("hold_outside_coverage"), coverage("yes"))
    end
    w = key_messages(Dict("tracers" => Dict("co2" => Dict(
        "init" => Dict("kind" => "uniform"),
        "surface_flux" => Dict("file" => "f.nc", "variable" => "FLUX", "scale" => 2.0)))))
    @test any(x -> occursin("[tracers.co2.surface_flux]", x) && occursin("no flux is emitted", x), w)
    # Flat tracer keys go through the same checks.
    w = key_messages(Dict("tracers" => Dict("co2" => Dict(
        "kind" => "uniform", "lon0_deg" => 3.0, "surface_flux_file" => "f.nc"))))
    @test any(x -> occursin("[tracers.co2]", x) && occursin("lon0_deg", x) &&
                   occursin("kind = \"uniform\"", x), w)
    @test any(x -> occursin("[tracers.co2] (flat surface_flux_* keys)", x) &&
                   occursin("no flux is emitted", x), w)

    @test isempty(key_messages(Dict("tracers" => Dict("co2" => Dict(
        "kind" => "gaussian_blob", "lon0_deg" => 3.0, "surface_flux_kind" => "edgar_sf6")))))
    # Keys a kind or mode leaves unread, and shadowed aliases.
    w = key_messages(Dict("tracers" => Dict(
        "a" => Dict("init" => Dict("kind" => "file", "file" => "f.nc", "variable" => "v",
                                   "background" => 4e-4)),
        "b" => Dict("init" => Dict("kind" => "latitude_step", "south" => 1.0, "south_value" => 2.0)),
        "c" => Dict("init" => Dict("kind" => "uniform"),
                    "surface_flux" => Dict("kind" => "lmdz_co2", "time_varying" => true,
                                           "files" => ["a.nc"], "file_pattern" => "x_{YYYYMM}.nc",
                                           "time_index" => 2)),
        "d" => Dict("init" => Dict("kind" => "uniform"),
                    "surface_flux" => Dict("kind" => "lmdz_co2", "time_index" => 2)))))
    @test any(x -> occursin("[tracers.a.init]", x) && occursin("background", x), w)
    @test any(x -> occursin("[tracers.b.init]", x) && occursin("south is ignored because `south_value`", x), w)
    @test any(x -> occursin("[tracers.c.surface_flux]", x) && occursin("file_pattern is ignored because `files`", x), w)
    @test any(x -> occursin("[tracers.c.surface_flux]", x) && occursin("time_index", x) &&
                   occursin("time_varying = true", x), w)
    @test any(x -> occursin("[tracers.d.surface_flux]", x) && occursin("time mean", x), w)
    # The run length counts every binary's windows.
    @test Runner._layout_run_hours([(3600.0, 12), (10800.0, 8)]) == 36.0
    @test Runner._layout_run_hours([(3600.0, 24), (3600.0, 24)]; stop_window_override = 6) == 12.0
    @test Runner._layout_run_hours([(3600.0, 24)]; start_window = 3) == 22.0
end
