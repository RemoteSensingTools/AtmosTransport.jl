using Test
using Dates
using JSON3
using TOML

# The lab lane and the shared naming/preset logic use Base, TOML, JSON3, and
# Dates only, so they load by path without the benchmark environment.
include(joinpath(@__DIR__, "..", "..", "benchmarking", "lab", "LabLane.jl"))
using .AtmosTransportLabLane
const LL = AtmosTransportLabLane

# The harness's Base + JSON3 parts (dashboard records, synthetic fields),
# loaded by path with the naming and case-option files they build on.
module BenchmarkRecords
const SRC = joinpath(@__DIR__, "..", "..", "benchmarking", "src")
include(joinpath(SRC, "labels.jl"))
include(joinpath(SRC, "case_options.jl"))
include(joinpath(SRC, "records.jl"))
include(joinpath(SRC, "synthetic_fields.jl"))
include(joinpath(SRC, "memory_model.jl"))
end
const BR = BenchmarkRecords

# Stands in for CUDA.jl's out-of-memory exception, recognised by type name.
struct OutOfGPUMemoryError <: Exception end

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const COMMIT = "0123456789abcdef0123456789abcdef01234567"
const OTHER_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
const PINNED_JULIA = "1.12.6"                 # machines.toml julia_version
const MANIFEST_SHA = repeat("ab", 32)
const OTHER_MANIFEST_SHA = repeat("cd", 32)

machines_from(text) = LL.parse_machines(TOML.parse(text); source = "test.toml")

# Lab executor that records every step and throws for steps whose description
# matches `fail`, standing in for commands that fail on a lab machine.
struct ScriptedExecutor <: LL.Executor
    log::Vector{String}
    fail::Regex
end
function LL.execute!(ex::ScriptedExecutor, step::LL.LabStep)
    push!(ex.log, step.description)
    occursin(ex.fail, step.description) && error("scripted failure: $(step.description)")
    return nothing
end
machine_named(ms, name) = only(filter(m -> m.name == name, ms))

const MINIMAL_MACHINE = """
[machines.box]
access = "local"
julia = "~/.juliaup/bin/julia"
julia_version = "1.12.6"
backend = "cuda"
float_types = ["Float32"]
code = { delivery = "shared", root = "~/atmostr_bench" }
"""

@testset "benchmark lab lane" begin
    @testset "machines.toml parses into typed machines" begin
        ms = LL.load_machines()
        @test sort([m.name for m in ms]) == ["curry", "mac", "wurst"]
        wurst, curry, mac = (machine_named(ms, n) for n in ("wurst", "curry", "mac"))

        @test wurst.access isa LL.LocalAccess
        @test wurst.code isa LL.SharedArchive
        @test wurst.code.root == expanduser("~/atmostr_bench")
        @test wurst.backend === :cuda && wurst.float_types == ["Float32"]
        # One deliberately updated Julia pin for every lab machine.
        @test all(m -> m.julia_version == VersionNumber(PINNED_JULIA), ms)
        @test ("CUDA_VISIBLE_DEVICES" => "0") in wurst.env

        @test curry.access == LL.SSHAccess("curry")
        @test curry.code isa LL.SharedArchive
        @test curry.float_types == ["Float32", "Float64"]
        @test curry.heap_size_hint !== nothing

        @test mac.access isa LL.SSHAccess
        @test mac.code == LL.CopiedArchive("~/atmostr_bench")
        @test mac.backend === :metal && mac.float_types == ["Float32"]
        @test mac.caffeinate
        # Everything the Mac writes stays under ~/atmostr_bench: code, io
        # temporaries, and the Julia depot (the trailing ":" adds only the
        # bundled stdlib depots, not ~/.julia).
        @test startswith(mac.tmpdir, "~/atmostr_bench/")
        depot = only(v for (k, v) in mac.env if k == "JULIA_DEPOT_PATH")
        @test all(p -> isempty(p) || startswith(p, "~/atmostr_bench/"), split(depot, ':'))

        @test [m.name for m in LL.select_machines(ms, ["mac", "wurst"])] == ["mac", "wurst"]
        @test_throws ArgumentError LL.select_machines(ms, ["nonexistent"])
    end

    @testset "machine configuration errors" begin
        @test length(machines_from(MINIMAL_MACHINE)) == 1
        bad(text) = (@test_throws LL.MachineConfigError machines_from(text))
        message(text) = try
            machines_from(text)
            ""
        catch err
            sprint(showerror, err)
        end

        typo = replace(MINIMAL_MACHINE, "float_types" => "flaot_types")
        bad(typo)
        @test occursin("flaot_types", message(typo))
        @test occursin("test.toml", message(typo))

        bad(MINIMAL_MACHINE * "\n[extra]\nkey = 1\n")
        bad(replace(MINIMAL_MACHINE, "root = \"~/atmostr_bench\"" =>
                                     "root = \"~/atmostr_bench\", mode = \"fast\""))
        bad(MINIMAL_MACHINE * "env = { TMPDIR = \"/scratch\" }\n")
        bad(MINIMAL_MACHINE * "env = { CUDA_VISIBLE_DEVICES = 0 }\n")
        bad(replace(MINIMAL_MACHINE, "\"~/atmostr_bench\"" => "\"~/scratch\""))
        bad(replace(MINIMAL_MACHINE, "\"~/atmostr_bench\"" => "\"~/atmostr_x/../etc\""))
        bad(replace(MINIMAL_MACHINE, "delivery = \"shared\"" => "delivery = \"copy\""))
        bad(replace(MINIMAL_MACHINE, "access = \"local\"" => "access = \"ssh\""))
        bad(replace(MINIMAL_MACHINE, "[\"Float32\"]" => "[\"Float16\"]"))
        bad(replace(MINIMAL_MACHINE, "[\"Float32\"]" => "[32]"))
        bad(replace(replace(MINIMAL_MACHINE, "\"cuda\"" => "\"metal\""),
                    "[\"Float32\"]" => "[\"Float32\", \"Float64\"]"))
        bad(MINIMAL_MACHINE * "device_label = \"L40S/48GB\"\n")
        bad(MINIMAL_MACHINE * "heap_size_hint = \"lots\"\n")
        bad(replace(MINIMAL_MACHINE, "julia_version = \"1.12.6\"\n" => ""))
        bad(replace(MINIMAL_MACHINE, "\"1.12.6\"" => "\"1.12\""))
        bad(replace(MINIMAL_MACHINE, "\"1.12.6\"" => "\"1.12.6-rc1\""))
        @test occursin("older than 1.11", message(replace(MINIMAL_MACHINE, "\"1.12.6\"" => "\"1.10.9\"")))
        bad("[machines]\n")
    end

    @testset "device label normalisation" begin
        for (name, label) in ("NVIDIA L40S" => "L40S",
                              "NVIDIA A100-PCIE-40GB" => "A100",
                              "NVIDIA A100-SXM4-80GB" => "A100",
                              "NVIDIA A100 80GB PCIe" => "A100",
                              "NVIDIA H100 80GB HBM3" => "H100",
                              "Tesla V100-PCIE-32GB" => "V100",
                              "NVIDIA GeForce RTX 4090" => "RTX 4090",
                              "NVIDIA RTX A6000" => "RTX A6000",
                              "Apple M5 Pro" => "M5 Pro",
                              "  Apple   M2  Max " => "M2 Max",
                              "Vendor X/Y" => "Vendor X-Y")
            @test normalize_device_label(name) == label
        end
        @test LL.backend_series_label("CPU-AMD", nothing) == "CPU-AMD"
        @test LL.backend_series_label("CPU-Intel", "ignored") == "CPU-Intel"
        @test LL.backend_series_label("GPU-CUDA", "L40S") == "GPU-CUDA L40S"
        @test LL.backend_series_label("GPU-Metal", "M5 Pro") == "GPU-Metal M5 Pro"
        @test_throws ArgumentError LL.backend_series_label("GPU-CUDA", nothing)
        @test_throws ArgumentError LL.validate_device_label("A100/80GB")
        @test_throws ArgumentError LL.validate_device_label(" ")
    end

    @testset "series names keep CPU history" begin
        cpu = LL.case_series_name(; group = "CPU Synthetic CS Sweep", grid = "C4", levels = 32,
                                  operator = "advection", tracers = 1, scheme = "ppm",
                                  backend_label = "CPU-AMD", float_type = "Float32")
        @test cpu == "CPU Synthetic CS Sweep / C4 L32 advection 1tr / CPU-AMD / Float32"
        gpu = LL.case_series_name(; group = "Lab Synthetic CS Sweep", grid = "C180", levels = 137,
                                  operator = "full", tracers = 12, scheme = "linrood7",
                                  backend_label = "GPU-CUDA A100", float_type = "Float64")
        @test gpu == "Lab Synthetic CS Sweep / C180 L137 full 12tr linrood7 / GPU-CUDA A100 / Float64"
        @test LL.breakdown_series_name(gpu, "convection") ==
              "Operator Breakdown / Lab Synthetic CS Sweep / C180 L137 full 12tr linrood7 / GPU-CUDA A100 / Float64 / convection"

        # Phase series of two groups and two schemes stay apart.
        names = [LL.breakdown_series_name(
                     LL.case_series_name(; group, grid = "C90", levels = 72, operator = "advection",
                                         tracers = 12, scheme, backend_label = "GPU-CUDA L40S",
                                         float_type = "Float32"), "transport")
                 for group in ("Lab Synthetic CS Sweep", "Lab Synthetic CS Sweep v2")
                 for scheme in ("ppm", "linrood7")]
        @test allunique(names)
        @test count(n -> occursin(" / Lab Synthetic CS Sweep v2 / ", n), names) == 2
        @test count(n -> occursin(" 12tr linrood7 / ", n), names) == 2
        @test all(n -> startswith(n, LL.BREAKDOWN_SERIES_PREFIX * " / "), names)
    end

    @testset "lab preset and case options" begin
        preset = LL.load_preset("lab")
        ms = LL.load_machines()
        # Per float type: 4 operators x 4 grids x 2 levels + (io + adjoint),
        # limited to C180, x 2 grids x 2 levels = 40 cases.
        for (name, count) in ("wurst" => 40, "curry" => 80, "mac" => 40)
            options = LL.machine_case_options(preset, machine_named(ms, name))
            @test LL.expected_case_count(options) == count
        end
        options = LL.machine_case_options(preset, machine_named(ms, "curry"))
        @test options.grids == [90, 180, 360, 720] && options.levels == [72, 137]
        @test options.tracers == [2] && options.group == "Lab Synthetic CS Sweep"
        @test options.max_grid == Dict(:io => 180, :adjoint => 180)
        cases = LL.case_matrix(options)
        @test length(cases) == LL.expected_case_count(options) && allunique(cases)
        # io carries every requested tracer; the adjoint footprint is one tracer.
        @test sort(unique(c.tracers for c in cases if c.operator === :io)) == [2]
        @test all(c -> c.tracers == 1, (c for c in cases if c.operator === :adjoint))
        @test count(c -> c.operator === :adjoint, cases) == 2 * 2 * 2   # float types x grids x levels
        # io and adjoint stop at their C180 limit; the model operators run on every grid.
        grids_of(op) = sort(unique(c.grid for c in cases if c.operator === op))
        @test grids_of(:io) == grids_of(:adjoint) == [90, 180]
        for op in (:advection, :diffusion, :convection, :full)
            @test grids_of(op) == [90, 180, 360, 720]
        end
        # Limits are typed: known operators, the max_grid key, a valid grid.
        limited(limits) = LL.resolve_case_options(Dict{String, Any}("grid" => "C24,C48",
                                                                    "operator" => "advection,io",
                                                                    "limits" => limits), nothing)
        @test [(c.operator, c.grid) for c in LL.case_matrix(limited(Dict("io" => Dict("max_grid" => 24))))] ==
              [(:advection, 24), (:io, 24), (:advection, 48)]
        @test LL.expected_case_count(limited(Dict{String, Any}())) == 4
        @test_throws ArgumentError limited(Dict("chemistry" => Dict("max_grid" => "C24")))
        @test_throws ArgumentError limited(Dict("io" => Dict("max_grids" => "C24")))
        @test_throws ArgumentError limited(Dict("io" => Dict("max_grid" => "C1")))
        @test_throws ArgumentError limited(Dict("io" => "C24"))
        @test_throws ArgumentError limited("C24")

        # Operators that ignore --scheme run once, under the default scheme;
        # single-tracer operators run at one tracer even if 1 is not listed.
        sweep = LL.resolve_case_options(Dict{String, Any}("grid" => "C4", "levels" => "8",
                                                          "tracers" => "4,4",
                                                          "operator" => "advection,diffusion,convection,full,io,adjoint",
                                                          "scheme" => "ppm,linrood7"), nothing)
        cases = LL.case_matrix(sweep)
        @test allunique(cases)
        bykey(op) = sort([(c.tracers, c.scheme) for c in cases if c.operator === op])
        @test bykey(:advection) == bykey(:full) == [(4, :linrood7), (4, :ppm)]
        for op in (:diffusion, :convection, :io)
            @test bykey(op) == [(4, :ppm)]
        end
        @test bykey(:adjoint) == [(1, :ppm)]
        @test LL.expected_case_count(sweep) == 8   # 2 advection + 2 full + 4 single cases
        @test issubset(LL.SINGLE_TRACER_OPERATORS, LL.SUPPORTED_OPERATORS)
        @test issubset(LL.SCHEME_FREE_OPERATORS, LL.SUPPORTED_OPERATORS)

        # The group is a series-name field.
        for bad_group in ("", "CPU/GPU sweep", "Operator Breakdown sweep")
            @test_throws ArgumentError LL.resolve_case_options(Dict{String, Any}("group" => bad_group), nothing)
        end

        defaults = LL.resolve_case_options(Dict{String, Any}(), nothing)
        @test defaults.grids == [24] && defaults.levels == [32] && defaults.convection_depth == 0
        cli = LL.resolve_case_options(Dict{String, Any}("grid" => "C4,c8", "levels" => nothing,
                                                         "float_type" => "f64"), preset)
        @test cli.grids == [4, 8] && cli.levels == [72, 137] && cli.float_types == ["Float64"]
        @test_throws ArgumentError LL.parse_preset(Dict("grids" => ["C90"]))
        @test_throws ArgumentError LL.resolve_case_options(Dict{String, Any}("operator" => "chemistry"), nothing)
        @test_throws ArgumentError LL.load_preset("no_such_preset")

        # Full-column forcing keeps the historical band; capped depths stay in the active levels.
        for Nz in (8, 32, 72)
            @test LL.convection_forcing_band(Nz, 0) == max(2, Nz ÷ 3):max(max(2, Nz ÷ 3), min(Nz - 1, 2Nz ÷ 3))
            @test LL.convection_forcing_band(Nz, 85) == LL.convection_forcing_band(Nz, 0)
        end
        band = LL.convection_forcing_band(137, 85)
        @test first(band) > 137 - 85 && last(band) <= 137
        @test LL.convection_forcing_band(72, 1) == 72:72
        @test LL.effective_convection_depth(85, 72) == 72
        @test LL.effective_convection_depth(85, 137) == 85
        @test LL.effective_convection_depth(0, 137) == 0
    end

    @testset "dashboard records and per-operator denominators" begin
        # A timed repeat is `steps` model steps, one adjoint run over `steps`
        # steps, or one io round trip.
        for op in (:advection, :diffusion, :convection, :full, :adjoint)
            @test BR.timed_steps_per_repeat(op, 5) == 5
            @test BR.timed_steps_per_repeat(String(op), 3) == 3
        end
        @test BR.timed_steps_per_repeat(:io, 5) == 1
        @test BR.timed_steps_per_repeat("io", 5) == 1

        steps, repeats = 5, 4
        # Raw records as the harness writes them: phase totals accumulated over
        # all timed repeats, `steps_per_repeat` from `timed_steps_per_repeat`.
        function raw_record(op, time_per_step, phases; status = "ok")
            name = BR.case_series_name(; group = "G", grid = "C4", levels = 8, operator = String(op),
                                       tracers = 1, scheme = "ppm", backend_label = "CPU-AMD",
                                       float_type = "Float32")
            Dict{String, Any}("name" => name, "status" => status, "operator" => String(op),
                              "device_name" => "cpu", "time_per_step_seconds" => time_per_step,
                              "cell_tracer_updates_per_second" => 1.0e6,
                              "steps_per_repeat" => BR.timed_steps_per_repeat(op, steps),
                              "repeats" => repeats, "phase_times_seconds" => phases,
                              "metadata" => Dict("commit" => COMMIT, "julia_version" => PINNED_JULIA,
                                                 "manifest_sha256" => MANIFEST_SHA))
        end
        # advection: 0.1 s/step over steps*repeats = 20 steps; io: 0.5 s per
        # round trip over 4 round trips; adjoint: 0.2 s/step, one 1.0 s run per
        # repeat over 5 steps.
        raw = [raw_record(:advection, 0.1, Dict("transport" => 0.1 * steps * repeats,
                                                "forcing_refresh" => 1.0)),
               raw_record(:io, 0.5, Dict("io" => 0.5 * repeats)),
               raw_record(:adjoint, 0.2, Dict("adjoint_forward_reverse" => 0.2 * steps * repeats)),
               raw_record(:full, 0.3, Dict("transport" => 1.0); status = "failed")]
        records = BR.dashboard_records(raw)
        byname = Dict(r["name"] => r for r in records)
        @test length(records) == 6
        for r in raw[1:3]
            @test byname[r["name"]]["value"] == r["time_per_step_seconds"]
            @test JSON3.read(byname[r["name"]]["extra"])["commit"] == COMMIT
        end
        # Every published record names its environment.
        for r in records
            extra = JSON3.read(r["extra"])
            @test extra["julia_version"] == PINNED_JULIA && extra["manifest_sha256"] == MANIFEST_SHA
        end
        @test LL.environment_fingerprint(records) == LL.EnvironmentFingerprint(PINNED_JULIA, MANIFEST_SHA)
        phase_value(i, phase) = byname[BR.breakdown_series_name(raw[i]["name"], phase)]["value"]
        @test phase_value(1, "transport") ≈ 0.1
        @test phase_value(2, "io") ≈ 0.5
        @test phase_value(3, "adjoint_forward_reverse") ≈ 0.2
        @test !any(r -> occursin("forcing_refresh", r["name"]), records)
        @test !any(r -> occursin(" full ", r["name"]), records)
        @test all(r -> r["unit"] == "s/step" && r["extra"] isa String, records)
        @test LL.validate_publishable_records(records, COMMIT) === nothing
    end

    @testset "synthetic fields: steady air mass, sane Courant number" begin
        Hp = 3
        for (Nc, Nz) in ((4, 8), (24, 32), (48, 72), (24, 137), (90, 72)), FT in (Float32, Float64)
            N = Nc + 2Hp
            m = ntuple(_ -> zeros(FT, N, N, Nz), 6)
            am = ntuple(_ -> zeros(FT, N + 1, N, Nz), 6)
            bm = ntuple(_ -> zeros(FT, N, N + 1, Nz), 6)
            cm = ntuple(_ -> zeros(FT, N, N, Nz + 1), 6)
            BR.fill_synthetic_mass!(m, Nc, Hp)
            BR.fill_synthetic_fluxes!(am, bm, cm, m, Nc, Hp)
            I = (Hp + 1):(Hp + Nc)
            # A full step's mass change of every cell (x, y, and z faces) is
            # zero to rounding, so the Courant number cannot drift over steps.
            change = maximum(maximum(abs.((am[p][I, I, :] .- am[p][I .+ 1, I, :]) .+
                                          (bm[p][I, I, :] .- bm[p][I, I .+ 1, :]) .+
                                          (cm[p][I, I, 1:Nz] .- cm[p][I, I, 2:(Nz + 1)])) ./
                                     m[p][I, I, :]) for p in 1:6)
            @test change < (FT === Float32 ? 1e-6 : 1e-12)
            # No flux crosses a panel edge, the model top, or the surface; halos stay zero.
            for p in 1:6
                @test all(iszero, am[p][Hp + 1, :, :]) && all(iszero, am[p][Hp + Nc + 1, :, :])
                @test all(iszero, bm[p][:, Hp + 1, :]) && all(iszero, bm[p][:, Hp + Nc + 1, :])
                @test all(iszero, cm[p][:, :, 1]) && all(iszero, cm[p][:, :, Nz + 1])
                @test all(iszero, am[p][1:Hp, :, :]) && all(iszero, bm[p][:, 1:Hp, :])
                @test all(iszero, cm[p][1:Hp, :, :])
            end
            # Every direction moves mass, at a Courant number that needs no subcycling.
            @test maximum(abs, am[1]) > 0 && maximum(abs, bm[1]) > 0 && maximum(abs, cm[1]) > 0
            courant = BR.synthetic_courant(m, am, bm, cm, Nc, Hp)
            @test 0.05 < courant <= BR.MAX_SYNTHETIC_COURANT
        end
        # Degenerate sizes: one level has a gyre but no divergent part or
        # vertical flux (zero scale, never NaN); C2 is the smallest grid.
        function synthetic(FT, Nc, Nz)
            N = Nc + 2Hp
            m = ntuple(_ -> zeros(FT, N, N, Nz), 6)
            am = ntuple(_ -> zeros(FT, N + 1, N, Nz), 6)
            bm = ntuple(_ -> zeros(FT, N, N + 1, Nz), 6)
            cm = ntuple(_ -> zeros(FT, N, N, Nz + 1), 6)
            BR.fill_synthetic_mass!(m, Nc, Hp)
            BR.fill_synthetic_fluxes!(am, bm, cm, m, Nc, Hp)
            return m, am, bm, cm
        end
        for (Nc, Nz) in ((4, 1), (2, 1), (2, 2), (24, 1)), FT in (Float32, Float64)
            m, am, bm, cm = synthetic(FT, Nc, Nz)
            @test all(f -> all(a -> all(isfinite, a), f), (am, bm, cm))
            @test maximum(abs, am[1]) > 0
            Nz == 1 && @test all(c -> all(iszero, c), cm)
            courant = BR.synthetic_courant(m, am, bm, cm, Nc, Hp)
            @test 0 < courant <= BR.MAX_SYNTHETIC_COURANT
        end
        @test_throws ArgumentError synthetic(Float32, 1, 8)
        @test_throws ArgumentError LL.resolve_case_options(Dict{String, Any}("grid" => "C1"), nothing)
        @test LL.resolve_case_options(Dict{String, Any}("grid" => "C2", "levels" => "1"), nothing).grids == [2]

        @test BR.check_synthetic_courant(0.2, "case"; stage = "initially") === nothing
        msg = try
            BR.check_synthetic_courant(0.7, "Lab / C180 L72 advection 1tr / GPU-CUDA L40S / Float32";
                                       stage = "after 2 warmup step(s)")
            ""
        catch err
            sprint(showerror, err)
        end
        @test occursin("C180 L72 advection", msg) && occursin("after 2 warmup step(s)", msg) &&
              occursin("exceeds $(BR.MAX_SYNTHETIC_COURANT)", msg)
    end

    @testset "device memory estimate and skip decision" begin
        # Memory measured on CPU (Base.summarysize of the model after
        # construction and one step, TM5 tile included), Float32, 2 tracers;
        # the estimate must never fall below it and stay within 20 %.
        measured = [(:advection, 90, 72, 155531816), (:advection, 180, 72, 581624216),
                    (:diffusion, 90, 72, 124588504), (:diffusion, 180, 72, 474087304),
                    (:convection, 90, 72, 354403472), (:convection, 180, 72, 1343629472),
                    (:full, 90, 72, 442113936), (:full, 180, 72, 1678227936)]
        for (op, Nc, Nz, bytes) in measured
            estimate = BR.estimate_case_bytes(op, 4, Nc, Nz, 2; steps = 1, tm5_legacy_tile = true)
            @test bytes <= estimate <= 1.2 * bytes
        end
        # Float64, 3 tracers (coefficients are in elements of the float type).
        @test 33543568 <= BR.estimate_case_bytes(:advection, 8, 32, 40, 3; steps = 1, tm5_legacy_tile = false)
        @test 65467280 <= BR.estimate_case_bytes(:full, 8, 32, 40, 3; steps = 1, tm5_legacy_tile = true)
        # The TM5 tile: measured 758344568 B at C180 L72 Float32, capped at about 1 GiB.
        @test abs(BR.tm5_legacy_tile_bytes(4, 180, 72) - 758344568) < 1024
        @test BR.tm5_legacy_tile_bytes(4, 720, 137) <= 2^30
        @test BR.estimate_case_bytes(:full, 4, 720, 137, 2; steps = 5, tm5_legacy_tile = true) -
              BR.estimate_case_bytes(:full, 4, 720, 137, 2; steps = 5, tm5_legacy_tile = false) ==
              BR.tm5_legacy_tile_bytes(4, 720, 137)
        # io: exactly the payload arrays.
        Hp, Nc, Nz, Nt = 3, 24, 32, 2
        N = Nc + 2Hp
        payload = 6 * sizeof(Float32) * ((1 + Nt) * N * N * Nz + 2 * (N + 1) * N * Nz + N * N * (Nz + 1))
        @test BR.io_payload_bytes(4, Nc, Nz, Nt) == payload
        @test BR.estimate_case_bytes(:io, 4, Nc, Nz, Nt; steps = 5, tm5_legacy_tile = false) ==
              payload + BR.MEMORY_FIXED_BYTES
        @test BR.estimate_case_bytes(:adjoint, 8, 180, 137, 1; steps = 5, tm5_legacy_tile = false) >
              BR.estimate_case_bytes(:adjoint, 8, 180, 137, 1; steps = 1, tm5_legacy_tile = false)

        # Skip decision against a device's available memory.
        required(op, fb, Nc, Nz, legacy) =
            BR.required_device_bytes(BR.estimate_case_bytes(op, fb, Nc, Nz, 2; steps = 5, tm5_legacy_tile = legacy))
        a100 = round(Int, 38.9 * 2^30)
        @test BR.memory_skip_reason(required(:full, 4, 720, 137, false), a100) === nothing
        reason = BR.memory_skip_reason(required(:full, 8, 720, 137, true), a100)
        @test reason isa String && occursin("GiB of device memory", reason) && occursin("38.9 GiB available", reason)
        @test BR.memory_skip_reason(required(:advection, 4, 90, 72, false), 2^29) isa String   # fake 0.5 GiB device
        @test BR.memory_skip_reason(10^12, nothing) === nothing      # unknown memory never skips
        @test BR.required_device_bytes(2^30) == ceil(Int, 1.05 * 2^30) + 2^29

        # Out-of-memory errors of Julia, CUDA (by type name), and Metal (by message).
        @test BR.is_out_of_memory(OutOfMemoryError())
        @test BR.is_out_of_memory(OutOfGPUMemoryError())
        @test BR.is_out_of_memory(ErrorException("Metal: failed to allocate a buffer of 8 GiB"))
        @test BR.is_out_of_memory(CompositeException([ErrorException("x"), OutOfMemoryError()]))
        @test !BR.is_out_of_memory(ErrorException("cubed-sphere palindrome subcycling exceeded max_n_sub=4096"))
        @test !BR.is_out_of_memory(ArgumentError("bad"))
    end

    @testset "record validation" begin
        machine = machine_named(LL.load_machines(), "curry")
        group = "Lab Synthetic CS Sweep"
        label = "GPU-CUDA A100"
        function record(ft, op; commit = COMMIT, status = "ok", backend_label = label,
                        julia = PINNED_JULIA, manifest = MANIFEST_SHA)
            name = LL.case_series_name(; group, grid = "C90", levels = 72, operator = op,
                                       tracers = 1, scheme = "ppm", backend_label, float_type = ft)
            Dict{String, Any}("name" => name, "status" => status, "group" => group,
                              "backend" => "cuda", "backend_class" => "GPU-CUDA",
                              "backend_label" => backend_label,
                              "device_name" => "NVIDIA A100-PCIE-40GB", "float_type" => ft,
                              "error" => status == "ok" ? nothing : "CUDA out of memory",
                              "metadata" => Dict("commit" => commit, "julia_version" => julia,
                                                 "manifest_sha256" => manifest))
        end
        dashboard_for(raw; commit = COMMIT) = [Dict{String, Any}(
            "name" => r["name"], "unit" => "s/step", "value" => 0.01,
            "extra" => JSON3.write(Dict("commit" => commit, "device" => r["device_name"],
                                        "julia_version" => r["metadata"]["julia_version"],
                                        "manifest_sha256" => r["metadata"]["manifest_sha256"])))
            for r in raw if r["status"] == "ok"]
        raw = [record(ft, op) for ft in ("Float32", "Float64") for op in ("advection", "full")]
        validate(raw, dashboard; n = length(raw)) =
            LL.validate_machine_results(raw, dashboard; machine, commit = COMMIT,
                                        expected_cases = n, group)
        failure(f) = try
            f()
            ""
        catch err
            err isa LL.LabValidationError || rethrow()
            sprint(showerror, err)
        end

        @test validate(raw, dashboard_for(raw)) == String[]
        # JSON3 objects (Symbol keys), as read back from disk, validate the same way.
        @test validate(JSON3.read(JSON3.write(raw)), JSON3.read(JSON3.write(dashboard_for(raw)))) == String[]

        mismatched = [raw[1:3]; record("Float64", "full"; commit = OTHER_COMMIT)]
        @test occursin("records commit", failure(() -> validate(mismatched, dashboard_for(raw))))

        failed = [raw[1:3]; record("Float64", "full"; status = "failed")]
        msg = failure(() -> validate(failed, dashboard_for(failed)))
        @test occursin("failed case", msg) && occursin("out of memory", msg)

        @test occursin("expected 5 cases", failure(() -> validate(raw, dashboard_for(raw); n = 5)))
        @test occursin("extra.commit", failure(() -> validate(raw, dashboard_for(raw; commit = OTHER_COMMIT))))
        @test occursin("missing from the dashboard", failure(() -> validate(raw, dashboard_for(raw)[1:2])))

        nodevice = [record(ft, op; backend_label = "GPU-CUDA") for ft in ("Float32", "Float64")
                    for op in ("advection", "full")]
        @test occursin("does not carry a device", failure(() -> validate(nodevice, dashboard_for(nodevice))))

        f16 = [raw[1:3]; record("Float16", "full")]
        @test occursin("float type", failure(() -> validate(f16, dashboard_for(f16))))

        # A case that did not fit in device memory is recorded, listed, and never published.
        function skipped_record(ft, op; reason = "needs an estimated 54.5 GiB of device memory, 38.9 GiB available")
            r = record(ft, op; status = "skipped_memory")
            delete!(r, "error")
            r["skip_reason"] = reason
            r["required_device_gib"] = 54.5
            r["available_device_gib"] = 38.9
            return r
        end
        with_skip = [raw[1:3]; skipped_record("Float64", "full")]
        @test validate(with_skip, dashboard_for(with_skip)) == [with_skip[4]["name"]]
        no_reason = [raw[1:3]; skipped_record("Float64", "full"; reason = "")]
        @test occursin("without a skip_reason", failure(() -> validate(no_reason, dashboard_for(no_reason))))
        published_skip = dashboard_for(raw)       # contains the name of the skipped case
        @test occursin("skipped case", failure(() -> validate(with_skip, published_skip)))
        @test occursin("expected 5 cases", failure(() -> validate(with_skip, dashboard_for(with_skip); n = 5)))

        # Environment fingerprint: the pinned Julia, a Manifest hash, one per machine.
        upgraded = [record(ft, op; julia = "1.12.7") for ft in ("Float32", "Float64")
                    for op in ("advection", "full")]
        @test occursin("machines.toml pins 1.12.6", failure(() -> validate(upgraded, dashboard_for(upgraded))))
        unhashed = [raw[1:3]; record("Float64", "full"; manifest = "")]
        @test occursin("no Manifest sha256", failure(() -> validate(unhashed, dashboard_for(unhashed))))
        mixed = [raw[1:3]; record("Float64", "full"; manifest = OTHER_MANIFEST_SHA)]
        @test occursin("several environment fingerprints", failure(() -> validate(mixed, dashboard_for(mixed))))
        other_env = [record(ft, op; manifest = OTHER_MANIFEST_SHA) for ft in ("Float32", "Float64")
                     for op in ("advection", "full")]
        @test occursin("differ from the raw records", failure(() -> validate(raw, dashboard_for(other_env))))

        # What the Benchmarks workflow re-checks before publishing.
        @test LL.validate_publishable_records(dashboard_for(raw), COMMIT) === nothing
        @test_throws LL.LabValidationError LL.validate_publishable_records(dashboard_for(raw), OTHER_COMMIT)
        @test_throws LL.LabValidationError LL.validate_publishable_records(Any[], COMMIT)
        @test_throws LL.LabValidationError LL.validate_publishable_records(dashboard_for(mixed), COMMIT)
        @test_throws LL.LabValidationError LL.validate_publishable_records(dashboard_for(unhashed), COMMIT)

        merged = LL.merge_dashboard_records(["curry" => dashboard_for(raw[1:2]),
                                             "wurst" => dashboard_for(raw[3:4])])
        @test length(merged) == 4
        @test_throws LL.LabValidationError LL.merge_dashboard_records(["curry" => dashboard_for(raw),
                                                                       "other" => dashboard_for(raw)])
        # Machines that resolved different environments are never merged or published.
        msg = failure(() -> LL.merge_dashboard_records(["curry" => dashboard_for(raw[1:2]),
                                                        "mac" => dashboard_for(other_env[3:4])]))
        @test occursin("different environments", msg) && occursin("curry: Julia 1.12.6", msg) &&
              occursin(OTHER_MANIFEST_SHA, msg)
    end

    @testset "dry-run command construction" begin
        ms = LL.load_machines()
        wurst, curry, mac = (machine_named(ms, n) for n in ("wurst", "curry", "mac"))
        results = joinpath(tempdir(), "lab_results_test", COMMIT)
        shared_work = expanduser("~/atmostr_bench/$(COMMIT)")

        # Shared archives are extracted once on this host for wurst and curry.
        prepared = Set{String}()
        prep_wurst = LL.preparation_steps(wurst, COMMIT, REPO, prepared)
        @test [s.argv[1] for s in prep_wurst] == ["mkdir", "tar"]
        @test prep_wurst[2].stdin_argv == ["git", "-C", REPO, "archive", "--format=tar", COMMIT]
        @test prep_wurst[2].argv == ["tar", "-x", "-f", "-", "-C", shared_work]
        @test isempty(LL.preparation_steps(curry, COMMIT, REPO, prepared))

        # Local machine: plain commands with environment and working directory.
        plan = LL.machine_plan(wurst, COMMIT; results_dir = results)
        run = only(filter(s -> startswith(s.description, "run preset"), plan))
        @test run.argv[1] == expanduser("~/.juliaup/bin/julia")
        @test run.dir == shared_work
        @test ("CUDA_VISIBLE_DEVICES" => "0") in run.env
        @test any(p -> first(p) == "TMPDIR", run.env)
        for arg in ("--preset=lab", "--backend=cuda", "--float-type=Float32", "--commit=$(COMMIT)",
                    "--project=$(shared_work)/benchmarking", "--threads=2",
                    "--output=$(shared_work)/lab_results/wurst/benchmark_results.json")
            @test arg in run.argv
        end
        copyback = plan[end]
        @test copyback.argv[1] == "cp" && copyback.argv[end] == joinpath(results, "wurst") * "/"

        # ssh + shared NFS root: absolute paths in the remote script.
        plan = LL.machine_plan(curry, COMMIT; results_dir = results)
        run = only(filter(s -> startswith(s.description, "run preset"), plan))
        @test run.argv[1:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=60", "curry"]
        script = run.argv[end]
        @test occursin("--project='$(shared_work)/benchmarking'", script)
        @test occursin("--float-type=Float32,Float64", script)
        @test occursin("--commit=$(COMMIT)", script)
        @test occursin("export CUDA_VISIBLE_DEVICES='0'", script)
        @test occursin("--heap-size-hint=", script)
        @test !occursin("caffeinate", script)
        @test plan[end].argv[1] == "cp"

        # ssh + copied archive: remote-home paths, caffeinate, scp back.
        prep_mac = LL.preparation_steps(mac, COMMIT, REPO, prepared)
        copy = only(prep_mac)
        @test copy.argv[1] == "ssh" && copy.stdin_argv[4] == "archive"
        @test occursin("tar -x -f - -C \"\$HOME\"/'atmostr_bench/$(COMMIT)'", copy.argv[end])
        plan = LL.machine_plan(mac, COMMIT; results_dir = results)
        run = only(filter(s -> startswith(s.description, "run preset"), plan))
        script = run.argv[end]
        @test occursin("caffeinate -i \"\$HOME\"/'.juliaup/bin/julia'", script)
        @test occursin("--backend=metal", script) && occursin("--float-type=Float32", script)
        @test occursin("--project=\"\$HOME\"/'atmostr_bench/$(COMMIT)/benchmarking'", script)
        @test occursin("export JULIA_DEPOT_PATH='~/atmostr_bench/julia_depot:'", script)
        instantiate = only(filter(s -> startswith(s.description, "instantiate"), plan))
        @test occursin("caffeinate -i", instantiate.argv[end])
        @test occursin("export JULIA_DEPOT_PATH=", instantiate.argv[end])
        @test occursin("VERSION == v\"$(PINNED_JULIA)\"", instantiate.argv[end])
        scp = plan[end]
        @test scp.argv[1] == "scp"
        @test "$(mac.access.host):atmostr_bench/$(COMMIT)/lab_results/mac/benchmark_results.json" in scp.argv

        # Instantiation runs only on the pinned Julia version.
        local_instantiate = only(filter(s -> startswith(s.description, "instantiate"),
                                        LL.machine_plan(wurst, COMMIT; results_dir = results)))
        expression = local_instantiate.argv[end]
        @test local_instantiate.argv[end - 1] == "-e"
        @test expression == LL.instantiate_expression(wurst)
        @test endswith(expression, "; using Pkg; Pkg.instantiate()")
        @test Meta.parseall(expression) isa Expr
        gate_at(version) = Core.eval(Module(), Meta.parse(
            "let VERSION = $(repr(version)); $(LL.julia_version_gate(wurst)) end"))
        @test gate_at(VersionNumber(PINNED_JULIA)) === true
        for other in (v"1.12.7", v"1.11.7", v"1.13.0", v"1.12.6-rc1")
            gate_error = try
                gate_at(other)
                ""
            catch err
                sprint(showerror, err)
            end
            @test occursin("wurst runs Julia $(other)", gate_error) &&
                  occursin("pins julia_version = $(PINNED_JULIA)", gate_error)
        end

        # A device_label override reaches the harness command line.
        labelled = only(machines_from(MINIMAL_MACHINE * "device_label = \"A100 80GB\"\n"))
        run = only(filter(s -> startswith(s.description, "run preset"),
                          LL.machine_plan(labelled, COMMIT; results_dir = results)))
        @test "--device-label=A100 80GB" in run.argv

        # The dry-run executor prints commands and runs nothing.
        io = IOBuffer()
        LL.execute!(LL.DryRunExecutor(io), copy)
        @test LL.capture(LL.DryRunExecutor(io), LL.LabStep("publish", "store blob", ["git", "hash-object"]),
                         "<blob>") == "<blob>"
        out = String(take!(io))
        @test occursin("[dry-run] [mac]", out) && occursin("git -C", out) && occursin("| ssh", out)
        @test occursin("[dry-run] [publish] store blob", out)
        @test !isdir(joinpath(results, "mac"))
    end

    @testset "results come back after a failed run" begin
        machine = machine_named(LL.load_machines(), "mac")       # ssh + copied archive
        results = joinpath(tempdir(), "lab_results_test", COMMIT)
        function run_scripted(fail)
            ex = ScriptedExecutor(String[], fail)
            err = try
                LL.run_machine!(ex, machine, COMMIT, REPO, Set{String}(); preset = "lab",
                                results_dir = results)
                nothing
            catch e
                e
            end
            return ex.log, err
        end
        copied(log) = any(startswith("copy results back"), log)

        log, err = run_scripted(r"^$^")                          # nothing fails
        @test err === nothing && copied(log)
        @test findfirst(startswith("run preset"), log) < findfirst(startswith("copy results back"), log)

        # The harness exits nonzero after writing partial JSON: copy, then report the run failure.
        log, err = run_scripted(r"^run preset")
        @test copied(log) && occursin("scripted failure: run preset", sprint(showerror, err))

        # Nothing was written either: the failed copy is logged, the run failure reported.
        log, err = @test_logs (:warn, r"could not copy result files back") match_mode = :any run_scripted(r"^(run preset|copy results back)")
        @test copied(log) && occursin("scripted failure: run preset", sprint(showerror, err))

        # A failed copy after a successful run is the error.
        log, err = run_scripted(r"^copy results back")
        @test occursin("scripted failure: copy results back", sprint(showerror, err))

        # Code delivery failed: there is nothing to copy back.
        log, err = run_scripted(r"^copy git archive")
        @test !copied(log) && occursin("scripted failure: copy git archive", sprint(showerror, err))
    end

    @testset "lab command-line options and publishing names" begin
        opts = LL.parse_lab_options(String[])
        @test opts.preset == "lab" && opts.remote == "origin" && opts.gh == "/usr/bin/gh"
        @test opts.commit === nothing && isempty(opts.machines) && opts.fetch && !opts.publish
        opts = LL.parse_lab_options(["--machines=wurst,curry,mac", "--dry-run", "--no-fetch",
                                     "--commit=HEAD~1", "--github-repo=o/r"])
        @test opts.machines == ["wurst", "curry", "mac"] && opts.dry_run && !opts.fetch
        @test opts.commit == "HEAD~1" && opts.github_repo == "o/r"
        @test_throws ArgumentError LL.parse_lab_options(["--publish", "--allow-unpublished"])
        @test_throws ArgumentError LL.parse_lab_options(["--machine=wurst"])
        @test_throws ArgumentError LL.parse_lab_options(["--commit="])

        @test LL.inbox_path(COMMIT, DateTime(2026, 10, 10, 1, 2, 3)) ==
              "lab/$(COMMIT)-20261010T010203Z.json"

        # The Benchmarks workflow accepts exactly these inbox paths and dispatch inputs.
        workflow = read(joinpath(REPO, ".github", "workflows", "Benchmarks.yml"), String)
        accepted = match(r"\$\{LAB_RESULTS\}\" =~ (\S+) \]\]", workflow)
        @test accepted !== nothing
        @test occursin(Regex(accepted.captures[1]), LL.inbox_path(COMMIT, DateTime(2026, 1, 2, 3, 4, 5)))
        @test !occursin(Regex(accepted.captures[1]), "lab/../$(COMMIT)-20260102T030405Z.json")
        @test occursin("origin $(LL.INBOX_BRANCH)", workflow)
        for input in ("lab_results:", "run_cpu:", "publish:")
            @test occursin(input, workflow)
        end
        # GPU numbers come only from the lab lane: no self-hosted runner jobs
        # or their inputs on this public repository.
        code = join(filter(line -> !startswith(lstrip(line), "#"), split(workflow, '\n')), '\n')
        for removed in ("self-hosted", "run_cuda", "run_metal", "gpu_float_types", "cuda:", "metal:")
            @test !occursin(removed, code)
        end
        @test all(m -> m.captures[1] == "ubuntu-latest", eachmatch(r"runs-on: (.+)", code))
        # Jobs that run pull-request code get a read-only token; only the
        # publish job (never run on pull requests) may write.
        @test !occursin("pull-requests:", code)
        @test count("contents: write", code) == 1
        jobs = split(code, r"\n  (?=[a-z]+:\n)")
        cpu_job = only(filter(j -> startswith(j, "cpu:"), jobs))
        publish_job = only(filter(j -> startswith(j, "publish:"), jobs))
        @test occursin("permissions:\n      contents: read\n", cpu_job)
        @test occursin("contents: write", publish_job) && occursin("needs: [cpu]", publish_job)
        # The dashboard entry is attributed to the dispatched commit, not to
        # whatever main points at when the publish step runs.
        @test occursin("ref: \${{ github.sha }}", workflow)
        @test LL.github_slug("https://github.com/RemoteSensingTools/AtmosTransport.jl.git") ==
              "RemoteSensingTools/AtmosTransport.jl"
        @test LL.github_slug("git@github.com:RemoteSensingTools/AtmosTransport.jl.git") ==
              "RemoteSensingTools/AtmosTransport.jl"
        @test_throws ArgumentError LL.github_slug("/srv/git/repo.git")
    end
end
