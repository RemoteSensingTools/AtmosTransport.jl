using Test, TOML
include(joinpath(@__DIR__, "..", "..", "benchmarking", "lab", "LabLane.jl"))
const LL = AtmosTransportLabLane
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const COMMIT = "0123456789abcdef0123456789abcdef01234567"
machine_named(ms, name) = only(filter(m -> m.name == name, ms))
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
        @test occursin("VERSION == v\"$(wurst.julia_version)\"", instantiate.argv[end])
        scp = plan[end]
        @test scp.argv[1] == "scp"
        @test "$(mac.access.host):atmostr_bench/$(COMMIT)/lab_results/mac/benchmark_results.json" in scp.argv

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
