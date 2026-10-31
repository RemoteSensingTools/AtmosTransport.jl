using Test, TOML

include(joinpath(@__DIR__, "..", "..", "scripts", "checks", "check_structure.jl"))
using .StructureCheck

@testset "repository structure ratchet" begin
    base = get(ENV, "ATMOSTR_STRUCTURE_BASE", "origin/main")
    # Package archives have no repository metadata; a worktree has a .git file.
    if !ispath(joinpath(StructureCheck.ROOT, ".git"))
        @info "Skipping structure ratchet comparison outside a git checkout; static structure checks still run"
        base = nothing
    end
    result = StructureCheck.check_structure(; base)
    @test isempty(result.errors)
    @test read(joinpath(StructureCheck.ROOT, StructureCheck.CODE_MAP), String) ==
          StructureCheck.code_map(result.inventory, result.findings)
end

@testset "JuliaSyntax remains a synchronized test-only dependency" begin
    root = TOML.parsefile(joinpath(StructureCheck.ROOT, "Project.toml"))
    tests = TOML.parsefile(joinpath(StructureCheck.ROOT, "test", "Project.toml"))
    @test !haskey(root["deps"], "JuliaSyntax")
    @test root["extras"]["JuliaSyntax"] == tests["deps"]["JuliaSyntax"]
    @test "JuliaSyntax" in root["targets"]["test"]
    @test root["compat"]["JuliaSyntax"] == tests["compat"]["JuliaSyntax"]
end

function write_toml(path, data)
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, data; sorted = true)
    end
end

function structure_fixture(f)
    mktempdir() do root
        mkpath(joinpath(root, "src"))
        write(joinpath(root, "src", "AtmosTransport.jl"), """
        module AtmosTransport
        include("Lower.jl")
        include("Upper.jl")
        end
        """)
        write(joinpath(root, "src", "Lower.jl"), """
        module Lower
        export base
        base(x) = x
        _hidden(x) = x - 1
        end
        """)
        write(joinpath(root, "src", "Upper.jl"), """
        module Upper
        using ..Lower: base
        caller(x) = base(x) + 1
        end
        """)
        inv = StructureCheck.inventory(root)
        found = StructureCheck.findings(inv)
        write_toml(joinpath(root, "src", "LAYERS.toml"), Dict(
            "order" => inv.order, "modules" => Dict(m => [b for (a,b) in found.edges if a == m] for m in inv.order),
            "folder_order" => ["src"], "folders" => Dict("src" => String[])))
        write_toml(joinpath(root, StructureCheck.FOLDER_SIZES), Dict("allow" => Dict()))
        write_toml(joinpath(root, StructureCheck.MIRRORS), Dict("pairs" => []))
        for file in values(StructureCheck.SNAPSHOTS)
            write_toml(joinpath(root, file), Dict("allow" => Dict{String,String}()))
        end
        f(root)
    end
end

check_fixture(root) = StructureCheck.check_structure(root; io = devnull, base = nothing, check_map = false)
function add_source(root, file, text)
    path = joinpath(root, "src", file)
    # Append definitions before the module's final end.
    content = read(path, String)
    write(path, replace(content, r"end\s*$" => text * "\nend\n"))
end

@testset "new violations fail and intended sharing passes" begin
    structure_fixture() do root
        @test isempty(check_fixture(root).errors)
        add_source(root, "Lower.jl", "later(x) = Upper.caller(x)")
        errors = check_fixture(root).errors
        @test any(occursin("new dependency", e) for e in errors)
        @test any(occursin("later module", e) for e in errors)
        @test any(occursin("cycle", e) for e in errors)
    end
    structure_fixture() do root
        add_source(root, "Upper.jl", "base(x::String) = length(x)")
        @test any(occursin("duplicate:", e) for e in check_fixture(root).errors)
        path = joinpath(root, "src", "Upper.jl")
        write(path, replace(read(path, String), "using ..Lower: base" => "import ..Lower: base"))
        @test isempty(check_fixture(root).errors) # imported method extends Lower.base
        add_source(root, "Upper.jl", "Lower.base(x::Symbol) = 3")
        @test isempty(check_fixture(root).errors) # qualified extension
    end
    structure_fixture() do root
        for file in ("Lower.jl", "Upper.jl")
            add_source(root, file, "import Base: length\nlength(x::Nothing) = 0")
        end
        @test isempty(check_fixture(root).errors) # external generic, not two new helpers
    end
    structure_fixture() do root
        # A one-line body with a multiline signature and a long-form body
        # produce identical streams with signatures excluded.
        body = "x + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13"
        add_source(root, "Lower.jl", "first_copy(\nx\n) = $body")
        add_source(root, "Upper.jl", "function second_copy(y)\n$body\nend")
        @test any(occursin("clone:", e) for e in check_fixture(root).errors)
        # The baseline permits these sites, but not another method with the
        # same name/body in an already allowlisted file.
        existing = StructureCheck.findings(StructureCheck.inventory(root)).clone
        write_toml(joinpath(root, StructureCheck.SNAPSHOTS["clone"]),
                   Dict("allow" => Dict(id => "fixture baseline" for id in keys(existing))))
        @test isempty(check_fixture(root).errors)
        add_source(root, "Lower.jl", "first_copy(x::String) = $body")
        @test any(occursin("clone:", e) for e in check_fixture(root).errors)
    end
    structure_fixture() do root
        a = join(["x + $i" for i in 1:70], " + ")
        b = join(["y + $i" for i in 2:71], " + ")
        add_source(root, "Lower.jl", "first_shape(x) = $a")
        add_source(root, "Upper.jl", "second_shape(y) = $b")
        result = check_fixture(root)
        @test isempty(result.errors)
        @test !isempty(result.findings.similar) # renamed/literal variants warn only
    end
    structure_fixture() do root
        add_source(root, "Upper.jl", "import ..Lower as L\nreach(x) = L._hidden(x)")
        @test any(occursin("private_use:", e) for e in check_fixture(root).errors)
        # README declarations and public declarations both authorize sharing.
        write(joinpath(root, "src", "README.md"), "# shared: _hidden\n")
        @test isempty(check_fixture(root).errors)
        rm(joinpath(root, "src", "README.md"))
        add_source(root, "Lower.jl", "public _hidden")
        @test isempty(check_fixture(root).errors)
    end
    structure_fixture() do root
        add_source(root, "Upper.jl", "using ..Lower: _hidden as shared_hidden\nf(x) = shared_hidden(x)")
        @test any(occursin("private_use:", e) for e in check_fixture(root).errors)
    end
    structure_fixture() do root
        add_source(root, "Upper.jl", "import ..Lower as L\nusing .L: _hidden\nf(x) = _hidden(x)")
        found = check_fixture(root)
        @test any(occursin("private_use:", e) for e in found.errors)
        @test haskey(found.findings.private_use, "src/Upper.jl|AtmosTransport.Upper|AtmosTransport.Lower|_hidden")
        add_source(root, "Lower.jl", "public _hidden")
        @test isempty(check_fixture(root).errors)
    end
    structure_fixture() do root
        body = "x + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13"
        add_source(root, "Lower.jl", "struct Op end\n(op::Op)(x) = $body")
        add_source(root, "Upper.jl", "struct Op end\nfunction (op::Op)(x)\n$body\nend")
        result = check_fixture(root)
        @test any(occursin("clone:", e) for e in result.errors)
        @test isempty(result.findings.duplicate) # callable methods own no generic
    end
    structure_fixture() do root
        entry = joinpath(root, "src", "AtmosTransport.jl")
        write(entry, replace(read(entry, String), "include(\"Upper.jl\")" =>
                             "import .Lower as Renamed\ninclude(\"Upper.jl\")"))
        add_source(root, "Upper.jl", "import ..AtmosTransport: Renamed as L\nf(x) = L._hidden(x)")
        found = StructureCheck.findings(StructureCheck.inventory(root))
        @test any(endswith(id, "|AtmosTransport.Lower|_hidden") for id in keys(found.private_use))
    end
    structure_fixture() do root
        add_source(root, "Upper.jl", "\"\"\"Attached.\"\"\"\nf(x) = x / 2")
        @test isempty(check_fixture(root).errors)
        add_source(root, "Upper.jl", "\"\"\"Dropped.\"\"\"\n\n# detached by whitespace/comment\ng(x) = x / 3")
        @test any(occursin("docstring:", e) for e in check_fixture(root).errors)
    end
    structure_fixture() do root
        write(joinpath(root, "src", "orphan.jl"), "orphan(x) = x\n")
        @test any(occursin("unreached source", e) for e in check_fixture(root).errors)
    end
    structure_fixture() do root
        add_source(root, "Lower.jl", "#" * repeat("size budget ", 20_000))
        @test any(occursin("exceeds 65000", e) for e in check_fixture(root).errors)
    end
    structure_fixture() do root
        write(joinpath(root, "src", "README.md"), "# tiny\n")
        output = IOBuffer()
        result = StructureCheck.check_structure(root; io = output, base = nothing, check_map = false)
        @test isempty(result.errors) # T6 is section presence, report only in P1
        @test occursin("README.md: Depends on", String(take!(output)))
    end
end

function folder_fixture(f)
    structure_fixture() do root
        for name in ("Lower", "Upper")
            mkpath(joinpath(root, "src", name))
            mv(joinpath(root, "src", "$name.jl"), joinpath(root, "src", name, "$name.jl"))
        end
        path = joinpath(root, "src", "AtmosTransport.jl")
        write(path, replace(read(path, String), "Lower.jl" => "Lower/Lower.jl", "Upper.jl" => "Upper/Upper.jl"))
        layers = TOML.parsefile(joinpath(root, "src", "LAYERS.toml"))
        found = StructureCheck.findings(StructureCheck.inventory(root))
        layers["folder_order"] = ["src/Lower", "src/Upper", "src"]
        layers["folders"] = Dict(d => sort!([b for (a,b) in found.folder_edges if a == d]) for d in layers["folder_order"])
        write_toml(joinpath(root, "src", "LAYERS.toml"), layers)
        f(root)
    end
end

@testset "folder order, sizes, and generated map" begin
    folder_fixture() do root
        @test isempty(check_fixture(root).errors)
        path = joinpath(root, "src", "LAYERS.toml")
        layers = TOML.parsefile(path)
        layers["folder_order"] = ["src/Upper", "src/Lower", "src"]
        write_toml(path, layers)
        @test any(occursin("later folder", e) for e in check_fixture(root).errors)
    end
    folder_fixture() do root
        # Same module, new folder: module-edge enforcement alone misses this.
        mkpath(joinpath(root, "src", "Lower", "helpers"))
        write(joinpath(root, "src", "Lower", "helpers", "helper.jl"), "helper(x) = x + 7\n")
        add_source(root, "Lower/Lower.jl", "include(\"helpers/helper.jl\")")
        layers = TOML.parsefile(joinpath(root, "src", "LAYERS.toml"))
        pushfirst!(layers["folder_order"], "src/Lower/helpers")
        layers["folders"]["src/Lower/helpers"] = String[]
        layers["folders"]["src/Lower"] = ["src/Lower/helpers"]
        write_toml(joinpath(root, "src", "LAYERS.toml"), layers)
        @test isempty(check_fixture(root).errors)
        add_source(root, "Upper/Upper.jl", "using ..Lower: helper\nf(x) = helper(x)")
        result = check_fixture(root)
        @test any(occursin("new folder dependency: src/Upper -> src/Lower/helpers", e) for e in result.errors)
        @test !any(occursin("T1 new dependency:", e) for e in result.errors)
    end
    folder_fixture() do root
        # Each folder fits separately, though the src subtree exceeds 65k.
        add_source(root, "Lower/Lower.jl", "#" * repeat("a", 120_000))
        add_source(root, "Upper/Upper.jl", "#" * repeat("b", 120_000))
        @test isempty(check_fixture(root).errors)
        add_source(root, "Lower/Lower.jl", "#" * repeat("c", 120_000))
        @test any(occursin("T5 src/Lower exceeds", e) for e in check_fixture(root).errors)
        inv = StructureCheck.inventory(root)
        bytes = StructureCheck.folder_bytes(inv, StructureCheck.source_folders(inv)["src/Lower"])
        write_toml(joinpath(root, StructureCheck.FOLDER_SIZES), Dict("allow" => Dict("src/Lower" =>
            Dict("bytes" => bytes, "reason" => "fixture existing debt"))))
        @test isempty(check_fixture(root).errors)
        add_source(root, "Lower/Lower.jl", "# growth")
        @test any(occursin("T5 src/Lower exceeds", e) for e in check_fixture(root).errors)
        path = joinpath(root, "src", "Lower", "Lower.jl")
        write(path, "module Lower\nexport base\nbase(x) = x\nend\n")
        @test any(occursin("T5 remove allowance", e) for e in check_fixture(root).errors)
        write_toml(joinpath(root, StructureCheck.FOLDER_SIZES), Dict("allow" => Dict()))
        @test isempty(check_fixture(root).errors)
    end
    folder_fixture() do root
        write_toml(joinpath(root, StructureCheck.MIRRORS), Dict("pairs" => [Dict(
            "forward" => "src/Lower/Lower.jl", "adjoint" => "src/Upper/Upper.jl")]))
        path = joinpath(root, StructureCheck.CODE_MAP)
        mkpath(dirname(path))
        map = StructureCheck.code_map(StructureCheck.inventory(root))
        write(path, map)
        @test isempty(StructureCheck.check_structure(root; io = devnull, base = nothing).errors)
        @test count("`src/Lower/Lower.jl` ↔ `src/Upper/Upper.jl`", map) == 2
        @test occursin("Public names: **1**", map)
        add_source(root, "Upper/Upper.jl", "more(x) = x - 3")
        @test any(occursin("code map is stale", e) for e in
                  StructureCheck.check_structure(root; io = devnull, base = nothing).errors)
        status = redirect_stdout(devnull) do
            StructureCheck.main(["--root=$root", "--write-code-map"])
        end
        @test status == 1 # map regenerated, missing default base still fails
        @test read(path, String) == StructureCheck.code_map(StructureCheck.inventory(root))
    end
end

@testset "core never depends on non-core modules" begin
    for mod in ("Visualization", "Analysis", "DataDownloads", "Extras")
        structure_fixture() do root
            # Load the non-core owner first so only the boundary rule is at issue.
            add_source(root, "Lower.jl", "")
            entry = joinpath(root, "src", "AtmosTransport.jl")
            write(entry, replace(read(entry, String), "include(\"Lower.jl\")" =>
                "module $mod\nexport exposed\nexposed(x) = x\nend\ninclude(\"Lower.jl\")"))
            add_source(root, "Upper.jl", "using ..$mod: exposed\ng(x) = exposed(x)")
            result = check_fixture(root)
            @test any(occursin("noncore:", e) for e in result.errors)
            @test any(endswith(id, "|AtmosTransport.$mod") for id in keys(result.findings.noncore))
        end
    end
    @test StructureCheck.is_noncore("AtmosTransportMakieExt")
    @test StructureCheck.is_noncore("AtmosTransport.Extras.Plotting")
    @test !StructureCheck.is_noncore("AtmosTransport.Diagnostics")
    structure_fixture() do root
        # A neutral module name cannot bypass the Extras path boundary.
        mkpath(joinpath(root, "src", "Extras"))
        write(joinpath(root, "src", "Extras", "Utilities.jl"),
              "module Utilities\nexport exposed\nexposed(x) = x\nend\n")
        entry = joinpath(root, "src", "AtmosTransport.jl")
        write(entry, replace(read(entry, String), "include(\"Lower.jl\")" =>
                            "include(\"Extras/Utilities.jl\")\ninclude(\"Lower.jl\")"))
        add_source(root, "Upper.jl", "using ..Utilities: exposed\ng(x) = exposed(x)")
        result = check_fixture(root)
        @test any(endswith(id, "|AtmosTransport.Utilities") for id in keys(result.findings.noncore))
    end
end

@testset "base ref validation and git snapshot ratchet" begin
    structure_fixture() do root
        # Reuse existing history read-only. Package archives have none; the
        # pure allowance tests below still exercise growth in that environment.
        repo = StructureCheck.ROOT
        if !ispath(joinpath(repo, ".git"))
            @info "Skipping Git history fixture outside a checkout"
            return
        end
        gitdir = strip(read(`git -C $repo rev-parse --absolute-git-dir`, String))
        write(joinpath(root, ".git"), "gitdir: $gitdir\n")
        base = get(ENV, "ATMOSTR_STRUCTURE_BASE", "origin/main")
        files = vcat(["src/LAYERS.toml", StructureCheck.FOLDER_SIZES], collect(values(StructureCheck.SNAPSHOTS)))
        errors = String[]
        StructureCheck.check_ratchet!(errors, root, "missing-ref", files)
        @test any(occursin("cannot validate structure base", e) for e in errors)
        @test any(occursin("git fetch --no-tags --depth=1 origin", e) for e in errors)
        empty!(errors)
        # An absent snapshot at an existing commit is first introduction. Use
        # a fixture-only path so this also works with depth-one checkouts.
        StructureCheck.check_ratchet!(errors, root, base, ["test/structure/fixture-absent.toml"])
        @test isempty(errors) # valid base, absent snapshot = first introduction
        # Copy the selected base's exact snapshots, then introduce hidden debt.
        for file in files
            if !isempty(read(`git -C $repo ls-tree $base -- $file`, String))
                write(joinpath(root, file), read(`git -C $repo show $base\:$file`, String))
            end
        end
        StructureCheck.check_ratchet!(errors, root, base, files)
        @test isempty(errors)
        write_toml(joinpath(root, StructureCheck.SNAPSHOTS["clone"]), Dict("allow" => Dict("new" => "hidden debt")))
        StructureCheck.check_ratchet!(errors, root, base, files)
        @test any(occursin("allowlists only shrink", e) for e in errors)
    end
end

@testset "allowances only shrink" begin
    original = Dict("allow" => Dict("existing" => "reason"))
    removed = Dict("allow" => Dict{String,String}())
    expanded = Dict("allow" => Dict("existing" => "reason", "new" => "reason"))
    @test isempty(StructureCheck.allowance_growth(original, removed, "clone"))
    @test length(StructureCheck.allowance_growth(original, expanded, "clone")) == 1
    original_layers = Dict("modules" => Dict("Upper" => ["Lower"]))
    shrunk_layers = Dict("modules" => Dict("Upper" => String[]))
    expanded_layers = Dict("modules" => Dict("Upper" => ["Lower", "Extra"]))
    @test isempty(StructureCheck.allowance_growth(original_layers, shrunk_layers, "src/LAYERS.toml"))
    @test length(StructureCheck.allowance_growth(original_layers, expanded_layers, "src/LAYERS.toml")) == 1
    folder_layers = Dict("folders" => Dict("Upper" => ["Lower"]), "folder_order_allow" => Dict())
    reverse_allowance = deepcopy(folder_layers)
    reverse_allowance["folder_order_allow"]["Lower|Upper"] = "hidden reverse edge"
    @test length(StructureCheck.allowance_growth(folder_layers, reverse_allowance, "src/LAYERS.toml")) == 1
    extra_edge = deepcopy(folder_layers)
    push!(extra_edge["folders"]["Upper"], "Extra")
    @test length(StructureCheck.allowance_growth(folder_layers, extra_edge, "src/LAYERS.toml")) == 1
    size_before = Dict("allow" => Dict("folder" => Dict("bytes" => 300_000, "reason" => "existing")))
    size_after = deepcopy(size_before)
    size_after["allow"]["folder"]["bytes"] = 290_000
    @test isempty(StructureCheck.allowance_growth(size_before, size_after, StructureCheck.FOLDER_SIZES))
    size_after["allow"]["folder"]["bytes"] = 300_001
    @test length(StructureCheck.allowance_growth(size_before, size_after, StructureCheck.FOLDER_SIZES)) == 1
end

@testset "X1 relocates identities without accepting new debt" begin
    private_file = StructureCheck.SNAPSHOTS["private_use"]
    old = "src/Analysis/run_context.jl|AtmosTransport.Analysis|AtmosTransport.Models.DrivenRunner|_cfg_float_type"
    moved = "src/Diagnostics/run_context.jl|AtmosTransport.Diagnostics|AtmosTransport.Models.DrivenRunner|_cfg_float_type"
    before = Dict("allow" => Dict(old => "existing site"))
    after = Dict("allow" => Dict(moved => "relocated site"))
    @test isempty(StructureCheck.allowance_growth(before, after, private_file))
    changed = Dict("allow" => Dict(replace(moved, "_cfg_float_type" => "_new_helper") => "new site"))
    @test !isempty(StructureCheck.allowance_growth(before, changed, private_file))
    copied = Dict("allow" => Dict(old => "old copy", moved => "new copy"))
    @test !isempty(StructureCheck.allowance_growth(before, copied, private_file))
    @test !isempty(StructureCheck.allowance_growth(after, copied, private_file))

    for file in (StructureCheck.SNAPSHOTS["clone"], StructureCheck.SNAPSHOTS["docstring"])
        old = "digest|src/Analysis/mass_totals.jl|AtmosTransport.Analysis|helper|signature"
        moved = replace(old, "src/Analysis/" => "src/Diagnostics/",
                        "AtmosTransport.Analysis" => "AtmosTransport.Diagnostics")
        before = Dict("allow" => Dict(old => "existing body"))
        after = Dict("allow" => Dict(moved => "relocated body"))
        @test isempty(StructureCheck.allowance_growth(before, after, file))
        for field in ("digest", "signature")
            changed = Dict("allow" => Dict(replace(moved, field => "changed") => "changed body"))
            @test !isempty(StructureCheck.allowance_growth(before, changed, file))
        end
    end
    before = Dict("allow" => Dict("src/Downloads" => Dict("bytes" => 300_000, "reason" => "existing")))
    after = Dict("allow" => Dict("src/Extras/Downloads" => Dict("bytes" => 300_000, "reason" => "moved")))
    @test isempty(StructureCheck.allowance_growth(before, after, StructureCheck.FOLDER_SIZES))
    after["allow"]["src/Extras/Downloads"]["bytes"] += 1
    @test !isempty(StructureCheck.allowance_growth(before, after, StructureCheck.FOLDER_SIZES))

    before = Dict("modules" => Dict("AtmosTransport" => ["AtmosTransport.Analysis"],
                                   "AtmosTransport.Analysis" => ["AtmosTransport.Models"]))
    after = StructureCheck.relocate_x1_baseline(before, "src/LAYERS.toml")
    @test isempty(StructureCheck.allowance_growth(before, after, "src/LAYERS.toml"))
    push!(after["modules"]["AtmosTransport.Diagnostics"], "AtmosTransport.Visualization")
    @test !isempty(StructureCheck.allowance_growth(before, after, "src/LAYERS.toml"))
end

@testset "O1 relocates observation dependencies without expanding debt budgets" begin
    before = Dict("modules" => Dict(
        "AtmosTransport" => ["AtmosTransport.Output"],
        "AtmosTransport.Output" => ["AtmosTransport", "AtmosTransport.Grids"],
        "AtmosTransport.Adjoints" => ["AtmosTransport.Output"]))
    after = StructureCheck.relocate_o1_layers(before, "src/LAYERS.toml")
    @test isempty(StructureCheck.allowance_growth(before, after, "src/LAYERS.toml"))
    @test StructureCheck.relocate_o1_layers(after, "src/LAYERS.toml") == after
    push!(after["modules"]["AtmosTransport.Observations.Sources"], "AtmosTransport.GriddedIO")
    @test !isempty(StructureCheck.allowance_growth(before, after, "src/LAYERS.toml"))
    debt = Dict("allow" => Dict("old" => "existing debt"))
    @test StructureCheck.relocate_o1_layers(debt, "test/structure/private_use_allowlist.toml") == debt
end

@testset "M1 relocates only the binary dependency budget" begin
    parent = "AtmosTransport.MetDrivers"
    format, readers = parent * ".BinaryFormat", parent * ".BinaryReaders"
    before = Dict("modules" => Dict(parent => ["AtmosTransport.Grids", "AtmosTransport.State"],
                                    "AtmosTransport.Preprocessing" => [parent]))
    after = StructureCheck.relocate_m1_layers(before, "src/LAYERS.toml")
    @test isempty(StructureCheck.allowance_growth(before, after, "src/LAYERS.toml"))
    @test format in after["modules"][readers]
    expanded = deepcopy(after)
    push!(expanded["modules"][format], "AtmosTransport.Operators")
    @test !isempty(StructureCheck.allowance_growth(before, expanded, "src/LAYERS.toml"))
    @test !isempty(StructureCheck.allowance_growth(after, expanded, "src/LAYERS.toml"))
    debt = Dict("allow" => Dict("old" => "unchanged debt"))
    @test StructureCheck.relocate_m1_layers(debt, "private") == debt
end

@testset "PP1 relocates ordered preprocessing stages without accepting reverse edges" begin
    parent = "AtmosTransport.Preprocessing"
    core, vertical, spectral = [parent * "." * n for n in ("Core", "Vertical", "Spectral")]
    before = Dict("modules" => Dict(parent => ["AtmosTransport.Grids"],
                                    "AtmosTransport" => [parent]),
                  "folders" => Dict("src/Preprocessing" => ["src/Grids", "src/Preprocessing/sources"]))
    after = StructureCheck.relocate_pp1_layers(before, "src/LAYERS.toml")
    @test isempty(StructureCheck.allowance_growth(before, after, "src/LAYERS.toml"))
    @test vertical in after["modules"][spectral]
    @test !(parent in after["modules"][core])
    @test !("src/Preprocessing/sources" in after["folders"]["src/Preprocessing/Core"])
    for (from, to) in ((core, vertical), (vertical, spectral), (core, parent))
        expanded = deepcopy(after)
        push!(expanded["modules"][from], to)
        @test !isempty(StructureCheck.allowance_growth(before, expanded, "src/LAYERS.toml"))
        @test !isempty(StructureCheck.allowance_growth(after, expanded, "src/LAYERS.toml"))
    end
    debt = Dict("allow" => Dict("old" => "unchanged debt"))
    @test StructureCheck.relocate_pp1_layers(debt, "clone") == debt
end

@testset "PP2 stages inherit lower dependencies without reverse edges" begin
    parent = "AtmosTransport.Preprocessing"
    core, spectral, closure, physics = parent .* [".Core", ".Spectral", ".MassClosure", ".PhysicsInputs"]
    before = Dict("modules" => Dict(parent => ["AtmosTransport.Grids", core, spectral],
                                    core => ["AtmosTransport.Grids"],
                                    spectral => [core]),
                  "folders" => Dict("src/Preprocessing" => ["src/Grids", "src/Preprocessing/Core"]))
    after = StructureCheck.relocate_pp2_layers(before, "src/LAYERS.toml")
    @test isempty(StructureCheck.allowance_growth(before, after, "src/LAYERS.toml"))
    @test spectral in after["modules"][closure]
    @test spectral in after["modules"][physics]
    @test !(closure in after["modules"][physics])
    @test after["modules"][core] == before["modules"][core]
    @test StructureCheck.relocate_pp2_layers(after, "src/LAYERS.toml") == after
    for (from, to) in ((closure, physics), (physics, closure), (closure, parent), (core, closure))
        expanded = deepcopy(after)
        push!(expanded["modules"][from], to)
        @test !isempty(StructureCheck.allowance_growth(before, expanded, "src/LAYERS.toml"))
    end
    debt = Dict("allow" => Dict("old" => "existing debt"))
    @test StructureCheck.relocate_pp2_layers(debt, "clone") == debt
end

@testset "A1 shared-module folder wiring preserves sibling order" begin
    home = StructureCheck.A1_HOME
    children = StructureCheck.A1_CHILDREN
    before = Dict("folders" => Dict(home => ["src/Grids"], "src/Adjoints" => [home]))
    after = StructureCheck.relocate_a1_layers(before, "src/LAYERS.toml")
    @test isempty(StructureCheck.allowance_growth(before, after, "src/LAYERS.toml"))
    @test StructureCheck.relocate_a1_layers(after, "src/LAYERS.toml") == after
    for (from, to) in ((children[1], children[3]), (children[2], "src/Adjoints"))
        expanded = deepcopy(after)
        push!(expanded["folders"][from], to)
        @test !isempty(StructureCheck.allowance_growth(before, expanded, "src/LAYERS.toml"))
    end
    debt = Dict("allow" => Dict("old" => "existing"))
    @test StructureCheck.relocate_a1_layers(debt, StructureCheck.SNAPSHOTS["private_use"]) == debt

    mktempdir() do root
        files = Dict(
            "src/AtmosTransport.jl" => "module AtmosTransport\ninclude(\"Operators/Operators.jl\")\nend\n",
            "src/Operators/Operators.jl" => "module Operators\ninclude(\"Advection/Advection.jl\")\nend\n",
            home * "/Advection.jl" => "module Advection\ninclude(\"Reconstruction/helpers.jl\")\ninclude(\"Structured/helpers.jl\")\ninclude(\"CubedSphere/helpers.jl\")\nend\n",
            children[1] * "/helpers.jl" => "reconstruct(x) = x + 1\n",
            children[2] * "/helpers.jl" => "structured(x) = reconstruct(x) * 2\n",
            children[3] * "/helpers.jl" => "cube(x) = structured(x) - 1\n")
        for (file, text) in files
            path = joinpath(root, file)
            mkpath(dirname(path)); write(path, text)
        end
        inv = StructureCheck.inventory(root)
        found = StructureCheck.findings(inv)
        order = ["src", "src/Operators", home, children...]
        layers = Dict("order" => inv.order,
            "modules" => Dict(m => [b for (a,b) in found.edges if a == m] for m in inv.order),
            "folder_order" => order,
            "folders" => Dict(f => [b for (a,b) in found.folder_edges if a == f] for f in order))
        write_toml(joinpath(root, "src/LAYERS.toml"), layers)
        write_toml(joinpath(root, StructureCheck.FOLDER_SIZES), Dict("allow" => Dict()))
        write_toml(joinpath(root, StructureCheck.MIRRORS), Dict("pairs" => []))
        for file in values(StructureCheck.SNAPSHOTS)
            write_toml(joinpath(root, file), Dict("allow" => Dict()))
        end
        @test isempty(check_fixture(root).errors)
        open(joinpath(root, children[1], "helpers.jl"), "a") do io
            println(io, "reverse_dependency(x) = cube(x)")
        end
        @test any(occursin("reference to later folder", e) for e in check_fixture(root).errors)
    end
end

@testset "relative child modules shadow inherited module names" begin
    structure_fixture() do root
        add_source(root, "Upper.jl", """
        module Lower
        export local_value
        local_value(x) = x * 2
        end
        using .Lower: local_value
        local_caller(x) = local_value(x)
        """)
        inv = StructureCheck.inventory(root)
        bindings, _, _ = StructureCheck.resolve_imports(inv)
        @test StructureCheck.binding_owner(bindings, "AtmosTransport.Upper", "local_value") ==
              ("AtmosTransport.Upper.Lower", "local_value")
    end
end

@testset "PP3 target and binary identities retain one-way dependencies" begin
    parent = "AtmosTransport.Preprocessing"
    stages = parent .* [".Core", ".Vertical", ".Spectral", ".MassClosure", ".PhysicsInputs"]
    children = parent .* [".TransportBinary", ".LatLon", ".ReducedGaussian", ".CubedSphere"]
    before = Dict("modules" => Dict(parent => ["AtmosTransport.Grids"; stages],
                                    stages[1] => ["AtmosTransport.Grids"]),
                  "folders" => Dict("src/Preprocessing" => ["src/Grids", "src/Preprocessing/Core"]))
    after = StructureCheck.relocate_pp3_layers(before, "src/LAYERS.toml")
    @test isempty(StructureCheck.allowance_growth(before, after, "src/LAYERS.toml"))
    @test StructureCheck.relocate_pp3_layers(after, "src/LAYERS.toml") == after
    @test after["modules"][stages[1]] == before["modules"][stages[1]]
    @test children[1] in after["modules"][children[2]]
    @test children[2] in after["modules"][children[3]]
    @test children[2] in after["modules"][children[4]]
    for (from,to) in ((children[1],children[2]),(children[2],children[4]),
                      (children[4],children[3]),(children[3],parent),(stages[1],children[1]))
        expanded=deepcopy(after); push!(expanded["modules"][from],to)
        @test !isempty(StructureCheck.allowance_growth(before, expanded, "src/LAYERS.toml"))
    end
    debt=Dict("allow"=>Dict("old"=>"existing debt"))
    @test StructureCheck.relocate_pp3_layers(debt,"clone") == debt
end

@testset "absolute package imports cannot bind a target module with the same leaf name" begin
    structure_fixture() do root
        write_toml(joinpath(root,"Project.toml"), Dict("deps"=>Dict("Lower"=>"external-package")))
        inv=StructureCheck.inventory(root)
        @test StructureCheck.module_path(inv,"AtmosTransport.Upper",["Lower","Geometry"],0) == ""
        @test StructureCheck.module_path(inv,"AtmosTransport.Upper",["Lower"],2) == "AtmosTransport.Lower"
    end
end
