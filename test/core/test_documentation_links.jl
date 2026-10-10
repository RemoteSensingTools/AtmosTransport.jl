using Test

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const MAINTAINED_DOCUMENTS = let
    documents = [
        joinpath(REPOSITORY_ROOT, "README.md"),
        joinpath(REPOSITORY_ROOT, "CONTRIBUTING.md"),
        joinpath(REPOSITORY_ROOT, "benchmarking", "README.md"),
        joinpath(REPOSITORY_ROOT, "config", "preprocessing", "README.md"),
        joinpath(REPOSITORY_ROOT, "config", "runs", "README.md"),
        joinpath(REPOSITORY_ROOT, "docs", "README.md"),
        joinpath(REPOSITORY_ROOT, "scripts", "downloads", "README.md"),
        joinpath(REPOSITORY_ROOT, "test", "README.md"),
    ]

    for name in (
        "00_SCOPE_AND_STATUS.md",
        "10_CORE_CONTRACTS.md",
        "20_RUNTIME_FLOW.md",
        "30_BINARY_AND_DRIVERS.md",
        "35_RUNTIME_STABILITY_AND_SUBCYCLING.md",
        "40_QUALITY_GATES.md",
    )
        push!(documents, joinpath(REPOSITORY_ROOT, "docs", name))
    end

    for directory in (joinpath(REPOSITORY_ROOT, "docs", "src"),)
        for (root, _, files) in walkdir(directory)
            append!(documents,
                    joinpath.(Ref(root), filter(file -> endswith(file, ".md"), files)))
        end
    end

    for (root, _, files) in walkdir(joinpath(REPOSITORY_ROOT, "src"))
        "README.md" in files && push!(documents, joinpath(root, "README.md"))
    end

    sort!(unique!(documents))
end

function local_document_targets(path::AbstractString)
    contents = read(path, String)
    raw_targets = [match.captures[1]
                   for match in eachmatch(r"!?\[[^\]]*\]\(([^)]+)\)", contents)]
    append!(raw_targets,
            [match.captures[1]
             for match in eachmatch(r"(?i)\b(?:src|href|poster)\s*=\s*[\"']([^\"']+)[\"']",
                                    contents)])

    targets = String[]
    for raw_target in raw_targets
        target = strip(raw_target)
        startswith(target, "<") && endswith(target, ">") &&
            (target = target[2:end-1])
        target = first(split(target))
        isempty(target) && continue
        windows_absolute = occursin(r"^[A-Za-z]:[\\/]", target)
        ignored_prefixes = ("#", "@ref", "@id")
        any(prefix -> startswith(target, prefix), ignored_prefixes) && continue
        occursin(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) &&
            !windows_absolute && continue
        push!(targets, replace(first(split(target, '#')), "%20" => " "))
    end
    return targets
end

@testset "maintained documentation has valid local links" begin
    missing = String[]
    for document in MAINTAINED_DOCUMENTS
        @test isfile(document)
        for target in local_document_targets(document)
            absolute = isabspath(target) || occursin(r"^[A-Za-z]:[\\/]", target)
            if absolute
                push!(missing,
                      "$(relpath(document, REPOSITORY_ROOT)) -> $target (absolute path)")
                continue
            end
            destination = normpath(joinpath(dirname(document), target))
            ispath(destination) ||
                push!(missing,
                      "$(relpath(document, REPOSITORY_ROOT)) -> $target")
        end
    end
    @test isempty(missing)
end

@testset "home page exposes VitePress front matter first" begin
    home = read(joinpath(REPOSITORY_ROOT, "docs", "src", "index.md"), String)
    @test startswith(home, "```@raw html\n---\nlayout: home\n")
end
