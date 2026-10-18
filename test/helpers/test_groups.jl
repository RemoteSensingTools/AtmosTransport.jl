# Test groups: the topics and the quick set of `test/groups.toml`, and the
# mapping from changed files to topics. Used by `test/runtests.jl` (the
# `--quick`, `--topics=` and `--changed` selectors) and checked by
# `core/test_test_groups.jl`. Only Base, TOML and git are needed.
module TestGroups

using TOML

export load_groups, topic_files, quick_files, changed_paths, select_changed, validate_groups

"""A topic: test entries (paths relative to `test/`) and repository path globs."""
struct Topic
    name::String
    description::String
    files::Vector{String}
    paths::Vector{String}
end

struct Groups
    topics::Vector{Topic}
    quick::Vector{String}
end

function load_groups(path::AbstractString)
    raw = TOML.parsefile(path)
    topics = [Topic(name, get(t, "description", ""), String.(get(t, "files", String[])),
                    String.(get(t, "paths", String[])))
              for (name, t) in sort!(collect(get(raw, "topics", Dict{String, Any}())); by = first)]
    return Groups(topics, String.(get(raw, "quick", String[])))
end

topic_names(g::Groups) = [t.name for t in g.topics]

function topic_files(g::Groups, names)
    files = Set{String}()
    for name in names
        i = findfirst(t -> t.name == name, g.topics)
        i === nothing && error("Unknown test topic '$name' (known: $(join(topic_names(g), ", ")))")
        union!(files, g.topics[i].files)
    end
    return files
end

quick_files(g::Groups) = Set(g.quick)

"""
    glob_regex(pattern) -> Regex

Repository-relative glob: `**` matches across folders (`dir/**` matches all of
`dir`), `*` and `?` match within one path component, `{a,b}` alternates.
"""
function glob_regex(pattern::AbstractString)
    out = IOBuffer()
    print(out, '^')
    i, n = firstindex(pattern), lastindex(pattern)
    depth = 0
    while i <= n
        c = pattern[i]
        if c == '*' && i < n && pattern[nextind(pattern, i)] == '*'
            i = nextind(pattern, i)
            # `**/` also matches no folder at all.
            if i < n && pattern[nextind(pattern, i)] == '/'
                i = nextind(pattern, i)
                print(out, "(?:.*/)?")
            else
                print(out, ".*")
            end
        elseif c == '*'
            print(out, "[^/]*")
        elseif c == '?'
            print(out, "[^/]")
        elseif c == '{'
            depth += 1; print(out, "(?:")
        elseif c == '}' && depth > 0
            depth -= 1; print(out, ")")
        elseif c == ',' && depth > 0
            print(out, "|")
        elseif c in ".+()|^\$[]\\"
            print(out, '\\', c)
        else
            print(out, c)
        end
        i = nextind(pattern, i)
    end
    print(out, '$')
    return Regex(String(take!(out)))
end

matches_glob(path::AbstractString, pattern::AbstractString) = occursin(glob_regex(pattern), path)

"""`a{b,c}d` as `["abd", "acd"]`, every brace group expanded (no nesting)."""
function expand_braces(pattern::AbstractString)
    m = match(r"\{([^{}]*)\}", pattern)
    m === nothing && return [String(pattern)]
    head = pattern[firstindex(pattern):prevind(pattern, m.offset)]
    tail = pattern[(m.offset + ncodeunits(m.match)):end]
    return reduce(vcat, [expand_braces(head * alt * tail) for alt in split(m.captures[1], ",")])
end

"""Topics whose `paths` match the repository-relative `path`."""
topics_for_path(g::Groups, path::AbstractString) =
    [t.name for t in g.topics if any(p -> matches_glob(path, p), t.paths)]

function _git_lines(repo_root, args)
    cmd = Cmd(`git -C $repo_root $args`)
    out = IOBuffer(); err = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
    proc.exitcode == 0 ||
        error("`git $(join(args, ' '))` failed in $repo_root:\n$(String(take!(err)))")
    return filter!(!isempty, split(String(take!(out)), '\n'))
end

"""
    changed_paths(repo_root, ref) -> Vector{String}

Repository-relative paths changed on this branch since its merge base with
`ref` (`git diff --name-only ref...HEAD`), plus uncommitted changes to
tracked files and untracked files that are not ignored. A renamed file
contributes its old and its new path.
"""
function changed_paths(repo_root::AbstractString, ref::AbstractString)
    paths = Set{String}()
    # Without rename detection a moved file lists both its old and new path.
    union!(paths, _git_lines(repo_root, ["diff", "--name-only", "--no-renames", "$(ref)...HEAD"]))
    union!(paths, _git_lines(repo_root, ["diff", "--name-only", "--no-renames", "HEAD"]))
    union!(paths, _git_lines(repo_root, ["ls-files", "--others", "--exclude-standard"]))
    return sort!(collect(paths))
end

# Test entries (paths relative to test/) of the default tiers.
function test_entries(test_root::AbstractString)
    core = joinpath(test_root, "core")
    entries = isdir(core) ? ["core/" * f for f in readdir(core) if endswith(f, ".jl")] : String[]
    isfile(joinpath(test_root, "regridding", "runtests.jl")) && push!(entries, "regridding/runtests.jl")
    return sort!(entries)
end

"""
    select_changed(groups, test_root, paths) -> (; files, topics, full, notes)

Map changed repository paths to test entries. A changed test entry selects
itself, and a file of `regridding/` selects `regridding/runtests.jl`. A path
matching a topic's `paths` selects the topic. Another changed file under
`test/` that no topic maps (a helper or fixture) selects every test entry that
mentions its file name. A `src/` file that no topic maps sets `full`: the
caller runs the full default tiers.
"""
function select_changed(g::Groups, test_root::AbstractString, paths::AbstractVector{<:AbstractString})
    files = Set{String}()
    topics = Set{String}()
    notes = String[]
    full = false
    entries = test_entries(test_root)
    unmapped = String[]
    for path in paths
        names = topics_for_path(g, path)
        union!(topics, names)
        if startswith(path, "test/")
            rel = path[6:end]
            if rel in entries
                push!(files, rel); continue
            elseif startswith(rel, "regridding/") && "regridding/runtests.jl" in entries
                push!(files, "regridding/runtests.jl"); continue
            end
            isempty(names) || continue
            base = basename(rel)
            users = [e for e in entries if occursin(base, read(joinpath(test_root, e), String))]
            union!(files, users)
            isempty(users) && push!(unmapped, path)
        elseif isempty(names)
            if startswith(path, "src/")
                full = true
                push!(notes, "$path maps to no test topic: running the full default tiers")
            else
                push!(unmapped, path)
            end
        end
    end
    isempty(unmapped) || push!(notes, "changed files that select no tests: " * join(unmapped, ", "))
    return (; files, topics, full, notes)
end

# Repository-relative paths of the files under `repo_root`, hidden folders skipped.
function repository_files(repo_root::AbstractString)
    files = String[]
    for (dir, subdirs, names) in walkdir(repo_root)
        filter!(s -> !startswith(s, ".") && s != "node_modules", subdirs)
        for n in names
            push!(files, relpath(joinpath(dir, n), repo_root))
        end
    end
    return files
end

"""
    validate_groups(groups, test_root, repo_root) -> Vector{String}

Problems with the groups: a default-tier test entry in no topic or in several,
a topic or quick entry that does not exist, a `src/` folder none of whose files
a topic maps, and a path glob that matches no file of the repository.
Folders and files are read from disk, so the check also runs outside git.
"""
function validate_groups(g::Groups, test_root::AbstractString, repo_root::AbstractString)
    problems = String[]
    entries = test_entries(test_root)
    owners = Dict{String, Vector{String}}()
    for t in g.topics
        occursin(r"^[a-z][a-z0-9_]*$", t.name) ||
            push!(problems, "topic name '$(t.name)' is not lower_snake_case")
        isempty(t.files) && push!(problems, "topic '$(t.name)' has no files")
        isempty(t.paths) && push!(problems, "topic '$(t.name)' has no paths")
        for f in t.files
            push!(get!(owners, f, String[]), t.name)
            f in entries || push!(problems, "topic '$(t.name)' lists '$f', which is not a test entry of core/ or regridding/")
        end
        length(unique(t.files)) == length(t.files) || push!(problems, "topic '$(t.name)' lists a file twice")
    end
    for e in entries
        n = length(get(owners, e, String[]))
        n == 0 && push!(problems, "'$e' is in no topic")
        n > 1 && push!(problems, "'$e' is in several topics: $(join(owners[e], ", "))")
    end
    for q in g.quick
        q in entries || push!(problems, "quick entry '$q' is not a test entry of core/ or regridding/")
    end
    length(unique(g.quick)) == length(g.quick) || push!(problems, "the quick set lists a file twice")
    files = repository_files(repo_root)
    # Each alternative of a `{a,b}` group must match a file: a misspelt
    # alternative would otherwise hide behind the ones that match.
    for t in g.topics, p in t.paths, alt in expand_braces(p)
        any(f -> matches_glob(f, alt), files) ||
            push!(problems, alt == p ?
                "topic '$(t.name)' path '$p' matches no file of the repository" :
                "topic '$(t.name)' path '$p': alternative '$alt' matches no file of the repository")
    end
    src_dirs = Dict{String, Bool}()
    for f in files
        startswith(f, "src/") && endswith(f, ".jl") || continue
        d = dirname(f)
        src_dirs[d] = get(src_dirs, d, false) || !isempty(topics_for_path(g, f))
    end
    for (d, mapped) in sort!(collect(src_dirs); by = first)
        mapped || push!(problems, "no topic maps a file of $d/")
    end
    return problems
end

end # module TestGroups
