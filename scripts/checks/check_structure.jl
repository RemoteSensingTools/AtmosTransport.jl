#!/usr/bin/env julia
# Static structure ratchet. Adapted from the cleanup inventory's JuliaSyntax
# include-tree walker; never includes or loads production code.
# Usage: julia --startup-file=no --project=test scripts/checks/check_structure.jl
#        [--root=<checkout>] [--base=<git-ref>] [--write-code-map]
module StructureCheck

using JuliaSyntax
using JuliaSyntax: SyntaxNode, kind, children, source_line, @K_str
using SHA, TOML

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SNAPSHOTS = Dict("duplicate" => "test/structure/duplicate_allowlist.toml",
                       "clone" => "test/structure/clone_allowlist.toml",
                       "private_use" => "test/structure/private_use_allowlist.toml",
                       "docstring" => "test/structure/docstring_allowlist.toml",
                       "noncore" => "test/structure/noncore_allowlist.toml")
const FOLDER_SIZES = "test/structure/folder_size_allowlist.toml"
const MIRRORS = "test/structure/mirrors.toml"
const MIGRATIONS = "test/structure/migrations.toml"
const CODE_MAP = "docs/src/development/code_map.md"
const README_SECTIONS = ("Depends on", "Used by", "Shared helpers owned here",
                         "Forward<->adjoint mirrors")

struct Definition
    file::String
    mod::String
    name::String
    line::Int
    qualified::Bool
    callable::Bool
    signature::String
    exact::Vector{String}
    shape::Set{String}
end

Base.@kwdef mutable struct Inventory
    root::String
    files::Dict{String,String} = Dict{String,String}()
    order::Vector{String} = String[]
    definitions::Vector{Definition} = Definition[]
    types::Set{Tuple{String,String}} = Set{Tuple{String,String}}()
    imports::Vector{NamedTuple} = NamedTuple[]
    references::Vector{NamedTuple} = NamedTuple[]
    identifiers::Dict{Tuple{String,String,String},Int} = Dict{Tuple{String,String,String},Int}()
    declared::Dict{String,Set{String}} = Dict{String,Set{String}}()
    exports::Dict{String,Set{String}} = Dict{String,Set{String}}()
    module_files::Dict{String,String} = Dict{String,String}()
    homes::Dict{Tuple{String,String},Set{String}} = Dict{Tuple{String,String},Set{String}}()
    includes::Set{Tuple{String,String}} = Set{Tuple{String,String}}()
    docs::Dict{String,String} = Dict{String,String}()
    errors::Vector{String} = String[]
end

record_home!(inv, mod, name, file) = push!(get!(inv.homes, (mod, name), Set{String}()), file)

node_text(n) = String(JuliaSyntax.sourcetext(n))
is_leaf(n) = (cs = children(n); cs === nothing || isempty(cs))
numchildren(n) = is_leaf(n) ? 0 : length(children(n))
identifier(n) = kind(n) in (K"Identifier", K"MacroName") ? string(n.val) : ""

function import_name(n)
    kind(n) == K"importpath" && return join(import_name.(children(n)), '.')
    return strip(node_text(n))
end

function dotchain(n)
    kind(n) == K"Identifier" && return [string(n.val)]
    if kind(n) == K"." && numchildren(n) == 2
        a, b = children(n)
        kind(b) == K"quote" && numchildren(b) == 1 && (b = only(children(b)))
        isempty(identifier(b)) && return String[]
        return vcat(dotchain(a), identifier(b))
    end
    return String[]
end

function signature_name(n)
    k = kind(n)
    k == K"call" && kind(first(children(n))) == K"::" && return "" # callable object
    k in (K"where", K"::", K"curly", K"call") && return signature_name(first(children(n)))
    k == K"Identifier" && return string(n.val)
    k == K"." && return join(dotchain(n), ".")
    return ""  # anonymous function / callable object
end

function callable_signature(n)
    kind(n) in (K"where", K"::") && return callable_signature(first(children(n)))
    return kind(n) == K"call" && kind(first(children(n))) == K"::"
end

function token_streams(text)
    exact, shape = String[], String[]
    for t in JuliaSyntax.tokenize(text)
        k = kind(t)
        k in (K"Whitespace", K"NewlineWs", K"Comment") && continue
        s = JuliaSyntax.untokenize(t, text)
        push!(exact, s)
        push!(shape, k == K"Identifier" ? "ID" : JuliaSyntax.is_literal(k) ? "LIT" : s)
    end
    return exact, Set(shape)
end

function record_definition!(inv, node, file, mod)
    sig, body = first(children(node)), last(children(node))
    name = signature_name(sig)
    callable = callable_signature(sig)
    callable && (name = "(callable)")
    isempty(name) && return
    # Use the AST body, not all lines after the signature's first newline:
    # this also handles multiline signatures and one-line definitions.
    exact, shape = numchildren(node) > 1 ? token_streams(node_text(body)) : (String[], Set{String}())
    signature = bytes2hex(sha256(join(first(token_streams(node_text(sig))), '\0')))[1:16]
    push!(inv.definitions, Definition(file, mod, name, source_line(node), occursin('.', name), callable, signature, exact, shape))
    callable || occursin('.', name) || record_home!(inv, mod, name, file)
end

# JuliaSyntax 0.4 (needed by JET on Julia 1.10) represents short methods as
# assignments; 1.x represents both forms with kind `function`.
function function_signature(n)
    kind(n) in (K"where", K"::") && return function_signature(first(children(n)))
    return kind(n) == K"call"
end

function record_import!(inv, node, file, mod)
    for c in children(node)
        p = kind(c) in (K":", K"as") ? first(children(c)) : c
        kind(p) == K"importpath" || continue
        names = [identifier(x) for x in children(p) if !isempty(identifier(x))]
        dots = count(x -> kind(x) == K".", children(p))
        bindings = Pair{String,String}[]
        if kind(c) == K":"
            for binding in children(c)[2:end]
                original = kind(binding) == K"as" ? first(children(binding)) : binding
                alias = kind(binding) == K"as" ? last(children(binding)) : binding
                push!(bindings, import_name(original) => import_name(alias))
            end
        end
        alias = kind(c) == K"as" ? import_name(last(children(c))) : ""
        push!(inv.imports, (; file, mod, names, dots, bindings, alias,
                            extend = kind(node) == K"import", line = source_line(node)))
    end
end

function walk_file!(inv, path, mod)
    path = normpath(path)
    file = relpath(path, inv.root)
    if haskey(inv.files, file)
        push!(inv.errors, "$file included more than once")
        return
    end
    inv.files[file] = mod
    if !isfile(path)
        push!(inv.errors, "missing include: $file")
        return
    end
    tree = try
        # Fix the grammar across Julia versions, including `public` declarations
        # that this static check must recognize when run on Julia 1.10.
        JuliaSyntax.parseall(SyntaxNode, read(path, String); filename = path,
                             version = v"1.12", ignore_warnings = true)
    catch e
        push!(inv.errors, "$file: $(sprint(showerror, e))")
        return
    end
    walk_node!(inv, tree, file, mod, true)
end

function walk_node!(inv, node, file, mod, top)
    k = kind(node)
    if k == K"module"
        name = identifier(first(children(node)))
        mod = isempty(mod) ? name : "$mod.$name"
        mod in inv.order || push!(inv.order, mod)
        inv.files[file] = mod
        inv.module_files[mod] = file
        for c in children(node)[2:end]
            walk_node!(inv, c, file, mod, true)
        end
        return
    elseif k == K"Identifier"
        get!(inv.identifiers, (file, mod, identifier(node)), source_line(node))
        return
    elseif k in (K"using", K"import")
        record_import!(inv, node, file, mod)
        return
    elseif k in (K"export", K"public")
        for c in children(node)
            name = node_text(c)
            push!(get!(inv.declared, mod, Set{String}()), name)
            k == K"export" && push!(get!(inv.exports, mod, Set{String}()), name)
        end
        return
    elseif k == K"call" && numchildren(node) >= 2 && identifier(first(children(node))) == "include"
        arg = children(node)[2]
        if kind(arg) == K"string" && numchildren(arg) == 1 && kind(only(children(arg))) == K"String"
            path = normpath(joinpath(inv.root, dirname(file), string(only(children(arg)).val)))
            push!(inv.includes, (file, relpath(path, inv.root)))
            walk_file!(inv, path, mod)
        else
            push!(inv.errors, "$file:$(source_line(node)): nonliteral include")
        end
        return
    elseif k == K"function" || (k == K"=" && function_signature(first(children(node))))
        top && record_definition!(inv, node, file, mod)
        top = false
    elseif k in (K"struct", K"abstract", K"primitive")
        sig = first(children(node))
        while kind(sig) in (K"<:", K"curly")
            sig = first(children(sig))
        end
        push!(inv.types, (mod, identifier(sig)))
        record_home!(inv, mod, identifier(sig), file)
    elseif top && k == K"=" && kind(first(children(node))) == K"Identifier"
        record_home!(inv, mod, identifier(first(children(node))), file)
    elseif k == K"." && !is_leaf(node)
        chain = dotchain(node)
        length(chain) >= 2 && push!(inv.references, (; file, mod, chain, line = source_line(node)))
        # Also descend so calls/qualified names in array indices aren't lost.
    end
    is_leaf(node) && return
    # A standalone triple-quoted string in a module/block was not attached by
    # Julia's doc macro. Strings inside assignments/calls/returns are excluded.
    if top && k in (K"toplevel", K"block")
        for c in children(node)
            if kind(c) == K"string" && startswith(strip(node_text(c)), "\"\"\"")
                id = file * "|" * bytes2hex(sha256(strip(node_text(c))))
                inv.docs[id] = "$file:$(source_line(c)): unattached docstring"
            end
        end
    end
    keep_top = top && k in (K"toplevel", K"block", K"if", K"elseif", K"else", K"macrocall", K"doc", K"const")
    for c in children(node)
        walk_node!(inv, c, file, mod, keep_top)
    end
end

function inventory(root)
    inv = Inventory(; root = abspath(root))
    walk_file!(inv, joinpath(root, "src", "AtmosTransport.jl"), "")
    ext = joinpath(root, "ext")
    if isdir(ext)
        for f in sort(readdir(ext; join = true))
            endswith(f, ".jl") && walk_file!(inv, f, "")
        end
    end
    for folder in ("src", "ext")
        isdir(joinpath(root, folder)) || continue
        for (dir, _, files) in walkdir(joinpath(root, folder)), f in files
            endswith(f, ".jl") || continue
            file = relpath(joinpath(dir, f), root)
            haskey(inv.files, file) || push!(inv.errors, "unreached source: $file")
        end
    end
    return inv
end

# Resolve relative paths and module aliases in lexical parent scopes. Internal
# bindings imported through a facade are followed to their defining module.
parent_module(m) = contains(m, '.') ? first(rsplit(m, '.'; limit = 2)) : ""

function module_path(inv, mod, names, dots)
    isempty(names) && return ""
    # Absolute package imports cannot resolve to a nested module with the same
    # leaf name. Dependency imports must start from the package root.
    # Relative imports still resolve through the ordinary lexical scope below.
    project = joinpath(inv.root, "Project.toml")
    if dots == 0 && isfile(project) &&
       haskey(get(TOML.parsefile(project), "deps", Dict()), first(names))
        return ""
    end
    if dots > 0
        parts = String.(split(mod, '.'))
        base = parts[1:max(0, length(parts) - dots + 1)]
        for i in length(names):-1:1
            candidate = join(vcat(base, names[1:i]), '.')
            candidate in inv.order && return candidate
        end
    end
    for i in length(names):-1:1
        candidate = join(names[1:i], '.')
        candidate in inv.order && return candidate
        scope = mod
        while !isempty(scope)
            "$scope.$candidate" in inv.order && return "$scope.$candidate"
            scope = parent_module(scope)
        end
        matches = filter(m -> last(split(m, '.')) == candidate, inv.order)
        length(matches) == 1 && return only(matches)
    end
    if dots > 0
        parts = String.(split(mod, '.'))
        base = join(parts[1:max(0, length(parts) - dots + 1)], '.')
        base in inv.order && return base # import ..f extends a parent binding
    end
    return ""
end

function resolve_imports(inv)
    bindings = Dict{Tuple{String,String},Tuple{String,String}}()
    extensions = Set{Tuple{String,String}}()
    imports = NamedTuple[]
    # Facades may acquire their bindings after a consumer is parsed. Resolve
    # to a fixed point so imports through aliases use the final owner too.
    for _ in 1:(length(inv.imports) + 1)
        previous = copy(bindings)
        empty!(imports)
        for imp in inv.imports
            # External imports extend an existing generic too; they need no
            # internal dependency edge, but must not become duplicate helpers.
            if imp.extend
                for (_, alias) in imp.bindings
                    push!(extensions, (imp.mod, alias))
                end
                isempty(imp.bindings) && length(imp.names) > 1 &&
                    push!(extensions, (imp.mod, isempty(imp.alias) ? last(imp.names) : imp.alias))
            end
            scope = imp.mod
            for _ in 2:imp.dots
                scope = parent_module(scope)
            end
            # An explicit relative module path resolves in its own scope before
            # inherited aliases. Inversion.Observations shadows the package's
            # Observations module; both are valid owners.
            direct = imp.dots > 0 ? join([split(scope, '.'); imp.names], '.') : ""
            direct in inv.order || (direct = "")
            target, member = isempty(direct) ?
                reference_target(inv, bindings, (; mod = scope, chain = imp.names)) : (direct, "")
            if isempty(target)
                target = module_path(inv, imp.mod, imp.names, imp.dots)
                member = !isempty(target) && last(imp.names) != last(split(target, '.')) ? last(imp.names) : ""
            end
            isempty(target) && continue
            push!(imports, (; imp..., target, member))
            if isempty(imp.bindings)
                alias = isempty(imp.alias) ? last(imp.names) : imp.alias
                bindings[(imp.mod, alias)] = (target, "")
                # import A.f (as g) extends the generic too.
                if !isempty(member)
                    bindings[(imp.mod, alias)] = (target, member)
                    imp.extend && push!(extensions, (imp.mod, alias))
                elseif !imp.extend
                    for name in get(inv.exports, target, Set{String}())
                        bindings[(imp.mod, name)] = (target, name)
                    end
                end
            else
                for (name, alias) in imp.bindings
                    child = "$target.$name"
                    bindings[(imp.mod, alias)] = child in inv.order ? (child, "") : (target, name)
                    imp.extend && push!(extensions, (imp.mod, alias))
                end
            end
        end
        bindings == previous && break
    end
    return bindings, extensions, imports
end

function binding_owner(bindings, target, name)
    seen = Set{Tuple{String,String}}()
    while !isempty(name) && haskey(bindings, (target, name)) && !((target, name) in seen)
        push!(seen, (target, name))
        target, name = bindings[(target, name)]
    end
    return target, name
end

function reference_target(inv, bindings, ref)
    chain = ref.chain
    scope = ref.mod
    target, name = "", ""
    while !isempty(scope)
        if haskey(bindings, (scope, first(chain)))
            target, name = bindings[(scope, first(chain))]
            break
        end
        scope = parent_module(scope)
    end
    if isempty(target)
        first(chain) == first(split(ref.mod, '.')) && (target = first(chain))
        # Direct module names, including references to modules loaded later.
        isempty(target) && (target = module_path(inv, ref.mod, [first(chain)], 0))
    end
    isempty(target) && return "", ""
    target, name = binding_owner(bindings, target, name)
    for part in chain[2:end]
        isempty(name) || return "", "" # a property of a value, not a module reference
        if "$target.$part" in inv.order
            target = "$target.$part"
        else
            target, name = binding_owner(bindings, target, part)
        end
    end
    return target, name
end

function shared_readme_names(inv)
    names = Dict{String,Set{String}}()
    for (file, mod) in inv.files
        path = joinpath(inv.root, dirname(file), "README.md")
        isfile(path) || continue
        for line in split(read(path, String), '\n')
            startswith(strip(line), "# shared:") || continue
            for m in eachmatch(r"_[A-Za-z0-9_!]+", split(line, "# shared:"; limit = 2)[2])
                push!(get!(names, mod, Set{String}()), m.match)
            end
        end
    end
    return names
end

function findings(inv)
    bindings, extensions, imports = resolve_imports(inv)
    edges = Set{Tuple{String,String}}()
    folder_edges = Set{Tuple{String,String}}()
    private = Dict{String,String}()
    noncore = Dict{String,String}()
    shared = shared_readme_names(inv)
    function folder_use!(file, owner, name)
        targets = get(inv.homes, (owner, name), Set{String}())
        isempty(targets) && haskey(inv.module_files, owner) && (targets = Set([inv.module_files[owner]]))
        for target in targets
            a, b = dirname(file), dirname(target)
            a == b || push!(folder_edges, (a, b))
        end
    end
    function use!(file, mod, owner, name, line)
        isempty(owner) || owner == mod || push!(edges, (mod, owner))
        isempty(owner) || folder_use!(file, owner, name)
        if !is_noncore(inv, mod) && is_noncore(inv, owner)
            id = join((file, mod, owner), '|')
            noncore[id] = "$file:$line: core $mod depends on non-core $owner"
        end
        if owner != mod && startswith(name, "_") &&
           !(name in get(inv.declared, owner, Set{String}())) &&
           !(name in get(shared, owner, Set{String}()))
            id = join((file, mod, owner, name), '|')
            private[id] = "$file:$line: $mod uses $owner.$name"
        end
    end
    for imp in imports
        use!(imp.file, imp.mod, imp.target, "", imp.line)
        for (name, _) in imp.bindings
            owner, name = binding_owner(bindings, imp.target, name)
            use!(imp.file, imp.mod, owner, name, imp.line)
        end
        if isempty(imp.bindings) && !isempty(imp.member)
            owner, name = binding_owner(bindings, imp.target, imp.member)
            use!(imp.file, imp.mod, owner, name, imp.line)
        end
    end
    for ref in inv.references
        target, name = reference_target(inv, bindings, ref)
        isempty(target) || use!(ref.file, ref.mod, target, name, ref.line)
    end
    for ((file, mod, name), line) in inv.identifiers
        if !haskey(bindings, (mod, name))
            haskey(inv.homes, (mod, name)) && folder_use!(file, mod, name)
            continue
        end
        owner, original = binding_owner(bindings, mod, name)
        use!(file, mod, owner, original, line)
    end
    owners = Dict{String,Set{String}}()
    clones = Dict{String,Vector{Definition}}()
    for d in inv.definitions
        if !d.callable && !d.qualified && !((d.mod, d.name) in extensions) && !((d.mod, d.name) in inv.types)
            push!(get!(owners, d.name, Set{String}()), d.mod)
        end
        length(d.exact) >= 25 || continue
        digest = bytes2hex(sha256(join(d.exact, '\0')))
        push!(get!(clones, digest, Definition[]), d)
    end
    duplicates = Dict{String,String}()
    for (name, mods) in owners
        length(mods) > 1 || continue
        for mod in mods
            duplicates["$name|$mod"] = "$name is independently defined in $mod"
        end
    end
    clone_sites = Dict{String,String}()
    for (digest, defs) in clones
        length(Set(d.file for d in defs)) > 1 || continue
        for d in defs
            id = join((digest, d.file, d.mod, d.name, d.signature), '|')
            clone_sites[id] = "$(d.file):$(d.line): $(d.name) shares a $(length(d.exact))-token body"
        end
    end
    # Jaccard on normalized token sets is deliberately advisory. Pair each
    # definition only across files, with at least 100 body tokens on both sides.
    large = filter(d -> length(d.exact) >= 100, inv.definitions)
    similar = String[]
    for i in eachindex(large), j in (i + 1):length(large)
        a, b = large[i], large[j]
        a.file == b.file && continue
        a.exact == b.exact && continue
        score = length(intersect(a.shape, b.shape)) / length(union(a.shape, b.shape))
        score >= 0.9 && push!(similar, "$(a.file):$(a.line) $(a.name) ~ $(b.file):$(b.line) $(b.name) (Jaccard $(round(score; digits=3)))")
    end
    for (a, b) in inv.includes
        dirname(a) == dirname(b) || push!(folder_edges, (dirname(a), dirname(b)))
    end
    return (; edges, folder_edges, duplicate = duplicates, clone = clone_sites, private_use = private,
              docstring = inv.docs, noncore, similar)
end

is_noncore(mod) = any(m -> mod == m || startswith(mod, "$m."),
    ("AtmosTransport.Visualization", "AtmosTransport.Analysis", "AtmosTransport.DataDownloads",
     "AtmosTransportMakieExt", "AtmosTransport.Extras"))

# Extras keeps its historical module names; future modules under that path
# must obey the same boundary even when their names do not mention Extras.
is_noncore(inv, mod) = is_noncore(mod) ||
    startswith(get(inv.module_files, mod, ""), "src/Extras/")

function allowlist(root, file)
    raw = TOML.parsefile(joinpath(root, file))
    entries = get(raw, "allow", Dict{String,Any}())
    all(v -> v isa String && !isempty(strip(v)), values(entries)) || error("$file: every entry needs a reason")
    return entries
end

function has_cycle(edges)
    graph = Dict{String,Set{String}}()
    for (a, b) in edges
        # Parent/child wiring isn't a dependency cycle (root re-exports children).
        startswith(b, "$a.") || push!(get!(graph, a, Set{String}()), b)
    end
    active, done = Set{String}(), Set{String}()
    function visit(a)
        a in active && return true
        a in done && return false
        push!(active, a)
        any(visit, get(graph, a, Set{String}())) && return true
        delete!(active, a); push!(done, a)
        return false
    end
    return any(visit, keys(graph))
end

# Kinds of the `|`-separated fields of each snapshot's allowance ids. Migration
# rows rename only module, folder and file fields, never helper names or digests.
const ID_FIELDS = Dict(SNAPSHOTS["duplicate"] => (:name, :module),
                       SNAPSHOTS["clone"] => (:digest, :file, :module, :name, :digest),
                       SNAPSHOTS["private_use"] => (:file, :module, :module, :name),
                       SNAPSHOTS["docstring"] => (:file, :digest),
                       SNAPSHOTS["noncore"] => (:file, :module, :module),
                       FOLDER_SIZES => (:folder,))

# Reviewed `[[migration]]` rows (from, to, reason) of test/structure/migrations.toml.
function migration_rows!(errors, raw)
    rows = NamedTuple{(:from, :to),Tuple{String,Vector{String}}}[]
    for row in get(raw, "migration", [])
        from, to, reason = row isa AbstractDict ? (get(row, k, "") for k in ("from", "to", "reason")) : ("", "", "")
        if !(from isa String && !isempty(from) && to isa Vector && !isempty(to) &&
             all(t -> t isa String && !isempty(t), to) && allunique(to))
            push!(errors, "migration rows need a from identity and distinct to identities: $row")
        elseif !(reason isa String && !isempty(strip(reason)))
            push!(errors, "migration $from needs a reason")
        else
            push!(rows, (; from, to = String.(to)))
        end
    end
    return rows
end

# A row moves one base module, folder or Julia file to current identities of the same kind.
function row_kind!(errors, row, base, head)
    for kind in (:module, :folder, :file)
        row.from in base[kind] || continue
        wrong = filter(t -> !(t in head[kind]), row.to)
        isempty(wrong) && return kind
        push!(errors, "migration $(row.from): $(join(wrong, ", ")) is not a current $kind")
        return nothing
    end
    push!(errors, "migration $(row.from) is not a module, folder or Julia file of the base")
    return nothing
end

function substitutions(rows)
    subs = Dict(kind => Dict{String,Vector{String}}() for kind in (:module, :folder, :file))
    foreach(row -> append!(get!(subs[row.kind], row.from, String[]), row.to), rows)
    return subs
end

# A field names its row's targets instead of `from`; a file also follows its folder.
function substitute(subs, kind, part)
    kind == :file || return get(get(subs, kind, Dict()), part, [part])
    haskey(subs[:file], part) && return subs[:file][part]
    folder = get(subs[:folder], dirname(part), nothing)
    return folder === nothing ? [part] : [t * "/" * basename(part) for t in folder]
end

# Current id => base ids it may come from (itself, or its migrated forms).
function origins(table, subs, fields)
    found = Dict{String,Vector{String}}()
    for id in keys(table), parts in Iterators.product(
            (substitute(subs, get(fields, i, :name), p) for (i, p) in enumerate(split(id, '|')))...)
        push!(get!(found, join(parts, '|'), String[]), id)
    end
    return found
end

# Current ids that cannot each be traced to a distinct base id (augmenting-path
# matching): an allowance or exception moves to at most one place, never copies.
function unmatched(ids, origin)
    owner = Dict{String,String}()
    claim(id, seen) = any(get(origin, id, String[])) do source
        source in seen && return false
        push!(seen, source)
        (!haskey(owner, source) || claim(owner[source], seen)) && (owner[source] = id; true)
    end
    return sort!([id for id in ids if !claim(id, Set{String}())])
end

# Each target receives from's base budget and may use the row's other targets
# (the owner's own use of itself is substituted); a base use of `from` may use
# any target; `from` keeps its budget and users only if it is also a target.
function migrated_budgets(budgets, subs)
    swap(id) = get(subs, id, [id])
    moved = Dict{String,Vector{String}}()
    for (owner, deps) in budgets, target in swap(owner)
        append!(get!(moved, target, String[]), filter(!=(target), reduce(vcat, swap.([deps; owner]))))
    end
    return moved
end

const NO_MIGRATIONS = substitutions([])

function allowance_growth(before, after, file, subs = NO_MIGRATIONS)
    before == after && return String[]
    if file == "src/LAYERS.toml"
        growth = String[]
        for (table, kind) in (("modules", :module), ("folders", :folder))
            budgets = migrated_budgets(get(before, table, Dict()), subs[kind])
            append!(growth, ["new $table allowance $mod -> $dep" for (mod, deps) in get(after, table, Dict())
                    for dep in setdiff(deps, get(budgets, mod, String[]))])
        end
        origin = origins(get(before, "folder_order_allow", Dict()), subs, (:folder, :folder))
        return [growth; "new folder order allowance " .* unmatched(keys(get(after, "folder_order_allow", Dict())), origin)]
    end
    old, new = get(before, "allow", Dict()), get(after, "allow", Dict())
    origin = origins(old, subs, get(ID_FIELDS, file, ()))
    file == FOLDER_SIZES || return "new or copied $file entry " .* unmatched(keys(new), origin)
    # The ceilings drawn from one base ceiling may not exceed it in sum.
    raised = [folder for folder in keys(new) if !haskey(origin, folder)]
    for (folder, entry) in old
        drawn = sum(new[f]["bytes"] for f in keys(new) if folder in get(origin, f, String[]); init = 0)
        drawn > entry["bytes"] && push!(raised, folder)
    end
    return "raised folder budget " .* sort!(unique(raised))
end

# Allowances only shrink relative to the base, except for transfers declared by
# migration rows. A row already in the base was applied there and is inert: a
# warning asks to delete it, and a PR that edits migrations.toml must do so. A
# row target that discharges no growth fails as unused or redundant.
function ratchet_errors(snapshots, migrations, applied = Dict{String,Any}();
                        files = (base = Set{String}(), head = Set{String}()), warnings = String[])
    errors = String[]
    layers = [(b, a) for (f, b, a) in snapshots if f == "src/LAYERS.toml"]
    before, after = isempty(layers) ? (Dict(), Dict()) : only(layers)
    ids(layers, paths) = Dict(:module => keys(get(layers, "modules", Dict())),
                              :folder => keys(get(layers, "folders", Dict())), :file => paths)
    done = [(row.from, sort(row.to)) for row in migration_rows!(String[], applied)]
    rows = []
    for row in migration_rows!(errors, migrations)
        if (row.from, sort(row.to)) in done
            push!(migrations == applied ? warnings : errors,
                  "migration $(row.from) was applied by the base; delete the row")
        elseif (kind = row_kind!(errors, row, ids(before, files.base), ids(after, files.head))) !== nothing
            push!(rows, (; row..., kind))
        end
    end
    growth(list) = unique(reduce(vcat, (allowance_growth(b, a, f, substitutions(list))
                                        for (f, b, a) in snapshots); init = String[]))
    found = growth(rows)
    append!(errors, "allowlists only shrink: " .* found)
    without(list, i, t) = [j == i ? (; r..., to = filter(!=(t), r.to)) : r for (j, r) in enumerate(list)]
    for (i, row) in enumerate(rows), target in row.to
        fewer = without(rows, i, target)
        issubset(growth(fewer), found) || continue
        twin = findfirst(j -> j != i && target in rows[j].to &&
                              !issubset(growth(without(fewer, j, target)), found), eachindex(rows))
        push!(errors, twin === nothing ? "migration $(row.from) -> $target is unused; remove it" :
              "migration $(row.from) -> $target is redundant with migration $(rows[twin].from) -> $target; keep one")
    end
    return errors
end

# Captured stdout of a git command, or nothing on failure. Fresh buffers: Julia
# 1.10 can retain a process-backed IOBuffer's contents after take!.
function git_output(root, args)
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(`git -C $root $args`); stdout = out, stderr = devnull))
    return success(proc) ? String(take!(out)) : nothing
end

# Parsed `file` at the base commit; nothing if absent (first introduction) or unreadable.
function base_toml!(errors, root, commit, base, file)
    listing = git_output(root, `ls-tree $commit -- $file`)
    listing === nothing && (push!(errors, "cannot inspect $file at structure base '$base'"); return nothing)
    isempty(listing) && return nothing
    text = git_output(root, `show $commit\:$file`)
    text === nothing && (push!(errors, "cannot read $file at structure base '$base'"); return nothing)
    return TOML.parse(text)
end

julia_files(paths) = Set(String(p) for p in paths if endswith(p, ".jl"))

function check_ratchet!(errors, root, base, files; io = stdout)
    base === nothing && return
    commit = git_output(root, `rev-parse --verify --quiet --end-of-options $(base * "^{commit}")`)
    if commit === nothing
        branch = startswith(base, "origin/") ? base[8:end] : base
        push!(errors, "cannot validate structure base ref '$base'; fetch the base branch with " *
                      "git fetch --no-tags --depth=1 origin '+refs/heads/$branch:refs/remotes/origin/$branch', " *
                      "then set ATMOSTR_STRUCTURE_BASE=origin/$branch (or pass an existing commit)")
        return
    end
    commit = strip(commit)
    snapshots = []
    for file in files
        before = base_toml!(errors, root, commit, base, file)
        path = joinpath(root, file)
        before === nothing || push!(snapshots, (file, before, isfile(path) ? TOML.parsefile(path) : Dict()))
    end
    base_files = julia_files(split(something(git_output(root, `ls-tree -r --name-only $commit -- src ext`), ""), '\n'))
    head_files = julia_files(relpath(joinpath(dir, f), root) for top in ("src", "ext") if isdir(joinpath(root, top))
                             for (dir, _, names) in walkdir(joinpath(root, top)) for f in names)
    path = joinpath(root, MIGRATIONS)
    applied = something(base_toml!(errors, root, commit, base, MIGRATIONS), Dict())
    warnings = String[]
    append!(errors, ratchet_errors(snapshots, isfile(path) ? TOML.parsefile(path) : Dict(), applied;
                                   files = (base = base_files, head = head_files), warnings))
    foreach(w -> println(io, "warning: ", w), warnings)
end

# A folder owns its immediate Julia files; subfolders have their own budgets.
# This keeps nested conceptual units independent rather than charging them twice.
function source_folders(inv)
    folders = Dict{String,Vector{String}}()
    for file in keys(inv.files)
        push!(get!(folders, dirname(file), String[]), file)
    end
    foreach(sort!, values(folders))
    return folders
end

folder_bytes(inv, files) = sum(f -> filesize(joinpath(inv.root, f)), files; init = 0)

# Shared-module folders whose wiring check_folders! permits explicitly (A1).
const A1_HOME = "src/Operators/Advection"
const A1_CHILDREN = A1_HOME .* ["/Reconstruction", "/Structured", "/CubedSphere"]

function check_folders!(errors, inv, found, layers; io = stdout)
    folders = source_folders(inv)
    order = get(layers, "folder_order", String[])
    positions = Dict(f => i for (i, f) in enumerate(order))
    length(order) == length(positions) || push!(errors, "T1 duplicate folder in folder_order")
    allowed = get(layers, "folders", Dict())
    exceptions = get(layers, "folder_order_allow", Dict())
    all(v -> v isa String && !isempty(strip(v)), values(exceptions)) ||
        push!(errors, "T1 folder order allowances need reasons")
    for folder in keys(folders)
        haskey(positions, folder) && haskey(allowed, folder) || push!(errors, "T1 undeclared folder: $folder")
    end
    needed = Set{String}()
    for (a, b) in sort!(collect(found.folder_edges))
        b in get(allowed, a, String[]) || push!(errors, "T1 new folder dependency: $a -> $b")
        # Module assembly may wire its children. A1 also permits its explicitly
        # named shared-module folders: fenced root types are used by the child
        # code, while root includes/Lin-Rood use child helpers. Sibling edges
        # still obey the ordinary order; no reverse exception is introduced.
        child_wiring = startswith(b, "$a/") && any(
            startswith(inv.files[child], inv.files[parent] * ".")
            for parent in get(folders, a, String[]), child in get(folders, b, String[]))
        shared_advection_wiring = a == A1_HOME && b in A1_CHILDREN &&
            all(inv.files[f] == "AtmosTransport.Operators.Advection"
                for f in get(folders, b, String[]))
        if !child_wiring && !shared_advection_wiring && get(positions, b, typemax(Int)) >= get(positions, a, 0)
            haskey(exceptions, "$a|$b") ? push!(needed, "$a|$b") : push!(errors, "T1 reference to later folder: $a -> $b")
        end
    end
    # An exception is kept only for a reverse edge the rules above would reject.
    for id in sort!(collect(setdiff(keys(exceptions), needed)))
        push!(errors, "T1 remove unneeded folder order allowance: $id")
    end
    println(io, "T1 folders: $(length(folders)) folders, $(length(found.folder_edges)) edges, $(length(exceptions)) existing order exceptions")
    sizes = TOML.parsefile(joinpath(inv.root, FOLDER_SIZES))["allow"]
    for (folder, entry) in sizes
        entry["bytes"] isa Integer && entry["bytes"] > 65_000 * 3.5 || push!(errors, "T5 invalid oversized budget: $folder")
        !isempty(strip(entry["reason"])) || push!(errors, "T5 folder allowance needs a reason: $folder")
        if !haskey(folders, folder) || folder_bytes(inv, folders[folder]) <= 65_000 * 3.5
            push!(errors, "T5 remove allowance for folder now within budget or removed: $folder")
        end
    end
    for (folder, files) in sort!(collect(folders))
        bytes = folder_bytes(inv, files)
        bytes <= 65_000 * 3.5 && continue
        if !haskey(sizes, folder) || bytes > sizes[folder]["bytes"]
            push!(errors, "T5 $folder exceeds 65000 tokens: $(round(bytes / 3.5; digits = 1)) ($bytes bytes)")
        elseif bytes < sizes[folder]["bytes"]
            push!(errors, "T5 shrink $folder allowance to current size: $bytes bytes")
        end
    end
    println(io, "T5 size: $(length(folders)) source folders, budget 65000 tokens (immediate Julia bytes / 3.5), $(length(sizes)) oversized allowances")
end

function mirror_pairs(root)
    raw = TOML.parsefile(joinpath(root, MIRRORS))
    return [(String(p["forward"]), String(p["adjoint"])) for p in raw["pairs"]]
end

function folder_public_names(inv)
    bindings, _, _ = resolve_imports(inv)
    names = Dict{String,Set{Tuple{String,String}}}()
    for (mod, declared) in inv.declared, name in declared
        owner, original = binding_owner(bindings, mod, name)
        files = get(inv.homes, (owner, original), Set{String}())
        if isempty(files)
            # Exported module bindings live at the child's entry point.
            entry = isempty(original) ? owner : "$owner.$original" in inv.order ? "$owner.$original" : owner
            haskey(inv.module_files, entry) || continue
            files = Set([inv.module_files[entry]])
        end
        for file in files
            push!(get!(names, dirname(file), Set{Tuple{String,String}}()), (owner, original))
        end
    end
    return names
end

function code_map(inv, found = findings(inv))
    io = IOBuffer()
    println(io, "# Code map\n")
    println(io, "Generated by `scripts/checks/check_structure.jl`; edit the source or mirror data, then run:\n")
    println(io, "```bash\nCUDA_VISIBLE_DEVICES= julia --startup-file=no --project=test scripts/checks/check_structure.jl --write-code-map\n```\n")
    println(io, "Read this map and the folder README before editing. Each folder owns its immediate Julia files; subfolders have separate budgets of 65,000 tokens (bytes / 3.5). Existing oversized folders must shrink.\n")
    println(io, "Dependencies include static imports, qualified references, shared-module name uses, and includes. They describe today's code; `src/LAYERS.toml` declares their allowed order and existing exceptions. Public counts are distinct resolved `(module, name)` pairs declared `public` or `export`, assigned to their definition folders (or the module entry folder if no definition is located); facade re-exports are counted at the owner. Mirrors come from `test/structure/mirrors.toml`, seeded from the cleanup inventory's units and helper index; an empty list means no recorded pair.\n")
    pairs = mirror_pairs(inv.root)
    public_names = folder_public_names(inv)
    for (folder, files) in sort!(collect(source_folders(inv)))
        bytes = folder_bytes(inv, files)
        deps = sort!([b for (a, b) in found.folder_edges if a == folder])
        users = sort!([a for (a, b) in found.folder_edges if b == folder])
        names = get(public_names, folder, Set{Tuple{String,String}}())
        println(io, "## `$folder`\n")
        println(io, "Size: **$bytes bytes / $(round(bytes / 3.5; digits = 1)) tokens**. Public names: **$(length(names))**.\n")
        println(io, "Depends on: ", isempty(deps) ? "none." : join(["`$d`" for d in deps], ", ") * ".", "\n")
        println(io, "Used by: ", isempty(users) ? "none." : join(["`$d`" for d in users], ", ") * ".", "\n")
        println(io, "Files:\n")
        for file in files
            println(io, "- `$(basename(file))` ($(filesize(joinpath(inv.root, file))) bytes)")
        end
        println(io, "\nForward ↔ adjoint mirrors:\n")
        local_pairs = filter(p -> dirname(p[1]) == folder || dirname(p[2]) == folder, pairs)
        if isempty(local_pairs)
            println(io, "None recorded.")
        else
            for (forward, adjoint) in sort!(local_pairs)
                println(io, "- `$forward` ↔ `$adjoint`")
            end
        end
        println(io)
    end
    return rstrip(String(take!(io))) * "\n"
end

function check_structure(root = ROOT; io = stdout, base = "origin/main", check_map = true)
    inv = inventory(root)
    found = findings(inv)
    errors = copy(inv.errors)
    layers = TOML.parsefile(joinpath(root, "src", "LAYERS.toml"))
    allowed = layers["modules"]
    for mod in inv.order
        haskey(allowed, mod) || push!(errors, "T1 undeclared module: $mod")
    end
    positions = Dict(m => i for (i, m) in enumerate(inv.order))
    for (a, b) in sort!(collect(found.edges))
        b in get(allowed, a, String[]) || push!(errors, "T1 new dependency: $a -> $b")
        # A parent's child wiring is allowed; a deferred qualified reference
        # to a later sibling is still forbidden even if LAYERS was edited.
        if !startswith(b, "$a.") && positions[b] > positions[a]
            push!(errors, "T1 reference to later module: $a -> $b")
        end
    end
    has_cycle(found.edges) && push!(errors, "T1 module dependency cycle")
    println(io, "T1 layers: $(length(inv.order)) modules, $(length(found.edges)) edges")
    for (topic, file) in sort!(collect(SNAPSHOTS))
        current = getproperty(found, Symbol(topic))
        allowed_entries = allowlist(root, file)
        new = setdiff(keys(current), keys(allowed_entries))
        for id in sort!(collect(new))
            push!(errors, "$topic: $(current[id]) [id: $id]")
        end
        stale = setdiff(keys(allowed_entries), keys(current))
        println(io, "$topic: $(length(current)) existing, $(length(new)) new, $(length(stale)) allowances can be removed")
    end
    println(io, "T3 warning: $(length(found.similar)) shape-similar pairs (advisory; first 10 below)")
    foreach(s -> println(io, "  warning: ", s), first(found.similar, 10))
    check_folders!(errors, inv, found, layers; io)
    for (forward, adjoint) in mirror_pairs(root)
        for file in (forward, adjoint)
            haskey(inv.files, file) || push!(errors, "code map mirror file is missing: $file")
        end
    end
    if check_map
        path = joinpath(root, CODE_MAP)
        if !isfile(path) || read(path, String) != code_map(inv, found)
            push!(errors, "code map is stale: run scripts/checks/check_structure.jl --write-code-map")
        end
    end
    missing = String[]
    for (dir, _, files) in walkdir(joinpath(root, "src"))
        "README.md" in files || continue
        text = read(joinpath(dir, "README.md"), String)
        headings = [strip(m.captures[1]) for m in eachmatch(r"(?m)^#{1,6}\s+(.+?)\s*$", text)]
        absent = filter(s -> !(s in headings), README_SECTIONS)
        isempty(absent) || push!(missing, "$(relpath(dir, root))/README.md: $(join(absent, ", "))")
    end
    println(io, "T6 README sections: $(length(missing)) incomplete (report only)")
    foreach(s -> println(io, "  missing: ", s), missing)
    check_ratchet!(errors, root, base, vcat(["src/LAYERS.toml", FOLDER_SIZES], collect(values(SNAPSHOTS))); io)
    foreach(e -> println(io, "ERROR: ", e), errors)
    println(io, isempty(errors) ? "structure check passed (existing debt reported)" : "structure check FAILED")
    return (; errors, inventory = inv, findings = found)
end

function main(args)
    root, base = ROOT, "origin/main"
    write_map = false
    for arg in args
        if startswith(arg, "--root=")
            root = abspath(split(arg, '='; limit = 2)[2])
        elseif startswith(arg, "--base=")
            base = split(arg, '='; limit = 2)[2]
        elseif arg == "--write-code-map"
            write_map = true
        else
            error("unknown option: $arg")
        end
    end
    if write_map
        path = joinpath(root, CODE_MAP)
        mkpath(dirname(path))
        write(path, code_map(inventory(root)))
        println("Wrote ", path)
    end
    return isempty(check_structure(root; base).errors) ? 0 : 1
end

end # module

(abspath(PROGRAM_FILE) == @__FILE__) && exit(StructureCheck.main(ARGS))
