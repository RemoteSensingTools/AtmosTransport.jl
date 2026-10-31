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
    # leaf name (e.g. CubedSphere.SphericalGeometry vs Preprocessing.CubedSphere).
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

# Compare tracked allowances with the selected base. P1's base has none; once
# landed, raising allowances in the same diff as a violation cannot hide it.
function relocate_x1_baseline(before, file)
    # One-way migration of P1 identities for X1's reviewed layout. This changes
    # only historical keys: no new digest, signature, helper or byte ceiling is
    # accepted. A base already using the X1 layout is compared without migration.
    extras(path) = replace(path, r"^src/(Analysis|Visualization|Downloads)(/|$)" => s"src/Extras/\1\2")
    if file == "src/LAYERS.toml"
        modules = get(before, "modules", Dict())
        haskey(modules, "AtmosTransport.Diagnostics") && return before
        haskey(modules, "AtmosTransport.Analysis") || return before
        migrated = deepcopy(before)
        mods = migrated["modules"]
        # The former Analysis dependencies now belong to core Diagnostics.
        # Analysis re-exports Diagnostics; the facade includes the new module.
        mods["AtmosTransport.Diagnostics"] = copy(modules["AtmosTransport.Analysis"])
        push!(mods["AtmosTransport.Analysis"], "AtmosTransport.Diagnostics")
        push!(mods["AtmosTransport"], "AtmosTransport.Diagnostics")
        if haskey(before, "folders")
            # SectionTimer moved to src so late Diagnostics can follow Models
            # without a new reverse-folder exception or a folder cycle.
            folder(path) = path == "src/Diagnostics" ? "src" : extras(path)
            migrated["folders"] = Dict(folder(a) => unique([folder(b) for b in deps if folder(b) != folder(a)])
                                       for (a, deps) in before["folders"] if a != "src/Diagnostics")
            migrated["folders"]["src/Diagnostics"] = extras.(before["folders"]["src/Analysis"])
            push!(migrated["folders"]["src/Extras/Analysis"], "src/Diagnostics")
            push!(migrated["folders"]["src"], "src/Diagnostics")
            # Reuse the old facade->timer exception for facade->Diagnostics.
            migrated["folder_order_allow"] = Dict(join(extras.(split(id, '|')), '|') => reason
                for (id, reason) in get(before, "folder_order_allow", Dict()))
        end
        return migrated
    end
    haskey(before, "allow") || return before
    entries = Dict{String,Any}()
    for (id, reason) in before["allow"]
        parts = split(id, '|')
        core = any(p -> p in ("src/Analysis/budget_closure.jl", "src/Analysis/run_context.jl",
                              "src/Analysis/snapshot_series.jl", "src/Analysis/mass_totals.jl"), parts) ||
               ("src/Analysis/Analysis.jl" in parts && last(parts) != "_neumaier_sum")
        parts = [core ? replace(p, "src/Analysis/Analysis.jl" => "src/Diagnostics/Diagnostics.jl",
                                  "src/Analysis/" => "src/Diagnostics/",
                                  "AtmosTransport.Analysis" => "AtmosTransport.Diagnostics") : p for p in parts]
        entries[join(extras.(parts), '|')] = reason
    end
    return Dict("allow" => entries)
end

function relocate_o1_layers(before, file)
    # O1 splits the former Output observation unit into three lower modules
    # and a facade. Translate only its historical dependency budget; duplicate,
    # clone, private-use and size allowances receive no new identities.
    file == "src/LAYERS.toml" || return before
    mods = get(before, "modules", Dict())
    haskey(mods, "AtmosTransport.Output") || return before
    haskey(mods, "AtmosTransport.Observations") && return before
    migrated = deepcopy(before)
    output = "AtmosTransport.Output"
    observations = "AtmosTransport.Observations"
    children = observations .* [".Retrievals", ".Sources", ".Sampling"]
    for deps in values(migrated["modules"])
        output in deps && append!(deps, [observations; children])
    end
    migrated["modules"][observations] = children
    migrated["modules"][children[1]] = String[]
    migrated["modules"][children[2]] = ["AtmosTransport", "AtmosTransport.ConfigChecks", children[1]]
    migrated["modules"][children[3]] = [mods[output]; output; children[1:2]]
    folders = get(migrated, "folders", Dict())
    old = "src/Output/observations"
    if haskey(folders, old)
        new = ["src/Observations", "src/Observations/Retrievals",
               "src/Observations/Sources", "src/Observations/Sampling"]
        for deps in values(folders)
            old in deps && append!(deps, new)
        end
        folders[new[1]] = new[2:4]
        folders[new[2]] = String[]
        folders[new[3]] = ["src", new[2]]
        folders[new[4]] = [folders[old]; new[2:3]]
        # The package facade now also includes the new facade module. This is
        # the same foundation/facade folder exception as src -> Output.
        "src/Output" in get(folders, "src", String[]) && push!(folders["src"], new[1])
        exceptions = get!(migrated, "folder_order_allow", Dict())
        haskey(exceptions, "src|src/Output") &&
            (exceptions["src|src/Observations"] = exceptions["src|src/Output"])
    end
    return migrated
end

function relocate_m1_layers(before, file)
    # M1 splits the former single-namespace binary implementation. Only its
    # historical dependency budget moves: no private/clone/size debt is added.
    file == "src/LAYERS.toml" || return before
    parent = "AtmosTransport.MetDrivers"
    fmt, readers = parent * ".BinaryFormat", parent * ".BinaryReaders"
    mods = get(before, "modules", Dict())
    haskey(mods, fmt) && return before
    haskey(mods, parent) || return before
    migrated = deepcopy(before)
    prior = filter(!=(parent * ".ERA5"), mods[parent])
    for deps in values(migrated["modules"])
        parent in deps && append!(deps, [fmt, readers])
    end
    migrated["modules"][parent] = [prior; fmt; readers]
    migrated["modules"][fmt] = [prior; parent]
    migrated["modules"][readers] = [prior; parent; fmt]
    folders = get(migrated, "folders", Dict())
    old, root = "src/MetDrivers/transport_binary", "src/MetDrivers"
    new = [root * "/BinaryFormat", root * "/BinaryReaders"]
    if haskey(folders, old)
        prior_folders = filter(x -> !(x in (old, "src/MetDrivers/ERA5")), folders[root])
        for deps in values(folders)
            old in deps && append!(deps, new)
        end
        folders[root] = [prior_folders; new]
        folders[new[1]] = [prior_folders; root]
        folders[new[2]] = [prior_folders; root; new[1]]
        # ERA5 GRIB geometry now belongs to preprocessing. The facade keeps
        # the same geometry bindings, whose defining file moved with them.
        "src/MetDrivers/ERA5" in get(folders, "src", String[]) &&
            push!(folders["src"], "src/Preprocessing/sources")
    end
    return migrated
end

# C1 splits one existing dependency budget into a facade and two children.
# Only LAYERS identities migrate: helper/clone/docstring allowances still shrink.
function relocate_c1_layers(before, file)
    file == "src/LAYERS.toml" || return before
    parent = "AtmosTransport.Operators.Convection"
    modules = get(before, "modules", Dict())
    haskey(modules, parent * ".TM5") && return before
    haskey(modules, parent) || return before
    migrated = deepcopy(before)
    for (table, home) in (("modules", parent), ("folders", "src/Operators/Convection"))
        budgets = get(migrated, table, Dict())
        haskey(budgets, home) || continue
        children = [home * (table == "modules" ? ".TM5" : "/TM5"),
                    home * (table == "modules" ? ".CMFMC" : "/CMFMC")]
        prior = copy(budgets[home])
        for deps in values(budgets)
            home in deps && append!(deps, children)
        end
        budgets[home] = [prior; children]
        budgets[children[1]] = [prior; home]
        budgets[children[2]] = [prior; home; children[1]]
    end
    return migrated
end

# PP1 relocates the existing preprocessing dependency budget to ordered stages.
# This applies only to LAYERS identities; clone/private/docstring debt still
# shrinks exactly. A base already containing Core gets no special treatment.
function relocate_pp1_layers(before, file)
    file == "src/LAYERS.toml" || return before
    parent = "AtmosTransport.Preprocessing"
    modules = get(before, "modules", Dict())
    haskey(modules, parent * ".Core") && return before
    haskey(modules, parent) || return before
    migrated = deepcopy(before)
    for (table, home, sep) in (("modules", parent, "."),
                               ("folders", "src/Preprocessing", "/"))
        budgets = get(migrated, table, Dict())
        haskey(budgets, home) || continue
        children = [home * sep * name for name in ("Core", "Vertical", "Spectral")]
        prior = copy(budgets[home])
        lower = filter(d -> d != home && !startswith(d, home * sep), prior)
        for deps in values(budgets)
            home in deps && append!(deps, children)
        end
        budgets[home] = [prior; children]
        for i in eachindex(children)
            budgets[children[i]] = [lower; children[1:i-1]]
        end
    end
    return migrated
end

# PP2 gives the next two pipeline stages their existing dependency budget.
# PhysicsInputs can use Spectral but does not depend on MassClosure.
function relocate_pp2_layers(before, file)
    file == "src/LAYERS.toml" || return before
    parent = "AtmosTransport.Preprocessing"
    modules = get(before, "modules", Dict())
    haskey(modules, parent * ".MassClosure") && return before
    haskey(modules, parent) || return before
    migrated = deepcopy(before)
    for (table, home, sep) in (("modules", parent, "."),
                               ("folders", "src/Preprocessing", "/"))
        budgets = get(migrated, table, Dict())
        haskey(budgets, home) || continue
        children = [home * sep * name for name in ("MassClosure", "PhysicsInputs")]
        prior = copy(budgets[home])
        lower = filter(d -> d != home && !startswith(d, home * sep), prior)
        for (owner, deps) in budgets
            # Existing consumers of the facade can follow its moved bindings.
            # PP1 children retain their already established one-way contracts.
            startswith(owner, home * sep) && owner in
                home .* sep .* ["Core", "Vertical", "Spectral"] && continue
            home in deps && append!(deps, children)
        end
        budgets[home] = [prior; children]
        budgets[children[1]] = [lower; home * sep * "Core";
                              home * sep * "Vertical"; home * sep * "Spectral"]
        budgets[children[2]] = [lower; home * sep * "Core";
                              home * sep * "Vertical"; home * sep * "Spectral"]
    end
    return migrated
end

# PP3 relocates target/binary identities only. The binary protocols load below
# LL, then RG/CS (which reuse LL spectral staging/logging). PP1/PP2 stages keep
# their existing budgets. GS1 can express these four ordered child mappings
# directly in migrations.toml; no clone/private/docstring/size debt migrates.
function relocate_pp3_layers(before, file)
    file == "src/LAYERS.toml" || return before
    parent = "AtmosTransport.Preprocessing"
    modules = get(before, "modules", Dict())
    haskey(modules, parent * ".TransportBinary") && return before
    haskey(modules, parent) || return before
    migrated = deepcopy(before)
    for (table, home, sep, names) in (
        ("modules", parent, ".", ["TransportBinary", "LatLon", "ReducedGaussian", "CubedSphere"]),
        ("folders", "src/Preprocessing", "/",
         ["TransportBinary", "Targets/LatLon", "Targets/ReducedGaussian", "Targets/CubedSphere"]))
        budgets = get(migrated, table, Dict())
        haskey(budgets, home) || continue
        children = home .* sep .* names
        stages = home .* sep .* ["Core", "Vertical", "Spectral", "MassClosure", "PhysicsInputs"]
        prior = copy(budgets[home])
        lower = filter(d -> d != home && !startswith(d, home * sep), prior)
        for (owner, deps) in budgets
            owner in stages && continue
            (home in deps || home * sep * "transport_binary" in deps) && append!(deps, children)
        end
        budgets[home] = [prior; children]
        for (i, child) in enumerate(children)
            earlier = i == 1 ? String[] : children[1:min(i - 1, 2)]
            budgets[child] = [lower; stages; earlier]
        end
    end
    return migrated
end

# A1 retains one module because the fenced root workspaces/Lin-Rood kernels
# and the topology files share types. Split its existing folder budget only;
# sibling dependencies remain Reconstruction <- Structured <- CubedSphere.
const A1_HOME = "src/Operators/Advection"
const A1_CHILDREN = A1_HOME .* ["/Reconstruction", "/Structured", "/CubedSphere"]
function relocate_a1_layers(before, file)
    file == "src/LAYERS.toml" || return before
    folders = get(before, "folders", Dict())
    haskey(folders, A1_HOME) || return before
    haskey(folders, first(A1_CHILDREN)) && return before
    migrated = deepcopy(before)
    budgets = migrated["folders"]
    prior = copy(budgets[A1_HOME])
    for deps in values(budgets)
        A1_HOME in deps && append!(deps, A1_CHILDREN)
    end
    budgets[A1_HOME] = [prior; A1_CHILDREN]
    for (i, child) in enumerate(A1_CHILDREN)
        budgets[child] = [prior; A1_HOME; A1_CHILDREN[1:i-1]]
    end
    return migrated
end

# T1 separates the existing Adjoints unit into core, replay, observation
# and solver modules. Children inherit only the old lower dependency budget.
# No clone/private-use/size identities migrate and no reverse exception is added.
function relocate_t1_layers(before, file)
    file == "src/LAYERS.toml" || return before
    parent = "AtmosTransport.Adjoints"
    modules = get(before, "modules", Dict())
    haskey(modules, parent * ".Footprint") && return before
    haskey(modules, parent) || return before
    migrated = deepcopy(before)
    mods = migrated["modules"]
    fp, inv = parent * ".Footprint", parent * ".Inversion"
    obs, sol = inv * ".Observations", inv * ".Solver"
    prior = copy(mods[parent])
    for deps in values(mods)
        parent in deps && append!(deps, [fp, inv, obs, sol])
    end
    mods[parent] = [prior; fp; inv]
    mods[fp] = [prior; parent]
    mods[inv] = [obs, sol]
    mods[obs] = [prior; parent; fp]
    mods[sol] = [prior; parent; fp; obs]
    folders = get(migrated, "folders", Dict())
    core, oldfp, oldinv = "src/Adjoints", "src/Footprint", "src/Inversion"
    haskey(folders, oldfp) && haskey(folders, oldinv) || return migrated
    newfp, newinv = core * "/Footprint", core * "/Inversion"
    newobs, newsol = newinv * "/Observations", newinv * "/Solver"
    # Step forcing moved down from Footprint into core; Inversion split up.
    expand(f) = f == oldfp ? [core, newfp] :
                f == oldinv ? [newinv, newobs, newsol] : [f]
    old = deepcopy(folders)
    for (f, deps) in old
        f in (core, oldfp, oldinv) && continue
        folders[f] = unique(reduce(vcat, expand.(deps); init = String[]))
    end
    prior_core = filter(f -> f ∉ (oldfp, oldinv), old[core])
    prior_fp = filter(f -> f != oldinv, old[oldfp])
    prior_inv = filter(f -> f != oldfp, old[oldinv])
    folders[core] = [prior_core; newfp; newinv]
    folders[newfp] = prior_fp
    folders[newinv] = [newobs, newsol]
    folders[newobs] = [prior_inv; prior_core; newfp]
    folders[newsol] = [prior_inv; prior_core; newfp; newobs]
    delete!(folders, oldfp); delete!(folders, oldinv)
    # These historical cycles are eliminated, not renamed.
    exceptions = get(migrated, "folder_order_allow", Dict())
    for id in ("$core|$oldfp", "$core|$oldinv", "$oldfp|$oldinv")
        delete!(exceptions, id)
    end
    return migrated
end

function allowance_growth(before, after, file)
    before == after && return String[]
    original_count = length(get(before, "allow", Dict()))
    before = relocate_x1_baseline(before, file)
    before = relocate_o1_layers(before, file)
    before = relocate_m1_layers(before, file)
    before = relocate_c1_layers(before, file)
    before = relocate_pp1_layers(before, file)
    before = relocate_pp2_layers(before, file)
    before = relocate_pp3_layers(before, file)
    before = relocate_a1_layers(before, file)
    before = relocate_t1_layers(before, file)
    if file == "src/LAYERS.toml"
        growth = String[]
        for table in ("modules", "folders")
            append!(growth, ["new $table allowance $mod -> $dep" for (mod, deps) in get(after, table, Dict())
                    for dep in setdiff(deps, get(get(before, table, Dict()), mod, String[]))])
        end
        append!(growth, ["new folder order allowance $id" for id in
            setdiff(keys(get(after, "folder_order_allow", Dict())), keys(get(before, "folder_order_allow", Dict())))])
        return growth
    elseif file == FOLDER_SIZES
        old = get(before, "allow", Dict())
        return ["raised folder budget $folder" for (folder, entry) in get(after, "allow", Dict())
                if !haskey(old, folder) || entry["bytes"] > old[folder]["bytes"]]
    end
    growth = ["new $file entry $id" for id in
              setdiff(keys(get(after, "allow", Dict())), keys(get(before, "allow", Dict())))]
    length(get(after, "allow", Dict())) > original_count && isempty(growth) &&
        push!(growth, "increased $file allowance count")
    return growth
end

function check_ratchet!(errors, root, base, files)
    base === nothing && return
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(`git -C $root rev-parse --verify --quiet --end-of-options $(base * "^{commit}")`);
                        stdout = out, stderr = devnull))
    if !success(proc)
        branch = startswith(base, "origin/") ? base[8:end] : base
        push!(errors, "cannot validate structure base ref '$base'; fetch the base branch with " *
                      "git fetch --no-tags --depth=1 origin '+refs/heads/$branch:refs/remotes/origin/$branch', " *
                      "then set ATMOSTR_STRUCTURE_BASE=origin/$branch (or pass an existing commit)")
        return
    end
    commit = strip(String(take!(out)))
    for file in files
        # Only a genuinely absent path at a valid commit is first introduction.
        # Use fresh buffers: Julia 1.10 can retain a process-backed IOBuffer's
        # contents after take!, unlike newer Julia versions.
        out = IOBuffer()
        proc = run(pipeline(ignorestatus(`git -C $root ls-tree $commit -- $file`);
                            stdout = out, stderr = devnull))
        if !success(proc)
            push!(errors, "cannot inspect $file at structure base '$base'")
            continue
        end
        isempty(String(take!(out))) && continue
        cmd = `git -C $root show $commit\:$file`
        out = IOBuffer()
        proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = devnull))
        if !success(proc)
            push!(errors, "cannot read $file at structure base '$base'")
            continue
        end
        before = TOML.parse(String(take!(out)))
        after = isfile(joinpath(root, file)) ? TOML.parsefile(joinpath(root, file)) : Dict{String,Any}()
        append!(errors, ["allowlists only shrink: $entry" for entry in allowance_growth(before, after, file)])
    end
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
        if !child_wiring && !shared_advection_wiring && get(positions, b, typemax(Int)) >= get(positions, a, 0) && !haskey(exceptions, "$a|$b")
            push!(errors, "T1 reference to later folder: $a -> $b")
        end
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
    check_ratchet!(errors, root, base, vcat(["src/LAYERS.toml", FOLDER_SIZES], collect(values(SNAPSHOTS))))
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
