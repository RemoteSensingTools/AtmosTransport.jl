#!/usr/bin/env julia
# Required-keyword call check.
#
# A keyword argument without a default is required, and a call that omits it
# fails only when it runs, with an `UndefKeywordError`. Scripts, documentation
# examples and rarely exercised paths can keep such a call long after a
# signature gained the keyword (the transport-binary writers' `mass_basis`,
# `surface_pressure_basis` and `planet_radius` are examples). This check parses
# `src/` for methods with required keywords, including `@kwdef` struct fields
# without defaults, and reports every call in the maintained Julia files that
# omits one.
#
# A call is compared with the methods of the same name whose positional arity
# admits it, and a keyword is reported only when every one of those methods
# requires it; methods a file defines itself count as candidates for the calls
# in that file. Calls that splat keywords (`f(x; kw...)`) are skipped, and so
# are calls inside `@test_throws`, which may omit a keyword on purpose. The
# scan covers src/, ext/, test/, scripts/, benchmarking/, examples/ and the
# Julia files of docs/literate and docs/scripts; `heritage/` and `archived/`
# folders (frozen records) are skipped.
#
# Usage:
#
#   julia --startup-file=no scripts/checks/check_keyword_calls.jl [--root=<repo>]
#
# Exit code 0 when every call passes, 1 otherwise. Only Base is needed.

module KeywordCallCheck

const SCAN_DIRS = ("src", "ext", "test", "scripts", "benchmarking", "examples",
                   joinpath("docs", "literate"), joinpath("docs", "scripts"))
const SKIP_DIRS = ("heritage", "archived")

"""One method signature: positional arity bounds and required keywords."""
struct Signature
    npos_min::Int
    npos_max::Float64          # Inf with a positional vararg
    required::Set{Symbol}
    file::String
    line::Int
end

"""A call that omits a keyword every candidate method requires."""
struct Finding
    file::String
    line::Int
    name::Symbol
    missing_keywords::Vector{Symbol}
    nmethods::Int
end

function julia_files(root::AbstractString, dirs = SCAN_DIRS)
    files = String[]
    for d in dirs
        top = joinpath(root, d)
        isdir(top) || continue
        for (dir, subdirs, names) in walkdir(top)
            filter!(s -> !(s in SKIP_DIRS) && !startswith(s, "."), subdirs)
            for n in names
                endswith(n, ".jl") && push!(files, joinpath(dir, n))
            end
        end
    end
    return sort!(files)
end

function parse_file(path::AbstractString)
    try
        return Meta.parseall(read(path, String); filename = path)
    catch err
        @warn "skipping $(path): it does not parse" exception = err
        return nothing
    end
end

# Name of a callee or of a defined function: `f`, `M.f`, `F{T}`; else nothing.
function callee_name(ex)
    ex isa Symbol && return ex
    if ex isa Expr
        ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode &&
            return ex.args[2].value isa Symbol ? ex.args[2].value : nothing
        ex.head === :curly && return callee_name(ex.args[1])
    end
    return nothing
end

# The call expression of a definition signature (strip `where` and return types).
function signature_call(sig)
    while sig isa Expr && sig.head in (:where, :(::))
        sig.head === :(::) && length(sig.args) == 1 && return nothing
        sig = sig.args[1]
    end
    return sig isa Expr && sig.head === :call ? sig : nothing
end

keyword_name(p::Symbol) = p
function keyword_name(p::Expr)
    p.head === :(::) && return keyword_name(p.args[1])
    p.head === :kw && return keyword_name(p.args[1])
    return nothing
end

function definition_signature(call::Expr, file, line)
    name = callee_name(call.args[1])
    name === nothing && return nothing
    npos_min, npos_max = 0, 0.0
    required = Set{Symbol}()
    for a in call.args[2:end]
        if a isa Expr && a.head === :parameters
            for p in a.args
                p isa Expr && p.head === :... && continue
                p isa Expr && p.head === :kw && continue
                k = keyword_name(p)
                k === nothing || push!(required, k)
            end
        elseif a isa Expr && a.head === :...
            npos_max = Inf
        elseif a isa Expr && a.head === :kw
            npos_max += 1
        else
            npos_min += 1
            npos_max += 1
        end
    end
    return name => Signature(npos_min, npos_max, required, file, line)
end

# `@kwdef struct S ... end`: a keyword constructor with the fields without
# defaults required, and the positional constructor over all fields.
function kwdef_signatures(st::Expr, file, line)
    head = st.args[2]
    head isa Expr && head.head === :(<:) && (head = head.args[1])
    name = callee_name(head)
    name === nothing && return Pair{Symbol, Signature}[]
    required = Set{Symbol}()
    nfields = 0
    for f in st.args[3].args
        f isa LineNumberNode && continue
        if f isa Symbol || (f isa Expr && f.head === :(::))
            push!(required, keyword_name(f)); nfields += 1
        elseif f isa Expr && f.head === :(=) && signature_call(f.args[1]) === nothing
            nfields += 1
        end
    end
    return [name => Signature(0, 0.0, required, file, line),
            name => Signature(nfields, Float64(nfields), Set{Symbol}(), file, line)]
end

is_macro(ex, names) = ex isa Expr && ex.head === :macrocall &&
    (ex.args[1] in names || (ex.args[1] isa Expr && ex.args[1].head === :. &&
                             ex.args[1].args[end] isa QuoteNode &&
                             ex.args[1].args[end].value in names))

const KWDEF = (Symbol("@kwdef"),)
const TEST_THROWS = (Symbol("@test_throws"),)

"""Collect `name => Signature` for every method defined in `ex`."""
function collect_definitions!(defs, ex, file, line = Ref(0))
    if ex isa LineNumberNode
        line[] = ex.line
    elseif ex isa Expr
        if is_macro(ex, KWDEF)
            for a in ex.args
                a isa Expr && a.head === :struct && append!(defs, kwdef_signatures(a, file, line[]))
            end
        end
        if ex.head in (:function, :(=)) && length(ex.args) >= 1
            call = signature_call(ex.args[1])
            if call !== nothing
                sig = definition_signature(call, file, line[])
                sig === nothing || push!(defs, sig)
            end
        end
        for a in ex.args
            collect_definitions!(defs, a, file, line)
        end
    end
    return defs
end

function signature_table(files)
    table = Dict{Symbol, Vector{Signature}}()
    for f in files
        ex = parse_file(f)
        ex === nothing && continue
        for (name, sig) in collect_definitions!(Pair{Symbol, Signature}[], ex, f)
            push!(get!(table, name, Signature[]), sig)
        end
    end
    return table
end

# Positional count (nothing when splatted), keywords given, and whether keywords are splatted.
function call_arguments(call::Expr, extra_positional::Int)
    npos = extra_positional
    splat_pos = false
    given = Set{Symbol}()
    splat_kw = false
    for a in call.args[2:end]
        if a isa Expr && a.head === :parameters
            for p in a.args
                if p isa Expr && p.head === :kw
                    k = keyword_name(p.args[1]); k === nothing ? (splat_kw = true) : push!(given, k)
                elseif p isa Symbol
                    push!(given, p)
                elseif p isa Expr && p.head === :. && p.args[end] isa QuoteNode
                    push!(given, p.args[end].value)
                else
                    splat_kw = true
                end
            end
        elseif a isa Expr && a.head === :kw
            k = keyword_name(a.args[1]); k === nothing ? (splat_kw = true) : push!(given, k)
        elseif a isa Expr && a.head === :...
            splat_pos = true
        else
            npos += 1
        end
    end
    return (; npos, splat_pos, given, splat_kw)
end

function check_call!(findings, table, local_defs, call::Expr, file, line, extra_positional)
    name = callee_name(call.args[1])
    name === nothing && return
    methods = vcat(get(table, name, Signature[]), get(local_defs, name, Signature[]))
    any(s -> !isempty(s.required), methods) || return
    args = call_arguments(call, extra_positional)
    args.splat_kw && return
    candidates = args.splat_pos ? filter(s -> s.npos_max >= args.npos, methods) :
                 filter(s -> s.npos_min <= args.npos <= s.npos_max, methods)
    isempty(candidates) && return
    required = intersect((s.required for s in candidates)...)
    missing_keywords = sort!(collect(setdiff(required, args.given)))
    isempty(missing_keywords) ||
        push!(findings, Finding(file, line, name, missing_keywords, length(candidates)))
    return
end

# Walk `ex`, checking calls; definition signatures and `@test_throws` bodies are skipped.
function check_calls!(findings, table, local_defs, ex, file, line = Ref(0))
    if ex isa LineNumberNode
        line[] = ex.line
        return findings
    end
    ex isa Expr || return findings
    if is_macro(ex, TEST_THROWS)
        return findings
    elseif ex.head in (:function, :(=)) && length(ex.args) == 2 &&
           signature_call(ex.args[1]) !== nothing
        sig = signature_call(ex.args[1])
        # Keyword defaults may call functions; walk them, not the signature itself.
        for a in sig.args[2:end]
            a isa Expr && a.head === :parameters && check_calls!(findings, table, local_defs, a, file, line)
        end
        return check_calls!(findings, table, local_defs, ex.args[2], file, line)
    elseif ex.head === :do && ex.args[1] isa Expr && ex.args[1].head === :call
        check_call!(findings, table, local_defs, ex.args[1], file, line[], 1)
        for a in ex.args[1].args[2:end]
            check_calls!(findings, table, local_defs, a, file, line)
        end
        return check_calls!(findings, table, local_defs, ex.args[2], file, line)
    elseif ex.head === :call
        check_call!(findings, table, local_defs, ex, file, line[], 0)
    end
    for a in ex.args
        check_calls!(findings, table, local_defs, a, file, line)
    end
    return findings
end

"""
    check_repository(root) -> (findings, nfiles, nfunctions)

Report every call in the scanned folders of `root` that omits a keyword all
candidate methods of `src/` (and of the calling file) require.
"""
function check_repository(root::AbstractString)
    table = signature_table(julia_files(root, ("src",)))
    filter!(kv -> any(s -> !isempty(s.required), kv.second), table)
    findings = Finding[]
    files = julia_files(root)
    for f in files
        ex = parse_file(f)
        ex === nothing && continue
        local_defs = Dict{Symbol, Vector{Signature}}()
        if !startswith(f, joinpath(root, "src") * "/")
            for (name, sig) in collect_definitions!(Pair{Symbol, Signature}[], ex, f)
                push!(get!(local_defs, name, Signature[]), sig)
            end
        end
        check_calls!(findings, table, local_defs, ex, f)
    end
    return findings, length(files), length(table)
end

function main(args = ARGS)
    root = normpath(joinpath(@__DIR__, "..", ".."))
    for a in args
        startswith(a, "--root=") ? (root = abspath(a[8:end])) :
            error("unknown argument $(repr(a)); usage: check_keyword_calls.jl [--root=<repo>]")
    end
    findings, nfiles, nfunctions = check_repository(root)
    for f in findings
        println(relpath(f.file, root), ":", f.line, ": `", f.name, "` call omits required keyword(s) ",
                join(("`$(k)`" for k in f.missing_keywords), ", "),
                " (required by all ", f.nmethods, " candidate method(s))")
    end
    println("keyword-call check: ", length(findings), " finding(s) in ", nfiles,
            " files; ", nfunctions, " functions with required keywords")
    return isempty(findings) ? 0 : 1
end

end # module KeywordCallCheck

if abspath(PROGRAM_FILE) == @__FILE__
    exit(KeywordCallCheck.main())
end
