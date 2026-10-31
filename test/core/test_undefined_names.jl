# Every global name that package code references must resolve in the module
# that references it.
#
# Julia binds a global name only when the line that uses it runs. A module
# split that drops an import therefore loads, precompiles and passes every test
# that skips the branch, and fails later in production with `UndefVarError`.
# PP3 shipped two such cases (`dimnames` in the reduced-Gaussian target,
# `coarsen_xface_sum!` in the facade for GEOS block coarsening). This scan
# reads the lowered code of every method defined in `AtmosTransport` and its
# submodules and reports each `GlobalRef` whose binding is undefined.
#
# Not reported: operands of `@isdefined` (a binding check cannot throw), and
# the `##sync#…` channel that `Threads.@spawn` references in its never-taken
# branch when it is not inside `@sync`.

using Test
using AtmosTransport

# Undefined references that are known not to execute. Every entry needs a
# reason, and the test fails when an entry no longer matches (shrink only).
const ALLOWED_UNDEFINED_NAMES = Dict(
    # KernelAbstractions also instantiates a CPU body (`cpu__*`) for the
    # GPU-only collaborative TM5 kernels. Locals that cross `@synchronize`
    # appear there as globals. TM5Convection never launches these kernels on
    # the CPU (it warns and takes the per-thread path).
    "AtmosTransport.Operators.Convection.TM5.bidiagonal_lower" => "GPU-only collaborative TM5 kernel",
    "AtmosTransport.Operators.Convection.TM5.icllfs" => "GPU-only collaborative TM5 kernel",
    "AtmosTransport.Operators.Convection.TM5.icltop" => "GPU-only collaborative TM5 kernel",
    "AtmosTransport.Operators.Convection.TM5.k_lo" => "GPU-only collaborative TM5 kernel",
    "AtmosTransport.Operators.Convection.TM5.last_row" => "GPU-only collaborative TM5 kernel",
    "AtmosTransport.Operators.Convection.TM5.n_batch" => "GPU-only collaborative TM5 kernel",
    "AtmosTransport.Operators.Convection.TM5.no_conv" => "GPU-only collaborative TM5 kernel",
    # `Base.@kwdef struct PBLPhysicsParameters{FT}` also defines an outer
    # keyword constructor without `{FT}` whose defaults (`FT(15.0)`, ...) see no
    # `FT`. Production builds `PBLPhysicsParameters{FT}(...)`; the outer form
    # works only when every field is passed.
    "AtmosTransport.Operators.Diffusion.FT" => "Base.@kwdef outer constructor of PBLPhysicsParameters",

)

function _package_modules(root::Module)
    found = Module[]
    pending = Module[root]
    while !isempty(pending)
        mod = pop!(pending)
        mod in found && continue
        push!(found, mod)
        for name in names(mod; all = true)
            isdefined(mod, name) || continue
            value = getfield(mod, name)
            value isa Module && value !== mod && parentmodule(value) === mod &&
                push!(pending, value)
        end
    end
    return found
end

# Julia >= 1.12 keeps one global method table. Older versions keep one table per
# type name: visit the tables of every function and type bound in the package
# modules, of the type constructors, and of every Base/Core function that
# package methods may extend.
function _package_methods(mods)
    selected = Set(mods)
    out = Method[]
    keep(m::Method) = (m.module in selected && push!(out, m); nothing)
    if isdefined(Core, :methodtable)
        Base.visit(keep, getfield(Core, :methodtable))
        return out
    end
    tables = IdDict{Any, Nothing}(DataType.name.mt => nothing)
    for mod in [collect(mods); Base; Core], name in names(mod; all = true, imported = true)
        isdefined(mod, name) || continue
        value = getfield(mod, name)
        T = value isa Function ? typeof(value) :
            value isa Type ? Base.unwrap_unionall(value) : nothing
        T isa DataType && isdefined(T.name, :mt) && (tables[T.name.mt] = nothing)
    end
    for mt in keys(tables), m in Base.MethodList(mt)
        keep(m)
    end
    return unique(out)
end

function _undefined_refs!(found, x, where_)
    if x isa GlobalRef
        isdefined(x.mod, x.name) || startswith(String(x.name), "##sync#") ||
            push!(get!(found, "$(x.mod).$(x.name)", String[]), where_)
    elseif x isa Expr
        x.head === :isdefined && return found
        foreach(a -> _undefined_refs!(found, a, where_), x.args)
    elseif x isa Core.ReturnNode
        isdefined(x, :val) && _undefined_refs!(found, x.val, where_)
    elseif x isa Core.GotoIfNot
        _undefined_refs!(found, x.cond, where_)
    end
    return found
end

@testset "package code references only defined global names" begin
    mods = _package_modules(AtmosTransport)
    methods_ = _package_methods(mods)
    @test length(methods_) > 1000
    found = Dict{String, Vector{String}}()
    for m in methods_
        src = try
            Base.uncompressed_ast(m)
        catch
            continue
        end
        where_ = "$(m.name) @ $(m.file):$(m.line)"
        foreach(stmt -> _undefined_refs!(found, stmt, where_), src.code)
    end
    unexpected = sort!([name => unique(sites) for (name, sites) in found
                        if !haskey(ALLOWED_UNDEFINED_NAMES, name)]; by = first)
    for (name, sites) in unexpected
        @error "Undefined global `$name`; add the import in the referencing module" sites
    end
    @test isempty(unexpected)
    stale = sort!([name for name in keys(ALLOWED_UNDEFINED_NAMES) if !haskey(found, name)])
    isempty(stale) || @error "Remove resolved entries from ALLOWED_UNDEFINED_NAMES" stale
    @test isempty(stale)
end
