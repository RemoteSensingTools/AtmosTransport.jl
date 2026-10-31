# The package reads only the environment variables documented in
# docs/src/config/environment.md, each from the part of `src/` that owns it.
# A new run setting belongs in the TOML configuration, not in the environment.
using Test

const ENV_REPO = normpath(joinpath(@__DIR__, "..", ".."))

# Variable => the `src/` files that may read it (a pattern, so file moves
# within the owning folder keep passing).
const ENV_ALLOWLIST = Dict(
    "ATMOSTR_TIMERS"                => r"src/SectionTimer\.jl$",
    "ATMOSTR_NVTX"                  => r"src/SectionTimer\.jl$",
    "ATMOSTR_ALLOC_TIMERS"          => r"src/SectionTimer\.jl$",
    "ATMOSTR_PROFILE_GPU"           => r"src/Operators/Advection/",
    "ATMOSTR_REGRID_CACHE_DIR"      => r"src/Regridding/",
    "NO_COLOR"                      => r"src/Models/Runner/",
    "TERM"                          => r"src/Models/Runner/",
    "ATMOSTR_INSTITUTION"           => r"src/Output/",
    "USER"                          => r"src/Output/",
    "USERNAME"                      => r"src/Output/",
    "ATMOSTR_SPECTRAL_CACHE_DIR"    => r"src/Preprocessing/",
    "ATMOS_OMEGA_TIMING"            => r"src/Preprocessing/",
    "ERA5_N320_PROFILE"             => r"src/Preprocessing/",
)

# Reads whose variable name is computed: path expansion (`$NAME` in paths)
# and the ERA5 scratch-directory candidates. Their names are documented here.
const ENV_DYNAMIC_READERS = Dict(
    r"src/AtmosTransport\.jl$" => ["ATMOSTRANSPORT_DATA_ROOT"],
    r"src/Preprocessing/PhysicsInputs/era5_surface_reader\.jl$" =>
        ["ATMOSTR_SCRATCH_DIR", "ATMOSTR_TMPDIR", "SCRATCH", "TMPDIR", "TEMP", "TMP"],
)

is_env(x) = x === :ENV || (x isa Expr && x.head === :. && x.args[end] == QuoteNode(:ENV))
# `get` or `Base.get`: the name of a call to a Base function, else nothing.
call_name(f) = f isa Symbol ? f :
               f isa Expr && f.head === :. && f.args[1] === :Base && f.args[end] isa QuoteNode ?
               f.args[end].value : nothing
const ENV_ACCESSORS = (:get, :haskey, :getindex, :get!, :delete!)
is_env_flag(f) = f === :env_flag ||
    (f isa Expr && f.head === :. && f.args[1] === :ConfigChecks &&
     f.args[end] == QuoteNode(:env_flag))

# Only this exact forwarding method may delegate its read to the literal-name
# call-site audit. A changed body or any other ENV read in ConfigChecks is
# scanned normally; this grants no file-wide computed-name exception.
const ENV_FLAG_FORWARDER_SOURCE = """
function env_flag(name::AbstractString;
                  true_values = ("1", "true", "on", "yes"), case_sensitive::Bool = false)
    value = get(ENV, name, "")
    return (case_sensitive ? value : lowercase(value)) in true_values
end
"""
const ENV_FLAG_FORWARDER = Base.remove_linenums!(Meta.parse(ENV_FLAG_FORWARDER_SOURCE))

# Every use of ENV in an expression tree: (variable name, `nothing` for a
# computed name, or `:other` for any use that is not a direct read, such as
# `env = ENV`), and its line.
function env_reads!(reads, ex, line = 0; signature = false, flag_forwarder = false)
    ex isa LineNumberNode && return ex.line
    is_env(ex) && (push!(reads, (:other, line)); return line)
    ex isa Expr || return line
    flag_forwarder && ex.head === :function &&
        Base.remove_linenums!(deepcopy(ex)) == ENV_FLAG_FORWARDER && return line
    args = ex.args
    key, skip = if ex.head === :ref && length(args) == 2 && is_env(args[1])
        args[2], 1
    elseif ex.head === :call && call_name(args[1]) in ENV_ACCESSORS &&
           length(args) >= 3 && is_env(args[2])
        args[3], 2
    elseif !signature && ex.head === :call && is_env_flag(args[1])
        positional = filter(a -> !(a isa Expr && a.head === :parameters), args[2:end])
        isempty(positional) ? (missing, 0) : (first(positional), 0)
    else
        missing, 0
    end
    key === missing || push!(reads, (key isa String ? key : nothing, line))
    for (i, arg) in enumerate(args)
        i == skip && continue          # the ENV of a recognized read
        # A method signature is not a call; its defaults may still read ENV.
        is_signature = i == 1 && (ex.head === :function ||
            (ex.head === :(=) && arg isa Expr && arg.head === :call))
        line = arg isa LineNumberNode ? arg.line :
               env_reads!(reads, arg, line; signature = is_signature, flag_forwarder)
    end
    return line
end

function env_reads(path)
    reads = Tuple{Union{String, Symbol, Nothing}, Int}[]
    env_reads!(reads, Meta.parseall(read(path, String); filename = path);
               flag_forwarder = normpath(path) == joinpath(ENV_REPO, "src", "ConfigChecks.jl"))
    return reads
end

@testset "the ENV scanner sees every form of read" begin
    scan(code; flag_forwarder = false) = (r = Tuple{Union{String, Symbol, Nothing}, Int}[];
        env_reads!(r, Meta.parseall(code); flag_forwarder); first.(r))
    @test scan("get(ENV, \"A\", \"\")") == ["A"]
    @test scan("Base.get(ENV, \"A\", \"\")") == ["A"]
    @test scan("haskey(Base.ENV, \"A\")") == ["A"]
    @test scan("getindex(ENV, \"A\")") == ["A"]
    @test scan("ENV[\"A\"]") == ["A"]
    @test scan("get(ENV, name, \"\")") == [nothing]
    @test scan("env_flag(\"A\")") == ["A"]
    @test scan("env_flag(\"A\"; true_values=(\"1\",), case_sensitive=true)") == ["A"]
    @test scan("ConfigChecks.env_flag(\"A\")") == ["A"]
    @test scan("env_flag(name)") == [nothing]
    @test scan("function env_flag(name); get(ENV, name, \"\"); end") == [nothing]
    @test scan("env_flag(name) = get(ENV, name, \"\")") == [nothing]
    forwarder = ENV_FLAG_FORWARDER_SOURCE
    @test scan(forwarder) == [nothing]
    @test isempty(scan(forwarder; flag_forwarder = true))
    @test scan(forwarder * "\nenv_flag(\"A\")"; flag_forwarder = true) == ["A"]
    @test scan(forwarder * "\nget(ENV, other, \"\")"; flag_forwarder = true) == [nothing]
    @test scan(replace(forwarder, "get(ENV, name," => "get(ENV, other,");
               flag_forwarder = true) == [nothing]
    @test scan(replace(forwarder, "value = get" => "get(ENV, other, \"\"); value = get");
               flag_forwarder = true) == [nothing, nothing]
    @test scan("env = ENV; get(env, \"A\", \"\")") == [:other]
    @test scan("f(ENV)") == [:other]
    @test scan("OtherModule.get(ENV, \"A\", \"\")") == [:other]
    @test isempty(scan("# ENV[\"A\"]\n\"\"\"ENV[\\\"A\\\"]\"\"\"\nx = 1"))
end

src_files = [joinpath(root, f) for (root, _, fs) in walkdir(joinpath(ENV_REPO, "src"))
             for f in fs if endswith(f, ".jl")]
docs = read(joinpath(ENV_REPO, "docs", "src", "config", "environment.md"), String)

@testset "environment variables read by src/ are allowed and documented" begin
    used = Set{Any}()
    unexpected = String[]
    for path in src_files
        rel = relpath(path, ENV_REPO)
        for (name, line) in env_reads(path)
            if name === :other
                push!(unexpected, "$(rel):$(line): ENV used other than as a direct read")
            elseif name === nothing
                pattern = findfirst(p -> occursin(p, rel), collect(keys(ENV_DYNAMIC_READERS)))
                pattern === nothing ? push!(unexpected, "$(rel):$(line): computed variable name") :
                                      push!(used, collect(keys(ENV_DYNAMIC_READERS))[pattern])
            elseif haskey(ENV_ALLOWLIST, name) && occursin(ENV_ALLOWLIST[name], rel)
                push!(used, name)
            else
                push!(unexpected, "$(rel):$(line): $(name)")
            end
        end
    end
    @test isempty(unexpected)
    # No stale entries.
    @test isempty(setdiff(Set(keys(ENV_ALLOWLIST)), used))
    @test isempty(setdiff(Set(keys(ENV_DYNAMIC_READERS)), used))
    for name in vcat(collect(keys(ENV_ALLOWLIST)), values(ENV_DYNAMIC_READERS)...)
        @test occursin("`$(name)`", docs)
    end
end

@testset "environment variables of the command-line runner are documented" begin
    reads = env_reads(joinpath(ENV_REPO, "scripts", "run_transport.jl"))
    @test all(((n, _),) -> n isa String, reads)     # no computed names, no aliasing
    names = [n for (n, _) in reads if n isa String]
    @test !isempty(names)
    for name in unique(names)
        @test occursin("`$(name)`", docs)
    end
end
