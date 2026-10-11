"""
    AtmosTransportLabLane

Lab lane for GPU benchmark numbers: benchmark one published commit on the lab
machines listed in `machines.toml`, copy the results back, validate them, and
optionally hand them to the `Benchmarks` workflow, which publishes them to the
dashboard. Uses Base, TOML, JSON3, and Dates only, so the test suite can load
it by path without the benchmark environment.

Entry point: `benchmarking/lab/run_lab_benchmarks.jl`.
"""
module AtmosTransportLabLane

using Dates
using JSON3
using TOML

include(joinpath(@__DIR__, "..", "src", "labels.jl"))
include(joinpath(@__DIR__, "..", "src", "case_options.jl"))

export LabMachine, LocalAccess, SSHAccess, SharedArchive, CopiedArchive,
    LabStep, LabOptions, load_machines, parse_machines, machine_plan,
    preparation_steps, validate_machine_results, merge_dashboard_records,
    validate_publishable_records, parse_lab_options, run_lab, render_step,
    normalize_device_label

"Default lab machine table, next to this module."
const DEFAULT_MACHINES_FILE = normpath(joinpath(@__DIR__, "machines.toml"))
"Directory (gitignored) receiving copied-back results, one subdirectory per commit."
const DEFAULT_RESULTS_DIR = normpath(joinpath(@__DIR__, "results"))
"Orphan branch that carries lab results to the Benchmarks workflow."
const INBOX_BRANCH = "benchmark-inbox"
"Directory inside the inbox branch holding one JSON file per lab run."
const INBOX_DIRECTORY = "lab"
"Workflow file dispatched to publish lab results."
const BENCHMARK_WORKFLOW = "Benchmarks.yml"
"The GitHub CLI; `gh` on PATH is shadowed on the lab hosts."
const DEFAULT_GH = "/usr/bin/gh"
"Name of the benchmark preset the lab lane runs."
const DEFAULT_PRESET = "lab"
"Prefix every code root's last path component must carry."
const ROOT_COMPONENT_PREFIX = "atmostr_"
"Raw results file written by run_benchmarks.jl on each machine."
const RAW_RESULTS_FILE = "benchmark_results.json"
"Dashboard file written next to the raw results."
const DASHBOARD_RESULTS_FILE = "benchmark_results.github-action-benchmark.json"
"ssh options: never prompt, keep long benchmark sessions alive."
const SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ServerAliveInterval=60"]
"Hardware class each backend's records must carry (CPU classes start with this)."
const BACKEND_CLASSES = Dict(:cpu => "CPU", :cuda => "GPU-CUDA", :metal => "GPU-Metal")
"""
Oldest Julia a machine's `julia_version` may pin. A `git archive` has no
`benchmarking/Manifest.toml` (it is gitignored), so the environment resolves
`AtmosTransport` through the `[sources]` path entry of
`benchmarking/Project.toml`, which Julia 1.10 ignores.
"""
const MINIMUM_LAB_JULIA = v"1.11"

# =========================================================================
# Machine configuration
# =========================================================================

"How the lab lane reaches a machine."
abstract type MachineAccess end
"Run on the host executing the lab lane."
struct LocalAccess <: MachineAccess end
"Run through `ssh host` with key authentication."
struct SSHAccess <: MachineAccess
    host::String
end

"How the benchmarked commit's source reaches a machine."
abstract type CodeDelivery end
"""
`git archive <commit>` extracted on the lab-lane host into `root/<commit>`; the
machine sees the same absolute path (NFS). `root` is absolute.
"""
struct SharedArchive <: CodeDelivery
    root::String
end
"""
`git archive <commit>` streamed over ssh into `root/<commit>` on the machine.
`root` is absolute or starts with `~/` (the machine's home).
"""
struct CopiedArchive <: CodeDelivery
    root::String
end

"""
    LabMachine

One `[machines.<name>]` table of `machines.toml`, validated.
"""
struct LabMachine
    name::String
    description::String
    access::MachineAccess
    code::CodeDelivery
    julia::String
    julia_version::VersionNumber
    backend::Symbol
    float_types::Vector{String}
    threads::Int
    heap_size_hint::Union{Nothing, String}
    tmpdir::Union{Nothing, String}
    caffeinate::Bool
    device_label::Union{Nothing, String}
    env::Vector{Pair{String, String}}
end

const _TOP_LEVEL_KEYS = ("machines",)
const _MACHINE_KEYS = ("description", "access", "ssh_host", "julia", "julia_version", "backend",
                       "float_types", "threads", "heap_size_hint", "tmpdir",
                       "caffeinate", "device_label", "code", "env")
const _CODE_KEYS = ("delivery", "root")
# Environment variables that have a dedicated machine key.
const _RESERVED_ENV = Dict("TMPDIR" => "tmpdir", "JULIA_NUM_THREADS" => "threads")
# Paths are interpolated into remote shell commands; keep them plain.
const _PATH_PATTERN = r"^[A-Za-z0-9_./~+-]+$"

struct MachineConfigError <: Exception
    msg::String
end
Base.showerror(io::IO, e::MachineConfigError) = print(io, "MachineConfigError: ", e.msg)

_config_error(where, msg) = throw(MachineConfigError("$(where): $(msg)"))

function _check_keys(table::AbstractDict, allowed, where)
    unknown = sort!([String(k) for k in keys(table) if !(String(k) in allowed)])
    isempty(unknown) ||
        _config_error(where, "unknown key(s) $(join(unknown, ", ")); allowed: $(join(allowed, ", "))")
    return nothing
end

function _require(table, key, ::Type{T}, where) where {T}
    haskey(table, key) || _config_error(where, "missing required key `$(key)`")
    return _typed(table[key], T, "$(where).$(key)")
end

_optional(table, key, ::Type{T}, where, default) where {T} =
    haskey(table, key) ? _typed(table[key], T, "$(where).$(key)") : default

function _typed(value, ::Type{T}, where) where {T}
    value isa T || _config_error(where, "expected $(T), got $(typeof(value)) ($(repr(value)))")
    return value
end

function _check_path(path::AbstractString, where; root = false)
    occursin(_PATH_PATTERN, path) ||
        _config_error(where, "path $(repr(path)) may only contain letters, digits, and _ . / ~ + -")
    (startswith(path, "/") || startswith(path, "~/") || (!root && !occursin('/', path))) ||
        _config_error(where, "path $(repr(path)) must be absolute or start with \"~/\"")
    occursin("..", path) && _config_error(where, "path $(repr(path)) must not contain \"..\"")
    if root
        startswith(basename(rstrip(path, '/')), ROOT_COMPONENT_PREFIX) || _config_error(where,
            "code root $(repr(path)) must end in a directory named $(ROOT_COMPONENT_PREFIX)*")
    end
    return String(path)
end

function _parse_access(table, where)
    access = _require(table, "access", String, where)
    if access == "local"
        haskey(table, "ssh_host") && _config_error(where, "`ssh_host` is only valid with access = \"ssh\"")
        return LocalAccess()
    elseif access == "ssh"
        host = _require(table, "ssh_host", String, where)
        occursin(r"^[A-Za-z0-9_.@-]+$", host) && !startswith(host, "-") ||
            _config_error(where, "invalid ssh_host $(repr(host))")
        return SSHAccess(host)
    end
    _config_error(where, "access must be \"local\" or \"ssh\", got $(repr(access))")
end

function _parse_code(table, access::MachineAccess, where)
    code = _require(table, "code", AbstractDict, where)
    cwhere = "$(where).code"
    _check_keys(code, _CODE_KEYS, cwhere)
    delivery = _require(code, "delivery", String, cwhere)
    root = _check_path(_require(code, "root", String, cwhere), "$(cwhere).root"; root = true)
    if delivery == "shared"
        return SharedArchive(rstrip(expanduser(root), '/'))
    elseif delivery == "copy"
        access isa SSHAccess ||
            _config_error(cwhere, "delivery = \"copy\" needs access = \"ssh\"; use \"shared\" locally")
        return CopiedArchive(rstrip(root, '/'))
    end
    _config_error(cwhere, "delivery must be \"shared\" or \"copy\", got $(repr(delivery))")
end

function _parse_env(table, where)
    env = _optional(table, "env", AbstractDict, where, Dict{String, Any}())
    pairs = Pair{String, String}[]
    for key in sort!(collect(String.(keys(env))))
        ewhere = "$(where).env.$(key)"
        occursin(r"^[A-Z_][A-Z0-9_]*$", key) || _config_error(ewhere, "invalid variable name")
        haskey(_RESERVED_ENV, key) &&
            _config_error(ewhere, "set the machine key `$(_RESERVED_ENV[key])` instead")
        push!(pairs, key => _typed(env[key], String, ewhere))
    end
    return pairs
end

function _parse_julia_version(table, where)
    s = _require(table, "julia_version", String, where)
    occursin(r"^[0-9]+\.[0-9]+\.[0-9]+$", s) ||
        _config_error(where, "julia_version $(repr(s)) must be an exact release such as \"1.12.6\"")
    v = VersionNumber(s)
    v >= MINIMUM_LAB_JULIA || _config_error(where,
        "julia_version $(s) is older than $(MINIMUM_LAB_JULIA); a git archive has no " *
        "benchmarking/Manifest.toml and Julia 1.10 ignores [sources] in benchmarking/Project.toml")
    return v
end

function _parse_machine(name::AbstractString, table::AbstractDict)
    where = "machines.$(name)"
    occursin(r"^[a-z][a-z0-9_-]*$", name) ||
        _config_error(where, "machine names are lowercase letters, digits, _ and -")
    _check_keys(table, _MACHINE_KEYS, where)
    access = _parse_access(table, where)
    code = _parse_code(table, access, where)
    backend = Symbol(_require(table, "backend", String, where))
    haskey(BACKEND_CLASSES, backend) ||
        _config_error(where, "backend must be one of $(join(sort!(collect(keys(BACKEND_CLASSES))), ", "))")
    float_types = _require(table, "float_types", AbstractVector, where)
    isempty(float_types) && _config_error(where, "float_types is empty")
    for ft in float_types
        ft in SUPPORTED_FLOAT_TYPES ||
            _config_error(where, "unsupported float type $(repr(ft)); use $(join(SUPPORTED_FLOAT_TYPES, ", "))")
    end
    float_types = String.(float_types)
    backend === :metal && float_types != ["Float32"] &&
        _config_error(where, "Metal supports Float32 only")
    threads = _optional(table, "threads", Integer, where, 1)
    threads >= 1 || _config_error(where, "threads must be positive")
    heap = _optional(table, "heap_size_hint", String, where, nothing)
    heap === nothing || occursin(r"^[0-9]+[KMGT]?$", heap) ||
        _config_error(where, "heap_size_hint $(repr(heap)) must look like 64G")
    tmpdir = _optional(table, "tmpdir", String, where, nothing)
    tmpdir === nothing || _check_path(tmpdir, "$(where).tmpdir")
    label = _optional(table, "device_label", String, where, nothing)
    if label !== nothing
        backend === :cpu && _config_error(where, "device_label only applies to GPU backends")
        try
            validate_device_label(label)
        catch err
            _config_error(where, sprint(showerror, err))
        end
    end
    return LabMachine(String(name),
                      _optional(table, "description", String, where, ""),
                      access, code,
                      _check_path(_require(table, "julia", String, where), "$(where).julia"),
                      _parse_julia_version(table, where), backend, float_types, Int(threads), heap, tmpdir,
                      _optional(table, "caffeinate", Bool, where, false),
                      label, _parse_env(table, where))
end

"""
    parse_machines(table; source) -> Vector{LabMachine}

Validate a parsed `machines.toml`. Unknown keys at any level are errors.
"""
function parse_machines(table::AbstractDict; source::AbstractString = "machines.toml")
    try
        _check_keys(table, _TOP_LEVEL_KEYS, "top level")
        machines = _require(table, "machines", AbstractDict, "top level")
        isempty(machines) && _config_error("top level", "no machines defined")
        return [_parse_machine(String(name), _typed(machines[name], AbstractDict, "machines.$(name)"))
                for name in sort!(collect(String.(keys(machines))))]
    catch err
        err isa MachineConfigError && throw(MachineConfigError("$(source): $(err.msg)"))
        rethrow()
    end
end

load_machines(path::AbstractString = DEFAULT_MACHINES_FILE) =
    parse_machines(TOML.parsefile(path); source = path)

"""
    select_machines(machines, names) -> Vector{LabMachine}

Machines named in `names`, in that order; all machines when `names` is empty.
"""
function select_machines(machines::Vector{LabMachine}, names::Vector{String})
    isempty(names) && return machines
    byname = Dict(m.name => m for m in machines)
    missing_names = [n for n in names if !haskey(byname, n)]
    isempty(missing_names) || throw(ArgumentError(
        "unknown machine(s) $(join(missing_names, ", ")); configured: $(join(sort!(collect(keys(byname))), ", "))"))
    return [byname[n] for n in unique(names)]
end

# =========================================================================
# Commands
# =========================================================================

"""
    LabStep

One command of a lab run. `argv` runs with `env` added and in `dir` (local
commands only; ssh commands carry their environment in the remote script).
`stdin_argv`, when set, is a local producer piped into `argv`
(`git archive ... | ssh ... tar -x`).
"""
struct LabStep
    machine::String
    description::String
    argv::Vector{String}
    env::Vector{Pair{String, String}}
    dir::Union{Nothing, String}
    stdin_argv::Union{Nothing, Vector{String}}
end

LabStep(machine, description, argv; env = Pair{String, String}[], dir = nothing,
        stdin_argv = nothing) =
    LabStep(String(machine), String(description), String.(argv), env, dir, stdin_argv)

_shell_quote(s::AbstractString) = "'" * replace(String(s), "'" => "'\\''") * "'"

"Shell word for `s`: plain words stay bare, anything else is single-quoted."
_shell_word(s::AbstractString) = occursin(r"^[A-Za-z0-9_./=:,+@-]+$", s) ? String(s) : _shell_quote(s)

"Remote shell word for a path: `~/x` expands to the remote home."
_remote_word(path::AbstractString) =
    startswith(path, "~/") ? "\"\$HOME\"/" * _shell_quote(path[3:end]) : _shell_quote(path)

"""
    PathArg(prefix, path)

Command-line argument ending in a path (`--project=<path>`); rendered with `~/`
expanded locally, or as a remote-shell word over ssh.
"""
struct PathArg
    prefix::String
    path::String
end
PathArg(path::AbstractString) = PathArg("", String(path))

_local_arg(a::AbstractString) = String(a)
_local_arg(a::PathArg) = a.prefix * expanduser(a.path)
_remote_arg(a::AbstractString) = _shell_word(a)
_remote_arg(a::PathArg) = (isempty(a.prefix) ? "" : _shell_word(a.prefix)) * _remote_word(a.path)

function render_step(step::LabStep)
    words = String[]
    step.dir === nothing || push!(words, "cd " * Base.shell_escape(step.dir), "&&")
    for (k, v) in step.env
        push!(words, "$(k)=$(Base.shell_escape(v))")
    end
    push!(words, Base.shell_escape(step.argv...))
    cmd = join(words, " ")
    step.stdin_argv === nothing && return cmd
    return Base.shell_escape(step.stdin_argv...) * " | " * cmd
end

function _cmd(step::LabStep)
    cmd = Cmd(step.argv)
    isempty(step.env) || (cmd = addenv(cmd, step.env...))
    step.dir === nothing || (cmd = Cmd(cmd; dir = step.dir))
    step.stdin_argv === nothing && return cmd
    return pipeline(Cmd(step.stdin_argv), cmd)
end

"Work directory of a commit on a machine (`~/` kept for copied archives)."
work_directory(code::CodeDelivery, commit::AbstractString) = string(code.root, "/", commit)

"Directory on the machine receiving that machine's benchmark outputs."
output_directory(m::LabMachine, commit) = string(work_directory(m.code, commit), "/lab_results/", m.name)

_ssh_argv(host::AbstractString, script::AbstractString) = vcat(["ssh"], SSH_OPTIONS, [host, script])
_short(commit) = commit[1:min(end, 12)]

"""
    preparation_steps(machine, commit, repo, prepared) -> Vector{LabStep}

Put the benchmarked commit's source on `machine`: a local extraction into a
shared work directory (skipped when `prepared`, the set of shared work
directories already extracted in this run, holds it; wurst and curry share
theirs) or an ssh copy.
"""
function preparation_steps(m::LabMachine, commit::AbstractString, repo::AbstractString,
                           prepared::Set{String} = Set{String}())
    work = work_directory(m.code, commit)
    m.code isa SharedArchive && work in prepared && return LabStep[]
    m.code isa SharedArchive && push!(prepared, work)
    archive = ["git", "-C", repo, "archive", "--format=tar", commit]
    return _prepare(m.code, m, work, archive, commit)
end

_prepare(::SharedArchive, m::LabMachine, work, archive, commit) = [
    LabStep(m.name, "create shared work directory", ["mkdir", "-p", work]),
    LabStep(m.name, "extract git archive of $(_short(commit)) into the shared work directory",
            ["tar", "-x", "-f", "-", "-C", work]; stdin_argv = archive)]

function _prepare(::CopiedArchive, m::LabMachine, work, archive, commit)
    host = (m.access::SSHAccess).host
    w = _remote_word(work)
    return [LabStep(m.name, "copy git archive of $(_short(commit)) to $(host)",
                    _ssh_argv(host, "set -eu; mkdir -p $(w); tar -x -f - -C $(w)");
                    stdin_argv = archive)]
end

_caffeinate(m::LabMachine) = m.caffeinate ? ["caffeinate", "-i"] : String[]

"""
    julia_version_gate(machine) -> String

Julia expression that throws, naming the machine, its Julia version, and the
pin, unless the Julia evaluating it is exactly the machine's `julia_version`.
"""
julia_version_gate(m::LabMachine) =
    "VERSION == v\"$(m.julia_version)\" || error(\"lab machine $(m.name) runs Julia \$(VERSION), " *
    "but machines.toml pins julia_version = $(m.julia_version); install that version or update the " *
    "pin deliberately (a Julia change starts a new environment fingerprint on the dashboard)\")"

"Julia expression run on a machine to prepare the benchmark environment."
instantiate_expression(m::LabMachine) =
    julia_version_gate(m) * "; using Pkg; Pkg.instantiate()"

"Julia command line that checks the Julia version and instantiates the benchmark environment."
_instantiate_args(m::LabMachine, work) =
    vcat(_caffeinate(m), [PathArg(m.julia), PathArg("--project=", work * "/benchmarking"),
                          "-e", instantiate_expression(m)])

"Julia command line that runs the preset and writes the raw and dashboard JSON."
function _benchmark_args(m::LabMachine, commit, preset)
    work = work_directory(m.code, commit)
    args = Any[_caffeinate(m)..., PathArg(m.julia), "--threads=$(m.threads)"]
    m.heap_size_hint === nothing || push!(args, "--heap-size-hint=$(m.heap_size_hint)")
    append!(args, [PathArg("--project=", work * "/benchmarking"),
                   PathArg(work * "/benchmarking/run_benchmarks.jl"),
                   "--preset=$(preset)", "--backend=$(m.backend)",
                   "--float-type=$(join(m.float_types, ","))", "--commit=$(commit)"])
    m.device_label === nothing || push!(args, "--device-label=$(m.device_label)")
    push!(args, PathArg("--output=", output_directory(m, commit) * "/" * RAW_RESULTS_FILE))
    return args
end

"""
    machine_plan(machine, commit; preset, results_dir) -> Vector{LabStep}

Commands that check the machine's Julia version (`julia_version`) and
instantiate the benchmark environment on `machine`, run the
preset there, and copy both result files into `results_dir/<machine>/`.
Code preparation is separate (`preparation_steps`). `run_machine!` executes
the same commands and copies the results back even after a failed run.
"""
function machine_plan(m::LabMachine, commit::AbstractString;
                      preset::AbstractString = DEFAULT_PRESET,
                      results_dir::AbstractString)
    return vcat(_run_steps(m.access, m, commit, preset), _fetch_steps(m.code, m, commit, results_dir))
end

function _run_steps(::LocalAccess, m::LabMachine, commit, preset)
    work = work_directory(m.code, commit)
    env = copy(m.env)
    steps = LabStep[]
    if m.tmpdir !== nothing
        push!(steps, LabStep(m.name, "create TMPDIR", ["mkdir", "-p", expanduser(m.tmpdir)]))
        push!(env, "TMPDIR" => expanduser(m.tmpdir))
    end
    push!(steps, LabStep(m.name, "create output directory", ["mkdir", "-p", output_directory(m, commit)]))
    push!(steps, LabStep(m.name, "instantiate benchmark environment",
                         _local_arg.(_instantiate_args(m, work)); env, dir = work))
    push!(steps, LabStep(m.name, "run preset $(preset)",
                         _local_arg.(_benchmark_args(m, commit, preset)); env, dir = work))
    return steps
end

function _remote_script(m::LabMachine, work, lines...)
    prelude = ["set -eu", "cd $(_remote_word(work))"]
    for (k, v) in m.env
        push!(prelude, "export $(k)=$(_shell_quote(v))")
    end
    if m.tmpdir !== nothing
        push!(prelude, "mkdir -p $(_remote_word(m.tmpdir))", "export TMPDIR=$(_remote_word(m.tmpdir))")
    end
    return join(vcat(prelude, collect(lines)), "\n")
end

_remote_command(args) = join(_remote_arg.(args), " ")

function _run_steps(access::SSHAccess, m::LabMachine, commit, preset)
    work = work_directory(m.code, commit)
    instantiate = _remote_script(m, work, _remote_command(_instantiate_args(m, work)))
    run = _remote_script(m, work, "mkdir -p $(_remote_word(output_directory(m, commit)))",
                         _remote_command(_benchmark_args(m, commit, preset)))
    return [LabStep(m.name, "instantiate benchmark environment on $(access.host)", _ssh_argv(access.host, instantiate)),
            LabStep(m.name, "run preset $(preset) on $(access.host)", _ssh_argv(access.host, run))]
end

_result_sources(m::LabMachine, commit) =
    [output_directory(m, commit) * "/" * f for f in (RAW_RESULTS_FILE, DASHBOARD_RESULTS_FILE)]

# The shared root is visible on this host at the same path.
function _fetch_steps(::SharedArchive, m::LabMachine, commit, results_dir)
    dest = joinpath(results_dir, m.name)
    return [LabStep(m.name, "create local results directory", ["mkdir", "-p", dest]),
            LabStep(m.name, "copy results back", vcat(["cp"], _result_sources(m, commit), [dest * "/"]))]
end

function _fetch_steps(::CopiedArchive, m::LabMachine, commit, results_dir)
    dest = joinpath(results_dir, m.name)
    host = (m.access::SSHAccess).host
    # scp resolves relative remote paths against the remote home.
    remote(p) = host * ":" * (startswith(p, "~/") ? p[3:end] : p)
    return [LabStep(m.name, "create local results directory", ["mkdir", "-p", dest]),
            LabStep(m.name, "copy results back from $(host)",
                    vcat(["scp", "-q"], SSH_OPTIONS, remote.(_result_sources(m, commit)), [dest * "/"]))]
end

# =========================================================================
# Validation
# =========================================================================

struct LabValidationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::LabValidationError) = print(io, "LabValidationError: ", e.msg)

# Field of a parsed JSON record, whether keys are Strings (Dict) or Symbols (JSON3).
function _get(r::AbstractDict, key::AbstractString, default = nothing)
    haskey(r, key) && return r[key]
    haskey(r, Symbol(key)) && return r[Symbol(key)]
    return default
end
_get(r, key::AbstractString, default = nothing) = default

function _extra_dict(record)
    extra = _get(record, "extra")
    extra isa AbstractString || return nothing
    try
        parsed = JSON3.read(extra)
        return parsed isa AbstractDict ? parsed : nothing
    catch
        return nothing
    end
end

"""
    EnvironmentFingerprint(julia_version, manifest_sha256)

The software environment a record was measured in: the Julia version and the
sha256 of the resolved `benchmarking/Manifest.toml`. The harness writes both
into each raw record's `metadata` and each dashboard record's `extra`.
"""
struct EnvironmentFingerprint
    julia_version::String
    manifest_sha256::String
end
Base.show(io::IO, f::EnvironmentFingerprint) =
    print(io, "Julia ", f.julia_version, ", Manifest sha256 ", f.manifest_sha256)

_fingerprint(d) = d === nothing ? EnvironmentFingerprint("", "") :
    EnvironmentFingerprint(string(something(_get(d, "julia_version"), "")),
                           string(something(_get(d, "manifest_sha256"), "")))

function _fingerprint_problems(f::EnvironmentFingerprint, name)
    problems = String[]
    occursin(r"^[0-9]+\.[0-9]+\.[0-9]+", f.julia_version) ||
        push!(problems, "$(name): no Julia version recorded")
    occursin(r"^[0-9a-f]{64}$", f.manifest_sha256) ||
        push!(problems, "$(name): no Manifest sha256 recorded")
    return problems
end

"""
    environment_fingerprint(dashboard_records) -> EnvironmentFingerprint

The one environment fingerprint shared by the `extra` of every record; throws
`LabValidationError` when records disagree or lack one.
"""
function environment_fingerprint(records)
    fingerprints = unique(_fingerprint(_extra_dict(d)) for d in records)
    length(fingerprints) == 1 || throw(LabValidationError(
        "records carry $(length(fingerprints)) environment fingerprints: $(join(string.(fingerprints), "; "))"))
    problems = _fingerprint_problems(only(fingerprints), "records")
    isempty(problems) || throw(LabValidationError(join(problems, "\n  ")))
    return only(fingerprints)
end

"""
    validate_machine_results(raw, dashboard; machine, commit, expected_cases, group)

Check one machine's copied-back results: the expected number of raw records,
each `ok` or `skipped_memory` with a reason (a case that did not fit in device
memory; never in the dashboard JSON) and none failed, every record attributed
to `commit`, the machine's backend
and float types, one device label of the right hardware class (equal to the
machine's `device_label` override when set), and a dashboard JSON whose every
record carries the commit and a positive time. Every record must carry one
environment fingerprint, whose Julia version is the machine's
`julia_version`. Throws `LabValidationError` listing every problem; returns
the names of the skipped cases.
"""
function validate_machine_results(raw, dashboard; machine::LabMachine, commit::AbstractString,
                                  expected_cases::Integer, group::AbstractString)
    problems = String[]
    raw isa AbstractVector || throw(LabValidationError("$(machine.name): raw results are not a JSON array"))
    length(raw) == expected_cases ||
        push!(problems, "expected $(expected_cases) cases, found $(length(raw))")
    class = BACKEND_CLASSES[machine.backend]
    labels = Set{String}()
    fingerprints = Set{EnvironmentFingerprint}()
    skipped = String[]
    for r in raw
        name = string(_get(r, "name", "<unnamed>"))
        fingerprint = _fingerprint(_get(r, "metadata"))
        append!(problems, _fingerprint_problems(fingerprint, "case $(name)"))
        fingerprint.julia_version == string(machine.julia_version) ||
            push!(problems, "case $(name) ran on Julia $(repr(fingerprint.julia_version)), " *
                            "machines.toml pins $(machine.julia_version)")
        push!(fingerprints, fingerprint)
        status = _get(r, "status", "missing")
        if status == "skipped_memory"
            reason = _get(r, "skip_reason")
            (reason isa AbstractString && !isempty(reason)) ||
                push!(problems, "case $(name) is skipped_memory without a skip_reason")
            _get(r, "required_device_gib") isa Real ||
                push!(problems, "case $(name) is skipped_memory without required_device_gib")
            push!(skipped, name)
        elseif status != "ok"
            push!(problems, "failed case $(name): $(_get(r, "error", "status $(status)"))")
        end
        rc = string(something(_get(_get(r, "metadata", Dict()), "commit"), ""))
        rc == commit || push!(problems, "case $(name) records commit $(repr(rc)), expected $(commit)")
        _get(r, "backend") == String(machine.backend) ||
            push!(problems, "case $(name) ran on backend $(repr(_get(r, "backend"))), expected $(machine.backend)")
        _get(r, "float_type") in machine.float_types ||
            push!(problems, "case $(name) used float type $(repr(_get(r, "float_type"))), not in $(machine.float_types)")
        _get(r, "group") == group ||
            push!(problems, "case $(name) is in group $(repr(_get(r, "group"))), expected $(repr(group))")
        device = string(something(_get(r, "device_name"), ""))
        isempty(strip(device)) && push!(problems, "case $(name) has no device name")
        rclass = string(something(_get(r, "backend_class"), ""))
        startswith(rclass, class) ||
            push!(problems, "case $(name) has hardware class $(repr(rclass)), expected $(class)")
        push!(labels, string(something(_get(r, "backend_label"), "")))
    end
    length(labels) > 1 &&
        push!(problems, "records carry several device labels: $(join(sort!(collect(labels)), ", "))")
    length(fingerprints) > 1 &&
        push!(problems, "raw records carry several environment fingerprints: $(join(string.(collect(fingerprints)), "; "))")
    for label in labels
        if machine.backend === :cpu
            startswith(label, class) || push!(problems, "series label $(repr(label)) is not a CPU class")
        elseif machine.device_label !== nothing
            expected = backend_series_label(class, machine.device_label)
            label == expected || push!(problems, "series label $(repr(label)), expected $(repr(expected))")
        else
            occursin(Regex("^" * class * " [^/]+\$"), label) ||
                push!(problems, "series label $(repr(label)) does not carry a device after $(class)")
        end
    end
    append!(problems, _dashboard_problems(dashboard, commit))
    if dashboard isa AbstractVector
        dashboard_fingerprints = Set(_fingerprint(_extra_dict(d)) for d in dashboard)
        issubset(dashboard_fingerprints, fingerprints) ||
            push!(problems, "dashboard fingerprints $(join(string.(collect(dashboard_fingerprints)), "; ")) " *
                            "differ from the raw records'")
        names = Set(string(_get(d, "name", "")) for d in dashboard)
        for r in raw
            _get(r, "status") == "ok" && !(string(_get(r, "name", "")) in names) &&
                push!(problems, "case $(_get(r, "name")) is missing from the dashboard JSON")
        end
        for name in skipped
            name in names && push!(problems, "skipped case $(name) is in the dashboard JSON")
        end
    end
    isempty(problems) ||
        throw(LabValidationError("$(machine.name): " * join(problems, "\n  ")))
    return skipped
end

function _dashboard_problems(records, commit)
    problems = String[]
    records isa AbstractVector || return ["dashboard results are not a JSON array"]
    isempty(records) && push!(problems, "dashboard JSON is empty")
    seen = Set{String}()
    for d in records
        name = _get(d, "name")
        name isa AbstractString || (push!(problems, "dashboard record without a name"); continue)
        name in seen && push!(problems, "duplicate dashboard series $(repr(name))")
        push!(seen, name)
        _get(d, "unit") == "s/step" || push!(problems, "$(name): unit $(repr(_get(d, "unit"))), expected \"s/step\"")
        v = _get(d, "value")
        (v isa Real && isfinite(v) && v > 0) || push!(problems, "$(name): value $(repr(v)) is not a positive time")
        extra = _extra_dict(d)
        rc = extra === nothing ? nothing : _get(extra, "commit")
        rc == commit || push!(problems, "$(name): extra.commit $(repr(rc)), expected $(commit)")
        append!(problems, _fingerprint_problems(_fingerprint(extra), name))
    end
    fingerprints = unique(_fingerprint(_extra_dict(d)) for d in records)
    length(fingerprints) > 1 &&
        push!(problems, "records carry $(length(fingerprints)) environment fingerprints: " *
                        join(string.(fingerprints), "; "))
    return problems
end

"""
    validate_publishable_records(records, commit)

The check the Benchmarks workflow applies to a lab results file before
publishing: a non-empty array of uniquely named `s/step` records with positive
values whose `extra` JSON names `commit` and one environment fingerprint
(Julia version and Manifest sha256) shared by every record.
"""
function validate_publishable_records(records, commit::AbstractString)
    problems = _dashboard_problems(records, commit)
    isempty(problems) || throw(LabValidationError(join(problems, "\n  ")))
    return nothing
end

"""
    merge_dashboard_records(per_machine) -> Vector

Concatenate validated dashboard records of several machines (`name => records`
pairs). Two machines producing the same series name (same device label) is an
error: give one of them a `device_label`. Machines whose environment
fingerprints (`environment_fingerprint`) differ are an error too, so a lab run
is never published with mixed Julia versions or dependency resolutions.
"""
function merge_dashboard_records(per_machine)
    by_machine = [machine => environment_fingerprint(records) for (machine, records) in per_machine]
    length(unique(last.(by_machine))) <= 1 || throw(LabValidationError(
        "machines resolved different environments; nothing is merged or published:\n  " *
        join(("$(machine): $(f)" for (machine, f) in by_machine), "\n  ") *
        "\n  Pin the same julia_version and let all machines resolve the Manifest afresh " *
        "(remove <code root>/<commit>/benchmarking/Manifest.toml on each), then rerun."))
    merged = Any[]
    owner = Dict{String, String}()
    for (machine, records) in per_machine
        for r in records
            name = string(_get(r, "name"))
            haskey(owner, name) && throw(LabValidationError(
                "series $(repr(name)) produced by both $(owner[name]) and $(machine); " *
                "set device_label on one of them"))
            owner[name] = machine
            push!(merged, r)
        end
    end
    return merged
end

# =========================================================================
# Command-line options
# =========================================================================

"""
    LabOptions

Options of `run_lab_benchmarks.jl`; see `LAB_USAGE`.
"""
Base.@kwdef struct LabOptions
    machines::Vector{String} = String[]
    commit::Union{Nothing, String} = nothing
    preset::String = DEFAULT_PRESET
    machines_file::String = DEFAULT_MACHINES_FILE
    results_dir::String = DEFAULT_RESULTS_DIR
    remote::String = "origin"
    github_repo::Union{Nothing, String} = nothing
    gh::String = DEFAULT_GH
    publish::Bool = false
    allow_unpublished::Bool = false
    dry_run::Bool = false
    fetch::Bool = true
    help::Bool = false
end

const LAB_USAGE = """
usage: julia --project=benchmarking benchmarking/lab/run_lab_benchmarks.jl [options]

Benchmark one commit on the lab machines and collect validated results in
benchmarking/lab/results/<commit>/.

  --machines=a,b       machines from machines.toml, run in this order (default: all)
  --commit=REV         commit to benchmark (default: <remote>/main after fetching)
  --preset=NAME        benchmark preset (default: $(DEFAULT_PRESET))
  --publish            push the merged dashboard JSON to the $(INBOX_BRANCH) branch and
                       dispatch the $(BENCHMARK_WORKFLOW) workflow to publish it; the commit
                       must be the head of <remote>/main
  --allow-unpublished  allow a dirty tree or a commit not on <remote>/main
                       (results stay local; incompatible with --publish)
  --dry-run            print every command instead of running it (read-only git
                       queries still run)
  --no-fetch           do not fetch <remote>/main first
  --machines-file=PATH machine table (default: benchmarking/lab/machines.toml)
  --results-dir=PATH   local results root (default: benchmarking/lab/results)
  --remote=NAME        git remote of the GitHub repository (default: origin)
  --github-repo=O/R    repository the workflow is dispatched in
                       (default: parsed from the remote URL)
  --gh=PATH            GitHub CLI (default: $(DEFAULT_GH))
  -h, --help           show this help
"""

"""
    parse_lab_options(args) -> LabOptions

Parse `--key=value` options and flags; unknown options are errors.
"""
function parse_lab_options(args::AbstractVector{<:AbstractString})
    kw = Dict{Symbol, Any}()
    flags = Dict("--publish" => :publish, "--allow-unpublished" => :allow_unpublished,
                 "--dry-run" => :dry_run, "-h" => :help, "--help" => :help)
    values = Dict("--machines" => :machines, "--commit" => :commit, "--preset" => :preset,
                  "--machines-file" => :machines_file, "--results-dir" => :results_dir,
                  "--remote" => :remote, "--github-repo" => :github_repo, "--gh" => :gh)
    for arg in args
        if haskey(flags, arg)
            kw[flags[arg]] = true
        elseif arg == "--no-fetch"
            kw[:fetch] = false
        elseif occursin('=', arg) && haskey(values, first(split(arg, '='; limit = 2)))
            key, value = split(arg, '='; limit = 2)
            isempty(value) && throw(ArgumentError("$(key) needs a value"))
            field = values[key]
            kw[field] = field === :machines ? String.(split(value, ','; keepempty = false)) :
                        field in (:machines_file, :results_dir) ? abspath(value) : String(value)
        else
            throw(ArgumentError("unknown option $(repr(arg))\n\n$(LAB_USAGE)"))
        end
    end
    opts = LabOptions(; kw...)
    opts.publish && opts.allow_unpublished && throw(ArgumentError(
        "--allow-unpublished disables publishing; drop one of --publish/--allow-unpublished"))
    return opts
end

# =========================================================================
# Execution
# =========================================================================

"Executes or prints lab commands."
abstract type Executor end
"Runs commands."
struct LiveExecutor <: Executor end
"Prints commands to `io`; returns placeholders for captured output."
struct DryRunExecutor <: Executor
    io::IO
end
DryRunExecutor() = DryRunExecutor(stdout)

function execute!(::LiveExecutor, step::LabStep)
    @info "[$(step.machine)] $(step.description)" command = render_step(step)
    t0 = time()
    run(_cmd(step))
    @info "[$(step.machine)] done: $(step.description)" seconds = round(time() - t0; digits = 1)
    return nothing
end

function execute!(ex::DryRunExecutor, step::LabStep)
    println(ex.io, "[dry-run] [$(step.machine)] $(step.description)\n    $(render_step(step))")
    return nothing
end

capture(::LiveExecutor, step::LabStep, placeholder) = readchomp(_cmd(step))
capture(ex::DryRunExecutor, step::LabStep, placeholder) = (execute!(ex, step); placeholder)

_git(repo, args...) = readchomp(Cmd(vcat(["git", "-C", repo], collect(String.(args)))))
_git_ok(repo, args...) =
    success(pipeline(Cmd(vcat(["git", "-C", repo], collect(String.(args)))); stdout = devnull, stderr = devnull))

resolve_commit(repo, rev) = _git(repo, "rev-parse", "--verify", "--quiet", "$(rev)^{commit}")
function _resolve(repo, rev)
    _git_ok(repo, "rev-parse", "--verify", "--quiet", "$(rev)^{commit}") ||
        throw(ArgumentError("cannot resolve $(repr(rev)) to a commit in $(repo)"))
    return resolve_commit(repo, rev)
end
is_dirty(repo) = !isempty(_git(repo, "status", "--porcelain", "--untracked-files=no"))

"`owner/repo` of a GitHub remote URL (https or ssh form)."
function github_slug(url::AbstractString)
    m = match(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    m === nothing && throw(ArgumentError("remote URL $(repr(url)) is not a GitHub repository"))
    return "$(m.captures[1])/$(m.captures[2])"
end

"Inbox file name of one lab run: `lab/<commit>-<yyyymmddTHHMMSSZ>.json`."
inbox_path(commit::AbstractString, utc::DateTime) =
    "$(INBOX_DIRECTORY)/$(commit)-$(Dates.format(utc, dateformat"yyyymmdd\THHMMSS"))Z.json"

"Case options the commit's preset produces on `machine`."
function machine_case_options(preset_table::AbstractDict, m::LabMachine)
    return resolve_case_options(Dict{String, Any}("float_type" => m.float_types), preset_table)
end

function _preset_at_commit(repo, commit, preset)
    occursin(r"^[A-Za-z0-9_-]+$", preset) ||
        throw(ArgumentError("the lab lane runs named presets; got $(repr(preset))"))
    rel = "benchmarking/presets/$(preset).toml"
    spec = "$(commit):$(rel)"
    _git_ok(repo, "cat-file", "-e", spec) ||
        throw(ArgumentError("commit $(commit) has no $(rel); it predates that preset"))
    return parse_preset(TOML.parse(_git(repo, "show", spec)); source = "$(_short(commit)):$(rel)")
end

_read_json(path) = JSON3.read(read(path, String))

"""
    run_machine!(ex, machine, commit, repo, prepared; preset, results_dir)

Deliver the code to `machine` (`preparation_steps`), run the preset there,
and copy the result files back into `results_dir/<machine>/`. The copy runs
even when the run fails: the harness writes both JSON files before it exits
with an error for a failed case, so the partial results come back for
inspection. The run's error is then rethrown; a copy that fails after a
failed run (the harness wrote nothing) is logged and does not mask it.
"""
function run_machine!(ex::Executor, m::LabMachine, commit::AbstractString, repo::AbstractString,
                      prepared::Set{String}; preset::AbstractString, results_dir::AbstractString)
    foreach(step -> execute!(ex, step), preparation_steps(m, commit, repo, prepared))
    run_error = nothing
    try
        foreach(step -> execute!(ex, step), _run_steps(m.access, m, commit, preset))
    catch err
        err isa InterruptException && rethrow()
        run_error = err
    end
    try
        foreach(step -> execute!(ex, step), _fetch_steps(m.code, m, commit, results_dir))
    catch err
        (run_error === nothing || err isa InterruptException) && rethrow()
        @warn "[$(m.name)] could not copy result files back after the failed run" exception = err
    end
    run_error === nothing || throw(run_error)
    return nothing
end

"""
    run_lab(opts; repo) -> (commit, merged_path)

Benchmark one commit on the selected machines (sequentially), validate and
merge their results, and publish them when `opts.publish`.
"""
function run_lab(opts::LabOptions; repo::AbstractString = _git(@__DIR__, "rev-parse", "--show-toplevel"))
    ex = opts.dry_run ? DryRunExecutor() : LiveExecutor()
    machines = select_machines(load_machines(opts.machines_file), opts.machines)
    @info "Lab lane" machines = join((m.name for m in machines), ", ") preset = opts.preset dry_run = opts.dry_run publish = opts.publish

    opts.fetch && execute!(ex, LabStep("lab", "fetch $(opts.remote)/main", ["git", "-C", repo, "fetch", "--quiet", opts.remote, "main"]))
    main_ref = "$(opts.remote)/main"
    main_head = _resolve(repo, main_ref)
    commit = _resolve(repo, something(opts.commit, main_ref))
    if is_dirty(repo)
        opts.allow_unpublished || throw(ArgumentError(
            "working tree has uncommitted changes; commit or stash them, or pass --allow-unpublished"))
        @warn "Working tree is dirty; machines.toml and this runner come from the working tree, " *
              "the benchmarked code is the archive of $(commit)"
    end
    if !_git_ok(repo, "merge-base", "--is-ancestor", commit, main_ref)
        opts.allow_unpublished || throw(ArgumentError(
            "commit $(commit) is not on $(main_ref); pass --allow-unpublished to benchmark it locally"))
        @warn "Commit $(commit) is not on $(main_ref); results stay local"
    end
    opts.publish && commit != main_head && throw(ArgumentError(
        "--publish needs the head of $(main_ref) ($(main_head)); the Benchmarks workflow attributes " *
        "the dashboard entry to the dispatched commit, got $(commit)"))

    preset_table = _preset_at_commit(repo, commit, opts.preset)
    results_dir = joinpath(opts.results_dir, commit)
    @info "Benchmarking commit" commit results_dir

    outcomes = Pair{String, Any}[]
    prepared = Set{String}()
    for m in machines
        options = machine_case_options(preset_table, m)
        @info "[$(m.name)] $(m.description)" cases = expected_case_count(options)
        try
            run_machine!(ex, m, commit, repo, prepared; preset = opts.preset, results_dir)
            if !opts.dry_run
                dir = joinpath(results_dir, m.name)
                raw = _read_json(joinpath(dir, RAW_RESULTS_FILE))
                dashboard = _read_json(joinpath(dir, DASHBOARD_RESULTS_FILE))
                skipped = validate_machine_results(raw, dashboard; machine = m, commit,
                                                   expected_cases = expected_case_count(options),
                                                   group = options.group)
                push!(outcomes, m.name => dashboard)
                isempty(skipped) ||
                    @warn "[$(m.name)] $(length(skipped)) case(s) skipped for device memory (not published)" cases = skipped
                @info "[$(m.name)] results valid" records = length(dashboard) skipped = length(skipped)
            end
        catch err
            err isa InterruptException && rethrow()
            @error "[$(m.name)] lab run failed" exception = (err, catch_backtrace())
            push!(outcomes, m.name => err)
        end
    end
    opts.dry_run && return _dry_run_publish(ex, opts, repo, commit)

    failed = [name for (name, o) in outcomes if o isa Exception]
    isempty(failed) || error("lab run failed on $(join(failed, ", ")); nothing published. " *
                             "Results copied back (including partial results of failed runs) are in $(results_dir)")
    merged = merge_dashboard_records(outcomes)
    utc = now(UTC)
    merged_path = joinpath(results_dir, "lab-$(Dates.format(utc, dateformat"yyyymmdd\THHMMSS"))Z.github-action-benchmark.json")
    open(merged_path, "w") do io
        JSON3.pretty(io, merged)
        println(io)
    end
    @info "Merged lab results" merged_path records = length(merged)
    opts.publish && publish!(ex, opts, repo, commit, merged_path, inbox_path(commit, utc))
    return commit, merged_path
end

function _dry_run_publish(ex, opts, repo, commit)
    path = joinpath(opts.results_dir, commit, "lab-<utc>.github-action-benchmark.json")
    println(ex.io, "[dry-run] validate and merge results into $(path)")
    opts.publish && publish!(ex, opts, repo, commit, path, "$(INBOX_DIRECTORY)/$(commit)-<utc>.json")
    return commit, path
end

"""
    publish!(ex, opts, repo, commit, merged_path, inbox_file)

Commit `merged_path` as `inbox_file` on the orphan `$(INBOX_BRANCH)` branch
(plumbing with a temporary index; the working tree is untouched), push it, and
dispatch the Benchmarks workflow on main with `lab_results=inbox_file`.
"""
function publish!(ex::Executor, opts::LabOptions, repo, commit, merged_path, inbox_file)
    ex isa LiveExecutor && validate_publishable_records(_read_json(merged_path), commit)
    remote, inbox = opts.remote, INBOX_BRANCH
    slug = opts.github_repo === nothing ? github_slug(_git(repo, "remote", "get-url", remote)) : opts.github_repo
    git(args...) = vcat(["git", "-C", repo], collect(String.(args)))
    # Main must still point at the benchmarked commit: the workflow checks it.
    execute!(ex, LabStep("publish", "re-fetch $(remote)/main", git("fetch", "--quiet", remote, "main")))
    head = resolve_commit(repo, "$(remote)/main")
    ex isa LiveExecutor && head != commit && error(
        "$(remote)/main moved to $(head) during the lab run; results for $(commit) stay in " *
        "$(dirname(merged_path)) and cannot be attributed by the workflow")
    inbox_ref = "refs/remotes/$(remote)/$(inbox)"
    parent = if ex isa LiveExecutor
        @info "[publish] fetch $(inbox) (absent on first use)"
        fetched = success(pipeline(Cmd(git("fetch", "--quiet", remote, "+refs/heads/$(inbox):$(inbox_ref)"));
                                   stderr = devnull))
        fetched ? resolve_commit(repo, inbox_ref) : nothing
    else
        execute!(ex, LabStep("publish", "fetch $(inbox) (absent on first use)",
                             git("fetch", "--quiet", remote, "+refs/heads/$(inbox):$(inbox_ref)")))
        "<inbox-parent>"
    end
    index = tempname()
    try
        env = ["GIT_INDEX_FILE" => index]
        execute!(ex, LabStep("publish", "load inbox tree into a temporary index",
                             parent === nothing ? git("read-tree", "--empty") : git("read-tree", parent); env))
        blob = capture(ex, LabStep("publish", "store results blob", git("hash-object", "-w", merged_path)), "<blob>")
        execute!(ex, LabStep("publish", "add $(inbox_file)",
                             git("update-index", "--add", "--cacheinfo", "100644,$(blob),$(inbox_file)"); env))
        tree = capture(ex, LabStep("publish", "write inbox tree", git("write-tree"); env), "<tree>")
        parent_args = parent === nothing ? String[] : ["-p", parent]
        new = capture(ex, LabStep("publish", "commit inbox tree",
                                  git("commit-tree", tree, parent_args...,
                                      "-m", "Lab benchmark results for $(commit)")), "<inbox-commit>")
        execute!(ex, LabStep("publish", "push $(inbox)", git("push", "--quiet", remote, "$(new):refs/heads/$(inbox)")))
    finally
        rm(index; force = true)
    end
    execute!(ex, LabStep("publish", "dispatch $(BENCHMARK_WORKFLOW) to publish $(inbox_file)",
                         [opts.gh, "workflow", "run", BENCHMARK_WORKFLOW, "--repo", slug, "--ref", "main",
                          "-f", "publish=true", "-f", "run_cpu=false", "-f", "lab_results=$(inbox_file)"]))
    ex isa LiveExecutor &&
        @info "Dispatched $(BENCHMARK_WORKFLOW); follow it with `$(opts.gh) run list --workflow=$(BENCHMARK_WORKFLOW) --repo $(slug)`"
    return nothing
end

end # module AtmosTransportLabLane
