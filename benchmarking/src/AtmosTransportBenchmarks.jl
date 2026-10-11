module AtmosTransportBenchmarks

using Adapt
using ArgParse
using Dates
using JSON3
using Printf
using SHA
using Serialization
using Statistics

using AtmosTransport
using AtmosTransport: AtmosGrid, CubedSphereMesh, CubedSphereState,
    DryBasis, HybridSigmaPressure, NoAdvection, UpwindScheme, SlopesScheme,
    PPMScheme, LinRoodPPMScheme, MonotoneLimiter, NoDiffusion,
    ImplicitVerticalDiffusion, ConstantField, NoConvection, TM5Convection,
    ConvectionForcing, TransportModel, step!, fill_panel_halos!
using AtmosTransport.Architectures: CPU, GPU, architecture_label,
    device_name, ensure_runtime!, array_adapter, synchronize_architecture!
using AtmosTransport.Adjoints: CSColumnMeanObjective, cs_surface_emission_footprint
using AtmosTransport.Grids: GEOSNativePanelConvention
using AtmosTransport.Operators.Advection: NoLimiter
using AtmosTransport.State: CubedSphereFaceFluxState, DryBasis
using AtmosTransport.State.Fields: CubedSphereField
using AtmosTransport.SectionTimer

export main, run_benchmarks

include("labels.jl")
include("case_options.jl")
include("records.jl")
include("synthetic_fields.jl")
include("memory_model.jl")
include("device_errors.jl")

const _HALO = BENCHMARK_HALO

"""
    BenchmarkRun

Identity shared by every case of one harness invocation: the backend, its
hardware class (`CPU-AMD`, `GPU-CUDA`, ...), the series label that goes into
record names (`CPU-AMD`, `GPU-CUDA L40S`, ...), the full device name, the
benchmarked commit, the preset name (empty without a preset), and the
environment fingerprint: the Julia version and the sha256 of the resolved
benchmark Manifest (`_manifest_sha256`).
"""
struct BenchmarkRun
    backend::Symbol
    backend_class::String
    backend_label::String
    device_name::String
    commit::String
    preset::String
    julia_version::String
    manifest_sha256::String
end

struct BenchmarkCase
    run::BenchmarkRun
    backend::Symbol
    float_type::DataType
    grid_nc::Int
    levels::Int
    tracers::Int
    operator::Symbol
    scheme::Symbol
    steps::Int
    warmup_steps::Int
    repeats::Int
    group::String
    convection_depth::Int
end

function _parse_float_type(s::AbstractString)
    name = _parse_float_name(s)
    name == "Float32" && return Float32
    name == "Float64" && return Float64
    throw(ArgumentError("unsupported float type $(s)"))
end

function _canonical_backend(s::AbstractString)
    key = lowercase(replace(String(s), '-' => '_'))
    key in ("cpu", "processor") && return :cpu
    key in ("cuda", "gpu_cuda") && return :cuda
    key in ("metal", "gpu_metal") && return :metal
    throw(ArgumentError("unsupported backend $(s); use cpu, cuda, or metal"))
end

function _adapter(backend::Symbol)
    arch = _architecture(backend)
    ensure_runtime!(arch)
    return array_adapter(arch)
end

function _sync(backend::Symbol)
    synchronize_architecture!(_architecture(backend))
    return nothing
end

function _architecture(backend::Symbol)
    backend === :cpu && return CPU()
    backend === :cuda && return GPU(:cuda)
    backend === :metal && return GPU(:metal)
    error("unsupported backend $(backend)")
end

function _cpu_vendor()
    model = try
        isempty(Sys.cpu_info()) ? "" : Sys.cpu_info()[1].model
    catch
        ""
    end
    occursin(r"AMD|EPYC|Ryzen"i, model) && return "AMD"
    occursin(r"Intel|Xeon|Core"i, model) && return "Intel"
    occursin(r"Apple|M[0-9]"i, model) && return "Apple"
    return "CPU"
end

function _backend_class(backend::Symbol)
    backend === :cpu && return "CPU-$(_cpu_vendor())"
    backend === :cuda && return "GPU-CUDA"
    backend === :metal && return "GPU-Metal"
    error("unsupported backend $(backend)")
end

_device_name(::Val{:cpu}) = _cpu_model()
# GPU series names carry the device, so a GPU run without one cannot be named.
_device_name(::Val{B}) where {B} = device_name(_architecture(B))
_device_name(backend::Symbol) = _device_name(Val(backend))

function _cpu_model()
    try
        isempty(Sys.cpu_info()) ? "CPU" : strip(Sys.cpu_info()[1].model)
    catch
        "CPU"
    end
end

"""
    _benchmark_run(backend; device_label, commit, preset) -> BenchmarkRun

Resolve the run identity. GPU series labels append the device label, which is
`normalize_device_label(device_name)` unless `device_label` overrides it; CPU
labels are the hardware class alone.
"""
function _benchmark_run(backend::Symbol; device_label = nothing,
                        commit::AbstractString = "", preset::AbstractString = "")
    class = _backend_class(backend)
    device = _device_name(backend)
    label = if backend === :cpu
        device_label === nothing ||
            @warn "--device-label is ignored for CPU runs; CPU series keep their class label" class
        backend_series_label(class, nothing)
    else
        backend_series_label(class, device_label === nothing ?
                                    normalize_device_label(device) : device_label)
    end
    return BenchmarkRun(backend, class, label, String(device),
                        isempty(commit) ? _git_commit() : String(commit), String(preset),
                        string(VERSION), _manifest_sha256())
end

"""
    _manifest_sha256() -> String

sha256 (hex) of the Manifest the active benchmark environment resolved to;
empty when the environment has no Manifest. Two runs with the same Julia
version and Manifest hash used the same package versions.
"""
function _manifest_sha256()
    project = Base.active_project()
    manifest = project === nothing ? nothing : Base.project_file_manifest_path(project)
    manifest === nothing && return ""
    return bytes2hex(open(sha256, manifest))
end

function _scheme(sym::Symbol)
    sym === :upwind && return UpwindScheme()
    sym === :slopes && return SlopesScheme(MonotoneLimiter())
    sym === :ppm && return PPMScheme(MonotoneLimiter())
    sym === :linrood5 && return LinRoodPPMScheme(5)
    sym === :linrood7 && return LinRoodPPMScheme(7)
    throw(ArgumentError("unsupported scheme $(sym)"))
end

function _haloed_panel(::Type{FT}, Nc::Int, Hp::Int, Nz::Int) where {FT}
    return zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
end

function _make_tracer(panels_m, mesh::CubedSphereMesh, ::Type{FT}, tracer_idx::Int) where {FT}
    Nc, Hp = mesh.Nc, mesh.Hp
    panels_rm = ntuple(_ -> similar(panels_m[1]), 6)
    @inbounds for p in 1:6
        m = panels_m[p]
        rm = panels_rm[p]
        fill!(rm, zero(FT))
        for k in axes(m, 3), j in 1:Nc, i in 1:Nc
            ii = Hp + i
            jj = Hp + j
            x = FT(i - 1) / max(FT(Nc - 1), one(FT))
            y = FT(j - 1) / max(FT(Nc - 1), one(FT))
            z = FT(k - 1) / max(FT(size(m, 3) - 1), one(FT))
            q = FT(1e-6) * FT(tracer_idx) *
                (FT(1) + FT(0.15) * sinpi(FT(2) * x + FT(0.1) * p) +
                 FT(0.10) * cospi(FT(2) * y) + FT(0.05) * z)
            rm[ii, jj, k] = m[ii, jj, k] * q
        end
    end
    fill_panel_halos!(panels_rm, mesh; dir = 0)
    return panels_rm
end

"""
    _synthetic_fields(FT, mesh, Nz, case_name) -> (panels_m, am, bm, cm)

Host arrays of the synthetic air mass (halos filled) and the steady face mass
fluxes (`synthetic_fields.jl`). Throws when their palindrome Courant number
exceeds `MAX_SYNTHETIC_COURANT`.
"""
function _synthetic_fields(::Type{FT}, mesh::CubedSphereMesh, Nz::Int,
                           case_name::AbstractString) where {FT}
    Nc, Hp = mesh.Nc, mesh.Hp
    panels_m = ntuple(_ -> _haloed_panel(FT, Nc, Hp, Nz), 6)
    fill_synthetic_mass!(panels_m, Nc, Hp)
    fill_panel_halos!(panels_m, mesh; dir = 0)
    N = Nc + 2Hp
    am = ntuple(_ -> zeros(FT, N + 1, N, Nz), 6)
    bm = ntuple(_ -> zeros(FT, N, N + 1, Nz), 6)
    cm = ntuple(_ -> zeros(FT, N, N, Nz + 1), 6)
    fill_synthetic_fluxes!(am, bm, cm, panels_m, Nc, Hp)
    check_synthetic_courant(synthetic_courant(panels_m, am, bm, cm, Nc, Hp), case_name;
                            stage = "of the initial fields")
    return panels_m, am, bm, cm
end

"""
    _check_steady_case(model, case, m0; stage)

The synthetic fluxes leave the air mass unchanged, so a case keeps its
initial Courant number. Throws, naming the case and `stage`, when the model's
air mass drifted from `m0` by more than `MAX_SYNTHETIC_MASS_DRIFT` or its
Courant number exceeds `MAX_SYNTHETIC_COURANT`. Copies to the host; call it
outside the timed region.
"""
function _check_steady_case(model, case::BenchmarkCase, m0; stage::AbstractString)
    Nc, Hp = case.grid_nc, _HALO
    interior = (Hp + 1):(Hp + Nc)
    m = map(Array, model.state.air_mass)
    drift = maximum(maximum(abs.(m[p][interior, interior, :] .- m0[p][interior, interior, :]) ./
                            m0[p][interior, interior, :]) for p in 1:6)
    fluxes = model.fluxes
    courant = synthetic_courant(m, map(Array, fluxes.am), map(Array, fluxes.bm),
                                map(Array, fluxes.cm), Nc, Hp)
    check_synthetic_courant(courant, _case_name(case); stage)
    drift <= MAX_SYNTHETIC_MASS_DRIFT || error(
        "synthetic case $(_case_name(case)): air mass drifted by $(round(drift; sigdigits = 3)) " *
        "(relative) $(stage), more than $(MAX_SYNTHETIC_MASS_DRIFT); the steps did not run the " *
        "steady workload the case is named after")
    return nothing
end

function _vertical_grid(::Type{FT}, Nz::Int) where {FT}
    A_ifc = zeros(FT, Nz + 1)
    B_ifc = FT.(range(FT(0), FT(1); length = Nz + 1))
    return HybridSigmaPressure(A_ifc, B_ifc)
end

function _build_model(case::BenchmarkCase)
    FT = case.float_type
    FT === Float64 && case.backend === :metal &&
        throw(ArgumentError("Metal benchmark runs require Float32"))

    Nc, Nz, Nt = case.grid_nc, case.levels, case.tracers
    mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = _HALO,
                           convention = GEOSNativePanelConvention())
    grid = AtmosGrid(mesh, _vertical_grid(FT, Nz), _architecture(case.backend); FT = FT)

    panels_m_cpu, am_cpu, bm_cpu, cm_cpu = _synthetic_fields(FT, mesh, Nz, _case_name(case))
    tracers_cpu = ntuple(t -> _make_tracer(panels_m_cpu, mesh, FT, t), Nt)

    adapter = _adapter(case.backend)
    panels_m = Adapt.adapt(adapter, panels_m_cpu)
    tracer_names = ntuple(t -> Symbol("tr", t), Nt)
    tracer_values = ntuple(t -> Adapt.adapt(adapter, tracers_cpu[t]), Nt)
    state = CubedSphereState(DryBasis, mesh, panels_m; NamedTuple{tracer_names}(tracer_values)...)
    fluxes = CubedSphereFaceFluxState{DryBasis}(
        Adapt.adapt(adapter, am_cpu),
        Adapt.adapt(adapter, bm_cpu),
        Adapt.adapt(adapter, cm_cpu))

    advection = case.operator === :diffusion || case.operator === :convection ? NoAdvection() : _scheme(case.scheme)
    diffusion = case.operator in (:diffusion, :full) ?
        ImplicitVerticalDiffusion(; kz_field = CubedSphereField(ConstantField{FT, 3}(FT(1.0)))) :
        NoDiffusion()
    # use_collab_lu=true is the production-representative path. The
    # collab-LU kernel parallelises the inner tridiagonal solve across the
    # GPU's wide warps; with use_collab_lu=false the per-column solve runs
    # serial inside each warp, making convection ~5x slower on GPU and
    # distorting the convection:advection ratio relative to CPU. See
    # `tm5_convection_perf_findings_2026_05_22.md` for the measurement.
    # `lmax_conv` caps the matrix to the bottom levels like production L137
    # runs; Float32 GPU collaborative solves need a depth of at most 85.
    convection = case.operator in (:convection, :full) ?
        TM5Convection(; use_collab_lu = true,
                      lmax_conv = effective_convection_depth(case.convection_depth, Nz)) :
        NoConvection()
    convection_forcing = _convection_forcing(case, mesh, adapter)

    model = TransportModel(state, fluxes, grid, advection;
                           diffusion = diffusion,
                           convection = convection,
                           convection_forcing = convection_forcing)
    # The model constructor allocates topology metrics from the CPU mesh.
    # Adapt the composed model once so workspace internals such as CS
    # convection cell areas live on the requested backend too.
    model = case.backend === :cpu ? model : Adapt.adapt(adapter, model)
    if !(diffusion isa NoDiffusion)
        dz = model.workspace.diffusion_ws.layer_thickness
        if dz isa Tuple
            foreach(panel -> fill!(panel, FT(100.0)), dz)
        else
            fill!(dz, FT(100.0))
        end
    end
    return model
end

"""
    _build_cs_payload(case)

Fields of the `io` and `adjoint` cases on the case's backend: the mesh, the
air mass, `case.tracers` tracer masses (`tracers`, one six-panel tuple per
tracer), and the three face-flux arrays.
"""
function _build_cs_payload(case::BenchmarkCase)
    FT, Nc, Nz, Nt = case.float_type, case.grid_nc, case.levels, case.tracers
    mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = _HALO,
                           convention = GEOSNativePanelConvention())
    panels_m_cpu, am_cpu, bm_cpu, cm_cpu = _synthetic_fields(FT, mesh, Nz, _case_name(case))
    tracers_cpu = ntuple(t -> _make_tracer(panels_m_cpu, mesh, FT, t), Nt)

    adapter = _adapter(case.backend)
    return (mesh = mesh,
            panels_m = Adapt.adapt(adapter, panels_m_cpu),
            tracers = Adapt.adapt(adapter, tracers_cpu),
            panels_am = Adapt.adapt(adapter, am_cpu),
            panels_bm = Adapt.adapt(adapter, bm_cpu),
            panels_cm = Adapt.adapt(adapter, cm_cpu))
end

# Device arrays are copied to the host for serialization; host arrays are
# serialized in place (no second copy of the payload).
_hostify(x::Array) = x
_hostify(x::AbstractArray) = Array(x)
_hostify(x::Tuple) = map(_hostify, x)
_hostify(x::NamedTuple) = map(_hostify, x)
_hostify(x) = x

function _write_panel_payload(path::AbstractString, payload)
    open(path, "w") do io
        serialize(io, _hostify(payload))
    end
    return filesize(path)
end

function _convection_forcing(case::BenchmarkCase, mesh, adapter)
    case.operator in (:convection, :full) || return ConvectionForcing()
    FT, Nc, Nz = case.float_type, case.grid_nc, case.levels
    band = convection_forcing_band(Nz, case.convection_depth)
    entu = ntuple(_ -> begin
        a = zeros(FT, Nc, Nc, Nz)
        a[:, :, band] .= FT(2e-5)
        a
    end, 6)
    detu = ntuple(_ -> begin
        a = zeros(FT, Nc, Nc, Nz)
        a[:, :, band] .= FT(2e-5)
        a
    end, 6)
    entd = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    detd = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    return Adapt.adapt(adapter, ConvectionForcing(nothing, nothing, (; entu, detu, entd, detd)))
end

function _phase_summary_from_csv(path)
    phases = Dict{String, Float64}()
    isfile(path) || return phases
    for (line_no, line) in enumerate(eachline(path))
        line_no == 1 && continue
        cols = split(line, ",")
        length(cols) >= 3 || continue
        phases[String(cols[1])] = parse(Float64, cols[3])
    end
    return phases
end

function _time_io_case(case::BenchmarkCase)
    payload = _build_cs_payload(case)
    times = Float64[]
    bytes = 0
    tmpdir = mktempdir(; prefix = "at_bench_io_")
    try
        for _ in 1:case.warmup_steps
            path = joinpath(tmpdir, "warmup.bin")
            _sync(case.backend)
            _write_panel_payload(path, payload)
            open(deserialize, path)
            rm(path; force = true)
        end
        for r in 1:case.repeats
            path = joinpath(tmpdir, "payload_$(r).bin")
            _sync(case.backend)
            t0 = time_ns()
            bytes = _write_panel_payload(path, payload)
            open(deserialize, path)
            _sync(case.backend)
            push!(times, (time_ns() - t0) / 1e9)
            rm(path; force = true)
        end
    finally
        rm(tmpdir; force = true, recursive = true)
    end
    # One repeat is one round trip (`timed_steps_per_repeat`), so the median
    # repeat time is the time per step; the phase total covers all repeats.
    median_s = median(times)
    mad_s = median(abs.(times .- median_s))
    return _result_dict(case, median_s, mad_s, Dict("io" => sum(times)),
                        bytes / median_s;
                        metric_name = "bytes_per_second")
end

# The footprint of one scalar objective of one tracer; `case_matrix` runs
# `adjoint` at one tracer only (`SINGLE_TRACER_OPERATORS`).
function _time_adjoint_case(case::BenchmarkCase)
    payload = _build_cs_payload(case)
    panels_rm = only(payload.tracers)
    nsteps = timed_steps_per_repeat(case.operator, case.steps)
    panels_am_steps = [payload.panels_am for _ in 1:nsteps]
    panels_bm_steps = [payload.panels_bm for _ in 1:nsteps]
    panels_cm_steps = [payload.panels_cm for _ in 1:nsteps]
    obj_i = max(1, min(2, case.grid_nc))
    objective = CSColumnMeanObjective(1, obj_i, obj_i)
    scheme = PPMScheme(NoLimiter())
    dt = case.float_type(600)

    for _ in 1:case.warmup_steps
        cs_surface_emission_footprint(panels_rm, payload.panels_m,
                                      panels_am_steps, panels_bm_steps,
                                      panels_cm_steps, payload.mesh, objective;
                                      scheme = scheme, dt = dt, tape_storage = :device)
    end

    times = Float64[]
    for _ in 1:case.repeats
        _sync(case.backend)
        t0 = time_ns()
        result = cs_surface_emission_footprint(panels_rm, payload.panels_m,
                                               panels_am_steps, panels_bm_steps,
                                               panels_cm_steps, payload.mesh, objective;
                                               scheme = scheme, dt = dt,
                                               tape_storage = :device)
        _sync(case.backend)
        push!(times, (time_ns() - t0) / 1e9)
        result === nothing && error("adjoint benchmark returned no result")
    end
    # One repeat is one forward and reverse run over `nsteps` steps; the phase
    # total covers all repeats, like the section times of the model cases.
    median_s = median(times)
    mad_s = median(abs.(times .- median_s))
    return _result_dict(case, median_s / nsteps, mad_s / nsteps,
                        Dict("adjoint_forward_reverse" => sum(times)),
                        1 / median_s;
                        metric_name = "adjoint_runs_per_second")
end

function _time_case(case::BenchmarkCase)
    case.operator === :io && return _time_io_case(case)
    case.operator === :adjoint && return _time_adjoint_case(case)

    model = _build_model(case)
    m0 = map(Array, model.state.air_mass)
    dt = case.float_type(600)
    for _ in 1:case.warmup_steps
        step!(model, dt)
    end
    _sync(case.backend)
    _check_steady_case(model, case, m0; stage = "after $(case.warmup_steps) warmup step(s)")

    SectionTimer.enable!()
    times = Float64[]
    try
        for _ in 1:case.repeats
            _sync(case.backend)
            t0 = time_ns()
            for _ in 1:case.steps
                step!(model, dt)
            end
            _sync(case.backend)
            push!(times, (time_ns() - t0) / 1e9 / case.steps)
        end
    finally
        SectionTimer.disable!()
    end
    _check_steady_case(model, case, m0;
                       stage = "after $(case.warmup_steps + case.steps * case.repeats) step(s)")
    csv_path = tempname() * ".csv"
    SectionTimer.write_csv(csv_path)
    phase_totals = _phase_summary_from_csv(csv_path)
    rm(csv_path; force = true)

    median_s = median(times)
    mad_s = median(abs.(times .- median_s))
    cells = 6 * case.grid_nc^2 * case.levels
    cell_tracer_updates = cells * case.tracers / median_s

    return _result_dict(case, median_s, mad_s, phase_totals, cell_tracer_updates;
                        metric_name = "cell_tracer_updates_per_second")
end

function _result_dict(case::BenchmarkCase, median_s::Real, mad_s::Real,
                      phase_totals, throughput::Real; metric_name::AbstractString)
    cells = 6 * case.grid_nc^2 * case.levels
    return Dict(
        "name" => _case_name(case),
        "status" => "ok",
        "group" => case.group,
        "backend_class" => case.run.backend_class,
        "backend_label" => case.run.backend_label,
        "backend" => String(case.backend),
        "device_name" => case.run.device_name,
        "float_type" => string(case.float_type),
        "grid" => "C$(case.grid_nc)",
        "levels" => case.levels,
        "tracers" => case.tracers,
        "operator" => String(case.operator),
        "scheme" => String(case.scheme),
        "convection_depth" => effective_convection_depth(case.convection_depth, case.levels),
        "steps_per_repeat" => timed_steps_per_repeat(case.operator, case.steps),
        "repeats" => case.repeats,
        "warmup_steps" => case.warmup_steps,
        "time_per_step_seconds" => median_s,
        "mad_seconds" => mad_s,
        "steps_per_second" => 1 / median_s,
        metric_name => throughput,
        "cell_tracer_updates_per_second" =>
            metric_name == "cell_tracer_updates_per_second" ?
            throughput : cells * case.tracers / median_s,
        "phase_times_seconds" => phase_totals,
        "metadata" => _metadata(case),
    )
end

function _metadata(case::BenchmarkCase)
    return Dict(
        "timestamp_utc" => string(Dates.now(Dates.UTC)),
        "julia_version" => case.run.julia_version,
        "manifest_sha256" => case.run.manifest_sha256,
        "julia_threads" => Threads.nthreads(),
        "atmostransport_version" => string(pkgversion(AtmosTransport)),
        "commit" => case.run.commit,
        "preset" => case.run.preset,
        "architecture" => try
            architecture_label(_architecture(case.backend))
        catch
            case.run.backend_class
        end,
    )
end

"Raw record of a case that threw (`err`) or was skipped (`err === nothing`); never published."
function _failed_result(case::BenchmarkCase, err)
    return Dict(
        "name" => _case_name(case),
        "status" => "failed",
        "error" => err === nothing ? nothing : first(sprint(showerror, err), 2000),
        "group" => case.group,
        "backend_class" => case.run.backend_class,
        "backend_label" => case.run.backend_label,
        "backend" => String(case.backend),
        "device_name" => case.run.device_name,
        "float_type" => string(case.float_type),
        "grid" => "C$(case.grid_nc)",
        "levels" => case.levels,
        "tracers" => case.tracers,
        "operator" => String(case.operator),
        "scheme" => String(case.scheme),
        "metadata" => _metadata(case),
    )
end

# Commit of the checkout holding this harness; empty in a git archive (the lab
# lane passes --commit there).
function _git_commit()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    try
        chomp(read(pipeline(`git -C $root rev-parse HEAD`; stderr = devnull), String))
    catch
        ""
    end
end

_case_name(case::BenchmarkCase) =
    case_series_name(; group = case.group, grid = "C$(case.grid_nc)",
                     levels = case.levels, operator = String(case.operator),
                     tracers = case.tracers, scheme = String(case.scheme),
                     backend_label = case.run.backend_label,
                     float_type = string(case.float_type))

function _write_outputs(raw_path::AbstractString, results)
    mkpath(dirname(abspath(raw_path)))
    open(raw_path, "w") do io
        JSON3.pretty(io, results)
        println(io)
    end
    dashboard_path = replace(raw_path, r"\.json$" => ".github-action-benchmark.json")
    dashboard_path == raw_path && (dashboard_path = raw_path * ".github-action-benchmark.json")
    open(dashboard_path, "w") do io
        JSON3.pretty(io, dashboard_records(results))
        println(io)
    end
    return raw_path, dashboard_path
end

# Case options default to `nothing` so a preset can fill them; the built-in
# values live in `DEFAULT_CASE_OPTIONS` (case_options.jl).
function _settings()
    d = DEFAULT_CASE_OPTIONS
    s = ArgParseSettings(description = "AtmosTransport benchmark runner")
    @add_arg_table! s begin
        "--backend"
            help = "Backend: cpu, cuda, or metal."
            default = "cpu"
        "--preset"
            help = "Named case matrix from benchmarking/presets (e.g. lab) or a preset .toml path. " *
                   "Explicit case options override the preset."
        "--float-type"
            help = "Comma-separated Float32/Float64 list (default $(join(d["float_type"], ",")))."
        "--grid"
            help = "Comma-separated CS grid sizes, e.g. C24,C90,C180 (default $(join(d["grid"], ",")))."
        "--levels"
            help = "Comma-separated vertical level counts, e.g. 72,137 (default $(join(d["levels"], ",")))."
        "--tracers"
            help = "Comma-separated tracer counts (default $(join(d["tracers"], ",")))."
        "--operator"
            help = "Comma-separated operators: $(join(SUPPORTED_OPERATORS, ",")) (default $(join(d["operator"], ",")))."
        "--scheme"
            help = "Comma-separated advection schemes: $(join(SUPPORTED_SCHEMES, ", ")) (default $(join(d["scheme"], ",")))."
        "--steps"
            help = "Steps per timed repeat (default $(d["steps"]))."
            arg_type = Int
        "--warmup-steps"
            help = "Warmup steps before timing (default $(d["warmup_steps"]))."
            arg_type = Int
        "--repeats"
            help = "Timed repeats (default $(d["repeats"]))."
            arg_type = Int
        "--convection-depth"
            help = "Bottom levels in the TM5 convection matrix, clipped to each case's levels; " *
                   "0 = full column (default $(d["convection_depth"]))."
            arg_type = Int
        "--group"
            help = "Dashboard group name (default $(repr(d["group"])))."
        "--device-label"
            help = "GPU device label for series names, overriding the normalised device name " *
                   "(e.g. \"A100 80GB\"). Ignored for CPU runs."
        "--device-memory-gib"
            help = "Device memory available to each case, in GiB, instead of querying the device " *
                   "(free CUDA memory; Metal's recommended working set minus its allocations; free " *
                   "host memory on the CPU). Cases estimated to need more are recorded as skipped_memory."
            arg_type = Float64
        "--commit"
            help = "Benchmarked commit recorded in each result (default: git rev-parse HEAD). " *
                   "Required when running from a git archive without .git."
            default = ""
        "--output"
            help = "Raw JSON output path. A github-action-benchmark JSON is written next to it."
            default = "benchmark_results.json"
    end
    return s
end

const _CASE_OPTION_FLAGS = ("float-type", "grid", "levels", "tracers", "operator", "scheme",
                            "steps", "warmup-steps", "repeats", "convection-depth", "group")

"""
    case_options(parsed) -> (CaseOptions, preset_name)

Resolve parsed command-line options against the optional preset and the
built-in defaults.
"""
function case_options(parsed::AbstractDict)
    preset_name = parsed["preset"] === nothing ? "" : String(parsed["preset"])
    preset = isempty(preset_name) ? nothing : load_preset(preset_name)
    overrides = Dict{String, Any}(replace(flag, "-" => "_") => parsed[flag]
                                  for flag in _CASE_OPTION_FLAGS)
    return resolve_case_options(overrides, preset), preset_name
end

function _cases(args)
    parsed = parse_args(args, _settings())
    backend = _canonical_backend(parsed["backend"])
    options, preset_name = case_options(parsed)
    if backend === :metal && "Float64" in options.float_types
        throw(ArgumentError("Metal benchmark runs require Float32; drop Float64 from --float-type"))
    end
    run = _benchmark_run(backend; device_label = parsed["device-label"],
                         commit = parsed["commit"], preset = preset_name)
    cases = [BenchmarkCase(run, backend, _parse_float_type(c.float_type), c.grid, c.levels,
                           c.tracers, c.operator, c.scheme, options.steps,
                           options.warmup_steps, options.repeats, options.group,
                           options.convection_depth)
             for c in case_matrix(options)]
    override = parsed["device-memory-gib"]
    override === nothing || override > 0 ||
        throw(ArgumentError("--device-memory-gib must be positive, got $(override)"))
    memory_override = override === nothing ? nothing : round(Int, override * 2^30)
    return cases, parsed["output"], memory_override
end

# =========================================================================
# Device memory
# =========================================================================

# The backend modules are loaded into Main by run_benchmarks.jl before this
# package, so they are reached through Main rather than as dependencies.
_main_module(name::Symbol) = Base.invokelatest(getfield, Main, name)

_query_available_bytes(::Val{:cpu}) = Int(Sys.free_memory())
_query_available_bytes(::Val{:cuda}) =
    Int(Base.invokelatest(getproperty(_main_module(:CUDA), :available_memory)))
function _query_available_bytes(::Val{:metal})
    device = Base.invokelatest(getproperty(_main_module(:Metal), :device))
    budget = Base.invokelatest(getproperty, device, :recommendedMaxWorkingSetSize)
    allocated = Base.invokelatest(getproperty, device, :currentAllocatedSize)
    return Int(budget) - Int(allocated)
end

"""
    _available_device_bytes(backend, override) -> Union{Int, Nothing}

Memory the next case may use: `override` (`--device-memory-gib`) when given,
else the device's free memory (`_query_available_bytes`). `nothing` when the
query fails; the case then runs without the memory check, and an
out-of-memory error still records it as skipped.
"""
function _available_device_bytes(backend::Symbol, override)
    override === nothing || return override
    try
        return _query_available_bytes(Val(backend))
    catch err
        @warn "Could not query the device memory; cases run without the memory check" backend exception = err maxlog = 1
        return nothing
    end
end

"Return freed device memory to the driver after a case or an out-of-memory error."
function _reclaim_device!(backend::Symbol)
    GC.gc(true)
    if backend === :cuda
        try
            Base.invokelatest(getproperty(_main_module(:CUDA), :reclaim))
        catch err
            @warn "CUDA.reclaim failed" exception = err maxlog = 1
        end
    end
    return nothing
end

"""
    _tm5_collaborative(case) -> Bool

Whether the case's TM5 convection runs the collaborative GPU solve, which
needs no column tile: a GPU backend and an effective convection depth within
the backend's limit for the float type.
"""
function _tm5_collaborative(case::BenchmarkCase)
    case.backend === :cpu && return false
    depth = effective_convection_depth(case.convection_depth, case.levels)
    depth = depth == 0 ? case.levels : depth
    try
        backend = AtmosTransport.Architectures.device(_architecture(case.backend))
        return 0 < depth <= AtmosTransport.Operators.Convection._tm5_collab_max_depth(case.float_type, backend)
    catch
        return false   # unknown: count the tile (conservative)
    end
end

"Device memory the case needs: the memory-model estimate with its safety margin."
_required_device_bytes(case::BenchmarkCase) =
    required_device_bytes(estimate_case_bytes(case.operator, sizeof(case.float_type), case.grid_nc,
                                              case.levels, case.tracers; steps = case.steps,
                                              tm5_legacy_tile = !_tm5_collaborative(case),
                                              device_is_host = case.backend === :cpu))

"""
    _run_case(case) -> Dict

`_time_case`, checked for device errors that do not raise. On Metal the
case's standard output is captured (`capture_stdout`, `device_errors.jl`)
and the command buffers Metal.jl reports as failed decide the outcome: an
out-of-memory buffer gives a `skipped_memory` record, any other failure a
`failed` one. Exceptions propagate to the caller as before.
"""
function _run_case(case::BenchmarkCase)
    case.backend === :metal || return _time_case(case)
    outcome, text = capture_stdout() do
        try
            _time_case(case)
        finally
            _drain_metal_command_buffers()
        end
    end
    status, message = command_buffer_outcome(command_buffer_failures(text))
    if status == "skipped_memory"
        @warn "Metal command buffer ran out of memory; case recorded as skipped" case=_case_name(case) reason=message
        return _skipped_result(case, message)
    elseif status == "failed"
        @error "Metal command buffer failed; case recorded as failed" case=_case_name(case) reason=message
        return _failed_result(case, ErrorException(message))
    end
    outcome isa CaughtError && throw(CapturedException(outcome.exception, outcome.backtrace))
    return outcome
end

"Seconds the Metal completion barrier waits before it fails the case."
const METAL_BARRIER_TIMEOUT = 600.0

"""
    _drain_metal_command_buffers()

Completion barrier (`completion_barrier`) after a Metal case: commit an
empty sentinel command buffer on the queue the case's kernels used (Metal.jl's
task-local `global_queue(device())`, so this runs on the case's task) with a
completion handler (`MTL.on_completed`) that signals the barrier. When it
returns, every completion handler of the case has run, so its error reports
are in the captured output and cannot reach the next case.
"""
function _drain_metal_command_buffers()
    metal = _main_module(:Metal)
    completion_barrier(timeout = METAL_BARRIER_TIMEOUT) do signal
        Base.invokelatest() do
            mtl = getproperty(metal, :MTL)
            # Like Metal.synchronize: create and commit inside an autorelease pool.
            metal.NSAutoreleasePool() do
                queue = metal.global_queue(metal.device())
                sentinel = mtl.MTLCommandBuffer(queue)
                mtl.on_completed(_ -> signal(), sentinel)
                mtl.commit!(sentinel)
            end
        end
    end
    return nothing
end

"Raw record of a case that was not run because it does not fit in device memory."
function _skipped_result(case::BenchmarkCase, reason::AbstractString)
    r = _failed_result(case, nothing)
    delete!(r, "error")
    r["status"] = "skipped_memory"
    r["skip_reason"] = String(reason)
    return r
end

"""
    run_benchmarks(args = ARGS) -> Vector

Run every case. A case whose estimated device memory
(`_required_device_bytes`) exceeds the available memory, or that runs out
of memory, is recorded with `"status" => "skipped_memory"` and a
`skip_reason`, and the run continues. A case that throws otherwise is
logged and kept with `"status" => "failed"`, and makes `main` fail after the
outputs are written. Only `"ok"` cases reach the dashboard JSON.
"""
function run_benchmarks(args = ARGS)
    cases, output, memory_override = _cases(args)
    isempty(cases) || @info "Benchmark run" backend=cases[1].backend series_label=cases[1].run.backend_label device=cases[1].run.device_name commit=cases[1].run.commit preset=cases[1].run.preset julia=cases[1].run.julia_version manifest_sha256=cases[1].run.manifest_sha256 ncases=length(cases)
    results = Any[]
    for (i, case) in enumerate(cases)
        @info "Running benchmark $(i)/$(length(cases))" backend=case.backend float_type=case.float_type grid=case.grid_nc levels=case.levels tracers=case.tracers operator=case.operator
        required = _required_device_bytes(case)
        available = _available_device_bytes(case.backend, memory_override)
        skip = memory_skip_reason(required, available)
        r = if skip !== nothing
            @warn "Skipping benchmark case: it does not fit in device memory" case=_case_name(case) reason=skip
            _skipped_result(case, skip)
        else
            try
                _run_case(case)
            catch err
                err isa InterruptException && rethrow()
                if is_out_of_memory(err)
                    @warn "Benchmark case ran out of memory; recorded as skipped" case=_case_name(case) exception=err
                    _reclaim_device!(case.backend)
                    _skipped_result(case, "ran out of device memory: " * first(sprint(showerror, err), 500))
                else
                    @error "Benchmark case failed" case=_case_name(case) exception=(err, catch_backtrace())
                    _failed_result(case, err)
                end
            end
        end
        r["required_device_gib"] = gib(required)
        r["available_device_gib"] = available === nothing ? nothing : gib(available)
        push!(results, r)
        if r["status"] == "ok"
            @printf("%-70s %10.6f s/step  %.3e cell-tracer/s\n",
                    r["name"], r["time_per_step_seconds"],
                    r["cell_tracer_updates_per_second"])
        elseif r["status"] == "skipped_memory"
            @printf("%-70s skipped: %s\n", r["name"], r["skip_reason"])
        end
        # Return the case's device buffers before the next (possibly larger) case.
        _reclaim_device!(case.backend)
    end
    raw, dashboard = _write_outputs(output, results)
    @info "Wrote benchmark results" raw dashboard
    return results
end

function main(args = ARGS)
    results = run_benchmarks(args)
    skipped = [r["name"] for r in results if r["status"] == "skipped_memory"]
    isempty(skipped) ||
        @warn "$(length(skipped)) benchmark case(s) skipped for device memory (not published)" cases = skipped
    failed = [r["name"] for r in results if r["status"] == "failed"]
    isempty(failed) ||
        error("$(length(failed)) benchmark case(s) failed:\n  " * join(failed, "\n  "))
    return results
end

end
