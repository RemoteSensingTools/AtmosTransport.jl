# Bitwise gate of the cubed-sphere adjoint tape (core/test_cs_adjoint_tape_bitwise.jl).
#
# Every tape storage and checkpoint schedule (`AdjointTape`) must give the same
# gradients, bit for bit, as the default tape (device storage, `FullCheckpoint`;
# the reference variant): footprints and the initial adjoint of the objective,
# seeded and multi-objective (Jacobian) entry points, the 4D-Var cost and
# gradient (layer and column means with a co-sample after the last assimilated
# step, and, for the cases marked `operators`, the observation operators: time
# windows, placed points, averaging-kernel columns with the rebin and the hat
# projection, a cell average, a column the surface-pressure screen leaves out,
# background terms and the preconditioned space), and a two-iteration 4D-Var
# optimization, whose evaluations share one tape pool. The linear schemes also
# run every entry point with `AirMassTape` (one recording reused by every
# evaluation of the optimization) and the Lin–Rood cases with its device tape.
# The cases use rough fields (random values, spikes, exact zeros and ties, sign
# changes, nonzero emission rates), so the limited PPM reverse meets every
# branch of its limiters; `limiter_code_counts` histograms the limiter codes
# the production code kernel wrote on the recorded tape and the test requires
# each branch to occur on every axis and next to the seams. Most cases pass
# their inputs directly; the `CapturedInputs` cases capture them with
# `cs_adjoint_inputs` from a production `DrivenSimulation` (rough met windows
# on the mock driver of `cs_adjoint_step_plan.jl`), so the gate also covers
# the capture: per-window flux sharing and scaling, per-window forcing.
#
# CPU kernels scatter with atomic adds, so results are bitwise reproducible
# only single-threaded: the test runs this file in a `--threads=1` subprocess.
# It writes one SHA-1 digest per (case, entry point, variant, quantity). The
# reference variant's digests are compared with
# `test/fixtures/cs_adjoint_tape_digests.toml`, recorded on the host named in
# its `[host]` table (SIMD reductions and codegen differ between CPUs and Julia
# versions, so other hosts compare variants within the run only).
#
#   julia --project=test test/helpers/cs_adjoint_tape_harness.jl [options]
#     --tier=core|full    the pairwise core set (default) or the full matrix
#     --out=FILE          write all digests and limiter-code counts (TOML)
#     --record=FILE       write the reference digests in the fixture format
#     --check=FILE        compare the reference digests with a recorded FILE
#     --reference-only    run only the reference variant, which passes no tape
#                         keyword (so it also runs on a checkout whose tape
#                         API differs, e.g. to record main), and skip the
#                         limiter-code histograms (they read this tree's tape)
#
# It prints every problem (an entry point that throws, a variant that differs
# from the reference, a limiter branch that never occurs, a digest that differs
# from `--check`) and exits with status 1 if there is one; a failing entry point
# does not stop the run. Every array the tape frees explicitly is filled with
# NaN (limiter codes: 0xff) when it is freed (`poison_freed_tape_arrays!`), so
# a free before the last read of an array (a use-after-free on a GPU) changes
# the digests. A same-host A/B of a branch against
# main: `--record=main.toml` on main, `--check=main.toml` on the branch, both
# with `SUBPROCESS_FLAGS` (the flags of the test).
module CSAdjointTapeHarness

using Logging: ConsoleLogger, Warn, with_logger
using SHA: sha1
using TOML
import KernelAbstractions
import AtmosTransport
const AT = AtmosTransport
const ADJ = AT.Adjoints
const TapeMod = AT.Tape
const Adv = AT.Operators.Advection

const HP = 3

# The small driven configuration of the step-plan parity tests (C4, 5 levels,
# Float64): its mock met driver, grid, convection forcing and operators.
module StepPlanConfig
include(joinpath(@__DIR__, "cs_adjoint_step_plan.jl"))
end
const SPC = StepPlanConfig

# ---------------------------------------------------------------------------
# Deterministic rough fields: a splitmix64 hash of the indices, the same on
# every Julia version (unlike `Random` streams).
# ---------------------------------------------------------------------------

function _mix(x::UInt64)
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end

function _u(idx::Integer...)
    h = 0x6a09e667f3bcc908
    for i in idx
        h = _mix(h ⊻ (UInt64(i) + 0x9e3779b97f4a7c15))
    end
    return Float64(h >> 11) * 0x1p-53          # [0, 1)
end

_sym(u) = 2u - 1                                  # [-1, 1)

# Air mass [kg per cell, unit cell areas]: 2-3 per layer with ±20 % noise.
function _air_mass(::Type{FT}, mesh, Nz, tag) where FT
    N = mesh.Nc + 2HP
    m = ntuple(p -> [FT((2.0 + 0.25k) * (1 + 0.2_sym(_u(tag, p, i, j, k))))
                     for i in 1:N, j in 1:N, k in 1:Nz], 6)
    Adv.fill_panel_halos!(m, mesh; dir = 0)
    return m
end

# Tracer mass with 5 % spikes, 5 % exact zeros, 5 % ties (the mixing ratio
# 2⁻⁴, exact after division by the air mass) and sign-changing noise elsewhere.
function _tracer_mass(m, mesh, tag)
    FT = eltype(m[1])
    N, Nz = size(m[1], 1), size(m[1], 3)
    q(p, i, j, k) = begin
        u = _u(tag, p, i, j, k)
        u < 0.05 ? 0.8 * _sym(_u(tag + 1, p, i, j, k)) :
        u < 0.10 ? 0.0 :
        u < 0.15 ? 0.0625 :
        0.1 * _sym(_u(tag + 2, p, i, j, k))
    end
    rm = ntuple(p -> [m[p][i, j, k] * FT(q(p, i, j, k)) for i in 1:N, j in 1:N, k in 1:Nz], 6)
    Adv.fill_panel_halos!(rm, mesh; dir = 0)
    return rm
end

# Face mass fluxes [kg per step] of step `s`. The horizontal fluxes are a
# rough circulation inside every panel: differences of a random streamfunction
# of amplitude `flow` on the panel's interior cell corners (zero on its edge
# corners), so they change sign from face to face, have no divergence, and
# vanish on the panel edges, where neighbouring panels exchange their fluxes.
# Rough noise of ±0.05 kg on every horizontal face and ±0.1 kg on the vertical
# interfaces is the only divergence: at most 0.8 kg over the six sweeps of a
# step, below the smallest air mass (1.8 kg).
function _step_fluxes(::Type{FT}, mesh, Nz, s, flow) where FT
    Nc = mesh.Nc
    N = Nc + 2HP
    ψ(p, a, b, k) = (a in (0, Nc) || b in (0, Nc)) ? 0.0 : flow * _sym(_u(101, s, p, a, b, k))
    am = ntuple(6) do p
        x = zeros(FT, N + 1, N, Nz)
        for k in 1:Nz, b in 0:Nc-1, a in 0:Nc
            x[HP+a+1, HP+b+1, k] = FT(ψ(p, a, b + 1, k) - ψ(p, a, b, k) +
                                      0.05_sym(_u(102, s, p, a, b, k)))
        end
        x
    end
    bm = ntuple(6) do p
        y = zeros(FT, N, N + 1, Nz)
        for k in 1:Nz, b in 0:Nc, a in 0:Nc-1
            y[HP+a+1, HP+b+1, k] = FT(ψ(p, a, b, k) - ψ(p, a + 1, b, k) +
                                      0.05_sym(_u(103, s, p, a, b, k)))
        end
        y
    end
    cm = ntuple(6) do p
        z = zeros(FT, N, N, Nz + 1)
        for k in 2:Nz, j in HP+1:HP+Nc, i in HP+1:HP+Nc
            z[i, j, k] = FT(0.1_sym(_u(104, s, p, i, j, k)))
        end
        z
    end
    return am, bm, cm
end

_nonzero(x) = abs(x) < 0.05 ? copysign(0.05, x) : x

# Surface rates [kg s⁻¹ per cell] of every step: nonzero, both signs.
_rates(::Type{FT}, Nc, nsteps) where FT =
    [ntuple(p -> [FT(1e-4 * _nonzero(_sym(_u(120, s, p, i, j)))) for i in 1:Nc, j in 1:Nc], 6)
     for s in 1:nsteps]

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

"""The case's inputs go to the entry points directly: rough fluxes of every step."""
struct DirectInputs end

"""
    CapturedInputs(; flux_kind = :full_window_mass_amount, vary_ps = true)

The case's inputs are `cs_adjoint_inputs` of a production `DrivenSimulation`
on the mock met driver of `cs_adjoint_step_plan.jl`, with rough met windows
(air mass, tracer, face fluxes of streamfunction amplitude `flow`, window mass
change) and nonzero rough emission rates: one shared flux set per met window,
scaled by production for `flux_kind = :full_window_mass_amount` (by
`1/(2 steps)`, different per window), and with `vary_ps` a surface pressure,
hence a diffusion layer thickness, per window. Float64, `Nc = 4`, `Nz = 5`.
"""
Base.@kwdef struct CapturedInputs
    flux_kind::Symbol = :full_window_mass_amount
    vary_ps::Bool = true
end

"""One configuration of the production step, run with every tape variant."""
Base.@kwdef struct Case
    name::String
    FT::DataType = Float64
    scheme = AT.PPMScheme(AT.MonotoneLimiter())
    conv::Symbol = :none                # :none, :tm5, :cmfmc, :cmfmc_matrix
    diffusion::Symbol = :none           # :none, :boundary, :split
    cadence::Symbol = :window           # physics cadence
    contract::Bool = true               # binary window contract (one subcycle per step)
    reset::Symbol = :none               # :none, :preserve_tracer_mass, :preserve_vmr
    flow::Float64 = 0.15                # streamfunction amplitude [kg]; 0.6 needs CFL subcycles
    steps::Vector{Int} = [2, 3, 2]      # steps per met window
    Nc::Int = 4
    Nz::Int = 5
    stride::Int = 2                     # StrideCheckpoint K of the variants
    storages::Vector{String} = ["pinned_host", "mmap"]   # host storages of the variants
    operators::Bool = false             # also run the 4D-Var observation operators (`fourdvar_operators`)
    inputs = DirectInputs()             # or `CapturedInputs` (cs_adjoint_inputs of a DrivenSimulation)
end

_limited(scheme) = scheme isa AT.PPMScheme &&
                   (scheme.limiter isa AT.MonotoneLimiter || scheme.limiter isa AT.CW84Limiter)
# The schemes whose reverse reads the air mass only (`AirMassTape`).
_linear(scheme) = scheme isa Union{AT.UpwindScheme, AT.SlopesScheme{AT.NoLimiter},
                                   AT.PPMScheme{AT.NoLimiter}}

# The cases whose inputs `cs_adjoint_inputs` captures from a production
# simulation, in both tiers. The first shares each met window's fluxes under
# CFL subcycles (counted from the shared arrays at every step), full-window
# flux storage scaled differently per window, substep CMFMC convection and a
# diffusion layer thickness per window; the second under the binary window
# contract with a window-end reset, window TM5 convection and substep flux
# storage. The 4D-Var control windows (steps 1:3 and 3:7) cross the met
# windows [2, 3, 2].
const CAPTURED_CASES = [
    Case(name = "captured_monotone_cmfmcsub_split_mass_cfl", scheme = AT.PPMScheme(AT.MonotoneLimiter()),
         conv = :cmfmc, diffusion = :split, cadence = :substep, contract = false,
         reset = :preserve_tracer_mass, flow = 2.4, storages = ["mmap"], inputs = CapturedInputs()),
    Case(name = "captured_cw84_tm5win_boundary_vmr", scheme = AT.PPMScheme(AT.CW84Limiter()),
         conv = :tm5, diffusion = :boundary, reset = :preserve_vmr, stride = 3, storages = ["pinned_host"],
         inputs = CapturedInputs(flux_kind = :substep_mass_amount)),
]

# Lin–Rood (device tape only; its horizontal records are device copies the
# drivers free explicitly): ORD 5 with upwind vertical sweeps, window TM5
# convection, boundary diffusion and a `:preserve_vmr` reset (both tiers); ORD
# 7 with FV3-profile vertical sweeps, substep CMFMC, split diffusion and a
# `:preserve_tracer_mass` reset under CFL subcycles (full tier).
const LINROOD_CORE = Case(name = "f64_linrood5_tm5win_boundary_vmr", scheme = AT.LinRoodPPMScheme(5),
                          conv = :tm5, diffusion = :boundary, reset = :preserve_vmr, storages = ["device"])
const LINROOD_FULL = Case(name = "f64_linrood7fv3_cmfmcsub_split_mass_cfl",
                          scheme = AT.LinRoodPPMScheme(7; vertical = AT.PPMScheme(; vertical = Adv.FV3ScalarProfile())),
                          conv = :cmfmc, diffusion = :split, cadence = :substep, contract = false,
                          reset = :preserve_tracer_mass, flow = 0.6, stride = 3, storages = ["device"])

# The pairwise core set: each scheme, both float types, every convection
# operator and diffusion coupling, both cadences, every reset mode, CFL
# subcycles, and both limiters with every physics family. Each case runs one
# host storage and one stride (compiling a recorder per storage is most of a
# case's time); across the set every limiter meets both host storages and
# both strides, and the mmap tape (sized by counting on the replay) meets CFL
# subcycles and substep CMFMC convection. The observation operators run on one
# Float64 case with CFL subcycles and the mmap tape and one Float32 case with
# the pinned-host tape. The captured cases (both tiers) close the set.
const CORE_CASES = [
    Case(name = "f64_cw84_tm5win_boundary_vmr", scheme = AT.PPMScheme(AT.CW84Limiter()),
         conv = :tm5, diffusion = :boundary, reset = :preserve_vmr, storages = ["mmap"]),
    Case(name = "f64_monotone_cmfmcsub_split_mass_cfl", scheme = AT.PPMScheme(AT.MonotoneLimiter()),
         conv = :cmfmc, diffusion = :split, cadence = :substep, contract = false,
         reset = :preserve_tracer_mass, flow = 0.6, stride = 3, storages = ["mmap"], operators = true),
    Case(name = "f32_cw84_matrixwin_split_none", FT = Float32, scheme = AT.PPMScheme(AT.CW84Limiter()),
         conv = :cmfmc_matrix, diffusion = :split, storages = ["pinned_host"], operators = true),
    Case(name = "f32_monotone_tm5sub_boundary_vmr", FT = Float32,
         scheme = AT.PPMScheme(AT.MonotoneLimiter()), conv = :tm5, diffusion = :boundary,
         cadence = :substep, reset = :preserve_vmr, stride = 3, storages = ["pinned_host"]),
    Case(name = "f64_upwind_cmfmcwin_split_vmr", scheme = AT.UpwindScheme(),
         conv = :cmfmc, diffusion = :split, reset = :preserve_vmr, stride = 3, storages = ["mmap"]),
    Case(name = "f64_slopes_tm5sub_boundary_none_cfl", scheme = AT.SlopesScheme(AT.NoLimiter()),
         conv = :tm5, diffusion = :boundary, cadence = :substep, contract = false, flow = 0.6,
         storages = ["pinned_host"]),
    Case(name = "f32_ppm_none_boundary_mass", FT = Float32, scheme = AT.PPMScheme(AT.NoLimiter()),
         diffusion = :boundary, reset = :preserve_tracer_mass, storages = ["mmap"]),
    LINROOD_CORE,
    CAPTURED_CASES...,
]

# The full matrix (opt-in): every scheme × float type × physics family.
function full_cases()
    schemes = (("upwind", AT.UpwindScheme()), ("slopes", AT.SlopesScheme(AT.NoLimiter())),
               ("ppm", AT.PPMScheme(AT.NoLimiter())), ("monotone", AT.PPMScheme(AT.MonotoneLimiter())),
               ("cw84", AT.PPMScheme(AT.CW84Limiter())))
    physics = ((conv = :none, diffusion = :none, cadence = :window, contract = true, reset = :none, flow = 0.15),
               (conv = :tm5, diffusion = :boundary, cadence = :window, contract = true, reset = :preserve_vmr, flow = 0.15),
               (conv = :cmfmc, diffusion = :split, cadence = :substep, contract = true, reset = :preserve_tracer_mass, flow = 0.15),
               (conv = :cmfmc_matrix, diffusion = :boundary, cadence = :window, contract = true, reset = :preserve_tracer_mass, flow = 0.15),
               (conv = :tm5, diffusion = :split, cadence = :substep, contract = false, reset = :none, flow = 0.6))
    cases = Case[]
    for FT in (Float64, Float32), (sname, scheme) in schemes, ph in physics
        name = lowercase("$(FT)_$(sname)_$(ph.conv)$(ph.cadence)_$(ph.diffusion)_$(ph.reset)" *
                         (ph.contract ? "" : "_cfl"))
        push!(cases, Case(; name, FT, scheme, ph...))
    end
    return append!(cases, [LINROOD_CORE, LINROOD_FULL], CAPTURED_CASES)
end

# The inputs of a case: the step's fluxes, schedule, reset targets, surface
# rates and the keywords of its operators (built fresh for every call: the
# workspaces carry scratch).
problem(c::Case) = problem(c, c.inputs)

function problem(c::Case, ::DirectInputs)
    FT = c.FT
    mesh = AT.CubedSphereMesh(Nc = c.Nc, Hp = HP, FT = FT)
    nsteps = sum(c.steps)
    m0 = _air_mass(FT, mesh, c.Nz, 1)
    rm0 = _tracer_mass(m0, mesh, 10)
    fluxes = [_step_fluxes(FT, mesh, c.Nz, s, c.flow) for s in 1:nsteps]
    am = [f[1] for f in fluxes]; bm = [f[2] for f in fluxes]; cm = [f[3] for f in fluxes]
    nw = length(c.steps)
    targets(tag) = [_air_mass(FT, mesh, c.Nz, tag + w) for w in 1:nw]
    start_targets = c.reset === :none ? nothing : targets(20)
    end_targets = c.reset === :none || !c.contract ? nothing : targets(40)
    schedule = AT.CSStepSchedule(; physics_cadence = c.cadence, binary_window_contract = c.contract,
                                 air_mass_reset = c.reset, steps_per_window = c.steps,
                                 window_dt = 1800.0, window_start_air_mass = start_targets,
                                 window_end_air_mass = end_targets)
    return (; case = c, mesh, m0, rm0, am, bm, cm, schedule, nsteps,
            rates = _rates(FT, c.Nc, nsteps))
end

physics_kwargs(P) = physics_kwargs(P, P.case.inputs)

function physics_kwargs(P, ::DirectInputs)
    c = P.case
    FT, Nc, Nz = c.FT, c.Nc, c.Nz
    kw = (;)
    if c.diffusion !== :none
        coupling = c.diffusion === :split ? AT.SplitSurfaceFluxCoupling() :
                                            AT.DiffusiveSurfaceFluxBoundary()
        kz = ntuple(p -> AT.PreComputedKzField(FT[30 * (1 + 0.5_sym(_u(121, p, i, j, k))) / k
                                                  for i in 1:Nc, j in 1:Nc, k in 1:Nz]), 6)
        op = AT.ImplicitVerticalDiffusion(; kz_field = AT.CubedSphereField(kz),
                                          surface_flux_coupling = coupling)
        ws = AT.DiffusionWorkspace(P.m0, HP, 0)
        for p in 1:6, k in 1:Nz
            ws.layer_thickness[p][:, :, k] .= FT(40 + 15k + 3p)
        end
        kw = (; kw..., diffusion_op = op, diffusion_workspace = ws)
    end
    c.conv === :none && return kw
    metrics = ntuple(_ -> ones(FT, Nc, Nc), 6)
    rough(tag, p, lo, hi, v, n) = (e = zeros(FT, Nc, Nc, n);
        for k in lo:min(hi, n), j in 1:Nc, i in 1:Nc
            e[i, j, k] = FT(v * (0.5 + _u(tag, p, i, j, k)))
        end; e)
    if c.conv === :tm5
        forcing = AT.ConvectionForcing(nothing, nothing,
            (; entu = ntuple(p -> rough(106, p, 2, 4, 1e-3, Nz), 6),
               detu = ntuple(p -> rough(107, p, 2, 4, 7e-4, Nz), 6),
               entd = ntuple(p -> rough(108, p, 3, 4, 3e-4, Nz), 6),
               detd = ntuple(p -> rough(109, p, 3, 4, 1.5e-4, Nz), 6)))
        return (; kw..., convection_op = AT.TM5Convection(), convection_forcing = forcing,
                convection_workspace = AT.TM5Workspace(P.m0; tile_columns = Nc * Nc,
                                                       cell_metrics = metrics))
    end
    forcing = AT.ConvectionForcing(ntuple(p -> rough(110, p, 2, 4, 2.5e-4, Nz + 1), 6),
                                   ntuple(p -> rough(111, p, 2, 3, 1e-4, Nz), 6), nothing)
    if c.conv === :cmfmc
        return (; kw..., convection_op = AT.CMFMCConvection(), convection_forcing = forcing,
                convection_workspace = AT.CMFMCWorkspace(P.m0; cell_metrics = metrics))
    end
    c.conv === :cmfmc_matrix || error("unknown convection $(c.conv)")
    return (; kw..., convection_op = AT.CMFMCMatrixConvection(), convection_forcing = forcing,
            convection_workspace = AT.Operators.Convection.CMFMCMatrixWorkspace(P.m0;
                cell_metrics = metrics, halo_width = HP))
end

# Met window `w` of a captured case: rough air mass, face fluxes and window
# mass change on the step-plan grid (mass in units of its `S`, about
# 1500 kg m⁻² per layer), the step plan's convection forcing of window `w`
# (its CMFMC 40 times stronger, so the CMFMC CFL needs 2-3 sub-steps) and a
# surface pressure that changes per window with `vary_ps`.
function _captured_window(c::Case, inputs::CapturedInputs, w::Int)
    mesh, Nc, Nz, S = SPC.MESH, SPC.NC, SPC.NZ, SPC.S
    N = Nc + 2HP
    m = map(a -> S .* a, _air_mass(Float64, mesh, Nz, 200 + w))
    am, bm, cm = map(f -> map(a -> S .* a, f), _step_fluxes(Float64, mesh, Nz, 200 + w, c.flow))
    dm = ntuple(p -> [S * 0.002 * _sym(_u(210 + w, p, i, j, k)) for i in 1:N, j in 1:N, k in 1:Nz], 6)
    cmfmc(f) = AT.ConvectionForcing(map(a -> 40 .* a, f.cmfmc), map(a -> 40 .* a, f.dtrain), nothing)
    forcing = c.conv === :tm5 ? SPC._tm5_forcing(w) :
              c.conv in (:cmfmc, :cmfmc_matrix) ? cmfmc(SPC._cmfmc_forcing(w)) : nothing
    ps = inputs.vary_ps ? 95_000.0 - 3000.0w : 95_000.0
    return AT.TransportWindow(m, ntuple(_ -> fill(ps, Nc, Nc), 6),
                              AT.CubedSphereFaceFluxState{AT.DryBasis}(am, bm, cm);
                              deltas = AT.MetDrivers.CubedSphereFluxDeltas(dm), convection = forcing,
                              dkg = ntuple(_ -> fill(S * 1e-5, Nc, Nc, Nz), 6))
end

# The production simulation of a captured case (not yet run): the case's
# scheme and operators on the mock driver, a rough tracer and nonzero rough
# emission rates of both signs.
function _captured_simulation(c::Case, inputs::CapturedInputs)
    c.FT === Float64 && c.Nc == SPC.NC && c.Nz == SPC.NZ ||
        error("$(c.name): captured cases run on the step-plan grid (Float64, Nc = $(SPC.NC), Nz = $(SPC.NZ))")
    windows = [_captured_window(c, inputs, w) for w in eachindex(c.steps)]
    driver = SPC._PlanDriver{Float64, typeof(SPC.GRID), eltype(windows)}(
        SPC.GRID, windows, SPC.WINDOW_DT, c.steps, c.contract, inputs.flux_kind)
    air = map(copy, windows[1].air_mass)
    state = AT.CubedSphereState(AT.DryBasis, SPC.MESH, air; CO2 = _tracer_mass(air, SPC.MESH, 10))
    model = AT.TransportModel(state, AT.allocate_face_fluxes(SPC.MESH, SPC.NZ; basis = AT.DryBasis),
                              SPC.GRID, c.scheme; diffusion = SPC._diffusion_op(c.diffusion),
                              convection = SPC._convection_op(c.conv))
    rates = ntuple(p -> [SPC.S * 1e-6 * _nonzero(_sym(_u(122, p, i, j))) for i in 1:SPC.NC, j in 1:SPC.NC], 6)
    return AT.DrivenSimulation(model, driver; surface_sources = (AT.SurfaceFluxSource(:CO2, rates),),
                               physics_cadence = c.cadence, air_mass_reset_mode = c.reset)
end

_capture(c::Case, inputs::CapturedInputs) =
    AT.cs_adjoint_inputs(_captured_simulation(c, inputs); tracer = :CO2)

function problem(c::Case, inputs::CapturedInputs)
    x = _capture(c, inputs)
    return (; case = c, mesh = x.mesh, m0 = x.panels_m0, rm0 = x.panels_rm0, am = x.panels_am_steps,
            bm = x.panels_bm_steps, cm = x.panels_cm_steps, schedule = x.kwargs.schedule,
            nsteps = length(x.panels_am_steps), rates = x.kwargs.emission_rates)
end

# The operator keywords of a fresh capture (its workspaces carry scratch).
physics_kwargs(P, inputs::CapturedInputs) =
    Base.structdiff(_capture(P.case, inputs).kwargs,
                    NamedTuple{(:scheme, :schedule, :surface_flux, :emission_rates)})

# ---------------------------------------------------------------------------
# Variants `checkpoint/storage` of the `AdjointTape`: the reference is the
# default tape (`StateTape`, device storage, FullCheckpoint), passed as no
# keyword. `strideK` is the case's stride (2 and 3 both cross the met windows
# [2, 3, 2]) and `S` each of its host storages. The Jacobian's two objectives
# share each stride window's recording; the optimization's evaluations share
# one tape pool (rewound, and an mmap file counted and reserved once).
# `airmass/S` is an `AirMassTape` (FullCheckpoint) for the linear schemes: one
# recording, reused by the optimization's later evaluations after its
# identity check.
# ---------------------------------------------------------------------------

const REFERENCE = "full/device"

const VARIANT_TEMPLATES = (
    footprint = [REFERENCE, "strideK/device", "full/S", "strideK/S"],
    seeded = [REFERENCE, "strideK/S"],
    jacobian = [REFERENCE, "strideK/device", "strideK/S"],
    fourdvar = [REFERENCE, "full/S", "strideK/device", "strideK/S"],
    fourdvar_operators = [REFERENCE, "full/S", "strideK/device", "strideK/S"],
    optimize = [REFERENCE, "full/S", "strideK/device", "strideK/S"],
)

# The entry points a case runs.
entries(c::Case) = c.operators ? keys(VARIANT_TEMPLATES) :
                   filter(!=(:fourdvar_operators), keys(VARIANT_TEMPLATES))

function variants(c::Case, entry::Symbol)
    vs = [replace(t, "K" => string(c.stride), "S" => s) for t in VARIANT_TEMPLATES[entry] for s in c.storages]
    _linear(c.scheme) && append!(vs, ["airmass/device"; ["airmass/$s" for s in c.storages]])
    return unique!(vs)
end

const STORAGES = Dict("device" => () -> AT.DeviceCSTapeStorage(),
                      "pinned_host" => () -> AT.PinnedHostCSTapeStorage())

function _variant_kwargs(variant::AbstractString, scratch::AbstractString)
    variant == REFERENCE && return (;)
    schedule, storage = split(variant, '/')
    content = schedule == "airmass" ? AT.AirMassTape() : AT.StateTape()
    checkpoint = schedule in ("full", "airmass") ? AT.FullCheckpoint() :
                 AT.StrideCheckpoint(parse(Int, chopprefix(schedule, "stride")))
    storage = storage == "mmap" ? AT.MmapCSTapeStorage(mktempdir(scratch)) : STORAGES[storage]()
    return (; tape = AT.AdjointTape(; content, storage, checkpoint))
end

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

_objectives(P) = [AT.CSColumnMeanObjective(1, 2, 3), AT.CSLayerMeanObjective(4, 3, 2, P.case.Nz)]

function _seed(P)
    N, Nz = P.case.Nc + 2HP, P.case.Nz
    return ntuple(p -> [P.case.FT(1e-3 * _sym(_u(112, p, i, j, k))) for i in 1:N, j in 1:N, k in 1:Nz], 6)
end

function _controls(P)
    FT, Nc, n = P.case.FT, P.case.Nc, P.nsteps
    value(tag) = ntuple(p -> [FT(1e-4 * _nonzero(_sym(_u(tag, p, i, j)))) for i in 1:Nc, j in 1:Nc], 6)
    late = 3:n
    return [AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:early, 1:3; normalize = true), value(113)),
            AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:late, late; weights = [1.0 / s for s in late]),
                                    value(114))]
end

# The last assimilated observation is before the last observation step (a
# co-sample at the final step), so the taped run stops early.
function _observations(P)
    n, Nz = P.nsteps, P.case.Nz
    return [AT.CSObservation(n - 4, AT.CSColumnMeanObjective(2, 3, 2), 0.02, 0.01),
            AT.CSObservation(n - 2, AT.CSLayerMeanObjective(5, 1, 4, Nz), -0.01, 0.02),
            AT.CSObservation(n - 2, AT.CSColumnMeanObjective(6, 4, 4), 0.03, 0.01),
            AT.CSObservation(n, AT.CSColumnMeanObjective(3, 2, 2), 0.0, 0.01; assimilate = false)]
end

# ---------------------------------------------------------------------------
# 4D-Var observation operators (`fourdvar_operators`). The interface pressures
# of a column are those of its dry air mass (`DryInterfaces`) on a cell of area
# `COLUMN_AREA`, about 1000 hPa at the surface for the case's air mass. The
# observations sample states through step `n - 1` and use every operator once:
# a time window, placed points by height and by pressure, averaging-kernel
# columns on retrieval layers (rebin) and on levels (hat projection, in a time
# window), a cell average of two columns, and at step `n - 1` the last column
# with `assimilate = true`, whose surface pressure the screen rejects
# (`SkipUnplaced`). The tape therefore records through step `n - 1` and the
# reverse pass starts at step `n - 2`; a co-sample at step `n` runs the
# forward to the end.
# ---------------------------------------------------------------------------

const COLUMN_P_TOP = 1000.0
const COLUMN_GRAVITY = 9.80665
const COLUMN_AREA = 1.36e-3                       # m²: g · 13.75 kg / A ≈ 990 hPa
const COLUMN_BASIS = AT.Observations.DryInterfaces()

_column_met(P) = AT.CSObservationColumnMet(; p_top = COLUMN_P_TOP, gravity = COLUMN_GRAVITY,
                                           temperature = 280.0)

# A retrieval column of `n` layers (rebin) or levels (hat) from `ps` up.
function _retrieval(ps; n = 4, hat = false, tag = 130)
    O = AT.Observations
    pressures = collect(range(ps, 0.0; length = hat ? n : n + 1))
    weights = hat ? fill(1.0 / n, n) : -diff(pressures) ./ ps
    kernel = [0.7 + 0.5 * _u(tag, k) for k in 1:n]
    prior = [0.02 + 0.06 * _u(tag + 1, k) for k in 1:n]
    return O.RetrievalColumn(; pressure_pa = pressures, pressure_weight = weights, averaging_kernel = kernel,
                             prior_profile = prior, mapping = hat ? O.HatLevelProjection() : O.LayerRebin())
end

function _column(panel, i, j, ps; limit = Inf, kwargs...)
    O = AT.Observations
    policy = isfinite(limit) ? O.SkipUnplaced() : O.ErrorOnUnplaced()
    return AT.CSRetrievalColumnObjective(panel, i, j, COLUMN_AREA, _retrieval(ps; kwargs...), COLUMN_BASIS;
                                         max_surface_pressure_mismatch_pa = limit, mismatch_policy = policy)
end

function _placed(panel, i, j; height = NaN, pressure = NaN)
    codes = AT.Observations.VERTICAL_PLACEMENT_CODES
    method = isnan(height) ? codes.pressure : codes.record_height
    placement = AT.Observations.VerticalPlacement(method, height, pressure, NaN)
    return AT.CSPlacedPointObjective(panel, i, j, COLUMN_AREA, placement, COLUMN_BASIS)
end

function _operator_observations(P)
    n, Nz, FT = P.nsteps, P.case.Nz, P.case.FT
    obs(s, objective, value, sigma; kw...) = AT.CSObservation(s, objective, FT(value), FT(sigma); kw...)
    window(steps, weights) = AT.CSTimeSampling(steps, weights)
    average = AT.CSCellAverageObjective(6, 1, 1, [_column(6, 1, 1, 98000.0; tag = 140),
                                                  _column(6, 1, 1, 101000.0; hat = true, n = 5, tag = 150)],
                                        [0.4, 0.6], [61, 62])
    return [obs(window([n - 5, n - 4], [0.25, 0.75]), AT.CSLayerMeanObjective(1, 2, 3, 2), 0.01, 0.02),
            obs(n - 4, _placed(2, 3, 1; height = 800.0), 0.02, 0.01),
            obs(n - 3, _placed(4, 1, 4; pressure = 60000.0), -0.01, 0.02),
            obs(n - 4, _column(3, 2, 2, 99000.0), 0.03, 0.01),
            obs(window([n - 3, n - 2], [0.5, 0.5]), _column(5, 4, 3, 100000.0; hat = true, n = 5, tag = 160),
                0.02, 0.01),
            obs(n - 2, average, 0.01, 0.01),
            obs(n - 1, _column(2, 2, 2, 200000.0; limit = 1000.0, tag = 170), 0.02, 0.01),
            obs(n, AT.CSColumnMeanObjective(3, 2, 2), 0.0, 0.01; assimilate = false)]
end

# The controls of `_controls` with background terms (scalar and per-cell
# sigma), and the same windows in the preconditioned space.
function _background_controls(P)
    FT, Nc = P.case.FT, P.case.Nc
    background(tag) = ntuple(p -> [FT(5e-5 * _sym(_u(tag, p, i, j))) for i in 1:Nc, j in 1:Nc], 6)
    sigma = ntuple(p -> [FT(1e-4 * (1 + _u(117, p, i, j))) for i in 1:Nc, j in 1:Nc], 6)
    early, late = _controls(P)
    return [AT.CSSurfaceFluxControl(early.window, early.value; background = background(115), sigma = FT(2e-4)),
            AT.CSSurfaceFluxControl(late.window, late.value; background = background(116), sigma)]
end

function _preconditioner(P)
    FT, Nc = P.case.FT, P.case.Nc
    sigma = ntuple(p -> [FT(1e-4 * (1 + _u(118, p, i, j))) for i in 1:Nc, j in 1:Nc], 6)
    background = ntuple(p -> [FT(2e-5 * _sym(_u(119, p, i, j))) for i in 1:Nc, j in 1:Nc], 6)
    return AT.CSSurfaceFluxPreconditioner(AT.IsotropicGaussianCSCovariance(sigma, 1.5), background,
                                          AT.LinearOptimType())
end

function _chi_controls(P)
    FT, Nc = P.case.FT, P.case.Nc
    chi(tag) = ntuple(p -> [FT(_sym(_u(tag, p, i, j))) for i in 1:Nc, j in 1:Nc], 6)
    return [AT.CSSurfaceFluxControl(c.window, chi(121 + k)) for (k, c) in enumerate(_controls(P))]
end

_common(P) = (; scheme = P.case.scheme, schedule = P.schedule, physics_kwargs(P)...)

_norm(tuples) = sqrt(sum(sum(abs2, a) for t in tuples for a in t))

function run_entry(P, entry::Symbol, variant::AbstractString, scratch::AbstractString)
    kw = (; _common(P)..., _variant_kwargs(variant, scratch)...)
    if entry === :footprint
        r = AT.cs_surface_emission_footprint(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh,
                                             first(_objectives(P)); base_emission_rates = P.rates, kw...)
        return ("footprints" => r.footprints, "initial_adjoint" => r.initial_adjoint)
    elseif entry === :seeded
        r = AT.cs_surface_emission_footprint_from_seed(_seed(P), P.m0, P.am, P.bm, P.cm, P.mesh;
                                                       base_panels_rm0 = P.rm0,
                                                       base_emission_rates = P.rates, kw...)
        return ("footprints" => r.footprints, "initial_adjoint" => r.initial_adjoint)
    elseif entry === :jacobian
        windows = [AT.CSSurfaceFluxWindow(:all, 1:P.nsteps), AT.CSSurfaceFluxWindow(:head, 1:3; normalize = true)]
        r = AT.cs_surface_flux_jacobian(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh, _objectives(P), windows;
                                        base_emission_rates = P.rates, kw...)
        return ("footprints" => [x.footprints for x in r.per_step_results],
                "initial_adjoint" => [x.initial_adjoint for x in r.per_step_results],
                "windows" => r.footprints)
    elseif entry === :fourdvar
        r = AT.cs_surface_flux_4dvar(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh, _observations(P),
                                     _controls(P); kw...)
        return ("cost" => [r.cost], "simulated" => r.simulated, "gradients" => r.gradients)
    elseif entry === :fourdvar_operators
        observations, column_met = _operator_observations(P), _column_met(P)
        r = AT.cs_surface_flux_4dvar(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh, observations,
                                     _background_controls(P); column_met, kw...)
        # The screen must reject the column at step n - 1 (the reverse pass
        # then starts before the end of the tape) and nothing else.
        [s.assimilated for s in r.stencils] == [trues(6); false; false] ||
            error("$(P.case.name): the surface-pressure screen of `_operator_observations` changed")
        kw = (; _common(P)..., _variant_kwargs(variant, scratch)...)
        chi = AT.cs_surface_flux_4dvar(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh, observations, _chi_controls(P);
                                       column_met, preconditioner = _preconditioner(P), kw...)
        return ("cost" => [r.cost], "simulated" => r.simulated, "gradients" => r.gradients,
                "chi_cost" => [chi.cost], "chi_gradients" => chi.gradients)
    elseif entry === :optimize
        # Fixed-step gradient descent: two iterations, three evaluations into
        # one tape pool. The step moves the controls by about a tenth of their
        # size along the default tape's gradient (the same for every variant).
        probe = AT.cs_surface_flux_4dvar(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh, _observations(P),
                                         _controls(P); _common(P)...)
        step = _norm([c.value for c in _controls(P)]) / (10 * _norm(probe.gradients))
        optimizer = AT.CSGradientDescent(; iterations = 2, initial_step = step, min_step = step,
                                         line_search = false)
        r = AT.cs_surface_flux_4dvar_optimize(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh, _observations(P),
                                              _controls(P); optimizer, kw...)
        length(r.cost_history) == 3 || error("the optimization took $(r.iterations) of 2 steps")
        return ("cost_history" => r.cost_history, "controls" => [c.value for c in r.controls],
                "gradients" => r.last.gradients)
    end
    error("unknown entry point $entry")
end

# ---------------------------------------------------------------------------
# Digests of the raw bits
# ---------------------------------------------------------------------------

_bytes!(io, x::AbstractArray{<:AbstractFloat}) = write(io, reinterpret(UInt8, vec(collect(x))))
_bytes!(io, x::Union{Tuple, AbstractArray}) = foreach(y -> _bytes!(io, y), x)

function digest(x)
    io = IOBuffer()
    _bytes!(io, x)
    return bytes2hex(sha1(take!(io)))
end

# ---------------------------------------------------------------------------
# Limiter branches of the recorded tape
# ---------------------------------------------------------------------------

# Branch counts of the limiter codes a limited sweep record holds (written by
# the production code kernel, `PPMLimiterCodes`), at the cells its reverse
# reads, per axis: x and y split into the cells next to a panel edge (whose
# stencils read the halo, "seam") and the others; z over the interior columns.
# `profile` counts the profile branches (bits 0-1: 0 extremum, 1 left edge
# reset, 2 right edge reset, 3 unchanged) of every limited scheme, `slope` the
# minmod branches of the CW84 edge slopes (bits 2-3: 0 zero, 1 centred,
# 2 forward, 3 backward).
function limiter_code_counts(ops, Nc, Nz)
    counts = Dict{String, Vector{Int}}()
    bump!(key, code) = (get!(() -> zeros(Int, 4), counts, key)[code + 1] += 1)
    for op in ops
        op isa TapeMod._CSSweepRecord && _limited(op.scheme) || continue
        codes = TapeMod._tape_panels(op.panels_tracer, TapeMod.LimiterCodeRead())
        cw84 = op.scheme.limiter isa AT.CW84Limiter
        count!(key, code) = (bump!("profile_$key", code & 0x03);
                             cw84 && bump!("slope_$key", (code >> 2) & 0x03))
        for p in 1:6
            c = codes[p]
            if op.direction === :z
                for k in 1:Nz, j in HP+1:HP+Nc, i in HP+1:HP+Nc
                    count!("z", c[i, j, k])
                end
                continue
            end
            axis = String(op.direction)
            for k in 1:Nz, a in HP:HP+Nc+1, b in HP+1:HP+Nc
                region = a in (HP, HP + 1, HP + Nc, HP + Nc + 1) ? "seam" : "interior"
                count!("$(axis)_$(region)", op.direction === :x ? c[a, b, k] : c[b, a, k])
            end
        end
    end
    return counts
end

# The limiter codes of a case's device tape.
function case_limiter_codes(P)
    replay, _ = ADJ.Footprint._cs_replay_for_call(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh;
                                                  _common(P)..., base_emission_rates = P.rates)
    pool = TapeMod._open_tape_pool(TapeMod.DeviceCSTapeStorage(), 0)
    ops = ADJ.Footprint._record_cs_tape!(replay, 1:P.nsteps, TapeMod.StateTape(), pool)
    return limiter_code_counts(ops, P.case.Nc, P.case.Nz)
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

host_fingerprint() = Dict{String, Any}(
    "cpu" => Sys.CPU_NAME, "arch" => string(Sys.ARCH), "julia" => string(VERSION),
    "kernelabstractions" => string(pkgversion(KernelAbstractions)))

"""
    run_cases(cases; scratch, reference_only = false) -> Dict

Digests of every (case, entry point, variant, quantity) and the limiter-code
counts of the limited-PPM cases, as written by `--out`. An entry point that
throws is reported in `failures` (one line per case, entry point and variant)
and the run continues; it has no digests.
"""
function run_cases(cases; scratch::AbstractString, reference_only::Bool = false)
    out = Dict{String, Any}("host" => host_fingerprint(), "threads" => Threads.nthreads())
    results = Dict{String, Any}()
    failures = String[]
    for c in cases
        t0 = time()
        P = problem(c)
        digests = Dict{String, Any}()
        for entry in entries(c)
            vs = reference_only ? [REFERENCE] : variants(c, entry)
            d = digests[String(entry)] = Dict{String, Any}()
            for v in vs
                try
                    d[v] = Dict{String, Any}(name => digest(x) for (name, x) in run_entry(P, entry, v, scratch))
                catch err
                    push!(failures, "$(c.name) $entry $v: " * first(sprint(showerror, err), 400))
                end
            end
        end
        entry = Dict{String, Any}("digests" => digests)
        if _limited(c.scheme) && !reference_only
            entry["limiter"] = c.scheme.limiter isa AT.CW84Limiter ? "cw84" : "monotone"
            entry["limiter_codes"] = case_limiter_codes(P)
        end
        results[c.name] = entry
        println(stderr, "  ", rpad(c.name, 44), round(time() - t0; digits = 1), " s")
    end
    out["cases"] = results
    out["failures"] = failures
    out["poisoned_arrays"] = POISONED[]
    return out
end

# The fixture format: the host and the reference variant's digests.
function reference_digests(out)
    cases = Dict{String, Any}()
    for (name, entry) in out["cases"]
        cases[name] = Dict{String, Any}(e => d[REFERENCE] for (e, d) in entry["digests"])
    end
    return Dict{String, Any}("host" => out["host"], "cases" => cases)
end

# Problems of a run (the `--out` table), one line each.

"""Variants whose digests differ from the reference variant of their entry point."""
function variant_mismatches(out)
    bad = String[]
    for (name, entry) in out["cases"], (e, d) in entry["digests"], (v, q) in d
        haskey(d, REFERENCE) || continue          # a failed reference is in `entry_failures`
        for (quantity, h) in q
            h == d[REFERENCE][quantity] || push!(bad, "$name $e $v: $quantity differs from $REFERENCE")
        end
    end
    return sort!(bad)
end

"""Entry points that threw: one line per case, entry point and variant."""
entry_failures(out) = sort!(String[f for f in get(out, "failures", String[])])

# The branch histograms of a limited case: profile branches on every axis and
# region, and the CW84 minmod slopes.
function expected_code_keys(limiter::AbstractString)
    regions = ["x_interior", "x_seam", "y_interior", "y_seam", "z"]
    kinds = limiter == "cw84" ? ["profile", "slope"] : ["profile"]
    return ["$(kind)_$(region)" for kind in kinds for region in regions]
end

"""Limiter branches that never occurred (or were never counted) on a limited-PPM case's tape."""
function missing_limiter_codes(out)
    bad = String[]
    for (name, entry) in out["cases"]
        haskey(entry, "limiter") || continue
        codes = get(entry, "limiter_codes", Dict())
        for key in expected_code_keys(entry["limiter"])
            counts = get(codes, key, zeros(Int, 4))
            zero_codes = findall(==(0), counts) .- 1
            isempty(zero_codes) || push!(bad, "$name $key: codes $zero_codes never occur ($counts)")
        end
    end
    return sort!(bad)
end

same_host(a, b) = a["host"] == b["host"]

"""Reference digests of the run that differ from (or are missing in) `reference`."""
function reference_mismatches(out, reference)
    bad = String[]
    run = reference_digests(out)["cases"]
    for (name, entries) in reference["cases"]
        haskey(run, name) || continue
        for (e, quantities) in entries, (quantity, h) in quantities
            got = get(get(run[name], e, Dict()), quantity, nothing)
            got == h || push!(bad, "$name $e $quantity: $(something(got, "missing")) != reference $h")
        end
    end
    return sort!(bad)
end

# ---------------------------------------------------------------------------
# Freed tape memory is poisoned
# ---------------------------------------------------------------------------

# On a GPU the tape frees its arrays explicitly (`_free_tape_array!`:
# `unsafe_free!` of device slabs, read caches, stride checkpoints, the
# recorder's limiter-code scratch and Lin–Rood records) and the memory is
# handed out again, so a free placed before the last read of an array is a
# use-after-free. On the CPU the free is a no-op, so the bitwise gate fills
# every freed float array with NaN, and every freed limiter-code array with
# 0xff, at that moment instead: a premature free then changes the digests. The method is added in the
# harness's own subprocess only (`main`), never in the test process, and is a
# no-op on trees without the hook.
const POISONED = Ref(0)
function poison_freed_tape_arrays!()
    isdefined(TapeMod, :_free_tape_array!) || return false
    @eval TapeMod._free_tape_array!(a::Array{<:AbstractFloat}) = (fill!(a, NaN); $POISONED[] += 1; nothing)
    @eval TapeMod._free_tape_array!(a::Array{UInt8}) = (fill!(a, 0xff); $POISONED[] += 1; nothing)
    return true
end

# The flags of every bitwise run: one thread, and the default bounds checks
# and optimization level whatever the parent process uses (they can change
# vectorized reduction orders).
const SUBPROCESS_FLAGS =
    `--startup-file=no --threads=1 --check-bounds=auto -O2 --inline=yes --code-coverage=none`

"""
    run_subprocess(tier) -> Dict

Run this file with `--tier=tier` in a fresh Julia process with
`SUBPROCESS_FLAGS` and the active project, and return its `--out` table.
Problems are left to the caller.
"""
function run_subprocess(tier::AbstractString)
    return mktempdir() do dir
        out = joinpath(dir, "out.toml")
        cmd = `$(Base.julia_cmd()) $SUBPROCESS_FLAGS --project=$(Base.active_project())
               $(@__FILE__) --tier=$tier --out=$out`
        run(pipeline(ignorestatus(cmd); stdout = stderr, stderr = stderr))
        isfile(out) || error("the adjoint tape harness wrote no results; see its output above")
        TOML.parsefile(out)
    end
end

function main(args = ARGS)
    tier = "core"; out_file = nothing; record_file = nothing; check_file = nothing
    reference_only = false
    for a in args
        if startswith(a, "--tier=")
            tier = a[8:end]
        elseif startswith(a, "--out=")
            out_file = a[7:end]
        elseif startswith(a, "--record=")
            record_file = a[10:end]
        elseif startswith(a, "--check=")
            check_file = a[9:end]
        elseif a == "--reference-only"
            reference_only = true
        else
            error("unknown option $a")
        end
    end
    tier in ("core", "full") || error("--tier must be core or full; got $tier")
    Threads.nthreads() == 1 ||
        @warn "run with --threads=1: CPU atomics are not bitwise reproducible across threads"
    cases = tier == "core" ? CORE_CASES : full_cases()
    poison_freed_tape_arrays!()
    # Production logs each growth of the CFL subcycle count; keep warnings.
    # `invokelatest`: the run sees the poisoning method defined just above.
    out = with_logger(ConsoleLogger(stderr, Warn)) do
        mktempdir(scratch -> Base.invokelatest(run_cases, cases; scratch, reference_only))
    end
    out_file === nothing || open(io -> TOML.print(io, out; sorted = true), out_file, "w")
    record_file === nothing || open(record_file, "w") do io
        println(io, "# Reference digests of test/helpers/cs_adjoint_tape_harness.jl (--record),")
        println(io, "# recorded from main on the [host] below; see test/README.md.")
        TOML.print(io, reference_digests(out); sorted = true)
    end
    problems = vcat(entry_failures(out), variant_mismatches(out), missing_limiter_codes(out))
    println(stderr, "freed tape arrays poisoned: ", out["poisoned_arrays"])
    if check_file !== nothing
        reference = TOML.parsefile(check_file)
        same_host(out, reference) ||
            @warn "the reference was recorded on another host or Julia version" reference["host"] out["host"]
        append!(problems, reference_mismatches(out, reference))
    end
    foreach(p -> println(stderr, "FAIL ", p), problems)
    println(stderr, isempty(problems) ? "all digests agree" : "$(length(problems)) problems")
    return isempty(problems)
end

end # module CSAdjointTapeHarness

if abspath(PROGRAM_FILE) == @__FILE__
    CSAdjointTapeHarness.main() || exit(1)
end
