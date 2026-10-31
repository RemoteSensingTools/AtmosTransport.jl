# Bitwise gate of the cubed-sphere adjoint tape (core/test_cs_adjoint_tape_bitwise.jl).
#
# Every tape storage and checkpoint schedule must give the same gradients, bit
# for bit, as the device tape with `FullCheckpoint` (the reference variant):
# footprints and the initial adjoint of the objective, seeded and
# multi-objective (Jacobian) entry points, and the 4D-Var cost and gradient
# (layer and column means with a co-sample after the last assimilated step,
# and, for the cases marked `operators`, the observation operators: time
# windows, placed points, averaging-kernel columns with the rebin and the hat
# projection, a cell average, a column the surface-pressure screen leaves out,
# background terms and the preconditioned space).
# The cases use rough fields (random values, spikes, exact zeros and ties, sign
# changes, nonzero emission rates), so the limited PPM reverse meets every
# branch of its limiters; `limiter_code_counts` histograms those branches on
# the recorded tape and the test requires each one to occur.
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
#     --reference-only    run only the reference variant (e.g. on an older
#                         checkout whose tape API differs)
#
# It prints every problem (a variant that differs from the reference, a
# limiter branch that never occurs, a digest that differs from `--check`) and
# exits with status 1 if there is one. A same-host A/B of a branch against
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
end

_limited(scheme) = scheme isa AT.PPMScheme &&
                   (scheme.limiter isa AT.MonotoneLimiter || scheme.limiter isa AT.CW84Limiter)

# The pairwise core set: each scheme, both float types, every convection
# operator and diffusion coupling, both cadences, every reset mode, CFL
# subcycles, and both limiters with every physics family. Each case runs one
# host storage and one stride (compiling a recorder per storage is most of a
# case's time); across the set every limiter meets both host storages and
# both strides, and the mmap tape (sized by counting on the replay) meets CFL
# subcycles and substep CMFMC convection. The observation operators run on one
# Float64 case with CFL subcycles and the mmap tape and one Float32 case with
# the pinned-host tape.
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
    return cases
end

# The inputs of a case: the step's fluxes, schedule, reset targets, surface
# rates and the keywords of its operators (built fresh for every call: the
# workspaces carry scratch).
function problem(c::Case)
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

function physics_kwargs(P)
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

# ---------------------------------------------------------------------------
# Variants `checkpoint/storage`: the reference is the device tape with
# FullCheckpoint. `strideK` is the case's stride (2 and 3 both cross the met
# windows [2, 3, 2]) and `S` each of its host storages.
# ---------------------------------------------------------------------------

const REFERENCE = "full/device"

const VARIANT_TEMPLATES = (
    footprint = [REFERENCE, "strideK/device", "revolve/device", "full/S", "strideK/S", "revolve/S"],
    seeded = [REFERENCE, "strideK/S"],
    jacobian = [REFERENCE, "revolve/S"],
    fourdvar = [REFERENCE, "full/S", "strideK/device", "strideK/S", "revolve/device", "revolve/S"],
    fourdvar_operators = [REFERENCE, "full/S", "strideK/device", "strideK/S", "revolve/device", "revolve/S"],
)

# The entry points a case runs.
entries(c::Case) = c.operators ? keys(VARIANT_TEMPLATES) :
                   filter(!=(:fourdvar_operators), keys(VARIANT_TEMPLATES))

variants(c::Case, entry::Symbol) =
    unique!([replace(t, "K" => string(c.stride), "S" => s)
             for t in VARIANT_TEMPLATES[entry] for s in c.storages])

function _variant_kwargs(variant::AbstractString, scratch::AbstractString)
    schedule, storage = split(variant, '/')
    checkpoint = schedule == "full" ? AT.FullCheckpoint() :
                 schedule == "revolve" ? AT.RevolveCheckpoint() :
                 AT.StrideCheckpoint(parse(Int, chopprefix(schedule, "stride")))
    tape_storage = storage == "mmap" ? AT.MmapCSTapeStorage(mktempdir(scratch)) : Symbol(storage)
    return (; checkpoint, tape_storage)
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

_basis(FT, i) = Adv._d6_basis(FT, i, one(FT))

# The branch `_ppm_limit_profile_monotone_ad` takes at a cell (0 extremum,
# 1 left edge fixed, 2 right edge fixed, 3 unchanged), read from its
# derivative with unit basis derivatives of the left edge, cell and right edge.
function _profile_code(q_L, c, q_R)
    FT = typeof(c)
    b1, b2, b3 = _basis(FT, 1), _basis(FT, 2), _basis(FT, 3)
    _, dq_L, _, dq_R = Adv._ppm_limit_profile_monotone_ad(q_L, b1, c, b2, q_R, b3)
    dq_L == b2 && dq_R == b2 && return 0
    dq_L == b1 && dq_R == b3 && return 3
    dq_R == b3 && return 1
    return 2
end

# The minmod branch `_limited_slope_monotone_ad` takes (0 zero slope, 1
# centred, 2 forward, 3 backward), read from its derivative.
function _slope_code(c_m, c_0, c_p)
    FT = typeof(c_0)
    _, d = Adv._limited_slope_monotone_ad(c_m, _basis(FT, 1), c_0, _basis(FT, 2), c_p, _basis(FT, 3))
    d == Adv._d6_zero(FT) && return 0
    d[2] == 0 && return 1
    d[1] == 0 && return 2
    return 3
end

function _edge(c_ll, c_l, c_r, c_rr, limiter)
    z = Adv._d6_zero(typeof(c_l))
    return first(Adv._ppm_edge_value_ad(c_ll, z, c_l, z, c_r, z, c_rr, z, limiter))
end

# Branch counts at the cells a limited sweep's reverse reads, per axis: x and
# y split into the cells next to a panel edge (whose stencils read the halo,
# "seam") and the others; z clamps the stencil at the column ends as the z
# reverse does. `profile` counts the profile branches of every limited scheme,
# `slope` the minmod branches of the CW84 edge slopes.
function limiter_code_counts(ops, Nc, Nz)
    counts = Dict{String, Vector{Int}}()
    bump!(key, code) = (get!(() -> zeros(Int, 4), counts, key)[code + 1] += 1)
    air, tracer = TapeMod.AirMassRead(), TapeMod.TracerMassRead()
    for op in ops
        op isa TapeMod._CSSweepRecord && op.panels_rm !== nothing || continue
        limiter = op.scheme.limiter
        m = TapeMod._tape_panels(op.panels_m, air)
        rm = TapeMod._tape_panels(op.panels_rm, tracer)
        FT = eltype(m[1])
        floor = Adv._advection_mass_floor(FT)
        cw84 = limiter isa AT.CW84Limiter
        for p in 1:6
            c = rm[p] ./ max.(m[p], floor)
            if op.direction === :z
                for k in 1:Nz, j in HP+1:HP+Nc, i in HP+1:HP+Nc
                    at(o) = c[i, j, clamp(k + o, 1, Nz)]
                    eL = _edge(at(-2), at(-1), at(0), at(1), limiter)
                    eR = _edge(at(-1), at(0), at(1), at(2), limiter)
                    bump!("profile_z", _profile_code(eL, at(0), eR))
                    cw84 && bump!("slope_z", _slope_code(at(-1), at(0), at(1)))
                end
                continue
            end
            axis = String(op.direction)
            for k in 1:Nz, a in HP:HP+Nc+1, b in HP+1:HP+Nc
                cell(o) = op.direction === :x ? c[a + o, b, k] : c[b, a + o, k]
                region = a in (HP, HP + 1, HP + Nc, HP + Nc + 1) ? "seam" : "interior"
                eL = _edge(cell(-2), cell(-1), cell(0), cell(1), limiter)
                eR = _edge(cell(-1), cell(0), cell(1), cell(2), limiter)
                bump!("profile_$(axis)_$(region)", _profile_code(eL, cell(0), eR))
                cw84 && bump!("slope_$(axis)_$(region)", _slope_code(cell(-1), cell(0), cell(1)))
            end
        end
    end
    return counts
end

function case_limiter_codes(P)
    replay, _ = ADJ.Footprint._cs_replay_for_call(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh;
                                                  _common(P)..., base_emission_rates = P.rates)
    ops = ADJ.Footprint._record_cs_tape!(replay, 1:P.nsteps, TapeMod.DeviceCSTapeStorage())
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
counts of the limited-PPM cases, as written by `--out`.
"""
function run_cases(cases; scratch::AbstractString, reference_only::Bool = false)
    out = Dict{String, Any}("host" => host_fingerprint(), "threads" => Threads.nthreads())
    results = Dict{String, Any}()
    for c in cases
        t0 = time()
        P = problem(c)
        digests = Dict{String, Any}()
        for entry in entries(c)
            vs = reference_only ? [REFERENCE] : variants(c, entry)
            digests[String(entry)] = Dict{String, Any}(
                v => Dict{String, Any}(name => digest(x) for (name, x) in run_entry(P, entry, v, scratch))
                for v in vs)
        end
        entry = Dict{String, Any}("digests" => digests)
        if _limited(c.scheme)
            entry["limiter"] = c.scheme.limiter isa AT.CW84Limiter ? "cw84" : "monotone"
            entry["limiter_codes"] = case_limiter_codes(P)
        end
        results[c.name] = entry
        println(stderr, "  ", rpad(c.name, 44), round(time() - t0; digits = 1), " s")
    end
    out["cases"] = results
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
        for (quantity, h) in q
            h == d[REFERENCE][quantity] || push!(bad, "$name $e $v: $quantity differs from $REFERENCE")
        end
    end
    return sort!(bad)
end

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
    # Production logs each growth of the CFL subcycle count; keep warnings.
    out = with_logger(ConsoleLogger(stderr, Warn)) do
        mktempdir(scratch -> run_cases(cases; scratch, reference_only))
    end
    out_file === nothing || open(io -> TOML.print(io, out; sorted = true), out_file, "w")
    record_file === nothing || open(record_file, "w") do io
        println(io, "# Reference digests of test/helpers/cs_adjoint_tape_harness.jl (--record),")
        println(io, "# recorded from main on the [host] below; see test/README.md.")
        TOML.print(io, reference_digests(out); sorted = true)
    end
    problems = vcat(variant_mismatches(out), missing_limiter_codes(out))
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
