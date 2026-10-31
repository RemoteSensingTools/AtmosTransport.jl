# ---------------------------------------------------------------------------
# Sampling observation objectives on a replayed state.
#
#   * `CSObservationColumnMet` — the column meteorology a placed point
#     observation (`CSPlacedPointObjective`) needs to choose its layer at a
#     sampled state: model-top pressure, gravity, layer temperature (for
#     hypsometric heights) and specific humidity (for total pressures and
#     the virtual temperature of the heights), per step or static.
#   * `_sample_objective(objective, step, panels_rm, panels_m, mesh, met,
#     build)` — the objective's value on the state after `step`, its linear
#     functional of that state (weights from `panels_m`, when `build`), and
#     the level it read. The forward replay `_observe_cs_forward!` calls it
#     once per sampled step of every observation.
#
# A placed point objective's layer comes from the shared rule of the forward
# observation sampler (`Observations.sample_column_layer!`): interface pressures from
# the replayed air mass of the sampled state, hypsometric heights above the
# model surface on the pressure basis (total pressures and the virtual
# temperature by default), then the record's height or pressure. The layer is fixed for
# the evaluation (the meteorology does not depend on the tracer), so the
# functional is the exact derivative of the evaluated value.
#
# Loaded into `Adjoints` after `Observations.jl`.
# ---------------------------------------------------------------------------

# The config table named in placement messages of the inversion path.
const _INVERSION_PLACEMENT_LABEL = "[observations.placement]"

# Near-surface layer temperature of the hypsometric heights when no
# temperature field is given, the default of the forward sampler's
# `layer_height_temperature_kelvin`.

"""
    CSObservationColumnMet(; p_top, gravity = STANDARD_GRAVITY,
                             temperature = 280.0, humidity = nothing, mass_basis = DryBasis())

The column meteorology with which [`CSPlacedPointObjective`](@ref)
observations choose their model layer at each sampled state
(`cs_surface_flux_4dvar(...; column_met)`). With the air mass ``m_k`` (kg per
cell, k = 1 at the top) of the sampled state and the cell area ``A``:

- interface pressures ``p_{1/2} = p_\\mathrm{top}``,
  ``p_{k+1/2} = p_{k-1/2} + g\\,m_k/A``, from `p_top` (Pa, the model-top
  interface, the pure-pressure `A_ifc[1]` of the met model's hybrid
  coordinate) and `gravity` (m s⁻²);
- hypsometric interface heights above the model surface,
  ``z_{k-1/2} = z_{k+1/2} + (R_d T_{v,k}/g)\\ln(p_{k+1/2}/p_{k-1/2})``, with the
  layer `temperature` ``T_k``: a number (K, every layer; the forward sampler's
  `layer_height_temperature_kelvin`, 280 K by default), an
  `Observations.AbstractLayerTemperature`, per-layer panels (an `NTuple{6}` of
  arrays laid out like the air mass, with or without the halo, K, for every
  step), or a vector of such panels with one entry per model step (the state
  after that step). With the pressure basis `TotalFromHumidity` (the default)
  on a dry basis the pressures are the total ones below and
  ``T_{v,k} = T_k(1 + 0.61\\,q_k)`` the virtual temperature; with
  `DryInterfaces` the dry interfaces above and ``T_{v,k} = T_k``;
- for `TotalFromHumidity` on a dry basis (the heights of every placed point
  observation, pressure placement, averaging-kernel columns), the total
  interface pressures ``p_{k+1/2} = p_{k-1/2} + g\\,m_k/(A\\,(1 - q_k))``
  with the specific `humidity` ``q_k`` (kg kg⁻¹ of moist air; `nothing`,
  static panels, or one set of panels per step, as `temperature`).

`mass_basis` is the basis of the air mass (`DryBasis()`, the default of
transport binaries, or `MoistBasis()`).
"""
struct CSObservationColumnMet{T, Q, B <: AbstractMassBasis}
    p_top::Float64
    gravity::Float64
    temperature::T
    humidity::Q
    mass_basis::B
end

function CSObservationColumnMet(; p_top::Real, gravity::Real = STANDARD_GRAVITY,
                                temperature = DEFAULT_OBSERVATION_LAYER_TEMPERATURE_KELVIN,
                                humidity = nothing, mass_basis::AbstractMassBasis = DryBasis())
    isfinite(p_top) && p_top >= 0 || throw(ArgumentError(
        "CSObservationColumnMet p_top must be a finite pressure >= 0 Pa, got $p_top"))
    isfinite(gravity) && gravity > 0 || throw(ArgumentError(
        "CSObservationColumnMet gravity must be finite and positive, got $gravity"))
    t = _layer_temperature_source(temperature)
    _check_temperature_source(t)
    _check_humidity_source(humidity)
    return CSObservationColumnMet(Float64(p_top), Float64(gravity), t, humidity, mass_basis)
end

function _layer_temperature_source(kelvin::Real)
    isfinite(kelvin) && kelvin > 0 || throw(ArgumentError(
        "CSObservationColumnMet temperature must be a positive number of kelvin, got $kelvin"))
    return ConstantLayerTemperature(Float64(kelvin))
end
_layer_temperature_source(t) = t

# The forms the temperature and the humidity may take (see the docstring).
const _ColumnPanels = Union{NTuple{6, AbstractArray{<:Real, 3}},
                            AbstractVector{<:NTuple{6, AbstractArray{<:Real, 3}}}}
_check_temperature_source(::Union{AbstractLayerTemperature, _ColumnPanels}) = nothing
_check_temperature_source(x) = throw(ArgumentError(
    "CSObservationColumnMet temperature must be a number of kelvin, an Observations.AbstractLayerTemperature, " *
    "an NTuple{6} of per-layer panel arrays, or a vector of them (one per model step); got $(typeof(x))"))
_check_humidity_source(::Union{Nothing, _ColumnPanels}) = nothing
_check_humidity_source(x) = throw(ArgumentError(
    "CSObservationColumnMet humidity must be nothing, an NTuple{6} of per-layer panel arrays of specific " *
    "humidity (kg/kg), or a vector of them (one per model step); got $(typeof(x))"))

# Shapes against the run, before the replay.
_check_column_field(::Nothing, name, mesh, Nz, nsteps) = nothing
_check_column_field(::AbstractLayerTemperature, name, mesh, Nz, nsteps) = nothing
function _check_column_field(t::ProfileLayerTemperature, name, mesh, Nz, nsteps)
    length(t.kelvin) == Nz || throw(DimensionMismatch(
        "CSObservationColumnMet $(name) profile has $(length(t.kelvin)) layers; the model has $Nz"))
    return nothing
end
function _check_column_field(panels::NTuple{6}, name, mesh, Nz, nsteps)
    for (p, a) in enumerate(panels)
        size(a, 3) == Nz && size(a, 1) == size(a, 2) && size(a, 1) in (mesh.Nc, mesh.Nc + 2mesh.Hp) ||
            throw(DimensionMismatch(
                "CSObservationColumnMet $(name) panel $p has size $(size(a)); expected " *
                "($(mesh.Nc), $(mesh.Nc), $Nz) or the halo-padded air-mass layout " *
                "($(mesh.Nc + 2mesh.Hp), $(mesh.Nc + 2mesh.Hp), $Nz)"))
    end
    return nothing
end
function _check_column_field(steps::AbstractVector{<:NTuple{6}}, name, mesh, Nz, nsteps)
    length(steps) == nsteps || throw(DimensionMismatch(
        "CSObservationColumnMet $(name) has $(length(steps)) steps; the run has $nsteps"))
    foreach(panels -> _check_column_field(panels, name, mesh, Nz, nsteps), steps)
    return nothing
end

# Before the replay: every placed point observation can choose its layer, and
# every averaging-kernel column can build its interface pressures.
function _validate_column_met(observations, met, mesh::CubedSphereMesh, Nz::Int, nsteps::Int)
    _validate_placed_column_met(CSPlacedPointObjective[o.objective for o in observations
                                                       if o.objective isa CSPlacedPointObjective],
                                met, mesh, Nz, nsteps)
    _validate_retrieval_column_met(CSRetrievalColumnObjective[r for o in observations
                                                              for r in _retrieval_columns(o.objective)],
                                   met, mesh, Nz, nsteps)
    return nothing
end

# The averaging-kernel columns an objective samples: itself, a cell average's
# members, or none.
_retrieval_columns(o::CSRetrievalColumnObjective) = (o,)
_retrieval_columns(o::CSCellAverageObjective) = Tuple(m for m in o.members if m isa CSRetrievalColumnObjective)
_retrieval_columns(::AbstractCSFootprintObjective) = ()

function _validate_retrieval_column_met(columns::Vector{CSRetrievalColumnObjective}, met, mesh::CubedSphereMesh,
                                        Nz::Int, nsteps::Int)
    isempty(columns) && return nothing
    met isa CSObservationColumnMet || throw(ArgumentError(
        "$(length(columns)) observations are averaging-kernel columns (CSRetrievalColumnObjective), which " *
        "build their weights from the interface pressures of each sampled state; pass " *
        "column_met = CSObservationColumnMet(; p_top, gravity, humidity)"))
    _check_column_field(met.humidity, "humidity", mesh, Nz, nsteps)
    for basis in unique(o.pressure_basis for o in columns)
        n = count(o -> o.pressure_basis == basis, columns)
        needs_column_humidity(basis, met.mass_basis) && met.humidity === nothing && throw(ArgumentError(
            "$(n) observations are averaging-kernel columns on total interface pressures " *
            "($(_INVERSION_PLACEMENT_LABEL).pressure_basis = \"total\", the default): a retrieval's pressures " *
            "are total pressures, and on a dry-basis run the model's are built from the dry air mass and the " *
            "specific humidity, but column_met has no humidity. Pass column_met = CSObservationColumnMet(; " *
            "..., humidity), or bind the records with pressure_basis = \"dry\" (the stretch onto the " *
            "retrieval surface pressure then also absorbs the weight of the water vapour, and the " *
            "surface-pressure mismatch includes it)"))
        # The dry interfaces need a dry-basis air mass.
        check_pressure_basis(basis, met.mass_basis, met.humidity !== nothing, n, _INVERSION_PLACEMENT_LABEL)
    end
    return nothing
end
function _validate_placed_column_met(placed::Vector{CSPlacedPointObjective}, met, mesh::CubedSphereMesh,
                                     Nz::Int, nsteps::Int)
    isempty(placed) && return nothing
    met isa CSObservationColumnMet || throw(ArgumentError(
        "$(length(placed)) observations are placed point samples (CSPlacedPointObjective), which choose " *
        "their layer from the column meteorology of each sampled state; pass " *
        "column_met = CSObservationColumnMet(; p_top, gravity, temperature, humidity)"))
    _check_column_field(met.temperature, "temperature", mesh, Nz, nsteps)
    _check_column_field(met.humidity, "humidity", mesh, Nz, nsteps)
    # Every placed observation builds its layer heights on its pressure basis.
    for basis in unique(o.pressure_basis for o in placed)
        check_pressure_basis(basis, met.mass_basis, met.humidity !== nothing,
                             count(o -> o.pressure_basis == basis, placed), _INVERSION_PLACEMENT_LABEL)
    end
    return nothing
end

# One column (k = 1 at the top) of per-layer panels at the interior cell
# `(p, i, j)`, as a host vector; the panels carry the halo or not. Range
# indexing keeps the read GPU-safe.
function _host_column(panels::NTuple{6}, p::Int, i::Int, j::Int, mesh::CubedSphereMesh)
    a = panels[p]
    offset = size(a, 1) == mesh.Nc ? 0 : mesh.Hp
    ii, jj = offset + i, offset + j
    return vec(Array(a[ii:ii, jj:jj, :]))
end

_step_panels(panels::NTuple{6}, ::Int) = panels
_step_panels(steps::AbstractVector{<:NTuple{6}}, step::Int) = steps[step]

_column_temperature(t::AbstractLayerTemperature, step::Int, obj, mesh) = t
_column_temperature(t, step::Int, obj, mesh) =
    ProfileLayerTemperature(Float64.(_host_column(_step_panels(t, step), obj.panel, obj.i, obj.j, mesh)))

"""
    _placed_level(objective::CSPlacedPointObjective, step, panels_m, mesh, met) -> Int

Layer (k = 1 at the top) of a placed point observation on the state after
`step`: the replayed air-mass column of its cell and the column meteorology
of that step through `Observations.sample_column_layer!`, the forward sampler's
rule.
"""
function _placed_level(obj::CSPlacedPointObjective, step::Int, panels_m, mesh::CubedSphereMesh,
                       met::CSObservationColumnMet)
    air = _host_column(panels_m, obj.panel, obj.i, obj.j, mesh)
    nlevel = length(air)
    column = PlacementColumn(nlevel)
    if needs_column_humidity(obj.pressure_basis, met.mass_basis)
        column.humidity .= _host_column(_step_panels(met.humidity, step), obj.panel, obj.i, obj.j, mesh)
    end
    p_half = Vector{Float64}(undef, nlevel + 1)
    z_half = Vector{Float64}(undef, nlevel + 1)
    return sample_column_layer!(column, p_half, z_half, obj.placement, air, obj.area, met.gravity,
                                met.p_top, _column_temperature(met.temperature, step, obj, mesh),
                                obj.pressure_basis, met.mass_basis)
end

# The value of `objective` on the state after `step`, its linear functional of
# that state (`nothing` unless `build`), the level it read (0 when it reads
# more than one level, as a column mean), and the step's diagnostics of an
# averaging-kernel column (`nothing` for other objectives). An objective with a
# fixed stencil needs only `evaluate_objective` and `_linear_functional`
# methods.
_sample_objective(obj::AbstractCSFootprintObjective, step::Int, panels_rm, panels_m, mesh, met, build::Bool) =
    (evaluate_objective(obj, panels_rm, panels_m, mesh),
     build ? _linear_functional(obj, step, panels_m, mesh) : nothing, _fixed_level(obj), nothing)
_fixed_level(obj::CSLayerMeanObjective) = obj.level
_fixed_level(::AbstractCSFootprintObjective) = 0
function _sample_objective(obj::CSPlacedPointObjective, step::Int, panels_rm, panels_m, mesh,
                           met::CSObservationColumnMet, build::Bool)
    layer = CSLayerMeanObjective(obj.panel, obj.i, obj.j, _placed_level(obj, step, panels_m, mesh, met))
    return (evaluate_objective(layer, panels_rm, panels_m, mesh),
            build ? _linear_functional(layer, step, panels_m, mesh) : nothing, layer.level, nothing)
end

"""
    _sample_objective(objective::CSRetrievalColumnObjective, step, panels_rm, panels_m, mesh, met, build)

An averaging-kernel column on the state after `step`. The interface pressures
of the objective's pressure basis come from the replayed air-mass column
``m_k`` of its cell and the column meteorology (`Observations.placement_pressures!`:
total pressures from ``m_k`` and the specific humidity, or the dry
interfaces); `Observations.retrieval_column_weights!` gives the weights ``H_k`` on
the dry mixing ratios ``x_k = r_k/\\max(m_k, ε)``, with the projection of the
column's mapping (the rebin of a layer column, or the hat projection of a
level column, which weights each layer by its replayed air mass ``m_k``). The
value is

``X = c + \\sum_k w_k\\, r_k``, ``w_k = H_k / \\max(m_k, ε)``,

with ``c`` the prior term (`Observations.retrieval_prior_offset`): linear in the
tracer mass with the weights ``w_k`` of its functional (the pressures depend
only on the meteorology), so the functional is the exact derivative of the
value. Also returns the model minus retrieval surface pressure and the
model profile on the retrieval layers or levels ``P x`` (surface first) of
the step.
"""
function _sample_objective(obj::CSRetrievalColumnObjective, step::Int, panels_rm, panels_m, mesh,
                           met::CSObservationColumnMet, build::Bool)
    FT = eltype(panels_m[obj.panel])
    air = _host_column(panels_m, obj.panel, obj.i, obj.j, mesh)
    tracer = _host_column(panels_rm, obj.panel, obj.i, obj.j, mesh)
    nz = length(air)
    humidity = needs_column_humidity(obj.pressure_basis, met.mass_basis) ?
        _host_column(_step_panels(met.humidity, step), obj.panel, obj.i, obj.j, mesh) : Float64[]
    p_half = placement_pressures!(Vector{Float64}(undef, nz + 1), obj.pressure_basis, met.mass_basis, air,
                                  humidity, obj.area, met.gravity, met.p_top)
    s = Vector{Float64}(undef, nz + 1)
    H = retrieval_column_weights!(Vector{Float64}(undef, nz), s, p_half, air, obj.retrieval)
    floor = Float64(_objective_mass_floor(FT))
    weight = FT[FT(H[k] / max(Float64(air[k]), floor)) for k in 1:nz]
    offset = FT(retrieval_prior_offset(obj.retrieval))
    # The value is the functional applied to this state, in its entry order.
    value = offset
    @inbounds for k in 1:nz
        value += weight[k] * tracer[k]
    end
    f = build ? CSLinearFunctional{FT}(step, fill(obj.panel, nz), fill(obj.i, nz), fill(obj.j, nz), 1:nz,
                                       weight, offset) : nothing
    x = Float64[Float64(tracer[k]) / max(Float64(air[k]), floor) for k in 1:nz]
    profile = retrieval_profile!(Vector{Float64}(undef, retrieval_layer_count(obj.retrieval)), s, p_half, air,
                                 obj.retrieval, x)
    sample = (; surface_pressure_mismatch_pa = p_half[end] - obj.retrieval.surface_pressure_pa, profile)
    return (value, f, 0, sample)
end

"""
    _sample_objective(objective::CSCellAverageObjective, step, panels_rm, panels_m, mesh, met, build)

A cell average on the state after `step`: each member sampled on that state
(`_sample_objective`), the value ``\\sum_n w_n h_n`` and the functional
``\\sum_n w_n f_n`` with the members' weights merged level by level (one cell)
and their offsets combined. The members' retrieval diagnostics are not
combined (`nothing`): the members are reported as co-samples.
"""
function _sample_objective(obj::CSCellAverageObjective, step::Int, panels_rm, panels_m, mesh,
                           met, build::Bool)
    FT = eltype(panels_m[obj.panel])
    nz = size(panels_m[obj.panel], 3)
    value = zero(FT)
    weight = zeros(FT, nz)
    offset = zero(FT)
    for (member, w) in zip(obj.members, obj.weights)
        v, f, _, _ = _sample_objective(member, step, panels_rm, panels_m, mesh, met, build)
        value += FT(w) * v
        build || continue
        @inbounds for e in eachindex(f.weight)
            weight[f.k[e]] += FT(w) * f.weight[e]
        end
        offset += FT(w) * f.offset
    end
    f = build ? CSLinearFunctional{FT}(step, fill(obj.panel, nz), fill(obj.i, nz), fill(obj.j, nz), 1:nz,
                                       weight, offset) : nothing
    return (value, f, 0, nothing)
end

# The stencil's placement code, height above the model surface and pressure,
# and its cell ((0, 0, 0) for an operator without a single cell).
_stencil_placement(::CSColumnMeanObjective) = (COLUMN_MEAN_PLACEMENT_CODE, NaN, NaN)
_stencil_placement(::CSRetrievalColumnObjective) = (RETRIEVAL_COLUMN_PLACEMENT_CODE, NaN, NaN)
_stencil_placement(::CSCellAverageObjective) = (CELL_AVERAGE_PLACEMENT_CODE, NaN, NaN)
_stencil_placement(::AbstractCSFootprintObjective) = (FIXED_LEVEL_PLACEMENT_CODE, NaN, NaN)
_stencil_placement(obj::CSPlacedPointObjective) =
    (obj.placement.method, obj.placement.height_agl_m, obj.placement.pressure_pa)
_objective_cell(o::Union{CSLayerMeanObjective, CSColumnMeanObjective, CSPlacedPointObjective,
                         CSRetrievalColumnObjective, CSCellAverageObjective}) = (o.panel, o.i, o.j)
_objective_cell(::AbstractCSFootprintObjective) = (0, 0, 0)

function _observation_stencil(obs::CSObservation, levels::Vector{Int},
                              surface_pressure_mismatch_pa::Float64 = NaN,
                              retrieval_profile::Vector{Float64} = Float64[], assimilated::Bool = true)
    method, height, pressure = _stencil_placement(obs.objective)
    return CSObservationStencil(_objective_cell(obs.objective)..., copy(obs.sampling.steps),
                                copy(obs.sampling.weights), levels, method, height, pressure,
                                surface_pressure_mismatch_pa, retrieval_profile, assimilated)
end

# The time-weighted diagnostics of an averaging-kernel column: add one sampled
# step's surface-pressure mismatch and rebinned profile with its weight.
_accumulate_retrieval_sample!(mismatch, profiles, idx::Int, ::Nothing, weight::Float64) = nothing
function _accumulate_retrieval_sample!(mismatch::Vector{Float64}, profiles::Vector{Vector{Float64}}, idx::Int,
                                       sample::NamedTuple, weight::Float64)
    if isempty(profiles[idx])
        profiles[idx] = zeros(length(sample.profile))
        mismatch[idx] = 0.0
    end
    mismatch[idx] += weight * sample.surface_pressure_mismatch_pa
    profiles[idx] .+= weight .* sample.profile
    return nothing
end

"""
    _screen_surface_pressure(observations, mismatch) -> Vector{Bool}

Whether each observation is assimilated: a co-sample (`assimilate = false`)
never is, and is not screened. An averaging-kernel column whose
time-weighted model minus retrieval surface pressure `mismatch[i]` exceeds
its `max_surface_pressure_mismatch_pa` is an `ArgumentError` under
`ErrorOnUnplaced()` (all such records are named, up to five) and is left out
of the cost under `SkipUnplaced()`. Every other observation is assimilated.
"""
function _screen_surface_pressure(observations, mismatch::Vector{Float64})
    assimilated = trues(length(observations))
    rejected = String[]
    n_rejected = 0
    for (idx, obs) in enumerate(observations)
        obs.assimilate || (assimilated[idx] = false; continue)
        obj = obs.objective
        obj isa CSRetrievalColumnObjective || continue
        abs(mismatch[idx]) <= obj.max_surface_pressure_mismatch_pa && continue
        if obj.mismatch_policy isa SkipUnplaced
            assimilated[idx] = false
        else
            n_rejected += 1
            length(rejected) < 5 && push!(rejected, "id $(obs.id) (Δps = $(round(mismatch[idx]; digits = 1)) Pa, " *
                                                     "limit $(obj.max_surface_pressure_mismatch_pa) Pa)")
        end
    end
    n_rejected == 0 || throw(ArgumentError(
        "$(n_rejected) averaging-kernel column observations differ from the model surface pressure by more " *
        "than max_surface_pressure_mismatch_pa: $(join(rejected, ", "))$(n_rejected > 5 ? ", …" : ""). " *
        "Raise the limit, or set $(_INVERSION_PLACEMENT_LABEL).unplaced_records = \"skip\" to leave them out " *
        "of the cost"))
    return collect(Bool, assimilated)
end
