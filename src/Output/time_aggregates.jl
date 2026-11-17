# ---------------------------------------------------------------------------
# Time-aggregated runtime output: `[output.time_mean]` and `[output.local_time]`.
#
# Both products are weighted sums of the selected diagnostics sampled at the
# met-window boundaries of one period (k = 0..n, spacing Δ = window length):
#
#   time mean   (TrapezoidalWeights):    w_k = 1/(2n) at k = 0 and k = n, else 1/n
#   local time  (LocalSolarTimeWeights): w_k(c) = max(0, 1 − |t_c − kΔ| / Δ),
#               t_c = mod(T − lon_c/15, 24) the UTC hour at which cell c reaches
#               local mean solar time T on the period's UTC day.
#
# The weights of every cell sum to 1 over k = 0..n. The boundary shared by two
# periods contributes to both. Sums are Float64 host arrays fed from the frames
# the selected-snapshot capture returns; nothing here runs on a device.
# ---------------------------------------------------------------------------

"Which time aggregate an `[output.<table>]` product writes."
abstract type AbstractTimeAggregate end

"""
    TimeMeanAggregate(period_hours)

Trapezoidal time mean over consecutive periods of `period_hours`, counted from
the run start (`[output.time_mean]`).
"""
struct TimeMeanAggregate <: AbstractTimeAggregate
    period_hours::Float64

    function TimeMeanAggregate(period_hours::Real)
        hours = Float64(period_hours)
        isfinite(hours) && hours > 0 || throw(ArgumentError(
            "[output.time_mean].period_hours must be finite and positive; got $(period_hours)"))
        return new(hours)
    end
end

"""
    LocalSolarTimeAggregate(hours)

One composite per UTC day and local mean solar time in `hours` (sorted;
`0 ≤ h < 24`), linearly interpolated in time per cell (`[output.local_time]`).
Local mean solar time is UTC + longitude/15 h, without the equation of time.
"""
struct LocalSolarTimeAggregate <: AbstractTimeAggregate
    hours::Vector{Float64}

    function LocalSolarTimeAggregate(hours::AbstractVector)
        out = sort!([config_float(h, "[output.local_time].hours") for h in hours])
        isempty(out) && throw(ArgumentError("[output.local_time].hours must not be empty"))
        bad = [h for h in out if !(isfinite(h) && 0 <= h < 24)]
        isempty(bad) || throw(ArgumentError(
            "[output.local_time].hours must lie in [0, 24); got $(join(bad, ", "))"))
        allunique(out) || throw(ArgumentError(
            "[output.local_time].hours lists an hour more than once: $(out)"))
        return new(out)
    end
end

"""
    period_hours(aggregate) -> Float64

Return the length of one averaging period in hours: `period_hours` of a time
mean, one UTC day of a local-time composite.
"""
period_hours(aggregate::TimeMeanAggregate) = aggregate.period_hours
period_hours(::LocalSolarTimeAggregate) = 24.0

"""
    AggregateOutputSpec

Parsed `[output.time_mean]` or `[output.local_time]` table: the aggregate, the
output path template (partitioned like `[output].path`), the NetCDF write
options of `[output]` and the field selection (default: tracer column means).
"""
struct AggregateOutputSpec{A <: AbstractTimeAggregate, P <: AbstractOutputPartition,
                           O <: SnapshotWriteOptions, F <: OutputFieldSpec}
    aggregate::A
    path::String
    partition::P
    options::O
    fields::F
end

const TIME_MEAN_OUTPUT_KEYS = ("enabled", "path", "period_hours", "fields")
const LOCAL_TIME_OUTPUT_KEYS = ("enabled", "path", "hours", "fields")
# TOML table → its keys and aggregate parser, in output order.
const _AGGREGATE_TABLES = (
    ("time_mean", TIME_MEAN_OUTPUT_KEYS,
     cfg -> TimeMeanAggregate(config_float(get(cfg, "period_hours", 24.0),
                                           "[output.time_mean].period_hours"))),
    ("local_time", LOCAL_TIME_OUTPUT_KEYS,
     cfg -> LocalSolarTimeAggregate(_local_time_hours(get(cfg, "hours", nothing)))),
)

function _local_time_hours(value)
    value isa AbstractVector || throw(ArgumentError(
        "[output.local_time].hours must be an array of local solar hours in [0, 24); " *
        "got $(repr(value))"))
    return value
end

_aggregate_table(::TimeMeanAggregate) = "time_mean"
_aggregate_table(::LocalSolarTimeAggregate) = "local_time"
aggregate_label(aggregate::AbstractTimeAggregate) = "[output.$(_aggregate_table(aggregate))]"
aggregate_label(spec::AggregateOutputSpec) = aggregate_label(spec.aggregate)

"""
    aggregate_output_specs(output_cfg, FT) -> Vector{AggregateOutputSpec}

Parse the enabled `[output.time_mean]` and `[output.local_time]` tables of
`[output]`. Unknown keys and a table that is not a table are errors whether or
not the product is enabled; the other checks (wrong types, a missing `path`, a
non-positive `period_hours`, local hours outside `[0, 24)`, selected layers
without `levels`) apply to enabled products. `[output].enabled` does not affect
these products. The files follow `[output].split` and use the `[output]`
NetCDF options at precision `FT`.
"""
function aggregate_output_specs(output_cfg::AbstractDict, ::Type{FT}) where {FT <: AbstractFloat}
    specs = AggregateOutputSpec[]
    for (key, keys, parse) in _AGGREGATE_TABLES
        haskey(output_cfg, key) || continue
        label = "[output.$(key)]"
        cfg = output_cfg[key]
        cfg isa AbstractDict || throw(ArgumentError("$(label) must be a TOML table"))
        check_known_keys(cfg, keys, label)
        config_bool(get(cfg, "enabled", true), "$(label).enabled") || continue
        path = get(cfg, "path", nothing)
        path isa AbstractString && !isempty(path) || throw(ArgumentError(
            "$(label) needs a `path` (a NetCDF file, or a {YYYYMMDD} template with " *
            "[output].split = \"daily\")"))
        fields_cfg = get(cfg, "fields", Dict{String, Any}())
        fields_cfg isa AbstractDict || throw(ArgumentError("$(label).fields must be a TOML table"))
        fields = output_field_spec(fields_cfg; label = "[output.$(key).fields]",
                                   column_means_only = true)
        check_selected_levels(fields, "[output.$(key).fields]")
        push!(specs, AggregateOutputSpec(parse(cfg), expand_data_path(String(path)),
                                         _output_partition(output_cfg),
                                         _output_options(output_cfg, FT, NetCDFOutput()), fields))
    end
    return specs
end

"""
    aggregate_output_path(spec, date_label, day_index) -> String

Return the file of `spec` for the UTC day `date_label` (`yyyymmdd`, day
`day_index` of the run): with `split = "daily"` the day template is
substituted as for snapshots; a single file substitutes a day token, if any,
with the run's first day, which the caller passes.
"""
aggregate_output_path(spec::AggregateOutputSpec{<:Any, SingleOutputFile},
                      date_label::AbstractString, day_index::Integer) =
    _has_day_token(spec.path) ? _substitute_day_template(spec.path, date_label, day_index) : spec.path
aggregate_output_path(spec::AggregateOutputSpec{<:Any, DailyOutputFiles},
                      date_label::AbstractString, day_index::Integer) =
    _substitute_day_template(spec.path, date_label, day_index)

"""
    check_aggregate_origin(aggregate, origin)

Reject a run whose CF time origin `origin` the aggregate cannot use: a
local-time composite covers whole UTC days, so its run must start at 00 UTC.
"""
check_aggregate_origin(::TimeMeanAggregate, origin::DateTime) = nothing
function check_aggregate_origin(::LocalSolarTimeAggregate, origin::DateTime)
    Time(origin) == Time(0) || throw(ArgumentError(
        "[output.local_time] composites whole UTC days, so the run must start at 00 UTC; " *
        "it starts at $(origin) (check [run].start_window)"))
    return nothing
end

# ---------------------------------------------------------------------------
# Boundary weights
# ---------------------------------------------------------------------------

"Weights of the window boundaries of one period."
abstract type AbstractTimeWeighting end

"""
    TrapezoidalWeights()

Trapezoidal time-mean weights: `1/(2n)` at the period's first and last
boundary, `1/n` in between.
"""
struct TrapezoidalWeights <: AbstractTimeWeighting end

"""
    CellWeights

The nonzero per-cell weights of one window boundary, per horizontal part (one
part on lat-lon and reduced-Gaussian grids, six cubed-sphere panels): part `p`
has `ncell[p]` cells, of which `cells[p]` (linear horizontal indices) carry
`weights[p]`.
"""
struct CellWeights
    ncell::Vector{Int}
    cells::Vector{Vector{Int}}
    weights::Vector{Vector{Float64}}
end

"""
    LocalSolarTimeWeights(local_hour, utc_hours, n, window_hours)

Linear-interpolation (hat) weights that sample every cell at `local_hour`
local mean solar time over a UTC day of `n` windows of `window_hours`.
`utc_hours` holds the per-cell UTC sample hour in the topology-native
horizontal layout of snapshot columns. The nonzero weights of each boundary
`k = 0..n` are precomputed once (`slots[k + 1]`); about `2/n` of the cells
have a nonzero weight at any boundary.
"""
struct LocalSolarTimeWeights{H} <: AbstractTimeWeighting
    local_hour::Float64
    utc_hours::H
    slots::Vector{CellWeights}
end

_horizontal_parts(x::Tuple) = collect(x)
_horizontal_parts(x::AbstractArray) = [x]

function LocalSolarTimeWeights(local_hour::Real, utc_hours, n::Integer, window_hours::Real)
    Δ = Float64(window_hours)
    parts = _horizontal_parts(utc_hours)
    slots = map(0:n) do k
        hats = [[max(0.0, 1.0 - abs(t - k * Δ) / Δ) for t in vec(part)] for part in parts]
        cells = [findall(>(0), hat) for hat in hats]
        CellWeights(length.(parts), cells, [hat[c] for (hat, c) in zip(hats, cells)])
    end
    return LocalSolarTimeWeights(Float64(local_hour), utc_hours, slots)
end

"""
    boundary_weights(weighting, k, n, window_hours)

Return the weight of boundary `k` (`0 ≤ k ≤ n`) of a period of `n` windows of
`window_hours`: a number for a time mean, the precomputed nonzero `CellWeights`
for a local-time composite.
"""
boundary_weights(::TrapezoidalWeights, k::Integer, n::Integer, window_hours::Real) =
    (k == 0 || k == n ? 0.5 : 1.0) / n
function boundary_weights(w::LocalSolarTimeWeights, k::Integer, n::Integer, window_hours::Real)
    length(w.slots) == n + 1 || throw(ArgumentError(
        "local-time weights were built for $(length(w.slots) - 1) windows per day, not $(n)"))
    return w.slots[k + 1]
end

_has_weight(w::Real) = !iszero(w)
_has_weight(w::CellWeights) = any(!isempty, w.cells)

# Longitude of every cell centre, in degrees and in the horizontal layout of
# snapshot columns, as the NetCDF coordinates record it.
_cell_longitudes(mesh::LatLonMesh) =
    [Float64(mesh.λᶜ[i]) for i in 1:nx(mesh), _ in 1:ny(mesh)]
_cell_longitudes(mesh::ReducedGaussianMesh) = first(cell_centers(mesh; FT = Float64))
function _cell_longitudes(mesh::CubedSphereMesh)
    lons, _ = _cs_panel_lonlat_arrays(panel_cell_center_lonlat, mesh, mesh.Nc)
    return ntuple(p -> lons[:, :, p], 6)
end

_utc_sample_hours(lons::AbstractArray, local_hour::Float64) =
    map(lon -> mod(local_hour - lon / 15, 24.0), lons)
_utc_sample_hours(lons::Tuple, local_hour::Float64) = map(p -> _utc_sample_hours(p, local_hour), lons)

"""
    time_weightings(aggregate, mesh, window_hours) -> Vector{<:AbstractTimeWeighting}

Return the weightings of one aggregate, one per record of a period: a single
trapezoidal mean, or one `LocalSolarTimeWeights` per requested local hour.
"""
time_weightings(::TimeMeanAggregate, mesh, window_hours::Real) = [TrapezoidalWeights()]
function time_weightings(aggregate::LocalSolarTimeAggregate, mesh, window_hours::Real)
    lons = _cell_longitudes(mesh)
    n = boundaries_per_period(aggregate, window_hours)
    return [LocalSolarTimeWeights(h, _utc_sample_hours(lons, h), n, window_hours)
            for h in aggregate.hours]
end

# ---------------------------------------------------------------------------
# Accumulated frames
# ---------------------------------------------------------------------------

"""
    AveragedSnapshotFrame

Host-resident Float64 weighted sum of selected snapshot diagnostics. Unlike
`SelectedSnapshotFrame`, which stores tracer storage and lets the writer form
mixing ratios, it stores the diagnostics themselves (per-layer VMR, column
means, column sums, air mass), so their average is the average of each
diagnostic. `levels` maps stored layers back to model levels.
`tracer_total_mass` holds weighted global totals (time means only).
"""
struct AveragedSnapshotFrame{A, C} <: AbstractSnapshotFrame
    time_hours::Float64
    air_mass::A
    vmr::Dict{Symbol, A}
    mass_basis::Symbol
    tracer_total_mass::Dict{Symbol, Float64}
    nlevel::Int
    levels::Vector{Int}
    column_air::Union{Nothing, C}
    column_mean::Dict{Symbol, C}
    column_tracers::Dict{Symbol, C}
end

_nlevel(frame::AveragedSnapshotFrame, mesh) = frame.nlevel
_layer_positions(frame::AveragedSnapshotFrame, levels) = _stored_indices(frame, levels)
_air_layers(frame::AveragedSnapshotFrame, levels) =
    _stored_layers(frame.air_mass, _stored_indices(frame, levels), length(frame.levels))
_frame_vmr(frame::AveragedSnapshotFrame, name, levels) =
    _stored_layers(frame.vmr[name], _stored_indices(frame, levels), length(frame.levels))
function _frame_column_air(frame::AveragedSnapshotFrame)
    frame.column_air === nothing && throw(ArgumentError("column air mass was not averaged"))
    return frame.column_air
end
_frame_column_tracer(frame::AveragedSnapshotFrame, name) = frame.column_tracers[name]
_frame_column_mean(frame::AveragedSnapshotFrame, name) = frame.column_mean[name]
_cs_stack_tracer_vmr!(buf, frame::AveragedSnapshotFrame, name, positions) =
    _cs_stack_layers!(buf, frame.vmr[name], positions)

_with_time(f::AveragedSnapshotFrame, time_hours::Real) =
    AveragedSnapshotFrame(Float64(time_hours), f.air_mass, f.vmr, f.mass_basis,
                          f.tracer_total_mass, f.nlevel, f.levels, f.column_air,
                          f.column_mean, f.column_tracers)

# What one product averages, fixed when the product is built: its tracers, the
# union of stored layers and the per-diagnostic tracer lists.
struct _AggregateLayout
    names::Vector{Symbol}
    nlevel::Int
    levels::Vector{Int}
    air::Bool
    layer_tracers::Vector{Symbol}
    column_air::Bool
    column_mean::Vector{Symbol}
    column_mass::Vector{Symbol}
end

function _AggregateLayout(fields::OutputFieldSpec, all_names::AbstractVector{Symbol}, Nz::Integer;
                         label::AbstractString)
    names = _select_tracer_keys(collect(Symbol, all_names), fields; label)
    air = (fields.air_mass || fields.air_mass_per_area) &&
          !isempty(_layer_indices(fields.air_mass_layers, fields, Nz))
    tf(name) = tracer_fields(fields, name)
    return _AggregateLayout(names, Int(Nz), _stored_levels(fields, names, Nz), air,
                            [n for n in names if !isempty(_layer_indices(tf(n).layers, fields, Nz))],
                            fields.column_air_mass_per_area,
                            [n for n in names if tf(n).column_mean],
                            [n for n in names if tf(n).column_mass_per_area])
end

_horizontal_zeros(mesh::LatLonMesh) = zeros(Float64, nx(mesh), ny(mesh))
_horizontal_zeros(mesh::ReducedGaussianMesh) = zeros(Float64, ncells(mesh))
_horizontal_zeros(mesh::CubedSphereMesh) = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
_layer_zeros(mesh::LatLonMesh, nlayer::Integer) = zeros(Float64, nx(mesh), ny(mesh), nlayer)
_layer_zeros(mesh::ReducedGaussianMesh, nlayer::Integer) = zeros(Float64, ncells(mesh), nlayer)
_layer_zeros(mesh::CubedSphereMesh, nlayer::Integer) =
    ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc, nlayer), 6)

function _zero_frame(layout::_AggregateLayout, mesh, mass_basis_sym::Symbol)
    nlayer = length(layout.levels)
    air = _layer_zeros(mesh, layout.air ? nlayer : 0)
    A, C = typeof(air), typeof(_horizontal_zeros(mesh))
    return AveragedSnapshotFrame{A, C}(
        NaN, air, Dict{Symbol, A}(name => _layer_zeros(mesh, nlayer) for name in layout.layer_tracers),
        mass_basis_sym, Dict{Symbol, Float64}(name => 0.0 for name in layout.names), layout.nlevel,
        copy(layout.levels), layout.column_air ? _horizontal_zeros(mesh) : nothing,
        Dict{Symbol, C}(name => _horizontal_zeros(mesh) for name in layout.column_mean),
        Dict{Symbol, C}(name => _horizontal_zeros(mesh) for name in layout.column_mass))
end

_fill_zero!(x::AbstractArray) = fill!(x, 0.0)
_fill_zero!(x::Tuple) = foreach(_fill_zero!, x)
_fill_zero!(::Nothing) = nothing
function _reset!(frame::AveragedSnapshotFrame)
    _fill_zero!(frame.air_mass)
    foreach(_fill_zero!, values(frame.vmr))
    _fill_zero!(frame.column_air)
    foreach(_fill_zero!, values(frame.column_mean))
    foreach(_fill_zero!, values(frame.column_tracers))
    for name in keys(frame.tracer_total_mass)
        frame.tracer_total_mass[name] = 0.0
    end
    return frame
end

# The diagnostics of one sampled frame that a product averages, as host
# arrays in snapshot layout (`_frame_*` accessors, so full, selected and
# shared frames give the same values).
function _sample_diagnostics(layout::_AggregateLayout, frame::AbstractSnapshotFrame)
    return (air = layout.air ? _air_layers(frame, layout.levels) : nothing,
            vmr = Dict(name => _frame_vmr(frame, name, layout.levels) for name in layout.layer_tracers),
            column_air = layout.column_air ? _frame_column_air(frame) : nothing,
            column_mean = Dict(name => _frame_column_mean(frame, name) for name in layout.column_mean),
            column_mass = Dict(name => _frame_column_tracer(frame, name) for name in layout.column_mass),
            totals = Dict(name => frame.tracer_total_mass[name] for name in layout.names))
end

# acc += w x for one weight, or for the nonzero per-cell weights only, so NaN
# cells (no air mass) never reach samples that do not weight them.
function _accumulate!(acc::Array{Float64}, x::AbstractArray, w::Real)
    size(acc) == size(x) || throw(DimensionMismatch("sample $(size(x)) vs sum $(size(acc))"))
    wf = Float64(w)
    @inbounds for i in eachindex(acc, x)
        acc[i] += wf * Float64(x[i])
    end
    return acc
end
# Only the cells with a nonzero weight are visited (trailing layers broadcast).
function _accumulate_cells!(acc::Array{Float64}, x::AbstractArray, ncell::Int,
                            cells::Vector{Int}, weights::Vector{Float64})
    size(acc) == size(x) || throw(DimensionMismatch("sample $(size(x)) vs sum $(size(acc))"))
    A = reshape(acc, ncell, :)
    X = reshape(x, ncell, :)
    @inbounds for k in axes(A, 2), i in eachindex(cells, weights)
        c = cells[i]
        A[c, k] += weights[i] * Float64(X[c, k])
    end
    return acc
end
_accumulate!(acc::Array{Float64}, x::AbstractArray, w::CellWeights) =
    _accumulate_cells!(acc, x, only(w.ncell), only(w.cells), only(w.weights))
_accumulate!(acc::Tuple, x::Tuple, w::CellWeights) =
    foreach(p -> _accumulate_cells!(acc[p], x[p], w.ncell[p], w.cells[p], w.weights[p]),
            eachindex(acc, x))
_accumulate!(acc::Tuple, x::Tuple, w::Real) = foreach((a, b) -> _accumulate!(a, b, w), acc, x)
_accumulate!(::Nothing, ::Nothing, w) = nothing

# Global totals are averaged only with one weight for the whole globe.
function _accumulate_totals!(acc::Dict{Symbol, Float64}, totals, w::Real)
    for (name, value) in totals
        acc[name] += Float64(w) * value
    end
    return acc
end
_accumulate_totals!(acc, totals, w) = acc

function _accumulate!(acc::AveragedSnapshotFrame, d, w)
    _accumulate!(_air_sum(acc, d), d.air, w)
    for (name, x) in d.vmr
        _accumulate!(acc.vmr[name], x, w)
    end
    _accumulate!(acc.column_air, d.column_air, w)
    for (name, x) in d.column_mean
        _accumulate!(acc.column_mean[name], x, w)
    end
    for (name, x) in d.column_mass
        _accumulate!(acc.column_tracers[name], x, w)
    end
    _accumulate_totals!(acc.tracer_total_mass, d.totals, w)
    return acc
end
# The air sum exists only when air layers are averaged.
_air_sum(acc::AveragedSnapshotFrame, d) = d.air === nothing ? nothing : acc.air_mass

# ---------------------------------------------------------------------------
# Capture coverage and union capture
# ---------------------------------------------------------------------------

# Why `frame` cannot provide the diagnostics `fields` selects for `names`, or
# `nothing` when it can.
function _capture_shortfall(frame::SelectedSnapshotFrame, fields::OutputFieldSpec, names, Nz::Integer)
    for name in names
        haskey(frame.tracers, name) || return "tracer $(name) was not captured"
    end
    all(level -> insorted(level, frame.levels), _stored_levels(fields, names, Nz)) ||
        return "writer requested levels that were not captured"
    needs_air = fields.column_air_mass_per_area ||
                any(name -> tracer_fields(fields, name).column_mean, names)
    needs_air && frame.column_air === nothing && return "column air mass was not captured"
    for name in names
        tf = tracer_fields(fields, name)
        (tf.column_mean || tf.column_mass_per_area) && !haskey(frame.column_tracers, name) &&
            return "column tracer $(name) was not captured"
    end
    return nothing
end
function _capture_shortfall(frame::AbstractSnapshotFrame, ::OutputFieldSpec, names, ::Integer)
    for name in names
        haskey(frame.tracers, name) || return "tracer $(name) was not captured"
    end
    return nothing
end

_union_layers(::FullLayerSelection, ::AbstractLayerSelection) = FullLayerSelection()
_union_layers(::AbstractLayerSelection, ::FullLayerSelection) = FullLayerSelection()
_union_layers(::FullLayerSelection, ::FullLayerSelection) = FullLayerSelection()
_union_layers(::SelectedLayerSelection, ::Union{SelectedLayerSelection, NoLayerSelection}) =
    SelectedLayerSelection()
_union_layers(::NoLayerSelection, ::SelectedLayerSelection) = SelectedLayerSelection()
_union_layers(::NoLayerSelection, ::NoLayerSelection) = NoLayerSelection()

# One capture selection that provides every product's diagnostics.
function _union_field_spec(specs, all_names::AbstractVector{Symbol})
    none = TracerOutputFields(NoLayerSelection(), false, false)
    per = Dict{Symbol, TracerOutputFields}()
    levels = Int[]
    air_layers = NoLayerSelection()
    air_mass = air_area = column_air = false
    for fields in specs
        names = _select_tracer_keys(collect(Symbol, all_names), fields)
        air = fields.air_mass || fields.air_mass_per_area
        uses_selected = air && fields.air_mass_layers isa SelectedLayerSelection ||
                        any(n -> tracer_fields(fields, n).layers isa SelectedLayerSelection, names)
        uses_selected && append!(levels, fields.selected_levels)
        if air
            air_layers = _union_layers(air_layers, fields.air_mass_layers)
            air_mass |= fields.air_mass
            air_area |= fields.air_mass_per_area
        end
        column_air |= fields.column_air_mass_per_area
        for name in names
            tf, prev = tracer_fields(fields, name), get(per, name, none)
            per[name] = TracerOutputFields(_union_layers(prev.layers, tf.layers),
                                           prev.column_mean || tf.column_mean,
                                           prev.column_mass_per_area || tf.column_mass_per_area)
        end
    end
    captured = [name for name in all_names if haskey(per, name)]
    return OutputFieldSpec(captured, sort!(unique!(levels)), none, per, air_layers,
                           air_mass, air_area, column_air)
end

# ---------------------------------------------------------------------------
# NetCDF sink
# ---------------------------------------------------------------------------

_time_weighting_description(::TimeMeanAggregate) = "trapezoidal over met-window boundaries"
_time_weighting_description(::LocalSolarTimeAggregate) =
    "linear interpolation between the two met-window boundaries bracketing the sample hour"
_aggregate_title(::TimeMeanAggregate) = "AtmosTransport runtime time mean"
_aggregate_title(::LocalSolarTimeAggregate) = "AtmosTransport runtime local solar time composite"
_aggregate_product(::TimeMeanAggregate) = "time_mean"
_aggregate_product(::LocalSolarTimeAggregate) = "local_solar_time"
_record_dims(::TimeMeanAggregate) = ("time",)
_record_dims(::LocalSolarTimeAggregate) = ("local_time", "time")
_record_index(::TimeMeanAggregate, slot::Integer, t::Integer) = (t,)
_record_index(::LocalSolarTimeAggregate, slot::Integer, t::Integer) = (slot, t)

_horizontal_dims(::LatLonMesh) = ("lon", "lat")
_horizontal_dims(::ReducedGaussianMesh) = ("cell",)
_horizontal_dims(::CubedSphereMesh) = ("Xdim", "Ydim", "nf")
_horizontal_coordinates(::LatLonMesh) = "lon lat"
_horizontal_coordinates(::ReducedGaussianMesh) = "cell_lon cell_lat"
_horizontal_coordinates(::CubedSphereMesh) = "lons lats"
_horizontal_values(::Union{LatLonMesh, ReducedGaussianMesh}, values) = values
_horizontal_values(::CubedSphereMesh, values) = _cs_stack2(values)

# Product-specific header: the local-time axis and the per-cell UTC sample hour.
_define_aggregate_axes!(ds, ::TimeMeanAggregate, weightings, mesh) = nothing
function _define_aggregate_axes!(ds, aggregate::LocalSolarTimeAggregate, weightings, mesh)
    defDim(ds, "local_time", length(aggregate.hours))
    defVar(ds, "local_time", Float64, ("local_time",),
           attrib = Dict("units" => "hours",
                         "long_name" => "local mean solar time",
                         "comment" => "UTC + longitude/15 h (no equation of time)"))[:] =
        aggregate.hours
    dims = _horizontal_dims(mesh)
    v = defVar(ds, "utc_sample_hour", Float64, (dims..., "local_time"),
               attrib = Dict("units" => "hours",
                             "long_name" => "UTC hour of the day at which the cell is sampled",
                             "coordinates" => _horizontal_coordinates(mesh),
                             "comment" => "mod(local_time - longitude/15, 24); each composite " *
                                          "interpolates linearly between the two met-window " *
                                          "boundaries of its UTC day that bracket this hour"))
    for (slot, w) in enumerate(weightings)
        v[ntuple(_ -> Colon(), length(dims))..., slot] = _horizontal_values(mesh, w.utc_hours)
    end
    ds.attrib["local_time_definition"] =
        "local mean solar time = UTC + longitude/15 h; one composite per UTC day " *
        "(time_bnds) and local_time"
    return nothing
end

# Time means: every time-varying variable is a mean over the record's bounds.
_annotate_aggregate_variables!(ds, ::LocalSolarTimeAggregate) = nothing
function _annotate_aggregate_variables!(ds, ::TimeMeanAggregate)
    for name in keys(ds)
        name in ("time", "time_bnds") && continue
        "time" in dimnames(ds[name]) || continue
        existing = String(get(ds[name].attrib, "cell_methods", ""))
        ds[name].attrib["cell_methods"] = isempty(existing) ? "time: mean" :
                                          existing * " time: mean"
    end
    return nothing
end

_write_aggregate_totals!(ds, ::LocalSolarTimeAggregate, frames, names, mass_basis_sym, offset) =
    nothing
_write_aggregate_totals!(ds, ::TimeMeanAggregate, frames, names, mass_basis_sym, offset) =
    _write_tracer_total_mass!(ds, frames, names, mass_basis_sym; time_offset = offset)

"Single-writer NetCDF file of one aggregate product; switches files with the partition."
mutable struct AggregateNetCDFSink
    path::String
    dataset::Union{Nothing, NCDataset}
    count::Int
    geometry::Any
    closed::Bool
end
AggregateNetCDFSink() = AggregateNetCDFSink("", nothing, 0, nothing, false)

function _close_dataset!(sink::AggregateNetCDFSink)
    dataset = sink.dataset
    sink.dataset = nothing
    dataset === nothing || with_netcdf_lock(() -> close(dataset))
    return nothing
end

function Base.close(sink::AggregateNetCDFSink)
    sink.closed && return nothing
    try
        _close_dataset!(sink)
    finally
        sink.closed = true
    end
    return nothing
end

# ---------------------------------------------------------------------------
# One product
# ---------------------------------------------------------------------------

"""
    TimeAggregateOutput

Runtime state of one `[output.time_mean]` or `[output.local_time]` product: the
weightings, one Float64 sum per record of the current period, the period
counter and the NetCDF sink. Feed it every window boundary in order with
`sample_window_boundary!` on its `TimeAggregateOutputs`.
"""
mutable struct TimeAggregateOutput{S <: AggregateOutputSpec, W <: AbstractTimeWeighting,
                                   F <: AveragedSnapshotFrame, G}
    spec::S
    weightings::Vector{W}
    sums::Vector{F}
    layout::_AggregateLayout
    grid::G
    mass_basis::Symbol
    physics_attributes::Dict{String, String}
    time_origin::DateTime
    window_hours::Float64
    boundaries_per_period::Int
    period::Int
    last_k::Int
    sink::AggregateNetCDFSink
end

"""
    boundaries_per_period(aggregate, window_hours) -> Int

Return the number of met windows in one period, which must be a whole number.
"""
function boundaries_per_period(aggregate::AbstractTimeAggregate, window_hours::Real)
    P = period_hours(aggregate)
    n = round(Int, P / window_hours)
    n >= 1 && abs(n * window_hours - P) <= 1e-6 * max(P, 1.0) || throw(ArgumentError(
        "$(aggregate_label(aggregate)): the period of $(P) h is not a positive multiple " *
        "of the $(window_hours) h met window"))
    return n
end

function TimeAggregateOutput(spec::AggregateOutputSpec, grid, tracer_names::AbstractVector{Symbol};
                             window_hours::Real, mass_basis::Symbol,
                             physics_attributes::AbstractDict{<:AbstractString, <:AbstractString},
                             time_origin::DateTime)
    mesh = grid.horizontal
    layout = _AggregateLayout(spec.fields, tracer_names, nlevels(grid);
                              label = "[output.$(_aggregate_table(spec.aggregate)).fields]")
    weightings = time_weightings(spec.aggregate, mesh, window_hours)
    sums = [_zero_frame(layout, mesh, mass_basis) for _ in weightings]
    return TimeAggregateOutput(spec, weightings, sums, layout, grid, mass_basis,
                               Dict{String, String}(physics_attributes), time_origin,
                               Float64(window_hours),
                               boundaries_per_period(spec.aggregate, window_hours),
                               0, -1, AggregateNetCDFSink())
end

# `diagnostics()` computes the frame's diagnostics on first use, so a boundary
# that no record weights costs nothing.
function _add_boundary!(out::TimeAggregateOutput, diagnostics, k::Integer)
    n = out.boundaries_per_period
    for (w, acc) in zip(out.weightings, out.sums)
        wk = boundary_weights(w, k, n, out.window_hours)
        _has_weight(wk) && _accumulate!(acc, diagnostics(), wk)
    end
    return nothing
end

# Diagnostics of one frame, computed once per distinct product layout.
_same_layout(a::_AggregateLayout, b::_AggregateLayout) =
    all(name -> getfield(a, name) == getfield(b, name), fieldnames(_AggregateLayout))
function _cached_diagnostics!(cache::Vector{Any}, layout::_AggregateLayout, frame)
    for (cached_layout, d) in cache
        _same_layout(cached_layout, layout) && return d
    end
    d = _sample_diagnostics(layout, frame)
    push!(cache, (layout, d))
    return d
end

function _sample_boundary!(out::TimeAggregateOutput, frame::AbstractSnapshotFrame, boundary::Integer,
                           cache::Vector{Any} = Any[])
    frame.mass_basis == out.mass_basis || throw(ArgumentError(
        "$(aggregate_label(out.spec)): frame mass_basis $(frame.mass_basis) differs from " *
        "$(out.mass_basis)"))
    p, k = fldmod(boundary, out.boundaries_per_period)
    diagnostics() = _cached_diagnostics!(cache, out.layout, frame)
    if k == 0 && p > 0
        # The boundary closes period p - 1 (as its k = n) and opens period p.
        _add_boundary!(out, diagnostics, out.boundaries_per_period)
        _write_period!(out)
        foreach(_reset!, out.sums)
        out.period = p
    end
    out.period == p || throw(ArgumentError(
        "$(aggregate_label(out.spec)): boundary $(boundary) is outside period $(out.period)"))
    _add_boundary!(out, diagnostics, k)
    out.last_k = k
    return nothing
end

_day_label(origin::DateTime, hours::Real) =
    Dates.format(origin + Millisecond(round(Int64, 3_600_000 * hours)), "yyyymmdd")

# The file of the period starting `start_hours` after the run start: one file
# (a day token takes the run's first day) or one per UTC day of the period start.
_aggregate_path(spec::AggregateOutputSpec{<:Any, SingleOutputFile}, origin, start_hours) =
    aggregate_output_path(spec, _day_label(origin, 0), 1)
_aggregate_path(spec::AggregateOutputSpec{<:Any, DailyOutputFiles}, origin, start_hours) =
    aggregate_output_path(spec, _day_label(origin, start_hours),
                          floor(Int, start_hours / 24 + 1e-9) + 1)

function _open_aggregate_file!(out::TimeAggregateOutput, path::AbstractString, frames)
    sink = out.sink
    _close_dataset!(sink)
    mesh = out.grid.horizontal
    aggregate = out.spec.aggregate
    _ensure_parent_dir(path)
    ds = sink.dataset = _create_netcdf_dataset(path)
    sink.path = String(path)
    sink.count = 0
    _define_common_attributes!(ds, mesh, frames, out.mass_basis; options = out.spec.options,
                               physics_attributes = out.physics_attributes)
    ds.attrib["title"] = _aggregate_title(aggregate)
    ds.attrib["output_product"] = _aggregate_product(aggregate)
    ds.attrib["output_fields"] = _fields_string(out.spec.fields, out.layout.names)
    ds.attrib["period_hours"] = period_hours(aggregate)
    ds.attrib["window_hours"] = out.window_hours
    ds.attrib["time_weighting"] = _time_weighting_description(aggregate)
    ds.attrib["completed_records"] = 0
    sink.geometry = _define_geometry!(ds, mesh, out.layout.nlevel, nothing, out.time_origin)
    ds["time"].attrib["bounds"] = "time_bnds"
    defDim(ds, "bnds", 2)
    defVar(ds, "time_bnds", Float64, ("bnds", "time"),
           attrib = Dict("units" => _time_units(out.time_origin), "calendar" => _TIME_CALENDAR))
    _define_aggregate_axes!(ds, aggregate, out.weightings, mesh)
    return ds
end

function _write_period!(out::TimeAggregateOutput)
    P = period_hours(out.spec.aggregate)
    t0, t1 = out.period * P, (out.period + 1) * P
    frames = [_with_time(acc, (t0 + t1) / 2) for acc in out.sums]
    path = _aggregate_path(out.spec, out.time_origin, t0)
    sink = out.sink
    sink.closed && throw(ArgumentError("cannot write to a closed time-aggregate output"))
    aggregate = out.spec.aggregate
    mesh = out.grid.horizontal
    with_netcdf_lock() do
        ds = path == sink.path && sink.dataset !== nothing ? sink.dataset :
             _open_aggregate_file!(out, path, frames)
        t = sink.count + 1
        _write_aggregate_totals!(ds, aggregate, frames, out.layout.names, out.mass_basis, t - 1)
        _write_snapshot_payload!(ds, mesh, frames, out.layout.names, sink.geometry,
                                 out.mass_basis, out.spec.options, out.spec.fields;
                                 record_dims = _record_dims(aggregate),
                                 records = [_record_index(aggregate, j, t) for j in eachindex(frames)],
                                 reuse = t > 1)
        t == 1 && _annotate_aggregate_variables!(ds, aggregate)
        ds["time"][t] = (t0 + t1) / 2
        ds["time_bnds"][:, t] = [t0, t1]
        NCDatasets.sync(ds)
        ds.attrib["completed_records"] = t
        ds.attrib["history"] = replace(String(ds.attrib["history"]),
            r"with \d+ (frame|record)\(s\)$" => "with $(t) record(s)")
        NCDatasets.sync(ds)
        sink.count = t
    end
    return path
end

function _finish!(out::TimeAggregateOutput)
    if out.last_k > 0
        P = period_hours(out.spec.aggregate)
        @warn "$(aggregate_label(out.spec)): the final period ($(out.period * P) h to " *
              "$((out.period + 1) * P) h after the run start) is incomplete " *
              "($(out.last_k) of $(out.boundaries_per_period) windows); it is not written."
    end
    close(out.sink)
    return nothing
end

Base.close(out::TimeAggregateOutput) = close(out.sink)

# ---------------------------------------------------------------------------
# All products of one run
# ---------------------------------------------------------------------------

"""
    TimeAggregateOutputs(specs, grid, tracer_names; window_hours, halo_width, mass_basis,
                         physics_attributes, time_origin)

The time-aggregated products of one run. `sample_window_boundary!` captures
the union of their field selections once per window boundary (or reuses a
snapshot frame captured at the same instant that covers it) and feeds every
product. Close it on every exit; `finish_time_aggregates!` ends a run that
completed.
"""
mutable struct TimeAggregateOutputs
    outputs::Vector{TimeAggregateOutput}
    capture_fields::Union{Nothing, OutputFieldSpec}
    halo_width::Int
    window_hours::Float64
    next_boundary::Int
end

TimeAggregateOutputs() = TimeAggregateOutputs(TimeAggregateOutput[], nothing, 0, 1.0, 0)

function TimeAggregateOutputs(specs::AbstractVector{<:AggregateOutputSpec}, grid,
                              tracer_names::AbstractVector{Symbol};
                              window_hours::Real, halo_width::Integer, mass_basis::Symbol,
                              physics_attributes::AbstractDict{<:AbstractString, <:AbstractString},
                              time_origin::DateTime)
    isempty(specs) && return TimeAggregateOutputs()
    outputs = TimeAggregateOutput[TimeAggregateOutput(spec, grid, tracer_names; window_hours,
                                                      mass_basis, physics_attributes, time_origin)
                                  for spec in specs]
    capture = _union_field_spec([spec.fields for spec in specs], collect(Symbol, tracer_names))
    return TimeAggregateOutputs(outputs, capture, Int(halo_width), Float64(window_hours), 0)
end

has_time_aggregates(aggregates::TimeAggregateOutputs) = !isempty(aggregates.outputs)

"""
    sample_window_boundary!(aggregates, model, time_hours; shared_frame=nothing)

Sample the run state at the next met-window boundary (hour 0 first, then every
window end) for every time-aggregated product, writing each period that the
boundary completes. `shared_frame` is a snapshot captured at the same instant;
it is used instead of a new capture when it holds every needed diagnostic.
Return the frame that was sampled (`nothing` without products).
"""
function sample_window_boundary!(aggregates::TimeAggregateOutputs, model, time_hours::Real;
                                 shared_frame::Union{Nothing, AbstractSnapshotFrame} = nothing)
    has_time_aggregates(aggregates) || return nothing
    boundary = aggregates.next_boundary
    abs(time_hours - boundary * aggregates.window_hours) <= 1e-6 * max(1.0, time_hours) ||
        throw(ArgumentError("time-aggregated output expected window boundary $(boundary) at " *
                            "$(boundary * aggregates.window_hours) h, got $(time_hours) h"))
    fields = aggregates.capture_fields
    Nz = first(aggregates.outputs).layout.nlevel
    frame = shared_frame !== nothing &&
            _capture_shortfall(shared_frame, fields, fields.tracers, Nz) === nothing ?
            shared_frame :
            capture_snapshot(model; time_hours, halo_width = aggregates.halo_width, fields)
    # Products with the same field selection share one set of diagnostics.
    cache = Any[]
    for out in aggregates.outputs
        _sample_boundary!(out, frame, boundary, cache)
    end
    aggregates.next_boundary += 1
    return frame
end

"""
    finish_time_aggregates!(aggregates)

End a completed run: warn about each product whose final period is incomplete
(it is not written) and close the files.
"""
function finish_time_aggregates!(aggregates::TimeAggregateOutputs)
    foreach(_finish!, aggregates.outputs)
    return nothing
end

function Base.close(aggregates::TimeAggregateOutputs)
    errors = Any[]
    for out in aggregates.outputs
        try
            close(out)
        catch err
            push!(errors, err)
        end
    end
    isempty(errors) || throw(CompositeException(errors))
    return nothing
end
