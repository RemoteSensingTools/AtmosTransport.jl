# ===========================================================================
# Time series from runtime snapshot files.
#
# A run writes one snapshot file per output period (one per day for the
# campaigns), each holding the frames of that period on the `time` axis in
# hours since the configured simulation start. Diagnostics need one series per
# variable over many files: `daily_snapshot_series` concatenates the frames in
# file order, stamps them with the run origin, and drops a frame whose time
# was already read (the boundary frame a file may repeat from its
# predecessor).
# ===========================================================================

"""
    daily_snapshot_series(paths, variables; origin::DateTime) -> (; times, values)

Concatenate the frames of runtime snapshot files into one series per variable.

- `paths` are read in the order given; pass them in time order (the daily
  files of a run are named `<prefix>_YYYYMMDD.nc`, so `sort` suffices).
- `variables` names snapshot variables whose last dimension is `time`, for
  example `"co2_natural_column_mean"` `(Xdim, Ydim, nf, time)` or
  `"air_mass_per_area"` `(Xdim, Ydim, nf, lev, time)`.
- `origin` is the simulation start. Snapshot files store `time` in hours since
  that start with the bare units `"hours"`, so the origin is not in the file
  and the caller passes it (`[input].start_date` of the run config). A `time`
  variable with any other units is an error. (Reading the origin from CF
  `"hours since …"` units follows when the snapshot writer records it.)

Each frame is stamped `origin + hours`. A frame whose hour was already read is
a duplicated boundary frame and is skipped: the first occurrence wins. A frame
earlier than the last kept frame that is not such a duplicate means the paths
are out of order or come from different runs, and is an error. The first
file of a run holds the initial state at hour 0 as well as its end-of-period
frame; both are kept. Callers that want end-of-day states only drop the frame
at `origin`.

Returns `times::Vector{DateTime}`, strictly increasing, and
`values::Dict{String, Array}`: for each variable an array with the variable's
non-time dimensions in NCDatasets order (for example `(Xdim, Ydim, nf)`)
followed by time. The element type is the stored floating-point type; values
equal to `_FillValue` or `missing_value` become NaN. Every file must hold each
variable with the same non-time shape and element type.
"""
function daily_snapshot_series(paths::AbstractVector{<:AbstractString}, variables;
                               origin::DateTime)
    isempty(paths) && throw(ArgumentError("daily_snapshot_series: no snapshot paths given"))
    names = String[String(v) for v in variables]
    isempty(names) && throw(ArgumentError("daily_snapshot_series: no variables given"))
    allunique(names) || throw(ArgumentError("daily_snapshot_series: variables repeat: $(names)"))

    # Pass 1: frame hours of every file and the frames to keep.
    hours = Float64[]
    kept = Set{Float64}()
    plan = Vector{Vector{Int}}(undef, length(paths))
    for (f, path) in enumerate(paths)
        file_hours = NCDataset(ds -> _snapshot_hours(ds, path), path, "r")
        keep = Int[]
        for (k, h) in enumerate(file_hours)
            h in kept && continue
            if !isempty(hours) && h < hours[end]
                throw(ArgumentError("$(path): frame $(k) at hour $(h) precedes hour " *
                                    "$(hours[end]) read before it; pass the files of one run in time order"))
            end
            push!(keep, k)
            push!(hours, h)
            push!(kept, h)
        end
        plan[f] = keep
    end

    # Pass 2: allocate from the first file, then copy the kept frames.
    values = NCDataset(paths[1], "r") do ds
        Dict{String, Array}(name => _allocate_series(ds, name, paths[1], length(hours))
                            for name in names)
    end
    column = 0
    for (f, path) in enumerate(paths)
        isempty(plan[f]) && continue
        NCDataset(path, "r") do ds
            for name in names
                _copy_frames!(values[name], ds, name, path, plan[f], column)
            end
        end
        column += length(plan[f])
    end

    ms_per_hour = Dates.toms(Dates.Hour(1))
    times = DateTime[origin + Millisecond(round(Int64, h * ms_per_hour)) for h in hours]
    return (; times, values)
end

function _snapshot_hours(ds, path)
    haskey(ds, "time") || throw(ArgumentError("$(path) has no `time` variable"))
    var = ds["time"]
    units = strip(String(get(var.attrib, "units", "")))
    units == "hours" ||
        throw(ArgumentError("$(path): `time` has units \"$(units)\"; expected the snapshot " *
                            "writer's \"hours\" (hours since the simulation start passed as `origin`)"))
    hours = Float64.(collect(var.var[:]))
    all(isfinite, hours) || throw(ArgumentError("$(path): `time` holds non-finite values"))
    return hours
end

# The variable's floating element type and non-time shape, checked.
function _series_layout(ds, name, path)
    haskey(ds, name) || throw(ArgumentError("$(path) has no variable `$(name)`"))
    var = ds[name]
    dims = dimnames(var)
    (!isempty(dims) && last(dims) == "time") ||
        throw(ArgumentError("$(path): `$(name)` has dimensions $(dims); its last dimension must be `time`"))
    T = nonmissingtype(eltype(var))
    T <: AbstractFloat ||
        throw(ArgumentError("$(path): `$(name)` is stored as $(T); a series needs a floating-point type"))
    return T, size(var)[1:end-1]
end

function _allocate_series(ds, name, path, ntime)
    T, shape = _series_layout(ds, name, path)
    return Array{T}(undef, shape..., ntime)
end

function _copy_frames!(dst::Array{T}, ds, name, path, frames, column) where {T}
    S, shape = _series_layout(ds, name, path)
    (S === T && shape == size(dst)[1:end-1]) ||
        throw(DimensionMismatch("$(path): `$(name)` is $(S) of size $(shape); the first file has " *
                                "$(T) of size $(size(dst)[1:end-1])"))
    var = ds[name]
    space = ntuple(_ -> Colon(), length(shape))
    # Read runs of consecutive frames as ranges: a range keeps the time axis, so
    # time-only variables (`<tracer>_total_mass`) come back as arrays too.
    j = 0
    for run in _consecutive_runs(frames)
        block = nomissing(var[space..., run], T(NaN))
        selectdim(dst, ndims(dst), column + j .+ (1:length(run))) .= block
        j += length(run)
    end
    return dst
end

# Split sorted frame indices into ranges of consecutive values.
function _consecutive_runs(frames::Vector{Int})
    runs = UnitRange{Int}[]
    for k in frames
        if !isempty(runs) && k == last(runs[end]) + 1
            runs[end] = first(runs[end]):k
        else
            push!(runs, k:k)
        end
    end
    return runs
end
