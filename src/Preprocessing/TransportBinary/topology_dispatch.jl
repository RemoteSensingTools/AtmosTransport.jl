# Multiple-dispatch extension points for source/topology preprocessing.

preprocessor_pair_supported(::AbstractTargetGeometry, _settings) = false

function ensure_preprocessor_pair_supported(grid::AbstractTargetGeometry,
                                            settings;
                                            context::AbstractString)
    ensure_supported_target(grid)
    preprocessor_pair_supported(grid, settings) && return nothing
    throw(ArgumentError(
        "$(context) preprocessing does not support source=$(nameof(typeof(settings))) " *
        "target=$(nameof(typeof(grid))). Add a `process_day(::Date, ::$(nameof(typeof(grid))), " *
        "::$(nameof(typeof(settings))), vertical; ...)` method and a matching " *
        "`preprocessor_pair_supported` method, or move this config under likely_legacy."))
end

"""
    process_day(date::Date, grid::AbstractTargetGeometry, settings, vertical; next_day_hour0=nothing)

Topology-specific daily transport-binary preprocessor extension point.

Concrete target geometries implement this method with ordinary Julia multiple
dispatch:

- `LatLonTargetGeometry` writes structured directional LL binaries.
- `ReducedGaussianTargetGeometry` writes face-indexed RG binaries.
- `CubedSphereTargetGeometry` writes panel-local CS binaries.

Every implementation must satisfy the same transport contract:

- use explicit forward endpoint mass targets for every window, including the
  final cross-day window when `next_day_hour0` is available;
- write declared payload semantics, including `delta_semantics`;
- run a write-time replay check unless explicitly disabled for diagnostics;
- produce binaries that the runtime driver can load with replay validation.

This fallback rejects unsupported source/target pairs after config parsing has
already produced an `AbstractTargetGeometry`.
"""
function process_day(date::Date,
                     grid::AbstractTargetGeometry,
                     settings,
                     vertical;
                     next_day_hour0=nothing)
    _ = date
    _ = vertical
    _ = next_day_hour0
    ensure_preprocessor_pair_supported(grid, settings;
                                       context = "transport-binary")
    throw(ArgumentError(
        "No transport-binary preprocessor implementation for " *
        "source=$(nameof(typeof(settings))) target=$(nameof(typeof(grid)))."))
end
