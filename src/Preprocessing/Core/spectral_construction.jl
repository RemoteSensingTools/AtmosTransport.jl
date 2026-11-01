"""Strategy for constructing ERA5 spectral transport fluxes."""
abstract type AbstractSpectralFluxConstruction end

"""Legacy centre/face sampling; the default keeps the existing arithmetic."""
struct PointSampledSpectralFluxes <: AbstractSpectralFluxConstruction end

"""Retained-spectrum target quadrature with independent vertical forcing."""
struct TM5SpectralFluxes <: AbstractSpectralFluxConstruction
    refinement_steps::Int
    vod2uv_radius::Float64
    function TM5SpectralFluxes(refinement_steps::Integer=2,
                              vod2uv_radius::Real=IFS_EARTH_RADIUS)
        refinement_steps > 0 || throw(ArgumentError("quadrature_refinement_steps must be positive"))
        vod2uv_radius == IFS_EARTH_RADIUS || throw(ArgumentError("TM5 spectral winds require IFS_EARTH_RADIUS"))
        new(Int(refinement_steps), Float64(vod2uv_radius))
    end
end

"""Synthesis implementation; device arrays are owned by the concrete plan."""
abstract type AbstractSpectralSynthesis end
abstract type AbstractSynthesisBackend end
struct CPUSynthesis <: AbstractSynthesisBackend end
struct GPUSynthesis <: AbstractSynthesisBackend end
struct AutoSynthesisPrecision end

spectral_construction(settings) = get(settings, :spectral_flux_construction, PointSampledSpectralFluxes())
spectral_resolution(::PointSampledSpectralFluxes, grid, source_T) = target_spectral_truncation(grid)
spectral_resolution(::TM5SpectralFluxes, grid, source_T) = source_T
# The reader uses zero to retain the source truncation before its header is known.
spectral_read_truncation(c::AbstractSpectralFluxConstruction, grid) = spectral_resolution(c,grid,0)
spectral_source_flux_sampling(::PointSampledSpectralFluxes) = :window_start_endpoint
spectral_source_flux_sampling(::TM5SpectralFluxes) = :window_mean
function balance_window_layers! end

# Extensions implement this without loading a GPU package in core.
validate_gpu_spectral_backend() = validate_gpu_spectral_backend(Val(:cuda))
validate_gpu_spectral_backend(::Val) = throw(ArgumentError(
    "spectral_synthesis=\"gpu\" requires a loaded, functional CUDA backend; use spectral_synthesis=\"cpu\" for CPU preprocessing"))

function resolve_spectral_construction(cfg)
    table = get(cfg, "preprocessing", Dict{String,Any}())
    check_known_keys(table, SPECTRAL_PREPROCESSING_KEYS, "[preprocessing]")
    c = parse_choice(get(table, "spectral_flux_construction", "point_sampled_spectral"),
        (point_sampled_spectral=PointSampledSpectralFluxes(), tm5_spectral=TM5SpectralFluxes()),
        "[preprocessing].spectral_flux_construction")
    c isa PointSampledSpectralFluxes && haskey(table, "quadrature_refinement_steps") &&
        throw(ArgumentError("quadrature_refinement_steps requires tm5_spectral"))
    # Do not add fields to historical settings: repr(settings) is fingerprinted.
    c isa PointSampledSpectralFluxes && !haskey(table, "spectral_synthesis") &&
        !haskey(table, "spectral_synthesis_float") && return NamedTuple()
    synthesis = parse_choice(get(table, "spectral_synthesis", c isa TM5SpectralFluxes ? "gpu" : "cpu"),
        (cpu=CPUSynthesis(), gpu=GPUSynthesis()), "[preprocessing].spectral_synthesis")
    precision = parse_choice(get(table, "spectral_synthesis_float", "auto"),
        (auto=AutoSynthesisPrecision(), float64=Float64, float32=Float32), "[preprocessing].spectral_synthesis_float")
    c isa TM5SpectralFluxes && (c = TM5SpectralFluxes(
        config_int(get(table, "quadrature_refinement_steps", 2), "[preprocessing].quadrature_refinement_steps"; min=1)))
    return (; spectral_flux_construction=c, spectral_synthesis=synthesis,
              spectral_synthesis_float=precision)
end

function validate_spectral_construction(c::TM5SpectralFluxes, grid, settings)
    grid isa Union{LatLonTargetGeometry,ReducedGaussianTargetGeometry} ||
        throw(ArgumentError("tm5_spectral supports lat-lon and reduced-Gaussian targets; cubed sphere is deferred"))
    if grid isa LatLonTargetGeometry
        isapprox(last(grid.mesh.λᶠ)-first(grid.mesh.λᶠ),360;atol=1e-9,rtol=0) &&
            first(grid.mesh.φᶠ)==-90 && last(grid.mesh.φᶠ)==90 ||
            throw(ArgumentError("tm5_spectral requires a global lat-lon target"))
    end
    settings.mass_basis === :dry || throw(ArgumentError("tm5_spectral requires mass_basis=\"dry\""))
    settings.flux_time_sampling === :window_mean || throw(ArgumentError("tm5_spectral requires flux_time_sampling=\"window_mean\""))
    get(settings,:met_interval,ERA5_MET_INTERVAL_SECONDS)==ERA5_MET_INTERVAL_SECONDS ||
        throw(ArgumentError("tm5_spectral requires the hourly ERA5 met_interval of $(ERA5_MET_INTERVAL_SECONDS) seconds"))
    get(settings, :spectral_synthesis, GPUSynthesis()) isa GPUSynthesis && validate_gpu_spectral_backend()
    return nothing
end
function validate_spectral_construction(::PointSampledSpectralFluxes,grid,settings)
    get(settings,:spectral_synthesis,CPUSynthesis()) isa GPUSynthesis && throw(ArgumentError(
        "GPU synthesis on the point-sampled path is not implemented; select tm5_spectral or cpu"))
    return nothing
end

"""Reject partial columns and pressure-overlap remaps before decoding meteorology."""
validate_spectral_vertical(::PointSampledSpectralFluxes, vertical) = nothing
function validate_spectral_vertical(::TM5SpectralFluxes, vertical)
    get(vertical, :vertical_mapping_method, :merge_map) === :merge_map ||
        throw(ArgumentError("tm5_spectral requires retained native interfaces (merge_map)"))
    vc = vertical.vc_native
    first(vertical.level_range) == 1 && first(vc.A) == 0 &&
        first(vc.B) == 0 && last(vc.B) == 1 || throw(ArgumentError(
        "tm5_spectral requires a complete native column: level_top=1, A_top=B_top=0, B_surface=1"))
    return nothing
end
