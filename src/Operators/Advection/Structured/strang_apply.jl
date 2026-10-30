# Model-facing apply! entry points of the advection operators (structured, face-indexed and cubed-sphere states).
# Split from StrangSplitting.jl (refactor phase 4); included by Advection.jl in this order.

# =========================================================================
# apply! entry points
# =========================================================================

"""
    apply!(state, fluxes, grid, scheme, dt; workspace)

Structured-mesh advection entry point.  Delegates to [`strang_split!`](@ref).

The `dt` argument is not used inside the kernels. Callers are responsible
for preparing `fluxes` so they already represent the intended substep forcing.
"""
function apply!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                grid::AtmosGrid{<:LatLonMesh},
                scheme::AbstractAdvectionScheme, dt;
                workspace::AdvectionWorkspace,
                cfl_limit::Real = DEFAULT_CFL_LIMIT,
                midpoint::AbstractMidpointHook = NoMidpoint(),
                meteo = nothing) where {B <: AbstractMassBasis}
    strang_split!(state, fluxes, grid, scheme;
                  workspace = workspace, cfl_limit = cfl_limit,
                  midpoint = midpoint)
    return nothing
end

# Advection alone is the identity; StepPlans owns diffusion-only composition.
@inline function apply!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                         grid::AtmosGrid{<:LatLonMesh}, ::NoAdvection, dt;
                         workspace = nothing,
                         cfl_limit::Real = DEFAULT_CFL_LIMIT,
                         meteo = nothing) where {B <: AbstractMassBasis}
    return nothing
end

@inline function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                         grid::AtmosGrid{<:AbstractHorizontalMesh}, ::NoAdvection, dt;
                         workspace = nothing,
                         cfl_limit::Real = DEFAULT_CFL_LIMIT,
                         meteo = nothing) where {B <: AbstractMassBasis}
    return nothing
end

function _preflight_advection_workspace(workspace::AdvectionWorkspace,
                                        tracers_raw::AbstractArray{FT, 4},
                                        air_mass::AbstractArray{FT, 3}) where FT
    tracer_shape = size(tracers_raw)
    mass_shape = size(air_mass)
    size(workspace.rm_4d_A) == tracer_shape &&
        size(workspace.rm_4d_B) == tracer_shape || throw(DimensionMismatch(
            "advection tracer workspace has shapes $(size(workspace.rm_4d_A)) and " *
            "$(size(workspace.rm_4d_B)); expected $(tracer_shape). Construct it with " *
            "`AdvectionWorkspace(state)`."))
    all(buffer -> size(buffer) == mass_shape,
        (workspace.rm_A, workspace.m_A, workspace.rm_B, workspace.m_B)) ||
        throw(DimensionMismatch(
            "advection mass workspace must match air-mass shape $(mass_shape)"))
    backend = typeof(get_backend(air_mass))
    all(buffer -> eltype(buffer) === FT && typeof(get_backend(buffer)) === backend,
        (workspace.rm_A, workspace.m_A, workspace.rm_B, workspace.m_B,
         workspace.rm_4d_A, workspace.rm_4d_B)) || throw(ArgumentError(
            "advection workspace and state must share one element type and backend"))
    return nothing
end

# Face-indexed transport retains its per-tracer H/V palindrome. The composer
# supplies one hook; each tracer's current arrays and index are passed to it.
for (scheme_type, h_sweep, v_sweep) in (
    (:AbstractConstantScheme,    :sweep_horizontal!, :sweep_vertical!),
)
    @eval function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                          grid::AtmosGrid{<:AbstractHorizontalMesh},
                          scheme::$scheme_type, dt;
                          workspace::AdvectionWorkspace,
                          cfl_limit::Real = DEFAULT_CFL_LIMIT,
                          midpoint::AbstractMidpointHook = NoMidpoint(),
                          meteo = nothing) where {B <: AbstractMassBasis}
        m = state.air_mass
        hflux, cm = fluxes.horizontal_flux, fluxes.cm
        cfl_limit_ft = convert(eltype(m), cfl_limit)

        n_tr = ntracers(state)
        n_tr == 0 && return nothing

        # Every tracer starts from the same air mass, so one count serves all.
        n_sub = _face_indexed_palindrome_subcycle_count(hflux, cm, m, grid.horizontal, cfl_limit_ft)
        fs = inv(eltype(m)(n_sub))

        m_save = n_tr > 1 ? similar(m) : m
        if n_tr > 1
            copyto!(m_save, m)
        end

        # Face-indexed path keeps a per-tracer loop (multi-tracer fusion
        # on unstructured grids is out of scope). Slices are
        # taken as views into state.tracers_raw so the algorithm is
        # identical to the earlier NamedTuple-iterating version; only
        # the data source changed.
        raw = state.tracers_raw
        last_dim = ndims(raw)
        for idx in 1:n_tr
            if idx > 1
                copyto!(m, m_save)
            end

            rm_tracer = selectdim(raw, last_dim, idx)

            for _ in 1:n_sub
                $h_sweep(rm_tracer, m, hflux, grid.horizontal, scheme, workspace, fs)
                $v_sweep(rm_tracer, m, cm, scheme, workspace, fs)
            end
            apply_midpoint!(midpoint, rm_tracer, m, idx)
            for _ in 1:n_sub
                $v_sweep(rm_tracer, m, cm, scheme, workspace, fs)
                $h_sweep(rm_tracer, m, hflux, grid.horizontal, scheme, workspace, fs)
            end
        end

        return nothing
    end
end

# Error stubs for unsupported face-indexed scheme families.
# Additional midpoint/meteorology keywords preserve the same error.
for (scheme_type, label) in (
    (:AbstractLinearScheme,            "linear-reconstruction"),
    (:AbstractQuadraticScheme,         "quadratic-reconstruction"),
)
    @eval function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                          grid::AtmosGrid{<:AbstractHorizontalMesh},
                          scheme::$scheme_type, dt; kwargs...) where {B <: AbstractMassBasis}
        throw(ArgumentError("Face-connected " * $label * " schemes not yet implemented for $(typeof(grid.horizontal))"))
    end
end
