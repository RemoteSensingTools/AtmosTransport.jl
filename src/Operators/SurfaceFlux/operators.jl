"""
    AbstractSurfaceFluxOperator

Top of the surface emission operator hierarchy. Concrete subtypes:

- [`NoSurfaceFlux`](@ref) — identity (default).
- [`SurfaceFluxOperator`](@ref) — wraps a
  [`PerTracerFluxMap`](@ref) and applies each tracer's surface flux
  to `k = Nz` of the conservative tracer-storage array.

Every concrete subtype implements two entry points:

- State-level `apply!(state, meteo, grid, op, dt; workspace)` →
  mutates `state.tracers_raw`.
- Array-level `apply_surface_flux!(q_raw, op, ws, dt, meteo, grid;
  tracer_names)` → mutates a raw tracer buffer directly. Supported
  layouts are:
- structured packed: `(Nx, Ny, Nz, Nt)`
- face-indexed packed: `(ncells, Nz, Nt)`
- face-indexed single-tracer slice: `(ncells, Nz)`
- cubed-sphere single-tracer panels: `NTuple{6}` of `(Nc + 2Hp, Nc + 2Hp, Nz)`
  Used by the structured multi-tracer palindrome and the
  reduced-Gaussian face-indexed transport block.
"""
abstract type AbstractSurfaceFluxOperator <: AbstractOperator end

"""
    NoSurfaceFlux()

Identity operator — `apply!` is a no-op. Default for configurations
without surface emissions, and the value `strang_split_mt!` sees when
the palindrome's S position is unoccupied.

`NoSurfaceFlux`'s `apply!` is literally `= state` (and the array-level
`apply_surface_flux!` is `= nothing`). Julia's dispatch turns the call
site into a dead branch with zero floating-point work and a bit-exact no-op.
"""
struct NoSurfaceFlux <: AbstractSurfaceFluxOperator end

"""
    SurfaceFluxOperator(flux_map::PerTracerFluxMap)
    SurfaceFluxOperator(sources::SurfaceFluxSource...)

Applies a `PerTracerFluxMap` of surface sources to the `k = Nz` slab
of the conservative tracer-storage array.

For every tracer named in the flux map, the operator launches the
layout-appropriate surface-flux kernel and adds `rate × dt` to the
surface layer `k = Nz` of the matching tracer:

- structured packed state `(Nx, Ny, Nz, Nt)` →
  `_surface_flux_kernel!` over `(Nx, Ny)`
- face-indexed packed state `(ncells, Nz, Nt)` →
  `_surface_flux_face_kernel!` over `ncells`

Tracer indices are resolved on the host from `state.tracer_names`.
Tracers absent from the map are untouched.

The rate is a model-storage amount per second per cell (already
area-integrated); no cell-area multiplier appears in the kernel. File-based
physical kg-species rates are converted by the source builder before they
reach this operator.

# `apply!` contract

    apply!(state, meteo, grid, op::SurfaceFluxOperator, dt; workspace)

- Walks `op.flux_map` in storage order.
- For each source, skips tracers not present in `state`.
- Dispatches to the array-level entry point with `tracer_names`
  pulled from the state.
- Synchronises the backend once at the end.

# Fields

- `flux_map :: M` — a [`PerTracerFluxMap`](@ref) of
  `SurfaceFluxSource`s. Emitting tracers are exactly those named in
  the map.
"""
struct SurfaceFluxOperator{M <: PerTracerFluxMap} <: AbstractSurfaceFluxOperator
    flux_map :: M

    SurfaceFluxOperator(flux_map::M) where {M <: PerTracerFluxMap} = new{M}(flux_map)
end

"""
    SurfaceFluxOperator(sources::SurfaceFluxSource...)

Convenience constructor: wraps a variadic list of sources in a
`PerTracerFluxMap` first. Empty variadic list is allowed (produces an
empty-map operator; equivalent to `NoSurfaceFlux` except type-
distinguishable).
"""
SurfaceFluxOperator(sources::SurfaceFluxSource...) =
    SurfaceFluxOperator(PerTracerFluxMap(sources...))

# =========================================================================
# Array-level entry point (for the palindrome)
# =========================================================================

"""
   apply_surface_flux!(q_raw, op::AbstractSurfaceFluxOperator, workspace, dt, meteo, grid;
                        tracer_names)

Array-level surface-flux application. Writes directly to the supplied
tracer buffer `q_raw`, adding each source's `rate × dt` contribution
to the surface slab `k = Nz`.

Supported layouts:

- structured packed: `q_raw :: (Nx, Ny, Nz, Nt)`
- face-indexed packed: `q_raw :: (ncells, Nz, Nt)`
- face-indexed single-tracer slice: `q_raw :: (ncells, Nz)`

`tracer_names::NTuple{Nt, Symbol}` is required as a keyword so the
function can resolve each source's name to a slab index without
reaching back into the caller's `CellState`. This lets the palindrome
integration point the operator at either the caller's
`state.tracers_raw` or the workspace's ping-pong buffer — whichever
currently holds the post-Z-sweep tracer state.

`workspace`, `meteo`, and `grid` are accepted but currently unused;
they are in the signature to match the operator-interface convention
(§"Workflow: Adding a new physics operator" in CLAUDE.md) and to leave
room for future extensions (e.g. meteorology-dependent emissions).

Returns `nothing` on success. For `NoSurfaceFlux`, returns `nothing`
immediately (zero floating-point work).
"""
function apply_surface_flux! end

# NoSurfaceFlux dead branch — dispatched away on the default config
apply_surface_flux!(::AbstractArray{<:Any, 4}, ::NoSurfaceFlux, ws, dt,
                     meteo, grid; tracer_names) = nothing
apply_surface_flux!(::AbstractArray{<:Any, 3}, ::NoSurfaceFlux, ws, dt,
                     meteo, grid; tracer_names) = nothing
apply_surface_flux!(::AbstractArray{<:Any, 2}, ::NoSurfaceFlux, ws, dt,
                     meteo, grid; tracer_names) = nothing
apply_surface_flux!(::NTuple{6}, ::NoSurfaceFlux, ws, dt,
                    meteo, grid; tracer_names, halo_width, deposit = nothing) = nothing

@inline function _check_surface_flux_rate_shape(src::SurfaceFluxSource,
                                                expected_shape::Tuple,
                                                q_raw_shape::Tuple)
    size(src.cell_mass_rate) == expected_shape || throw(DimensionMismatch(
        "surface source $(src.tracer_name) has shape $(size(src.cell_mass_rate)) " *
        "but q_raw surface shape is $(expected_shape) for buffer $(q_raw_shape)"))
    return nothing
end

function apply_surface_flux!(q_raw::AbstractArray{FT, 4},
                             op::SurfaceFluxOperator,
                             workspace,
                             dt::Real,
                             meteo,
                             grid;
                             tracer_names::Tuple) where FT
    Nx, Ny, Nz, _ = size(q_raw)
    backend = get_backend(q_raw)
    dt_FT   = FT(dt)

    for src in op.flux_map.sources
        t_idx = findfirst(==(src.tracer_name), tracer_names)
        t_idx === nothing && continue   # tracer not in this state; skip
        _check_surface_flux_rate_shape(src, (Nx, Ny), size(q_raw))
        launch!(_surface_flux_kernel!, backend, (16, 16), (Nx, Ny), q_raw, src.cell_mass_rate,
                src.compensation, dt_FT, t_idx, Nz; sync = false)
    end

    synchronize(backend)
    return nothing
end

function apply_surface_flux!(q_raw::AbstractArray{FT, 3},
                             op::SurfaceFluxOperator,
                             workspace,
                             dt::Real,
                             meteo,
                             grid;
                             tracer_names::Tuple) where FT
    ncells, Nz, _ = size(q_raw)
    backend = get_backend(q_raw)
    dt_FT   = FT(dt)

    for src in op.flux_map.sources
        t_idx = findfirst(==(src.tracer_name), tracer_names)
        t_idx === nothing && continue
        _check_surface_flux_rate_shape(src, (ncells,), size(q_raw))
        launch!(_surface_flux_face_kernel!, backend, 256, ncells, q_raw, src.cell_mass_rate,
                src.compensation, dt_FT, t_idx, Nz; sync = false)
    end

    synchronize(backend)
    return nothing
end

function apply_surface_flux!(q_raw::AbstractArray{FT, 2},
                             op::SurfaceFluxOperator,
                             workspace,
                             dt::Real,
                             meteo,
                             grid;
                             tracer_names::Tuple) where FT
    length(tracer_names) == 1 || throw(ArgumentError(
        "apply_surface_flux!: single-tracer face-indexed buffer requires " *
        "exactly one tracer name, got $(tracer_names)"))

    ncells, Nz = size(q_raw)
    tracer_name = tracer_names[1]
    src = flux_for(op.flux_map, tracer_name)
    src === nothing && return nothing
    _check_surface_flux_rate_shape(src, (ncells,), size(q_raw))

    backend = get_backend(q_raw)
    launch!(_surface_flux_face_single_kernel!, backend, 256, ncells, q_raw, src.cell_mass_rate,
            src.compensation, FT(dt), Nz)
    return nothing
end

"""
    apply_surface_flux!(q_raw::NTuple{6}, op, workspace, dt, meteo, grid;
                        tracer_names, halo_width, deposit = SurfaceLayerDeposit())

Cubed-sphere surface fluxes on halo-padded panels, single-tracer
`(Nc + 2Hp, Nc + 2Hp, Nz)` (exactly one name in `tracer_names`) or packed
`(…, Nz, Nt)`. `deposit` places each column's emitted mass:
`SurfaceLayerDeposit()` or an `NTuple{6}` of per-panel `ProfileDeposit`s.
"""
function apply_surface_flux!(q_raw::NTuple{6, A}, op::SurfaceFluxOperator, workspace,
                             dt::Real, meteo, grid; tracer_names::Tuple, halo_width::Integer,
                             deposit = SurfaceLayerDeposit()) where {FT, A <: AbstractArray{FT}}
    ndims(A) == 3 && length(tracer_names) != 1 && throw(ArgumentError(
        "apply_surface_flux!: cubed-sphere single-tracer panels require exactly one tracer name, got $(tracer_names)"))
    backend = get_backend(q_raw[1])
    for src in op.flux_map.sources
        t_idx = findfirst(==(src.tracer_name), tracer_names)
        t_idx === nothing && continue
        _apply_cs_surface_flux!(q_raw, src, FT(dt), t_idx, Int(halo_width), backend, meteo, deposit)
    end
    synchronize(backend)
    return nothing
end

"""
    emission_deposit(diffusion_op) -> SurfaceLayerDeposit() or NTuple{6, ProfileDeposit}

Where the diffusion scheme puts fresh surface emissions: the surface layer,
except for GEOS-Chem's non-local VDIFF, whose counter-gradient transport
spreads them over the boundary layer (`GCHPNonlocalPBLField.emission_profile`).
"""
emission_deposit(_diffusion_op) = SurfaceLayerDeposit()
emission_deposit(op::ImplicitVerticalDiffusion{<:Any, <:GCHPNonlocalPBLField}) =
    map(ProfileDeposit, op.kz_field.emission_profile)

@inline _panel_deposit(d::SurfaceLayerDeposit, _p) = d
@inline _panel_deposit(d::NTuple{6, <:ProfileDeposit}, p) = d[p]

function _cs_interior_shape(q_panel, Hp, rate_panel, src, p)
    Nc, Ny = size(q_panel, 1) - 2Hp, size(q_panel, 2) - 2Hp
    size(rate_panel)[1:2] == (Nc, Ny) || throw(DimensionMismatch(
        "surface source $(src.tracer_name) panel $p has shape $(size(rate_panel)) " *
        "but cubed-sphere interior panel shape is $((Nc, Ny))"))
    return Nc, Ny, size(q_panel, 3)
end

# Static per-cell rate.
function _apply_cs_surface_flux!(q_raw::NTuple{6}, src::SurfaceFluxSource, dt_FT, t_idx,
                                 Hp, backend, meteo, deposit)
    rates = src.cell_mass_rate
    rates isa NTuple{6} || throw(ArgumentError(
        "apply_surface_flux!: cubed-sphere source $(src.tracer_name) must provide NTuple{6} panel rates"))
    kernel = _surface_flux_cs_kernel!(backend, (16, 16))
    for p in 1:6
        Nc, Ny, Nz = _cs_interior_shape(q_raw[p], Hp, rates[p], src, p)
        kernel(q_raw[p], rates[p], src.compensation[p], dt_FT, t_idx, Nz, Hp,
               _panel_deposit(deposit, p); ndrange = (Nc, Ny))
    end
    return nothing
end

# Time-varying series: the bracketing slices and weights come from the sim
# clock (`current_time(meteo)`; `nothing` → 0).
function _apply_cs_surface_flux!(q_raw::NTuple{6}, src::TimeVaryingSurfaceFluxSource, dt_FT::FT,
                                 t_idx, Hp, backend, meteo, deposit) where FT
    series = src.cell_mass_rate_series
    series isa NTuple{6} || throw(ArgumentError(
        "apply_surface_flux!: cubed-sphere time-varying source $(src.tracer_name) must provide NTuple{6} panel series"))
    segments = _flux_temporal_segments(src.scheme, src.times, Float64(current_time(meteo)),
                                       Float64(dt_FT))
    kernel = _surface_flux_cs_interp_kernel!(backend, (16, 16))
    for p in 1:6
        Nc, Ny, Nz = _cs_interior_shape(q_raw[p], Hp, series[p], src, p)
        for (i0, i1, w0, w1, frac) in segments
            kernel(q_raw[p], series[p], src.compensation[p], FT(w0), FT(w1), i0, i1,
                   dt_FT * FT(frac), t_idx, Nz, Hp, _panel_deposit(deposit, p); ndrange = (Nc, Ny))
        end
    end
    return nothing
end

# =========================================================================
# State-level entry point
# =========================================================================

# NoSurfaceFlux: identity
function apply!(state::CellState, meteo, grid, ::NoSurfaceFlux, dt;
                workspace = nothing)
    return state
end

function apply!(state::CubedSphereState, meteo, grid, ::NoSurfaceFlux, dt;
                workspace = nothing)
    return state
end

function apply!(state::CellState, meteo, grid, op::SurfaceFluxOperator, dt;
                workspace = nothing)
    apply_surface_flux!(state.tracers_raw, op, workspace, dt, meteo, grid;
                        tracer_names = state.tracer_names)
    return state
end

function apply!(state::CubedSphereState, meteo, grid, op::SurfaceFluxOperator, dt;
                workspace = nothing)
    apply_surface_flux!(state.tracers_raw, op, workspace, dt, meteo, grid;
                        tracer_names = state.tracer_names,
                        halo_width = state.halo_width)
    return state
end

# =========================================================================
# Helpers
# =========================================================================

"""
    emitting_tracer_indices(op::SurfaceFluxOperator, state::CellState) -> NTuple

Ordered tuple of tracer indices in `state.tracer_names` for the
tracers present in `op.flux_map`. Tracers in the map but missing
from the state are skipped (returned as `nothing` slots). Useful for
testing / introspection and unchanged by `apply!`.
"""
function emitting_tracer_indices(op::SurfaceFluxOperator, state::CellState)
    return map(src -> findfirst(==(src.tracer_name), state.tracer_names),
               op.flux_map.sources)
end

emitting_tracer_indices(::NoSurfaceFlux, ::CellState) = ()

# =========================================================================
# Adapt — GPU dispatch
# =========================================================================

Adapt.adapt_structure(to, op::NoSurfaceFlux) = op   # bits-stable singleton

Adapt.adapt_structure(to, op::SurfaceFluxOperator) =
    SurfaceFluxOperator(Adapt.adapt(to, op.flux_map))
