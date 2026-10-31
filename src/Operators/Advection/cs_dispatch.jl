# Cubed-sphere state entrypoints; palindrome and recorder calls are unchanged.

@inline function _copy_cs_storage!(dest::NTuple{6}, src::NTuple{6})
    @inbounds for p in 1:6
        copyto!(dest[p], src[p])
    end
    return dest
end

@inline function _similar_cs_storage(src::NTuple{6})
    return ntuple(p -> similar(src[p]), 6)
end

function _cs_transport_step!(::CSSplitSweepStyle,
                             rm_4d::NTuple{6, <:AbstractArray{<:Any, 4}},
                             m,
                             fluxes::CubedSphereFaceFluxState,
                             mesh::CubedSphereMesh,
                             scheme::AbstractAdvectionScheme,
                             workspace::CSAdvectionWorkspace;
                             cfl_limit::Real = DEFAULT_CFL_LIMIT,
                             subcycle_count::Union{Nothing, Integer} = nothing,
                             midpoint::AbstractMidpointHook = NoMidpoint(),
                             recorder::AbstractStepRecorder = NoStepRecorder())
    strang_split_cs_mt!(rm_4d, m, fluxes.am, fluxes.bm, fluxes.cm,
                        mesh, scheme, workspace;
                        cfl_limit = cfl_limit,
                        subcycle_count = subcycle_count,
                        midpoint = midpoint,
                        recorder = recorder)
    return nothing
end

function _cs_transport_step!(::CSSplitSweepStyle,
                             _rm_tracer, _m,
                             _fluxes::CubedSphereFaceFluxState,
                             _mesh::CubedSphereMesh,
                             scheme::AbstractAdvectionScheme,
                             workspace;
                             kwargs...)
    throw(ArgumentError(
        "Cubed-sphere split-sweep advection with $(typeof(scheme)) requires " *
        "`CSAdvectionWorkspace`; got $(typeof(workspace))."))
end

function _cs_transport_step!(::CSLinRoodStyle,
                             rm_tracer, m,
                             fluxes::CubedSphereFaceFluxState,
                             mesh::CubedSphereMesh,
                             scheme::LinRoodPPMScheme{ORD},
                             workspace::CSLinRoodAdvectionWorkspace;
                             cfl_limit::Real = DEFAULT_CFL_LIMIT,
                             subcycle_count::Union{Nothing, Integer} = nothing,
                             midpoint = nothing,
                             recorder::AbstractStepRecorder = NoStepRecorder()) where ORD
    _ = subcycle_count
    # The runtime driver Hp-pads the flux panels; the LinRood cross-term kernels
    # expect the interior faces only. Strip the halo here (see
    # `_cs_flux_*_interior`) so the kernels read the correct cell. `cm` stays
    # padded — the shared vertical `_sweep_z!` indexes it with the Hp offset like
    # the rest of the CS sweeps.
    Nc, Hp = mesh.Nc, mesh.Hp
    am = ntuple(p -> _cs_flux_x_interior(fluxes.am[p], Nc, Hp), 6)
    bm = ntuple(p -> _cs_flux_y_interior(fluxes.bm[p], Nc, Hp), 6)
    _strang_split_linrood_ppm_cs!(rm_tracer, m, am, bm, fluxes.cm,
                                  mesh, Val(ORD), workspace;
                                  cfl_limit = cfl_limit,
                                  midpoint! = midpoint,
                                  vertical = scheme.vertical,
                                  recorder = recorder)
    return nothing
end

function _cs_transport_step!(::CSLinRoodStyle,
                             _rm_tracer, _m,
                             _fluxes::CubedSphereFaceFluxState,
                             _mesh::CubedSphereMesh,
                             scheme::LinRoodPPMScheme,
                             workspace;
                             kwargs...)
    throw(ArgumentError(
        "Cubed-sphere Lin-Rood advection with $(typeof(scheme)) requires " *
        "`CSLinRoodAdvectionWorkspace`; got $(typeof(workspace))."))
end

@inline _meteo_driver_for_substeps(::Nothing) = nothing
@inline function _meteo_driver_for_substeps(meteo)
    return hasproperty(meteo, :driver) ? getproperty(meteo, :driver) : meteo
end

@inline function _cs_runtime_subcycle_count(meteo)
    return uses_binary_substep_contract(_meteo_driver_for_substeps(meteo)) ? 1 : nothing
end

function strang_split!(state::CubedSphereState{B}, fluxes::CubedSphereFaceFluxState{B},
                       grid::AtmosGrid{<:CubedSphereMesh},
                       scheme::AbstractAdvectionScheme;
                       workspace,
                       cfl_limit::Real = DEFAULT_CFL_LIMIT,
                       midpoint::AbstractMidpointHook = NoMidpoint(),
                       meteo = nothing,
                       recorder::AbstractStepRecorder = NoStepRecorder()) where {B <: AbstractMassBasis}
    n_tr = ntracers(state)
    n_tr == 0 && return nothing

    m = state.air_mass
    subcycle_count = _cs_runtime_subcycle_count(meteo)
    if cs_advection_style(scheme) isa CSSplitSweepStyle
        # Queued only: the first sweep follows on the same device queue.
        fill_cs_step_start_halos!(state.tracers_raw, m, grid.horizontal; sync = false)
        record_halo!(recorder, 1)
        _cs_transport_step!(CSSplitSweepStyle(), state.tracers_raw, m, fluxes,
                            grid.horizontal, scheme, workspace;
                            cfl_limit = cfl_limit,
                            subcycle_count = subcycle_count,
                            midpoint = midpoint,
                            recorder = recorder)
        return nothing
    end

    # Lin-Rood transports one tracer at a time from the same initial air mass.
    # Its horizontal operator (`fv_tp_2d_cs!`) refills the edge and corner
    # halos of rm and m before it reads them, so these fills are not read
    # (and a recorder gets no record of them: their output never reaches the
    # result, so their tangent is zero).
    fill_panel_halos!(m, grid.horizontal; dir=1)
    m_save = n_tr > 1 ? _similar_cs_storage(m) : m
    if n_tr > 1
        _copy_cs_storage!(m_save, m)
    end

    for idx in 1:n_tr
        if idx > 1
            _copy_cs_storage!(m, m_save)
        end

        # One tracer at a time: `tracer = idx` selects its diffusion ledger carry.
        rm_tracer = get_tracer(state, idx)
        fill_panel_halos!(rm_tracer, grid.horizontal; dir=1)
        _cs_transport_step!(cs_advection_style(scheme),
                            rm_tracer, m, fluxes, grid.horizontal, scheme, workspace;
                            cfl_limit = cfl_limit,
                            subcycle_count = subcycle_count,
                            midpoint = BoundMidpoint(midpoint, rm_tracer, m, idx),
                            recorder = recorder)
    end

    return nothing
end

function apply!(state::CubedSphereState{B}, fluxes::CubedSphereFaceFluxState{B},
                grid::AtmosGrid{<:CubedSphereMesh},
                scheme::AbstractAdvectionScheme, dt;
                workspace,
                cfl_limit::Real = DEFAULT_CFL_LIMIT,
                midpoint::AbstractMidpointHook = NoMidpoint(),
                meteo = nothing,
                recorder::AbstractStepRecorder = NoStepRecorder()) where {B <: AbstractMassBasis}
    strang_split!(state, fluxes, grid, scheme;
                  workspace = workspace,
                  cfl_limit = cfl_limit,
                  midpoint = midpoint,
                  meteo = meteo,
                  recorder = recorder)
    return nothing
end
