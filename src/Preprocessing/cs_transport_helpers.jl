# ---------------------------------------------------------------------------
# Cubed-sphere transport binary preprocessing helpers.
#
# Line-for-line port from the legacy preprocessing runner (git commit
# ec2d2c0, path scripts_legacy/preprocessing/transport_binary_v2_cs_conservative.jl),
# adapted for the modern streaming pipeline in src/Preprocessing/.
#
# Key functions:
#   - regrid_3d_to_cs_panels!, regrid_2d_to_cs_panels! — conservative LL→CS
#     (intensive fields as given; extensive fields such as `m` through density)
#   - CubedSpherePreprocessWorkspace — all per-window scratch arrays
# The CS face fluxes from the regridded winds (reconstruct_cs_fluxes!) are in
# MassClosure/cs_flux_reconstruction.jl; LL wind recovery lives beside it in
# MassClosure/cs_wind_rotation.jl.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

"""
    CubedSpherePreprocessWorkspace{FT}

Pre-allocated workspace for one or two CS transport windows. Holds all per-panel
arrays needed by the regrid → wind recovery → flux reconstruction → balance → cm
pipeline.
"""
struct CubedSpherePreprocessWorkspace{FT}
    # Current window CS fields
    m_panels      :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz)
    ps_panels     :: NTuple{CS_PANEL_COUNT, Matrix{FT}}      # (Nc, Nc)
    am_panels     :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc+1, Nc, Nz)
    bm_panels     :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc+1, Nz)
    cm_panels     :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz+1)
    # Next-window mass for Poisson balance look-ahead
    m_next_panels :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz)
    # Regridded cell-center winds on CS panels
    u_cs_panels   :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz)
    v_cs_panels   :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz)
    # Pressure thickness on CS panels
    dp_panels     :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz)
    # Mass tendency for cm diagnosis
    dm_panels     :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz)
    # LL cell-center winds (staging grid)
    u_cc          :: Array{FT, 3}   # (Nx_stg, Ny_stg, Nz)
    v_cc          :: Array{FT, 3}   # (Nx_stg, Ny_stg, Nz)
    # Regridder flat I/O buffers
    src_flat_3d   :: Matrix{FT}     # (n_src, Nz)
    dst_flat_3d   :: Matrix{FT}     # (n_dst, Nz)
    src_flat_2d   :: Vector{FT}     # (n_src,)
    dst_flat_2d   :: Vector{FT}     # (n_dst,)
    # Density of one source column for extensive regrids (`apply_regridder!` scratch)
    density_scratch :: Vector{FT}   # (n_src,)
end

function allocate_cs_preprocess_workspace(Nc::Int, Nx_stg::Int, Ny_stg::Int,
                                          Nz::Int, n_src::Int, n_dst::Int,
                                          ::Type{FT}) where FT <: AbstractFloat
    return CubedSpherePreprocessWorkspace{FT}(
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT),
        zeros(FT, Nx_stg, Ny_stg, Nz),
        zeros(FT, Nx_stg, Ny_stg, Nz),
        zeros(FT, n_src, Nz),
        zeros(FT, n_dst, Nz),
        zeros(FT, n_src),
        zeros(FT, n_dst),
        zeros(FT, n_src),
    )
end

# ---------------------------------------------------------------------------
# Panel packing/unpacking
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Conservative LL → CS regridding
# ---------------------------------------------------------------------------

"""
    regrid_3d_to_cs_panels!(panels, regridder, src_3d, ws, Nc,
                             kind::QuantityKind = IntensiveCellField())

Conservatively regrid a 3D LL field `(Nx, Ny, Nz)` to 6 CS panels `(Nc, Nc, Nz)`.

`kind` declares the field's regrid semantics and is passed to
`Regridding.apply_regridder!(dst, regridder, src, kind)`:

- `IntensiveCellField` (default): regridded as given.
  Use for `ps`, `qv`, `T`, cell-center wind components, mixing ratios.
- `ExtensiveCellField`: regridded as the density `src ./ regridder.src_areas`,
  then multiplied by `regridder.dst_areas`. Required for `m` (kg/cell) and any
  per-cell integral; without it the LL polar cells (230× smaller area than
  equator) drag CS polar mass to ~8% of physical and trigger runtime CFL
  inflation.
- `HorizontalVectorField` / `HorizontalFluxField`: not scalar regrids; they
  throw an `ArgumentError`. Winds need component-wise intensive regrid plus
  rotation (`rotate_winds_to_panel_local!`); face fluxes are reconstructed
  from regridded winds.

Uses `ws.src_flat_3d` and `ws.dst_flat_3d` as I/O buffers and
`ws.density_scratch` for the density of extensive fields, so repeated calls
allocate nothing.
"""
function regrid_3d_to_cs_panels!(panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                  regridder,
                                  src_3d::AbstractArray{<:Real, 3},
                                  ws::CubedSpherePreprocessWorkspace{FT},
                                  Nc::Int,
                                  kind::QuantityKind = IntensiveCellField()) where FT
    Nz = size(src_3d, 3)
    _copy_flat!(ws.src_flat_3d, src_3d)
    apply_regridder!(ws.dst_flat_3d, regridder, ws.src_flat_3d, kind; scratch = ws.density_scratch)
    return unpack_flat_to_panels_3d!(panels, ws.dst_flat_3d, Nc, Nz)
end

"""
    regrid_2d_to_cs_panels!(panels, regridder, src_2d, ws, Nc,
                             kind::QuantityKind = IntensiveCellField())

2D analogue of [`regrid_3d_to_cs_panels!`](@ref); same dispatch contract.
"""
function regrid_2d_to_cs_panels!(panels::NTuple{CS_PANEL_COUNT, Matrix{FT}},
                                  regridder,
                                  src_2d::AbstractArray{<:Real, 2},
                                  ws::CubedSpherePreprocessWorkspace{FT},
                                  Nc::Int,
                                  kind::QuantityKind = IntensiveCellField()) where FT
    _copy_flat!(ws.src_flat_2d, src_2d)
    apply_regridder!(ws.dst_flat_2d, regridder, ws.src_flat_2d, kind; scratch = ws.density_scratch)
    return unpack_flat_to_panels_2d!(panels, ws.dst_flat_2d, Nc)
end

# Copy a staging-grid field into a flat regridder buffer in column-major order
# (the flat horizontal index is `i + (j - 1)Nx`). `copyto!` copies linearly, so
# no reshaped view is needed; `reshape` allocates an array header per call on
# Julia 1.10.
function _copy_flat!(flat::AbstractArray, field::AbstractArray)
    length(field) == length(flat) || throw(DimensionMismatch(
        "field of size $(size(field)) does not fill the regridder buffer of size $(size(flat))"))
    return copyto!(flat, field)
end

# ---------------------------------------------------------------------------
# Wind recovery from LL mass fluxes
# ---------------------------------------------------------------------------
