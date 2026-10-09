# CSAdvectionWorkspace: the per-panel buffers of the cubed-sphere palindrome.
# Split from CubedSphereStrang.jl (refactor phase 4); included by Advection.jl in this order.

# =========================================================================
# CS workspace — pre-allocated buffers for one panel
# =========================================================================

"""
    CSAdvectionWorkspace{FT, A, P3, A4, P4}

Pre-allocated cubed-sphere transport workspace.

- `rm_A`, `m_A` are the halo-padded single-tracer advection ping-pong
  buffers shared across panels.
- `rm_4d_A` is the packed-tracer panel buffer used by the production
  split-sweep path so CS follows the same packed `tracers_raw` paradigm
  as structured grids.
- `m_pp_buf`, `rm_4d_pp_buf` are full-panel spare buffers for the packed
  ping-pong path, avoiding the per-sweep copy-back kernels.
- `seam_flux` holds one air and N-tracer transfer per physical panel edge;
  its storage scales with edge length, not panel area. Lin–Rood workspaces
  disable this split-sweep cache (`seam_transport=false`).
- `column_scratch` is the per-column, per-tracer working storage of the FV3
  vertical profile (`Nc × Nc × (Nz+1) × 3 max(Nt_s, 1)`, `Nt_s =
  column_scratch_tracers`, the packed tracer count by default); it is empty
  unless the workspace is built with `column_scratch=true`.
- `max_subcycles` tracks this workspace's high-water mark for CFL diagnostics;
  keeping it with the workspace prevents unrelated simulations sharing state.
"""
struct CSAdvectionWorkspace{FT, A <: AbstractArray{FT, 3},
                            P3 <: NTuple{6, <:AbstractArray{FT, 3}},
                            A4 <: AbstractArray{FT, 4},
                            P4 <: NTuple{6, <:AbstractArray{FT, 4}}}
    rm_A       :: A
    m_A        :: A
    rm_4d_A    :: A4
    m_pp_buf   :: P3
    rm_4d_pp_buf :: P4
    seam_flux  :: A4
    column_scratch :: A4
    max_subcycles :: Base.RefValue{NTuple{3, Int}}
end

function CSAdvectionWorkspace(mesh::CubedSphereMesh, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              array_type::Type{<:AbstractArray} = Array,
                              n_tracers::Integer = 0,
                              seam_transport::Bool = true,
                              column_scratch::Bool = false,
                              column_scratch_tracers::Integer = n_tracers)
    N = mesh.Nc + 2 * mesh.Hp
    Nt = Int(n_tracers)
    Nt >= 0 || throw(ArgumentError("CSAdvectionWorkspace: n_tracers must be non-negative, got $n_tracers"))
    rm_A = array_type(zeros(FT, N, N, Nz))
    m_A  = array_type(zeros(FT, N, N, Nz))
    rm_4d_A = array_type(zeros(FT, N, N, Nz, Nt))
    m_pp_buf = Nt > 0 ? ntuple(_ -> array_type(zeros(FT, N, N, Nz)), 6) :
                         ntuple(_ -> m_A, 6)
    rm_4d_pp_buf = Nt > 0 ? ntuple(_ -> array_type(zeros(FT, N, N, Nz, Nt)), 6) :
                            ntuple(_ -> rm_4d_A, 6)
    seam_flux = similar(rm_4d_A, FT, mesh.Nc, Nz, max(Nt, 1) + 1, seam_transport ? 12 : 0)
    scratch = _column_scratch(rm_4d_A, mesh.Nc, Nz, Int(column_scratch_tracers), column_scratch)
    return CSAdvectionWorkspace{FT, typeof(rm_A),
                                typeof(m_pp_buf), typeof(rm_4d_A),
                                typeof(rm_4d_pp_buf)}(
        rm_A, m_A, rm_4d_A, m_pp_buf, rm_4d_pp_buf, seam_flux, scratch,
        Ref((1, 1, 1)))
end

function CSAdvectionWorkspace(mesh::CubedSphereMesh,
                              prototype::AbstractArray{FT, 3};
                              n_tracers::Integer = 0,
                              seam_transport::Bool = true,
                              column_scratch::Bool = false,
                              column_scratch_tracers::Integer = n_tracers) where {FT <: AbstractFloat}
    N = mesh.Nc + 2 * mesh.Hp
    Nz = size(prototype, 3)
    Nt = Int(n_tracers)
    Nt >= 0 || throw(ArgumentError("CSAdvectionWorkspace: n_tracers must be non-negative, got $n_tracers"))
    rm_A = similar(prototype, FT, N, N, Nz)
    m_A = similar(prototype, FT, N, N, Nz)
    rm_4d_A = similar(prototype, FT, N, N, Nz, Nt)
    m_pp_buf = Nt > 0 ? ntuple(_ -> similar(prototype, FT, N, N, Nz), 6) :
                         ntuple(_ -> m_A, 6)
    rm_4d_pp_buf = Nt > 0 ? ntuple(_ -> similar(prototype, FT, N, N, Nz, Nt), 6) :
                            ntuple(_ -> rm_4d_A, 6)
    seam_flux = similar(rm_4d_A, FT, mesh.Nc, Nz, max(Nt, 1) + 1, seam_transport ? 12 : 0)
    scratch = _column_scratch(rm_4d_A, mesh.Nc, Nz, Int(column_scratch_tracers), column_scratch)
    return CSAdvectionWorkspace{FT, typeof(rm_A),
                                typeof(m_pp_buf), typeof(rm_4d_A),
                                typeof(rm_4d_pp_buf)}(
        rm_A, m_A, rm_4d_A, m_pp_buf, rm_4d_pp_buf, seam_flux, scratch,
        Ref((1, 1, 1)))
end

# Per-column, per-tracer working storage of the FV3 vertical profile
# (`FV3Column` in `vertical_fv3_profile.jl`); empty unless a scheme needs it.
_column_scratch(prototype, Nc, Nz, Nt, needed::Bool) =
    needed ? similar(prototype, Nc, Nc, Nz + 1, 3 * max(Nt, 1)) : similar(prototype, 0, 0, 0, 0)

function Adapt.adapt_structure(to, ws::CSAdvectionWorkspace{FT}) where FT
    rm_A = Adapt.adapt(to, ws.rm_A)
    m_A = Adapt.adapt(to, ws.m_A)
    rm_4d_A = Adapt.adapt(to, ws.rm_4d_A)
    m_pp_buf = Adapt.adapt(to, ws.m_pp_buf)
    rm_4d_pp_buf = Adapt.adapt(to, ws.rm_4d_pp_buf)
    seam_flux = Adapt.adapt(to, ws.seam_flux)
    column_scratch = Adapt.adapt(to, ws.column_scratch)
    return CSAdvectionWorkspace{FT, typeof(rm_A),
                                typeof(m_pp_buf), typeof(rm_4d_A),
                                typeof(rm_4d_pp_buf)}(
        rm_A, m_A, rm_4d_A, m_pp_buf, rm_4d_pp_buf, seam_flux, column_scratch,
        Ref(ws.max_subcycles[]))
end

@inline function _record_cs_subcycle_growth!(workspace::CSAdvectionWorkspace,
                                              n_x::Int, n_y::Int, n_z::Int)
    mx, my, mz = workspace.max_subcycles[]
    if n_x > mx || n_y > my || n_z > mz
        workspace.max_subcycles[] = (max(mx, n_x), max(my, n_y), max(mz, n_z))
        @info "strang_split_cs! subcycle count grew" n_x n_y n_z
    end
    return nothing
end
