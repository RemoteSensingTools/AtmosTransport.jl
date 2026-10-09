# AdvectionWorkspace: the pre-allocated double buffers of the structured and face-indexed sweeps.
# Split from StrangSplitting.jl (refactor phase 4); included by Advection.jl in this order.

using KernelAbstractions: get_backend, synchronize, @kernel, @index, @Const, @atomic, CPU as KA_CPU

# =========================================================================
# AdvectionWorkspace — pre-allocated double buffers
# =========================================================================

"""
    AdvectionWorkspace{FT, A, V1, A4}

Pre-allocated buffers for mass-flux Strang splitting.  Eliminates all
array allocations from the inner time-stepping loop.

The workspace provides TWO complete buffer pairs (`(rm_A, m_A)` and
`(rm_B, m_B)`) used as ping-pong source/destination by
[`strang_split!`](@ref). Each directional sweep reads from one pair and
writes to the other; the palindrome's six sweeps flip parity an even
number of times, so the caller's home arrays receive the final result
naturally.

Kernels always write to a DIFFERENT array than they read from — this is
the double-buffer contract, and in-place updates would violate the
stencil's read-before-write assumption and break mass conservation by
~10% per step.

# Fields
- `rm_A::A`, `rm_B::A` — 3D tracer-mass ping-pong pair (same size as `rm`)
- `m_A::A`, `m_B::A`   — 3D air-mass ping-pong pair (same size as `m`)
- `cluster_sizes::V1` — per-latitude clustering factors for reduced grids
  (`Int32[Ny]`; all ones for uniform grids; empty for face-indexed meshes)
- `face_left::V1`, `face_right::V1` — face connectivity for face-indexed meshes
- `rm_4d_A::A4`, `rm_4d_B::A4` — 4D tracer-mass ping-pong pair for the
  multi-tracer fused path (`(Nx, Ny, Nz, Nt)`). Both are allocated to
  size 0×0×0×0 when `n_tracers == 0`.

# Constructors

    AdvectionWorkspace(m::AbstractArray{FT,3}; cluster_sizes_cpu=nothing, n_tracers=0)

Create workspace for a 3D structured grid, allocating both buffer pairs
matching the size of `m`.  If `cluster_sizes_cpu` is nothing, defaults to
uniform (all ones).

    AdvectionWorkspace(m::AbstractArray{FT,2}; cluster_sizes_cpu=nothing, mesh=nothing)

Create workspace for a 2D face-indexed grid (cell × level layout).
"""
struct AdvectionWorkspace{FT, A <: AbstractArray{FT}, V1 <: AbstractVector{Int32}, A4}
    rm_A           :: A
    m_A            :: A
    rm_B           :: A
    m_B            :: A
    cluster_sizes  :: V1
    face_left      :: V1
    face_right     :: V1
    rm_4d_A        :: A4
    rm_4d_B        :: A4
end

function _face_connectivity_vectors(mesh::AbstractHorizontalMesh)
    left = Vector{Int32}(undef, nfaces(mesh))
    right = Vector{Int32}(undef, nfaces(mesh))
    @inbounds for f in eachindex(left)
        l, r = face_cells(mesh, f)
        left[f] = Int32(l)
        right[f] = Int32(r)
    end
    return left, right
end

function AdvectionWorkspace(m::AbstractArray{FT,3};
                            cluster_sizes_cpu::Union{Nothing, Vector{Int32}} = nothing,
                            n_tracers::Int = 0) where FT
    Nx, Ny, Nz = size(m)
    cs_cpu = cluster_sizes_cpu !== nothing ? cluster_sizes_cpu : ones(Int32, Ny)
    cs_dev = similar(m, Int32, Ny)
    copyto!(cs_dev, cs_cpu)
    face_left = similar(m, Int32, 0)
    face_right = similar(m, Int32, 0)
    rm_4d_A = n_tracers > 0 ? similar(m, Nx, Ny, Nz, n_tracers) : similar(m, 0, 0, 0, 0)
    rm_4d_B = n_tracers > 0 ? similar(m, Nx, Ny, Nz, n_tracers) : similar(m, 0, 0, 0, 0)
    AdvectionWorkspace{FT, typeof(m), typeof(cs_dev), typeof(rm_4d_A)}(
        similar(m), similar(m),                       # rm_A, m_A
        similar(m), similar(m),                       # rm_B, m_B
        cs_dev, face_left, face_right,
        rm_4d_A, rm_4d_B)
end

function AdvectionWorkspace(m::AbstractArray{FT,2};
                            cluster_sizes_cpu::Union{Nothing, Vector{Int32}} = nothing,
                            mesh::Union{Nothing, AbstractHorizontalMesh} = nothing,
                            n_tracers::Int = 0) where FT
    cs_dev = similar(m, Int32, 0)
    if mesh === nothing
        face_left = similar(m, Int32, 0)
        face_right = similar(m, Int32, 0)
    else
        left_cpu, right_cpu = _face_connectivity_vectors(mesh)
        face_left = similar(m, Int32, length(left_cpu))
        face_right = similar(m, Int32, length(right_cpu))
        copyto!(face_left, left_cpu)
        copyto!(face_right, right_cpu)
    end
    # Face-indexed path does NOT use strang_split_mt!; it loops over
    # tracer slices via selectdim. 4D ping-pong buffers stay 0-sized.
    rm_4d_A = similar(m, 0, 0)
    rm_4d_B = similar(m, 0, 0)
    AdvectionWorkspace{FT, typeof(m), typeof(cs_dev), typeof(rm_4d_A)}(
        similar(m), similar(m),                       # rm_A, m_A
        similar(m), similar(m),                       # rm_B, m_B
        cs_dev, face_left, face_right,
        rm_4d_A, rm_4d_B)
end

"""
    AdvectionWorkspace(state::CellState; cluster_sizes_cpu=nothing, mesh=nothing)

Construct a workspace sized for `state`: infers `n_tracers` from
`ntracers(state)` so the 4D ping-pong buffers match the packed tracer
storage. This is the preferred form; the raw `AdvectionWorkspace(m;
n_tracers=…)` is kept for low-level callers.
"""
function AdvectionWorkspace(state::CellState;
                            cluster_sizes_cpu::Union{Nothing, Vector{Int32}} = nothing,
                            mesh::Union{Nothing, AbstractHorizontalMesh} = nothing)
    nt = ntracers(state)
    m = state.air_mass
    if ndims(m) == 3
        return AdvectionWorkspace(m; cluster_sizes_cpu = cluster_sizes_cpu,
                                  n_tracers = nt)
    elseif ndims(m) == 2
        return AdvectionWorkspace(m; cluster_sizes_cpu = cluster_sizes_cpu,
                                  mesh = mesh, n_tracers = nt)
    else
        throw(ArgumentError("unsupported air_mass rank $(ndims(m))"))
    end
end

function Adapt.adapt_structure(to, ws::AdvectionWorkspace{FT}) where {FT}
    rm_A           = Adapt.adapt(to, getfield(ws, :rm_A))
    m_A            = Adapt.adapt(to, getfield(ws, :m_A))
    rm_B           = Adapt.adapt(to, getfield(ws, :rm_B))
    m_B            = Adapt.adapt(to, getfield(ws, :m_B))
    cluster_sizes  = Adapt.adapt(to, getfield(ws, :cluster_sizes))
    face_left      = Adapt.adapt(to, getfield(ws, :face_left))
    face_right     = Adapt.adapt(to, getfield(ws, :face_right))
    rm_4d_A        = Adapt.adapt(to, getfield(ws, :rm_4d_A))
    rm_4d_B        = Adapt.adapt(to, getfield(ws, :rm_4d_B))
    return AdvectionWorkspace{FT, typeof(rm_A), typeof(cluster_sizes), typeof(rm_4d_A)}(
        rm_A, m_A, rm_B, m_B,
        cluster_sizes, face_left, face_right,
        rm_4d_A, rm_4d_B)
end
