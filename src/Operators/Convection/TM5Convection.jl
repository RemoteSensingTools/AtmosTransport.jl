# ---------------------------------------------------------------------------
# TM5Convection — TM5-style four-field Tiedtke 1989 mass-flux convection.
#
# Real kernel launches go via `_tm5_{…}_kernel!` in tm5_kernels.jl,
# which wraps `_tm5_solve_column!` (tm5_column_solve.jl) per thread.
# All three topologies (LatLon, ReducedGaussian, CubedSphere) are
# supported.
# ---------------------------------------------------------------------------

"""
    TM5Convection(; tile_workspace_gib=1.0, use_collab_lu=false,
                    lmax_conv=0, n_merge=1)

TM5-style convective transport operator. Four-field mass-flux scheme
following Tiedtke (1989) as implemented in TM5-4DVAR: two entrainment
and two detrainment fields (updraft + downdraft). The backward-Euler
transport matrix `conv1 = I - dt·D` is dense within the cloud window
and identity above; the solver assembles and factorizes only the active
lower-right cloud block. The legacy workspace retains its pivot vector for
supported adjoint replay; collaborative factors live only in shared memory.

`use_collab_lu` selects the workgroup-collaborative solver. `lmax_conv=0`
uses all vertical levels; a positive value limits the active lower atmosphere.
`n_merge` optionally aggregates adjacent active layers before the solve.
Float32 GPU matrices support effective depths 1..85. Float64 CUDA supports
unmerged (`n_merge=1`) depths 1..73, using Float64 shared arrays and arithmetic.
CPU and unsupported Float64 configurations retain the legacy solver with a
warning. Tracers use batches of six; total tracer count does not set this depth
limit. Defaults remain the full-column legacy solve. CS adjoint footprints
still require the full-column, unmerged legacy variant.

The forcing arrays `(entu, detu, entd, detd)` arrive via
`TransportModel.convection_forcing.tm5_fields`, populated each
substep by `DrivenSimulation._refresh_forcing!` from
`sim.window.convection`.

# Memory budget

`tile_workspace_gib` (binary GiB) is the per-topology target for
the TM5 column-tile workspace.
`_convection_workspace_for(::TM5Convection, ...)` reads this field
and derives a tile size `B` via [`derive_tile_columns`](@ref); the
[`TM5Workspace`](@ref) uses a single `(Nz, Nz, B)` matrix slab plus
matching pivot, cloud-dimension, and flux-vector slabs. A larger budget
means fewer legacy kernel launches at the cost of more memory.
When collaborative LU is requested, global scratch is deferred because
the kernels use shared memory. Unsupported configurations allocate the configured
tile on first use and reuse it. The default legacy budget is 1.0 GiB.

# Basis convention

`TM5Convection` is **basis-polymorphic**, identical to
[`CMFMCConvection`](@ref). The four forcing fields
must be on the same basis as `state.air_mass`. ECMWF and TM5 archive
convection fields are total-air (moist) fluxes upstream; maintained
preprocessors convert them to the binary's declared basis before writing and
record that basis as `tm5_convection_mass_basis`. Opening a binary rejects a
recorded mismatch. Legacy dry binaries without that key warn because their
unconverted moist fluxes make rates about 1 % too large.

# Fields required on `ConvectionForcing`

- `forcing.tm5_fields :: NamedTuple{(:entu, :detu, :entd, :detd)}`
  with all four arrays at layer centers in AtmosTransport
  orientation (k=1=TOA, k=Nz=surface). Units kg / m² / s.
  Shapes per topology:
  - Structured LatLon: `(Nx, Ny, Nz)` per field.
  - Face-indexed ReducedGaussian: `(ncells, Nz)` per field.
  - Panel-native CubedSphere: `NTuple{6, AbstractArray{FT, 3}}`
    per field, with per-panel shape `(Nc, Nc, Nz)`.

Orientation conversion + sign flip on `entd` happen in the preprocessor
(`src/Preprocessing/tm5_convection_conversion.jl`). The operator performs
zero runtime orientation gymnastics.

# Solver class

Partial-pivot Gaussian elimination on the active lower-right block per
column (see `test/core/test_tm5_sparsity_above_icltop.jl` for the structure
survey).
Identity rows above the effective cloud top and the lower-left zero
quadrant are skipped by both the factorization and the tracer solve.

Pivoting is kept even though the matrix is diagonally dominant by
construction (upstream Fortran comment says pivoting "not needed").
The pivot vector is stored in [`TM5Workspace`](@ref) so the adjoint
can replay the same factorization with `trans='T'`.

# CFL sub-cycling

None. The backward-Euler matrix solve is unconditionally stable
for any `dt`, unlike `CMFMCConvection`'s forward-Euler two-pass
update which requires sub-cycling when the CMFMC profile is
strong. The kernel launches once per tile and calls
`synchronize(backend)` once per `apply!`.
"""
struct TM5Convection{FT} <: AbstractConvection
    # Budget for the legacy column tile, in binary GiB.
    tile_workspace_gib :: FT
    # Factor one column per workgroup and reuse it across tracer batches.
    # Shared storage and arithmetic retain the tracer precision.
    use_collab_lu :: Bool
    # Zero uses all Nz levels. A positive ceiling selects that many bottom
    # levels; layers above it pass through unchanged. Choose this from the
    # forcing's cloud depths and transported-field accuracy, not speed alone.
    lmax_conv :: Int
    # One keeps the transport grid. Larger values sum adjacent fine layers
    # into floor(L/n_merge) super-layers, where L is the active depth (Nz
    # when lmax_conv=0), for the collaborative solve.
    # The effective fine span can therefore lose up to n_merge-1 top layers.
    # Results are redistributed using each tracer's prior fine-layer profile,
    # with uniform weights when the prior super-layer total is zero. This
    # preserves its total up to rounding but changes vertical mixing.
    n_merge :: Int

    # Validate every construction path, including Adapt and explicit type calls.
    function TM5Convection{FT}(tile_workspace_gib::FT,
                                use_collab_lu::Bool,
                                lmax_conv::Int,
                                n_merge::Int) where {FT}
        n_merge >= 1 || throw(ArgumentError(
            "TM5Convection: `n_merge` must be ≥ 1 (got $(n_merge))"))
        return new{FT}(tile_workspace_gib, use_collab_lu, lmax_conv, n_merge)
    end
end

function TM5Convection(; tile_workspace_gib::Real = 1.0,
                         use_collab_lu::Bool = false,
                         lmax_conv::Integer = 0,
                         n_merge::Integer = 1)
    # Validation is delegated to the inner constructor so the
    # parametric `TM5Convection{FT}(args...)` path is guarded too.
    return TM5Convection{typeof(tile_workspace_gib)}(
        tile_workspace_gib, use_collab_lu, Int(lmax_conv), Int(n_merge))
end

# =========================================================================
# Array-level entry: apply_convection!
# =========================================================================

# Effective lmax_conv: 0 means "no truncation, use the full Nz";
# otherwise the operator's explicit cap is used.
@inline _effective_lmax_conv(op::TM5Convection, Nz::Integer) =
    op.lmax_conv == 0 ? Int(Nz) : op.lmax_conv

# Shared storage and arithmetic use the tracer precision. The backend trait
# gates matrix depth; Float64 currently supports only unmerged CUDA columns.
# CPU retains the serial solve because KA cannot lower these group indices.
@inline function _use_collab_path(op::TM5Convection, Nz::Integer, Nt::Integer,
                                   ::Type{FT}, backend) where FT
    L = _effective_lmax_conv(op, Nz)
    nm = max(1, op.n_merge)
    # Aggregation rounds down so the active fine span cannot exceed the
    # requested ceiling. It is a scientific configuration, not an automatic
    # shared-memory optimization.
    L_super = nm > 1 ? fld(L, nm) : L
    L_padded = L_super * nm
    op.use_collab_lu && (FT !== Float64 || nm == 1) && L > 0 && L_super > 0 &&
        L_padded <= Int(Nz) &&
        _tm5_collab_supports(L_super, Nt, FT, backend) &&
        !(backend isa KernelAbstractions.CPU)
end

# Compute the super-grid size and the padded fine-span size for the
# vertical-aggregation path. For n_merge=1, L_super == L, L_padded == L.
@inline function _tm5_super_dims(op::TM5Convection, Nz::Integer)
    L = _effective_lmax_conv(op, Nz)
    nm = max(1, op.n_merge)
    L_super = nm > 1 ? fld(L, nm) : L
    L_padded = L_super * nm
    return L_super, L_padded, nm
end

# Preserve the existing contract: invalid Float32 GPU envelopes are errors;
# CPU and unsupported precisions retain a warning and the legacy solve.
# Float64 CUDA is opt-in within depth 1..`_TM5_COLLAB_MAX_DEPTH_FLOAT64_CUDA` (73)
# with n_merge=1. Unsupported Float64 configurations keep their previous
# fallback rather than changing the grid.
@noinline function _collab_contract_violation(op::TM5Convection, Nz::Integer,
                                              Nt::Integer, ::Type{FT}, backend) where {FT}
    L = _effective_lmax_conv(op, Nz)
    L_super, _, nm = _tm5_super_dims(op, Nz)
    if FT === Float64
        @warn "TM5Convection: Float64 collaborative LU requires CUDA, depth 1..$(_TM5_COLLAB_MAX_DEPTH_FLOAT64_CUDA) within Nz, n_merge=1, and positive Nt; got depth=$(L), Nz=$(Nz), n_merge=$(nm), Nt=$(Nt), backend=$(typeof(backend)). Using the per-thread path. Set use_collab_lu=false to silence this." maxlog=1
    elseif FT !== Float32
        @warn "TM5Convection: collaborative LU does not support FT=$(FT); using the per-thread path. Set use_collab_lu=false to silence this." maxlog=1
    elseif backend isa KernelAbstractions.CPU
        @warn "TM5Convection: use_collab_lu=true but backend is CPU — the collaborative kernel is GPU-only; using the per-thread path. Set use_collab_lu=false to silence this." maxlog=1
    else
        throw(ArgumentError(
            "TM5Convection: use_collab_lu=true requested but the collaborative LU path " *
            "cannot engage on this F32 GPU run — L_super=$(L_super) (lmax_conv=$(L), " *
            "n_merge=$(nm)), Nz=$(Nz), Nt=$(Nt). The effective depth must fit " *
            "1..$(_TM5_COLLAB_MAX_DEPTH_FLOAT32) " *
            "and the model's vertical extent, and Nt must be positive. Tracers are " *
            "processed in batches with no upper count limit. Choose a convection " *
            "depth/aggregation consistent with the forcing's active cloud depth. " *
            "To deliberately use the per-thread path, set use_collab_lu=false."))
    end
    return nothing
end

# Single choke point for the three `apply_convection!` dispatch sites: enforce
# the contract above, then return whether the collaborative path engages.
@inline function _should_use_collab(op::TM5Convection, Nz::Integer, Nt::Integer,
                                    ::Type{FT}, backend) where {FT}
    use = _use_collab_path(op, Nz, Nt, FT, backend)
    (op.use_collab_lu && !use) && _collab_contract_violation(op, Nz, Nt, FT, backend)
    return use
end

# Vertical-aggregation helper. Aggregates the active fine layers of
# `(q_raw, air_mass, entu, detu, entd, detd)` along the `k_dim`-th
# axis into a coarse super-grid of `L_super` layers (each spanning
# `n_merge` fine layers), runs the supplied solve closure on the
# super arrays, then disaggregates the new tracer values back to the
# fine layers proportionally to the pre-convection distribution.
# Redistribution preserves the new super-layer total up to rounding
# (uniform fallback when the old super-layer total was zero).
#
# `q_raw` has one more axis than the forcings (the tracer axis).
# `solve!` is a closure that takes the super-grid arrays
# `(q_super, m_super, entu_super, detu_super, entd_super, detd_super)`
# and mutates `q_super` in place.
@inline function _tm5_merge_aggregate_solve_disaggregate!(
        q_raw, air_mass, entu, detu, entd, detd,
        k_shift::Int, L_padded::Int, L_super::Int, n_merge::Int,
        k_dim::Int, solve!::F;
        Hp::Int = 0,
    ) where {F}
    # Build slicing tuples for the active vertical span.  The
    # `Nz`-axis is `k_dim` for q_raw and the same for forcings
    # (forcings have one fewer axis since they lack the tracer
    # dimension).  We slice the active sub-range, reshape it with
    # the extra `n_merge` axis inserted at position `k_dim`, then
    # sum over that axis to produce the super-grid array.
    #
    # `Hp` is the cubed-sphere panel halo half-width. When > 0,
    # the write-back is restricted to the interior `Hp+1:end-Hp`
    # cells in the horizontal dims so the halo region (which holds
    # neighbouring-panel tracer values for the next advection
    # step) is preserved bit-exact. Aggregation can still scan the
    # full padded view — the kernel only reads / writes interior
    # positions anyway, so halo-aggregated values are harmless
    # junk in `q_super`.
    q_active    = selectdim(q_raw,    k_dim, (k_shift + 1):(k_shift + L_padded))
    m_active    = selectdim(air_mass, k_dim, (k_shift + 1):(k_shift + L_padded))
    entu_active = selectdim(entu,     k_dim, (k_shift + 1):(k_shift + L_padded))
    detu_active = selectdim(detu,     k_dim, (k_shift + 1):(k_shift + L_padded))
    entd_active = selectdim(entd,     k_dim, (k_shift + 1):(k_shift + L_padded))
    detd_active = selectdim(detd,     k_dim, (k_shift + 1):(k_shift + L_padded))

    # Sum-aggregate along the `n_merge` super-internal axis.
    q_super    = _agg_sum(q_active,    k_dim, n_merge, L_super)
    m_super    = _agg_sum(m_active,    k_dim, n_merge, L_super)
    entu_super = _agg_sum(entu_active, k_dim, n_merge, L_super)
    detu_super = _agg_sum(detu_active, k_dim, n_merge, L_super)
    entd_super = _agg_sum(entd_active, k_dim, n_merge, L_super)
    detd_super = _agg_sum(detd_active, k_dim, n_merge, L_super)

    # Snapshot fine q for the proportional disaggregation.
    q_fine_save = copy(q_active)

    # Solve on the coarse grid (mutates `q_super`).
    solve!(q_super, m_super, entu_super, detu_super, entd_super, detd_super)

    # Disaggregate: new_fine = super_new * (fine_old / super_old)
    # with uniform fallback `1/n_merge` when super_old == 0.
    n_inv = Float32(1) / Float32(n_merge)
    _disaggregate!(q_active, q_super, q_fine_save, k_dim, n_merge, L_super, n_inv, Hp)
    return nothing
end

# Sum-aggregate `arr` (already sliced to the active vertical span)
# along the layer axis `k_dim`, producing an array of the same shape
# except the layer dim is `L_super` instead of `L_super * n_merge`.
# Uses `reshape + sum + dropdims` so the work runs on the underlying
# backend (CuArray broadcasts on GPU).
@inline function _agg_sum(arr, k_dim::Int, n_merge::Int, L_super::Int)
    # Materialise the view into a contiguous fresh array before
    # reshape. `selectdim` produces a SubArray whose `t`-stride is
    # the parent's `Nz`-stride (not the active-slice's contiguous
    # stride), so reshape can return a `ReshapedArray` wrapper whose
    # downstream `sum(...; dims=k_dim)` interaction with CUDA's
    # broadcast machinery is fragile. Copying once at the start
    # gives a contiguous `(N1, …, L_padded, Nt)` array; reshape on
    # *that* is a plain layout transform and the subsequent sum is
    # bit-stable.
    src = copy(arr)
    sh = size(src)
    # New shape: insert `n_merge` axis just before `k_dim` and replace
    # the `k_dim` size with `L_super`.
    new_shape = ntuple(d ->
        d <  k_dim ? sh[d] :
        d == k_dim ? n_merge :
        d == k_dim + 1 ? L_super :
                         sh[d - 1],
        length(sh) + 1)
    return dropdims(sum(reshape(src, new_shape); dims = k_dim); dims = k_dim)
end

# Proportional disaggregation. Writes `q_active` (the fine view of
# q_raw) with `q_super .* (fine_old / super_old)`, broadcasting the
# super-axis across `n_merge` fine slots. Uniform fallback when
# `super_old == 0`.
@inline function _disaggregate!(q_active, q_super, q_fine_save,
                                 k_dim::Int, n_merge::Int, L_super::Int,
                                 n_inv::Float32, Hp::Int)
    sh_fine = size(q_fine_save)
    # 5D view of the fine snapshot with the (n_merge, L_super) split.
    sh5_fine = ntuple(d ->
        d <  k_dim ? sh_fine[d] :
        d == k_dim ? n_merge :
        d == k_dim + 1 ? L_super :
                         sh_fine[d - 1],
        length(sh_fine) + 1)
    fine_old_5d = reshape(q_fine_save, sh5_fine)
    # super_old: sum over the n_merge axis.
    super_old   = dropdims(sum(fine_old_5d; dims = k_dim); dims = k_dim)
    # 5D broadcast view of super_old + q_super: insert a length-1
    # axis at `k_dim` so they broadcast across n_merge fine slots.
    sh5_super = ntuple(d ->
        d <  k_dim ? size(super_old, d) :
        d == k_dim ? 1 :
                     size(super_old, d - 1),
        length(sh_fine) + 1)
    super_old_5d = reshape(super_old, sh5_super)
    super_new_5d = reshape(q_super,   sh5_super)

    ratio_5d   = @. ifelse(super_old_5d == 0f0, n_inv,
                            fine_old_5d / super_old_5d)
    new_fine_5d = super_new_5d .* ratio_5d
    new_fine_fine = reshape(new_fine_5d, sh_fine)
    if Hp == 0
        # LL / RG: no halo, write the whole active slice.
        copyto!(q_active, new_fine_fine)
    else
        # CS panel: restrict the write-back to the interior cells
        # `Hp+1 : end-Hp` in the leading two dims. The halo region
        # of `q_active` must NOT be overwritten — it carries
        # neighbouring-panel tracer values that the next advection
        # step's halo exchange relies on; clobbering it with
        # convection-derived junk amplifies mass across substeps
        # via the halo loop.
        n1 = size(q_active, 1)
        n2 = size(q_active, 2)
        copyto!(view(q_active, (Hp + 1):(n1 - Hp), (Hp + 1):(n2 - Hp),
                     :, ntuple(_ -> :, ndims(q_active) - 3)...),
                view(new_fine_fine, (Hp + 1):(n1 - Hp), (Hp + 1):(n2 - Hp),
                     :, ntuple(_ -> :, ndims(q_active) - 3)...))
    end
    return nothing
end

"""
    apply_convection!(q_raw, air_mass, forcing::ConvectionForcing,
                       op::TM5Convection, dt, workspace::TM5Workspace,
                       grid::AtmosGrid) -> nothing

Array-level entry point — parallels the CMFMC contract at
`operators.jl:70-89`. Dispatches on grid
mesh type and launches the matching KA kernel from
`tm5_kernels.jl`. Single `synchronize(backend)`
at the end (TM5 matrix solve is unconditionally stable; no
sub-cycling).
"""
function apply_convection!(q_raw::AbstractArray{FT, 4},
                            air_mass::AbstractArray{FT, 3},
                            forcing::ConvectionForcing,
                            op::TM5Convection,
                            dt::Real,
                            workspace::TM5Workspace,
                            grid::AtmosGrid{<:LatLonMesh}) where {FT}
    _assert_tm5_forcing(forcing)
    tm5 = forcing.tm5_fields
    cell_areas_y = _tm5_require_cell_metrics(workspace, "LatLon")
    Nx, Ny, Nz, Nt = size(q_raw)
    N_total = Nx * Ny
    backend = get_backend(q_raw)
    dt_ft   = FT(dt)
    if _should_use_collab(op, Nz, Nt, FT, backend)
        L_super, L_padded, nm = _tm5_super_dims(op, Nz)
        k_shift = Nz - L_padded
        kernel = _tm5_column_collab_kernel!(backend, _TM5_COLLAB_WG_SIZE)
        if nm == 1
            # No vertical aggregation — run the existing kernel
            # directly on the fine grid.
            kernel(q_raw, air_mass,
                   tm5.entu, tm5.detu, tm5.entd, tm5.detd,
                   cell_areas_y,
                   Int(Nx), Int(Nz), Int(Nt), FT(dt), Val(L_super);
                   ndrange = _TM5_COLLAB_WG_SIZE * N_total,
                   workgroupsize = _TM5_COLLAB_WG_SIZE)
        else
            # Aggregate → coarse solve → disaggregate.  The closure
            # captures the kernel-launch parameters; `_tm5_merge_…`
            # builds the super arrays, invokes it, and redistributes
            # the new tracer mass back to the fine layers.
            _tm5_merge_aggregate_solve_disaggregate!(
                q_raw, air_mass,
                tm5.entu, tm5.detu, tm5.entd, tm5.detd,
                k_shift, L_padded, L_super, nm, 3,
                function (qS, mS, eU, dU, eD, dD)
                    kernel(qS, mS, eU, dU, eD, dD, cell_areas_y,
                           Int(Nx), Int(L_super), Int(Nt),
                           FT(dt), Val(L_super);
                           ndrange = _TM5_COLLAB_WG_SIZE * N_total,
                           workgroupsize = _TM5_COLLAB_WG_SIZE)
                end)
        end
    else
        kernel = _tm5_column_kernel!(backend)
        # Tile loop — KA stream ordering serializes panels safely
        # because the workspace is shared. `synchronize(backend)`
        # after the loop, not per launch.
        _ensure_tm5_scratch!(workspace)
        B = size(workspace.conv1, 3)
        for tile_off in 0:B:(N_total - 1)
            n = min(B, N_total - tile_off)
            kernel(q_raw, air_mass,
                   tm5.entu, tm5.detu, tm5.entd, tm5.detd,
                   cell_areas_y,
                   workspace.conv1, workspace.pivots, workspace.cloud_dims,
                   workspace.f_scratch,
                   workspace.amu_scratch, workspace.amd_scratch,
                   Int(tile_off), Int(Nx), dt_ft;
                   ndrange = n)
        end
    end
    synchronize(backend)
    return nothing
end

function apply_convection!(q_raw::AbstractArray{FT, 3},
                            air_mass::AbstractArray{FT, 2},
                            forcing::ConvectionForcing,
                            op::TM5Convection,
                            dt::Real,
                            workspace::TM5Workspace,
                            grid::AtmosGrid{<:ReducedGaussianMesh}) where {FT}
    _assert_tm5_forcing(forcing)
    tm5     = forcing.tm5_fields
    cell_areas = _tm5_require_cell_metrics(workspace, "face-indexed TM5Convection")
    N_total, Nz, Nt = size(q_raw)
    backend = get_backend(q_raw)
    dt_ft   = FT(dt)
    if _should_use_collab(op, Nz, Nt, FT, backend)
        L_super, L_padded, nm = _tm5_super_dims(op, Nz)
        k_shift = Nz - L_padded
        kernel = _tm5_faceindexed_column_collab_kernel!(backend, _TM5_COLLAB_WG_SIZE)
        if nm == 1
            kernel(q_raw, air_mass,
                   tm5.entu, tm5.detu, tm5.entd, tm5.detd,
                   cell_areas,
                   Int(Nz), Int(Nt), FT(dt), Val(L_super);
                   ndrange = _TM5_COLLAB_WG_SIZE * N_total,
                   workgroupsize = _TM5_COLLAB_WG_SIZE)
        else
            # RG: q is (ncells, Nz, Nt), forcings (ncells, Nz),
            # air_mass (ncells, Nz). Layer axis is k_dim = 2.
            _tm5_merge_aggregate_solve_disaggregate!(
                q_raw, air_mass,
                tm5.entu, tm5.detu, tm5.entd, tm5.detd,
                k_shift, L_padded, L_super, nm, 2,
                function (qS, mS, eU, dU, eD, dD)
                    kernel(qS, mS, eU, dU, eD, dD, cell_areas,
                           Int(L_super), Int(Nt),
                           FT(dt), Val(L_super);
                           ndrange = _TM5_COLLAB_WG_SIZE * N_total,
                           workgroupsize = _TM5_COLLAB_WG_SIZE)
                end)
        end
    else
        kernel = _tm5_faceindexed_column_kernel!(backend)
        _ensure_tm5_scratch!(workspace)
        B = size(workspace.conv1, 3)
        for tile_off in 0:B:(N_total - 1)
            n = min(B, N_total - tile_off)
            kernel(q_raw, air_mass,
                   tm5.entu, tm5.detu, tm5.entd, tm5.detd,
                   cell_areas,
                   workspace.conv1, workspace.pivots, workspace.cloud_dims,
                   workspace.f_scratch,
                   workspace.amu_scratch, workspace.amd_scratch,
                   Int(tile_off), dt_ft;
                   ndrange = n)
        end
    end
    synchronize(backend)
    return nothing
end

function apply_convection!(q_raw::NTuple{6, <:AbstractArray{FT, 4}},
                            air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                            forcing::ConvectionForcing,
                            op::TM5Convection,
                            dt::Real,
                            workspace::TM5Workspace,
                            grid::AtmosGrid{<:CubedSphereMesh}) where {FT}
    _assert_tm5_forcing(forcing)
    tm5     = forcing.tm5_fields
    cell_areas = _tm5_require_cell_metrics(workspace, "cubed-sphere TM5Convection")
    mesh    = grid.horizontal
    Nc      = mesh.Nc
    Hp      = mesh.Hp
    N_total = Nc * Nc
    backend = get_backend(q_raw[1])
    dt_ft   = FT(dt)
    # Per-panel Nz / Nt: panels share a vertical resolution by
    # construction, so we read them off panel 1.
    Nz = size(air_mass[1], 3)
    Nt = size(q_raw[1], 4)
    if _should_use_collab(op, Nz, Nt, FT, backend)
        L_super, L_padded, nm = _tm5_super_dims(op, Nz)
        k_shift = Nz - L_padded
        kernel = _tm5_cs_panel_column_collab_kernel!(backend, _TM5_COLLAB_WG_SIZE)
        # The workspace is no longer needed by the collab kernel
        # (its scratch lives in `@localmem`), but the panel loop
        # remains so each panel's halo offset and forcing arrays
        # are passed individually.
        for p in 1:6
            if nm == 1
                kernel(q_raw[p], air_mass[p],
                       tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p],
                       cell_areas[p],
                       Int(Hp), Int(Nc), Int(Nz), Int(Nt), FT(dt), Val(L_super);
                       ndrange = _TM5_COLLAB_WG_SIZE * N_total,
                       workgroupsize = _TM5_COLLAB_WG_SIZE)
            else
                # CS panel: q_raw[p] is (Nc+2Hp, Nc+2Hp, Nz, Nt), air_mass[p]
                # is (Nc+2Hp, Nc+2Hp, Nz), forcings are (Nc, Nc, Nz).
                # All share `k_dim = 3` for the vertical layer axis.
                # Pass `Hp` so the disaggregation write-back stays
                # inside the interior and the halo region is left
                # bit-exact (any disturbance there propagates to
                # neighbouring panels via the next advection
                # step's halo exchange).
                _tm5_merge_aggregate_solve_disaggregate!(
                    q_raw[p], air_mass[p],
                    tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p],
                    k_shift, L_padded, L_super, nm, 3,
                    function (qS, mS, eU, dU, eD, dD)
                        kernel(qS, mS, eU, dU, eD, dD, cell_areas[p],
                               Int(Hp), Int(Nc), Int(L_super), Int(Nt),
                               FT(dt), Val(L_super);
                               ndrange = _TM5_COLLAB_WG_SIZE * N_total,
                               workgroupsize = _TM5_COLLAB_WG_SIZE)
                    end;
                    Hp = Int(Hp))
            end
        end
    else
        kernel = _tm5_cs_panel_column_kernel!(backend)
        _ensure_tm5_scratch!(workspace)
        B = size(workspace.conv1, 3)
        # The workspace is shared across panels; KA stream ordering
        # makes that safe (panel n+1 can't start until panel n's
        # writes are visible). The `for p` loop sits *outside* the
        # tile loop so the workspace is reused per panel rather than
        # cloned six times.
        for p in 1:6
            for tile_off in 0:B:(N_total - 1)
                n = min(B, N_total - tile_off)
                kernel(q_raw[p], air_mass[p],
                       tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p],
                       cell_areas[p],
                       workspace.conv1, workspace.pivots,
                       workspace.cloud_dims,
                       workspace.f_scratch,
                       workspace.amu_scratch, workspace.amd_scratch,
                       Int(Hp), Int(tile_off), Int(Nc), dt_ft;
                       ndrange = n)
            end
        end
    end
    synchronize(backend)
    return nothing
end

# Clear error message when `forcing.tm5_fields === nothing` — this
# is the only "missing input" case because the validator
# in DrivenSimulation rejects it at window-load time.  Direct
# callers (e.g. tests that build forcing by hand) go through this
# guard.
function _assert_tm5_forcing(forcing::ConvectionForcing)
    forcing.tm5_fields === nothing &&
        throw(ArgumentError(
            "TM5Convection requires `forcing.tm5_fields` " *
            "(NamedTuple with :entu, :detu, :entd, :detd) to be populated. " *
            "Use `with_convection_forcing(model, ConvectionForcing(nothing, nothing, tm5_fields))` " *
            "or ensure the driver populates `window.convection.tm5_fields`."))
    return nothing
end

function _tm5_require_cell_metrics(workspace::TM5Workspace, context::AbstractString)
    metrics = workspace.cell_metrics
    metrics === nothing && throw(ArgumentError(
        "$context requires `workspace.cell_metrics` to carry cell areas " *
        "so TM5 can convert state air mass from kg/cell to kg/m². " *
        "Build the model with `with_convection` or construct " *
        "`TM5Workspace(air_mass; cell_metrics=...)`."))
    return metrics
end

# =========================================================================
# State-level entry: apply!
# =========================================================================

"""
    apply!(state::CellState, forcing::ConvectionForcing, grid::AtmosGrid,
           op::TM5Convection, dt::Real; workspace) -> state

State-level delegate — matches the CMFMC contract at
`CMFMCConvection.jl:296-316`.
Dispatches on grid mesh type plus the `Raw` parameter of
`CellState{B, A, Raw}`.
"""
function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:LatLonMesh},
                op::TM5Convection,
                dt::Real;
                workspace::TM5Workspace) where {B, A, Raw <: AbstractArray{<:Any, 4}, Names}
    apply_convection!(state.tracers_raw, state.air_mass, forcing,
                        op, dt, workspace, grid)
    return state
end

function apply!(state::CellState{B, A, Raw, Names},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:ReducedGaussianMesh},
                op::TM5Convection,
                dt::Real;
                workspace::TM5Workspace) where {B, A, Raw <: AbstractArray{<:Any, 3}, Names}
    apply_convection!(state.tracers_raw, state.air_mass, forcing,
                        op, dt, workspace, grid)
    return state
end

function apply!(state::CubedSphereState{B},
                forcing::ConvectionForcing,
                grid::AtmosGrid{<:CubedSphereMesh},
                op::TM5Convection,
                dt::Real;
                workspace::TM5Workspace) where {B}
    apply_convection!(state.tracers_raw, state.air_mass, forcing,
                        op, dt, workspace, grid)
    return state
end
