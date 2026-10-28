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
`n_merge` optionally aggregates adjacent active layers before the solve (see
Level merging below).
Float32 GPU matrices support effective depths 1..85. Float64 CUDA supports
unmerged (`n_merge=1`) depths 1..73, using Float64 shared arrays and arithmetic.
CPU and unsupported Float64 configurations retain the legacy solver with a
warning. Tracers use batches of six; total tracer count does not set this depth
limit. Defaults remain the full-column legacy solve. CS adjoint footprints
still require the full-column, unmerged legacy variant.

The forcing arrays `(entu, detu, entd, detd)` arrive via
`TransportModel.convection_forcing.tm5_fields`, populated at the first
substep of each window by `DrivenSimulation._refresh_forcing!` from
`sim.window.convection`.

# Level merging (`n_merge > 1`)

An AtmosTransport approximation; TM5 itself merges levels once, in its met
preprocessing, and never reconstructs a fine profile. Each group `s` of
`n_merge` adjacent fine layers `k` becomes one super-layer with tracer mass
`R_s = Σ_k r_k`, air mass `M_s = Σ_k m_k`, and summed forcings. After the
backward-Euler solve on the super grid gives `R'_s`, each fine layer receives a
share of the super-layer's change:

    r'_k = r_k + (R'_s - R_s) * w_k,   w_k = |r_k| / Σ_{j∈s} |r_j|

with `w_k = m_k / M_s` if `Σ_{j∈s} |r_j| = 0`, and `w_k = 1 / n_merge` if
`M_s = 0` as well.

The weights lie in [0, 1] and sum to one. The split conserves the column
total, keeps a uniform mixing ratio, leaves columns and layers without
convection unchanged bit for bit, and changes no cell by more than
`|R'_s - R_s|`. If a super-layer's fine values share one sign, it equals the
proportional split `r'_k = R'_s (r_k / R_s)` and is evaluated in that form, so
non-negative tracers stay non-negative in floating point as well. A column
mass ledger over the whole merged operation returns the split's rounding to
the column (see the `ledger_carry` of [`TM5Workspace`](@ref)). Only
super-layers holding values of both signs can increase the column's `Σ|r|`:
by at most `2 Σ_{s mixed} |R'_s - R_s|` in total. The split is not linear in
the tracer; merged solves have no adjoint (CS footprints require
`n_merge = 1`).

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
    # Each super-layer's tracer-mass change is split over its fine layers
    # with weights |r_k|/Σ|r_j| (air-mass weights if all are zero, equal
    # weights if the air mass is zero too); this
    # conserves mass up to rounding but changes vertical mixing.
    n_merge :: Int

    # Validate every construction path, including Adapt and explicit type calls.
    function TM5Convection{FT}(tile_workspace_gib::FT,
                                use_collab_lu::Bool,
                                lmax_conv::Int,
                                n_merge::Int) where {FT}
        lmax_conv >= 0 || throw(ArgumentError(
            "TM5Convection: `lmax_conv` must be >= 0 (got $(lmax_conv))"))
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
        @warn "TM5Convection: Float64 collaborative LU requires CUDA, depth 1..$(_TM5_COLLAB_MAX_DEPTH_FLOAT64_CUDA) within Nz, n_merge=1, and positive Nt; got depth=$(L), Nz=$(Nz), n_merge=$(nm), Nt=$(Nt), backend=$(typeof(backend)). Using the per-thread path with the full column and no level merging; requested lmax_conv/n_merge approximations are dropped. Set use_collab_lu=false to silence this." maxlog=1
    elseif FT !== Float32
        @warn "TM5Convection: collaborative LU does not support FT=$(FT); using the per-thread path with the full column and no level merging; requested lmax_conv/n_merge approximations are dropped. Set use_collab_lu=false to silence this." maxlog=1
    elseif backend isa KernelAbstractions.CPU
        @warn "TM5Convection: use_collab_lu=true but backend is CPU — the collaborative kernel is GPU-only; using the per-thread path with the full column and no level merging; requested lmax_conv/n_merge approximations are dropped. Set use_collab_lu=false to silence this." maxlog=1
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
# super arrays, then splits each super-layer's tracer-mass change back
# to its fine layers (`_split_merged_increment!`).
#
# `q_raw` has one more axis than the forcings (the tracer axis).
# `solve!` is a closure that takes the super-grid arrays
# `(q_super, m_super, entu_super, detu_super, entd_super, detd_super)`
# and mutates `q_super` in place.
#
# `ledger_carry` (the workspace carry, one per column and tracer) adds a
# column mass ledger over the whole merged operation: the split rounds every
# fine layer outside the coarse solve's own ledger, while in exact arithmetic
# aggregation, solve and split leave the fine column total unchanged. The
# operator always passes it; `nothing` leaves the split as computed.
@inline function _tm5_merge_aggregate_solve_disaggregate!(
        q_raw, air_mass, entu, detu, entd, detd,
        k_shift::Int, L_padded::Int, L_super::Int, n_merge::Int,
        k_dim::Int, solve!::F;
        Hp::Int = 0, ledger_carry = nothing,
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

    # Sum-aggregate along the `n_merge` super-internal axis. The contiguous
    # fine tracer and air-mass copies (see `_agg_sum`) are kept for the split,
    # as is the super-layer tracer mass before the solve.
    q_fine_old  = copy(q_active)
    m_fine      = copy(m_active)
    q_super     = _sum_layers(q_fine_old, k_dim, n_merge, L_super)
    m_super     = _sum_layers(m_fine,     k_dim, n_merge, L_super)
    entu_super  = _agg_sum(entu_active, k_dim, n_merge, L_super)
    detu_super  = _agg_sum(detu_active, k_dim, n_merge, L_super)
    entd_super  = _agg_sum(entd_active, k_dim, n_merge, L_super)
    detd_super  = _agg_sum(detd_active, k_dim, n_merge, L_super)
    q_super_old = copy(q_super)

    # Solve on the coarse grid (mutates `q_super`; reads `m_super`).
    solve!(q_super, m_super, entu_super, detu_super, entd_super, detd_super)

    _split_merged_increment!(q_active, q_super, q_super_old, q_fine_old,
                             m_fine, m_super, k_dim, n_merge, L_super, Hp)
    ledger_carry === nothing ||
        _tm5_merged_column_ledger!(q_active, q_fine_old, ledger_carry, Hp)
    return nothing
end

# Fine-grid column ledger of the merged operation: compensated sums of each
# interior column before (`q_fine_old`) and after (`q_active`), residual and
# carry into the largest fine layer (`_tm5_restore_column_mass!`). As in the
# unmerged solve, only the changed part of the column takes part: from its
# first changed layer down. Layers above (passed through) stay bit-exact and
# do not enter the leak bound, and an unchanged column (no convection) is not
# ledgered and keeps its carry.
function _tm5_merged_column_ledger!(q_active::AbstractArray{<:Any, 4}, q_fine_old, carry, Hp::Int)
    N1, N2, _, Nt = size(q_active)
    launch!(_tm5_merged_ledger_kernel!, get_backend(q_fine_old), (8, 8, 1),
            (N1 - 2Hp, N2 - 2Hp, Nt), q_active, carry, q_fine_old, Hp; sync = false)
    return nothing
end

function _tm5_merged_column_ledger!(q_active::AbstractArray{<:Any, 3}, q_fine_old, carry, Hp::Int)
    launch!(_tm5_merged_ledger_rg_kernel!, get_backend(q_fine_old), (64, 1),
            (size(q_active, 1), size(q_active, 3)), q_active, carry, q_fine_old; sync = false)
    return nothing
end

@inline function _tm5_merged_column_ledger!(column, old, carry, cell...)
    FT = eltype(column)
    L = length(column)
    lo = L + 1
    @inbounds for k in 1:L
        if !isequal(column[k], old[k])
            lo = k
            break
        end
    end
    lo > L && return nothing
    before = _neumaier_sum(k -> @inbounds(old[k]), lo:L, FT)
    c = _tm5_restore_column_mass!(column, lo, L, 1, before, _load_ledger_carry(carry, FT, cell...))
    _store_ledger_carry!(carry, c, cell...)
    return nothing
end

@kernel function _tm5_merged_ledger_kernel!(q_active, carry, q_fine_old, Hp::Int)
    i, j, t = @index(Global, NTuple)
    _tm5_merged_column_ledger!(view(q_active, i + Hp, j + Hp, :, t),
                               view(q_fine_old, i + Hp, j + Hp, :, t), carry, i, j, t)
end

@kernel function _tm5_merged_ledger_rg_kernel!(q_active, carry, q_fine_old)
    c, t = @index(Global, NTuple)
    _tm5_merged_column_ledger!(view(q_active, c, :, t), view(q_fine_old, c, :, t), carry, c, t)
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
    return _sum_layers(copy(arr), k_dim, n_merge, L_super)
end

# Super-layer sums of a contiguous array whose layer axis `k_dim` holds
# `n_merge * L_super` fine layers.
@inline _sum_layers(src, k_dim::Int, n_merge::Int, L_super::Int) =
    dropdims(sum(reshape(src, _split_layer_axis(size(src), k_dim, n_merge, L_super));
                 dims = k_dim); dims = k_dim)

# Shape with the layer axis `k_dim` (length `n_merge * L_super`) split into
# `(n_merge, L_super)`: the fine layers of one super-layer share an index
# along the new axis `k_dim + 1`.
@inline _split_layer_axis(sh::Tuple, k_dim::Int, n_merge::Int, L_super::Int) =
    ntuple(d -> d <  k_dim     ? sh[d] :
                d == k_dim     ? n_merge :
                d == k_dim + 1 ? L_super :
                                 sh[d - 1],
           length(sh) + 1)

# Increment split (OF-107). Each fine layer k of super-layer s receives a share
# of the super-layer's tracer-mass change ΔR_s = R'_s - R_s from the coarse solve:
#
#     r'_k = r_k + ΔR_s * w_k,   w_k = |r_k| / Σ_{j∈s} |r_j|
#
# (w_k = m_k / M_s if that sum is 0; 1 / n_merge if M_s is 0 as well).
#
# The weights lie in [0, 1] and sum to one, so the split conserves the
# super-layer total exactly in exact arithmetic, keeps a uniform mixing
# ratio, leaves layers the solve did not change (ΔR_s == 0) bit for bit, and
# changes no cell by more than |ΔR_s|. For a super-layer whose fine values
# share one sign it equals the proportional split r'_k = R'_s (r_k / R_s),
# which is how it is evaluated there: the weight r_k / R_s lies in [0, 1] and
# the product keeps its sign in floating point, while the increment form can
# round a nearly emptied fine layer to a tiny negative. (The form
# r_k (R'_s / R_s) is not used: R'_s / R_s overflows when a nearly empty
# super-layer receives tracer, and 0 * Inf is NaN.) Unlike the old proportional
# split of every super-layer, no 1/R_s factor amplifies near-cancelling values
# of both signs. See the TM5Convection docstring.
# `pos` and `neg` are the super-layer sums of the positive and negative fine
# values; one of them is zero exactly when the fine values share one sign.
@inline function _merged_split_value(r, R_old, R_new, pos, neg, m, m_sum, n_inv)
    R_new == R_old && return r
    (iszero(pos) || iszero(neg)) && !iszero(R_old) && return R_new * (r / R_old)
    abs_sum = pos - neg
    w = abs_sum > 0 ? abs(r) / abs_sum :
        m_sum   > 0 ? m / m_sum : n_inv
    return r + (R_new - R_old) * w
end

# Shape `sh` with a length-1 axis inserted at `k_dim`, so super-layer arrays
# broadcast across the fine layers of the `_split_layer_axis` reshape.
@inline _insert_singleton_axis(sh::Tuple, k_dim::Int) =
    ntuple(d -> d < k_dim ? sh[d] : d == k_dim ? 1 : sh[d - 1], length(sh) + 1)

# Broadcasts over `(…, n_merge, L_super, …)` reshapes run on the arrays' backend.
@inline function _split_merged_increment!(q_active, q_super_new, q_super_old,
                                          q_fine_old, m_fine, m_super,
                                          k_dim::Int, n_merge::Int, L_super::Int,
                                          Hp::Int)
    sh_fine = size(q_fine_old)
    r_old = reshape(q_fine_old, _split_layer_axis(sh_fine, k_dim, n_merge, L_super))
    m_old = reshape(m_fine, _split_layer_axis(size(m_fine), k_dim, n_merge, L_super))
    # Super-layer quantities keep a length-1 axis at `k_dim`; the air masses
    # lack the trailing tracer axis.
    sh_super = _insert_singleton_axis(size(q_super_old), k_dim)
    R_old   = reshape(q_super_old, sh_super)
    R_new   = reshape(q_super_new, sh_super)
    m_sum   = reshape(m_super, _insert_singleton_axis(size(m_super), k_dim))
    pos     = sum(x -> max(x, zero(x)), r_old; dims = k_dim)
    neg     = sum(x -> min(x, zero(x)), r_old; dims = k_dim)
    n_inv   = inv(eltype(q_fine_old)(n_merge))
    r_new = _merged_split_value.(r_old, R_old, R_new, pos, neg, m_old, m_sum, n_inv)
    new_fine = reshape(r_new, sh_fine)
    if Hp == 0
        # LL / RG: no halo, write the whole active slice.
        copyto!(q_active, new_fine)
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
                view(new_fine, (Hp + 1):(n1 - Hp), (Hp + 1):(n2 - Hp),
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
    carry   = _tm5_ledger_carry!(workspace, (Nx, Ny, Nt))
    if _should_use_collab(op, Nz, Nt, FT, backend)
        L_super, L_padded, nm = _tm5_super_dims(op, Nz)
        k_shift = Nz - L_padded
        kernel = _tm5_column_collab_kernel!(backend, _TM5_COLLAB_WG_SIZE)
        if nm == 1
            # No vertical aggregation — run the existing kernel
            # directly on the fine grid.
            kernel(q_raw, carry, air_mass,
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
            # The ledger carry covers the whole merged operation (the coarse
            # solve keeps its own ledger without a carry).
            _tm5_merge_aggregate_solve_disaggregate!(
                q_raw, air_mass,
                tm5.entu, tm5.detu, tm5.entd, tm5.detd,
                k_shift, L_padded, L_super, nm, 3,
                function (qS, mS, eU, dU, eD, dD)
                    kernel(qS, nothing, mS, eU, dU, eD, dD, cell_areas_y,
                           Int(Nx), Int(L_super), Int(Nt),
                           FT(dt), Val(L_super);
                           ndrange = _TM5_COLLAB_WG_SIZE * N_total,
                           workgroupsize = _TM5_COLLAB_WG_SIZE)
                end; ledger_carry = carry)
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
            kernel(q_raw, carry, air_mass,
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
    carry   = _tm5_ledger_carry!(workspace, (N_total, Nt))
    if _should_use_collab(op, Nz, Nt, FT, backend)
        L_super, L_padded, nm = _tm5_super_dims(op, Nz)
        k_shift = Nz - L_padded
        kernel = _tm5_faceindexed_column_collab_kernel!(backend, _TM5_COLLAB_WG_SIZE)
        if nm == 1
            kernel(q_raw, carry, air_mass,
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
                    kernel(qS, nothing, mS, eU, dU, eD, dD, cell_areas,
                           Int(L_super), Int(Nt),
                           FT(dt), Val(L_super);
                           ndrange = _TM5_COLLAB_WG_SIZE * N_total,
                           workgroupsize = _TM5_COLLAB_WG_SIZE)
                end; ledger_carry = carry)
        end
    else
        kernel = _tm5_faceindexed_column_kernel!(backend)
        _ensure_tm5_scratch!(workspace)
        B = size(workspace.conv1, 3)
        for tile_off in 0:B:(N_total - 1)
            n = min(B, N_total - tile_off)
            kernel(q_raw, carry, air_mass,
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
    carry = _tm5_ledger_carry!(workspace, (Nc, Nc, Nt))
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
                kernel(q_raw[p], carry[p], air_mass[p],
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
                        kernel(qS, nothing, mS, eU, dU, eD, dD, cell_areas[p],
                               Int(Hp), Int(Nc), Int(L_super), Int(Nt),
                               FT(dt), Val(L_super);
                               ndrange = _TM5_COLLAB_WG_SIZE * N_total,
                               workgroupsize = _TM5_COLLAB_WG_SIZE)
                    end;
                    Hp = Int(Hp), ledger_carry = carry[p])
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
                kernel(q_raw[p], carry[p], air_mass[p],
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
