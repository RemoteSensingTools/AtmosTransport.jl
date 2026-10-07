# ---------------------------------------------------------------------------
# TM5 column solver.
#
# Backend-agnostic Julia implementation of the Tiedtke 1989 mass-flux
# convection matrix builder + partial-pivot LU solve, transcribed once
# from the TM5-4DVAR reference at deps/tm5/base/src/tm5_conv.F90:32–341
# into AtmosTransport orientation (k=1=TOA, k=Nz=surface — the
# preprocessor writes forcings in this orientation, so
# the solver has zero runtime orientation logic).
#
# Algorithm overview (matches the Fortran line-by-line):
#
#   1. Diagnose cloud dims (icltop, iclbas, icllfs) from `detu` / `entd`.
#   2. If icltop > Nz (no detu-positive layer), return immediately:
#      this column has identity convection.
#   3. Build the flux-divergence matrix `conv1 = I - dt·D`. Rows above
#      the effective cloud window are explicit identity rows and the
#      lower-left quadrant below that window is zero.
#   4. Factorize the active lower-right block with partial pivoting. Store the
#      permutation vector in `pivots`; the adjoint replays this
#      factorization with trans='T', so do NOT fuse the permutation
#      into the solved coefficients.
#   5. Back-substitute each tracer's active vertical profile.
#
# General downdraft columns use dense LU in the active cloud window.
# Without a diagnosed downdraft, the matrix is upper-Hessenberg and needs
# only quadratic factorization work. Both paths retain the same factor and
# pivot representation for the forward and adjoint triangular solves.
# ---------------------------------------------------------------------------

"""
    _tm5_diagnose_cloud_dims(detu, entd, Nz) -> (icltop, iclbas, icllfs)

Diagnose cloud top, cloud base, and level of free sinking from the
updraft detrainment and downdraft entrainment profiles. All
indices in AtmosTransport orientation (k=1=TOA, k=Nz=surface).

- `icltop` = smallest k with `detu[k] > 0` (highest-altitude active
  updraft detrainment level; set to `Nz + 1` if no convection).
- `iclbas` = largest k with `detu[k] > 0` (lowest-altitude active
  updraft detrainment level; set to 0 if no convection).
- `icllfs` = smallest k with `entd[k] > 0` (highest-altitude
  downdraft entrainment level; set to `Nz + 1` if no downdraft).

Maps to TM5-4DVAR's `ConvCloudDim`
([`tmphys_convec.F90:39–132`](../../../deps/tm5/base/src/tmphys_convec.F90)).
TM5's "cloud top is HIGHEST level with detu > 0 searching top-down"
corresponds in AtmosTransport to "smallest k with detu[k] > 0
searching from k=1". The invariant `iclbas >= icltop` (Fortran
`iclbas > icltop`) becomes `iclbas >= icltop` in AtmosTransport
indexing.

`entu` is not needed for cloud-dim diagnosis — TM5 upstream uses
detu for the cloud top/base and entd for the LFS.
"""
@inline function _tm5_diagnose_cloud_dims(detu_col, entd_col,
                                           Nz::Integer)
    icltop = Nz + 1   # sentinel "no convection"
    iclbas = 0
    icllfs = Nz + 1

    @inbounds for k in 1:Nz
        if detu_col[k] > 0
            icltop == Nz + 1 && (icltop = k)
            iclbas = k
        end
        if entd_col[k] > 0 && icllfs == Nz + 1
            icllfs = k
        end
    end
    return (icltop, iclbas, icllfs)
end

"""
    _tm5_build_conv1!(conv1, entu_col, detu_col, entd_col, detd_col,
                      m_col, icltop, icllfs, dt, Nz) -> nothing

Build the transport matrix `conv1 = I - dt · D` in place. On entry,
`conv1` may hold arbitrary data; this function zeros it and writes
the identity + active-window entries. Indices run 1..Nz in
AtmosTransport orientation.

Follows [`tm5_conv.F90:32–191`](../../../deps/tm5/base/src/tm5_conv.F90)
block by block. Intermediate `fu` (updraft subplane) and `f`
(downdraft + subsidence) arrays are stack-allocated via `MVector`
from StaticArrays… or are they? For an arbitrary `Nz` we can't.
This uses plain `Array`s sized `(Nz+1, Nz)` (TM5 Fortran's
`f(0:lmx, 1:lmx)` pattern, shifted by +1); these can either be
batch-allocated inside the workspace or kept per-column (the matrices
are small — `Nz × Nz` floats ≤ 72² × 4 bytes ≈ 20 KiB per column).

Reindex rules (AtmosTransport ↔ TM5; applied inline):

- TM5 `do k = 1, li` (surface → cloud top) becomes AtmosTransport
  `for k_atm in Nz:-1:icltop_atm` (surface → cloud top, k decreasing).
- TM5 `do k = ld, 2, -1` (level of free sinking → 2) becomes
  AtmosTransport `for k_atm in icllfs_atm:(Nz-1)` (level of free
  sinking → second-from-bottom, k increasing).
- TM5 `do k = 1, lmx-1` (surface → TOA-1) becomes AtmosTransport
  `for k_atm in Nz:-1:2`.

Mass-flux boundary: TM5 `amu(0) = 0` ("flux below the bottom
boundary") maps to AtmosTransport `amu_atIFC[Nz+1] = 0` (flux at
the surface interface, below the bottom-most air-mass layer).
"""
function _tm5_build_conv1!(conv1::AbstractMatrix{FT},
                           entu_col::AbstractVector{FT},
                           detu_col::AbstractVector{FT},
                           entd_col::AbstractVector{FT},
                           detd_col::AbstractVector{FT},
                           m_col::AbstractVector{FT},
                           icltop::Integer, icllfs::Integer,
                           dt::FT, Nz::Integer;
                           cell_area::FT = one(FT),
                           f::AbstractMatrix{FT}   = zeros(FT, Nz + 1, Nz),
                           amu::AbstractVector{FT} = zeros(FT, Nz + 1),
                           amd::AbstractVector{FT} = zeros(FT, Nz + 1),
                           ) where {FT}
    # Working storage: `f` holds the merged updraft + downdraft
    # contribution. Rather than carrying a separate `fu` buffer to
    # mirror TM5's two-array Fortran layout, the updraft is folded
    # directly into `f` because the two passes write disjoint index
    # ranges (updraft at `kk_atm ≥ k_atm`, downdraft at
    # `kk_atm < k_atm`). `amu`/`amd` are length-(Nz+1).
    #
    # Shape contract: callers pass either `(Nz+1, Nz)` (default
    # standalone path) or `(Nz, Nz)` aliased to `conv1`. Production
    # uses the alias — see `convection_workspace.jl`'s
    # `f_scratch === conv1` aliasing. Both shapes are valid because
    # the loop body never reads `f[Nz+1, ...]` (the surface row is
    # supplied as a literal zero in the propagation step at
    # `k_atm == Nz`, and the final assembly uses the same guard).
    fill!(amu, zero(FT))
    fill!(amd, zero(FT))

    # Index convention in AtmosTransport orientation (k=1=TOA,
    # k=Nz=surface):
    # - Layers use k ∈ 1:Nz with k_atm = Nz+1-k_tm5.
    # - Interface-valued quantities (amu, amd) use a length-(Nz+1)
    #   vector where amu[k_atm] is the mass flux at the top of layer
    #   k_atm, and amu[Nz+1] is the bottom boundary (TM5 amu(0) = 0).
    # - `f[k_atm, kk_atm]` is TM5 `f(k, kk)` (or `fu(k, kk)` for
    #   updraft entries) with `k_atm = Nz+1-k_tm5`. The implicit
    #   "TM5 row 0" / surface row at index Nz+1 is treated as zero;
    #   no shape needs to materialize it.
    icltop_eff = min(Int(icllfs), max(Int(icltop), 2) - 1)
    k_lo = max(icltop_eff, 1)

    # `f` may alias `conv1` and can contain stale values from the previous
    # column. Only the active lower-right block can be read by the final
    # assembly and active LU; clear that block instead of the full matrix.
    @inbounds for kk_atm in k_lo:Nz
        for k_atm in k_lo:Nz
            f[k_atm, kk_atm] = zero(FT)
        end
    end

    # --- Updraft pass (AtmosTransport surface → cloud top) ------
    # For each layer k_atm iterated surface-up, compute amu[k_atm],
    # zxi, and propagate the updraft contribution. Writes go
    # directly into `f` at column indices `kk_atm ≥ k_atm` (upper
    # triangle plus diagonal); the downdraft pass below writes
    # disjointly at `kk_atm < k_atm`, so the combine pass no longer
    # needs `f += fu`.
    @inbounds for k_atm in Nz:-1:icltop
        # TM5:  amu(k)   = amu(k-1) + entu(k) - detu(k)
        # Atm:  amu[k_atm] = amu[k_atm+1] + entu[k_atm] - detu[k_atm]
        # Cloud-top closure (mirrors the collab kernels): at the topmost
        # active layer force the updraft to fully detrain so no residual
        # `amu` escapes into the passthrough region above. A clipping
        # `lmax_conv` (active-region top below the cloud top) otherwise leaves
        # an uncompensated residual updraft flux → emission-driven mass
        # blow-up over many substeps. No-op when the updraft already closes
        # at the cloud top (the native detrainment then equals amu+entu).
        detu_k = (icltop == 1 && k_atm == icltop) ? (amu[k_atm + 1] + entu_col[k_atm]) : detu_col[k_atm]
        amu[k_atm] = amu[k_atm + 1] + entu_col[k_atm] - detu_k
        if amu[k_atm] > 0
            denom = amu[k_atm + 1] + entu_col[k_atm]
            zxi = max(zero(FT), one(FT) - detu_k / denom)
        else
            amu[k_atm] = zero(FT)
            zxi = zero(FT)
        end
        # Propagate updraft row k_atm from row k_atm+1; the surface
        # boundary (TM5 fu(0, kk) ≡ 0) is a literal zero so the
        # `(Nz, Nz)`-aliased shape never touches a phantom row.
        for kk_atm in (k_atm + 1):Nz
            f_below = k_atm == Nz ? zero(FT) : f[k_atm + 1, kk_atm]
            f[k_atm, kk_atm] = f_below * zxi
        end
        # TM5's `m(k)` is layer mass per unit horizontal area
        # (kg/m²). Runtime state stores total layer mass per cell
        # (kg), so topology kernels pass `cell_area` to recover the
        # original TM5 units before dividing the kg/m²/s fluxes.
        bmass_k = m_col[k_atm] / cell_area
        f[k_atm, k_atm] = entu_col[k_atm] / bmass_k * zxi
    end

    # --- Downdraft pass (AtmosTransport icllfs → Nz-1) ----------
    # TM5: do k = ld, 2, -1    (ld = icllfs_tm5; iterate top-down)
    # Atm: iterate k_atm in icllfs:(Nz-1), bottom-up.
    # TM5 "k-1" = layer below current in TM5 = k_atm+1 in AtmosTransport.
    # TM5 kk = k+1..ld = layers ABOVE current in TM5 = layers TOA-ward =
    #   kk_atm in icllfs:(k_atm-1)  (above current in AtmosTransport).
    # The "result row" `f(k-1, kk)` is "layer below current in TM5" =
    #   f[k_atm+1, kk_atm] in AtmosTransport.
    if icllfs <= Nz
        @inbounds for k_atm in icllfs:(Nz - 1)
            # TM5:  amd(k-1) = amd(k) - entd(k) + detd(k)
            # Atm:  amd[k_atm+1] = amd[k_atm] - entd[k_atm] + detd[k_atm]
            amd[k_atm + 1] = amd[k_atm] - entd_col[k_atm] + detd_col[k_atm]
            if amd[k_atm + 1] < 0
                denom = amd[k_atm] - entd_col[k_atm]
                zxi = max(zero(FT), one(FT) + detd_col[k_atm] / denom)
            else
                amd[k_atm + 1] = zero(FT)
                zxi = zero(FT)
            end
            # TM5:  f(k-1, kk) = f(k, kk) * zxi   for kk = k+1..ld
            # Atm:  f[k_atm+1, kk_atm] = f[k_atm, kk_atm] * zxi
            #       for kk_atm in icllfs:(k_atm-1)
            for kk_atm in icllfs:(k_atm - 1)
                f[k_atm + 1, kk_atm] = f[k_atm, kk_atm] * zxi
            end
            # TM5:  f(k-1, k) = -entd(k) / m(k) * zxi
            # Atm:  f[k_atm+1, k_atm] = -entd[k_atm] / bmass[k_atm] * zxi
            bmass_k = m_col[k_atm] / cell_area
            f[k_atm + 1, k_atm] = -entd_col[k_atm] / bmass_k * zxi
        end
    end

    # --- Subsidence subtraction --------------------------------
    # TM5's combine pass was `f(k, kk) += fu(k, kk)` followed by
    # subsidence at f(k, k+1) and f(k, k). The += step is gone:
    # the updraft pass above writes its contribution directly into
    # `f`, and the disjoint index ranges (updraft at
    # `kk_atm ≥ k_atm`, downdraft at `kk_atm < k_atm`) mean no
    # position is written by both passes. Only the subsidence
    # corrections remain.
    @inbounds for k_atm in Nz:-1:2
        bmass_above = m_col[k_atm - 1] / cell_area
        bmass_k     = m_col[k_atm]     / cell_area
        f[k_atm, k_atm - 1] -= amu[k_atm] / bmass_above
        f[k_atm, k_atm]     -= amd[k_atm] / bmass_k
    end

    # --- Final assembly: conv1(k, kk) = -dt · (f(k-1, kk) - f(k, kk)) + I
    # TM5: do k=1, lmx → Atm: for k_atm = 1:Nz (any order).
    # TM5 "f(k-1, kk)" = "row below current in TM5" = f[k_atm+1, kk_atm].
    #
    # `f` is allowed to alias `conv1` in the production workspace. Iterate
    # from top to bottom so overwriting row k never destroys row k+1 before it
    # is read by a later iteration. The implicit bottom row f[Nz+1, :] is zero.
    #
    # Rows above `icltop_eff` are provably identity rows and the active
    # lower-left quadrant is provably zero
    # (`test/test_tm5_sparsity_above_icltop.jl`). Write those explicitly and
    # only assemble the lower-right active block. This keeps the full-matrix
    # public contract while avoiding O(Nz²) dead work for shallow convection.
    @inbounds for k_atm in 1:(k_lo - 1)
        for kk_atm in 1:Nz
            conv1[k_atm, kk_atm] = k_atm == kk_atm ? one(FT) : zero(FT)
        end
    end
    @inbounds for k_atm in k_lo:Nz
        for kk_atm in 1:(k_lo - 1)
            conv1[k_atm, kk_atm] = zero(FT)
        end
        for kk_atm in k_lo:Nz
            f_below = k_atm == Nz ? zero(FT) : f[k_atm + 1, kk_atm]
            fdiff = f_below - f[k_atm, kk_atm]
            if fdiff != zero(FT)
                conv1[k_atm, kk_atm] = -dt * fdiff
            else
                conv1[k_atm, kk_atm] = zero(FT)
            end
        end
        conv1[k_atm, k_atm] += one(FT)
    end
    return nothing
end

"""
    _tm5_lu!(conv1, pivots, Nz) -> nothing

In-place partial-pivot Gaussian elimination on `conv1[1:Nz, 1:Nz]`.
Writes the L and U factors into the same matrix and the row
permutation into `pivots[1:Nz]`.

Only the active lower-right block beginning at `icltop_eff` is factored.
Partial pivoting is retained for forward and adjoint replay.

Columns with no diagnosed downdraft use `_tm5_hessenberg_lu!` instead.
This routine remains the general dense reference and handles all downdrafts.
"""
function _tm5_lu!(conv1::AbstractMatrix{FT},
                  pivots::AbstractVector{<:Integer},
                  Nz::Integer;
                  icltop_eff::Integer = 1) where {FT}
    Nz == 0 && return nothing
    # Active-window LU: rows above `icltop_eff` are
    # pure identity by `_tm5_build_conv1!`'s structure (proven by
    # `test/test_tm5_sparsity_above_icltop.jl`), and the lower-left
    # quadrant `(k ≥ icltop_eff, j < icltop_eff)` is zero. So the LU
    # factorization is block-diagonal — only the lower-right
    # `(Nz - icltop_eff + 1)`-by-`(Nz - icltop_eff + 1)` block needs
    # work. Inner loops below already iterate from `k+1` (or `1:k-1`
    # in the solve), so `k_lo := max(icltop_eff, 1)` is the only
    # change vs the bit-exact full-Nz path.
    k_lo = max(Int(icltop_eff), 1)

    @inbounds for k in k_lo:Nz
        # Find the pivot row within column k, range [k, Nz].
        piv = k
        pivmag = abs(conv1[k, k])
        for r in (k + 1):Nz
            mag = abs(conv1[r, k])
            if mag > pivmag
                piv = r
                pivmag = mag
            end
        end
        pivots[k] = piv
        if piv != k
            # Both rows share zeros in columns [1, k_lo-1]; swap only
            # the active columns to skip the dead-zero work.
            for c in k_lo:Nz
                tmp = conv1[k, c]
                conv1[k, c] = conv1[piv, c]
                conv1[piv, c] = tmp
            end
        end
        # Diagonal is guaranteed non-zero by diagonal dominance,
        # but protect against degenerate input just in case.
        diag = conv1[k, k]
        if diag == zero(FT)
            # Singular column — leave matrix untouched; solve will
            # propagate NaN/Inf. Upstream validators (preprocessor
            # sanity probe) should have caught this.
            continue
        end
        invd = one(FT) / diag
        for r in (k + 1):Nz
            conv1[r, k] *= invd
        end
        for r in (k + 1):Nz
            lrk = conv1[r, k]
            for c in (k + 1):Nz
                conv1[r, c] -= lrk * conv1[k, c]
            end
        end
    end
    return nothing
end

"""
    _tm5_hessenberg_lu!(A, pivots, Nz; icltop_eff=1)

Partial-pivot LU for an active upper-Hessenberg block. The caller guarantees
that entries below the first subdiagonal are zero. At step k, only rows k and
k+1 can supply a nonzero pivot, and only row k+1 needs elimination, reducing
factorization from cubic to quadratic work. Full active-row swaps preserve
previous L multipliers. General triangular solves handle every pivot sequence;
identity pivots allow specialized bidiagonal L and transpose-L solves.
"""
function _tm5_hessenberg_lu!(A::AbstractMatrix{FT},
                            pivots::AbstractVector{<:Integer}, Nz::Integer;
                            icltop_eff::Integer=1) where FT
    lo = max(Int(icltop_eff), 1)
    @inbounds for k in lo:Nz
        p = k < Nz && abs(A[k+1,k]) > abs(A[k,k]) ? k+1 : k
        pivots[k] = p
        if p != k
            for j in lo:Nz
                A[k,j], A[p,j] = A[p,j], A[k,j]
            end
        end
        diag = A[k,k]
        (iszero(diag) || k == Nz) && continue
        A[k+1,k] *= one(FT) / diag
        multiplier = A[k+1,k]
        for j in (k+1):Nz
            A[k+1,j] -= multiplier * A[k,j]
        end
    end
    return nothing
end

# Cloud diagnosis supplies an exact structural condition: if there is no
# downdraft pass, the builder creates no entries below the first subdiagonal.
# There is no small-flux threshold. Keep this branch outside both LU loops.
@inline function _tm5_factorize!(A, pivots, Nz, icllfs; icltop_eff=1)
    if icllfs > Nz
        _tm5_hessenberg_lu!(A, pivots, Nz; icltop_eff)
    else
        _tm5_lu!(A, pivots, Nz; icltop_eff)
    end
    return nothing
end

"""
    _tm5_solve!(rm_col, conv1, pivots, Nz, Nt) -> nothing

Apply the factored `conv1` + stored pivots to `rm_col` (shape
`(Nz, Nt)`). Back-substitutes in place for each tracer `1..Nt`.
"""
function _tm5_solve!(rm_col::AbstractMatrix{FT},
                     conv1::AbstractMatrix{FT},
                     pivots::AbstractVector{<:Integer},
                     Nz::Integer,
                     Nt::Integer;
                     icltop_eff::Integer = 1) where {FT}
    Nz == 0 && return nothing
    # Rows [1, icltop_eff-1] are identity, so
    # rm_col[1:icltop_eff-1, :] passes through unchanged. Lower-left
    # quadrant `conv1[k ≥ icltop_eff, j < icltop_eff] == 0`, so the
    # forward solve's `j` loop can start at icltop_eff. Pivots are
    # only written by `_tm5_lu!` for `k ≥ icltop_eff`; entries below
    # are unchanged from the previous column's run, so we must skip
    # them here too.
    k_lo = max(Int(icltop_eff), 1)
    for t in 1:Nt
        _tm5_solve_shared_tracer!(rm_col, conv1, pivots, Nz, k_lo, t)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Column mass ledger. Float32 elimination on the near-identity convection
# matrix loses 1–2e-8 of the active column per solve, always with the same
# sign (≈1e-5 of every tracer per month on the L40S). Compensated sums of the
# active window (`lo:n`, slot `slot` of `q`) before and after the solve give
# the residual, a few ulps of the largest cell, which is added back there;
# spread over the column it would round away again. What remains is unbiased.
# ---------------------------------------------------------------------------
@inline _tm5_column_sum(q, lo, n, slot) = _neumaier_sum(k -> @inbounds(q[k, slot]), lo:n, eltype(q))

@inline function _tm5_restore_column_mass!(q, lo, n, slot, before)
    kmax, largest = lo, abs(@inbounds q[lo, slot])
    @inbounds for k in (lo + 1):n
        value = abs(q[k, slot])
        value > largest && ((kmax, largest) = (k, value))
    end
    after = _tm5_column_sum(q, lo, n, slot)
    @inbounds q[kmax, slot] += _ledger_residual(before, after, largest, n - lo + 1)
    return nothing
end

# A thread owns one complete shared-memory RHS. There are no workgroup
# operations here, so KA can inline this helper in each topology kernel.
@inline function _tm5_solve_shared_tracer!(q, A, pivots, n, lo, slot)
    @inbounds begin
        for k in lo:n
            p = Int(pivots[k])
            if p != k
                q[k, slot], q[p, slot] = q[p, slot], q[k, slot]
            end
        end
        for k in lo:n
            value = q[k, slot]
            for j in lo:(k - 1)
                value -= A[k, j] * q[j, slot]
            end
            q[k, slot] = value
        end
        for k in n:-1:lo
            value = q[k, slot]
            for j in (k + 1):n
                value -= A[k, j] * q[j, slot]
            end
            q[k, slot] = value / A[k, k]
        end
    end
    return nothing
end

# With no row swaps, Hessenberg elimination leaves a lower-bidiagonal L.
# Check the actual pivot sequence; the structural property alone is not enough.
@inline function _tm5_identity_pivots(pivots, Nz, lo)
    @inbounds for k in lo:Nz
        pivots[k] == k || return false
    end
    return true
end

# One independent RHS, shared by serial CPU and collaborative GPU callers.
# Preconditions: Hessenberg factors, identity pivots, finite tracer input.
@inline function _tm5_solve_bidiagonal_tracer!(q, A, Nz, lo, tracer)
    @inbounds begin
        for k in (lo+1):Nz
            q[k,tracer] -= A[k,k-1] * q[k-1,tracer]
        end
        for k in Nz:-1:lo
            value = q[k,tracer]
            for j in (k+1):Nz
                value -= A[k,j] * q[j,tracer]
            end
            q[k,tracer] = value / A[k,k]
        end
    end
    return nothing
end

"""
    _tm5_conserving_solve_tracer!(q, A, pivots, n, lo, tracer, bidiagonal)

Production convection solve of one tracer column with the mass ledger. TM5's
convection matrix conserves column mass (unit column sums; the cloud-top
closure keeps the active window closed), so the residual of the Float32 solve
is rounding and is returned to the column (see `_ledger_residual`). The plain
solves above stay exact linear algebra for any matrix.
"""
@inline function _tm5_conserving_solve_tracer!(q, A, pivots, n, lo, tracer, bidiagonal::Bool)
    before = _tm5_column_sum(q, lo, n, tracer)
    if bidiagonal
        _tm5_solve_bidiagonal_tracer!(q, A, n, lo, tracer)
    else
        _tm5_solve_shared_tracer!(q, A, pivots, n, lo, tracer)
    end
    _tm5_restore_column_mass!(q, lo, n, tracer, before)
    return nothing
end

function _tm5_solve_bidiagonal!(q, A, Nz, Nt; icltop_eff=1)
    lo = max(Int(icltop_eff), 1)
    for tracer in 1:Nt
        _tm5_solve_bidiagonal_tracer!(q, A, Nz, lo, tracer)
    end
    return nothing
end

"""
    _tm5_solve_column!(rm_col, m_col, entu_col, detu_col, entd_col, detd_col,
                       conv1_buf, pivots_buf, cloud_dims, dt) -> nothing

Backend-agnostic full column solver. Shapes:

- `rm_col      :: AbstractMatrix{FT}`  (Nz, Nt) — tracer masses; updated in place.
- `m_col       :: AbstractVector{FT}`  (Nz,)    — layer air mass per cell (kg); interpreted
                                                  as kg/m² when `cell_area = 1`.
- `entu_col, detu_col, entd_col, detd_col :: AbstractVector{FT}` (Nz,) — forcings.
- `conv1_buf   :: AbstractMatrix{FT}`  (Nz, Nz) — scratch matrix.
- `pivots_buf  :: AbstractVector{Int}` (Nz,)    — scratch permutation.
- `cloud_dims  :: AbstractVector{Int}` (3,)     — output (icltop, iclbas, icllfs)
                                                  in AtmosTransport orientation.
- `dt          :: FT`                            — step length in seconds.
- `cell_area   :: FT`                            — grid-cell area (m²) used to convert
                                                  `m_col` to kg/m².

Zero-forcing (`entu = detu = entd = detd = 0`) short-circuits to
identity. Zero-size columns (`Nz = 0`) are a no-op.
"""
function _tm5_solve_column!(rm_col::AbstractMatrix{FT},
                            m_col::AbstractVector{FT},
                            entu_col::AbstractVector{FT},
                            detu_col::AbstractVector{FT},
                            entd_col::AbstractVector{FT},
                            detd_col::AbstractVector{FT},
                            conv1_buf::AbstractMatrix{FT},
                            pivots_buf::AbstractVector{<:Integer},
                            cloud_dims::AbstractVector{<:Integer},
                            dt::FT;
                            cell_area::FT = one(FT),
                            f_buf::AbstractMatrix{FT}   = zeros(FT, length(m_col) + 1, length(m_col)),
                            amu_buf::AbstractVector{FT} = zeros(FT, length(m_col) + 1),
                            amd_buf::AbstractVector{FT} = zeros(FT, length(m_col) + 1),
                            ) where {FT}
    Nz = length(m_col)
    Nz == 0 && return nothing
    Nt = size(rm_col, 2)

    icltop, iclbas, icllfs = _tm5_diagnose_cloud_dims(detu_col, entd_col, Nz)
    cloud_dims[1] = icltop
    cloud_dims[2] = iclbas
    cloud_dims[3] = icllfs

    # No convection in this column → identity transform.
    if icltop > Nz
        return nothing
    end

    # Active-window LU. `icltop_eff` is the smallest
    # row index with any non-identity entry, derived from the builder
    # in `_tm5_build_conv1!`. Proven correct by
    # `test/test_tm5_sparsity_above_icltop.jl`. The `max(icltop, 2)`
    # guard keeps the formula well-defined at icltop = 1 (no skip).
    icltop_eff = min(Int(icllfs), max(Int(icltop), 2) - 1)

    _tm5_build_conv1!(conv1_buf,
                      entu_col, detu_col, entd_col, detd_col, m_col,
                      icltop, icllfs, dt, Nz;
                      cell_area = cell_area,
                      f = f_buf,
                      amu = amu_buf, amd = amd_buf)
    _tm5_factorize!(conv1_buf, pivots_buf, Nz, icllfs; icltop_eff = icltop_eff)
    bidiagonal = icllfs > Nz && _tm5_identity_pivots(pivots_buf, Nz, icltop_eff)
    for t in 1:Nt
        _tm5_conserving_solve_tracer!(rm_col, conv1_buf, pivots_buf, Nz, max(icltop_eff, 1), t, bidiagonal)
    end
    return nothing
end
