# ---------------------------------------------------------------------------
# CMFMCMatrixConvection derivation kernels — convert GEOS (cmfmc, dtrain)
# into TM5-form (entu, detu, 0, 0) per column.
#
# Continuity at layer k (k=1=TOA, k=Nz=surface; cmfmc[k] is the flux at the
# *top* interface of layer k, cmfmc[k+1] is the flux at its *bottom*):
#
#     M_c[k] - M_c[k+1]  =  entu[k] - detu[k]
#
# GCHP supplies dtrain[k] directly (the explicit detrainment rate), so by
# closing the system with detu[k] = dtrain[k] the entrainment falls out:
#
#     raw_E[k] = cmfmc[k] - cmfmc[k+1] + dtrain[k]   (= GCHP's "entrn")
#
# `raw_E` can go negative near cloud top (cloud is shrinking faster than
# `dtrain` accounts for). Since TM5 requires `entu ≥ 0`, fold the negative
# excess into the detrainment:
#
#     entu[k] = max(0, raw_E[k])
#     detu[k] = dtrain[k] - min(0, raw_E[k])     (= dtrain + max(0, -raw_E))
#
# Both branches preserve continuity layer by layer. Expansion:
#   raw_E ≥ 0:  entu - detu = raw_E - dtrain
#                            = (cmfmc[k] - cmfmc[k+1] + dtrain) - dtrain
#                            = cmfmc[k] - cmfmc[k+1]                       ✓
#   raw_E < 0:  entu - detu = 0 - (dtrain - raw_E)
#                            = raw_E - dtrain
#                            = cmfmc[k] - cmfmc[k+1]                       ✓
#
# Column closure `Σ entu = Σ detu`: telescoping gives
#     δ = Σ detu - Σ entu = cmfmc[Nz+1] - cmfmc[1]
# which is zero when the binary's boundary cmfmc values are zero (TOA +
# surface). Any numerical residual is absorbed at layer Nz (surface) by
# adding to whichever side keeps the result non-negative:
#     δ > 0  (more detrainment than entrainment):  entu[Nz] += δ
#     δ < 0  (more entrainment than detrainment):  detu[Nz] -= δ
# This always preserves entu ≥ 0 and detu ≥ 0 (which the TM5 LU column-
# stochasticity argument relies on), even on a binary with non-zero TOA
# cmfmc.
#
# These kernels run ONCE per met-window advance (cmfmc + dtrain are
# window-constant). The CMFMCMatrixWorkspace caches the derived arrays and
# the operator's `apply!` reuses them across all substeps inside the
# window.
# ---------------------------------------------------------------------------

# =========================================================================
# LatLon — (Nx, Ny, Nz+1) cmfmc, (Nx, Ny, Nz) dtrain/entu/detu
# =========================================================================

@inline function _derive_cmfmc_matrix_rates_column!(entu_out, detu_out, cmfmc, dtrain, Nz, cell...)
    FT = eltype(entu_out)
    sum_e = zero(FT)
    sum_d = zero(FT)
    @inbounds for k in 1:Nz
        d_k = dtrain[cell..., k]
        ec = cmfmc[cell..., k] - cmfmc[cell..., k + 1]
        raw_E = ec + d_k
        if raw_E >= zero(FT)
            entu_out[cell..., k] = raw_E
            detu_out[cell..., k] = d_k
            sum_e += raw_E
            sum_d += d_k
        else
            entu_out[cell..., k] = zero(FT)
            detu_out[cell..., k] = d_k - raw_E
            sum_d += d_k - raw_E
        end
    end
    # Defensive column closure. Forces `Σ entu == Σ detu` exactly so the
    # TM5 LU matrix is column-stochastic and `Σ(m·q)` is preserved to
    # roundoff for any inert tracer. Split into pos/neg branches so the
    # absorbed residual never drives entu or detu negative — important
    # when the binary delivers non-zero cmfmc at TOA (telescope leaves
    # δ < 0, which we'd otherwise add to entu[Nz] and risk going below
    # zero on a small surface raw_E).
    delta = sum_d - sum_e
    if delta > zero(FT)
        @inbounds entu_out[cell..., Nz] += delta
    elseif delta < zero(FT)
        @inbounds detu_out[cell..., Nz] -= delta
    end
    return nothing
end

@kernel function _derive_cmfmc_matrix_rates_ll_kernel!(
    entu_out, detu_out, @Const(cmfmc), @Const(dtrain), Nz::Int)
    i, j = @index(Global, NTuple)
    _derive_cmfmc_matrix_rates_column!(entu_out, detu_out, cmfmc, dtrain, Nz, i, j)
end

@kernel function _derive_cmfmc_matrix_rates_rg_kernel!(
    entu_out, detu_out, @Const(cmfmc), @Const(dtrain), Nz::Int)
    c = @index(Global)
    _derive_cmfmc_matrix_rates_column!(entu_out, detu_out, cmfmc, dtrain, Nz, c)
end

# =========================================================================
# Host-side dispatch helpers — launch the right kernel for the topology.
# Called by `CMFMCMatrixConvection.apply!` exactly once per met-window
# refresh.
# =========================================================================

# LatLon
@inline function _launch_cmfmc_matrix_derivation!(
    entu_out::AbstractArray{<:Any, 3},
    detu_out::AbstractArray{<:Any, 3},
    cmfmc::AbstractArray{<:Any, 3},
    dtrain::AbstractArray{<:Any, 3},
)
    backend = get_backend(entu_out)
    Nx, Ny, Nz = size(entu_out)
    size(detu_out) == (Nx, Ny, Nz) || throw(DimensionMismatch(
        "_launch_cmfmc_matrix_derivation!: detu_out shape mismatch."))
    size(cmfmc)    == (Nx, Ny, Nz + 1) || throw(DimensionMismatch(
        "_launch_cmfmc_matrix_derivation!: cmfmc shape mismatch."))
    size(dtrain)   == (Nx, Ny, Nz) || throw(DimensionMismatch(
        "_launch_cmfmc_matrix_derivation!: dtrain shape mismatch."))
    kernel = _derive_cmfmc_matrix_rates_ll_kernel!(backend)
    kernel(entu_out, detu_out, cmfmc, dtrain, Int(Nz); ndrange = (Nx, Ny))
    synchronize(backend)
    return nothing
end

# Face-indexed ReducedGaussian
@inline function _launch_cmfmc_matrix_derivation!(
    entu_out::AbstractArray{<:Any, 2},
    detu_out::AbstractArray{<:Any, 2},
    cmfmc::AbstractArray{<:Any, 2},
    dtrain::AbstractArray{<:Any, 2},
)
    backend = get_backend(entu_out)
    ncells, Nz = size(entu_out)
    size(detu_out) == (ncells, Nz)     || throw(DimensionMismatch(
        "_launch_cmfmc_matrix_derivation!: detu_out shape mismatch."))
    size(cmfmc)    == (ncells, Nz + 1) || throw(DimensionMismatch(
        "_launch_cmfmc_matrix_derivation!: cmfmc shape mismatch."))
    size(dtrain)   == (ncells, Nz)     || throw(DimensionMismatch(
        "_launch_cmfmc_matrix_derivation!: dtrain shape mismatch."))
    kernel = _derive_cmfmc_matrix_rates_rg_kernel!(backend)
    kernel(entu_out, detu_out, cmfmc, dtrain, Int(Nz); ndrange = ncells)
    synchronize(backend)
    return nothing
end

# CubedSphere panel tuple
@inline function _launch_cmfmc_matrix_derivation!(
    entu_out::NTuple{6, <:AbstractArray{<:Any, 3}},
    detu_out::NTuple{6, <:AbstractArray{<:Any, 3}},
    cmfmc::NTuple{6, <:AbstractArray{<:Any, 3}},
    dtrain::NTuple{6, <:AbstractArray{<:Any, 3}},
)
    backend = get_backend(entu_out[1])
    Nc, _, Nz = size(entu_out[1])
    for p in 1:6
        size(entu_out[p]) == (Nc, Nc, Nz)     || throw(DimensionMismatch(
            "_launch_cmfmc_matrix_derivation!: entu_out panel $p shape mismatch."))
        size(detu_out[p]) == (Nc, Nc, Nz)     || throw(DimensionMismatch(
            "_launch_cmfmc_matrix_derivation!: detu_out panel $p shape mismatch."))
        size(cmfmc[p])    == (Nc, Nc, Nz + 1) || throw(DimensionMismatch(
            "_launch_cmfmc_matrix_derivation!: cmfmc panel $p shape mismatch."))
        size(dtrain[p])   == (Nc, Nc, Nz)     || throw(DimensionMismatch(
            "_launch_cmfmc_matrix_derivation!: dtrain panel $p shape mismatch."))
    end
    kernel = _derive_cmfmc_matrix_rates_ll_kernel!(backend)
    for p in 1:6
        kernel(entu_out[p], detu_out[p], cmfmc[p], dtrain[p], Int(Nz);
               ndrange = (Nc, Nc))
    end
    synchronize(backend)
    return nothing
end
