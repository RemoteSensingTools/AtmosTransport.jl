# ---------------------------------------------------------------------------
# Adjoint of CS implicit vertical diffusion.
#
# Reverse-mode of `ImplicitVerticalDiffusion`. Builds the same Thomas-
# solve coefficients as the forward pass on the fly, then runs the
# transpose solve (upper-triangular sweep first, then back-substitution).
# Mass-aware: enters and exits in lambda-on-tracer-mass space.
#
# The CS adjoint kernel (commit `bff8933`, 2026-05-25) transposes the
# TM5-style mass-flux forward coefficients. The forward+adjoint share the
# same `(dkg, m_k, dt)` ingredients; `a_T[k] = c[k-1]` and `c_T[k] = a[k+1]`
# reduce to mass-flux entries with the "other layer's" m as normalizer. See
# file-top doc of `src/Operators/Diffusion/diffusion_kernels.jl` for the
# forward derivation and `memory/diffusion_full_pipeline_audit_2026_05_25.md`
# for the audit chain.
# ---------------------------------------------------------------------------

@inline function _adjoint_diffusion_time(::Type{FT}, meteo) where FT
    return meteo === nothing ? zero(FT) : FT(current_time(meteo))
end

# Mass-flux adjoint: transpose of the forward `Ã = M⁻¹·A·M` (on VMR),
# which is equivalent to the column-stochastic A on tracer mass. The
# transposed tridiagonal has:
#
#     a_T[k] = c[k-1]    super-of-row-(k-1) becomes sub-of-row-k
#     b_T[k] = b[k]
#     c_T[k] = a[k+1]    sub-of-row-(k+1) becomes super-of-row-k
#
# Substituting the forward coefficients
#     a[k]   = -dt·dkg[k-½]/m_k
#     c[k]   = -dt·dkg[k+½]/m_k
# at the transposed indices gives the adjoint entries in terms of the
# `dkg` evaluated at the same interface but with the OTHER layer's
# air mass as the normalizer:
#     a_T[k] = c[k-1] = -dt·dkg[k-½]/m_{k-1}      (note m_{k-1}, not m_k)
#     c_T[k] = a[k+1] = -dt·dkg[k+½]/m_{k+1}      (note m_{k+1}, not m_k)
#     b_T[k] = 1 + dt·(dkg[k-½] + dkg[k+½])/m_k
@kernel function _vertical_diffusion_cs_single_adjoint_kernel!(
    lambda, @Const(air_mass), kz_field, @Const(dz), w_scratch,
    dt, Nz::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    FT = eltype(lambda)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        dt_ft = FT(dt)

        # Scalar-carry pattern matching the forward kernel: amortize
        # global-memory loads of (Kz, dz, air_mass) across consecutive
        # vertical levels. Without this, every iteration re-loaded the
        # k-1 and k+1 neighbors, tripling the load count on the column.
        Kz_prev = zero(FT)
        dz_prev = zero(FT)
        m_prev  = zero(FT)
        w_prev  = zero(FT)
        g_prev  = zero(FT)

        Kz_k = field_value(kz_field, (ii, jj, 1))
        dz_k = dz[ii, jj, 1]
        m_k  = air_mass[i, j, 1]

        for k in 1:Nz
            dkg_above = zero(FT)
            dkg_below = zero(FT)
            Kz_next   = zero(FT)
            dz_next   = zero(FT)
            m_next    = zero(FT)
            a_T = zero(FT)
            c_T = zero(FT)

            if k > 1
                sum_dz_above = dz_prev + dz_k
                dkg_above = (m_prev + m_k) * (Kz_prev + Kz_k) /
                            (sum_dz_above * sum_dz_above)
                inv_m_prev = m_prev > zero(FT) ? one(FT) / m_prev : zero(FT)
                a_T = -dt_ft * dkg_above * inv_m_prev
            end

            if k < Nz
                Kz_next = field_value(kz_field, (ii, jj, k + 1))
                dz_next = dz[ii, jj, k + 1]
                m_next  = air_mass[i, j, k + 1]
                sum_dz_below = dz_k + dz_next
                dkg_below = (m_k + m_next) * (Kz_k + Kz_next) /
                            (sum_dz_below * sum_dz_below)
                inv_m_next = m_next > zero(FT) ? one(FT) / m_next : zero(FT)
                c_T = -dt_ft * dkg_below * inv_m_next
            end

            inv_m_k = m_k > zero(FT) ? one(FT) / m_k : zero(FT)
            b_T = one(FT) + dt_ft * (dkg_above + dkg_below) * inv_m_k
            d_k = m_k > zero(FT) ? m_k * lambda[i, j, k] : zero(FT)

            if k == 1
                denom = b_T
                w_k = c_T / denom
                g_k = d_k / denom
            else
                denom = b_T - a_T * w_prev
                w_k = c_T / denom
                g_k = (d_k - a_T * g_prev) / denom
            end

            w_scratch[ii, jj, k] = w_k
            lambda[i, j, k] = g_k

            if k < Nz
                w_prev  = w_k
                g_prev  = g_k
                Kz_prev = Kz_k
                dz_prev = dz_k
                m_prev  = m_k
                Kz_k    = Kz_next
                dz_k    = dz_next
                m_k     = m_next
            end
        end

        for k in (Nz - 1):-1:1
            lambda[i, j, k] = lambda[i, j, k] -
                              w_scratch[ii, jj, k] * lambda[i, j, k + 1]
        end

        for k in 1:Nz
            m_div = air_mass[i, j, k]
            lambda[i, j, k] = m_div > zero(FT) ? lambda[i, j, k] / m_div : zero(FT)
        end
    end
end

# Transpose each partition using its smaller weight directly. This preserves
# both constant mass-objective seeds and weak recipient sensitivities.
@inline function _dkg_transpose_partition(seed, neighbor_seed, ratio)
    iszero(ratio) && return seed
    if ratio > one(ratio)
        fraction = one(ratio) / (one(ratio) + ratio)
        return neighbor_seed + fraction * (seed - neighbor_seed)
    else
        fraction = ratio / (one(ratio) + ratio)
        return seed + fraction * (neighbor_seed - seed)
    end
end

@kernel function _vertical_diffusion_cs_single_dkg_adjoint_kernel!(
    lambda, @Const(air_mass), dkg_field, w_scratch,
    dt, Nz::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    FT = eltype(lambda)
    i, j = ii + Hp, jj + Hp
    @inbounds begin
        coupling = zero(FT)
        previous_seed = zero(FT)
        for k in 1:Nz
            m = air_mass[i, j, k]
            next_m = k < Nz ? air_mass[i, j, k + 1] : zero(FT)
            exchange = k < Nz ? FT(dt) * field_value(dkg_field, (ii, jj, k)) : zero(FT)
            forward_ratio, backward_ratio, coupling = _dkg_transfer_ratios(m, next_m, exchange, coupling)
            w_scratch[ii, jj, k] = forward_ratio
            seed = m > zero(FT) ? lambda[i, j, k] : zero(FT)
            previous_seed = _dkg_transpose_partition(seed, previous_seed, backward_ratio)
            lambda[i, j, k] = previous_seed
        end
        next_seed = zero(FT)
        for k in Nz:-1:1
            next_seed = _dkg_transpose_partition(lambda[i, j, k], next_seed, w_scratch[ii, jj, k])
            lambda[i, j, k] = air_mass[i, j, k] > zero(FT) ? next_seed : zero(FT)
        end
    end
end

function _require_cs_diffusion_workspace(workspace)
    workspace === nothing && throw(ArgumentError(
        "CS adjoint diffusion requires a panel-native DiffusionWorkspace"))
    hasproperty(workspace, :factors) && hasproperty(workspace, :layer_thickness) ||
        throw(ArgumentError(
            "CS adjoint diffusion requires a workspace with panel-native " *
            "`factors` and `layer_thickness` tuples"))
    w_scratch = workspace.factors
    dz_scratch = workspace.layer_thickness
    length(w_scratch) == 6 && length(dz_scratch) == 6 ||
        throw(DimensionMismatch("CS adjoint diffusion workspace must provide 6 panel scratch arrays"))
    return w_scratch, dz_scratch
end

function _diffusion_sequence_at(value, step::Int, nsteps::Int,
                                name::AbstractString)
    if value isa AbstractVector
        length(value) == nsteps || throw(ArgumentError(
            "$name length $(length(value)) does not match nsteps $nsteps"))
        return value[step]
    else
        return value
    end
end

@inline _validate_cs_diffusion_kz_for_adjoint(_op) = nothing

_validate_cs_diffusion_kz_for_adjoint(::ImplicitVerticalDiffusion{FT, <:GCHPNonlocalPBLField}) where FT =
    throw(ArgumentError(
        "GEOS-Chem non-local VDIFF has no adjoint yet: the footprint tape neither refreshes " *
        "the field nor transposes its counter-gradient emission profile."))

function _validate_cs_diffusion_kz_for_adjoint(
    op::ImplicitVerticalDiffusion{FT, <:LocalHoltslagBovilleKzField}) where FT
    @inbounds for p in 1:6
        data = panel_field(op.kz_field, p).data
        all(isfinite, data) || throw(ArgumentError(
            "Local Holtslag-Boville VDIFF diffusion adjoint requires a finite Kz " *
            "cache; panel $p contains NaN or Inf. Refresh with " *
            "`refresh_local_holtslag_boville_kz_cache!` before recording the tape."))
        maximum(data) > zero(FT) || throw(ArgumentError(
            "Local Holtslag-Boville VDIFF diffusion adjoint received an all-zero " *
            "Kz cache on panel $p. This usually means the VDIFF Kz field was " *
            "constructed but never refreshed from surface/vdiff forcing before " *
            "the adjoint tape was recorded."))
    end
    return nothing
end

function _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace,
                                       nsteps::Int)
    checked_kz = IdDict{Any, Bool}()
    for step in 1:nsteps
        op = _diffusion_sequence_at(diffusion_op, step, nsteps, "diffusion_op")
        if !(op isa NoDiffusion)
            if !haskey(checked_kz, op.kz_field)
                _validate_cs_diffusion_kz_for_adjoint(op)
                checked_kz[op.kz_field] = true
            end
            ws = _diffusion_sequence_at(diffusion_workspace, step, nsteps,
                                        "diffusion_workspace")
            _require_cs_diffusion_workspace(ws)
        end
    end
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels, panels_m, ::NoDiffusion,
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh)
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels::NTuple{6, A},
                                      panels_m::NTuple{6},
                                      op::ImplicitVerticalDiffusion{FT, KzF},
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh) where {
                                          FT, A <: AbstractArray{FT, 3},
                                          KzF <: AbstractCSDkgField{FT}}
    w_scratch, _ = _require_cs_diffusion_workspace(workspace)
    update_field!(op.kz_field, _adjoint_diffusion_time(FT, meteo))
    Hp, Nc = mesh.Hp, mesh.Nc
    @inbounds for p in 1:6
        panel_lambda = lambda_panels[p]
        panel_m = panels_m[p]
        Nz = size(panel_lambda, 3)
        panel_dkg = panel_field(op.kz_field, p)
        backend = get_backend(panel_lambda)
        kernel! = _vertical_diffusion_cs_single_dkg_adjoint_kernel!(backend, (8, 8))
        kernel!(panel_lambda, panel_m, panel_dkg, w_scratch[p], FT(dt), Nz, Hp;
                ndrange = (Nc, Nc))
        synchronize(backend)
    end
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels::NTuple{6, A},
                                      panels_m::NTuple{6},
                                      op::ImplicitVerticalDiffusion{FT, KzF},
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh) where {
                                          FT, A <: AbstractArray{FT, 3},
                                          KzF <: AbstractCubedSphereField{FT}}
    w_scratch, dz_scratch = _require_cs_diffusion_workspace(workspace)
    update_field!(op.kz_field, _adjoint_diffusion_time(FT, meteo))

    Hp = mesh.Hp
    Nc = mesh.Nc
    @inbounds for p in 1:6
        panel_lambda = lambda_panels[p]
        panel_m = panels_m[p]
        size(panel_lambda) == size(panel_m) || throw(DimensionMismatch(
            "adjoint tracer panel $p shape $(size(panel_lambda)) does not match " *
            "air_mass shape $(size(panel_m))"))
        Nz = size(panel_lambda, 3)
        expected = (Nc, Nc, Nz)
        size(w_scratch[p]) == size(dz_scratch[p]) ||
            throw(DimensionMismatch("CS adjoint diffusion w_scratch and dz_scratch sizes differ on panel $p"))
        size(w_scratch[p]) == expected ||
            throw(DimensionMismatch(
                "CS adjoint diffusion workspace panel $p has shape $(size(w_scratch[p])); " *
                "expected $expected"))

        panel_kz = panel_field(op.kz_field, p)
        backend = get_backend(panel_lambda)
        kernel! = _vertical_diffusion_cs_single_adjoint_kernel!(backend, (8, 8))
        kernel!(panel_lambda, panel_m, panel_kz, dz_scratch[p], w_scratch[p],
                FT(dt), Nz, Hp; ndrange = (Nc, Nc))
        synchronize(backend)
    end
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels, panels_m,
                                      op::ImplicitVerticalDiffusion,
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh)
    throw(ArgumentError(
        "CS adjoint diffusion requires `ImplicitVerticalDiffusion` with a " *
        "`CubedSphereField` Kz field; got $(typeof(op.kz_field))"))
end
