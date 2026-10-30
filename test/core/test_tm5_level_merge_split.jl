#!/usr/bin/env julia
#
# TM5 convection level merging (`n_merge > 1`, OF-107): the split of each
# super-layer's tracer-mass change back to its fine layers,
#
#     r'_k = r_k + (R'_s - R_s) |r_k| / Σ_{j∈s} |r_j|      (m_k / M_s if that sum is 0).
#
# For super-layers whose fine values share one sign this is evaluated as
# r'_k = R'_s (r_k / R_s). The merged path runs only in the collaborative GPU
# solver, so the CPU tests drive `_tm5_merge_aggregate_solve_disaggregate!`
# directly with the serial column solve on the super grid; a CUDA testset runs
# the production `apply!`. The previous proportional split
# `r'_k = R'_s r_k / R_s` fails the near-cancellation testset (column Σ|r| grew
# 130-200-fold in one step) and the unchanged-layer testset.

using Test, Random

using AtmosTransport

const _HAS_CUDA = try
    @eval using CUDA
    @eval CUDA.functional()
catch
    false
end
if _HAS_CUDA
    @eval using Adapt
end

const Cv = AtmosTransport.Operators.Convection

# Convective forcing (kg m⁻² s⁻¹) on `Nz` layers, k = 1 at the top: an updraft
# entraining in the bottom three layers and detraining in `top:top+2`, and a
# downdraft entraining just below that and detraining at the surface.
function _merge_test_forcing(::Type{FT}, Nz; top = 4) where {FT}
    entu = zeros(FT, Nz); detu = zeros(FT, Nz); entd = zeros(FT, Nz); detd = zeros(FT, Nz)
    entu[(Nz - 2):Nz]      .= FT(0.02)
    detu[top:(top + 2)]    .= FT(0.02)
    entd[(top + 3):(top + 4)] .= FT(0.004)
    detd[(Nz - 1):Nz]      .= FT(0.004)
    return entu, detu, entd, detd
end

# Serial coarse solve of every column of LL-layout super arrays (…, L_super, Nt).
function _serial_super_solve!(dt, qS, mS, eU, dU, eD, dD)
    FT = eltype(qS)
    L = size(qS, 3)
    conv1 = zeros(FT, L, L); piv = zeros(Int, L); dims = zeros(Int, 3)
    for j in axes(qS, 2), i in axes(qS, 1)
        Cv.TM5._tm5_solve_column!(view(qS, i, j, :, :), mS[i, j, :], eU[i, j, :], dU[i, j, :],
                              eD[i, j, :], dD[i, j, :], conv1, piv, dims, FT(dt))
    end
    return nothing
end

# Merged step on LL-layout arrays q (Nx, Ny, Nz, Nt), m (Nx, Ny, Nz).
function _merged_step!(q, m, forcing, n_merge; L = size(m, 3), dt = 3600)
    Nz = size(m, 3)
    L_super = fld(L, n_merge); L_padded = L_super * n_merge
    Cv.TM5._tm5_merge_aggregate_solve_disaggregate!(
        q, m, forcing..., Nz - L_padded, L_padded, L_super, n_merge, 3,
        (args...) -> _serial_super_solve!(dt, args...))
    return q
end

# Independent coarse result R'_s for one column (aggregate, solve), used to
# check the split against the solve it distributes.
function _coarse_column(r, m, forcing, n_merge; dt = 3600)
    FT = eltype(r); Ls = length(m) ÷ n_merge
    agg(x) = [sum(x[((s - 1) * n_merge + 1):(s * n_merge)]) for s in 1:Ls]
    R = reshape(agg(r), Ls, 1); R0 = copy(R)
    Cv.TM5._tm5_solve_column!(R, agg(m), map(agg, forcing)..., zeros(FT, Ls, Ls), zeros(Int, Ls),
                          zeros(Int, 3), FT(dt))
    return vec(R0), vec(R)
end

_column_forcing(::Type{FT}, Nx, Ny, Nz; kw...) where {FT} =
    map(f -> repeat(reshape(f, 1, 1, Nz), Nx, Ny, 1), _merge_test_forcing(FT, Nz; kw...))

# Shared CPU/CUDA fixture: layer 10 receives detrainment with cloud top 9.
# For n_merge = 2 or 4, its super-layer contains only this subnormal seed.
function _nearly_empty_merge_tracer(m)
    FT = eltype(m); Nz = size(m, 3)
    q = zeros(FT, size(m))
    q[:, :, (Nz - 3):Nz] .= FT(1e-5) .* m[:, :, (Nz - 3):Nz]
    q[:, :, 10] .= floatmin(FT) / 2^20
    return q
end

@testset "TM5 level merging: increment split" begin
    Nz = 12

    @testset "near-cancelling mixed-sign pairs do not grow Σ|r| ($FT)" for FT in (Float32, Float64)
        m = FT.(collect(range(800, 1200; length = Nz)))
        forcing = _merge_test_forcing(FT, Nz)
        base = FT(1e-6) .* m
        columns = Vector{Vector{FT}}()
        # Pairs (2s-1, 2s) holding ±a with a residual 1e-6 a, 1e-3 a, and an
        # exact cancellation, inside and above the cloud window.
        for (s, rel) in ((6, 1e-6), (5, 1e-3), (4, 1e-6), (3, 0.0), (1, 1e-6))
            r = copy(base)
            r[2s - 1] = FT(1e-3) * m[2s - 1]
            r[2s] = -r[2s - 1] * (1 - FT(rel))
            push!(columns, r)
        end
        for r in columns
            q = reshape(copy(r), 1, 1, Nz, 1)
            _merged_step!(q, reshape(m, 1, 1, Nz), map(f -> reshape(f, 1, 1, Nz), forcing), 2)
            r_new = vec(q)
            R0, R1 = _coarse_column(r, m, forcing, 2)
            dR = R1 .- R0
            tol = 64 * eps(FT) * sum(abs, r)
            # The split distributes exactly the coarse change of each super-layer.
            for s in eachindex(dR)
                ks = (2s - 1):(2s)
                @test isapprox(sum(r_new[ks]), R1[s]; atol = tol)
                @test all(abs.(r_new[ks] .- r[ks]) .<= abs(dR[s]) + tol)
            end
            mixed = [s for s in eachindex(dR) if minimum(r[(2s - 1):(2s)]) < 0 < maximum(r[(2s - 1):(2s)])]
            @test sum(abs, r_new) <= sum(abs, r) + 2 * sum(abs, dR[mixed]; init = zero(FT)) + tol
            # The coarse change of a near-cancelling pair is small next to its
            # fine values, so the column's Σ|r| barely moves.
            @test sum(abs, r_new) <= FT(1.05) * sum(abs, r)
        end
    end

    @testset "conservation, signed random columns ($FT, n_merge = $nm)" for FT in (Float32, Float64), nm in (2, 3, 4)
        rng = MersenneTwister(107)
        Nx, Ny = 5, 4
        m = FT(500) .+ FT(1000) .* rand(rng, FT, Nx, Ny, Nz)
        q = randn(rng, FT, Nx, Ny, Nz, 2) .* reshape(m, Nx, Ny, Nz, 1) .* FT(1e-6)
        q0 = copy(q)
        _merged_step!(q, m, _column_forcing(FT, Nx, Ny, Nz), nm)
        @test q != q0
        for t in 1:2
            @test isapprox(sum(Float64, q[:, :, :, t]), sum(Float64, q0[:, :, :, t]);
                           atol = 64 * eps(FT) * sum(abs, q0[:, :, :, t]))
        end
    end

    @testset "uniform mixing ratio is kept ($FT)" for FT in (Float32, Float64)
        Nx, Ny = 3, 2
        m = FT(500) .+ FT(1000) .* rand(MersenneTwister(1), FT, Nx, Ny, Nz)
        q = cat(FT(4e-4) .* m, FT(-2e-9) .* m; dims = 4)
        _merged_step!(q, m, _column_forcing(FT, Nx, Ny, Nz), 2)
        @test maximum(abs, q[:, :, :, 1] ./ m ./ FT(4e-4) .- 1) <= 64 * eps(FT)
        @test maximum(abs, q[:, :, :, 2] ./ m ./ FT(-2e-9) .- 1) <= 64 * eps(FT)
    end

    @testset "non-negative tracer stays non-negative, proportional shape ($FT)" for FT in (Float32, Float64)
        Nx, Ny = 4, 3
        rng = MersenneTwister(2)
        m = FT(500) .+ FT(1000) .* rand(rng, FT, Nx, Ny, Nz)
        # Surface plume: strong gradient within the bottom pairs, zeros aloft.
        q = zeros(FT, Nx, Ny, Nz, 1)
        q[:, :, (Nz - 3):Nz, 1] .= rand(rng, FT, Nx, Ny, 4) .* FT(1e-5) .* m[:, :, (Nz - 3):Nz]
        q[:, :, Nz - 2, 1] .= 0
        q0 = copy(q)
        _merged_step!(q, m, _column_forcing(FT, Nx, Ny, Nz), 2)
        @test minimum(q) >= 0
        @test q != q0
        # Pairs that held tracer keep their fine ratio r'_k / r_k; empty pairs
        # that receive detrained tracer (layers 3-6) get a uniform mixing ratio.
        for j in 1:Ny, i in 1:Nx, s in 2:(Nz ÷ 2)
            k1, k2 = 2s - 1, 2s
            if q0[i, j, k1, 1] > 0 && q0[i, j, k2, 1] > 0
                @test isapprox(q[i, j, k1, 1] / q0[i, j, k1, 1], q[i, j, k2, 1] / q0[i, j, k2, 1];
                               rtol = 64 * eps(FT))
            elseif s in (2, 3)
                @test q[i, j, k1, 1] > 0
                @test isapprox(q[i, j, k1, 1] / m[i, j, k1], q[i, j, k2, 1] / m[i, j, k2];
                               rtol = 64 * eps(FT))
            end
        end
    end

    @testset "nearly empty super-layer receiving tracer stays finite ($FT, n_merge = $nm, lmax = $lmax)" for FT in (Float32, Float64), nm in (2, 4), lmax in (0, 8)
        Nx, Ny, Nz_receiving = 4, 3, 16
        m = FT(500) .+ FT(1000) .* rand(MersenneTwister(6), FT, Nx, Ny, Nz_receiving)
        forcing = _column_forcing(FT, Nx, Ny, Nz_receiving; top = 9)
        q = reshape(_nearly_empty_merge_tracer(m), Nx, Ny, Nz_receiving, 1)
        q0 = copy(q)
        _merged_step!(q, m, forcing, nm; L = lmax == 0 ? Nz_receiving : lmax)
        @test all(isfinite, q)
        @test minimum(q) >= 0
        @test all(q[:, :, 10, 1] .> q0[:, :, 10, 1])
        # Confirm the unsafe R_new/R_old scale would overflow in every column.
        @test all(isinf, q[:, :, 10, 1] ./ q0[:, :, 10, 1])
        @test isapprox(sum(Float64, q), sum(Float64, q0); rtol = 64 * eps(FT))
    end

    @testset "split value: depletion, empty and massless super-layers" begin
        # Arguments: r, R_old, R_new, positive sum, negative sum, m, M, 1/n.
        sv(r, Ro, Rn, pos, neg; m = 1f0, M = 2f0) =
            Cv.TM5._merged_split_value(r, Ro, Rn, pos, neg, m, M, 0.5f0)
        # A nearly emptied fine layer next to a full one: the increment form
        # rounds r + ΔR w to a tiny negative in Float32; the single-sign
        # super-layer uses R' (r/R) and stays non-negative.
        r = (1f-10, 1f10); R = r[1] + r[2]
        @test r[1] + (0f0 - R) * (r[1] / R) < 0
        for R_new in (0f0, 1f-3, 1f0)
            @test all(x -> sv(x, R, R_new, R, 0f0) >= 0, r)
        end
        @test sv(r[2], R, R, R, 0f0) == r[2]
        # A nearly empty single-sign super-layer receiving tracer: R'/R would
        # overflow Float32 (0 * Inf = NaN), the weights r/R do not.
        @test sv(0f0, 1f-40, 1f3, 1f-40, 0f0) == 0f0
        @test sv(1f-40, 1f-40, 1f3, 1f-40, 0f0) == 1f3
        @test sv(-0f0, -0f0, -0f0, 0f0, 0f0) === -0f0
        # All-zero super-layer: air-mass weights; massless: equal weights.
        @test sv(0f0, 0f0, 6f0, 0f0, 0f0; m = 1f0, M = 3f0) == 2f0
        @test sv(0f0, 0f0, 6f0, 0f0, 0f0; m = 0f0, M = 0f0) == 3f0
        @test sv(0f0, 0f0, 0f0, 0f0, 0f0; m = 0f0, M = 0f0) == 0f0
        # Mixed signs: the change is bounded by |ΔR|, whatever R.
        @test abs(Cv.TM5._merged_split_value(-1.0, 1e-9, 0.5, 1.0 + 1e-9, -1.0, 1.0, 2.0, 0.5) + 1.0) <= 0.5
        # A negative value below rounding of the positive one: Σ|r| == |R|
        # in Float32, but the pair holds both signs, so the bounded increment
        # applies and the negative is not scaled by R'/R.
        pos, neg = 1f0, -1f-8; R = pos + neg
        @test R == pos - neg
        @test sv(neg, R, 1000f0, pos, neg) > 0
        @test sv(pos, R, 1000f0, pos, neg) + sv(neg, R, 1000f0, pos, neg) ≈ 1000f0
    end

    @testset "no-convection columns and layers above the cloud are unchanged ($FT)" for FT in (Float32, Float64)
        Nx, Ny = 3, 2
        rng = MersenneTwister(3)
        m = FT(500) .+ FT(1000) .* rand(rng, FT, Nx, Ny, Nz)
        # Cloud top at layer 8: super-layers 1-2 (fine layers 1-4) lie above
        # the solved window.
        forcing = _column_forcing(FT, Nx, Ny, Nz; top = 7)
        for f in forcing
            f[1, 1, :] .= 0                       # column (1, 1) has no convection
        end
        q = randn(rng, FT, Nx, Ny, Nz, 1) .* reshape(m, Nx, Ny, Nz, 1) .* FT(1e-6)
        q[:, :, 1, 1] .= FT(3e-6) .* m[:, :, 1]   # exactly cancelling pair above the cloud
        q[:, :, 2, 1] .= .-q[:, :, 1, 1]
        q0 = copy(q)
        _merged_step!(q, m, forcing, 2)
        @test q[1, 1, :, :] == q0[1, 1, :, :]
        @test q[:, :, 1:4, :] == q0[:, :, 1:4, :]
        @test q[2:end, :, 5:end, :] != q0[2:end, :, 5:end, :]
    end

    @testset "layouts: reduced-Gaussian axis and cubed-sphere halo" begin
        FT = Float64
        rng = MersenneTwister(4)
        Nx, Ny, Hp = 3, 3, 1
        m = FT(500) .+ FT(1000) .* rand(rng, FT, Nx, Ny, Nz)
        q = randn(rng, FT, Nx, Ny, Nz, 2) .* reshape(m, Nx, Ny, Nz, 1) .* FT(1e-6)
        forcing = _column_forcing(FT, Nx, Ny, Nz)
        q_ll = _merged_step!(copy(q), m, forcing, 2)

        # Reduced Gaussian: (ncells, Nz, Nt), layer axis 2.
        nc = Nx * Ny
        q_rg = reshape(copy(q), nc, Nz, 2)
        rg(f) = reshape(f, nc, Nz)
        Cv.TM5._tm5_merge_aggregate_solve_disaggregate!(
            q_rg, rg(m), map(rg, forcing)..., 0, Nz, Nz ÷ 2, 2, 2,
            (qS, mS, eU, dU, eD, dD) -> _serial_super_solve!(3600,
                reshape(qS, Nx, Ny, Nz ÷ 2, 2), reshape(mS, Nx, Ny, :),
                (reshape(x, Nx, Ny, :) for x in (eU, dU, eD, dD))...))
        @test reshape(q_rg, Nx, Ny, Nz, 2) == q_ll

        # Cubed-sphere panel: halo of width Hp around the (Nx, Ny) interior;
        # forcings carry no halo. The halo must come back untouched.
        q_cs = fill(FT(7), Nx + 2Hp, Ny + 2Hp, Nz, 2)
        m_cs = fill(FT(900), Nx + 2Hp, Ny + 2Hp, Nz)
        q_cs[(Hp + 1):(Hp + Nx), (Hp + 1):(Hp + Ny), :, :] .= q
        m_cs[(Hp + 1):(Hp + Nx), (Hp + 1):(Hp + Ny), :] .= m
        halo0 = copy(q_cs)
        interior = (Hp + 1):(Hp + Nx), (Hp + 1):(Hp + Ny)
        Cv.TM5._tm5_merge_aggregate_solve_disaggregate!(
            q_cs, m_cs, forcing..., 0, Nz, Nz ÷ 2, 2, 3,
            (qS, mS, eU, dU, eD, dD) -> _serial_super_solve!(3600,
                view(qS, interior..., :, :), view(mS, interior..., :), eU, dU, eD, dD);
            Hp = Hp)
        @test q_cs[interior..., :, :] == q_ll
        q_cs[interior..., :, :] .= halo0[interior..., :, :]
        @test q_cs == halo0
    end

    @testset "CUDA apply!: near-cancelling pairs on the collaborative path" begin
        if !_HAS_CUDA
            @test_skip "CUDA backend required"
        else
            FT = Float32
            Nz = 16
            G = AtmosTransport.Grids
            Nx, Ny = 4, 3
            A_ifc = collect(FT, range(0, 5f4; length = Nz + 1))
            B_ifc = collect(FT, range(1, 0; length = Nz + 1))
            A_ifc[1] = 0; B_ifc[end] = 0; B_ifc[1] = 0; A_ifc[end] = 0
            grid = G.AtmosGrid(G.LatLonMesh(; Nx = Nx, Ny = Ny, FT = FT),
                               G.HybridSigmaPressure(A_ifc, B_ifc), AtmosTransport.CPU(); FT = FT)
            m = FT(500) .+ FT(1000) .* rand(MersenneTwister(6), FT, Nx, Ny, Nz)
            forcing = _column_forcing(FT, Nx, Ny, Nz; top=9)
            q = FT(1e-6) .* m
            q[:, :, 9] .= FT(1e-3) .* m[:, :, 9]                # pair (9, 10) nearly cancels
            q[:, :, 10] .= .-q[:, :, 9] .* (1 - FT(1e-6))
            q_empty = _nearly_empty_merge_tracer(m)
            for nm in (2,4,8), lmax in (0,8)
                q_input = nm == 2 ? q : abs.(q)
                state = AtmosTransport.State.CellState(m; CO2 = copy(q_input), Rn = copy(q_empty))
                ws = AtmosTransport.Operators.TM5Workspace(state.air_mass; cell_metrics = ones(FT, Ny))
                state_d = Adapt.adapt(CUDA.CuArray, state)
                ws_d = Adapt.adapt(CUDA.CuArray, ws)
                tm5_d = map(CUDA.CuArray, NamedTuple{(:entu, :detu, :entd, :detd)}(forcing))
                forcing_d = AtmosTransport.MetDrivers.ConvectionForcing(nothing, nothing, tm5_d)
                AtmosTransport.Operators.apply!(state_d, forcing_d, grid,
                    AtmosTransport.Operators.TM5Convection(use_collab_lu = true, n_merge = nm, lmax_conv = lmax),
                    FT(3600); workspace = ws_d)
                out = Array(state_d.tracers_raw)
                q_gpu = out[:, :, :, 1]
                q_cpu = _merged_step!(reshape(copy(q_input), Nx, Ny, Nz, 1), m, forcing, nm; L=lmax == 0 ? Nz : lmax)[:, :, :, 1]
                @test isapprox(sum(Float64, q_gpu), sum(Float64, q_input); rtol = 64 * eps(FT))
                @test sum(abs, q_gpu) <= FT(1.05) * sum(abs, q_input)
                @test isapprox(q_gpu, q_cpu; rtol = 64 * eps(FT), atol = 64 * eps(FT) * maximum(abs, q_cpu))
                @test all(isfinite, out[:, :, :, 2]) && minimum(out[:, :, :, 2]) >= 0
                @test isapprox(sum(Float64, out[:, :, :, 2]), sum(Float64, q_empty); rtol = 64 * eps(FT))
                if nm in (2, 4)
                    @test all(out[:, :, 10, 2] .> q_empty[:, :, 10])
                end
                if lmax == 8
                    @test q_gpu[:,:,1:8] == q_input[:,:,1:8]
                    if nm == 8
                        @test q_gpu == q_input
                    end
                end
            end
        end
    end

    @testset "CUDA broadcasts match the CPU split" begin
        if !_HAS_CUDA
            @test_skip "CUDA backend required"
        else
            FT = Float32
            rng = MersenneTwister(5)
            Nx, Ny = 6, 5
            m = FT(500) .+ FT(1000) .* rand(rng, FT, Nx, Ny, Nz)
            q = randn(rng, FT, Nx, Ny, Nz, 3) .* reshape(m, Nx, Ny, Nz, 1) .* FT(1e-6)
            forcing = _column_forcing(FT, Nx, Ny, Nz)
            q_cpu = _merged_step!(copy(q), m, forcing, 2)
            # Device arrays throughout; only the coarse column solve runs on the
            # host, so the split's reductions and broadcasts run on the GPU.
            q_gpu = CUDA.CuArray(q)
            Cv.TM5._tm5_merge_aggregate_solve_disaggregate!(
                q_gpu, CUDA.CuArray(m), map(CUDA.CuArray, forcing)..., 0, Nz, Nz ÷ 2, 2, 3,
                function (qS, args...)
                    h = Array(qS)
                    _serial_super_solve!(3600, h, map(Array, args)...)
                    copyto!(qS, h)
                end)
            @test isapprox(Array(q_gpu), q_cpu; rtol = 16 * eps(FT), atol = 16 * eps(FT) * maximum(abs, q_cpu))
        end
    end
end
