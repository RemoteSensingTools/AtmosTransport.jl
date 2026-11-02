# Reduced-Gaussian column Poisson balance (`balance_compressed_column_fluxes!`).
#
# One window of `n` substeps takes `m_cur` to `m_next` through the horizontal
# outflow divergence `D_c(F_k)` and the vertical mass flux `cm`:
#
#     m_next[c, k] = m_cur[c, k] − 2n (D_c(F_k) + cm[c, k+1] − cm[c, k]).
#
# The balance closes only the column budget, D_c(Σ_k F_k) = Σ_k (m_cur − m_next)/(2n),
# and spreads each face's correction over the layers by air mass, so the part of
# the per-level imbalance with zero column sum stays in `cm` (vertical motion).
# Balancing every level to its own tendency, as the reduced-Gaussian path did
# before issue #11, forces `cm ≈ 0`.

using Test
using Random
using AtmosTransport
using .AtmosTransport.Preprocessing: build_compressed_laplacian, balance_compressed_column_fluxes!,
    spread_column_correction!, MassWeightedColumn, HybridBWeightedColumn,
    HybridMassWeightedColumn
using .AtmosTransport.MetDrivers: verify_window_continuity_rg, recompute_cm_from_dm_target!,
    faceindexed_replay_layout

# A small reduced-Gaussian-like graph: rings of 8, 12, 12 and 8 cells, zonal faces
# within rings, meridional faces between overlapping cells of adjacent rings, and
# one polar stub face (no right neighbour), which must keep zero flux.
function ring_graph()
    rings = [8, 12, 12, 8]
    starts = cumsum([0; rings[1:end-1]])
    face_left, face_right = Int32[], Int32[]
    for (r, n) in enumerate(rings), i in 1:n
        push!(face_left, starts[r] + i); push!(face_right, starts[r] + mod1(i + 1, n))
    end
    for r in 1:length(rings)-1, i in 1:rings[r], j in 1:rings[r+1]
        lo = max((i - 1) / rings[r], (j - 1) / rings[r+1])
        hi = min(i / rings[r], j / rings[r+1])
        hi > lo && (push!(face_left, starts[r] + i); push!(face_right, starts[r+1] + j))
    end
    push!(face_left, 1); push!(face_right, 0)          # polar stub
    return face_left, face_right, sum(rings)
end

function outflow_divergence(hflux, face_left, face_right, nc)
    div = zeros(nc, size(hflux, 2))
    for k in axes(hflux, 2), f in eachindex(face_left)
        face_left[f] > 0 && (div[face_left[f], k] += hflux[f, k])
        face_right[f] > 0 && (div[face_right[f], k] -= hflux[f, k])
    end
    return div
end

# `cm` from the explicit tendency (as `balance_window!` does), then the
# write-time replay of the window.
function close_window(hflux, m_cur, m_next, face_left, face_right, steps)
    nc, Nz = size(m_cur)
    cm = zeros(nc, Nz + 1)
    div = zeros(nc, Nz)
    recompute_cm_from_dm_target!(faceindexed_replay_layout(face_left, face_right), div, cm,
                                 m_cur, (m_next .- m_cur) ./ (2steps), hflux)
    replay = verify_window_continuity_rg(m_cur, hflux, cm, m_next, face_left, face_right,
                                         steps)
    return cm, replay.max_rel_err
end

new_scratch(nc) = (psi = zeros(nc), rhs = zeros(nc), r = zeros(nc), p = zeros(nc),
                   Ap = zeros(nc), z = zeros(nc))

@testset "RG column Poisson balance" begin
    face_left, face_right, nc = ring_graph()
    nf = length(face_left)
    stub = nf
    L = build_compressed_laplacian(face_left, face_right, nc)
    Nz, steps = 4, 4
    rng = Xoshiro(11)
    # Air mass increasing downward (top first), varying between cells.
    m_cur = [1e10 * k * (1 + 0.2rand(rng)) for c in 1:nc, k in 1:Nz]

    @testset "levels divergent, column balanced: cm carries the vertical motion" begin
        hflux = 1e6 .* randn(rng, nf, Nz)
        hflux[stub, :] .= 0
        # Per-level tendency = divergence + a vertical exchange e with zero column sum.
        e = 1e5 .* randn(rng, nc, Nz)
        e .-= sum(e; dims = 2) ./ Nz
        m_next = m_cur .- 2steps .* (outflow_divergence(hflux, face_left, face_right, nc) .+ e)
        hflux0 = copy(hflux)
        balance_compressed_column_fluxes!(hflux, m_cur, m_next, face_left, face_right, L, steps,
                                          new_scratch(nc))
        @test maximum(abs, hflux .- hflux0) <= 1e-9 * maximum(abs, hflux0)   # nothing to correct
        cm, replay = close_window(hflux, m_cur, m_next, face_left, face_right, steps)
        # cm[k+1] − cm[k] = e[k]: the analytic profile, not just the column residual.
        cm_expected = hcat(zeros(nc), cumsum(e; dims = 2))
        @test maximum(abs, cm .- cm_expected) <= 1e-8 * maximum(abs, e)
        # This fails under the removed per-layer balance, which gives `cm ≈ 0`.
        @test maximum(abs, cm) >= 0.5 * maximum(abs, cm_expected)
        @test maximum(abs, cm[:, Nz + 1]) <= 1e-8 * maximum(abs, e)
        @test replay <= 1e-12
    end

    @testset "column budget closes; correction spread by air mass" begin
        hflux = 1e6 .* randn(rng, nf, Nz)
        hflux[stub, :] .= 0
        tendency = 1e5 .* randn(rng, nc, Nz)
        tendency .-= sum(tendency) / length(tendency)    # reachable: zero global total
        m_next = m_cur .- 2steps .* tendency
        hflux0 = copy(hflux)
        diag = balance_compressed_column_fluxes!(hflux, m_cur, m_next, face_left, face_right, L,
                                                 steps, new_scratch(nc))
        column_div = vec(sum(outflow_divergence(hflux, face_left, face_right, nc); dims = 2))
        column_target = vec(sum(tendency; dims = 2))
        @test maximum(abs, column_div .- column_target) <= 1e-9 * maximum(abs, column_target)
        @test diag.max_post_raw_residual <= 1e-9 * maximum(abs, column_target)
        @test diag.max_pre_raw_residual > 1e3 * diag.max_post_raw_residual
        @test diag.max_face_delta > 0
        @test all(iszero, hflux[stub, :])
        # Each interior face's increments are its column correction times the
        # layer air-mass fractions of its two cells.
        for f in 1:nf-1
            l, r = face_left[f], face_right[f]
            Δ = hflux[f, :] .- hflux0[f, :]
            w = (m_cur[l, :] .+ m_cur[r, :]) ./ sum(m_cur[l, :] .+ m_cur[r, :])
            @test isapprox(Δ, sum(Δ) .* w; rtol = 1e-9, atol = 1e-9 * maximum(abs, hflux0))
        end
        cm, replay = close_window(hflux, m_cur, m_next, face_left, face_right, steps)
        @test replay <= 1e-12
        @test maximum(abs, cm[:, 1]) == 0
        # The per-level imbalance left after the column balance is vertical motion.
        @test maximum(abs, cm) > 0.1 * maximum(abs, tendency)
    end

    @testset "column weights reach the balance and the cm residual" begin
        B = [0.0, 0.0, 0.4, 0.8, 1.0]                  # layer 1 pure pressure
        weights = HybridMassWeightedColumn(B)
        hflux = 1e6 .* randn(rng, nf, Nz)
        hflux[stub, :] .= 0
        tendency = 1e5 .* randn(rng, nc, Nz)
        tendency .-= sum(tendency) / length(tendency)
        m_next = m_cur .- 2steps .* tendency
        hflux0 = copy(hflux)
        balance_compressed_column_fluxes!(hflux, m_cur, m_next, face_left, face_right, L,
                                          steps, new_scratch(nc); weights)
        @test hflux[:, 1] == hflux0[:, 1]
        @test hflux[:, 2:end] != hflux0[:, 2:end]
        # The residual of the explicit-tendency closure is folded with the
        # same weights: a pure-pressure interface keeps its integrated value.
        cm_w, cm_m = zeros(nc, Nz + 1), zeros(nc, Nz + 1)
        layout = faceindexed_replay_layout(face_left, face_right)
        dm = (m_next .- m_cur) ./ (2steps)
        dm_skew = copy(dm); dm_skew[:, end] .+= 1e3          # a column residual to fold
        recompute_cm_from_dm_target!(layout, zeros(nc, Nz), cm_m, m_cur, dm_skew, hflux)
        recompute_cm_from_dm_target!(layout, zeros(nc, Nz), cm_w, m_cur, dm_skew, hflux;
                                     residual_weight = AtmosTransport.Preprocessing.ColumnResidualWeight(weights))
        raw = zeros(nc, Nz + 1)
        div = outflow_divergence(hflux, face_left, face_right, nc)
        for c in 1:nc, k in 1:Nz
            raw[c, k + 1] = raw[c, k] - div[c, k] - dm_skew[c, k]
        end
        @test cm_w[:, 1:2] == raw[:, 1:2]
        @test cm_m[:, 2] != raw[:, 2]
        @test maximum(abs, cm_w[:, end]) <= 1e-6 * maximum(abs, raw)
        # The default residual weight is the layer mass, bit for bit.
        cm_d = zeros(nc, Nz + 1)
        recompute_cm_from_dm_target!(layout, zeros(nc, Nz), cm_d, m_cur, dm_skew, hflux;
                                     residual_weight = AtmosTransport.Preprocessing.ColumnResidualWeight(
                                         MassWeightedColumn()))
        @test cm_d == cm_m
    end

    @testset "argument checks" begin
        scratch = new_scratch(nc)
        @test_throws DimensionMismatch balance_compressed_column_fluxes!(
            zeros(nf, Nz + 1), m_cur, m_cur, face_left, face_right, L, steps, scratch)
        @test_throws DimensionMismatch balance_compressed_column_fluxes!(
            zeros(nf, Nz), m_cur, m_cur[:, 1:Nz-1], face_left, face_right, L, steps, scratch)
        @test_throws DimensionMismatch balance_compressed_column_fluxes!(
            zeros(nf, Nz), m_cur, m_cur, face_left, face_right, L, steps, scratch;
            weights = HybridBWeightedColumn([0.0, 0.5, 1.0]))
    end
end

@testset "spread_column_correction!" begin
    m_a = [1.0, 2.0, 3.0, 4.0]
    m_b = [3.0, 2.0, 1.0, 0.0]
    for FT in (Float64, Float32)
        flux = FT[1, 2, 3, 4]
        spread_column_correction!(flux, 10.0, MassWeightedColumn(), m_a, m_b)
        @test flux ≈ FT[1, 2, 3, 4] .+ FT.(10 .* (m_a .+ m_b) ./ sum(m_a .+ m_b))
        @test isapprox(sum(Float64, flux) - 10, 10; atol = 4 * eps(FT) * 10)
    end
    # Negative masses carry no weight; a correction with no eligible weight is undefined.
    flux = zeros(4)
    spread_column_correction!(flux, 4.0, MassWeightedColumn(), [-1.0, 1.0, 0.0, 0.0], zeros(4))
    @test flux == [0.0, 4.0, 0.0, 0.0]
    @test_throws DomainError spread_column_correction!(zeros(4), 4.0,
        MassWeightedColumn(), zeros(4), zeros(4))
    @test_throws DomainError spread_column_correction!(zeros(4), 4.0,
        MassWeightedColumn(), fill(NaN, 4), zeros(4))
    @test_throws DomainError spread_column_correction!(zeros(4), 4.0,
        MassWeightedColumn(), fill(Inf, 4), zeros(4))
    # Hybrid-B weights: pure-pressure layers (ΔB = 0) receive nothing.
    flux = zeros(4)
    spread_column_correction!(flux, 1.0, HybridBWeightedColumn([0.0, 0.0, 0.25, 0.5, 1.0]),
                              m_a, m_b)
    @test flux ≈ [0.0, 0.25, 0.25, 0.5]
    # The remainder goes to the last eligible layer, never outside a hybrid mask.
    flux = zeros(Float32, 4)
    spread_column_correction!(flux, 1.0, HybridMassWeightedColumn([0.0, 0.5, 1.0, 1.0, 1.0]),
                              ones(4), ones(4))
    @test flux[3:4] == zeros(Float32, 2)
    @test sum(flux) == 1.0f0
    @test_throws DomainError spread_column_correction!(zeros(4), 1.0,
        HybridMassWeightedColumn([0.0, 0.0, 0.0, 1.0, 1.0]),
        [1.0, 1.0, 0.0, 0.0], zeros(4))
end

@testset "RG global mode follows column mass" begin
    left, right, nc = ring_graph()
    nz, steps = 3, 2
    L = build_compressed_laplacian(left, right, nc)
    m = [1e12 * k * (c <= 8 ? 0.001 : 1.0) for c in 1:nc, k in 1:nz]
    next = m .* (1 + 3e-15)
    for direct in (nothing, AtmosTransport.Preprocessing.MassClosure.DirectRingPoisson(L))
        flux = zeros(length(left), nz)
        @test_throws ErrorException balance_compressed_column_fluxes!(
            flux, m, m .* (1 + 1e-10), left, right, L, 100,
            new_scratch(nc); direct_solver=direct, fixed_global_mass=true)
        balance_compressed_column_fluxes!(flux, m, next, left, right, L, steps,
                                          new_scratch(nc); tol=1e-12, direct_solver=direct, fixed_global_mass=true)
        _, error = close_window(flux, m, next, left, right, steps)
        @test error < 1e-13
        @test all(iszero, flux[end, :])
    end
end
