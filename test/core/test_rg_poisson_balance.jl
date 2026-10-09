# Reduced-Gaussian Poisson balance: after balancing, every level's horizontal
# outflow divergence equals the mass loss per substep (minus the forward tendency
# dm/dt = (m_next − m_cur) / (2 · steps_per_window)),
#
#     div_c = Σ_out F − Σ_in F = (m_cur − m_next) / (2 · steps_per_window),
#
# the continuity the write-time replay gate checks (the palindrome applies each
# window flux twice per substep); the replay checker itself must then close
# without vertical flux. Regression test for the sign of the balance target, section A item 1 of
# docs/memos/2026-10-08_code_structure_and_duplication_plan.md.

using Test
using AtmosTransport
using .AtmosTransport.Preprocessing: build_compressed_laplacian, balance_compressed_horizontal_fluxes!
using .AtmosTransport.MetDrivers: verify_window_continuity_rg

# Outflow divergence of each cell at every level (face f carries hflux[f, k] from left to right).
function outflow_divergence(hflux, face_left, face_right, nc)
    div = zeros(nc, size(hflux, 2))
    for k in axes(hflux, 2), f in eachindex(face_left)
        div[face_left[f], k] += hflux[f, k]
        div[face_right[f], k] -= hflux[f, k]
    end
    return div
end

# Relative error of the write-time replay of the balanced fluxes with no vertical flux.
replay_error(hflux, m_cur, m_next, face_left, face_right, steps) =
    verify_window_continuity_rg(m_cur, hflux, zeros(size(m_cur, 1), size(m_cur, 2) + 1), m_next,
                                face_left, face_right, similar(m_cur), steps).max_rel_err

# Six cells (a ring and two chords), a target that sums to zero over the cells
# (reachable by interior fluxes) and arbitrary starting fluxes.
function balance_problem()
    face_left  = Int32[1, 2, 3, 4, 5, 6, 1, 2]
    face_right = Int32[2, 3, 4, 5, 6, 1, 4, 5]
    nc, Nz, steps = 6, 3, 4
    m_cur = [1e9 * (1 + 0.1c + 0.01k) for c in 1:nc, k in 1:Nz]
    tendency = [1e5 * k * sinpi(2c / nc) for c in 1:nc, k in 1:Nz]
    m_next = m_cur .- 2steps .* tendency
    hflux = [1e6 * cos(3f + k) for f in eachindex(face_left), k in 1:Nz]
    scratch = (psi = zeros(nc), rhs = zeros(nc), r = zeros(nc), p = zeros(nc), Ap = zeros(nc), z = zeros(nc))
    return (; face_left, face_right, nc, steps, m_cur, m_next, tendency, hflux, scratch)
end

@testset "RG Poisson balance reaches the continuity target" begin
    @testset "compressed Laplacian" begin
        (; face_left, face_right, nc, steps, m_cur, m_next, tendency, hflux, scratch) = balance_problem()
        L = build_compressed_laplacian(face_left, face_right, nc)
        diag = balance_compressed_horizontal_fluxes!(hflux, m_cur, m_next, face_left, face_right, L, steps, scratch)
        residual = outflow_divergence(hflux, face_left, face_right, nc) .- tendency
        @test maximum(abs, residual) <= 1e-9 * maximum(abs, tendency)
        @test diag.max_post_raw_residual <= 1e-9 * maximum(abs, tendency)   # the solver's own diagnostic agrees
        @test replay_error(hflux, m_cur, m_next, face_left, face_right, steps) <= 1e-12
    end
end
