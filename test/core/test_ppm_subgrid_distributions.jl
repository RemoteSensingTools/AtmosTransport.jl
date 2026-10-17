#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Regression tests for the PPM cell-edge reconstruction used by the LinRood
# (ORD=5 / ORD=7) cubed-sphere advection path.
#
# History: ORD=5 (and therefore ORD=7, which reuses it) silently degenerated to
# a 2nd-order 2-point average — `huynh_second_constraint(q_im, q_i, q_i, …)`
# passed `q_i` as both centre and right edge, so the clamp collapsed to the plain
# difference `q_i - q_im` and the parabola edges became `(q_im+q_i)/2`. ORD=6's
# coefficients also summed to 58/60, breaking constant preservation. These tests
# pin the corrected 4th-order interpolation, constant/linear exactness for every
# ORD, the genuine higher-order behaviour of ORD=5/7, and forward/adjoint
# (dual-number) consistency.
# ---------------------------------------------------------------------------

using Test

import AtmosTransport
const ADV = AtmosTransport.Operators.Advection

const _edges = ADV._ppm_edge_values

@testset "PPM edge reconstruction" begin
    @testset "constant field is preserved exactly (every ORD)" begin
        for ord in (4, 5, 6, 7)
            qL, qR = _edges(1.0, 1.0, 1.0, 1.0, 1.0, Val(ord))
            @test qL ≈ 1.0
            @test qR ≈ 1.0
        end
    end

    @testset "linear field is reconstructed exactly (every ORD)" begin
        # q = x at centres x = -2..2 → edge at -1/2 is -0.5, at +1/2 is +0.5.
        for ord in (4, 5, 6, 7)
            qL, qR = _edges(-2.0, -1.0, 0.0, 1.0, 2.0, Val(ord))
            @test qL ≈ -0.5
            @test qR ≈ 0.5
        end
    end

    @testset "ORD=5/7 use the 4th-order interpolation, not 2-point averaging" begin
        # The exact 4th-order PPM weights are (-1/12, 7/12, 7/12, -1/12) for q_L
        # and (0, -1/12, 7/12, 7/12, -1/12) for q_R.
        q = (0.3, 1.1, 2.7, 1.9, 0.4)
        p1, p2 = 7 / 12, -1 / 12
        wantL = p2 * (q[1] + q[4]) + p1 * (q[2] + q[3])
        wantR = p2 * (q[2] + q[5]) + p1 * (q[3] + q[4])
        for ord in (5, 7)
            qL, qR = _edges(q..., Val(ord))
            @test qL ≈ wantL
            @test qR ≈ wantR
            # ...and decidedly NOT the degenerate 2-point average (the old bug).
            @test !(qL ≈ (q[2] + q[3]) / 2)
            @test !(qR ≈ (q[3] + q[4]) / 2)
        end

        # On a cubic the degenerate scheme returned the minmod/average value
        # (±0.5); the 4th-order scheme returns ≈0 — a concrete higher-order check.
        qL5, qR5 = _edges(-8.0, -1.0, 0.0, 1.0, 8.0, Val(5))
        @test abs(qL5) < 1e-12
        @test abs(qR5) < 1e-12
    end

    @testset "ORD=6 weights sum to one (constant preservation restored)" begin
        # Regression: the old (1/30,13/60,13/60,9/20,1/20) weights summed to 58/60.
        qL, qR = _edges(1.0, 1.0, 1.0, 1.0, 1.0, Val(6))
        @test qL ≈ 1.0
        @test qR ≈ 1.0
        # 5th-order upwind right-edge stencil (2,-13,47,27,-3)/60.
        q = (0.3, 1.1, 2.7, 1.9, 0.4)
        _, qR6 = _edges(q..., Val(6))
        @test qR6 ≈ (2q[1] - 13q[2] + 47q[3] + 27q[4] - 3q[5]) / 60
    end

    @testset "Float32 stays finite and consistent" begin
        for ord in (4, 5, 6, 7)
            qL, qR = _edges(1.0f0, 2.0f0, 3.0f0, 2.0f0, 1.0f0, Val(ord))
            @test isfinite(qL) && isfinite(qR)
            @test qL isa Float32 && qR isa Float32
        end
        # constant preserved at Float32
        for ord in (4, 5, 6, 7)
            qL, qR = _edges(5.0f0, 5.0f0, 5.0f0, 5.0f0, 5.0f0, Val(ord))
            @test qL ≈ 5.0f0
            @test qR ≈ 5.0f0
        end
    end

    @testset "ORD=5 forward/adjoint (dual-number) consistency" begin
        # The dual-number edge reconstruction must return the forward value and a
        # gradient matching a central finite difference of the forward kernel.
        q = [0.3, 1.1, 2.7, 1.9, 0.4]
        ds = ntuple(n -> ADV._d6_var(q[n], Val(n)), 5)
        qL_d6, qR_d6 = ADV._ppm_edge_values_ord5_d6(ds...)
        fL, fR = _edges(q..., Val(5))
        @test qL_d6.v ≈ fL
        @test qR_d6.v ≈ fR
        h = 1e-6
        for (slot, dual) in ((1, qL_d6), (2, qR_d6))
            for n in 1:5
                qp = copy(q); qp[n] += h
                qm = copy(q); qm[n] -= h
                fd = (_edges(qp..., Val(5))[slot] - _edges(qm..., Val(5))[slot]) / (2h)
                @test dual.g[n] ≈ fd atol = 1e-6
            end
        end
    end

    @testset "extremum flattening still applies (quasi-monotone limiter)" begin
        # _apply_monotonicity flattens the reconstruction to the cell mean when the
        # edges straddle a local extremum (q_R, q_L on opposite sides of c).
        c = 2.0
        qLf, qRf = ADV._apply_monotonicity(1.5, 1.5, c)  # both below c → extremum
        @test qLf == c
        @test qRf == c
        # monotone stencil: edges pass through unchanged
        qLp, qRp = ADV._apply_monotonicity(1.0, 3.0, c)
        @test qLp == 1.0
        @test qRp == 3.0
    end
end

# ---------------------------------------------------------------------------
# Lin-Rood ORD=7 cube-edge treatment (issue #62). The face kernels rebuild the
# parabolas of both donor cells at every face. Each cell must get the same
# parabola at its two faces: the seam value on its seam face, and edges next
# to a seam built only from the panel's own cells (FV3 `tp_core.F90` `xppm`:
# `al(1)`, `al(2) = c3 q1 + c2 q2 + c1 q3`). Previously cell 1 took the seam
# value at face 1 but a cross-seam fourth-order left edge at face 2.
#
# The donor parabola is read back from the kernel output: with unit air mass
# and a uniform flux ±a, the face value is
#   F > 0 (donor = left cell):  c + (1-a)((1-a) br - a bl)
#   F < 0 (donor = right cell): c + (1-a)((1-a) bl - a br)
# with bl = q_L - c, br = q_R - c, so two Courant fractions give (bl, br) of
# every cell as seen through its right face and through its left face.
# ---------------------------------------------------------------------------
@testset "Lin-Rood ORD=7 gives each cell one parabola (seam consistency)" begin
    Nc, Hp, Nz = 8, 3, 1
    N = Nc + 2Hp
    f(x) = exp(0.15x) + 0.1x                        # monotone: no extremum flattening
    q1d = [f(i - Hp - 0.5) for i in 1:N]            # local cell i - Hp, halos continue the profile
    c(i) = q1d[Hp + i]
    m = ones(N, N, Nz)
    backend = ADV.get_backend(m)

    function face_values(kind, dir, F)
        q = dir == 1 ? [q1d[i] for i in 1:N, j in 1:N, k in 1:Nz] :
                       [q1d[j] for i in 1:N, j in 1:N, k in 1:Nz]
        flux = dir == 1 ? fill(F, Nc + 1, Nc, Nz) : fill(F, Nc, Nc + 1, Nz)
        out = zero(flux)
        kernel! = dir == 1 ? (kind === :rm ? ADV._ppm_x_face_kernel! : ADV._ppm_x_face_from_q_kernel!) :
                             (kind === :rm ? ADV._ppm_y_face_kernel! : ADV._ppm_y_face_from_q_kernel!)
        args = kind === :rm ? (out, q, m, flux) : (out, q, flux, m)   # rm-input: (rm, m, flux)
        kernel!(backend, 64)(args..., Hp, Nc, Val(7); ndrange = size(flux))
        ADV.synchronize(backend)
        return dir == 1 ? out[:, 2, 1] : out[2, :, 1]
    end

    a1, a2 = 0.25, 0.6
    one_sided(q1, q2, q3) = 5 / 14 * q1 + 11 / 14 * q2 - 2 / 14 * q3   # FV3 c3 q1 + c2 q2 + c1 q3
    for kind in (:rm, :q), dir in (1, 2)
        @testset "$(kind)-input, $(dir == 1 ? "x" : "y") faces" begin
            vp = (face_values(kind, dir, a1), face_values(kind, dir, a2))
            vn = (face_values(kind, dir, -a1), face_values(kind, dir, -a2))
            # (bl, br) of cell i through its right face i+1 and through its left face i
            through_right(i) = [-a1 1-a1; -a2 1-a2] \ [(vp[1][i+1] - c(i)) / (1 - a1),
                                                        (vp[2][i+1] - c(i)) / (1 - a2)]
            through_left(i)  = [1-a1 -a1; 1-a2 -a2] \ [(vn[1][i] - c(i)) / (1 - a1),
                                                        (vn[2][i] - c(i)) / (1 - a2)]
            @test all(i -> isapprox(through_right(i), through_left(i); atol = 1e-12), 1:Nc)

            # Cell 1: seam value on the left, one-sided value on the right.
            bl, br = through_left(1)
            @test bl + c(1) ≈ ADV._ppm_face_edge_value_ord7_discontinuous(c(0), c(-1), c(1), c(2)) atol = 1e-12
            @test br + c(1) ≈ one_sided(c(1), c(2), c(3)) atol = 1e-12
            # Cell Nc: one-sided value on the left, seam value on the right.
            bl, br = through_right(Nc)
            @test bl + c(Nc) ≈ one_sided(c(Nc), c(Nc - 1), c(Nc - 2)) atol = 1e-12
            @test br + c(Nc) ≈ ADV._ppm_face_edge_value_ord7_discontinuous(c(Nc), c(Nc - 1),
                                                                            c(Nc + 1), c(Nc + 2)) atol = 1e-12
        end
    end
end
