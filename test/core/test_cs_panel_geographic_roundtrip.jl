#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan geos-followups Section D, Commit 1.
#
# Cover the two rotation helpers in `cs_wind_rotation.jl`:
#
#   - `rotate_panel_to_geographic!`     — inverse of
#       `rotate_winds_to_panel_local!`. The forward path projects geographic
#       winds onto CS face normals, not onto non-orthogonal panel tangents.
#   - `recover_cs_cell_center_winds!`   — peer of
#       `recover_ll_cell_center_winds!`. Inverts the v4 face-flux
#       layout back to cell-center `(u, v)` for the cross-topology
#       preprocessor (CS source → LL/RG target).
#
# Tests:
#   1. Roundtrip rotation is bit-exact when fed a synthetic orthonormal
#      basis (since transpose = inverse).
#   2. Rotation by 90° (synthetic basis) gives the expected formula
#      output, catching axis-mix-up regressions.
#   3. `reconstruct_cs_fluxes!` ∘ `recover_cs_cell_center_winds!` is
#      bit-exact on a synthetic uniform mesh (Δx≡Δy≡1, dp uniform).
#   4. On a real `CubedSphereMesh`, the face-normal rotate-roundtrip is
#      near machine precision even where panel tangents are non-orthogonal.
#   5. The mesh wrapper for `rotate_panel_to_geographic!` matches the
#      explicit-tangent-basis form.
# ---------------------------------------------------------------------------

using Test

import AtmosTransport
using .AtmosTransport.Grids: CubedSphereMesh, GnomonicPanelConvention,
                              GEOSNativePanelConvention,
                              panel_cell_local_tangent_basis
using .AtmosTransport.Preprocessing: rotate_winds_to_panel_local!,
                                      rotate_panel_to_geographic!,
                                      recover_cs_cell_center_winds!,
                                      reconstruct_cs_fluxes!

const FT_TEST = Float64
const NC      = 6
const NZ      = 4
const NPANEL  = 6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_panels_3d(Nc, Nz; FT=FT_TEST) = ntuple(_ -> zeros(FT, Nc, Nc, Nz),     NPANEL)
_panels_3d_x(Nc, Nz; FT=FT_TEST) = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), NPANEL)
_panels_3d_y(Nc, Nz; FT=FT_TEST) = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), NPANEL)
_panels_2d(Nc; FT=FT_TEST) = ntuple(_ -> zeros(FT, Nc, Nc), NPANEL)

# Build a synthetic identity tangent basis: x_east=1, x_north=0,
# y_east=0, y_north=1. This is orthonormal so transpose = inverse.
function _identity_tangent_basis(Nc::Int; FT=FT_TEST)
    return ntuple(_ -> (ones(FT, Nc, Nc), zeros(FT, Nc, Nc),
                        zeros(FT, Nc, Nc), ones(FT, Nc, Nc)), NPANEL)
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "CS panel ↔ geographic helpers (Section D, Commit 1)" begin

    @testset "Roundtrip rotation: orthonormal synthetic basis" begin
        tb = _identity_tangent_basis(NC)
        ue = _panels_3d(NC, NZ); vn = _panels_3d(NC, NZ)
        for p in 1:NPANEL, k in 1:NZ, j in 1:NC, i in 1:NC
            ue[p][i, j, k] = sin(0.1 * (i + 2j + 3k + 5p))
            vn[p][i, j, k] = cos(0.1 * (i + 2j + 3k + 5p) + 0.7)
        end
        up = _panels_3d(NC, NZ); vp = _panels_3d(NC, NZ)
        ue_rt = _panels_3d(NC, NZ); vn_rt = _panels_3d(NC, NZ)

        rotate_winds_to_panel_local!(up, vp, ue, vn, tb, NC, NZ)
        rotate_panel_to_geographic!(ue_rt, vn_rt, up, vp, tb, NC, NZ)

        for p in 1:NPANEL
            @test ue_rt[p] == ue[p]
            @test vn_rt[p] == vn[p]
        end
    end

    @testset "Rotation by 90° (synthetic basis)" begin
        # Build a 90° rotation as the tangent basis: panel-x = north,
        # panel-y = -east, equivalent to x_east=0, x_north=1, y_east=-1,
        # y_north=0. Forward should map (ue=1, vn=0) → (up=0, vp=-1) and
        # reverse should invert it.
        tb = ntuple(_ -> (zeros(FT_TEST, NC, NC),
                          ones(FT_TEST,  NC, NC),
                          fill(FT_TEST(-1), NC, NC),
                          zeros(FT_TEST, NC, NC)), NPANEL)

        ue = _panels_3d(NC, NZ); vn = _panels_3d(NC, NZ)
        for p in 1:NPANEL
            fill!(ue[p], FT_TEST(1.0))
            fill!(vn[p], FT_TEST(0.0))
        end

        up = _panels_3d(NC, NZ); vp = _panels_3d(NC, NZ)
        rotate_winds_to_panel_local!(up, vp, ue, vn, tb, NC, NZ)
        for p in 1:NPANEL
            @test all(up[p] .== FT_TEST(0))
            @test all(vp[p] .== FT_TEST(-1))
        end

        ue_rt = _panels_3d(NC, NZ); vn_rt = _panels_3d(NC, NZ)
        rotate_panel_to_geographic!(ue_rt, vn_rt, up, vp, tb, NC, NZ)
        for p in 1:NPANEL
            @test ue_rt[p] == ue[p]
            @test vn_rt[p] == vn[p]
        end
    end

    @testset "Face-normal projection on a non-orthogonal synthetic basis" begin
        # Local x is east; local y is a unit vector 53.13° counter-clockwise
        # from x. The local-x face normal is therefore (0.8, -0.6) and the
        # local-y face normal is due north. A due-east geographic wind should
        # project to (0.8, 0.0), not the tangent-dot pair (1.0, 0.6).
        tb = ntuple(_ -> (ones(FT_TEST, NC, NC),
                          zeros(FT_TEST, NC, NC),
                          fill(FT_TEST(0.6), NC, NC),
                          fill(FT_TEST(0.8), NC, NC)), NPANEL)
        ue = _panels_3d(NC, NZ); vn = _panels_3d(NC, NZ)
        for p in 1:NPANEL
            fill!(ue[p], FT_TEST(1.0))
            fill!(vn[p], FT_TEST(0.0))
        end

        up = _panels_3d(NC, NZ); vp = _panels_3d(NC, NZ)
        rotate_winds_to_panel_local!(up, vp, ue, vn, tb, NC, NZ)
        for p in 1:NPANEL
            @test up[p] ≈ fill(FT_TEST(0.8), NC, NC, NZ) rtol = 1e-12
            @test vp[p] ≈ zeros(FT_TEST, NC, NC, NZ) atol = 1e-12
        end

        ue_rt = _panels_3d(NC, NZ); vn_rt = _panels_3d(NC, NZ)
        rotate_panel_to_geographic!(ue_rt, vn_rt, up, vp, tb, NC, NZ)
        for p in 1:NPANEL
            @test ue_rt[p] ≈ ue[p] rtol = 1e-12
            @test vn_rt[p] ≈ vn[p] atol = 1e-12
        end
    end

    @testset "Roundtrip rotation on real CS mesh" begin
        # The cell-local tangent basis is unit-normalized but not exactly
        # orthogonal off the panel center. The implementation derives face
        # normals from those tangents and solves the inverse Gram system, so
        # geographic -> face-normal -> geographic is a true local roundtrip.
        for conv in (GnomonicPanelConvention(), GEOSNativePanelConvention())
            mesh = CubedSphereMesh(; Nc = NC, FT = FT_TEST, convention = conv)
            tb = ntuple(p -> panel_cell_local_tangent_basis(mesh, p), NPANEL)

            ue = _panels_3d(NC, NZ); vn = _panels_3d(NC, NZ)
            for p in 1:NPANEL, k in 1:NZ, j in 1:NC, i in 1:NC
                ue[p][i, j, k] = sin(0.1 * (i + 2j + 3k + 5p))
                vn[p][i, j, k] = cos(0.1 * (i + 2j + 3k + 5p) + 0.7)
            end
            up = _panels_3d(NC, NZ); vp = _panels_3d(NC, NZ)
            ue_rt = _panels_3d(NC, NZ); vn_rt = _panels_3d(NC, NZ)

            rotate_winds_to_panel_local!(up, vp, ue, vn, tb, NC, NZ)
            rotate_panel_to_geographic!(ue_rt, vn_rt, up, vp, tb, NC, NZ)

            for p in 1:NPANEL
                @test ue_rt[p] ≈ ue[p] rtol = 1e-12 atol = 1e-12
                @test vn_rt[p] ≈ vn[p] rtol = 1e-12 atol = 1e-12
            end
        end
    end

    @testset "recover_cs_cell_center_winds! inverts uniform-mesh forward" begin
        # Synthetic uniform mesh: Δx ≡ Δy ≡ 1. Under uniform u, v, dp the
        # forward `reconstruct_cs_fluxes!` produces a constant am, bm
        # everywhere; recovery then divides by the same area_factor and
        # gets back the original (u, v) bit-exactly.
        Δx_uniform = ones(FT_TEST, NC, NC)
        Δy_uniform = ones(FT_TEST, NC, NC)

        gravity   = FT_TEST(9.80665)
        dt_factor = FT_TEST(1800.0)

        # dp_panels filled directly (reconstruct_cs_fluxes! overwrites it
        # using A, B; pick A, B so the result is uniform = 2500 Pa).
        dp_panels = _panels_3d(NC, NZ)
        u_in      = _panels_3d(NC, NZ)
        v_in      = _panels_3d(NC, NZ)
        for p in 1:NPANEL
            fill!(dp_panels[p], FT_TEST(2_500.0))
            fill!(u_in[p], FT_TEST(15.0))
            fill!(v_in[p], FT_TEST(-7.0))
        end
        ps_panels = _panels_2d(NC)
        for p in 1:NPANEL
            fill!(ps_panels[p], FT_TEST(101_325.0))
        end
        # A_ifc, B_ifc are chosen so the dp recomputed inside
        # reconstruct_cs_fluxes! comes out to 2500 Pa per layer.
        A_ifc = collect(FT_TEST.(0:NZ) .* FT_TEST(-2_500.0))
        B_ifc = zeros(FT_TEST, NZ + 1)

        am_v4 = _panels_3d_x(NC, NZ)
        bm_v4 = _panels_3d_y(NC, NZ)

        reconstruct_cs_fluxes!(am_v4, bm_v4, u_in, v_in, dp_panels, ps_panels,
                                A_ifc, B_ifc, Δx_uniform, Δy_uniform,
                                gravity, dt_factor, NC, NZ)

        u_out = _panels_3d(NC, NZ)
        v_out = _panels_3d(NC, NZ)
        recover_cs_cell_center_winds!(u_out, v_out, am_v4, bm_v4, dp_panels,
                                       Δx_uniform, Δy_uniform,
                                       gravity, dt_factor, NC, NZ)

        for p in 1:NPANEL
            @test u_out[p] ≈ u_in[p] rtol = 1e-12
            @test v_out[p] ≈ v_in[p] rtol = 1e-12
        end
    end

    @testset "rotate_panel_to_geographic! mesh wrapper matches explicit-basis form" begin
        mesh = CubedSphereMesh(; Nc = NC, FT = FT_TEST,
                                convention = GnomonicPanelConvention())
        tb = ntuple(p -> panel_cell_local_tangent_basis(mesh, p), NPANEL)

        up = _panels_3d(NC, NZ); vp = _panels_3d(NC, NZ)
        for p in 1:NPANEL, k in 1:NZ, j in 1:NC, i in 1:NC
            up[p][i, j, k] = sin(0.3 * (i + j))
            vp[p][i, j, k] = cos(0.4 * (i - j) + 0.1k)
        end

        ue_a = _panels_3d(NC, NZ); vn_a = _panels_3d(NC, NZ)
        ue_b = _panels_3d(NC, NZ); vn_b = _panels_3d(NC, NZ)

        rotate_panel_to_geographic!(ue_a, vn_a, up, vp, tb, NC, NZ)
        rotate_panel_to_geographic!(ue_b, vn_b, up, vp, mesh, NZ)

        for p in 1:NPANEL
            @test ue_a[p] == ue_b[p]
            @test vn_a[p] == vn_b[p]
        end
    end
end
