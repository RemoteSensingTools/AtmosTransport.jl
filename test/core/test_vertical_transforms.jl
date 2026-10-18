#!/usr/bin/env julia
# Plan 41 P0b — typed vertical-transform surface tests.
#
# Verifies that:
#   (1) The six concrete `AbstractVerticalTransform` types produce
#       `VerticalPlan{FT, T}` whose `merged_vc`/`merge_map`/`Nz_output`
#       are bit-exact with today's `merge_thin_levels` /
#       `select_levels_echlevs` (for the merge-map flavors).
#   (2) `MergeAbovePressure` correctly coarsens the upper atmosphere on
#       a realistic L72 hybrid coordinate while preserving troposphere /
#       stratosphere at native resolution.
#   (3) `apply_vertical!` honors the per-`FieldKind` rule:
#         * extensive (MassField / TracerMassField / MassFluxField /
#           ConvectionTendencyField) → sum native layers within group;
#         * interface (PressureFluxField / ConvectionInterfaceFlux) →
#           select kept half-level interfaces;
#         * IntensiveCenterField → mass-weighted mean within group;
#         * SurfaceField → 2D passthrough.
#   (4) Shape mismatches and `PressureOverlap.apply_vertical!` error
#       with clear messages.

using Test

import AtmosTransport
const AT  = AtmosTransport
const Pre = AtmosTransport.Preprocessing

# ---------------------------------------------------------------------------
# Small synthetic native vc — five layers, deliberately uneven thickness so
# `MergeLayersThinnerThan` and `MergeAbovePressure` actually have something
# to do.
# ---------------------------------------------------------------------------

const _TEST_A = Float64[0.0, 100.0, 500.0, 2000.0, 10000.0, 101325.0]
const _TEST_B = Float64[0.0, 0.0,   0.0,   0.0,    0.0,    1.0]

_test_vc() = Pre.HybridSigmaPressure(_TEST_A, _TEST_B)

@testset "Plan 41 P0b — typed vertical-transform surface" begin

    # -----------------------------------------------------------------------
    # IdentityVertical
    # -----------------------------------------------------------------------

    @testset "IdentityVertical: no-op plan" begin
        vc = _test_vc()
        plan = Pre.plan_vertical(Pre.IdentityVertical(), vc)
        @test plan isa Pre.VerticalPlan{Float64, Pre.IdentityVertical}
        @test plan.Nz_native == 5
        @test plan.Nz_output == 5
        @test plan.merge_map == 1:5
        @test plan.groups == [k:k for k in 1:5]
        # merged_vc IS native_vc (same reference is fine; equal coefficients required).
        @test plan.merged_vc.A == vc.A
        @test plan.merged_vc.B == vc.B
    end

    # -----------------------------------------------------------------------
    # MergeByIndex: contiguity validation + group → merged_vc construction
    # -----------------------------------------------------------------------

    @testset "MergeByIndex: explicit groups produce correct merge_map + merged_vc" begin
        vc = _test_vc()
        plan = Pre.plan_vertical(Pre.MergeByIndex([1:2, 3:5]), vc)
        @test plan.Nz_output == 2
        @test plan.merge_map == [1, 1, 2, 2, 2]
        @test plan.groups == [1:2, 3:5]
        # Merged half-levels = native indices [1, 3, 6].
        @test plan.merged_vc.A == [_TEST_A[1], _TEST_A[3], _TEST_A[6]]
        @test plan.merged_vc.B == [_TEST_B[1], _TEST_B[3], _TEST_B[6]]
    end

    @testset "MergeByIndex: validation errors on bad input" begin
        vc = _test_vc()
        # Doesn't start at 1
        @test_throws ErrorException Pre.plan_vertical(Pre.MergeByIndex([2:5]), vc)
        # Doesn't end at Nz
        @test_throws ErrorException Pre.plan_vertical(Pre.MergeByIndex([1:3]), vc)
        # Non-contiguous (gap)
        @test_throws ErrorException Pre.plan_vertical(Pre.MergeByIndex([1:2, 4:5]), vc)
        # Empty groups
        @test_throws ErrorException Pre.plan_vertical(Pre.MergeByIndex(UnitRange{Int}[]), vc)
    end

    # -----------------------------------------------------------------------
    # MergeLayersThinnerThan: bit-exact wrap of `merge_thin_levels`
    # -----------------------------------------------------------------------

    @testset "MergeLayersThinnerThan: bit-exact vs merge_thin_levels" begin
        vc = _test_vc()
        for min_thr in (1000.0, 5000.0, 30000.0)
            plan = Pre.plan_vertical(
                Pre.MergeLayersThinnerThan(min_thickness_Pa = min_thr), vc)
            merged_ref, mm_ref = Pre.merge_thin_levels(vc; min_thickness_Pa = min_thr)
            @test plan.merged_vc.A == merged_ref.A
            @test plan.merged_vc.B == merged_ref.B
            @test plan.merge_map == mm_ref
            @test plan.Nz_output == Pre.n_levels(merged_ref)
        end
    end

    # -----------------------------------------------------------------------
    # merge_thin_levels: the thickness rule, pinned on ERA5 L137 (issue #63).
    # Groups close from the surface up as soon as they reach the minimum; the
    # remnant above the last group is kept as the top layer and may be
    # thinner. At 100 Pa (the L117 CATRINE set,
    # `era5_n320_arco_diffusion_to_c90_l117_li.toml`) that top layer is 6.8 Pa;
    # in a real L117 binary (2022-01-15) at most 0.005 of its mass crosses its
    # lower interface per stored substep amount (layers 2–4: 0.012–0.05), so
    # it is numerically benign. Folding it into layer 2 (L116) changes the
    # level set and waits for the next regeneration; until then these numbers
    # must not change.
    # -----------------------------------------------------------------------

    @testset "merge_thin_levels: layer thicknesses on ERA5 L137" begin
        l137 = Pre.load_hybrid_coefficients(
            joinpath(@__DIR__, "..", "..", "config", "era5_L137_coefficients.toml"))
        ps = 101325.0
        for (min_dp, Nz, top_dp, top_native, groups) in
                ((100.0, 117, 6.8, 4, [4, 10, 4, 3, 2, 2, 2, 1]),
                 (1000.0, 68, 33.0, 9, [9, 20, 7, 5, 4, 4, 3, 3]))
            merged, mm = Pre.merge_thin_levels(l137; min_thickness_Pa = min_dp, p_surface = ps)
            dp = [Pre.level_thickness(merged, k, ps) for k in 1:Pre.n_levels(merged)]
            native = [Pre.level_thickness(l137, k, ps) for k in 1:Pre.n_levels(l137)]
            @test length(dp) == Nz
            @test round(dp[1]; digits = 1) == top_dp        # the remnant: thinner than min_dp
            @test count(==(1), mm) == top_native
            @test [count(==(l), mm) for l in 1:8] == groups
            @test all(>=(min_dp), dp[2:end])                # every other layer reaches min_dp
            # Each group closes at the first native layer that reaches min_dp.
            @test all(l -> sum(native[findall(==(l), mm)[2:end]]; init = 0.0) < min_dp, 2:Nz)
            @test issubset(merged.A, l137.A) && sum(dp) ≈ sum(native)
        end
        # Without a remnant the top group reaches the minimum as well.
        merged, mm = Pre.merge_thin_levels(_test_vc(); min_thickness_Pa = 100.0)
        @test Pre.n_levels(merged) == 5 && mm == 1:5
    end

    # -----------------------------------------------------------------------
    # MergeAbovePressure: GEOS-IT L72 mesospheric scenario
    # -----------------------------------------------------------------------

    @testset "MergeAbovePressure: L72 mesosphere coarsening" begin
        l72_path = joinpath(@__DIR__, "..", "..", "config", "geos_L72_coefficients.toml")
        vc_l72 = Pre.load_hybrid_coefficients(l72_path)
        @test Pre.n_levels(vc_l72) == 72

        # 100 Pa cutoff + 50 Pa target. The upper-mesospheric native layers
        # (~14 Pa each) get merged; troposphere/stratosphere stay native.
        plan = Pre.plan_vertical(
            Pre.MergeAbovePressure(pressure_Pa = 100.0,
                                    target_min_thickness_Pa = 50.0),
            vc_l72)
        @test plan.Nz_output < plan.Nz_native
        # The top group should swallow several native layers.
        @test length(plan.groups[1]) >= 2
        # Bottom layers should be 1-to-1 (passthrough).
        @test plan.groups[end] == 72:72
        @test plan.groups[end - 1] == 71:71
        # merge_map must be monotone non-decreasing on contiguous groups.
        for k in 1:(plan.Nz_native - 1)
            @test plan.merge_map[k + 1] >= plan.merge_map[k]
        end
        @test plan.merge_map[1] == 1
        @test plan.merge_map[end] == plan.Nz_output
    end

    @testset "MergeAbovePressure: cutoff = 0 falls back to identity" begin
        vc = _test_vc()
        # No level has midpoint pressure < 0 → eligible set is empty →
        # `plan_vertical` should return an identity plan.
        plan = Pre.plan_vertical(
            Pre.MergeAbovePressure(pressure_Pa = 0.0,
                                    target_min_thickness_Pa = 50.0), vc)
        @test plan.Nz_output == plan.Nz_native
        @test plan.merge_map == 1:5
    end

    # -----------------------------------------------------------------------
    # LevelSelection: bit-exact vs select_levels_echlevs
    # -----------------------------------------------------------------------

    @testset "LevelSelection: bit-exact vs select_levels_echlevs (CFL85)" begin
        # Build a realistic native vc for L137 (just construct uniform spacing
        # — we only need shape correctness, not real ERA5 coefficients).
        Nz_native = 137
        A = collect(range(0.0; stop = 101325.0, length = Nz_native + 1))
        B = zeros(Nz_native + 1)
        vc_native = Pre.HybridSigmaPressure(A, B)

        plan = Pre.plan_vertical(
            Pre.LevelSelection(Pre.ECHLEVS_ML137_CFL85), vc_native)
        selected_ref, mm_ref = Pre.select_levels_echlevs(
            vc_native, Pre.ECHLEVS_ML137_CFL85)
        @test plan.merged_vc.A == selected_ref.A
        @test plan.merged_vc.B == selected_ref.B
        @test plan.merge_map == mm_ref
    end

    # -----------------------------------------------------------------------
    # PressureOverlap: plan today, apply_vertical! defers to P1
    # -----------------------------------------------------------------------

    @testset "PressureOverlap: plan constructs target vc; apply errors with P1 message" begin
        vc = _test_vc()
        l72_path = joinpath(@__DIR__, "..", "..", "config", "geos_L72_coefficients.toml")
        plan = Pre.plan_vertical(Pre.PressureOverlap(l72_path), vc)
        @test plan isa Pre.VerticalPlan{Float64, Pre.PressureOverlap}
        @test plan.Nz_output == 72
        @test isempty(plan.merge_map)
        # apply_vertical! must error with the documented P1 message.
        buf_in  = zeros(Float64, 1, 1, 5)
        buf_out = zeros(Float64, 1, 1, 72)
        err = try
            Pre.apply_vertical!(buf_out, buf_in, plan, Pre.MassField())
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("P1", err.msg)
    end

    # -----------------------------------------------------------------------
    # apply_vertical! field-kind rules (extensive)
    # -----------------------------------------------------------------------

    @testset "apply_vertical!(MassField): sum native layers within group" begin
        vc = _test_vc()
        plan = Pre.plan_vertical(Pre.MergeByIndex([1:2, 3:5]), vc)
        m_native = ones(Float64, 2, 3, 5)
        m_out = zeros(Float64, 2, 3, 2)
        Pre.apply_vertical!(m_out, m_native, plan, Pre.MassField())
        # All-ones input: group 1 has 2 native layers → 2.0; group 2 has 3 → 3.0.
        @test all(m_out[:, :, 1] .== 2.0)
        @test all(m_out[:, :, 2] .== 3.0)
    end

    @testset "apply_vertical!(MassFluxField): same sum rule" begin
        vc = _test_vc()
        plan = Pre.plan_vertical(Pre.MergeByIndex([1:2, 3:5]), vc)
        am_native = reshape(collect(1.0:30.0), 2, 3, 5)
        am_out = zeros(Float64, 2, 3, 2)
        Pre.apply_vertical!(am_out, am_native, plan, Pre.MassFluxField())
        # Verify the (1,1,:) column directly: native [1, 7, 13, 19, 25]
        # → group 1 = 1+7 = 8; group 2 = 13+19+25 = 57.
        @test am_out[1, 1, 1] == 8.0
        @test am_out[1, 1, 2] == 57.0
        # Total mass conservation: sum(am_out) == sum(am_native).
        @test sum(am_out) == sum(am_native)
    end

    @testset "apply_vertical!(TracerMassField + ConvectionTendencyField): identical extensive rule" begin
        vc = _test_vc()
        plan = Pre.plan_vertical(Pre.MergeByIndex([1:2, 3:5]), vc)
        in3 = reshape(collect(1.0:30.0), 2, 3, 5)
        for kind in (Pre.TracerMassField(), Pre.ConvectionTendencyField())
            out3 = zeros(Float64, 2, 3, 2)
            Pre.apply_vertical!(out3, in3, plan, kind)
            @test sum(out3) == sum(in3)  # conservation
        end
    end

    # -----------------------------------------------------------------------
    # apply_vertical! field-kind rules (interface)
    # -----------------------------------------------------------------------

    @testset "apply_vertical!(PressureFluxField): select kept interfaces" begin
        vc = _test_vc()
        plan = Pre.plan_vertical(Pre.MergeByIndex([1:2, 3:5]), vc)
        cm_native = reshape(collect(1.0:36.0), 2, 3, 6)
        cm_out = zeros(Float64, 2, 3, 3)
        Pre.apply_vertical!(cm_out, cm_native, plan, Pre.PressureFluxField())
        # Kept native interface indices: 1, 3, 6 (boundaries of groups 1:2 and 3:5).
        # cm_native[1,1,:] = [1, 7, 13, 19, 25, 31] → out[1,1,:] = [1, 13, 31].
        @test cm_out[1, 1, :] == [1.0, 13.0, 31.0]
        # TOA + surface boundary interfaces preserved.
        @test cm_out[:, :, 1] == cm_native[:, :, 1]
        @test cm_out[:, :, end] == cm_native[:, :, end]
    end

    @testset "apply_vertical!(ConvectionInterfaceFlux): same rule as PressureFluxField" begin
        vc = _test_vc()
        plan = Pre.plan_vertical(Pre.MergeByIndex([1:2, 3:5]), vc)
        in_iface = reshape(collect(1.0:36.0), 2, 3, 6)
        out_pf = zeros(Float64, 2, 3, 3)
        out_ci = zeros(Float64, 2, 3, 3)
        Pre.apply_vertical!(out_pf, in_iface, plan, Pre.PressureFluxField())
        Pre.apply_vertical!(out_ci, in_iface, plan, Pre.ConvectionInterfaceFlux())
        @test out_pf == out_ci
    end

    # -----------------------------------------------------------------------
    # apply_vertical! field-kind rules (intensive + surface)
    # -----------------------------------------------------------------------

    @testset "apply_vertical!(IntensiveCenterField): mass-weighted mean" begin
        vc = _test_vc()
        plan = Pre.plan_vertical(Pre.MergeByIndex([1:2, 3:5]), vc)
        T_native = fill(300.0, 2, 3, 5)
        T_native[:, :, 4] .= 200.0     # one cooler layer in group 2
        # Uniform weights → simple mean.
        weights = ones(Float64, 2, 3, 5)
        T_out = zeros(Float64, 2, 3, 2)
        Pre.apply_vertical!(T_out, T_native, plan, Pre.IntensiveCenterField(), weights)
        @test all(T_out[:, :, 1] .== 300.0)      # group 1 mean = 300
        @test all(T_out[:, :, 2] .≈ (300 + 200 + 300) / 3)
        # Non-uniform weights → weighted mean.
        weights2 = ones(Float64, 2, 3, 5)
        weights2[:, :, 4] .= 9.0    # 200K layer gets weight 9
        T_out2 = zeros(Float64, 2, 3, 2)
        Pre.apply_vertical!(T_out2, T_native, plan, Pre.IntensiveCenterField(), weights2)
        @test all(T_out2[:, :, 2] .≈ (300 + 9 * 200 + 300) / 11)
    end

    @testset "apply_vertical!(SurfaceField): 2D identity passthrough" begin
        vc = _test_vc()
        plan = Pre.plan_vertical(Pre.MergeByIndex([1:2, 3:5]), vc)
        ps_in  = [1.0 2.0 3.0; 4.0 5.0 6.0]
        ps_out = zeros(2, 3)
        Pre.apply_vertical!(ps_out, ps_in, plan, Pre.SurfaceField())
        @test ps_out == ps_in
    end

    # -----------------------------------------------------------------------
    # Mesospheric end-to-end: MergeAbovePressure preserves total mass under
    # MassField apply.
    # -----------------------------------------------------------------------

    @testset "MergeAbovePressure + apply_vertical!(MassField): mass conservation" begin
        l72_path = joinpath(@__DIR__, "..", "..", "config", "geos_L72_coefficients.toml")
        vc_l72 = Pre.load_hybrid_coefficients(l72_path)
        plan = Pre.plan_vertical(
            Pre.MergeAbovePressure(pressure_Pa = 100.0,
                                    target_min_thickness_Pa = 50.0), vc_l72)
        # Random column with positive native mass.
        Nx, Ny = 2, 2
        m_native = rand(Float64, Nx, Ny, plan.Nz_native) .+ 0.1
        m_out = zeros(Float64, Nx, Ny, plan.Nz_output)
        Pre.apply_vertical!(m_out, m_native, plan, Pre.MassField())
        # Column-total mass is preserved exactly (FP summation).
        for i in 1:Nx, j in 1:Ny
            @test sum(m_out[i, j, :]) ≈ sum(m_native[i, j, :])
        end
    end

    # -----------------------------------------------------------------------
    # Shape validation
    # -----------------------------------------------------------------------

    @testset "Shape-mismatch errors are explicit" begin
        vc = _test_vc()
        plan = Pre.plan_vertical(Pre.MergeByIndex([1:2, 3:5]), vc)
        # Wrong Nz_native
        bad_in  = zeros(Float64, 2, 3, 4)
        good_out = zeros(Float64, 2, 3, 2)
        @test_throws ErrorException Pre.apply_vertical!(
            good_out, bad_in, plan, Pre.MassField())
        # Wrong Nz_output
        good_in  = zeros(Float64, 2, 3, 5)
        bad_out = zeros(Float64, 2, 3, 3)
        @test_throws ErrorException Pre.apply_vertical!(
            bad_out, good_in, plan, Pre.MassField())
        # Wrong horizontal shape
        bad_h   = zeros(Float64, 3, 3, 5)
        good_out2 = zeros(Float64, 2, 3, 2)
        @test_throws ErrorException Pre.apply_vertical!(
            good_out2, bad_h, plan, Pre.MassField())
        # Interface mismatch
        bad_iface_in = zeros(Float64, 2, 3, 5)  # should be 6 for Nz=5
        good_iface_out = zeros(Float64, 2, 3, 3)
        @test_throws ErrorException Pre.apply_vertical!(
            good_iface_out, bad_iface_in, plan, Pre.PressureFluxField())
        # Intensive shape-mismatch on weights
        good_int_in  = zeros(Float64, 2, 3, 5)
        good_int_out = zeros(Float64, 2, 3, 2)
        bad_weights  = zeros(Float64, 2, 3, 4)
        @test_throws ErrorException Pre.apply_vertical!(
            good_int_out, good_int_in, plan, Pre.IntensiveCenterField(), bad_weights)
    end
end
