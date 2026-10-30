#!/usr/bin/env julia
# Production vertical plans, extensive conservation, interface selection,
# mass-weighted intensive reduction, and shape contracts.

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
_merge_test_vc() = Pre.HybridSigmaPressure([0.0, 100.0, 300.0, 400.0, 500.0, 600.0], zeros(6))

@testset "Plan 41 P0b — typed vertical-transform surface" begin

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

    @testset "MergeLayersThinnerThan: production groups produce correct merge_map + merged_vc" begin
        vc = _merge_test_vc()
        plan = Pre.plan_vertical(Pre.MergeLayersThinnerThan(min_thickness_Pa = 300.0), vc)
        @test plan.Nz_output == 2
        @test plan.merge_map == [1, 1, 2, 2, 2]
        @test plan.groups == [1:2, 3:5]
        # Merged half-levels = native indices [1, 3, 6].
        @test plan.merged_vc.A == vc.A[[1, 3, 6]]
        @test plan.merged_vc.B == vc.B[[1, 3, 6]]
    end

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

    @testset "MergeAbovePressure: non-positive or NaN target thickness rejected" begin
        # The greedy merge never advances past such a threshold (infinite loop).
        for bad in (0.0, -1.0, -Inf, NaN)
            @test_throws ArgumentError Pre.MergeAbovePressure(
                pressure_Pa = 100.0, target_min_thickness_Pa = bad)
        end
        # Production config path: the constructor rejects 0 and negatives;
        # `_vertical_float` already rejects NaN and -Inf.
        for (bad, err) in ((0.0, ArgumentError), (-1.0, ArgumentError),
                           (NaN, ErrorException), ("-inf", ErrorException))
            cfg = Dict{String, Any}("transform" => "merge_above_pressure",
                                    "pressure_Pa" => 100.0,
                                    "target_min_thickness_Pa" => bad)
            @test_throws err Pre._build_native_vertical_setup(cfg, _test_vc(), Float64)
        end
        @test Pre.MergeAbovePressure(pressure_Pa = 100.0).target_min_thickness_Pa == Inf
        @test Pre.MergeAbovePressure(pressure_Pa = 100.0,
                                     target_min_thickness_Pa = 50.0).target_min_thickness_Pa == 50.0
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

    @testset "apply_vertical!(MassField): sum native layers within group" begin
        vc = _merge_test_vc()
        plan = Pre.plan_vertical(Pre.MergeLayersThinnerThan(min_thickness_Pa = 300.0), vc)
        m_native = ones(Float64, 2, 3, 5)
        m_out = zeros(Float64, 2, 3, 2)
        Pre.apply_vertical!(m_out, m_native, plan, Pre.MassField())
        # All-ones input: group 1 has 2 native layers → 2.0; group 2 has 3 → 3.0.
        @test all(m_out[:, :, 1] .== 2.0)
        @test all(m_out[:, :, 2] .== 3.0)
    end

    @testset "apply_vertical!(MassFluxField): same sum rule" begin
        vc = _merge_test_vc()
        plan = Pre.plan_vertical(Pre.MergeLayersThinnerThan(min_thickness_Pa = 300.0), vc)
        am_native = reshape(collect(1.0:30.0), 2, 3, 5)
        am_out = zeros(Float64, 2, 3, 2)
        Pre.apply_vertical!(am_out, am_native, plan, Pre.MassFluxField())
        # Verify the (1,1,:) column directly: native [1, 7, 13, 19, 25]
        # → group 1 = 1+7 = 8; group 2 = 13+19+25 = 57.
        @test am_out[1, 1, 1] == 8.0
        @test am_out[1, 1, 2] == 57.0
        # Total mass conservation: sum(am_out) == sum(am_native).
        @test sum(am_out) == sum(am_native)
        for FT in (Float32, Float64)
            big = FT === Float32 ? 2.0^24 : 2.0^53
            native = reshape(FT[big, 1, -big], 1, 1, 3)
            vc = Pre.HybridSigmaPressure([0.0, 1.0, 2.0, 3.0], zeros(4))
            merged = zeros(FT, 1, 1, 1)
            order_plan = Pre.plan_vertical(Pre.MergeLayersThinnerThan(min_thickness_Pa = 3.0), vc)
            Pre.apply_vertical!(merged, native, order_plan, Pre.MassFluxField())
            @test merged[1] === zero(FT)
        end

    end

    @testset "apply_vertical!(ConvectionTendencyField): identical extensive rule" begin
        vc = _merge_test_vc()
        plan = Pre.plan_vertical(Pre.MergeLayersThinnerThan(min_thickness_Pa = 300.0), vc)
        in3 = reshape(collect(1.0:30.0), 2, 3, 5)
        for kind in (Pre.ConvectionTendencyField(),)
            out3 = zeros(Float64, 2, 3, 2)
            Pre.apply_vertical!(out3, in3, plan, kind)
            @test sum(out3) == sum(in3)  # conservation
        end
    end

    @testset "apply_vertical!(ConvectionInterfaceFlux): select kept interfaces" begin
        vc = _merge_test_vc()
        plan = Pre.plan_vertical(Pre.MergeLayersThinnerThan(min_thickness_Pa = 300.0), vc)
        cm_native = reshape(collect(1.0:36.0), 2, 3, 6)
        cm_out = zeros(Float64, 2, 3, 3)
        Pre.apply_vertical!(cm_out, cm_native, plan, Pre.ConvectionInterfaceFlux())
        # Kept native interface indices: 1, 3, 6 (boundaries of groups 1:2 and 3:5).
        # cm_native[1,1,:] = [1, 7, 13, 19, 25, 31] → out[1,1,:] = [1, 13, 31].
        @test cm_out[1, 1, :] == [1.0, 13.0, 31.0]
        # TOA + surface boundary interfaces preserved.
        @test cm_out[:, :, 1] == cm_native[:, :, 1]
        @test cm_out[:, :, end] == cm_native[:, :, end]
    end

    @testset "apply_vertical!(IntensiveCenterField): mass-weighted mean" begin
        vc = _merge_test_vc()
        plan = Pre.plan_vertical(Pre.MergeLayersThinnerThan(min_thickness_Pa = 300.0), vc)
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

    @testset "Shape-mismatch errors are explicit" begin
        vc = _merge_test_vc()
        plan = Pre.plan_vertical(Pre.MergeLayersThinnerThan(min_thickness_Pa = 300.0), vc)
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
            good_iface_out, bad_iface_in, plan, Pre.ConvectionInterfaceFlux())
        # Intensive shape-mismatch on weights
        good_int_in  = zeros(Float64, 2, 3, 5)
        good_int_out = zeros(Float64, 2, 3, 2)
        bad_weights  = zeros(Float64, 2, 3, 4)
        @test_throws ErrorException Pre.apply_vertical!(
            good_int_out, good_int_in, plan, Pre.IntensiveCenterField(), bad_weights)
    end
end
