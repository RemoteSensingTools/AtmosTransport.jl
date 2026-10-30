using Test

using AtmosTransport

const Pre = AtmosTransport.Preprocessing

function _fields(Nx, Ny, Nz; offset = 0.0)
    panels3() = ntuple(p -> reshape(
        [offset + 100p + 10k + j + 0.1i
         for i in 1:Nx, j in 1:Ny, k in 1:Nz], Nx, Ny, Nz), 6)
    return Pre.ERA5C180RegridFields{Float64}(
        ntuple(p -> fill(90_000.0 + p, Nx, Ny), 6),
        panels3(), panels3(), panels3(), panels3())
end

@testset "ERA5 native pin sensitivity is merged as an extensive field" begin
    Nx = Ny = 1
    native_vc = Pre.HybridSigmaPressure(
        [0.0, 1000.0, 6000.0, 6000.0], [0.0, 0.2, 0.6, 1.0])
    plan = Pre.plan_vertical(Pre.LevelSelection([0, 2, 3]), native_vc)
    @test Pre._era5_pin_sensitivity_strategy(plan) isa Pre.NativeMergedPinSensitivity
    area = fill(2.0e10, Nx, Ny)
    qv = ntuple(p -> reshape([0.01p, 0.20 + 0.01p, 0.05p], Nx, Ny, 3), 6)
    native_sensitivity = ntuple(_ -> zeros(Nx, Ny, 3), 6)
    merged_sensitivity = ntuple(_ -> zeros(Nx, Ny, 2), 6)

    Pre._fill_era5_pin_sensitivity!(native_sensitivity, qv,
                                    diff(native_vc.B), area,
                                    AtmosTransport.Parameters.STANDARD_GRAVITY)
    Pre._merge_cs_center_extensive!(merged_sensitivity, native_sensitivity,
                                    plan, Pre.MassField())
    for p in 1:6
        @test merged_sensitivity[p][1, 1, 1] ==
              native_sensitivity[p][1, 1, 1] + native_sensitivity[p][1, 1, 2]
        @test merged_sensitivity[p][1, 1, 2] == native_sensitivity[p][1, 1, 3]
    end

    m = ntuple(p -> reshape([1.0e12 + p, 2.0e12 + p], Nx, Ny, 2), 6)
    before = ntuple(p -> copy(m[p]), 6)
    delta_ps = 37.0
    target = sum(sum(Float64, panel) for panel in m) +
             delta_ps * sum(sum(Float64, panel) for panel in merged_sensitivity)
    stats = Pre._pin_cs_global_air_mass_from_sensitivity!(m, merged_sensitivity, target)
    @test stats.delta_ps_pa ≈ delta_ps rtol = 1e-12
    for p in 1:6
        @test m[p] ≈ before[p] .+ delta_ps .* merged_sensitivity[p] rtol = 2e-15
    end

    # A moist-dp-weighted merged q is not interchangeable when dA differs
    # within a group: dB_merged*(1-qbar) loses the native covariance.
    ps = 100_000.0
    dp = diff(native_vc.A) .+ diff(native_vc.B) .* ps
    p = 1
    qbar = sum(qv[p][1, 1, k] * dp[k] for k in 1:2) / sum(dp[1:2])
    old = diff(plan.merged_vc.B)[1] * (1 - qbar) * area[1, 1] /
          AtmosTransport.Parameters.STANDARD_GRAVITY
    @test old != merged_sensitivity[p][1, 1, 1]
end


@testset "pure-pressure-only merges retain compatibility strategy" begin
    vc = Pre.HybridSigmaPressure([0.0, 100.0, 300.0, 300.0],
                                 [0.0, 0.0, 0.0, 1.0])
    plan = Pre.plan_vertical(Pre.LevelSelection([0, 2, 3]), vc)
    @test Pre._era5_pin_sensitivity_strategy(plan) isa
          Pre.PurePressureMergePinCompatibility

    for FT in (Float32, Float64)
        strategy = Pre._era5_pin_sensitivity_strategy(plan)
        buffers = Pre._allocate_era5_pin_sensitivity(strategy, FT, 2, 3, 2)
        area = fill(FT(2.5e10), 2, 2)
        qv = ntuple(p -> FT[0.001p + 0.0001i + 0.0002j + 0.0003k
                             for i in 1:2, j in 1:2, k in 1:2], 6)
        initial = ntuple(p -> FT[1.0e12 + 1000p + 100i + 10j + k
                                  for i in 1:2, j in 1:2, k in 1:2], 6)
        dispatched = map(copy, initial)
        target = sum(sum(Float64, panel) for panel in initial) + 1.25e8
        gravity = FT(AtmosTransport.Parameters.STANDARD_GRAVITY)

        dispatched_stats = Pre._pin_era5_endpoint_mass!(
            strategy, dispatched, nothing, qv, buffers.native, buffers.merged,
            plan, area, gravity, target)
        @test sum(sum(Float64, panel) for panel in dispatched) ≈ target rtol = 2eps(FT)

    end


end

@testset "native ERA5 config resolves named L66 preset" begin
    coeff_path = joinpath(@__DIR__, "..", "..", "config",
                          "era5_L137_coefficients.toml")
    vc = Pre.load_hybrid_coefficients(coeff_path)
    vertical = Pre._build_native_vertical_setup(
        Dict{String, Any}(
            "transform" => "level_selection",
            "preset" => "ml137_66L",
        ),
        vc, Float32)
    @test vertical.Nz_native == 137
    @test vertical.Nz == 66
    @test length(vertical.merge_map) == 137
    @test extrema(vertical.merge_map) == (1, 66)
    @test vertical.plan.transform isa Pre.LevelSelection
    vc(n) = AtmosTransport.Grids.HybridSigmaPressure(collect(range(0.0, 0.0; length = n + 1)),
                                                     collect(range(0.0, 1.0; length = n + 1)))
    cfg = Dict{String, Any}("transform" => "level_selection", "preset" => " ml137_66L ")
    setup = Pre._build_native_vertical_setup(cfg, vc(137), Float64)
    @test setup.Nz == 66
    @test_throws ErrorException Pre._build_native_vertical_setup(cfg, vc(72), Float64)
end

@testset "ERA5 N320 target-grid L137 merge contract" begin
    Nx, Ny, Nz_native = 2, 2, 4
    vc = Pre.HybridSigmaPressure(collect(0.0:100.0:400.0), zeros(5))
    plan = Pre.plan_vertical(Pre.LevelSelection([0, 2, 4]), vc)

    native_fields = _fields(Nx, Ny, Nz_native)
    out_fields = _fields(Nx, Ny, plan.Nz_output; offset = -1.0)
    native_m = ntuple(p -> reshape(
        [1000p + 100k + 10j + i
         for i in 1:Nx, j in 1:Ny, k in 1:Nz_native], Nx, Ny, Nz_native), 6)
    native_delp_dry = ntuple(_ -> reshape(
        [5.0k + j + i for i in 1:Nx, j in 1:Ny, k in 1:Nz_native],
        Nx, Ny, Nz_native), 6)
    native_delp_moist = ntuple(_ -> reshape(
        [7.0k + 2j + i for i in 1:Nx, j in 1:Ny, k in 1:Nz_native],
        Nx, Ny, Nz_native), 6)
    out_m = ntuple(_ -> zeros(Nx, Ny, plan.Nz_output), 6)
    out_delp = ntuple(_ -> zeros(Nx, Ny, plan.Nz_output), 6)

    Pre._merge_era5_c180_state!(
        out_fields, out_m, out_delp, native_fields, native_m,
        native_delp_dry, native_delp_moist, plan)

    for p in 1:6, l in 1:plan.Nz_output, j in 1:Ny, i in 1:Nx
        group = plan.groups[l]
        @test out_m[p][i, j, l] == sum(native_m[p][i, j, k] for k in group)
        @test out_delp[p][i, j, l] ==
              sum(native_delp_dry[p][i, j, k] for k in group)
        wsum = sum(native_delp_moist[p][i, j, k] for k in group)
        expected_u = sum(native_fields.u[p][i, j, k] *
                         native_delp_moist[p][i, j, k] for k in group) / wsum
        @test out_fields.u[p][i, j, l] ≈ expected_u
        @test out_fields.ps[p][i, j] == native_fields.ps[p][i, j]
    end
    @test sum(sum, out_m) == sum(sum, native_m)
    @test sum(sum, out_delp) == sum(sum, native_delp_dry)

    native_flux = ntuple(p -> reshape(
        [100p + 10k + j + 0.1i
         for i in 1:(Nx + 1), j in 1:Ny, k in 1:Nz_native],
        Nx + 1, Ny, Nz_native), 6)
    out_flux = ntuple(_ -> zeros(Nx + 1, Ny, plan.Nz_output), 6)
    Pre._merge_cs_center_extensive!(
        out_flux, native_flux, plan, Pre.MassFluxField())
    @test sum(sum, out_flux) ≈ sum(sum, native_flux)
    for p in 1:6, l in 1:plan.Nz_output, j in 1:Ny, i in 1:(Nx + 1)
        @test out_flux[p][i, j, l] ==
              sum(native_flux[p][i, j, k] for k in plan.groups[l])
    end

    native_tm5 = Pre.ERA5C180TM5ConvectionFields{Float64}(
        native_flux, native_flux, native_flux, native_flux)
    out_tm5_panels() = ntuple(_ -> zeros(Nx + 1, Ny, plan.Nz_output), 6)
    out_tm5 = Pre.ERA5C180TM5ConvectionFields{Float64}(
        out_tm5_panels(), out_tm5_panels(), out_tm5_panels(), out_tm5_panels())
    Pre._merge_era5_tm5_fields!(out_tm5, native_tm5, plan)
    for name in (:entu, :detu, :entd, :detd)
        @test sum(sum, getproperty(out_tm5, name)) ≈
              sum(sum, getproperty(native_tm5, name))
    end
end
