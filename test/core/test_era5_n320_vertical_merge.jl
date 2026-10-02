using Test

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))

const Pre = AtmosTransport.Preprocessing

function _fields(Nx, Ny, Nz; offset = 0.0)
    panels3() = ntuple(p -> reshape(
        [offset + 100p + 10k + j + 0.1i
         for i in 1:Nx, j in 1:Ny, k in 1:Nz], Nx, Ny, Nz), 6)
    return Pre.ERA5C180RegridFields{Float64}(
        ntuple(p -> fill(90_000.0 + p, Nx, Ny), 6),
        panels3(), panels3(), panels3(), panels3())
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
end

@testset "ERA5 N320 target-grid L137 merge contract" begin
    Nx, Ny, Nz_native = 2, 2, 4
    vc = Pre.HybridSigmaPressure(collect(0.0:100.0:400.0), zeros(5))
    plan = Pre.plan_vertical(Pre.MergeByIndex([1:2, 3:4]), vc)

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
