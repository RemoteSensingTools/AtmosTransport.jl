using Test

using AtmosTransport.Grids: HybridSigmaPressure
using AtmosTransport.Preprocessing:
    remap_extensive_field_pressure_overlap!,
    remap_qv_pressure_overlap!,
    fill_xface_surface_pressure!,
    fill_yface_surface_pressure!

@testset "pressure-overlap vertical remap" begin
    src_vc = HybridSigmaPressure([0.0, 50.0, 100.0], [0.0, 0.0, 0.0])
    dst_vc = HybridSigmaPressure([0.0, 25.0, 75.0, 100.0], [0.0, 0.0, 0.0, 0.0])
    ps = fill(90000.0, 1, 1)

    src = reshape(Float64[10.0, 20.0], 1, 1, 2)
    dst = zeros(Float64, 1, 1, 3)
    remap_extensive_field_pressure_overlap!(dst, src, ps, src_vc, dst_vc)

    @test dst[1, 1, :] ≈ [5.0, 15.0, 10.0]
    @test sum(dst) ≈ sum(src)

    q_src = reshape(Float64[1.0, 3.0], 1, 1, 2)
    q_dst = zeros(Float64, 1, 1, 3)
    remap_qv_pressure_overlap!(q_dst, q_src, src, ps, src_vc, dst_vc)

    @test q_dst[1, 1, 1] ≈ 1.0
    @test q_dst[1, 1, 2] ≈ (1.0 * 5.0 + 3.0 * 10.0) / 15.0
    @test q_dst[1, 1, 3] ≈ 3.0
end

@testset "face surface pressure reconstruction" begin
    ps = [100.0 400.0; 900.0 1600.0]
    psx = zeros(Float64, 3, 2)
    psy = zeros(Float64, 2, 3)

    fill_xface_surface_pressure!(psx, ps)
    fill_yface_surface_pressure!(psy, ps)

    @test psx[1, 1] ≈ sqrt(900.0 * 100.0)
    @test psx[2, 1] ≈ sqrt(100.0 * 900.0)
    @test psx[3, 2] ≈ sqrt(1600.0 * 400.0)

    @test psy[1, 1] ≈ 100.0
    @test psy[1, 2] ≈ sqrt(100.0 * 400.0)
    @test psy[1, 3] ≈ 400.0
end
