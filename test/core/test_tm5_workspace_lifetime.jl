using Test, Adapt
using AtmosTransport
const C = AtmosTransport.Operators.Convection



@testset "Deferred adaptation preserves persistent convection data" begin
    mass = zeros(Float32, 2, 3, 8)
    ws = C.CMFMC.CMFMCMatrixWorkspace(mass; tile_columns=5, defer_scratch=true,
                               cell_metrics=ones(Float32,3))
    ws.derived_entu .= 0.03f0
    ws.derived_detu .= 0.02f0
    ws.derived_valid[] = true
    C.TM5._ensure_tm5_scratch!(ws.tm5_workspace)
    adapted = Adapt.adapt(Array, ws)
    @test adapted.derived_entu == ws.derived_entu
    @test adapted.derived_detu == ws.derived_detu
    @test adapted.derived_valid[]
    @test all(iszero, adapted.zero_entd)
    @test all(iszero, adapted.zero_detd)
    C.CMFMC.invalidate_cmfmc_matrix_cache!(adapted)
    @test ws.derived_valid[] # Sentinel ownership is independent after adaptation.
end
