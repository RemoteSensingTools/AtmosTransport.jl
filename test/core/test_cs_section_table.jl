# One table gives the element count of every cubed-sphere binary section, for the
# writer and (through the header) the reader: cell fields Nc × Nc per level,
# x faces (Nc + 1) × Nc, y faces Nc × (Nc + 1), interface fields Nz + 1 levels,
# 2-D fields one level, all times the panel count.

using Test
using AtmosTransport

const TB = AtmosTransport.MetDrivers

@testset "cubed-sphere section element counts" begin
    Nc, np, Nz = 5, 6, 4
    cells = np * Nc * Nc
    expected = Dict(
        (s => cells * Nz for s in (:m, :dm, :dkg, :dtrain, :entu, :detu, :entd, :detd,
                                   :vdiff_u, :vdiff_v, :vdiff_t, :vdiff_qv))...,
        (s => np * (Nc + 1) * Nc * Nz for s in (:am, :dam))...,
        (s => np * Nc * (Nc + 1) * Nz for s in (:bm, :dbm))...,
        (s => cells * (Nz + 1) for s in (:cm, :dcm, :cmfmc))...,
        (s => cells for s in (:ps, :pblh, :ustar, :pbl_hflux, :t2m, :pbl_eflux, :cmfmc_cloud_base))...)
    for (section, n) in expected
        @test TB._cs_section_elements(Nc, np, Nz, section) == n
    end
    @test_throws ErrorException TB._cs_section_elements(Nc, np, Nz, :not_a_section)
end
