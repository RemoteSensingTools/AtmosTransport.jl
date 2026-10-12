#!/usr/bin/env julia

using Test

using AtmosTransport
using .AtmosTransport.Preprocessing: ERA5C180RegridFields,
    ERA5C180RawConvectionFields, ERA5C180TM5ConvectionFields,
    allocate_tm5_convection_column_scratch, derive_c180_tm5_convection!,
    ec2tm_from_rates!, dz_hydrostatic_virtual!, TM5CleanupStats

@testset "ERA convection converts after target-grid mapping" begin
    FT = Float64
    Nc, Nz = 2, 5
    panels3() = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    panels2(x) = ntuple(_ -> fill(FT(x), Nc, Nc), 6)
    thermo = ERA5C180RegridFields{FT}(
        panels2(100_000), panels3(), panels3(), panels3(), panels3())
    raw = ERA5C180RawConvectionFields{FT}(panels3(), panels3(), panels3(), panels3())
    out = ERA5C180TM5ConvectionFields{FT}(panels3(), panels3(), panels3(), panels3())

    for p in 1:6, j in 1:Nc, i in 1:Nc, k in 1:Nz
        thermo.t[p][i,j,k] = 215 + 14k + p
        thermo.qv[p][i,j,k] = 0.001k
        raw.udmf[p][i,j,k] = k in 2:4 ? 0.01 * (5 - abs(3 - k)) * (1 + 0.1p) : 0
        raw.ddmf[p][i,j,k] = k in 3:4 ? -0.004 * (5 - k) : 0
        raw.udrf[p][i,j,k] = k in 2:4 ? 1e-5 * (i + j + p) : 0
        raw.ddrf[p][i,j,k] = k in 3:4 ? 5e-6 * (i + p) : 0
    end
    A = FT[0, 20, 100, 500, 2000, 10_000]
    B = FT[0, 0.01, 0.08, 0.3, 0.65, 1]
    vc = HybridSigmaPressure(A, B)
    scratches = [allocate_tm5_convection_column_scratch(FT, Nz)
                 for _ in 1:Threads.maxthreadid()]
    stats = TM5CleanupStats()
    derive_c180_tm5_convection!(out, raw, thermo, vc, scratches; basis = DryBasis(), stats)
    @test stats.columns_processed[] == 6 * Nc * Nc

    # Every target column must equal an independent ec2tm call on that target
    # column's already-mapped raw diagnostics, converted to dry air (issue #19:
    # interface fluxes × (1 − q at the interface), centre rates × (1 − q)), and
    # thermodynamic geometry.
    p, i, j = 5, 2, 1
    q = vec(thermo.qv[p][i,j,:])
    q_interface = [k == 1 ? q[1] : k == Nz + 1 ? q[Nz] : (q[k-1] + q[k]) / 2 for k in 1:Nz+1]
    udmf = vcat(0.0, vec(raw.udmf[p][i,j,:])) .* (1 .- q_interface)
    ddmf = vcat(0.0, vec(raw.ddmf[p][i,j,:])) .* (1 .- q_interface)
    udrf = vec(raw.udrf[p][i,j,:]) .* (1 .- q)
    ddrf = vec(raw.ddrf[p][i,j,:]) .* (1 .- q)
    dz = zeros(FT, Nz)
    dz_hydrostatic_virtual!(dz, vec(thermo.t[p][i,j,:]),
                            vec(thermo.qv[p][i,j,:]), thermo.ps[p][i,j], A, B, Nz)
    expected = ntuple(_ -> zeros(FT, Nz), 4)
    ec2tm_from_rates!(expected..., udmf, ddmf, udrf, ddrf, dz, Nz)
    @test vec(out.entu[p][i,j,:]) == expected[1]
    @test vec(out.detu[p][i,j,:]) == expected[2]
    @test vec(out.entd[p][i,j,:]) == expected[3]
    @test vec(out.detd[p][i,j,:]) == expected[4]

    # Closure is imposed independently on every C180 column after mapping.
    for p in 1:6, j in 1:Nc, i in 1:Nc
        @test sum(out.entu[p][i,j,:]) ≈ sum(out.detu[p][i,j,:]) atol=1e-12
        @test sum(out.entd[p][i,j,:]) ≈ sum(out.detd[p][i,j,:]) atol=1e-12
        @test all(>=(0), out.entu[p][i,j,:])
        @test all(>=(0), out.detu[p][i,j,:])
    end
end
