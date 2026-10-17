#!/usr/bin/env julia

using Test

using AtmosTransport
using .AtmosTransport.Preprocessing: ERA5C180RegridFields,
    ERA5C180RawConvectionFields, ERA5C180TM5ConvectionFields,
    allocate_tm5_convection_column_scratch, derive_c180_tm5_convection!,
    dz_hydrostatic_virtual!, TM5CleanupStats

include(joinpath(@__DIR__, "..", "helpers", "tm5_ecconv_reference.jl"))

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
        # GRIB level k is the top of layer k; level Nz (the top of the lowest
        # layer) carries updraft and downdraft flux, as in real ERA5 columns.
        raw.udmf[p][i,j,k] = k in 2:Nz ? 0.01 * (5 - abs(3 - k)) * (1 + 0.1p) : 0
        raw.ddmf[p][i,j,k] = k in 3:Nz ? -0.004 * (6 - k) : 0
        raw.udrf[p][i,j,k] = k in 1:4 ? 1e-5 * (i + j + p) : 0
        raw.ddrf[p][i,j,k] = k in 3:Nz ? 5e-6 * (i + p) : 0
    end
    A = FT[0, 20, 100, 500, 2000, 10_000]
    B = FT[0, 0.01, 0.08, 0.3, 0.65, 1]
    vc = HybridSigmaPressure(A, B)
    scratches = [allocate_tm5_convection_column_scratch(FT, Nz)
                 for _ in 1:Threads.maxthreadid()]
    stats = TM5CleanupStats()
    derive_c180_tm5_convection!(out, raw, thermo, vc, scratches; basis = DryBasis(), stats)
    @test stats.columns_processed[] == 6 * Nc * Nc

    # Every target column equals TM5's own conversion (an independent
    # transcription of tmm.F90 + ECconv_to_TMconv + meteo.F90) of that target
    # column's already-mapped raw diagnostics, converted to dry air (issue #19:
    # GRIB level k × (1 − q at interface k, the top of layer k), centre rates
    # × (1 − q)), and its thermodynamic geometry; the fluxes rebuilt from the
    # TM5 fields are the converted GRIB fluxes, with zero through the ground.
    for (p, i, j) in ((5, 2, 1), (1, 1, 1), (3, 2, 2))
        q = vec(thermo.qv[p][i,j,:])
        q_interface = [k == 1 ? q[1] : (q[k-1] + q[k]) / 2 for k in 1:Nz]
        grib_udmf = vec(raw.udmf[p][i,j,:]) .* (1 .- q_interface)
        grib_ddmf = vec(raw.ddmf[p][i,j,:]) .* (1 .- q_interface)
        udrf = vec(raw.udrf[p][i,j,:]) .* (1 .- q)
        ddrf = vec(raw.ddrf[p][i,j,:]) .* (1 .- q)
        dz = zeros(FT, Nz)
        dz_hydrostatic_virtual!(dz, vec(thermo.t[p][i,j,:]), q, thermo.ps[p][i,j], A, B, Nz)
        expected = tm5_ecconv_reference(grib_udmf, grib_ddmf, udrf, ddrf, dz)
        column = map(f -> vec(f[p][i,j,:]), (out.entu, out.detu, out.entd, out.detd))
        for (field, reference) in zip(column, expected)
            @test field ≈ reference atol = 1e-15
        end
        @test implied_updraft_flux(column[1], column[2]) ≈ vcat(grib_udmf, 0.0) atol = 1e-15
        @test implied_downdraft_flux(column[3], column[4]) ≈ vcat(grib_ddmf, 0.0) atol = 1e-15
        # Layer 1, above the first interface with flux, is the cloud top.
        @test column[2][1] > 0
    end

    # Closure is imposed independently on every C180 column after mapping.
    for p in 1:6, j in 1:Nc, i in 1:Nc
        @test sum(out.entu[p][i,j,:]) ≈ sum(out.detu[p][i,j,:]) atol=1e-12
        @test sum(out.entd[p][i,j,:]) ≈ sum(out.detd[p][i,j,:]) atol=1e-12
        @test all(>=(0), out.entu[p][i,j,:])
        @test all(>=(0), out.detu[p][i,j,:])
    end
end
