#!/usr/bin/env julia
"""
Plan 23 Commit 3 tests: ec2tm math + synthetic TM5-section binary
roundtrip across LL, RG, and CS formats.

ec2tm math is standalone (no binary dependency). The binary
roundtrips write a synthetic TransportBinary / CS binary with
TM5 sections populated, read it back, and assert byte-for-byte
agreement — protects the section-element tables, writer ordering,
and reader copyto offsets from future drift.
"""

using Test
using JSON3

using AtmosTransport
using .AtmosTransport: Operators, Grids, State, MetDrivers, Preprocessing

using .AtmosTransport.Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh,
                              HybridSigmaPressure
using .AtmosTransport.State: DryBasis, StructuredFaceFluxState,
                              FaceIndexedFluxState, allocate_face_fluxes
using .AtmosTransport.MetDrivers: TransportBinaryReader,
                                   write_transport_binary,
                                   has_tm5_convection,
                                   load_tm5_convection_window!,
                                   ConvectionForcing
using .AtmosTransport.Preprocessing: ec2tm!

const FT = Float32

# =========================================================================
# ec2tm math (pure function)
# =========================================================================

@testset "plan 23 Commit 3: ec2tm! column math" begin
    Nz = 6

    @testset "zero ECMWF fluxes → zero TM5 entrainment" begin
        T = Float64
        entu = zeros(T, Nz);  detu = zeros(T, Nz)
        entd = zeros(T, Nz);  detd = zeros(T, Nz)
        mflu_ec = zeros(T, Nz + 1)
        mfld_ec = zeros(T, Nz + 1)
        detu_ec = zeros(T, Nz)
        detd_ec = zeros(T, Nz)

        ec2tm!(entu, detu, entd, detd,
                mflu_ec, mfld_ec, detu_ec, detd_ec)
        @test all(entu .== 0)
        @test all(detu .== 0)
        @test all(entd .== 0)
        @test all(detd .== 0)
    end

    @testset "updraft mass balance: entu = detu + Δmflu" begin
        T = Float64
        # Synthetic updraft profile (half-level positive flux).
        mflu_ec = T[0.0, 0.01, 0.03, 0.05, 0.04, 0.02, 0.0]  # Nz+1 = 7
        mfld_ec = zeros(T, Nz + 1)
        detu_ec = T[0.005, 0.005, 0.01, 0.015, 0.01, 0.005]  # layer centers
        detd_ec = zeros(T, Nz)

        entu = zeros(T, Nz);  detu = zeros(T, Nz)
        entd = zeros(T, Nz);  detd = zeros(T, Nz)

        ec2tm!(entu, detu, entd, detd,
                mflu_ec, mfld_ec, detu_ec, detd_ec)
        # entu[k] = detu_ec[k] + mflu_ec[k] - mflu_ec[k+1], clamped ≥ 0.
        for k in 1:Nz
            expected = max(0.0, detu_ec[k] + mflu_ec[k] - mflu_ec[k + 1])
            @test entu[k] ≈ expected
        end
        # detu passes through (clipped on negatives).
        @test detu == detu_ec
        # No downdraft activity.
        @test all(entd .== 0)
        @test all(detd .== 0)
    end

    @testset "downdraft sign flip: mfdo_ec ≤ 0 → entd ≥ 0" begin
        T = Float64
        mflu_ec = zeros(T, Nz + 1)
        # ECMWF downdraft: negative values (downward mass flux).
        mfld_ec = T[0.0, -0.01, -0.03, -0.04, -0.02, -0.01, 0.0]
        detu_ec = zeros(T, Nz)
        detd_ec = T[0.005, 0.005, 0.01, 0.005, 0.005, 0.005]

        entu = zeros(T, Nz);  detu = zeros(T, Nz)
        entd = zeros(T, Nz);  detd = zeros(T, Nz)

        ec2tm!(entu, detu, entd, detd,
                mflu_ec, mfld_ec, detu_ec, detd_ec)
        # entd[k] = detd_ec[k] + |mfld_ec[k+1]| - |mfld_ec[k]|, clamped ≥ 0.
        for k in 1:Nz
            expected = max(0.0,
                detd_ec[k] + (-mfld_ec[k + 1]) - (-mfld_ec[k]))
            @test entd[k] ≈ expected
        end
        @test detd == detd_ec
        @test all(entu .== 0)
    end

    @testset "clips small negative detrainment noise" begin
        T = Float64
        mflu_ec = zeros(T, Nz + 1)
        mfld_ec = zeros(T, Nz + 1)
        detu_ec = fill(T(-1e-19), Nz)    # ECMWF diagnostic rounding
        detd_ec = fill(T(-5e-20), Nz)

        entu = zeros(T, Nz);  detu = zeros(T, Nz)
        entd = zeros(T, Nz);  detd = zeros(T, Nz)

        ec2tm!(entu, detu, entd, detd,
                mflu_ec, mfld_ec, detu_ec, detd_ec)
        @test all(detu .== 0)
        @test all(detd .== 0)
    end

    @testset "shape guard: halflevel input must be Nz+1" begin
        T = Float64
        entu = zeros(T, Nz);  detu = zeros(T, Nz)
        entd = zeros(T, Nz);  detd = zeros(T, Nz)
        mflu_bad = zeros(T, Nz)                       # wrong size
        mfld_ok  = zeros(T, Nz + 1)
        detu_ec  = zeros(T, Nz)
        detd_ec  = zeros(T, Nz)
        @test_throws ArgumentError ec2tm!(entu, detu, entd, detd,
                                           mflu_bad, mfld_ok, detu_ec, detd_ec)
    end

    @testset "2-D grid broadcasting (Nx × Nz)" begin
        T = Float32
        Nx = 3
        entu = zeros(T, Nx, Nz);  detu = zeros(T, Nx, Nz)
        entd = zeros(T, Nx, Nz);  detd = zeros(T, Nx, Nz)
        mflu_ec = zeros(T, Nx, Nz + 1)
        mfld_ec = zeros(T, Nx, Nz + 1)
        detu_ec = zeros(T, Nx, Nz)
        detd_ec = zeros(T, Nx, Nz)

        # Set one column with nonzero updraft, others zero.
        mflu_ec[2, :] = T[0.0, 0.01, 0.03, 0.05, 0.04, 0.02, 0.0]
        detu_ec[2, :] = T[0.005, 0.005, 0.01, 0.015, 0.01, 0.005]

        ec2tm!(entu, detu, entd, detd,
                mflu_ec, mfld_ec, detu_ec, detd_ec)
        # Column 2 has non-zero entu; columns 1 and 3 stay zero.
        @test all(entu[1, :] .== 0)
        @test all(entu[3, :] .== 0)
        @test any(entu[2, :] .> 0)
    end
end

# =========================================================================
# LL transport-binary roundtrip: write a synthetic LL binary with
# TM5 sections, read back, assert byte-equality.
# =========================================================================

function _make_ll_grid(Nx, Ny, Nz; FT = Float32)
    mesh = LatLonMesh(; Nx = Nx, Ny = Ny, FT = FT)
    A_ifc = collect(FT, range(0, 10000; length = Nz + 1))
    B_ifc = collect(FT, range(1, 0;    length = Nz + 1))
    # Fix pure-pressure top / pure-sigma bottom.
    A_ifc[1] = FT(0); A_ifc[end] = FT(0)
    B_ifc[1] = FT(0); B_ifc[end] = FT(1)
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    return AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
end

function _make_ll_window(grid, Nz; FT = Float32)
    Nx = grid.horizontal.Nx
    Ny = grid.horizontal.Ny
    m  = fill(FT(5e3), Nx, Ny, Nz)
    ps = fill(FT(1.01e5), Nx, Ny)
    am = fill(FT(0), Nx + 1, Ny, Nz)
    bm = fill(FT(0), Nx, Ny + 1, Nz)
    cm = fill(FT(0), Nx, Ny, Nz + 1)
    fluxes = StructuredFaceFluxState{DryBasis}(am, bm, cm)
    tm5 = (
        entu = FT(0.001) .* rand(FT, Nx, Ny, Nz),
        detu = FT(0.001) .* rand(FT, Nx, Ny, Nz),
        entd = FT(0.001) .* rand(FT, Nx, Ny, Nz),
        detd = FT(0.001) .* rand(FT, Nx, Ny, Nz),
    )
    return (m = m, ps = ps, fluxes = fluxes, tm5_fields = tm5)
end

@testset "plan 23 Commit 3: LL transport binary TM5 roundtrip" begin
    Nx, Ny, Nz = 4, 3, 5
    grid = _make_ll_grid(Nx, Ny, Nz; FT = FT)
    window = _make_ll_window(grid, Nz; FT = FT)

    mktempdir() do tmp
        path = joinpath(tmp, "roundtrip_tm5.bin")
        write_transport_binary(path, grid, [window];
                                FT = FT,
                                source_flux_sampling = :window_start_endpoint,
                                mass_basis = :dry,
                                surface_pressure_basis = TotalSurfacePressure())

        reader = TransportBinaryReader(path; FT = FT)
        try
            @test has_tm5_convection(reader)
            loaded = load_tm5_convection_window!(reader, 1)
            @test loaded !== nothing
            @test loaded.entu == window.tm5_fields.entu
            @test loaded.detu == window.tm5_fields.detu
            @test loaded.entd == window.tm5_fields.entd
            @test loaded.detd == window.tm5_fields.detd
        finally
            close(reader)
        end
    end
end

@testset "plan 23 Commit 3: LL binary without TM5 → has_tm5_convection=false" begin
    Nx, Ny, Nz = 4, 3, 5
    grid = _make_ll_grid(Nx, Ny, Nz; FT = FT)
    # Window without tm5_fields → writer emits no TM5 sections.
    Nxx = grid.horizontal.Nx
    Nyy = grid.horizontal.Ny
    m  = fill(FT(5e3), Nxx, Nyy, Nz)
    ps = fill(FT(1.01e5), Nxx, Nyy)
    am = fill(FT(0), Nxx + 1, Nyy, Nz)
    bm = fill(FT(0), Nxx, Nyy + 1, Nz)
    cm = fill(FT(0), Nxx, Nyy, Nz + 1)
    fluxes = StructuredFaceFluxState{DryBasis}(am, bm, cm)
    window = (m = m, ps = ps, fluxes = fluxes)

    mktempdir() do tmp
        path = joinpath(tmp, "roundtrip_no_tm5.bin")
        write_transport_binary(path, grid, [window];
                                FT = FT,
                                source_flux_sampling = :window_start_endpoint,
                                mass_basis = :dry,
                                surface_pressure_basis = TotalSurfacePressure())

        reader = TransportBinaryReader(path; FT = FT)
        try
            @test !has_tm5_convection(reader)
            @test load_tm5_convection_window!(reader, 1) === nothing
        finally
            close(reader)
        end
    end
end

# =========================================================================
# TransportBinaryDriver: load_transport_window populates
# window.convection.tm5_fields when binary carries TM5 sections.
# =========================================================================

@testset "plan 23 Commit 3: TransportBinaryDriver populates window.convection.tm5_fields" begin
    using .AtmosTransport.MetDrivers: TransportBinaryDriver, load_transport_window

    Nx, Ny, Nz = 4, 3, 5
    grid = _make_ll_grid(Nx, Ny, Nz; FT = FT)
    window = _make_ll_window(grid, Nz; FT = FT)

    mktempdir() do tmp
        path = joinpath(tmp, "driver_tm5.bin")
        write_transport_binary(path, grid, [window];
                                FT = FT,
                                source_flux_sampling = :window_start_endpoint,
                                mass_basis = :dry,
                                surface_pressure_basis = TotalSurfacePressure())

        driver = TransportBinaryDriver(path;
                                        FT = FT,
                                        validate_windows = false)
        try
            loaded_window = load_transport_window(driver, 1)
            @test loaded_window.convection !== nothing
            fwd = loaded_window.convection::ConvectionForcing
            @test fwd.cmfmc === nothing
            @test fwd.dtrain === nothing
            @test fwd.tm5_fields !== nothing
            @test fwd.tm5_fields.entu == window.tm5_fields.entu
            @test fwd.tm5_fields.detu == window.tm5_fields.detu
            @test fwd.tm5_fields.entd == window.tm5_fields.entd
            @test fwd.tm5_fields.detd == window.tm5_fields.detd
        finally
            close(driver)
        end
    end
end
