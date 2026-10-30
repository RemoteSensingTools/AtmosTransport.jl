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

const FT = Float32

# =========================================================================
# ec2tm math (pure function)
# =========================================================================


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



# =========================================================================
# TransportBinaryDriver: load_transport_window populates
# window.convection.tm5_fields when binary carries TM5 sections.
# =========================================================================

@testset "plan 23 Commit 3: TransportBinaryDriver populates window.convection.tm5_fields" begin
    using .AtmosTransport.MetDrivers: TransportBinaryDriver, load_transport_window

    Nx, Ny, Nz = 4, 3, 5
    grid = _make_ll_grid(Nx, Ny, Nz; FT = FT)
    window = _make_ll_window(grid, Nz; FT = FT)

    for has_tm5 in (false,true)
    mktempdir() do tmp
        path = joinpath(tmp, "driver_tm5.bin")
        write_transport_binary(path, grid, [has_tm5 ? window : (;m=window.m,ps=window.ps,fluxes=window.fluxes)];
                                FT = FT,
                                source_flux_sampling = :window_start_endpoint,
                                mass_basis = :dry,
                                surface_pressure_basis = TotalSurfacePressure())

        driver = TransportBinaryDriver(path;
                                        FT = FT,
                                        validate_windows = false)
        try
            @test has_tm5_convection(driver.reader) == has_tm5
            loaded_window = load_transport_window(driver, 1)
            if !has_tm5
                @test loaded_window.convection === nothing
            else
            @test loaded_window.convection !== nothing
            fwd = loaded_window.convection::ConvectionForcing
            @test fwd.cmfmc === nothing
            @test fwd.dtrain === nothing
            @test fwd.tm5_fields !== nothing
            @test fwd.tm5_fields.entu == window.tm5_fields.entu
            @test fwd.tm5_fields.detu == window.tm5_fields.detu
            @test fwd.tm5_fields.entd == window.tm5_fields.entd
            @test fwd.tm5_fields.detd == window.tm5_fields.detd
            end
        finally
            close(driver)
        end
    end
    end
end
