using Test, Adapt
using AtmosTransport
using AtmosTransport.MetDrivers: copy_convection_forcing!, allocate_convection_forcing_like

@testset "Convection forcing survives allocation, refresh and adaptation" begin
    for FT in (Float32, Float64), panels in (false, true), has_dtrain in (false, true)
        array(v, dims...) = panels ? ntuple(_ -> fill(FT(v), dims...), 6) : fill(FT(v), dims...)
        cmfmc = array(0.02, 3, 3, 5)
        dtrain = has_dtrain ? array(0.01, 3, 3, 4) : nothing
        tm5 = NamedTuple{(:entu,:detu,:entd,:detd)}(ntuple(k -> array(k/100, 3, 3, 4),4))
        src = ConvectionForcing(cmfmc, dtrain, tm5)
        dst = allocate_convection_forcing_like(src, array(0, 3, 3, 4))
        @test (panels ? map(size,dst.cmfmc) : size(dst.cmfmc)) == (panels ? map(size,src.cmfmc) : size(src.cmfmc))
        @test dst.cmfmc !== src.cmfmc
        @test (dst.dtrain === nothing) == !has_dtrain
        copy_convection_forcing!(dst, src)
        for f in (dst, Adapt.adapt(Array, dst))
            @test f.cmfmc == src.cmfmc
            @test f.dtrain == src.dtrain
            @test f.tm5_fields == src.tm5_fields
        end
    end
    # TM5-only forcing must also copy all four rates.
    src = ConvectionForcing(nothing, nothing,
        NamedTuple{(:entu,:detu,:entd,:detd)}(ntuple(k -> fill(Float32(k),3,3,4),4)))
    dst = allocate_convection_forcing_like(src, zeros(Float32,3,3,4))
    copy_convection_forcing!(dst, src)
    @test dst.tm5_fields == src.tm5_fields
    @test Adapt.adapt(Array, src).tm5_fields == src.tm5_fields

    FT = Float32
    mesh = LatLonMesh(; FT, Nx=3, Ny=3)
    grid = AtmosGrid(mesh, HybridSigmaPressure(FT[0,100,300], FT[0,0,1]), CPU(); FT)
    m, ps = ones(FT,3,3,2), ones(FT,3,3)
    fluxes = allocate_face_fluxes(mesh, 2; FT, basis=DryBasis)
    forcing = ConvectionForcing(fill(0.02f0,3,3,3), fill(0.01f0,3,3,2),
        NamedTuple{(:entu,:detu,:entd,:detd)}(ntuple(k -> fill(FT(k),3,3,2),4)))
    win = TransportWindow(m, ps, fluxes; convection=forcing)
    @test Adapt.adapt(Array, win).convection.dtrain == forcing.dtrain
    @test Adapt.adapt(Array, win).convection.tm5_fields == forcing.tm5_fields
    @test Adapt.adapt(Array, TransportWindow(m,ps,fluxes)).convection === nothing
    state = CellState(m; CO2=fill(4f-4,3,3,2))
    model = TransportModel(state, fluxes, grid, UpwindScheme();
                           convection=CMFMCConvection(), convection_forcing=forcing)
    adapted = Adapt.adapt(Array, model)
    @test adapted.convection_forcing.cmfmc == forcing.cmfmc
    @test adapted.convection_forcing.dtrain == forcing.dtrain
    @test adapted.convection_forcing.tm5_fields == forcing.tm5_fields
end

@testset "Convection refresh rejects changed capabilities" begin
    cmfmc, dtrain = zeros(3,3,5), zeros(3,3,4)
    with_dtrain = ConvectionForcing(cmfmc, dtrain, nothing)
    without_dtrain = ConvectionForcing(copy(cmfmc), nothing, nothing)
    tm5_only = ConvectionForcing(nothing, nothing,
        NamedTuple{(:entu,:detu,:entd,:detd)}(ntuple(_ -> zeros(3,3,4),4)))
    for (dst,src) in ((with_dtrain,without_dtrain), (without_dtrain,with_dtrain),
                      (tm5_only,without_dtrain))
        @test_throws ArgumentError copy_convection_forcing!(dst,src)
    end
end

@testset "cloud_base requires CMFMC and travels with the forcing" begin
    Nc, Nz = 3, 2
    cmfmc = ntuple(_ -> fill(0.01f0, Nc, Nc, Nz+1),6)
    cloud_base = ntuple(p -> fill(Float32(p),Nc,Nc),6)
    @test_throws ArgumentError ConvectionForcing(nothing, zeros(Float32,Nc,Nc,Nz), nothing)
    @test_throws ArgumentError ConvectionForcing(nothing,nothing,nothing,cloud_base)
    src = ConvectionForcing(cmfmc,nothing,nothing,cloud_base)
    dst = allocate_convection_forcing_like(src,ntuple(_ -> zeros(Float32,Nc+2,Nc+2,Nz),6))
    @test dst.cloud_base[1] !== src.cloud_base[1]
    buffer = dst.cloud_base[4]
    copy_convection_forcing!(dst,src)
    @test dst.cloud_base[4] === buffer
    @test dst.cloud_base == cloud_base
    @test Adapt.adapt(Array,src).cloud_base == cloud_base
    no_base = ConvectionForcing(ntuple(_ -> zeros(Float32,Nc,Nc,Nz+1),6),nothing,nothing)
    @test_throws ArgumentError copy_convection_forcing!(no_base,src)
end
