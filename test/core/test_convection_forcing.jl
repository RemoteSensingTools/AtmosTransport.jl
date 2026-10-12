#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 18 Commit 2 — ConvectionForcing + window/model extensions
#
# Test coverage per plan 18 v5.1 §3 Commit 2:
#
# 1-11 struct / window / model extensions, Adapt round-trips, CMFMC+TM5
#      dual capability (Decision 28 — supersedes §2.4's "no mixing").
#      The capability tuple `_cap` is (cmfmc, dtrain, tm5_fields,
#      cloud_base) since the cloud-base slot was added (4a5daf38).
# 12-17 copy_convection_forcing! semantics including strict capability
#       match (Decision 27 invariance regression).
# 18-19 allocate_convection_forcing_like shape/backend match
#       (Decision 26).
# 20-21 with_convection / with_convection_forcing helpers.
# 22   cloud_base: requires CMFMC, is copied and allocated with the
#      forcing, and is part of the capability match.
# ---------------------------------------------------------------------------

using Test
using Adapt

using AtmosTransport
# `copy_convection_forcing!` and `allocate_convection_forcing_like` are
# exported by MetDrivers but not re-exported by AtmosTransport.
using AtmosTransport.MetDrivers: _cap, copy_convection_forcing!,
                                 allocate_convection_forcing_like

# Custom dummy convection operator for test 20 (CMFMCConvection lands
# in Commit 3).
struct _DummyConvOp <: AbstractConvection end

# ---------------------------------------------------------------------------
# Struct / capability basics
# ---------------------------------------------------------------------------

@testset "Default window construction has convection === nothing" begin
    FT = Float64
    mesh = LatLonMesh(; FT = FT, Nx = 4, Ny = 3)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    m = ones(FT, 4, 3, 2)
    ps = ones(FT, 4, 3)
    fluxes = allocate_face_fluxes(grid.horizontal, 2; FT = FT, basis = DryBasis)

    # Default kwargs — no convection
    win = TransportWindow(m, ps, fluxes)
    @test win.convection === nothing
    @test !has_convection_forcing(win)

    # Type parameter `C` binds to Nothing
    @test win isa TransportWindow{DryBasis, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing}
end

@testset "ConvectionForcing CMFMC+DTRAIN construction" begin
    FT = Float64
    cmfmc = zeros(FT, 4, 3, 5)    # (Nx, Ny, Nz+1)
    dtrain = zeros(FT, 4, 3, 4)   # (Nx, Ny, Nz)

    f = ConvectionForcing(cmfmc, dtrain, nothing)
    @test f.cmfmc === cmfmc
    @test f.dtrain === dtrain
    @test f.tm5_fields === nothing
    @test has_convection_forcing(f)
    @test _cap(f) == (true, true, false, false)
end

@testset "ConvectionForcing TM5-mode construction" begin
    FT = Float64
    entu = zeros(FT, 4, 3, 4)
    detu = zeros(FT, 4, 3, 4)
    entd = zeros(FT, 4, 3, 4)
    detd = zeros(FT, 4, 3, 4)
    tm5 = (; entu = entu, detu = detu, entd = entd, detd = detd)

    f = ConvectionForcing(nothing, nothing, tm5)
    @test f.cmfmc === nothing
    @test f.dtrain === nothing
    @test f.tm5_fields === tm5
    @test has_convection_forcing(f)
    @test _cap(f) == (false, false, true, false)
end

@testset "ConvectionForcing dual-capability (Decision 28)" begin
    # §2.4 legacy language said "no mixing CMFMC with TM5", but
    # Decision 28 explicitly allows dual-capability binaries. The
    # inner constructor permits all combinations as long as DTRAIN
    # has CMFMC alongside it.
    FT = Float64
    cmfmc = zeros(FT, 4, 3, 5)
    dtrain = zeros(FT, 4, 3, 4)
    tm5 = (; entu = zeros(FT, 4, 3, 4), detu = zeros(FT, 4, 3, 4),
             entd = zeros(FT, 4, 3, 4), detd = zeros(FT, 4, 3, 4))

    f = ConvectionForcing(cmfmc, dtrain, tm5)
    @test _cap(f) == (true, true, true, false)
    @test has_convection_forcing(f)
end

@testset "ConvectionForcing CMFMC-only Tiedtke fallback is valid" begin
    # Per Decision 2: `dtrain === nothing` with cmfmc present triggers
    # Tiedtke-style single-flux transport in CMFMCConvection. The
    # struct constructor must accept this.
    FT = Float64
    cmfmc = zeros(FT, 4, 3, 5)
    f = ConvectionForcing(cmfmc, nothing, nothing)
    @test _cap(f) == (true, false, false, false)
    @test has_convection_forcing(f)
end

@testset "ConvectionForcing invariant: DTRAIN requires CMFMC" begin
    FT = Float64
    dtrain = zeros(FT, 4, 3, 4)
    # DTRAIN without CMFMC is meaningless; must throw.
    @test_throws ArgumentError ConvectionForcing(nothing, dtrain, nothing)
end

@testset "has_convection_forcing: struct + window overloads" begin
    @test !has_convection_forcing(ConvectionForcing())

    FT = Float64
    cmfmc = zeros(FT, 4, 3, 5)
    @test has_convection_forcing(ConvectionForcing(cmfmc, nothing, nothing))

    # Window-level overload
    mesh = LatLonMesh(; FT = FT, Nx = 4, Ny = 3)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
    m = ones(FT, 4, 3, 2); ps = ones(FT, 4, 3)
    fluxes = allocate_face_fluxes(grid.horizontal, 2; FT = FT, basis = DryBasis)

    win_none = TransportWindow(m, ps, fluxes)
    @test !has_convection_forcing(win_none)

    win_with = TransportWindow(m, ps, fluxes;
                                          convection = ConvectionForcing(cmfmc, nothing, nothing))
    @test has_convection_forcing(win_with)
end

# ---------------------------------------------------------------------------
# Adapt round-trips
# ---------------------------------------------------------------------------

@testset "Adapt.adapt_structure: ConvectionForcing CPU→CPU preserves shape" begin
    FT = Float64
    cmfmc = zeros(FT, 4, 3, 5)
    dtrain = zeros(FT, 4, 3, 4)
    f = ConvectionForcing(cmfmc, dtrain, nothing)

    # Adapt to Array — should be a no-op round-trip (still Array-backed)
    f2 = Adapt.adapt(Array, f)
    @test _cap(f2) == _cap(f)
    @test size(f2.cmfmc) == size(cmfmc)
    @test size(f2.dtrain) == size(dtrain)
    @test f2.tm5_fields === nothing

    # All-nothing placeholder round-trips as all-nothing
    f_none = ConvectionForcing()
    f_none2 = Adapt.adapt(Array, f_none)
    @test _cap(f_none2) == (false, false, false, false)

    # TM5 mode round-trips the NamedTuple
    tm5 = (; entu = zeros(FT, 4, 3, 4), detu = zeros(FT, 4, 3, 4),
             entd = zeros(FT, 4, 3, 4), detd = zeros(FT, 4, 3, 4))
    f_tm5 = ConvectionForcing(nothing, nothing, tm5)
    f_tm5_2 = Adapt.adapt(Array, f_tm5)
    @test f_tm5_2.tm5_fields isa NamedTuple
    @test keys(f_tm5_2.tm5_fields) == (:entu, :detu, :entd, :detd)
end

@testset "Adapt.adapt_structure: TransportWindow preserves convection" begin
    FT = Float64
    mesh = LatLonMesh(; FT = FT, Nx = 4, Ny = 3)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    m = ones(FT, 4, 3, 2); ps = ones(FT, 4, 3)
    fluxes = allocate_face_fluxes(grid.horizontal, 2; FT = FT, basis = DryBasis)
    cmfmc = zeros(FT, 4, 3, 3)
    forcing = ConvectionForcing(cmfmc, nothing, nothing)

    win = TransportWindow(m, ps, fluxes; convection = forcing)
    win2 = Adapt.adapt(Array, win)
    @test win2.convection !== nothing
    @test _cap(win2.convection) == (true, false, false, false)
    @test size(win2.convection.cmfmc) == size(cmfmc)

    # Round-trip of a `convection === nothing` window stays `nothing`.
    win_none = TransportWindow(m, ps, fluxes)
    win_none2 = Adapt.adapt(Array, win_none)
    @test win_none2.convection === nothing
end

@testset "Adapt.adapt_structure: TransportModel preserves convection + forcing" begin
    FT = Float64
    mesh = LatLonMesh(; FT = FT, Nx = 4, Ny = 3)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
    state = CellState(ones(FT, 4, 3, 2); CO2 = fill(FT(400e-6), 4, 3, 2))
    fluxes = allocate_face_fluxes(grid.horizontal, 2; FT = FT, basis = DryBasis)

    cmfmc = zeros(FT, 4, 3, 3)
    forcing = ConvectionForcing(cmfmc, nothing, nothing)

    model = TransportModel(state, fluxes, grid, UpwindScheme();
                            convection = _DummyConvOp(),
                            convection_forcing = forcing)

    model2 = Adapt.adapt(Array, model)
    @test model2.convection isa _DummyConvOp
    @test _cap(model2.convection_forcing) == (true, false, false, false)
    @test size(model2.convection_forcing.cmfmc) == size(cmfmc)
end

# ---------------------------------------------------------------------------
# Face-indexed window also carries convection
# ---------------------------------------------------------------------------

@testset "TransportWindow supports convection field" begin
    FT = Float64
    ncell = 8
    Nz = 2
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT = FT)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    m = ones(FT, ncell, Nz)
    ps = ones(FT, ncell)
    fluxes = allocate_face_fluxes(grid.horizontal, Nz; FT = FT, basis = MoistBasis)

    win_none = TransportWindow(m, ps, fluxes)
    @test win_none.convection === nothing
    @test !has_convection_forcing(win_none)

    # With an explicit convection forcing (face-indexed payloads are
    # (ncell, Nz+1) for cmfmc, (ncell, Nz) for dtrain — but the struct
    # doesn't shape-check, so a placeholder CMFMC works for the field-
    # carrying test).
    cmfmc = zeros(FT, ncell, Nz + 1)
    forcing = ConvectionForcing(cmfmc, nothing, nothing)
    win_with = TransportWindow(m, ps, fluxes; convection = forcing)
    @test has_convection_forcing(win_with)
    @test _cap(win_with.convection) == (true, false, false, false)
end

# ---------------------------------------------------------------------------
# copy_convection_forcing! semantics (Decision 27 invariance)
# ---------------------------------------------------------------------------

@testset "copy_convection_forcing!: CMFMC+DTRAIN match" begin
    FT = Float64
    cmfmc_src = fill(FT(1.0), 4, 3, 5)
    dtrain_src = fill(FT(0.5), 4, 3, 4)
    src = ConvectionForcing(cmfmc_src, dtrain_src, nothing)

    cmfmc_dst = zeros(FT, 4, 3, 5)
    dtrain_dst = zeros(FT, 4, 3, 4)
    dst = ConvectionForcing(cmfmc_dst, dtrain_dst, nothing)

    result = copy_convection_forcing!(dst, src)
    @test result === dst
    @test dst.cmfmc == cmfmc_src
    @test dst.dtrain == dtrain_src
    # Destination arrays are the SAME buffers (identity preserved)
    @test dst.cmfmc === cmfmc_dst
    @test dst.dtrain === dtrain_dst
end

@testset "copy_convection_forcing!: CMFMC-only (no dtrain) match" begin
    FT = Float64
    cmfmc_src = fill(FT(2.0), 4, 3, 5)
    src = ConvectionForcing(cmfmc_src, nothing, nothing)

    cmfmc_dst = zeros(FT, 4, 3, 5)
    dst = ConvectionForcing(cmfmc_dst, nothing, nothing)

    copy_convection_forcing!(dst, src)
    @test dst.cmfmc == cmfmc_src
    @test dst.cmfmc === cmfmc_dst
end

@testset "copy_convection_forcing!: cubed-sphere panel tuples" begin
    FT = Float32
    Nc, Nz = 4, 2
    cmfmc_src = ntuple(_ -> fill(FT(1), Nc, Nc, Nz + 1), 6)
    dtrain_src = ntuple(_ -> fill(FT(0.5), Nc, Nc, Nz), 6)
    src = ConvectionForcing(cmfmc_src, dtrain_src, nothing)

    cmfmc_dst = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
    dtrain_dst = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    dst = ConvectionForcing(cmfmc_dst, dtrain_dst, nothing)

    copy_convection_forcing!(dst, src)
    @test dst.cmfmc[2] == cmfmc_src[2]
    @test dst.dtrain[5] == dtrain_src[5]
    @test dst.cmfmc[1] === cmfmc_dst[1]
    @test dst.dtrain[6] === dtrain_dst[6]
end

@testset "copy_convection_forcing!: TM5 fields copy via NamedTuple loop" begin
    FT = Float64
    src = ConvectionForcing(nothing, nothing,
        (; entu = fill(FT(1.0), 4, 3, 4),
           detu = fill(FT(2.0), 4, 3, 4),
           entd = fill(FT(3.0), 4, 3, 4),
           detd = fill(FT(4.0), 4, 3, 4)))
    dst = ConvectionForcing(nothing, nothing,
        (; entu = zeros(FT, 4, 3, 4),
           detu = zeros(FT, 4, 3, 4),
           entd = zeros(FT, 4, 3, 4),
           detd = zeros(FT, 4, 3, 4)))

    copy_convection_forcing!(dst, src)
    @test all(dst.tm5_fields.entu .== FT(1.0))
    @test all(dst.tm5_fields.detu .== FT(2.0))
    @test all(dst.tm5_fields.entd .== FT(3.0))
    @test all(dst.tm5_fields.detd .== FT(4.0))
end

@testset "copy_convection_forcing! mismatch: src has dtrain, dst doesn't" begin
    # Silent-stale-values regression: dst without dtrain accepting a
    # src with dtrain would silently drop dtrain updates.
    FT = Float64
    cmfmc_src = fill(FT(1.0), 4, 3, 5)
    dtrain_src = fill(FT(0.5), 4, 3, 4)
    src = ConvectionForcing(cmfmc_src, dtrain_src, nothing)

    cmfmc_dst = zeros(FT, 4, 3, 5)
    dst = ConvectionForcing(cmfmc_dst, nothing, nothing)

    @test_throws ArgumentError copy_convection_forcing!(dst, src)
end

@testset "copy_convection_forcing! mismatch: dst has dtrain, src doesn't" begin
    # Missing-destination regression: if src loses its dtrain (e.g.,
    # because the binary schema changed mid-run), dst's stale
    # preallocated dtrain would persist unchanged. Decision 27 says
    # that's impossible; this test enforces it.
    FT = Float64
    cmfmc_src = fill(FT(1.0), 4, 3, 5)
    src = ConvectionForcing(cmfmc_src, nothing, nothing)

    cmfmc_dst = zeros(FT, 4, 3, 5)
    dtrain_dst = zeros(FT, 4, 3, 4)
    dst = ConvectionForcing(cmfmc_dst, dtrain_dst, nothing)

    @test_throws ArgumentError copy_convection_forcing!(dst, src)
end

@testset "copy_convection_forcing! mismatch: CMFMC vs TM5 mode" begin
    FT = Float64
    src = ConvectionForcing(fill(FT(1.0), 4, 3, 5), nothing, nothing)
    dst = ConvectionForcing(nothing, nothing,
        (; entu = zeros(FT, 4, 3, 4), detu = zeros(FT, 4, 3, 4),
           entd = zeros(FT, 4, 3, 4), detd = zeros(FT, 4, 3, 4)))

    @test_throws ArgumentError copy_convection_forcing!(dst, src)
end

# ---------------------------------------------------------------------------
# allocate_convection_forcing_like (Decision 26)
# ---------------------------------------------------------------------------

@testset "allocate_convection_forcing_like: identical capability + shape" begin
    FT = Float64
    cmfmc = zeros(FT, 4, 3, 5)
    dtrain = zeros(FT, 4, 3, 4)
    src = ConvectionForcing(cmfmc, dtrain, nothing)

    backend_hint = zeros(FT, 4, 3, 2)   # CPU Array
    dst = allocate_convection_forcing_like(src, backend_hint)

    @test _cap(dst) == _cap(src)
    @test size(dst.cmfmc) == size(cmfmc)
    @test size(dst.dtrain) == size(dtrain)
    # `similar()` produces a FRESH buffer, not the same reference
    @test dst.cmfmc !== src.cmfmc
    @test dst.dtrain !== src.dtrain
    @test dst.tm5_fields === nothing
    @test dst.cmfmc isa Array{FT, 3}
    @test dst.dtrain isa Array{FT, 3}
end

@testset "allocate_convection_forcing_like: dtrain-absent src → dst" begin
    FT = Float64
    cmfmc = zeros(FT, 4, 3, 5)
    src = ConvectionForcing(cmfmc, nothing, nothing)

    dst = allocate_convection_forcing_like(src, zeros(FT, 4, 3, 2))
    @test _cap(dst) == (true, false, false, false)
    @test dst.dtrain === nothing
    @test size(dst.cmfmc) == size(cmfmc)
end

@testset "allocate_convection_forcing_like: cubed-sphere panel tuples" begin
    FT = Float32
    Nc, Nz = 4, 2
    cmfmc = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
    dtrain = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    src = ConvectionForcing(cmfmc, dtrain, nothing)

    backend_hint = ntuple(_ -> zeros(FT, Nc + 2, Nc + 2, Nz), 6)
    dst = allocate_convection_forcing_like(src, backend_hint)

    @test _cap(dst) == _cap(src)
    @test size(dst.cmfmc[1]) == size(cmfmc[1])
    @test size(dst.dtrain[1]) == size(dtrain[1])
    @test dst.cmfmc[1] !== src.cmfmc[1]
    @test dst.dtrain[1] !== src.dtrain[1]
    @test eltype(dst.cmfmc[1]) === FT
end

# ---------------------------------------------------------------------------
# with_convection / with_convection_forcing helpers
# ---------------------------------------------------------------------------

@testset "with_convection preserves convection_forcing (no reallocation)" begin
    FT = Float64
    mesh = LatLonMesh(; FT = FT, Nx = 4, Ny = 3)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
    state = CellState(ones(FT, 4, 3, 2); CO2 = fill(FT(400e-6), 4, 3, 2))
    fluxes = allocate_face_fluxes(grid.horizontal, 2; FT = FT, basis = DryBasis)
    model = TransportModel(state, fluxes, grid, UpwindScheme())

    # Default: NoConvection + ConvectionForcing() placeholder
    @test model.convection isa NoConvection
    forcing_before = model.convection_forcing

    model2 = with_convection(model, _DummyConvOp())
    @test model2.convection isa _DummyConvOp
    # Decision 26: with_convection does NOT allocate forcing; placeholder preserved.
    @test model2.convection_forcing === forcing_before
    # Other fields preserved by identity
    @test model2.state === model.state
    @test model2.fluxes === model.fluxes
    @test model2.grid === model.grid
    @test model2.advection === model.advection
    @test model2.workspace === model.workspace
    @test model2.chemistry === model.chemistry
    @test model2.diffusion === model.diffusion
    @test model2.emissions === model.emissions
end

@testset "with_convection_forcing replaces only the forcing" begin
    FT = Float64
    mesh = LatLonMesh(; FT = FT, Nx = 4, Ny = 3)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
    state = CellState(ones(FT, 4, 3, 2); CO2 = fill(FT(400e-6), 4, 3, 2))
    fluxes = allocate_face_fluxes(grid.horizontal, 2; FT = FT, basis = DryBasis)
    model = TransportModel(state, fluxes, grid, UpwindScheme();
                            convection = _DummyConvOp())

    cmfmc = zeros(FT, 4, 3, 5)
    new_forcing = ConvectionForcing(cmfmc, nothing, nothing)

    model2 = with_convection_forcing(model, new_forcing)
    @test model2.convection_forcing === new_forcing
    # operator preserved
    @test model2.convection === model.convection
    # everything else preserved by identity
    @test model2.state === model.state
    @test model2.fluxes === model.fluxes
    @test model2.grid === model.grid
    @test model2.advection === model.advection
    @test model2.workspace === model.workspace
    @test model2.chemistry === model.chemistry
    @test model2.diffusion === model.diffusion
    @test model2.emissions === model.emissions
end

# ---------------------------------------------------------------------------
# Cloud base (fourth capability slot)
# ---------------------------------------------------------------------------

@testset "cloud_base requires CMFMC and travels with the forcing" begin
    FT = Float32
    Nc, Nz = 3, 2
    cmfmc = ntuple(_ -> fill(FT(0.01), Nc, Nc, Nz + 1), 6)
    cloud_base = ntuple(p -> fill(FT(p), Nc, Nc), 6)

    @test_throws ArgumentError ConvectionForcing(nothing, nothing, nothing, cloud_base)

    src = ConvectionForcing(cmfmc, nothing, nothing, cloud_base)
    @test _cap(src) == (true, false, false, true)

    dst = allocate_convection_forcing_like(src, ntuple(_ -> zeros(FT, Nc + 2, Nc + 2, Nz), 6))
    @test _cap(dst) == _cap(src)
    @test dst.cloud_base[1] !== src.cloud_base[1]
    buffer = dst.cloud_base[4]
    copy_convection_forcing!(dst, src)
    @test dst.cloud_base[4] === buffer
    @test dst.cloud_base == cloud_base

    # A destination without the cloud base is a capability mismatch.
    no_base = ConvectionForcing(ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6), nothing, nothing)
    @test_throws ArgumentError copy_convection_forcing!(no_base, src)
end
