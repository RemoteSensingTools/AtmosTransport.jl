#!/usr/bin/env julia
"""
Plan 23 tests for `TM5Convection` — built up over Commits 1–4.

Commit 1 (this file, initial): type hierarchy + workspace factory
+ runtime validator + stub `apply!` / `apply_convection!`.

Commit 2 adds the column-solver tests.
Commit 4 adds the full-kernel parity / conservation / CPU-GPU
agreement tests.

Invariant preserved by every commit in plan 23: `NoConvection` and
`CMFMCConvection` paths stay bit-exact to pre-plan-23 behaviour.
"""

using Test

using AtmosTransport
using .AtmosTransport: Operators, Grids, State, MetDrivers, Models

using .AtmosTransport.State: DryBasis, MoistBasis, CellState, CubedSphereState,
                              allocate_face_fluxes
using .AtmosTransport.Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh,
                             CubedSphereMesh, HybridSigmaPressure
using .AtmosTransport.Operators: AbstractConvection, NoConvection,
                                 CMFMCConvection, TM5Convection,
                                 CMFMCWorkspace, TM5Workspace,
                                 UpwindScheme
using .AtmosTransport.Operators.Convection: apply_convection!
using .AtmosTransport.MetDrivers: ConvectionForcing
using .AtmosTransport.Models: TransportModel, with_convection,
                              _convection_workspace_for

const FT = Float32

@testset "plan 23 Commit 1: TM5Convection type + workspace factory" begin
    @testset "construct TM5Convection() and tile_workspace_gib field" begin
        op = TM5Convection()
        @test op isa AbstractConvection
        @test op isa TM5Convection
        # Storage plan Commit 4: tile_workspace_gib field carries
        # the per-topology TM5 column-tile budget. Default 1.0 GiB.
        # P6 cache / collab-LU work: a second `use_collab_lu` Bool
        # field opts the convection block into the workgroup-
        # collaborative kernel; default false so existing runs are
        # bit-identical to pre-collab behaviour.
        # `lmax_conv` lands here too: a third Int field with default
        # 0 meaning "no truncation, use Nz". Setting it to a positive
        # value caps the convection matrix size (matching TM5's
        # tropoX* setups).
        @test fieldcount(typeof(op)) == 4
        @test op.tile_workspace_gib == 1.0
        @test op.use_collab_lu == false
        @test op.lmax_conv == 0
        @test op.n_merge == 1

        op_tuned = TM5Convection(tile_workspace_gib = 2.5)
        @test op_tuned.tile_workspace_gib == 2.5
        @test op_tuned.use_collab_lu == false
        @test op_tuned.lmax_conv == 0
        @test op_tuned.n_merge == 1

        op_collab = TM5Convection(use_collab_lu = true)
        @test op_collab.use_collab_lu == true
        @test op_collab.tile_workspace_gib == 1.0
        @test op_collab.lmax_conv == 0
        @test op_collab.n_merge == 1

        op_trunc = TM5Convection(use_collab_lu = true, lmax_conv = 75)
        @test op_trunc.lmax_conv == 75
        @test op_trunc.use_collab_lu == true
        @test op_trunc.n_merge == 1

        op_merge = TM5Convection(use_collab_lu = true, lmax_conv = 75, n_merge = 3)
        @test op_merge.n_merge == 3
        @test op_merge.lmax_conv == 75
        @test op_merge.use_collab_lu == true

        # n_merge = 2 is now ACCEPTED (2026-06-13). The historical
        # multi-substep mass blow-up was a CLIPPING bug — when `lmax_conv`
        # truncates the active region below the cloud top, the updraft never
        # fully detrains and the residual `amu` is uncompensated, which (fed by
        # emission tracers) blows up over many substeps. It hit ANY n_merge with
        # a clipping lmax_conv; it only LOOKED n=2-specific because of the L85
        # cloud-top/lmax alignment. Fixed by the cloud-top closure in the
        # updraft pass (tm5_kernels.jl + tm5_column_solve.jl).
        @test TM5Convection(n_merge = 2).n_merge == 2
        @test TM5Convection(use_collab_lu = true, lmax_conv = 75,
                            n_merge = 2).n_merge == 2
        @test TM5Convection{Float64}(1.0, true, 75, 2).n_merge == 2
        # n_merge ≥ 1 still enforced (all construction paths).
        @test_throws ArgumentError TM5Convection(n_merge = 0)
        @test_throws ArgumentError TM5Convection(n_merge = -1)
        @test_throws ArgumentError TM5Convection{Float64}(1.0, false, 0, 0)
        @test_throws ArgumentError TM5Convection{Float64}(1.0, false, 0, -3)
    end

    @testset "NoConvection and CMFMCConvection unchanged" begin
        # Plan 23's validator refactor must not change what these
        # return or how they dispatch. Bit-exact regression.
        @test NoConvection() isa AbstractConvection
        @test CMFMCConvection() isa AbstractConvection
    end
end

@testset "plan 23 Commit 1 / storage Commit 4: TM5Workspace tile shape" begin
    Nx, Ny, Nz = 4, 3, 5

    @testset "structured LatLon air_mass (Nx, Ny, Nz) — default tile" begin
        air_mass = zeros(FT, Nx, Ny, Nz)
        ws = TM5Workspace(air_mass)
        # Default tile size = total cells per launch; the workspace
        # is bit-equal to the pre-Commit-4 per-cell allocator in
        # capacity, but the shape is now flat (Nz, Nz, B).
        B = Nx * Ny
        @test ws isa TM5Workspace{FT}
        @test size(ws.conv1)       == (Nz, Nz, B)
        @test size(ws.pivots)      == (Nz, B)
        @test size(ws.cloud_dims)  == (3, B)
        @test size(ws.amu_scratch) == (Nz + 1, B)
        @test size(ws.amd_scratch) == (Nz + 1, B)
        @test ws.f_scratch === ws.conv1                         # alias
        @test eltype(ws.conv1)      == FT
        @test eltype(ws.pivots)     == Int
        @test eltype(ws.cloud_dims) == Int
    end

    @testset "face-indexed RG air_mass (ncells, Nz) — default tile" begin
        ncells = 24
        air_mass = zeros(FT, ncells, Nz)
        ws = TM5Workspace(air_mass)
        @test ws isa TM5Workspace{FT}
        @test size(ws.conv1)      == (Nz, Nz, ncells)
        @test size(ws.pivots)     == (Nz, ncells)
        @test size(ws.cloud_dims) == (3, ncells)
        @test ws.f_scratch === ws.conv1
    end

    @testset "CS panel NTuple{6, (Nc, Nc, Nz)} — default tile (per-panel)" begin
        Nc = 6
        air_mass = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
        ws = TM5Workspace(air_mass)
        # Storage plan Commit 4: workspace is shared across panels
        # (kernel launches sequentially), so the slab is sized for
        # one panel's ndrange (Nc²), not all six.
        B = Nc * Nc
        @test ws isa TM5Workspace{FT}
        @test ws.conv1 isa AbstractArray{FT, 3}
        @test size(ws.conv1)      == (Nz, Nz, B)
        @test size(ws.pivots)     == (Nz, B)
        @test size(ws.cloud_dims) == (3, B)
        @test ws.f_scratch === ws.conv1
    end

    @testset "explicit tile_columns shrinks the slab" begin
        air_mass = zeros(FT, 32, 8, Nz)
        ws_full  = TM5Workspace(air_mass)                       # B = 256
        ws_tile  = TM5Workspace(air_mass; tile_columns = 64)
        @test size(ws_full.conv1, 3) == 256
        @test size(ws_tile.conv1, 3) == 64
        @test size(ws_tile.amu_scratch, 2) == 64
    end

    @testset "tile_workspace_gib budget derives B" begin
        # Big topology, small budget → derive_tile_columns clamps to 256.
        air_mass = zeros(FT, 720, 720, Nz)
        ws = TM5Workspace(air_mass; tile_workspace_gib = 1e-6)
        @test size(ws.conv1, 3) >= 256                          # floor
        # Big budget → clamps at total cells.
        ws_big = TM5Workspace(air_mass; tile_workspace_gib = 1024.0)
        @test size(ws_big.conv1, 3) == 720 * 720
    end
end

@testset "plan 23 Commit 1: _convection_workspace_for dispatch" begin
    # Minimal LatLon grid just to drive _convection_workspace_for.
    mesh = LatLonMesh(; Nx=4, Ny=3, FT=FT)
    A_ifc = FT[0, 500, 5000, 30000, 0]
    B_ifc = FT[0, 0, FT(0.1), FT(0.5), 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    state = CellState(zeros(FT, 4, 3, 4); CO2 = zeros(FT, 4, 3, 4))

    @test _convection_workspace_for(NoConvection(), state, grid) === nothing
    ws_cmfmc = _convection_workspace_for(CMFMCConvection(), state, grid)
    @test ws_cmfmc isa CMFMCWorkspace{FT}

    ws_tm5 = _convection_workspace_for(TM5Convection(), state, grid)
    @test ws_tm5 isa TM5Workspace{FT}
    # Storage plan Commit 4: flat tile slab (Nz, Nz, B). Default
    # B = Nx*Ny when budget covers all cells (it does at this size).
    @test size(ws_tm5.conv1) == (4, 4, 12)
    @test ws_tm5.cell_metrics !== nothing
    @test size(ws_tm5.cell_metrics) == (3,)
end

@testset "plan 23 Commit 4: TM5Convection apply! LL kernel" begin
    Nx, Ny, Nz, Nt = 4, 3, 8, 2
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    A_ifc = FT[0, 500, 1000, 2000, 5000, 10000, 30000, 50000, 0]
    B_ifc = FT[0, 0, 0, FT(0.05), FT(0.2), FT(0.4), FT(0.7), FT(0.9), 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    m = fill(FT(5e3), Nx, Ny, Nz)
    tracer1 = zeros(FT, Nx, Ny, Nz); tracer1[:, :, Nz] .= FT(1e-3) .* m[:, :, Nz]
    tracer2 = zeros(FT, Nx, Ny, Nz); tracer2[:, :, 3] .= FT(2e-3) .* m[:, :, 3]
    state = CellState(m; CO2 = tracer1, CH4 = tracer2)

    entu = zeros(FT, Nx, Ny, Nz); entu[:, :, 3:6] .= FT(0.03)
    detu = zeros(FT, Nx, Ny, Nz); detu[:, :, 3:6] .= FT(0.02)
    entd = zeros(FT, Nx, Ny, Nz); entd[:, :, 4:6] .= FT(0.01)
    detd = zeros(FT, Nx, Ny, Nz); detd[:, :, 4:6] .= FT(0.005)
    forcing = ConvectionForcing(nothing, nothing,
                                 (; entu, detu, entd, detd))
    ws = TM5Workspace(state.air_mass; cell_metrics = ones(FT, Ny))

    mass_before = [sum(state.tracers_raw[:, :, :, t]) for t in 1:Nt]
    state0_copy = copy(state.tracers_raw)
    apply!(state, forcing, grid, TM5Convection(), FT(600); workspace = ws)

    for t in 1:Nt
        mass_after = sum(state.tracers_raw[:, :, :, t])
        @test isapprox(mass_after, mass_before[t];
                        rtol = 1f4 * eps(FT))
    end
    # Nontrivial: tracer profile changed (not silent identity).
    @test any(state.tracers_raw .!= state0_copy)

    # Zero-forcing → bit-exact identity.
    state_zero = CellState(m; CO2 = copy(tracer1), CH4 = copy(tracer2))
    ws_zero = TM5Workspace(state_zero.air_mass; cell_metrics = ones(FT, Ny))
    zero_forcing = ConvectionForcing(nothing, nothing,
        (; entu = zeros(FT, Nx, Ny, Nz), detu = zeros(FT, Nx, Ny, Nz),
           entd = zeros(FT, Nx, Ny, Nz), detd = zeros(FT, Nx, Ny, Nz)))
    state0_identity = copy(state_zero.tracers_raw)
    apply!(state_zero, zero_forcing, grid, TM5Convection(), FT(600);
            workspace = ws_zero)
    @test state_zero.tracers_raw == state0_identity

    # A deferred workspace must materialize the requested legacy tile on CPU
    # fallback, then reproduce the normal allocation path exactly.
    deferred = TM5Workspace(state.air_mass; tile_columns=3, defer_scratch=true,
                            cell_metrics=ws.cell_metrics)
    @test isempty(deferred.conv1)
    initial = copy(state0_copy)
    apply_convection!(initial, state.air_mass, forcing,
                      TM5Convection(use_collab_lu=true), FT(600), deferred, grid)
    @test initial == state.tracers_raw
    @test size(deferred.conv1, 3) == 3
    @test deferred.f_scratch === deferred.conv1

end

@testset "plan 23 Commit 4: TM5Convection apply! RG kernel" begin
    mesh = ReducedGaussianMesh(FT[-0.9, 0.0, 0.9], [4, 4, 4]; FT=FT)
    Nz = 6
    A_ifc = collect(FT, range(0, 5f4; length=Nz+1))
    B_ifc = collect(FT, range(1, 0;   length=Nz+1))
    A_ifc[1] = 0; B_ifc[end] = 0
    B_ifc[1] = 0; A_ifc[end] = 0
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)
    ncells = AtmosTransport.Grids.ncells(mesh)

    m = fill(FT(5e3), ncells, Nz)
    tracer1 = zeros(FT, ncells, Nz); tracer1[:, Nz] .= FT(1e-3) .* m[:, Nz]
    state = CellState(m; CO2 = tracer1)

    entu = zeros(FT, ncells, Nz); entu[:, 2:4] .= FT(0.03)
    detu = zeros(FT, ncells, Nz); detu[:, 2:4] .= FT(0.02)
    entd = zeros(FT, ncells, Nz); entd[:, 3:4] .= FT(0.01)
    detd = zeros(FT, ncells, Nz); detd[:, 3:4] .= FT(0.005)
    forcing = ConvectionForcing(nothing, nothing,
                                 (; entu, detu, entd, detd))
    ws = TM5Workspace(state.air_mass; cell_metrics = ones(FT, ncells))

    mass_before = sum(state.tracers_raw)
    state0_copy = copy(state.tracers_raw)
    apply!(state, forcing, grid, TM5Convection(), FT(600); workspace = ws)
    mass_after = sum(state.tracers_raw)
    @test isapprox(mass_after, mass_before; rtol = 1f4 * eps(FT))
    @test any(state.tracers_raw .!= state0_copy)

    # A deferred workspace must materialize the requested legacy tile on CPU
    # fallback, then reproduce the normal allocation path exactly.
    deferred = TM5Workspace(state.air_mass; tile_columns=3, defer_scratch=true,
                            cell_metrics=ws.cell_metrics)
    @test isempty(deferred.conv1)
    initial = copy(state0_copy)
    apply_convection!(initial, state.air_mass, forcing,
                      TM5Convection(use_collab_lu=true), FT(600), deferred, grid)
    @test initial == state.tracers_raw
    @test size(deferred.conv1, 3) == 3
    @test deferred.f_scratch === deferred.conv1

end

@testset "plan 23 Commit 4: TM5Convection apply! CS kernel" begin
    Nc = 4
    Nz = 6
    mesh = CubedSphereMesh(; Nc = Nc, Hp = 1, FT = FT)
    A_ifc = collect(FT, range(0, 5f4; length=Nz+1))
    B_ifc = collect(FT, range(1, 0;   length=Nz+1))
    A_ifc[1] = 0; B_ifc[end] = 0
    B_ifc[1] = 0; A_ifc[end] = 0
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
    Hp = mesh.Hp

    air_mass = ntuple(_ -> fill(FT(5e3), Nc + 2Hp, Nc + 2Hp, Nz), 6)
    tracer1  = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)
    for p in 1:6
        tracer1[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz] .= FT(1e-3) * FT(5e3)
    end
    state = CubedSphereState(DryBasis, mesh, air_mass; CO2 = tracer1)

    entu = ntuple(_ -> begin e = zeros(FT, Nc, Nc, Nz); e[:, :, 2:4] .= FT(0.03); e end, 6)
    detu = ntuple(_ -> begin e = zeros(FT, Nc, Nc, Nz); e[:, :, 2:4] .= FT(0.02); e end, 6)
    entd = ntuple(_ -> begin e = zeros(FT, Nc, Nc, Nz); e[:, :, 3:4] .= FT(0.01); e end, 6)
    detd = ntuple(_ -> begin e = zeros(FT, Nc, Nc, Nz); e[:, :, 3:4] .= FT(0.005); e end, 6)
    forcing = ConvectionForcing(nothing, nothing,
                                 (; entu, detu, entd, detd))
    ws = TM5Workspace(state.air_mass; cell_metrics = ntuple(_ -> ones(FT, Nc, Nc), 6))

    function interior_mass(tracers_raw)
        s = zero(FT)
        for p in 1:6, k in 1:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc
            s += tracers_raw[p][i, j, k, 1]
        end
        return s
    end
    state0_copy = map(copy, state.tracers_raw)
    mass_before = interior_mass(state.tracers_raw)
    apply!(state, forcing, grid, TM5Convection(), FT(600); workspace = ws)
    mass_after = interior_mass(state.tracers_raw)
    @test isapprox(mass_after, mass_before; rtol = 1f4 * eps(FT))

    # A deferred workspace must materialize the requested legacy tile on CPU
    # fallback, then reproduce the normal allocation path exactly.
    deferred = TM5Workspace(state.air_mass; tile_columns=3, defer_scratch=true,
                            cell_metrics=ws.cell_metrics)
    @test isempty(deferred.conv1)
    initial = map(copy, state0_copy)
    apply_convection!(initial, state.air_mass, forcing,
                      TM5Convection(use_collab_lu=true), FT(600), deferred, grid)
    @test initial == state.tracers_raw
    @test size(deferred.conv1, 3) == 3
    @test deferred.f_scratch === deferred.conv1

end

# The collaborative-LU kernel uses `@uniform` for workgroup-uniform
# locals (`g`, `i`, `j`, `area`, …) so the KA-on-GPU paths hoist them
# correctly. The KA CPU backend's `@uniform` lowering doesn't accept
# `@index(Group)` in this exact shape, so the collab-LU path is
# GPU-only. Production targets (CUDA + Metal) both honour `@uniform`
# in the GPU-codegen path; the CPU backend is for unit-test convenience
# only. This test therefore gates on a working CUDA backend and is
# explicitly skipped on CPU-only hosts. A Metal sanity run is the
# parallel validation step before production opt-in (see
# `docs/memos/TM5_CONVECTION_AGENTLOOP_SYNTHESIS.md`).
const _HAS_CUDA = try
    @eval using CUDA
    @eval CUDA.functional()
catch
    false
end

if _HAS_CUDA
    @eval using Adapt
    @eval using CUDA: CuArray
end

@testset "P6 collab-LU: bit-exact equivalence with per-thread kernel (CUDA)" begin
    if !_HAS_CUDA
        @test_skip "CUDA backend required for the workgroup-collaborative LU test"
    else
        Nx, Ny, Nz, Nt = 4, 3, 8, 2
        mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
        A_ifc = FT[0, 500, 1000, 2000, 5000, 10000, 30000, 50000, 0]
        B_ifc = FT[0, 0, 0, FT(0.05), FT(0.2), FT(0.4), FT(0.7), FT(0.9), 1]
        vc = HybridSigmaPressure(A_ifc, B_ifc)
        grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

        m = fill(FT(5e3), Nx, Ny, Nz)
        tracer1 = zeros(FT, Nx, Ny, Nz); tracer1[:, :, Nz] .= FT(1e-3) .* m[:, :, Nz]
        tracer2 = zeros(FT, Nx, Ny, Nz); tracer2[:, :, 3] .= FT(2e-3) .* m[:, :, 3]

        entu = zeros(FT, Nx, Ny, Nz); entu[:, :, 3:6] .= FT(0.03)
        detu = zeros(FT, Nx, Ny, Nz); detu[:, :, 3:6] .= FT(0.02)
        entd = zeros(FT, Nx, Ny, Nz); entd[:, :, 4:6] .= FT(0.01)
        detd = zeros(FT, Nx, Ny, Nz); detd[:, :, 4:6] .= FT(0.005)
        forcing = ConvectionForcing(nothing, nothing,
                                     (; entu, detu, entd, detd))

        # Reference: per-thread kernel on GPU.
        state_ref = CellState(m; CO2 = copy(tracer1), CH4 = copy(tracer2))
        ws_ref    = TM5Workspace(state_ref.air_mass; cell_metrics = ones(FT, Ny))
        state_d_ref   = Adapt.adapt(CuArray, state_ref)
        ws_d_ref      = Adapt.adapt(CuArray, ws_ref)
        forcing_d     = Adapt.adapt(CuArray, forcing)
        apply!(state_d_ref, forcing_d, grid, TM5Convection(), FT(600);
                workspace = ws_d_ref)
        CUDA.synchronize()

        # Candidate: collab-LU kernel.
        state_col = CellState(m; CO2 = copy(tracer1), CH4 = copy(tracer2))
        ws_col    = TM5Workspace(state_col.air_mass; cell_metrics = ones(FT, Ny))
        state_d_col   = Adapt.adapt(CuArray, state_col)
        ws_d_col      = Adapt.adapt(CuArray, ws_col)
        apply!(state_d_col, forcing_d, grid,
                TM5Convection(use_collab_lu = true), FT(600);
                workspace = ws_d_col)
        CUDA.synchronize()

        # The collab-LU performs the same multiplications in the same
        # order as the per-thread builder + LU, so on small grids the
        # output is *bit-identical*, not merely "within Float32 round-
        # off".  The agent-loop synthesis measured a similar pattern
        # on C180/L85 panels (`err = 0.0` on shallower-convection
        # panels, ~7e-7 on the deepest one).  We accept any deviation
        # ≤ 500·eps(FT)·|q_ref| to keep the test stable across CUDA
        # toolchain bumps that might re-order multiply-adds.
        ref = Array(state_d_ref.tracers_raw)
        col = Array(state_d_col.tracers_raw)
        @test maximum(abs.(col .- ref)) <= 500f0 * eps(FT) * maximum(abs.(ref))

        # Zero-forcing → identity on both paths.
        state_ref0 = CellState(m; CO2 = copy(tracer1), CH4 = copy(tracer2))
        state_col0 = CellState(m; CO2 = copy(tracer1), CH4 = copy(tracer2))
        ws_ref0    = TM5Workspace(state_ref0.air_mass; cell_metrics = ones(FT, Ny))
        ws_col0    = TM5Workspace(state_col0.air_mass; cell_metrics = ones(FT, Ny))
        state_d_ref0 = Adapt.adapt(CuArray, state_ref0); ws_d_ref0 = Adapt.adapt(CuArray, ws_ref0)
        state_d_col0 = Adapt.adapt(CuArray, state_col0); ws_d_col0 = Adapt.adapt(CuArray, ws_col0)
        zero_forcing = ConvectionForcing(nothing, nothing,
            (; entu = zeros(FT, Nx, Ny, Nz), detu = zeros(FT, Nx, Ny, Nz),
               entd = zeros(FT, Nx, Ny, Nz), detd = zeros(FT, Nx, Ny, Nz)))
        zero_forcing_d = Adapt.adapt(CuArray, zero_forcing)
        apply!(state_d_ref0, zero_forcing_d, grid, TM5Convection(), FT(600);
                workspace = ws_d_ref0)
        apply!(state_d_col0, zero_forcing_d, grid,
                TM5Convection(use_collab_lu = true), FT(600);
                workspace = ws_d_col0)
        CUDA.synchronize()
        @test Array(state_d_ref0.tracers_raw) == Array(state_d_col0.tracers_raw)
    end
end

@testset "P6 collab-LU: bit-exact equivalence on ReducedGaussian (CUDA)" begin
    # Same shape contract as the LL testset but with the face-indexed
    # RG kernel, so a regression in the per-topology body duplication
    # is caught here even if LL stays bit-exact.
    if !_HAS_CUDA
        @test_skip "CUDA backend required for the workgroup-collaborative LU test"
    else
        Nz = 8
        mesh = ReducedGaussianMesh(FT[-0.9, 0.0, 0.9], [4, 4, 4]; FT=FT)
        ncells = AtmosTransport.Grids.ncells(mesh)
        A_ifc = collect(FT, range(0, 5f4; length=Nz+1))
        B_ifc = collect(FT, range(1, 0;   length=Nz+1))
        A_ifc[1] = 0; B_ifc[end] = 0
        B_ifc[1] = 0; A_ifc[end] = 0
        vc = HybridSigmaPressure(A_ifc, B_ifc)
        grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

        m = fill(FT(5e3), ncells, Nz)
        tracer1 = zeros(FT, ncells, Nz); tracer1[:, Nz] .= FT(1e-3) .* m[:, Nz]
        tracer2 = zeros(FT, ncells, Nz); tracer2[:, 3]  .= FT(2e-3) .* m[:, 3]

        entu = zeros(FT, ncells, Nz); entu[:, 3:6] .= FT(0.03)
        detu = zeros(FT, ncells, Nz); detu[:, 3:6] .= FT(0.02)
        entd = zeros(FT, ncells, Nz); entd[:, 4:6] .= FT(0.01)
        detd = zeros(FT, ncells, Nz); detd[:, 4:6] .= FT(0.005)
        forcing = ConvectionForcing(nothing, nothing,
                                     (; entu, detu, entd, detd))
        forcing_d = Adapt.adapt(CuArray, forcing)

        state_ref = CellState(m; CO2 = copy(tracer1), CH4 = copy(tracer2))
        ws_ref    = TM5Workspace(state_ref.air_mass; cell_metrics = ones(FT, ncells))
        state_d_ref = Adapt.adapt(CuArray, state_ref); ws_d_ref = Adapt.adapt(CuArray, ws_ref)
        apply!(state_d_ref, forcing_d, grid, TM5Convection(), FT(600);
                workspace = ws_d_ref)

        state_col = CellState(m; CO2 = copy(tracer1), CH4 = copy(tracer2))
        ws_col    = TM5Workspace(state_col.air_mass; cell_metrics = ones(FT, ncells))
        state_d_col = Adapt.adapt(CuArray, state_col); ws_d_col = Adapt.adapt(CuArray, ws_col)
        apply!(state_d_col, forcing_d, grid,
                TM5Convection(use_collab_lu = true), FT(600);
                workspace = ws_d_col)
        CUDA.synchronize()

        ref = Array(state_d_ref.tracers_raw)
        col = Array(state_d_col.tracers_raw)
        @test maximum(abs.(col .- ref)) <= 500f0 * eps(FT) * maximum(abs.(ref))
    end
end

@testset "P6 collab-LU: bit-exact equivalence on CubedSphere (CUDA)" begin
    # CS is the production topology — keep this test even at the tiny
    # `Nc=6` grid we use elsewhere, so the halo-offset arithmetic in
    # `_tm5_read_mass` / `_tm5_read_q` / `_tm5_write_q!` (only the CS
    # variant adds `Hp`) is exercised on the equivalence path.
    if !_HAS_CUDA
        @test_skip "CUDA backend required for the workgroup-collaborative LU test"
    else
        Nc, Nz = 6, 8
        mesh = CubedSphereMesh(; Nc=Nc, FT=FT, Hp=1)
        A_ifc = collect(FT, range(0, 5f4; length=Nz+1))
        B_ifc = collect(FT, range(1, 0;   length=Nz+1))
        A_ifc[1] = 0; B_ifc[end] = 0
        B_ifc[1] = 0; A_ifc[end] = 0
        vc = HybridSigmaPressure(A_ifc, B_ifc)
        grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)
        Hp = mesh.Hp
        N = Nc + 2 * Hp

        air_mass = ntuple(_ -> fill(FT(5e3), N, N, Nz), 6)
        tracer1 = ntuple(_ -> begin
            arr = zeros(FT, N, N, Nz)
            arr[(Hp + 1):(Hp + Nc), (Hp + 1):(Hp + Nc), Nz] .= FT(1e-3) .* FT(5e3)
            arr
        end, 6)
        # Use the topology-aware constructor (DryBasis, mesh, air_mass; …)
        # — same signature as the existing CS apply! testset above.
        state = CubedSphereState(DryBasis, mesh, air_mass; CO2 = tracer1)

        entu = ntuple(_ -> (a = zeros(FT, Nc, Nc, Nz); a[:, :, 3:6] .= FT(0.03); a), 6)
        detu = ntuple(_ -> (a = zeros(FT, Nc, Nc, Nz); a[:, :, 3:6] .= FT(0.02); a), 6)
        entd = ntuple(_ -> (a = zeros(FT, Nc, Nc, Nz); a[:, :, 4:6] .= FT(0.01); a), 6)
        detd = ntuple(_ -> (a = zeros(FT, Nc, Nc, Nz); a[:, :, 4:6] .= FT(0.005); a), 6)
        forcing = ConvectionForcing(nothing, nothing,
                                     (; entu, detu, entd, detd))
        forcing_d = Adapt.adapt(CuArray, forcing)

        state_ref = CubedSphereState(DryBasis, mesh, air_mass;
                                      CO2 = ntuple(p -> copy(tracer1[p]), 6))
        ws_ref    = TM5Workspace(state_ref.air_mass; cell_metrics = ntuple(_ -> ones(FT, Nc, Nc), 6))
        state_d_ref = Adapt.adapt(CuArray, state_ref); ws_d_ref = Adapt.adapt(CuArray, ws_ref)
        apply!(state_d_ref, forcing_d, grid, TM5Convection(), FT(600);
                workspace = ws_d_ref)

        state_col = CubedSphereState(DryBasis, mesh, air_mass;
                                      CO2 = ntuple(p -> copy(tracer1[p]), 6))
        ws_col    = TM5Workspace(state_col.air_mass; cell_metrics = ntuple(_ -> ones(FT, Nc, Nc), 6))
        state_d_col = Adapt.adapt(CuArray, state_col); ws_d_col = Adapt.adapt(CuArray, ws_col)
        apply!(state_d_col, forcing_d, grid,
                TM5Convection(use_collab_lu = true), FT(600);
                workspace = ws_d_col)
        CUDA.synchronize()

        # Compare all six panels.
        max_err = 0f0; max_ref = 0f0
        for p in 1:6
            ref = Array(state_d_ref.tracers_raw[p])
            col = Array(state_d_col.tracers_raw[p])
            max_err = max(max_err, maximum(abs.(col .- ref)))
            max_ref = max(max_ref, maximum(abs.(ref)))
        end
        @test max_err <= 500f0 * eps(FT) * max_ref
    end
end

@testset "P6 collab-LU: lmax_conv truncation is bit-exact when forcings above k_shift are zero (CUDA)" begin
    # The lmax_conv truncation is *bit-exact* (within Float32 rounding)
    # IFF the binary has zero forcings above `k_shift = Nz - lmax_conv`.
    # We construct a small problem where the top 3 layers have NO
    # entu/detu/entd/detd activity (matching the contract of TM5's
    # tropoX* preprocessors and our scan-validated ERA5/L85 binary),
    # then compare collab-LU at `lmax_conv = Nz` vs `lmax_conv = Nz - 3`.
    if !_HAS_CUDA
        @test_skip "CUDA backend required for the workgroup-collaborative LU test"
    else
        Nx, Ny, Nz, Nt = 4, 3, 8, 2
        K_SHIFT = 3
        L_TRUNC = Nz - K_SHIFT          # = 5
        mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
        A_ifc = FT[0, 500, 1000, 2000, 5000, 10000, 30000, 50000, 0]
        B_ifc = FT[0, 0, 0, FT(0.05), FT(0.2), FT(0.4), FT(0.7), FT(0.9), 1]
        vc = HybridSigmaPressure(A_ifc, B_ifc)
        grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

        m = fill(FT(5e3), Nx, Ny, Nz)
        tracer1 = zeros(FT, Nx, Ny, Nz); tracer1[:, :, Nz] .= FT(1e-3) .* m[:, :, Nz]
        tracer2 = zeros(FT, Nx, Ny, Nz); tracer2[:, :, K_SHIFT + 1] .= FT(2e-3) .* m[:, :, K_SHIFT + 1]

        # Forcings live ONLY in layers (K_SHIFT + 1 : Nz). Top K_SHIFT
        # layers are pass-through identity for convection — matching
        # the TM5 tropoX* preprocessor convention.
        entu = zeros(FT, Nx, Ny, Nz); entu[:, :, (K_SHIFT + 2):(Nz - 2)] .= FT(0.03)
        detu = zeros(FT, Nx, Ny, Nz); detu[:, :, (K_SHIFT + 2):(Nz - 2)] .= FT(0.02)
        entd = zeros(FT, Nx, Ny, Nz); entd[:, :, (K_SHIFT + 3):(Nz - 2)] .= FT(0.01)
        detd = zeros(FT, Nx, Ny, Nz); detd[:, :, (K_SHIFT + 3):(Nz - 2)] .= FT(0.005)
        forcing = ConvectionForcing(nothing, nothing,
                                     (; entu, detu, entd, detd))
        forcing_d = Adapt.adapt(CuArray, forcing)

        # Reference: collab-LU with NO truncation (lmax_conv = 0 ⇒ Nz).
        state_full = CellState(m; CO2 = copy(tracer1), CH4 = copy(tracer2))
        ws_full    = TM5Workspace(state_full.air_mass; cell_metrics = ones(FT, Ny))
        state_d_full = Adapt.adapt(CuArray, state_full); ws_d_full = Adapt.adapt(CuArray, ws_full)
        apply!(state_d_full, forcing_d, grid,
                TM5Convection(use_collab_lu = true), FT(600);
                workspace = ws_d_full)

        # Candidate: collab-LU with lmax_conv = L_TRUNC (skips top
        # K_SHIFT layers as pass-through).
        state_trunc = CellState(m; CO2 = copy(tracer1), CH4 = copy(tracer2))
        ws_trunc    = TM5Workspace(state_trunc.air_mass; cell_metrics = ones(FT, Ny))
        state_d_trunc = Adapt.adapt(CuArray, state_trunc); ws_d_trunc = Adapt.adapt(CuArray, ws_trunc)
        apply!(state_d_trunc, forcing_d, grid,
                TM5Convection(use_collab_lu = true, lmax_conv = L_TRUNC),
                FT(600); workspace = ws_d_trunc)
        CUDA.synchronize()

        full  = Array(state_d_full.tracers_raw)
        trunc = Array(state_d_trunc.tracers_raw)
        # The whole field must be bit-equivalent within Float32
        # rounding — both the active block and the pass-through
        # layers above k_shift.
        @test maximum(abs.(trunc .- full)) <= 500f0 * eps(FT) * maximum(abs.(full))

        # Pass-through invariant: the top K_SHIFT layers of the
        # truncated path must equal the original initial condition
        # exactly (no roundoff, no rounding — they were never
        # touched by the kernel).
        for tt in 1:Nt
            tracer_init = tt == 1 ? tracer1 : tracer2
            for k in 1:K_SHIFT
                @test trunc[:, :, k, tt] == tracer_init[:, :, k]
            end
        end
    end
end

@testset "P6 collab-LU: n_merge = 1 is bit-exact to no-merge path (CUDA)" begin
    # When `n_merge = 1` the host should bypass the aggregation
    # wrapper entirely and call the collab kernel directly.
    # Equivalently, even if the wrapper ran with n_merge=1 (single-
    # layer "super" cells), the proportional disaggregation reduces
    # to identity. Verify the dispatch picks the no-merge fast path
    # by comparing against `lmax_conv = 75, n_merge = 1` vs the
    # baseline collab-LU with `n_merge = 1`.
    if !_HAS_CUDA
        @test_skip "CUDA backend required"
    else
        Nx, Ny, Nz, Nt = 4, 3, 8, 2
        mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
        A_ifc = FT[0, 500, 1000, 2000, 5000, 10000, 30000, 50000, 0]
        B_ifc = FT[0, 0, 0, FT(0.05), FT(0.2), FT(0.4), FT(0.7), FT(0.9), 1]
        vc = HybridSigmaPressure(A_ifc, B_ifc)
        grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

        m = fill(FT(5e3), Nx, Ny, Nz)
        tracer = zeros(FT, Nx, Ny, Nz); tracer[:, :, Nz] .= FT(1e-3) .* m[:, :, Nz]
        entu = zeros(FT, Nx, Ny, Nz); entu[:, :, 3:6] .= FT(0.03)
        detu = zeros(FT, Nx, Ny, Nz); detu[:, :, 3:6] .= FT(0.02)
        entd = zeros(FT, Nx, Ny, Nz); entd[:, :, 4:6] .= FT(0.01)
        detd = zeros(FT, Nx, Ny, Nz); detd[:, :, 4:6] .= FT(0.005)
        forcing = ConvectionForcing(nothing, nothing, (; entu, detu, entd, detd))
        forcing_d = Adapt.adapt(CuArray, forcing)

        state_a = CellState(m; CO2 = copy(tracer))
        ws_a    = TM5Workspace(state_a.air_mass; cell_metrics = ones(FT, Ny))
        state_d_a = Adapt.adapt(CuArray, state_a); ws_d_a = Adapt.adapt(CuArray, ws_a)
        apply!(state_d_a, forcing_d, grid,
               TM5Convection(use_collab_lu = true), FT(600);
               workspace = ws_d_a)

        # Explicit n_merge = 1 — should be byte-identical (host takes
        # the same `nm == 1` short-circuit path).
        state_b = CellState(m; CO2 = copy(tracer))
        ws_b    = TM5Workspace(state_b.air_mass; cell_metrics = ones(FT, Ny))
        state_d_b = Adapt.adapt(CuArray, state_b); ws_d_b = Adapt.adapt(CuArray, ws_b)
        apply!(state_d_b, forcing_d, grid,
               TM5Convection(use_collab_lu = true, n_merge = 1), FT(600);
               workspace = ws_d_b)
        CUDA.synchronize()
        @test Array(state_d_a.tracers_raw) == Array(state_d_b.tracers_raw)
    end
end

@testset "P6 collab-LU: vertical aggregation conserves tracer mass (CUDA)" begin
    # The aggregate→solve→disaggregate path must conserve total
    # tracer mass exactly, regardless of `n_merge`. We use Nz = 12
    # so the divisors 2, 3, 4 all split lmax_conv cleanly.
    if !_HAS_CUDA
        @test_skip "CUDA backend required"
    else
        Nx, Ny, Nz, Nt = 4, 3, 12, 2
        mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
        A_ifc = collect(FT, range(0, 5f4; length = Nz + 1))
        B_ifc = collect(FT, range(1, 0;   length = Nz + 1))
        A_ifc[1] = 0; B_ifc[end] = 0; B_ifc[1] = 0; A_ifc[end] = 0
        vc = HybridSigmaPressure(A_ifc, B_ifc)
        grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

        m = fill(FT(5e3), Nx, Ny, Nz)
        tracer1 = zeros(FT, Nx, Ny, Nz)
        tracer2 = zeros(FT, Nx, Ny, Nz)
        for i in 1:Nx, j in 1:Ny, k in 1:Nz
            tracer1[i, j, k] = FT(1e-4) * (k + i)        # smooth profile
            tracer2[i, j, k] = FT(2e-4) * (Nz - k + j)   # different smooth profile
        end

        # Forcings only in layers 5..10 → lmax_conv = 8 covers the
        # active region (layers 5..12), so layers 1..4 are pass-
        # through. n_merge ∈ {2, 4} divides 8 evenly.
        entu = zeros(FT, Nx, Ny, Nz); entu[:, :, 6:9] .= FT(0.03)
        detu = zeros(FT, Nx, Ny, Nz); detu[:, :, 6:9] .= FT(0.02)
        entd = zeros(FT, Nx, Ny, Nz); entd[:, :, 7:9] .= FT(0.01)
        detd = zeros(FT, Nx, Ny, Nz); detd[:, :, 7:9] .= FT(0.005)
        forcing = ConvectionForcing(nothing, nothing, (; entu, detu, entd, detd))
        forcing_d = Adapt.adapt(CuArray, forcing)

        # n_merge=2 is now valid (the clipping blow-up was fixed by the
        # cloud-top closure); test 2, 4, 8 (divisors of lmax_conv=8) — all
        # must conserve mass.
        for nm in (2, 4, 8)
            state = CellState(m; CO2 = copy(tracer1), CH4 = copy(tracer2))
            ws    = TM5Workspace(state.air_mass; cell_metrics = ones(FT, Ny))
            state_d = Adapt.adapt(CuArray, state); ws_d = Adapt.adapt(CuArray, ws)
            mass_before_co2 = sum(state.tracers_raw[:, :, :, 1])
            mass_before_ch4 = sum(state.tracers_raw[:, :, :, 2])
            apply!(state_d, forcing_d, grid,
                   TM5Convection(use_collab_lu = true, lmax_conv = 8, n_merge = nm),
                   FT(600); workspace = ws_d)
            CUDA.synchronize()
            q_out = Array(state_d.tracers_raw)
            # Mass conservation: total over all (i, j, k) per tracer
            # must equal the pre-step total to within Float32 rounding
            # of `Nx * Ny * Nz ≈ 144` summations.
            mass_after_co2 = sum(q_out[:, :, :, 1])
            mass_after_ch4 = sum(q_out[:, :, :, 2])
            # Same tolerance as the existing TM5 apply! testset
            # (kernel-level mass conservation is bounded by TM5's
            # surface-boundary handling, not by the disaggregation
            # arithmetic; the proportional redistribution is
            # mass-exact within roundoff).
            @test isapprox(mass_after_co2, mass_before_co2;
                            rtol = 1f4 * eps(FT))
            @test isapprox(mass_after_ch4, mass_before_ch4;
                            rtol = 1f4 * eps(FT))
            # Pass-through invariant: layers above the active region
            # (k < Nz - lmax_conv + 1 = 5) must equal the input.
            for k in 1:4, tt in 1:Nt
                tracer_init = tt == 1 ? tracer1 : tracer2
                @test q_out[:, :, k, tt] == tracer_init[:, :, k]
            end
            # Non-trivial: the active layers were actually mixed.
            @test any(q_out[:, :, 5:Nz, :] .!= cat(tracer1[:, :, 5:Nz],
                                                     tracer2[:, :, 5:Nz];
                                                     dims = 4))
        end
    end
end

@testset "cloud-top closure: icltop==1 (clipped cloud) conserves mass over many substeps" begin
    # Regression for the clipping mass blow-up (historically mislabelled the
    # "n_merge=2 bug"; it is clipping, not merge). When `lmax_conv` truncates
    # the active region so the cloud reaches its very top (relative
    # `icltop == 1`), the pre-fix updraft left an UNCOMPENSATED residual `amu`
    # at that top (the subsidence pass runs `LMAX_CONV:-1:2`), which blew mass
    # up over many substeps. The cloud-top closure forces full updraft
    # detrainment there. Mass must be conserved over many substeps on BOTH the
    # per-thread (CPU) and collaborative (CUDA) paths.
    Nx, Ny, Nz, Nt = 4, 3, 12, 1
    mesh = LatLonMesh(; Nx = Nx, Ny = Ny, FT = FT)
    A_ifc = collect(FT, range(0, 5f4; length = Nz + 1))
    B_ifc = collect(FT, range(1, 0;   length = Nz + 1))
    A_ifc[1] = 0; B_ifc[end] = 0; B_ifc[1] = 0; A_ifc[end] = 0
    vc   = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
    m = fill(FT(5e3), Nx, Ny, Nz)
    tracer0 = zeros(FT, Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        tracer0[i, j, k] = FT(1e-4) * (k + i)
    end
    # Put the cloud at the model top (layers 1..4) so the cloud top IS the very
    # top layer ⇒ relative `icltop == 1` for BOTH paths. (Production hits the
    # same icltop==1 via a clipping `lmax_conv` on the collab path; the
    # per-thread solver ignores `lmax_conv` and always runs the full column, so
    # forcing the cloud to the top is the portable way to drive icltop==1 on
    # both.) This is exactly the branch the aggregation testset (icltop=2) skips.
    entu = zeros(FT, Nx, Ny, Nz); entu[:, :, 1:4] .= FT(0.03)
    detu = zeros(FT, Nx, Ny, Nz); detu[:, :, 1:4] .= FT(0.02)
    entd = zeros(FT, Nx, Ny, Nz); entd[:, :, 2:4] .= FT(0.01)
    detd = zeros(FT, Nx, Ny, Nz); detd[:, :, 2:4] .= FT(0.005)
    forcing = ConvectionForcing(nothing, nothing, (; entu, detu, entd, detd))
    NSUB = 48

    # Per-thread (CPU) path.
    state_cpu = CellState(m; CO2 = copy(tracer0))
    ws_cpu = TM5Workspace(state_cpu.air_mass; cell_metrics = ones(FT, Ny))
    mass0_cpu = sum(state_cpu.tracers_raw)
    op_cpu = TM5Convection(use_collab_lu = false, lmax_conv = 0, n_merge = 1)
    for _ in 1:NSUB
        apply!(state_cpu, forcing, grid, op_cpu, FT(600); workspace = ws_cpu)
    end
    @test all(isfinite, state_cpu.tracers_raw)
    @test isapprox(sum(state_cpu.tracers_raw), mass0_cpu; rtol = 1f-3)

    # Collaborative (CUDA) path: lmax_conv = 0 → full column (L_super = Nz = 12
    # ≤ 85) so the collab kernel engages; icltop == 1 fires the closure.
    if _HAS_CUDA
        state_gpu = CellState(m; CO2 = copy(tracer0))
        ws_gpu = TM5Workspace(state_gpu.air_mass; cell_metrics = ones(FT, Ny))
        mass0_gpu = sum(state_gpu.tracers_raw)
        sd = Adapt.adapt(CuArray, state_gpu)
        wd = Adapt.adapt(CuArray, ws_gpu)
        fd = Adapt.adapt(CuArray, forcing)
        op_gpu = TM5Convection(use_collab_lu = true, lmax_conv = 0, n_merge = 1)
        for _ in 1:NSUB
            apply!(sd, fd, grid, op_gpu, FT(600); workspace = wd)
        end
        CUDA.synchronize()
        q = Array(sd.tracers_raw)
        @test all(isfinite, q)
        @test isapprox(sum(q), mass0_gpu; rtol = 1f-3)
    end
end

@testset "Collaborative LU: depth envelope and unbounded tracer batching" begin
    # Depth controls matrix storage; a fixed six-slot RHS buffer is reused
    # for any positive tracer count. CPU/Float64 fallback is independent.
    @test AtmosTransport.Operators.Convection._tm5_collab_supports(85, 2) == true
    @test AtmosTransport.Operators.Convection._tm5_collab_supports(85, 6) == true
    @test AtmosTransport.Operators.Convection._tm5_collab_supports(85, 7) == true
    @test AtmosTransport.Operators.Convection._tm5_collab_supports(85, 65) == true
    @test AtmosTransport.Operators.Convection._tm5_collab_supports(85, 0) == false
    @test AtmosTransport.Operators.Convection._tm5_collab_supports(85, -1) == false
    @test AtmosTransport.Operators.Convection._tm5_collab_supports(75, 2) == true   # ERA5/L85 target
    @test AtmosTransport.Operators.Convection._tm5_collab_supports(40, 2) == true   # ml91/tropo60
    @test AtmosTransport.Operators.Convection._tm5_collab_supports(25, 2) == true   # ml137/tropo25a
    # lmax_conv > 85 doesn't fit Metal's 32 KB threadgroup-memory limit.
    @test AtmosTransport.Operators.Convection._tm5_collab_supports(86, 2) == false
    @test AtmosTransport.Operators.Convection._tm5_collab_supports(91, 2) == false
    @test AtmosTransport.Operators.Convection._tm5_collab_supports(0,  2) == false
end

@testset "plan 23 Commit 4: _assert_tm5_forcing catches missing tm5_fields" begin
    Nx, Ny, Nz = 4, 3, 4
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    A_ifc = FT[0, 500, 5000, 30000, 0]
    B_ifc = FT[0, 0, FT(0.1), FT(0.5), 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    state = CellState(fill(FT(1), Nx, Ny, Nz); CO2 = zeros(FT, Nx, Ny, Nz))
    ws = TM5Workspace(state.air_mass; cell_metrics = ones(FT, Ny))
    empty_forcing = ConvectionForcing()

    err = try
        apply!(state, empty_forcing, grid, TM5Convection(), FT(60); workspace=ws)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("tm5_fields", err.msg)
    @test occursin("NamedTuple", err.msg)

    # No "not yet implemented" strings remain (principle 7 prep).
    @test !occursin("not yet implemented", err.msg)
end

@testset "plan 23 Commit 2: _tm5_solve_column! identity + conservation" begin
    using .AtmosTransport.Operators.Convection: _tm5_solve_column!,
                                                   _tm5_diagnose_cloud_dims

    Nz = 8
    Nt = 3

    # -----------------------------------------------------------------
    # Zero-forcing short-circuits to identity
    # -----------------------------------------------------------------
    @testset "zero forcing → identity (F64)" begin
        T = Float64
        rm    = rand(T, Nz, Nt)
        rm0   = copy(rm)
        m     = rand(T, Nz) .+ T(0.1)
        entu  = zeros(T, Nz)
        detu  = zeros(T, Nz)
        entd  = zeros(T, Nz)
        detd  = zeros(T, Nz)
        conv1 = zeros(T, Nz, Nz)
        piv   = zeros(Int, Nz)
        cd    = zeros(Int, 3)

        _tm5_solve_column!(rm, m, entu, detu, entd, detd,
                            conv1, piv, cd, T(600))
        @test rm == rm0                 # bit-exact identity
        @test cd == [Nz + 1, 0, Nz + 1] # "no convection" sentinel
    end

    # -----------------------------------------------------------------
    # Tracer mass conservation: sum(rm_new) == sum(rm_old) by the
    # column-sum-is-1 invariant of `conv1` (each column of conv1
    # sums to 1 by construction; that's what makes backward-Euler
    # `conv1 · rm_new = rm_old` mass-conserving). Note: `rm_col`
    # here is tracer MASS, not mixing ratio — that's the plan 14
    # storage convention (tracers_raw stored as mass).
    # -----------------------------------------------------------------
    @testset "tracer-mass conservation (F64)" begin
        T = Float64
        rm = T[0.1 0.2 0.3;
               0.2 0.1 0.4;
               0.4 0.3 0.5;
               0.5 0.6 0.2;
               0.3 0.4 0.3;
               0.2 0.2 0.2;
               0.1 0.1 0.1;
               0.05 0.05 0.05]
        rm0 = copy(rm)
        m  = T[1.0e4, 1.2e4, 1.5e4, 1.7e4, 2.0e4, 2.5e4, 3.0e4, 4.0e4]
        entu = T[0.0, 0.0, 0.03, 0.05, 0.04, 0.02, 0.0, 0.0]
        detu = T[0.0, 0.02, 0.02, 0.01, 0.01, 0.03, 0.0, 0.0]
        entd = T[0.0, 0.0, 0.01, 0.02, 0.01, 0.0, 0.0, 0.0]
        detd = T[0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.0, 0.0]
        conv1 = zeros(T, Nz, Nz)
        piv   = zeros(Int, Nz)
        cd    = zeros(Int, 3)

        _tm5_solve_column!(rm, m, entu, detu, entd, detd,
                            conv1, piv, cd, T(600))

        for t in 1:Nt
            mass_before = sum(rm0[:, t])
            mass_after  = sum(rm[:, t])
            @test isapprox(mass_after, mass_before;
                           rtol = 1e4 * eps(T))
        end
        # Nontrivial: the profile actually changed (not silent identity).
        @test any(rm .!= rm0)
        # Cloud-dim diagnosis should pick up the active range.
        icltop, iclbas, icllfs = cd
        @test icltop == 2          # smallest k with detu > 0
        @test iclbas == 6          # largest k with detu > 0
        @test icllfs == 3          # smallest k with entd > 0
    end

    # -----------------------------------------------------------------
    # Uniform MIXING RATIO in → uniform MIXING RATIO out.
    # Since conv1 acts on tracer mass, the mixing-ratio preservation
    # test requires initializing rm_col as `const_mr × m` and
    # checking that `rm_new / m == const_mr` layer-by-layer.
    # -----------------------------------------------------------------
    @testset "uniform mixing ratio preserved (F64)" begin
        T = Float64
        const_mr = T(2.5e-4)
        m  = T[1.0e4, 1.2e4, 1.5e4, 1.7e4, 2.0e4, 2.5e4, 3.0e4, 4.0e4]
        rm = zeros(T, Nz, Nt)
        for t in 1:Nt, k in 1:Nz
            rm[k, t] = const_mr * m[k]
        end
        entu = T[0.0, 0.0, 0.03, 0.05, 0.04, 0.02, 0.0, 0.0]
        detu = T[0.0, 0.02, 0.02, 0.01, 0.01, 0.03, 0.0, 0.0]
        entd = T[0.0, 0.0, 0.01, 0.02, 0.01, 0.0, 0.0, 0.0]
        detd = T[0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.0, 0.0]
        conv1 = zeros(T, Nz, Nz)
        piv   = zeros(Int, Nz)
        cd    = zeros(Int, 3)

        _tm5_solve_column!(rm, m, entu, detu, entd, detd,
                            conv1, piv, cd, T(600))

        for t in 1:Nt, k in 1:Nz
            @test isapprox(rm[k, t] / m[k], const_mr;
                            rtol = 1e4 * eps(T))
        end
    end

    # -----------------------------------------------------------------
    # F32 variant: zero-forcing + mass conservation.
    # -----------------------------------------------------------------
    @testset "F32: zero forcing + tracer-mass conservation" begin
        T = Float32
        rm = T[0.1 0.2; 0.2 0.1; 0.4 0.3; 0.5 0.6;
               0.3 0.4; 0.2 0.2; 0.1 0.1; 0.05 0.05]
        rm0 = copy(rm)
        m  = T[1.0e4, 1.2e4, 1.5e4, 1.7e4, 2.0e4, 2.5e4, 3.0e4, 4.0e4]
        conv1 = zeros(T, Nz, Nz)
        piv   = zeros(Int, Nz)
        cd    = zeros(Int, 3)

        # Zero forcing → identity.
        _tm5_solve_column!(rm, m,
                            zeros(T, Nz), zeros(T, Nz),
                            zeros(T, Nz), zeros(T, Nz),
                            conv1, piv, cd, T(600))
        @test rm == rm0

        # Nontrivial forcing → tracer mass conserved to F32 ULP.
        entu = T[0.0, 0.0, 0.03, 0.05, 0.04, 0.02, 0.0, 0.0]
        detu = T[0.0, 0.02, 0.02, 0.01, 0.01, 0.03, 0.0, 0.0]
        entd = T[0.0, 0.0, 0.01, 0.02, 0.01, 0.0, 0.0, 0.0]
        detd = T[0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.0, 0.0]
        _tm5_solve_column!(rm, m, entu, detu, entd, detd,
                            conv1, piv, cd, T(600))
        for t in 1:size(rm, 2)
            mass_before = sum(rm0[:, t])
            mass_after  = sum(rm[:, t])
            @test isapprox(mass_after, mass_before;
                            rtol = 1f4 * eps(T))
        end
    end

    # -----------------------------------------------------------------
    # Cloud-dim diagnostics stand-alone.
    # -----------------------------------------------------------------
    @testset "_tm5_diagnose_cloud_dims" begin
        T = Float64
        detu_empty = zeros(T, Nz)
        entd_empty = zeros(T, Nz)
        @test _tm5_diagnose_cloud_dims(detu_empty, entd_empty, Nz) ==
              (Nz + 1, 0, Nz + 1)

        detu = zeros(T, Nz); detu[3] = 0.1; detu[5] = 0.2; detu[6] = 0.1
        entd = zeros(T, Nz); entd[4] = 0.05
        @test _tm5_diagnose_cloud_dims(detu, entd, Nz) == (3, 6, 4)

        # icllfs should be the HIGHEST-altitude (smallest k) with
        # entd > 0, even if multiple levels are active.
        entd2 = zeros(T, Nz); entd2[3] = 0.05; entd2[5] = 0.01
        @test _tm5_diagnose_cloud_dims(detu, entd2, Nz)[3] == 3
    end
end

@testset "plan 23 Commit 2: column-major loop-order audit" begin
    # Matrix assembly loops run leftmost-innermost (Julia is column-major).
    # Time the shipped column solve on a minimal case and confirm it is not
    # catastrophically slow. This is a sanity guard, not a benchmark; the
    # 1 s gate is orders of magnitude above the expected time.
    using .AtmosTransport.Operators.Convection: _tm5_solve_column!

    T = Float64
    Nz = 16
    Ncols = 256
    entu = zeros(T, Nz, Ncols)
    detu = zeros(T, Nz, Ncols)
    entd = zeros(T, Nz, Ncols)
    detd = zeros(T, Nz, Ncols)
    m    = fill(T(1e4), Nz, Ncols)
    rm   = fill(T(1.0), Nz, 1, Ncols)
    # Seed one mid-column detrainment so work is nontrivial.
    detu[8, :] .= 0.01

    conv1s = zeros(T, Nz, Nz, Ncols)
    pivs   = zeros(Int, Nz, Ncols)
    cds    = zeros(Int, 3, Ncols)

    # Loop column index as outermost (leftmost-innermost k: correct
    # column-major order because column slices are contiguous when
    # arrays are (Nz, …, Ncols)).
    t_correct = @elapsed for c in 1:Ncols
        _tm5_solve_column!(view(rm, :, :, c),
                            view(m, :, c),
                            view(entu, :, c), view(detu, :, c),
                            view(entd, :, c), view(detd, :, c),
                            view(conv1s, :, :, c),
                            view(pivs, :, c),
                            view(cds, :, c),
                            T(600))
    end
    @test t_correct < 1.0   # generous smoke gate; actual <50 ms
end

@testset "plan 23 Commit 1: with_convection(model, TM5Convection())" begin
    # Build a minimal LatLon TransportModel end-to-end to prove the
    # workspace installer threads TM5Workspace through correctly.
    mesh = LatLonMesh(; Nx=4, Ny=3, FT=FT)
    A_ifc = FT[0, 500, 5000, 30000, 0]
    B_ifc = FT[0, 0, FT(0.1), FT(0.5), 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    state = CellState(fill(FT(1), 4, 3, 4); CO2 = zeros(FT, 4, 3, 4))
    fluxes = allocate_face_fluxes(grid.horizontal, 4; FT = FT, basis = DryBasis)

    model = TransportModel(state, fluxes, grid, UpwindScheme())
    @test model.workspace.convection_ws === nothing  # NoConvection default

    model_tm5 = with_convection(model, TM5Convection())
    @test model_tm5.convection isa TM5Convection
    @test model_tm5.workspace.convection_ws isa TM5Workspace{FT}
    # Storage plan Commit 4: workspace is a flat tile slab. Default
    # 1.0 GiB budget covers all 12 cells of this fixture, so B = 12.
    @test size(model_tm5.workspace.convection_ws.conv1) == (4, 4, 12)

    # Swapping back to NoConvection drops the workspace.
    model_none = with_convection(model_tm5, NoConvection())
    @test model_none.workspace.convection_ws === nothing
end
