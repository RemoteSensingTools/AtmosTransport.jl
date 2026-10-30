#!/usr/bin/env julia
# `[advection] check_binary_cfl` (`BinaryCFLCheck`): a cubed-sphere split-sweep
# step that takes its subcycle count from the binary recomputes the runtime CFL
# budget and refuses a schedule that asks for fewer subcycles.

using Test
using Adapt
using AtmosTransport
using .AtmosTransport.Operators.Advection: fill_panel_halos!,
    strang_split_cs_mt!, CSAdvectionWorkspace,
    _cs_static_palindrome_subcycle_count
using .AtmosTransport.State: DryBasis
using .AtmosTransport.Operators: NoBinaryCFLCheck, BinaryCFLCheck

const M = AtmosTransport.Models
const CS_STYLE = M.CubedSphereRuntimeRecipeStyle()
const LL_STYLE = M.LatLonRuntimeRecipeStyle()

# Air mass 2 in level 1 (1 below) with `cm = 0.6` below every interior level-1
# cell: the cell ends the step with 2 − 2 × 0.6 = 0.8, so the palindrome budget
# is 2 × 0.6 / min(2, 0.8) = 1.5 and a 0.95 CFL limit needs 2 subcycles.
function steep_cm_fixture(; Nc = 4, Hp = 1, Nz = 3)
    mesh = CubedSphereMesh(; Nc, Hp, FT = Float64)
    N = Nc + 2Hp
    panels_m = ntuple(_ -> (m = ones(N, N, Nz); m[:, :, 1] .= 2.0; m), 6)
    panels_rm = ntuple(_ -> fill(4e-4, N, N, Nz), 6)
    fill_panel_halos!(panels_m, mesh; dir = 0)
    fill_panel_halos!(panels_rm, mesh; dir = 0)
    panels_am = ntuple(_ -> zeros(N + 1, N, Nz), 6)
    panels_bm = ntuple(_ -> zeros(N, N + 1, Nz), 6)
    panels_cm = ntuple(6) do _
        cm = zeros(N, N, Nz + 1)
        cm[(Hp + 1):(Hp + Nc), (Hp + 1):(Hp + Nc), 2] .= 0.6
        cm
    end
    return (; mesh, Nz, panels_m, panels_rm, panels_am, panels_bm, panels_cm)
end

@testset "split-sweep operators check the binary's subcycle count" begin
    f = steep_cm_fixture()
    (; mesh, Nz) = f
    @test _cs_static_palindrome_subcycle_count(f.panels_am, f.panels_bm, f.panels_cm,
                                               f.panels_m, mesh.Nc, mesh.Hp, Nz, 0.95) == 2

    packed(op, ws; kw...) = begin
        rm = ntuple(p -> reshape(copy(f.panels_rm[p]), size(f.panels_rm[p])..., 1), 6)
        m = deepcopy(f.panels_m)
        op(rm, m, f.panels_am, f.panels_bm, f.panels_cm, mesh, UpwindScheme(), ws; kw...)
        (rm, m)
    end
    checked() = CSAdvectionWorkspace(mesh, Nz; n_tracers = 1, binary_cfl_check = BinaryCFLCheck())
    trusting() = CSAdvectionWorkspace(mesh, Nz; n_tracers = 1)

    runs = (("strang_split_cs_mt_pingpong!", ws -> packed(strang_split_cs_mt!, ws; subcycle_count = 1),
                                             ws -> packed(strang_split_cs_mt!, ws; subcycle_count = 2),
                                             ws -> packed(strang_split_cs_mt!, ws)),)
    for (caller, too_few, enough, runtime) in runs
        err = try too_few(checked()); nothing catch e; e end
        @test err isa ArgumentError && contains(err.msg, caller) &&
              contains(err.msg, "check_binary_cfl")
        @test too_few(trusting()) isa Tuple                  # the default trusts the binary
        @test enough(checked()) == enough(trusting())        # a passing check changes nothing
        @test runtime(checked()) == runtime(trusting())      # runtime counts are never checked
    end
end

@testset "TransportModel and workspaces carry the check" begin
    mesh = CubedSphereMesh(; FT = Float64, Nc = 4)
    grid = AtmosGrid(mesh, HybridSigmaPressure([0.0, 100.0, 300.0], [0.0, 0.0, 1.0]),
                     AtmosTransport.CPU(); FT = Float64)
    N, Nz = mesh.Nc + 2mesh.Hp, 2
    state = CubedSphereState(DryBasis, mesh, ntuple(_ -> ones(N, N, Nz), 6);
                             CO2 = ntuple(_ -> fill(4e-4, N, N, Nz), 6))
    fluxes = allocate_face_fluxes(mesh, Nz; FT = Float64, basis = DryBasis)
    ws_of(model) = model.workspace.advection_ws

    @test ws_of(TransportModel(state, fluxes, grid, UpwindScheme())).binary_cfl_check ===
          NoBinaryCFLCheck()
    model = TransportModel(state, fluxes, grid, UpwindScheme(); binary_cfl_check = BinaryCFLCheck())
    @test ws_of(model).binary_cfl_check === BinaryCFLCheck()
    @test Adapt.adapt(Array, ws_of(model)).binary_cfl_check === BinaryCFLCheck()
    # Schemes without a binary subcycle count reject the check.
    @test_throws ArgumentError TransportModel(state, fluxes, grid, LinRoodPPMScheme(5);
                                              binary_cfl_check = BinaryCFLCheck())
    @test_throws ArgumentError TransportModel(state, fluxes, grid, NoAdvection();
                                              binary_cfl_check = BinaryCFLCheck())
    # A supplied workspace must carry a requested check.
    @test_throws ArgumentError TransportModel(state, fluxes, grid, UpwindScheme();
                                              binary_cfl_check = BinaryCFLCheck(),
                                              advection_workspace = CSAdvectionWorkspace(mesh, Nz; n_tracers = 1))
end
