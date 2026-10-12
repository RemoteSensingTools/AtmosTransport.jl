#!/usr/bin/env julia
"""
Plan 23 Commit 5: cross-scheme consistency of TM5Convection and
CMFMCConvection.

Both schemes transport tracers through cumulus convection but have
different implementations and unit conventions:

- CMFMCConvection: GCHP RAS-style updraft mixing (`qc`) followed by an
  explicit, CFL-subcycled interface-flux divergence
  `Φ(k) = cmfmc[k]·(qc[k] − q_env(k−1))`, with a well-mixed sub-cloud
  layer. `air_mass` in kg per cell (the kernel divides by `cell_area`).
- TM5Convection: implicit backward-Euler matrix solve,
  unconditionally stable. Production `air_mass` is kg per cell,
  and the kernel uses `cell_metrics` to divide by kg/m²; these
  idealized checks pass unit-area metrics so their synthetic
  `m = 5e3` columns still represent kg/m².

Because the natural unit conventions differ, this test does NOT
attempt a profile-by-profile numerical agreement between the two
schemes. Driven by matched idealized profiles on one lat-lon grid, both
must preserve a uniform mixing ratio at the state level (`apply!` on a
`CellState`).

Mass conservation and the non-identity response are tested per scheme
elsewhere: CMFMC in `test_transport_model_convection.jl` (lat-lon and
reduced Gaussian, rtol 1e-12) and `test_cmfmc_convection.jl`, TM5 in
`test_tm5_convection.jl` (lat-lon, reduced Gaussian and cubed sphere).
"""

using Test

using AtmosTransport

using .AtmosTransport.State: CellState
using .AtmosTransport.Grids: AtmosGrid, LatLonMesh, HybridSigmaPressure,
                              cell_areas_by_latitude
using .AtmosTransport.Operators: CMFMCConvection, TM5Convection,
                                 CMFMCWorkspace, TM5Workspace
using .AtmosTransport.MetDrivers: ConvectionForcing

const FT = Float64

# -------------------------------------------------------------------------
# Idealized forcing profile for a single column.
# -------------------------------------------------------------------------
function _cmfmc_column(Nz::Int, k_top::Int; FT = Float64, peak = FT(0.05))
    cmfmc  = zeros(FT, Nz + 1)
    dtrain = zeros(FT, Nz)
    for k in k_top:Nz
        frac = FT((Nz - k + 1) / (Nz - k_top + 1))
        cmfmc[k + 1] = peak * frac * (1 - frac)
    end
    cmfmc[Nz + 1] = zero(FT)
    for k in (k_top + 1):(k_top + 2)
        dtrain[k] = FT(0.01)
    end
    return (cmfmc, dtrain)
end

function _tm5_column(Nz::Int, k_top::Int; FT = Float64)
    # Matching idealized entrainment/detrainment profile for TM5:
    # entu peaks at mid-cloud, detu at cloud top, no downdraft.
    entu = zeros(FT, Nz)
    detu = zeros(FT, Nz)
    entd = zeros(FT, Nz)
    detd = zeros(FT, Nz)
    for k in k_top:Nz
        frac = FT((Nz - k + 1) / (Nz - k_top + 1))
        entu[k] = FT(0.03) * frac * (1 - frac)
    end
    for k in (k_top + 1):(k_top + 2)
        detu[k] = FT(0.01)
    end
    return (; entu, detu, entd, detd)
end

# -------------------------------------------------------------------------
# Structured LatLon
# -------------------------------------------------------------------------

@testset "plan 23 Commit 5: CMFMC and TM5 preserve uniform mixing ratio on LatLon" begin
    Nx, Ny, Nz = 2, 2, 8
    k_top = 3

    mesh = LatLonMesh(; Nx = Nx, Ny = Ny, FT = FT)
    A_ifc = FT[0, 500, 1000, 2000, 5000, 10000, 30000, 50000, 0]
    B_ifc = FT[0, 0, 0, 0.05, 0.2, 0.4, 0.7, 0.9, 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
    cell_areas = cell_areas_by_latitude(mesh)

    # Forcings (same column tiled over all cells).
    cmfmc_col, dtrain_col = _cmfmc_column(Nz, k_top; FT = FT)
    tm5_col = _tm5_column(Nz, k_top; FT = FT)
    cmfmc_3d  = repeat(reshape(cmfmc_col, 1, 1, Nz + 1),  Nx, Ny, 1)
    dtrain_3d = repeat(reshape(dtrain_col, 1, 1, Nz), Nx, Ny, 1)
    entu_3d   = repeat(reshape(tm5_col.entu, 1, 1, Nz), Nx, Ny, 1)
    detu_3d   = repeat(reshape(tm5_col.detu, 1, 1, Nz), Nx, Ny, 1)
    entd_3d   = repeat(reshape(tm5_col.entd, 1, 1, Nz), Nx, Ny, 1)
    detd_3d   = repeat(reshape(tm5_col.detd, 1, 1, Nz), Nx, Ny, 1)

    # -- CMFMC path — `m` in kg/cell --
    m_cmfmc = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m_cmfmc[i, j, k] = FT(5e3) * cell_areas[j]
    end
    dt = FT(60)

    @testset "CMFMC: preserves uniform mixing ratio" begin
        mr = FT(2.5e-4)
        tracer_init = mr .* m_cmfmc
        state = CellState(m_cmfmc; CO2 = copy(tracer_init))
        forcing = ConvectionForcing(cmfmc_3d, dtrain_3d, nothing)
        ws = CMFMCWorkspace(state.air_mass; cell_metrics = cell_areas)
        AtmosTransport.Operators.apply!(state, forcing, grid,
                                         CMFMCConvection(), dt;
                                         workspace = ws)
        mr_after = state.tracers_raw[:, :, :, 1] ./ m_cmfmc
        @test all(isapprox.(mr_after, mr; rtol = 1e-8))
    end

    # -- TM5 path — `m` in kg/m² --
    m_tm5 = fill(FT(5e3), Nx, Ny, Nz)

    @testset "TM5: preserves uniform mixing ratio" begin
        mr = FT(2.5e-4)
        tracer_init = mr .* m_tm5
        state = CellState(m_tm5; CO2 = copy(tracer_init))
        tm5_fields = (entu = entu_3d, detu = detu_3d,
                       entd = entd_3d, detd = detd_3d)
        forcing = ConvectionForcing(nothing, nothing, tm5_fields)
        ws = TM5Workspace(state.air_mass; cell_metrics = ones(FT, Ny))
        AtmosTransport.Operators.apply!(state, forcing, grid,
                                         TM5Convection(), dt;
                                         workspace = ws)
        mr_after = state.tracers_raw[:, :, :, 1] ./ m_tm5
        @test all(isapprox.(mr_after, mr; rtol = 1e-10))
    end
end
