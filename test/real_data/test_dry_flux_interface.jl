#!/usr/bin/env julia
"""
Validation tests for dry-mass face-flux transport architecture.

Tests:
1. Module loading and type construction
2. Geometry API correctness for LatLonMesh
3. CellState and FaceFluxState construction
4. Uniform field invariance through Strang splitting
5. Mass conservation (total tracer mass preserved)
6. Vertical flux diagnosis (cm from horizontal continuity)
7. Numerical equivalence across mass-flux advection paths
"""

using Test

# Load module
include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport: Grids, State, Operators, MetDrivers
using .AtmosTransport.Grids: FaceIndexedFluxTopology, StructuredFluxTopology,
                             cell_areas_by_latitude, cell_faces, face_cells,
                             face_length, face_normal, flux_topology, n_levels
using .AtmosTransport.MetDrivers: diagnose_cm_from_continuity!
using .AtmosTransport.Operators: MonotoneLimiter, strang_split!
using .AtmosTransport.State: AbstractFaceFluxState,
                             AbstractStructuredFaceFluxState,
                             DryBasis, MoistBasis,
                             StructuredFaceFluxState, face_flux_x,
                             face_flux_y, face_flux_z, mass_basis

# =========================================================================
# Test 1: Module loading and type construction
# =========================================================================
@testset "module loading" begin
    @test isdefined(AtmosTransport, :LatLonMesh)
    @test isdefined(AtmosTransport, :CellState)
    @test isdefined(AtmosTransport, :AbstractFaceFluxState)
    @test isdefined(AtmosTransport, :AbstractStructuredFaceFluxState)
    @test isdefined(AtmosTransport, :StructuredFaceFluxState)
    @test isdefined(AtmosTransport, :FaceIndexedFluxState)
    @test isdefined(AtmosTransport, :SlopesScheme)
    @test isdefined(AtmosTransport, :AtmosGrid)
    @test isdefined(AtmosTransport, :HybridSigmaPressure)
    @test isdefined(AtmosTransport, :StructuredFluxTopology)
    @test isdefined(AtmosTransport, :FaceIndexedFluxTopology)
    @test isdefined(AtmosTransport, :flux_topology)
    # Basis types
    @test isdefined(AtmosTransport, :AbstractMassBasis)
    @test isdefined(AtmosTransport, :MoistBasis)
    @test isdefined(AtmosTransport, :DryBasis)
    @test isdefined(AtmosTransport, :mass_basis)
end

@testset "Flux topology trait" begin
    mesh_ll = LatLonMesh(; Nx=10, Ny=8, FT=Float64)
    @test flux_topology(mesh_ll) isa StructuredFluxTopology

    sfs = StructuredFaceFluxState(zeros(11,8,4), zeros(10,9,4), zeros(10,8,5))
    @test sfs isa AbstractStructuredFaceFluxState
    @test sfs isa AbstractFaceFluxState
    @test sfs isa StructuredFaceFluxState{DryBasis}

    @test face_flux_x(sfs, 1, 1, 1) == 0.0
    @test face_flux_y(sfs, 1, 1, 1) == 0.0
    @test face_flux_z(sfs, 1, 1, 1) == 0.0
end

@testset "Mass basis type safety" begin
    am = zeros(11, 8, 4); bm = zeros(10, 9, 4); cm = zeros(10, 8, 5)

    dry = StructuredFaceFluxState{DryBasis}(am, bm, cm)
    @test mass_basis(dry) isa DryBasis

    moist = StructuredFaceFluxState{MoistBasis}(am, bm, cm)
    @test mass_basis(moist) isa MoistBasis

    default = StructuredFaceFluxState(am, bm, cm)
    @test mass_basis(default) isa DryBasis

    @test !(moist isa StructuredFaceFluxState{DryBasis})
    @test !(dry isa StructuredFaceFluxState{MoistBasis})

    alloc_dry = allocate_face_fluxes(StructuredFluxTopology(), 10, 8, 4;
                                      basis=DryBasis)
    @test mass_basis(alloc_dry) isa DryBasis

    alloc_moist = allocate_face_fluxes(StructuredFluxTopology(), 10, 8, 4;
                                        basis=MoistBasis)
    @test mass_basis(alloc_moist) isa MoistBasis
end

# =========================================================================
# Test 1b: Basis safety — strang_split! only accepts DryBasis.
#           A mock subtype of AbstractStructuredFaceFluxState (not
#           StructuredFaceFluxState) is rejected by dispatch.
#           StructuredFaceFluxState{MoistBasis} is also rejected.
# =========================================================================
struct MockStructuredFluxState{B, AX, AY, AZ} <: AbstractStructuredFaceFluxState{B}
    am :: AX
    bm :: AY
    cm :: AZ
end

MockStructuredFluxState{B}(am, bm, cm) where {B} =
    MockStructuredFluxState{B, typeof(am), typeof(bm), typeof(cm)}(am, bm, cm)

@testset "Basis safety: strang_split! rejects non-dry flux states" begin
    FT = Float64
    Nx, Ny, Nz = 10, 8, 4

    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    A = FT[0.0, 500.0, 5000.0, 30000.0, 0.0]
    B = FT[0.0, 0.0,   0.1,    0.5,     1.0]
    vc = HybridSigmaPressure(A, B)
    grid = AtmosGrid(mesh, vc, AtmosTransport.Architectures.CPU(); FT=FT)

    g = grid.gravity
    ps_val = grid.reference_pressure
    areas = cell_areas_by_latitude(mesh)

    m = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp_k = level_thickness(vc, k, ps_val)
        m[i, j, k] = dp_k * areas[j] / g
    end
    rm = m .* FT(400e-6)
    state = CellState(m; CO2=rm)

    am = zeros(FT, Nx+1, Ny, Nz)
    bm = zeros(FT, Nx, Ny+1, Nz)
    cm = zeros(FT, Nx, Ny, Nz+1)

    scheme = SlopesScheme(MonotoneLimiter())
    ws = AdvectionWorkspace(state)

    # Mock subtype → MethodError (not StructuredFaceFluxState at all)
    mock_fluxes = MockStructuredFluxState{DryBasis}(am, bm, cm)
    @test mock_fluxes isa AbstractStructuredFaceFluxState
    @test !(mock_fluxes isa StructuredFaceFluxState)
    @test mass_basis(mock_fluxes) isa DryBasis
    @test_throws MethodError strang_split!(state, mock_fluxes, grid, scheme; workspace=ws)

    # Moist basis → MethodError (wrong basis)
    moist_fluxes = StructuredFaceFluxState{MoistBasis}(am, bm, cm)
    @test_throws MethodError strang_split!(state, moist_fluxes, grid, scheme; workspace=ws)

    # Dry basis → works
    dry_fluxes = StructuredFaceFluxState{DryBasis}(am, bm, cm)
    m_before = sum(state.air_mass)
    rm_before = total_mass(state, :CO2)
    strang_split!(state, dry_fluxes, grid, scheme; workspace=ws)
    @test sum(state.air_mass) ≈ m_before atol=eps(FT)*m_before*1000
    @test total_mass(state, :CO2)  ≈ rm_before atol=eps(FT)*rm_before*1000

    # Scoped accessors still work for mock and moist types
    @test face_flux_x(mock_fluxes, 1, 1, 1) == 0.0
    @test face_flux_y(moist_fluxes, 1, 1, 1) == 0.0
    @test face_flux_z(dry_fluxes, 1, 1, 1) == 0.0
end

# =========================================================================
# Test 1c: Universal face geometry API
# =========================================================================
@testset "Universal face_length and face_normal" begin
    mesh = LatLonMesh(; Nx=10, Ny=8, FT=Float64)
    R = mesh.radius

    n_xfaces = (mesh.Nx + 1) * mesh.Ny
    total = nfaces(mesh)
    @test total == n_xfaces + mesh.Nx * (mesh.Ny + 1)

    fl_x = face_length(mesh, 1)
    @test fl_x ≈ R * deg2rad(mesh.Δφ)

    # y-face at index n_xfaces+1 is j=1 (south pole), where cos(φ)=0
    fl_y_pole = face_length(mesh, n_xfaces + 1)
    @test fl_y_pole >= 0.0
    # A y-face away from the pole should have positive length
    fl_y_mid = face_length(mesh, n_xfaces + mesh.Nx * div(mesh.Ny, 2) + 1)
    @test fl_y_mid > 0.0

    nx_x, ny_x = face_normal(mesh, 1)
    @test nx_x == 1.0
    @test ny_x == 0.0

    nx_y, ny_y = face_normal(mesh, n_xfaces + 1)
    @test nx_y == 0.0
    @test ny_y == 1.0
end

# =========================================================================
# Test 2: Geometry API correctness for LatLonMesh
# =========================================================================
@testset "LatLonMesh geometry" begin
    mesh = LatLonMesh(; Nx=60, Ny=45, FT=Float64)

    @test nx(mesh) == 60
    @test ny(mesh) == 45
    @test ncells(mesh) == 60 * 45

    # Cell area at equator should be approximately R² Δλ Δφ
    R = 6.371e6
    Δλ_rad = deg2rad(mesh.Δλ)
    area_eq_j = div(mesh.Ny, 2) + 1  # near equator
    a = cell_area(mesh, 1, area_eq_j)
    @test a > 0.0
    # Rough check: equatorial area ~ R² × Δλ_rad × sin(Δφ/2)*2 ≈ R² Δλ Δφ (cos≈1)
    Δφ_rad = deg2rad(mesh.Δφ)
    expected_equator = R^2 * Δλ_rad * abs(sind(mesh.φᶠ[area_eq_j+1]) - sind(mesh.φᶠ[area_eq_j]))
    @test a ≈ expected_equator

    # Total area should approximate Earth's surface (4πR²)
    total_area = sum(cell_area(mesh, 1, j) for j in 1:mesh.Ny) * mesh.Nx
    earth_area = 4π * R^2
    @test abs(total_area - earth_area) / earth_area < 1e-10

    # dx at equator should be R × Δλ_rad × cos(0) ≈ R × Δλ_rad
    dx_eq = Grids.dx(mesh, area_eq_j)
    @test dx_eq > 0.0

    # dy should be uniform
    dy_val = Grids.dy(mesh)
    @test dy_val ≈ R * Δφ_rad

    # face_cells connectivity
    f = 1
    lc, rc = face_cells(mesh, f)
    @test lc isa Integer
    @test rc isa Integer

    # cell_faces — returns (west, east, south, north) face indices for a cell
    Nx, Ny = mesh.Nx, mesh.Ny
    n_xfaces = (Nx + 1) * Ny

    # Interior cell (i=3, j=4)
    w, e, s, n = cell_faces(mesh, (3, 4))
    @test w == (4-1)*(Nx+1) + 3
    @test e == (4-1)*(Nx+1) + 4
    @test s == n_xfaces + (4-1)*Nx + 3
    @test n == n_xfaces + 4*Nx + 3

    # Flat-index version: cell c = i + (j-1)*Nx = 3 + 3*60 = 183
    c_flat = 3 + (4-1)*Nx
    w2, e2, s2, n2 = cell_faces(mesh, c_flat)
    @test (w2, e2, s2, n2) == (w, e, s, n)

    # Round-trip: face_cells ∘ cell_faces should be consistent
    # The west face of cell (3,4) should have (3,4) as its right cell
    _, rc_w = face_cells(mesh, w)
    @test rc_w == 3 + (4-1)*Nx  # flat index of cell (3,4)

    # The east face of cell (3,4) should have (3,4) as its left cell
    lc_e, _ = face_cells(mesh, e)
    @test lc_e == 3 + (4-1)*Nx

    # Corner cell (1,1): west face wraps to Nx
    w1, _, _, _ = cell_faces(mesh, (1, 1))
    lc1, _ = face_cells(mesh, w1)
    @test lc1 == Nx + (1-1)*Nx  # flat index of (Nx, 1)
end

# =========================================================================
# Test 3: AtmosGrid composite construction
# =========================================================================
@testset "AtmosGrid construction" begin
    mesh = LatLonMesh(; Nx=10, Ny=8, FT=Float64)

    A = Float64[0.0, 1000.0, 5000.0, 20000.0, 50000.0]
    B = Float64[1.0, 0.9, 0.6, 0.2, 0.0]
    vc = HybridSigmaPressure(A, B)
    @test n_levels(vc) == 4

    grid = AtmosGrid(mesh, vc, AtmosTransport.Architectures.CPU();
                     FT=Float64, radius=6.371e6, gravity=9.80665)
    @test nlevels(grid) == 4
    @test floattype(grid) == Float64
end

# =========================================================================
# Helpers: create test grid and state
# =========================================================================
function make_test_grid(; Nx=60, Ny=30, Nz=4, FT=Float64)
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)

    # Simple 4-level vertical: A + B*ps = pressure at interfaces
    A = FT[0.0, 5000.0, 20000.0, 50000.0, 80000.0]
    B = FT[1.0, 0.75, 0.4, 0.1, 0.0]
    # Normalize so p_surface = A[end]+B[end]*ps matches
    # With these values: p_top = A[1]+B[1]*ps = ps, p_bot = A[5]+B[5]*ps = 80000
    # Let's use proper hybrid: TOA is index 1
    A_proper = FT[0.0, 500.0, 5000.0, 30000.0, 0.0]
    B_proper = FT[0.0, 0.0,   0.1,    0.5,     1.0]
    vc = HybridSigmaPressure(A_proper, B_proper)

    # Use a simple CPU() arch. Since the module doesn't re-export legacy Architectures,
    # we create a minimal stand-in.
    grid = AtmosGrid(mesh, vc, AtmosTransport.Architectures.CPU();
                     FT=FT, radius=FT(6.371e6), gravity=FT(9.80665))
    return grid
end

function make_test_state(grid; χ=400e-6)
    FT = floattype(grid)
    mesh = grid.horizontal
    vc = grid.vertical
    Nx, Ny = nx(mesh), ny(mesh)
    Nz = n_levels(vc)
    g = grid.gravity
    ps_val = grid.reference_pressure

    areas = cell_areas_by_latitude(mesh)

    # Compute dry air mass (no humidity for simplicity)
    m = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp_k = level_thickness(vc, k, ps_val)
        m[i, j, k] = dp_k * areas[j] / g
    end

    # Tracer: uniform mixing ratio
    rm = m .* FT(χ)

    state = CellState(m; CO2=rm)
    return state
end

function make_zero_fluxes(grid)
    FT = floattype(grid)
    mesh = grid.horizontal
    Nx, Ny = nx(mesh), ny(mesh)
    Nz = n_levels(grid.vertical)
    return allocate_face_fluxes(StructuredFluxTopology(), Nx, Ny, Nz; FT=FT)
end

function make_simple_fluxes(grid; scale=1e8)
    FT = floattype(grid)
    mesh = grid.horizontal
    Nx, Ny = nx(mesh), ny(mesh)
    Nz = n_levels(grid.vertical)

    fluxes = allocate_face_fluxes(StructuredFluxTopology(), Nx, Ny, Nz; FT=FT)

    # Simple zonal flow: am is uniform positive, small enough for CFL < 1
    m_state = make_test_state(grid)
    m_min = minimum(m_state.air_mass)
    am_val = FT(0.1) * m_min  # CFL ~ 0.1

    for k in 1:Nz, j in 1:Ny, i in 1:Nx+1
        fluxes.am[i, j, k] = am_val
    end
    # Zero out polar am at j=1 and j=Ny boundaries
    fluxes.am[:, 1, :] .= zero(FT)

    # Diagnose consistent cm from am (bm=0)
    bt = FT[Grids.b_diff(grid.vertical, k) for k in 1:Nz]
    for j in 1:Ny, i in 1:Nx
        pit = zero(FT)
        for k in 1:Nz
            pit += fluxes.am[i, j, k] - fluxes.am[i+1, j, k] +
                   fluxes.bm[i, j, k] - fluxes.bm[i, j+1, k]
        end
        acc = zero(FT)
        fluxes.cm[i, j, 1] = acc
        for k in 1:Nz
            conv_k = fluxes.am[i, j, k] - fluxes.am[i+1, j, k] +
                     fluxes.bm[i, j, k] - fluxes.bm[i, j+1, k]
            acc += conv_k - bt[k] * pit
            fluxes.cm[i, j, k+1] = acc
        end
    end

    return fluxes
end

# =========================================================================
# Test 4: CellState and FaceFluxState construction
# =========================================================================
@testset "State construction" begin
    grid = make_test_grid()
    state = make_test_state(grid)

    @test size(state.air_mass) == (60, 30, 4)
    @test :CO2 in tracer_names(state)
    @test total_air_mass(state) > 0.0
    @test total_mass(state, :CO2) > 0.0

    # Mixing ratio should be uniform
    χ = mixing_ratio(state, :CO2)
    @test maximum(χ) ≈ 400e-6
    @test minimum(χ) ≈ 400e-6

    fluxes = make_zero_fluxes(grid)
    @test size(fluxes.am) == (61, 30, 4)
    @test size(fluxes.bm) == (60, 31, 4)
    @test size(fluxes.cm) == (60, 30, 5)
end

# =========================================================================
# Test 5: Uniform field invariance (zero fluxes)
# =========================================================================
@testset "Uniform field invariance (zero flux)" begin
    grid = make_test_grid()
    state = make_test_state(grid; χ=400e-6)
    fluxes = make_zero_fluxes(grid)

    scheme = SlopesScheme(MonotoneLimiter())
    ws = AdvectionWorkspace(state)

    m_before = sum(state.air_mass)
    rm_before = total_mass(state, :CO2)

    strang_split!(state, fluxes, grid, scheme; workspace=ws)

    m_after = sum(state.air_mass)
    rm_after = total_mass(state, :CO2)

    @test m_after ≈ m_before atol=eps(Float64)*m_before*1000
    @test rm_after ≈ rm_before atol=eps(Float64)*rm_before*1000

    # Mixing ratio should remain uniform
    χ = mixing_ratio(state, :CO2)
    @test maximum(χ) ≈ minimum(χ) atol=1e-15
end

# =========================================================================
# Test 6: Mass conservation with non-zero fluxes
# =========================================================================
@testset "Mass conservation (zonal flow)" begin
    grid = make_test_grid()
    state = make_test_state(grid; χ=400e-6)
    fluxes = make_simple_fluxes(grid)

    scheme = SlopesScheme(MonotoneLimiter())
    ws = AdvectionWorkspace(state)

    m_before = sum(state.air_mass)
    rm_before = total_mass(state, :CO2)

    strang_split!(state, fluxes, grid, scheme; workspace=ws)

    m_after = sum(state.air_mass)
    rm_after = total_mass(state, :CO2)

    # Air mass should be conserved to machine precision
    @test abs(m_after - m_before) / m_before < 1e-12

    # Tracer mass should be conserved to machine precision
    @test abs(rm_after - rm_before) / rm_before < 1e-12
end

# =========================================================================
# Test 7: Vertical flux diagnosis
# =========================================================================
@testset "Vertical flux diagnosis (cm from continuity)" begin
    grid = make_test_grid()
    FT = Float64
    mesh = grid.horizontal
    Nx, Ny = nx(mesh), ny(mesh)
    Nz = n_levels(grid.vertical)

    # Random am, bm
    am = randn(FT, Nx+1, Ny, Nz) .* 1e6
    bm = randn(FT, Nx, Ny+1, Nz) .* 1e6
    cm = zeros(FT, Nx, Ny, Nz+1)

    bt = FT[Grids.b_diff(grid.vertical, k) for k in 1:Nz]

    # Diagnose cm
    diagnose_cm_from_continuity!(cm, am, bm, bt, Nx, Ny, Nz)

    # Verify: cm[1] = 0 (TOA), cm[Nz+1] ≈ 0 (surface, if bt sums properly)
    for j in 1:Ny, i in 1:Nx
        @test cm[i, j, 1] == 0.0
    end

    # Verify continuity: for each cell, horizontal convergence + vertical flux
    # divergence should equal bt[k] * pit
    for j in 1:Ny, i in 1:Nx
        pit = zero(FT)
        for k in 1:Nz
            pit += am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
        end
        for k in 1:Nz
            conv_h = am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
            div_v = cm[i, j, k+1] - cm[i, j, k]
            # conv_h - div_v should = bt[k] * pit
            @test abs(conv_h - div_v - bt[k] * pit) < 1e-10 * max(abs(pit), 1.0)
        end
    end
end

# =========================================================================
# Test 8 (numerical equivalence v1 vs v2) was removed: the v1/v2
# distinction no longer exists after the `src_v2 → src` promotion.
# The v1 paths (`LatitudeLongitudeGrid`, `strang_split_massflux!`,
# `allocate_massflux_workspace`) are gone from `src/`. Removed per
# plan 18 PRE_PLAN_18_FIXES.md §A2.
# =========================================================================

# =========================================================================
# Test 9: cell_area with flat integer index
# =========================================================================
@testset "cell_area flat index" begin
    mesh = LatLonMesh(; Nx=60, Ny=45, FT=Float64)
    Nx, Ny = mesh.Nx, mesh.Ny

    for j in 1:Ny
        c_flat = 1 + (j - 1) * Nx
        @test cell_area(mesh, c_flat) ≈ cell_area(mesh, 1, j)
    end

    for j in [1, div(Ny, 2), Ny], i in [1, div(Nx, 2), Nx]
        c_flat = i + (j - 1) * Nx
        @test cell_area(mesh, c_flat) ≈ cell_area(mesh, (i, j))
    end
end

# =========================================================================
# Test 10: Multi-tracer mass conservation
# =========================================================================
@testset "Multi-tracer mass conservation" begin
    grid = make_test_grid(; Nx=60, Ny=30, Nz=4)
    FT = floattype(grid)
    mesh = grid.horizontal
    vc = grid.vertical
    Nx, Ny = nx(mesh), ny(mesh)
    Nz = n_levels(vc)
    g = grid.gravity
    ps_val = grid.reference_pressure
    areas = cell_areas_by_latitude(mesh)

    m = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp_k = level_thickness(vc, k, ps_val)
        m[i, j, k] = dp_k * areas[j] / g
    end

    rm_co2 = m .* FT(400e-6)
    rm_sf6 = m .* FT(10e-12)
    rm_ch4 = m .* FT(1.8e-6)

    state = CellState(m; CO2=rm_co2, SF6=rm_sf6, CH4=rm_ch4)
    @test length(tracer_names(state)) == 3
    @test :CO2 in tracer_names(state)
    @test :SF6 in tracer_names(state)
    @test :CH4 in tracer_names(state)

    fluxes = make_simple_fluxes(grid)
    scheme = SlopesScheme(MonotoneLimiter())
    ws = AdvectionWorkspace(state)

    m_before = sum(state.air_mass)
    co2_before = total_mass(state, :CO2)
    sf6_before = total_mass(state, :SF6)
    ch4_before = total_mass(state, :CH4)

    strang_split!(state, fluxes, grid, scheme; workspace=ws)

    m_after = sum(state.air_mass)
    co2_after = total_mass(state, :CO2)
    sf6_after = total_mass(state, :SF6)
    ch4_after = total_mass(state, :CH4)

    @test abs(m_after - m_before) / m_before < 1e-12
    @test abs(co2_after - co2_before) / co2_before < 1e-12
    @test abs(sf6_after - sf6_before) / sf6_before < 1e-12
    @test abs(ch4_after - ch4_before) / ch4_before < 1e-12

    # Each tracer should have a different field (not accidentally shared)
    χ_co2 = mixing_ratio(state, :CO2)
    χ_sf6 = mixing_ratio(state, :SF6)
    @test maximum(abs.(χ_co2 .- χ_sf6)) > 1e-6
end

# =========================================================================
# Test 11: Multi-window time-stepping (synthetic)
# =========================================================================
@testset "Multi-window time stepping" begin
    grid = make_test_grid(; Nx=60, Ny=30, Nz=4)
    FT = floattype(grid)
    mesh = grid.horizontal
    vc = grid.vertical
    Nx, Ny = nx(mesh), ny(mesh)
    Nz = n_levels(vc)
    g = grid.gravity
    ps_val = grid.reference_pressure
    areas = cell_areas_by_latitude(mesh)

    m_init = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp_k = level_thickness(vc, k, ps_val)
        m_init[i, j, k] = dp_k * areas[j] / g
    end

    state = CellState(copy(m_init); CO2 = m_init .* FT(400e-6))
    scheme = SlopesScheme(MonotoneLimiter())
    ws = AdvectionWorkspace(state)

    m_total_0 = sum(state.air_mass)
    rm_total_0 = total_mass(state, :CO2)

    n_windows = 4
    for win in 1:n_windows
        # Rebuild fluxes each window (simulating fresh met data)
        fluxes = make_simple_fluxes(grid)

        m_before = sum(state.air_mass)
        rm_before = total_mass(state, :CO2)

        strang_split!(state, fluxes, grid, scheme; workspace=ws)

        m_after = sum(state.air_mass)
        rm_after = total_mass(state, :CO2)

        @test abs(m_after - m_before) / m_before < 1e-12
        @test abs(rm_after - rm_before) / rm_before < 1e-12
    end

    # Cumulative conservation across all windows
    m_total_final = sum(state.air_mass)
    rm_total_final = total_mass(state, :CO2)
    @test abs(m_total_final - m_total_0) / m_total_0 < 1e-11
    @test abs(rm_total_final - rm_total_0) / rm_total_0 < 1e-11
end

# =========================================================================
# Test 12: sweep functions are NOT exported (basis discipline)
# =========================================================================
@testset "Sweep functions not in public API" begin
    @test !isdefined(AtmosTransport, :sweep_x!)
    @test !isdefined(AtmosTransport, :sweep_y!)
    @test !isdefined(AtmosTransport, :sweep_z!)
    @test !isdefined(AtmosTransport, :max_cfl_x)
    @test !isdefined(AtmosTransport, :max_cfl_y)
    @test !isdefined(AtmosTransport, :max_cfl_z)
    @test !isdefined(AtmosTransport, :minmod)
    @test !isdefined(AtmosTransport, :van_leer_slope)

    # But strang_split! and AdvectionWorkspace ARE exported
    @test isdefined(AtmosTransport, :strang_split!)
    @test isdefined(AtmosTransport, :AdvectionWorkspace)
end

println("\n✓ All dry-flux interface tests passed!")
