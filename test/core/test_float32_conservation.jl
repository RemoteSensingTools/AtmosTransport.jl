#!/usr/bin/env julia
# Float32 conservation contracts (docs/src/theory/float32_conservation.md):
# error-free and compensated sums, the column-ledger guard and its persistent
# carry, geometry evaluated in Float64 and rounded once, the decay decrement,
# exact global extents for regridding sources, and the emission profile
# deposit. The ledgers on production matrices are checked in
# test_tm5_hessenberg.jl and helpers/conservative_dkg.jl.

using Test, Random, LinearAlgebra
using AtmosTransport
using AtmosTransport.Architectures: _two_sum, _neumaier_add, compensated_total, _ledger_deposit
using AtmosTransport.Parameters: RN222_HALF_LIFE
using AtmosTransport.Grids: AtmosGrid, CubedSphereMesh, GEOSNativePanelConvention, LatLonMesh,
                            ReducedGaussianMesh, HybridSigmaPressure, ncells
using AtmosTransport.MetDrivers: ConvectionForcing
using AtmosTransport.Operators.SurfaceFlux: SurfaceFluxSource, SurfaceFluxOperator,
                                            ProfileDeposit, apply_surface_flux!
const Chem = AtmosTransport.Operators.Chemistry
const TM5 = AtmosTransport.Operators.Convection
const Diff = AtmosTransport.Operators.Diffusion
const RG = AtmosTransport.Regridding
include(joinpath(@__DIR__, "..", "helpers", "ledger_carry.jl"))
const ICIO = AtmosTransport.Models.InitialConditionIO

@testset "Error-free sums" begin
    rng = MersenneTwister(2026)
    draw() = Float32(randn(rng) * 10.0^rand(rng, -3:3))
    @test all(1:10_000) do _
        a, b = draw(), draw()
        s, e = _two_sum(a, b)
        big(s) + big(e) == big(a) + big(b)
    end

    # 0.01 is below half an ulp of 1e6 in Float32: a running sum drops all of
    # it, while the compensated sum stays within a fraction of that ulp.
    x = [1f6; fill(0.01f0, 10_000)]
    s, c = foldl((sc, v) -> _neumaier_add(sc..., v), x; init = (0f0, 0f0))
    exact = sum(big, x)
    @test abs(foldl(+, x) - exact) > 50
    @test abs(big(s) + big(c) - exact) < eps(1f6) / 8
end

@testset "Compensated Float64 totals" begin
    rng = MersenneTwister(11)
    panels = ntuple(_ -> 4f-4 .* (1 .+ rand(rng, Float32, 9, 9, 5)), 6)
    exact = sum(p -> sum(big, p), panels)
    @test compensated_total(panels[1]) isa Float64
    @test abs(compensated_total(panels) - exact) <= eps(Float64) * exact
    interior = view(panels[1], 2:8, 2:8, :)
    @test abs(compensated_total(interior) - sum(big, interior)) <= eps(Float64) * sum(big, interior)
end

@testset "Column ledger returns rounding, not leaks" begin
    n, rng = 40, MersenneTwister(7)
    for FT in (Float32, Float64), leak in (zero(FT), FT(1e-3))
        # I + B, where B exchanges mass between neighbours (columns sum to 1);
        # `leak` makes column n sum to 1 + leak.
        A = Matrix{FT}(I, n, n)
        for (k, a) in enumerate(FT(0.3) .* rand(rng, FT, n - 1))
            A[k, k] += a; A[k + 1, k] -= a; A[k + 1, k + 1] += a; A[k, k + 1] -= a
        end
        A[n, n] += leak
        pivots = zeros(Int, n)
        TM5._tm5_lu!(A, pivots, n)
        q = FT(4) .* (1 .+ rand(rng, FT, n, 1))
        q0 = copy(q)
        carry = TM5._tm5_conserving_solve_tracer!(q, A, pivots, n, 1, 1, false)
        gap = sum(big, q) - sum(big, q0)
        if iszero(leak)
            @test abs(gap) <= eps(FT) * maximum(q)
        else
            @test abs(gap) > 1e-5 * sum(big, q0)     # the leak stays visible
        end
        @test abs(carry) <= eps(maximum(q)) / 2      # the carry holds rounding only
    end
end

@testset "Ledger carry keeps what the largest cell cannot hold" begin
    # 0.01 is below half an ulp of 1e6 in Float32 (0.03125): added to the cell
    # alone it vanishes every time. The carry keeps it until whole ulps can be
    # added, so the cell plus the carry tracks the exact sum.
    value, carry, plain = 1f6, 0f0, 1f6
    for k in 1:100
        value, carry = _ledger_deposit(value, 0.01f0, carry, 40)
        plain += 0.01f0
        @test abs(big(value) + big(carry) - (big(1f6) + k * big(0.01f0))) <= k * eps(0.01f0)
        @test abs(carry) <= eps(value)                # at most half an ulp plus the deposit
    end
    @test plain == 1f6
    @test value == 1f6 + 1                            # 100 × 0.01 = 16 ulps of 1e6

    # With a zero carry the cell receives exactly the rounded sum, as without a carry.
    rng = MersenneTwister(5)
    @test all(1:1000) do _
        v, r = 1f3 * randn(rng, Float32), randn(rng, Float32)
        first(_ledger_deposit(v, r, 0f0, 10)) === v + r
    end
    # A carry above the residual bound (16n ulps of the cell) is stale and dropped.
    @test _ledger_deposit(3f0, 0f0, 1f0, 1) == (3f0, 0f0)
    @test _ledger_deposit(0f0, 0f0, 1f-30, 4) == (0f0, 0f0)
    value, carry = _ledger_deposit(1.0, 1e-17, 1e-17, 2)
    @test value == 1.0 && carry == 2e-17
end

# Repeated Float32 solves of surface-peaked columns. Without the carry (as in
# v0.5.0: a zero carry at every solve), the part of each residual below the
# largest cell's spacing is lost with one sign and the column totals drift
# (7.7e-9 for TM5 and 1.9e-9 for dkg here). With it, the field plus the carry
# is conserved to rounding of the carry, and the field alone differs from the
# start only by the outstanding carry, about half an ulp of each column's
# largest cell. Each cell stays as accurate against Float64 as in the
# carry-free run (measured: 7.54e-6, 3.37e-6 and 1.8 eps in both).
@testset "Repeated solves do not drift (persistent ledger carry)" begin
    rng = MersenneTwister(2026)
    n, ncol, solves = 40, 64, 200
    A = Matrix{Float32}(I, n, n)
    for (k, a) in enumerate(Float32(3) .* rand(rng, Float32, n - 1))
        A[k, k] += a; A[k + 1, k] -= a; A[k + 1, k + 1] += a; A[k, k + 1] -= a
    end
    q0 = Float32[1f9 * exp(-(n - k) / 4f0) * (1 + rand(rng, Float32)) for k in 1:n, _ in 1:ncol]
    function convect(FT; keep_carry = true)
        B, q, pivots, carry = FT.(A), FT.(q0), zeros(Int, n), zeros(FT, ncol)
        TM5._tm5_lu!(B, pivots, n)
        for _ in 1:solves, c in 1:ncol
            c′ = TM5._tm5_conserving_solve_tracer!(q, B, pivots, n, 1, c, false, carry[c])
            keep_carry && (carry[c] = c′)
        end
        return q, carry
    end
    (q32, carry), (plain32, _), (q64, _) = convect(Float32), convect(Float32; keep_carry = false),
                                           convect(Float64)
    total0 = sum(big, q0)
    drift(q) = abs(Float64((sum(big, q) - total0) / total0))
    cell_error(q) = maximum(abs.(q .- q64) ./ q64)
    bound = 2e-10           # the carry-free run fails it (its size varies with the Julia version)
    @test drift(plain32) > bound
    @test drift(q32) < bound
    @test abs(Float64((sum(big, q32) + sum(big, carry) - total0) / total0)) < 1e-14
    @test cell_error(q32) <= 1.1 * cell_error(plain32)

    Nc, Nz, Hp = 6, 40, 1
    N = Nc + 2
    m = ntuple(_ -> [1f9 * (1 + 40 * Float32(k / Nz)^2) * (1 + 0.1f0 * rand(rng, Float32))
                     for i in 1:N, j in 1:N, k in 1:Nz], 6)
    d = ntuple(p -> [10 < k < Nz ? (k > Nz - 15 ? 3f0 : 0.02f0) *
                     min(m[p][i+Hp,j+Hp,k], m[p][i+Hp,j+Hp,k+1]) * (0.5f0 + rand(rng, Float32)) : 0f0
                     for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6)
    # Tracer 1 is peaked at the surface, tracer 2 is a uniform mixing ratio.
    rm0 = ntuple(p -> [m[p][i,j,k] * (t == 1 ? 1f-5 * exp(-(Nz - k) / 4f0) * (1 + rand(rng, Float32)) : 4f-4)
                       for i in 1:N, j in 1:N, k in 1:Nz, t in 1:2], 6)
    function diffuse(FT; keep_carry = true)
        mass, rm = map(a -> FT.(a), m), map(a -> FT.(a), rm0)
        field = AtmosTransport.State.PrecomputedCSDkgField(map(a -> FT.(a), d))
        op = Diff.ImplicitVerticalDiffusion(; kz_field = field)
        ws = Diff.DiffusionWorkspace(mass, Hp, 2)
        for _ in 1:solves
            keep_carry || foreach(c -> fill!(c, zero(FT)), ws.ledger_carry)
            Diff.apply_vertical_diffusion_vmr!(rm, mass, op, ws, one(FT); halo_width = Hp)
        end
        return map(a -> a[1+Hp:Nc+Hp, 1+Hp:Nc+Hp, :, :], rm), ws.ledger_carry
    end
    (r32, carry), (plain32, _), (r64, _) = diffuse(Float32), diffuse(Float32; keep_carry = false),
                                           diffuse(Float64)
    interior(a) = a[1+Hp:Nc+Hp, 1+Hp:Nc+Hp, :, :]
    total(x, t) = sum(p -> sum(big, @view x[p][:, :, :, t]), 1:6)
    dkg_total0 = total(map(interior, rm0), 1)
    held = sum(p -> sum(big, @view carry[p][:, :, 1]), 1:6)
    dkg_drift(r) = abs(Float64((total(r, 1) - dkg_total0) / dkg_total0))
    dkg_cell_error(r) = maximum(p -> maximum(abs.(r[p][:,:,:,1] .- r64[p][:,:,:,1]) ./ r64[p][:,:,:,1]), 1:6)
    uniform_error(r) = maximum(p -> maximum(abs, r[p][:,:,:,2] ./ m[p][1+Hp:Nc+Hp, 1+Hp:Nc+Hp, :] .- 4f-4), 1:6)
    @test dkg_drift(plain32) > bound                      # 1.9e-9 on Julia 1.12, 4.9e-10 on 1.10
    @test dkg_drift(r32) < bound
    @test abs(Float64((total(r32, 1) + held - dkg_total0) / dkg_total0)) < 1e-14
    @test dkg_cell_error(r32) <= 1.1 * dkg_cell_error(plain32)
    @test uniform_error(r32) <= 1.1 * uniform_error(plain32)
end

# The production operators keep the carry in their workspaces: after many
# calls, each column's tracer mass plus its carry still holds the initial mass
# (helpers/ledger_carry.jl, also run on CUDA and Metal).
@testset "Operator workspaces keep the ledger carry" begin
    rng = MersenneTwister(77)
    FT, Nt = Float32, 2
    check_tm5_ledger_carry(Array)       # CPU: per-thread kernels on every topology
    check_dkg_ledger_carry(Array)       # packed and one-tracer (`tracer = t`) calls agree

    # Without `tracer` a one-tracer Dkg call keeps no carry (adjoint replays);
    # a slot outside the workspace is an error.
    Nc, Nz, Hp = 3, 10, 1
    N = Nc + 2Hp
    mass = ntuple(_ -> [1f9 * (1 + 40 * Float32(k / Nz)^2) * (1 + 0.1f0 * rand(rng, FT))
                        for i in 1:N, j in 1:N, k in 1:Nz], 6)
    dkg = ntuple(p -> [k < Nz ? 0.5f0 * min(mass[p][i+Hp, j+Hp, k], mass[p][i+Hp, j+Hp, k+1]) *
                       rand(rng, FT) : 0f0 for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6)
    rm = ntuple(p -> [mass[p][i, j, k] * 1f-3 * exp(-(Nz - k) / 3f0) for i in 1:N, j in 1:N, k in 1:Nz], 6)
    op = Diff.ImplicitVerticalDiffusion(; kz_field = AtmosTransport.State.PrecomputedCSDkgField(dkg))
    ws = Diff.DiffusionWorkspace(mass, Hp, Nt)
    Diff.apply_vertical_diffusion_vmr!(rm, mass, op, ws, 1f0; halo_width = Hp, tracer = 1)
    kept = map(copy, ws.ledger_carry)
    @test any(c -> any(!iszero, c), kept)
    Diff.apply_vertical_diffusion_vmr!(rm, mass, op, ws, 1f0; halo_width = Hp)
    @test ws.ledger_carry == kept
    @test_throws ArgumentError Diff.apply_vertical_diffusion_vmr!(rm, mass, op, ws, 1f0;
                                                                  halo_width = Hp, tracer = Nt + 1)

    # GEOS-Chem non-local VDIFF solves through the same Dkg path: with the same
    # exchange it keeps the same field and the same carry.
    panels() = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    gchp = AtmosTransport.State.GCHPNonlocalPBLField(
        AtmosTransport.State.PrecomputedCSDkgField(dkg), panels(), panels(), panels(),
        AtmosTransport.State.GCHPVdiffParameters{FT}(), zeros(FT, Nc, Nc))
    op_gchp = Diff.ImplicitVerticalDiffusion(; kz_field = gchp)
    packed = ntuple(p -> cat(rm[p], 2 .* rm[p]; dims = 4), 6)
    a, b = map(copy, packed), map(copy, packed)
    ws_a, ws_b = Diff.DiffusionWorkspace(mass, Hp, Nt), Diff.DiffusionWorkspace(mass, Hp, Nt)
    for _ in 1:10
        Diff.apply_vertical_diffusion_vmr!(a, mass, op, ws_a, 1f0; halo_width = Hp)
        Diff.apply_vertical_diffusion_vmr!(b, mass, op_gchp, ws_b, 1f0; halo_width = Hp)
    end
    @test a == b && ws_a.ledger_carry == ws_b.ledger_carry
    @test any(c -> any(!iszero, c), ws_b.ledger_carry)

    # The per-tracer Lin–Rood transport step diffuses each tracer with its own carry.
    A = collect(FT, range(0, 5f4; length = Nz + 1)); B = collect(FT, range(1, 0; length = Nz + 1))
    A[1] = 0; B[end] = 0; B[1] = 0; A[end] = 0
    mesh = CubedSphereMesh(; Nc, Hp = 3, FT)
    grid = AtmosGrid(mesh, HybridSigmaPressure(A, B), AtmosTransport.CPU(); FT)
    N = Nc + 2mesh.Hp
    air = ntuple(_ -> [1f9 * (1 + 40 * Float32(k / Nz)^2) for i in 1:N, j in 1:N, k in 1:Nz], 6)
    state = CubedSphereState(DryBasis, mesh, air;
                             A = ntuple(p -> air[p] .* [1f-3 * exp(-(Nz - k) / 3f0) for i in 1:N, j in 1:N, k in 1:Nz], 6),
                             B = ntuple(p -> air[p] .* 1f-4 .* (1 .+ rand(rng, FT, N, N, Nz)), 6))
    model = TransportModel(state, allocate_face_fluxes(mesh, Nz; FT, basis = DryBasis), grid,
                           LinRoodPPMScheme(5); diffusion = op)
    for _ in 1:5
        step!(model, 900f0)
    end
    carry = model.workspace.diffusion_ws.ledger_carry
    @test all(t -> any(p -> any(!iszero, carry[p][:, :, t]), 1:6), 1:Nt)
    @test carry[1][:, :, 1] != carry[1][:, :, 2]
end

@testset "Cubed-sphere geometry is Float64, rounded once" begin
    m32, m64 = (CubedSphereMesh(; FT, Nc = 48, convention = GEOSNativePanelConvention())
                for FT in (Float32, Float64))
    @test m32.cell_areas == Float32.(m64.cell_areas)
    @test m32.Δx == Float32.(m64.Δx) && m32.Δy == Float32.(m64.Δy)
    @test RG.cubed_sphere_face_corners(m32) == RG.cubed_sphere_face_corners(m64)
    @test RG.GOCore.best_manifold(m32) == RG.GOCore.best_manifold(m64)
end

@testset "Decay decrement" begin
    λ = Float32(log(2) / RN222_HALF_LIFE)
    for dt in (400.0, 450.0, 900.0)
        d = Chem.decay_decrement(Float32, λ, dt)
        @test d isa Float32
        @test d == Float32(expm1(-Float64(λ) * dt))
    end
end

@testset "Global source grids snap to exact extents" begin
    # GridFED stores its 0.1° centres in Float32.
    lon = Float64.(Float32.(range(-179.95, 179.95; length = 3600)))
    lat = Float64.(Float32.(range(-89.95, 89.95; length = 1800)))
    GIO = AtmosTransport.GriddedIO
    R = AtmosTransport.Parameters.EARTH_RADIUS
    mesh = GIO.source_latlon_mesh(lon, lat; radius = R)
    @test mesh isa LatLonMesh{Float64}
    @test (first(mesh.φᶠ), last(mesh.φᶠ)) == (-90.0, 90.0)
    @test last(mesh.λᶠ) - first(mesh.λᶠ) ≈ 360 atol = 1e-9
    @test RG._latlon_full_sphere(mesh)

    regional = GIO.source_latlon_mesh(collect(0.5:1.0:29.5), collect(-9.5:1.0:9.5);
                                      radius = R)
    @test (first(regional.φᶠ), last(regional.φᶠ)) == (-10.0, 10.0)
    @test !RG._latlon_full_sphere(regional)
end

@testset "Profile deposit conserves column mass in Float32" begin
    FT, Nc, Nz, Hp = Float32, 2, 40, 1
    # 0.45 kg profile shares into 2e6 kg cells (400 ppm of 5e9 kg): ~4 ulps each.
    q = ntuple(_ -> fill(FT(400e-6 * 5e9), Nc + 2Hp, Nc + 2Hp, Nz), 6)
    q0 = map(copy, q)
    src = SurfaceFluxSource(:x, ntuple(_ -> fill(FT(0.01), Nc, Nc), 6))   # kg s⁻¹
    op = SurfaceFluxOperator(src)
    profile = zeros(FT, Nc, Nc, Nz)
    profile[:, :, Nz-4:Nz] .= reshape(FT[0.1, 0.15, 0.2, 0.25, 0.3], 1, 1, 5)
    deposit = ntuple(_ -> ProfileDeposit(profile), 6)
    dt, steps = 450.0, 240
    for _ in 1:steps
        apply_surface_flux!(q, op, nothing, dt, nothing, nothing; tracer_names = (:x,),
                            halo_width = Hp, deposit)
    end
    # Kahan: mass added = steps·x + final compensation (mass deposited ahead).
    emitted = steps * 6Nc^2 * big(FT(0.01) * FT(dt))
    added = sum(p -> sum(big, p), q) - sum(p -> sum(big, p), q0)
    ahead = sum(p -> sum(big, p), src.compensation)
    @test abs(added - (emitted + ahead)) < 1e-7 * emitted
end

# CMFMC convection: the flux-form update conserves each column by construction
# (interface fluxes telescope), but each layer's new mass is written once as
# `q·m`. Without a ledger, sub-ulp changes vanish with one sign: over these 200
# updates a surface-peaked tracer drifted by -1.9e-7, its cells by up to 1.7e-5
# against Float64, and a uniform mixing ratio by 76 eps. With the ledger and
# its carry the field plus the carry is exact, the field alone is off by the
# outstanding carries (6.6e-10), and the cells stay within 1.3e-6 and 4.3 eps.
@testset "CMFMC column ledger and carry" begin
    rng = MersenneTwister(2026)
    Nx, Ny, Nz, calls = 8, 8, 40, 200
    function cmfmc_grid(FT)
        A = collect(FT, range(0, 5f4; length = Nz + 1)); B = collect(FT, range(1, 0; length = Nz + 1))
        A[1] = 0; B[end] = 0; B[1] = 0; A[end] = 0
        return AtmosGrid(LatLonMesh(; FT, Nx, Ny), HybridSigmaPressure(A, B), AtmosTransport.CPU(); FT)
    end
    m0 = [1f16 * (1 + 40 * Float32(k / Nz)^2) / 41 * (1 + 0.1f0 * rand(rng, Float32))
          for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    q0 = [m0[i, j, k] * (t == 1 ? 1f-5 * exp(-(Nz - k) / 4f0) * (1 + rand(rng, Float32)) : 4f-4)
          for i in 1:Nx, j in 1:Ny, k in 1:Nz, t in 1:2]
    cmfmc = zeros(Float32, Nx, Ny, Nz + 1)            # zero at the model top
    for i in 1:Nx, j in 1:Ny, k in 12:36
        cmfmc[i, j, k] = 0.01f0 * (0.5f0 + rand(rng, Float32))
    end
    function convect(FT, cm; steps = calls)
        state = CellState(FT.(m0); A = FT.(q0[:, :, :, 1]), B = FT.(q0[:, :, :, 2]))
        ws = TM5.CMFMCWorkspace(state.air_mass)
        forcing = ConvectionForcing(FT.(cm), nothing, nothing)
        for _ in 1:steps
            TM5.apply_convection!(state.tracers_raw, state.air_mass, forcing, TM5.CMFMCConvection(),
                                  FT(900), ws, cmfmc_grid(FT))
        end
        return state.tracers_raw, ws.ledger_carry[]
    end
    (r32, carry), (r64, _) = convect(Float32, cmfmc), convect(Float64, cmfmc)
    total0 = sum(big, q0[:, :, :, 1])
    @test size(carry) == (Nx, Ny, 2) && any(!iszero, carry)
    @test abs(Float64((sum(big, r32[:, :, :, 1]) - total0) / total0)) < 2e-9
    @test abs(Float64((sum(big, r32[:, :, :, 1]) + sum(big, carry[:, :, 1]) - total0) / total0)) < 1e-13
    @test maximum(abs.(r32[:, :, :, 1] .- r64[:, :, :, 1]) ./ r64[:, :, :, 1]) <= 2e-6
    @test maximum(abs, r32[:, :, :, 2] ./ m0 .- 4f-4) <= 8 * eps(Float32) * 4f-4

    # A flux through the model top leaks by construction: not ledgered, so the
    # loss stays visible and no carry is kept.
    leaking = copy(cmfmc); leaking[:, :, 1] .= 0.005f0
    r, carry = convect(Float32, leaking; steps = 1)
    @test all(iszero, carry)
    @test sum(big, r[:, :, :, 2]) < sum(big, q0[:, :, :, 2]) * (1 - 1e-5)

    # Every topology keeps the carry in the workspace (helpers/ledger_carry.jl,
    # also run on CUDA and Metal); the clamped cubed-sphere variant rescales its
    # columns instead and is not ledgered.
    check_cmfmc_ledger_carry(Array)
end
