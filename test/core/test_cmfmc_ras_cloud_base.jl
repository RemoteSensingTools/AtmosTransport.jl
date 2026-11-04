#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# CMFMCConvection (RAS) at the cloud base, checked against a literal Julia port
# of GEOS-Chem's `DO_RAS_CLOUD_CONVECTION` (convection_mod.F90 of GEOS-Chem
# 14.8.0, the transport part, one insoluble tracer).
#
# GEOS-Chem mixes the layers below the cloud base with the cloud-base inflow,
# lets the updraft entrain the cloud-base layer as it was, and gives that layer
# the sub-cloud exchange through its bottom edge in the same level update
# (T1 + T4). Before this test, AtmosTransport gave the cloud-base layer the
# exchange first and then let the updraft entrain the already exchanged air,
# which counts the inflow twice in that layer: conservative, but the layer goes
# negative. The literal port reproduces that order with
# `entrain_after_exchange = true`.
#
# The literal port keeps GEOS-Chem's bottom-up indexing (K = 1 is the surface
# layer, CMFMC(K) the flux through the upper edge of layer K), its cloud base
# CLDBASE, the in-place level loop K = CLDBASE … KTOP = NLAY − 1 and its
# `Q + DELQ < 0` clip (counted; the cases here never trigger it). The cases
# keep the model-top layer out of the updraft (GEOS-Chem never updates it).
# ---------------------------------------------------------------------------

using Test
using Random
using AtmosTransport
using AtmosTransport.Operators: CMFMCWorkspace, CMFMCConvection, apply_convection!
using AtmosTransport.Operators.Convection: CMFMCEdgeCloudBase, ArchivedCloudBase
using AtmosTransport.MetDrivers: ConvectionForcing
using AtmosTransport.Adjoints: _apply_cs_convection_adjoint!

"""
    ras_literal!(Q, CMFMC, DTRAIN, BMASS, CLDBASE, NDT, NS; tinynum, entrain_after_exchange, clip) -> n_clipped

Literal port of the transport part of GEOS-Chem's `DO_RAS_CLOUD_CONVECTION`
for one column, bottom-up (`K = 1` = surface). `Q` mixing ratio, `CMFMC(K)`
upward mass flux through the upper edge of layer `K` [kg/m²/s], `DTRAIN(K)`
[kg/m²/s], `BMASS(K)` air mass per area [kg/m²], `CLDBASE` the cloud-base
layer. `NS` internal steps of `SDT = NDT / NS`. With `entrain_after_exchange`
the updraft entrains the cloud-base layer after its sub-cloud exchange (the
order AtmosTransport used before); `clip = false` drops the `Q + DELQ < 0`
clip. Returns how often the clip fired.
"""
function ras_literal!(Q, CMFMC, DTRAIN, BMASS, CLDBASE, NDT, NS; tinynum = 1e-14,
                      entrain_after_exchange = false, clip = true)
    NLAY = length(Q)
    KTOP = NLAY - 1
    SDT = NDT / NS
    MB = sum(BMASS[1:CLDBASE-1]; init = 0.0)
    n_clipped = 0
    for ISTEP in 1:NS
        QC = Q[CLDBASE]
        if CLDBASE > 1 && CMFMC[CLDBASE - 1] > tinynum
            QB = sum(Q[K] * BMASS[K] for K in 1:CLDBASE-1) / sum(BMASS[1:CLDBASE-1])
            QC = (MB * QB + CMFMC[CLDBASE - 1] * Q[CLDBASE] * SDT) / (MB + CMFMC[CLDBASE - 1] * SDT)
            Q[1:CLDBASE-1] .= QC
        end
        for K in CLDBASE:KTOP
            CMFMC_BELOW = K > 1 ? CMFMC[K - 1] : 0.0
            if CMFMC_BELOW > tinynum
                CMOUT = CMFMC[K] + DTRAIN[K]
                ENTRN = CMOUT - CMFMC_BELOW
                QC_PRES = QC
                QENV = Q[K]
                if entrain_after_exchange && K == CLDBASE
                    QENV = Q[K] + SDT / BMASS[K] * CMFMC_BELOW * (QC_PRES - Q[K])
                end
                if ENTRN >= 0 && CMOUT > 0
                    QC = (CMFMC_BELOW * QC_PRES + ENTRN * QENV) / CMOUT
                end
                T1 = CMFMC_BELOW * QC_PRES
                T2 = -CMFMC[K] * QC
                T3 = CMFMC[K] * Q[K + 1]
                T4 = -CMFMC_BELOW * Q[K]
                DELQ = (SDT / BMASS[K]) * (T1 + T2 + T3 + T4)
                if clip && Q[K] + DELQ < 0
                    DELQ = -Q[K]; n_clipped += 1
                end
                Q[K] = Q[K] + DELQ
            else
                QC = Q[K]
                if CMFMC[K] > tinynum
                    DELQ = (SDT / BMASS[K]) * (-CMFMC[K] * QC + CMFMC[K] * Q[K + 1])
                    if clip && Q[K] + DELQ < 0
                        DELQ = -Q[K]; n_clipped += 1
                    end
                    Q[K] = Q[K] + DELQ
                end
            end
        end
    end
    return n_clipped
end

# The literal port on a top-down column (k = 1 = model top, our layout):
# `cmfmc[k]` is the flux through the top edge of layer k (Nz + 1 entries),
# `cb` the top-down cloud-base layer.
function ras_reference(q, cmfmc, dtrain, bmass, cb, dt, n_sub; kwargs...)
    Nz = length(q)
    Q = Float64.(reverse(q))
    CMFMC = Float64[cmfmc[Nz + 1 - K] for K in 1:Nz]
    n = ras_literal!(Q, CMFMC, Float64.(reverse(dtrain)), Float64.(reverse(bmass)),
                     Nz + 1 - cb, Float64(dt), n_sub; kwargs...)
    return reverse(Q), n
end

# --- Single-column cases (top-down, Nz = 8) --------------------------------
const NZ = 8
const BMASS = [50.0, 80.0, 120.0, 180.0, 250.0, 300.0, 350.0, 400.0]
const AREA = 1e12
# Lowest-layer tracer (the C3 initial condition) and a smooth profile.
const Q_SFC = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
const Q_PROF = [1.0, 1.0, 1.1, 1.3, 1.8, 2.5, 4.0, 9.0]

# Updraft from edge 7 (top of layer 7) up to edge 3, entraining at the cloud
# base without detraining there (as in GCHP's MERRA-2 columns). The CMFMC-edge
# cloud base is layer 6, the DQRCU-like archived base used below layer 5.
function elevated(scale)
    c = zeros(NZ + 1); d = zeros(NZ)
    c[3:7] .= [0.4, 0.8, 0.9, 0.7, 0.3] .* scale
    d[2] = 0.4scale; d[3] = 0.4scale; d[5] = 0.2scale
    return c, d
end
# Updraft from the surface layer up to layer 2 (no flux under the top layer,
# which GEOS-Chem never updates: KTOP = NLAY - 1). CMFMC-edge base: layer 7.
function deep(scale)
    c = zeros(NZ + 1); d = zeros(NZ)
    c[3:8] .= [0.5, 1.0, 1.0, 0.9, 0.8, 0.6] .* scale
    d[2] = 0.5scale; d[3] = 0.5scale
    return c, d
end
const DT = 240.0
const CASES = (("elevated, weak", elevated(0.002)),
               ("elevated, strong", elevated(0.05)),
               ("deep, strong", deep(0.05)))

# Top-down CMFMC-edge cloud base: lowest layer with inflow through its bottom edge.
edge_base(c) = findlast(k -> c[k + 1] > 0, 1:NZ)

# --- One column replicated over a grid of each topology --------------------
# Every column carries the same profile; returns the first column's mixing
# ratio, the CFL sub-step count and the raw tracer arrays.

function run_cs(::Type{FT}, q0, cmfmc, dtrain, dt; archived = nothing) where FT
    Nc, Hp = 2, 1
    N = Nc + 2Hp
    mesh = CubedSphereMesh(; Nc = Nc, Hp = Hp, FT = FT)
    vc = HybridSigmaPressure(zeros(FT, NZ + 1), collect(range(zero(FT), one(FT); length = NZ + 1)))
    grid = AtmosGrid(mesh, vc, CPU(); FT = FT)
    areas = ntuple(_ -> fill(FT(AREA), Nc, Nc), 6)
    air = ntuple(_ -> repeat(reshape(FT.(BMASS) .* FT(AREA), 1, 1, NZ), N, N, 1), 6)
    q = ntuple(_ -> repeat(reshape(FT.(q0) .* FT.(BMASS) .* FT(AREA), 1, 1, NZ, 1), N, N, 1, 1), 6)
    cm = ntuple(_ -> repeat(reshape(FT.(cmfmc), 1, 1, :), Nc, Nc, 1), 6)
    dtr = ntuple(_ -> repeat(reshape(FT.(dtrain), 1, 1, :), Nc, Nc, 1), 6)
    cb = archived === nothing ? nothing : ntuple(_ -> fill(FT(archived), Nc, Nc), 6)
    op = CMFMCConvection(; cloud_base = archived === nothing ? CMFMCEdgeCloudBase() : ArchivedCloudBase())
    ws = CMFMCWorkspace(q; cell_metrics = areas)
    apply_convection!(q, air, ConvectionForcing(cm, dtr, nothing, cb), op, FT(dt), ws, grid)
    cols = [q[p][i, j, :, 1] ./ air[p][i, j, :] for p in 1:6, i in (Hp + 1):(Hp + Nc), j in (Hp + 1):(Hp + Nc)]
    return cols[1], ws.cached_n_sub[], [sum(Float64, q[p][i, j, :, 1]) for p in 1:6, i in (Hp + 1):(Hp + Nc), j in (Hp + 1):(Hp + Nc)], cols
end

function run_ll(::Type{FT}, q0, cmfmc, dtrain, dt) where FT
    Nx, Ny = 3, 2
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vc = HybridSigmaPressure(zeros(FT, NZ + 1), collect(range(zero(FT), one(FT); length = NZ + 1)))
    grid = AtmosGrid(mesh, vc, CPU(); FT = FT)
    air = repeat(reshape(FT.(BMASS) .* FT(AREA), 1, 1, NZ), Nx, Ny, 1)
    q = repeat(reshape(FT.(q0) .* FT.(BMASS) .* FT(AREA), 1, 1, NZ, 1), Nx, Ny, 1, 1)
    cm = repeat(reshape(FT.(cmfmc), 1, 1, :), Nx, Ny, 1)
    dtr = repeat(reshape(FT.(dtrain), 1, 1, :), Nx, Ny, 1)
    ws = CMFMCWorkspace(air; cell_metrics = fill(FT(AREA), Ny))
    apply_convection!(q, air, ConvectionForcing(cm, dtr, nothing), CMFMCConvection(), FT(dt), ws, grid)
    cols = [q[i, j, :, 1] ./ air[i, j, :] for i in 1:Nx, j in 1:Ny]
    return cols[1], ws.cached_n_sub[], [sum(Float64, q[i, j, :, 1]) for i in 1:Nx, j in 1:Ny], cols
end

function run_rg(::Type{FT}, q0, cmfmc, dtrain, dt) where FT
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT = FT)
    vc = HybridSigmaPressure(zeros(FT, NZ + 1), collect(range(zero(FT), one(FT); length = NZ + 1)))
    grid = AtmosGrid(mesh, vc, CPU(); FT = FT)
    nc = ncells(mesh)
    air = repeat(reshape(FT.(BMASS) .* FT(AREA), 1, NZ), nc, 1)
    q = repeat(reshape(FT.(q0) .* FT.(BMASS) .* FT(AREA), 1, NZ, 1), nc, 1, 1)
    cm = repeat(reshape(FT.(cmfmc), 1, :), nc, 1)
    dtr = repeat(reshape(FT.(dtrain), 1, :), nc, 1)
    ws = CMFMCWorkspace(air; cell_metrics = fill(FT(AREA), nc))
    apply_convection!(q, air, ConvectionForcing(cm, dtr, nothing), CMFMCConvection(), FT(dt), ws, grid)
    cols = [q[c, :, 1] ./ air[c, :] for c in 1:nc]
    return cols[1], ws.cached_n_sub[], [sum(Float64, q[c, :, 1]) for c in 1:nc], cols
end

const TOPOLOGIES = (("cubed sphere", run_cs), ("lat-lon", run_ll), ("reduced Gaussian", run_rg))

@testset "CMFMCConvection RAS cloud base in GEOS-Chem's order" begin

@testset "cloud-base layer stays non-negative ($topo)" for (topo, run) in TOPOLOGIES
    # Lowest-layer tracer, entraining cloud base: the old order drove the
    # cloud-base layer (6) to -1.0e-4 here; GEOS-Chem's order keeps it at zero.
    c, d = elevated(0.05)
    q, n_sub, _, cols = run(Float64, Q_SFC, c, d, DT)
    old, _ = ras_reference(Q_SFC, c, d, BMASS, edge_base(c), DT, n_sub;
                           entrain_after_exchange = true, clip = false)
    @test minimum(old) < -1e-4                    # what the old order gave
    @test all(col -> minimum(col) >= 0, cols)
    @test abs(q[edge_base(c)]) < 1e-15            # only subsidence of tracer-free air
end

@testset "matches the literal GEOS-Chem port, Float64 ($topo)" for (topo, run) in TOPOLOGIES
    for (name, (c, d)) in CASES, q0 in (Q_SFC, Q_PROF)
        q, n_sub, _, _ = run(Float64, q0, c, d, DT)
        ref, n_clipped = ras_reference(q0, c, d, BMASS, edge_base(c), DT, n_sub)
        @test n_clipped == 0
        @test q ≈ ref rtol = 1e-13 atol = 1e-16
    end
    # A long step: several CFL sub-steps, GEOS-Chem's NS internal steps.
    c, d = elevated(0.05)
    q, n_sub, _, _ = run(Float64, Q_PROF, c, d, 1800.0)
    @test n_sub > 1
    ref, _ = ras_reference(Q_PROF, c, d, BMASS, edge_base(c), 1800.0, n_sub)
    @test q ≈ ref rtol = 1e-13
end

@testset "archived (DQRCU) cloud base matches the literal port (cubed sphere)" begin
    for (name, (c, d)) in CASES, q0 in (Q_SFC, Q_PROF)
        cb = edge_base(c) - 1                     # one layer above the lowest inflow edge
        q, n_sub, _, _ = run_cs(Float64, q0, c, d, DT; archived = cb)
        ref, n_clipped = ras_reference(q0, c, d, BMASS, cb, DT, n_sub)
        @test n_clipped == 0
        @test q ≈ ref rtol = 1e-13 atol = 1e-16
    end
    # Cloud base in the surface layer: no sub-cloud exchange.
    c, d = deep(0.05)
    q, n_sub, _, _ = run_cs(Float64, Q_PROF, c, d, DT; archived = NZ)
    ref, _ = ras_reference(Q_PROF, c, d, BMASS, NZ, DT, n_sub)
    @test q ≈ ref rtol = 1e-13
end

@testset "column mass conserved, uniform field unchanged ($topo)" for (topo, run) in TOPOLOGIES
    for (name, (c, d)) in CASES
        _, _, _, cols = run(Float64, fill(4.2e-4, NZ), c, d, DT)
        @test all(col -> all(x -> isapprox(x, 4.2e-4; rtol = 1e-13), col), cols)
        for q0 in (Q_SFC, Q_PROF)
            _, _, masses, cols = run(Float64, q0, c, d, DT)
            before = sum(q0 .* BMASS .* AREA)
            @test all(m -> isapprox(m, before; rtol = 1e-14), masses)
            @test all(col -> minimum(col) >= 0, cols)
        end
    end
end

@testset "Float32 ($topo)" for (topo, run) in TOPOLOGIES
    for (name, (c, d)) in CASES
        q, n_sub, masses, _ = run(Float32, Q_PROF, c, d, DT)
        ref, _ = ras_reference(Q_PROF, c, d, BMASS, edge_base(c), DT, n_sub)
        @test eltype(q) == Float32
        @test Float64.(q) ≈ ref rtol = 2e-5
        before = sum(Float64.(Float32.(Q_PROF) .* Float32.(BMASS) .* Float32(AREA)))
        @test all(m -> abs(m - before) / before < 1e-6, masses)
        # Non-negative to Float32 rounding of the unit surface value (the old
        # order gave -1e-4 here).
        q, _, _, cols = run(Float32, Q_SFC, c, d, DT)
        @test all(col -> minimum(col) >= -2eps(Float32), cols)
    end
end

# The adjoint kernel against the forward operator, column by column: the
# forward is linear in the tracer mass, so its matrix follows from unit
# vectors, and the adjoint applied to unit vectors must give its transpose.
function cs_column_matrices(::Type{FT}, cmfmc, dtrain, dt) where FT
    Nc, Hp = 2, 1
    N = Nc + 2Hp
    mesh = CubedSphereMesh(; Nc = Nc, Hp = Hp, FT = FT)
    vc = HybridSigmaPressure(zeros(FT, NZ + 1), collect(range(zero(FT), one(FT); length = NZ + 1)))
    grid = AtmosGrid(mesh, vc, CPU(); FT = FT)
    areas = ntuple(_ -> fill(FT(AREA), Nc, Nc), 6)
    air = ntuple(_ -> repeat(reshape(FT.(BMASS) .* FT(AREA), 1, 1, NZ), N, N, 1), 6)
    cm = ntuple(_ -> repeat(reshape(FT.(cmfmc), 1, 1, :), Nc, Nc, 1), 6)
    dtr = ntuple(_ -> repeat(reshape(FT.(dtrain), 1, 1, :), Nc, Nc, 1), 6)
    forcing = ConvectionForcing(cm, dtr, nothing)
    i0 = j0 = Hp + 1
    L = zeros(Float64, NZ, NZ)
    LT = zeros(Float64, NZ, NZ)
    n_sub = 0
    for k in 1:NZ
        q = ntuple(_ -> zeros(FT, N, N, NZ, 1), 6)
        q[1][i0, j0, k, 1] = one(FT)
        ws = CMFMCWorkspace(q; cell_metrics = areas)
        apply_convection!(q, air, forcing, CMFMCConvection(), FT(dt), ws, grid)
        n_sub = ws.cached_n_sub[]
        L[:, k] .= q[1][i0, j0, :, 1]
        λ = ntuple(_ -> zeros(FT, N, N, NZ), 6)
        λ[1][i0, j0, k] = one(FT)
        _apply_cs_convection_adjoint!(λ, air, forcing, CMFMCConvection(), FT(dt),
                                      CMFMCWorkspace(λ; cell_metrics = areas), mesh)
        LT[k, :] .= λ[1][i0, j0, :]
    end
    return L, LT, n_sub
end

@testset "adjoint is the transpose of the forward column ($FT)" for FT in (Float64, Float32)
    tol = FT === Float64 ? 1e-13 : 1e-5
    for (name, (c, d)) in CASES, dt in (DT, 1800.0)
        L, LT, n_sub = cs_column_matrices(FT, c, d, dt)
        dt == 1800.0 && !occursin("weak", name) && @test n_sub > 1
        @test maximum(abs.(LT .- L)) <= tol * maximum(abs.(L))
        # Columns of the forward matrix sum to one (tracer mass conserved).
        @test all(isapprox.(vec(sum(L; dims = 1)), 1; atol = 10tol))
    end
end

end  # CMFMCConvection RAS cloud base
