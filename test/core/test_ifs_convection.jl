#!/usr/bin/env julia
# IFSConvection: the tracer transport of the IFS convection scheme (CUCTRACER +
# CUBIDIAG) on the TM5 convection fields. The production column code is checked
# against the compiled OpenIFS routine (fixture) and the plain-loop reference
# (`test/helpers/ifs_convection_reference.jl`), then the operator through the
# three topologies, TransportModel, DrivenSimulation, configuration and the
# adjoint guard. GPU sections run where CUDA is functional.

using Test
using Random

using AtmosTransport
using .AtmosTransport: Operators, Models
using .AtmosTransport.Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
                             HybridSigmaPressure, ncells
using .AtmosTransport.State: CellState, CubedSphereState, DryBasis, allocate_face_fluxes
using .AtmosTransport.Operators: IFSConvection, IFSWorkspace, IFSCuctracer, IFSMonotoneUpstream,
                                 UpwindScheme
using .AtmosTransport.Operators.Convection: apply_convection!, convection_workspace_for
using .AtmosTransport.MetDrivers: ConvectionForcing

const IFSC = AtmosTransport.Operators.Convection.IFS
const Physics = AtmosTransport.Models.Physics
const Runner = AtmosTransport.Models.DrivenRunner
const RunnerCore = AtmosTransport.Models.Runner

include(joinpath(@__DIR__, "..", "helpers", "ifs_convection_reference.jl"))
include(joinpath(@__DIR__, "..", "helpers", "convection_runtime.jl"))

const VARIANTS = ((IFSCuctracer(), false), (IFSMonotoneUpstream(), true))

# --- Column helpers ---------------------------------------------------------

function read_fixture(path)
    lines = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(path))
    pos = Ref(1)
    next() = (l = lines[pos[]]; pos[] += 1; l)
    numbers() = parse.(Float64, split(next()))
    columns = []
    for _ in 1:parse(Int, next())
        header = split(next())
        N, nt, dt = parse(Int, header[1]), parse(Int, header[2]), parse(Float64, header[3])
        m, entu, detu, entd, detd = (numbers() for _ in 1:5)
        before = [numbers() for _ in 1:nt]
        after = [numbers() for _ in 1:nt]
        @assert all(length.((m, entu, detu, entd, detd)) .== N)
        push!(columns, (; N, dt, m, entu, detu, entd, detd, before, after))
    end
    return columns
end

# Interface fluxes, active interfaces and Courant number of one column.
function column_setup(m, entu, detu, entd, detd, dt; area = one(eltype(m)))
    N = length(m)
    mu, md, cu, cd = (zeros(eltype(m), N) for _ in 1:4)
    courant = IFSC._ifs_column_fluxes!(mu, md, m, entu, detu, entd, detd, area, dt)
    lo = IFSC._ifs_first_active_interface(mu, md)
    lfs = lo == 0 ? N + 1 : IFSC._ifs_downdraft_top(md, lo)
    return (; mu, md, cu, cd, courant, lo, lfs)
end

# One production sub-step of mixing ratio `c` (no ledger); returns the new mixing ratio.
function production_step(c, m, entu, detu, entd, detd, dt, variant; area = one(eltype(m)))
    s = column_setup(m, entu, detu, entd, detd, dt; area)
    q = reshape(c .* m, :, 1)
    s.lo == 0 || IFSC._ifs_substep!(q, 1, m, detu, detd, s.mu, s.md, s.cu, s.cd, area, dt,
                                    s.lo, s.lfs, variant)
    return q[:, 1] ./ m, q[:, 1]
end

# Random columns, including ones that break the plume assumptions.
function random_column(rng, N, kind)
    m = 40 .+ 400 .* rand(rng, N)
    entu, detu, entd, detd = (zeros(N) for _ in 1:4)
    if kind === :inconsistent
        for f in (entu, detu, entd, detd), k in 1:N
            rand(rng) < 0.6 && (f[k] = 0.05 * rand(rng))
        end
        return (; m, entu, detu, entd, detd)
    end
    # Updraft: flux at interfaces kt+1..kb (top of layer kb), zero at kt.
    kt = rand(rng, 2:max(2, min(N ÷ 2, N - 5)))
    kb = kind === :elevated_base ? rand(rng, (kt + 2):(N - 3)) : N
    Mu = zeros(N + 1)
    for i in (kt + 1):kb
        Mu[i] = 0.02 + 0.08 * rand(rng)
    end
    for k in kt:kb
        diff = Mu[k] - Mu[k + 1]
        entu[k] = max(diff, 0.0) + 0.02 * rand(rng)
        detu[k] = entu[k] - diff
    end
    # Downdraft: flux at interfaces ld+1..N, zero at the ground.
    ld = kind === :downdraft_above ? rand(rng, 1:max(1, kt - 1)) : rand(rng, kt:(N - 2))
    Dn = zeros(N + 1)
    for i in (ld + 1):N
        Dn[i] = 0.01 + 0.03 * rand(rng)
    end
    for k in ld:N
        diff = Dn[k + 1] - Dn[k]
        entd[k] = max(diff, 0.0) + 0.01 * rand(rng)
        detd[k] = entd[k] - diff
    end
    if kind === :excess_detrainment
        # Detrainment above the entering flux, entrainment raised to match.
        for k in kt:kb, (e, d, M) in ((entu, detu, Mu[k + 1]), (entd, detd, Dn[k]))
            if rand(rng) < 0.5
                extra = M + 0.05 * rand(rng)
                e[k] += extra
                d[k] += extra
            end
        end
    end
    return (; m, entu, detu, entd, detd)
end

const KINDS = (:consistent, :excess_detrainment, :inconsistent, :downdraft_above, :elevated_base)

random_profile(rng, N) = rand(rng) < 0.5 ? rand(rng, N) :
                         (c = zeros(N); c[rand(rng, 1:N)] = 1.0; c)

# --- Fixture and reference -----------------------------------------------------

@testset "IFSConvection column: compiled OpenIFS fixture" begin
    columns = read_fixture(joinpath(@__DIR__, "..", "fixtures", "ifs_cuctracer_columns.txt"))
    @test length(columns) == 12
    worst_step = 0.0
    worst_column = 0.0
    for col in columns
        s = column_setup(col.m, col.entu, col.detu, col.entd, col.detd, col.dt)
        for (c0, c1) in zip(col.before, col.after)
            scale = maximum(abs, c1)
            c, _ = production_step(c0, col.m, col.entu, col.detu, col.entd, col.detd,
                                   col.dt, IFSCuctracer())
            worst_step = max(worst_step, maximum(abs, c .- c1) / scale)
            # The whole column routine, with its ledger, where one step suffices.
            if s.courant <= 1
                q = reshape(c0 .* col.m, :, 1)
                IFSC._ifs_column!(q, nothing, col.m, col.detu, col.detd, s.mu, s.md, s.cu, s.cd,
                                  1.0, col.dt, s.courant, 1.0, IFSCuctracer())
                worst_column = max(worst_column, maximum(abs, q[:, 1] ./ col.m .- c1) / scale)
            end
        end
    end
    @info "IFS fixture: largest relative deviation" worst_step worst_column
    @test worst_step < 1e-12
    @test worst_column < 1e-12
end

@testset "IFSConvection column: plain-loop reference, both variants" begin
    rng = Xoshiro(20261111)
    worst = Dict{Bool, Float64}(false => 0.0, true => 0.0)
    for trial in 1:400, (variant, monotone) in VARIANTS
        N = rand(rng, 8:40)
        kind = KINDS[mod1(trial, length(KINDS))]
        col = random_column(rng, N, kind)
        dt = rand(rng, (300.0, 900.0, 1800.0, 3600.0))
        c0 = random_profile(rng, N)
        ref = ifs_convection_step(c0, col.entu, col.detu, col.entd, col.detd, col.m, dt; monotone)
        got, _ = production_step(c0, col.m, col.entu, col.detu, col.entd, col.detd, dt, variant)
        # Relative to the larger of input and result: in one step at Courant numbers
        # above one the cuctracer arithmetic can amplify the input by orders of
        # magnitude where the downdraft exceeds the updraft.
        scale = max(maximum(abs, c0), maximum(abs, ref))
        worst[monotone] = max(worst[monotone], maximum(abs, got .- ref) / scale)
    end
    @info "IFS reference, one step: largest relative deviation" cuctracer = worst[false] monotone = worst[true]
    @test worst[false] < 1e-12
    @test worst[true] < 1e-12
end

@testset "IFSConvection column: sub-steps equal repeated reference steps" begin
    rng = Xoshiro(7)
    worst = 0.0
    substeps = Int[]
    for trial in 1:200, (variant, monotone) in VARIANTS
        N = rand(rng, 8:30)
        col = random_column(rng, N, KINDS[mod1(trial, length(KINDS))])
        dt = rand(rng, (900.0, 3600.0, 10800.0))
        cfl = rand(rng, (0.8, 0.5, 1.0))
        c0 = random_profile(rng, N)
        n = max(1, ceil(Int, ifs_courant(col.entu, col.detu, col.entd, col.detd, col.m, dt) / cfl))
        ref = copy(c0)
        for _ in 1:n
            ref = ifs_convection_step(ref, col.entu, col.detu, col.entd, col.detd, col.m, dt / n; monotone)
        end
        s = column_setup(col.m, col.entu, col.detu, col.entd, col.detd, dt)
        @test IFSC._ifs_substeps(s.courant, cfl) == n
        push!(substeps, n)
        q = reshape(c0 .* col.m, :, 1)
        IFSC._ifs_column!(q, nothing, col.m, col.detu, col.detd, s.mu, s.md, s.cu, s.cd,
                          1.0, dt, s.courant, cfl, variant)
        # Where the downdraft exceeds the updraft, the cuctracer arithmetic takes the
        # compensating flow from downstream and grows by about
        # 1 / (1 - dt_sub (Md - Mu) / m) per sub-step; compare relative to the
        # larger of input and result.
        scale = max(maximum(abs, c0), maximum(abs, ref))
        worst = max(worst, maximum(abs, q[:, 1] ./ col.m .- ref) / scale)
    end
    @info "IFS reference with sub-steps" worst max_substeps = maximum(substeps)
    @test maximum(substeps) > 4
    @test worst < 1e-12
end

# --- Properties ----------------------------------------------------------------

# `n` production sub-steps of length dt / n without the ledger; returns tracer mass.
function production_steps(c, m, f, dt, n, variant; area)
    s = column_setup(m, f.entu, f.detu, f.entd, f.detd, dt; area)
    q = reshape(c .* m, :, 1)
    for _ in 1:n
        IFSC._ifs_substep!(q, 1, m, f.detu, f.detd, s.mu, s.md, s.cu, s.cd, area, dt / n,
                           s.lo, s.lfs, variant)
    end
    return q[:, 1]
end

@testset "IFSConvection column: conservation, uniform field, extrema" begin
    # One call: sub-steps at Courant number <= 1, as the operator takes them; no ledger.
    rng = Xoshiro(11)
    worst_mass = Dict(false => 0.0, true => 0.0)
    worst_uniform = Dict(false => 0.0, true => 0.0)
    worst_extrema = 0.0
    for trial in 1:300, (variant, monotone) in VARIANTS
        N = rand(rng, 8:40)
        kind = KINDS[mod1(trial, length(KINDS))]
        col = random_column(rng, N, kind)
        dt = rand(rng, (900.0, 3600.0))
        area = 1e9 * (1 + rand(rng))
        m = col.m .* area                       # kg per cell
        s = column_setup(m, col.entu, col.detu, col.entd, col.detd, dt; area)
        s.lo == 0 && continue
        n = IFSC._ifs_substeps(s.courant, 1.0)
        c0 = random_profile(rng, N)
        q = production_steps(c0, m, col, dt, n, variant; area)
        # Relative to the larger column: the cuctracer arithmetic can grow (see below).
        scale = max(sum(abs, c0 .* m), sum(abs, q))
        worst_mass[monotone] = max(worst_mass[monotone], abs(sum(q) - sum(c0 .* m)) / scale)
        # A uniform field is a fixed point of both variants. The cuctracer arithmetic
        # amplifies rounding wherever the downdraft exceeds the updraft at an
        # interface; on the independent random fields of `:inconsistent` columns
        # this is visible within one call, so it is checked on the other kinds.
        if monotone || kind !== :inconsistent
            q_u = production_steps(fill(0.37, N), m, col, dt, n, variant; area)
            worst_uniform[monotone] = max(worst_uniform[monotone],
                                          maximum(abs, q_u ./ m .- 0.37) / 0.37)
        end
        if monotone                         # no new extrema
            c = q ./ m
            lo, hi = extrema(c0)
            worst_extrema = max(worst_extrema, max(lo - minimum(c), maximum(c) - hi, 0.0) / hi)
        end
    end
    @info "IFS column properties (Float64, one call, no ledger)" worst_mass worst_uniform worst_extrema
    @test worst_mass[false] < 1e-13 && worst_mass[true] < 1e-13
    @test worst_uniform[false] < 1e-12 && worst_uniform[true] < 1e-12
    @test worst_extrema < 1e-14
end

@testset "IFSConvection: many calls per column kind (Float64)" begin
    # Frozen forcing, 200 calls with the operator's sub-steps (cfl_limit 0.8).
    # The monotone variant stays within the initial bounds and conserves mass for
    # every kind. The cuctracer arithmetic amplifies wherever the downdraft
    # exceeds the updraft at an interface, by about 1 / (1 - dt_sub (Md - Mu) / m)
    # per sub-step; the plume-consistent kinds `:elevated_base` (downdraft below
    # cloud base) and `:downdraft_above` have such interfaces too. For it only
    # what holds is asserted: conservation and agreement with the transcription.
    ncall, ncol, dt = 200, 30, 900.0
    rng = Xoshiro(2026)
    for kind in KINDS
        worst_bounds = 0.0; worst_mass_mono = 0.0
        worst_ref = 0.0; worst_mass_cuc = 0.0; grown = 0
        for _ in 1:ncol
            N = rand(rng, 12:30)
            col = random_column(rng, N, kind)
            c0 = random_profile(rng, N)
            s = column_setup(col.m, col.entu, col.detu, col.entd, col.detd, dt)
            s.lo == 0 && continue
            n = IFSC._ifs_substeps(s.courant, 0.8)
            for (variant, monotone) in VARIANTS
                q = reshape(c0 .* col.m, :, 1)
                ref = copy(c0)
                largest = maximum(abs, q)
                for _ in 1:ncall
                    IFSC._ifs_column!(q, nothing, col.m, col.detu, col.detd, s.mu, s.md, s.cu, s.cd,
                                      1.0, dt, s.courant, 0.8, variant)
                    largest = max(largest, maximum(abs, q))
                    monotone && continue
                    for _ in 1:n
                        ref = ifs_convection_step(ref, col.entu, col.detu, col.entd, col.detd,
                                                  col.m, dt / n)
                    end
                end
                c = q[:, 1] ./ col.m
                mass = abs(sum(q) - sum(c0 .* col.m)) / largest / N
                if monotone
                    lo, hi = extrema(c0)
                    worst_bounds = max(worst_bounds, max(lo - minimum(c), maximum(c) - hi, 0.0) / hi)
                    worst_mass_mono = max(worst_mass_mono, mass)
                elseif all(isfinite, ref) && all(isfinite, c)
                    worst_mass_cuc = max(worst_mass_cuc, mass)
                    scale = max(maximum(abs, c0), maximum(abs, ref))
                    worst_ref = max(worst_ref, maximum(abs, c .- ref) / scale)
                    grown += maximum(abs, c) > 10 * maximum(abs, c0)
                else                        # overflowed: both or neither
                    @test !all(isfinite, ref) && !all(isfinite, c)
                    grown += 1
                end
            end
        end
        @info "IFS $(ncall) calls, kind $(kind)" worst_bounds worst_mass_mono worst_ref worst_mass_cuc cuctracer_grown_10x = "$(grown) of $(ncol)"
        @test worst_bounds < 1e-12
        @test worst_mass_mono < 1e-14
        @test worst_mass_cuc < 1e-13
        @test worst_ref < 1e-9
    end
end

@testset "IFSConvection: the cuctracer arithmetic can go negative, the monotone one cannot" begin
    # A tracer confined to one layer inside a downdraft-dominated column.
    rng = Xoshiro(3)
    found_negative = false
    for _ in 1:200
        N = 20
        col = random_column(rng, N, :consistent)
        c0 = zeros(N); c0[rand(rng, 2:N)] = 1.0
        dt = 900.0
        s = column_setup(col.m, col.entu, col.detu, col.entd, col.detd, dt)
        s.courant <= 1 || continue
        c_ifs, _ = production_step(c0, col.m, col.entu, col.detu, col.entd, col.detd, dt, IFSCuctracer())
        c_mon, _ = production_step(c0, col.m, col.entu, col.detu, col.entd, col.detd, dt,
                                   IFSMonotoneUpstream())
        found_negative |= minimum(c_ifs) < -1e-6
        @test minimum(c_mon) >= -1e-15
    end
    @test found_negative
end

# --- Float32: flux closure, ledger, uniform field, bounds -----------------------

# A plume-consistent column in Float32 whose cumulative sums do not close: the
# fields are rounded, and entu, detu are perturbed by a relative 1e-6.
function unclosed_column(rng, N, kind)
    col = random_column(rng, N, kind)
    jitter(x) = Float32.(x .* (1 .+ 1e-6 .* randn(rng, length(x))))
    return (; m = Float32.(col.m), entu = jitter(col.entu), detu = jitter(col.detu),
            entd = Float32.(col.entd), detd = Float32.(col.detd))
end

@testset "IFSConvection: flux closure on Float32 sums that do not close" begin
    rng = Xoshiro(8)
    open_sums = 0
    for trial in 1:200
        N = 30
        col = unclosed_column(rng, N, trial % 2 == 0 ? :consistent : :elevated_base)
        kt = findfirst(>(0), col.detu)
        # The plain cumulative sum leaves flux above the cloud top in many columns.
        below = 0f0
        for k in N:-1:2
            below = max(below + col.entu[k] - col.detu[k], 0f0)
            k <= kt && below > 0 && (open_sums += 1; break)
        end
        s = column_setup(col.m, col.entu, col.detu, col.entd, col.detd, 900f0)
        @test all(iszero, s.mu[1:kt])
        @test s.lo == kt + 1
        # A tracer that grows towards the ground: the largest cell is in the cloud.
        q = reshape(Float32.(collect(1:N) .^ 2) .* col.m, :, 1)
        q0 = copy(q)
        for variant in (IFSMonotoneUpstream(), IFSCuctracer())
            IFSC._ifs_column!(q, zeros(Float32, 1), col.m, col.detu, col.detd, s.mu, s.md,
                              s.cu, s.cd, 1f0, 900f0, s.courant, 0.8f0, variant)
        end
        @test all(map(===, q[1:(kt - 1), 1], q0[1:(kt - 1), 1]))
    end
    @info "IFS Float32 columns whose updraft sum stays open above the cloud top" open_sums
    @test open_sums > 20
end

# Column fields as in the drift reproduction: an updraft from the ground to the
# top of layer `kt`, no downdraft, Float32.
function drift_column(rng, N, kt)
    entu, detu, entd, detd = (zeros(Float32, N) for _ in 1:4)
    Mu = zeros(N + 1)
    for i in (kt + 1):N
        Mu[i] = 0.02 * (0.2 + 0.8 * rand(rng))
    end
    for k in kt:N
        lo = max(0.0, Mu[k + 1] - Mu[k]); hi = Mu[k + 1]
        d = k == N ? 0.0 : lo + (hi - lo) * rand(rng)
        detu[k] = d; entu[k] = Mu[k] - Mu[k + 1] + d
    end
    return entu, detu, entd, detd
end

@testset "IFSConvection: the downdraft ends in its lowest detraining layer" begin
    rng = Xoshiro(21)
    ended_above_ground = 0
    for trial in 1:100
        N = 30
        col = random_column(rng, N, :consistent)
        kb = rand(rng, (N - 6):(N - 1))                  # no downdraft forcing below layer kb
        entd = copy(col.entd); detd = copy(col.detd)
        entd[(kb + 1):N] .= 0; detd[(kb + 1):N] .= 0
        last_detraining = findlast(>(0), detd)
        last_detraining === nothing && continue
        s = column_setup(col.m, col.entu, col.detu, entd, detd, 900.0)
        @test all(iszero, s.md[(last_detraining + 1):N])
        Mu, Dn = ifs_interface_fluxes(col.entu, col.detu, entd, detd)
        @test s.md ≈ Dn[1:N] atol = 0 rtol = 1e-14
        ended_above_ground += last_detraining < N
    end
    @test ended_above_ground > 50
end

@testset "IFSConvection: the ledger bound comes from the rewritten layers" begin
    # A loss of 1e-3 of an active cell is not rounding, also under a cell 1000 times larger
    # above the cloud: the ledger must leave it in the field. A loss of one ulp is returned.
    N = 117; top = 60
    q = ones(Float32, N, 1); q[10, 1] = 1000f0
    before = IFSC._tm5_column_sum(q, 1, N, 1)
    q[80, 1] -= 1f-3
    lost = copy(q)
    carry = IFSC._ifs_restore_column_mass!(q, top, N, 1, before, 0f0)
    @test q == lost
    @test carry == 0
    q = ones(Float32, N, 1); q[10, 1] = 1000f0
    before = IFSC._tm5_column_sum(q, 1, N, 1)
    q[80, 1] = prevfloat(q[80, 1])
    carry = IFSC._ifs_restore_column_mass!(q, top, N, 1, before, 0f0)
    @test sum(big.(q)) + big(carry) == big(1000) + (N - 1)
end

@testset "IFSConvection: Float32 column drift with a cloud top that changes ($label)" for (label, release, tops) in
        (("above every cloud top", 10, 20:40), ("inside, top varies around it", 30, 20:40),
         ("inside, fixed top", 30, 20:20), ("at the ground", 60, 20:40),
         ("just above a fixed top", 19, 20:20))
    # A one-layer tracer; exact sums. What the column misses must be its carry,
    # about half an ulp of its largest cell, whatever the number of calls.
    rng = MersenneTwister(1)
    N, ncol, ncall = 60, 60, 300
    worst = 0.0
    total0 = big(0.0); total1 = big(0.0)
    for variant in (IFSMonotoneUpstream(), IFSCuctracer()), _ in 1:ncol
        m = Float32.(2f3 .+ 6f3 .* rand(rng, N))
        q = zeros(Float32, N, 1); q[release, 1] = Float32(1e6 * (1 + rand(rng)))
        carry = zeros(Float32, 1)
        mu, md, cu, cd = (zeros(Float32, N) for _ in 1:4)
        before = sum(big, q)
        for _ in 1:ncall
            entu, detu, entd, detd = drift_column(rng, N, rand(rng, tops))
            courant = IFSC._ifs_column_fluxes!(mu, md, m, entu, detu, entd, detd, 1f0, 900f0)
            IFSC._ifs_column!(q, carry, m, detu, detd, mu, md, cu, cd, 1f0, 900f0, courant, 0.8f0, variant)
        end
        after = sum(big, q)
        worst = max(worst, Float64(abs(after - before) / maximum(abs, q)))
        total0 += before; total1 += after
    end
    drift = Float64((total1 - total0) / total0)
    @info "IFS Float32 drift, release $(label), $(ncall) calls" worst_in_largest_cell = worst drift
    @test worst <= eps(Float32)
end

# Float32 uniform field over many calls; frozen or changing forcing, plume-consistent kinds.
function uniform_drift(variant; changing, ncall = 300, ncol = 40, seed = 42)
    rng = Xoshiro(seed)
    N = 40
    kinds = (:consistent, :excess_detrainment, :downdraft_above, :elevated_base)
    worst = 0.0
    for c in 1:ncol
        cols = [random_column(rng, N, kinds[mod1(c + i, 4)]) for i in 1:(changing ? 8 : 1)]
        area = 1f10
        m = Float32.(cols[1].m .* area)
        q = reshape(0.37f0 .* m, :, 1)
        mu, md, cu, cd = (zeros(Float32, N) for _ in 1:4)
        carry = zeros(Float32, 1)
        for call in 1:ncall
            col = cols[mod1(call, length(cols))]
            f = map(x -> Float32.(x), (col.entu, col.detu, col.entd, col.detd))
            courant = IFSC._ifs_column_fluxes!(mu, md, m, f..., area, 900f0)
            IFSC._ifs_column!(q, carry, m, f[2], f[4], mu, md, cu, cd, area, 900f0, courant, 0.8f0, variant)
        end
        worst = max(worst, maximum(abs, q[:, 1] ./ m .- 0.37f0) / 0.37)
    end
    return worst
end

@testset "IFSConvection: Float32 uniform field over many calls" begin
    frozen = uniform_drift(IFSMonotoneUpstream(); changing = false)
    changing = uniform_drift(IFSMonotoneUpstream(); changing = true)
    @info "IFS Float32 uniform field, 300 calls, monotone variant" frozen changing
    @test frozen <= 4 * eps(Float32)
    @test changing <= 4 * eps(Float32)
end

@testset "IFSConvection: Float32 bounds of the monotone variant" begin
    rng = Xoshiro(21)
    lowest = 0.0; highest = 0.0
    for trial in 1:400
        N = rand(rng, 12:40)
        col = random_column(rng, N, KINDS[mod1(trial, length(KINDS))])
        f = map(x -> Float32.(x), (col.entu, col.detu, col.entd, col.detd))
        m = Float32.(col.m)
        c0 = zeros(Float32, N); c0[rand(rng, 1:N)] = 1f0           # a unit tracer in one layer
        q = reshape(c0 .* m, :, 1)
        mu, md, cu, cd = (zeros(Float32, N) for _ in 1:4)
        for _ in 1:20
            courant = IFSC._ifs_column_fluxes!(mu, md, m, f..., 1f0, 1800f0)
            IFSC._ifs_column!(q, zeros(Float32, 1), m, f[2], f[4], mu, md, cu, cd, 1f0, 1800f0,
                              courant, 0.8f0, IFSMonotoneUpstream())
        end
        c = q[:, 1] ./ m
        lowest = min(lowest, minimum(c)); highest = max(highest, maximum(c) - 1)
    end
    @info "IFS Float32 monotone variant, unit tracer, 20 calls: excursions" lowest highest
    # Up to rounding: a few ulps of the unit maximum.
    @test lowest >= -2 * eps(Float32)
    @test highest <= 2 * eps(Float32)
end

# --- Topologies and the operator -------------------------------------------------

# Twelve columns; each topology places them in its own cells with their own areas.
function topology_columns(FT; Nz = 12, seed = 5, kinds = KINDS)
    rng = Xoshiro(seed)
    cols = [random_column(rng, Nz, kinds[mod1(c, length(kinds))]) for c in 1:12]
    cols[7] = (; m = cols[7].m, entu = zeros(Nz), detu = zeros(Nz), entd = zeros(Nz), detd = zeros(Nz))
    tracer = [hcat(random_profile(rng, Nz), fill(1.0, Nz)) for _ in 1:12]
    return (; cols, tracer, Nz)
end

# Fill one cell: air mass per cell from kg m-2 and the cell's area.
function fill_cell!(m, q, fields, I, J, col, tracer, area, FT)
    m[I..., :] .= FT.(col.m .* area)
    for (f, src) in zip(fields, (col.entu, col.detu, col.entd, col.detd))
        f[J..., :] .= FT.(src)
    end
    q[I..., :, :] .= FT.(tracer .* (col.m .* area))
end

const LL_AREAS = (1.7e9, 2.6e9, 3.1e9)                 # one per latitude row

function ll_case(FT, tc)
    Nx, Ny, Nz = 4, 3, tc.Nz
    m = zeros(FT, Nx, Ny, Nz); q = zeros(FT, Nx, Ny, Nz, 2)
    fields = ntuple(_ -> zeros(FT, Nx, Ny, Nz), 4)
    for j in 1:Ny, i in 1:Nx
        c = i + Nx * (j - 1)
        fill_cell!(m, q, fields, (i, j), (i, j), tc.cols[c], tc.tracer[c], LL_AREAS[j], FT)
    end
    forcing = ConvectionForcing(nothing, nothing, NamedTuple{(:entu, :detu, :entd, :detd)}(fields))
    ws = IFSWorkspace(m; cell_metrics = collect(FT, LL_AREAS))
    return (; q, m, forcing, ws, area = (i, j) -> FT(LL_AREAS[j]))
end

rg_area(c) = 1e9 * (1 + 0.37 * c)

function rg_case(FT, tc)
    Nz = tc.Nz
    m = zeros(FT, 12, Nz); q = zeros(FT, 12, Nz, 2)
    fields = ntuple(_ -> zeros(FT, 12, Nz), 4)
    for c in 1:12
        fill_cell!(m, q, fields, (c,), (c,), tc.cols[c], tc.tracer[c], rg_area(c), FT)
    end
    forcing = ConvectionForcing(nothing, nothing, NamedTuple{(:entu, :detu, :entd, :detd)}(fields))
    ws = IFSWorkspace(m; cell_metrics = FT.(rg_area.(1:12)))
    return (; q, m, forcing, ws, area = c -> FT(rg_area(c)))
end

cs_area(i, j, p) = 1e9 * (1 + 0.1 * i + 0.3 * j + 0.05 * p)

# Nc = 2: 24 columns, column (i, j) of panel p holds topology column mod1(i + 2(j-1) + 4(p-1), 12).
function cs_case(FT, tc; Hp = 1)
    Nc, Nz = 2, tc.Nz
    n = Nc + 2Hp
    m = ntuple(_ -> fill(FT(1), n, n, Nz), 6)
    q = ntuple(_ -> fill(FT(-7), n, n, Nz, 2), 6)       # halo values must stay
    fields = ntuple(_ -> ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6), 4)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        c = mod1(i + Nc * (j - 1) + 4 * (p - 1), 12)
        fill_cell!(m[p], q[p], map(f -> f[p], fields), (i + Hp, j + Hp), (i, j), tc.cols[c],
                   tc.tracer[c], cs_area(i, j, p), FT)
    end
    forcing = ConvectionForcing(nothing, nothing, NamedTuple{(:entu, :detu, :entd, :detd)}(fields))
    ws = IFSWorkspace(m; cell_metrics = ntuple(p -> [FT(cs_area(i, j, p)) for i in 1:Nc, j in 1:Nc], 6))
    mesh = CubedSphereMesh(; Nc, Hp, FT)
    return (; q, m, forcing, ws, mesh, Hp, Nc)
end

function vertical(FT, Nz)
    b = collect(FT, range(0, 1; length = Nz + 1))
    return HybridSigmaPressure(FT(2e4) .* b .* (1 .- b), b)
end

ll_grid(FT, Nz) = AtmosGrid(LatLonMesh(; Nx = 4, Ny = 3, FT), vertical(FT, Nz), AtmosTransport.CPU(); FT)
rg_grid(FT, Nz) = AtmosGrid(ReducedGaussianMesh(FT[-45, 0, 45], [4, 4, 4]; FT), vertical(FT, Nz),
                            AtmosTransport.CPU(); FT)

# The column routine on one column of host arrays, with a fresh carry.
function direct_column(q, m, entu, detu, entd, detd, area, dt, op)
    FT = eltype(q)
    q = copy(q)
    N = length(m)
    mu, md, cu, cd = (zeros(FT, N) for _ in 1:4)
    courant = IFSC._ifs_column_fluxes!(mu, md, m, entu, detu, entd, detd, area, FT(dt))
    IFSC._ifs_column!(q, zeros(FT, size(q, 2)), m, detu, detd, mu, md, cu, cd, area, FT(dt),
                      courant, FT(op.cfl_limit), op.variant)
    return q
end

@testset "IFSConvection: each topology equals the column routine with its cell area ($FT)" for FT in (Float32, Float64)
    tc = topology_columns(FT)
    dt = 2700
    for variant in (IFSCuctracer(), IFSMonotoneUpstream())
        op = IFSConvection(; variant)
        ll = ll_case(FT, tc); rg = rg_case(FT, tc); cs = cs_case(FT, tc)
        ll0 = copy(ll.q); rg0 = copy(rg.q); cs0 = map(copy, cs.q)
        apply_convection!(ll.q, ll.m, ll.forcing, op, dt, ll.ws, ll_grid(FT, tc.Nz))
        apply_convection!(rg.q, rg.m, rg.forcing, op, dt, rg.ws, rg_grid(FT, tc.Nz))
        apply_convection!(cs.q, cs.m, cs.forcing, op, dt, cs.ws,
                          AtmosGrid(cs.mesh, vertical(FT, tc.Nz), AtmosTransport.CPU(); FT))
        f = ll.forcing.tm5_fields
        for j in 1:3, i in 1:4
            @test ll.q[i, j, :, :] == direct_column(ll0[i, j, :, :], ll.m[i, j, :], f.entu[i, j, :],
                f.detu[i, j, :], f.entd[i, j, :], f.detd[i, j, :], ll.area(i, j), dt, op)
        end
        f = rg.forcing.tm5_fields
        for c in 1:12
            @test rg.q[c, :, :] == direct_column(rg0[c, :, :], rg.m[c, :], f.entu[c, :], f.detu[c, :],
                                                 f.entd[c, :], f.detd[c, :], rg.area(c), dt, op)
        end
        f = cs.forcing.tm5_fields
        H = cs.Hp
        for p in 1:6, j in 1:2, i in 1:2
            @test cs.q[p][i + H, j + H, :, :] == direct_column(cs0[p][i + H, j + H, :, :],
                cs.m[p][i + H, j + H, :], f.entu[p][i, j, :], f.detu[p][i, j, :],
                f.entd[p][i, j, :], f.detd[p][i, j, :], FT(cs_area(i, j, p)), dt, op)
        end
        # Halo cells and the column without convection are untouched bit for bit.
        for p in 1:6
            interior = falses(size(cs.q[p])); interior[2:3, 2:3, :, :] .= true
            @test cs.q[p][.!interior] == cs0[p][.!interior]
        end
        @test all(map(===, ll.q[3, 2, :, :], ll0[3, 2, :, :]))
        @test ll.q != ll0
        # Column mass with the ledger, exact sums: what is missing is the carry.
        for t in 1:2, j in 1:3, i in 1:4
            before = sum(big, ll0[i, j, :, t]); after = sum(big, ll.q[i, j, :, t])
            @test abs(after - before) <= eps(FT) * maximum(abs, ll0[i, j, :, t])
        end
        # The uniform tracer stays uniform.
        @test maximum(abs, ll.q[:, :, :, 2] ./ ll.m .- 1) <= 16 * eps(FT)
    end
end

@testset "IFSConvection: Float32 column ledger and its persistent state" begin
    FT = Float32
    tc = topology_columns(FT; Nz = 30, seed = 99, kinds = (:consistent, :elevated_base))
    ll = ll_case(FT, tc)
    op = IFSConvection()
    grid = ll_grid(FT, tc.Nz)
    before = [sum(big, ll.q[i, j, :, t]) for i in 1:4, j in 1:3, t in 1:2]
    for _ in 1:50
        apply_convection!(ll.q, ll.m, ll.forcing, op, 1800, ll.ws, grid)
    end
    after = [sum(big, ll.q[i, j, :, t]) for i in 1:4, j in 1:3, t in 1:2]
    largest = [maximum(abs, ll.q[i, j, :, t]) for i in 1:4, j in 1:3, t in 1:2]
    drift = Float64(maximum(abs.(after .- before) ./ largest))
    @info "IFS Float32 column drift after 50 calls (units of the largest cell)" drift
    @test drift <= eps(FT)
    @test size(ll.ws.ledger_carry) == (4, 3, 2)
    snap = Operators.operator_state(ll.ws)
    Operators.reset_operator_state!(ll.ws)
    @test all(iszero, ll.ws.ledger_carry)
    Operators.restore_operator_state!(ll.ws, snap)
    @test ll.ws.ledger_carry == snap.carry
end

@testset "IFSConvection: construction, forcing and Courant checks" begin
    @test IFSConvection() === IFSConvection(; cfl_limit = 0.8, variant = IFSMonotoneUpstream())
    @test IFSConvection(; cfl_limit = 1).cfl_limit === 1.0
    for bad in (0.0, -0.5, 1.01, NaN, Inf)
        @test_throws ArgumentError IFSConvection(; cfl_limit = bad)
    end
    FT = Float64
    tc = topology_columns(FT)
    grid = ll_grid(FT, tc.Nz)
    message(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end
    ll = ll_case(FT, tc)
    @test occursin("tm5_fields", message(() ->
        apply_convection!(ll.q, ll.m, ConvectionForcing(), IFSConvection(), 900, ll.ws, grid)))
    @test occursin("cell_metrics", message(() ->
        apply_convection!(ll.q, ll.m, ll.forcing, IFSConvection(), 900, IFSWorkspace(ll.m), grid)))
    # Every rejection stops the call before any tracer moves.
    q0 = copy(ll.q)
    for (label, edit!, expected) in (
            ("NaN field", c -> (c.forcing.tm5_fields.entu[2, 2, 5] = NaN), "non-finite"),
            ("negative field", c -> (c.forcing.tm5_fields.detd[1, 3, 9] = -1e-4), "negative"),
            ("layer without air under a flux", c -> (c.m[1, 1, 11] = 0), "without air"),
            ("layer without air below every flux", c -> (c.m[2, 1, end] = 0; foreach(f -> (f[2, 1, end-1:end] .= 0), c.forcing.tm5_fields)), "without air"))
        bad = ll_case(FT, tc)
        edit!(bad)
        msg = message(() -> apply_convection!(bad.q, bad.m, bad.forcing, IFSConvection(), 900, bad.ws, grid))
        @test occursin(expected, msg)
        @test bad.q == q0
    end
    @test occursin("sub-steps", message(() ->
        apply_convection!(ll.q, ll.m, ll.forcing, IFSConvection(), 1e9, ll.ws, grid)))
    @test ll.q == q0
    # A layer without air above every flux is accepted and left alone.
    quiet = ll_case(FT, tc)
    for f in quiet.forcing.tm5_fields
        f[:, :, 1:2] .= 0
    end
    quiet.m[:, :, 1] .= 0; quiet.q[:, :, 1, :] .= 0
    apply_convection!(quiet.q, quiet.m, quiet.forcing, IFSConvection(), 900, quiet.ws, grid)
    @test all(iszero, quiet.q[:, :, 1, :])
    @test all(isfinite, quiet.q)
end

@testset "IFSConvection through TransportModel and DrivenSimulation" begin
    FT = Float64
    tc = topology_columns(FT)
    ll = ll_case(FT, tc)
    grid = ll_grid(FT, tc.Nz)
    # The model's workspace takes the grid's cell areas: rescale the cells to them.
    areas = AtmosTransport.Grids.cell_areas_by_latitude(grid.horizontal)
    for j in 1:3
        ll.m[:, j, :] .*= areas[j] / LL_AREAS[j]
        ll.q[:, j, :, :] .*= areas[j] / LL_AREAS[j]
    end
    state = CellState(copy(ll.m); CO2 = copy(ll.q[:, :, :, 1]), UNI = copy(ll.q[:, :, :, 2]))
    fluxes = allocate_face_fluxes(grid.horizontal, tc.Nz; FT, basis = DryBasis)
    model = Models.TransportModel(state, fluxes, grid, UpwindScheme();
                                  convection = IFSConvection(), convection_forcing = ll.forcing)
    @test model.workspace.convection_ws isa IFSWorkspace
    direct = CellState(copy(ll.m); CO2 = copy(ll.q[:, :, :, 1]), UNI = copy(ll.q[:, :, :, 2]))
    Operators.apply!(direct, ll.forcing, grid, IFSConvection(), FT(1800);
                     workspace = convection_workspace_for(IFSConvection(), direct, grid))
    total = sum(model.state.tracers_raw[:, :, :, 1])
    Models.step!(model, FT(1800))
    @test model.state.tracers_raw ≈ direct.tracers_raw rtol = 1e-14
    @test model.state.tracers_raw[:, :, :, 1] != ll.q[:, :, :, 1]
    @test isapprox(sum(model.state.tracers_raw[:, :, :, 1]), total; rtol = 1e-14)

    for FT in (Float32, Float64)
        driver, _, _ = _make_convection_window_driver(; FT, tm5 = true)
        m = copy(driver.windows[1].air_mass)
        tracer = zeros(FT, 4, 3, 5); tracer[:, :, 5] .= FT(1e-6) .* m[:, :, 5]
        model = Models.TransportModel(CellState(m; CO2 = tracer),
                                      allocate_face_fluxes(driver.grid.horizontal, 5; FT, basis = DryBasis),
                                      driver.grid, UpwindScheme(); convection = IFSConvection())
        sim = Models.DrivenSimulation(model, driver; start_window = 1, stop_window = 2)
        initial = copy(sim.model.state.tracers_raw)
        Models.step!(sim); Models.step!(sim)
        @test isapprox(sum(Float64, sim.model.state.tracers_raw), sum(Float64, initial); rtol = 10eps(FT))
        @test sum(sim.model.state.tracers_raw[:, :, 5, 1]) < sum(initial[:, :, 5, 1])
        # A window without the TM5 fields is rejected.
        cmfmc_driver, _, _ = _make_convection_window_driver(; FT)
        @test_throws ArgumentError RunnerCore._validate_convection_window!(
            IFSConvection(), cmfmc_driver.windows[1], cmfmc_driver)
        @test RunnerCore._validate_convection_window!(IFSConvection(), driver.windows[1], driver) === nothing
    end
end

@testset "IFSConvection configuration" begin
    spec = Models.convection_spec(Dict("kind" => "ifs"))
    @test spec isa Models.IFSConvectionSpec
    @test spec.cfl_limit == 0.8 && spec.variant === IFSMonotoneUpstream()
    style = Physics.LatLonRuntimeRecipeStyle()
    @test Models.materialize(spec, style) === IFSConvection()
    cfg = Dict("convection" => Dict("kind" => "ifs", "cfl_limit" => 0.5, "variant" => "cuctracer"))
    op = Models.build_runtime_convection(cfg, style)
    @test op === IFSConvection(; cfl_limit = 0.5, variant = IFSCuctracer())
    @test Physics._convection_attributes(Models.convection_spec(cfg["convection"])) == Dict(
        "physics_convection" => "ifs", "physics_convection_variant" => "cuctracer",
        "physics_convection_cfl_limit" => "0.5")
    @test Physics._convection_attributes(spec)["physics_convection_variant"] == "monotone"
    for bad in (Dict("kind" => "ifs", "cfl_limit" => 0), Dict("kind" => "ifs", "cfl_limit" => 1.5),
                Dict("kind" => "ifs", "variant" => "ifs"), Dict("kind" => "ifs", "variant" => "upstream"),
                Dict("kind" => "ifs", "cloud_base" => "cmfmc"))
        @test_throws ArgumentError Models.convection_spec(bad)
    end
    # Unknown keys are errors; keys of another kind are warnings.
    findings(section) = Runner._config_key_findings(Dict("convection" => section))
    @test isempty(findings(Dict("kind" => "ifs", "cfl_limit" => 0.5, "variant" => "monotone")).errors)
    @test isempty(findings(Dict("kind" => "ifs", "cfl_limit" => 0.5, "variant" => "monotone")).warnings)
    @test !isempty(findings(Dict("kind" => "ifs", "cfl_limt" => 0.5)).errors)
    @test !isempty(findings(Dict("kind" => "tm5", "variant" => "monotone")).warnings)
    @test !isempty(findings(Dict("kind" => "ifs", "use_collab_lu" => true)).warnings)
    # Binary capability and runtime forcing source.
    caps = (; tm5_convection = false, payload_sections = (:m, :am))
    @test_throws ArgumentError Runner._validate_convection_capability(IFSConvection(), caps)
    @test Runner._validate_convection_capability(IFSConvection(), merge(caps, (; tm5_convection = true))) === nothing
    err = try Models.validate_runtime_convection(style, IFSConvection(), nothing); nothing catch e; e end
    @test err isa ArgumentError && occursin("kind = \"ifs\"", err.msg)
    # The run-log header names the variant.
    @test occursin("IFSConvection(variant=monotone, cfl_limit=0.8)",
                   Runner._convection_label(IFSConvection()))
end

@testset "IFSConvection has no adjoint" begin
    reason = AtmosTransport.Adjoints.adjoint_support(IFSConvection())
    @test reason isa String && occursin("IFSConvection has no adjoint", reason)
    err = try
        AtmosTransport.Adjoints._require_cs_convection_workspace(IFSConvection(), nothing); nothing
    catch e; e end
    @test err isa ArgumentError && occursin("IFSConvection has no adjoint", err.msg)
    FT = Float64
    tc = topology_columns(FT)
    ll = ll_case(FT, tc)
    grid = ll_grid(FT, tc.Nz)
    state = CellState(copy(ll.m); CO2 = copy(ll.q[:, :, :, 1]))
    model = Models.TransportModel(state, allocate_face_fluxes(grid.horizontal, tc.Nz; FT, basis = DryBasis),
                                  grid, UpwindScheme(); convection = IFSConvection(),
                                  convection_forcing = ll.forcing)
    err = try AtmosTransport.Adjoints.require_adjoint_support(model); nothing catch e; e end
    @test err isa ArgumentError && occursin("IFSConvection has no adjoint", err.msg)
    # The tape size estimate refuses it before it looks at any input.
    mesh = CubedSphereMesh(; Nc = 2, Hp = 1, FT)
    err = try
        AtmosTransport.Adjoints.cs_tape_byte_estimate(nothing, nothing, nothing, nothing, mesh,
                                                      UpwindScheme(); flux_scale = 1.0,
                                                      convection_op = IFSConvection())
        nothing
    catch e; e end
    @test err isa ArgumentError && occursin("IFSConvection has no adjoint", err.msg)
end

# --- GPU -----------------------------------------------------------------------

const _HAS_CUDA = try
    @eval using CUDA
    @eval CUDA.functional()
catch
    false
end

@testset "IFSConvection on CUDA matches the CPU ($FT)" for FT in (Float32, Float64)
    if !_HAS_CUDA
        @test_skip "CUDA backend required"
    else
        tc = topology_columns(FT; Nz = 40, seed = 17)
        for variant in (IFSCuctracer(), IFSMonotoneUpstream())
            op = IFSConvection(; variant)
            ll = ll_case(FT, tc); rg = rg_case(FT, tc); cs = cs_case(FT, tc)
            dev(x::AbstractArray) = CuArray(x)
            dev(x::Tuple) = map(dev, x)
            dev(f::ConvectionForcing) = ConvectionForcing(nothing, nothing, map(dev, f.tm5_fields))
            cases = ((ll, ll_grid(FT, tc.Nz)), (rg, rg_grid(FT, tc.Nz)),
                     (cs, AtmosGrid(cs.mesh, vertical(FT, tc.Nz), AtmosTransport.CPU(); FT)))
            for (case, grid) in cases
                q_dev = dev(case.q); m_dev = dev(case.m)
                ws_dev = IFSWorkspace(m_dev; cell_metrics = case.ws.cell_metrics)
                for _ in 1:3
                    apply_convection!(case.q, case.m, case.forcing, op, 5400, case.ws, grid)
                    apply_convection!(q_dev, m_dev, dev(case.forcing), op, 5400, ws_dev, grid)
                end
                panels(x) = x isa Tuple ? x : (x,)
                gpu = map(Array, panels(q_dev))
                cpu = panels(case.q)
                diff = maximum(map((a, b) -> maximum(abs, a .- b), gpu, cpu))
                scale = maximum(map(a -> maximum(abs, a), cpu))
                @info "IFS CUDA vs CPU" FT variant typeof(grid.horizontal) rel = diff / scale
                @test diff <= 64 * eps(FT) * scale
                @test all(c -> c isa CuArray, panels(ws_dev.ledger_carry))
            end
        end
    end
end
