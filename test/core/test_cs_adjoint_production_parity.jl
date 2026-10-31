# Host tape read-role regression (#57). Production FD/checkpoint coverage is
# in test_cs_adjoint_step_plan_parity.jl.
using Test, Random
import AtmosTransport
const AT = AtmosTransport
const TapeMod = AT.Tape
const Adv = AT.Operators.Advection
const NC = 4
const NZ = 5
const NSTEPS = 6

function _parity_problem(; Nc = NC, Nz = NZ, nsteps = NSTEPS, FT = Float64,
                         mass_scale = 1.0)
    mesh = AT.CubedSphereMesh(Nc = Nc, Hp = 3, FT = FT)
    Hp = mesh.Hp
    N = Nc + 2Hp
    S = mass_scale
    m = ntuple(p -> [FT(S * (2.0 + 0.25k + 0.01p + 0.03sin(0.7i + 0.3j + p)))
                     for i in 1:N, j in 1:N, k in 1:Nz], 6)
    rm = ntuple(p -> [m[p][i, j, k] * FT(0.08 + 0.01sin(0.27i + 0.13j + 0.19k + 0.07p))
                      for i in 1:N, j in 1:N, k in 1:Nz], 6)
    Adv.fill_panel_halos!(m, mesh; dir = 0)
    Adv.fill_panel_halos!(rm, mesh; dir = 0)
    am = [ntuple(p -> begin
              a = zeros(FT, N + 1, N, Nz)
              for k in 1:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc+1
                  a[i, j, k] = FT(S * 0.05sin(0.2s + 0.3p + 0.7i + 0.4j + 0.2k))
              end
              a
          end, 6) for s in 1:nsteps]
    bm = [ntuple(p -> begin
              b = zeros(FT, N, N + 1, Nz)
              for k in 1:Nz, j in Hp+1:Hp+Nc+1, i in Hp+1:Hp+Nc
                  b[i, j, k] = FT(S * 0.04cos(0.3s + 0.5p + 0.4i + 0.6j + 0.1k))
              end
              b
          end, 6) for s in 1:nsteps]
    cm = [ntuple(p -> begin
              c = zeros(FT, N, N, Nz + 1)
              for k in 2:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc
                  c[i, j, k] = FT(-S * 0.03(1 + 0.3sin(i + j + k + p + s)))
              end
              c
          end, 6) for s in 1:nsteps]
    return (; mesh, m, rm, am, bm, cm)
end

const PB57 = _parity_problem()
const MESH = PB57.mesh
const HP = MESH.Hp

_footprint_maxrel(a, b) =
    maximum(maximum(abs.(a.footprints[s][p] .- b.footprints[s][p]))
            for s in eachindex(a.footprints), p in 1:6) /
    maximum(maximum(abs.(b.footprints[s][p])) for s in eachindex(b.footprints), p in 1:6)

# Scratch directory of the mmap tapes. Host pools (pinned host, mmap) copy
# every read into a cache per read role, as on a GPU.
const SCRATCH = mktempdir()

@testset "host tape storage gives distinct air-mass and tracer-mass reads (#57)" begin
    @testset "slot reads ($(nameof(typeof(pool))))" for
            pool in (TapeMod._open_tape_pool(AT.PinnedHostCSTapeStorage(), 0),
                     TapeMod._MmapTapeFile(SCRATCH, 1 << 26))
        m_slot = TapeMod._stage_panels(pool, PB57.m)
        rm_slot = TapeMod._stage_panels(pool, PB57.rm)
        # Both operands of one limited-PPM sweep record, read as the reverse
        # loop reads them.
        m_read = TapeMod._tape_panels(m_slot, TapeMod.AirMassRead())
        rm_read = TapeMod._tape_panels(rm_slot, TapeMod.TracerMassRead())
        @test all(m_read[p] !== rm_read[p] for p in 1:6)
        @test all(m_read[p] == PB57.m[p] for p in 1:6)
        @test all(rm_read[p] == PB57.rm[p] for p in 1:6)
        # A read stays valid until the next read with the same role.
        TapeMod._tape_panels(rm_slot, TapeMod.TracerMassRead())
        @test all(m_read[p] == PB57.m[p] for p in 1:6)
        TapeMod._release_tape_pool!(pool)
    end

    objective = AT.CSLayerMeanObjective(2, 2, 3, NZ)
    nsteps = 4
    am, bm, cm = PB57.am[1:nsteps], PB57.bm[1:nsteps], PB57.cm[1:nsteps]
    @testset "footprints, PPM $(nameof(typeof(limiter)))" for
            limiter in (AT.MonotoneLimiter(), AT.CW84Limiter())
        scheme = AT.PPMScheme(limiter)
        footprint(storage, checkpoint) = AT.cs_surface_emission_footprint(
            PB57.rm, PB57.m, am, bm, cm, MESH, objective;
            scheme, dt = 30.0, tape = AT.AdjointTape(; storage, checkpoint))
        for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2))
            reference = footprint(AT.DeviceCSTapeStorage(), checkpoint)
            for storage in (AT.PinnedHostCSTapeStorage(), AT.MmapCSTapeStorage(SCRATCH))
                @test _footprint_maxrel(footprint(storage, checkpoint), reference) <= 1e-12
            end
        end

        seed = ntuple(p -> (a = zeros(size(PB57.rm[p])); a[HP + 2, HP + 2, NZ] = 1.0; a), 6)
        from_seed(storage, checkpoint) = AT.cs_surface_emission_footprint_from_seed(
            seed, PB57.m, am, bm, cm, MESH;
            scheme, dt = 30.0, base_panels_rm0 = PB57.rm, tape = AT.AdjointTape(; storage, checkpoint))
        for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2))
            @test _footprint_maxrel(from_seed(AT.PinnedHostCSTapeStorage(), checkpoint),
                                    from_seed(AT.DeviceCSTapeStorage(), checkpoint)) <= 1e-12
        end
    end

    @testset "4D-Var gradient, PPM MonotoneLimiter" begin
        scheme = AT.PPMScheme(AT.MonotoneLimiter())
        rng = MersenneTwister(5)
        control = AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:all, 1:nsteps; normalize = true),
            ntuple(_ -> 0.02 .* rand(rng, NC, NC), 6))
        observations = [AT.CSObservation(nsteps, objective, 0.09, 0.01),
                        AT.CSObservation(3, AT.CSColumnMeanObjective(4, 1, 2), 0.07, 0.01)]
        direction = ntuple(_ -> randn(rng, NC, NC), 6)
        shifted(h) = AT.CSSurfaceFluxControl(control.window,
            ntuple(p -> control.value[p] .+ h .* direction[p], 6);
            background = control.background, sigma = control.sigma)
        fourdvar(c; storage = AT.DeviceCSTapeStorage()) = AT.cs_surface_flux_4dvar(
            PB57.rm, PB57.m, am, bm, cm, MESH, observations, [c];
            scheme, dt = 30.0, tape = AT.AdjointTape(; storage))
        gd(result) = sum(sum(result.gradients[1][p] .* direction[p]) for p in 1:6)
        h = 1e-7
        fd = (fourdvar(shifted(h)).cost - fourdvar(shifted(-h)).cost) / 2h
        reference = gd(fourdvar(control))
        @test abs(reference - fd) <= 1e-8 * abs(fd)
        for storage in (AT.PinnedHostCSTapeStorage(), AT.MmapCSTapeStorage(SCRATCH))
            @test abs(gd(fourdvar(control; storage)) - reference) <= 1e-12 * abs(reference)
        end
    end
end
