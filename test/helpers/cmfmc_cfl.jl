using AtmosTransport, Random, Test
const CMFMCConv = AtmosTransport.Operators.Convection

# Fields for the CMFMC CFL scan in all three layouts, with zero-mass columns
# (skipped by the scan) and the largest ratio wherever the random draw puts it;
# `single_hot_positions` places it at chosen interfaces instead.
function cmfmc_cfl_fixture(FT; Nx = 7, Ny = 5, Nz = 6, Nc = 5, Hp = 3, seed = 2718)
    rng = MersenneTwister(seed)
    flux(dims...) = FT(0.02) .* randn(rng, FT, dims...)
    mass(dims...) = FT(5e3) .+ FT(1e4) .* rand(rng, FT, dims...)
    ll = (cmfmc = flux(Nx, Ny, Nz + 1), air_mass = mass(Nx, Ny, Nz),
          areas = FT(1e6) .+ FT(1e5) .* rand(rng, FT, Ny))
    ll.air_mass[2, 3, :] .= 0
    ncell = Nx * Ny
    fi = (cmfmc = flux(ncell, Nz + 1), air_mass = mass(ncell, Nz),
          areas = FT(1e6) .+ FT(1e5) .* rand(rng, FT, ncell))
    fi.air_mass[4, :] .= 0
    N = Nc + 2Hp
    cs = (cmfmc = ntuple(_ -> flux(Nc, Nc, Nz + 1), 6),
          air_mass = ntuple(_ -> mass(N, N, Nz), 6),
          areas = ntuple(_ -> FT(1e6) .+ FT(1e5) .* rand(rng, FT, Nc, Nc), 6))
    cs.air_mass[3][Hp + 2, Hp + 1, :] .= 0
    # Halo cells must not enter the scan: make them tiny so a wrong offset
    # produces a huge ratio.
    for p in 1:6
        a = cs.air_mass[p]
        a[1:Hp, :, :] .= FT(1e-3); a[Hp + Nc + 1:end, :, :] .= FT(1e-3)
        a[:, 1:Hp, :] .= FT(1e-3); a[:, Hp + Nc + 1:end, :] .= FT(1e-3)
    end
    return (; ll, fi, cs, Hp)
end

# Independent scalar reference: every interface against its thinner adjacent layer.
function reference_cmfmc_max_cfl(cmfmc, m, area, dt)
    FT = eltype(m)
    Nz = size(m, ndims(m))
    worst = zero(FT)
    for col in CartesianIndices(size(m)[1:end-1]), k in 1:Nz + 1
        mk = k == 1 ? m[col, 1] : k > Nz ? m[col, Nz] : min(m[col, k - 1], m[col, k])
        bmass = mk / FT(area(col))
        bmass > 0 && (worst = max(worst, abs(cmfmc[col, k]) * FT(dt) / bmass))
    end
    return worst
end

function reference_cmfmc_max_cfl(f)
    dt = 900
    ll = reference_cmfmc_max_cfl(f.ll.cmfmc, f.ll.air_mass, c -> f.ll.areas[c[2]], dt)
    fi = reference_cmfmc_max_cfl(f.fi.cmfmc, f.fi.air_mass, c -> f.fi.areas[c[1]], dt)
    Hp = f.Hp
    cs = maximum(1:6) do p
        Nc = size(f.cs.cmfmc[p], 1)
        interior = f.cs.air_mass[p][Hp + 1:Hp + Nc, Hp + 1:Hp + Nc, :]
        reference_cmfmc_max_cfl(f.cs.cmfmc[p], interior, c -> f.cs.areas[p][c], dt)
    end
    return (; ll, fi, cs, dt)
end

scan_cmfmc_max_cfl(f, adapt, dt) = (
    ll = CMFMCConv._cmfmc_max_cfl(adapt(f.ll.cmfmc), adapt(f.ll.air_mass), adapt(f.ll.areas), dt),
    fi = CMFMCConv._cmfmc_max_cfl(adapt(f.fi.cmfmc), adapt(f.fi.air_mass), adapt(f.fi.areas), dt),
    cs = CMFMCConv._cmfmc_max_cfl(map(adapt, f.cs.cmfmc), map(adapt, f.cs.air_mass),
                                  map(adapt, f.cs.areas), dt))

# Interfaces at the ends of every index range (and on every panel) where a
# dominant flux is placed in turn: with the maximum pinned there, a wrong
# column, level or halo offset in the scan returns a different value.
function single_hot_positions(f)
    Nx, Ny, Nz1 = size(f.ll.cmfmc)
    ncell = size(f.fi.cmfmc, 1)
    Nc = size(f.cs.cmfmc[1], 1)
    ll = [(i, j, k) for i in (1, Nx), j in (1, Ny), k in (1, 2, Nz1)]
    fi = [(c, k) for c in (1, ncell), k in (1, 2, Nz1)]
    cs = [(p, i, j, k) for p in 1:6, (i, j) in ((1, 1), (Nc, 2), (2, Nc)), k in (1, Nz1)]
    return (; ll, fi, cs)
end

function check_single_hot(f, adapt; dt = 900)
    FT = eltype(f.ll.air_mass)
    hot = FT(50)
    ok = Bool[]
    for pos in single_hot_positions(f).ll
        g = deepcopy(f); g.ll.cmfmc[pos...] = hot
        ref = reference_cmfmc_max_cfl(g)
        push!(ok, scan_cmfmc_max_cfl(g, adapt, dt).ll === ref.ll)
    end
    for pos in single_hot_positions(f).fi
        g = deepcopy(f); g.fi.cmfmc[pos...] = hot
        ref = reference_cmfmc_max_cfl(g)
        push!(ok, scan_cmfmc_max_cfl(g, adapt, dt).fi === ref.fi)
    end
    for (p, i, j, k) in single_hot_positions(f).cs
        g = deepcopy(f); g.cs.cmfmc[p][i, j, k] = hot
        ref = reference_cmfmc_max_cfl(g)
        push!(ok, scan_cmfmc_max_cfl(g, adapt, dt).cs === ref.cs)
    end
    return ok
end

# A NaN layer mass skips both interfaces of that layer (as Julia's `min` does on
# the host). The dominant flux sits exactly on those interfaces, so a backend
# whose `min` returns the finite neighbour (Metal's `fmin`) gives a different,
# much larger value. Returns one Bool per layout.
function check_nan_mass_skips_interfaces(f, adapt; dt = 900)
    FT = eltype(f.ll.air_mass)
    hot = FT(50)
    g = deepcopy(f)
    g.ll.air_mass[3, 2, 4] = FT(NaN); g.ll.cmfmc[3, 2, 4] = hot; g.ll.cmfmc[3, 2, 5] = hot
    g.fi.air_mass[6, 4] = FT(NaN);    g.fi.cmfmc[6, 4] = hot;    g.fi.cmfmc[6, 5] = hot
    g.cs.air_mass[4][g.Hp + 3, g.Hp + 2, 4] = FT(NaN)
    g.cs.cmfmc[4][3, 2, 4] = hot;     g.cs.cmfmc[4][3, 2, 5] = hot
    ref = reference_cmfmc_max_cfl(g)
    got = scan_cmfmc_max_cfl(g, adapt, dt)
    # The skipped interfaces would give ratios above 1e6 with the fixture's masses.
    return (got.ll === ref.ll && ref.ll < 1e6, got.fi === ref.fi && ref.fi < 1e6,
            got.cs === ref.cs && ref.cs < 1e6)
end

# Device checks shared by the opt-in GPU diagnostics: `adapt` moves a host array
# to the device (e.g. `CuArray`) and `float_types` lists the element types to
# test, so a backend without Float64 passes `float_types = (Float32,)`.
function check_cmfmc_cfl_on_device(adapt; float_types = (Float32, Float64))
    for FT in float_types
        f = cmfmc_cfl_fixture(FT; Nx = 37, Ny = 29, Nz = 72, Nc = 24)
        ref = reference_cmfmc_max_cfl(f)
        dev = scan_cmfmc_max_cfl(f, adapt, ref.dt)
        @test dev.ll === ref.ll
        @test dev.fi === ref.fi
        @test dev.cs === ref.cs
        @test all(check_single_hot(f, adapt))
        @test all(check_nan_mass_skips_interfaces(f, adapt))
        # NaN handling matches the host: a NaN mass skips its interfaces,
        # a NaN flux makes the scan NaN.
        f.ll.air_mass[5, 4, 3] = FT(NaN)
        f.cs.air_mass[2][f.Hp + 1, f.Hp + 3, 2] = FT(NaN)
        f.fi.cmfmc[3, 2] = FT(NaN)
        ref_nan = reference_cmfmc_max_cfl(f)
        dev_nan = scan_cmfmc_max_cfl(f, adapt, ref.dt)
        @test dev_nan.ll === ref_nan.ll && isfinite(dev_nan.ll)
        @test dev_nan.cs === ref_nan.cs && isfinite(dev_nan.cs)
        @test isnan(dev_nan.fi) && isnan(ref_nan.fi)
        # Mixed residency falls back to the host scan.
        @test CMFMCConv._cmfmc_max_cfl(adapt(f.ll.cmfmc), adapt(f.ll.air_mass),
                                       f.ll.areas, ref.dt) === ref_nan.ll
    end
end
