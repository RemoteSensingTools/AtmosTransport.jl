# Regenerates `test/fixtures/ifs_cuctracer_columns.txt`: random columns that satisfy the plume
# assumptions of the IFS convection scheme, and the result of one step of the unmodified OpenIFS
# tracer routine for each. See the README in this folder for the build of `ifs_driver`.
#
#     julia test/fixtures/ifs_cuctracer_generator/generate.jl <path to ifs_driver> <work dir>
using Random, Printf
include(joinpath(@__DIR__, "..", "..", "helpers", "ifs_convection_reference.jl"))
const GRAVITY = 9.80665      # value in stubs.F90

# Layer fields of one column. `detached`: cloud base above the two lowest layers, which lets the
# downdraft exceed the updraft near the ground and exercises the correction of the lowest layer.
function random_column(rng; N, downdraft, cfl, detached, dt = 900.0)
    m = 80 .+ 500 .* rand(rng, N)
    kt = rand(rng, 4:N÷2)                         # highest layer of the updraft
    Mu = zeros(N + 1)
    for i in kt+1:N
        Mu[i] = cfl * minimum(m[max(i - 1, 1):i]) / dt * (0.15 + 0.85 * rand(rng))
    end
    detached && (Mu[N-1:N] .= 0.0)
    entu = zeros(N); detu = zeros(N)
    for k in kt:N
        lo = max(0.0, Mu[k + 1] - Mu[k]); hi = Mu[k + 1]
        detu[k] = k == N ? 0.0 : lo + (hi - lo) * rand(rng) * (rand(rng) < 0.3 ? 0.0 : 1.0)
        detu[k] = max(detu[k], lo)
        entu[k] = Mu[k] - Mu[k + 1] + detu[k]
    end
    entd = zeros(N); detd = zeros(N); Dn = zeros(N + 1)
    if downdraft
        il = rand(rng, kt+3:N-3)                  # first interface of the downdraft
        for i in il:N
            Dn[i] = 0.3 * max(Mu[i], Mu[kt + 1]) * (0.2 + 0.8 * rand(rng))
        end
        for k in il-1:N
            lo = max(0.0, Dn[k] - Dn[k + 1]); hi = Dn[k]
            detd[k] = k == N ? Dn[k] : lo + (hi - lo) * rand(rng)
            entd[k] = Dn[k + 1] - Dn[k] + detd[k]
        end
    end
    return (; N, m, dt, entu, detu, entd, detd)
end

function tracers(rng, N)
    random = rand(rng, N)
    surface_maximum = collect(reverse(exp.(-range(0, 6; length = N))))
    one_layer = zeros(N); one_layer[rand(rng, 3:N-6)] = 1.0
    uniform = ones(N)
    empty_near_ground = rand(rng, N); empty_near_ground[end-2:end] .= 0.0
    above_lowest = zeros(N); above_lowest[N - 1] = 1.0
    return [random, surface_maximum, one_layer, uniform, empty_near_ground, above_lowest]
end

fmt(v) = join((@sprintf("%.17e", x) for x in v), " ")

function main(driver, dir)
    rng = MersenneTwister(4801)
    cols = [random_column(rng; N = rand(rng, 18:28), downdraft = n % 3 != 0, cfl = n % 2 == 0 ? 0.75 : 0.4,
                          detached = n in (4, 7, 10)) for n in 1:12]
    trs = [tracers(rng, col.N) for col in cols]
    open(joinpath(dir, "columns.txt"), "w") do io          # half-level input of the Fortran driver
        println(io, length(cols))
        for (col, tr) in zip(cols, trs)
            Mu, Dn = ifs_interface_fluxes(col.entu, col.detu, col.entd, col.detd)
            N = col.N
            kctop = findfirst(>(0), Mu[1:N]); kdtop = findfirst(>(0), Dn[1:N])
            println(io, N, " ", length(tr), " ", col.dt, " ", kctop, " ", something(kdtop, N + 1), " 1 ", kdtop === nothing ? 0 : 1)
            println(io, fmt([0.0; cumsum(col.m) .* GRAVITY])); println(io, fmt(Mu[1:N])); println(io, fmt(-Dn[1:N]))
            println(io, fmt(col.detu)); println(io, fmt([0.0; col.detd[1:N-1]]))
            foreach(t -> println(io, fmt(t)), tr)
        end
    end
    cd(() -> run(`$driver`), dir)
    lines = readlines(joinpath(dir, "tendencies.txt"))
    row = 0; worst = 0.0
    open(joinpath(@__DIR__, "..", "ifs_cuctracer_columns.txt"), "w") do io
        println(io, "# Columns for the IFS convection operator and the result of one step of the tracer routine of")
        println(io, "# OpenIFS (cuctracer.F90 and cubidiag.F90, commit 48c1db541c1839b3b8f55f1b005df1930d29f41b,")
        println(io, "# Apache-2.0), compiled unmodified with gfortran in double precision, RMFSOLCT = 1, RMFCMIN = 1e-8,")
        println(io, "# RMFADVW = 0, no scavenging, g = 9.80665. Layer 1 is the top. The Fortran half-level inputs were")
        println(io, "# built from the layer fields below: mass flux from the cumulative entrainment minus detrainment,")
        println(io, "# downdraft detrainment of layer k given on half level k + 1. Written by")
        println(io, "# test/fixtures/ifs_cuctracer_generator/generate.jl.")
        println(io, "# First line: number of columns. Per column: `N ntracers dt[s]`, then one line each of air mass")
        println(io, "# per area [kg m-2], entu, detu, entd, detd [kg m-2 s-1], then ntracers lines of mixing ratio before")
        println(io, "# and ntracers lines after the step.")
        println(io, length(cols))
        for (col, tr) in zip(cols, trs)
            println(io, col.N, " ", length(tr), " ", col.dt)
            foreach(v -> println(io, fmt(v)), (col.m, col.entu, col.detu, col.entd, col.detd))
            foreach(t -> println(io, fmt(t)), tr)
            for c in tr
                row += 1
                fortran = c .+ col.dt .* parse.(Float64, split(lines[row]))
                println(io, fmt(fortran))
                reference = ifs_convection_step(c, col.entu, col.detu, col.entd, col.detd, col.m, col.dt)
                worst = max(worst, maximum(abs, reference .- fortran))
            end
        end
    end
    println("rows: ", row, ", largest |reference - Fortran| = ", worst)
end

main(abspath(ARGS[1]), abspath(ARGS[2]))
