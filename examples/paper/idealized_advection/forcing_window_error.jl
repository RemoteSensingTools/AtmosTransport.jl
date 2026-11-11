#!/usr/bin/env julia
# Error of window-mean forcing in the deformational flows.
#
#   julia --project=. -t 16 examples/paper/idealized_advection/forcing_window_error.jl \
#         [--results <dir>] [--nlon 360]
#
# An offline model sees the wind as one mass transfer per met window. Inside a window the
# transfer is applied at a constant rate, so the model's parcels follow the window-mean wind and
# not the continuously time-dependent analytic wind that defines the reference solution. The
# difference does not depend on the grid: it is the error that remains with that window length
# when the spatial error has converged away. This script measures it without the model. Parcels are traced
# back from T (and T/2) through the window-mean wind (6-point Gauss–Legendre in time, RK4 with
# four steps per window) and the initial field at the departure point is compared with the exact
# solution on a regular sample of the sphere (normalized l2, as in `analyze.py`).
#
# For flow B (stream function, constant air mass) the window-mean wind is exactly what the
# transfers of `generate.jl` encode. For flow C the transfers are window means of density times
# wind, so the window-mean wind is an estimate of the same effect. Flow A is steady: no such error.
# The error scales with the square of the window length (see the printed table).
include(joinpath(@__DIR__, "IdealizedAdvection.jl"))
using .IdealizedAdvection
using LinearAlgebra, Printf, StaticArrays
using FastGaussQuadrature: gausslegendre
const IA = IdealizedAdvection

const TIME_NODES, TIME_WEIGHTS = gausslegendre(6)
const STEPS_PER_WINDOW = 4

window_mean_wind(flow, x, t0, t1) =
    sum(TIME_WEIGHTS[q] / 2 * IA.velocity(flow, x, t0 + (t1 - t0) * (TIME_NODES[q] + 1) / 2) for q in eachindex(TIME_NODES))

"""Position at `t0` of the parcel that the window-mean wind of `[t0, t1]` brings to `x` at `t1`."""
function back_through_window(flow, x, t0, t1)
    h = -(t1 - t0) / STEPS_PER_WINDOW
    wind(y) = (n = normalize(y); v = window_mean_wind(flow, n, t0, t1); v - dot(v, n) * n)
    for _ in 1:STEPS_PER_WINDOW
        k1 = wind(x)
        k2 = wind(x + h / 2 * k1)
        k3 = wind(x + h / 2 * k2)
        k4 = wind(x + h * k3)
        x = normalize(x + h / 6 * (k1 + 2k2 + 2k3 + k4))
    end
    return x
end

"""Normalized l2 difference at `fraction * T` between window-mean and exact advection, per field."""
function window_error(flow, fields, window_hours, fraction; nlon)
    nlat = nlon ÷ 2
    period = flow.period
    nwindow = round(Int, period / (3600 * window_hours))
    last_window = round(Int, fraction * nwindow)
    num = zeros(length(fields), nlat)
    den = zeros(length(fields), nlat)
    Threads.@threads for j in 1:nlat
        θ = -π / 2 + (j - 0.5) * π / nlat
        for i in 1:nlon
            λ = (i - 0.5) * 2π / nlon
            arrival = SVector(cos(θ) * cos(λ), cos(θ) * sin(λ), sin(θ))
            x = arrival
            for w in last_window:-1:1
                x = back_through_window(flow, x, (w - 1) * period / nwindow, w * period / nwindow)
            end
            exact = first(IA.trace_back(flow, arrival, last_window * period / nwindow))
            for (k, field) in enumerate(fields)
                num[k, j] += cos(θ) * (field(x) - field(exact))^2
                den[k, j] += cos(θ) * field(exact)^2
            end
        end
    end
    return sqrt.(vec(sum(num; dims = 2)) ./ vec(sum(den; dims = 2)))
end

function main()
    results = joinpath(@__DIR__, "results")
    nlon = 360
    for (i, a) in enumerate(ARGS)
        a == "--results" && (results = ARGS[i + 1])
        a == "--nlon" && (nlon = parse(Int, ARGS[i + 1]))
    end
    names = ("gaussian_hills", "cosine_bells")
    rows = String[]
    for id in ("B", "C")
        flow, tracers = flow_case(id)
        table = Dict(tracers)
        fields = [table[n] for n in names]
        for window_hours in (4.0, 2.0, 1.0, 0.5, 0.25), (label, fraction) in (("T/2", 0.5), ("T", 1.0))
            l2 = window_error(flow, fields, window_hours, fraction; nlon)
            for (n, e) in zip(names, l2)
                push!(rows, @sprintf("%s,%.2f,%s,%s,%.6e", id, window_hours, n, label, e))
            end
            @printf("flow %s, %5.2f h windows, %-3s: l2 %s\n", id, window_hours, label,
                    join((@sprintf("%s %.3e", n, e) for (n, e) in zip(names, l2)), ", "))
        end
    end
    mkpath(results)
    path = joinpath(results, "forcing_window_error.csv")
    open(path, "w") do io
        println(io, "flow,window_hours,tracer,time,l2")
        foreach(r -> println(io, r), rows)
    end
    println("wrote $path")
end

main()
