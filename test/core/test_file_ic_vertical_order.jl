#!/usr/bin/env julia
# File-based initial conditions start the right way up: files with hybrid
# coefficients are oriented by their interface pressures and conservatively
# remapped, whatever their level count; files without them must match the
# model's levels and say which end is the top.

using Test
import NCDatasets: NCDataset, defDim, defVar
using AtmosTransport

const NLEV = 4
# Model levels, top first: A [Pa], B; interface pressure A + B ps.
const A_TOP = [0.0, 5000.0, 10000.0, 5000.0, 0.0]
const B_TOP = [0.0, 0.0, 0.2, 0.6, 1.0]
const PS = 1.0e5
# A profile that tells the levels apart: k = 1 is the top.
const Q_TOP = [1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6]

# A horizontally uniform field on a 4 x 3 lon-lat grid, levels as given.
function write_ic(path; q, ap = nothing, bp = nothing, lev = Float64.(0:length(q) - 1),
                  lev_attrib = Dict{String, String}())
    NCDataset(path, "c") do ds
        nlev = length(q)
        defDim(ds, "lon", 4); defDim(ds, "lat", 3); defDim(ds, "lev", nlev)
        defVar(ds, "lon", Float64, ("lon",))[:] = [0.0, 90.0, 180.0, 270.0]
        defVar(ds, "lat", Float64, ("lat",))[:] = [-60.0, 0.0, 60.0]
        vlev = defVar(ds, "lev", Float64, ("lev",))
        vlev[:] = lev
        for (k, v) in lev_attrib
            vlev.attrib[k] = v
        end
        if ap !== nothing
            defDim(ds, "ilev", nlev + 1)
            defVar(ds, "ap", Float64, ("ilev",))[:] = ap
            defVar(ds, "bp", Float64, ("ilev",))[:] = bp
        end
        field = Array{Float64}(undef, 4, 3, nlev)
        for k in 1:nlev
            field[:, :, k] .= q[k]
        end
        defVar(ds, "TRACER", Float64, ("lon", "lat", "lev"))[:, :, :] = field
    end
    return path
end

grid = AtmosGrid(LatLonMesh(; FT = Float64, Nx = 4, Ny = 3), HybridSigmaPressure(A_TOP, B_TOP),
                 AtmosTransport.CPU(); FT = Float64)
column(path, extra...) = begin
    cfg = Dict{String, Any}("kind" => "file", "file" => path, "variable" => "TRACER", extra...)
    q = AtmosTransport.Models.InitialConditionIO.build_initial_mixing_ratio(
        ones(4, 3, NLEV), grid, cfg; surface_pressure = fill(PS, 4, 3))
    q[2, 2, :]
end

@testset "file initial conditions: vertical order" begin
    mktempdir() do dir
        # Hybrid coefficients, the model's level count: surface first (as the
        # CATRINE start files) and top first give the same column.
        sfc = write_ic(joinpath(dir, "sfc.nc"); q = reverse(Q_TOP), ap = reverse(A_TOP), bp = reverse(B_TOP))
        top = write_ic(joinpath(dir, "top.nc"); q = Q_TOP, ap = A_TOP, bp = B_TOP)
        @test column(sfc) ≈ Q_TOP
        @test column(top) ≈ Q_TOP
        # Another level count: conservative means over eight finer sigma
        # layers are reproduced from top-first and surface-first files alike.
        # Fine interfaces include the target interfaces 0, .05, .30, .65, 1.
        bp8 = [0.0, 0.01, 0.05, 0.15, 0.3, 0.5, 0.65, 0.9, 1.0]             # top first
        q8 = 1e-6 .* collect(1.0:8.0)
        expected = 1e-6 .* [
            (1 * 0.01 + 2 * 0.04) / 0.05,
            (3 * 0.10 + 4 * 0.15) / 0.25,
            (5 * 0.20 + 6 * 0.15) / 0.35,
            (7 * 0.25 + 8 * 0.10) / 0.35,
        ]
        fine_top = write_ic(joinpath(dir, "fine_top.nc"); q = q8, ap = zeros(9), bp = bp8)
        fine_sfc = write_ic(joinpath(dir, "fine_sfc.nc"); q = reverse(q8), ap = zeros(9), bp = reverse(bp8))
        @test column(fine_top) ≈ expected rtol = 1e-12
        @test column(fine_sfc) ≈ expected rtol = 1e-12
        # The level coordinate's values do not matter with hybrid coefficients.
        rev = write_ic(joinpath(dir, "rev.nc"); q = reverse(Q_TOP), ap = reverse(A_TOP),
                       bp = reverse(B_TOP), lev = Float64.(NLEV - 1:-1:0))
        @test column(rev) ≈ Q_TOP
        @test column(sfc, "vertical_order" => "surface_first") ≈ Q_TOP
        err = try column(sfc, "vertical_order" => "toa_first"); nothing catch e; e end
        @test err isa ArgumentError && contains(err.msg, "contradicts")

        # No hybrid coefficients: the coordinate must say which end is the top.
        hpa = write_ic(joinpath(dir, "hpa.nc"); q = Q_TOP, lev = [10.0, 200.0, 600.0, 950.0],
                       lev_attrib = Dict("units" => "hPa"))
        @test column(hpa) ≈ Q_TOP
        height = write_ic(joinpath(dir, "height.nc"); q = reverse(Q_TOP),
                          lev = [100.0, 2000.0, 8000.0, 20000.0], lev_attrib = Dict("units" => "m"))
        @test column(height) ≈ Q_TOP
        up = write_ic(joinpath(dir, "up.nc"); q = reverse(Q_TOP), lev_attrib = Dict("positive" => "up"))
        @test column(up) ≈ Q_TOP
        plain = write_ic(joinpath(dir, "plain.nc"); q = reverse(Q_TOP))
        err = try column(plain); nothing catch e; e end
        @test err isa ArgumentError && contains(err.msg, "vertical_order")
        @test column(plain, "vertical_order" => "surface_first") ≈ Q_TOP
        @test_throws ArgumentError column(plain, "vertical_order" => "bottom_up")
        # ... and the model's level count (no zero-filled levels).
        short = write_ic(joinpath(dir, "short.nc"); q = Q_TOP[1:3], lev_attrib = Dict("units" => "hPa"),
                         lev = [10.0, 200.0, 600.0])
        err = try column(short); nothing catch e; e end
        @test err isa ArgumentError && contains(err.msg, "no hybrid coefficients")
    end
end
