"""
Core run-level conservation with decay, through catrine_true_mass_balance.jl.
The exact decay update matches its closed form; a two-binary synthetic run
closes both the stable-tracer and Rn-222 budgets against independent flux totals.
"""

using Test
using Dates
using Logging
using NCDatasets
using TOML
using AtmosTransport
using AtmosTransport: HybridSigmaPressure
using AtmosTransport.Parameters: DRY_AIR_MOLAR_MASS, RN222_HALF_LIFE,
                                 SPECIES_MOLAR_MASS, EARTH_RADIUS
const MD = AtmosTransport.MetDrivers

const SCRIPT = joinpath(@__DIR__, "..", "..", "scripts", "diagnostics", "catrine_true_mass_balance.jl")
const MB = Module()
Base.include(MB, SCRIPT)
mb_call(f::Symbol, args...; kw...) = Base.invokelatest(getproperty(MB, f), args...; kw...)

const RUN_NC, RUN_NZ, RUN_AIR = 3, 2, 1e16

function write_run_binary(path)
    vc = HybridSigmaPressure(fill(100.0, RUN_NZ + 1), collect(range(0.0, 1.0; length = RUN_NZ + 1)))
    writer = MD.open_streaming_cs_transport_binary(path, RUN_NC, 6, RUN_NZ, 8, vc; planet_radius = EARTH_RADIUS,
                                                   FT = Float64, dt_met_seconds = 10800.0,
                                                   steps_per_window = 2, mass_basis = :dry,
                                                   surface_pressure_basis = TotalSurfacePressure())
    try
        window = (; m = ntuple(_ -> fill(RUN_AIR, RUN_NC, RUN_NC, RUN_NZ), 6),
                  am = ntuple(_ -> zeros(RUN_NC + 1, RUN_NC, RUN_NZ), 6),
                  bm = ntuple(_ -> zeros(RUN_NC, RUN_NC + 1, RUN_NZ), 6),
                  cm = ntuple(_ -> zeros(RUN_NC, RUN_NC, RUN_NZ + 1), 6),
                  ps = ntuple(_ -> fill(1e5, RUN_NC, RUN_NC), 6))
        for _ in 1:8
            MD.write_streaming_cs_window!(writer, window, RUN_NC, 6)
        end
    finally
        MD.close_streaming_transport_binary!(writer)
    end
    return path
end

co2_density(i, j, p, day) = 1e-8 * (0.5 + 0.1i - 0.07j + 0.03p) * (day == 1 ? 1.0 : -0.5)
rn_density(i, j, p) = 2e-20 * (1 + 0.2i + 0.05j + 0.01p)

function write_run_flux(path, areas)
    NCDataset(path, "c") do ds
        defDim(ds, "Xdim", RUN_NC); defDim(ds, "Ydim", RUN_NC); defDim(ds, "nf", 6); defDim(ds, "time", 3)
        defVar(ds, "time", Float64, ("time",),
               attrib = Dict("units" => "hours since 2021-01-01 00:00:00",
                             "calendar" => "proleptic_gregorian"))[:] = [0.0, 24.0, 48.0]
        attrib = Dict("units" => "kg m-2 s-1")
        co2 = defVar(ds, "CO2_FLUX", Float64, ("Xdim", "Ydim", "nf", "time"); attrib)
        rn = defVar(ds, "RN_FLUX", Float64, ("Xdim", "Ydim", "nf", "time"); attrib)
        for (k, day) in enumerate((1, 2, 2))
            co2[:, :, :, k] = [co2_density(i, j, p, day) for i in 1:RUN_NC, j in 1:RUN_NC, p in 1:6]
            rn[:, :, :, k] = [rn_density(i, j, p) for i in 1:RUN_NC, j in 1:RUN_NC, p in 1:6]
        end
        defVar(ds, "cell_area", Float64, ("Xdim", "Ydim", "nf"))[:, :, :] = areas
    end
    return path
end

function run_config(dir, binaries, flux)
    surface(variable, species) = Dict{String, Any}("kind" => "cs_native", "file" => flux, "variable" => variable,
                                                   "species" => species, "time_varying" => true,
                                                   "temporal_scheme" => "stepwise")
    return Dict{String, Any}(
        "input" => Dict{String, Any}("binary_paths" => binaries, "start_date" => "2021-01-01"),
        "architecture" => Dict{String, Any}("use_gpu" => false),
        "numerics" => Dict{String, Any}("float_type" => "Float64"),
        "advection" => Dict{String, Any}("scheme" => "upwind"),
        "diffusion" => Dict{String, Any}("kind" => "none"),
        "convection" => Dict{String, Any}("kind" => "none"),
        "chemistry" => Dict{String, Any}("kind" => "decay",
                                         "half_lives_seconds" => Dict{String, Any}("rn222" => RN222_HALF_LIFE)),
        "tracers" => Dict{String, Any}(
            "co2" => Dict{String, Any}("init" => Dict{String, Any}("kind" => "uniform", "background" => 4e-4),
                                       "surface_flux" => surface("CO2_FLUX", "co2")),
            "rn222" => Dict{String, Any}("init" => Dict{String, Any}("kind" => "uniform", "background" => 0.0),
                                         "surface_flux" => surface("RN_FLUX", "rn222"))),
        "output" => Dict{String, Any}("format" => "netcdf", "split" => "daily",
                                      "snapshot_interval_hours" => 6,
                                      "path" => joinpath(dir, "output", "tiny_{YYYYMMDD}.nc")))
end

@testset "exact decay" begin
    λ, E, B0, Δ = log(2) / RN222_HALF_LIFE, 3.0e-8, 2.0e-3, 10800.0
    B = B0
    for _ in 1:40
        B = mb_call(:exact_decay, B, E, λ, Δ)
    end
    T = 40Δ
    @test B ≈ B0 * exp(-λ * T) + E / λ * (1 - exp(-λ * T)) rtol = 1e-13
    @test mb_call(:exact_decay, 2.0, 0.5, 0.0, 4.0) == 4.0
end

@testset "catrine_true_mass_balance.jl (atmostransport) on a tiny run" begin
    rdir = mktempdir()
    binaries = [write_run_binary(joinpath(rdir, "tiny_20210101.bin")),
                write_run_binary(joinpath(rdir, "tiny_20210102.bin"))]
    mesh = MD.driver_grid(MD.TransportBinaryDriver(binaries[1]; FT = Float64)).horizontal
    areas = repeat(Float64.(mesh.cell_areas), 1, 1, 6)
    cfg = run_config(rdir, binaries, write_run_flux(joinpath(rdir, "flux.nc"), areas))
    with_logger(NullLogger()) do
        redirect_stdout(devnull) do
            redirect_stderr(() -> AtmosTransport.run_driven_simulation(cfg), devnull)
        end
    end
    cfg_path = joinpath(rdir, "run.toml")
    open(io -> TOML.print(io, cfg), cfg_path, "w")
    out = joinpath(rdir, "balance.csv")
    io = IOBuffer()
    mb = with_logger(NullLogger()) do
        mb_call(:main, [cfg_path, joinpath(rdir, "output", "tiny_*.nc"), out]; io)
    end
    text = String(take!(io))
    lines = readlines(out)
    rows = [parse.(Float64, split(l, ",")) for l in lines[2:end]]
    hours = [r[1] for r in rows]
    @test hours == collect(6.0:6.0:48.0)                                     # from start_hours = 3; boundary frame once
    @test all(r -> isapprox(r[2], RUN_AIR * RUN_NC^2 * 6 * RUN_NZ; rtol = 1e-12), rows)

    # The source: Σ density × area × M_air / M over the stepwise daily slices.
    storage(density, M) = sum(density .* areas) * DRY_AIR_MOLAR_MASS / M
    co2_rate(day) = storage([co2_density(i, j, p, day) for i in 1:RUN_NC, j in 1:RUN_NC, p in 1:6], SPECIES_MOLAR_MASS.co2)
    rn_rate = storage([rn_density(i, j, p) for i in 1:RUN_NC, j in 1:RUN_NC, p in 1:6], SPECIES_MOLAR_MASS.rn222)
    for r in rows
        s1 = (min(r[1], 24.0) - 6.0) * 3600
        s2 = max(r[1] - 24.0, 0.0) * 3600
        @test r[4] ≈ co2_rate(1) * s1 + co2_rate(2) * s2 rtol = 1e-12
        @test r[7] ≈ rn_rate * (r[1] - 6.0) * 3600 rtol = 1e-12
        # Transport moves nothing, so the run conserves: r ≈ 0 for the stable tracer
        # and, with the model's symmetric decay/emission split, for Rn-222.
        @test abs(r[5]) <= 1e-12 * r[3]
        @test abs(r[8]) <= 1e-12 * max(r[6], rn_rate * 10800)
    end
    @test rows[1][[4, 5, 7, 8]] == zeros(4)
    # Midpoint emissions leave only the negative O((λT)²) Strang-splitting bias.
    m = match(r"rn222\s+splitting bias of the model's decay vs the exact solution: ([+-]\d\.\d+e[+-]\d+)", text)
    @test m !== nothing
    @test -1e-4 < parse(Float64, only(m.captures)) < 0
    @test mb.tracers == [:co2, :rn222]
end
