#!/usr/bin/env julia

using AtmosTransport

"""
    generate_synthetic_quickstart([path]) -> String

Write a small version-4 lat-lon transport binary for the newcomer tutorial.
The forcing has constant eastward mass flux, zero mass divergence, four hourly
windows, and no dependency on external meteorological data.
"""
function generate_synthetic_quickstart(
        path::AbstractString = joinpath("data", "quickstart", "synthetic_latlon_v4.bin"))
    FT = Float64
    Nx, Ny, Nz = 36, 18, 3
    nwindow = 4

    mesh = LatLonMesh(; FT, Nx, Ny)
    vertical = HybridSigmaPressure(
        FT[0, 100, 300, 600],
        FT[0, 0, 0, 1],
    )
    grid = AtmosGrid(mesh, vertical, CPU(); FT)

    windows = [
        (; m = ones(FT, Nx, Ny, Nz),
           am = fill(FT(0.03), Nx + 1, Ny, Nz),
           bm = zeros(FT, Nx, Ny + 1, Nz),
           cm = zeros(FT, Nx, Ny, Nz + 1),
           ps = fill(FT(95_000), Nx, Ny),
           qv_start = zeros(FT, Nx, Ny, Nz),
           qv_end = zeros(FT, Nx, Ny, Nz),
           dam = zeros(FT, Nx + 1, Ny, Nz),
           dbm = zeros(FT, Nx, Ny + 1, Nz),
           dcm = zeros(FT, Nx, Ny, Nz + 1),
           dm = zeros(FT, Nx, Ny, Nz))
        for _ in 1:nwindow
    ]

    output_path = abspath(expanduser(path))
    mkpath(dirname(output_path))
    isfile(output_path) && rm(output_path)
    write_transport_binary(
        output_path,
        grid,
        windows;
        FT,
        dt_met_seconds = 3600.0,
        half_dt_seconds = 1800.0,
        steps_per_window = 2,
        mass_basis = :dry,
        surface_pressure_basis = DrySurfacePressure(),
        source_flux_sampling = :window_start_endpoint,
        flux_sampling = :window_constant,
        extra_header = Dict(
            "poisson_balance_target_scale" => 0.25,
            "poisson_balance_target_semantics" =>
                "forward_window_mass_difference / (2 * steps_per_window)",
        ),
    )
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    path = isempty(ARGS) ? generate_synthetic_quickstart() :
                          generate_synthetic_quickstart(only(ARGS))
    println("Wrote current transport binary: $path")
    println("Next: julia --project=. scripts/run_transport.jl config/examples/minimal_template.toml")
end
