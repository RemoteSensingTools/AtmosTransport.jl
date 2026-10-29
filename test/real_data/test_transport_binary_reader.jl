#!/usr/bin/env julia

using Test
using JSON3

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.MetDrivers: TRANSPORT_BINARY_FORMAT_VERSION,
    source_flux_sampling, air_mass_sampling, flux_sampling, flux_kind,
    humidity_sampling, load_window!, load_qv_pair_window!,
    load_flux_delta_window!, load_surface_window!, interpolate_fluxes!,
    expected_air_mass!, interpolate_qv!, open_streaming_transport_binary,
    set_streaming_steps_per_window_schedule!, write_streaming_window!,
    close_streaming_transport_binary!, PBLSurfaceForcing

function rewrite_transport_header!(path::AbstractString;
                                   updates = Dict{String, Any}(),
                                   delete_keys = String[])
    open(path, "r+") do io
        raw = read(io, min(filesize(path), 262144))
        json_end = something(findfirst(==(0x00), raw), length(raw) + 1) - 1
        header = Dict{String, Any}(String(k) => v for (k, v) in
                                   pairs(JSON3.read(String(raw[1:json_end]))))
        for key in delete_keys
            delete!(header, key)
        end
        merge!(header, Dict{String, Any}(updates))
        header_bytes = Int(header["header_bytes"])
        header_json = JSON3.write(header)
        pad = header_bytes - ncodeunits(header_json)
        pad >= 0 || error("patched header does not fit")
        seek(io, 0)
        write(io, header_json)
        write(io, zeros(UInt8, pad))
    end
    return nothing
end

function write_test_transport_binary_reduced(path::AbstractString;
                                             FT::Type{<:AbstractFloat}=Float64,
                                             source_flux_sampling::Symbol=:window_start_endpoint,
                                             cm_fill::Union{Nothing, Real}=nothing,
                                             surface_pressure_basis=TotalSurfacePressure(),
                                             binary_kwargs...)
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)
    ncell = ncells(mesh)
    nface_h = nfaces(mesh)
    nlevel = nlevels(grid)

    windows = [
        begin
            m = reshape(FT.(1:(ncell * nlevel)) .* FT(win), ncell, nlevel)
            hflux = zeros(FT, nface_h, nlevel)
            cm = zeros(FT, ncell, nlevel + 1)
            cm_fill === nothing || fill!(cm, FT(cm_fill))
            ps = fill(FT(90000 + 1000win), ncell)
            qv_start = fill(FT(0.01win), ncell, nlevel)
            qv_end = fill(FT(0.01win + 0.001), ncell, nlevel)
            (; m, hflux, cm, ps, qv_start, qv_end)
        end for win in 1:2
    ]

    write_transport_binary(path, grid, windows;
                           FT=FT,
                           dt_met_seconds=3600.0,
                           half_dt_seconds=1800.0,
                           steps_per_window=2,
                           mass_basis=:moist,
                           surface_pressure_basis,
                           source_flux_sampling=source_flux_sampling,
                           extra_header=Dict(
                               "poisson_balance_target_scale" => 0.25,
                               "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
                           ),
                           binary_kwargs...)
    return grid
end

function write_test_transport_binary_latlon(path::AbstractString;
                                            FT::Type{<:AbstractFloat}=Float64,
                                            source_flux_sampling::Symbol=:window_start_endpoint,
                                            flux_sampling::Symbol=:window_constant,
                                            include_poisson_metadata::Bool=true,
                                            cm_fill::Union{Nothing, Real}=nothing,
                                            surface_pressure_basis=TotalSurfacePressure(),
                                            binary_kwargs...)
    Nx, Ny, Nz = 6, 4, 3
    mesh = LatLonMesh(; FT=FT, Nx=Nx, Ny=Ny)
    vertical = HybridSigmaPressure(FT[0, 100, 300, 1000], FT[0, 0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)

    windows = [
        begin
            m = reshape(FT.(1:(Nx * Ny * Nz)) .* FT(win), Nx, Ny, Nz)
            am = zeros(FT, Nx + 1, Ny, Nz)
            bm = zeros(FT, Nx, Ny + 1, Nz)
            cm = zeros(FT, Nx, Ny, Nz + 1)
            cm_fill === nothing || fill!(cm, FT(cm_fill))
            ps = fill(FT(95000 + 1000win), Nx, Ny)
            qv_start = fill(FT(0.002win), Nx, Ny, Nz)
            qv_end = fill(FT(0.002win + 0.0005), Nx, Ny, Nz)
            dam = fill(FT(0.1win), Nx + 1, Ny, Nz)
            dbm = fill(FT(0.2win), Nx, Ny + 1, Nz)
            dcm = fill(FT(0.3win), Nx, Ny, Nz + 1)
            dm = fill(FT(0.4win), Nx, Ny, Nz)
            (; m, am, bm, cm, ps, qv_start, qv_end, dam, dbm, dcm, dm)
        end for win in 1:2
    ]

    extra_header = include_poisson_metadata ? Dict(
        "poisson_balance_target_scale" => 0.25,
        "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
    ) : Dict{String, Any}()

    write_transport_binary(path, grid, windows;
                           FT=FT,
                           dt_met_seconds=3600.0,
                           half_dt_seconds=1800.0,
                           steps_per_window=2,
                           mass_basis=:moist,
                           surface_pressure_basis,
                           source_flux_sampling=source_flux_sampling,
                           flux_sampling=flux_sampling,
                           extra_header=extra_header,
                           binary_kwargs...)
    return grid
end

@testset "TransportBinaryReader structured lat-lon path" begin
    mktemp() do path, io
        close(io)
        grid_ref = write_test_transport_binary_latlon(path; FT=Float64)

        reader = TransportBinaryReader(path; FT=Float64)
        @test reader.header.format_version == TRANSPORT_BINARY_FORMAT_VERSION
        @test grid_type(reader) == :latlon
        @test horizontal_topology(reader) == :structureddirectional
        @test window_count(reader) == 2
        @test reader.header.steps_per_window_by_window == [2, 2]
        @test reader.header.poisson_balance_target_scale_by_window == [0.25, 0.25]
        @test binary_mass_basis(reader) == :moist
        @test has_qv_endpoints(reader)
        @test has_flux_delta(reader)
        @test source_flux_sampling(reader) == :window_start_endpoint
        @test air_mass_sampling(reader) == :window_start_endpoint
        @test flux_sampling(reader) == :window_constant
        @test flux_kind(reader) == :substep_mass_amount
        @test humidity_sampling(reader) == :window_endpoints
        @test delta_semantics(reader) == :forward_window_endpoint_difference
        @test reader.header.poisson_balance_target_scale == 0.25
        @test reader.header.poisson_balance_target_semantics == "forward_window_mass_difference / (2 * steps_per_window)"

        header_repr = sprint(show, reader.header)
        @test occursin("TransportBinaryHeader", header_repr)
        @test occursin("qv_start/qv_end", header_repr)
        @test occursin("substep_mass_amount", header_repr)

        reader_repr = sprint(show, reader)
        @test occursin("TransportBinaryReader", reader_repr)
        @test occursin("latlon/structureddirectional", reader_repr)

        grid = load_grid(reader; FT=Float64, arch=CPU())
        @test grid.horizontal isa LatLonMesh
        @test nx(grid.horizontal) == nx(grid_ref.horizontal)
        @test ny(grid.horizontal) == ny(grid_ref.horizontal)

        m, ps, fluxes = load_window!(reader, 1)
        @test size(m) == (6, 4, 3)
        @test size(ps) == (6, 4)
        @test size(fluxes.am) == (7, 4, 3)
        @test size(fluxes.bm) == (6, 5, 3)
        @test size(fluxes.cm) == (6, 4, 4)
        @test mass_basis(fluxes) isa MoistBasis

        qv_pair = load_qv_pair_window!(reader, 1)
        @test qv_pair !== nothing
        @test size(qv_pair.qv_start) == (6, 4, 3)
        @test size(qv_pair.qv_end) == (6, 4, 3)
        @test all(qv_pair.qv_start .== 0.002)
        @test all(qv_pair.qv_end .== 0.0025)

        deltas = load_flux_delta_window!(reader, 1)
        @test deltas !== nothing
        @test size(deltas.dam) == (7, 4, 3)
        @test size(deltas.dbm) == (6, 5, 3)
        @test size(deltas.dcm) == (6, 4, 4)
        @test size(deltas.dm) == (6, 4, 3)
        @test all(deltas.dam .== 0.1)
        @test all(deltas.dbm .== 0.2)
        @test all(deltas.dcm .== 0.3)
        @test all(deltas.dm .== 0.4)

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        @test total_windows(driver) == 2
        @test window_dt(driver) == 3600.0
        @test steps_per_window(driver) == 2
        @test air_mass_basis(driver) == :moist
        @test driver_grid(driver).horizontal isa LatLonMesh

        driver_repr = sprint(show, driver)
        @test occursin("TransportBinaryDriver", driver_repr)
        @test occursin("steps/window=2", driver_repr)

        window = load_transport_window(driver, 1)
        @test window isa TransportWindow{MoistBasis}
        @test has_humidity_endpoints(window)
        @test has_flux_delta(window)

        flux_interp = allocate_face_fluxes(driver_grid(driver).horizontal, 3; FT=Float64, basis=MoistBasis)
        interpolate_fluxes!(flux_interp, window, 0.5)
        @test all(flux_interp.am .== 0.05)
        @test all(flux_interp.bm .== 0.1)
        @test all(flux_interp.cm .== 0.15)

        m_interp = similar(window.air_mass)
        expected_air_mass!(m_interp, window, 0.5)
        @test all(m_interp .== window.air_mass .+ 0.2)

        qv_interp = similar(window.qv_start)
        interpolate_qv!(qv_interp, window, 0.5)
        @test all(isapprox.(qv_interp, 0.00225; atol=eps(Float64) * 10))

        close(driver)

        state = CellState(MoistBasis, copy(m); CO2=copy(m) .* 400e-6)
        model = TransportModel(state, fluxes, grid, UpwindScheme())
        sim = Simulation(model; Δt=1800.0, stop_time=3600.0)
        m0 = total_air_mass(state)
        rm0 = total_mass(state, :CO2)
        run!(sim)
        @test total_air_mass(sim.model.state) ≈ m0 atol=eps(Float64) * m0 * 10
        @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(Float64) * rm0 * 10
    end
end

@testset "Generic transport binaries reject CS-only adaptive runtime contract" begin
    mktemp() do path, io
        close(io)
        write_test_transport_binary_latlon(path; FT=Float64)
        rewrite_transport_header!(path; updates = Dict(
            "runtime_substep_contract" => "binary_schedule",
        ))
        @test_throws ArgumentError TransportBinaryDriver(path; FT=Float64, arch=CPU())
    end

    mktemp() do path, io
        close(io)
        write_test_transport_binary_reduced(path; FT=Float64)
        rewrite_transport_header!(path; updates = Dict(
            "runtime_substep_contract" => "binary_schedule",
        ))
        @test_throws ArgumentError TransportBinaryDriver(path; FT=Float64, arch=CPU())
    end
end

@testset "TransportBinaryReader reduced-Gaussian path" begin
    mktemp() do path, io
        close(io)
        grid_ref = write_test_transport_binary_reduced(path; FT=Float64)

        reader = TransportBinaryReader(path; FT=Float64)
        @test grid_type(reader) == :reduced_gaussian
        @test horizontal_topology(reader) == :faceindexed
        @test window_count(reader) == 2
        @test binary_mass_basis(reader) == :moist
        @test has_qv_endpoints(reader)
        @test !has_flux_delta(reader)
        @test source_flux_sampling(reader) == :window_start_endpoint
        @test delta_semantics(reader) == :none

        reduced_reader_repr = sprint(show, reader)
        @test occursin("reduced_gaussian/faceindexed", reduced_reader_repr)
        @test occursin("qv_start/qv_end", reduced_reader_repr)

        grid = load_grid(reader; FT=Float64, arch=CPU())
        @test grid.horizontal isa ReducedGaussianMesh
        @test ncells(grid.horizontal) == ncells(grid_ref.horizontal)
        @test nfaces(grid.horizontal) == reader.header.nface_h

        m, ps, fluxes = load_window!(reader, 1)
        @test size(m) == (8, 2)
        @test size(ps) == (8,)
        @test size(fluxes.horizontal_flux) == (reader.header.nface_h, 2)
        @test size(fluxes.cm) == (8, 3)
        @test mass_basis(fluxes) isa MoistBasis

        qv_pair = load_qv_pair_window!(reader, 1)
        @test qv_pair !== nothing
        @test all(qv_pair.qv_start .== 0.01)
        @test all(qv_pair.qv_end .== 0.011)

        state = CellState(MoistBasis, copy(m); CO2=copy(m) .* 400e-6)
        model = TransportModel(state, fluxes, grid, UpwindScheme())
        sim = Simulation(model; Δt=1800.0, stop_time=3600.0)
        m0 = total_air_mass(state)
        rm0 = total_mass(state, :CO2)
        run!(sim)
        @test total_air_mass(sim.model.state) ≈ m0 atol=eps(Float64) * m0 * 10
        @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(Float64) * rm0 * 10
    end
end

@testset "Reduced-Gaussian hflux does not imply PBL surface payload" begin
    mktemp() do path, io
        close(io)
        write_test_transport_binary_reduced(path; FT=Float64)

        reader = TransportBinaryReader(path; FT=Float64)
        try
            @test :hflux in reader.header.payload_sections
            @test !(:pbl_hflux in reader.header.payload_sections)
            @test !has_surface(reader)
            @test load_surface_window!(reader, 1) === nothing
        finally
            close(reader)
        end
    end
end

@testset "Reduced-Gaussian PBL surface heat flux uses pbl_hflux" begin
    FT = Float64
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)
    ncell = ncells(mesh)
    nface_h = nfaces(mesh)
    nlevel = nlevels(grid)

    horizontal_hflux = reshape(FT.(1:(nface_h * nlevel)), nface_h, nlevel)
    surface_hflux = fill(FT(80), ncell)
    surface = PBLSurfaceForcing(
        fill(FT(900), ncell),
        fill(FT(0.3), ncell),
        surface_hflux,
        fill(FT(290), ncell),
    )
    window = (
        m = ones(FT, ncell, nlevel),
        hflux = horizontal_hflux,
        cm = zeros(FT, ncell, nlevel + 1),
        ps = fill(FT(90_000), ncell),
        surface = surface,
    )

    mktemp() do path, io
        close(io)
        write_transport_binary(path, grid, [window];
                               FT=FT,
                               dt_met_seconds=3600.0,
                               half_dt_seconds=1800.0,
                               steps_per_window=2,
                               mass_basis=:dry,
                               surface_pressure_basis=DrySurfacePressure(),
                               source_flux_sampling=:window_start_endpoint)

        reader = TransportBinaryReader(path; FT=FT)
        try
            sections = reader.header.payload_sections
            @test count(==(:hflux), sections) == 1
            @test count(==(:pbl_hflux), sections) == 1
            @test has_surface(reader)

            _, _, fluxes = load_window!(reader, 1)
            @test fluxes.horizontal_flux == horizontal_hflux

            loaded_surface = load_surface_window!(reader, 1)
            @test loaded_surface !== nothing
            @test loaded_surface.hflux == surface_hflux
        finally
            close(reader)
        end
    end
end

@testset "Streaming transport binary carries per-window step schedule" begin
    mktemp() do path, io
        close(io)
        FT = Float64
        mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
        vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
        grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)
        ncell = ncells(mesh)
        nface_h = nfaces(mesh)
        nlevel = nlevels(grid)
        window = (
            m = fill(FT(1), ncell, nlevel),
            hflux = zeros(FT, nface_h, nlevel),
            cm = zeros(FT, ncell, nlevel + 1),
            ps = fill(FT(90000), ncell),
        )
        writer = open_streaming_transport_binary(
            path, grid, 2, window;
            FT = FT,
            dt_met_seconds = 3600.0,
            steps_per_window = 2,
            source_flux_sampling = :window_start_endpoint,
            mass_basis = :moist,
            surface_pressure_basis = TotalSurfacePressure())
        set_streaming_steps_per_window_schedule!(writer, [2, 3])
        write_streaming_window!(writer, window)
        write_streaming_window!(writer, window)
        close_streaming_transport_binary!(writer)

        reader = TransportBinaryReader(path; FT=FT)
        @test reader.header.steps_per_window == 3
        @test reader.header.steps_per_window_by_window == [2, 3]
        @test reader.header.poisson_balance_target_scale_by_window == [0.25, 1 / 6]
        @test reader.header.poisson_balance_target_semantics ==
              "forward_window_mass_difference / (2 * steps_per_window_by_window[win])"
        close(reader)

        driver = TransportBinaryDriver(path; FT=FT, arch=CPU())
        @test steps_per_window(driver) == 3
        @test steps_per_window(driver, 1) == 2
        @test steps_per_window(driver, 2) == 3
        @test steps_per_window_schedule(driver) == [2, 3]
        close(driver)
    end
end

@testset "Transport binary current contract rejects obsolete or incomplete headers" begin
    mktemp() do path, io
        close(io)
        write_test_transport_binary_latlon(path; FT=Float64)
        rewrite_transport_header!(path; updates = Dict("format_version" => 1))
        @test_throws ArgumentError TransportBinaryReader(path; FT=Float64)
    end

    mktemp() do path, io
        close(io)
        write_test_transport_binary_latlon(path; FT=Float64)
        rewrite_transport_header!(path; delete_keys = ["steps_per_window_by_window"])
        @test_throws ArgumentError TransportBinaryReader(path; FT=Float64)
    end

    mktemp() do path, io
        close(io)
        write_test_transport_binary_latlon(path; FT=Float64)
        rewrite_transport_header!(path; updates = Dict(
            "steps_per_window" => 2,
            "steps_per_window_by_window" => [2, 3],
            "time_step_schedule" => "per_window",
            "poisson_balance_target_scale" => 0.25,
            "poisson_balance_target_scale_by_window" => [0.25, 1 / 6],
            "poisson_balance_target_semantics" =>
                "forward_window_mass_difference / (2 * steps_per_window_by_window[win])",
        ))
        @test_throws ArgumentError TransportBinaryReader(path; FT=Float64)
    end
end

@testset "Transport binary writers reject contract-breaking header overrides" begin
    mktemp() do path, io
        close(io)
        FT = Float64
        mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
        vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
        grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)
        ncell = ncells(mesh)
        nface_h = nfaces(mesh)
        nlevel = nlevels(grid)
        window = (
            m = fill(FT(1), ncell, nlevel),
            hflux = zeros(FT, nface_h, nlevel),
            cm = zeros(FT, ncell, nlevel + 1),
            ps = fill(FT(90000), ncell),
        )
        @test_throws ArgumentError write_transport_binary(
            path, grid, [window];
            FT = FT,
            dt_met_seconds = 3600.0,
            half_dt_seconds = 1800.0,
            steps_per_window = 2,
            source_flux_sampling = :window_start_endpoint,
            mass_basis = :moist,
            surface_pressure_basis = TotalSurfacePressure(),
            extra_header = Dict("format_version" => 1))
    end
end

@testset "TransportBinaryDriver timing semantics validation" begin
    mktemp() do path, io
        close(io)
        @test_throws ArgumentError write_test_transport_binary_latlon(path; FT=Float64, flux_kind=:mass_rate)
    end
end

@testset "writer fills Poisson metadata by default" begin
    mktemp() do path, io
        close(io)
        write_test_transport_binary_latlon(path; FT=Float64, include_poisson_metadata=false)
        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        @test total_windows(driver) == 2
        @test window_dt(driver) == 3600.0
        @test driver.reader.header.poisson_balance_target_scale == 0.25
        @test driver.reader.header.poisson_balance_target_semantics ==
              "forward_window_mass_difference / (2 * steps_per_window)"
        close(driver)
    end
end

@testset "TransportBinaryDriver rejects binaries with oversized cm relative to cell mass" begin
    mktemp() do path, io
        close(io)
        write_test_transport_binary_latlon(path; FT=Float64, cm_fill=1.05)
        @test_throws ArgumentError TransportBinaryDriver(path; FT=Float64, arch=CPU())

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU(), validate_windows=false)
        @test total_windows(driver) == 2
        close(driver)
    end

    mktemp() do path, io
        close(io)
        write_test_transport_binary_reduced(path; FT=Float64, cm_fill=1.05)
        @test_throws ArgumentError TransportBinaryDriver(path; FT=Float64, arch=CPU())
    end
end
