#!/usr/bin/env julia

using Test
using JSON3

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Architectures: CPU

function rewrite_cs_header!(path::AbstractString; updates = Dict{String, Any}())
    open(path, "r+") do io
        raw = read(io, min(filesize(path), 262144))
        json_end = something(findfirst(==(0x00), raw), length(raw) + 1) - 1
        header = Dict{String, Any}(String(k) => v for (k, v) in
                                   pairs(JSON3.read(String(raw[1:json_end]))))
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

function write_driven_cs_binary(path::AbstractString;
                                FT::Type{<:AbstractFloat} = Float64,
                                Nc::Int = 4,
                                Nz::Int = 2,
                                window_mass_scales::Tuple{Vararg{Real}} = (1, 1),
                                window_dm_panels = nothing,
                                surface_windows = nothing,
                                vdiff_windows = nothing,
                                convection_windows = nothing,
                                dtrain_windows = nothing,
                                tm5_windows = nothing,
                                steps_per_window::Int = 2,
                                steps_per_window_by_window = nothing)
    window_dm_panels !== nothing &&
        length(window_dm_panels) == length(window_mass_scales) ||
        window_dm_panels === nothing || throw(ArgumentError("window_dm_panels length must match window_mass_scales"))
    surface_windows !== nothing &&
        length(surface_windows) == length(window_mass_scales) ||
        surface_windows === nothing || throw(ArgumentError("surface_windows length must match window_mass_scales"))
    vdiff_windows !== nothing &&
        length(vdiff_windows) == length(window_mass_scales) ||
        vdiff_windows === nothing || throw(ArgumentError("vdiff_windows length must match window_mass_scales"))
    convection_windows !== nothing &&
        length(convection_windows) == length(window_mass_scales) ||
        convection_windows === nothing || throw(ArgumentError("convection_windows length must match window_mass_scales"))
    dtrain_windows !== nothing &&
        length(dtrain_windows) == length(window_mass_scales) ||
        dtrain_windows === nothing || throw(ArgumentError("dtrain_windows length must match window_mass_scales"))
    tm5_windows !== nothing &&
        length(tm5_windows) == length(window_mass_scales) ||
        tm5_windows === nothing || throw(ArgumentError("tm5_windows length must match window_mass_scales"))
    vc = if Nz == 2
        AtmosTransport.HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    elseif Nz == 5
        AtmosTransport.HybridSigmaPressure(FT[0, 100, 300, 600, 1000, 2000],
                                           FT[0, 0, 0.1, 0.3, 0.7, 1])
    else
        AtmosTransport.HybridSigmaPressure(
            FT.(collect(range(0, stop = 2000, length = Nz + 1))),
            FT.(collect(range(0, stop = 1, length = Nz + 1))))
    end
    writer = AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
        path, Nc, 6, Nz, length(window_mass_scales), vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
        FT = FT,
        dt_met_seconds = 3600.0,
        steps_per_window = steps_per_window,
        include_flux_delta = window_dm_panels !== nothing,
        mass_basis = :dry,
        include_surface = surface_windows !== nothing,
        include_cmfmc = convection_windows !== nothing,
        include_dtrain = dtrain_windows !== nothing,
        include_tm5conv = tm5_windows !== nothing,
        include_gchp_vdiff = vdiff_windows !== nothing,
    )
    steps_per_window_by_window !== nothing &&
        AtmosTransport.MetDrivers.set_streaming_steps_per_window_schedule!(
            writer, steps_per_window_by_window)

    for (win, scale) in enumerate(window_mass_scales)
        window = (
            m  = ntuple(_ -> fill(FT(scale), Nc, Nc, Nz), 6),
            am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), 6),
            bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), 6),
            cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6),
            ps = ntuple(_ -> fill(FT(90000), Nc, Nc), 6),
        )
        if convection_windows !== nothing
            window = merge(window, (cmfmc = convection_windows[win],))
        end
        if surface_windows !== nothing
            window = merge(window, (surface = surface_windows[win],))
        end
        if vdiff_windows !== nothing
            window = merge(window, (vdiff = vdiff_windows[win],))
        end
        if dtrain_windows !== nothing
            window = merge(window, (dtrain = dtrain_windows[win],))
        end
        if tm5_windows !== nothing
            window = merge(window, (tm5_fields = tm5_windows[win],))
        end
        if window_dm_panels !== nothing
            window = merge(window, (dm = window_dm_panels[win],))
        end
        AtmosTransport.MetDrivers.write_streaming_cs_window!(writer, window, Nc, 6)
    end

    AtmosTransport.MetDrivers.close_streaming_transport_binary!(writer)
    return nothing
end

@testset "CubedSphere driven runtime honors per-window step schedule" begin
    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path; FT=Float64, Nc=4, Nz=2,
                               window_mass_scales=(1, 1),
                               steps_per_window=3,
                               steps_per_window_by_window=[2, 3])
        caps = inspect_binary(path; io=devnull)
        @test caps.variable_step_schedule === true
        @test caps.steps_per_window == 3

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU(), Hp=1)
        @test steps_per_window(driver) == 3
        @test steps_per_window(driver, 1) == 2
        @test steps_per_window(driver, 2) == 3
        @test steps_per_window_schedule(driver) == [2, 3]

        window = load_transport_window(driver, 1)
        mesh = driver_grid(driver).horizontal
        state = CubedSphereState(DryBasis, mesh, window.air_mass;
                                 CO2 = ntuple(p -> window.air_mass[p] .* 1e-6, 6))
        fluxes = allocate_face_fluxes(mesh, 2; FT=Float64, basis=DryBasis)
        model = TransportModel(state, fluxes, driver_grid(driver), UpwindScheme())
        sim = DrivenSimulation(model, driver; start_window=1, stop_window=2)

        @test sim.final_iteration == 5
        @test sim.steps_per_window == 2
        @test sim.Δt == 1800.0
        run_window!(sim)
        @test sim.iteration == 2
        @test window_index(sim) == 1

        step!(sim)
        @test window_index(sim) == 2
        @test sim.steps_per_window == 3
        @test sim.Δt == 1200.0
        @test sim.time == 4800.0

        run!(sim)
        @test sim.iteration == 5
        @test sim.time == 7200.0
        @test window_index(sim) == 2
        close(driver)
    end
end

@testset "CubedSphere reader rejects obsolete transport format version" begin
    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path; FT=Float64, Nc=4, Nz=2,
                               window_mass_scales=(1, 1),
                               steps_per_window=2)
        rewrite_cs_header!(path; updates = Dict("format_version" => 1))
        @test_throws ArgumentError TransportBinaryReader(path; FT=Float64)
    end
end

@testset "CubedSphere binary schedule can own runtime subcycling" begin
    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path; FT=Float64, Nc=4, Nz=2,
                               window_mass_scales=(1,),
                               steps_per_window=4)
        driver_legacy = TransportBinaryDriver(path; FT=Float64, arch=CPU(), Hp=1)
        @test !AtmosTransport.MetDrivers.uses_binary_substep_contract(driver_legacy)
        close(driver_legacy)

        rewrite_cs_header!(path; updates = Dict(
            "preprocessor_contract" => "plan41_variable_substeps",
            "adaptive_substeps" => true,
            "runtime_substep_contract" => "binary_schedule",
        ))
        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU(), Hp=1)
        @test AtmosTransport.MetDrivers.uses_binary_substep_contract(driver)
        close(driver)
    end
end

function make_cs_surface_panels(FT::Type{<:AbstractFloat}, Nc::Int;
                                pblh = FT(1000),
                                ustar = FT(0.35),
                                hflux = FT(120),
                                t2m = FT(295))
    return PBLSurfaceForcing(
        ntuple(_ -> fill(pblh,  Nc, Nc), 6),
        ntuple(_ -> fill(ustar, Nc, Nc), 6),
        ntuple(_ -> fill(hflux, Nc, Nc), 6),
        ntuple(_ -> fill(t2m,   Nc, Nc), 6),
    )
end

function make_cs_vdiff_panels(FT::Type{<:AbstractFloat}, Nc::Int, Nz::Int;
                              u0 = FT(5),
                              v0 = FT(-2),
                              t0 = FT(285),
                              qv0 = FT(0.004))
    return (
        u = ntuple(p -> [FT(u0 + 0.1p + 0.01k) for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6),
        v = ntuple(p -> [FT(v0 - 0.1p + 0.02k) for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6),
        t = ntuple(p -> [FT(t0 - 6k + 0.05p) for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6),
        qv = ntuple(p -> [max(FT(0), FT(qv0 / k)) for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6),
    )
end

function zero_cs_dm_panels(FT::Type{<:AbstractFloat}, Nc::Int, Nz::Int)
    return ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
end

function make_cs_cmfmc_panels(FT::Type{<:AbstractFloat}, Nc::Int, Nz::Int;
                              peak = FT(0.02),
                              top_detrain = FT(0.01))
    cmfmc = ntuple(6) do _
        arr = zeros(FT, Nc, Nc, Nz + 1)
        if Nz >= 5
            arr[:, :, 4] .= peak * FT(0.5)
            arr[:, :, 3] .= peak
            arr[:, :, 2] .= peak * FT(0.5)
        else
            arr[:, :, min(2, Nz)] .= peak
        end
        arr
    end
    dtrain = ntuple(6) do _
        arr = zeros(FT, Nc, Nc, Nz)
        arr[:, :, 1] .= top_detrain
        arr
    end
    return cmfmc, dtrain
end

function make_cs_tm5_panels(FT::Type{<:AbstractFloat}, Nc::Int, Nz::Int;
                            peak_entu = FT(0.02),
                            peak_detu = FT(0.01))
    entu = ntuple(6) do _
        arr = zeros(FT, Nc, Nc, Nz)
        arr[:, :, Nz] .= peak_entu * FT(0.3)
        if Nz >= 5
            arr[:, :, 4] .= peak_entu * FT(0.5)
            arr[:, :, 3] .= peak_entu
        else
            arr[:, :, max(1, Nz - 1)] .= peak_entu
        end
        arr
    end
    detu = ntuple(6) do _
        arr = zeros(FT, Nc, Nc, Nz)
        arr[:, :, max(1, Nz - 3)] .= peak_detu
        arr
    end
    entd = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    detd = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    return (entu = entu, detu = detu, entd = entd, detd = detd)
end

@testset "CubedSphere transport driver + DrivenSimulation" begin
    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path; FT=Float64, window_mass_scales=(1, 1))

        reader = TransportBinaryReader(path; FT=Float64)
        @test grid_type(reader) == :cubed_sphere
        @test horizontal_topology(reader) == :structureddirectional
        @test window_count(reader) == 2
        @test mass_basis(reader) == :dry
        @test !has_qv_endpoints(reader)
        @test !has_flux_delta(reader)
        @test !has_cmfmc(reader)
        @test !has_surface(reader)

        grid = load_grid(reader; FT=Float64, arch=CPU(), Hp=1)
        @test grid.horizontal isa CubedSphereMesh
        @test grid.horizontal.Nc == 4
        @test grid.horizontal.Hp == 1

        close(reader)

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU(), Hp=1)
        @test total_windows(driver) == 2
        @test window_dt(driver) == 3600.0
        @test steps_per_window(driver) == 2
        @test air_mass_basis(driver) == :dry

        window = load_transport_window(driver, 1)
        @test window isa TransportWindow{DryBasis}
        @test size(window.air_mass[1]) == (6, 6, 2)
        @test size(window.fluxes.am[1]) == (7, 6, 2)
        @test size(window.fluxes.bm[1]) == (6, 7, 2)
        @test size(window.fluxes.cm[1]) == (6, 6, 3)
        @test !has_humidity_endpoints(window)

        mesh = driver_grid(driver).horizontal
        tracer_panels = ntuple(p -> window.air_mass[p] .* 400e-6, 6)
        state = CubedSphereState(DryBasis, mesh, window.air_mass; CO2 = tracer_panels)
        fluxes = allocate_face_fluxes(mesh, 2; FT=Float64, basis=DryBasis)
        model = @inferred TransportModel(state, fluxes, driver_grid(driver), UpwindScheme())
        sim = DrivenSimulation(model, driver; start_window=1, stop_window=2)

        @test sim.interpolate_fluxes_within_window == false
        @test sim.Δt == 1800.0
        @test sim.steps_per_window == 2
        @test window_index(sim) == 1
        @test current_qv(sim) === nothing

        m0 = total_air_mass(sim.model.state)
        rm0 = total_mass(sim.model.state, :CO2)

        run!(sim)

        @test sim.iteration == 4
        @test sim.time == 7200.0
        @test window_index(sim) == 2
        @test total_air_mass(sim.model.state) ≈ m0 atol=eps(Float64) * m0 * 10
        @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(Float64) * rm0 * 10

        close(driver)
    end
end

@testset "CubedSphere binary carries raw PBL surface fields and drives pbl diffusion" begin
    FT = Float64
    Nc, Nz = 4, 5
    surface = make_cs_surface_panels(FT, Nc)

    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path;
                               FT = FT,
                               Nc = Nc,
                               Nz = Nz,
                               window_mass_scales = (FT(5e15),),
                               surface_windows = (surface,))

        reader = TransportBinaryReader(path; FT = FT)
        @test has_surface(reader)
        @test :pblh in reader.header.payload_sections
        @test :pbl_hflux in reader.header.payload_sections
        @test !(:hflux in reader.header.payload_sections)
        raw = load_window!(reader, 1)
        @test raw.surface isa PBLSurfaceForcing
        @test raw.surface.pblh[1][1, 1] == FT(1000)
        @test raw.surface.ustar[1][1, 1] == FT(0.35)
        close(reader)

        driver = TransportBinaryDriver(path; FT = FT, arch = CPU(), Hp = 1)
        recipe = build_runtime_physics_recipe(
            Dict("diffusion" => Dict("kind" => "pbl")),
            driver,
            FT,
        )
        @test recipe.diffusion isa ImplicitVerticalDiffusion
        @test recipe.diffusion.kz_field isa WindowPBLKzField

        window = load_transport_window(driver, 1)
        mesh = driver_grid(driver).horizontal
        tracer_panels = ntuple(6) do p
            rm = zeros(FT, size(window.air_mass[p]))
            @views rm[mesh.Hp + 1:mesh.Hp + mesh.Nc,
                      mesh.Hp + 1:mesh.Hp + mesh.Nc,
                      Nz] .= FT(1e-6) .* window.air_mass[p][mesh.Hp + 1:mesh.Hp + mesh.Nc,
                                                           mesh.Hp + 1:mesh.Hp + mesh.Nc,
                                                           Nz]
            rm
        end

        state = CubedSphereState(DryBasis, mesh, window.air_mass; CO2 = tracer_panels)
        fluxes = allocate_face_fluxes(mesh, Nz; FT = FT, basis = DryBasis)
        model = TransportModel(state, fluxes, driver_grid(driver), UpwindScheme();
                               diffusion = recipe.diffusion)
        sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 1)

        @test maximum(sim.model.diffusion.kz_field.host_cache[1]) > 0
        run!(sim)
        @test sim.iteration == 2
        @test all(isfinite, sim.model.diffusion.kz_field.host_cache[1])
        @test all(isfinite, get_tracer(sim.model.state, :CO2)[1])

        close(driver)
    end
end

@testset "CubedSphere GCHP VDIFF payload drives Holtslag-Boville local Kz" begin
    FT = Float64
    Nc, Nz = 4, 5
    surface = make_cs_surface_panels(FT, Nc)
    vdiff = make_cs_vdiff_panels(FT, Nc, Nz)

    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path;
                               FT = FT,
                               Nc = Nc,
                               Nz = Nz,
                               window_mass_scales = (FT(5e15),),
                               surface_windows = (surface,),
                               vdiff_windows = (vdiff,))

        reader = TransportBinaryReader(path; FT = FT)
        @test has_surface(reader)
        @test has_vdiff_fields(reader)
        raw = load_window!(reader, 1)
        @test raw.vdiff !== nothing
        @test raw.vdiff.t[2] == vdiff.t[2]
        close(reader)

        driver = TransportBinaryDriver(path; FT = FT, arch = CPU(), Hp = 1)
        recipe = build_runtime_physics_recipe(
            Dict("diffusion" => Dict("kind" => "geoschem_holtslag_boville_vdiff")),
            driver,
            FT,
        )
        @test recipe.diffusion isa ImplicitVerticalDiffusion
        @test recipe.diffusion.kz_field isa LocalHoltslagBovilleKzField

        window = load_transport_window(driver, 1)
        mesh = driver_grid(driver).horizontal
        tracer_panels = ntuple(6) do p
            rm = zeros(FT, size(window.air_mass[p]))
            @views rm[mesh.Hp + 1:mesh.Hp + mesh.Nc,
                      mesh.Hp + 1:mesh.Hp + mesh.Nc,
                      Nz] .= FT(1e-6) .* window.air_mass[p][mesh.Hp + 1:mesh.Hp + mesh.Nc,
                                                           mesh.Hp + 1:mesh.Hp + mesh.Nc,
                                                           Nz]
            rm
        end

        state = CubedSphereState(DryBasis, mesh, window.air_mass; CO2 = tracer_panels)
        fluxes = allocate_face_fluxes(mesh, Nz; FT = FT, basis = DryBasis)
        model = TransportModel(state, fluxes, driver_grid(driver), UpwindScheme();
                               diffusion = recipe.diffusion)
        sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 1)

        @test maximum(sim.model.diffusion.kz_field.host_cache[1]) > 0
        @test all(isfinite, sim.model.diffusion.kz_field.host_cache[1])
        run!(sim)
        @test sim.iteration == 2
        @test all(isfinite, get_tracer(sim.model.state, :CO2)[1])

        close(driver)
    end
end

@testset "CubedSphere optional payload sections round-trip panel/index sentinels" begin
    FT = Float64
    Nc, Nz = 3, 4
    pblh = ntuple(p -> [FT(1000p + 10i + j) for i in 1:Nc, j in 1:Nc], 6)
    ustar = ntuple(p -> [FT(0.1p + 0.01i + 0.001j) for i in 1:Nc, j in 1:Nc], 6)
    hflux = ntuple(p -> [FT(-50p + 2i - j) for i in 1:Nc, j in 1:Nc], 6)
    t2m = ntuple(p -> [FT(250 + p + 0.1i + 0.01j) for i in 1:Nc, j in 1:Nc], 6)
    surface = PBLSurfaceForcing(pblh, ustar, hflux, t2m)
    cmfmc = ntuple(p -> [FT(0.001p + 0.0001i + 0.00001j + 0.000001k)
                         for i in 1:Nc, j in 1:Nc, k in 1:(Nz + 1)], 6)
    dtrain = ntuple(p -> [FT(0.002p + 0.0002i + 0.00002j + 0.000002k)
                          for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6)

    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path;
                               FT = FT,
                               Nc = Nc,
                               Nz = Nz,
                               window_mass_scales = (FT(1e16),),
                               surface_windows = (surface,),
                               convection_windows = (cmfmc,),
                               dtrain_windows = (dtrain,))

        reader = TransportBinaryReader(path; FT = FT)
        raw = load_window!(reader, 1)
        @test has_surface(reader)
        @test has_cmfmc(reader)
        for p in 1:6
            @test raw.surface.pblh[p] == pblh[p]
            @test raw.surface.ustar[p] == ustar[p]
            @test raw.surface.hflux[p] == hflux[p]
            @test raw.surface.t2m[p] == t2m[p]
            @test raw.cmfmc[p] == cmfmc[p]
            @test raw.dtrain[p] == dtrain[p]
        end
        close(reader)
    end
end

@testset "CubedSphere multi-binary runner rejects partial-day carryover" begin
    mktempdir() do dir
        path1 = joinpath(dir, "day1.bin")
        path2 = joinpath(dir, "day2.bin")
        write_driven_cs_binary(path1; FT = Float64, Nc = 4, Nz = 2,
                               window_mass_scales = (1, 1))
        write_driven_cs_binary(path2; FT = Float64, Nc = 4, Nz = 2,
                               window_mass_scales = (1, 1))
        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path1, path2]),
            "architecture" => Dict("use_gpu" => false, "backend" => "cpu"),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("stop_window" => 1, "Hp" => 1),
            "advection" => Dict("scheme" => "upwind"),
            "tracers" => Dict("co2" => Dict("init" => Dict("kind" => "uniform",
                                                            "background" => 4e-4))),
            "output" => Dict("snapshot_hours" => Float64[]),
        )
        @test_throws ArgumentError run_driven_simulation(cfg)
    end
end

@testset "CubedSphere transport replay gate validates dm-bearing binaries" begin
    mktemp() do path, io
        close(io)
        dm_zero = zero_cs_dm_panels(Float64, 4, 2)
        write_driven_cs_binary(path; FT=Float64, Nc=4, Nz=2,
                               window_mass_scales=(1,),
                               window_dm_panels=(dm_zero,))

        reader = TransportBinaryReader(path; FT=Float64)
        @test has_flux_delta(reader)
        close(reader)

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU(), Hp=1,
                                            validate_replay=true)
        close(driver)
    end
end

@testset "CubedSphere transport replay gate catches final-window target mismatch" begin
    mktemp() do path, io
        close(io)
        dm_bad = ntuple(_ -> fill(1.0e4, 4, 4, 2), 6)
        write_driven_cs_binary(path; FT=Float64, Nc=4, Nz=2,
                               window_mass_scales=(1,),
                               window_dm_panels=(dm_bad,))

        @test_throws ArgumentError TransportBinaryDriver(
            path; FT=Float64, arch=CPU(), Hp=1, validate_replay=true)
    end
end

@testset "CubedSphere runtime supports CMFMC convection without promoting to Float64" begin
    FT = Float32
    Nc, Nz = 4, 5
    cmfmc, dtrain = make_cs_cmfmc_panels(FT, Nc, Nz)

    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path;
                               FT = FT,
                               Nc = Nc,
                               Nz = Nz,
                               window_mass_scales = (FT(1e16),),
                               convection_windows = (cmfmc,),
                               dtrain_windows = (dtrain,))

        reader = TransportBinaryReader(path; FT = FT)
        @test has_cmfmc(reader)
        close(reader)

        driver = TransportBinaryDriver(path; FT = FT, arch = CPU(), Hp = 1)
        window = load_transport_window(driver, 1)
        mesh = driver_grid(driver).horizontal
        @test window.convection !== nothing
        @test eltype(window.convection.cmfmc[1]) === FT

        tracer_panels = ntuple(6) do p
            rm = zeros(FT, size(window.air_mass[p]))
            @views rm[mesh.Hp + 1:mesh.Hp + mesh.Nc, mesh.Hp + 1:mesh.Hp + mesh.Nc, Nz] .=
                FT(1e-6) .* window.air_mass[p][mesh.Hp + 1:mesh.Hp + mesh.Nc, mesh.Hp + 1:mesh.Hp + mesh.Nc, Nz]
            rm
        end

        state = CubedSphereState(DryBasis, mesh, window.air_mass; CO2 = tracer_panels)
        fluxes = allocate_face_fluxes(mesh, Nz; FT = FT, basis = DryBasis)
        model = TransportModel(state, fluxes, driver_grid(driver), UpwindScheme();
                               convection = CMFMCConvection())
        sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 1)

        @test typeof(sim.Δt) === FT
        @test eltype(sim.model.convection_forcing.cmfmc[1]) === FT

        rm0 = total_mass(sim.model.state, :CO2)
        run!(sim)

        @test sim.iteration == 2
        @test total_mass(sim.model.state, :CO2) ≈ rm0 rtol = 1f-5
        for p in 1:6
            panel = get_tracer(sim.model.state, :CO2)[p]
            @test any(panel[mesh.Hp + 1:mesh.Hp + mesh.Nc, mesh.Hp + 1:mesh.Hp + mesh.Nc, 1] .> 0)
        end

        close(driver)
    end
end

@testset "CubedSphere full physics preserves uniform mixing ratio" begin
    FT = Float64
    Nc, Nz = 4, 5
    q0 = FT(4.0e-4)
    surface = make_cs_surface_panels(FT, Nc;
                                     pblh = FT(1200),
                                     ustar = FT(0.45),
                                     hflux = FT(180),
                                     t2m = FT(292))
    cmfmc, dtrain = make_cs_cmfmc_panels(FT, Nc, Nz;
                                         peak = FT(0.015),
                                         top_detrain = FT(0.006))

    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path;
                               FT = FT,
                               Nc = Nc,
                               Nz = Nz,
                               window_mass_scales = (FT(1e16),),
                               surface_windows = (surface,),
                               convection_windows = (cmfmc,),
                               dtrain_windows = (dtrain,))

        io_report = IOBuffer()
        caps = inspect_binary(path; io = io_report)
        report = String(take!(io_report))
        @test caps.grid_type === :cubed_sphere
        @test caps.pbl_diffusion === true
        @test caps.cmfmc_convection === true
        @test occursin("PBL diffusion", report)
        @test occursin("CMFMC convection", report)

        driver = TransportBinaryDriver(path; FT = FT, arch = CPU(), Hp = 1)
        recipe = build_runtime_physics_recipe(
            Dict("diffusion" => Dict("kind" => "pbl"),
                 "convection" => Dict("kind" => "cmfmc")),
            driver,
            FT,
        )
        window = load_transport_window(driver, 1)
        mesh = driver_grid(driver).horizontal

        tracer_panels = ntuple(p -> q0 .* window.air_mass[p], 6)
        state = CubedSphereState(DryBasis, mesh, window.air_mass; CO2 = tracer_panels)
        fluxes = allocate_face_fluxes(mesh, Nz; FT = FT, basis = DryBasis)
        model = TransportModel(state, fluxes, driver_grid(driver), UpwindScheme();
                               diffusion = recipe.diffusion,
                               convection = recipe.convection)
        sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 1)

        @test all(panel -> all(isfinite, panel), sim.model.diffusion.kz_field.host_cache)
        @test maximum(sim.model.diffusion.kz_field.host_cache[1]) > zero(FT)
        rm0 = total_mass(sim.model.state, :CO2)

        run!(sim)

        @test sim.iteration == 2
        @test total_mass(sim.model.state, :CO2) ≈ rm0 rtol = 1e-10
        for p in 1:6
            rm = get_tracer(sim.model.state, :CO2)[p]
            m = sim.model.state.air_mass[p]
            interior = @views rm[mesh.Hp + 1:mesh.Hp + mesh.Nc,
                                 mesh.Hp + 1:mesh.Hp + mesh.Nc,
                                 :] ./ m[mesh.Hp + 1:mesh.Hp + mesh.Nc,
                                        mesh.Hp + 1:mesh.Hp + mesh.Nc,
                                        :]
            @test maximum(abs.(interior .- q0)) < 1e-10
        end

        close(driver)
    end
end

@testset "Transport binary rejects partial raw PBL surface payloads" begin
    FT = Float64
    Nx, Ny, Nz = 3, 2, 2
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = AtmosTransport.HybridSigmaPressure(FT[0, 100, 1000], FT[0, 0.2, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
    window = (
        m = fill(FT(1e12), Nx, Ny, Nz),
        am = zeros(FT, Nx + 1, Ny, Nz),
        bm = zeros(FT, Nx, Ny + 1, Nz),
        cm = zeros(FT, Nx, Ny, Nz + 1),
        ps = fill(FT(90_000), Nx, Ny),
        pblh = fill(FT(900), Nx, Ny),
        ustar = fill(FT(0.3), Nx, Ny),
        hflux = fill(FT(100), Nx, Ny),
    )

    mktemp() do path, io
        close(io)
        @test_throws ArgumentError write_transport_binary(
            path, grid, [window];
            FT = FT,
            dt_met_seconds = 3600.0,
            half_dt_seconds = 1800.0,
            steps_per_window = 2,
            mass_basis = :dry,
            source_flux_sampling = :window_start_endpoint,
            flux_sampling = :window_constant,
        )
    end
end

@testset "CubedSphere runtime supports diffusion plus surface sources" begin
    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path; FT=Float64, window_mass_scales=(1,))

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU(), Hp=1)
        window = load_transport_window(driver, 1)
        mesh = driver_grid(driver).horizontal

        tracer_panels = ntuple(6) do p
            rm = zeros(Float64, size(window.air_mass[p]))
            @views rm[mesh.Hp + 1:mesh.Hp + mesh.Nc, mesh.Hp + 1:mesh.Hp + mesh.Nc, 2] .= 100.0
            rm
        end

        state = CubedSphereState(DryBasis, mesh, window.air_mass; CO2=tracer_panels)
        fluxes = allocate_face_fluxes(mesh, 2; FT=Float64, basis=DryBasis)
        kz = CubedSphereField(ntuple(_ -> ConstantField{Float64, 3}(1.0), 6))
        diffusion = ImplicitVerticalDiffusion(; kz_field=kz)
        model = TransportModel(state, fluxes, driver_grid(driver), UpwindScheme();
                               diffusion=diffusion)
        for p in 1:6
            fill!(model.workspace.diffusion_ws.layer_thickness[p], 100.0)
        end

        source = SurfaceFluxSource(:CO2, ntuple(_ -> fill(2.0, mesh.Nc, mesh.Nc), 6))
        sim = DrivenSimulation(model, driver;
                               start_window=1,
                               stop_window=1,
                               surface_sources=(source,))

        m0 = total_mass(sim.model.state, :CO2)
        run!(sim)

        @test sim.iteration == 2
        @test sim.time == 3600.0
        for p in 1:6
            panel = get_tracer(sim.model.state, :CO2)[p]
            @test all(panel[mesh.Hp + 1:mesh.Hp + mesh.Nc, mesh.Hp + 1:mesh.Hp + mesh.Nc, 1] .> 0.0)
        end
        @test total_mass(sim.model.state, :CO2) ≈ m0 + 6 * mesh.Nc * mesh.Nc * 2.0 * 1800.0 * 2 rtol=1e-7

        close(driver)
    end
end

@testset "CubedSphere diffusion preserves flat mixing ratio on nonuniform air mass" begin
    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path; FT=Float64, Nc=4, Nz=5, window_mass_scales=(1,))

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU(), Hp=1)
        window = load_transport_window(driver, 1)
        grid = driver_grid(driver)
        mesh = grid.horizontal
        Nz = 5

        air_mass = ntuple(6) do p
            m = copy(window.air_mass[p])
            for k in 1:Nz
                @views m[:, :, k] .= 1.0e12 * (1 + k)
            end
            m
        end
        tracer_panels = ntuple(p -> 4.0e-4 .* air_mass[p], 6)

        state = CubedSphereState(DryBasis, mesh, air_mass; CO2=tracer_panels)
        fluxes = allocate_face_fluxes(mesh, Nz; FT=Float64, basis=DryBasis)
        kz = CubedSphereField(ntuple(_ -> ConstantField{Float64, 3}(10.0), 6))
        diffusion = ImplicitVerticalDiffusion(; kz_field=kz)
        model = TransportModel(state, fluxes, grid, UpwindScheme(); diffusion=diffusion)
        for p in 1:6
            fill!(model.workspace.diffusion_ws.layer_thickness[p], 100.0)
        end

        step!(model, 600.0)

        for p in 1:6
            rm = get_tracer(model.state, :CO2)[p]
            m = model.state.air_mass[p]
            @views vmr = rm[mesh.Hp + 1:mesh.Hp + mesh.Nc,
                            mesh.Hp + 1:mesh.Hp + mesh.Nc, :] ./
                         m[mesh.Hp + 1:mesh.Hp + mesh.Nc,
                           mesh.Hp + 1:mesh.Hp + mesh.Nc, :]
            @test maximum(abs.(vmr .- 4.0e-4)) < 1e-12
        end

        close(driver)
    end
end

@testset "CubedSphere runtime supports Lin-Rood advection plus diffusion plus TM5 convection" begin
    FT = Float64
    Nc, Nz = 4, 5
    tm5 = make_cs_tm5_panels(FT, Nc, Nz)

    mktemp() do path, io
        close(io)
        write_driven_cs_binary(path;
                               FT = FT,
                               Nc = Nc,
                               Nz = Nz,
                               window_mass_scales = (FT(1e16),),
                               tm5_windows = (tm5,))

        reader = TransportBinaryReader(path; FT = FT)
        @test has_tm5_convection(reader)
        close(reader)

        driver = TransportBinaryDriver(path; FT = FT, arch = CPU(), Hp = 3)
        window = load_transport_window(driver, 1)
        mesh = driver_grid(driver).horizontal
        @test window.convection !== nothing
        @test window.convection.tm5_fields !== nothing
        @test eltype(window.convection.tm5_fields.entu[1]) === FT

        tracer_panels = ntuple(6) do p
            rm = zeros(FT, size(window.air_mass[p]))
            @views rm[mesh.Hp + 1:mesh.Hp + mesh.Nc,
                      mesh.Hp + 1:mesh.Hp + mesh.Nc,
                      Nz] .= FT(1e-6) .* window.air_mass[p][mesh.Hp + 1:mesh.Hp + mesh.Nc,
                                                           mesh.Hp + 1:mesh.Hp + mesh.Nc,
                                                           Nz]
            rm
        end

        state = CubedSphereState(DryBasis, mesh, window.air_mass; CO2 = tracer_panels)
        fluxes = allocate_face_fluxes(mesh, Nz; FT = FT, basis = DryBasis)
        kz = CubedSphereField(ntuple(_ -> ConstantField{FT, 3}(FT(1.0)), 6))
        diffusion = ImplicitVerticalDiffusion(; kz_field = kz)
        model = TransportModel(state, fluxes, driver_grid(driver), LinRoodPPMScheme(7);
                               diffusion = diffusion,
                               convection = TM5Convection())
        @test model.workspace.advection_ws isa CSLinRoodAdvectionWorkspace
        for p in 1:6
            fill!(model.workspace.diffusion_ws.layer_thickness[p], FT(100.0))
        end

        sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 1)
        @test sim.model.workspace.convection_ws isa TM5Workspace{FT}
        @test sim.model.convection_forcing.tm5_fields !== nothing

        rm0 = total_mass(sim.model.state, :CO2)
        run!(sim)

        @test sim.iteration == 2
        @test total_mass(sim.model.state, :CO2) ≈ rm0 rtol = 2e-4
        for p in 1:6
            panel = get_tracer(sim.model.state, :CO2)[p]
            @test any(panel[mesh.Hp + 1:mesh.Hp + mesh.Nc,
                            mesh.Hp + 1:mesh.Hp + mesh.Nc,
                            1] .> 0)
        end

        close(driver)
    end
end
