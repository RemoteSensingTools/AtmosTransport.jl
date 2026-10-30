#!/usr/bin/env julia

using Test
using JSON3

using AtmosTransport
using AtmosTransport.Architectures: CPU
using AtmosTransport.MetDrivers: PBLSurfaceForcing, load_window!
using AtmosTransport.Models: current_qv
using AtmosTransport.State: CubedSphereField
using AtmosTransport.Operators.Diffusion: WindowPBLKzField, LocalHoltslagBovilleKzField
using AtmosTransport.Operators.Advection: CSLinRoodAdvectionWorkspace
using AtmosTransport.Operators.Convection: TM5Workspace

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

include(joinpath(@__DIR__, "..", "fixtures", "driven_cs_binary.jl"))

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
        driver_written = TransportBinaryDriver(path; FT=Float64, arch=CPU(), Hp=1)
        @test AtmosTransport.MetDrivers.uses_binary_substep_contract(driver_written)
        close(driver_written)

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
        @test binary_mass_basis(reader) == :dry
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
            Dict("diffusion" => Dict("kind" => "tm5_beljaars_viterbo_local_kz")),
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
            Dict("diffusion" => Dict("kind" => "tm5_beljaars_viterbo_local_kz"),
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
