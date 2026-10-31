#!/usr/bin/env julia
# Binary capability and inspector contracts against a tiny lat-lon fixture.

using Test

using AtmosTransport
using .AtmosTransport.Parameters: GLOBAL_MEAN_DRY_SURFACE_PRESSURE

const _INSPECTOR_AIR_MASS = 1e16

function _inspector_fixture_binary(path::AbstractString;
                                   FT::Type{<:AbstractFloat} = Float64,
                                   with_tm5::Bool = false,
                                   with_surface::Bool = false,
                                   with_single_qv::Bool = false,
                                   with_humidity::Bool = false,
                                   mass_basis::Symbol = :dry,
                                   mass_fix_enabled = true,
                                   mass_fix_flags = nothing,
                                   recorded_target = GLOBAL_MEAN_DRY_SURFACE_PRESSURE,
                                   flux_kind::Symbol = :substep_mass_amount)
    Nx, Ny, Nz = 4, 3, 5
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(
        FT[0, 100, 300, 600, 1000, 2000],
        FT[0, 0, 0.1, 0.3, 0.7, 1],
    )
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    m  = fill(FT(_INSPECTOR_AIR_MASS), Nx, Ny, Nz)
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    ps = fill(FT(95_000), Nx, Ny)

    optional = if with_tm5
        tm5 = (entu = zeros(FT, Nx, Ny, Nz),
               detu = zeros(FT, Nx, Ny, Nz),
               entd = zeros(FT, Nx, Ny, Nz),
               detd = zeros(FT, Nx, Ny, Nz))
        (; tm5_fields = tm5)
    else
        NamedTuple()
    end
    if with_surface
        surface = AtmosTransport.MetDrivers.PBLSurfaceForcing(
            fill(FT(900), Nx, Ny), fill(FT(0.3), Nx, Ny),
            fill(FT(80), Nx, Ny), fill(FT(290), Nx, Ny))
        optional = merge(optional, (; surface))
    end
    with_single_qv &&
        (optional = merge(optional, (; qv = zeros(FT, Nx, Ny, Nz))))
    with_humidity && (optional = merge(optional, (
        qv_start = fill(FT(0.02), Nx, Ny, Nz),
        qv_end = fill(FT(0.02), Nx, Ny, Nz),
    )))
    window = merge((; m, am, bm, cm, ps), optional)

    mass_fix_header = Dict{String, Any}()
    if mass_fix_flags === nothing
        mass_fix_enabled === nothing ||
            (mass_fix_header["mass_fix_enabled"] = mass_fix_enabled)
    else
        merge!(mass_fix_header, mass_fix_flags)
    end
    recorded_target === nothing ||
        (mass_fix_header["mass_fix_target_ps_dry_pa"] = recorded_target)
    write_transport_binary(path, grid, [window];
                           FT = FT,
                           dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0,
                           steps_per_window = 1,
                           mass_basis,
                           surface_pressure_basis = TotalSurfacePressure(),
                           extra_header = mass_fix_header,
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant,
                           flux_kind)
    return nothing
end

@testset "binary_capabilities + inspect_binary" begin



    @testset "LL writer rejects unsupported full-window flux storage" begin
        @test_throws ArgumentError _inspector_fixture_binary(
            tempname(); flux_kind = :full_window_mass_amount)
    end

    @testset "LL writer rejects obsolete single-field humidity" begin
        @test_throws ArgumentError _inspector_fixture_binary(
            tempname(); with_single_qv = true)
    end

    @testset "basic LL fixture (no TM5)" begin
        mktempdir() do dir
            path = joinpath(dir, "fixture_base.bin")
            _inspector_fixture_binary(path; with_tm5 = false)

            reader = TransportBinaryReader(path; FT = Float64)
            caps = binary_capabilities(reader)
            @test caps.advection === true
            @test caps.tm5_convection === false
            @test caps.cmfmc_convection === false
            @test caps.surface_pressure === true
            @test caps.mass_basis === :dry
            @test caps.grid_type === :latlon
            @test caps.nwindow == 1 && caps.window_seconds == 3600.0
            @test binary_geometry(reader) isa LatLonBinaryGeometry
            @test caps.flux_kind === :substep_mass_amount
            @test caps.dry_mass_target_ps_pa == GLOBAL_MEAN_DRY_SURFACE_PRESSURE
            @test caps.dry_mass_target_source === :header
            @test caps.dry_mass_pinned === true
            @test caps.dry_mass_target_is_current
            @test :m in caps.payload_sections
            @test :am in caps.payload_sections
            close(reader)
        end
    end

    @testset "LL fixture with TM5 convection sections" begin
        mktempdir() do dir
            path = joinpath(dir, "fixture_tm5.bin")
            _inspector_fixture_binary(path; with_tm5 = true)

            reader = TransportBinaryReader(path; FT = Float64)
            caps = binary_capabilities(reader)
            @test caps.tm5_convection === true
            @test :entu in caps.payload_sections
            @test :detu in caps.payload_sections
            @test :entd in caps.payload_sections
            @test :detd in caps.payload_sections
            close(reader)
        end
    end

    @testset "LL surface payload does not advertise unsupported diffusion" begin
        mktempdir() do dir
            path = joinpath(dir, "fixture_surface.bin")
            _inspector_fixture_binary(path; with_surface = true)

            reader = TransportBinaryReader(path; FT = Float64)
            @test has_surface(reader)
            caps = binary_capabilities(reader)
            @test caps.pbl_diffusion === false
            @test caps.gchp_vdiff === false
            driver = TransportBinaryDriver(reader)
            @test AtmosTransport.Models.Physics._runtime_has_surface(driver) === false
            close(driver)
        end
    end


    @testset "pinned, unpinned, mixed, and legacy target classification" begin
        mktempdir() do dir
            pinned_path = joinpath(dir, "pinned.bin")
            unpinned_path = joinpath(dir, "unpinned.bin")
            legacy_path = joinpath(dir, "legacy.bin")
            legacy_pinned_path = joinpath(dir, "legacy_pinned.bin")
            equal_pinned_path = joinpath(dir, "equal_pinned.bin")
            other_pinned_path = joinpath(dir, "other_pinned.bin")
            _inspector_fixture_binary(pinned_path)
            # A stale target must not classify an explicitly unpinned file.
            _inspector_fixture_binary(unpinned_path; mass_fix_enabled = false,
                                      recorded_target = GLOBAL_MEAN_DRY_SURFACE_PRESSURE)
            _inspector_fixture_binary(legacy_path; mass_fix_enabled = nothing,
                                      recorded_target = nothing)
            _inspector_fixture_binary(legacy_pinned_path;
                mass_fix_enabled = nothing, recorded_target = nothing,
                mass_fix_flags = Dict("global_mass_pin_enabled" => true))

            pinned = inspect_binary(pinned_path; io = devnull)
            unpinned = inspect_binary(unpinned_path; io = devnull)
            legacy = inspect_binary(legacy_path; io = devnull)
            legacy_pinned = inspect_binary(legacy_pinned_path; io = devnull)
            carried = _INSPECTOR_AIR_MASS * 4 * 3 * 5 *
                      AtmosTransport.Parameters.STANDARD_GRAVITY /
                      (4π * AtmosTransport.Parameters.EARTH_RADIUS^2)
            _inspector_fixture_binary(equal_pinned_path; recorded_target = carried)
            _inspector_fixture_binary(other_pinned_path; recorded_target = carried + 1)
            equal_pinned = inspect_binary(equal_pinned_path; io = devnull)
            other_pinned = inspect_binary(other_pinned_path; io = devnull)
            @test pinned.dry_mass_target_source === :header
            @test pinned.dry_mass_pinned === true
            @test unpinned.dry_mass_target_source === :unpinned_window1
            @test unpinned.dry_mass_pinned === false
            @test unpinned.dry_mass_target_ps_pa ≈ carried
            @test legacy.dry_mass_target_source === :legacy_window1
            @test legacy.dry_mass_pinned === nothing
            @test legacy.dry_mass_target_ps_pa == unpinned.dry_mass_target_ps_pa
            @test legacy_pinned.dry_mass_target_source === :legacy_window1
            @test legacy_pinned.dry_mass_pinned === true
            @test legacy_pinned.dry_mass_target_ps_pa ≈ carried

            item(path, caps) = (; path, caps)
            check(xs...) = AtmosTransport.Models.DrivenRunner._check_run_dry_mass_targets(
                collect(xs))
            drifting_unpinned = merge(unpinned,
                (dry_mass_target_ps_pa = unpinned.dry_mass_target_ps_pa + 2.0,))
            @test check(item(unpinned_path, unpinned),
                        item("next_unpinned.bin", drifting_unpinned)) === nothing
            @test check(item(unpinned_path, unpinned), item(legacy_path, legacy)) ≈ carried
            @test check(item(pinned_path, pinned), item(pinned_path, pinned)) ==
                  GLOBAL_MEAN_DRY_SURFACE_PRESSURE
            err = try
                check(item(pinned_path, pinned), item(unpinned_path, unpinned))
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("cannot mix explicitly pinned and unpinned", sprint(showerror, err))

            # Equal window-1 mass must not hide conflicting explicit states.
            equal_err = try
                check(item(equal_pinned_path, equal_pinned), item(unpinned_path, unpinned))
            catch e
                e
            end
            @test equal_pinned.dry_mass_target_ps_pa == unpinned.dry_mass_target_ps_pa
            @test equal_err isa ArgumentError
            @test occursin("cannot mix explicitly pinned and unpinned",
                           sprint(showerror, equal_err))

            pressure_err = try
                check(item(equal_pinned_path, equal_pinned),
                      item(other_pinned_path, other_pinned))
            catch e
                e
            end
            @test pressure_err isa ArgumentError
            @test occursin("must share one dry-mass target", sprint(showerror, pressure_err))
            @test occursin("GLOBAL_MEAN_DRY_SURFACE_PRESSURE=98325.0 Pa",
                           sprint(showerror, pressure_err))

            check_basis(xs...) = AtmosTransport.Models.DrivenRunner._check_run_surface_pressure_bases(
                collect(xs))
            dry_ps_caps = merge(pinned, (surface_pressure_basis=DrySurfacePressure(),))
            total_ps_caps = merge(pinned, (surface_pressure_basis=TotalSurfacePressure(),))
            @test check_basis(item("dry.bin", dry_ps_caps),
                              item("dry2.bin", dry_ps_caps)) isa DrySurfacePressure
            @test_throws ArgumentError check_basis(item("dry.bin", dry_ps_caps),
                                                    item("total.bin", total_ps_caps))

            # Warm compilation before measuring. The fallback may allocate a
            # tiny SubArray descriptor, but never any payload-sized arrays.
            reader = TransportBinaryReader(legacy_path; FT = Float64)
            AtmosTransport.MetDrivers.binary_dry_mass_target(reader)
            bytes = @allocated AtmosTransport.MetDrivers.binary_dry_mass_target(reader)
            @test bytes < 4096
            close(reader)

            moist_qv_path = joinpath(dir, "moist_qv.bin")
            moist_unknown_path = joinpath(dir, "moist_unknown.bin")
            _inspector_fixture_binary(moist_qv_path; mass_basis = :moist,
                with_humidity = true, mass_fix_enabled = false,
                recorded_target = nothing)
            _inspector_fixture_binary(moist_unknown_path; mass_basis = :moist,
                mass_fix_enabled = false, recorded_target = nothing)
            moist_qv = inspect_binary(moist_qv_path; io = devnull)
            moist_unknown = inspect_binary(moist_unknown_path; io = devnull)
            @test moist_qv.dry_mass_target_ps_pa ≈ carried * 0.98
            @test moist_qv.dry_mass_target_source === :unpinned_window1
            @test moist_unknown.dry_mass_target_ps_pa === nothing
            @test moist_unknown.dry_mass_target_source === :unclassifiable
        end
    end

    @testset "all dry-mass pin aliases are validated together" begin
        mktempdir() do dir
            keys = ("mass_fix_enabled", "global_mass_pin_enabled",
                    "geos_global_mass_pin_enabled")
            states = (nothing, false, true)
            case_id = 0
            for values in Iterators.product(states, states, states)
                case_id += 1
                flags = Dict{String, Any}(keys[i] => values[i]
                                          for i in eachindex(keys)
                                          if values[i] !== nothing)
                path = joinpath(dir, "truth_$(case_id).bin")
                _inspector_fixture_binary(path; mass_fix_enabled = nothing,
                                          mass_fix_flags = flags)
                present = filter(!isnothing, collect(values))
                conflicting = !isempty(present) && any(!=(first(present)), present)
                if conflicting
                    err = try
                        inspect_binary(path; io = devnull)
                    catch e
                        e
                    end
                    @test err isa ArgumentError
                    @test occursin("conflicting dry-mass pin flags", sprint(showerror, err))
                else
                    caps = inspect_binary(path; io = devnull)
                    expected = isempty(present) ? nothing : first(present)
                    @test caps.dry_mass_pinned === expected
                    @test caps.dry_mass_target_source ===
                          (expected === true ? :header :
                           expected === false ? :unpinned_window1 : :legacy_window1)
                end
            end

            for (idx, malformed_key) in enumerate(keys)
                flags = Dict{String, Any}("mass_fix_enabled" => false,
                                          malformed_key => "false")
                path = joinpath(dir, "malformed_$(idx).bin")
                _inspector_fixture_binary(path; mass_fix_enabled = nothing,
                                          mass_fix_flags = flags)
                err = try
                    inspect_binary(path; io = devnull)
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("non-Boolean $(malformed_key)", sprint(showerror, err))
            end

            bool_target = joinpath(dir, "bool_target.bin")
            _inspector_fixture_binary(bool_target; recorded_target = true)
            err = try
                inspect_binary(bool_target; io = devnull)
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("non-numeric mass_fix_target_ps_dry_pa",
                           sprint(showerror, err))
        end
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
            surface_pressure_basis = TotalSurfacePressure(),
            source_flux_sampling = :window_start_endpoint,
            flux_sampling = :window_constant,
        )
    end
end
