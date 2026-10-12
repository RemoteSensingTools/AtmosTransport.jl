using Test
using NCDatasets
using JSON3
using Dates

using AtmosTransport
# AtmosTransport re-exports only a subset of Output's names (e.g.
# `runtime_output_spec`); the output-spec accessors used below
# (`output_enabled`, `output_field_spec`, `snapshot_hours`, `layer_selection`,
# `output_fields`, `output_path_for_day`, …) live in the Output submodule, so
# pull them in directly to keep this file self-sufficient.
using .AtmosTransport.Output

@testset "Output NetCDF snapshots" begin
    @testset "write options validate compression settings" begin
        @test SnapshotWriteOptions(float_type=Float32, deflate_level=1).deflate_level == 1
        @test_throws ArgumentError SnapshotWriteOptions(float_type=Float32, deflate_level=10)
        @test_throws ArgumentError SnapshotWriteOptions(float_type=Int32)
        @test_throws ArgumentError SnapshotWriteOptions(float_type=Float16)
    end

    @testset "runtime output spec parses cadence and partitions" begin
        mktempdir() do dir
            path = joinpath(dir, "run.nc")
            spec = runtime_output_spec(Dict{String, Any}(
                                           "path" => path,
                                           "cadence_hours" => 3,
                                           "stop_hour" => 12,
                                           "split" => "single",
                                           "deflate_level" => 1),
                                       Float32)
            @test output_enabled(spec)
            @test spec.partition isa SingleOutputFile
            @test output_path(spec) == path
            @test snapshot_hours(spec) == [0.0, 3.0, 6.0, 9.0, 12.0]
            @test spec.options.float_type === Float32
            @test spec.options.deflate_level == 1
            @test layer_selection(output_fields(spec).default_tracer) === :full

            daily = runtime_output_spec(Dict{String, Any}(
                                            "path" => joinpath(dir, "run_{date}.nc"),
                                            "hours" => [0, 6, 24],
                                            "split" => "daily"),
                                        Float64)
            @test output_enabled(daily)
            @test daily.partition isa DailyOutputFiles
            @test snapshot_hours(daily) == [0.0, 6.0, 24.0]
            @test output_path_for_day(daily, "20211202", 2) ==
                  joinpath(dir, "run_20211202.nc")

            suffixed = runtime_output_spec(Dict{String, Any}(
                                               "path" => joinpath(dir, "run.nc"),
                                               "snapshot_interval_hours" => 6,
                                               "stop_hour" => 6,
                                               "split" => "daily"),
                                           Float32)
            @test output_path_for_day(suffixed, "20211203", 3) ==
                  joinpath(dir, "run_20211203.nc")

            # On-disk snapshot precision is decoupled from the model compute type.
            # The ATMSNAP/binary writer is Float32-only on disk, so a Float64 run
            # with `format = "binary_mmap"` must coerce the on-disk dtype to
            # Float32 (the F64 precision lives in the in-run transport, not the
            # snapshot). NetCDF still honors the compute type.
            f64_binary = runtime_output_spec(Dict{String, Any}(
                                                 "path" => joinpath(dir, "run_{date}.atmsnap"),
                                                 "snapshot_interval_hours" => 6,
                                                 "stop_hour" => 6,
                                                 "split" => "daily",
                                                 "format" => "binary_mmap"),
                                             Float64)
            @test f64_binary.format isa AtmosTransport.Output.BinaryMmapOutput
            @test f64_binary.options.float_type === Float32
            f64_netcdf = runtime_output_spec(Dict{String, Any}(
                                                 "path" => joinpath(dir, "run.nc"),
                                                 "snapshot_interval_hours" => 6,
                                                 "stop_hour" => 6,
                                                 "split" => "single"),
                                             Float64)
            @test f64_netcdf.format isa AtmosTransport.Output.NetCDFOutput
            @test f64_netcdf.options.float_type === Float64

            disabled = runtime_output_spec(Dict{String, Any}("enabled" => false,
                                                             "path" => path,
                                                             "cadence_hours" => 1),
                                           Float32)
            @test !output_enabled(disabled)
            @test_throws ArgumentError runtime_output_spec(Dict{String, Any}(
                                                               "path" => path,
                                                               "cadence_hours" => 0),
                                                           Float32)

            fields = output_field_spec(Dict{String, Any}(
                "tracers" => ["co2"],
                "layers" => "selected",
                "levels" => [3, 1, 3],
                "column_mean" => true,
                "column_mass_per_area" => false,
                "air_mass_layers" => "none",
                "air_mass" => false,
                "air_mass_per_area" => false,
                "column_air_mass_per_area" => true,
                "per_tracer" => Dict("co2" => Dict("layers" => "full"))))
            @test fields.tracers == [:co2]
            @test fields.selected_levels == [1, 3]
            @test air_mass_layer_selection(fields) === :none
            @test layer_selection(fields.default_tracer) === :selected
            @test layer_selection(tracer_fields(fields, :co2)) === :full
            @test !tracer_fields(fields, :co2).column_mass_per_area
        end
    end

    @testset "writer rejects inconsistent snapshot contracts" begin
        mktempdir() do dir
            mesh = LatLonMesh(; FT=Float64, Nx=4, Ny=3)
            grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1, 2], Float64[0, 0.5, 1]),
                             CPU(); FT=Float64)
            air = fill(2.0, 4, 3, 2)
            frame = SnapshotFrame(0.0, air, Dict(:co2 => copy(air)), :dry)
            bad_basis = SnapshotFrame(1.0, air, Dict(:co2 => copy(air)), :moist)
            bad_shape = SnapshotFrame(1.0, air, Dict(:co2 => fill(1.0, 4, 3, 1)), :dry)

            @test_throws ArgumentError write_snapshot_netcdf(joinpath(dir, "empty.nc"),
                                                             SnapshotFrame[], grid)
            @test_throws ArgumentError write_snapshot_netcdf(joinpath(dir, "basis.nc"),
                                                             [frame, bad_basis], grid;
                                                             mass_basis=:dry)
            @test_throws DimensionMismatch write_snapshot_netcdf(joinpath(dir, "shape.nc"),
                                                                  [frame, bad_shape], grid;
                                                                  mass_basis=:dry)
        end
    end

    @testset "signed tracer totals survive Float32 output cancellation" begin
        mktempdir() do dir
            mesh = LatLonMesh(; FT=Float32, Nx=4, Ny=2)
            grid = AtmosGrid(mesh,
                             HybridSigmaPressure(Float32[0, 1], Float32[0, 1]),
                             CPU(); FT=Float32)
            air = ones(Float32, 4, 2, 1)
            signed = zeros(Float32, 4, 2, 1)
            signed[1, 1, 1] = 1f8
            signed[2, 1, 1] = 1f0
            signed[3, 1, 1] = -1f8

            frame = SnapshotFrame(0.0, air, Dict(:signed => signed), :dry)
            @test frame.tracer_total_mass[:signed] == 1.0
            @test sum(signed) == 0f0
            @test_throws ArgumentError SnapshotFrame(
                0.0, air, Dict(:signed => signed), :dry;
                tracer_total_mass = Dict(:other => 1.0))

            path = write_snapshot_netcdf(joinpath(dir, "signed.nc"), [frame], grid;
                                         mass_basis = :dry,
                                         options = SnapshotWriteOptions(float_type = Float32))
            NCDataset(path, "r") do ds
                total = ds["signed_total_mass"]
                @test eltype(total) === Float64
                @test total[:] == [1.0]
                @test total.attrib["units"] == "kg"
                @test total.attrib["storage_contract"] ==
                      "mixing_ratio_times_carrier_air_mass"
            end
        end
    end

    @testset "ATMSNAP header preserves authoritative signed totals" begin
        mktempdir() do dir
            mesh = CubedSphereMesh(; FT=Float32, Nc=2)
            grid = AtmosGrid(mesh,
                             HybridSigmaPressure(Float32[0, 1], Float32[0, 1]),
                             CPU(); FT=Float32)
            air = ntuple(_ -> ones(Float32, 2, 2, 1), 6)
            signed = ntuple(6) do panel
                values = zeros(Float32, 2, 2, 1)
                if panel == 1
                    values[1] = 1f8
                    values[2] = 1f0
                    values[3] = -1f8
                end
                values
            end
            frame = SnapshotFrame(0.0, air, Dict(:signed => signed), :dry)
            @test frame.tracer_total_mass[:signed] == 1.0

            path = write_snapshot_binary(joinpath(dir, "signed.atmsnap"),
                                         [frame], grid; mass_basis = :dry,
                                         time_origin = DateTime(2021, 12, 1),
                                         physics_attributes = Dict(
                                             "physics_advection_scheme" => "upwind"))
            header = open(path, "r") do io
                @test String(read(io, 8)) == "ATMSNAP1"
                header_size = read(io, UInt64)
                JSON3.read(String(read(io, Int(header_size))), Dict{String, Any})
            end
            @test header["tracer_total_mass_dtype"] == "Float64"
            @test header["tracer_total_mass_storage"] ==
                  "mixing_ratio_times_carrier_air_mass"
            @test Float64(header["tracer_total_mass"]["signed"][1]) == 1.0
            @test header["time_units"] == "hours since 2021-12-01 00:00:00"
            @test header["time_calendar"] == "proleptic_gregorian"
            @test header["physics_advection_scheme"] == "upwind"
        end
    end

    @testset "LL writer carries full fields and derived columns" begin
        mktempdir() do dir
            mesh = LatLonMesh(; FT=Float64, Nx=4, Ny=3)
            grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1, 2], Float64[0, 0.5, 1]),
                             CPU(); FT=Float64)
            air = fill(2.0, 4, 3, 2)
            rm = similar(air)
            rm[:, :, 1] .= 300e-6 .* air[:, :, 1]
            rm[:, :, 2] .= 600e-6 .* air[:, :, 2]
            frame = SnapshotFrame(0.0, air, Dict(:co2 => rm), :dry)
            path = write_snapshot_netcdf(joinpath(dir, "ll.nc"), [frame], grid;
                                         mass_basis=:dry,
                                         options=SnapshotWriteOptions(float_type=Float64,
                                                                      deflate_level=1),
                                         time_origin=DateTime(2021, 12, 1),
                                         physics_attributes=Dict(
                                             "physics_advection_scheme" => "upwind"))

            NCDataset(path, "r") do ds
                @test ds.attrib["output_contract"] == "AtmosTransport snapshot v2"
                @test ds.attrib["grid_type"] == "latlon"
                @test ds.attrib["physics_advection_scheme"] == "upwind"
                @test ds["time"].attrib["units"] == "hours since 2021-12-01 00:00:00"
                @test ds["time"].attrib["calendar"] == "proleptic_gregorian"
                @test haskey(ds, "air_mass")
                @test haskey(ds, "air_mass_per_area")
                @test haskey(ds, "column_air_mass_per_area")
                @test haskey(ds, "co2")
                @test haskey(ds, "co2_column_mean")
                @test haskey(ds, "co2_column_mass_per_area")
                @test NCDatasets.deflate(ds["co2"].var) == (true, true, 1)
                @test ds["co2_column_mean"][1, 1, 1] ≈ 450e-6
                @test ds["co2"][1, 1, 2, 1] ≈ 600e-6
            end
        end
    end

    @testset "writer honors tracer and layer field selection" begin
        mktempdir() do dir
            mesh = LatLonMesh(; FT=Float64, Nx=4, Ny=3)
            grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1, 2, 3], Float64[0, 0.3, 0.6, 1]),
                             CPU(); FT=Float64)
            air = fill(2.0, 4, 3, 3)
            co2 = similar(air)
            sf6 = similar(air)
            for k in 1:3
                co2[:, :, k] .= (300e-6 + 100e-6 * k) .* air[:, :, k]
                sf6[:, :, k] .= (5e-12 + 1e-12 * k) .* air[:, :, k]
            end
            frame = SnapshotFrame(0.0, air, Dict(:co2 => co2, :sf6 => sf6), :dry)
            fields = output_field_spec(Dict{String, Any}(
                "tracers" => ["co2"],
                "layers" => "selected",
                "levels" => [1, 3],
                "column_mean" => true,
                "column_mass_per_area" => false,
                "air_mass_layers" => "none",
                "air_mass" => false,
                "air_mass_per_area" => false,
                "column_air_mass_per_area" => true))
            path = write_snapshot_netcdf(joinpath(dir, "selected.nc"), [frame], grid;
                                         mass_basis=:dry,
                                         options=SnapshotWriteOptions(float_type=Float64),
                                         fields=fields)

            NCDataset(path, "r") do ds
                @test haskey(ds, "lev_selected")
                @test ds["lev_selected"][:] == [1.0, 3.0]
                @test !haskey(ds, "air_mass")
                @test !haskey(ds, "air_mass_per_area")
                @test haskey(ds, "column_air_mass_per_area")
                @test haskey(ds, "co2")
                @test size(ds["co2"]) == (4, 3, 2, 1)
                @test haskey(ds, "co2_column_mean")
                @test !haskey(ds, "co2_column_mass_per_area")
                @test !haskey(ds, "sf6")
                @test !haskey(ds, "sf6_column_mean")
                @test ds["co2"][1, 1, 1, 1] ≈ 400e-6
                @test ds["co2"][1, 1, 2, 1] ≈ 600e-6
            end
        end
    end

    @testset "RG writer preserves native cells plus legacy raster" begin
        mktempdir() do dir
            mesh = ReducedGaussianMesh([-45.0, 45.0], [4, 8]; FT=Float64)
            grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1, 2], Float64[0, 0.5, 1]),
                             CPU(); FT=Float64)
            air = fill(2.0, ncells(mesh), 2)
            rm = fill(400e-6, ncells(mesh), 2) .* air
            frame = SnapshotFrame(3.0, air, Dict(:co2 => rm), :dry)
            path = write_snapshot_netcdf(joinpath(dir, "rg.nc"), [frame], grid;
                                         mass_basis=:dry,
                                         options=SnapshotWriteOptions(float_type=Float64))

            NCDataset(path, "r") do ds
                @test ds.attrib["grid_type"] == "reduced_gaussian"
                @test haskey(ds, "cell_lon")
                @test haskey(ds, "cell_lat")
                @test haskey(ds, "cell_lon_bounds")
                @test haskey(ds, "cell_lat_bounds")
                @test ds["cell_lon"].attrib["bounds"] == "cell_lon_bounds"
                @test ds["cell_lat"].attrib["bounds"] == "cell_lat_bounds"
                @test haskey(ds, "co2_column_mean_native")
                @test haskey(ds, "co2_column_mean")
                @test size(ds["cell_lon_bounds"]) == (ncells(mesh), 4)
                @test size(ds["cell_lat_bounds"]) == (ncells(mesh), 4)
                @test ds["cell_lon_bounds"][1, :] ≈ [0.0, 90.0, 90.0, 0.0]
                @test ds["cell_lat_bounds"][1, :] ≈ [-90.0, -90.0, 0.0, 0.0]
                @test ds["cell_lon_bounds"][5, :] ≈ [0.0, 45.0, 45.0, 0.0]
                @test ds["cell_lat_bounds"][5, :] ≈ [0.0, 0.0, 90.0, 90.0]
                @test size(ds["co2_column_mean_native"]) == (ncells(mesh), 1)
                @test size(ds["co2_column_mean"]) == (maximum(mesh.nlon_per_ring), nrings(mesh), 1)
                @test all(isapprox.(ds["co2_column_mean_native"][:, 1], 400e-6; atol=0))
            end
        end
    end

    @testset "CS writer emits GEOS-native coordinates and panel diagnostics" begin
        mktempdir() do dir
            mesh = CubedSphereMesh(; FT=Float64, Nc=4, convention=GEOSNativePanelConvention())
            grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1, 2], Float64[0, 0.5, 1]),
                             CPU(); FT=Float64)
            air = ntuple(_ -> fill(2.0, mesh.Nc, mesh.Nc, 2), 6)
            rm = ntuple(_ -> begin
                a = Array{Float64}(undef, mesh.Nc, mesh.Nc, 2)
                a[:, :, 1] .= 300e-6 .* 2.0
                a[:, :, 2] .= 600e-6 .* 2.0
                a
            end, 6)
            frame = SnapshotFrame(6.0, air, Dict(:co2 => rm), :dry)
            path = write_snapshot_netcdf(joinpath(dir, "cs.nc"), [frame], grid;
                                         mass_basis=:dry,
                                         options=SnapshotWriteOptions(float_type=Float64))

            NCDataset(path, "r") do ds
                @test ds.attrib["grid_type"] == "cubed_sphere"
                @test ds.attrib["panel_convention"] == "geos_native"
                @test ds.attrib["longitude_of_central_meridian"] == -10.0
                @test haskey(ds, "lons")
                @test haskey(ds, "corner_lons")
                @test haskey(ds, "cubed_sphere")
                @test ds["cubed_sphere"].attrib["panel_convention"] == "geos_native"
                @test ds["corner_lons"][1, 1, 1] ≈ 305.0 atol=1e-10
                @test ds["corner_lons"][1, 1, 3] ≈ 35.0 atol=1e-10
                @test ds["corner_lats"][1, 1, 3] ≈ rad2deg(asin(1 / sqrt(3))) atol=1e-10
                @test ds["co2_column_mean"][1, 1, 1, 1] ≈ 450e-6
                @test ds["co2"][1, 1, 1, 2, 1] ≈ 600e-6
                @test ds["co2"].attrib["grid_mapping"] == "cubed_sphere"
                @test ds["co2"].attrib["coordinates"] == "lons lats"
            end
        end
    end

    @testset "driven runner delegates snapshot layout to Output" begin
        mktempdir() do dir
            mesh = LatLonMesh(; FT=Float64, Nx=2, Ny=2)
            grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1], Float64[0, 1]),
                             CPU(); FT=Float64)
            m = fill(2.0, 2, 2, 1)
            windows = [(
                m = m,
                am = zeros(Float64, 3, 2, 1),
                bm = zeros(Float64, 2, 3, 1),
                cm = zeros(Float64, 2, 2, 2),
                ps = fill(95000.0, 2, 2),
            )]
            bin = joinpath(dir, "transport.bin")
            write_transport_binary(bin, grid, windows;
                                   FT=Float64,
                                   dt_met_seconds=3600.0,
                                   half_dt_seconds=1800.0,
                                   steps_per_window=1,
                                   mass_basis=:dry,
                                   surface_pressure_basis=TotalSurfacePressure(),
                                   source_flux_sampling=:window_start_endpoint,
                                   flux_sampling=:window_constant)
            out = joinpath(dir, "snapshot.nc")
            cfg = Dict{String, Any}(
                "input" => Dict("binary_paths" => [bin], "start_date" => "2021-12-01"),
                "numerics" => Dict("float_type" => "Float64"),
                "run" => Dict("tracer_name" => "co2"),
                "init" => Dict("kind" => "uniform", "background" => 400e-6),
                "output" => Dict("path" => out,
                                  "hours" => [0.0, 1.0],
                                  "split" => "single",
                                  "deflate_level" => 1),
            )
            run_driven_simulation(cfg)
            NCDataset(out, "r") do ds
                @test ds.attrib["output_contract"] == "AtmosTransport snapshot v2"
                @test ds.attrib["physics_advection_scheme"] == "upwind"
                @test ds.attrib["physics_advection_limiter"] == "not_applicable"
                @test ds.attrib["physics_vertical_reconstruction"] == "upwind"
                @test ds.attrib["physics_convection"] == "none"
                @test ds.attrib["physics_diffusion"] == "none"
                @test ds.attrib["physics_surface_fluxes"] == "none"
                @test ds.attrib["physics_chemistry"] == "none"
                @test ds.attrib["physics_mass_basis"] == "dry"
                @test ds.attrib["physics_cadence"] == "window"
                @test ds.attrib["physics_air_mass_reset_mode"] == "preserve_tracer_mass"
                @test ds["time"].attrib["units"] == "hours since 2021-12-01 00:00:00"
                @test ds["time"].attrib["calendar"] == "proleptic_gregorian"
                @test haskey(ds, "co2")
                @test haskey(ds, "co2_column_mean")
                @test NCDatasets.deflate(ds["co2"].var) == (true, true, 1)
                @test size(ds["co2_column_mean"]) == (2, 2, 2)
                @test all(isapprox.(ds["co2_column_mean"][:, :, :], 400e-6; atol=1e-14))
            end

            daily_base = joinpath(dir, "daily.nc")
            daily_cfg = copy(cfg)
            daily_cfg["output"] = Dict("path" => daily_base,
                                       "hours" => [0.0, 1.0],
                                       "split" => "daily")
            run_driven_simulation(daily_cfg)
            daily_out = joinpath(dir, "daily_001.nc")
            @test isfile(daily_out)
            NCDataset(daily_out, "r") do ds
                @test size(ds["co2_column_mean"]) == (2, 2, 2)
                @test length(ds["time"][:]) == 2
            end
        end
    end
end
