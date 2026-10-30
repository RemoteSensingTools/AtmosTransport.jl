#!/usr/bin/env julia
# Which routes may put TM5 convection into a transport binary.
#
# The lat-lon ERA5 physics-file route (`[tm5_convection]` in an ERA5 spectral
# preprocessing config) placed the TM5 fields 180° off in longitude and one
# hour early (issue #55); it was removed. A config that still asks for it must
# fail before any input is read and name the supported routes.

using Test
using TOML
using Logging
using AtmosTransport

const Pre = AtmosTransport.Preprocessing
const PREPROCESSING_CONFIGS = joinpath(pkgdir(AtmosTransport), "config", "preprocessing")

function _config_error(cfg)
    return try
        with_logger(NullLogger()) do
            Pre.validate_preprocessing_config(cfg)
        end
        nothing
    catch err
        err
    end
end

@testset "retired lat-lon TM5 physics-file route is rejected" begin
    spectral = TOML.parsefile(joinpath(PREPROCESSING_CONFIGS, "era5_ll72x37_advresln_dec2021_f32.toml"))
    # The unmodified configs validate.
    @test _config_error(spectral) === nothing

    for base in (spectral,), section in (
            Dict{String, Any}("enable" => true, "physics_bin_dir" => "/nonexistent/physics_bin"),
            Dict{String, Any}("enable" => false))
        cfg = deepcopy(base)
        cfg["tm5_convection"] = section
        err = _config_error(cfg)
        @test err isa ArgumentError
    end
end

# ERA5-derived TM5 sections record the ec2tm convention. Before the fix of
# issue #54 the conversion placed ERA5 UDMF/DDMF level k at the bottom of
# layer k and dropped the cloud-top layer; such binaries, and those of the
# retired lat-lon route, are rejected on load with a regenerate message.
# Sections attached from the TM5-converted CATRINE archive (TM5's own
# ECconv_to_TMconv) stay accepted, and so do sections without any recorded
# source (synthetic or externally written binaries).
const MD = AtmosTransport.MetDrivers
const P = AtmosTransport.Parameters
const VC = HybridSigmaPressure([0.0, 1000, 2000], [0.0, 0.2, 1.0])

function write_cs_tm5_fixture(path, extra_header)
    Nc, np, Nz = 2, 6, 2
    panels3() = ntuple(_ -> zeros(Float64, Nc, Nc, Nz), np)
    window = (m = ntuple(_ -> fill(1e15, Nc, Nc, Nz), np),
              am = ntuple(_ -> zeros(Nc + 1, Nc, Nz), np),
              bm = ntuple(_ -> zeros(Nc, Nc + 1, Nz), np),
              cm = ntuple(_ -> zeros(Nc, Nc, Nz + 1), np),
              ps = ntuple(_ -> fill(95_000.0, Nc, Nc), np),
              tm5_fields = (entu = panels3(), detu = panels3(), entd = panels3(), detd = panels3()))
    writer = MD.open_streaming_cs_transport_binary(
        path, Nc, np, Nz, 1, VC; planet_radius = P.EARTH_RADIUS,
        steps_per_window = 1, include_tm5conv = true, mass_basis = :dry,
        surface_pressure_basis = DrySurfacePressure(),
        extra_header = Dict{String, Any}(extra_header))
    MD.write_streaming_cs_window!(writer, window, Nc, np)
    MD.close_streaming_transport_binary!(writer)
    return path
end

function write_ll_tm5_fixture(path, extra_header)
    Nx, Ny, Nz = 4, 3, 2
    grid = AtmosGrid(LatLonMesh(; FT = Float64, Nx, Ny), VC, CPU(); FT = Float64)
    tm5() = zeros(Nx, Ny, Nz)
    window = (m = fill(1e16, Nx, Ny, Nz), am = zeros(Nx + 1, Ny, Nz),
              bm = zeros(Nx, Ny + 1, Nz), cm = zeros(Nx, Ny, Nz + 1),
              ps = fill(95_000.0, Nx, Ny),
              tm5_fields = (entu = tm5(), detu = tm5(), entd = tm5(), detd = tm5()))
    write_transport_binary(path, grid, [window]; FT = Float64, steps_per_window = 1,
        mass_basis = :dry, surface_pressure_basis = DrySurfacePressure(),
        source_flux_sampling = :window_start_endpoint,
        extra_header = Dict{String, Any}(extra_header))
    return path
end

function open_error(path)
    return try
        close(TransportBinaryReader(path; FT = Float64))
        nothing
    catch err
        err
    end
end

@testset "TM5 convection provenance is checked on load" begin
    # The on-disk names are pinned here as literals.
    stamp = "tm5_ec2tm_convention" => "tm5_halflevel_v2"
    ec2tm_source = "tm5_convection_source" => "ec2tm_from_rates(udmf,ddmf,udrf,ddrf)"
    mktempdir() do dir
        accepted = (
            "current ec2tm"     => (write_cs_tm5_fixture, [ec2tm_source, stamp]),
            "CATRINE attach v3" => (write_cs_tm5_fixture,
                                    ["convection_attachment" => "legacy_catrine_1deg_3hour_tm5_to_cs_v3"]),
            "CATRINE attach v4" => (write_cs_tm5_fixture,
                                    ["convection_attachment" => "legacy_catrine_1deg_3hour_tm5_to_cs_v4"]),
            "CATRINE attach, stale source key" =>
                (write_cs_tm5_fixture, [ec2tm_source,
                                        "convection_attachment" => "legacy_catrine_1deg_3hour_tm5_to_cs_v4"]),
            "no recorded source" => (write_cs_tm5_fixture, Pair{String, Any}[]),
            "synthetic LL"      => (write_ll_tm5_fixture, Pair{String, Any}[]),
            "LL→CS regrid, current source" =>
                (write_cs_tm5_fixture, ["preprocessor" => "regrid_ll_binary_to_cs",
                                        "tm5_convection_source" => "unrecorded"]),
        )
        rejected = (
            "pre-#54 ec2tm"     => (write_cs_tm5_fixture, [ec2tm_source], r"regenerate"i),
            "unknown convention" => (write_cs_tm5_fixture,
                                     [ec2tm_source, "tm5_ec2tm_convention" => "tm5_halflevel_v1"],
                                     r"regenerate"i),
            "retired LL route"  => (write_ll_tm5_fixture, ["spectral_half_dt_seconds" => 450.0],
                                    r"#55"),
            "pre-fix LL→CS regrid" => (write_cs_tm5_fixture,
                                       ["preprocessor" => "regrid_ll_binary_to_cs"], r"#55"),
        )
        for (label, (write, header)) in accepted
            path = write(joinpath(dir, replace(label, r"\W+" => "_") * ".bin"), header)
            @test open_error(path) === nothing
        end
        for (label, (write, header, pattern)) in rejected
            path = write(joinpath(dir, replace(label, r"\W+" => "_") * ".bin"), header)
            err = open_error(path)
            @test err isa ArgumentError
        end
        @test Pre.era5_tm5_convection_header(true) == Dict(ec2tm_source, stamp)
        @test Pre.era5_tm5_convection_header(false) == Dict("tm5_convection_source" => "none")
    end
end
