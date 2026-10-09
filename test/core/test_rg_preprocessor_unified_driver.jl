#!/usr/bin/env julia
# Plan 41 - RG spectral unified-driver byte stability.

using Test
using Dates
using JSON3

using AtmosTransport
using .AtmosTransport.Preprocessing: build_target_geometry, process_day,
                                      ERA5SpectralSettings

const Pre = AtmosTransport.Preprocessing

function _write_fake_spectral_cache!(spectral_dir::String,
                                     cache_dir::String,
                                     date::Date)
    mkpath(spectral_dir)
    mkpath(cache_dir)
    ds = Dates.format(date, "yyyymmdd")
    vo_d_path = joinpath(spectral_dir, "era5_spectral_$(ds)_vo_d.gb")
    lnsp_path = joinpath(spectral_dir, "era5_spectral_$(ds)_lnsp.gb")
    touch(vo_d_path)
    touch(lnsp_path)

    lnsp = fill(complex(log(100000.0), 0.0), 1, 1)
    hours = collect(0:23)
    zero_levels = zeros(ComplexF64, 1, 1, 137)
    spec = (
        hours = hours,
        lnsp_all = Dict(hour => copy(lnsp) for hour in hours),
        vo_by_hour = Dict(hour => copy(zero_levels) for hour in hours),
        d_by_hour = Dict(hour => copy(zero_levels) for hour in hours),
        T = 0,
        n_times = length(hours),
    )
    path = Pre.spectral_day_cache_path(cache_dir, vo_d_path, lnsp_path;
                                       T_target = 0)
    Pre._write_spectral_day_cache(path, spec)
    return path
end

function _stable_binary_parts(path)
    bytes = read(path)
    json_end = something(findfirst(==(0x00), bytes), length(bytes) + 1) - 1
    header = Dict{Symbol, Any}(JSON3.read(String(bytes[1:json_end])))
    header_bytes = Int(header[:header_bytes])
    delete!(header, :creation_time)
    delete!(header, :generation_fingerprint)
    return header, bytes[header_bytes + 1:end]
end

function _rg_test_vertical(::Type{FT}) where FT
    vc = AtmosTransport.HybridSigmaPressure(FT[0, 0], FT[0, 1])
    return (
        Nz_native = 1,
        Nz = 1,
        level_range = 1:1,
        ab = (dA = FT[0], dB = FT[1], b_ifc = FT[0, 1]),
        merge_map = [1],
        merged_vc = vc,
    )
end

function _rg_test_settings(::Type{FT}, spectral_dir, cache_dir, out_dir;
                           include_qv::Bool = false, mass_fix::Bool = false) where FT
    return ERA5SpectralSettings((
        output_float_type = FT,
        spectral_dir = spectral_dir,
        spectral_cache_dir = cache_dir,
        T_target = 0,
        min_dp = 0.0,
        include_qv = include_qv,
        mass_basis = :moist,
        mass_fix_enable = mass_fix,
        target_ps_dry_pa = 98726.0,
        qv_global_climatology = 0.0,
        thermo_dir = dirname(out_dir),
        half_dt = 450.0,
        met_interval = 3600.0,
        dt = 900.0,
        out_dir = out_dir,
    ))
end

@testset "RG spectral unified driver emits reproducible bytes" begin
    mktempdir() do tmp
        FT = Float64
        date = Date(2021, 12, 1)
        spectral_dir = joinpath(tmp, "spectral")
        cache_dir = joinpath(tmp, "cache")
        _write_fake_spectral_cache!(spectral_dir, cache_dir, date)

        grid = build_target_geometry(Val(:synthetic_reduced_gaussian),
                                     Dict{String, Any}("gaussian_number" => 1),
                                     FT)
        vertical = _rg_test_vertical(FT)
        unsupported_qv = _rg_test_settings(
            FT, spectral_dir, cache_dir, joinpath(tmp, "unsupported_qv");
            include_qv = true)
        @test_throws ArgumentError process_day(
            date, grid, unsupported_qv, vertical; positivity_cfl_limit = 0.95)
        first_settings = _rg_test_settings(FT, spectral_dir, cache_dir,
                                           joinpath(tmp, "first"))
        second_settings = _rg_test_settings(FT, spectral_dir, cache_dir,
                                            joinpath(tmp, "second"))

        first_path = process_day(date, grid, first_settings, vertical;
                                 positivity_cfl_limit = 0.95)
        second_path = process_day(date, grid, second_settings, vertical;
                                  positivity_cfl_limit = 0.95)

        @test isfile(first_path)
        @test isfile(second_path)
        @test filesize(first_path) == filesize(second_path)
        first_header, first_payload = _stable_binary_parts(first_path)
        second_header, second_payload = _stable_binary_parts(second_path)
        @test second_header == first_header
        @test second_payload == first_payload

        # The last window ends at the next day's 00 UTC state, pinned like every window.
        pinned = _rg_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "pinned"); mass_fix = true)
        next_day = (lnsp = fill(complex(log(101000.0), 0.0), 1, 1), vo = zeros(ComplexF64, 1, 1, 137),
                    d = zeros(ComplexF64, 1, 1, 137), T = 0)
        header, _ = _stable_binary_parts(process_day(date, grid, pinned, vertical;
                                                     positivity_cfl_limit = 0.95,
                                                     next_day_hour0 = next_day))
        @test header[:mass_fix_qv_mode] == "global_qv_climatology"
        @test all(≈(98726.0 - 100000.0), header[:ps_offsets_pa_per_window])
        @test header[:ps_offsets_next_day_hour0_pa] ≈ 98726.0 - 101000.0
    end
end
