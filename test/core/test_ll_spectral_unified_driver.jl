#!/usr/bin/env julia
# Plan 41 - LL spectral unified-driver byte stability.

using Test
using Dates
using JSON3

using AtmosTransport
using .AtmosTransport.Preprocessing: build_target_geometry, process_day, HEADER_SIZE,
                                      ERA5SpectralSettings

const Pre = AtmosTransport.Preprocessing

function _write_fake_ll_spectral_cache!(spectral_dir::String,
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

function _ll_test_vertical(::Type{FT}) where FT
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

function _ll_test_settings(::Type{FT}, spectral_dir, cache_dir, out_dir;
                           horizontal_balance = nothing) where FT
    return ERA5SpectralSettings((
        horizontal_balance = horizontal_balance,
        output_float_type = FT,
        spectral_dir = spectral_dir,
        spectral_cache_dir = cache_dir,
        T_target = 0,
        min_dp = 0.0,
        include_qv = false,
        mass_basis = :moist,
        mass_fix_enable = false,
        target_ps_dry_pa = 98726.0,
        qv_global_climatology = 0.0,
        thermo_dir = dirname(out_dir),
        half_dt = 450.0,
        met_interval = 3600.0,
        dt = 900.0,
        out_dir = out_dir,
        tm5_convection_enable = false,
        include_surface = false,
    ))
end

function _header_without_creation_time(path)
    header_bytes = read(path)[1:HEADER_SIZE]
    json_len = findfirst(==(0x00), header_bytes)
    json_len = json_len === nothing ? HEADER_SIZE : json_len - 1
    header = Dict{Symbol, Any}(JSON3.read(String(header_bytes[1:json_len])))
    delete!(header, :creation_time)
    delete!(header, :generation_fingerprint)
    return header
end

@testset "LL spectral unified driver emits reproducible payload bytes" begin
    mktempdir() do tmp
        FT = Float64
        date = Date(2021, 12, 1)
        spectral_dir = joinpath(tmp, "spectral")
        cache_dir = joinpath(tmp, "cache")
        _write_fake_ll_spectral_cache!(spectral_dir, cache_dir, date)

        grid = build_target_geometry(Val(:latlon),
                                     Dict{String, Any}("nlon" => 4,
                                                       "nlat" => 3),
                                     FT)
        vertical = _ll_test_vertical(FT)
        first_settings = _ll_test_settings(FT, spectral_dir, cache_dir,
                                           joinpath(tmp, "first"))
        second_settings = _ll_test_settings(FT, spectral_dir, cache_dir,
                                            joinpath(tmp, "second"))

        first_path, first_last = process_day(date, grid, first_settings,
                                             vertical;
                                             positivity_cfl_limit = 0.95)
        second_path, second_last = process_day(date, grid, second_settings,
                                               vertical;
                                               positivity_cfl_limit = 0.95)

        @test isfile(first_path)
        @test isfile(second_path)
        @test filesize(first_path) == filesize(second_path)
        @test _header_without_creation_time(second_path) ==
              _header_without_creation_time(first_path)
        @test read(second_path)[HEADER_SIZE + 1:end] ==
              read(first_path)[HEADER_SIZE + 1:end]
        @test second_last.m == first_last.m
        @test second_last.am == first_last.am
        @test second_last.bm == first_last.bm

        # The balance mode is recorded in the header (column by default).
        @test _header_without_creation_time(first_path)[:horizontal_balance] == "column"
        layer = _ll_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "layer");
                                  horizontal_balance = AtmosTransport.Preprocessing.LayerBalance())
        layer_path, _ = process_day(date, grid, layer, vertical; positivity_cfl_limit = 0.95)
        @test _header_without_creation_time(layer_path)[:horizontal_balance] == "per_layer"
    end
end
