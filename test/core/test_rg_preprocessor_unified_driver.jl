#!/usr/bin/env julia
# Plan 41 - RG spectral unified-driver byte stability.

using Test
using Dates
using JSON3

using AtmosTransport
using .AtmosTransport.Parameters: GLOBAL_MEAN_DRY_SURFACE_PRESSURE
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

function _write_vertically_compensating_spectral_cache!(spectral_dir::String,
                                                         cache_dir::String,
                                                         date::Date)
    mkpath(spectral_dir)
    mkpath(cache_dir)
    ds = Dates.format(date, "yyyymmdd")
    vo_d_path = joinpath(spectral_dir, "era5_spectral_$(ds)_vo_d.gb")
    lnsp_path = joinpath(spectral_dir, "era5_spectral_$(ds)_lnsp.gb")
    touch(vo_d_path)
    touch(lnsp_path)

    lnsp = zeros(ComplexF64, 2, 2)
    lnsp[1, 1] = complex(log(100000.0))
    vo = zeros(ComplexF64, 2, 2, 137)
    d = zeros(ComplexF64, 2, 2, 137)
    # Equal-pressure layers with opposite D(1, 0) coefficients have equal and
    # opposite horizontal mass tendencies. Their column tendency is zero, so
    # a column balance preserves the level mismatch for cm to close.
    d[2, 1, 1] = 1.0e-6
    d[2, 1, 2] = -1.0e-6
    hours = collect(0:23)
    spec = (
        hours,
        lnsp_all = Dict(hour => copy(lnsp) for hour in hours),
        vo_by_hour = Dict(hour => copy(vo) for hour in hours),
        d_by_hour = Dict(hour => copy(d) for hour in hours),
        T = 1,
        n_times = length(hours),
    )
    path = Pre.spectral_day_cache_path(cache_dir, vo_d_path, lnsp_path;
                                       T_target = 1)
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

function _rg_two_level_vertical(::Type{FT}) where FT
    vc = AtmosTransport.HybridSigmaPressure(FT[0, 0, 0], FT[0, 0.5, 1])
    return (
        Nz_native = 2,
        Nz = 2,
        level_range = 1:2,
        ab = (dA = FT[0, 0], dB = FT[0.5, 0.5], b_ifc = FT[0, 0.5, 1]),
        merge_map = [1, 2],
        merged_vc = vc,
    )
end

function _rg_test_settings(::Type{FT}, spectral_dir, cache_dir, out_dir;
                           include_qv::Bool = false, mass_fix::Bool = false,
                           column_balance_weights = :mass,
                           T_target::Int = 0) where FT
    return ERA5SpectralSettings((
        column_balance_weights = column_balance_weights,
        output_float_type = FT,
        spectral_dir = spectral_dir,
        spectral_cache_dir = cache_dir,
        T_target = T_target,
        min_dp = 0.0,
        include_qv = include_qv,
        mass_basis = :moist,
        mass_fix_enable = mass_fix,
        target_ps_dry_pa = GLOBAL_MEAN_DRY_SURFACE_PRESSURE,
        qv_global_climatology = 0.0,
        thermo_dir = dirname(out_dir),
        half_dt = 450.0,
        met_interval = 3600.0,
        dt = 900.0,
        out_dir = out_dir,
    ))
end

@testset "RG end-to-end column balance retains vertical motion" begin
    for FT in (Float64, Float32)
        mktempdir() do tmp
            date = Date(2021, 12, 1)
            spectral_dir = joinpath(tmp, "spectral")
            cache_dir = joinpath(tmp, "cache")
            _write_vertically_compensating_spectral_cache!(spectral_dir, cache_dir, date)
            grid = build_target_geometry(Val(:synthetic_reduced_gaussian),
                                         Dict{String, Any}("gaussian_number" => 1), Float64)
            vertical = _rg_two_level_vertical(Float64)
            settings = _rg_test_settings(
                FT, spectral_dir, cache_dir, joinpath(tmp, "out"); T_target = 1)
            path = process_day(date, grid, settings, vertical;
                               positivity_cfl_limit = 0.95)

            reader = AtmosTransport.MetDrivers.TransportBinaryReader(path; FT)
            try
                m_cur, _, fluxes = AtmosTransport.MetDrivers.load_window!(reader, 1)
                m_next, _, _ = AtmosTransport.MetDrivers.load_window!(reader, 2)
                hflux, cm = fluxes.horizontal_flux, fluxes.cm
                @test maximum(abs, cm[:, 2]) > zero(FT)

                nf = AtmosTransport.Grids.nfaces(grid.mesh)
                face_left = Vector{Int32}(undef, nf)
                face_right = similar(face_left)
                for f in 1:nf
                    left, right = AtmosTransport.Grids.face_cells(grid.mesh, f)
                    face_left[f] = left
                    face_right[f] = right
                end
                div = zeros(Float64, size(m_cur))
                for k in axes(hflux, 2), f in axes(hflux, 1)
                    flux = Float64(hflux[f, k])
                    face_left[f] > 0 && (div[face_left[f], k] += flux)
                    face_right[f] > 0 && (div[face_right[f], k] -= flux)
                end
                level_scale = maximum(abs, div)
                @test level_scale > 0
                @test maximum(abs, vec(sum(div; dims = 2))) <=
                      (FT === Float64 ? 1e-10 : 1e-5) * level_scale

                replay = AtmosTransport.MetDrivers.verify_window_continuity_rg(
                    m_cur, hflux, cm, m_next, face_left, face_right,
                    reader.header.steps_per_window_by_window[1])
                @test replay.max_rel_err <= AtmosTransport.MetDrivers.replay_tolerance(FT)
            finally
                close(reader)
            end
        end
    end
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
        @test !isfile(first_path * ".tmp")              # staged, then renamed
        @test filesize(first_path) == filesize(second_path)
        first_header, first_payload = _stable_binary_parts(first_path)
        second_header, second_payload = _stable_binary_parts(second_path)
        @test second_header == first_header
        @test second_payload == first_payload
        @test first_header[:surface_pressure_basis] == "total"

        # `write_replay_check = false` skips the gate and is recorded; same payload.
        off = _rg_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "off"))
        off_path = @test_logs (:info, r"replay gate SKIPPED") match_mode = :any process_day(
            date, grid, off, vertical; positivity_cfl_limit = 0.95, write_replay_check = false)
        off_header, off_payload = _stable_binary_parts(off_path)
        @test off_header[:write_replay_check] === false
        @test !haskey(first_header, :write_replay_check)
        @test off_payload == first_payload

        # The reduced-Gaussian column balance spreads its correction by air mass only.
        hyb = _rg_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "hyb");
                                column_balance_weights = :hybrid_b)
        @test_throws ArgumentError process_day(date, grid, hyb, vertical; positivity_cfl_limit = 0.95)

        # The last window ends at the next day's 00 UTC state, pinned like every window.
        pinned = _rg_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "pinned"); mass_fix = true)
        next_day = (lnsp = fill(complex(log(101000.0), 0.0), 1, 1), vo = zeros(ComplexF64, 1, 1, 137),
                    d = zeros(ComplexF64, 1, 1, 137), T = 0)
        header, _ = _stable_binary_parts(process_day(date, grid, pinned, vertical;
                                                     positivity_cfl_limit = 0.95,
                                                     next_day_hour0 = next_day))
        @test header[:mass_fix_qv_mode] == "global_qv_climatology"
        @test header[:horizontal_balance] == "column"
        @test all(≈(GLOBAL_MEAN_DRY_SURFACE_PRESSURE - 100000.0),
                  header[:ps_offsets_pa_per_window])
        @test header[:ps_offsets_next_day_hour0_pa] ≈
              GLOBAL_MEAN_DRY_SURFACE_PRESSURE - 101000.0

        # A failed day keeps an existing binary and removes its staging file. Without
        # the pin, the last window's global mass change cannot be closed by horizontal
        # fluxes, so the replay gate fails after the earlier windows were written.
        failing = _rg_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "failing"))
        final = Pre.output_binary_path(date, failing.out_dir, failing.min_dp, FT)
        mkpath(dirname(final))
        write(final, "existing binary")
        @test_throws ErrorException process_day(date, grid, failing, vertical;
                                                positivity_cfl_limit = 0.95,
                                                next_day_hour0 = next_day)
        @test read(final, String) == "existing binary"
        @test !isfile(final * ".tmp")
    end
end

# The write-time contract replays what the writer stores: the buffer slot's
# fluxes in the output float type, not the Float64 balance work arrays they are
# rounded from. A tolerance below Float32 rounding therefore fails a Float32
# window and passes a Float64 one.
@testset "RG write-time contract checks the stored output arrays" begin
    mktempdir() do tmp
        date = Date(2021, 12, 1)
        for FT in (Float64, Float32)
            grid = build_target_geometry(Val(:synthetic_reduced_gaussian),
                                         Dict{String, Any}("gaussian_number" => 2), FT)
            settings = _rg_test_settings(FT, tmp, tmp, joinpath(tmp, "out"))
            ws = Pre.allocate_window_workspace(grid, settings, _rg_test_vertical(FT),
                                               (T = 0, n_times = 2), date, FT)
            nc, Nz = size(ws.buf.m[1])
            nf = size(ws.buf.hflux[1], 1)
            # Integer masses below 2^24 are exact in Float32, so the endpoints close
            # globally in both float types (the tendency sums to zero over the cells).
            m_cur = [1_000_000.0 + 1000c for c in 1:nc, k in 1:Nz]
            tendency = [isodd(c) ? 500.0 : -500.0 for c in 1:nc, k in 1:Nz]
            ws.buf.m[ws.cur] .= FT.(m_cur)
            ws.buf.m[ws.nxt] .= FT.(m_cur .+ tendency)
            interior(f) = ws.work.face_left[f] > 0 && ws.work.face_right[f] > 0
            ws.buf.hflux[ws.cur] .= FT.([interior(f) ? 1e5 * cospi(0.37f + k) : 0.0
                                         for f in 1:nf, k in 1:Nz])
            contract = Pre.ReducedGaussianContract{FT}(
                replay_tol = 1e-12, steps_per_window = ws.steps_schedule[1],
                face_left = ws.work.face_left, face_right = ws.work.face_right)
            policy = Pre.SubstepSchedulePolicy(adaptive_substeps = false,
                                               substep_cfl_target = 0.95)
            drain() = Pre.drain_ready_windows!(ws, contract, 1, ws.steps_schedule[1];
                                               write_replay_on = true,
                                               substep_policy = policy)
            if FT === Float64
                @test drain().contract.replay.max_rel_err <= 1e-12
            else
                err = try drain(); nothing catch e; e end
                @test err isa ErrorException
                @test occursin("Write-time replay gate FAILED for RG window 1", err.msg)
            end
        end
    end
end
