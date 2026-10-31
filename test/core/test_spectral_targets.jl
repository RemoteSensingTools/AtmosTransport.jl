# Spectral target regressions share deterministic cache and vertical fixtures.
# Plan 41 - LL spectral unified-driver byte stability.

using Test
using Dates
using JSON3
import NCDatasets

using AtmosTransport
using .AtmosTransport.Parameters: GLOBAL_MEAN_DRY_SURFACE_PRESSURE
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
    Pre.write_spectral_day_cache(path, spec)
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
                           column_balance_weights = :mass) where FT
    return ERA5SpectralSettings((
        column_balance_weights = column_balance_weights,
        output_float_type = FT,
        spectral_dir = spectral_dir,
        spectral_cache_dir = cache_dir,
        T_target = 0,
        min_dp = 0.0,
        include_qv = false,
        mass_basis = :moist,
        mass_fix_enable = false,
        target_ps_dry_pa = GLOBAL_MEAN_DRY_SURFACE_PRESSURE,
        qv_global_climatology = 0.0,
        thermo_dir = dirname(out_dir),
        half_dt = 450.0,
        met_interval = 3600.0,
        dt = 900.0,
        out_dir = out_dir,
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

        first_path = process_day(date, grid, first_settings,
                                             vertical;
                                             positivity_cfl_limit = 0.95)
        second_path = process_day(date, grid, second_settings,
                                               vertical;
                                               positivity_cfl_limit = 0.95)

        @test isfile(first_path)
        @test isfile(second_path)
        @test filesize(first_path) == filesize(second_path)
        @test _header_without_creation_time(second_path) ==
              _header_without_creation_time(first_path)
        @test read(second_path)[HEADER_SIZE + 1:end] ==
              read(first_path)[HEADER_SIZE + 1:end]
        @test _header_without_creation_time(first_path)[:surface_pressure_basis] == "total"

        # The column balance is recorded in the header.
        @test _header_without_creation_time(first_path)[:horizontal_balance] == "column"

        # `write_replay_check = false` reaches the gate and the header; the payload
        # is the same. A binary that passed the gate records nothing.
        off = _ll_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "off"))
        off_path = process_day(
            date, grid, off, vertical; positivity_cfl_limit = 0.95, write_replay_check = false)
        @test _header_without_creation_time(off_path)[:write_replay_check] === false
        @test !haskey(_header_without_creation_time(first_path), :write_replay_check)
        @test read(off_path)[HEADER_SIZE + 1:end] == read(first_path)[HEADER_SIZE + 1:end]

        # The lat-lon column balance takes every column weighting; the header
        # records a non-mass one. With one (hybrid) level every weighting
        # puts the whole correction there, so the payload is unchanged.
        hyb = _ll_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "hyb");
                                column_balance_weights = :hybrid_b)
        hyb_path = process_day(date, grid, hyb, vertical; positivity_cfl_limit = 0.95)
        @test _header_without_creation_time(hyb_path)[:column_balance_weights] == "hybrid_b"
        @test !haskey(_header_without_creation_time(first_path), :column_balance_weights)
        @test read(hyb_path)[HEADER_SIZE + 1:end] == read(first_path)[HEADER_SIZE + 1:end]
        bad = _ll_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "bad");
                                column_balance_weights = :pressure)
        err = try process_day(date, grid, bad, vertical; positivity_cfl_limit = 0.95); nothing catch e; e end
        @test err isa ArgumentError && contains(err.msg, "column_balance_weights must be one of")
    end
end

# Plan 41 - CS spectral unified-driver byte stability.

using Test
using Dates
using JSON3

using AtmosTransport
using .AtmosTransport.Parameters: GLOBAL_MEAN_DRY_SURFACE_PRESSURE
using .AtmosTransport.Preprocessing: build_target_geometry, process_day,
                                      ERA5SpectralSettings

@testset "spectral endpoint flux mean is rebuilt from raw endpoints" begin
    panels(value) = ntuple(_ -> fill(Float64(value), 2, 2, 1), 6)
    ws = (endpoint_am = panels(2), endpoint_bm = panels(4),
          cur_am = panels(0), cur_bm = panels(0))
    end_am, end_bm = panels(6), panels(12)
    Pre.rebuild_spectral_window_fluxes!(ws, end_am, end_bm, :window_mean, 4, 4)
    @test all(==(4.0), Iterators.flatten(ws.cur_am))
    @test all(==(8.0), Iterators.flatten(ws.cur_bm))
    foreach(a -> fill!(a, 999.0), ws.cur_am) # stand in for Poisson-balanced faces
    foreach(a -> fill!(a, 999.0), ws.cur_bm)
    Pre.rebuild_spectral_window_fluxes!(ws, end_am, end_bm, :window_mean, 4, 8)
    @test all(==(2.0), Iterators.flatten(ws.cur_am))
    @test all(==(4.0), Iterators.flatten(ws.cur_bm))
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

function _cs_test_settings(::Type{FT}, spectral_dir, cache_dir, out_dir;
                           column_balance_weights = :mass,
                           face_lengths = :edge,
                           face_fluxes = :vector,
                           face_interpolation = :fv3,
                           wind_regrid = :cartesian,
                           flux_time_sampling = :window_mean) where FT
    return ERA5SpectralSettings((
        column_balance_weights = column_balance_weights,
        output_float_type = FT,
        spectral_dir = spectral_dir,
        spectral_cache_dir = cache_dir,
        T_target = 0,
        min_dp = 0.0,
        include_qv = false,
        mass_basis = :moist,
        mass_fix_enable = false,
        target_ps_dry_pa = GLOBAL_MEAN_DRY_SURFACE_PRESSURE,
        qv_global_climatology = 0.0,
        thermo_dir = dirname(out_dir),
        half_dt = 450.0,
        met_interval = 3600.0,
        dt = 900.0,
        out_dir = out_dir,
        face_lengths,
        face_fluxes,
        face_interpolation,
        wind_regrid,
        flux_time_sampling,
    ))
end

@testset "CS spectral unified driver emits reproducible bytes" begin
    mktempdir() do tmp
        FT = Float64
        date = Date(2021, 12, 1)
        spectral_dir = joinpath(tmp, "spectral")
        cache_dir = joinpath(tmp, "cache")
        regrid_cache_dir = joinpath(tmp, "regrid_cache")
        _write_fake_ll_spectral_cache!(spectral_dir, cache_dir, date)

        grid = build_target_geometry(
            Val(:cubed_sphere),
            Dict{String, Any}(
                "Nc" => 2,
                "staging_nlon" => 4,
                "staging_nlat" => 3,
                "regridder_cache_dir" => regrid_cache_dir,
            ),
            FT)
        vertical = _ll_test_vertical(FT)
        first_settings = _cs_test_settings(FT, spectral_dir, cache_dir,
                                           joinpath(tmp, "first"))
        second_settings = _cs_test_settings(FT, spectral_dir, cache_dir,
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
        @test first_header[:mass_fix_enabled] === false
        @test first_header[:surface_pressure_basis] == "total"
        @test first_header[:face_fluxes] == "vector"
        @test first_header[:flux_face_lengths] == "edge"
        @test first_header[:face_interpolation] == "fv3"
        @test first_header[:wind_regrid] == "cartesian"
        @test first_header[:flux_time_sampling] == "window_mean"
        @test first_header[:flux_time_sampling_effective] ==
              "window_start_boundary_fallback"
        @test !haskey(first_header, :mass_fix_target_ps_dry_pa)
        @test length(first_header[:ps_offsets_pa_per_window]) == 24
        @test first_header[:ps_offsets_next_day_hour0_pa] isa Real

        # `write_replay_check = false` skips the gate and is recorded; same payload.
        off = _cs_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "off"))
        off_path = process_day(
            date, grid, off, vertical; positivity_cfl_limit = 0.95, write_replay_check = false)
        off_header, off_payload = _stable_binary_parts(off_path)
        @test off_header[:write_replay_check] === false
        @test !haskey(first_header, :write_replay_check)
        @test off_payload == first_payload

        # Both alternatives dispatch through the same spectral endpoint
        # lifecycle: source-grid line integrals and the fully historical path.
        li = _cs_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "li");
                               face_fluxes = :line_integral,
                               face_interpolation = :linear)
        li_header, li_payload = _stable_binary_parts(
            process_day(date, grid, li, vertical; positivity_cfl_limit = 0.95))
        @test li_header[:face_fluxes] == "line_integral"
        @test li_header[:wind_regrid] == "source_line_integral"
        @test li_header[:line_integral_samples_per_face] == 16
        @test li_header[:line_integral_source_grid] == "latlon_staging"
        @test li_payload == first_payload

        legacy = _cs_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "legacy");
                                   face_lengths = :cell_centerline,
                                   face_fluxes = :panel_average,
                                   face_interpolation = :linear,
                                   wind_regrid = :scalar,
                                   flux_time_sampling = :window_start)
        legacy_header, legacy_payload = _stable_binary_parts(
            process_day(date, grid, legacy, vertical; positivity_cfl_limit = 0.95))
        @test legacy_header[:face_fluxes] == "panel_average"
        @test legacy_header[:flux_face_lengths] == "cell_centerline"
        @test legacy_header[:flux_time_sampling] == "window_start"
        @test legacy_payload == first_payload

        # Column weights are recorded only when not the default mass weights;
        # with one level every weighting gives the same fluxes.
        hyb = _cs_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "hyb");
                                column_balance_weights = :hybrid_b)
        hyb_header, hyb_payload = _stable_binary_parts(
            process_day(date, grid, hyb, vertical; positivity_cfl_limit = 0.95))
        @test hyb_header[:column_balance_weights] == "hybrid_b"
        @test !haskey(first_header, :column_balance_weights)
        @test hyb_payload == first_payload
    end
end

# Plan 41 - RG spectral unified-driver byte stability.

using Test
using Dates
using JSON3

using AtmosTransport
using .AtmosTransport.Parameters: GLOBAL_MEAN_DRY_SURFACE_PRESSURE
using .AtmosTransport.Preprocessing: build_target_geometry, process_day,
                                      ERA5SpectralSettings

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
    Pre.write_spectral_day_cache(path, spec)
    return path
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
                           mass_basis::Symbol = :moist,
                           qv_preload::Bool = false,
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
        mass_basis = mass_basis,
        mass_fix_enable = mass_fix,
        target_ps_dry_pa = GLOBAL_MEAN_DRY_SURFACE_PRESSURE,
        qv_global_climatology = 0.0,
        qv_preload = qv_preload,
        qv_preload_max_bytes = qv_preload ? Int64(8) * 1024^3 : Int64(0),
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

# Dry basis is the default: the RG target reads hourly humidity from the ERA5
# thermo sidecar, interpolates it to the RG cells and converts the native mass
# and fluxes before the level merge. The sidecar here holds point values on the
# ERA5 layout (longitude 0..315°, latitude 90..-90°) with one humidity per level,
# so every stored dry cell mass is (1 - q_k) times the moist run's.
function _write_rg_level_thermo!(thermo_dir, date, q_by_level; hours = 24)
    lons = collect(0.0:45.0:315.0)
    lats = collect(90.0:-45.0:-90.0)
    path = joinpath(thermo_dir, "era5_thermo_ml_$(Dates.format(date, "yyyymmdd")).nc")
    NCDatasets.NCDataset(path, "c") do ds
        NCDatasets.defDim(ds, "longitude", length(lons))
        NCDatasets.defDim(ds, "latitude", length(lats))
        NCDatasets.defDim(ds, "hybrid", length(q_by_level))
        NCDatasets.defDim(ds, "time", hours)
        NCDatasets.defVar(ds, "longitude", Float32.(lons), ("longitude",))
        NCDatasets.defVar(ds, "latitude", Float32.(lats), ("latitude",))
        NCDatasets.defVar(ds, "q", Float32[q_by_level[k] for _ in lons, _ in lats,
                                                k in eachindex(q_by_level), _ in 1:hours],
                          ("longitude", "latitude", "hybrid", "time"))
    end
    return path
end

@testset "RG dry basis converts native mass with the thermo humidity" begin
    mktempdir() do tmp
        date = Date(2021, 12, 1)
        spectral_dir = joinpath(tmp, "spectral")
        cache_dir = joinpath(tmp, "cache")
        _write_vertically_compensating_spectral_cache!(spectral_dir, cache_dir, date)
        q_by_level = (0.01f0, 0.003f0)
        _write_rg_level_thermo!(tmp, date, q_by_level)
        grid = build_target_geometry(Val(:synthetic_reduced_gaussian),
                                     Dict{String, Any}("gaussian_number" => 1), Float64)
        vertical = _rg_two_level_vertical(Float64)
        stored = Dict{Tuple{Symbol, Bool}, Any}()
        # Hourly reads and the preloaded day are both production paths.
        for (basis, preload) in ((:moist, false), (:dry, false), (:dry, true))
            settings = _rg_test_settings(Float64, spectral_dir, cache_dir,
                                         joinpath(tmp, "out_$(basis)_$(preload)");
                                         T_target = 1, mass_basis = basis,
                                         qv_preload = preload)
            path = process_day(date, grid, settings, vertical; positivity_cfl_limit = 0.95)
            reader = AtmosTransport.MetDrivers.TransportBinaryReader(path; FT = Float64)
            try
                @test reader.header.mass_basis == basis
                m, _, fluxes = AtmosTransport.MetDrivers.load_window!(reader, 1)
                @test maximum(abs, fluxes.cm[:, 2]) > 0
                stored[(basis, preload)] = copy(m)
            finally
                close(reader)
            end
        end
        moist = stored[(:moist, false)]
        @test stored[(:dry, true)] == stored[(:dry, false)]
        for k in eachindex(q_by_level)
            @test isapprox(stored[(:dry, false)][:, k],
                           (1 - Float64(q_by_level[k])) .* moist[:, k]; rtol = 1e-12)
        end
    end
end

@testset "RG spectral unified driver emits reproducible bytes" begin
    mktempdir() do tmp
        FT = Float64
        date = Date(2021, 12, 1)
        spectral_dir = joinpath(tmp, "spectral")
        cache_dir = joinpath(tmp, "cache")
        _write_fake_ll_spectral_cache!(spectral_dir, cache_dir, date)

        grid = build_target_geometry(Val(:synthetic_reduced_gaussian),
                                     Dict{String, Any}("gaussian_number" => 1),
                                     FT)
        vertical = _ll_test_vertical(FT)
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
        off_path = process_day(
            date, grid, off, vertical; positivity_cfl_limit = 0.95, write_replay_check = false)
        off_header, off_payload = _stable_binary_parts(off_path)
        @test off_header[:write_replay_check] === false
        @test !haskey(first_header, :write_replay_check)
        @test off_payload == first_payload

        # The reduced-Gaussian column balance takes every column weighting and
        # records a non-mass one; with one hybrid level the payload is the same.
        hyb = _rg_test_settings(FT, spectral_dir, cache_dir, joinpath(tmp, "hyb");
                                column_balance_weights = :hybrid_b)
        hyb_header, hyb_payload = _stable_binary_parts(
            process_day(date, grid, hyb, vertical; positivity_cfl_limit = 0.95))
        @test hyb_header[:column_balance_weights] == "hybrid_b"
        @test !haskey(first_header, :column_balance_weights)
        @test hyb_payload == first_payload

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
            ws = Pre.allocate_window_workspace(grid, settings, _ll_test_vertical(FT),
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

# ---------------------------------------------------------------------------
# ERA5 spectral preprocessing: the two pole rows of a lat-lon mesh are
# ordinary cells (issue #58).
#
# The rows j = 1 and j = Ny of a `LatLonMesh` are cells centred at
# ±(90° − Δφ/2); only their outer meridional faces lie on the pole and have
# zero length. Their zonal faces carry the synthesized flux. Zeroing those
# zonal fluxes closes the pole-row cells to zonal flow, so a flow across the
# pole has no path: the lat-lon column balance then re-routes `bm` at every
# latitude, and the cubed-sphere staging recovers `u = 0` in the pole rows.
# The zeroing was removed once on 2026-04-03
# (docs/resources/bug_archive/SESSION1_GRID_CM_FIX_2026-04-03.md, section 3)
# and came back with the v4 spectral preprocessor; these tests keep it out.
#
# Flow: solid-body rotation about an axis tilted 45° from the pole
# (u₀ = 20 m/s, 14 m/s at the pole), non-divergent, constant surface pressure,
# one full-column layer, written as a spectral day cache (no GRIB needed).
# ---------------------------------------------------------------------------

using Test
using Dates

using AtmosTransport
using .AtmosTransport.Parameters: GLOBAL_MEAN_DRY_SURFACE_PRESSURE, IFS_EARTH_RADIUS,
                                  STANDARD_GRAVITY
using .AtmosTransport.Preprocessing: build_target_geometry, process_day,
                                      ERA5SpectralSettings, SubstepSchedulePolicy
using .AtmosTransport.MetDrivers: TransportBinaryReader, load_window!, load_grid,
                                  steps_per_window, window_count
using .AtmosTransport.Grids: panel_cell_corner_lonlat

const PoleRowsPre = AtmosTransport.Preprocessing

const POLE_U0 = 20.0                    # m/s, equatorial speed of the rotation
const POLE_TILT = deg2rad(45.0)         # rotation axis tilted from the pole
const POLE_PS = 1.0e5                   # Pa, constant surface pressure
const POLE_T = 7                        # spectral truncation of the cache

pole_u(λ, φ) = POLE_U0 * (cosd(φ) * cos(POLE_TILT) + sind(φ) * cosd(λ) * sin(POLE_TILT))
pole_v(λ) = -POLE_U0 * sind(λ) * sin(POLE_TILT)
# Stream function on a sphere of radius R; the flux through a face from corner
# A to corner B of a layer of thickness Δp over Δt is (Δp Δt/g)(ψ(A) − ψ(B))
# for the cubed-sphere face orientation used below.
pole_ψ(λ, φ, R) = -R * POLE_U0 * (sind(φ) * cos(POLE_TILT) - cosd(φ) * cosd(λ) * sin(POLE_TILT))

# Vorticity of the rotation in ECMWF's fully normalised spherical harmonics
# (`spec[n+1, m+1]`): ζ = (2u₀/a)(sin φ cos α − cos φ cos λ sin α), with
# P̃₁⁰ = √3 sin φ and P̃₁¹ = √(3/2) cos φ (the m = 1 term enters twice).
function _rotation_vorticity(T)
    a = IFS_EARTH_RADIUS
    vo = zeros(ComplexF64, T + 1, T + 1)
    vo[2, 1] = 2POLE_U0 / a * cos(POLE_TILT) / sqrt(3)
    vo[2, 2] = -(2POLE_U0 / a) * sin(POLE_TILT) / (2sqrt(1.5))
    return vo
end

function _write_rotation_cache!(spectral_dir, cache_dir, date; T = POLE_T)
    mkpath(spectral_dir)
    mkpath(cache_dir)
    ds = Dates.format(date, "yyyymmdd")
    vo_d_path = joinpath(spectral_dir, "era5_spectral_$(ds)_vo_d.gb")
    lnsp_path = joinpath(spectral_dir, "era5_spectral_$(ds)_lnsp.gb")
    touch(vo_d_path)
    touch(lnsp_path)
    lnsp = zeros(ComplexF64, T + 1, T + 1)
    lnsp[1, 1] = log(POLE_PS)
    vo = zeros(ComplexF64, T + 1, T + 1, 137)
    vo[:, :, 1] .= _rotation_vorticity(T)
    hours = collect(0:23)
    spec = (hours = hours,
            lnsp_all = Dict(h => copy(lnsp) for h in hours),
            vo_by_hour = Dict(h => copy(vo) for h in hours),
            d_by_hour = Dict(h => zeros(ComplexF64, T + 1, T + 1, 137) for h in hours),
            T = T, n_times = length(hours))
    PoleRowsPre.write_spectral_day_cache(
        PoleRowsPre.spectral_day_cache_path(cache_dir, vo_d_path, lnsp_path; T_target = T), spec)
    return nothing
end

# One full-column layer: Δp = ps.
_pole_vertical(::Type{FT}) where FT = (
    Nz_native = 1, Nz = 1, level_range = 1:1,
    ab = (dA = FT[0], dB = FT[1], b_ifc = FT[0, 1]),
    merge_map = [1],
    merged_vc = AtmosTransport.HybridSigmaPressure(FT[0, 0], FT[0, 1]))

_pole_settings(spectral_dir, cache_dir, out_dir; extra...) = ERA5SpectralSettings(merge((
    column_balance_weights = :mass,
    output_float_type = Float64,
    spectral_dir, spectral_cache_dir = cache_dir,
    T_target = POLE_T, min_dp = 0.0,
    include_qv = false, mass_basis = :moist,
    mass_fix_enable = false,
    target_ps_dry_pa = GLOBAL_MEAN_DRY_SURFACE_PRESSURE,
    qv_global_climatology = 0.0,
    thermo_dir = dirname(out_dir),
    half_dt = 450.0, met_interval = 3600.0, dt = 900.0,
    out_dir,
    tm5_convection_enable = false, include_surface = false), (; extra...)))

# Least-squares ratio of a stored face row to its reference.
_row_ratio(stored, ref) = sum(stored .* ref) / sum(abs2, ref)

@testset "ERA5 spectral pole rows are ordinary cells (issue #58)" begin
    mktempdir() do tmp
        date = Date(2021, 12, 1)
        spectral_dir = joinpath(tmp, "spectral")
        cache_dir = joinpath(tmp, "cache")
        _write_rotation_cache!(spectral_dir, cache_dir, date)
        vertical = _pole_vertical(Float64)
        g = STANDARD_GRAVITY

        @testset "lat-lon 96×48: stored fluxes match the rotation at every latitude" begin
            grid = build_target_geometry(Val(:latlon),
                Dict{String, Any}("nlon" => 96, "nlat" => 48), Float64)
            # The narrow pole-row cells need more than the 4 configured
            # substeps; the adaptive schedule (the production default) adds them.
            path = process_day(date, grid,
                _pole_settings(spectral_dir, cache_dir, joinpath(tmp, "ll")), vertical;
                positivity_cfl_limit = 0.95,
                substep_policy = SubstepSchedulePolicy(adaptive_substeps = true,
                                                       substep_cfl_target = 0.95))
            reader = TransportBinaryReader(path; FT = Float64)
            Nx, Ny = 96, 48
            R = grid.mesh.radius
            λc = collect(grid.lons); φc = collect(grid.lats); λf = collect(grid.mesh.λᶠ)
            m = zeros(Nx, Ny, 1); ps = zeros(Nx, Ny)
            am = zeros(Nx + 1, Ny, 1); bm = zeros(Nx, Ny + 1, 1); cm = zeros(Nx, Ny, 2)
            u_cc = zeros(Nx, Ny, 1); v_cc = zeros(Nx, Ny, 1)
            worst_am = worst_bm = worst_u = 0.0
            for win in (1, 12, window_count(reader))
                load_window!(reader, win; m, ps, am, bm, cm)
                dt_amount = 3600.0 / (2 * steps_per_window(reader, win))
                # Reference amounts sample the rotation where the C grid does:
                # u at the west faces of each row, v·cos φ averaged from the
                # two adjacent cell centres to the face. The column balance
                # moves the balanced fluxes off these by its O(Δ²) correction
                # of the discrete divergence (5e-4 here, 9e-4 on 72×37).
                for j in 1:Ny
                    ref = [pole_u(λf[i], φc[j]) * POLE_PS * R * grid.dlat / g * dt_amount for i in 1:Nx]
                    worst_am = max(worst_am, abs(_row_ratio(am[1:Nx, j, 1], ref) - 1))
                end
                for j in 2:Ny
                    ref = [pole_v(λc[i]) * (cosd(φc[j - 1]) + cosd(φc[j])) / 2 *
                           POLE_PS * R * grid.dlon / g * dt_amount for i in 1:Nx]
                    worst_bm = max(worst_bm, abs(_row_ratio(bm[:, j, 1], ref) - 1))
                end
                @test all(iszero, bm[:, 1, 1]) && all(iszero, bm[:, Ny + 1, 1])
                # The cubed-sphere staging and the LL → CS regrid recover the
                # cell-centre winds from these fluxes; the pole-row u must be
                # the rotation's (it was 0 while the pole-row am was zeroed).
                PoleRowsPre.recover_ll_cell_center_winds!(u_cc, v_cc, am, bm, ps,
                    [0.0, 0.0], [0.0, 1.0], φc, R * grid.dlat, grid.dlon, R, g, dt_amount)
                for j in (1, Ny)
                    exact = [pole_u(λ, φc[j]) for λ in λc]
                    worst_u = max(worst_u, maximum(abs, u_cc[:, j, 1] .- exact) / maximum(abs, exact))
                end
            end
            # worst_* are max |ratio − 1| over the rows (u: max error / max |u|).
            @test worst_am < 1e-3          # pole rows included; main: 0.998
            @test worst_bm < 1e-3          # main: 0.998 (ratio 0.002 at the first interior face row)
            @test worst_u < 5e-3           # main: 1.000 (u = 0 in the pole rows)
            close(reader)
        end

        @testset "cubed sphere C24 (96×48 staging): polar face amounts" begin
            grid = build_target_geometry(Val(:cubed_sphere), Dict{String, Any}(
                "Nc" => 24, "staging_nlon" => 96, "staging_nlat" => 48,
                "regridder_cache_dir" => joinpath(tmp, "regrid_cache")), Float64)
            path = process_day(date, grid,
                _pole_settings(spectral_dir, cache_dir, joinpath(tmp, "cs");
                               face_lengths = :edge, face_fluxes = :vector,
                               face_interpolation = :fv3, wind_regrid = :cartesian,
                               flux_time_sampling = :window_mean), vertical;
                positivity_cfl_limit = 0.95)
            reader = TransportBinaryReader(path; FT = Float64)
            mesh = load_grid(reader).horizontal
            Nc = mesh.Nc
            R = mesh.radius
            win = 12
            w = load_window!(reader, win)
            c = POLE_PS / g * 3600.0 / (2 * steps_per_window(reader, win))
            err = zeros(2); ref = zeros(2)   # |lat| < 60, |lat| > 75
            for p in 1:6
                lon, lat = panel_cell_corner_lonlat(mesh, p)
                for j in 1:Nc, i in 1:Nc
                    φ = abs(sum(@view lat[i:i+1, j:j+1]) / 4)
                    band = φ > 75 ? 2 : φ < 60 ? 1 : 0
                    band == 0 && continue
                    ψ(a, b) = pole_ψ(lon[a, b], lat[a, b], R)
                    for (F, e) in ((w.am[p][i, j, 1], c * (ψ(i, j) - ψ(i, j + 1))),
                                   (w.bm[p][i, j, 1], c * (ψ(i + 1, j) - ψ(i, j))))
                        err[band] += (F - e)^2
                        ref[band] += e^2
                    end
                end
            end
            rel = sqrt.(err ./ ref)
            @test rel[1] < 5e-3
            @test rel[2] < 5e-3            # main: 5.1e-2
            close(reader)
        end
    end
end

@testset "LL → CS regrid: closure record of rescaled fluxes (adaptive vs explicit substeps)" begin
    # The regrid balances and diagnoses `cm` for one substep, then the adaptive
    # schedule rescales am/bm/cm to N. The header `column_closure_*` keys
    # describe the fluxes as stored, so they must not depend on whether N came
    # from the adaptive schedule or from an explicit `steps_per_window = N`.
    # The `cm` ratio describes the diagnosis (run for 1 substep in the adaptive
    # case) and is an upper bound of the explicit-N ratio.
    mktempdir() do tmp
        date = Date(2021, 12, 1)
        spectral_dir = joinpath(tmp, "spectral")
        cache_dir = joinpath(tmp, "cache")
        _write_rotation_cache!(spectral_dir, cache_dir, date)
        grid = build_target_geometry(Val(:latlon),
            Dict{String, Any}("nlon" => 72, "nlat" => 37), Float64)
        ll_path = process_day(date, grid,
            _pole_settings(spectral_dir, cache_dir, joinpath(tmp, "ll")), _pole_vertical(Float64);
            positivity_cfl_limit = 0.95,
            substep_policy = SubstepSchedulePolicy(adaptive_substeps = true,
                                                   substep_cfl_target = 0.95))
        cs_grid = build_target_geometry(Val(:cubed_sphere), Dict{String, Any}(
            "Nc" => 24, "regridder_cache_dir" => joinpath(tmp, "regrid_cache")), Float64)
        # A tight limit makes the adaptive schedule leave one substep (N = 4 here).
        lim = 0.1
        adaptive = joinpath(tmp, "adaptive.bin")
        PoleRowsPre.regrid_ll_binary_to_cs(ll_path, cs_grid, adaptive; FT = Float64,
                                           positivity_cfl_limit = lim)
        ra = TransportBinaryReader(adaptive; FT = Float64)
        sched = Int.(ra.header.steps_per_window_by_window)
        ha = ra.header.raw_header
        wa = load_window!(ra, 12)
        close(ra)
        N = maximum(sched)
        @test N > 1 && all(==(N), sched)
        explicit = joinpath(tmp, "explicit.bin")
        PoleRowsPre.regrid_ll_binary_to_cs(ll_path, cs_grid, explicit; FT = Float64,
                                           positivity_cfl_limit = lim, steps_per_window = N)
        rb = TransportBinaryReader(explicit; FT = Float64)
        hb = rb.header.raw_header
        wb = load_window!(rb, 12)
        close(rb)
        # Same stored fluxes ...
        for p in 1:6
            @test wa.am[p] ≈ wb.am[p] rtol = 1e-9
            @test wa.bm[p] ≈ wb.bm[p] rtol = 1e-9
        end
        # ... same record of their closure (on 45e86d6a the adaptive kg values
        # and the projected ratio were N times the explicit ones).
        for key in ("column_closure_projected_residual_max_kg",
                    "column_closure_residual_to_tolerance_max",
                    "column_closure_booked_residual_max_kg")
            @test ha[key] ≈ hb[key] rtol = 1e-6
        end
        @test ha["column_closure_windows"] == hb["column_closure_windows"]
        @test ha["column_closure_cm_residual_to_tolerance_max"] >=
              hb["column_closure_cm_residual_to_tolerance_max"]
    end
end
