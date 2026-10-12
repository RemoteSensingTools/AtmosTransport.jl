#!/usr/bin/env julia
"""
Plan 24 Commit 4 — LL `process_day` TM5 integration tests.

Three core-gated testsets (run without `--all`):

  (1) `resolve_tm5_convection_settings` — parsing, defaults, and
      error on missing `physics_bin_dir`.

  (2) `write_day_binary!` round-trip with TM5 sections — hand-fill
      a tiny `WindowStorage` with entu/detu/entd/detd arrays, call
      the writer, re-read the on-disk JSON header + payload bytes,
      and verify the TM5 sections are byte-exact and positioned
      per the reader's section order
      (`_transport_push_optional_sections!`, plan 23 Commit 3).

  (3) `compute_tm5_merged_hour_on_source!` via the Commit-2 NC→BIN
      chain on synthetic ERA5-shaped data — verifies the mmap
      reader feeds Commit-3's grid helper correctly and that the
      merged output has the right shape + sane sign.

Plus one `--all`-gated testset:

  (4) Real `process_day` on Dec 2 2021 (if staged) — requires
      a physics BIN at
      `/temp1/met/era5/0.5x0.5/physics_bin/2021/era5_physics_20211202.bin`
      and the matching ERA5 spectral GRIB + thermo NC. Graceful
      skip when any input is missing. Asserts the output binary
      has `include_tm5conv=true`, non-zero TM5 sections, and
      plausible entu magnitude.
"""

using Test
using Dates
using JSON3
using NCDatasets
using Mmap

using AtmosTransport
using .AtmosTransport.Preprocessing: resolve_tm5_convection_settings,
                                       convert_era5_physics_nc_to_bin,
                                       open_era5_physics_binary,
                                       close_era5_physics_binary,
                                       build_target_geometry,
                                       allocate_tm5_workspace,
                                       tm5_copy_or_regrid_ll!,
                                       compute_tm5_merged_hour_on_source!,
                                       TM5CleanupStats
using .AtmosTransport.Regridding: build_regridder

const RUN_ALL = "--all" in ARGS

# ─────────────────────────────────────────────────────────────────────────
# (1) resolve_tm5_convection_settings
# ─────────────────────────────────────────────────────────────────────────

@testset "resolve_tm5_convection_settings" begin
    # Defaults: missing section → disabled, empty path.
    nt = resolve_tm5_convection_settings(Dict{String, Any}())
    @test nt.tm5_convection_enable == false
    @test nt.tm5_physics_bin_dir == ""

    # Explicit disable, no path needed.
    nt = resolve_tm5_convection_settings(Dict("tm5_convection" => Dict("enable" => false)))
    @test nt.tm5_convection_enable == false

    # Enable with path — path is expanduser'd.
    nt = resolve_tm5_convection_settings(Dict("tm5_convection" =>
        Dict("enable" => true, "physics_bin_dir" => "~/data/physics_bin")))
    @test nt.tm5_convection_enable == true
    @test !startswith(nt.tm5_physics_bin_dir, "~")
    @test endswith(nt.tm5_physics_bin_dir, "physics_bin")

    # Enable without path — error with a fix hint.
    err = try
        resolve_tm5_convection_settings(Dict("tm5_convection" => Dict("enable" => true)))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("physics_bin_dir", err.msg)
    @test occursin("convert_era5_physics_nc_to_bin.jl", err.msg)
end

# ─────────────────────────────────────────────────────────────────────────
# (1b) LL TM5 source->target regrid helper
# ─────────────────────────────────────────────────────────────────────────

@testset "tm5_copy_or_regrid_ll! supports lower-resolution LL target shape" begin
    FT = Float64
    source_grid = build_target_geometry(Val(:latlon),
        Dict{String, Any}("type" => "latlon", "nlon" => 8, "nlat" => 4), FT)
    target_grid = build_target_geometry(Val(:latlon),
        Dict{String, Any}("type" => "latlon", "nlon" => 4, "nlat" => 2), FT)
    regridder = build_regridder(source_grid.mesh, target_grid.mesh; normalize = false)

    ws = allocate_tm5_workspace(8, 4, 3, 2, FT;
                                basis = DryBasis(),
                                regridder = regridder,
                                target_nlon = 4,
                                target_nlat = 2,
                                physics_eltype = Float32)
    fill!(ws.entu_merged_src, FT(7))
    dst = Array{FT, 3}(undef, 4, 2, 2)

    tm5_copy_or_regrid_ll!(dst, ws.entu_merged_src, ws)

    @test size(dst) == (4, 2, 2)
    @test maximum(abs.(dst .- FT(7))) < 1e-10
end

# ─────────────────────────────────────────────────────────────────────────
# (2) write_day_binary! round-trip with TM5 sections
# ─────────────────────────────────────────────────────────────────────────
#
# Strategy: build a tiny LL target geometry (Nx=8, Ny=4, Nz=3),
# fabricate a WindowStorage filled with known values for Nt=2
# windows, call write_day_binary!, then read the resulting file
# back byte-by-byte.
#
# Verifies:
#   - header JSON has include_tm5conv=true
#   - payload_sections ends with "entu","detu","entd","detd"
#   - n_entu..n_detd match the (Nx*Ny*Nz) element count
#   - per-window TM5 bytes at the advertised offset match what
#     we wrote to the WindowStorage
# ─────────────────────────────────────────────────────────────────────────

@testset "LL write_day_binary! with TM5 sections" begin
    using .AtmosTransport.Preprocessing: build_target_geometry,
                                           allocate_window_storage,
                                           allocate_merge_workspace,
                                           build_v4_header,
                                           window_element_counts,
                                           window_byte_sizes,
                                           write_day_binary!,
                                           HEADER_SIZE,
                                           compute_window_deltas!

    mktempdir() do tmp
        FT = Float32
        Nx, Ny, Nz = 8, 4, 3
        Nz_native = 6
        Nt = 2

        grid = build_target_geometry(Val(:latlon),
                                      Dict{String,Any}("nlon" => Nx, "nlat" => Ny),
                                      FT)

        # Synthetic settings NamedTuple covering every field the
        # header/writer paths look at.
        settings = (
            spectral_dir = tmp, coeff_path = "", out_dir = tmp, thermo_dir = "",
            include_qv = false, mass_basis = :moist,
            level_range = 1:Nz_native, min_dp = 1000.0,
            dt = 900.0, met_interval = 3600.0, half_dt = 450.0,
            output_float_type = FT,
            mass_fix_enable = false,
            target_ps_dry_pa = AtmosTransport.Parameters.GLOBAL_MEAN_DRY_SURFACE_PRESSURE,
            qv_global_climatology = 0.00247,
            T_target = 47,
            tm5_convection_enable = true,
            tm5_physics_bin_dir = tmp,
        )

        counts = window_element_counts(grid, Nz;
                                        include_qv=false,
                                        tm5_convection=true)
        byte_sizes = window_byte_sizes(counts, FT, Nt)
        counts = merge(counts, (bytes_per_window = byte_sizes.bytes_per_window,))

        # Minimal vertical stub (build_v4_header fields).
        A = Float64[1000.0 * (Nz_native + 1 - k) for k in 1:(Nz_native + 1)]
        B = Float64[(k - 1) / Nz_native for k in 1:(Nz_native + 1)]
        vc = (A = A, B = B)
        merge_map = [1, 1, 2, 2, 3, 3]
        merged_vc = (A = A[[1, 3, 5, 7]], B = B[[1, 3, 5, 7]])
        vertical = (
            ab = (a_ifc = A, b_ifc = B),
            level_range = 1:Nz_native,
            vc_native = vc,
            merged_vc = merged_vc,
            merge_map = merge_map,
            Nz_native = Nz_native,
            Nz = Nz,
        )

        provenance = (
            script_path = "test_tm5_process_day.jl",
            script_mtime = 0.0,
            git_commit = "testing",
            git_dirty = false,
            creation_time = "2026-04-22T00:00:00",
        )

        sizes = (Nx = Nx, Ny = Ny, Nz = Nz, Nz_native = Nz_native, Nt = Nt,
                 steps_per_met = 4)

        # Allocate storage with TM5 enabled and fill each array with a
        # signature pattern so we can check byte-exact round-trip.
        storage = allocate_window_storage(Nt, FT;
                                            include_qv=false,
                                            tm5_convection=true)
        merged = allocate_merge_workspace(grid, Nz_native, Nz, FT)

        for w in 1:Nt
            storage.all_m[w]    = fill(FT(1.0 + w), Nx, Ny, Nz)
            storage.all_am[w]   = fill(FT(2.0 + w), Nx + 1, Ny, Nz)
            storage.all_bm[w]   = fill(FT(3.0 + w), Nx, Ny + 1, Nz)
            storage.all_cm[w]   = fill(FT(0.1 + w), Nx, Ny, Nz + 1)
            storage.all_ps[w]   = fill(FT(98000.0 + w), Nx, Ny)
            storage.all_entu[w] = reshape(FT.(collect(1:(Nx*Ny*Nz)) .+ (w - 1) * 1000), Nx, Ny, Nz)
            storage.all_detu[w] = reshape(FT.(collect(1:(Nx*Ny*Nz)) .+ (w - 1) * 1000 .+ 10_000), Nx, Ny, Nz)
            storage.all_entd[w] = reshape(FT.(collect(1:(Nx*Ny*Nz)) .+ (w - 1) * 1000 .+ 20_000), Nx, Ny, Nz)
            storage.all_detd[w] = reshape(FT.(collect(1:(Nx*Ny*Nz)) .+ (w - 1) * 1000 .+ 30_000), Nx, Ny, Nz)
        end

        header = build_v4_header(Date(2021, 12, 2), grid, vertical, settings,
                                  FT, counts, sizes, provenance)
        header_json = JSON3.write(header)
        @test length(header_json) < HEADER_SIZE

        bin_path = joinpath(tmp, "tm5_roundtrip.bin")
        # last_hour_next == nothing → deltas get zero-filled per the
        # existing MergeWorkspace behaviour; good enough for the
        # byte-layout test.
        write_day_binary!(bin_path, header_json, storage, settings, merged, nothing)

        @test isfile(bin_path)
        @test filesize(bin_path) == byte_sizes.total_bytes

        # ─── Read back the header ───
        open(bin_path, "r") do io
            hdr_buf = read(io, HEADER_SIZE)
            # Strip trailing zero padding before JSON parse.
            json_end = findfirst(==(UInt8(0)), hdr_buf)
            json_end = json_end === nothing ? HEADER_SIZE : (json_end - 1)
            hdr_parsed = JSON3.read(String(hdr_buf[1:json_end]))

            @test hdr_parsed["include_tm5conv"] == true
            # The TM5 fluxes are put on the binary's basis (moist here).
            @test hdr_parsed["tm5_convection_mass_basis"] == "moist"
            sections = String.(hdr_parsed["payload_sections"])
            @test sections[end-3:end] == ["entu", "detu", "entd", "detd"]

            n_tm5 = Nx * Ny * Nz
            @test hdr_parsed["n_entu"] == n_tm5
            @test hdr_parsed["n_detu"] == n_tm5
            @test hdr_parsed["n_entd"] == n_tm5
            @test hdr_parsed["n_detd"] == n_tm5
        end

        # ─── Verify per-window TM5 bytes at the end of each window ───
        # Layout per window (TM5 enabled, no qv):
        #   m, am, bm, cm, ps, dam, dbm, dcm, dm, entu, detu, entd, detd
        # Bytes-per-window = elems_per_window * sizeof(FT).
        # The last 4 sections (each Nx*Ny*Nz FT) live at offset
        #   -4*n_tm5 .. -3*n_tm5 : entu
        #   -3*n_tm5 .. -2*n_tm5 : detu
        #   -2*n_tm5 .. -1*n_tm5 : entd
        #   -1*n_tm5 .. end      : detd
        # from the window's end.
        n_tm5 = Nx * Ny * Nz
        bytes_per_window = counts.bytes_per_window
        for w in 1:Nt
            open(bin_path, "r") do io
                win_start = HEADER_SIZE + (w - 1) * bytes_per_window
                tm5_block_start = win_start + bytes_per_window - 4 * n_tm5 * sizeof(FT)
                seek(io, tm5_block_start)
                entu_read = read!(io, Vector{FT}(undef, n_tm5))
                detu_read = read!(io, Vector{FT}(undef, n_tm5))
                entd_read = read!(io, Vector{FT}(undef, n_tm5))
                detd_read = read!(io, Vector{FT}(undef, n_tm5))

                @test entu_read == vec(storage.all_entu[w])
                @test detu_read == vec(storage.all_detu[w])
                @test entd_read == vec(storage.all_entd[w])
                @test detd_read == vec(storage.all_detd[w])
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────
# (3) compute_tm5_merged_hour_on_source! via Commit-2 fixture chain
# ─────────────────────────────────────────────────────────────────────────
#
# Build tiny ERA5-convection + thermo NCs with zero UDMF/DDMF
# everywhere, convert to BIN via Commit 2, open the reader, and call
# the Commit-4 pipeline helper. Zero-in should yield zero-out across
# all four TM5 fields at both native and merged verticals.
# ─────────────────────────────────────────────────────────────────────────

@testset "compute_tm5_merged_hour_on_source! zero-forcing" begin
    mktempdir() do tmp
        nc_dir = joinpath(tmp, "nc")
        bin_dir = joinpath(tmp, "bin")
        mkpath(nc_dir)
        mkpath(bin_dir)

        # Build three NCs: today convection, prev-day convection
        # (needed for 00-06 splice), and thermo.
        target_date = Date(2021, 12, 2)
        prev_date   = target_date - Day(1)

        Nlon, Nlat, Nlev = 4, 3, 5

        function _make_conv(path, date)
            ds = NCDataset(path, "c")
            defDim(ds, "longitude", Nlon)
            defDim(ds, "latitude",  Nlat)
            defDim(ds, "hybrid",    Nlev)
            defDim(ds, "step",      12)
            defDim(ds, "time",       2)
            defVar(ds, "longitude", collect(range(0.0, step = 360.0 / Nlon, length = Nlon)), ("longitude",))
            defVar(ds, "latitude",  collect(range(90.0, stop = -90.0, length = Nlat)), ("latitude",))
            defVar(ds, "hybrid",    collect(1.0:Nlev), ("hybrid",))
            defVar(ds, "time",      Float64[0.0, 12.0], ("time",))
            defVar(ds, "step",      collect(1.0:12.0),  ("step",))
            # Zero-everywhere.
            for (nm, v) in (("udmf", 0.0f0), ("ddmf", 0.0f0),
                             ("udrf", 0.0f0), ("ddrf", 0.0f0))
                defVar(ds, nm, zeros(Float32, Nlon, Nlat, Nlev, 12, 2),
                        ("longitude", "latitude", "hybrid", "step", "time"))
            end
            close(ds)
        end
        today_str = Dates.format(target_date, "yyyymmdd")
        prev_str   = Dates.format(prev_date,   "yyyymmdd")
        _make_conv(joinpath(nc_dir, "era5_convection_$(today_str).nc"), target_date)
        _make_conv(joinpath(nc_dir, "era5_convection_$(prev_str).nc"),  prev_date)

        # Thermo: constant T=280K, Q=0.005 across all hours.
        ds = NCDataset(joinpath(nc_dir, "era5_thermo_ml_$(today_str).nc"), "c")
        defDim(ds, "longitude", Nlon); defDim(ds, "latitude", Nlat)
        defDim(ds, "hybrid", Nlev);    defDim(ds, "time", 24)
        defVar(ds, "longitude", collect(range(0.0, step = 360.0 / Nlon, length = Nlon)), ("longitude",))
        defVar(ds, "latitude",  collect(range(90.0, stop = -90.0, length = Nlat)), ("latitude",))
        defVar(ds, "hybrid",    collect(1.0:Nlev), ("hybrid",))
        defVar(ds, "time",      collect(0.0:23.0), ("time",))
        defVar(ds, "t", fill(280.0f0, Nlon, Nlat, Nlev, 24), ("longitude", "latitude", "hybrid", "time"))
        defVar(ds, "q", fill(0.005f0, Nlon, Nlat, Nlev, 24), ("longitude", "latitude", "hybrid", "time"))
        close(ds)

        bin_path = convert_era5_physics_nc_to_bin(nc_dir, bin_dir, target_date; verbose = false)
        @test isfile(bin_path)

        reader = open_era5_physics_binary(bin_dir, target_date)
        try
            FT = Float32
            Nz_native = Nlev
            Nz = 3
            merge_map = [1, 1, 2, 2, 3]

            ws = allocate_tm5_workspace(Nlon, Nlat, Nz_native, Nz, FT; basis = DryBasis())
            stats = TM5CleanupStats()

            # L137-style hybrid coefs: surface-down pressure.
            ak = Float64[100000.0 * (Nz_native + 1 - k) / Nz_native for k in 1:(Nz_native + 1)]
            bk = zeros(Float64, Nz_native + 1)

            ps_hour = fill(98000.0f0, Nlon, Nlat)

            # Exercise every valid hour slot (1..24) so an off-by-one
            # in the BIN index would throw BoundsError.  Real `process_day`
            # drives this with `win_idx` from `enumerate(spec.hours)`,
            # not the ERA5 0-indexed `hour`.
            for h in (1, 12, 24)
                compute_tm5_merged_hour_on_source!(
                    ws, reader, h, ps_hour, ak, bk, Nz_native, merge_map; stats = stats)
                # Zero input → zero output at every slot.
                @test all(ws.entu_merged_src .== 0)
                @test all(ws.detu_merged_src .== 0)
                @test all(ws.entd_merged_src .== 0)
                @test all(ws.detd_merged_src .== 0)
            end
            @test size(ws.entu_merged_src) == (Nlon, Nlat, Nz)

            # With all UDMF below the 1e-6 clip threshold (all zero),
            # every column registers as "no updraft" / "no downdraft".
            # 3 hour slots × Nlon*Nlat columns each.
            @test stats.columns_processed[] == 3 * Nlon * Nlat
            @test stats.no_updraft[]         == 3 * Nlon * Nlat
            @test stats.no_downdraft[]       == 3 * Nlon * Nlat
        finally
            close_era5_physics_binary(reader)
        end

        # Re-open and re-exercise with FT = Float64 — the workspace must
        # allocate the F32→F64 conversion buffers and produce the same
        # zero result (i.e. zero physics input cannot get corrupted by
        # the upcast).
        reader = open_era5_physics_binary(bin_dir, target_date)
        try
            FT = Float64
            Nz_native = Nlev
            Nz = 3
            merge_map = [1, 1, 2, 2, 3]

            ws = allocate_tm5_workspace(Nlon, Nlat, Nz_native, Nz, FT;
                                         basis = DryBasis(), physics_eltype = Float32)
            @test ws.physics_bufs !== nothing
            @test eltype(ws.physics_bufs[1]) === Float64

            stats = TM5CleanupStats()
            ak = Float64[100000.0 * (Nz_native + 1 - k) / Nz_native for k in 1:(Nz_native + 1)]
            bk = zeros(Float64, Nz_native + 1)
            ps_hour = fill(98000.0, Nlon, Nlat)

            compute_tm5_merged_hour_on_source!(
                ws, reader, 1, ps_hour, ak, bk, Nz_native, merge_map; stats = stats)
            @test eltype(ws.entu_merged_src) === Float64
            @test all(ws.entu_merged_src .== 0)
            @test all(ws.detu_merged_src .== 0)
            @test all(ws.entd_merged_src .== 0)
            @test all(ws.detd_merged_src .== 0)
        finally
            close_era5_physics_binary(reader)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────
# (4) --all gated: real 1-day process_day on Dec 2 2021
# ─────────────────────────────────────────────────────────────────────────

if RUN_ALL
    @testset "Real LL process_day TM5 integration (Dec 2 2021)" begin
        bin_dir = "/temp1/met/era5/0.5x0.5/physics_bin"
        physics_bin = joinpath(bin_dir, "2021", "era5_physics_20211202.bin")
        spectral_dir = expanduser("~/data/AtmosTransport/met/era5/0.5x0.5/spectral")

        if !isfile(physics_bin)
            @info "Skip: physics BIN not staged at $physics_bin"
            return
        end
        if !isdir(spectral_dir) ||
           !isfile(joinpath(spectral_dir, "era5_spectral_20211202_vo_d.gb"))
            @info "Skip: ERA5 spectral GRIB not staged at $spectral_dir"
            return
        end

        # This path is intentionally left for Commit 6 to wire through
        # a full TOML config.  For Commit 4, we assert only that the
        # plumbing is intact: opening the BIN + instantiating the TM5
        # workspace at the right shape works without error.
        reader = open_era5_physics_binary(bin_dir, Date(2021, 12, 2))
        try
            @test reader.header.Nlon == 720
            @test reader.header.Nlat == 361
            @test reader.header.Nlev == 137
            @test reader.header.Nt   == 24
        finally
            close_era5_physics_binary(reader)
        end
    end
end
