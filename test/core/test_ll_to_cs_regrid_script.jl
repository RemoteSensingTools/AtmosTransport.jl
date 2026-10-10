#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# test_ll_to_cs_regrid_script.jl — plan 40 Commit 3
#
# Exercises `regrid_ll_binary_to_cs` end-to-end: writes a tiny LL v4 fixture,
# regrids it to a C4 cubed-sphere binary, and verifies
#   1. the output file exists and its header is readable by
#      `TransportBinaryReader` / `inspect_binary`,
#   2. unchanged windows retain negligible `cm`, while explicit endpoint
#      mass tendencies diagnose bounded vertical `cm`,
#   3. the total CS air mass matches the LL source to within conservative-
#      regrid tolerance.
#
# This is the unit-level validation the existing `regrid_ll_binary_to_cs`
# function has lacked: today it is only exercised transitively through
# `process_day`. The thin CLI wrapper
# `scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl` is the
# deployment surface; this test locks in the numerical contract.
# ---------------------------------------------------------------------------

using Test
using JSON3
using Dates: Date

using AtmosTransport
using .AtmosTransport.Preprocessing: regrid_ll_binary_to_cs, build_target_geometry,
                                      PreprocessorRunCache, CubedSphereTargetGeometry
using .AtmosTransport.MetDrivers: load_window!, has_surface

# LL fixture: small but non-trivial. Pressure varies with latitude so
# `recover_ll_cell_center_winds!` sees a genuine ∇p field, and air mass
# varies with B-coefficient so the regridded totals aren't all identical.
function _ll_fixture_binary(path::AbstractString;
                            FT::Type{<:AbstractFloat} = Float64,
                            Nx::Int = 24, Ny::Int = 13, Nz::Int = 4,
                            nwindow::Int = 2,
                            final_dm_fraction::Real = 0,
                            surface_uniform = nothing,
                            tm5_uniform::Real = 0,
                            am_uniform::Real = 0,
                            bm_uniform::Real = 0,
                            radius::Real = AtmosTransport.Parameters.IFS_EARTH_RADIUS)
    # Default: the preprocessors' sphere, which the CS target shares.
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny, radius)
    A_ifc = FT[0, 2500, 5000, 7500, 10000]
    B_ifc = FT[0, 0.1, 0.3, 0.6, 1.0]
    vertical = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT, radius)

    # ps varies with latitude: lower at poles, higher at equator.
    ps = Array{FT}(undef, Nx, Ny)
    for j in 1:Ny
        φ = FT(mesh.φᶜ[j])
        ps_col = FT(101_300) - FT(2_000) * FT(sind(φ)^2)
        for i in 1:Nx
            ps[i, j] = ps_col
        end
    end

    # m[i,j,k] = (A[k+1]-A[k]) + (B[k+1]-B[k]) * ps[i,j], divided by g ≈ 9.81.
    g = FT(9.81)
    m = Array{FT}(undef, Nx, Ny, Nz)
    for k in 1:Nz
        dA = A_ifc[k + 1] - A_ifc[k]
        dB = B_ifc[k + 1] - B_ifc[k]
        for j in 1:Ny, i in 1:Nx
            m[i, j, k] = (dA + dB * ps[i, j]) / g
        end
    end

    # Start with zero fluxes — tests the core regrid + balance + cm-diagnose
    # path without committing to a particular wind field. Windows with no
    # endpoint mass change keep `cm` at solver tolerance; the final optional
    # `dm` perturbation should diagnose a bounded nonzero vertical `cm`.
    am = fill(FT(am_uniform), Nx + 1, Ny, Nz)
    bm = fill(FT(bm_uniform), Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)

    dm_final = zeros(FT, Nx, Ny, Nz)
    if final_dm_fraction != 0
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            dm_final[i, j, k] =
                FT(final_dm_fraction) * m[i, j, k] *
                sin(2π * FT(i - 1) / FT(Nx)) *
                cospi(FT(j - 1) / FT(max(Ny - 1, 1))) *
                FT(k / Nz)
        end
        for k in 1:Nz
            level_mean = sum(@view dm_final[:, :, k]) / (Nx * Ny)
            @views dm_final[:, :, k] .-= level_mean
        end
    end

    # Optional TM5 sections (uniform per-window) so the regrid can be
    # validated as a near-identity on a uniform field — conservative
    # regrid + LL→CS → CS uniformity at machine precision.
    tm5_block = if tm5_uniform != 0
        u = fill(FT(tm5_uniform), Nx, Ny, Nz)
        (entu = u, detu = u, entd = u, detd = u)
    else
        nothing
    end
    surface_block = if surface_uniform !== nothing
        vals = surface_uniform
        (pblh = fill(FT(vals.pblh), Nx, Ny),
         ustar = fill(FT(vals.ustar), Nx, Ny),
         hflux = fill(FT(vals.hflux), Nx, Ny),
         t2m = fill(FT(vals.t2m), Nx, Ny))
    else
        nothing
    end

    windows = Vector{NamedTuple}(undef, nwindow)
    for win in 1:nwindow
        base = (
            m = copy(m),
            am = am,
            bm = bm,
            cm = cm,
            ps = ps,
            dm = win < nwindow ? zeros(FT, Nx, Ny, Nz) : copy(dm_final),
        )
        win_nt = base
        surface_block !== nothing && (win_nt = merge(win_nt, (surface = surface_block,)))
        tm5_block !== nothing && (win_nt = merge(win_nt, (tm5_fields = tm5_block,)))
        windows[win] = win_nt
    end

    write_transport_binary(path, grid, windows;
                           FT = FT,
                           dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0,
                           steps_per_window = 2,
                           mass_basis = :dry,
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant,
                           delta_semantics = :forward_window_endpoint_difference)
    return (; Nx, Ny, Nz, nwindow,
            final_dm_fraction = Float64(final_dm_fraction),
            window_totals = fill(sum(m), nwindow))
end

@testset "plan 40 Commit 3 — regrid_ll_binary_to_cs end-to-end" begin

    @testset "F64 static LL → C4 CS, 2 windows" begin
        mktempdir() do dir
            ll_path = joinpath(dir, "ll_fixture.bin")
            cs_path = joinpath(dir, "cs_fixture.bin")

            meta = _ll_fixture_binary(ll_path; FT = Float64,
                                       Nx = 24, Ny = 13, Nz = 4, nwindow = 2,
                                       final_dm_fraction = 1e-3)

            # Build a C4 CS target geometry with a tempdir regridder cache
            # so the test is hermetic and does not pollute the user's cache.
            cfg_grid = Dict{String, Any}(
                "Nc" => 4,
                "regridder_cache_dir" => joinpath(dir, "cr_cache"),
            )
            cs_grid = build_target_geometry(Val(:cubed_sphere), cfg_grid, Float64)

            # Timestep metadata now comes from the source header; mass_basis
            # defaults to "match source" (no silent relabeling).
            regrid_ll_binary_to_cs(ll_path, cs_grid, cs_path; FT = Float64)

            @test isfile(cs_path)

            # `inspect_binary` must round-trip a CS binary written by the
            # regridder. This also exercises the plan-40-review fix to
            # `_peek_grid_type` + the CS `binary_capabilities` method.
            caps = inspect_binary(cs_path; io = devnull)
            @test caps.grid_type === :cubed_sphere
            @test caps.mass_basis === :dry
            @test caps.advection === true
            @test caps.surface_pressure === true
            @test caps.tm5_convection === false
            @test caps.replay_gate === true

            # Numerical invariants: load via the CS reader and check the
            # total air mass and post-balance cm/m.
            reader = TransportBinaryReader(cs_path; FT = Float64)
            @test reader.header.geometry.Nc == 4
            @test reader.header.geometry.npanel == 6
            @test reader.header.nwindow == meta.nwindow
            @test reader.header.nlevel == meta.Nz
            @test has_flux_delta(reader)
            @test delta_semantics(reader) == :forward_window_endpoint_difference

            # For each window, probe cm/m on the interior panel and confirm
            # global mass agrees with the LL total within conservative-regrid
            # tolerance. We use the raw data segmented by `_cs_section_offset`
            # through the reader's own slicing helpers via a compact ad-hoc
            # walk — load_cs_transport_window! would be the canonical route
            # but is driver-owned; testing at the reader level keeps the
            # dependency minimal.
            Nc = reader.header.geometry.Nc
            Nz = reader.header.nlevel
            npanel = reader.header.geometry.npanel

            for win in 1:reader.header.nwindow
                # Offset within the window for the `m` section (first in
                # payload_sections for CS: [m, am, bm, cm, ps]).
                # elems_per_window × (win − 1) gets us to this window's base.
                base = reader.header.elems_per_window * (win - 1)
                n_m = npanel * Nc * Nc * Nz
                m_slice = @view reader.data[base + 1 : base + n_m]
                m_total_cs = sum(Float64, m_slice)

                # Conservative LL→CS regrid preserves total mass to the
                # weight-map resolution. Tolerance: 1e-6 relative.
                @test isapprox(m_total_cs, meta.window_totals[win];
                               rtol = 1e-6)

                # `cm` section sits after m + am + bm.
                n_am = npanel * (Nc + 1) * Nc * Nz
                n_bm = npanel * Nc * (Nc + 1) * Nz
                n_cm = npanel * Nc * Nc * (Nz + 1)
                cm_offset = base + n_m + n_am + n_bm
                cm_slice = @view reader.data[cm_offset + 1 : cm_offset + n_cm]
                max_abs_cm = maximum(abs, cm_slice)

                m_ref = Float64(maximum(m_slice))
                cm_ratio = max_abs_cm / m_ref
                if win < reader.header.nwindow
                    # No endpoint mass change: zero input fluxes should leave
                    # vertical flux at solver tolerance.
                    @test cm_ratio < 1e-10
                else
                    # The final source window carries an explicit layer mass
                    # endpoint perturbation. Column balance closes the column
                    # budget horizontally and `diagnose_cs_cm!` carries the
                    # remaining layer redistribution vertically.
                    @test cm_ratio > 0
                    @test cm_ratio < meta.final_dm_fraction
                end
            end

            close(reader)

            driver = TransportBinaryDriver(cs_path; FT = Float64, arch = CPU(),
                                                Hp = 1, validate_replay = true)
            close(driver)
        end
    end

    # TM5 carry-through: when the source binary has the four convection
    # sections, the regridder must pass them through to the CS output.
    # A uniform input field round-trips to a near-uniform output through
    # conservative regrid; any drop to zero would mean the writer didn't
    # see the regridded panels.
    @testset "TM5 sections carried LL→CS" begin
        mktempdir() do dir
            ll_path = joinpath(dir, "ll_fixture.bin")
            cs_path = joinpath(dir, "cs_fixture.bin")
            tm5_val = 0.05  # kg/m²/s, plausible mid-tropo updraft entrainment

            _ll_fixture_binary(ll_path; FT = Float64,
                                Nx = 24, Ny = 13, Nz = 4, nwindow = 2,
                                tm5_uniform = tm5_val)

            cfg_grid = Dict{String, Any}(
                "Nc" => 4,
                "regridder_cache_dir" => joinpath(dir, "cr_cache"),
            )
            cs_grid = build_target_geometry(Val(:cubed_sphere), cfg_grid, Float64)

            regrid_ll_binary_to_cs(ll_path, cs_grid, cs_path; FT = Float64)

            caps = inspect_binary(cs_path; io = devnull)
            @test caps.tm5_convection === true

            # Load window 1 via the CS reader's `load_window!` (which
            # surfaces TM5 fields as panel-tuples). A uniform LL input
            # round-trips to near-uniform CS output through conservative
            # regrid, so each panel cell should be close to `tm5_val`.
            reader = TransportBinaryReader(cs_path; FT = Float64)
            win = load_window!(reader, 1)
            @test haskey(win, :tm5_fields) && win.tm5_fields !== nothing
            for fld in (:entu, :detu, :entd, :detd)
                panels = getfield(win.tm5_fields, fld)
                @test length(panels) == reader.header.geometry.npanel
                @test all(size(p) == (4, 4, 4) for p in panels)
                vals = vcat(map(vec, panels)...)
                @test isapprox(maximum(vals), tm5_val; atol = 5e-3)
                @test isapprox(minimum(vals), tm5_val; atol = 5e-3)
            end
            close(reader)
        end
    end

    @testset "PBL surface sections carried LL→CS" begin
        mktempdir() do dir
            ll_path = joinpath(dir, "ll_fixture.bin")
            cs_path = joinpath(dir, "cs_fixture.bin")
            sfc = (pblh = 900.0, ustar = 0.4, hflux = 80.0, t2m = 289.0)

            _ll_fixture_binary(ll_path; FT = Float64,
                                Nx = 24, Ny = 13, Nz = 4, nwindow = 2,
                                surface_uniform = sfc)

            cfg_grid = Dict{String, Any}(
                "Nc" => 4,
                "regridder_cache_dir" => joinpath(dir, "cr_cache"),
            )
            cs_grid = build_target_geometry(Val(:cubed_sphere), cfg_grid, Float64)

            regrid_ll_binary_to_cs(ll_path, cs_grid, cs_path; FT = Float64)

            caps = inspect_binary(cs_path; io = devnull)
            @test caps.pbl_diffusion === true

            reader = TransportBinaryReader(cs_path; FT = Float64)
            @test has_surface(reader)
            @test :pbl_hflux in reader.header.payload_sections
            @test !(:hflux in reader.header.payload_sections)
            win = load_window!(reader, 1)
            @test haskey(win, :surface) && win.surface !== nothing
            for (fld, expected) in pairs(sfc)
                panels = getfield(win.surface, fld)
                @test length(panels) == reader.header.geometry.npanel
                @test all(size(p) == (4, 4) for p in panels)
                vals = vcat(map(vec, panels)...)
                @test isapprox(maximum(vals), expected; atol = 1e-8)
                @test isapprox(minimum(vals), expected; atol = 1e-8)
            end
            close(reader)
        end
    end

    @testset "steps_per_window override rescales reconstructed CS fluxes" begin
        mktempdir() do dir
            ll_path = joinpath(dir, "ll_fixture.bin")
            cs_default_path = joinpath(dir, "cs_default.bin")
            cs_override_path = joinpath(dir, "cs_override.bin")

            _ll_fixture_binary(ll_path; FT = Float64,
                                Nx = 24, Ny = 13, Nz = 4, nwindow = 2,
                                am_uniform = 100.0)

            cfg_grid = Dict{String, Any}(
                "Nc" => 4,
                "regridder_cache_dir" => joinpath(dir, "cr_cache"),
            )
            cs_grid = build_target_geometry(Val(:cubed_sphere), cfg_grid, Float64)
            run_cache = PreprocessorRunCache(CubedSphereTargetGeometry, Float64)

            regrid_ll_binary_to_cs(ll_path, cs_grid, cs_default_path;
                                   FT = Float64, run_cache = run_cache)
            @test length(run_cache.entries) == 1
            regrid_ll_binary_to_cs(ll_path, cs_grid, cs_override_path;
                                   FT = Float64, steps_per_window = 4,
                                   run_cache = run_cache)
            @test length(run_cache.entries) == 1

            function _horizontal_flux_l1(path)
                reader = TransportBinaryReader(path; FT = Float64)
                win = load_window!(reader, 1)
                steps = reader.header.steps_per_window
                total = sum(map(p -> sum(abs, p), win.am)) +
                        sum(map(p -> sum(abs, p), win.bm))
                close(reader)
                return steps, total
            end

            steps_default, flux_default = _horizontal_flux_l1(cs_default_path)
            steps_override, flux_override = _horizontal_flux_l1(cs_override_path)

            @test steps_default == 2
            @test steps_override == 4
            @test flux_default > 0
            @test isapprox(flux_override / flux_default, 0.5; rtol = 1e-6)
        end
    end

    # Regression: requesting an output mass_basis that differs from the
    # source must error. Without this guard the function silently relabels
    # dry bytes as moist (invariant-14 violation).
    @testset "basis-mismatch raises ArgumentError" begin
        mktempdir() do dir
            ll_path = joinpath(dir, "ll_fixture.bin")
            cs_path = joinpath(dir, "cs_fixture.bin")
            _ll_fixture_binary(ll_path; FT = Float64,
                                Nx = 24, Ny = 13, Nz = 4, nwindow = 2)

            cfg_grid = Dict{String, Any}(
                "Nc" => 4,
                "regridder_cache_dir" => joinpath(dir, "cr_cache"),
            )
            cs_grid = build_target_geometry(Val(:cubed_sphere), cfg_grid, Float64)

            # Source fixture is `:dry`; request `:moist` on the same data.
            @test_throws ArgumentError regrid_ll_binary_to_cs(
                ll_path, cs_grid, cs_path;
                FT = Float64, mass_basis = :moist)
        end
    end

    @testset "a skipped write-time replay gate is recorded and reported" begin
        Pre = AtmosTransport.Preprocessing
        @test Pre._with_replay_record(Dict{String, Any}(), true) == Dict{String, Any}()
        @test Pre._with_replay_record(Dict{String, Any}(), false) ==
              Dict{String, Any}("write_replay_check" => false)
        # `[numerics] write_replay_check` of a preprocessing config (default on).
        resolve(numerics) = Pre._resolve_write_replay_check(Dict{String, Any}("numerics" => numerics))
        @test Pre._resolve_write_replay_check(Dict{String, Any}())
        @test resolve(Dict{String, Any}("write_replay_check" => true))
        @test !resolve(Dict{String, Any}("write_replay_check" => false))
        err = try resolve(Dict{String, Any}("write_replay_check" => "no")); nothing catch e; e end
        @test err isa ArgumentError && contains(err.msg, "[numerics].write_replay_check")
        # Every writer takes the keyword; the MERRA-2 and ERA5 N320 adapters end in
        # `kwargs...` and must name it, or it would be silently dropped.
        # (The unsupported-pair fallback takes untyped settings.)
        writers = [m for m in methods(Pre.process_day) if length(m.sig.parameters) == 5 &&
                   m.sig.parameters[2] === Date && m.sig.parameters[4] !== Any]
        @test length(writers) == 6
        for m in writers
            @test :write_replay_check in Base.kwarg_decl(m)
        end
        for f in (Pre.process_merra2_to_cs_day, Pre.process_era5_n320_to_cs_day,
                  regrid_ll_binary_to_cs)
            @test all(m -> :write_replay_check in Base.kwarg_decl(m), methods(f))
        end
        mktempdir() do dir
            cfg_grid = Dict{String, Any}("Nc" => 4, "regridder_cache_dir" => joinpath(dir, "cr_cache"))
            cs_grid = build_target_geometry(Val(:cubed_sphere), cfg_grid, Float64)
            ll_path = joinpath(dir, "ll.bin")
            _ll_fixture_binary(ll_path)
            header_of(path) = (r = AtmosTransport.MetDrivers.TransportBinaryReader(path; FT = Float64);
                               h = r.header.raw_header; close(r); h)

            # Default: the gate ran, nothing is recorded (headers unchanged).
            on_path = joinpath(dir, "cs_on.bin")
            regrid_ll_binary_to_cs(ll_path, cs_grid, on_path; FT = Float64)
            @test !haskey(header_of(on_path), "write_replay_check")
            @test inspect_binary(on_path; io = devnull).write_replay_check

            # Gate off (cubed sphere): recorded, reported, and warned about.
            off_path = joinpath(dir, "cs_off.bin")
            regrid_ll_binary_to_cs(ll_path, cs_grid, off_path; FT = Float64,
                                   write_replay_check = false)
            @test header_of(off_path)["write_replay_check"] === false
            report = IOBuffer()
            @test !inspect_binary(off_path; io = report).write_replay_check
            @test occursin("write-time replay check disabled", String(take!(report)))
            @test_logs (:warn, r"write-time replay check disabled") match_mode = :any begin
                close(TransportBinaryDriver(off_path; FT = Float64, arch = CPU(), Hp = 1))
            end

            # The script flag reaches the options.
            script_mod = Module()
            Base.include(script_mod, joinpath(@__DIR__, "..", "..", "scripts", "preprocessing",
                                              "regrid_ll_transport_binary_to_cs.jl"))
            args = ["--input", ll_path, "--output", joinpath(dir, "x.bin"), "--Nc", "4"]
            parse_args = Base.invokelatest(getproperty, script_mod, :_parse_args)
            @test Base.invokelatest(parse_args, args).write_replay_check
            @test !Base.invokelatest(parse_args, vcat(args, "--no-write-replay-check")).write_replay_check

            # Output reuse: an existing binary is reused only when its replay
            # record matches the run's (an absent key means the gate ran).
            reuse_path = joinpath(dir, "reuse.bin")
            _ll_fixture_binary(reuse_path)
            r = AtmosTransport.MetDrivers.TransportBinaryReader(reuse_path; FT = Float64)
            nb = r.header.header_bytes
            close(r)
            function patch_header!(path, f)
                raw = open(io -> read(io, nb), path)
                h = JSON3.read(String(raw[1:findfirst(==(0x00), raw) - 1]), Dict{String, Any})
                f(h)
                json = Vector{UInt8}(JSON3.write(h))
                open(io -> write(io, json, zeros(UInt8, nb - length(json))), path, "r+")
                return h
            end
            # The reuse contract keys must exist in the header to compare.
            disk = patch_header!(reuse_path, h -> foreach(k -> get!(h, k, 0), Pre._OUTPUT_REUSE_CONTRACT_KEYS))
            sections = disk["payload_sections"]
            bytes = filesize(reuse_path)
            matches(expected) = first(Pre.existing_output_schema_matches(reuse_path, bytes, sections, expected))
            @test matches(copy(disk))                                              # absent / absent
            @test !matches(merge(copy(disk), Dict("write_replay_check" => false)))   # absent / false
            patch_header!(reuse_path, h -> h["write_replay_check"] = false)
            @test matches(merge(copy(disk), Dict("write_replay_check" => false)))    # false / false
            @test !matches(copy(disk))                                             # false / absent

            # A lat-lon binary with the record is reported the same way.
            ll_off = joinpath(dir, "ll_off.bin")
            _ll_fixture_binary(ll_off)
            reader = AtmosTransport.MetDrivers.TransportBinaryReader(ll_off; FT = Float64)
            nbytes = reader.header.header_bytes
            close(reader)
            raw = open(io -> read(io, nbytes), ll_off)
            header = JSON3.read(String(raw[1:findfirst(==(0x00), raw) - 1]), Dict{String, Any})
            header["write_replay_check"] = false
            json = Vector{UInt8}(JSON3.write(header))
            open(io -> write(io, json, zeros(UInt8, nbytes - length(json))), ll_off, "r+")
            @test !inspect_binary(ll_off; io = devnull).write_replay_check
            @test_logs (:warn, r"write-time replay check disabled") match_mode = :any begin
                close(TransportBinaryDriver(ll_off; FT = Float64, arch = CPU()))
            end
        end
    end

    @testset "source and target share one sphere" begin
        mktempdir() do dir
            cfg_grid = Dict{String, Any}("Nc" => 4, "regridder_cache_dir" => joinpath(dir, "cr_cache"))
            cs_grid = build_target_geometry(Val(:cubed_sphere), cfg_grid, Float64)

            # A source on another sphere is rejected.
            ll_path = joinpath(dir, "ll_earth.bin")
            _ll_fixture_binary(ll_path; radius = AtmosTransport.Parameters.EARTH_RADIUS)
            @test_throws ArgumentError regrid_ll_binary_to_cs(ll_path, cs_grid, joinpath(dir, "cs_earth.bin");
                                                              FT = Float64)

            # A source without the radius key was built on the IFS sphere.
            ll_path = joinpath(dir, "ll_keyless.bin")
            _ll_fixture_binary(ll_path; final_dm_fraction = 1e-3)
            reader = AtmosTransport.MetDrivers.TransportBinaryReader(ll_path; FT = Float64)
            nbytes = reader.header.header_bytes
            close(reader)
            raw = open(io -> read(io, nbytes), ll_path)
            header = JSON3.read(String(raw[1:findfirst(==(0x00), raw) - 1]), Dict{String, Any})
            delete!(header, "planet_radius_m")
            json = Vector{UInt8}(JSON3.write(header))
            open(io -> write(io, json, zeros(UInt8, nbytes - length(json))), ll_path, "r+")
            cs_path = joinpath(dir, "cs_keyless.bin")
            regrid_ll_binary_to_cs(ll_path, cs_grid, cs_path; FT = Float64)
            reader = AtmosTransport.MetDrivers.TransportBinaryReader(cs_path; FT = Float64)
            @test reader.header.planet_radius_m == AtmosTransport.Parameters.IFS_EARTH_RADIUS
            close(reader)
        end
    end

    # CLI integration: invoke the script through a fresh Julia process so
    # arg-parse, default handling, and the main() path are actually
    # exercised. Keeps the cost down by only running for the tiny fixture
    # and setting --cache-dir to the tempdir (hermetic).
    @testset "CLI wrapper dispatches to library" begin
        mktempdir() do dir
            ll_path = joinpath(dir, "ll_fixture.bin")
            cs_path = joinpath(dir, "cs_fixture.bin")
            _ll_fixture_binary(ll_path; FT = Float64,
                                Nx = 24, Ny = 13, Nz = 4, nwindow = 2)

            script = joinpath(@__DIR__, "..", "..", "scripts", "preprocessing",
                              "regrid_ll_transport_binary_to_cs.jl")
            project = dirname(Base.active_project())

            # Run synchronously, capture stdout+stderr for debugging on
            # failure. The process should exit 0 and produce the output.
            cmd = `$(Base.julia_cmd()) --project=$(project) $(script)
                   --input $(ll_path) --output $(cs_path) --Nc 4
                   --convention geos_native
                   --cache-dir $(joinpath(dir, "cr_cache"))`
            buf = IOBuffer()
            proc = run(pipeline(cmd; stdout = buf, stderr = buf);
                       wait = true)
            @test proc.exitcode == 0
            @test isfile(cs_path)

            # Sanity: the CLI-produced binary matches source basis.
            caps = inspect_binary(cs_path; io = devnull)
            @test caps.grid_type === :cubed_sphere
            @test caps.mass_basis === :dry
            reader = TransportBinaryReader(cs_path; FT = Float64)
            @test reader.header.geometry.panel_convention === :geos_native
            close(reader)
        end
    end

end
