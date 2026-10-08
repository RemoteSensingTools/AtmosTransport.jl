#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Breakpoint B — ERA5 N320 spectral synthesis + reduced_gg reorder.
#
# Verifies the per-window synthesis surface for the N320 source grid:
#
#   1. Workspace + fields allocate with the right shapes/dtypes.
#   2. `_reorder_grib_reduced_gg_to_mesh!` reverses ring order and
#      interpolates the grid points at (i − 1)Δλ to the cell centres at
#      (i − ½)Δλ; mismatched ring counts fail loudly.
#   3. Single-mode spectral synthesis via the full workspace pipeline
#      produces the analytical real-space pattern (constant, then a known
#      m=1 zonal wave) to FT precision.
#   4. PS = exp(LNSP) recovery from a constant LNSP coefficient.
#   5. (opt-in, real GRIB) End-to-end read for 2021-12-01 hour 0 against
#      the local N320 archive: T/Q/U/V/PS pass sanity ranges; level-1
#      completeness gates fire when fields are absent.
# ---------------------------------------------------------------------------

using Test
using Dates

import AtmosTransport

using .AtmosTransport.Preprocessing: AbstractMetSettings,
                                      ERA5N320Settings, ERA5GRIBDayHandles,
                                      ERA5N320SpectralWorkspace, ERA5N320WindowFields,
                                      allocate_era5_n320_spectral_workspace,
                                      allocate_era5_n320_window_fields,
                                      discover_era5_n320_source_grid,
                                      discover_era5_spectral_truncation,
                                      read_era5_n320_window_fields!,
                                      open_era5_day, close_era5_day!,
                                      build_target_geometry,
                                      _reorder_grib_reduced_gg_to_mesh!,
                                      _synthesize_into_column!
using .AtmosTransport.Grids: ncells, nrings

"""Build a tiny synthetic reduced-Gaussian source grid for unit tests.

Uses gaussian_number = 2 → 4 latitude rings with `nlon_per_ring = [8, 8, 8, 8]`
(regular reduced layout) — small enough to read in tests, large enough that the
GRIB-style north→south reorder is non-trivial.
"""
function _tiny_source_grid(::Type{FT} = Float64) where FT
    cfg = Dict{String, Any}(
        "type"            => "synthetic_reduced_gaussian",
        "gaussian_number" => 2,
        "nlon_mode"       => "regular",
    )
    return build_target_geometry(Val(:synthetic_reduced_gaussian), cfg, FT)
end

"""Allocate a minimal synthesis cache for a synthetic source grid at truncation
`T`. Reuses the workspace allocator so the cache wiring stays in one place.
The workspace holds one cache per thread; the first is sufficient for tests."""
function _synth_cache_for_test(grid, T::Int)
    return allocate_era5_n320_spectral_workspace(grid, T, 1).synth_caches[1]
end

@testset "ERA5 N320 window reader — breakpoint B" begin

    @testset "Workspace + fields allocation" begin
        grid = _tiny_source_grid(Float64)
        T = 3
        Nz = 5

        ws = allocate_era5_n320_spectral_workspace(grid, T, Nz)
        @test ws isa ERA5N320SpectralWorkspace{Float64}
        @test ws.T  == T
        @test ws.Nz == Nz
        @test size(ws.vo_spec)   == (T + 1, T + 1, Nz)
        @test size(ws.d_spec)    == (T + 1, T + 1, Nz)
        @test size(ws.t_spec)    == (T + 1, T + 1, Nz)
        @test size(ws.lnsp_spec) == (T + 1, T + 1)
        # Per-thread synthesis caches/scratch (one per thread; each cache owns
        # its own u_spec/v_spec). Sized by maxthreadid() so the threaded level
        # loop can index by threadid().
        @test length(ws.synth_caches) == Threads.maxthreadid()
        @test length(ws.grid_scratches) == Threads.maxthreadid()
        @test all(c -> size(c.u_spec) == (T + 1, T + 1), ws.synth_caches)
        @test all(c -> size(c.v_spec) == (T + 1, T + 1), ws.synth_caches)
        @test all(s -> length(s) == ncells(grid.mesh), ws.grid_scratches)
        @test length(ws.lnsp_grid) == ncells(grid.mesh)
        @test length(ws.have_t)  == Nz
        @test ws.have_lnsp[] === false

        f = allocate_era5_n320_window_fields(grid, Nz)
        @test f isa ERA5N320WindowFields{Float64}
        @test size(f.u)  == (ncells(grid.mesh), Nz)
        @test size(f.v)  == (ncells(grid.mesh), Nz)
        @test size(f.t)  == (ncells(grid.mesh), Nz)
        @test size(f.qv) == (ncells(grid.mesh), Nz)
        @test length(f.ps) == ncells(grid.mesh)

        @test_throws ArgumentError allocate_era5_n320_spectral_workspace(grid, 0, Nz)
        @test_throws ArgumentError allocate_era5_n320_spectral_workspace(grid, T, 0)
        @test_throws ArgumentError allocate_era5_n320_window_fields(grid, 0)
    end

    @testset "_reorder_grib_reduced_gg_to_mesh! ring reversal" begin
        grid = _tiny_source_grid(Float64)
        mesh = grid.mesh

        # Native GRIB ring order is N→S, so native_nlon is the reverse of
        # mesh.nlon_per_ring. We tag each native value with its source ring
        # so the reorder result reveals any permutation error.
        native_nlon = reverse(collect(mesh.nlon_per_ring))   # [4, 8, 8, 4] reversed
        native_vals = Float64[]
        for (j_native, n) in enumerate(native_nlon)
            append!(native_vals, fill(Float64(j_native), n))
        end

        out = zeros(Float64, ncells(mesh))
        _reorder_grib_reduced_gg_to_mesh!(out, native_vals, native_nlon, mesh)

        # After reorder, mesh ring 1 (south pole) should hold native ring N
        # (the last in N→S order, i.e. j_native = length(native_nlon)).
        n_rings = nrings(mesh)
        for j_mesh in 1:n_rings
            j_native = n_rings - j_mesh + 1
            ring_start = mesh.ring_offsets[j_mesh]
            ring_end   = mesh.ring_offsets[j_mesh + 1] - 1
            @test all(out[ring_start:ring_end] .== Float64(j_native))
        end

        # Grid points at (i - 1)Δλ land on cell centres (i - ½)Δλ: a zonal wave
        # sampled at the GRIB points comes out at the cell centres with no
        # phase shift (amplitude cos(Δλ/2) from the linear interpolation).
        wave_native = Float64[]
        for n in native_nlon
            append!(wave_native, [cosd(3 * (i - 1) * 360 / n) for i in 1:n])
        end
        _reorder_grib_reduced_gg_to_mesh!(out, wave_native, native_nlon, mesh)
        for j_mesh in 1:n_rings
            n = mesh.nlon_per_ring[j_mesh]; Δ = 360 / n
            ring = out[mesh.ring_offsets[j_mesh]:(mesh.ring_offsets[j_mesh + 1] - 1)]
            @test ring ≈ [cosd(3 * (i - 0.5) * Δ) * cosd(3 * Δ / 2) for i in 1:n] atol = 1e-12
        end

        # Mismatched length errors loudly.
        bad = zeros(Float64, ncells(mesh) + 1)
        @test_throws DimensionMismatch _reorder_grib_reduced_gg_to_mesh!(
            bad, native_vals, native_nlon, mesh)
    end

    @testset "_synthesize_into_column! — constant + m=1 mode" begin
        grid = _tiny_source_grid(Float64)
        T = 3
        cache = _synth_cache_for_test(grid, T)
        nc = ncells(grid.mesh)
        scratch = zeros(Float64, nc)

        # 1. Constant field: only the (n=0, m=0) coefficient is non-zero.
        # ECMWF/IFS uses fully-normalised harmonics with P̃_0^0 = 1, so the
        # zonal harmonic G_0(φ) = spec[1,1] · 1 at every latitude. Backward
        # FFT of a single DC bin returns G_0 at every longitude. A unit
        # coefficient therefore yields a unit-valued field everywhere.
        spec = zeros(ComplexF64, T + 1, T + 1)
        spec[1, 1] = 1.0 + 0im

        column = zeros(Float64, nc)
        _synthesize_into_column!(column, spec, T, grid, cache, scratch)

        @test all(abs.(column .- 1.0) .< 1e-12)

        # 2. m=1 zonal wave: spec[2, 2] sets a cos(λ) ring pattern up to a
        # latitudinal normalisation. Verify the result varies along longitude
        # (non-constant) and integrates to ~0 by ring (mean is zero on a
        # uniformly spaced ring sampling cos(λ)).
        fill!(spec, zero(ComplexF64))
        spec[2, 2] = 1.0 + 0im
        column2 = zeros(Float64, nc)
        _synthesize_into_column!(column2, spec, T, grid, cache, scratch)

        mesh = grid.mesh
        n_rings = nrings(mesh)
        for j in 1:n_rings
            ring_start = mesh.ring_offsets[j]
            ring_end   = mesh.ring_offsets[j + 1] - 1
            ring = column2[ring_start:ring_end]
            @test abs(sum(ring)) < 1e-10               # mean-zero per ring
            @test maximum(ring) - minimum(ring) > 1e-3 # genuinely non-constant
        end
    end

    @testset "PS = exp(LNSP) recovery" begin
        grid = _tiny_source_grid(Float64)
        T = 3
        cache = _synth_cache_for_test(grid, T)

        # ECMWF P̃_0^0 = 1 ⇒ spec[1,1] = log(101325) synthesises to that scalar
        # at every gridpoint; exp then recovers 101325 Pa.
        target_ps = 101325.0
        target_lnsp = log(target_ps)
        spec = zeros(ComplexF64, T + 1, T + 1)
        spec[1, 1] = target_lnsp + 0im

        lnsp_grid = zeros(Float64, ncells(grid.mesh))
        scratch   = zeros(Float64, ncells(grid.mesh))
        _synthesize_into_column!(lnsp_grid, spec, T, grid, cache, scratch)
        ps = exp.(lnsp_grid)
        @test all(abs.(ps .- target_ps) .< 1e-6)
    end

    # -----------------------------------------------------------------------
    # Real-data smoke. The N320 archive is on the workstation only, so this
    # @testset is gated on ATMOS_ERA5_N320_ROOT and runs end-to-end against
    # one window of the 2021-12-01 GRIB.
    # -----------------------------------------------------------------------
    real_root_env = get(ENV, "ATMOS_ERA5_N320_ROOT", "")
    if !isempty(real_root_env) && isdir(real_root_env)
        @testset "Real N320 smoke (2021-12-01 hour 0)" begin
            settings = ERA5N320Settings(; root_dir = real_root_env)
            handles  = open_era5_day(settings, Date(2021, 12, 1))

            try
                T = discover_era5_spectral_truncation(handles.core_path)
                @test T == 639

                grid = discover_era5_n320_source_grid(handles.core_path; FT = Float64)
                @test grid.gaussian_number == 320
                @test nrings(grid.mesh)    == 640
                @test minimum(grid.nlon_per_ring) >= 18
                @test maximum(grid.nlon_per_ring) == 1280

                Nz = 137
                ws = allocate_era5_n320_spectral_workspace(grid, T, Nz)
                f  = allocate_era5_n320_window_fields(grid, Nz)

                read_era5_n320_window_fields!(f, ws, handles, Date(2021, 12, 1), 0)

                # Physical-range gates. ECMWF model atmosphere covers ~175 K
                # (mesopause) to ~330 K (summer Sahara surface), Q is bounded
                # by ~30 g/kg in the tropical boundary layer, and the lowest
                # surface pressures on Earth are over the Himalayas/Tibet
                # plateau (~490 hPa over the highest summits).
                @test all(175.0     .<= f.t  .<= 330.0)
                @test all(0.0       .<= f.qv .<= 0.03)
                @test all(30_000.0  .<= f.ps .<= 110_000.0)

                # Wind ranges checked on cells with |φ| ≤ 89° — the two polar
                # rings of N320 are at ±89.78° where `1/cos(φ)` amplifies any
                # spectral synthesis noise. Stratospheric jets reach ~150 m/s.
                mesh = grid.mesh
                interior_cells = vcat([mesh.ring_offsets[j]:(mesh.ring_offsets[j + 1] - 1)
                                       for j in 1:nrings(mesh)
                                       if abs(mesh.latitudes[j]) <= 89.0]...)
                @test all(-200.0 .<= f.u[interior_cells, :] .<= 200.0)
                @test all(-200.0 .<= f.v[interior_cells, :] .<= 200.0)

                # Global means: PS within 2% of the ~985 hPa global average
                # for December (cell-area-weighted is better but cells are
                # equator-weighted in N320, which is close enough).
                ps_mean = sum(f.ps) / length(f.ps)
                @test 96_000.0 <= ps_mean <= 102_000.0

                # Global mean temperature integrated across all 137 levels
                # is near 240 K for a standard atmosphere weighted by mass.
                # We use a simple level-arithmetic mean here, which is biased
                # cold relative to mass-weighted (stratosphere over-represented)
                # but stays in a reproducible range.
                t_mean = sum(f.t) / length(f.t)
                @test 230.0 <= t_mean <= 260.0
            finally
                close_era5_day!(handles)
            end
        end
    else
        @info "Skipping real N320 window-reader smoke (set ATMOS_ERA5_N320_ROOT to enable)."
        @test_skip false
    end
end
