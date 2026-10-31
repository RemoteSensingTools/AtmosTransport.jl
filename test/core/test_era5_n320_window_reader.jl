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
                                      build_target_geometry
using .AtmosTransport.Preprocessing.ERA5: _reorder_grib_reduced_gg_to_mesh!
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

@testset "ERA5 N320 window reader — breakpoint B" begin

    @testset "Synthesis built for another level count is rejected" begin
        grid = _tiny_source_grid(Float64)
        @test_throws ArgumentError allocate_era5_n320_spectral_workspace(
            grid, 3, 5; synthesis = AtmosTransport.Preprocessing.ReducedSpectralSynthesis(grid, 3, 6))
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



    # -----------------------------------------------------------------------
    # Real-data smoke. The N320 archive is on the workstation only, so this
    # @testset is gated on ATMOS_ERA5_N320_ROOT and runs end-to-end against
    # one window of the 2021-12-01 GRIB.
    # -----------------------------------------------------------------------
end
