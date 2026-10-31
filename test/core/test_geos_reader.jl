#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan indexed-baking-valiant Commit 3 — GEOS NetCDF reader
#
# Tests against a synthetic 6-panel C8 fixture with controlled values, so
# the assertions are exact (not data-dependent). Coverage:
#
#   1. read_window! produces both endpoints + window-integrated fluxes.
#   2. Level-flip turns bottom-up source into top-down output.
#   3. mass_flux_dt scaling: am[i] = MFXC[i]/dt and similarly for bm.
#   4. Endpoint dry-mass closure: Σ DELP_dry[k] == PS_dry exactly.
#   5. Window-edge plumbing: the right endpoint of window 1 equals the
#      left endpoint of window 2 (i.e. CTM_I1 hour 2 used for both).
#   6. detect_level_orientation auto-detect on bottom-up DELP.
# ---------------------------------------------------------------------------

using Test
using Dates
using NCDatasets
using Random
using Statistics: mean

import AtmosTransport
using .AtmosTransport.Preprocessing: GEOSITSettings, open_day, close_day!,
                                      open_geos_day, close_geos_day!,
                                      read_window!, allocate_raw_window,
                                      detect_level_orientation,
                                       endpoint_dry_mass!,
                                      windows_per_day, has_convection

const FT_TEST = Float64
const NC = 8         # synthetic C8
const NPANEL = 6
const NZ = 4         # synthetic 4-level
const NWIN = 2       # 2 windows = 3 instantaneous PS/QV slices on day; we test windows 1..2

# ---------------------------------------------------------------------------
# Helpers — write a synthetic GEOS-IT day under /tmp.
# ---------------------------------------------------------------------------

"""Synthetic hybrid coefficients (top-to-bottom) for `nz=4` layers.

Constructed so:

  * `A[TOA] = 0`, `A[surface] = 0`         ⇒ `ΣDELP = PS_total` exactly.
  * Non-linear B values                    ⇒ DELP varies layer-by-layer
                                              (so auto-detect can distinguish
                                              surface from TOA).
"""
function _synthetic_hybrid_coefs(nz::Int)
    nz == 4 || error("synthetic coefs only defined for nz=4 in this test")
    A = [0.0, 0.0, 0.0, 0.0, 0.0]              # Pa
    B = [0.0, 0.05, 0.20, 0.55, 1.0]
    return A, B
end

"""Write a Julia 4D field (Nc, Nc, 6, Nz, time) to a NetCDF variable, given dims."""
function _defvar3d!(ds, name, A; units::String="", attrib=Dict{String,Any}())
    v = defVar(ds, name, eltype(A),
               ("Xdim", "Ydim", "nf", "lev", "time");
               attrib = ["units" => units; collect(attrib)...])
    v[:, :, :, :, :] = A
    return v
end

function _defvar2d_xyz_t!(ds, name, A; units::String="", attrib=Dict{String,Any}())
    v = defVar(ds, name, eltype(A),
               ("Xdim", "Ydim", "nf", "time");
               attrib = ["units" => units; collect(attrib)...])
    v[:, :, :, :] = A
    return v
end

"""
Generate a synthetic GEOS-IT day under `dir`. Levels are stored
**bottom-to-top** (k=1 = surface) to mirror the real GEOS-IT convention so
the auto-detect + flip path is exercised. Writes CTM_A1.nc and CTM_I1.nc
covering 2 windows (3 instantaneous endpoints at time indices 1, 2, 3).

PS_total is uniformly 100_000 Pa (= 1000 hPa); QV is uniformly `qv_const`.
DELP follows the synthetic hybrid coords with ΔA = -25 Pa per layer and
ΔB = +0.25 per layer. MFXC and MFYC are filled with constant `mfxc_const`
and `mfyc_const` respectively.
"""
function _write_synthetic_geosit_day!(dir::String, datestr::String;
                                      qv_const::Float64 = 0.005,
                                      mfxc_const::Float64 = 50.0,
                                      mfyc_const::Float64 = 30.0,
                                      ntimes_a1::Int = 2,
                                      ntimes_i1::Int = 3)
    mkpath(dir)
    A_top_down, B_top_down = _synthetic_hybrid_coefs(NZ)
    # Bottom-up storage (reverse the half-level coefficient arrays semantically;
    # in practice we just produce DELP fields that are larger at k=1 = surface).
    delp_per_layer_bottom_up = Float64[]
    # In top-down convention: ΔA[k] = A[k+1]-A[k], ΔB[k] = B[k+1]-B[k]
    # In bottom-up storage: layer 1 is surface (large mass), layer Nz is TOA (tiny mass)
    for k in NZ:-1:1
        ΔA = A_top_down[k+1] - A_top_down[k]
        ΔB = B_top_down[k+1] - B_top_down[k]
        push!(delp_per_layer_bottom_up, ΔA + ΔB * 100_000.0)
    end

    # ---- CTM_A1: MFXC, MFYC, DELP, time ----
    a1_path = joinpath(dir, "GEOSIT.$(datestr).CTM_A1.C$(NC).nc")
    NCDataset(a1_path, "c") do ds
        ds.dim["Xdim"] = NC
        ds.dim["Ydim"] = NC
        ds.dim["nf"]   = NPANEL
        ds.dim["lev"]  = NZ
        ds.dim["time"] = ntimes_a1

        delp_bu = fill(0.0, NC, NC, NPANEL, NZ, ntimes_a1)
        for k in 1:NZ
            delp_bu[:, :, :, k, :] .= delp_per_layer_bottom_up[k]
        end
        _defvar3d!(ds, "DELP", delp_bu; units="Pa")

        mfxc = fill(mfxc_const, NC, NC, NPANEL, NZ, ntimes_a1)
        mfyc = fill(mfyc_const, NC, NC, NPANEL, NZ, ntimes_a1)
        _defvar3d!(ds, "MFXC", mfxc; units="Pa m+2")
        _defvar3d!(ds, "MFYC", mfyc; units="Pa m+2")

        # Bottom-up lev coordinate: lev[1] = surface, lev[end] = TOA.
        # GEOS-IT stamps `lev:positive = "down"` regardless of array orientation.
        defVar(ds, "lev", collect(1:NZ), ("lev",); attrib=["positive" => "down"])
    end

    # ---- CTM_I1: PS, QV, time ----
    i1_path = joinpath(dir, "GEOSIT.$(datestr).CTM_I1.C$(NC).nc")
    NCDataset(i1_path, "c") do ds
        ds.dim["Xdim"] = NC
        ds.dim["Ydim"] = NC
        ds.dim["nf"]   = NPANEL
        ds.dim["lev"]  = NZ
        ds.dim["time"] = ntimes_i1

        ps_hpa = fill(1000.0, NC, NC, NPANEL, ntimes_i1)
        _defvar2d_xyz_t!(ds, "PS", ps_hpa; units="hPa")

        qv = fill(qv_const, NC, NC, NPANEL, NZ, ntimes_i1)
        _defvar3d!(ds, "QV", qv; units="kg kg-1")

        defVar(ds, "lev", collect(1:NZ), ("lev",); attrib=["positive" => "down"])
    end

    return a1_path, i1_path
end

"""Write a synthetic 4-layer hybrid coefficients TOML at `path`."""
function _write_synthetic_coefs_toml!(path::String)
    A, B = _synthetic_hybrid_coefs(NZ)
    open(path, "w") do io
        write(io, """
            [metadata]
            n_levels = $(NZ)
            n_interfaces = $(NZ + 1)

            [coefficients]
            a = $(repr(A))
            b = $(repr(B))
            """)
    end
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "GEOS reader" begin

    tmpdir   = mktempdir()
    daydir   = joinpath(tmpdir, "20211201")
    nextdir  = joinpath(tmpdir, "20211202")
    a1_path1, i1_path1 = _write_synthetic_geosit_day!(daydir,  "20211201")
    a1_path2, i1_path2 = _write_synthetic_geosit_day!(nextdir, "20211202")
    coef_path = joinpath(tmpdir, "synthetic_coefs.toml")
    _write_synthetic_coefs_toml!(coef_path)

    # Override the GEOS-72 file with our 4-layer one inside the settings.
    settings = GEOSITSettings(
        root_dir = tmpdir, Nc = NC,
        mass_flux_dt = 450.0,
        coefficients_file = coef_path,
    )


    # Override windows_per_day for our 2-window synthetic via dispatch on
    # a custom subtype is overkill; instead we just test win_idx=1 directly,
    # since our fixture only covers 2 windows. The reader still expects the
    # NCDataset "time" dim ≥ win_idx + 1 for the right endpoint.

    @testset "open_day auto-detects bottom-up orientation" begin
        handles = open_day(settings, Date(2021,12,1))
        @test handles.orientation === :bottom_up
        @test handles.next_ctm_i1 !== nothing
        close_day!(handles)
        ds = NCDataset(a1_path1, "r")
        @test detect_level_orientation(ds) === :bottom_up
        close(ds)
    end



    @testset "read_window! produces both endpoints, top-down" begin
        handles = open_day(settings, Date(2021,12,1))
        try
            raw = allocate_raw_window(settings; FT=FT_TEST, Nz=NZ)
            read_window!(raw, settings, handles, Date(2021,12,1), 1)
            @test length(raw.m)       == 6
            @test length(raw.m_next)  == 6
            @test length(raw.ps)      == 6
            @test length(raw.ps_next) == 6
            @test size(raw.m[1])      == (NC, NC, NZ)
            @test size(raw.ps[1])     == (NC, NC)

            # After flip, k=1 is TOA → smallest DELP. k=NZ is surface → largest.
            d_toa = mean(raw.m[1][:, :, 1])
            d_sfc = mean(raw.m[1][:, :, NZ])
            @test d_toa < d_sfc

            # qv unchanged in magnitude (constant 0.005) but flipped to top-down.
            @test all(isapprox.(raw.qv[1],      0.005))
            @test all(isapprox.(raw.qv_next[1], 0.005))
        finally
            close_day!(handles)
        end
    end

    @testset "mass_flux_dt scaling: am = MFXC / mass_flux_dt (no humidity correction)" begin
        # GEOS-IT MFXC and MFYC are already DRY mass fluxes per GMAO; the
        # reader only scales by `1 / mass_flux_dt` and does NOT multiply by
        # (1 - qv). Multiplying by (1-qv) would double-dry, biasing transport
        # low (codex P2 finding 2026-04-24).
        handles = open_day(settings, Date(2021,12,1))
        try
            raw = allocate_raw_window(settings; FT=FT_TEST, Nz=NZ)
            read_window!(raw, settings, handles, Date(2021,12,1), 1)
            @test all(isapprox.(raw.am[1], 50.0 / 450.0; rtol = 1e-12))
            @test all(isapprox.(raw.bm[1], 30.0 / 450.0; rtol = 1e-12))
        finally
            close_day!(handles)
        end
    end

    @testset "endpoint dry-mass closure: Σ DELP_dry[k] == PS_dry" begin
        handles = open_day(settings, Date(2021,12,1))
        try
            raw = allocate_raw_window(settings; FT=FT_TEST, Nz=NZ)
            read_window!(raw, settings, handles, Date(2021,12,1), 1)
            for p in 1:6
                Σ_delp_dry = sum(raw.m[p], dims=3)[:, :, 1]      # (NC, NC)
                @test isapprox(Σ_delp_dry, raw.ps[p]; rtol = 1e-13)

                Σ_delp_dry_next = sum(raw.m_next[p], dims=3)[:, :, 1]
                @test isapprox(Σ_delp_dry_next, raw.ps_next[p]; rtol = 1e-13)
            end
        finally
            close_day!(handles)
        end
    end

    @testset "endpoint_dry_mass: ΣDELP_dry == PS_dry == (1-qv)*PS_total" begin
        # With uniform qv, PS_dry should equal (1-qv) * PS_total exactly.
        A, B = _synthetic_hybrid_coefs(NZ)
        vc = AtmosTransport.Grids.HybridSigmaPressure(A, B)
        ps_total = fill(100_000.0, 4, 4)
        qv = fill(0.005, 4, 4, NZ)
        delp_dry = similar(qv)
        ps_dry = similar(ps_total)
        endpoint_dry_mass!(delp_dry, ps_dry, ps_total, qv, vc)
        @test all(isapprox.(ps_dry, (1 - 0.005) * 100_000.0; rtol = 1e-13))
        Σ_delp_dry = sum(delp_dry, dims=3)[:, :, 1]
        @test isapprox(Σ_delp_dry, ps_dry; rtol = 1e-13)
    end

    @testset "right endpoint of window 1 == left endpoint of window 2" begin
        # Use two separate raw buffers so the in-place reader doesn't clobber
        # window 1's data on the window 2 call.
        handles = open_day(settings, Date(2021,12,1))
        try
            raw1 = allocate_raw_window(settings; FT=FT_TEST, Nz=NZ)
            raw2 = allocate_raw_window(settings; FT=FT_TEST, Nz=NZ)
            read_window!(raw1, settings, handles, Date(2021,12,1), 1)
            read_window!(raw2, settings, handles, Date(2021,12,1), 2)
            for p in 1:6
                @test isapprox(raw1.m_next[p],  raw2.m[p];  rtol = 1e-13)
                @test isapprox(raw1.ps_next[p], raw2.ps[p]; rtol = 1e-13)
                @test isapprox(raw1.qv_next[p], raw2.qv[p]; rtol = 1e-13)
            end
        finally
            close_day!(handles)
        end
    end
end
