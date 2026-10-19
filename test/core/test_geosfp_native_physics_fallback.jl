#!/usr/bin/env julia

using Test
using Dates
using NCDatasets

import AtmosTransport
using .AtmosTransport.Preprocessing: GEOSFPSettings, open_day, close_day!,
                                      read_window!, allocate_raw_window,
                                      process_day, build_target_geometry,
                                      load_hybrid_coefficients
using .AtmosTransport.MetDrivers: TransportBinaryReader

const FP_NC = 4
const FP_NP = 6
const FP_NZ = 4

function _fp_coefs(nz::Int)
    nz == 4 || error("synthetic coefs only defined for nz=4")
    return [0.0, 500.0, 1000.0, 500.0, 0.0], [0.0, 0.05, 0.20, 0.55, 1.0]
end

function _write_fp_coefs!(path::String)
    A, B = _fp_coefs(FP_NZ)
    open(path, "w") do io
        write(io, """
            [metadata]
            n_levels = $(FP_NZ)
            n_interfaces = $(FP_NZ + 1)

            [coefficients]
            a = $(repr(A))
            b = $(repr(B))
            """)
    end
end

function _def_fp_3d!(ds, name, A; units="")
    v = defVar(ds, name, eltype(A), ("Xdim", "Ydim", "nf", "lev", "time");
               attrib = ["units" => units])
    v[:, :, :, :, :] = A
    return v
end

function _def_fp_2d!(ds, name, A; units="")
    v = defVar(ds, name, eltype(A), ("Xdim", "Ydim", "nf", "time");
               attrib = ["units" => units])
    v[:, :, :, :] = A
    return v
end

function _write_fp_hour!(root::String, date::Date, hour::Int; bottom_up=false)
    datestr = Dates.format(date, "yyyymmdd")
    daydir = joinpath(root, datestr)
    mkpath(daydir)
    fname = "GEOS.fp.asm.tavg_1hr_ctm_c$(lpad(FP_NC, 4, '0'))_v72.$(datestr)_$(lpad(hour, 2, '0'))30.V01.nc4"
    path = joinpath(daydir, fname)
    A, B = _fp_coefs(FP_NZ)
    delp = zeros(Float64, FP_NC, FP_NC, FP_NP, FP_NZ, 1)
    for k in 1:FP_NZ
        delp[:, :, :, k, 1] .= (A[k + 1] - A[k]) + (B[k + 1] - B[k]) * 100_000.0
    end
    # Encode the tavg hour, top-down level, and panel independently of PS/QV.
    mfxc = [Float64(hour * 10_000 + k * 100 + p)
            for i in 1:FP_NC, j in 1:FP_NC, p in 1:FP_NP, k in 1:FP_NZ, _ in 1:1]
    mfyc = @. -(2 * mfxc + 50)
    NCDataset(path, "c") do ds
        ds.dim["Xdim"] = FP_NC
        ds.dim["Ydim"] = FP_NC
        ds.dim["nf"] = FP_NP
        ds.dim["lev"] = FP_NZ
        ds.dim["time"] = 1
        _def_fp_3d!(ds, "DELP", bottom_up ? reverse(delp; dims=4) : delp; units = "Pa")
        _def_fp_3d!(ds, "MFXC", bottom_up ? reverse(mfxc; dims=4) : mfxc; units = "Pa m+2")
        _def_fp_3d!(ds, "MFYC", bottom_up ? reverse(mfyc; dims=4) : mfyc; units = "Pa m+2")
        _def_fp_2d!(ds, "PS", fill(100_000.0, FP_NC, FP_NC, FP_NP, 1); units = "Pa")
    end
    return path
end

# Real tavg CTM has no QV. Keep instantaneous PS/QV in separate HH00 files.
function _write_fp_inst_hour!(root::String, date::Date, hour::Int;
                              varying=false, bottom_up=false)
    datestr = Dates.format(date, "yyyymmdd")
    daydir = joinpath(root, datestr)
    mkpath(daydir)
    path = joinpath(daydir, "GEOS.fp.asm.inst_1hr_ctm_c$(lpad(FP_NC, 4, '0'))_v72.$(datestr)_$(lpad(hour, 2, '0'))00.V01.nc4")
    t = Dates.value(date - Date(2021, 12, 1)) * 24 + hour
    ps = [varying ? 98_000.0 + 100t + 10p + i + 0.1j : 100_000.0
          for i in 1:FP_NC, j in 1:FP_NC, p in 1:FP_NP, _ in 1:1]
    qv = [varying ? 0.001k + 0.0001p + 0.00001t : 0.01
          for i in 1:FP_NC, j in 1:FP_NC, p in 1:FP_NP, k in 1:FP_NZ, _ in 1:1]
    # Different endpoint units exercise the same units-aware path as GEOS-IT.
    hpa = varying && isodd(t)
    NCDataset(path, "c") do ds
        ds.dim["Xdim"] = FP_NC
        ds.dim["Ydim"] = FP_NC
        ds.dim["nf"] = FP_NP
        ds.dim["lev"] = FP_NZ
        ds.dim["time"] = 1
        _def_fp_2d!(ds, "PS", hpa ? ps ./ 100 : ps; units = hpa ? "hPa" : "Pa")
        _def_fp_3d!(ds, "QV", bottom_up ? reverse(qv; dims=4) : qv; units = "kg kg-1")
    end
    return path
end

function _write_fp_native_ctm_day!(root::String, date::Date; varying=false, bottom_up=false)
    for h in 0:23
        _write_fp_hour!(root, date, h; bottom_up)
        _write_fp_inst_hour!(root, date, h; varying, bottom_up)
    end
    # No next-day tavg: only the instantaneous endpoint is required.
    _write_fp_inst_hour!(root, date + Day(1), 0; varying, bottom_up)
end

@testset "GEOS-FP instantaneous hourly dry endpoints" begin
    for bottom_up in (false, true)
        mktempdir() do tmp
            date = Date(2021, 12, 1)
            _write_fp_native_ctm_day!(tmp, date; varying=true, bottom_up)
            coefs = joinpath(tmp, "coefs.toml")
            _write_fp_coefs!(coefs)
            settings = GEOSFPSettings(; root_dir=tmp, Nc=FP_NC, coefficients_file=coefs)
            @test settings.mass_flux_dt == 450.0
            # Exercise the flat-root search alongside per-day directories.
            inst0 = AtmosTransport.Preprocessing.geosfp_native_hourly_inst_path(settings, date, 0)
            mv(inst0, joinpath(tmp, basename(inst0)))
            handles = open_day(settings, date)
            try
                @test handles.orientation === (bottom_up ? :bottom_up : :top_down)
                @test AtmosTransport.Preprocessing._geos_next_endpoint_available(handles)
                raw = allocate_raw_window(settings; FT=Float64, Nz=FP_NZ)
                A, B = _fp_coefs(FP_NZ)
                for win in (1, 4, 24)
                    read_window!(raw, settings, handles, date, win)
                    for (t, masses, pressures, humidities) in
                        ((win-1, raw.m, raw.ps, raw.qv), (win, raw.m_next, raw.ps_next, raw.qv_next))
                        for p in 1:FP_NP
                            ps = [98_000.0 + 100t + 10p + i + 0.1j for i in 1:FP_NC, j in 1:FP_NC]
                            expected = [(1 - (0.001k + 0.0001p + 0.00001t)) *
                                        (A[k+1]-A[k] + (B[k+1]-B[k])*ps[i,j])
                                        for i in 1:FP_NC, j in 1:FP_NC, k in 1:FP_NZ]
                            @test masses[p] ≈ expected
                            @test pressures[p] ≈ dropdims(sum(expected; dims=3); dims=3)
                            @test humidities[p][1,1,:] ≈ [0.001k + 0.0001p + 0.00001t for k in 1:FP_NZ]
                            # Tavg PS=100 kPa and its DELP differ from both endpoints.
                            @test !isapprox(masses[p][1,1,end], (1-humidities[p][1,1,end]) * (A[end]-A[end-1] + (B[end]-B[end-1])*100_000.0))
                        end
                    end
                    # Window w uses tavg hour w-1. Both file orders must return
                    # top-down dry fluxes, scaled only by the 450 s dynamics step.
                    for p in 1:FP_NP
                        expected_am = [Float64((win - 1) * 10_000 + k * 100 + p) * inv(450.0)
                                       for i in 1:FP_NC, j in 1:FP_NC, k in 1:FP_NZ]
                        expected_bm = [Float64(-(2 * ((win - 1) * 10_000 + k * 100 + p) + 50)) * inv(450.0)
                                       for i in 1:FP_NC, j in 1:FP_NC, k in 1:FP_NZ]
                        @test raw.am[p] == expected_am
                        @test raw.bm[p] == expected_bm
                    end
                end
            finally
                close_day!(handles)
            end

            handles = open_day(settings, date; next_day_handle=false)
            try
                raw = allocate_raw_window(settings; FT=Float64, Nz=FP_NZ)
                @test_throws r"inst_1hr_ctm_c0004_v72.20211202_0000" read_window!(raw, settings, handles, date, 24)
            finally
                close_day!(handles)
            end

            for (missing_date, hour) in ((date, 7), (date+Day(1), 0))
                path = AtmosTransport.Preprocessing.geosfp_native_hourly_inst_path(settings, missing_date, hour)
                mv(path, path * ".hidden")
                try
                    @test_throws Regex("instantaneous hourly CTM endpoint file not found:.*" * basename(path)) open_day(settings, date)
                finally
                    mv(path * ".hidden", path)
                end
            end
        end
    end
end

function _def_ll_2d!(ds, name, value; units="")
    v = defVar(ds, name, Float64, ("lon", "lat", "time"); attrib = ["units" => units, "_FillValue" => 1.0e15])
    v[:, :, :] = fill(value, length(ds["lon"][:]), length(ds["lat"][:]), size(v, 3))
    return v
end

function _def_ll_3d!(ds, name, base; nlev::Int, units="")
    v = defVar(ds, name, Float64, ("lon", "lat", "lev", "time"); attrib = ["units" => units, "_FillValue" => 1.0e15])
    data = zeros(Float64, length(ds["lon"][:]), length(ds["lat"][:]), nlev, size(v, 4))
    for t in axes(data, 4)
        data[:, :, :, t] .= base + t
    end
    v[:, :, :, :] = data
    return v
end

function _write_fp_latlon_physics!(root::String, date::Date)
    mkpath(root)
    datestr = Dates.format(date, "yyyymmdd")
    lons = collect(0.0:45.0:315.0)
    lats = collect(90.0:-45.0:-90.0)

    a1 = joinpath(root, "GEOSFP.$(datestr).A1.025x03125.nc")
    NCDataset(a1, "c") do ds
        ds.dim["lon"] = length(lons)
        ds.dim["lat"] = length(lats)
        ds.dim["time"] = 24
        defVar(ds, "lon", lons, ("lon",))
        defVar(ds, "lat", lats, ("lat",))
        _def_ll_2d!(ds, "PBLH", 1000.0; units = "m")
        _def_ll_2d!(ds, "USTAR", 0.25; units = "m s-1")
        _def_ll_2d!(ds, "HFLUX", 80.0; units = "W m-2")
        _def_ll_2d!(ds, "T2M", 290.0; units = "K")
    end

    a3mste = joinpath(root, "GEOSFP.$(datestr).A3mstE.025x03125.nc")
    NCDataset(a3mste, "c") do ds
        ds.dim["lon"] = length(lons)
        ds.dim["lat"] = length(lats)
        ds.dim["lev"] = FP_NZ + 1
        ds.dim["time"] = 8
        defVar(ds, "lon", lons, ("lon",))
        defVar(ds, "lat", lats, ("lat",))
        defVar(ds, "lev", collect(1:(FP_NZ + 1)), ("lev",))
        _def_ll_3d!(ds, "CMFMC", 10.0; nlev = FP_NZ + 1, units = "kg m-2 s-1")
    end

    a3dyn = joinpath(root, "GEOSFP.$(datestr).A3dyn.025x03125.nc")
    NCDataset(a3dyn, "c") do ds
        ds.dim["lon"] = length(lons)
        ds.dim["lat"] = length(lats)
        ds.dim["lev"] = FP_NZ
        ds.dim["time"] = 8
        defVar(ds, "lon", lons, ("lon",))
        defVar(ds, "lat", lats, ("lat",))
        defVar(ds, "lev", collect(1:FP_NZ), ("lev",))
        _def_ll_3d!(ds, "DTRAIN", 20.0; nlev = FP_NZ, units = "kg m-2 s-1")
    end
    return a1, a3mste, a3dyn
end

@testset "GEOS-FP native C720-style physics fallback" begin
    tmp = mktempdir()
    ctm_root = joinpath(tmp, "ctm")
    physics_root = joinpath(tmp, "physics")
    _write_fp_native_ctm_day!(ctm_root, Date(2021, 12, 1))
    _write_fp_latlon_physics!(physics_root, Date(2021, 12, 1))
    coef_path = joinpath(tmp, "coefs.toml")
    _write_fp_coefs!(coef_path)

    settings = GEOSFPSettings(
        root_dir = ctm_root,
        Nc = FP_NC,
        coefficients_file = coef_path,
        include_surface = true,
        include_convection = true,
        physics_dir = physics_root,
        physics_layout = :latlon_025,
    )

    handles = open_day(settings, Date(2021, 12, 1))
    try
        @test handles.orientation === :top_down
        @test handles.next_ctm_i1 !== nothing
        raw = allocate_raw_window(settings; FT = Float64, Nz = FP_NZ)
        read_window!(raw, settings, handles, Date(2021, 12, 1), 4)
        @test all(isapprox.(raw.surface.pblh[1], 1000.0; rtol = 1e-12))
        @test all(isapprox.(raw.surface.ustar[1], 0.25; rtol = 1e-12))
        @test all(isapprox.(raw.surface.hflux[1], 80.0; rtol = 1e-12))
        @test all(isapprox.(raw.surface.t2m[1], 290.0; rtol = 1e-12))
        # Window 4 binds to 3-hourly record 2; qv=0.01, so dry-basis factor is 0.99.
        @test all(isapprox.(raw.cmfmc[1], (10.0 + 2.0) * 0.99; rtol = 1e-12))
        @test all(isapprox.(raw.dtrain[1], (20.0 + 2.0) * 0.99; rtol = 1e-12))
    finally
        close_day!(handles)
    end

    grid = build_target_geometry(
        Dict{String,Any}("type" => "cubed_sphere", "Nc" => FP_NC,
                         "panel_convention" => "geos_native"),
        Float64,
    )
    vertical = (merged_vc = load_hybrid_coefficients(coef_path),
                Nz = FP_NZ, Nz_native = FP_NZ)
    out = joinpath(tmp, "geosfp_with_physics.bin")
    process_day(Date(2021, 12, 1), grid, settings, vertical;
                out_path = out, dt_met_seconds = 3600.0,
                FT = Float64, mass_basis = :dry, replay_tol = 1e-12)

    reader = TransportBinaryReader(out; FT = Float64)
    try
        for section in (:pblh, :ustar, :pbl_hflux, :t2m, :cmfmc, :dtrain)
            @test section in reader.header.payload_sections
        end
    finally
        close(reader)
    end
end
