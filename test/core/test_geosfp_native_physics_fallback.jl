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

function _def_ll_3d!(ds, name, base; nlev::Int, units="", lev_dim="lev")
    v = defVar(ds, name, Float64, ("lon", "lat", lev_dim, "time"); attrib = ["units" => units, "_FillValue" => 1.0e15])
    data = zeros(Float64, length(ds["lon"][:]), length(ds["lat"][:]), nlev, size(v, 4))
    for t in axes(data, 4)
        for k in axes(data, 3)
            # Values encode the archive index, independent of CTM orientation.
            data[:, :, k, t] .= base + t + k
        end
    end
    v[:, :, :, :] = data
    return v
end

function _write_fp_latlon_physics!(root::String, date::Date;
                                  bottom_up=true, nz=FP_NZ, dual_axes=false)
    mkpath(root)
    datestr = Dates.format(date, "yyyymmdd")
    lons = collect(0.0:45.0:315.0)
    lats = [89.9375, 89.75, 45.0, 0.0, -45.0, -89.75, -89.9375]

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
        cm_dim = dual_axes ? "ilev" : "lev"
        if dual_axes
            # Unused layer axis comes first and suggests the opposite order.
            ds.dim["lev"] = nz
            defVar(ds, "lev", collect(1:nz), ("lev",);
                   attrib=["positive" => (bottom_up ? "down" : "up")])
        end
        ds.dim[cm_dim] = nz + 1
        ds.dim["time"] = 8
        defVar(ds, "lon", lons, ("lon",))
        defVar(ds, "lat", lats, ("lat",))
        defVar(ds, cm_dim, collect(1:(nz + 1)), (cm_dim,))
        _def_ll_3d!(ds, "CMFMC", 10.0; nlev = nz + 1, units = "kg m-2 s-1", lev_dim=cm_dim)
    end

    a3dyn = joinpath(root, "GEOSFP.$(datestr).A3dyn.025x03125.nc")
    NCDataset(a3dyn, "c") do ds
        if dual_axes
            ds.dim["ilev"] = nz + 1
            defVar(ds, "ilev", collect(1:(nz + 1)), ("ilev",);
                   attrib=["positive" => (bottom_up ? "down" : "up")])
        end
        ds.dim["lon"] = length(lons)
        ds.dim["lat"] = length(lats)
        ds.dim["lev"] = nz
        ds.dim["time"] = 8
        defVar(ds, "lon", lons, ("lon",))
        defVar(ds, "lat", lats, ("lat",))
        defVar(ds, "lev", collect(1:nz), ("lev",))
        _def_ll_3d!(ds, "DTRAIN", 20.0; nlev = nz, units = "kg m-2 s-1")
    end
    i3 = joinpath(root, "GEOSFP.$(datestr).I3.025x03125.nc")
    NCDataset(i3, "c") do ds
        if dual_axes
            ds.dim["ilev"] = nz + 1
            defVar(ds, "ilev", collect(1:(nz + 1)), ("ilev",);
                   attrib=["positive" => (bottom_up ? "down" : "up")])
        end
        ds.dim["lon"] = length(lons)
        ds.dim["lat"] = length(lats)
        ds.dim["lev"] = nz
        ds.dim["time"] = 8
        defVar(ds, "lon", lons, ("lon",))
        defVar(ds, "lat", lats, ("lat",))
        defVar(ds, "lev", collect(1:nz), ("lev",))
        q = defVar(ds, "QV", Float64, ("lon", "lat", "lev", "time"))
        for k in 1:nz
            value = bottom_up ? 0.01 * 1000.0^(-(k-1)/(nz-1)) :
                                0.01 * 1000.0^(-(nz-k)/(nz-1))
            q[:, :, k, :] = fill(value,length(lons),length(lats),8)
        end
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
        @test handles.physics.level_orientation === :bottom_up
        for k in 1:(FP_NZ+1)
            @test all(isapprox.(raw.cmfmc[1][:,:,k], (10.0 + 2.0 + FP_NZ+2-k) * 0.99; rtol=1e-12))
        end
        for k in 1:FP_NZ
            @test all(isapprox.(raw.dtrain[1][:,:,k], (20.0 + 2.0 + FP_NZ+1-k) * 0.99; rtol=1e-12))
        end
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


@testset "OF-119 coordinate-aware physics interpolation" begin
    pp = AtmosTransport.Preprocessing
    lons = [-170.0, -80.0, 0.0, 45.0, 170.0] # nonuniform, seam spans 20 degrees
    lats = vcat(-89.9375, collect(-89.75:0.25:89.75), 89.9375)
    lat_ramp = [φ for λ in lons, φ in lats]
    lon_ramp = [λ for λ in lons, φ in lats]
    for λ in (-170.0, -100.0, -5.0, 32.52337311, 169.0),
        φ in (-90.0, -89.9, -89.8, -60.0, 0.0, 4.95077576, 89.8, 89.9, 90.0)
        @test pp._interp_regular_ll(lat_ramp, lons, lats, λ, φ) ≈ clamp(φ, first(lats), last(lats)) atol=3e-14
        @test pp._interp_regular_ll(lon_ramp, lons, lats, λ, φ) ≈ λ atol=3e-14
    end
    # Unwrap the field at the seam independently: 170..190 (= -170).
    seam = [λ == -170 ? 190.0 : λ for λ in lons, φ in lats]
    for λ in (175.0, 180.0, 185.0, -175.0, 535.0)
        expected = mod(λ-170, 360) + 170
        @test pp._interp_regular_ll(seam, lons, lats, λ, 4.9) ≈ expected
    end
    @test pp._interp_regular_ll(reshape([7.0],1,1), [0.0], [0.0], 720, -90) == 7
    mktempdir() do tmp
        path = joinpath(tmp,"coordinate_ramps.nc")
        # Odd, nonuniform longitudes with a non-centred origin exercise the
        # coordinate/field permutation, rather than assuming a half-grid roll.
        raw_lons = [0.0,40.0,150.0,220.0,350.0]
        raw_lats = reverse(lats)
        NCDataset(path,"c") do ds
            ds.dim["longitude"] = length(raw_lons)
            ds.dim["latitude"] = length(raw_lats)
            ds.dim["lev"] = 2
            ds.dim["time"] = 8
            defVar(ds,"longitude",raw_lons,("longitude",))
            defVar(ds,"latitude",raw_lats,("latitude",))
            latvar = defVar(ds,"latitude_ramp",Float64,("longitude","latitude","time"))
            latvar[:,:,:] = [φ for λ in raw_lons,φ in raw_lats,t in 1:8]
            lonvar = defVar(ds,"longitude_ramp",Float64,("longitude","latitude","lev","time"))
            lonvar[:,:,:,:] = [mod(λ+180,360)-180+k+100t for λ in raw_lons,φ in raw_lats,k in 1:2,t in 1:8]
        end
        NCDataset(path,"r") do ds
            xs,ys = pp._geosfp_latlon_axes(ds)
            latfield = pp._read_latlon_slice(ds,"latitude_ramp",4,Val(2),Float64)
            lonfield = pp._read_latlon_slice(ds,"longitude_ramp",4,Val(3),Float64)
            @test latfield == [φ for λ in xs,φ in ys]
            @test lonfield == [λ+k+200 for λ in xs,φ in ys,k in 1:2]
            for φ in (-89.9,4.95077576,89.9)
                @test pp._interp_regular_ll(latfield,xs,ys,32.52337311,φ) ≈ φ
                @test pp._interp_regular_ll(lonfield,xs,ys,32.52337311,φ,2) ≈ 234.52337311
            end
        end
    end
end

@testset "OF-119 independent physics/CTM orientations and humidity pairing" begin
    pp = AtmosTransport.Preprocessing
    date = Date(2021,12,1)
    for ctm_bottom_up in (false, true), physics_bottom_up in (false, true)
        mktempdir() do tmp
            ctm = joinpath(tmp,"ctm"); phys = joinpath(tmp,"physics")
            _write_fp_native_ctm_day!(ctm, date; varying=true, bottom_up=ctm_bottom_up)
            _write_fp_latlon_physics!(phys, date; bottom_up=physics_bottom_up)
            coefs = joinpath(tmp,"coefs.toml"); _write_fp_coefs!(coefs)
            settings = GEOSFPSettings(; root_dir=ctm, Nc=FP_NC, coefficients_file=coefs,
                include_convection=true, physics_dir=phys, physics_layout=:latlon_025)
            handles = open_day(settings,date)
            try
                raw = allocate_raw_window(settings; FT=Float64, Nz=FP_NZ)
                read_window!(raw,settings,handles,date,4)
                @test handles.physics.level_orientation === (physics_bottom_up ? :bottom_up : :top_down)
                for p in 1:6, k in 1:FP_NZ
                    src_k = physics_bottom_up ? FP_NZ+1-k : k
                    q = 0.001k + 0.0001p + 0.00001*3.5
                    @test raw.dtrain[p][1,1,k] ≈ (20+2+src_k)*(1-q)
                end
                for p in 1:6, k in 1:FP_NZ+1
                    src_k = physics_bottom_up ? FP_NZ+2-k : k
                    ka, kb = max(1,k-1), min(FP_NZ,k)
                    q = 0.001*(ka+kb)/2 + 0.0001p + 0.00001*3.5
                    @test raw.cmfmc[p][1,1,k] ≈ (10+2+src_k)*(1-q)
                end
            finally
                close_day!(handles)
            end
        end
    end
end

@testset "OF-119 L72 edge/layer formulas and validated CS order" begin
    pp = AtmosTransport.Preprocessing
    date = Date(2021,12,1)
    for layout in (:latlon_025, :cubed_sphere), bottom_up in (false,true)
        mktempdir() do tmp
            if layout === :latlon_025
                _write_fp_latlon_physics!(tmp,date; bottom_up, nz=72)
            else
                for (collection, name, nz, base) in (("A3mstE","CMFMC",73,10), ("A3dyn","DTRAIN",72,20))
                    path = joinpath(tmp,"GEOSFP.20211201.$collection.C4.nc")
                    NCDataset(path,"c") do ds
                        for (d,n) in (("Xdim",4),("Ydim",4),("nf",6),("lev",nz),("time",8))
                            ds.dim[d] = n
                        end
                        defVar(ds,"lev",collect(1:nz),("lev",);
                            attrib=["positive" => (bottom_up ? "up" : "down")])
                        data = [Float64(base+t+k) for i in 1:4,j in 1:4,p in 1:6,k in 1:nz,t in 1:8]
                        _def_fp_3d!(ds,name,data)
                    end
                end
            end
            settings = GEOSFPSettings(; root_dir=tmp, Nc=4, include_convection=true,
                physics_dir=tmp, physics_layout=layout)
            phys = pp._open_geosfp_physics_fallback(settings,date)
            try
                raw = allocate_raw_window(settings; FT=Float64, Nz=72)
                for p in 1:6,k in 1:72
                    raw.qv[p][:,:,k] .= 0.0001k
                    raw.qv_next[p][:,:,k] .= 0.0002k
                end
                # Deliberately opposite CTM orientation; raw humidity is already normalized.
                pp._read_geosfp_convection_window!(raw,phys,4,bottom_up ? :top_down : :bottom_up)
                for k in 1:73
                    src_k = bottom_up ? 74-k : k
                    q = 0.00015*(max(1,k-1)+min(72,k))/2
                    @test raw.cmfmc[1][1,1,k] ≈ (10+2+src_k)*(1-q)
                end
                for k in 1:72
                    src_k = bottom_up ? 73-k : k
                    @test raw.dtrain[1][1,1,k] ≈ (20+2+src_k)*(1-0.00015k)
                end
                if layout === :latlon_025
                    # Cached 3-D stencil must reproduce a geographic ramp at every target.
                    ramp = [φ for λ in phys.lons, φ in phys.lats, k in 1:72]
                    pp._interpolate_ll_to_panels!(raw.dtrain,ramp,phys)
                    @test all(p -> raw.dtrain[p][:,:,1] ≈ phys.target_lats[p], 1:6)
                    panels = ntuple(_ -> zeros(4,4),6)
                    pp._interpolate_ll_to_panels!(panels,ramp[:,:,1],phys)
                    @test all(p -> panels[p] ≈ phys.target_lats[p], 1:6)
                end
            finally
                pp.close_geosfp_physics!(phys)
            end
            if layout === :latlon_025
                path = joinpath(tmp,"GEOSFP.20211201.I3.025x03125.nc")
                NCDataset(path,"a") do ds
                    ds["QV"][:,:,:,:] = fill(0.001,size(ds["QV"]))
                end
                @test_throws r"Cannot detect GEOS-FP physics level order" pp._open_geosfp_physics_fallback(settings,date)
                rm(path)
                @test_throws r"Cannot validate GEOS-FP physics level order" pp._open_geosfp_physics_fallback(settings,date)
            else
                path = joinpath(tmp,"GEOSFP.20211201.A3dyn.C4.nc")
                NCDataset(path,"a") do ds
                    ds["lev"].attrib["positive"] = bottom_up ? "down" : "up"
                end
                @test_throws r"Conflicting GEOS-FP physics" pp._open_geosfp_physics_fallback(settings,date)
            end
        end
    end
end

@testset "OF-119 dual vertical axes use each field's dimension" begin
    pp = AtmosTransport.Preprocessing
    date = Date(2021, 12, 1)
    for bottom_up in (false, true)
        mktempdir() do tmp
            _, cm_path, dt_path = _write_fp_latlon_physics!(tmp, date;
                bottom_up, nz=72, dual_axes=true)
            q_path = joinpath(tmp, "GEOSFP.20211201.I3.025x03125.nc")
            settings = GEOSFPSettings(; root_dir=tmp, Nc=4, include_convection=true,
                physics_dir=tmp, physics_layout=:latlon_025)
            expected = bottom_up ? :bottom_up : :top_down
            positive = bottom_up ? "up" : "down"
            opposite = bottom_up ? "down" : "up"
            phys = pp._open_geosfp_physics_fallback(settings, date)
            try
                @test phys.level_orientation === expected
                # Exercise all 73 edges and 72 layers through the actual reader.
                raw = allocate_raw_window(settings; FT=Float64, Nz=72)
                for p in 1:6
                    fill!(raw.qv[p], 0)
                    fill!(raw.qv_next[p], 0)
                end
                pp._read_geosfp_convection_window!(raw, phys, 1, :top_down)
                for k in 1:73
                    @test raw.cmfmc[1][1,1,k] ≈ 11 + (bottom_up ? 74-k : k)
                end
                for k in 1:72
                    @test raw.dtrain[1][1,1,k] ≈ 21 + (bottom_up ? 73-k : k)
                end
            finally
                pp.close_geosfp_physics!(phys)
            end

            # The unused axes now agree with QV. They cannot hide conflicting
            # physical metadata on any of the three fields' actual axes.
            for (path, axis) in ((cm_path, "lev"), (dt_path, "ilev"), (q_path, "ilev"))
                NCDataset(path, "a") do ds
                    ds[axis].attrib["positive"] = positive
                end
            end
            for (path, axis) in ((cm_path, "ilev"), (dt_path, "lev"), (q_path, "lev"))
                NCDataset(path, "a") do ds
                    ds[axis].attrib["positive"] = opposite
                end
                @test_throws r"Conflicting GEOS-FP physics level orientations" pp._open_geosfp_physics_fallback(settings, date)
                NCDataset(path, "a") do ds
                    delete!(ds[axis].attrib, "positive")
                end
            end

            # Dimensionless edge direction must match the actual layer axis.
            NCDataset(cm_path, "a") do ds
                ds["ilev"][:] = collect(73:-1:1)
            end
            @test_throws r"CMFMC edges and DTRAIN layers" pp._open_geosfp_physics_fallback(settings, date)
            NCDataset(cm_path, "a") do ds
                ds["ilev"][:] = collect(1:73)
            end
            NCDataset(q_path, "a") do ds
                ds["lev"][:] = collect(72:-1:1)
            end
            @test_throws r"I3 and A3dyn level coordinates disagree" pp._open_geosfp_physics_fallback(settings, date)

            # Physical metadata on the actual A3 axes also works without I3;
            # unrelated axes still suggest the opposite orientation.
            rm(q_path)
            for (path, axis, unused) in ((cm_path, "ilev", "lev"), (dt_path, "lev", "ilev"))
                NCDataset(path, "a") do ds
                    ds[axis].attrib["positive"] = positive
                    ds[unused].attrib["positive"] = opposite
                end
            end
            phys = pp._open_geosfp_physics_fallback(settings, date)
            try
                @test phys.level_orientation === expected
            finally
                pp.close_geosfp_physics!(phys)
            end
        end
    end
end

@testset "GEOS-FP physics download defaults satisfy dimensionless reader" begin
    pp = AtmosTransport.Preprocessing
    dd = AtmosTransport.DataDownloads
    date = Date(2021, 12, 1)
    mktempdir() do tmp
        fixture_dir = joinpath(tmp, "fixtures")
        _write_fp_latlon_physics!(fixture_dir, date; dual_axes=true)
        source = dd.GEOSFPSource(Dict{String,Any}(), "geosfp_025")
        protocol = dd.HTTPProtocol("https://example.invalid")
        output = dd.OutputConfig(tmp, "geosfp", "C4", "daily", "raw", false, false, "")
        tasks = dd.build_tasks(source, protocol, [date], output, Any[])
        @test Set(basename(t.dest_path) for t in tasks) == Set(
            "GEOSFP.20211201.$c.025x03125.nc" for c in ("A1", "A3mstE", "A3dyn", "I3"))
        # Stage exactly what the default task list downloads, without network I/O.
        for task in tasks
            mkpath(dirname(task.dest_path))
            cp(joinpath(fixture_dir, basename(task.dest_path)), task.dest_path)
        end
        settings = GEOSFPSettings(; root_dir=tmp, Nc=4,
            include_surface=true, include_convection=true,
            physics_dir=dirname(first(tasks).dest_path), physics_layout=:latlon_025)
        phys = pp._open_geosfp_physics_fallback(settings, date)
        try
            @test phys.level_orientation === :bottom_up
            @test phys.a1 !== nothing
        finally
            pp.close_geosfp_physics!(phys)
        end
    end
end
