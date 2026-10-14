#!/usr/bin/env julia
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
    PoleRowsPre._write_spectral_day_cache(
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
            path, _ = process_day(date, grid,
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

@testset "compute_mass_fluxes! keeps the pole-row zonal flux (regression, issue #58)" begin
    # A uniform eastward wind: every row, the pole rows included, carries the
    # same zonal flux per unit Δp. Pole zeroing must not come back silently
    # (bug archive SESSION1_GRID_CM_FIX_2026-04-03.md, section 3).
    grid = build_target_geometry(Val(:latlon), Dict{String, Any}("nlon" => 8, "nlat" => 5), Float64)
    Nx, Ny, Nz = 8, 5, 1
    u_stag = [10.0 * cosd(grid.lats[j]) for i in 1:Nx + 1, j in 1:Ny, k in 1:Nz]   # U cos φ
    v_stag = zeros(Nx, Ny + 1, Nz)
    ps = fill(POLE_PS, Nx, Ny)
    dp = fill(POLE_PS, Nx, Ny, Nz)
    am = zeros(Nx + 1, Ny, Nz); bm = zeros(Nx, Ny + 1, Nz); cm = zeros(Nx, Ny, Nz + 1)
    PoleRowsPre.compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, dp, ps, [0.0], [1.0],
                                     grid, 450.0, Nz)
    @test all(>(0), am)
    @test am[:, 1, 1] ≈ am[:, 3, 1] rtol = 1e-12
    @test am[:, Ny, 1] ≈ am[:, 3, 1] rtol = 1e-12
end
