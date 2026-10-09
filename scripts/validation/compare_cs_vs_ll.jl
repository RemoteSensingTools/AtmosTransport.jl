#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# compare_cs_vs_ll.jl — plan 40 Commit 7
#
# Post-run cross-grid validation. Loads a CS snapshot NetCDF and a matching
# LL snapshot NetCDF (same physics, same dates, same tracer IC), conservatively
# regrids the LL column-mean onto the CS grid at each snapshot, and emits a
# CSV with per-(snapshot × tracer) diagnostics:
#
#   - cs_mass_drift_pct : (Σ air_mass(t) - Σ air_mass(0)) / Σ air_mass(0) × 100.
#     Computed from the CS snapshot's `air_mass` variable (LL snapshots do
#     not carry air_mass, so the reference column reports NA). Soft acceptance:
#     |drift| < 1e-5 per day F32.
#   - cs_col_mean, ll_col_mean : area-weighted global means of the CS and
#     regridded-LL column-mean VMR fields (mol mol⁻¹). Sanity check that
#     the two reference fields agree on the integrated quantity.
#   - col_diff_rmse, col_diff_max_abs : per-cell RMSE and max-abs of
#     (regridded_LL - CS) column-mean VMR. Plan acceptance: column CO₂ mean
#     within 0.2 ppm = 2e-7 at day 3 for C180.
#   - cs_gpu_verified, ll_gpu_verified : did each run's stdout log contain
#     the `[gpu verified]` marker printed by `_assert_gpu_residency!` ?
#     Reads optional `--cs-log` / `--ll-log` flags; `false` if unset or
#     missing from the file.
#
# CS snapshots may be either `panel_convention=gnomonic` or
# `panel_convention=geos_native`. The validation path uses the same
# convention-aware `CubedSphereMesh` and `Trees.treeify` implementation as
# runtime output and visualization, so GEOS-native panels are not remapped or
# special-cased here.
#
# Usage:
#   julia --project=. scripts/validation/compare_cs_vs_ll.jl \
#       --cs  ~/data/AtmosTransport/output/catrine_c48_3d/advonly.nc  \
#       --ll  ~/data/AtmosTransport/output/catrine_ll720_3d/advonly.nc \
#       --out artifacts/validation/catrine_c48_vs_ll720_3d_advonly.csv
#       [--cs-log /tmp/run_cs48_advonly.log]
#       [--ll-log /tmp/run_ll720_advonly.log]
# ---------------------------------------------------------------------------

using Printf
using Statistics: mean
using NCDatasets

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Grids: LatLonMesh, CubedSphereMesh,
                             GnomonicPanelConvention, GEOSNativePanelConvention
using .AtmosTransport.Regridding: build_regridder, apply_regridder!

# Earth radius [m] (ECMWF/TM5 convention; matches preprocessing constants).
const R_EARTH_M = 6.371229e6

const USAGE = """
Usage: julia --project=. scripts/validation/compare_cs_vs_ll.jl \\
           --cs <cs.nc> --ll <ll.nc> --out <diff.csv>
           [--cs-log <path>] [--ll-log <path>]
"""

function _parse_args(argv)
    cs = nothing; ll = nothing; out = nothing
    cs_log = ""; ll_log = ""
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--cs" && i + 1 <= length(argv)
            cs = expanduser(argv[i + 1]); i += 2
        elseif arg == "--ll" && i + 1 <= length(argv)
            ll = expanduser(argv[i + 1]); i += 2
        elseif arg == "--out" && i + 1 <= length(argv)
            out = expanduser(argv[i + 1]); i += 2
        elseif arg == "--cs-log" && i + 1 <= length(argv)
            cs_log = expanduser(argv[i + 1]); i += 2
        elseif arg == "--ll-log" && i + 1 <= length(argv)
            ll_log = expanduser(argv[i + 1]); i += 2
        elseif arg in ("-h", "--help")
            println(USAGE); exit(0)
        else
            error("Unknown argument `$(arg)`.\n$(USAGE)")
        end
    end
    cs  === nothing && error("--cs required.\n$(USAGE)")
    ll  === nothing && error("--ll required.\n$(USAGE)")
    out === nothing && error("--out required.\n$(USAGE)")
    isfile(cs) || error("CS snapshot not found: $(cs)")
    isfile(ll) || error("LL snapshot not found: $(ll)")
    return (; cs, ll, out, cs_log, ll_log)
end

# Return `true` iff `path` is a non-empty file containing the literal
# `[gpu verified]` tag emitted by `_assert_gpu_residency!`. Empty
# path or missing file → `false`.
function _gpu_verified(path::AbstractString)
    isempty(path) && return false
    isfile(path)  || return false
    return occursin("[gpu verified]", read(path, String))
end

# Refuse to proceed unless the snapshot was written on dry basis. Air-mass-
# weighted column-mean VMRs are only physically meaningful when the weights
# are dry air mass — otherwise water vapour gets double-counted through
# the tracer's own (1 − qv) scaling. The writers in `DrivenRunner.jl`
# emit `ds.attrib["mass_basis"]` as of plan 40 Commit 7; older snapshots
# get a loud warning rather than a silent pass.
# Reconstruct the (west_edge, east_edge) × (south_edge, north_edge) mesh
# intervals from cell-center coordinate arrays.
#
# ASSUMPTION: `lons`/`lats` are **cell-centered** with uniform spacing,
# which is what `_write_snapshot_ll!` emits (`v_lat[:] = mesh.φᶜ`,
# `v_lon[:] = mesh.λᶜ` — both cell centers by LatLonMesh construction).
# Both ERA5 and GEOS LL products written by the runner satisfy this.
#
# A raw endpoint-inclusive lat layout (`[-90, -89.5, …, 90]`) would
# infer `south_edge = -90 - Δφ/2 = -90.25°`, outside the sphere, and
# downstream `LatLonMesh` rejects it with an `ArgumentError`. That
# failure is loud and immediate — not a silent mis-registration — so
# the script refuses to run against raw-GRIB-style snapshots rather
# than silently comparing on a wrong grid. If that use case ever
# matters, add an explicit branch.
#
# The tiny `abs(... - ±90) < 1e-6` clamp handles floating-point drift
# in cell-centered inputs (not endpoint-inclusive ones).
function _infer_ll_interval(lons::AbstractVector{<:Real}, lats::AbstractVector{<:Real})
    length(lons) >= 2 || error("LL snapshot has <2 lon points; cannot infer interval")
    length(lats) >= 2 || error("LL snapshot has <2 lat points; cannot infer interval")
    dlon = Float64(lons[2] - lons[1])
    dlat = Float64(lats[2] - lats[1])
    lon_west = Float64(lons[1])   - dlon / 2
    lon_east = Float64(lons[end]) + dlon / 2
    lat_south = Float64(lats[1])   - dlat / 2
    lat_north = Float64(lats[end]) + dlat / 2
    # Pole drift fix: if the array actually spans pole-to-pole the edges
    # should be exact ±90°. Tolerance 1e-4 covers Float32 roundoff in
    # the NetCDF (the writer casts to the run's `float_type`, which is
    # often Float32; e.g. LL720×361 snapshot has lats[1]≈-89.7506943
    # giving a drift of ~4e-6° after the edge shift). Far below the
    # grid's own Δφ≈0.5°, so no risk of misclassifying an endpoint-
    # inclusive raw GRIB as cell-centered.
    abs(lat_south + 90.0) < 1e-4 && (lat_south = -90.0)
    abs(lat_north - 90.0) < 1e-4 && (lat_north =  90.0)
    return (lon_west, lon_east), (lat_south, lat_north)
end

function _require_dry_mass_basis(ds, path::AbstractString)
    if haskey(ds.attrib, "mass_basis")
        basis = String(ds.attrib["mass_basis"])
        basis == "dry" || error(
            "$(path): mass_basis=$(basis) — compare_cs_vs_ll.jl requires " *
            "dry-basis snapshots (dry VMR is the transport contract). Regenerate the run with " *
            "a dry binary, or extend this script to convert moist→dry " *
            "via qv before computing column means.")
    else
        @warn "$(path) has no `mass_basis` attribute — assuming dry per " *
              "the dry-basis contract. Regenerate with a current runner to get the " *
              "attribute set. If this is a moist-basis run, the column " *
              "means below are physically wrong."
    end
end

function _cs_panel_convention(ds, path::AbstractString)
    if haskey(ds.attrib, "panel_convention")
        conv = lowercase(String(ds.attrib["panel_convention"]))
        conv in ("gnomonic", "gnomic") && return GnomonicPanelConvention()
        conv in ("geos_native", "geosnative", "geos-native") && return GEOSNativePanelConvention()
        error("$(path): unsupported panel_convention=$(conv); expected gnomonic or geos_native")
    else
        @warn "$(path) has no `panel_convention` attribute — assuming " *
              "gnomonic (the preprocessor default). Regenerate with a " *
              "current runner to pin this."
        return GnomonicPanelConvention()
    end
end

const CS_NON_TRACER_VARS = Set([
    "time", "nf", "air_mass", "air_mass_per_area", "column_air_mass_per_area",
    "Xdim", "Ydim", "Xcorner", "Ycorner", "lev",
    "lons", "lats", "corner_lons", "corner_lats", "cell_area",
    "cubed_sphere",
])

function _cs_tracer_names(ds)
    names = Symbol[]
    for n in keys(ds)
        s = String(n)
        s in CS_NON_TRACER_VARS && continue
        endswith(s, "_column_mean") && continue
        endswith(s, "_column_mass_per_area") && continue
        ndims(ds[s]) == 5 && push!(names, Symbol(s))
    end
    return sort!(unique(names))
end

# CS column-mean **dry** VMR per cell, air-mass-weighted:
#   col_vmr[i,j,p] = Σ_k (vmr_dry[k] × m_dry[k]) / Σ_k m_dry[k]
#
# The CS NetCDF `{tracer}` variable already stores per-level VMR
# (`_write_snapshot_cs!` at DrivenRunner.jl writes `rm_p ./ m_p`), and
# `air_mass` stores the carrier-mass array the runtime held. Both are
# **dry** on dry-basis runs, the default (the writer records this as
# `ds.attrib["mass_basis"] = "dry"`; the caller of this function is
# responsible for verifying that attribute first).
#
# Do NOT divide by Σm without first weighting by m — `Σ vmr_k / Σ m_k`
# has units of 1/kg and is physically meaningless. This bug was caught
# in code review 2026-04-24 before the first CS validation run.
#
# Air-mass-weighted VMRs should always be **dry**-air-mass-weighted:
# the air column's dry-mass distribution is what the trace gas
# actually rides on. A moist-basis weight would double-count water
# vapour through the tracer's own (1 − qv) scaling.
#
# Input: vmr and m with shape (Nc, Nc, nf, Nz, ntime). Must be dry.
# Output: (Nc, Nc, nf, ntime).
function _cs_column_mean_dry_vmr(vmr::Array{T, 5}, m::Array{T, 5}) where T
    Nc1, Nc2, nf, Nz, ntime = size(vmr)
    out = zeros(Float64, Nc1, Nc2, nf, ntime)
    @inbounds for t in 1:ntime, p in 1:nf, j in 1:Nc2, i in 1:Nc1
        num = 0.0; den = 0.0
        for k in 1:Nz
            vmr_k = Float64(vmr[i, j, p, k, t])
            m_k   = Float64(m[i,   j, p, k, t])
            num += vmr_k * m_k
            den += m_k
        end
        out[i, j, p, t] = den > 0 ? num / den : 0.0
    end
    return out
end

# Global area-weighted mean of a CS panel field (Nc, Nc, nf). Supported panel
# conventions share the same per-panel `(Nc, Nc)` cell-area matrix, so we pass
# that and reuse it across the 6 panels.
function _cs_area_mean(field::AbstractArray{<:Real, 3}, areas_2d::AbstractMatrix)
    num = 0.0; den = 0.0
    Nc1, Nc2, nf = size(field)
    @inbounds for p in 1:nf, j in 1:Nc2, i in 1:Nc1
        a = Float64(areas_2d[i, j]); v = Float64(field[i, j, p])
        num += v * a; den += a
    end
    return den > 0 ? num / den : 0.0
end

function main()
    opts = _parse_args(ARGS)

    @info "Loading CS snapshot: $(opts.cs)"
    ds_cs = NCDataset(opts.cs, "r")
    _require_dry_mass_basis(ds_cs, opts.cs)
    cs_convention = _cs_panel_convention(ds_cs, opts.cs)
    Nc    = Int(ds_cs.attrib["Nc"])
    time  = Float64.(collect(ds_cs["time"][:]))
    air   = Array{Float64}(ds_cs["air_mass"][:, :, :, :, :])   # (Nc,Nc,nf,Nz,ntime)
    tracer_names = _cs_tracer_names(ds_cs)
    cs_vmr_full = Dict{Symbol, Array{Float64, 5}}()
    for name in tracer_names
        cs_vmr_full[name] = Array{Float64}(ds_cs[String(name)][:, :, :, :, :])
    end
    close(ds_cs)

    @info "Loading LL snapshot: $(opts.ll)"
    ds_ll   = NCDataset(opts.ll, "r")
    _require_dry_mass_basis(ds_ll, opts.ll)
    ll_lons = Float64.(collect(ds_ll["lon"][:]))
    ll_lats = Float64.(collect(ds_ll["lat"][:]))
    ll_time = Float64.(collect(ds_ll["time"][:]))
    # All `<name>_column_mean` variables in the LL snapshot.
    ll_tracer_names = Set{Symbol}()
    for n in keys(ds_ll)
        s = String(n)
        endswith(s, "_column_mean") || continue
        push!(ll_tracer_names, Symbol(s[1:end - length("_column_mean")]))
    end

    ll_col = Dict{Symbol, Array{Float64, 3}}()
    for name in tracer_names
        name in ll_tracer_names || continue
        ll_col[name] = Array{Float64}(ds_ll["$(String(name))_column_mean"][:, :, :])
    end
    close(ds_ll)

    # Fail loudly on tracer-set mismatch in either direction. Silently
    # dropping a CS-only tracer (finding round-5) would hide half the
    # validation; leaving an LL-only tracer uncompared (finding round-6)
    # would hide a dropped CS tracer. The two snapshots must describe
    # the same tracer set or the CSV is misleading.
    missing_in_ll = [n for n in tracer_names     if !(n in ll_tracer_names)]
    extra_in_ll   = [n for n in ll_tracer_names if !(n in Set(tracer_names))]
    if !isempty(missing_in_ll) || !isempty(extra_in_ll)
        error("tracer-set mismatch between CS and LL snapshots:\n" *
              "  missing in LL: $(missing_in_ll)\n" *
              "  extra in LL:   $(extra_in_ll)\n" *
              "Re-run both with the same tracer set.")
    end

    # Sanity checks on matching snapshots.
    length(time) == length(ll_time) ||
        error("snapshot counts differ: CS=$(length(time)) LL=$(length(ll_time))")
    all(abs.(time .- ll_time) .< 0.5) ||
        error("CS/LL snapshot hours differ beyond 0.5h: CS=$(time) LL=$(ll_time)")

    # Build meshes + LL→CS conservative regridder.
    @info "Building LL→CS regridder"
    Nx_ll = length(ll_lons); Ny_ll = length(ll_lats)
    # Derive the mesh interval from the NetCDF's cell-center arrays rather
    # than hard-coding (-180, 180) × (-90, 90). The LL writer stores
    # λᶜ / φᶜ, so edges are λᶜ[1] ± Δλ/2. This handles both (-180, 180)
    # and (0, 360) longitude conventions plus any future non-global or
    # sub-polar lat interval without silently mis-registering the regrid.
    lon_interval, lat_interval = _infer_ll_interval(ll_lons, ll_lats)
    ll_mesh = LatLonMesh(; FT = Float64, Nx = Nx_ll, Ny = Ny_ll,
                          longitude = lon_interval, latitude = lat_interval,
                          radius = R_EARTH_M)
    cs_mesh = CubedSphereMesh(; FT = Float64, Nc = Nc,
                               radius = R_EARTH_M,
                               convention = cs_convention)
    regridder = build_regridder(ll_mesh, cs_mesh; normalize = false)
    n_src = length(regridder.src_areas); n_dst = length(regridder.dst_areas)
    @info @sprintf("Regridder: %d → %d", n_src, n_dst)

    # CS cell-area matrix (gnomonic symmetry: same `(Nc, Nc)` on all panels).
    cs_areas_2d = Array{Float64}(cs_mesh.cell_areas)

    cs_gpu_ok = _gpu_verified(opts.cs_log)
    ll_gpu_ok = _gpu_verified(opts.ll_log)

    # Precompute CS column-mean VMR per tracer once across all snapshots —
    # reading per-snapshot inside the loop would be O(T²) in the 5-D array.
    cs_col_dry_by_name = Dict(name => _cs_column_mean_dry_vmr(cs_vmr_full[name], air)
                         for name in tracer_names)

    # Baseline air-mass (snapshot 1) for drift.
    m0 = sum(Float64, view(air, :, :, :, :, 1))
    mkpath(dirname(opts.out))

    open(opts.out, "w") do io
        println(io, "time_hours,tracer,cs_mass_drift_pct,cs_col_mean_dry_vmr,ll_col_mean_dry_vmr,col_diff_rmse,col_diff_max_abs,cs_gpu_verified,ll_gpu_verified")

        regrid_buf   = zeros(Float64, n_dst)
        ll_on_cs_pan = Array{Float64, 3}(undef, Nc, Nc, 6)

        for (ti, hour) in enumerate(time)
            mt = sum(Float64, view(air, :, :, :, :, ti))
            drift_pct = 100.0 * (mt - m0) / m0

            for name in tracer_names
                # Every tracer is guaranteed present in `ll_col` by the
                # up-front tracer-set assertion above.
                cs_cm = @view cs_col_dry_by_name[name][:, :, :, ti]

                # Regrid LL[:,:,ti] onto CS panels. The CR.jl regridder
                # works on flat vectors; apply_regridder! writes Nc*Nc*6
                # panel cells contiguously, panel-major.
                ll_flat = vec(view(ll_col[name], :, :, ti))
                fill!(regrid_buf, 0.0)
                apply_regridder!(regrid_buf, regridder, ll_flat)
                copyto!(ll_on_cs_pan, regrid_buf)

                diff = ll_on_cs_pan .- cs_cm
                rmse = sqrt(mean(abs2, diff))
                max_abs = maximum(abs, diff)
                cs_mean = _cs_area_mean(cs_cm, cs_areas_2d)
                ll_mean = _cs_area_mean(ll_on_cs_pan, cs_areas_2d)

                @printf(io, "%.2f,%s,%.6e,%.6e,%.6e,%.6e,%.6e,%s,%s\n",
                        hour, String(name), drift_pct, cs_mean, ll_mean,
                        rmse, max_abs, cs_gpu_ok, ll_gpu_ok)
            end
        end
    end

    @info "Wrote $(opts.out)"
    return opts.out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
