# ===========================================================================
# verify_emissions_vs_geoschem.jl
#
# GATING CHECK: confirm OUR applied surface emissions match GEOS-Chem's,
# per species and per timestep, on the C180 cube, before a 3-month campaign.
#
# For each of the 4 catrine species (co2_natural=lmdz_co2,
# co2_fossil=gridfed_fossil_co2, sf6=edgar_sf6, rn222=zhang_rn222):
#   1. Build OUR applied per-cell physical emission flux [kg species/s] using
#      the model's OWN loader + conservative LL->CS regridder + temporal
#      scheme (the EXACT machinery a run uses). The storage scale
#      (M_dryair / M_species) is divided back out to recover physical kg/s.
#   2. Read GC's Emis<species> [kg/m2/s] x Met_AREAM2 [m2] -> per-cell kg/s.
#   3. Compare: global total (kg/s) per sampled time; per-cell spatial
#      ratio / max-rel-diff / correlation; temporal variation.
#
# Geometry: our CS mesh (from the actual run binary) is index-aligned with
# GC's (nf, Ydim, Xdim) layout; the script verifies that alignment via
# panel_cell_center_lonlat vs GC `lons`/`lats` before trusting per-cell maps.
#
# Usage:
#   julia --project=. scripts/completed_experiments/verify_emissions_vs_geoschem.jl
#
# Read-only: no transport runs, no binaries written.
# ===========================================================================

using AtmosTransport
using AtmosTransport.MetDrivers: TransportBinaryDriver, driver_grid
using AtmosTransport.Grids: panel_cell_center_lonlat, cell_area
using AtmosTransport.Models.InitialConditionIO:
    build_surface_flux_source, _surface_flux_storage_scale,
    _load_file_surface_flux_field, _load_timevarying_surface_flux_field,
    _build_surface_flux_regridder, _apply_surface_flux_regridder
using AtmosTransport.Operators.SurfaceFlux:
    SurfaceFluxSource, TimeVaryingSurfaceFluxSource,
    StepwiseFlux, LinearInterpFlux, ConservativeMeanFlux
using NCDatasets
using Dates
using Printf
using Statistics

const FT = Float64
const GC_DIR  = expanduser("~/data/AtmosTransport/catrine-geoschem-runs")
const BIN     = expanduser("~/data/AtmosTransport/met/geosit/C180/" *
                           "transport_binary_dec2021_catrine_f32_fullL72/" *
                           "geos_transport_20211201_float32.bin")
const RUN_START = DateTime(2021, 12, 1, 0, 0, 0)

# species symbol -> (GC Emis var, surface_flux cfg) ; cfg mirrors the
# fullphys config exactly (defaults from _resolve_surface_flux_file).
const SPECIES = [
    (:co2_natural, "EmisCO2_Total",
        Dict("kind" => "lmdz_co2", "time_varying" => true)),  # default temporal=linear
    (:co2_fossil,  "Emis_FossilCO2_Total",
        Dict("kind" => "gridfed_fossil_co2", "time_index" => 12)),
    (:sf6,         "EmisSF6",
        Dict("kind" => "edgar_sf6", "scale" => 1.0116635)),  # match GC 10.10 kt/yr
    (:rn222,       "EmisRn_Soil",
        Dict("kind" => "zhang_rn222", "time_index" => 12)),
]

# ---------------------------------------------------------------------------
# GC file helpers
# ---------------------------------------------------------------------------

gc_file(dt::DateTime) = joinpath(GC_DIR,
    "GEOSChem.CATRINE_inst." * Dates.format(dt, "yyyymmdd_HHMM") * "z.nc4")

# Read GC Emis var at a file -> (per-cell flux kg/m2/s, per-cell kg/s, area)
# all as (Xdim, Ydim, nf) = (i, j, p) to match our mesh order. GC stores
# (time, nf, Ydim, Xdim); NCDatasets reads column-major reversed ->
# array indexed [Xdim, Ydim, nf, time]. We squeeze time.
function read_gc_emis(path::String, var::String)
    NCDataset(path) do ds
        raw  = Array{Float64}(ds[var][:, :, :, 1])        # (Xdim, Ydim, nf)
        area = Array{Float64}(ds["Met_AREAM2"][:, :, :, 1])
        lons = Array{Float64}(ds["lons"][:, :, :])         # (Xdim, Ydim, nf)
        lats = Array{Float64}(ds["lats"][:, :, :])
        flux = replace(raw, NaN => 0.0)
        mass = flux .* area                                # kg/s per cell
        return (flux = flux, mass = mass, area = area, lons = lons, lats = lats)
    end
end

# ---------------------------------------------------------------------------
# OUR side: per-cell physical kg/s as (Xdim, Ydim, nf), sampled at clock time.
# build_surface_flux_source stores storage units (x storage_scale); divide
# it out to recover physical kg species/s.
# ---------------------------------------------------------------------------

# Stack 6 panels (each Nc x Nc) into (Nc, Nc, 6).
stack_panels(panels) = cat(panels...; dims = 3)

function our_static_field(grid, name::Symbol, cfg)
    src = build_surface_flux_source(grid, name, cfg, FT)
    scale = FT(_surface_flux_storage_scale(name, cfg))
    panels = src.cell_mass_rate                       # NTuple{6} Nc x Nc, storage units
    phys = ntuple(p -> panels[p] ./ scale, 6)         # physical kg/s
    return stack_panels(phys)
end

# Time-varying (lmdz): sample at clock time t (sec since RUN_START) with the
# source's temporal scheme, exactly as the runtime kernel does for a single
# instantaneous step (dt->0 so Linear/Conservative reduce to point eval at t;
# Stepwise to the containing block). Returns physical kg/s (Nc,Nc,6).
function our_timevarying_field(grid, name::Symbol, cfg, t_sec::Float64)
    src = build_surface_flux_source(grid, name, cfg, FT)
    @assert src isa TimeVaryingSurfaceFluxSource
    scale = FT(_surface_flux_storage_scale(name, cfg))
    times = src.times
    scheme = src.scheme
    # two-slice blend at time t (dt=0 -> point evaluation, matches a 3-hourly
    # GC instantaneous snapshot at exactly t).
    i0, i1, w0, w1 = _flux_bracket(scheme, times, t_sec)
    series = src.cell_mass_rate_series                # NTuple{6} Nc x Nc x ntime
    out = ntuple(p -> (w0 .* series[p][:, :, i0] .+ w1 .* series[p][:, :, i1]) ./ scale, 6)
    return stack_panels(out), (i0, i1, w0, w1)
end

# Replicate _flux_temporal_weights at dt=0 for the bracket.
function _flux_bracket(::LinearInterpFlux, times, t)
    n = length(times)
    n == 1 && return (1, 1, 1.0, 0.0)
    t <= times[1]   && return (1, 1, 1.0, 0.0)
    t >= times[end] && return (n, n, 1.0, 0.0)
    k = searchsortedlast(times, t)
    t0 = Float64(times[k]); t1 = Float64(times[k+1]); span = t1 - t0
    frac = span > 0 ? (Float64(t) - t0) / span : 0.0
    return (k, k+1, 1.0 - frac, frac)
end
function _flux_bracket(::StepwiseFlux, times, t)
    t <= times[1] && return (1, 1, 1.0, 0.0)
    k = searchsortedlast(times, t)
    return (k, k, 1.0, 0.0)
end
_flux_bracket(::ConservativeMeanFlux, times, t) = _flux_bracket(LinearInterpFlux(), times, t)

# ---------------------------------------------------------------------------
# comparison metrics
# ---------------------------------------------------------------------------

function compare_fields(ours::Array{Float64,3}, gc::Array{Float64,3})
    # global totals
    tot_ours = sum(ours)
    tot_gc   = sum(gc)
    ratio    = tot_ours / tot_gc
    # per-cell: relative diff where GC nonzero & magnitude significant
    absgc = abs.(gc)
    thr = 1e-6 * maximum(absgc)              # ignore noise floor
    mask = absgc .> thr
    nmask = count(mask)
    reldiff = abs.(ours .- gc) ./ max.(absgc, thr)
    maxrel = nmask > 0 ? maximum(reldiff[mask]) : NaN
    medrel = nmask > 0 ? median(reldiff[mask]) : NaN
    # correlation over significant cells
    ov = ours[mask]; gv = gc[mask]
    corr = (length(ov) > 2 && std(ov) > 0 && std(gv) > 0) ? cor(ov, gv) : NaN
    return (tot_ours = tot_ours, tot_gc = tot_gc, ratio = ratio,
            maxrel = maxrel, medrel = medrel, corr = corr,
            nmask = nmask, ncell = length(gc))
end

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

function main()
    isdir(GC_DIR) || error("GC dir not found: $GC_DIR")
    isfile(BIN)   || error("transport binary not found: $BIN")

    @info "Opening transport binary to get the run mesh..." BIN
    # CS transport binaries are read by TransportBinaryDriver; it builds
    # the exact run mesh from the binary's own cs-definition. We only need the
    # mesh geometry, so skip the replay-consistency gate.
    driver = TransportBinaryDriver(BIN; FT = FT, arch = AtmosTransport.CPU())
    grid = driver_grid(driver)
    mesh = grid.horizontal
    Nc = mesh.Nc
    @info "mesh" Nc

    # --- geometry alignment check vs GC ---
    gc_geom = read_gc_emis(gc_file(DateTime(2021,12,1,3,0,0)), "EmisCO2_Total")
    our_lons = Array{Float64}(undef, Nc, Nc, 6)
    our_lats = Array{Float64}(undef, Nc, Nc, 6)
    for p in 1:6
        lonp, latp = panel_cell_center_lonlat(mesh, p)
        our_lons[:, :, p] = lonp
        our_lats[:, :, p] = latp
    end
    dlon = our_lons .- gc_geom.lons
    dlon = mod.(dlon .+ 180, 360) .- 180          # wrap
    dlat = our_lats .- gc_geom.lats
    @info @sprintf("GEOMETRY index-alignment vs GC: max|dlon|=%.4f deg  max|dlat|=%.4f deg",
                   maximum(abs.(dlon)), maximum(abs.(dlat)))
    aligned = maximum(abs.(dlon)) < 0.5 && maximum(abs.(dlat)) < 0.5
    @info "  index-aligned (per-cell maps comparable directly)?" aligned

    # --- sampled GC times in Dec 2021 (incl a month boundary) ---
    sample_times = [
        DateTime(2021,12,1,3,0,0),
        DateTime(2021,12,1,12,0,0),
        DateTime(2021,12,2,0,0,0),
        DateTime(2021,12,15,12,0,0),
        DateTime(2021,12,31,21,0,0),
    ]
    # keep only those whose GC file exists
    sample_times = filter(dt -> isfile(gc_file(dt)), sample_times)
    @info "Sampling GC times" sample_times

    println("\n", repeat("=", 110))
    for (name, gcvar, cfg) in SPECIES
        println("\n#### SPECIES: $name   (our kind=$(cfg["kind"]))   GC var=$gcvar")
        is_tv = get(cfg, "time_varying", false)

        # build OUR source once (also report molar-mass / storage scale)
        scale = _surface_flux_storage_scale(name, cfg)
        @info @sprintf("  storage_scale(M_dryair/M_%s) = %.6f", String(name), scale)

        @printf("  %-19s | %-13s | %-13s | %-8s | %-9s | %-9s | %-7s | %-6s\n",
                "time(UTC)", "OURS tot kg/s", "GC tot kg/s", "ratio", "maxrel", "medrel", "corr", "nsig")
        results = []
        for dt in sample_times
            t_sec = Float64(Dates.value(dt - RUN_START)) / 1000.0
            ours = if is_tv
                f, _ = our_timevarying_field(grid, name, cfg, t_sec)
                f
            else
                our_static_field(grid, name, cfg)
            end
            gc = read_gc_emis(gc_file(dt), gcvar)
            m = compare_fields(ours, gc.mass)
            push!(results, (dt, m))
            @printf("  %-19s | %12.5e | %12.5e | %7.4f | %8.2e | %8.2e | %6.3f | %d\n",
                    Dates.format(dt, "yyyy-mm-dd HH:MM"),
                    m.tot_ours, m.tot_gc, m.ratio, m.maxrel, m.medrel, m.corr, m.nmask)
        end

        # timing summary: does OUR global total vary in time like GC's?
        ours_tots = [r[2].tot_ours for r in results]
        gc_tots   = [r[2].tot_gc   for r in results]
        ours_var = maximum(ours_tots) - minimum(ours_tots)
        gc_var   = maximum(gc_tots)   - minimum(gc_tots)
        @printf("  TIMING: OUR total varies by %.3e kg/s across samples (%.2f%% of mean); GC by %.3e (%.2f%%)\n",
                ours_var, 100*ours_var/mean(abs.(ours_tots) .+ eps()),
                gc_var,   100*gc_var/mean(abs.(gc_tots) .+ eps()))
    end
    println("\n", repeat("=", 110))
    @info "DONE (read-only; no runs, no binaries written)."
end

main()
