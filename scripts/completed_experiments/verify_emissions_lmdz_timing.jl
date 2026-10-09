# ===========================================================================
# verify_emissions_lmdz_timing.jl
#
# Focused follow-up to verify_emissions_vs_geoschem.jl: the co2_natural
# (lmdz_co2) GLOBAL TOTAL ratio looked wild (sign-flipping, |ratio|>>1)
# while the per-cell MEDIAN rel-diff was tiny (~7e-4). That is the classic
# signature of a near-cancelling NET flux (gross +/- biosphere) where the
# global total ~ 0 makes any ratio explode. This script disentangles:
#
#   (A) per-cell spatial fidelity at each time, using magnitude-robust
#       metrics (area-weighted RMS / RMS(GC), corr, slope) instead of a
#       global-total ratio;
#   (B) TIMING: OUR linearly-interpolated lmdz field evaluated at each GC
#       3-hourly stamp on Dec 1 vs GC's own value -> does our temporal
#       application track GC's per-timestep variation?
#   (C) sanity: at an EXACT lmdz slice time, OUR field should equal the raw
#       CAMS slice (x44/12) regridded -> near machine-zero residual confirms
#       no interpolation/units error.
#
# Read-only. No runs, no binaries.
# ===========================================================================

using AtmosTransport
using AtmosTransport.MetDrivers: TransportBinaryDriver, driver_grid
using AtmosTransport.Grids: panel_cell_center_lonlat
using AtmosTransport.Models.InitialConditionIO:
    build_surface_flux_source, _surface_flux_storage_scale
using AtmosTransport.Operators.SurfaceFlux:
    TimeVaryingSurfaceFluxSource, LinearInterpFlux
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

gc_file(dt) = joinpath(GC_DIR,
    "GEOSChem.CATRINE_inst." * Dates.format(dt, "yyyymmdd_HHMM") * "z.nc4")

function read_gc(path, var)
    NCDataset(path) do ds
        flux = replace(Array{Float64}(ds[var][:, :, :, 1]), NaN => 0.0)
        area = Array{Float64}(ds["Met_AREAM2"][:, :, :, 1])
        return flux .* area, area              # kg/s per cell, area
    end
end

stack(p) = cat(p...; dims = 3)

function main()
    driver = TransportBinaryDriver(BIN; FT = FT, arch = AtmosTransport.CPU())
    grid = driver_grid(driver)

    cfg = Dict("kind" => "lmdz_co2", "time_varying" => true)
    src = build_surface_flux_source(grid, :co2_natural, cfg, FT;
                                    reference_time = RUN_START)
    @assert src isa TimeVaryingSurfaceFluxSource
    scale = FT(_surface_flux_storage_scale(:co2_natural, cfg))
    times = src.times
    series = src.cell_mass_rate_series        # NTuple{6} Nc x Nc x ntime
    @info "lmdz source loaded" ntime=length(times) first_times=times[1:4] scheme=src.scheme

    # physical kg/s field at clock time t, sampled to MATCH the source's actual
    # temporal scheme so the diagnostic reflects what the model applies. lmdz_co2
    # now defaults to StepwiseFlux (piecewise-constant CAMS hold = GC parity);
    # fall back to a linear point-blend for LinearInterpFlux sources.
    function ours_at(t)
        n = length(times)
        if src.scheme isa StepwiseFlux
            k = t <= times[1] ? 1 : searchsortedlast(times, t)   # hold block containing t
            i0=i1=k; w0=1.0; w1=0.0
        elseif t <= times[1]; i0=i1=1; w0=1.0; w1=0.0
        elseif t >= times[end]; i0=i1=n; w0=1.0; w1=0.0
        else
            k = searchsortedlast(times, t)
            t0=times[k]; t1=times[k+1]; frac=(t-t0)/(t1-t0)
            i0=k; i1=k+1; w0=1-frac; w1=frac
        end
        out = ntuple(p -> (w0.*series[p][:,:,i0] .+ w1.*series[p][:,:,i1]) ./ scale, 6)
        return stack(out)
    end

    # robust per-cell metrics (area-weighted, magnitude-normalized)
    function metrics(ours, gc, area)
        d = ours .- gc
        wrms_abs = sqrt(sum((d.^2).*area)/sum(area))
        wrms_gc  = sqrt(sum((gc.^2).*area)/sum(area))
        nrmse = wrms_abs/wrms_gc
        ov=vec(ours); gv=vec(gc)
        c = cor(ov,gv)
        slope = sum(ov.*gv)/sum(gv.*gv)         # least-squares ours~slope*gc
        maxabs_gc = maximum(abs.(gc))
        maxabs_d  = maximum(abs.(d))
        return (nrmse=nrmse, corr=c, slope=slope, maxabs_d=maxabs_d, maxabs_gc=maxabs_gc,
                tot_ours=sum(ours), tot_gc=sum(gc))
    end

    println("\n=== (A)+(B) co2_natural lmdz: ALL 8 GC stamps on Dec 1 (timing + spatial) ===")
    @printf("%-17s | %-12s | %-12s | %-7s | %-6s | %-6s | %-10s\n",
            "GC time","tot_ours kg/s","tot_gc kg/s","nrmse","corr","slope","max|d| kg/s")
    dec1_stamps = [DateTime(2021,12,1,h,0,0) for h in 3:3:21]
    push!(dec1_stamps, DateTime(2021,12,2,0,0,0))   # the 0000z next-day file
    for dt in dec1_stamps
        isfile(gc_file(dt)) || continue
        t = Float64(Dates.value(dt - RUN_START))/1000.0
        ours = ours_at(t)
        gc, area = read_gc(gc_file(dt), "EmisCO2_Total")
        m = metrics(ours, gc, area)
        @printf("%-17s | %12.4e | %12.4e | %6.4f | %5.3f | %5.3f | %10.3e\n",
                Dates.format(dt,"mm-dd HH:MM"), m.tot_ours, m.tot_gc, m.nrmse, m.corr, m.slope, m.maxabs_d)
    end

    println("\n=== (C) EXACT-slice residual: t at an lmdz knot (12:00 = slice 5) ===")
    # 12:00 UTC Dec 1 is exactly lmdz time=12h (knot) -> our interp == that slice.
    dt = DateTime(2021,12,1,12,0,0)
    t = Float64(Dates.value(dt - RUN_START))/1000.0
    ours = ours_at(t)
    gc, area = read_gc(gc_file(dt), "EmisCO2_Total")
    m = metrics(ours, gc, area)
    @printf("knot t=12h: nrmse=%.4f corr=%.4f slope=%.4f tot_ours=%.4e tot_gc=%.4e\n",
            m.nrmse, m.corr, m.slope, m.tot_ours, m.tot_gc)

    # GC field flux-density range vs ours, to confirm units (kg CO2 vs kg C)
    println("\n=== field magnitude / units sanity at 03:00z ===")
    gc3, area3 = read_gc(gc_file(DateTime(2021,12,1,3,0,0)), "EmisCO2_Total")
    o3 = ours_at(Float64(Dates.value(DateTime(2021,12,1,3,0,0)-RUN_START))/1000.0)
    fd_gc = gc3 ./ area3; fd_o = o3 ./ area3
    @printf("GC flux-density kg/m2/s: min=%.3e max=%.3e ; OURS: min=%.3e max=%.3e\n",
            minimum(fd_gc), maximum(fd_gc), minimum(fd_o), maximum(fd_o))
    @printf("ratio of max|flux density| OURS/GC = %.4f  (1.0 => same kgCO2 basis; 0.27 or 3.67 => kgC mixup)\n",
            maximum(abs.(fd_o))/maximum(abs.(fd_gc)))
    @info "DONE (read-only)."
end

main()
