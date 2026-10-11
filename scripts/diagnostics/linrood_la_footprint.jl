#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 25 demo — 6-hour accumulated LinRood adjoint footprint for LA.
#
# Builds a C48 cubed-sphere synthetic-meteorology problem and runs
# `cs_surface_emission_footprint(scheme=LinRoodPPMScheme())` backward
# from a column-mean observation at LA (panel 4, gnomonic).
# Accumulates the per-substep surface footprints over the final 6 hours
# (48 substeps × dt = 450 s) and prints summary statistics + saves the
# accumulated footprint to NetCDF for plotting.
#
# Why C48 + synthetic meteo: production C180 GEOS-IT binaries are Hp=1
# (LinRood requires Hp=3), and a 2-day C180 LinRood tape exceeds ~100 GB
# of memory. C48 with synthetic-but-realistic winds gives a tractable
# demo of the shipped capability while real-meteo C24/C48 binaries
# come online from the active downloads.
# ---------------------------------------------------------------------------

using Printf
using Random

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

const FT = Float64
const Nc = 24                 # cubed-sphere panel resolution
const Hp = 3                  # LinRood PPM stencil halo
const Nz = 6                  # vertical levels (toy)
const dt_substep = 450.0      # seconds — model time step of this synthetic problem (not the GEOS mass-flux dt)
const six_hours_substeps = 48 # 48 × 450 s = 6.0 h

# Locate LA: panel 4 in GnomonicPanelConvention is the Americas-
# equatorial panel; LA is at ~34°N, -118°W (i.e. lon=242°E). Build
# the panel cell-center lat/lon grid and find the closest cell.
function _find_la_cell(mesh::AT.CubedSphereMesh)
    la_lat = 34.0
    la_lon = -118.0
    best_panel, best_i, best_j = 0, 0, 0
    best_dist2 = Inf
    for p in 1:6
        lons, lats = AT.Grids.panel_cell_center_lonlat(mesh, p)
        for j in 1:mesh.Nc, i in 1:mesh.Nc
            dlon = lons[i, j] - la_lon
            dlon = mod(dlon + 180, 360) - 180  # wrap to [-180, 180]
            dlat = lats[i, j] - la_lat
            d2 = dlon^2 + dlat^2
            if d2 < best_dist2
                best_dist2 = d2
                best_panel = p
                best_i = i
                best_j = j
            end
        end
    end
    return (panel=best_panel, i=best_i, j=best_j, dist_deg=sqrt(best_dist2))
end

# Build a synthetic-but-realistic cubed-sphere meteorology with:
#   * non-trivial horizontal velocities (am, bm) varying per substep,
#     scaled so peak CFL ~ 0.5
#   * weak vertical mass flux (cm) representing residual convection
#   * uniform air mass m and zero tracer rm to start
function _build_problem(; nsteps::Int)
    mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
    N = Nc + 2Hp
    rng = MersenneTwister(2026_05_11)

    panels_m = ntuple(_ -> begin
        m = Array{FT, 3}(undef, N, N, Nz)
        @inbounds for k in 1:Nz, j in 1:N, i in 1:N
            m[i, j, k] = FT(1.0) * FT(2.0)^(Nz - k + 1)   # ~exponential with height
        end
        m
    end, Val(6))

    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), Val(6))
    Adv.fill_panel_halos!(panels_m, mesh; dir=0)
    Adv.fill_panel_halos!(panels_rm, mesh; dir=0)

    # Time-varying mass fluxes. Scale to keep CFL < 0.5.
    cfl_scale = FT(0.1)
    panels_am_steps = Vector{NTuple{6, Array{FT, 3}}}(undef, nsteps)
    panels_bm_steps = Vector{NTuple{6, Array{FT, 3}}}(undef, nsteps)
    panels_cm_steps = Vector{NTuple{6, Array{FT, 3}}}(undef, nsteps)
    for step in 1:nsteps
        t = FT(step)
        panels_am_steps[step] = ntuple(6) do p
            am = zeros(FT, N + 1, N, Nz)
            @inbounds for k in 1:Nz, j in (Hp + 1):(Hp + Nc),
                          i in (Hp + 1):(Hp + Nc + 1)
                am[i, j, k] = cfl_scale * panels_m[p][i, j, k] *
                              sin(FT(0.03) * t + FT(0.07) * i + FT(0.05) * j +
                                  FT(0.11) * k + FT(0.17) * p)
            end
            am
        end
        panels_bm_steps[step] = ntuple(6) do p
            bm = zeros(FT, N, N + 1, Nz)
            @inbounds for k in 1:Nz, j in (Hp + 1):(Hp + Nc + 1),
                          i in (Hp + 1):(Hp + Nc)
                bm[i, j, k] = cfl_scale * panels_m[p][i, j, k] *
                              cos(FT(0.04) * t + FT(0.06) * i + FT(0.08) * j +
                                  FT(0.13) * k + FT(0.19) * p)
            end
            bm
        end
        # Small vertical mass flux (residual convection ~ 1% of horizontal).
        panels_cm_steps[step] = ntuple(6) do p
            cm = zeros(FT, N, N, Nz + 1)
            @inbounds for k in 2:Nz, j in (Hp + 1):(Hp + Nc),
                          i in (Hp + 1):(Hp + Nc)
                cm[i, j, k] = FT(0.005) * panels_m[p][i, j, k] *
                              sin(FT(0.5) * t + FT(0.3) * i + FT(0.2) * j +
                                  FT(0.7) * k + FT(0.4) * p)
            end
            cm
        end
    end

    return (mesh=mesh, panels_m=panels_m, panels_rm=panels_rm,
            panels_am=panels_am_steps, panels_bm=panels_bm_steps,
            panels_cm=panels_cm_steps)
end

function main()
    println("=" ^ 72)
    println("Plan 25 demo — 6-hour LinRood adjoint footprint at Los Angeles")
    println("=" ^ 72)

    nsteps = six_hours_substeps
    @printf("Resolution:        C%d, Hp=%d, Nz=%d\n", Nc, Hp, Nz)
    @printf("Time window:       %d substeps × %.0f s = %.2f h (backward)\n",
            nsteps, dt_substep, nsteps * dt_substep / 3600)
    @printf("Scheme:            LinRoodPPMScheme(ORD=5)\n")

    print("\nBuilding problem... ")
    problem = _build_problem(nsteps=nsteps)
    println("done.")

    print("Locating LA on the cubed sphere... ")
    la = _find_la_cell(problem.mesh)
    @printf("\n  panel=%d  (i,j)=(%d, %d)\n  nearest cell ≈ %.2f° from (34°N, -118°W)\n",
            la.panel, la.i, la.j, la.dist_deg)

    # Set up a CSColumnMeanObjective at the LA cell (mass-weighted
    # vertical mean of the final tracer mixing ratio).
    obj = AT.CSColumnMeanObjective(la.panel, la.i, la.j)

    print("\nRunning forward + adjoint pass (LinRoodPPMScheme)...\n")
    t0 = time()
    result = AT.cs_surface_emission_footprint(
        problem.panels_rm, problem.panels_m,
        problem.panels_am, problem.panels_bm, problem.panels_cm,
        problem.mesh, obj;
        scheme = Adv.LinRoodPPMScheme(),
        dt = FT(dt_substep),
    )
    elapsed = time() - t0
    @printf("Reverse pass completed in %.1f s\n", elapsed)

    # `result.footprints[t]` is an NTuple{6, Matrix{FT}} of dJ/dE for
    # surface emission rates applied at the MIDPOINT of model substep
    # t. With nsteps=48 we already cover the full 6-hour window.
    # Accumulate over all substeps.
    nsubsteps = length(result.footprints)
    accumulated = ntuple(6) do p
        acc = zero(result.footprints[1][p])
        for t in 1:nsubsteps
            acc .+= result.footprints[t][p]
        end
        acc
    end

    println("\n" * "=" ^ 72)
    println("6-hour accumulated surface flux footprint summary")
    println("=" ^ 72)
    total_sensitivity = sum(sum(acc) for acc in accumulated)
    max_sensitivity = maximum(maximum(abs.(acc)) for acc in accumulated)
    @printf("Total cells with |footprint| > 0   : %d / %d\n",
            sum(count(!iszero, acc) for acc in accumulated),
            6 * Nc * Nc)
    @printf("Σ over all cells (signed)          : %+.4e\n", total_sensitivity)
    @printf("Max |dJ/dE| anywhere on the sphere : %+.4e\n", max_sensitivity)

    # Per-panel summary.
    println("\nPer-panel sensitivity (Σ |dJ/dE| over panel):")
    for p in 1:6
        s = sum(abs, accumulated[p])
        marker = p == la.panel ? "  ← LA panel" : ""
        @printf("  panel %d : %+.4e%s\n", p, s, marker)
    end

    # Top-10 most sensitive cells.
    println("\nTop-10 most sensitive (panel, i, j) cells:")
    all_cells = [(p, i, j, accumulated[p][i, j])
                 for p in 1:6 for j in 1:Nc for i in 1:Nc]
    sort!(all_cells, by = t -> -abs(t[4]))
    for (rank, (p, i, j, v)) in enumerate(all_cells[1:min(10, end)])
        rank > 10 && break
        lons, lats = AT.Grids.panel_cell_center_lonlat(problem.mesh, p)
        @printf("  #%2d  panel=%d (i,j)=(%2d,%2d)  lon=%+8.2f lat=%+7.2f  dJ/dE=%+10.3e\n",
                rank, p, i, j, lons[i, j], lats[i, j], v)
    end

    # Save accumulated footprint per panel to a small binary file for
    # downstream plotting (one Float64 array per panel, layout
    # (panel_idx, footprint_Nc_by_Nc)).
    out_path = joinpath(@__DIR__, "..", "..",
                        "artifacts", "linrood_la_footprint_c$(Nc)_6h.bin")
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        # Header: Nc, Nz, nsteps as Int64, then panel index + footprint Nc×Nc as Float64.
        write(io, Int64(Nc))
        write(io, Int64(Nz))
        write(io, Int64(nsteps))
        write(io, Int64(la.panel))
        write(io, Int64(la.i))
        write(io, Int64(la.j))
        for p in 1:6
            write(io, accumulated[p])
        end
    end
    @printf("\nFootprint saved to %s (%.1f KB)\n", out_path,
            filesize(out_path) / 1024)
end

main()
