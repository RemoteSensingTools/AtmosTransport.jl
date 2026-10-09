#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 25 — 6-hour LinRood adjoint footprint at LA on the production C180
# cubed sphere, driven by the pre-processed ERA5 transport binary.
#
# Loads `era5_transport_20211202_merged1000Pa_float32.bin` (24 windows ×
# 8 substeps × 450 s = 1 day) and runs the adjoint backward for the
# first 6 hours = 48 substeps.
#
# Uses the same loaders as `scripts/benchmarks/run_cs_transport.jl`
# (`TransportBinaryReader`,
# `load_window!`, `_pad` to Hp=3). Memory budget is bounded by the
# LinRood horizontal tape (~5-10 GB) and `_CSSweepRecord` Z snapshots
# (~10-15 GB). Total comfortably fits in 64 GB+ host memory.
# ---------------------------------------------------------------------------

using Printf
using Dates

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport: CubedSphereMesh
using .AtmosTransport.Adjoints: cs_surface_emission_footprint,
                                CSColumnMeanObjective
using .AtmosTransport.MetDrivers: TransportBinaryReader, load_window!,
                                  mesh_definition
using .AtmosTransport.Operators.Advection: LinRoodPPMScheme, fill_panel_halos!
using .AtmosTransport.Grids: panel_cell_center_lonlat

const FT = Float64
const Hp = 3
const BIN_PATH = "/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/era5_transport_20211202_merged1000Pa_float32.bin"
const SIX_HOUR_WINDOWS = 6   # 6 hourly met windows
const ARTIFACT_DIR = joinpath(@__DIR__, "..", "..", "artifacts")

# Pad an interior (Nc, Nc, Nz) panel into a haloed (Nc+2Hp, Nc+2Hp, Nz)
# panel by zero-filling the halo cells. Halos get refreshed at the
# start of each LinRood substep via fill_panel_halos! inside the tape
# recorder.
function _pad_panel(a::AbstractArray{T, 3}, Hp) where {T}
    Nx, Ny, Nz = size(a)
    p = zeros(T, Nx + 2Hp, Ny + 2Hp, Nz)
    @inbounds p[Hp+1:Hp+Nx, Hp+1:Hp+Ny, :] .= a
    return p
end
# Face arrays (already Nc±1 × Nc × Nz on disk): pad the cell-shape
# dimension into Hp arrays. For LinRood:
#   am: (Nc+1, Nc, Nz) → (Nc+1+2Hp_face_x, Nc+2Hp, Nz)?
# Actually looking at fv_tp_2d_cs!, the face panels are referenced
# as am[i, j, k], bm[i, j, k] with i, j the FACE indices 1..Nc+1 in
# the active direction and 1..Nc in the orthogonal direction.
# The kernels do NOT read halo face cells — they only touch interior
# faces. So face arrays should be supplied at the (Nc+1, Nc, Nz) /
# (Nc, Nc+1, Nz) shape directly, no padding needed. BUT the C180
# binary stores them at (Nc, Nc, Nz) and (Nc, Nc, Nz) — let me check
# the actual on-disk shapes via load_window!.

function _build_meteo_step_sequence(reader, nsteps_total, AT)
    # Load N windows × M substeps, where each substep within a window
    # shares the same (am, bm, cm). nsteps_total = N × M.
    steps_per_window = reader.header.steps_per_window
    length(unique(reader.header.steps_per_window_by_window)) == 1 ||
        error("linrood_la_footprint_c180.jl currently assumes a constant " *
              "steps_per_window schedule; variable-step binaries need a " *
              "per-step dt/metadata path in this diagnostic.")
    @assert nsteps_total % steps_per_window == 0
        "nsteps_total ($nsteps_total) must be a multiple of steps_per_window ($steps_per_window)"
    n_windows = nsteps_total ÷ steps_per_window

    # Storage. Use the panel-tuple type returned by load_window!.
    pm_w1, _, _, _, _ = load_window!(reader, 1)
    Nc, _, Nz = size(pm_w1[1])
    panels_am_steps = Vector{Any}(undef, nsteps_total)
    panels_bm_steps = Vector{Any}(undef, nsteps_total)
    panels_cm_steps = Vector{Any}(undef, nsteps_total)

    for win in 1:n_windows
        @printf("  loading window %2d / %d\n", win, n_windows)
        _, _, pam_w, pbm_w, pcm_w = load_window!(reader, win)
        # Convert per-window fluxes to a per-substep "rate" by dividing
        # by steps_per_window. Face arrays remain UNPADDED — the LinRood
        # kernels read am/bm at kernel-local indices `[i, jf, k]`
        # (no Hp offset), so face arrays should be shaped
        # (Nc+1, Nc, Nz) for am, (Nc, Nc+1, Nz) for bm,
        # (Nc, Nc, Nz+1) for cm. Cell arrays (m) ARE padded to Hp=3.
        # flux_kind-aware: convert STORED amounts to per-palindrome-
        # application amounts (1 for legacy per-substep storage; 1/(2*steps)
        # for full_window_mass_amount binaries) so the rest of the script
        # always sees legacy units. am/bm stay UNPADDED for the LinRood
        # kernel convention; cm gets padded for the Z-sweep convention.
        fs = AtmosTransport.MetDrivers.flux_storage_substep_scale(
            FT, reader.header.steps_per_window_by_window[win],
            AtmosTransport.MetDrivers.flux_kind(reader))
        pam = ntuple(p -> AT(FT.(pam_w[p]) .* fs), 6)
        pbm = ntuple(p -> AT(FT.(pbm_w[p]) .* fs), 6)
        pcm = ntuple(p -> AT(_pad_panel(FT.(pcm_w[p]) .* fs, Hp)), 6)
        for sub in 1:steps_per_window
            step_idx = (win - 1) * steps_per_window + sub
            panels_am_steps[step_idx] = pam
            panels_bm_steps[step_idx] = pbm
            panels_cm_steps[step_idx] = pcm
        end
    end
    return panels_am_steps, panels_bm_steps, panels_cm_steps, Nc, Nz
end

function _find_la_cell(mesh::CubedSphereMesh)
    la_lat = 34.0
    la_lon = -118.0
    best_panel, best_i, best_j = 0, 0, 0
    best_dist2 = Inf
    for p in 1:6
        lons, lats = panel_cell_center_lonlat(mesh, p)
        for j in 1:mesh.Nc, i in 1:mesh.Nc
            dlon = lons[i, j] - la_lon
            dlon = mod(dlon + 180, 360) - 180
            dlat = lats[i, j] - la_lat
            d2 = dlon^2 + dlat^2
            if d2 < best_dist2
                best_dist2 = d2
                best_panel, best_i, best_j = p, i, j
            end
        end
    end
    return (panel=best_panel, i=best_i, j=best_j, dist_deg=sqrt(best_dist2))
end

function main()
    println("=" ^ 72)
    println("Plan 25 — 6-hour LinRood adjoint footprint at LA on C180 ERA5")
    println("=" ^ 72)

    @printf("Binary:  %s\n", BIN_PATH)
    reader = TransportBinaryReader(BIN_PATH; FT)
    h = reader.header
    Nc = h.geometry.Nc; Nz = h.nlevel
    steps_per_window = h.steps_per_window
    nsteps_total = SIX_HOUR_WINDOWS * steps_per_window
    dt_substep = h.dt_met_seconds / steps_per_window
    @printf("Grid:    C%d × %d levels, Hp=%d, dt_substep=%.0f s\n",
            Nc, Nz, Hp, dt_substep)
    @printf("Window:  %d windows × %d substeps = %d steps = %.2f h\n",
            SIX_HOUR_WINDOWS, steps_per_window, nsteps_total,
            SIX_HOUR_WINDOWS * h.dt_met_seconds / 3600)

    AT = Array  # CPU run; switch to CuArray for GPU
    mesh = CubedSphereMesh(; Nc, Hp, definition=mesh_definition(reader))

    la = _find_la_cell(mesh)
    lons, lats = panel_cell_center_lonlat(mesh, la.panel)
    @printf("LA cell: panel=%d (i,j)=(%d, %d)  lon=%+8.3f  lat=%+7.3f  (%.2f° from LA)\n",
            la.panel, la.i, la.j, lons[la.i, la.j], lats[la.i, la.j],
            la.dist_deg)

    # Initial state from the binary's first window (interior-only),
    # padded to Hp=3 and halo-filled across panels.
    print("Loading initial state... ")
    pm_w1, _, _, _, _ = load_window!(reader, 1)
    panels_m0 = ntuple(p -> AT(_pad_panel(FT.(pm_w1[p]), Hp)), 6)
    fill_panel_halos!(panels_m0, mesh; dir=0)
    # Tracer starts at zero — surface emissions seeded by base_emission_rates
    # would only matter if we wanted the J value; here we only want dJ/dE.
    panels_rm0 = ntuple(p -> begin
        a = similar(panels_m0[p]); fill!(a, zero(FT)); a
    end, 6)
    println("done.")

    print("Loading meteo step sequence ($nsteps_total substeps)...\n")
    panels_am, panels_bm, panels_cm, _, _ =
        _build_meteo_step_sequence(reader, nsteps_total, AT)
    println("done.")

    obj = CSColumnMeanObjective(la.panel, la.i, la.j)

    print("\nRunning forward + adjoint pass (LinRoodPPMScheme, C$Nc × $Nz, $nsteps_total substeps)...\n")
    flush(stdout)
    t0 = time()
    result = cs_surface_emission_footprint(
        panels_rm0, panels_m0,
        panels_am, panels_bm, panels_cm,
        mesh, obj;
        scheme = LinRoodPPMScheme(),
        dt = FT(dt_substep),
    )
    elapsed = time() - t0
    @printf("Adjoint reverse pass completed in %.1f s\n", elapsed)

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
    abs_total = sum(sum(abs, acc) for acc in accumulated)
    max_sensitivity = maximum(maximum(abs.(acc)) for acc in accumulated)
    nonzero = sum(count(!iszero, acc) for acc in accumulated)
    total_cells = 6 * Nc * Nc
    @printf("Total cells with |dJ/dE| > 0   : %d / %d  (%.2f%%)\n",
            nonzero, total_cells, 100 * nonzero / total_cells)
    @printf("Σ |dJ/dE|                       : %+.4e\n", abs_total)
    @printf("Peak |dJ/dE|                    : %+.4e\n", max_sensitivity)

    println("\nPer-panel Σ |dJ/dE|:")
    for p in 1:6
        s = sum(abs, accumulated[p])
        marker = p == la.panel ? "  ← LA panel" : ""
        @printf("  panel %d : %+.4e%s\n", p, s, marker)
    end

    # Save: raw bin + CSV
    mkpath(ARTIFACT_DIR)
    bin_path = joinpath(ARTIFACT_DIR, "linrood_la_footprint_c180_6h.bin")
    open(bin_path, "w") do io
        write(io, Int64(Nc))
        write(io, Int64(Nz))
        write(io, Int64(nsubsteps))
        write(io, Int64(la.panel)); write(io, Int64(la.i)); write(io, Int64(la.j))
        for p in 1:6
            write(io, Array{Float64}(accumulated[p]))
        end
    end
    @printf("\nBinary saved to %s (%.2f MB)\n",
            bin_path, filesize(bin_path) / 1024^2)

    csv_path = joinpath(ARTIFACT_DIR, "linrood_la_footprint_c180_6h.csv")
    open(csv_path, "w") do out
        @printf(out, "# Nc=%d Nz=%d nsteps=%d la_panel=%d la_i=%d la_j=%d\n",
                Nc, Nz, nsubsteps, la.panel, la.i, la.j)
        println(out, "panel,i,j,lon,lat,dJdE")
        for p in 1:6
            lons_p, lats_p = panel_cell_center_lonlat(mesh, p)
            for j in 1:Nc, i in 1:Nc
                @printf(out, "%d,%d,%d,%.6f,%.6f,%.6e\n",
                        p, i, j, lons_p[i, j], lats_p[i, j], accumulated[p][i, j])
            end
        end
    end
    @printf("CSV saved to %s (%.2f MB)\n",
            csv_path, filesize(csv_path) / 1024^2)
end

main()
