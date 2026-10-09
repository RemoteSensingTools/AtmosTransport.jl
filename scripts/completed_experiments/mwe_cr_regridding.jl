#!/usr/bin/env julia
# ===========================================================================
# MWE: Conservative regridding LL → "reduced" grid using CR.jl directly
#
# Tests whether ConservativeRegridding.jl preserves total mass when
# regridding from a regular lat-lon grid to a reduced grid (variable
# cells per latitude ring, like reduced Gaussian).
#
# Uses ONLY CR.jl types — no AtmosTransport wrappers.
# ===========================================================================

using ConservativeRegridding
using ConservativeRegridding: Regridder, regrid!, CellBasedGrid
using ConservativeRegridding.Trees
using GeometryOps
using GeometryOps.UnitSpherical: UnitSphericalPoint
using SparseArrays
using Statistics
using Printf

# --- Helper: build a UnitSphericalPoint grid from face coordinates ---------
#     Returns (Nx+1 × Ny+1) matrix — CR.jl infers Spherical manifold from
#     the UnitSphericalPoint element type.

const TO_SPHERE = GeometryOps.UnitSphereFromGeographic()

function ll_point_matrix(lon_faces, lat_faces)
    pts = Matrix{UnitSphericalPoint}(undef, length(lon_faces), length(lat_faces))
    for (j, lat) in enumerate(lat_faces), (i, lon) in enumerate(lon_faces)
        pts[i, j] = TO_SPHERE((Float64(lon), Float64(lat)))
    end
    return pts
end

# --- Helper: build a (nlon+1 × 2) point matrix for one latitude ring ------

function ring_point_matrix(nlon, lat_south, lat_north)
    lon_faces = range(0.0, 360.0, length=nlon+1)
    lat_s = max(lat_south, -89.999)
    lat_n = min(lat_north,  89.999)
    pts = Matrix{UnitSphericalPoint}(undef, nlon+1, 2)
    for (i, lon) in enumerate(lon_faces)
        pts[i, 1] = TO_SPHERE((Float64(lon), lat_s))
        pts[i, 2] = TO_SPHERE((Float64(lon), lat_n))
    end
    return pts
end

# --- Diagnostic: check regridder mass conservation -------------------------

function check_regridder(label, r)
    n_src = length(r.src_areas)
    n_dst = length(r.dst_areas)

    src_field = ones(n_src)
    dst_field = zeros(n_dst)
    regrid!(dst_field, r, src_field)

    src_total = sum(src_field .* r.src_areas)
    dst_total = sum(dst_field .* r.dst_areas)

    covered_src = vec(sum(r.intersections, dims=1))
    frac_a = covered_src ./ r.src_areas

    rel_err = abs(dst_total - src_total) / src_total
    status = rel_err < 1e-6 ? "PASS" : (rel_err < 0.01 ? "WARN" : "FAIL")

    @printf("  [%s] %s\n", status, label)
    @printf("    src=%d  dst=%d  rel_err=%.2e\n", n_src, n_dst, rel_err)
    @printf("    frac_a: min=%.4f mean=%.4f max=%.4f  |  zero=%d (%.1f%%)\n",
            minimum(frac_a), mean(frac_a), maximum(frac_a),
            count(frac_a .== 0), 100*count(frac_a .== 0)/n_src)
    return (; rel_err, frac_a)
end

# ===========================================================================

function main()
    println("=" ^ 70)
    println("MWE: ConservativeRegridding.jl — mass conservation tests")
    println("All grids use UnitSphericalPoint (Spherical manifold).")
    println("=" ^ 70)

    # ---- Test 1: LL → LL (coarsening, same type) ----
    println("\n--- Test 1: LL 36×18 → LL 12×6 ---")
    src1 = ll_point_matrix(range(0.0, 360.0, 37), range(-89.999, 89.999, 19))
    dst1 = ll_point_matrix(range(0.0, 360.0, 13), range(-89.999, 89.999, 7))
    r1 = Regridder(dst1, src1)
    check_regridder("LL 36×18 → LL 12×6", r1)

    # ---- Test 2: LL → LL self-regrid ----
    println("\n--- Test 2: LL 36×18 self-regrid ---")
    r2 = Regridder(src1, src1)
    check_regridder("LL 36×18 self-regrid", r2)

    # ---- Test 3: LL → single equatorial ring ----
    println("\n--- Test 3: LL 36×18 → single equatorial ring (nlon=12) ---")
    ring_eq = ring_point_matrix(12, -5.0, 5.0)
    r3 = Regridder(ring_eq, src1)
    check_regridder("LL → equatorial ring", r3)

    # ---- Test 4: LL → full reduced grid (ring-by-ring) ---
    # This mimics what our RG regridding does: one Regridder per ring
    println("\n--- Test 4: LL 120×60 → Reduced 30 rings (ring-by-ring) ---")
    Nx_s, Ny_s = 120, 60
    src4 = ll_point_matrix(range(0.0, 360.0, Nx_s+1), range(-89.999, 89.999, Ny_s+1))

    # Build "reduced" grid: nlon tapers with cos(lat)
    nrings = 30
    lat_faces4 = range(-90.0, 90.0, nrings+1) |> collect
    nlon_eq = 120

    n_src4 = Nx_s * Ny_s
    covered_total = zeros(n_src4)
    all_dst_areas = Float64[]
    all_dst_vals = Float64[]
    src4_field = ones(n_src4)

    for j in 1:nrings
        lat_s, lat_n = lat_faces4[j], lat_faces4[j+1]
        lat_c = 0.5 * (lat_s + lat_n)
        nlon = max(20, round(Int, nlon_eq * cosd(lat_c)))

        rp = ring_point_matrix(nlon, lat_s, lat_n)
        rr = Regridder(rp, src4)

        append!(all_dst_areas, rr.dst_areas)
        ring_dst = zeros(nlon)
        regrid!(ring_dst, rr, src4_field)
        append!(all_dst_vals, ring_dst)

        covered_total .+= vec(sum(rr.intersections, dims=1))
    end

    # Source areas from self-regrid
    r_self4 = Regridder(src4, src4)
    src_areas4 = r_self4.src_areas

    src_total4 = sum(src4_field .* src_areas4)
    dst_total4 = sum(all_dst_vals .* all_dst_areas)
    frac_a4 = covered_total ./ src_areas4
    rel_err4 = abs(dst_total4 - src_total4) / src_total4
    status4 = rel_err4 < 1e-6 ? "PASS" : (rel_err4 < 0.01 ? "WARN" : "FAIL")

    @printf("  [%s] LL 120×60 → Reduced 30 rings\n", status4)
    @printf("    src=%d  dst=%d  rel_err=%.2e\n", n_src4, length(all_dst_areas), rel_err4)
    @printf("    frac_a: min=%.4f mean=%.4f max=%.4f  |  zero=%d (%.1f%%)\n",
            minimum(frac_a4), mean(frac_a4), maximum(frac_a4),
            count(frac_a4 .== 0), 100*count(frac_a4 .== 0)/n_src4)

    # ---- Test 5: LL → full reduced (as single CellBasedGrid) ----
    # Instead of ring-by-ring, build one big ExplicitPolygonGrid
    # Actually, let's just test a single wide ring that's the full grid
    println("\n--- Test 5: LL 36×18 → single full-sphere ring (nlon=36) ---")
    full_ring = ring_point_matrix(36, -89.999, 89.999)
    r5 = Regridder(full_ring, src1)
    check_regridder("LL → full-sphere single ring (36 cells)", r5)

    # ---- Test 6: higher res LL → reduced ring-by-ring ----
    println("\n--- Test 6: LL 360×180 → Reduced 90 rings ---")
    Nx_s6, Ny_s6 = 360, 180
    src6 = ll_point_matrix(range(0.0, 360.0, Nx_s6+1), range(-89.999, 89.999, Ny_s6+1))
    n_src6 = Nx_s6 * Ny_s6
    covered6 = zeros(n_src6)
    dst_areas6 = Float64[]
    dst_vals6 = Float64[]
    src6_field = ones(n_src6)

    nrings6 = 90
    lat_faces6 = range(-90.0, 90.0, nrings6+1) |> collect
    for j in 1:nrings6
        lat_s, lat_n = lat_faces6[j], lat_faces6[j+1]
        lat_c = 0.5 * (lat_s + lat_n)
        nlon = max(20, round(Int, 360 * cosd(lat_c)))

        rp = ring_point_matrix(nlon, lat_s, lat_n)
        rr = Regridder(rp, src6)

        append!(dst_areas6, rr.dst_areas)
        ring_dst = zeros(nlon)
        regrid!(ring_dst, rr, src6_field)
        append!(dst_vals6, ring_dst)
        covered6 .+= vec(sum(rr.intersections, dims=1))
    end

    r_self6 = Regridder(src6, src6)
    src_areas6 = r_self6.src_areas
    src_total6 = sum(src6_field .* src_areas6)
    dst_total6 = sum(dst_vals6 .* dst_areas6)
    frac_a6 = covered6 ./ src_areas6
    rel_err6 = abs(dst_total6 - src_total6) / src_total6
    status6 = rel_err6 < 1e-6 ? "PASS" : (rel_err6 < 0.01 ? "WARN" : "FAIL")

    @printf("  [%s] LL 360×180 → Reduced 90 rings\n", status6)
    @printf("    src=%d  dst=%d  rel_err=%.2e\n", n_src6, length(dst_areas6), rel_err6)
    @printf("    frac_a: min=%.4f mean=%.4f max=%.4f  |  zero=%d (%.1f%%)\n",
            minimum(frac_a6), mean(frac_a6), maximum(frac_a6),
            count(frac_a6 .== 0), 100*count(frac_a6 .== 0)/n_src6)

    println("\n" * "=" ^ 70)
    println("Done.")
    println("=" ^ 70)
end

main()
