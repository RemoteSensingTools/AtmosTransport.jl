#!/usr/bin/env julia
# Cubed-sphere face-flux reconstruction from cell-centre winds: face lengths,
# the legacy centreline choice, the divergence of a solid-body rotation, vector
# face fluxes, and the vector regridding of the winds.

using Test, Statistics
using AtmosTransport
const G = AtmosTransport.Grids
const P = AtmosTransport.Preprocessing

gmao_mesh(Nc) = let conv = G.GEOSNativePanelConvention()
    CubedSphereMesh(; Nc, Hp = 0, FT = Float64, convention = conv,
                    definition = G.GMAOCubedSphereDefinition(; convention = conv))
end

# Solid-body rotation about an axis tilted by `tilt` toward lon = 0, at cell centres.
function solid_body_winds(mesh, U, tilt)
    Nc = mesh.Nc
    u_e = ntuple(_ -> zeros(Nc, Nc, 1), 6); v_n = ntuple(_ -> zeros(Nc, Nc, 1), 6)
    for p in 1:6
        lon, lat = G.panel_cell_center_lonlat(mesh, p)
        for j in 1:Nc, i in 1:Nc
            λ, φ = deg2rad(lon[i, j]), deg2rad(lat[i, j])
            u_e[p][i, j, 1] = U * (cos(φ) * cos(tilt) + sin(φ) * cos(λ) * sin(tilt))
            v_n[p][i, j, 1] = -U * sin(λ) * sin(tilt)
        end
    end
    u = ntuple(_ -> zeros(Nc, Nc, 1), 6); v = ntuple(_ -> zeros(Nc, Nc, 1), 6)
    P.rotate_winds_to_panel_local!(u, v, u_e, v_n, mesh, 1)
    return u, v
end

# Exact face fluxes R ∫ u·n ds of a tangent wind `wind(x)` along each face's
# great-circle arc (3-point Gauss–Legendre), oriented from left to right cell.
function exact_face_fluxes(mesh, ft, wind)
    centre = P._cs_cell_bases(mesh)[1]
    gx, gw = (-sqrt(0.6), 0.0, sqrt(0.6)), (5 / 9, 8 / 9, 5 / 9)
    map(1:ft.nf) do f
        a, b = P._cs_face_corners(mesh, ft, f)
        n = P._unit3(P._cross3(a, b))
        n = P._dot3(n, centre[ft.face_right[f]] .- centre[ft.face_left[f]]) >= 0 ? n : .-n
        θ = P._arc3(a, b)
        s = sum(w * P._dot3(wind((sin(θ * (1 - x) / 2) .* a .+ sin(θ * (1 + x) / 2) .* b) ./ sin(θ)), n)
                for (x, w) in zip(gx, gw))
        mesh.radius * s * θ / 2
    end
end

# East/north components of `wind(x)` at the cell centres.
function cell_winds(mesh, wind)
    Nc = mesh.Nc
    u_e = ntuple(_ -> zeros(Nc, Nc, 1), 6); v_n = ntuple(_ -> zeros(Nc, Nc, 1), 6)
    for p in 1:6
        lon, lat = G.panel_cell_center_lonlat(mesh, p)
        for j in 1:Nc, i in 1:Nc
            λ, φ = deg2rad(lon[i, j]), deg2rad(lat[i, j])
            w = wind((cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ)))
            u_e[p][i, j, 1] = P._dot3(w, (-sin(λ), cos(λ), 0.0))
            v_n[p][i, j, 1] = P._dot3(w, (-sin(φ) * cos(λ), -sin(φ) * sin(λ), cos(φ)))
        end
    end
    return u_e, v_n
end

canonical_fluxes(am, bm, ft) =
    [ft.face_dir[f] == 1 ? am[ft.face_panel[f]][ft.face_idx_i[f], ft.face_idx_j[f], 1] :
                           bm[ft.face_panel[f]][ft.face_idx_i[f], ft.face_idx_j[f], 1] for f in 1:ft.nf]

interior_divergence(am, bm, Nc) =
    [am[p][i, j, 1] - am[p][i + 1, j, 1] + bm[p][i, j, 1] - bm[p][i, j + 1, 1]
     for p in 1:6, i in 2:Nc-1, j in 2:Nc-1]

@testset "CS face fluxes" begin
    @testset "face edge lengths" begin
        mesh = gmao_mesh(24)
        Lx, Ly = G.cs_face_edge_lengths(mesh)
        @test size(Lx) == (25, 24) && size(Ly) == (24, 25)
        # A panel edge is a great-circle arc between two cube corners: acos(1/3).
        @test sum(Lx[1, :]) ≈ mesh.radius * acos(1 / 3) rtol = 1e-12
        @test sum(Ly[:, end]) ≈ mesh.radius * acos(1 / 3) rtol = 1e-12
        @test Lx ≈ reverse(Lx; dims = 1) rtol = 1e-12     # panel symmetry
        @test Lx' ≈ Ly rtol = 1e-12
    end

    @testset "default keeps the centreline lengths" begin
        mesh, Nc, Nz = gmao_mesh(8), 8, 2
        u = ntuple(p -> rand(Nc, Nc, Nz), 6); v = ntuple(p -> rand(Nc, Nc, Nz), 6)
        ps = ntuple(_ -> 1e5 .+ 100rand(Nc, Nc), 6)
        A, B = [0.0, 5000.0, 0.0], [0.0, 0.3, 1.0]
        new = [ntuple(_ -> zeros(Nc + 1, Nc, Nz), 6), ntuple(_ -> zeros(Nc, Nc + 1, Nz), 6)]
        P.reconstruct_cs_fluxes!(new[1], new[2], u, v, ntuple(_ -> zeros(Nc, Nc, Nz), 6), ps, A, B,
                                 mesh.Δx, mesh.Δy, 9.8, 0.5, Nc, Nz)
        # the historical formula: the high-side cell's centreline width, edge cells at the panel edge
        for p in 1:6, k in 1:Nz, j in 1:Nc, i in 2:Nc
            dp(ii) = abs((A[k] - A[k + 1]) + (B[k] - B[k + 1]) * ps[p][ii, j])
            ref = 0.5 * (u[p][i - 1, j, k] + u[p][i, j, k]) * 0.5 * (dp(i - 1) + dp(i)) * mesh.Δy[i, j] / 9.8 * 0.5
            @test new[1][p][i, j, k] == ref
        end
    end

    @testset "solid-body rotation is nearly divergence-free with face edge lengths" begin
        mesh = gmao_mesh(48); Nc = mesh.Nc
        dps = ntuple(_ -> fill(1000.0, Nc, Nc, 1), 6)
        scale = 30.0 * 1000.0 * mean(mesh.Δy) / 9.80665
        for tilt in (0.0, π / 4)
            u, v = solid_body_winds(mesh, 30.0, tilt)
            rms = map((P.CellCenterlineLengths(mesh.Δx, mesh.Δy), P.EdgeLengths(mesh))) do L
                am = ntuple(_ -> zeros(Nc + 1, Nc, 1), 6); bm = ntuple(_ -> zeros(Nc, Nc + 1, 1), 6)
                P.cs_face_fluxes!(am, bm, u, v, dps, L, 9.80665, 1.0, Nc, 1)
                sqrt(mean(abs2, interior_divergence(am, bm, Nc))) / scale
            end
            @test rms[2] < rms[1] / 20
            @test rms[2] < 1e-5
        end
    end

    @testset "vector face fluxes: orientation, seams and mirrors" begin
        mesh = gmao_mesh(48); Nc = mesh.Nc
        conn = G.panel_connectivity_for(mesh.convention)
        ft = P.build_cs_global_face_table(Nc, conn)
        geom = P.CSVectorFaceGeometry(mesh, ft)
        dps = ntuple(_ -> fill(1000.0, Nc, Nc, 1), 6)
        scale = 30.0 * 1000.0 * mean(mesh.Δy) / 9.80665
        seam = [min(i, j, Nc + 1 - i, Nc + 1 - j) == 1 for p in 1:6, i in 1:Nc, j in 1:Nc]
        for tilt in (0.0, π / 4)
            u_e = ntuple(_ -> zeros(Nc, Nc, 1), 6); v_n = ntuple(_ -> zeros(Nc, Nc, 1), 6)
            for p in 1:6
                lon, lat = G.panel_cell_center_lonlat(mesh, p)
                for j in 1:Nc, i in 1:Nc
                    λ, φ = deg2rad(lon[i, j]), deg2rad(lat[i, j])
                    u_e[p][i, j, 1] = 30.0 * (cos(φ) * cos(tilt) + sin(φ) * cos(λ) * sin(tilt))
                    v_n[p][i, j, 1] = -30.0 * sin(λ) * sin(tilt)
                end
            end
            vec_am = ntuple(_ -> zeros(Nc + 1, Nc, 1), 6); vec_bm = ntuple(_ -> zeros(Nc, Nc + 1, 1), 6)
            P.cs_vector_face_fluxes!(vec_am, vec_bm, u_e, v_n, dps, geom, ft, 9.80665, 1.0, Nc, 1)
            u, v = solid_body_winds(mesh, 30.0, tilt)
            avg_am = ntuple(_ -> zeros(Nc + 1, Nc, 1), 6); avg_bm = ntuple(_ -> zeros(Nc, Nc + 1, 1), 6)
            P.cs_face_fluxes!(avg_am, avg_bm, u, v, dps, P.EdgeLengths(mesh), 9.80665, 1.0, Nc, 1)
            P._sync_cs_mirrors!(avg_am, avg_bm, ft, 1)
            # same orientation and size at interior faces (they differ only in how
            # the two cells' winds are combined)
            interior = [vec_am[p][i, j, 1] - avg_am[p][i, j, 1] for p in 1:6, i in 3:Nc-1, j in 2:Nc-1]
            @test maximum(abs, interior) < 1e-3 * scale
            div(am, bm) = [am[p][i, j, 1] - am[p][i + 1, j, 1] + bm[p][i, j, 1] - bm[p][i, j + 1, 1]
                           for p in 1:6, i in 1:Nc, j in 1:Nc] ./ scale
            d_vec, d_avg = div(vec_am, vec_bm), div(avg_am, avg_bm)
            @test sqrt(mean(abs2, d_vec[.!seam])) < 3e-6      # second order: 2e-7 at C90
            # the along-edge seam interpolation brings seam cells to near-interior accuracy
            @test sqrt(mean(abs2, d_vec[seam])) < sqrt(mean(abs2, d_avg[seam])) / 50
            @test sqrt(mean(abs2, d_vec[seam])) < 5e-5
            # global sum of the divergence vanishes (shared faces cancel exactly)
            @test abs(sum(d_vec)) < 1e-9 * length(d_vec)
        end
    end

    @testset "every seam face has an along-edge partner" begin
        for Nc in (24, 48)
            mesh = gmao_mesh(Nc)
            ft = P.build_cs_global_face_table(Nc, G.panel_connectivity_for(mesh.convention))
            geom = P.CSVectorFaceGeometry(mesh, ft)
            # even Nc: every seam face is off the edge midpoint and gets a partner
            seam = ft.mirror_panel .!= 0
            @test all(geom.slot[seam] .!= 0) && all(geom.slot[.!seam] .== 0)
            @test all(0 .< geom.w_partner .< 0.5)
        end
    end

    @testset "fourth-order wind interpolation at interior faces" begin
        mesh = gmao_mesh(48); Nc = mesh.Nc
        ft = P.build_cs_global_face_table(Nc, G.panel_connectivity_for(mesh.convention))
        geo2, geo4 = P.CSVectorFaceGeometry(mesh, ft), P.CSVectorFaceGeometry(mesh, ft; order = 4)
        @test_throws ArgumentError P.CSVectorFaceGeometry(mesh, ft; order = 3)
        # the stencil needs two cells on each side in the panel row
        @test count(!=(0), geo4.outer.slot) == 12 * Nc * (Nc - 3) && geo2.outer === nothing
        @test all(f -> geo4.coef[f] == geo2.coef[f], findall(==(0), geo4.outer.slot))
        # a smooth divergent and rotational wind: interior face fluxes against exact line integrals
        a, b = (0.3, 0.5, 0.8) ./ sqrt(0.98), (0.1, -0.9, 0.3) ./ sqrt(0.91)
        wind(x) = let k = 12.0, w = 20 .* (cos(k * P._dot3(x, a)) .* a .+ P._cross3(x, cos(k * P._dot3(x, b)) .* b))
            w .- P._dot3(w, x) .* x
        end
        u_e, v_n = cell_winds(mesh, wind)
        dps = ntuple(_ -> fill(1.0, Nc, Nc, 1), 6)
        exact = exact_face_fluxes(mesh, ft, wind)
        interior = ft.mirror_panel .== 0
        err = map((geo2, geo4)) do geo
            am = ntuple(_ -> zeros(Nc + 1, Nc, 1), 6); bm = ntuple(_ -> zeros(Nc, Nc + 1, 1), 6)
            P.cs_vector_face_fluxes!(am, bm, u_e, v_n, dps, geo, ft, 1.0, 1.0, Nc, 1)
            sqrt(mean(abs2, (canonical_fluxes(am, bm, ft) .- exact)[interior]))
        end
        @test err[2] < err[1] / 2
    end

    @testset "FV3's filter along the faces" begin
        mesh = gmao_mesh(24); Nc = mesh.Nc
        ft = P.build_cs_global_face_table(Nc, G.panel_connectivity_for(mesh.convention))
        geo = P.CSVectorFaceGeometry(mesh, ft; order = 4, along_face_filter = true)
        face(p, d, i, j) = findfirst(f -> (ft.face_panel[f], ft.face_dir[f], ft.face_idx_i[f], ft.face_idx_j[f]) ==
                                          (p, d, i, j), 1:ft.nf)
        f = face(1, 1, 12, 12)                        # deep interior: the five faces along the grid line
        @test geo.filter.w[f] == (-1, 8, 18, 8, -1) ./ 32
        @test collect(geo.filter.face[f]) == [face(1, 1, 12, j) for j in 10:14]
        @test all(f -> sum(geo.filter.w[f]) ≈ 1, findall(==(0), ft.mirror_panel))   # interior: a weighted mean
        # a wind alternating along the faces of panel 1 (the 2Δ wave the filter removes)
        u_e = ntuple(p -> p == 1 ? [(-1.0)^j for i in 1:Nc, j in 1:Nc, k in 1:1] : zeros(Nc, Nc, 1), 6)
        v_n = ntuple(_ -> zeros(Nc, Nc, 1), 6); dps = ntuple(_ -> ones(Nc, Nc, 1), 6)
        flux(geom) = (am = ntuple(_ -> zeros(Nc + 1, Nc, 1), 6); bm = ntuple(_ -> zeros(Nc, Nc + 1, 1), 6);
                      P.cs_vector_face_fluxes!(am, bm, u_e, v_n, dps, geom, ft, 1.0, 1.0, Nc, 1); am[1])
        raw, filtered = flux(P.CSVectorFaceGeometry(mesh, ft; order = 4)), flux(geo)
        @test maximum(abs, filtered[3:Nc-1, 2:Nc-1, 1]) < 0.01 * maximum(abs, raw[3:Nc-1, 2:Nc-1, 1])
        @test filtered[3:Nc-1, [1, Nc], 1] == raw[3:Nc-1, [1, Nc], 1]       # first and last rows unfiltered
        # seam faces stay single-valued: mirror entries carry the filtered canonical flux
        u_e, v_n = cell_winds(mesh, x -> 30.0 .* (0.0, -x[3], x[2]))   # rotation about the x axis
        am = ntuple(_ -> zeros(Nc + 1, Nc, 2), 6); bm = ntuple(_ -> zeros(Nc, Nc + 1, 2), 6)
        u2, v2 = ntuple(p -> cat(u_e[p], u_e[p]; dims = 3), 6), ntuple(p -> cat(v_n[p], v_n[p]; dims = 3), 6)
        P.cs_vector_face_fluxes!(am, bm, u2, v2, ntuple(_ -> ones(Nc, Nc, 2), 6), geo, ft, 1.0, 1.0, Nc, 2)
        for f in findall(!=(0), ft.mirror_panel)
            get(a, b, d, p, i, j) = d == 1 ? a[p][i, j, 2] : b[p][i, j, 2]
            @test get(am, bm, ft.mirror_dir[f], ft.mirror_panel[f], ft.mirror_idx_i[f], ft.mirror_idx_j[f]) ==
                  ft.mirror_sign[f] * get(am, bm, ft.face_dir[f], ft.face_panel[f], ft.face_idx_i[f], ft.face_idx_j[f])
        end
        # the divergence-free rotation stays nearly divergence-free through the filter
        div = [am[p][i, j, 2] - am[p][i + 1, j, 2] + bm[p][i, j, 2] - bm[p][i, j + 1, 2] for p in 1:6, i in 1:Nc, j in 1:Nc]
        @test maximum(abs, div) < 1e-3 * maximum(abs, am[1])
    end

    @testset "layer thickness from the dry mass" begin
        mesh = gmao_mesh(8); Nc, Nz, g = 8, 3, 9.80665
        m = ntuple(_ -> 1e3 .* rand(Nc, Nc, Nz), 6)
        dp = ntuple(_ -> zeros(Nc, Nc, Nz), 6)
        x = (dp = dp,)
        P._fill_flux_thickness!(P.DryMassFluxThickness(), x, nothing, m, nothing, mesh, Nc, Nz)
        @test all(p -> dp[p] ≈ g .* m[p] ./ mesh.cell_areas, 1:6)
        ps = ntuple(_ -> fill(1e5, Nc, Nc), 6)
        vc = (A = [0.0, 1000.0, 3000.0, 0.0], B = [0.0, 0.0, 0.2, 1.0])
        P._fill_flux_thickness!(P.MoistFluxThickness(), x, ps, m, vc, mesh, Nc, Nz)
        @test dp[1][1, 1, :] ≈ [1000.0, 2000.0 + 0.2e5, 0.8e5 - 3000.0]
    end

    @testset "winds regridded as vectors near the poles" begin
        # 5° latitude-longitude source with pole points (half-width polar caps, as MERRA-2)
        nx, ny = 72, 37
        src = G.LatLonMesh{Float64}(nx, ny, 5.0, 5.0, [-180 + 5.0 * (i - 1) for i in 1:nx],
                                    [-182.5 + 5.0 * (i - 1) for i in 1:nx + 1],
                                    [-90 + 5.0 * (j - 1) for j in 1:ny],
                                    [-90; [-92.5 + 5.0 * (j - 1) for j in 2:ny]; 90], 6.371e6)
        mesh = CubedSphereMesh(; Nc = 12, Hp = 0, FT = Float64, radius = 6.371e6,
                               convention = G.GEOSNativePanelConvention(),
                               definition = G.GMAOCubedSphereDefinition(; convention = G.GEOSNativePanelConvention()))
        Nc, Nz = 12, 1
        R = AtmosTransport.Regridding.build_regridder(src, mesh; normalize = false)
        ws = P.allocate_cs_preprocess_workspace(Nc, nx, ny, Nz, length(R.src_areas), length(R.dst_areas), Float64)
        pipe = (u = ntuple(_ -> zeros(Nc, Nc, Nz), 6), v = ntuple(_ -> zeros(Nc, Nc, Nz), 6))
        V = (3.0, 11.0, 0.0)                                       # uniform flow across the pole
        en(λ, φ) = ((-sin(λ), cos(λ), 0.0), (-sin(φ) * cos(λ), -sin(φ) * sin(λ), cos(φ)))
        u = zeros(nx, ny, Nz); v = zeros(nx, ny, Nz)
        for j in 1:ny, i in 1:nx
            e, n = en(deg2rad(src.λᶜ[i]), deg2rad(src.φᶜ[j]))
            u[i, j, 1], v[i, j, 1] = P._dot3(V, e), P._dot3(V, n)
        end
        regrid!(panels, f) = P.regrid_3d_to_cs_panels!(panels, R, f, ws, Nc)
        polar_error = map((P.ScalarWindRegrid(), P.CartesianWindRegrid(src, mesh, Nz, Float64))) do w
            P._regrid_winds!(w, pipe.u, pipe.v, regrid!, u, v)
            lon, lat = G.panel_cell_center_lonlat(mesh, 3)       # the north-polar panel
            maximum(Iterators.product(1:Nc, 1:Nc)) do (i, j)
                e, n = en(deg2rad(lon[i, j]), deg2rad(lat[i, j]))
                hypot(pipe.u[3][i, j, 1] - P._dot3(V, e), pipe.v[3][i, j, 1] - P._dot3(V, n))
            end
        end
        @test polar_error[1] > 0.05 * hypot(V...)      # u and v remapped separately: > 5 % of |V|
        @test polar_error[2] < 0.1 * polar_error[1]   # the vector remap recovers the flow
        @test P._wind_regrid(:scalar, src, mesh, Nz, Float64) isa P.ScalarWindRegrid
        # Float32 buffers, as in production
        ws32 = P.allocate_cs_preprocess_workspace(Nc, nx, ny, Nz, length(R.src_areas), length(R.dst_areas), Float32)
        u32, v32 = ntuple(_ -> zeros(Float32, Nc, Nc, Nz), 6), ntuple(_ -> zeros(Float32, Nc, Nc, Nz), 6)
        P._regrid_winds!(P.CartesianWindRegrid(src, mesh, Nz, Float32), u32, v32,
                         (panels, f) -> P.regrid_3d_to_cs_panels!(panels, R, f, ws32, Nc),
                         Float32.(u), Float32.(v))
        @test maximum(p -> maximum(abs, u32[p] .- pipe.u[p]), 1:6) < 1e-4 * hypot(V...)
    end

    @testset "winds regridded as vectors from a reduced Gaussian grid (ERA5)" begin
        # rings 3° apart south to north, few longitudes near the poles as in N320
        # (cell centres at (i - ½) Δλ)
        lats = [-88.5 + 3.0 * (j - 1) for j in 1:60]
        rg = G.ReducedGaussianMesh(lats, [max(8, round(Int, 120 * cosd(φ))) for φ in lats];
                                   FT = Float64, radius = 6.371e6)
        mesh = CubedSphereMesh(; Nc = 12, Hp = 0, FT = Float64, radius = 6.371e6,
                               convention = G.GEOSNativePanelConvention(),
                               definition = G.GMAOCubedSphereDefinition(; convention = G.GEOSNativePanelConvention()))
        Nc, Nz = 12, 2
        R = AtmosTransport.Regridding.build_regridder(rg, mesh; normalize = false)
        lon, lat = P._source_cell_centers(rg)
        @test length(lon) == length(R.src_areas) && lat[1] == lats[1] && lat[end] == lats[end]
        @test lon[1:8] ≈ [(i - 0.5) * 45 for i in 1:8]       # the southernmost ring has 8 cells
        V = (3.0, 11.0, 0.0)
        en(λ, φ) = ((-sin(λ), cos(λ), 0.0), (-sin(φ) * cos(λ), -sin(φ) * sin(λ), cos(φ)))
        u = zeros(length(lon), Nz); v = similar(u)
        for c in eachindex(lon)
            e, n = en(deg2rad(lon[c]), deg2rad(lat[c]))
            u[c, :] .= P._dot3(V, e); v[c, :] .= P._dot3(V, n)
        end
        src_flat, dst_flat = zeros(length(lon), Nz), zeros(length(R.dst_areas), Nz)
        # Few-longitude polar rings leave the polar target cells a few percent short of
        # full coverage; `_regrid_intensive!` divides by the regridded constant.
        coverage = P.apply_regridder!(zeros(length(R.dst_areas)), R, ones(length(lon)))
        @test minimum(coverage) < 0.99
        @test P._regrid_intensive!(dst_flat, src_flat, R, coverage, fill(400.0, length(lon), Nz)) ≈
              fill(400.0, size(dst_flat)) rtol = 1e-12
        regrid!(panels, f) = (P._regrid_intensive!(dst_flat, src_flat, R, coverage, f);
                              P._unpack_flat_to_cs_panels_3d!(panels, dst_flat, Nc, Nz))
        u_cs, v_cs = ntuple(_ -> zeros(Nc, Nc, Nz), 6), ntuple(_ -> zeros(Nc, Nc, Nz), 6)
        polar_error = map((P.ScalarWindRegrid(), P.CartesianWindRegrid(rg, mesh, Nz, Float64))) do w
            P._regrid_winds!(w, u_cs, v_cs, regrid!, u, v)
            lonp, latp = G.panel_cell_center_lonlat(mesh, 3)
            maximum(Iterators.product(1:Nc, 1:Nc, 1:Nz)) do (i, j, k)
                e, n = en(deg2rad(lonp[i, j]), deg2rad(latp[i, j]))
                hypot(u_cs[3][i, j, k] - P._dot3(V, e), v_cs[3][i, j, k] - P._dot3(V, n))
            end
        end
        # a C12 cell touching the pole spans 90° of longitude: the scalar regrid is off by
        # ~13 % of |V| here, the vector regrid by < 1 %
        @test polar_error[1] > 0.05 * hypot(V...)
        @test polar_error[2] < 0.1 * polar_error[1]
        # winds must come level last: a transposed array is rejected, not scrambled
        @test_throws DimensionMismatch P._regrid_winds!(P.CartesianWindRegrid(rg, mesh, Nz, Float64),
                                                        u_cs, v_cs, regrid!, permutedims(u), permutedims(v))
    end

    @testset "seam thickness is interpolated along the edge too" begin
        # zonal solid-body wind times a smooth thickness field: the flux at a seam
        # face must use the thickness at the face midpoint, not between the centres
        mesh = gmao_mesh(48); Nc = mesh.Nc
        ft = P.build_cs_global_face_table(Nc, G.panel_connectivity_for(mesh.convention))
        geom = P.CSVectorFaceGeometry(mesh, ft)
        centre = P._cs_cell_bases(mesh)[1]
        thickness(x) = 1000.0 * (1 + 0.3 * x[3] + 0.2 * x[1])
        u_e = ntuple(_ -> zeros(Nc, Nc, 1), 6); v_n = ntuple(_ -> zeros(Nc, Nc, 1), 6)
        dps = ntuple(_ -> zeros(Nc, Nc, 1), 6)
        for p in 1:6
            lon, lat = G.panel_cell_center_lonlat(mesh, p)
            for j in 1:Nc, i in 1:Nc
                u_e[p][i, j, 1] = 30.0 * cos(deg2rad(lat[i, j]))
                dps[p][i, j, 1] = thickness(centre[P._cs_global_cell(i, j, p, Nc)])
            end
        end
        am = ntuple(_ -> zeros(Nc + 1, Nc, 1), 6); bm = ntuple(_ -> zeros(Nc, Nc + 1, 1), 6)
        P.cs_vector_face_fluxes!(am, bm, u_e, v_n, dps, geom, ft, 1.0, 1.0, Nc, 1)
        rel = Float64[]
        for f in findall(!=(0), ft.mirror_panel)
            a, b = P._cs_face_corners(mesh, ft, f)
            mid = P._unit3(a .+ b)
            n = P._unit3(P._cross3(a, b))
            n = P._dot3(n, centre[ft.face_right[f]] .- centre[ft.face_left[f]]) >= 0 ? n : .-n
            exact = 30.0 * P._dot3((-mid[2], mid[1], 0.0), n) * thickness(mid) * geom.len[f]
            p, i, j = Int(ft.face_panel[f]), Int(ft.face_idx_i[f]), Int(ft.face_idx_j[f])
            got = ft.face_dir[f] == 1 ? am[p][i, j, 1] : bm[p][i, j, 1]
            push!(rel, (got - exact) / (30.0 * 1000.0 * geom.len[f]))
        end
        @test sqrt(mean(abs2, rel)) < 1e-4
    end

    @testset "default flux construction is the historical reconstruct_cs_fluxes!" begin
        # the ERA5 N320 driver now calls the shared methods; the defaults must be bit-identical
        mesh = gmao_mesh(8); Nc, Nz = 8, 3
        ft = P.build_cs_global_face_table(Nc, G.panel_connectivity_for(mesh.convention))
        u_e = ntuple(_ -> randn(Nc, Nc, Nz), 6); v_n = ntuple(_ -> randn(Nc, Nc, Nz), 6)
        ps = ntuple(_ -> 1e5 .+ 500 .* rand(Nc, Nc), 6)
        A, B = [0.0, 3000.0, 8000.0, 0.0], [0.0, 0.0, 0.4, 1.0]
        defaults = (face_fluxes = :panel_average, face_lengths = :cell_centerline, face_interpolation = :linear)
        method = P._face_flux_method(defaults, (mesh = mesh, face_table = ft))
        panels(n) = ntuple(_ -> zeros(Nc + n[1], Nc + n[2], Nz), 6)
        x = (am = panels((1, 0)), bm = panels((0, 1)), u_local = panels((0, 0)), v_local = panels((0, 0)),
             dp = panels((0, 0)))
        P._prepare_cell_winds!(method, x, u_e, v_n, mesh, Nz)
        P.fill_cs_layer_thickness!(x.dp, ps, A, B, Nc, Nz)
        P._face_fluxes!(method, x, 9.80665, 0.25, Nc, Nz)
        old = (am = panels((1, 0)), bm = panels((0, 1)), u = panels((0, 0)), v = panels((0, 0)), dp = panels((0, 0)))
        P.rotate_winds_to_panel_local!(old.u, old.v, u_e, v_n, mesh, Nz)
        P.reconstruct_cs_fluxes!(old.am, old.bm, old.u, old.v, old.dp, ps, A, B, mesh.Δx, mesh.Δy,
                                 9.80665, 0.25, Nc, Nz)
        @test x.am == old.am && x.bm == old.bm
    end

    @testset "each flux method gets the winds it expects" begin
        mesh = gmao_mesh(8); Nc, Nz = 8, 2
        ft = P.build_cs_global_face_table(Nc, G.panel_connectivity_for(mesh.convention))
        u_e = ntuple(_ -> rand(Nc, Nc, Nz), 6); v_n = ntuple(_ -> rand(Nc, Nc, Nz), 6)
        x = (u_local = ntuple(_ -> zeros(Nc, Nc, Nz), 6), v_local = ntuple(_ -> zeros(Nc, Nc, Nz), 6))
        P._prepare_cell_winds!(P.VectorFaceFluxes(P.CSVectorFaceGeometry(mesh, ft), ft), x, u_e, v_n, mesh, Nz)
        @test x.u_local == u_e && x.v_local == v_n                     # east/north as they are
        P._prepare_cell_winds!(P.PanelAverageFluxes(P.EdgeLengths(mesh)), x, u_e, v_n, mesh, Nz)
        u_ref = ntuple(_ -> zeros(Nc, Nc, Nz), 6); v_ref = ntuple(_ -> zeros(Nc, Nc, Nz), 6)
        P.rotate_winds_to_panel_local!(u_ref, v_ref, u_e, v_n, mesh, Nz)
        @test x.u_local == u_ref && x.v_local == v_ref                 # panel-local components
    end

    @testset "flux options in the met-source TOML" begin
        mktempdir() do dir
            write_toml(name, body) = (path = joinpath(dir, name); write(path, body); path)
            merra = read(joinpath(pkgdir(AtmosTransport), "config", "met_sources", "merra2_geoschem.toml"), String)
            ok = write_toml("ok.toml", merra * "\n")
            @test P.load_met_settings(ok; root_dir = dir).face_fluxes === :panel_average
            # vector ignores face_lengths, so an explicit centreline choice is rejected
            bad = replace(merra, "[preprocessing]" => "[preprocessing]\nface_fluxes = \"vector\"\nface_lengths = \"cell_centerline\"")
            @test_throws ArgumentError P.load_met_settings(write_toml("bad.toml", bad); root_dir = dir)
            wrong = replace(merra, "[preprocessing]" => "[preprocessing]\nface_fluxes = \"spline\"")
            @test_throws ArgumentError P.load_met_settings(write_toml("wrong.toml", wrong); root_dir = dir)
            # ERA5 N320 takes the shared flux-construction keys, but not the MERRA-2 flux thickness
            era = write_toml("era.toml", "[source]\nname = \"ERA5-N320\"\n[preprocessing]\nface_fluxes = \"vector\"\n" *
                                         "face_interpolation = \"fv3\"\nwind_regrid = \"cartesian\"\n" *
                                         "column_balance_weights = \"hybrid_b\"\n")
            e = P.load_met_settings(era; root_dir = dir)
            @test (e.face_fluxes, e.face_interpolation, e.wind_regrid, e.column_balance_weights) ==
                  (:vector, :fv3, :cartesian, :hybrid_b)
            @test P.load_met_settings(write_toml("era0.toml", "[source]\nname = \"ERA5-N320\"\n"); root_dir = dir).face_fluxes ===
                  :panel_average
            era_dry = write_toml("era_dry.toml", "[source]\nname = \"ERA5-N320\"\n[preprocessing]\nflux_thickness = \"dry_mass\"\n")
            @test_throws ArgumentError P.load_met_settings(era_dry; root_dir = dir)
            era_bad = write_toml("era_bad.toml", "[source]\nname = \"ERA5-N320\"\n[preprocessing]\nwind_regrid = \"bilinear\"\n")
            @test_throws ArgumentError P.load_met_settings(era_bad; root_dir = dir)
            vec = replace(merra, "[preprocessing]" => "[preprocessing]\nface_fluxes = \"vector\"\nface_interpolation = \"fv3\"\nwind_regrid = \"cartesian\"")
            s = P.load_met_settings(write_toml("vec.toml", vec); root_dir = dir)
            @test (s.face_interpolation, s.wind_regrid) == (:fv3, :cartesian)
            # the cubic stencil belongs to the vector method
            cubic = replace(merra, "[preprocessing]" => "[preprocessing]\nface_interpolation = \"cubic\"")
            @test_throws ArgumentError P.load_met_settings(write_toml("cubic.toml", cubic); root_dir = dir)
            bad_regrid = replace(merra, "[preprocessing]" => "[preprocessing]\nwind_regrid = \"bilinear\"")
            @test_throws ArgumentError P.load_met_settings(write_toml("regrid.toml", bad_regrid); root_dir = dir)
        end
    end
end

