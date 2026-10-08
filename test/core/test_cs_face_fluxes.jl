#!/usr/bin/env julia
# Cubed-sphere face-flux reconstruction from cell-centre winds: face lengths,
# the legacy centreline choice, and the divergence of a solid-body rotation.

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
            era = write_toml("era.toml", "[source]\nname = \"ERA5-N320\"\n[preprocessing]\nface_fluxes = \"vector\"\n")
            @test_throws ArgumentError P.load_met_settings(era; root_dir = dir)
        end
    end
end

