#!/usr/bin/env julia
#
# Unit tests for the Poisson mass-flux balance solvers.
# Tests synthetic cases where the exact solution is known.

using Test

using AtmosTransport
Prep = AtmosTransport.Preprocessing

# ---------------------------------------------------------------------------
# LL FFT Poisson balance
# ---------------------------------------------------------------------------

@testset "LL FFT Poisson balance" begin
    @testset "Already-balanced fluxes → no change" begin
        Nx, Ny, Nz = 24, 12, 2
        am = zeros(Float64, Nx+1, Ny, Nz)
        bm = zeros(Float64, Nx, Ny+1, Nz)
        dm_dt = zeros(Float64, Nx, Ny, Nz)

        ws = Prep.LLPoissonWorkspace(Nx, Ny)
        am0 = copy(am); bm0 = copy(bm)
        Prep.balance_mass_fluxes!(am, bm, dm_dt, ws)

        @test am ≈ am0 atol=1e-14
        @test bm ≈ bm0 atol=1e-14
    end

    @testset "Uniform divergence → corrected to match dm_dt" begin
        Nx, Ny, Nz = 36, 18, 4
        # Create fluxes with known divergence
        am = zeros(Float64, Nx+1, Ny, Nz)
        bm = zeros(Float64, Nx, Ny+1, Nz)
        dm_dt = zeros(Float64, Nx, Ny, Nz)

        # Put some divergence in the fluxes
        for k in 1:Nz, j in 1:Ny, i in 1:Nx+1
            am[i, j, k] = 0.01 * sin(2π * i / Nx) * cos(2π * j / Ny)
        end
        for k in 1:Nz, j in 1:Ny+1, i in 1:Nx
            bm[i, j, k] = 0.01 * cos(2π * i / Nx) * sin(2π * j / Ny)
        end

        # Target: zero mass tendency
        fill!(dm_dt, 0.0)

        ws = Prep.LLPoissonWorkspace(Nx, Ny)
        Prep.balance_mass_fluxes!(am, bm, dm_dt, ws)

        # After balance, convergence should match dm_dt at each cell
        max_residual = 0.0
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            conv = (am[i,j,k] - am[i+1,j,k]) + (bm[i,j,k] - bm[i,j+1,k])
            residual = abs(conv - dm_dt[i,j,k])
            max_residual = max(max_residual, residual)
        end

        # Should be at machine precision
        @test max_residual < 1e-12
    end

    @testset "Non-zero dm_dt → convergence matches target" begin
        Nx, Ny, Nz = 48, 24, 2
        am = zeros(Float64, Nx+1, Ny, Nz)
        bm = zeros(Float64, Nx, Ny+1, Nz)
        dm_dt = zeros(Float64, Nx, Ny, Nz)

        # Non-zero mass tendency pattern
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            dm_dt[i,j,k] = 1e-3 * sin(4π * i / Nx) * sin(2π * j / Ny)
        end
        # Subtract global mean (Poisson can't fix the mean)
        for k in 1:Nz
            dm_dt[:,:,k] .-= sum(dm_dt[:,:,k]) / (Nx * Ny)
        end

        ws = Prep.LLPoissonWorkspace(Nx, Ny)
        Prep.balance_mass_fluxes!(am, bm, dm_dt, ws)

        max_residual = 0.0
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            conv = (am[i,j,k] - am[i+1,j,k]) + (bm[i,j,k] - bm[i,j+1,k])
            residual = abs(conv - dm_dt[i,j,k])
            max_residual = max(max_residual, residual)
        end

        @test max_residual < 1e-12
    end

    @testset "Workspace reuse across levels" begin
        Nx, Ny, Nz = 24, 12, 8
        am = rand(Float64, Nx+1, Ny, Nz) .* 0.01
        bm = rand(Float64, Nx, Ny+1, Nz) .* 0.01
        dm_dt = zeros(Float64, Nx, Ny, Nz)

        ws = Prep.LLPoissonWorkspace(Nx, Ny)

        # Run twice — workspace should produce same result both times
        am1 = copy(am); bm1 = copy(bm)
        Prep.balance_mass_fluxes!(am1, bm1, dm_dt, ws)

        am2 = copy(am); bm2 = copy(bm)
        Prep.balance_mass_fluxes!(am2, bm2, dm_dt, ws)

        @test am1 ≈ am2 atol=1e-15
        @test bm1 ≈ bm2 atol=1e-15
    end
end

# ---------------------------------------------------------------------------
# CS global Poisson balance
# ---------------------------------------------------------------------------

@testset "CS global Poisson balance" begin
    if isdefined(Prep, :balance_cs_global_mass_fluxes!)
        using .AtmosTransport.Grids: GnomonicPanelConvention,
            GEOSNativePanelConvention, panel_connectivity_for

        cs_conventions = (GnomonicPanelConvention(), GEOSNativePanelConvention())

        @testset "Zero fluxes → zero correction" begin
            for convention in cs_conventions
                @testset "$(nameof(typeof(convention)))" begin
                    Nc = 8; Nz = 2
                    conn = panel_connectivity_for(convention)
                    ft = Prep.build_cs_global_face_table(Nc, conn)
                    degree = Prep.cs_cell_face_degree(ft)
                    scratch = Prep.CSPoissonScratch(ft.nc)

                    m = ntuple(_ -> ones(Float64, Nc, Nc, Nz), 6)
                    m_next = ntuple(_ -> ones(Float64, Nc, Nc, Nz), 6)
                    am = ntuple(_ -> zeros(Float64, Nc+1, Nc, Nz), 6)
                    bm = ntuple(_ -> zeros(Float64, Nc, Nc+1, Nz), 6)

                    Prep.balance_cs_global_mass_fluxes!(am, bm, m, m_next, ft, degree, 4, scratch)

                    for p in 1:6
                        @test maximum(abs, am[p]) < 1e-14
                        @test maximum(abs, bm[p]) < 1e-14
                    end
                end
            end
        end

        @testset "Random fluxes → post-balance residual near machine precision" begin
            for convention in cs_conventions
                @testset "$(nameof(typeof(convention)))" begin
                    Nc = 12; Nz = 2; steps = 4
                    conn = panel_connectivity_for(convention)
                    ft = Prep.build_cs_global_face_table(Nc, conn)
                    degree = Prep.cs_cell_face_degree(ft)
                    scratch = Prep.CSPoissonScratch(ft.nc)

                    # Same mass at both endpoints (zero target) with random initial fluxes
                    m = ntuple(_ -> ones(Float64, Nc, Nc, Nz), 6)
                    am = ntuple(_ -> rand(Float64, Nc+1, Nc, Nz) .* 0.01, 6)
                    bm = ntuple(_ -> rand(Float64, Nc, Nc+1, Nz) .* 0.01, 6)

                    diag = Prep.balance_cs_global_mass_fluxes!(am, bm, m, m,
                                                                ft, degree, steps, scratch;
                                                                tol=1e-14, max_iter=10000)

                    # Post-balance: convergence matches zero target to machine precision
                    @test diag.max_post_residual < 1e-10
                end
            end
        end

        @testset "All cells have degree 4" begin
            for convention in cs_conventions, Nc in [4, 8, 16]
                @testset "$(nameof(typeof(convention))) C$(Nc)" begin
                    conn = panel_connectivity_for(convention)
                    ft = Prep.build_cs_global_face_table(Nc, conn)
                    degree = Prep.cs_cell_face_degree(ft)
                    @test all(degree .== 4)
                    @test length(degree) == 6 * Nc^2
                end
            end
        end

        @testset "cm diagnosis gives zero bottom boundary" begin
            Nc = 8; Nz = 4
            am = ntuple(_ -> zeros(Float64, Nc+1, Nc, Nz), 6)
            bm = ntuple(_ -> zeros(Float64, Nc, Nc+1, Nz), 6)
            dm = ntuple(_ -> zeros(Float64, Nc, Nc, Nz), 6)
            m  = ntuple(_ -> ones(Float64, Nc, Nc, Nz), 6)
            cm = ntuple(_ -> zeros(Float64, Nc, Nc, Nz+1), 6)

            Prep.diagnose_cs_cm!(cm, am, bm, dm, m, Nc, Nz)

            # With zero fluxes and zero dm, all cm should be zero
            for p in 1:6
                @test maximum(abs, cm[p]) < 1e-14
            end
        end
        @testset "mirror_sign: inflow positions get +1, outflow get -1" begin
            for convention in cs_conventions, Nc in [4, 8, 12]
                @testset "$(nameof(typeof(convention))) C$(Nc)" begin
                    conn = panel_connectivity_for(convention)
                    ft = Prep.build_cs_global_face_table(Nc, conn)

                    for f in 1:ft.nf
                        mq = Int(ft.mirror_panel[f])
                        mq == 0 && continue  # interior face

                        cdir = Int(ft.face_dir[f])
                        ci   = Int(ft.face_idx_i[f])
                        cj   = Int(ft.face_idx_j[f])
                        mdir = Int(ft.mirror_dir[f])
                        mi   = Int(ft.mirror_idx_i[f])
                        mj   = Int(ft.mirror_idx_j[f])
                        ms   = Int(ft.mirror_sign[f])

                        # `mirror_sign` negates when canonical + mirror land on the
                        # same position type (both inflow or both outflow), and
                        # stays positive when they land on opposite types.
                        can_at_outflow = (cdir == 1 && ci == Nc + 1) ||
                                         (cdir == 2 && cj == Nc + 1)
                        at_outflow = (mdir == 1 && mi == Nc + 1) ||
                                     (mdir == 2 && mj == Nc + 1)
                        expected = (can_at_outflow == at_outflow) ? -1 : 1
                        @test ms == expected
                    end
                end
            end
        end

        @testset "mirror_sign: per-panel div matches global face-table div" begin
            for convention in cs_conventions
                @testset "$(nameof(typeof(convention)))" begin
                    Nc = 8; Nz = 3; steps = 4
                    conn = panel_connectivity_for(convention)
                    ft = Prep.build_cs_global_face_table(Nc, conn)
                    degree = Prep.cs_cell_face_degree(ft)
                    scratch = Prep.CSPoissonScratch(ft.nc)
                    nc = 6 * Nc^2

                    # Non-trivial mass fields to create a real balance problem
                    m      = ntuple(p -> 1.0 .+ 0.1 .* rand(Float64, Nc, Nc, Nz), 6)
                    m_next = ntuple(p -> 1.0 .+ 0.1 .* rand(Float64, Nc, Nc, Nz), 6)
                    am = ntuple(_ -> rand(Float64, Nc+1, Nc, Nz) .* 0.01, 6)
                    bm = ntuple(_ -> rand(Float64, Nc, Nc+1, Nz) .* 0.01, 6)

                    Prep.balance_cs_global_mass_fluxes!(am, bm, m, m_next,
                        ft, degree, steps, scratch; tol=1e-14, max_iter=20000)

                    # After balance (which calls _sync_cs_mirrors!), compare divergences.
                    for k in 1:Nz
                        # Global face-table divergence
                        div_global = zeros(Float64, nc)
                        @inbounds for f in 1:ft.nf
                            p   = Int(ft.face_panel[f])
                            dir = Int(ft.face_dir[f])
                            i   = Int(ft.face_idx_i[f])
                            j   = Int(ft.face_idx_j[f])
                            flux = dir == 1 ? am[p][i, j, k] : bm[p][i, j, k]

                            left  = Int(ft.face_left[f])
                            right = Int(ft.face_right[f])
                            div_global[left]  += flux
                            div_global[right] -= flux
                        end

                        # Per-panel divergence (outflow convention) from am/bm arrays.
                        # div = (am[i+1] - am[i]) + (bm[j+1] - bm[j]) = net outflow.
                        # Boundary faces use mirror entries — this is what mirror_sign
                        # must get right.
                        div_panel = zeros(Float64, nc)
                        @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
                            c = (p - 1) * Nc^2 + (j - 1) * Nc + i
                            div_panel[c] = (am[p][i+1, j, k] - am[p][i, j, k]) +
                                           (bm[p][i, j+1, k] - bm[p][i, j, k])
                        end

                        # They must match at every cell — this fails without mirror_sign
                        max_diff = maximum(abs, div_global .- div_panel)
                        @test max_diff < 1e-13
                    end
                end
            end
        end

        @testset "mirror_sign: balance + cm diagnosis conserves mass per column" begin
            for convention in cs_conventions
                @testset "$(nameof(typeof(convention)))" begin
                    Nc = 8; Nz = 4; steps = 4
                    conn = panel_connectivity_for(convention)
                    ft = Prep.build_cs_global_face_table(Nc, conn)
                    degree = Prep.cs_cell_face_degree(ft)
                    scratch = Prep.CSPoissonScratch(ft.nc)

                    m      = ntuple(p -> 1.0 .+ 0.1 .* rand(Float64, Nc, Nc, Nz), 6)
                    m_next = ntuple(p -> 1.0 .+ 0.1 .* rand(Float64, Nc, Nc, Nz), 6)

                    # Make per-level global mass match (Poisson requires Σ target = 0 per level)
                    for k in 1:Nz
                        total_cur  = sum(p -> sum(m[p][:,:,k]), 1:6)
                        total_next = sum(p -> sum(m_next[p][:,:,k]), 1:6)
                        offset = (total_cur - total_next) / (6 * Nc^2)
                        for p in 1:6; m_next[p][:,:,k] .+= offset; end
                    end

                    am = ntuple(_ -> rand(Float64, Nc+1, Nc, Nz) .* 0.01, 6)
                    bm = ntuple(_ -> rand(Float64, Nc, Nc+1, Nz) .* 0.01, 6)

                    Prep.balance_cs_global_mass_fluxes!(am, bm, m, m_next,
                        ft, degree, steps, scratch; tol=1e-14, max_iter=20000)

                    # Compute dm and diagnose cm
                    dm = ntuple(_ -> zeros(Float64, Nc, Nc, Nz), 6)
                    cm = ntuple(_ -> zeros(Float64, Nc, Nc, Nz+1), 6)
                    inv_scale = 1.0 / (2 * steps)
                    for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
                        dm[p][i, j, k] = (m_next[p][i, j, k] - m[p][i, j, k]) * inv_scale
                    end
                    Prep.diagnose_cs_cm!(cm, am, bm, dm, m, Nc, Nz)

                    # cm[k=1] = 0 by construction, cm[k=Nz+1] should be near zero
                    # after residual redistribution
                    max_bottom = maximum(p -> maximum(abs, cm[p][:,:,Nz+1]), 1:6)
                    @test max_bottom < 1e-12
                end
            end
        end

    else
        @test true  # skip if CS balance not available
    end
end

# ---------------------------------------------------------------------------
# Column-balance weights: how the column correction is spread over levels
# ---------------------------------------------------------------------------

@testset "CS column-balance weights" begin
    using Random
    using .AtmosTransport.Grids: GnomonicPanelConvention, panel_connectivity_for
    Nc, Nz, steps = 8, 6, 4
    B = [0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0]   # layers 1–2 pure pressure (ΔB = 0)
    pure, hybrid = 1:2, 3:Nz
    ft = Prep.build_cs_global_face_table(Nc, panel_connectivity_for(GnomonicPanelConvention()))
    degree = Prep.cs_cell_face_degree(ft)
    rng = MersenneTwister(2)
    m      = ntuple(_ -> 1 .+ 0.1 .* rand(rng, Nc, Nc, Nz), 6)
    m_next = ntuple(_ -> 1 .+ 0.1 .* rand(rng, Nc, Nc, Nz), 6)
    am0 = ntuple(_ -> 0.01 .* rand(rng, Nc + 1, Nc, Nz), 6)
    bm0 = ntuple(_ -> 0.01 .* rand(rng, Nc, Nc + 1, Nz), 6)
    Prep._sync_cs_mirrors!(am0, bm0, ft, Nz)          # consistent shared faces

    function balanced(; kwargs...)
        am, bm = map(copy, am0), map(copy, bm0)
        diag = Prep.balance_cs_column_mass_fluxes!(am, bm, m, m_next, ft, degree, steps,
                                                   Prep.CSPoissonScratch(ft.nc); kwargs...)
        return am, bm, diag
    end
    am_def, bm_def, _ = balanced()
    am_mass, bm_mass, diag_mass = balanced(; weights = Prep.column_weights(:mass, B))
    @test am_def == am_mass && bm_def == bm_mass          # default is mass weighting

    for kind in (:hybrid_b, :hybrid_mass)
        am, bm, diag = balanced(; weights = Prep.column_weights(kind, B))
        # the column budget closes as well as with mass weights
        @test diag.final_column_projected_residual <= 10 * max(diag_mass.final_column_projected_residual, 1e-12)
        # pure-pressure layers receive no correction
        @test all(p -> am[p][:, :, pure] == am0[p][:, :, pure], 1:6)
        @test all(p -> bm[p][:, :, pure] == bm0[p][:, :, pure], 1:6)
        @test any(p -> am[p][:, :, hybrid] != am0[p][:, :, hybrid], 1:6)
    end

    # hybrid_b spreads the correction in proportion to ΔB
    am_hb, _, _ = balanced(; weights = Prep.column_weights(:hybrid_b, B))
    inc_b = am_hb[1][4, 3, hybrid[1:end-1]] .- am0[1][4, 3, hybrid[1:end-1]]
    dB = diff(B)[hybrid[1:end-1]]
    @test inc_b ./ dB ≈ fill(inc_b[1] / dB[1], length(dB)) rtol = 1e-10

    # hybrid_mass spreads the correction like mass weights within hybrid layers
    am_hm, _, _ = balanced(; weights = Prep.column_weights(:hybrid_mass, B))
    p, i, j = 1, 4, 3
    inc = am_hm[p][i, j, hybrid[1:end-1]] .- am0[p][i, j, hybrid[1:end-1]]
    w = m[p][i - 1, j, hybrid[1:end-1]] .+ m[p][i, j, hybrid[1:end-1]]
    @test inc ./ w ≈ fill(inc[1] / w[1], length(w)) rtol = 1e-10

    # cm diagnosis: the bottom residual is spread over hybrid layers only
    dm = ntuple(p -> (m_next[p] .- m[p]) ./ (2steps), 6)
    cm_mass, cm_hyb = ntuple(_ -> ntuple(_ -> zeros(Nc, Nc, Nz + 1), 6), 2)
    Prep.diagnose_cs_cm!(cm_mass, am0, bm0, dm, m, Nc, Nz)
    Prep.diagnose_cs_cm!(cm_hyb, am0, bm0, dm, m, Nc, Nz, Prep.column_weights(:hybrid_mass, B))
    raw = ntuple(6) do p
        c = zeros(Nc, Nc, Nz + 1)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            div_h = (am0[p][i, j, k] - am0[p][i + 1, j, k]) + (bm0[p][i, j, k] - bm0[p][i, j + 1, k])
            c[i, j, k + 1] = c[i, j, k] + div_h - dm[p][i, j, k]
        end
        c
    end
    @test all(p -> cm_hyb[p][:, :, 1:3] == raw[p][:, :, 1:3], 1:6)    # top and pure-pressure interfaces
    @test any(p -> cm_mass[p][:, :, 2:3] != raw[p][:, :, 2:3], 1:6)   # mass weights reach them
    @test maximum(p -> maximum(abs, cm_hyb[p][:, :, Nz + 1]), 1:6) < 1e-12

    # ΔB weights are layer thicknesses in B, normalized, for any vector type
    for Bv in (B, Float32.(B), collect(B))
        w = Prep.column_weights(:hybrid_b, Bv)
        @test length(w.dB) == Nz && isapprox(w.dB, diff(B) ./ (B[end] - B[1]); rtol = 1e-6)
    end
    @test Prep.column_weights(:hybrid_mass, B).hybrid == (diff(B) .> 0)

    # weights for a different number of layers are rejected
    @test_throws DimensionMismatch balanced(; weights = Prep.column_weights(:hybrid_b, B[2:end]))
    @test_throws DimensionMismatch Prep.diagnose_cs_cm!(cm_hyb, am0, bm0, dm, m, Nc, Nz,
                                                        Prep.column_weights(:hybrid_mass, B[2:end]))

    # TOML wiring: MERRA-2 and ERA5 N320 read the key, the GEOS sources reject it
    mktempdir() do dir
        merra2 = Prep.load_met_settings(joinpath(pkgdir(AtmosTransport), "config", "met_sources",
                                                 "merra2_geoschem_hybridmass.toml"); root_dir = dir)
        @test merra2.column_balance_weights === :hybrid_mass
        toml = joinpath(dir, "era5.toml")
        write(toml, "[source]\nname = \"ERA5-N320\"\n[preprocessing]\ncolumn_balance_weights = \"hybrid_b\"\n")
        @test Prep.load_met_settings(toml; root_dir = dir).column_balance_weights === :hybrid_b
        geos = read(joinpath(pkgdir(AtmosTransport), "config", "met_sources", "geosit.toml"), String)
        geos_toml = joinpath(dir, "geos.toml")
        write(geos_toml, replace(geos, r"\[preprocessing\]" => "[preprocessing]\ncolumn_balance_weights = \"hybrid_b\"", count = 1))
        @test_throws ArgumentError Prep.load_met_settings(geos_toml; root_dir = dir)
    end

    @test_throws ArgumentError Prep.column_weights(:pressure, B)
    @test_throws ArgumentError Prep.HybridBWeightedColumn(reverse(B))
    @test_throws ArgumentError Prep.HybridMassWeightedColumn(reverse(B))
end
