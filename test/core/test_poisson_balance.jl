#!/usr/bin/env julia
#
# Unit tests for the Poisson mass-flux balance solvers.
# Tests synthetic cases where the exact solution is known.

using Test
using Random
Random.seed!(20261030)

using AtmosTransport
Prep = AtmosTransport.Preprocessing

# ---------------------------------------------------------------------------
# LL FFT Poisson balance
# ---------------------------------------------------------------------------

@testset "LL FFT Poisson balance" begin

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


    @testset "Column balance spreads by the column weights" begin
        Nx, Ny, Nz = 16, 9, 5
        B = [0.0, 0.0, 0.3, 0.6, 0.85, 1.0]          # layer 1 pure pressure
        m = [1e10 * k * (1 + 0.1sin(i + 2j)) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        am0 = [1e7 * cos(0.3i + 0.7j + k) for i in 1:Nx+1, j in 1:Ny, k in 1:Nz]
        bm0 = [1e7 * sin(0.5i - 0.2j + k) for i in 1:Nx, j in 1:Ny+1, k in 1:Nz]
        bm0[:, 1, :] .= 0; bm0[:, Ny+1, :] .= 0
        dm_dt = zeros(Nx, Ny, Nz)
        ws = Prep.LLPoissonWorkspace(Nx, Ny)
        function balanced(weights)
            am, bm = copy(am0), copy(bm0)
            diag = Prep.balance_column_mass_fluxes!(am, bm, m, dm_dt, ws; weights)
            return am, bm, diag
        end
        am_m, bm_m, diag_m = balanced(Prep.MassWeightedColumn())
        am_h, bm_h, diag_h = balanced(Prep.column_weights(:hybrid_mass, B))
        @test diag_h.max_post_residual <= 10 * max(diag_m.max_post_residual, 1e-6)
        @test am_h[:, :, 1] == am0[:, :, 1] && bm_h[:, :, 1] == bm0[:, :, 1]
        @test am_m[:, :, 1] != am0[:, :, 1]
        @test sum(am_h; dims = 3) ≈ sum(am_m; dims = 3) rtol = 1e-10
        @test_throws DimensionMismatch balanced(Prep.column_weights(:hybrid_mass, B[2:end]))
    end
end

# ---------------------------------------------------------------------------
# CS global Poisson balance
# ---------------------------------------------------------------------------

@testset "CS global Poisson balance" begin
    using .AtmosTransport.Grids: GnomonicPanelConvention,
        GEOSNativePanelConvention, panel_connectivity_for

    cs_conventions = (GnomonicPanelConvention(), GEOSNativePanelConvention())


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

                # After balance (which calls sync_cs_mirrors!), compare divergences.
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

    @testset "CS Poisson PCG projection cadence" begin
        Nc = 12
        mesh = CubedSphereMesh(; Nc=Nc, convention=GEOSNativePanelConvention())
        ft = Prep.build_cs_global_face_table(Nc, mesh.connectivity)
        degree = Prep.cs_cell_face_degree(ft)

        rng = MersenneTwister(20260429)
        rhs0 = randn(rng, ft.nc)
        rhs0 .-= sum(rhs0) / length(rhs0)

        function solve_with(project_every)
            scratch = Prep.CSPoissonScratch(ft.nc)
            psi = zeros(Float64, ft.nc)
            rhs = copy(rhs0)
            res, it = Prep.solve_cs_poisson_pcg!(
                psi, rhs, ft, degree, (r=scratch.r, p=scratch.p, Ap=scratch.Ap, z=scratch.z);
                tol=1e-11, max_iter=2000, project_every=project_every)
            Prep.cs_graph_laplacian_mul!(scratch.Ap, psi, ft, degree)
            return (; psi, residual=maximum(abs.(scratch.Ap .- rhs0)), solver_res=res, it)
        end

        legacy = solve_with(1)
        periodic = solve_with(8)
        default = solve_with(AtmosTransport.Parameters.DEFAULT_POISSON_PROJECT_EVERY)

        @test default.residual < 1e-9
        @test default.psi ≈ legacy.psi rtol=1e-8 atol=1e-8
        @test legacy.residual < 1e-9
        @test periodic.residual < 1e-9
        @test isapprox(periodic.psi, legacy.psi; rtol=1e-8, atol=1e-8)
        @test abs(sum(periodic.psi) / length(periodic.psi)) < 1e-12
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
    Prep.sync_cs_mirrors!(am0, bm0, ft, Nz)          # consistent shared faces

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

    # A seam uses the two physical cells from the global face table, not two
    # copies of whichever boundary cell happens to own the canonical face.
    f = findfirst(!=(0), ft.mirror_panel)
    fp, fd = Int(ft.face_panel[f]), Int(ft.face_dir[f])
    fi, fj = Int(ft.face_idx_i[f]), Int(ft.face_idx_j[f])
    left, right = Int(ft.face_left[f]), Int(ft.face_right[f])
    cell_view(c) = begin
        panel = (c - 1) ÷ Nc^2 + 1
        local_idx = (c - 1) % Nc^2
        i = local_idx % Nc + 1
        j = local_idx ÷ Nc + 1
        view(m[panel], i, j, :)
    end
    seam_inc = fd == 1 ? am_mass[fp][fi, fj, :] .- am0[fp][fi, fj, :] :
                         bm_mass[fp][fi, fj, :] .- bm0[fp][fi, fj, :]
    seam_w = cell_view(left) .+ cell_view(right)
    @test seam_inc[1:end-1] ./ seam_w[1:end-1] ≈
          fill(seam_inc[1] / seam_w[1], Nz - 1) rtol = 1e-10

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

    # Residual folding is explicit and bounded: the raw value is returned,
    # while an out-of-tolerance value fails before it can be hidden in cm.
    cm_probe = ntuple(_ -> zeros(Nc, Nc, Nz + 1), 6)
    dm_probe = ntuple(_ -> zeros(Nc, Nc, Nz), 6)
    dm_probe[1][1, 1, 1] = 0.25
    cm_diag = Prep.diagnose_cs_cm!(cm_probe, ntuple(_ -> zeros(Nc + 1, Nc, Nz), 6),
                                   ntuple(_ -> zeros(Nc, Nc + 1, Nz), 6),
                                   dm_probe, m, Nc, Nz; residual_tol = 0.25)
    @test cm_diag.max_raw_residual == 0.25
    @test cm_diag.max_booked_residual == 0.25
    @test cm_diag.max_raw_residual_ratio <= 1
    # The folded residual closes the column to rounding of the 0.25 kg residual
    # (Julia 1.10 and 1.12 sum in different orders: 0 vs one ulp).
    @test abs(cm_probe[1][1, 1, end]) <= 4eps(0.25)
    @test_throws ErrorException Prep.diagnose_cs_cm!(cm_probe,
        ntuple(_ -> zeros(Nc + 1, Nc, Nz), 6),
        ntuple(_ -> zeros(Nc, Nc + 1, Nz), 6), dm_probe, m, Nc, Nz;
        residual_tol = 0.24)
    # A per-cell tolerance applies to its own column only.
    per_cell = fill(0.24, 6Nc^2); per_cell[1] = 0.25
    @test Prep.diagnose_cs_cm!(cm_probe, ntuple(_ -> zeros(Nc + 1, Nc, Nz), 6),
        ntuple(_ -> zeros(Nc, Nc + 1, Nz), 6), dm_probe, m, Nc, Nz;
        residual_tol = per_cell).max_raw_residual_ratio <= 1
    per_cell[1] = 0.24
    @test_throws ErrorException Prep.diagnose_cs_cm!(cm_probe,
        ntuple(_ -> zeros(Nc + 1, Nc, Nz), 6), ntuple(_ -> zeros(Nc, Nc + 1, Nz), 6),
        dm_probe, m, Nc, Nz; residual_tol = per_cell)
    @test_throws DimensionMismatch Prep.diagnose_cs_cm!(cm_probe,
        ntuple(_ -> zeros(Nc + 1, Nc, Nz), 6), ntuple(_ -> zeros(Nc, Nc + 1, Nz), 6),
        dm_probe, m, Nc, Nz; residual_tol = per_cell[2:end])
    # A residual common to every column (a global air-mass change, as left by
    # an unpinned run) must fail the closure gate.
    dm_global = ntuple(_ -> fill(1.0, Nc, Nc, Nz), 6)
    @test_throws ErrorException Prep.diagnose_cs_cm!(ntuple(_ -> zeros(Nc, Nc, Nz + 1), 6),
        ntuple(_ -> zeros(Nc + 1, Nc, Nz), 6), ntuple(_ -> zeros(Nc, Nc + 1, Nz), 6),
        dm_global, m, Nc, Nz; residual_tol = 1.0, context = "unpinned test")

    # Float32: the closure tolerance is per cell (a global maximum would let a
    # small-flux cell keep a residual set by the largest-flux cell) and the
    # result is exposed for a mandatory caller-side check.
    m32 = map(x -> Float32.(1e11 .* x), m)
    # A tendency with zero global sum (a pinned run): panel pairs exchange mass.
    exchange = Float32.(1e9 .* (m_next[1] .- m[1]))
    m32_next = ntuple(p -> m32[p] .+ (isodd(p) ? exchange : -exchange), 6)
    am32 = map(x -> Float32.(1e10 .* x), am0)
    bm32 = map(x -> Float32.(1e10 .* x), bm0)
    am32[1][:, :, :] .*= 1000                  # one panel with 1000× larger fluxes
    bm32[1][:, :, :] .*= 1000
    Prep.sync_cs_mirrors!(am32, bm32, ft, Nz)
    scratch32 = Prep.CSPoissonScratch(ft.nc)
    diag32 = Prep.balance_cs_column_mass_fluxes!(
        am32, bm32, m32, m32_next, ft, degree, steps, scratch32; closure_passes = 8)
    @test diag32.final_column_residual_ratio <= 1
    @test diag32.final_column_tolerance_by_cell === scratch32.tol
    @test diag32.final_column_tolerance_min >= 10.0
    @test diag32.final_column_tolerance_max > 5 * diag32.final_column_tolerance_min
    @test minimum(scratch32.tol) == diag32.final_column_tolerance_min
    @test Prep.require_cs_column_balance(diag32, "Float32 test") === diag32
    bad_diag = merge(diag32, (final_column_residual_ratio = 1.5,))
    @test_throws ErrorException Prep.require_cs_column_balance(bad_diag, "bad test")
    @test_throws ErrorException Prep.require_cs_column_balance(
        merge(diag32, (final_column_residual_ratio = NaN,)), "nan test")
    dm32 = ntuple(p -> (m32_next[p] .- m32[p]) ./ Float32(2steps), 6)
    cm32 = ntuple(_ -> zeros(Float32, Nc, Nc, Nz + 1), 6)
    cm_diag32 = Prep.diagnose_balanced_cs_cm!(cm32, am32, bm32, dm32, m32, Nc, Nz,
                                               diag32, "Float32 test")
    @test cm_diag32.max_raw_residual_ratio <= 1
    record = Prep.column_closure_record(diag32, cm_diag32)
    @test record.residual_to_tolerance == diag32.final_column_residual_ratio
    @test record.booked_residual_kg == cm_diag32.max_booked_residual
    # Fluxes changed after the balance are re-checked against their own tolerances.
    recheck = Prep.recheck_cs_column_closure(am32, bm32, m32, m32_next, ft, steps,
                                             scratch32, diag32)
    @test recheck.final_column_residual_ratio <= 1
    # Full-window payloads (GEOS) are re-checked as stored, in per-application
    # units: an exact (power-of-two) scaling reproduces the per-application
    # closure, kg values are stated in stored amounts.
    full = 2steps
    @test ispow2(full)
    stored_am = map(x -> x .* Float32(full), am32)
    stored_bm = map(x -> x .* Float32(full), bm32)
    stored = Prep.full_window_column_closure_record(record, stored_am, stored_bm, m32,
                                                    m32_next, ft, steps, scratch32)
    @test stored.residual_to_tolerance == recheck.final_column_residual_ratio
    @test stored.projected_residual_kg == full * recheck.final_column_projected_residual
    @test stored.booked_residual_kg == full * record.booked_residual_kg
    @test stored.cm_residual_to_tolerance == record.cm_residual_to_tolerance
    @test Prep.full_window_column_closure_record(Prep.NoColumnClosureRecord(), stored_am,
              stored_bm, m32, m32_next, ft, steps, scratch32) === Prep.NoColumnClosureRecord()
    # A factor that is not a power of two rounds every stored value; the check
    # sees the stored payload, not the fluxes before the scaling.
    rounded = Prep.recheck_cs_column_closure(map(x -> x .* 18f0, am32),
                                             map(x -> x .* 18f0, bm32), m32, m32_next,
                                             ft, steps, scratch32, (;); flux_scale = 1 / 18)
    @test rounded.final_column_residual_ratio <= 1
    @test rounded.final_column_projected_residual != recheck.final_column_projected_residual
    @test_throws ArgumentError Prep.recheck_cs_column_closure(am32, bm32, m32, m32_next, ft,
                                                              steps, scratch32, (;);
                                                              flux_scale = 0.0)
    moved = map(copy, am32); moved[2][3, 3, 1] += 1f9; moved[2][3, 3, 2] -= 1f9  # same column sum
    @test Prep.recheck_cs_column_closure(moved, bm32, m32, m32_next, ft, steps,
                                         scratch32, diag32).final_column_residual_ratio <= 1
    broken = map(copy, am32); broken[2][3, 3, 1] += 1f9                        # column budget broken
    @test Prep.recheck_cs_column_closure(broken, bm32, m32, m32_next, ft, steps,
                                         scratch32, diag32).final_column_residual_ratio > 1
    # An exactly closed column with zero tolerance (closure_tol = 0, no flux) is closed.
    zf = ntuple(_ -> zeros(Nc + 1, Nc, Nz), 6), ntuple(_ -> zeros(Nc, Nc + 1, Nz), 6)
    zdiag = Prep.balance_cs_column_mass_fluxes!(zf..., m, m, ft, degree, steps,
                                                Prep.CSPoissonScratch(ft.nc); closure_tol = 0.0)
    @test zdiag.final_column_residual_ratio == 0

    # ΔB weights are layer thicknesses in B, normalized, for any vector type
    for Bv in (B, Float32.(B), collect(B))
        w = Prep.column_weights(:hybrid_b, Bv)
        @test length(w.dB) == Nz && isapprox(w.dB, diff(B) ./ (B[end] - B[1]); rtol = 1e-6)
    end
    @test Prep.column_weights(:hybrid_mass, B).hybrid ==
          (diff(B) .>= AtmosTransport.Parameters.HYBRID_LAYER_MIN_DB)

    # Near-pure-pressure layers: a positive ΔB below HYBRID_LAYER_MIN_DB (a
    # rounding residue such as GEOS L72 layer 41, ΔB = 8.2e-9) is treated as
    # pure pressure, by the horizontal spread and by the cm residual booking.
    B_tiny = [0.0, 0.0, 8.175413e-9, 0.2, 0.5, 0.8, 1.0]
    w_tiny = Prep.column_weights(:hybrid_mass, B_tiny)
    @test w_tiny.hybrid == BitVector([false, false, true, true, true, true])
    am_t, bm_t, _ = balanced(; weights = w_tiny)
    @test all(p -> am_t[p][:, :, 1:2] == am0[p][:, :, 1:2], 1:6)
    @test all(p -> bm_t[p][:, :, 1:2] == bm0[p][:, :, 1:2], 1:6)
    cm_tiny = ntuple(_ -> zeros(Nc, Nc, Nz + 1), 6)
    Prep.diagnose_cs_cm!(cm_tiny, am0, bm0, dm, m, Nc, Nz, w_tiny)
    @test all(p -> cm_tiny[p][:, :, 1:3] == raw[p][:, :, 1:3], 1:6)
    @test_throws ArgumentError Prep.HybridMassWeightedColumn([0.0, 1e-9, 2e-9])
    for (file, layer, hybrid) in (("geos_L72_coefficients.toml", 41, false),
                                  ("geos_L72_coefficients.toml", 42, true),
                                  ("era5_L137_coefficients.toml", 54, false),
                                  ("era5_L137_coefficients.toml", 55, true))
        vc = Prep.load_hybrid_coefficients(joinpath(pkgdir(AtmosTransport), "config", file))
        @test Prep.HybridMassWeightedColumn(vc.B).hybrid[layer] == hybrid
    end

    # weights for a different number of layers are rejected
    @test_throws DimensionMismatch balanced(; weights = Prep.column_weights(:hybrid_b, B[2:end]))
    @test_throws DimensionMismatch Prep.diagnose_cs_cm!(cm_hyb, am0, bm0, dm, m, Nc, Nz,
                                                        Prep.column_weights(:hybrid_mass, B[2:end]))

    @test_throws ArgumentError Prep.column_weights(:pressure, B)
    @test_throws ArgumentError Prep.HybridBWeightedColumn(reverse(B))
    @test_throws ArgumentError Prep.HybridMassWeightedColumn(reverse(B))
end
