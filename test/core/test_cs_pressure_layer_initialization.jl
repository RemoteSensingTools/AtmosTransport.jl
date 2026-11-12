using AtmosTransport, Test
const PressureIC = AtmosTransport.Models.InitialConditionIO

@testset "Cubed-sphere pressure-layer initialization" begin
    for FT in (Float32, Float64)
        Nc, Hp, Nz = 3, 2, 4
        mesh = CubedSphereMesh(; Nc, Hp, FT)
        A = FT[0, 5000, 8000, 3000, 0]
        B = FT[0, 0.1, 0.3, 0.65, 1]
        grid = AtmosGrid(mesh, HybridSigmaPressure(A, B), CPU(); FT)
        ps = ntuple(p -> FT[65000 + 2500i + 1500j + 2000p for i in 1:Nc, j in 1:Nc], 6)
        air = ntuple(p -> begin
            a = fill(FT(NaN), Nc + 2Hp, Nc + 2Hp, Nz)
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                a[Hp+i, Hp+j, k] = FT(1000 + 30p + 20k + 2j + i)
            end
            a
        end, 6)
        for lowest in (false, true), fraction in (0.01, 0.2, 0.55, 1.0)
            cfg = Dict{String,Any}("kind" => "pressure_layer", "lowest_layer" => lowest,
                                   "psurf_fraction" => fraction, "total_molecules" => 1e28)
            q = PressureIC.build_initial_mixing_ratio(air, grid, cfg; surface_pressure=ps)
            @test all(size(panel) == (Nc, Nc, Nz) && eltype(panel) == FT for panel in q)
            # Independently choose the nearest log-pressure midpoint. Halos contain
            # NaNs so a misplaced dry-mass read cannot silently pass normalization.
            selected_mass = 0.0
            for p in 1:6, j in 1:Nc, i in 1:Nc
                edges = Float64.(A) .+ Float64.(B) .* Float64(ps[p][i,j])
                mids = sqrt.(max.(edges[1:end-1], eps(Float64)) .* edges[2:end])
                k = lowest ? Nz : argmin(abs.(mids .- FT(fraction) * Float64(ps[p][i,j])))
                @test findall(!iszero, q[p][i,j,:]) == [k]
                selected_mass += Float64(air[p][Hp+i,Hp+j,k])
            end
            expected_vmr = FT(1e28 * 0.0289644 / (6.02214076e23 * selected_mass))
            @test all(x == zero(FT) || x == expected_vmr for panel in q for x in panel)
            molecules = sum(Float64(q[p][i,j,k]) * Float64(air[p][Hp+i,Hp+j,k])
                            for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc) *
                        6.02214076e23 / 0.0289644
            @test isapprox(molecules, 1e28; rtol=4eps(FT))
        end

        # The pressure selector must use total hybrid-coordinate pressure even
        # when the binary stores dry column pressure.
        qv = ntuple(_ -> reshape(repeat(FT[0.001, 0.004, 0.012, 0.02], Nc * Nc),
                                 Nc, Nc, Nz), 6)
        ps_dry = ntuple(p -> FT[
            sum((1 - qv[p][i, j, k]) *
                ((A[k + 1] - A[k]) + (B[k + 1] - B[k]) * ps[p][i, j])
                for k in 1:Nz)
            for i in 1:Nc, j in 1:Nc], 6)
        basis_cfg = Dict{String, Any}("kind" => "pressure_layer",
                                      "psurf_fraction" => 0.72,
                                      "total_molecules" => 1e28)
        from_total = PressureIC.build_initial_mixing_ratio(
            air, grid, basis_cfg; surface_pressure = ps,
            pressure_basis = TotalSurfacePressure(), qv)
        from_dry = PressureIC.build_initial_mixing_ratio(
            air, grid, basis_cfg; surface_pressure = ps_dry,
            pressure_basis = DrySurfacePressure(), qv)
        @test from_dry == from_total

        cfg = Dict{String,Any}("kind" => "pressure_layer", "total_molecules" => 1e28)
        @test_throws ArgumentError PressureIC.build_initial_mixing_ratio(air, grid, cfg)
        for fraction in (0, -0.1, 1.1, NaN)
            bad = merge(cfg, Dict("psurf_fraction" => fraction))
            @test_throws ArgumentError PressureIC.build_initial_mixing_ratio(air, grid, bad; surface_pressure=ps)
        end
        for amount in (0, -1, NaN)
            bad = merge(cfg, Dict("total_molecules" => amount))
            @test_throws ArgumentError PressureIC.build_initial_mixing_ratio(air, grid, bad; surface_pressure=ps)
        end
        empty_air = map(a -> zero(a), air)
        @test_throws ArgumentError PressureIC.build_initial_mixing_ratio(empty_air, grid, cfg; surface_pressure=ps)

        # Equal midpoint distances retain the first layer; lowest_layer bypasses
        # pressure-fraction parsing, including an otherwise invalid value.
        tie_grid = AtmosGrid(mesh, HybridSigmaPressure(zeros(FT, Nz+1), FT[0.1,0.1,0.1,0.8,1]), CPU(); FT)
        tie = merge(cfg, Dict("psurf_fraction" => 0.1))
        q = PressureIC.build_initial_mixing_ratio(air, tie_grid, tie; surface_pressure=ps)
        @test all(all(!iszero, panel[:,:,1]) && all(iszero, panel[:,:,2:end]) for panel in q)
        lowest = merge(cfg, Dict("lowest_layer" => true, "psurf_fraction" => "unused"))
        q = PressureIC.build_initial_mixing_ratio(air, grid, lowest; surface_pressure=ps)
        @test all(all(!iszero, panel[:,:,end]) && all(iszero, panel[:,:,1:end-1]) for panel in q)
    end
end


@testset "Cubed-sphere pressure-Gaussian initialization" begin
    # Mean of exp(-(s - c)^2 / (2 w^2)) over [a, b] by a fine midpoint sum.
    function reference_mean(a, b, c, w; n = 20_000)
        h = (b - a) / n
        return sum(exp(-0.5 * ((a + (m - 0.5) * h - c) / w)^2) for m in 1:n) / n
    end
    Nc, Hp = 3, 2
    # A coarse pure-sigma grid and a refinement of it (every layer split in four).
    coarse_edges = [0.0, 0.03, 0.08, 0.16, 0.27, 0.40, 0.52, 0.64, 0.75, 0.84, 0.91, 0.96, 1.0]
    fine_edges = vcat([range(coarse_edges[k], coarse_edges[k + 1]; length = 5)[1:4] for k in 1:12]..., 1.0)
    for FT in (Float32, Float64)
        mesh = CubedSphereMesh(; Nc, Hp, FT)
        ps = ntuple(p -> FT[65000 + 2500i + 1500j + 2000p for i in 1:Nc, j in 1:Nc], 6)
        function setup(edges)
            Nz = length(edges) - 1
            grid = AtmosGrid(mesh, HybridSigmaPressure(zeros(FT, Nz + 1), FT.(edges)), CPU(); FT)
            # Dry air mass proportional to the layer's pressure thickness; NaN halos.
            air = ntuple(p -> begin
                a = fill(FT(NaN), Nc + 2Hp, Nc + 2Hp, Nz)
                for k in 1:Nz, j in 1:Nc, i in 1:Nc
                    a[Hp+i, Hp+j, k] = FT((edges[k + 1] - edges[k]) * ps[p][i, j] * (1 + 0.1p))
                end
                a
            end, 6)
            return grid, air, Nz
        end
        molecules(q, air, Nz) = sum(Float64(q[p][i,j,k]) * Float64(air[p][Hp+i,Hp+j,k])
                                    for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc) * 6.02214076e23 / 0.0289644
        for (center, width) in ((0.5, 0.04), (0.85, 0.02), (1.0, 0.02), (0.07, 0.015), (0.7, 0.004))
            cfg = Dict{String,Any}("kind" => "pressure_gaussian", "psurf_fraction" => center,
                                   "width" => width, "total_molecules" => 1e28)
            grid_c, air_c, Nz_c = setup(coarse_edges)
            grid_f, air_f, Nz_f = setup(fine_edges)
            q_c = PressureIC.build_initial_mixing_ratio(air_c, grid_c, cfg; surface_pressure = ps)
            q_f = PressureIC.build_initial_mixing_ratio(air_f, grid_f, cfg; surface_pressure = ps)
            @test all(size(panel) == (Nc, Nc, Nz_c) && eltype(panel) == FT for panel in q_c)
            # The tolerance covers the summation order of up to 2592 terms.
            @test isapprox(molecules(q_c, air_c, Nz_c), 1e28; rtol = 64eps(FT))
            @test isapprox(molecules(q_f, air_f, Nz_f), 1e28; rtol = 64eps(FT))
            # Every column has the same profile: the layer means of the Gaussian,
            # up to the one global factor.
            means = [reference_mean(coarse_edges[k], coarse_edges[k + 1], center, width) for k in 1:Nz_c]
            factor = Float64(maximum(q_c[1][1, 1, :])) / maximum(means)
            for p in 1:6, j in 1:Nc, i in 1:Nc
                @test isapprox(Float64.(q_c[p][i, j, :]), factor .* means;
                               rtol = 0, atol = max(64eps(FT), 1e-7) * factor * maximum(means))
            end
            # The tracer in a coarse layer is the same on the refined grid.
            for p in (1, 4), j in 1:Nc, i in 1:Nc, k in 1:Nz_c
                on_coarse = Float64(q_c[p][i, j, k]) * Float64(air_c[p][Hp+i, Hp+j, k])
                on_fine = sum(Float64(q_f[p][i, j, kk]) * Float64(air_f[p][Hp+i, Hp+j, kk])
                              for kk in (4k - 3):(4k))
                @test isapprox(on_fine, on_coarse; rtol = 0,
                               atol = 64eps(FT) * 1e28 * 0.0289644 / 6.02214076e23 / (6 * Nc^2))
            end
        end
        # Center at the surface: largest in the lowest layer, falling monotonically upward.
        grid_f, air_f, Nz_f = setup(fine_edges)
        q = PressureIC.build_initial_mixing_ratio(air_f, grid_f,
            Dict{String,Any}("kind" => "pressure_gaussian", "psurf_fraction" => 1.0, "width" => 0.03);
            surface_pressure = ps)
        column = q[2][2, 3, :]
        @test argmax(column) == Nz_f
        @test issorted(column)
        @test column[1] == 0
    end

    # The layers are placed with the total hybrid-coordinate pressure, also
    # when the binary stores the dry column pressure.
    FT = Float64
    Nz = 4
    mesh = CubedSphereMesh(; Nc, Hp, FT)
    A = FT[0, 5000, 8000, 3000, 0]
    B = FT[0, 0.1, 0.3, 0.65, 1]
    grid = AtmosGrid(mesh, HybridSigmaPressure(A, B), CPU(); FT)
    ps = ntuple(p -> FT[65000 + 2500i + 1500j + 2000p for i in 1:Nc, j in 1:Nc], 6)
    # Air mass that differs in every cell and level, with NaN halos: a transposed
    # or shifted read changes the normalisation or returns NaN.
    air = ntuple(p -> begin
        a = fill(FT(NaN), Nc + 2Hp, Nc + 2Hp, Nz)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            a[Hp+i, Hp+j, k] = FT(1000 + 30p + 20k + 2j + i)
        end
        a
    end, 6)
    qv = ntuple(_ -> reshape(repeat(FT[0.001, 0.004, 0.012, 0.02], Nc * Nc), Nc, Nc, Nz), 6)
    ps_dry = ntuple(p -> FT[sum((1 - qv[p][i, j, k]) * ((A[k + 1] - A[k]) + (B[k + 1] - B[k]) * ps[p][i, j])
                                for k in 1:Nz) for i in 1:Nc, j in 1:Nc], 6)
    cfg = Dict{String,Any}("kind" => "pressure_gaussian", "psurf_fraction" => 0.72, "width" => 0.1,
                           "total_molecules" => 1e28)
    from_total = PressureIC.build_initial_mixing_ratio(air, grid, cfg; surface_pressure = ps,
                                                       pressure_basis = TotalSurfacePressure(), qv)
    from_dry = PressureIC.build_initial_mixing_ratio(air, grid, cfg; surface_pressure = ps_dry,
                                                     pressure_basis = DrySurfacePressure(), qv)
    @test all(isapprox(from_dry[p], from_total[p]; rtol = 1e-12) for p in 1:6)
    # On hybrid levels every column has its own sigma edges (A + B ps) / ps: check each
    # column against the layer means on those edges and the molecule count.
    weighted = 0.0
    means = [zeros(Nz) for _ in 1:6, _ in 1:Nc, _ in 1:Nc]
    for p in 1:6, j in 1:Nc, i in 1:Nc
        edges = (Float64.(A) .+ Float64.(B) .* ps[p][i, j]) ./ ps[p][i, j]
        means[p, i, j] .= [reference_mean(edges[k], edges[k + 1], 0.72, 0.1) for k in 1:Nz]
        weighted += sum(means[p, i, j][k] * air[p][Hp+i, Hp+j, k] for k in 1:Nz)
    end
    factor = 1e28 * 0.0289644 / (6.02214076e23 * weighted)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        @test isapprox(from_total[p][i, j, :], factor .* means[p, i, j]; rtol = 1e-7)
    end
    @test isapprox(sum(from_total[p][i,j,k] * air[p][Hp+i,Hp+j,k] for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc) *
                   6.02214076e23 / 0.0289644, 1e28; rtol = 64eps(FT))

    # Rejected inputs.
    bad(pairs...) = merge(cfg, Dict{String,Any}(pairs...))
    @test_throws ArgumentError PressureIC.build_initial_mixing_ratio(
        air, grid, delete!(copy(cfg), "width"); surface_pressure = ps)
    @test_throws ArgumentError PressureIC.build_initial_mixing_ratio(
        air, grid, bad("width" => 0.0); surface_pressure = ps)
    @test_throws ArgumentError PressureIC.build_initial_mixing_ratio(
        air, grid, bad("psurf_fraction" => 1.2); surface_pressure = ps)
    @test_throws ArgumentError PressureIC.build_initial_mixing_ratio(
        air, grid, bad("total_molecules" => -1.0); surface_pressure = ps)
    @test_throws ArgumentError PressureIC.build_initial_mixing_ratio(air, grid, cfg)
end
