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
