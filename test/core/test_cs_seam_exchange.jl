using Test, LinearAlgebra
include(joinpath(@__DIR__, "..", "helpers", "cs_seam_exchange.jl"))

@testset "CS seam exchange agrees with independent scalar loops" begin
    for FT in (Float32, Float64), Nt in (1, 7), convention in
            (CSSeamGrids.GnomonicPanelConvention(), CSSeamGrids.GEOSNativePanelConvention()),
            scheme in (UpwindScheme(), SlopesScheme(MonotoneLimiter()), PPMScheme())
        mesh, m, packed, am, bm = cs_seam_fixture(FT, 5, 3, Nt, convention)
        ws = CSSeamAdv.CSAdvectionWorkspace(mesh, m[1]; n_tracers=Nt)
        original = map(copy, packed)
        for direction in (1, 2)
            reference = [reference_cs_group(map(a -> a[:, :, :, t], original),
                m, am, bm, mesh, scheme, direction, FT(1)) for t in 1:Nt]
            actual, air = map(copy, original), map(copy, m)
            CSSeamAdv._sweep_cs_horizontal!(actual, air, direction == 1 ? am : bm,
                mesh, scheme, ws, Val(direction))
            tol = FT == Float32 ? 5e-6 : 2e-14
            @test cs_seam_maxdiff(air, reference[1][2], mesh) < tol
            for t in 1:Nt
                @test cs_seam_maxdiff(map(a -> a[:, :, :, t], actual), reference[t][1], mesh) < tol
            end
        end
    end
end

# The seam application in its former structure: one pass per seam of the
# group, in seam order, each subtracting on the lower panel and adding on the
# neighbor (face index Nc + 1 maps to cell Nc).
function seam_by_seam_apply!(rm, m, cache, mesh, direction)
    Nc, Hp, Nz, Nt = mesh.Nc, mesh.Hp, size(m[1], 3), size(rm[1], 4)
    conn, seam = mesh.connectivity, 0
    for p in 1:6, edge in 1:4
        q = conn.neighbors[p][edge].panel
        p < q || continue
        seam += 1
        CSSeamAdv._cs_seam_axis(edge) == direction || continue
        other = CSSeamGrids.reciprocal_edge(conn, p, edge)
        for k in 1:Nz, s in 1:Nc
            t = conn.neighbors[p][edge].orientation != 0 ? Nc + 1 - s : s
            i, j = Hp .+ min.(CSSeamAdv._cs_seam_face_index(edge, s, Nc), Nc)
            u, v = Hp .+ min.(CSSeamAdv._cs_seam_face_index(other, t, Nc), Nc)
            m[p][i, j, k] -= cache[s, k, Nt + 1, seam]
            m[q][u, v, k] += cache[s, k, Nt + 1, seam]
            for tr in 1:Nt
                rm[p][i, j, k, tr] += -cache[s, k, tr, seam]
                rm[q][u, v, k, tr] += cache[s, k, tr, seam]
            end
        end
    end
end

# Raw bits, so that the sign of a zero counts.
seam_bits(a) = reinterpret(UInt32, a)

@testset "CS seam application: one launch, seam order at shared corners" begin
    rng = MersenneTwister(4711)
    for convention in (CSSeamGrids.GnomonicPanelConvention(), CSSeamGrids.GEOSNativePanelConvention()),
            Nc in (1, 5), Nt in (1, 3), direction in (1, 2)
        Hp, Nz = 3, 2
        mesh = CubedSphereMesh(; FT = Float32, Nc, Hp, convention)
        N = Nc + 2Hp
        m = ntuple(_ -> 10f0 .+ rand(rng, Float32, N, N, Nz), 6)
        rm = ntuple(_ -> rand(rng, Float32, N, N, Nz, Nt), 6)
        # Transfers much larger than the cell values, so that a different
        # addition order at a shared corner changes the rounded result, and
        # signed zeros in the cells and the transfers.
        cache = 1f7 .* randn(rng, Float32, Nc, Nz, Nt + 1, 12)
        cache[rand(rng, eachindex(cache), 40)] .= -0f0
        cache[rand(rng, eachindex(cache), 40)] .= 0f0
        foreach(a -> a[rand(rng, eachindex(a), 40)] .= -0f0, rm)
        ops = CSSeamAdv._seam_apply_ops(mesh.connectivity, Val(direction), Nc)
        if Nc > 1
            # Fused contract: the operations write pairwise-disjoint cells.
            writes = Dict{NTuple{4, Int}, Int}()
            for op in ops.ops
                air = map(zero, m)
                # Distinct per-seam transfers: two contributions cannot cancel.
                marks = Float32.(reshape(2 .^ (1:12), 1, 1, 1, 12)) .* ones(Float32, size(cache))
                ctx = (; panels_rm = map(zero, rm), panels_m = air, cache = marks, Nc, Hp, Nt)
                CSSeamAdv.launch!(op, ctx, CSSeamAdv.get_backend(air[1]))
                for p in 1:6, I in findall(!iszero, air[p])
                    writes[(p, Tuple(I)...)] = get(writes, (p, Tuple(I)...), 0) + 1
                end
            end
            @test all(==(1), values(writes))
            # Corners shared by two seams of the group (worked out by hand from
            # the connectivity tables): GEOS 4 per group; gnomonic X none, Y the
            # 8 corners of the polar panels.
            shared = sum(op -> count(>(0), op.partner_seam), ops.ops) ÷ 2
            @test shared == (convention isa CSSeamGrids.GEOSNativePanelConvention ? 4 :
                             direction == 1 ? 0 : 8)
            @test length(writes) == (count(op -> op.active, ops.ops) * 2Nc - shared) * Nz
        else
            shared = 0
            @test all(op -> all(iszero, op.partner_seam), ops.ops)   # applied seam by seam
        end
        expected_rm, expected_m = map(copy, rm), map(copy, m)
        seam_by_seam_apply!(expected_rm, expected_m, cache, mesh, direction)
        actual_rm, actual_m = map(copy, rm), map(copy, m)
        panels_rm = Nt == 1 ? map(a -> a[:, :, :, 1], actual_rm) : actual_rm
        CSSeamAdv._apply_cs_seams!(panels_rm, actual_m, cache, mesh, Val(direction))
        @test all(p -> seam_bits(actual_m[p]) == seam_bits(expected_m[p]), 1:6)
        @test all(p -> seam_bits(panels_rm[p]) ==
                       seam_bits(Nt == 1 ? expected_rm[p][:, :, :, 1] : expected_rm[p]), 1:6)
        # The test detects a reversed order at the shared corners.
        if shared > 0
            reversed_rm, reversed_m = map(copy, rm), map(copy, m)
            for op in reverse(ops.ops)
                CSSeamAdv.launch!(CSSeamAdv.SeamTransferApply{direction}(op.p, op.q, op.edge,
                    op.other_edge, op.reversed, op.seam, op.active, (0, 0, 0, 0), (0, 0, 0, 0),
                    (false, false, false, false)),
                    (; panels_rm = reversed_rm, panels_m = reversed_m, cache, Nc, Hp, Nt),
                    CSSeamAdv.get_backend(m[1]))
            end
            @test reversed_m != expected_m || reversed_rm != expected_rm
        end
    end
end

@testset "CS seam adjoint differentiates the coupled production sweep" begin
    Adj = AtmosTransport.Adjoints
    for convention in (CSSeamGrids.GnomonicPanelConvention(), CSSeamGrids.GEOSNativePanelConvention()),
            scheme in (UpwindScheme(), SlopesScheme(CSSeamAdv.NoLimiter()),
                       PPMScheme(CSSeamAdv.NoLimiter()), PPMScheme()), axis in (1, 2)
        mesh, m, packed, am, bm = cs_seam_fixture(Float64, 5, 3, 2, convention)
        rm = map(a -> a[:, :, :, 2], packed)
        rng = MersenneTwister(2917)
        direction = map(a -> randn(rng, size(a)), rm)
        seed = map(a -> zero(a), rm)
        for p in 1:6
            cs_seam_interior(seed[p], mesh) .= randn(rng, mesh.Nc, mesh.Nc, 3)
            # Only physical cells are independent inputs; halo values are copied.
            interior = copy(cs_seam_interior(direction[p], mesh))
            fill!(direction[p], 0)
            cs_seam_interior(direction[p], mesh) .= interior
        end
        flux = axis == 1 ? am : bm
        symbol = axis == 1 ? :x : :y
        ws = CSSeamAdv.CSAdvectionWorkspace(mesh, m[1])
        aws = Adj.CSAdjointWorkspace(mesh, m[1])
        function objective(h)
            r = map((a, d) -> a .+ h .* d, rm, direction)
            mass = map(copy, m)
            CSSeamAdv.fill_panel_halos!(r, mesh)
            CSSeamAdv._sweep_cs_horizontal!(r, mass, flux, mesh, scheme, ws, Val(axis))
            return sum(dot(seed[p], r[p]) for p in 1:6)
        end
        gradient = map(copy, seed)
        if scheme isa PPMScheme{MonotoneLimiter}
            Adj._adjoint_scheme_sweep!(gradient, m, rm, flux, symbol, scheme, mesh, aws, 1.0)
        else
            Adj._adjoint_scheme_sweep!(gradient, m, flux, symbol, scheme, mesh, aws, 1.0)
        end
        Adj._adjoint_fill_panel_halos!(gradient, mesh)
        expected = sum(dot(gradient[p], direction[p]) for p in 1:6)
        h = 1e-5
        finite_difference = (objective(h) - objective(-h)) / (2h)
        @test expected ≈ finite_difference rtol=2e-7 atol=2e-8
    end
end

@testset "CS seam cache follows its scheme and rejects incompatible geometry" begin
    mesh, m, packed, am, bm = cs_seam_fixture(Float64, 5, 3, 7, CSSeamGrids.GEOSNativePanelConvention())
    lr = CSSeamAdv.CSLinRoodAdvectionWorkspace(mesh, m[1]; n_tracers=7)
    @test isempty(lr.cs.seam_flux)
    original = map(copy, packed)
    @test_throws DimensionMismatch CSSeamAdv._sweep_cs_horizontal!(
        packed, m, am, mesh, PPMScheme(), lr.cs, Val(1))
    narrow = CSSeamAdv.CSAdvectionWorkspace(mesh, m[1])
    @test_throws DimensionMismatch CSSeamAdv._sweep_cs_horizontal!(
        packed, m, am, mesh, PPMScheme(), narrow, Val(1))
    @test packed == original
end
