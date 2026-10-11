# ---------------------------------------------------------------------------
# test_nested_cs_regrid.jl
#
# Nested cubed spheres (`src/Regridding/cs_nesting.jl`):
#   1. `meshes_nested` and the corner check;
#   2. `build_regridder` returns `NestedCubedSphereRegrid` for nested pairs
#      (after identity, before CR.jl) and caches nothing;
#   3. C4 ↔ C8 restriction and prolongation conserve to 1e-13 and equal a
#      ConservativeRegridding.jl `Regridder` of the same pair to 1e-12;
#   4. the merged block kernels `coarsen_sum!` / `coarsen_area_weighted!` and
#      the face sums against verbatim copies of the two kernel sets they
#      replace (Set A: `cubed_sphere_coarsen.jl`, Set B:
#      `geos_cs_resolution.jl`, both at commit 5a9930fb): bitwise equal to
#      Set A, and to Set B for sums, face sums and Float64 means; Float32
#      means within 5 ulp of Set B on positive data (Set B accumulated in
#      Float32; see docs/memos/2026-10-10_python_flux_port_p0_facts.md §8).
# ---------------------------------------------------------------------------

using Test
using Random
using AtmosTransport
using AtmosTransport.Regridding
using AtmosTransport.Quantities: IntensiveCellField, ExtensiveCellField
using AtmosTransport.Grids: GnomonicPanelConvention, GEOSNativePanelConvention,
                            EquiangularCubedSphereDefinition, GMAOCubedSphereDefinition
import ConservativeRegridding

const NestRG = AtmosTransport.Regridding
const NestPP = AtmosTransport.Preprocessing

nest_bits(x::Float32) = reinterpret(UInt32, x)
nest_bits(x::Float64) = reinterpret(UInt64, x)
nest_bitwise(a, b) = size(a) == size(b) && all(nest_bits.(a) .== nest_bits.(b))
# Distance in units in the last place of the larger magnitude.
nest_ulp(a, b) = a == b ? 0.0 : Float64(abs(big(a) - big(b)) / big(eps(max(abs(a), abs(b)))))

# --- Set A (cubed_sphere_coarsen.jl at 5a9930fb), verbatim -------------------
function setA_sum_cells3!(dst::AbstractArray{FT, 3}, src::AbstractArray{FT, 3}, ratio::Int) where FT
    Nc, Nz = size(dst, 1), size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        total = zero(FT)
        i0 = (i - 1) * ratio
        j0 = (j - 1) * ratio
        for jj in 1:ratio, ii in 1:ratio
            total += src[i0 + ii, j0 + jj, k]
        end
        dst[i, j, k] = total
    end
    return dst
end

function setA_weighted_cells2!(dst::AbstractMatrix{FT}, src::AbstractMatrix{FT},
                               source_area::AbstractMatrix, ratio::Int) where FT
    Nc = size(dst, 1)
    @inbounds for j in 1:Nc, i in 1:Nc
        numerator = 0.0
        denominator = 0.0
        i0 = (i - 1) * ratio
        j0 = (j - 1) * ratio
        for jj in 1:ratio, ii in 1:ratio
            area = Float64(source_area[i0 + ii, j0 + jj])
            numerator += Float64(src[i0 + ii, j0 + jj]) * area
            denominator += area
        end
        dst[i, j] = FT(numerator / denominator)
    end
    return dst
end

function setA_xfaces!(dst::AbstractArray{FT, 3}, src::AbstractArray{FT, 3}, ratio::Int) where FT
    Nc, Nz = size(dst, 2), size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:(Nc + 1)
        fine_i = (i - 1) * ratio + 1
        j0 = (j - 1) * ratio
        total = zero(FT)
        for jj in 1:ratio
            total += src[fine_i, j0 + jj, k]
        end
        dst[i, j, k] = total
    end
    return dst
end

function setA_yfaces!(dst::AbstractArray{FT, 3}, src::AbstractArray{FT, 3}, ratio::Int) where FT
    Nc, Nz = size(dst, 1), size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:(Nc + 1), i in 1:Nc
        fine_j = (j - 1) * ratio + 1
        i0 = (i - 1) * ratio
        total = zero(FT)
        for ii in 1:ratio
            total += src[i0 + ii, fine_j, k]
        end
        dst[i, j, k] = total
    end
    return dst
end

# --- Set B (geos_cs_resolution.jl at 5a9930fb), verbatim ---------------------
function setB_sum3!(dst::AbstractArray{FT, 3}, src::AbstractArray{FT, 3}, ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        s = zero(FT)
        for jj in ((j - 1) * R + 1):(j * R), ii in ((i - 1) * R + 1):(i * R)
            s += src[ii, jj, k]
        end
        dst[i, j, k] = s
    end
    return dst
end

function setB_area_weighted2!(dst::AbstractMatrix{FT}, src::AbstractMatrix{FT},
                              src_area::AbstractMatrix{FT}, ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    @inbounds for j in 1:Nc, i in 1:Nc
        num = zero(FT)
        den = zero(FT)
        for jj in ((j - 1) * R + 1):(j * R), ii in ((i - 1) * R + 1):(i * R)
            a = src_area[ii, jj]
            num += src[ii, jj] * a
            den += a
        end
        dst[i, j] = num / den
    end
    return dst
end

function setB_area_weighted3!(dst::AbstractArray{FT, 3}, src::AbstractArray{FT, 3},
                              src_area::AbstractMatrix{FT}, ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        num = zero(FT)
        den = zero(FT)
        for jj in ((j - 1) * R + 1):(j * R), ii in ((i - 1) * R + 1):(i * R)
            a = src_area[ii, jj]
            num += src[ii, jj, k] * a
            den += a
        end
        dst[i, j, k] = num / den
    end
    return dst
end

function setB_xface_sum!(dst::AbstractArray{FT, 3}, src::AbstractArray{FT, 3}, ::Val{R}) where {FT, R}
    Nc = size(dst, 2)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:(Nc + 1)
        fi = (i - 1) * R + 1
        s = zero(FT)
        for fj in ((j - 1) * R + 1):(j * R)
            s += src[fi, fj, k]
        end
        dst[i, j, k] = s
    end
    return dst
end

function setB_yface_sum!(dst::AbstractArray{FT, 3}, src::AbstractArray{FT, 3}, ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:(Nc + 1), i in 1:Nc
        fj = (j - 1) * R + 1
        s = zero(FT)
        for fi in ((i - 1) * R + 1):(i * R)
            s += src[fi, fj, k]
        end
        dst[i, j, k] = s
    end
    return dst
end

# Test data of the P0 probe: uniform [0, 1), signed over 12 decades, and
# air-mass-like (1e9 + 1e13·U).
function nest_data(rng, ::Type{FT}, dims, kind::Symbol) where FT
    kind === :unit && return rand(rng, FT, dims...)
    kind === :signed_wide && return FT.(randn(rng, dims...) .* 10.0 .^ (12 .* rand(rng, dims...) .- 6))
    kind === :air_mass && return FT.(1e9 .+ 1e13 .* rand(rng, dims...))
    error("unknown data kind $(kind)")
end

nest_mesh(Nc, def; FT = Float64) = CubedSphereMesh(; Nc, FT, Hp = 0, definition = def)

const NEST_DEFINITIONS = (equiangular = EquiangularCubedSphereDefinition(),
                          gmao = GMAOCubedSphereDefinition())

@testset "nested cubed-sphere regridding" begin

    @testset "meshes_nested" begin
        for def in NEST_DEFINITIONS
            c4, c8, c12 = nest_mesh(4, def), nest_mesh(8, def), nest_mesh(12, def)
            @test meshes_nested(c4, c8) && meshes_nested(c8, c4)
            @test meshes_nested(c4, c12)
            @test !meshes_nested(c8, c12)                 # 12/8 is not an integer
            @test !meshes_nested(c4, nest_mesh(4, def))   # equal Nc: equivalent, not nested
            @test meshes_equivalent(c4, nest_mesh(4, def))
            # Precision does not change the geometry.
            @test meshes_nested(nest_mesh(4, def; FT = Float32), c8)
        end
        e4 = nest_mesh(4, NEST_DEFINITIONS.equiangular)
        g8 = nest_mesh(8, NEST_DEFINITIONS.gmao)
        @test !meshes_nested(e4, g8)                      # different definitions
        @test !meshes_nested(e4, CubedSphereMesh(; Nc = 8, Hp = 0, radius = 6.0e6))
        @test !meshes_nested(e4, LatLonMesh(Nx = 36, Ny = 18))
        # The corner check rejects corner laws that do not nest.
        @test !NestRG._corners_nest(nest_mesh(8, NEST_DEFINITIONS.equiangular),
                                    nest_mesh(4, NEST_DEFINITIONS.gmao), 2)
        # Production pairs nest to round-off.
        for (Nf, Nc) in ((90, 30), (180, 90))
            @test meshes_nested(nest_mesh(Nf, NEST_DEFINITIONS.gmao),
                                nest_mesh(Nc, NEST_DEFINITIONS.gmao))
        end
    end

    @testset "build_regridder selects the nested regridder" begin
        c4 = nest_mesh(4, NEST_DEFINITIONS.equiangular)
        c8 = nest_mesh(8, NEST_DEFINITIONS.equiangular)
        mktempdir() do cache
            down = build_regridder(c8, c4; cache_dir = cache)
            up = build_regridder(c4, c8; cache_dir = cache)
            @test down isa NestedCubedSphereRegrid{Restriction}
            @test up isa NestedCubedSphereRegrid{Prolongation}
            @test down.ratio == up.ratio == 2
            @test isempty(readdir(cache))                 # nothing is cached
            @test build_regridder(c4, nest_mesh(4, NEST_DEFINITIONS.equiangular)) isa IdentityRegrid
        end
        r = build_regridder(c8, c4)
        @test r.src_areas == repeat(vec(c8.cell_areas), 6)
        @test r.dst_areas == repeat(vec(c4.cell_areas), 6)
        # A Float32 mesh gives the same (Float64) areas.
        r32 = build_regridder(nest_mesh(8, NEST_DEFINITIONS.equiangular; FT = Float32), c4)
        @test r32.src_areas == r.src_areas
        # Inconsistent sizes are rejected.
        @test_throws ArgumentError NestedCubedSphereRegrid{Restriction}(1, r.src_areas, r.dst_areas)
        @test_throws DimensionMismatch NestedCubedSphereRegrid{Restriction}(3, r.src_areas, r.dst_areas)
        @test_throws DimensionMismatch NestedCubedSphereRegrid{Prolongation}(2, r.src_areas, r.dst_areas)
        @test_throws ArgumentError NestedCubedSphereRegrid(c4, nest_mesh(6, NEST_DEFINITIONS.equiangular))
        @test_throws DimensionMismatch apply_regridder!(zeros(10), r, zeros(length(r.src_areas)))
    end

    @testset "C4 ↔ C8 conservation and agreement with CR.jl ($(name))" for (name, def) in pairs(NEST_DEFINITIONS)
        c4, c8 = nest_mesh(4, def), nest_mesh(8, def)
        rng = MersenneTwister(20261010)
        for (src_mesh, dst_mesh, D) in ((c8, c4, Restriction), (c4, c8, Prolongation))
            r = build_regridder(src_mesh, dst_mesh)
            @test r isa NestedCubedSphereRegrid{D}
            cr = ConservativeRegridding.Regridder(dst_mesh, src_mesh; normalize = false)
            n_src, n_dst = length(r.src_areas), length(r.dst_areas)

            # Extensive (kg per cell), several columns: totals conserved to 1e-13.
            mass = 1e12 .* (0.5 .+ rand(rng, n_src, 3))
            out = zeros(n_dst, 3)
            apply_regridder!(out, r, mass, ExtensiveCellField())
            @test abs(sum(out) / sum(mass) - 1) < 1e-13
            rep = regrid_conservation_report(r, mass, out, ExtensiveCellField())
            @test rep.rel_error < 1e-13
            @test rep.src_total == sum(Float64, mass)
            out_cr = zeros(n_dst, 3)
            apply_regridder!(out_cr, cr, mass, ExtensiveCellField())
            @test maximum(abs.(out .- out_cr)) / maximum(abs.(out_cr)) < 1e-12

            # Intensive, signed: area integral conserved; equal to CR.jl to 1e-12.
            q = randn(rng, n_src)
            q_out = zeros(n_dst)
            apply_regridder!(q_out, r, q)
            @test regrid_conservation_report(r, q, q_out, IntensiveCellField()).rel_error < 1e-13
            q_cr = zeros(n_dst)
            apply_regridder!(q_cr, cr, q)
            @test maximum(abs.(q_out .- q_cr)) / maximum(abs.(q)) < 1e-12
            # The kind-tagged intensive call is the plain call.
            q_tag = zeros(n_dst)
            apply_regridder!(q_tag, r, q, IntensiveCellField())
            @test nest_bitwise(q_tag, q_out)

            # A uniform intensive field stays uniform to the nesting round-off.
            u_out = zeros(n_dst)
            apply_regridder!(u_out, r, fill(3.0, n_src))
            @test maximum(abs.(u_out .- 3.0)) < 3e-13

            # Columns are independent: the matrix call equals column calls.
            for col in 1:3
                one_col = zeros(n_dst)
                apply_regridder!(one_col, r, mass[:, col], ExtensiveCellField())
                @test nest_bitwise(one_col, out[:, col])
            end
        end
        # Restriction is a block sum of extensive amounts; prolongation splits a
        # coarse amount by area fraction.
        down = build_regridder(c8, c4)
        up = build_regridder(c4, c8)
        mass8 = 1e12 .* (0.5 .+ rand(rng, length(down.src_areas)))
        mass4 = zeros(length(down.dst_areas))
        apply_regridder!(mass4, down, mass8, ExtensiveCellField())
        panels8 = reshape(mass8, 8, 8, 6)
        block = zeros(4, 4, 6)
        coarsen_sum!(block, panels8, 2)
        @test maximum(abs.(vec(block) ./ mass4 .- 1)) < 1e-14
        split8 = zeros(length(up.dst_areas))
        apply_regridder!(split8, up, mass4, ExtensiveCellField())
        areas8 = reshape(up.dst_areas, 8, 8, 6)
        area_blocks = zeros(4, 4, 6)
        coarsen_sum!(area_blocks, areas8, 2)
        expected = [mass4[(p - 1) * 16 + cld(i, 2) + (cld(j, 2) - 1) * 4] * areas8[i, j, p] /
                    area_blocks[cld(i, 2), cld(j, 2), p] for i in 1:8, j in 1:8, p in 1:6]
        @test maximum(abs.(vec(expected) ./ split8 .- 1)) < 1e-14
    end

    @testset "merged block kernels against Set A and Set B" begin
        # Float32 data with Float64 areas (what the flux path holds).
        let src = rand(MersenneTwister(1), Float32, 8, 8), area = 1e8 .* (0.5 .+ rand(MersenneTwister(2), 8, 8))
            a = zeros(Float32, 4, 4); b = zeros(Float32, 4, 4)
            coarsen_area_weighted!(a, src, area, 2)
            setA_weighted_cells2!(b, src, area, 2)
            @test nest_bitwise(a, b)
        end
        # 2-D cell sums are the 3-D kernel with one level.
        let src = rand(MersenneTwister(3), Float32, 12, 12)
            a = zeros(Float32, 4, 4); b = zeros(Float32, 4, 4, 1)
            coarsen_sum!(a, src, 3)
            coarsen_sum!(b, reshape(src, 12, 12, 1), 3)
            @test nest_bitwise(a, b[:, :, 1])
        end
        @test_throws DimensionMismatch coarsen_sum!(zeros(4, 4), zeros(9, 8), 2)
        @test_throws DimensionMismatch coarsen_area_weighted!(zeros(4, 4), zeros(8, 8), ones(4, 4), 2)
        @test_throws DimensionMismatch NestPP._coarsen_xface_sum!(zeros(5, 4, 1), zeros(8, 8, 1), 2)

        Nz = 7
        worst_ulp = 0.0
        for (Nsrc, Ndst) in ((8, 4), (12, 4), (90, 30)), FT in (Float32, Float64),
            kind in (:unit, :signed_wide, :air_mass)
            R = Nsrc ÷ Ndst
            rng = MersenneTwister(hash((Nsrc, FT, kind)))
            label = "C$(Nsrc)→C$(Ndst) $(FT) $(kind)"

            src3 = nest_data(rng, FT, (Nsrc, Nsrc, Nz), kind)
            new = zeros(FT, Ndst, Ndst, Nz)
            coarsen_sum!(new, src3, R)
            @test nest_bitwise(new, setA_sum_cells3!(zeros(FT, Ndst, Ndst, Nz), src3, R))
            @test nest_bitwise(new, setB_sum3!(zeros(FT, Ndst, Ndst, Nz), src3, Val(R)))

            srcx = nest_data(rng, FT, (Nsrc + 1, Nsrc, Nz), kind)
            newx = NestPP._coarsen_xface_sum!(zeros(FT, Ndst + 1, Ndst, Nz), srcx, R)
            @test nest_bitwise(newx, setA_xfaces!(zeros(FT, Ndst + 1, Ndst, Nz), srcx, R))
            @test nest_bitwise(newx, setB_xface_sum!(zeros(FT, Ndst + 1, Ndst, Nz), srcx, Val(R)))
            srcy = nest_data(rng, FT, (Nsrc, Nsrc + 1, Nz), kind)
            newy = NestPP._coarsen_yface_sum!(zeros(FT, Ndst, Ndst + 1, Nz), srcy, R)
            @test nest_bitwise(newy, setA_yfaces!(zeros(FT, Ndst, Ndst + 1, Nz), srcy, R))
            @test nest_bitwise(newy, setB_yface_sum!(zeros(FT, Ndst, Ndst + 1, Nz), srcy, Val(R)))

            areas = (nest_mesh(Nsrc, NEST_DEFINITIONS.gmao; FT).cell_areas,
                     nest_mesh(Nsrc, NEST_DEFINITIONS.equiangular; FT).cell_areas,
                     FT.(1e8 .* (0.5 .+ rand(rng, Nsrc, Nsrc))))
            for area in areas
                src2 = nest_data(rng, FT, (Nsrc, Nsrc), kind)
                new2 = coarsen_area_weighted!(zeros(FT, Ndst, Ndst), src2, area, R)
                a2 = setA_weighted_cells2!(zeros(FT, Ndst, Ndst), src2, area, R)
                b2 = setB_area_weighted2!(zeros(FT, Ndst, Ndst), src2, area, Val(R))
                @test nest_bitwise(new2, a2)

                s3 = nest_data(rng, FT, (Nsrc, Nsrc, Nz), kind)
                new3 = coarsen_area_weighted!(zeros(FT, Ndst, Ndst, Nz), s3, area, R)
                a3 = zeros(FT, Ndst, Ndst, Nz)
                for k in 1:Nz
                    setA_weighted_cells2!(view(a3, :, :, k), view(s3, :, :, k), area, R)
                end
                b3 = setB_area_weighted3!(zeros(FT, Ndst, Ndst, Nz), s3, area, Val(R))
                @test nest_bitwise(new3, a3)

                if FT === Float64
                    @test nest_bitwise(new2, b2)
                    @test nest_bitwise(new3, b3)
                elseif kind !== :signed_wide
                    # Set B accumulated Float32 means in Float32: not bitwise.
                    # 5 ulp is the bound measured in P0 on these data (it is not
                    # a theoretical bound: Set B's error can reach ~2R² ulp).
                    worst = max(maximum(nest_ulp.(new2, b2)), maximum(nest_ulp.(new3, b3)))
                    worst_ulp = max(worst_ulp, worst)
                    @test worst <= 5
                end
            end
        end
        @info "merged Float32 area-weighted means vs Set B (positive data)" max_ulp = worst_ulp
    end
end
