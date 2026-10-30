# ---------------------------------------------------------------------------
# test_serialization.jl
#
# Serialization round-trip tests for:
#   1. JLD2 save/load (strict identity of intersections + areas)
#   2. On-disk cache (build_regridder hits the cached path on 2nd call)
# ---------------------------------------------------------------------------


@testset "serialization round-trips" begin
    Nx, Ny, Nc = 36, 18, 4
    src = LatLonMesh(Nx = Nx, Ny = Ny)
    dst = CubedSphereMesh(Nc = Nc, convention = GnomonicPanelConvention())

    r = build_regridder(src, dst; normalize = false)

    @testset "JLD2 explicit save/load" begin
        mktempdir() do tmp
            path = joinpath(tmp, "explicit.jld2")
            save_regridder(path, r)
            @test isfile(path)
            r2 = load_regridder(path)
            @test r2.intersections == r.intersections
            @test r2.dst_areas == r.dst_areas
            @test r2.src_areas == r.src_areas
        end
    end

    @testset "build_regridder cache_dir hit" begin
        mktempdir() do tmp
            @test_throws ArgumentError build_regridder(
                src, dst; cache_dir = tmp, intersection_operator = :custom)
            # First call builds
            r1 = build_regridder(src, dst; normalize = false, cache_dir = tmp)
            # Second call should return an equivalent regridder from cache
            r2 = build_regridder(src, dst; normalize = false, cache_dir = tmp)
            @test r1.intersections == r2.intersections
            @test r1.dst_areas == r2.dst_areas
            @test r1.src_areas == r2.src_areas
        end
    end

end
