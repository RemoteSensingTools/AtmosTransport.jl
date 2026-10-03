# The legacy CATRINE TM5 convection files store ERA5 L137 levels surface
# first. The attach script must map each file level to its ERA5 level before
# applying the binary's top-first merge_map; summing them in file order put
# the convection upside down (entrainment near the model top).
using Test, Dates, NCDatasets, AtmosTransport

module AttachScript
include(joinpath(@__DIR__, "..", "..", "scripts", "preprocessing",
                 "attach_catrine_tm5_convection_cs.jl"))
end

const PP = AtmosTransport.Preprocessing

@testset "legacy TM5 convection level order" begin
    vc = PP.load_hybrid_coefficients(AttachScript.ERA5_L137_COEFFICIENTS)
    A, B = Float64.(vc.A), Float64.(vc.B)
    amid = (A[1:end-1] .+ A[2:end]) ./ 2
    bmid = (B[1:end-1] .+ B[2:end]) ./ 2

    @test AttachScript.era5_level_of_file_level(amid, bmid, A, B) == 1:137
    surface_first = AttachScript.era5_level_of_file_level(reverse(amid), reverse(bmid), A, B)
    @test surface_first == 137:-1:1
    @test_throws ErrorException AttachScript.era5_level_of_file_level(amid .+ 50, bmid, A, B)
    @test_throws ErrorException AttachScript.era5_level_of_file_level(amid[1:136], bmid[1:136], A, B)

    # A surface-first profile with all its mass in the lowest file level lands
    # in the bottom target level, whatever the merge.
    merge_map = [cld(k, 2) for k in 1:137]                 # 137 -> 69, top first
    Nz = maximum(merge_map)
    native = zeros(2, 1, 137); native[:, :, 1] .= 1.0      # file level 1 = surface
    merged = AttachScript.merge_levels!(zeros(2, 1, Nz), native, merge_map, surface_first)
    @test merged[:, :, Nz] == ones(2, 1) && sum(merged) == 2
    @test AttachScript.level_order_label(surface_first) == "surface_first_reversed"
    @test AttachScript.level_order_label(1:137) == "toa_first"

    # The post-merge guard: entrainment near the surface passes, near the top fails.
    bottom = ntuple(_ -> (a = zeros(Float32, 2, 2, 10); a[:, :, 9:10] .= 1; a), 6)
    top = ntuple(_ -> (a = zeros(Float32, 2, 2, 10); a[:, :, 1:2] .= 1; a), 6)
    @test AttachScript.updraft_mean_level(bottom) == 9.5
    @test AttachScript.updraft_mean_level(top) == 1.5
    @test isnan(AttachScript.updraft_mean_level(ntuple(_ -> zeros(Float32, 2, 2, 10), 6)))
end

@testset "legacy TM5 convection file checks" begin
    mktempdir() do dir
        vc = PP.load_hybrid_coefficients(AttachScript.ERA5_L137_COEFFICIENTS)
        A, B = Float64.(vc.A), Float64.(vc.B)
        function write_file(path; lon = collect(-179.5:1.0:179.5), start = DateTime(2014, 1, 1))
            NCDataset(path, "c") do ds
                defDim(ds, "lon", 360); defDim(ds, "lat", 180); defDim(ds, "lev", 137); defDim(ds, "time", 8)
                defVar(ds, "lon", Float64, ("lon",))[:] = lon
                defVar(ds, "lat", Float64, ("lat",))[:] = collect(-89.5:1.0:89.5)
                defVar(ds, "ap", Float64, ("lev",))[:] = reverse((A[1:end-1] .+ A[2:end]) ./ 2)
                defVar(ds, "b", Float64, ("lev",))[:] = reverse((B[1:end-1] .+ B[2:end]) ./ 2)
                # Corrupt CF time, as in the 2022-04-30 to 2023-12-31 files: must be ignored.
                defVar(ds, "time", Float64, ("time",);
                       attrib = Dict("units" => "seconds since 1900-01-01 00:00:00"))[:] = zeros(8)
                defDim(ds, "timeval", 6); defDim(ds, "nv", 2)
                parts(t) = [year(t), month(t), day(t), hour(t), minute(t), second(t)]
                tb = defVar(ds, "timevalues_bounds", Int32, ("nv", "timeval", "time"))
                for k in 1:8
                    tb[1, :, k] = parts(start + Hour(3 * (k - 1)))
                    tb[2, :, k] = parts(start + Hour(3 * k))
                end
                for v in ("eu", "du", "ed", "dd")
                    defVar(ds, v, Float32, ("lon", "lat", "lev", "time"))
                end
            end
            return path
        end
        binary = "/x/era5_n320_transport_20140101_float32.bin"
        check(path; input = binary) = NCDataset(ds -> AttachScript.check_convection_file(ds, path, input), path)
        good = write_file(joinpath(dir, "convec_20140101_00p03.nc"))
        @test check(good) == 137:-1:1
        @test_throws ErrorException check(good; input = "/x/era5_n320_transport_20140102_float32.bin")
        # Same file name in separate folders, so only the tested property differs.
        variant(name; kwargs...) = write_file(joinpath(mkpath(joinpath(dir, name)),
                                                       "convec_20140101_00p03.nc"); kwargs...)
        @test_throws ErrorException check(variant("lon_0_360"; lon = collect(0.5:1.0:359.5)))
        @test_throws ErrorException check(variant("late_slots"; start = DateTime(2014, 1, 1, 1)))
    end
end
