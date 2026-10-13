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
const MD = AtmosTransport.MetDrivers

@testset "humidity remap uses the ERA5 manifold" begin
    transport_mesh = LatLonMesh(; FT = Float64, Nx = 4, Ny = 2,
                                radius = AtmosTransport.Parameters.EARTH_RADIUS)
    opened = Ref{Any}()
    opener(root, date, target; cache_dir) =
        (opened[] = (; root, date, target, cache_dir); :mock_remap)
    result = AttachScript.humidity_remap(
        DryBasis(), Dict{String, Any}(), "/era5", Date(2021, 12, 1),
        transport_mesh, "/cache"; opener)
    @test result === :mock_remap
    humidity_mesh = opened[].target
    @test humidity_mesh.λᶠ == transport_mesh.λᶠ
    @test humidity_mesh.φᶠ == transport_mesh.φᶠ
    @test humidity_mesh.radius == AtmosTransport.Parameters.IFS_EARTH_RADIUS
    @test opened[].root == "/era5"
    @test opened[].cache_dir == "/cache"
end

@testset "attachment preserves dry-mass pin metadata" begin
    raw = Dict{String, Any}(
        "geos_global_mass_pin_enabled" => true,
        "mass_fix_target_ps_dry_pa" => 98_325.0,
        "unrelated" => "drop",
    )
    metadata = AttachScript.copy_source_metadata!(Dict{String, Any}(), raw)
    @test metadata == Dict{String, Any}(
        "mass_fix_enabled" => true,
        "mass_fix_target_ps_dry_pa" => 98_325.0,
    )

    stale = AttachScript.copy_source_metadata!(Dict{String, Any}(), Dict{String, Any}(
        "global_mass_pin_enabled" => false,
        "mass_fix_target_ps_dry_pa" => 98_325.0,
    ))
    @test stale == Dict{String, Any}("mass_fix_enabled" => false)
end

# The attachment copies m/am/bm/cm unchanged, so the source binary's write-time
# certificate (C6) still describes the attached file and is carried over.
function _source_certificate(nwindow, nlevel)
    FT = Float32
    G = PP.CubedSphereTargetGeometry
    acc = PP.FluxFidelityAccumulator()
    for win in 1:nwindow
        samples = [PP.FluxFidelitySample(f, k, FT(1e9 + f), FT(1e9 + f), FT(1.1e9 + f))
                   for f in (3, 7) for k in 1:nlevel]
        flux = PP.WindDerivedFluxEvidence{G,FT,String,typeof(samples)}(
            "vector", samples, 2, "window_mean", 450.0)
        omega = PP.NoSourceOmegaEvidence{G,FT,PP.SourceHasNoOmega}(PP.SourceHasNoOmega())
        PP.verify_flux_fidelity!(acc, PP.FluxFidelityEvidenceBundle(
            flux, omega, PP.ColumnClosureRecord(1.0e5, 0.25, 2.0e4, 0.5)), win)
    end
    return PP.flux_fidelity_header!(Dict{String, Any}("nwindow" => nwindow,
                                                      "nlevel" => nlevel), acc)
end

@testset "attachment carries the source flux certificate" begin
    certificate = _source_certificate(1, 1)
    delete!(certificate, "nwindow"); delete!(certificate, "nlevel")
    raw = merge(Dict{String, Any}("unrelated" => "drop", "horizontal_balance" => "column",
                                  "column_balance_weights" => "hybrid_mass"), certificate)
    metadata = AttachScript.copy_source_metadata!(Dict{String, Any}(), raw)
    @test metadata == merge(certificate, Dict{String, Any}(
        "horizontal_balance" => "column", "column_balance_weights" => "hybrid_mass"))
    @test metadata["flux_fidelity_status"] == "passed"
    @test metadata["column_closure_windows"] == 1
end

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

@testset "legacy TM5 convection moist-to-dry attachment" begin
    nlon, nlat, nlev = 2, 1, 4
    era5_level = nlev:-1:1                         # archive is surface first
    shape = (nlon, nlat, nlev)
    # Two closed TM5 drafts in top-first order, then store them file-order.
    top_first = (
        eu = [0.0, 0.2, 0.1, 0.0], du = [0.0, 0.0, 0.1, 0.2],
        ed = [0.0, 0.03, 0.02, 0.0], dd = [0.0, 0.0, 0.02, 0.03],
    )
    file_field(v) = repeat(reshape(reverse(v), 1, 1, nlev), nlon, nlat, 1)
    slot = map(file_field, top_first)
    out = map(x -> similar(x), slot)
    qcol = [0.001, 0.004, 0.010, 0.018]
    q = repeat(reshape(qcol, 1, 1, nlev), nlon, nlat, 1)
    columns = ntuple(_ -> zeros(nlev), 5)

    AttachScript.file_fields_to_basis!(out, slot, DryBasis(), q, era5_level, columns)
    expected = map(copy, top_first)
    PP.tm5_convection_to_basis!(DryBasis(), expected..., qcol)
    for (actual, wanted) in zip(out, expected)
        @test vec(actual[1, 1, :]) == reverse(wanted)
        @test actual[2, 1, :] == actual[1, 1, :]
    end
    @test sum(out.eu) < sum(slot.eu)              # the unfixed attach route fails here
    @test sum(out.eu) ≈ sum(out.du) rtol = 1e-14
    @test sum(out.ed) ≈ sum(out.dd) rtol = 1e-14
end

@testset "legacy TM5 attachment records its converted basis" begin
    mktempdir() do dir
        Nc, np, Nz = 1, 6, 1
        panels3(value, nz = Nz) = ntuple(_ -> fill(Float32(value), Nc, Nc, nz), np)
        panels2(value) = ntuple(_ -> fill(Float32(value), Nc, Nc), np)
        surface = (pblh = panels2(900), ustar = panels2(0.3),
                   hflux = panels2(20), t2m = panels2(285))
        tm5 = (entu = panels3(0.01), detu = panels3(0.01),
               entd = panels3(0.002), detd = panels3(0.002))
        window = (
            m = panels3(1e15),
            am = ntuple(_ -> zeros(Float32, Nc + 1, Nc, Nz), np),
            bm = ntuple(_ -> zeros(Float32, Nc, Nc + 1, Nz), np),
            cm = panels3(0, Nz + 1), ps = panels2(95_000), dm = panels3(0),
            surface, tm5_fields = tm5, dkg = panels3(0),
        )
        header = (
            nwindow = 1, planet_radius_m = AtmosTransport.Parameters.EARTH_RADIUS,
            dt_met_seconds = 3600.0, half_dt_seconds = 1800.0,
            source_flux_sampling = :window_start_endpoint,
            air_mass_sampling = :window_start_endpoint,
            flux_sampling = :window_constant, flux_kind = :substep_mass_amount,
            mass_basis = :dry,
            surface_pressure_basis = DrySurfacePressure(),
        )
        geometry = (
            panel_convention = "gnomonic", definition = "equiangular_gnomonic",
            coordinate_law = "equiangular_gnomonic", center_law = "angular_midpoint",
            longitude_offset_deg = 0.0,
        )
        metadata = Dict{String, Any}(
            "convection_attachment" => "legacy_catrine_1deg_3hour_tm5_to_cs_v4")
        certificate = _source_certificate(1, Nz)
        AttachScript.copy_source_metadata!(metadata, certificate)
        path = joinpath(dir, "attached.bin")
        writer = AttachScript.open_attachment_writer(
            path, Nc, np, Nz, header, geometry,
            HybridSigmaPressure(Float32[0, 2000], Float32[0, 1]), metadata)
        MD.write_streaming_cs_window!(writer, window, Nc, np)
        MD.close_streaming_transport_binary!(writer)

        @test !haskey(metadata, MD.TM5_CONVECTION_MASS_BASIS_KEY)
        reader = MD.TransportBinaryReader(path)
        @test MD.tm5_convection_mass_basis(reader) isa DryBasis
        @test reader.header.raw_header[MD.TM5_CONVECTION_MASS_BASIS_KEY] == "dry"
        @test MD.surface_pressure_basis(reader) isa DrySurfacePressure
        # The source certificate reaches the attached file and parses as typed.
        @test reader.header.flux_fidelity.status isa MD.FidelityPassed
        @test reader.header.flux_fidelity.kind isa MD.WindRederivedFidelity
        @test reader.header.flux_fidelity.sample_count == 2
        @test reader.header.omega_fidelity.status isa MD.FidelityNotApplicable
        for (key, value) in certificate
            key in ("nwindow", "nlevel") && continue
            @test reader.header.raw_header[key] == value
        end
        close(reader)
    end
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
