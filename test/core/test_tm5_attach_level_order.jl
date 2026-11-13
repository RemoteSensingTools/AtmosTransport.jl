# The legacy CATRINE TM5 convection files store ERA5 L137 levels surface
# first. The attach script must map each file level to its ERA5 level before
# applying the binary's top-first merge_map; summing them in file order put
# the convection upside down (entrainment near the model top).
using Test, Dates, NCDatasets, Random, AtmosTransport

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
            payload_sections = [:m, :am, :bm, :cm, :ps, :dm, :pblh, :ustar, :pbl_hflux, :t2m, :dkg],
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

# A legacy-format convection file (ERA5 L137, surface first, eight slots).
# `fields(name, slot)` returns the `(360, 180, 137)` file-order values of
# variable `name`; without it the variables are left unwritten.
function write_convection_file(path; lon = collect(-179.5:1.0:179.5),
                               start = DateTime(2014, 1, 1), fields = nothing)
    vc = PP.load_hybrid_coefficients(AttachScript.ERA5_L137_COEFFICIENTS)
    A, B = Float64.(vc.A), Float64.(vc.B)
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
            var = defVar(ds, v, Float32, ("lon", "lat", "lev", "time");
                         deflatelevel = 1, chunksizes = [360, 180, 1, 1])
            fields === nothing && continue
            for slot in 1:8
                var[:, :, :, slot] = fields(v, slot)
            end
        end
    end
    return path
end

@testset "legacy TM5 convection file checks" begin
    mktempdir() do dir
        binary = "/x/era5_n320_transport_20140101_float32.bin"
        check(path; input = binary) = NCDataset(ds -> AttachScript.check_convection_file(ds, path, input), path)
        good = write_convection_file(joinpath(dir, "convec_20140101_00p03.nc"))
        @test check(good) == 137:-1:1
        @test_throws ErrorException check(good; input = "/x/era5_n320_transport_20140102_float32.bin")
        # Same file name in separate folders, so only the tested property differs.
        variant(name; kwargs...) = write_convection_file(joinpath(mkpath(joinpath(dir, name)),
                                                                  "convec_20140101_00p03.nc"); kwargs...)
        @test_throws ErrorException check(variant("lon_0_360"; lon = collect(0.5:1.0:359.5)))
        @test_throws ErrorException check(variant("late_slots"; start = DateTime(2014, 1, 1, 1)))
    end
end

# Pressure-overlap mode: ERA5 L137 convection on a binary with other hybrid
# levels (for example MERRA-2 L72). The fields are layer totals, so the
# column-wise remap sums them in proportion to the pressure overlap.
@testset "pressure-overlap vertical mapping" begin
    remap!(dest, src, p_src, p_dst) =
        AttachScript.remap_layers_by_pressure_overlap!(dest, src, p_src, p_dst)

    # Identity: coinciding levels copy the column exactly.
    p = [0.0, 10.0, 100.0, 1_000.0, 50_000.0, 100_000.0]
    src = ([0.0, 0.1, 0.5, 2.0, 1.0], [0.3, 0.0, 0.0, 1.5, 7.0])
    dest = (zeros(5), zeros(5))
    remap!(dest, src, p, p)
    @test dest == src

    # Analytic case: one source layer split 1:3 by a target interface, and two
    # source layers merged into one target layer.
    dest = (zeros(2),)
    remap!(dest, ([4.0, 1.0, 2.0],), [0.0, 400.0, 700.0, 1_000.0], [0.0, 100.0, 1_000.0])
    @test dest[1] ≈ [1.0, 3.0 + 1.0 + 2.0]

    # A field uniform per unit pressure on the source levels stays uniform on
    # the target levels: value = c Δp on both sides.
    c = 2.5e-6
    p_src = collect(range(0.0, 95_000.0; length = 31))
    p_dst = [0.0; sort(rand(MersenneTwister(1), 12) .* 95_000.0); 95_000.0]
    dest = (zeros(13),)
    remap!(dest, (c .* diff(p_src),), p_src, p_dst)
    @test dest[1] ≈ c .* diff(p_dst) rtol = 1e-12

    # Source pressure above the target top goes to the top layer, below the
    # target bottom to the bottom layer; the column sum is kept.
    dest = (zeros(2),)
    remap!(dest, ([1.0, 2.0, 4.0],), [0.0, 2.0, 50.0, 120.0], [1.0, 60.0, 100.0])
    @test dest[1][1] ≈ 1.0 + 2.0 + 4.0 * 10 / 70
    @test dest[1][2] ≈ 4.0 * 60 / 70
    @test sum(dest[1]) ≈ 7.0

    # Level order: interfaces must run from the top down.
    @test_throws ArgumentError remap!((zeros(2),), ([1.0, 2.0],), [100.0, 10.0, 0.0], [0.0, 50.0, 100.0])
    @test_throws DimensionMismatch remap!((zeros(3),), ([1.0, 2.0],), [0.0, 10.0, 100.0], [0.0, 50.0, 100.0])

    # ERA5 L137 onto a coarse hybrid target (50 hPa pure-pressure top part,
    # sigma below) over varying surface pressure: column sums are preserved,
    # the surface layer lands at the bottom, and the updraft mass flux, the
    # cumulative entrainment minus detrainment from the ground, stays the
    # linear interpolation of the source flux.
    vc = PP.load_hybrid_coefficients(AttachScript.ERA5_L137_COEFFICIENTS)
    A_src, B_src = Float64.(vc.A), Float64.(vc.B)
    A_dst = [1.0, 500.0, 2_000.0, 5_000.0, 4_000.0, 1_500.0, 0.0]
    B_dst = [0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 1.0]
    rng = MersenneTwister(7)
    ns, nd = length(A_src) - 1, length(A_dst) - 1
    p_src, p_dst = zeros(ns + 1), zeros(nd + 1)
    for ps in (50_000.0, 85_000.0, 101_325.0, 104_000.0)
        AttachScript.hybrid_interfaces!(p_src, A_src, B_src, ps)
        AttachScript.hybrid_interfaces!(p_dst, A_dst, B_dst, ps)
        @test p_src[end] == p_dst[end] == ps
        entu = zeros(ns); entu[end-20:end] .= rand(rng, 21)
        detu = zeros(ns); detu[60:100] .= rand(rng, 41)
        detu .*= sum(entu) / sum(detu)
        fields = (entu, detu, rand(rng, ns), rand(rng, ns))
        dest = ntuple(_ -> zeros(nd), 4)
        AttachScript.remap_layers_by_pressure_overlap!(dest, fields, p_src, p_dst)
        for (d, s) in zip(dest, fields)
            @test sum(d) ≈ sum(s) rtol = 1e-13
            @test all(>=(0), d)
        end
        surface = zeros(ns); surface[end] = 1.0
        bottom = (zeros(nd),)
        AttachScript.remap_layers_by_pressure_overlap!(bottom, (surface,), p_src, p_dst)
        @test bottom[1][end] ≈ 1.0 && sum(bottom[1][1:end-1]) == 0
        # Updraft flux at each target interface = the source flux at that pressure.
        flux_src = vcat(reverse(cumsum(reverse(entu .- detu))), 0.0)
        flux_dst = vcat(reverse(cumsum(reverse(dest[1] .- dest[2]))), 0.0)
        for k in 2:nd
            s = searchsortedlast(p_src, p_dst[k])
            w = (p_dst[k] - p_src[s]) / (p_src[s + 1] - p_src[s])
            @test flux_dst[k] ≈ (1 - w) * flux_src[s] + w * flux_src[s + 1] atol = 1e-12
        end
    end
end

@testset "pressure-overlap column surface pressure" begin
    A = [0.0, 5_000.0, 0.0]; B = [0.0, 0.3, 1.0]
    @test AttachScript.total_surface_pressure(TotalSurfacePressure(), 98_000f0, A, B, nothing) == 98_000.0
    qv = [0.001, 0.02]
    ps_dry = (1 - qv[1]) * (5_000 + 0.3 * 100_000) + (1 - qv[2]) * (-5_000 + 0.7 * 100_000)
    @test AttachScript.total_surface_pressure(DrySurfacePressure(), ps_dry, A, B, qv) ≈ 100_000.0
end

@testset "archive slot and humidity hour of a transport window" begin
    @test [AttachScript.convection_slot(w, 3600) for w in 1:24] == repeat(1:8; inner = 3)
    @test [AttachScript.convection_slot(w, 10_800) for w in 1:8] == 1:8
    @test [AttachScript.convection_slot(w, 1800) for w in (1, 6, 7, 48)] == [1, 1, 2, 8]
    @test_throws ArgumentError AttachScript.convection_slot(2, 7200)    # 2-4 h straddles 3 h
    @test_throws ArgumentError AttachScript.convection_slot(9, 10_800)  # after the day
    @test [AttachScript.humidity_hour(w, 3600) for w in (1, 24)] == [0, 23]
    @test AttachScript.humidity_hour(3, 10_800) == 6
    @test_throws ArgumentError AttachScript.humidity_hour(2, 1800)
end

@testset "attachment keeps GEOS physics sections next to TM5" begin
    mktempdir() do dir
        Nc, np, Nz = 2, 6, 3
        rng = MersenneTwister(3)
        panels3(nz = Nz) = ntuple(_ -> rand(rng, Float32, Nc, Nc, nz), np)
        panels2() = ntuple(_ -> rand(rng, Float32, Nc, Nc), np)
        window = (
            m = panels3(), am = ntuple(_ -> rand(rng, Float32, Nc + 1, Nc, Nz), np),
            bm = ntuple(_ -> rand(rng, Float32, Nc, Nc + 1, Nz), np),
            cm = panels3(Nz + 1), ps = panels2(), dm = panels3(),
            surface = (pblh = panels2(), ustar = panels2(), hflux = panels2(),
                       t2m = panels2(), eflux = panels2()),
            cmfmc = panels3(Nz + 1), dtrain = panels3(),
            vdiff = (u = panels3(), v = panels3(), t = panels3(), qv = panels3()),
            cmfmc_cloud_base = panels2(),
            tm5_fields = (entu = panels3(), detu = panels3(), entd = panels3(), detd = panels3()),
        )
        sections = [:m, :am, :bm, :cm, :ps, :dm, :pblh, :ustar, :pbl_hflux, :t2m,
                    :cmfmc, :dtrain, :vdiff_u, :vdiff_v, :vdiff_t, :vdiff_qv,
                    :pbl_eflux, :cmfmc_cloud_base]
        header = (
            payload_sections = sections, nwindow = 1,
            planet_radius_m = AtmosTransport.Parameters.EARTH_RADIUS,
            dt_met_seconds = 3600.0, half_dt_seconds = 1800.0,
            source_flux_sampling = :window_start_endpoint,
            air_mass_sampling = :window_start_endpoint,
            flux_sampling = :window_constant, flux_kind = :substep_mass_amount,
            mass_basis = :dry, surface_pressure_basis = DrySurfacePressure(),
        )
        geometry = (panel_convention = "geos_native", definition = "gmao_equal_distance",
                    coordinate_law = "gmao_equal_distance_gnomonic",
                    center_law = "four_corner_normalized", longitude_offset_deg = -10.0)
        path = joinpath(dir, "foreign.bin")
        host = Dict{String, Any}("source_type" => "merra2_native_latlon", "merra2_archive" => "x",
                                 "git_commit" => "abc", "git_dirty" => true, "unrelated" => 1,
                                 "include_tm5conv" => false, "nlevel" => 99)
        mapping = AttachScript.PressureOverlapLevels([0.0, 1.0], [0.0, 1.0])
        metadata = AttachScript.mapping_metadata(mapping, host, header, DrySurfacePressure(), DryBasis())
        @test metadata["host_git_commit"] == "abc" && metadata["host_git_dirty"] == true
        @test !haskey(metadata, "git_commit")
        writer = AttachScript.open_attachment_writer(
            path, Nc, np, Nz, header, geometry,
            HybridSigmaPressure(Float32[1, 2000, 0, 0], Float32[0, 0, 0.5, 1]), metadata)
        AttachScript.add_host_header_keys!(writer, mapping, host)
        MD.write_streaming_cs_window!(writer, window, Nc, np)
        MD.close_streaming_transport_binary!(writer)

        reader = MD.TransportBinaryReader(path)
        @test reader.header.payload_sections ==
              [sections[1:12]; [:entu, :detu, :entd, :detd]; sections[13:end]]
        @test MD.has_tm5_convection(reader) && MD.has_cmfmc(reader)
        @test MD.tm5_convection_mass_basis(reader) isa DryBasis
        raw = reader.header.raw_header
        @test raw["convection_host_meteorology"] == "merra2_native_latlon"
        # Host keys the output does not set are kept; the output's own win.
        @test raw["merra2_archive"] == "x" && raw["unrelated"] == 1
        @test raw["include_tm5conv"] == true && raw["nlevel"] == Nz && raw["git_commit"] != "abc"
        back = MD.load_window!(reader, 1)
        @test back.cmfmc == window.cmfmc && back.dtrain == window.dtrain
        @test back.vdiff.qv == window.vdiff.qv && back.surface.eflux == window.surface.eflux
        @test back.cmfmc_cloud_base == window.cmfmc_cloud_base
        @test back.tm5_fields.entd == window.tm5_fields.entd && back.m == window.m
        close(reader)

        # An input with part of a section group is rejected.
        @test_throws ErrorException AttachScript.open_attachment_writer(
            joinpath(dir, "partial.bin"), Nc, np, Nz,
            merge(header, (; payload_sections = [sections[1:12]; :vdiff_u])), geometry,
            HybridSigmaPressure(Float32[1, 2000, 0, 0], Float32[0, 0, 0.5, 1]), metadata)
    end
end

@testset "pressure-overlap zero-thickness layers and labels" begin
    # A source layer of zero thickness goes whole to the target layer around it.
    dest = (zeros(3),)
    AttachScript.remap_layers_by_pressure_overlap!(
        dest, ([1.0, 2.0, 4.0],), [0.0, 100.0, 100.0, 300.0], [0.0, 50.0, 200.0, 300.0])
    @test dest[1] ≈ [0.5, 0.5 + 2.0 + 2.0, 2.0]
    err = try
        AttachScript.remap_layers_by_pressure_overlap!(
            (zeros(1),), ([1.0, 1.0],), [0.0, 20.0, 10.0], [0.0, 30.0]; column = (3, 1, 2))
    catch e
        e
    end
    @test err isa ArgumentError && occursin("(3, 1, 2)", err.msg)

    # Humidity hours exist only for the dry-air conversion.
    @test AttachScript.humidity_hours(MoistBasis(), 48, 1800.0) === nothing
    @test_throws ArgumentError AttachScript.humidity_hours(DryBasis(), 48, 1800.0)
    @test AttachScript.humidity_hours(DryBasis(), 8, 10_800.0) == 0:3:21
    @test AttachScript.temporal_mapping_label(3600.0) ==
          "three-hour slots centred at 01:30, 04:30, ... UTC, each held for its three hourly transport windows"
    @test occursin("10800.0-second", AttachScript.temporal_mapping_label(10_800.0))

    # The host is named by source_type, else preprocessor, else the attachment stops.
    @test AttachScript.host_label(Dict{String, Any}("source_type" => "a", "preprocessor" => "b")) == "a"
    @test AttachScript.host_label(Dict{String, Any}("preprocessor" => "geos_native_cs")) == "geos_native_cs"
    @test_throws ErrorException AttachScript.host_label(Dict{String, Any}("source_type" => ""))
    mapping = AttachScript.PressureOverlapLevels([0.0, 1.0], [0.0, 1.0])
    header = (dt_met_seconds = 3600.0,)
    moist = AttachScript.mapping_metadata(mapping, Dict{String, Any}("preprocessor" => "geos_native_cs"),
                                          header, TotalSurfacePressure(), MoistBasis())
    @test moist["convection_host_meteorology"] == "geos_native_cs"
    @test !occursin("dry air", moist["experimental_caveat"])
    @test occursin("different meteorology", moist["experimental_caveat"])
    @test !occursin("humidity", moist["convection_temporal_mapping"])
    dry = AttachScript.mapping_metadata(mapping, Dict{String, Any}("source_type" => "era5_n320_native_grib"),
                                        header, DrySurfacePressure(), DryBasis())
    @test occursin("dry air", dry["experimental_caveat"])
    @test !occursin("different meteorology", dry["experimental_caveat"])
    @test occursin("vdiff_qv", dry["convection_vertical_surface_pressure"])
end

# The panel driver of the pressure-overlap mapping, checked cell by cell
# against the column functions: values differ in every cell and level, and
# surface pressure and humidity differ in every column.
@testset "pressure-overlap panels follow the column functions" begin
    Nc, np = 2, 6
    A_src = [0.0, 3_000.0, 9_000.0, 12_000.0, 6_000.0, 0.0]
    B_src = [0.0, 0.0, 0.05, 0.25, 0.6, 1.0]
    A_dst = [1.0, 5_000.0, 8_000.0, 0.0]
    B_dst = [0.0, 0.0, 0.3, 1.0]
    nsrc, ndst = length(A_src) - 1, length(A_dst) - 1
    ncell = np * Nc * Nc
    flats = ntuple(f -> [1000.0 * f + 10.0 * c + s for c in 1:ncell, s in 1:nsrc], 4)
    ps = ntuple(p -> Float32[80_000 + 2_000 * p + 300 * i + 70 * j for i in 1:Nc, j in 1:Nc], np)
    qv = ntuple(p -> Float32[0.001 * p + 0.0005 * i + 0.0002 * j + 0.003 * k
                             for i in 1:Nc, j in 1:Nc, k in 1:ndst], np)
    new_tm5() = NamedTuple{(:entu, :detu, :entd, :detd)}(
        ntuple(_ -> ntuple(_ -> zeros(Float32, Nc, Nc, ndst), np), 4))
    function expected(basis, humidity, p, i, j)
        column_qv = humidity === nothing ? nothing : humidity[p][i, j, :]
        ps_total = AttachScript.total_surface_pressure(basis, ps[p][i, j], A_dst, B_dst, column_qv)
        p_src = AttachScript.hybrid_interfaces!(zeros(nsrc + 1), A_src, B_src, ps_total)
        p_dst = AttachScript.hybrid_interfaces!(zeros(ndst + 1), A_dst, B_dst, ps_total)
        cell = (p - 1) * Nc * Nc + (j - 1) * Nc + i
        src = ntuple(f -> flats[f][cell, :], 4)
        dest = ntuple(_ -> zeros(ndst), 4)
        return AttachScript.remap_layers_by_pressure_overlap!(dest, src, p_src, p_dst)
    end
    for (basis, humidity) in ((DrySurfacePressure(), qv), (TotalSurfacePressure(), nothing))
        tm5 = AttachScript.pressure_overlap_to_panels!(new_tm5(), flats, ps, humidity, basis,
                                                       A_src, B_src, A_dst, B_dst, Nc)
        mismatches = 0
        for p in 1:np, j in 1:Nc, i in 1:Nc
            want = expected(basis, humidity, p, i, j)
            for (f, name) in enumerate((:entu, :detu, :entd, :detd))
                mismatches += getfield(tm5, name)[p][i, j, :] != Float32.(want[f])
            end
        end
        @test mismatches == 0
    end
    # The dry surface pressure is not the total one: humidity matters.
    dry = AttachScript.pressure_overlap_to_panels!(new_tm5(), flats, ps, qv, DrySurfacePressure(),
                                                   A_src, B_src, A_dst, B_dst, Nc)
    total = AttachScript.pressure_overlap_to_panels!(new_tm5(), flats, ps, nothing, TotalSurfacePressure(),
                                                     A_src, B_src, A_dst, B_dst, Nc)
    @test dry.entu[1] != total.entu[1]

    # A bad surface pressure stops the remap and names its column.
    bad = map(copy, ps); bad[3][2, 1] = NaN32
    err = try
        AttachScript.pressure_overlap_to_panels!(new_tm5(), flats, bad, nothing, TotalSurfacePressure(),
                                                 A_src, B_src, A_dst, B_dst, Nc)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("panel 3, column (2, 1)", err.msg)
    err = try
        AttachScript.pressure_overlap_to_panels!(new_tm5(), flats, bad, qv, DrySurfacePressure(),
                                                 A_src, B_src, A_dst, B_dst, Nc)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("panel 3, column (2, 1)", err.msg)
    # Sizes are checked before any column is touched.
    @test_throws DimensionMismatch AttachScript.pressure_overlap_to_panels!(
        new_tm5(), map(x -> x[:, 1:end-1], flats), ps, qv, DrySurfacePressure(),
        A_src, B_src, A_dst, B_dst, Nc)
    @test_throws DimensionMismatch AttachScript.pressure_overlap_to_panels!(
        new_tm5(), flats, ps, map(x -> x[:, :, 1:end-1], qv), DrySurfacePressure(),
        A_src, B_src, A_dst, B_dst, Nc)
end

@testset "native-level regrid keeps the file level order" begin
    mktempdir() do dir
        cube = CubedSphereMesh(; FT = Float64, Nc = 2, Hp = 0)
        latlon = LatLonMesh(; FT = Float64, size = (8, 4), longitude = (-180, 180),
                            latitude = (-90, 90), radius = cube.radius)
        regridder = AtmosTransport.Regridding.build_regridder(latlon, cube; normalize = false,
                                                              cache_dir = dir)
        nlev = 5
        native = zeros(8, 4, nlev); native[:, :, 1] .= 3.0     # file level 1 = surface
        src_flat = zeros(32, nlev)
        dst_flat = zeros(6 * 4, nlev)
        AttachScript.native_tm5_to_cube!(dst_flat, src_flat, native, nlev:-1:1, regridder)
        @test all(x -> isapprox(x, 3.0; rtol = 1e-10), dst_flat[:, nlev])   # bottom, top first
        @test all(iszero, dst_flat[:, 1:nlev-1])
        @test_throws DimensionMismatch AttachScript.native_tm5_to_cube!(
            dst_flat, zeros(32, nlev - 1), native, 1:nlev - 1, regridder)
    end
end

# End to end: `main` on a toy moist binary (total surface pressure, so no ERA5
# humidity is read) with three-hour windows and the GEOS physics sections.
@testset "pressure-overlap attachment end to end" begin
    mktempdir() do dir
        Nc, np, Nz, nwindow = 2, 6, 4, 8
        A = Float32[1, 2_000, 5_000, 3_000, 0]
        B = Float32[0, 0, 0, 0.5, 1]
        rng = MersenneTwister(11)
        panels3(nz = Nz) = ntuple(_ -> rand(rng, Float32, Nc, Nc, nz), np)
        panels2() = ntuple(_ -> rand(rng, Float32, Nc, Nc), np)
        ps_of(win) = ntuple(p -> Float32[90_000 + 1_000 * p + 100 * i + 10 * j + win
                                         for i in 1:Nc, j in 1:Nc], np)
        windows = [(m = panels3(), am = ntuple(_ -> rand(rng, Float32, Nc + 1, Nc, Nz), np),
                    bm = ntuple(_ -> rand(rng, Float32, Nc, Nc + 1, Nz), np),
                    cm = panels3(Nz + 1), ps = ps_of(win), dm = panels3(),
                    surface = (pblh = panels2(), ustar = panels2(), hflux = panels2(),
                               t2m = panels2(), eflux = panels2()),
                    cmfmc = panels3(Nz + 1), dtrain = panels3(),
                    vdiff = (u = panels3(), v = panels3(), t = panels3(), qv = panels3()),
                    cmfmc_cloud_base = panels2()) for win in 1:nwindow]
        input = joinpath(dir, "toy_transport_20140101_float32.bin")
        writer = MD.open_streaming_cs_transport_binary(
            input, Nc, np, Nz, nwindow, HybridSigmaPressure(A, B);
            FT = Float32, planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
            dt_met_seconds = 10_800.0, steps_per_window = 2, include_flux_delta = true,
            mass_basis = :moist, surface_pressure_basis = TotalSurfacePressure(),
            include_surface = true, include_cmfmc = true, include_dtrain = true,
            include_gchp_vdiff = true, include_pbl_eflux = true, include_cmfmc_cloud_base = true,
            extra_header = Dict{String, Any}("source_type" => "toy_source", "toy_key" => "kept",
                                             "git_commit" => "hostsha"))
        foreach(w -> MD.write_streaming_cs_window!(writer, w, Nc, np), windows)
        MD.close_streaming_transport_binary!(writer)

        # Horizontally uniform, closed drafts, scaled by the slot number (top-first ERA5 levels).
        top_first(slot) = (eu = (137, 2f-3 * slot), du = (100, 2f-3 * slot),
                           ed = (110, 5f-4 * slot), dd = (136, 5f-4 * slot))
        function fields(name, slot)
            level, value = getfield(top_first(slot), Symbol(name))
            out = zeros(Float32, 360, 180, 137)
            out[:, :, 138 - level] .= value                  # file order: surface first
            return out
        end
        conv = write_convection_file(joinpath(dir, "convec_20140101_00p03.nc"); fields)
        output = joinpath(dir, "out", "toy_transport_20140101_float32.bin")
        cache = mkpath(joinpath(dir, "cache"))

        # Without a merge_map the default mode points at the pressure-overlap mode.
        err = try
            AttachScript.main([input, conv, output, "--cache-dir", cache])
        catch e
            e
        end
        @test err isa ErrorException && occursin("--vertical-mapping pressure_overlap", err.msg)

        AttachScript.main([input, conv, output, "--vertical-mapping", "pressure_overlap",
                           "--cache-dir", cache])
        rin = MD.TransportBinaryReader(input; FT = Float32)
        rout = MD.TransportBinaryReader(output; FT = Float32)
        @test rout.header.payload_sections ==
              [rin.header.payload_sections[1:12]; [:entu, :detu, :entd, :detd];
               rin.header.payload_sections[13:end]]
        same = all(MD.section_view(rin, win, s) == MD.section_view(rout, win, s)
                   for win in 1:nwindow for s in rin.header.payload_sections)
        @test same
        raw = rout.header.raw_header
        @test raw["convection_host_meteorology"] == "toy_source"
        @test raw["toy_key"] == "kept" && raw["host_git_commit"] == "hostsha"
        @test raw["tm5_convection_mass_basis"] == "moist"
        @test raw["convection_humidity_source"] == "none (moist binary)"
        @test !occursin("dry air", raw["experimental_caveat"])
        @test occursin("10800.0-second", raw["convection_temporal_mapping"])
        @test rout.header.A_ifc == rin.header.A_ifc && rout.header.B_ifc == rin.header.B_ifc
        @test all(k -> haskey(raw, k), keys(rin.header.raw_header))

        # The placed fields: each window takes its slot, each column its own ps.
        vc = PP.load_hybrid_coefficients(AttachScript.ERA5_L137_COEFFICIENTS)
        A_src, B_src = Float64.(vc.A), Float64.(vc.B)
        worst = 0.0
        for win in 1:nwindow
            tm5 = MD.load_window!(rout, win).tm5_fields
            values = top_first(win)
            for p in 1:np, j in 1:Nc, i in 1:Nc
                ps = Float64(windows[win].ps[p][i, j])
                p_src = AttachScript.hybrid_interfaces!(zeros(138), A_src, B_src, ps)
                p_dst = AttachScript.hybrid_interfaces!(zeros(Nz + 1), Float64.(A), Float64.(B), ps)
                for name in (:eu, :du, :ed, :dd)
                    level, value = getfield(values, name)
                    column = zeros(137); column[level] = value
                    want = AttachScript.remap_layers_by_pressure_overlap!((zeros(Nz),), (column,),
                                                                         p_src, p_dst)[1]
                    got = getfield(tm5, Dict(:eu => :entu, :du => :detu, :ed => :entd, :dd => :detd)[name])[p][i, j, :]
                    worst = max(worst, maximum(abs.(got .- want)) / value)
                end
            end
            @test tm5.entu[1][1, 1, Nz] ≈ 2f-3 * win rtol = 1e-5      # surface entrainment at the bottom
        end
        @test worst < 1e-5
        close(rin); close(rout)

        # Inputs the attachment cannot copy faithfully are rejected.
        input64 = joinpath(dir, "f64", "toy_transport_20140101_float32.bin")
        mkpath(dirname(input64))
        writer = MD.open_streaming_cs_transport_binary(
            input64, Nc, np, Nz, nwindow, HybridSigmaPressure(A, B);
            FT = Float64, planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
            dt_met_seconds = 10_800.0, steps_per_window = 2, include_flux_delta = true,
            mass_basis = :moist, surface_pressure_basis = TotalSurfacePressure())
        for w in windows
            MD.write_streaming_cs_window!(writer, map(x -> x isa Tuple ? map(Array{Float64}, x) : x,
                (m = w.m, am = w.am, bm = w.bm, cm = w.cm, ps = w.ps, dm = w.dm)), Nc, np)
        end
        MD.close_streaming_transport_binary!(writer)
        err = try
            AttachScript.main([input64, conv, joinpath(dir, "out64.bin"),
                               "--vertical-mapping", "pressure_overlap", "--cache-dir", cache])
        catch e
            e
        end
        @test err isa ErrorException && occursin("Float32 only", err.msg)
    end
end
