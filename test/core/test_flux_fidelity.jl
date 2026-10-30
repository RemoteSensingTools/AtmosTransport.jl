#!/usr/bin/env julia
# Write-time flux-fidelity certificates (C6): per-window verification, the
# aggregate header, and its read-side consistency checks.

using Test
using Random
using AtmosTransport
using JSON3

const Pre = AtmosTransport.Preprocessing
const MD = AtmosTransport.MetDrivers
const Grids = AtmosTransport.Grids
const CS = Pre.CubedSphereTargetGeometry

no_omega(::Type{FT}) where FT =
    Pre.NoSourceOmegaEvidence{CS,FT,Pre.SourceHasNoOmega}(Pre.SourceHasNoOmega())

function wind_evidence(::Type{FT}, faces, nlevel; error = 0.0,
                       time_sampling = "window_mean") where FT
    samples = Pre.FluxFidelitySample{FT}[
        Pre.FluxFidelitySample(f, k, FT(1e9 * f), FT(1e9 * f * (1 + error)), FT(1.1e9 * f))
        for f in faces for k in 1:nlevel]
    return Pre.WindDerivedFluxEvidence{CS,FT,String,typeof(samples)}(
        "vector", samples, length(faces), time_sampling, 450.0)
end

bundle(flux, omega = no_omega(Float32)) =
    Pre.FluxFidelityEvidenceBundle(flux, omega, Pre.NoColumnClosureRecord())

# The aggregate certificate of `nwindow` windows, as a header the reader parses.
function certificate(evidence_for_window, nwindow, nlevel)
    acc = Pre.FluxFidelityAccumulator()
    for win in 1:nwindow
        Pre.verify_flux_fidelity!(acc, evidence_for_window(win), win)
    end
    return Pre.flux_fidelity_header!(Dict{String,Any}("nwindow" => nwindow,
                                                      "nlevel" => nlevel), acc)
end

parse_flux(h; nwindow = h["nwindow"], nlevel = h["nlevel"]) =
    MD.BinaryFormat.parse_flux_fidelity_header(JSON3.read(JSON3.write(h)), nwindow, nlevel)
parse_omega(h) = MD.BinaryFormat.parse_omega_fidelity_header(JSON3.read(JSON3.write(h)))

# OMEGA evidence with `n` cells × `nz` interfaces; `candidate(src, k)` sets ω_cm.
function omega_evidence(candidate; n = 64, nz = 20, seed = 3)
    rng = Xoshiro(seed)
    samples = Pre.OmegaFidelitySample{Float64}[]
    for c in 1:n, k in 1:nz
        src = 0.1 * randn(rng) * (1 + k / nz)
        push!(samples, Pre.OmegaFidelitySample(src, candidate(src, k, rng),
                                               1000.0 * k / nz, 0.0))
    end
    return Pre.SourceOmegaEvidence{CS,Float64,typeof(samples)}(
        samples, "fixture", "instantaneous", "dry", Pre.MERRA2_WIND_OMEGA_BASELINE)
end

function omega_header(candidate; kwargs...)
    acc = Pre.FluxFidelityAccumulator()
    flux = Pre.NoFluxFidelityEvidence{CS,Float64,Pre.SourceHasNoWinds}(Pre.SourceHasNoWinds())
    Pre.verify_flux_fidelity!(acc, Pre.FluxFidelityEvidenceBundle(
        flux, omega_evidence(candidate; kwargs...), Pre.NoColumnClosureRecord()), 1)
    header = Dict{String,Any}()
    err = try
        Pre.flux_fidelity_header!(header, acc)
        nothing
    catch e
        e
    end
    return header, err
end

@testset "typed write-time flux fidelity" begin
    @testset "sample maps are deterministic and canonical" begin
        Nc = 4
        mesh = Grids.CubedSphereMesh(; Nc, FT=Float64)
        table = Pre.build_cs_global_face_table(
            Nc, Grids.panel_connectivity_for(mesh.convention))
        a = Pre.FidelitySampleMap(table; samples_per_window=32)
        b = Pre.FidelitySampleMap(table; samples_per_window=32)
        @test a.version == 1
        @test a.seed == b.seed
        @test a.faces == b.faces
        @test length(a.faces) == length(unique(a.faces)) == 32
        @test all(f -> 1 <= f <= table.nf, a.faces)
        # Structured lat-lon: ranked once, the same rule as before the cache.
        s = Pre.StructuredFidelitySampler(12, 7; samples_per_window = 20)
        nface = 13 * 7 + 12 * 8
        ranked = sort(collect(1:nface); by = f -> Pre._splitmix64(
            xor(Pre.FLUX_FIDELITY_SAMPLE_SEED, UInt64(f))))
        @test s.faces == sort(ranked[1:20])
        @test Pre.StructuredFidelitySampler(12, 7; samples_per_window = 20).faces == s.faces
        cells = Pre.OmegaFidelityCells(4; samples_per_window = 16)
        @test length(cells.cells) == 16 && issorted(cells.cells)
    end

    @testset "wind evidence passes; injected errors fail before write" begin
        for FT in (Float32, Float64)
            h = certificate(_ -> bundle(wind_evidence(FT, (7, 9), 2), no_omega(FT)), 3, 2)
            @test h["flux_fidelity_status"] == "passed"
            @test h["flux_fidelity_kind"] == "wind_rederived"
            @test h["flux_fidelity_sample_rule"] == "cs_stratified_faces_v1"
            @test h["flux_fidelity_sample_count"] == 3 * 2 * 2
            @test h["flux_fidelity_time_sampling"] == "window_mean"
            # Balanced = 1.1 × reference at every sample: 10 % flux-weighted correction.
            @test h["balance_correction_rms_relative"] ≈ 0.1 rtol = 1e-5
            @test h["balance_correction_max_level_rms_relative"] ≈ 0.1 rtol = 1e-5
            @test h["balance_correction_max_level"] in 1:2
            @test h["omega_fidelity_status"] == "not_applicable"
            @test h["omega_fidelity_reason"] == "source_has_no_omega"
            parsed = parse_flux(h)
            @test parsed.status isa MD.FidelityPassed
            @test parsed.kind isa MD.WindRederivedFidelity
            @test isempty(parsed.time_sampling_by_window)
            @test_throws ArgumentError Pre.verify_flux_fidelity!(
                Pre.FluxFidelityAccumulator(), bundle(wind_evidence(FT, (1,), 1; error = 0.01),
                                                      no_omega(FT)), 1)
        end
    end

    @testset "write-time consistency" begin
        e = wind_evidence(Float32, (7, 9), 2)
        # Coverage: windows and levels of the header.
        @test_throws ArgumentError certificate(_ -> bundle(e), 2, 3)
        acc = Pre.FluxFidelityAccumulator()
        Pre.verify_flux_fidelity!(acc, bundle(e), 1)
        @test_throws ArgumentError Pre.flux_fidelity_header!(
            Dict{String,Any}("nwindow" => 2, "nlevel" => 2), acc)
        # Empty evidence and changed sample maps.
        empty = Pre.WindDerivedFluxEvidence{CS,Float32,String,Vector{Pre.FluxFidelitySample{Float32}}}(
            "vector", Pre.FluxFidelitySample{Float32}[], 2, "window_mean", 450.0)
        @test_throws ArgumentError Pre.verify_flux_fidelity!(Pre.FluxFidelityAccumulator(),
                                                             bundle(empty), 1)
        acc = Pre.FluxFidelityAccumulator()
        Pre.verify_flux_fidelity!(acc, bundle(e), 1)
        @test_throws ArgumentError Pre.verify_flux_fidelity!(
            acc, bundle(wind_evidence(Float32, (7, 9, 11), 2)), 2)
        # Mixed kinds of evidence.
        acc = Pre.FluxFidelityAccumulator()
        Pre.verify_flux_fidelity!(acc, bundle(e), 1)
        Pre.verify_flux_fidelity!(acc, Pre.default_fidelity_evidence(CS, Float32), 2)
        @test_throws ArgumentError Pre.flux_fidelity_header!(Dict{String,Any}(), acc)
    end

    @testset "effective time sampling: boundary fallback is recorded" begin
        h = certificate(win -> bundle(wind_evidence(Float32, (7, 9), 2; time_sampling =
            win == 3 ? "window_start_boundary_fallback" : "window_mean")), 3, 2)
        @test h["flux_fidelity_time_sampling"] == "mixed"
        @test h["flux_fidelity_time_sampling_by_window"] ==
              ["window_mean", "window_mean", "window_start_boundary_fallback"]
        @test parse_flux(h).time_sampling_by_window[3] == "window_start_boundary_fallback"
        bad = copy(h); bad["flux_fidelity_time_sampling_by_window"] = ["window_mean"]
        @test_throws ArgumentError parse_flux(bad)
    end

    @testset "native fluxes are not applicable, with the balance correction" begin
        FT = Float32
        # Every sample is corrected by 2e8 kg; one level-1 face had almost no
        # pre-balance flux (1 kg). A per-sample ratio would report 2e8 for it;
        # the flux-weighted metric reports the correction relative to the flux.
        pre(f, k) = f == 5 && k == 1 ? FT(1) : FT(2e9)
        samples = [Pre.FluxBalanceSample(f, k, pre(f, k), pre(f, k) + FT(2e8))
                   for f in (4, 5) for k in 1:3]
        native = Pre.NativeFluxEvidence{CS,FT,typeof(samples)}(
            samples, 2, "geos_native_scaling_vertical_map", 3600.0)
        h = certificate(_ -> Pre.FluxFidelityEvidenceBundle(
            native, Pre.NoSourceOmegaEvidence{CS,FT,Pre.SourceNotOpened}(Pre.SourceNotOpened()),
            Pre.NoColumnClosureRecord()), 2, 3)
        @test h["flux_fidelity_status"] == "not_applicable"
        @test h["flux_fidelity_kind"] == "native_passthrough"
        @test h["flux_fidelity_reason"] == "native_fluxes_not_rederivable"
        @test h["balance_correction_max_abs_kg"] ≈ 2e8 rtol = 1e-6
        @test h["balance_correction_rms_relative"] ≈ sqrt(6 / 5) * 0.1 rtol = 1e-6
        @test h["balance_correction_max_level_rms_relative"] ≈ sqrt(2) * 0.1 rtol = 1e-6
        @test h["balance_correction_max_level"] == 1
        @test !haskey(h, "flux_fidelity_max_abs_kg")             # nothing was compared
        @test h["omega_fidelity_status"] == "unavailable"
        parsed = parse_flux(h)
        @test parsed.status isa MD.FidelityNotApplicable
        @test parsed.kind isa MD.NativePassthroughFidelity
        @test parsed.reason.code == "native_fluxes_not_rederivable"
        # The native capture records the pre-balance values only.
        Nc, Nz = 3, 2
        table = Pre.build_cs_global_face_table(
            Nc, Grids.panel_connectivity_for(Grids.CubedSphereMesh(; Nc, FT=Float64).convention))
        am = ntuple(p -> fill(Float32(p), Nc + 1, Nc, Nz), 6)
        bm = ntuple(p -> fill(Float32(-p), Nc, Nc + 1, Nz), 6)
        smap = Pre.FidelitySampleMap(table; samples_per_window = 8)
        captured = Pre.capture_native_flux_samples(am, bm, table; sample_map = smap)
        @test captured isa Pre.CapturedNativeFluxSamples
        @test length(captured.native) == 8 * Nz
        @test Pre.scale_captured_flux_samples(captured, 2).native == 2 .* captured.native
    end

    @testset "line-integral samples rebuild the midpoint quadrature" begin
        Nc, Nz = 3, 2
        source = Grids.LatLonMesh(; Nx=12, Ny=6, FT=Float64)
        target = Grids.CubedSphereMesh(; Nc, FT=Float64)
        table = Pre.build_cs_global_face_table(
            Nc, Grids.panel_connectivity_for(target.convention))
        method = Pre.LineIntegralFaceFluxes(source, target, table;
                                            samples_per_face=4)
        nsrc = Grids.ncells(source)
        u = reshape(range(-10.0, 10.0; length=nsrc * Nz), nsrc, Nz)
        v = reshape(range(5.0, -5.0; length=nsrc * Nz), nsrc, Nz)
        dp = reshape(range(100.0, 1000.0; length=nsrc * Nz), nsrc, Nz)
        am = ntuple(_ -> zeros(Float64, Nc + 1, Nc, Nz), 6)
        bm = ntuple(_ -> zeros(Float64, Nc, Nc + 1, Nz), 6)
        Pre.line_integral_face_fluxes!(am, bm, method, u, v, dp,
                                       9.80665, 60.0, Nz)
        smap = Pre.FidelitySampleMap(table; samples_per_window=12)
        # The certificate must not depend on the production sparse operator
        # after the reference payload has been formed.
        L = copy(method.L)
        method.L.nzval .= NaN
        captured = Pre.capture_cs_wind_flux_samples(
            method, am, bm, u, v, dp, table, 9.80665, 60.0, Nc, Nz;
            sample_map=smap)
        @test captured.reference ≈ captured.candidate rtol=2e-14
        evidence = Pre.wind_derived_evidence(
            CS, captured, am, bm, table;
            method="line_integral", samples_per_window=length(smap.faces),
            time_sampling="window_start_endpoint", amount_seconds=60.0)
        @test isnothing(Pre.verify_flux_fidelity!(
            Pre.FluxFidelityAccumulator(), evidence, 1))

        # Moist thickness evaluated lazily at the stencil cells equals the
        # production moist line integral (no source-grid dp array).
        method.L.nzval .= L.nzval
        A, B = [0.0, 3000.0, 1000.0], [0.0, 0.2, 1.0]
        ps = collect(range(9.5e4, 1.02e5; length = nsrc))
        Pre.line_integral_face_fluxes!(am, bm, method, u, v, ps, A, B, 9.80665, 60.0, Nz)
        lazy = Pre.capture_cs_wind_flux_samples(
            method, am, bm, u, v, Pre.HybridLayerThickness(A, B, ps), table,
            9.80665, 60.0, Nc, Nz; sample_map = smap)
        @test lazy.reference ≈ lazy.candidate rtol = 2e-14
    end

    @testset "reduced-Gaussian samples synthesize only their lines" begin
        mesh = AtmosTransport.ReducedGaussianMesh(Float64[-60, -20, 20, 60], [8, 12, 12, 8])
        grid = Pre.ReducedGaussianTargetGeometry(
            mesh, "synthetic", 2, copy(mesh.nlon_per_ring), copy(mesh.latitudes),
            [collect(range(0.0, 360.0; length=n + 1))[1:end-1] for n in mesh.nlon_per_ring])
        T = 3
        work = Pre.allocate_reduced_transform_workspace(grid, T, 1)
        rng = Xoshiro(5)
        u_spec = complex.(randn(rng, T + 1, T + 1), randn(rng, T + 1, T + 1))
        v_spec = complex.(randn(rng, T + 1, T + 1), randn(rng, T + 1, T + 1))
        lnsp = log.(1e5 .+ 1e3 .* randn(rng, Grids.ncells(mesh)))
        cache = work.caches[1]
        full = zeros(Grids.nfaces(mesh))
        Pre.compute_reduced_horizontal_fluxes!(full, lnsp, u_spec, v_spec, T, 300.0, 0.2,
                                               grid, 450.0, cache)
        for sampler in (work.fidelity_sampler,
                        Pre.ReducedFidelitySampler(grid, work.face_left, work.face_right;
                                                   samples = 9, lines = 3))
            out = zeros(length(sampler.faces))
            Pre.sample_reduced_horizontal_fluxes!(out, sampler, lnsp, u_spec, v_spec, T,
                                                  300.0, 0.2, grid, 450.0, cache)
            @test out ≈ full[sampler.faces] rtol = 1e-12
            @test all(f -> work.face_left[f] > 0 && work.face_right[f] > 0, sampler.faces)
            @test length(sampler.line_slots) <= 7
        end
        small = Pre.ReducedFidelitySampler(grid, work.face_left, work.face_right;
                                           samples = 9, lines = 3)
        @test length(small.line_slots) == 3
        @test small.faces == Pre.ReducedFidelitySampler(
            grid, work.face_left, work.face_right; samples = 9, lines = 3).faces
        @test size(work.fidelity_native) == (length(work.fidelity_sampler.faces), 1)
    end

    @testset "OMEGA gate: NRMSE < 1, correlation > 0.5, 0.5 < slope < 1.5" begin
        good, err = omega_header((src, k, rng) -> 0.9src + 0.02randn(rng))
        @test err === nothing
        @test good["omega_fidelity_status"] == "passed"
        @test good["omega_fidelity_nrmse"] < 1
        @test good["omega_fidelity_nrmse_limit"] == AtmosTransport.Parameters.OMEGA_FIDELITY_NRMSE_LIMIT
        @test good["omega_fidelity_correlation_min"] == 0.5
        @test good["omega_fidelity_slope_min"] == 0.5
        @test good["omega_fidelity_slope_max"] == 1.5
        @test good["omega_fidelity_baseline_id"] == "merra2_wind_omega_v3"
        @test parse_omega(good).status isa MD.FidelityPassed
        # Zero, sign-flipped and level-flipped vertical transport fail.
        for (label, candidate) in (("zero", (src, k, rng) -> 0.0),
                                   ("sign flip", (src, k, rng) -> -src),
                                   ("level flip", (src, k, rng) -> 0.1randn(rng) * (2 - k / 20)))
            h, e = omega_header(candidate)
            @test e isa ArgumentError
            @test h["omega_fidelity_status"] == "failed"
            @test h["omega_fidelity_reason"] == "baseline_gate_failed"
            @test h["omega_fidelity_passed"] === false
            @test parse_omega(h).status isa MD.FidelityFailed
        end
        # A uniformly mis-scaled `cm` (an amount or time-convention error: 2n
        # against n substeps, the 8× `mass_flux_dt` error) keeps the
        # correlation and, when too small, NRMSE < 1; only the two-sided slope
        # bound rejects it. Source and noise tuned to the MERRA-2 C90 golden
        # (slope 0.865, NRMSE 0.47, r 0.89), through the production accumulator.
        rng = Xoshiro(1)
        src = 0.1 .* randn(rng, 20_000)
        noise = 0.0436 .* randn(rng, 20_000)
        function scaled_cm_header(scale)
            samples = [Pre.OmegaFidelitySample{Float64}(s, scale * (0.865s + e), 500.0, 0.0)
                       for (s, e) in zip(src, noise)]
            evidence = Pre.SourceOmegaEvidence{CS,Float64,typeof(samples)}(
                samples, "fixture", "instantaneous", "dry", Pre.MERRA2_WIND_OMEGA_BASELINE)
            acc = Pre.FluxFidelityAccumulator()
            Pre.verify_flux_fidelity!(acc, Pre.FluxFidelityEvidenceBundle(
                Pre.NoFluxFidelityEvidence{CS,Float64,Pre.SourceHasNoWinds}(Pre.SourceHasNoWinds()),
                evidence, Pre.NoColumnClosureRecord()), 1)
            header = Dict{String,Any}()
            err = try
                Pre.flux_fidelity_header!(header, acc)
                nothing
            catch e
                e
            end
            return header, err
        end
        unscaled, err = scaled_cm_header(1.0)
        @test err === nothing
        @test unscaled["omega_fidelity_slope"] ≈ 0.865 atol = 0.02
        @test unscaled["omega_fidelity_nrmse"] ≈ 0.47 atol = 0.03
        @test parse_omega(unscaled).status isa MD.FidelityPassed
        for scale in (2.0, 0.5, 1 / 8)
            h, e = scaled_cm_header(scale)
            @test e isa ArgumentError
            @test h["omega_fidelity_status"] == "failed"
            @test h["omega_fidelity_correlation"] > 0.5
            @test scale > 1 || h["omega_fidelity_nrmse"] < 1
            @test !(0.5 < h["omega_fidelity_slope"] < 1.5)
            @test parse_omega(h).status isa MD.FidelityFailed
        end
        zero_header, _ = omega_header((src, k, rng) -> 0.0)
        @test zero_header["omega_fidelity_nrmse"] ≈ 1.0
        @test zero_header["omega_fidelity_correlation"] == 0.0
        flip_header, _ = omega_header((src, k, rng) -> -src)
        @test flip_header["omega_fidelity_nrmse"] ≈ 2.0
        # No samples: rejected per window, never certified.
        empty = Pre.SourceOmegaEvidence{CS,Float64,Vector{Pre.OmegaFidelitySample{Float64}}}(
            Pre.OmegaFidelitySample{Float64}[], "fixture", "instantaneous", "dry",
            Pre.MERRA2_WIND_OMEGA_BASELINE)
        @test_throws ArgumentError Pre.verify_flux_fidelity!(Pre.FluxFidelityAccumulator(),
            Pre.FluxFidelityEvidenceBundle(
                Pre.NoFluxFidelityEvidence{CS,Float64,Pre.SourceHasNoWinds}(Pre.SourceHasNoWinds()),
                empty, Pre.NoColumnClosureRecord()), 1)
    end

    @testset "column-closure records reach the header" begin
        FT = Float32
        no_flux = Pre.NoFluxFidelityEvidence{CS,FT,Pre.SourcePayloadOnly}(Pre.SourcePayloadOnly())
        acc = Pre.FluxFidelityAccumulator()
        for (win, r) in enumerate((Pre.ColumnClosureRecord(2.0e5, 0.4, 3.0e4, 0.2),
                                   Pre.ColumnClosureRecord(1.0e5, 0.7, 6.0e4, 0.1)))
            Pre.verify_flux_fidelity!(acc, Pre.FluxFidelityEvidenceBundle(no_flux, no_omega(FT), r), win)
        end
        header = Pre.flux_fidelity_header!(Dict{String,Any}("column_balance_weights" => "hybrid_mass"), acc)
        @test header["column_closure_rule"] == Pre.COLUMN_CLOSURE_RULE
        @test header["column_closure_windows"] == 2
        @test header["column_closure_projected_residual_max_kg"] == 2.0e5
        @test header["column_closure_residual_to_tolerance_max"] == 0.7
        @test header["column_closure_booked_residual_max_kg"] == 6.0e4
        @test header["column_closure_cm_residual_to_tolerance_max"] == 0.2
        @test header["column_balance_weights"] == "hybrid_mass"      # not a certificate key
        @test header["flux_fidelity_sample_rule"] == "none"
        # Re-patching replaces, never accumulates, the certificate keys.
        @test Pre.flux_fidelity_header!(copy(header), acc) == header
        @test_throws ArgumentError Pre.flux_fidelity_header!(
            Dict{String,Any}("nwindow" => 3), acc)
        # The reader decodes and checks the record; absent keys mean "not recorded".
        parse_closure(h, nwindow = 2) =
            MD.BinaryFormat.parse_column_closure_header(JSON3.read(JSON3.write(h)), nwindow)
        closure = parse_closure(header)
        @test closure isa MD.ColumnClosureHeader
        @test closure.rule == MD.COLUMN_CLOSURE_RULE
        @test (closure.windows, closure.projected_residual_max_kg,
               closure.residual_to_tolerance_max, closure.booked_residual_max_kg,
               closure.cm_residual_to_tolerance_max) == (2, 2.0e5, 0.7, 6.0e4, 0.2)
        @test parse_closure(Dict{String,Any}("nwindow" => 2)) === MD.NoColumnClosureHeader()
        @test_throws ArgumentError parse_closure(header, 24)
        for mutate in (h -> h["column_closure_residual_to_tolerance_max"] = 1.2,
                       h -> h["column_closure_cm_residual_to_tolerance_max"] = 3.0,
                       h -> h["column_closure_projected_residual_max_kg"] = "NaN",
                       h -> h["column_closure_booked_residual_max_kg"] = -1.0,
                       h -> h["column_closure_rule"] = "global_max_v0",
                       h -> delete!(h, "column_closure_booked_residual_max_kg"))
            h = copy(header); mutate(h)
            @test_throws ArgumentError parse_closure(h)
        end
        # Windows without a column balance write no closure keys.
        none = Pre.FluxFidelityAccumulator()
        Pre.verify_flux_fidelity!(none, Pre.FluxFidelityEvidenceBundle(
            no_flux, no_omega(FT), Pre.NoColumnClosureRecord()), 1)
        @test !any(startswith("column_closure_"), keys(Pre.flux_fidelity_header!(Dict{String,Any}(), none)))
        # Partial coverage and out-of-tolerance records are rejected.
        Pre.verify_flux_fidelity!(none, Pre.FluxFidelityEvidenceBundle(
            no_flux, no_omega(FT), Pre.ColumnClosureRecord(1.0, 0.1, 0.0, 0.0)), 2)
        @test_throws ArgumentError Pre.flux_fidelity_header!(Dict{String,Any}(), none)
        @test_throws ArgumentError Pre.verify_flux_fidelity!(Pre.FluxFidelityAccumulator(),
            Pre.FluxFidelityEvidenceBundle(no_flux, no_omega(FT),
                                           Pre.ColumnClosureRecord(1.0, 1.5, 0.0, 0.0)), 1)
        @test_throws ArgumentError Pre.verify_flux_fidelity!(Pre.FluxFidelityAccumulator(),
            Pre.FluxFidelityEvidenceBundle(no_flux, no_omega(FT),
                                           Pre.ColumnClosureRecord(NaN, 0.1, 0.0, 0.0)), 1)
        # A payload-only writer gets the same serializer's not-applicable certificate.
        payload = Pre.payload_only_fidelity_header!(Dict{String,Any}(), CS, Float64)
        @test payload["flux_fidelity_reason"] == "source_payload_only"
        @test payload["omega_fidelity_reason"] == "source_payload_only"
        @test parse_flux(payload; nwindow = 24, nlevel = 72).status isa MD.FidelityNotApplicable
    end

    @testset "read-side parsing rejects inconsistent certificates" begin
        good = certificate(_ -> bundle(wind_evidence(Float32, (7, 9), 2)), 3, 2)
        @test parse_flux(good).status isa MD.FidelityPassed
        broken(f) = (h = copy(good); f(h); h)
        for (label, mutate) in (
                ("zero windows", h -> h["flux_fidelity_windows"] = 0),
                ("short coverage", h -> h["flux_fidelity_sample_count"] = 4),
                ("no samples per window", h -> h["flux_fidelity_samples_per_window"] = 0),
                ("kind none", h -> h["flux_fidelity_kind"] = "none"),
                ("native kind", h -> h["flux_fidelity_kind"] = "native_passthrough"),
                ("missing metric", h -> delete!(h, "flux_fidelity_max_abs_kg")),
                ("missing balance metric", h -> delete!(h, "balance_correction_rms_relative")),
                ("balance level outside the binary", h -> h["balance_correction_max_level"] = 3),
                ("non-finite metric", h -> h["flux_fidelity_rms_rel"] = "NaN"),
                ("reason on passed", h -> h["flux_fidelity_reason"] = "x"))
            @test_throws Exception parse_flux(broken(mutate))
        end
        @test_throws ArgumentError parse_flux(good; nwindow = 24)
        @test_throws ArgumentError parse_flux(good; nlevel = 72)
        not_applicable = Dict{String,Any}(
            "flux_fidelity_schema" => MD.FLUX_FIDELITY_SCHEMA, "flux_fidelity_status" => "not_applicable",
            "flux_fidelity_kind" => "none", "flux_fidelity_reason" => "source_payload_only",
            "flux_fidelity_samples_per_window" => 0, "flux_fidelity_sample_count" => 0,
            "flux_fidelity_windows" => 0)
        @test parse_flux(not_applicable; nwindow = 24, nlevel = 72).kind isa MD.NoFluxFidelityKind
        # Schema-1 headers (unreleased development builds) are refused, not misread.
        @test_throws ArgumentError parse_flux(merge(not_applicable, Dict("flux_fidelity_schema" => 1));
                                              nwindow = 24, nlevel = 72)
        @test_throws ArgumentError parse_flux(merge(not_applicable,
            Dict{String,Any}("flux_fidelity_max_abs_kg" => 0.0)); nwindow = 24, nlevel = 72)
        @test_throws ArgumentError parse_flux(merge(not_applicable,
            Dict{String,Any}("flux_fidelity_kind" => "wind_rederived")); nwindow = 24, nlevel = 72)
        @test_throws ArgumentError MD.BinaryFormat.parse_fidelity_status("maybe")
        @test_throws ArgumentError MD.BinaryFormat.parse_flux_fidelity_kind("symbol_switch")

        passed, _ = omega_header((src, k, rng) -> 0.9src + 0.02randn(rng))
        @test parse_omega(passed).status isa MD.FidelityPassed
        for mutate in (h -> h["omega_fidelity_sample_count"] = 0,
                       h -> h["omega_fidelity_passed"] = false,
                       h -> h["omega_fidelity_nrmse"] = 1.2,
                       h -> h["omega_fidelity_correlation"] = 0.2,
                       h -> h["omega_fidelity_slope"] = -0.1,
                       h -> h["omega_fidelity_slope"] = 1.6,
                       h -> delete!(h, "omega_fidelity_slope_max"),
                       h -> delete!(h, "omega_fidelity_nrmse_limit"),
                       h -> (delete!(h, "omega_fidelity_pressure_bands_hpa");
                             delete!(h, "omega_fidelity_band_nrmse")),
                       h -> h["omega_fidelity_band_nrmse"] = [0.3, "NaN", 0.2, 0.1, 0.4])
            h = copy(passed); mutate(h)
            @test_throws ArgumentError parse_omega(h)
        end
        unavailable = Dict{String,Any}(
            "omega_fidelity_status" => "unavailable", "omega_fidelity_reason" => "source_not_opened",
            "omega_fidelity_source" => "none", "omega_fidelity_sampling" => "none",
            "omega_fidelity_basis" => "none", "omega_fidelity_sample_count" => 0)
        @test parse_omega(unavailable).reason.code == "source_not_opened"
        @test_throws ArgumentError parse_omega(merge(unavailable,
            Dict{String,Any}("omega_fidelity_nrmse" => 0.3)))
        # Binaries written before certificates read as unavailable.
        legacy = MD.BinaryFormat.parse_flux_fidelity_header(JSON3.read("{}"), 24, 72)
        @test legacy.status isa MD.FidelityUnavailable
        @test legacy.reason.code == "legacy_header_missing"
    end
end
