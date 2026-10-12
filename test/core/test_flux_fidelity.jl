#!/usr/bin/env julia

using Test
using AtmosTransport
using JSON3

const Pre = AtmosTransport.Preprocessing
const MD = AtmosTransport.MetDrivers
const G = AtmosTransport.Grids

@testset "typed write-time flux fidelity" begin
    @testset "sample map is deterministic and canonical" begin
        Nc = 4
        mesh = G.CubedSphereMesh(; Nc, FT=Float64)
        table = Pre.build_cs_global_face_table(
            Nc, G.panel_connectivity_for(mesh.convention))
        a = Pre.FidelitySampleMap(table; samples_per_window=32)
        b = Pre.FidelitySampleMap(table; samples_per_window=32)
        @test a.version == 1
        @test a.seed == b.seed
        @test a.faces == b.faces
        @test length(a.faces) == length(unique(a.faces)) == 32
        @test all(f -> 1 <= f <= table.nf, a.faces)
    end

    @testset "typed evidence passes and injected errors fail before write" begin
        for FT in (Float32, Float64)
            samples = Pre.FluxFidelitySample{FT}[
                Pre.FluxFidelitySample(7, 2, FT(3), FT(3), FT(3.25)),
                Pre.FluxFidelitySample(9, 1, FT(-2), FT(-2), FT(-1.75)),
            ]
            evidence = Pre.WindDerivedFluxEvidence{
                Pre.CubedSphereTargetGeometry,FT,String,typeof(samples)}(
                    "vector", samples, 2, "window_mean", 150.0)
            acc = Pre.FluxFidelityAccumulator()
            Pre.verify_flux_fidelity!(acc, evidence, 4)
            header = Dict{String,Any}()
            omega = Pre.NoSourceOmegaEvidence{
                Pre.CubedSphereTargetGeometry,FT,Pre.SourceHasNoOmega}(
                    Pre.SourceHasNoOmega())
            Pre.verify_omega_fidelity!(acc, omega, 4)
            Pre.flux_fidelity_header!(header, acc)
            @test header["flux_fidelity_status"] == "passed"
            @test header["flux_fidelity_kind"] == "wind_rederived"
            @test header["flux_fidelity_sample_count"] == 2
            @test header["balance_correction_max_abs_kg"] == FT(0.25)
            @test header["omega_fidelity_status"] == "not_applicable"
            @test header["omega_fidelity_reason"] == "source_has_no_omega"

            bad_samples = Pre.FluxFidelitySample{FT}[
                Pre.FluxFidelitySample(1, 1, FT(1), FT(1.01), FT(1))]
            bad = Pre.WindDerivedFluxEvidence{
                Pre.CubedSphereTargetGeometry,FT,String,typeof(bad_samples)}(
                    "vector", bad_samples, 1, "window_start_endpoint", 1.0)
            @test_throws ArgumentError Pre.verify_flux_fidelity!(
                Pre.FluxFidelityAccumulator(), bad, 1)
        end
    end

    @testset "line-integral samples rebuild the midpoint quadrature" begin
        Nc, Nz = 3, 2
        source = G.LatLonMesh(; Nx=12, Ny=6, FT=Float64)
        target = G.CubedSphereMesh(; Nc, FT=Float64)
        table = Pre.build_cs_global_face_table(
            Nc, G.panel_connectivity_for(target.convention))
        method = Pre.LineIntegralFaceFluxes(source, target, table;
                                            samples_per_face=4)
        nsrc = G.ncells(source)
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
        method.L.nzval .= NaN
        captured = Pre.capture_cs_wind_flux_samples(
            method, am, bm, u, v, dp, table, 9.80665, 60.0, Nc, Nz;
            sample_map=smap)
        @test captured.reference ≈ captured.candidate rtol=2e-14
        evidence = Pre.wind_derived_evidence(
            Pre.CubedSphereTargetGeometry, captured, am, bm, table;
            method="line_integral", samples_per_window=length(smap.faces),
            time_sampling="window_start_endpoint", amount_seconds=60.0)
        @test isnothing(Pre.verify_flux_fidelity!(
            Pre.FluxFidelityAccumulator(), evidence, 1))
    end

    @testset "aggregate OMEGA failure is fail-closed" begin
        FT = Float64
        flux_samples = Pre.FluxFidelitySample{FT}[
            Pre.FluxFidelitySample(1, 1, 1.0, 1.0, 1.0)]
        flux = Pre.NativeFluxEvidence{
            Pre.CubedSphereTargetGeometry,FT,typeof(flux_samples)}(
                flux_samples, 1, "native_scaling_mapping", 300.0)
        omega_samples = Pre.OmegaFidelitySample{FT}[
            Pre.OmegaFidelitySample(1.0, 2.0, 500.0, 0.0)]
        omega = Pre.SourceOmegaEvidence{
            Pre.CubedSphereTargetGeometry,FT,typeof(omega_samples)}(
                omega_samples, "fixture", "instantaneous", "dry", "fixture-v1", 0.1)
        acc = Pre.FluxFidelityAccumulator()
        Pre.verify_flux_fidelity!(acc, Pre.FluxFidelityEvidenceBundle(flux, omega), 1)
        header = Dict{String,Any}()
        @test_throws ArgumentError Pre.flux_fidelity_header!(header, acc)
        @test header["omega_fidelity_status"] == "failed"
        @test header["omega_fidelity_reason"] == "baseline_limit_exceeded"
    end

    @testset "read-side parsing uses nominal status and kind types" begin
        raw = Dict{String,Any}(
            "flux_fidelity_schema" => 1,
            "flux_fidelity_status" => "passed",
            "flux_fidelity_kind" => "wind_rederived",
            "flux_fidelity_sample_rule" => Pre.FLUX_FIDELITY_SAMPLE_RULE,
            "flux_fidelity_sample_seed" => string(Pre.FLUX_FIDELITY_SAMPLE_SEED),
            "flux_fidelity_samples_per_window" => 2,
            "flux_fidelity_sample_count" => 4,
            "flux_fidelity_windows" => 2,
            "omega_fidelity_status" => "unavailable",
            "omega_fidelity_reason" => "source_not_opened",
            "omega_fidelity_source" => "none",
            "omega_fidelity_sampling" => "none",
            "omega_fidelity_basis" => "none",
            "omega_fidelity_sample_count" => 0,
        )
        json = JSON3.read(JSON3.write(raw))
        flux = MD._parse_flux_fidelity_header(json)
        omega = MD._parse_omega_fidelity_header(json)
        @test flux.status isa MD.FidelityPassed
        @test flux.kind isa MD.WindRederivedFidelity
        @test flux.reason === nothing
        @test omega.status isa MD.FidelityUnavailable
        @test omega.reason.code == "source_not_opened"
        @test_throws ArgumentError MD._parse_fidelity_status("maybe")
        @test_throws ArgumentError MD._parse_flux_fidelity_kind("symbol_switch")
    end
end
