#!/usr/bin/env julia
#
# OCO-2/OCO-3 level retrievals (issue #20 follow-up, decision D30): the hat
# projection of the model profile onto the retrieval levels, TM5-4DVar's
# 19-layer mode, the converter of the OCO-2 v11 MIP 10-second files and the
# reader of their retrieval fields.
#
# OCO-2 profile values sit on 20 levels ("defined on layer boundaries"), the
# profile linear in pressure between them; the pressure weights are the
# levels' hat integrals (trapezoid rule, half-thickness end levels, with a
# dry-air correction). The model column, stretched onto the retrieval levels,
# is projected onto the hat functions φ_j in dry air (the L2, Galerkin,
# projection):
#
#     M c = b,  M_ji = ∫ φ_j φ_i dp_dry,  b_j = ∫ φ_j x dp_dry,
#     X = Σ_j h_j a_j c_j + Σ_j h_j (1 − a_j) x_a,j.
#
# Checked here:
#   * the projection matrix against an independent construction (exact
#     trapezoid and Simpson integration of the hats and their products over
#     every stretched layer);
#   * rows sum to 1; column conservation: Σ_j h_j c_j is the model's
#     air-mass-weighted column mean with h the hat integrals;
#   * reproduction: a constant profile exactly, and a profile piecewise linear
#     on the levels (the retrieval's own representation) at every level, ends
#     and kinks included, to the second-order discretization of the model's
#     layer means;
#   * with the file's pressure weights, c = prior gives xco2_apriori, and a
#     model profile equal to the retrieval's prior gives the prior column;
#   * TM5 mode against a transcription of TM5-4DVar's Observations.py
#     (7016-7052) and Satellite.py (707-717);
#   * schema v2 round trip of level records (retrieval_mapping);
#   * the converter: a synthetic MIP file always, a subset of the real file
#     when it exists (read only); units, orientation, flags.

using Test
using Dates: DateTime, Second, Minute, Hour
using LinearAlgebra: dot
using Random: MersenneTwister
using NCDatasets: NCDataset

import AtmosTransport
const AT = AtmosTransport
const O = AT.Output

include(joinpath(@__DIR__, "..", "helpers", "oco2_mip_fixture.jl"))

const HAT = O.HatLevelProjection()

# OCO-2 levels at surface pressure `ps` (Pa), surface first.
_oco2_levels(ps) = reverse(OCO2_SIGMA .* ps)

function _level_column(ps; ak = collect(range(0.98, 0.32; length = 20)), prior = fill(400e-6, 20),
                       h = reverse(oco2_pressure_weights(ps)), sigma = Float64[])
    return O.RetrievalColumn(; pressure_pa = _oco2_levels(ps), pressure_weight = h, averaging_kernel = ak,
                             prior_profile = prior, profile_sigma = sigma, mapping = HAT)
end

# A random model column (top first) and its air masses.
function _random_model(rng; nz = rand(rng, 6:72), top = rand(rng, (0.0, 1.0, 100.0, 1000.0)),
                       ps = 1e5 * (0.6 + 0.45rand(rng)))
    p_half = vcat(top, sort!(top .+ (ps - top) .* rand(rng, nz - 1)), ps)
    q = 1e-2 .* rand(rng, nz)
    return (; p_half, air = diff(p_half) .* (1 .- q), x = 400e-6 .+ 2e-5 .* randn(rng, nz))
end

@testset "hat projection == independent construction; rows sum to 1; column conserved" begin
    rng = MersenneTwister(30)
    for _ in 1:200
        m = _random_model(rng)
        column = _level_column(m.p_half[end] * (1 + 0.1 * (rand(rng) - 0.5)))
        P = O.retrieval_projection_matrix(m.p_half, m.air, column)
        ref = hat_projection_reference(m.p_half, m.air, column.pressure_pa)
        @test P ≈ ref.P atol = 1e-12                                  # a tridiagonal solve: rounding × condition
        @test all(r -> isapprox(r, 1; atol = 1e-14), sum(P; dims = 2))
        # Column conservation: with the hat integrals of the model's air as
        # weights, the projected profile has the model's column mean.
        @test dot(ref.hat, P * m.x) ≈ dot(m.air, m.x) / sum(m.air) rtol = 1e-14
        # The weights of the column are (h a)ᵀ P.
        H = O.retrieval_column_weights!(zeros(length(m.x)), zeros(length(m.p_half)), m.p_half, m.air, column)
        @test H ≈ vec((column.pressure_weight .* column.averaging_kernel)' * ref.P) atol = 1e-14
        @test O.retrieval_column_value(m.p_half, m.air, column, m.x) ≈
              dot(H, m.x) + O.retrieval_prior_offset(column) rtol = 1e-15
    end
end

@testset "hat projection: profiles on the retrieval's basis are reproduced at every level" begin
    ps = 98000.0
    levels = _oco2_levels(ps)
    column = _level_column(ps)
    Δ = ps / 19
    # A profile piecewise linear on the levels (a kink at every level), sampled
    # as the layer means of a model grid that splits every interval between
    # levels into N equal layers (no humidity; the model column spans exactly
    # the retrieval levels, so the stretch is the identity).
    v = 4.0e-4 .+ 1e-6 .* sin.(1:20)                                  # level values, surface first
    slopes = [(v[j + 1] - v[j]) / (levels[j + 1] - levels[j]) for j in 1:19]
    interp(p) = (j = clamp(searchsortedlast(-levels, -p), 1, 19);
                 v[j] + (v[j + 1] - v[j]) * (p - levels[j]) / (levels[j + 1] - levels[j]))
    errors = map((8, 32)) do N
        p_half = sort!(unique!(vcat([collect(range(levels[j + 1], levels[j]; length = N + 1)) for j in 1:19]...)))
        air = diff(p_half)
        nz = length(air)
        P = O.retrieval_projection_matrix(p_half, air, column)
        # The layer mean of a linear piece is its midpoint value.
        x = [interp((p_half[k] + p_half[k + 1]) / 2) for k in 1:nz]
        c = P * x
        # A constant profile is reproduced exactly at every level.
        @test P * fill(4.07e-4, nz) ≈ fill(4.07e-4, 20) rtol = 1e-14
        # The piecewise-linear profile is reproduced at every level, the end
        # levels and the kinks included, to the discretization of the layer
        # means: ∫φ_j x over a layer of thickness δ with x its mean differs by
        # δ³ φ'_j v'/12, which gives |c − v| ≲ Δ max|v'| / N² (measured: 0.21 of it).
        @test maximum(abs.(c .- v)) <= Δ * maximum(abs.(slopes)) / N^2
        # The column is conserved, with the hat integrals of the model's air.
        hat = hat_projection_reference(p_half, air, levels).hat
        @test dot(hat, c) ≈ dot(air, x) / sum(air) rtol = 1e-14
        maximum(abs.(c .- v))
    end
    @test errors[1] / errors[2] ≈ 16 rtol = 0.05                       # second order in the layer thickness
    # A profile linear in pressure is reproduced exactly at every level (its
    # layer means integrate exactly against the hats where the layering is
    # symmetric, and the end levels are exact too: the projection reproduces
    # the basis).
    p_half = sort!(unique!(vcat([collect(range(levels[j + 1], levels[j]; length = 33)) for j in 1:19]...)))
    nz = length(p_half) - 1
    x = 3.9e-4 .+ 2e-10 .* (p_half[1:nz] .+ p_half[2:end]) ./ 2
    c = O.retrieval_projection_matrix(p_half, diff(p_half), column) * x
    @test maximum(abs.(c .- (3.9e-4 .+ 2e-10 .* levels))) <= Δ * 2e-10 / 32^2
end

# The column X of a model profile equal to the retrieval's own prior (piecewise
# linear on the levels), sampled as the layer means of a model grid with N
# layers per interval between levels, minus the prior column Σ h x_a (ppm).
function _prior_profile_departure(column; N = 40)
    levels = column.pressure_pa
    n = length(levels)
    v = column.prior_profile
    p_half = sort!(unique!(vcat([collect(range(levels[j + 1], levels[j]; length = N + 1)) for j in 1:(n - 1)]...)))
    interp(p) = (j = clamp(searchsortedlast(-levels, -p), 1, n - 1);
                 v[j] + (v[j + 1] - v[j]) * (p - levels[j]) / (levels[j + 1] - levels[j]))
    x = [interp((p_half[k] + p_half[k + 1]) / 2) for k in 1:(length(p_half) - 1)]
    X = O.retrieval_column_value(p_half, diff(p_half), column, x)
    return (X - dot(column.pressure_weight, v)) * 1e6
end

@testset "file pressure weights: c = prior gives xco2_apriori" begin
    mktempdir() do dir
        retrieval = oco2_fixture_retrieval(985.0)
        rec = (; id = 2018070112000011, time = DateTime(2018, 7, 1, 12), lat = 10.0, lon = 20.0, xco2 = 405.1,
               unc = 0.6, model_error = 0.3, psurf = 985.0, altitude = 120.0, data_type = 6, assimilate_flag = 1,
               retrieval...)
        path = write_oco2_mip_fixture(joinpath(dir, "fixture.nc4"), [rec])
        column = NCDataset(path) do ds
            only(O.read_oco2_retrieval_columns(ds, [1]))
        end
        apriori = NCDataset(path) do ds
            Float64(ds["xco2_apriori"][1])
        end
        # The column of the prior itself (c = x_a) is the retrieval's prior
        # column, within the Float32 precision of the file.
        prior_column = dot(column.pressure_weight .* column.averaging_kernel, column.prior_profile) +
                       O.retrieval_prior_offset(column)
        @test abs(prior_column * 1e6 - apriori) < 1e-4
        # A model profile equal to the retrieval's prior gives the prior column
        # (the projection reproduces the retrieval's representation; a hat
        # average without the mass matrix is 0.02 ppm low here).
        @test abs(_prior_profile_departure(column)) < 1e-5
        # A model profile equal to a constant c gives c at every level.
        p_half = vcat(1.0, collect(range(2000.0, 99500.0; length = 30)))
        air = diff(p_half)
        @test O.retrieval_projection_matrix(p_half, air, column) * fill(4.1e-4, 30) ≈ fill(4.1e-4, 20) rtol = 1e-14
    end
end

# ---------------------------------------------------------------------------
# TM5 mode
# ---------------------------------------------------------------------------

# TM5-4DVar's construction for the OCO-2 10-s files, transcribed:
# Observations.py 7016-7052 (levels σ·psurf, TOA first; kernel and prior
# interpolated linearly to the layer midpoints) and Satellite.py 707-717 (flip
# to surface first; h = Δp/p_surface; stretch and inv_rebin; the column).
function _tm5_transcription(psurf_hpa, sigma, ak_toa, prior_toa, model_profile_sfc, model_levels_sfc)
    pres_levels = sigma .* (100.0 * psurf_hpa)                       # TOA first
    mid = 0.5 .* (pres_levels[2:end] .+ pres_levels[1:(end - 1)])
    lerp(x, y, t) = [y[i] + (y[i + 1] - y[i]) * (t[i] - x[i]) / (x[i + 1] - x[i]) for i in eachindex(t)]
    avg_kernel = lerp(pres_levels, ak_toa, mid)
    co2_apri = lerp(pres_levels, prior_toa, mid)
    # Satellite.py: the first index is TOA, flip.
    p = reverse(pres_levels)
    ak = reverse(avg_kernel)
    prior = reverse(co2_apri)
    dP = -diff(p)
    pres_weights = dP ./ p[1]
    nmod = length(model_profile_sfc)
    m = (p[1] - p[end]) / (model_levels_sfc[1] - model_levels_sfc[end])
    c = 0.5 * ((p[1] - m * model_levels_sfc[1]) + (p[end] - m * model_levels_sfc[end]))
    in_x = m .* model_levels_sfc .+ c
    out_x = vcat(in_x[1], p[2:(end - 1)], in_x[end])
    noco = length(ak)
    R = zeros(noco, nmod)
    for i in 1:noco, j in 1:nmod
        ((in_x[j + 1] >= out_x[i]) || (out_x[i + 1] >= in_x[j])) && continue
        R[i, j] = (min(out_x[i], in_x[j]) - max(in_x[j + 1], out_x[i + 1])) / (out_x[i] - out_x[i + 1])
    end
    W = ak .* pres_weights
    model_tc = dot(W, R * model_profile_sfc) + dot(pres_weights .- W, prior)
    return (; model_tc, levels = p, ak, prior, pres_weights)
end

@testset "TM5 mode == transcription of TM5-4DVar's Observations.py / Satellite.py" begin
    rng = MersenneTwister(5)
    for psurf in (1013.2, 962.5, 701.0)
        r = oco2_fixture_retrieval(psurf; tilt = 0.2 * randn(rng))
        level = O.RetrievalColumn(; pressure_pa = _oco2_levels(100psurf), pressure_weight = reverse(r.pw),
                                  averaging_kernel = reverse(r.ak), prior_profile = reverse(r.prior) .* 1e-6,
                                  mapping = HAT)
        tm5 = O.tm5_layer_column(level)
        @test tm5.mapping isa O.LayerRebin && O.retrieval_layer_count(tm5) == 19
        # A model column (surface first, as TM5) and its profile.
        model_sfc = vcat(100psurf, sort!(100psurf .* (0.001 .+ 0.998 .* rand(rng, 30)); rev = true), 1.0)
        profile = 4e-4 .+ 1e-5 .* randn(rng, 31)
        t = _tm5_transcription(psurf, OCO2_SIGMA, r.ak, r.prior .* 1e-6, profile, model_sfc)
        @test tm5.pressure_pa == t.levels
        @test tm5.averaging_kernel ≈ t.ak rtol = 1e-14
        @test tm5.prior_profile ≈ t.prior rtol = 1e-14
        # TM5's weights sum to 1 − σ_top; the mode normalizes them.
        S = sum(t.pres_weights)
        @test S ≈ 1 - 1e-4 rtol = 1e-12
        @test tm5.pressure_weight .* S ≈ t.pres_weights rtol = 1e-14
        X = O.retrieval_column_value(reverse(model_sfc), tm5, reverse(profile))
        @test X * S ≈ t.model_tc rtol = 1e-13
    end
end

# ---------------------------------------------------------------------------
# Level records in observation schema v2
# ---------------------------------------------------------------------------

_record(id, retrieval; kind = "column", flag = 0) =
    AT.CSObservationRecord(; id, date_components = (2018, 7, 1, 12, 0, 0), lat = 10.0, lon = 20.0, kind,
                           value = 405e-6, value_sigma = 0.6e-6, model_sigma = 0.3e-6, flag,
                           instrument_type = "OCO2_OG", tracer = "CO2", retrieval)

_same(a, b) = all(f -> isequal(getfield(a, f), getfield(b, f)), fieldnames(typeof(a)))

@testset "schema v2: level records (retrieval_mapping = hat_levels)" begin
    level = _level_column(98000.0; sigma = fill(1e-6, 20))
    layer = O.tm5_layer_column(level)
    mktempdir() do dir
        set = AT.CSObservationSet([_record(1, level), _record(2, layer), _record(3, nothing),
                                   _record(4, _level_column(101000.0))]; time_origin = "1970-01-01 00:00:00")
        path = AT.write_observations(joinpath(dir, "levels.nc"), set; attributes = Dict("source_file" => "x"))
        NCDataset(path) do ds
            @test ds["retrieval_mapping"][:] == ["hat_levels", "layer_rebin", "", "hat_levels"]
            @test ds.dim["retrieval_level"] == 20 && ds.dim["retrieval_interface"] == 21
            @test all(isnan, ds["retrieval_pressure_pa"][21, [1, 4]])     # 20 levels, NaN-padded
            @test ds.attrib["source_file"] == "x"
        end
        back = AT.read_observations(path)
        @test all(_same(a, b) for (a, b) in zip(set.records, back.records))
        @test back[1].retrieval.mapping isa O.HatLevelProjection && back[2].retrieval.mapping isa O.LayerRebin
        # A mapping without a block, or a block without a mapping, is an error.
        NCDataset(path, "a") do ds
            ds["retrieval_mapping"][3] = "hat_levels"
        end
        @test_throws ArgumentError AT.read_observations(path)
        NCDataset(path, "a") do ds
            ds["retrieval_mapping"][3] = ""
            ds["retrieval_mapping"][1] = ""
        end
        @test_throws ArgumentError AT.read_observations(path)
        NCDataset(path, "a") do ds
            ds["retrieval_mapping"][1] = "midpoint_boxes"
        end
        @test_throws ArgumentError AT.read_observations(path)
        @test_throws ArgumentError AT.write_observations(joinpath(dir, "bad.nc"), set;
                                                         attributes = Dict("time_origin" => "x"))
    end
    # Level-column checks.
    @test_throws DimensionMismatch O.RetrievalColumn(; pressure_pa = _oco2_levels(1e5)[1:19],
                                                     pressure_weight = fill(0.05, 20), averaging_kernel = ones(20),
                                                     prior_profile = fill(4e-4, 20), mapping = HAT)
    @test_throws ArgumentError O.RetrievalColumn(; pressure_pa = [1e5], pressure_weight = [1.0],
                                                 averaging_kernel = [1.0], prior_profile = [4e-4], mapping = HAT)
    # The hat projection weights layers by their air mass, which it needs.
    @test_throws ArgumentError O.retrieval_column_weights!(zeros(3), zeros(4), [10.0, 3e4, 7e4, 1e5], level)
end

# ---------------------------------------------------------------------------
# The converter of the OCO-2 v11 MIP 10-second files
# ---------------------------------------------------------------------------

function _fixture_records()
    base = DateTime(2018, 7, 1, 12)
    specs = [(1, 6, 1, 985.0), (2, 1, 1, 812.0), (3, 2, 2, 1003.0), (4, 3, 0, 967.0), (5, 6, 1, 1011.0),
             (6, 2, 1, 955.0)]
    return [(; id = 2018070112000000 + 10k + dt, time = base + Second(10k), lat = 10.0 + k,
             lon = -50.0 + 2k, xco2 = 404.0 + 0.3k, unc = 0.5 + 0.05k, model_error = 0.2 + 0.01k, psurf,
             altitude = 50.0k, data_type = dt, assimilate_flag = af, oco2_fixture_retrieval(psurf; tilt = 0.02k)...)
            for (k, dt, af, psurf) in specs]
end

@testset "converter: synthetic MIP file → schema v2" begin
    mktempdir() do dir
        recs = _fixture_records()
        path = write_oco2_mip_fixture(joinpath(dir, "OCO2_b11.2_10sec_GOOD_r2.nc4"), recs)
        set = AT.read_oco2_mip_observations(path)
        @test length(set) == length(recs)
        for (r, rec) in zip(set.records, recs)
            @test r.id == rec.id && r.kind isa AT.ColumnObservationKind && r.time_window_s == 0
            @test DateTime(Int.(r.date_components)...) == rec.time
            @test r.lat == Float64(Float32(rec.lat)) && r.lon == Float64(Float32(rec.lon))
            @test r.elevation_m == Float64(Float32(rec.altitude))
            # ppm → mol/mol
            @test r.value ≈ Float64(Float32(rec.xco2)) * 1e-6 rtol = 1e-15
            @test r.value_sigma ≈ Float64(Float32(rec.unc)) * 1e-6 rtol = 1e-15
            @test r.model_sigma ≈ Float64(Float32(rec.model_error)) * 1e-6 rtol = 1e-15
            # Surface first: hPa → Pa, the bottom level the surface.
            c = r.retrieval
            @test c.mapping isa O.HatLevelProjection
            @test c.surface_pressure_pa ≈ 100 * Float64(Float32(rec.psurf)) rtol = 1e-7
            @test c.pressure_pa ≈ reverse(Float64.(Float32.(OCO2_SIGMA)) .* 100 .* Float64(Float32(rec.psurf))) rtol = 1e-7
            @test c.averaging_kernel == reverse(Float64.(Float32.(rec.ak)))   # not multiplied by the weights
            @test c.pressure_weight == reverse(Float64.(Float32.(rec.pw)))
            @test c.prior_profile ≈ reverse(Float64.(Float32.(rec.prior))) .* 1e-6 rtol = 1e-15
        end
        # Flags: 0 = assimilated (af ∈ {1}, LN/LG/OG); else 1 + assimilate_flag.
        @test [r.flag for r in set.records] == [0, 0, 3, 1, 0, 0]
        @test [r.instrument_type for r in set.records] == ["OCO2_OG", "OCO2_LN", "OCO2_LG", "OCO2_LT", "OCO2_OG",
                                                           "OCO2_LG"]
        # Options: data types, assimilate flags, model sigma, time range, TM5 mode.
        only_og = AT.read_oco2_mip_observations(path; data_types = ("OG",), assimilate_flags = (1, 2),
                                                model_sigma = 0.5, mapping = :tm5,
                                                time_range = (DateTime(2018, 7, 1, 12, 0, 15),
                                                              DateTime(2018, 7, 1, 12, 1)))
        @test [r.id for r in only_og.records] == [r.id for r in recs[2:5]]
        @test [r.flag for r in only_og.records] == [2, 3, 1, 0]
        @test all(r -> r.model_sigma ≈ 0.5e-6, only_og.records)
        @test all(r -> r.retrieval.mapping isa O.LayerRebin && O.retrieval_layer_count(r.retrieval) == 19,
                  only_og.records)
        @test_throws ArgumentError AT.read_oco2_mip_observations(path; data_types = ("XX",))
        # Files per day, round trip.
        paths = AT.convert_oco2_mip_observations(path, joinpath(dir, "obs_{YYYYMMDD}.nc"))
        @test paths == [joinpath(dir, "obs_20180701.nc")]
        back = AT.read_observations(only(paths))
        @test all(_same(a, b) for (a, b) in zip(set.records, back.records))
        NCDataset(only(paths)) do ds
            @test ds.attrib["source_file"] == basename(path)
            @test occursin("hat", ds.attrib["oco2_mip_mapping"])
            @test ds["value"].attrib["units"] == "mol mol-1"
            @test ds["retrieval_pressure_pa"].attrib["ordering"] == "surface_to_top"
        end
        monthly = AT.convert_oco2_mip_observations(path, joinpath(dir, "obs_{YYYYMM}.nc"); partition = :month)
        @test monthly == [joinpath(dir, "obs_201807.nc")]
        @test_throws ArgumentError AT.convert_oco2_mip_observations(path, joinpath(dir, "obs.nc"))
        # The command-line script.
        script = joinpath(@__DIR__, "..", "..", "scripts", "preprocessing", "oco2_mip_to_observations.jl")
        cli = Module(:OCO2MIPConverterCLI)
        Base.include(cli, script)
        args = [path, joinpath(dir, "cli_{YYYYMMDD}.nc"), "--data-types", "OG", "--mapping", "tm5", "--model-sigma", "0.4"]
        written = Base.invokelatest(() -> cli.main(args))
        cli_set = AT.read_observations(only(written))
        @test [r.flag for r in cli_set.records] == [0, 2, 3, 1, 0, 2]
        @test all(r -> r.model_sigma ≈ 0.4e-6 && r.retrieval.mapping isa O.LayerRebin, cli_set.records)
    end
end

const MIP_FILE = "/kiwi-data/Data/model/OCO2MIP/observation_input/OCO2_b11.2_10sec_GOOD_r2.nc4"

@testset "converter: real MIP records (read only, when the file exists)" begin
    if !isfile(MIP_FILE)
        @info "skipping the real-record converter check: $(MIP_FILE) not found"
        @test_skip isfile(MIP_FILE)
    else
        # Two hours of mid-2019 records.
        range = (DateTime(2019, 7, 1, 0), DateTime(2019, 7, 1, 2))
        set = AT.read_oco2_mip_observations(MIP_FILE; time_range = range)
        @test length(set) > 10
        apriori, af, dtype, psurf = NCDataset(MIP_FILE) do ds
            ids = Array(ds["sounding_id"].var)
            rows = [findfirst(==(r.id), ids) for r in set.records]
            lo, hi = extrema(rows)
            idx = rows .- lo .+ 1
            (Float64.(ds["xco2_apriori"].var[lo:hi][idx]), Int.(ds["assimilate_flag"].var[lo:hi][idx]),
             Int.(ds["data_type"].var[lo:hi][idx]), Float64.(ds["psurf"].var[lo:hi][idx]))
        end
        worst = 0.0
        for (k, r) in enumerate(set.records)
            c = r.retrieval
            @test issorted(c.pressure_pa; rev = true)
            @test c.surface_pressure_pa ≈ 100 * psurf[k] rtol = 1e-6
            @test abs(sum(c.pressure_weight) - 1) < 1e-5
            # The file's own prior column (c = prior) is xco2_apriori, up to the
            # 10-s averaging of the weights and the prior separately.
            worst = max(worst, abs(dot(c.pressure_weight, c.prior_profile) * 1e6 - apriori[k]))
            # A model profile equal to the record's prior gives its prior column.
            k <= 10 && @test abs(_prior_profile_departure(c)) < 1e-5
            @test r.flag == ((af[k] == 1 && dtype[k] in (1, 2, 6)) ? 0 : 1 + af[k])
        end
        @test worst < 0.02
        mktempdir() do dir
            path = AT.write_observations(joinpath(dir, "real.nc"), set)
            back = AT.read_observations(path)
            @test all(_same(a, b) for (a, b) in zip(set.records, back.records))
        end
    end
end
