#!/usr/bin/env julia
#
# Issue #20, part 4: averaging-kernel (retrieval) column observations in the
# 4D-Var, with TM5-4DVar's observation operator.
#
# A column record with a retrieval block (layer interfaces from the surface up,
# pressure weights h, column averaging kernel a, prior x_a; `RetrievalColumn`)
# is bound to a `CSRetrievalColumnObjective`. At every sampled state the model's
# total interface pressures (dry air mass plus humidity) are stretched onto the
# retrieval column and the dry mixing ratios are rebinned onto its layers by
# pressure overlap (R):
#
#     X = Σᵢ hᵢ aᵢ (R x)ᵢ + Σᵢ hᵢ (1 − aᵢ) x_a,ᵢ = Σₖ Hₖ xₖ + c.
#
# Checked here:
#   * agreement with TM5-4DVar's `averaging_kernels.F90` (OCO-2 routine and
#     `inv_rebin`): reference values computed with the Fortran itself on fixed
#     columns (model surface below and above the retrieval's, retrieval top
#     above and below the model top), and a line-by-line transcription of the
#     routine on random columns;
#   * R conserves the column integral, and with a = 1 the operator returns the
#     model's pressure-weighted column mean whatever the surface-pressure
#     mismatch; on the model's own levels it equals the column mean;
#   * the retrieval block of observation schema v2: round trips with and without
#     it, units, ordering, NaN padding, validation errors;
#   * the single sweep equals the per-observation reference (a seeded footprint
#     per observation and sampled step, weights from the transcription);
#   * central finite differences of the cost, with window records;
#   * the surface-pressure mismatch policy (error, skip) and the frozen
#     model-error term;
#   * departures v2 with the retrieval diagnostics.

using Test
using Dates: DateTime, Second, year, month, day, hour, minute, second
using LinearAlgebra: dot
using Random: MersenneTwister
using NCDatasets: NCDataset, defDim, defVar
using Logging: NullLogger, with_logger

import AtmosTransport
const AT = AtmosTransport
const ADJ = AT.Adjoints
const O = AT.Observations

# Evaluate production profile interpolation on unit vectors for an independent matrix comparison.
function projection_from_profiles(p_half, air, column)
    nz = length(p_half) - 1
    P = zeros(O.retrieval_layer_count(column), nz)
    for k in 1:nz
        x = zeros(nz); x[k] = 1.0
        O.retrieval_profile!(view(P, :, k), zeros(nz + 1), p_half, air, column, x)
    end
    return P
end


const Adv = AT.Operators.Advection

include(joinpath(@__DIR__, "..", "helpers", "cs_observation_columns.jl"))
include(joinpath(@__DIR__, "..", "helpers", "cs_inversion_reference.jl"))

const T0 = DateTime(2024, 6, 15)
const DT = 600.0

_dc(s) = (t = T0 + Second(round(Int, s)); (year(t), month(t), day(t), hour(t), minute(t), second(t)))

# ---------------------------------------------------------------------------
# TM5-4DVar reference
# ---------------------------------------------------------------------------

# TM5-4DVar's OCO-2 operator transcribed line by line, in its surface-first
# indexing: the stretch (averaging_kernels.F90:213-219), the distribution matrix
# of `inv_rebin` (103-142) and the column (233-235, 245). `h` defaults to the
# routine's own pressure weights Δp/p_surface (233).


# Reference values of TM5-4DVar's OCO-2 operator: `applyaveragingkernel` and
# `inv_rebin` of deps/tm5-cy3-4dvar/base/f2py/src/averaging_kernels.F90 (file
# sha256 e6c34700…fb136; modules `utility_functions` and `oco2`, lines 1-272,
# compiled unchanged with GNU Fortran 11.5.0, -O0 -ffp-contract=off) called
# from a driver program with these inputs, the measured column 0 and its error
# 1, so that the departures are H·X. TM5 order (surface first); model edges
# p = AT + BT·p_s.
const TM5_BT = [1.0, 0.97, 0.90, 0.78, 0.60, 0.38, 0.17, 0.04, 0.0]
const TM5_AT = [0.0, 0.0, 400.0, 1800.0, 4500.0, 7600.0, 9200.0, 6000.0, 100.0]
const TM5_CASES = (
    # Model surface below the retrieval's (98000 < 100000 Pa); retrieval top
    # (0 Pa) above the model top (100 Pa).
    (psurf = 98000.0,
     profile = [4.12e-4, 4.09e-4, 4.06e-4, 4.03e-4, 4.01e-4, 3.99e-4, 3.96e-4, 3.92e-4],
     levels = [100000.0, 85000.0, 70000.0, 50000.0, 30000.0, 10000.0, 0.0],
     ak = [0.62, 0.81, 0.95, 1.03, 1.08, 1.12],
     prior = [4.05e-4, 4.04e-4, 4.02e-4, 4.00e-4, 3.98e-4, 3.90e-4],
     model_tc = 4.00388092951991864e-4,
     departures = [7.45482549369653405e-6, 1.63803308466937530e-5, 3.02129419538119930e-5,
                   5.25436778652707662e-5, 7.30996598141502697e-5, 8.06906745161747142e-5,
                   7.04061399161096124e-5, 4.49759748295367566e-5],
     stretched = [9.99999999999999854e4, 9.69969356486210309e4, 9.03983656792645488e4, 7.98161389172625059e4,
                  6.45556690500510740e4, 4.56996935648621075e4, 2.63125638406537328e4, 1.00306435137895878e4,
                  7.77333752921549603e-12],
     squeezed = [4.08520939734422872e-4, 4.04036772216547521e-4, 4.01544433094994861e-4,
                 3.99430030643513757e-4, 3.96546986721143927e-4, 3.92000000000000475e-4]),
    # Model surface above the retrieval's (101500 > 96000 Pa); retrieval top
    # (1000 Pa) below the model top (100 Pa).
    (psurf = 101500.0,
     profile = [4.31e-4, 4.18e-4, 4.07e-4, 4.02e-4, 4.00e-4, 3.98e-4, 3.97e-4, 3.95e-4],
     levels = [96000.0, 80000.0, 62000.0, 45000.0, 28000.0, 12000.0, 1000.0],
     ak = [0.45, 0.70, 0.92, 1.05, 1.10, 0.97],
     prior = [4.10e-4, 4.06e-4, 4.03e-4, 4.01e-4, 3.99e-4, 3.92e-4],
     model_tc = 3.97258863694732105e-4,
     departures = [5.31236398831792066e-6, 1.16976684865917161e-5, 2.21528827792746144e-5,
                   4.22546652863909313e-5, 6.88772816193544484e-5, 8.07329962376413988e-5,
                   6.90208504764466325e-5, 3.74558007399651720e-5],
     stretched = [9.60000000000000000e4, 9.31471893491124356e4, 8.68653846153846243e4, 7.67657790927021706e4,
                  6.21785009861932995e4, 4.41622287968441888e4, 2.56915680473372813e4, 1.03313609467455644e4,
                  1.00000000000000171e3],
     squeezed = [4.15597956730769287e-4, 4.02878561253561165e-4, 4.00000000000000019e-4,
                 3.98098561318018375e-4, 3.97144276997041452e-4, 3.95303388918773332e-4]),
    # Equal surface pressures; retrieval top (50 Pa) above the model top.
    (psurf = 100000.0,
     profile = [4.05e-4, 4.05e-4, 4.04e-4, 4.04e-4, 4.03e-4, 4.01e-4, 3.97e-4, 3.90e-4],
     levels = [100000.0, 90000.0, 72000.0, 55000.0, 33000.0, 15000.0, 50.0],
     ak = [0.80, 0.90, 1.00, 1.00, 1.05, 1.20],
     prior = [4.00e-4, 4.00e-4, 4.00e-4, 3.99e-4, 3.97e-4, 3.91e-4],
     model_tc = 4.00484595795795779e-4,
     departures = [9.61644092487440202e-6, 2.11561700347237197e-5, 3.80670830883676815e-5,
                   5.81850799922281389e-5, 7.57294722833860980e-5, 7.91019414389035330e-5,
                   7.11872192935021479e-5, 4.76013825781283981e-5],
     stretched = [1.00000000000000000e5, 9.69984984984985058e4, 9.03951951951952069e4, 7.97898898898898915e4,
                  6.44822322322322361e4, 4.55727727727727761e4, 2.61630630630630658e4, 9.95495495495495561e3,
                  5.00000000000003695e1],
     squeezed = [4.04960480480480434e-4, 4.04000000000000008e-4, 4.03442221633398053e-4,
                 4.01857020657020675e-4, 3.98519319319319532e-4, 3.92362228449184586e-4]),
)

# The production operator on TM5's inputs: our model columns are top first.
function _production(profile, tm5_levels, levels, ak, prior; h = (levels[1:(end - 1)] .- levels[2:end]) ./ levels[1])
    p_half = reverse(tm5_levels)
    x = reverse(profile)
    nz = length(x)
    H = zeros(nz)
    s = zeros(nz + 1)
    total_h = sum(h)
    column = O.RetrievalColumn(; pressure_pa = levels, pressure_weight = h ./ total_h,
                               averaging_kernel = ak, prior_profile = prior)
    O.retrieval_column_weights!(H, s, p_half, nothing, column)
    H .*= total_h
    X = dot(H, x) + O.retrieval_prior_offset(h, ak, prior)
    R = projection_from_profiles(p_half, nothing, column)
    return (; X, H, s, R, profile = R * x)
end

@testset "agreement with TM5-4DVar's averaging_kernels.F90 (Fortran reference values)" begin
    for case in TM5_CASES
        tm5_levels = TM5_AT .+ TM5_BT .* case.psurf
        H_ref = case.departures ./ case.model_tc         # TM5 order
        # The transcription reproduces the Fortran.
        t = _tm5_oco2(case.profile, tm5_levels, case.levels, case.ak, case.prior)
        @test t.model_tc ≈ case.model_tc rtol = 1e-14
        @test t.H ≈ H_ref rtol = 1e-13
        @test t.stretched ≈ case.stretched rtol = 1e-15 atol = 1e-10
        @test t.squeezed ≈ case.squeezed rtol = 1e-13
        # Production agrees with both.
        p = _production(case.profile, tm5_levels, case.levels, case.ak, case.prior)
        @test p.X ≈ case.model_tc rtol = 1e-14
        @test reverse(p.H) ≈ H_ref rtol = 1e-13
        @test reverse(p.s) ≈ case.stretched rtol = 1e-15 atol = 1e-10
        @test p.profile ≈ case.squeezed rtol = 1e-13
        @test p.R ≈ t.R[:, end:-1:1] atol = 1e-15
    end
    # The cases differ in what the stretch absorbs.
    @test TM5_CASES[1].psurf < TM5_CASES[1].levels[1] && TM5_CASES[2].psurf > TM5_CASES[2].levels[1]
    @test TM5_CASES[1].levels[end] < TM5_AT[end] < TM5_CASES[2].levels[end]
end

# A random model column (surface first, as TM5) and retrieval column.
function _random_columns(rng)
    nmod, noco = rand(rng, 4:40), rand(rng, 1:22)
    top_mod = rand(rng, (0.0, 1.0, 100.0, 1000.0))
    ps_mod = 1e5 * (0.6 + 0.45rand(rng))
    model = vcat(ps_mod, sort!(top_mod .+ (ps_mod - top_mod) .* rand(rng, nmod - 1); rev = true), top_mod)
    top_ret = rand(rng, (0.0, 50.0, 500.0, 3000.0))
    ps_ret = ps_mod * (1 + 0.15 * (rand(rng) - 0.5))
    levels = vcat(ps_ret, sort!(top_ret .+ (ps_ret - top_ret) .* rand(rng, noco - 1); rev = true), top_ret)
    profile = 4e-4 .+ 2e-5 .* randn(rng, nmod)
    ak = 0.3 .+ rand(rng, noco)
    prior = 4e-4 .+ 1e-5 .* randn(rng, noco)
    return (; model, levels, profile, ak, prior)
end

@testset "production == TM5 transcription on random columns" begin
    rng = MersenneTwister(20)
    worst = 0.0
    for _ in 1:300
        c = _random_columns(rng)
        h = rand(rng, length(c.ak))
        h ./= sum(h)
        t = _tm5_oco2(c.profile, c.model, c.levels, c.ak, c.prior; h)
        p = _production(c.profile, c.model, c.levels, c.ak, c.prior; h)
        worst = max(worst, abs(p.X - t.model_tc) / t.model_tc, maximum(abs.(reverse(p.H) .- t.H)))
        @test p.R ≈ t.R[:, end:-1:1] atol = 1e-14
    end
    @test worst < 1e-13
end

@testset "the rebin conserves the column; a = 1 gives the pressure-weighted column mean" begin
    rng = MersenneTwister(4)
    for _ in 1:200
        c = _random_columns(rng)
        p_half = reverse(c.model)
        x = reverse(c.profile)
        h = -diff(c.levels) ./ (c.levels[1] - c.levels[end])
        column = O.RetrievalColumn(; pressure_pa = c.levels, pressure_weight = h,
                                   averaging_kernel = c.ak, prior_profile = c.prior)
        R = projection_from_profiles(p_half, nothing, column)
        s = O.stretch_model_interfaces!(zeros(length(p_half)), p_half, c.levels[1], c.levels[end])
        # TM5's outer retrieval interfaces are the stretched model's.
        edges = vcat(s[end], c.levels[2:(end - 1)], s[1])
        Δr = edges[1:(end - 1)] .- edges[2:end]
        Δs = diff(s)
        @test all(x -> isapprox(x, 1; atol = 1e-13), sum(R; dims = 2))
        @test vec(sum(R .* Δr; dims = 1)) ≈ Δs rtol = 1e-12
        @test dot(Δr, R * x) ≈ dot(Δs, x) rtol = 1e-13
        # The stretch keeps each layer's share of the column: with a = 1 and
        # weights normalized to the retrieval column, X is the model's
        # pressure-weighted mean, whatever the surface-pressure mismatch.
        n = length(c.ak)
        h = -diff(c.levels) ./ (c.levels[1] - c.levels[end])
        column = O.RetrievalColumn(; pressure_pa = c.levels, pressure_weight = h, averaging_kernel = ones(n),
                                   prior_profile = clamp.(c.prior, 0, 1))
        Δp = diff(p_half)
        @test O.retrieval_column_value(p_half, nothing, column, x) ≈ dot(Δp, x) / sum(Δp) rtol = 1e-13
    end
end

# ---------------------------------------------------------------------------
# RetrievalColumn checks
# ---------------------------------------------------------------------------

_edges(ps; top = 0.0, n = 6) = collect(range(ps, top; length = n + 1))
function _retrieval(ps; top = 0.0, n = 6, ak = collect(range(0.7, 1.1; length = n)),
                    prior = collect(range(4.05e-4, 3.95e-4; length = n)), sigma = Float64[])
    edges = _edges(ps; top, n)
    return O.RetrievalColumn(; pressure_pa = edges, pressure_weight = -diff(edges) ./ (edges[1] - edges[end]),
                             averaging_kernel = ak, prior_profile = prior, profile_sigma = sigma)
end

@testset "RetrievalColumn validation" begin
    good = _retrieval(98000.0)
    @test O.retrieval_layer_count(good) == 6 && good.surface_pressure_pa == 98000.0
    kw = (; pressure_pa = good.pressure_pa, pressure_weight = good.pressure_weight,
          averaging_kernel = good.averaging_kernel, prior_profile = good.prior_profile)
    bad(msg; kwargs...) = (err = try O.RetrievalColumn(; kw..., kwargs...) catch e e end;
                           err isa Union{ArgumentError, DimensionMismatch} && occursin(msg, sprint(showerror, err)))
    @test bad("from the surface up"; pressure_pa = reverse(good.pressure_pa),
              surface_pressure_pa = good.pressure_pa[1])
    @test bad("sum to 1"; pressure_weight = good.pressure_weight .* 0.99)
    @test bad(">= 0"; pressure_weight = [0.5, -0.1, 0.2, 0.2, 0.1, 0.1])
    @test bad("ppm"; prior_profile = good.prior_profile .* 1e6)
    @test bad("finite"; averaging_kernel = [1.0, NaN, 1, 1, 1, 1])
    @test bad("differs from the first"; surface_pressure_pa = 97000.0)
    @test bad("interface pressures"; pressure_pa = good.pressure_pa[1:end - 1])
    @test bad("profile_sigma"; profile_sigma = [1e-6])
    @test bad("top pressure"; pressure_pa = vcat(good.pressure_pa[1:end - 1], -10.0))
    # The model-error variance Σ (h a σ)², zero without σ.
    σ = fill(2e-6, 6)
    with_sigma = O.RetrievalColumn(; kw..., profile_sigma = σ)
    @test O.retrieval_model_error_variance(with_sigma) ≈ sum((good.pressure_weight .* good.averaging_kernel .* σ) .^ 2)
    @test O.retrieval_model_error_variance(good) == 0
    @test bad("within [0, 1]"; profile_sigma = fill(2.0, 6))               # an uncertainty in ppm
end

# ---------------------------------------------------------------------------
# Observation schema v2: the retrieval block
# ---------------------------------------------------------------------------

function _record(id; lat = 10.0, lon = 20.0, t = 450.0, kind = "column", value = 401e-6, value_sigma = 1e-6,
                 kwargs...)
    return AT.CSObservationRecord(; id, date_components = _dc(t), lat, lon, kind, value, value_sigma,
                                  instrument_type = "test", tracer = "CO2", kwargs...)
end

_same(a::AT.CSObservationRecord, b::AT.CSObservationRecord) =
    all(f -> isequal(getfield(a, f), getfield(b, f)), fieldnames(AT.CSObservationRecord))

# `value_units` are the units stated on value, value_sigma and model_sigma
# (`nothing` = none), whose data are multiplied by `value_scale`.
const _MOLMOL = Dict{String, Any}("value" => "mol mol-1", "value_sigma" => "mol mol-1", "model_sigma" => "mol mol-1")
function _restate_values!(ds; value_units = _MOLMOL, value_scale = 1.0)
    for (name, units) in value_units
        units === nothing || (ds[name].attrib["units"] = units)
        value_scale == 1 || (ds[name][:] = ds[name][:] .* value_scale)
    end
    return ds
end

@testset "schema v2 round trip with and without the retrieval block" begin
    mktempdir() do dir
        plain = AT.CSObservationSet([_record(1; kind = "point", intake_height_m = 10.0), _record(2)];
                                    time_origin = "2024-06-15 00:00:00")
        path = AT.write_observations(joinpath(dir, "plain.nc"), plain)
        NCDataset(path) do ds
            @test !haskey(ds, "retrieval_pressure_pa") && !haskey(ds.dim, "retrieval_level")
        end
        back = AT.read_observations(path)
        @test all(_same(a, b) for (a, b) in zip(plain.records, back.records))
        @test all(r -> r.retrieval === nothing, back.records)

        records = [_record(1; kind = "point", intake_height_m = 10.0),
                   _record(2),
                   _record(3; retrieval = _retrieval(98000.0; n = 6)),
                   _record(4; retrieval = _retrieval(95000.0; top = 100.0, n = 4, sigma = fill(1.5e-6, 4))),
                   _record(5; retrieval = _retrieval(101000.0; n = 9), model_sigma = 0.3e-6)]
        set = AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00")
        path = AT.write_observations(joinpath(dir, "block.nc"), set)
        NCDataset(path) do ds
            @test ds.dim["retrieval_level"] == 9 && ds.dim["retrieval_interface"] == 10
            @test ds["retrieval_pressure_pa"].attrib["ordering"] == "surface_to_top"
            @test ds["retrieval_prior_profile"].attrib["units"] == "mol mol-1"
            @test haskey(ds, "retrieval_profile_sigma")
            @test all(isnan, ds["retrieval_pressure_pa"][:, 1]) && isnan(ds["retrieval_surface_pressure_pa"][2])
            @test all(isnan, ds["retrieval_pressure_weight"][5:end, 4])        # NaN padding
        end
        back = AT.read_observations(path)
        @test all(_same(a, b) for (a, b) in zip(set.records, back.records))
        @test back[4].retrieval.profile_sigma == fill(1.5e-6, 4) && isempty(back[3].retrieval.profile_sigma)
        # Only column records carry a block, and their values are mole fractions in mol/mol.
        @test_throws ArgumentError _record(9; kind = "point", intake_height_m = 10.0, retrieval = _retrieval(9e4))
        err = try _record(9; value = 401.0, retrieval = _retrieval(9e4)) catch e e end
        @test err isa ArgumentError && occursin("mol/mol", err.msg)
        # Errors may exceed the value (a low-signal retrieval), but not 1 mol/mol.
        @test _record(9; value_sigma = 1e-3, retrieval = _retrieval(9e4)).value_sigma == 1e-3
        @test _record(9; model_sigma = 0.5, retrieval = _retrieval(9e4)).model_sigma == 0.5
        @test_throws ArgumentError _record(9; value_sigma = 1.5, retrieval = _retrieval(9e4))
        @test_throws ArgumentError _record(9; model_sigma = 2.0, retrieval = _retrieval(9e4))
        # A file with blocks states mol/mol on the record values.
        NCDataset(path) do ds
            @test all(v -> ds[v].attrib["units"] == "mol mol-1", ("value", "value_sigma", "model_sigma"))
        end
        # Files without blocks: no units read as they are; a stated mole-fraction
        # unit is converted, and must be stated on all three.
        plain_path = AT.write_observations(joinpath(dir, "plain_ppb.nc"),
            AT.CSObservationSet([_record(1; kind = "point", intake_height_m = 10.0, value = 1.9e-6,
                                         value_sigma = 1e-8, model_sigma = 2e-9)]; time_origin = "2024-06-15 00:00:00"))
        NCDataset(plain_path) do ds
            @test !haskey(ds["value"].attrib, "units")
        end
        NCDataset(plain_path, "a") do ds
            _restate_values!(ds; value_units = Dict{String, Any}("value" => "ppb", "value_sigma" => "ppb",
                                                                 "model_sigma" => "ppb"), value_scale = 1e9)
        end
        got = only(AT.read_observations(plain_path).records)
        @test got.value ≈ 1.9e-6 && got.value_sigma ≈ 1e-8 && got.model_sigma ≈ 2e-9
        NCDataset(plain_path, "a") do ds
            delete!(ds["model_sigma"].attrib, "units")
        end
        err = try AT.read_observations(plain_path) catch e e end
        @test err isa ArgumentError && occursin("on all three or on none", err.msg)
    end
end

# A v2 file without a block, then the block variables written as given:
# `pressure` (interfaces × records), `weight`, `kernel`, `prior` (layers ×
# records), `surface`, optional `sigma`, with per-variable attributes.
function _write_block_file(path, records; pressure, weight, kernel, prior, surface, sigma = nothing,
                           attrib = Dict{String, Dict{String, Any}}(), skip = (), value_units = _MOLMOL,
                           value_scale = 1.0)
    AT.write_observations(path, AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00"))
    NCDataset(path, "a") do ds
        _restate_values!(ds; value_units, value_scale)
        defDim(ds, "retrieval_level", size(weight, 1))
        defDim(ds, "retrieval_interface", size(pressure, 1))
        default = Dict("retrieval_pressure_pa" => Dict{String, Any}("units" => "Pa", "ordering" => "surface_to_top"),
                       "retrieval_pressure_weight" => Dict{String, Any}("ordering" => "surface_to_top"),
                       "retrieval_averaging_kernel" => Dict{String, Any}("ordering" => "surface_to_top"),
                       "retrieval_prior_profile" => Dict{String, Any}("units" => "mol mol-1",
                                                                      "ordering" => "surface_to_top"),
                       "retrieval_profile_sigma" => Dict{String, Any}("units" => "mol mol-1",
                                                                    "ordering" => "surface_to_top"),
                       "retrieval_surface_pressure_pa" => Dict{String, Any}("units" => "Pa"))
        for (name, data, dims) in (("retrieval_pressure_pa", pressure, ("retrieval_interface", "obs")),
                                   ("retrieval_pressure_weight", weight, ("retrieval_level", "obs")),
                                   ("retrieval_averaging_kernel", kernel, ("retrieval_level", "obs")),
                                   ("retrieval_prior_profile", prior, ("retrieval_level", "obs")),
                                   ("retrieval_profile_sigma", sigma, ("retrieval_level", "obs")),
                                   ("retrieval_surface_pressure_pa", surface, ("obs",)))
            data === nothing && continue
            name in skip && continue
            attrs = merge(default[name], get(attrib, name, Dict{String, Any}()))
            filter!(kv -> kv.second !== nothing, attrs)
            defVar(ds, name, data, dims; attrib = attrs)
        end
    end
    return path
end

@testset "schema v2 retrieval block: units, ordering, padding, validation errors" begin
    rc = _retrieval(98000.0; n = 4)
    blockdata(c) = (; pressure = reshape(c.pressure_pa, :, 1), weight = reshape(c.pressure_weight, :, 1),
                    kernel = reshape(c.averaging_kernel, :, 1), prior = reshape(c.prior_profile, :, 1),
                    surface = [c.surface_pressure_pa])
    mktempdir() do dir
        file(name) = joinpath(dir, name)
        b = blockdata(rc)
        # Stated units are converted: hPa and ppm.
        path = _write_block_file(file("units.nc"), [_record(1)]; b..., pressure = b.pressure ./ 100,
                                 surface = b.surface ./ 100, prior = b.prior .* 1e6,
                                 attrib = Dict("retrieval_pressure_pa" => Dict{String, Any}("units" => "hPa"),
                                               "retrieval_surface_pressure_pa" => Dict{String, Any}("units" => "hPa"),
                                               "retrieval_prior_profile" => Dict{String, Any}("units" => "ppm")))
        got = only(AT.read_observations(path).records).retrieval
        @test got.pressure_pa ≈ rc.pressure_pa && got.prior_profile ≈ rc.prior_profile
        @test got.surface_pressure_pa ≈ rc.surface_pressure_pa

        failure(name, msg; kwargs...) = begin
            path = _write_block_file(file(name), [_record(1)]; b..., kwargs...)
            err = try AT.read_observations(path) catch e e end
            err isa ArgumentError && occursin(msg, sprint(showerror, err))
        end
        @test failure("nounits.nc", "must state its unit";
                      attrib = Dict("retrieval_prior_profile" => Dict{String, Any}("units" => nothing)))
        @test failure("order.nc", "ordering";
                      attrib = Dict("retrieval_averaging_kernel" => Dict{String, Any}("ordering" => nothing)))
        @test failure("topfirst.nc", "from the surface up"; pressure = reverse(b.pressure, dims = 1),
                      surface = [b.pressure[end]])
        @test failure("percent.nc", "dimensionless";
                      attrib = Dict("retrieval_pressure_weight" => Dict{String, Any}("units" => "%")))
        @test failure("partial.nc", "without"; skip = ("retrieval_prior_profile",))
        @test failure("sum.nc", "sum to 1"; weight = b.weight .* 1.01)
        hole = copy(b.kernel)
        hole[2] = NaN
        @test failure("hole.nc", "NaN after them"; kernel = hole)
        @test failure("molppm.nc", "ppm"; prior = b.prior .* 1e6)       # ppm labelled mol/mol
        # The record values: in ppm (converted), without units (rejected with a block).
        path = _write_block_file(file("valueppm.nc"), [_record(1; model_sigma = 0.5e-6)]; b..., value_scale = 1e6,
                                 value_units = Dict{String, Any}("value" => "ppm", "value_sigma" => "ppm",
                                                                 "model_sigma" => "umol/mol"))
        got = only(AT.read_observations(path).records)
        @test got.value ≈ 401e-6 && got.value_sigma ≈ 1e-6 && got.model_sigma ≈ 0.5e-6
        @test failure("valuenounits.nc", "must state a dry mole-fraction unit";
                      value_units = Dict{String, Any}("value" => "ppm", "value_sigma" => nothing,
                                                      "model_sigma" => "ppm"))
        @test failure("valueother.nc", "must state a dry mole-fraction unit";
                      value_units = Dict{String, Any}("value" => "kg m-2", "value_sigma" => "kg m-2",
                                                      "model_sigma" => "kg m-2"))
        # A low-signal retrieval, errors larger than the value (10 ± 20 ppb), is
        # legitimate and read as given.
        path = _write_block_file(file("lowsignal.nc"), [_record(1; model_sigma = 0.5e-6)]; b...)
        NCDataset(path, "a") do ds
            ds["value"][:] = [1e-8]
            ds["value_sigma"][:] = [2e-8]
            ds["model_sigma"][:] = [3e-8]
        end
        got = only(AT.read_observations(path).records)
        @test got.value == 1e-8 && got.value_sigma == 2e-8 && got.model_sigma == 3e-8
        # An error beyond 1 mol/mol cannot be a dry mole fraction: rejected.
        NCDataset(path, "a") do ds
            ds["value_sigma"][:] = [2.0]
        end
        err = try AT.read_observations(path) catch e e end
        @test err isa ArgumentError && occursin("at most 1 mol/mol", sprint(showerror, err))
        # A point record must not carry a block.
        path = _write_block_file(file("point.nc"), [_record(1; kind = "point", intake_height_m = 10.0)]; b...)
        @test_throws ArgumentError AT.read_observations(path)
        # NaN padding: a 3-layer record in a 4-layer block, and a record without one.
        rc3 = _retrieval(97000.0; n = 3)
        pad(v, n) = vcat(v, fill(NaN, n - length(v)))
        path = _write_block_file(file("pad.nc"), [_record(1), _record(2), _record(3)];
                                 pressure = hcat(rc.pressure_pa, pad(rc3.pressure_pa, 5), fill(NaN, 5)),
                                 weight = hcat(rc.pressure_weight, pad(rc3.pressure_weight, 4), fill(NaN, 4)),
                                 kernel = hcat(rc.averaging_kernel, pad(rc3.averaging_kernel, 4), fill(NaN, 4)),
                                 prior = hcat(rc.prior_profile, pad(rc3.prior_profile, 4), fill(NaN, 4)),
                                 surface = [rc.surface_pressure_pa, rc3.surface_pressure_pa, NaN])
        got = AT.read_observations(path).records
        @test got[1].retrieval == rc && got[2].retrieval == rc3 && got[3].retrieval === nothing
    end
end

# ---------------------------------------------------------------------------
# Binding and the column operator in the 4D-Var
# ---------------------------------------------------------------------------

_bind(P, records; kwargs...) =
    AT.bind_to_mesh(AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00"), P.mesh, T0, DT;
                    nsteps = P.nsteps, kwargs...)

function _controls(P; scale = 1.0)
    value(phase) = ntuple(p -> [P.FT(scale * 5e8 * (1 + 0.3sin(phase + i + 2j + p))) for i in 1:P.Nc, j in 1:P.Nc], 6)
    return [AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:all, 1:P.nsteps), value(0.0)),
            AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:late, 2:P.nsteps; normalize = true), value(1.3))]
end

_fourdvar(P, observations, controls; scheme = AT.UpwindScheme(), met = column_met(P), kwargs...) =
    AT.cs_surface_flux_4dvar(P.rm, P.m, P.am, P.bm, P.cm, P.mesh, observations, controls;
                             scheme, dt = DT, column_met = met, kwargs...)

function _site(P; panel = 2, i = 2, j = 3)
    lons, lats = AT.panel_cell_center_lonlat(P.mesh, panel)
    return (lat = lats[i, j], lon = lons[i, j])
end

_relerr(g, ref) = maximum(maximum(abs.(g[c][p] .- ref[c][p])) for c in eachindex(ref), p in 1:6) /
                  maximum(maximum(abs.(ref[c][p])) for c in eachindex(ref), p in 1:6)

@testset "binding: retrieval records become averaging-kernel columns" begin
    P = hydrostatic_cs_problem()
    site = _site(P)
    σ = fill(2e-6, 6)
    rc = _retrieval(99000.0; sigma = σ)
    obs = _bind(P, [_record(1; site...), _record(2; site..., retrieval = rc, model_sigma = 0.4e-6)];
                max_surface_pressure_mismatch_pa = 3000.0,
                placement = O.VerticalPlacementSpec(; unplaced = O.SkipUnplaced()))
    @test obs[1].objective isa AT.CSColumnMeanObjective
    o = obs[2].objective
    @test o isa AT.CSRetrievalColumnObjective && o.retrieval == rc
    @test (o.panel, o.i, o.j) == (2, 2, 3) && o.area == AT.cell_area(P.mesh, 2, 3)
    @test o.pressure_basis isa O.TotalFromHumidity && o.mismatch_policy isa O.SkipUnplaced
    @test o.max_surface_pressure_mismatch_pa == 3000.0
    # σ² = value_sigma² + model_sigma² + Σ (h a σ)², fixed by the record.
    @test obs[2].sigma ≈ sqrt(1e-12 + 0.16e-12 + sum((rc.pressure_weight .* rc.averaging_kernel .* σ) .^ 2))
    @test obs[1].sigma == 1e-6
    for bad in (0.0, -1.0, NaN)
        @test_throws ArgumentError _bind(P, [_record(2; site..., retrieval = rc)];
                                         max_surface_pressure_mismatch_pa = bad)
    end
    # A retrieval objective has no final-time footprint, and needs the column meteorology.
    @test_throws ArgumentError AT.cs_surface_emission_footprint(P.rm, P.m, P.am, P.bm, P.cm, P.mesh, o;
                                                                scheme = AT.UpwindScheme(), dt = DT)
    err = try _fourdvar(P, obs, _controls(P); met = nothing) catch e e end
    @test err isa ArgumentError && occursin("column_met", sprint(showerror, err))
    err = try _fourdvar(P, obs, _controls(P); met = column_met(P; humidity = nothing)) catch e e end
    @test err isa ArgumentError && occursin("pressure_basis = \"dry\"", sprint(showerror, err))
end

@testset "column-mean equivalence: a = 1, h = Δp/Δp_column on the model's levels" begin
    # No transport: the air mass stays hydrostatic, so the model's interfaces
    # are known; the surface emissions change the tracer.
    P = hydrostatic_cs_problem(; flux_fraction = 0.0, nsteps = 2)
    site = _site(P)
    area = AT.cell_area(P.mesh, 2, 3)
    nz = P.Nz
    total = reverse(COLUMN_P_HALF)                                      # surface first
    dry = reverse(O.interface_pressures!(zeros(nz + 1), P.m[2][P.mesh.Hp + 2, P.mesh.Hp + 3, :], area,
                                         COLUMN_GRAVITY, P.p_top))
    model_levels(edges) = O.RetrievalColumn(; pressure_pa = edges,
                                            pressure_weight = -diff(edges) ./ (edges[1] - edges[end]),
                                            averaging_kernel = ones(nz), prior_profile = fill(0.5, nz))
    # A coarser retrieval grid 3 % below the model surface, also a = 1.
    coarse = _retrieval(0.97 * total[1]; n = 5, ak = ones(5))
    for step_time in (450.0, 1000.0)
        records = [_record(1; site..., t = step_time, retrieval = model_levels(total)),
                   _record(2; site..., t = step_time, retrieval = model_levels(dry)),
                   _record(3; site..., t = step_time),
                   _record(4; site..., t = step_time, retrieval = coarse),
                   (_record(10 + k; site..., t = step_time, kind = "point", pressure_pa = (total[nz + 2 - k] +
                                                                                       total[nz + 1 - k]) / 2)
                    for k in 1:nz)...]
        obs_total = _bind(P, records)
        obs_dry = _bind(P, records; placement = O.VerticalPlacementSpec(; pressure_basis = O.DryInterfaces()))
        r_total = _fourdvar(P, obs_total, _controls(P))
        r_dry = _fourdvar(P, obs_dry, _controls(P))
        x = r_total.simulated[5:end]                                    # layer k = 1 … nz (top first)
        @test [s.levels[1] for s in r_total.stencils[5:end]] == 1:nz
        Δp = diff(COLUMN_P_HALF)
        # Total pressures: the pressure-weighted column mean.
        @test r_total.simulated[1] ≈ dot(Δp, x) / sum(Δp) rtol = 1e-13
        # Dry interfaces: the air-mass-weighted column mean (CSColumnMeanObjective).
        @test r_dry.simulated[2] ≈ r_dry.simulated[3] rtol = 1e-13
        # Any retrieval grid and surface pressure, a = 1: the same mean.
        @test r_total.simulated[4] ≈ r_total.simulated[1] rtol = 1e-13
        @test r_total.stencils[1].placement_method == AT.RETRIEVAL_COLUMN_PLACEMENT_CODE
        @test r_total.stencils[1].levels == [0]
        @test abs(r_total.stencils[1].surface_pressure_mismatch_pa) < 1e-6
        @test r_total.stencils[4].surface_pressure_mismatch_pa ≈ 0.03 * total[1] rtol = 1e-10
        @test r_total.stencils[1].retrieval_profile ≈ reverse(x) rtol = 1e-13
    end
end

# ---------------------------------------------------------------------------
# Single sweep vs the per-observation reference
# ---------------------------------------------------------------------------

# The air mass and tracer mass after every step of the production step plan,
# replayed with the emissions of `controls`.
_layers(P, ks, v) = ntuple(_ -> (e = zeros(P.FT, P.Nc, P.Nc, P.Nz); e[:, :, ks] .= P.FT(v); e), 6)

# Averaging-kernel columns (instantaneous and windowed, model surface above and
# below the retrieval's, retrieval top above and below the model top, with a
# profile uncertainty), a column mean and a placed point.


@testset "single sweep == per-observation reference for placed, window, layer and level columns" begin
    P = hydrostatic_cs_problem()
    observations = _mixed_observations(P)
    controls = _controls(P)
    tm5(P) = (; convection_op = AT.TM5Convection(),
                convection_forcing = AT.ConvectionForcing(nothing, nothing,
                    (; entu = _layers(P, 5:8, 0.4), detu = _layers(P, 5:8, 0.28),
                       entd = _layers(P, 6:8, 0.12), detd = _layers(P, 6:8, 0.06))),
                convection_workspace = AT.TM5Workspace(P.m; tile_columns = P.Nc^2,
                    cell_metrics = ntuple(_ -> [AT.cell_area(P.mesh, i, j) for i in 1:P.Nc, j in 1:P.Nc], 6)))
    @testset "$(nameof(typeof(scheme))) $(label)" for (scheme, label, physics) in (
            (AT.UpwindScheme(), "no physics", P -> (;)),
            (AT.PPMScheme(AT.MonotoneLimiter()), "no physics", P -> (;)),
            (AT.LinRoodPPMScheme(), "no physics", P -> (;)),
            (AT.UpwindScheme(), "TM5 convection", tm5))
        full = _fourdvar(P, observations, controls; scheme, physics(P)...)
        @test all(s -> s.assimilated, full.stencils)
        @test full.stencils[2].steps == [1, 2, 3] && length(full.stencils[3].steps) == 2
        ref = _reference(P, observations, full.stencils, controls; scheme, physics)
        @test full.simulated ≈ ref.simulated rtol = 1e-14
        @test _relerr(full.gradients, ref.gradients) <= 1e-12
        # The time-weighted diagnostics, with pressures that change from step to step.
        for idx in (1, 2, 3, 11, 12, 13, 14)
            @test full.stencils[idx].surface_pressure_mismatch_pa ≈ ref.mismatch[idx] rtol = 1e-12
            @test full.stencils[idx].retrieval_profile ≈ ref.profiles[idx] rtol = 1e-13
        end
        @test observations[11].objective.retrieval.mapping isa O.HatLevelProjection
        @test observations[14].objective.retrieval.mapping isa O.LayerRebin
        @test length(full.stencils[11].retrieval_profile) == 20
        @test length(full.stencils[14].retrieval_profile) == 19
        @test isnan(full.stencils[4].surface_pressure_mismatch_pa) && isempty(full.stencils[5].retrieval_profile)
        stride = _fourdvar(P, observations, controls; scheme, checkpoint = AT.StrideCheckpoint(2), physics(P)...)
        @test stride.simulated == full.simulated
        @test _relerr(stride.gradients, full.gradients) <= 1e-12
    end
end

@testset "Float32: single sweep == per-observation reference" begin
    P = hydrostatic_cs_problem(; FT = Float32)
    observations = _mixed_observations(P)
    controls = _controls(P)
    full = _fourdvar(P, observations, controls)
    @test eltype(full.simulated) == Float32
    ref = _reference(P, observations, full.stencils, controls; scheme = AT.UpwindScheme(), physics = P -> (;))
    @test full.simulated ≈ ref.simulated rtol = 1e-6
    @test _relerr(full.gradients, ref.gradients) <= 1e-5
    # Close to the Float64 values.
    P64 = hydrostatic_cs_problem()
    full64 = _fourdvar(P64, _mixed_observations(P64), _controls(P64))
    @test Float64.(full.simulated) ≈ full64.simulated rtol = 1e-5
end

@testset "per-step humidity: each sampled step uses its own total pressures" begin
    # A moister second step lowers the dry air mass share of the total
    # pressure there: the column at step 2 must use step 2's humidity.
    P = hydrostatic_cs_problem(; flux_fraction = 0.0, nsteps = 2)
    site = _site(P)
    moist = ntuple(q -> P.qpanels[q] .* 1.5, 6)
    record = [_record(1; site..., t = 900.0, retrieval = _retrieval(98000.0; n = 5, ak = [0.5, 0.8, 1.0, 1.2, 1.4]))]
    obs = _bind(P, record)
    stepwise = _fourdvar(P, obs, _controls(P); met = column_met(P; humidity = [P.qpanels, moist]))
    static_moist = _fourdvar(P, obs, _controls(P); met = column_met(P; humidity = moist))
    static_first = _fourdvar(P, obs, _controls(P))
    @test stepwise.simulated == static_moist.simulated
    @test stepwise.stencils[1].surface_pressure_mismatch_pa == static_moist.stencils[1].surface_pressure_mismatch_pa
    @test stepwise.stencils[1].surface_pressure_mismatch_pa > static_first.stencils[1].surface_pressure_mismatch_pa
    @test stepwise.simulated != static_first.simulated
end

@testset "central finite differences of the cost match the gradient (with windows)" begin
    P = hydrostatic_cs_problem()
    observations = _mixed_observations(P)
    @test count(o -> o.objective isa AT.CSRetrievalColumnObjective && length(o.sampling.steps) > 1,
                observations) == 4
    controls = _controls(P)
    direction = _controls(P; scale = 0.37)
    shifted(ε) = [AT.CSSurfaceFluxControl(c.window, ntuple(q -> c.value[q] .+ ε .* d.value[q], 6))
                  for (c, d) in zip(controls, direction)]
    for (scheme, tol) in ((AT.UpwindScheme(), 1e-8), (AT.PPMScheme(AT.MonotoneLimiter()), 1e-6))
        base = _fourdvar(P, observations, controls; scheme)
        directional = sum(sum(sum(g[q] .* d.value[q]) for q in 1:6) for (g, d) in zip(base.gradients, direction))
        ε = 1e-2
        fd = (_fourdvar(P, observations, shifted(ε); scheme).cost -
              _fourdvar(P, observations, shifted(-ε); scheme).cost) / 2ε
        @test abs(fd - directional) <= tol * abs(directional)
        @test base.observation_cost > 1
        # The cost uses the frozen σ of the binding (model-error term included).
        @test base.observation_cost ≈ sum(0.5 * (r / o.sigma)^2 for (r, o) in zip(base.residuals, observations)) rtol = 1e-14
    end
end

# ---------------------------------------------------------------------------
# Surface-pressure mismatch policy
# ---------------------------------------------------------------------------

@testset "surface-pressure mismatch: the stretch absorbs it, or reject / skip" begin
    P = hydrostatic_cs_problem(; flux_fraction = 0.0, nsteps = 2)
    site = _site(P)
    ps = COLUMN_P_HALF[end]                                     # the model's total surface pressure
    records = [_record(1; site..., retrieval = _retrieval(ps - 3000.0), value = 402e-6),   # Δps = +3000 Pa
               _record(2; site..., retrieval = _retrieval(ps - 1000.0), value = 401e-6),   # Δps = +1000 Pa
               _record(3; site..., t = 900.0, kind = "point", intake_height_m = 10.0, value = 420e-6)]
    controls = _controls(P)
    # Default: no limit; the stretch absorbs both.
    free = _fourdvar(P, _bind(P, records), controls)
    @test all(s -> s.assimilated, free.stencils)
    @test free.stencils[1].surface_pressure_mismatch_pa ≈ 3000 rtol = 1e-9
    @test free.stencils[2].surface_pressure_mismatch_pa ≈ 1000 rtol = 1e-9
    # Reject (unplaced_records = "error", the default).
    err = try _fourdvar(P, _bind(P, records; max_surface_pressure_mismatch_pa = 2000.0), controls) catch e e end
    @test err isa ArgumentError
    @test occursin("id 1", err.msg) && !occursin("id 2", err.msg) && occursin("max_surface_pressure_mismatch_pa", err.msg)
    # The screen rejects after the forward run, which recorded into the tape:
    # an mmap tape is released with the error (unmapped and closed, no GC).
    mktempdir() do root
        dir = joinpath(root, "tape")
        err = try
            _fourdvar(P, _bind(P, records; max_surface_pressure_mismatch_pa = 2000.0), controls;
                      tape_storage = AT.MmapCSTapeStorage(dir))
        catch e
            e
        end
        @test err isa ArgumentError && occursin("max_surface_pressure_mismatch_pa", err.msg)
        @test isdir(dir)                                   # the tape was opened before the screen
        Sys.islinux() && @test count(line -> occursin(realpath(dir), line), eachline("/proc/self/maps")) == 0
        @test isempty(readdir(dir))
    end
    # Skip: reported, not assimilated, counted.
    skip = O.VerticalPlacementSpec(; unplaced = O.SkipUnplaced())
    screened = _bind(P, records; max_surface_pressure_mismatch_pa = 2000.0, placement = skip)
    r = with_logger(NullLogger()) do
        _fourdvar(P, screened, controls)
    end
    @test [s.assimilated for s in r.stencils] == [false, true, true]
    @test r.simulated == free.simulated
    kept = _fourdvar(P, screened[2:3], controls)
    @test r.observation_cost == kept.observation_cost
    @test gradients_match(r.gradients, kept.gradients)        # bit for bit single-threaded (helper)
    @test r.residuals[1] == r.simulated[1] - screened[1].value
    # The count is logged.
    @test_logs (:info, r"1 averaging-kernel column observations are not assimilated") match_mode = :any _fourdvar(P, screened, controls)
    # Every observation screened: zero observation cost and gradient.
    none = _fourdvar(P, screened[1:1], controls)
    @test none.observation_cost == 0 && all(iszero(none.gradients[c][q]) for c in eachindex(controls), q in 1:6)
end

# ---------------------------------------------------------------------------
# Departures v2
# ---------------------------------------------------------------------------

@testset "departures v2: placement code, surface-pressure mismatch, retrieval profile" begin
    P = hydrostatic_cs_problem(; flux_fraction = 0.0, nsteps = 3)
    site = _site(P)
    ps = COLUMN_P_HALF[end]
    σ = fill(2e-6, 6)
    records = [_record(1; site..., retrieval = _retrieval(ps - 3000.0), value = 402e-6),
               _record(2; site..., t = 1000.0, time_window_s = 1200.0, value = 401e-6,
                       retrieval = _retrieval(ps + 500.0; sigma = σ), model_sigma = 0.2e-6),
               _record(3; site..., value = 403e-6),
               _record(4; site..., kind = "point", intake_height_m = 10.0, value = 420e-6)]
    set = AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00")
    obs = AT.bind_to_mesh(set, P.mesh, T0, DT; nsteps = P.nsteps, max_surface_pressure_mismatch_pa = 2000.0,
                          placement = O.VerticalPlacementSpec(; unplaced = O.SkipUnplaced()))
    result = with_logger(NullLogger()) do
        _fourdvar(P, obs, _controls(P))
    end
    deps = AT.build_departure_set(set, result, P.mesh, "2024-06-15 00:00:00", DT, P.nsteps)
    @test [d.vertical_placement for d in deps.records] ==
          [AT.RETRIEVAL_COLUMN_PLACEMENT_CODE, AT.RETRIEVAL_COLUMN_PLACEMENT_CODE, AT.COLUMN_MEAN_PLACEMENT_CODE,
           O.VERTICAL_PLACEMENT_CODES.record_height]
    @test [d.assimilated for d in deps.records] == [false, true, true, true]
    @test deps[1].surface_pressure_mismatch_pa ≈ 3000 rtol = 1e-9
    @test deps[2].surface_pressure_mismatch_pa ≈ -500 rtol = 1e-9
    @test deps[2].retrieval_profile == result.stencils[2].retrieval_profile && length(deps[2].retrieval_profile) == 6
    @test isnan(deps[3].surface_pressure_mismatch_pa) && isempty(deps[3].retrieval_profile)
    profile_sigma = sqrt(sum((records[2].retrieval.pressure_weight .* records[2].retrieval.averaging_kernel .* σ) .^ 2))
    @test deps[2].profile_model_sigma ≈ profile_sigma
    @test deps[2].normalized_departure ≈ deps[2].departure / sqrt(1e-12 + 0.04e-12 + profile_sigma^2)
    mktempdir() do dir
        path = AT.write_departures(joinpath(dir, "deps.nc"), deps)
        NCDataset(path) do ds
            @test ds.dim["retrieval_level"] == 6
            @test ds["assimilated"][:] == Int8[0, 1, 1, 1]
            @test occursin("-3 = retrieval_column", ds.attrib["vertical_placement_codes"])
        end
        back = AT.read_departures(path)
        for (a, b) in zip(deps.records, back.records)
            @test all(f -> isequal(getfield(a, f), getfield(b, f)), fieldnames(AT.CSDepartureRecord))
        end
        # A file without averaging-kernel rows has no retrieval block.
        plain = AT.CSDepartureSet(deps.records[3:4]; mesh_Nc = deps.mesh_Nc,
                                  mesh_panel_convention = deps.mesh_panel_convention,
                                  mesh_cs_definition_tag = deps.mesh_cs_definition_tag, t_start = deps.t_start,
                                  dt_seconds = deps.dt_seconds, nsteps = deps.nsteps)
        path = AT.write_departures(joinpath(dir, "plain.nc"), plain)
        NCDataset(path) do ds
            @test !haskey(ds, "assimilated") && !haskey(ds.dim, "retrieval_level")
        end
        @test all(r -> r.assimilated && isempty(r.retrieval_profile), AT.read_departures(path).records)
        # The reader checks the profile's ordering and units and the mismatch's
        # units, converting ppm and hPa.
        restated = AT.write_departures(joinpath(dir, "restated.nc"), deps)
        NCDataset(restated, "a") do ds
            ds["retrieval_model_profile"].attrib["units"] = "ppm"
            ds["retrieval_model_profile"][:, :] = ds["retrieval_model_profile"][:, :] .* 1e6
            ds["surface_pressure_mismatch_pa"].attrib["units"] = "hPa"
            ds["surface_pressure_mismatch_pa"][:] = ds["surface_pressure_mismatch_pa"][:] ./ 100
        end
        again = AT.read_departures(restated)
        @test again[2].retrieval_profile ≈ deps[2].retrieval_profile rtol = 1e-14
        @test again[1].surface_pressure_mismatch_pa ≈ deps[1].surface_pressure_mismatch_pa rtol = 1e-14
        for (attr, value, msg) in (("ordering", "top_to_surface", "ordering"), ("units", "kg", "units"))
            bad = AT.write_departures(joinpath(dir, "bad_$(attr).nc"), deps)
            NCDataset(bad, "a") do ds
                ds["retrieval_model_profile"].attrib[attr] = value
            end
            err = try AT.read_departures(bad) catch e e end
            @test err isa ArgumentError && occursin(msg, sprint(showerror, err))
        end
        bad = AT.write_departures(joinpath(dir, "bad_mismatch.nc"), deps)
        NCDataset(bad, "a") do ds
            ds["surface_pressure_mismatch_pa"].attrib["units"] = "m"
        end
        @test_throws ArgumentError AT.read_departures(bad)
    end
    # Row checks: only averaging-kernel rows carry the diagnostics.
    row = deps[3]
    kw = Dict(f => getfield(row, f) for f in fieldnames(AT.CSDepartureRecord))
    @test_throws ArgumentError AT.CSDepartureRecord(; kw..., assimilated = false)
    @test_throws ArgumentError AT.CSDepartureRecord(; kw..., surface_pressure_mismatch_pa = 1.0)
    kw2 = Dict(f => getfield(deps[2], f) for f in fieldnames(AT.CSDepartureRecord))
    @test_throws ArgumentError AT.CSDepartureRecord(; kw2..., retrieval_profile = Float64[])
    @test_throws ArgumentError AT.CSDepartureRecord(; kw2..., profile_model_sigma = 0.0)   # σ no longer matches
    # The record's block must be the one the observation was bound with.
    other = AT.CSObservationSet([records[1:1]; _record(2; site..., t = 1000.0, time_window_s = 1200.0,
                                                       value = 401e-6, retrieval = _retrieval(ps), model_sigma = 0.2e-6);
                                 records[3:4]]; time_origin = "2024-06-15 00:00:00")
    @test_throws ArgumentError AT.build_departure_set(other, result, P.mesh, "2024-06-15 00:00:00", DT, P.nsteps)
end
