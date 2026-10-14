#!/usr/bin/env julia
"""
`ec2tm_from_rates!` (ERA5 → TM5 convection, issue #54) and the hydrostatic
layer thickness.

The conversion is checked against an independent transcription of TM5's own
code (`test/helpers/tm5_ecconv_reference.jl`: `tmm.F90` packing,
`phys_convec_ec2tm.F90` `ECconv_to_TMconv`, `meteo.F90` column closure), not
against a copy of the Julia loop. ERA5 archives UDMF/DDMF level k at the TOP of
layer k with no flux through the ground, so the interface vector the
preprocessor passes is `[GRIB levels 1..Nz; 0]`, and the highest layer carrying
updraft is the one ABOVE the first interface with flux.

1. Known answers on an exactly representable column: TM5's fields, the input
   fluxes recovered from them, and no closure.
2. Agreement with the TM5 transcription on the upstream 6-level column and on
   random deep columns.
3. Clipping, negative redistribution, the column closure (now only for a flux
   through the model top), and shape and boundary guards.
4. `dz_hydrostatic_virtual!` / `dz_hydrostatic_constT!`.

All tests run core (no real data).
"""

using Test
using Random

using AtmosTransport

using .AtmosTransport.Preprocessing: ec2tm!, ec2tm_from_rates!,
                                       TM5CleanupStats,
                                       dz_hydrostatic_virtual!,
                                       dz_hydrostatic_constT!

include(joinpath(@__DIR__, "..", "helpers", "tm5_ecconv_reference.jl"))

# Run ec2tm_from_rates! on GRIB-level columns packed as the preprocessor packs
# them: interface k = top of layer k, interface Nz + 1 (the ground) = 0.
function _ec2tm_grib(grib_udmf, grib_ddmf, udrf, ddrf, dz; stats = nothing)
    Nz = length(dz)
    out = ntuple(_ -> zeros(Nz), 4)
    ec2tm_from_rates!(out..., vcat(grib_udmf, 0.0), vcat(grib_ddmf, 0.0),
                      copy(udrf), copy(ddrf), copy(dz), Nz; stats)
    return out
end

# ─────────────────────────────────────────────────────────────────────────
# Known answers. Every value is a dyadic fraction, so all sums and products
# are exact and the column closure must not fire.
#
#   interface (top of layer k)    1   2      3      4       5      6      7
#   updraft   udmf                0   0   1/32   1/16   3/64   1/64      0
#   downdraft ddmf                0   0      0  -1/128 -1/64  -1/128     0
#
# The updraft reaches the top of layer 2 (interface 3), so layer 2 is the
# cloud-top layer and detrains the 1/32 that enters it; GRIB level 6 (the top
# of the lowest layer) carries flux. The downdraft starts in layer 3.
# ─────────────────────────────────────────────────────────────────────────
const _KA_DZ        = [256.0, 256.0, 256.0, 128.0, 128.0, 64.0]
const _KA_GRIB_UDMF = [0.0, 0.0, 1/32, 1/16, 3/64, 1/64]
const _KA_GRIB_DDMF = [0.0, 0.0, 0.0, -1/128, -1/64, -1/128]
const _KA_DETU      = [0.0, 1/32, 3/64, 1/64, 1/128, 0.0]        # kg m⁻² s⁻¹
const _KA_DETD      = [0.0, 0.0, 0.0, 0.0, 1/128, 1/128]
const _KA_UDRF      = _KA_DETU ./ _KA_DZ                            # kg m⁻³ s⁻¹
const _KA_DDRF      = _KA_DETD ./ _KA_DZ

@testset "ec2tm_from_rates! known answer (TM5 half-level convention)" begin
    stats = TM5CleanupStats()
    entu, detu, entd, detd = _ec2tm_grib(_KA_GRIB_UDMF, _KA_GRIB_DDMF,
                                         _KA_UDRF, _KA_DDRF, _KA_DZ; stats)
    # TM5's fields, by hand: entu = udmf[k] - udmf[k+1] + detu (top minus bottom).
    @test entu == [0.0, 0.0, 1/64, 1/32, 5/128, 1/64]
    @test detu == _KA_DETU                  # including the cloud-top layer 2
    @test entd == [0.0, 0.0, 1/128, 1/128, 0.0, 0.0]
    @test detd == _KA_DETD
    # The fluxes rebuilt from the fields are the input, through every interface.
    @test implied_updraft_flux(entu, detu) == vcat(_KA_GRIB_UDMF, 0.0)
    @test implied_downdraft_flux(entd, detd) == vcat(_KA_GRIB_DDMF, 0.0)
    # Consistent input: the closure has nothing to move.
    @test stats.columns_closure_entu[] == 0
    @test stats.columns_closure_entd[] == 0
    @test stats.levels_entu_neg[] == 0 && stats.levels_detu_neg[] == 0
    # Same answer as the independent TM5 transcription, bit for bit.
    @test (entu, detu, entd, detd) ==
          tm5_ecconv_reference(_KA_GRIB_UDMF, _KA_GRIB_DDMF, _KA_UDRF, _KA_DDRF, _KA_DZ)
end

# The upstream column (issue #54, Part 1): interface fluxes given directly.
const _NZ_REF     = 6
const _UDMF_REF   = Float64[0.00, 0.00, 0.02, 0.05, 0.03, 0.00, 0.00]
const _DDMF_REF   = Float64[0.00, 0.00, 0.00, -0.01, -0.02, 0.00, 0.00]
const _UDRF_RATE  = Float64[0.0, 0.0, 1.0e-4, 2.0e-4, 1.0e-4, 0.0]
const _DDRF_RATE  = Float64[0.0, 0.0, 0.0, 5.0e-5, 1.0e-4, 0.0]
const _DZ_REF     = Float64[200.0, 200.0, 200.0, 150.0, 120.0, 80.0]

@testset "ec2tm_from_rates! matches TM5 on the upstream column" begin
    Nz = _NZ_REF
    out = ntuple(_ -> zeros(Nz), 4)
    ec2tm_from_rates!(out..., copy(_UDMF_REF), copy(_DDMF_REF),
                      copy(_UDRF_RATE), copy(_DDRF_RATE), copy(_DZ_REF), Nz)
    entu, detu, entd, detd = out
    ref = tm5_ecconv_reference(_UDMF_REF[1:Nz], _DDMF_REF[1:Nz],
                               _UDRF_RATE, _DDRF_RATE, _DZ_REF)
    scale = maximum(abs, _UDMF_REF)
    for (field, expected) in zip(out, ref)
        @test field ≈ expected atol = 1e-15
    end
    # Layer 2 is the cloud-top layer: it detrains the 0.02 entering from below.
    @test detu[2] ≈ 0.02 atol = 1e-15
    @test detu[1] == 0.0
    @test implied_updraft_flux(entu, detu) ≈ _UDMF_REF atol = 1e-15
    @test implied_downdraft_flux(entd, detd) ≈ _DDMF_REF atol = 1e-15
    @test all(>=(0), entu) && all(>=(0), detu) && all(>=(0), entd) && all(>=(0), detd)
end

@testset "ec2tm_from_rates! matches TM5 on random deep columns" begin
    rng = Xoshiro(54)
    Nz = 40
    worst_field = 0.0
    worst_flux = 0.0
    for column in 1:500
        top, base = rand(rng, 4:20), rand(rng, 30:Nz)
        # Updraft through interfaces top+1..Nz (GRIB level Nz included), zero
        # at and above interface `top` and at the ground.
        grib_udmf = zeros(Nz)
        for k in (top + 1):Nz
            grib_udmf[k] = 0.05 * rand(rng) * (k <= base ? 1.0 : (Nz + 1 - k) / (Nz + 1 - base))
        end
        grib_udmf[grib_udmf .< 1e-4] .= 0.0
        grib_udmf[top + 1] = max(grib_udmf[top + 1], 1e-3)
        dtop = rand(rng, (top + 2):Nz)
        grib_ddmf = [k > dtop ? -0.01 * rand(rng) - 1e-4 : 0.0 for k in 1:Nz]
        udrf = [k >= top ? 2e-5 * rand(rng) : 0.0 for k in 1:Nz]
        ddrf = [k >= dtop ? 1e-5 * rand(rng) : 0.0 for k in 1:Nz]
        dz = 100.0 .+ 400.0 .* rand(rng, Nz)
        out = _ec2tm_grib(grib_udmf, grib_ddmf, udrf, ddrf, dz)
        ref = tm5_ecconv_reference(grib_udmf, grib_ddmf, udrf, ddrf, dz)
        scale = sum(abs, grib_udmf) + sum(abs, grib_ddmf)     # the closure's scale
        worst_field = max(worst_field,
                          maximum(maximum(abs, a .- b) for (a, b) in zip(out, ref)) / scale)
        worst_flux = max(worst_flux,
            maximum(abs, implied_updraft_flux(out[1], out[2]) .- vcat(grib_udmf, 0.0)) / scale,
            maximum(abs, implied_downdraft_flux(out[3], out[4]) .- vcat(grib_ddmf, 0.0)) / scale)
        # The cloud-top layer (above the first interface with flux) detrains.
        @test out[2][top] > 0
        @test all(iszero, out[1][1:top-1]) && all(iszero, out[2][1:top-1])
    end
    @test worst_field < 16eps()
    @test worst_flux < 16eps()
end

@testset "ec2tm_from_rates! zero forcing" begin
    Nz = 5
    entu = zeros(Float64, Nz); detu = zeros(Float64, Nz)
    entd = zeros(Float64, Nz); detd = zeros(Float64, Nz)
    udmf = zeros(Float64, Nz + 1); ddmf = zeros(Float64, Nz + 1)
    udrf = zeros(Float64, Nz); ddrf = zeros(Float64, Nz)
    dz   = fill(100.0, Nz)

    ec2tm_from_rates!(entu, detu, entd, detd,
                       udmf, ddmf, udrf, ddrf, dz, Nz)
    @test all(entu .== 0)
    @test all(detu .== 0)
    @test all(entd .== 0)
    @test all(detd .== 0)
end

@testset "ec2tm_from_rates! bad-data cleanup" begin
    Nz = 5
    entu = zeros(Float64, Nz); detu = zeros(Float64, Nz)
    entd = zeros(Float64, Nz); detd = zeros(Float64, Nz)

    # Sub-threshold updraft noise: should all get clipped to 0 ⇒
    # no updraft at all ⇒ all outputs zero.
    udmf = fill(0.5e-6, Nz + 1)     # all < 1e-6 → zeroed
    ddmf = fill(-0.5e-6, Nz + 1)    # all > -1e-6 → zeroed
    udrf = fill(0.5e-10, Nz)        # all < 1e-10 → zeroed
    ddrf = fill(0.5e-10, Nz)
    dz   = fill(100.0, Nz)

    stats = TM5CleanupStats()
    ec2tm_from_rates!(entu, detu, entd, detd,
                       udmf, ddmf, udrf, ddrf, dz, Nz; stats = stats)

    @test all(entu .== 0) && all(detu .== 0)
    @test all(entd .== 0) && all(detd .== 0)
    @test stats.levels_udmf_clipped[] == Nz + 1
    @test stats.levels_ddmf_clipped[] == Nz + 1
    @test stats.levels_udrf_clipped[] == Nz
    @test stats.levels_ddrf_clipped[] == Nz
    @test stats.no_updraft[]    == 1
    @test stats.no_downdraft[]  == 1
    @test stats.columns_processed[] == 1
end

@testset "ec2tm_from_rates! negative redistribution" begin
    # The flux grows upward from interface 3 (0.05) to interface 2 (0.03 is
    # smaller, so layer 2 must detrain 0.02 more than its small rate gives):
    # the budget makes entu[1] and entu[2] negative, and TM5 moves the
    # magnitude to detu, keeping entu - detu.
    Nz = 4
    grib_udmf = Float64[0.0, 0.03, 0.05, 0.0]
    grib_ddmf = zeros(Float64, Nz)
    udrf = Float64[0.0, 1.0e-5, 5.0e-4, 0.0]   # kg/m³/s, small in layer 2
    ddrf = zeros(Float64, Nz)
    dz   = fill(200.0, Nz)

    stats = TM5CleanupStats()
    entu, detu, entd, detd = _ec2tm_grib(grib_udmf, grib_ddmf, udrf, ddrf, dz; stats)
    ref_entu, ref_detu, _, _ = tm5_ecconv_reference(grib_udmf, grib_ddmf, udrf, ddrf, dz)

    @test all(entu .>= 0)
    @test all(detu .>= 0)
    @test stats.levels_entu_neg[] == 2
    @test entu ≈ ref_entu atol = 1e-15
    @test detu ≈ ref_detu atol = 1e-15
    # The budget holds in every layer: no layer absorbs a closure residual.
    udmf = vcat(grib_udmf, 0.0)
    for k in 1:Nz
        @test entu[k] - detu[k] ≈ udmf[k] - udmf[k + 1] atol = 1e-15
    end
    @test detu[1] ≈ 0.03 atol = 1e-15       # cloud-top layer
end

@testset "TM5 closure absorbs a flux through the model top" begin
    # Consistent ERA5 input has no flux through the model top, and its column
    # budget closes by itself. TM5 does not check interface 1, though; a flux
    # there leaves Σentu - Σdetu = udmf[1], which the meteo.F90 closure moves
    # to the layer of largest entrainment (on the detrainment side when it is
    # positive, the documented deviation that keeps entu non-negative).
    Nz = 4
    grib_udmf = Float64[0.01, 0.03, 0.05, 0.02]
    grib_ddmf = Float64[-0.005, -0.01, -0.03, -0.02]
    udrf = Float64[0.0, 5.0e-5, 5.0e-5, 5.0e-5]
    ddrf = Float64[0.0, 5.0e-5, 5.0e-5, 5.0e-5]
    dz   = fill(200.0, Nz)

    stats = TM5CleanupStats()
    entu, detu, entd, detd = _ec2tm_grib(grib_udmf, grib_ddmf, udrf, ddrf, dz; stats)

    @test isapprox(sum(entu), sum(detu); atol = 1e-15)
    @test isapprox(sum(entd), sum(detd); atol = 1e-15)
    @test all(entu .>= 0) && all(detu .>= 0) && all(entd .>= 0) && all(detd .>= 0)
    @test stats.columns_closure_entu[] == 1
    @test stats.columns_closure_entd[] == 1
    ref = tm5_ecconv_reference(grib_udmf, grib_ddmf, udrf, ddrf, dz)
    for (field, expected) in zip((entu, detu, entd, detd), ref)
        @test field ≈ expected atol = 1e-15
    end
    # The updraft residual 0.01 lands on detu of the largest-entrainment layer 3.
    @test detu[3] ≈ 0.01 + 0.01 atol = 1e-15
    # The downdraft residual 0.005 lands on entd of its largest-entrainment layer 2.
    @test entd[2] ≈ 0.03 + 0.005 atol = 1e-15
end

@testset "TM5 closure: no-op when column is already balanced" begin
    Nz = 4
    entu = zeros(Float64, Nz); detu = zeros(Float64, Nz)
    entd = zeros(Float64, Nz); detd = zeros(Float64, Nz)
    # No updraft anywhere → no closure work to do.
    udmf = zeros(Float64, Nz + 1)
    ddmf = zeros(Float64, Nz + 1)
    udrf = zeros(Float64, Nz)
    ddrf = zeros(Float64, Nz)
    dz   = fill(200.0, Nz)

    stats = TM5CleanupStats()
    ec2tm_from_rates!(entu, detu, entd, detd,
                       udmf, ddmf, udrf, ddrf, dz, Nz; stats = stats)

    @test all(entu .== 0) && all(detu .== 0)
    @test all(entd .== 0) && all(detd .== 0)
    @test stats.columns_closure_entu[] == 0
    @test stats.columns_closure_entd[] == 0
end

@testset "plan 24 Commit 1: dz_hydrostatic_virtual! sanity" begin
    # Typical ERA5 L137-like column: realistic T/Q + ps.
    Nz = 5
    # Descending ak/bk so p_top < p_bot at each level.
    ak = Float64[0.0, 100.0, 500.0, 1000.0, 5000.0, 30000.0]
    bk = Float64[0.0, 0.0,   0.05,  0.2,    0.6,    1.0]
    ps = 101325.0
    T  = Float64[210, 220, 240, 260, 280]  # K
    Q  = Float64[0.0, 0.0, 0.001, 0.005, 0.010]  # kg/kg
    dz_virt  = zeros(Float64, Nz)
    dz_const = zeros(Float64, Nz)

    dz_hydrostatic_virtual!(dz_virt,  T, Q, ps, ak, bk, Nz)
    dz_hydrostatic_constT!(dz_const, ps, ak, bk, Nz; T_ref = 260)

    # All dz strictly positive.
    @test all(dz_virt .> 0)
    @test all(dz_const .> 0)

    # Virtual-temperature variant gives larger dz where air is warm/moist
    # (lower density → more height per hPa).  Surface layer k=Nz has
    # highest T and Q.
    @test dz_virt[Nz] > dz_const[Nz]    # surface warmer than 260 → more dz

    # Divergence bounded in troposphere: within 30% everywhere, typical
    # 10-20% at surface.
    for k in 1:Nz
        rel = abs(dz_virt[k] - dz_const[k]) / dz_const[k]
        @test rel < 0.30
    end
end

@testset "TM5 dz uses the model's R, g and virtual-temperature factor" begin
    P = AtmosTransport.Parameters
    ak = [0.0, 100.0, 5000.0, 0.0]
    bk = [0.0, 0.0, 0.3, 1.0]
    for FT in (Float32, Float64)
        T, Q, ps = FT[220, 250, 285], FT[0, 0.002, 0.015], 101325.0
        R_over_g = FT(P.R_DRY_AIR / P.STANDARD_GRAVITY)
        p_half = [FT(ak[k]) + FT(bk[k]) * FT(ps) for k in 1:4]
        dp = diff(p_half)
        p_mid = FT(0.5) .* (p_half[1:3] .+ p_half[2:4])
        dz = zeros(FT, 3)
        dz_hydrostatic_virtual!(dz, T, Q, ps, ak, bk, 3)
        @test dz == R_over_g .* (T .* (one(FT) .+ FT(P.VIRTUAL_TEMPERATURE_FACTOR) .* Q)) .* dp ./ p_mid
        dz_hydrostatic_constT!(dz, ps, ak, bk, 3; T_ref = 260)
        @test dz == R_over_g .* FT(260) .* dp ./ p_mid
    end
end

@testset "plan 24 Commit 1: dz_hydrostatic_virtual! shape guards" begin
    ak = Float64[0, 100, 500]; bk = Float64[0, 0.1, 1.0]
    ps = 101325.0
    T = Float64[220, 260]; Q = Float64[0.001, 0.005]
    dz = zeros(Float64, 2)

    # Correct shapes → no throw.
    dz_hydrostatic_virtual!(dz, T, Q, ps, ak, bk, 2)
    @test all(dz .> 0)

    # Mismatched sizes → clear errors.
    @test_throws ArgumentError dz_hydrostatic_virtual!(
        zeros(3), T, Q, ps, ak, bk, 2)
    @test_throws ArgumentError dz_hydrostatic_virtual!(
        dz, T, Float64[0.001], ps, ak, bk, 2)
    @test_throws ArgumentError dz_hydrostatic_virtual!(
        dz, T, Q, ps, Float64[0, 100], bk, 2)
end

@testset "plan 24 Commit 1: ec2tm! (plan-23 minimal) unchanged" begin
    # Regression: the plan-23 ec2tm! signature and behaviour must stay
    # identical.  One zero-forcing test is enough to catch accidental
    # behaviour drift from the new ec2tm_from_rates! additions.
    Nz = 4
    entu = zeros(Float64, Nz); detu = zeros(Float64, Nz)
    entd = zeros(Float64, Nz); detd = zeros(Float64, Nz)
    mflu = zeros(Float64, Nz + 1); mfld = zeros(Float64, Nz + 1)
    detu_ec = zeros(Float64, Nz); detd_ec = zeros(Float64, Nz)

    ec2tm!(entu, detu, entd, detd, mflu, mfld, detu_ec, detd_ec)
    @test all(entu .== 0) && all(detu .== 0)
    @test all(entd .== 0) && all(detd .== 0)
end

@testset "plan 24 Commit 1: ec2tm_from_rates! shape guards" begin
    Nz = 4
    entu = zeros(Float64, Nz); detu = zeros(Float64, Nz)
    entd = zeros(Float64, Nz); detd = zeros(Float64, Nz)
    udmf = zeros(Float64, Nz + 1); ddmf = zeros(Float64, Nz + 1)
    udrf = zeros(Float64, Nz); ddrf = zeros(Float64, Nz)
    dz   = fill(100.0, Nz)

    # Wrong entu length → clear error.
    @test_throws ArgumentError ec2tm_from_rates!(
        zeros(Float64, 3), detu, entd, detd,
        udmf, ddmf, udrf, ddrf, dz, Nz)

    # Wrong udmf length → clear error.
    @test_throws ArgumentError ec2tm_from_rates!(
        entu, detu, entd, detd,
        zeros(Float64, Nz), ddmf, udrf, ddrf, dz, Nz)

    # A flux through the ground (interface Nz + 1) is a packing error: ERA5
    # level k is the top of layer k, and the surface interface carries none.
    for (u, d) in ((vcat(zeros(Nz), 0.01), zeros(Nz + 1)),
                   (zeros(Nz + 1), vcat(zeros(Nz), -0.01)))
        err = try
            ec2tm_from_rates!(entu, detu, entd, detd, u, d, copy(udrf), copy(ddrf), dz, Nz)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test err !== nothing && occursin("ground", sprint(showerror, err))
    end
    # Sub-threshold noise there is clipped to zero first and accepted.
    ec2tm_from_rates!(entu, detu, entd, detd, vcat(zeros(Nz), 5e-7),
                      vcat(zeros(Nz), -5e-7), copy(udrf), copy(ddrf), dz, Nz)
    @test all(iszero, entu) && all(iszero, detu)
end

@testset "ec2tm_from_rates! sign-aware closure (regression)" begin
    # TM5's meteo.F90 closure writes `entu[lsave] += totd - tote`. A flux
    # through the model top (interface 1, 0.2 here) leaves tote - totd = 0.2,
    # more than the largest entrainment (0.15 in layer 1), so TM5's subtraction
    # would make entu negative. The sign-aware closure adds the 0.2 to detu
    # there instead: the same net entu - detu, both non-negative.
    Nz = 4
    grib_udmf = Float64[0.2, 0.05, 0.1, 0.0]
    stats = TM5CleanupStats()
    entu, detu, entd, detd = _ec2tm_grib(grib_udmf, zeros(Nz), zeros(Nz), zeros(Nz),
                                         fill(100.0, Nz); stats)
    @test stats.columns_closure_entu[] == 1
    @test all(entu .>= 0) && all(detu .>= 0) && all(entd .>= 0) && all(detd .>= 0)
    @test entu ≈ [0.15, 0.0, 0.1, 0.0] atol = 1e-15
    @test detu ≈ [0.2, 0.05, 0.0, 0.0] atol = 1e-15
    @test sum(entu) ≈ sum(detu) atol = 1e-15
end
