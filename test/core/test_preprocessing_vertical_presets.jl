using Test
using TOML

import AtmosTransport
using .AtmosTransport.Grids: HybridSigmaPressure, n_levels
using .AtmosTransport.Parameters:
    ECHLEVS_ML137_TROPO34,
    ECHLEVS_ML137_TROPO34A,
    ECHLEVS_ML137_66L,
    ECHLEVS_ML137_CFL94,
    ECHLEVS_ML137_CFL85
using .AtmosTransport.Preprocessing:
    build_vertical_setup,
    echlevs_preset,
    select_levels_echlevs

function merge_map_from_spans(spans::Vector{UnitRange{Int}}, nz::Int)
    mm = zeros(Int, nz)
    for (g, span) in enumerate(spans)
        mm[span] .= g
    end
    return mm
end


interfaces_from_spans(spans::Vector{UnitRange{Int}}, vc) =
    vcat([vc.A[first(span)] for span in spans], vc.A[last(spans[end]) + 1])

function synthetic_l137_vc()
    a = collect(Float64, 0:137)
    b = zeros(Float64, 138)
    return HybridSigmaPressure(a, b)
end

function write_synthetic_l137_coefficients(path::AbstractString)
    open(path, "w") do io
        TOML.print(io, Dict(
            "coefficients" => Dict(
                "a" => collect(Float64, 0:137),
                "b" => zeros(Float64, 138),
            ),
        ))
    end
end


@testset "TM5-numbered ERA5 L137 echlev presets" begin
    # Verbatim from TM5-MP r1112 levels/ml137/tropo34
    tm5_mp_r1112_tropo34 = [
        137, 134, 129, 124, 119, 114, 110, 105, 101,  97,
         93,  88,  84,  81,  78,  76,  73,  70,  67,  65,
         62,  59,  57,  54,  51,  46,  42,  37,  32,  27,
         22,  17,  12,   7,   0,
    ]
    # Verbatim from TM5 cy3-4dvar proj/levels/ml137/tropo34a/src/dims_levels.F90.
    tm5_cy34dvar_tropo34a = [
        137, 134, 131, 126, 122, 117, 113, 109, 105, 102,
         98,  96,  94,  92,  90,  88,  85,  83,  81,  78,
         75,  73,  70,  67,  64,  61,  57,  54,  51,  44,
         34,  26,  20,  14,   0,
    ]

    @test collect(ECHLEVS_ML137_TROPO34) == tm5_mp_r1112_tropo34
    @test collect(ECHLEVS_ML137_TROPO34A) == tm5_cy34dvar_tropo34a
    @test echlevs_preset("ml137_tropo34") == tm5_mp_r1112_tropo34
    @test echlevs_preset("ml137_tropo34a") == tm5_cy34dvar_tropo34a

    coeff_path = joinpath(@__DIR__, "..", "..", "config", "era5_L137_coefficients.toml")
    coefficients = TOML.parsefile(coeff_path)["coefficients"]
    a = Float64.(coefficients["a"])
    b = Float64.(coefficients["b"])
    vc = HybridSigmaPressure(a, b)

    # TM5 geometry.F90 uses at(l) = a_ec(echlevs(l-1)): e selects e + 1
    # in Julia's top-first arrays. The pre-fix selector instead chose 138 - e.
    for levels in (tm5_mp_r1112_tropo34, tm5_cy34dvar_tropo34a)
        selected, _ = select_levels_echlevs(vc, levels)
        coefficient_indices = sort(levels) .+ 1
        @test selected.A == a[coefficient_indices]
        @test selected.B == b[coefficient_indices]
    end

    # AtmosTransport's own recipes were written for the mirrored selector.
    # Reversing each recipe as 137 - e must preserve the old physical grid
    # exactly, including every Float64 bit of the real ERA5 coefficients.
    legacy_66 = [
        137, 135, 130, 125, 120, 115, 110, 105, 100,
         96,  93,  90,  87,  84,  81,  78,  75,  72,  69,  66,
         63,  60,  58,  56,  54,  52,  50,  48,  46,  44,
         42,  40,  38,  36,  34,  32,  30,  29,  28,  27,
         26,  25,  24,  23,  22,  21,  20,  19,  18,  17,
         16,  15,  14,  13,  12,  11,  10,   9,   8,   7,
          6,   5,   4,   3,   2,   1,   0,
    ]
    legacy_cfl94 = vcat(
        [137, 125, 113, 109, 105, 101, 97, 93, 89, 85], collect(84:-1:0))
    legacy_cfl85 = vcat(
        [137, 125, 113, 109, 105, 101, 97, 93, 89, 85,
          84,  82,  80,  78,  76,  74, 72, 70, 68, 66], collect(65:-1:0))

    for (corrected, legacy) in ((ECHLEVS_ML137_66L, legacy_66),
                                (ECHLEVS_ML137_CFL94, legacy_cfl94),
                                (ECHLEVS_ML137_CFL85, legacy_cfl85))
        @test collect(corrected) == 137 .- legacy
        selected, _ = select_levels_echlevs(vc, corrected)
        old_coefficient_indices = sort(138 .- legacy)
        @test selected.A == a[old_coefficient_indices]
        @test selected.B == b[old_coefficient_indices]
    end

    # Development-mode cleanup: the descriptive names are the only API.
    @test_throws ErrorException echlevs_preset("ml137_94L")
    @test_throws ErrorException echlevs_preset("ml137_85L")
    vc = synthetic_l137_vc()

    spans_94 = vcat(
        [1:12, 13:24],
        [s:min(s + 3, 53) for s in 25:4:53],
        [k:k for k in 54:137],
    )
    selected_94, merge_map_94 = select_levels_echlevs(vc, ECHLEVS_ML137_CFL94)

    @test n_levels(selected_94) == 94
    @test length(ECHLEVS_ML137_CFL94) == 95
    @test merge_map_94 == merge_map_from_spans(spans_94, 137)
    @test selected_94.A == interfaces_from_spans(spans_94, vc)

    spans_85 = vcat(
        [1:12, 13:24],
        [s:min(s + 3, 53) for s in 25:4:53],
        [s:(s + 1) for s in 54:2:70],
        [k:k for k in 72:137],
    )
    selected_85, merge_map_85 = select_levels_echlevs(vc, ECHLEVS_ML137_CFL85)

    @test n_levels(selected_85) == 85
    @test length(ECHLEVS_ML137_CFL85) == 86
    @test merge_map_85 == merge_map_from_spans(spans_85, 137)
    @test selected_85.A == interfaces_from_spans(spans_85, vc)
end
@testset "CFL-oriented echlev presets are available through config" begin
    mktempdir() do dir
        coeff_path = joinpath(dir, "synthetic_l137.toml")
        write_synthetic_l137_coefficients(coeff_path)

        vertical_cfl94 = build_vertical_setup(
            coeff_path, 1:137, 1000.0, Dict("echlevs" => "ml137_cfl94"))
        vertical_cfl85 = build_vertical_setup(
            coeff_path, 1:137, 1000.0, Dict("echlevs" => "ml137_cfl85"))

        @test vertical_cfl94.Nz == 94
        @test vertical_cfl85.Nz == 85
        @test vertical_cfl94.vertical_mapping_method == :merge_map
        @test vertical_cfl85.vertical_mapping_method == :merge_map
    end
end
