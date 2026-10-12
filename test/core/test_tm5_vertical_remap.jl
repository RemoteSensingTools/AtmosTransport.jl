#!/usr/bin/env julia
"""
Plan 24 Commit 3 — grid-level TM5 pipeline + vertical-remap tests.

Exercises the two plumbing functions that Commit 4's `process_day`
will call to produce merged TM5 fields from native-L137 ERA5
physics data:

- `tm5_native_fields_for_hour!` — for one hour, run
  `dz_hydrostatic_virtual!` + `ec2tm_from_rates!` across the full
  2D grid, writing native-level entu/detu/entd/detd.

- `merge_tm5_field_3d!` — sum native-level fluxes into the
  merged-grid buckets via `merge_map`, preserving the column-
  integrated flux exactly.

Tests run core (no real data). Commit 3's invariant: after
running native-level ec2tm and then merging to Nz, the
column-sum of each TM5 field equals the column-sum at native
resolution — mass-budget preservation in the vertical-reduction
step.
"""

using Test

using AtmosTransport

using .AtmosTransport.Preprocessing: tm5_native_fields_for_hour!,
                                       merge_tm5_field_3d!,
                                       TM5CleanupStats

# ─────────────────────────────────────────────────────────────────────────

@testset "plan 24 Commit 3: merge_tm5_field_3d! preserves column sum" begin
    FT = Float64
    Nlon, Nlat, Nz_native, Nz_merged = 4, 3, 10, 4

    # Simple merge_map: native levels 1-3 → merged 1, 4-6 → merged 2,
    # 7-8 → merged 3, 9-10 → merged 4.
    merge_map = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    @test length(merge_map) == Nz_native
    @test maximum(merge_map) == Nz_merged

    native = rand(FT, Nlon, Nlat, Nz_native)
    merged = zeros(FT, Nlon, Nlat, Nz_merged)

    merge_tm5_field_3d!(merged, native, merge_map)

    # Column-sum preservation: for each (i, j), sum(native[i,j,:])
    # == sum(merged[i,j,:]).
    for j in 1:Nlat, i in 1:Nlon
        sum_native = sum(native[i, j, :])
        sum_merged = sum(merged[i, j, :])
        @test isapprox(sum_merged, sum_native; rtol = 1e-12)
    end

    # Per-bucket values should match the partial sums.
    for j in 1:Nlat, i in 1:Nlon
        @test merged[i, j, 1] ≈ sum(native[i, j, 1:3])
        @test merged[i, j, 2] ≈ sum(native[i, j, 4:6])
        @test merged[i, j, 3] ≈ sum(native[i, j, 7:8])
        @test merged[i, j, 4] ≈ sum(native[i, j, 9:10])
    end
end

@testset "plan 24 Commit 3: merge_tm5_field_3d! zeros unused buckets" begin
    FT = Float32
    Nlon, Nlat, Nz_native, Nz_merged = 2, 2, 6, 5
    # merge_map that leaves Nz_merged=5 unused (only 1..4 reached).
    merge_map = [1, 1, 2, 3, 3, 4]
    native = fill(FT(1), Nlon, Nlat, Nz_native)
    # Seed the output with nonzero garbage to make sure merge clears it.
    merged = fill(FT(999), Nlon, Nlat, Nz_merged)
    merge_tm5_field_3d!(merged, native, merge_map)
    # Bucket 5 should be zero.
    @test all(merged[:, :, 5] .== 0)
    # Other buckets: 1=2 (two ones), 2=1, 3=2 (two ones), 4=1.
    @test all(merged[:, :, 1] .== 2)
    @test all(merged[:, :, 2] .== 1)
    @test all(merged[:, :, 3] .== 2)
    @test all(merged[:, :, 4] .== 1)
end

@testset "plan 24 Commit 3: merge_tm5_field_3d! shape guards" begin
    FT = Float64
    native = zeros(FT, 4, 3, 10)
    merged = zeros(FT, 4, 3, 4)
    merge_map = collect(1:10)   # length must equal Nz_native

    # Correct shapes.
    merge_tm5_field_3d!(merged, native, [1, 1, 1, 2, 2, 2, 3, 3, 4, 4])

    @test_throws ArgumentError merge_tm5_field_3d!(
        zeros(FT, 3, 3, 4), native, [1, 1, 1, 2, 2, 2, 3, 3, 4, 4])  # Nlon mismatch
    @test_throws ArgumentError merge_tm5_field_3d!(
        merged, zeros(FT, 4, 3, 8), [1, 1, 2, 2, 3, 3])  # Nz_native=8 vs map=6
end

# ─────────────────────────────────────────────────────────────────────────

@testset "plan 24 Commit 3: tm5_native_fields_for_hour! grid loop" begin
    FT = Float64
    Nlon, Nlat = 3, 2
    Nz_native = 5

    # Simple hybrid: linear in sigma (bk=1 at surface).
    ak_full = Float64[0.0, 100.0, 500.0, 1000.0, 5000.0, 30000.0]
    bk_full = Float64[0.0, 0.0, 0.05, 0.2, 0.6, 1.0]

    # All-column constant forcing: produces predictable, uniform outputs.
    udmf = zeros(FT, Nlon, Nlat, Nz_native)
    ddmf = zeros(FT, Nlon, Nlat, Nz_native)
    udrf = zeros(FT, Nlon, Nlat, Nz_native)
    ddrf = zeros(FT, Nlon, Nlat, Nz_native)
    udmf[:, :, 3] .= 0.05   # updraft flux in middle of column
    udrf[:, :, 4] .= 0.001  # detrainment rate near surface

    t = fill(FT(260), Nlon, Nlat, Nz_native)
    q = fill(FT(0.003), Nlon, Nlat, Nz_native)
    ps = fill(FT(101325), Nlon, Nlat)

    entu = zeros(FT, Nlon, Nlat, Nz_native)
    detu = zeros(FT, Nlon, Nlat, Nz_native)
    entd = zeros(FT, Nlon, Nlat, Nz_native)
    detd = zeros(FT, Nlon, Nlat, Nz_native)

    stats = TM5CleanupStats()
    tm5_native_fields_for_hour!(
        entu, detu, entd, detd,
        udmf, ddmf, udrf, ddrf,
        t, q, ps, ak_full, bk_full, Nz_native;
        basis = DryBasis(), stats = stats)

    # Every column processed.
    @test stats.columns_processed[] == Nlon * Nlat

    # All columns identical inputs → identical outputs.
    for k in 1:Nz_native
        @test all(entu[:, :, k] .≈ entu[1, 1, k])
        @test all(detu[:, :, k] .≈ detu[1, 1, k])
    end

    # Non-negative outputs.
    @test all(entu .>= 0) && all(detu .>= 0)
    @test all(entd .>= 0) && all(detd .>= 0)

    # Some levels see nonzero updraft entrainment (below udmf's level).
    @test sum(entu) > 0
    @test sum(detu) > 0
end

@testset "plan 24 Commit 3: tm5_native_fields_for_hour! output-shape guard" begin
    FT = Float32
    Nlon, Nlat = 4, 3
    Nz_native = 5
    ak_full = collect(Float64, 0:100.0:(Nz_native * 100))
    bk_full = collect(Float64, 1:-(1/Nz_native):0)
    udmf = zeros(FT, Nlon, Nlat, Nz_native)
    ddmf = zeros(FT, Nlon, Nlat, Nz_native)
    udrf = zeros(FT, Nlon, Nlat, Nz_native)
    ddrf = zeros(FT, Nlon, Nlat, Nz_native)
    t    = fill(FT(250), Nlon, Nlat, Nz_native)
    q    = fill(FT(0.002), Nlon, Nlat, Nz_native)
    ps   = fill(FT(100000), Nlon, Nlat)

    # Correct entu shape.
    tm5_native_fields_for_hour!(
        zeros(FT, Nlon, Nlat, Nz_native),
        zeros(FT, Nlon, Nlat, Nz_native),
        zeros(FT, Nlon, Nlat, Nz_native),
        zeros(FT, Nlon, Nlat, Nz_native),
        udmf, ddmf, udrf, ddrf, t, q, ps,
        ak_full, bk_full, Nz_native; basis = DryBasis())

    # Wrong Nz_native arg → clear error.
    @test_throws ArgumentError tm5_native_fields_for_hour!(
        zeros(FT, Nlon, Nlat, Nz_native),
        zeros(FT, Nlon, Nlat, Nz_native),
        zeros(FT, Nlon, Nlat, Nz_native),
        zeros(FT, Nlon, Nlat, Nz_native),
        udmf, ddmf, udrf, ddrf, t, q, ps,
        ak_full, bk_full, 3; basis = DryBasis())   # wrong Nz_native
end

# ─────────────────────────────────────────────────────────────────────────

@testset "plan 24 Commit 3: end-to-end native → merged column sum preserved" begin
    # Column-integrated flux must be preserved by the full pipeline:
    # ec2tm at native 137L → merge_map to Nz_merged.
    FT = Float64
    Nlon, Nlat = 4, 3
    Nz_native = 8
    Nz_merged = 3

    # Merge 2 native levels per merged level below, 4 above:
    merge_map = [1, 1, 2, 2, 2, 3, 3, 3]
    ak_full = collect(Float64, 0:5000.0:(Nz_native * 5000))
    bk_full = reverse(collect(Float64, 0:(1/Nz_native):1))

    # Nonzero updraft profile to exercise ec2tm.
    udmf = zeros(FT, Nlon, Nlat, Nz_native)
    udmf[:, :, 3] .= 0.03
    udmf[:, :, 4] .= 0.05
    udmf[:, :, 5] .= 0.02
    ddmf = zeros(FT, Nlon, Nlat, Nz_native)
    udrf = zeros(FT, Nlon, Nlat, Nz_native)
    udrf[:, :, 4:5] .= 0.001
    ddrf = zeros(FT, Nlon, Nlat, Nz_native)
    t  = fill(FT(260), Nlon, Nlat, Nz_native)
    q  = fill(FT(0.003), Nlon, Nlat, Nz_native)
    ps = fill(FT(101325), Nlon, Nlat)

    # Compute native-level fields.
    entu_n = zeros(FT, Nlon, Nlat, Nz_native)
    detu_n = zeros(FT, Nlon, Nlat, Nz_native)
    entd_n = zeros(FT, Nlon, Nlat, Nz_native)
    detd_n = zeros(FT, Nlon, Nlat, Nz_native)
    tm5_native_fields_for_hour!(
        entu_n, detu_n, entd_n, detd_n,
        udmf, ddmf, udrf, ddrf,
        t, q, ps, ak_full, bk_full, Nz_native; basis = DryBasis())

    # Merge to Nz_merged.
    entu_m = zeros(FT, Nlon, Nlat, Nz_merged)
    detu_m = zeros(FT, Nlon, Nlat, Nz_merged)
    entd_m = zeros(FT, Nlon, Nlat, Nz_merged)
    detd_m = zeros(FT, Nlon, Nlat, Nz_merged)
    merge_tm5_field_3d!(entu_m, entu_n, merge_map)
    merge_tm5_field_3d!(detu_m, detu_n, merge_map)
    merge_tm5_field_3d!(entd_m, entd_n, merge_map)
    merge_tm5_field_3d!(detd_m, detd_n, merge_map)

    # Column-sum invariant (the plan 24 Commit 3 core claim).
    for j in 1:Nlat, i in 1:Nlon
        @test isapprox(sum(entu_m[i, j, :]), sum(entu_n[i, j, :]); rtol = 1e-12)
        @test isapprox(sum(detu_m[i, j, :]), sum(detu_n[i, j, :]); rtol = 1e-12)
        @test isapprox(sum(entd_m[i, j, :]), sum(entd_n[i, j, :]); rtol = 1e-12)
        @test isapprox(sum(detd_m[i, j, :]), sum(detd_n[i, j, :]); rtol = 1e-12)
    end
end
