#!/usr/bin/env julia
# TM5 storage redesign Commit 7 — sparsity-above-icltop precondition.
#
# Codex's directive: this test is *deterministic* — it constructs
# fixture columns with the failure-mode geometries directly, rather
# than sampling from a production binary. The test gates the
# `icltop_eff = min(icllfs, max(icltop, 2) - 1)` formula
# (`_tm5_effective_cloud_top` in src/Adjoints/ConvectionAdjoint.jl; the
# forward `_tm5_solve_column!` in src/Operators/Convection/tm5_column_solve.jl
# writes the same expression inline) and the active-window LU
# implementation.
#
# Assertions per fixture:
# 1. Upper rows are identity:  conv1[k, j] == (k == j) for k ∈ [1, icltop_eff-1].
# 2. Lower-left quadrant zero: conv1[k, j] == 0 for k ∈ [icltop_eff, Nz]
#    and j ∈ [1, icltop_eff-1].
# 3. The production `f === conv1` alias path matches the standalone
#    `(Nz+1, Nz)` scratch path.
#
# Failing this test means the `_tm5_build_conv1!` builder changed and
# the `icltop_eff` formula needs revisiting. It does NOT mean "fall
# back to full-Nz"; the fallback was rejected in the 2026-04-30 review.

using Test

using AtmosTransport.Operators.Convection: _tm5_diagnose_cloud_dims,
                                              _tm5_build_conv1!
# The active-window pivot `icltop_eff` used by the TM5 LU solves.
# `max(icltop, 2)` guards `icltop = 1` (no skip possible); a value `< 1`
# means no skip (already handled by the LU loop).
using AtmosTransport.Adjoints: _tm5_effective_cloud_top

const FT = Float32

"""
Build a column with prescribed `(icltop_target, icllfs_target)`.
- `entu`, `detu` are nonzero only on `[icltop_target, Nz]`.
- `entd`, `detd` are nonzero only on `[icllfs_target, Nz]`.
- `m_col` is unit-everywhere positive.
Returns the diagnosed `(icltop, iclbas, icllfs)` so callers can verify
the geometry didn't drift through the diagnosis.
"""
function _make_column(Nz::Int; icltop_target::Int, icllfs_target::Int)
    entu = zeros(FT, Nz); detu = zeros(FT, Nz)
    entd = zeros(FT, Nz); detd = zeros(FT, Nz)
    if icltop_target ≤ Nz
        for k in icltop_target:Nz
            entu[k] = FT(0.02 + 0.001 * (k - icltop_target))
            detu[k] = FT(0.015 + 0.002 * (k - icltop_target))
        end
    end
    if icllfs_target ≤ Nz
        for k in icllfs_target:Nz
            entd[k] = FT(0.01 + 0.0005 * (k - icllfs_target))
            detd[k] = FT(0.012 + 0.0004 * (k - icllfs_target))
        end
    end
    m_col = ones(FT, Nz) .* FT(1000.0)
    return entu, detu, entd, detd, m_col
end

function _build_and_check(name::String; Nz::Int,
                          icltop_target::Int, icllfs_target::Int,
                          dt::FT = FT(900.0))
    entu, detu, entd, detd, m_col = _make_column(Nz;
        icltop_target = icltop_target, icllfs_target = icllfs_target)
    icltop, _, icllfs = _tm5_diagnose_cloud_dims(detu, entd, Nz)

    if icltop > Nz
        # No convection — short-circuited in `_tm5_solve_column!` before
        # `_tm5_build_conv1!` is called. Nothing to assert at the matrix
        # level.
        @info "[$name] no convection — short-circuited; skipping matrix asserts"
        return
    end

    conv1 = zeros(FT, Nz, Nz)
    f_buf = zeros(FT, Nz + 1, Nz)
    amu = zeros(FT, Nz + 1)
    amd = zeros(FT, Nz + 1)
    _tm5_build_conv1!(conv1, entu, detu, entd, detd, m_col,
                      icltop, icllfs, dt, Nz;
                      f = f_buf, amu = amu, amd = amd)

    conv1_alias = zeros(FT, Nz, Nz)
    amu_alias = zeros(FT, Nz + 1)
    amd_alias = zeros(FT, Nz + 1)
    _tm5_build_conv1!(conv1_alias, entu, detu, entd, detd, m_col,
                      icltop, icllfs, dt, Nz;
                      f = conv1_alias, amu = amu_alias, amd = amd_alias)
    @test conv1_alias == conv1

    icltop_eff = _tm5_effective_cloud_top(icltop, icllfs)
    upper_ok = true
    for k in 1:max(0, icltop_eff - 1)
        for j in 1:Nz
            expected = (k == j) ? one(FT) : zero(FT)
            if conv1[k, j] != expected
                upper_ok = false
                @info "[$name] upper-row violation conv1[$k,$j]=$(conv1[k,j]) expected=$expected"
                @goto upper_done
            end
        end
    end
    @label upper_done
    @test upper_ok

    lower_left_ok = true
    for k in icltop_eff:Nz
        for j in 1:max(0, icltop_eff - 1)
            if conv1[k, j] != zero(FT)
                lower_left_ok = false
                @info "[$name] lower-left violation conv1[$k,$j]=$(conv1[k,j])"
                @goto lower_left_done
            end
        end
    end
    @label lower_left_done
    @test lower_left_ok
end

@testset "TM5 conv1 sparsity above icltop_eff" begin
    Nz = 72
    @testset "icllfs > icltop  (typical)"        _build_and_check("icllfs>icltop";  Nz=Nz, icltop_target=40, icllfs_target=55)
    @testset "icllfs == icltop"                  _build_and_check("icllfs==icltop"; Nz=Nz, icltop_target=40, icllfs_target=40)
    @testset "icllfs < icltop  (historical fail)" _build_and_check("icllfs<icltop"; Nz=Nz, icltop_target=50, icllfs_target=30)
    @testset "icltop = 1 (full-depth)"           _build_and_check("icltop=1";       Nz=Nz, icltop_target=1,  icllfs_target=20)
    @testset "icltop = 2 (boundary)"             _build_and_check("icltop=2";       Nz=Nz, icltop_target=2,  icllfs_target=10)
    @testset "icltop > Nz (no convection)"       _build_and_check("no convection";  Nz=Nz, icltop_target=Nz + 5, icllfs_target=Nz + 5)

    # L137 cases mirroring the C180/L137 binary scan.
    Nz137 = 137
    @testset "L137 typical median (icltop=106)"  _build_and_check("L137 median";  Nz=Nz137, icltop_target=106, icllfs_target=120)
    @testset "L137 deepest seen (icltop=54)"     _build_and_check("L137 deepest"; Nz=Nz137, icltop_target=54,  icllfs_target=70)
end
