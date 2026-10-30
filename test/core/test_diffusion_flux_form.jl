#!/usr/bin/env julia
# Implicit Kz vertical diffusion updates tracer mass in flux form (finding
# OF-113; see the header of src/Operators/Diffusion/diffusion_kernels.jl).
#
# The column solve gives the new mixing ratio g; the tracer mass then changes by
# the interface fluxes F_{k+½} = Δt D_{k+½} (g_k − g_{k+1}), each applied with
# opposite signs to its two layers, so the column total telescopes. Writing back
# m_k (g_k + c) instead (the matrix form, the historical VMR-level solve)
# loses mass with one sign in Float32 where x = Δt D/m is the same in every
# column, as in pure-pressure levels. Checked on every layout:
#
#   1. one solve agrees with an independent Float64 Backward-Euler solution;
#   2. uniform columns and layers without exchange keep their mass bit for bit;
#   3. over a day of Float32 solves the flux form
#      stays at the rounding level;
#   4. the same on CUDA when a GPU is available.

using Test, Random, LinearAlgebra
using AtmosTransport
using .AtmosTransport.Operators: ImplicitVerticalDiffusion, DiffusionWorkspace
using .AtmosTransport.Operators.Diffusion: apply_vertical_diffusion_vmr!
using .AtmosTransport.State: PreComputedKzField, CubedSphereField
using Adapt: adapt

const HAS_CUDA_FLUX_FORM = try
    using CUDA
    CUDA.functional()
catch
    false
end

# --- Column fixture -------------------------------------------------------------
#
# `ncol` columns of `Nz` layers, top first. The upper `ntop` layers are pure
# pressure: their pressure thickness and layer depth are the same in every
# column, and the air mass differs only by the cell area, which cancels in
# x = Δt D/m. Below, every column differs. Tracer 1 is non-negative with no
# background (a blob in the upper levels plus a surface source), tracer 2 a
# 400 ppm background with a small non-negative anomaly.
function flux_form_columns(::Type{FT}, ncol, Nz; ntop = Nz ÷ 2, kz = 1.0,
                           kz_top = kz, seed = 113) where FT
    rng = MersenneTwister(seed)
    m, dz, Kz = zeros(FT, ncol, Nz), zeros(FT, ncol, Nz), zeros(FT, ncol, Nz)
    rm = zeros(FT, ncol, Nz, 2)
    for c in 1:ncol
        area = 1 + 0.5 * sin(c)
        for k in 1:Nz
            pure = k <= ntop
            dp = pure ? 400.0 + 150k : (2500.0 + 400(k - ntop)) * (1 + 0.05 * sin(3c + k))
            m[c, k] = FT(area * 1e10 * dp / 9.80665)
            dz[c, k] = FT(pure ? 3000.0 - 100k : (900.0 - 25(k - ntop)) * (1 + 0.05 * cos(c + k)))
            Kz[c, k] = FT(pure ? kz_top : kz)
        end
        for k in 1:Nz
            blob = 1e-6 * exp(-(k - 3)^2 / 4) * (1 + c % 3) + (k >= Nz - 1 ? 1e-6 * (1 + c % 5) : 0.0)
            rm[c, k, 1] = FT(m[c, k] * blob)
            rm[c, k, 2] = FT(m[c, k] * (4e-4 + 1e-6 * rand(rng)))
        end
    end
    return (; m, dz, Kz, rm)
end

# Independent reference: the Backward-Euler tracer-mass system of one column in
# Float64, (M + Δt L) q′ = rm, with L the interface exchange Laplacian.
function backward_euler_column(rm, m, Kz, dz, dt)
    n = length(m)
    m, Kz, dz = Float64.(m), Float64.(Kz), Float64.(dz)
    A = Matrix(Diagonal(m))
    for k in 1:n-1
        D = (m[k] + m[k + 1]) * (Kz[k] + Kz[k + 1]) / (dz[k] + dz[k + 1])^2
        A[k, k] += dt * D; A[k + 1, k + 1] += dt * D
        A[k, k + 1] -= dt * D; A[k + 1, k] -= dt * D
    end
    return m .* (A \ Float64.(rm))
end

# --- Layouts: pack the columns, run a solve, unpack -----------------------------
#
# Each layout stores the same columns; `unpack` returns (ncol, Nz, Nt) tracer
# mass. `solve!(layout, form, dt)` applies the flux form (`:flux`, production
# `apply_vertical_diffusion_vmr!`) or the matrix form (`:matrix`, the VMR-level
# independent Backward-Euler solutions on tracer mass).

struct LatLonLayout{R, M, O, W}
    rm::R; m::M; op::O; ws::W; Nx::Int; Ny::Int
end
struct FaceLayout{R, M, O, W}
    rm::R; m::M; op::O; ws::W
end
struct PanelLayout{R, M, O, W}
    rm::R; m::M; op::O; ws::W; Nc::Int; Hp::Int
end

function latlon_layout(cols, Nt, to)
    ncol, Nz = size(cols.m)
    Nx = 6; Ny = ncol ÷ Nx
    m = to(reshape(cols.m, Nx, Ny, Nz))
    rm = to(reshape(cols.rm[:, :, 1:Nt], Nx, Ny, Nz, Nt))
    op = ImplicitVerticalDiffusion(; kz_field = adapt(to, PreComputedKzField(reshape(cols.Kz, Nx, Ny, Nz))))
    ws = DiffusionWorkspace(m, Nt)
    copyto!(ws.layer_thickness, to(reshape(cols.dz, Nx, Ny, Nz)))
    return LatLonLayout(rm, m, op, ws, Nx, Ny)
end

function face_layout(cols, Nt, to)
    m = to(copy(cols.m))
    rm = to(Nt == 0 ? cols.rm[:, :, 1] : cols.rm[:, :, 1:Nt])   # Nt = 0: single-tracer (ncells, Nz)
    op = ImplicitVerticalDiffusion(; kz_field = adapt(to, PreComputedKzField(copy(cols.Kz))))
    ws = DiffusionWorkspace(m, max(Nt, 1))
    copyto!(ws.layer_thickness, to(cols.dz))
    return FaceLayout(rm, m, op, ws)
end

# Cubed sphere: 6 Nc² columns with a one-cell halo; halo tracer cells hold a
# sentinel the solve must not touch. Nt = 0 packs single-tracer panels.
function panel_layout(cols, Nt, to; Hp = 1)
    ncol, Nz = size(cols.m)
    Nc = isqrt(ncol ÷ 6)
    N = Nc + 2Hp
    column(p, i, j) = (p - 1) * Nc^2 + (j - 1) * Nc + i
    m_h = ntuple(_ -> fill(eltype(cols.m)(1e12), N, N, Nz), 6)
    rm_h = ntuple(_ -> fill(eltype(cols.m)(-7), N, N, Nz, max(Nt, 1)), 6)
    kz_h = ntuple(_ -> zeros(eltype(cols.m), Nc, Nc, Nz), 6)
    dz_h = ntuple(_ -> zeros(eltype(cols.m), Nc, Nc, Nz), 6)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        c = column(p, i, j)
        m_h[p][i + Hp, j + Hp, :] .= cols.m[c, :]
        kz_h[p][i, j, :] .= cols.Kz[c, :]
        dz_h[p][i, j, :] .= cols.dz[c, :]
        for t in 1:max(Nt, 1)
            rm_h[p][i + Hp, j + Hp, :, t] .= cols.rm[c, :, t]
        end
    end
    m = map(to, m_h)
    rm = Nt == 0 ? map(a -> to(a[:, :, :, 1]), rm_h) : map(to, rm_h)
    field = adapt(to, CubedSphereField(ntuple(p -> PreComputedKzField(kz_h[p]), 6)))
    op = ImplicitVerticalDiffusion(; kz_field = field)
    ws = DiffusionWorkspace(m, Hp, max(Nt, 1))
    foreach((d, s) -> copyto!(d, to(s)), ws.layer_thickness, dz_h)
    return PanelLayout(rm, m, op, ws, Nc, Hp)
end

function unpack(l::LatLonLayout)
    a = Array(l.rm)
    return reshape(a, l.Nx * l.Ny, size(a, 3), size(a, 4))
end
unpack(l::FaceLayout) = (a = Array(l.rm); ndims(a) == 2 ? reshape(a, size(a)..., 1) : a)
function unpack(l::PanelLayout)
    Nc, Hp = l.Nc, l.Hp
    panels = map(Array, l.rm)
    Nz, Nt = size(panels[1], 3), size(panels[1], 4)
    out = zeros(eltype(panels[1]), 6Nc^2, Nz, Nt)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        out[(p - 1) * Nc^2 + (j - 1) * Nc + i, :, :] .= panels[p][i + Hp, j + Hp, :, :]
    end
    return out
end

# Halo cells of a panel layout (they must keep the sentinel).
function halo_values(l::PanelLayout)
    Nc, Hp = l.Nc, l.Hp
    return [a[i, j, k, t] for a in map(Array, l.rm) for t in axes(a, 4), k in axes(a, 3),
            j in axes(a, 2), i in axes(a, 1)
            if !(Hp < i <= Hp + Nc && Hp < j <= Hp + Nc)]
end

# Single-tracer arrays name their ledger carry slot as production does
# (`tracer = idx`); `slot = nothing` keeps no carry.
single(l::FaceLayout) = ndims(l.rm) == 2
single(l::PanelLayout) = ndims(l.rm[1]) == 3
solve!(l::LatLonLayout, ::Val{:flux}, dt) = apply_vertical_diffusion_vmr!(l.rm, l.m, l.op, l.ws, dt)
solve!(l::FaceLayout, ::Val{:flux}, dt; slot = 1) = single(l) ?
    apply_vertical_diffusion_vmr!(l.rm, l.m, l.op, l.ws, dt; tracer = slot) :
    apply_vertical_diffusion_vmr!(l.rm, l.m, l.op, l.ws, dt)
solve!(l::PanelLayout, ::Val{:flux}, dt; slot = 1) = single(l) ?
    apply_vertical_diffusion_vmr!(l.rm, l.m, l.op, l.ws, dt; halo_width = l.Hp, tracer = slot) :
    apply_vertical_diffusion_vmr!(l.rm, l.m, l.op, l.ws, dt; halo_width = l.Hp)

const LAYOUTS = (("lat-lon packed", latlon_layout, 2), ("reduced-Gaussian packed", face_layout, 2),
                 ("reduced-Gaussian single", face_layout, 0), ("cubed-sphere packed", panel_layout, 2),
                 ("cubed-sphere single", panel_layout, 0))

# --- 1. Agreement with the Backward-Euler solution --------------------------------

function check_backward_euler(to; FT, dt = 900.0)
    cols = flux_form_columns(FT, 54, 14; kz = 30.0)
    for (name, layout, Nt) in LAYOUTS
        l = layout(cols, Nt, to)
        solve!(l, Val(:flux), FT(dt))
        actual = unpack(l)
        # Allowed error: a fraction of the column's change plus the rounding of
        # the stored masses, which the column ledger returns to the largest
        # layer (2Nz ulps of it at most). The 400 ppm tracer changes by
        # 1e-5–1e-3 of its mass, well above that bound, so a solve that did
        # nothing fails.
        rtol = FT == Float64 ? 1e-10 : 2e-4
        worst = 0.0
        for t in axes(actual, 3), c in axes(actual, 1)
            before = cols.rm[c, :, t]
            expected = backward_euler_column(before, cols.m[c, :], cols.Kz[c, :], cols.dz[c, :], dt)
            change = maximum(abs, expected .- before)
            allowed = rtol * change + 2length(before) * eps(FT) * maximum(abs, before)
            change > 4 * allowed || error("fixture column $c, tracer $t changes too little")
            worst = max(worst, maximum(abs, actual[c, :, t] .- expected) / allowed)
        end
        @test worst <= 1
        # The merged mass-conservation tests retain both precisions and every layout.
        for t in axes(actual, 3)
            carry = column_carry(l, t)
            for c in axes(actual, 1)
                before_total = sum(Float64, cols.rm[c, :, t])
                residual = sum(Float64, actual[c, :, t]) - before_total + carry[c]
                @test abs(residual) / before_total < 1e-12
            end
        end
        l isa PanelLayout && @test all(==(FT(-7)), halo_values(l))
    end
end

# --- 2. Bit-exact uniform columns and isolated layers ------------------------------

function check_bit_exact(to; FT = Float32)
    cols = flux_form_columns(FT, 54, 14; kz = 30.0, kz_top = 0.0)
    for c in axes(cols.m, 1)
        # Tracer 2: a uniform mixing ratio per column, a power of two so that
        # rm/m reproduces it exactly in every layer.
        cols.rm[c, :, 2] .= cols.m[c, :] .* FT(2.0^(-11 - c % 3))
    end
    ntop = 14 ÷ 2
    for (name, layout, Nt) in LAYOUTS
        l = layout(cols, Nt, to)
        for _ in 1:24
            solve!(l, Val(:flux), FT(900))
        end
        actual = unpack(l)
        # Kz = 0 above layer ntop, so layers 1 … ntop-1 exchange nothing.
        @test actual[:, 1:ntop-1, 1] == cols.rm[:, 1:ntop-1, 1]
        Nt == 2 && @test actual[:, :, 2] == cols.rm[:, :, 2]
        if l isa Union{FaceLayout, PanelLayout} && single(l)
            # Without a carry slot a single-tracer solve is a function of its
            # input alone (the adjoint replays rely on this): repeating it on
            # the same input with the same workspace gives the same values.
            r = layout(cols, Nt, to)
            input = r.rm isa Tuple ? map(copy, r.rm) : copy(r.rm)
            results = map(1:2) do _
                r.rm isa Tuple ? foreach(copyto!, r.rm, input) : copyto!(r.rm, input)
                solve!(r, Val(:flux), FT(900); slot = nothing)
                unpack(r)
            end
            @test results[1] == results[2]
        end
    end
end

# --- 3. Float32 drift over a day of solves ---------------------------------------

const FLUXFORM_CARRY_LIMIT = 1e-12

relative_change(after, before) = (sum(Float64, after) - sum(Float64, before)) / sum(Float64, before)

# Outstanding ledger carries of tracer 1 (slot 1), in Float64.
carry_total(l::Union{LatLonLayout, FaceLayout}) =
    sum(Float64, Array(selectdim(l.ws.ledger_carry, ndims(l.ws.ledger_carry), 1)))
carry_total(l::PanelLayout) = sum(c -> sum(Float64, Array(c)[:, :, 1]), l.ws.ledger_carry)

column_carry(l::Union{LatLonLayout, FaceLayout}, t) =
    vec(Array(selectdim(l.ws.ledger_carry, ndims(l.ws.ledger_carry), t)))
column_carry(l::PanelLayout, t) = vcat((vec(Array(c)[:, :, t]) for c in l.ws.ledger_carry)...)

# 192 solves of 450 s of tracer 1 (no background). Returns the relative change
# of the global total for the flux form, of the flux form's total plus its
# outstanding ledger carries.
function day_of_solves(layout, Nt, to; kz = 1.0)
    cols = flux_form_columns(Float32, 216, 20; kz)
    before = cols.rm[:, :, 1]
    l = layout(cols, Nt, to)
    for _ in 1:192
        solve!(l, Val(:flux), 450f0)
    end
    flux = relative_change(unpack(l)[:, :, 1], before)
    with_carry = flux + carry_total(l) / sum(Float64, before)
    # Both the perturbation tracer and a large background must conserve.
    for t in 1:max(Nt, 1)
        after = unpack(l)[:, :, t]
        before_t = cols.rm[:, :, t]
        carry = column_carry(l, t)
        for c in axes(after, 1)
            total = sum(Float64, before_t[c, :])
            residual = sum(Float64, after[c, :]) - total + carry[c]
            @test abs(residual) / total < FLUXFORM_CARRY_LIMIT
        end
    end
    return flux, with_carry
end

@testset "Kz diffusion flux form agrees with Backward Euler" begin
    for FT in (Float64, Float32)
        check_backward_euler(Array; FT)
    end
end

@testset "Kz diffusion keeps uniform columns and isolated layers bit for bit" begin
    check_bit_exact(Array)
end

@testset "Float32 Kz diffusion: flux form and ledger conserve" begin
    # The remaining rounding is held in the outstanding ledger carries.
    for (name, layout, Nt) in LAYOUTS, kz in (0.3, 1.0, 3.0)
        flux, with_carry = day_of_solves(layout, Nt, Array; kz)
        @test abs(flux) < 5e-9
        @test abs(with_carry) < FLUXFORM_CARRY_LIMIT
    end
end

if HAS_CUDA_FLUX_FORM
    @testset "CUDA Kz diffusion flux form" begin
        for FT in (Float64, Float32)
            check_backward_euler(CuArray; FT)
        end
        check_bit_exact(CuArray)
        for (name, layout, Nt) in LAYOUTS
            flux, with_carry = day_of_solves(layout, Nt, CuArray)
            @test abs(flux) < 5e-9
            @test abs(with_carry) < FLUXFORM_CARRY_LIMIT
        end
    end
end

@testset "packed diffusion rejects a workspace without tracer ledger slots before touching the state" begin
    m = ones(Float32, 4, 3, 5)
    rm = fill(4f-4, 4, 3, 5, 2)
    before = copy(rm)
    op = ImplicitVerticalDiffusion(; kz_field = ConstantField{Float32, 3}(50f0))
    @test_throws DimensionMismatch apply_vertical_diffusion_vmr!(rm, m, op, DiffusionWorkspace(m), 900f0)
    @test rm == before
    @test apply_vertical_diffusion_vmr!(similar(rm, 4, 3, 5, 0), m, op, DiffusionWorkspace(m), 900f0) === nothing
end
