#!/usr/bin/env julia
"""
Tests for the mass-flux reformulation of the CS vertical-diffusion
kernel (D1 fix). Two properties are verified end-to-end:

  1. Column tracer-mass conservation to roundoff for inert tracers.
     The legacy geometric kernel `D = Kz/(dz·dz)` conserved `Σ q·dz` but
     leaked `Σ m·q` whenever density varied with height. The new mass-flux
     kernel makes the implicit matrix column-stochastic on `rm = m·q`,
     so `Σ m·q_new = Σ m·q_old` exactly.
  2. Adjoint identity `⟨y, L·x⟩ = ⟨Lᵀ·y, x⟩` on the cubed-sphere panel
     path. The forward+adjoint share the same `(dkg, m, dt)` ingredients
     and the adjoint kernel implements the transpose of the column-
     stochastic A; the identity should hold to roundoff.

Both checks use a small CS panel with a non-uniform Kz profile and a
non-uniform air-mass column (density decreases with height) so that
the OLD geometric kernel would have visibly leaked mass.
"""

using Test
using Random
using LinearAlgebra: dot

using AtmosTransport
using .AtmosTransport.Operators: ImplicitVerticalDiffusion, DiffusionWorkspace,
                                  NoSurfaceFlux
using .AtmosTransport.Operators.Diffusion: apply_vertical_diffusion_vmr!,
                                            _cs_scale_tracer_mass_to_vmr!,
                                            _cs_scale_vmr_to_tracer_mass!
using .AtmosTransport.State: PreComputedKzField, CubedSphereField, panel_field
using .AtmosTransport.Adjoints: _vertical_diffusion_cs_single_adjoint_kernel!
using .AtmosTransport.Grids: CubedSphereMesh

using KernelAbstractions: get_backend, synchronize

# Build a flat (non-CS) air-mass column for LL/RG: density decreases with
# altitude exponentially.
function _build_air_mass_column_flat(::Type{FT}, shape::Tuple) where {FT}
    arr = zeros(FT, shape)
    Nz = shape[end]
    @views for k in 1:Nz
        scale = exp(FT(k - Nz) * FT(0.05))      # surface (k=Nz) → 1
        selectdim(arr, length(shape), k) .= FT(1e15) * scale
    end
    return arr
end

# Build a CS air-mass profile that decreases with height (typical
# tropospheric column) so the OLD geometric kernel would leak.
function _build_air_mass_column(::Type{FT}, Nc::Int, Hp::Int, Nz::Int) where {FT}
    N = Nc + 2 * Hp
    panels = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    # In our orientation (k=1=TOA, k=Nz=surface), mass should be largest at
    # k=Nz and smallest at k=1. `(k - Nz)` is 0 at the surface and negative
    # at TOA; the positive exponent puts the largest value at the surface.
    for p in 1:6, k in 1:Nz
        scale = exp(FT(k - Nz) * FT(0.05))   # k=Nz → 1, k=1 → exp(-(Nz-1)·0.05)
        panels[p][:, :, k] .= FT(1e15) * scale
    end
    return panels
end

function _build_kz_field(::Type{FT}, Nc::Int, Hp::Int, Nz::Int) where {FT}
    N = Nc + 2 * Hp
    panel_arrays = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    # Non-uniform Kz: peaks in mid-PBL, decays toward TOA + surface
    for p in 1:6, k in 1:Nz
        z_frac = FT(k - 1) / FT(Nz - 1)        # 0 at TOA, 1 at surface
        # Triangular profile peaking at z_frac ≈ 0.7 (lower troposphere)
        bump = max(zero(FT), one(FT) - abs(z_frac - FT(0.7)) / FT(0.3))
        panel_arrays[p][:, :, k] = fill(FT(50.0) * bump + FT(0.5), Nc, Nc)
    end
    panel_fields = ntuple(p -> PreComputedKzField(panel_arrays[p]), 6)
    return CubedSphereField(panel_fields)
end

# Build dz_scratch (constant T_ref hydrostatic) per the current code path.
function _build_dz_scratch(::Type{FT}, Nc::Int, Nz::Int) where {FT}
    # Simple uniform per-cell dz that increases slightly with altitude
    # (matches a typical atmospheric hydrostatic column).
    out = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    for p in 1:6, k in 1:Nz
        # Layer thickness grows from ~100 m near surface to ~2000 m near TOA
        out[p][:, :, k] .= FT(100) + FT(1900) * (one(FT) - FT(k - 1) / FT(Nz - 1))
    end
    return out
end

@testset "CS mass-flux diffusion — column tracer-mass conservation" begin
    FT = Float64
    Nc, Hp, Nz = 4, 1, 8
    N = Nc + 2 * Hp

    panels_m = _build_air_mass_column(FT, Nc, Hp, Nz)
    kz_field = _build_kz_field(FT, Nc, Hp, Nz)
    dz_scratch = _build_dz_scratch(FT, Nc, Nz)
    w_scratch = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)

    # Random initial tracer mass on the interior cells.
    rng = MersenneTwister(2026)
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    for p in 1:6, k in 1:Nz,
        j in (Hp + 1):(Hp + Nc),
        i in (Hp + 1):(Hp + Nc)
        panels_rm[p][i, j, k] = abs(randn(rng, FT))
    end

    # Pre-step total per column (interior only).
    sum_pre = ntuple(p -> dropdims(sum(@view(panels_rm[p][(Hp+1):(Hp+Nc),
                                                          (Hp+1):(Hp+Nc), :]);
                                        dims = 3); dims = 3), 6)

    # Build the operator + workspace.
    op = ImplicitVerticalDiffusion(; kz_field = kz_field)
    workspace = DiffusionWorkspace(w_scratch, dz_scratch, nothing)

    apply_vertical_diffusion_vmr!(panels_rm, panels_m, op, workspace,
                                   FT(450.0); halo_width = Hp)

    sum_post = ntuple(p -> dropdims(sum(@view(panels_rm[p][(Hp+1):(Hp+Nc),
                                                            (Hp+1):(Hp+Nc), :]);
                                         dims = 3); dims = 3), 6)

    for p in 1:6, j in 1:Nc, i in 1:Nc
        @test isapprox(sum_pre[p][i, j], sum_post[p][i, j];
                       rtol = 1e-12, atol = 0)
    end
end

@testset "CS packed diffusion matches independent tracer solves" begin
    for FT in (Float32, Float64)
        Nc, Hp, Nz, Nt = 3, 1, 7, 4
        N = Nc + 2 * Hp
        panels_m = _build_air_mass_column(FT, Nc, Hp, Nz)
        kz_field = _build_kz_field(FT, Nc, Hp, Nz)
        dz_scratch = _build_dz_scratch(FT, Nc, Nz)
        rng = MersenneTwister(1701)
        packed = ntuple(_ -> abs.(randn(rng, FT, N, N, Nz, Nt)), 6)
        expected = ntuple(p -> copy(packed[p]), 6)
        op = ImplicitVerticalDiffusion(; kz_field)
        single_workspace = DiffusionWorkspace(
            ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6), dz_scratch, nothing)

        for t in 1:Nt
            tracer = ntuple(p -> @view(expected[p][:, :, :, t]), 6)
            apply_vertical_diffusion_vmr!(tracer, panels_m, op,
                                           single_workspace, FT(450);
                                           halo_width = Hp)
        end

        packed_workspace = DiffusionWorkspace(
            ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6), dz_scratch,
            ntuple(_ -> zeros(FT, Nc, Nc, Nt), 6))
        apply_vertical_diffusion_vmr!(packed, panels_m, op, packed_workspace,
                                       FT(450); halo_width = Hp)

        @test packed == expected
    end
end

@testset "CS mass-flux diffusion — adjoint identity ⟨y, L·x⟩ = ⟨Lᵀ·y, x⟩" begin
    FT = Float64
    Nc, Hp, Nz = 4, 1, 8
    N = Nc + 2 * Hp
    mesh = CubedSphereMesh(; Nc = Nc, Hp = Hp, FT = FT)

    panels_m = _build_air_mass_column(FT, Nc, Hp, Nz)
    kz_field = _build_kz_field(FT, Nc, Hp, Nz)
    dz_scratch = _build_dz_scratch(FT, Nc, Nz)
    w_scratch = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)

    op = ImplicitVerticalDiffusion(; kz_field = kz_field)
    workspace = DiffusionWorkspace(w_scratch, dz_scratch, nothing)

    # Random x (tracer-mass-like) and adjoint seed y on the interior.
    rng = MersenneTwister(31337)
    x_rm   = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    y_seed = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    for p in 1:6, k in 1:Nz,
        j in (Hp + 1):(Hp + Nc),
        i in (Hp + 1):(Hp + Nc)
        x_rm[p][i, j, k]   = randn(rng, FT)
        y_seed[p][i, j, k] = randn(rng, FT)
    end

    dt = FT(450.0)

    # Forward: copy x → rm, apply L (= forward diffusion).
    rm_new = ntuple(p -> copy(x_rm[p]), 6)
    apply_vertical_diffusion_vmr!(rm_new, panels_m, op, workspace, dt;
                                   halo_width = Hp)

    # Adjoint: copy y → λ, apply Lᵀ. Direct call to the CS adjoint kernel
    # which mirrors the forward's mass-flux coefficient construction.
    lambda = ntuple(p -> copy(y_seed[p]), 6)
    backend = get_backend(lambda[1])
    kernel = _vertical_diffusion_cs_single_adjoint_kernel!(backend, (8, 8))
    @inbounds for p in 1:6
        kernel(lambda[p], panels_m[p],
               panel_field(kz_field, p),
               dz_scratch[p], w_scratch[p], FT(dt), Nz, Hp;
               ndrange = (Nc, Nc))
        synchronize(backend)
    end

    # Inner products over the interior.
    function inner_interior(a, b)
        s = zero(FT)
        @inbounds for p in 1:6, k in 1:Nz,
            j in (Hp + 1):(Hp + Nc),
            i in (Hp + 1):(Hp + Nc)
            s += a[p][i, j, k] * b[p][i, j, k]
        end
        return s
    end
    lhs = inner_interior(y_seed, rm_new)   # ⟨y, L·x⟩
    rhs = inner_interior(lambda, x_rm)     # ⟨Lᵀ·y, x⟩

    @test isapprox(lhs, rhs; rtol = 1e-10, atol = 1e-10 * abs(lhs))
end

# ─────────────────────────────────────────────────────────────────────
# LL packed + RG (face-indexed) — D1 follow-up. Same conservation bar
# as CS but on the structured / face-indexed apply paths via the new
# `apply_vertical_diffusion_vmr!` wrappers.
# ─────────────────────────────────────────────────────────────────────

using .AtmosTransport.Operators.Diffusion: apply_vertical_diffusion_vmr!
using .AtmosTransport.State: ConstantField

@testset "LL packed mass-flux diffusion — column tracer-mass conservation" begin
    FT = Float64
    Nx, Ny, Nz, Nt = 4, 4, 8, 2

    air_mass = _build_air_mass_column_flat(FT, (Nx, Ny, Nz))

    # Non-uniform Kz at cell centers.
    Kz_arr = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz
        z_frac = FT(k - 1) / FT(Nz - 1)
        bump = max(zero(FT), one(FT) - abs(z_frac - FT(0.7)) / FT(0.3))
        Kz_arr[:, :, k] .= FT(50) * bump + FT(0.5)
    end
    kz_field = PreComputedKzField(Kz_arr)

    # Non-uniform dz hydrostatic-ish.
    dz_scratch = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz
        dz_scratch[:, :, k] .= FT(100) + FT(1900) * (one(FT) - FT(k - 1) / FT(Nz - 1))
    end
    w_scratch = zeros(FT, Nx, Ny, Nz)

    # Random tracer mass.
    rng = MersenneTwister(2026)
    rm = abs.(randn(rng, FT, Nx, Ny, Nz, Nt))

    sum_pre = dropdims(sum(rm; dims = 3); dims = 3)   # (Nx, Ny, Nt)

    op = ImplicitVerticalDiffusion(; kz_field = kz_field)
    workspace = DiffusionWorkspace(w_scratch, dz_scratch, zeros(FT, Nx, Ny, Nt))
    apply_vertical_diffusion_vmr!(rm, air_mass, op, workspace, FT(450.0))

    sum_post = dropdims(sum(rm; dims = 3); dims = 3)
    for t in 1:Nt, j in 1:Ny, i in 1:Nx
        @test isapprox(sum_pre[i, j, t], sum_post[i, j, t];
                       rtol = 1e-12, atol = 0)
    end
end

@testset "RG (face-indexed) packed mass-flux diffusion — tracer-mass conservation" begin
    FT = Float64
    ncells, Nz, Nt = 12, 8, 2

    air_mass = _build_air_mass_column_flat(FT, (ncells, Nz))

    Kz_arr = zeros(FT, ncells, Nz)
    for k in 1:Nz
        z_frac = FT(k - 1) / FT(Nz - 1)
        bump = max(zero(FT), one(FT) - abs(z_frac - FT(0.7)) / FT(0.3))
        Kz_arr[:, k] .= FT(50) * bump + FT(0.5)
    end
    kz_field = PreComputedKzField(Kz_arr)

    dz_scratch = zeros(FT, ncells, Nz)
    for k in 1:Nz
        dz_scratch[:, k] .= FT(100) + FT(1900) * (one(FT) - FT(k - 1) / FT(Nz - 1))
    end
    w_scratch = zeros(FT, ncells, Nz)

    rng = MersenneTwister(101)
    rm = abs.(randn(rng, FT, ncells, Nz, Nt))
    sum_pre = dropdims(sum(rm; dims = 2); dims = 2)   # (ncells, Nt)

    op = ImplicitVerticalDiffusion(; kz_field = kz_field)
    workspace = DiffusionWorkspace(w_scratch, dz_scratch, zeros(FT, ncells, Nt))
    apply_vertical_diffusion_vmr!(rm, air_mass, op, workspace, FT(450.0))

    sum_post = dropdims(sum(rm; dims = 2); dims = 2)
    for t in 1:Nt, c in 1:ncells
        @test isapprox(sum_pre[c, t], sum_post[c, t]; rtol = 1e-12, atol = 0)
    end
end

@testset "RG (face-indexed) single mass-flux diffusion — tracer-mass conservation" begin
    FT = Float64
    ncells, Nz = 12, 8

    air_mass = _build_air_mass_column_flat(FT, (ncells, Nz))

    Kz_arr = zeros(FT, ncells, Nz)
    for k in 1:Nz
        z_frac = FT(k - 1) / FT(Nz - 1)
        bump = max(zero(FT), one(FT) - abs(z_frac - FT(0.7)) / FT(0.3))
        Kz_arr[:, k] .= FT(50) * bump + FT(0.5)
    end
    kz_field = PreComputedKzField(Kz_arr)

    dz_scratch = zeros(FT, ncells, Nz)
    for k in 1:Nz
        dz_scratch[:, k] .= FT(100) + FT(1900) * (one(FT) - FT(k - 1) / FT(Nz - 1))
    end
    w_scratch = zeros(FT, ncells, Nz)

    rng = MersenneTwister(73)
    rm = abs.(randn(rng, FT, ncells, Nz))
    sum_pre = dropdims(sum(rm; dims = 2); dims = 2)   # (ncells,)

    op = ImplicitVerticalDiffusion(; kz_field = kz_field)
    workspace = DiffusionWorkspace(w_scratch, dz_scratch, nothing)
    apply_vertical_diffusion_vmr!(rm, air_mass, op, workspace, FT(450.0))

    sum_post = dropdims(sum(rm; dims = 2); dims = 2)
    for c in 1:ncells
        @test isapprox(sum_pre[c], sum_post[c]; rtol = 1e-12, atol = 0)
    end
end
