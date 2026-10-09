using Test

import AtmosTransport
const AT = AtmosTransport

using .AtmosTransport: AdvectionWorkspace, DiffusionWorkspace,
    ConstantField, ImplicitVerticalDiffusion,
    SplitSurfaceFluxCoupling, DiffusiveSurfaceFluxBoundary,
    SurfaceFluxSource, SurfaceFluxOperator, UpwindScheme
using .AtmosTransport.Operators.Advection: strang_split_mt!
using .AtmosTransport.Operators.Diffusion: apply_vertical_diffusion_vmr!

function _ll_diffusion_fixture(::Type{FT}; Nx=2, Ny=2, Nz=6, Nt=1) where FT
    m = FT[1 + 0.08k for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    rm = Array{FT}(undef, Nx, Ny, Nz, Nt)
    for t in 1:Nt, k in 1:Nz, j in 1:Ny, i in 1:Nx
        rm[i, j, k, t] = m[i, j, k] *
            (FT(0.1t) + exp(-FT(0.4) * (k - FT(Nz + 1) / 2)^2))
    end
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    advection = AdvectionWorkspace(m; n_tracers=Nt)
    diffusion = DiffusionWorkspace(m, Nt)
    fill!(diffusion.layer_thickness, FT(100))
    return rm, m, am, bm, cm, advection, diffusion
end

@testset "LL palindrome diffusion equals its standalone midpoint operator" begin
    FT = Float64
    rm, m, am, bm, cm, advection, diffusion = _ll_diffusion_fixture(FT)
    expected = copy(rm)
    expected_m = copy(m)
    op = ImplicitVerticalDiffusion(; kz_field=ConstantField{FT, 3}(FT(1.5)))

    strang_split_mt!(rm, m, am, bm, cm, UpwindScheme(), advection;
                     diffusion_op=op, diffusion_workspace=diffusion, dt=FT(5))
    apply_vertical_diffusion_vmr!(expected, expected_m, op, diffusion, FT(5))

    @test rm ≈ expected rtol=1e-13 atol=0
    @test m == expected_m
end

@testset "Backward-Euler full and half-step maps differ at second order" begin
    FT = Float64
    function split_error(dt)
        full, m, _, _, _, _, full_ws = _ll_diffusion_fixture(FT)
        half = copy(full)
        half_ws = DiffusionWorkspace(m, size(full, 4))
        fill!(half_ws.layer_thickness, FT(100))
        op = ImplicitVerticalDiffusion(; kz_field=ConstantField{FT, 3}(FT(1.5)))
        apply_vertical_diffusion_vmr!(full, m, op, full_ws, FT(dt))
        apply_vertical_diffusion_vmr!(half, m, op, half_ws, FT(dt / 2))
        apply_vertical_diffusion_vmr!(half, m, op, half_ws, FT(dt / 2))
        return maximum(abs, full .- half)
    end

    large = split_error(4)
    small = split_error(2)
    @test large > small > 0
    @test 2.5 < large / small < 5.5
end

@testset "Diffusive surface coupling adds mass before the implicit solve" begin
    FT = Float64
    rm, m, am, bm, cm, advection, diffusion = _ll_diffusion_fixture(FT)
    fill!(rm, zero(FT))
    dt = FT(3)
    rate = fill(FT(2), size(m, 1), size(m, 2))
    emissions = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rate))
    boundary_op = ImplicitVerticalDiffusion(;
        kz_field=ConstantField{FT, 3}(one(FT)),
        surface_flux_coupling=DiffusiveSurfaceFluxBoundary())

    strang_split_mt!(rm, m, am, bm, cm, UpwindScheme(), advection;
                     diffusion_op=boundary_op,
                     diffusion_workspace=diffusion,
                     emissions_op=emissions,
                     tracer_names=(:CO2,), dt)

    expected = zeros(FT, size(rm))
    expected[:, :, end, 1] .= rate .* dt
    reference_op = ImplicitVerticalDiffusion(;
        kz_field=ConstantField{FT, 3}(one(FT)),
        surface_flux_coupling=SplitSurfaceFluxCoupling())
    apply_vertical_diffusion_vmr!(expected, m, reference_op, diffusion, dt)

    @test rm ≈ expected rtol=1e-13 atol=0
end
