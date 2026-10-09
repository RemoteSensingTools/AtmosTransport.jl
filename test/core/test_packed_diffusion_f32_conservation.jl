# Lat-lon and reduced-Gaussian diffusion in Float32 must conserve each tracer's
# column mass as well as the cubed-sphere kernels: the solve works on the
# departure from each column's minimum (the operator keeps a uniform column),
# so a 400 ppm background does not swamp the rounding of a small anomaly.

using Test
using AtmosTransport
using .AtmosTransport.Operators: ImplicitVerticalDiffusion, DiffusionWorkspace
using .AtmosTransport.Operators.Diffusion: apply_vertical_diffusion_vmr!
using .AtmosTransport.State: ConstantField

# Air mass and tracer mass on `dims = (columns..., Nz, Nt)`: tracer t has a background
# of (400 + 10t) ppm and a surface anomaly that varies from column to column.
function background_column_mass(::Type{FT}, dims) where FT
    columns, Nz, Nt = dims[1:end-2], dims[end-1], dims[end]
    m = [FT(1e4 * (1 + 0.1k)) for _ in CartesianIndices(columns), k in 1:Nz]
    m = reshape(m, columns..., Nz)
    rm = similar(m, FT, dims)
    for I in CartesianIndices(rm)
        idx = Tuple(I)
        c, k, t = idx[1:end-2], idx[end-1], idx[end]
        anomaly = k == Nz ? FT(1e-6) * (1 + sum(c) % 5) * t : zero(FT)
        rm[I] = m[c..., k] * (FT(4e-4 + 1e-5t) + anomaly)
    end
    return rm, m
end

# Relative change of every column's mass of every tracer after one day of 15-minute steps.
function column_mass_change(rm, m, ws)
    N = ndims(m)
    op = ImplicitVerticalDiffusion(; kz_field = ConstantField{Float32, N}(50f0))
    r = copy(rm)
    for _ in 1:96
        apply_vertical_diffusion_vmr!(r, m, op, ws, 900f0)
    end
    column_sum(x) = sum(Float64, x; dims = N)
    return maximum(abs, column_sum(r) ./ column_sum(rm) .- 1)
end

@testset "LL/RG diffusion conserves background-dominated tracers in Float32" begin
    for dims in ((12, 8, 20, 3), (40, 20, 3))                  # lat-lon (Nx, Ny, Nz, Nt), reduced-Gaussian (ncells, Nz, Nt)
        rm, m = background_column_mass(Float32, dims)
        ws = DiffusionWorkspace(m, dims[end])
        fill!(ws.layer_thickness, 300f0)
        @test column_mass_change(rm, m, ws) < 2e-7
    end
    # reduced-Gaussian single-tracer kernel (2-D tracer array)
    rm, m = background_column_mass(Float32, (40, 20, 1))
    ws = DiffusionWorkspace(m)
    fill!(ws.layer_thickness, 300f0)
    @test column_mass_change(rm[:, :, 1], m, ws) < 2e-7
end

@testset "packed diffusion rejects a workspace without tracer references before touching the state" begin
    rm, m = background_column_mass(Float32, (4, 3, 5, 2))
    before = copy(rm)
    op = ImplicitVerticalDiffusion(; kz_field = ConstantField{Float32, 3}(50f0))
    @test_throws DimensionMismatch apply_vertical_diffusion_vmr!(rm, m, op, DiffusionWorkspace(m), 900f0)
    @test rm == before
    @test apply_vertical_diffusion_vmr!(similar(rm, 4, 3, 5, 0), m, op, DiffusionWorkspace(m), 900f0) === nothing
end
