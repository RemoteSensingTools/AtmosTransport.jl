using AtmosTransport, Test, Random, LinearAlgebra, Adapt
using KernelAbstractions: get_backend, synchronize
const DkgDiff = AtmosTransport.Operators.Diffusion
const DkgState = AtmosTransport.State
const DkgAdj = AtmosTransport.Adjoints

function dkg_mass_fixture(FT, Nc, Nz, Nt, strength)
    rng = MersenneTwister(1973)
    Hp, N = 1, Nc + 2
    m = ntuple(_ -> FT(1e10) .* (FT(1) .+ FT(30) .* rand(rng, FT, N, N, Nz)), 6)
    d = ntuple(p -> [k < Nz ? FT(strength) * min(m[p][i+Hp,j+Hp,k], m[p][i+Hp,j+Hp,k+1]) : zero(FT)
                     for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6)
    rm = ntuple(_ -> fill(FT(17), N, N, Nz, Nt), 6)
    for p in 1:6, t in 1:Nt, k in 1:Nz, j in 1:Nc, i in 1:Nc
        q = t == 7 ? -FT(4e-4) + FT(1e-9) * randn(rng, FT) : t == 6 ? (k == cld(Nz,2) ? FT(0.4) : zero(FT)) : t % 5 == 0 ? zero(FT) : t % 5 == 1 ? rand(rng, FT) :
            t % 5 == 2 ? randn(rng, FT) : t % 5 == 3 ? FT(4e-4) :
            FT(4e-4) + FT(1e-9) * randn(rng, FT)
        rm[p][i+Hp,j+Hp,k,t] = m[p][i+Hp,j+Hp,k] * q
    end
    return m, d, rm
end

# Independent direct-mass backward-Euler matrix (column sums equal one).
function dkg_mass_reference(rm, m, d)
    n = length(m)
    u = Float64.(d[1:n-1]) ./ Float64.(m[1:n-1])
    v = Float64.(d[1:n-1]) ./ Float64.(m[2:n])
    diagonal = ones(n)
    diagonal[1:n-1] .+= u
    diagonal[2:n] .+= v
    return Tridiagonal(-u, diagonal, -v) \ Float64.(rm)
end

function check_conservative_dkg(FT, Nc, Nz, Nt, strength; array_type=Array)
    m, d, rm = dkg_mass_fixture(FT, Nc, Nz, Nt, strength)
    device_m, device_rm = map(array_type, m), map(array_type, rm)
    field = Adapt.adapt(array_type, DkgState.PrecomputedCSDkgField(d))
    op = DkgDiff.ImplicitVerticalDiffusion(; kz_field=field)
    ws = DkgDiff.DiffusionWorkspace(device_m, 1, Nt)
    DkgDiff.apply_vertical_diffusion_vmr!(device_rm, device_m, op, ws, one(FT); halo_width=1)
    actual = map(Array, device_rm)
    strength == 0 && @test all(p -> isequal(actual[p], rm[p]), 1:6)
    tol = FT == Float32 ? 8e-7 : 5e-12
    for p in 1:6
        @test actual[p][1,:,:,:] == rm[p][1,:,:,:]
        for j in 1:Nc, i in 1:Nc
            source = rm[p][i+1,j+1,:,:]
            expected = dkg_mass_reference(source, m[p][i+1,j+1,:], d[p][i,j,:])
            value = actual[p][i+1,j+1,:,:]
            @test all(isfinite, value)
            @test norm(Float64.(value) - expected) <= tol * norm(expected)
            for t in 1:Nt
                # Column mass ledger: exact up to rounding the residual into one layer.
                @test abs(sum(big, value[:,t]) - sum(big, source[:,t])) <=
                    eps(FT) * maximum(abs, value[:,t])
                t % 5 != 2 && @test minimum(value[:,t]) >= 0
                t % 5 == 0 && @test all(iszero, value[:,t])
                t % 5 == 3 && @test maximum(abs, value[:,t] ./ m[p][i+1,j+1,:] .- FT(4e-4)) <=
                    (FT == Float32 ? 8eps(FT) : 2e-14) * FT(4e-4)
            end
        end
    end
    # Scalar and packed kernels retain identical arithmetic, including signed data.
    for t in unique((1, min(2, Nt)))
        scalar = map(p -> array_type(p[:,:,:,t]), rm)
        DkgDiff.apply_vertical_diffusion_vmr!(scalar, device_m, op, ws, one(FT); halo_width=1)
        @test map(Array, scalar) == map(p -> p[:,:,:,t], actual)
    end

    rng = MersenneTwister(1974)
    seed = ntuple(_ -> randn(rng, FT, Nc+2, Nc+2, Nz), 6)
    gradient = map(array_type, seed)
    kernel! = DkgAdj._vertical_diffusion_cs_single_dkg_adjoint_kernel!(get_backend(gradient[1]), (8,8))
    for p in 1:6
        kernel!(gradient[p], device_m[p], DkgState.panel_field(field,p), ws.factors[p],
                one(FT), Nz, 1; ndrange=(Nc,Nc))
    end
    synchronize(get_backend(gradient[1]))
    gradient = map(Array, gradient)
    strength == 0 && @test all(p -> isequal(gradient[p], seed[p]), 1:6)
    t = min(2, Nt)
    inside = a -> view(a, 2:Nc+1, 2:Nc+1, :)
    lhs = sum(dot(Float64.(inside(seed[p])), Float64.(inside(actual[p][:,:,:,t]))) for p in 1:6)
    rhs = sum(dot(Float64.(inside(gradient[p])), Float64.(inside(rm[p][:,:,:,t]))) for p in 1:6)
    scale = sum(norm(Float64.(inside(seed[p]))) * norm(Float64.(inside(rm[p][:,:,:,t]))) for p in 1:6)
    @test abs(lhs-rhs) <= (FT == Float32 ? 5e-7 : 3e-15) * scale

    # Total mass has a constant unit gradient, preserved exactly by the transpose.
    constant = map(p -> array_type(ones(FT, size(p))), m)
    for p in 1:6
        kernel!(constant[p], device_m[p], DkgState.panel_field(field,p), ws.factors[p],
                one(FT), Nz, 1; ndrange=(Nc,Nc))
    end
    synchronize(get_backend(constant[1]))
    @test all(p -> all(==(one(FT)), inside(Array(p))), constant)
    return nothing
end

function check_dkg_isolated_layers(FT; array_type=Array)
    m,d,rm=dkg_mass_fixture(FT,2,4,7,40)
    for p in 1:6
        d[p][:,:,2:3] .= 0
    end
    device_m,device_rm=map(array_type,m),map(array_type,rm)
    field=Adapt.adapt(array_type,DkgState.PrecomputedCSDkgField(d))
    op=DkgDiff.ImplicitVerticalDiffusion(;kz_field=field)
    ws=DkgDiff.DiffusionWorkspace(device_m,1,7)
    for _ in 1:20
        DkgDiff.apply_vertical_diffusion_vmr!(device_rm,device_m,op,ws,one(FT);halo_width=1)
    end
    actual=map(Array,device_rm)
    @test all(p -> isequal(actual[p][2:3,2:3,3:4,:],rm[p][2:3,2:3,3:4,:]),1:6)
end

function check_dkg_weak_exchange(FT; array_type=Array)
    for strength in (1e-2, 1e-8, 1e-14)
        m = ntuple(_ -> fill(FT(1e10),3,3,2),6)
        d = ntuple(_ -> reshape(FT[FT(strength)*FT(1e10),0],1,1,2),6)
        rm = ntuple(_ -> zeros(FT,3,3,2,2),6)
        for p in 1:6
            rm[p][2,2,1,1] = FT(4e6)
            rm[p][2,2,2,2] = -FT(4e6)
        end
        device_m,device_rm = map(array_type,m),map(array_type,rm)
        field = Adapt.adapt(array_type,DkgState.PrecomputedCSDkgField(d))
        op = DkgDiff.ImplicitVerticalDiffusion(;kz_field=field)
        ws = DkgDiff.DiffusionWorkspace(device_m,1,2)
        DkgDiff.apply_vertical_diffusion_vmr!(device_rm,device_m,op,ws,one(FT);halo_width=1)
        actual = map(Array,device_rm)
        u = Float64(d[1][1,1,1])/Float64(m[1][2,2,1])
        weight = u/(1+2u)
        expected = Float64(FT(4e6))*weight
        tolerance = FT == Float32 ? 6eps(FT) : 4e-15
        for p in 1:6
            # A global field norm would hide complete loss of these recipients.
            @test actual[p][2,2,2,1] ≈ expected rtol=tolerance atol=0
            @test actual[p][2,2,1,2] ≈ -expected rtol=tolerance atol=0
        end
        lambda = ntuple(_ -> zeros(FT,3,3,2),6)
        for p in 1:6
            lambda[p][2,2,2] = 1
        end
        gradient = map(array_type,lambda)
        kernel! = DkgAdj._vertical_diffusion_cs_single_dkg_adjoint_kernel!(get_backend(gradient[1]),(8,8))
        for p in 1:6
            kernel!(gradient[p],device_m[p],DkgState.panel_field(field,p),ws.factors[p],
                    one(FT),2,1;ndrange=(1,1))
        end
        synchronize(get_backend(gradient[1]))
        @test Array(gradient[1])[2,2,1] ≈ weight rtol=tolerance atol=0
    end
end
