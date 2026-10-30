include(joinpath(@__DIR__, "..", "helpers", "conservative_dkg.jl"))

@testset "Conservative Dkg mass solve and transpose" begin
    for FT in (Float32, Float64), Nz in (1, 2, 66), Nt in (1, 7), strength in (0, 0.1, 40, 1e4)
        @testset "$FT Nz=$Nz Nt=$Nt exchange=$strength" begin
            check_conservative_dkg(FT, 2, Nz, Nt, strength)
        end
    end
end

@testset "Dkg resolves weak transfers into empty layers" begin
    for FT in (Float32,Float64)
        check_dkg_weak_exchange(FT)
    end
end

@testset "Split Dkg kernels run on CPU with GPU launch tile shapes" begin
    # Normal CPU dispatch uses the packed column kernel. Launch the split
    # factor/tracer kernels explicitly so CI also checks their indexing and
    # read-only factor reuse, including partially filled CUDA-shaped tiles.
    for FT in (Float32, Float64), (Nc, Nz, Nt) in ((3,1,3), (3,5,7), (35,3,33)),
        strength in (0, 40)
        m, d, original = dkg_mass_fixture(FT, Nc, Nz, Nt, strength)
        actual = map(copy, original)
        field = DkgState.PrecomputedCSDkgField(d)
        ws = DkgDiff.DiffusionWorkspace(m, 1, Nt)
        backend = get_backend(actual[1])
        factor! = DkgDiff._vertical_diffusion_cs_dkg_factors_panels_kernel!(backend, (32,2))
        solve! = DkgDiff._vertical_diffusion_cs_mass_dkg_tracers_panels_kernel!(backend, (32,1,2))
        panels = ntuple(p -> DkgState.panel_field(field, p), 6)
        factor!(ws.factors, m, panels, one(FT), Nz, 1; ndrange=(Nc,Nc,6))
        synchronize(backend)
        factors_before = map(copy, ws.factors)
        solve!(actual, ws.ledger_carry, m, panels, ws.factors, one(FT), Nz, Nt, 1;
               ndrange=(Nc,Nc,6Nt))
        synchronize(backend)
        for p in 1:6
            @test isequal(ws.factors[p], factors_before[p])
            @test actual[p][[1,end],:,:,:] == original[p][[1,end],:,:,:]
            @test actual[p][:,[1,end],:,:] == original[p][:,[1,end],:,:]
            for i in (1,Nc), j in (1,Nc)
                source = original[p][i+1,j+1,:,:]
                expected = dkg_mass_reference(source, m[p][i+1,j+1,:], d[p][i,j,:])
                value = actual[p][i+1,j+1,:,:]
                @test all(isfinite, value)
                @test norm(Float64.(value)-expected) <=
                    (FT == Float32 ? 8e-7 : 5e-12) * norm(expected)
            end
        end
        strength == 0 && @test isequal(actual, original)
    end
end

@testset "Dkg leaves isolated layers bit-exact" begin
    for FT in (Float32,Float64)
        check_dkg_isolated_layers(FT)
    end
end

@testset "Dkg validates all workspace panels before mutation" begin
    m, d, rm = dkg_mass_fixture(Float64, 2, 3, 7, 1)
    op = DkgDiff.ImplicitVerticalDiffusion(; kz_field=DkgState.PrecomputedCSDkgField(d))
    good = DkgDiff.DiffusionWorkspace(m, 1, 7)
    factors = ntuple(p -> p == 6 ? zeros(1,1,1) : good.factors[p], 6)
    bad = DkgDiff.DiffusionWorkspace(factors, good.layer_thickness, good.ledger_carry)
    original = map(copy, rm)
    @test_throws DimensionMismatch DkgDiff.apply_vertical_diffusion_vmr!(rm,m,op,bad,1;halo_width=1)
    @test rm == original
    @test_throws ArgumentError DkgDiff.apply_vertical_diffusion_vmr!(rm,m,op,good,1;halo_width=-1)
    @test rm == original
end

@testset "Dkg zero-carrier convention uses an absorbing empty layer" begin
    m, d, rm = dkg_mass_fixture(Float64, 2, 4, 2, 0.7)
    for p in 1:6
        m[p][2,2,2] = 0
    end
    op = DkgDiff.ImplicitVerticalDiffusion(; kz_field=DkgState.PrecomputedCSDkgField(d))
    ws = DkgDiff.DiffusionWorkspace(m,1,2)
    expected = map(copy, rm)
    for p in 1:6, j in 2:3, i in 2:3
        mass = m[p][i, j, :]
        invmass = [v > 0 ? 1 / v : 0.0 for v in mass]
        exchange = d[p][i-1, j-1, 1:3]
        lower = -exchange .* invmass[2:end]
        upper = -exchange .* invmass[1:end-1]
        diagonal = ones(4)
        diagonal[1:3] .+= exchange .* invmass[1:3]
        diagonal[2:4] .+= exchange .* invmass[2:4]
        matrix = Tridiagonal(lower, diagonal, upper)
        for t in 1:2
            q = [mass[k] > 0 ? rm[p][i, j, k, t] / mass[k] : 0.0 for k in 1:4]
            expected[p][i, j, :, t] .= mass .* (matrix \ q)
        end
    end
    DkgDiff.apply_vertical_diffusion_vmr!(rm,m,op,ws,1;halo_width=1)
    @test all(iszero, rm[1][2, 2, 2, :])
    @test all(p -> isapprox(rm[p],expected[p];rtol=3e-15),1:6)
end

@testset "Dkg closed interfaces do not pass rounding residuals" begin
    for FT in (Float32, Float64)
        m, d, rm = dkg_mass_fixture(FT,2,4,7,40)
        for p in 1:6
            d[p][:,:,2] .= 0
            rm[p][2:3,2:3,3:4,:] .= 0
        end
        op=DkgDiff.ImplicitVerticalDiffusion(;kz_field=DkgState.PrecomputedCSDkgField(d))
        ws=DkgDiff.DiffusionWorkspace(m,1,7)
        for _ in 1:20
            DkgDiff.apply_vertical_diffusion_vmr!(rm,m,op,ws,one(FT);halo_width=1)
        end
        @test all(p -> all(iszero,rm[p][2:3,2:3,3:4,:]),1:6)
    end
end
