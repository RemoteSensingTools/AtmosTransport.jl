# Set CUDA_VISIBLE_DEVICES explicitly and opt in before running.
using Test
if get(ENV, "ATMOSTR_RUN_DKG_GPU_TESTS", "0") != "1"
    @info "Skipping opt-in conservative Dkg GPU tests"
else
    using CUDA
    CUDA.allowscalar(false)
    expected = get(ENV,"ATMOSTR_DKG_GPU_NAME","A100")
    @assert !isempty(expected) && occursin(expected,CUDA.name(CUDA.device()))
    include(joinpath(@__DIR__,"..","helpers","conservative_dkg.jl"))
    include(joinpath(@__DIR__,"..","helpers","ledger_carry.jl"))
    @testset "CUDA Dkg launch preserves serial results and halos" begin
        for FT in (Float32, Float64), (Nc, Nz) in ((1, 1), (3, 66), (35, 3)), Nt in (1, 2, 3, 7, 32, 65)
            m, d, rm = dkg_mass_fixture(FT, Nc, Nz, Nt, 40)
            air, actual = map(CuArray, m), map(CuArray, rm)
            expected = map(copy, actual)
            field = Adapt.adapt(CuArray, DkgState.PrecomputedCSDkgField(d))
            op = DkgDiff.ImplicitVerticalDiffusion(; kz_field=field)
            workspace = DkgDiff.DiffusionWorkspace(air, 1, Nt)
            serial! = DkgDiff._vertical_diffusion_cs_mass_dkg_packed_kernel!(
                get_backend(actual[1]), (8, 8))
            serial_carry = map(c -> zero(c), workspace.ledger_carry)
            for p in 1:6
                serial!(expected[p], serial_carry[p], air[p], DkgState.panel_field(field, p),
                        workspace.factors[p], one(FT), Nz, Nt, 1; ndrange=(Nc, Nc))
            end
            CUDA.synchronize()
            DkgDiff.apply_vertical_diffusion_vmr!(actual, air, op, workspace, one(FT); halo_width=1)
            @test map(Array, actual) == map(Array, expected)
            @test map(Array, workspace.ledger_carry) == map(Array, serial_carry)
        end
    end
    @testset "CUDA conservative Dkg through 65 tracers" begin
        for FT in (Float32,Float64), (Nc,Nz) in ((3,66),(35,3)), Nt in (1,7,32,65), strength in (0,40)
            check_conservative_dkg(FT,Nc,Nz,Nt,strength;array_type=CuArray)
        end
        for FT in (Float32,Float64)
            check_dkg_isolated_layers(FT;array_type=CuArray)
            check_dkg_weak_exchange(FT;array_type=CuArray)
        end
    end
    @testset "CUDA Dkg keeps the column ledger carry" begin
        for FT in (Float32,Float64)
            check_dkg_ledger_carry(CuArray; FT)
        end
    end
end
