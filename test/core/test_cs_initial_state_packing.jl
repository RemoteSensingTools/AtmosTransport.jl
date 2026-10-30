using AtmosTransport, Test
import NCDatasets: NCDataset, defDim, defVar
const PackingIC = AtmosTransport.Models.InitialConditionIO
const PackingRunner = AtmosTransport.Models.DrivenRunner

@testset "Cubed-sphere initialization into packed storage" begin
    @testset "Signed dry/moist conversion and independent slots" begin
        for FT in (Float32, Float64), Hp in (0, 2)
            Nc, Nz = 3, 4
            mesh = CubedSphereMesh(; Nc, Hp, FT)
            grid = AtmosGrid(mesh, HybridSigmaPressure(zeros(FT,Nz+1),
                             FT.(range(0,1;length=Nz+1))), CPU(); FT)
            air = ntuple(p -> FT[100p + 10k + 2j + i
                                for i in 1:Nc+2Hp, j in 1:Nc+2Hp, k in 1:Nz], 6)
            vmr = ntuple(p -> FT[(-1)^(i+j+k+p) * (p+i+j+k)*1e-6
                                for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6)
            qv = ntuple(p -> FT[(p+i+j+k)*1e-3
                               for i in 1:Nc+2Hp, j in 1:Nc+2Hp, k in 1:Nz], 6)
            originals = deepcopy((air, vmr, qv))
            for humidity in (nothing, qv)
                raw = ntuple(p -> fill(FT(-99), size(air[p])..., 3), 6)
                destination = map(a -> selectdim(a, 4, 2), raw)
                result = PackingIC._cs_pack_interior_into_halo!(destination, grid, air, vmr, humidity)
                @test result === destination
                allocated = pack_initial_tracer_mass(grid, air, vmr;
                    mass_basis = humidity === nothing ? DryBasis() : MoistBasis(), qv=humidity)
                for p in 1:6
                    expected = zeros(FT, size(air[p]))
                    for k in 1:Nz, j in 1:Nc, i in 1:Nc
                        value = vmr[p][i,j,k] * air[p][Hp+i,Hp+j,k]
                        expected[Hp+i,Hp+j,k] = humidity === nothing ? value :
                            value * (1 - humidity[p][Hp+i,Hp+j,k])
                    end
                    @test destination[p] == expected
                    @test allocated[p] == expected
                    @test all(==(FT(-99)), raw[p][:,:,:,1])
                    @test all(==(FT(-99)), raw[p][:,:,:,3])
                end
                @test isequal((air,vmr,qv), originals)
            end
            bad_out = ntuple(_ -> zeros(FT,1,1,1),6)
            @test_throws DimensionMismatch PackingIC._cs_pack_interior_into_halo!(bad_out, grid, air, vmr, nothing)
            bad_vmr = ntuple(_ -> zeros(FT,1,1,1),6)
            @test_throws DimensionMismatch pack_initial_tracer_mass(grid, air, bad_vmr; mass_basis=DryBasis())
            bad_qv = ntuple(_ -> zeros(FT,1,1,1),6)
            @test_throws DimensionMismatch pack_initial_tracer_mass(grid, air, vmr; mass_basis=MoistBasis(),qv=bad_qv)
        end
    end

    @testset "Runner preserves packed order and per-tracer values" begin
        for FT in (Float32, Float64), nt in (1, 6, 7, 17, 32, 64)
            Nc, Hp, Nz = 3, 1, 4
            mesh = CubedSphereMesh(; Nc, Hp, FT)
            grid = AtmosGrid(mesh, HybridSigmaPressure(zeros(FT,Nz+1),
                             FT.(range(0,1;length=Nz+1))), CPU(); FT)
            air = ntuple(p -> fill(FT(1000+p),Nc+2Hp,Nc+2Hp,Nz),6)
            ps = ntuple(p -> fill(FT(80000+1000p),Nc,Nc),6)
            # The packed order is the iteration order of the specs (the runner
            # passes them sorted by name); reversed here so a sorting or
            # dictionary round trip inside the initializer would show.
            configs = [Symbol("tracer", lpad(i,2,'0')) =>
                (i % 3 == 0 ? Dict{String,Any}("kind"=>"pressure_layer", "psurf_fraction"=>0.5,
                                               "total_molecules"=>1e28) :
                              Dict{String,Any}("kind"=>"uniform", "background"=>(-1)^i * i*1e-6))
                for i in nt:-1:1]
            masses = [name => pack_initial_tracer_mass(grid,air,
                          build_initial_mixing_ratio(air,grid,cfg;surface_pressure=ps);
                          mass_basis=DryBasis())
                      for (name, cfg) in configs]
            reference = CubedSphereState(DryBasis,mesh,air;masses...)
            state = PackingRunner._initialize_cs_dry_state(grid,air,configs;surface_pressure=ps)
            @test tracer_names(state) == Tuple(first.(configs))
            @test tracer_names(state) == tracer_names(reference)
            @test state.air_mass === air
            @test AtmosTransport.State.halo_width(state) == Hp
            for name in tracer_names(state)
                @test get_tracer(state,name) == get_tracer(reference,name)
            end
            @test all(size(a) == (Nc+2Hp,Nc+2Hp,Nz,nt) for a in state.tracers_raw)
            first_name = first(tracer_names(state))
            get_tracer(state,first_name)[1][Hp+1,Hp+1,1] = FT(123)
            @test get_tracer(reference,first_name)[1][Hp+1,Hp+1,1] != FT(123)
            @test air[1][Hp+1,Hp+1,1] == FT(1001)
            @test_throws ArgumentError PackingRunner._initialize_cs_dry_state(grid,air,Dict();surface_pressure=ps)
            @test_throws ArgumentError PackingRunner._initialize_cs_dry_state(
                grid,air,[configs[1], configs[1]];surface_pressure=ps)
        end
    end
end

@testset "Private CS VMR reuse preserves public ownership and mixed initializers" begin
    for FT in (Float32, Float64)
        Nc, Hp, Nz = 3, 1, 4
        mesh = CubedSphereMesh(; Nc, Hp, FT)
        grid = AtmosGrid(mesh, HybridSigmaPressure(zeros(FT, Nz + 1),
                         FT.(range(0, 1; length=Nz + 1))), CPU(); FT)
        air = ntuple(p -> fill(FT(1000 + p), Nc + 2Hp, Nc + 2Hp, Nz), 6)
        ps = ntuple(p -> fill(FT(80000 + 1000p), Nc, Nc), 6)
        configs = [
            Dict{String, Any}("kind"=>"uniform", "background"=>-2e-4),
            Dict{String, Any}("kind"=>"pressure_layer", "psurf_fraction"=>0.2, "total_molecules"=>1e28),
            Dict{String, Any}("kind"=>"gaussian_blob", "background"=>-1e-4, "amplitude"=>3e-4),
            Dict{String, Any}("kind"=>"latitude_step", "background"=>2e-4),
            Dict{String, Any}("kind"=>"pressure_layer", "lowest_layer"=>true, "total_molecules"=>1e27),
        ]
        scratch = ntuple(_ -> fill(FT(NaN), Nc, Nc, Nz), 6)
        first_owned = build_initial_mixing_ratio(air, grid, first(configs); surface_pressure=ps)
        first_values = map(copy, first_owned)
        for cfg in configs
            expected = build_initial_mixing_ratio(air, grid, cfg; surface_pressure=ps)
            result = PackingIC._build_cs_initial_mixing_ratio(air, grid, cfg, scratch; surface_pressure=ps)
            @test result === scratch
            @test result == expected
            @test all(result[p] !== expected[p] for p in 1:6)
            @test first_owned == first_values
        end
        # Validate every reuse panel before writing any, including the last one.
        bad = ntuple(p -> p == 6 ? zeros(FT, 1, 1, 1) : copy(scratch[p]), 6)
        original = map(copy, bad)
        @test_throws DimensionMismatch PackingIC._build_cs_initial_mixing_ratio(
            air, grid, first(configs), bad; surface_pressure=ps)
        @test bad == original

        specs = Dict(Symbol("tracer", i) => cfg for (i, cfg) in enumerate(configs))
        state = PackingRunner._initialize_cs_dry_state(grid, air, specs; surface_pressure=ps)
        for (name, cfg) in specs
            vmr = build_initial_mixing_ratio(air, grid, cfg; surface_pressure=ps)
            expected = pack_initial_tracer_mass(grid, air, vmr; mass_basis=DryBasis())
            @test get_tracer(state, name) == expected
        end
    end
end

@testset "Native CS file initialization can replace private reuse buffers" begin
    for FT in (Float32, Float64)
        Nc, Hp, Nz = 2, 1, 3
        mesh = CubedSphereMesh(; Nc, Hp, FT)
        grid = AtmosGrid(mesh, HybridSigmaPressure(zeros(FT, Nz + 1),
                         FT.(range(0, 1; length=Nz + 1))), CPU(); FT)
        air = ntuple(_ -> fill(FT(1000), Nc + 2Hp, Nc + 2Hp, Nz), 6)
        ps = ntuple(_ -> fill(FT(80000), Nc, Nc), 6)
        scratch = ntuple(_ -> fill(FT(99), Nc, Nc, Nz), 6)
        raw = FT[(-1)^(i+j+p+k) * (i+j+p+k)*1e-6
                 for i in 1:Nc, j in 1:Nc, p in 1:6, k in 1:Nz]
        mktempdir() do directory
            path = joinpath(directory, "native.nc")
            NCDataset(path, "c") do ds
                for (name, size) in (("x", Nc), ("y", Nc), ("nf", 6), ("lev", Nz))
                    defDim(ds, name, size)
                end
                defVar(ds, "signed", FT, ("x", "y", "nf", "lev"))[:] = raw
            end
            cfg = Dict{String, Any}("kind"=>"cs_native", "file"=>path,
                                    "variable"=>"signed", "vertical_order"=>"toa_first")
            native = PackingIC._build_cs_initial_mixing_ratio(air, grid, cfg, scratch; surface_pressure=ps)
            @test native == ntuple(p -> raw[:, :, p, :], 6)
            @test all(native[p] !== scratch[p] for p in 1:6)
            @test all(all(==(FT(99)), panel) for panel in scratch)
            original = map(copy, native)
            specs = Dict(:native=>cfg, :uniform=>Dict("kind"=>"uniform", "background"=>-2e-4),
                         :layer=>Dict("kind"=>"pressure_layer", "lowest_layer"=>true))
            state = PackingRunner._initialize_cs_dry_state(grid, air, specs; surface_pressure=ps)
            expected = pack_initial_tracer_mass(grid, air, original; mass_basis=DryBasis())
            @test get_tracer(state, :native) == expected
            uniform = Dict("kind"=>"uniform", "background"=>-2e-4)
            reused = PackingIC._build_cs_initial_mixing_ratio(air, grid, uniform, native; surface_pressure=ps)
            @test reused === native
            @test all(all(==(FT(-2e-4)), panel) for panel in reused)
            @test get_tracer(state, :native) == expected
        end
    end
end

@testset "Moist CS packing requires panel humidity" begin
    FT, Nc, Hp, Nz = Float64, 4, 1, 3
    grid = AtmosGrid(CubedSphereMesh(; FT, Nc, Hp),
                     HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0]), CPU(); FT)
    air_mass = ntuple(_ -> fill(1.2e10, Nc + 2Hp, Nc + 2Hp, Nz), 6)
    vmr_interior = ntuple(_ -> fill(4.11e-4, Nc, Nc, Nz), 6)
    # MoistBasis without qv errors
    @test_throws ArgumentError pack_initial_tracer_mass(grid, air_mass, vmr_interior;
                                                        mass_basis = MoistBasis())
    # MoistBasis with wrong qv type errors
    qv_flat = fill(FT(0.02), Nc + 2 * Hp, Nc + 2 * Hp, Nz)   # not an NTuple{6}
    @test_throws ArgumentError pack_initial_tracer_mass(grid, air_mass, vmr_interior;
                                                        mass_basis = MoistBasis(),
                                                        qv = qv_flat)
end
