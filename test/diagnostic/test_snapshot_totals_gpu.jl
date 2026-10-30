using Test
if get(ENV, "ATMOSTR_RUN_SNAPSHOT_GPU_TESTS", "0") == "1"
    using CUDA, AtmosTransport
    using AtmosTransport.Output
    const O = AtmosTransport.Output
    expected = get(ENV, "ATMOSTR_SNAPSHOT_GPU_NAME", "A100")
    isempty(expected) && error("Select the authorized GPU by name")
    CUDA.functional() && occursin(expected, CUDA.name(CUDA.device())) ||
        error("This test requires the selected $expected GPU")
    CUDA.allowscalar(false)

    @testset "Column reductions retain signed residuals with and without device Float64" begin
        for shape in ((7, 33), (4, 3, 33)), levels in ((1, 2, 3), (15, 16, 17), (16, 17, 33))
            values = zeros(Float32, shape)
            axis = ndims(values)
            selectdim(values, axis, levels[1]) .= 2f0^24
            selectdim(values, axis, levels[2]) .= 1f0
            selectdim(values, axis, levels[3]) .= -2f0^24
            device = CuArray(values)
            # Exercise the host-slab policy used by Metal on real device arrays.
            @test O._backend_column_sum(device, Float32) == O._column_sum(values)
            @test O._backend_column_sum(device) == O._column_sum(values)
        end
        parent_values = zeros(Float32, 10, 10, 33)
        parent_values[4:7, 4:7, 15] .= 2f0^24
        parent_values[4:7, 4:7, 16] .= 1f0
        parent_values[4:7, 4:7, 17] .= -2f0^24
        interior = view(CuArray(parent_values), 4:7, 4:7, :)
        @test O._backend_column_sum(interior, Float32) == ones(4, 4)
        @test O._backend_column_sum(interior) == ones(4, 4)
    end

    @testset "Compensated device totals retain signed residuals" begin
        for FT in (Float32,Float64), n in (3,257,2049,1048577)
            values = zeros(FT,n)
            lanes = min(4096,cld(n,256))
            for indices in ((1,2,3),(1,1+lanes,1+2lanes))
                maximum(indices) <= n || continue
                fill!(values,0)
                values[indices[1]]=FT(1e30)
                values[indices[2]]=FT(1)
                values[indices[3]]=FT(-1e30)
                exact = Float64(sum(BigFloat.(values)))
                device = CuArray(values)
                @test O._backend_tracer_total(device) == exact == 1.0
                @test O._backend_tracer_total((device,device,device,device,device,device)) == 6exact
            end
        end
        @test_throws ArgumentError O._backend_tracer_total(CuArray([Inf,1.0]))
    end

    @testset "Selected GPU capture preserves signed totals" begin
        for FT in (Float32,Float64), topology in (:ll,:rg,:cs)
            Nz=4
            mesh = topology === :ll ? LatLonMesh(;Nx=4,Ny=2,FT) :
                   topology === :rg ? ReducedGaussianMesh(FT[-45,45],[4,8];FT) :
                   CubedSphereMesh(;Nc=4,Hp=3,FT)
            grid = AtmosGrid(mesh,HybridSigmaPressure(zeros(FT,Nz+1),FT.(0:1/Nz:1)),CPU();FT)
            shape = topology === :ll ? (4,2,Nz) : topology === :rg ? (12,Nz) : (10,10,Nz)
            air = ones(FT,shape)
            rm = zeros(FT,shape)
            if topology === :cs
                rm[4,4,1]=FT(1e30);rm[5,4,1]=1;rm[6,4,1]=FT(-1e30)
                air = ntuple(_->CuArray(air),6);rm=ntuple(_->CuArray(rm),6)
                state = CubedSphereState(DryBasis,air; signed=rm,halo_width=3)
            else
                rm[1]=FT(1e30);rm[2]=1;rm[3]=FT(-1e30)
                state = CellState(DryBasis,CuArray(air);signed=CuArray(rm))
            end
            model=(;state,grid)
            halo_width=topology===:cs ? 3 : 0
            full=capture_snapshot(model;halo_width)
            for mode in ("none","selected","full")
                fields=output_field_spec(Dict("layers"=>mode,"levels"=>[1,3]))
                frame=capture_snapshot(model;halo_width,fields)
                @test frame.tracer_total_mass == full.tracer_total_mass
                @test frame.tracer_total_mass[:signed] == (topology===:cs ? 6.0 : 1.0)
            end
        end
    end
else
    @testset "GPU snapshot reductions (opt-in)" begin
        @test_skip "Set ATMOSTR_RUN_SNAPSHOT_GPU_TESTS=1 and select the authorized GPU"
    end
end
