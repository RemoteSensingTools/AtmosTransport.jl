using Test
if get(ENV, "ATMOSTR_RUN_SNAPSHOT_GPU_TESTS", "0") == "1"
    using CUDA, AtmosTransport, Random
    using AtmosTransport.Output: ObservationGatherBuffers, gather_columns!, gather_field!
    using AtmosTransport.State: CellState, CubedSphereState, DryBasis

    expected = get(ENV, "ATMOSTR_SNAPSHOT_GPU_NAME", "")
    CUDA.functional() || error("ATMOSTR_RUN_SNAPSHOT_GPU_TESTS=1 requires a functional CUDA device")
    isempty(expected) || occursin(expected, CUDA.name(CUDA.device())) ||
        error("ATMOSTR_SNAPSHOT_GPU_NAME=$(expected) does not match $(CUDA.name(CUDA.device()))")
    CUDA.allowscalar(false)

    @testset "observation gather on CUDA matches the host gather" begin
        rng = MersenneTwister(5)
        for FT in (Float32, Float64)
            Nc, Hp, Nz = 6, 3, 4
            Np = Nc + 2Hp
            air = ntuple(p -> FT.(rand(rng, Np, Np, Nz) .+ p), 6)
            co2 = map(m -> m .* FT(400e-6), air)
            ch4 = map(m -> m .* FT(1.9e-6), air)
            host = CubedSphereState(DryBasis, air; co2 = co2, dummy = air, ch4 = ch4, halo_width = Hp)
            device = CubedSphereState(DryBasis, map(CuArray, air); co2 = map(CuArray, co2),
                                      dummy = map(CuArray, air), ch4 = map(CuArray, ch4), halo_width = Hp)
            requests = sort!([(p, rand(rng, 1:Nc), rand(rng, 1:Nc)) for p in 1:6 for _ in 1:rand(rng, 0:5)]; by = first)
            columns = Int32[(Hp + i) + (Hp + j - 1) * Np for (_, i, j) in requests]
            ranges = [let idx = findall(r -> r[1] == p, requests)
                          isempty(idx) ? (1:0) : (first(idx):last(idx))
                      end for p in 1:6]
            slots = [3, 1]
            buf_device = ObservationGatherBuffers(device.air_mass[1], Nz, slots; capacity = 2)
            buf_host = ObservationGatherBuffers(host.air_mass[1], Nz, slots; capacity = 2)
            n = gather_columns!(buf_device, device, columns, ranges)
            @test gather_columns!(buf_host, host, columns, ranges) == n
            @test buf_device.air isa CuArray
            @test buf_device.air_host[:, 1:n] == buf_host.air_host[:, 1:n]
            @test buf_device.tracers_host[:, :, 1:n] == buf_host.tracers_host[:, :, 1:n]
            for (r, (p, i, j)) in enumerate(requests), k in 1:Nz
                @test buf_device.air_host[k, r] == air[p][Hp + i, Hp + j, k]
            end
            # Zero-slot field gather on the device, and a backend mismatch is refused.
            field_device = ObservationGatherBuffers(device.air_mass[1], Nz, Int[])
            @test gather_field!(field_device, device.air_mass, columns, ranges) == n
            @test field_device.air_host[:, 1:n] == buf_host.air_host[:, 1:n]
            @test_throws ArgumentError gather_columns!(buf_host, device, columns, ranges)
            @test_throws ArgumentError gather_columns!(buf_device, host, columns, ranges)

            air_ll = FT.(rand(rng, 8, 5, Nz) .+ 1)
            ll_host = CellState(DryBasis, air_ll; co2 = air_ll .* FT(4e-4), ch4 = air_ll .* FT(2e-6))
            ll_device = CellState(DryBasis, CuArray(air_ll); co2 = CuArray(air_ll .* FT(4e-4)),
                                  ch4 = CuArray(air_ll .* FT(2e-6)))
            cols = Int32[1, 8, 40, 17, 23]
            bd = ObservationGatherBuffers(ll_device.air_mass, Nz, [2, 1])
            bh = ObservationGatherBuffers(ll_host.air_mass, Nz, [2, 1])
            gather_columns!(bd, ll_device, cols)
            gather_columns!(bh, ll_host, cols)
            @test bd.air_host[:, 1:5] == bh.air_host[:, 1:5]
            @test bd.tracers_host[:, :, 1:5] == bh.tracers_host[:, :, 1:5]
        end
        # Growth path on the device, lat-lon and cubed sphere.
        big = CuArray(rand(Float32, 20, 20, 3))
        state = CellState(DryBasis, big; co2 = big .* 1f-6)
        buf = ObservationGatherBuffers(state.air_mass, 3, [1]; capacity = 4)
        @test gather_columns!(buf, state, Int32.(collect(1:400))) == 400
        @test buf.capacity == 512
        @test buf.air_host[2, 400] == Array(big)[20, 20, 2]
        panels = ntuple(p -> CuArray(rand(Float32, 10, 10, 3) .+ p), 6)
        cs = CubedSphereState(DryBasis, panels; co2 = map(a -> a .* 1f-6, panels), halo_width = 2)
        cbuf = ObservationGatherBuffers(panels[1], 3, [1]; capacity = 1)
        cols = Int32[(2 + i) + (2 + j - 1) * 10 for p in 1:6 for i in 1:6 for j in 1:6]
        ranges = [(1 + 36 * (p - 1)):(36 * p) for p in 1:6]
        @test gather_columns!(cbuf, cs, cols, ranges) == 216
        @test cbuf.capacity == 256
        @test cbuf.air_host[1, 37] == Array(panels[2])[3, 3, 1]
    end
else
    @info "Skipping observation gather GPU test (set ATMOSTR_RUN_SNAPSHOT_GPU_TESTS=1 on a CUDA host)"
end
