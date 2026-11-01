# Opt in on a ROCm host:
# ATMOSTR_RUN_ROCM_TESTS=1 HIP_VISIBLE_DEVICES=0 julia --threads=8 --project=. test/diagnostic/test_rocm_smoke.jl
using Test
if get(ENV, "ATMOSTR_RUN_ROCM_TESTS", "0") != "1"
    @info "Skipping opt-in ROCm smoke tests (set ATMOSTR_RUN_ROCM_TESTS=1)"
else
    using AMDGPU                  # load the extension before constructing models
    using AtmosTransport
    AMDGPU.versioninfo()
    AMDGPU.functional() || error("ROCm is unavailable; check AMDGPU.versioninfo()")
    AMDGPU.allowscalar(false)
    helpers = joinpath(@__DIR__, "..", "helpers")
    include(joinpath(helpers, "backend_smoke.jl"))
    include(joinpath(helpers, "point_ops.jl"))
    include(joinpath(helpers, "cs_halo_fill.jl"))
    include(joinpath(helpers, "cmfmc_cfl.jl"))
    include(joinpath(helpers, "ledger_carry.jl"))
    @testset "ROCm forward smoke: CPU parity, mass, uniform tracer" begin
        BackendSmoke.check_backend_smoke(ROCArray; backend = "rocm")
        check_point_ops_on_device(ROCArray)
        @test all(check_halo_fill(library_fill!, library_corners!, ROCArray;
                                  cases = ((Float32, 3), (Float64, 4))))
        check_cmfmc_cfl_on_device(ROCArray; float_types = (Float32, Float64))
        check_tm5_ledger_carry(ROCArray; calls = 3)
        check_dkg_ledger_carry(ROCArray; calls = 3)
        check_cmfmc_ledger_carry(ROCArray; calls = 3)
    end
    @testset "ROCm pinned staging and tape round trips" begin
        A = AtmosTransport.Architectures
        T = AtmosTransport.Tape
        panels = ntuple(p -> ROCArray(fill(Float64(p), 4, 4, 3)), 6)
        host = ntuple(_ -> zeros(Float64, 4, 4, 3), 6)
        A.pin_host_memory!(host, panels)
        foreach(copyto!, host, panels)
        A.synchronize_storage!(panels)
        @test all(p -> host[p] == Array(panels[p]), 1:6)
        # The pinned-host arena registers its chunk once; the registration
        # lives until the arena is released.
        arena = T._reserve_tape_pool(AtmosTransport.PinnedHostCSTapeStorage(), [(panels, eltype(first(panels)))])
        try
            slot = T._stage_panels(arena, panels)
            GC.gc()
            @test length(arena.chunks) == 1 && arena.chunks[1].registration !== nothing
            @test all(p -> slot.panels[p] == Array(panels[p]), 1:6)
            @test map(Array, T._tape_panels(slot, T.TracerMassRead())) == map(Array, panels)
            T._rewind_tape_pool!(arena)
            again = T._stage_panels(arena, panels)
            @test length(arena.chunks) == 1 && pointer(again.panels[1]) == pointer(slot.panels[1])
        finally
            T._release_tape_pool!(arena)
        end
        @test isempty(arena.chunks)
        mktempdir() do dir
            storage = T._MmapTapeFile(dir, T._tape_slot_bytes(panels))
            try
                slot = T._stage_panels(storage, panels)
                @test map(Array, T._tape_panels(slot, T.TracerMassRead())) == map(Array, panels)
            finally
                T._release_tape_pool!(storage)
            end
        end
        A.reclaim_backend_pool!(panels)
    end
end
