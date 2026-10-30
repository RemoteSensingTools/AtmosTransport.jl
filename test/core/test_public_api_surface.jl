#!/usr/bin/env julia

using Test

using AtmosTransport

@testset "curated top-level public API" begin
    exported = Set(names(AtmosTransport))

    # Soft cap on top-level export count. Original threshold was 100; the
    # 2026-05 additions (`NoAdvection`, `CMFMCMatrixConvection`,
    # `AbstractConvection`, `ConvectionForcing`, `apply_convection!`,
    # `AbstractMetDriver`, `lonlat_to_panel_xy`)
    # took the count to 107; the 2026-06 typed runtime-physics specs and
    # config-facing additions took it to 144. Bump to 160 to give breathing
    # room. Next API audit should bring it back down (good candidates: internal
    # accessors that escaped through `using .Submodule` re-exports).
    @test length(exported) < 160
    @test :run_driven_simulation in exported
    @test :validate_config in exported
    @test :inspect_binary in exported
    @test :TransportModel in exported
    @test :open_snapshot in exported

    # Advanced internals stay reachable by qualification but no longer fill
    # `using AtmosTransport` completions.
    @test isdefined(AtmosTransport, :CSTapeSlot)
    @test isdefined(AtmosTransport, :apply_B_half!)
    @test isdefined(AtmosTransport, :set_streaming_steps_per_window_schedule!)
    @test :CSTapeSlot ∉ exported
    @test :apply_B_half! ∉ exported
    @test :set_streaming_steps_per_window_schedule! ∉ exported
end
