using Test
using Dates
import AtmosTransport
const Pre = AtmosTransport.Preprocessing
struct _MockMetSettings <: Pre.AbstractMetSettings
    name::String
end

@testset "Reader scheduling and archive boundaries" begin
    @testset "supports_day_threading trait" begin
        # Default is conservative — unknown sources serialize. ERA5 GRIB
        # N320 keeps the conservative serial policy because its inner kernels
        # use static schedules; GEOS also stays serial because pressure-flux
        # endpoint chaining couples consecutive days.
        @test Pre.supports_day_threading(_MockMetSettings("default")) == false
        era5_n320 = Pre.ERA5N320Settings(root_dir = "/tmp/nowhere")
        @test Pre.supports_day_threading(era5_n320) == false

        # Exercise the exact outer-driver decision for a two-day request. Run
        # this file with `--threads=2` so the thread-count side of the policy is
        # live; N320 must still decline the outer day loop.
        @test Pre.day_threading_enabled(era5_n320, 2, false, false, NaN) == false
    end

    @testset "zero-tendency endpoint only at configured last day" begin
        first_day = Date(2021, 12, 1)
        last_day = first_day + Day(1)
        @test Pre.require_configured_terminal_day(
            last_day, last_day, "test source", "next-day endpoint") === nothing
        @test_throws ErrorException Pre.require_configured_terminal_day(
            first_day, last_day, "test source", "next-day endpoint")
        @test_throws ErrorException Pre.require_configured_terminal_day(
            last_day, nothing, "test source", "next-day endpoint")
    end
end
