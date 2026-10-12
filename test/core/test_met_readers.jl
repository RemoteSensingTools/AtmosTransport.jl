#!/usr/bin/env julia
# Plan 41 P0a — typed met-reader surface tests.
#
# Exercises the abstract surface (`AbstractMetReader{FT, S, CP}`) and the
# ChainPolicy type-system invariants. The full GEOS-IT bit-exact smoke
# (round-trip a 1-day read against the existing `process_day` path) needs
# the catrine archive on disk; that lives in the `--all` real-data set.
# Core tests below use a mock reader so they run without external data.

using Test
using Dates

import AtmosTransport
const AT  = AtmosTransport
const Pre = AtmosTransport.Preprocessing

# ---------------------------------------------------------------------------
# Mock reader + mock settings — exercise the trait surface without I/O.
# ---------------------------------------------------------------------------

struct _MockMetSettings <: Pre.AbstractMetSettings
    name :: String
end

Pre.windows_per_day(::_MockMetSettings, ::Date) = 4
Pre.has_convection(::_MockMetSettings) = false
Pre.has_surface(::_MockMetSettings)    = false

@testset "Public capability queries include preprocessing settings" begin
    settings = _MockMetSettings("capability probe")
    @test AT.has_surface === Pre.has_surface === AT.MetDrivers.has_surface
    @test AT.has_vdiff_fields === Pre.has_vdiff_fields === AT.MetDrivers.has_vdiff_fields
    @test AT.has_surface(settings) === false
    @test AT.has_vdiff_fields(settings) === false
    for (surface, vdiff, expected_surface) in ((false,false,false), (true,false,true),
                                              (false,true,true), (true,true,true))
        geos = Pre.GEOSITSettings(; root_dir="/tmp/does_not_exist", Nc=24,
                                   include_surface=surface, include_vdiff_fields=vdiff,
                                   include_convection=false)
        @test AT.has_surface(geos) === expected_surface
        @test AT.has_vdiff_fields(geos) === vdiff
    end
end

mutable struct _MockReader{FT, CP, V} <:
                Pre.AbstractMetReader{FT, _MockMetSettings, CP}
    settings :: _MockMetSettings
    date     :: Date
    seed     :: V
    closed   :: Bool
end

Pre.windows_per_day(reader::_MockReader) =
    Pre.windows_per_day(reader.settings, reader.date)
Pre.native_vertical(reader::_MockReader) =
    error("_MockReader has no vertical coordinate")
Pre.window_metadata(reader::_MockReader{FT}) where FT =
    (windows = Pre.windows_per_day(reader), substeps = 1, dt_substep = 3600.0)
Pre.close_reader!(reader::_MockReader) = (reader.closed = true; nothing)

# Chained-mass specialization: declare a typed seed return.
@inline function Pre.end_of_day_seed(
    reader::_MockReader{FT, Pre.ChainedMass{Vector{Float64}}, V},
) where {FT, V}
    seed = reader.seed
    seed === nothing && return nothing
    return seed::Vector{Float64}
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "Plan 41 P0a — typed met-reader surface" begin

    @testset "abstract-type hierarchy" begin
        # ChainPolicy hierarchy
        @test Pre.NoChain <: Pre.AbstractChainPolicy
        @test Pre.ChainedMass{Vector{Float64}} <: Pre.AbstractChainPolicy
        @test Pre.ChainedMass{NTuple{6, Array{Float32, 3}}} <: Pre.AbstractChainPolicy
        # Reader hierarchy must be checked on concrete instantiations —
        # Julia's `<:` returns `false` when comparing bare UnionAll types
        # with different arities, even when every concrete instantiation
        # is a subtype (`AbstractMetReader` has 3 params, the readers
        # have 4–5).
        @test Pre.GEOSNativeReader{Float64, Pre.GEOSITSettings,
                                    Pre.NoChain, Nothing, Nothing} <:
              Pre.AbstractMetReader{Float64, Pre.GEOSITSettings, Pre.NoChain}
        @test Pre.ERA5SpectralReader{Float64, Pre.ERA5SpectralSettings} <:
              Pre.AbstractMetReader{Float64, Pre.ERA5SpectralSettings, Pre.NoChain}
        @test _MockReader{Float64, Pre.NoChain, Nothing} <:
              Pre.AbstractMetReader{Float64, _MockMetSettings, Pre.NoChain}
        # Supertype invariant: each concrete reader's `supertype(...)`
        # is `AbstractMetReader{FT, S, CP}` — the V and H parameters are
        # stripped at the abstract level.
        @test supertype(Pre.GEOSNativeReader{Float64, Pre.GEOSITSettings,
                                              Pre.NoChain, Nothing, Nothing}) ===
              Pre.AbstractMetReader{Float64, Pre.GEOSITSettings, Pre.NoChain}
    end

    @testset "NoChain reader — end_of_day_seed returns nothing (statically)" begin
        reader = _MockReader{Float64, Pre.NoChain, Nothing}(
            _MockMetSettings("nochain"), Date(2021, 12, 1), nothing, false)
        # Default fallback for NoChain in met_readers.jl: always nothing.
        @test Pre.end_of_day_seed(reader) === nothing
        # Type-level: the return type is statically `Nothing` for NoChain.
        # This is the foot-gun (D) closure invariant — `end_of_day_seed`
        # cannot accidentally return a non-nothing value from a NoChain
        # reader path.
        @test @inferred(Pre.end_of_day_seed(reader)) === nothing
    end

    @testset "ChainedMass reader — end_of_day_seed returns typed seed" begin
        seed = [1.0, 2.0, 3.0]
        reader = _MockReader{Float64, Pre.ChainedMass{Vector{Float64}},
                              Union{Nothing, Vector{Float64}}}(
            _MockMetSettings("chain"), Date(2021, 12, 1), seed, false)
        out = Pre.end_of_day_seed(reader)
        @test out isa Vector{Float64}
        @test out == [1.0, 2.0, 3.0]
        # Pre-seed: returns nothing (until end-of-day fills the slot).
        reader_empty = _MockReader{Float64, Pre.ChainedMass{Vector{Float64}},
                                    Union{Nothing, Vector{Float64}}}(
            _MockMetSettings("chain"), Date(2021, 12, 1), nothing, false)
        @test Pre.end_of_day_seed(reader_empty) === nothing
    end

    @testset "windows_per_day and lifecycle" begin
        reader = _MockReader{Float64, Pre.NoChain, Nothing}(
            _MockMetSettings("noch"), Date(2021, 12, 1), nothing, false)
        @test Pre.windows_per_day(reader) == 4
        @test reader.closed == false
        Pre.close_reader!(reader)
        @test reader.closed == true
        # close_reader! is idempotent.
        Pre.close_reader!(reader)
        @test reader.closed == true
    end

    @testset "window_metadata returns NamedTuple with required keys" begin
        reader = _MockReader{Float64, Pre.NoChain, Nothing}(
            _MockMetSettings("meta"), Date(2021, 12, 1), nothing, false)
        meta = Pre.window_metadata(reader)
        @test meta isa NamedTuple
        @test haskey(meta, :windows)
        @test haskey(meta, :substeps)
        @test haskey(meta, :dt_substep)
        @test meta.windows == 4
    end

    # -----------------------------------------------------------------------
    # ERA5SpectralReader — typed nominal exists; read_window! remains explicit.
    # -----------------------------------------------------------------------

    @testset "ERA5SpectralReader nominal" begin
        nt = (mass_basis = :dry, T_target = 159, level_range = 1:137)
        settings = Pre.ERA5SpectralSettings(nt)
        # Property forwarding to the wrapped NamedTuple.
        @test settings.mass_basis === :dry
        @test settings.T_target == 159
        @test :mass_basis in propertynames(settings)

        reader = Pre.open_reader(settings, Date(2021, 12, 1), Float64;
                                  seed = nothing, chain_mass = false)
        @test reader isa Pre.ERA5SpectralReader{Float64, Pre.ERA5SpectralSettings}
        @test reader isa Pre.AbstractMetReader{Float64, Pre.ERA5SpectralSettings, Pre.NoChain}
        @test Pre.windows_per_day(reader) == 24
        @test Pre.end_of_day_seed(reader) === nothing

        # read_window! remains unsupported with a clear error.
        err = try
            Pre.read_window!(nothing, reader, 1)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("not implemented yet", err.msg)

        # ChainPolicy is fixed at NoChain for the spectral path; passing a
        # non-nothing seed is a static API violation.
        @test_throws ArgumentError Pre.open_reader(
            settings, Date(2021, 12, 1), Float64; seed = [1.0])
        @test_throws ArgumentError Pre.open_reader(
            settings, Date(2021, 12, 1), Float64; chain_mass = true)

        Pre.close_reader!(reader)
        @test reader.closed == true
    end

    # -----------------------------------------------------------------------
    # GEOSNativeReader — typed-construction tests that do NOT open a file.
    # We construct the reader directly (skipping open_reader → open_day) so
    # we exercise the static type-parameter mechanics without needing the
    # GEOS NetCDF archive. The real-data smoke that actually opens a day
    # and round-trips read_window! against the existing path lives in the
    # `--all` real-data test set.
    # -----------------------------------------------------------------------

    @testset "GEOSNativeReader type parameters and end_of_day_seed dispatch" begin
        settings = Pre.GEOSITSettings(;
            root_dir = "/tmp/does_not_exist",
            Nc = 24,
            mass_flux_dt = 450.0,
            level_orientation = :bottom_up,
            include_surface = false,
            include_convection = false,
            physics_dir = "",
            physics_layout = :auto,
            coefficients_file = "config/geos_L72_coefficients.toml",
        )
        date = Date(2021, 12, 1)
        # NoChain construction (chain_mass = false). Synthetic
        # construction uses `H = Nothing` since we pass `nothing` as
        # the handles — Julia-style review round-1 promoted the field
        # type to a fifth `H` type parameter for static dispatch.
        reader_nc = Pre.GEOSNativeReader{Float64, typeof(settings),
                                          Pre.NoChain, Nothing, Nothing}(
            settings, nothing, date, nothing, Ref{Nothing}(nothing))
        @test reader_nc isa Pre.AbstractMetReader{Float64, typeof(settings), Pre.NoChain}
        @test Pre.end_of_day_seed(reader_nc) === nothing
        # set_end_of_day_seed! is a no-op for NoChain.
        @test Pre.set_end_of_day_seed!(reader_nc, [1.0, 2.0]) === nothing

        # ChainedMass construction
        SeedT = NTuple{6, Array{Float64, 3}}
        FieldT = Union{Nothing, SeedT}
        reader_ch = Pre.GEOSNativeReader{Float64, typeof(settings),
                                          Pre.ChainedMass{SeedT}, FieldT, Nothing}(
            settings, nothing, date, nothing, Ref{FieldT}(nothing))
        @test reader_ch isa Pre.AbstractMetReader{Float64, typeof(settings),
                                                   Pre.ChainedMass{SeedT}}
        # Before set_end_of_day_seed!: seed is nothing, end_of_day_seed
        # returns nothing.
        @test Pre.end_of_day_seed(reader_ch) === nothing
        # After: seed value flows through.
        Nc = 24
        Nz = 3
        seed_val = ntuple(_ -> zeros(Float64, Nc, Nc, Nz), 6)
        seed_val[1][1, 1, 1] = 42.0
        Pre.set_end_of_day_seed!(reader_ch, seed_val)
        carry = Pre.end_of_day_seed(reader_ch)
        @test carry isa SeedT
        @test carry[1][1, 1, 1] == 42.0
    end

    @testset "GEOSNativeReader rejects mismatched settings type at open" begin
        # `open_reader(::AbstractGEOSSettings, ...)` dispatches on the
        # settings type; an unrelated AbstractMetSettings has no
        # `open_reader` method (until P2 lands one). Sanity-check that
        # the dispatch surface is the one we expect.
        bogus = _MockMetSettings("bogus")
        @test_throws MethodError Pre.open_reader(bogus, Date(2021, 12, 1),
                                                  Float64; chain_mass = false)
    end

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
        @test Pre._day_threading_enabled(era5_n320, 2, false, false, NaN) == false
    end

    @testset "zero-tendency endpoint only at configured last day" begin
        first_day = Date(2021, 12, 1)
        last_day = first_day + Day(1)
        @test Pre._require_configured_terminal_day(
            last_day, last_day, "test source", "next-day endpoint") === nothing
        @test_throws ErrorException Pre._require_configured_terminal_day(
            first_day, last_day, "test source", "next-day endpoint")
        @test_throws ErrorException Pre._require_configured_terminal_day(
            last_day, nothing, "test source", "next-day endpoint")
    end
end
