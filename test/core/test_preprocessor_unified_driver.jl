#!/usr/bin/env julia
# Plan 41 - focused tests for the unified-preprocessor lifecycle.

using Test

using AtmosTransport
include(joinpath(@__DIR__, "..", "helpers", "preprocessing_evidence.jl"))
using .AtmosTransport.Preprocessing: AbstractMetSettings,
                                       AbstractMetReader,
                                       AbstractWindowContract,
                                       AbstractWindowWorkspace,
                                       AbstractBinaryWriter,
                                       LatLonTargetGeometry,
                                       DryBasis,
                                       NoChain,
                                       ReadyWindow,
                                       PreverifiedWindow,
                                       UnifiedPreprocessorDay,
                                       run_unified_preprocessor_day!

import .AtmosTransport.Preprocessing: windows_per_day,
                                       close_reader!,
                                       ingest_window!,
                                       drain_ready_windows!,
                                       flush_final_windows!,
                                       verify_window!,
                                       update_accumulator!,
                                       summarize_status!,
                                       write_window!,
                                       close_streaming_binary!,
                                       promote_streaming_binary!,
                                       quarantine_streaming_binary!,
                                       driver_after_write_window!

struct FakeSettings <: AbstractMetSettings end

mutable struct FakeReader{FT} <: AbstractMetReader{FT, FakeSettings, NoChain}
    windows :: Int
    closed  :: Bool
end

windows_per_day(reader::FakeReader) = reader.windows
close_reader!(reader::FakeReader) = (reader.closed = true; nothing)

mutable struct FakeWorkspace{FT} <: AbstractWindowWorkspace{LatLonTargetGeometry, FT}
    pending       :: Vector{Any}
    ingested      :: Vector{Int}
    after_written :: Vector{Int}
    flush_final   :: Bool
    preverified   :: Bool
    accumulated   :: Bool
end

FakeWorkspace{FT}(; flush_final::Bool=false, preverified::Bool=false,
                  accumulated::Bool=false) where FT =
    FakeWorkspace{FT}(Any[], Int[], Int[], flush_final, preverified, accumulated)

_fake_ready(::Type{FT}, index::Int) where FT =
    ReadyWindow{LatLonTargetGeometry, FT}(index, (token = index,), payload_only_evidence(LatLonTargetGeometry, FT))

_fake_diag() = (replay = (max_rel_err = 0.0, max_abs_err = 0.0),
                positivity = (ok = true, ratio = 0.0, direction = :none,
                              location = (0, 0, 0)))

_fake_event(ws::FakeWorkspace{FT}, index::Int) where FT =
    ws.preverified ? PreverifiedWindow(_fake_ready(FT, index), _fake_diag();
                                       accumulated = ws.accumulated) :
                     _fake_ready(FT, index)

function ingest_window!(ws::FakeWorkspace{FT}, _reader::FakeReader{FT},
                        win::Int) where FT
    push!(ws.ingested, win)
    push!(ws.pending, _fake_event(ws, win))
    return nothing
end

function drain_ready_windows!(ws::FakeWorkspace)
    ready = copy(ws.pending)
    empty!(ws.pending)
    return ready
end

function flush_final_windows!(ws::FakeWorkspace{FT}, reader::FakeReader{FT},
                              _contract) where FT
    ws.flush_final || return ()
    return (_fake_event(ws, reader.windows + 1),)
end

driver_after_write_window!(ws::FakeWorkspace, _reader::FakeReader,
                           ready::ReadyWindow, _context) =
    (push!(ws.after_written, ready.index); nothing)

mutable struct FakeContract{FT} <: AbstractWindowContract{LatLonTargetGeometry, FT}
    updates         :: Vector{Int}
    summarized      :: Bool
    fail_summary    :: Bool
    verify_calls    :: Int
    quarantine_path :: Any
end

FakeContract{FT}(; fail_summary::Bool=false) where FT =
    FakeContract{FT}(Int[], false, fail_summary, 0, nothing)

function verify_window!(ready::ReadyWindow{LatLonTargetGeometry, FT},
                        contract::FakeContract{FT},
                        win_idx::Int) where FT
    @test ready.index == win_idx
    contract.verify_calls += 1
    return _fake_diag()
end

function update_accumulator!(contract::FakeContract, _positivity, win_idx::Int)
    push!(contract.updates, win_idx)
    return nothing
end

function summarize_status!(contract::FakeContract; quarantine_path = nothing)
    contract.summarized = true
    contract.quarantine_path = quarantine_path
    contract.fail_summary && error("summary failed")
    return nothing
end

mutable struct FakeWriter{FT, Basis} <: AbstractBinaryWriter{LatLonTargetGeometry, FT, Basis}
    path        :: String
    final_path  :: String
    closed      :: Bool
    promoted    :: Bool
    quarantined :: Bool
    written     :: Vector{Int}
    fail_on     :: Int
end

FakeWriter{FT}(; fail_on::Int=0) where FT =
    FakeWriter{FT, DryBasis}("stage.bin", "final.bin", false, false,
                             false, Int[], fail_on)

function write_window!(writer::FakeWriter{FT},
                       ready::ReadyWindow{LatLonTargetGeometry, FT}) where FT
    push!(writer.written, ready.index)
    ready.index == writer.fail_on && error("write failed at $(ready.index)")
    return 1
end

function close_streaming_binary!(writer::FakeWriter)
    writer.closed = true
    return writer.path
end

function promote_streaming_binary!(writer::FakeWriter)
    writer.closed || close_streaming_binary!(writer)
    writer.promoted = true
    return writer.final_path
end

function quarantine_streaming_binary!(writer::FakeWriter)
    writer.closed || close_streaming_binary!(writer)
    writer.quarantined = true
    return writer.path
end

@testset "unified driver writes, summarizes, promotes, and closes reader" begin
    FT = Float64
    reader = FakeReader{FT}(3, false)
    workspace = FakeWorkspace{FT}(flush_final = true)
    contract = FakeContract{FT}()
    writer = FakeWriter{FT}()
    day = UnifiedPreprocessorDay(reader, workspace, contract, writer)

    result = run_unified_preprocessor_day!(day)

    @test result.windows_written == 4
    @test result.last_ready_index == 4
    @test result.out_path == writer.final_path
    @test result.promoted
    @test workspace.ingested == [1, 2, 3]
    @test workspace.after_written == [1, 2, 3, 4]
    @test writer.written == [1, 2, 3, 4]
    @test contract.updates == [1, 2, 3, 4]
    @test contract.verify_calls == 4
    @test contract.summarized
    @test contract.quarantine_path == writer.path
    @test writer.closed
    @test writer.promoted
    @test !writer.quarantined
    @test reader.closed
end

@testset "unified driver accepts preverified ready events" begin
    FT = Float64
    reader = FakeReader{FT}(2, false)
    workspace = FakeWorkspace{FT}(preverified = true)
    contract = FakeContract{FT}()
    writer = FakeWriter{FT}()

    result = run_unified_preprocessor_day!(
        UnifiedPreprocessorDay(reader, workspace, contract, writer))

    @test result.windows_written == 2
    @test writer.written == [1, 2]
    @test contract.updates == [1, 2]
    @test contract.verify_calls == 0
    @test writer.promoted
    @test reader.closed
end

@testset "unified driver can skip already-accumulated ready events" begin
    FT = Float64
    reader = FakeReader{FT}(2, false)
    workspace = FakeWorkspace{FT}(preverified = true, accumulated = true)
    contract = FakeContract{FT}()
    writer = FakeWriter{FT}()

    result = run_unified_preprocessor_day!(
        UnifiedPreprocessorDay(reader, workspace, contract, writer))

    @test result.windows_written == 2
    @test writer.written == [1, 2]
    @test isempty(contract.updates)
    @test contract.verify_calls == 0
    @test writer.promoted
    @test reader.closed
end

@testset "unified driver quarantines when summary fails" begin
    FT = Float64
    reader = FakeReader{FT}(1, false)
    workspace = FakeWorkspace{FT}()
    contract = FakeContract{FT}(fail_summary = true)
    writer = FakeWriter{FT}()

    @test_throws ErrorException run_unified_preprocessor_day!(
        UnifiedPreprocessorDay(reader, workspace, contract, writer))

    @test writer.closed
    @test writer.quarantined
    @test !writer.promoted
    @test contract.summarized
    @test reader.closed
end

@testset "unified driver quarantines when a write fails" begin
    FT = Float64
    reader = FakeReader{FT}(3, false)
    workspace = FakeWorkspace{FT}()
    contract = FakeContract{FT}()
    writer = FakeWriter{FT}(fail_on = 2)

    @test_throws ErrorException run_unified_preprocessor_day!(
        UnifiedPreprocessorDay(reader, workspace, contract, writer))

    @test writer.written == [1, 2]
    @test contract.updates == [1, 2]
    @test !contract.summarized
    @test writer.closed
    @test writer.quarantined
    @test !writer.promoted
    @test reader.closed
end
