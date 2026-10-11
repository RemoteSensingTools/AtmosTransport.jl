#!/usr/bin/env julia
#
# AtomicFiles: a file is published only when complete, from a sibling staging
# file created with O_EXCL and mode 0o666 & ~umask, by one replacing rename;
# failures leave neither the target nor the staging file, and a failed rename
# keeps the previous target. The transport-binary writers and the spectral day
# cache publish through it. Staging names `<target>.tmp-<pid>-<n>` and file
# modes are POSIX-only: on Windows the staging file comes from `mktemp`.

using Test
using Random

using AtmosTransport
const AT = AtmosTransport
const AF = AT.AtomicFiles
const MD = AT.MetDrivers
const PP = AT.Preprocessing

using .AtmosTransport.Architectures: CPU
using .AtmosTransport.Grids: AtmosGrid, HybridSigmaPressure, ncells, nfaces

"""Run `f()` with the process umask set to `mask`, restoring the previous one."""
function with_umask(f, mask)
    old = ccall(:umask, Cuint, (Cuint,), mask)
    try
        return f()
    finally
        ccall(:umask, Cuint, (Cuint,), old)
    end
end

mode_of(path) = filemode(path) & 0o777

"""Files in `dir` that look like staging files of any target."""
staging_leftovers(dir) = filter(name -> occursin(".tmp-", name), readdir(dir))

function rg_fixture()
    FT = Float32
    mesh = AT.ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT)
    vc = HybridSigmaPressure(FT[0, 1000], FT[0, 1])
    grid = AtmosGrid(mesh, vc, CPU(); FT)
    window = (m = ones(FT, ncells(mesh), 1),
              hflux = zeros(FT, nfaces(mesh), 1),
              cm = zeros(FT, ncells(mesh), 2),
              ps = fill(FT(90_000), ncells(mesh)))
    return grid, window
end

const RG_KWARGS = (FT = Float32, header_bytes = 4096, steps_per_window = 1,
                   source_flux_sampling = :window_start_endpoint,
                   humidity_sampling = :none, delta_semantics = :none, mass_basis = :dry)

@testset "atomic_publish writes the target through a sibling staging file" begin
    mktempdir() do dir
        path = joinpath(dir, "field.bin")
        seen = Ref("")
        result = AF.atomic_publish(path) do tmp
            seen[] = tmp
            @test isfile(tmp)
            @test filesize(tmp) == 0
            @test dirname(tmp) == dirname(abspath(path))
            Sys.iswindows() || @test startswith(basename(tmp), "field.bin.tmp-$(getpid())-")
            @test !ispath(path)                       # nothing published yet
            open(io -> write(io, "first"), tmp, "w")
        end
        @test result == path
        @test read(path, String) == "first"
        @test !ispath(seen[])
        @test readdir(dir) == ["field.bin"]

        AF.atomic_publish(tmp -> open(io -> write(io, "second"), tmp, "w"), path)
        @test read(path, String) == "second"          # previous file replaced
        @test readdir(dir) == ["field.bin"]
    end
end

@testset "a failure mid-write leaves neither target nor staging file" begin
    mktempdir() do dir
        path = joinpath(dir, "new.bin")
        @test_throws DomainError AF.atomic_publish(path) do tmp
            open(tmp, "w") do io
                write(io, "partial payload")
                flush(io)
                throw(DomainError(-1, "simulated failure after a partial write"))
            end
        end
        @test !ispath(path)
        @test isempty(readdir(dir))

        existing = joinpath(dir, "existing.bin")
        write(existing, "previous valid file")
        @test_throws ErrorException AF.atomic_publish(existing) do tmp
            write(tmp, "half of the new file")
            error("simulated failure")
        end
        @test read(existing, String) == "previous valid file"
        @test readdir(dir) == ["existing.bin"]

        # The staged-stream API: abandon removes the staging file only.
        tmp, io = AF.open_atomic_staging(existing)
        write(io, "unfinished")
        close(io)
        @test AF.abandon_atomic_staging(tmp) === nothing
        @test !ispath(tmp)
        @test AF.abandon_atomic_staging(tmp) === nothing      # already gone: no error
        @test read(existing, String) == "previous valid file"
        @test readdir(dir) == ["existing.bin"]
    end
end

@testset "commit replaces the target and cleans up when it cannot publish" begin
    mktempdir() do dir
        path = joinpath(dir, "staged.bin")
        write(path, "old")
        tmp, io = AF.open_atomic_staging(path)
        write(io, "new")
        close(io)
        @test AF.commit_atomic_staging(tmp, path) == path
        @test read(path, String) == "new"
        @test !ispath(tmp)

        # A directory that appeared at the target is never replaced.
        target = joinpath(dir, "became_a_directory")
        tmp, io = AF.open_atomic_staging(target)
        close(io)
        mkdir(target)
        write(joinpath(target, "keep.txt"), "kept")
        @test_throws ArgumentError AF.commit_atomic_staging(tmp, target)
        @test !ispath(tmp)
        @test read(joinpath(target, "keep.txt"), String) == "kept"
    end
end

@testset "commit is one rename; a failed rename keeps the previous target" begin
    mktempdir() do dir
        path = joinpath(dir, "swap.bin")
        write(path, "old content")
        tmp, io = AF.open_atomic_staging(path)
        write(io, "new content")
        close(io)
        @test read(path, String) == "old content"     # readable until the swap
        if Sys.iswindows()
            AF.commit_atomic_staging(tmp, path)
        else
            reader = open(path, "r")                  # a reader of the previous file
            staged_inode = stat(tmp).inode
            AF.commit_atomic_staging(tmp, path)
            @test stat(path).inode == staged_inode     # the staging file itself: renamed, not copied
            @test read(reader, String) == "old content"
            close(reader)
        end
        @test read(path, String) == "new content"
        @test readdir(dir) == ["swap.bin"]

        # The rename fails (ENOENT: the staging file is gone). `mv(…; force = true)`
        # on Julia 1.10 deleted the target before renaming.
        tmp, io = AF.open_atomic_staging(path)
        close(io)
        rm(tmp)
        err = try
            AF.commit_atomic_staging(tmp, path)
        catch e
            e
        end
        @test err isa Base.IOError
        @test err.code == Base.UV_ENOENT
        @test read(path, String) == "new content"
        @test readdir(dir) == ["swap.bin"]

        # The rename fails with the staging entry present (ENOTDIR: rename(2)
        # cannot move a directory onto a file). `mv(…; force = true)` deleted the
        # target and renamed again on every Julia version. Not on Windows, where
        # MoveFileEx may move a directory onto a file.
        if !Sys.iswindows()
            tmp, io = AF.open_atomic_staging(path)
            close(io)
            rm(tmp)
            mkdir(tmp)
            err = try
                AF.commit_atomic_staging(tmp, path)
            catch e
                e
            end
            @test err isa Base.IOError
            @test err.code == Base.UV_ENOTDIR
            @test read(path, String) == "new content"
            @test !ispath(tmp)                        # staging entry removed
            @test readdir(dir) == ["swap.bin"]
        end
    end
end

@testset "bad targets are refused before anything is created" begin
    mktempdir() do dir
        @test_throws ArgumentError AF.open_atomic_staging(joinpath(dir, "missing", "x.bin"))
        @test_throws ArgumentError AF.atomic_publish(tmp -> write(tmp, "x"),
                                                     joinpath(dir, "missing", "x.bin"))
        sub = joinpath(dir, "subdir")
        mkdir(sub)
        write(joinpath(sub, "keep.txt"), "kept")
        @test_throws ArgumentError AF.atomic_publish(tmp -> write(tmp, "x"), sub)
        @test read(joinpath(sub, "keep.txt"), String) == "kept"
        @test readdir(dir) == ["subdir"]
    end
end

if !Sys.iswindows()
    @testset "published mode is 0o666 & ~umask" begin
        mktempdir() do dir
            for mask in (0o022, 0o027, 0o002, 0o077)
                expected = 0o666 & ~UInt16(mask)    # octal literals are unsigned; widen first
                path = joinpath(dir, "mode_$(string(mask; base = 8)).bin")
                with_umask(mask) do
                    AF.atomic_publish(path) do tmp
                        @test mode_of(tmp) == expected          # staging file already has it
                        open(io -> write(io, "x"), tmp, "w")
                    end
                end
                @test mode_of(path) == expected

                staged = joinpath(dir, "staged_$(string(mask; base = 8)).bin")
                with_umask(mask) do
                    tmp, io = AF.open_atomic_staging(staged)
                    @test mode_of(tmp) == expected
                    write(io, "x")
                    close(io)
                    AF.commit_atomic_staging(tmp, staged)
                end
                @test mode_of(staged) == expected
            end
            @test isempty(staging_leftovers(dir))
        end
    end

    @testset "staging files are created exclusively and never clobber" begin
        mktempdir() do dir
            path = joinpath(dir, "target.bin")
            staged = [AF.open_atomic_staging(path) for _ in 1:200]
            names = first.(staged)
            @test allunique(names)
            foreach(close ∘ last, staged)
            foreach(AF.abandon_atomic_staging, names)
            @test isempty(readdir(dir))

            # Occupy the first name the generator will draw: O_EXCL refuses it,
            # a new name is drawn, and the occupant is left untouched.
            seed = 0x2545f491
            Libc.srand(seed)
            occupied = string(abspath(path), ".tmp-", getpid(), "-", Libc.rand(UInt32))
            write(occupied, "someone else's staging file")
            Libc.srand(seed)
            tmp, io = AF.open_atomic_staging(path)
            Libc.srand()
            close(io)
            @test tmp != occupied
            @test read(occupied, String) == "someone else's staging file"
            AF.abandon_atomic_staging(tmp)
            rm(occupied)
        end
    end
end

@testset "staging names do not consume the task-local Random stream" begin
    mktempdir() do dir
        Random.seed!(20261010)
        reference = rand(4)
        Random.seed!(20261010)
        AF.atomic_publish(tmp -> write(tmp, "x"), joinpath(dir, "rng.bin"))
        @test rand(4) == reference
    end
end

@testset "transport-binary writers publish through AtomicFiles" begin
    mktempdir() do dir
        grid, window = rg_fixture()
        streamed = joinpath(dir, "streamed.bin")
        eager = joinpath(dir, "eager.bin")
        write(streamed, "previous stream")

        function run_writers()
            writer = MD.open_streaming_transport_binary(streamed, grid, 2, window; RG_KWARGS...)
            @test dirname(writer.staging_path) == dir
            Sys.iswindows() ||
                @test startswith(basename(writer.staging_path), "streamed.bin.tmp-$(getpid())-")
            Sys.iswindows() || @test mode_of(writer.staging_path) == 0o640
            MD.write_streaming_window!(writer, window)
            @test read(streamed, String) == "previous stream"    # not published mid-stream
            MD.write_streaming_window!(writer, window)
            @test MD.close_streaming_transport_binary!(writer) == streamed
            @test !ispath(writer.staging_path)
            MD.write_transport_binary(eager, grid, [window, window]; RG_KWARGS...)
        end
        Sys.iswindows() ? run_writers() : with_umask(run_writers, 0o027)

        # The streaming writer and the eager writer produce the same file.
        @test read(streamed) == read(eager)
        if !Sys.iswindows()
            @test mode_of(streamed) == 0o640
            @test mode_of(eager) == 0o640
        end
        @test sort(readdir(dir)) == ["eager.bin", "streamed.bin"]

        # The cubed-sphere streaming writer uses the same staging file.
        cs_path = joinpath(dir, "cs.bin")
        vc = HybridSigmaPressure(Float32[0, 1000], Float32[0, 1])
        cs_writer = MD.open_streaming_cs_transport_binary(
            cs_path, 2, 6, 1, 1, vc; planet_radius = AT.Parameters.EARTH_RADIUS,
            FT = Float32, header_bytes = 4096, steps_per_window = 1, mass_basis = :dry)
        @test dirname(cs_writer.staging_path) == dir
        Sys.iswindows() ||
            @test startswith(basename(cs_writer.staging_path), "cs.bin.tmp-$(getpid())-")
        @test_throws ArgumentError MD.close_streaming_transport_binary!(cs_writer)   # no windows
        @test !ispath(cs_path)
        @test !ispath(cs_writer.staging_path)
        @test sort(readdir(dir)) == ["eager.bin", "streamed.bin"]     # no staging file left
    end
end

@testset "the spectral day cache is published atomically" begin
    mktempdir() do dir
        hours = collect(0:23)                 # the cache validates a full ERA5 day
        spec = (hours = hours,
                lnsp_all = Dict(h => ComplexF64[1 0; 2 3] for h in hours),
                vo_by_hour = Dict(h => zeros(ComplexF64, 2, 2, 137) for h in hours),
                d_by_hour = Dict(h => zeros(ComplexF64, 2, 2, 137) for h in hours),
                T = 1, n_times = length(hours))
        path = joinpath(dir, "cache", "spectral.jld2")
        write_cache() = PP._write_spectral_day_cache(path, spec)
        result = Sys.iswindows() ? write_cache() : with_umask(write_cache, 0o027)
        @test result == path
        Sys.iswindows() || @test mode_of(path) == 0o640
        @test PP._load_spectral_day_cache(path).hours == hours
        @test readdir(dirname(path)) == ["spectral.jld2"]

        # A spec that fails mid-write keeps the previous cache and leaves no staging file.
        broken = (; spec.hours, spec.lnsp_all)
        @test_throws Exception PP._write_spectral_day_cache(path, broken)
        @test PP._load_spectral_day_cache(path).hours == hours
        @test readdir(dirname(path)) == ["spectral.jld2"]
    end
end
