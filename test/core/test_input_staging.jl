# Rolling NVMe input stager: rolling/eviction/disk-bound, disabled passthrough,
# copy-failure fallback, and cross-run reuse. CPU-only, no transport binaries.
using Test

using AtmosTransport
using .AtmosTransport.Models: InputStager, staged_path_for!, cleanup_staging!
const _IS = AtmosTransport.Models.InputStaging   # for the internal name helper

# fake NAS files with distinct content/size per day
function _fake_nas(n)
    dir = mktempdir()
    paths = [joinpath(dir, "geos_transport_2021120$(i)_float32.bin") for i in 1:n]
    for (i, p) in enumerate(paths)
        write(p, rand(UInt8, 400_000 + 7 * i))
    end
    return paths
end

@testset "InputStaging — rolling NVMe stager" begin
    paths = _fake_nas(5)

    @testset "enabled: rolling stage, eviction, disk bound, content" begin
        stage = mktempdir()
        mgr = InputStager(paths, Dict("enabled" => true, "dir" => stage,
                                      "lookahead_days" => 2, "keep_behind_days" => 0))
        for idx in 1:5
            sp = staged_path_for!(mgr, idx)
            @test dirname(sp) == stage
            @test endswith(sp, basename(paths[idx]))
            @test isfile(sp) && read(sp) == read(paths[idx])          # bit-identical copy
            @test count(f -> endswith(f, ".bin"), readdir(stage)) <= 3 # ≤ lookahead+1
            for j in 1:idx-1                                            # keep_behind=0 ⇒ evicted
                @test !isfile(joinpath(stage, _IS._staged_basename(paths[j])))
            end
        end
        cleanup_staging!(mgr)
        @test isempty(filter(f -> endswith(f, ".bin") || occursin(".tmp-", f), readdir(stage)))
    end

    @testset "disabled: returns NAS paths, no staging" begin
        stage = mktempdir()
        mgr = InputStager(paths, Dict("enabled" => false))
        for idx in 1:5
            @test staged_path_for!(mgr, idx) == paths[idx]
        end
        @test isempty(readdir(stage))
    end

    @testset "fallback: missing source ⇒ NAS path, no crash" begin
        stage = mktempdir()
        bad = copy(paths); bad[3] = joinpath(dirname(paths[1]), "MISSING.bin")
        mgr = InputStager(bad, Dict("enabled" => true, "dir" => stage, "lookahead_days" => 1))
        @test (@test_logs (:warn,) match_mode = :any staged_path_for!(mgr, 3)) == bad[3]
        cleanup_staging!(mgr)
    end

    @testset "reuse correct + replace stale across runs" begin
        stage = mktempdir()
        cp(paths[1], joinpath(stage, _IS._staged_basename(paths[1])))             # same size, no metadata: re-copy
        write(joinpath(stage, _IS._staged_basename(paths[2])), rand(UInt8, 9))    # wrong size: re-copy
        mgr = InputStager(paths, Dict("enabled" => true, "dir" => stage,
                                      "lookahead_days" => 0, "cleanup_on_exit" => false))
        @test read(staged_path_for!(mgr, 1)) == read(paths[1])
        @test read(staged_path_for!(mgr, 2)) == read(paths[2])
        cleanup_staging!(mgr)
    end

    @testset "config validation" begin
        @test_throws ArgumentError InputStager(paths, Dict("enabled" => true))  # missing dir
    end
end

@testset "InputStaging — source identity and directory ownership" begin
    mktempdir() do dir
        source = joinpath(dir,"source.bin")
        write(source,fill(UInt8(1),4096))
        stage = joinpath(dir,"stage")
        cfg = Dict("enabled"=>true,"dir"=>stage,"lookahead_days"=>0,
                   "cleanup_on_exit"=>false)
        blocked_dir = joinpath(dir,"not-a-directory")
        write(blocked_dir,"occupied")
        unavailable = @test_logs (:warn,r"Input staging directory") InputStager(
            [source],merge(cfg,Dict("dir"=>blocked_dir)))
        @test !unavailable.enabled
        @test staged_path_for!(unavailable,1) == source

        first = InputStager([source],cfg)
        staged = staged_path_for!(first,1)
        @test _IS._can_reuse_staged(source,staged)
        @test isfile(_IS._staged_metadata_path(staged))
        original_inode = stat(staged).inode
        # A simultaneous run must not write into or evict this run's cache.
        second = @test_logs (:warn,r"Input staging directory") InputStager([source],cfg)
        @test !second.enabled
        @test staged_path_for!(second,1) == source
        cleanup_staging!(second)
        @test isfile(staged)
        unrelated_partial = joinpath(stage,"another-producer.part")
        write(unrelated_partial,"owned elsewhere")
        cleanup_staging!(first)
        @test !isfile(joinpath(stage,".atmostransport-staging.pid"))
        @test read(unrelated_partial,String) == "owned elsewhere"
        @test_throws ArgumentError staged_path_for!(first,1)
        @test cleanup_staging!(first) === nothing

        # An unchanged source really is reused, preserving the staged inode.
        reused = InputStager([source],cfg)
        @test staged_path_for!(reused,1) == staged
        @test stat(staged).inode == original_inode
        cleanup_staging!(reused)

        # Replacing the source with the same size must invalidate cached data.
        replacement = joinpath(dir,"replacement.bin")
        write(replacement,fill(UInt8(2),4096))
        mv(replacement,source;force=true)
        @test filesize(source) == filesize(staged)
        @test !_IS._can_reuse_staged(source,staged)
        changed = InputStager([source],cfg)
        @test read(staged_path_for!(changed,1)) == fill(UInt8(2),4096)
        @test _IS._can_reuse_staged(source,staged)
        cleanup_staging!(changed)

        # Corrupt metadata never authorizes reuse, and cleanup removes only
        # files owned by this stager (including their source metadata).
        write(_IS._staged_metadata_path(staged),"invalid [ metadata")
        @test !_IS._can_reuse_staged(source,staged)
        final = InputStager([source],merge(cfg,Dict("cleanup_on_exit"=>true)))
        @test read(staged_path_for!(final,1)) == read(source)
        cleanup_staging!(final)
        @test !isfile(staged)
        @test !isfile(_IS._staged_metadata_path(staged))
        @test read(unrelated_partial,String) == "owned elsewhere"

        repeated = InputStager(fill(source,5),merge(cfg,Dict(
            "cleanup_on_exit"=>true,"lookahead_days"=>2)))
        for idx in 1:5
            path = staged_path_for!(repeated,idx)
            @test isfile(path)
            @test read(path) == read(source)
        end
        cleanup_staging!(repeated)
        @test !isfile(staged)
    end
end
