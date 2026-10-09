#!/usr/bin/env julia
# The source revision recorded in output files and transport binaries:
#   1. a git checkout reports HEAD and whether it has local changes;
#   2. `git archive` writes the commit into src/REVISION (export-subst), which
#      an exported tree reports when it has no `.git`.

using Test
import AtmosTransport

@testset "source revision" begin
    root = pkgdir(AtmosTransport)
    rev = AtmosTransport.source_revision()
    if ispath(joinpath(root, ".git"))
        @test rev.commit == readchomp(`git -C $root rev-parse HEAD`)
        @test rev.dirty in ("clean", "dirty")
        # an archive of HEAD carries HEAD's commit in src/REVISION (once the file is committed)
        committed(path) = !isempty(readchomp(`git -C $root ls-tree HEAD $path`))
        if committed("src/REVISION") && committed(".gitattributes")
            exported = mktempdir() do dir
                run(pipeline(`git -C $root archive HEAD src/REVISION`, `tar -x -C $dir`))
                strip(read(joinpath(dir, "src", "REVISION"), String))
            end
            @test exported == readchomp(`git -C $root rev-parse HEAD`)
        end
    elseif rev.commit == "unknown"
        @test_skip false            # an install whose REVISION was not expanded on export
    else
        @test rev.dirty == "clean"  # exported tree
    end
    # the unexpanded placeholder of a checkout is never reported as a commit
    @test occursin("Format", read(joinpath(root, "src", "REVISION"), String)) || !ispath(joinpath(root, ".git"))
end
