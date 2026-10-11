#!/usr/bin/env julia
#
# GriddedIO staging of compressed inputs (design 2.4; P0 facts memo Section 4):
# `.nc` passes through, `.nc.gz` is decompressed with `gzip -dc`, a
# single-member `.nc.tar.gz` with `tar -xzOf`; the result is cached by the
# source's path, size and modification time and published atomically.

using Test
using NCDatasets

using AtmosTransport
using AtmosTransport.GriddedIO: stage_input

const GZIP = Sys.which("gzip")
const TAR = Sys.which("tar")

function small_netcdf(path; value = 1.5)
    NCDataset(path, "c") do ds
        defDim(ds, "x", 3)
        defVar(ds, "v", Float64, ("x",))[:] = [value, 2value, 3value]
    end
    return path
end

gzip_copy(src, dst) = (run(pipeline(`$GZIP -c -- $src`; stdout = dst)); dst)
tar_gz(dst, dir, members...) = (run(`$TAR -czf $dst -C $dir $(collect(members))`); dst)

# Files in `dir` other than the given ones (leftover staging files).
others(dir, keep) = filter(f -> !(joinpath(dir, f) in keep), readdir(dir))

if GZIP === nothing || TAR === nothing
    @info "gzip or tar not on the PATH; staging tests skipped"
    @test_skip false
else
@testset "stage_input" begin
    src = mktempdir()
    staging = joinpath(mktempdir(), "staging")       # created on first use
    nc = small_netcdf(joinpath(src, "SDGVM_S3_gpp.nc"))

    @testset "uncompressed files pass through" begin
        @test stage_input(nc; staging_dir = staging) == nc
        @test stage_input(joinpath(src, "missing.nc"); staging_dir = staging) == joinpath(src, "missing.nc")
        @test !isdir(staging)                         # nothing staged
    end

    @testset ".nc.gz" begin
        gz = gzip_copy(nc, joinpath(src, "SDGVM_S3_gpp.nc.gz"))
        staged = stage_input(gz; staging_dir = staging)
        @test dirname(staged) == staging
        @test endswith(staged, "-SDGVM_S3_gpp.nc")
        @test read(staged) == read(nc)                 # bit-identical
        @test NCDataset(ds -> ds["v"][:], staged) == [1.5, 3.0, 4.5]
        # Published like any file: mode 0o666 & ~umask, no staging leftovers.
        mask = ccall(:umask, UInt32, (UInt32,), 0o022); ccall(:umask, UInt32, (UInt32,), mask)
        @test filemode(staged) & 0o777 == 0o666 & ~mask
        @test others(staging, [staged]) == String[]
        # Cached: a second call reuses the file without rewriting it.
        t0 = mtime(staged)
        sleep(0.05)
        @test stage_input(gz; staging_dir = staging) == staged
        @test mtime(staged) == t0
        # A changed source (new content and modification time) gets a new key.
        small_netcdf(joinpath(src, "v2.nc"); value = 7.0)
        sleep(1.1)                                     # a distinct mtime
        gzip_copy(joinpath(src, "v2.nc"), gz)
        restaged = stage_input(gz; staging_dir = staging)
        @test restaged != staged
        @test NCDataset(ds -> ds["v"][:], restaged) == [7.0, 14.0, 21.0]
        @test isfile(staged)                           # old entries are the caller's to remove
    end

    @testset ".nc.tar.gz with one member" begin
        inner = mktempdir()
        small_netcdf(joinpath(inner, "iMAPLE_S3_gpp.nc"); value = 2.0)
        tgz = tar_gz(joinpath(src, "iMAPLE_S3_gpp.nc.tar.gz"), inner, "iMAPLE_S3_gpp.nc")
        staged = stage_input(tgz; staging_dir = staging)
        @test endswith(staged, "-iMAPLE_S3_gpp.nc")
        @test read(staged) == read(joinpath(inner, "iMAPLE_S3_gpp.nc"))
        @test stage_input(tgz; staging_dir = staging) == staged
    end

    @testset "refused inputs leave nothing behind" begin
        inner = mktempdir()
        small_netcdf(joinpath(inner, "a.nc")); small_netcdf(joinpath(inner, "b.nc"))
        two = tar_gz(joinpath(src, "two.nc.tar.gz"), inner, "a.nc", "b.nc")
        before = readdir(staging)
        @test_throws ArgumentError stage_input(two; staging_dir = staging)
        # Not NetCDF after decompression.
        txt = joinpath(src, "notes.txt"); write(txt, "not a netcdf file\n")
        bad = gzip_copy(txt, joinpath(src, "notes.nc.gz"))
        @test_throws ArgumentError stage_input(bad; staging_dir = staging)
        # A corrupt archive: gzip fails.
        corrupt = joinpath(src, "corrupt.nc.gz"); write(corrupt, rand(UInt8, 64))
        @test_throws Exception stage_input(corrupt; staging_dir = staging)
        @test readdir(staging) == before               # no partial or staging files
        @test_throws ArgumentError stage_input(joinpath(src, "absent.nc.gz"); staging_dir = staging)
    end

    @testset "no default staging directory" begin
        @test_throws UndefKeywordError stage_input(joinpath(src, "SDGVM_S3_gpp.nc.gz"))
    end
end
end
