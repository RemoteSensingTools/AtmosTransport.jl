# Refilling a host window in place (`load_transport_window!`, the GPU staging
# path) must give exactly the window `load_transport_window` builds, halos included.
using Test, AtmosTransport, Mmap
include(joinpath(@__DIR__, "..", "fixtures", "cs_multifile.jl"))
using .CSDriverHandoffFixtures
const StagingMD = AtmosTransport.MetDrivers
const Readers = StagingMD.BinaryReaders

# Field-by-field equality of two windows (arrays by value, `nothing` by identity).
windows_equal(a::AbstractArray, b::AbstractArray) = a == b
windows_equal(a::Tuple, b::Tuple) = length(a) == length(b) && all(map(windows_equal, a, b))
windows_equal(a::NamedTuple, b::NamedTuple) = keys(a) == keys(b) && all(map(windows_equal, values(a), values(b)))
windows_equal(::Nothing, ::Nothing) = true
windows_equal(a, b) = typeof(a) == typeof(b) &&
    all(f -> windows_equal(getfield(a, f), getfield(b, f)), fieldnames(typeof(a)))

@testset "In-place cubed-sphere window load equals a fresh load" begin
    mktempdir() do dir
        for FT in (Float32, Float64)
            path = joinpath(dir, "staging_$(FT).bin")
            cs_handoff_fixture(path, [1.0, 3.0, 8.0]; FT)
            driver = TransportBinaryDriver(path; FT, arch = CPU(), Hp = 3)
            staging = load_transport_window(driver, 1)
            for win in (2, 3, 1)
                fresh = load_transport_window(driver, win)
                @test StagingMD.load_transport_window!(staging, driver, win) === staging
                @test windows_equal(staging, fresh)
            end
            # The halos of padded fields stay zero (they are never written).
            Hp = 3
            m = staging.air_mass[1]
            @test all(iszero, m[1:Hp, :, :]) && all(iszero, m[end-Hp+1:end, :, :])
            close(driver)
        end
    end
end

@testset "In-place load covers every optional cubed-sphere section" begin
    Nc, Nz, np, nwin, Hp = 3, 4, 6, 3, 2
    vc = HybridSigmaPressure([0.0, 20, 100, 400, 1000], [0.0, 0.05, 0.3, 0.7, 1])
    panels(w, dims...) = ntuple(p -> w .+ p / 10 .+ rand(dims...) ./ 100, np)
    # (on-disk, loaded) precision; the loader converts Float32 payloads to Float64.
    @testset "disk $(disk_FT), load $(FT)" for (disk_FT, FT) in
            ((Float64, Float64), (Float32, Float64), (Float32, Float32))
    mktempdir() do dir
        path = joinpath(dir, "all_sections.bin")
        writer = StagingMD.open_streaming_cs_transport_binary(
            path, Nc, np, Nz, nwin, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
            FT = disk_FT, mass_basis = :dry, include_flux_delta = true, include_cmfmc = true,
            surface_pressure_basis = TotalSurfacePressure(),
            include_dtrain = true, include_surface = true, include_tm5conv = true,
            include_gchp_vdiff = true, include_precomputed_dkg = true,
            include_pbl_eflux = true, include_cmfmc_cloud_base = true)
        try
            for w in 1:nwin
                window = (m = panels(1e6 * w, Nc, Nc, Nz),
                          am = panels(w, Nc + 1, Nc, Nz), bm = panels(w, Nc, Nc + 1, Nz),
                          cm = panels(w, Nc, Nc, Nz + 1), ps = panels(9e4 + w, Nc, Nc),
                          dm = panels(w, Nc, Nc, Nz),
                          cmfmc = panels(w, Nc, Nc, Nz + 1), dtrain = panels(w, Nc, Nc, Nz),
                          cmfmc_cloud_base = ntuple(_ -> fill(Float64(Nz - mod(w, 2)), Nc, Nc), np),
                          surface = (pblh = panels(500w, Nc, Nc), ustar = panels(w, Nc, Nc),
                                     hflux = panels(20w, Nc, Nc), t2m = panels(280 + w, Nc, Nc),
                                     eflux = panels(50w, Nc, Nc)),
                          tm5_fields = (entu = panels(w, Nc, Nc, Nz), detu = panels(w, Nc, Nc, Nz),
                                        entd = panels(w, Nc, Nc, Nz), detd = panels(w, Nc, Nc, Nz)),
                          vdiff = (u = panels(w, Nc, Nc, Nz), v = panels(w, Nc, Nc, Nz),
                                   t = panels(250 + w, Nc, Nc, Nz), qv = panels(w / 1000, Nc, Nc, Nz)),
                          dkg = panels(w, Nc, Nc, Nz))
                StagingMD.write_streaming_cs_window!(writer, window, Nc, np)
            end
        finally
            StagingMD.close_streaming_transport_binary!(writer)
        end
        driver = TransportBinaryDriver(path; FT, arch = CPU(), Hp)
        staging = load_transport_window(driver, 1)
        # Raw staging (the device path, here with host arrays): one read of the
        # window payload, unpacked into a window whose halos hold stale values.
        raw = StagingMD.raw_window_staging(driver, staging.air_mass)
        # Another on-disk element type converts on the host instead.
        @test (raw === nothing) == (disk_FT !== FT)
        if raw !== nothing
            h = driver.reader.header
            @test eltype(raw.host) === disk_FT && length(raw.host) == h.elems_per_window
            @test length(raw.device) == maximum(s -> StagingMD.section_elements(h, s), h.payload_sections)
            unpacked = deepcopy(staging)
            foreach(a -> fill!(a, NaN), (unpacked.air_mass..., unpacked.fluxes.am..., unpacked.deltas.dm...,
                                         unpacked.surface.pblh..., unpacked.convection.cmfmc...))
            for win in (3, 1, 2)
                StagingMD.read_window_payload!(raw, driver, win)
                @test StagingMD.unpack_window_payload!(unpacked, driver, raw) === unpacked
                @test windows_equal(unpacked, load_transport_window(driver, win))
            end
            @test_throws BoundsError StagingMD.read_window_payload!(raw, driver, nwin + 1)
            # Unified-memory staging (Metal): the payload buffer is the device
            # buffer, unpacked from each section's window offset with no upload
            # (here one host vector stands for both views).
            @test AtmosTransport.Architectures.host_memory_model(staging.air_mass) ===
                  AtmosTransport.Architectures.DiscreteHostMemory()
            @test_throws ArgumentError AtmosTransport.Architectures.host_mapped_vector(
                disk_FT, 4, staging.air_mass)
            payload = Vector{disk_FT}(undef, h.elems_per_window)
            mapped = StagingMD.MappedWindowStaging(payload, payload)
            mapped_window = deepcopy(unpacked)
            foreach(a -> fill!(a, NaN), (mapped_window.air_mass..., mapped_window.fluxes.cm...,
                                         mapped_window.surface.t2m..., mapped_window.dkg...))
            for win in (2, 3, 1)
                @test StagingMD.read_window_payload!(mapped, driver, win) === mapped
                @test StagingMD.unpack_window_payload!(mapped_window, driver, mapped) === mapped_window
                @test windows_equal(mapped_window, load_transport_window(driver, win))
            end
            short = StagingMD.MappedWindowStaging(payload, payload[1:end-1])
            @test_throws DimensionMismatch StagingMD.unpack_window_payload!(mapped_window, driver, short)
            # Air mass a fraction into the window: start mass plus that fraction of dm.
            expected = map(similar, unpacked.air_mass)
            StagingMD.expected_air_mass!(expected, unpacked, 0.3)
            @test all(p -> expected[p] == unpacked.air_mass[p] .+ FT(0.3) .* unpacked.deltas.dm[p], 1:6)
        end
        @test staging.deltas !== nothing && staging.surface !== nothing &&
              staging.surface.eflux !== nothing && staging.vdiff !== nothing &&
              staging.dkg !== nothing && staging.convection.tm5_fields !== nothing &&
              staging.convection.cloud_base !== nothing
        @test eltype(staging.air_mass[1]) === FT
        for win in (2, 3, 1)
            StagingMD.load_transport_window!(staging, driver, win)
            @test windows_equal(staging, load_transport_window(driver, win))
        end
        close(driver)
    end
    end
end

# A raw window read is split into concurrent `pread`s: four on local file
# systems (and when the file system cannot be queried); on NFS four for a
# window already in the page cache, else one. The bytes read do not depend on
# the split.
@testset "Raw window reads: split by file system, same bytes" begin
    mktempdir() do dir   # a local directory on CI (no NFS there)
        local_dir = !Readers._on_nfs(dir)
        @test Readers.window_read_split(dir) isa Readers.ConcurrentWindowReads || !local_dir
        @test Readers.window_read_split(joinpath(dir, "missing", "file.bin")) isa Readers.ConcurrentWindowReads
        @test Readers.window_read_split("") isa Readers.ConcurrentWindowReads
        path = joinpath(dir, "chunks.bin")
        cs_handoff_fixture(path, [1.0, 3.0, 8.0]; FT = Float32)
        driver = TransportBinaryDriver(path; FT = Float32, arch = CPU(), Hp = 1)
        reader = driver.reader
        if local_dir
            @test reader.read_split isa Readers.ConcurrentWindowReads
            @test occursin("window reads:  4 concurrent preads", sprint(show, reader))
        end
        h = reader.header
        for win in (2, 1, 3)
            offset = Readers._transport_window_offset(reader, win)
            nbytes = h.elems_per_window * sizeof(Float32)
            local_dir && @test Readers.window_read_chunks(reader.read_split, reader.data,
                                                          offset * sizeof(Float32), nbytes) == 4
            expected = reader.data[offset .+ (1:h.elems_per_window)]
            position = h.header_bytes + offset * sizeof(Float32)
            for chunks in (Readers.NFS_WINDOW_READ_CHUNKS, 2, 3, Readers.LOCAL_WINDOW_READ_CHUNKS)
                buffer = fill(NaN32, h.elems_per_window)
                Readers._concurrent_pread!(buffer, reader.io, position, chunks)
                @test reinterpret(UInt32, buffer) == reinterpret(UInt32, expected)
            end
        end
        close(driver)
    end
end

# The NFS rule's page-cache query (`mincore` on a file mapping): a freshly
# written file is cached; after `fsync` and `POSIX_FADV_DONTNEED` Linux drops
# it unless the file lives in memory (tmpfs). Processes that cannot see a
# file's page cache read NFS windows with one `pread`.
@testset "Window residency from the page cache" begin
    supported = Sys.islinux() || Sys.isapple()
    mktempdir() do dir
        page = Readers._page_size()
        npages = 4 * Readers.RESIDENCY_SAMPLE_PAGES   # sampled, not queried whole
        path = joinpath(dir, "resident.bin")
        write(path, rand(UInt8, npages * page))
        open(path) do io
            data = Mmap.mmap(io, Vector{UInt8}, npages * page)
            split = Readers.ResidencyWindowReads()
            fraction(first, n) = Readers.resident_fraction!(split.pages, data, first, n)
            if supported
                @test fraction(0, sizeof(data)) == 1.0                 # sampled runs
                @test fraction(100, 5 * page) == 1.0                   # unaligned, whole
                @test Readers.window_read_chunks(split, data, 0, sizeof(data)) == 4
            else
                @test fraction(0, sizeof(data)) === nothing
                @test Readers.window_read_chunks(split, data, 0, sizeof(data)) == 1
            end
            @test_throws BoundsError fraction(1, sizeof(data))
            @test Readers.window_read_chunks(Readers.SequentialWindowReads(), data, 0, sizeof(data)) == 1
            # Linux shows the page cache only to the owner or a writer.
            @test Readers.page_cache_visible(path) == supported
            @test_throws ArgumentError fraction(0, -1)
            if Sys.islinux() && ccall(:geteuid, Cuint, ()) != 0 && isfile("/etc/passwd") &&
               stat("/etc/passwd").uid != ccall(:geteuid, Cuint, ())
                @test !Readers.page_cache_visible("/etc/passwd")
            end
            @test !Readers.page_cache_visible(joinpath(dir, "missing.bin")) || Sys.isapple()
            if Sys.islinux() && Base.Filesystem.diskstat(dir).ftype != 0x01021994   # not tmpfs
                @test ccall(:fsync, Cint, (Cint,), fd(io)) == 0
                @test ccall(:posix_fadvise, Cint, (Cint, Int64, Int64, Cint), fd(io), 0, 0, 4) == 0
                # The kernel may keep pages it was advised to drop; the split
                # follows whatever residency it reports.
                cold = fraction(0, sizeof(data)) < Readers.NFS_RESIDENT_FRACTION_FOR_SPLIT
                @test Readers.window_read_chunks(split, data, 0, sizeof(data)) == (cold ? 1 : 4)
            end
            finalize(data)
        end
    end
end

# A runner reuses one binary's window storage for the next binary
# (`DrivenSimulation(...; reuse_buffers_of)`) only when `same_window_layout`
# holds; the in-place refill then equals a fresh load of the new binary.
@testset "Window storage refills in place across binaries with the same layout" begin
    mktempdir() do dir
        first, second = joinpath.(dir, ["first.bin", "second.bin"])
        cs_handoff_fixture(first, [1.0, 1.0]; FT = Float32)
        cs_handoff_fixture(second, [8.0, 2.0]; FT = Float32)
        d1 = TransportBinaryDriver(first; FT = Float32, arch = CPU(), Hp = 3)
        d2 = TransportBinaryDriver(second; FT = Float32, arch = CPU(), Hp = 3)
        @test StagingMD.same_window_layout(d1, d2) && StagingMD.same_window_layout(d2, d1)
        staging = load_transport_window(d1, 2)
        for win in (1, 2)
            StagingMD.load_transport_window!(staging, d2, win)
            @test windows_equal(staging, load_transport_window(d2, win))
        end
        # Another halo, element type or payload layout does not qualify.
        halo2 = TransportBinaryDriver(second; FT = Float32, arch = CPU(), Hp = 2)
        float64 = TransportBinaryDriver(second; FT = Float64, arch = CPU(), Hp = 3)
        @test !StagingMD.same_window_layout(d1, halo2)
        @test !StagingMD.same_window_layout(d1, float64)
        other = joinpath(dir, "no_convection.bin")
        vc = HybridSigmaPressure(fill(100.0f0, 6), Float32.(range(0, 1; length = 6)))
        writer = StagingMD.open_streaming_cs_transport_binary(
            other, 3, 6, 5, 1, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
            FT = Float32, dt_met_seconds = 3600.0, steps_per_window = 2, mass_basis = :dry,
            surface_pressure_basis = DrySurfacePressure())
        try
            StagingMD.write_streaming_cs_window!(writer,
                (m = ntuple(_ -> fill(1f16, 3, 3, 5), 6), am = ntuple(_ -> zeros(Float32, 4, 3, 5), 6),
                 bm = ntuple(_ -> zeros(Float32, 3, 4, 5), 6), cm = ntuple(_ -> zeros(Float32, 3, 3, 6), 6),
                 ps = ntuple(_ -> fill(1f5, 3, 3), 6)), 3, 6)
        finally
            StagingMD.close_streaming_transport_binary!(writer)
        end
        sections = TransportBinaryDriver(other; FT = Float32, arch = CPU(), Hp = 3)
        @test !StagingMD.same_window_layout(d1, sections)
        foreach(close, (d1, d2, halo2, float64, sections))
    end
end
