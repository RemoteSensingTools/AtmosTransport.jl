# Refilling a host window in place (`load_transport_window!`, the GPU staging
# path) must give exactly the window `load_transport_window` builds, halos included.
using Test, AtmosTransport
include(joinpath(@__DIR__, "..", "fixtures", "cs_multifile.jl"))
using .CSDriverHandoffFixtures
const StagingMD = AtmosTransport.MetDrivers

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
