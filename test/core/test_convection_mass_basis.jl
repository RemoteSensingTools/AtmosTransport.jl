#!/usr/bin/env julia
# Convective mass fluxes on the binary's air-mass basis (issue #19).
#
# ERA5 UDMF/DDMF/UDRF/DDRF and the TM5-archive eu/du/ed/dd are per unit of
# total (moist) air, while the default binary and the runtime air mass are
# dry. `TM5Convection` divides every flux by the air mass, so moist fluxes on
# dry air inflate the convective rates by 1/(1 − q). The preprocessors
# multiply them by (1 − q) (interfaces: the GEOS face rule) and record the
# basis in the header; a dry binary without the record warns at load time.

using Test
using Random
using JSON3

using AtmosTransport
using .AtmosTransport.State: CellState
using .AtmosTransport.Operators: TM5Convection, TM5Workspace, apply!
using .AtmosTransport.MetDrivers: ConvectionForcing, TM5_CONVECTION_MASS_BASIS_KEY,
    tm5_convection_mass_basis
using .AtmosTransport.Preprocessing: interface_specific_humidity, ecmwf_convection_to_basis!,
    tm5_convection_to_basis!, ec2tm_from_rates!, dz_hydrostatic_virtual!,
    tm5_native_fields_for_hour!, _moist_to_dry_cmfmc!

@testset "interface humidity rule (shared with GEOS CMFMC)" begin
    q = [0.001, 0.004, 0.010]
    @test [interface_specific_humidity(k, 3, q) for k in 1:4] == [0.001, 0.0025, 0.007, 0.010]

    # GEOS CMFMC keeps its four-corner arithmetic bit for bit.
    rng = Xoshiro(19)
    for FT in (Float32, Float64)
        Nc, Nz = 5, 7
        q1 = ntuple(_ -> FT(0.02) .* rand(rng, FT, Nc, Nc, Nz), 6)
        q2 = ntuple(_ -> FT(0.02) .* rand(rng, FT, Nc, Nc, Nz), 6)
        cmfmc = ntuple(_ -> rand(rng, FT, Nc, Nc, Nz + 1), 6)
        legacy = map(copy, cmfmc)
        for p in 1:6, k in 1:(Nz + 1), j in 1:Nc, i in 1:Nc
            qf = k == 1 ? (q1[p][i, j, 1] + q2[p][i, j, 1]) * FT(0.5) :
                 k == Nz + 1 ? (q1[p][i, j, Nz] + q2[p][i, j, Nz]) * FT(0.5) :
                 (q1[p][i, j, k-1] + q1[p][i, j, k] + q2[p][i, j, k-1] + q2[p][i, j, k]) * FT(0.25)
            legacy[p][i, j, k] *= one(FT) - qf
        end
        @test _moist_to_dry_cmfmc!(map(copy, cmfmc), q1, q2) == legacy
    end
end

@testset "ecmwf_convection_to_basis!" begin
    Nz = 4
    q = [0.0002, 0.003, 0.009, 0.016]
    udmf = [0.0, 0.01, 0.03, 0.02, 0.0]
    ddmf = [0.0, 0.0, -0.004, -0.006, 0.0]
    udrf = [1e-5, 2e-5, 0.0, 0.0]
    ddrf = [0.0, 1e-6, 3e-6, 0.0]
    fields = map(copy, (udmf, ddmf, udrf, ddrf))
    ecmwf_convection_to_basis!(MoistBasis(), fields..., q)
    @test fields == (udmf, ddmf, udrf, ddrf)

    ecmwf_convection_to_basis!(DryBasis(), fields..., q)
    q_interface = [q[1], (q[1] + q[2]) / 2, (q[2] + q[3]) / 2, (q[3] + q[4]) / 2, q[4]]
    @test fields[1] == udmf .* (1 .- q_interface)
    @test fields[2] == ddmf .* (1 .- q_interface)
    @test fields[3] == udrf .* (1 .- q)
    @test fields[4] == ddrf .* (1 .- q)

    dry_zero_q = map(copy, (udmf, ddmf, udrf, ddrf))
    ecmwf_convection_to_basis!(DryBasis(), dry_zero_q..., zeros(Nz))
    @test dry_zero_q == (udmf, ddmf, udrf, ddrf)
end

# One humid tropical column (the reporter's set-up): deep updraft and a
# downdraft, unit tracer in the lowest layer, TM5Convection on a 4×3 grid of
# identical columns.
const GRAVITY = 9.80665
const NZ = 40
const PS = 101_000.0
const B_IFC = collect(range(0.0, 1.0; length = NZ + 1))
const A_IFC = zeros(NZ + 1)
const P_IFC = A_IFC .+ B_IFC .* PS
const P_MID = 0.5 .* (P_IFC[1:end-1] .+ P_IFC[2:end])
const DP = diff(P_IFC)
const TEMPERATURE = max.(300.0 .* (P_MID ./ PS) .^ 0.19, 195.0)
const NX, NY = 4, 3

ramp(x, a, b) = clamp((x - a) / (b - a), 0.0, 1.0)
updraft_flux(p) = 0.04 * ramp(p, 1.0e5, 8.0e4) * ramp(p, 1.5e4, 4.0e4)
downdraft_flux(p) = -0.012 * ramp(p, 5.5e4, 8.0e4) * ramp(p, 1.0e5, 9.0e4)
bcast(v) = repeat(reshape(v, 1, 1, :), NX, NY, 1)

# ECMWF-like hourly fields on the grid; `tm5_native_fields_for_hour!` takes
# UDMF/DDMF with level k at the bottom interface of layer k.
function ecmwf_grid_fields(q)
    dz = zeros(NZ)
    dz_hydrostatic_virtual!(dz, TEMPERATURE, q, PS, A_IFC, B_IFC, NZ)
    mu, md = updraft_flux.(P_IFC), downdraft_flux.(P_IFC)
    udrf = [max(mu[k + 1] - mu[k], 0.0) / dz[k] for k in 1:NZ]
    ddrf = [max(md[k] - md[k + 1], 0.0) / dz[k] for k in 1:NZ]
    return (udmf = bcast(mu[2:end]), ddmf = bcast(md[2:end]), udrf = bcast(udrf),
            ddrf = bcast(ddrf), t = bcast(TEMPERATURE), q = bcast(q), ps = fill(PS, NX, NY))
end

function tm5_fields(basis, q)
    f = ecmwf_grid_fields(q)
    out = ntuple(_ -> zeros(NX, NY, NZ), 4)
    tm5_native_fields_for_hour!(out..., f.udmf, f.ddmf, f.udrf, f.ddrf, f.t, f.q, f.ps,
                                A_IFC, B_IFC, NZ; basis)
    return (entu = out[1], detu = out[2], entd = out[3], detd = out[4])
end

function convect(fields, m, tracer; nstep)
    grid = AtmosGrid(LatLonMesh(; Nx = NX, Ny = NY, FT = Float64),
                     HybridSigmaPressure(A_IFC, B_IFC), AtmosTransport.CPU(); FT = Float64)
    state = CellState(DryBasis, bcast(m); TR = bcast(tracer))
    forcing = ConvectionForcing(nothing, nothing, fields)
    ws = TM5Workspace(state.air_mass; cell_metrics = ones(NY))
    for _ in 1:nstep
        apply!(state, forcing, grid, TM5Convection(), 900.0; workspace = ws)
    end
    return state.tracers_raw[1, 1, :, 1]
end

above_500hPa(r) = sum(r[P_MID .< 5.0e4])

@testset "dry-basis TM5 fields on a humid column" begin
    surface_unit = [k == NZ ? 1.0 : 0.0 for k in 1:NZ]

    # Control: without humidity the conversion is the identity.
    @test tm5_fields(DryBasis(), zeros(NZ)) == tm5_fields(MoistBasis(), zeros(NZ))

    q = 0.018 .* (P_MID ./ PS) .^ 3
    dry, moist = tm5_fields(DryBasis(), q), tm5_fields(MoistBasis(), q)
    m_dry, m_moist = DP ./ GRAVITY .* (1 .- q), DP ./ GRAVITY
    for f in (dry, moist), j in 1:NY, i in 1:NX       # ec2tm column closure
        @test sum(f.entu[i, j, :]) ≈ sum(f.detu[i, j, :]) rtol = 1e-12
        @test sum(f.entd[i, j, :]) ≈ sum(f.detd[i, j, :]) rtol = 1e-12
        @test all(>=(0), f.entu) && all(>=(0), f.detu) && all(>=(0), f.entd) && all(>=(0), f.detd)
    end
    # Entrainment below 800 hPa (q = 0.009-0.018) is about 1 % smaller in dry air.
    low = P_MID .> 8.0e4
    @test 0.98 < sum(dry.entu[1, 1, low]) / sum(moist.entu[1, 1, low]) < 0.995

    for nstep in (1, 24)
        dry_on_dry = convect(dry, m_dry, surface_unit; nstep)
        moist_on_dry = convect(moist, m_dry, surface_unit; nstep)       # before the fix
        tm5 = convect(moist, m_moist, surface_unit; nstep)            # TM5: moist on moist
        @test sum(dry_on_dry) ≈ 1 rtol = 1e-12                           # tracer mass conserved
        # Basis-consistent rates match TM5's own; moist fluxes on dry air are >1 % fast.
        @test abs(above_500hPa(dry_on_dry) / above_500hPa(tm5) - 1) < 0.005
        @test above_500hPa(moist_on_dry) / above_500hPa(tm5) - 1 > 0.01
    end

    # A uniform mixing ratio stays uniform under dry fluxes on dry air.
    uniform = convect(dry, m_dry, 4e-4 .* m_dry; nstep = 24)
    @test maximum(abs, uniform ./ (4e-4 .* m_dry) .- 1) < 1e-10
end

@testset "tm5_convection_to_basis! (TM5-archive fields)" begin
    q = 0.018 .* (P_MID ./ PS) .^ 3
    moist = tm5_fields(MoistBasis(), q)
    column(f) = map(x -> collect(x[1, 1, :]), f)
    entu, detu, entd, detd = column(moist)

    unchanged = map(copy, (entu, detu, entd, detd))
    tm5_convection_to_basis!(MoistBasis(), unchanged..., q)
    @test unchanged == (entu, detu, entd, detd)

    # Uniform humidity scales every field by (1 − q).
    uniform = map(copy, (entu, detu, entd, detd))
    tm5_convection_to_basis!(DryBasis(), uniform..., fill(0.01, NZ))
    for (converted, original) in zip(uniform, (entu, detu, entd, detd))
        @test converted ≈ 0.99 .* original rtol = 1e-12 atol = 1e-15 * maximum(original)
    end

    converted = map(copy, (entu, detu, entd, detd))
    tm5_convection_to_basis!(DryBasis(), converted..., q)
    e_u, d_u, e_d, d_d = converted
    @test all(>=(0), e_u) && all(>=(0), d_u) && all(>=(0), e_d) && all(>=(0), d_d)
    @test sum(e_u) ≈ sum(d_u) rtol = 1e-12                  # closure kept
    @test sum(e_d) ≈ sum(d_d) rtol = 1e-12
    # Nothing appears above each draft (TM5Convection reads the cloud top from
    # the first non-zero detu, the downdraft top from the first non-zero entd).
    top_u = findfirst(k -> entu[k] != 0 || detu[k] != 0, 1:NZ)
    top_d = findfirst(k -> entd[k] != 0 || detd[k] != 0, 1:NZ)
    @test all(iszero, e_u[1:top_u-1]) && all(iszero, d_u[1:top_u-1])
    @test all(iszero, e_d[1:top_d-1]) && all(iszero, d_d[1:top_d-1])
    @test findfirst(>(0), d_u) == findfirst(>(0), detu)
    @test findfirst(>(0), e_d) == findfirst(>(0), entd)
    # The draft fluxes implied by the converted fields are the moist ones times
    # (1 − q at the interface); detrainment is converted where nothing was
    # redistributed.
    q_interface = [interface_specific_humidity(k, NZ, q) for k in 1:NZ+1]
    up_moist = [sum(entu[k:end] .- detu[k:end]) for k in 1:NZ]          # interface k, upward
    up_dry = [sum(e_u[k:end] .- d_u[k:end]) for k in 1:NZ]
    down_moist = [sum(entd[1:k] .- detd[1:k]) for k in 1:NZ]            # interface k+1, downward
    down_dry = [sum(e_d[1:k] .- d_d[1:k]) for k in 1:NZ]
    @test up_dry ≈ (1 .- q_interface[1:NZ]) .* up_moist atol = 1e-12 * maximum(abs, up_moist)
    @test down_dry ≈ (1 .- q_interface[2:NZ+1]) .* down_moist atol = 1e-12 * maximum(abs, down_moist)
    @test d_u[e_u .> 0] == detu[e_u .> 0] .* (1 .- q[e_u .> 0])
    # Converting the ECMWF inputs before ec2tm gives the same boundary-layer
    # entrainment and total detrainment. (Per level the two differ where TM5's
    # closure puts its cloud-top residual: the layer of largest entrainment,
    # which moves within this column's plateau when the fluxes are converted.)
    dry = column(tm5_fields(DryBasis(), q))
    low = P_MID .> 8.0e4
    @test sum(e_u[low]) ≈ sum(dry[1][low]) rtol = 1e-3
    @test sum(d_u) ≈ sum(dry[2]) rtol = 1e-3
    @test sum(d_d) ≈ sum(dry[4]) rtol = 1e-3
end

@testset "TM5 convection basis in the binary header" begin
    MD = AtmosTransport.MetDrivers
    basis_kw(tm5_basis) = tm5_basis === :default ? (;) :
                           (; tm5_convection_mass_basis = tm5_basis)
    add_tm5(window, field) = merge(window, (;
        tm5_fields = (entu = field(), detu = field(), entd = field(), detd = field())))

    function ll_fixture(include_tm5)
        Nx, Ny, Nz = 4, 3, 2
        grid = AtmosGrid(LatLonMesh(; FT = Float64, Nx, Ny),
                         HybridSigmaPressure([0.0, 1000, 2000], [0.0, 0.2, 1.0]),
                         CPU(); FT = Float64)
        window = (m = fill(1e16, Nx, Ny, Nz), am = zeros(Nx + 1, Ny, Nz),
                  bm = zeros(Nx, Ny + 1, Nz), cm = zeros(Nx, Ny, Nz + 1),
                  ps = fill(95_000.0, Nx, Ny))
        return grid, include_tm5 ? add_tm5(window, () -> zeros(Nx, Ny, Nz)) : window
    end

    function rg_fixture(include_tm5)
        Nz = 2
        mesh = ReducedGaussianMesh([-45.0, 45.0], [4, 4]; FT = Float64)
        grid = AtmosGrid(mesh, HybridSigmaPressure([0.0, 1000, 2000], [0.0, 0.2, 1.0]),
                         CPU(); FT = Float64)
        window = (m = fill(1e16, ncells(mesh), Nz),
                  hflux = zeros(nfaces(mesh), Nz),
                  cm = zeros(ncells(mesh), Nz + 1), ps = fill(95_000.0, ncells(mesh)))
        return grid, include_tm5 ? add_tm5(window, () -> zeros(ncells(mesh), Nz)) : window
    end

    function write_eager_ll(path; include_tm5, tm5_basis)
        grid, window = ll_fixture(include_tm5)
        write_transport_binary(path, grid, [window]; FT = Float64,
            dt_met_seconds = 3600.0, half_dt_seconds = 1800.0, steps_per_window = 1,
            mass_basis = :dry, source_flux_sampling = :window_start_endpoint,
            flux_sampling = :window_constant, basis_kw(tm5_basis)...)
    end

    function write_eager_rg(path; include_tm5, tm5_basis)
        grid, window = rg_fixture(include_tm5)
        write_transport_binary(path, grid, [window]; FT = Float64,
            dt_met_seconds = 3600.0, half_dt_seconds = 1800.0, steps_per_window = 1,
            mass_basis = :dry, source_flux_sampling = :window_start_endpoint,
            flux_sampling = :window_constant, basis_kw(tm5_basis)...)
    end

    function write_streaming_rg(path; include_tm5, tm5_basis)
        grid, window = rg_fixture(include_tm5)
        writer = MD.open_streaming_transport_binary(
            path, grid, 1, window; FT = Float64, steps_per_window = 1,
            mass_basis = :dry, source_flux_sampling = :window_start_endpoint,
            flux_sampling = :window_constant, basis_kw(tm5_basis)...)
        MD.write_streaming_window!(writer, window)
        MD.close_streaming_transport_binary!(writer)
    end

    function write_streaming_cs(path; include_tm5, tm5_basis)
        Nc, np, Nz = 2, 6, 2
        panels3() = ntuple(_ -> zeros(Float64, Nc, Nc, Nz), np)
        window = (m = ntuple(_ -> fill(1e15, Nc, Nc, Nz), np),
                  am = ntuple(_ -> zeros(Nc + 1, Nc, Nz), np),
                  bm = ntuple(_ -> zeros(Nc, Nc + 1, Nz), np),
                  cm = ntuple(_ -> zeros(Nc, Nc, Nz + 1), np),
                  ps = ntuple(_ -> fill(95_000.0, Nc, Nc), np))
        include_tm5 && (window = add_tm5(window, panels3))
        writer = MD.open_streaming_cs_transport_binary(
            path, Nc, np, Nz, 1,
            HybridSigmaPressure([0.0, 1000, 2000], [0.0, 0.2, 1.0]);
            planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
            steps_per_window = 1, include_tm5conv = include_tm5,
            mass_basis = :dry, basis_kw(tm5_basis)...)
        MD.write_streaming_cs_window!(writer, window, Nc, np)
        MD.close_streaming_transport_binary!(writer)
    end

    function raw_header(path)
        raw = open(io -> MD._read_transport_header_json(io; source = path), path)
        return JSON3.read(String(raw), Dict{String, Any})
    end

    function rewrite_tm5_basis!(path, basis)
        header = raw_header(path)
        header[TM5_CONVECTION_MASS_BASIS_KEY] = String(basis)
        header_bytes = Int(header["header_bytes"])
        encoded = Vector{UInt8}(codeunits(JSON3.write(header)))
        length(encoded) < header_bytes || error("test header no longer fits")
        open(path, "r+") do io
            write(io, encoded)
            write(io, zeros(UInt8, header_bytes - length(encoded)))
        end
        return path
    end

    mktempdir() do dir
        writers = (("eager LL", write_eager_ll), ("eager RG", write_eager_rg),
                   ("streaming RG", write_streaming_rg), ("streaming CS", write_streaming_cs))
        cases = (("inferred TM5 basis", true, :default, "dry"),
                 ("explicit keyless TM5", true, nothing, nothing),
                 ("non-TM5 payload", false, :default, nothing))
        @testset "$writer_name: $case_name" for (writer_name, write_fixture) in writers,
                                                  (case_name, include_tm5, tm5_basis, expected) in cases
            path = joinpath(dir, replace(lowercase("$writer_name-$case_name"), ' ' => '_') * ".bin")
            write_fixture(path; include_tm5, tm5_basis)
            header = raw_header(path)
            if expected === nothing
                @test !haskey(header, TM5_CONVECTION_MASS_BASIS_KEY)
            else
                @test header[TM5_CONVECTION_MASS_BASIS_KEY] == expected
            end
        end

        recorded = joinpath(dir, "eager_ll-inferred_tm5_basis.bin")
        reader = TransportBinaryReader(recorded; FT = Float64)
        @test tm5_convection_mass_basis(reader) === :dry
        @test binary_capabilities(reader).tm5_convection_mass_basis === :dry
        close(reader)
        @test_logs min_level = Base.CoreLogging.Warn close(TransportBinaryDriver(recorded))

        legacy = joinpath(dir, "eager_ll-explicit_keyless_tm5.bin")
        reader = nothing
        @test_logs (:warn, r"tm5_convection_mass_basis.*1/\(1-q\)") begin
            reader = TransportBinaryReader(legacy; FT = Float64)
        end
        @test tm5_convection_mass_basis(reader) === nothing
        report = nothing
        @test_logs (:warn, r"tm5_convection_mass_basis.*1/\(1-q\)") begin
            report = sprint(io -> inspect_binary(legacy; io))
        end
        @test occursin("tm5_convection_mass_basis = not recorded", report)
        close(reader)

        # A moist binary without the record is consistent (fluxes and mass moist).
        grid, window = ll_fixture(true)
        legacy_moist = joinpath(dir, "legacy_moist.bin")
        write_transport_binary(legacy_moist, grid, [window]; FT = Float64,
            steps_per_window = 1, mass_basis = :moist,
            source_flux_sampling = :window_start_endpoint,
            tm5_convection_mass_basis = nothing)
        @test_logs min_level = Base.CoreLogging.Warn close(TransportBinaryReader(legacy_moist))

        @test_throws ArgumentError write_eager_ll(
            joinpath(dir, "writer_mismatch.bin"); include_tm5 = true, tm5_basis = :moist)

        # Simulate an externally generated binary. This bypasses every writer
        # guard so the reader's independent basis check must reject it.
        mismatch = joinpath(dir, "external_mismatch.bin")
        cp(recorded, mismatch)
        rewrite_tm5_basis!(mismatch, :moist)
        reader_error = try
            TransportBinaryReader(mismatch; FT = Float64)
            nothing
        catch err
            err
        end
        @test reader_error isa ArgumentError
        @test occursin("TM5 convection fluxes are on a moist basis", sprint(showerror, reader_error))
    end
end
