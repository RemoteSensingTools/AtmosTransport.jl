using Test, AtmosTransport, Random
using KernelAbstractions: CPU as KA_CPU
using AtmosTransport.Output: ObservationGatherBuffers, gather_columns!, gather_field!, interface_pressures!,
                             layer_heights_agl!, intake_layer_index, column_mean_vmr,
                             mixing_ratio_profile!, ConstantLayerTemperature,
                             SurfaceLapseTemperature, ProfileLayerTemperature
const O = AtmosTransport.Output

function fake_state(topology, FT; Nz = 5, Hp = 3)
    rng = MersenneTwister(3)
    if topology === :cs
        Nc = 4
        Np = Nc + 2Hp
        air = ntuple(p -> FT.(rand(rng, Np, Np, Nz) .+ p), 6)
        co2 = map(m -> m .* FT(400e-6), air)
        ch4 = map(m -> m .* FT(1.9e-6), air)
        return CubedSphereState(DryBasis, air; co2 = co2, dummy = air, ch4 = ch4, halo_width = Hp), Nc, Hp
    elseif topology === :rg
        air = FT.(rand(rng, 12, Nz) .+ 1)
        return CellState(DryBasis, air; co2 = air .* FT(400e-6), dummy = air, ch4 = air .* FT(1.9e-6)), 12, 0
    else
        air = FT.(rand(rng, 6, 4, Nz) .+ 1)
        return CellState(DryBasis, air; co2 = air .* FT(400e-6), dummy = air, ch4 = air .* FT(1.9e-6)), (6, 4), 0
    end
end

@testset "gather matches direct indexing on every topology and precision" begin
    for FT in (Float64, Float32), topology in (:ll, :rg, :cs)
        state, shape, Hp = fake_state(topology, FT)
        slots = [AtmosTransport.State.tracer_index(state, :ch4), AtmosTransport.State.tracer_index(state, :co2)]
        reference = topology === :cs ? state.air_mass[1] : state.air_mass
        buf = ObservationGatherBuffers(reference, 5, slots; capacity = 2)
        @test O.nslots(buf) == 2
        if topology === :cs
            Nc = shape
            Np = Nc + 2Hp
            # 7 requests grouped by panel (panels 1, 3, 3, 3, 5, 6, 6).
            requests = [(1, 1, 1), (3, 2, 3), (3, 4, 4), (3, 1, 2), (5, 3, 3), (6, 4, 1), (6, 2, 2)]
            columns = Int32[(Hp + i) + (Hp + j - 1) * Np for (_, i, j) in requests]
            ranges = [1:1, 2:1, 2:4, 5:4, 5:5, 6:7]
            n = gather_columns!(buf, state, columns, ranges)
            @test n == 7
            @test buf.capacity >= 7                    # grew from 2
            for (r, (p, i, j)) in enumerate(requests), k in 1:5
                @test buf.air_host[k, r] == state.air_mass[p][Hp + i, Hp + j, k]
                @test buf.tracers_host[k, 1, r] == state.tracers_raw[p][Hp + i, Hp + j, k, slots[1]]
                @test buf.tracers_host[k, 2, r] == state.tracers_raw[p][Hp + i, Hp + j, k, slots[2]]
            end
            # Ranges must tile 1:n in order: wrong count, overlap, gap, out of bounds.
            @test_throws ArgumentError gather_columns!(buf, state, columns, ranges[1:5])
            @test_throws ArgumentError gather_columns!(buf, state, columns, [1:2, 2:1, 2:4, 5:4, 5:5, 6:7])
            @test_throws ArgumentError gather_columns!(buf, state, columns, [1:1, 2:1, 3:4, 5:4, 5:5, 6:7])
            @test_throws ArgumentError gather_columns!(buf, state, columns, [1:1, 2:1, 2:4, 5:4, 5:5, 6:9])
            # A column outside the slab is rejected before any device read.
            bad = copy(columns); bad[3] = Int32(Np * Np + 1)
            @test_throws BoundsError gather_columns!(buf, state, bad, ranges)
            # A field gather over the same plan.
            field_buf = ObservationGatherBuffers(reference, 5, Int[])
            @test gather_field!(field_buf, state.air_mass, columns, ranges) == 7
            @test field_buf.air_host[:, 1:7] == buf.air_host[:, 1:7]
        elseif topology === :rg
            cells = Int32[1, 5, 12, 7, 7]
            n = gather_columns!(buf, state, cells)
            @test n == 5
            for (r, c) in enumerate(cells), k in 1:5
                @test buf.air_host[k, r] == state.air_mass[c, k]
                @test buf.tracers_host[k, 2, r] == state.tracers_raw[c, k, slots[2]]
            end
            @test_throws BoundsError gather_columns!(buf, state, Int32[13])
            @test_throws BoundsError gather_columns!(buf, state, Int32[0])
        else
            Nx, Ny = shape
            requests = [(1, 1), (6, 4), (3, 2), (6, 1)]
            columns = Int32[i + (j - 1) * Nx for (i, j) in requests]
            n = gather_columns!(buf, state, columns, nothing)
            @test n == 4
            for (r, (i, j)) in enumerate(requests), k in 1:5
                @test buf.air_host[k, r] == state.air_mass[i, j, k]
                @test buf.tracers_host[k, 1, r] == state.tracers_raw[i, j, k, slots[1]]
            end
            # Empty requests are a no-op; a smaller follow-up gather after growth is exact.
            @test gather_columns!(buf, state, Int32[]) == 0
            @test gather_columns!(buf, state, Int32[columns[2]]) == 1
            @test buf.air_host[:, 1] == state.air_mass[6, 4, :]
            # Slot validation against the state: a buffer asking for slot 4 of a 3-tracer state.
            too_many = ObservationGatherBuffers(reference, 5, [4])
            @test_throws ArgumentError gather_columns!(too_many, state, Int32[1])
        end
    end
end

@testset "the kernel body matches the host loop on the CPU backend" begin
    state, (Nx, Ny), _ = fake_state(:ll, Float32)
    slots = Int32[3, 1]
    columns = Int32[1, 7, 24, 13]
    ncolumn = Nx * Ny
    ref_air = zeros(Float32, 5, 6); ref_tr = zeros(Float32, 5, 2, 6)
    O._gather_columns_host!(ref_air, ref_tr, state.air_mass, state.tracers_raw, columns, slots, ncolumn, 5, 1, 3)
    out_air = zeros(Float32, 5, 6); out_tr = zeros(Float32, 5, 2, 6)
    O._gather_columns!(KA_CPU())(out_air, out_tr, state.air_mass, state.tracers_raw, columns, slots,
                                 ncolumn, 5, 1; ndrange = (5, 3))
    @test out_air == ref_air
    @test out_tr == ref_tr
    @test all(ref_air[:, 1] .== 0) && all(ref_air[:, 5:6] .== 0)         # offset = 1, count = 3
    @test ref_tr[:, 1, 3] == state.tracers_raw[6, 4, :, 3]                  # output slot 3 holds column 24 = (6, 4)
    # Column vectors that are views are accepted (converted before upload).
    vbuf = ObservationGatherBuffers(state.air_mass, 5, [1])
    @test gather_columns!(vbuf, state, view(columns, 2:3)) == 2
    @test vbuf.air_host[:, 1] == state.air_mass[1, 2, :]
end

@testset "buffer validation" begin
    air = rand(3, 2, 4)
    @test_throws ArgumentError ObservationGatherBuffers(air, 0, [1])
    @test_throws ArgumentError ObservationGatherBuffers(air, 4, [0])
    state = CellState(DryBasis, air; co2 = air .* 1e-6)
    buf = ObservationGatherBuffers(air, 3, [1])
    @test_throws DimensionMismatch gather_columns!(buf, state, Int32[1])
    field_buf = ObservationGatherBuffers(air, 4, Int[])
    @test O.nslots(field_buf) == 0
    @test gather_field!(field_buf, air, Int32[6, 1]) == 2
    @test field_buf.air_host[:, 1] == air[3, 2, :]   # column 6 of a 3×2 slab
    @test field_buf.air_host[:, 2] == air[1, 1, :]
    @test_throws ArgumentError gather_field!(ObservationGatherBuffers(air, 4, [1]), air, Int32[1])
end

@testset "pressure, heights, intake layer, mixing ratios" begin
    g = 9.80665
    area = 2.5e9
    ps = 98_000.0
    p_top = 1.0
    nlevel = 6
    p_half_ref = collect(range(p_top, ps; length = nlevel + 1))
    dp = diff(p_half_ref)
    air = dp .* area ./ g
    p_half = zeros(nlevel + 1)
    interface_pressures!(p_half, air, area, g, p_top)
    @test p_half ≈ p_half_ref rtol = 1e-14
    @test p_half[1] == p_top
    @test interface_pressures!(zeros(nlevel + 1), Float32.(air), area, g, p_top) ≈ p_half_ref rtol = 1e-6
    @test_throws DimensionMismatch interface_pressures!(zeros(3), air, area, g, p_top)

    # Isothermal column: z = R T / g ln(ps / p).
    T = 280.0
    z_half = zeros(nlevel + 1)
    layer_heights_agl!(z_half, p_half, ConstantLayerTemperature(T), g)
    @test z_half[end] == 0
    @test z_half ≈ O.OBSERVATION_R_DRY * T / g .* log.(ps ./ p_half) rtol = 1e-12
    @test issorted(z_half; rev = true)
    # Zero top pressure → infinite top height, finite elsewhere; non-monotone or NaN
    # pressures poison everything above them.
    p_zero_top = copy(p_half); p_zero_top[1] = 0.0
    layer_heights_agl!(z_half, p_zero_top, ConstantLayerTemperature(T), g)
    @test isinf(z_half[1]) && all(isfinite, z_half[2:end])
    p_bad = copy(p_half); p_bad[3] = p_bad[4] + 1
    layer_heights_agl!(z_half, p_bad, ConstantLayerTemperature(T), g)
    @test all(isinf, z_half[1:3]) && all(isfinite, z_half[4:end])
    p_nan = copy(p_half); p_nan[2] = NaN
    layer_heights_agl!(z_half, p_nan, ConstantLayerTemperature(T), g)
    @test all(isinf, z_half[1:2]) && all(isfinite, z_half[3:end]) && !any(isnan, z_half)
    # Lapse-rate and profile temperatures.
    lapse = SurfaceLapseTemperature(290.0)
    @test lapse.lapse_rate == 0.0065
    @test O.layer_temperature(lapse, 3, 1000.0) == 290.0 - 6.5
    @test O.layer_temperature(lapse, 3, 50_000.0) == O.STANDARD_TROPOPAUSE_KELVIN   # floored aloft
    profile = ProfileLayerTemperature(collect(220.0:10.0:270.0))
    @test_throws DimensionMismatch layer_heights_agl!(zeros(nlevel + 1), p_half, ProfileLayerTemperature([250.0]), g)
    @test O.layer_temperature(profile, 1, 0.0) == 220.0
    @test O.layer_temperature(profile, 6, 0.0) == 270.0
    @test O.height_method_label(ConstantLayerTemperature(1.0)) === :constant
    @test O.height_method_label(lapse) === :surface_lapse
    @test O.height_method_label(profile) === :profile
    z_lapse = layer_heights_agl!(zeros(nlevel + 1), p_half, lapse, g)
    z_const = layer_heights_agl!(zeros(nlevel + 1), p_half, ConstantLayerTemperature(290.0), g)
    @test z_lapse[end-1] == z_const[end-1]
    @test all(z_lapse[1:end-2] .< z_const[1:end-2])
    # A tall 72-level column with a 0.01 Pa top stays monotone and finite under the floor.
    tall = exp.(range(log(0.01), log(101_325.0); length = 73))
    z_tall = layer_heights_agl!(zeros(73), tall, SurfaceLapseTemperature(288.15), g)
    @test all(isfinite, z_tall) && issorted(z_tall; rev = true)
    @test 60_000 < z_tall[1] < 120_000

    z = [1000.0, 500.0, 100.0, 0.0]
    @test intake_layer_index(z, 50.0) == 3
    @test intake_layer_index(z, 100.0) == 2
    @test intake_layer_index(z, 499.9) == 2
    @test intake_layer_index(z, 500.0) == 1
    @test intake_layer_index(z, 2000.0) == 1
    @test intake_layer_index(z, NaN) == 3
    @test intake_layer_index(z, 0.0) == 3
    @test intake_layer_index(z, -5.0) == 3

    m = [2.0, 0.0, 4.0]
    rm = [2.0 * 400e-6, 0.0, 4.0 * 410e-6]
    @test column_mean_vmr(m, rm) ≈ (2 * 400e-6 + 4 * 410e-6) / 6
    @test isnan(column_mean_vmr([0.0, 0.0], [0.0, 0.0]))
    out = zeros(3)
    mixing_ratio_profile!(out, m, rm)
    @test out[1] ≈ 400e-6 && isnan(out[2]) && out[3] ≈ 410e-6
    @test column_mean_vmr(Float32[1, 1], Float32[1f-6, 3f-6]) ≈ 2e-6 rtol = 1e-6
end
