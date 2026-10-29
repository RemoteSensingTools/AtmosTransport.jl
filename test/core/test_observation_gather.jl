using Test, AtmosTransport, Random, TOML
using KernelAbstractions: CPU as KA_CPU
using AtmosTransport.Observations: ObservationGatherBuffers, gather_columns!, gather_field!,
                                   interface_pressures!, layer_heights_agl!, intake_layer_index,
                                   column_mean_vmr, mixing_ratio_profile!, ConstantLayerTemperature,
                                   SurfaceLapseTemperature, ProfileLayerTemperature
const O = AtmosTransport.Observations

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
    O.Sampling._gather_columns_host!(ref_air, ref_tr, state.air_mass, state.tracers_raw, columns, slots, ncolumn, 5, 1, 3)
    out_air = zeros(Float32, 5, 6); out_tr = zeros(Float32, 5, 2, 6)
    O.Sampling._gather_columns!(KA_CPU())(out_air, out_tr, state.air_mass, state.tracers_raw, columns, slots,
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
    @test z_half ≈ AtmosTransport.Parameters.R_DRY_AIR * T / g .* log.(ps ./ p_half) rtol = 1e-12
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
    @test O.layer_temperature(lapse, 3, 50_000.0) ==
          AtmosTransport.Parameters.STANDARD_TROPOPAUSE_TEMPERATURE   # floored aloft
    profile = ProfileLayerTemperature(collect(220.0:10.0:270.0))
    @test_throws DimensionMismatch layer_heights_agl!(zeros(nlevel + 1), p_half, ProfileLayerTemperature([250.0]), g)
    @test O.layer_temperature(profile, 1, 0.0) == 220.0
    @test O.layer_temperature(profile, 6, 0.0) == 270.0
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
    # A missing height is never a silent lowest-layer fallback (issue #56).
    @test_throws ArgumentError intake_layer_index(z, NaN)
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

@testset "intake layers on the real GEOS L72 grid (k = 1 at the top)" begin
    cfg = TOML.parsefile(joinpath(@__DIR__, "..", "..", "config", "geos_L72_coefficients.toml"))
    A = Float64.(cfg["coefficients"]["a"])
    B = Float64.(cfg["coefficients"]["b"])
    @test B[1] == 0 && B[end] == 1                  # file is top-down like the model
    ps, g, area = 101325.0, 9.80665, 1e10
    air = diff(A .+ B .* ps) .* area ./ g
    p_half = interface_pressures!(zeros(73), air, area, g, A[1])
    @test p_half[end] ≈ ps rtol = 1e-12
    z = layer_heights_agl!(zeros(73), p_half, ConstantLayerTemperature(280.0), g)
    @test 110 < z[72] < 140                          # surface layer ~124 m thick
    @test intake_layer_index(z, 10.0) == 72
    @test intake_layer_index(z, 100.0) == 72
    @test intake_layer_index(z, 300.0) == 70
    @test intake_layer_index(z, 500.0) == 69
end

@testset "total interface pressures from the dry air mass and specific humidity (issue #20)" begin
    # A tropical L72 column: total interfaces A + B·ps, specific humidity
    # 18 g/kg at the surface falling as (p/ps)^2.7 (about 50 kg m⁻² of
    # precipitable water), dry air mass (1 − q) Δp A / g per layer.
    cfg = TOML.parsefile(joinpath(@__DIR__, "..", "..", "config", "geos_L72_coefficients.toml"))
    A = Float64.(cfg["coefficients"]["a"])
    B = Float64.(cfg["coefficients"]["b"])
    ps, g, area = 101000.0, 9.80665, 1e10
    p_ref = A .+ B .* ps
    q = 0.018 .* ((p_ref[1:end-1] .+ p_ref[2:end]) ./ (2ps)) .^ 2.7
    dry = (1 .- q) .* diff(p_ref) .* area ./ g
    p_total = O.total_interface_pressures!(zeros(73), dry, q, area, g, A[1])
    @test p_total ≈ p_ref rtol = 1e-13
    @test O.total_interface_pressures!(zeros(73), Float32.(dry), Float32.(q), area, g, A[1]) ≈ p_ref rtol = 1e-6
    # The dry interfaces lie lower by the weight of the vapour above: 4.9 hPa at the surface here.
    p_dry = interface_pressures!(zeros(73), dry, area, g, A[1])
    @test p_ref .- p_dry ≈ vcat(0.0, cumsum(q .* diff(p_ref))) rtol = 1e-10
    @test 480 < p_ref[end] - p_dry[end] < 500
    # The pressure basis selects the interfaces; a moist-basis air mass already is the total mass.
    @test O.placement_pressures!(zeros(73), O.TotalFromHumidity(), DryBasis(), dry, q, area, g, A[1]) == p_total
    @test O.placement_pressures!(zeros(73), O.DryInterfaces(), DryBasis(), dry, q, area, g, A[1]) == p_dry
    moist = diff(p_ref) .* area ./ g
    @test O.placement_pressures!(zeros(73), O.TotalFromHumidity(), MoistBasis(), moist, fill(NaN, 72), area, g,
                                 A[1]) ≈ p_ref rtol = 1e-13
    # Invalid humidity is refused; a small negative value (interpolation undershoot) is used as is.
    bad = copy(q); bad[40] = 1.0
    @test_throws ArgumentError O.total_interface_pressures!(zeros(73), dry, bad, area, g, A[1])
    bad[40] = NaN
    @test_throws ArgumentError O.total_interface_pressures!(zeros(73), dry, bad, area, g, A[1])
    @test_throws DimensionMismatch O.total_interface_pressures!(zeros(73), dry, q[1:71], area, g, A[1])
    @test_throws DimensionMismatch O.total_interface_pressures!(zeros(72), dry, q, area, g, A[1])
    undershoot = copy(q); undershoot[1] = -1e-9
    @test issorted(O.total_interface_pressures!(zeros(73), dry, undershoot, area, g, A[1]))

    # The sampler-free layer rule: pressure placements against the basis's interfaces, the rest by height.
    column = O.PlacementColumn(72)
    column.humidity .= q
    z_half = layer_heights_agl!(zeros(73), p_dry, ConstantLayerTemperature(280.0), g)
    p_record = (p_dry[70] + p_ref[70]) / 2                    # between the dry and total bottom of layer 69
    pressure = O.VerticalPlacement(O.VERTICAL_PLACEMENT_CODES.pressure, NaN, p_record, NaN)
    @test O.column_layer!(column, pressure, z_half, dry, area, g, A[1], O.TotalFromHumidity(), DryBasis()) == 69
    @test column.p_half == p_total
    @test O.column_layer!(column, pressure, z_half, dry, area, g, A[1], O.DryInterfaces(), DryBasis()) == 70
    height = O.VerticalPlacement(O.VERTICAL_PLACEMENT_CODES.record_height, 100.0, NaN, NaN)
    @test O.column_layer!(column, height, z_half, dry, area, g, A[1], O.TotalFromHumidity(), DryBasis()) ==
          intake_layer_index(z_half, 100.0)
end

@testset "layer heights of an analytic hydrostatic column with humidity (OF-117)" begin
    # A column given by its interface heights above the surface, layer temperatures
    # and specific humidities (top first). Moist air of temperature T has the
    # density of dry air at T_v = T (1 + c q), c = R_v/R_d - 1, so each layer of
    # uniform T and q is exactly p_{k-1/2} = p_{k+1/2} exp(-g Δz / (R_d T_v)) on
    # total pressures, from ps = 1010 hPa; its dry air mass is (1 - q) Δp A / g.
    R, c, g, area = AtmosTransport.Parameters.R_DRY_AIR, AtmosTransport.Parameters.VIRTUAL_TEMPERATURE_FACTOR,
                    9.80665, 1e10
    z_exact = [12000.0, 8000.0, 5000.0, 3000.0, 1500.0, 800.0, 300.0, 100.0, 0.0]
    T = [220.0, 245.0, 262.0, 275.0, 285.0, 292.0, 297.0, 300.0]
    q = [1e-5, 1e-4, 1e-3, 4e-3, 8e-3, 12e-3, 15e-3, 16e-3]
    p = zeros(9)
    p[end] = 101000.0
    for k in 8:-1:1
        p[k] = p[k + 1] * exp(-g * (z_exact[k] - z_exact[k + 1]) / (R * T[k] * (1 + c * q[k])))
    end
    dry = (1 .- q) .* diff(p) .* area ./ g
    placement = O.VerticalPlacement(O.VERTICAL_PLACEMENT_CODES.record_height, 296.0, NaN, NaN)
    layer(basis; humidity = q) = begin
        column = O.PlacementColumn(8)
        column.humidity .= humidity
        p_half, z_half = zeros(9), zeros(9)
        level = O.sample_column_layer!(column, p_half, z_half, placement, dry, area, g, p[1],
                                       ProfileLayerTemperature(T), basis, DryBasis())
        (; level, p_half, z_half)
    end
    # The default basis recovers the heights from the dry air mass and the humidity.
    total = layer(O.TotalFromHumidity())
    @test total.z_half ≈ z_exact rtol = 1e-12 atol = 1e-9
    @test total.p_half ≈ p[1] .+ vcat(0.0, cumsum(diff(p) .* (1 .- q))) rtol = 1e-13   # dry interfaces
    # The dry basis keeps the old rule (dry interfaces, T): heights 1.9-2.3 % low in the lowest
    # three layers here (the dry pressure ratio and T for T_v; 293.3 m for 300 m, with 26 kg m⁻² of
    # precipitable water), and the record at 296 m lands one layer higher (6 instead of 7).
    dry_rule = layer(O.DryInterfaces(); humidity = fill(NaN, 8))
    @test dry_rule.z_half == layer_heights_agl!(zeros(9), dry_rule.p_half, ProfileLayerTemperature(T), g)
    low = 1 .- dry_rule.z_half[1:8] ./ z_exact[1:8]
    @test all(0.019 .< low[6:8] .< 0.024) && all(0.0 .< low[1:5] .< 0.019)
    @test total.level == 7 && dry_rule.level == 6
    # With q = 0 the two bases agree bit for bit (total pressures = dry ones, T_v = T).
    dry0 = (1 .- 0.0) .* diff(p) .* area ./ g
    zero_q(basis) = begin
        column = O.PlacementColumn(8)
        column.humidity .= 0.0
        z_half = zeros(9)
        O.sample_column_layer!(column, zeros(9), z_half, placement, dry0, area, g, p[1],
                               ProfileLayerTemperature(T), basis, DryBasis())
        z_half
    end
    @test zero_q(O.TotalFromHumidity()) == zero_q(O.DryInterfaces())
    # A moist-basis air mass already gives total pressures; its heights use T (no humidity is gathered).
    moist = diff(p) .* area ./ g
    moist_z = zeros(9)
    O.sample_column_layer!(O.PlacementColumn(8), zeros(9), moist_z, placement, moist, area, g, p[1],
                           ProfileLayerTemperature(T), O.TotalFromHumidity(), MoistBasis())
    @test moist_z ≈ layer_heights_agl!(zeros(9), p, ProfileLayerTemperature(T), g) rtol = 1e-12
end

@testset "virtual layer temperature (OF-117)" begin
    q = [0.0, 0.01, 0.02]
    c = AtmosTransport.Parameters.VIRTUAL_TEMPERATURE_FACTOR
    virtual = O.VirtualLayerTemperature(ProfileLayerTemperature([250.0, 270.0, 290.0]), q)
    @test O.layer_temperature(virtual, 1, 0.0) == 250.0
    @test O.layer_temperature(virtual, 3, 0.0) == 290.0 * (1 + c * 0.02)
    lapse = O.VirtualLayerTemperature(SurfaceLapseTemperature(290.0), q)
    @test O.layer_temperature(lapse, 2, 1000.0) == (290.0 - 6.5) * (1 + c * 0.01)
    @test O.layer_temperature(O.VirtualLayerTemperature(ConstantLayerTemperature(280.0), q), 2, 0.0) ==
          280.0 * (1 + c * 0.01)
    @test_throws DimensionMismatch layer_heights_agl!(zeros(5), collect(1.0:5.0), virtual, 9.80665)
    @test_throws DimensionMismatch layer_heights_agl!(zeros(4), collect(1.0:4.0),
        O.VirtualLayerTemperature(ConstantLayerTemperature(280.0), [0.0, 0.0]), 9.80665)
    # Which bases need the humidity: total pressures on a dry basis only.
    @test O.needs_column_humidity(O.TotalFromHumidity(), DryBasis())
    @test !O.needs_column_humidity(O.DryInterfaces(), DryBasis())
    @test !O.needs_column_humidity(O.TotalFromHumidity(), MoistBasis())
end
