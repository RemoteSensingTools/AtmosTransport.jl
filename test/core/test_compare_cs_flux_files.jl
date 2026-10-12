#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# test_compare_cs_flux_files.jl
#
# The cs_native flux-file comparator `scripts/validation/compare_cs_flux_files.jl`
# (surface-flux port, validation level L2). Two synthetic C4 files with known
# differences are written into a temporary directory; every reported metric is
# checked against a direct computation in this file:
#   time stamps, per-slice / month / year totals (including slices that cross a
#   month boundary and `time_bnds`), the coastal / polar / interior partition
#   (neighbours found from shared cell corners, independent of the panel
#   connectivity table), per-cell relative-difference quantiles, the 60
#   sin-latitude bands, Float32 ulp distances and the bitwise-equal fraction,
#   fill and non-finite values (including a NaN fill value), tolerance gates
#   (including checks that pass as n/a on an empty sample) and the shared
#   tolerance validation (through the function, the API and the command line),
#   the reference scale
#   and its preset, variables without a time dimension, the units rule (series
#   must declare units), series present in only one file, bitwise equality only
#   within one stored type, the quantiles over all cells selected without a
#   merged copy, layout errors (swapped X/Y, swapped panels, time bounds that
#   are centre-stamped, gapped or overlapping, one-sided `lats`/`lons`), time
#   slices matched as (start, end) intervals, a coastal mask read from another
#   file, the JSON report, the mesh from a transport binary, and the exit status
#   of the command line.
# The script is included into a module, as in `test_ll_to_cs_regrid_script.jl`.
# ---------------------------------------------------------------------------

using Test
using Dates
using JSON3
using NCDatasets
using Statistics: quantile
using AtmosTransport
using AtmosTransport.Grids: CubedSphereMesh, GEOSNativePanelConvention,
                            panel_cell_center_lonlat, panel_cell_corner_lonlat
using AtmosTransport.Parameters: EARTH_RADIUS, IFS_EARTH_RADIUS, SPECIES_MOLAR_MASS, CARBON_MOLAR_MASS
const MD = AtmosTransport.MetDrivers

const COMPARATOR = joinpath(@__DIR__, "..", "..", "scripts", "validation", "compare_cs_flux_files.jl")
const CMP = Module()
Base.include(CMP, COMPARATOR)
cmp_call(f::Symbol, args...; kw...) = Base.invokelatest(getproperty(CMP, f), args...; kw...)

# Message of the exception `f()` throws ("no error" when it returns).
function error_message(f)
    try
        f()
        return "no error"
    catch err
        return sprint(showerror, err)
    end
end

# The C -> CO2 factor of the Python flux files written before commit 71a08395:
# a legacy encoding the reference-scale preset undoes, not a physical constant.
const LEGACY_CO2_PER_CARBON = 44 / 12

const NC = 4
const MESH = CubedSphereMesh(; Nc = NC, FT = Float64, Hp = 0,
                             convention = GEOSNativePanelConvention(), radius = EARTH_RADIUS)
const AREA = Float64.(MESH.cell_areas)

# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

# `fields` are name => (Xdim, Ydim, nf, time) arrays stored as `T` with
# dimensions `field_dims`; time is encoded as `unit since origin` raw numbers.
# `extra` holds name => (array, dims) pairs. `units = nothing` writes no units
# attribute; `bounds_variable = false` declares `time_bnds` without writing it.
function write_flux_file(path, times::Vector{DateTime}, fields;
                         origin = DateTime(2021, 12, 1), unit = Hour(1), unit_name = "hours",
                         origin_suffix = "", bounds = nothing, bounds_variable = true,
                         extra = Pair{String, Any}[], fillvalue = nothing, units = "kg CO2 m-2 s-1",
                         T = Float32, field_dims = ("Xdim", "Ydim", "nf", "time"))
    NCDataset(path, "c") do ds
        defDim(ds, "Xdim", NC); defDim(ds, "Ydim", NC); defDim(ds, "nf", 6)
        defDim(ds, "time", length(times))
        attrib = Dict{String, Any}("units" => "$(unit_name) since $(Dates.format(origin, "yyyy-mm-dd HH:MM:SS"))$(origin_suffix)",
                                   "calendar" => "proleptic_gregorian")
        bounds === nothing || (attrib["bounds"] = "time_bnds")
        encode(t) = Millisecond(t - origin) / Millisecond(unit)
        defVar(ds, "time", Float64, ("time",); attrib)[:] = encode.(times)
        if bounds !== nothing && bounds_variable
            defDim(ds, "nv", 2)
            defVar(ds, "time_bnds", Float64, ("nv", "time"))[:, :] = encode.(bounds)
        end
        for (name, data) in fields
            kw = fillvalue === nothing ? (;) : (; fillvalue = T(fillvalue))
            v = defVar(ds, name, T, field_dims;
                       attrib = units === nothing ? Dict{String, Any}() : Dict("units" => units), kw...)
            v.var[:, :, :, :] = data
        end
        for (name, (data, dims)) in extra
            defVar(ds, name, eltype(data), dims)[ntuple(_ -> Colon(), ndims(data))...] = data
        end
    end
    return path
end

# A file of (Xdim, Ydim, nf) variables only, without a time axis: name =>
# (array, dims) pairs; dimensions other than Xdim, Ydim, nf are made as needed.
function write_static_file(path, fields)
    NCDataset(path, "c") do ds
        sizes = Dict("Xdim" => NC, "Ydim" => NC, "nf" => 6)
        for (name, (data, dims)) in fields
            for (d, n) in zip(dims, size(data))
                haskey(ds.dim, d) || defDim(ds, d, get(sizes, d, n))
            end
            defVar(ds, name, eltype(data), dims)[ntuple(_ -> Colon(), ndims(data))...] = data
        end
    end
    return path
end

# Mesh cell centres in degrees as (Xdim, Ydim, nf) arrays, the `lons`/`lats` of a
# file; `panels` reorders the panels (to make centres of another convention).
function centre_fields(mesh; panels = 1:6)
    lon = zeros(NC, NC, 6); lat = zeros(NC, NC, 6)
    for (q, p) in enumerate(panels)
        lon[:, :, q], lat[:, :, q] = panel_cell_center_lonlat(mesh, p)
    end
    return lon, lat
end

# Land: panel 1 columns i <= 2 and a 2x2 island on the north polar panel.
function land_mask()
    m = zeros(Float32, NC, NC, 6)
    m[1:2, :, 1] .= 1
    m[2:3, 2:3, 3] .= 1
    return m
end

# Smooth reference field: land 1e-8, ocean 1e-10 kg m-2 s-1, varying in i, j, p, k.
function reference_field(nt, mask)
    A = Array{Float32}(undef, NC, NC, 6, nt)
    for k in 1:nt, p in 1:6, j in 1:NC, i in 1:NC
        base = mask[i, j, p] > 0 ? 1e-8 : 1e-10
        A[i, j, p, k] = Float32(base * (1 + 0.1i + 0.01j + 0.2p) * (1 + 0.05k))
    end
    return A
end

# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------

unit_xyz(lon, lat) = (cosd(lat) * cosd(lon), cosd(lat) * sind(lon), sind(lat))

# Cells sharing two corners are edge neighbours (found from the geometry alone).
function neighbour_pairs(mesh)
    Nc = mesh.Nc
    corners = Dict{NTuple{3, Int}, Vector{NTuple{3, Float64}}}()
    for p in 1:6
        lon, lat = panel_cell_corner_lonlat(mesh, p, Float64)
        for j in 1:Nc, i in 1:Nc
            corners[(i, j, p)] = [unit_xyz(lon[a, b], lat[a, b]) for (a, b) in
                                  ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]
        end
    end
    cells = collect(keys(corners))
    shared(c1, c2) = count(x -> any(y -> maximum(abs.(x .- y)) < 1e-9, corners[c2]), corners[c1])
    return [(c1, c2) for (n, c1) in enumerate(cells) for c2 in cells[n + 1:end] if shared(c1, c2) == 2]
end

function expected_categories(mask, threshold)
    cat = Dict{NTuple{3, Int}, Symbol}()
    for p in 1:6, j in 1:NC, i in 1:NC
        cat[(i, j, p)] = p in (3, 6) ? :polar : :interior
    end
    for (c1, c2) in neighbour_pairs(MESH)
        if (mask[c1...] > threshold) != (mask[c2...] > threshold)
            cat[c1] = :coastal
            cat[c2] = :coastal
        end
    end
    return cat
end

rel_cell(a, b) = (m = max(abs(Float64(a)), abs(Float64(b)));
                  m == 0 ? nothing : abs(Float64(b) - Float64(a)) / m)

# Float32 ulp distance from the bit patterns of the magnitudes (+0 == -0).
function ulps(a::Float32, b::Float32)
    a == b && return 0
    ia, ib = Int64(reinterpret(UInt32, abs(a))), Int64(reinterpret(UInt32, abs(b)))
    return signbit(a) == signbit(b) ? abs(ia - ib) : ia + ib
end

const BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 16), (17, 256), (257, typemax(Int64)))

band_of(lat) = clamp(floor(Int, (sind(lat) + 1) * 30) + 1, 1, 60)

slice_total(X, k) = sum(Float64(X[i, j, p, k]) * AREA[i, j] for p in 1:6, j in 1:NC, i in 1:NC)

@testset "compare_cs_flux_files.jl" begin
    mktempdir() do dir
        times = [DateTime(2021, 12, 30), DateTime(2021, 12, 31), DateTime(2022, 1, 1), DateTime(2022, 1, 2)]
        nt = length(times)
        mask = land_mask()
        A = reference_field(nt, mask)
        B = copy(A)
        # Known differences (cell (i, j, panel), slice k).
        C1, C2, C3 = (3, 3, 2), (2, 3, 2), (1, 1, 4)        # k = 2: 1 ulp, 3 ulps, +0 vs -0
        C4, C5, C6 = (2, 2, 2), (2, 1, 1), (2, 2, 6)        # k = 3: interior, coastal, polar
        Z1 = (2, 2, 4)                                     # k = 4: zero only in the reference
        B[C1..., 2] = nextfloat(A[C1..., 2])
        B[C2..., 2] = nextfloat(A[C2..., 2], 3)
        A[C3..., 2] = 0.0f0; B[C3..., 2] = -0.0f0
        B[C4..., 3] = Float32(A[C4..., 3] * 1.001)
        B[C5..., 3] = Float32(A[C5..., 3] * 1.1)
        B[C6..., 3] = Float32(A[C6..., 3] * (1 + 1e-4))
        A[Z1..., 4] = 0.0f0; B[Z1..., 4] = 1.0f-9
        other = fill(1.0f-9, NC, NC, 6, nt)

        area_file = repeat(AREA, 1, 1, 6)
        area_file[3, 2, 5] *= 1 + 1e-5
        ref = write_flux_file(joinpath(dir, "ref.nc"), times, ["FLUX" => A];
                              origin = DateTime(2021, 12, 30), origin_suffix = " UTC",
                              extra = ["land_fraction" => (mask, ("Xdim", "Ydim", "nf")),
                                       "cell_area" => (area_file, ("Xdim", "Ydim", "nf"))])
        # Same stamps, another encoding: days since another origin.
        cand = write_flux_file(joinpath(dir, "cand.nc"), times, ["FLUX" => B];
                               origin = DateTime(2021, 12, 1), unit = Day(1), unit_name = "days")
        # Each with one more series, under different names.
        ref_other = write_flux_file(joinpath(dir, "ref_other.nc"), times, ["FLUX" => A, "OTHER" => other])
        cand_other = write_flux_file(joinpath(dir, "cand_other.nc"), times, ["FLUX" => B, "OTHER_B" => other])

        cats = expected_categories(mask, 0.0)
        @testset "categories from shared corners" begin
            @test length(neighbour_pairs(MESH)) == 12 * NC^2
            codes, polar = cmp_call(:cell_categories, MESH; mask = Float64.(mask))
            names = Dict(UInt8(1) => :interior, UInt8(2) => :coastal, UInt8(3) => :polar)
            @test polar == [3, 6]
            @test all(names[codes[c...]] == cats[c] for c in keys(cats))
            @test cats[C5] == :coastal && cats[C4] == :interior && cats[C6] == :polar && cats[Z1] == :interior
            # Cross-panel coast: panel 1 west column (land) touches panel 5 (ocean).
            @test count(c -> c[3] == 5 && cats[c] == :coastal, keys(cats)) == NC
            # The land threshold decides the class of a partial cell.
            partial = Float64.(mask); partial[4, 4, 2] = 0.3
            for threshold in (0.0, 0.5)
                codes, _ = cmp_call(:cell_categories, MESH; mask = partial, land_threshold = threshold)
                expected = expected_categories(partial, threshold)
                @test all(names[codes[c...]] == expected[c] for c in keys(expected))
            end
            nomask, _ = cmp_call(:cell_categories, MESH)
            @test count(==(UInt8(2)), nomask) == 0 && count(==(UInt8(3)), nomask) == 2 * NC^2
        end

        tolerances = ["slice_total" => 0.1, "year_total" => 1.0, "cell_max.interior" => 0.5,
                      "cell_p99.polar" => 1e-3, "bitwise_fraction" => 0.98, "ulp_max.polar" => 1e4]
        report = cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "land_fraction",
                          tolerances)

        @testset "time axes and variables" begin
            @test report.time.identical && report.time.pass
            @test report.time.n_compared == nt
            @test report.time.first == "2021-12-30T00:00:00" && report.time.last == "2022-01-02T00:00:00"
            @test report.reference.time_units == "hours since 2021-12-30 00:00:00 UTC"
            @test report.candidate.time_units == "days since 2021-12-01 00:00:00"
            @test occursin("one spacing", report.reference.slice_ends)
            @test length(report.variables) == 1
            v = only(report.variables)
            @test v.reference_variable == v.candidate_variable == "FLUX"
            sel = report.variable_selection
            @test sel.pass && !sel.common_variables_only && occursin("either file", sel.rule)
            @test isempty(sel.series_only_in_reference) && isempty(sel.series_only_in_candidate)
            @test v.units.equal && v.stored_types.reference == "Float32" && v.stored_types.equal
            @test report.mesh.Nc == NC && report.mesh.radius_m == EARTH_RADIUS
            @test report.mesh.total_area_m2 ≈ 4π * EARTH_RADIUS^2 rtol = 1e-12
            @test report.reference.cell_area_vs_mesh.max_rel_diff ≈ 1e-5 rtol = 1e-6
            @test report.candidate.cell_area_vs_mesh === nothing
            @test report.categories.n_cells.coastal == count(==(:coastal), values(cats))
            @test report.categories.n_cells.polar == count(==(:polar), values(cats))
            @test report.categories.n_cells.interior == count(==(:interior), values(cats))
        end

        v = only(report.variables)
        @testset "global totals per slice, month and year" begin
            ta = [slice_total(A, k) for k in 1:nt]
            tb = [slice_total(B, k) for k in 1:nt]
            @test v.slice_totals.reference ≈ ta rtol = 1e-13
            @test v.slice_totals.candidate ≈ tb rtol = 1e-13
            rel = (tb .- ta) ./ abs.(ta)
            @test v.slice_totals.rel_diff ≈ rel rtol = 1e-9
            @test v.slice_totals.max_abs_rel_diff ≈ maximum(abs, rel) rtol = 1e-9
            @test v.slice_totals.worst_time == Dates.format(times[argmax(abs.(rel))], "yyyy-mm-ddTHH:MM:SS")
            day = 86400.0
            months = v.month_totals.months
            @test [m.month for m in months] == ["2021-12", "2022-01"]
            @test months[1].reference ≈ (ta[1] + ta[2]) * day rtol = 1e-13
            @test months[2].candidate ≈ (tb[3] + tb[4]) * day rtol = 1e-13
            @test !months[1].complete && !months[2].complete
            @test months[2].rel_diff ≈ (tb[3] + tb[4] - ta[3] - ta[4]) / (ta[3] + ta[4]) rtol = 1e-9
            years = v.year_totals.years
            @test [y.year for y in years] == [2021, 2022]
            @test years[1].reference ≈ months[1].reference rtol = 1e-15
            @test years[2].candidate ≈ months[2].candidate rtol = 1e-15
            @test !years[1].complete
        end

        @testset "per-cell relative differences by category" begin
            samples = Dict(c => Float64[] for c in (:all, :interior, :coastal, :polar))
            both_zero = one_zero = 0
            for k in 1:nt, p in 1:6, j in 1:NC, i in 1:NC
                a, b = A[i, j, p, k], B[i, j, p, k]
                r = rel_cell(a, b)
                if r === nothing
                    both_zero += 1
                    continue
                end
                (a == 0) != (b == 0) && (one_zero += 1)
                push!(samples[:all], r)
                push!(samples[cats[(i, j, p)]], r)
            end
            @test v.cells.all.n_both_zero == both_zero == 1
            @test v.cells.all.n_one_zero == one_zero == 1
            for c in (:all, :interior, :coastal, :polar)
                s = getproperty(v.cells, c)
                @test s.n_sample == length(samples[c])
                @test s.p50 ≈ quantile(samples[c], 0.5) rtol = 1e-6 atol = 1e-30
                @test s.p99 ≈ quantile(samples[c], 0.99) rtol = 1e-6
                @test s.max == maximum(samples[c])
            end
            @test v.cells.all.max == 1.0
            @test v.cells.all.worst.panel == Z1[3] && (v.cells.all.worst.i, v.cells.all.worst.j) == Z1[1:2]
            @test v.cells.all.worst.time == "2022-01-02T00:00:00"
            @test v.cells.all.worst.reference == 0 && v.cells.all.worst.candidate == Float64(1.0f-9)
            @test v.cells.coastal.max ≈ rel_cell(A[C5..., 3], B[C5..., 3]) rtol = 1e-15
            @test v.cells.polar.max ≈ rel_cell(A[C6..., 3], B[C6..., 3]) rtol = 1e-15
            @test 0.0909 < v.cells.coastal.max < 0.0910 && 9e-5 < v.cells.polar.max < 1.1e-4
        end

        @testset "Float32 ulp distance and bitwise equality" begin
            @test ulps(A[C1..., 2], B[C1..., 2]) == 1
            @test ulps(A[C2..., 2], B[C2..., 2]) == 3
            @test ulps(A[C3..., 2], B[C3..., 2]) == 0
            for c in (:all, :interior, :coastal, :polar)
                d = Int64[]; bitwise = 0
                for k in 1:nt, p in 1:6, j in 1:NC, i in 1:NC
                    (c == :all || cats[(i, j, p)] == c) || continue
                    a, b = A[i, j, p, k], B[i, j, p, k]
                    push!(d, ulps(a, b))
                    bitwise += reinterpret(UInt32, a) == reinterpret(UInt32, b)
                end
                s = getproperty(v.cells, c)
                @test s.n_values == length(d)
                @test s.ulp_max == maximum(d)
                @test s.bitwise_equal_fraction == bitwise / length(d)
                @test s.within_1ulp_fraction == count(<=(1), d) / length(d)
                @test [h.count for h in s.ulp_histogram] == [count(x -> lo <= x <= hi, d) for (lo, hi) in BUCKETS]
            end
            @test v.cells.all.bitwise_equal_fraction == (6 * NC^2 * nt - 7) / (6 * NC^2 * nt)
            @test [h.ulps for h in v.cells.all.ulp_histogram] == ["0", "1", "2", "3-4", "5-16", "17-256", ">=257"]
            @test v.cells.all.ulp_histogram[2].count == 1 && v.cells.all.ulp_histogram[4].count == 1
        end

        @testset "sin-latitude bands" begin
            band = Array{Int}(undef, NC, NC, 6)
            for p in 1:6
                _, lat = panel_cell_center_lonlat(MESH, p)
                band[:, :, p] .= band_of.(lat)
            end
            @test v.bands.n_bands == 60
            @test v.bands.n_cells == [count(==(q), band) for q in 1:60]
            @test sum(v.bands.n_cells) == 6 * NC^2
            sums(X, k) = [sum((Float64(X[i, j, p, k]) * AREA[i, j] for p in 1:6, j in 1:NC, i in 1:NC
                               if band[i, j, p] == q); init = 0.0) for q in 1:60]
            mean_a = sum(sums(A, k) for k in 1:nt) ./ nt     # equal (daily) slice weights
            mean_b = sum(sums(B, k) for k in 1:nt) ./ nt
            @test v.bands.reference ≈ mean_a rtol = 1e-12
            @test v.bands.candidate ≈ mean_b rtol = 1e-12
            @test v.bands.max_abs_diff_over_max_band ≈ maximum(abs.(mean_b .- mean_a)) / maximum(abs, mean_a) rtol = 1e-6
            slice_max = [maximum(k -> (a = sums(A, k)[q]; b = sums(B, k)[q];
                                       a == b ? 0.0 : abs(b - a) / abs(a)), 1:nt) for q in 1:60]
            @test v.bands.max_slice_abs_rel_diff ≈ slice_max rtol = 1e-6 atol = 1e-15
        end

        @testset "tolerance checks" begin
            checks = Dict(c.metric => c for c in v.checks)
            @test [c.metric for c in v.checks] == ["finite", "slice_total", "year_total", "cell_max.interior",
                                                   "cell_p99.polar", "bitwise_fraction.all", "ulp_max.polar"]
            @test checks["finite"].pass && checks["finite"].value == 0
            @test checks["slice_total"].value == v.slice_totals.max_abs_rel_diff < 0.1 && checks["slice_total"].pass
            @test checks["year_total"].value == v.year_totals.max_abs_rel_diff && checks["year_total"].pass
            # The one-zero interior cell (relative difference 1) fails cell_max.interior <= 0.5.
            @test checks["cell_max.interior"].value == 1.0 && !checks["cell_max.interior"].pass &&
                  checks["cell_max.interior"].bound == "<="
            @test checks["cell_p99.polar"].value == v.cells.polar.p99 < 1e-3 && checks["cell_p99.polar"].pass
            @test checks["bitwise_fraction.all"].value == v.cells.all.bitwise_equal_fraction > 0.98
            @test checks["bitwise_fraction.all"].pass && checks["bitwise_fraction.all"].bound == ">="
            @test checks["ulp_max.polar"].value == v.cells.polar.ulp_max == ulps(A[C6..., 3], B[C6..., 3])
            @test checks["ulp_max.polar"].pass
            @test !v.pass && !report.pass
            strict = cmp_call(:compare_cs_flux_files, ref, cand, MESH;
                              tolerances = ["cell_max.coastal" => 0.05, "bitwise_fraction" => 0.999],
                              mask_var = "land_fraction")
            sv = only(strict.variables)
            @test !strict.pass && !sv.pass
            @test [c.pass for c in sv.checks] == [true, false, false]
            relaxed = cmp_call(:compare_cs_flux_files, ref, cand, MESH;
                               tolerances = ["cell_max" => 1.0, "bitwise_fraction" => 0.98, "slice_total" => 0.1])
            @test relaxed.pass
            @test relaxed.categories.n_cells.coastal == 0
        end

        @testset "tolerance validation: function, API and command line" begin
            validate(k, x) = cmp_call(:validate_tolerance, k, x)
            @test validate("cell_p99", 1e-6) == ("cell_p99.all" => 1e-6)
            @test validate(" ulp_max.polar ", 2) == ("ulp_max.polar" => 2.0)
            @test validate("bitwise_fraction", 1) == ("bitwise_fraction.all" => 1.0)
            @test validate("bitwise_fraction", 0) == ("bitwise_fraction.all" => 0.0)
            @test validate("slice_total", 0.0) == ("slice_total" => 0.0) && validate("year_total", 1 // 10).second == 0.1
            bad = [("slice_total", Inf), ("year_total", -Inf), ("cell_max", NaN), ("month_total", -1e-12),
                   ("cell_p99.interior", -1.0), ("bitwise_fraction", -1), ("bitwise_fraction", 1.5),
                   ("bitwise_fraction", Inf), ("ulp_max", 0.5), ("ulp_max", -1), ("ulp_max", Inf),
                   ("band", true), ("cell_max", "1"), ("cell_max", nothing), ("nonsense", 1.0)]
            for (k, x) in bad
                @test_throws ArgumentError validate(k, x)
            end
            @test occursin("finite and non-negative", error_message(() -> validate("slice_total", Inf)))
            @test occursin("must lie in [0, 1]", error_message(() -> validate("bitwise_fraction", 1.5)))
            @test occursin("whole number", error_message(() -> validate("ulp_max", 0.5)))
            @test_throws ArgumentError cmp_call(:validate_tolerances, ["cell_max" => 1.0, "cell_max.all" => 2.0])
            # The API: an infinite or negative tolerance would pass its gate whatever the discrepancy.
            for tol in (["slice_total" => Inf], ["bitwise_fraction" => -1.0], ["bitwise_fraction" => 2.0],
                        ["ulp_max" => 1.5], ["cell_max" => -0.5], ["cell_max" => 1.0, "cell_max.all" => 2.0])
                @test_throws ArgumentError cmp_call(:compare_cs_flux_files, ref, cand, MESH; tolerances = tol)
            end
            # The command line.
            parse = getproperty(CMP, :_parse_args)
            base = ["--reference", ref, "--candidate", cand, "--nc", "4"]
            for spec in ("slice_total=Inf", "slice_total=inf", "year_total=-1e-9", "bitwise_fraction=-1",
                         "bitwise_fraction=1.000001", "ulp_max=2.5", "cell_p99.polar=-0.1", "cell_max=NaN")
                @test_throws ArgumentError Base.invokelatest(parse, vcat(base, "--tol", spec))
            end
            @test occursin("(--tol)", error_message(() -> Base.invokelatest(parse, vcat(base, "--tol", "slice_total=Inf"))))
            opts = Base.invokelatest(parse, vcat(base, "--tol", "ulp_max=3", "--tol", "bitwise_fraction=0.5"))
            @test opts.tolerances == ["ulp_max.all" => 3.0, "bitwise_fraction.all" => 0.5]
        end

        @testset "named variable pairs" begin
            r = cmp_call(:compare_cs_flux_files, ref_other, cand_other, MESH; variables = ["OTHER" => "OTHER_B"])
            w = only(r.variables)
            @test (w.reference_variable, w.candidate_variable) == ("OTHER", "OTHER_B")
            @test w.cells.all.bitwise_equal_fraction == 1.0 && w.cells.all.max == 0.0
            @test w.slice_totals.max_abs_rel_diff == 0.0 && r.pass
            @test r.variable_selection.rule == "named variables" && r.variable_selection.pass
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, ref, cand, MESH; variables = ["MISSING"])
        end

        @testset "series in only one file" begin
            # Each file has a series the other lacks: compared are the common ones, and the comparison fails.
            r = cmp_call(:compare_cs_flux_files, ref_other, cand_other, MESH)
            @test [w.reference_variable for w in r.variables] == ["FLUX"]
            @test r.variable_selection.series_only_in_reference == ["OTHER"]
            @test r.variable_selection.series_only_in_candidate == ["OTHER_B"]
            @test !r.variable_selection.pass && !r.pass && only(r.variables).pass
            @test occursin("series in one file only: reference [OTHER], candidate [OTHER_B]  FAIL",
                           cmp_call(:summary_text, r))
            # A candidate that drops a series (TER) while keeping GPP fails.
            two = write_flux_file(joinpath(dir, "gpp_ter.nc"), times, ["GPP" => A, "TER" => A])
            gpp = write_flux_file(joinpath(dir, "gpp.nc"), times, ["GPP" => A])
            r = cmp_call(:compare_cs_flux_files, two, gpp, MESH; tolerances = ["bitwise_fraction" => 1.0])
            @test !r.pass && only(r.variables).pass && r.variable_selection.series_only_in_reference == ["TER"]
            # Unless only the common series are asked for.
            r = cmp_call(:compare_cs_flux_files, two, gpp, MESH; common_variables_only = true,
                         tolerances = ["bitwise_fraction" => 1.0])
            @test r.pass && r.variable_selection.pass && r.variable_selection.common_variables_only
            @test r.variable_selection.series_only_in_reference == ["TER"]
            @test occursin("(not compared: common variables only)", cmp_call(:summary_text, r))
            # The option goes with the default selection only.
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, two, gpp, MESH; variables = ["GPP"],
                                                common_variables_only = true)
            # Named variables are compared without looking at the other series.
            @test cmp_call(:compare_cs_flux_files, two, gpp, MESH; variables = ["GPP"]).pass
        end

        @testset "time stamps that differ" begin
            shifted = copy(times); shifted[end] = DateTime(2022, 1, 3)
            other_time = write_flux_file(joinpath(dir, "shifted.nc"), shifted, ["FLUX" => B])
            r = cmp_call(:compare_cs_flux_files, ref, other_time, MESH)
            @test !r.time.identical && !r.time.pass && !r.pass
            @test r.time.only_in_reference == ["2022-01-02T00:00:00"]
            @test r.time.only_in_candidate == ["2022-01-03T00:00:00"]
            # The 2022-01-01 slice lasts one day in the reference and two in the candidate: not matched.
            @test r.time.n_compared == nt - 2 && r.time.n_end_differs == 1
            @test only(r.time.end_differs) == (; start = "2022-01-01T00:00:00", reference_end = "2022-01-02T00:00:00",
                                                 candidate_end = "2022-01-03T00:00:00")
            @test length(only(r.variables).slice_totals.reference) == nt - 2
            r = cmp_call(:compare_cs_flux_files, ref, other_time, MESH; common_times = true)
            @test r.time.pass && r.pass
            # A short build against a long one: common stamps only.
            short = write_flux_file(joinpath(dir, "short.nc"), times[2:3], ["FLUX" => A[:, :, :, 2:3]])
            r = cmp_call(:compare_cs_flux_files, ref, short, MESH; common_times = true)
            @test r.pass && r.time.n_compared == 2 && r.time.n_only_in_reference == 2
            @test only(r.variables).slice_totals.reference ≈ [slice_total(A, 2), slice_total(A, 3)] rtol = 1e-13
            # The short file's last slice ends one spacing later, as the long file's.
            @test only(r.variables).month_totals.max_abs_rel_diff == 0.0
        end

        @testset "slice ends: month-start stamps, month splits and time bounds" begin
            monthly = [DateTime(2021, m, 1) for m in 1:12]
            X = reference_field(12, mask)
            ma = write_flux_file(joinpath(dir, "monthly_a.nc"), monthly, ["FLUX" => X];
                                 origin = DateTime(2021, 1, 1))
            mb = write_flux_file(joinpath(dir, "monthly_b.nc"), monthly, ["FLUX" => 2 .* X];
                                 origin = DateTime(2021, 1, 1))
            r = cmp_call(:compare_cs_flux_files, ma, mb, MESH)
            w = only(r.variables)
            @test occursin("month", r.reference.slice_ends)
            @test all(m -> m.complete, w.month_totals.months)
            @test only(w.year_totals.years).complete
            @test [m.reference for m in w.month_totals.months] ≈
                  [slice_total(X, m) * Dates.daysinmonth(2021, m) * 86400 for m in 1:12] rtol = 1e-13
            @test only(w.year_totals.years).reference ≈
                  sum(slice_total(X, m) * Dates.daysinmonth(2021, m) * 86400 for m in 1:12) rtol = 1e-13
            @test all(==(1.0), w.slice_totals.rel_diff) && w.year_totals.max_abs_rel_diff == 1.0
            @test w.cells.all.max == 0.5 && w.cells.all.p50 == 0.5      # |2a - a| / 2a
            @test w.cells.all.ulp_max == 2^23                             # one binade
            @test w.cells.all.bitwise_equal_fraction == 0.0
            @test w.bands.max_abs_diff_over_max_band ≈ 1.0 rtol = 1e-14

            # Noon stamps: the first slice is split between March and April.
            noon = [DateTime(2021, 3, 31, 12), DateTime(2021, 4, 1, 12)]
            Y = reference_field(2, mask)
            na = write_flux_file(joinpath(dir, "noon_a.nc"), noon, ["FLUX" => Y]; origin = DateTime(2021, 3, 1))
            w = only(cmp_call(:compare_cs_flux_files, na, na, MESH).variables)
            months = w.month_totals.months
            @test [m.month for m in months] == ["2021-03", "2021-04"]
            @test months[1].reference ≈ slice_total(Y, 1) * 43200 rtol = 1e-13
            @test months[2].reference ≈ slice_total(Y, 1) * 43200 + slice_total(Y, 2) * 86400 rtol = 1e-13
            # With time bounds the last slice is 6 hours long.
            bnds = [DateTime(2021, 3, 31, 12) DateTime(2021, 4, 1, 12);
                    DateTime(2021, 4, 1, 12) DateTime(2021, 4, 1, 18)]
            nb = write_flux_file(joinpath(dir, "noon_b.nc"), noon, ["FLUX" => Y]; origin = DateTime(2021, 3, 1),
                                 bounds = bnds)
            r = cmp_call(:compare_cs_flux_files, nb, nb, MESH)
            @test occursin("time bounds", r.reference.slice_ends)
            months = only(r.variables).month_totals.months
            @test months[2].reference ≈ slice_total(Y, 1) * 43200 + slice_total(Y, 2) * 21600 rtol = 1e-13
            # Band means are weighted by the slice durations (24 h and 6 h).
            band_w = only(r.variables).bands.reference
            band_u = only(cmp_call(:compare_cs_flux_files, na, na, MESH).variables).bands.reference
            @test sum(band_w) ≈ (slice_total(Y, 1) * 24 + slice_total(Y, 2) * 6) / 30 rtol = 1e-12
            @test sum(band_u) ≈ (slice_total(Y, 1) + slice_total(Y, 2)) / 2 rtol = 1e-12
            # Bounds must describe start-stamped, contiguous slices.
            bad_bounds = (
                centre = [DateTime(2021, 3, 31) DateTime(2021, 4, 1);              # noon stamps in day intervals
                          DateTime(2021, 4, 1) DateTime(2021, 4, 2)],
                gap = [DateTime(2021, 3, 31, 12) DateTime(2021, 4, 1, 12);
                       DateTime(2021, 3, 31, 18) DateTime(2021, 4, 1, 18)],
                overlap = [DateTime(2021, 3, 31, 12) DateTime(2021, 4, 1, 12);
                           DateTime(2021, 4, 1, 18) DateTime(2021, 4, 2, 12)],
                empty = [DateTime(2021, 3, 31, 12) DateTime(2021, 4, 1, 12);
                         DateTime(2021, 4, 1, 12) DateTime(2021, 4, 1, 12)])
            messages = Dict{Symbol, String}()
            for (label, b) in pairs(bad_bounds)
                f = write_flux_file(joinpath(dir, "bounds_$(label).nc"), noon, ["FLUX" => Y];
                                    origin = DateTime(2021, 3, 1), bounds = b)
                messages[label] = try
                    cmp_call(:compare_cs_flux_files, f, f, MESH)
                    "no error"
                catch err
                    err isa ArgumentError ? sprint(showerror, err) : "unexpected $(typeof(err))"
                end
            end
            @test occursin("do not start at the stamps (slice 1", messages[:centre])
            @test occursin("leave a gap between slices 1 and 2", messages[:gap])
            @test occursin("overlap between slices 1 and 2", messages[:overlap])
            @test occursin("do not end after their start (slice 2", messages[:empty])
            # Declared bounds that are not in the file.
            undeclared = write_flux_file(joinpath(dir, "bounds_missing.nc"), noon, ["FLUX" => Y];
                                         origin = DateTime(2021, 3, 1), bounds = bnds, bounds_variable = false)
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, undeclared, undeclared, MESH)
            # A single slice without bounds has no duration: no month or year totals.
            single = write_flux_file(joinpath(dir, "single.nc"), times[1:1], ["FLUX" => A[:, :, :, 1:1]])
            r = cmp_call(:compare_cs_flux_files, single, single, MESH; tolerances = ["month_total" => 1.0])
            w = only(r.variables)
            @test w.month_totals.months === nothing && w.month_totals.max_abs_rel_diff === nothing
            @test !r.pass && !w.checks[end].pass                         # a requested metric that is unavailable fails
            @test r.time.identical                                       # an unknown end matches an unknown end

            # Slices are matched as (start, end) intervals: equal stamps with unequal final bounds differ.
            bnds12 = [bnds[1, 1] bnds[1, 2]; bnds[2, 1] DateTime(2021, 4, 2)]
            nb12 = write_flux_file(joinpath(dir, "noon_b12.nc"), noon, ["FLUX" => Y]; origin = DateTime(2021, 3, 1),
                                   bounds = bnds12)
            r = cmp_call(:compare_cs_flux_files, nb, nb12, MESH; tolerances = ["bitwise_fraction" => 1.0])
            @test !r.time.identical && !r.time.pass && !r.pass && only(r.variables).pass   # bitwise, yet different mass
            @test r.time.n_end_differs == 1 && r.time.n_only_in_reference == r.time.n_only_in_candidate == 0
            @test only(r.time.end_differs) == (; start = "2021-04-01T12:00:00", reference_end = "2021-04-01T18:00:00",
                                                 candidate_end = "2021-04-02T00:00:00")
            @test r.time.n_compared == 1 && length(only(r.variables).slice_totals.reference) == 1
            @test occursin("1 with other slice ends", cmp_call(:summary_text, r))
            # With --common-times only the matched slice is compared.
            r = cmp_call(:compare_cs_flux_files, nb, nb12, MESH; common_times = true, tolerances = ["bitwise_fraction" => 1.0])
            @test r.time.pass && r.pass && r.time.n_compared == 1 && r.time.first == "2021-03-31T12:00:00"
            # One file with bounds, the other without: the last slice ends 24 h (one spacing) vs 6 h after its stamp.
            r = cmp_call(:compare_cs_flux_files, na, nb, MESH; tolerances = ["bitwise_fraction" => 1.0])
            @test !r.time.identical && !r.time.pass && !r.pass && r.time.n_end_differs == 1
            @test only(r.time.end_differs).reference_end == "2021-04-02T12:00:00"
            @test only(r.time.end_differs).candidate_end == "2021-04-01T18:00:00"
            # Bounds equal to the inferred ends describe the same slices.
            bnds24 = [bnds[1, 1] bnds[1, 2]; bnds[2, 1] DateTime(2021, 4, 2, 12)]
            nb24 = write_flux_file(joinpath(dir, "noon_b24.nc"), noon, ["FLUX" => Y]; origin = DateTime(2021, 3, 1),
                                   bounds = bnds24)
            r = cmp_call(:compare_cs_flux_files, na, nb24, MESH; tolerances = ["bitwise_fraction" => 1.0])
            @test r.time.identical && r.time.n_end_differs == 0 && r.pass
            # An unknown end (one slice without bounds) matches no known end.
            r = cmp_call(:compare_cs_flux_files, single, ref, MESH; common_times = true)
            @test r.time.n_end_differs == 1 && r.time.n_compared == 0 && !r.time.pass && !r.pass
            @test only(r.time.end_differs).reference_end == "unknown"
        end

        @testset "fill and non-finite values fail" begin
            F = copy(B); F[1, 1, 1, 1] = -999.0f0; F[2, 1, 1, 2] = NaN32
            filled = write_flux_file(joinpath(dir, "filled.nc"), times, ["FLUX" => F]; fillvalue = -999.0f0)
            r = cmp_call(:compare_cs_flux_files, ref, filled, MESH)
            w = only(r.variables)
            @test w.invalid.fill_values_candidate == 1 && w.invalid.fill_values_reference == 0
            @test w.invalid.nonfinite_pairs == 1
            @test !w.checks[1].pass && w.checks[1].value == 2 && !r.pass
            @test isnan(w.slice_totals.candidate[2])
            # A NaN fill value matches every NaN (as NCDatasets decides), so a NaN
            # value counts as a fill value and as a non-finite value.
            G = copy(B); G[3, 1, 2, 3] = NaN32
            nanfill = write_flux_file(joinpath(dir, "nan_fill.nc"), times, ["FLUX" => G]; fillvalue = NaN32)
            NCDataset(nanfill) do ds
                @test isnan(ds["FLUX"].attrib["_FillValue"]) && ismissing(ds["FLUX"][3, 1, 2, 3])
            end
            w = only(cmp_call(:compare_cs_flux_files, ref, nanfill, MESH).variables)
            @test w.invalid.fill_values_candidate == 1 && w.invalid.nonfinite_pairs == 1
            @test !w.checks[1].pass && w.checks[1].value == 2
            is_fill(x, fills) = Base.invokelatest(getproperty(CMP, :_is_fill), x, fills)
            @test is_fill(NaN32, Float32[NaN32]) && !is_fill(1.0f0, Float32[NaN32])
            @test is_fill(-0.0f0, Float32[0.0f0]) && !is_fill(NaN32, Float32[-999.0f0])
        end

        @testset "bitwise equality needs one stored type" begin
            bits_equal(a, b) = Base.invokelatest(getproperty(CMP, :_bits_equal), a, b)
            @test bits_equal(1.0f0, 1.0f0) && !bits_equal(0.0f0, -0.0f0)
            @test !bits_equal(1.0f0, 1.0) && !bits_equal(1.0, 1.0f0)
            # The same values stored as Float64: equal numbers, no stored bits to compare.
            wide = write_flux_file(joinpath(dir, "float64.nc"), times, ["FLUX" => Float64.(A)]; T = Float64)
            r = cmp_call(:compare_cs_flux_files, ref, wide, MESH;
                         tolerances = ["bitwise_fraction" => 0.0, "ulp_max" => 0, "cell_max" => 0.0])
            w = only(r.variables)
            @test w.stored_types == (; reference = "Float32", candidate = "Float64", equal = false)
            @test w.cells.all.bitwise_equal_fraction === nothing && w.cells.interior.bitwise_equal_fraction === nothing
            @test w.cells.all.ulp_max == 0 && w.cells.all.max == 0.0 && w.slice_totals.max_abs_rel_diff == 0.0
            checks = Dict(c.metric => c for c in w.checks)
            @test checks["bitwise_fraction.all"].status == "fail" && !checks["bitwise_fraction.all"].pass
            @test occursin("stored types differ (Float32 vs Float64)", checks["bitwise_fraction.all"].note)
            @test checks["ulp_max.all"].pass && checks["cell_max.all"].pass && !r.pass
            @test occursin("bitwise n/a", cmp_call(:summary_text, r))
            # A static field stored in two types.
            lf64 = write_static_file(joinpath(dir, "lf64.nc"), ["land_fraction" => (Float64.(mask), ("Xdim", "Ydim", "nf"))])
            w = only(cmp_call(:compare_cs_flux_files, ref, lf64, MESH; variables = ["land_fraction"]).variables)
            @test w.cells.all.bitwise_equal_fraction === nothing && w.cells.all.max == 0.0
        end

        @testset "quantiles over all cells without a merged copy" begin
            kth(vs, k) = Base.invokelatest(getproperty(CMP, :kth_smallest), vs, k)
            qz(vs, nzero, p) = Base.invokelatest(getproperty(CMP, :_quantile_with_zeros), vs, nzero, p)
            parts = [sort!(Float32.(rand(1:50, 400)) ./ 7), Float32[], sort!(rand(Float32, 301)),
                     sort!(vcat(fill(0.0f0, 3), Float32.(rand(1:50, 99)) ./ 7))]   # ties and zeros
            merged = sort!(reduce(vcat, parts))
            @test all(k -> kth(parts, k) === merged[k], eachindex(merged))
            @test kth(parts[1:1], 17) === parts[1][17]
            @test_throws BoundsError kth(parts, length(merged) + 1)
            for nzero in (0, 25), p in (0.0, 0.01, 0.5, 0.99, 1.0)
                @test qz(parts, nzero, p) ≈ quantile(vcat(zeros(nzero), Float64.(merged)), p) rtol = 1e-14
            end
            @test qz([Float32[], Float32[]], 0, 0.5) === nothing && qz([Float32[]], 2, 0.5) == 0.0
            # The summary over all cells copies no relative differences.
            fresh() = [(s = Base.invokelatest(getproperty(CMP, :CategoryStats)); s.n = 10^6;
                        s.rel = rand(Float32, 10^6); s) for _ in 1:3]
            summaries(stats) = Base.invokelatest(getproperty(CMP, :_cell_summaries), stats, DateTime[], true)
            summaries(fresh())
            stats = fresh()
            bytes = @allocated summaries(stats)
            @test bytes < 2^20                                       # 12 MB of differences, no copy
            all_summary, per_category = summaries(fresh())
            @test all_summary.n_sample == 3 * 10^6 && per_category[1].n_sample == 10^6
        end

        @testset "block reading" begin
            # One slice per block gives the same report as the default block.
            r1 = cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "land_fraction", block_slices = 1)
            w1 = only(r1.variables)
            @test w1.slice_totals.reference == v.slice_totals.reference
            @test w1.cells.all == v.cells.all && w1.cells.coastal == v.cells.coastal
            @test w1.bands.reference == v.bands.reference
        end

        K = (SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS) / LEGACY_CO2_PER_CARBON
        @testset "reference scale" begin
            preset = getproperty(CMP, :REFERENCE_SCALE_PRESETS).carbon_44_12_to_iupac
            @test preset.factor == K && K < 1
            @test getproperty(CMP, :LEGACY_CO2_PER_CARBON) == LEGACY_CO2_PER_CARBON
            resolve(x) = cmp_call(:resolve_reference_scale, x)
            @test resolve("carbon_44_12_to_iupac").factor == K
            @test resolve("carbon_44_12_to_iupac").preset == "carbon_44_12_to_iupac"
            @test resolve(0.5) == (; factor = 0.5, preset = nothing, definition = nothing)
            @test resolve("2.5").factor == 2.5 && resolve(2).factor == 2.0
            for bad in ("nonsense", 0, NaN, Inf, true, "0")
                @test_throws ArgumentError resolve(bad)
            end

            # A candidate written as T(Float64(a) × K) equals the scaled reference bitwise.
            S = Float32.(Float64.(A) .* K)
            scaled = write_flux_file(joinpath(dir, "scaled.nc"), times, ["FLUX" => S])
            r = cmp_call(:compare_cs_flux_files, ref, scaled, MESH;
                         reference_scale = ["FLUX" => "carbon_44_12_to_iupac"],
                         tolerances = ["bitwise_fraction" => 1.0, "slice_total" => 0.0])
            w = only(r.variables)
            @test r.pass && w.cells.all.bitwise_equal_fraction == 1.0 && w.cells.all.max == 0.0
            @test w.reference_scale.factor == K && w.reference_scale.preset == "carbon_44_12_to_iupac"
            @test w.slice_totals.reference ≈ [slice_total(S, k) for k in 1:nt] rtol = 1e-13
            @test r.reference_scale_rule !== nothing
            @test occursin("reference × ", cmp_call(:summary_text, r))
            # Without the scale the totals differ by K - 1 (up to Float32 rounding).
            u = only(cmp_call(:compare_cs_flux_files, ref, scaled, MESH).variables)
            @test u.reference_scale === nothing
            @test u.slice_totals.rel_diff ≈ [(slice_total(S, k) - slice_total(A, k)) / slice_total(A, k) for k in 1:nt] rtol = 1e-9
            @test all(x -> abs(x - (K - 1)) < 1e-6, u.slice_totals.rel_diff)

            # The legacy case: a carbon field converted with 44/12 (reference) and
            # with M(CO2)/M(C) (candidate), each rounded once to Float32.
            carbon = Float64.(A) ./ 3.7
            old = Float32.(carbon .* LEGACY_CO2_PER_CARBON)
            new = Float32.(carbon .* (SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS))
            f_old = write_flux_file(joinpath(dir, "legacy_44_12.nc"), times, ["FLUX" => old])
            f_new = write_flux_file(joinpath(dir, "iupac.nc"), times, ["FLUX" => new])
            w = only(cmp_call(:compare_cs_flux_files, f_old, f_new, MESH;
                              reference_scale = Dict("FLUX" => "carbon_44_12_to_iupac")).variables)
            rescaled = Float32.(Float64.(old) .* K)
            d = [ulps(rescaled[c], new[c]) for c in eachindex(new)]
            @test w.cells.all.ulp_max == maximum(d) <= 1
            @test w.cells.all.bitwise_equal_fraction ==
                  count(c -> reinterpret(UInt32, rescaled[c]) == reinterpret(UInt32, new[c]), eachindex(new)) / length(new)
            @test only(cmp_call(:compare_cs_flux_files, f_old, f_new, MESH).variables).cells.all.bitwise_equal_fraction < 0.1

            # Fill values are found on the stored reference values, before scaling.
            F = copy(A); F[1, 1, 1, 1] = -999.0f0
            filled = write_flux_file(joinpath(dir, "filled_ref.nc"), times, ["FLUX" => F]; fillvalue = -999.0f0)
            w = only(cmp_call(:compare_cs_flux_files, filled, ref, MESH; reference_scale = ["FLUX" => 0.5]).variables)
            @test w.invalid.fill_values_reference == 1 && !w.checks[1].pass

            # A scale for a variable that is not compared, or given twice, is an error.
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, ref, cand, MESH; reference_scale = ["OTHER" => 2.0])
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, ref, cand, MESH;
                                                reference_scale = ["FLUX" => 2.0, "FLUX" => 3.0])
        end

        @testset "variables without a time dimension" begin
            area_mesh = repeat(AREA, 1, 1, 6)
            cand_static = write_flux_file(joinpath(dir, "cand_static.nc"), times, ["FLUX" => B];
                                          extra = ["land_fraction" => (mask, ("Xdim", "Ydim", "nf")),
                                                   "cell_area" => (area_mesh, ("Xdim", "Ydim", "nf"))])
            r = cmp_call(:compare_cs_flux_files, ref, cand_static, MESH; variables = ["land_fraction", "cell_area"],
                         mask_var = "land_fraction",
                         tolerances = ["bitwise_fraction" => 1.0, "slice_total" => 0.0, "cell_max" => 1e-4])
            @test r.time === nothing && r.reference.n_slices === nothing && r.reference.time_units === nothing
            lf, ca = r.variables
            @test lf.kind == ca.kind == "static"
            @test lf.slice_totals === nothing && lf.month_totals === nothing && lf.bands === nothing
            @test lf.cells.all.n_values == 6 * NC^2 && lf.cells.all.bitwise_equal_fraction == 1.0
            @test lf.cells.all.n_both_zero == count(==(0), mask)
            @test lf.pass && [c.status for c in lf.checks] == ["pass", "pass", "n/a", "pass"]
            @test occursin("no time dimension", lf.checks[3].note) && lf.checks[3].pass
            # cell_area differs in the one perturbed cell.
            @test ca.stored_types.reference == "Float64"
            @test ca.cells.all.bitwise_equal_fraction == (6 * NC^2 - 1) / (6 * NC^2)
            @test ca.cells.all.max ≈ rel_cell(area_file[3, 2, 5], area_mesh[3, 2, 5]) rtol = 1e-15
            @test (ca.cells.all.worst.i, ca.cells.all.worst.j, ca.cells.all.worst.panel) == (3, 2, 5)
            @test ca.cells.all.worst.time === nothing
            @test ca.cells.all.ulp_max == ulps(Float32(area_file[3, 2, 5]), Float32(area_mesh[3, 2, 5])) > 0
            @test getproperty(ca.cells, cats[(3, 2, 5)]).max == ca.cells.all.max
            @test !ca.pass && !r.pass && [c.status for c in ca.checks] == ["pass", "fail", "n/a", "pass"]
            text = cmp_call(:summary_text, r)
            @test occursin("none read", text) && occursin("land_fraction [] (no time dimension): PASS", text)
            @test occursin("n/a (not applicable", text)

            # Series and fields together: the time axis is read for the series.
            r = cmp_call(:compare_cs_flux_files, ref, cand_static, MESH; variables = ["FLUX", "land_fraction"])
            @test [w.kind for w in r.variables] == ["series", "static"] && r.time.n_compared == nt
            # A file without a time axis.
            lf_only = write_static_file(joinpath(dir, "lf_only.nc"), ["land_fraction" => (mask, ("Xdim", "Ydim", "nf"))])
            r = cmp_call(:compare_cs_flux_files, ref, lf_only, MESH; variables = ["land_fraction"],
                         tolerances = ["bitwise_fraction" => 1.0])
            @test r.pass && r.candidate.time_units === nothing
            region = write_static_file(joinpath(dir, "region.nc"),
                                       ["W" => (zeros(Float32, NC, NC, 6, 2), ("Xdim", "Ydim", "nf", "region"))])
            @test_throws DimensionMismatch cmp_call(:compare_cs_flux_files, region, region, MESH; variables = ["W"])
            # A field against a series is an error.
            @test_throws DimensionMismatch cmp_call(:compare_cs_flux_files, ref, cand_static, MESH;
                                                    variables = ["land_fraction" => "FLUX"])
        end

        @testset "checks with an empty sample pass as n/a" begin
            # Every land cell of the mask is coastal or polar: land_fraction is 0 on all interior cells.
            @test all(c -> cats[c] != :interior || mask[c...] == 0, keys(cats))
            r = cmp_call(:compare_cs_flux_files, ref, ref, MESH; variables = ["land_fraction"],
                         mask_var = "land_fraction",
                         tolerances = ["cell_p50.interior" => 0.0, "cell_max.interior" => 0.0,
                                       "bitwise_fraction.interior" => 1.0, "ulp_max.interior" => 0])
            w = only(r.variables)
            @test w.cells.interior.n_sample == 0 && w.cells.interior.n_values == count(==(:interior), values(cats))
            @test [c.status for c in w.checks] == ["pass", "n/a", "n/a", "pass", "pass"]
            @test all(c -> c.pass, w.checks) && r.pass
            @test occursin("every value is zero", w.checks[2].note) && w.checks[2].value === nothing
            # A mask without coastal cells (all land).
            all_land = write_static_file(joinpath(dir, "all_land.nc"), ["lf" => (ones(Float32, NC, NC, 6), ("Xdim", "Ydim", "nf"))])
            r = cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "lf", mask_file = all_land,
                         tolerances = ["cell_p99.coastal" => 0.0, "ulp_max.coastal" => 0, "bitwise_fraction.coastal" => 1.0])
            w = only(r.variables)
            @test r.categories.n_cells.coastal == 0 && w.cells.coastal.n_values == 0
            @test [c.status for c in w.checks] == ["pass", "n/a", "n/a", "n/a"] && w.pass && r.pass
            @test all(c -> occursin("no cells", c.note), w.checks[2:end])
            # The same coastal tolerance without a mask is an error, not n/a.
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, ref, cand, MESH; tolerances = ["cell_p99.coastal" => 0.0])
            # A metric that cannot be computed still fails: one slice without time bounds.
            single = joinpath(dir, "single.nc")
            r = cmp_call(:compare_cs_flux_files, single, single, MESH; tolerances = ["year_total" => 1.0])
            chk = only(r.variables).checks[end]
            @test chk.status == "fail" && !chk.pass && occursin("durations unknown", chk.note) && !r.pass
            @test occursin("FAIL (slice durations unknown", cmp_call(:summary_text, r))
        end

        @testset "units must agree" begin
            carbon_units = write_flux_file(joinpath(dir, "carbon_units.nc"), times, ["FLUX" => B]; units = "kg C m-2 s-1")
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, ref, carbon_units, MESH)
            r = cmp_call(:compare_cs_flux_files, ref, carbon_units, MESH; allow_unit_mismatch = true)
            w = only(r.variables)
            @test !w.units.equal && w.units.reference == "kg CO2 m-2 s-1" && w.units.candidate == "kg C m-2 s-1"
            @test w.cells.all.bitwise_equal_fraction == v.cells.all.bitwise_equal_fraction
            @test w.slice_totals.candidate == v.slice_totals.candidate          # compared as stored
            @test occursin("[kg CO2 m-2 s-1 vs kg C m-2 s-1]", cmp_call(:summary_text, r))
            # A series must declare its units, even when both files leave them out.
            no_units = write_flux_file(joinpath(dir, "no_units.nc"), times, ["FLUX" => B]; units = nothing)
            blank_units = write_flux_file(joinpath(dir, "blank_units.nc"), times, ["FLUX" => B]; units = " ")
            for (a, b) in ((ref, no_units), (no_units, ref), (no_units, no_units), (blank_units, blank_units))
                @test_throws ArgumentError cmp_call(:compare_cs_flux_files, a, b, MESH; allow_unit_mismatch = true)
            end
            # (Xdim, Ydim, nf) fields such as masks may be unitless.
            r = cmp_call(:compare_cs_flux_files, ref, ref, MESH; variables = ["land_fraction"])
            @test r.pass && only(r.variables).units == (; reference = "", candidate = "", equal = true)
        end

        @testset "layout: dimension names and stored cell centres" begin
            dims3 = ("Xdim", "Ydim", "nf")
            # Swapped X and Y dimensions: rejected for a named variable and in the default selection.
            swapped_xy = write_flux_file(joinpath(dir, "swapped_xy.nc"), times, ["FLUX" => permutedims(B, (2, 1, 3, 4))];
                                         field_dims = ("Ydim", "Xdim", "nf", "time"))
            @test_throws DimensionMismatch cmp_call(:compare_cs_flux_files, ref, swapped_xy, MESH; variables = ["FLUX"])
            @test_throws DimensionMismatch cmp_call(:compare_cs_flux_files, ref, swapped_xy, MESH)
            @test_throws DimensionMismatch cmp_call(:compare_cs_flux_files, swapped_xy, ref, MESH)
            lf_xy = write_static_file(joinpath(dir, "lf_xy.nc"), ["land_fraction" => (permutedims(mask, (2, 1, 3)), ("Ydim", "Xdim", "nf"))])
            @test_throws DimensionMismatch cmp_call(:compare_cs_flux_files, ref, lf_xy, MESH; variables = ["land_fraction"])
            # Stored centres that match the mesh: accepted and reported.
            lon, lat = centre_fields(MESH)
            centred = write_flux_file(joinpath(dir, "centred.nc"), times, ["FLUX" => B];
                                      extra = ["lons" => (lon, dims3), "lats" => (lat, dims3)])
            r = cmp_call(:compare_cs_flux_files, ref, centred, MESH)
            @test r.candidate.cell_centre_vs_mesh_deg < 1e-9 && r.reference.cell_centre_vs_mesh_deg === nothing
            @test only(r.variables).cells.all == v.cells.all
            # Swapped panels (data and centres in another panel order): rejected.
            order = [2, 1, 3, 4, 5, 6]
            lon_p, lat_p = centre_fields(MESH; panels = order)
            swapped_panels = write_flux_file(joinpath(dir, "swapped_panels.nc"), times, ["FLUX" => B[:, :, order, :]];
                                             extra = ["lons" => (lon_p, dims3), "lats" => (lat_p, dims3)])
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, ref, swapped_panels, MESH)
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, swapped_panels, ref, MESH; variables = ["FLUX"])
            # X and Y transposed under the right dimension names: the stored centres give it away.
            transposed = write_flux_file(joinpath(dir, "transposed_xy.nc"), times, ["FLUX" => permutedims(B, (2, 1, 3, 4))];
                                         extra = ["lons" => (permutedims(lon, (2, 1, 3)), dims3),
                                                  "lats" => (permutedims(lat, (2, 1, 3)), dims3)])
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, ref, transposed, MESH)
            # Centres stored in another layout cannot be checked: rejected.
            other_layout = write_flux_file(joinpath(dir, "centres_other_layout.nc"), times, ["FLUX" => B];
                                           extra = ["lons" => (permutedims(lon, (3, 2, 1)), ("nf", "Ydim", "Xdim")),
                                                    "lats" => (permutedims(lat, (3, 2, 1)), ("nf", "Ydim", "Xdim"))])
            @test_throws DimensionMismatch cmp_call(:compare_cs_flux_files, ref, other_layout, MESH)
            # Only one of `lats` and `lons`: rejected, in either file.
            only_lats = write_flux_file(joinpath(dir, "only_lats.nc"), times, ["FLUX" => B];
                                        extra = ["lats" => (lat, dims3)])
            @test occursin("stores `lats` without `lons`",
                           error_message(() -> cmp_call(:compare_cs_flux_files, ref, only_lats, MESH)))
            @test occursin("stores `lats` without `lons`",
                           error_message(() -> cmp_call(:compare_cs_flux_files, only_lats, ref, MESH; variables = ["FLUX"])))
        end

        @testset "coastal mask from another file" begin
            dims = ("Xdim", "Ydim", "nf")
            lon, lat = centre_fields(MESH)
            mf = write_static_file(joinpath(dir, "mask.nc"), ["lf" => (mask, dims), "lons" => (lon, dims), "lats" => (lat, dims)])
            r = cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "lf", mask_file = mf, mask_threshold = 0.5)
            @test r.categories.mask_file == abspath(mf) && r.categories.mask_var == "lf"
            @test r.categories.mask_threshold == 0.5
            @test r.categories.mask_cell_centre_vs_mesh_deg < 1e-9
            @test r.categories.n_cells == report.categories.n_cells
            w = only(r.variables)
            @test w.cells.coastal.max == v.cells.coastal.max && w.cells.coastal.n_values == v.cells.coastal.n_values
            @test report.reference.cell_centre_vs_mesh_deg === nothing    # the flux fixtures store no centres
            # 1 on ocean and 0 on land gives the same coastal cells.
            ocean = write_static_file(joinpath(dir, "ocean.nc"), ["ocean" => (1 .- mask, dims)])
            r = cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "ocean", mask_file = ocean)
            @test r.categories.n_cells == report.categories.n_cells
            @test r.categories.mask_cell_centre_vs_mesh_deg === nothing
            # (Xdim, Ydim, nf, time) with one slice is accepted.
            m4 = write_static_file(joinpath(dir, "mask4.nc"), ["lf" => (reshape(mask, NC, NC, 6, 1), (dims..., "time"))])
            r = cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "lf", mask_file = m4)
            @test r.categories.n_cells == report.categories.n_cells
            # Other layouts are rejected.
            transposed = write_static_file(joinpath(dir, "mask_t.nc"), ["lf" => (permutedims(mask, (3, 2, 1)), ("nf", "Ydim", "Xdim"))])
            @test_throws DimensionMismatch cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "lf", mask_file = transposed)
            two = write_static_file(joinpath(dir, "mask2.nc"), ["lf" => (cat(mask, mask; dims = 4), (dims..., "time"))])
            @test_throws DimensionMismatch cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "lf", mask_file = two)
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "nope", mask_file = mf)
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_file = mf)
            # Stored centres must lie within a tenth of the nominal cell width of the mesh centres.
            limit = cmp_call(:centre_deviation_limit_deg, NC)
            @test limit == 0.1 * 90 / NC
            lon2, lat2 = centre_fields(MESH; panels = [2, 1, 3, 4, 5, 6])
            swapped = write_static_file(joinpath(dir, "mask_swapped.nc"),
                                        ["lf" => (mask, dims), "lons" => (lon2, dims), "lats" => (lat2, dims)])
            @test_throws ArgumentError cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "lf", mask_file = swapped)
            only_lons = write_static_file(joinpath(dir, "mask_only_lons.nc"), ["lf" => (mask, dims), "lons" => (lon, dims)])
            @test occursin("stores `lons` without `lats`",
                           error_message(() -> cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "lf",
                                                        mask_file = only_lons)))
            for (shift, ok) in ((0.5, true), (2.0, false))
                lat_s = copy(lat); lat_s[1, 1, 1] += shift * limit
                shifted = write_static_file(joinpath(dir, "mask_shift_$(shift).nc"),
                                            ["lf" => (mask, dims), "lons" => (lon, dims), "lats" => (lat_s, dims)])
                if ok
                    r = cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "lf", mask_file = shifted)
                    @test r.categories.mask_cell_centre_vs_mesh_deg ≈ shift * limit rtol = 1e-9
                else
                    @test_throws ArgumentError cmp_call(:compare_cs_flux_files, ref, cand, MESH; mask_var = "lf",
                                                        mask_file = shifted)
                end
            end
        end

        @testset "command line, JSON and summary" begin
            parse = getproperty(CMP, :_parse_args)
            base = ["--reference", ref, "--candidate", cand, "--nc", "4"]
            opts = Base.invokelatest(parse, vcat(base, "--tol", "cell_p99=1e-6", "--var", "FLUX",
                                                 "--var", "OTHER:OTHER_B"))
            @test opts.tolerances == ["cell_p99.all" => 1e-6]
            @test opts.variables == ["FLUX" => "FLUX", "OTHER" => "OTHER_B"]
            @test opts.Nc == 4 && opts.radius === nothing && opts.mask_threshold == 0.0
            @test !opts.common_variables_only
            @test Base.invokelatest(parse, vcat(base, "--common-variables-only")).common_variables_only
            for bad in (vcat(base, "--tol", "nonsense=1"),
                        vcat(base, "--tol", "cell_p99.coastal=1"),            # coastal without a mask
                        vcat(base, "--tol", "slice_total.interior=1"),        # no categories for totals
                        vcat(base, "--tol", "cell_max=1", "--tol", "cell_max.all=2"),
                        vcat(base, "--binary", ref),                          # two meshes
                        vcat(base, "--mask-threshold", "0.5"),                # threshold without a mask
                        vcat(base, "--tol", "cell_max=abc"),
                        vcat(base, "--reference-scale", "FLUX=nonsense"),
                        vcat(base, "--reference-scale", "FLUX"),
                        vcat(base, "--reference-scale", "FLUX=0"),
                        vcat(base, "--reference-scale", "FLUX=1", "--reference-scale", "FLUX=2"),
                        vcat(base, "--mask-file", ref),                       # mask file without a variable
                        vcat(base, "--mask-var", "land_fraction", "--mask-threshold", "NaN"),
                        vcat(base, "--common-variables-only", "--var", "FLUX"),  # explicit selection
                        ["--reference", ref, "--candidate", cand])            # no mesh
                @test_throws ArgumentError Base.invokelatest(parse, bad)
            end
            opts = Base.invokelatest(parse, vcat(base, "--reference-scale", "FLUX=carbon_44_12_to_iupac",
                                                 "--reference-scale", "OTHER=0.5", "--allow-unit-mismatch",
                                                 "--mask-var", "land_fraction", "--mask-file", ref,
                                                 "--mask-threshold", "0.5"))
            @test [s.first for s in opts.reference_scale] == ["FLUX", "OTHER"]
            @test opts.reference_scale[1].second.factor == K
            @test opts.reference_scale[1].second.preset == "carbon_44_12_to_iupac"
            @test opts.reference_scale[2].second.factor == 0.5
            @test opts.allow_unit_mismatch && opts.mask_var == "land_fraction" && opts.mask_file == ref
            @test opts.mask_threshold == 0.5 && !Base.invokelatest(parse, base).allow_unit_mismatch

            main = getproperty(CMP, :main)
            json = joinpath(dir, "out", "report.json")
            summary = joinpath(dir, "out", "summary.txt")
            args = vcat(base, "--mask-var", "land_fraction", "--json", json, "--summary", summary,
                        "--tol", "cell_max.interior=1.0", "--tol", "bitwise_fraction=0.98")
            @test redirect_stdout(devnull) do
                Base.invokelatest(main, args)
            end == 0
            parsed = JSON3.read(read(json, String))
            @test parsed.pass === true
            jv = only(parsed.variables)
            @test jv.cells.all.p99 ≈ v.cells.all.p99 rtol = 1e-15
            @test jv.cells.coastal.max == v.cells.coastal.max
            @test jv.slice_totals.reference == v.slice_totals.reference
            @test jv.month_totals.months[1].month == "2021-12"
            @test length(jv.bands.reference) == 60
            @test [c.metric for c in jv.checks] == ["finite", "cell_max.interior", "bitwise_fraction.all"]
            @test [c.status for c in jv.checks] == ["pass", "pass", "pass"]
            @test jv.kind == "series" && jv.reference_scale === nothing
            @test parsed.mesh.source == "constructed" && parsed.categories.polar_panels == [3, 6]
            @test parsed.categories.mask_var == "land_fraction" && parsed.categories.mask_file == abspath(ref)
            text = read(summary, String)
            @test startswith(text, "compare_cs_flux_files: PASS")
            @test occursin("FLUX [kg CO2 m-2 s-1]: PASS", text) && occursin("coastal", text)
            @test !any(f -> occursin(".tmp-", f), readdir(dirname(json)))
            # Non-finite numbers become strings in JSON.
            @test JSON3.write(Base.invokelatest(getproperty(CMP, :_jsonable), (; x = Inf, y = [1.0, NaN]))) ==
                  "{\"x\":\"Inf\",\"y\":[1.0,\"NaN\"]}"
            @test redirect_stdout(devnull) do
                Base.invokelatest(main, vcat(base, "--tol", "bitwise_fraction=0.999"))
            end == 1
            # A reference scale from the command line.
            json2 = joinpath(dir, "out", "scaled.json")
            @test redirect_stdout(devnull) do
                Base.invokelatest(main, ["--reference", ref, "--candidate", joinpath(dir, "scaled.nc"), "--nc", "4",
                                         "--reference-scale", "FLUX=carbon_44_12_to_iupac",
                                         "--tol", "bitwise_fraction=1", "--json", json2])
            end == 0
            js = only(JSON3.read(read(json2, String)).variables)
            @test js.reference_scale.preset == "carbon_44_12_to_iupac" && js.reference_scale.factor == K
            # Different units stop the command line unless allowed.
            units_args = ["--reference", ref, "--candidate", joinpath(dir, "carbon_units.nc"), "--nc", "4"]
            @test_throws ArgumentError redirect_stdout(devnull) do
                Base.invokelatest(main, units_args)
            end
            @test redirect_stdout(devnull) do
                Base.invokelatest(main, vcat(units_args, "--allow-unit-mismatch"))
            end == 0
            # A dropped series fails the command line unless --common-variables-only.
            dropped_args = ["--reference", joinpath(dir, "gpp_ter.nc"), "--candidate", joinpath(dir, "gpp.nc"),
                            "--nc", "4", "--json", json]
            @test redirect_stdout(devnull) do
                Base.invokelatest(main, dropped_args)
            end == 1
            @test JSON3.read(read(json, String)).variable_selection.series_only_in_reference == ["TER"]
            @test redirect_stdout(devnull) do
                Base.invokelatest(main, vcat(dropped_args, "--common-variables-only"))
            end == 0
            @test JSON3.read(read(json, String)).variable_selection.common_variables_only === true
        end

        @testset "cell areas from a transport binary" begin
            bin = joinpath(dir, "c4.bin")
            vertical = HybridSigmaPressure([0.0, 500.0, 0.0], [0.0, 0.5, 1.0])
            window = (m = ntuple(_ -> fill(1e16, NC, NC, 2), 6), am = ntuple(_ -> zeros(NC + 1, NC, 2), 6),
                      bm = ntuple(_ -> zeros(NC, NC + 1, 2), 6), cm = ntuple(_ -> zeros(NC, NC, 3), 6),
                      ps = ntuple(_ -> fill(1e5, NC, NC), 6))
            writer = MD.open_streaming_cs_transport_binary(bin, NC, 6, 2, 1, vertical; mass_basis = :dry,
                                                           surface_pressure_basis = DrySurfacePressure(),
                                                           panel_convention = "geos_native",
                                                           planet_radius = IFS_EARTH_RADIUS)
            MD.write_streaming_cs_window!(writer, window, NC, 6)
            MD.close_streaming_transport_binary!(writer)
            mesh, info = cmp_call(:load_target_mesh; binary = bin)
            @test info.source == "binary" && mesh.radius == IFS_EARTH_RADIUS
            ifs = CubedSphereMesh(; Nc = NC, FT = Float64, Hp = 0, convention = GEOSNativePanelConvention(),
                                  radius = IFS_EARTH_RADIUS)
            @test mesh.cell_areas ≈ ifs.cell_areas rtol = 1e-14
            @test cmp_call(:load_target_mesh; Nc = NC, radius = IFS_EARTH_RADIUS)[1].cell_areas == ifs.cell_areas
            r = cmp_call(:compare_cs_flux_files, ref, cand, mesh; mesh_info = info)
            @test r.mesh.source == "binary" && r.mesh.radius_m == IFS_EARTH_RADIUS
            @test only(r.variables).slice_totals.reference[1] ≈
                  sum(Float64(A[i, j, p, 1]) * ifs.cell_areas[i, j] for p in 1:6, j in 1:NC, i in 1:NC) rtol = 1e-13
            # The area check reports the file's EARTH_RADIUS areas against the IFS mesh.
            @test r.reference.cell_area_vs_mesh.max_rel_diff ≈ 1 - (EARTH_RADIUS / IFS_EARTH_RADIUS)^2 rtol = 1e-3
        end

        @testset "exit status of the command line" begin
            project = dirname(Base.active_project())
            cmd = `$(Base.julia_cmd()) --project=$(project) $(COMPARATOR) --reference $(ref)
                   --candidate $(cand) --nc 4 --tol bitwise_fraction=0.999`
            out = IOBuffer()
            proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out); wait = true)
            text = String(take!(out))
            @test proc.exitcode == 1
            @test occursin("compare_cs_flux_files: FAIL", text)
            @test occursin("check bitwise_fraction.all", text)
        end
    end
end
