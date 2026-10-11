#!/usr/bin/env julia
#
# GriddedIO coordinates: `find_coordinate` by variable name only, lat-lon
# orientation, and the fill policies of `read_raw`. The cases are those measured
# in step P0 of the surface-flux port
# (docs/memos/2026-10-10_python_flux_port_p0_facts.md, Sections 3 and 5):
# FLUXCOM-X labels its `lon` variable `units = degrees_north`,
# `long_name = latitude`; GFED5 NRT coordinates carry no attributes; VISIT,
# VISIT-UT and iMAPLE mask only through `missing_value = -99999`; ELM-FATES and
# LPJwsl use `_FillValue = NaN`; DLEM has no fill attribute; the IBIS land
# fraction declares a wrong `missing_value = 0`.
#
# The runtime behaviour the functions replace (the runtime loaders' coordinate
# search and lat-flip / lon-roll, `nomissing(…, 0)`) is frozen below and
# compared bit for bit.

using Test
using NCDatasets

using AtmosTransport
using AtmosTransport.GriddedIO: find_coordinate, LongitudeAxis, LatitudeAxis, TimeAxis,
                                VerticalAxis, LatLonOrientation, latlon_orientation, orient,
                                orient_coordinates, read_raw, FillValueOnly, MaskedAsNaN,
                                MaskedAsZero

# ---- Frozen runtime behaviour (src/Models/InitialConditionIO.jl and the
# ---- static/time-varying surface-flux loaders, before the GriddedIO move) ----

function frozen_ic_find_coord(ds, candidates::Vector{String})
    for name in candidates
        haskey(ds, name) && return name
    end
    return nothing
end

# The lat-flip / lon-roll block of `_load_file_surface_flux_field` (raw field
# and optional cell area) and `_load_single_timevarying_surface_flux_field`.
function frozen_reorient(raw, lon_src, lat_src, cell_area_src)
    trailing = ntuple(_ -> Colon(), ndims(raw) - 2)
    if length(lat_src) > 1 && lat_src[1] > lat_src[end]
        raw = raw[:, end:-1:1, trailing...]
        lat_src = reverse(lat_src)
        cell_area_src === nothing || (cell_area_src = cell_area_src[:, end:-1:1])
    end
    if minimum(lon_src) < 0
        split = findfirst(>=(0), lon_src)
        if split !== nothing
            idx = vcat(split:length(lon_src), 1:split-1)
            lon_src = mod.(lon_src[idx], 360.0)
            raw = raw[idx, :, trailing...]
            cell_area_src === nothing || (cell_area_src = cell_area_src[idx, :])
        end
    end
    return raw, lon_src, lat_src, cell_area_src
end

function new_reorient(raw, lon, lat, cell_area)
    o = latlon_orientation(lon, lat)
    lon_o, lat_o = orient_coordinates(lon, lat, o)
    return orient(raw, o), lon_o, lat_o, cell_area === nothing ? nothing : orient(cell_area, o)
end

function write_coordinate_file(path; lon_attrib = Dict{String, Any}(), lat_attrib = Dict{String, Any}(),
                               lon = collect(-179.5:1.0:179.5), lat = collect(-89.5:1.0:89.5),
                               extra = ds -> nothing)
    NCDataset(path, "c") do ds
        defDim(ds, "lon", length(lon))
        defDim(ds, "lat", length(lat))
        defVar(ds, "lon", Float64, ("lon",); attrib = lon_attrib)[:] = lon
        defVar(ds, "lat", Float64, ("lat",); attrib = lat_attrib)[:] = lat
        extra(ds)
    end
    return path
end

@testset "find_coordinate: by variable name only" begin
    dir = mktempdir()

    # FLUXCOM-X: `lon` is labelled as a latitude; the names decide.
    fx = write_coordinate_file(joinpath(dir, "fluxcom_like.nc");
        lon_attrib = Dict{String, Any}("units" => "degrees_north", "long_name" => "latitude"),
        lat_attrib = Dict{String, Any}("units" => "degrees_north", "long_name" => "latitude"),
        lat = collect(89.875:-0.25:-89.875), lon = collect(-179.875:0.25:179.875))
    NCDataset(fx) do ds
        @test find_coordinate(ds, LongitudeAxis()) == "lon"
        @test find_coordinate(ds, LatitudeAxis()) == "lat"
        @test find_coordinate(ds, LongitudeAxis(); length = 1440) == "lon"
        @test_throws DimensionMismatch find_coordinate(ds, LatitudeAxis(); length = 721)
        @test_throws ArgumentError find_coordinate(ds, TimeAxis())
        @test find_coordinate(ds, TimeAxis(); required = false) === nothing
    end

    # GFED5 NRT: coordinates without any attribute.
    nrt = write_coordinate_file(joinpath(dir, "gfed_nrt_like.nc");
                                lon = collect(-179.875:0.25:179.875), lat = collect(-89.875:0.25:89.875))
    NCDataset(nrt) do ds
        @test isempty(collect(keys(ds["lon"].attrib)))
        @test find_coordinate(ds, LongitudeAxis()) == "lon"
        @test find_coordinate(ds, LatitudeAxis()) == "lat"
    end

    # A variable that only an attribute calls a latitude is not found.
    attr_only = joinpath(dir, "attribute_only.nc")
    NCDataset(attr_only, "c") do ds
        defDim(ds, "row", 3)
        defVar(ds, "nav_lat", Float64, ("row",);
               attrib = Dict("standard_name" => "latitude", "units" => "degrees_north",
                             "axis" => "Y"))[:] = [-60.0, 0.0, 60.0]
    end
    NCDataset(attr_only) do ds
        @test_throws ArgumentError find_coordinate(ds, LatitudeAxis())
    end

    # Long names, the legacy single-letter names, the vertical axis, a 2-D
    # coordinate and bounds variables.
    other = joinpath(dir, "other_names.nc")
    NCDataset(other, "c") do ds
        defDim(ds, "x", 4); defDim(ds, "nb", 2); defDim(ds, "t", 2); defDim(ds, "lev", 3)
        defVar(ds, "latitude", Float64, ("x", "nb"))[:, :] = zeros(4, 2)    # 2-D
        defVar(ds, "x", Float64, ("x",))[:] = [0.0, 90.0, 180.0, 270.0]
        defVar(ds, "lat_bnds", Float64, ("nb", "x"))[:, :] = zeros(2, 4)
        defVar(ds, "t", Float64, ("t",))[:] = [0.0, 1.0]
        defVar(ds, "lev", Float64, ("lev",))[:] = [1.0, 2.0, 3.0]
    end
    NCDataset(other) do ds
        @test find_coordinate(ds, LongitudeAxis()) == "x"
        @test find_coordinate(ds, TimeAxis()) == "t"
        @test find_coordinate(ds, VerticalAxis()) == "lev"
        @test_throws ArgumentError find_coordinate(ds, LatitudeAxis())   # `latitude` is 2-D
        # Same answers as the runtime's search wherever it found a name.
        @test frozen_ic_find_coord(ds, ["lon", "longitude", "x"]) == find_coordinate(ds, LongitudeAxis())
        @test frozen_ic_find_coord(ds, ["time", "t"]) == find_coordinate(ds, TimeAxis())
        @test frozen_ic_find_coord(ds, ["lev", "level", "plev", "z", "hybrid", "nhym"]) ==
              find_coordinate(ds, VerticalAxis())
    end
    NCDataset(fx) do ds
        @test frozen_ic_find_coord(ds, ["lon", "longitude", "x"]) == find_coordinate(ds, LongitudeAxis())
        @test frozen_ic_find_coord(ds, ["lat", "latitude", "y"]) == find_coordinate(ds, LatitudeAxis())
    end
end

@testset "lat-lon orientation reproduces the runtime loaders" begin
    cases = [
        # (name, lon, lat)
        ("VISIT: descending lat, -180..180 lon", collect(-179.75:0.5:179.75), collect(89.75:-0.5:-89.75)),
        ("CABLE-POP: ascending, -180..180", collect(-179.5:1.0:179.5), collect(-89.5:1.0:89.5)),
        ("CLM-FATES: wrapped lon", [collect(0.0:2.5:177.5); collect(-180.0:2.5:-2.5)],
         collect(range(-59.684, 90.0; length = 80))),
        ("CLM: 0..360 lon", collect(0.0:1.25:358.75), collect(range(-90.0, 90.0; length = 192))),
        ("ELM: centre on -180", collect(-180.0:1.25:178.75), collect(range(-90.0, 90.0; length = 192))),
        ("regional, all negative lon", collect(-99.5:1.0:-50.5), collect(20.5:1.0:49.5)),
        ("GridFED: Float32 centres", Float64.(Float32.(collect(-179.95:0.1:179.95))),
         Float64.(Float32.(collect(-89.95:0.1:89.95)))),
    ]
    for (name, lon, lat) in cases
        @testset "$name" begin
            Nx, Ny = length(lon), length(lat)
            raw3 = reshape(Float32.(1:(Nx * Ny * 3)), Nx, Ny, 3)
            raw2 = reshape(Float64.(1:(Nx * Ny)) .* 0.5, Nx, Ny)
            area = reshape(Float64.(1:(Nx * Ny)) .+ 1e3, Nx, Ny)
            for (raw, ca) in ((raw3, area), (raw2, nothing), (raw3, nothing))
                old = frozen_reorient(raw, lon, lat, ca)
                new = new_reorient(raw, lon, lat, ca)
                for (a, b) in zip(old, new)
                    @test typeof(a) == typeof(b)
                    @test isequal(a, b)
                end
            end
        end
    end

    # Orientation fields.
    @test latlon_orientation(collect(-179.5:1.0:179.5), [-1.0, 1.0]) == LatLonOrientation(false, 180, true)
    @test latlon_orientation(collect(0.5:1.0:359.5), [1.0, -1.0]) == LatLonOrientation(true, 0, false)
    @test latlon_orientation([0.0, 90.0, -180.0, -90.0], [0.0]) == LatLonOrientation(false, 0, true)
    @test latlon_orientation([-10.0, -5.0], [0.0, 1.0]) == LatLonOrientation(false, 0, false)
    # A trailing dimension is untouched; a 1-D field is refused.
    o = LatLonOrientation(true, 1, true)
    A = reshape(collect(1:24), 3, 2, 4)
    @test orient(A, o)[:, :, 4] == A[[2, 3, 1], [2, 1], 4]
    @test_throws ArgumentError orient(collect(1:3), o)
    @test_throws ArgumentError orient(zeros(2, 2), LatLonOrientation(false, 2, true))
    @test_throws ArgumentError latlon_orientation(Float64[], [0.0])
end

@testset "fill policies" begin
    dir = mktempdir()
    path = joinpath(dir, "fills.nc")
    Nx, Ny, Nt = 4, 3, 2
    data = reshape(Float32.(1:(Nx * Ny * Nt)) .* 1.5f0, Nx, Ny, Nt)
    NCDataset(path, "c") do ds
        defDim(ds, "lon", Nx); defDim(ds, "lat", Ny); defDim(ds, "time", Nt)
        dims = ("lon", "lat", "time")
        # VISIT/iMAPLE: only missing_value.
        v = defVar(ds, "visit", Float32, dims; attrib = Dict("missing_value" => -99999f0))
        d = copy(data); d[1, 1, 1] = -99999f0; d[4, 3, 2] = -99999f0
        v.var[:, :, :] = d
        # IBIS land fraction: a real _FillValue and a wrong missing_value = 0.
        v = defVar(ds, "ibis_fraction", Float32, ("lon", "lat");
                   fillvalue = -99999f0, attrib = Dict("missing_value" => 0f0))
        f = Float32[0.0 0.25 0.5; 1.0 -99999 0.0; 0.75 0.0 1.0; 0.5 0.5 -99999]
        v.var[:, :] = f
        # ELM-FATES/LPJwsl: _FillValue = NaN.
        v = defVar(ds, "elm", Float32, dims; fillvalue = NaN32)
        d = copy(data); d[2, 2, 1] = NaN32
        v.var[:, :, :] = d
        # CABLE-POP: both attributes, same value.
        v = defVar(ds, "cable", Float32, dims; fillvalue = -99999f0,
                   attrib = Dict("missing_value" => -99999f0))
        d = copy(data); d[3, 1, 2] = -99999f0
        v.var[:, :, :] = d
        # DLEM: no fill attribute (a -99999 is a value).
        v = defVar(ds, "dlem", Float64, dims)
        d = Float64.(data); d[1, 2, 1] = -99999.0
        v.var[:, :, :] = d
        # Float64 values that Float32 does not hold (rounding, subnormal, overflow).
        v = defVar(ds, "f64", Float64, dims; fillvalue = -1.0)
        d = reshape([0.1, 1 / 3, 1e-40, 3.5e38, -2.0^-149 * 0.75, 1 + 2.0^-24, 1 + 3 * 2.0^-25,
                     -1.0, 7.0, 1e300, -0.0, 0.2, 0.3, 0.7, 2.0^-126, 1e-45, 9.99, 123.456,
                     1e10 + 1, -5.5, 6.25, 0.0, 1 / 7, 22 / 7], Nx, Ny, Nt)
        v.var[:, :, :] = d
        # Packed Int16 with a Float32 scale factor (a CF view of eltype Float32).
        v = defVar(ds, "packed", Int16, dims; attrib = Dict("scale_factor" => 0.1f0, "add_offset" => 1f0))
        v.var[:, :, :] = Int16.(reshape(1:(Nx * Ny * Nt), Nx, Ny, Nt))
        # A larger static field for the allocation check.
        defDim(ds, "x", 200); defDim(ds, "y", 100)
        defVar(ds, "big", Float32, ("x", "y"))[:, :] = reshape(Float32.(1:20000) ./ 7, 200, 100)
    end

    NCDataset(path) do ds
        # missing_value only (VISIT-like): masked by MaskedAsZero and MaskedAsNaN.
        z = read_raw(ds, "visit", MaskedAsZero())
        @test z isa Array{Float64, 3}
        @test z[1, 1, 1] == 0.0 && z[4, 3, 2] == 0.0
        @test z[2, 1, 1] == Float64(data[2, 1, 1])
        n = read_raw(ds, "visit", MaskedAsNaN())
        @test isnan(n[1, 1, 1]) && isnan(n[4, 3, 2])
        @test count(isnan, n) == 2
        @test !any(==(-99999.0), z) && !any(==(-99999.0), n)
        # FillValueOnly ignores missing_value: the -99999 of a missing_value-only
        # variable passes through (that is why fluxes use the masking policies).
        @test read_raw(ds, "visit", FillValueOnly())[1, 1, 1] == -99999.0

        # IBIS land fraction: FillValueOnly keeps the valid zeros.
        fv = read_raw(ds, "ibis_fraction", FillValueOnly())
        @test count(isnan, fv) == 2
        @test count(==(0.0), fv) == 3
        mz = read_raw(ds, "ibis_fraction", MaskedAsZero())
        @test count(==(0.0), mz) == 5                  # the zeros were masked too
        @test count(isnan, read_raw(ds, "ibis_fraction", MaskedAsNaN())) == 5

        # NaN fill values.
        @test isnan(read_raw(ds, "elm", MaskedAsNaN())[2, 2, 1])
        @test read_raw(ds, "elm", MaskedAsZero())[2, 2, 1] == 0.0
        @test read_raw(ds, "cable", MaskedAsZero())[3, 1, 2] == 0.0
        @test read_raw(ds, "dlem", MaskedAsZero())[1, 2, 1] == -99999.0

        # Indices, as ds[var][idx...]; all-integer indices give a 0-d array.
        @test isequal(read_raw(ds, "cable", MaskedAsNaN(), :, :, 2),
                      read_raw(ds, "cable", MaskedAsNaN())[:, :, 2])
        s = read_raw(ds, "cable", MaskedAsZero(), 2, 3, 1)
        @test s isa Array{Float64, 0} && s[] == Float64(data[2, 3, 1])
        @test_throws ArgumentError read_raw(ds, "no_such_variable", MaskedAsZero())

        # MaskedAsZero reproduces the runtime's nomissing(…, 0) bit for bit.
        for name in ("visit", "elm", "cable", "dlem"), t in 1:Nt
            runtime = Float32.(nomissing(ds[name][:, :, t], 0f0))       # FT = Float32 runtime
            @test isequal(Float32.(read_raw(ds, name, MaskedAsZero(), :, :, t)), runtime)
            @test isequal(read_raw(ds, name, MaskedAsZero(), :, :, t),
                          Float64.(nomissing(ds[name][:, :, t], 0.0)))
        end
    end

    # Element type T: each stored value converted once, straight to T, exactly
    # as the runtime's T.(nomissing(ds[name][idx...], zero(T))); no Float64 copy.
    raw_bytes(T, ds, name) = (read_raw(T, ds, name, MaskedAsZero());
                              @allocated read_raw(T, ds, name, MaskedAsZero()))
    NCDataset(path) do ds
        for FT in (Float32, Float64), name in ("visit", "elm", "cable", "dlem", "f64", "packed"), t in 1:Nt
            r = read_raw(FT, ds, name, MaskedAsZero(), :, :, t)
            @test r isa Array{FT, 2}
            @test isequal(r, FT.(nomissing(ds[name][:, :, t], zero(FT))))
            @test isequal(read_raw(FT, ds, name, MaskedAsZero()), FT.(nomissing(ds[name][:, :, :], zero(FT))))
        end
        @test read_raw(Float64, ds, "f64", MaskedAsNaN()) isa Array{Float64, 3}
        @test isequal(read_raw(ds, "f64", MaskedAsNaN()), read_raw(Float64, ds, "f64", MaskedAsNaN()))
        n32 = read_raw(Float32, ds, "visit", MaskedAsNaN())
        @test n32 isa Array{Float32, 3} && isnan(n32[1, 1, 1]) && count(isnan, n32) == 2
        @test read_raw(Float32, ds, "f64", MaskedAsZero(), 4, 1, 1) isa Array{Float32, 0}
        @test read_raw(Float32, ds, "f64", MaskedAsZero(), 4, 1, 1)[] == Inf32     # 3.5e38 overflows
        # A Float32 read allocates no Float64 array: at least 4 bytes per value
        # less than the Float64 read (whose Float32 conversion would cost more).
        @test raw_bytes(Float32, ds, "big") + 4 * 20000 <= raw_bytes(Float64, ds, "big")
        @test_throws ArgumentError read_raw(Float32, ds, "no_such_variable", MaskedAsZero())
    end

    # The policy, not the dataset's masking value, decides.
    NCDataset(path; maskingvalue = NaN) do ds
        @test read_raw(ds, "visit", MaskedAsZero())[1, 1, 1] == 0.0
        @test isnan(read_raw(ds, "visit", MaskedAsNaN())[1, 1, 1])
    end
end
