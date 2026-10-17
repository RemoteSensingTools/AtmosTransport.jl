#!/usr/bin/env julia

# Build the per-grid model surface height dataset for observation placement
# (`[output.observations].surface_height_file`): the surface geopotential (or
# height) of a met source on its latitude-longitude grid, divided by the
# standard gravity and regridded conservatively (area-weighted mean per model
# cell) onto the mesh of a transport binary. The file is written by
# `GriddedIO.write_model_surface_height` and read by the runtime through
# `GriddedIO.read_model_surface_height`.
#
# Use the surface field of the met source the binary was built from, so the
# model ground matches the surface pressure the run carries:
#   MERRA-2 / GEOS: PHIS (m2 s-2) of any inst3_3d_asm_Nv file (time-invariant);
#   ERA5: the invariant surface geopotential `z` (m2 s-2).
#
#   julia --project=. scripts/preprocessing/build_model_surface_height.jl \
#       --binary <any transport binary of the target grid> \
#       --source ~/data/AtmosTransport/met/merra2/M2I3NVASM/2021/12/MERRA2_400.inst3_3d_asm_Nv.20211201.nc4 \
#       --variable PHIS --out <dir>/surface_height_c90_merra2.nc \
#       [--compare <file on the target grid> --compare-variable Met_PHIS]
#
# `--compare` reports the difference to an independent field already on the
# target cubed sphere (for example GEOS-Chem's `Met_PHIS` diagnostic, the
# geopotential height GCHP uses on the same C<N> grid); its `lons`/`lats` must
# be the mesh's.

using ArgParse, Printf, Statistics
using NCDatasets
using AtmosTransport
using AtmosTransport.GriddedIO: find_coordinate, LongitudeAxis, LatitudeAxis, read_raw, MaskedAsNaN,
                                latlon_orientation, orient, orient_coordinates, source_latlon_mesh,
                                write_model_surface_height
using AtmosTransport.Regridding: build_regridder, apply_regridder!, default_regrid_cache_dir

const MD = AtmosTransport.MetDrivers
const G0 = AtmosTransport.Parameters.STANDARD_GRAVITY

function settings()
    s = ArgParseSettings(description = "Per-grid model surface height for observation placement")
    @add_arg_table! s begin
        "--binary"
            required = true
            help = "transport binary that defines the target mesh"
        "--source"
            required = true
            help = "lat-lon NetCDF file with the surface geopotential or height"
        "--variable"
            required = true
            help = "variable name (e.g. PHIS, z)"
        "--time-index"
            arg_type = Int
            default = 1
            help = "time index when the variable has a time dimension"
        "--out"
            required = true
        "--compare"
            default = ""
            help = "optional field on the target cubed sphere to compare with"
        "--compare-variable"
            default = "Met_PHIS"
    end
    return s
end

# Units → metres per unit of the stored value: heights as they are,
# geopotential divided by the standard gravity (GEOS MAPL_GRAV and ECMWF both
# use 9.80665 m s-2).
function metres_per_unit(units, label)
    u = replace(lowercase(strip(String(something(units, "")))), r"[\s\*\^\+]" => "")
    u in ("m", "meter", "meters", "metre", "metres") && return 1.0
    u in ("m2s-2", "m2/s2", "m2/s-2") && return 1 / G0
    throw(ArgumentError("$(label) has units $(repr(units)); expected metres or m2 s-2 (geopotential)"))
end

function read_source(path, name, time_index)
    NCDataset(path, "r") do ds
        haskey(ds, name) || throw(ArgumentError("$(path) has no variable $(name)"))
        var = ds[name]
        lon_name = find_coordinate(ds, LongitudeAxis(); length = size(var, 1))
        lat_name = find_coordinate(ds, LatitudeAxis(); length = size(var, 2))
        scale = metres_per_unit(get(var.attrib, "units", nothing), "$(path):$(name)")
        idx = ndims(var) == 2 ? () : ndims(var) == 3 ? (:, :, time_index) :
              throw(ArgumentError("$(name) must be (lon, lat) or (lon, lat, time); got $(ndims(var)) dimensions"))
        raw = read_raw(ds, name, MaskedAsNaN(), idx...)
        all(isfinite, raw) || throw(ArgumentError("$(path):$(name) has fill values"))
        lon, lat = read_raw(ds, lon_name, MaskedAsNaN()), read_raw(ds, lat_name, MaskedAsNaN())
        o = latlon_orientation(lon, lat)
        olon, olat = orient_coordinates(lon, lat, o)
        return orient(raw, o) .* scale, olon, olat
    end
end

# Flat regridder order → the writer's layout.
to_layout(flat, mesh::CubedSphereMesh) =
    cat(AtmosTransport.Preprocessing.unpack_flat_to_panels_2d!(ntuple(_ -> Matrix{Float64}(undef, mesh.Nc, mesh.Nc), 6),
                                                                flat, mesh.Nc)...; dims = 3)
to_layout(flat, mesh::LatLonMesh) = reshape(flat, mesh.Nx, mesh.Ny)
to_layout(flat, mesh::ReducedGaussianMesh) = flat

area_mean(x, areas) = sum(x .* areas) / sum(areas)

unit_xyz(lon, lat) = (cosd(lat) * cosd(lon), cosd(lat) * sind(lon), sind(lat))

function compare(field, mesh::CubedSphereMesh, path, name)
    ref = NCDataset(path, "r") do ds
        lons, lats = read_raw(ds, "lons", MaskedAsNaN()), read_raw(ds, "lats", MaskedAsNaN())
        for p in 1:6
            mlon, mlat = AtmosTransport.Grids.panel_cell_center_lonlat(mesh, p)
            chord = maximum(sqrt(sum(abs2, unit_xyz(lons[i, j, p], lats[i, j, p]) .- unit_xyz(mlon[i, j], mlat[i, j])))
                            for i in 1:mesh.Nc, j in 1:mesh.Nc)
            chord < 1e-5 || throw(ArgumentError(
                "$(path) panel $(p) centres are up to $(rad2deg(chord))° from the mesh centres"))
        end
        var = ds[name]
        scale = metres_per_unit(get(var.attrib, "units", nothing), "$(path):$(name)")
        raw = read_raw(ds, name, MaskedAsNaN())
        (ndims(raw) == 4 ? raw[:, :, :, 1] : raw) .* scale
    end
    area = cat((Float64.(mesh.cell_areas) for _ in 1:6)...; dims = 3)
    d = field .- ref
    @printf("compare with %s:%s (m): mean(ours) %.2f  mean(ref) %.2f  mean diff %.3f  rms %.2f  max |diff| %.1f  corr %.6f\n",
            basename(path), name, area_mean(field, area), area_mean(ref, area), area_mean(d, area),
            sqrt(area_mean(d .^ 2, area)), maximum(abs, d), cor(vec(field), vec(ref)))
    land = ref .> 50
    @printf("  cells with ref > 50 m: %d, rms diff %.2f m, 99th percentile |diff| %.1f m\n",
            count(land), sqrt(mean(d[land] .^ 2)), quantile(abs.(d[land]), 0.99))
    return nothing
end
compare(field, mesh, path, name) = throw(ArgumentError("--compare supports cubed-sphere targets only"))

"""
    build_model_surface_height(source, variable, mesh, out; time_index = 1, cache_dir, attributes = Dict())

Regrid the surface geopotential or height `variable` of the lat-lon file
`source` conservatively onto `mesh` (m above sea level) and write it to `out`
with `write_model_surface_height`. Returns `(path, field, source_mean,
target_mean)`, the area means in m.
"""
function build_model_surface_height(source::AbstractString, variable::AbstractString, mesh, out::AbstractString;
                                    time_index::Integer = 1, cache_dir = default_regrid_cache_dir(),
                                    attributes = Dict{String, Any}())
    height, lon, lat = read_source(source, variable, time_index)
    src_mesh = source_latlon_mesh(lon, lat; radius = mesh.radius)
    regridder = build_regridder(src_mesh, mesh; cache_dir)
    flat = zeros(Float64, length(regridder.dst_areas))
    apply_regridder!(flat, regridder, vec(height))
    field = to_layout(flat, mesh)
    src_mean = area_mean(vec(height), regridder.src_areas)
    dst_mean = area_mean(flat, regridder.dst_areas)
    @printf("%s → %s: area-mean height source %.4f m, target %.4f m (rel. diff %.2e); range %.1f .. %.1f m\n",
            basename(source), summary(mesh), src_mean, dst_mean, abs(dst_mean - src_mean) / max(abs(src_mean), eps()),
            minimum(flat), maximum(flat))
    path = write_model_surface_height(out, mesh, field; attributes = merge(Dict{String, Any}(
        "source_file" => abspath(source), "source_variable" => String(variable),
        "source_time_index" => Int(time_index),
        "regridding" => "conservative (area-weighted mean per model cell), Regridding.build_regridder",
        "geopotential_to_height" => "divided by STANDARD_GRAVITY = $(G0) m s-2 when the source is geopotential"),
        Dict{String, Any}(String(k) => v for (k, v) in attributes)))
    return (; path, field, source_mean = src_mean, target_mean = dst_mean)
end

function main(argv)
    args = parse_args(argv, settings())
    reader = MD.TransportBinaryReader(args["binary"]; FT = Float32)
    mesh = MD.load_grid(reader; FT = Float64).horizontal
    close(reader)
    result = build_model_surface_height(args["source"], args["variable"], mesh, args["out"];
                                        time_index = args["time-index"],
                                        attributes = Dict("target_binary" => abspath(args["binary"])))
    @info "wrote $(result.path)"
    isempty(args["compare"]) || compare(result.field, mesh, args["compare"], args["compare-variable"])
    return result.path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
