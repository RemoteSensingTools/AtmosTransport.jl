#!/usr/bin/env julia
# Extract one tracer's panel-native column-mean dry VMR from daily ATMSNAP1
# files into a compact (cell,time) NetCDF for diagnostics and animation.

using Dates
using JSON3
using NCDatasets

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Grids: CubedSphereMesh, EquiangularCubedSphereDefinition,
    GMAOCubedSphereDefinition, GnomonicPanelConvention, GEOSNativePanelConvention,
    panel_cell_center_lonlat
using .AtmosTransport.Output: column_mean_mixing_ratio

const MAGIC = "ATMSNAP1"

panel_convention(tag) =
    tag == "gnomonic" ? GnomonicPanelConvention() :
    tag == "geos_native" ? GEOSNativePanelConvention() :
    error("unsupported panel convention $tag")

definition(tag, convention) =
    tag == "equiangular_gnomonic" ? EquiangularCubedSphereDefinition(convention = convention) :
    tag == "gmao_equal_distance" ? GMAOCubedSphereDefinition(convention = convention) :
    error("unsupported cubed-sphere definition $tag")

float_type(tag) = tag == "Float32" ? Float32 : tag == "Float64" ? Float64 :
    error("unsupported snapshot float type $tag")

function read_header(io)
    String(read(io, length(MAGIC))) == MAGIC || error("not an ATMSNAP1 file")
    nbytes = Int(read(io, UInt64))
    return JSON3.read(String(read(io, nbytes)), Dict{String, Any})
end

function parse_args()
    length(ARGS) >= 3 || error(
        "usage: extract_cs_xco2.jl INPUT_DIR OUTPUT.nc TRACER [CARRIER_PPM] [START_ISO]")
    input_dir = abspath(expanduser(ARGS[1]))
    output = abspath(expanduser(ARGS[2]))
    tracer = String(ARGS[3])
    carrier_ppm = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.0
    start_time = length(ARGS) >= 5 ? DateTime(ARGS[5]) : DateTime(2021, 12, 1)
    return input_dir, output, tracer, carrier_ppm, start_time
end

function main()
    input_dir, output, tracer, carrier_ppm, start_time = parse_args()
    files = sort(filter(f -> endswith(f, ".atmsnap"), readdir(input_dir; join = true)))
    isempty(files) && error("no .atmsnap files under $input_dir")

    first_header = open(read_header, first(files), "r")
    grid_header = first_header["grid"]
    Nc = Int(grid_header["Nc"])
    Nz = Int(grid_header["Nz"])
    conv = panel_convention(String(grid_header["panel_convention"]))
    mesh = CubedSphereMesh(
        ; FT = float_type(String(first_header["float_dtype"])), Nc, Hp = 0,
        definition = definition(String(grid_header["definition"]), conv))
    lon = vcat((vec(panel_cell_center_lonlat(mesh, p)[1]) for p in 1:6)...)
    lat = vcat((vec(panel_cell_center_lonlat(mesh, p)[2]) for p in 1:6)...)
    area = vcat((vec(mesh.cell_areas) for _ in 1:6)...)
    ncell = length(lon)

    times = Float64[]
    headers = Dict{String, Any}[]
    for file in files
        header = open(read_header, file, "r")
        push!(headers, header)
        append!(times, Float64.(header["times_hours"]))
    end
    length(unique(times)) == length(times) || error("snapshot times are not unique")
    issorted(times) || error("snapshot times are not ascending")

    mkpath(dirname(output))
    ds = NCDataset(output, "c")
    try
        defDim(ds, "cell", ncell)
        defDim(ds, "time", length(times))
        defVar(ds, "cs_lon", Float64, ("cell",),
               attrib = Dict("units" => "degrees_east"))[:] = lon
        defVar(ds, "cs_lat", Float64, ("cell",),
               attrib = Dict("units" => "degrees_north"))[:] = lat
        defVar(ds, "cs_area", Float64, ("cell",),
               attrib = Dict("units" => "m2"))[:] = area
        time_var = defVar(ds, "time_hours", Float64, ("time",),
                          attrib = Dict("units" =>
                              "hours since $(Dates.format(start_time, dateformat"yyyy-mm-dd HH:MM:SS")) UTC"))
        time_var[:] = times
        xco2_var = defVar(ds, "xco2", Float32, ("cell", "time"),
                          attrib = Dict("units" => "ppm", "long_name" =>
                              "column-mean dry-air mole fraction of $tracer"))
        anomaly_var = defVar(ds, "xco2_anomaly", Float32, ("cell", "time"),
                             attrib = Dict("units" => "ppm", "carrier_removed_ppm" => carrier_ppm,
                                 "long_name" => "XCO2 signal relative to positivity carrier"))
        global_var = defVar(ds, "global_mean_xco2_anomaly", Float64, ("time",),
                            attrib = Dict("units" => "ppm", "weighting" => "cubed-sphere cell area"))

        panel_values = Nc * Nc * Nz
        global_index = 0
        for (file, header) in zip(files, headers)
            fields = String.(header["fields"])
            tracer in fields || error("tracer '$tracer' absent from $file; fields=$fields")
            "air_mass" in fields || error("air_mass absent from $file")
            nframes = Int(header["n_frames"])
            open(file, "r") do io
                seek(io, Int(header["payload_offset"]))
                for _ in 1:nframes
                    global_index += 1
                    kept = Dict{String, NTuple{6, Array{Float32, 3}}}()
                    for field in fields
                        panels = ntuple(6) do _
                            buffer = Vector{Float32}(undef, panel_values)
                            read!(io, buffer)
                            reshape(buffer, Nc, Nc, Nz)
                        end
                        field in ("air_mass", tracer) && (kept[field] = panels)
                    end
                    cm = column_mean_mixing_ratio(kept["air_mass"], kept[tracer])
                    absolute_ppm = vcat((vec(cm[p]) for p in 1:6)...) .* 1.0f6
                    anomaly_ppm = absolute_ppm .- Float32(carrier_ppm)
                    xco2_var[:, global_index] = absolute_ppm
                    anomaly_var[:, global_index] = anomaly_ppm
                    global_var[global_index] = sum(Float64.(anomaly_ppm) .* area) / sum(area)
                end
            end
            @info "extracted XCO2" file = basename(file) frames = nframes
        end

        ds.attrib["source_directory"] = input_dir
        ds.attrib["tracer"] = tracer
        ds.attrib["carrier_removed_ppm"] = carrier_ppm
        ds.attrib["grid"] = "GEOS-native C$(Nc)"
        ds.attrib["mass_basis"] = String(first_header["mass_basis"])
        ds.attrib["start_time"] = string(start_time)
    finally
        close(ds)
    end
    @info "wrote compact XCO2 time series" output frames = length(times) cells = ncell
end

main()
