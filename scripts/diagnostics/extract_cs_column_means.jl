#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Extract per-frame column-mean dry VMR (XCO2, ppm) for co2_ocean/natural/fossil
# from a set of daily ATMSNAP1 files into ONE compact NetCDF (cell, time), plus
# per-cell CS lon/lat. Avoids writing full 3-D NetCDFs. For downstream viz.
#
#   julia --project=. scripts/diagnostics/extract_cs_column_means.jl <dir> <out.nc>
# ---------------------------------------------------------------------------
using JSON3, Printf, NCDatasets
include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Grids: CubedSphereMesh, EquiangularCubedSphereDefinition,
    GMAOCubedSphereDefinition, GnomonicPanelConvention, GEOSNativePanelConvention,
    panel_cell_center_lonlat
using .AtmosTransport.Output: column_mean_mixing_ratio

const MAGIC = "ATMSNAP1"
const OCEAN_CARRIER_PPM = 400.0f0
_pc(t) = t == "gnomonic" ? GnomonicPanelConvention() : t == "geos_native" ? GEOSNativePanelConvention() : error("pc $t")
_df(t, c) = t == "equiangular_gnomonic" ? EquiangularCubedSphereDefinition(convention = c) :
            t == "gmao_equal_distance" ? GMAOCubedSphereDefinition(convention = c) : error("def $t")
_ft(t) = t == "Float32" ? Float32 : Float64
_hdr(io) = (String(read(io, length(MAGIC))) == MAGIC || error("magic"); JSON3.read(String(read(io, Int(read(io, UInt64)))), Dict{String, Any}))

const TRACERS = ["co2_ocean", "co2_natural", "co2_fossil"]

function main()
    dir, out = ARGS[1], ARGS[2]
    files = sort(filter(f -> endswith(f, ".atmsnap"), readdir(dir; join = true)))
    isempty(files) && error("no atmsnap in $dir")

    h0 = open(_hdr, files[1], "r")
    conv = _pc(String(h0["grid"]["panel_convention"]))
    radius = Float64(get(h0["grid"], "planet_radius_m", AtmosTransport.Parameters.EARTH_RADIUS))
    mesh = CubedSphereMesh(; FT = _ft(String(h0["float_dtype"])), Nc = Int(h0["grid"]["Nc"]),
                           Hp = 0, definition = _df(String(h0["grid"]["definition"]), conv), radius)
    Nc = Int(h0["grid"]["Nc"]); Nz = Int(h0["grid"]["Nz"])
    cs_lon = vcat((vec(panel_cell_center_lonlat(mesh, p)[1]) for p in 1:6)...)
    cs_lat = vcat((vec(panel_cell_center_lonlat(mesh, p)[2]) for p in 1:6)...)
    cs_area = vcat((vec(mesh.cell_areas) for _ in 1:6)...)
    ncell = length(cs_lon)

    # first pass: total frame count
    times = Float64[]
    for f in files
        h = open(_hdr, f, "r"); append!(times, Float64.(h["times_hours"]))
    end
    nt = length(times)
    @info "frames" nt ncell first=times[1] last=times[end]

    data = Dict(t => Array{Float32}(undef, ncell, nt) for t in TRACERS)
    pf = Nc * Nc * Nz
    gi = 0
    for f in files
        open(f, "r") do io
            h = _hdr(io); fields = String.(h["fields"]); nfr = Int(h["n_frames"])
            seek(io, Int(h["payload_offset"]))
            for _ in 1:nfr
                gi += 1
                buffers = Dict{String, NTuple{6, Array{Float32, 3}}}()
                for fn in fields
                    keep = fn == "air_mass" || fn in TRACERS
                    panels = ntuple(6) do _
                        b = Vector{Float32}(undef, pf); read!(io, b)
                        keep ? reshape(b, (Nc, Nc, Nz)) : reshape(Float32[], (0, 0, 0))
                    end
                    keep && (buffers[fn] = panels)
                end
                for t in TRACERS
                    cm = column_mean_mixing_ratio(buffers["air_mass"], buffers[t])  # 6-panel (Nc,Nc)
                    values_ppm = vcat((vec(cm[p]) for p in 1:6)...) .* 1f6
                    t == "co2_ocean" && (values_ppm .-= OCEAN_CARRIER_PPM)
                    data[t][:, gi] .= values_ppm
                end
            end
        end
        @info "read" file=basename(f)
    end

    ds = NCDataset(out, "c")
    defDim(ds, "cell", ncell); defDim(ds, "time", nt)
    defVar(ds, "cs_lon", Float64, ("cell",))[:] = cs_lon
    defVar(ds, "cs_lat", Float64, ("cell",))[:] = cs_lat
    area_var = defVar(ds, "cs_area", Float64, ("cell",))
    area_var.attrib["units"] = "m2"
    area_var[:] = cs_area
    defVar(ds, "time_hours", Float64, ("time",))[:] = times
    for t in TRACERS
        v = defVar(ds, t, Float32, ("cell", "time"))
        v.attrib["units"] = "ppm (dry VMR)"
        t == "co2_ocean" &&
            (v.attrib["carrier_removed_ppm"] = Float64(OCEAN_CARRIER_PPM))
        v[:, :] = data[t]
    end
    ds.attrib["note"] = "column-mean dry VMR (XCO2) per CS cell; " *
                        "co2_ocean has its 400 ppm transport carrier removed; " *
                        "run start 2021-12-01T00:00"
    close(ds)
    @info "wrote" out size_mb=round(filesize(out) / 1e6, digits = 1)
end
main()
