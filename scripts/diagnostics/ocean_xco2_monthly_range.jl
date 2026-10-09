#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Report the monthly evolution of column-mean dry VMR (XCO2) for each tracer in
# a set of daily ATMSNAP1 snapshots, focusing on the ocean (ECCO-Darwin) tracer.
# Reads only the LAST frame of each daily file (targeted seek) for speed.
#
# Usage:
#   julia --project=. scripts/diagnostics/ocean_xco2_monthly_range.jl <dir-or-files...>
# ---------------------------------------------------------------------------
using JSON3, Printf
include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Grids: CubedSphereMesh, CubedSphereDefinition,
    EquiangularCubedSphereDefinition, GMAOCubedSphereDefinition,
    GnomonicPanelConvention, GEOSNativePanelConvention,
    AbstractCubedSpherePanelConvention, panel_cell_center_lonlat
using .AtmosTransport.Output: column_mean_mixing_ratio

const MAGIC = "ATMSNAP1"
const OCEAN_CARRIER_VMR = 4.0e-4

_panel_conv(t) = t == "gnomonic" ? GnomonicPanelConvention() :
                 t == "geos_native" ? GEOSNativePanelConvention() :
                 error("bad panel_convention $(repr(t))")
function _def(t, conv)
    t == "equiangular_gnomonic" && return EquiangularCubedSphereDefinition(convention = conv)
    t == "gmao_equal_distance"  && return GMAOCubedSphereDefinition(convention = conv)
    error("bad definition $(repr(t))")
end
_ftype(t) = t == "Float32" ? Float32 : t == "Float64" ? Float64 : error("bad float $t")

function _header(io)
    String(read(io, length(MAGIC))) == MAGIC || error("bad magic")
    hs = read(io, UInt64)
    return JSON3.read(String(read(io, Int(hs))), Dict{String, Any})
end

# Read air_mass + requested tracers from the LAST frame only.
function _last_frame(path, want)
    open(path, "r") do io
        h = _header(io)
        Nc, Nz = Int(h["grid"]["Nc"]), Int(h["grid"]["Nz"])
        fields = String.(h["fields"])
        nfr = Int(h["n_frames"])
        pf = Nc * Nc * Nz
        frame_bytes = length(fields) * 6 * pf * 4
        seek(io, Int(h["payload_offset"]) + (nfr - 1) * frame_bytes)
        out = Dict{String, NTuple{6, Array{Float32, 3}}}()
        for fn in fields
            keep = fn == "air_mass" || fn in want
            panels = ntuple(6) do _
                buf = Vector{Float32}(undef, pf)
                read!(io, buf)
                return keep ? reshape(buf, (Nc, Nc, Nz)) : reshape(Float32[], (0, 0, 0))
            end
            keep && (out[fn] = panels)
        end
        return out, h, Float64(h["times_hours"][nfr])
    end
end

function main()
    args = ARGS
    files = String[]
    for a in args
        isdir(a) ? append!(files, sort(filter(f -> endswith(f, ".atmsnap"),
                          readdir(a; join = true)))) : push!(files, a)
    end
    isempty(files) && error("no atmsnap files")
    tracers = ["co2_ocean", "co2_natural", "co2_fossil"]

    # mesh (for cell areas + coords) from the first file's header
    h0 = open(_header, files[1], "r")
    conv = _panel_conv(String(h0["grid"]["panel_convention"]))
    def = _def(String(h0["grid"]["definition"]), conv)
    mesh = CubedSphereMesh(; FT = _ftype(String(h0["float_dtype"])),
                           Nc = Int(h0["grid"]["Nc"]), Hp = 0, definition = def,
                           radius = Float64(get(h0["grid"], "planet_radius_m", AtmosTransport.Parameters.EARTH_RADIUS)))
    area = mesh.cell_areas                      # (Nc, Nc), identical per panel
    Atot = 6 * sum(area)

    println("day        ", join([@sprintf("%-34s", t) for t in tracers]))
    println("           ", join(["  min      max      range   amean" for _ in tracers]))
    last_fields = nothing
    for f in files
        flds, _, _ = _last_frame(f, tracers)
        last_fields = flds
        cols = String[]
        for t in tracers
            cm = column_mean_mixing_ratio(flds["air_mass"], flds[t])  # 6-panel (Nc,Nc) VMR
            t == "co2_ocean" && foreach(p -> (cm[p] .-= OCEAN_CARRIER_VMR), 1:6)
            mn = minimum(minimum, cm) * 1e6
            mx = maximum(maximum, cm) * 1e6
            am = sum(p -> sum(cm[p] .* area), 1:6) / Atot * 1e6        # area-wtd mean, ppm
            push!(cols, @sprintf("%+7.3f  %+7.3f  %6.3f  %+7.4f", mn, mx, mx - mn, am))
        end
        tag = basename(f)[end-12:end-8]   # YYYYMMDD -> ...MMDD
        println(rpad(basename(f)[end-11:end-8], 11), join(cols, "  "))
    end

    # final-day spatial sign check for the ocean tracer
    println("\n[final day] co2_ocean XCO2 (ppm) at key sites:")
    cm = column_mean_mixing_ratio(last_fields["air_mass"], last_fields["co2_ocean"])
    foreach(p -> (cm[p] .-= OCEAN_CARRIER_VMR), 1:6)
    lons = ntuple(p -> panel_cell_center_lonlat(mesh, p)[1], 6)
    lats = ntuple(p -> panel_cell_center_lonlat(mesh, p)[2], 6)
    function at(lon, lat)
        lon = mod(lon, 360.0); best = (Inf, 0.0); bv = NaN
        for p in 1:6, j in axes(cm[p], 2), i in axes(cm[p], 1)
            dlon = abs(mod(lons[p][i, j], 360.0) - lon); dlon = min(dlon, 360 - dlon)
            d = dlon^2 + (lats[p][i, j] - lat)^2
            d < best[1] && (best = (d, 0.0); bv = cm[p][i, j] * 1e6)
        end
        return bv
    end
    for (nm, lo, la) in [("EqPacific(src,+)", -140, 0), ("EqAtlantic(src,+)", -20, 0),
                         ("N.Atlantic(sink,-)", -30, 55), ("N.Pacific(sink,-)", -150, 45),
                         ("SouthernOcean", 0, -60)]
        @printf("  %-20s %+8.4f\n", nm, at(lo, la))
    end
end

main()
