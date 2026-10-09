#!/usr/bin/env julia

# Build CATRINE-protocol (D7.1) surface-flux series on a cubed-sphere transport
# grid, as `kind = "cs_native"` time-varying inputs (NetCDF (time, nf, Ydim,
# Xdim), kg species m-2 s-1, one slice per calendar period, held stepwise).
#
# Protocol choices that the built-in flux kinds cannot express:
#   * fossil CO2 starts on 2022-01-01 (null state then), so its December 2021
#     spin-up slice is zero; GridFED monthly TOTAL afterwards;
#   * Rn-222 uses the monthly Zhang et al. (2021) climatology for every month;
#   * SF6 uses the EDGAR v8 2022 map for every year, scaled so each year's
#     global total matches the NOAA global growth rate (sf6_gr_gl.txt);
#     the December 2021 spin-up uses the 2021 growth rate.
#
# Regridding reuses the model's own loaders and conservative regridder, so
# units handling is identical to the built-in kinds. Global totals are printed
# for verification.
#
#   julia --project=. scripts/preprocessing/catrine_protocol_cs_fluxes.jl \
#       --binary <any C90 transport binary> --outdir <dir>

using ArgParse, Dates, NCDatasets, Printf
using AtmosTransport

const MD = AtmosTransport.MetDrivers
const ICIO = AtmosTransport.Models.InitialConditionIO

const CATRINE = joinpath(homedir(), "data", "AtmosTransport", "catrine", "Emissions")
const ORIGIN = DateTime(2021, 12, 1)
const M_SF6 = 0.146055            # kg/mol
# ppt -> kg conversion for the NOAA growth rates: the dry-air molar mass and the
# 5.135e18 kg global dry-air mass pin of the preprocessing give 25.89 kt/ppt.
const M_DRY_AIR = AtmosTransport.Parameters.DRY_AIR_MOLAR_MASS   # kg/mol
const DRY_AIR_MASS = 5.135e18     # kg

function settings()
    s = ArgParseSettings(description = "CATRINE protocol cs_native flux series")
    @add_arg_table! s begin
        "--binary"
            required = true
            help = "cubed-sphere transport binary that defines the target mesh"
        "--outdir"
            required = true
        "--last-month"
            default = "2023-12"
            help = "last month (YYYY-MM) of the series"
    end
    return s
end

# Per-cell density (kg m-2 s-1) on the cubed sphere, (Nc, Nc, 6), from one
# static surface-flux configuration of a built-in kind.
function cs_density(cfg::Dict{String, Any}, mesh)
    source = ICIO._load_file_surface_flux_field(cfg, Float64)
    rate = ICIO._conservative_surface_flux_rate(source, mesh, Float64)   # kg/s per cell
    Nc = mesh.Nc
    panels = ntuple(_ -> Matrix{Float64}(undef, Nc, Nc), 6)
    AtmosTransport.Preprocessing.unpack_flat_to_panels_2d!(panels, rate, Nc)
    area = Float64.(mesh.cell_areas)
    density = Array{Float64}(undef, Nc, Nc, 6)
    for p in 1:6
        density[:, :, p] .= panels[p] ./ area
    end
    return density, source.native_total_mass_rate
end

global_rate(density, mesh) = sum(density[:, :, p] .* mesh.cell_areas for p in 1:6) |> sum

function write_series(path, mesh, starts::Vector{DateTime}, slices::Vector{Array{Float64, 3}};
                      name, long_name, attributes = Dict{String, Any}())
    Nc = mesh.Nc
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        defDim(ds, "Xdim", Nc); defDim(ds, "Ydim", Nc); defDim(ds, "nf", 6); defDim(ds, "time", length(starts))
        defVar(ds, "time", Float64, ("time",);
               attrib = Dict("units" => "seconds since 2021-12-01 00:00:00",
                             "calendar" => "proleptic_gregorian",
                             "long_name" => "start of the period each slice is held for"))[:] =
            [Dates.value(t - ORIGIN) / 1000 for t in starts]
        area = defVar(ds, "cell_area", Float64, ("Xdim", "Ydim", "nf"); attrib = Dict("units" => "m2"))
        for p in 1:6
            area[:, :, p] = Float64.(mesh.cell_areas)
        end
        v = defVar(ds, name, Float32, ("Xdim", "Ydim", "nf", "time");
                   attrib = Dict("units" => "kg m-2 s-1", "long_name" => long_name))
        for (k, slice) in enumerate(slices)
            v[:, :, :, k] = Float32.(slice)
        end
        ds.attrib["protocol"] = "CATRINE D7.1 (2024-05-20)"
        ds.attrib["grid"] = "C$(Nc) $(mesh.convention) panels, cell areas from the runtime mesh"
        ds.attrib["temporal_scheme"] = "stepwise: each slice holds from its time until the next"
        for (key, value) in attributes
            ds.attrib[key] = value
        end
    end
    return path
end

month_starts(first::Date, last::Date) = [DateTime(d) for d in first:Month(1):last]

function noaa_sf6_growth()
    growth = Dict{Int, Float64}()
    for line in eachline(joinpath(CATRINE, "sf6_gr_gl.txt"))
        startswith(strip(line), "#") && continue
        fields = split(strip(line))
        length(fields) >= 2 || continue
        year = tryparse(Int, fields[1]); rate = tryparse(Float64, fields[2])
        year === nothing || rate === nothing || (growth[year] = rate)
    end
    return growth
end

function main(argv)
    args = parse_args(argv, settings())
    outdir = mkpath(abspath(args["outdir"]))
    last_month = Date(args["last-month"], dateformat"yyyy-mm")
    reader = MD.TransportBinaryReader(args["binary"]; FT = Float32)
    mesh = MD.load_grid(reader; FT = Float64, Hp = 0).horizontal
    close(reader)
    months = month_starts(Date(2021, 12, 1), last_month)
    @info "Target mesh C$(mesh.Nc); $(length(months)) monthly slices from $(first(months)) to $(last(months))"

    # -- fossil CO2: zero in the December 2021 spin-up, GridFED monthly after.
    fossil = Array{Float64, 3}[]
    for t in months
        if t < DateTime(2022, 1, 1)
            push!(fossil, zeros(mesh.Nc, mesh.Nc, 6))
            continue
        end
        file = joinpath(CATRINE, "gridfed", "GCP-GridFEDv2024.0_$(year(t)).short.nc")
        isfile(file) || error("no GridFED file for $(year(t)): $file")
        density, native = cs_density(Dict{String, Any}("kind" => "gridfed_fossil_co2", "file" => file,
                                                       "time_index" => month(t)), mesh)
        push!(fossil, density)
        @printf("fossil %s  %.6e kg/s (native %.6e)\n", Dates.format(t, "yyyy-mm"), global_rate(density, mesh), native)
    end
    write_series(joinpath(outdir, "catrine_fossil_co2_c$(mesh.Nc).nc"), mesh, months, fossil;
                 name = "fossil_co2", long_name = "GridFEDv2024.0 TOTAL CO2; zero before 2022-01-01",
                 attributes = Dict("source" => "GCP-GridFEDv2024.0_{2022,2023}.short.nc"))

    # -- Rn-222: monthly climatology for every month.
    rn = Array{Float64, 3}[]
    for t in months
        density, native = cs_density(Dict{String, Any}("kind" => "zhang_rn222", "time_index" => month(t)), mesh)
        push!(rn, density)
        @printf("rn222  %s  %.6e kg/s (native %.6e)\n", Dates.format(t, "yyyy-mm"), global_rate(density, mesh), native)
    end
    write_series(joinpath(outdir, "catrine_rn222_c$(mesh.Nc).nc"), mesh, months, rn;
                 name = "rn222", long_name = "Zhang et al. (2021) monthly Rn-222 climatology",
                 attributes = Dict("source" => "Rn222_Emis_Zhang_Liu_et_al_05x05_mass.nc"))

    # -- SF6: EDGAR 2022 map, yearly totals from NOAA growth rates.
    growth = noaa_sf6_growth()
    kg_per_ppt = DRY_AIR_MASS / M_DRY_AIR * 1e-12 * M_SF6
    base, _ = cs_density(Dict{String, Any}("kind" => "edgar_sf6"), mesh)
    base_kg_per_year = global_rate(base, mesh) * 365.25 * 86400
    sf6_starts = DateTime[DateTime(2021, 12, 1)]
    scale_years = [2021]
    for y in 2022:year(last_month)
        push!(sf6_starts, DateTime(y, 1, 1)); push!(scale_years, y)
    end
    sf6 = Array{Float64, 3}[]
    scales = Float64[]
    for y in scale_years
        haskey(growth, y) || error("no NOAA SF6 growth rate for $y")
        target = growth[y] * kg_per_ppt
        push!(scales, target / base_kg_per_year)
        push!(sf6, base .* last(scales))
        @printf("sf6    %d  growth %.2f ppt/yr -> %.3f kt/yr (EDGAR 2022 %.3f kt/yr, scale %.6f)\n",
                y, growth[y], target / 1e6, base_kg_per_year / 1e6, last(scales))
    end
    write_series(joinpath(outdir, "catrine_sf6_c$(mesh.Nc).nc"), mesh, sf6_starts, sf6;
                 name = "sf6", long_name = "EDGARv8.0 2022 SF6 TOTALS scaled to NOAA global growth",
                 attributes = Dict("source" => "v8.0_FT2022_GHG_SF6_2022_TOTALS_emi.nc",
                                   "scale_years" => scale_years, "scale_factors" => scales,
                                   "kg_per_ppt" => kg_per_ppt))
    println("wrote ", outdir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
