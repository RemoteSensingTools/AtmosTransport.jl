#!/usr/bin/env julia

# Attach the legacy CATRINE 1-degree, 3-hourly TM5 convection archive to a
# current cubed-sphere transport binary (for example C30 or C90).
#
# This is a physically usable TM5 convection driver, but it is coarser in
# space/time than the native ERA5-N320 hourly transport forcing. The output
# records that provenance explicitly so downstream experiments cannot confuse
# the attached product with native-N320 convection.

using ArgParse
using Dates
using NCDatasets
using AtmosTransport

const MD = AtmosTransport.MetDrivers
const PP = AtmosTransport.Preprocessing
const RG = AtmosTransport.Regridding

function settings()
    s = ArgParseSettings(description = "Attach legacy 1-degree TM5 convection to a CS binary")
    @add_arg_table! s begin
        "input"
            required = true
            help = "cubed-sphere transport binary without convection"
        "convection_nc"
            required = true
            help = "legacy CATRINE convec_YYYYMMDD_00p03.nc"
        "output"
            required = true
            help = "output CS binary with entu/detu/entd/detd"
        "--cache-dir"
            dest_name = "cache_dir"
            default = "/temp2/catrine-runs/met/regrid_cache"
            help = "conservative-regrid weight cache"
        "--force"
            action = :store_true
            help = "replace an existing output"
    end
    return s
end

function merged_tm5_to_panels!(panels, dst_flat, src_flat, merged, native,
                               merge_map, regridder, Nc)
    fill!(merged, 0.0)
    @inbounds for k in eachindex(merge_map)
        km = merge_map[k]
        @views merged[:, :, km] .+= native[:, :, k]
    end
    copyto!(src_flat, reshape(merged, size(src_flat)))
    RG.apply_regridder!(dst_flat, regridder, src_flat)
    Nz = size(merged, 3)
    cells_per_panel = Nc * Nc
    @inbounds for p in eachindex(panels), k in 1:Nz, j in 1:Nc, i in 1:Nc
        flat_cell = (p - 1) * cells_per_panel + (j - 1) * Nc + i
        panels[p][i, j, k] = Float32(dst_flat[flat_cell, k])
    end
    return panels
end

function closure_error(fields)
    worst = 0.0
    for p in eachindex(fields.entu)
        entu, detu = fields.entu[p], fields.detu[p]
        entd, detd = fields.entd[p], fields.detd[p]
        for j in axes(entu, 2), i in axes(entu, 1)
            up_e = sum(@view entu[i, j, :])
            up_d = sum(@view detu[i, j, :])
            dn_e = sum(@view entd[i, j, :])
            dn_d = sum(@view detd[i, j, :])
            worst = max(worst, abs(up_e - up_d) / max(up_e, up_d, eps(Float64)))
            worst = max(worst, abs(dn_e - dn_d) / max(dn_e, dn_d, eps(Float64)))
        end
    end
    return worst
end

function main(argv)
    args = parse_args(argv, settings())
    input = abspath(args["input"])
    conv_path = abspath(args["convection_nc"])
    output = abspath(args["output"])
    isfile(input) || error("input binary missing: $input")
    isfile(conv_path) || error("convection file missing: $conv_path")
    input == output && error("input and output paths must differ")
    isfile(output) && !args["force"] && error("output exists: $output")

    @warn "Attaching legacy 1-degree, 3-hourly TM5 convection; this is not native ERA5-N320 hourly convection"
    reader = MD.TransportBinaryReader(input; FT = Float32)
    writer = nothing
    started = time()
    try
        h = reader.header
        g = h.geometry
        MD.grid_type(reader) === :cubed_sphere || error("input must be cubed-sphere")
        MD.has_tm5_convection(reader) && error("input already contains TM5 convection")
        all(s in h.payload_sections for s in (:dm, :pblh, :ustar, :pbl_hflux, :t2m, :dkg)) ||
            error("input must contain dm, complete PBL surface fields, and dkg")
        merge_map = Int.(collect(get(h.raw_header, "merge_map", Int[])))
        length(merge_map) == 137 || error("expected a 137-entry merge_map; found $(length(merge_map))")
        maximum(merge_map) == h.nlevel || error("merge_map target does not match nlevel=$(h.nlevel)")

        target_grid = MD.load_grid(reader; FT = Float64, Hp = 0)
        source_mesh = AtmosTransport.LatLonMesh(
            ; FT = Float64, size = (360, 180), longitude = (-180, 180),
              latitude = (-90, 90))
        regridder = RG.build_regridder(
            source_mesh, target_grid.horizontal; normalize = false,
            cache_dir = abspath(args["cache_dir"]))

        Nc, Nz, np = g.Nc, h.nlevel, g.npanel
        nsrc, ndst = length(regridder.src_areas), length(regridder.dst_areas)
        merged = zeros(Float64, 360, 180, Nz)
        src_flat = zeros(Float64, nsrc, Nz)
        dst_flat = zeros(Float64, ndst, Nz)
        tm5 = (
            entu = ntuple(_ -> zeros(Float32, Nc, Nc, Nz), np),
            detu = ntuple(_ -> zeros(Float32, Nc, Nc, Nz), np),
            entd = ntuple(_ -> zeros(Float32, Nc, Nc, Nz), np),
            detd = ntuple(_ -> zeros(Float32, Nc, Nc, Nz), np),
        )

        mkpath(dirname(output))
        raw = h.raw_header
        metadata = Dict{String, Any}(
            "experimental" => true,
            "validation_status" => "legacy_1deg_3hour_convection_attached; transport_archive_validated_separately",
            "experimental_caveat" => "Legacy 1-degree, 3-hourly CATRINE TM5 convection conservatively remapped to the target cubed sphere; native ERA5-N320 hourly convection comparison remains required.",
            "convection_attachment" => "legacy_catrine_1deg_3hour_tm5_to_cs_v2",
            "convection_target_grid" => "C$(Nc)_L$(Nz)",
            "convection_source" => conv_path,
            "convection_temporal_mapping" => "three-hour field at HH:30 held for three hourly transport windows",
            "source_binary" => input,
            "merge_map" => merge_map,
        )
        for key in ("target_coefficients", "vertical_mapping_method", "target_vertical_name",
                    "mass_fix_enabled", "global_mass_pin_enabled", "coarsening_method",
                    "coarsening_source_Nc", "coarsening_target_Nc", "coarsening_ratio")
            haskey(raw, key) && (metadata[key] = raw[key])
        end
        vc = AtmosTransport.HybridSigmaPressure(Float32.(h.A_ifc), Float32.(h.B_ifc))
        writer = MD.open_streaming_cs_transport_binary(
            output, Nc, np, Nz, h.nwindow, vc;
            FT = Float32, dt_met_seconds = h.dt_met_seconds,
            half_dt_seconds = h.half_dt_seconds, steps_per_window = 1,
            source_flux_sampling = h.source_flux_sampling,
            air_mass_sampling = h.air_mass_sampling,
            flux_sampling = h.flux_sampling, flux_kind = h.flux_kind,
            include_flux_delta = true, mass_basis = h.mass_basis,
            include_surface = true, include_tm5conv = true,
            include_precomputed_dkg = true,
            panel_convention = String(g.panel_convention),
            cs_definition = String(g.definition),
            cs_coordinate_law = String(g.coordinate_law),
            cs_center_law = String(g.center_law),
            longitude_offset_deg = g.longitude_offset_deg,
            extra_header = metadata)

        worst_closure = 0.0
        NCDataset(conv_path, "r") do ds
            for name in ("eu", "du", "ed", "dd")
                haskey(ds, name) || error("convection NC missing $name")
                size(ds[name]) == (360, 180, 137, 8) ||
                    error("unexpected $name shape $(size(ds[name]))")
            end
            for win in 1:h.nwindow
                tidx = clamp(fld(win - 1, 3) + 1, 1, 8)
                if win == 1 || tidx != clamp(fld(win - 2, 3) + 1, 1, 8)
                    merged_tm5_to_panels!(tm5.entu, dst_flat, src_flat, merged,
                        ds["eu"][:, :, :, tidx], merge_map, regridder, Nc)
                    merged_tm5_to_panels!(tm5.detu, dst_flat, src_flat, merged,
                        ds["du"][:, :, :, tidx], merge_map, regridder, Nc)
                    merged_tm5_to_panels!(tm5.entd, dst_flat, src_flat, merged,
                        ds["ed"][:, :, :, tidx], merge_map, regridder, Nc)
                    merged_tm5_to_panels!(tm5.detd, dst_flat, src_flat, merged,
                        ds["dd"][:, :, :, tidx], merge_map, regridder, Nc)
                    worst_closure = max(worst_closure, closure_error(tm5))
                end
                source = MD.load_window!(reader, win)
                dm = MD.load_flux_delta_window!(reader, win).dm
                payload = (m = source.m, am = source.am, bm = source.bm,
                           cm = source.cm, ps = source.ps, dm,
                           surface = source.surface, dkg = source.dkg,
                           tm5_fields = tm5)
                MD.write_streaming_cs_window!(writer, payload, Nc, np)
            end
        end
        MD.set_streaming_steps_per_window_schedule!(writer, h.steps_per_window_by_window)
        MD.close_streaming_transport_binary!(writer)
        writer = nothing
        check = MD.TransportBinaryReader(output; FT = Float32)
        MD.has_tm5_convection(check) || error("written output lacks TM5 capability")
        close(check)
        println("TM5 ATTACH COMPLETE")
        println("  output: ", output)
        println("  size:   ", round(filesize(output) / 2.0^30; digits = 3), " GiB")
        println("  closure:", worst_closure)
        println("  time:   ", round(time() - started; digits = 2), " s")
    catch
        if writer !== nothing
            isopen(writer.io) && close(writer.io)
            isfile(writer.staging_path) && rm(writer.staging_path; force = true)
        end
        rethrow()
    finally
        close(reader)
    end
end

main(ARGS)
