#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# scripts/postprocess/binary_to_netcdf.jl — convert ATMSNAP1 binary snapshots
# (produced by `[output].format = "binary_mmap"` runs) to the canonical
# NetCDF output that downstream tooling expects.
#
# Usage:
#   julia --project=. scripts/postprocess/binary_to_netcdf.jl <input.atmsnap> [output.nc]
#
# If `[output.nc]` is omitted, the converter writes alongside the input with
# the `.atmsnap` (or `.bin`) suffix replaced by `.nc`. The converter never
# overwrites the source binary; pass an explicit output path if you want to
# write next to it under a different name.
# ---------------------------------------------------------------------------

using Printf
using Dates

using AtmosTransport
using AtmosTransport.Output: read_snapshot_binary, write_snapshot_netcdf,
                             SnapshotWriteOptions, output_field_spec

function _default_output_path(input::AbstractString)
    # Strip a single trailing .atmsnap or .bin suffix; otherwise append .nc.
    m = match(r"\.(atmsnap|bin)$", input)
    return m === nothing ? string(input, ".nc") :
                            string(input[1:end - length(m.match)], ".nc")
end

function main()
    if length(ARGS) < 1
        println(stderr, "Usage: julia --project=. scripts/postprocess/binary_to_netcdf.jl <input.atmsnap> [output.nc]")
        exit(1)
    end
    input = ARGS[1]
    isfile(input) || error("input file not found: $(input)")
    output = length(ARGS) >= 2 ? ARGS[2] : _default_output_path(input)
    abspath(input) == abspath(output) &&
        error("output path collides with input — pass an explicit output path")

    @info "reading binary header" input
    (; header, grid, frames, mass_basis) = read_snapshot_binary(input)
    mesh = grid.horizontal
    @info @sprintf("loaded %d frame(s), %d field(s), %s",
                   length(frames), length(header["fields"]), summary(mesh))

    options = SnapshotWriteOptions(float_type = Float32)
    fields = output_field_spec()
    time_units = String(get(header, "time_units", "hours"))
    time_origin = if startswith(time_units, "hours since ")
        DateTime(time_units[length("hours since ") + 1:end], dateformat"yyyy-mm-dd HH:MM:SS")
    else
        nothing
    end
    physics_attributes = Dict{String, String}(
        key => String(value) for (key, value) in pairs(header) if startswith(key, "physics_"))
    t0 = time()
    write_snapshot_netcdf(output, frames, grid;
                          mass_basis = mass_basis,
                          options = options,
                          fields = fields,
                          time_origin,
                          physics_attributes)
    elapsed = time() - t0
    @info @sprintf("wrote NetCDF: %s (%.1fs)", output, elapsed)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
