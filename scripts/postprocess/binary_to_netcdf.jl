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

using JSON3
using Printf

using AtmosTransport
using AtmosTransport.Architectures: CPU
using AtmosTransport.Grids: AtmosGrid, CubedSphereMesh, CubedSphereDefinition,
                              EquiangularCubedSphereDefinition,
                              GMAOCubedSphereDefinition,
                              GnomonicPanelConvention,
                              GEOSNativePanelConvention,
                              AbstractCubedSpherePanelConvention,
                              HybridSigmaPressure
using AtmosTransport.Output: SnapshotFrame, write_snapshot_netcdf,
                               SnapshotWriteOptions, output_field_spec

const _ATMSNAP_MAGIC = "ATMSNAP1"

function _read_atmsnap_header(io::IO)
    magic = String(read(io, length(_ATMSNAP_MAGIC)))
    magic == _ATMSNAP_MAGIC || error("not an ATMSNAP binary: bad magic $(repr(magic))")
    header_size = read(io, UInt64)
    header_bytes = read(io, Int(header_size))
    return JSON3.read(String(header_bytes), Dict{String, Any})
end

function _panel_convention_from_tag(tag::AbstractString)
    tag == "gnomonic" && return GnomonicPanelConvention()
    tag == "geos_native" && return GEOSNativePanelConvention()
    error("unknown panel_convention tag: $(repr(tag))")
end

# Map the `cs_definition_tag` Symbol (round-tripped through JSON as a String)
# back to a fully-constructed `CubedSphereDefinition`. The binary header stores
# only the definition tag, panel convention, and the two law tags; we use the
# convention-aware default-constructors for each definition to rebuild a
# self-consistent object rather than reaching into the law factories directly.
function _definition_from_tags(def_tag::AbstractString,
                               convention::AbstractCubedSpherePanelConvention)
    if def_tag == "equiangular_gnomonic"
        return EquiangularCubedSphereDefinition(convention = convention)
    elseif def_tag == "gmao_equal_distance"
        return GMAOCubedSphereDefinition(convention = convention)
    else
        error("unknown cs_definition tag: $(repr(def_tag))")
    end
end

_float_from_dtype_tag(tag::AbstractString) =
    tag == "Float32" ? Float32 :
    tag == "Float64" ? Float64 :
    error("unsupported float_dtype tag: $(repr(tag))")

function _rebuild_cs_mesh(header::Dict{String, Any})
    g = header["grid"]
    Nc = Int(g["Nc"])
    conv = _panel_convention_from_tag(String(g["panel_convention"]))
    def = _definition_from_tags(String(g["definition"]), conv)
    # Match the original run's mesh `FT` so geometry-derived diagnostics
    # (cell_area, lat/lon corner positions, etc.) round-trip bit-exact.
    FT = _float_from_dtype_tag(String(header["float_dtype"]))
    # Hp = 0: the binary stores only interior cells. Downstream output paths
    # don't need halos for write-only consumers. Snapshots written before the
    # radius was recorded come from meshes of radius EARTH_RADIUS.
    radius = Float64(get(g, "planet_radius_m", AtmosTransport.Parameters.EARTH_RADIUS))
    return CubedSphereMesh(; FT = FT, Nc = Nc, Hp = 0, definition = def, radius = FT(radius))
end

function _read_frames(header::Dict{String, Any}, path::AbstractString)
    Nc = Int(header["grid"]["Nc"])
    Nz = Int(header["grid"]["Nz"])
    payload_offset = Int(header["payload_offset"])
    fields = String.(header["fields"])
    n_frames = Int(header["n_frames"])
    times = Float64.(header["times_hours"])
    mass_basis = Symbol(String(header["mass_basis"]))
    stored_totals = get(header, "tracer_total_mass", nothing)

    panel_floats = Nc * Nc * Nz

    frames = Vector{SnapshotFrame}(undef, n_frames)
    open(path, "r") do io
        seek(io, payload_offset)
        for fi in 1:n_frames
            field_arrays = Dict{String, NTuple{6, Array{Float32, 3}}}()
            for field_name in fields
                panels = ntuple(6) do _
                    buf = Vector{Float32}(undef, panel_floats)
                    read!(io, buf)
                    return reshape(buf, (Nc, Nc, Nz))
                end
                field_arrays[field_name] = panels
            end
            air = field_arrays["air_mass"]
            tracers = Dict{Symbol, NTuple{6, Array{Float32, 3}}}()
            for name in fields
                name == "air_mass" && continue
                tracers[Symbol(name)] = field_arrays[name]
            end
            if stored_totals === nothing
                frames[fi] = SnapshotFrame(times[fi], air, tracers, mass_basis)
            else
                totals = Dict(Symbol(name) => Float64(stored_totals[name][fi])
                              for name in fields if name != "air_mass")
                frames[fi] = SnapshotFrame(times[fi], air, tracers, mass_basis;
                                           tracer_total_mass = totals)
            end
        end
    end
    return frames, mass_basis
end

# The `write_snapshot_netcdf` writer reads Nz from the frame, not from the
# vertical grid, so the converter only needs to construct an `AtmosGrid` with
# any valid vertical coordinate of the right level count. Placeholder hybrid
# A/B coefficients are fine; the netcdf writer never queries them.
function _build_grid(header::Dict{String, Any}, mesh::CubedSphereMesh)
    Nz = Int(header["grid"]["Nz"])
    A = zeros(Float64, Nz + 1)
    B = collect(range(1.0, stop = 0.0, length = Nz + 1))
    vertical = HybridSigmaPressure(A, B)
    return AtmosGrid(mesh, vertical, CPU(); radius = mesh.radius)
end

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
    header = open(_read_atmsnap_header, input, "r")
    header["grid_type"] == "cubed_sphere" ||
        error("converter currently supports cubed-sphere only; got $(header["grid_type"])")

    mesh = _rebuild_cs_mesh(header)
    grid = _build_grid(header, mesh)
    frames, mass_basis = _read_frames(header, input)
    @info @sprintf("loaded %d frame(s), %d field(s), %s",
                   length(frames), length(header["fields"]), summary(mesh))

    options = SnapshotWriteOptions(float_type = Float32)
    fields = output_field_spec()
    t0 = time()
    write_snapshot_netcdf(output, frames, grid;
                          mass_basis = mass_basis,
                          options = options,
                          fields = fields)
    elapsed = time() - t0
    @info @sprintf("wrote NetCDF: %s (%.1fs)", output, elapsed)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
