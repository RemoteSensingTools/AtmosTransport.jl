function _ensure_parent_dir(path::AbstractString)
    parent = dirname(path)
    isempty(parent) || mkpath(parent)
    return nothing
end

function _nlevel(frame::SnapshotFrame, ::LatLonMesh)
    return size(frame.air_mass, 3)
end

function _nlevel(frame::SnapshotFrame, ::ReducedGaussianMesh)
    return size(frame.air_mass, 2)
end

function _nlevel(frame::SnapshotFrame, ::CubedSphereMesh)
    return size(frame.air_mass[1], 3)
end

# ---------------------------------------------------------------------------
# Automated provenance metadata.
#
# Everything below is computed at write time so a downstream consumer
# (or "what produced this binary?" forensic question) has a stable
# answer without bookkeeping at the call site. Git SHA / dirty status
# fall back to `"unknown"` when the source tree isn't a git checkout or
# `git` isn't on PATH, so users running from a tarball get a working
# file with explicit "unknown" provenance instead of a hard error.
# ---------------------------------------------------------------------------

const _OUTPUT_CONTRACT = "AtmosTransport snapshot v2"

function _git_command(args::Vector{String})
    # `@__DIR__` is `src/Output/`; pin the repo root via -C so callers
    # running from arbitrary CWDs still resolve the right tree.
    repo_dir = joinpath(@__DIR__, "..", "..")
    return Cmd(vcat(["git", "-C", repo_dir], args))
end

function _git_commit_sha()
    try
        return strip(read(pipeline(_git_command(["rev-parse", "HEAD"]); stderr=devnull), String))
    catch
        return "unknown"
    end
end

function _git_dirty_flag()
    try
        out = strip(read(pipeline(_git_command(["status", "--porcelain"]); stderr=devnull), String))
        return isempty(out) ? "clean" : "dirty"
    catch
        return "unknown"
    end
end

function _iso8601_utc_now()
    # Dates.now(UTC) is portable; format with explicit "Z" suffix.
    return Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "Z"
end

function _runtime_environment_string()
    return @sprintf("Julia %s on %s (%s/%s)",
                    VERSION, Sys.MACHINE,
                    Sys.KERNEL, Sys.iswindows() ? "Windows" :
                                Sys.isapple()   ? "macOS"   :
                                Sys.islinux()   ? "Linux"   : "Unknown")
end

function _options_string(options)
    return @sprintf("float_type=%s, deflate_level=%d, shuffle=%s",
                    options.float_type, options.deflate_level, options.shuffle)
end

# Automated provenance shared by snapshot and observation files: timestamp,
# source-tree state, runtime, host, and user. Each value is best-effort and
# falls back to `"unknown"` rather than throwing, so non-git checkouts still
# produce a complete metadata block.
function _define_provenance_attributes!(ds)
    creation_date = _iso8601_utc_now()
    git_commit    = _git_commit_sha()
    git_dirty     = _git_dirty_flag()
    hostname      = try
        Base.Libc.gethostname()
    catch
        "unknown"
    end
    ds.attrib["creation_date"]    = creation_date
    ds.attrib["framework"]        = "AtmosTransport.jl"
    ds.attrib["framework_commit"] = git_commit
    ds.attrib["framework_dirty"]  = git_dirty
    ds.attrib["runtime"]          = _runtime_environment_string()
    ds.attrib["hostname"]         = hostname
    ds.attrib["user"]             = get(ENV, "USER", get(ENV, "USERNAME", "unknown"))
    return creation_date, git_commit, git_dirty
end

function _define_common_attributes!(ds, mesh, frames, mass_basis_sym::Symbol;
                                     options = nothing)
    ds.attrib["Conventions"]    = "CF-1.8"
    ds.attrib["title"]          = "AtmosTransport runtime snapshot"
    ds.attrib["source"]         = "AtmosTransport.jl"
    ds.attrib["institution"]    = get(ENV, "ATMOSTR_INSTITUTION",
                                       "Caltech / Frankenberg group")
    ds.attrib["grid"]            = summary(mesh)
    ds.attrib["grid_type"]       = _grid_type_string(mesh)
    ds.attrib["mass_basis"]      = String(mass_basis_sym)
    ds.attrib["output_contract"] = _OUTPUT_CONTRACT

    creation_date, git_commit, git_dirty = _define_provenance_attributes!(ds)
    options === nothing ||
        (ds.attrib["output_options"] = _options_string(options))

    # `history` is the CF-canonical "what touched this file" chain. We
    # prepend a single line; readers that pass the file through other
    # tools (CDO, NCO, xarray) append their own lines downstream.
    ds.attrib["history"] = @sprintf("%s: written by AtmosTransport.Output (commit %s%s) with %d frame(s)",
                                     creation_date,
                                     git_commit == "unknown" ? "unknown" : first(git_commit, 12),
                                     git_dirty == "dirty"    ? "+dirty"  : "",
                                     length(frames))
    return nothing
end

_grid_type_string(::LatLonMesh) = "latlon"
_grid_type_string(::ReducedGaussianMesh) = "reduced_gaussian"
_grid_type_string(::CubedSphereMesh) = "cubed_sphere"

function _define_time!(ds, times)
    defDim(ds, "time", length(times))
    v = defVar(ds, "time", Float64, ("time",),
               attrib = Dict("units" => "hours",
                             "long_name" => "elapsed simulation time",
                             "comment" => "hours since the configured simulation start"))
    v[:] = Float64.(times)
    return v
end

# Streaming files grow along time; batch output keeps its fixed dimension.
function _define_time!(ds, ::Nothing)
    defDim(ds, "time", Inf)
    return defVar(ds, "time", Float64, ("time",),
                  attrib = Dict("units" => "hours",
                                "long_name" => "elapsed simulation time",
                                "comment" => "hours since the configured simulation start"))
end

function _define_lev!(ds, Nz::Integer)
    defDim(ds, "lev", Int(Nz))
    # `axis = "Z"` + `positive = "down"` is what Panoply / ncview look
    # for to identify the vertical axis (and to render vertical profiles
    # when the user clicks a horizontal cell). `standard_name =
    # "model_level_number"` is the CF-canonical choice for integer-level
    # vertical coords; `coordinate = "eta"` is a GEOS-Chem hint readers
    # may consult for hybrid-sigma context.
    v = defVar(ds, "lev", Float64, ("lev",),
               attrib = Dict("units" => "1",
                             "long_name" => "model level",
                             "standard_name" => "model_level_number",
                             "axis" => "Z",
                             "positive" => "down",
                             "coordinate" => "eta"))
    v[:] = collect(1.0:Float64(Nz))
    return v
end

function _bounds_from_faces(faces)
    n = length(faces) - 1
    out = Array{Float64}(undef, n, 2)
    @inbounds for i in 1:n
        out[i, 1] = Float64(faces[i])
        out[i, 2] = Float64(faces[i + 1])
    end
    return out
end

function _define_latlon_geometry!(ds, mesh::LatLonMesh, Nz::Integer, times)
    defDim(ds, "lon", nx(mesh))
    defDim(ds, "lat", ny(mesh))
    defDim(ds, "nv", 2)
    _define_lev!(ds, Nz)
    _define_time!(ds, times)

    v_lon = defVar(ds, "lon", Float64, ("lon",),
                   attrib = Dict("units" => "degrees_east",
                                 "long_name" => "longitude",
                                 "standard_name" => "longitude",
                                 "bounds" => "lon_bounds"))
    v_lat = defVar(ds, "lat", Float64, ("lat",),
                   attrib = Dict("units" => "degrees_north",
                                 "long_name" => "latitude",
                                 "standard_name" => "latitude",
                                 "bounds" => "lat_bounds"))
    v_lon[:] = Float64.(mesh.λᶜ)
    v_lat[:] = Float64.(mesh.φᶜ)

    defVar(ds, "lon_bounds", Float64, ("lon", "nv"),
           attrib = Dict("units" => "degrees_east"))[:, :] = _bounds_from_faces(mesh.λᶠ)
    defVar(ds, "lat_bounds", Float64, ("lat", "nv"),
           attrib = Dict("units" => "degrees_north"))[:, :] = _bounds_from_faces(mesh.φᶠ)

    area = Array{Float64}(undef, nx(mesh), ny(mesh))
    @inbounds for j in 1:ny(mesh), i in 1:nx(mesh)
        area[i, j] = Float64(cell_area(mesh, i, j))
    end
    defVar(ds, "cell_area", Float64, ("lon", "lat"),
           attrib = Dict("units" => "m2",
                         "long_name" => "horizontal cell area",
                         "standard_name" => "cell_area"))[:, :] = area
    return (;)
end

function _rg_native_coordinates(mesh::ReducedGaussianMesh)
    lons = Array{Float64}(undef, ncells(mesh))
    lats = Array{Float64}(undef, ncells(mesh))
    areas = Array{Float64}(undef, ncells(mesh))
    lon_bounds = Array{Float64}(undef, ncells(mesh), 4)
    lat_bounds = Array{Float64}(undef, ncells(mesh), 4)
    @inbounds for j in 1:nrings(mesh)
        ring_lons = ring_longitudes(mesh, j)
        dlon = 360.0 / mesh.nlon_per_ring[j]
        lat_s = Float64(mesh.lat_faces[j])
        lat_n = Float64(mesh.lat_faces[j + 1])
        for i in eachindex(ring_lons)
            c = cell_index(mesh, i, j)
            lons[c] = Float64(ring_lons[i])
            lats[c] = Float64(mesh.latitudes[j])
            areas[c] = Float64(cell_area(mesh, c))
            lon_w = (i - 1) * dlon
            lon_e = i * dlon
            lon_bounds[c, :] .= (lon_w, lon_e, lon_e, lon_w)
            lat_bounds[c, :] .= (lat_s, lat_s, lat_n, lat_n)
        end
    end
    return lons, lats, areas, lon_bounds, lat_bounds
end

function _rg_legacy_raster_map(mesh::ReducedGaussianMesh)
    nr = nrings(mesh)
    nlon = maximum(mesh.nlon_per_ring)
    dlon = 360.0 / nlon
    out_lons = [(i - 0.5) * dlon for i in 1:nlon]
    out_lats = Float64.(mesh.latitudes)
    nn_map = zeros(Int, nlon, nr)
    @inbounds for j in 1:nr
        nlon_ring = mesh.nlon_per_ring[j]
        dlon_ring = 360.0 / nlon_ring
        for i in 1:nlon
            i_ring = clamp(round(Int, out_lons[i] / dlon_ring + 0.5), 1, nlon_ring)
            nn_map[i, j] = cell_index(mesh, i_ring, j)
        end
    end
    return out_lons, out_lats, nn_map
end

function _define_rg_geometry!(ds, mesh::ReducedGaussianMesh, Nz::Integer, times)
    defDim(ds, "cell", ncells(mesh))
    defDim(ds, "nv", 4)
    _define_lev!(ds, Nz)
    _define_time!(ds, times)

    lons, lats, areas, lon_bounds, lat_bounds = _rg_native_coordinates(mesh)
    defVar(ds, "cell", Int32, ("cell",),
           attrib = Dict("long_name" => "native reduced-Gaussian cell index"))[:] =
        Int32.(1:ncells(mesh))
    defVar(ds, "cell_lon", Float64, ("cell",),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "native cell longitude",
                         "standard_name" => "longitude",
                         "bounds" => "cell_lon_bounds"))[:] = lons
    defVar(ds, "cell_lat", Float64, ("cell",),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "native cell latitude",
                         "standard_name" => "latitude",
                         "bounds" => "cell_lat_bounds"))[:] = lats
    defVar(ds, "cell_lon_bounds", Float64, ("cell", "nv"),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "native cell corner longitudes",
                         "comment" => "Corners are ordered SW, SE, NE, NW."))[:, :] = lon_bounds
    defVar(ds, "cell_lat_bounds", Float64, ("cell", "nv"),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "native cell corner latitudes",
                         "comment" => "Corners are ordered SW, SE, NE, NW."))[:, :] = lat_bounds
    defVar(ds, "cell_area", Float64, ("cell",),
           attrib = Dict("units" => "m2",
                         "long_name" => "horizontal cell area",
                         "standard_name" => "cell_area",
                         "coordinates" => "cell_lon cell_lat"))[:] = areas

    out_lons, out_lats, nn_map = _rg_legacy_raster_map(mesh)
    defDim(ds, "lon", length(out_lons))
    defDim(ds, "lat", length(out_lats))
    defVar(ds, "lon", Float64, ("lon",),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "diagnostic longitude",
                         "standard_name" => "longitude"))[:] = out_lons
    defVar(ds, "lat", Float64, ("lat",),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "diagnostic latitude",
                         "standard_name" => "latitude"))[:] = out_lats

    ds.attrib["nrings"] = nrings(mesh)
    ds.attrib["regridding"] = "nearest-neighbor reduced-Gaussian diagnostic raster"
    return (; nn_map)
end

function _cs_convention_tag(mesh::CubedSphereMesh)
    mesh.convention isa GnomonicPanelConvention && return "gnomonic"
    mesh.convention isa GEOSNativePanelConvention && return "geos_native"
    return "unknown"
end

function _cs_central_meridian(mesh::CubedSphereMesh)
    return longitude_offset_deg(cs_definition(mesh))
end

function _cs_panel_center_arrays(mesh::CubedSphereMesh)
    Nc = mesh.Nc
    lons = Array{Float64}(undef, Nc, Nc, 6)
    lats = Array{Float64}(undef, Nc, Nc, 6)
    for p in 1:6
        lonp, latp = panel_cell_center_lonlat(mesh, p)
        lons[:, :, p] = lonp
        lats[:, :, p] = latp
    end
    return lons, lats
end

function _cs_panel_corner_arrays(mesh::CubedSphereMesh)
    Nc = mesh.Nc
    lons = Array{Float64}(undef, Nc + 1, Nc + 1, 6)
    lats = Array{Float64}(undef, Nc + 1, Nc + 1, 6)
    for p in 1:6
        lonp, latp = panel_cell_corner_lonlat(mesh, p)
        lons[:, :, p] = lonp
        lats[:, :, p] = latp
    end
    return lons, lats
end

function _define_cs_geometry!(ds, mesh::CubedSphereMesh, Nz::Integer, times)
    Nc = mesh.Nc
    defDim(ds, "Xdim", Nc)
    defDim(ds, "Ydim", Nc)
    defDim(ds, "Xcorner", Nc + 1)
    defDim(ds, "Ycorner", Nc + 1)
    defDim(ds, "nf", 6)
    _define_lev!(ds, Nz)
    _define_time!(ds, times)

    convention_tag = _cs_convention_tag(mesh)
    central_meridian = _cs_central_meridian(mesh)
    coordinate_tag = coordinate_law_tag(coordinate_law(mesh))
    center_tag = center_law_tag(center_law(mesh))
    definition_tag = String(cs_definition_tag(cs_definition(mesh)))
    ds.attrib["Nc"] = Nc
    ds.attrib["cs_definition"] = definition_tag
    ds.attrib["cs_coordinate_law"] = coordinate_tag
    ds.attrib["cs_center_law"] = center_tag
    ds.attrib["panel_convention"] = convention_tag
    ds.attrib["grid_mapping_name"] = "gnomonic cubed-sphere"
    ds.attrib["longitude_of_central_meridian"] = central_meridian

    # GEOS-Chem-compatibility trick: tag the 1D panel-index axes with
    # `degrees_east` / `degrees_north` so Panoply (and GrADS, ncview, …)
    # auto-recognize the dataset as geographic and render a per-face
    # map. The numeric values stay panel indices `1..Nc`; geographic
    # truth lives in the auxiliary `lons`/`lats` 3D coords below. The
    # "Fake …" long_name is the documented GEOS-Chem convention so a
    # reader doesn't mistake these for true 1D longitudes/latitudes.
    defVar(ds, "Xdim", Float64, ("Xdim",),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "Fake Longitude for GrADS Compatibility",
                         "axis" => "X"))[:] = collect(1.0:Nc)
    defVar(ds, "Ydim", Float64, ("Ydim",),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "Fake Latitude for GrADS Compatibility",
                         "axis" => "Y"))[:] = collect(1.0:Nc)
    defVar(ds, "nf", Int32, ("nf",),
           attrib = Dict("axis" => "e",
                         "grads_dim" => "e",
                         "long_name" => "cubed-sphere face"))[:] = Int32.(1:6)

    lons, lats = _cs_panel_center_arrays(mesh)
    clons, clats = _cs_panel_corner_arrays(mesh)
    defVar(ds, "lons", Float64, ("Xdim", "Ydim", "nf"),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "cell-center longitude",
                         "standard_name" => "longitude"))[:, :, :] = lons
    defVar(ds, "lats", Float64, ("Xdim", "Ydim", "nf"),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "cell-center latitude",
                         "standard_name" => "latitude"))[:, :, :] = lats
    defVar(ds, "corner_lons", Float64, ("Xcorner", "Ycorner", "nf"),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "cell-corner longitude"))[:, :, :] = clons
    defVar(ds, "corner_lats", Float64, ("Xcorner", "Ycorner", "nf"),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "cell-corner latitude"))[:, :, :] = clats

    area = Array{Float64}(undef, Nc, Nc, 6)
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        area[i, j, p] = Float64(cell_area(mesh, i, j))
    end
    defVar(ds, "cell_area", Float64, ("Xdim", "Ydim", "nf"),
           attrib = Dict("units" => "m2",
                         "long_name" => "horizontal cell area",
                         "standard_name" => "cell_area",
                         "coordinates" => "lons lats"))[:, :, :] = area

    gm = defVar(ds, "cubed_sphere", Int32, (),
                attrib = Dict("grid_mapping_name" => "gnomonic cubed-sphere",
                              "cs_definition" => definition_tag,
                              "cs_coordinate_law" => coordinate_tag,
                              "cs_center_law" => center_tag,
                              "panel_convention" => convention_tag,
                              "longitude_of_central_meridian" => central_meridian,
                              "semi_major_axis" => Float64(mesh.radius),
                              "inverse_flattening" => 0.0,
                              "comment" => "Use lons/lats and corner_lons/corner_lats as authoritative native-cell coordinates."))
    gm[] = Int32(0)
    return (;)
end

function _define_geometry!(ds, mesh::LatLonMesh, Nz::Integer, times)
    return _define_latlon_geometry!(ds, mesh, Nz, times)
end

function _define_geometry!(ds, mesh::ReducedGaussianMesh, Nz::Integer, times)
    return _define_rg_geometry!(ds, mesh, Nz, times)
end

function _define_geometry!(ds, mesh::CubedSphereMesh, Nz::Integer, times)
    return _define_cs_geometry!(ds, mesh, Nz, times)
end
