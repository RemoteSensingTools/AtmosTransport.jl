# ERA5 single-level surface reader for raw PBL diffusion fields.
#
# Supports NetCDF files that are either plain `.nc` or CDS ZIP payloads
# saved with an `.nc` suffix. Fields are normalized to the lat-lon
# transport-binary convention: longitude centered on [-180, 180) and
# latitude south-to-north.

const ERA5_SURFACE_VAR_CANDIDATES = Dict(
    :pblh   => ("pblh", "blh", "boundary_layer_height"),
    :ustar  => ("ustar", "zust", "friction_velocity"),
    :hflux  => ("hflux", "sshf", "surface_sensible_heat_flux"),
    :lhflux => ("lhflux", "slhf", "surface_latent_heat_flux"),
    :t2m    => ("t2m", "2t", "2m_temperature"),
    :u10    => ("u10", "10u", "10m_u_component_of_wind"),
    :v10    => ("v10", "10v", "10m_v_component_of_wind"),
)

const ERA5_SURFACE_TIME_DIMS = ("time", "valid_time")
const ERA5_USTAR_DRAG_COEFFICIENT = 1.2e-3

mutable struct ERA5SurfaceReader
    datasets :: Vector{NCDataset}
    path     :: String
    tmp_dir  :: Union{Nothing, String}
    date     :: Date
    Nx       :: Int
    Ny       :: Int
    time_len :: Int
    lons     :: Vector{Float64}
    lats     :: Vector{Float64}
end

function _resolve_era5_surface_path(surface_dir::AbstractString, date::Date)
    root = expand_data_path(surface_dir)
    ym = Dates.format(date, "yyyymm")
    ymd = Dates.format(date, "yyyymmdd")
    candidates = String[
        joinpath(root, "era5_surface_$(ymd).nc"),
        joinpath(root, "era5_surface_$(ym).nc"),
        joinpath(root, "surface_$(ymd).nc"),
        joinpath(root, "surface_$(ym).nc"),
        joinpath(root, "sfc_an_native", "era5_surface_$(ymd).nc"),
        joinpath(root, "sfc_an_native", "era5_surface_$(ym).nc"),
    ]
    for path in candidates
        isfile(path) && return path
    end
    throw(ArgumentError(
        "ERA5 surface fields not found for $(date). Looked for: " *
        join(candidates, ", ")))
end

function _is_zip_payload(path::String)
    open(path, "r") do io
        sig = read(io, min(4, filesize(path)))
        return length(sig) >= 2 && sig[1] == 0x50 && sig[2] == 0x4b
    end
end

function _surface_extract_dir(path::String)
    parent = dirname(abspath(path))
    try
        return mktempdir(parent; prefix = ".surface_extract_")
    catch err
        @warn "Could not create ERA5 surface extraction directory next to source; using scratch volume" path exception=(err, catch_backtrace())
        return _surface_scratch_extract_dir()
    end
end

function _surface_scratch_extract_dir()
    candidates = String[]
    for key in ("ATMOSTR_SCRATCH_DIR", "ATMOSTR_TMPDIR", "SCRATCH", "TMPDIR", "TEMP", "TMP")
        value = strip(get(ENV, key, ""))
        isempty(value) && continue
        append!(candidates, split(value, Sys.iswindows() ? ';' : ':'))
    end
    push!(candidates, tempdir())

    seen = Set{String}()
    for parent in candidates
        parent = isempty(parent) ? parent : abspath(expanduser(parent))
        isempty(parent) && continue
        parent in seen && continue
        push!(seen, parent)
        isdir(parent) || continue
        try
            return mktempdir(parent; prefix = "era5_surface_extract_")
        catch
            continue
        end
    end
    return mktempdir(; prefix = "era5_surface_extract_")
end

function _surface_nc_paths(path::String)
    _is_zip_payload(path) || return String[path], nothing

    entries = strip.(readlines(`unzip -Z1 $path`))
    nc_entries = [entry for entry in entries
                  if !isempty(entry) &&
                     !endswith(entry, "/") &&
                     endswith(lowercase(entry), ".nc")]
    isempty(nc_entries) && throw(ArgumentError(
        "ERA5 surface ZIP $(path) contains no NetCDF entries"))

    tmp_dir = _surface_extract_dir(path)
    nc_paths = String[]
    try
        for (idx, entry) in enumerate(nc_entries)
            out_path = joinpath(tmp_dir, @sprintf("%02d_%s", idx, basename(entry)))
            open(out_path, "w") do out
                run(pipeline(`unzip -p $path $entry`; stdout = out))
            end
            push!(nc_paths, out_path)
        end
    catch
        rm(tmp_dir; recursive = true, force = true)
        rethrow()
    end
    return nc_paths, tmp_dir
end

function open_era5_surface_reader(surface_dir::AbstractString, date::Date,
                                  Nx::Int, Ny::Int)
    # ARCO layout: one netCDF per variable under
    # `<surface_dir>/arco/YYYYMMDD/<variable>.nc` (blh, zust, sshf, slhf, t2m, …).
    # The var-binding below already searches across a Vector of datasets, so we
    # just open them all. tmp_dir stays `nothing` — these are the real download,
    # not an extracted temp, so they must NOT be deleted on close.
    arco_dir = joinpath(expand_data_path(surface_dir), "arco",
                        Dates.format(date, "yyyymmdd"))
    if isdir(arco_dir)
        arco_ncs = sort!(filter(f -> endswith(lowercase(f), ".nc"),
                                readdir(arco_dir; join = true)))
        isempty(arco_ncs) ||
            return _open_era5_surface_reader_from(arco_ncs, nothing, arco_dir,
                                                  date, Nx, Ny)
    end

    path = _resolve_era5_surface_path(surface_dir, date)
    nc_paths, tmp_dir = _surface_nc_paths(path)
    return _open_era5_surface_reader_from(nc_paths, tmp_dir, path, date, Nx, Ny)
end

function _open_era5_surface_reader_from(nc_paths::Vector{String},
                                        tmp_dir::Union{Nothing, String},
                                        path::AbstractString, date::Date,
                                        Nx::Int, Ny::Int)
    datasets = NCDataset[]
    try
        for nc_path in nc_paths
            push!(datasets, NCDataset(nc_path, "r"))
        end
    catch
        for ds in datasets
            close(ds)
        end
        tmp_dir === nothing || rm(tmp_dir; recursive = true, force = true)
        rethrow()
    end
    try
        time_len = isempty(datasets) ? 0 : maximum(_surface_time_len, datasets)
        lons, lats = _era5_surface_coordinates(datasets, path, Nx, Ny)
        return ERA5SurfaceReader(datasets, String(path), tmp_dir, date, Nx, Ny,
                                 time_len, lons, lats)
    catch
        for ds in datasets
            close(ds)
        end
        tmp_dir === nothing || rm(tmp_dir; recursive = true, force = true)
        rethrow()
    end
end

function close_era5_surface_reader(reader::ERA5SurfaceReader)
    for ds in reader.datasets
        close(ds)
    end
    reader.tmp_dir === nothing || rm(reader.tmp_dir; recursive = true, force = true)
    return nothing
end

function _surface_time_len(ds::NCDataset)
    for dim_name in ERA5_SURFACE_TIME_DIMS
        if haskey(ds.dim, dim_name)
            return Int(ds.dim[dim_name])
        elseif haskey(ds, dim_name)
            return length(ds[dim_name])
        end
    end
    return 1
end

function _normalize_surface_coordinates(lon, lat)
    any(ismissing, lon) && throw(ArgumentError(
        "ERA5 surface longitude coordinate contains missing values"))
    any(ismissing, lat) && throw(ArgumentError(
        "ERA5 surface latitude coordinate contains missing values"))
    lons = Float64.(lon)
    lats = Float64.(lat)
    if length(lats) > 1 && lats[1] > lats[end]
        reverse!(lats)
    end
    # Keep the coordinate order exactly synchronized with
    # `_normalize_surface_ll!`: a 0..360 field is rolled by half a grid into
    # the centred -180..180 order before it is regridded.
    if length(lons) > 1 && lons[1] >= 0 && lons[end] > 180
        lons = circshift(lons, length(lons) ÷ 2)
        @. lons = ifelse(lons >= 180, lons - 360, lons)
    end
    return lons, lats
end

function _era5_surface_coordinates(datasets::Vector{NCDataset}, path,
                                    Nx::Int, Ny::Int)
    isempty(datasets) && throw(ArgumentError(
        "ERA5 surface input $(path) contains no NetCDF datasets"))
    reference = nothing
    for ds in datasets
        haskey(ds, "longitude") || throw(ArgumentError(
            "ERA5 surface dataset in $(path) is missing longitude coordinates"))
        haskey(ds, "latitude") || throw(ArgumentError(
            "ERA5 surface dataset in $(path) is missing latitude coordinates"))
        lons, lats = _normalize_surface_coordinates(
            ds["longitude"][:], ds["latitude"][:])
        length(lons) == Nx || throw(DimensionMismatch(
            "ERA5 surface longitude coordinate has length $(length(lons)), expected $(Nx)"))
        length(lats) == Ny || throw(DimensionMismatch(
            "ERA5 surface latitude coordinate has length $(length(lats)), expected $(Ny)"))
        if reference === nothing
            reference = (lons, lats)
        elseif lons != reference[1] || lats != reference[2]
            throw(ArgumentError(
                "ERA5 surface datasets in $(path) do not share one longitude/latitude grid"))
        end
    end
    return reference
end

"""
    era5_surface_source_mesh(reader, radius) -> LatLonMesh

Build the ERA5 single-level source mesh from the coordinates stored in the
open file. `SnappedFaces` preserves the point-registered 0.25° grid: the
longitude points remain cell centres and the latitude rows at ±90° become
half-width polar caps. The returned coordinate order matches the normalized
surface arrays produced by [`load_era5_surface_window`](@ref).
"""
function era5_surface_source_mesh(reader::ERA5SurfaceReader, radius::Real)
    return source_latlon_mesh(reader.lons, reader.lats, SnappedFaces(); radius)
end

function _surface_time_index(reader::ERA5SurfaceReader, ds::NCDataset, win_idx::Int)
    time_len = _surface_time_len(ds)
    time_len <= 1 && return 1
    records_per_day = round(Int, 86400 / ERA5_MET_INTERVAL_SECONDS)
    if time_len == records_per_day
        return win_idx
    end
    idx = (day(reader.date) - 1) * records_per_day + win_idx
    idx <= time_len || throw(ArgumentError(
        "ERA5 surface file $(reader.path) has $(time_len) time slots; " *
        "need index $(idx) for $(reader.date) window $(win_idx)."))
    return idx
end

function _surface_var_name(ds::NCDataset, key::Symbol)
    for candidate in ERA5_SURFACE_VAR_CANDIDATES[key]
        for name in keys(ds)
            lowercase(String(name)) == lowercase(candidate) && return String(name)
        end
    end
    return nothing
end

function _surface_var_binding(reader::ERA5SurfaceReader, key::Symbol; required::Bool = true)
    for ds in reader.datasets
        name = _surface_var_name(ds, key)
        name === nothing || return ds, name
    end
    required || return nothing
    throw(ArgumentError(
        "ERA5 surface file $(reader.path) is missing $(key); tried " *
        join(ERA5_SURFACE_VAR_CANDIDATES[key], ", ")))
end

function _read_surface_slice(ds::NCDataset, var_name::String, time_idx::Int)
    v = ds[var_name]
    dims = dimnames(v)
    lon_dim = findfirst(==("longitude"), dims)
    lat_dim = findfirst(==("latitude"), dims)
    time_dim = findfirst(d -> d in ERA5_SURFACE_TIME_DIMS, dims)
    lon_dim === nothing && throw(ArgumentError("surface variable $(var_name) lacks longitude dimension"))
    lat_dim === nothing && throw(ArgumentError("surface variable $(var_name) lacks latitude dimension"))

    idx = ntuple(d -> d == lon_dim || d == lat_dim ? Colon() :
                      d == time_dim ? time_idx : 1,
                 length(dims))
    raw = Array(v[idx...])
    kept = [dims[d] for d in eachindex(dims) if d == lon_dim || d == lat_dim]
    field = kept == ["longitude", "latitude"] ? raw :
            kept == ["latitude", "longitude"] ? permutedims(raw, (2, 1)) :
            throw(ArgumentError("unsupported surface variable dimension order for $(var_name): $(dims)"))
    return field
end

function _normalize_surface_ll!(field::AbstractMatrix{FT},
                                ds::NCDataset) where FT
    out = field
    if haskey(ds, "latitude")
        lats = Float64.(ds["latitude"][:])
        if length(lats) == size(out, 2) && lats[1] > lats[end]
            out = out[:, end:-1:1]
        end
    end
    if haskey(ds, "longitude")
        tmp3 = reshape(out, size(out, 1), size(out, 2), 1)
        out = @view normalize_lon_to_centered(tmp3, Float64.(ds["longitude"][:]))[:, :, 1]
    end
    return Array{FT}(out)
end

"""
    _surface_flux_to_upward_wm2(raw, ds, var_name, FT) -> Matrix{FT}

Normalise an ERA5 surface turbulent heat flux to **upward-positive W m⁻²**.

ERA5 reports `sshf` (sensible) and `slhf` (latent) as time-accumulated
energy over the output interval (J m⁻², downward-positive). Two corrections
recover an instantaneous, upward-positive flux:

  * **divide by 3600** (`ERA5_MET_INTERVAL_SECONDS`) — the accumulation window
    is one hour, so J m⁻² becomes W m⁻²;
  * **flip the sign** — ERA5 uses downward-positive, while the boundary-layer
    scheme (and atmospheric convention) treats an upward surface flux of heat
    into the atmosphere as positive.

Fields already stored as a rate (units without `J`, e.g. a pre-divided
`W m⁻²` export) pass through unchanged. The accumulated case is detected
either from the `units` attribute containing `J` or from the canonical
short/long names of the two accumulated fluxes.
"""
function _surface_flux_to_upward_wm2(raw, ds::NCDataset, var_name::String, ::Type{FT}) where FT
    units = lowercase(String(get(ds[var_name].attrib, "units", "")))
    lname = lowercase(var_name)
    accumulated = occursin("j", units) || lname in (
        "sshf", "surface_sensible_heat_flux",
        "slhf", "surface_latent_heat_flux")
    # The Int divisor keeps Float32 data in Float32 before the FT conversion.
    return accumulated ? FT.(-raw ./ round(Int, ERA5_MET_INTERVAL_SECONDS)) : FT.(raw)
end

# ERA5 surface slices occasionally carry a handful of `_FillValue`-masked cells
# (observed on some Dec-2021 hours for `sshf`); NCDatasets surfaces those as
# `missing`, so a bare `FT.(...)` cast throws `Float32(::Missing)`. Fill the masked
# cells from their finite neighbours (see `_fill_masked_surface`) so a stray cell
# takes a physically-local value rather than a flat global mean — preserving the
# downstream finiteness/range validators (`_validate_era5_surface!`) — log the
# count, and REFUSE to fill a slice where more than
# `ERA5_SURFACE_MAX_MASKED_FRACTION` of the field is masked (a genuinely broken
# slice, not a few stray cells: regenerate the download instead). The guard is set
# tight: a "handful" of stray cells is far below 1% even on a coarse grid, so 1%
# still admits the observed cases while catching wholesale corruption.
const ERA5_SURFACE_MAX_MASKED_FRACTION = 0.01

# Fill `missing` cells in a (lon × lat) surface slice from their finite 4-neighbours
# (longitude wraps, latitude clamps), sweeping inward until none remain. Each masked
# cell becomes the mean of its currently-finite neighbours, so the fill follows the
# local gradient instead of collapsing stray cells to the global mean. A region with
# no finite neighbour after the sweeps (fully isolated) falls back to the global
# finite mean. Returns a dense `Matrix{FT}` with no remaining `missing`.
function _fill_masked_surface(raw::AbstractMatrix, ::Type{FT}) where FT
    nlon, nlat = size(raw)
    masked = ismissing.(raw)
    out = Matrix{FT}(undef, nlon, nlat)
    @inbounds for k in eachindex(raw)
        out[k] = masked[k] ? zero(FT) : FT(raw[k])
    end
    gmean = FT(sum(skipmissing(raw)) / count(!, masked))
    remaining = count(masked)
    sweep = 0
    while remaining > 0 && sweep < nlon + nlat
        sweep += 1
        prevmask = copy(masked)        # snapshot: only consult last sweep's finite cells
        prevout = copy(out)
        filled = 0
        @inbounds for j in 1:nlat, i in 1:nlon
            prevmask[i, j] || continue
            s = zero(FT); n = 0
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ii = mod1(i + di, nlon)            # longitude is periodic
                jj = j + dj
                (1 <= jj <= nlat) || continue      # latitude is not
                prevmask[ii, jj] && continue
                s += prevout[ii, jj]; n += 1
            end
            if n > 0
                out[i, j] = s / n
                masked[i, j] = false
                filled += 1
            end
        end
        remaining -= filled
        filled == 0 && break           # no progress → isolated region, fall back
    end
    if remaining > 0
        @inbounds for k in eachindex(out)
            masked[k] && (out[k] = gmean)
        end
    end
    return out
end

function _surface_slice_to_FT(ds::NCDataset, var_name::String, time_idx::Int,
                              ::Type{FT}) where FT
    raw = _read_surface_slice(ds, var_name, time_idx)
    nmiss = count(ismissing, raw)
    nmiss == 0 && return FT.(raw)
    frac = nmiss / length(raw)
    frac <= ERA5_SURFACE_MAX_MASKED_FRACTION || error(
        "ERA5 surface $(var_name) time $(time_idx): $(nmiss)/$(length(raw)) cells " *
        "($(round(100 * frac; digits = 2))%) are _FillValue-masked, exceeding the " *
        "$(round(100 * ERA5_SURFACE_MAX_MASKED_FRACTION; digits = 1))% guard — refusing to " *
        "fill a broken slice; regenerate the surface download for this day.")
    @warn "ERA5 surface $(var_name) time $(time_idx): filling $(nmiss) masked " *
          "cell(s) from finite neighbours" maxlog = 8
    return _fill_masked_surface(raw, FT)
end

function _load_surface_field(reader::ERA5SurfaceReader,
                             key::Symbol,
                             win_idx::Int,
                             ::Type{FT}) where FT
    ds, var_name = _surface_var_binding(reader, key)
    time_idx = _surface_time_index(reader, ds, win_idx)
    field = _normalize_surface_ll!(_surface_slice_to_FT(ds, var_name, time_idx, FT), ds)
    return field, ds, var_name
end

function _load_surface_raw_field(reader::ERA5SurfaceReader,
                                 key::Symbol,
                                 win_idx::Int,
                                 ::Type{FT}) where FT
    ds, var_name = _surface_var_binding(reader, key)
    time_idx = _surface_time_index(reader, ds, win_idx)
    field = _normalize_surface_ll!(_surface_slice_to_FT(ds, var_name, time_idx, FT), ds)
    return field, ds, var_name
end

function _load_surface_ustar(reader::ERA5SurfaceReader,
                             win_idx::Int,
                             ::Type{FT}) where FT
    direct = _surface_var_binding(reader, :ustar; required = false)
    if direct !== nothing
        ds, var_name = direct
        time_idx = _surface_time_index(reader, ds, win_idx)
        return _normalize_surface_ll!(_surface_slice_to_FT(ds, var_name, time_idx, FT), ds)
    end

    u10, _, _ = _load_surface_field(reader, :u10, win_idx, FT)
    v10, _, _ = _load_surface_field(reader, :v10, win_idx, FT)
    size(u10) == size(v10) || throw(DimensionMismatch(
        "ERA5 surface u10 has size $(size(u10)); v10 has size $(size(v10))"))

    ustar = similar(u10, FT)
    drag = FT(sqrt(ERA5_USTAR_DRAG_COEFFICIENT))
    @. ustar = drag * hypot(u10, v10)
    return ustar
end

function _validate_era5_surface!(surface, path::String, win_idx::Int)
    all(isfinite, surface.pblh) || error("ERA5 surface PBLH contains non-finite values in $(path) window $(win_idx)")
    all(isfinite, surface.ustar) || error("ERA5 surface USTAR contains non-finite values in $(path) window $(win_idx)")
    all(isfinite, surface.hflux) || error("ERA5 surface HFLUX contains non-finite values in $(path) window $(win_idx)")
    all(isfinite, surface.t2m) || error("ERA5 surface T2M contains non-finite values in $(path) window $(win_idx)")
    minimum(surface.pblh) > 0 || error("ERA5 surface PBLH must be positive in $(path) window $(win_idx)")
    minimum(surface.ustar) >= 0 || error("ERA5 surface USTAR must be nonnegative in $(path) window $(win_idx)")
    t_lo, t_hi = SURFACE_AIR_TEMPERATURE_RANGE
    t_lo < minimum(surface.t2m) < t_hi || error("ERA5 surface T2M minimum is out of range in $(path) window $(win_idx)")
    t_lo < maximum(surface.t2m) < t_hi || error("ERA5 surface T2M maximum is out of range in $(path) window $(win_idx)")
    maximum(abs, surface.hflux) < SURFACE_HEAT_FLUX_MAX_ABS || error("ERA5 surface HFLUX magnitude is out of range in $(path) window $(win_idx)")
    if haskey(surface, :lhflux)
        all(isfinite, surface.lhflux) || error("ERA5 surface LHFLUX contains non-finite values in $(path) window $(win_idx)")
        maximum(abs, surface.lhflux) < SURFACE_HEAT_FLUX_MAX_ABS || error("ERA5 surface LHFLUX magnitude is out of range in $(path) window $(win_idx)")
    end
    return nothing
end

function _validate_surface_shapes!(surface, expected::Tuple{Int, Int})
    for (name, field) in pairs(surface)
        size(field) == expected ||
            throw(DimensionMismatch("ERA5 surface $(name) has size $(size(field)), expected $(expected)"))
    end
    return nothing
end

"""
    load_era5_surface_window(reader, win_idx, FT; with_latent = false) -> NamedTuple

Read one time window of ERA5 surface fields, normalised to the lat-lon
transport-binary convention (longitude on `[-180, 180)`, latitude south→north).

Returns `(pblh, t2m, ustar, hflux)`:

  * `pblh`  — boundary-layer height (m),
  * `t2m`   — 2 m temperature (K),
  * `ustar` — friction velocity (m s⁻¹), read directly when ERA5 `zust` is
    present, otherwise estimated from the 10 m wind via a constant drag
    coefficient,
  * `hflux` — surface **sensible** heat flux, upward-positive (W m⁻²).

With `with_latent = true` the result additionally carries

  * `lhflux` — surface **latent** heat flux, upward-positive (W m⁻²),

which the TM5 boundary-layer diffusion scheme (`bldiff`) needs to form the
surface virtual heat flux. It is opt-in so existing callers that only need the
four PBL fields neither require `slhf` in the file nor change behaviour.
"""
function load_era5_surface_window(reader::ERA5SurfaceReader,
                                  win_idx::Int,
                                  ::Type{FT};
                                  with_latent::Bool = false) where FT
    pblh, _, _ = _load_surface_field(reader, :pblh, win_idx, FT)
    ustar = _load_surface_ustar(reader, win_idx, FT)
    hflux_raw, hflux_ds, hflux_name = _load_surface_raw_field(reader, :hflux, win_idx, FT)
    hflux = _surface_flux_to_upward_wm2(hflux_raw, hflux_ds, hflux_name, FT)
    t2m, _, _ = _load_surface_field(reader, :t2m, win_idx, FT)

    surface = (pblh = pblh, t2m = t2m, ustar = ustar, hflux = hflux)
    if with_latent
        lh_raw, lh_ds, lh_name = _load_surface_raw_field(reader, :lhflux, win_idx, FT)
        surface = merge(surface,
                        (lhflux = _surface_flux_to_upward_wm2(lh_raw, lh_ds, lh_name, FT),))
    end

    _validate_surface_shapes!(surface, (reader.Nx, reader.Ny))
    _validate_era5_surface!(surface, reader.path, win_idx)
    return surface
end
