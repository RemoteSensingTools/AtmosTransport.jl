#!/usr/bin/env julia

# Attach the legacy CATRINE 1-degree, 3-hourly TM5 convection archive to a
# current cubed-sphere transport binary (for example C30 or C90).
#
# This is a physically usable TM5 convection driver, but it is coarser in
# space/time than the native ERA5-N320 hourly transport forcing. The output
# records that provenance explicitly so downstream experiments cannot confuse
# the attached product with native-N320 convection.
#
# The archive's fluxes are per unit of total (moist) air. For a dry-basis
# binary they are converted to dry air on the 1-degree grid, at the native
# ERA5 levels, with the ERA5 N320 specific humidity at the start of each
# transport window (`tm5_convection_to_basis!`), before the vertical mapping
# and the remap.
#
# Two vertical mappings (`--vertical-mapping`):
#
# - `merge_map` (default, `MergeMapLevels`): the binary's levels are merged
#   ERA5 L137 levels and its header `merge_map` sends each ERA5 level to one
#   target level. ERA5-derived binaries only.
# - `pressure_overlap` (`PressureOverlapLevels`): the binary's levels are not
#   ERA5 levels (for example MERRA-2 L72). The fields are remapped
#   horizontally at the native ERA5 levels and then, column by column on the
#   cube, distributed over the target layers in proportion to the pressure
#   overlap, with both level sets placed over the column's total surface
#   pressure. The output is ERA5 convection on the binary's own meteorology
#   and says so in its header; every section and header key of the input
#   binary is kept.

using ArgParse
using Dates
using NCDatasets
using AtmosTransport
using AtmosTransport.Parameters: ERA5_NATIVE_LEVEL_COUNT, ERA5_MET_INTERVAL_SECONDS,
                                  IFS_EARTH_RADIUS, SECONDS_PER_DAY

const MD = AtmosTransport.MetDrivers
const PP = AtmosTransport.Preprocessing
const RG = AtmosTransport.Regridding

const VERTICAL_MAPPINGS = ("merge_map", "pressure_overlap")
# The legacy archive has eight three-hour slots per day, from 00 UTC.
const CONVECTION_SLOTS_PER_DAY = 8
const CONVECTION_SLOT = Hour(3)
const CONVECTION_SLOT_SECONDS = Dates.value(Second(CONVECTION_SLOT))
@assert CONVECTION_SLOTS_PER_DAY * CONVECTION_SLOT_SECONDS == SECONDS_PER_DAY

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
        "--era5-root"
            dest_name = "era5_root"
            default = ""
            help = "ERA5 N320 native GRIB root (ml_an_native_core/) supplying the specific " *
                   "humidity for the dry-air conversion; default: the input binary's source_root " *
                   "(ERA5-derived binaries only, so required for any other dry-basis binary)"
        "--vertical-mapping"
            dest_name = "vertical_mapping"
            default = "merge_map"
            range_tester = in(VERTICAL_MAPPINGS)
            help = "merge_map: ERA5-derived binary with a header merge_map; " *
                   "pressure_overlap: binary on other hybrid levels (e.g. MERRA-2), " *
                   "column-wise pressure-overlap remap; keeps every input section"
        "--force"
            action = :store_true
            help = "replace an existing output"
    end
    return s
end

"""
    era5_level_of_file_level(ap, b, A_ifc, B_ifc) -> AbstractRange{Int}

ERA5 L137 model level (1 = top of atmosphere) of each level of a legacy TM5
convection file, from the file's layer-midpoint coefficients `ap`/`b`
compared with the ERA5 interface coefficients `A_ifc`/`B_ifc` (138 entries,
top first). The CATRINE files store levels surface first (TM5 order); the
level order is verified, never assumed, and anything other than ERA5 L137 in
either order is an error.
"""
function era5_level_of_file_level(ap::AbstractVector, b::AbstractVector,
                                  A_ifc::AbstractVector, B_ifc::AbstractVector)
    nlev = length(A_ifc) - 1
    length(ap) == length(b) == nlev ||
        error("convection file has $(length(ap)) levels; expected ERA5 L$(nlev)")
    amid = (A_ifc[1:end-1] .+ A_ifc[2:end]) ./ 2
    bmid = (B_ifc[1:end-1] .+ B_ifc[2:end]) ./ 2
    # Per-level tolerances: 1 mPa on ap, 1e-6 on b (the files match ERA5 exactly).
    for order in (1:nlev, nlev:-1:1)
        maximum(abs, ap .- amid[order]) <= 1e-3 && maximum(abs, b .- bmid[order]) <= 1e-6 &&
            return order
    end
    error("convection file levels match neither ERA5 L$(nlev) order (ap[1] = $(ap[1]), b[1] = $(b[1]))")
end

level_order_label(era5_level::AbstractRange) = step(era5_level) > 0 ? "toa_first" : "surface_first_reversed"

"""
    merge_levels!(merged, native, merge_map, era5_level)

Sum file levels into target levels: file level k is ERA5 level
`era5_level[k]`, which `merge_map` (top first) sends to a target level.
"""
function merge_levels!(merged::AbstractArray{<:Any, 3}, native::AbstractArray{<:Any, 3},
                       merge_map::AbstractVector{<:Integer}, era5_level::AbstractVector{<:Integer})
    fill!(merged, 0)
    @inbounds for k in eachindex(era5_level)
        @views merged[:, :, merge_map[era5_level[k]]] .+= native[:, :, k]
    end
    return merged
end

function merged_tm5_to_panels!(panels, dst_flat, src_flat, merged, native,
                               merge_map, era5_level, regridder, Nc)
    merge_levels!(merged, native, merge_map, era5_level)
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

"""
    convection_slot(win, dt_met_seconds) -> Int

Return the three-hour archive slot (1 to 8) that holds transport window `win`.

Window `win` covers `[(win - 1) dt, win dt)` seconds after 00 UTC. A window
that does not lie inside one slot, or that ends after the day, is an error.
"""
function convection_slot(win::Integer, dt_met_seconds::Real)
    win >= 1 && dt_met_seconds > 0 ||
        throw(ArgumentError("window $win of length $dt_met_seconds s is invalid"))
    start = (win - 1) * dt_met_seconds
    stop = win * dt_met_seconds
    slot = fld(start, CONVECTION_SLOT_SECONDS) + 1
    stop <= slot * CONVECTION_SLOT_SECONDS && slot <= CONVECTION_SLOTS_PER_DAY ||
        throw(ArgumentError("transport window $win ($start-$stop s after 00 UTC) is not " *
                            "inside one three-hour convection slot of the day"))
    return Int(slot)
end

"""
    humidity_hour(win, dt_met_seconds) -> Int

Return the UTC hour of the ERA5 humidity record at the start of window `win`.

The ERA5 N320 humidity is hourly, so a window that does not start on the hour
is an error.
"""
function humidity_hour(win::Integer, dt_met_seconds::Real)
    hours = (win - 1) * dt_met_seconds / ERA5_MET_INTERVAL_SECONDS
    isinteger(hours) && 0 <= hours < SECONDS_PER_DAY / ERA5_MET_INTERVAL_SECONDS ||
        throw(ArgumentError("transport window $win does not start on an hour of the day"))
    return Int(hours)
end

"""
    humidity_hours(basis, nwindow, dt_met_seconds) -> Union{Vector{Int}, Nothing}

Return the ERA5 humidity hour of every window, or `nothing` for a moist binary.

Only the dry-air conversion reads humidity, so only a dry-basis binary needs
windows that start on the hour.
"""
humidity_hours(::AtmosTransport.MoistBasis, nwindow, dt_met_seconds) = nothing
humidity_hours(::AtmosTransport.DryBasis, nwindow, dt_met_seconds) =
    [humidity_hour(win, dt_met_seconds) for win in 1:nwindow]

window_hour(::Nothing, win) = nothing
window_hour(hours::AbstractVector, win) = hours[win]

"""
    hybrid_interfaces!(p, A, B, ps) -> p

Fill `p` with the interface pressures `A[k] + B[k] * ps` (Pa, top first).
"""
function hybrid_interfaces!(p::AbstractVector, A::AbstractVector, B::AbstractVector, ps::Real)
    length(p) == length(A) == length(B) ||
        throw(DimensionMismatch("interface vector lengths $(length(p)), $(length(A)), $(length(B)) differ"))
    for k in eachindex(p, A, B)
        p[k] = A[k] + B[k] * ps
    end
    return p
end

_column_label(::Nothing) = ""
_column_label(column) = " in column $(column)"

"""
    remap_layers_by_pressure_overlap!(dest, src, p_src, p_dst; column = nothing) -> dest

Distribute layer totals `src` over the layers of `p_dst` in proportion to the pressure overlap.

`dest` and `src` are tuples of vectors that share one column geometry (for
example the four TM5 fields); levels and interfaces (`p_src` for the source
layers) run from the top down, so both interface vectors must be
non-decreasing. The fields are layer totals (kg m⁻² s⁻¹), so source layer `s`
gives target layer `k` the fraction `overlap(s, k) / Δp_s` of its value: each
field's column sum is preserved up to rounding. Source pressure above the top
target interface counts as part of the top target layer and pressure below the
bottom target interface as part of the bottom layer, so nothing is lost when
the two level sets do not span the same pressure range. A source layer of zero
thickness goes whole to the target layer that contains its pressure. `column`
labels error messages.
"""
function remap_layers_by_pressure_overlap!(dest::Tuple, src::Tuple,
                                           p_src::AbstractVector, p_dst::AbstractVector;
                                           column = nothing)
    length(dest) == length(src) ||
        throw(DimensionMismatch("$(length(dest)) destination and $(length(src)) source fields"))
    ns, nd = length(p_src) - 1, length(p_dst) - 1
    ns >= 1 && nd >= 1 || throw(DimensionMismatch("need at least one source and one target layer"))
    all(x -> length(x) == ns, src) ||
        throw(DimensionMismatch("source fields must have $(ns) layers"))
    all(x -> length(x) == nd, dest) ||
        throw(DimensionMismatch("destination fields must have $(nd) layers"))
    issorted(p_src) || throw(ArgumentError(
        "source interfaces must increase from the top down$(_column_label(column)): $(p_src)"))
    issorted(p_dst) || throw(ArgumentError(
        "target interfaces must increase from the top down$(_column_label(column)): $(p_dst)"))
    foreach(x -> fill!(x, 0), dest)
    k = 1
    @inbounds for s in 1:ns
        top, bottom = p_src[s], p_src[s + 1]
        # First target layer whose bottom interface lies below the source top.
        while k < nd && p_dst[k + 1] <= top
            k += 1
        end
        thickness = bottom - top
        if thickness == 0
            for f in eachindex(dest)
                dest[f][k] += src[f][s]
            end
            continue
        end
        level, upper = k, top
        while true
            lower = level == nd ? bottom : min(bottom, p_dst[level + 1])
            weight = (lower - upper) / thickness
            for f in eachindex(dest)
                dest[f][level] += weight * src[f][s]
            end
            (lower >= bottom || level == nd) && break
            upper = lower
            level += 1
        end
    end
    return dest
end

"""
    surface_pressure_basis_label(basis) -> String

Describe the surface pressure that places the hybrid interfaces of both level sets.
"""
surface_pressure_basis_label(::TotalSurfacePressure) =
    "total surface pressure stored in the binary (ps)"
surface_pressure_basis_label(::DrySurfacePressure) =
    "total surface pressure recovered from the binary's dry ps and the window-start " *
    "VDIFF humidity (vdiff_qv): ps_dry = sum_k (1 - qv_k) (dA_k + dB_k ps_total)"

"""
    total_surface_pressure(basis, ps, A, B, qv) -> Float64

Return the total surface pressure of one target column from its stored `ps`.

The binary's hybrid levels are total-pressure levels, `A + B ps_total`; a dry
`ps` is converted with the column humidity `qv`, a total `ps` is returned as
stored.
"""
# This private function is used on purpose: it is the total-pressure recovery
# of the initial conditions, so both place the binary's levels identically.
# The test of `total_surface_pressure` fails if it is renamed or changed.
total_surface_pressure(basis::AtmosTransport.AbstractSurfacePressureBasis, ps, A, B, qv) =
    AtmosTransport.Models.InitialConditionIO._total_surface_pressure(basis, ps, A, B, qv)

"""
    column_humidity(basis, window) -> humidity panels or nothing

Return the humidity panels that a stored surface pressure of `basis` needs.

A dry `ps` needs the window-start VDIFF humidity `vdiff_qv`; a total `ps`
needs none.
"""
column_humidity(::TotalSurfacePressure, window) = nothing
column_humidity(::DrySurfacePressure, window) = window.vdiff.qv

"""
    total_surface_pressure_panels(basis, ps, qv, A, B) -> NTuple of matrices

Return the total surface pressure of every column, checked to be finite and positive.

Errors name the panel and the column `(i, j)`.
"""
function total_surface_pressure_panels(basis, ps, qv, A, B)
    return ntuple(length(ps)) do p
        out = zeros(Float64, size(ps[p]))
        for j in axes(out, 2), i in axes(out, 1)
            column_qv = qv === nothing ? nothing : view(qv[p], i, j, :)
            value = try
                total_surface_pressure(basis, ps[p][i, j], A, B, column_qv)
            catch err
                err isa ArgumentError || rethrow()
                throw(ArgumentError("panel $p, column ($i, $j): $(err.msg)"))
            end
            isfinite(value) && value > 0 || throw(ArgumentError(
                "panel $p, column ($i, $j): total surface pressure $(value) Pa " *
                "(stored ps $(ps[p][i, j])) is not finite and positive"))
            out[i, j] = value
        end
        out
    end
end

"""
    pressure_overlap_to_panels!(tm5, dst_flats, ps, qv, basis, A_src, B_src, A_dst, B_dst, Nc) -> tm5

Remap the regridded native-level TM5 fields to the target levels of every cube column.

`dst_flats` holds the four fields on the cube at the source levels,
`(ncell, nsrc)` each with cells in panel-major order and levels top first.
For each column, both level sets are placed over the column's total surface
pressure (from the stored `ps` panels, humidity panels `qv` or `nothing`, and
the stored surface-pressure `basis`) and
[`remap_layers_by_pressure_overlap!`](@ref) writes the target-level fields
into the Float32 panels of `tm5`. Panels run on separate threads.
"""
function pressure_overlap_to_panels!(tm5, dst_flats, ps, qv, basis,
                                     A_src, B_src, A_dst, B_dst, Nc)
    nsrc, ndst = length(A_src) - 1, length(A_dst) - 1
    np = length(tm5.entu)
    length(B_src) == nsrc + 1 && length(B_dst) == ndst + 1 ||
        throw(DimensionMismatch("hybrid A and B coefficients differ in length"))
    length(dst_flats) == 4 && all(f -> size(f) == (np * Nc * Nc, nsrc), dst_flats) ||
        throw(DimensionMismatch("regridded fields must be four ($(np * Nc * Nc), $nsrc) arrays"))
    all(fields -> length(fields) == np && all(x -> size(x) == (Nc, Nc, ndst), fields), tm5) ||
        throw(DimensionMismatch("TM5 panels must be $np arrays of size ($Nc, $Nc, $ndst)"))
    length(ps) == np && all(x -> size(x) == (Nc, Nc), ps) ||
        throw(DimensionMismatch("surface pressure must be $np ($Nc, $Nc) panels"))
    qv === nothing || length(qv) == np && all(x -> size(x) == (Nc, Nc, ndst), qv) ||
        throw(DimensionMismatch("humidity must be $np ($Nc, $Nc, $ndst) panels"))
    ps_total = total_surface_pressure_panels(basis, ps, qv, A_dst, B_dst)
    cells_per_panel = Nc * Nc
    Threads.@threads for p in 1:np
        src = ntuple(_ -> zeros(nsrc), 4)
        dest = ntuple(_ -> zeros(ndst), 4)
        p_src = zeros(nsrc + 1)
        p_dst = zeros(ndst + 1)
        panels = (tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p])
        # Every index below lies inside the sizes checked above.
        @inbounds for j in 1:Nc, i in 1:Nc
            hybrid_interfaces!(p_src, A_src, B_src, ps_total[p][i, j])
            hybrid_interfaces!(p_dst, A_dst, B_dst, ps_total[p][i, j])
            flat_cell = (p - 1) * cells_per_panel + (j - 1) * Nc + i
            for f in 1:4, s in 1:nsrc
                src[f][s] = dst_flats[f][flat_cell, s]
            end
            remap_layers_by_pressure_overlap!(dest, src, p_src, p_dst; column = (p, i, j))
            for f in 1:4, k in 1:ndst
                panels[f][i, j, k] = Float32(dest[f][k])
            end
        end
    end
    return tm5
end

"""
    native_tm5_to_cube!(dst_flat, src_flat, native, era5_level, regridder) -> dst_flat

Regrid one file-order TM5 field to the cube at the native ERA5 levels, top first.

`native` is `(nlon, nlat, nlev)` in file order; column `era5_level[k]` of
`src_flat` takes file level `k`, so `dst_flat` is `(ncell, nlev)` with cells in
panel-major order and levels top first.
"""
function native_tm5_to_cube!(dst_flat, src_flat, native::AbstractArray{<:Any, 3},
                             era5_level, regridder)
    ncell = size(native, 1) * size(native, 2)
    size(src_flat) == (ncell, length(era5_level)) == (ncell, size(native, 3)) ||
        throw(DimensionMismatch("source buffer $(size(src_flat)) for a $(size(native)) field"))
    for (k, level) in enumerate(era5_level)
        copyto!(view(src_flat, :, level), vec(view(native, :, :, k)))
    end
    RG.apply_regridder!(dst_flat, regridder, src_flat)
    return dst_flat
end

# ---------------------------------------------------------------------------
# Vertical mappings
# ---------------------------------------------------------------------------

"""
    AbstractVerticalMapping

How the attachment maps ERA5 L137 convection onto the levels of the input binary.
"""
abstract type AbstractVerticalMapping end

"""
    MergeMapLevels(merge_map)

Map ERA5 levels through the binary's `merge_map` (ERA5-derived binaries).
"""
struct MergeMapLevels <: AbstractVerticalMapping
    merge_map :: Vector{Int}
end

"""
    PressureOverlapLevels(A_src, B_src)

Map ERA5 L137 layers (interfaces `A_src + B_src ps`) onto any hybrid levels by pressure overlap.
"""
struct PressureOverlapLevels <: AbstractVerticalMapping
    A_src :: Vector{Float64}
    B_src :: Vector{Float64}
end

"""
    vertical_mapping(name, h, ps_basis) -> AbstractVerticalMapping

Build the mapping `name` (`"merge_map"` or `"pressure_overlap"`) for header `h` and check that the binary supports it.
"""
function vertical_mapping(name::AbstractString, h, ps_basis)
    name == "merge_map" && return merge_map_levels(h)
    name == "pressure_overlap" && return pressure_overlap_levels(h, ps_basis)
    throw(ArgumentError("unknown vertical mapping $(repr(name)); expected one of $(VERTICAL_MAPPINGS)"))
end

function merge_map_levels(h)
    # The merge_map decides whether this mode applies at all, so it is checked first.
    merge_map = Int.(collect(get(h.raw_header, "merge_map", Int[])))
    length(merge_map) == ERA5_NATIVE_LEVEL_COUNT ||
        error("expected a $(ERA5_NATIVE_LEVEL_COUNT)-entry merge_map; found $(length(merge_map)); " *
              "use --vertical-mapping pressure_overlap for binaries on other levels")
    maximum(merge_map) == h.nlevel || error("merge_map target does not match nlevel=$(h.nlevel)")
    issorted(merge_map) && first(merge_map) == 1 ||
        error("merge_map must run from target level 1 (top) to $(h.nlevel) (surface)")
    all(s in h.payload_sections for s in (:pblh, :ustar, :pbl_hflux, :t2m, :dkg)) ||
        error("input must contain dm, complete PBL surface fields, and dkg")
    return MergeMapLevels(merge_map)
end

function pressure_overlap_levels(h, ps_basis)
    ps_basis isa DrySurfacePressure && !(:vdiff_qv in h.payload_sections) &&
        error("a dry surface pressure needs the vdiff_qv humidity to place the hybrid " *
              "levels over total pressure; the input has none")
    vc = PP.load_hybrid_coefficients(ERA5_L137_COEFFICIENTS)
    return PressureOverlapLevels(Float64.(vc.A), Float64.(vc.B))
end

"""
    mapping_workspace(mapping, nsrc, ndst, Nz) -> NamedTuple

Allocate the buffers of `mapping` for `nsrc` 1-degree cells, `ndst` cube cells and `Nz` target levels.
"""
mapping_workspace(::MergeMapLevels, nsrc, ndst, Nz) =
    (merged = zeros(Float64, 360, 180, Nz), src_flat = zeros(Float64, nsrc, Nz),
     dst_flat = zeros(Float64, ndst, Nz))
mapping_workspace(::PressureOverlapLevels, nsrc, ndst, Nz) =
    (src_flat = zeros(Float64, nsrc, ERA5_NATIVE_LEVEL_COUNT),
     native_flats = ntuple(_ -> zeros(Float64, ndst, ERA5_NATIVE_LEVEL_COUNT), 4))

"""
    cube_fields!(tm5, mapping, workspace, fields, era5_level, regridder, window, h, ps_basis) -> tm5

Put the file-order 1-degree TM5 `fields` of one window on the cube and the binary's levels.

`window` is the input binary's window (its surface pressure and humidity
place the levels in the pressure-overlap mapping).
"""
function cube_fields!(tm5, mapping::MergeMapLevels, ws, fields, era5_level, regridder,
                      window, h, ps_basis)
    for (panels, field) in zip(tm5, fields)
        merged_tm5_to_panels!(panels, ws.dst_flat, ws.src_flat, ws.merged,
            field, mapping.merge_map, era5_level, regridder, h.geometry.Nc)
    end
    return tm5
end

function cube_fields!(tm5, mapping::PressureOverlapLevels, ws, fields, era5_level, regridder,
                      window, h, ps_basis)
    for (flat, field) in zip(ws.native_flats, fields)
        native_tm5_to_cube!(flat, ws.src_flat, field, era5_level, regridder)
    end
    return pressure_overlap_to_panels!(tm5, ws.native_flats, window.ps,
                                       column_humidity(ps_basis, window), ps_basis,
                                       mapping.A_src, mapping.B_src, h.A_ifc, h.B_ifc,
                                       h.geometry.Nc)
end

"""
    host_label(raw) -> String

Name the meteorology of the input binary from its `source_type`, else its `preprocessor`.
"""
function host_label(raw::AbstractDict)
    for key in ("source_type", "preprocessor")
        value = get(raw, key, "")
        value isa AbstractString && !isempty(value) && return String(value)
    end
    error("the input binary records neither source_type nor preprocessor; its meteorology " *
          "cannot be named in the output header")
end

"""
    temporal_mapping_label(dt_met_seconds) -> String

Describe how the three-hour slots are held over transport windows of `dt_met_seconds`.
"""
temporal_mapping_label(dt_met_seconds) =
    dt_met_seconds == ERA5_MET_INTERVAL_SECONDS ?
    "three-hour slots centred at 01:30, 04:30, ... UTC, each held for its three hourly transport windows" :
    "three-hour slots from 00 UTC, each held for the $(dt_met_seconds)-second transport windows inside it"

"""
    mapping_metadata(mapping, raw, h, ps_basis, basis) -> Dict{String, Any}

Return the header entries that describe `mapping`.

The pressure-overlap entries say that the convection is ERA5-based (legacy
1-degree, 3-hourly TM5 archive, ERA5 L137) on the input binary's meteorology
(named by [`host_label`](@ref)), how it was mapped in time and in the
vertical, and record the input's `git_commit` and `git_dirty` as
`host_git_commit` and `host_git_dirty`. The input's other header keys are
added by [`add_host_header_keys!`](@ref).
"""
mapping_metadata(mapping::MergeMapLevels, raw, h, ps_basis, basis) =
    Dict{String, Any}("merge_map" => mapping.merge_map)

function mapping_metadata(::PressureOverlapLevels, raw::AbstractDict, h, ps_basis, basis)
    host = host_label(raw)
    conversion = basis isa AtmosTransport.DryBasis ?
        ", converted to dry air with the ERA5 N320 humidity" : ""
    relation = startswith(lowercase(host), "era5") ?
        "an ERA5-based binary on other levels (host: $host); the convection's 1-degree, " *
        "3-hourly sampling and levels differ from the binary's" :
        "a binary of different meteorology (host: $host); winds, air mass, humidity, boundary " *
        "layer and any other convection sections are the host's, and the TM5 fields are not " *
        "consistent with them"
    humidity_timing = basis isa AtmosTransport.DryBasis ?
        "; humidity of the hour at each window start" : ""
    metadata = Dict{String, Any}(
        "validation_status" => "era5_convection_attached_by_pressure_overlap; not_a_consistent_product",
        "experimental_caveat" =>
            "ERA5-based TM5 convection (legacy CATRINE 1-degree, 3-hourly TM5 archive on " *
            "ERA5 L137$conversion) attached by pressure overlap to $relation. " *
            "Sensitivity experiments only.",
        "convection_attachment" => "legacy_catrine_1deg_3hour_tm5_to_cs_pressure_overlap_v1",
        "convection_meteorology" => "ERA5 (legacy CATRINE TM5 archive, 1 degree, 3-hourly, L137)",
        "convection_host_meteorology" => host,
        "convection_source_levels" => "ERA5 L137",
        "convection_vertical_mapping" =>
            "per cube column after the horizontal regrid: ERA5 L137 interfaces A + B ps_total and " *
            "target interfaces A_ifc + B_ifc ps_total over the same column total surface pressure; " *
            "each ERA5 layer total distributed over the target layers in proportion to pressure " *
            "overlap (column sums preserved); ERA5 pressure above the target top goes to the top layer",
        "convection_vertical_surface_pressure" => surface_pressure_basis_label(ps_basis),
        "convection_temporal_mapping" => temporal_mapping_label(h.dt_met_seconds) * humidity_timing,
    )
    for key in ("git_commit", "git_dirty")
        haskey(raw, key) && (metadata["host_" * key] = raw[key])
    end
    return metadata
end

"""
    add_host_header_keys!(writer, mapping, raw) -> writer

Copy every input header key that the output header does not set into the open `writer`.

Only the pressure-overlap mapping keeps the host's own keys (the provenance of
its meteorology); keys the writer or the attachment set, including the
structural ones, keep the output's values. The header is written again when
the writer closes.
"""
add_host_header_keys!(writer, ::MergeMapLevels, raw) = writer
function add_host_header_keys!(writer, ::PressureOverlapLevels, raw::AbstractDict)
    for (key, value) in raw
        key = String(key)
        key in ("git_commit", "git_dirty") && continue     # kept as host_git_*
        haskey(writer.header, key) || (writer.header[key] = value)
    end
    return writer
end

# Revision of the code that wrote the attachment ("" outside a git checkout).
_git_revision() = try
    String(strip(read(pipeline(`git -C $(@__DIR__) rev-parse HEAD`; stderr = devnull), String)))
catch
    ""
end

const ERA5_L137_COEFFICIENTS = joinpath(@__DIR__, "..", "..", "config", "era5_L137_coefficients.toml")
function copy_source_metadata!(metadata::AbstractDict, raw::AbstractDict,
                               dry_target = nothing)
    for key in ("adaptive_substeps", "preprocessor_contract", "runtime_substep_contract",
                "poisson_balanced", "vertical_transform",
                "target_coefficients", "vertical_mapping_method", "target_vertical_name",
                "coarsening_method", "coarsening_source_Nc", "coarsening_target_Nc",
                "coarsening_ratio", "horizontal_balance", "column_balance_weights")
        haskey(raw, key) && (metadata[key] = raw[key])
    end
    # The write-time flux certificate (flux and OMEGA fidelity, balance
    # correction, column closure) describes the m/am/bm/cm payload, which the
    # attachment copies unchanged: carry it over verbatim.
    for (key, value) in raw
        MD.is_certificate_key(String(key)) && (metadata[String(key)] = value)
    end
    pin_enabled = AtmosTransport.MetDrivers.dry_mass_pin_enabled(raw, "source metadata")
    if pin_enabled !== nothing
        metadata["mass_fix_enabled"] = pin_enabled
        if pin_enabled
            recorded = get(raw, "mass_fix_target_ps_dry_pa",
                           dry_target === nothing ? nothing : dry_target.ps_dry_pa)
            recorded isa Real && !(recorded isa Bool) || error(
                "pinned source metadata lacks numeric mass_fix_target_ps_dry_pa")
            target = Float64(recorded)
            isfinite(target) && target > 0 || error(
                "invalid source mass_fix_target_ps_dry_pa=$(repr(recorded))")
            metadata["mass_fix_target_ps_dry_pa"] = target
        end
    end
    return metadata
end

# The YYYYMMDD field of `convec_YYYYMMDD_00p03.nc` / `..._transport_YYYYMMDD_float32.bin`.
_date_label(path) = (m = match(r"_(\d{8})[_.]", basename(path)); m === nothing ? nothing : m.captures[1])

"""
    check_convection_file(ds, conv_path, input) -> era5_level

Verify the legacy file before any output is written: the 1-degree grid the
regridder assumes (longitudes -179.5..179.5, latitudes -89.5..89.5, both
ascending), eight three-hour slots starting at 00 UTC of the file's date, the
same date as the transport binary, and the level order (returned as the ERA5
level of each file level). Slot times come from `timevalues_bounds`: the CF
`time` variable is corrupt in the 2022-04-30 to 2023-12-31 files (it decodes
to the 1880s) and changes epoch in 2024.
"""
function check_convection_file(ds, conv_path, input)
    for name in ("eu", "du", "ed", "dd")
        haskey(ds, name) || error("convection NC missing $name")
        size(ds[name]) == (360, 180, ERA5_NATIVE_LEVEL_COUNT, CONVECTION_SLOTS_PER_DAY) ||
            error("unexpected $name shape $(size(ds[name]))")
    end
    ds["lon"][:] == collect(-179.5:1.0:179.5) ||
        error("convection longitudes are not -179.5:1:179.5 in $conv_path")
    ds["lat"][:] == collect(-89.5:1.0:89.5) ||
        error("convection latitudes are not -89.5:1:89.5 (south to north) in $conv_path")
    day = _date_label(conv_path)
    day === nothing && error("cannot read a YYYYMMDD date from $conv_path")
    day == _date_label(input) ||
        error("convection file date $day does not match transport binary $(basename(input))")
    start = DateTime(Date(day, dateformat"yyyymmdd"))
    # (nv, timeval, time) in Julia order; timeval = year month day hour minute second
    bounds = Array(ds["timevalues_bounds"])
    size(bounds) == (2, 6, CONVECTION_SLOTS_PER_DAY) ||
        error("unexpected timevalues_bounds shape $(size(bounds)) in $conv_path")
    stamp(v) = DateTime(v...)
    for k in 1:CONVECTION_SLOTS_PER_DAY
        first_hour, last_hour = (k - 1) * CONVECTION_SLOT, k * CONVECTION_SLOT
        covers = stamp(bounds[1, :, k]) == start + first_hour &&
                 stamp(bounds[2, :, k]) == start + last_hour
        covers || error("convection slot $k in $conv_path does not cover hours " *
                        "$(Dates.value(first_hour))-$(Dates.value(last_hour)) UTC of $day")
    end
    vc = PP.load_hybrid_coefficients(ERA5_L137_COEFFICIENTS)
    return era5_level_of_file_level(Float64.(ds["ap"][:]), Float64.(ds["b"][:]),
                                    Float64.(vc.A), Float64.(vc.B))
end

# Entrainment-weighted mean target level of the updraft (k = 1 top, Nz
# surface). Updrafts entrain in the lower troposphere, so a mean in the upper
# half of the column means the profile was stored upside down.
function updraft_mean_level(entu)
    weight = 0.0
    level = 0.0
    for panel in entu, k in axes(panel, 3)
        w = sum(Float64, @view panel[:, :, k])
        weight += w
        level += k * w
    end
    return weight > 0 ? level / weight : NaN
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

# Moist (total-air) archive fluxes are converted to the binary's air-mass
# basis with the specific humidity of each window; a moist binary needs none.
humidity_target_mesh(mesh::AtmosTransport.LatLonMesh) =
    AtmosTransport.LatLonMesh(mesh.λᶠ, mesh.φᶠ;
                              FT = eltype(mesh), radius = IFS_EARTH_RADIUS)

humidity_remap(::AtmosTransport.MoistBasis, raw_header, era5_root, date, mesh, cache_dir) = nothing
function humidity_remap(::AtmosTransport.DryBasis, raw_header, era5_root, date,
                        mesh, cache_dir; opener = PP.open_era5_humidity_remap)
    # Specific humidity is intensive, so changing only the sphere radius does
    # not change its target values. ConservativeRegridding requires both
    # manifolds to match: use the N320/IFS radius here, while the subsequent
    # extensive convection-field remap keeps the transport binary's radius.
    target = humidity_target_mesh(mesh)
    return opener(
        isempty(era5_root) ? era5_root_of(raw_header) : era5_root,
        date, target; cache_dir)
end

# The ERA5 N320 archive a no-convection binary was built from.
function era5_root_of(raw_header)
    get(raw_header, "source_type", "") == "era5_n320_native_grib" && haskey(raw_header, "source_root") ||
        error("the input binary does not record its ERA5 N320 source_root; pass --era5-root")
    return String(raw_header["source_root"])
end

"""
    humidity_source_label(remap, dt_met_seconds) -> String

Describe the humidity of the dry-air conversion for transport windows of `dt_met_seconds`.
"""
humidity_source_label(::Nothing, dt_met_seconds) = "none (moist binary)"
humidity_source_label(remap, dt_met_seconds) =
    "ERA5 N320 specific humidity at the start of each " *
    (dt_met_seconds == ERA5_MET_INTERVAL_SECONDS ? "hourly window" :
                                                   "$(dt_met_seconds)-second transport window") *
    " ($(remap.handles.core_path)), conservatively remapped to the 1-degree convection grid"

"""
    window_fields_to_basis!(out, slot, basis, remap, hour, era5_level, columns) -> out

Copy the file-level TM5 fields of a three-hour slot (`slot`, NamedTuple of
`(360, 180, 137)` arrays `eu, du, ed, dd`, surface first) into `out` and put
them on the binary's air-mass `basis` with the specific humidity of `hour`
from `remap`: column by column at the native ERA5 levels (top first), by
`tm5_convection_to_basis!`. `columns` holds five length-137 scratch vectors.
"""
function window_fields_to_basis!(out, slot, ::AtmosTransport.MoistBasis, ::Nothing,
                                 hour, era5_level, columns)
    foreach(copyto!, out, slot)
    return out
end

function window_fields_to_basis!(out, slot, basis::AtmosTransport.DryBasis, remap,
                                 hour, era5_level, columns)
    nlon, nlat, nlev = size(slot.eu)
    q = reshape(PP.remap_era5_humidity!(remap, hour), nlon, nlat, nlev)   # top first
    return file_fields_to_basis!(out, slot, basis, q, era5_level, columns)
end

"""
    file_fields_to_basis!(out, slot, basis, q, era5_level, columns) -> out

Convert one file-order TM5 archive slot using top-first layer-centre humidity
`q`. This is the pure conversion step used after the N320 humidity remap and
is separate so the attachment route can be regression-tested without GRIB
input.
"""
function file_fields_to_basis!(out, slot, basis::AtmosTransport.DryBasis,
                               q::AbstractArray{<:Any, 3}, era5_level, columns)
    nlon, nlat, nlev = size(slot.eu)
    size(q) == (nlon, nlat, nlev) ||
        throw(DimensionMismatch("humidity shape $(size(q)) does not match TM5 slot $(size(slot.eu))"))
    eu, du, ed, dd, qc = columns
    @inbounds for j in 1:nlat, i in 1:nlon
        for (k, level) in enumerate(era5_level)
            eu[level] = slot.eu[i, j, k]; du[level] = slot.du[i, j, k]
            ed[level] = slot.ed[i, j, k]; dd[level] = slot.dd[i, j, k]
        end
        for level in 1:nlev
            qc[level] = q[i, j, level]
        end
        PP.tm5_convection_to_basis!(basis, eu, du, ed, dd, qc)
        for (k, level) in enumerate(era5_level)
            out.eu[i, j, k] = eu[level]; out.du[i, j, k] = du[level]
            out.ed[i, j, k] = ed[level]; out.dd[i, j, k] = dd[level]
        end
    end
    return out
end

"""
    open_attachment_writer(output, Nc, np, Nz, h, g, vc, metadata) -> writer

Open the output binary with every payload section of the input header `h` plus the TM5 sections.

The optional sections (`dm`, PBL surface fields, `cmfmc`, `dtrain`, GCHP
VDIFF, `dkg`, `pbl_eflux`, `cmfmc_cloud_base`) follow `h.payload_sections`; the
writer's section order puts the TM5 sections between `dtrain` and the VDIFF
fields. An input with the TM5 PBL group (version 5 only) is rejected.
"""
function open_attachment_writer(output, Nc, np, Nz, h, g, vc, metadata)
    sections = h.payload_sections
    has(section) = section in sections
    hasall(group) = all(has, group)
    any(has, MD.TM5_PBL_PAYLOAD_SECTIONS) &&
        error("the input carries the TM5 PBL group; attaching to version-5 binaries is not supported")
    for group in (MD.PBL_SURFACE_PAYLOAD_SECTIONS, MD.GCHP_VDIFF_PAYLOAD_SECTIONS)
        any(has, group) == hasall(group) ||
            error("the input carries part of the section group $(group)")
    end
    writer = MD.open_streaming_cs_transport_binary(
        output, Nc, np, Nz, h.nwindow, vc;
        FT = Float32, planet_radius = h.planet_radius_m, dt_met_seconds = h.dt_met_seconds,
        half_dt_seconds = h.half_dt_seconds, steps_per_window = 1,
        source_flux_sampling = h.source_flux_sampling,
        air_mass_sampling = h.air_mass_sampling,
        flux_sampling = h.flux_sampling, flux_kind = h.flux_kind,
        include_flux_delta = has(:dm), mass_basis = h.mass_basis,
        surface_pressure_basis = h.surface_pressure_basis,
        tm5_convection_mass_basis = AtmosTransport.State.mass_basis(h.mass_basis),
        include_surface = hasall(MD.PBL_SURFACE_PAYLOAD_SECTIONS),
        include_cmfmc = has(:cmfmc), include_dtrain = has(:dtrain),
        include_tm5conv = true,
        include_gchp_vdiff = hasall(MD.GCHP_VDIFF_PAYLOAD_SECTIONS),
        include_precomputed_dkg = has(:dkg),
        include_pbl_eflux = has(:pbl_eflux),
        include_cmfmc_cloud_base = has(:cmfmc_cloud_base),
        panel_convention = String(g.panel_convention),
        cs_definition = String(g.definition),
        cs_coordinate_law = String(g.coordinate_law),
        cs_center_law = String(g.center_law),
        longitude_offset_deg = g.longitude_offset_deg,
        extra_header = metadata)
    kept = filter(s -> !(s in MD.TM5_CONVECTION_PAYLOAD_SECTIONS), writer.payload_sections)
    if kept != collect(sections)
        MD.abandon_streaming_transport_binary!(writer)
        error("output sections $(writer.payload_sections) do not keep the input sections $(sections)")
    end
    return writer
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
    humidity = nothing
    started = time()
    try
        h = reader.header
        g = h.geometry
        MD.grid_type(reader) === :cubed_sphere || error("input must be cubed-sphere")
        MD.has_tm5_convection(reader) && error("input already contains TM5 convection")
        # The reader and the writer work in Float32 and the writer emits version
        # 4: anything else would be changed while the header claims a copy.
        h.format_version == MD.TRANSPORT_BINARY_FORMAT_VERSION ||
            error("input is format_version $(h.format_version); the attachment writes version " *
                  "$(MD.TRANSPORT_BINARY_FORMAT_VERSION) and would drop its flux-construction " *
                  "identity, so only version-$(MD.TRANSPORT_BINARY_FORMAT_VERSION) inputs are accepted")
        h.on_disk_float_type === :Float32 ||
            error("input stores $(h.on_disk_float_type); the attachment copies sections as Float32 only")
        ps_basis = MD.surface_pressure_basis(reader)
        mapping = vertical_mapping(args["vertical_mapping"], h, ps_basis)
        mapping isa PressureOverlapLevels &&
            @warn "Pressure-overlap mode: ERA5-based convection on the input binary's own " *
                  "meteorology; the result is not a consistent product"
        :dm in h.payload_sections || error("input must contain dm")
        h.nwindow * h.dt_met_seconds == SECONDS_PER_DAY ||
            error("expected windows covering one day; found $(h.nwindow) x $(h.dt_met_seconds) s")
        basis = AtmosTransport.State.mass_basis(h.mass_basis)
        slots = [convection_slot(win, h.dt_met_seconds) for win in 1:h.nwindow]
        hours = humidity_hours(basis, h.nwindow, h.dt_met_seconds)
        era5_level = NCDataset(ds -> check_convection_file(ds, conv_path, input), conv_path, "r")
        level_order = level_order_label(era5_level)
        @info "Convection file level order: $(level_order)"

        target_grid = MD.load_grid(reader; FT = Float64, Hp = 0)
        source_mesh = AtmosTransport.LatLonMesh(
            ; FT = Float64, size = (360, 180), longitude = (-180, 180),
              latitude = (-90, 90), radius = target_grid.horizontal.radius)
        regridder = RG.build_regridder(
            source_mesh, target_grid.horizontal; normalize = false,
            cache_dir = abspath(args["cache_dir"]))

        humidity = humidity_remap(basis, h.raw_header, args["era5_root"],
                                  Date(_date_label(input), dateformat"yyyymmdd"),
                                  source_mesh, abspath(args["cache_dir"]))
        humidity_label = humidity_source_label(humidity, h.dt_met_seconds)
        @info "Convection fluxes on the $(h.mass_basis) basis; humidity: $(humidity_label)"

        Nc, Nz, np = g.Nc, h.nlevel, g.npanel
        ws = mapping_workspace(mapping, length(regridder.src_areas), length(regridder.dst_areas), Nz)
        native_shape = (360, 180, ERA5_NATIVE_LEVEL_COUNT)
        slot = (eu = zeros(native_shape), du = zeros(native_shape),
                ed = zeros(native_shape), dd = zeros(native_shape))
        on_basis = map(similar, slot)
        columns = ntuple(_ -> zeros(ERA5_NATIVE_LEVEL_COUNT), 5)
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
            # v4: fluxes put on the binary's air-mass basis (v3: level order fixed).
            "convection_attachment" => "legacy_catrine_1deg_3hour_tm5_to_cs_v4",
            "convection_humidity_source" => humidity_label,
            "convection_source_level_order" => level_order,
            "convection_attach_git_revision" => _git_revision(),
            "convection_target_grid" => "C$(Nc)_L$(Nz)",
            "convection_source" => conv_path,
            "convection_temporal_mapping" => temporal_mapping_label(h.dt_met_seconds),
            "source_binary" => input,
        )
        merge!(metadata, mapping_metadata(mapping, raw, h, ps_basis, basis))
        copy_source_metadata!(metadata, raw, MD.binary_dry_mass_target(reader))
        # The input's own coefficients, unrounded: a Float32 copy would change
        # the header of a binary whose A/B are not Float32 values (MERRA-2).
        vc = AtmosTransport.HybridSigmaPressure(Vector{Float64}(h.A_ifc), Vector{Float64}(h.B_ifc))
        writer = open_attachment_writer(output, Nc, np, Nz, h, g, vc, metadata)
        add_host_header_keys!(writer, mapping, raw)

        worst_closure = 0.0
        NCDataset(conv_path, "r") do ds
            for win in 1:h.nwindow
                tidx = slots[win]
                if win == 1 || tidx != slots[win - 1]
                    for (field, name) in zip(slot, ("eu", "du", "ed", "dd"))
                        field .= ds[name][:, :, :, tidx]
                    end
                end
                # The humidity of a dry binary changes every window start.
                window_fields_to_basis!(on_basis, slot, basis, humidity, window_hour(hours, win),
                                        era5_level, columns)
                source = MD.load_window!(reader, win)
                cube_fields!(tm5, mapping, ws, on_basis, era5_level, regridder, source, h, ps_basis)
                worst_closure = max(worst_closure, closure_error(tm5))
                mean_level = updraft_mean_level(tm5.entu)
                isfinite(mean_level) ||
                    error("updraft entrainment in slot $tidx of $conv_path is all zero or not finite")
                mean_level > Nz / 2 ||
                    error("updraft entrainment mean level $(round(mean_level; digits = 1)) of $Nz " *
                          "is in the upper half of the column (slot $tidx); level order is wrong")
                dm = MD.load_flux_delta_window!(reader, win).dm
                # Every input section passes through unchanged; only the TM5 fields are new.
                payload = merge(source, (; dm, tm5_fields = tm5))
                MD.write_streaming_cs_window!(writer, payload, Nc, np)
            end
        end
        # Updraft and downdraft columns must each close (entrainment = detrainment);
        # the legacy files close to ~3e-7.
        isfinite(worst_closure) && worst_closure < 1e-4 ||
            error("convection column closure error $(worst_closure) exceeds 1e-4")
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
        writer === nothing || MD.abandon_streaming_transport_binary!(writer)
        rethrow()
    finally
        humidity === nothing || close(humidity)
        close(reader)
    end
end

# Run only as a script, so tests can include the helpers.
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
