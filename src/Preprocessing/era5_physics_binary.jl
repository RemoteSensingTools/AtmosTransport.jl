# ---------------------------------------------------------------------------
# ERA5 physics binary format.
#
# Two-layer design per DATA_LAYOUT.md's NVMe staging strategy:
#
#   1. Archive tier — HDD, permanent, zlib-compressed NetCDF:
#      `~/data/.../met/era5/<grid>/physics/era5_convection_YYYYMMDD.nc`
#      `~/data/.../met/era5/<grid>/physics/era5_thermo_ml_YYYYMMDD.nc`
#
#   2. Staging tier — NVMe, rolling window, uncompressed float32:
#      `/temp1/.../met/era5/<grid>/physics_bin/YYYY/era5_physics_YYYYMMDD.bin`
#
# The staging BIN is what the preprocessor reads.  It's mmap-friendly
# (no decompression overhead), ~18 GB per day at 0.5° × 137L × 24h,
# and gets pruned after 1-2 days post-use by a housekeeping script.
# Peak NVMe occupancy ≤ ~60 GB (3 days × 18 GB) independent of
# simulation length.
#
# Format:
#   - 4 KB JSON header (printf-padded with trailing nulls).
#   - Flat float32 payload, one variable at a time:
#       udmf[Nlon, Nlat, Nlev, 24]
#       ddmf[Nlon, Nlat, Nlev, 24]
#       udrf_rate[Nlon, Nlat, Nlev, 24]
#       ddrf_rate[Nlon, Nlat, Nlev, 24]
#       t[Nlon, Nlat, Nlev, 24]
#       q[Nlon, Nlat, Nlev, 24]
#
# Calendar-day aligned: BIN for date D covers hours 00:00-23:00 of D.
# Thermo NC is already calendar-day aligned; convection NC is
# forecast-based (07:00 day D through 06:00 day D+1) so building a
# calendar-day BIN requires TWO adjacent convection NCs (D-1 for
# hours 00-06, D for hours 07-23).
#
# Latitude: stored S→N in the BIN (AtmosTransport orientation).
# ERA5 NCs are N→S; the converter flips on write.
# Hybrid level: 1 = TOA, Nlev = surface (AtmosTransport orientation,
# matches ERA5 native).
# ---------------------------------------------------------------------------

using Mmap
using JSON3
using NCDatasets
using Dates

const ERA5_PHYSICS_BINARY_MAGIC        = "ERA5PHYS"
const ERA5_PHYSICS_BINARY_VERSION      = 1
const ERA5_PHYSICS_BINARY_HEADER_BYTES = 4096
const ERA5_PHYSICS_BINARY_VARS         = (:udmf, :ddmf, :udrf_rate, :ddrf_rate, :t, :q)

"""
    ERA5PhysicsBinaryHeader

In-memory view of the ERA5 physics BIN header. Parsed from / serialized
to a 4 KB JSON block at the start of each BIN file. Fields:

- `format_version` — int, currently 1. Readers error on mismatch.
- `date` — `Date`, the calendar day the BIN represents (hours 00-23).
- `Nlon, Nlat, Nlev, Nt` — grid shape. `Nt` is always 24 for the
  hourly calendar-day BIN.
- `var_offsets_bytes` — NamedTuple mapping `var name → byte offset`
  into the file (where that variable's payload starts, after the
  header block).
- `var_nelems` — NamedTuple mapping `var name → total element count`.
- `latitude_convention` — `:S_to_N` (AtmosTransport orientation).
- `longitude_range` — `(first, last, step)` in degrees.
- `latitude_range` — `(first, last, step)` in degrees (S to N).
- `provenance` — Dict with source NC paths, timestamp, git sha.
"""
struct ERA5PhysicsBinaryHeader
    format_version        :: Int
    date                  :: Date
    Nlon                  :: Int
    Nlat                  :: Int
    Nlev                  :: Int
    Nt                    :: Int
    var_offsets_bytes     :: NamedTuple{ERA5_PHYSICS_BINARY_VARS, NTuple{6, Int}}
    var_nelems            :: NamedTuple{ERA5_PHYSICS_BINARY_VARS, NTuple{6, Int}}
    latitude_convention   :: Symbol
    longitude_range       :: NTuple{3, Float64}
    latitude_range        :: NTuple{3, Float64}
    provenance            :: Dict{String, Any}
end

# ---------------------------------------------------------------------------
# Helpers (variable-name detection, lat-flip, time-slicing)
# ---------------------------------------------------------------------------

"""
    _era5_detect_var(ds, candidates) -> String | nothing

Find the first variable in `ds` whose lowercase name matches any of
`candidates`. Returns `nothing` if none match. Handles ERA5's
evolving variable naming: legacy `p71.162`, `var71`, `var235009`,
canonical `udmf`, etc.
"""
function _era5_detect_var(ds, candidates)
    for vname in keys(ds)
        lname = lowercase(vname)
        for c in candidates
            if lname == lowercase(c)
                return vname
            end
        end
    end
    return nothing
end

const _CANDIDATES = Dict(
    :udmf => ("udmf", "avg_umf", "mumf", "mu", "var235009", "p235009", "p71.162", "var71"),
    :ddmf => ("ddmf", "avg_dmf", "mdmf", "md", "var235010", "p235010", "p72.162", "var72"),
    :udrf => ("udrf", "avg_udr", "var235011", "p235011", "p214.162", "var214"),
    :ddrf => ("ddrf", "avg_ddr", "var235012", "p235012", "p215.162", "var215"),
    :t    => ("t", "var130", "p130"),
    :q    => ("q", "var133", "p133"),
)

function _era5_requires_var(ds, key::Symbol)
    name = _era5_detect_var(ds, _CANDIDATES[key])
    name === nothing && error(
        "ERA5 physics NC is missing variable `$key` " *
        "(tried candidates: $(_CANDIDATES[key])). " *
        "If this file was downloaded with a different script, " *
        "extend `_CANDIDATES` in " *
        "`src/Preprocessing/era5_physics_binary.jl` with the new name.")
    return name
end

"""
    _era5_latitude_needs_flip(ds) -> Bool

True when the NC's latitude dim runs N→S (ERA5 default). The
converter must flip to S→N on write.
"""
function _era5_latitude_needs_flip(ds)
    haskey(ds, "latitude") || return false
    lats = ds["latitude"][:]
    length(lats) > 1 && lats[1] > lats[end]
end

"""
    _era5_unroll_convection_hours(var::Array{FT, 5}) -> Array{FT, 4}

Convection NC variables are shaped `(Nlon, Nlat, Nlev, step=12, time=2)`
covering 24 hourly values from 07:00 of the base date through 06:00
of the next day. Unroll to a 4D `(Nlon, Nlat, Nlev, 24)` array in
that time order.
"""
function _era5_unroll_convection_hours(var::AbstractArray{FT, 5}) where {FT}
    Nlon, Nlat, Nlev, Nstep, Ntime = size(var)
    Nstep == 12 || throw(ArgumentError("Expected step=12, got $Nstep"))
    Ntime == 2  || throw(ArgumentError("Expected time=2, got $Ntime"))
    out = Array{FT}(undef, Nlon, Nlat, Nlev, 24)
    @inbounds for t in 1:Ntime, s in 1:Nstep
        t_idx = (t - 1) * Nstep + s
        @views out[:, :, :, t_idx] .= var[:, :, :, s, t]
    end
    return out
end

# ---------------------------------------------------------------------------
# Writer: NC → BIN
# ---------------------------------------------------------------------------

"""
    convert_era5_physics_nc_to_bin(nc_dir, bin_dir, date;
                                    force_rewrite=false, verbose=true) -> bin_path

Build one calendar-day ERA5 physics BIN from the archive NCs.

- `nc_dir`: directory containing
  `era5_convection_YYYYMMDD.nc` + `era5_thermo_ml_YYYYMMDD.nc`.
- `bin_dir`: output staging directory; BIN lands at
  `<bin_dir>/<YYYY>/era5_physics_<YYYYMMDD>.bin`.
- `date`: calendar day (00:00-23:00) the BIN represents.
- `force_rewrite`: when `false` and the BIN already exists with a
  valid header, skip (idempotent). When `true`, overwrite.
- `verbose`: log progress (default `true`).

Requires the convection NC for both `date - 1 day` and `date`
(because convection is forecast-based and a calendar-day BIN
needs hours 00-06 from the previous day's file and hours 07-23
from the current day's file). Raises if either is missing.

The thermo NC is calendar-day aligned so only the target-date
file is needed.

Writes the BIN through `AtomicFiles.atomic_publish`, using a unique sibling
staging file and one replacing rename. This is safe to run concurrently with
a reader: it sees either the previous complete file or the new complete file.
"""
function convert_era5_physics_nc_to_bin(nc_dir::AbstractString,
                                         bin_dir::AbstractString,
                                         date::Date;
                                         force_rewrite::Bool = false,
                                         verbose::Bool = true)
    nc_dir   = expand_data_path(nc_dir)
    bin_dir  = expand_data_path(bin_dir)
    year_dir = joinpath(bin_dir, string(year(date)))
    mkpath(year_dir)
    date_str = Dates.format(date, "yyyymmdd")
    bin_path = joinpath(year_dir, "era5_physics_$(date_str).bin")

    if !force_rewrite && isfile(bin_path)
        verbose && @info "ERA5PhysicsBIN: skip $(basename(bin_path)) (exists; force_rewrite=false)"
        return bin_path
    end

    conv_today = joinpath(nc_dir, "era5_convection_$(date_str).nc")
    prev_date  = date - Day(1)
    conv_prev  = joinpath(nc_dir, "era5_convection_$(Dates.format(prev_date, "yyyymmdd")).nc")
    thermo     = joinpath(nc_dir, "era5_thermo_ml_$(date_str).nc")

    isfile(conv_today) || error(
        "ERA5PhysicsBIN: missing convection NC for $date at $conv_today. " *
        "Run `scripts/downloads/download_era5_physics.py --fields convection --start $date --end $date`.")
    isfile(conv_prev) || error(
        "ERA5PhysicsBIN: missing previous-day convection NC for $prev_date " *
        "at $conv_prev (needed for calendar-day hours 00-06 of $date " *
        "because convection fields are forecast-based from base times " *
        "06/18 UTC and the first 7 hours of day $date come from day $prev_date's NC).")
    isfile(thermo) || error(
        "ERA5PhysicsBIN: missing thermo NC for $date at $thermo. " *
        "Run `scripts/downloads/download_era5_physics.py --fields thermodynamics --start $date --end $date`.")

    verbose && @info "ERA5PhysicsBIN: building $(basename(bin_path)) from NCs..."

    conv_today_arrays, conv_prev_arrays, Nlon, Nlat, Nlev, lon_range, lat_range =
        _read_convection_pair(conv_prev, conv_today)
    thermo_arrays = _read_thermo(thermo, Nlon, Nlat, Nlev)

    # Splice: hours 00-06 from prev-day NC (steps 6-12 of time=2),
    # hours 07-23 from today's NC (steps 1-12 of time=1 + steps 1-5
    # of time=2).
    udmf = _splice_calendar_day(conv_prev_arrays.udmf, conv_today_arrays.udmf)
    ddmf = _splice_calendar_day(conv_prev_arrays.ddmf, conv_today_arrays.ddmf)
    udrf = _splice_calendar_day(conv_prev_arrays.udrf, conv_today_arrays.udrf)
    ddrf = _splice_calendar_day(conv_prev_arrays.ddrf, conv_today_arrays.ddrf)

    # Shape: (Nlon, Nlat, Nlev, 24), Float32.
    var_nelems = (
        udmf      = length(udmf),
        ddmf      = length(ddmf),
        udrf_rate = length(udrf),
        ddrf_rate = length(ddrf),
        t         = length(thermo_arrays.t),
        q         = length(thermo_arrays.q),
    )

    header_bytes   = ERA5_PHYSICS_BINARY_HEADER_BYTES
    offset         = header_bytes
    var_offsets    = Int[]
    for key in ERA5_PHYSICS_BINARY_VARS
        push!(var_offsets, offset)
        offset += getproperty(var_nelems, key) * sizeof(Float32)
    end
    var_offsets_nt = NamedTuple{ERA5_PHYSICS_BINARY_VARS}(
        ntuple(i -> var_offsets[i], length(ERA5_PHYSICS_BINARY_VARS)))

    header = ERA5PhysicsBinaryHeader(
        ERA5_PHYSICS_BINARY_VERSION, date, Nlon, Nlat, Nlev, 24,
        var_offsets_nt, var_nelems,
        :S_to_N, lon_range, lat_range,
        Dict{String, Any}(
            "source_convection_today" => conv_today,
            "source_convection_prev"  => conv_prev,
            "source_thermo"           => thermo,
            "nc_to_bin_timestamp"     => string(now(UTC)),
            "generator_version"       => "plan24_commit2_v1",
            "git_sha"                 => _git_sha_or_empty(),
        ),
    )

    atomic_publish(bin_path) do tmp_path
        open(tmp_path, "w") do io
            _write_header!(io, header, header_bytes)
            seek(io, header_bytes)
            _write_payload!(io, udmf)
            _write_payload!(io, ddmf)
            _write_payload!(io, udrf)
            _write_payload!(io, ddrf)
            _write_payload!(io, thermo_arrays.t)
            _write_payload!(io, thermo_arrays.q)
        end
    end

    verbose && @info "ERA5PhysicsBIN: wrote $(basename(bin_path)) " *
                      "($(round(filesize(bin_path) / 1e9; digits = 2)) GB)"
    return bin_path
end

@inline function _splice_calendar_day(prev_day::AbstractArray{FT, 4},
                                       today::AbstractArray{FT, 4}) where {FT}
    # prev_day[:, :, :, 18:24] covers calendar-day 00-06 UTC
    # today[:, :, :, 1:17]  covers calendar-day 07-23 UTC
    # where 1..24 = hours 07:00 day-N through 06:00 day-N+1.
    Nlon, Nlat, Nlev = size(today)[1:3]
    out = Array{FT}(undef, Nlon, Nlat, Nlev, 24)
    @inbounds begin
        # Hours 00-06 UTC of the target day = hours 18-24 of prev's unrolled array.
        # Unrolled index h in 1..24 corresponds to UTC hour (h + 6) mod 24.
        # So UTC 00 = h = 18, UTC 06 = h = 24.
        out[:, :, :,  1:7]  .= prev_day[:, :, :, 18:24]
        # Hours 07-23 UTC of the target day = hours 1-17 of today's unrolled array.
        out[:, :, :, 8:24]  .= today[:, :, :,  1:17]
    end
    return out
end

function _read_convection_pair(conv_prev::AbstractString,
                                conv_today::AbstractString)
    ds_today = NCDataset(conv_today, "r")
    ds_prev  = NCDataset(conv_prev,  "r")
    try
        lon_range = _geom_range(ds_today, "longitude")
        lat_nc    = ds_today["latitude"][:]
        needs_flip = _era5_latitude_needs_flip(ds_today)

        Nlon = ds_today.dim["longitude"]
        Nlat = ds_today.dim["latitude"]
        Nlev = ds_today.dim["hybrid"]

        today_arrs  = _read_convection_vars(ds_today, needs_flip)
        prev_arrs   = _read_convection_vars(ds_prev,  needs_flip)

        # latitude_range is the S-to-N orientation we write to disk.
        lat_sorted = needs_flip ? reverse(lat_nc) : lat_nc
        lat_range  = (Float64(lat_sorted[1]),
                      Float64(lat_sorted[end]),
                      length(lat_sorted) > 1 ? Float64(lat_sorted[2] - lat_sorted[1]) : 0.0)

        return today_arrs, prev_arrs, Nlon, Nlat, Nlev, lon_range, lat_range
    finally
        close(ds_today)
        close(ds_prev)
    end
end

function _read_convection_vars(ds, needs_flip)
    udmf_5d = Float32.(ds[_era5_requires_var(ds, :udmf)][:, :, :, :, :])
    ddmf_5d = Float32.(ds[_era5_requires_var(ds, :ddmf)][:, :, :, :, :])
    udrf_5d = Float32.(ds[_era5_requires_var(ds, :udrf)][:, :, :, :, :])
    ddrf_5d = Float32.(ds[_era5_requires_var(ds, :ddrf)][:, :, :, :, :])

    # Unroll the (step, time) dimensions into a flat 24-hour index.
    udmf = _era5_unroll_convection_hours(udmf_5d)
    ddmf = _era5_unroll_convection_hours(ddmf_5d)
    udrf = _era5_unroll_convection_hours(udrf_5d)
    ddrf = _era5_unroll_convection_hours(ddrf_5d)

    if needs_flip
        udmf = udmf[:, end:-1:1, :, :]
        ddmf = ddmf[:, end:-1:1, :, :]
        udrf = udrf[:, end:-1:1, :, :]
        ddrf = ddrf[:, end:-1:1, :, :]
    end
    return (udmf = udmf, ddmf = ddmf, udrf = udrf, ddrf = ddrf)
end

function _read_thermo(thermo::AbstractString, Nlon, Nlat, Nlev)
    ds = NCDataset(thermo, "r")
    try
        Nt = ds.dim["time"]
        Nt == 24 || error("ERA5PhysicsBIN: expected 24 hours in thermo NC, got $Nt")
        ds.dim["longitude"] == Nlon || error("ERA5PhysicsBIN: thermo Nlon mismatch")
        ds.dim["latitude"]  == Nlat || error("ERA5PhysicsBIN: thermo Nlat mismatch")
        ds.dim["hybrid"]    == Nlev || error("ERA5PhysicsBIN: thermo Nlev mismatch")

        needs_flip = _era5_latitude_needs_flip(ds)
        t = Float32.(ds[_era5_requires_var(ds, :t)][:, :, :, :])
        q = Float32.(ds[_era5_requires_var(ds, :q)][:, :, :, :])
        if needs_flip
            t = t[:, end:-1:1, :, :]
            q = q[:, end:-1:1, :, :]
        end
        return (t = t, q = q)
    finally
        close(ds)
    end
end

function _geom_range(ds, dim::AbstractString)
    vals = ds[dim][:]
    length(vals) >= 2 ||
        return (Float64(vals[1]), Float64(vals[1]), 0.0)
    return (Float64(vals[1]), Float64(vals[end]),
            Float64(vals[2] - vals[1]))
end

function _write_header!(io::IO, h::ERA5PhysicsBinaryHeader, header_bytes::Int)
    d = Dict{String, Any}(
        "magic"               => ERA5_PHYSICS_BINARY_MAGIC,
        "format_version"      => h.format_version,
        "date"                => string(h.date),
        "Nlon"                => h.Nlon,
        "Nlat"                => h.Nlat,
        "Nlev"                => h.Nlev,
        "Nt"                  => h.Nt,
        "var_offsets_bytes"   => Dict(string(k) => v for (k, v) in pairs(h.var_offsets_bytes)),
        "var_nelems"          => Dict(string(k) => v for (k, v) in pairs(h.var_nelems)),
        "latitude_convention" => string(h.latitude_convention),
        "longitude_range"     => collect(h.longitude_range),
        "latitude_range"      => collect(h.latitude_range),
        "provenance"          => h.provenance,
    )
    json = JSON3.write(d)
    nb = ncodeunits(json)
    nb < header_bytes ||
        error("ERA5PhysicsBIN: header JSON ($nb B) exceeds budget ($header_bytes B)")
    write(io, json)
    write(io, zeros(UInt8, header_bytes - nb))
    return nothing
end

function _write_payload!(io::IO, arr::AbstractArray{Float32})
    expected = sizeof(arr)
    written  = write(io, arr)
    written == expected ||
        error("ERA5PhysicsBIN: short write ($written/$expected bytes). " *
              "Disk full? Check `df -h` on the staging directory. " *
              "Partial BIN is discarded by the outer atomic publication.")
    return nothing
end

function _git_sha_or_empty()
    try
        read(pipeline(`git -C $(@__DIR__) rev-parse --short HEAD`; stderr=devnull), String) |> strip
    catch
        ""
    end
end

# ---------------------------------------------------------------------------
# Reader: mmap-friendly BIN access
# ---------------------------------------------------------------------------

"""
    ERA5PhysicsBinaryReader

Mmap view of one ERA5 physics BIN file. Per-variable getters return
reshaped views with zero allocation on subsequent calls.

# Fields

- `path` — absolute BIN path.
- `io` — open `IOStream` (keep alive for mmap lifetime).
- `header` — parsed `ERA5PhysicsBinaryHeader`.
- `mmap` — single flat `Vector{Float32}` over the entire payload.
  Individual variable views reshape into this.

# Usage

```julia
reader = open_era5_physics_binary(bin_dir, Date(2021, 12, 1))
try
    udmf = get_era5_physics_field(reader, :udmf)    # 4D (Nlon, Nlat, Nlev, 24)
    slab = @view udmf[:, :, :, 5]                    # one hour
    # ... use slab ...
finally
    close_era5_physics_binary(reader)
end
```
"""
mutable struct ERA5PhysicsBinaryReader
    path   :: String
    io     :: IOStream
    header :: ERA5PhysicsBinaryHeader
    mmap   :: Vector{Float32}
end

"""
    open_era5_physics_binary(bin_dir, date) -> ERA5PhysicsBinaryReader

Open the BIN for `date` under `bin_dir` (with YYYY subdir), mmap
the payload, parse the header. Caller must `close_era5_physics_binary`
when done (or wrap in a try/finally).
"""
function open_era5_physics_binary(bin_dir::AbstractString, date::Date)
    year_dir = joinpath(expand_data_path(bin_dir), string(year(date)))
    date_str = Dates.format(date, "yyyymmdd")
    bin_path = joinpath(year_dir, "era5_physics_$(date_str).bin")
    isfile(bin_path) || error(
        "ERA5PhysicsBIN: missing $bin_path. Run " *
        "`julia --project=. scripts/preprocessing/convert_era5_physics_nc_to_bin.jl " *
        "--nc-dir <NC dir> --bin-dir $bin_dir --start $date --end $date`.")

    io = open(bin_path, "r")
    header = _read_header(io)
    # Size the mmap at payload end.
    last_var_end = maximum(getproperty(header.var_offsets_bytes, k) +
                            getproperty(header.var_nelems, k) * sizeof(Float32)
                            for k in ERA5_PHYSICS_BINARY_VARS)
    nfloats = (last_var_end - ERA5_PHYSICS_BINARY_HEADER_BYTES) ÷ sizeof(Float32)
    mmap_data = Mmap.mmap(io, Vector{Float32}, nfloats,
                          ERA5_PHYSICS_BINARY_HEADER_BYTES)
    return ERA5PhysicsBinaryReader(bin_path, io, header, mmap_data)
end

"""
    close_era5_physics_binary(reader) -> nothing

Release the mmap and close the underlying `IOStream`.
"""
function close_era5_physics_binary(reader::ERA5PhysicsBinaryReader)
    reader.mmap = Float32[]
    close(reader.io)
    return nothing
end

function _read_header(io::IO)
    seek(io, 0)
    raw = read(io, ERA5_PHYSICS_BINARY_HEADER_BYTES)
    # JSON3 stops at the first '\0' in our padded block.
    nul_idx = findfirst(==(0x00), raw)
    json_bytes = nul_idx === nothing ? raw : view(raw, 1:(nul_idx - 1))
    d = JSON3.read(String(json_bytes))
    get(d, :magic, "") == ERA5_PHYSICS_BINARY_MAGIC ||
        error("ERA5PhysicsBIN: magic mismatch (file is not an ERA5 physics BIN).")
    v = Int(d.format_version)
    v == ERA5_PHYSICS_BINARY_VERSION ||
        error("ERA5PhysicsBIN: format_version $v ≠ expected $ERA5_PHYSICS_BINARY_VERSION. " *
              "Regenerate the BIN with the current converter.")
    var_offsets = NamedTuple{ERA5_PHYSICS_BINARY_VARS}(
        ntuple(i -> Int(getproperty(d.var_offsets_bytes, ERA5_PHYSICS_BINARY_VARS[i])),
               length(ERA5_PHYSICS_BINARY_VARS)))
    var_nelems = NamedTuple{ERA5_PHYSICS_BINARY_VARS}(
        ntuple(i -> Int(getproperty(d.var_nelems, ERA5_PHYSICS_BINARY_VARS[i])),
               length(ERA5_PHYSICS_BINARY_VARS)))
    return ERA5PhysicsBinaryHeader(
        v, Date(String(d.date)),
        Int(d.Nlon), Int(d.Nlat), Int(d.Nlev), Int(d.Nt),
        var_offsets, var_nelems,
        Symbol(String(d.latitude_convention)),
        Tuple(Float64.(d.longitude_range)),
        Tuple(Float64.(d.latitude_range)),
        Dict{String, Any}(String(k) => v for (k, v) in pairs(d.provenance)),
    )
end

"""
    get_era5_physics_field(reader, var::Symbol) -> Array view

Return a zero-allocation reshaped view into the mmap for
`var ∈ (:udmf, :ddmf, :udrf_rate, :ddrf_rate, :t, :q)`. Shape
`(Nlon, Nlat, Nlev, Nt)`.
"""
function get_era5_physics_field(reader::ERA5PhysicsBinaryReader, var::Symbol)
    var in ERA5_PHYSICS_BINARY_VARS ||
        throw(ArgumentError("Unknown var $var; expected one of $ERA5_PHYSICS_BINARY_VARS"))
    h = reader.header
    nelems = getproperty(h.var_nelems, var)
    off_bytes = getproperty(h.var_offsets_bytes, var)
    # Offset into the mmap'd vector (mmap starts at header_bytes).
    off_floats = (off_bytes - ERA5_PHYSICS_BINARY_HEADER_BYTES) ÷ sizeof(Float32)
    flat = view(reader.mmap, (off_floats + 1):(off_floats + nelems))
    return reshape(flat, h.Nlon, h.Nlat, h.Nlev, h.Nt)
end
