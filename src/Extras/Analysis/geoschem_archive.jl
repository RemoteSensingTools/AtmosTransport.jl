# ===========================================================================
# The output archive of the GEOS-Chem (GCHP) CATRINE reference run.
#
# The run writes one file of instantaneous fields per output time,
# `GEOSChem.CATRINE_inst.YYYYMMDD_HHMMz.nc4` (every 3 hours), either flat in
# one directory or in `YYYY/MM/` subdirectories. Each file holds one time on
# GEOS-Chem's cubed sphere, in NCDatasets order (Xdim, Ydim, nf[, lev], time)
# with levels surface first. The diagnostics use:
#
#   Met_AD           dry-air mass of each cell [kg]
#   Met_AREAM2       cell area [m²]
#   SpeciesConcVV_*  dry mole fraction [mol mol⁻¹]
#   Emis*            instantaneous surface emission flux [kg species m⁻² s⁻¹]
#
# netcdf-c is not thread-safe, so every NetCDF call runs under the package's
# NetCDF lock. The file itself is read into memory first, outside the lock:
# the archive sits on network storage, where one sequential read of a whole
# file is several times faster than HDF5's chunk-by-chunk reads, and threads
# can read files concurrently while only the in-memory decoding is serialised.
# ===========================================================================

"""
    GEOSChemTracer(name, concentration, emission, molar_mass; half_life = Inf)

One tracer of a GEOS-Chem run and the output variables that describe it:

- `name`: the diagnostic's label for the tracer (a `Symbol`, for example
  `:co2_natural`);
- `concentration`: the dry mole fraction variable [mol mol⁻¹], for example
  `"SpeciesConcVV_CO2"`;
- `emission`: the surface emission flux variable [kg species m⁻² s⁻¹], for
  example `"EmisCO2_Total"`;
- `molar_mass`: GEOS-Chem's molar mass of the species [kg mol⁻¹], from
  `Parameters.GEOSCHEM_SPECIES_MOLAR_MASS`;
- `half_life`: radioactive half-life [s]; `Inf` for a stable tracer.
"""
struct GEOSChemTracer
    name          :: Symbol
    concentration :: String
    emission      :: String
    molar_mass    :: Float64
    half_life     :: Float64
end

GEOSChemTracer(name::Symbol, concentration::AbstractString, emission::AbstractString,
               molar_mass::Real; half_life::Real = Inf) =
    GEOSChemTracer(name, concentration, emission, molar_mass, half_life)

"""
    CATRINE_GEOSCHEM_TRACERS

The four tracers of the GEOS-Chem CATRINE reference run, labelled as in the
AtmosTransport CATRINE runs: `co2_natural` (`SpeciesConcVV_CO2`,
`EmisCO2_Total`), `co2_fossil` (`SpeciesConcVV_FossilCO2`,
`Emis_FossilCO2_Total`), `sf6` (`SpeciesConcVV_SF6`, `EmisSF6`) and `rn222`
(`SpeciesConcVV_Rn222`, `EmisRn_Soil`, half-life `RN222_HALF_LIFE`). Molar
masses are GEOS-Chem's (`GEOSCHEM_SPECIES_MOLAR_MASS`; the CO₂ value for both
CO₂ tracers).
"""
const CATRINE_GEOSCHEM_TRACERS = (
    GEOSChemTracer(:co2_natural, "SpeciesConcVV_CO2", "EmisCO2_Total", GEOSCHEM_SPECIES_MOLAR_MASS.co2),
    GEOSChemTracer(:co2_fossil, "SpeciesConcVV_FossilCO2", "Emis_FossilCO2_Total",
                   GEOSCHEM_SPECIES_MOLAR_MASS.co2),
    GEOSChemTracer(:sf6, "SpeciesConcVV_SF6", "EmisSF6", GEOSCHEM_SPECIES_MOLAR_MASS.sf6),
    GEOSChemTracer(:rn222, "SpeciesConcVV_Rn222", "EmisRn_Soil", GEOSCHEM_SPECIES_MOLAR_MASS.rn222;
                   half_life = RN222_HALF_LIFE),
)

"""
    GEOSChemCATRINEArchive(dir; tracers = CATRINE_GEOSCHEM_TRACERS, interval = Hour(3))

The output of the GEOS-Chem CATRINE reference run under `dir`: one file
`GEOSChem.CATRINE_inst.YYYYMMDD_HHMMz.nc4` per output time, every `interval`,
flat in `dir` or in `dir/YYYY/MM/`. `tracers` is a tuple of
[`GEOSChemTracer`](@ref)s.

The archive records the files it could not read: [`with_gc_file`](@ref) (and
[`global_totals`](@ref), which uses it) returns `nothing` for a missing,
truncated or otherwise unreadable file or one that lacks a variable, and
[`unreadable_files`](@ref) lists them with the reason. Reads are safe from
several threads.
"""
struct GEOSChemCATRINEArchive{T <: Tuple{Vararg{GEOSChemTracer}}}
    dir        :: String
    tracers    :: T
    interval   :: Millisecond
    unreadable :: Dict{DateTime, String}
    lock       :: ReentrantLock
end

function GEOSChemCATRINEArchive(dir::AbstractString; tracers::Tuple{Vararg{GEOSChemTracer}} = CATRINE_GEOSCHEM_TRACERS,
                                interval::Dates.Period = Hour(3))
    step = Millisecond(interval)
    step > Millisecond(0) || throw(ArgumentError("GEOSChemCATRINEArchive: interval must be positive; got $(interval)"))
    isempty(tracers) && throw(ArgumentError("GEOSChemCATRINEArchive: no tracers given"))
    names = [tr.name for tr in tracers]
    allunique(names) || throw(ArgumentError("GEOSChemCATRINEArchive: tracer names repeat: $(names)"))
    for tr in tracers
        (isfinite(tr.molar_mass) && tr.molar_mass > 0) ||
            throw(ArgumentError("GEOSChemCATRINEArchive: $(tr.name) molar mass must be finite and positive; got $(tr.molar_mass)"))
        tr.half_life > 0 ||
            throw(ArgumentError("GEOSChemCATRINEArchive: $(tr.name) half-life must be positive (Inf for a stable tracer); got $(tr.half_life)"))
    end
    return GEOSChemCATRINEArchive(String(dir), tracers, step, Dict{DateTime, String}(), ReentrantLock())
end

Base.show(io::IO, a::GEOSChemCATRINEArchive) =
    print(io, "GEOSChemCATRINEArchive(", repr(a.dir), "; ", length(a.tracers), " tracers, every ",
          Dates.canonicalize(Dates.CompoundPeriod(a.interval)), ")")

"""
    gc_path(archive, t::DateTime) -> String

Path of the archive's file for time `t`: `dir/YYYY/MM/<name>` if that file
exists, otherwise `dir/<name>` (which need not exist), with
`<name> = GEOSChem.CATRINE_inst.YYYYMMDD_HHMMz.nc4`.
"""
function gc_path(a::GEOSChemCATRINEArchive, t::DateTime)
    name = "GEOSChem.CATRINE_inst." * Dates.format(t, dateformat"yyyymmdd_HHMM") * "z.nc4"
    nested = joinpath(a.dir, Dates.format(t, dateformat"yyyy"), Dates.format(t, dateformat"mm"), name)
    return isfile(nested) ? nested : joinpath(a.dir, name)
end

"""
    gc_times(archive, start::DateTime, stop::DateTime) -> Vector{DateTime}

The archive's output times `start, start + interval, …` up to and including
`stop` (when it falls on the grid).
"""
function gc_times(a::GEOSChemCATRINEArchive, start::DateTime, stop::DateTime)
    stop >= start || throw(ArgumentError("gc_times: stop $(stop) precedes start $(start)"))
    return collect(start:a.interval:stop)
end

# A file that lacks what a reader needs (for example a variable without a time
# record): raised by the readers of this file, recorded as unreadable.
struct GEOSChemFileError <: Exception
    msg :: String
end
Base.showerror(io::IO, err::GEOSChemFileError) = print(io, err.msg)

# One-line reasons for the unreadable-file record.
_unreadable_reason(err::NetCDFError) = err.msg
_unreadable_reason(err::SystemError) = "cannot read the file: " * Libc.strerror(err.errnum)
_unreadable_reason(err) = sprint(showerror, err)

function _unreadable!(a, t, err)
    reason = _unreadable_reason(err)
    lock(() -> (a.unreadable[t] = reason), a.lock)
    return nothing
end

"""
    with_gc_file(f, archive, t::DateTime)

Open the archive's file for time `t` ([`gc_path`](@ref)) and return `f(ds)`,
where `ds` is the `NCDataset`; return `nothing` if the file cannot be read.

The file is read into memory first (without the lock, so threads read files
concurrently), then opened from memory and passed to `f` under the package's
NetCDF lock (`Output.with_netcdf_lock`; netcdf-c is not thread-safe). `f`
should only copy the arrays it needs out of `ds`; computing on them after
`with_gc_file` returns keeps the lock short. Memory use is the file size plus
whatever `f` returns.

The file is unreadable when it is missing or cannot be read (`SystemError`,
`EOFError`), when it is not a NetCDF/HDF5 file or is truncated (a
`NetCDFError` on opening), and when `f` hits a `NetCDFError` (a missing
variable, a failed read) or a variable without a time record: then the time
and the reason are recorded in the archive (see [`unreadable_files`](@ref))
and `nothing` is returned. Any other error propagates, including a `KeyError`
or `BoundsError` raised by `f` itself.
"""
function with_gc_file(f, a::GEOSChemCATRINEArchive, t::DateTime)
    path = gc_path(a, t)
    bytes = try
        read(path)
    catch err
        err isa Union{SystemError, EOFError} || rethrow()
        return _unreadable!(a, t, err)
    end
    return with_netcdf_lock() do
        ds = try
            NCDataset(path, "r"; memory = bytes)
        catch err
            err isa NetCDFError || rethrow()
            return _unreadable!(a, t, err)
        end
        try
            return f(ds)
        catch err
            err isa Union{NetCDFError, GEOSChemFileError} || rethrow()
            return _unreadable!(a, t, err)
        finally
            close(ds)
        end
    end
end

"""
    unreadable_files(archive) -> Vector{Pair{DateTime, String}}

The times whose files [`with_gc_file`](@ref) could not read so far, in time
order, each with the reason (the path is [`gc_path`](@ref)).
"""
unreadable_files(a::GEOSChemCATRINEArchive) =
    lock(() -> sort!(collect(a.unreadable); by = first), a.lock)

# The first time record of `name`, with `_FillValue` / `missing_value` as zero
# (as the masked arrays filled with 0 of the Python diagnostics).
function _gc_field(ds, name::AbstractString)
    var = ds[name]
    dims = dimnames(var)
    (!isempty(dims) && last(dims) == "time") ||
        throw(ArgumentError("GEOS-Chem variable `$(name)` has dimensions $(dims); expected `time` last"))
    size(var, length(dims)) >= 1 || throw(GEOSChemFileError("`$(name)` has no time record"))
    raw = var[ntuple(_ -> Colon(), length(dims) - 1)..., 1]
    return nomissing(raw, zero(nonmissingtype(eltype(raw))))
end

# Σ x and Σ x·y in Float64, in linear-index order (pairwise).
_sum64(x) = sum(i -> Float64(x[i]), eachindex(x))
_dot64(x, y) = sum(i -> Float64(x[i]) * Float64(y[i]), eachindex(x, y))

_totals_type(::GEOSChemCATRINEArchive{T}) where {T} =
    @NamedTuple{dry_air::Float64, burden::NTuple{fieldcount(T), Float64},
                emission_rate::NTuple{fieldcount(T), Float64}}

"""
    global_totals(archive, t::DateTime) -> (; dry_air, burden, emission_rate) or nothing

Global sums of one output file of a GEOS-Chem archive, in GEOS-Chem's
dry-air-equivalent storage units (kg of dry air carrying the tracer's mole
fraction, the units of AtmosTransport's `<tracer>_total_mass`):

- `dry_air` ``= \\sum \\mathrm{Met\\_AD}`` [kg];
- `burden[i]` ``= \\sum \\chi_i \\, \\mathrm{Met\\_AD}`` [kg] for the dry mole
  fraction ``\\chi_i`` of tracer `i` of `archive.tracers`;
- `emission_rate[i]` ``= \\left(\\sum E_i \\, \\mathrm{Met\\_AREAM2}\\right) M_\\mathrm{air} / M_i``
  [kg s⁻¹], the emission flux ``E_i`` [kg species m⁻² s⁻¹] integrated over the
  sphere and converted with GEOS-Chem's molar masses
  (``M_\\mathrm{air}`` = `DRY_AIR_MOLAR_MASS`, GEOS-Chem's `AIRMW`; ``M_i`` the
  tracer's `molar_mass`). The emission fields are instantaneous: the rate at
  time `t`.

Products and sums are taken in Float64 over the Float32 fields; fill values
count as zero. Returns `nothing` when the file is unreadable (recorded in the
archive, see [`with_gc_file`](@ref)).
"""
function global_totals(a::GEOSChemCATRINEArchive, t::DateTime)
    fields = with_gc_file(a, t) do ds
        (; air = _gc_field(ds, "Met_AD"), area = _gc_field(ds, "Met_AREAM2"),
           conc = map(tr -> _gc_field(ds, tr.concentration), a.tracers),
           emis = map(tr -> _gc_field(ds, tr.emission), a.tracers))
    end
    fields === nothing && return nothing
    (; air, area, conc, emis) = fields
    burden = map(q -> _dot64(q, air), conc)
    emission_rate = map((e, tr) -> _dot64(e, area) * DRY_AIR_MOLAR_MASS / tr.molar_mass, emis, a.tracers)
    return _totals_type(a)((_sum64(air), burden, emission_rate))
end

"""
    global_totals(archive, times::AbstractVector{DateTime}) -> Vector

[`global_totals`](@ref) of each time, on all Julia threads (start Julia with
`-t N`); an element is `nothing` where the file is unreadable. Each thread
holds one file in memory while it is decoded.
"""
function global_totals(a::GEOSChemCATRINEArchive, times::AbstractVector{DateTime})
    ts = collect(DateTime, times)
    out = Vector{Union{Nothing, _totals_type(a)}}(nothing, length(ts))
    Threads.@threads :dynamic for k in eachindex(ts)
        out[k] = global_totals(a, ts[k])
    end
    return out
end
