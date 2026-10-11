# ===========================================================================
# legacy_budget_emission.jl: harness-only expected emissions for the budget
# closure parity of `check_trendy_npp_rh_run.py` (design memo 2026-10-10, §4.3)
#
# `Analysis.budget_closure` takes the expected storage change of every tracer
# from an `AbstractExpectedEmission`. The library default integrates the run's
# own sources. The two methods here exist only to compare the Julia port with
# the Python reports and are deleted with the harness emulators in step P11:
#
# - `LegacyDailyFluxFileEmission(dry_air_molar_mass)`: the Python arithmetic.
#   For a `cs_native` tracer with daily slices stamped at 00 UTC, the change
#   over [t0, t1] is
#       E = Σ_d [Σ_c F_{c,d} A_c^file] · 86400 · M_air^legacy / M_tracer ,
#   with the flux file's `cell_area` A^file, the config's `molar_mass_kg_mol`
#   and the retired dry-air molar mass 0.02896546 (the runtime constant
#   `DRY_AIR_MOLAR_MASS` is 0.0289644 since d552d3e0). Days are summed in
#   blocks of 64 in Float64, left to right as `x*86400*M_air/M`, the order of
#   the Python script.
# - `ScaledEmission(inner, factor)`: `factor` times the change of `inner`, for
#   example the runtime sources rescaled by 0.02896546 / DRY_AIR_MOLAR_MASS to
#   the storage scale a run made before d552d3e0 applied.
# - `TabulatedEmission(values)`: fixed changes per tracer (kg), to reuse the
#   changes of an earlier report without rebuilding the sources.
#
# Usage: `include` this file, then pass an instance as
# `budget_closure(run; emission = ...)`.
# ===========================================================================
module LegacyBudgetEmission

using Dates: DateTime, Millisecond
using NCDatasets: NCDataset
using AtmosTransport: TransportTracerSpec
using AtmosTransport.Analysis: Analysis, AbstractExpectedEmission, RunContext

export LegacyDailyFluxFileEmission, ScaledEmission, TabulatedEmission

"Retired dry-air molar mass of `check_trendy_npp_rh_run.py` and `prepare_trendy_s3_npp_rh.py` (kg mol⁻¹)."
const LEGACY_DRY_AIR_MOLAR_MASS = 0.02896546

struct LegacyDailyFluxFileEmission <: AbstractExpectedEmission
    dry_air_molar_mass :: Float64
end
LegacyDailyFluxFileEmission() = LegacyDailyFluxFileEmission(LEGACY_DRY_AIR_MOLAR_MASS)

const _SECONDS_PER_DAY = 86400.0
const _BLOCK_DAYS = 64

function Analysis.expected_storage_change(m::LegacyDailyFluxFileEmission, ctx::RunContext,
                                          spec::TransportTracerSpec, t0::Real, t1::Real)
    cfg = spec.surface_flux_cfg
    isempty(cfg) && return 0.0
    file, variable = String(cfg["file"]), String(cfg["variable"])
    molar_mass = Float64(cfg["molar_mass_kg_mol"])
    days = (Float64(t1) - Float64(t0)) / _SECONDS_PER_DAY
    isinteger(days) || throw(ArgumentError("legacy budget: [$(t0), $(t1)] s is not a whole number of days"))
    NCDataset(file, "r") do ds
        stamps = collect(DateTime, ds["time"][:])
        first_stamp = ctx.reference_time + Millisecond(round(Int64, 1000 * Float64(t0)))
        first_day = findfirst(==(first_stamp), stamps)
        first_day === nothing && throw(ArgumentError("legacy budget: $(file) has no slice at $(first_stamp)"))
        last_day = first_day + Int(days) - 1
        last_day <= length(stamps) || throw(ArgumentError("legacy budget: $(file) ends before day $(last_day)"))
        all(k -> stamps[k + 1] - stamps[k] == Millisecond(86_400_000), first_day:last_day-1) ||
            throw(ArgumentError("legacy budget: $(file) is not daily between $(first_stamp) and day $(last_day)"))
        area = Float64.(ds["cell_area"][:, :, :])
        expected = 0.0
        for i in first_day:_BLOCK_DAYS:last_day
            block = i:min(i + _BLOCK_DAYS - 1, last_day)
            flux = Float64.(ds[variable][:, :, :, block])
            expected += sum(flux .* area) * _SECONDS_PER_DAY * m.dry_air_molar_mass / molar_mass
        end
        return expected
    end
end

struct ScaledEmission{E <: AbstractExpectedEmission} <: AbstractExpectedEmission
    inner  :: E
    factor :: Float64
end

Analysis.expected_storage_change(m::ScaledEmission, ctx::RunContext, spec::TransportTracerSpec,
                                 t0::Real, t1::Real) =
    m.factor * Analysis.expected_storage_change(m.inner, ctx, spec, t0, t1)

struct TabulatedEmission <: AbstractExpectedEmission
    values :: Dict{Symbol, Float64}
end

Analysis.expected_storage_change(m::TabulatedEmission, ::RunContext, spec::TransportTracerSpec,
                                 ::Real, ::Real) = m.values[spec.name]

Base.show(io::IO, m::LegacyDailyFluxFileEmission) =
    print(io, "LegacyDailyFluxFileEmission(dry_air_molar_mass = ", m.dry_air_molar_mass, ")")
Base.show(io::IO, m::ScaledEmission) = print(io, "ScaledEmission(", m.inner, ", ", m.factor, ")")
Base.show(io::IO, m::TabulatedEmission) = print(io, "TabulatedEmission(", length(m.values), " tracers)")

end # module LegacyBudgetEmission
