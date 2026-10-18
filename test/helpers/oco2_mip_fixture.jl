# Test helpers for OCO-2 level retrievals: a synthetic OCO-2 v11 MIP
# 10-second file (the layout of OCO2_b11.2_10sec_GOOD_r2.nc4, levels from
# space to the surface) for the converter, the reader of retrieval fields and
# the forward sampler's averaging-kernel columns, and an independent
# construction of the hat projection (test/core/test_retrieval_hat_projection.jl,
# test_cs_4dvar_oco2_mip.jl). Include after `import AtmosTransport`.

using NCDatasets: NCDataset, defDim, defVar
using Dates: DateTime, datetime2unix, year, month, day, hour, minute, second, millisecond

# The OCO-2 retrieval sigma levels, from space to the surface.
const OCO2_SIGMA = vcat(1e-4, collect(1:19) ./ 19)

"""
    oco2_pressure_weights(psurf_pa; q = nothing) -> Vector{Float64}

The pressure weights of the OCO-2 levels (from space to the surface) at
surface pressure `psurf_pa`: the integrals of the levels' hat functions
(trapezoid rule, half-thickness end levels), weighted by `1 - q` per level
(a dry-air correction) and normalized to 1.
"""
function oco2_pressure_weights(psurf_pa; q = nothing)
    p = OCO2_SIGMA .* psurf_pa
    n = length(p)
    w = [((j == n ? p[n] : (p[j] + p[j + 1]) / 2) - (j == 1 ? p[1] : (p[j - 1] + p[j]) / 2)) for j in 1:n]
    q === nothing || (w .*= (1 .- q))
    return w ./ sum(w)
end

"""
    write_oco2_mip_fixture(path, records; short_name = "OCO2_v11.2_XCO2_10sec_avg") -> path

`records` are named tuples with `id`, `time::DateTime`, `lat`, `lon`, `xco2`,
`unc`, `model_error` (ppm), `psurf` (hPa), `altitude`, `data_type`,
`assimilate_flag`, and the 20-level `ak`, `prior` (ppm) and `pw`, from space to
the surface. Fields are stored as the MIP files store them (Float32, Int8).
"""
function write_oco2_mip_fixture(path, records; short_name = "OCO2_v11.2_XCO2_10sec_avg")
    n = length(records)
    NCDataset(path, "c") do ds
        defDim(ds, "sounding_id", n)
        defDim(ds, "levels", 20)
        defDim(ds, "epoch_dimension", 7)
        ds.attrib["ShortName"] = short_name
        defVar(ds, "sounding_id", Int64, ("sounding_id",))[:] = [r.id for r in records]
        defVar(ds, "date", Int16, ("epoch_dimension", "sounding_id"))[:, :] =
            Int16[c == 1 ? year(r.time) : c == 2 ? month(r.time) : c == 3 ? day(r.time) : c == 4 ? hour(r.time) :
                  c == 5 ? minute(r.time) : c == 6 ? second(r.time) : millisecond(r.time) for c in 1:7, r in records]
        defVar(ds, "time", Float64, ("sounding_id",);
               attrib = Dict("units" => "seconds since 1970-01-01 00:00:00"))[:] = [datetime2unix(r.time) for r in records]
        defVar(ds, "latitude", Float32, ("sounding_id",); attrib = Dict("units" => "degrees_north"))[:] =
            Float32[r.lat for r in records]
        defVar(ds, "longitude", Float32, ("sounding_id",); attrib = Dict("units" => "degrees_east"))[:] =
            Float32[r.lon for r in records]
        for (name, field, units) in (("xco2_2019_scale", :xco2, "ppm"), ("xco2_uncertainty", :unc, "ppm"),
                                     ("model_error", :model_error, "ppm"), ("psurf", :psurf, "hPa"),
                                     ("altitude", :altitude, "meters"))
            defVar(ds, name, Float32, ("sounding_id",); attrib = Dict("units" => units))[:] =
                Float32[getfield(r, field) for r in records]
        end
        defVar(ds, "xco2_apriori", Float32, ("sounding_id",); attrib = Dict("units" => "ppm"))[:] =
            Float32[sum(Float32.(r.pw) .* Float32.(r.prior)) for r in records]
        defVar(ds, "data_type", Int8, ("sounding_id",))[:] = Int8[r.data_type for r in records]
        defVar(ds, "assimilate_flag", Int8, ("sounding_id",))[:] = Int8[r.assimilate_flag for r in records]
        defVar(ds, "sigma_levels", Float32, ("levels",); attrib = Dict("units" => "none"))[:] = Float32.(OCO2_SIGMA)
        for (name, field, units) in (("co2_profile_apriori", :prior, "ppm"), ("xco2_averaging_kernel", :ak, "none"),
                                     ("pressure_weight", :pw, "none"))
            defVar(ds, name, Float32, ("levels", "sounding_id"); attrib = Dict("units" => units))[:, :] =
                Float32[getfield(r, field)[l] for l in 1:20, r in records]
        end
    end
    return path
end

# A plausible OCO-2 retrieval at surface pressure `psurf` (hPa), from space to
# the surface: a kernel rising from the top, a prior falling into the
# stratosphere, dry-air-corrected weights.
function oco2_fixture_retrieval(psurf; tilt = 0.0)
    ak = [0.32 + 0.7 * (1 - exp(-6σ)) - 0.05σ + tilt * (σ - 0.5) for σ in OCO2_SIGMA]
    prior = [411.2 - 14.0 * exp(-OCO2_SIGMA[l] / 0.08) + 0.4 * OCO2_SIGMA[l] for l in 1:20]
    q = [2e-6 + 1.2e-2 * σ^3 for σ in OCO2_SIGMA]
    return (; ak, prior, pw = oco2_pressure_weights(100 * psurf; q))
end

# An independent construction of the hat projection: stretch, then for every
# stretched layer the exact integral of each hat by the trapezoid rule on the
# union of the layer edges and the levels (the hats are linear between them).
function hat_projection_reference(p_half, air, levels)
    nz = length(p_half) - 1
    n = length(levels)
    s = AtmosTransport.Output.stretch_model_interfaces!(zeros(nz + 1), p_half, levels[1], levels[end])
    nodes = vcat(s[end], levels[2:(n - 1)], s[1])                  # surface first
    φ(j, p) = begin
        hi = j == 1 ? Inf : nodes[j - 1]
        lo = j == n ? -Inf : nodes[j + 1]
        p >= nodes[j] ? (j == 1 ? 1.0 : (hi - p) / (hi - nodes[j])) :
                        (j == n ? 1.0 : (p - lo) / (nodes[j] - lo))
    end
    integral = zeros(n, nz)
    for k in 1:nz
        pts = sort!(unique!(vcat(s[k], s[k + 1], filter(p -> s[k] < p < s[k + 1], nodes))))
        for j in 1:n
            integral[j, k] = sum((pts[t + 1] - pts[t]) * (max(φ(j, pts[t]), 0) + max(φ(j, pts[t + 1]), 0)) / 2
                                 for t in 1:(length(pts) - 1))
        end
    end
    F = integral ./ reshape(diff(s), 1, :)
    under = F * air
    return (; P = (F .* reshape(air, 1, :)) ./ under, hat = under ./ sum(under), s)
end
