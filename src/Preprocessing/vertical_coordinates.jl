# HybridSigmaPressure, n_levels, pressure_at_interface, level_thickness
# are imported from ..Grids via the parent Preprocessing module.

"""
    merge_thin_levels(vc; min_thickness_Pa=1000, p_surface=STANDARD_PRESSURE)

Merge adjacent vertical levels that are thinner than `min_thickness_Pa` at
the reference surface pressure. Returns `(merged_vc, merge_map)` where
`merge_map[k]` maps native level `k` to the merged level index.
"""
function merge_thin_levels(vc::HybridSigmaPressure{FT};
                           min_thickness_Pa::Real = FT(1000),
                           p_surface::Real = FT(STANDARD_PRESSURE)) where FT
    Nz = n_levels(vc)
    ps = FT(p_surface)
    min_dp = FT(min_thickness_Pa)
    dp = [level_thickness(vc, k, ps) for k in 1:Nz]

    top_ifaces = Int[1]
    acc = zero(FT)
    for k in 1:Nz
        acc += dp[k]
        if acc >= min_dp
            push!(top_ifaces, k + 1)
            acc = zero(FT)
        end
    end

    bot_ifaces = Int[Nz + 1]
    acc = zero(FT)
    for k in Nz:-1:1
        acc += dp[k]
        if acc >= min_dp
            pushfirst!(bot_ifaces, k)
            acc = zero(FT)
        end
    end

    first_bot = bot_ifaces[1]
    keep = Int[]
    for i in top_ifaces
        i <= first_bot && push!(keep, i)
    end
    (isempty(keep) || keep[end] < first_bot) && push!(keep, first_bot)
    for i in bot_ifaces
        i > keep[end] && push!(keep, i)
    end

    merged_vc = HybridSigmaPressure(FT[vc.A[k] for k in keep], FT[vc.B[k] for k in keep])
    Nz_merged = n_levels(merged_vc)
    mm = Vector{Int}(undef, Nz)
    km = 1
    for k in 1:Nz
        while km < Nz_merged && keep[km + 1] <= k
            km += 1
        end
        mm[k] = km
    end

    return merged_vc, mm
end

"""
    select_levels_echlevs(vc_native, echlevs)

TM5-style level selection (TM5's `echlevs`): keep a subset of the native
interfaces. The selected interfaces keep their native `A` and `B`
coefficients. The native layers between two selected interfaces become one
output layer, and the preprocessor sums their layer quantities (air mass,
horizontal mass fluxes) through the returned `merge_map`, as TM5 does.

`echlevs` uses TM5/ECMWF half-level numbering: `e = 0` is the model top and
`e = 137` is the surface for ERA5 L137. Following TM5 `geometry.F90`, which
assigns `at(l) = a_ec(echlevs(l-1))`, an entry `e` selects Julia coefficient
index `e + 1`. Thus each retained interface obeys

```math
p_{e+1/2}(p_s) = A_e + B_e p_s,
```

and an output-layer extensive quantity is

```math
X_l = \\sum_{k=e_l+1}^{e_{l+1}} X_k.
```

TM5 files conventionally list interfaces from surface to top, e.g.
`[137, 134, 129, ..., 7, 0]` for `"ml137_tropo34"`; order does not matter to
this function because it sorts the indices into the model's top-to-surface
array order.

Returns `(selected_vc, merge_map)`, where `merge_map[k]` is the output layer
that contains native layer `k` (both counted from the top).
"""
function select_levels_echlevs(vc_native::HybridSigmaPressure{FT},
                               echlevs::AbstractVector{<:Integer}) where FT
    Nz_native = n_levels(vc_native)
    isempty(echlevs) && throw(ArgumentError("echlevs must not be empty"))
    all(e -> 0 <= e <= Nz_native, echlevs) ||
        throw(ArgumentError("echlevs interface indices must lie in 0:$Nz_native"))
    length(unique(echlevs)) == length(echlevs) ||
        throw(ArgumentError("echlevs interface indices must be unique"))
    keep = sort!(Int[e + 1 for e in echlevs])
    first(keep) == 1 && last(keep) == Nz_native + 1 ||
        throw(ArgumentError("echlevs must include model top 0 and surface $Nz_native"))

    selected_vc = HybridSigmaPressure(FT[vc_native.A[k] for k in keep],
                                      FT[vc_native.B[k] for k in keep])
    Nz_selected = n_levels(selected_vc)

    mm = Vector{Int}(undef, Nz_native)
    km = 1
    for k in 1:Nz_native
        while km < Nz_selected && keep[km + 1] <= k
            km += 1
        end
        mm[k] = km
    end

    @info "echlevs level selection: $(Nz_native) → $(Nz_selected) levels " *
          "($(length(echlevs)) interfaces)"
    return selected_vc, mm
end

select_levels_echlevs(vc_native::HybridSigmaPressure,
                      echlevs::Tuple{Vararg{Integer}}) =
    select_levels_echlevs(vc_native, collect(Int, echlevs))

# Keep the exported, pinned recipes as immutable tuples, but put a concrete
# type boundary in front of dynamic name lookup.  Looking up a runtime Symbol
# in ERA5_L137_ECHLEVS_PRESETS directly makes inference consider every tuple
# length (35 through 138); all fields here instead have the same Vector{Int}
# type.  echlevs_preset returns a copy, so these private vectors cannot be
# mutated through the public accessor.
const _ERA5_L137_ECHLEVS_PRESET_VECTORS =
    map(echlevs -> collect(Int, echlevs), ERA5_L137_ECHLEVS_PRESETS)

"""
    echlevs_preset(name) -> Vector{Int}

Resolve a named ERA5 L137 interface-selection preset. The returned vector is
copied from the immutable process-wide preset library. An unknown name is an
error that lists the available presets.

The two 34-layer TM5 selections are `"ml137_tropo34"` (TM5-MP r1112
`levels/ml137/tropo34`) and `"ml137_tropo34a"` (TM5 cy3-4dvar
`levels/ml137/tropo34a`, the CAMS73 selection); both store TM5's lists
verbatim. The AtmosTransport-native presets store TM5-numbered indices chosen
to retain their grids from before the selector's TM5-numbering correction.
"""
function echlevs_preset(name::AbstractString)
    key = Symbol(name)
    hasproperty(_ERA5_L137_ECHLEVS_PRESET_VECTORS, key) ||
        error("Unknown echlevs config: $(String(name)). Available: " *
              join(sort(collect(string.(propertynames(ERA5_L137_ECHLEVS_PRESETS)))), ", "))
    return copy(getproperty(_ERA5_L137_ECHLEVS_PRESET_VECTORS, key))
end

function load_era5_vertical_coordinate(coeff_path::String, level_top::Int, level_bot::Int)
    isfile(coeff_path) || error("Coefficients not found: $coeff_path")
    cfg = TOML.parsefile(coeff_path)
    a_all = Float64.(cfg["coefficients"]["a"])
    b_all = Float64.(cfg["coefficients"]["b"])
    return HybridSigmaPressure(a_all[level_top:level_bot+1], b_all[level_top:level_bot+1])
end

"""
    load_hybrid_coefficients(coeff_path::String) -> HybridSigmaPressure

Load all hybrid sigma-pressure interface coefficients from a TOML file.
Unlike `load_era5_vertical_coordinate`, this does not slice — useful for
sources whose level count comes from the file rather than a config knob
(e.g. GEOS-72, MERRA-2, native ERA5 L137 without sub-tropo selection).
"""
function load_hybrid_coefficients(coeff_path::String)
    isfile(coeff_path) || error("Coefficients not found: $coeff_path")
    cfg = TOML.parsefile(coeff_path)
    return HybridSigmaPressure(Float64.(cfg["coefficients"]["a"]),
                               Float64.(cfg["coefficients"]["b"]))
end

function load_ab_coefficients(coeff_path::String, level_range)
    isfile(coeff_path) || error("Coefficients not found: $coeff_path")
    cfg = TOML.parsefile(coeff_path)
    a_all = Float64.(cfg["coefficients"]["a"])
    b_all = Float64.(cfg["coefficients"]["b"])
    Nz = length(level_range)
    i_start = level_range[1]
    i_end = level_range[end] + 1
    a_ifc = a_all[i_start:i_end]
    b_ifc = b_all[i_start:i_end]
    dA = diff(a_ifc)
    dB = diff(b_ifc)
    A_center = [(a_ifc[k] + a_ifc[k + 1]) / 2 for k in 1:Nz]
    B_center = [(b_ifc[k] + b_ifc[k + 1]) / 2 for k in 1:Nz]
    return (; a_ifc, b_ifc, dA, dB, A_center, B_center)
end
