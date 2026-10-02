# HybridSigmaPressure, n_levels, pressure_at_interface, level_thickness
# are imported from ..Grids via the parent Preprocessing module.

"""
    merge_thin_levels(vc; min_thickness_Pa=1000, p_surface=101325)

Merge adjacent vertical levels that are thinner than `min_thickness_Pa` at
the reference surface pressure. Returns `(merged_vc, merge_map)` where
`merge_map[k]` maps native level `k` to the merged level index.
"""
function merge_thin_levels(vc::HybridSigmaPressure{FT};
                           min_thickness_Pa::Real = FT(1000),
                           p_surface::Real = FT(101325)) where FT
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

TM5 r1112-style level selection: pick specific native levels by index.
No summation — am/bm at selected levels are the native values at those levels,
and cm at selected interfaces are the native cm at those interfaces.

`echlevs` is a vector of native level INTERFACE indices (0-based, bottom-up),
e.g., [137, 134, 129, ..., 7, 0] for TM5's ml137→tropo34 config.
Index 0 = TOA, 137 = surface.

Returns (selected_vc, merge_map) where merge_map[k] = merged level index
for native level k. Levels between selected interfaces are summed (like TM5).
"""
function select_levels_echlevs(vc_native::HybridSigmaPressure{FT},
                               echlevs::Vector{Int}) where FT
    Nz_native = n_levels(vc_native)
    ifaces_0based = sort(echlevs)
    keep = Int[Nz_native + 1 - i for i in reverse(ifaces_0based)]

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

const ECHLEVS_ML137_TROPO34 = [
    137, 134, 129, 124, 119, 114, 110, 105, 101,  97,
     93,  88,  84,  81,  78,  76,  73,  70,  67,  65,
     62,  59,  57,  54,  51,  46,  42,  37,  32,  27,
     22,  17,  12,   7,   0]

const ECHLEVS_ML137_66L = [
    137, 135, 130, 125, 120, 115, 110, 105, 100,
     96,  93,  90,  87,  84,  81,  78,  75,  72,  69,  66,
     63,  60,  58,  56,  54,  52,  50,  48,  46,  44,
     42,  40,  38,  36,  34,  32,  30,  29,  28,  27,
     26,  25,  24,  23,  22,  21,  20,  19,  18,  17,
     16,  15,  14,  13,  12,  11,  10,   9,   8,   7,
      6,   5,   4,   3,   2,   1,   0]

# CFL-oriented ERA5 L137 presets from the cubed-sphere subcycling analysis.
# Native ERA5 groups, top down:
#   ml137_cfl94: 1:24 x12, 25:53 x4, 54:71 x1, 72:137 x1
#   ml137_cfl85: 1:24 x12, 25:53 x4, 54:71 x2, 72:137 x1
const ECHLEVS_ML137_CFL94 = vcat(
    [137, 125, 113, 109, 105, 101, 97, 93, 89, 85],
    collect(84:-1:0),
)

const ECHLEVS_ML137_CFL85 = vcat(
    [137, 125, 113, 109, 105, 101, 97, 93, 89, 85,
      84,  82,  80,  78,  76,  74, 72, 70, 68, 66],
    collect(65:-1:0),
)

const ECHLEVS_ML137_PRESETS = Dict(
    "ml137_tropo34" => ECHLEVS_ML137_TROPO34,
    "ml137_66L"     => ECHLEVS_ML137_66L,
    "ml137_cfl94"   => ECHLEVS_ML137_CFL94,
    "ml137_cfl85"   => ECHLEVS_ML137_CFL85,
    "ml137_94L"     => ECHLEVS_ML137_CFL94,
    "ml137_85L"     => ECHLEVS_ML137_CFL85,
    "ml137_full"    => collect(137:-1:0),
)

"""
    echlevs_preset(name) -> Vector{Int}

Resolve a named ERA5 L137 interface-selection preset. The returned vector is
copied so callers cannot mutate the process-wide preset library.
"""
function echlevs_preset(name::AbstractString)
    key = String(name)
    haskey(ECHLEVS_ML137_PRESETS, key) ||
        error("Unknown echlevs config: $key. Available: " *
              join(sort!(collect(keys(ECHLEVS_ML137_PRESETS))), ", "))
    return copy(ECHLEVS_ML137_PRESETS[key])
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
