# ===========================================================================
# P2 proof: the `:omega_consistent` OMEGA/QV PCHIP target at the day-boundary
# windows is now INTERPOLATED across midnight (using the prev/next-day A3dyn/I3
# nodes) instead of CONSTANT-extrapolated to the nearest same-day node.
#
# For each boundary window we print the PCHIP target at a representative cell
# computed two ways:
#   BEFORE = legacy bracket clamped to same-day nodes [1..n3] (constant-extrap)
#   AFTER  = cross-day bracket spanning prev/next-day nodes (this fix)
# A nonzero BEFORE→AFTER delta at win 1/23/24 (and ~0 at an interior window)
# proves the midnight discontinuity is removed.
#
# Usage:
#   julia --project=. scripts/completed_experiments/omega_midnight_pchip_proof.jl \
#       config/preprocessing/<omega>.toml 2021-12-01
# ===========================================================================
using AtmosTransport
using AtmosTransport.Preprocessing
const P = AtmosTransport.Preprocessing
using Dates, Printf, TOML

# Legacy same-day-only bracket (the committed behaviour before P2).
function legacy_bracket(valid_min, n3, t)
    a = 1
    for ai in 1:n3
        valid_min(ai) <= t && (a = ai)
    end
    a0 = clamp(a, 1, n3); a1 = clamp(a + 1, 1, n3)
    t0 = Float64(valid_min(a0)); t1 = Float64(valid_min(a1))
    f = (t1 == t0) ? 0.0 : clamp((t - t0) / (t1 - t0), 0.0, 1.0)
    am1 = clamp(a0 - 1, 1, n3); ap2 = clamp(a1 + 1, 1, n3)
    return (am1, a0, a1, ap2), f
end

# Evaluate a PCHIP target at one (i,j,k) cell, panel 1, for a given bracket and
# field, resolving global node indices across prev/today/next datasets.
function eval_cell(var, valid_min, n3, t, today, prev, next, or, cell, cross::Bool)
    if cross
        gmin = prev === nothing ? 1 : 1 - prev.dim["time"]
        gmax = next === nothing ? n3 : n3 + next.dim["time"]
        (nodes, f, _) = P._pchip_bracket_global(valid_min, n3, t, gmin, gmax)
        resolve = g -> P._resolve_global_node(g, n3, today, prev, next)
    else
        (nodes, f) = legacy_bracket(valid_min, n3, t)
        resolve = g -> (today, g)
    end
    ys = map(nodes) do g
        (ds, loc) = resolve(g)
        P._read_panels_3d(ds[var], loc, or; FT = Float64)
    end
    (i, j, k) = cell
    y1 = ys[1][1][i, j, k]; y2 = ys[2][1][i, j, k]
    y3 = ys[3][1][i, j, k]; y4 = ys[4][1][i, j, k]
    return P._pchip_eval(y1, y2, y3, y4, f), nodes, f
end

function main()
    cfg = TOML.parsefile(abspath(ARGS[1]))
    date = Date(ARGS[2])
    src = cfg["source"]
    toml_path = joinpath(dirname(abspath(ARGS[1])), "..", "..", String(src["toml"]))
    skw = (root_dir = P.expand_data_path(String(src["root_dir"])),
           include_surface = true, include_convection = true,
           include_vdiff_fields = true)
    cfg_vertical = get(cfg, "vertical", Dict())
    haskey(cfg_vertical, "coefficients") &&
        (skw = (skw..., coefficients_file = P.expand_data_path(String(cfg_vertical["coefficients"]))))
    settings = P.load_met_settings(toml_path; skw...)
    # `adjacent_omega=true` opens the prev/next-day A3dyn+I3 handles the
    # `:omega_consistent` closure needs for the cross-midnight PCHIP bracket.
    h = P.open_geos_day(settings, date; next_day_handle = true, adjacent_omega = true)
    or = h.orientation
    n3_a3 = h.a3dyn.dim["time"]; n3_i3 = h.i3.dim["time"]
    @printf("date=%s orientation=%s  n3_a3=%d n3_i3=%d\n", date, or, n3_a3, n3_i3)
    @printf("prev_a3dyn=%s next_a3dyn=%s prev_i3=%s next_i3=%s\n",
            h.prev_a3dyn !== nothing, h.next_a3dyn !== nothing,
            h.prev_i3 !== nothing, h.next_i3 !== nothing)
    cell = (90, 90, 36)   # mid-panel, mid-level representative cell
    for win in (1, 12, 23, 24)
        t = Float64(P._ctm_valid_min(win))
        oa, na, fa = eval_cell("OMEGA", P._a3_valid_min, n3_a3, t, h.a3dyn,
                               h.prev_a3dyn, h.next_a3dyn, or, cell, true)
        ob, nb, fb = eval_cell("OMEGA", P._a3_valid_min, n3_a3, t, h.a3dyn,
                               h.prev_a3dyn, h.next_a3dyn, or, cell, false)
        qa, _, _ = eval_cell("QV", P._i3_valid_min, n3_i3, t, h.i3,
                             h.prev_i3, h.next_i3, or, cell, true)
        qb, _, _ = eval_cell("QV", P._i3_valid_min, n3_i3, t, h.i3,
                             h.prev_i3, h.next_i3, or, cell, false)
        @printf("\nwin %2d (valid %02d:%02d):\n", win, t ÷ 60, t % 60)
        @printf("  OMEGA  BEFORE(same-day clamp)=% .6e  nodes=%s f=%.3f\n", ob, nb, fb)
        @printf("  OMEGA  AFTER (cross-midnight)=% .6e  nodes=%s f=%.3f\n", oa, na, fa)
        @printf("  OMEGA  Δ(after-before)=% .3e\n", oa - ob)
        @printf("  QV     BEFORE=% .6e  AFTER=% .6e  Δ=% .3e\n", qb, qa, qa - qb)
    end
    P.close_geos_day!(h)
end
main()
