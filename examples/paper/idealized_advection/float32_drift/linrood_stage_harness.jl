# Stage-level Float32 mass budget of the production Lin–Rood horizontal step (`fv_tp_2d_cs!`) in
# the idealized flow B on the cubed sphere, uniform tracer by default. After every call the cell
# update is recomputed exactly (Float64) from the Float32 face values the kernel used, which splits
# the change of the global tracer mass into
#   rounding : sum over cells of (stored Float32 result − exact result of the same inputs)
#   mismatch : sum over cells of (exact result − old value) = net of the face fluxes, zero when
#              every face flux enters its two cells with opposite sign (also across panel seams).
# Modes: `production` (the kernel as is) and two prototypes applied to the kernel's fluxes:
# `single_rounding` (store the correctly rounded exact update) and `carry` (also keep what the
# rounding removed and add it to the next update of the same cell).
#
#   julia --project=. -t 8 examples/paper/idealized_advection/float32_drift/linrood_stage_harness.jl \
#         [grid=c48] [field=uniform] [Float32|Float64] [production|single_rounding|carry]
using Printf, Statistics
include(joinpath(@__DIR__, "..", "IdealizedAdvection.jl"))
using .IdealizedAdvection
const IA = IdealizedAdvection
using AtmosTransport
using AtmosTransport.Grids: CubedSphereMesh, GEOSNativePanelConvention
using AtmosTransport.Parameters: GLOBAL_MEAN_DRY_SURFACE_PRESSURE
const Adv = AtmosTransport.Operators.Advection

function exact_update(rm_old, am, bm, lr, p, Nc, Hp, Nz)
    out = zeros(Float64, Nc, Nc, Nz)
    fxi, fxo, fyi, fyo = lr.fx_in[p], lr.fx_out[p], lr.fy_in[p], lr.fy_out[p]
    h = 0.5f0
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        aw = Float64(h * (fxo[i, j, k] + fxi[i, j, k]))
        ae = Float64(h * (fxo[i+1, j, k] + fxi[i+1, j, k]))
        as = Float64(h * (fyo[i, j, k] + fyi[i, j, k]))
        an = Float64(h * (fyo[i, j+1, k] + fyi[i, j+1, k]))
        out[i, j, k] = Float64(rm_old[Hp+i, Hp+j, k]) +
            (Float64(am[i, j, k]) * aw - Float64(am[i+1, j, k]) * ae) +
            (Float64(bm[i, j, k]) * as - Float64(bm[i, j+1, k]) * an)
    end
    return out
end

interior(a, Nc, Hp) = @view a[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]

function main(; label = "c48", ord = 7, field = "uniform", days = 12, FT = Float32, mode = :production)
    g = IA.build_geometry(label)
    flow, tracers = IA.flow_case("B")
    Nc = g.grid.mesh.Nc; Hp = 3
    B = [0.0, 0.15, 0.4, 0.7, 1.0]; frac = diff(B); Nz = length(frac)
    nw = 24 * days
    windows = [((w - 1) * 3600.0, w * 3600.0) for w in 1:nw]
    F, M = IA.column_transfers(flow, g, windows; ps = GLOBAL_MEAN_DRY_SURFACE_PRESSURE)
    steps = ceil(Int, IA.max_outflow_fraction(g, F, M) / 0.5)      # outflow ≤ 0.25 per application
    mesh = CubedSphereMesh(; Nc, Hp, FT, convention = GEOSNativePanelConvention())
    N = Nc + 2Hp
    m0 = IA.pack_cells(g, M[:, 1], frac)
    χ = field == "uniform" ? ones(IA.ncell(g)) : [Dict(tracers)[field](x) for x in g.cell_center]
    q0 = IA.pack_cells(g, χ, ones(Nz))
    m = ntuple(p -> (a = zeros(FT, N, N, Nz); interior(a, Nc, Hp) .= m0[p]; a), 6)
    rm = ntuple(p -> (a = zeros(FT, N, N, Nz); interior(a, Nc, Hp) .= FT.(q0[p]) .* FT.(m0[p]); a), 6)
    ws = Adv.CSLinRoodAdvectionWorkspace(mesh, m[1])
    total(x) = sum(p -> sum(Float64, interior(x[p], Nc, Hp)), 1:6)
    S0 = total(rm)
    round_sum = 0.0; mis_sum = 0.0; nupd = 0; ulp_bias = 0.0
    edge_round = 0.0; inner_round = 0.0
    carry = ntuple(_ -> zeros(Float64, Nc, Nc, Nz), 6)
    @printf("%s flow B, Lin–Rood %d, %s, %s, mode %s: %d substeps per window\n",
            label, ord, field, FT, mode, steps)
    for w in 1:nw
        f = IA.pack_faces(g, F[:, w], frac)
        am = map(a -> FT.(a ./ (2steps)), f.am); bm = map(a -> FT.(a ./ (2steps)), f.bm)
        for p in 1:6
            interior(m[p], Nc, Hp) .= m0[p]
        end
        for _ in 1:2steps
            rm_old = map(copy, rm)
            Adv.fv_tp_2d_cs!(rm, m, am, bm, mesh, Val(ord), ws.cs, ws.linrood)
            for p in 1:6
                ex = exact_update(rm_old[p], am[p], bm[p], ws.linrood, p, Nc, Hp, Nz)
                d = Float64.(interior(rm[p], Nc, Hp)) .- ex
                round_sum += sum(d)
                mis_sum += sum(ex .- Float64.(interior(rm_old[p], Nc, Hp)))
                ulp_bias += sum(d ./ Float64.(eps.(interior(rm_old[p], Nc, Hp))))
                nupd += length(d)
                edge = sum(@view d[1, :, :]) + sum(@view d[Nc, :, :]) +
                       sum(@view d[2:Nc-1, 1, :]) + sum(@view d[2:Nc-1, Nc, :])
                edge_round += edge; inner_round += sum(d) - edge
                if mode != :production
                    target = mode == :carry ? ex .+ carry[p] : ex
                    stored = FT.(target)
                    mode == :carry && (carry[p] .= target .- Float64.(stored))
                    interior(rm[p], Nc, Hp) .= stored
                end
            end
        end
        w % (nw ÷ 4) == 0 || continue
        if mode == :production
            @printf("  hour %3d: total %+.3e   rounding %+.3e (panel-edge cells %+.3e, others %+.3e)   ",
                    w, total(rm) / S0 - 1, round_sum / S0, edge_round / S0, inner_round / S0)
            @printf("flux mismatch %+.3e   mean %+.5f ulp/update\n", mis_sum / S0, ulp_bias / nupd)
        else
            χerr = maximum(p -> maximum(abs, Float64.(interior(rm[p], Nc, Hp)) ./ Float64.(m0[p]) .- q0[p]),
                           1:6)
            @printf("  hour %3d: total %+.3e  (carry held %+.2e; max |chi - 1| %.2e)\n",
                    w, total(rm) / S0 - 1, sum(sum, carry) / S0, χerr)
        end
    end
end

label = length(ARGS) >= 1 ? ARGS[1] : "c48"
field = length(ARGS) >= 2 ? ARGS[2] : "uniform"
FT = length(ARGS) >= 3 && ARGS[3] == "Float64" ? Float64 : Float32
mode = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :production
main(; label, field, FT, mode)
