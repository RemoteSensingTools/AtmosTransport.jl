# ===========================================================================
# Head-to-head: :endpoint_balanced vs :moisture_filtered cm closure, same day,
# same merged C180 grid. Confirms (1) the SH-UTLS cm fingering DROPS, (2) the
# cm fields actually DIFFER, and (3) surface pressure is conserved per column.
#
#   julia --project=. scripts/completed_experiments/compare_moisturefiltered_vs_endpoint.jl
# ===========================================================================
using AtmosTransport
using AtmosTransport.MetDrivers: TransportBinaryReader, load_window!, load_grid
using AtmosTransport.Grids: panel_cell_center_lonlat
using AtmosTransport.Architectures: CPU
using Printf

const BIN_A = "/home/cfranken/data/AtmosTransport/met/geosit/C180/transport_binary_dec2021_catrine_f32/geos_transport_20211211_float32.bin"            # endpoint
const BIN_B = "/home/cfranken/data/AtmosTransport/met/geosit/C180/transport_binary_dec2021_catrine_f32_moisturefiltered/geos_transport_20211211_float32.bin"  # filtered

_std(x) = (m = sum(x)/length(x); sqrt(sum(v -> (v-m)^2, x) / max(1, length(x)-1)))

# Normalized grid-scale roughness: RMS(5-pt Laplacian)/std over a masked region.
function rough_masked(field_panels, mask_panels, Nc)
    lap = Float64[]; vals = Float64[]
    for p in 1:6
        f = field_panels[p]; msk = mask_panels[p]
        for j in 2:Nc-1, i in 2:Nc-1
            msk[i, j] || continue
            l = 4f[i,j] - f[i-1,j] - f[i+1,j] - f[i,j-1] - f[i,j+1]
            push!(lap, Float64(l)); push!(vals, Float64(f[i,j]))
        end
    end
    s = _std(vals); s == 0 && return 0.0
    return sqrt(sum(abs2, lap) / length(lap)) / s
end

function main()
    println("Opening readers …")
    rA = TransportBinaryReader(BIN_A; FT = Float32)
    rB = TransportBinaryReader(BIN_B; FT = Float32)
    Nc = rA.header.geometry.Nc; Nz = rA.header.nlevel
    @printf("  Nc=%d  Nz=%d (merged, TOA-first)\n", Nc, Nz)

    mesh = load_grid(rA; FT = Float32, arch = CPU(), Hp = 0).horizontal
    lats = ntuple(p -> Float64.(panel_cell_center_lonlat(mesh, p)[2]), 6)
    sh   = ntuple(p -> lats[p] .< -30.0, 6)
    @printf("  SH cells (lat<-30): %d of %d\n", sum(sum, sh), 6Nc*Nc)

    for win in (1, 6, 12)
        wA = load_window!(rA, win); wB = load_window!(rB, win)
        # Direct: do the cm fields differ at all? (max over full sphere + SH)
        dmax = 0.0; dmax_sh = 0.0; cmscale = 0.0
        for p in 1:6, k in 1:Nz+1, j in 1:Nc, i in 1:Nc
            d = abs(Float64(wB.cm[p][i,j,k]) - Float64(wA.cm[p][i,j,k]))
            dmax = max(dmax, d)
            cmscale = max(cmscale, abs(Float64(wA.cm[p][i,j,k])))
            sh[p][i,j] && (dmax_sh = max(dmax_sh, d))
        end
        @printf("\n=== window %d ===\n", win)
        @printf("  max|cm_B - cm_A| = %.3e   (cm scale %.3e ⇒ rel %.2e)   SH-only %.3e\n",
                dmax, cmscale, cmscale == 0 ? NaN : dmax/cmscale, dmax_sh)
        println("  cm SH roughness by interface level: endpoint  filtered  ratio")
        for iface in 30:2:46
            iface > Nz && continue
            fA = ntuple(p -> wA.cm[p][:, :, iface], 6)
            fB = ntuple(p -> wB.cm[p][:, :, iface], 6)
            rgA = rough_masked(fA, sh, Nc); rgB = rough_masked(fB, sh, Nc)
            @printf("    lev %3d   %.4f   %.4f   %.3f×\n", iface, rgA, rgB,
                    rgA == 0 ? NaN : rgB/rgA)
        end
    end

    # ps (column dry mass) conservation: window-2 start = window-1 endpoint.
    mA = load_window!(rA, 2).m; mB = load_window!(rB, 2).m
    max_ps = 0.0; max_lay = 0.0; dcol = 0.0; dlay = 0.0
    for p in 1:6, j in 1:Nc, i in 1:Nc
        cA = 0.0; cB = 0.0
        for k in 1:Nz
            cA += Float64(mA[p][i,j,k]); cB += Float64(mB[p][i,j,k])
            max_lay = max(max_lay, abs(Float64(mB[p][i,j,k]) - Float64(mA[p][i,j,k])))
            dlay = max(dlay, abs(Float64(mA[p][i,j,k])))
        end
        dcol = max(dcol, abs(cA)); max_ps = max(max_ps, abs(cB - cA))
    end
    println("\n=== ps / per-layer mass: window-2 start (= window-1 endpoint) ===")
    @printf("  max |Δ column mass| / max column   = %.3e   (ps drift; expect ~0)\n", max_ps/dcol)
    @printf("  max |Δ per-layer mass| / max layer = %.3e   (the filtering; expect > 0)\n", max_lay/dlay)
    println("\nDone.")
end

main()
