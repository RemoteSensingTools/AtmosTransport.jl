# ===========================================================================
# OMEGA probe: is the model's own PHYSICAL vertical motion (OMEGA, on the C180
# cube, A3dyn) much smoother at the SH-UTLS than the quantity the diagnosed cm
# is forced to integrate (div_h(MFXC), the fingering driver)?
#
# The diagnosed cm[k+1]=cm[k]-div_h[k]-dm[k] absorbs the grid-noisy MFXC<->DELP
# residual M into cm -> fingering. OMEGA (Pa/s) is the model's resolved vertical
# pressure velocity -- a smooth physical field. If rel-roughness(OMEGA) <<
# rel-roughness(div_h MFXC) at the SH-UTLS, then ingesting a PHYSICAL vertical
# flux (archived MFZ, or an OMEGA-derived cm) on the NATIVE grid would cut the
# fingering -- the cubed-sphere alternative to the lat-lon wind path.
#
# Metric: SH-interior relative roughness RMS(d2)/std (scale-invariant), as in
# wind_vs_mfxc_divh_roughness.jl. Both fields read through the GEOS reader so
# they share the (i,j) panel convention + level ordering.
#   julia --project=. scripts/completed_experiments/omega_vs_cm_driver_roughness.jl [ctm_win] [a3_win]
# ===========================================================================
using AtmosTransport, AtmosTransport.Preprocessing, AtmosTransport.Grids
using Dates, NCDatasets, Statistics, Printf, TOML
const P = AtmosTransport.Preprocessing
const CONFIG = joinpath(@__DIR__, "..", "..",
    "config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml")
const DATE = Date(2021, 12, 11)
const CTM_WIN = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
const A3_WIN  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4

function main()
    cfg = TOML.parsefile(CONFIG); FT = Float64
    grid = P.build_target_geometry(cfg["grid"], FT); Nc = grid.Nc
    src = cfg["source"]
    settings = P.load_met_settings(joinpath(@__DIR__, "..", "..", String(src["toml"]));
        root_dir = AtmosTransport.expand_data_path(String(src["root_dir"])),
        include_surface = false, include_convection = false, include_vdiff_fields = true,
        coefficients_file = AtmosTransport.expand_data_path(String(cfg["vertical"]["coefficients"])))
    vc = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(settings.coefficients_file))
    vertical = P._build_native_vertical_setup(cfg["vertical"], vc, FT); Nz = vertical.Nz
    Aifc = Float64.(vertical.merged_vc.A); Bifc = Float64.(vertical.merged_vc.B)
    sh = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh, p)[2]) .< -30.0, 6)

    reader = P.open_reader(settings, DATE, FT; chain_mass = false, next_day_handle = true)
    or = reader.handles.orientation
    reader.handles.a3dyn === nothing && error("A3dyn handle not open (need include_vdiff_fields=true)")
    mfxc = P._read_panels_3d(reader.handles.ctm_a1, "MFXC", CTM_WIN, or; FT = FT)
    mfyc = P._read_panels_3d(reader.handles.ctm_a1, "MFYC", CTM_WIN, or; FT = FT)
    omega = P._read_panels_3d(reader.handles.a3dyn, "OMEGA", A3_WIN, or; FT = FT)
    @printf("Nc=%d Nz=%d  ctm_win=%d  a3_win=%d\n", Nc, Nz, CTM_WIN, A3_WIN)

    # div_h(MFXC) = the diagnosed-cm driver (the fingering source)
    div_nat = ntuple(p -> begin
        d = fill(NaN, Nc, Nc, Nz)
        @inbounds for k in 1:Nz, j in 2:Nc, i in 2:Nc
            d[i, j, k] = (mfxc[p][i-1, j, k] - mfxc[p][i, j, k]) +
                         (mfyc[p][i, j-1, k] - mfyc[p][i, j, k])
        end; d
    end, 6)

    function sh_rel_rough(f3, k)
        lap = Float64[]; vals = Float64[]
        for p in 1:6, j in 3:Nc-2, i in 3:Nc-2
            sh[p][i, j] || continue
            f = f3[p]
            (isnan(f[i,j,k])||isnan(f[i+1,j,k])||isnan(f[i-1,j,k])||isnan(f[i,j+1,k])||isnan(f[i,j-1,k])) && continue
            push!(lap, f[i,j,k]-0.25*(f[i+1,j,k]+f[i-1,j,k]+f[i,j+1,k]+f[i,j-1,k]))
            push!(vals, f[i,j,k])
        end
        (isempty(lap)||length(vals)<2) && return NaN
        sd = std(vals); sd > 0 ? sqrt(mean(abs2,lap))/sd : NaN
    end

    println("\n  TOAlev  p(hPa)   div_h(MFXC) rel   OMEGA rel   OMEGA/div_h")
    psref = 1.0e5
    for k in 30:2:50
        rd = sh_rel_rough(div_nat, k)
        ro = sh_rel_rough(omega, k)
        pmid = 0.5*(Aifc[k]+Aifc[k+1] + (Bifc[k]+Bifc[k+1])*psref)/100
        @printf("   %3d   %6.1f      %.4f          %.4f      %.3f\n", k, pmid, rd, ro, ro/rd)
    end
    println("\n(OMEGA/div_h << 1 ⇒ the physical vertical motion is far smoother than the cm driver,")
    println(" so ingesting a physical vertical flux (MFZ / OMEGA-derived cm) on the NATIVE cube")
    println(" would cut the SH-UTLS fingering — the native alternative to the lat-lon wind path.)")
end
main()
