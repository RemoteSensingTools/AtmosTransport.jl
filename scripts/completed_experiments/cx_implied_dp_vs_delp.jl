# ===========================================================================
# CX/CY M-localization: GMAO computes MFXC = (Courant CX)·(model dp)·area/dt, so
#   dp_flux = MFXC / CX / area   (the Courant-weighted dp the FLUX transported)
# Comparing dp_flux to the archived DELP localizes where the accumulated mass
# flux and the analyzed pressure thickness DISAGREE — i.e. the source of the
# residual M = Σ_k dm_dry − pit_native that drives the SH-UTLS fingering.
# Reports, by pressure level: the SH-mean relative gap |dp_flux−DELP|/DELP and
# its grid-scale roughness, so we can see if the disagreement is UTLS-localized.
#   julia --project=. scripts/completed_experiments/cx_implied_dp_vs_delp.jl [window]
# ===========================================================================
using AtmosTransport, AtmosTransport.Preprocessing, AtmosTransport.Grids
using Dates, NCDatasets, Statistics, Printf, TOML
const P = AtmosTransport.Preprocessing
const CONFIG = joinpath(@__DIR__,"..","..","config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml")
const DATE = Date(2021,12,11); WIN = length(ARGS)>=1 ? parse(Int,ARGS[1]) : 12
const CX_MIN = 1.0e-3   # mask near-zero Courant (division blows up where flow ~0)

function main()
    cfg = TOML.parsefile(CONFIG); FT=Float64
    grid = P.build_target_geometry(cfg["grid"], FT); Nc = grid.Nc
    src = cfg["source"]
    settings = P.load_met_settings(joinpath(@__DIR__,"..","..",String(src["toml"]));
        root_dir=AtmosTransport.expand_data_path(String(src["root_dir"])),
        include_surface=false, include_convection=false, include_vdiff_fields=false,
        coefficients_file=AtmosTransport.expand_data_path(String(cfg["vertical"]["coefficients"])))
    vc = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(settings.coefficients_file))
    vertical = P._build_native_vertical_setup(cfg["vertical"], vc, FT); Nz=vertical.Nz
    Aifc=Float64.(vertical.merged_vc.A); Bifc=Float64.(vertical.merged_vc.B)
    areas = grid.mesh.cell_areas
    sh = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh,p)[2]) .< -30.0, 6)
    reader = P.open_reader(settings, DATE, FT; chain_mass=false, next_day_handle=true)
    or = reader.handles.orientation
    @printf("Nc=%d Nz=%d win=%d  (CX_MIN=%.0e)\n", Nc, Nz, WIN, CX_MIN)

    mfxc = P._read_panels_3d(reader.handles.ctm_a1, "MFXC", WIN, or; FT=FT)
    cx   = P._read_panels_3d(reader.handles.ctm_a1, "CX",   WIN, or; FT=FT)
    delp = P._read_panels_3d(reader.handles.ctm_a1, "DELP", WIN, or; FT=FT)

    # dp_flux on the east face of cell i = MFXC[i]/CX[i]/area; the implied dp
    # (×const) the flux carried. Compare to the cell DELP (the analyzed dp).
    # Report SH per-level: median ratio dp_flux/DELP (≈ const if consistent),
    # the grid-scale roughness of that ratio (0 if consistent), and the rel gap.
    function sh_lap_rough(f)  # RMS 5-pt Laplacian / median over SH interior, masked NaN
        lap=Float64[]; vals=Float64[]
        for p in 1:6, j in 2:Nc-1, i in 2:Nc-1
            sh[p][i,j] || continue
            (isnan(f[p][i,j])||isnan(f[p][i+1,j])||isnan(f[p][i-1,j])||isnan(f[p][i,j+1])||isnan(f[p][i,j-1])) && continue
            push!(lap, f[p][i,j]-0.25*(f[p][i+1,j]+f[p][i-1,j]+f[p][i,j+1]+f[p][i,j-1]))
            push!(vals, f[p][i,j])
        end
        (isempty(lap)||isempty(vals)) && return (NaN,NaN)
        return sqrt(mean(abs2,lap)), median(filter(isfinite,vals))
    end

    println("\n  TOAlev  p(hPa)   SH median dp_flux/DELP   rough(ratio)/median   masked%")
    psref=1.0e5
    for k in 30:2:48
        ratio = ntuple(p -> begin
            r = fill(NaN, Nc, Nc)
            @inbounds for j in 1:Nc, i in 1:Nc
                c = cx[p][i,j,k]
                (abs(c) < CX_MIN || delp[p][i,j,k] <= 0) && continue
                r[i,j] = (mfxc[p][i,j,k]/c/areas[i,j]) / delp[p][i,j,k]
            end; r
        end, 6)
        rough, med = sh_lap_rough(ratio)
        nmask = sum(count(isnan, ratio[p][sh[p]]) for p in 1:6; init=0)
        nsh = sum(count, sh; init=0)
        pmid = 0.5*(Aifc[k]+Aifc[k+1]+(Bifc[k]+Bifc[k+1])*psref)/100
        @printf("   %3d   %6.1f      %.4e            %.4f               %.0f%%\n",
                k, pmid, med, isnan(med) ? NaN : rough/abs(med), 100*nmask/nsh)
    end
    println("\n(If rough(ratio)/median is FLAT across levels ⇒ MFXC and CX are internally")
    println(" consistent and the M residual is time-sampling, not a flux↔dp mismatch.")
    println(" If it SPIKES at the UTLS ⇒ the flux-implied dp disagrees with DELP there.)")
end
main()
