# Synthetic CS columns in runtime order. Deliberately poison mass halos and dz.
using AtmosTransport
using AtmosTransport.Operators.Diffusion: TM5CalcKzzCSField, refresh_tm5_calc_kzz!,
    ImplicitVerticalDiffusion, DiffusionWorkspace, apply_vertical_diffusion_vmr!
const CSMD = AtmosTransport.MetDrivers
const CSBF = CSMD.BinaryFormat
const CSTD = AtmosTransport.Operators.Diffusion
const CSPH = AtmosTransport.Models.Physics

function tm5_cs_fixture(::Type{FT}=Float64;Nc=3,Hp=2,chunk_columns=65_536,
                         convention=AtmosTransport.Grids.GnomonicPanelConvention()) where FT
    mesh=CubedSphereMesh(;FT,Nc,Hp,convention)
    vc=HybridSigmaPressure(FT[0,10000,30000,60000,80000,95000,100000],zeros(FT,7))
    grid=AtmosGrid(mesh,vc,CPU();FT)
    metrics=CSPH._tm5_metrics(mesh,FT)
    input=NamedTuple{CSBF.TM5_PBL_FIELD_NAMES}(map(CSBF.TM5_PBL_FIELD_NAMES) do name
        value=get((;t=280,q=.005,lsm=.4,sr=.03,fsr=.1,u10=6,v10=2,ewss=.1,nsss=.05,sshf=-100,slhf=-30,oro=0),name,0)
        ntuple(_->fill(FT(value),name in (:t,:q) ? (Nc,Nc,6) : (Nc,Nc)),6)
    end)
    mass=ntuple(_->fill(FT(NaN),Nc+2Hp,Nc+2Hp,6),6)
    for p in 1:6,k in 1:6,j in 1:Nc,i in 1:Nc
        mass[p][i+Hp,j+Hp,k]=FT(metrics.area[i,j]*(vc.A[k+1]-vc.A[k])/9.80665*(1-.005))
        input.t[p][i,j,k]=FT(300*((vc.A[k]+vc.A[k+1])/2/1e5)^(287.2978576364893/1004))
    end
    fluxes=allocate_face_fluxes(mesh,6;FT,basis=DryBasis)
    for arr in (fluxes.am...,fluxes.bm...,fluxes.cm...)
        fill!(arr,0)
    end
    field=TM5CalcKzzCSField(ntuple(_->zeros(FT,Nc,Nc,6),6),metrics,vc;chunk_columns)
    return (;grid,mass,input,fluxes,field)
end

function tm5_cs_window(f;input=f.input)
    h=f.grid.horizontal.Hp; n=f.grid.horizontal.Nc
    interior(a)=copy(view(a,h+1:h+n,h+1:h+n,:))
    return (;m=map(interior,f.mass),ps=ntuple(_->fill(1e5,n,n),6),
        am=map(a->copy(view(a,h+1:h+n+1,h+1:h+n,:)),f.fluxes.am),
        bm=map(a->copy(view(a,h+1:h+n,h+1:h+n+1,:)),f.fluxes.bm),
        cm=map(a->copy(view(a,h+1:h+n,h+1:h+n,:)),f.fluxes.cm),tm5_pbl=input)
end
