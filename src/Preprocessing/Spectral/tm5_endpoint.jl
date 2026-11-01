"""Vertical reduction prepared once for every endpoint of a TM5 day."""
struct TM5EndpointPlan{V,P}
    vertical::V
    reduction::P
end
function TM5EndpointPlan(vertical)
    validate_spectral_vertical(TM5SpectralFluxes(), vertical)
    interfaces = [0; [findlast(==(k), vertical.merge_map) for k in 1:vertical.Nz]]
    return TM5EndpointPlan(vertical, plan_vertical(LevelSelection(interfaces), vertical.vc_native))
end

"""Pin integrated PS after integration, merge moist fields and humidity, retain W interfaces."""
function construct_tm5_endpoint(ws::SpectralIntegralWorkspace,H,vo,d,T,q_native,endpoint_plan,settings)
    vertical = endpoint_plan.vertical
    c=spectral_construction(settings)
    nz=vertical.Nz_native
    nc=length(ws.quadrature.areas)
    raw=integrate_face_fluxes!(ws,H,vo,d,T,vertical.ab.dA,vertical.ab.dB,c)
    q=reshape(clamp.(Float64.(q_native),0,MAX_SPECIFIC_HUMIDITY),nc,nz)
    offset=settings.mass_fix_enable ? pin_global_mean_ps_using_qv!(raw.ps,ws.quadrature.areas,
        vertical.ab.dA,vertical.ab.dB,q;
        target_ps_dry_pa=settings.target_ps_dry_pa) : 0.0
    m=zeros(nc,nz)
    for k in 1:nz, cell in 1:nc
        m[cell,k]=ws.quadrature.areas[cell]/STANDARD_GRAVITY*(vertical.ab.dA[k]+vertical.ab.dB[k]*raw.ps[cell])
    end
    all(>(0),m) || throw(ArgumentError("tm5_spectral integrated layer mass must be positive"))
    W=zeros(nc,nz+1)
    integrate_vertical_flux!(W,raw.C,vertical.vc_native.B)
    plan = endpoint_plan.reduction
    function merge_extensive(input,kind)
        out=zeros(size(input,1),1,vertical.Nz)
        apply_vertical!(out,reshape(input,size(input,1),1,nz),plan,kind)
        return reshape(out,size(input,1),vertical.Nz)
    end
    wm=zeros(nc,1,vertical.Nz+1)
    apply_vertical!(wm,reshape(W,nc,1,nz+1),plan,InterfaceMassFluxField())
    qm=zeros(nc,1,vertical.Nz)
    merge_qv!(qm,reshape(q,nc,1,nz),reshape(m,nc,1,nz),vertical.merge_map)
    return (;ps=raw.ps,m=merge_extensive(m,MassField()),q=reshape(qm,nc,vertical.Nz),
        u=merge_extensive(raw.u,MassFluxField()),v=merge_extensive(raw.v,MassFluxField()),
        W=reshape(wm,nc,vertical.Nz+1),offset)
end
