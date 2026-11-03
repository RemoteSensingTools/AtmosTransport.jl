# CPU-only raw 1° LL face quadrature diagnostic for T639 ERA5; see README.md.
# `harmonics OUT` writes the analytic/direct trapezoid response sweep.
# `day SPECTRAL_DIR YYYY-MM-DD OUT` measures all faces at native levels
# 1/70/137 at 24 hourly analyses and all 137 levels at hour 00. Each component
# uses identical complete wind coefficients on both sides, isolating integration
# (including TM5's meridian polar mean) from other spectral-transform changes.
# `native-day` extends all 24 analyses to all 137 native levels.
# Reference: 16-node Gauss latitude / Fourier-integrated longitude products;
# hour 00 checks doubled reference resolution. No transport stages are run.
# Independent face-only LL quadrature probe; no pin, balance, drying or merging.
using AtmosTransport, LinearAlgebra, Dates, TOML, FFTW, Statistics
const P=AtmosTransport.Preprocessing;const S=P.Spectral
const RV=AtmosTransport.Parameters.IFS_EARTH_RADIUS
const SCALE=RV/AtmosTransport.Parameters.STANDARD_GRAVITY
BLAS.set_num_threads(1)
const CSV_HEADER="hour,level,component,rms_delta_kg_s,rms_reference_kg_s,relative_l2_percent,cap_rms_delta_kg_s,cap_rms_reference_kg_s,cap_relative_l2_percent,nonpolar_rms_delta_kg_s,nonpolar_relative_l2_percent,innercap_rms_delta_kg_s,innercap_relative_l2_percent"

function plans(T,ng=16,Jv=5760)
 t,w=S.gauss_legendre(ng)
 latu=vcat(deg2rad.(collect(-90.:.125:90.)),[deg2rad(-90+j+(x+1)/2) for j in 0:179 for x in t])
 latv=deg2rad.(collect(-89.:89.))
 ru=S.SynthesisRings(latu,fill(1440,length(latu)),zeros(length(latu)))
 rv=S.SynthesisRings(latv,fill(Jv,length(latv)),zeros(length(latv)))
 su=S.CPURingSynthesis(ru,T+1,3);sv=S.CPURingSynthesis(rv,T+1,3)
 hu=S.surface_synthesis(su,1);hv=S.surface_synthesis(sv,1)
 longitude=S.fourier_integral_plan(rv,fill(360,length(latv)),sv,3)
 (;su,sv,hu,hv,w,ng,Jv,longitude)
end
function surfaces(pl,H,T)
 h=zeros(ComplexF64,T+2,T+2,1);h[1:T+1,1:T+1,1].=H
 pu=reshape(exp.(S.synthesize_rings(h,pl.hu)),1440,:)[1:4:end,:]
 pv=reshape(exp.(S.synthesize_rings(h,pl.hv)),pl.Jv,:)
 (;pu,pv)
end
function u_integrals(samples,ps,da,db,pl)
 ng,w=pl.ng,pl.w
 old=zeros(360,180,3);exact=similar(old)
 for k in 1:3
  f=reshape(view(samples,:,k),1440,:)[1:4:end,:]
  f .*= da[k].+db[k].*ps
  for j in axes(f,2)
   cp=cos(pl.su.plan.rings.latitudes[j]);abs(cp)>1e-14 && (f[:,j]./=cp)
  end
  # TM5 includes duplicate longitude 360 in the polar endpoint mean.
  f[:,1].=(sum(f[:,2])+f[1,2])/361
  f[:,1441].=(sum(f[:,1440])+f[1,1440])/361
  for j in 1:180, i in 1:360
   first=1+(j-1)*8
   old[i,j,k]=SCALE*deg2rad(.125)*(.5f[i,first]+sum(view(f,i,first+1:first+7))+.5f[i,first+8])
   first=1442+(j-1)*ng
   exact[i,j,k]=SCALE*deg2rad(.5)*sum(w[q]*f[i,first+q-1] for q in 1:ng)
  end
 end
 old,exact
end
function v_integrals(samples,ps,da,db,pl)
 J=pl.Jv;r=J÷360;old=zeros(360,181,3);exact=copy(old);L=pi/180
 values=reshape(samples,J,:,3).*(reshape(da,1,1,3).+reshape(db,1,1,3).*ps)
 # The diagnostic and production use the same exact Fourier integrator.
 integrals=S.fourier_integrals!(reshape(values,:,3),pl.longitude,3)
 exact[:,2:180,:].=SCALE.*reshape(integrals,360,179,3)
 for k in 1:3
  f=view(values,:,:,k)
  for j in 1:179, i in 1:360
   first=1+(i-1)*r
   old[i,j+1,k]=SCALE*L/8*(.5f[first,j]+sum(f[first+q*(r÷8),j] for q in 1:7)+.5f[mod1(first+r,J),j])
  end
 end
 old,exact
end
function evaluate(pl,surf,vo,d,T,da,db)
 u,v=S.complete_wind_coefficients(vo,d,T,RV)
 us=S.synthesize_rings(u,pl.su);vs=S.synthesize_rings(v,pl.sv)
 u_integrals(us,surf.pu,da,db,pl),v_integrals(vs,surf.pv,da,db,pl)
end
function stats(old,ref,cap)
 delta=old.-ref
 (sqrt(mean(abs2,delta)),sqrt(mean(abs2,ref)),sqrt(sum(abs2,delta)/sum(abs2,ref))*100,
 sqrt(mean(abs2,delta[:,cap])),sqrt(mean(abs2,ref[:,cap])),sqrt(sum(abs2,delta[:,cap])/sum(abs2,ref[:,cap]))*100)
end
function write_stats(io,hour,levels,result)
 for (kind,(old,ref)) in zip(("u","v"),result)
  lat=kind=="u" ? collect(-89.5:89.5) : collect(-90.:90.)
  cap=abs.(lat).>85
  for (k,level) in enumerate(levels)
   row=stats(old[:,:,k],ref[:,:,k],cap)
   delta=old[:,:,k]-ref[:,:,k]
   ordinary=abs.(lat).<89
   innercap=cap .& ordinary
   extra=(sqrt(mean(abs2,delta[:,ordinary])),100norm(delta[:,ordinary])/norm(ref[:,ordinary,k]),
     sqrt(mean(abs2,delta[:,innercap])),100norm(delta[:,innercap])/norm(ref[:,innercap,k]))
   println(io,join((hour,level,kind,row...,extra...),','))
  end
 end
 flush(io)
end
function write_native_statistics(io,hour,pl,surf,endpoint,da,db)
 T=endpoint.T
 for firstlevel in 1:3:137
  levels=collect(firstlevel:min(firstlevel+2,137));n=length(levels)
  vp=zeros(ComplexF64,T+1,T+1,3);dp=copy(vp)
  vp[:,:,1:n].=endpoint.vo[:,:,levels];dp[:,:,1:n].=endpoint.d[:,:,levels]
  aa=zeros(3);bb=zeros(3);aa[1:n].=da[levels];bb[1:n].=db[levels]
  native=evaluate(pl,surf,vp,dp,T,aa,bb)
  write_stats(io,hour,levels,native)
 end
end

function measure_day(spectral_dir, date, out; seconds_limit=3300)
 mkpath(out)
 start=time();paths=S.era5_spectral_paths(spectral_dir,date)
 ab=TOML.parsefile(joinpath(pkgdir(AtmosTransport),"config/era5_L137_coefficients.toml"))["coefficients"]
 da=diff(ab["a"]);db=diff(ab["b"])
 pl=plans(639);@info "LL plans ready" elapsed=time()-start
 open(joinpath(out,"ll_day.csv"),"w") do io
  println(io,CSV_HEADER)
  S.with_spectral_hours(paths) do readhour
   for hour in 0:23
    endpoint=readhour(hour);T=endpoint.T
    T==639 || error("This quadrature diagnostic requires T639 ERA5")
    surf=surfaces(pl,endpoint.lnsp,T)
    selected=[1,70,137]
    result=evaluate(pl,surf,endpoint.vo[:,:,selected],endpoint.d[:,:,selected],T,da[selected],db[selected])
    write_stats(io,hour,selected,result)
    if hour==1
     @info "Review meridian cross-check" default_kg_s=result[1][1][81,121,3] converged_kg_s=result[1][2][81,121,3]
    end
    @info "LL day hour complete" hour elapsed=time()-start
    if hour==0
     # Convergence check: double Gauss nodes and Fourier product grid.
     check=plans(T,32,11520);sf=surfaces(check,endpoint.lnsp,T)
     refined=evaluate(check,sf,endpoint.vo[:,:,selected],endpoint.d[:,:,selected],T,da[selected],db[selected])
     for i in 1:2
      difference=refined[i][2]-result[i][2]
      @info "Reference convergence" component=i relative_l2=norm(difference)/norm(refined[i][2]) max_abs_kg_s=maximum(abs,difference)
     end
     check=nothing;sf=nothing;refined=nothing;GC.gc()
     open(joinpath(out,"ll_hour00_all_levels.csv"),"w") do allio
      println(allio,CSV_HEADER)
      write_native_statistics(allio,0,pl,surf,endpoint,da,db)
     end
    end
    endpoint=nothing;surf=nothing;GC.gc()
    if time()-start>seconds_limit
     @warn "CPU-time budget reached" elapsed=time()-start
     break
    end
   end
  end
 end
 @info "LL measurement complete" elapsed=time()-start peak_gb=Sys.maxrss()/1e9
end

function measure_harmonics(out)
 mkpath(out)
 open(joinpath(out,"trapezoid_modes.csv"),"w") do io
  println(io,"T,m,nstep,r,spacing_degrees,samples_per_wavelength,measured_error_percent,analytic_error_percent")
  for T in (255,639), m in (1,(T-2)÷4,(T-2)÷2,T-2), nstep in (2,3,5,8)
   L=pi/180;r=ceil(Int,nstep*2(T+1)/360);h=L/r;x=m*h/2
   exact=sin(m*L)/m
   trap=h*sum((i==0 || i==r ? .5 : 1.)*cos(m*h*i) for i in 0:r)
   measured=trap/exact-1;analytic=x*cot(x)-1
   @assert abs(measured-analytic)<1e-12
   println(io,join((T,m,nstep,r,1/r,360r/m,100measured,100analytic),','))
  end
 end
end

function measure_native_day(spectral_dir,date,out; seconds_limit=2600)
 start=time();T=639
 mkpath(out)
 ab=TOML.parsefile(joinpath(pkgdir(AtmosTransport),"config/era5_L137_coefficients.toml"))["coefficients"]
 da=diff(ab["a"]);db=diff(ab["b"])
 pl=plans(T)
 paths=S.era5_spectral_paths(spectral_dir,date)
 open(joinpath(out,"ll_day_all_levels.csv"),"w") do io
  println(io,CSV_HEADER)
  S.with_spectral_hours(paths) do readhour
   for hour in 0:23
    endpoint=readhour(hour)
    endpoint.T==T || error("This quadrature diagnostic requires T639 ERA5")
    surf=surfaces(pl,endpoint.lnsp,T)
    write_native_statistics(io,hour,pl,surf,endpoint,da,db)
    @info "Full native day progress" hour elapsed=time()-start
    flush(stderr)
    endpoint=nothing;surf=nothing;GC.gc()
    if time()-start>seconds_limit
     @warn "CPU-time budget reached" elapsed=time()-start
     break
    end
   end
  end
 end
 @info "Native day measurement finished" elapsed=time()-start peak_gb=Sys.maxrss()/1e9
end

if length(ARGS)==2 && ARGS[1]=="harmonics"
 measure_harmonics(ARGS[2])
elseif length(ARGS)==4 && ARGS[1]=="native-day"
 measure_native_day(ARGS[2],Date(ARGS[3]),ARGS[4])
elseif length(ARGS)==4 && ARGS[1]=="day"
 measure_day(ARGS[2],Date(ARGS[3]),ARGS[4])
else
 error("Usage: tm5_default_quadrature.jl harmonics OUT | day/native-day SPECTRAL_DIR YYYY-MM-DD OUT")
end
