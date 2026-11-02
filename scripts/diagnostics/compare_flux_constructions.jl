#!/usr/bin/env julia
#=
Real-day flux-construction diagnostics. Run from the repository root:

    julia --project=. scripts/diagnostics/compare_flux_constructions.jl binary REF.bin TM5.bin OUT [PS_END.nc]
    julia --project=. scripts/diagnostics/compare_flux_constructions.jl factorial D0.nc D1.nc T0.nc T1.nc OUT
    julia --project=. -t8 scripts/diagnostics/compare_flux_constructions.jl stages PRE.toml YYYY-MM-DD OUT [HOURS]
    julia --project=. -t8 scripts/diagnostics/compare_flux_constructions.jl humidity N320_ROOT YYYY-MM-DD OUT

OUT is a directory. Differences are candidate minus reference; factorial inputs
are default flux/default PBL, default flux/TM5 PBL, TM5 flux/default PBL,
TM5 flux/TM5 PBL. CSVs contain SI mass flux rates and native dry VMRs.
No production methods are replaced. `stages` calls the production endpoint,
layer-balance and column-balance functions with alternative inputs (R0–R4).
It accepts a comma-separated list of UTC window starts, default all 24.
`humidity` exports the production conservative N320 humidity remap to the
thermo NetCDF schema on the same 360×180 target (q only, no invented T).
CPU only; use the canonical preprocessing/runtime CLIs for full-day binaries
and transport. See docs/memos/2026-11-01_tm5_flux_real_day.md.
=#
using AtmosTransport, NCDatasets, Dates, Statistics, Printf, TOML, GRIB
using AtmosTransport.Parameters: STANDARD_GRAVITY, ERA5_NATIVE_LEVEL_COUNT
const MD = AtmosTransport.MetDrivers
const PP = AtmosTransport.Preprocessing
const LL = PP.LatLonTarget
const SP = PP.Spectral
const G = AtmosTransport.Grids

const REGIONS = ("global", "tropics", "midlatitudes", "north_cap", "south_cap")
inregion(lat, r) = r == "global" || (r == "tropics" && abs(lat) <= 30) ||
    (r == "midlatitudes" && 30 < abs(lat) <= 80) ||
    (r == "north_cap" && lat > 80) || (r == "south_cap" && lat < -80)

mutable struct Moments
    weight::Float64
    d::Float64
    d2::Float64
    r2::Float64
    maxd::Float64
end
Moments() = Moments(0,0,0,0,0)
function add!(a::Moments, r, c, w)
    all(isfinite, (r,c,w)) || error("nonfinite diagnostic input")
    d = c-r
    a.weight += w; a.d += w*d; a.d2 += w*d*d; a.r2 += w*r*r
    a.maxd = max(a.maxd,abs(d))
end
ratio(a,b) = b == 0 ? NaN : a/b
function csvrow(io, xs...)
    println(io, join(xs, ','))
end

# Cell fields use area weights; face rates use equal weight per unique face.
# Zonal means use one row per native ring and vertical level.
function accumulate!(acc, zon, tag, ref, cand, lat, weights, rings)
    size(ref) == size(cand) || error("shape mismatch for $tag")
    n = length(lat)
    length(ref) % n == 0 || error("geometry mismatch for $tag")
    R = reshape(ref,n,:); C = reshape(cand,n,:)
    for region in REGIONS
        indices=findall(x->inregion(x,region),lat)
        isempty(indices) && continue
        for k in axes(R,2)
            a=get!(Moments,acc,(tag,region,k))
            for i in indices
                add!(a,R[i,k],C[i,k],weights[i])
            end
        end
    end
    bins=Dict{Float64,Vector{Int}}()
    for i in 1:n; push!(get!(()->Int[],bins,Float64(rings[i])),i); end
    for (ring,indices) in bins, k in axes(R,2)
        a=get!(Moments,zon,(tag,ring,k))
        for i in indices
            add!(a,R[i,k],C[i,k],weights[i])
        end
    end
end
function write_moments(path,acc; zonal=false)
    open(path,"w") do io
        csvrow(io,"field",zonal ? "latitude_deg" : "region","level","mean_difference",
            "rms_difference","rms_reference","relative_rms","max_abs_difference","weight")
        for (key,a) in sort!(collect(acc);by=first)
            rd=sqrt(a.d2/a.weight); rr=sqrt(a.r2/a.weight)
            csvrow(io,key...,a.d/a.weight,rd,rr,ratio(rd,rr),a.maxd,a.weight)
        end
    end
end
function summary(acc)
    out=Dict{Tuple{String,String,Int},Moments}()
    for ((tag,region,k),a) in acc
        b=get!(Moments,out,(tag,region,0))
        b.weight+=a.weight; b.d+=a.d; b.d2+=a.d2; b.r2+=a.r2; b.maxd=max(b.maxd,a.maxd)
    end
    return out
end

function write_pressures(path,A,B)
    open(path,"w") do io
        csvrow(io,"level","mid_pressure_at_ps1000_hPa","interface_pressure_at_ps1000_hPa")
        p=(A .+ 1e5.*B)./100
        for k in 1:length(A)-1; csvrow(io,k,(p[k]+p[k+1])/2,p[k]); end
    end
end

function geometry(h)
    g=h.geometry
    if g isa MD.LatLonBinaryGeometry
        nx,ny=g.Nx,g.Ny
        mesh=LatLonMesh(;size=(nx,ny),longitude=g.longitude_interval,
            latitude=g.latitude_interval,radius=h.planet_radius_m)
        lat=vec(repeat(reshape(g.latitudes,1,:),nx,1))
        area=vec(Float64.(G.cell_areas(mesh)))
        alat=copy(lat)  # exclude east periodic mirror below
        blat=vec(repeat(reshape(mesh.φᶠ,1,:),nx,1))
        return (;lat,area,rings=lat,face=Dict(:am=>(alat,ones(length(alat)),alat),
            :bm=>(blat,ones(length(blat)),blat)),mesh)
    else
        mesh=ReducedGaussianMesh(g.latitudes,g.nlon_per_ring;FT=Float64,radius=h.planet_radius_m)
        lat=reduce(vcat,[fill(g.latitudes[j],g.nlon_per_ring[j]) for j in eachindex(g.latitudes)])
        fl=map(1:G.nfaces(mesh)) do f
            l,r=G.face_cells(mesh,f)
            l==0 ? -90.0 : r==0 ? 90.0 : (lat[l]+lat[r])/2
        end
        return (;lat,area=vec(Float64.(G.cell_areas(mesh))),rings=lat,
            face=Dict(:horizontal_flux=>(fl,ones(length(fl)),fl)),mesh)
    end
end

function mass_delta(reader,w,m,convergence,cm,seconds)
    dm=similar(m)
    if :dm in reader.header.payload_sections
        MD.BinaryReaders._load_sections!((;dm),reader,w)
        return (;dm,method="explicit_dm")
    elseif w<reader.header.nwindow
        next=similar(m)
        MD.BinaryReaders._load_sections!((;m=next),reader,w+1)
        dm .= next .- m
        return (;dm,method="stored_endpoints")
    else
        # The RG layout omits the final mass endpoint as well as dm.
        # This is its encoded transport budget, not an independent replay check.
        dm .= reader.header.dt_met_seconds .* (convergence .- diff(cm./seconds;dims=ndims(cm)))
        return (;dm,method="final_flux_budget")
    end
end

function binary_compare(refpath,candpath,out;pressure_end="")
    mkpath(out)
    r=TransportBinaryReader(refpath;FT=Float64); c=TransportBinaryReader(candpath;FT=Float64)
    h=r.header; hc=c.header
    h.nwindow==hc.nwindow && h.nlevel==hc.nlevel && h.mass_basis==hc.mass_basis || error("incompatible binaries")
    h.dt_met_seconds==hc.dt_met_seconds && get(h.raw_header,"date",nothing)==get(hc.raw_header,"date",nothing) ||
        error("binary dates or window durations differ")
    h.mass_basis==:dry || error("this comparison expects dry carrier mass")
    h.A_ifc==hc.A_ifc && h.B_ifc==hc.B_ifc || error("vertical coordinates differ")
    geo=geometry(h); gc=geometry(hc)
    geo.lat==gc.lat && geo.area==gc.area || error("horizontal coordinates differ")
    acc=Dict{Tuple{String,String,Int},Moments}(); zon=Dict{Tuple{String,Float64,Int},Moments}()
    previous_ps=nothing
    end_ps=nothing
    if !isempty(pressure_end)
        end_ps=NCDataset(pressure_end) do ds
            h.geometry isa MD.LatLonBinaryGeometry || error("pressure endpoint currently requires lat-lon")
            Array(ds["lat"])==h.geometry.latitudes || error("pressure endpoint latitude mismatch")
            Array(ds["lon"])==h.geometry.longitudes || error("pressure endpoint longitude mismatch")
            string(ds.attrib["source_day"])==string(h.raw_header["date"])==string(hc.raw_header["date"]) ||
                error("pressure endpoint source day mismatch")
            reference=Array(ds["point_ps_end"]); candidate=Array(ds["tm5_ps_end"])
            h.on_disk_float_type===:Float32 && (reference=Float64.(Float32.(reference)))
            hc.on_disk_float_type===:Float32 && (candidate=Float64.(Float32.(candidate)))
            (;reference,candidate)
        end
    end
    open(joinpath(out,"windows.csv"),"w") do io
        csvrow(io,"window","steps_reference","steps_candidate","dry_mass_ref_kg","dry_mass_candidate_kg",
            "mass_tendency_ref_kg_s","mass_tendency_candidate_kg_s","mass_delta_reference","mass_delta_candidate")
        for w in 1:h.nwindow
            mr,pr,fr=MD.load_window!(r,w); mc,pc,fc=MD.load_window!(c,w)
            sr=MD.flux_application_seconds(h.dt_met_seconds,h.steps_per_window_by_window[w],h.flux_kind)
            sc=MD.flux_application_seconds(hc.dt_met_seconds,hc.steps_per_window_by_window[w],hc.flux_kind)
            for (name,R,C) in (("m",mr,mc),("ps",pr,pc),("cm",fr.cm/sr,fc.cm/sc))
                accumulate!(acc,zon,name,R,C,geo.lat,geo.area,geo.rings)
            end
            if previous_ps !== nothing
                accumulate!(acc,zon,"ps_tendency",(pr-previous_ps[1])/h.dt_met_seconds,
                    (pc-previous_ps[2])/hc.dt_met_seconds,geo.lat,geo.area,geo.rings)
            end
            previous_ps=(copy(pr),copy(pc))
            if w==h.nwindow && end_ps !== nothing
                size(end_ps.reference)==size(pr)==size(end_ps.candidate) || error("pressure endpoint shape mismatch")
                accumulate!(acc,zon,"ps_tendency",(end_ps.reference-pr)/h.dt_met_seconds,
                    (end_ps.candidate-pc)/hc.dt_met_seconds,geo.lat,geo.area,geo.rings)
            end
            convr=zeros(size(mr)); convc=zeros(size(mc))
            for (v,(lat,weight,rings)) in geo.face
                R=getproperty(fr,v)/sr; C=getproperty(fc,v)/sc
                if v===:am
                    convr .+= R[1:end-1,:,:].-R[2:end,:,:]
                    convc .+= C[1:end-1,:,:].-C[2:end,:,:]
                    R=R[1:end-1,:,:]; C=C[1:end-1,:,:]
                elseif v===:bm
                    convr .+= R[:,1:end-1,:].-R[:,2:end,:]
                    convc .+= C[:,1:end-1,:].-C[:,2:end,:]
                else
                    for f in axes(R,1)
                        l,rr=G.face_cells(geo.mesh,f)
                        l>0 && (convr[l,:] .-= R[f,:]; convc[l,:] .-= C[f,:])
                        rr>0 && (convr[rr,:] .+= R[f,:]; convc[rr,:] .+= C[f,:])
                    end
                end
                accumulate!(acc,zon,String(v),R,C,lat,weight,rings)
                accumulate!(acc,zon,String(v)*"_column",sum(R;dims=ndims(R)),sum(C;dims=ndims(C)),lat,weight,rings)
            end
            accumulate!(acc,zon,"horizontal_convergence",convr,convc,geo.lat,geo.area,geo.rings)
            dr=mass_delta(r,w,mr,convr,fr.cm,sr); dc=mass_delta(c,w,mc,convc,fc.cm,sc)
            accumulate!(acc,zon,"mass_tendency",dr.dm/h.dt_met_seconds,dc.dm/hc.dt_met_seconds,geo.lat,geo.area,geo.rings)
            accumulate!(acc,zon,"ps_dry_tendency",sum(dr.dm;dims=ndims(dr.dm)).*STANDARD_GRAVITY./reshape(geo.area,size(pr))./h.dt_met_seconds,
                sum(dc.dm;dims=ndims(dc.dm)).*STANDARD_GRAVITY./reshape(geo.area,size(pc))./hc.dt_met_seconds,geo.lat,geo.area,geo.rings)
            csvrow(io,w,h.steps_per_window_by_window[w],hc.steps_per_window_by_window[w],sum(mr),sum(mc),sum(dr.dm)/h.dt_met_seconds,sum(dc.dm)/hc.dt_met_seconds,dr.method,dc.method)
        end
    end
    write_moments(joinpath(out,"levels.csv"),acc)
    write_moments(joinpath(out,"summary.csv"),summary(acc))
    write_moments(joinpath(out,"zonal.csv"),zon;zonal=true)
    write_pressures(joinpath(out,"levels_pressure.csv"),h.A_ifc,h.B_ifc)
    for (name,head) in (("reference",h),("candidate",hc))
        open(joinpath(out,name*"_header.toml"),"w") do io
            # Keep the schedule and tendency coverage beside the numerical tables.
            TOML.print(io,Dict("binary"=>name=="reference" ? refpath : candpath,
                "source_flux_sampling"=>string(head.source_flux_sampling),
                "steps"=>head.steps_per_window_by_window,"mass_basis"=>string(head.mass_basis),
                "ps_tendency_windows"=>h.nwindow-(end_ps===nothing ? 1 : 0),
                "pressure_endpoint"=>pressure_end))
        end
    end
end

function ncgeometry(ds)
    if haskey(ds.dim,"cell")
        return vec(Array(ds["cell_lat"])),vec(Array(ds["cell_area"]))
    end
    area=Array(ds["cell_area"])
    return vec(repeat(reshape(Array(ds["lat"]),1,:),size(area,1),1)),vec(area)
end
function ncfetch(ds,name,t)
    v=ds[name]
    return Array(v[ntuple(_->Colon(),ndims(v)-1)...,t])
end
function factorial(paths,out)
    length(paths) in (2,4) || error("expected two runs or a four-run factorial")
    mkpath(out)
    ds=map(NCDataset,paths)
    try
        lat,area=ncgeometry(ds[1]); nz=ds[1].dim["lev"]; nt=ds[1].dim["time"]
        all(d->ncgeometry(d)==(lat,area) && d.dim["lev"]==nz && Array(d["time"])==Array(ds[1]["time"]),ds) || error("run grids or times differ")
        acc=Dict{Tuple{String,String,Int},Moments}(); zon=Dict{Tuple{String,Float64,Int},Moments}()
        initial=Dict{String,Any}()
        effects=length(paths)==2 ? (("flux_default_pbl",1,2),) :
            (("flux_default_pbl",1,3),("flux_tm5_pbl",2,4),
             ("pbl_default_flux",1,2),("pbl_tm5_flux",3,4),("interaction",0,0))
        open(joinpath(out,"tracers.csv"),"w") do io
            csvrow(io,"effect","tracer","field","region","snapshot","elapsed_hours","mean_difference","rms_difference","rms_reference","relative_rms","max_abs_difference",
                "initial_rms_difference","rms_increment_difference","rms_reference_increment")
            for tr in ("uniform","co2","fossil","rn222","blob"), t in 1:nt
                x=map(d->reshape(Float64.(ncfetch(d,tr,t)),length(lat),nz),ds)
                suffix=haskey(ds[1].dim,"cell") ? "_column_mean_native" : "_column_mean"
                col=map(d->Float64.(ncfetch(d,tr*suffix,t)),ds)
                t==1 && (initial[tr]=(;x,col))
                x0=initial[tr].x; col0=initial[tr].col
                for (effect,a,b) in effects
                    R=a==0 ? x[2]-x[1] : x[a]
                    C=a==0 ? x[4]-x[3] : x[b]
                    rc=a==0 ? col[2]-col[1] : col[a]
                    cc=a==0 ? col[4]-col[3] : col[b]
                    R0=a==0 ? x0[2]-x0[1] : x0[a]
                    C0=a==0 ? x0[4]-x0[3] : x0[b]
                    rc0=a==0 ? col0[2]-col0[1] : col0[a]
                    cc0=a==0 ? col0[4]-col0[3] : col0[b]
                    if t==nt
                        accumulate!(acc,zon,effect*"/"*tr,R,C,lat,area,lat)
                        accumulate!(acc,zon,effect*"/"*tr*"/increment",R-R0,C-C0,lat,area,lat)
                    end
                    for (field,r,c,r0,c0) in (("surface",view(R,:,nz),view(C,:,nz),view(R0,:,nz),view(C0,:,nz)),
                        ("column",vec(rc),vec(cc),vec(rc0),vec(cc0))), region in REGIONS
                        z=Moments(); zi=Moments(); increment=Moments()
                        for i in eachindex(lat)
                            if inregion(lat[i],region)
                                add!(z,r[i],c[i],area[i])
                                add!(zi,r0[i],c0[i],area[i])
                                add!(increment,r[i]-r0[i],c[i]-c0[i],area[i])
                            end
                        end
                        z.weight==0 && continue
                        rd=sqrt(z.d2/z.weight); rr=sqrt(z.r2/z.weight)
                        times=Array(ds[1]["time"])
                        hours=Dates.value(times[t]-times[1])/3.6e6
                        csvrow(io,effect,tr,field,region,t,hours,z.d/z.weight,rd,rr,ratio(rd,rr),z.maxd,
                            sqrt(zi.d2/zi.weight),sqrt(increment.d2/increment.weight),sqrt(increment.r2/increment.weight))
                    end
                end
            end
        end
        write_moments(joinpath(out,"final_levels.csv"),acc)
        write_moments(joinpath(out,"final_zonal.csv"),zon;zonal=true)
        open(joinpath(out,"budgets.csv"),"w") do io
            csvrow(io,"run","tracer","initial_mass_kg","final_mass_kg","fractional_mass_change","uniform_max_relative_drift")
            for j in eachindex(ds), tr in ("uniform","co2","fossil","rn222","blob")
                mass=Array(ds[j][tr*"_total_mass"])
                dev=tr=="uniform" ? maximum(maximum(abs,ncfetch(ds[j],tr,t).-4e-4)/4e-4 for t in 1:nt) : NaN
                csvrow(io,j,tr,mass[1],mass[end],ratio(mass[end]-mass[1],mass[1]),dev)
            end
        end
    finally
        foreach(close,ds)
    end
end

function humidity(root,date,out;records=24)
    mkpath(out)
    grid=PP.Common.build_target_geometry(Dict("type"=>"latlon","nlon"=>360,"nlat"=>180),Float64)
    remap=PP.ERA5.open_era5_humidity_remap(root,date,grid.mesh;
        cache_dir=joinpath(out,"regrid_cache"))
    path=joinpath(out,"era5_thermo_ml_$(Dates.format(date,"yyyymmdd")).nc")
    try
        NCDataset(path,"c") do ds
            defDim(ds,"longitude",360); defDim(ds,"latitude",180); defDim(ds,"hybrid",ERA5_NATIVE_LEVEL_COUNT); defDim(ds,"time",records)
            defVar(ds,"longitude",Float64,("longitude",))[:]=grid.lons
            defVar(ds,"latitude",Float64,("latitude",))[:]=grid.lats
            defVar(ds,"hybrid",Int32,("hybrid",))[:]=1:ERA5_NATIVE_LEVEL_COUNT
            defVar(ds,"time",Float64,("time",);attrib=Dict("units"=>"hours since $date 00:00:00","calendar"=>"standard"))[:]=0:records-1
            q=defVar(ds,"q",Float32,("longitude","latitude","hybrid","time");
                deflatelevel=1,attrib=Dict("units"=>"kg/kg"))
            for hour in 0:records-1
                x=PP.ERA5.remap_era5_humidity!(remap,hour)
                all(isfinite,x) || error("nonfinite remapped humidity")
                q[:,:,:,hour+1]=reshape(Float32.(x),360,180,ERA5_NATIVE_LEVEL_COUNT)
                @info "N320 humidity export" date hour extrema=extrema(x)
            end
        end
    finally
        close(remap)
    end
    return path
end

"Copy packed spectral messages unchanged; q uses the production conservative remap."
function native_inputs(root,date,out;records=24)
    stamp=Dates.format(date,"yyyymmdd")
    source=joinpath(root,"ml_an_native_core","era5_core_$stamp.grib")
    spectral=joinpath(out,"spectral"); mkpath(spectral)
    vo=joinpath(spectral,"era5_spectral_$(stamp)_vo_d.gb")
    ln=joinpath(spectral,"era5_spectral_$(stamp)_lnsp.gb")
    !ispath(vo) && !ispath(ln) || error("spectral output exists; use a fresh destination")
    counts=Dict("vo"=>0,"d"=>0,"lnsp"=>0)
    GRIB.GribFile(source) do f
        for msg in f
            try
                name=String(msg["shortName"])
                haskey(counts,name) && Int(msg["dataTime"])<100records || continue
                GRIB.writemessage(msg,name=="lnsp" ? ln : vo;mode="ab")
                counts[name]+=1
            finally
                finalize(msg)
            end
        end
    end
    counts==Dict("vo"=>ERA5_NATIVE_LEVEL_COUNT*records,"d"=>ERA5_NATIVE_LEVEL_COUNT*records,"lnsp"=>records) || error("incomplete native spectra: $counts")
    open(joinpath(out,"native_$stamp.toml"),"w") do io
        TOML.print(io,Dict("source"=>source,"source_bytes"=>filesize(source),"records"=>records,
            "counts"=>counts,"humidity_method"=>"production conservative N320 to 360x180; Float32 NetCDF"))
    end
    return humidity(root,date,joinpath(out,"thermo");records)
end

"Export real T/q at target points; record and zero negative interpolated humidity."
function thermo_inputs(source,date,out;records=24)
    mkpath(out)
    grid=PP.Common.build_target_geometry(Dict("type"=>"latlon","nlon"=>360,"nlat"=>180),Float64)
    path=SP.era5_thermo_path(source,date)
    lons,lats=SP.read_thermo_coordinates(path)
    weights=LL.LatLonBilinearWeights(lons,lats,grid.lons,grid.lats)
    dest=SP.era5_thermo_path(out,date)
    ispath(dest) && error("thermo destination already exists")
    open(joinpath(out,"quality_$(Dates.format(date,"yyyymmdd")).csv"),"w") do log
        csvrow(log,"hour","negative_count","min_q_before","max_q_before","sum_abs_q_adjustment")
        NCDataset(dest,"c") do ds
            defDim(ds,"longitude",360); defDim(ds,"latitude",180); defDim(ds,"hybrid",ERA5_NATIVE_LEVEL_COUNT); defDim(ds,"time",records)
            defVar(ds,"longitude",Float64,("longitude",))[:]=grid.lons
            defVar(ds,"latitude",Float64,("latitude",))[:]=grid.lats
            defVar(ds,"hybrid",Int32,("hybrid",))[:]=1:ERA5_NATIVE_LEVEL_COUNT
            defVar(ds,"time",Float64,("time",);attrib=Dict("units"=>"hours since $date 00:00:00","calendar"=>"standard"))[:]=0:records-1
            t=defVar(ds,"t",Float32,("longitude","latitude","hybrid","time");deflatelevel=1,attrib=Dict("units"=>"K"))
            q=defVar(ds,"q",Float32,("longitude","latitude","hybrid","time");deflatelevel=1,attrib=Dict("units"=>"kg/kg"))
            dst=zeros(Float64,360,180,ERA5_NATIVE_LEVEL_COUNT)
            for hour in 0:records-1
                for (name,var) in (("t",t),("q",q))
                    src=SP.read_qv_from_thermo(path,hour+1,length(lons),length(lats),ERA5_NATIVE_LEVEL_COUNT;variable=name,FT=Float64)
                    LL.interpolate_ll_qv!(dst,src,weights)
                    if name=="q"
                        all(isfinite,dst) && maximum(dst)<1 || error("invalid input humidity")
                        csvrow(log,hour,count(<(0),dst),minimum(dst),maximum(dst),-sum(x->min(x,0),dst))
                        dst .= max.(dst,0)
                    end
                    var[:,:,:,hour+1]=Float32.(dst)
                end
                @info "Thermo export" date hour
            end
        end
    end
    return dest
end

# The retained-spectrum point rung gathers an oversampled FFT at the target
# points. A target-sized FFT would silently drop m>179 on the 1-degree mesh.
function point_plans(grid,T,vertical)
    nx,ny=length(grid.lons),length(grid.lats)
    J=nx*ceil(Int,(2T+3)/nx)
    rings=SP.SynthesisRings(deg2rad.(grid.lats),fill(J,ny),fill(deg2rad(first(grid.mesh.λᶠ)),ny))
    synth=SP.CPURingSynthesis(rings,T,8)
    scalar=SP.surface_synthesis(synth,1)
    return (;synth,scalar,J,transform=LL.allocate_transform_workspace(grid,min(T,nx÷2-1),vertical.Nz_native),
        merged=LL.allocate_merge_workspace(grid,vertical.Nz_native,vertical.Nz,Float64))
end

function point_endpoint(h,qv,grid,vertical,settings,p;retained=false)
    nx,ny=length(grid.lons),length(grid.lats); nz=vertical.Nz_native
    tr=p.transform
    if !retained
        LL.spectral_to_native_fields!(tr,h.lnsp,h.vo,h.d,min(h.T,nx÷2-1),vertical,grid,settings)
    else
        T=h.T; stride=p.J÷nx
        phase=reshape([cis(m*deg2rad(grid.dlon/2)) for m in 0:T],1,T+1,1)
        H=reshape(h.lnsp,T+1,T+1,1).*phase
        samples=reshape(SP.synthesize_rings(H,p.scalar),p.J,ny)
        tr.sp .= exp.(samples[1:stride:end,:])
        for lo in 1:8:nz
            levels=lo:min(lo+7,nz); nb=length(levels)
            U=zeros(ComplexF64,T+1,T+1,8); V=similar(U); fill!(V,0)
            for (b,k) in enumerate(levels)
                u=zeros(ComplexF64,T+1,T+1); v=similar(u)
                SP.vod2uv!(u,v,view(h.vo,:,:,k),view(h.d,:,:,k),T)
                U[:,:,b].=u; V[:,:,b].=v
            end
            us=reshape(SP.synthesize_rings(U,p.synth),p.J,ny,8)
            vs=reshape(SP.synthesize_rings(V.*phase,p.synth),p.J,ny,8)
            tr.u_cc[:,:,levels].=us[1:stride:end,:,1:nb]
            tr.v_cc[:,:,levels].=vs[1:stride:end,:,1:nb]
        end
        SP.stagger_winds!(tr.u_stag,tr.v_stag,tr.u_cc,tr.v_cc,nx,ny,nz)
        SP.compute_dp!(tr.dp,tr.sp,vertical.ab.dA,vertical.ab.dB,nx,ny,nz)
        SP.compute_air_mass!(tr.m_arr,tr.dp,grid.area,nx,ny,nz)
        SP.compute_mass_fluxes!(tr.am_arr,tr.bm_arr,tr.u_stag,tr.v_stag,tr.sp,
            vertical.ab.dA,vertical.ab.dB,grid,settings.half_dt,nz)
    end
    LL.apply_mass_fix_if_needed!(qv,tr,grid,vertical,settings,zeros(1),1)
    LL.apply_dry_basis_if_needed!(settings.mass_basis,tr,qv)
    LL.merge_native_window!(p.merged,tr,qv,vertical,settings)
    return (;m=reshape(copy(p.merged.m_merged),nx*ny,vertical.Nz),
        ps=vec(copy(tr.sp)),am=copy(p.merged.am_merged)/settings.half_dt,
        bm=copy(p.merged.bm_merged)/settings.half_dt)
end

# stages keeps the binary/runtime analysis usable without a spectral workspace.
function stages(config,date,out,hours)
    mkpath(out)
    cfg=TOML.parsefile(config)
    plan=PP.validate_preprocessing_config(cfg)
    grid=PP.Common.build_target_geometry(cfg["grid"],Float64)
    grid isa PP.Common.LatLonTargetGeometry || error("stages currently requires lat-lon")
    settings=plan.settings; vertical=plan.vertical
    write_pressures(joinpath(out,"levels_pressure.csv"),vertical.merged_vc.A,vertical.merged_vc.B)
    c=PP.Common.TM5SpectralFluxes()
    ep=SP.TM5EndpointPlan(vertical)
    paths=SP.era5_spectral_paths(settings.spectral_dir,date)
    qv=LL.allocate_qv_workspace(grid,settings,date,vertical.Nz_native,vertical.Nz,Float64)
    solver=SP.LLPoissonWorkspace(length(grid.lons),length(grid.lats))
    acc=Dict{Tuple{String,String,Int},Moments}(); zon=Dict{Tuple{String,Float64,Int},Moments}()
    corrections=NamedTuple[]
    lat=vec(repeat(reshape(grid.lats,1,:),length(grid.lons),1)); area=vec(grid.area)
    SP.with_spectral_hours(paths) do read_hour
        initial=read_hour(first(hours))
        ws=SP.SpectralIntegralWorkspace(grid,initial.T,c;synthesis=PP.Common.CPUSynthesis(),batch=8)
        point=point_plans(grid,initial.T,vertical)
        function endpoint(hour)
            h=hour==24 ? SP.read_hour0_spectral(settings.spectral_dir,date+Day(1)) : read_hour(hour)
            if hour==24
                LL.read_next_day_qv!(qv,date,settings,length(grid.lons),length(grid.lats),vertical.Nz_native)
            else
                LL.read_window_qv!(qv,hour+1,length(grid.lons),length(grid.lats),vertical.Nz_native)
            end
            p0=point_endpoint(h,qv,grid,vertical,settings,point)
            p1=point_endpoint(h,qv,grid,vertical,settings,point;retained=true)
            tm5=SP.construct_tm5_endpoint(ws,h.lnsp,h.vo,h.d,h.T,qv.qv_native,ep,settings)
            if hour==24
                NCDataset(joinpath(out,"ps_end.nc"),"c") do ds
                    ds.attrib["source_day"]=string(date)
                    ds.attrib["endpoint_time"]=string(date+Day(1))*"T00:00:00"
                    ds.attrib["config"]=abspath(config)
                    defDim(ds,"lon",length(grid.lons)); defDim(ds,"lat",length(grid.lats))
                    defVar(ds,"lon",Float64,("lon",))[:]=grid.lons
                    defVar(ds,"lat",Float64,("lat",))[:]=grid.lats
                    defVar(ds,"point_ps_end",Float64,("lon","lat");attrib=Dict("units"=>"Pa"))[:,:]=reshape(p0.ps,size(grid.area))
                    defVar(ds,"tm5_ps_end",Float64,("lon","lat");attrib=Dict("units"=>"Pa"))[:,:]=reshape(tm5.ps,size(grid.area))
                end
            end
            return (;tm5,p0,p1)
        end
        initial=nothing
        open(joinpath(out,"balance.csv"),"w") do io
            csvrow(io,"hour","endpoint","pre_rms","pre_max","post_rms","post_max","passes","max_layer_residual_kg_s")
            a=nothing; previous=-2
            for hour in hours
                aa=previous==hour-1 ? a : endpoint(hour)
                bb=endpoint(hour+1)
                a=aa.tm5; b=bb.tm5
                win=LL.balance_window_layers!(c,a,b,grid,solver,settings.met_interval,area)
                # R2: independently dry the raw endpoint rates, then window mean.
                u0=copy(a.u); v0=copy(a.v); u1=copy(b.u); v1=copy(b.v)
                LL.dry_target_fluxes!(u0,v0,a.q,grid); LL.dry_target_fluxes!(u1,v1,b.q,grid)
                rawu=(u0+u1)/2; rawv=(v0+v1)/2
                for (idx,d) in enumerate(win.diagnostics)
                    csvrow(io,hour,idx,d.pre.rms,d.pre.maximum,d.post.rms,d.post.maximum,d.passes,d.max_layer_residual)
                end
                for (tag,ru,rv,cu,cv) in (("layer_balance",rawu,rawv,win.u,win.v),
                    ("sampling",win.diagnostics[1].u,win.diagnostics[1].v,win.u,win.v))
                    ra,rb=LL.directional_fluxes(ru,rv,grid); ca,cb=LL.directional_fluxes(cu,cv,grid)
                    # Shared column repair on each rung, with the identical dry mass target.
                    target=(reshape(win.m_next,size(grid.area)...,vertical.Nz)-reshape(win.m,size(grid.area)...,vertical.Nz))/settings.met_interval
                    for (side,(am,bm)) in enumerate(((ra,rb),(ca,cb)))
                        diag=SP.balance_column_mass_fluxes!(am,bm,reshape(win.m,size(target)),target,solver;
                            weights=LL.spectral_column_weights(grid,settings,vertical))
                        push!(corrections,(;hour,rung=tag*"_"*string(side),diag...))
                    end
                    for (v,R,C,lats) in (("am",ra[1:end-1,:,:],ca[1:end-1,:,:],lat),
                        ("bm",rb,cb,vec(repeat(reshape(grid.mesh.φᶠ,1,:),length(grid.lons),1))))
                        accumulate!(acc,zon,tag*"/"*v,R,C,lats,ones(length(lats)),lats)
                        accumulate!(acc,zon,tag*"/"*v*"_column",sum(R;dims=3),sum(C;dims=3),lats,ones(length(lats)),lats)
                    end
                    # Production explicit-dm closure, including its residual weights.
                    function cm(am,bm)
                        result=zeros(size(grid.area)...,vertical.Nz+1)
                        MD.recompute_cm_from_dm_target!(MD.structured_replay_layout(),zeros(size(target)),
                            result,reshape(win.m,size(target)),target,am,bm;
                            residual_weight=PP.MassClosure.ColumnResidualWeight(LL.spectral_column_weights(grid,settings,vertical)))
                        result[:,:,end].=0
                        return result
                    end
                    accumulate!(acc,zon,tag*"/cm",cm(ra,rb),cm(ca,cb),lat,area,lat)
                end
                # R0 and R1 retain the production point route's start sampling.
                function repaired(am,bm,m,mnext,rung,ps,psnext)
                    target=reshape((mnext-m)/settings.met_interval,size(grid.area)...,vertical.Nz)
                    mass=reshape(m,size(target))
                    diag=SP.balance_column_mass_fluxes!(am,bm,mass,target,solver;
                        weights=LL.spectral_column_weights(grid,settings,vertical))
                    push!(corrections,(;hour,rung,diag...))
                    cm=zeros(size(grid.area)...,vertical.Nz+1)
                    MD.recompute_cm_from_dm_target!(MD.structured_replay_layout(),zeros(size(target)),
                        cm,mass,target,am,bm;residual_weight=PP.MassClosure.ColumnResidualWeight(
                            LL.spectral_column_weights(grid,settings,vertical)))
                    cm[:,:,end].=0
                    conv=am[1:end-1,:,:]-am[2:end,:,:]+bm[:,1:end-1,:]-bm[:,2:end,:]
                    return (;am,bm,cm,m,ps,
                        horizontal_convergence=reshape(conv,size(m)),
                        ps_dry_tendency=vec(sum(target;dims=3)).*STANDARD_GRAVITY./area,
                        ps_tendency=(psnext-ps)/settings.met_interval)
                end
                r0=repaired(copy(aa.p0.am),copy(aa.p0.bm),aa.p0.m,bb.p0.m,"R0",aa.p0.ps,bb.p0.ps)
                r1=repaired(copy(aa.p1.am),copy(aa.p1.bm),aa.p1.m,bb.p1.m,"R1",aa.p1.ps,bb.p1.ps)
                rawam,rawbm=LL.directional_fluxes(rawu,rawv,grid)
                r2=repaired(rawam,rawbm,win.m,win.m_next,"R2",a.ps,b.ps)
                fullam,fullbm=LL.directional_fluxes(win.u,win.v,grid)
                r3=repaired(fullam,fullbm,win.m,win.m_next,"R3",a.ps,b.ps)
                for (tag,r,c) in (("retained_spectrum",r0,r1),("integrated_and_mean",r1,r2),("full_tm5",r0,r3))
                    for v in (:am,:bm,:cm,:m,:horizontal_convergence,:ps,:ps_dry_tendency,:ps_tendency)
                        R=getproperty(r,v); C=getproperty(c,v)
                        lats=v===:bm ? vec(repeat(reshape(grid.mesh.φᶠ,1,:),length(grid.lons),1)) : lat
                        if v===:am
                            R=R[1:end-1,:,:]; C=C[1:end-1,:,:]
                        end
                        weight=v in (:am,:bm) ? ones(length(lats)) : area
                        accumulate!(acc,zon,tag*"/"*String(v),R,C,lats,weight,lats)
                    end
                end
                a=bb; previous=hour
                flush(io)
                @info "Attribution window complete" date hour
                flush(stderr)
            end
        end
    end
    write_moments(joinpath(out,"levels.csv"),acc)
    write_moments(joinpath(out,"summary.csv"),summary(acc))
    write_moments(joinpath(out,"zonal.csv"),zon;zonal=true)
    open(joinpath(out,"column_balance.csv"),"w") do io
        csvrow(io,"hour","rung","max_pre_residual_kg_s","max_post_residual_kg_s","max_face_delta_kg_s")
        for row in corrections; csvrow(io,values(row)...); end
    end
end

function main(args=ARGS)
    if !isempty(args) && args[1] in ("-h","--help")
        println("Usage: compare_flux_constructions.jl binary REF CAND OUT [PS_END.nc] | factorial D0 D1 T0 T1 OUT | pair REF CAND OUT | stages CONFIG DAY OUT [HOURS] | humidity ROOT DAY OUT | native ROOT DAY OUT [RECORDS] | thermo SOURCE DAY OUT [RECORDS]")
        return
    end
    isempty(args) && error("expected a command; run with --help")
    cmd=args[1]
    if cmd=="binary" && length(args) in (4,5)
        binary_compare(args[2:4]...;pressure_end=length(args)==5 ? args[5] : "")
    elseif cmd=="factorial" && length(args)==6
        factorial(args[2:5],args[6])
    elseif cmd=="pair" && length(args)==4
        factorial(args[2:3],args[4])
    elseif cmd=="stages" && length(args) in (4,5)
        hours=length(args)==5 ? parse.(Int,split(args[5],',')) : collect(0:23)
        all(h->0<=h<=23,hours) && issorted(hours) && allunique(hours) || error("hours must be unique, sorted, in 0:23")
        stages(args[2],Date(args[3]),args[4],hours)
    elseif cmd=="humidity" && length(args)==4
        humidity(args[2],Date(args[3]),args[4])
    elseif cmd=="native" && length(args) in (4,5)
        records=length(args)==5 ? parse(Int,args[5]) : 24
        1<=records<=24 || error("records outside 1:24")
        native_inputs(args[2],Date(args[3]),args[4];records)
    elseif cmd=="thermo" && length(args) in (4,5)
        records=length(args)==5 ? parse(Int,args[5]) : 24
        1<=records<=24 || error("records outside 1:24")
        thermo_inputs(args[2],Date(args[3]),args[4];records)
    else
        error("invalid command or arguments; run with --help")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
