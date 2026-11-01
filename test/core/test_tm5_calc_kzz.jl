using Test, TOML, KernelAbstractions
using AtmosTransport
const TD = AtmosTransport.Operators.Diffusion
const TP = AtmosTransport.Parameters

function tm5_scratch(::Type{FT}, n) where FT
    return (; g=zeros(FT,n+1), pf=zeros(FT,n), theta=zeros(FT,n),
        theta_v=zeros(FT,n), z=zeros(FT,n), ri=zeros(FT,n-1),
        shear2=zeros(FT,n-1), k_free=zeros(FT,n-1), kvh=zeros(FT,n-1))
end
function synthetic_tm5_column(::Type{FT}=Float64; sshf=20, lsm=1, sr=0.1, fsr=0.1) where FT
    n = 8
    p = FT[100000, 98000, 95000, 90000, 80000, 60000, 30000, 10000, 0]
    q = fill(FT(0.005),n)
    th = FT[290, 292, 295, 300, 310, 330, 360, 420]
    c = TD.TM5BLDiffParameters(FT; lmax=6).constants
    t = th .* ((p[1:end-1] .+ p[2:end]) ./ FT(2) ./ c.p_ref).^(c.r_air / c.cp_air)
    surface = (; lsm=FT(lsm),sr=FT(sr),fsr=FT(fsr),u10=FT(5),v10=FT(2),
        ewss=FT(0.1),nsss=FT(0.05),sshf=FT(sshf),slhf=FT(-30),oro=FT(0))
    return (;p,p_geometry=p,t,q,mass=(p[1:end-1].-p[2:end]).*FT(1e8),
        u=FT.(1:n),v=zeros(FT,n),surface)
end

# Independent TM5 K:883–962 fallback, without the singular entrainment quotient.
function tm5_background_k(r,c,l)
    zi = r.g[l+1]-r.g[1]
    zi > r.h && return r.k_free[l]
    if r.hv < 0
        ell = 1/(1/(c.karman*zi)+1/450)
        f = r.ri[l] >= 0 ? 1/(1+10*r.ri[l]*sqrt(1+r.ri[l])) : 1
        return max(ell^2*sqrt(r.shear2[l])*f,c.k_bl_min)
    end
    a = 1-c.surface_fraction*c.beta*r.h/r.l_obukhov
    eta = zi/r.h
    wm = r.ustr*a^(1/3)
    km = c.karman*zi*(1-eta)^2 * (eta <= c.surface_fraction ?
        r.ustr*(1-c.beta*(zi/r.l_obukhov))^(1/3) : wm)
    pr = a^(1/3)/sqrt(a)+c.surface_fraction*c.karman*c.fakn*r.w_star/(r.ustr*a^(1/3))
    return max(km/pr,c.k_bl_min)
end

@testset "TM5 Calc_Kzz analytic contracts" begin
    for FT in (Float64,Float32), heat in (-100, 0, 20)
        col=synthetic_tm5_column(FT;sshf=heat)
        prm=TD.TM5BLDiffParameters(FT;lmax=6)
        out=tm5_scratch(FT,8); D=zeros(FT,8)
        r=@inferred TD.tm5_calc_kzz_column!(D,out,col,prm)
        @test all(isfinite,D) && all(>=(0),D)
        @test all(iszero,D[6:end])
        @test r.h >= 100
        @test r.g[end] === FT(200000)
        @test r.g[2] ≈ col.t[1]*(1+prm.geometry.virtual_factor*col.q[1])*prm.geometry.r_air*log(col.p[1]/col.p[2])/prm.constants.gravity
        @test r.ustr >= FT(0.01)
        @test r.h isa FT && r.hv isa FT && r.l_obukhov isa FT
        @test all(>=(prm.constants.k_free_min), r.k_free)
        for l in 1:5
            @test D[l] ≈ max(0,r.kvh[l])*2*(col.mass[l+1]+col.mass[l])/(r.g[l+2]-r.g[l])^2
        end
    end
    col=synthetic_tm5_column(;lsm=0,sr=0,fsr=0)
    @test isfinite(TD.tm5_ustar(col.surface,TD.TM5BLDiffParameters(;lmax=6).constants))
    @test TD.tm5_jqif(0.0,-0.0)
    @test !TD.tm5_jqif(-0.0,0.0)
    @test TD.tm5_jqif(1.0,1.0)
    # Finite selected branches for constant theta_v, zero wind and no Ri crossing.
    out=tm5_scratch(Float64,8); prm=TD.TM5BLDiffParameters(;lmax=6)
    chi=1/(prm.constants.r_air/prm.constants.r_vap)-1
    col=synthetic_tm5_column(;sshf=-100)
    t=300 ./ (1 .+ chi.*col.q) .* (((col.p[1:end-1].+col.p[2:end])./2)./1e5).^(prm.constants.r_air/prm.constants.cp_air)
    flat=merge(col,(;t,u=zeros(8),v=zeros(8)))
    r=TD.tm5_calc_kzz_column!(zeros(8),out,flat,prm)
    @test isfinite(r.h) && all(isfinite,r.kvh)
    @test r.crossing1 == 0 && r.h_pass1 == r.z[end]
    @test r.guarded > 0
    affected = 0
    for l in 1:7
        if r.theta_v[l+1] == r.theta_v[l]
            @test r.kvh[l] ≈ tm5_background_k(r,prm.constants,l) rtol=1e-12
            affected += r.hv >= 0 && r.z[l] <= r.h <= r.z[l+1] && l < prm.lmax
        end
    end
    @test r.guarded_active == affected
    @test r.guarded_active < r.guarded
    # Reference pressure equivalent of tropo34a layer 26.
    coeff=TOML.parsefile(joinpath(@__DIR__,"../../config/era5_L137_coefficients.toml"))
    @test TP.TM5_PBL_CUTOFF_PRESSURE ≈ coeff["coefficients"]["a"][58]+coeff["coefficients"]["b"][58]*TP.TM5_PBL_REFERENCE_PRESSURE rtol=0 atol=1e-10
    lev=reverse(TP.ECHLEVS_ML137_TROPO34A)
    @test TD.pbl_layer_count(Float64.(coeff["coefficients"]["a"])[collect(lev).+1],Float64.(coeff["coefficients"]["b"])[collect(lev).+1],TP.TM5_PBL_CUTOFF_PRESSURE,TP.TM5_PBL_REFERENCE_PRESSURE)==26
end

@testset "TM5 local pydyn Float64 CPU parity" begin
    root=get(ENV,"ATMOSTR_TM5_FIXTURES_DIR","")
    files=("calc_kzz_tropo34a.toml","ustar_cases.toml","jqif_cases.toml")
    if isempty(root) || !all(f->isfile(joinpath(root,"calc_kzz",f)),files)
        @info "Skipping local pydyn parity: set ATMOSTR_TM5_FIXTURES_DIR to the external TM5 fixture directory (fixtures are never committed)" root
        @test_skip false
    else
        prm=TD.TM5BLDiffParameters(;lmax=26)
        for c in TOML.parsefile(joinpath(root,"calc_kzz","ustar_cases.toml"))["case"]
            s=(;lsm=c["lsm"]/100,sr=c["sr"],fsr=c["srols"],u10=c["u10m"],v10=c["v10m"],ewss=c["ewss"],nsss=c["nsss"])
            @test TD.tm5_ustar(s,prm.constants) ≈ c["ustar"] rtol=1e-12
        end
        jq=TOML.parsefile(joinpath(root,"calc_kzz","jqif_cases.toml"))["cases"]
        @test Float64.(TD.tm5_jqif.(jq["x"],jq["y"])) == jq["jqif"]
        data=TOML.parsefile(joinpath(root,"calc_kzz","calc_kzz_tropo34a.toml"))
        errors=Dict{String,Tuple{Float64,Float64}}()
        flips=0
        guard_changes=0
        d32_error=0.; d32_l1=0.; k32_error=0.; k32_active_error=0.
        d32_floor_error=0.; d32_floor_l1=0.; singular_interfaces=0
        pu=reshape(data["grid"]["pu"]["data"],Tuple(data["grid"]["pu"]["shape"]))
        pv=reshape(data["grid"]["pv"]["data"],Tuple(data["grid"]["pv"]["shape"]))
        grid=data["grid"]
        # Use the exact TM5 lengths to isolate arithmetic from the documented A/L metric difference.
        dx=TP.EARTH_RADIUS*deg2rad(grid["dx"])*cosd(grid["ybeg"]+grid["dy"]/2)
        dy=TP.EARTH_RADIUS*deg2rad(grid["dy"])
        wind_error=(0.,0.)
        for col in data["column"]
            x,y,mid=col["inputs"],col["outputs"],col["intermediates"]
            i=col["index"]
            for l in 1:34
                u,v=TD.tm5_wind_from_faces(pu[i,1,l],pu[i+1,1,l],pv[i,1,l],pv[i,2,l],y["m"][l],dx,dy,1.)
                for (a,b) in ((u,y["uwind"][l]),(v,y["vwind"][l]))
                    wind_error=(max(wind_error[1],abs(a-b)),max(wind_error[2],abs(a-b)/max(abs(b),1e-300)))
                    @test isapprox(a,b;rtol=1e-12,atol=1e-14)
                end
            end
            s=(;lsm=x["lsm"]/100,sr=x["sr"],fsr=x["srols"],u10=x["u10m"],v10=x["v10m"],ewss=x["ewss"],nsss=x["nsss"],sshf=x["sshf"],slhf=x["slhf"],oro=x["oro"])
            pg=data["grid"]["at"] .+ data["grid"]["bt"].*x["sp"]
            input=(;p=y["phlb"],p_geometry=pg,t=x["t"],q=x["q"],mass=y["m"],u=y["uwind"],v=y["vwind"],surface=s)
            D=zeros(34); r=TD.tm5_calc_kzz_column!(D,tm5_scratch(Float64,34),input,prm)
            pairs=Dict("ustar"=>(r.ustar,y["ustar"]),"gph"=>(r.g,y["gph"]),"kvh"=>(r.kvh,y["kvh"]),"dkg"=>(D[1:26],y["dkg"]),"pblh"=>(r.h,y["pblh"]))
            for key in (:pf,:theta,:theta_v,:ri,:shear2,:k_free,:w_theta,:w_q,:ustr,:theta_s,:hv,:l_obukhov,:theta_excess,:theta_parcel,:w_star,:h_pass1,:h_pass2_raw)
                pairs[string(key)]=(getproperty(r,key),mid[string(key)])
            end
            pairs["z_centre"]=(r.z,mid["z_centre"])
            for (key,(actual,reference)) in pairs
                av=actual isa Number ? [actual] : actual
                rv=reference isa Number ? [reference] : reference
                for (l,(a,b)) in enumerate(zip(av,rv))
                    if isfinite(b)
                        err=abs(a-b); rel=err/max(abs(b),1e-300)
                        old=get(errors,key,(0.,0.)); errors[key]=(max(old[1],err),max(old[2],rel))
                        @test isapprox(a,b;rtol=1e-10,atol=1e-12)
                    else
                        @test isfinite(a)
                        if key == "kvh"
                            @test a ≈ tm5_background_k(r,prm.constants,l) rtol=1e-12
                        elseif key == "dkg"
                            kb = tm5_background_k(r,prm.constants,l)
                            expected = l < prm.lmax ? max(0,kb)*2*(input.mass[l+1]+input.mass[l])/(r.g[l+2]-r.g[l])^2 : 0.
                            @test a ≈ expected rtol=1e-12
                        end
                    end
                end
            end
            @test all(iszero,D[26:end])
            compact=vcat(reverse(D[1:33]),0.)
            full=vcat(0.,compact)
            @test all(iszero,compact[1:8]) && iszero(compact[34])
            @test all(iszero,full[1:9]) && iszero(full[35])
            @test full[10:34]==reverse(D[1:25])
            input32=map(v->v isa NamedTuple ? map(Float32,v) : Float32.(v),input)
            D32=zeros(Float32,34)
            r32=TD.tm5_calc_kzz_column!(D32,tm5_scratch(Float32,34),input32,TD.TM5BLDiffParameters(Float32;lmax=26))
            @test all(isfinite,r32.kvh)
            # Float64 arithmetic on the Float32-rounded inputs: the floor for any Float32 column.
            Dr=zeros(34)
            TD.tm5_calc_kzz_column!(Dr,tm5_scratch(Float64,34),map(v->v isa NamedTuple ? map(Float64,v) : Float64.(v),input32),prm)
            # Interfaces built with exactly equal Float64 θv are singular: Ri = 0 picks the
            # stable branch and the entrainment quotient is non-finite. Input rounding moves
            # them off that point (also in Float64 on the rounded inputs), so they are
            # counted, not scored.
            regular=[l<34 && r.theta_v[l+1]!=r.theta_v[l] for l in 1:34]
            singular_interfaces += count(l->!regular[l],1:33)
            for (a,b,ok) in zip(D32,D,regular)
                ok && b != 0 && (d32_error=max(d32_error,abs(Float64(a)-b)/abs(b)))
            end
            for (a,b,ok) in zip(D32,Dr,regular)
                ok && b != 0 && (d32_floor_error=max(d32_floor_error,abs(Float64(a)-b)/abs(b)))
            end
            for (l,(a,b)) in enumerate(zip(r32.kvh,r.kvh))
                if b != 0 && regular[l]
                    e=abs(Float64(a)-b)/abs(b)
                    k32_error=max(k32_error,e)
                    l < prm.lmax && (k32_active_error=max(k32_active_error,e))
                end
            end
            d32_l1=max(d32_l1,sum(abs.(Float64.(D32[regular]).-D[regular]))/sum(abs,D[regular]))
            d32_floor_l1=max(d32_floor_l1,sum(abs.(Float64.(D32[regular]).-Dr[regular]))/sum(abs,Dr[regular]))
            flips += (r.crossing1 != r32.crossing1) + (r.crossing2 != r32.crossing2)
            guard_changes += r.guarded != r32.guarded
        end
        for key in sort!(collect(keys(errors)))
            println("TM5 parity ",key," max_abs=",errors[key][1]," max_rel=",errors[key][2])
        end
        println("TM5 parity winds max_abs=",wind_error[1]," max_rel=",wind_error[2])
        @test d32_error <= 2e-4
        @test d32_l1 <= 3e-5          # Float64 on the rounded inputs alone: 2.6e-5
        @test k32_error <= 4e-3       # set by the top interface, above the cutoff
        @test k32_active_error <= 2e-4
        # Float32 arithmetic itself (literal Float32 differences gave 1.5e-4 and 4.0e-5).
        @test d32_floor_error <= 2e-5
        @test d32_floor_l1 <= 1e-5
        println("TM5 Float32 D max_entry_rel=",d32_error," max_column_L1_rel=",d32_l1," K max_entry_rel=",k32_error," active_K max_entry_rel=",k32_active_error)
        println("TM5 Float32 D against Float64 on Float32 inputs: max_entry_rel=",d32_floor_error," max_column_L1_rel=",d32_floor_l1)
        println("TM5 Float32 PBL crossing changes: ",flips," / 32 passes")
        println("TM5 Float32 guarded-interface count changes: ",guard_changes," / 16 columns (",singular_interfaces," singular interfaces)")
    end
end
