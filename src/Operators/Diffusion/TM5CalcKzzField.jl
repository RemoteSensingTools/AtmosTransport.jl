"""
    TM5CalcKzzField(data, metrics, vertical)

Window-cached live TM5 interface exchange [kg/s] for lat-lon (rank 3) or
reduced Gaussian (rank 2). `data[...,k]` exchanges layers k and k+1, top-down;
the last plane is zero. Used inside `ImplicitVerticalDiffusion`. Construct
on the host with mesh metrics materialized by the runtime recipe, then Adapt
with the rest of the model for device execution.
"""
struct TM5CalcKzzField{FT,N,D<:AbstractArray{FT,N},S,P,M} <: AbstractTimeVaryingField{FT,N}
    dkg::D
    scratch::S
    params::P
    metrics::M
    guard_warned::Base.RefValue{Bool}
end

struct TM5LLMetrics{A,X,Y}
    area::A
    dx::X
    dy::Y
    nx::Int
end
struct TM5RGMetrics{A,X,Y,O,F}
    area::A
    dx::X
    dy::Y
    offsets::O
    faces::F
end
Adapt.@adapt_structure TM5LLMetrics
Adapt.@adapt_structure TM5RGMetrics
Adapt.@adapt_structure TM5CalcKzzField


function TM5CalcKzzField(data::AbstractArray{FT,N}, metrics::Union{TM5LLMetrics,TM5RGMetrics}, vertical) where {FT,N}
    L = size(data,N)
    L >= 2 || throw(ArgumentError("TM5 Calc_Kzz needs at least two layers"))
    nc = length(metrics.area)
    prod(size(data)[1:N-1]) == nc || throw(DimensionMismatch("TM5 cache does not match metrics"))
    centres = (;p=zeros(FT,nc,L+1),t=zeros(FT,nc,L),q=zeros(FT,nc,L),
        mass=zeros(FT,nc,L),u=zeros(FT,nc,L),v=zeros(FT,nc,L),D=zeros(FT,nc,L))
    out = (;g=zeros(FT,nc,L+1),pf=zeros(FT,nc,L),theta=zeros(FT,nc,L),theta_v=zeros(FT,nc,L),
        z=zeros(FT,nc,L),ri=zeros(FT,nc,L-1),shear2=zeros(FT,nc,L-1),
        k_free=zeros(FT,nc,L-1),kvh=zeros(FT,nc,L-1))
    scratch = (;centres,out,guarded=zeros(Int,nc))
    lmax = pbl_layer_count(vertical.A,vertical.B,TM5_PBL_CUTOFF_PRESSURE,TM5_PBL_REFERENCE_PRESSURE)
    params = TM5BLDiffParameters(FT;lmax)
    return TM5CalcKzzField{FT,N,typeof(data),typeof(scratch),typeof(params),typeof(metrics)}(
        data,scratch,params,metrics,Ref(false))
end

@inline field_value(f::TM5CalcKzzField, I::Tuple) = @inbounds f.dkg[I...]
update_field!(f::TM5CalcKzzField, ::Real) = f

@inline function _tm5_window_winds(metrics::TM5LLMetrics,fluxes,c,k,m,seconds)
    i=mod1(c,metrics.nx); j=(c-1)÷metrics.nx+1
    @inbounds return tm5_wind_from_faces(fluxes.am[i,j,k],fluxes.am[i+1,j,k],
        fluxes.bm[i,j,k],fluxes.bm[i,j+1,k],m,metrics.dx[c],metrics.dy[c],seconds)
end
@inline function _tm5_window_winds(metrics::TM5RGMetrics,fluxes,c,k,m,seconds)
    sums = ntuple(4) do side
        x = zero(m)
        @inbounds for pos in metrics.offsets[4*(c-1)+side]:metrics.offsets[4*(c-1)+side+1]-1
            x += fluxes.horizontal_flux[metrics.faces[pos],k]
        end
        x
    end
    @inbounds return tm5_wind_from_faces(sums...,m,metrics.dx[c],metrics.dy[c],seconds)
end

@inline _tm5_pressure_mass(::DryBasis,m,q) = m/(one(q)-q)
@inline _tm5_pressure_mass(::MoistBasis,m,_q) = m

@kernel function _tm5_refresh_kernel!(data, centres, out, guards,
                                      @Const(inputs),@Const(mass),fluxes,metrics,params,
                                      p_top,seconds,basis)
    c = @index(Global,Linear)
    FT = eltype(data)
    L = size(data,2)
    @inbounds begin
        centres.p[c,L+1] = p_top
        for k in 1:L
            l = L-k+1
            q = inputs.q[c,k]
            m = mass[c,k]
            centres.t[c,l] = inputs.t[c,k]
            centres.q[c,l] = q
            centres.mass[c,l] = m
            centres.p[c,l] = centres.p[c,l+1] + params.constants.gravity * _tm5_pressure_mass(basis,m,q) / metrics.area[c]
            u,v = _tm5_window_winds(metrics,fluxes,c,k,m,seconds)
            centres.u[c,l],centres.v[c,l] = u,v
        end
        surface = (;lsm=inputs.lsm[c],sr=inputs.sr[c],fsr=inputs.fsr[c],u10=inputs.u10[c],
            v10=inputs.v10[c],ewss=inputs.ewss[c],nsss=inputs.nsss[c],sshf=inputs.sshf[c],
            slhf=inputs.slhf[c],oro=inputs.oro[c])
        col = (;p=view(centres.p,c,:),p_geometry=view(centres.p,c,:),t=view(centres.t,c,:),
            q=view(centres.q,c,:),mass=view(centres.mass,c,:),u=view(centres.u,c,:),v=view(centres.v,c,:),surface)
        column_out = map(a->view(a,c,:),out)
        result = tm5_calc_kzz_column!(view(centres.D,c,:),column_out,col,params)
        guards[c] = result.guarded_active
        for k in 1:L-1
            data[c,k] = centres.D[c,L-k]
        end
        data[c,L] = zero(FT)
    end
end

"""
    refresh_tm5_calc_kzz!(field, inputs, mass, fluxes, vertical, seconds, basis)

Refresh LL/RG exchange from the runtime window-start carrier mass and face
amounts. `seconds` is `flux_application_seconds` of the stored amounts.
`basis` identifies whether the pressure mass is dry (divide by 1-q) or total.
Warn once per field/run when a non-finite entrainment expression changes an
active exchange. Unused branches and interfaces above the cutoff do not count.
"""
function refresh_tm5_calc_kzz!(field::TM5CalcKzzField{FT,N}, inputs, mass, fluxes,
                                vertical, seconds, basis) where {FT,N}
    inputs === nothing && throw(ArgumentError("tm5_calc_kzz requires the complete v5 tm5_pbl group"))
    seconds > 0 || throw(ArgumentError("TM5 flux application seconds must be positive"))
    nc = length(field.metrics.area); L = size(field.dkg,N)
    flat = map(inputs) do a
        ndims(a)==N ? reshape(a,nc,L) : reshape(a,nc)
    end
    s = field.scratch
    backend = get_backend(field.dkg)
    launch!(_tm5_refresh_kernel!,backend,256,nc,reshape(field.dkg,nc,L),s.centres,s.out,s.guarded,
        flat,reshape(mass,nc,L),fluxes,field.metrics,field.params,FT(first(vertical.A)),FT(seconds),basis)
    guarded = sum(s.guarded)
    if guarded > 0 && !field.guard_warned[]
        @warn "TM5 Calc_Kzz kept BL K at non-finite active entrainment interfaces (further warnings suppressed for this run)" guarded
        field.guard_warned[] = true
    end
    return field
end
