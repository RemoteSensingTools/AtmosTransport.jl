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


# Scratch is shared by independent column chunks, including all six CS panels.
function _tm5_column_scratch(::Type{FT}, nc::Int, L::Int, chunk_columns::Int) where FT
    chunk_columns > 0 || throw(ArgumentError("TM5 chunk_columns must be positive"))
    n = min(nc, chunk_columns)
    centres = (;p=zeros(FT,n,L+1),t=zeros(FT,n,L),q=zeros(FT,n,L),
        mass=zeros(FT,n,L),u=zeros(FT,n,L),v=zeros(FT,n,L),D=zeros(FT,n,L))
    out = (;g=zeros(FT,n,L+1),pf=zeros(FT,n,L),theta=zeros(FT,n,L),theta_v=zeros(FT,n,L),
        z=zeros(FT,n,L),ri=zeros(FT,n,L-1),shear2=zeros(FT,n,L-1),
        k_free=zeros(FT,n,L-1),kvh=zeros(FT,n,L-1))
    return (;centres,out,guarded=zeros(Int,n))
end

function TM5CalcKzzField(data::AbstractArray{FT,N}, metrics::Union{TM5LLMetrics,TM5RGMetrics}, vertical;
                         chunk_columns::Int=65_536) where {FT,N}
    L = size(data,N)
    L >= 2 || throw(ArgumentError("TM5 Calc_Kzz needs at least two layers"))
    nc = length(metrics.area)
    prod(size(data)[1:N-1]) == nc || throw(DimensionMismatch("TM5 cache does not match metrics"))
    scratch = _tm5_column_scratch(FT,nc,L,chunk_columns)
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

struct TM5FlatLayout end
@inline _tm5_mass(::TM5FlatLayout, mass, c, k) = @inbounds mass[c,k]
@inline _tm5_input(::TM5FlatLayout, input, c, k) = @inbounds input[c,k]
@inline _tm5_input(::TM5FlatLayout, input, c) = @inbounds input[c]
@inline _tm5_area(metrics::Union{TM5LLMetrics,TM5RGMetrics}, c) = @inbounds metrics.area[c]
@inline _tm5_store!(::TM5FlatLayout, data, c, k, value) = (@inbounds data[c,k] = value)

@inline function _tm5_refresh_column!(data, centres, out, guards, inputs, mass,
                                     fluxes, metrics, params, p_top, seconds, basis,
                                     layout, c, s, L)
    FT = typeof(p_top)
    @inbounds begin
        centres.p[s,L+1] = p_top
        # Top-down pressure: TM5's plain sum in Float64, Kahan-compensated otherwise.
        p_sum, p_comp = p_top, zero(FT)
        for k in 1:L
            l = L-k+1
            q = _tm5_input(layout,inputs.q,c,k)
            m = _tm5_mass(layout,mass,c,k)
            centres.t[s,l] = _tm5_input(layout,inputs.t,c,k)
            centres.q[s,l] = q
            centres.mass[s,l] = m
            p_sum, p_comp = _kahan_add(p_sum, p_comp, params.constants.gravity * _tm5_pressure_mass(basis,m,q) / _tm5_area(metrics,c))
            centres.p[s,l] = p_sum
            u,v = _tm5_window_winds(metrics,fluxes,c,k,m,seconds)
            centres.u[s,l],centres.v[s,l] = u,v
        end
        surface = (;lsm=_tm5_input(layout,inputs.lsm,c),sr=_tm5_input(layout,inputs.sr,c),fsr=_tm5_input(layout,inputs.fsr,c),u10=_tm5_input(layout,inputs.u10,c),
            v10=_tm5_input(layout,inputs.v10,c),ewss=_tm5_input(layout,inputs.ewss,c),nsss=_tm5_input(layout,inputs.nsss,c),sshf=_tm5_input(layout,inputs.sshf,c),
            slhf=_tm5_input(layout,inputs.slhf,c),oro=_tm5_input(layout,inputs.oro,c))
        col = (;p=view(centres.p,s,:),p_geometry=view(centres.p,s,:),t=view(centres.t,s,:),
            q=view(centres.q,s,:),mass=view(centres.mass,s,:),u=view(centres.u,s,:),v=view(centres.v,s,:),surface)
        column_out = map(a->view(a,s,:),out)
        result = tm5_calc_kzz_column!(view(centres.D,s,:),column_out,col,params)
        guards[s] = result.guarded_active
        for k in 1:L-1
            _tm5_store!(layout,data,c,k,centres.D[s,L-k])
        end
        _tm5_store!(layout,data,c,L,zero(FT))
    end
end

@kernel function _tm5_refresh_kernel!(data, centres, out, guards,
                                      @Const(inputs),@Const(mass),fluxes,metrics,params,
                                      p_top,seconds,basis,layout,offset,L)
    s = @index(Global,Linear)
    _tm5_refresh_column!(data,centres,out,guards,inputs,mass,fluxes,metrics,params,
                         p_top,seconds,basis,layout,offset+s,s,L)
end

function _tm5_refresh_chunks!(field, data, inputs, mass, fluxes, vertical, seconds, basis,
                              layout, nc, L)
    FT = typeof(field.params.constants.gravity)
    s = field.scratch
    s === nothing && throw(ArgumentError("cannot refresh a frozen TM5 Calc_Kzz field"))
    backend = get_backend(s.centres.p)
    chunk = length(s.guarded)
    guarded = 0
    for offset in 0:chunk:nc-1
        n = min(chunk,nc-offset)
        launch!(_tm5_refresh_kernel!,backend,256,n,data,s.centres,s.out,s.guarded,
            inputs,mass,fluxes,field.metrics,field.params,FT(first(vertical.A)),FT(seconds),
            basis,layout,offset,L)
        guarded += sum(view(s.guarded,1:n))
    end
    if guarded > 0 && !field.guard_warned[]
        @warn "TM5 Calc_Kzz kept BL K at non-finite active entrainment interfaces (further warnings suppressed for this run)" guarded
        field.guard_warned[] = true
    end
    return field
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
    return _tm5_refresh_chunks!(field,reshape(field.dkg,nc,L),flat,reshape(mass,nc,L),
        fluxes,vertical,seconds,basis,TM5FlatLayout(),nc,L)
end
