"Cubed-sphere areas, face lengths, tangent-angle cosine/sine and forcing halo width."
struct TM5CSMetrics{A,L,B}
    area::A
    lx::L
    ly::L
    basis::B
    Hp::Int
end

"""
    TM5CalcKzzCSField(dkg, metrics, vertical; chunk_columns=65_536)

Live TM5 interface exchange [kg/s] on six interior `(Nc,Nc,Nz)` panels.
One bounded column scratch serves all panels. The last D plane is zero.
Deliberately has no `panel_field` method: D must never enter a generic Kz
path. Adjoint capture stores copies of D with `scratch = metrics = nothing`.
"""
struct TM5CalcKzzCSField{FT,D<:AbstractArray{FT,3},S,P,M} <: AbstractCubedSphereField{FT}
    dkg::NTuple{6,D}
    scratch::S
    params::P
    metrics::M
    guard_warned::Base.RefValue{Bool}
end
Adapt.@adapt_structure TM5CSMetrics
Adapt.@adapt_structure TM5CalcKzzCSField

function TM5CalcKzzCSField(dkg::NTuple{6,AbstractArray{FT,3}}, metrics::TM5CSMetrics, vertical;
                          chunk_columns::Int=65_536) where FT
    nx,ny,L = size(first(dkg))
    L >= 2 || throw(ArgumentError("TM5 Calc_Kzz needs at least two layers"))
    nx == ny && size(metrics.area) == (nx,ny) && all(a->size(a)==(nx,ny,L),dkg) ||
        throw(DimensionMismatch("TM5 CS cache does not match metrics"))
    scratch = _tm5_column_scratch(FT,6nx*ny,L,chunk_columns)
    lmax = pbl_layer_count(vertical.A,vertical.B,TM5_PBL_CUTOFF_PRESSURE,TM5_PBL_REFERENCE_PRESSURE)
    return TM5CalcKzzCSField(dkg,scratch,TM5BLDiffParameters(FT;lmax),metrics,Ref(false))
end
update_field!(f::TM5CalcKzzCSField, ::Real) = f

struct TM5CSLayout
    Nc::Int
    Hp::Int
end
@inline function _tm5_cs_indices(Nc,c)
    local_c = (c-1) % (Nc*Nc)
    return local_c % Nc + 1, local_c ÷ Nc + 1, (c-1) ÷ (Nc*Nc) + 1
end
@inline function _tm5_mass(layout::TM5CSLayout, mass, c, k)
    i,j,p = _tm5_cs_indices(layout.Nc,c)
    return @inbounds select_panel(mass,p)[i+layout.Hp,j+layout.Hp,k]
end
@inline function _tm5_input(layout::TM5CSLayout, input, c, k)
    i,j,p = _tm5_cs_indices(layout.Nc,c)
    return @inbounds select_panel(input,p)[i,j,k]
end
@inline function _tm5_input(layout::TM5CSLayout, input, c)
    i,j,p = _tm5_cs_indices(layout.Nc,c)
    return @inbounds select_panel(input,p)[i,j]
end
@inline function _tm5_area(metrics::TM5CSMetrics,c)
    i,j,_ = _tm5_cs_indices(size(metrics.area,1),c)
    return @inbounds metrics.area[i,j]
end
@inline function _tm5_store!(layout::TM5CSLayout, data, c, k, value)
    i,j,p = _tm5_cs_indices(layout.Nc,c)
    @inbounds select_panel(data,p)[i,j,k] = value
end

@inline function _tm5_window_winds(metrics::TM5CSMetrics,fluxes,c,k,m,seconds)
    i,j,p = _tm5_cs_indices(size(metrics.area,1),c)
    h = metrics.Hp
    am,bm = select_panel(fluxes.am,p),select_panel(fluxes.bm,p)
    cosine,sine = select_panel(metrics.basis,p)
    @inbounds begin
        scale = metrics.area[i,j]/(2*m*seconds)
        u1 = scale*(am[i+h,j+h,k]/metrics.lx[i,j] + am[i+h+1,j+h,k]/metrics.lx[i+1,j])
        u2 = scale*(bm[i+h,j+h,k]/metrics.ly[i,j] + bm[i+h,j+h+1,k]/metrics.ly[i,j+1])
        # Face-normal components -> orthonormal frame (ex, ex-perpendicular).
        return (u1+cosine[i,j]*u2)/sine[i,j],u2
    end
end

function refresh_tm5_calc_kzz!(field::TM5CalcKzzCSField, inputs, mass, fluxes,
                               vertical, seconds, basis)
    inputs === nothing && throw(ArgumentError("tm5_calc_kzz requires the complete v5 tm5_pbl group"))
    seconds > 0 || throw(ArgumentError("TM5 flux application seconds must be positive"))
    field.metrics === nothing && throw(ArgumentError("cannot refresh a frozen TM5 Calc_Kzz field"))
    Nc,_,L = size(first(field.dkg))
    Hp = field.metrics.Hp
    for (name, panels, shape) in ((:mass,mass,(Nc+2Hp,Nc+2Hp,L)),
                                  (:am,fluxes.am,(Nc+1+2Hp,Nc+2Hp,L)),
                                  (:bm,fluxes.bm,(Nc+2Hp,Nc+1+2Hp,L)))
        length(panels) == 6 && all(a->size(a)==shape,panels) ||
            throw(DimensionMismatch("TM5 CS $name panels must have shape $shape (halo width $Hp)"))
    end
    for name in (:t,:q,:lsm,:sr,:fsr,:u10,:v10,:ewss,:nsss,:sshf,:slhf,:oro)
        panels = getproperty(inputs,name)
        shape = name in (:t,:q) ? (Nc,Nc,L) : (Nc,Nc)
        length(panels) == 6 && all(a->size(a)==shape,panels) ||
            throw(DimensionMismatch("TM5 CS $name input panels must have shape $shape"))
    end
    return _tm5_refresh_chunks!(field,field.dkg,inputs,mass,fluxes,vertical,seconds,basis,
        TM5CSLayout(Nc,field.metrics.Hp),6Nc*Nc,L)
end
