function _diffusion_sequence_at(value, step::Int, nsteps::Int,
                                name::AbstractString)
    if value isa AbstractVector
        length(value) == nsteps || throw(ArgumentError(
            "$name length $(length(value)) does not match nsteps $nsteps"))
        return value[step]
    else
        return value
    end
end

@inline _validate_cs_diffusion_kz_for_adjoint(_op) = nothing

_validate_cs_diffusion_kz_for_adjoint(::ImplicitVerticalDiffusion{FT, <:GCHPNonlocalPBLField}) where FT =
    throw(ArgumentError(
        "GEOS-Chem non-local VDIFF has no adjoint yet: the footprint tape neither refreshes " *
        "the field nor transposes its counter-gradient emission profile."))

function _validate_cs_diffusion_kz_for_adjoint(
    op::ImplicitVerticalDiffusion{FT, <:LocalHoltslagBovilleKzField}) where FT
    @inbounds for p in 1:6
        data = panel_field(op.kz_field, p).data
        all(isfinite, data) || throw(ArgumentError(
            "Local Holtslag-Boville VDIFF diffusion adjoint requires a finite Kz " *
            "cache; panel $p contains NaN or Inf. Refresh with " *
            "`refresh_local_holtslag_boville_kz_cache!` before recording the tape."))
        maximum(data) > zero(FT) || throw(ArgumentError(
            "Local Holtslag-Boville VDIFF diffusion adjoint received an all-zero " *
            "Kz cache on panel $p. This usually means the VDIFF Kz field was " *
            "constructed but never refreshed from surface/vdiff forcing before " *
            "the adjoint tape was recorded."))
    end
    return nothing
end

# Whether the forward solve of `op` reads the layer thickness: Kz fields convert
# Kz into interface exchange with it; Dkg fields and TM5 Calc_Kzz carry their
# exchange. The one definition: validation here and the capture in
# Models.Runner (`cs_adjoint_inputs`) both use it.
_reads_layer_thickness(_op) = true
_reads_layer_thickness(::ImplicitVerticalDiffusion{FT, <:Union{TM5CalcKzzCSField, AbstractCSDkgField}}) where FT =
    false

# The diffusion inputs of a call: each step's operator with a finite Kz, and
# the step's layer thickness (six panels [m]) where its solve reads one.
function _validate_cs_diffusion_inputs(diffusion_op, diffusion_layer_thickness,
                                       nsteps::Int)
    checked_kz = IdDict{Any, Bool}()
    for step in 1:nsteps
        op = _diffusion_sequence_at(diffusion_op, step, nsteps, "diffusion_op")
        op isa NoDiffusion && continue
        if !haskey(checked_kz, op.kz_field)
            _validate_cs_diffusion_kz_for_adjoint(op)
            checked_kz[op.kz_field] = true
        end
        dz = _diffusion_sequence_at(diffusion_layer_thickness, step, nsteps,
                                    "diffusion_layer_thickness")
        dz isa NTuple{6, AbstractArray} || (dz === nothing && !_reads_layer_thickness(op)) ||
            throw(ArgumentError(
                "`diffusion_layer_thickness` of $(nameof(typeof(op.kz_field))) diffusion at step $step " *
                "is a $(typeof(dz)); pass six (Nc, Nc, Nz) panels [m], one tuple or one per step" *
                (_reads_layer_thickness(op) ? "" : ", or nothing (this operator reads none)")))
    end
    return nothing
end

# The column arrays of a reverse solve (writable scratch and, for Kz fields,
# the layer thickness): one (Nc, Nc, Nz) panel per adjoint panel.
function _check_cs_diffusion_columns(lambda_panels, panels_m, mesh::CubedSphereMesh, columns...)
    for p in 1:6
        size(lambda_panels[p]) == size(panels_m[p]) || throw(DimensionMismatch(
            "adjoint tracer panel $p shape $(size(lambda_panels[p])) does not match air_mass " *
            "shape $(size(panels_m[p]))"))
        expected = (mesh.Nc, mesh.Nc, size(lambda_panels[p], 3))
        for a in columns
            length(a) == 6 || throw(DimensionMismatch(
                "CS adjoint diffusion column arrays need six panels; got $(length(a))"))
            size(a[p]) == expected || throw(DimensionMismatch(
                "CS adjoint diffusion column panel $p has shape $(size(a[p])); expected $expected"))
        end
    end
    return nothing
end

# Reverse launchers: `_apply_cs_diffusion_adjoint!(lambda, m, op, layer_thickness,
# scratch, dt, meteo, mesh)` transposes one forward solve of `op` on air mass
# `m`. `layer_thickness` is the solve's (read only); `scratch` six writable
# (Nc, Nc, Nz) panels, overwritten column by column before they are read (any
# contents, shared by every solve of the reverse pass).

_apply_cs_diffusion_adjoint!(lambda_panels, panels_m, ::NoDiffusion, _layer_thickness,
                             _scratch, dt, meteo, mesh::CubedSphereMesh) = nothing

function _apply_cs_diffusion_adjoint!(lambda_panels::NTuple{6, A},
                                      panels_m::NTuple{6},
                                      op::ImplicitVerticalDiffusion{FT, KzF},
                                      _layer_thickness, w_scratch, dt, meteo,
                                      mesh::CubedSphereMesh) where {
                                          FT, A <: AbstractArray{FT, 3},
                                          KzF <: AbstractCSDkgField{FT}}
    _check_cs_diffusion_columns(lambda_panels, panels_m, mesh, w_scratch)
    update_field!(op.kz_field, current_time(meteo))
    Hp, Nc = mesh.Hp, mesh.Nc
    @inbounds for p in 1:6
        panel_lambda = lambda_panels[p]
        panel_m = panels_m[p]
        Nz = size(panel_lambda, 3)
        panel_dkg = panel_field(op.kz_field, p)
        backend = get_backend(panel_lambda)
        kernel! = _vertical_diffusion_cs_single_dkg_adjoint_kernel!(backend, (8, 8))
        kernel!(panel_lambda, panel_m, panel_dkg, w_scratch[p], FT(dt), Nz, Hp;
                ndrange = (Nc, Nc))
    end
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels::NTuple{6, A},
                                      panels_m::NTuple{6},
                                      op::ImplicitVerticalDiffusion{FT, KzF},
                                      layer_thickness, w_scratch, dt, meteo,
                                      mesh::CubedSphereMesh) where {
                                          FT, A <: AbstractArray{FT, 3},
                                          KzF <: AbstractCubedSphereField{FT}}
    layer_thickness isa NTuple{6, AbstractArray} || throw(ArgumentError(
        "CS adjoint Kz diffusion needs the solve's layer thickness (six panels); got " *
        "$(typeof(layer_thickness))"))
    _check_cs_diffusion_columns(lambda_panels, panels_m, mesh, w_scratch, layer_thickness)
    update_field!(op.kz_field, current_time(meteo))

    Hp = mesh.Hp
    Nc = mesh.Nc
    @inbounds for p in 1:6
        panel_lambda = lambda_panels[p]
        panel_m = panels_m[p]
        Nz = size(panel_lambda, 3)

        panel_kz = panel_field(op.kz_field, p)
        backend = get_backend(panel_lambda)
        kernel! = _vertical_diffusion_cs_single_adjoint_kernel!(backend, (8, 8))
        kernel!(panel_lambda, panel_m, panel_kz, layer_thickness[p], w_scratch[p],
                FT(dt), Nz, Hp; ndrange = (Nc, Nc))
    end
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels, panels_m,
                                      op::ImplicitVerticalDiffusion,
                                      _layer_thickness, _scratch, dt, meteo,
                                      mesh::CubedSphereMesh)
    throw(ArgumentError(
        "CS adjoint diffusion requires `ImplicitVerticalDiffusion` with a " *
        "`CubedSphereField` Kz field; got $(typeof(op.kz_field))"))
end

function _validate_cs_diffusion_kz_for_adjoint(
    op::ImplicitVerticalDiffusion{FT,<:TM5CalcKzzCSField}) where FT
    f = op.kz_field
    for d in f.dkg
        all(x->isfinite(x) && x>=zero(FT),d) || throw(ArgumentError(
            "TM5 Calc_Kzz adjoint requires finite nonnegative D"))
        L = size(d,3)
        # Top-down: interfaces above the bottom lmax layers and terminal plane.
        all(iszero,view(d,:,:,1:max(0,L-f.params.lmax))) &&
            all(iszero,view(d,:,:,L)) || throw(ArgumentError(
            "TM5 Calc_Kzz adjoint requires zero D above lmax and in the last plane"))
    end
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels::NTuple{6,A},panels_m::NTuple{6},
    op::ImplicitVerticalDiffusion{FT,<:TM5CalcKzzCSField},_layer_thickness,scratch,dt,meteo,
    mesh::CubedSphereMesh) where {FT,A<:AbstractArray{FT,3}}
    _check_cs_diffusion_columns(lambda_panels,panels_m,mesh,scratch)
    for p in 1:6
        kernel! = _vertical_diffusion_cs_single_direct_dkg_adjoint_kernel!(
            get_backend(lambda_panels[p]),(8,8))
        kernel!(lambda_panels[p],panels_m[p],op.kz_field.dkg[p],scratch[p],FT(dt),
                size(lambda_panels[p],3),mesh.Hp;ndrange=(mesh.Nc,mesh.Nc))
    end
    return nothing
end
