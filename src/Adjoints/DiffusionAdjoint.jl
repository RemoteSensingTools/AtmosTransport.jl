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

function _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace,
                                       nsteps::Int)
    checked_kz = IdDict{Any, Bool}()
    for step in 1:nsteps
        op = _diffusion_sequence_at(diffusion_op, step, nsteps, "diffusion_op")
        if !(op isa NoDiffusion)
            if !haskey(checked_kz, op.kz_field)
                _validate_cs_diffusion_kz_for_adjoint(op)
                checked_kz[op.kz_field] = true
            end
            ws = _diffusion_sequence_at(diffusion_workspace, step, nsteps,
                                        "diffusion_workspace")
            check_diffusion_workspace(ws)
        end
    end
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels, panels_m, ::NoDiffusion,
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh)
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels::NTuple{6, A},
                                      panels_m::NTuple{6},
                                      op::ImplicitVerticalDiffusion{FT, KzF},
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh) where {
                                          FT, A <: AbstractArray{FT, 3},
                                          KzF <: AbstractCSDkgField{FT}}
    w_scratch, _ = check_diffusion_workspace(workspace)
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
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh) where {
                                          FT, A <: AbstractArray{FT, 3},
                                          KzF <: AbstractCubedSphereField{FT}}
    w_scratch, dz_scratch = check_diffusion_workspace(workspace)
    update_field!(op.kz_field, current_time(meteo))

    Hp = mesh.Hp
    Nc = mesh.Nc
    @inbounds for p in 1:6
        panel_lambda = lambda_panels[p]
        panel_m = panels_m[p]
        Nz = size(panel_lambda, 3)
        check_diffusion_workspace(workspace, panel_lambda, panel_m; panel = p, columns = (Nc, Nc))

        panel_kz = panel_field(op.kz_field, p)
        backend = get_backend(panel_lambda)
        kernel! = _vertical_diffusion_cs_single_adjoint_kernel!(backend, (8, 8))
        kernel!(panel_lambda, panel_m, panel_kz, dz_scratch[p], w_scratch[p],
                FT(dt), Nz, Hp; ndrange = (Nc, Nc))
    end
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels, panels_m,
                                      op::ImplicitVerticalDiffusion,
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh)
    throw(ArgumentError(
        "CS adjoint diffusion requires `ImplicitVerticalDiffusion` with a " *
        "`CubedSphereField` Kz field; got $(typeof(op.kz_field))"))
end
