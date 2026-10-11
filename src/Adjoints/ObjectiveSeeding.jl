# ---------------------------------------------------------------------------
# Footprint-objective seeding + evaluation.
#
# Defines the abstract objective type and three concrete subtypes
# (`CSLayerMeanObjective`, `CSColumnMeanObjective`, `CSSeedObjective`),
# plus the kernels that:
#
#   * validate an objective against mesh/Nz bounds,
#   * forward-evaluate it on a `(panels_rm, panels_m)` pair to get the
#     scalar value of the cost function,
#   * seed `lambda_panels` from the objective at final time
#     (`_seed_objective!`), and
#   * accumulate the per-step surface footprint
#     (`_accumulate_surface_footprint!`).
# ---------------------------------------------------------------------------

abstract type AbstractCSFootprintObjective end

"""
    CSLayerMeanObjective(panel, i, j, level)

Scalar objective equal to the final-layer mixing ratio
`rm[panel][i, j, level] / m[panel][i, j, level]` on the physical CS
interior indices. `level = Nz` is the surface layer.
"""
struct CSLayerMeanObjective <: AbstractCSFootprintObjective
    panel::Int
    i::Int
    j::Int
    level::Int
end

"""
    CSColumnMeanObjective(panel, i, j)

Scalar objective equal to the final air-mass-weighted column mean mixing
ratio at one physical CS interior cell.
"""
struct CSColumnMeanObjective <: AbstractCSFootprintObjective
    panel::Int
    i::Int
    j::Int
end

"""
    CSSeedObjective()

Marker objective used when the caller supplies an explicit final adjoint
seed (`dJ/drm_final`) instead of one of the built-in scalar objectives.
"""
struct CSSeedObjective <: AbstractCSFootprintObjective end

# ---------------------------------------------------------------------------
# Objective validation + forward evaluation
# ---------------------------------------------------------------------------

function _validate_objective(obj::CSLayerMeanObjective, mesh::CubedSphereMesh, Nz::Int)
    1 <= obj.panel <= 6 || throw(ArgumentError("panel must be in 1:6, got $(obj.panel)"))
    1 <= obj.i <= mesh.Nc || throw(ArgumentError("i must be in 1:$(mesh.Nc), got $(obj.i)"))
    1 <= obj.j <= mesh.Nc || throw(ArgumentError("j must be in 1:$(mesh.Nc), got $(obj.j)"))
    1 <= obj.level <= Nz || throw(ArgumentError("level must be in 1:$Nz, got $(obj.level)"))
    return nothing
end

function _validate_objective(obj::CSColumnMeanObjective, mesh::CubedSphereMesh, Nz::Int)
    1 <= obj.panel <= 6 || throw(ArgumentError("panel must be in 1:6, got $(obj.panel)"))
    1 <= obj.i <= mesh.Nc || throw(ArgumentError("i must be in 1:$(mesh.Nc), got $(obj.i)"))
    1 <= obj.j <= mesh.Nc || throw(ArgumentError("j must be in 1:$(mesh.Nc), got $(obj.j)"))
    return nothing
end

function _validate_objective(::CSSeedObjective, mesh::CubedSphereMesh, Nz::Int)
    throw(ArgumentError(
        "`CSSeedObjective` is reserved for explicit final adjoint seeds; " *
        "use `cs_surface_emission_footprint_from_seed(final_adjoint_rm, ...)`"))
end

# Air-mass floor of the objectives, `eps(FT)`. The layer objective's evaluation
# `rm / max(m, floor)` and its adjoint seed `1 / max(m, floor)` both read it here,
# so the seed stays the derivative of the evaluated objective.
@inline _objective_mass_floor(::Type{FT}) where {FT} = eps(FT)

@kernel function _evaluate_layer_objective_kernel!(out, @Const(rm), @Const(m),
                                                   i, j, k)
    _ = @index(Global)
    FT = eltype(rm)
    @inbounds out[1] = rm[i, j, k] / max(m[i, j, k], _objective_mass_floor(FT))
end

@kernel function _evaluate_column_objective_kernel!(out, @Const(rm), @Const(m),
                                                    i, j, Nz::Int)
    _ = @index(Global)
    FT = eltype(rm)
    num = zero(FT)
    den = zero(FT)
    @inbounds for k in 1:Nz
        num += rm[i, j, k]
        den += m[i, j, k]
    end
    @inbounds out[1] = num / max(den, _objective_mass_floor(FT))
end

_host_scalar(a) = Array(a)[1]

function evaluate_objective(obj::CSLayerMeanObjective, panels_rm, panels_m,
                            mesh::CubedSphereMesh)
    Hp = mesh.Hp
    p = obj.panel
    ii = Hp + obj.i
    jj = Hp + obj.j
    k = obj.level
    FT = eltype(panels_rm[p])
    out = similar(panels_rm[p], FT, 1)
    backend = get_backend(panels_rm[p])
    kernel! = _evaluate_layer_objective_kernel!(backend, 1)
    kernel!(out, panels_rm[p], panels_m[p], Int32(ii), Int32(jj), Int32(k);
            ndrange = 1)
    synchronize(backend)
    return _host_scalar(out)
end

function evaluate_objective(obj::CSColumnMeanObjective, panels_rm, panels_m,
                            mesh::CubedSphereMesh)
    Hp = mesh.Hp
    p = obj.panel
    ii = Hp + obj.i
    jj = Hp + obj.j
    FT = eltype(panels_rm[p])
    out = similar(panels_rm[p], FT, 1)
    backend = get_backend(panels_rm[p])
    kernel! = _evaluate_column_objective_kernel!(backend, 1)
    kernel!(out, panels_rm[p], panels_m[p], Int32(ii), Int32(jj),
            size(panels_rm[p], 3); ndrange = 1)
    synchronize(backend)
    return _host_scalar(out)
end

# ---------------------------------------------------------------------------
# Adjoint seeding + per-step surface accumulator
# ---------------------------------------------------------------------------

@kernel function _seed_layer_objective_kernel!(lambda, @Const(m), value, i, j, k)
    _ = @index(Global)
    @inbounds lambda[i, j, k] = value / max(m[i, j, k], _objective_mass_floor(eltype(lambda)))
end

@kernel function _seed_column_objective_kernel!(lambda, @Const(m), denom, i, j, Hp)
    k = @index(Global, Linear)
    @inbounds lambda[i, j, k] = one(eltype(lambda)) / denom
end

function _seed_objective!(lambda_panels, obj::CSLayerMeanObjective, final_m,
                          mesh::CubedSphereMesh)
    FT = eltype(lambda_panels[1])
    @inbounds for p in 1:6
        fill!(lambda_panels[p], zero(FT))
    end
    p = obj.panel
    ii = mesh.Hp + obj.i
    jj = mesh.Hp + obj.j
    backend = get_backend(lambda_panels[p])
    kernel! = _seed_layer_objective_kernel!(backend, 1)
    kernel!(lambda_panels[p], final_m[p], one(FT), Int32(ii), Int32(jj), Int32(obj.level);
            ndrange=1)
    synchronize(backend)
    return nothing
end

function _seed_objective!(lambda_panels, obj::CSColumnMeanObjective, final_m,
                          mesh::CubedSphereMesh)
    FT = eltype(lambda_panels[1])
    @inbounds for p in 1:6
        fill!(lambda_panels[p], zero(FT))
    end
    p = obj.panel
    ii = mesh.Hp + obj.i
    jj = mesh.Hp + obj.j
    denom = sum(@view final_m[p][ii, jj, :])
    backend = get_backend(lambda_panels[p])
    kernel! = _seed_column_objective_kernel!(backend, 256)
    kernel!(lambda_panels[p], final_m[p], FT(denom), Int32(ii), Int32(jj), Int32(mesh.Hp);
            ndrange=size(final_m[p], 3))
    synchronize(backend)
    return nothing
end

function _seed_objective!(lambda_panels, ::CSSeedObjective, final_m,
                          mesh::CubedSphereMesh)
    throw(ArgumentError(
        "`CSSeedObjective` is reserved for explicit final adjoint seeds; " *
        "use `cs_surface_emission_footprint_from_seed(final_adjoint_rm, ...)`"))
end

@kernel function _accumulate_surface_footprint_kernel!(footprint, @Const(lambda), dt, Hp, Nz)
    i, j = @index(Global, NTuple)
    @inbounds footprint[i, j] = dt * lambda[i + Hp, j + Hp, Nz]
end

function _accumulate_surface_footprint!(footprint, lambda_panels, dt, mesh::CubedSphereMesh)
    Hp = mesh.Hp
    Nz = size(lambda_panels[1], 3)
    @inbounds for p in 1:6
        backend = get_backend(lambda_panels[p])
        kernel! = _accumulate_surface_footprint_kernel!(backend, (16, 16))
        kernel!(footprint[p], lambda_panels[p], eltype(lambda_panels[p])(dt), Int32(Hp), Int32(Nz);
                ndrange=(mesh.Nc, mesh.Nc))
        synchronize(backend)
    end
    return nothing
end
