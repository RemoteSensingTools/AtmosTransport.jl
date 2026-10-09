# CFL subcycling of the structured and face-indexed sweeps: pass counts and subcycled sweeps.
# Split from StrangSplitting.jl (refactor phase 4); included by Advection.jl in this order.

# =========================================================================
# Structured-grid CFL subcycling helpers
# =========================================================================

@inline function _subcycling_pass_count(max_cfl::FT, cfl_limit::FT) where FT
    cfl_limit > zero(FT) || throw(ArgumentError("structured advection requires cfl_limit > 0, got $(cfl_limit)"))
    return max(1, ceil(Int, max_cfl / cfl_limit))
end

function _horizontal_face_outgoing_ratio(horizontal_flux::AbstractArray{FT,2},
                                         m::AbstractArray{FT,2},
                                         mesh::AbstractHorizontalMesh) where FT
    nc, Nz = size(m)
    outgoing = zeros(FT, nc)
    max_ratio = zero(FT)

    @inbounds for k in 1:Nz
        outgoing .= zero(FT)
        for f in 1:nfaces(mesh)
            left, right = face_cells(mesh, f)
            if left > 0 && right > 0
                flux = horizontal_flux[f, k]
                if flux >= zero(FT)
                    outgoing[left] += flux
                else
                    outgoing[right] -= flux
                end
            end
        end
        max_ratio = max(max_ratio, maximum(outgoing ./ max.(m[:, k], eps(FT))))
    end

    return max_ratio
end

function _vertical_face_outgoing_ratio(cm::AbstractArray{FT,2},
                                       m::AbstractArray{FT,2}) where FT
    # cm is (nc, Nz+1). `cm[:, k]` is the flux through the TOP face of cell
    # (:, k) (positive = downward = inflow from above). `cm[:, k+1]` is the
    # flux through the BOTTOM face (positive = downward = outflow to below).
    # So per-cell OUTFLOW = upward through top (max(-cm[:,k], 0)) + downward
    # through bottom (max(cm[:,k+1], 0)). Single broadcast over all (:, :)
    # stays on device for GPU callers. Corrects an earlier bug
    # that summed inflow, not outflow (GPU and CPU saw CFL half the true
    # value in flows where inflow ≠ outflow).
    Nz = size(m, 2)
    out = max.(.- @view(cm[:, 1:Nz]), zero(FT)) .+ max.(@view(cm[:, 2:Nz+1]), zero(FT))
    return maximum(out ./ max.(m, eps(FT)))
end

# Face-indexed pilots — unified static algorithm.
#
# The horizontal path requires mesh connectivity (face_cells) which lives on
# CPU, so device arrays are materialized via Array(...) before the static
# ratio is computed. For realistic problem sizes the transfer is ~1–10 MB and
# happens once per sweep — far cheaper than the old evolving-mass iteration
# that transferred on every pilot pass.
#
# The vertical path is a pure broadcast reduction — stays on device.
function _horizontal_face_subcycling_pass_count(horizontal_flux::AbstractArray{FT,2},
                                                m::AbstractArray{FT,2},
                                                mesh::AbstractHorizontalMesh,
                                                ws::AdvectionWorkspace{FT},
                                                cfl_limit::FT; max_n_sub::Int = 4096) where FT
    isinf(cfl_limit) && return 1
    static_cfl = m isa Array ?
        _horizontal_face_outgoing_ratio(horizontal_flux, m, mesh) :
        _horizontal_face_outgoing_ratio(Array(horizontal_flux), Array(m), mesh)
    n_sub = _subcycling_pass_count(static_cfl, cfl_limit)
    n_sub <= max_n_sub || throw(ArgumentError("face-indexed horizontal subcycling exceeded max_n_sub=$(max_n_sub)"))
    return n_sub
end

function _vertical_face_subcycling_pass_count(cm::AbstractArray{FT,2},
                                              m::AbstractArray{FT,2},
                                              ws::AdvectionWorkspace{FT},
                                              cfl_limit::FT; max_n_sub::Int = 4096) where FT
    isinf(cfl_limit) && return 1
    static_cfl = _vertical_face_outgoing_ratio(cm, m)
    n_sub = _subcycling_pass_count(static_cfl, cfl_limit)
    n_sub <= max_n_sub || throw(ArgumentError("face-indexed vertical subcycling exceeded max_n_sub=$(max_n_sub)"))
    return n_sub
end

@inline function _sweep_horizontal_face_subcycled!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                                   horizontal_flux::AbstractArray{FT,2},
                                                   mesh::AbstractHorizontalMesh,
                                                   scheme::AbstractAdvectionScheme,
                                                   ws::AdvectionWorkspace{FT},
                                                   cfl_limit::FT) where FT
    n_sub = _horizontal_face_subcycling_pass_count(horizontal_flux, m, mesh, ws, cfl_limit)
    if n_sub == 1
        sweep_horizontal!(rm, m, horizontal_flux, mesh, scheme, ws)
        return 1
    end
    flux_scale = inv(FT(n_sub))
    for _ in 1:n_sub
        sweep_horizontal!(rm, m, horizontal_flux, mesh, scheme, ws, flux_scale)
    end
    return n_sub
end

@inline function _sweep_vertical_face_subcycled!(rm::AbstractArray{FT,2}, m::AbstractArray{FT,2},
                                                 cm::AbstractArray{FT,2},
                                                 scheme::AbstractAdvectionScheme,
                                                 ws::AdvectionWorkspace{FT},
                                                 cfl_limit::FT) where FT
    n_sub = _vertical_face_subcycling_pass_count(cm, m, ws, cfl_limit)
    if n_sub == 1
        sweep_vertical!(rm, m, cm, scheme, ws)
        return 1
    end
    flux_scale = inv(FT(n_sub))
    for _ in 1:n_sub
        sweep_vertical!(rm, m, cm, scheme, ws, flux_scale)
    end
    return n_sub
end

# Unified CFL pilot — single static algorithm for CPU and GPU.
#
# For each cell (i,j,k), CFL = total_outflow / cell_mass. Total outflow is
# the sum of flux leaving the cell through its two faces in the active
# direction; the maximum over all cells determines n_sub.
#
# For the structured x direction, face flux `am[face, j, k]` lives between
# cells (face-1, j, k) and (face, j, k):
#   positive am[i]   = rightward flow   → leaves cell i-1 through its right face
#   negative am[i]   = leftward flow    → leaves cell i   through its left face
# so outflow(cell i) = max(-am[i], 0) + max(am[i+1], 0). y and z are analogous
# (positive cm = downward; cell k's outflow is upward through cm[:,:,k] and
# downward through cm[:,:,k+1]).
#
# This replaces an earlier dual-path (CPU evolving-mass,
# GPU static-inflow) with one algorithm that (a) is backend-agnostic via
# pure broadcast, (b) computes OUTFLOW correctly (the prior GPU path
# summed inflow — a sign bug that under-estimated CFL on device).
function _x_subcycling_pass_count(am::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                  ws::AdvectionWorkspace{FT},
                                  cfl_limit::FT; max_n_sub::Int = 4096) where FT
    isinf(cfl_limit) && return 1
    Nx = size(m, 1)
    out = max.(.- @view(am[1:Nx, :, :]), zero(FT)) .+ max.(@view(am[2:Nx+1, :, :]), zero(FT))
    static_cfl = maximum(out ./ max.(m, eps(FT)))
    n_sub = _subcycling_pass_count(static_cfl, cfl_limit)
    n_sub <= max_n_sub || throw(ArgumentError("x-direction subcycling exceeded max_n_sub=$(max_n_sub)"))
    return n_sub
end

function _y_subcycling_pass_count(bm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                  ws::AdvectionWorkspace{FT},
                                  cfl_limit::FT; max_n_sub::Int = 4096) where FT
    isinf(cfl_limit) && return 1
    Ny = size(m, 2)
    out = max.(.- @view(bm[:, 1:Ny, :]), zero(FT)) .+ max.(@view(bm[:, 2:Ny+1, :]), zero(FT))
    static_cfl = maximum(out ./ max.(m, eps(FT)))
    n_sub = _subcycling_pass_count(static_cfl, cfl_limit)
    n_sub <= max_n_sub || throw(ArgumentError("y-direction subcycling exceeded max_n_sub=$(max_n_sub)"))
    return n_sub
end

function _z_subcycling_pass_count(cm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                  ws::AdvectionWorkspace{FT},
                                  cfl_limit::FT; max_n_sub::Int = 4096) where FT
    isinf(cfl_limit) && return 1
    Nz = size(m, 3)
    out = max.(.- @view(cm[:, :, 1:Nz]), zero(FT)) .+ max.(@view(cm[:, :, 2:Nz+1]), zero(FT))
    static_cfl = maximum(out ./ max.(m, eps(FT)))
    n_sub = _subcycling_pass_count(static_cfl, cfl_limit)
    n_sub <= max_n_sub || throw(ArgumentError("z-direction subcycling exceeded max_n_sub=$(max_n_sub)"))
    return n_sub
end
