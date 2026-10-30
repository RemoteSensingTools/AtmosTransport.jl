# CS preprocessing: east/north ↔ panel-local wind rotation and face fluxes → cell-center winds.
# Split from cs_transport_helpers.jl (refactor phase 4); included by Preprocessing.jl in this order.

# ---------------------------------------------------------------------------
# East/north → panel-local wind rotation
# ---------------------------------------------------------------------------

"""
    rotate_winds_to_panel_local!(u_panel, v_panel, u_east, v_north,
                                  mesh, Nz)

Rotate geographic `(east, north)` wind components to cubed-sphere face-normal
components for all 6 panels.

`u_panel` is the wind normal to local-x faces, scaled with `Δy` when
reconstructing `am`; `v_panel` is the wind normal to local-y faces, scaled with
`Δx` when reconstructing `bm`.  The panel tangent directions on the gnomonic
cubed sphere are not generally orthogonal, so these are not simple dot products
onto the local x/y tangents.  The routine derives the two face normals from the
convention-aware tangent basis and projects onto those normals.
"""
function rotate_winds_to_panel_local!(u_panel::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       v_panel::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       u_east::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       v_north::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       tangent_basis::NTuple{CS_PANEL_COUNT, <:Any},
                                       Nc::Int, Nz::Int) where FT
    for p in 1:CS_PANEL_COUNT
        x_east, x_north, y_east, y_north = tangent_basis[p]
        for j in 1:Nc, i in 1:Nc
            xe = x_east[i, j]
            xn = x_north[i, j]
            ye = y_east[i, j]
            yn = y_north[i, j]
            c = clamp(xe * ye + xn * yn, -one(FT), one(FT))
            s = sqrt(max(one(FT) - c * c, FT(eps(Float64))))
            nx_east = (xe - c * ye) / s
            nx_north = (xn - c * yn) / s
            ny_east = (ye - c * xe) / s
            ny_north = (yn - c * xn) / s
            @inbounds for k in 1:Nz
                ue = u_east[p][i, j, k]
                vn = v_north[p][i, j, k]
                u_panel[p][i, j, k] = ue * nx_east + vn * nx_north
                v_panel[p][i, j, k] = ue * ny_east + vn * ny_north
            end
        end
    end
    return nothing
end

function rotate_winds_to_panel_local!(u_panel::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       v_panel::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       u_east::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       v_north::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       mesh::CubedSphereMesh{FT}, Nz::Int) where FT
    tangent_basis = ntuple(p -> panel_cell_local_tangent_basis(mesh, p), CS_PANEL_COUNT)
    return rotate_winds_to_panel_local!(u_panel, v_panel, u_east, v_north,
                                        tangent_basis, mesh.Nc, Nz)
end

"""
    recover_ll_cell_center_winds!(u_cc, v_cc, am_ll, bm_ll, ps_ll,
                                   A_ifc, B_ifc, lats_deg,
                                   Δy_ll, Δlon_ll, radius, gravity, dt_factor)

Recover cell-center u,v wind components from LL mass fluxes by dividing
by the cross-sectional area factor: `am / (Δy × dp/g × dt_factor)`.

`dt_factor = dt_met / (2 × steps_per_window)` is the flux scaling used
in the binary.
"""
function recover_ll_cell_center_winds!(u_cc::Array{FT, 3},
                                        v_cc::Array{FT, 3},
                                        am_ll::AbstractArray{FT, 3},
                                        bm_ll::AbstractArray{FT, 3},
                                        ps_ll::AbstractArray{<:Real, 2},
                                        A_ifc::AbstractVector, B_ifc::AbstractVector,
                                        lats_deg::AbstractVector,
                                        Δy_ll::FT, Δlon_ll::FT,
                                        radius::FT, gravity::FT,
                                        dt_factor::FT) where FT
    Nx, Ny, Nz = size(u_cc)

    # u from am: average face fluxes to cell centers, divide by area factor
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp = abs((A_ifc[k] - A_ifc[k + 1]) +
                 (B_ifc[k] - B_ifc[k + 1]) * ps_ll[i, j])
        area_factor = Δy_ll * dp / gravity * dt_factor
        u_cc[i, j, k] = area_factor > FT(1e-10) ?
            FT(0.5) * (am_ll[i, j, k] + am_ll[i + 1, j, k]) / area_factor :
            zero(FT)
    end

    # v from bm: average face fluxes to cell centers, divide by area factor
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        cos_lat = cosd(lats_deg[j])
        Δx_loc = radius * Δlon_ll * max(cos_lat, FT(1e-6))
        dp = abs((A_ifc[k] - A_ifc[k + 1]) +
                 (B_ifc[k] - B_ifc[k + 1]) * ps_ll[i, j])
        area_factor = Δx_loc * dp / gravity * dt_factor
        jn = j + 1
        v_cc[i, j, k] = area_factor > FT(1e-10) ?
            FT(0.5) * (bm_ll[i, j, k] + bm_ll[i, jn, k]) / area_factor :
            zero(FT)
    end

    return nothing
end
