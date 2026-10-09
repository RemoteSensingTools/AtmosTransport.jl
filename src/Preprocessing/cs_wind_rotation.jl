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
    rotate_winds_to_panel_local!(..., Nc, Nz)

Backward-compatible gnomonic wrapper. New preprocessing code should pass the
actual `CubedSphereMesh` so panel convention is explicit.
"""
function rotate_winds_to_panel_local!(u_panel::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       v_panel::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       u_east::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       v_north::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       Nc::Int, Nz::Int) where FT
    mesh = CubedSphereMesh(; Nc=Nc, FT=FT, radius=FT(IFS_EARTH_RADIUS),
                            convention=GnomonicPanelConvention())
    return rotate_winds_to_panel_local!(u_panel, v_panel, u_east, v_north, mesh, Nz)
end

# ---------------------------------------------------------------------------
# Panel-local → east/north wind rotation
# ---------------------------------------------------------------------------

"""
    rotate_panel_to_geographic!(u_east, v_north, u_panel, v_panel,
                                 tangent_basis, Nc, Nz)

Inverse of [`rotate_winds_to_panel_local!`](@ref): rotate cubed-sphere
face-normal wind components back to geographic `(east, north)` for all panels.

`u_panel` and `v_panel` are interpreted as normal velocities through local-x
and local-y faces, matching the `am`/`bm` reconstruction convention.  Because
the two face normals are non-orthogonal whenever the panel coordinate tangents
are non-orthogonal, the inverse solves the local two-vector Gram system rather
than applying a transpose.
"""
function rotate_panel_to_geographic!(u_east::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                      v_north::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                      u_panel::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                      v_panel::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
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
            denom = max(one(FT) - c * c, FT(eps(Float64)))
            s = sqrt(denom)
            nx_east = (xe - c * ye) / s
            nx_north = (xn - c * yn) / s
            ny_east = (ye - c * xe) / s
            ny_north = (yn - c * xn) / s
            @inbounds for k in 1:Nz
                up = u_panel[p][i, j, k]
                vp = v_panel[p][i, j, k]
                ax = (up + c * vp) / denom
                ay = (vp + c * up) / denom
                u_east[p][i, j, k] = ax * nx_east + ay * ny_east
                v_north[p][i, j, k] = ax * nx_north + ay * ny_north
            end
        end
    end
    return nothing
end

function rotate_panel_to_geographic!(u_east::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                      v_north::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                      u_panel::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                      v_panel::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                      mesh::CubedSphereMesh{FT}, Nz::Int) where FT
    tangent_basis = ntuple(p -> panel_cell_local_tangent_basis(mesh, p), CS_PANEL_COUNT)
    return rotate_panel_to_geographic!(u_east, v_north, u_panel, v_panel,
                                       tangent_basis, mesh.Nc, Nz)
end

# ---------------------------------------------------------------------------
# CS face fluxes → cell-center winds (peer of recover_ll_cell_center_winds!)
# ---------------------------------------------------------------------------

"""
    recover_cs_cell_center_winds!(u_cc, v_cc, am_v4, bm_v4, dp_panels,
                                   Δx, Δy, gravity, dt_factor, Nc, Nz)

Recover panel-local cell-center `(u, v)` wind components from v4
face-staggered mass fluxes. This is the CS peer of
[`recover_ll_cell_center_winds!`](@ref), used in the cross-topology
preprocessor (CS source → LL/RG target) right after
[`geos_native_to_face_flux!`](@ref) lays MFXC/MFYC into v4 layout.

For each cell `(i, j)` on each panel, average the two adjacent face
fluxes and divide by the cross-sectional area:

    u_cc[i, j, k] = ½ (am_v4[i, j, k] + am_v4[i+1, j, k])
                    / (Δy[i, j] · dp[i, j, k] / g · dt_factor)

    v_cc[i, j, k] = ½ (bm_v4[i, j, k] + bm_v4[i, j+1, k])
                    / (Δx[i, j] · dp[i, j, k] / g · dt_factor)

Boundary cells use the same averaging — the v4 layout already supplies
both the canonical face (i=Nc+1, j=Nc+1) and the halo face (i=1, j=1)
populated by `_propagate_cs_outflow_to_halo!`, so no special edge
treatment is needed. `Δx, Δy` are the per-cell face-length matrices
sourced from `mesh.Δx`, `mesh.Δy`.
"""
function recover_cs_cell_center_winds!(u_cc::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                        v_cc::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                        am_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                        bm_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                        dp_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                        Δx, Δy, gravity::FT, dt_factor::FT,
                                        Nc::Int, Nz::Int) where FT
    eps_area = FT(1e-10)
    for p in 1:CS_PANEL_COUNT
        ap, bp = am_v4[p], bm_v4[p]
        dp     = dp_panels[p]
        u, v   = u_cc[p], v_cc[p]
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            dpk     = dp[i, j, k]
            area_y  = FT(Δy[i, j]) * dpk / gravity * dt_factor
            area_x  = FT(Δx[i, j]) * dpk / gravity * dt_factor
            u[i, j, k] = area_y > eps_area ?
                FT(0.5) * (ap[i, j, k] + ap[i + 1, j, k]) / area_y :
                zero(FT)
            v[i, j, k] = area_x > eps_area ?
                FT(0.5) * (bp[i, j, k] + bp[i, j + 1, k]) / area_x :
                zero(FT)
        end
    end
    return nothing
end
