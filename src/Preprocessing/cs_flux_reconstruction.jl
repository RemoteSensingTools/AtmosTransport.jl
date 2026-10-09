# CS preprocessing: face mass fluxes reconstructed from cell-center winds.
# Split from cs_transport_helpers.jl (refactor phase 4); included by Preprocessing.jl in this order.

# ---------------------------------------------------------------------------
# CS flux reconstruction from cell-center winds
# ---------------------------------------------------------------------------

"""
Lengths of the panel faces that multiply the face-normal wind in the flux
reconstruction.

- `CellCenterlineLengths(Δx, Δy)`: the centerline width of the cell on one
  side of the face (`mesh.Δx`, `mesh.Δy`), the historical choice. These differ
  from the face's own length by up to ±0.7 % at C90, which adds a spurious
  divergence of about 1e-3 of the face flux.
- `EdgeLengths(Lx, Ly)`: the great-circle length of the face itself
  (`cs_face_edge_lengths(mesh)`).
"""
abstract type AbstractFaceLengths end
struct CellCenterlineLengths{A} <: AbstractFaceLengths
    Δx :: A
    Δy :: A
end
struct EdgeLengths{A} <: AbstractFaceLengths
    Lx :: A      # (Nc + 1) × Nc
    Ly :: A      # Nc × (Nc + 1)
end
EdgeLengths(mesh::CubedSphereMesh) = EdgeLengths(cs_face_edge_lengths(mesh)...)

# Length of the x-face `i` (between cells i − 1 and i) of row `j`, and of the
# y-face `j` of column `i`. Interior faces take the cell on the high side,
# panel-edge faces the adjacent cell.
@inline _xface_length(l::CellCenterlineLengths, i, j, Nc) = l.Δy[min(i, Nc), j]
@inline _yface_length(l::CellCenterlineLengths, i, j, Nc) = l.Δx[i, min(j, Nc)]
@inline _xface_length(l::EdgeLengths, i, j, Nc) = l.Lx[i, j]
@inline _yface_length(l::EdgeLengths, i, j, Nc) = l.Ly[i, j]

"""
    reconstruct_cs_fluxes!(am_panels, bm_panels, u_cs, v_cs, dp_panels,
                            ps_panels, A_ifc, B_ifc, Δx, Δy,
                            gravity, dt_factor, Nc, Nz;
                            face_lengths = CellCenterlineLengths(Δx, Δy))

Reconstruct per-panel horizontal mass fluxes from regridded cell-center winds,
with the layer thickness `dp = A_{k+1} − A_k + (B_{k+1} − B_k) ps` of each cell
(see [`cs_face_fluxes!`](@ref)).
"""
function reconstruct_cs_fluxes!(am_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                 bm_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                 u_cs::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                 v_cs::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                 dp_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                 ps_panels::NTuple{CS_PANEL_COUNT, Matrix{FT}},
                                 A_ifc::AbstractVector, B_ifc::AbstractVector,
                                 Δx, Δy, gravity::FT, dt_factor::FT,
                                 Nc::Int, Nz::Int;
                                 face_lengths::AbstractFaceLengths = CellCenterlineLengths(Δx, Δy)) where FT
    fill_cs_layer_thickness!(dp_panels, ps_panels, A_ifc, B_ifc, Nc, Nz)
    return cs_face_fluxes!(am_panels, bm_panels, u_cs, v_cs, dp_panels, face_lengths,
                           gravity, dt_factor, Nc, Nz)
end

"""
    fill_cs_layer_thickness!(dp_panels, ps_panels, A_ifc, B_ifc, Nc, Nz)

Layer pressure thickness `|A_k − A_{k+1} + (B_k − B_{k+1}) ps|` of every cell.
"""
function fill_cs_layer_thickness!(dp_panels, ps_panels, A_ifc, B_ifc, Nc, Nz)
    FT = eltype(dp_panels[1])
    for p in 1:CS_PANEL_COUNT
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            dp_panels[p][i, j, k] = abs(FT(A_ifc[k] - A_ifc[k + 1]) +
                                        FT(B_ifc[k] - B_ifc[k + 1]) * ps_panels[p][i, j])
        end
    end
    return dp_panels
end

"""
    cs_face_fluxes!(am_panels, bm_panels, u_cs, v_cs, dp_panels, face_lengths,
                    gravity, dt_factor, Nc, Nz)

Face mass fluxes from cell-center face-normal winds and layer thicknesses:
the two adjacent cells are averaged and multiplied by the face length,

    am[i,j,k] = ū × d̄p × L_x[i,j] / g × dt_factor
    bm[i,j,k] = v̄ × d̄p × L_y[i,j] / g × dt_factor

Panel-edge faces use the adjacent cell's values; CS edge mirrors are
synchronized before the binary is written.
"""
function cs_face_fluxes!(am_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                         bm_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                         u_cs::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                         v_cs::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                         dp_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                         face_lengths::AbstractFaceLengths,
                         gravity::FT, dt_factor::FT, Nc::Int, Nz::Int) where FT
    L = face_lengths
    for p in 1:CS_PANEL_COUNT
        u, v, dp = u_cs[p], v_cs[p], dp_panels[p]
        @inbounds for k in 1:Nz, j in 1:Nc
            am_panels[p][1, j, k] = u[1, j, k] * dp[1, j, k] * _xface_length(L, 1, j, Nc) / gravity * dt_factor
            for i in 2:Nc
                u_face  = FT(0.5) * (u[i - 1, j, k] + u[i, j, k])
                dp_face = FT(0.5) * (dp[i - 1, j, k] + dp[i, j, k])
                am_panels[p][i, j, k] = u_face * dp_face * _xface_length(L, i, j, Nc) / gravity * dt_factor
            end
            am_panels[p][Nc + 1, j, k] = u[Nc, j, k] * dp[Nc, j, k] *
                                         _xface_length(L, Nc + 1, j, Nc) / gravity * dt_factor
        end
        @inbounds for k in 1:Nz, i in 1:Nc
            bm_panels[p][i, 1, k] = v[i, 1, k] * dp[i, 1, k] * _yface_length(L, i, 1, Nc) / gravity * dt_factor
            for j in 2:Nc
                v_face  = FT(0.5) * (v[i, j - 1, k] + v[i, j, k])
                dp_face = FT(0.5) * (dp[i, j - 1, k] + dp[i, j, k])
                bm_panels[p][i, j, k] = v_face * dp_face * _yface_length(L, i, j, Nc) / gravity * dt_factor
            end
            bm_panels[p][i, Nc + 1, k] = v[i, Nc, k] * dp[i, Nc, k] *
                                         _yface_length(L, i, Nc + 1, Nc) / gravity * dt_factor
        end
    end
    return nothing
end

"""
    OuterCellPair

The outer cells of FV3's fourth-order stencil `(−1, 9, 9, −1)/16` across a face
(see [`CSVectorFaceGeometry`](@ref)): for face `f` with `slot[f] ≠ 0`, the cells
`cell[slot[f]]` two rows from the face and their weighted face-normal
coefficients `coef[slot[f]]` (the `-1/16` terms).
"""
struct OuterCellPair
    slot :: Vector{Int32}
    cell :: Vector{NTuple{6, Int32}}
    coef :: Vector{NTuple{4, Float64}}
end

"""
    AlongFaceFilter

The filter that GCHP's restaggering of A-grid winds applies along each face
(`A2D2C` in `GEOS_FV3_Utilities.F90`, then `d2a2c_vect` in `sw_core.F90`). The
D-grid wind is the two-cell average; the fourth-order D → A step then returns
the A-grid wind smoothed along the face direction, `(−1, 8, 18, 8, −1)/32` in the
panel interior and `(1, 2, 1)/4` within three cells of a panel edge. It removes
the 2Δ wave along the face and keeps 62% of the 4Δ wave.

It acts here on the face-normal winds of the faces on the same grid line of the
face's panel: face `f` becomes `Σₘ w[f][m] uₙ(face[f][m])`. Two approximations
of FV3: the stencil is chosen per face rather than per contributing cell, and
the faces in the first and last row of a panel, whose stencil would leave the
panel, are not filtered.
"""
struct AlongFaceFilter
    face :: Vector{NTuple{5, Int32}}
    w    :: Vector{NTuple{5, Float64}}
end

"""
    CSVectorFaceGeometry(mesh, face_table; order = 2, along_face_filter = false)

Geometry for building face fluxes from cell-centre winds as vectors, for every
unique face of `face_table` (interior and panel-seam faces):

- `len`: the face's great-circle length;
- `cell`: its two cells `(pl, il, jl, pr, ir, jr)`, lower-index cell first;
- `coef`: the face-normal components of the two cells' east and north unit
  vectors, times the weights that interpolate to the face midpoint (inverse
  great-circle distance from the two cell centres). The face-normal wind is
  `coef[1] u_l + coef[2] v_l + coef[3] u_r + coef[4] v_r` with `u` east and `v`
  north, and the layer thickness `dpw[1] dp_l + dpw[2] dp_r`.

At a panel seam the grid lines kink and the cells on both sides are skewed
along the edge, so the point between the two cell centres lies up to a quarter
of a face length along the edge from the face midpoint (2e-4 in the panel
interior). As in FV3's cube-edge treatment, a seam face also takes the value of
the neighbouring seam face on the side of its midpoint, projected on its own
normal, and interpolates along the edge with weight `w_partner`. These partner
data are stored for the seam faces only; `slot[f]` indexes them (0 for interior
faces and for a seam face whose midpoint needs no correction).

With `order = 4` the wind at interior faces with two cells on each side in the
same panel row is interpolated with FV3's interior fourth-order stencil
`(−1, 9, 9, −1)/16` (`outer`, a [`OuterCellPair`](@ref)); the layer thickness
stays linear. The faces next to a seam (index 2 and `Nc`) and the seam faces
stay second order. The cells between a second- and a fourth-order face keep
an O(h²) divergence error, 1e-5 of `U Δx` for a solid-body rotation at C90
(the linear construction leaves 1e-7). FV3 avoids it with one-sided stencils
next to the seams and fourth-order interpolation across them
(`edge_interpolate4`). `along_face_filter = true` adds GCHP's filter along the faces
(`filter`, an [`AlongFaceFilter`](@ref)). Either is `nothing` when off.
"""
struct CSVectorFaceGeometry{O <: Union{Nothing, OuterCellPair}, F <: Union{Nothing, AlongFaceFilter}}
    len       :: Vector{Float64}
    cell      :: Vector{NTuple{6, Int32}}
    coef      :: Vector{NTuple{4, Float64}}
    dpw       :: Vector{NTuple{2, Float64}}
    slot      :: Vector{Int32}
    w_partner :: Vector{Float64}
    pcell     :: Vector{NTuple{6, Int32}}
    pcoef     :: Vector{NTuple{4, Float64}}
    pdpw      :: Vector{NTuple{2, Float64}}
    outer     :: O
    filter    :: F
end

# Small 3-vector helpers on tuples.
@inline _dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
@inline _cross3(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
@inline _unit3(a) = a ./ sqrt(_dot3(a, a))
# Angle between unit vectors from their chord (accurate at small angles).
@inline _arc3(a, b) = 2 * asin(min(1.0, sqrt(_dot3(a .- b, a .- b)) / 2))

@inline function _cs_cell_ijp(c, Nc)
    p, r = divrem(c - 1, Nc * Nc)
    j, i = divrem(r, Nc)
    return i + 1, j + 1, p + 1
end

# Cell centres and local east/north unit vectors, by global cell index.
function _cs_cell_bases(mesh::CubedSphereMesh)
    Nc = mesh.Nc
    centre, east, north = (Vector{NTuple{3, Float64}}(undef, 6Nc^2) for _ in 1:3)
    for p in 1:6
        lon, lat = panel_cell_center_lonlat(mesh, p)
        for j in 1:Nc, i in 1:Nc
            c = _cs_global_cell(i, j, p, Nc)
            λ, φ = deg2rad(Float64(lon[i, j])), deg2rad(Float64(lat[i, j]))
            centre[c] = (cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ))
            east[c]   = (-sin(λ), cos(λ), 0.0)
            north[c]  = (-sin(φ) * cos(λ), -sin(φ) * sin(λ), cos(φ))
        end
    end
    return centre, east, north
end

# The two corners of face `f` in its canonical panel.
function _cs_face_corners(mesh, ft, f)
    p, i, j = Int(ft.face_panel[f]), Int(ft.face_idx_i[f]), Int(ft.face_idx_j[f])
    b = ft.face_dir[f] == 1 ? cs_corner_xyz(mesh, i, j + 1, p) : cs_corner_xyz(mesh, i + 1, j, p)
    return cs_corner_xyz(mesh, i, j, p), b
end

function CSVectorFaceGeometry(mesh::CubedSphereMesh, ft::CSGlobalFaceTable; order::Integer = 2,
                              along_face_filter::Bool = false)
    order in (2, 4) || throw(ArgumentError("CSVectorFaceGeometry: order must be 2 or 4, got $order"))
    Nc, R = mesh.Nc, Float64(mesh.radius)
    centre, east, north = _cs_cell_bases(mesh)
    corners = [_cs_face_corners(mesh, ft, f) for f in 1:ft.nf]
    mid = [_unit3(a .+ b) for (a, b) in corners]
    normal = map(1:ft.nf) do f
        n = _unit3(_cross3(corners[f]...))
        _dot3(n, centre[ft.face_right[f]] .- centre[ft.face_left[f]]) >= 0 ? n : .-n
    end
    cells(g) = (Int(ft.face_left[g]), Int(ft.face_right[g]))
    function weights(g)
        l, r = cells(g)
        d_l, d_r = _arc3(centre[l], mid[g]), _arc3(centre[r], mid[g])
        return (d_r / (d_l + d_r), d_l / (d_l + d_r))
    end
    function cell_tuple(g)
        (il, jl, pl), (ir, jr, pr) = _cs_cell_ijp(cells(g)[1], Nc), _cs_cell_ijp(cells(g)[2], Nc)
        return Int32.((pl, il, jl, pr, ir, jr))
    end
    function coefficients(g, n)    # cells of face g, projected on normal n
        (l, r), (wl, wr) = cells(g), weights(g)
        return (wl * _dot3(east[l], n), wl * _dot3(north[l], n), wr * _dot3(east[r], n), wr * _dot3(north[r], n))
    end

    partner, w_partner = _seam_partners(ft, corners, mid, centre)
    nf = ft.nf
    len = [R * _arc3(corners[f]...) for f in 1:nf]
    cell = [cell_tuple(f) for f in 1:nf]
    coef = [coefficients(f, normal[f]) for f in 1:nf]
    dpw = [weights(f) for f in 1:nf]
    corrected = findall(!=(0), partner)
    slot = zeros(Int32, nf); slot[corrected] .= 1:length(corrected)
    pcell = [cell_tuple(partner[f]) for f in corrected]
    pcoef = [coefficients(partner[f], normal[f]) for f in corrected]
    pdpw = [weights(partner[f]) for f in corrected]

    outer = order == 4 ? _outer_cell_pair!(coef, ft, cells, normal, east, north, Nc) : nothing
    filter = along_face_filter ? AlongFaceFilter(ft) : nothing
    return CSVectorFaceGeometry(len, cell, coef, dpw, slot, w_partner[corrected], pcell, pcoef, pdpw,
                                outer, filter)
end

# Fourth-order stencil across the interior faces with two cells on each side in
# the panel row: the inner pair's coefficients become 9/16, the outer pair −1/16.
function _outer_cell_pair!(coef, ft, cells, normal, east, north, Nc)
    proj(c, n, w) = (w * _dot3(east[c], n), w * _dot3(north[c], n))
    faces = [f for f in 1:ft.nf if ft.mirror_panel[f] == 0 &&
             3 <= (ft.face_dir[f] == 1 ? ft.face_idx_i[f] : ft.face_idx_j[f]) <= Nc - 1]
    slot = zeros(Int32, ft.nf); slot[faces] .= 1:length(faces)
    cell, outer_coef = NTuple{6, Int32}[], NTuple{4, Float64}[]
    for f in faces
        p, i, j = Int(ft.face_panel[f]), Int(ft.face_idx_i[f]), Int(ft.face_idx_j[f])
        (il, jl), (ir, jr) = ft.face_dir[f] == 1 ? ((i - 2, j), (i + 1, j)) : ((i, j - 2), (i, j + 1))
        l, r = cells(f)
        coef[f] = (proj(l, normal[f], 9 / 16)..., proj(r, normal[f], 9 / 16)...)
        push!(cell, Int32.((p, il, jl, p, ir, jr)))
        push!(outer_coef, (proj(_cs_global_cell(il, jl, p, Nc), normal[f], -1 / 16)...,
                           proj(_cs_global_cell(ir, jr, p, Nc), normal[f], -1 / 16)...))
    end
    return OuterCellPair(slot, cell, outer_coef)
end

# Panel-local position (panel, direction, i, j) of a face's canonical entry.
_face_key(ft, f) = (Int(ft.face_panel[f]), Int(ft.face_dir[f]), Int(ft.face_idx_i[f]), Int(ft.face_idx_j[f]))

# The faces along one panel edge are all canonical in the same panel, so every
# neighbour on a face's grid line is a canonical face of the face's own panel.
function AlongFaceFilter(ft::CSGlobalFaceTable)
    Nc = ft.Nc
    at = Dict(_face_key(ft, f) => f for f in 1:ft.nf)   # canonical position → face
    s5, s3, none = (-1, 8, 18, 8, -1) ./ 32, (0, 1, 2, 1, 0) ./ 4, (0.0, 0.0, 1.0, 0.0, 0.0)
    face, w = Vector{NTuple{5, Int32}}(undef, ft.nf), Vector{NTuple{5, Float64}}(undef, ft.nf)
    for f in 1:ft.nf
        p, d, i, j = _face_key(ft, f)
        along, across = d == 1 ? (j, i) : (i, j)          # position along and across the grid line
        w[f] = 4 <= along <= Nc - 3 && 5 <= across <= Nc - 3 ? s5 :
               2 <= along <= Nc - 1 ? s3 : none
        face[f] = ntuple(5) do m
            a = clamp(along + m - 3, 1, Nc)               # out-of-panel entries have weight 0
            Int32(at[d == 1 ? (p, d, i, a) : (p, d, a, j)])
        end
    end
    return AlongFaceFilter(face, w)
end

# Along-edge interpolation partners of the panel-seam faces (see `CSVectorFaceGeometry`).
function _seam_partners(ft, corners, mid, centre)
    key(x) = round.(Int, x .* 1e9)              # integer keys: no signed zeros
    seam_faces = [f for f in 1:ft.nf if ft.mirror_panel[f] != 0]
    by_corner = Dict{NTuple{3, Int}, Vector{Int}}()
    for f in seam_faces, c in corners[f]
        push!(get!(by_corner, key(c), Int[]), f)
    end
    between(g) = _unit3(centre[ft.face_left[g]] .+ centre[ft.face_right[g]])
    partner, w_partner = zeros(Int, ft.nf), zeros(ft.nf)
    for f in seam_faces
        plane = _unit3(_cross3(corners[f]...))
        F, P = mid[f], between(f)
        for c in corners[f], g in by_corner[key(c)]
            g == f && continue
            other = key(corners[g][1]) == key(c) ? corners[g][2] : corners[g][1]
            abs(_dot3(plane, other)) < 1e-9 || continue                  # same seam (great circle)
            Pg = between(g)
            _dot3(F .- P, Pg .- P) > 0 || continue                        # on the midpoint's side
            partner[f], w_partner[f] = g, _arc3(P, F) / (_arc3(P, F) + _arc3(Pg, F))
            break
        end
    end
    return partner, w_partner
end

# Face-normal wind and layer thickness from a face's two cells at level `k`.
@inline _face_wind((pl, il, jl, pr, ir, jr), (cl_e, cl_n, cr_e, cr_n), u_east, v_north, k) =
    @inbounds cl_e * u_east[pl][il, jl, k] + cl_n * v_north[pl][il, jl, k] +
              cr_e * u_east[pr][ir, jr, k] + cr_n * v_north[pr][ir, jr, k]
@inline function _face_value(cell, coef, (wl, wr), u_east, v_north, dp_panels, k)
    pl, il, jl, pr, ir, jr = cell
    dp = @inbounds wl * dp_panels[pl][il, jl, k] + wr * dp_panels[pr][ir, jr, k]
    return _face_wind(cell, coef, u_east, v_north, k), dp
end

"""
    cs_vector_face_fluxes!(am_panels, bm_panels, u_east, v_north, dp_panels,
                           geom, face_table, gravity, dt_factor, Nc, Nz)

Face mass fluxes from cell-centre east/north winds, treated as vectors (see
[`CSVectorFaceGeometry`](@ref)): the two cells' winds, interpolated to the face
midpoint and projected onto the face normal, times the interpolated layer
thickness and the face's great-circle length,

    F = u_n × dp × L / g × dt_factor,

with the along-edge interpolation at panel seams. Mirror entries of the seam
faces are written from the canonical ones with their signs (`_sync_cs_mirrors!`).
"""
function cs_vector_face_fluxes!(am_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                bm_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                u_east::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                v_north::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                dp_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                geom::CSVectorFaceGeometry, ft::CSGlobalFaceTable,
                                gravity::FT, dt_factor::FT, Nc::Int, Nz::Int) where FT
    scale = Float64(dt_factor) / Float64(gravity)
    un, dp = zeros(ft.nf), zeros(ft.nf)              # one level of face values
    @inbounds for k in 1:Nz
        for f in 1:ft.nf
            un[f], dp[f] = _vector_face_value(geom, f, u_east, v_north, dp_panels, k)
        end
        for f in 1:ft.nf
            p, i, j = Int(ft.face_panel[f]), Int(ft.face_idx_i[f]), Int(ft.face_idx_j[f])
            target = ft.face_dir[f] == 1 ? am_panels[p] : bm_panels[p]
            target[i, j, k] = FT(_filtered_wind(geom.filter, un, f) * dp[f] * geom.len[f] * scale)
        end
    end
    _sync_cs_mirrors!(am_panels, bm_panels, ft, Nz)
    return nothing
end

@inline _filtered_wind(::Nothing, un, f) = @inbounds un[f]
@inline _filtered_wind(F::AlongFaceFilter, un, f) =
    @inbounds sum(F.w[f][m] * un[F.face[f][m]] for m in 1:5)

# Adds the outer pair of the fourth-order stencil (no-op without one).
@inline _with_outer(::Nothing, un, f, u_east, v_north, k) = un
@inline function _with_outer(o::OuterCellPair, un, f, u_east, v_north, k)
    s = @inbounds o.slot[f]
    return s == 0 ? un : un + @inbounds _face_wind(o.cell[s], o.coef[s], u_east, v_north, k)
end

# Face-normal wind and layer thickness at the midpoint of face `f`, level `k`.
@inline function _vector_face_value(geom::CSVectorFaceGeometry, f, u_east, v_north, dp_panels, k)
    un, dp = _face_value(geom.cell[f], geom.coef[f], geom.dpw[f], u_east, v_north, dp_panels, k)
    un = _with_outer(geom.outer, un, f, u_east, v_north, k)
    s = geom.slot[f]
    if s != 0      # seam face: interpolate along the edge to the face midpoint
        w = geom.w_partner[s]
        un_g, dp_g = _face_value(geom.pcell[s], geom.pcoef[s], geom.pdpw[s], u_east, v_north, dp_panels, k)
        un, dp = (1 - w) * un + w * un_g, (1 - w) * dp + w * dp_g
    end
    return un, dp
end

# ---------------------------------------------------------------------------
# (REMOVED) Per-level mass consistency correction
#
# The old `_enforce_perlevel_mass_consistency!` applied a uniform additive
# offset per level after regridding `m` directly through `apply_regridder!`.
# It made the per-level *sum* match the source but did not fix the spatial
# distortion introduced by treating an extensive field (`m` in kg/cell) as
# intensive (kg/m²). On the LL→C180 path that distortion produced ~12×
# polar mass deficits.
#
# The fix is now upstream: callers regrid `m` via
# `regrid_3d_to_cs_panels!(..., ExtensiveCellField())`, which converts
# through density on both sides and preserves the spatial distribution by
# construction. With that path in place the band-aid is unnecessary and
# was removed (see commit history for the previous implementation).
# ---------------------------------------------------------------------------
