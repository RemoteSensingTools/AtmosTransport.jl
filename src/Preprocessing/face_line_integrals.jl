# ---------------------------------------------------------------------------
# Cube-face mass fluxes as line integrals of the source-grid flow.
#
# The horizontal mass flux through a cube face f in one layer is
#
#     F_f = (Δt / g) ∫_f (V · N_f) Δp dl,
#
# with V the horizontal wind, N_f the unit normal of the face and Δp the layer
# thickness. Building F_f from cell-centre winds interpolated to the face
# (`VectorFaceFluxes`) smooths the flow at the cube grid scale: on ERA5 the
# vertical motion diagnosed from such fluxes has a regression slope of
# 0.73–0.75 against ERA5's own ω per C90 cell, but 0.92–0.94 on 3 × 3 blocks.
# Integrating the source flow along each face keeps the convergence of every
# cube cell at the source resolution instead (divergence theorem): slope
# 0.99–1.00 and RMSE 3 % of ERA5's ω at 10–71 hPa (2022-01-15; see
# docs/src/theory/vertical_transport.md, section 1).
# ---------------------------------------------------------------------------

"""
    LineIntegralFaceFluxes(source_mesh, target_mesh, face_table; samples_per_face = 16)

Face mass fluxes integrated along each cube face from the source-grid winds,

    F_f = (Δt / g) ∫_f (V · N_f) Δp dl ≈ (Δt / g) (ℓ_f / n) Σ_q N_f · (V Δp)(x_q).

A cube face is a great-circle arc, so its unit normal `N_f` (oriented from the
face's left to its right cell, as in [`CSVectorFaceGeometry`](@ref)) is the same
3-vector at every point of the arc. The integrand is sampled at `n =
samples_per_face` equally spaced midpoints `x_q` of the arc of length `ℓ_f`,
where the Cartesian components of `V Δp` are interpolated bilinearly from the
source cells. The integral is therefore a fixed linear map of the source fields,
stored as one sparse matrix `L = [L_X L_Y L_Z]` (faces × 3 source cells),

    F = (Δt / g) L [X Δp; Y Δp; Z Δp],

with `X = −sin λ u − sin φ cos λ v`, `Y = cos λ u − sin φ sin λ v`,
`Z = cos φ v` the Cartesian wind components of each source cell. Cartesian
components are smooth across the poles, unlike `u` and `v`.

Only a `ReducedGaussianMesh` source (ERA5 N320) is implemented. With 16 samples
a C90 face (≤ 1.2°) is sampled every ≤ 0.08°, about four times per N320 cell.
"""
struct LineIntegralFaceFluxes
    L          :: SparseMatrixCSC{Float64, Int}    # (n_faces, 3 n_source_cells): X, Y, Z blocks
    face_table :: CSGlobalFaceTable
    src_trig   :: NamedTuple{(:sinλ, :cosλ, :sinφ, :cosφ), NTuple{4, Vector{Float64}}}
    src_flux   :: Vector{Vector{Float64}}          # per thread: [X Δp; Y Δp; Z Δp] of one level
    face_flux  :: Vector{Vector{Float64}}          # per thread: F of one level
end

function LineIntegralFaceFluxes(source::ReducedGaussianMesh, target::CubedSphereMesh,
                                ft::CSGlobalFaceTable; samples_per_face::Integer = 16)
    samples_per_face >= 1 || throw(ArgumentError("samples_per_face must be ≥ 1, got $samples_per_face"))
    R = Float64(target.radius)
    nsrc = ncells(source)
    centre, _, _ = _cs_cell_bases(target)
    I, J, V = Int[], Int[], Float64[]
    for f in 1:ft.nf
        a, b = _cs_face_corners(target, ft, f)
        n = _unit3(_cross3(a, b))                       # normal of the face's great circle,
        s = _dot3(n, centre[ft.face_right[f]] .- centre[ft.face_left[f]]) >= 0 ? 1 : -1   # pointing left → right
        θ = _arc3(a, b)
        w = s * R * θ / samples_per_face                # arc length per sample (midpoint rule)
        for q in 1:samples_per_face
            x = _slerp3(a, b, θ, (q - 0.5) / samples_per_face)
            for (c, wc) in _point_weights(source, x), d in 1:3
                push!(I, f); push!(J, (d - 1) * nsrc + c); push!(V, w * wc * n[d])
            end
        end
    end
    lon, lat = _source_cell_centers(source)
    nt = Threads.maxthreadid()
    return LineIntegralFaceFluxes(sparse(I, J, V, ft.nf, 3nsrc),   # repeated (face, cell) pairs add up
                                  ft, _trig(deg2rad.(Float64.(lon)), deg2rad.(Float64.(lat))),
                                  [zeros(3nsrc) for _ in 1:nt], [zeros(ft.nf) for _ in 1:nt])
end

# Point at fraction t of the great-circle arc from unit vector a to b (θ = arc).
@inline _slerp3(a, b, θ, t) = (sin((1 - t) * θ) .* a .+ sin(t * θ) .* b) ./ sin(θ)

"""
    _point_weights(mesh::ReducedGaussianMesh, x) -> NTuple{4, Tuple{Int, Float64}}

Bilinear interpolation weights at the unit vector `x` from the cell centres of a
reduced Gaussian mesh: linear in longitude along the two rings that bracket the
latitude (cell centres at `(i − ½) Δλ`), then linear in latitude between them.
Poleward of the outermost ring the outermost ring is used alone.
"""
function _point_weights(m::ReducedGaussianMesh, x)
    λ = mod(atand(x[2], x[1]), 360.0)
    φ = asind(clamp(x[3], -1.0, 1.0))
    lats = m.latitudes
    j = clamp(searchsortedlast(lats, φ), 1, length(lats) - 1)
    t = clamp((φ - Float64(lats[j])) / (Float64(lats[j + 1]) - Float64(lats[j])), 0.0, 1.0)
    (c1, c2), (w1, w2) = _ring_weights(m, j, λ)
    (c3, c4), (w3, w4) = _ring_weights(m, j + 1, λ)
    return ((c1, (1 - t) * w1), (c2, (1 - t) * w2), (c3, t * w3), (c4, t * w4))
end

# Linear interpolation in longitude within ring j: the two bracketing cells and their weights.
function _ring_weights(m::ReducedGaussianMesh, j, λ)
    n = m.nlon_per_ring[j]
    s = λ * n / 360 - 0.5                               # position in cell-centre units
    i0 = floor(Int, s)
    t = s - i0
    off = m.ring_offsets[j]
    return (off + mod(i0, n), off + mod(i0 + 1, n)), (1 - t, t)
end

"""
    line_integral_face_fluxes!(am, bm, m::LineIntegralFaceFluxes, u, v, ps, A, B,
                               gravity, dt_factor, Nz)

Fill the face mass fluxes `am`, `bm` (per panel, levels `1:Nz`) with the line
integrals of [`LineIntegralFaceFluxes`](@ref), from the source east/north winds
`u`, `v` (source cell × level) and surface pressure `ps` (Pa). The layer
thickness of source cell `c` is `Δp = |ΔA_k + ΔB_k ps_c|` from the hybrid
interfaces `A`, `B`, as for the cell-centre flux methods. Mirror entries of the
panel-seam faces are written from the canonical ones.
"""
function line_integral_face_fluxes!(am::NTuple{6, Array{FT, 3}}, bm::NTuple{6, Array{FT, 3}},
                                    m::LineIntegralFaceFluxes, u::AbstractMatrix, v::AbstractMatrix,
                                    ps::AbstractVector, A, B, gravity, dt_factor, Nz::Int) where FT
    ft = m.face_table
    (; sinλ, cosλ, sinφ, cosφ) = m.src_trig
    size(u) == size(v) && size(u, 1) == length(ps) == length(sinλ) && size(u, 2) >= Nz ||
        throw(DimensionMismatch("source winds $(size(u)), $(size(v)) and ps $(length(ps)) do not match the operator"))
    scale = Float64(dt_factor) / Float64(gravity)
    nsrc = length(ps)
    Threads.@threads :static for k in 1:Nz              # levels are independent
        XYZ = m.src_flux[Threads.threadid()]
        F = m.face_flux[Threads.threadid()]
        dA = Float64(A[k + 1]) - Float64(A[k])
        dB = Float64(B[k + 1]) - Float64(B[k])
        @inbounds for c in 1:nsrc                       # Cartesian components of V Δp per source cell
            dp = abs(dA + dB * Float64(ps[c]))
            uu, vv = Float64(u[c, k]) * dp, Float64(v[c, k]) * dp
            XYZ[c]         = -sinλ[c] * uu - sinφ[c] * cosλ[c] * vv    # X
            XYZ[nsrc + c]  =  cosλ[c] * uu - sinφ[c] * sinλ[c] * vv    # Y
            XYZ[2nsrc + c] =  cosφ[c] * vv                             # Z
        end
        mul!(F, m.L, XYZ)                               # ∫ (V · N) Δp dl for every face
        @inbounds for f in 1:ft.nf                      # canonical entry, positive from left to right cell
            p, i, j = Int(ft.face_panel[f]), Int(ft.face_idx_i[f]), Int(ft.face_idx_j[f])
            target = ft.face_dir[f] == 1 ? am[p] : bm[p]
            target[i, j, k] = FT(F[f] * scale)
        end
    end
    _sync_cs_mirrors!(am, bm, ft, Nz)
    return nothing
end
