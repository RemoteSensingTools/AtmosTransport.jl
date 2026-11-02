"""Regular, cell-centred refined rings for great-circle CS face quadrature."""
struct CSSpectralQuadrature{M}
    rings::SynthesisRings
    mesh::M
    areas::Vector{Float64}
    radius::Float64
    samples_per_face::Int
end

function CSSpectralQuadrature(grid::CubedSphereTargetGeometry, T::Integer, c::TM5SpectralFluxes)
    lx, ly = cs_face_edge_lengths(grid.mesh)
    edge = rad2deg(min(minimum(lx), minimum(ly))/grid.mesh.radius)
    refinement = quadrature_refinement(T, edge, c.refinement_steps)
    spacing = edge/refinement
    ny = max(2, ceil(Int, 180/spacing))
    nx = max(2ny, 2(T+1)+1)
    mesh = LatLonMesh(; Nx=nx, Ny=ny, FT=Float64, radius=grid.mesh.radius,
        longitude=(0, 360), latitude=(-90, 90))
    rings = SynthesisRings(deg2rad.(mesh.φᶜ), fill(nx, ny), fill(deg2rad(first(mesh.λᶜ)), ny))
    return CSSpectralQuadrature(rings, mesh, cell_areas(grid.mesh), Float64(grid.mesh.radius),
        max(16, 4refinement))
end

# CSC of the transpose provides rows for a portable gather, without atomics.
# Geometry is prepared on the host and transferred once, never per level.
function cs_gather_operator(matrix, adapter, ::Type{FT}) where FT
    rows = sparse(transpose(matrix))
    return (; offsets=adapter(Int32.(rows.colptr)), indices=adapter(Int32.(rows.rowval)),
        weights=adapter(FT.(rows.nzval)))
end

@kernel function cs_apply_integral_kernel!(out, values, offsets, indices, weights)
    c, k = @index(Global, NTuple)
    if c <= size(out, 1) && k <= size(out, 2)
        total = zero(eltype(out))
        for t in offsets[c]:offsets[c+1]-1
            total += weights[t]*values[indices[t], k]
        end
        out[c, k] = total
    end
end

function cs_apply_integral!(out, values, op)
    backend = get_backend(out)
    cs_apply_integral_kernel!(backend, 256)(out, values, op.offsets, op.indices, op.weights; ndrange=size(out))
    return out
end

"""CS integration plans; face and overlap matrices are supplied by the target."""
struct CSSpectralIntegralWorkspace{Q, S, F, G}
    quadrature::Q
    synthesis::S
    surface::F
    geometry::G
end

function CSSpectralIntegralWorkspace(q::CSSpectralQuadrature, T::Integer, face_matrix, overlap;
                                     synthesis=CPUSynthesis(), precision=AutoSynthesisPrecision(), batch=8)
    s = build_ring_synthesis(synthesis, q.rings, T+1, batch, precision)
    surface = surface_synthesis(s, 3)
    p = s.plan
    FT = eltype(p.coeffs)
    adapt = p.adapter
    nx, ny = q.mesh.Nx, q.mesh.Ny
    lon = repeat(deg2rad.(q.mesh.λᶜ), ny)
    lat = repeat(deg2rad.(q.mesh.φᶜ); inner=nx)
    # PS is an intensive box mean; C density is integrated over the CS area.
    means = spdiagm(0 => 1 ./ overlap.dst_areas)*overlap.intersections
    g = (; face=cs_gather_operator(face_matrix, adapt, FT),
        mean=cs_gather_operator(means, adapt, Float64),
        area=cs_gather_operator(spdiagm(0 => q.areas)*means, adapt, FT),
        sinlon=adapt(FT.(sin.(lon))), coslon=adapt(FT.(cos.(lon))),
        sinlat=adapt(FT.(sin.(lat))), coslat=adapt(FT.(cos.(lat))))
    return CSSpectralIntegralWorkspace(q, s, surface, g)
end

# Unlike LL's latitude integrand, this is a density per m². All arithmetic
# follows the selected 3-D plan precision; no implicit Float64 kernel promotion.
@kernel function cs_spectral_integrands!(xyz, forcing, U, V, D, pressure, surface,
                                         sinlon, coslon, sinlat, coslat, dA, dB, R, Rv, gravity)
    c, k = @index(Global, NTuple)
    ns = size(forcing, 1)
    if c <= ns && k <= size(forcing, 2)
        FT = eltype(forcing)
        cosine = coslat[c]
        p = FT(pressure[c, 1])
        dp = dA[k]+dB[k]*p
        u = FT(U[c, k])/cosine
        v = FT(V[c, k])/cosine
        scale = dp/gravity
        xyz[c, k] = (-sinlon[c]*u-sinlat[c]*coslon[c]*v)*scale
        xyz[ns+c, k] = (coslon[c]*u-sinlat[c]*sinlon[c]*v)*scale
        xyz[2ns+c, k] = cosine*v*scale
        forcing[c, k] = (Rv/(R*gravity))*FT(D[c, k])*dp +
            p*dB[k]/(R*gravity*cosine*cosine)*
            (FT(U[c, k])*FT(surface[c, 2])+FT(V[c, k])*FT(surface[c, 3]))
    end
end

"""Synthesize PS and exact LNSP gradients on the open refined CS rings."""
function integrate_surface_pressure!(ws::CSSpectralIntegralWorkspace, H, T)
    gradients = lnsp_gradient(H, T)
    spectra = zeros(ComplexF64, T+2, T+2, 3)
    spectra[1:T+1, 1:T+1, 1] .= H
    spectra[:, :, 2] .= gradients[1]
    spectra[:, :, 3] .= gradients[2]
    surface = synthesize_rings(spectra, ws.surface)
    ps = similar(surface, size(surface, 1), 1)
    backend = get_backend(ps)
    surface_pressure_samples!(backend, 256)(ps, surface; ndrange=length(ps))
    return (; surface, ps)
end

"""Great-circle face rates and conservative box means; samples stay on device."""
function integrate_face_fluxes!(ws::CSSpectralIntegralWorkspace, H, vo, d, T, dA, dB, c::TM5SpectralFluxes)
    q, g, p = ws.quadrature, ws.geometry, ws.synthesis.plan
    FT = eltype(p.coeffs)
    backend = get_backend(p.coeffs)
    surf = integrate_surface_pressure!(ws, H, T)
    nc, nf, nz = length(q.areas), p.batch, length(dA)
    ns = q.rings.offsets[end]-1
    nfaces = length(g.face.offsets)-1
    ps_out = similar(surf.ps, nc, 1)
    cs_apply_integral!(ps_out, surf.ps, g.mean)
    u_out, C_out = zeros(nfaces, nz), zeros(nc, nz)
    for firstlevel in 1:nf:nz
        levels = firstlevel:min(firstlevel+nf-1, nz)
        nb = length(levels)
        vb = zeros(ComplexF64, T+1, T+1, nf)
        db = zeros(ComplexF64, T+1, T+1, nf)
        vb[:, :, 1:nb] .= view(vo, :, :, levels)
        db[:, :, 1:nb] .= view(d, :, :, levels)
        U, V, D = synthesize_native_fields(vb, db, T, c.vod2uv_radius, ws.synthesis)
        xyz = similar(p.coeffs, FT, 3ns, nf)
        forcing = similar(p.coeffs, FT, ns, nf)
        aa, bb = zeros(FT, nf), zeros(FT, nf)
        aa[1:nb] .= dA[levels]
        bb[1:nb] .= dB[levels]
        cs_spectral_integrands!(backend, 256)(xyz, forcing, U, V, D, surf.ps, surf.surface,
            g.sinlon, g.coslon, g.sinlat, g.coslat, p.adapter(aa), p.adapter(bb),
            FT(q.radius), FT(c.vod2uv_radius), FT(STANDARD_GRAVITY); ndrange=(ns, nf))
        faces = similar(p.coeffs, FT, nfaces, nf)
        cells = similar(p.coeffs, FT, nc, nf)
        cs_apply_integral!(faces, xyz, g.face)
        cs_apply_integral!(cells, forcing, g.area)
        synchronize(backend)
        u_out[:, levels] .= Array(faces)[:, 1:nb]
        C_out[:, levels] .= Array(cells)[:, 1:nb]
    end
    synchronize(backend)
    # A zero-row v keeps the common endpoint/balance interface, with all
    # unique CS faces in u (their canonical orientation is in the face table).
    return (; ps=vec(Array(ps_out)), u=u_out, v=zeros(0, nz), C=C_out)
end
