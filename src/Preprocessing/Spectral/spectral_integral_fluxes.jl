# Direct target integrals. TM5 tmm.F90:1866–1876, 2135–2147, 2167–2179,
# 2428–2473. R*Rv/g (not R²/g) follows the two-radius Gauss identity.

"""Convert synthesized log surface pressure to pressure [Pa] at each sample."""
@kernel function surface_pressure_samples!(ps, surface, scale=nothing)
    i = @index(Global)
    i <= length(ps) && (ps[i, 1] = surface_pressure_value(surface[i, 1],scale))
end

"""
Form the TM5 face integrands and independent layer forcing at refined samples.

`U,V` are cos(latitude)-scaled winds, `D` is divergence, and surface columns
2–3 are the longitude and cos-scaled latitude LNSP gradients. Sample-local
Δp retains wind/pressure covariance. `R` belongs to the mesh, `Rv` to IFS;
their product in the divergence term preserves the two-radius Gauss identity.
"""
@kernel function spectral_flux_integrands!(fu, fv, forcing, U, V, D, ps, surface,
                                          offsets, lengths, coslatitudes, dA, dB, R, Rv, nf)
    i, row, k = @index(Global, NTuple)
    FT = eltype(fu)
    if row <= length(lengths) && i <= lengths[row] && k <= nf
        c = offsets[row]+i-1
        cosine = coslatitudes[row]
        if abs(cosine) < eps(FT)
            fu[c, k] = 0
            fv[c, k] = 0
            forcing[c, k] = 0
        else
            p = ps[c, 1]
            dp = dA[k] + dB[k]*p
            u = FT(U[c, k])
            v = FT(V[c, k])
            fu[c, k] = (R / FT(STANDARD_GRAVITY)) * u * dp / cosine
            fv[c, k] = (R / FT(STANDARD_GRAVITY)) * v * dp
            forcing[c, k] = (R*Rv / FT(STANDARD_GRAVITY)) * FT(D[c, k]) * dp * cosine +
                (R / FT(STANDARD_GRAVITY)) * p * dB[k] / cosine * (u*surface[c, 2] + v*surface[c, 3])
        end
    end
end

"""
Integrate cell-mean PS [Pa], zonal face rates and cell forcing [kg/s].

PS uses cos-linear latitude weights and cell-area normalization; forcing
uses plain latitude weights. Longitude intervals use periodic primitives.
"""
@kernel function integrate_cell_rows!(ps_out, u_out, C_out, ps, psprefix, fu, C, Cprefix,
                                     lengths, offsets, node_offsets, node_rows, cell_offsets, nlon,
                                     weights, cosweights, areas, R, nf)
    i, j, k = @index(Global, NTuple)
    if j <= length(nlon) && i <= nlon[j] && k <= nf
        cell = cell_offsets[j]+i-1
        pressure = zero(eltype(ps_out))
        zonal = zero(eltype(u_out))
        forcing = zero(eltype(C_out))
        for node in node_offsets[j]:node_offsets[j+1]-1
            row = node_rows[node]
            J = lengths[row]
            start = offsets[row]
            a = eltype(ps_out)(i-1)*J/nlon[j]
            b = eltype(ps_out)(i)*J/nlon[j]
            k == 1 && (pressure += cosweights[node]*(
                ring_primitive(ps, psprefix, start, J, 1, b)-ring_primitive(ps, psprefix, start, J, 1, a)))
            sample = start+round(Int, a)
            zonal += weights[node]*fu[sample, k]
            forcing += weights[node]*(ring_primitive(C, Cprefix, start, J, k, b)-
                                      ring_primitive(C, Cprefix, start, J, k, a))
        end
        k == 1 && (ps_out[cell] = pressure*R^2/areas[cell])
        u_out[cell, k] = zonal
        C_out[cell, k] = forcing
    end
end

"""Integrate northward face rates [kg/s], with zero flow across polar boundaries."""
@kernel function integrate_boundary_rows!(v_out, fv, prefix, lengths, offsets, rows, counts, bounds, nf)
    i, b, k = @index(Global, NTuple)
    if b <= length(counts) && i <= counts[b] && k <= nf
        f = bounds[b]+i-1
        if b == 1 || b == length(counts)
            v_out[f, k] = zero(eltype(v_out))
        else
            row = rows[b]
            J = lengths[row]
            start = offsets[row]
            a = eltype(v_out)(i-1)*J/counts[b]
            z = eltype(v_out)(i)*J/counts[b]
            v_out[f, k] = ring_primitive(fv, prefix, start, J, k, z)-ring_primitive(fv, prefix, start, J, k, a)
        end
    end
end

"""Reusable quadrature and synthesis plans. Batches bound the refined-grid working set."""
struct SpectralIntegralWorkspace{S, F, Q, L, G, B, P}
    quadrature::G
    synthesis::S
    surface::F
    geometry::Q
    longitude::L
    batch::Int
    bands::B
    polar_cache::P
end
SpectralIntegralWorkspace(q, s, surface, geometry, longitude, batch, bands) =
    SpectralIntegralWorkspace(q, s, surface, geometry, longitude, batch, bands, nothing)

function SpectralIntegralWorkspace(grid, T, c::TM5SpectralFluxes; synthesis = CPUSynthesis(),
                                   precision = AutoSynthesisPrecision(), batch::Integer = 8,
                                   latitude_band_cells::Integer = 32)
    q = SpectralQuadrature(grid, T, c.face_quadrature, c.refinement_steps)
    latitude_band_cells >= 0 || throw(ArgumentError("latitude_band_cells must be nonnegative"))
    cells = latitude_band_cells == 0 ? 0 : synthesis isa CPUSynthesis ?
        max(1, latitude_band_cells÷cpu_band_workers()) : latitude_band_cells
    bands = quadrature_bands(q.rule, q, cells)
    work = spectral_integral_workspace(bands === nothing ? q : first(bands).quadrature,
        T, c; synthesis, precision, batch, plan_polar=bands === nothing)
    polar_cache = bands === nothing ? nothing : band_polar_cache(bands, work.synthesis, T, batch)
    return SpectralIntegralWorkspace(q, work.synthesis, work.surface, work.geometry,
        work.longitude, Int(batch), bands, polar_cache)
end

function spectral_integral_workspace(q, T, c; synthesis, precision, batch, plan_polar=true)
    s = build_ring_synthesis(synthesis, q.rings, T+1, native_transform_batch(q.rule, batch), precision;
        stable_poles=c.face_quadrature isa GaussFourier)
    surface = surface_synthesis(s, 3; precision=eltype(s.plan.coeffs))
    longitude = longitude_plans(q.rule, q, s, surface, batch)
    return SpectralIntegralWorkspace(q, s, surface, integral_geometry(q, s; plan_polar), longitude, Int(batch), nothing)
end

function integral_geometry(q, s; plan_polar=true)
    FT = eltype(s.plan.coeffs)
    adapt = s.plan.adapter
    # Float32 scalar vector transforms lose small polar winds to cancellation.
    # Reuse CS's stable vector harmonics on device plans only; CPU bits stay fixed.
    polar = plan_polar && s isa GPUSpectralSynthesis && FT === Float32 ?
        cs_polar_synthesis(s, s.plan.T-1; batch=s.plan.batch÷native_transform_batch(q.rule, 1), rings=q.rings, threshold=0.5) : nothing
    g = (; polar, lengths = adapt(q.rings.nlon), offsets = adapt(q.rings.offsets),
        coslatitudes = adapt(FT.(cos.(q.rings.latitudes))), node_offsets = adapt(q.node_offsets), node_rows = adapt(q.node_rows),
        cell_offsets = adapt(q.cell_offsets), nlon = adapt(q.nlon),
        weights = adapt(FT.(q.plain_weights)), cosweights = adapt(FT.(q.cosine_weights)),
        areas = adapt(FT.(q.areas)), rows = adapt(q.boundary_rows), counts = adapt(q.boundary_counts),
        bounds = adapt(q.boundary_offsets), integral_offsets=adapt(cumsum([1; q.interval_counts])),
        boundary_active=adapt([1 < b < length(q.boundary_counts) for b in eachindex(q.boundary_counts)]))
    return g
end

"""Sample exp(LNSP), then integrate cell means with the selected longitude rule."""
function integrate_surface_pressure!(ws::SpectralIntegralWorkspace, H, T; layout=nothing)
    spectra, pressure_scale = surface_pressure_spectra(H, T, ws.surface)
    surface = layout === nothing ? synthesize_rings(spectra, ws.surface) :
        band_synthesis(spectra, ws.surface; layout)
    p = ws.synthesis.plan
    ps = similar(p.coeffs, eltype(p.coeffs), size(surface, 1), 1)
    backend = get_backend(ps)
    surface_pressure_samples!(backend, 256)(ps, surface, pressure_scale; ndrange = length(ps))
    prefix = longitude_integrals!(ws.quadrature.rule, ps, ws, 1, ws.longitude.surface)
    synchronize(backend)
    return (; surface, ps, prefix)
end

"""Integrate native-level face rates and independent spectral divergence C_k."""
function integrate_backend_face_fluxes!(ws::SpectralIntegralWorkspace, H, vo, d, T, dA, dB, c::TM5SpectralFluxes;
                                spectra=nothing, samples=nothing, surface_layout=nothing, host_output=true)
    ws.bands === nothing || return integrate_latitude_bands!(ws, H, vo, d, T, dA, dB, c; host_output)
    q = ws.quadrature
    g = ws.geometry
    p = ws.synthesis.plan
    surf = integrate_surface_pressure!(ws, H, T; layout=surface_layout)
    nc = length(q.areas)
    nv = q.boundary_offsets[end]-1
    nz = length(dA)
    FT = eltype(p.coeffs)
    ps_out = similar(p.coeffs, FT, nc)
    u_out = similar(p.coeffs, FT, nc, nz)
    v_out = similar(p.coeffs, FT, nv, nz)
    C_out = similar(p.coeffs, FT, nc, nz)
    backend = get_backend(ps_out)
    nf = ws.batch
    for firstlevel in 1:nf:nz
        levels = firstlevel:min(firstlevel+nf-1, nz)
        nb = length(levels)
        U, V, D = integral_field_samples(ws, vo, d, T, c, levels, spectra, samples, surface_layout)
        # Samples are consumed once; form the three integrands in place.
        fu, fv, forcing = U, V, D
        aa = zeros(FT, nf)
        bb = zeros(FT, nf)
        aa[1:nb].= dA[levels]
        bb[1:nb].= dB[levels]
        spectral_flux_integrands!(backend, 256)(fu, fv, forcing, U, V, D, surf.ps, surf.surface,
            g.offsets, g.lengths, g.coslatitudes, p.adapter(aa), p.adapter(bb), FT(q.radius), FT(c.vod2uv_radius), nf;
            ndrange = (maximum(q.rings.nlon), length(q.rings.nlon), nf))
        fp = longitude_integrals!(q.rule, fv, ws, nf, ws.longitude.faces)
        cp = longitude_integrals!(q.rule, forcing, ws, nf, ws.longitude.fields)
        uu = similar(p.coeffs, FT, nc, nf)
        vv = similar(p.coeffs, FT, nv, nf)
        cc = similar(p.coeffs, FT, nc, nf)
        integrate_rows!(q.rule, backend, ps_out, uu, vv, cc, surf, fu, forcing, fv, cp, fp, g, q, nf)
        synchronize(backend)
        view(u_out, :, levels) .= view(uu, :, 1:nb)
        view(v_out, :, levels) .= view(vv, :, 1:nb)
        view(C_out, :, levels) .= view(cc, :, 1:nb)
    end
    raw = (; ps=ps_out, u=u_out, v=v_out, C=C_out)
    return host_output ? face_host_result(raw) : raw
end

"""Upward independent interface flux W_e = Σ_{k<e} C_k − B_e Σ_k C_k."""
function integrate_vertical_flux!(W, C, B)
    nc, nz = size(C)
    first(B) == 0 && last(B) == 1 || throw(ArgumentError("TM5 W requires B=0 at the top and B=1 at the surface"))
    size(W) == (nc, nz+1) && length(B) == nz+1 || throw(DimensionMismatch("vertical integral interfaces"))
    for cell in 1:nc
        total = sum(view(C, cell, :))
        acc = 0.0
        W[cell, 1] = 0.0
        for k in 1:nz-1
            acc += C[cell, k]
            W[cell, k+1] = acc-B[k+1]*total
        end
        W[cell, end] = 0.0
    end
    return W
end

longitude_plans(::TM5Trapezoid, q, s, surface, batch) = (; fields=nothing, faces=nothing, surface=nothing)
function longitude_plans(::GaussFourier, q, s, surface, batch)
    selected_rows=unique(q.node_rows)
    # The shared capacity includes boundaries; a boundary-only group can have
    # more rows than the interior group on a small or reduced target.
    capacity = fourier_integral_plan(q.rings, q.interval_counts, s, batch)
    scratch = Dict((g.J, g.n) => (; g.samples, g.spectrum, g.half, g.integrals) for g in capacity.groups)
    return (; fields=fourier_integral_plan(q.rings, q.interval_counts, s, batch; selected_rows, scratch),
        faces=fourier_integral_plan(q.rings, q.interval_counts, s, batch; selected_rows=unique(q.boundary_rows), scratch),
        surface=fourier_integral_plan(q.rings, q.interval_counts, surface, 1; selected_rows, scratch))
end
function longitude_integrals!(::TM5Trapezoid, values, ws, nf, plan)
    prefix = similar(values)
    return ring_prefix!(prefix, values, ws.geometry, nf, maximum(ws.quadrature.rings.nlon))
end
longitude_integrals!(::GaussFourier, values, ws, nf, plan) = fourier_integrals!(values, plan, nf)

function integrate_rows!(::TM5Trapezoid, backend, ps_out, uu, vv, cc, surf, fu, forcing, fv, cp, fp, g, q, nf)
    integrate_cell_rows!(backend, 256)(ps_out, uu, cc, surf.ps, surf.prefix, fu, forcing, cp,
        g.lengths, g.offsets, g.node_offsets, g.node_rows, g.cell_offsets, g.nlon, g.weights, g.cosweights,
        g.areas, eltype(ps_out)(q.radius), nf;
        ndrange=(maximum(q.nlon), length(q.nlon), nf))
    integrate_boundary_rows!(backend, 256)(vv, fv, fp, g.lengths, g.offsets, g.rows, g.counts, g.bounds, nf;
        ndrange=(maximum(q.boundary_counts), length(q.boundary_counts), nf))
end

@kernel function integrate_gauss_cells!(ps_out, u_out, C_out, pressure, fu, forcing,
                                       lengths, offsets, integral_offsets, node_offsets, node_rows,
                                       cell_offsets, nlon, weights, cosweights, areas, R, nf, accumulation=Val(false))
    i, j, k = @index(Global, NTuple)
    if j <= length(nlon) && i <= nlon[j] && k <= nf
        cell = cell_offsets[j]+i-1
        ps = zero(eltype(ps_out))
        u = zero(eltype(u_out))
        C = zero(eltype(C_out))
        ep, eu, ec = zero(ps), zero(u), zero(C)
        for node in node_offsets[j]:node_offsets[j+1]-1
            row = node_rows[node]
            interval = integral_offsets[row]+i-1
            sample = offsets[row]+(i-1)*(lengths[row]÷nlon[j])
            k == 1 && ((ps, ep) = face_product_step(ps, ep, cosweights[node],pressure[interval, 1], accumulation))
            u, eu = face_product_step(u, eu, weights[node],fu[sample, k], accumulation)
            C, ec = face_product_step(C, ec, weights[node],forcing[interval, k], accumulation)
        end
        k == 1 && (ps_out[cell] = face_sum_result(ps, ep, accumulation)*R^2/areas[cell])
        u_out[cell, k] = face_sum_result(u, eu, accumulation)
        C_out[cell, k] = face_sum_result(C, ec, accumulation)
    end
end

@kernel function integrate_fourier_boundaries!(out, integrals, offsets, rows, counts, bounds, active, nf)
    i, b, k = @index(Global, NTuple)
    if b <= length(counts) && i <= counts[b] && k <= nf
        out[bounds[b]+i-1, k] = !active[b] ? zero(eltype(out)) :
            integrals[offsets[rows[b]]+i-1, k]
    end
end

function integrate_rows!(::GaussFourier, backend, ps_out, uu, vv, cc, surf, fu, forcing, fv, cp, fp, g, q, nf)
    integrate_gauss_cells!(backend, 256)(ps_out, uu, cc, surf.prefix, fu, cp,
        g.lengths, g.offsets, g.integral_offsets, g.node_offsets, g.node_rows, g.cell_offsets,
        g.nlon, g.weights, g.cosweights, g.areas, eltype(ps_out)(q.radius), nf,
        face_accumulation(backend, eltype(ps_out));
        ndrange=(maximum(q.nlon), length(q.nlon), nf))
    integrate_fourier_boundaries!(backend, 256)(vv, fp, g.integral_offsets, g.rows, g.counts, g.bounds, g.boundary_active, nf;
        ndrange=(maximum(q.boundary_counts), length(q.boundary_counts), nf))
end

function spectral_synthesis_metadata(ws::SpectralIntegralWorkspace)
    metadata = Dict{String,Any}("spectral_synthesis"=>synthesis_provenance(ws.synthesis),
        "spectral_synthesis_float"=>string(eltype(ws.synthesis.plan.coeffs)),
        "spectral_surface_float"=>string(eltype(ws.surface.plan.coeffs)),
        "face_quadrature"=>face_quadrature_name(ws.quadrature.rule))
    merge!(metadata, face_rule_metadata(ws.quadrature.rule, ws.quadrature))
    return metadata
end
face_rule_metadata(::TM5Trapezoid, q) = Dict{String,Any}()
face_rule_metadata(rule::GaussFourier, q) = Dict{String,Any}(
    "face_quadrature_gauss_order"=>rule.order,
    "face_quadrature_longitude_oversampling"=>rule.oversampling,
    "face_quadrature_latitude_nodes_min"=>minimum(diff(q.node_offsets)),
    "face_quadrature_latitude_nodes_max"=>maximum(diff(q.node_offsets)))
