# Keep whole target cells together: each latitude sum retains its original
# node order. Slice the existing geometry, never reconstruct latitude faces.
quadrature_bands(::TM5Trapezoid, q, cells) = nothing
function quadrature_bands(::GaussFourier, q, cells)
    cells == 0 && return nothing
    ranges = UnitRange{Int}[]
    lo = 1
    while lo <= length(q.nlon)
        hi = lo
        # Coarse target cells have more Gauss nodes. Bound interior rows too,
        # allowing one complete cell when it alone exceeds the row budget.
        while hi < min(lo+cells-1, length(q.nlon)) && q.node_offsets[hi+2]-q.node_offsets[lo] <= 128
            hi += 1
        end
        push!(ranges, lo:hi)
        lo = hi+1
    end
    length(ranges) == 1 && return nothing
    return [quadrature_band(q, rows) for rows in ranges]
end

function quadrature_band(q, latitudes)
    lo, hi = first(latitudes), last(latitudes)
    nodes = q.node_offsets[lo]:q.node_offsets[hi+1]-1
    boundaries = lo:hi+1
    rows = sort(unique([q.node_rows[nodes]; q.boundary_rows[boundaries]]))
    localrow = zeros(Int, length(q.rings.nlon))
    localrow[rows] .= eachindex(rows)
    cells = q.cell_offsets[lo]:q.cell_offsets[hi+1]-1
    faces = q.boundary_offsets[lo]:q.boundary_offsets[hi+2]-1
    rings = q.rings
    band = SpectralQuadrature(q.rule, q.interval_counts[rows],
        SynthesisRings(rings.latitudes[rows], rings.nlon[rows], rings.origins[rows]),
        q.nlon[latitudes], cumsum([1; q.nlon[latitudes]]),
        q.node_offsets[lo:hi+1].-first(nodes).+1, localrow[q.node_rows[nodes]],
        q.plain_weights[nodes], q.cosine_weights[nodes], localrow[q.boundary_rows[boundaries]],
        q.boundary_counts[boundaries], cumsum([1; q.boundary_counts[boundaries]]), q.areas[cells], q.radius)
    active = [1 < b < length(q.boundary_counts) for b in boundaries]
    return (; quadrature=band, cells, faces, active, rows)
end

function integral_field_samples(ws, vo, d, T, c, levels, spectra, samples, layout)
    nf = ws.batch
    if spectra === nothing
        vb = zeros(ComplexF64, T+1, T+1, nf)
        db = zero(vb)
        vb[:, :, 1:length(levels)] .= view(vo, :, :, levels)
        db[:, :, 1:length(levels)] .= view(d, :, :, levels)
        return synthesize_quadrature_fields(ws.quadrature.rule, vb, db, T, c.vod2uv_radius, ws.synthesis)
    end
    source = spectra[cld(first(levels), nf)]
    values = band_synthesis(source, ws.synthesis; out=samples, layout, native_panels=true)
    return view(values, :, 1:nf), view(values, :, nf+1:2nf), view(values, :, 2nf+1:3nf)
end

# Wind coefficients depend on level, not latitude. Prepare each padded native
# batch once, then reuse it across bands. No global sample or table array lives
# alongside these coefficients.
function band_spectra(ws, vo, d, T, radius)
    nf = ws.batch
    return map(1:nf:size(vo, 3)) do lo
        levels = lo:min(lo+nf-1, size(vo, 3))
        vb = zeros(ComplexF64, T+1, T+1, nf)
        db = zero(vb)
        vb[:, :, 1:length(levels)] .= view(vo, :, :, levels)
        db[:, :, 1:length(levels)] .= view(d, :, :, levels)
        quadrature_spectra(vb, db, T, radius, ws.synthesis)
    end
end

function refill_band_tables!(s, q)
    p = s.plan
    # Host recurrence has exactly the same Float64 intermediates as initial
    # construction. Copy only the bounded band if this is a device plan.
    host = get_backend(p.coeffs) isa KernelAbstractions.CPU ? p.legendre :
        [zeros(eltype(p.coeffs), p.T-m+1, length(q.rings.nlon)) for m in 0:p.T]
    fill_ring_legendre!(host, q.rings, p.T; stable_poles=true)
    if host !== p.legendre
        foreach(copyto!, p.legendre, host)
    end
    return s
end

function band_workspace(ws, previous, band, T, c)
    q = band.quadrature
    old = previous.quadrature
    if old.rings.nlon == q.rings.nlon && old.rings.origins == q.rings.origins &&
       old.interval_counts == q.interval_counts && old.node_rows == q.node_rows && old.boundary_rows == q.boundary_rows
        s = refill_band_tables!(previous.synthesis, q)
        work = SpectralIntegralWorkspace(q, s, previous.surface, integral_geometry(q, s),
            previous.longitude, ws.batch, nothing)
    else
        backend = ws.synthesis isa CPURingSynthesis ? CPUSynthesis() : GPUSynthesis()
        work = spectral_integral_workspace(q, T, c; synthesis=backend,
            precision=eltype(ws.synthesis.plan.coeffs), batch=ws.batch)
    end
    g = merge(work.geometry, (; boundary_active=work.synthesis.plan.adapter(band.active)))
    return SpectralIntegralWorkspace(q, work.synthesis, work.surface, g, work.longitude, ws.batch, nothing)
end

function integrate_latitude_bands!(ws, H, vo, d, T, dA, dB, c)
    q = ws.quadrature
    FT = eltype(ws.synthesis.plan.coeffs)
    nz = length(dA)
    result = (; ps=zeros(FT, length(q.areas)), u=zeros(FT, length(q.areas), nz),
        v=zeros(FT, q.boundary_offsets[end]-1, nz), C=zeros(FT, length(q.areas), nz))
    spectra = band_spectra(ws, vo, d, T, c.vod2uv_radius)
    firstq = first(ws.bands).quadrature
    initial = SpectralIntegralWorkspace(firstq, ws.synthesis, ws.surface, ws.geometry,
        ws.longitude, ws.batch, nothing)
    largest = maximum(length(b.quadrature.node_rows) for b in ws.bands)
    workers = ws.synthesis isa CPURingSynthesis ?
        min(cpu_band_workers(), length(ws.bands), max(1, 512÷largest)) : 1
    works = [initial for _ in 1:workers]
    # Distinct mutable plans for each worker; packed coefficients are read-only.
    for k in 2:workers
        backend = CPUSynthesis()
        works[k] = spectral_integral_workspace(ws.bands[k].quadrature, T, c;
            synthesis=backend, precision=FT, batch=ws.batch)
    end
    buffers = [similar(w.synthesis.plan.coeffs, FT, w.quadrature.rings.offsets[end]-1, 3ws.batch) for w in works]
    for lo in 1:workers:length(ws.bands)
        group = ws.bands[lo:min(lo+workers-1, length(ws.bands))]
        # Table recurrence runs on the caller before workers start, so its
        # static thread schedule is never nested inside a worker task.
        for k in eachindex(group)
            work = band_workspace(ws, works[k], group[k], T, c)
            if work.synthesis !== works[k].synthesis
                buffers[k] = similar(work.synthesis.plan.coeffs, FT, work.quadrature.rings.offsets[end]-1, 3ws.batch)
            end
            works[k] = work
        end
        jobs = @sync begin
            map(eachindex(group)) do k
                work, samples, band = works[k], buffers[k], group[k]
                surface_layout = (; rows=band.rows, total=length(q.rings.nlon))
                if workers == 1
                    integrate_face_fluxes!(work, H, vo, d, T, dA, dB, c; spectra, samples, surface_layout)
                else
                    Threads.@spawn integrate_face_fluxes!($work, H, vo, d, T, dA, dB, c;
                        spectra, samples=$samples, surface_layout=$surface_layout)
                end
            end
        end
        # Copy on the caller, including shared band boundaries, to avoid
        # concurrent writes even when the two values are bitwise equal.
        for k in eachindex(group)
            raw = workers == 1 ? jobs[k] : fetch(jobs[k])
            band = group[k]
            result.ps[band.cells] .= raw.ps
            result.u[band.cells, :] .= raw.u
            result.C[band.cells, :] .= raw.C
            result.v[band.faces, :] .= raw.v
        end
    end
    return result
end
