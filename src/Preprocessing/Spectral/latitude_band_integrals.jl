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
        vb = spectral_level_batch(vo, levels, nf, ws.synthesis.plan)
        db = spectral_level_batch(d, levels, nf, ws.synthesis.plan)
        values = synthesize_quadrature_fields(ws.quadrature.rule, vb, db, T, c.vod2uv_radius, ws.synthesis)
        cs_stable_polar_winds!(values[1], values[2], vb, db, c.vod2uv_radius, ws.geometry.polar)
        return values
    end
    source = spectra[cld(first(levels), nf)]
    values = band_synthesis(source.native, ws.synthesis; out=samples, layout, native_panels=true)
    U, V, D = view(values, :, 1:nf), view(values, :, nf+1:2nf), view(values, :, 2nf+1:3nf)
    if ws.geometry.polar !== nothing
        cs_stable_polar_winds!(U, V, nothing, nothing, c.vod2uv_radius, ws.geometry.polar;
            coefficients=source.polar)
    end
    return U, V, D
end

# Wind coefficients depend on level, not latitude. Prepare each padded native
# batch once, then reuse it across bands. No global sample or table array lives
# alongside these coefficients.
function band_spectra(ws, vo, d, T, radius)
    nf = ws.batch
    return map(1:nf:size(vo, 3)) do lo
        levels = lo:min(lo+nf-1, size(vo, 3))
        vb = spectral_level_batch(vo, levels, nf, ws.synthesis.plan)
        db = spectral_level_batch(d, levels, nf, ws.synthesis.plan)
        native = quadrature_spectra(vb, db, T, radius, ws.synthesis)
        p = ws.synthesis.plan
        polar = ws.polar_cache === nothing ? nothing : cs_polar_coefficients(get_backend(p.coeffs), vb, db, radius,
            (; T, batch=2nf, coeffs=p.coeffs, adapter=p.adapter))
        (; native, polar)
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
        work = SpectralIntegralWorkspace(q, s, previous.surface, integral_geometry(q, s; plan_polar=ws.polar_cache === nothing),
            previous.longitude, ws.batch, nothing)
    else
        backend = ws.synthesis
        work = spectral_integral_workspace(q, T, c; synthesis=backend,
            precision=eltype(ws.synthesis.plan.coeffs), batch=ws.batch, plan_polar=ws.polar_cache === nothing)
    end
    polar = ws.polar_cache === nothing ? work.geometry.polar : refill_band_polar!(ws.polar_cache, q.rings, T)
    g = merge(work.geometry, (; polar, boundary_active=work.synthesis.plan.adapter(band.active)))
    return SpectralIntegralWorkspace(q, work.synthesis, work.surface, g, work.longitude, ws.batch, nothing)
end

function integrate_latitude_bands!(ws, H, vo, d, T, dA, dB, c; host_output=true)
    q = ws.quadrature
    FT = eltype(ws.synthesis.plan.coeffs)
    nz = length(dA)
    prototype = ws.synthesis.plan.coeffs
    result = (; ps=similar(prototype, FT, length(q.areas)), u=similar(prototype, FT, length(q.areas), nz),
        v=similar(prototype, FT, q.boundary_offsets[end]-1, nz), C=similar(prototype, FT, length(q.areas), nz))
    spectra = band_spectra(ws, vo, d, T, c.vod2uv_radius)
    firstq = first(ws.bands).quadrature
    initial = SpectralIntegralWorkspace(firstq, ws.synthesis, ws.surface, ws.geometry,
        ws.longitude, ws.batch, nothing)
    largest = maximum(length(b.quadrature.node_rows) for b in ws.bands)
    workers = ws.synthesis isa CPURingSynthesis ?
        min(cpu_band_workers(), length(ws.bands), max(1, 512÷largest)) : 1
    works = SpectralIntegralWorkspace[initial for _ in 1:workers]
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
                    integrate_face_fluxes!(work, H, vo, d, T, dA, dB, c; spectra, samples, surface_layout, host_output=false)
                else
                    Threads.@spawn integrate_face_fluxes!($work, H, vo, d, T, dA, dB, c;
                        spectra, samples=$samples, surface_layout=$surface_layout, host_output=false)
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
    return host_output ? face_host_result(result) : result
end

# Polar transforms retain their FFT plans across endpoints. Different band
# shapes share one bounded pair of Legendre tables; only one band uses them
# at a time on device. Coefficients and sample scratch are also reused.
function band_polar_cache(bands, s, T, batch)
    s isa GPUSpectralSynthesis{Float32} || return nothing
    layouts = map(bands) do band
        rings = band.quadrature.rings
        rows = findall(phi -> abs(cos(phi)) < 0.5, rings.latitudes)
        (; rings, rows)
    end
    capacity = maximum(length(l.rows) for l in layouts)
    capacity == 0 && return nothing
    host = ntuple(_ -> [zeros(T-m+1, capacity) for m in 0:T], 2)
    tables = map(family -> [s.plan.adapter(Float32.(a)) for a in family], host)
    plans = Dict{Tuple,Any}()
    for (; rings, rows) in layouts
        isempty(rows) && continue
        key = (Tuple(rings.nlon[rows]), Tuple(rings.origins[rows]))
        haskey(plans, key) && continue
        views = map(family -> [view(a, :, 1:length(rows)) for a in family], tables)
        plans[key] = cs_polar_synthesis(s, T; batch, rings, threshold=0.5, tables=views)
    end
    return (; host, tables, plans)
end

function refill_band_polar!(cache, rings, T)
    rows = findall(phi -> abs(cos(phi)) < 0.5, rings.latitudes)
    isempty(rows) && return nothing
    key = (Tuple(rings.nlon[rows]), Tuple(rings.origins[rows]))
    polar = cache.plans[key]
    r = SynthesisRings(rings.latitudes[rows], rings.nlon[rows], rings.origins[rows])
    host = map(family -> [view(a, :, 1:length(rows)) for a in family], cache.host)
    cs_vector_legendre!(host[1], host[2], r, T)
    for (target, source) in zip((polar.derivative.plan.legendre, polar.divided.plan.legendre), host)
        foreach((a, b) -> copyto!(a, Float32.(b)), target, source)
    end
    copyto!(polar.indices, vcat([collect(rings.offsets[j]:rings.offsets[j+1]-1) for j in rows]...))
    copyto!(polar.cosine, vcat([fill(Float32(cos(rings.latitudes[j])), rings.nlon[j]) for j in rows]...))
    return polar
end
