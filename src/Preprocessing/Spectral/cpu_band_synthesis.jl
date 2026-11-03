# Small CPU bands do not benefit from a threaded KA launch for each mode.
# Use host loops for the elementwise pack/scatter/gather. Native fields use
# fixed GEMM panels; surface fields retain the full contraction's shape.
band_synthesis(spec, s; out=nothing, layout=nothing, native_panels=false) = synthesize_rings(spec, s; out)

# Anchor native GEMMs to global latitude columns. Both full and streamed
# execution use 32-column panels, including padding at either end of a band.
# Keeping M/N/K, leading dimensions and column positions identical avoids
# OpenBLAS edge microkernels changing the reduction order on different CPUs.
function native_panel_scratch(p)
    FT = eltype(p.coeffs)
    return (; table=zeros(FT, p.T+1, 32), sums=zeros(FT, 2p.batch, 32))
end

function native_panel_product!(p, m, scratch, layout)
    K = p.T-m+1
    rows = layout === nothing ? axes(p.sums, 2) : layout.rows
    table = view(scratch.table, 1:K, :)
    j = 1
    while j <= length(rows)
        tile = (rows[j]-1)÷32
        last = j
        while last < length(rows) && (rows[last+1]-1)÷32 == tile
            last += 1
        end
        fill!(table, 0)
        for column in j:last
            copyto!(view(table, :, mod1(rows[column], 32)), view(p.legendre[m+1], :, column))
        end
        mul!(scratch.sums, view(p.coeffs, :, 1:K), table)
        for column in j:last
            copyto!(view(p.sums, :, column), view(scratch.sums, :, mod1(rows[column], 32)))
        end
        j = last+1
    end
    return p.sums
end

function band_synthesis(spec, s::CPURingSynthesis; out=nothing, layout=nothing, native_panels=false)
    p = s.plan
    FT = eltype(p.coeffs)
    nf = p.batch
    size(spec) == (p.T+1, p.T+1, nf) || throw(DimensionMismatch("ring spectral batch"))
    source = eltype(spec) === Complex{FT} ? spec : Complex{FT}.(spec)
    out === nothing && (out = similar(p.coeffs, p.rings.offsets[end]-1, nf))
    # OpenBLAS changes the six-row surface contraction's accumulation when
    # N shrinks. Preserve its original M/N/K and column positions using one
    # reusable mode panel, rather than retaining every latitude table.
    native = native_panels
    scratch = native ? native_panel_scratch(p) : nothing
    surface_layout = native ? nothing : layout
    panel = surface_layout === nothing ? nothing : Vector{FT}(undef, (p.T+1)*layout.total)
    sums = surface_layout === nothing ? p.sums : zeros(FT, 2nf, layout.total)
    for group in p.groups
        fill!(group.half, 0)
    end
    for m in 0:p.T
        K = p.T-m+1
        @inbounds for q in 1:K, k in 1:nf
            z = source[q+m, m+1, k]
            p.coeffs[k, q] = real(z)
            p.coeffs[nf+k, q] = imag(z)
        end
        table = p.legendre[m+1]
        if surface_layout !== nothing
            table = reshape(view(panel, 1:K*layout.total), K, layout.total)
            fill!(table, 0)
            for (j, row) in enumerate(layout.rows)
                copyto!(view(table, :, row), view(p.legendre[m+1], :, j))
            end
        end
        if native
            native_panel_product!(p, m, scratch, layout)
        else
            mul!(sums, view(p.coeffs, :, 1:K), table)
        end
        for group in p.groups
            if m <= min(p.T, (group.J-1)÷2)
                @inbounds for k in 1:nf, r in eachindex(group.rows)
                    j = group.rows[r]
                    column = surface_layout === nothing ? j : layout.rows[j]
                    z = complex(sums[k, column], sums[nf+k, column])
                    group.half[m+1, k, r] = z*p.phases[m+1, j]
                end
            end
        end
    end
    for group in p.groups
        ring_fft_mul!(group.samples, group.plan, group.half)
        for r in eachindex(group.rows), k in 1:nf
            copyto!(out, p.rings.offsets[group.rows[r]]+(k-1)*size(out, 1),
                group.samples, 1+(k-1)*group.J+(r-1)*group.J*nf, group.J)
        end
    end
    return out
end
