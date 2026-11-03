# Small CPU bands do not benefit from a threaded KA launch for each mode.
# Keep the same coefficient pack, GEMM and complex multiply as the portable
# path; only schedule the elementwise pack/scatter/gather with host loops.
band_synthesis(spec, s; out=nothing, layout=nothing) = synthesize_rings(spec, s; out)
function band_synthesis(spec, s::CPURingSynthesis; out=nothing, layout=nothing)
    p = s.plan
    FT = eltype(p.coeffs)
    nf = p.batch
    size(spec) == (p.T+1, p.T+1, nf) || throw(DimensionMismatch("ring spectral batch"))
    source = eltype(spec) === Complex{FT} ? spec : Complex{FT}.(spec)
    out === nothing && (out = similar(p.coeffs, p.rings.offsets[end]-1, nf))
    # OpenBLAS changes the six-row surface contraction's accumulation when
    # N shrinks. Preserve its original M/N/K and column positions using one
    # reusable mode panel, rather than retaining every latitude table.
    panel = layout === nothing ? nothing : Vector{FT}(undef, (p.T+1)*layout.total)
    sums = layout === nothing ? p.sums : zeros(FT, 2nf, layout.total)
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
        if layout !== nothing
            table = reshape(view(panel, 1:K*layout.total), K, layout.total)
            fill!(table, 0)
            for (j, row) in enumerate(layout.rows)
                copyto!(view(table, :, row), view(p.legendre[m+1], :, j))
            end
        end
        mul!(sums, view(p.coeffs, :, 1:K), table)
        for group in p.groups
            if m <= min(p.T, (group.J-1)÷2)
                @inbounds for k in 1:nf, r in eachindex(group.rows)
                    j = group.rows[r]
                    column = layout === nothing ? j : layout.rows[j]
                    z = complex(sums[k, column], sums[nf+k, column])
                    group.half[m+1, k, r] = z*p.phases[m+1, j]
                end
            end
        end
    end
    for group in p.groups
        mul!(group.samples, group.plan, group.half)
        for r in eachindex(group.rows), k in 1:nf
            copyto!(out, p.rings.offsets[group.rows[r]]+(k-1)*size(out, 1),
                group.samples, 1+(k-1)*group.J+(r-1)*group.J*nf, group.J)
        end
    end
    return out
end
