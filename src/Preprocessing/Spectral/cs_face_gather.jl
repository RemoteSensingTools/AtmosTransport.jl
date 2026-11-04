"""Shared scalar face stencils with normals and optional Float32 rounding offsets."""
struct CSFaceGather{O,I,W,N,C}
    offsets::O
    indices::I
    weights::W
    normals::N
    rounding::C
    nsource::Int
end

function cs_gather_operator(geometry::NamedTuple{(:stencils,:normals)}, adapter, ::Type{FT}) where FT
    rows, normals = geometry.stencils, geometry.normals
    IT = max(size(rows,1), length(rows.nzval)+1) <= typemax(Int32) ? Int32 : Int64
    weights = FT.(rows.nzval)
    rounded_normals = FT.(normals)
    rounding = cs_face_rounding(rows, normals, weights, rounded_normals)
    return CSFaceGather(adapter(IT.(rows.colptr)), adapter(IT.(rows.rowval)),
        adapter(weights), adapter(rounded_normals),
        rounding === nothing ? nothing : adapter(rounding), size(rows,1))
end

cs_face_rounding(rows, normals, weights::Vector{Float64}, rounded_normals) = nothing

function cs_face_rounding(rows, normals, weights::Vector{Float32}, rounded_normals)
    # The old map rounded Float64(normal*weight) once. Multiplying separately
    # rounded factors can differ by an ulp. Store only the integer bit offset
    # needed to reproduce that rounding, avoiding Float64 device arithmetic.
    rounding = Matrix{Int8}(undef, length(weights), 3)
    for f in axes(normals,2), d in 1:3, t in rows.colptr[f]:rows.colptr[f+1]-1
        expected = Float32(normals[d,f]*rows.nzval[t])
        product = rounded_normals[d,f]*weights[t]
        delta = Int64(reinterpret(Int32,expected))-Int64(reinterpret(Int32,product))
        rounding[t,d] = Int8(delta) # checked conversion; never silently truncate
    end
    return rounding
end

@inline cs_face_weight(weight, normal, ::Nothing, t, d) = normal*weight
@inline function cs_face_weight(weight::Float32, normal::Float32, rounding::AbstractMatrix{Int8}, t, d)
    bits = reinterpret(Int32,normal*weight)
    return reinterpret(Float32,bits+Int32(rounding[t,d]))
end

@kernel function cs_apply_face_kernel!(out, values, offsets, indices, weights, normals, rounding, nsource, accumulation=Val(false))
    f, k = @index(Global, NTuple)
    if f <= size(out,1) && k <= size(out,2)
        total = zero(eltype(out))
        correction = zero(total)
        # Preserve every multiplication and addition in the expanded map,
        # including its X, Y, Z order and explicit zero-normal entries.
        for d in 1:3
            normal = normals[d,f]
            for t in offsets[f]:offsets[f+1]-1
                w = cs_face_weight(weights[t], normal, rounding, t, d)
                total, correction = face_product_step(total, correction, w,values[(d-1)*nsource+indices[t],k], accumulation)
            end
        end
        out[f,k] = face_sum_result(total, correction, accumulation)
    end
end

function cs_apply_integral!(out, values, op::CSFaceGather; accumulation=face_accumulation(get_backend(out), eltype(out)))
    backend = get_backend(out)
    cs_apply_face_kernel!(backend, 256)(out, values, op.offsets, op.indices, op.weights,
        op.normals, op.rounding, op.nsource, accumulation; ndrange=size(out))
    return out
end
