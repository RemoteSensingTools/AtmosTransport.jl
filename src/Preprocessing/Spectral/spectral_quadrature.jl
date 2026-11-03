"""Closed linear quadrature; the second weights integrate the interpolant times cosφ."""
function latitude_quadrature(south::Real, north::Real, n::Integer)
    polar = abs(south+pi/2) < 1e-12 || abs(north-pi/2) < 1e-12
    h = (north-south)/n
    if polar
        # Open midpoint rule in each polar subinterval; no pole wind division.
        nodes = [south+(i-0.5)*h for i in 1:n]
        return nodes, fill(h, n), [sin(south+i*h)-sin(south+(i-1)*h) for i in 1:n]
    end
    nodes = collect(range(south, north; length = n+1))
    plain = fill(h, n+1)
    plain[1] /= 2
    plain[end] /= 2
    weighted = zeros(n+1)
    for i in 1:n
        a, b = nodes[i], nodes[i+1]
        # Stable exact integrals of the two linear basis functions times cosφ.
        mid = (a+b)/2
        half = (b-a)/2
        total = 2cos(mid)*sin(half)
        # ∫ (φ-mid)cosφ dφ / h, avoiding cancellation at small h.
        moment = 2sin(mid)*(half*cos(half)-sin(half))/(b-a)
        weighted[i] += total/2-moment
        weighted[i+1] += total/2+moment
    end
    return nodes, plain, weighted
end

"""Target boxes, refined rows, latitude rules and boundary-segment geometry."""
struct SpectralQuadrature{Q<:AbstractFaceQuadrature}
    rule::Q
    interval_counts::Vector{Int}
    rings::SynthesisRings
    nlon::Vector{Int}
    cell_offsets::Vector{Int}
    node_offsets::Vector{Int}
    node_rows::Vector{Int}
    plain_weights::Vector{Float64}
    cosine_weights::Vector{Float64}
    boundary_rows::Vector{Int}
    boundary_counts::Vector{Int}
    boundary_offsets::Vector{Int}
    areas::Vector{Float64}
    radius::Float64
end

quadrature_geometry(g::LatLonTargetGeometry) =
    (fill(nlon(g), nlat(g)), deg2rad.(Float64.(g.mesh.φᶠ)), deg2rad(Float64(g.mesh.λᶠ[1])),
     fill(nlon(g), nlat(g)+1), Float64(g.mesh.radius))
quadrature_geometry(g::ReducedGaussianTargetGeometry) =
    (g.mesh.nlon_per_ring, deg2rad.(Float64.(g.mesh.lat_faces)), 0.0,
     g.mesh.boundary_counts, Float64(g.mesh.radius))

SpectralQuadrature(grid, T::Integer, steps::Integer=2) = SpectralQuadrature(grid, T, TM5Trapezoid(), steps)

latitude_rule(::TM5Trapezoid, south, north, T, steps) =
    latitude_quadrature(south, north, quadrature_refinement(T, rad2deg(north-south), steps))
latitude_rule(rule::GaussFourier, south, north, T, steps) = gauss_latitude_quadrature(south, north, T, rule)
ring_length(::TM5Trapezoid, nx, T, steps) =
    oversampled_ring_length(nx*quadrature_refinement(T, 360/nx, steps), T+1)
ring_length(rule::GaussFourier, nx, T, steps) = nx*cld(rule.oversampling*(T+2)+1, nx)

function SpectralQuadrature(grid, T::Integer, rule::AbstractFaceQuadrature, steps::Integer=2)
    nlon, faces, origin, counts, R = quadrature_geometry(grid)
    lats = Float64[]
    lengths = Int[]
    interval_counts = Int[]
    origins = Float64[]
    weights = Float64[]
    cosweights = Float64[]
    node_offsets = [1]
    areas = Float64[]
    node_rows = Int[]
    ring_ids = Dict{Tuple{Float64, Int, Int}, Int}()
    function ring_row(phi, J, count)
        return get!(ring_ids, (phi, J, count)) do
            push!(lats, phi)
            push!(lengths, J)
            push!(interval_counts, count)
            push!(origins, origin)
            length(lats)
        end
    end
    for j in eachindex(nlon)
        nx = nlon[j]
        J = ring_length(rule, nx, T, steps)
        nodes, w, cw = latitude_rule(rule, faces[j], faces[j+1], T, steps)
        append!(node_rows, [ring_row(phi, J, nx) for phi in nodes])
        append!(weights, w)
        append!(cosweights, cw)
        push!(node_offsets, length(node_rows)+1)
        append!(areas, fill(R^2*2pi/nx*(sin(faces[j+1])-sin(faces[j])), nx))
    end
    boundary_rows = Int[]
    for b in eachindex(faces)
        # LCM segments need no LCM-sized FFT: prefix primitives evaluate any segment.
        nx = b == 1 ? nlon[1] : b > length(nlon) ? nlon[end] : max(nlon[b-1], nlon[b])
        J = ring_length(rule, nx, T, steps)
        push!(boundary_rows, ring_row(faces[b], J, counts[b]))
    end
    return SpectralQuadrature(rule, interval_counts, SynthesisRings(lats, lengths, origins), Int.(nlon),
        cumsum([1; nlon]), node_offsets, node_rows, weights, cosweights, boundary_rows, Int.(counts),
        cumsum([1; counts]), areas, R)
end

# The primitive of a periodic piecewise-linear ring. Prefix entry i is the
# integral up to sample i; partial intervals include the exact quadratic term.
@inline function ring_primitive(values, prefix, start, J, k, x)
    FT = eltype(values)
    t = clamp(x, zero(FT), FT(J))
    i = min(floor(Int, t), J-1)
    a = t-i
    f0 = FT(values[start+i, k])
    f1 = FT(values[start+mod(i+1, J), k])
    return prefix[start+i, k] + (FT(2pi)/J)*(a*f0+FT(0.5)*a*a*(f1-f0))
end

# Each workgroup scans one 256-sample tile. The second pass adds the (short)
# prefix of tile totals, replacing a full serial longitude loop per ring.
@kernel function ring_prefix_tiles!(prefix, totals, values, offsets, lengths, nf)
    tile, row, k = @index(Group, NTuple)
    lane = @index(Local, Linear)
    scratch = @localmem eltype(prefix) (256, )
    next = @localmem eltype(prefix) (256, )
    J = lengths[row]
    i = (tile-1)*256+lane
    start = offsets[row]
    FT = eltype(prefix)
    value = zero(FT)
    if i <= J && k <= nf
        value = (FT(pi)/J)*(FT(values[start+i-1, k])+FT(values[start+mod(i, J), k]))
    end
    scratch[lane] = value
    @synchronize
    for shift in (1, 2, 4, 8, 16, 32, 64, 128)
        next[lane] = scratch[lane] + (lane > shift ? scratch[lane-shift] : zero(eltype(prefix)))
        @synchronize
        scratch[lane] = next[lane]
        @synchronize
    end
    # Recompute index-derived scalars after barriers for the KA CPU lowering.
    FT = eltype(prefix)
    J = lengths[row]
    i = (tile-1)*256+lane
    start = offsets[row]
    if i <= J && k <= nf
        value = (FT(pi)/J)*(FT(values[start+i-1, k])+FT(values[start+mod(i, J), k]))
        prefix[start+i-1, k] = scratch[lane]-value
    end
    if lane == 256
        totals[tile, row, k] = scratch[lane]
    end
end

@kernel function ring_prefix_offsets!(prefix, totals, offsets, lengths, nf)
    i, row, k = @index(Global, NTuple)
    if row <= length(lengths) && i <= lengths[row] && k <= nf
        tile = (i-1)÷256+1
        offset = zero(eltype(prefix))
        for t in 1:tile-1
            offset += totals[t, row, k]
        end
        prefix[offsets[row]+i-1, k] += offset
    end
end

"""Periodic trapezoid prefix with a tiled parallel scan on CPU and device arrays."""
function ring_prefix!(prefix, values, geometry, nf, max_length)
    backend = get_backend(values)
    nr = length(geometry.lengths)
    ntile = cld(max_length, 256)
    totals = similar(values, eltype(values), ntile, nr, nf)
    ring_prefix_tiles!(backend, (256, 1, 1))(prefix, totals, values, geometry.offsets, geometry.lengths, nf;
        ndrange = (ntile*256, nr, nf))
    ring_prefix_offsets!(backend, 256)(prefix, totals, geometry.offsets, geometry.lengths, nf;
        ndrange = (max_length, nr, nf))
    synchronize(backend)
    return prefix
end
