"""
    IdealizedAdvection

Analytic flows, initial fields and grid geometry for the idealized advection tests of the GMD
paper (Sect. 4.1). The module turns an analytic wind into ordinary transport binaries, writes
initial-condition files that the existing `file` and `cs_native` initial-condition kinds read,
and evaluates the analytic solution as cell means for the error norms. Nothing here is used by
the runtime: the tests run through `scripts/run_transport.jl` like any other case.

Conventions (all on the unit sphere unless stated):

- `x` is a unit position vector; longitude `λ`, latitude `θ` in radians.
- The velocity is `V = R ẋ` with `ẋ = velocity(flow, x, t)` [s⁻¹].
- A stream function `ψ` [m² s⁻¹] gives `V = r̂ × ∇ψ`, i.e. `u = -∂ψ/(R∂θ)`,
  `v = ∂ψ/(R cos θ ∂λ)`.
- A face is a path `A → B` whose right-hand normal `n = t̂ × r̂` points from the face's
  `left` cell to its `right` cell. Its volume transfer is `∫ V·n dl`, which for a stream
  function is `-(ψ(B) - ψ(A))`.

Flows: solid-body rotation (Williamson et al., 1992, case 1), and the non-divergent and the
divergent deformational flow of Lauritzen et al. (2012, their Eqs. 18-19 and 21-22; Nair and
Lauritzen, 2010, cases 4 and 3 with a constant background wind). Their nondimensional forms
(R = 1, T = 5) are rescaled to a period `T` in seconds.
"""
module IdealizedAdvection

using LinearAlgebra
using Printf
using Dates
using StaticArrays
using FastGaussQuadrature: gausslegendre
using NCDatasets

using AtmosTransport
using AtmosTransport.Grids
using AtmosTransport.Grids: CubedSphereMesh, LatLonMesh, ReducedGaussianMesh, GEOSNativePanelConvention,
                            cs_corner_xyz, cs_flat_index, build_cs_global_face_table, face_cells,
                            nfaces, ncells, nrings, cell_area, HybridSigmaPressure, AtmosGrid
using AtmosTransport.Parameters: EARTH_RADIUS, STANDARD_GRAVITY, SECONDS_PER_DAY

const MD = AtmosTransport.MetDrivers
const Vec3 = SVector{3, Float64}

export SolidBodyRotation, NonDivergentDeformation, DivergentDeformation
export CosineBells, GaussianHills, SlottedCylinders, UniformField
export GridGeometry, build_geometry, flow_case

# ---------------------------------------------------------------------------
# Sphere helpers
# ---------------------------------------------------------------------------

@inline unit_xyz(λ, θ) = Vec3(cos(θ) * cos(λ), cos(θ) * sin(λ), sin(θ))
@inline lon_of(x) = mod(atan(x[2], x[1]), 2π)
@inline lat_of(x) = asin(clamp(x[3], -1.0, 1.0))
@inline east_unit(λ) = Vec3(-sin(λ), cos(λ), 0.0)
@inline north_unit(λ, θ) = Vec3(-sin(θ) * cos(λ), -sin(θ) * sin(λ), cos(θ))
@inline great_circle_distance(a, b) = atan(norm(cross(a, b)), dot(a, b))

"""Area of the spherical triangle `abc` on the unit sphere (Van Oosterom and Strackee, 1983)."""
@inline function spherical_triangle_area(a, b, c)
    numerator = abs(dot(a, cross(b, c)))
    denominator = 1 + dot(a, b) + dot(b, c) + dot(c, a)
    return 2 * atan(numerator, denominator)
end

"""Rotate `x` about the unit axis `k` by `angle` (Rodrigues)."""
@inline rotate(x, k, angle) = x * cos(angle) + cross(k, x) * sin(angle) + k * dot(k, x) * (1 - cos(angle))

# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------

abstract type AbstractFlow end
"""Flows with a stream function: discretely non-divergent face transfers, constant air mass."""
abstract type StreamFunctionFlow <: AbstractFlow end

"""
    SolidBodyRotation(; tilt_deg = 45, period = 12 days)

Williamson et al. (1992) case 1: rotation about the axis `k = (-sin α, 0, cos α)` with one
revolution per `period`; `α = 0` is eastward flow along the equator.
"""
Base.@kwdef struct SolidBodyRotation <: StreamFunctionFlow
    tilt_deg::Float64 = 45.0
    period::Float64 = 12 * SECONDS_PER_DAY
end
rotation_axis(f::SolidBodyRotation) = Vec3(-sind(f.tilt_deg), 0.0, cosd(f.tilt_deg))
angular_speed(f::SolidBodyRotation) = 2π / f.period
streamfunction(f::SolidBodyRotation, x, t) = -angular_speed(f) * EARTH_RADIUS^2 * dot(rotation_axis(f), x)
velocity(f::SolidBodyRotation, x, t) = angular_speed(f) * cross(rotation_axis(f), x)
is_steady(::AbstractFlow) = false
is_steady(::SolidBodyRotation) = true
"""Exact departure point at time 0 of the parcel at `x` at time `t`."""
departure_point(f::SolidBodyRotation, x, t) = rotate(x, rotation_axis(f), -angular_speed(f) * t)

"""
    NonDivergentDeformation(; κ = 2, period = 12 days)

Lauritzen et al. (2012), Eqs. (18)-(20): deformational flow with a background zonal rotation;
the field returns to its initial position at `t = period`. Nondimensional stream function with
`s = t/T`, `λ' = λ - 2πs`:  `ψ̃ = κ sin²λ' cos²θ cos(πs) - (2π/5) sin θ`; time unit `T/5`, so
`κ = 2` is their coefficient `10 R/T`.
"""
Base.@kwdef struct NonDivergentDeformation <: StreamFunctionFlow
    κ::Float64 = 2.0
    period::Float64 = 12 * SECONDS_PER_DAY
end
time_unit(f::AbstractFlow) = f.period / 5     # the nondimensional period of the reference cases is 5
function streamfunction(f::NonDivergentDeformation, x, t)
    λ, θ = lon_of(x), lat_of(x)
    s = t / f.period
    λp = λ - 2π * s
    ψ̃ = f.κ * sin(λp)^2 * cos(θ)^2 * cos(π * s) - (2π / 5) * sin(θ)
    return ψ̃ * EARTH_RADIUS^2 / time_unit(f)
end
function velocity(f::NonDivergentDeformation, x, t)
    λ, θ = lon_of(x), lat_of(x)
    s = t / f.period
    λp = λ - 2π * s
    u = f.κ * sin(λp)^2 * sin(2θ) * cos(π * s) + (2π / 5) * cos(θ)
    v = f.κ * sin(2λp) * cos(θ) * cos(π * s)
    return (u * east_unit(λ) + v * north_unit(λ, θ)) / time_unit(f)
end

"""
    DivergentDeformation(; κ = 1, period = 12 days)

Lauritzen et al. (2012), Eqs. (21)-(22): divergent deformational flow with a background zonal
rotation, `ũ = -κ sin²(λ'/2) sin 2θ cos²θ cos(πs) + (2π/5) cos θ`,
`ṽ = (κ/2) sin λ' cos³θ cos(πs)`, `λ' = λ - 2πs`; `κ = 1` is their coefficient `5 R/T`. In the
frame rotating with the background wind the flow reverses at `T/2`, so the field and the air
mass return to their initial state at `t = period`.
"""
Base.@kwdef struct DivergentDeformation <: AbstractFlow
    κ::Float64 = 1.0
    period::Float64 = 12 * SECONDS_PER_DAY
end
function velocity(f::DivergentDeformation, x, t)
    λ, θ = lon_of(x), lat_of(x)
    s = t / f.period
    λp = λ - 2π * s
    u = -f.κ * sin(λp / 2)^2 * sin(2θ) * cos(θ)^2 * cos(π * s) + (2π / 5) * cos(θ)
    v = (f.κ / 2) * sin(λp) * cos(θ)^3 * cos(π * s)
    return (u * east_unit(λ) + v * north_unit(λ, θ)) / time_unit(f)
end

# ---------------------------------------------------------------------------
# Trajectories and density of the deformational flows (pseudo time in the rotating frame)
# ---------------------------------------------------------------------------
#
# Both deformational flows are a steady field W in the frame rotating with the background wind
# (`λ' = λ − 2πt/T`), modulated by `cos(πt/T)`: `V' = cos(πt/T) W(x')`. A parcel therefore
# follows W in the pseudo time `τ(t) = (T/π) sin(πt/T)` (`dτ = cos(πt/T) dt`), and its density
# obeys `d ln ρ/dτ = −div W` (the background rotation is non-divergent). In time units of `T/5`
# (the nondimensional time of the reference cases) `τ̃(t) = (5/π) sin(πt/T)`.

const DeformationalFlow = Union{NonDivergentDeformation, DivergentDeformation}
# RK4 step in τ̃ (measured on 2000 random points at T/2: positions converge as h⁴, 1.5e-9 (B) and
# 9.5e-12 (C) at h = 0.005; the face transfers of C use 0.01, log-density error 1.5e-10).
const DEFAULT_PSEUDO_STEP = 0.005
const FLUX_PSEUDO_STEP = 0.01

"""Position in the frame rotating with the background wind (`λ' = λ − 2πt/T`)."""
to_rotating(f::DeformationalFlow, x, t) = rotate(x, Vec3(0.0, 0.0, 1.0), -2π * t / f.period)
"""Pseudo time `τ̃(t) = (5/π) sin(πt/T)` in units of `T/5`."""
pseudo_time(f::DeformationalFlow, t) = (5 / π) * sin(π * t / f.period)

"""Steady deformational wind `W̃(x')` in the rotating frame (unit sphere per `T/5`)."""
function deformation_wind(f::NonDivergentDeformation, x)
    λ, θ = lon_of(x), lat_of(x)
    return f.κ * sin(λ)^2 * sin(2θ) * east_unit(λ) + f.κ * sin(2λ) * cos(θ) * north_unit(λ, θ)
end
function deformation_wind(f::DivergentDeformation, x)
    λ, θ = lon_of(x), lat_of(x)
    u = -f.κ * sin(λ / 2)^2 * sin(2θ) * cos(θ)^2
    v = (f.κ / 2) * sin(λ) * cos(θ)^3
    return u * east_unit(λ) + v * north_unit(λ, θ)
end
"""`div W̃` on the unit sphere: `(1/cos θ)(∂u/∂λ + ∂(v cos θ)/∂θ)`."""
deformation_divergence(::NonDivergentDeformation, x) = 0.0
function deformation_divergence(f::DivergentDeformation, x)
    λ, θ = lon_of(x), lat_of(x)
    return -3 * f.κ * sin(λ) * sin(θ) * cos(θ)^2
end

"""
    trace_back(flow, x, t; h) -> (x0, log(ρ/ρ0))

Departure point at time 0 of the parcel at `x` at time `t` and the log of its density ratio:
RK4 backward in pseudo time with `ceil(τ̃/h)` steps, integrating `div W̃` along the path.
"""
function trace_back(f::DeformationalFlow, x, t; h = DEFAULT_PSEUDO_STEP)
    xr = to_rotating(f, x, t)
    τ = pseudo_time(f, t)
    n = ceil(Int, abs(τ) / h)
    n == 0 && return xr, 0.0
    dτ = -τ / n
    W(y) = deformation_wind(f, y)
    D(y) = deformation_divergence(f, y)
    logρ = 0.0
    @inbounds for _ in 1:n
        y2 = normalize(xr + dτ / 2 * W(xr))
        y3 = normalize(xr + dτ / 2 * W(y2))
        y4 = normalize(xr + dτ * W(y3))
        logρ += dτ / 6 * (D(xr) + 2D(y2) + 2D(y3) + D(y4))
        xr = normalize(xr + dτ / 6 * (W(xr) + 2W(y2) + 2W(y3) + W(y4)))
    end
    return xr, logρ
end
trace_back(f::SolidBodyRotation, x, t; h = 0.0) = departure_point(f, x, t), 0.0

"""Density ratio `ρ(x, t)/ρ0` (1 for the non-divergent flows)."""
density_ratio(f::DivergentDeformation, x, t; h = DEFAULT_PSEUDO_STEP) = exp(last(trace_back(f, x, t; h)))

"""
Departure point by RK4 in physical time with the full time-dependent wind (720 steps by default);
kept as an independent check of the pseudo-time trajectories.
"""
function departure_point_rk4(f::AbstractFlow, x, t; steps::Int = 720)
    t == 0 && return x
    h = -t / steps
    τ = t
    @inbounds for _ in 1:steps
        k1 = velocity(f, x, τ)
        k2 = velocity(f, normalize(x + h / 2 * k1), τ + h / 2)
        k3 = velocity(f, normalize(x + h / 2 * k2), τ + h / 2)
        k4 = velocity(f, normalize(x + h * k3), τ + h)
        x = normalize(x + h / 6 * (k1 + 2k2 + 2k3 + k4))
        τ += h
    end
    return x
end

# ---------------------------------------------------------------------------
# Initial fields (mixing ratios; values follow the reference papers)
# ---------------------------------------------------------------------------

abstract type AbstractTracerField end

"""Cosine bells `b + c Σᵢ hᵢ`, `hᵢ = (hmax/2)(1 + cos(π rᵢ/r))` inside great-circle radius `r`."""
struct CosineBells <: AbstractTracerField
    centers::Vector{Vec3}
    radius::Float64
    hmax::Float64
    background::Float64
    scale::Float64
end
function (f::CosineBells)(x)
    h = 0.0
    for c in f.centers
        r = great_circle_distance(x, c)
        r < f.radius && (h += f.hmax / 2 * (1 + cos(π * r / f.radius)))
    end
    return f.background + f.scale * h
end

"""Gaussian hills `hmax Σᵢ exp(-b0 |x - xᵢ|²)` (Euclidean distance on the unit sphere)."""
struct GaussianHills <: AbstractTracerField
    centers::Vector{Vec3}
    b0::Float64
    hmax::Float64
end
(f::GaussianHills)(x) = f.hmax * sum(exp(-f.b0 * sum(abs2, x - c)) for c in f.centers)

"""Slotted cylinders of Lauritzen et al. (2012): value `c` inside, `b` outside and in the slots."""
struct SlottedCylinders <: AbstractTracerField
    centers::Vector{Tuple{Float64, Float64}}   # (λ, θ)
    radius::Float64
    background::Float64
    value::Float64
end
function (f::SlottedCylinders)(x)
    λ, θ = lon_of(x), lat_of(x)
    r = f.radius
    for (i, (λi, θi)) in enumerate(f.centers)
        great_circle_distance(x, unit_xyz(λi, θi)) <= r || continue
        abs(λ - λi) >= r / 6 && return f.value
        i == 1 && θ - θi < -5r / 12 && return f.value
        i == 2 && θ - θi > 5r / 12 && return f.value
    end
    return f.background
end

struct UniformField <: AbstractTracerField
    value::Float64
end
(f::UniformField)(x) = f.value

const LAURITZEN_CENTERS = ((5π / 6, 0.0), (7π / 6, 0.0))
lauritzen_centers_xyz() = [unit_xyz(λ, θ) for (λ, θ) in LAURITZEN_CENTERS]

"""
    flow_case(id) -> (flow, tracers)

The flows of this stage and the tracers each run carries (name => field).
"""
function flow_case(id::AbstractString; tilt_deg::Real = 45.0)
    if id == "A"
        flow = SolidBodyRotation(; tilt_deg = Float64(tilt_deg))
        tracers = ["cosine_bell" => CosineBells([unit_xyz(3π / 2, 0.0)], 1 / 3, 1.0, 0.0, 1.0),
                   "gaussian_hill" => GaussianHills([unit_xyz(π / 2, 0.0)], 5.0, 1.0),
                   "uniform" => UniformField(1.0)]
    elseif id == "B"
        flow = NonDivergentDeformation()
        tracers = ["gaussian_hills" => GaussianHills(lauritzen_centers_xyz(), 5.0, 0.95),
                   "cosine_bells" => CosineBells(lauritzen_centers_xyz(), 0.5, 1.0, 0.1, 0.9),
                   "slotted_cylinders" => SlottedCylinders(collect(LAURITZEN_CENTERS), 0.5, 0.1, 1.0),
                   "uniform" => UniformField(1.0)]
    elseif id == "C"
        flow = DivergentDeformation()
        tracers = ["gaussian_hills" => GaussianHills(lauritzen_centers_xyz(), 5.0, 0.95),
                   "cosine_bells" => CosineBells(lauritzen_centers_xyz(), 0.5, 1.0, 0.1, 0.9),
                   "uniform" => UniformField(1.0)]
    else
        throw(ArgumentError("unknown flow '$id' (A, B or C)"))
    end
    return flow, tracers
end

flow_description(f::SolidBodyRotation) = @sprintf("solid-body rotation, tilt %.0f deg, period %.1f d",
                                                  f.tilt_deg, f.period / SECONDS_PER_DAY)
flow_description(f::NonDivergentDeformation) = @sprintf("non-divergent deformation (kappa %.2f), period %.1f d",
                                                        f.κ, f.period / SECONDS_PER_DAY)
flow_description(f::DivergentDeformation) = @sprintf("divergent deformation (kappa %.2f), period %.1f d",
                                                     f.κ, f.period / SECONDS_PER_DAY)

# ---------------------------------------------------------------------------
# Face paths
# ---------------------------------------------------------------------------

"""Great-circle arc from corner `a` to corner `b` (indices into the corner table)."""
struct GreatCircleArc
    a::Int
    b::Int
end
"""Arc along latitude `θ` from longitude `λa` to `λb` (radians), corner indices `a`, `b`."""
struct LatitudeArc
    a::Int
    b::Int
    θ::Float64
    λa::Float64
    λb::Float64
end
const FacePath = Union{GreatCircleArc, LatitudeArc}
reverse_path(p::GreatCircleArc) = GreatCircleArc(p.b, p.a)
reverse_path(p::LatitudeArc) = LatitudeArc(p.b, p.a, p.θ, p.λb, p.λa)

"""Point and derivative `dx/ds` of a path at `s ∈ [0, 1]`."""
function path_point(p::GreatCircleArc, corners, s)
    a, b = corners[p.a], corners[p.b]
    ω = atan(norm(cross(a, b)), dot(a, b))
    x = (sin((1 - s) * ω) * a + sin(s * ω) * b) / sin(ω)
    dx = ω * (-cos((1 - s) * ω) * a + cos(s * ω) * b) / sin(ω)
    return x, dx
end
function path_point(p::LatitudeArc, corners, s)
    λ = p.λa + s * (p.λb - p.λa)
    return unit_xyz(λ, p.θ), (p.λb - p.λa) * cos(p.θ) * east_unit(λ)
end

"""`∫ ẋ·(dx/ds × x) ds` along the path: the transfer `∫ V·n dl` divided by `R²`."""
function path_flux(p::FacePath, corners, flow, t, nodes, weights)
    acc = 0.0
    for (ξ, w) in zip(nodes, weights)
        x, dx = path_point(p, corners, (ξ + 1) / 2)
        acc += w / 2 * dot(velocity(flow, x, t), cross(dx, x))
    end
    return acc
end

# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------

abstract type AbstractIdealizedGrid end
struct CubedSphereGrid <: AbstractIdealizedGrid
    mesh::CubedSphereMesh
    table::Any
end
struct LatLonGrid <: AbstractIdealizedGrid
    mesh::LatLonMesh
end
struct ReducedGaussianGrid <: AbstractIdealizedGrid
    mesh::ReducedGaussianMesh
end

"""
    GridGeometry

Flat geometry of one horizontal grid: cells (native flat order of the model output), unique
corners, and faces with left/right cells and a path oriented so that positive transfer goes
from left to right. Faces with a pole on one side (reduced-Gaussian pole caps) are omitted.
"""
struct GridGeometry{G <: AbstractIdealizedGrid}
    grid::G
    label::String
    cell_area::Vector{Float64}       # m²
    cell_center::Vector{Vec3}
    corners::Vector{Vec3}
    face_left::Vector{Int}
    face_right::Vector{Int}
    face_path::Vector{FacePath}
end
ncell(g::GridGeometry) = length(g.cell_area)
nface(g::GridGeometry) = length(g.face_left)

"""Unique-corner registry: points that coincide mathematically share one index and one value."""
struct CornerRegistry
    index::Dict{NTuple{3, Int}, Int}
    points::Vector{Vec3}
end
CornerRegistry() = CornerRegistry(Dict{NTuple{3, Int}, Int}(), Vec3[])
function register!(r::CornerRegistry, x)
    key = (round(Int, x[1] * 1e10), round(Int, x[2] * 1e10), round(Int, x[3] * 1e10))
    return get!(r.index, key) do
        push!(r.points, Vec3(x))
        length(r.points)
    end
end

function orient(path::FacePath, corners, cl, cr)
    x, dx = path_point(path, corners, 0.5)
    return dot(cross(dx, x), cr - cl) >= 0 ? path : reverse_path(path)
end

"""
    build_geometry(label) -> GridGeometry

`label`: `c<N>` (GEOS-native cubed sphere), `ll<Nx>x<Ny>` (regular latitude-longitude) or
`n<N>` (ECMWF reduced Gaussian; ring sizes read from `n<N>_pl.txt` in `pl_dir`).
"""
function build_geometry(label::AbstractString; pl_dir::AbstractString = "")
    if startswith(label, "c")
        Nc = parse(Int, label[2:end])
        mesh = CubedSphereMesh(; Nc, convention = GEOSNativePanelConvention())
        return _cs_geometry(CubedSphereGrid(mesh, build_cs_global_face_table(Nc, mesh.connectivity)), label)
    elseif startswith(label, "ll")
        Nx, Ny = parse.(Int, split(label[3:end], "x"))
        return _ll_geometry(LatLonGrid(LatLonMesh(; Nx, Ny)), label)
    elseif startswith(label, "n")
        N = parse(Int, label[2:end])
        file = joinpath(pl_dir, "n$(N)_pl.txt")
        isfile(file) || throw(ArgumentError("ring-size table $file not found (pass pl_dir)"))
        nlon = parse.(Int, filter(!isempty, strip.(readlines(file))))
        length(nlon) == 2N || throw(ArgumentError("$file has $(length(nlon)) rings, expected $(2N)"))
        lats = reverse(asind.(gausslegendre(2N)[1]))          # north -> south ...
        mesh = ReducedGaussianMesh(sort(lats), nlon)          # ... stored south -> north
        return _rg_geometry(ReducedGaussianGrid(mesh), label)
    end
    throw(ArgumentError("unknown grid label '$label'"))
end

function _cs_geometry(grid::CubedSphereGrid, label)
    mesh, table = grid.mesh, grid.table
    Nc = mesh.Nc
    reg = CornerRegistry()
    corner_id = Array{Int}(undef, Nc + 1, Nc + 1, 6)
    for p in 1:6, j in 1:Nc+1, i in 1:Nc+1
        corner_id[i, j, p] = register!(reg, Vec3(cs_corner_xyz(mesh, i, j, p)...))
    end
    ncells_total = 6 * Nc^2
    area = Vector{Float64}(undef, ncells_total)
    center = Vector{Vec3}(undef, ncells_total)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        c = cs_flat_index(i, j, p, Nc)
        area[c] = Float64(mesh.cell_areas[i, j])
        center[c] = normalize(reg.points[corner_id[i, j, p]] + reg.points[corner_id[i+1, j, p]] +
                              reg.points[corner_id[i+1, j+1, p]] + reg.points[corner_id[i, j+1, p]])
    end
    left = Int.(table.face_left)
    right = Int.(table.face_right)
    paths = Vector{FacePath}(undef, table.nf)
    for f in 1:table.nf
        p, i, j = table.face_panel[f], table.face_idx_i[f], table.face_idx_j[f]
        a = corner_id[i, j, p]
        b = table.face_dir[f] == 1 ? corner_id[i, j+1, p] : corner_id[i+1, j, p]
        paths[f] = orient(GreatCircleArc(a, b), reg.points, center[left[f]], center[right[f]])
    end
    return GridGeometry(grid, label, area, center, reg.points, left, right, paths)
end

function _ll_geometry(grid::LatLonGrid, label)
    mesh = grid.mesh
    Nx, Ny = mesh.Nx, mesh.Ny
    λf, θf = deg2rad.(Float64.(mesh.λᶠ)), deg2rad.(Float64.(mesh.φᶠ))
    λc, θc = deg2rad.(Float64.(mesh.λᶜ)), deg2rad.(Float64.(mesh.φᶜ))
    R = Float64(mesh.radius)
    cell(i, j) = i + (j - 1) * Nx
    area = [R^2 * (λf[i+1] - λf[i]) * (sin(θf[j+1]) - sin(θf[j])) for i in 1:Nx, j in 1:Ny][:]
    center = [unit_xyz(λc[i], θc[j]) for i in 1:Nx, j in 1:Ny][:]
    reg = CornerRegistry()
    cid = [register!(reg, unit_xyz(λf[i], θf[j])) for i in 1:Nx+1, j in 1:Ny+1]
    left, right, paths = Int[], Int[], FacePath[]
    for j in 1:Ny, i in 1:Nx                                   # x faces: west face of cell i
        l, r = cell(i == 1 ? Nx : i - 1, j), cell(i, j)
        push!(left, l); push!(right, r)
        push!(paths, orient(GreatCircleArc(cid[i, j], cid[i, j+1]), reg.points, center[l], center[r]))
    end
    for j in 2:Ny, i in 1:Nx                                   # y faces: south face of row j
        l, r = cell(i, j - 1), cell(i, j)
        push!(left, l); push!(right, r)
        path = LatitudeArc(cid[i, j], cid[i+1, j], θf[j], λf[i], λf[i+1])
        push!(paths, orient(path, reg.points, center[l], center[r]))
    end
    return GridGeometry(grid, label, area, center, reg.points, left, right, paths)
end

# Longitudes of ring partitions as exact fractions of the circle, so shared corners coincide.
@inline _ring_lon(k, n) = 2π * Float64((k - 1) // n)

function _rg_geometry(grid::ReducedGaussianGrid, label)
    mesh = grid.mesh
    θf = deg2rad.(Float64.(mesh.lat_faces))
    nc = ncells(mesh)
    area = [Float64(cell_area(mesh, c)) for c in 1:nc]
    center = Vector{Vec3}(undef, nc)
    for j in 1:nrings(mesh), i in 1:mesh.nlon_per_ring[j]
        n = mesh.nlon_per_ring[j]
        center[mesh.ring_offsets[j] + i - 1] = unit_xyz(2π * (i - 0.5) / n, deg2rad(Float64(mesh.latitudes[j])))
    end
    reg = CornerRegistry()
    left, right, paths = Int[], Int[], FacePath[]
    for f in 1:nfaces(mesh)
        l, r = face_cells(mesh, f)
        (l == 0 || r == 0) && continue                         # pole caps carry no transfer
        if f <= nc                                             # zonal (x) face: west face of cell r
            j = searchsortedlast(mesh.ring_offsets, r)
            i = r - mesh.ring_offsets[j] + 1
            λ = _ring_lon(i, mesh.nlon_per_ring[j])
            a = register!(reg, unit_xyz(λ, θf[j]))
            b = register!(reg, unit_xyz(λ, θf[j+1]))
            path = GreatCircleArc(a, b)
        else                                                   # meridional (y) segment
            yf = f - nc
            bnd = searchsortedlast(mesh.boundary_offsets, yf)
            seg = yf - mesh.boundary_offsets[bnd] + 1
            nseg = mesh.boundary_counts[bnd]
            λa, λb = _ring_lon(seg, nseg), _ring_lon(seg + 1, nseg)
            seg == nseg && (λb = 2π)
            a = register!(reg, unit_xyz(λa, θf[bnd]))
            b = register!(reg, unit_xyz(λb, θf[bnd]))
            path = LatitudeArc(a, b, θf[bnd], λa, λb)
        end
        push!(left, l); push!(right, r)
        push!(paths, orient(path, reg.points, center[l], center[r]))
    end
    return GridGeometry(grid, label, area, center, reg.points, left, right, paths)
end

"""Index of each geometry face in the RG mesh's face numbering (pole caps excluded)."""
function rg_face_index(g::GridGeometry{ReducedGaussianGrid})
    mesh = g.grid.mesh
    idx = Int[]
    for f in 1:nfaces(mesh)
        l, r = face_cells(mesh, f)
        (l == 0 || r == 0) || push!(idx, f)
    end
    return idx
end

# ---------------------------------------------------------------------------
# Cell quadrature for the analytic cell means
# ---------------------------------------------------------------------------

"""
    cell_quadrature(g, k; cells = 1:ncell(g)) -> (points, weights)

`k²` points per cell of `cells` (cell-major, in the order of `cells`), weights normalized per
cell. Lat-lon and reduced-Gaussian cells are longitude × sin(latitude) rectangles:
Gauss-Legendre in both, exact area weights. Cubed-sphere cells are split into `k × k` sub-cells
of the nested `C(kN)` mesh (the GMAO equal-distance grid nests exactly): sub-cell centre, with
the spherical sub-cell area as weight.
"""
function cell_quadrature(g::GridGeometry{CubedSphereGrid}, k::Int; cells = 1:ncell(g))
    Nc = g.grid.mesh.Nc
    fine = CubedSphereMesh(; Nc = k * Nc, convention = GEOSNativePanelConvention())
    q = k * k
    points = Vector{Vec3}(undef, length(cells) * q)
    weights = Vector{Float64}(undef, length(cells) * q)
    corner(i, j, p) = Vec3(cs_corner_xyz(fine, i, j, p)...)
    Threads.@threads for n in eachindex(cells)
        c = cells[n]
        p = (c - 1) ÷ (Nc * Nc) + 1
        j = ((c - 1) % (Nc * Nc)) ÷ Nc + 1
        i = (c - 1) % Nc + 1
        base = (n - 1) * q
        m = 0
        for b in 1:k, a in 1:k
            I, J = k * (i - 1) + a, k * (j - 1) + b
            x1, x2 = corner(I, J, p), corner(I + 1, J, p)
            x3, x4 = corner(I + 1, J + 1, p), corner(I, J + 1, p)
            m += 1
            points[base+m] = normalize(x1 + x2 + x3 + x4)
            weights[base+m] = spherical_triangle_area(x1, x2, x3) + spherical_triangle_area(x1, x3, x4)
        end
        s = sum(@view weights[base+1:base+q])
        weights[base+1:base+q] ./= s
    end
    return points, weights
end

function _rectangle_quadrature(bounds, k)
    ξ, w = gausslegendre(k)
    q = k * k
    points = Vector{Vec3}(undef, length(bounds) * q)
    weights = Vector{Float64}(undef, length(bounds) * q)
    for (c, (λa, λb, θa, θb)) in enumerate(bounds)
        μa, μb = sin(θa), sin(θb)
        n = 0
        for b in 1:k, a in 1:k
            n += 1
            λ = λa + (ξ[a] + 1) / 2 * (λb - λa)
            μ = μa + (ξ[b] + 1) / 2 * (μb - μa)
            points[(c-1)*q+n] = unit_xyz(λ, asin(μ))
            weights[(c-1)*q+n] = w[a] * w[b] / 4
        end
    end
    return points, weights
end

function cell_quadrature(g::GridGeometry{LatLonGrid}, k::Int; cells = 1:ncell(g))
    mesh = g.grid.mesh
    λf, θf = deg2rad.(Float64.(mesh.λᶠ)), deg2rad.(Float64.(mesh.φᶠ))
    bounds = [(λf[i], λf[i+1], θf[j], θf[j+1]) for i in 1:mesh.Nx, j in 1:mesh.Ny][:]
    return _rectangle_quadrature(bounds[cells], k)
end

function cell_quadrature(g::GridGeometry{ReducedGaussianGrid}, k::Int; cells = 1:ncell(g))
    mesh = g.grid.mesh
    θf = deg2rad.(Float64.(mesh.lat_faces))
    bounds = NTuple{4, Float64}[]
    for j in 1:nrings(mesh), i in 1:mesh.nlon_per_ring[j]
        n = mesh.nlon_per_ring[j]
        push!(bounds, (2π * (i - 1) / n, 2π * i / n, θf[j], θf[j+1]))
    end
    return _rectangle_quadrature(bounds[cells], k)
end

"""
    trace(flow, points, t; h) -> (departure points, log density ratios)

Departure point at time 0 and `log(ρ/ρ0)` of the parcel at each point at time `t` (exact for the
solid-body rotation; pseudo-time RK4 for the deformational flows).
"""
function trace(flow, points, t; h = DEFAULT_PSEUDO_STEP)
    dep = similar(points)
    logρ = Vector{Float64}(undef, length(points))
    Threads.@threads for n in eachindex(points)
        dep[n], logρ[n] = trace_back(flow, points[n], t; h)
    end
    return dep, logρ
end

"""
    reference_means(field, dep, logρ, weights, q) -> means

Air-mass-weighted cell means `Σ w ρ χ / Σ w ρ` of `field` at the departure points (`q` points
per cell).
"""
function reference_means(field, dep, logρ, weights, q)
    nc = length(dep) ÷ q
    means = Vector{Float64}(undef, nc)
    Threads.@threads for c in 1:nc
        num = den = 0.0
        for n in (c-1)*q+1:c*q
            w = weights[n] * exp(logρ[n])
            num += w * field(dep[n])
            den += w
        end
        means[c] = num / den
    end
    return means
end

"""Smallest and largest sample of `field` in each cell (`q` departure points per cell)."""
function sample_range(field, dep, q)
    nc = length(dep) ÷ q
    lo = Vector{Float64}(undef, nc)
    hi = Vector{Float64}(undef, nc)
    Threads.@threads for c in 1:nc
        a, b = Inf, -Inf
        for n in (c-1)*q+1:c*q
            v = field(dep[n])
            a, b = min(a, v), max(b, v)
        end
        lo[c], hi[c] = a, b
    end
    return lo, hi
end

"""Fields whose cell means need a finer quadrature where a cell is cut by a discontinuity."""
is_discontinuous(::AbstractTracerField) = false
is_discontinuous(::SlottedCylinders) = true

const CUT_SCREEN_POINTS = 16     # samples per direction in the second ring of cells around a discontinuity

"""
    cut_cells(flow, field, g, t, lo, hi; h) -> cells

Cells that a discontinuity of `field` may cross at time `t`, to be evaluated with the fine
quadrature; `lo` and `hi` are the sample ranges of the coarse quadrature. A discontinuity can
clip a thin strip or a corner of a cell between the coarse samples, so the selection is wider
than the cells whose samples differ:

- every cell whose coarse samples differ, every face neighbour of such a cell, and both cells of
  a face whose two sides have different values (a strip along a face is found from the cell on
  its other side);
- of the face neighbours of those cells, the ones in which `CUT_SCREEN_POINTS²` samples differ
  or differ from the coarse value (a corner clipped through a vertex).

A cell the fine quadrature evaluates without being cut keeps its value.
"""
function cut_cells(flow, field, g::GridGeometry, t, lo, hi; h = DEFAULT_PSEUDO_STEP)
    mixed = lo .!= hi
    cut = copy(mixed)
    for f in 1:nface(g)
        a, b = g.face_left[f], g.face_right[f]
        if mixed[a] || mixed[b] || lo[a] != lo[b]
            cut[a] = cut[b] = true
        end
    end
    ring = falses(length(cut))
    for f in 1:nface(g)
        a, b = g.face_left[f], g.face_right[f]
        cut[a] && !cut[b] && (ring[b] = true)
        cut[b] && !cut[a] && (ring[a] = true)
    end
    screened = findall(ring)
    if !isempty(screened)
        points, _ = cell_quadrature(g, CUT_SCREEN_POINTS; cells = screened)
        dep, _ = trace(flow, points, t; h)
        slo, shi = sample_range(field, dep, CUT_SCREEN_POINTS^2)
        for (n, c) in enumerate(screened)
            cut[c] = slo[n] != shi[n] || slo[n] != lo[c]
        end
    end
    return findall(cut)
end

"""
    analytic_cell_means(flow, tracers, g, t; k, k_fine, h) -> (means, cell_air, refined)

Air-mass-weighted analytic cell means of every tracer at time `t`, with `k × k` points per cell;
for discontinuous fields the cells cut by the discontinuity (`cut_cells`) are evaluated again
with `k_fine × k_fine` points. Also returns the exact cell integral of `ρ/ρ0` (per unit `ρ0`,
m²) and the number of refined cells per tracer.
"""
function analytic_cell_means(flow, tracers, g::GridGeometry, t; k = 6, k_fine = 48, h = DEFAULT_PSEUDO_STEP)
    points, weights = cell_quadrature(g, k)
    dep, logρ = trace(flow, points, t; h)
    q = k * k
    cell_air = [g.cell_area[c] * sum(weights[n] * exp(logρ[n]) for n in (c-1)*q+1:c*q) for c in 1:ncell(g)]
    means = Dict{String, Vector{Float64}}()
    refined = Dict{String, Int}()
    for (name, field) in tracers
        m = reference_means(field, dep, logρ, weights, q)
        if is_discontinuous(field)
            cut = cut_cells(flow, field, g, t, sample_range(field, dep, q)...; h)
            if !isempty(cut)
                pf, wf = cell_quadrature(g, k_fine; cells = cut)
                df, lf = trace(flow, pf, t; h)
                m[cut] .= reference_means(field, df, lf, wf, k_fine^2)
            end
            refined[name] = length(cut)
        end
        means[name] = m
    end
    return means, cell_air, refined
end

# ---------------------------------------------------------------------------
# Face transfers
# ---------------------------------------------------------------------------

"""Column air mass per area [kg m⁻²] of the reference state (pure sigma, `ps` uniform)."""
column_mass_per_area(ps) = ps / STANDARD_GRAVITY

"""
    window_streamfunction(flow, corners, t0, t1; nquad = 8) -> Vector

`∫_{t0}^{t1} ψ(x, t) dt` at every corner (Gauss-Legendre in time, exact for a steady flow).
"""
function window_streamfunction(flow::StreamFunctionFlow, corners, t0, t1; nquad = 8)
    out = Vector{Float64}(undef, length(corners))
    if is_steady(flow)
        Threads.@threads for n in eachindex(corners)
            out[n] = streamfunction(flow, corners[n], t0) * (t1 - t0)
        end
        return out
    end
    ξ, w = gausslegendre(nquad)
    Threads.@threads for n in eachindex(corners)
        acc = 0.0
        for (s, ws) in zip(ξ, w)
            acc += ws * streamfunction(flow, corners[n], t0 + (s + 1) / 2 * (t1 - t0))
        end
        out[n] = acc * (t1 - t0) / 2
    end
    return out
end

"""
    column_transfers(flow, g, windows; ps) -> (F, M)

Column air-mass transfer [kg] through every face for each window `(t0, t1)` (matrix
`nface × nwindow`), and column air mass [kg] at every window boundary (`ncell × (nwindow+1)`).

Stream-function flows: `F = σ (Ψ(A) − Ψ(B))` with `Ψ` the window-integrated stream function at
the face's corners; air mass is constant. The divergent flow has its own method below.
"""
function column_transfers(flow::StreamFunctionFlow, g::GridGeometry, windows; ps)
    σ = column_mass_per_area(ps)
    F = Matrix{Float64}(undef, nface(g), length(windows))
    for (w, (t0, t1)) in enumerate(windows)
        Ψ = window_streamfunction(flow, g.corners, t0, t1)
        @inbounds for f in 1:nface(g)
            p = g.face_path[f]
            F[f, w] = -σ * (Ψ[p.b] - Ψ[p.a])
        end
    end
    M = repeat(σ .* g.cell_area, 1, length(windows) + 1)
    return F, M
end

function _divergence!(div, g, flux)
    fill!(div, 0.0)
    @inbounds for f in eachindex(flux)
        div[g.face_left[f]] += flux[f]
        div[g.face_right[f]] -= flux[f]
    end
    return div
end

"""
    window_volume_transfer!(volume, flow, g, t0, t1; ntime = 4, nspace = 4)

`∫_{t0}^{t1} ∫_face V·n dl dt` [m²] for every face (Gauss-Legendre in time and along the face).
"""
function window_volume_transfer!(volume, flow::AbstractFlow, g::GridGeometry, t0, t1; ntime = 4, nspace = 4)
    ξt, wt = gausslegendre(ntime)
    ξs, ws = gausslegendre(nspace)
    Threads.@threads for f in 1:nface(g)
        acc = 0.0
        for (s, wts) in zip(ξt, wt)
            acc += wts / 2 * path_flux(g.face_path[f], g.corners, flow, t0 + (s + 1) / 2 * (t1 - t0), ξs, ws)
        end
        volume[f] = acc * EARTH_RADIUS^2 * (t1 - t0)
    end
    return volume
end

# Measured on C24 at T/4 against 6 × 8 nodes: (3, 3) 1.0e-8, (3, 4) 2.8e-10, (4, 4) 2.8e-10.
const DEFAULT_NTIME = 3
const DEFAULT_NSPACE = 4

"""
    window_mass_transfer!(F, flow::DivergentDeformation, g, t0, t1; σ0, ntime, nspace, h)

Air-mass transfer `σ0 R² ∫∫ (ρ/ρ0) V·n dl dt` [kg] of every face over `[t0, t1]` with the exact
density (Gauss-Legendre with `ntime` nodes in time and `nspace` along the face).
"""
function window_mass_transfer!(F, flow::DivergentDeformation, g::GridGeometry, t0, t1; σ0,
                               ntime = DEFAULT_NTIME, nspace = DEFAULT_NSPACE, h = FLUX_PSEUDO_STEP)
    ξt, wt = gausslegendre(ntime)
    ξs, ws = gausslegendre(nspace)
    Threads.@threads :dynamic for f in 1:nface(g)
        acc = 0.0
        for (a, wa) in zip(ξt, wt)
            t = t0 + (a + 1) / 2 * (t1 - t0)
            for (b, wb) in zip(ξs, ws)
                x, dx = path_point(g.face_path[f], g.corners, (b + 1) / 2)
                acc += wa * wb / 4 * density_ratio(flow, x, t; h) * dot(velocity(flow, x, t), cross(dx, x))
            end
        end
        F[f] = σ0 * EARTH_RADIUS^2 * (t1 - t0) * acc
    end
    return F
end

"""
Divergent flow: transfers with the exact density, and the cell air mass by closure,
`M(t1) = M(t0) − div F` from `M(0) = σ0 A`.
"""
function column_transfers(flow::DivergentDeformation, g::GridGeometry, windows; ps,
                          ntime = DEFAULT_NTIME, nspace = DEFAULT_NSPACE, h = FLUX_PSEUDO_STEP)
    σ0 = column_mass_per_area(ps)
    nw = length(windows)
    F = Matrix{Float64}(undef, nface(g), nw)
    M = Matrix{Float64}(undef, ncell(g), nw + 1)
    M[:, 1] .= σ0 .* g.cell_area
    div = zeros(ncell(g))
    Fw = zeros(nface(g))
    for (w, (t0, t1)) in enumerate(windows)
        window_mass_transfer!(Fw, flow, g, t0, t1; σ0, ntime, nspace, h)
        F[:, w] .= Fw
        M[:, w+1] .= M[:, w] .- _divergence!(div, g, Fw)
    end
    return F, M
end

"""
    max_outflow_fraction(g, F, M) -> Float64

Largest fraction of a cell's column air mass that leaves it through its faces in one window
(sum of outgoing transfers over `min(M(t0), M(t1))`). One flux application carries `1/(2N)`.
"""
function max_outflow_fraction(g::GridGeometry, F, M)
    out = zeros(ncell(g))
    worst = 0.0
    for w in axes(F, 2)
        fill!(out, 0.0)
        @inbounds for f in 1:nface(g)
            v = F[f, w]
            v > 0 ? (out[g.face_left[f]] += v) : (out[g.face_right[f]] -= v)
        end
        @inbounds for c in 1:ncell(g)
            worst = max(worst, out[c] / min(M[c, w], M[c, w+1]))
        end
    end
    return worst
end

# ---------------------------------------------------------------------------
# Packing into the binary layouts
# ---------------------------------------------------------------------------

"""Layer fractions of the column (pure sigma: `ΔB`)."""
layer_fractions(B_ifc) = diff(B_ifc)

pack_cells(g::GridGeometry{LatLonGrid}, v, frac) =
    [v[i + (j - 1) * g.grid.mesh.Nx] * frac[k] for i in 1:g.grid.mesh.Nx, j in 1:g.grid.mesh.Ny, k in eachindex(frac)]
pack_cells(g::GridGeometry{ReducedGaussianGrid}, v, frac) = [v[c] * frac[k] for c in eachindex(v), k in eachindex(frac)]
function pack_cells(g::GridGeometry{CubedSphereGrid}, v, frac)
    Nc = g.grid.mesh.Nc
    return ntuple(p -> [v[cs_flat_index(i, j, p, Nc)] * frac[k] for i in 1:Nc, j in 1:Nc, k in eachindex(frac)], 6)
end
pack_surface(g::GridGeometry{LatLonGrid}, v) = reshape(copy(v), g.grid.mesh.Nx, g.grid.mesh.Ny)
pack_surface(g::GridGeometry{ReducedGaussianGrid}, v) = copy(v)
function pack_surface(g::GridGeometry{CubedSphereGrid}, v)
    Nc = g.grid.mesh.Nc
    return ntuple(p -> [v[cs_flat_index(i, j, p, Nc)] for i in 1:Nc, j in 1:Nc], 6)
end

"""Horizontal face arrays of one window: `(am, bm)` (lat-lon, cubed sphere) or `hflux` (RG)."""
function pack_faces(g::GridGeometry{LatLonGrid}, F, frac)
    Nx, Ny, Nz = g.grid.mesh.Nx, g.grid.mesh.Ny, length(frac)
    am = zeros(Nx + 1, Ny, Nz)
    bm = zeros(Nx, Ny + 1, Nz)
    nx_faces = Nx * Ny
    for k in 1:Nz
        for j in 1:Ny, i in 1:Nx
            am[i, j, k] = F[i + (j - 1) * Nx] * frac[k]
        end
        am[Nx+1, :, k] .= am[1, :, k]
        for j in 2:Ny, i in 1:Nx
            bm[i, j, k] = F[nx_faces + i + (j - 2) * Nx] * frac[k]
        end
    end
    return (; am, bm)
end
function pack_faces(g::GridGeometry{ReducedGaussianGrid}, F, frac; face_index = rg_face_index(g))
    hflux = zeros(nfaces(g.grid.mesh), length(frac))
    for k in eachindex(frac), (n, f) in enumerate(face_index)
        hflux[f, k] = F[n] * frac[k]
    end
    return (; hflux)
end
function pack_faces(g::GridGeometry{CubedSphereGrid}, F, frac)
    Nc, t = g.grid.mesh.Nc, g.grid.table
    Nz = length(frac)
    am = ntuple(_ -> zeros(Nc + 1, Nc, Nz), 6)
    bm = ntuple(_ -> zeros(Nc, Nc + 1, Nz), 6)
    put!(panel, dir, i, j, k, v) = dir == 1 ? (am[panel][i, j, k] = v) : (bm[panel][i, j, k] = v)
    for k in 1:Nz, f in 1:t.nf
        v = F[f] * frac[k]
        put!(t.face_panel[f], t.face_dir[f], t.face_idx_i[f], t.face_idx_j[f], k, v)
        t.mirror_panel[f] == 0 && continue
        put!(t.mirror_panel[f], t.mirror_dir[f], t.mirror_idx_i[f], t.mirror_idx_j[f], k, t.mirror_sign[f] * v)
    end
    return (; am, bm)
end

zero_cm(g::GridGeometry{LatLonGrid}, Nz) = zeros(g.grid.mesh.Nx, g.grid.mesh.Ny, Nz + 1)
zero_cm(g::GridGeometry{ReducedGaussianGrid}, Nz) = zeros(ncell(g), Nz + 1)
zero_cm(g::GridGeometry{CubedSphereGrid}, Nz) = ntuple(_ -> zeros(g.grid.mesh.Nc, g.grid.mesh.Nc, Nz + 1), 6)

_sub(a::Array, b::Array) = a .- b
_sub(a::NTuple, b::NTuple) = map(_sub, a, b)

# ---------------------------------------------------------------------------
# Transport binaries
# ---------------------------------------------------------------------------

"""
    write_binaries(g, F, M; outdir, prefix, start_date, windows_per_day, dt, steps, B_ifc, extra)

Write one transport binary per day. Window `w` stores the window-start air mass `M[:, w]`, the
per-application transfers `F[:, w] / (2 steps)` and the forward endpoint differences.
Returns the list of paths.
"""
function write_binaries(g::GridGeometry, F, M; outdir, prefix, start_date::Date, windows_per_day::Int,
                        dt::Float64, steps::Int, B_ifc, extra = Dict{String, Any}())
    nw = size(F, 2)
    nw % windows_per_day == 0 || throw(ArgumentError("window count $nw is not a multiple of $windows_per_day"))
    frac = layer_fractions(B_ifc)
    Nz = length(frac)
    vc = HybridSigmaPressure(zeros(Nz + 1), Float64.(B_ifc))
    mkpath(outdir)
    paths = String[]
    scale = 1 / (2 * steps)
    surface_pressure(w) = pack_surface(g, M[:, w] .* STANDARD_GRAVITY ./ g.cell_area)
    for day in 1:(nw ÷ windows_per_day)
        date = start_date + Day(day - 1)
        path = joinpath(outdir, "$(prefix)_$(Dates.format(date, "yyyymmdd")).bin")
        ws = (day - 1) * windows_per_day + 1:day * windows_per_day
        header = merge(Dict{String, Any}("date" => Dates.format(date, "yyyy-mm-dd")), extra)
        _write_day(g, path, ws, F, M, frac, vc, scale, steps, dt, header, surface_pressure)
        push!(paths, path)
    end
    return paths
end

function _window(g::GridGeometry, w, F, M, frac, scale, Nz, surface_pressure)
    m = pack_cells(g, M[:, w], frac)
    m_next = pack_cells(g, M[:, w+1], frac)
    faces = pack_faces(g, view(F, :, w) .* scale, frac)
    return (; m, faces, cm = zero_cm(g, Nz), ps = surface_pressure(w), dm = _sub(m_next, m))
end

function _write_day(g::GridGeometry{CubedSphereGrid}, path, ws, F, M, frac, vc, scale, steps, dt, header,
                    surface_pressure)
    Nc, Nz = g.grid.mesh.Nc, length(frac)
    writer = MD.open_streaming_cs_transport_binary(path, Nc, 6, Nz, length(ws), vc;
        FT = Float64, dt_met_seconds = dt, steps_per_window = steps,
        mass_basis = :dry, surface_pressure_basis = DrySurfacePressure(),
        include_flux_delta = true, panel_convention = "geos_native",
        planet_radius = EARTH_RADIUS, extra_header = header)
    try
        for w in ws
            win = _window(g, w, F, M, frac, scale, Nz, surface_pressure)
            MD.write_streaming_cs_window!(writer, (; m = win.m, am = win.faces.am, bm = win.faces.bm,
                                                   cm = win.cm, ps = win.ps, dm = win.dm), Nc, 6)
        end
    catch
        MD.abandon_streaming_transport_binary!(writer)
        rethrow()
    end
    MD.close_streaming_transport_binary!(writer)
    return path
end

function _structured_grid(g::GridGeometry{LatLonGrid}, vc)
    return AtmosGrid(g.grid.mesh, vc, AtmosTransport.CPU(); FT = Float64)
end
function _structured_grid(g::GridGeometry{ReducedGaussianGrid}, vc)
    return AtmosGrid(g.grid.mesh, vc, AtmosTransport.CPU(); FT = Float64)
end

function _write_day(g::GridGeometry, path, ws, F, M, frac, vc, scale, steps, dt, header, surface_pressure)
    Nz = length(frac)
    grid = _structured_grid(g, vc)
    target = MD.poisson_balance_target(:substep_mass_amount, [steps])
    header = merge(header, Dict{String, Any}("poisson_balance_target_scale" => target.scale,
                                             "poisson_balance_target_semantics" => target.semantics))
    windows = map(ws) do w
        win = _window(g, w, F, M, frac, scale, Nz, surface_pressure)
        next = w + 1 <= size(F, 2) ? w + 1 : w
        faces_next = pack_faces(g, view(F, :, next) .* scale, frac)
        _structured_window(win, faces_next)
    end
    MD.write_transport_binary(path, grid, windows; FT = Float64, dt_met_seconds = dt,
        steps_per_window = steps, source_flux_sampling = :window_start_endpoint,
        flux_sampling = :window_constant, mass_basis = :dry,
        surface_pressure_basis = DrySurfacePressure(), extra_header = header)
    return path
end

_structured_window(win, next) = haskey(win.faces, :am) ?
    (; m = win.m, am = win.faces.am, bm = win.faces.bm, cm = win.cm, ps = win.ps,
       dam = next.am .- win.faces.am, dbm = next.bm .- win.faces.bm, dcm = zero(win.cm), dm = win.dm) :
    (; m = win.m, hflux = win.faces.hflux, cm = win.cm, ps = win.ps,
       dhflux = next.hflux .- win.faces.hflux, dcm = zero(win.cm), dm = win.dm)

# ---------------------------------------------------------------------------
# Replay check of the written windows (the same per-cell test as the runtime gate)
# ---------------------------------------------------------------------------

"""Worst per-cell replay residual over all windows, computed on the packed arrays."""
function replay_residual(g::GridGeometry, F, M, B_ifc, steps)
    frac = layer_fractions(B_ifc)
    scale = 1 / (2 * steps)
    worst = (max_rel_err = 0.0, window = 0)
    rg_args = g isa GridGeometry{ReducedGaussianGrid} ? Grids.face_connectivity(g.grid.mesh) : nothing
    for w in axes(F, 2)
        m0 = pack_cells(g, M[:, w], frac)
        m1 = pack_cells(g, M[:, w+1], frac)
        faces = pack_faces(g, view(F, :, w) .* scale, frac)
        cm = zero_cm(g, length(frac))
        d = _replay(g, m0, faces, cm, m1, steps, rg_args)
        d.max_rel_err > worst.max_rel_err && (worst = (max_rel_err = d.max_rel_err, window = w))
    end
    return worst
end
_replay(::GridGeometry{CubedSphereGrid}, m0, faces, cm, m1, steps, _) =
    MD.verify_window_continuity_cs(m0, faces.am, faces.bm, cm, m1, steps)
_replay(::GridGeometry{LatLonGrid}, m0, faces, cm, m1, steps, _) =
    MD.verify_window_continuity(MD.structured_replay_layout(), MD.ReplayScratch(size(m0)), m0, cm, m1, steps,
                                faces.am, faces.bm)
_replay(::GridGeometry{ReducedGaussianGrid}, m0, faces, cm, m1, steps, (left, right)) =
    MD.verify_window_continuity_rg(m0, faces.hflux, cm, m1, left, right, steps)

# ---------------------------------------------------------------------------
# Initial-condition files and analytic references
# ---------------------------------------------------------------------------

"""
    write_initial_conditions(g, path, fields::Vector{Pair{String, Vector}}, Nz)

Write the initial cell means in a form the existing IC kinds copy without changing them:
- cubed sphere: `cs_native` file `(Xdim, Ydim, nf, lev)`, copied cell for cell;
- lat-lon: `file` kind on the model's own cell centres (bilinear weights 0 or 1);
- reduced Gaussian: `file` kind on the ring latitudes and a fine longitude axis (0.1°) holding
  each ring's cell means piecewise constant, so the bilinear sample at a cell centre returns
  that cell's mean exactly (both neighbouring nodes lie inside the cell).
"""
function write_initial_conditions(g::GridGeometry{CubedSphereGrid}, path, fields, Nz)
    Nc = g.grid.mesh.Nc
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        defDim(ds, "Xdim", Nc); defDim(ds, "Ydim", Nc); defDim(ds, "nf", 6); defDim(ds, "lev", Nz)
        lev = defVar(ds, "lev", Float64, ("lev",);
                     attrib = ["positive" => "down", "long_name" => "model level, 1 = top"])
        lev[:] = 1:Nz
        for (name, v) in fields
            a = reshape(v, Nc, Nc, 6)
            var = defVar(ds, name, Float64, ("Xdim", "Ydim", "nf", "lev"); attrib = ["units" => "mol mol-1"])
            var[:, :, :, :] = repeat(a, 1, 1, 1, Nz)
        end
        ds.attrib["description"] = "idealized advection test: initial cell means ($(g.label))"
    end
    return path
end

function write_initial_conditions(g::GridGeometry{LatLonGrid}, path, fields, Nz)
    mesh = g.grid.mesh
    Nx, Ny = mesh.Nx, mesh.Ny
    lon = mod.(Float64.(mesh.λᶜ), 360)
    order = sortperm(lon)
    _write_latlon_ic(path, lon[order], Float64.(mesh.φᶜ), Nz,
                     [name => reshape(v, Nx, Ny)[order, :] for (name, v) in fields], g.label)
end

function write_initial_conditions(g::GridGeometry{ReducedGaussianGrid}, path, fields; Nz, dlon = 0.1)
    mesh = g.grid.mesh
    nlon_fine = round(Int, 360 / dlon)
    lon = [(i - 0.5) * dlon for i in 1:nlon_fine]
    lats = Float64.(mesh.latitudes)
    out = map(fields) do (name, v)
        a = Matrix{Float64}(undef, nlon_fine, nrings(mesh))
        for j in 1:nrings(mesh)
            n = mesh.nlon_per_ring[j]
            for (i, l) in enumerate(lon)
                cell = clamp(floor(Int, l / (360 / n)) + 1, 1, n)
                a[i, j] = v[mesh.ring_offsets[j] + cell - 1]
            end
        end
        name => a
    end
    _write_latlon_ic(path, lon, lats, Nz, out, g.label)
end
write_initial_conditions(g::GridGeometry{ReducedGaussianGrid}, path, fields, Nz) =
    write_initial_conditions(g, path, fields; Nz)

function _write_latlon_ic(path, lon, lat, Nz, fields, label)
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        defDim(ds, "lon", length(lon)); defDim(ds, "lat", length(lat)); defDim(ds, "lev", Nz)
        defVar(ds, "lon", lon, ("lon",); attrib = ["units" => "degrees_east"])
        defVar(ds, "lat", lat, ("lat",); attrib = ["units" => "degrees_north"])
        defVar(ds, "lev", Float64.(1:Nz), ("lev",); attrib = ["positive" => "down"])
        for (name, a) in fields
            defVar(ds, name, repeat(a, 1, 1, Nz), ("lon", "lat", "lev"); attrib = ["units" => "mol mol-1"])
        end
        ds.attrib["description"] = "idealized advection test: initial cell means ($label)"
    end
    return path
end

"""
    write_reference(path, g, times, fields)

Analytic cell means `fields[name][:, n]` at `times[n]` [h] in the flat cell order of the model
output (lat-lon `(lon, lat)` column-major, reduced Gaussian `cell`, cubed sphere
`(Xdim, Ydim, nf)`), with the cell areas.
"""
function write_reference(path, g::GridGeometry, times, fields; attributes = Dict{String, Any}())
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        defDim(ds, "cell", ncell(g)); defDim(ds, "time", length(times))
        start = get(attributes, "start_date", "2001-01-01")
        defVar(ds, "time", Float64.(times), ("time",); attrib = ["units" => "hours since $(start) 00:00:00"])
        defVar(ds, "cell_area", g.cell_area, ("cell",); attrib = ["units" => "m2"])
        lon = [rad2deg(lon_of(x)) for x in g.cell_center]
        lat = [rad2deg(lat_of(x)) for x in g.cell_center]
        defVar(ds, "lon", lon, ("cell",); attrib = ["units" => "degrees_east"])
        defVar(ds, "lat", lat, ("cell",); attrib = ["units" => "degrees_north"])
        for (name, a) in fields
            defVar(ds, name, a, ("cell", "time"); attrib = ["long_name" => "analytic cell mean"])
        end
        for (k, v) in attributes
            ds.attrib[k] = v
        end
        ds.attrib["grid"] = g.label
    end
    return path
end

end # module
