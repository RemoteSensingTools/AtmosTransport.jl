# Shared helpers for the inversion's placed point observations
# (test/core/test_cs_observation_binding.jl, test_cs_4dvar_placed_observations.jl):
# a cubed-sphere problem whose air mass is hydrostatic, m = (1 − q) Δp A / g per
# cell (dry basis), with smooth mass fluxes of a few percent of the cell mass.

using AtmosTransport
const _COL_AT = AtmosTransport

const COLUMN_GRAVITY = 9.80665
# Ten layers, total interface pressures (Pa, top first) of a 1000 hPa column.
const COLUMN_P_HALF = [1000.0, 10000.0, 20000.0, 30000.0, 45000.0, 60000.0, 75000.0, 85000.0,
                       92000.0, 97000.0, 100000.0]
# Specific humidity (kg/kg), top first: moist near the surface.
const COLUMN_Q = [1e-5, 1e-5, 5e-5, 2e-4, 6e-4, 1.5e-3, 3e-3, 6e-3, 1e-2, 1.4e-2]

"""
    hydrostatic_cs_problem(; FT, Nc, Hp, nsteps, flux_fraction, p_half, q)

Halo-filled panels of dry air mass `m[k] = (1 − q[k]) (p[k+1] − p[k]) A / g`
(A the cell area), tracer mass `rm = m · c(k, i, j)` with a mixing ratio that
rises toward the surface, and per-step face mass fluxes of about
`flux_fraction` of the cell mass (zero for `flux_fraction = 0`). `qpanels` are
the humidity panels in the air-mass layout.
"""
function hydrostatic_cs_problem(; FT = Float64, Nc = 4, Hp = 3, nsteps = 3, flux_fraction = 0.03,
                                p_half = COLUMN_P_HALF, q = COLUMN_Q)
    mesh = _COL_AT.CubedSphereMesh(Nc = Nc, Hp = Hp, FT = FT)
    N = Nc + 2Hp
    Nz = length(q)
    area(i, j) = _COL_AT.cell_area(mesh, clamp(i - Hp, 1, Nc), clamp(j - Hp, 1, Nc))
    m = ntuple(p -> [FT((1 - q[k]) * (p_half[k + 1] - p_half[k]) * area(i, j) / COLUMN_GRAVITY)
                     for i in 1:N, j in 1:N, k in 1:Nz], 6)
    rm = ntuple(p -> [m[p][i, j, k] * FT(400e-6 + 2e-6 * k + 0.5e-6 * sin(0.7i + 0.3j + p))
                      for i in 1:N, j in 1:N, k in 1:Nz], 6)
    _COL_AT.Operators.Advection.fill_panel_halos!(m, mesh; dir = 0)
    _COL_AT.Operators.Advection.fill_panel_halos!(rm, mesh; dir = 0)
    scale(k) = FT(flux_fraction * (1 - q[k]) * (p_half[k + 1] - p_half[k]) * area(Hp + 1, Hp + 1) /
                  COLUMN_GRAVITY)
    am = [ntuple(p -> begin
              a = zeros(FT, N + 1, N, Nz)
              for k in 1:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc+1
                  a[i, j, k] = scale(k) * FT(sin(0.2s + 0.3p + 0.7i + 0.4j + 0.2k))
              end
              a
          end, 6) for s in 1:nsteps]
    bm = [ntuple(p -> begin
              b = zeros(FT, N, N + 1, Nz)
              for k in 1:Nz, j in Hp+1:Hp+Nc+1, i in Hp+1:Hp+Nc
                  b[i, j, k] = scale(k) * FT(0.8cos(0.3s + 0.5p + 0.4i + 0.6j + 0.1k))
              end
              b
          end, 6) for s in 1:nsteps]
    cm = [ntuple(p -> begin
              c = zeros(FT, N, N, Nz + 1)
              for k in 2:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc
                  c[i, j, k] = -FT(0.5) * scale(k) * FT(1 + 0.3sin(i + j + k + p + s))
              end
              c
          end, 6) for s in 1:nsteps]
    qpanels = ntuple(_ -> repeat(reshape(FT.(q), 1, 1, Nz), N, N, 1), 6)
    return (; mesh, m, rm, am, bm, cm, qpanels, FT, Nc, Nz, nsteps, p_top = p_half[1])
end

column_met(P; kwargs...) = _COL_AT.CSObservationColumnMet(; p_top = P.p_top, gravity = COLUMN_GRAVITY,
                                                          humidity = P.qpanels, kwargs...)

# The halo-adjoint and Lin-Rood face adjoint kernels accumulate with atomics,
# so a multithreaded reverse pass reproduces itself only to rounding (two
# identical calls differ at ~1e-22 of the largest gradient entry). Gradient
# equalities are therefore bit for bit single-threaded and to 1e-13 relative
# otherwise. `a`, `b`: vectors of per-control panel tuples (`CS4DVarResult.gradients`).
function gradients_match(a, b)
    Threads.nthreads() == 1 && return all(a[c][q] == b[c][q] for c in eachindex(b), q in 1:6)
    scale = maximum(maximum(abs.(b[c][q])) for c in eachindex(b), q in 1:6)
    return maximum(maximum(abs.(a[c][q] .- b[c][q])) for c in eachindex(b), q in 1:6) <= 1e-13 * scale
end
