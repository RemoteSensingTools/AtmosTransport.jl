# Plain-loop reference of the tracer transport of the IFS convection scheme (OpenIFS CY48R1.1
# `cuctracer.F90` and `cubidiag.F90`, implicit setting RMFSOLCT = 1, no scavenging), written for
# layer fields: entrainment and detrainment of the updraft and the downdraft per layer
# (kg m-2 s-1), layer air mass per area `m` (kg m-2), mixing ratio `c`. Layer 1 is the top.
# Interface `i` is the top of layer `i`; interface `N + 1` is the ground.
#
# The mass fluxes are the cumulative sums of entrainment minus detrainment. The updraft is closed at
# the top of its highest detraining layer and the downdraft at the bottom of its lowest detraining
# layer: sums of archived fields do not return to zero exactly, and the remainder must not travel on
# to the model top or the ground.
#
# `literal = true` keeps the Fortran arithmetic where the detrainment of a layer exceeds the
# mass flux entering it; `literal = false` lets the excess leave as entrained air of the same layer.
#
# `monotone = true` is a variant that cannot create negative values or new extrema as long as
# `ifs_courant(...) <= 1`: the downdraft starts with air of the layer above its first interface
# only (IFS: one tenth updraft air), the compensating flow takes the value of the layer it comes
# from (IFS: always the layer above), and the correction of the lowest layer is not needed.
#
# `ifs_courant` is the number the sub-stepping of the operator limits.
function ifs_courant(entu, detu, entd, detd, m, dt)
    Mu, Dn = ifs_interface_fluxes(entu, detu, entd, detd)
    return maximum(dt * (max(Mu[k], Mu[k + 1]) + max(Dn[k], Dn[k + 1])) / m[k] for k in eachindex(m))
end

function ifs_interface_fluxes(entu, detu, entd, detd)
    N = length(entu)
    Mu = zeros(N + 1); Dn = zeros(N + 1)
    top = findfirst(>(0), detu)             # the updraft ends in the highest layer that detrains
    if top !== nothing
        for k in N:-1:top+1
            Mu[k] = max(Mu[k + 1] + entu[k] - detu[k], 0.0)
        end
    end
    bottom = findlast(>(0), detd)           # the downdraft ends in the lowest layer that detrains
    if bottom !== nothing
        for k in 1:min(bottom, N)-1
            Dn[k + 1] = max(Dn[k] + entd[k] - detd[k], 0.0)
        end
    end
    return Mu, Dn                           # nothing crosses the model top or the ground
end

function ifs_convection_step(c, entu, detu, entd, detd, m, dt; literal = false, monotone = false)
    N = length(c)
    Mu, Dn = ifs_interface_fluxes(entu, detu, entd, detd)
    Cu = zeros(N + 1); Cd = zeros(N + 1)
    for i in 2:N
        Cu[i] = c[i - 1]; Cd[i] = c[i - 1]
    end
    Cu[N] = c[N]
    for k in N-1:-1:2                       # updraft, upwards
        Mu[k] > 0 || continue
        through = literal ? Mu[k + 1] - detu[k] : Mu[k + 1] - min(detu[k], Mu[k + 1])
        entrained = literal ? entu[k] : Mu[k] - through
        Cu[k] = (through * Cu[k + 1] + entrained * c[k]) / Mu[k]
    end
    lfs = findfirst(>(0), Dn)               # interface where the downdraft starts
    if lfs !== nothing
        Cd[lfs] = monotone ? c[lfs - 1] : 0.1 * Cu[lfs] + 0.9 * c[lfs - 1]
        for i in lfs+1:N                    # downdraft, downwards, through layer i - 1
            Dn[i] > 0 || continue
            through = literal ? Dn[i - 1] - detd[i - 1] : Dn[i - 1] - min(detd[i - 1], Dn[i - 1])
            entrained = literal ? entd[i - 1] : Dn[i] - through
            Cd[i] = (through * Cd[i - 1] + entrained * c[i - 1]) / Dn[i]
        end
        if Dn[N] > 0 && !monotone           # keep the lowest layer from going negative
            change = dt / m[N] * (-Mu[N] * Cu[N] + Dn[N] * Cd[N] + (Mu[N] - Dn[N]) * c[N - 1])
            if c[N] + change < 0
                Cd[N] = (Mu[N] * Cu[N] - (Mu[N] - Dn[N]) * c[N - 1] - c[N] * m[N] / dt) / Dn[N]
            end
        end
    end
    plume = zeros(N + 1); net = zeros(N + 1)
    for i in 2:N
        plume[i] = Mu[i] * Cu[i] - Dn[i] * Cd[i]
        net[i] = Mu[i] - Dn[i]
    end
    cnew = similar(c)
    if monotone                             # implicit compensating flow, upstream by its direction
        lower = zeros(N); diag = ones(N); upper = zeros(N); rhs = zeros(N)
        for k in 1:N
            z = dt / m[k]
            rhs[k] = c[k] + z * (plume[k + 1] - plume[k])
            lower[k] = -z * max(net[k], 0.0)
            diag[k] = 1 - z * min(net[k], 0.0) + z * max(net[k + 1], 0.0)
            upper[k] = z * min(net[k + 1], 0.0)
        end
        for k in 2:N                        # Thomas algorithm
            w = lower[k] / diag[k - 1]
            diag[k] -= w * upper[k - 1]
            rhs[k] -= w * rhs[k - 1]
        end
        cnew[N] = rhs[N] / diag[N]
        for k in N-1:-1:1
            cnew[k] = (rhs[k] - upper[k] * cnew[k + 1]) / diag[k]
        end
        return cnew
    end
    for k in 1:N                            # implicit compensating subsidence, top down
        z = dt / m[k]
        rhs = c[k] + z * (plume[k + 1] - plume[k])
        above = k == 1 ? 0.0 : cnew[k - 1]
        cnew[k] = (rhs + z * net[k] * above) / (1 + z * net[k + 1])
    end
    return cnew
end
