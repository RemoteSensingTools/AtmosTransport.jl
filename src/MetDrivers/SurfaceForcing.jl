# ---------------------------------------------------------------------------
# PBLSurfaceForcing - raw per-window surface fields used to derive PBL Kz.
#
# The transport binary stores the raw meteorological fields, not Kz. Runtime
# diffusion derives a panel-native Kz cache from these fields and the current
# dry air mass whenever the met window advances.
# ---------------------------------------------------------------------------

"""
    PBLSurfaceForcing(pblh, ustar, hflux, t2m[, eflux])

Container for raw surface fields used by the PBL diffusion closure.

Fields are topology-shaped 2D arrays:

- structured: `(Nx, Ny)`
- face-indexed: `(ncell,)` when such a path is added
- cubed sphere: `NTuple{6, <:AbstractMatrix}` with one `(Nc, Nc)` panel

Units follow the canonical runtime contract:

- `pblh`  - boundary-layer height [m]
- `ustar` - friction velocity [m s^-1]
- `hflux` - upward sensible heat flux [W m^-2]
- `t2m`   - 2 m air temperature [K]
- `eflux` - upward latent heat flux [W m^-2], or `nothing` (only GEOS-Chem's
  non-local VDIFF scheme needs it)
"""
struct PBLSurfaceForcing{P, U, H, T, E}
    pblh  :: P
    ustar :: U
    hflux :: H
    t2m   :: T
    eflux :: E
end

PBLSurfaceForcing(pblh, ustar, hflux, t2m) = PBLSurfaceForcing(pblh, ustar, hflux, t2m, nothing)
PBLSurfaceForcing(; pblh, ustar, hflux, t2m, eflux = nothing) =
    PBLSurfaceForcing(pblh, ustar, hflux, t2m, eflux)

has_pbl_surface_forcing(f::PBLSurfaceForcing) = true
has_pbl_surface_forcing(::Nothing) = false

function Adapt.adapt_structure(_to, f::PBLSurfaceForcing)
    return PBLSurfaceForcing(Adapt.adapt(_to, f.pblh),
                             Adapt.adapt(_to, f.ustar),
                             Adapt.adapt(_to, f.hflux),
                             Adapt.adapt(_to, f.t2m),
                             Adapt.adapt(_to, f.eflux))
end

export PBLSurfaceForcing, has_pbl_surface_forcing
