# =============================================================================
# Numerical defaults
# =============================================================================
#
# The one default of each numerical setting that several code paths offer (a
# keyword argument, a config key, a command-line option). Code that offers the
# setting takes its default from here, so the paths cannot drift apart. Settings
# that belong to one scheme stay in that scheme's parameter struct.
# =============================================================================

"""
Runtime palindrome outflow budget of split-sweep advection (lat-lon,
reduced-Gaussian and cubed-sphere): a transport step is split into `n`
subcycles so that one palindrome's worth of outflow,
`2 (out_x + out_y + out_z) / n`, is at most this fraction of
`min(m_start, m_end)`, the smaller of the cell's masses at the start and end of
the step. Every pass then removes at most half this fraction of the mass the
cell holds when the pass starts.
"""
const DEFAULT_CFL_LIMIT = 0.95

"""
Positivity limit checked when a transport binary is written: the palindrome
outflow of a substep, `2 (out_x + out_y + out_z)`, must stay below this fraction
of `min(m, m_next)`.
"""
const DEFAULT_POSITIVITY_CFL_LIMIT = 0.95

"Courant number that the adaptive write-time substep schedule aims for."
const DEFAULT_SUBSTEP_CFL_TARGET = 0.95

"Adaptive substep target of the ERA5 N320 → cubed-sphere preprocessor."
const ERA5_N320_SUBSTEP_CFL_TARGET = 0.85

"Adaptive substep target of the cubed-sphere binary coarsener."
const COARSEN_SUBSTEP_CFL_TARGET = 0.90

"Length of a met window when the source or config does not give one [s]."
const DEFAULT_MET_INTERVAL_SECONDS = 3600.0

"Dry volume mixing ratio of tracers whose initial condition sets none (400 ppm)."
const DEFAULT_BACKGROUND_VMR = 4.0e-4

"""
Smoothing sweeps of the GEOS preprocessing moisture filter and column-drift
correction.
"""
const DEFAULT_GEOS_SMOOTHING_SWEEPS = 8

"""
Most CFL subcycles one split-sweep advection step may need before the run
stops: a guard against mass fluxes that would need absurdly many passes.
"""
const DEFAULT_MAX_SUBCYCLES = 4096

"Most write-time substeps per window the cubed-sphere binary coarsener accepts."
const DEFAULT_MAX_STEPS_PER_WINDOW = 4096

"""
Most times the adaptive write-time substep schedule raises a window's substep
count (to the count the measured Courant ratio implies) before accepting it; one
or two normally suffice.
"""
const ADAPTIVE_SUBSTEP_MAX_REFINEMENTS = 8

"Residual tolerance of the horizontal Poisson flux balance (conjugate-gradient solve)."
const DEFAULT_POISSON_BALANCE_TOL = 1.0e-14

"Iteration limit of the horizontal Poisson flux balance."
const DEFAULT_POISSON_BALANCE_MAX_ITER = 20000

"""
Interval, in conjugate-gradient iterations, at which the cubed-sphere Poisson
solve projects out the null-space (global-mean) component.
"""
const DEFAULT_POISSON_PROJECT_EVERY = 50

"""
Value of the transport-binary header key `horizontal_balance` written by every
preprocessing path that Poisson-balances its horizontal mass fluxes: `"column"`.

Every path that applies a Poisson balance closes a window's horizontal fluxes
against its endpoint mass tendency with one solve for the column-integrated
divergence and spreads each face's correction over the layers, so the layer
winds keep their vertical structure and the vertical mass flux diagnosed from
the explicit tendency carries the vertical motion. GEOS `pressure_fixer` and
`pfix_corrected` keep their native horizontal fluxes and record `"none"`. A
per-layer balance would leave almost nothing for the vertical flux because
preprocessing has no independent vertical-flux input (issue #11).
"""
const HORIZONTAL_BALANCE_TAG = "column"

"""
Smallest increment `ΔB_k = B_{k+1} − B_k` of hybrid `B` at which a layer counts
as hybrid in the column-balance weighting `HybridMassWeightedColumn`: a layer
with `ΔB_k < 1e-6` is treated as pure-pressure and receives no part of a column
correction. `ΔB_k` is the share of a surface-pressure change that changes the
layer's thickness, `Δ(Δp_k) = ΔB_k Δp_s`, so at the threshold a 1000 hPa
(1e5 Pa) change of `p_s` changes the thickness by 0.1 Pa. GEOS/MERRA-2 L72
layer 41 (150–177 hPa) has `ΔB = 8.2e-9` (8.2e-4 Pa per 1000 hPa, 3.1e-7 of
its 2654 Pa thickness at `p_s` = 1000 hPa), a rounding residue of its `B`
table. The smallest genuine hybrid increment of the shipped tables is ERA5
L137 layer 55 (73–77 hPa, the topmost layer with `ΔB > 0`): `ΔB = 7.0e-6`,
0.7 Pa per 1000 hPa, 0.17 % of its 416 Pa thickness. The ERA5 level presets
keep it at `ΔB = 7.0e-6` (`ml137_full`, `ml137_cfl94`, `ml137_cfl85`) or merge
it into a layer with `2.4e-5` (`ml137_66L`) or `5.9e-5` (`ml137_tropo34`,
`ml137_tropo34a`); the next GEOS layer has `6.96e-3` (696 Pa per 1000 hPa).
"""
const HYBRID_LAYER_MIN_DB = 1.0e-6

"""
Absolute floor of the per-cell closure tolerance of the cubed-sphere column
balance [kg per flux application]:
`τ_c = max(COLUMN_CLOSURE_TOL_KG, 8 ε_FT Σ_{f∂c} Σ_k |F_{f,k}|)` bounds the
mean-free column residual of every cell after the balance
(`balance_cs_column_mass_fluxes!`, `recheck_cs_column_closure`) and, plus the
rounding bound of the recursion, the `cm` residual folded into the column
(`diagnose_cs_cm!`). The floor binds for Float64 payloads (ERA5 C24 golden:
largest projected residual 0.10 kg against `τ_c = 10` kg); Float32 payloads
are bound by the rounding term (smallest `τ_c` of window 1 of the MERRA-2 C90
golden: 6.1e5 kg).
"""
const COLUMN_CLOSURE_TOL_KG = 10.0

"""
Write-time OMEGA-fidelity gate: the normalized RMS difference between the
vertical velocity diagnosed from a binary's `cm` and the source OMEGA,
`NRMSE = RMS(ω_cm − ω_src) / RMS(ω_src)`, must be below 1.0. `NRMSE = 1` is
the score of `cm ≡ 0`, so the limit accepts only a vertical transport that is
closer to the source than none at all. A sign-flipped `cm` scores about 2, a
level-flipped one about 1.4; MERRA-2 wind-derived C90 scores 0.47 (2.1×
margin; 2022-01 golden).
"""
const OMEGA_FIDELITY_NRMSE_LIMIT = 1.0

"""
Write-time OMEGA-fidelity gate: the correlation between `ω_cm` and the source
OMEGA must exceed 0.5 (MERRA-2 wind-derived C90: 0.886). A zero `cm` has
correlation 0, a sign-flipped one −1.
"""
const OMEGA_FIDELITY_CORRELATION_MIN = 0.5

"""
Write-time OMEGA-fidelity gate: the regression slope of `ω_cm` on the source
OMEGA must exceed 0.5 (and stay below `OMEGA_FIDELITY_SLOPE_MAX`). An amount
or time-convention error (`2n` against `n` substeps, the 8× `mass_flux_dt`
error, a factor `g`) scales `cm` uniformly: it scales the slope by the same
factor but leaves the correlation unchanged, and NRMSE stays below 1 for
factors down to 1/20, so only a bound on the slope catches it. MERRA-2
wind-derived C90 measures 0.865 (2022-01 golden); `0.5 = 0.865/1.73` and
`OMEGA_FIDELITY_SLOPE_MAX = 1.5 = 0.865 × 1.73` are symmetric about it in
log space, so a factor-2 error (slope 0.43 or 1.73) fails. Zero and
sign-flipped `cm` fail too.
"""
const OMEGA_FIDELITY_SLOPE_MIN = 0.5

"""
Write-time OMEGA-fidelity gate: the regression slope of `ω_cm` on the source
OMEGA must stay below 1.5 (see `OMEGA_FIDELITY_SLOPE_MIN`: a `cm` twice too
large has slope 1.73 at MERRA-2 C90).
"""
const OMEGA_FIDELITY_SLOPE_MAX = 1.5

"""
Replay-check tolerance for Float64 transport-binary data. The check replays each
window's air-mass budget cell by cell,
`r = m_cur − 2n (Σ F_out − Σ F_in) − m_next` over the cell's faces, and accepts
the window when every cell has
`|r| ≤ τ (|m_cur| + |m_next| + 2n Σ |F|)`: the residual relative to the summed
magnitudes of the budget's own terms, the scale that the rounding error of the
sum is bounded by. Stored Float64 binaries (2026-10 goldens) measure at most
6.5e-14, set by the residual of the Poisson flux balance.
"""
const REPLAY_TOLERANCE_FLOAT64 = 1.0e-12

"""
Replay-check tolerance for Float32 transport-binary data, in the per-cell metric
of `REPLAY_TOLERANCE_FLOAT64`. Stored Float32 binaries measure at most 1.0e-6
(about 9 eps(Float32): Float32 rounding) in C90 and C180 cubed-sphere binaries
from ERA5, MERRA-2 and GEOS-IT alike (2026-10 goldens and production samples).
"""
const REPLAY_TOLERANCE_FLOAT32 = 1.0e-5

"Weight of one GEOS moisture-filter / column-drift smoothing sweep (see `DEFAULT_GEOS_SMOOTHING_SWEEPS`)."
const DEFAULT_GEOS_SMOOTHING_WEIGHT = 0.5

"Lin–Rood PPM edge-value family when the config sets none: 5 (Colella–Woodward fourth-order edges) or 7."
const DEFAULT_LINROOD_PPM_ORDER = 5

"Width (standard deviation) of the Gaussian-blob initial condition when the config sets none [degrees]."
const DEFAULT_GAUSSIAN_BLOB_SIGMA_DEG = 10.0

export DEFAULT_CFL_LIMIT, DEFAULT_POSITIVITY_CFL_LIMIT, DEFAULT_SUBSTEP_CFL_TARGET,
       ERA5_N320_SUBSTEP_CFL_TARGET, COARSEN_SUBSTEP_CFL_TARGET, DEFAULT_MET_INTERVAL_SECONDS,
       DEFAULT_BACKGROUND_VMR, DEFAULT_GEOS_SMOOTHING_SWEEPS, DEFAULT_MAX_SUBCYCLES,
       DEFAULT_MAX_STEPS_PER_WINDOW, ADAPTIVE_SUBSTEP_MAX_REFINEMENTS, DEFAULT_POISSON_BALANCE_TOL,
       DEFAULT_POISSON_BALANCE_MAX_ITER, DEFAULT_POISSON_PROJECT_EVERY, HORIZONTAL_BALANCE_TAG,
       HYBRID_LAYER_MIN_DB, COLUMN_CLOSURE_TOL_KG, OMEGA_FIDELITY_NRMSE_LIMIT, OMEGA_FIDELITY_CORRELATION_MIN,
       OMEGA_FIDELITY_SLOPE_MIN, OMEGA_FIDELITY_SLOPE_MAX,
       DEFAULT_GEOS_SMOOTHING_WEIGHT,
       DEFAULT_GAUSSIAN_BLOB_SIGMA_DEG, DEFAULT_LINROOD_PPM_ORDER,
       REPLAY_TOLERANCE_FLOAT64, REPLAY_TOLERANCE_FLOAT32

"Near-surface fallback temperature (K) for observation hypsometric layer heights."
const DEFAULT_OBSERVATION_LAYER_TEMPERATURE_KELVIN = 280.0
export DEFAULT_OBSERVATION_LAYER_TEMPERATURE_KELVIN
