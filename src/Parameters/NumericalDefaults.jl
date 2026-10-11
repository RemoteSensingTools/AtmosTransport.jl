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
Courant-number limit of one advection subcycle pass at run time: a window's
mass fluxes are split into passes with outflow at most this fraction of the
cell's mass.
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
Most CFL subcycle passes one advection step may need (one direction on
structured grids, one Strang palindrome on the cubed sphere) before the run
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

"Weight of one GEOS moisture-filter / column-drift smoothing sweep (see `DEFAULT_GEOS_SMOOTHING_SWEEPS`)."
const DEFAULT_GEOS_SMOOTHING_WEIGHT = 0.5

"Order of the Lin–Rood PPM edge reconstruction when the config sets none: 5 (Huynh-constrained) or 7."
const DEFAULT_LINROOD_PPM_ORDER = 5

"Width (standard deviation) of the Gaussian-blob initial condition when the config sets none [degrees]."
const DEFAULT_GAUSSIAN_BLOB_SIGMA_DEG = 10.0

export DEFAULT_CFL_LIMIT, DEFAULT_POSITIVITY_CFL_LIMIT, DEFAULT_SUBSTEP_CFL_TARGET,
       ERA5_N320_SUBSTEP_CFL_TARGET, COARSEN_SUBSTEP_CFL_TARGET, DEFAULT_MET_INTERVAL_SECONDS,
       DEFAULT_BACKGROUND_VMR, DEFAULT_GEOS_SMOOTHING_SWEEPS, DEFAULT_MAX_SUBCYCLES,
       DEFAULT_MAX_STEPS_PER_WINDOW, ADAPTIVE_SUBSTEP_MAX_REFINEMENTS, DEFAULT_POISSON_BALANCE_TOL,
       DEFAULT_POISSON_BALANCE_MAX_ITER, DEFAULT_POISSON_PROJECT_EVERY, HORIZONTAL_BALANCE_TAG,
       DEFAULT_GEOS_SMOOTHING_WEIGHT,
       DEFAULT_GAUSSIAN_BLOB_SIGMA_DEG, DEFAULT_LINROOD_PPM_ORDER
