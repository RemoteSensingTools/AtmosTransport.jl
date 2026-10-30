# Runtime Stability, Subcycling, and Precompile Notes

## Scope

This memo records the first real-data `src` ERA5 lat-lon results and the
conclusions they force on the next implementation steps.

It is intentionally narrow:

- pure advection only
- `src` only
- ERA5 lat-lon reference path first
- `UpwindScheme` as the first runtime truth baseline
- `SlopesScheme` validated against the same binaries after the upwind path was
  stable

## Current Real-Data Status

The current standalone `src` ERA5 lat-lon reference path now works for the
first full 2-day real-data smoke test on `2021-12-01` to `2021-12-02`.

Verified results:

- `UpwindScheme` completes both days on the rebuilt reference binaries
- `SlopesScheme` completes the same 2-day run on the same binaries
- both runs stay finite for all 48 windows
- the day-boundary air-mass mismatch before day 2 is `9.153e-07`
- the first-day worst hourly endpoint mismatch is `7.671e-06` local and
  `9.492e-06` global across all 23 handoffs
- tracer-mass drift for a uniform tracer is `0` for upwind and `1.219e-16` for
  slopes

So the original multi-day mass-loss bug on the real `0.5°` lat-lon reference
path is fixed for the current forcing contract.

## Why The Final Air-Mass Number Is Not The Main Metric

The runner reports a final air-mass change relative to the initial state. That
number is not a pure conservation metric here because the driver resets air mass
from the met window at each new window boundary.

The important diagnostics for multi-day correctness are instead:

- does the runtime state at the end of window `n` land near the stored air-mass
  state at window `n+1`?
- do uniform-tracer runs preserve tracer mass on the same forced air-mass path?

Those are now the key acceptance criteria for the lat-lon runtime.

## What The Current Flux Contract Means

The new transport-binary runtime contract currently says:

- `flux_kind = :substep_mass_amount`
- flux interpretation is done upstream in the driver/runtime layer
- advection kernels only consume prepared flux fields

That architectural split is still correct.

The current ERA5 lat-lon reference binaries reuse the legacy spectral pipeline
convention where `am`, `bm`, and `cm` are scaled to `half_dt`, but after
preprocessing they should be treated as **window-constant substep transport
amounts**, not as endpoint states to interpolate toward the next hour.

Relevant source:

- spectral mass-flux construction uses `half_dt` in
  [`spectral_synthesis.jl`](../src/Preprocessing/Spectral/spectral_synthesis.jl)
- the runtime then applies these prepared sweep fluxes through the Strang
  sequence in
  [`StrangSplitting.jl`](../src/Operators/Advection/Structured/StrangSplitting.jl)

So the right reading of the current ERA5 lat-lon contract is:

- the binary stores prepared Strang-sweep transport amounts for one substep
- the runtime should not multiply fluxes by `dt`
- the current reference path should keep those fluxes constant within the hour
- endpoint interpolation inside the current hour was the remaining bug, not the
  intended contract

## Structured Subcycling Status

The structured `src` path now includes a first CPU-side evolving-mass pilot
for x, y, and z directional subcycling.

That was enough to remove the original full-day instability on the real
`0.5°` lat-lon binary, but it is still only a first stage:

- it is global per direction, not row-adaptive
- it is CPU-only; GPU arrays currently fall back to a single pass
- it improves stability, but it was not the final endpoint-fidelity fix

So the current state is:

- stable enough for the current 2-day lat-lon reference run
- still not the final word on long-run efficiency or TM5-style latitude-adaptive
  policies

## Where TM5 Still Matters

TM5 does not rely only on one global subdivision factor.

In the newer MPI transport:

- x transport uses evolving-mass local `nloop(j,l)` in
  `deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advectx__slopes.F90`
- y transport uses evolving-mass local `nloop(l)` in
  `deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advecty__slopes.F90`

Those comparison sources live in the optional, git-ignored `deps/` checkout;
they are not part of the AtmosTransport source distribution.

Important implications:

- x-direction stiffness on lat-lon is naturally latitude dependent
- a row- or row/level-adaptive x policy is closer to TM5 and more efficient than
  one global x subdivision based on the polar maximum
- y-direction refinement can often stay simpler than x

But `src` should still stay generic. That argues for:

1. a clean subcycling policy layer
2. scheme-independent CFL handling
3. topology-aware implementations behind a stable API

## Root Cause Found

The multi-day endpoint-fidelity problem turned out to be two coupled forcing
bugs, not an advection-kernel bug.

### Bug 1: Poisson-balance target normalization

What was wrong:

- the Poisson-balance target used the raw forward window mass difference
  `m_next - m_curr`
- but the stored horizontal fluxes are half-sweep transport amounts consumed
  over `2 * steps_per_window` horizontal half-sweeps per met window
- so the correct balance target is:
  - `(m_next - m_curr) / (2 * steps_per_window)`

This fix alone reduced hourly mismatch by about an order of magnitude, but it
was not sufficient.

### Bug 2: wrong intra-window flux interpretation

Even after Bug 1, the runtime was still interpolating `am`, `bm`, and `cm`
toward the next hour inside the current met window.

That was wrong for the current reference binaries. The balanced flux fields are
only consistent when interpreted as **window-constant** substep transport
amounts.

Key evidence:

- old interpolated runtime on rebuilt binaries still showed hourly mismatch of
  `O(1e-3)`
- constant-flux replay of the same window dropped those mismatches to
  `O(1e-7)` to `O(1e-6)`
- the same fix made the day-boundary mismatch collapse from `5.3e-2` to
  `9.153e-07`

## Current Reference Conclusion

For the ERA5 lat-lon `src` reference path:

- `source_flux_sampling = :window_start_endpoint`
- `flux_kind = :substep_mass_amount`
- `flux_sampling = :window_constant`
- `poisson_balance_target_scale = 1 / (2 * steps_per_window)`

The driver must keep fluxes constant within each met window for this path.
Advection kernels should not diagnose closure or reinterpret timing semantics.

## PrecompileTools: Useful, But Not For This Bug

`PrecompileTools.jl` can help with:

- first-run latency of `src` smoke tests
- JIT time for `TransportModel`, `DrivenSimulation`, and scheme dispatch
- repeated local development loops

It does **not** solve:

- forcing-contract mistakes
- large x-direction CFL by itself
- slow GRIB or NetCDF I/O
- expensive spectral transforms

So it is still worth adding later for ergonomics, but it is not the next
numerical priority.

## Preprocessing Speed Notes

The first real `2021-12-01` and `2021-12-02` lat-lon preprocesses showed that
the biggest costs are not binary packing:

- GRIB spectral reads are expensive
- repeated QV reads are also expensive

Representative timing for one day was about:

- total: `~303-312 s`
- spectral GRIB read: `~126-131 s`
- spectral transforms: `~76-80 s`
- QV loads: `~43-44 s`
- write: `~18 s`

So the first preprocessing optimizations should be:

1. keep the daily thermo/QV file open across the day
2. keep payload packing threaded and deterministic
3. parallelize across days before trying fine-grained variable-level process
   fan-out

The natural parallel units are:

- first: day-level parallelism across processes
- second: small worker-pool hourly-window parallelism

Not recommended as a first move:

- distributing by spectral variable (`VO`, `D`, `LNSP`) because they are tightly
  coupled in one transform path
