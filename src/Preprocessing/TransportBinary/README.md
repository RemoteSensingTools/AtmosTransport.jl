# TransportBinary

Binary metadata, write-time evidence, window protocols, writer adapters and the unified day lifecycle.

## File map

- [`TransportBinary.jl`](TransportBinary.jl) — module imports, exports and load order.
- [`core.jl`](core.jl) — binary v4 metadata, provenance, sizing and output reuse.
- [`driver.jl`](driver.jl) — unified day lifecycle and dispatch hooks.
- [`flux_fidelity.jl`](flux_fidelity.jl) — typed evidence, deterministic sampling and certificates.
- [`positivity_status.jl`](positivity_status.jl) — shared accumulator and summary diagnostics.
- [`structured_positivity.jl`](structured_positivity.jl) — shared LL/CS rectangle scan.
- [`topology_dispatch.jl`](topology_dispatch.jl) — unsupported-pair fallback and process_day generic.
- [`window_contracts.jl`](window_contracts.jl) — window/writer protocols, step policy and event/cache types.
- [`writer_adapters.jl`](writer_adapters.jl) — typed writer adapters, publication and staged-size validation.

## Depends on

Core target/source protocols; Vertical extensive sums; Spectral pressure/remap helpers; MassClosure flux geometry and closure diagnostics. Lower package APIs provide binary v4 IO, replay types, geometry, constants, mass basis and atomic publication. It imports no target, source or parent implementation.

## Used by

All three targets and the deferred source day drivers; the Preprocessing facade preserves script, Models and test names.

## Shared helpers owned here

`ReadyWindow`, `PreverifiedWindow`, `SubstepSchedulePolicy`, the window/writer interfaces and driver hooks; fidelity samplers/capture and evidence builders; `verify_substep_positivity_cs!` (the shared rectangular-block scan), positivity accumulation/summaries, `validate_replay_tol`, `validate_cfl_limit`, `validate_staged_size`, `splitmix64`; binary sizing, provenance and header helpers; `cached_or_build!` and
`ranked_fidelity_ids` (each caller retains its key, seed and salt).

## Forward<->adjoint mirrors

No adjoint. Certificate writing mirrors MetDrivers/BinaryFormat header parsing; writer adapters preserve its v4 payload layout. LL/CS share the palindrome scan; RG keeps its separate per-direction scan.

## Numerical contract

Binary format remains v4. Shared helpers preserve conversion boundaries, native-layer order, sample salts, tolerances and the summary escape hatch before integer rescue arithmetic. Arithmetic variants stay separate.
