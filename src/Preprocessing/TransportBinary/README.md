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

Common target/source protocols; Vertical extensive sums; Spectral pressure/remap helpers; MassClosure flux geometry and closure diagnostics. Lower package APIs provide binary v4 IO, replay types, geometry, constants, mass basis and atomic publication. It imports no target, source or parent implementation.

## Used by

All three targets and the deferred source day drivers; the Preprocessing facade preserves script, Models and test names.

## Shared helpers owned here

`ReadyWindow`, `PreverifiedWindow`, `SubstepSchedulePolicy`, the window/writer interfaces and driver hooks; fidelity samplers/capture and evidence builders; `verify_substep_positivity_cs!` (the shared rectangular-block scan), positivity accumulation/summaries, `validate_replay_tol`, `validate_cfl_limit`, `validate_staged_size`, `splitmix64`; binary sizing, provenance and header helpers; `cached_or_build!` and
`ranked_fidelity_ids` (each caller retains its key, seed and salt).

## Forward<->adjoint mirrors

No adjoint. Certificate writing mirrors MetDrivers/BinaryFormat header parsing; writer adapters preserve its v4 payload layout. LL/CS share the palindrome scan; RG keeps its separate per-direction scan.

## Numerical contract

Default binaries remain v4; the TM5 construction records its v5 identity. Shared helpers preserve conversion boundaries, native-layer order, sample salts, tolerances and the summary escape hatch before integer rescue arithmetic. Arithmetic variants stay separate.

# shared: tm5_construction_metadata, tm5_flux_evidence

`tm5_spectral_metadata.jl` records the v5 spectral construction identity and
honest unavailable flux-fidelity evidence where no independent spectral
re-evaluator is attached. Legacy outputs stay v4. The new LL/RG construction
uses the existing window-constant and explicit-dm runtime contracts.

The LL writer adapter accepts a `storage_index` payload slot independent of the
logical window index, allowing two-window streaming with the existing deltas.
TM5 v5 provenance records the actual synthesis backend and float precision.

Opt-in LL live-Calc_Kzz inputs add a complete `tm5_pbl` group and a v5
construction identity: point-sampled by default, or TM5 spectral when selected.
`writer_adapters.jl` appends its arrays after the unchanged v4 sections. Default writers still emit v4. RG uses BinaryFormat's
existing optional-section stream packing.
