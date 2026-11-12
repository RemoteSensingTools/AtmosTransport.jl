# Tests and checks

The tiers, topics, the quick gate, the golden outputs and the CI policy are in
`test/README.md` and `test/golden/README.md`; `CLAUDE.md` has the rules for observing state
and for refactors.

## What a test is for

- **Tests call production code paths.** No test-only code in `src/`; a test that guards
  nothing is removed.
- **An independent reference.** A test written beside the code shares its errors. Compare
  with a closed form, an independent transcription, or the original routine compiled
  unchanged (the IFS convection fixture: `test/fixtures/ifs_cuctracer_generator/`).
- **A test must be able to fail.** Ask which wrong implementation would pass: a midpoint
  instead of a mean, a transposed index, a dropped term. Use fields that differ in every cell
  and level, and NaN halos, so a misplaced read cannot pass by symmetry.
- **Honest tolerances.** State what the tolerance covers. Do not widen one or shorten a run
  to make a test pass; if a scheme misbehaves, the test documents it.
- **Properties with the result:** conservation, a uniform field, bounds, the no-op path bit
  for bit, the same column result on every grid, device against host.
- **Error tests check the message** where the message is what helps the user:
  `@test_throws "what to check" f(x)`, with or without the exception type.

## Running

```bash
julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'   # once per checkout
julia --project=test test/core/test_<name>.jl
scripts/dev/quick_gate.sh                               # before a push
julia --startup-file=no --project=test scripts/checks/check_structure.jl --write-code-map
ATMOSTR_DOCS_BUILD_ONLY=true julia docs/build.jl
```

A new `test/core/` file belongs to exactly one topic in `test/groups.toml`. The Manifests are
not tracked: a worktree whose results will be compared with a golden reference copies the
`Manifest.toml` that reference was recorded with before `Pkg.instantiate()`.

## Result changes

A change that alters results says so in the CHANGELOG, with the size of the change on a
named case.
