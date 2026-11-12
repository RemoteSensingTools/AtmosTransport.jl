---
name: atmostransport-julia
description: Use when writing, reviewing or refactoring any .jl file in this repository (src/, ext/, test/, scripts/, benchmarking/, examples/) - types and dispatch, KernelAbstractions kernels, Float32 conservation, docstrings, comments, tests.
---

# Julia in AtmosTransport

`CLAUDE.md` holds the project facts, invariants and the rules for extending the code. This
skill is about how the Julia is written. The files next to this one hold the details; read
the one a task touches:

- `docstrings.md`: how a docstring is written.
- `style.md`: argument annotations, structs, indexing and `@inbounds`, comments.
- `kernels.md`: KernelAbstractions kernels, CPU and GPU.
- `float32.md`: single precision as a first-class target.
- `testing.md`: what a test must show, and the checks before a push.

## Rules

1. **Dispatch on types, not on names or flags.** A new scheme is a new type with its own
   methods; the model loop and the other operators gain no `if` on it. The wiring it does
   need (config kind, runtime spec, workspace factory, capability checks, adjoint support or
   refusal, schema) is listed in `CLAUDE.md`, Extending The Code, and the folder READMEs.
2. **Type-stable hot paths, shown by measurement.** Check the real call with
   `@code_warntype` or JET. Type annotations on arguments do not make code faster.
   Exception: `f::F ... where {F}` forces specialization on a function or type argument that
   is only passed on.
3. **Kernels allocate nothing and never promote `Float32` to `Float64`.** See `kernels.md`.
4. **Immutable structs** unless mutation is needed; array fields are type parameters, so the
   same struct holds host and device arrays.
5. **No type piracy, no test-only code in `src/`, no deprecation shims.** A replaced option
   is removed.
6. **Single precision is a first-class target.** Read `float32.md` before writing an update
   that adds small increments to large values or sums many terms.
7. **Docstrings follow the Julia manual.** See `docstrings.md`. Apply it to docstrings you
   write or touch; do not rewrite a file for it.
8. **Physics is documented in the manual first (`CLAUDE.md`) and summarized in the
   docstring:** equations with defined symbols, the reference, and the differences from TM5
   or GCHP.
9. **Comments say what is true now**, not how the code came to be. See `style.md`.
10. **`@inbounds` is earned, not added by reflex.** See `style.md`.
11. **Fail fast.** Reject bad input with a clear error before any state changes.
12. **Forward and adjoint stay mirrors.** If a folder README lists a mirror for a file,
    change both or state why not. A new operator declares its adjoint support or refusal.
13. **Dependencies change through Pkg.** Never edit a Manifest by hand; do not run
    `Pkg.update()` on the root environment casually.
14. **Performance claims come from measurements** on representative inputs, with the
    machine load checked first.

## Know, do not assume

Language knowledge from memory can be wrong or out of date. Settle a question by running it:

| Question | Check |
|---|---|
| Is this type stored inline, how large is it, where is the padding? | `isbitstype(T)`, `sizeof(T)`, `fieldoffset(T, i)` |
| Is this call type-stable? | `@code_warntype f(args...)`, or `@inferred` in a test |
| Does it allocate? | `@allocated` inside a function, after a warm-up call |
| What does this macro produce? | `@macroexpand` |
| Does a `Float32` kernel compute in `Float64`? | the checks in `kernels.md` |
| Does behaviour depend on the Julia version? | the package supports Julia 1.10, so nothing newer may be required; the full suite runs on 1.10 and 1.13 |

The authority for the language is the Julia manual: Performance Tips, Metaprogramming,
Interfaces, Documentation. For KernelAbstractions and the GPU packages, read their current
documentation instead of recalling an API.

Macros and `@generated` functions are rare here. Use one only where a function with dispatch
cannot do the job, and show its expansion in the review.

## Before you finish

- Focused tests for the touched module, then `scripts/dev/quick_gate.sh`.
- `julia --startup-file=no --project=test scripts/checks/check_structure.jl --write-code-map`
  after any change to a `.jl` file under `src/` or `ext/`, or to a `# shared:` line of a
  README; the structure test compares the code map byte for byte. A private `_helper` used
  from another module needs a `# shared:` line in the owner's README.
- `ATMOSTR_DOCS_BUILD_ONLY=true julia docs/build.jl` if `docs/` or a docstring changed.
- README or manual updated if behaviour, config keys, scripts or file maps changed.
- A change of results is stated with its size; a refactor reproduces the golden outputs
  (`test/golden/README.md`).
- An independent review of the diff for any non-trivial change (`CLAUDE.md`).
