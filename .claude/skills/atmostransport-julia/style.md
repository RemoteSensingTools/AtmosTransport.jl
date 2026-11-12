# Style: arguments, structs, indexing, comments

Several of these rules are adapted from Tim Holy's public agent configuration
(https://github.com/timholy/claude_config, MIT licence).

## Arguments and constructors

- **Annotate an argument only as far as the method needs.** `f(A::Matrix{Float64})` shuts out
  device arrays, views and `Float32`. Use `AbstractArray{FT, 3}` where the shape matters and
  nothing where the method works for any input. Annotations select methods; they do not
  document intent.
- **`AbstractArray` is a promise** that the code works for any such array. Loops here assume
  1-based arrays with explicit halo offsets. Where a host function relies on that for an
  argument that could be something else, say so with `Base.require_one_based_indexing(A)`
  (on the host, not in a kernel).
- **New inner constructors take unconstrained arguments** and let the fields convert; outer
  constructors only compute type parameters.
- **Keyword shorthand:** `f(; max_iter)` when the variable has the keyword's name.
- **Fail fast.** An unexpected state is an error with a message that says what to check, not
  a silent default.

## Structs

- Immutable by default. A struct needs `mutable` only if a field is rebound after
  construction or its identity matters (`===`, finalizers).
- In a `mutable struct`, mark every field that is never rebound `const`.
- Check the layout of a struct that is stored in arrays: `isbitstype`, `sizeof`,
  `fieldoffset`. A struct passed to a kernel is checked as the kernel receives it, after
  adaptation to the device.

## Indexing and `@inbounds`

- Iterate `eachindex(A)` or `axes(A, d)` where the loop follows an array; write explicit
  ranges where the algorithm is about levels and interfaces (`k in 1:Nz`), with the sizes
  checked at the entry point.
- `@inbounds` is not a speed switch: an out-of-range access under it reads or writes other
  memory instead of throwing. Outside kernels, do not add it without a measurement that
  shows the bounds check costs something.
- Kernels and column routines here use `@inbounds`, because their indices come from the
  launch range and from sizes validated before the launch. For new code of that kind the
  validation must exist, and an index computed from data (a cloud top, a level found by a
  search) needs a guard.
- `Pkg.test()` and CI run the host tests with bounds checks forced on, which ignores every
  `@inbounds` and turns an out-of-range access into a `BoundsError`. The quick gate and a
  direct `julia --project=test test/core/test_<name>.jl` do not; add `--check-bounds=yes`
  to run a new kernel's host test that way once before review. It recompiles everything.

## Comments

- A comment states what is true of the code now: an invariant, a non-obvious reason, a
  constraint a later editor would otherwise break. It must read correctly to someone who has
  only the repository.
- No history ("previously", "used to", "was changed to"), no plans, no task or session
  labels, no "for now". If the history is the only thing that prevents a mistake, write the
  constraint as a present-tense fact: "must stay above zero: zero makes the solve singular".
  The one exception is the `# HACK:` marker with its TODO that `CLAUDE.md` requires for an
  unavoidable shortcut.
- An issue or PR number is a fine pointer (`# see #NNN`) but does not replace saying what
  the code does.
- Match the density of the surrounding code. Do not narrate what the next line does.
- Docstrings follow the same rules. Commit messages and the CHANGELOG carry the history, but
  no task or session labels.
