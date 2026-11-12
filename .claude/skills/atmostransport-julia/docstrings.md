# Docstrings

The conventions are those of the Julia manual
(https://docs.julialang.org/en/v1/manual/documentation/).

## Form

1. **Signature first, indented four spaces**, so it renders as code. Simplify where it helps:
   defaults as `f(x, y=1)`, an optional argument without default as `f(x[, y])`, many keywords
   as `f(x; <keyword arguments>)` with the list under `# Arguments`. Most signatures need no
   return annotation. Where one helps, name the returned value: `f(x) -> y`. Existing
   signatures of the form `f(...) -> Type` stay as they are.
2. **One sentence in the imperative**, ending with a period: "Return the mean of ...", not
   "Returns ..." and not a bare noun phrase. Details follow in a second paragraph after a
   blank line. A function that cannot be summarized in one sentence may want splitting.
3. **Do not repeat the signature**: no "The function `f` ...", no types the signature shows.
4. **`# Arguments` only when needed** (complex functions, many keywords): one `-` bullet per
   argument with type and default.
5. **Related functions** in a "See also" paragraph. Link with ``[`name`](@ref)`` only to
   exported names of documented modules; the docs build fails on a link it cannot resolve.
   Private helpers go in plain backticks.
6. **`# Examples`** as `jldoctest` blocks with `julia>` prompts, self-contained, on exported
   names only (doctests run only for docstrings in the manual). No bare `rand`: seed your own
   generator. A plain `julia` block for examples that cannot be tested.
7. **Backticks** around code and identifiers, double backticks around equations, Unicode
   characters instead of LaTeX escapes (`σ`, not `\sigma`).
8. **Opening and closing `"""` on their own lines.**
9. **At most 92 characters per line.**
10. **`# Implementation`** for what a developer of a new subtype must provide.
11. **`# Extended help`** splits a long docstring: `?f` shows the part above it.

## Where

- Document the generic function or its most general method. Document a specific method only
  where its behaviour differs.
- The docstring sits directly above the definition, with no blank line or comment between;
  the structure check fails on a detached docstring. Struct fields may carry their own.
- Do not document an alias (`const g = f`); document `f`.
- `$` and `\` are interpolation and escape characters inside a docstring. Escape them, or use
  `@doc raw"""..."""` for text with much LaTeX.

## Example

```julia
"""
    column_mean(q, m)

Return the air-mass-weighted mean of `q` over one column.

`q` is the dry mole fraction per layer and `m` the dry air mass of the same layers; layers
without air do not contribute. The result has the element type of `q`.
"""
column_mean(q, m) = sum(k -> q[k] * m[k], eachindex(q, m)) / sum(m)
```

Not this: no signature, third person, restated types, text on the quote line.

```julia
"""The function column_mean computes the mean. Takes a Vector{Float64} q and a
Vector{Float64} m and returns a Float64."""
column_mean(q, m) = sum(q .* m) / sum(m)
```

## What a physics docstring adds

The equation with every symbol defined, the units, the reference, and what differs from the
TM5 or GCHP routine it follows. Derivations belong in the manual under `docs/src/`.
