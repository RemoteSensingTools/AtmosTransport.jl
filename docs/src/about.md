# About these docs

This site is the AtmosTransport.jl reference site. It is built with
[Documenter.jl](https://documenter.juliadocs.org/) and
[DocumenterVitepress.jl](https://luxdl.github.io/DocumenterVitepress.jl/),
which produces a VitePress static site with native Mermaid rendering.

## What you'll find here

- **Getting Started** — install AtmosTransport.jl, run the synthetic
  smoke tests, and inspect the output of a real run.
- **For TM5 & GCHP users** — five pages mapping TM5 / GCHP / GIGC
  vocabulary onto AtmosTransport's design: philosophy, the binary
  pipeline, operator dispatch, adjoints, and the kernel + I/O
  architecture.
- **Concepts** — grids, state and basis, operators, binary format.
- **Tutorials** — end-to-end runnable examples (CI-safe, executed at
  build time).
- **Preprocessing** — turning raw met data (ERA5 spectral GRIB, GEOS
  native NetCDF) into a transport binary.
- **Theory & Verification** — discrete mass conservation, advection
  schemes, conservation budgets, validation status, adjoint status.
- **Configuration & Runtime** — TOML schema, NetCDF output schema,
  data sources.
- **Development** — the performance benchmarks, the lab machines that run
  them, and how to read the performance dashboard.
- **API Reference** — auto-generated docstrings per submodule.

## Source of truth

When this site and the source code disagree, **the source code wins**.
Open an issue if you find a contradiction and we'll patch the doc.
Concrete invariants and file-level pointers live in the repository `README.md`,
the numbered engineering-contract documents under `docs/`, and source-side
module docstrings.

## Build the docs locally

```bash
ATMOSTR_DOCS_BUILD_ONLY=true julia docs/build.jl
```

`docs/build.jl` activates and instantiates the isolated documentation
environment, develops the checkout in place, executes the tutorial and
doctests, and renders the production VitePress site. The build fails on broken
references, doctests, or exported docstrings omitted from the manual.

The rendered VitePress site lands in `docs/build/`. To serve it
locally for live preview:

```bash
cd docs && julia --project -e 'using DocumenterVitepress; DocumenterVitepress.dev_docs("build")'
```

CI builds the same target on pull requests and pushes to `main`; deployment to
`gh-pages` happens on pushes to `main`.

## How to contribute documentation

- **Concepts and theory pages.** Prose-heavy. Cite the relevant source file for
  any concrete claim; if a claim depends
  on a specific function, link the function in the API reference
  rather than copying its signature inline (signatures drift; the
  autodoc does not).
- **Tutorials.** Add a new `.jl` under `docs/literate/`; the build
  picks it up automatically and executes it. Keep tutorials
  CI-friendly (no external downloads, no GPU-only paths).
- **API reference.** Edit the source-side docstring rather than the
  generated `api/*.md`; the autodoc surfaces whatever is in the
  source.
- **Status tracker.** The top-level [`README.md`](https://github.com/RemoteSensingTools/AtmosTransport.jl)
  carries the canonical "what works now, what's experimental" table.
  Update that table when a capability moves between status tiers.
