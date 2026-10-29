# AtmosTransport documentation

The maintained manual lives in [`src/`](src/) and is published at the
[AtmosTransport.jl documentation site](https://RemoteSensingTools.github.io/AtmosTransport.jl/dev/).
The same pages serve new users, experienced atmospheric-model developers, and
API readers; there is no second, competing reference manual.

## Start here

1. [Installation](src/getting_started/installation.md)
2. [Julia orientation](src/getting_started/julia_basics.md) if Julia is new to you
3. [Quickstart](src/getting_started/quickstart.md)
4. [Architecture tour](src/concepts/architecture.md)
5. [Run with real meteorology](src/getting_started/first_run.md)
6. [Learn adjoints and inversions](src/getting_started/adjoints.md)

The quickstart is deterministic, uses the current version-4 binary contract,
and downloads no external data.

## Documentation structure

```text
docs/
├── src/             rendered user manual and API reference
├── literate/        executable tutorial sources
├── memos/           dated design and investigation records
├── validation/      dated validation records
├── make.jl          Documenter navigation and build definition
└── build.jl         reproducible local/CI build entry point
```

The numbered top-level documents (`00_…` through `40_…`) are engineering
contracts for contributors. Dated bug reports and memos preserve decision
history; they are not statements of the current user API.

## Build locally

From the repository root:

```bash
ATMOSTR_DOCS_BUILD_ONLY=true julia docs/build.jl
```

The build uses the isolated docs environment, executes the Literate tutorial
and doctests, checks exported docstrings and cross-references, and renders the
VitePress site beneath `docs/build/`.

To preview the rendered site:

```bash
cd docs
julia --project -e 'using DocumenterVitepress; DocumenterVitepress.dev_docs("build")'
```

## Contribute

- Edit prose pages under `docs/src/`.
- Edit API documentation in source docstrings; `docs/src/api/` selects which
  symbols Documenter renders.
- Add runnable tutorials under `docs/literate/`. They must be CPU-safe and
  independent of external data.
- Read the [generated code map](src/development/code_map.md) and the folder
  README before editing source. Regenerate the map after source/mirror changes
  with `julia --startup-file=no --project=test scripts/checks/check_structure.jl --write-code-map`;
  the structure test checks that the checked-in map matches the source.
- Run the strict docs build before opening a pull request.

If behavior and prose disagree, treat the code and tests as evidence and fix
the manual in the same change.
