# Structure snapshots

The test-only checker in `scripts/checks/check_structure.jl` parses the
`src/AtmosTransport.jl` include tree and `ext/*.jl` with JuliaSyntax. It never
loads the package. Root test extras and `test/Project.toml` use matching
exact compat pins for JuliaSyntax 0.4.10 and 1.0.2: JET requires 0.4 on Julia
1.10 and 1.x on newer Julia. The checker normalizes their short-method AST
representations and uses a fixed grammar so snapshots agree. Run it from the
root with:

```bash
CUDA_VISIBLE_DEVICES= julia --startup-file=no --project=test scripts/checks/check_structure.jl
```

`test/core/test_structure.jl` exercises the checker and introduces deliberate
violations in temporary repositories. The quick set and `quick_gate.sh` both
run it. The standalone check accepts `--root=<checkout>` and `--base=<git-ref>`
(default `origin/main`), plus `--write-code-map` to regenerate only the
developer map. The quick gate passes its selected base ref. A missing or
unreadable base fails; only an absent snapshot path at a valid commit counts
as first introduction. Synthetic fixtures explicitly opt out of base checking.

- T1: `src/LAYERS.toml` records allowed internal module edges. Imports,
  imported/re-exported bindings, and qualified references are resolved through
  the include tree. New edges, later-module references, cycles (excluding
  parent-to-child wiring), unparsed includes, and unreached Julia files fail.
  `order` records the snapshot's include order; later-module checks use the
  actual include order, even if the TOML is edited.
  `folders` and `folder_order` additionally declare allowed folder edges and
  their one-directional order (dependencies must point to earlier folders).
  Folder edges come from imports/references, includes, and uses of known
  definitions across folders that currently share a module. Module imports
  point to the module's entry folder; named bindings also point to definition
  folders. Existing reverse edges in this unaligned layout have reasons in
  `folder_order_allow`; these exceptions and allowed edges only shrink. An
  exception is kept only for a reverse edge that the order rules would
  otherwise reject; module child wiring and shared-module wiring need none.
  Folder-order enforcement also rejects new cycles: a cycle needs a reverse
  edge. Existing exceptions can preserve current cycles until folder cleanup.
  `noncore_allowlist.toml` snapshots current core-to-non-core dependencies.
  Core must not import/refer to Visualization, Analysis, DataDownloads, the
  Makie extension, or any module defined under `src/Extras/`, including through
  facade aliases. `AtmosTransport.Diagnostics` is core and follows Models;
  foundation `SectionTimer` lives in `src/SectionTimer.jl` so its early users
  do not form a folder cycle with the later conservation module.
  Today's three exceptions are package-facade re-exports, retained until the
  planned Extras split; implementation modules have no exceptions.
- T2: `duplicate_allowlist.toml` records each existing independent generic
  name and its owning modules. Explicitly imported and qualified extensions,
  callable objects, and constructors are excluded from the duplicate-name
  check. A new owner of an existing duplicate still fails.
- T3: `clone_allowlist.toml` records each site of token-identical function
  bodies with at least 25 tokens, across different files. Signatures and
  whitespace/comments are excluded from body comparison. A stable SHA-256
  body digest and signature digest identify each method without line numbers.
  Jaccard similarity of normalized token sets (at least 100 body tokens,
  score at least 0.9) is advisory; the report prints the count and first ten
  pairs. Exact clone debt and new violations are checked without truncation.
  Callable-object methods participate in T3 even though they own no T2 generic.
- T4: `private_use_allowlist.toml` records existing cross-module private uses
  by file, consumer, owner, and name. `public`/`export` in the owner or an
  explicit `# shared: _helper, _other!` line in the owner's folder README
  authorizes sharing. Reasons retain the baseline locations for review;
  moving a line does not create a new violation.
- T5: every source folder (including `src` and `ext`) must stay within 65,000
  approximate tokens (immediate Julia source bytes / 3.5). Subfolders have
  separate budgets; files are never charged twice. `folder_size_allowlist.toml`
  records the seven existing oversized folders with exact byte ceilings and
  reasons. A ceiling cannot grow relative to the base; when source shrinks,
  reduce its ceiling to the exact current size; remove it once in budget or
  when the folder is removed. Retaining a stale size allowance fails.
  New files are automatically counted in their folders. The inventory's 30
  review units have no role in enforcement.
- T6: missing README headings `Depends on`, `Used by`,
  `Shared helpers owned here`, and `Forward<->adjoint mirrors` are reported
  only. Dependency-section contents are not checked in P1. Add the sections
  folder by folder with the relevant cleanup.

`docstring_allowlist.toml` additionally snapshots standalone triple-quoted
strings at module scope. A blank line or comment separating a docstring from
its definition can silently detach it; a new detached string fails.

All allowances snapshot P1's source state and contain reasons. Remove entries
as cleanup eliminates debt; stale entries are reported. The checker compares
allowance keys and dependency edges against `--base` and rejects additions
where that base already contains the snapshot. P1's base has no snapshots, so
this first introduction is reviewed in the PR. There is no allowance
regeneration or accept-all flag. New allowances require an explicit owner decision.

## Migrations

Snapshots are keyed by module names, folder paths and file paths. A PR that
moves, renames or splits a unit declares each budget transfer in
`migrations.toml` as a `[[migration]]` row with `from`, `to` and `reason`.
`from` must be a module or folder of the base `src/LAYERS.toml`, or a Julia
file of the base. Every target must be a current identity of the same kind.
In the base snapshots the checker substitutes the targets for `from`:

- Targets receive the base dependencies of `from` and may use each other. The
  order checks still reject references to later modules or folders. A child
  module may use its parent facade, which precedes it in include order.
- Base edges to `from` may point to any target.
- `from` keeps its budget only if it is also a target, so a reused vacated
  name inherits nothing.
- Only module, folder and file fields of allowance ids follow rows. Each base
  allowance and folder-order exception moves to at most one place, so the
  number of entries never grows.
- Byte ceilings drawn from one base ceiling may not exceed it in sum.
- Rows are not chained.

The checker rejects rows without a reason, invalid identities, and unused or
redundant row targets.

A row identical to one in the base `migrations.toml` was applied there and is
inert. On an unchanged file, as on a clean `main`, it only produces a warning.
A PR that edits `migrations.toml` must delete such rows.

CI compares a push to `main` with the commit before the push, so the merge
validates its rows. `docs/src/development/structure.md` gives the details and
the rows of past cleanups. The checker contains no code specific to a past
cleanup.

`docs/src/development/code_map.md` is generated from the same inventory and
checked byte for byte by the gate and repository structure test. Each folder
lists its immediate files, bytes/tokens, dependency folders, folders using it,
and distinct public/export name count, resolved to definition folders
(falling back to the module entry folder when no definition is located).
Facade re-exports are counted at their owner, without double-counting.
`mirrors.toml` records forward/adjoint file pairs seeded from cleanup
`units.md` U09–U14 and `helper_index.md` §0.3; both folders show each pair.
Maintain those pairs when files move. Regenerate with:

```bash
CUDA_VISIBLE_DEVICES= julia --startup-file=no --project=test scripts/checks/check_structure.jl --write-code-map
```

This writes only the code map and then runs the checks, so it cannot approve
new violations. Read the code map and the folder README before editing.

This is a static syntax check, not Julia name inference: local shadowing of an
imported binding can produce a conservative finding, while computed property
access and dynamic code generation can hide a dependency. Macro argument
syntax is scanned without expansion. It checks source and extension structure,
including non-core modules, without adding numerical tests of non-core APIs.

Folder edges point to earlier folders unless `folder_order_allow` records an
exception. A module entry file may wire child modules in its subfolders, as
for module edges. One shared-module layout also wires its folders:
`src/Operators/Advection` with its Reconstruction, Structured and CubedSphere
folders (A1: fenced root types and Lin-Rood share one module). The Models
folders are ordinary submodules (MD2 removed MD1's shared-Models exemption).
Edges between the Advection sibling folders and all external dependencies keep
the ordinary order. Explicit
relative imports resolve local child modules before inherited aliases.
`Adjoints.Inversion.ObservationBinding` owns the inversion observation flow;
the package's `Observations` owns forward placement, retrievals and sampling.
