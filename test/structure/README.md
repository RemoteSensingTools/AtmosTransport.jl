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
  `folder_order_allow`; these exceptions and allowed edges only shrink.
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

X1 migrates P1's historical identities in the comparison, without accepting
new debt: the old Analysis conservation files and their module-assembly
private imports map to Diagnostics, other non-core paths map into Extras,
and SectionTimer folder edges map to `src`. Diagnostics inherits the former
Analysis dependencies on earlier core modules; Analysis gains only its
Diagnostics re-export edge. The old facade-to-timer folder exception now
covers the facade's Diagnostics include. No reverse exception is added.
Clone/docstring digests, method signatures, private helper names and byte
ceilings survive this migration unchanged, and allowance counts cannot grow.
Bases already using the new identities receive the ordinary exact-key
comparison. Entries eliminated by declared sharing or other cleanup are
removed. Duplicate, docstring, folder-size and mirror snapshots currently
have no moved file keys; their identities and counts remain unchanged.

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

M1 recognizes parent-folder child wiring by the same ancestry rule used for
module wiring. The dependency-budget migration replaces the old single
MetDrivers binary unit with BinaryFormat and BinaryReaders, inheriting only
its earlier dependencies and parent interfaces. The ERA5 geometry facade
binding follows its moved Preprocessing source. This migration applies only
to LAYERS edges; private uses, duplicates, clones and byte ceilings only shrink.

C1 splits Convection into the facade and TM5/CMFMC children. The LAYERS-only
baseline migration inherits the old lower dependencies and maps existing
consumers to the children. CMFMC may call the earlier TM5 solve; no new reverse
edge, clone digest, private-use allowance or folder budget is accepted.

PP1 splits the preprocessing dependency budget into Core, Vertical and Spectral.
The LAYERS-only migration grants each child the former lower dependencies;
Vertical may call Core and Spectral may call Core and Vertical. Existing facade
consumers may follow the moved bindings. No child may import the parent or a
later stage, and no reverse exception or clone/private/docstring allowance is
added. The migration does not apply when the base already contains Core.
