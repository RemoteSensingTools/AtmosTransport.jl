# Structure ratchet and migrations

`scripts/checks/check_structure.jl` parses the include tree with JuliaSyntax
and never loads the package. Run it from the checkout root against the branch
your change will merge into:

```bash
CUDA_VISIBLE_DEVICES= julia --startup-file=no --project=test scripts/checks/check_structure.jl --base=origin/main
```

`src/LAYERS.toml` records the allowed module and folder dependency edges, the
folder order and reasoned folder-order exceptions. The files in
`test/structure/` snapshot existing debt: duplicate helpers, body clones,
private uses, detached docstrings, core-to-non-core uses and oversized
folders. Every snapshot may only shrink relative to the base. Editing a
snapshot in the same diff cannot grant a new edge or allowance. The order
checks use the actual include order and `folder_order` independently of the
ratchet. A folder-order exception is kept only for a reverse edge that the
order rules, including module child wiring and shared-module wiring, would
otherwise reject.

## Moves, renames and splits

Snapshots are keyed by names: module names, folder paths and source file
paths. When a change moves code to a new name, it declares each transfer as a
reviewed row in `test/structure/migrations.toml`:

```toml
[[migration]]
from = "src/Preprocessing"
to = ["src/Preprocessing", "src/Preprocessing/Core", "src/Preprocessing/Vertical"]
reason = "Split preprocessing into ordered stage folders"
```

`from` must be a module or folder declared in the base `src/LAYERS.toml`, or
a Julia file of the base. Every `to` entry must be a current identity of the
same kind. In the base snapshots, the checker substitutes the `to` identities
for `from`:

- Each target may use the base dependencies of `from`, plus its own base
  dependencies if it already existed. The targets of one row may also use
  each other: they were one unit before.
- Base edges to `from` may now point to any target, so existing users can
  follow the moved code.
- `from` keeps its budget and users only if it is listed in `to`. A vacated
  name that is later reused starts with no budget; an `Empty -> Rich` rename
  gives the new `Rich` only the old budget of `Empty`.
- In allowance ids, only module, folder and file fields follow a row. A file
  field also follows its folder's row. Helper names, signatures and digests
  never change.
- Each base allowance and folder-order exception moves to at most one place.
  It is never copied, so these entries cannot grow in number. If a retained
  `from` and one of its pieces both need an exception, the second one is a
  new allowance. Module child wiring already exempts facade edges, so main
  keeps no exceptions for them that a split could carry to a new folder.
- The byte ceilings drawn from one base ceiling may not exceed it in sum. Each
  ceiling must still equal its folder's current size.

Rows are not chained: a target of one row never receives the budget that
another row grants to its `from`.

### Direction among the pieces

Peer grants are symmetric, so the ordinary order checks decide the direction:

- **Modules.** A reference to a later module fails. A child module may use its
  parent, because the parent precedes it in include order. This is the facade
  usage that `LAYERS.toml` already contains for many modules.
- **Cycles.** Parent and child wiring is not counted as a module cycle.
- **Folders.** Direction follows the PR's `folder_order`. That order is
  declared, not ratcheted; review it together with the rows.

## Row validation

The checker rejects:

- rows without a reason or without distinct `to` identities;
- a `from` that is not a base identity, or a target of the wrong kind;
- an unused row target, whose removal exposes no allowance growth;
- a target that another row also covers. This is reported as redundant with
  the other row; keep one.

A row identical to one in the base `migrations.toml` (same `from` and `to`)
was applied there and is **inert**. When `migrations.toml` equals the base
copy, as on a clean `main`, the checker only warns that such rows can be
deleted. A pull request that edits `migrations.toml` must delete them.

Pull requests compare with their target branch. CI compares a push to `main`
with the commit `main` had before the push (`github.event.before`). It falls
back to `HEAD^1` when that commit is all zeros or cannot be fetched. A merge, including a multi-commit rebase merge, therefore validates
its rows once more. Afterwards they are inert. No Julia code is specific to a
past cleanup.

## Examples

These rows reproduce the merged folder cleanups against each merge's first
parent. Those trees still carry the facade exceptions that module child wiring
makes unneeded; the rows pass once those exceptions are deleted, as on `main`
today.

| Cleanup | Rows |
|---|---|
| X1 #108 | `AtmosTransport.Analysis -> [Analysis, Diagnostics]`, `src/Analysis -> [src/Extras/Analysis, src/Diagnostics]`, `src/Diagnostics -> [src]` (the timer moves to `src`), `src/Analysis/Analysis.jl -> [src/Diagnostics/Diagnostics.jl]`, one row each moving Visualization, Downloads and `Downloads/sources` into `src/Extras` |
| O1 #110 | `AtmosTransport.Output -> [Output, Observations, .Retrievals, .Sources, .Sampling]`, `src/Output/observations -> [Retrievals, Sources, Sampling]`, `src/Output -> [src/Output, src/Observations]` |
| M1 #111 | `AtmosTransport.MetDrivers -> [MetDrivers, .BinaryFormat, .BinaryReaders]`, `src/MetDrivers/transport_binary -> [BinaryFormat, BinaryReaders]`, `src/MetDrivers/ERA5 -> [src/Preprocessing/sources]` |
| C1 #114 | `AtmosTransport.Operators.Convection` and `src/Operators/Convection`, each to itself plus `TM5` and `CMFMC` |
| A1 #115 | `src/Operators/Advection -> [Advection, Reconstruction, Structured, CubedSphere]` |
| T1 #116 | `AtmosTransport.Adjoints -> [Adjoints, .Footprint, .Inversion, .Inversion.Observations, .Inversion.Solver]`, `src/Footprint -> [src/Adjoints/Footprint]`, `src/Inversion -> [src/Adjoints/Inversion, .../Solver]`, `src/Adjoints -> [src/Adjoints, src/Adjoints/Inversion/Observations]` |
| PP1 #117 | `AtmosTransport.Preprocessing` and `src/Preprocessing`, each to itself plus `Core`, `Vertical` and `Spectral` |
| PP2 #118 | `AtmosTransport.Preprocessing` and `src/Preprocessing`, each to itself plus `MassClosure` and `PhysicsInputs` |
| MD1 #119 | `src/Models -> [src/Models, Physics, Core, Runner]`, `src/Models/runner -> [src/Models/Runner]`, `src/Models/initial_conditions -> [src/Models/InitialConditions]` |

## Shared-module folder wiring

Folder edges must point to earlier folders unless `folder_order_allow`
records an exception. Two wiring cases are exempt:

- A module entry file may include or re-export child modules in its
  subfolders.
- `src/Operators/Advection` may wire its `Reconstruction`, `Structured` and
  `CubedSphere` folders while they share one module.

Models uses the ordinary child-module wiring rule. MD2 removes MD1's shared-module
exception: its siblings follow `Physics → Core → InitialConditionIO → Runner`,
with driven stepping, forcing refresh and capture owned by Runner.

Sibling order and all external dependencies still follow the ordinary rules.
