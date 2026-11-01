# Adjoints and checkpointing API

The adjoint surface covers cubed-sphere objectives, observations, footprints,
preconditioning, and 4D-Var solvers. `Tape` owns checkpoint schedules and
device, pinned-host, and memory-mapped tape storage.
`Adjoints.Inversion.ObservationBinding` owns inversion records, column binding,
observation/departure IO and OCO-2 MIP conversion. `Observations` owns forward
placement, retrievals and sampling; inversion exports remain available through
the `Inversion` and `Adjoints` facades.

## Adjoints

Configure optimizer settings and logging on `CSGradientDescent` or `CSLBFGS`
and pass it as `optimizer` to `cs_surface_flux_4dvar_optimize`. One
`cs_surface_flux_4dvar` evaluation runs the forward model once: the run that
samples the observations records the tape of the reverse pass;
`evaluation = CSCostOnly()` runs it without a tape for the cost and simulated
values alone. Footprint
functions compute analytic adjoints; finite-difference perturbations belong
to the caller's validation code. `CSFootprintResult` contains `objective`,
`footprints`, `lag_steps`, `dt`, and `initial_adjoint`.

```@autodocs
Modules = [AtmosTransport.Adjoints, AtmosTransport.Adjoints.Footprint,
           AtmosTransport.Adjoints.Inversion.ObservationBinding, AtmosTransport.Adjoints.Inversion.Solver]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

The solver exposes the forward preconditioner and its adjoint. The unused
inverse and tangent interfaces have been removed.

## Tape and checkpointing

Every reverse-pass entry point defaults to `tape = AutoTape()`. It resolves
one concrete `AdjointTape` after building the replay and reserves its pools
before recording. The binary's per-window step schedule fixes a closed-form
size; CFL-driven schedules and Lin–Rood use one counting replay, cached for
all evaluations of an optimization. Auto reads the backend's free memory
after reclaiming its pool, subtracts reverse-workspace and output memory and
`max(10% of capacity, 4 GiB)` on GPUs, and checks host availability against
cgroup headroom, excluding reclaimable file cache. CPU selection holds back 10%
of the memory limit; pinned-host selection alongside a GPU holds back
`max(10% of limit, 8 GiB)`. These margins apply only to Auto selection; explicit
`AdjointTape` choices attempt reservation directly, with a hard headroom check
for pinned memory.

Supplied `AutoTape` fields constrain the choice: `storage`, `checkpoint`,
`content`, `budget_bytes` (overrides the measured device tape/snapshot budget
and margin, including shared host headroom on CPU)
and `scratch_dir` (enables mmap). The order is device Full, device Stride
with the largest fitting resident tail and device snapshots, device Stride
with pinned-host snapshots, then mmap candidates only when `scratch_dir` is set. Pinned tapes are available explicitly; no pin-once GPU
measurement has qualified them ahead of recomputation. Relative-time estimates use
measured device forward/record/walk costs; unmeasured host and Lin–Rood
relative times are logged as `missing`.

An explicit `tape = AdjointTape(; content, storage, checkpoint)` fixes what each record stages and how long a recording serves
(`StateTape()`, recorded by every forward run, or `AirMassTape()`, recorded
once per call or optimization for the schemes whose reverse reads the air mass
only, `tape_dependence`), where the staged states live
(`DeviceCSTapeStorage()`, `PinnedHostCSTapeStorage()`,
`MmapCSTapeStorage(dir)`) and how much of the run is taped at once
(`FullCheckpoint()`, `StrideCheckpoint(K; resident_tail = 0,
snapshots = DeviceCSTapeStorage())`). A resident tail uses the tape’s selected storage and is recorded in the
forward run; only the head's `K`-step windows are recomputed. Snapshots may
use `PinnedHostCSTapeStorage()` for their tracer and air masses; persistent
operator state retains its backend types. The value holds no memory: each
call opens its tape's memory, reuses it for every stride window (and
`cs_surface_flux_4dvar_optimize` for every evaluation) and frees it when it
returns. The on-disk storage `MmapCSTapeStorage(dir)` takes a required
scratch directory; each call reserves its tape there before recording.

A failed Auto reservation tries another allowed candidate. A failed explicit
choice raises `TapeReservationError` with its estimate; it is never replaced.
Device slabs, pinned chunks and mmap extents are reserved before recording;
growth beyond a pool reservation raises `TapeReservationExceeded`. Lin–Rood's
horizontal copies remain outside the pools and are counted in its device
peak. Each runtime emits one `@info "adjoint tape"` with its choice, estimated
and reserved bytes, budget and expected relative time. Mmap is scratch
storage: persisted per-window tapes and bisection Revolve are absent.

A split-sweep tape stages each state once: the palindrome center's diffusion
records and a convection call share the slots of the sweeps that read the same
air mass, and a monotone or CW84 PPM sweep stages one byte of limiter branches
per cell instead of the tracer mass (see
[what the tape stages](@ref adjoint-tape-compaction)).

```@autodocs
Modules = [AtmosTransport.Tape]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
