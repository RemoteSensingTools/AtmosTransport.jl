# Kernels

Kernels are KernelAbstractions `@kernel` functions that run unchanged on the host and on
CUDA, ROCm and Metal devices. How the model launches them is described in
`docs/src/concepts/architecture.md`; the helpers are in `src/Architectures.jl`. The loop
order, the source-and-destination rule for sweeps and the backend dispatch rule are in
`CLAUDE.md`.

For KernelAbstractions, CUDA.jl, AMDGPU.jl and Metal.jl, read the current package
documentation instead of recalling an API, and measure before claiming that a change is
faster. Launch latency, synchronization and memory traffic often decide, not arithmetic.

## Rules

- **No allocation in a kernel.** Scratch lives in a workspace sized like the field, built
  once. A kernel body builds no message from values and prints nothing.
- **No `Float64` in a `Float32` kernel.** A bare literal such as `0.5` is `Float64` and
  promotes the expression. Write `FT(0.5)` or `oftype(x, 0.5)`, or combine an integer
  directly with the value (`x / 2`, `2x`). These are `Float64` too: `1/2`, `2π`, `sqrt(2)`,
  `Inf`, `NaN`, `eps()`, `max(x, 0.0)`; use `FT(Inf)`, `eps(FT)`, `zero(FT)`. Metal has no
  `Float64` at all.
- **One work item writes only its own cells.** Scratch is not shared between work items;
  neighbouring work items should touch neighbouring memory.
- **Launch through `launch!`** (`launch!(kernel, backend, workgroup, ndrange, args...; sync)`),
  so workgroup and synchronization are explicit at the call site. Several launches can share
  one synchronization with `sync = false`; do not remove a synchronization without showing
  that no host read depends on it. The cubed-sphere advection sweeps keep their own profiled
  launch.
- **Point operations** (`AbstractPointOp` with `apply_point!`, combined with `Fused` and
  `Sequence`) give one kernel body for many small operations; the backend decides how a
  fused set is launched.
- **A six-panel tuple inside a kernel** is indexed with `select_panel(t, p)` when the panel
  number is a run-time value; `p` must be in 1:6 and is not checked there.
- **Host checks happen outside the kernel**: a kernel returns a status value per work item;
  the host reduces it and throws before any state changes.
- **A column kernel shares one body across grids.** The lat-lon, reduced-Gaussian and
  cubed-sphere wrappers differ only in how a column is addressed (see
  `src/Operators/Convection/IFS/ifs_kernels.jl`).

## Checks

- The same test on the host and on a device, in `Float32` and `Float64`, with the device
  result compared against the host. GPU sections run only where the device is functional;
  CI has no GPU, so run them on a GPU machine.
- No `Float64`: `@test (@inferred column_fn(args32...)) isa Float32` for each function a
  kernel calls, as `test/core/test_fv3_vertical_profile.jl` does. For a whole body, the typed
  or LLVM code with `Float32` arguments shows the element types (`@code_warntype`,
  `@code_llvm`; on a device, the package's device-code reflection); `double` in an error
  path alone is not a promotion.
- `@allocated` inside a function, after a warm-up call: zero for a column function or host
  loop; for a kernel launch on the host, a bound far below one field, as
  `test/core/test_replay_consistency.jl` does.
- For a performance claim: the timing on the target device with the machine otherwise idle,
  before and after, with the kernel and its synchronization both inside the measurement.
