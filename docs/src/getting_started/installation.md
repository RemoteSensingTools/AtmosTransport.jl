# [Installation](@id Installation)

This guide installs AtmosTransport from source and verifies that Julia can load
it. The first simulation on the next page uses the CPU and synthetic data, so
you do not need a GPU, meteorological credentials, Python, or NetCDF command-line
tools.

## 1. Install Julia and Git

AtmosTransport requires Julia 1.10 or later. Install Julia with
[juliaup](https://github.com/JuliaLang/juliaup), the installer recommended by
the Julia project, then open a new terminal and check the version:

```bash
julia --version
```

Any result beginning with `julia version 1.10` or newer is suitable. You also
need [Git](https://git-scm.com/downloads):

```bash
git --version
```

!!! tip "New to terminals or Julia?"
    Read [Julia orientation](@ref Julia-orientation) for the difference between
    terminal commands, the `julia>` prompt, and Julia's `pkg>` prompt. You can
    return here after the two-minute overview.

## 2. Clone the repository

AtmosTransport is not installed from Julia's General registry yet. Clone the
source and enter the repository directory:

```bash
git clone https://github.com/RemoteSensingTools/AtmosTransport.jl.git
cd AtmosTransport.jl
```

The default clone follows `main`. For a reproducible experiment, check out a
published release tag with `git checkout <tag>` and record it with your run
configuration. The `/dev/` manual follows development; consult the
repository's `CHANGELOG.md` when moving between releases. Do not assume that
an unreleased branch's numerical changes are present in an older tag.

All commands in this documentation assume that your terminal remains in this
directory. A quick check is:

```bash
test -f Project.toml && echo "repository root found"
```

On Windows PowerShell, use `Test-Path Project.toml` instead.

## 3. Install the Julia environment

Instantiate the packages declared by `Project.toml`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The first run downloads packages and precompiles them. It can take several
minutes and may appear quiet for short periods. Later starts reuse the compiled
cache. The transport runner, transport preprocessor, LL-to-CS binary regridder,
CS inversion driver, and binary-to-NetCDF converter import the installed project
package and reuse that cache. Launch these scripts with `--project=.` from this
checkout so Julia selects the matching source and dependencies.

## 4. Verify the package loads

```bash
julia --project=. -e 'using AtmosTransport; println("AtmosTransport is ready")'
```

Expected final line:

```text
AtmosTransport is ready
```

That is enough to continue. The [Quickstart](@ref Quickstart) performs a more
useful end-to-end check by creating a current transport binary, running the
model, and writing NetCDF output.

## Optional: run the regression suite

The complete synthetic CPU suite is intended for contributors and takes much
longer than the load check:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

It requires no private meteorological data. Real-data and large diagnostic
campaigns are separate opt-in test tiers.

## Optional: GPU support

Start on the CPU. Once the quickstart works, a run can request an available
backend through TOML:

```toml
[architecture]
use_gpu = true
backend = "auto"    # or "cuda" / "rocm" / "metal"

[numerics]
float_type = "Float32"
```

- NVIDIA runs use CUDA. `Float32` is the practical production default on most
  GPUs; `Float64` is supported where hardware performance permits.
- AMD Instinct runs use `backend = "rocm"` and AMDGPU.jl, with Float32 or
  Float64. Hardware validation is pending; see the [ROCm runbook](../development/rocm.md).
- Apple Silicon runs use Metal and require `Float32`. The extension has
  passed a C90/L66 forward smoke test on an M5 Pro with six and 32 tracers, including PPM, Dkg diffusion and full TM5 convection.
  Broader operator coverage and Metal adjoints remain unverified; see
  [Validation status](@ref).
- The runtime fails clearly if a requested GPU backend is unavailable; it does
  not silently change the execution path to CPU.

Backend packages are optional weak dependencies, so a fresh CPU installation
does not fetch them. Create a separate local environment containing this
checkout and the backend you need. Run **one** of these from the repository root:

```bash
julia -e 'using Pkg; Pkg.activate("gpu-env"); Pkg.develop(path="."); Pkg.add("CUDA")'
# Apple Silicon instead:
julia -e 'using Pkg; Pkg.activate("gpu-env"); Pkg.develop(path="."); Pkg.add("Metal")'
# AMD Instinct
julia -e 'using Pkg; Pkg.activate("gpu-env"); Pkg.develop(path="."); Pkg.add("AMDGPU")'
```

Then diagnose it after the base installation succeeds:

```bash
julia --project=gpu-env -e 'using CUDA; CUDA.versioninfo()'
julia --project=gpu-env -e 'using Metal; Metal.versioninfo()'
julia --project=gpu-env -e 'using AMDGPU; AMDGPU.versioninfo()'
```

Run only the command appropriate for your hardware. Use
`julia --project=gpu-env scripts/run_transport.jl my_run.toml` for GPU runs;
keep using `--project=.` for the base CPU environment. Preserve the local GPU
environment's `Project.toml` and `Manifest.toml` with your experiment records.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `julia: command not found` | Open a new terminal after installing Julia, or finish the juliaup PATH setup. |
| `Package AtmosTransport not found` | Run from the cloned repository and include `--project=.`. |
| A dependency “does not seem to be installed” | Re-run `julia --project=. -e 'using Pkg; Pkg.instantiate()'`. |
| Julia loads the wrong checkout | Run `Base.active_project()` in the REPL; it should end in this repository's `Project.toml`. |
| GPU package fails to load | Check `--project=gpu-env`; to finish the CPU quickstart, set `backend = "cpu"` and `use_gpu = false`. |

## Next step

Run the [Quickstart](@ref Quickstart). If the syntax is unfamiliar, keep the
[Julia orientation](@ref Julia-orientation) page open alongside it.
