# [Run with real meteorology](@id Run-with-real-meteorology)

The [Quickstart](@ref Quickstart) uses synthetic forcing so that everyone can
run it. A scientific run uses the same runner and TOML structure, but points at
one or more transport binaries generated from ERA5 or GEOS meteorology.

## What you need

A forward run has three inputs:

1. **Transport binaries**—one or more version-4 files produced by the current
   preprocessor. They contain grid geometry, air mass, mass fluxes, timing, and
   any requested diffusion or convection fields.
2. **A run TOML**—selects precision, architecture, operators, tracers, initial
   conditions, emissions, and output.
3. **Optional scientific data**—file-backed initial conditions or emissions
   inventories referenced by the TOML.

Raw ERA5 GRIB or GEOS NetCDF is not opened by the runtime. See
[Preprocessing overview](@ref Preprocessing-overview) when you need to create
transport binaries.

## 1. Inspect the forcing first

Before writing a run configuration, inspect one file:

```bash
julia --project=. scripts/diagnostics/inspect_transport_binary.jl \
    /path/to/era5_transport_20211201_float32.bin
```

Confirm:

- `format_version` is 4 (older files are rejected);
- `grid_type` and vertical level count match the intended experiment;
- `mass_basis` is `dry` for the standard runtime;
- the required capability is present for every requested operator—for example,
  TM5 convection needs `entu`, `detu`, `entd`, and `detd`.

This check catches incompatible data before model allocation or GPU startup.

## 2. Copy the canonical template

```bash
cp config/examples/minimal_template.toml my_run.toml
```

The template uses the current section names. Replace its synthetic input and
output paths, then adjust the physics and tracers for the experiment.

### Input: explicit files

Use an explicit list for a short run or files without a date naming pattern:

```toml
[input]
binary_paths = [
    "/data/transport/era5_transport_20211201_float32.bin",
    "/data/transport/era5_transport_20211202_float32.bin",
]
```

The list is consumed in the order you provide. Path expansion does not sort
explicit lists or check their dates; list consecutive files chronologically.

For complete daily files, omit `[run].stop_window`. Window indices belong to
each file: two 24-window daily files make a 48-hour run, not a file with window
48. Multi-file runs must consume each file's full range so forcing is not
skipped at the handoff. Output schedules use hours since the whole run began.

### Input: folder and date range

Use folder expansion for daily production files:

```toml
[input]
folder = "$ATMOSTRANSPORT_DATA_ROOT/met/era5/ll72x37/transport"
start_date = "2021-12-01"
end_date = "2021-12-10"

# Add this only when the filename needs an exact pattern:
# file_pattern = "era5_transport_{YYYYMMDD}_float32.bin"
```

The range is inclusive. The runner verifies that every date exists and rejects
duplicates or gaps.

### Architecture and precision

Start on CPU when validating a new configuration:

```toml
[architecture]
use_gpu = false
backend = "cpu"

[numerics]
float_type = "Float32"
```

After the CPU smoke run succeeds, set `use_gpu = true` and choose `"cuda"`,
`"metal"`, or `"auto"`. Metal requires `Float32`.
If you created the local GPU environment in [Installation](@ref), use
`--project=gpu-env` instead of `--project=.` for that run.

### Operators

```toml
[advection]
scheme = "slopes"

[diffusion]
kind = "none"

[convection]
kind = "none"
```

Operator selection must agree with the binary capabilities. Omitting an
optional physics block or using `kind = "none"` selects its typed no-op. See
[Operators](@ref Operator-concepts) for the available schemes and [TOML schema](@ref) for every
key.

### Tracers and output

```toml
[tracers.co2.init]
kind = "uniform"
background = 400.0e-6       # dry mol/mol = 400 ppm

[output]
cadence_hours = 6
path = "$ATMOSTRANSPORT_DATA_ROOT/output/my_run.nc"
split = "single"
```

Tracer initial conditions are dry volume mixing ratios. Internally the model
converts them to conservative `mixing ratio × carrier-air-mass` storage.
Output converts back to mixing ratio and writes
the mass diagnostics needed to reproduce column means.

## 3. Validate and run

The command-line runner performs inexpensive configuration checks before
opening the full simulation:

```bash
julia --project=. scripts/run_transport.jl my_run.toml
```

You can also check a configuration from Julia without opening its binaries or
allocating model state:

```julia
using AtmosTransport, TOML
cfg = TOML.parsefile("my_run.toml")
ok, errors = validate_config(cfg)
foreach(println, errors)
```

This reports malformed tables, missing input paths, incompatible precision and
backend settings, invalid window indices, and invalid physics or output
settings; keys the run would ignore (often typos) are logged as warnings. For example, `co2 = 0.0004`
directly inside `[tracers]` is rejected: use the `[tracers.co2.init]` table shown
above. Window indices must be integers; `start_window = true` and
`start_window = 1.0` are errors. Shape errors are reported first, before the
remaining value checks.

`ok = true` does not validate binary contents or establish that the requested
physics matches the forcing. Keep the binary inspection in step 1. GPU
auto-detection can load and probe runtime packages, and the CLI preloads a
requested GPU package during startup.

The script may restart itself with two Julia threads so NetCDF writes can
overlap computation. Set `ATMOSTR_NO_AUTO_THREADS=1` only when debugging a
single-threaded path.

A successful startup summary identifies the choices that matter:

```text
Driven runtime
|-- grid:      ...
|-- numerics:  scheme=Slopes, FT=Float32, backend=CPU
|-- physics:   diffusion=NoDiffusion, convection=NoConvection
|-- schedule:  window_dt=3600s, steps/window=..., binaries=...
|-- tracers:   co2
`-- output:    .../my_run.nc
```

At completion, check the final air-mass and per-tracer storage-budget lines. Their
tolerance depends on precision and enabled sources/sinks; unexplained drift in
a closed advection-only run is a failure, not a cosmetic warning.

## Data roots

`$ATMOSTRANSPORT_DATA_ROOT` defaults to `~/data/AtmosTransport`. Set it once to
keep configs portable across machines:

```bash
export ATMOSTRANSPORT_DATA_ROOT=/scratch/$USER/AtmosTransport
```

Paths beginning with `~/` and environment variables are expanded by the
runtime. Relative paths are interpreted from the directory where the command is
run, which is another reason to run from the repository root.

## Create current forcing from raw data

The canonical preprocessing command is:

```bash
julia --project=. --threads=8 \
    scripts/preprocessing/preprocess_transport_binary.jl \
    config/preprocessing/<source-and-grid>.toml \
    --day 2021-12-01
```

Use `--start` and `--end` where the selected source supports a range. The
preprocessor performs dry-air conversion, topology/vertical transforms,
continuity balancing, adaptive substep selection, and write-time replay and
positivity gates before committing the binary.

Choose the source-specific guide next:

- [ERA5 spectral path](@ref) for lat-lon, reduced Gaussian, or cubed sphere.
- [GEOS native cubed-sphere](@ref) for GEOS-IT or GEOS-FP native panels.
- [Data sources](@ref) for authentication and raw-file layout.

## Common first-run failures

| Message or symptom | Meaning and first action |
|---|---|
| `unsupported transport binary version` | Regenerate the forcing with the current version-4 preprocessor. |
| `resolved path does not exist` | Check the active data root and run from the repository root. |
| Missing convection/diffusion payload | The preprocessing config did not include the fields required by the selected operator. |
| GPU backend unavailable | Set CPU in TOML, verify the run, then diagnose the optional backend separately. |
| Date range has gaps | Add the missing daily binary or use an explicit `binary_paths` list. |
| Extreme CFL or positivity failure | Inspect the binary schedule and preprocessing gates; do not work around it by suppressing the error. |

## Next steps

- [Inspecting output](@ref) explains variables and dimension order.
- [Architecture tour](@ref Architecture-tour) connects the runner to the typed
  source structure.
- [TOML schema](@ref) is the complete configuration reference.
