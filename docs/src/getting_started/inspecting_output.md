# [Inspecting input and output](@id Inspecting-output)

AtmosTransport reads a **transport binary** and writes **NetCDF snapshots**.
The two formats have different jobs:

| File | Contains | Typical question |
|---|---|---|
| `*.bin` | grid, air mass, mass fluxes, and optional physics fields | Can this forcing run the operators I requested? |
| `*.nc` | tracer and air-mass snapshots | What did the simulated atmosphere look like? |

The [Quickstart](@ref) creates both kinds beneath `data/quickstart/`.

## Check a transport binary before running

Use the inspector on any binary, synthetic or preprocessed:

```bash
julia --project=. scripts/diagnostics/inspect_transport_binary.jl \
    data/quickstart/synthetic_latlon_v4.bin
```

It reports the grid, precision, mass basis, time windows, payload fields, and
operator capabilities. For example, the synthetic quickstart has advection and
replay fields but no convection or diffusion fields. Asking the runtime to use
an unsupported operator therefore fails immediately with a useful error.

All current inputs must have `format_version = 4` or `5` (version 5 adds the
flux-construction identity; see [Binary format](../concepts/binary_format.md)).
Regenerate an older binary with the current preprocessor; the runtime does not
guess old semantics.

From Julia, the same check is:

```julia
using AtmosTransport

capabilities = inspect_binary("data/quickstart/synthetic_latlon_v4.bin")
@show capabilities.grid_type capabilities.mass_basis
```

## Open the quickstart output in Julia

`NCDatasets` is already a project dependency:

```julia
using NCDatasets

NCDataset("data/quickstart/synthetic_output.nc") do ds
    println(keys(ds.variables))
    co2 = ds["co2_bl_column_mean"][:, :, :]
    println("shape: ", size(co2))
    println("range: ", extrema(co2))
end
```

For the maintained example, the column-mean array has shape `(36, 18, 5)`:
longitude × latitude × output time. Its values begin at `4e-4` and develop a
longitudinal pattern as the synthetic flow transports the tracer.

If you have NetCDF command-line tools installed, this gives a quick structural
check without starting Julia:

```bash
ncdump -h data/quickstart/synthetic_output.nc | less
```

## Understand variable names and dimensions

Tracer names come from the TOML table name. For `[tracers.co2_bl]`, output
variables begin with `co2_bl`:

| Variable | Meaning |
|---|---|
| `co2_bl` | tracer mixing ratio on every model level |
| `co2_bl_column_mean` | air-mass-weighted column mean |
| `co2_bl_column_mass_per_area` | conservative tracer storage per unit area; not physical kg of CO₂ |
| `co2_bl_total_mass` | signed Float64 total of conservative tracer storage; use this for mass-budget checks |
| `air_mass` | air mass on every model level |
| `column_air_mass_per_area` | vertically integrated air mass per unit area |

The horizontal dimensions depend on the grid:

| Grid | Full three-dimensional tracer | Column mean |
|---|---|---|
| Latitude–longitude | `(lon, lat, lev, time)` | `(lon, lat, time)` |
| Reduced Gaussian | native cells plus rasterized products | native and `(lon, lat, time)` products |
| Cubed sphere | `(Xdim, Ydim, nf, lev, time)` | `(Xdim, Ydim, nf, time)` |

Some Python tools display dimensions in the opposite order because they follow
row-major conventions. Read the variable's named dimensions instead of
assuming an axis order.

The `[output.fields]` table can intentionally omit full three-dimensional
fields, select tracers, or select model levels. Check that table first when an
expected variable is absent. See [Output schema](@ref) for every option.

Runtime NetCDF spatial fields use the simulation's configured precision; the
independent total-mass series always uses Float64. For streamed NetCDF output, also check the global
`completed_snapshots` attribute and the last output time: an interrupted run
can leave a readable partial file. See [Output schema](@ref) before using it.

## Interpret the run summary

At shutdown the runner prints relative air-mass and conservative tracer-storage
changes. The
synthetic quickstart is closed and has no sources, so both should be near the
floating-point noise floor. A source-driven experiment can legitimately change
model storage; compare the change with the reported integrated source rather
than expecting zero. This storage is `mixing ratio × carrier-air-mass`, not
physical kilograms of the tracer species.

## Common problems

| Symptom | First check |
|---|---|
| Binary rejected before the run | Confirm `format_version` is 4 or 5 and inspect its capabilities. |
| Requested convection or diffusion is unavailable | Reprocess forcing with the required fields enabled. |
| A tracer variable is missing | Check `[output.fields]` and the tracer table name. |
| Extreme CFL values or non-finite output | Check vertical ordering, flux time units, and whether the binary was produced by the current preprocessor. |
| A closed run has appreciable mass drift | Check the conservation budget and replay gate before interpreting tracer results. |

Continue with [Architecture tour](@ref) for the model's moving pieces, or
[Run with real meteorology](@ref) to replace the synthetic forcing.
