# Advection in prescribed flows

These benchmarks exercise horizontal tracer transport on the sphere. Each flow returns the tracer to its starting position after 12 days. The initial cell means therefore provide a reference for every grid and scheme. The generator writes face transfers derived from the prescribed flow to ordinary transport files. Four identical layers have zero vertical mass flux. The runs use the production transport driver and require no meteorological archive. [Case definitions and reproduction scripts](https://github.com/RemoteSensingTools/AtmosTransport.jl/tree/0b13756fdd979632987da801d9778ec2adb9276d/examples/paper/idealized_advection)

| Flow | Tracer fields | What to examine |
| --- | --- | --- |
| Tilted solid-body rotation | Gaussian hill, cosine bell, uniform tracer | Shape and displacement after one revolution across high latitudes and cubed-sphere panel edges |
| Nondivergent deformation | Gaussian hills, cosine bells, slotted cylinders, uniform tracer | Filament width at day 6, recovery at day 12, overshoots and undershoots at sharp edges |
| Divergent deformation | Gaussian hills, cosine bells, uniform tracer | Preservation of uniform mole fraction as the air mass in each cell changes |

## Resolution and shape preservation

The main comparison contains 87 matched FP32/FP64 pairs. On the cubed sphere all three flows were run at C48, C96 and C192. Solid-body rotation and nondivergent deformation were also run at three latitude–longitude resolutions and two reduced-Gaussian resolutions. The divergent case used one resolution on each of those two grids. The summary gives errors and observed orders by grid and scheme, including the grid spacing at which the cosine-bell error crosses the Lauritzen threshold. A missing order means there was no second resolution for that case. [Full results table](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/0b13756fdd979632987da801d9778ec2adb9276d/examples/paper/idealized_advection/results/summary.csv) · [Convergence curves](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/0b13756fdd979632987da801d9778ec2adb9276d/examples/paper/idealized_advection/figures/convergence.png)

For the deforming cosine bells, the normalized L2 error at day 12 crosses 0.033 at about 0.8° for Lin–Rood and 0.55° for the Colella–Woodward PPM option on the cubed sphere. The slopes and default PPM options remain above that threshold at C192. The slotted-cylinder maps show a different aspect of accuracy. Upwind smooths the sharp edges, while Lin–Rood retains the slots but can produce negative mole fractions near the edges. [Threshold estimates](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/0b13756fdd979632987da801d9778ec2adb9276d/examples/paper/idealized_advection/results/minimal_resolution.csv) · [All tracer maps](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/0b13756fdd979632987da801d9778ec2adb9276d/examples/paper/idealized_advection/figures/deformation_maps.png)

Reducing the averaging window from one hour to 15 minutes changed the day-12 normalized L2 errors in the tested deformational cases by less than 1%. Doubling the substep count at C96 increased the Gaussian-hill and cosine-bell errors by 3–19%. The reduced convergence rate of Lin–Rood at the finer resolutions remains unexplained. The benchmark directory includes the rotation-axis, window-length, substep, and panel-edge comparisons. [Sensitivity results](https://github.com/RemoteSensingTools/AtmosTransport.jl/tree/0b13756fdd979632987da801d9778ec2adb9276d/examples/paper/idealized_advection/results)

## Conservation and precision

Over the 12-day runs the largest relative change in global tracer mass was about 4 × 10⁻⁶ in FP32 and below 3 × 10⁻¹⁵ in FP64. The largest local FP32 departure from a uniform tracer reached 7.6 × 10⁻⁴ in the finest latitude–longitude rotation case, near the poles of the rotation axis. The precision diagnostics report the largest local error, the area affected, and the global burden separately. The real-meteorology column tests and the earlier 761-day comparison are documented in [Float32 conservation](float32_conservation.md). [Conservation figure](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/0b13756fdd979632987da801d9778ec2adb9276d/examples/paper/idealized_advection/figures/conservation.png) · [Uniform-tracer diagnostics](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/0b13756fdd979632987da801d9778ec2adb9276d/examples/paper/idealized_advection/results/uniform_tracer_stats.csv)

## Reproduce the tests

From the repository root, choose a work directory with space for generated forcing, model output, and reference fields. The analysis and plotting scripts use Python with NumPy, netCDF4, and Matplotlib. Python versions below 3.11 also require tomli. To generate the inputs, run the model, calculate the errors, and plot the results:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
export IDEALIZED_ADVECTION_WORKDIR=/path/to/scratch/idealized-advection
bash examples/paper/idealized_advection/run_matrix.sh all
```

The driver also accepts individual stages. With completed runs already in the work directory, use `analyze` to regenerate the tables or `plot` to redraw the figures. Each results row identifies its grid, scheme, tracer, precision, and sampling time. [Benchmark setup and file layout](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/0b13756fdd979632987da801d9778ec2adb9276d/examples/paper/idealized_advection/README.md)


The two compact paper figures can be regenerated directly from the archived CSV and the original deformation maps. This step also uses Pillow and Poppler's `pdfimages` utility. Run the following from the repository root. The optional `--output` argument chooses a different destination.

```bash
python3 examples/paper/idealized_advection/plot_paper_summary.py
```
