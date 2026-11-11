# Sources of the GMD paper figures and tables

Every figure and table of the model-description paper has its source here: the scripts that
produce the inputs, the run configurations (or the generator that writes them), the analysis,
and the small results (CSV tables, PDF figures with PNG previews). Large files — transport
binaries, model output, regenerated intermediate files — are not committed; each script takes
their location from an argument or an environment variable.

| Folder | Reproduces | Large outputs |
|---|---|---|
| [`idealized_advection/`](idealized_advection/) | Sect. 4.1, standard transport tests: solid-body rotation and the deformational flows of Nair and Lauritzen (2010) on cubed-sphere, latitude–longitude and reduced-Gaussian grids, all advection schemes, FP32 and FP64 | `$IDEALIZED_ADVECTION_WORKDIR` (current case table, 202 runs: binaries about 150 GB, run output about 35 GB) |
| [`grids/`](grids/) | Figure of the three transport grids (latitude–longitude, reduced Gaussian, cubed sphere) coloured by cell area | none (`cs_c<N>_corners.csv` is regenerated) |

All commands run from the repository root with the project environment
(`julia --project=. ...`); the Python scripts need numpy, matplotlib and netCDF4 (the grid
figure also cartopy for the coastlines).

## Idealized advection (Sect. 4.1)

Analytic winds enter as ordinary transport binaries and run through
`scripts/run_transport.jl`; nothing in `src/` is test specific. See
[`idealized_advection/README.md`](idealized_advection/README.md) for the method, the files and
the checks.

```bash
export IDEALIZED_ADVECTION_WORKDIR=/path/to/scratch       # large outputs go here
export GPU_LOCK_DIR=/path/to/locks                        # only for the Float32 runs that cases.toml puts on GPUs
JOBS=8 THREADS=8 GPUS="0 1" examples/paper/idealized_advection/run_matrix.sh all
# or stage by stage: generate | configs | run | analyze | plot
```

Results: `idealized_advection/results/*.csv`; figures: `idealized_advection/figures/`.

## Grid figure

```bash
julia --project=. examples/paper/grids/export_cs.jl 48 examples/paper/grids/cs_c48_corners.csv
python3 examples/paper/grids/make_grid_figure.py <output dir> 48
```

`make_grid_figure.py` reads `cs_c<N>_corners.csv` and the ECMWF ring sizes `n<N>_pl.txt` from
its own folder and writes `grids_c<N>.pdf` and `.png` to the output directory (default: its own
folder).
