# Scripts

Command-line entry points and tools around the package. Library code lives in
`src/`; a script that needs a reusable function should call the library (or
move the function there), not copy it.

| Folder | Contents |
|---|---|
| [`run_transport.jl`](run_transport.jl) | The runtime entry point: `julia --project=. scripts/run_transport.jl <config.toml>` |
| [`preprocessing/`](preprocessing/) | Building transport binaries (`preprocess_transport_binary.jl`, the canonical CLI), regridding and attaching fields, and preparing surface-flux inputs |
| [`downloads/`](downloads/) | `download_data.jl` with the recipes in `config/downloads/` |
| [`diagnostics/`](diagnostics/) | Inspection and comparison tools that take paths as arguments (`inspect_transport_binary.jl`, the CATRINE benchmark suite against GEOS-Chem, mass-balance and Float32 checks) |
| [`validation/`](validation/) | Checks of binaries and runs against references (continuity, cubed sphere vs lat-lon, GEOS-IT) |
| [`visualization/`](visualization/) | Plots and animations; `cs_regrid_utils.jl` is shared by several of them |
| [`benchmarks/`](benchmarks/) | Synthetic and real-input performance benchmarks (results in `benchmarks/results/`) |
| [`postprocess/`](postprocess/), [`inversions/`](inversions/), [`checks/`](checks/) | Snapshot conversion, inversion prototypes, documentation checks |
| [`completed_experiments/`](completed_experiments/) | Scripts of closed investigations, kept for reference; they hard-code the dates and paths of their investigation |
| [`deprecated/`](deprecated/) | Superseded entry points |

The campaign launchers at the top level (`run_*_campaign.py`,
`run_catrine_c30_*.sh`, `run_campaign5d.sh`) drive specific production
campaigns; their defaults point at the machines and directories of those
campaigns.
