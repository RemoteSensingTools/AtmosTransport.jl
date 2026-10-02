# CATRINE C30 2022 MacBook bundle

This bundle contains the C30, 66-level, dry-air full-physics transport
binaries from 2021-12-01 through 2022-12-31. December 2021 is required
spin-up; the analysis year is 2022. It also includes the 13 monthly LMDZ CO2
flux inputs, two annual GridFED files, EDGAR SF6 and Zhang Rn-222 fluxes,
CO2/SF6 initial conditions, and the format-4 runtime source.

The bundle is a directory rather than a compressed archive because the input
data are already compressed/binary and total about 133 GiB. Copy it with
`rsync`; it can resume an interrupted transfer.

On the Mac, install Julia 1.10 or newer, then run from the bundle directory:

```bash
julia --project=runtime/AtmosTransportModel-era-c90 -e 'using Pkg; Pkg.instantiate()'
./run_catrine_c30_2022_macbook.sh
```

The supplied run is CPU-only for compatibility across MacBook models. It uses
all logical CPU cores unless `JULIA_NUM_THREADS` is already set. Daily
NetCDF output with four snapshots per day (00, 06, 12, 18 UTC) is written to
`output/`, and the terminal stream is retained in `output/run.log`.

`manifest.files` lists the complete payload. The run configuration has
portable paths rooted at `ATMOSTRANSPORT_DATA_ROOT`; it must be run using
the included format-4 runtime, not the older local checkout that rejects
these transport binaries.
