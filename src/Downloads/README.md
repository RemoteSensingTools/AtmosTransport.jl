# Downloads

TOML-driven download pipeline for raw meteorological input: ERA5 (from CDS,
MARS, or Google ARCO), GEOS-FP, GEOS-IT, and MERRA-2.

The module is `DataDownloads`, not `Downloads`, because
[`verification.jl`](verification.jl) uses the `Downloads` stdlib (as `DL`).
`src/AtmosTransport.jl` loads it last, after `Models`. It imports no other
AtmosTransport module; it uses only `Dates`, `Logging`, `Printf`, `SHA`,
`TOML`, and the `Downloads` stdlib. `scripts/downloads/download_data.jl`
`include`s [`Downloads.jl`](Downloads.jl) on its own, without the package, so
the module must stay free of `..` imports.

## Entry Points

- CLI: `julia --project=. scripts/downloads/download_data.jl config/downloads/<recipe>.toml [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--dry-run] [--verify]`
- `download_data!(cfg; start_date, end_date, dry_run, verify_only)` in
  [`pipeline.jl`](pipeline.jl) parses the recipe, groups dates, builds tasks,
  then skips, verifies, or executes each one
- `parse_download_config(cfg) -> DownloadConfig` in
  [`configuration.jl`](configuration.jl)
- There are two dispatch points:
  - `build_tasks(source, protocol, dates, output, requests)` has a fallback in
    [`pipeline.jl`](pipeline.jl) and methods in `sources/`
  - `execute!(task, protocol; max_retries, retry_wait)` has methods in
    [`pipeline.jl`](pipeline.jl) and [`sources/era5_arco.jl`](sources/era5_arco.jl)
- `verified_download(url, dest; max_retries)` and `verify_downloads` are in
  [`verification.jl`](verification.jl)
- `detect_python_env(python) -> PythonEnvironment` is in
  [`python_interop.jl`](python_interop.jl)

## File Map

- [`Downloads.jl`](Downloads.jl): module `DataDownloads`, includes, exports
- [`types.jl`](types.jl): source and protocol types, `PythonEnvironment`,
  `DownloadTask`, `OutputConfig` with `canonical_output_dir`,
  `ScheduleConfig`, `DownloadConfig`
- [`configuration.jl`](configuration.jl): recipe parsing. It resolves
  `[source].met_source` against the project root, maps the met-source
  `[source].name` to a source type, and builds protocol, output, schedule, and
  requests.
- [`pipeline.jl`](pipeline.jl): `download_data!`, date chunking, verification
  sidecars, `execute!` for HTTP/S3/CDS/MARS/OPeNDAP, `_with_retries`, console
  output
- [`python_interop.jl`](python_interop.jl): Python probe, generation of
  `cdsapi` / `ecmwfapi` scripts, subprocess runner
- [`verification.jl`](verification.jl): HTTP download checked against
  Content-Length, and a directory scan

## Sources (`sources/`)

Each file adds a `build_tasks` method for one source and its protocol.

| File | Source type | Met-source `name` | Default protocol | Output files |
|---|---|---|---|---|
| [`sources/era5.jl`](sources/era5.jl) | `ERA5Source` | `ERA5` | `cds` (`mars` also accepted) | one GRIB per request group per chunk; `filename_template` tokens `{request_name}`, `{year_month}`, `{date}` |
| [`sources/era5_arco.jl`](sources/era5_arco.jl) | `ERA5ARCOSource` | `ERA5-ARCO` | `gcs` | `ml_an_native_core/era5_core_YYYYMMDD.grib` (concatenated ARCO parts), `sfc_an_native/arco/YYYYMMDD/<var>.nc` |
| [`sources/geosfp.jl`](sources/geosfp.jl) | `GEOSFPSource` | `GEOS-FP` | `http` | products `geosfp_c720` (24 hourly CTM files per day) and `geosfp_025` |
| [`sources/geosit.jl`](sources/geosit.jl) | `GEOSITSource` | `GEOS-IT` | `s3` | `GEOSIT.YYYYMMDD.<coll>.C180.nc`; the default collection list includes `CTM_I1` |
| [`sources/merra2.jl`](sources/merra2.jl) | `MERRA2Source` | `MERRA-2` | `opendap` | `MERRA2_<stream>.<collection>.YYYYMMDD.nc4` |

- [`sources/era5_arco.jl`](sources/era5_arco.jl) also defines `execute!` for
  `GCSProtocol`. With `assemble="single"` it is a verified download. With
  `assemble="concat"` it fetches the component objects concurrently and
  joins them with `cat`.
- OPeNDAP downloads are not implemented (`protocol_can_download`): the MERRA-2
  recipe runs with `--dry-run` or `--verify`; without them `download_data!`
  refuses it before downloading anything.

## Conventions

- Recipes have two levels. A recipe in `config/downloads/*.toml` points at a
  met-source description in `config/met_sources/*.toml` through
  `[source].met_source`. The recipe holds `[download]`, `[output]`,
  `[schedule]`, and `[options]`; the met source holds `[source].name`,
  `[access]`, and (for MERRA-2) `[collections]`.
- Output goes to
  `<data_root>/met/<met_source>/<grid_name>/<cadence>/<payload_type>/`
  (`canonical_output_dir`). `data_root` is passed through `expanduser` only;
  environment variables are not substituted.
- A successful download writes the sidecar `<dest>.download.toml`. It holds a
  format version, the task identity (protocol identity, source URL, and the
  request sorted by key), the size, and the SHA-256.
- `_existing_task_status` classifies a file as `:missing`, `:corrupt`,
  `:verified`, or `:unverifiable`. A sidecar wins. With no sidecar, HTTP and
  unsigned S3 fall back to an HTTP `HEAD` Content-Length check.
- With `skip_existing` (the default), `:verified` and `:unverifiable` files are
  skipped; the second case logs a warning. `--verify` counts `:unverifiable`
  as a failure.
- Downloads are staged as `<dest>.part` and then `mv`d into place.
- `chunk` is `monthly`, `daily`, or `per_file`, where `per_file` groups like
  `daily`. Tasks run one after another.

## Tests And Docs

- [`../../test/core/test_persistence_hardening.jl`](../../test/core/test_persistence_hardening.jl):
  sidecar resume semantics (identity, size, and checksum changes)
- Recipes and usage: [`../../scripts/downloads/README.md`](../../scripts/downloads/README.md),
  [`../../docs/src/config/data_sources.md`](../../docs/src/config/data_sources.md)
- API page: [`../../docs/src/api/downloads.md`](../../docs/src/api/downloads.md)
