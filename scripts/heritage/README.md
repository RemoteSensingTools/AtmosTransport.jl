# Heritage campaign launchers

Scripts in `heritage/` are not part of a maintained workflow. They are kept
for reference, as the method or evidence of a finished study, until they are
trimmed. They may hard-code the paths, dates and data of that study, and they
are not tested; check imports and inputs before running one. Retired scripts
are listed under "Retired scripts" in [`scripts/README.md`](../README.md).

| Script | Purpose | Why it is here |
|---|---|---|
| `run_trendy_s3_2021_batches.sh` | Runs the 23-model TRENDY v14 S3 2021 C90 ensemble as four GPU-memory-safe batches in two concurrent waves (one per GPU) via `run_transport.jl` | Launcher of the completed run; its four batch configs remain in `config/runs/` and the TRENDY re-run guide memo names it. Hard-codes `RUNTIME_DIR=/tmp/AtmosTransportModel-era-c90` |
