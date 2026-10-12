# Legacy Runtime Configs

This directory holds old runtime TOMLs that do not satisfy the canonical
`scripts/run_transport.jl` input schema.

Active run configs must declare `[input]` with either `binary_paths` or
`folder` plus `start_date` and `end_date`. Files here are preserved for
provenance only; migrate one back to `config/runs/` only after updating it to
the current schema.

The nine `era5_*_catrine_48h_*.toml` files were moved here during the 0.4 review.
They use legacy `[run].tracer_name`, top-level `[init]` and
`stop_window=48` for two daily files. Their referenced binaries are unavailable
on the review host, so no real-input acceptance is claimed. Start a current
two-day run from `config/examples/minimal_template.toml`, supply consecutive
version-4 files, and omit `stop_window` to consume both complete files.

`catrine5d_c180_full137_f32_pbl_tm5_smoke_gpu.toml` is also provenance-only:
its pre-PBL-rename input directory was quarantined and contains no compatible
transport binary. It must not be reactivated without a regenerated L137 input
and an evidence-based collaborative-LU depth or aggregation choice.
