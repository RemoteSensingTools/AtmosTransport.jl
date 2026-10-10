# Heritage preprocessing

Scripts in `heritage/` are not part of a maintained workflow. They are kept
for reference, as the method or evidence of a finished study, until they are
trimmed. They may hard-code the paths, dates and data of that study, and they
are not tested; check imports and inputs before running one. Retired scripts
are listed under "Retired scripts" in [`scripts/README.md`](../../README.md).

| Script | Purpose | Why it is here |
|---|---|---|
| `build_catrine_c30_2022_macbook_bundle.sh` | Stages a transfer-ready CATRINE C30 2021-12 to 2022 bundle for a MacBook (hardlinked format-4 met, emissions, ICs, runtime checkout, config, run script, manifest) | Only recipe for rebuilding the bundle staged under `/temp2/catrine-runs/bundles`; its config, `docs/memos/CATRINE_C30_2022_MACBOOK_BUNDLE.md` and `scripts/run_catrine_c30_2022_macbook.sh` exist. Copies the runtime from a `/tmp` checkout |
| `build_era5_geos_c180_cfl85_tm5_surface.sh` | Builds Dec 2-4 2021 ERA5 LL cfl85 TM5+surface binaries and regrids them to GEOS-native C180 | Build recipe of the May 2026 C180 cfl85 campaign, cited by its configs and `docs/c180_ppm_campaign_locations.md`. Hard-codes a `cd` into the main checkout and personal data paths |
