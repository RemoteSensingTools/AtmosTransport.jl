#!/usr/bin/env bash
# Stage a transfer-ready CATRINE C30 package. C30 binary files are hardlinked
# on /temp2, so staging does not duplicate the ~128 GiB local met payload.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
bundle_root=${1:-/temp2/catrine-runs/bundles/catrine_c30_2022_macbook}
met_source=/temp2/catrine-runs/met/era5_c30_dec2021_through_2022_fullphysics_experimental
runtime_source=/tmp/AtmosTransportModel-era-c90
data_root=/home/cfranken/data/AtmosTransport/catrine

[[ -d "$met_source" ]] || { echo "missing met source: $met_source" >&2; exit 1; }
[[ -d "$runtime_source" ]] || { echo "missing format-4 runtime: $runtime_source" >&2; exit 1; }

mkdir -p "$bundle_root"/{met,emissions/LMDZ_fluxes,emissions/gridfed,emissions/edgar_v8,emissions/ZHANG_Rn222,initial_conditions,runtime,config,output}

met_count=0
while IFS= read -r -d '' source_file; do
    target_file="$bundle_root/met/$(basename "$source_file")"
    if [[ ! -e "$target_file" ]]; then
        ln "$(realpath "$source_file")" "$target_file"
    fi
    met_count=$((met_count + 1))
done < <(find -L "$met_source" -maxdepth 1 -type f -name '*.bin' -print0 | sort -z)
[[ "$met_count" == 396 ]] || { echo "expected 396 C30 binaries, found $met_count" >&2; exit 1; }

rsync -a --ignore-existing \
  "$data_root/Emissions/LMDZ_fluxes/z_cams_l_cams55_202112_FT24r2_ra_sfc_3h_co2_flux.nc" \
  "$data_root/Emissions/LMDZ_fluxes/"*2022*_co2_flux.nc \
  "$bundle_root/emissions/LMDZ_fluxes/"
rsync -a --ignore-existing \
  "$data_root/Emissions/gridfed/GCP-GridFEDv2024.0_2021.short.nc" \
  "$data_root/Emissions/gridfed/GCP-GridFEDv2024.0_2022.short.nc" \
  "$bundle_root/emissions/gridfed/"
rsync -a --ignore-existing \
  "$data_root/Emissions/edgar_v8/v8.0_FT2022_GHG_SF6_2022_TOTALS_emi.nc" \
  "$bundle_root/emissions/edgar_v8/"
rsync -a --ignore-existing \
  "$data_root/Emissions/ZHANG_Rn222/Rn222_Emis_Zhang_Liu_et_al_05x05_mass.nc" \
  "$bundle_root/emissions/ZHANG_Rn222/"
rsync -a --ignore-existing \
  "$data_root/InitialConditions/startCO2_202112010000.nc" \
  "$data_root/InitialConditions/startSF6_202112010000.nc" \
  "$bundle_root/initial_conditions/"

rsync -a --exclude='.git' "$runtime_source/" "$bundle_root/runtime/AtmosTransportModel-era-c90/"
install -m 0644 "$repo_root/config/runs/catrine_c30_dec2021_2022_fullphysics_column_macbook.toml" "$bundle_root/config/"
install -m 0644 "$repo_root/docs/memos/CATRINE_C30_2022_MACBOOK_BUNDLE.md" "$bundle_root/README.md"
install -m 0755 "$repo_root/scripts/run_catrine_c30_2022_macbook.sh" "$bundle_root/"

(
    cd "$bundle_root"
    find config emissions initial_conditions met runtime -type f -printf '%P\t%s bytes\n' | LC_ALL=C sort
) > "$bundle_root/manifest.files"
{
    printf 'C30 transport binaries: %s\n' "$met_count"
    printf 'Transport binary format: 4\n'
    printf 'Period: 2021-12-01 through 2022-12-31 (December 2021 spin-up; 2022 analysis)\n'
} > "$bundle_root/BUNDLE_INFO.txt"
du -sh "$bundle_root"
echo "Bundle ready: $bundle_root"
