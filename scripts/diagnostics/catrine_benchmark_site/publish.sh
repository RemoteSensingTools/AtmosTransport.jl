#!/usr/bin/env bash
# Build the AtmosTransport v0.5 CATRINE benchmark site into the staging folder, then copy it to the
# public web folder.
#
#   scripts/diagnostics/catrine_benchmark_site/publish.sh                 # build + publish
#   scripts/diagnostics/catrine_benchmark_site/publish.sh --build-only    # build into staging only
#
# Environment: STAGING (default /temp1/cfranken/plan/w3/signoff/site_staging), PUBLIC (default
# ~/www/catrine/benchmark_v0.5). Further arguments go to catrine_benchmark_v05_site.py.
# The copy deletes files in PUBLIC that the build no longer has, so PUBLIC must be empty, absent, or a
# folder this script published before (it carries the marker file .catrine_benchmark_site).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGING="${STAGING:-/temp1/cfranken/plan/w3/signoff/site_staging}"
PUBLIC="${PUBLIC:-$HOME/www/catrine/benchmark_v0.5}"
BUILD_ONLY=0
if [[ "${1:-}" == "--build-only" ]]; then BUILD_ONLY=1; shift; fi
for arg in "$@"; do
  [[ "$arg" == --out || "$arg" == --out=* ]] && { echo "error: set STAGING instead of passing --out" >&2; exit 1; }
done

python3 "$HERE/../catrine_benchmark_v05_site.py" --out "$STAGING" "$@"
[[ -f "$STAGING/.catrine_benchmark_site" ]] || { echo "error: $STAGING has no build marker" >&2; exit 1; }
if (( BUILD_ONLY )); then echo "built $STAGING (not published)"; exit 0; fi

if [[ -d "$PUBLIC" && -n "$(ls -A "$PUBLIC")" && ! -f "$PUBLIC/.catrine_benchmark_site" ]]; then
  echo "error: $PUBLIC is not empty and was not published by this script; refusing to overwrite it" >&2
  exit 1
fi
mkdir -p "$PUBLIC"
rsync -a --delete --exclude '_cache/' "$STAGING"/ "$PUBLIC"/
chmod -R a+rX "$PUBLIC"
echo "published $STAGING -> $PUBLIC"
