#!/usr/bin/env bash
# Quick gate: the fast checks to run before committing or handing work back.
#
#   1. whitespace errors in the changes (`git diff --check`): committed since
#      the merge base with <ref>, uncommitted, and untracked files;
#   2. calls that omit a required keyword (scripts/checks/check_keyword_calls.jl),
#      in src/, ext/, test/, scripts/, benchmarking/, examples/, docs/literate
#      and docs/scripts;
#   3. source structure ratchet (scripts/checks/check_structure.jl);
#   4. the quick test set plus the topics of the changed files, in one Julia
#      process (`test/runtests.jl --quick --changed=<ref>`).
#
# Every step runs; the script prints a summary and exits non-zero if any step
# failed. CI runs the full suite for changes outside documentation; see
# test/README.md for the documentation-only exception and merge requirements.
#
# Usage (from anywhere in the checkout):
#
#   scripts/dev/quick_gate.sh [<git-ref>]      # default: ATMOSTR_STRUCTURE_BASE or origin/main
#
# Environment: JULIA (default `julia`) selects the Julia binary, e.g.
# JULIA="julia +1.10". JULIA_NUM_THREADS defaults to 1 when unset, matching CI
# and keeping reverse-halo atomic summation deterministic. An explicit thread
# count is preserved. The test environment must be prepared once:
#   julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

set -u
ref="${1:-${ATMOSTR_STRUCTURE_BASE:-origin/main}}"
export ATMOSTR_STRUCTURE_BASE="$ref"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS-1}"
julia_cmd=(${JULIA:-julia})
root="$(git rev-parse --show-toplevel)" || exit 2
cd "$root" || exit 2

if ! git rev-parse --verify --quiet "${ref}^{commit}" > /dev/null; then
    echo "quick_gate: unknown git ref '${ref}' (fetch it, or pass another ref)" >&2
    exit 2
fi

declare -a summary
status=0
record() {  # record <step> <exit code> [<seconds>]
    if [ "$2" -eq 0 ]; then
        summary+=("PASS  $1${3:+ (${3} s)}")
    else
        summary+=("FAIL  $1${3:+ (${3} s)}")
        status=1
    fi
}

echo "== 1/4 git diff --check (${ref}...HEAD, uncommitted, untracked)"
whitespace=0
git --no-pager diff --check "${ref}...HEAD" || whitespace=1
git --no-pager diff --check HEAD || whitespace=1
# Untracked files: `--no-index` exits with bit 1 for any difference and bit 2
# for whitespace errors.
while IFS= read -r -d '' f; do
    git --no-pager diff --no-index --check /dev/null "$f"
    (( $? & 2 )) && whitespace=1
done < <(git ls-files -z --others --exclude-standard)
record "whitespace (git diff --check)" "$whitespace"

echo "== 2/4 required-keyword calls"
start=$SECONDS
"${julia_cmd[@]}" --startup-file=no scripts/checks/check_keyword_calls.jl
record "required-keyword calls" "$?" "$((SECONDS - start))"

echo "== 3/4 source structure"
start=$SECONDS
"${julia_cmd[@]}" --startup-file=no --project=test scripts/checks/check_structure.jl "--base=${ref}"
record "source structure" "$?" "$((SECONDS - start))"

echo "== 4/4 tests: --quick --changed=${ref}"
start=$SECONDS
"${julia_cmd[@]}" --startup-file=no --project=test test/runtests.jl --quick "--changed=${ref}"
record "tests (--quick --changed=${ref})" "$?" "$((SECONDS - start))"

echo
echo "quick gate summary (${ref}):"
printf '  %s\n' "${summary[@]}"
if [ "$status" -ne 0 ]; then
    echo "quick gate FAILED"
else
    echo "quick gate passed; see test/README.md for CI merge requirements"
fi
exit "$status"
