#!/usr/bin/env bash
# Quick gate: the fast checks to run before committing or handing work back.
#
#   1. whitespace errors in the changes (`git diff --check`): committed since
#      the merge base with <ref>, uncommitted, and untracked files;
#   2. calls that omit a required keyword (scripts/checks/check_keyword_calls.jl),
#      in src/, ext/, test/, scripts/, benchmarking/, examples/, docs/literate
#      and docs/scripts;
#   3. the quick test set plus the topics of the changed files, in one Julia
#      process (`test/runtests.jl --quick --changed=<ref>`).
#
# Every step runs; the script prints a summary and exits non-zero if any step
# failed. The full suite (no flags) remains the merge requirement in CI.
#
# Usage (from anywhere in the checkout):
#
#   scripts/dev/quick_gate.sh [<git-ref>]      # default ref: origin/main
#
# Environment: JULIA (default `julia`) selects the Julia binary, e.g.
# JULIA="julia +1.10". The test environment must be prepared once:
#   julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

set -u
ref="${1:-origin/main}"
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

echo "== 1/3 git diff --check (${ref}...HEAD, uncommitted, untracked)"
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

echo "== 2/3 required-keyword calls"
start=$SECONDS
"${julia_cmd[@]}" --startup-file=no scripts/checks/check_keyword_calls.jl
record "required-keyword calls" "$?" "$((SECONDS - start))"

echo "== 3/3 tests: --quick --changed=${ref}"
start=$SECONDS
"${julia_cmd[@]}" --startup-file=no --project=test test/runtests.jl --quick "--changed=${ref}"
record "tests (--quick --changed=${ref})" "$?" "$((SECONDS - start))"

echo
echo "quick gate summary (${ref}):"
printf '  %s\n' "${summary[@]}"
if [ "$status" -ne 0 ]; then
    echo "quick gate FAILED"
else
    echo "quick gate passed; CI runs the full suite before merge"
fi
exit "$status"
