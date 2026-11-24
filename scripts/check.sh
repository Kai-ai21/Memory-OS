#!/usr/bin/env bash
# make check: the whole verification block, every step run even after one fails,
# and a table at the end that says which.
#
# Stopping at the first failure would hide the rest, and the rest is usually
# the point: a health check is read, not just obeyed.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -t 1 ]; then
  BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; RESET=$'\033[0m'
else
  BOLD=; RED=; GREEN=; RESET=
fi

RESULTS=""
FAILED=0
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

step() {
  # step NAME COMMAND...   — runs the command, records pass or fail.
  local name="$1" started status
  shift
  printf '\n%s━━ %s%s\n' "$BOLD" "$name" "$RESET"
  started="$(date +%s)"
  "$@"
  status=$?
  record "$name" "$status" "$(( $(date +%s) - started ))"
}

record() {
  local name="$1" status="$2" secs="$3" mark
  if [ "$status" -eq 0 ]; then mark="${GREEN}pass${RESET}"; else mark="${RED}FAIL${RESET}"; FAILED=1; fi
  RESULTS="${RESULTS}$(printf '  %s  %-44s %4ss' "$mark" "$name" "$secs")
"
}

types_unchanged() {
  make types && git diff --exit-code -- web/src/api/openapi.json web/src/api/schema.d.ts
}

# The refusal is the guardrail this whole project is organised around, so it is
# asserted rather than eyeballed: exit 0 *and* the verifier labelled it a
# refusal. An answer, or a model error, fails the step.
refusal_holds() {
  local out="$SCRATCH/ask.txt" status
  uv run memoryos ask "what is our AWS billing setup" >"$out" 2>/dev/null
  status=$?
  grep -v $'\033' "$out" | grep -vE '^(No device|Loading|The Transformer|\[transformers\])'
  if [ "$status" -eq 0 ] && grep -q '· refusal' "$out"; then
    return 0
  fi
  printf '%sthe out-of-corpus question was not refused%s\n' "$RED" "$RESET"
  return 1
}

step "containers healthy"          docker compose up -d --wait --wait-timeout 180
step "alembic upgrade head"        uv run alembic upgrade head
step "alembic check"               uv run alembic check
step "ruff"                        uv run ruff check .
step "mypy"                        uv run mypy
step "pytest"                      uv run pytest
step "vitest"                      npm --prefix web run test -- --run
step "tsc -b"                      sh -c 'cd web && npx tsc -b'
step "vite build"                  npm --prefix web run build
step "make types (no drift)"       types_unchanged
step "memoryos doctor"             uv run memoryos doctor
step "memoryos verify-replay"      uv run memoryos verify-replay
step "memoryos graph verify"       uv run memoryos graph verify
step "evaluate vs baseline-hybrid" uv run memoryos evaluate --k 10 --compare var/baseline-hybrid.json
step "ask refuses out-of-corpus"   refusal_holds

printf '\n%s━━ summary%s\n%s' "$BOLD" "$RESET" "$RESULTS"
exit "$FAILED"
