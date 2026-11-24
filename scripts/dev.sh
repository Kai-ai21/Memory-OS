#!/usr/bin/env bash
# MEMO in one terminal.
#
#   scripts/dev.sh          (make dev)    containers, migrations, API + worker + UI
#   scripts/dev.sh stop     (make stop)   stop all of it; volumes are kept
#   scripts/dev.sh reset    (make reset)  destroy the volumes and rebuild the corpus
#
# A plain script rather than honcho or foreman: three processes do not justify a
# dependency, and the part that is actually hard here (every child, grandchild and
# reloader subprocess gone on Ctrl-C, with nothing left holding a port) is a few
# lines of `trap` either way.
#
# Written for the bash macOS ships (3.2): no `wait -n`, no associative arrays.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_PORT=8000
WEB_PORT=5173
WEB_URL="http://localhost:${WEB_PORT}"
API_URL="http://localhost:${API_PORT}"

if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'; RESET=$'\033[0m'
  C_API=$'\033[36m'; C_WORKER=$'\033[35m'; C_WEB=$'\033[33m'
else
  BOLD=; DIM=; RED=; GREEN=; RESET=; C_API=; C_WORKER=; C_WEB=
fi

say()  { printf '%s==>%s %s\n' "$BOLD" "$RESET" "$*"; }
warn() { printf '%swarning:%s %s\n' "$RED" "$RESET" "$*" >&2; }
die()  { printf '\n%serror:%s %s\n\n' "$RED" "$RESET" "$*" >&2; exit 1; }
now()  { date +%s; }

# --------------------------------------------------------------------------
# Preflight: every check here exists to replace a stack trace with a sentence.
# --------------------------------------------------------------------------

require_tools() {
  local missing="" tool
  for tool in docker uv npm lsof curl; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
  done
  [ -z "$missing" ] || die "missing on PATH:$missing. See Prerequisites in README.md."
}

require_docker() {
  docker info >/dev/null 2>&1 || die "Docker is not running.
  Start Docker Desktop (macOS: open -a Docker), wait for it to settle, and run this again."
}

# The PID listening on a TCP port, or nothing.
port_owner() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1; }

# Refuse to start rather than let uvicorn or Vite die with a bind error halfway
# through a screen of log output.
require_free_ports() {
  local busy=0 port label pid cmd
  for port in "$API_PORT" "$WEB_PORT"; do
    pid="$(port_owner "$port")"
    [ -n "$pid" ] || continue
    if [ "$port" = "$API_PORT" ]; then label="API"; else label="web UI"; fi
    cmd="$(ps -o command= -p "$pid" 2>/dev/null | cut -c1-100)"
    printf '\n%sport %s (%s) is already in use%s by PID %s:\n    %s\n' \
      "$RED" "$port" "$label" "$RESET" "$pid" "$cmd" >&2
    case "$cmd" in
      *"$ROOT"*) printf '  That is an earlier MEMO still running. Free it with:\n    make stop\n' >&2 ;;
      *)         printf '  Free it with:\n    kill %s\n  or, for whatever holds the port:\n    lsof -ti tcp:%s | xargs kill\n' "$pid" "$port" >&2 ;;
    esac
    busy=1
  done
  [ "$busy" = 0 ] || { echo >&2; exit 1; }
}

# Nothing starts without MEMOS_MASTER_KEY (M11.2), and the application's own
# message explains how to make one, so let it say so once, here, rather than
# three times in three interleaved tracebacks.
require_master_key() {
  local out
  if ! out="$(uv run --quiet python -c '
import sys
from memoryos.config import get_settings
from memoryos.crypto import MissingMasterKey, load_master_key
try:
    load_master_key(settings=get_settings())
except MissingMasterKey as exc:
    print(exc)
    sys.exit(1)
' 2>&1)"; then
    [ -f .env ] || out="There is no .env. Start from the template:

    cp .env.example .env

then set MEMOS_MASTER_KEY in it.

$out"
    die "$out"
  fi
}

require_dependencies() {
  if [ ! -d .venv ]; then
    say "installing Python dependencies (first run)"
    uv sync --frozen --extra dev || die "uv sync failed"
  fi
  if [ ! -d web/node_modules ]; then
    say "installing web dependencies (first run)"
    (cd web && npm ci) || die "npm ci failed"
  fi
}

containers_up() {
  # `--wait` blocks on the healthchecks in docker-compose.yml (pg_isready, and a
  # real bolt query for Neo4j), not on a guessed number of seconds.
  say "starting Postgres and Neo4j, waiting until both report healthy"
  docker compose up -d --wait --wait-timeout 180 \
    || die "the containers did not become healthy. Look at:
    docker compose ps
    docker compose logs --tail 50
  A 'port is already allocated' error means something else holds 5433, 7474 or 7687."
}

migrate() {
  say "applying pending migrations"
  uv run alembic upgrade head || die "alembic upgrade head failed (output above)"
}

# --------------------------------------------------------------------------
# Process supervision
# --------------------------------------------------------------------------

PIDS=""
NAMES=""
STOPPING=""

# Start `$@` in the background with every line prefixed by its name.
#
# Process substitution rather than a pipe, so that `$!` is the command itself
# and not a subshell around a pipeline: a pipeline stays alive for as long as
# anything holds its write end, which is how a dead `uv run` with a surviving
# Python child went unnoticed.
#
# The prefixer ignores INT and TERM so it outlives the process it is reading:
# it ends at EOF, which means a server's shutdown messages are still printed
# instead of being cut off by a pipe that closed first (and a SIGPIPE landing in
# the middle of that shutdown).
start() {
  local name="$1" color="$2" label
  shift 2
  label="$(printf '%-6s' "$name")"
  PYTHONUNBUFFERED=1 FORCE_COLOR=1 "$@" \
    > >(trap '' INT TERM; exec awk -v p="${color}${label}|${RESET} " '{ print p $0; fflush() }') 2>&1 &
  PIDS="$PIDS $!"
  NAMES="$NAMES $name"
}

# Processes from this checkout whose parent has already gone (reparented to
# PID 1). `uv run` does not exec, so if it dies first its Python child is left
# running with nobody above it; this is what finds that child. Orphans only, so
# a `memoryos worker` somebody started by hand in another terminal is left alone.
orphans() {
  local pid
  for pid in $(our_processes); do
    [ "$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" = "1" ] && echo "$pid"
  done
}

# Every process below $1, depth first. Gathered *before* anything is signalled,
# because once a parent exits its children are reparented and `pgrep -P` stops
# finding them — which is exactly how a uvicorn reloader orphans its worker.
descendants() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do
    descendants "$child"
    echo "$child"
  done
}

alive_among() {
  local pid alive=""
  for pid in "$@"; do
    kill -0 "$pid" 2>/dev/null && alive="$alive $pid"
  done
  echo "$alive"
}

terminate() {
  # $@: pids. TERM, up to ten seconds of grace, then KILL whatever is left.
  local i alive
  [ $# -gt 0 ] || return 0
  kill -TERM "$@" 2>/dev/null
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    alive="$(alive_among "$@")"
    [ -n "$alive" ] || return 0
    sleep 0.5
  done
  warn "still running after 10s, killing:$alive"
  # shellcheck disable=SC2086
  kill -KILL $alive 2>/dev/null
}

cleanup() {
  local status="${1:-0}" all="" pid
  [ -z "$STOPPING" ] || return 0
  STOPPING=1
  trap '' INT TERM
  trap - EXIT
  echo
  say "stopping api, worker and web"
  for pid in $PIDS $(orphans); do
    all="$all $(descendants "$pid") $pid"
  done
  # shellcheck disable=SC2086
  terminate $all
  wait 2>/dev/null

  local port leftover=""
  for port in "$API_PORT" "$WEB_PORT"; do
    [ -z "$(port_owner "$port")" ] || leftover="$leftover $port"
  done
  if [ -n "$leftover" ]; then
    warn "something still holds port(s)$leftover — make stop will clear it"
  else
    say "stopped; ports $API_PORT and $WEB_PORT are free"
  fi
  printf '%s    Postgres and Neo4j are still running, so the next make dev starts fast.\n    make stop brings them down too.%s\n' "$DIM" "$RESET"
  exit "$status"
}

dev() {
  local started
  started="$(now)"
  require_tools
  require_docker
  require_free_ports
  require_dependencies
  require_master_key
  containers_up
  migrate

  # Ctrl-C is how this is meant to be stopped, so it exits 0 rather than make
  # reporting "Error 130" for a normal shutdown.
  trap 'cleanup 0' INT
  trap 'cleanup 143' TERM
  trap 'cleanup $?' EXIT

  say "starting api (:${API_PORT}), worker, and web (:${WEB_PORT})"
  # The UI calls the API across an origin, which is the reason
  # MEMOS_CORS_ORIGINS exists. --reload-dir keeps the watcher off var/ (model
  # weights, blobs) and web/node_modules.
  start api "$C_API" env MEMOS_CORS_ORIGINS="[\"${WEB_URL}\"]" \
    uv run uvicorn "memoryos.api.app:create_app" --factory \
      --port "$API_PORT" --reload --reload-dir src
  start worker "$C_WORKER" uv run memoryos worker
  start web "$C_WEB" npm --prefix web run dev

  # Supervise: announce readiness once, and if any one process dies, take the
  # others down with it rather than leave a UI talking to nothing.
  local ready="" i name pid
  while :; do
    i=0
    for pid in $PIDS; do
      i=$((i + 1))
      if ! kill -0 "$pid" 2>/dev/null; then
        name="$(echo $NAMES | cut -d' ' -f"$i")"
        echo
        warn "$name exited on its own — stopping the rest (its output is above)"
        cleanup 1
      fi
    done
    if [ -z "$ready" ] \
      && curl -fsS -o /dev/null --max-time 2 "${API_URL}/health/ready" 2>/dev/null \
      && curl -fsS -o /dev/null --max-time 2 "${WEB_URL}/" 2>/dev/null; then
      ready=1
      printf '\n%s%s  MEMO is up in %ss  →  %s%s   %s(API %s · Ctrl-C stops everything)%s\n\n' \
        "$BOLD" "$GREEN" "$(( $(now) - started ))" "$WEB_URL" "$RESET" "$DIM" "$API_URL" "$RESET"
    fi
    # Backgrounded so the wait is interruptible: a trapped signal ends `wait`
    # at once, where a foreground `sleep` would finish first.
    sleep 1 &
    wait $! 2>/dev/null
  done
}

# --------------------------------------------------------------------------
# stop
# --------------------------------------------------------------------------

# Only processes started from *this* checkout: the patterns are anchored on its
# absolute path, so another project's Vite or uvicorn is never touched.
our_processes() {
  {
    pgrep -f "${ROOT}/\.venv/bin/uvicorn memoryos\.api\.app" 2>/dev/null
    pgrep -f "${ROOT}/\.venv/bin/memoryos worker" 2>/dev/null
    pgrep -f "${ROOT}/web/node_modules/.*vite" 2>/dev/null
  } | sort -u
}

stop() {
  local pids all="" pid
  pids="$(our_processes)"
  if [ -n "$pids" ]; then
    say "stopping MEMO processes: $(echo $pids)"
    for pid in $pids; do
      all="$all $(descendants "$pid") $pid"
    done
    # shellcheck disable=SC2086
    terminate $all
  fi
  if docker info >/dev/null 2>&1; then
    say "stopping containers (volumes are kept)"
    docker compose down
  else
    say "Docker is not running, so there are no containers to stop"
  fi
}

# --------------------------------------------------------------------------
# reset
# --------------------------------------------------------------------------

reset() {
  require_tools
  require_docker
  [ -t 0 ] || die "make reset destroys data and has to be confirmed at a terminal."
  if [ -n "$(port_owner "$API_PORT")" ] || [ -n "$(our_processes)" ]; then
    die "MEMO is running. Stop it first (Ctrl-C in its terminal, or make stop)."
  fi
  require_dependencies
  require_master_key

  cat <<EOF

${RED}${BOLD}make reset destroys the Postgres and Neo4j volumes.${RESET} That is:

  - every account and session, every memory and chunk, every chat
  - the tables no rebuild can reproduce: decisions, outcomes, assumption
    verdicts and query judgements
  - the graph

Kept: .env (and so MEMOS_MASTER_KEY), the blob store in var/blobs, and the
model weights in var/hf.

Afterwards it migrates, asks you to create an account, ingests this repository
as the source "self" and embeds it. Expect several minutes.

EOF
  local answer
  read -r -p "Type ${BOLD}reset${RESET} to destroy the volumes, anything else to cancel: " answer
  [ "$answer" = "reset" ] || { say "cancelled; nothing was touched"; exit 1; }

  say "destroying the volumes"
  docker compose down -v || die "docker compose down -v failed"
  containers_up
  migrate

  say "every row belongs to an account (M11.1), so the corpus needs one before it can be ingested"
  local email=""
  while [ -z "$email" ]; do read -r -p "Email for the account: " email; done
  uv run memoryos auth create-user --email "$email" || die "account creation failed"

  say "re-ingesting this repository as the source 'self'"
  uv run memoryos source add --kind filesystem --name self --root . || die "source add failed"
  uv run memoryos sync --source self --full || die "sync failed"

  # Only the two jobs that make the corpus searchable. Entity extraction is a
  # model call per memory against a free tier that serves a few a minute; the
  # worker that make dev starts picks those jobs up afterwards.
  say "normalizing and embedding"
  uv run memoryos worker --drain --only normalize_memory,embed_memory || die "embedding failed"
  uv run memoryos stats
  say "done. make dev to start it; sign in as $email."
}

case "${1:-dev}" in
  dev)   dev ;;
  stop)  stop ;;
  reset) reset ;;
  *)     die "usage: scripts/dev.sh [dev|stop|reset]" ;;
esac
