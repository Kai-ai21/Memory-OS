#!/usr/bin/env bash
# Restore MEMO from a backup made by deploy/backup.sh (M12.3).
#
#   deploy/restore.sh backups/memos-<stamp>.dump backups/blobs-<stamp>.tar.gz
#
# Run on the server, from the repository root, with the same .env.production
# (and so the same MEMOS_MASTER_KEY) the backup was taken under: the dump holds
# data keys wrapped by that key, and nothing else can unwrap them.
#
# It replaces the database. It asks first.
set -euo pipefail

# One env file normally; MEMO_ENV_FILES may name several, space-separated,
# later ones winning (the local rehearsal layers test secrets over a dev .env).
COMPOSE=(docker compose)
for env_file in ${MEMO_ENV_FILES:-.env.production}; do
  COMPOSE+=(--env-file "$env_file")
done
COMPOSE+=(-f docker-compose.prod.yml)

dump="${1:?usage: deploy/restore.sh <memos-*.dump> <blobs-*.tar.gz>}"
blobs="${2:?usage: deploy/restore.sh <memos-*.dump> <blobs-*.tar.gz>}"
[ -f "$dump" ] || { echo "no such file: $dump" >&2; exit 1; }
[ -f "$blobs" ] || { echo "no such file: $blobs" >&2; exit 1; }

if [ "${MEMO_RESTORE_CONFIRM:-}" != "restore" ]; then
  [ -t 0 ] || { echo "refusing: confirm at a terminal, or set MEMO_RESTORE_CONFIRM=restore" >&2; exit 1; }
  echo "This REPLACES the memos database with $dump"
  echo "and adds the blobs in $blobs to the blob store."
  read -r -p "Type restore to continue: " answer
  [ "$answer" = "restore" ] || { echo "cancelled; nothing touched"; exit 1; }
fi

echo "==> stopping everything that writes"
"${COMPOSE[@]}" stop web api worker backup

echo "==> recreating the database"
"${COMPOSE[@]}" up -d --wait postgres
"${COMPOSE[@]}" exec -T postgres dropdb -U memos --force --if-exists memos
"${COMPOSE[@]}" exec -T postgres createdb -U memos memos
# Roles live in the cluster, not the database, so memos_app is still there and
# the dump's grants and row-level-security policies land on it.
"${COMPOSE[@]}" exec -T postgres pg_restore -U memos -d memos --exit-on-error < "$dump"

echo "==> restoring the blob store"
# Content-addressed: a file's name is its hash, so extracting over the store
# can only add files or rewrite identical ones. Nothing has to be deleted first.
"${COMPOSE[@]}" run --rm --no-deps -T --entrypoint tar \
  -v "$(cd "$(dirname "$blobs")" && pwd)":/restore:ro \
  --user root worker -xzf "/restore/$(basename "$blobs")" -C /data
"${COMPOSE[@]}" run --rm --no-deps -T --entrypoint chown --user root worker -R 10001:10001 /data/blobs

echo "==> starting, migrations first"
"${COMPOSE[@]}" up -d --wait
echo "restored from $dump"
