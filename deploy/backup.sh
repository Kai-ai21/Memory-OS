#!/bin/sh
# Scheduled backups (M12.3): the database and the blob store, together.
#
# Both, because neither restores MEMO alone. The database holds the log, the
# accounts, the decisions and the wrapped data keys; the blob store holds the
# original bytes every replay rebuilds from. A pg_dump without the blobs
# restores a corpus that cannot be replayed.
#
# Not in here, on purpose: MEMOS_MASTER_KEY. Every memory is encrypted under it,
# so these files are unreadable without it — which is the point of keeping it
# somewhere else. Back the key up separately (see deploy/README.md).
#
# Each file is written under a `.partial` name and renamed when complete, so a
# backup interrupted halfway never looks like a good one.
set -eu

INTERVAL="${BACKUP_INTERVAL_SECONDS:-86400}"
KEEP="${BACKUP_KEEP:-14}"
DIR=/backups

prune() {
  # Newest first; everything past the newest $KEEP goes.
  ls -1t "$DIR"/$1 2>/dev/null | tail -n +"$((KEEP + 1))" | while read -r old; do
    rm -f -- "$old"
  done
}

backup_once() {
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  pg_dump --format=custom --file="$DIR/memos-$stamp.dump.partial"
  mv "$DIR/memos-$stamp.dump.partial" "$DIR/memos-$stamp.dump"
  tar -C /data -czf "$DIR/blobs-$stamp.tar.gz.partial" blobs
  mv "$DIR/blobs-$stamp.tar.gz.partial" "$DIR/blobs-$stamp.tar.gz"
  prune 'memos-*.dump'
  prune 'blobs-*.tar.gz'
  echo "backup $stamp: $(du -h "$DIR/memos-$stamp.dump" | cut -f1) database, $(du -h "$DIR/blobs-$stamp.tar.gz" | cut -f1) blobs"
}

if [ "${1:-}" = "--once" ]; then
  backup_once
  exit 0
fi

while :; do
  backup_once || echo "backup failed; will retry in ${INTERVAL}s" >&2
  sleep "$INTERVAL"
done
