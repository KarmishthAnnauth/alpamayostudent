#!/usr/bin/env bash
# Extract Bench2Drive-Base tarballs in place, deleting each archive once verified.
# Usage: nohup bash scripts/01c_b2d_extract.sh > logs/b2d_extract.log 2>&1 &
set -uo pipefail
ROOT=/bulk/datasets/bench2drive
OUT=$ROOT/extracted
JOBS=${JOBS:-6}
mkdir -p "$OUT"

one() {
    f=$1; clip=$(basename "$f" .tar.gz); dst=$OUT/$clip
    [ -f "$dst/.done" ] && { echo "SKIP $clip"; return 0; }
    rm -rf "$dst"
    expected=$(pigz -dc "$f" | tar t | grep -vc '/$')
    if ! pigz -dc "$f" | tar x -C "$OUT"; then echo "FAIL extract $clip"; return 1; fi
    [ -d "$dst" ] || { echo "FAIL no-dir $clip (archive root differs from name)"; return 1; }
    found=$(find "$dst" -type f | wc -l)
    if [ "$found" -gt 0 ] && [ "$found" = "$expected" ]; then
        touch "$dst/.done" && rm -f "$f" && echo "OK   $clip $found files"
    else
        echo "FAIL count $clip expected=$expected found=$found (tarball kept)"; return 1
    fi
}
export -f one; export OUT
ls "$ROOT"/*.tar.gz 2>/dev/null | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {}
echo "DONE $(date -Is): $(ls "$OUT" | wc -l) dirs, $(ls "$ROOT"/*.tar.gz 2>/dev/null | wc -l) tarballs left"
