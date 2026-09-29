#!/usr/bin/env bash
# tools/lock.sh <name> <cmd...>: run cmd holding a machine-wide lock, so parallel projects take turns on the Mac
# (e.g. `tools/lock.sh render scripts/render-parallel.sh 4 12 ...`: one full render at a time). Waits while another
# live process holds it; a holder that died is cleared. Lock dirs live in out/locks/ (gitignored).
set -euo pipefail
L="$(cd "$(dirname "$0")/.." && pwd)/out/locks/$1"; shift
mkdir -p "$(dirname "$L")"
waited=0
until mkdir "$L" 2>/dev/null; do
  pid=$(cat "$L/pid" 2>/dev/null || true)
  if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then rm -rf "$L"; continue; fi
  [ $((waited % 300)) -eq 0 ] && echo "lock $(basename "$L"): waiting on pid ${pid:-?} ($(cat "$L/what" 2>/dev/null || true))"
  sleep 15; waited=$((waited + 15))
done
echo $$ > "$L/pid"; echo "$PWD: $*" > "$L/what"
trap 'rm -rf "$L"' EXIT
"$@"
