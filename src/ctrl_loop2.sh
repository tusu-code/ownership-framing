#!/bin/sh
# 2026-11-01: like ctrl_loop.sh, but passes extra run_p2.py arguments verbatim (e.g. the same-batch retest).
# Usage: src/ctrl_loop2.sh <model> <run_p2.py review args...>
cd "$(dirname "$0")/.."
PY=python3; [ -x venv/bin/python ] && PY=venv/bin/python
export TMPDIR="$PWD/tmp" TMUX_TMPDIR="$PWD/tmux" XDG_CACHE_HOME="$PWD/tmp/cache"
[ -d tiktoken_cache ] && export TIKTOKEN_CACHE_DIR="$PWD/tiktoken_cache"
m=$1; shift
log="logs/ctrl_${m}.log"
echo $$ > "logs/ctrl_loop_$m.pid"
i=0
while [ $i -lt 12 ]; do
  i=$((i+1))
  out=$($PY -u src/run_p2.py review --full --model "$m" "$@" 2>&1 | tee -a "$log")
  echo "$out" | grep -qE "^\[$m\] 0 calls" && break
  sleep 120
done
echo "CTRL LOOP DONE $m pass=$i $(date "+%F %T")" >> "$log"
