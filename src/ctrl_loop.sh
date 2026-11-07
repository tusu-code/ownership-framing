#!/bin/sh
# 2026-10-31 control asks (colleague, neutral). Usage: src/ctrl_loop.sh <model> <cond> [<cond> ...]
# Repeats passes until one reports 0 calls left, at most 12 passes (FatalGuard stops a pass on no balance etc.).
cd "$(dirname "$0")/.."
PY=python3; [ -x venv/bin/python ] && PY=venv/bin/python
export TMPDIR="$PWD/tmp" TMUX_TMPDIR="$PWD/tmux" XDG_CACHE_HOME="$PWD/tmp/cache"
[ -d tiktoken_cache ] && export TIKTOKEN_CACHE_DIR="$PWD/tiktoken_cache"
m=$1; shift
args=""; for c in "$@"; do args="$args --cond $c"; done
log="logs/ctrl_${m}.log"
echo $$ > "logs/ctrl_loop_$m.pid"
i=0
while [ $i -lt 12 ]; do
  i=$((i+1))
  out=$($PY -u src/run_p2.py review --full --model "$m" $args 2>&1 | tee -a "$log")
  echo "$out" | grep -qE "^\[$m\] 0 calls" && break
  sleep 120
done
echo "CTRL LOOP DONE $m pass=$i $(date "+%F %T")" >> "$log"
