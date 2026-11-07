#!/bin/sh
# Re-run the full-study review for one model until the disguise chain has ended and a pass adds nothing.
cd "$(dirname "$0")/.."
PY=python3; [ -x venv/bin/python ] && PY=venv/bin/python
export TMPDIR="$PWD/tmp" TMUX_TMPDIR="$PWD/tmux" PIP_CACHE_DIR="$PWD/tmp/pip" XDG_CACHE_HOME="$PWD/tmp/cache"
[ -d tiktoken_cache ] && export TIKTOKEN_CACHE_DIR="$PWD/tiktoken_cache"
m=$1
echo $$ > "logs/bg_loop_$m.pid"
while true; do
  $PY -u src/run_p2.py review --full --model "$m" >> "logs/p3_${m}_sail.log" 2>&1
  if ! kill -0 "$(cat logs/bg_disguise.pid)" 2>/dev/null; then
    left=$($PY -u src/run_p2.py review --full --model "$m" 2>&1 | tee -a "logs/p3_${m}_sail.log" | grep -oE "^\[$m\] [0-9]+ calls" | grep -oE "[0-9]+ calls")
    [ "$left" = "0 calls" ] && break
  fi
  sleep 300
done
echo "LOOP DONE $m $(date '+%F %T')" >> "logs/p3_${m}_sail.log"
