#!/bin/sh
# 2026-11-03, second review: point 3 (unrelated-context control) and point 5 (DeepSeek reasoning).
# Usage: src/rev2_chain.sh <claude|gemini|deepseek>
cd "$(dirname "$0")/.."
PY=python3; [ -x venv/bin/python ] && PY=venv/bin/python
export TMPDIR="$PWD/tmp" TMUX_TMPDIR="$PWD/tmux" XDG_CACHE_HOME="$PWD/tmp/cache" TIKTOKEN_CACHE_DIR="$PWD/tiktoken_cache"
c=$1; log="logs/rev2_$c.log"; echo $$ > "logs/rev2_$c.pid"
step() { echo "== $(date '+%F %T') $*" >> "$log"; "$@" >> "$log" 2>&1; }
review() { m=$1; shift; i=0
  while [ $i -lt 6 ]; do i=$((i+1))
    out=$($PY -u src/run_p2.py review --full --model "$m" "$@" 2>&1 | tee -a "$log")
    echo "$out" | grep -qE "^\[$m\] 0 calls" && break; sleep 60; done; }
case $c in
  claude)   review claude --sample retest200.csv --retest-sample retest200.csv --retest-cond struct_filler --retest-cond struct_venue --retest-rep 3 --retest-only ;;
  gemini)   review gemini --sample sub80.csv --cond open_filler
            step $PY -u src/followup.py --model gemini --sample sub80.csv --cond open_filler ;;
  deepseek) review deepseek --sample sub60.csv --retest-sample sub60.csv --retest-cond open_author --retest-cond open_reviewer --retest-rep 4 --retest-only
            step $PY -u src/followup.py --model deepseek --sample sub60.csv --cond open_author --cond open_reviewer --rep 4
            step $PY -u src/reasoning_check.py ;;
esac
echo "REV2 DONE $c $(date '+%F %T')" >> "$log"
