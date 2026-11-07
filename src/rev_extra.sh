#!/bin/sh
# 2026-11-03 follow-on to rev_chain.sh (user approved both extras): Gemini instruction ablation on sub60, and the
# second-turn self-score for every fixed-instruction and ablation free-text answer. Each part waits for the chain
# that uses the same model's keys. Writes logs/ctrl_loop_rev_extra_<part>.pid so the judge loop stays alive.
# Usage: src/rev_extra.sh <claude|glm|gemini>
cd "$(dirname "$0")/.."
PY=python3; [ -x venv/bin/python ] && PY=venv/bin/python
export TMPDIR="$PWD/tmp" TMUX_TMPDIR="$PWD/tmux" XDG_CACHE_HOME="$PWD/tmp/cache" TIKTOKEN_CACHE_DIR="$PWD/tiktoken_cache"
c=$1; log="logs/rev_extra_$c.log"; echo $$ > "logs/ctrl_loop_rev_extra_$c.pid"
FO="--cond fixedopen_author --cond fixedopen_reviewer"
AB="--cond ablorder_author --cond ablorder_reviewer --cond ablrole_author --cond ablrole_reviewer --cond ablmerits_author --cond ablmerits_reviewer"
step() { echo "== $(date '+%F %T') $*" >> "$log"; "$@" >> "$log" 2>&1; }
waitfor() { while kill -0 "$(cat "$1")" 2>/dev/null; do sleep 120; done; }
review() { m=$1; shift; i=0
  while [ $i -lt 6 ]; do i=$((i+1))
    out=$($PY -u src/run_p2.py review --full --model "$m" "$@" 2>&1 | tee -a "$log")
    echo "$out" | grep -qE "^\[$m\] 0 calls" && break; sleep 60; done; }
case $c in
  claude) step $PY -u src/followup.py --model claude --sample sub60.csv $FO ;;
  glm)    waitfor logs/ctrl_loop_rev_glm.pid
          step $PY -u src/followup.py --model glm --sample sub80.csv $FO ;;
  gemini) waitfor logs/ctrl_loop_rev_gemini.pid
          review gemini --sample sub60.csv $AB
          step $PY -u src/followup.py --model gemini --sample sub60.csv $AB
          step $PY -u src/followup.py --model gemini --sample sub80.csv $FO ;;
esac
echo "REV EXTRA DONE $c $(date '+%F %T')" >> "$log"
