#!/bin/sh
# 2026-11-03 revision experiments (reviewer M1a, M2, M4, M6), one chain per model so that a model's keys are never
# used by two processes at once. Each chain writes logs/ctrl_loop_rev_<chain>.pid, which keeps ctrl_judge_loop.sh
# alive until the chain ends. Steps of the four optional upgrades come last in each chain.
# Usage: src/rev_chain.sh <glm|gemini|claude|deepseek|minimax|tone>
cd "$(dirname "$0")/.."
PY=python3; [ -x venv/bin/python ] && PY=venv/bin/python
export TMPDIR="$PWD/tmp" TMUX_TMPDIR="$PWD/tmux" XDG_CACHE_HOME="$PWD/tmp/cache" TIKTOKEN_CACHE_DIR="$PWD/tiktoken_cache"
c=$1; log="logs/rev_$c.log"
# Chains that create answers keep the judge loop alive; minimax waits for that loop, so it must not.
case $c in minimax|tone) echo $$ > "logs/rev_$c.pid" ;; *) echo $$ > "logs/ctrl_loop_rev_$c.pid" ;; esac
A4="--cond open_author --cond open_colleague --cond open_venue --cond open_reviewer"
AR="--cond open_author --cond open_reviewer"
FO="--cond fixedopen_author --cond fixedopen_reviewer"
step() { echo "== $(date '+%F %T') $*" >> "$log"; "$@" >> "$log" 2>&1; }
review() {  # review until no calls remain (at most 6 passes)
  m=$1; shift; i=0
  while [ $i -lt 6 ]; do i=$((i+1))
    out=$($PY -u src/run_p2.py review --full --model "$m" "$@" 2>&1 | tee -a "$log")
    echo "$out" | grep -qE "^\[$m\] 0 calls" && break; sleep 60; done; }
case $c in
  glm)    step $PY -u src/followup.py --model glm --sample sub80.csv $A4
          review glm --sample sub80.csv $FO
          step $PY -u src/recognize.py --model glm ;;
  gemini) while kill -0 "$(cat logs/ctrl_loop_gemini.pid)" 2>/dev/null; do sleep 120; done
          step $PY -u src/followup.py --model gemini --sample sub80.csv $A4
          review gemini --sample sub80.csv $FO
          step $PY -u src/recognize.py --model gemini ;;
  claude) head -n 41 data/sub80.csv > data/sub40.csv
          step $PY -u src/followup.py --model claude --sample sub40.csv $AR
          step $PY -u src/followup.py --model claude --sample sub80.csv $AR
          review claude --sample sub60.csv $FO
          step $PY -u src/recognize.py --model claude ;;
  deepseek) step $PY -u src/recognize.py --model deepseek
          review deepseek --sample sub80.csv --retest-sample sub80.csv --retest-cond open_author --retest-cond open_reviewer --retest-rep 3 --retest-only
          step $PY -u src/followup.py --model deepseek --sample sub80.csv $AR --rep 3 ;;
  tone)   step $PY -u src/tone_swap.py rewrite; step $PY -u src/tone_swap.py judge ;;
  minimax) step $PY -u src/judge3.py neutral
          while [ -e logs/ctrl_judge_loop.pid ] && kill -0 "$(cat logs/ctrl_judge_loop.pid)" 2>/dev/null; do sleep 300; done
          step $PY -u src/judge3.py neutral; step $PY -u src/judge3.py raw ;;
esac
echo "REV CHAIN DONE $c $(date '+%F %T')" >> "$log"
