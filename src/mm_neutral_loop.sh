#!/bin/sh
# 2026-11-03 19:35: replaces the neutral part of rev_chain.sh minimax (stopped by mistake). Runs the MiniMax judge on
# neutralized answers now, then once more after the main judge loop has finished neutralizing the last new answers.
cd "$(dirname "$0")/.."
PY=python3; [ -x venv/bin/python ] && PY=venv/bin/python
export TMPDIR="$PWD/tmp" TMUX_TMPDIR="$PWD/tmux" XDG_CACHE_HOME="$PWD/tmp/cache" TIKTOKEN_CACHE_DIR="$PWD/tiktoken_cache"
log=logs/rev_minimax_neutral.log; echo $$ > logs/rev_minimax_neutral.pid
echo "== $(date '+%F %T') pass 1" >> $log; $PY -u src/judge3.py neutral >> $log 2>&1
while [ -e logs/ctrl_judge_loop.pid ] && kill -0 "$(cat logs/ctrl_judge_loop.pid)" 2>/dev/null; do sleep 300; done
echo "== $(date '+%F %T') pass 2" >> $log; $PY -u src/judge3.py neutral >> $log 2>&1
echo "MM NEUTRAL DONE $(date '+%F %T')" >> $log
