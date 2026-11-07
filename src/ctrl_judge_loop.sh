#!/bin/sh
# 2026-10-31: judge new open answers (raw judge, neutralize, neutral judge) and raw-judge all struct answers,
# every 30 min while any control review loop is alive, then one final pass.
cd "$(dirname "$0")/.."
PY=python3; [ -x venv/bin/python ] && PY=venv/bin/python
export TMPDIR="$PWD/tmp" TMUX_TMPDIR="$PWD/tmux" XDG_CACHE_HOME="$PWD/tmp/cache"
[ -d tiktoken_cache ] && export TIKTOKEN_CACHE_DIR="$PWD/tiktoken_cache"
echo $$ > logs/ctrl_judge_loop.pid
alive() { for f in logs/ctrl_loop_*.pid; do kill -0 "$(cat $f)" 2>/dev/null && return 0; done; return 1; }
while alive; do
  $PY -u src/run_p3_judge.py run --struct > logs/ctrl_judge_last.log 2>&1; tail -n 3 logs/ctrl_judge_last.log >> logs/ctrl_judge.log
  sleep 1800
done
$PY -u src/run_p3_judge.py run --struct > logs/ctrl_judge_last.log 2>&1; tail -n 40 logs/ctrl_judge_last.log >> logs/ctrl_judge.log
echo "JUDGE LOOP DONE $(date "+%F %T")" >> logs/ctrl_judge.log
