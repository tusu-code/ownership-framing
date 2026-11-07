#!/bin/sh
# Preregistered sample run for one model: rerun until every cell is valid (at most 8 passes, 7-day limit in the
# plan), then the recognition check.   usage: mitig/src/mitig_loop.sh claude|gemini
cd "$(dirname "$0")/../.."
export TMPDIR=$PWD/tmp TIKTOKEN_CACHE_DIR=$PWD/tiktoken_cache
m=$1
for i in 1 2 3 4 5 6 7 8; do
  echo "== $m pass $i $(date)"
  nice -n 19 venv/bin/python -u mitig/src/run_mitig.py run --model $m --role sample
  venv/bin/python -u mitig/src/run_mitig.py status | grep "^$m:"
  venv/bin/python -u mitig/src/run_mitig.py status | grep "^$m:" | grep -q "'sample': 2880" && break
  sleep 600
done
nice -n 19 venv/bin/python -u mitig/src/run_mitig.py recognize --model $m --role sample
echo "== $m done $(date)"
