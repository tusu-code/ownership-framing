#!/bin/sh
# Disguise the remaining papers. Waits while the rewrite model is down (probe every 10 min, up to 24 h),
# and runs up to 3 passes so fallback chunks and failed renames get redone.
cd "$(dirname "$0")/.."
PY=python3; [ -x venv/bin/python ] && PY=venv/bin/python
export TMPDIR="$PWD/tmp" TMUX_TMPDIR="$PWD/tmux" PIP_CACHE_DIR="$PWD/tmp/pip" XDG_CACHE_HOME="$PWD/tmp/cache"
[ -d tiktoken_cache ] && export TIKTOKEN_CACHE_DIR="$PWD/tiktoken_cache"
echo $$ > logs/bg_disguise.pid
waited=0; passes=0
while [ $passes -lt 3 ]; do
  if $PY src/probe_model.py "$($PY -c 'import yaml;print(yaml.safe_load(open("config.yaml"))["disguise_model"])')" >> logs/disguise_probe.log 2>&1; then
    passes=$((passes + 1))
    $PY -u src/disguise.py run >> logs/disguise_sail_run$passes.log 2>&1
  else
    [ $waited -ge 144 ] && break
    waited=$((waited + 1)); sleep 600
  fi
done
echo "CHAIN DONE passes=$passes waited=${waited}x10min $(date '+%F %T')" >> logs/disguise_probe.log
