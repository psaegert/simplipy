#!/bin/bash
# Every ~5h: pull the growing 7-4 ruleset from solomon and benchmark it on valkyrie
# (pinned core, nice 10, paired vs dev_7-3). Exits after the round that sees the
# mine's 'finished' stamp.
cd "$(dirname "$0")" || exit 1
export SSH_AUTH_SOCK=$HOME/.ssh/agent.sock
{
    while true; do
        /home/psaegert/miniconda3/envs/flash-ansr/bin/python -u pull_and_bench_74.py --workdir bench74
        /home/psaegert/miniconda3/envs/flash-ansr/bin/python -c "
import json, sys
try:
    rows = [json.loads(l) for l in open('bench74/bench_timeline.jsonl')]
    sys.exit(0 if rows and rows[-1].get('mine_done') else 1)
except FileNotFoundError:
    sys.exit(1)" && { echo "[loop] mine finished; final benchmark recorded; exiting"; break; }
        sleep 18000
    done
} >> bench_loop.log 2>&1
