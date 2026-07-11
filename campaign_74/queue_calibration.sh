#!/bin/bash
# Queue the 7-4 L4/L5 calibration behind the running hybrid_grid_gen timing experiment.
# Waits until no hybrid_grid_gen worker is left (3 consecutive checks, 5 min apart, so a
# between-problems gap is not mistaken for completion), settles 2 min, then launches the
# calibration with capped threads. The pgrep pattern uses the [n] bracket trick so this
# script (and any monitor greps) never self-match.
cd "$(dirname "$0")" || exit 1
{
    echo "[queue] $(date) waiting for hybrid_grid_ge[n].py workers to exit"
    MISS=0
    while [ "$MISS" -lt 3 ]; do
        if pgrep -f "hybrid_grid_ge[n].py" > /dev/null; then
            MISS=0
        else
            MISS=$((MISS + 1))
        fi
        sleep 300
    done
    echo "[queue] $(date) hybrid grid gen finished; settling 120s"
    sleep 120
    echo "[queue] $(date) launching calibration (RAYON=28, nice 5)"
    RAYON_NUM_THREADS=28 OMP_NUM_THREADS=1 CALIB_PARITY_THREADS=24 nice -n 5 \
        /home/psaegert/miniconda3/envs/flash-ansr/bin/python -u calibrate_74_l4l5.py \
        --out calib_out > calibration.log 2>&1
    echo "[queue] $(date) calibration exit code $?"
} >> calib_queue.log 2>&1
