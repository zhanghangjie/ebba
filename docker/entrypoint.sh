#!/bin/sh
# EBBA verification entrypoint.
#
#   verify [--fast] [--only NAME] [--timeout SEC] [--device cpu|cuda]
#   test
#   retrain --attack badnets|blend
#   report
#   env
#   shell
#   train-full --all --resume auto     # ResNet-18 / full CIFAR-10 / 200 epochs
#   verify-full [--device cpu|cuda]    # full_eval + notebook 07
#   notebook-full                      # execute only notebook 07
#
# `verify` exits non-zero only on execution failure. CONTRADICTED verdicts are
# expected results, not errors.

set -eu

cd /app

DEVICE="${EBBA_DEVICE:-cpu}"
PAPERMILL_ARGS=""

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

if [ "$#" -eq 0 ]; then
    set -- verify
fi

CMD="$1"
shift

case "$CMD" in
    verify)
        # The proxy path is deliberately isolated from the full profile.
        export EBBA_PROFILE=proxy
        export EBBA_ARTIFACTS=/app/artifacts
        export EBBA_REPORTS=/app/reports
        while [ "$#" -gt 0 ]; do
            case "$1" in
                --device) DEVICE="$2"; shift 2 ;;
                --device=*) DEVICE="${1#*=}"; shift ;;
                --fast) PAPERMILL_ARGS="$PAPERMILL_ARGS --fast"; shift ;;
                --timeout) PAPERMILL_ARGS="$PAPERMILL_ARGS --timeout $2"; shift 2 ;;
                --only)
                    shift
                    names=""
                    while [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; do
                        names="$names $1"; shift
                    done
                    if [ -z "$names" ]; then
                        echo "verify: --only needs at least one notebook name" >&2
                        exit 2
                    fi
                    PAPERMILL_ARGS="$PAPERMILL_ARGS --only$names"
                    ;;
                *) echo "unknown option for verify: $1" >&2; exit 2 ;;
            esac
        done
        echo "== environment =="
        python scripts/check_env.py
        echo "== notebooks (device=$DEVICE) =="
        # shellcheck disable=SC2086
        python scripts/run_papermill.py --profile proxy --device "$DEVICE" $PAPERMILL_ARGS
        echo "== report =="
        python scripts/make_report.py
        echo
        echo "Report: /app/reports/report.html"
        echo "Verdicts: /app/reports/verdicts.json"
        ;;
    verify-full)
        export EBBA_PROFILE=full
        export EBBA_ARTIFACTS=/app/artifacts-full
        export EBBA_REPORTS=/app/reports-full
        # Append the container device only when the caller did not choose one.
        case " $* " in
            *" --device "*) ;;
            *) set -- --device "$DEVICE" "$@" ;;
        esac
        # Echo the device full_eval.py actually receives, not the env default.
        FULL_DEVICE="$DEVICE"
        prev=""
        for a in "$@"; do
            if [ "$prev" = "--device" ]; then FULL_DEVICE="$a"; fi
            case "$a" in --device=*) FULL_DEVICE="${a#*=}" ;; esac
            prev="$a"
        done
        echo "== full evaluation (device=$FULL_DEVICE) =="
        python scripts/full_eval.py "$@"
        echo "== full notebook =="
        python scripts/run_papermill.py --profile full --only 07_full_comparison
        echo
        echo "Full report: /app/reports-full/full_report.html"
        echo "Full metrics: /app/reports-full/full_metrics.json"
        ;;
    train-full)
        export EBBA_PROFILE=full
        export EBBA_ARTIFACTS="${EBBA_ARTIFACTS:-/app/artifacts-full}"
        export EBBA_REPORTS="${EBBA_REPORTS:-/app/reports-full}"
        export EBBA_WORK="${EBBA_WORK:-/app/checkpoints-full}"
        exec python scripts/train_full.py "$@"
        ;;
    notebook-full)
        export EBBA_PROFILE=full
        export EBBA_ARTIFACTS="${EBBA_ARTIFACTS:-/app/artifacts-full}"
        export EBBA_REPORTS="${EBBA_REPORTS:-/app/reports-full}"
        exec python scripts/run_papermill.py --profile full "$@"
        ;;
    test)
        export EBBA_PROFILE=proxy
        exec python -m pytest -q tests "$@"
        ;;
    retrain)
        export EBBA_PROFILE=proxy
        exec python scripts/train.py "$@"
        ;;
    report)
        export EBBA_PROFILE=proxy
        exec python scripts/make_report.py "$@"
        ;;
    env)
        exec python scripts/check_env.py "$@"
        ;;
    shell)
        exec /bin/bash
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "unknown command: $CMD" >&2
        usage >&2
        exit 2
        ;;
esac
