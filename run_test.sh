#!/bin/bash

set -euo pipefail

TASK="ResDexHand"
DATA_INDICES="[75a7a@1]"
IMITATION_EXPERIMENT_BASE="write_on_paper_imitation_first"
RESIDUAL_EXPERIMENT_BASE="write_on_paper_residual_first"
TEST_EXPERIMENT_BASE="write_on_paper_test_first"

LH_BASE="assets/imitator_lh_inspire.pth"
DEXHAND="inspire"
SIDE="RH"
HEADLESS="false"
NUM_ENVS=6
TEST="true"
RANDOM_INIT="true"
LR="2e-4"
ACT_MA="0.4"
IMITATION_EPOCH=""

# Usage:
#   ./run_test.sh                      # uses residual checkpoints built from best imitation checkpoints
#   ./run_test.sh 10 30 50             # tests these suffixes
#   ./run_test.sh --epoch 200 10 30 50 # tests residual checkpoints trained with imitation epoch 200
PAIR_SUFFIXES=()
while (( $# > 0 )); do
    case "$1" in
        --epoch)
            if (( $# < 2 )); then
                echo "[error] --epoch requires a value"
                exit 1
            fi
            IMITATION_EPOCH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--epoch N] [pair suffixes...]"
            exit 0
            ;;
        *)
            PAIR_SUFFIXES+=("$1")
            shift
            ;;
    esac
done

if [[ -n "${IMITATION_EPOCH}" ]] && ! [[ "${IMITATION_EPOCH}" =~ ^[0-9]+$ ]] ; then
    echo "[error] invalid epoch: ${IMITATION_EPOCH}"
    exit 1
fi

if (( ${#PAIR_SUFFIXES[@]} == 0 )); then
    PAIR_SUFFIXES=(10 30 50)
fi

resolve_imitation_checkpoint() {
    local suffix="$1"
    local epoch="${2:-}"
    local imitation_name="${IMITATION_EXPERIMENT_BASE}_${suffix}"

    local matches=()
    shopt -s nullglob
    if [[ -n "${epoch}" ]]; then
        matches=(runs/${imitation_name}__*/nn/last_${imitation_name}_ep_${epoch}_*.pth)
    else
        matches=(runs/${imitation_name}__*/nn/${imitation_name}.pth)
    fi
    shopt -u nullglob

    if (( ${#matches[@]} == 0 )); then
        return 1
    fi

    ls -t "${matches[@]}" 2>/dev/null | head -n 1
}

resolve_residual_checkpoint() {
    local suffix="$1"
    local epoch="${2:-}"
    local residual_name

    if [[ -n "${epoch}" ]]; then
        residual_name="${RESIDUAL_EXPERIMENT_BASE}_${suffix}_ep_${epoch}"
    else
        residual_name="${RESIDUAL_EXPERIMENT_BASE}_${suffix}_best"
    fi

    local matches=()
    shopt -s nullglob
    matches=(runs/${residual_name}__*/nn/${residual_name}.pth)
    shopt -u nullglob

    if (( ${#matches[@]} == 0 )); then
        return 1
    fi

    ls -t "${matches[@]}" 2>/dev/null | head -n 1
}

for suffix in "${PAIR_SUFFIXES[@]}"; do
    if ! [[ "${suffix}" =~ ^[0-9]+$ ]] || (( suffix < 1 )); then
        echo "[skip] Invalid pair suffix: ${suffix}"
        continue
    fi

    if [[ -n "${IMITATION_EPOCH}" ]]; then
        residual_name="${RESIDUAL_EXPERIMENT_BASE}_${suffix}_ep_${IMITATION_EPOCH}"
        test_experiment="${TEST_EXPERIMENT_BASE}_${suffix}_ep_${IMITATION_EPOCH}"
    else
        residual_name="${RESIDUAL_EXPERIMENT_BASE}_${suffix}_best"
        test_experiment="${TEST_EXPERIMENT_BASE}_${suffix}_best"
    fi

    if ! RH_BASE=$(resolve_imitation_checkpoint "${suffix}" "${IMITATION_EPOCH}"); then
        echo "[skip] Missing imitation checkpoint for suffix=${suffix}"
        if [[ -n "${IMITATION_EPOCH}" ]]; then
            echo "       expected pattern: runs/${IMITATION_EXPERIMENT_BASE}_${suffix}__*/nn/last_${IMITATION_EXPERIMENT_BASE}_${suffix}_ep_${IMITATION_EPOCH}_*.pth"
        else
            echo "       expected pattern: runs/${IMITATION_EXPERIMENT_BASE}_${suffix}__*/nn/${IMITATION_EXPERIMENT_BASE}_${suffix}.pth"
        fi
        continue
    fi

    if ! CHECKPOINT=$(resolve_residual_checkpoint "${suffix}" "${IMITATION_EPOCH}"); then
        echo "[skip] Missing residual checkpoint for suffix=${suffix}"
        echo "       expected pattern: runs/${residual_name}__*/nn/${residual_name}.pth"
        continue
    fi

    echo "[run] pair=${suffix} dataIndices=${DATA_INDICES} rh_base=${RH_BASE} checkpoint=${CHECKPOINT}"

    python main/rl/train.py \
        task=${TASK} \
        dexhand=${DEXHAND} \
        side=${SIDE} \
        headless=${HEADLESS} \
        num_envs=${NUM_ENVS} \
        learning_rate=${LR} \
        test=${TEST} \
        randomStateInit=${RANDOM_INIT} \
        dataIndices=${DATA_INDICES} \
        actionsMovingAverage=${ACT_MA} \
        rh_base_model_checkpoint=${RH_BASE} \
        lh_base_model_checkpoint=${LH_BASE} \
        checkpoint=${CHECKPOINT} \
        experiment=${test_experiment} \
        capture_video=True
done