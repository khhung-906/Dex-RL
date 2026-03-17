#!/bin/bash

set -euo pipefail

TASK="ResDexHand"
# DATA_INDICES="[1be0e@1,1dc51@1,419a3@5,5fde1@3,7fa33@2,847ba@2,8adcc@2,9f2cb@1,a5345@1,ab61c@1,ebd70@5,ed9bc@1,f5a3d@10,ff220@3]"
# DATA_INDICES="[20034@0,67cb3@1,6c1ba@3,86fb0@1,99369@0,f3fe2@0]"
DATA_INDICES="[75a7a@1]"
EXPERIMENT_BASE="write_on_paper_residual_collected_first"
IMITATION_EXPERIMENT_BASE="write_on_paper_imitation_collected_first"

DEXHAND="inspire"
SIDE="RH"
HEADLESS="true"
NUM_ENVS=4096
LR="2e-4"
TEST="false"
RANDOM_INIT="true"
EARLY_STOP=100
ACT_MA="0.4"
UNCERTAINTY_COEFF="0.0"        # env reward shaping coefficient (set > 0 to enable)
UNCERTAINTY_LOSS_COEFF="0.0"   # direct PPO loss coefficient (set > 0 to enable)
LH_BASE="assets/imitator_lh_inspire.pth"
IMITATION_EPOCH=""

# Usage:
#   ./run_residual.sh                    # use best imitation checkpoint, default suffixes below
#   ./run_residual.sh 1 7 14             # use best imitation checkpoint for each suffix
#   ./run_residual.sh --epoch 200 1 7 14 # use imitation checkpoint saved at epoch 200
IMITATION_SUFFIXES=()
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
            echo "Usage: $0 [--epoch N] [imitation suffixes...]"
            exit 0
            ;;
        *)
            IMITATION_SUFFIXES+=("$1")
            shift
            ;;
    esac
done

if [[ -n "${IMITATION_EPOCH}" ]] && ! [[ "${IMITATION_EPOCH}" =~ ^[0-9]+$ ]] ; then
    echo "[error] invalid epoch: ${IMITATION_EPOCH}"
    exit 1
fi

if (( ${#IMITATION_SUFFIXES[@]} == 0 )); then
    IMITATION_SUFFIXES=(1 7 14)
fi

resolve_rh_base_checkpoint() {
    local subset_size="$1"
    local epoch="${2:-}"
    local imitation_name="${IMITATION_EXPERIMENT_BASE}_${subset_size}"

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

for imitation_suffix in "${IMITATION_SUFFIXES[@]}"; do
    if ! [[ "${imitation_suffix}" =~ ^[0-9]+$ ]] || (( imitation_suffix < 1 )); then
        echo "[skip] Invalid imitation suffix: ${imitation_suffix}"
        continue
    fi

    imitation_name="${IMITATION_EXPERIMENT_BASE}_${imitation_suffix}"
    if [[ -n "${IMITATION_EPOCH}" ]]; then
        experiment_name="${EXPERIMENT_BASE}_${imitation_suffix}_ep_${IMITATION_EPOCH}"
    else
        experiment_name="${EXPERIMENT_BASE}_${imitation_suffix}_best"
    fi

    if ! RH_BASE=$(resolve_rh_base_checkpoint "${imitation_suffix}" "${IMITATION_EPOCH}"); then
        echo "[skip] No imitation checkpoint found for suffix=${imitation_suffix}"
        if [[ -n "${IMITATION_EPOCH}" ]]; then
            echo "       expected pattern: runs/${IMITATION_EXPERIMENT_BASE}_${imitation_suffix}__*/nn/last_${IMITATION_EXPERIMENT_BASE}_${imitation_suffix}_ep_${IMITATION_EPOCH}_*.pth"
        else
            echo "       expected pattern: runs/${IMITATION_EXPERIMENT_BASE}_${imitation_suffix}__*/nn/${IMITATION_EXPERIMENT_BASE}_${imitation_suffix}.pth"
        fi
        continue
    fi

    echo "[run] experiment=${experiment_name} dataIndices=${DATA_INDICES} rh_base=${RH_BASE}"

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
        early_stop_epochs=${EARLY_STOP} \
        actionsMovingAverage=${ACT_MA} \
        experiment=${experiment_name} \
        rh_base_model_checkpoint=${RH_BASE} \
        lh_base_model_checkpoint=${LH_BASE} \
        uncertaintyCoeff=${UNCERTAINTY_COEFF} \
        train.params.config.uncertainty_loss_coeff=${UNCERTAINTY_LOSS_COEFF}
done