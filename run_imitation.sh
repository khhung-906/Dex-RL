#!/bin/bash

set -euo pipefail

TASK="DexHandImitator"
# DATA_INDICES="[1be0e@1,1dc51@1,419a3@5,5fde1@3,7fa33@2,847ba@2,8adcc@2,9f2cb@1,a5345@1,ab61c@1,ebd70@5,ed9bc@1,f5a3d@10,ff220@3]"
# DATA_INDICES="[20034@0,67cb3@1,6c1ba@3,86fb0@1,99369@0,f3fe2@0]"
DATA_INDICES="[]"
COLLECTED_DATA_BASE="sr_75a7a_2_r"
EXPERIMENT_BASE="write_on_paper_imitation_collected"

DEXHAND="inspire"
SIDE="RH"
HEADLESS="true"
NUM_ENVS=2048
LR="2e-4"
TEST="false"
RANDOM_INIT="true"
ACT_MA="0.4"
USE_PID="False"

REGISTRY_PATH="data/collected_rollouts/data_id_registry.json"

# runtime subset selection mode: first | middle | last | random
RUNTIME_SELECT_MODE="first"
# default fixed seed for runtime random sampling; can be overridden by --seed
RUNTIME_RANDOM_SEED="42"

append_collected_indices_by_base() {
    local base="$1"
    local registry_path="$2"

    if [[ -z "${base}" ]]; then
        return 0
    fi

    if [[ ! -f "${registry_path}" ]]; then
        echo "[warn] collected registry not found: ${registry_path}" >&2
        return 0
    fi

    local extra_csv
    extra_csv=$(python - "${registry_path}" "${base}" <<'PY'
import json
import sys

registry_path = sys.argv[1]
base = sys.argv[2]

try:
    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    print("")
    raise SystemExit(0)

items = data.get("items", {}) if isinstance(data, dict) else {}
if not isinstance(items, dict):
    print("")
    raise SystemExit(0)

collected = []
seen = set()

# Prefer explicit base + child_rollout_ids ordering if available.
base_item = items.get(base)
if isinstance(base_item, dict):
    for data_id in [base] + list(base_item.get("child_rollout_ids", [])):
        if isinstance(data_id, str) and data_id.startswith(base) and data_id not in seen:
            seen.add(data_id)
            collected.append(data_id)

# Fallback/extra: include any other registered ids with same prefix.
for key in sorted(items.keys()):
    if isinstance(key, str) and key.startswith(base) and key not in seen:
        seen.add(key)
        collected.append(key)

print(",".join(collected))
PY
)

    if [[ -z "${extra_csv}" ]]; then
        return 0
    fi

    local existing_csv="${DATA_INDICES#[}"
    existing_csv="${existing_csv%]}"

    local combined_csv
    if [[ -n "${existing_csv}" ]]; then
        combined_csv="${existing_csv},${extra_csv}"
    else
        combined_csv="${extra_csv}"
    fi

    # Keep insertion order while removing duplicates.
    DATA_INDICES=$(python - "${combined_csv}" <<'PY'
import sys

vals = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
seen = set()
ordered = []
for v in vals:
    if v not in seen:
        seen.add(v)
        ordered.append(v)

print("[" + ",".join(ordered) + "]")
PY
)
}

append_collected_indices_by_base "${COLLECTED_DATA_BASE}" "${REGISTRY_PATH}"

# Usage:
#   ./run_imitation.sh                                  # uses default subset sizes below, mode=first
#   ./run_imitation.sh --mode first 1 3 5 10           # pick first N
#   ./run_imitation.sh --mode middle 10 30 50          # pick middle N
#   ./run_imitation.sh --mode last 10 30 50            # pick last N
#   ./run_imitation.sh --mode random --seed 123 10 30  # pick random N with custom seed
SUBSET_SIZES=()
while (( $# > 0 )); do
    case "$1" in
        --mode)
            if (( $# < 2 )); then
                echo "[error] --mode requires a value: first | middle | last | random"
                exit 1
            fi
            RUNTIME_SELECT_MODE="${2,,}"
            shift 2
            ;;
        --seed)
            if (( $# < 2 )); then
                echo "[error] --seed requires a value"
                exit 1
            fi
            RUNTIME_RANDOM_SEED="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--mode first|middle|last|random] [--seed N] [subset sizes...]"
            exit 0
            ;;
        *)
            SUBSET_SIZES+=("$1")
            shift
            ;;
    esac
done

if [[ ! "${RUNTIME_SELECT_MODE}" =~ ^(first|middle|last|random)$ ]]; then
    echo "[error] invalid mode: ${RUNTIME_SELECT_MODE} (expected: first|middle|last|random)"
    exit 1
fi

if (( ${#SUBSET_SIZES[@]} == 0 )); then
    SUBSET_SIZES=(1 3 5)
fi

indices_csv="${DATA_INDICES#[}"
indices_csv="${indices_csv%]}"
IFS=',' read -r -a DATA_INDEX_ARRAY <<< "${indices_csv}"
TOTAL_INDICES=${#DATA_INDEX_ARRAY[@]}

for subset_size in "${SUBSET_SIZES[@]}"; do
    if ! [[ "${subset_size}" =~ ^[0-9]+$ ]] || (( subset_size < 1 )); then
        echo "[skip] Invalid subset size: ${subset_size}"
        continue
    fi

    if (( subset_size > TOTAL_INDICES )); then
        echo "[skip] subset_size=${subset_size} exceeds total indices (${TOTAL_INDICES})"
        continue
    fi

    if [[ "${RUNTIME_SELECT_MODE}" == "random" ]]; then
        subset_joined=$(python - "${indices_csv}" "${subset_size}" "${RUNTIME_RANDOM_SEED}" <<'PY'
import random
import sys

vals = [x.strip() for x in sys.argv[1].split(',') if x.strip()]
k = int(sys.argv[2])
seed = sys.argv[3]

rng = random.Random(seed)
picked = rng.sample(vals, k)
print(",".join(picked))
PY
)
    elif [[ "${RUNTIME_SELECT_MODE}" == "middle" ]]; then
        start_idx=$(( (TOTAL_INDICES - subset_size) / 2 ))
        subset_joined=$(IFS=','; echo "${DATA_INDEX_ARRAY[*]:start_idx:subset_size}")
    elif [[ "${RUNTIME_SELECT_MODE}" == "last" ]]; then
        start_idx=$(( TOTAL_INDICES - subset_size ))
        subset_joined=$(IFS=','; echo "${DATA_INDEX_ARRAY[*]:start_idx:subset_size}")
    else
        subset_joined=$(IFS=','; echo "${DATA_INDEX_ARRAY[*]:0:subset_size}")
    fi

    subset_indices="[${subset_joined}]"
    experiment_name="${EXPERIMENT_BASE}_${RUNTIME_SELECT_MODE}_${subset_size}"

    echo "[run] experiment=${experiment_name} dataIndices=${subset_indices}"

    python main/rl/train.py \
        task=${TASK} \
        dexhand=${DEXHAND} \
        side=${SIDE} \
        headless=${HEADLESS} \
        num_envs=${NUM_ENVS} \
        learning_rate=${LR} \
        test=${TEST} \
        randomStateInit=${RANDOM_INIT} \
        dataIndices=${subset_indices} \
        actionsMovingAverage=${ACT_MA} \
        usePIDControl=${USE_PID} \
        experiment=${experiment_name}
done