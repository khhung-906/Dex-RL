#!/bin/bash

set -euo pipefail

TASK="DexHandImitator"
DATA_INDICES="[]"
COLLECTED_DATA_BASE="sr_75a7a_1_r"
EXPERIMENT_BASE="write_on_paper_imitation"

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

# runtime subset selection mode: first | middle | last | random | custom
RUNTIME_SELECT_MODE="first"
# default fixed seed for runtime random sampling; can be overridden by --seed
RUNTIME_RANDOM_SEED="42"
CUSTOM_FILE="data/collected_rollouts/cluster1_top10.csv"

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
#   ./run_imitation.sh                                       # uses default subset sizes below, mode=first
#   ./run_imitation.sh --mode first 10 30 50                 # pick first N
#   ./run_imitation.sh --mode middle 10 30 50                # pick middle N
#   ./run_imitation.sh --mode last 10 30 50                  # pick last N
#   ./run_imitation.sh --mode random --seed 123 10 30        # pick random N with custom seed
#   ./run_imitation.sh --mode custom --custom-file path.csv  # run once with IDs from CSV
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
        --custom-file)
            if (( $# < 2 )); then
                echo "[error] --custom-file requires a path"
                exit 1
            fi
            CUSTOM_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--mode first|middle|last|random|custom] [--seed N] [--custom-file PATH] [subset sizes...]"
            exit 0
            ;;
        *)
            SUBSET_SIZES+=("$1")
            shift
            ;;
    esac
done

if [[ ! "${RUNTIME_SELECT_MODE}" =~ ^(first|middle|last|random|custom)$ ]]; then
    echo "[error] invalid mode: ${RUNTIME_SELECT_MODE} (expected: first|middle|last|random|custom)"
    exit 1
fi

if [[ "${RUNTIME_SELECT_MODE}" != "custom" ]] && (( ${#SUBSET_SIZES[@]} == 0 )); then
    SUBSET_SIZES=(10 30 50)
fi

load_custom_indices_from_csv() {
    local csv_path="$1"
    if [[ ! -f "${csv_path}" ]]; then
        echo "[error] custom csv file not found: ${csv_path}"
        return 1
    fi

    python - "${csv_path}" <<'PY'
import csv
import sys

path = sys.argv[1]
ids = []
seen = set()

with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row:
            continue

        candidate = None
        for key in ("sr_id", "collection_id", "data_id", "id"):
            val = (row.get(key) or "").strip()
            if val:
                candidate = val
                break

        if candidate is None:
            for val in row.values():
                val = (val or "").strip()
                if val.startswith("sr_"):
                    candidate = val
                    break

        if candidate and candidate not in seen:
            seen.add(candidate)
            ids.append(candidate)

print("[" + ",".join(ids) + "]")
PY
}

if [[ "${RUNTIME_SELECT_MODE}" == "custom" ]]; then
    DATA_INDICES=$(load_custom_indices_from_csv "${CUSTOM_FILE}")
    if [[ -z "${DATA_INDICES}" || "${DATA_INDICES}" == "[]" ]]; then
        echo "[error] no valid collected IDs found in custom csv: ${CUSTOM_FILE}"
        exit 1
    fi

    custom_stem=$(basename "${CUSTOM_FILE}")
    custom_stem="${custom_stem%.*}"
    experiment_name="${EXPERIMENT_BASE}_custom_${custom_stem}"

    echo "[run] experiment=${experiment_name} dataIndices=${DATA_INDICES}"

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
        usePIDControl=${USE_PID} \
        experiment=${experiment_name}

    exit 0
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