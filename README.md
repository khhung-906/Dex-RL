# Residual RL with Uncertainty-Aware Adaptation

This repository contains the official implementation of our work on studying the interaction between imitation learning and residual reinforcement learning, with a focus on uncertainty-aware adaptation for dexterous manipulation tasks.

Our codebase is built upon and modified from the [ManipTrans](https://github.com/ManipTrans/ManipTrans) codebase, with additional components for controlled imitation training and uncertainty-aware residual RL.

---

## 0. Setup
Follow [ManipTrans](https://github.com/ManipTrans/ManipTrans) README.md.

## 1. Imitation Learning Training Scripts

Recommended order:
1. Train imitation policy with `run_imitation.sh`
2. Train residual policy with `run_residual.sh`
3. Evaluate with `run_test.sh`

### 1.1 `run_imitation.sh`
Train imitation policy (`task=DexHandImitator`) with selected collected rollout IDs.

Basic usage:
```bash
# Default: mode=first, subset sizes 10 30 50
bash run_imitation.sh

# Choose selection mode and subset sizes
bash run_imitation.sh --mode first 10 30 50
bash run_imitation.sh --mode middle 10 30 50
bash run_imitation.sh --mode last 10 30 50

# Random sampling with fixed seed
bash run_imitation.sh --mode random --seed 123 10 30

# Use IDs from custom CSV (run once)
bash run_imitation.sh --mode custom --custom-file data/collected_rollouts/cluster1_top10.csv
```

Arguments:
- `--mode first|middle|last|random|custom`: subset selection strategy.
- `--seed N`: random seed used only when `--mode random`.
- `--custom-file PATH`: CSV path used when `--mode custom`.
- trailing numbers (e.g., `10 30 50`): subset sizes to run.

### 1.2 `run_residual.sh`
Train residual policy (`task=ResDexHand`) using checkpoints from imitation runs.

Basic usage:
```bash
# Default: use best imitation checkpoint for suffixes 10 30 50
bash run_residual.sh

# Train only selected suffixes
bash run_residual.sh 10 30

# Use imitation checkpoint from a specific epoch
bash run_residual.sh --epoch 200 10 30 50
```

Arguments:
- `--epoch N`: load imitation checkpoint `last_*_ep_N_*.pth`.
- trailing numbers: suffixes that match imitation experiment names.

### 1.3 `run_test.sh`
Run test/inference using trained residual checkpoints (and matched imitation base checkpoints).

Basic usage:
```bash
# Default: test suffixes 10 30 50 from "best" residual checkpoints
bash run_test.sh

# Test selected suffixes
bash run_test.sh 10 30

# Test residual checkpoints associated with imitation epoch N
bash run_test.sh --epoch 200 10 30 50
```

Arguments:
- `--epoch N`: pick residual experiments named with `_ep_N`.
- trailing numbers: pair suffixes to test.

Notes:
- `run_residual.sh` and `run_test.sh` expect imitation experiments named as `${IMITATION_EXPERIMENT_BASE}_<suffix>`.
- If a required checkpoint is missing, scripts will print `[skip]` with expected file pattern.
- You can edit defaults (e.g., `DATA_INDICES`, `EXPERIMENT_BASE`, `NUM_ENVS`) directly in each script.


## 2. Uncertainty Aware Training Scripts

### 2.1 `train_oak_good.sh`
Train residual RL on OakInk-V2 with uncertainty-aware terms enabled.

Basic usage:
```bash
# Run with default task and uncertainty coefficients defined in the script
bash train_oak_good.sh
```

Default key settings in script:
- `UNCERTAINTY_COEFF=0.0`: uncertainty reward shaping coefficient.
- `UNCERTAINTY_LOSS_COEFF=0.01`: uncertainty-related PPO loss coefficient.

Notes:
- This script currently uses fixed values; edit variables directly in `train_oak_good.sh` when switching tasks or hyperparameters.
- `dataIndices` is passed as `[${OAKINK_IDX}]`, so `OAKINK_IDX` should follow `hash@stage` format (e.g., `0b3d1@0`).
- Base imitation checkpoints are loaded from `assets/imitator_rh_inspire.pth` and `assets/imitator_lh_inspire.pth`.


---

## Acknowledgements

This codebase is modified from the [ManipTrans](https://github.com/ManipTrans/ManipTrans) repository. We thank the original authors for their implementation and contributions.
