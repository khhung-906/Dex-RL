#!/bin/bash
# Evaluation script for imperfect imitation experiments
# Runs test=true with the best checkpoint from each experiment
# Output: per-episode reward and success rate (deterministic policy)

set -e

export PATH="/user/maniptrans/bin:$PATH"
export LD_LIBRARY_PATH=/home/ava/miniconda3/envs/maniptrans/lib:${LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=9

PYTHON=/user/maniptrans/bin/python
NUM_ENVS=${NUM_ENVS:-512}
DEXHAND=${DEXHAND:-inspire}
NOISE_STD=${NOISE_STD:-0.02}

cd /user/ManipTrans

echo "=== Imperfect Imitation Evaluation ==="
echo "Noise std: ${NOISE_STD} | Num envs: ${NUM_ENVS} | Dexhand: ${DEXHAND}"
echo ""

# Exp 1: Remove test tube from rack (1be0e@0)
echo ">>> Eval 1: 1be0e@0 (remove test tube from rack)"
$PYTHON main/rl/train.py \
  task=ResDexHand \
  dexhand=${DEXHAND} \
  side=RH \
  headless=true \
  test=true \
  num_envs=${NUM_ENVS} \
  sim_device=cuda:0 \
  rl_device=cuda:0 \
  graphics_device_id=0 \
  dataIndices='[1be0e@0]' \
  demo_noise_std=${NOISE_STD} \
  randomStateInit=false \
  rh_base_model_checkpoint=assets/imitator_rh_${DEXHAND}.pth \
  lh_base_model_checkpoint=assets/imitator_lh_${DEXHAND}.pth \
  checkpoint=runs/noisy_1be0e_at_0_inspire__02-28-15-03-20/nn/noisy_1be0e_at_0_inspire.pth \
  wandb_activate=False \
  experiment=eval_noisy_1be0e_at_0_${DEXHAND}
echo ">>> Done Eval 1"
echo ""

# Exp 2: Place test tube on rack (1be0e@2)
echo ">>> Eval 2: 1be0e@2 (place test tube on rack)"
$PYTHON main/rl/train.py \
  task=ResDexHand \
  dexhand=${DEXHAND} \
  side=RH \
  headless=true \
  test=true \
  num_envs=${NUM_ENVS} \
  sim_device=cuda:0 \
  rl_device=cuda:0 \
  graphics_device_id=0 \
  dataIndices='[1be0e@2]' \
  demo_noise_std=${NOISE_STD} \
  randomStateInit=false \
  rh_base_model_checkpoint=assets/imitator_rh_${DEXHAND}.pth \
  lh_base_model_checkpoint=assets/imitator_lh_${DEXHAND}.pth \
  checkpoint=runs/noisy_1be0e_at_2_inspire__02-28-16-35-03/nn/noisy_1be0e_at_2_inspire.pth \
  wandb_activate=False \
  experiment=eval_noisy_1be0e_at_2_${DEXHAND}
echo ">>> Done Eval 2"
echo ""

# Exp 3: Heat beaker (c3c69@0)
echo ">>> Eval 3: c3c69@0 (heat beaker)"
$PYTHON main/rl/train.py \
  task=ResDexHand \
  dexhand=${DEXHAND} \
  side=RH \
  headless=true \
  test=true \
  num_envs=${NUM_ENVS} \
  sim_device=cuda:0 \
  rl_device=cuda:0 \
  graphics_device_id=0 \
  dataIndices='[c3c69@0]' \
  demo_noise_std=${NOISE_STD} \
  randomStateInit=false \
  rh_base_model_checkpoint=assets/imitator_rh_${DEXHAND}.pth \
  lh_base_model_checkpoint=assets/imitator_lh_${DEXHAND}.pth \
  checkpoint=runs/noisy_c3c69_at_0_inspire__02-28-17-20-44/nn/noisy_c3c69_at_0_inspire.pth \
  wandb_activate=False \
  experiment=eval_noisy_c3c69_at_0_${DEXHAND}
echo ">>> Done Eval 3"
echo ""
echo "$(date) - === Evaluation complete ==="
