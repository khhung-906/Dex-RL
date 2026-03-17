#!/bin/bash
# Train on OakInk-V2 single-hand task (format: data_idx = hash@stage, e.g. 0b3d1@0 or 20aed@0).
# List available OakInk tasks: python list_tasks.py oakink2

#SBATCH --partition=iris
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --account=iris
#SBATCH --output=runs/results/%A.out
#SBATCH --error=runs/results/%A.err
#SBATCH --job-name="rl"
#SBATCH --exclude=iliad1,iliad2,iliad3,iliad4,iliad-hgx-1,iris1,iris2,iris3,iris4,iris5

export LD_LIBRARY_PATH=/iliad/u/khhung/miniconda3/envs/rlgpu/lib:$LD_LIBRARY_PATH
export WANDB_MODE=online

# OakInk task: first 5 digits of sequence hash + @ + stage index (e.g. 0b3d1@0, 20aed@0)
OAKINK_IDX=75a7a@1

# # Step 2: Train RL policy on OakInk task
UNCERTAINTY_COEFF="0.0"       # env reward shaping: + uncertainty * residual_norm bonus
UNCERTAINTY_LOSS_COEFF="0.0001"  # PPO loss:           - uncertainty * ||mu|| gradient term

python main/rl/train.py \
    task=ResDexHand \
    dexhand=inspire \
    side=RH \
    headless=true \
    num_envs=4096 \
    learning_rate=2e-4 \
    test=false \
    randomStateInit=true \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    dataIndices=[${OAKINK_IDX}] \
    early_stop_epochs=1000 \
    rl_train.params.config.save_frequency=50 \
    actionsMovingAverage=0.4 \
    uncertaintyCoeff=${UNCERTAINTY_COEFF} \
    rl_train.params.config.uncertainty_loss_coeff=${UNCERTAINTY_LOSS_COEFF} \
    experiment=oakink_${OAKINK_IDX//@/_}_inspire_uncertain_${UNCERTAINTY_COEFF}_${UNCERTAINTY_LOSS_COEFF} \
    wandb_activate=true \
    wandb_project=dexrl
