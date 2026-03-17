#!/bin/bash
# Train on OakInk-V2 single-hand task (format: data_idx = hash@stage, e.g. 0b3d1@0 or 20aed@0).
# List available OakInk tasks: python list_tasks.py oakink2

#SBATCH --partition=iliad
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --account=iliad
#SBATCH --output=runs/results/%A.out
#SBATCH --error=runs/results/%A.err
#SBATCH --job-name="rl"
#SBATCH --exclude=iliad1,iliad2,iliad3,iliad4,iliad-hgx-1

export LD_LIBRARY_PATH=/iliad/u/khhung/miniconda3/envs/rlgpu/lib:$LD_LIBRARY_PATH

# OakInk task: first 5 digits of sequence hash + @ + stage index (e.g. 0b3d1@0, 20aed@0)
OAKINK_IDX=75a7a@1

# # Step 1: Retarget MANO to dex hand for OakInk (right hand for RH task)
# python main/dataset/mano2dexhand.py --data_idx ${OAKINK_IDX} --side right --dexhand inspire --headless --iter 7000

# # Step 2: Train RL policy on OakInk task
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
    experiment=oakink_${OAKINK_IDX//@/_}_inspire \
    wandb_activate=true \
    wandb_project=dexrl
