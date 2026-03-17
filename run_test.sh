#!/bin/bash

TASK="ResDexHand"
DATA_INDICES="[05aa2@0,75a7a@1]"
EXPERIMENT="write_on_paper_0.4"
RH_BASE="assets/imitator_rh_inspire.pth"

DEXHAND="inspire"
SIDE="RH"
HEADLESS="true"
NUM_ENVS=4096
LR="2e-4"
TEST="false"
RANDOM_INIT="true"
EARLY_STOP=300
ACT_MA="0.4"
LH_BASE="assets/imitator_lh_inspire.pth"

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
    experiment=${EXPERIMENT} \
    rh_base_model_checkpoint=${RH_BASE} \
    lh_base_model_checkpoint=${LH_BASE} \
    wandb_activate=true \
    wandb_project=dexrl