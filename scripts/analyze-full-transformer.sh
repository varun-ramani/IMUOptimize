#!/bin/zsh

source .env
cd src
python -m analysis \
          --data $ARTIFACTS/data/AMASS_t_all \
          --checkpoints $ARTIFACTS/checkpoints/full/transformer \
          --model transformer \
          --stage full \
          --smpl-model $ARTIFACTS/smpl/SMPL_male.pkl \
          --subset 1000 \
          --num-sensors 24 \
          --output $ARTIFACTS/analysis/full/transformer \
          --recurse