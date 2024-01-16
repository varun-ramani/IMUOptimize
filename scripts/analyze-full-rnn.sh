#!/bin/zsh

source .env
cd src
python -m analysis \
          --data $ARTIFACTS/data/AMASS_t_all \
          --checkpoints $ARTIFACTS/checkpoints/full/birnn \
          --model birnn \
          --stage full \
          --smpl-model $ARTIFACTS/smpl/SMPL_male.pkl \
          --subset 100 \
          --num-sensors 24 \
          --output $ARTIFACTS/analysis/full/birnn \
          --recurse