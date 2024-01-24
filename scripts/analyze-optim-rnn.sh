#!/bin/zsh

source .env
cd src
python -m analysis \
          --data $ARTIFACTS/data/AMASS_t_optim_birnn \
          --checkpoints $ARTIFACTS/checkpoints/optim/birnn \
          --model birnn \
          --stage optim \
          --smpl-model $ARTIFACTS/smpl/SMPL_male.pkl \
          --subset 100 \
          --num-sensors 6 \
          --output $ARTIFACTS/analysis/optim/birnn \
          --recurse