#!/bin/zsh

source .env
cd src
python -m model \
          --data $ARTIFACTS/data/AMASS_t_optim_birnn \
          --checkpoints $ARTIFACTS/checkpoints/optim/birnn \
          --model birnn \
          --stage optim \
          --epochs 5 \
          --num-sensors 6