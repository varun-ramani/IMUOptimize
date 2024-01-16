#!/bin/zsh

source .env
cd src
python -m model \
          --data $ARTIFACTS/data/AMASS_t_optim \
          --checkpoints $ARTIFACTS/checkpoints/optim/birnn \
          --model birnn \
          --stage optim \
          --epochs 5 \
          --num-sensors 6