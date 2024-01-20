#!/bin/zsh

source .env
cd src
python -m model \
          --data $ARTIFACTS/data/AMASS_t_optim_transformer \
          --checkpoints $ARTIFACTS/checkpoints/optim/transformer \
          --model transformer \
          --stage optim \
          --epochs 5 \
          --num-sensors 6