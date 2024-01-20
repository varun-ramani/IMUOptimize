#!/bin/zsh

source .env
cd src
python -m model \
          --data $ARTIFACTS/data/AMASS_t_all \
          --checkpoints $ARTIFACTS/checkpoints/full/transformer \
          --model transformer \
          --stage full \
          --epochs 5 \
          --num-sensors 24