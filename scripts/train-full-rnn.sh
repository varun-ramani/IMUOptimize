#!/bin/zsh

source .env
cd src
python -m model \
          --data $ARTIFACTS/data/AMASS_t_all \
          --checkpoints $ARTIFACTS/checkpoints/full/birnn \
          --model birnn \
          --stage full \
          --epochs 5 \
          --num-sensors 24