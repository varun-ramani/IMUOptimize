#!/bin/zsh

source .env
cd src
python -m data.synthesize \
          --input $ARTIFACTS/data/AMASS \
          --output $ARTIFACTS/data/AMASS_t_optim \
          --model $ARTIFACTS/smpl/SMPL_male.pkl \
          --joints "9 21 2 12 15 0" \
          --purge-existing \
          --keep-subdirectories