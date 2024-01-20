#!/bin/zsh

source .env
cd src
python -m data.synthesize \
          --input $ARTIFACTS/data/AMASS \
          --output $ARTIFACTS/data/AMASS_t_optim_transformer \
          --model $ARTIFACTS/smpl/SMPL_male.pkl \
          --joints "0 6 3 17 13 21" \
          --purge-existing \
          --keep-subdirectories