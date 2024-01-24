#!/bin/zsh

source .env
cd src
python -m data.synthesize \
          --input $ARTIFACTS/data/AMASS \
          --output $ARTIFACTS/data/AMASS_t_optim_birnn \
          --model $ARTIFACTS/smpl/SMPL_male.pkl \
          --joints "0 16 20 5 9 4" \
          --purge-existing \
          --keep-subdirectories