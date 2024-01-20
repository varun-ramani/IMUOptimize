#!/bin/zsh

source .env
cd src
python -m data.synthesize \
          --input $ARTIFACTS/data/AMASS \
          --output $ARTIFACTS/data/AMASS_t_optim_birnn \
          --model $ARTIFACTS/smpl/SMPL_male.pkl \
          --joints "0 9 21 2 12 15" \
          --purge-existing \
          --keep-subdirectories