#!/bin/zsh

source .env
cd src
python -m data.synthesize \
          --input $ARTIFACTS/data/AMASS \
          --output $ARTIFACTS/data/AMASS_t_all \
          --model $ARTIFACTS/smpl/SMPL_male.pkl \
          --joints "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23" \
          --purge-existing \
          --keep-subdirectories