cd src
python -m data.synthesize \
          --input '../artifacts/data/AMASS' \
          --output '../artifacts/data/AMASS_t_optim' \
          --model '../artifacts/smpl/SMPL_male.pkl' \
          --joints "9 21 2 12 15 0" \
          --purge-existing \
          --keep-subdirectories