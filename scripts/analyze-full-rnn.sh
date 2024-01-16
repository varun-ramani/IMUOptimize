cd src
python -m analysis \
          --data ../artifacts/data/AMASS_t_all \
          --checkpoints ../artifacts/checkpoints/full/birnn \
          --model birnn \
          --stage full \
          --smpl-model ../artifacts/smpl/SMPL_male.pkl \
          --subset 50 \
          --num-sensors 24 \
          --output ../artifacts/analysis/full/birnn \
          --recurse