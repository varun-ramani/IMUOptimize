cd src
python -m model \
          --data ../artifacts/data/AMASS_t_optim \
          --checkpoints ../artifacts/checkpoints/optim/birnn \
          --model birnn \
          --stage optim \
          --epochs 5