cd src
python -m model \
          --data ../artifacts/data/AMASS_t_all \
          --checkpoints ../artifacts/checkpoints/full/birnn \
          --model birnn \
          --stage full \
          --epochs 5