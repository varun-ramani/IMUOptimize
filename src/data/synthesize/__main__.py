from .parallelize import produce_synthetic_dataset

produce_synthetic_dataset('../artifacts/data/AMASS', '../artifacts/data/AMASS_t_all', '../artifacts/smpl/SMPL_male.pkl', list(range(24)), purge_existing=True)