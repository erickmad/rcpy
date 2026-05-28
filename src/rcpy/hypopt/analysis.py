import numpy as np
import optuna

def optimization_history(system, reservoir_units, offset, results_dir, seeds):

    best_so_far_all = []
    for seed in seeds:
        study_name = f"{system}_N{reservoir_units}_S{int(seed)}"
        storage = f"sqlite:///{results_dir}/hypopt/{study_name}.db"

        study = optuna.load_study(study_name=study_name, storage=storage)

        trials = study.trials
        values = np.array([t.value for t in trials if t.value is not None])

        best_so_far = np.minimum.accumulate(values)
        best_so_far_all.append(best_so_far)

    return np.array(best_so_far_all)