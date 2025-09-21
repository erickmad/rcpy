import numpy as np
import optuna
from optuna.samplers import TPESampler
from rcpy.models import create_model
from rcpy.forecasting import forecast_rcpy
from rcpy.hypopt import standard_loss, soft_horizon_loss
import json, os


# 2) Define search_space
def search_space(trial):
    # Sample leak_rate
    leak_rate = trial.suggest_float("leak_rate", 0.1, 1.0)

    # Compute lower bound
    min_spectral_radius = max(1.0 - leak_rate, 1e-2)  # Avoid values ≥ 2

    # Guard against invalid ranges
    if min_spectral_radius >= 2.0:
        raise optuna.exceptions.TrialPruned()

    spectral_radius = trial.suggest_float(
        "spectral_radius",
        min_spectral_radius,
        1.5,
        log=True
    )

    return {
        "spectral_radius": spectral_radius,
        "alpha": trial.suggest_float("alpha", 1e-7, 1e-2, log=True),
        "leak_rate": leak_rate,
        "input_scaling": trial.suggest_float("input_scaling", 0.1, 1.0),
        "p": trial.suggest_float("p", 0.01, 0.1),
    }

# Forecasting function
def get_loss(data, val_length, model_config, loss_function, seed):

    #dim = data["train_data"].shape[1]
    model_config["seed"] = seed

    model = create_model(model_config=model_config)
    model.fit(data["train_data"][:-1], data["train_data"][1:], warmup=240)

    Y_pred = forecast_rcpy(
        model=model,
        warmup_data=data['warmup_data'],
        forecast_length=val_length
    )

    if loss_function == "soft_horizon":
        return soft_horizon_loss(data["val_data"], Y_pred, metric="rmse")
    elif loss_function == "rmse":
        return standard_loss(data["val_data"], Y_pred, metric="rmse")

# Building objective function
def build_objective(data, val_length, reservoir_units, loss_function, seed):
    def objective(trial):
        hyperparams = search_space(trial)
        hyperparams["reservoir_units"] = reservoir_units

        loss = get_loss(data, val_length, hyperparams, loss_function, seed)

        trial.report(loss, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return loss
    return objective


def run_optimization(study_name, db_file, objective_func, total_trials, timeout_hours):
    sampler = optuna.samplers.TPESampler(multivariate=True)

    if db_file:  # Save to disk if db_file is provided
        storage = f"sqlite:///{db_file}"
        study_name_final = study_name
        load_if_exists = True
    else:  # Use in-memory study
        storage = None
        study_name_final = None
        load_if_exists = False

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name_final,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )

    study.optimize(objective_func, n_trials=total_trials, timeout=timeout_hours * 3600)
    return study

# Save best parameters
""" def save_best_params(study, seed, output_file):
    best_params = study.best_params
    best_params["seed"] = seed
    with open(output_file, "w") as f:
        json.dump(best_params, f, indent=4) """

def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def save_best_params(study, filename):
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    best_params = study.best_trial.params
    #best_params["seed"] = int(seed)  # ensure seed is a native int too
    with open(filename, "w") as f:
        json.dump(best_params, f, indent=4, default=convert_numpy)
