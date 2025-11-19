import numpy as np
import optuna
from optuna.samplers import TPESampler
from rcpy.models import create_model, save_trained_model
from rcpy.forecasting import forecast_rcpy
#from rcpy.hypopt import standard_loss, soft_horizon_loss
from rcpy.analysis import compute_skill
import json, os


# 2) Define search_space
def search_space(trial, reservoir_units, hypopt_config):
    """
    Sample hyperparameters for optimization based on config.
    - hypopt_config: dictionary of hyperparameters from config["optimization"]["hyperparameters"]
    """

    params = {}

    for name, settings in hypopt_config.items():

        # 2️⃣ If 'fixed' key present in YAML config
        if "fixed" in settings:
            params[name] = settings["fixed"]
            print(f"⚡ {name} fixed from YAML: {params[name]}")
            continue

        # 3️⃣ Otherwise sample according to defined range
        low, high = map(float, settings["range"])
        log_scale = settings.get("log", False)

        # Validate
        if low > high:
            raise ValueError(f"Invalid range for {name}: low={low}, high={high}")

        if log_scale and low <= 0:
            raise ValueError(f"Log-scale hyperparameter {name} must have low > 0, got low={low}")

        if log_scale:
            params[name] = trial.suggest_float(name, low, high, log=True)
        else:
            params[name] = trial.suggest_float(name, low, high)

        print(f"⚙️  {name} optimized: {params[name]:.4g}")

    # Handle sparsity default
    if "p" not in params:
        params["p"] = 50 / (reservoir_units - 1)
        print(f"⚙️  p fixed at {params['p']:.4g}")

    return params


# Forecasting function
def get_loss(data, val_length, model_config, loss_function, seed):

    #dim = data["train_data"].shape[1]
    model_config["seed"] = seed
    washout_training = 500
    warmup_training = 500

    model = create_model(model_config=model_config)
    model.fit(data["train_data"][:-1], data["train_data"][1:], warmup=washout_training)


    Y_pred = forecast_rcpy(
        model=model,
        warmup_data=data['train_data'][-warmup_training:],
        forecast_length=val_length
    )

    if loss_function == "soft_horizon":
        #return soft_horizon_loss(data["val_data"], Y_pred, metric="rmse")
        return -compute_skill(data["val_data"], Y_pred, method="efh", threshold=0.2, softness=0.02)
    elif loss_function == "rmse":
        #return standard_loss(data["val_data"], Y_pred, metric="rmse")
        return compute_skill(data["val_data"], Y_pred, method="error", metric="rmse")

# Building objective function
def build_objective(data, val_length, reservoir_units, loss_function, seed, hypopt_config):

    def objective(trial):
        # 1️⃣ Get hyperparameters from search space
        hyperparams = search_space(trial, reservoir_units, hypopt_config)

        # 2️⃣ Always include reservoir size
        hyperparams["reservoir_units"] = reservoir_units

        # 3️⃣ Compute loss
        loss = get_loss(data, val_length, hyperparams, loss_function, seed)

        trial.report(loss, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return loss

    return objective



def run_optimization(study_name, db_file, objective_func, total_trials, timeout_hours):
    sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)

    if db_file:  # Save to disk if db_file is provided
        storage = f"sqlite:///{db_file}"
        study_name_final = study_name
        load_if_exists = True
    else:  # Use in-memory study
        storage = None
        study_name_final = study_name
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
