import numpy as np
import optuna
from optuna.samplers import TPESampler
from rcpy.models import create_model, save_trained_model
from rcpy.training import train_model
from rcpy.forecasting import forecast_rcpy
#from rcpy.hypopt import standard_loss, soft_horizon_loss
from rcpy.analysis import compute_skill
import json, os
from pathlib import Path
from typing import Any



def search_space(trial, hypopt_search):
    """
    Build Optuna search space from a unified config structure.
    Supports:
      - fixed values
      - continuous ranges (linear or log)
      - future extensibility (choices, integer ranges, etc.)
    """

    params = {}

    for name, cfg in hypopt_search.items():

        # ----------------------
        # 1. Fixed hyperparameter
        # ----------------------
        if "fixed" in cfg:
            params[name] = cfg["fixed"]
            #print(f"⚡ {name} fixed at: {params[name]}")
            continue

        # ----------------------
        # 2. Range-based hyperparameter
        # ----------------------
        if "range" in cfg:
            low, high = map(float, cfg["range"])
            log_scale = bool(cfg["log"])

            if low > high:
                raise ValueError(f"Invalid range for {name}: low={low} > high={high}")

            if log_scale and low <= 0:
                raise ValueError(
                    f"Log-scale parameter '{name}' must have low > 0. Got low={low}"
                )

            params[name] = trial.suggest_float(
                name,
                low,
                high,
                log=log_scale
            )
            #print(f"⚙️  {name} sampled: {params[name]:.4g}")
            continue

        # ----------------------
        # 3. Unsupported hyperparameter type
        # ----------------------
        raise ValueError(
            f"Hyperparameter '{name}' must contain either 'fixed' or 'range'. "
            f"Got keys: {list(cfg.keys())}"
        )

    return params


def get_loss(data, hyperparams, washout_training, forecast_config, loss_config, seed):

    #dim = data["train_data"].shape[1]
    hyperparams["seed"] = seed

    warmup_length = forecast_config["warmup_length"]
    val_length = len(data["val_data"])

    # Check for potential warmup overlap with validation
    if warmup_length > val_length:
        import warnings
        warnings.warn(
            f"Warmup length ({warmup_length}) is longer than validation length "
            f"({val_length}). This may lead to data leakage!"
        )

    model = create_model(hyperparams=hyperparams, output_dim=data["train_data"].shape[1])
    trained_model = train_model(model=model, train_data=data["train_data"],
                                forecasting_step=forecast_config["forecasting_step"],
                                washout_training=washout_training,)

    Y_pred = forecast_rcpy(
        model=trained_model,
        warmup_data=data['train_data'][-warmup_length:],
        forecast_length=val_length
    )
    

    if loss_config["function"] == "soft_horizon":
        threshold = loss_config["threshold"]
        softness = loss_config["softness"]
        return -compute_skill(data["val_data"], Y_pred, method="efh", threshold=threshold, softness=softness)
    elif loss_config["function"] == "rmse":
        return compute_skill(data["val_data"], Y_pred, method="error", metric="rmse")

def build_objective(data, reservoir_units, seed, hypopt_config, train_config, forecast_config):

    def objective(trial):
        # 1️⃣ Get hyperparameters from search space
        hyperparams = search_space(trial, hypopt_config["search_space"])

        # 2️⃣ Always include reservoir size
        hyperparams["reservoir_units"] = reservoir_units

        # 3️⃣ Compute loss
        loss = get_loss(data, hyperparams, train_config["washout_training"], forecast_config, hypopt_config["loss"], seed)

        trial.report(loss, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return loss

    return objective



def run_optimization(study_name, db_file, objective_func, hypopt_config):

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

    study.optimize(objective_func, n_trials=hypopt_config["trials"], timeout=hypopt_config["timeout_hours"] * 3600)
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

def save_best_params(best_params, filename):
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    #best_params = study.best_trial.params
    #best_params["seed"] = int(seed)  # ensure seed is a native int too
    with open(filename, "w") as f:
        json.dump(best_params, f, indent=4, default=convert_numpy)


""" def load_seed_hyperparams(
    seeds: list[int],
    out_dir: str,
    system: str,
    reservoir_units: int,
    offset: int,
    loss_function: str,
) -> dict[int, dict]:
    seed_hyperparams = {}

    for seed in seeds:
        param_path = Path(
            f"{out_dir}/hypopt/"
            f"{system}_N{reservoir_units}_T{offset}_S{seed}_params.json"
        )

        if not param_path.exists():
            raise FileNotFoundError(f"Missing parameter file: {param_path}")

        with param_path.open("r") as f:
            seed_hyperparams[seed] = json.load(f)

    return seed_hyperparams """

def load_seed_hyperparams(
    seeds: list[int],
    out_dir: str,
    filename_template: str,
    **template_params: Any,
) -> dict[int, dict]:
    seed_hyperparams = {}
    base_dir = Path(out_dir) / "hypopt"

    for seed in seeds:
        try:
            filename = filename_template.format(seed=seed, **template_params)
        except KeyError as e:
            raise ValueError(
                f"Filename template requires missing parameter: {e}"
            ) from None

        param_path = base_dir / filename

        if not param_path.exists():
            raise FileNotFoundError(f"Missing parameter file: {param_path}")

        with param_path.open("r") as f:
            seed_hyperparams[seed] = json.load(f)

    return seed_hyperparams