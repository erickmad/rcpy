import numpy as np
from rcpy.models import create_model
import random, json, os
import pandas as pd

def forecast_rcpy(model, warmup_data, forecast_length):

    dim = 1
    # Warm up the model
    warmup_y = model.run(warmup_data, reset=True)

    Y_pred = np.empty((forecast_length, dim))
    x = warmup_y[-1].reshape(1, -1)

    for i in range(forecast_length):
        x = model(x)
        Y_pred[i] = x

    return Y_pred


def multiple_forecasts_rcpy(
    data: dict,
    config: dict,
    params: dict = None,
    seeds: list[int] = None,
    params_path: str = None,
    params_filename_template: str = "{system}_N{reservoir_units}_S{seed}_params.json",
) -> list:
    """
    Generate multiple forecasts using different reservoir seeds.

    Parameters
    ----------
    data : dict
        Dictionary with 'train_data', 'train_target', and 'warmup_data'.
    config : dict
        Full experiment configuration (from config file).
    params : dict, optional
        Shared hyperparameters (used only when seeds is None).
    seeds : list of int, optional
        List of seeds to use. If given, params will be loaded per seed from JSON.
    params_path : str, optional
        Directory where per-seed parameter JSON files are stored.
    params_filename_template : str, optional
        Template for per-seed JSON filenames. Can include {system}, {reservoir_units}, {seed}.
        Default: "{system}_N{reservoir_units}_S{seed}_params.json"

    Returns
    -------
    forecasts : list of np.ndarray
        Forecasts from each model instance.
    """
    forecasts = []
    forecast_length = config["forecasting"]["length"]
    num_realizations = config["forecasting"].get("num_reservoirs", 10)
    system = config["system"]["name"]
    reservoir_units = config["reservoir"]["units"]

    if seeds is None:
        assert params is not None, "params must be provided when seeds is None."
        for seed in random.sample(range(1_000_000), num_realizations):
            model_params = params.copy()
            model_params["seed"] = seed
            model = create_model(model_config=model_params)
            model.fit(
                data["train_data"][:-1],
                data["train_data"][1:],
                warmup=config["training"]["discard_training"],
            )
            pred = forecast_rcpy(
                model=model,
                warmup_data=data["warmup_data"],
                forecast_length=forecast_length,
            )
            forecasts.append(pred)
    else:
        for seed in seeds:
            filename = params_filename_template.format(
                system=system, reservoir_units=reservoir_units, seed=seed
            )
            if params_path is not None:
                filename = os.path.join(params_path, filename)

            with open(filename, "r") as file:
                loaded_params = json.load(file)

            model_params = {
                "reservoir_units": reservoir_units,
                "p": loaded_params["p"],
                "leak_rate": loaded_params["leak_rate"],
                "spectral_radius": loaded_params["spectral_radius"],
                "input_scaling": loaded_params["input_scaling"],
                "alpha": loaded_params["alpha"],
                "seed": seed,
            }

            model = create_model(model_config=model_params)
            model.fit(
                data["train_data"][:-1],
                data["train_data"][1:],
                warmup=config["training"]["discard_training"],
            )
            pred = forecast_rcpy(
                model=model,
                warmup_data=data["warmup_data"],
                forecast_length=forecast_length,
            )
            forecasts.append(pred)

    return forecasts



def save_multiforecasts(forecasts: list, config: dict, seeds: list[int], mode: str = "per_seed", filename: str = None):
    """
    Save multiple forecasts to disk.

    Parameters
    ----------
    forecasts : list of np.ndarray
        List of forecast arrays (each shape: forecast_length,).
    config : dict
        Experiment configuration (must include system, reservoir, results).
    seeds : list of int
        Seeds corresponding to forecasts.
    mode : str, optional
        "per_seed" -> one file per seed
        "single_file"   -> one file with all forecasts (rows = seeds, cols = time steps)
    """
    out_dir = os.path.join(config["results"]["output_dir"], "forecasts")
    os.makedirs(out_dir, exist_ok=True)

    system = config["system"]["name"]
    units = config["reservoir"]["units"]

    if mode == "per_seed":
        for forecast, seed in zip(forecasts, seeds):
            filename = f"{system}_N{units}_forecast_seed{seed}.csv"
            filepath = os.path.join(out_dir, filename)

            pd.DataFrame(forecast).to_csv(filepath, index=False, header=False)
            print(f"✅ Saved forecast for seed {seed} to {filepath}")

    elif mode == "single_file":
        Y_matrix = np.array(forecasts)

        # Ensure Y_matrix is 2D: (n_seeds, forecast_length)
        if Y_matrix.ndim == 1:
            # Only one forecast, make it (1, forecast_length)
            Y_matrix = Y_matrix[np.newaxis, :]
        elif Y_matrix.ndim == 3:
            # Often (n_seeds, forecast_length, dim=1)
            Y_matrix = Y_matrix.squeeze(-1)

        n_seeds, n_steps = Y_matrix.shape
        print("Y_matrix shape:", Y_matrix.shape)
        df = pd.DataFrame(
            Y_matrix,
            index=[f"seed{seed}" for seed in seeds],
            columns=[f"t{i}" for i in range(n_steps)],
        )

        if filename is None:
            filename = f"{system}_N{units}_all_forecasts.csv"
        filepath = os.path.join(out_dir, filename)
        df.to_csv(filepath)

        print(f"✅ Saved all forecasts to {filepath}")

    else:
        raise ValueError("mode must be 'per_seed' or 'single_file'")
